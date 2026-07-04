"""
src/deep_value.py
==================

BIST "Derin Değer & Kontraryan Dip Avcısı" motoru.

Felsefe
-------
Bu modül, mevcut momentum/trend-takip botunun (src/scanner.py, src/model.py)
TAM TERSİ bir bakış açısını kodlar:

    "Yükselişi / momentumu kovalamak yerine, TEKNİK olarak aşırı dövülmüş
     (ucuz kalmış) ama TEMEL olarak çürük OLMAYAN hisseleri avla."

"Malı ucuza almak kazandırır" prensibi — AMA sadece mal sağlamsa. Ucuz + çürük
= değer tuzağı (value trap). Bütün mesele, "ucuz-ve-sağlam" ile
"ucuz-çünkü-batıyor"u ayırmaktır. Bu yüzden iki BAĞIMSIZ eksende puanlarız ve
ikisini de geçmeyen eleniyor:

    Eksen A — Teknik Ucuzluk (0-100): Hisse ne kadar dövülmüş / aşırı satımda?
    Eksen B — Temel Değer & Kalite (0-100): Ucuzluğu hak mı ediyor, yoksa
              gerçekten değerli mi ve nakit üretiyor mu?

Nihai skor iki eksenin AĞIRLIKLI GEOMETRİK ortalamasıdır: biri düşükse toplam
düşer (hem ucuz HEM sağlam olması şart). Ayrıca "value-trap" bayrakları nihai
skoru çarpan cezasıyla düşürür ya da diskalifiye eder.

Alım tarafında Fibonacci "kademeli alım merdiveni" üretilir: dövülmüş hissenin
mevcut düşüş yapısının Fibonacci seviyelerine göre pozisyon 3-4 dilime bölünür,
her dilim ATR bazlı stop ile korunur — "düşen bıçağı" tek hamlede tutmayız.

Tasarım ilkeleri
----------------
* SAF (pure) puanlama fonksiyonları ağdan bağımsızdır; plain input alır, test
  edilebilir. Ağ/veri katmanı (İş Yatırım / tvdatafeed / yfinance) ayrı
  adapter fonksiyonlarındadır ve Colab/açık internette çalışır.
* Tüm indikatörler yalnızca geçmişe bakar (look-ahead yok).
* BIST temel verisi EKSİK ve GÜRÜLTÜLÜdür (İş Yatırım bazı oranları vermez,
  Yahoo trailing F/K çevrimsel diplerde 500+ olabilir). Bu yüzden puanlama
  "eldeki metriklerin ortalaması" mantığıyla eksiğe dayanıklıdır.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("bist_bot.deep_value")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ====================================================================== #
# 0. Yardımcı: güvenli sayı / normalizasyon
# ====================================================================== #
def _num(x) -> Optional[float]:
    """None / NaN / '' / string sayıyı güvenle float'a çevirir; olmazsa None."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _ramp(x: Optional[float], lo: float, hi: float) -> Optional[float]:
    """x'i [lo, hi] aralığından 0..100'e lineer ölçekler (hi'de 100).

    lo > hi ise ters ölçekler (küçük x = yüksek skor). x aralık dışıysa
    0/100'e sabitlenir. x None ise None döner (eksik veri).
    """
    if x is None:
        return None
    if lo == hi:
        return 50.0
    t = (x - lo) / (hi - lo)
    return float(max(0.0, min(1.0, t)) * 100.0)


def _mean_available(values: list[Optional[float]]) -> Optional[float]:
    """Yalnızca None olmayan değerlerin ortalaması (eksiğe dayanıklı)."""
    present = [v for v in values if v is not None]
    if not present:
        return None
    return float(np.mean(present))


def _wmean(pairs: list[tuple[Optional[float], float]]) -> Optional[float]:
    """Ağırlıklı ARİTMETİK ortalama (None atlanır, ağırlıklar yeniden normalize).

    Alt-skor BLEND'leri (value/quality) için: tek bir kötü metrik toplamı
    sıfırlamaz; sadece aşağı çeker. Eksenler-arası KAPILAMA için _wgeomean kullan.
    """
    usable = [(s, w) for s, w in pairs if s is not None and w > 0]
    if not usable:
        return None
    total_w = sum(w for _, w in usable)
    return float(sum(s * w for s, w in usable) / total_w)


def _wgeomean(pairs: list[tuple[Optional[float], float]]) -> Optional[float]:
    """Ağırlıklı geometrik ortalama (0-100 skorlar için).

    pairs: (skor, ağırlık) listesi. None skorlar atlanır (ağırlıkları
    yeniden normalize edilir). Herhangi bir skor 0 ise sonuç ~0 olur —
    "her iki eksen de geçmeli" mantığını doğal olarak dayatır.
    """
    usable = [(max(1e-9, s), w) for s, w in pairs if s is not None and w > 0]
    if not usable:
        return None
    total_w = sum(w for _, w in usable)
    log_sum = sum(w * math.log(s) for s, w in usable)
    return float(math.exp(log_sum / total_w))


# ====================================================================== #
# 1. Teknik indikatörler (saf, OHLCV DataFrame'den)
# ====================================================================== #
def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def williams_r(df: pd.DataFrame, window: int = 14) -> pd.Series:
    hh = df["high"].rolling(window).max()
    ll = df["low"].rolling(window).min()
    return -100 * (hh - df["close"]) / (hh - ll).replace(0, np.nan)


def bollinger_percent_b(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    lower = mid - n_std * std
    upper = mid + n_std * std
    return (close - lower) / (upper - lower).replace(0, np.nan)


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    return line - sig


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def _weekly_close(df: pd.DataFrame) -> pd.Series:
    """Günlük close'u haftalığa resample eder (haftalık RSI için)."""
    if not isinstance(df.index, pd.DatetimeIndex):
        return df["close"]
    return df["close"].resample("W-FRI").last().dropna()


# --- Para akışı / "banker" (akıllı para) göstergeleri --------------------
def money_flow_index(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """MFI: hacim ağırlıklı RSI. Düşük = aşırı satım + para çıkışı bitmek üzere."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    rmf = tp * df["volume"]
    pos = rmf.where(tp > tp.shift(1), 0.0)
    neg = rmf.where(tp < tp.shift(1), 0.0)
    pos_sum = pos.rolling(window).sum()
    neg_sum = neg.rolling(window).sum().replace(0, np.nan)
    mfr = pos_sum / neg_sum
    return (100 - 100 / (1 + mfr)).fillna(50.0)


def chaikin_money_flow(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """CMF: bar içi kapanışın yerine göre hacim baskısı. >0 = birikim (banker topluyor)."""
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl
    mfv = mfm * df["volume"]
    return (mfv.rolling(window).sum() / df["volume"].rolling(window).sum().replace(0, np.nan)).fillna(0.0)


def accum_dist(df: pd.DataFrame) -> pd.Series:
    """Accumulation/Distribution çizgisi (kümülatif hacim baskısı)."""
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl
    return (mfm.fillna(0.0) * df["volume"]).cumsum()


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume: yön işaretli kümülatif hacim."""
    sign = np.sign(df["close"].diff().fillna(0.0))
    return (sign * df["volume"]).cumsum()


# ====================================================================== #
# 2. EKSEN A — Teknik Ucuzluk Skoru (0-100)
# ====================================================================== #
# Alt bileşen ağırlıkları (toplam 1.0). "Dibe yakınlık" ve aşırı-satım
# ağırlıklı; MACD dönüşü bonus/teyit olarak ayrı raporlanır.
TECH_WEIGHTS = {
    "pos_52w": 0.28,   # 52 haftalık banttaki konum (dibe yakınlık) — kalp
    "rsi_d": 0.20,     # günlük RSI aşırı satım
    "rsi_w": 0.12,     # haftalık RSI aşırı satım (kalıcı ucuzluk teyidi)
    "drawdown": 0.15,  # 52h zirveden düşüş büyüklüğü
    "bb_b": 0.10,      # Bollinger alt banda sarkma
    "ema200": 0.10,    # 200 günlük ortalamanın altında kalma derinliği
    "williams": 0.05,  # Williams %R ek aşırı-satım teyidi
}


@dataclass
class TechnicalScore:
    total: float
    components: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)
    dip_confirm: bool = False   # MACD histogramı yukarı dönüyor mu (düşüş ivmesi kırıldı)


def technical_cheapness(df: pd.DataFrame) -> TechnicalScore:
    """Günlük OHLCV'den 0-100 teknik ucuzluk skoru.

    100 = maksimum dövülmüş / aşırı satımda (en "ucuz"). Trend-takip
    mantığının tersine: DÜŞÜK RSI, dipteki konum, derin drawdown yüksek puan alır.
    """
    df = df.copy()
    close = df["close"]
    n = len(df)

    rsi_d = rsi(close, 14)
    win52 = min(252, n)
    high_52 = df["high"].rolling(win52, min_periods=max(20, win52 // 4)).max()
    low_52 = df["low"].rolling(win52, min_periods=max(20, win52 // 4)).min()
    pos_52w = (close - low_52) / (high_52 - low_52).replace(0, np.nan)  # 0=dip, 1=tepe
    drawdown = (high_52 - close) / high_52.replace(0, np.nan)           # zirveden % düşüş
    pctb = bollinger_percent_b(close, 20, 2.0)
    ema200 = close.ewm(span=200, adjust=False).mean()
    ema200_dev = (close - ema200) / ema200                              # negatif = altında
    wr = williams_r(df, 14)
    mh = macd_hist(close)

    # Haftalık RSI
    wclose = _weekly_close(df)
    rsi_w_val = float(rsi(wclose, 14).iloc[-1]) if len(wclose) >= 15 else None

    raw = {
        "close": float(close.iloc[-1]),
        "rsi_d": float(rsi_d.iloc[-1]),
        "rsi_w": rsi_w_val,
        "pos_52w": float(pos_52w.iloc[-1]) if pd.notna(pos_52w.iloc[-1]) else None,
        "drawdown": float(drawdown.iloc[-1]) if pd.notna(drawdown.iloc[-1]) else None,
        "bb_percent_b": float(pctb.iloc[-1]) if pd.notna(pctb.iloc[-1]) else None,
        "ema200_dev": float(ema200_dev.iloc[-1]) if pd.notna(ema200_dev.iloc[-1]) else None,
        "williams_r": float(wr.iloc[-1]) if pd.notna(wr.iloc[-1]) else None,
        "high_52w": float(high_52.iloc[-1]) if pd.notna(high_52.iloc[-1]) else None,
        "low_52w": float(low_52.iloc[-1]) if pd.notna(low_52.iloc[-1]) else None,
    }

    # Alt bileşenleri 0-100 ucuzluk skoruna çevir (yüksek = daha ucuz)
    comp = {}
    # 52h konum: 0 (dip) -> 100, 1 (tepe) -> 0
    comp["pos_52w"] = _ramp(raw["pos_52w"], 0.85, 0.05) if raw["pos_52w"] is not None else None
    # RSI 50 -> 0, 15 -> 100
    comp["rsi_d"] = _ramp(raw["rsi_d"], 50.0, 15.0)
    comp["rsi_w"] = _ramp(raw["rsi_w"], 55.0, 25.0) if raw["rsi_w"] is not None else None
    # Drawdown: %10 -> 0, %60 -> 100
    comp["drawdown"] = _ramp(raw["drawdown"], 0.10, 0.60) if raw["drawdown"] is not None else None
    # Bollinger %B: 0.5 -> 0, -0.05 (alt bandın altı) -> 100
    comp["bb_b"] = _ramp(raw["bb_percent_b"], 0.50, -0.05) if raw["bb_percent_b"] is not None else None
    # 200EMA sapması: 0 -> 0, -%25 -> 100 (yani -dev'i ölçekle)
    comp["ema200"] = _ramp(-raw["ema200_dev"], 0.0, 0.25) if raw["ema200_dev"] is not None else None
    # Williams %R: -20 -> 0, -85 -> 100
    comp["williams"] = _ramp(-raw["williams_r"], 20.0, 85.0) if raw["williams_r"] is not None else None

    total = 0.0
    wsum = 0.0
    for k, w in TECH_WEIGHTS.items():
        if comp.get(k) is not None:
            total += w * comp[k]
            wsum += w
    total = total / wsum if wsum > 0 else 0.0

    # Dip teyidi: MACD histogramı son 3 barda yukarı dönüyorsa (düşüş ivmesi kırıldı)
    dip_confirm = bool(len(mh) >= 4 and mh.iloc[-1] > mh.iloc[-2] > mh.iloc[-3])

    return TechnicalScore(total=round(total, 1), components=comp, raw=raw, dip_confirm=dip_confirm)


# ====================================================================== #
# 2b. BANKER / AKILLI PARA (birikim) skoru — EKSTRA katman
# ====================================================================== #
@dataclass
class BankerScore:
    total: float
    components: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)
    accumulation: bool = False   # düşüşe rağmen birikim (pozitif diverjans) var mı


def banker_accumulation(df: pd.DataFrame) -> BankerScore:
    """"Banker"/akıllı para birikim skoru (0-100) — EKSTRA puanlama.

    Fikir: Aşırı ucuz bir hissede asıl aradığımız, fiyat düşerken ya da dipte
    yatarken GÜÇLÜ ELLERİN sessizce topladığının izidir. Klasik dip-avı sinyali
    "pozitif diverjans"tır: fiyat düşük/yatay AMA para akışı (CMF, A/D, OBV)
    yukarı. Bu skor yüksekse teknik ucuzluk daha güvenilirdir; düşükse hisse
    hâlâ dağıtım (satış) altındadır — acele etme.
    """
    n = len(df)
    close = df["close"]
    mfi = money_flow_index(df, 14)
    cmf = chaikin_money_flow(df, 20)
    ad = accum_dist(df)
    obv_s = obv(df)

    look = min(20, n - 1)
    price_chg = (close.iloc[-1] / close.iloc[-1 - look] - 1) if look > 0 else 0.0
    ad_chg = (ad.iloc[-1] - ad.iloc[-1 - look]) if look > 0 else 0.0
    obv_chg = (obv_s.iloc[-1] - obv_s.iloc[-1 - look]) if look > 0 else 0.0
    ad_norm = ad.abs().tail(60).mean() or 1.0
    obv_norm = obv_s.abs().tail(60).mean() or 1.0

    raw = {
        "mfi": float(mfi.iloc[-1]),
        "cmf": float(cmf.iloc[-1]),
        "price_chg_20": float(price_chg),
        "ad_slope": float(ad_chg / ad_norm) if ad_norm else 0.0,
        "obv_slope": float(obv_chg / obv_norm) if obv_norm else 0.0,
    }

    comp = {}
    # CMF pozitife döndükçe birikim: -0.10 -> 0, +0.15 -> 100
    comp["cmf"] = _ramp(raw["cmf"], -0.10, 0.15)
    # MFI aşırı satımdan dönüş: MFI 15 (dip) düşük puan, 45 civarı (para dönüyor) yüksek.
    #   Dikkat: burada "para GİRİYOR mu" istiyoruz; çok düşük MFI hâlâ çıkış demek.
    comp["mfi"] = _ramp(raw["mfi"], 15.0, 45.0)
    # A/D eğimi yukarı = birikim
    comp["ad_slope"] = _ramp(raw["ad_slope"], -0.5, 0.5)
    # OBV eğimi yukarı = birikim
    comp["obv_slope"] = _ramp(raw["obv_slope"], -0.5, 0.5)
    # Pozitif diverjans bonusu: fiyat düşmüş AMA para akışı yukarı
    divergence = raw["price_chg_20"] < -0.02 and (raw["ad_slope"] > 0.05 or raw["cmf"] > 0.02)
    comp["divergence"] = 100.0 if divergence else 40.0

    weights = {"cmf": 0.28, "mfi": 0.18, "ad_slope": 0.20, "obv_slope": 0.14, "divergence": 0.20}
    total = sum(weights[k] * comp[k] for k in weights)
    return BankerScore(total=round(total, 1), components=comp, raw=raw, accumulation=bool(divergence))


# ====================================================================== #
# 3. EKSEN B — Temel Değer & Kalite Skoru (0-100)
# ====================================================================== #
# Beklenen `ratios` sözlüğü anahtarları (hepsi opsiyonel, eksiğe dayanıklı):
#   pe_ratio, pb_ratio, ev_ebitda, ev_sales, owner_earnings, oe_yield,
#   safety_margin, buffett_score, net_income, roe, net_debt_to_equity
_BUFFETT_MAP = {"STRONG_BUY": 100.0, "BUY": 75.0, "HOLD": 50.0, "AVOID": 15.0, "SELL": 5.0}


@dataclass
class FundamentalScore:
    total: float
    value: Optional[float]
    quality: Optional[float]
    components: dict = field(default_factory=dict)


def _value_subscore(r: dict) -> tuple[Optional[float], dict]:
    """Fiyatlama ucuzluğu (0-100, yüksek = ucuz). Eldeki metriklerin ortalaması."""
    pe = _num(r.get("pe_ratio"))
    pb = _num(r.get("pb_ratio"))
    ev_ebitda = _num(r.get("ev_ebitda"))
    ev_sales = _num(r.get("ev_sales"))

    comp = {}
    # PD/DD: 0.6 -> 100, 3.0 -> 0
    comp["pb"] = _ramp(pb, 3.0, 0.6) if pb is not None and pb > 0 else None
    # EV/EBITDA: 3 -> 100, 15 -> 0
    comp["ev_ebitda"] = _ramp(ev_ebitda, 15.0, 3.0) if ev_ebitda is not None and ev_ebitda > 0 else None
    # F/K: 4 -> 100, 25 -> 0. Negatif ya da >100 (çevrimsel dip / anlamsız) -> yok say
    comp["pe"] = _ramp(pe, 25.0, 4.0) if pe is not None and 0 < pe <= 100 else None
    # EV/Satış: 0.4 -> 100, 4.0 -> 0 (düşük ağırlık; sektöre bağlı)
    comp["ev_sales"] = _ramp(ev_sales, 4.0, 0.4) if ev_sales is not None and ev_sales > 0 else None

    # Ağırlıklı ARİTMETİK: PD/DD ve EV/EBITDA daha güvenilir (çevrimsele dayanıklı).
    # Aritmetik ki tek bir pahalı metrik (ör. çevrimsel EV/EBITDA) skoru sıfırlamasın.
    weighted = [(comp["pb"], 0.35), (comp["ev_ebitda"], 0.35), (comp["pe"], 0.20), (comp["ev_sales"], 0.10)]
    value = _wmean([(s, w) for s, w in weighted]) if any(c is not None for c in comp.values()) else None
    return (round(value, 1) if value is not None else None), comp


def _quality_subscore(r: dict) -> tuple[Optional[float], dict]:
    """Sağlamlık/kalite (0-100). Nakit üretimi + DCF güvenlik marjı + Buffett + borç."""
    oe = _num(r.get("owner_earnings"))
    oe_yield = _num(r.get("oe_yield"))
    safety = _num(r.get("safety_margin"))
    buffett = r.get("buffett_score")
    ndte = _num(r.get("net_debt_to_equity"))
    roe = _num(r.get("roe"))

    comp = {}
    # DCF güvenlik marjı: -%30 -> 0, +%50 -> 100  (içsel değere göre iskonto)
    comp["safety"] = _ramp(safety, -0.30, 0.50) if safety is not None else None
    # Buffett skoru
    comp["buffett"] = _BUFFETT_MAP.get(str(buffett).upper()) if buffett else None
    # Nakit getirisi (OE yield): 0 -> 0, %15 -> 100
    if oe is not None and oe < 0:
        comp["cash"] = 0.0          # nakit YAKIYOR -> kalite sıfır sinyali
    elif oe_yield is not None:
        comp["cash"] = _ramp(oe_yield, 0.0, 0.15)
    elif oe is not None:
        comp["cash"] = 60.0 if oe > 0 else 0.0
    else:
        comp["cash"] = None
    # ROE varsa: %5 -> 0, %30 -> 100
    comp["roe"] = _ramp(roe, 0.05, 0.30) if roe is not None else None
    # Net borç/özsermaye varsa (düşük iyi): 2.0 -> 0, 0 -> 100
    comp["leverage"] = _ramp(ndte, 2.0, 0.0) if ndte is not None else None

    weighted = [
        (comp["safety"], 0.34),
        (comp["buffett"], 0.30),
        (comp["cash"], 0.24),
        (comp["roe"], 0.06),
        (comp["leverage"], 0.06),
    ]
    quality = _wmean([(s, w) for s, w in weighted]) if any(c is not None for c in comp.values()) else None
    return (round(quality, 1) if quality is not None else None), comp


def fundamental_value_quality(r: dict) -> FundamentalScore:
    """Temel Değer & Kalite skoru: value ve quality'nin ağırlıklı geo. ortalaması.

    Geometrik ortalama, KALİTE kapısı görevi görür: ucuz ama kalitesiz (value
    trap) hisselerde quality düşük olduğu için toplam da düşer.
    """
    value, vcomp = _value_subscore(r)
    quality, qcomp = _quality_subscore(r)
    # Temel biraz kalite-ağırlıklı (tuzaktan korunmak öncelik)
    total = _wgeomean([(value, 0.45), (quality, 0.55)])
    return FundamentalScore(
        total=round(total, 1) if total is not None else None,
        value=value,
        quality=quality,
        components={"value": vcomp, "quality": qcomp},
    )


# ====================================================================== #
# 4. Value-trap (değer tuzağı) elemesi
# ====================================================================== #
@dataclass
class TrapAssessment:
    flags: list[str]
    multiplier: float          # nihai skora uygulanacak çarpan (1.0 = temiz)
    disqualified: bool


def assess_value_trap(
    r: dict,
    tech: TechnicalScore,
    min_tl_volume: float = 5_000_000.0,
    avg_tl_volume: Optional[float] = None,
) -> TrapAssessment:
    """30 yıllık trader refleksi: 'ucuz ama neden ucuz?' Tuzak bayrakları.

    HARD tuzaklar nihai skoru ağır cezalandırır (çarpan) veya diskalifiye eder;
    böylece 'düşen bıçağı' körlemesine tutmayız.
    """
    flags: list[str] = []
    mult = 1.0
    disq = False

    oe = _num(r.get("owner_earnings"))
    ni = _num(r.get("net_income"))
    safety = _num(r.get("safety_margin"))
    buffett = str(r.get("buffett_score") or "").upper()
    pe = _num(r.get("pe_ratio"))
    pb = _num(r.get("pb_ratio"))
    ev_ebitda = _num(r.get("ev_ebitda"))

    # 1) Nakit yakıyor: negatif owner earnings — en ağır tuzak
    if oe is not None and oe < 0:
        flags.append("NAKIT_YAKIYOR (negatif owner earnings)")
        mult *= 0.30
    # 2) Zarar açıklamış
    if ni is not None and ni < 0:
        flags.append("ZARAR (negatif net kar)")
        mult *= 0.55
    # 3) DCF'e göre pahalı + AVOID: ucuz görünse de içsel değerin üstünde
    if buffett == "AVOID" and safety is not None and safety < 0:
        flags.append("DCF_PAHALI (icsel degerin uzerinde, AVOID)")
        mult *= 0.55
    # 4) Zarar nedeniyle ucuz görünüyor: F/K yok/negatif + düşük PD/DD
    if (pe is None or pe <= 0) and pb is not None and pb < 1.0 and (oe is None or oe <= 0):
        flags.append("UCUZ_CUNKU_ZARAR (F/K yok, dusuk PD/DD, nakit yok)")
        mult *= 0.50
    # 5) İşletme değeri pahalı (defterde ucuz ama EV/EBITDA yüksek)
    if ev_ebitda is not None and ev_ebitda > 15:
        flags.append("EV/EBITDA_PAHALI (isletme degeri yuksek)")
        mult *= 0.75
    # 6) Likidite tuzağı: TL bazlı ortalama hacim çok düşük — alınıp satılamaz
    if avg_tl_volume is not None and avg_tl_volume < min_tl_volume:
        flags.append(f"LIKIDITE_TUZAGI (ort. islem hacmi < {min_tl_volume:,.0f} TL)")
        mult *= 0.60
    # 7) Düşen bıçak: dip konumda, teyit yok VE haftalık RSI de dipte
    pos = tech.raw.get("pos_52w")
    if pos is not None and pos < 0.03 and not tech.dip_confirm:
        flags.append("DUSEN_BICAK (52h dibinde, dip teyidi yok)")
        mult *= 0.80

    # Diskalifiye: nakit yakan + zarar eden + AVOID kombinasyonu -> tamamen ele
    if oe is not None and oe < 0 and buffett == "AVOID":
        disq = True

    return TrapAssessment(flags=flags, multiplier=round(mult, 3), disqualified=disq)


# ====================================================================== #
# 5. Nihai birleşik skor
# ====================================================================== #
# Mimari (kullanıcı kurgusu):
#   ÖNCELİKLİ ÇEKİRDEK = teknik ucuzluk + banker (akıllı para). Sıralamayı bunlar belirler.
#   PUANLAMA KATKISI    = temel değer/kalite -> yalnızca ÖLÇÜLÜ çarpan katkısı (±%20).
#   VALUE-TRAP          = ceza çarpanı (çürük malı ele).
#
# final = çekirdek × trap_çarpanı × temel_katkı_çarpanı
#   çekirdek = CORE_W_TECH·teknik + CORE_W_BANKER·banker   (ikisi de öncelikli)
#   temel_katkı_çarpanı = 1 + FUND_BONUS_STRENGTH × (temel-50)/50   (±%20, sadece katkı)
# "Aşırı ucuz" kapısı teknik ucuzluğa bakar (fiyat ne kadar dövülmüş); banker çekirdeğin
# ikinci öncelikli bileşenidir (akıllı para topluyor mu). Temel yalnızca ince ayar yapar.
TECH_GATE = 60.0             # "aşırı ucuz" eşiği (teknik): bunun altı derin-değer adayı sayılmaz
CORE_W_TECH = 0.60           # ÖNCELİKLİ çekirdekte teknik ucuzluk ağırlığı
CORE_W_BANKER = 0.40         # ÖNCELİKLİ çekirdekte banker/akıllı para ağırlığı
FUND_BONUS_STRENGTH = 0.20   # temel kriter YALNIZCA puanlama katkısı (±%20)


@dataclass
class DeepValueResult:
    symbol: str
    final_score: float
    technical: TechnicalScore
    banker: BankerScore
    fundamental: FundamentalScore
    trap: TrapAssessment
    core_score: float                   # öncelikli çekirdek (teknik+banker)
    fund_bonus: float                   # temel kriterin katkı çarpanı (≈0.80–1.20)
    qualifies: bool                     # teknik kapıyı geçti mi (aşırı ucuz mu)
    ladder: Optional["FibLadder"] = None
    sector: Optional[str] = None
    intrinsic_value: Optional[float] = None   # içsel değer (TL/hisse) — temel, teknikten AYRI
    safety_margin: Optional[float] = None     # (içsel değer / fiyat - 1)
    intrinsic_method: Optional[str] = None     # "Graham" / "FCF×8" vb.


def composite_score(
    symbol: str,
    df: pd.DataFrame,
    ratios: dict,
    sector: Optional[str] = None,
    avg_tl_volume: Optional[float] = None,
) -> DeepValueResult:
    """Bir hisse için tam derin-değer değerlendirmesi.

    ÖNCELİKLİ ÇEKİRDEK = teknik ucuzluk + banker (akıllı para). Temel değer/kalite
    bunun üstüne YALNIZCA ölçülü bir puanlama katkısı (±%20) uygular. Value-trap
    bayrakları ayrıca ceza çarpanı getirir.
    """
    tech = technical_cheapness(df)
    banker = banker_accumulation(df)
    fund = fundamental_value_quality(ratios)
    trap = assess_value_trap(ratios, tech, avg_tl_volume=avg_tl_volume)

    # Öncelikli çekirdek: teknik + banker (ikisi de öncelik)
    core = _wmean([(tech.total, CORE_W_TECH), (banker.total, CORE_W_BANKER)]) or 0.0

    # Temel kriter: yalnızca ölçülü katkı çarpanı (temel yoksa nötr)
    if fund.total is not None:
        fund_bonus = 1.0 + FUND_BONUS_STRENGTH * (fund.total - 50.0) / 50.0
    else:
        fund_bonus = 1.0

    final = core * trap.multiplier * fund_bonus
    if trap.disqualified:
        final = 0.0

    return DeepValueResult(
        symbol=symbol,
        final_score=round(final, 1),
        technical=tech,
        banker=banker,
        fundamental=fund,
        trap=trap,
        core_score=round(core, 1),
        fund_bonus=round(fund_bonus, 3),
        qualifies=bool(tech.total >= TECH_GATE and not trap.disqualified),
        sector=sector,
        intrinsic_value=_num(ratios.get("dcf_intrinsic_value")),
        safety_margin=_num(ratios.get("safety_margin")),
        intrinsic_method=ratios.get("intrinsic_method"),
    )


# ====================================================================== #
# 6. Fibonacci kademeli alım merdiveni
# ====================================================================== #
@dataclass
class FibLadder:
    swing_high: float
    swing_low: float
    current_price: float
    rungs: list[dict]          # her biri: {ratio, price, weight_pct, note}
    hard_stop: float
    dcf_target: Optional[float] = None
    expected_upside_pct: Optional[float] = None


# Düşüşün Fibonacci oranları (H->L düşüşünün kesri olarak) ve dilim ağırlıkları.
# Daha derin (ucuz) seviyeye daha çok ağırlık: "ne kadar ucuzsa o kadar al".
_FIB_RATIOS = [
    (0.382, "sığ geri çekilme"),
    (0.500, "orta geri çekilme"),
    (0.618, "altın oran desteği"),
    (0.786, "derin geri çekilme"),
    (1.000, "swing dip retesti"),
    (1.272, "kapitülasyon uzantısı"),
    (1.618, "aşırı kapitülasyon"),
]


def fibonacci_ladder(
    df: pd.DataFrame,
    lookback: int = 180,
    n_rungs: int = 3,
    atr_stop_mult: float = 1.5,
    dcf_target: Optional[float] = None,
) -> FibLadder:
    """Dövülmüş hisse için kademeli alım merdiveni.

    Mantık: son `lookback` bardaki baskın SWING'i (en yüksek tepe H, en düşük
    dip L) al. H->L düşüşünün Fibonacci seviyeleri referans alınır. Alım
    kademeleri yalnızca GÜNCEL FİYATIN ALTINDAKİ (ya da hemen üstündeki)
    seviyelere konur — hisse düştükçe kademeli alırsın, ortalama maliyet düşer.
    Daha derin kademeye daha çok ağırlık verilir.

    Stop: en derin kademenin ATR katı kadar altına konur ("bu da tutmazsa tez
    yanlış" seviyesi).
    """
    win = df.tail(lookback)
    H = float(win["high"].max())
    L = float(win["low"].min())
    price = float(df["close"].iloc[-1])
    rng = max(H - L, 1e-9)
    atr_val = float(atr(df, 14).iloc[-1]) if len(df) > 15 else rng * 0.03

    # Aday seviyeler (fiyat cinsinden). ratio, düşüşün kesri: price_r = H - ratio*(H-L)
    candidates = []
    for ratio, note in _FIB_RATIOS:
        lvl = H - ratio * rng
        candidates.append({"ratio": ratio, "price": round(lvl, 4), "note": note})

    # Yalnızca güncel fiyatın %2 üstü ve altındaki seviyeler alım kademesi olur
    buyable = [c for c in candidates if c["price"] <= price * 1.02]
    # Güncel fiyata en yakından başlayıp aşağı doğru sırala
    buyable.sort(key=lambda c: -c["price"])

    # İlk kademe güncel fiyata yakın olsun: en yakın buyable seviye %6'dan fazla
    # aşağıdaysa, ilk kademe olarak MEVCUT FİYATı ekle (hemen birikime başla).
    if not buyable or buyable[0]["price"] < price * 0.94:
        buyable.insert(0, {"ratio": 0.0, "price": round(price, 4), "note": "mevcut fiyat (ilk kademe)"})

    rungs = buyable[:n_rungs]
    # 3 kademeye tamamlanamadıysa swing dip / uzantı ile doldur
    if len(rungs) < n_rungs:
        for ratio, note in [(1.0, "swing dip retesti"), (1.272, "kapitülasyon uzantısı"), (1.618, "aşırı kapitülasyon")]:
            lvl = round(H - ratio * rng, 4)
            if lvl < rungs[-1]["price"] and all(abs(lvl - r["price"]) > 1e-6 for r in rungs):
                rungs.append({"ratio": ratio, "price": lvl, "note": note})
            if len(rungs) >= n_rungs:
                break

    # Ağırlıklar: derine daha çok. n_rungs'a göre artan profil.
    base_weights = {
        1: [1.0],
        2: [0.4, 0.6],
        3: [0.25, 0.35, 0.40],
        4: [0.20, 0.25, 0.30, 0.25],
        5: [0.15, 0.20, 0.25, 0.25, 0.15],
    }.get(len(rungs), None)
    if base_weights is None:
        base_weights = [1.0 / len(rungs)] * len(rungs)
    for rung, w in zip(rungs, base_weights):
        rung["weight_pct"] = round(w * 100, 1)

    deepest = min(r["price"] for r in rungs)
    hard_stop = round(deepest - atr_stop_mult * atr_val, 4)

    upside = None
    if dcf_target is not None and dcf_target > 0:
        # Merdivenin ağırlıklı ortalama maliyetine göre beklenen getiri
        avg_cost = sum(r["price"] * r["weight_pct"] for r in rungs) / sum(r["weight_pct"] for r in rungs)
        upside = round((dcf_target / avg_cost - 1) * 100, 1)

    return FibLadder(
        swing_high=round(H, 4),
        swing_low=round(L, 4),
        current_price=round(price, 4),
        rungs=rungs,
        hard_stop=hard_stop,
        dcf_target=dcf_target,
        expected_upside_pct=upside,
    )


# ====================================================================== #
# 7. Universe tarama orkestrasyonu (rapor DataFrame'i)
# ====================================================================== #
def result_to_row(res: DeepValueResult) -> dict:
    """Tek bir DeepValueResult'ı rapor tablosu satırına indirger."""
    t = res.technical
    f = res.fundamental
    b = res.banker
    return {
        "symbol": res.symbol,
        "sector": res.sector,
        "final_score": res.final_score,
        "asiri_ucuz": res.qualifies,
        "cekirdek": res.core_score,     # ÖNCELİK: teknik+banker
        "tech_ucuzluk": t.total,        # öncelik 1
        "banker": b.total,              # öncelik 2: akıllı para birikimi
        "temel_skor": f.total,          # katkı: değer & kalite
        "temel_carpan": res.fund_bonus, # temelin nihai skora çarpan katkısı
        "value": f.value,
        "quality": f.quality,
        "icsel_deger": res.intrinsic_value,                    # temel: içsel değer (TL/hisse)
        "guvenlik_marji_%": round(res.safety_margin * 100, 1) if res.safety_margin is not None else None,
        "deger_yontemi": res.intrinsic_method,
        "rsi_d": round(t.raw.get("rsi_d"), 1) if t.raw.get("rsi_d") is not None else None,
        "rsi_w": round(t.raw["rsi_w"], 1) if t.raw.get("rsi_w") is not None else None,
        "pos_52w": round(t.raw["pos_52w"], 3) if t.raw.get("pos_52w") is not None else None,
        "drawdown": round(t.raw["drawdown"], 3) if t.raw.get("drawdown") is not None else None,
        "dip_confirm": t.dip_confirm,
        "birikim": b.accumulation,      # pozitif diverjans (banker topluyor)
        "trap_mult": res.trap.multiplier,
        "trap_flags": "; ".join(res.trap.flags) if res.trap.flags else "",
        "disqualified": res.trap.disqualified,
    }


def build_report(results: list[DeepValueResult]) -> "pd.DataFrame":
    rows = [result_to_row(r) for r in results]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    return df
