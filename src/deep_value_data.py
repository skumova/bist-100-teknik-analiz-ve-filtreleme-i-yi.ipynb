"""
src/deep_value_data.py
======================

Derin-değer motoru (src/deep_value.py) için AĞ / VERİ ADAPTER katmanı.

Bu dosya açık internette (Colab) çalışır. İki tür veri sağlar:

1. OHLCV geçmişi  -> mevcut `src.data_loader.BistDataLoader` (tvdatafeed
   [rongardF] -> yfinance yedek) üzerinden günlük barlar.
2. Temel oranlar  -> ÖNCELİK: İş Yatırım public API (F/K, PD/DD, EV/EBITDA...);
   YEDEK: yfinance `.IS` .info (priceToBook, enterpriseToEbitda, trailingPE...).

Not: Bu ortamın (Claude oturumu) ağ politikası İş Yatırım/Yahoo'yu bloke eder;
bu yüzden fonksiyonlar burada değil, Colab'da çalıştırılmak üzere tasarlanmıştır.
Tümü try/except ile sarılıdır: bir kaynak başarısız olursa diğerine düşülür,
en kötü ihtimalle eksik alanlar None döner (motor eksiğe dayanıklıdır).
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger("bist_bot.deep_value_data")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------- #
# OHLCV
# ---------------------------------------------------------------------- #
def load_daily_ohlcv(symbol: str, n_bars: int = 600, loader=None) -> pd.DataFrame:
    """Günlük OHLCV döndürür (tvdatafeed -> yfinance yedek).

    `loader` verilmezse yeni bir BistDataLoader('1d') oluşturur. Toplu taramada
    tek bir loader'ı yeniden kullanmak (önbellek paylaşımı için) önerilir.
    """
    if loader is None:
        from src.data_loader import BistDataLoader

        loader = BistDataLoader(interval="1d")
    return loader.get_history(symbol, n_bars=n_bars)


def average_tl_volume(df: pd.DataFrame, window: int = 20) -> Optional[float]:
    """Son `window` bardaki ortalama TL bazlı işlem hacmi (likidite tuzağı için)."""
    if df is None or df.empty:
        return None
    tl = (df["close"] * df["volume"]).tail(window)
    return float(tl.mean()) if len(tl) else None


# ---------------------------------------------------------------------- #
# Temel oranlar - İş Yatırım (öncelik)
# ---------------------------------------------------------------------- #
_ISYATIRIM_URL = (
    "https://www.isyatirim.com.tr/_layouts/15/IsYatirim.Website/Common/"
    "Data.aspx/StockAnalysisRatios?hisse={sym}"
)


def fetch_ratios_isyatirim(symbol: str, timeout: int = 15) -> dict:
    """İş Yatırım'dan temel oranları çeker (best-effort).

    İş Yatırım'ın public JSON uçları zamanla değişebilir; bu yüzden fonksiyon
    başarısız olursa BOŞ dict döner ve çağıran taraf yfinance yedeğine düşer.
    Beklenen çıktı anahtarları motorun (deep_value) beklediği isimlerdir.
    """
    import requests

    try:
        url = _ISYATIRIM_URL.format(sym=symbol.upper())
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        payload = r.json()
        rows = payload.get("value") or payload.get("d") or []
        if not rows:
            return {}
        rec = rows[0] if isinstance(rows, list) else rows
        out = {
            "pe_ratio": rec.get("FK") or rec.get("FiyatKazanc"),
            "pb_ratio": rec.get("PDDD") or rec.get("PiyasaDegeriDefterDegeri"),
            "ev_ebitda": rec.get("FDFAVOK") or rec.get("EVEBITDA"),
            "ev_sales": rec.get("FDSatis") or rec.get("EVSatis"),
        }
        return {k: v for k, v in out.items() if v is not None}
    except Exception as exc:  # noqa: BLE001
        logger.debug("[%s] İş Yatırım oranları alınamadı: %s", symbol, exc)
        return {}


# ---------------------------------------------------------------------- #
# Temel oranlar - yfinance yedek (portatif, Colab'da güvenilir çalışır)
# ---------------------------------------------------------------------- #
def fetch_ratios_yfinance(symbol: str) -> dict:
    """yfinance `.IS` .info'dan temel oranlar (yedek kaynak).

    Motorun beklediği anahtar isimlerine map'ler. Eksik alanlar döndürülmez.
    """
    try:
        import yfinance as yf

        sym = symbol.upper()
        if not sym.endswith(".IS"):
            sym = f"{sym}.IS"
        info = yf.Ticker(sym).info or {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("[%s] yfinance info alınamadı: %s", symbol, exc)
        return {}

    out = {
        "pe_ratio": info.get("trailingPE"),
        "pb_ratio": info.get("priceToBook"),
        "ev_ebitda": info.get("enterpriseToEbitda"),
        "ev_sales": info.get("enterpriseToRevenue"),
        "roe": info.get("returnOnEquity"),
        "net_debt_to_equity": (info.get("debtToEquity") / 100.0) if info.get("debtToEquity") else None,
        "net_income": info.get("netIncomeToCommon"),
        "owner_earnings": info.get("freeCashflow"),  # yaklaşık: serbest nakit akışı
        "sector": info.get("sector"),
    }
    return {k: v for k, v in out.items() if v is not None}


def fetch_fundamentals(symbol: str, prefer_isyatirim: bool = True) -> dict:
    """İş Yatırım + yfinance oranlarını birleştirir (İş Yatırım öncelikli)."""
    yf_data = fetch_ratios_yfinance(symbol)
    if not prefer_isyatirim:
        return yf_data
    isy = fetch_ratios_isyatirim(symbol)
    merged = dict(yf_data)
    merged.update({k: v for k, v in isy.items() if v is not None})  # İş Yatırım üstün gelir
    return merged


# ---------------------------------------------------------------------- #
# Universe tarama (uçtan uca)
# ---------------------------------------------------------------------- #
def screen_universe(
    symbols: list[str],
    n_bars: int = 600,
    fib_lookback: int = 180,
    prefer_isyatirim: bool = True,
    with_ladder_top_n: int = 15,
    tech_prefilter: bool = True,
    prefilter_margin: float = 10.0,
):
    """Sembol listesini uçtan uca tarar: veri çek -> skorla -> raporla.

    ANA belirleyici teknik ucuzluk olduğu için önce teknik hesaplanır; teknik
    kapının (TECH_GATE) belirgin altında kalan hisseler için pahalı temel veri
    çekilmez (`tech_prefilter=True`). Böylece 100 hissede gereksiz API isteği
    yapılmaz — sadece "aşırı ucuz" adaylara temel + banker ekstra puanı işlenir.

    Dönüş: (rapor_df, results) — rapor_df sıralı özet tablo, results ise
    her sembol için tam DeepValueResult listesi (Fibonacci merdivenleri dahil).
    """
    from src import deep_value as dv
    from src.data_loader import BistDataLoader

    loader = BistDataLoader(interval="1d")
    results = []
    gate = dv.TECH_GATE - prefilter_margin
    for sym in symbols:
        try:
            df = load_daily_ohlcv(sym, n_bars=n_bars, loader=loader)
            if df is None or len(df) < 60:
                logger.warning("[%s] yetersiz OHLCV, atlanıyor.", sym)
                continue
            # Teknik ön-eleme: kapıyı geçemeyecek kadar pahalıysa temel veri çekme
            if tech_prefilter and dv.technical_cheapness(df).total < gate:
                ratios = {}
            else:
                ratios = fetch_fundamentals(sym, prefer_isyatirim=prefer_isyatirim)
            res = dv.composite_score(
                sym, df, ratios,
                sector=ratios.get("sector"),
                avg_tl_volume=average_tl_volume(df),
            )
            results.append((res, df, ratios))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] taranamadı: %s", sym, exc)

    ranked = sorted((r[0] for r in results), key=lambda x: x.final_score, reverse=True)
    # En iyi N adaya Fibonacci merdiveni ekle
    df_by_sym = {r[0].symbol: r[1] for r in results}
    ratios_by_sym = {r[0].symbol: r[2] for r in results}
    for res in ranked[:with_ladder_top_n]:
        d = df_by_sym[res.symbol]
        dcf = None
        r = ratios_by_sym[res.symbol]
        if r.get("dcf_intrinsic_value"):
            dcf = dv._num(r.get("dcf_intrinsic_value"))
        res.ladder = dv.fibonacci_ladder(d, lookback=fib_lookback, dcf_target=dcf)

    report = dv.build_report(ranked)
    return report, ranked
