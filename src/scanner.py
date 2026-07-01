"""
src/scanner.py
================

BIST piyasa taraması: tek bir hisse yerine, Borsa İstanbul'da işlem gören
TÜM hisseleri (~600+, sadece BIST-100 endeksi değil) tarar ve XGBoost
modelinin ürettiği "yukarı yön olasılığına" (proba_up) göre en güvenilir
adayları sıralar.

Sembol evreni: `fetch_bist_symbols()` TradingView'in genel tarayıcı
API'sinden BIST'teki tüm hisseleri dinamik olarak çeker (tvdatafeed ile aynı
kaynak). Bu, borsaya yeni giren/çıkan hisseler olsa bile listenin güncel
kalmasını sağlar. Ağ erişimi kısıtlıysa sırasıyla HTML tabanlı yedek
kaynaklara ve son olarak `config.BIST100_SYMBOLS` çekirdek listesine düşer.

Tasarım felsefesi - "eğit bir kere, tara sık sık":
- Her sembol için model, `ensure_symbol_model()` ile diske önbelleğe alınır.
  Model dosyası `MODEL_MAX_AGE_DAYS` günden eskiyse (veya hiç yoksa) otomatik
  olarak yeniden eğitilir; bu da sistemin "kendi kendini güncelleyen" yapısını
  ~600 hisse ölçeğinde sürdürür.
- Model eğitimi (Optuna hiperparametre araması dahil) görece pahalıdır ve
  periyodik olarak (varsayılan: haftada bir) yapılır.
- Asıl tarama (scan_market) ise ucuzdur: BistDataLoader'ın önbelleği sayesinde
  sadece son bar(lar) çekilir, özellikler/filtreler hesaplanır ve önbellekteki
  modelle tek satırlık bir tahmin yapılır. Modeller zaten eğitilmişse tüm
  BIST'in taranması birkaç dakikayı geçmez.
- `max_workers` > 1 ile semboller paralel taranabilir (~600 hisselik tam
  taramada süreyi kısaltır); CPU aşırı-abonelikten kaçınmak için bu modda
  XGBoost eğitimi otomatik olarak tek çekirdekte çalışır.
- Bir sembolün verisi çekilemez veya model eğitilemezse tarama tamamen
  durmaz; o sembol "error" statüsüyle işaretlenip atlanır.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from src import config
from src.data_loader import BistDataLoader
from src.features import FEATURE_COLUMNS, build_feature_matrix
from src.filters import build_filter_mask
from src.model import (
    WalkForwardEngine,
    evaluate_predictions,
    make_labels,
    optimize_hyperparameters,
    predict_latest,
    train_xgb,
)

logger = logging.getLogger("bist_bot.scanner")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------- #
# BIST TAM LİSTESİ — sadece BIST-100 değil, borsadaki TÜM hisseler (~600+)
# ---------------------------------------------------------------------- #
def _fetch_from_tradingview_scanner(min_len: int = 2, max_len: int = 7, timeout: int = 20) -> list[str]:
    """TradingView'in genel (kimlik doğrulama gerektirmeyen) tarayıcı API'sinden
    BIST'te işlem gören TÜM hisseleri çeker. tvdatafeed ile aynı kaynaktır."""
    import requests

    url = "https://scanner.tradingview.com/turkey/scan"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0",
        "Origin": "https://www.tradingview.com",
        "Referer": "https://www.tradingview.com/",
    }
    payload = {
        "filter": [],
        "options": {"lang": "en"},
        "symbols": {"query": {"types": ["stock"]}, "tickers": []},
        "columns": ["name", "description", "type", "subtype", "exchange"],
        "sort": {"sortBy": "name", "sortOrder": "asc"},
        "range": [0, 2000],
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    symbols: list[str] = []
    for row in data.get("data", []):
        cols = row.get("d") or []
        ticker = cols[0] if cols else None
        if ticker and isinstance(ticker, str):
            clean = ticker.split(":")[-1].strip().upper()
            if clean.isalpha() and min_len <= len(clean) <= max_len:
                symbols.append(clean)
    return list(dict.fromkeys(symbols))  # tekrarları kaldır, sırayı koru


def _fetch_bist_fallback_html(min_len: int = 2, max_len: int = 7, timeout: int = 15) -> list[str]:
    """TradingView API'ye ulaşılamazsa HTML tablo tabanlı yedek kaynakları dener."""
    import requests

    sources = [
        "https://www.isyatirim.com.tr/analysis/fundamental/equity/index",
        "https://finans.mynet.com/borsa/hisseler/",
    ]
    headers = {"User-Agent": "Mozilla/5.0"}
    for url in sources:
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            tables = pd.read_html(resp.text)
            symbols = []
            for table in tables:
                for col in table.select_dtypes("object").columns:
                    for val in table[col].dropna().astype(str):
                        val = val.strip().upper()
                        if val.isalpha() and min_len <= len(val) <= max_len:
                            symbols.append(val)
            unique = list(dict.fromkeys(symbols))
            if len(unique) > 50:
                return unique
        except Exception as exc:  # noqa: BLE001
            logger.debug("Yedek kaynak başarısız (%s): %s", url, exc)
            continue
    return []


def fetch_bist_symbols() -> list[str]:
    """BIST'te işlem gören TÜM hisselerin (~600+, sadece BIST-100 değil) güncel
    listesini döndürür.

    Öncelik sırası:
    1. TradingView Scanner API (dinamik, güncel, kimlik doğrulama gerekmez)
    2. HTML tablo tabanlı yedek kaynaklar (isyatirim.com.tr, mynet finans)
    3. `config.BIST100_SYMBOLS` (statik ama doğrulanmış 100 hisselik çekirdek liste)

    Ağ erişimi kısıtlı ortamlarda (ör. bazı kurumsal/izole ağlar) otomatik
    olarak 3. seçeneğe düşer; böylece tarama asla tamamen başarısız olmaz.
    """
    try:
        symbols = _fetch_from_tradingview_scanner()
        if symbols:
            logger.info("TradingView Scanner: %d BIST hissesi çekildi.", len(symbols))
            return symbols
    except Exception as exc:  # noqa: BLE001
        logger.warning("TradingView Scanner API başarısız: %s", exc)

    symbols = _fetch_bist_fallback_html()
    if symbols:
        logger.info("Yedek kaynaktan %d BIST hissesi çekildi.", len(symbols))
        return symbols

    logger.warning(
        "Tüm dinamik kaynaklar başarısız; config.BIST100_SYMBOLS kullanılıyor (%d hisse).",
        len(config.BIST100_SYMBOLS),
    )
    return list(config.BIST100_SYMBOLS)


def _model_path(symbol: str) -> Path:
    return config.MODEL_CACHE_DIR / f"{symbol.upper()}_xgb.json"


def _is_model_fresh(model_path: Path, max_age_days: float) -> bool:
    if not model_path.exists():
        return False
    age_seconds = time.time() - model_path.stat().st_mtime
    return age_seconds < max_age_days * 86400


def train_symbol_model(
    symbol: str,
    loader: BistDataLoader,
    train_bars: int = config.SCANNER_TRAIN_BARS,
    optuna_trials: int = config.SCANNER_OPTUNA_TRIALS,
    use_optuna: bool = True,
    holdout_bars: int = config.WALK_FORWARD_MIN_TEST_BARS,
    xgb_n_jobs: int = -1,
) -> WalkForwardEngine:
    """Tek bir sembol için (walk-forward döngüsü olmadan) tek seferlik model eğitir.

    Son `holdout_bars` bar hiperparametre seçimi ve raporlama için ayrılır;
    nihai model, seçilen parametrelerle TÜM veriyle (train+holdout) yeniden
    eğitilir - böylece dağıtılan (deploy edilen) model en güncel veriyi de
    görmüş olur. Bu, `WalkForwardEngine.run()`'ın yaptığı tarihsel doğrulama
    döngüsünden farklı olarak, canlı tarama için ucuz ve hızlı bir eğitim
    şeklidir; periyodik olarak (bkz. MODEL_MAX_AGE_DAYS) tekrarlanması
    walk-forward'ın "kendi kendini güncelleme" felsefesini canlıda sürdürür.
    """
    n_bars = train_bars + config.LABEL_HORIZON_BARS + 60  # + indikatör ısınma payı
    raw = loader.get_history(symbol, n_bars=n_bars)
    feat = build_feature_matrix(raw)
    feat["label"] = make_labels(feat)
    feat = feat.dropna(subset=list(FEATURE_COLUMNS) + ["label"])
    feat["label"] = feat["label"].astype(int)

    if len(feat) < holdout_bars + 100:
        raise ValueError(f"[{symbol}] eğitim için yetersiz veri: {len(feat)} satır")

    train_slice = feat.iloc[:-holdout_bars]
    holdout_slice = feat.iloc[-holdout_bars:]

    X_train, y_train = train_slice[FEATURE_COLUMNS], train_slice["label"]
    X_holdout, y_holdout = holdout_slice[FEATURE_COLUMNS], holdout_slice["label"]

    base_params = dict(config.XGB_DEFAULT_PARAMS)
    base_params["n_jobs"] = xgb_n_jobs  # paralel tarama sırasında CPU aşırı-abonelikten kaçınmak için

    if use_optuna:
        params = optimize_hyperparameters(X_train, y_train, n_trials=optuna_trials, base_params=base_params)
    else:
        params = base_params
    params["n_jobs"] = xgb_n_jobs

    holdout_model = train_xgb(X_train, y_train, params)
    holdout_proba = holdout_model.predict_proba(X_holdout)[:, 1]
    holdout_preds = (holdout_proba >= config.ENTRY_PROBABILITY_THRESHOLD).astype(int)
    metrics = evaluate_predictions(y_holdout, holdout_preds)

    # Nihai (dağıtılacak) model: seçilen parametrelerle tüm veriyle yeniden eğitilir.
    final_model = train_xgb(feat[FEATURE_COLUMNS], feat["label"], params)

    engine = WalkForwardEngine(feature_columns=list(FEATURE_COLUMNS))
    engine.model = final_model
    engine.current_params = params
    engine.save_model(_model_path(symbol))

    logger.info(
        "[%s] model eğitildi | holdout precision=%.3f recall=%.3f f1=%.3f | params=%s",
        symbol, metrics["precision"], metrics["recall"], metrics["f1"], params,
    )
    return engine


def ensure_symbol_model(
    symbol: str,
    loader: BistDataLoader,
    force_retrain: bool = False,
    max_age_days: float = config.MODEL_MAX_AGE_DAYS,
    **train_kwargs,
) -> WalkForwardEngine:
    """Önbellekteki modeli yükler; yoksa veya bayatsa yeniden eğitip önbelleğe alır."""
    model_path = _model_path(symbol)
    if not force_retrain and _is_model_fresh(model_path, max_age_days):
        engine = WalkForwardEngine(feature_columns=list(FEATURE_COLUMNS))
        engine.load_model(model_path)
        return engine
    return train_symbol_model(symbol, loader, **train_kwargs)


@dataclass
class ScanResult:
    symbol: str
    status: str  # "ok" | "error"
    as_of: Optional[pd.Timestamp] = None
    close: Optional[float] = None
    proba_up: Optional[float] = None
    signal: Optional[str] = None
    tradable: Optional[bool] = None
    adx: Optional[float] = None
    volume_ratio: Optional[float] = None
    rsi: Optional[float] = None
    trend_up: Optional[bool] = None
    atr_pct: Optional[float] = None
    error: Optional[str] = None


def _scan_single_symbol(
    symbol: str,
    loader: BistDataLoader,
    force_retrain: bool,
    max_age_days: float,
    apply_filters: bool,
    train_kwargs: dict,
) -> ScanResult:
    try:
        engine = ensure_symbol_model(symbol, loader, force_retrain=force_retrain, max_age_days=max_age_days, **train_kwargs)

        raw = loader.get_history(symbol, n_bars=config.SCANNER_TRAIN_BARS)
        feat = build_feature_matrix(raw)
        if apply_filters:
            feat = build_filter_mask(feat)

        latest = feat.iloc[-1]
        proba_up = predict_latest(engine.model, latest, engine.feature_columns)
        is_tradable = bool(latest["tradable"]) if apply_filters else None
        signal = "AL" if (proba_up >= config.ENTRY_PROBABILITY_THRESHOLD and (is_tradable is not False)) else "BEKLE"

        return ScanResult(
            symbol=symbol,
            status="ok",
            as_of=latest.name,
            close=float(latest["close"]),
            proba_up=proba_up,
            signal=signal,
            tradable=is_tradable,
            adx=float(latest.get(f"adx_{config.ADX_WINDOW}", float("nan"))) if apply_filters else None,
            volume_ratio=float(latest.get("volume_ratio", float("nan"))) if apply_filters else None,
            rsi=float(latest.get(f"rsi_{config.RSI_WINDOW}", float("nan"))),
            trend_up=bool(latest["trend_up"]) if apply_filters and "trend_up" in latest else None,
            atr_pct=float(latest.get("atr_pct", float("nan"))),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] tarama başarısız: %s", symbol, exc)
        return ScanResult(symbol=symbol, status="error", error=str(exc))


def scan_market(
    symbols: Optional[list[str]] = None,
    loader: Optional[BistDataLoader] = None,
    force_retrain: bool = False,
    max_age_days: float = config.MODEL_MAX_AGE_DAYS,
    apply_filters: bool = True,
    max_workers: int = 1,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    train_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    """BIST hisselerini tarayıp yukarı yön olasılığına göre sıralı bir DataFrame döndürür.

    Parameters
    ----------
    symbols: taranacak sembol listesi. None ise `fetch_bist_symbols()` ile
        BIST'teki TÜM hisseler (~600+, sadece BIST-100 değil) dinamik olarak
        çekilir. Daha hızlı/küçük bir tarama için `config.BIST100_SYMBOLS`
        veya kendi listenizi geçebilirsiniz.
    loader: paylaşılan bir BistDataLoader (önbelleği tekrar kullanmak için).
        None ise yeni bir tane oluşturulur.
    force_retrain: True ise tüm semboller için model yaşına bakılmaksızın
        yeniden eğitim yapılır.
    apply_filters: True ise ADX/hacim/volatilite/trend/seans filtreleri
        uygulanır ve `tradable` kolonu eklenir (bkz. src.filters).
    max_workers: 1'den büyükse semboller `ThreadPoolExecutor` ile paralel
        taranır (~600 hisselik tam taramada süreyi önemli ölçüde kısaltır).
        CPU aşırı-abonelikten kaçınmak için >1 olduğunda XGBoost eğitimi
        otomatik olarak `n_jobs=1` ile çalışır (train_kwargs içinde
        `xgb_n_jobs` açıkça verilmediği sürece).
    progress_callback: (index, total, symbol) imzalı, her sembol sonrası
        çağrılan opsiyonel ilerleme fonksiyonu (Colab'de canlı ilerleme
        göstermek için kullanışlıdır). Paralel modda çağrılar tamamlanma
        sırasına göre gelir (girdi sırasına göre değil).

    Returns
    -------
    En yüksek `proba_up` en üstte olacak şekilde sıralanmış DataFrame.
    Başarısız semboller `status == "error"` ile en altta yer alır.
    """
    symbols = list(symbols) if symbols is not None else fetch_bist_symbols()
    loader = loader or BistDataLoader()
    train_kwargs = dict(train_kwargs or {})
    if max_workers > 1:
        train_kwargs.setdefault("xgb_n_jobs", 1)

    total = len(symbols)
    results: list[ScanResult] = []

    if max_workers <= 1:
        for i, symbol in enumerate(symbols, start=1):
            result = _scan_single_symbol(symbol, loader, force_retrain, max_age_days, apply_filters, train_kwargs)
            results.append(result)
            if progress_callback is not None:
                progress_callback(i, total, symbol)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _scan_single_symbol, symbol, loader, force_retrain, max_age_days, apply_filters, train_kwargs
                ): symbol
                for symbol in symbols
            }
            for i, future in enumerate(as_completed(futures), start=1):
                symbol = futures[future]
                results.append(future.result())
                if progress_callback is not None:
                    progress_callback(i, total, symbol)

    df = pd.DataFrame([r.__dict__ for r in results])
    ok_mask = df["status"] == "ok"
    ranked = pd.concat(
        [
            df[ok_mask].sort_values("proba_up", ascending=False),
            df[~ok_mask],
        ],
        ignore_index=True,
    )
    n_ok = int(ok_mask.sum())
    logger.info("Tarama tamamlandı: %d/%d sembol başarılı.", n_ok, total)
    return ranked


def top_candidates(scan_df: pd.DataFrame, n: int = config.SCANNER_TOP_N, require_tradable: bool = True) -> pd.DataFrame:
    """Taramadan en yüksek yukarı-yön olasıklı ilk n adayı döndürür.

    require_tradable=True ise (varsayılan), sadece ADX/hacim/volatilite/trend/
    seans filtrelerini de geçen ("tradable") adaylar döndürülür - bu, sadece
    model olasılığına değil ek doğrulamalara da dayanan en güvenilir seçimdir.
    """
    ok = scan_df[scan_df["status"] == "ok"].copy()
    if require_tradable and "tradable" in ok.columns and ok["tradable"].notna().any():
        ok = ok[ok["tradable"] != False]  # noqa: E712 (None değerleri de dahil edilsin diye == True kullanılmadı)
    return ok.sort_values("proba_up", ascending=False).head(n).reset_index(drop=True)
