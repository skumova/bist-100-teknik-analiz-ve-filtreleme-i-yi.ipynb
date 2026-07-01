"""
src/scanner.py
================

BIST piyasa taraması: tek bir hisse yerine, verilen sembol listesindeki
(varsayılan: BIST-100) TÜM hisseleri tarar ve XGBoost modelinin ürettiği
"yukarı yön olasılığına" (proba_up) göre en güvenilir adayları sıralar.

Tasarım felsefesi - "eğit bir kere, tara sık sık":
- Her sembol için model, `ensure_symbol_model()` ile diske önbelleğe alınır.
  Model dosyası `MODEL_MAX_AGE_DAYS` günden eskiyse (veya hiç yoksa) otomatik
  olarak yeniden eğitilir; bu da sistemin "kendi kendini güncelleyen" yapısını
  100 hisse ölçeğinde sürdürür.
- Model eğitimi (Optuna hiperparametre araması dahil) görece pahalıdır ve
  periyodik olarak (varsayılan: haftada bir) yapılır.
- Asıl tarama (scan_market) ise ucuzdur: BistDataLoader'ın önbelleği sayesinde
  sadece son bar(lar) çekilir, özellikler/filtreler hesaplanır ve önbellekteki
  modelle tek satırlık bir tahmin yapılır. 100 hissenin taranması, modeller
  zaten eğitilmişse birkaç dakikayı geçmez.
- Bir sembolün verisi çekilemez veya model eğitilemezse tarama tamamen
  durmaz; o sembol "error" statüsüyle işaretlenip atlanır.
"""

from __future__ import annotations

import logging
import time
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

    if use_optuna:
        params = optimize_hyperparameters(X_train, y_train, n_trials=optuna_trials)
    else:
        params = dict(config.XGB_DEFAULT_PARAMS)

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
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    train_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    """BIST hisselerini tarayıp yukarı yön olasılığına göre sıralı bir DataFrame döndürür.

    Parameters
    ----------
    symbols: taranacak sembol listesi (varsayılan: config.BIST100_SYMBOLS).
    loader: paylaşılan bir BistDataLoader (önbelleği tekrar kullanmak için).
        None ise yeni bir tane oluşturulur.
    force_retrain: True ise tüm semboller için model yaşına bakılmaksızın
        yeniden eğitim yapılır.
    apply_filters: True ise ADX/hacim/volatilite/trend/seans filtreleri
        uygulanır ve `tradable` kolonu eklenir (bkz. src.filters).
    progress_callback: (index, total, symbol) imzalı, her sembol sonrası
        çağrılan opsiyonel ilerleme fonksiyonu (Colab'de canlı ilerleme
        göstermek için kullanışlıdır).

    Returns
    -------
    En yüksek `proba_up` en üstte olacak şekilde sıralanmış DataFrame.
    Başarısız semboller `status == "error"` ile en altta yer alır.
    """
    symbols = symbols or list(config.BIST100_SYMBOLS)
    loader = loader or BistDataLoader()
    train_kwargs = train_kwargs or {}

    results: list[ScanResult] = []
    total = len(symbols)
    for i, symbol in enumerate(symbols, start=1):
        result = _scan_single_symbol(symbol, loader, force_retrain, max_age_days, apply_filters, train_kwargs)
        results.append(result)
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
