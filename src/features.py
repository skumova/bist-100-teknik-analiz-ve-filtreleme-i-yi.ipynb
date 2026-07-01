"""
src/features.py
================

Teknik indikatörler, fiyat dönüşümleri ve GARCH tabanlı volatilite tahmini
içeren özellik (feature) mühendisliği modülü.

Tüm fonksiyonlar sadece geçmişe bakar (t anındaki değer sadece t ve öncesi
barlardan hesaplanır) böylece walk-forward eğitimde "look-ahead bias" oluşmaz.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger("bist_bot.features")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------- #
# Temel fiyat dönüşümleri
# ---------------------------------------------------------------------- #
def add_log_returns(df: pd.DataFrame, column: str = "close") -> pd.DataFrame:
    df = df.copy()
    df["log_return"] = np.log(df[column] / df[column].shift(1))
    return df


def add_ema_deviation(df: pd.DataFrame, windows=config.EMA_WINDOWS, column: str = "close") -> pd.DataFrame:
    """EMA'lardan yüzdesel sapma: (fiyat - EMA) / EMA."""
    df = df.copy()
    for w in windows:
        ema = df[column].ewm(span=w, adjust=False).mean()
        df[f"ema_{w}"] = ema
        df[f"ema_dev_{w}"] = (df[column] - ema) / ema
    return df


# ---------------------------------------------------------------------- #
# Momentum / trend indikatörleri
# ---------------------------------------------------------------------- #
def add_rsi(df: pd.DataFrame, window: int = config.RSI_WINDOW, column: str = "close") -> pd.DataFrame:
    df = df.copy()
    delta = df[column].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    df[f"rsi_{window}"] = rsi.fillna(50.0)
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = config.MACD_FAST,
    slow: int = config.MACD_SLOW,
    signal: int = config.MACD_SIGNAL,
    column: str = "close",
) -> pd.DataFrame:
    df = df.copy()
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = macd_line - signal_line
    return df


def add_bollinger_bands(
    df: pd.DataFrame,
    window: int = config.BOLLINGER_WINDOW,
    n_std: float = config.BOLLINGER_STD,
    column: str = "close",
) -> pd.DataFrame:
    df = df.copy()
    mid = df[column].rolling(window).mean()
    std = df[column].rolling(window).std()

    upper = mid + n_std * std
    lower = mid - n_std * std

    df["bb_mid"] = mid
    df["bb_upper"] = upper
    df["bb_lower"] = lower
    band_range = (upper - lower).replace(0, np.nan)
    df["bb_percent_b"] = (df[column] - lower) / band_range
    df["bb_bandwidth"] = band_range / mid
    return df


def add_atr(
    df: pd.DataFrame,
    window: int = config.ATR_WINDOW,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.DataFrame:
    """Average True Range - kısa vadeli volatilite ölçümü ve stop-loss mesafesi için."""
    df = df.copy()
    prev_close = df[close_col].shift(1)
    tr = pd.concat(
        [
            df[high_col] - df[low_col],
            (df[high_col] - prev_close).abs(),
            (df[low_col] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df[f"atr_{window}"] = tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    df["atr_pct"] = df[f"atr_{window}"] / df[close_col]
    return df


# ---------------------------------------------------------------------- #
# GARCH tabanlı koşullu volatilite (risk yönetimi feature'ı)
# ---------------------------------------------------------------------- #
def add_garch_volatility(
    df: pd.DataFrame,
    return_column: str = "log_return",
    p: int = config.GARCH_P,
    q: int = config.GARCH_Q,
    scale: float = 100.0,
) -> pd.DataFrame:
    """GARCH(p, q) modeli ile koşullu varyans/oynaklık tahmini ekler.

    Not: Model, verilen `df` penceresindeki TÜM getirilerle bir kez eğitilir
    (in-sample fit). Walk-forward eğitimde look-ahead oluşmaması için bu
    fonksiyon her zaman sadece o anki eğitim penceresine (geçmiş veriye)
    uygulanmalıdır -- bkz. src/model.py walk-forward döngüsü, her adımda
    GARCH'ı yalnızca o adımın eğitim penceresiyle yeniden fit eder.
    """
    df = df.copy()
    returns = df[return_column].dropna() * scale  # arch, küçük sayılarda daha iyi yakınsar

    if len(returns) < max(50, 10 * (p + q)):
        logger.warning("GARCH için yetersiz veri (%d satır); volatilite rolling-std ile dolduruluyor.", len(returns))
        df["garch_variance"] = df[return_column].rolling(20).std().pow(2)
        df["garch_vol"] = df[return_column].rolling(20).std()
        return df

    try:
        from arch import arch_model

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(returns, vol="GARCH", p=p, q=q, mean="Zero", dist="normal", rescale=False)
            res = am.fit(disp="off", show_warning=False)

        cond_vol = res.conditional_volatility / scale  # ölçeği geri al
        cond_var = cond_vol.pow(2)

        df["garch_vol"] = cond_vol.reindex(df.index)
        df["garch_variance"] = cond_var.reindex(df.index)
        df["garch_vol"] = df["garch_vol"].ffill().bfill()
        df["garch_variance"] = df["garch_variance"].ffill().bfill()
    except Exception as exc:  # noqa: BLE001
        logger.warning("GARCH fit başarısız (%s); rolling-std'e düşülüyor.", exc)
        df["garch_variance"] = df[return_column].rolling(20).std().pow(2)
        df["garch_vol"] = df[return_column].rolling(20).std()

    return df


# ---------------------------------------------------------------------- #
# Tüm feature'ları tek adımda üreten pipeline
# ---------------------------------------------------------------------- #
FEATURE_COLUMNS = (
    ["log_return"]
    + [f"ema_dev_{w}" for w in config.EMA_WINDOWS]
    + [f"rsi_{config.RSI_WINDOW}", "macd", "macd_signal", "macd_hist"]
    + ["bb_percent_b", "bb_bandwidth"]
    + [f"atr_{config.ATR_WINDOW}", "atr_pct"]
    + ["garch_vol", "garch_variance"]
)


def build_feature_matrix(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    """OHLCV DataFrame'inden tüm modelleme feature'larını üretir.

    Parameters
    ----------
    df: en az `open, high, low, close, volume` kolonlarını içeren, datetime
        index'li DataFrame (bkz. src.data_loader.BistDataLoader).
    dropna: indikatör ısınma (warm-up) periyodundan kaynaklanan NaN
        satırların atılıp atılmayacağı.
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"build_feature_matrix: eksik kolonlar: {missing}")

    out = df.copy()
    out = add_log_returns(out)
    out = add_ema_deviation(out)
    out = add_rsi(out)
    out = add_macd(out)
    out = add_bollinger_bands(out)
    out = add_atr(out)
    out = add_garch_volatility(out)

    if dropna:
        out = out.dropna(subset=FEATURE_COLUMNS)

    return out
