"""
src/filters.py
===============

Sinyal kalitesini artırmak için ek (opsiyonel) filtreler.

Bu filtreler, XGBoost'un ürettiği yukarı yön olasılığı eşiği geçse bile,
piyasa koşulları işlem açmaya uygun değilse girişi engeller:

- ADX: trend gücü zayıfsa (yatay/gürültülü piyasa) işlem açılmaz.
- Hacim filtresi: ortalamanın belirgin altında hacimli (likit olmayan/tatil
  öncesi vb.) barlarda işlem açılmaz.
- Volatilite rejimi: GARCH koşullu oynaklığı aşırı düşük (sinyal-gürültü
  oranı kötü) veya aşırı yüksek (kayma/gap riski yüksek) rejimlerdeyken
  işlem açılmaz.
- Trend hizası: sadece EMA(fast) > EMA(slow) olan yükseliş rejiminde long
  açılır (BUY-only stratejide karşı-trend işlemleri eler).
- İşlem seansı: gün içi verilerde açılış/kapanışa yakın (gürültülü) dakikalar
  ile BIST seans dışı barlar elenir. Günlük barlarda otomatik olarak devre dışı
  kalır (no-op).

Tüm fonksiyonlar yalnızca geçmişe bakar; look-ahead bias oluşturmaz.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger("bist_bot.filters")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------- #
# ADX - trend gücü
# ---------------------------------------------------------------------- #
def add_adx(df: pd.DataFrame, window: int = config.ADX_WINDOW) -> pd.DataFrame:
    df = df.copy()
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_smooth = tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    plus_dm_smooth = pd.Series(plus_dm, index=df.index).ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=df.index).ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    plus_di = 100 * (plus_dm_smooth / atr_smooth.replace(0, np.nan))
    minus_di = 100 * (minus_dm_smooth / atr_smooth.replace(0, np.nan))

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx = dx.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    df[f"adx_{window}"] = adx.fillna(0.0)
    df["plus_di"] = plus_di.fillna(0.0)
    df["minus_di"] = minus_di.fillna(0.0)
    return df


# ---------------------------------------------------------------------- #
# Hacim filtresi
# ---------------------------------------------------------------------- #
def add_volume_ratio(df: pd.DataFrame, window: int = config.VOLUME_WINDOW) -> pd.DataFrame:
    df = df.copy()
    avg_volume = df["volume"].rolling(window, min_periods=window // 2).mean()
    df["volume_ratio"] = df["volume"] / avg_volume.replace(0, np.nan)
    df["volume_ratio"] = df["volume_ratio"].fillna(1.0)
    return df


# ---------------------------------------------------------------------- #
# Volatilite rejimi (GARCH koşullu oynaklığın rolling percentile rank'i)
# ---------------------------------------------------------------------- #
def add_volatility_regime(
    df: pd.DataFrame,
    vol_column: str = "garch_vol",
    window: int = config.VOL_REGIME_WINDOW,
) -> pd.DataFrame:
    df = df.copy()
    if vol_column not in df.columns:
        logger.warning("'%s' kolonu yok; volatilite rejimi filtresi atlanıyor.", vol_column)
        df["vol_regime_pct"] = 0.5
        return df

    def _last_percentile(window_values: np.ndarray) -> float:
        return (window_values <= window_values[-1]).mean()

    df["vol_regime_pct"] = (
        df[vol_column]
        .rolling(window, min_periods=max(10, window // 4))
        .apply(_last_percentile, raw=True)
    )
    df["vol_regime_pct"] = df["vol_regime_pct"].fillna(0.5)
    return df


# ---------------------------------------------------------------------- #
# Trend hizası
# ---------------------------------------------------------------------- #
def add_trend_alignment(
    df: pd.DataFrame,
    fast: int = config.TREND_FAST_EMA,
    slow: int = config.TREND_SLOW_EMA,
) -> pd.DataFrame:
    df = df.copy()
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["trend_up"] = ema_fast > ema_slow
    return df


# ---------------------------------------------------------------------- #
# İşlem seansı filtresi (yalnızca gün-içi veriler için anlamlı)
# ---------------------------------------------------------------------- #
def add_session_filter(
    df: pd.DataFrame,
    market_open: str = config.MARKET_OPEN,
    market_close: str = config.MARKET_CLOSE,
    edge_exclude_minutes: int = config.SESSION_EDGE_EXCLUDE_MINUTES,
) -> pd.DataFrame:
    df = df.copy()

    is_intraday = df.index.to_series().dt.time.nunique() > 1
    if not is_intraday:
        df["in_session"] = True
        return df

    open_t = pd.to_datetime(market_open).time()
    close_t = pd.to_datetime(market_close).time()
    times = df.index.to_series().dt.time

    open_dt = pd.to_datetime(market_open)
    close_dt = pd.to_datetime(market_close)
    edge_open = (open_dt + pd.Timedelta(minutes=edge_exclude_minutes)).time()
    edge_close = (close_dt - pd.Timedelta(minutes=edge_exclude_minutes)).time()

    df["in_session"] = (times >= edge_open) & (times <= edge_close) & (times >= open_t) & (times <= close_t)
    return df


# ---------------------------------------------------------------------- #
# Tüm filtreleri birleştiren pipeline
# ---------------------------------------------------------------------- #
FILTER_DIAGNOSTIC_COLUMNS = [
    f"adx_{config.ADX_WINDOW}",
    "volume_ratio",
    "vol_regime_pct",
    "trend_up",
    "in_session",
]


def build_filter_mask(
    df: pd.DataFrame,
    min_adx: float = config.MIN_ADX,
    min_volume_ratio: float = config.MIN_VOLUME_RATIO,
    vol_regime_lower: float = config.VOL_REGIME_LOWER_PCT,
    vol_regime_upper: float = config.VOL_REGIME_UPPER_PCT,
    require_trend_alignment: bool = config.REQUIRE_TREND_ALIGNMENT,
    session_filter_enabled: bool = config.SESSION_FILTER_ENABLED,
) -> pd.DataFrame:
    """Tüm ek filtreleri hesaplayıp tek bir `tradable` boolean kolonuna indirger.

    Girdi `df`, en azından `build_feature_matrix` çıktısını (OHLCV + teknik
    indikatörler + garch_vol) içermelidir. Çıktı, orijinal kolonlara ek olarak
    her filtrenin ara sonucunu ve nihai `tradable` kolonunu içerir.
    """
    out = df.copy()
    out = add_adx(out)
    out = add_volume_ratio(out)
    out = add_volatility_regime(out)
    out = add_trend_alignment(out)
    out = add_session_filter(out)

    adx_col = f"adx_{config.ADX_WINDOW}"
    conditions = [
        out[adx_col] >= min_adx,
        out["volume_ratio"] >= min_volume_ratio,
        out["vol_regime_pct"].between(vol_regime_lower, vol_regime_upper),
        out["in_session"],
    ]
    if require_trend_alignment:
        conditions.append(out["trend_up"])

    tradable = conditions[0]
    for cond in conditions[1:]:
        tradable = tradable & cond

    out["tradable"] = tradable.fillna(False)
    return out
