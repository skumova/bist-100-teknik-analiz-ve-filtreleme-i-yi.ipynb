"""
src/data_loader.py
===================

Önbellek (memory + disk) destekli, minimum sayıda API isteği yapan veri katmanı.

Tasarım ilkesi: "Bir kere indir, hep kullan."
- İlk çağrıda tüm geçmiş veri (n_bars) indirilir ve hem RAM'de (pandas DataFrame)
  hem de diskte (parquet) saklanır.
- Sonraki her çağrıda sadece son birkaç bar TradingView/yfinance'tan çekilir,
  zaten bellekte olan DataFrame'e "append" edilir; tüm geçmiş tekrar indirilmez.
- Birincil kaynak: tvdatafeed (TradingView). Kimlik doğrulama başarısız olur,
  paket kurulu değilse veya istek hata verirse otomatik olarak yfinance'a düşer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger("bist_bot.data_loader")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# tvdatafeed <-> yfinance <-> insan-okunur interval eşlemesi
_TV_INTERVAL_NAMES = {
    "1m": "in_1_minute",
    "5m": "in_5_minute",
    "15m": "in_15_minute",
    "30m": "in_30_minute",
    "1h": "in_1_hour",
    "4h": "in_4_hour",
    "1d": "in_daily",
}
_YF_INTERVAL_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "60m",
    "4h": "60m",  # yfinance 4h desteklemez; 60m indirilip sonradan resample edilebilir
    "1d": "1d",
}


def _bist_symbol_for_yfinance(symbol: str) -> str:
    """BIST sembollerini yfinance formatına çevirir (GARAN -> GARAN.IS)."""
    symbol = symbol.upper().strip()
    if symbol.endswith(".IS"):
        return symbol
    return f"{symbol}.IS"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Farklı kaynaklardan gelen kolon isimlerini standart hale getirir."""
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    rename_map = {"adj close": "close", "vol": "volume"}
    df = df.rename(columns=rename_map)
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Beklenen OHLCV kolonları eksik: {missing}. Mevcut: {list(df.columns)}")
    df = df[OHLCV_COLUMNS]
    df.index.name = "datetime"
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.sort_index()


class BistDataLoader:
    """BIST hisseleri için önbellekli (memory + disk) OHLCV veri yükleyici.

    Parameters
    ----------
    exchange: TradingView borsa kodu (varsayılan "BIST").
    interval: "1m", "5m", "15m", "30m", "1h", "4h", "1d" değerlerinden biri.
    tv_username / tv_password: tvdatafeed için opsiyonel TradingView kimlik bilgileri.
        Belirtilmezse tvdatafeed anonim (sınırlı) modda dener; başarısız olursa
        otomatik olarak yfinance kullanılır.
    """

    def __init__(
        self,
        exchange: str = config.DEFAULT_EXCHANGE,
        interval: str = config.DEFAULT_INTERVAL,
        tv_username: Optional[str] = None,
        tv_password: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        if interval not in _TV_INTERVAL_NAMES:
            raise ValueError(f"Desteklenmeyen interval: {interval}. Seçenekler: {list(_TV_INTERVAL_NAMES)}")

        self.exchange = exchange
        self.interval = interval
        self.tv_username = tv_username
        self.tv_password = tv_password
        self.cache_dir = Path(cache_dir) if cache_dir else config.DATA_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._memory: dict[str, pd.DataFrame] = {}
        self._tv_client = None
        self._tv_unavailable = False

    # ------------------------------------------------------------------ #
    # Kamuya açık API
    # ------------------------------------------------------------------ #
    def get_history(
        self,
        symbol: str,
        n_bars: int = config.DEFAULT_LOOKBACK_BARS,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Sembol için OHLCV geçmişini döndürür.

        - Bellekte veri varsa: sadece son bar(lar) çekilip mevcut DataFrame'e eklenir.
        - Bellekte yoksa ama diskte varsa: diskten yüklenir, sonra son bar güncellenir.
        - Hiçbiri yoksa: tam n_bars kadar geçmiş indirilir.
        """
        key = symbol.upper()

        if force_refresh or key not in self._memory:
            disk_df = None if force_refresh else self._load_disk_cache(key)
            if disk_df is not None and not disk_df.empty:
                logger.info("[%s] Disk önbelleğinden %d bar yüklendi.", key, len(disk_df))
                self._memory[key] = disk_df
            else:
                logger.info("[%s] Önbellek bulunamadı, %d barlık tam geçmiş indiriliyor.", key, n_bars)
                full_df = self._fetch_raw(key, n_bars=n_bars)
                self._memory[key] = full_df
                self._save_disk_cache(key, full_df)
                return self._memory[key]

        self._append_latest_bar(key)
        return self._memory[key]

    def get_latest_bar(self, symbol: str) -> Optional[pd.Series]:
        """Sadece en güncel (son) barı döndürür, DataFrame'i günceller."""
        key = symbol.upper()
        if key not in self._memory:
            self.get_history(key)
        else:
            self._append_latest_bar(key)
        if self._memory[key].empty:
            return None
        return self._memory[key].iloc[-1]

    def warm_cache(self, symbols: list[str], n_bars: int = config.DEFAULT_LOOKBACK_BARS) -> None:
        """Birden çok sembol için önbelleği önceden ısıtır (toplu ilk indirme)."""
        for sym in symbols:
            try:
                self.get_history(sym, n_bars=n_bars)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%s] önbellek ısıtma başarısız: %s", sym, exc)

    # ------------------------------------------------------------------ #
    # Bellek güncelleme (append-only)
    # ------------------------------------------------------------------ #
    def _append_latest_bar(self, key: str) -> None:
        """Sadece en son barları çekip mevcut DataFrame'e ekler (tam indirme yapmaz)."""
        current = self._memory.get(key)
        fetch_n = 5  # son kapanan barı garanti yakalamak için küçük bir pencere
        try:
            latest = self._fetch_raw(key, n_bars=fetch_n)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] canlı bar güncellemesi başarısız, mevcut önbellek korunuyor: %s", key, exc)
            return

        if current is None or current.empty:
            self._memory[key] = latest
            self._save_disk_cache(key, latest)
            return

        last_ts = current.index[-1]
        new_rows = latest[latest.index > last_ts]
        # Son barın kendisi kapanmadan tekrar geldiyse (aynı timestamp) güncelle
        overlapping = latest[latest.index == last_ts]

        updated = current
        if not overlapping.empty:
            updated = updated.copy()
            updated.loc[last_ts] = overlapping.iloc[-1]

        if not new_rows.empty:
            updated = pd.concat([updated, new_rows])
            logger.info("[%s] %d yeni bar belleğe eklendi (append).", key, len(new_rows))
            self._save_disk_cache(key, updated)

        self._memory[key] = updated

    # ------------------------------------------------------------------ #
    # Disk önbelleği
    # ------------------------------------------------------------------ #
    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}_{self.interval}.parquet"

    def _load_disk_cache(self, key: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] disk önbelleği okunamadı (%s), yeniden indirilecek.", key, exc)
            return None

    def _save_disk_cache(self, key: str, df: pd.DataFrame) -> None:
        try:
            df.to_parquet(self._cache_path(key))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] disk önbelleğine yazılamadı: %s", key, exc)

    # ------------------------------------------------------------------ #
    # Sağlayıcılar: tvdatafeed (öncelikli) ve yfinance (yedek)
    # ------------------------------------------------------------------ #
    def _fetch_raw(self, symbol: str, n_bars: int) -> pd.DataFrame:
        if not self._tv_unavailable:
            try:
                df = self._fetch_tvdatafeed(symbol, n_bars)
                if df is not None and not df.empty:
                    return df
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[%s] tvdatafeed başarısız (%s). yfinance yedeğine geçiliyor.", symbol, exc
                )
                self._tv_unavailable = True

        return self._fetch_yfinance(symbol, n_bars)

    def _get_tv_client(self):
        if self._tv_client is not None:
            return self._tv_client
        from tvDatafeed import TvDatafeed  # type: ignore

        if self.tv_username and self.tv_password:
            self._tv_client = TvDatafeed(self.tv_username, self.tv_password)
        else:
            self._tv_client = TvDatafeed()
        return self._tv_client

    def _fetch_tvdatafeed(self, symbol: str, n_bars: int) -> pd.DataFrame:
        from tvDatafeed import Interval  # type: ignore

        client = self._get_tv_client()
        tv_interval = getattr(Interval, _TV_INTERVAL_NAMES[self.interval])
        raw = client.get_hist(
            symbol=symbol,
            exchange=self.exchange,
            interval=tv_interval,
            n_bars=n_bars,
        )
        if raw is None or raw.empty:
            raise RuntimeError("tvdatafeed boş veri döndürdü")
        raw = raw.drop(columns=["symbol"], errors="ignore")
        return _normalize_columns(raw)

    def _fetch_yfinance(self, symbol: str, n_bars: int) -> pd.DataFrame:
        import yfinance as yf

        yf_symbol = _bist_symbol_for_yfinance(symbol) if self.exchange.upper() == "BIST" else symbol
        yf_interval = _YF_INTERVAL_MAP[self.interval]

        # yfinance dakikalık barlarda geriye dönük süreyi sınırlar; makul bir pencere seçelim.
        period = self._period_for_yfinance(n_bars, yf_interval)
        raw = yf.download(
            yf_symbol,
            period=period,
            interval=yf_interval,
            progress=False,
            auto_adjust=False,
            multi_level_index=False,
        )
        if raw is None or raw.empty:
            raise RuntimeError(f"yfinance boş veri döndürdü: {yf_symbol}")
        df = _normalize_columns(raw)
        return df.tail(n_bars)

    @staticmethod
    def _period_for_yfinance(n_bars: int, yf_interval: str) -> str:
        if yf_interval == "1d":
            years = max(1, int(np.ceil(n_bars / 252)) + 1)
            return f"{years}y"
        intraday_max_days = {"1m": 7, "5m": 60, "15m": 60, "30m": 60, "60m": 730}
        return f"{intraday_max_days.get(yf_interval, 60)}d"
