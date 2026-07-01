"""
src/live_scanner.py
=====================

BIST seans saatlerine (varsayılan 10:00-18:00, Europe/Istanbul) kilitli,
sürekli çalışan canlı tarama döngüsü.

Tasarım gerekçesi: ücretsiz veri kaynakları (TradingView/yfinance) genellikle
~15 dakika gecikmeli veri sunar. Bu yüzden daha sık taramanın bir faydası
yoktur - aynı gecikmeli veriyi tekrar tekrar çekmiş oluruz. Bu modül,
tarama periyodunu (`LIVE_SCAN_INTERVAL_MINUTES`) veri gecikmesiyle
(`DATA_DELAY_MINUTES`) eşleştirir ve her döngüde SADECE yeni ortaya çıkan
AL sinyallerini (bir önceki turda zaten bildirilmemiş olanları) bildirir -
böylece aynı sinyal için tekrar tekrar bildirim spam'i oluşmaz.

Önemli: Bu modül yalnızca BİLDİRİM üretir (Telegram/konsol); gerçek para ile
emir gönderme veya broker/API entegrasyonu içermez.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from datetime import time as dt_time
from typing import Callable, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from src import config
from src.data_loader import BistDataLoader
from src.notifier import notify
from src.scanner import scan_market

logger = logging.getLogger("bist_bot.live_scanner")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _tz() -> ZoneInfo:
    return ZoneInfo(config.MARKET_TIMEZONE)


def is_bist_open(now: Optional[datetime] = None) -> bool:
    """BIST'in şu an (varsayılan: gerçek zaman) açık olup olmadığını döndürür.

    Hafta sonu (Cumartesi/Pazar) ve seans saatleri (MARKET_OPEN-MARKET_CLOSE)
    dışındaki zamanları kapalı sayar. Resmi tatiller dahil değildir.
    """
    now = now or datetime.now(_tz())
    if now.weekday() >= 5:  # 5=Cumartesi, 6=Pazar
        return False
    open_t = dt_time.fromisoformat(config.MARKET_OPEN)
    close_t = dt_time.fromisoformat(config.MARKET_CLOSE)
    return open_t <= now.time() <= close_t


def seconds_until_market_open(now: Optional[datetime] = None) -> float:
    """Bir sonraki BIST açılışına kadar kalan saniyeyi hesaplar (hafta sonlarını atlar)."""
    now = now or datetime.now(_tz())
    open_t = dt_time.fromisoformat(config.MARKET_OPEN)
    candidate = now.replace(hour=open_t.hour, minute=open_t.minute, second=0, microsecond=0)

    if now.time() >= open_t or now.weekday() >= 5:
        candidate += timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)

    return max(0.0, (candidate - now).total_seconds())


def _current_al_signals(scan_df: pd.DataFrame, require_tradable: bool) -> pd.DataFrame:
    """scan_market() çıktısından şu an AL sinyali veren adayları döndürür (proba_up'a göre sıralı)."""
    ok = scan_df[(scan_df["status"] == "ok") & (scan_df["signal"] == "AL")].copy()
    if require_tradable and "tradable" in ok.columns and ok["tradable"].notna().any():
        ok = ok[ok["tradable"] != False]  # noqa: E712
    return ok.sort_values("proba_up", ascending=False)


def format_alert(row: pd.Series) -> str:
    """Tek bir AL sinyali için Telegram/konsol bildirim mesajı oluşturur."""
    atr_pct = row.get("atr_pct")
    close = row.get("close")
    stop_line = ""
    if pd.notna(atr_pct) and pd.notna(close):
        atr_abs = atr_pct * close
        stop = close - config.ATR_STOP_MULTIPLIER * atr_abs
        take_profit = close + config.ATR_TAKE_PROFIT_MULTIPLIER * atr_abs
        stop_line = f"\nÖnerilen Stop-Loss: {stop:.2f}\nÖnerilen Take-Profit: {take_profit:.2f}"

    as_of = row.get("as_of")
    return (
        f"🟢 <b>{row['symbol']}</b> - AL sinyali\n"
        f"Yukarı yön olasılığı: {row['proba_up']:.1%}\n"
        f"Son fiyat: {close:.2f}"
        f"{stop_line}\n"
        f"Bar zamanı: {as_of} (veri ~{config.DATA_DELAY_MINUTES} dk gecikmeli olabilir)"
    )


def run_live_scan_loop(
    symbols: Optional[list[str]] = None,
    loader: Optional[BistDataLoader] = None,
    probability_threshold: float = config.ENTRY_PROBABILITY_THRESHOLD,
    poll_interval_minutes: int = config.LIVE_SCAN_INTERVAL_MINUTES,
    require_tradable: bool = True,
    telegram_bot_token: Optional[str] = None,
    telegram_chat_id: Optional[str] = None,
    max_workers: int = 1,
    max_iterations: Optional[int] = None,
    scan_fn: Optional[Callable[..., pd.DataFrame]] = None,
    now_fn: Optional[Callable[[], datetime]] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    """BIST seans saatleri boyunca sürekli tarayıp yeni AL sinyallerini bildirir.

    Parameters
    ----------
    symbols: taranacak semboller (None ise src.scanner.fetch_bist_symbols()
        ile TÜM BIST hisseleri kullanılır).
    poll_interval_minutes: tarama periyodu (varsayılan 15 dk - veri
        gecikmesiyle eşleşir).
    telegram_bot_token / telegram_chat_id: verilirse bildirimler Telegram'a
        gönderilir; verilmezse konsola yazdırılır (bkz. src.notifier).
    max_iterations: yalnızca test/tek seferlik çalıştırma için; None ise
        sınırsız (piyasa kapanana kadar, kapalıysa açılışı bekleyerek) çalışır.
    scan_fn / now_fn / sleep_fn: test edilebilirlik için enjekte edilebilir
        bağımlılıklar (varsayılanları gerçek tarama/saat/uyku fonksiyonlarıdır).

    Not: Bu döngü yalnızca bildirim üretir; gerçek emir göndermez. Google
    Colab'in ücretsiz oturumları günler boyu kesintisiz çalışmayı garanti
    etmez - pratikte her gün yeniden başlatmanız gerekebilir.
    """
    now_fn = now_fn or (lambda: datetime.now(_tz()))
    loader = loader or BistDataLoader(interval="15m")

    def _default_scan_fn() -> pd.DataFrame:
        return scan_market(
            symbols,
            loader=loader,
            force_retrain=False,  # canlı döngüde yeniden eğitim yapılmaz; bkz. MODEL_MAX_AGE_DAYS
            apply_filters=True,
            max_workers=max_workers,
        )

    scan_fn = scan_fn or _default_scan_fn

    previously_signaled: set[str] = set()
    iteration = 0

    logger.info(
        "Canlı tarama döngüsü başlıyor (periyot=%d dk, eşik=%.0f%%, seans=%s-%s %s).",
        poll_interval_minutes, probability_threshold * 100, config.MARKET_OPEN, config.MARKET_CLOSE, config.MARKET_TIMEZONE,
    )

    try:
        while max_iterations is None or iteration < max_iterations:
            now = now_fn()

            if not is_bist_open(now):
                wait_s = seconds_until_market_open(now)
                logger.info("Piyasa kapalı (%s). Açılışa kadar bekleniyor: ~%.0f dk.", now, wait_s / 60)
                sleep_fn(wait_s if max_iterations is not None else min(wait_s, 3600))
                iteration += 1
                if max_iterations is not None:
                    continue
                previously_signaled = set()  # yeni seans -> önceki gün sinyalleri sıfırlanır
                continue

            scan_df = scan_fn()
            current_signaled = set(_current_al_signals(scan_df, require_tradable)["symbol"])
            new_signals = current_signaled - previously_signaled

            candidates = _current_al_signals(scan_df, require_tradable)
            for symbol in sorted(new_signals):
                row = candidates[candidates["symbol"] == symbol].iloc[0]
                message = format_alert(row)
                channel = notify(message, telegram_bot_token, telegram_chat_id)
                logger.info("[%s] yeni AL sinyali bildirildi (kanal=%s).", symbol, channel)

            if not new_signals:
                logger.info("Yeni sinyal yok (%d aktif AL sinyali izleniyor).", len(current_signaled))

            previously_signaled = current_signaled
            iteration += 1
            if max_iterations is not None and iteration >= max_iterations:
                break
            sleep_fn(poll_interval_minutes * 60)

    except KeyboardInterrupt:
        logger.info("Canlı tarama döngüsü kullanıcı tarafından durduruldu.")
