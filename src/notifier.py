"""
src/notifier.py
=================

Canlı tarama sinyalleri için basit bildirim katmanı.

Birincil kanal Telegram'dır (kurulumu ücretsiz ve hızlıdır: @BotFather'dan bir
bot token'ı ve kendi chat ID'nizi almanız yeterlidir). Telegram kimlik
bilgileri verilmezse (veya gönderim başarısız olursa) mesaj otomatik olarak
konsola yazdırılır - böylece bildirim kanalı henüz kurulmamışken bile
canlı tarama döngüsü sorunsuz çalışır ve test edilebilir.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("bist_bot.notifier")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def send_telegram_message(message: str, bot_token: str, chat_id: str, timeout: int = 10) -> bool:
    """Telegram Bot API üzerinden mesaj gönderir. Başarılıysa True döner."""
    import requests

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram API hata döndürdü: {data}")
    return True


def notify(
    message: str,
    telegram_bot_token: Optional[str] = None,
    telegram_chat_id: Optional[str] = None,
) -> str:
    """Mesajı yapılandırılmış kanaldan gönderir; yapılandırılmamışsa/başarısız olursa
    konsola yazdırır.

    Returns
    -------
    "telegram" veya "console" - mesajın nihai olarak hangi kanaldan iletildiği.
    """
    if telegram_bot_token and telegram_chat_id:
        try:
            send_telegram_message(message, telegram_bot_token, telegram_chat_id)
            return "telegram"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Telegram bildirimi başarısız, konsola yazdırılıyor: %s", exc)

    print(f"[BİLDİRİM]\n{message}\n")
    return "console"
