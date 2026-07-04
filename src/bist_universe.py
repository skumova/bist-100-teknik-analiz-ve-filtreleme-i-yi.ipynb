"""
src/bist_universe.py
====================

Tüm BIST sembol evrenini döndüren hafif yardımcı (yalnızca `requests`/`pandas`
ve `config`'e bağlıdır; `src.scanner`'ın ağır ML bağımlılıklarını çekmez).

Bu modül, notebook'a gömülebilecek kadar bağımsızdır. `src.scanner.fetch_bist_symbols`
ile aynı işi görür ama izole çalışır.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("bist_bot.bist_universe")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _from_tradingview(min_len: int = 2, max_len: int = 7, timeout: int = 20) -> list[str]:
    """TradingView genel tarayıcı API'sinden BIST'te işlem gören tüm hisseler."""
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
    symbols: list[str] = []
    for row in resp.json().get("data", []):
        cols = row.get("d") or []
        ticker = cols[0] if cols else None
        if ticker and isinstance(ticker, str):
            clean = ticker.split(":")[-1].strip().upper()
            if clean.isalpha() and min_len <= len(clean) <= max_len:
                symbols.append(clean)
    return list(dict.fromkeys(symbols))


def fetch_bist_symbols() -> list[str]:
    """BIST'te işlem gören TÜM hisselerin (~600+) güncel listesi.

    1) TradingView Scanner API → 2) `config.BIST100_SYMBOLS` çekirdek yedeği.
    """
    try:
        symbols = _from_tradingview()
        if symbols:
            logger.info("TradingView Scanner: %d BIST hissesi çekildi.", len(symbols))
            return symbols
    except Exception as exc:  # noqa: BLE001
        logger.warning("TradingView Scanner API başarısız: %s", exc)

    try:
        from src import config
        core = list(config.BIST100_SYMBOLS)
    except Exception:  # noqa: BLE001
        core = []
    logger.warning("Dinamik kaynak başarısız; çekirdek liste kullanılıyor (%d hisse).", len(core))
    return core
