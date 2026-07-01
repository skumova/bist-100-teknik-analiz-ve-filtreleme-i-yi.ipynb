"""
Merkezi konfigürasyon modülü.

Bu modül, projenin çalıştığı ortamı (yerel makine / Google Colab / Google Drive)
otomatik olarak algılar ve tüm modüllerin ortak kullanacağı yol, sabit ve
parametre değerlerini tek bir yerden yönetir.
"""

from __future__ import annotations

import os
from pathlib import Path


def _detect_project_root() -> Path:
    """Çalışma ortamına göre proje kök dizinini bulur.

    Öncelik sırası:
    1. BIST_BOT_ROOT ortam değişkeni (kullanıcı elle set ederse)
    2. Google Colab + Google Drive mount edilmişse Drive altındaki proje klasörü
    3. Google Colab ama Drive mount edilmemişse /content/ altı
    4. Bu dosyanın bulunduğu paketin bir üst dizini (yerel / GitHub çalışması)
    """
    env_root = os.environ.get("BIST_BOT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    try:
        import google.colab  # noqa: F401

        drive_candidates = [
            Path("/content/drive/MyDrive/bist-swing-trading-bot"),
            Path("/content/drive/My Drive/bist-swing-trading-bot"),
        ]
        for candidate in drive_candidates:
            if candidate.parent.exists():
                candidate.mkdir(parents=True, exist_ok=True)
                return candidate
        content_root = Path("/content/bist-swing-trading-bot")
        content_root.mkdir(parents=True, exist_ok=True)
        return content_root
    except ImportError:
        pass

    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = _detect_project_root()
CACHE_DIR = PROJECT_ROOT / "cache"
DATA_CACHE_DIR = CACHE_DIR / "data"
MODEL_CACHE_DIR = CACHE_DIR / "models"
LOG_DIR = PROJECT_ROOT / "logs"

for _dir in (DATA_CACHE_DIR, MODEL_CACHE_DIR, LOG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --- Veri kaynağı ayarları ---
DEFAULT_EXCHANGE = "BIST"
DEFAULT_INTERVAL = "15m"          # tvdatafeed: in_15_minute, yfinance: 15m
DEFAULT_LOOKBACK_BARS = 5000

# --- Özellik mühendisliği ayarları ---
EMA_WINDOWS = (5, 21, 50)
RSI_WINDOW = 14
ATR_WINDOW = 14
BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2.0
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
GARCH_P, GARCH_Q = 1, 1

# --- Model / walk-forward ayarları ---
WALK_FORWARD_TRAIN_BARS = 1000
WALK_FORWARD_STEP_BARS = 100
WALK_FORWARD_MIN_TEST_BARS = 50
LABEL_HORIZON_BARS = 3          # kaç bar ileriye bakarak yön etiketi üretilecek
LABEL_UP_THRESHOLD = 0.0015     # yukarı yön için minimum log getiri eşiği

XGB_PARAM_SPACE = {
    "n_estimators": (100, 600),
    "max_depth": (3, 8),
    "learning_rate": (0.01, 0.3),
}
XGB_DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_jobs": -1,
}

# --- Risk yönetimi / scalping kuralları ---
ENTRY_PROBABILITY_THRESHOLD = 0.65
EXIT_PROBABILITY_THRESHOLD = 0.35   # model görüşü tersine dönerse erken çıkış
ATR_STOP_MULTIPLIER = 1.5
ATR_TRAILING_MULTIPLIER = 1.0
ATR_TAKE_PROFIT_MULTIPLIER = 2.5
RISK_FREE_RATE_ANNUAL = 0.0

# --- Pozisyon boyutlandırma ve devre kesici (circuit breaker) ---
POSITION_SIZE_MODE = "vol_target"   # "fixed" (sabit %) veya "vol_target" (risk bazlı)
RISK_PER_TRADE_PCT = 0.01           # vol_target modunda işlem başına riske edilen sermaye oranı
MAX_CONSECUTIVE_LOSSES = 3          # bu sayıda üst üste zararlı işlemden sonra soğuma
COOLDOWN_BARS_AFTER_LOSSES = 10     # soğuma süresi (bar sayısı)
MAX_DAILY_LOSS_PCT = 0.03           # günlük zarar bu oranı aşarsa o gün yeni işlem açılmaz

# --- Ek sinyal filtreleri (gerekirse devre dışı bırakılabilir) ---
ADX_WINDOW = 14
MIN_ADX = 20.0                      # bu değerin altında (yatay piyasa) işlem açılmaz
VOLUME_WINDOW = 20
MIN_VOLUME_RATIO = 0.7              # ortalama hacmin bu oranın altındaki barlar elenir
VOL_REGIME_WINDOW = 100
VOL_REGIME_LOWER_PCT = 0.05         # aşırı sakin (düşük oynaklık) rejimi ele
VOL_REGIME_UPPER_PCT = 0.95         # aşırı oynak (kayma/gap riski yüksek) rejimi ele
REQUIRE_TREND_ALIGNMENT = True      # sadece EMA(fast) > EMA(slow) iken long aç
TREND_FAST_EMA = 21
TREND_SLOW_EMA = 50
SESSION_FILTER_ENABLED = True
MARKET_OPEN = "10:00"               # BIST seans açılışı
MARKET_CLOSE = "18:00"              # BIST seans kapanışı
SESSION_EDGE_EXCLUDE_MINUTES = 15   # açılış/kapanışa yakın gürültülü dakikalar

# --- Genel ---
RANDOM_STATE = 42
