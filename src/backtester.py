"""
src/backtester.py
==================

Risk yönetimi ve backtest motoru.

- Giriş: XGBoost'un ürettiği yukarı yön olasılığı (`proba_up`) belirlenen
  eşiği (varsayılan %65) aştığında ve (varsa) `tradable` filtre maskesi
  True ise long pozisyon açılır (sadece BUY / vur-kaç, açığa satış yoktur).
  `tradable` kolonu src.filters.build_filter_mask() ile üretilir (ADX trend
  gücü, hacim oranı, volatilite rejimi, trend hizası, işlem seansı).
- Çıkış: ATR tabanlı dinamik stop-loss, ATR tabanlı take-profit, fiyat lehe
  hareket ettikçe yükselen (iz süren) trailing stop ve model görüşü tersine
  dönerse (proba_up < exit_probability_threshold) erken çıkış.
- Pozisyon boyutlandırma: "fixed" (sermayenin sabit yüzdesi) veya
  "vol_target" (ATR'ye göre işlem başına sabit risk yüzdesi - kurumsal
  standartta risk bazlı boyutlandırma).
- Devre kesici (circuit breaker): ardışık kayıp sayısı eşiği aşarsa soğuma
  periyodu, günlük zarar sermayenin belirli bir yüzdesini aşarsa o gün için
  yeni işlem açılmaz.
- Her çalıştırmada Sharpe Ratio, Maximum Drawdown ve Win/Loss Rate hesaplanır;
  bu fonksiyonlar src.model.WalkForwardEngine'in `financial_metrics_fn`
  callback'i olarak da kullanılabilir.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger("bist_bot.backtester")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


_BARS_PER_YEAR = {
    "1m": 252 * 60 * 8,
    "5m": 252 * 12 * 8,
    "15m": 252 * 4 * 8,
    "30m": 252 * 2 * 8,
    "1h": 252 * 8,
    "4h": 252 * 2,
    "1d": 252,
}


def estimate_periods_per_year(interval: str = config.DEFAULT_INTERVAL) -> int:
    return _BARS_PER_YEAR.get(interval, 252 * 4 * 8)


@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str
    shares: float
    pnl: float
    pnl_pct: float


def _open_new_position(
    row: pd.Series,
    ts: pd.Timestamp,
    capital: float,
    fee_bps: float,
    atr_col: str,
    stop_mult: float,
    tp_mult: float,
    position_size_mode: str,
    position_size_pct: float,
    risk_per_trade_pct: float,
) -> dict:
    entry_price = float(row["close"])
    atr_at_entry = float(row[atr_col])
    stop_distance = stop_mult * atr_at_entry

    if position_size_mode == "vol_target" and stop_distance > 0:
        risk_amount = capital * risk_per_trade_pct
        shares = risk_amount / stop_distance
        allocation = shares * entry_price
        if allocation > capital:  # kaldıraçsız: sermayeyi aşan tahsisatı kırp
            allocation = capital
            shares = allocation / entry_price
    else:
        allocation = capital * position_size_pct
        shares = allocation / entry_price

    entry_fee = allocation * (fee_bps / 10_000.0)

    return {
        "entry_time": ts,
        "entry_price": entry_price,
        "atr_at_entry": atr_at_entry,
        "shares": shares,
        "entry_fee": entry_fee,
        "stop_loss": entry_price - stop_mult * atr_at_entry,
        "take_profit": entry_price + tp_mult * atr_at_entry,
        "trailing_stop": entry_price - stop_mult * atr_at_entry,
    }


def run_backtest(
    df: pd.DataFrame,
    proba_col: str = "proba_up",
    atr_col: str = f"atr_{config.ATR_WINDOW}",
    tradable_col: str = "tradable",
    probability_threshold: float = config.ENTRY_PROBABILITY_THRESHOLD,
    exit_probability_threshold: Optional[float] = config.EXIT_PROBABILITY_THRESHOLD,
    stop_mult: float = config.ATR_STOP_MULTIPLIER,
    trail_mult: float = config.ATR_TRAILING_MULTIPLIER,
    tp_mult: float = config.ATR_TAKE_PROFIT_MULTIPLIER,
    initial_capital: float = 100_000.0,
    position_size_mode: str = config.POSITION_SIZE_MODE,
    position_size_pct: float = 1.0,
    risk_per_trade_pct: float = config.RISK_PER_TRADE_PCT,
    fee_bps: float = 5.0,
    max_consecutive_losses: Optional[int] = config.MAX_CONSECUTIVE_LOSSES,
    cooldown_bars_after_losses: int = config.COOLDOWN_BARS_AFTER_LOSSES,
    max_daily_loss_pct: Optional[float] = config.MAX_DAILY_LOSS_PCT,
    periods_per_year: Optional[int] = None,
) -> dict:
    """XGBoost sinyalleriyle ATR tabanlı risk yönetimini birleştiren backtest.

    Parameters
    ----------
    df: `open, high, low, close, atr_...` ve `proba_up` (XGBoost olasılığı)
        kolonlarını içeren, zaman sırasına göre sıralı DataFrame. `tradable`
        kolonu varsa (bkz. src.filters.build_filter_mask) ek filtre olarak
        uygulanır; yoksa tüm barlar filtre açısından uygun kabul edilir.
    exit_probability_threshold: pozisyon açıkken modelin yukarı yön olasılığı
        bu değerin altına düşerse (görüş tersine döndüyse) pozisyon erken
        kapatılır. None verilirse bu davranış devre dışı kalır.
    position_size_mode: "fixed" (sermayenin `position_size_pct` kadarı) veya
        "vol_target" (her işlemde sermayenin `risk_per_trade_pct` kadarını
        stop mesafesine göre riske eden, ATR'ye duyarlı boyutlandırma).
    max_consecutive_losses / cooldown_bars_after_losses: art arda bu kadar
        zararlı işlemden sonra belirtilen bar sayısı kadar yeni işlem açılmaz.
    max_daily_loss_pct: günün başındaki sermayeye göre günlük zarar bu oranı
        aşarsa, o gün için yeni işlem açılmaz (devre kesici).
    fee_bps: işlem başına baz puan cinsinden komisyon (varsayılan 5 bps).

    Returns
    -------
    dict: {"trades": DataFrame, "equity_curve": Series, "metrics": dict}
    """
    required = {"open", "high", "low", "close", atr_col, proba_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"run_backtest: eksik kolonlar: {missing}")

    periods_per_year = periods_per_year or estimate_periods_per_year()
    has_tradable_filter = tradable_col in df.columns

    capital = initial_capital
    position: Optional[dict] = None
    trades: list[Trade] = []
    equity_curve = []
    equity_index = []

    consecutive_losses = 0
    cooldown_until_idx = -1
    current_day = None
    capital_at_day_start = capital
    daily_pnl = 0.0
    day_halted = False

    for i, (ts, row) in enumerate(df.iterrows()):
        bar_date = ts.date() if hasattr(ts, "date") else None
        if bar_date is not None and bar_date != current_day:
            current_day = bar_date
            capital_at_day_start = capital
            daily_pnl = 0.0
            day_halted = False

        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        atr = float(row[atr_col]) if pd.notna(row[atr_col]) else None
        proba_up = float(row[proba_col]) if pd.notna(row[proba_col]) else 0.0

        if position is not None:
            # Trailing stop'u fiyat lehte hareket ettikçe yukarı çek (asla aşağı çekme)
            if atr is not None:
                candidate_trail = close - trail_mult * atr
                position["trailing_stop"] = max(position["trailing_stop"], candidate_trail)

            effective_stop = max(position["stop_loss"], position["trailing_stop"])

            exit_price = None
            exit_reason = None
            # Kötümser (konservatif) varsayım: aynı barda hem stop hem TP'ye
            # değinilmişse önce stop-loss tetiklenmiş kabul edilir.
            if low <= effective_stop:
                exit_price = min(effective_stop, high)
                exit_reason = "trailing_stop" if effective_stop == position["trailing_stop"] else "stop_loss"
            elif high >= position["take_profit"]:
                exit_price = position["take_profit"]
                exit_reason = "take_profit"
            elif exit_probability_threshold is not None and proba_up < exit_probability_threshold:
                exit_price = close
                exit_reason = "model_exit"

            if exit_price is not None:
                exit_fee = position["shares"] * exit_price * (fee_bps / 10_000.0)
                gross_pnl = (exit_price - position["entry_price"]) * position["shares"]
                pnl = gross_pnl - position["entry_fee"] - exit_fee
                pnl_pct = pnl / (position["entry_price"] * position["shares"])

                capital += pnl
                daily_pnl += pnl
                trades.append(
                    Trade(
                        entry_time=position["entry_time"],
                        entry_price=position["entry_price"],
                        exit_time=ts,
                        exit_price=exit_price,
                        exit_reason=exit_reason,
                        shares=position["shares"],
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                    )
                )
                position = None

                consecutive_losses = consecutive_losses + 1 if pnl <= 0 else 0
                if max_consecutive_losses is not None and consecutive_losses >= max_consecutive_losses:
                    cooldown_until_idx = i + cooldown_bars_after_losses
                    consecutive_losses = 0
                if (
                    max_daily_loss_pct is not None
                    and capital_at_day_start > 0
                    and daily_pnl <= -max_daily_loss_pct * capital_at_day_start
                ):
                    day_halted = True

        is_tradable = bool(row[tradable_col]) if has_tradable_filter else True
        can_open = (
            position is None
            and proba_up >= probability_threshold
            and atr is not None
            and atr > 0
            and is_tradable
            and not day_halted
            and i >= cooldown_until_idx
        )
        if can_open:
            position = _open_new_position(
                row, ts, capital, fee_bps, atr_col, stop_mult, tp_mult,
                position_size_mode, position_size_pct, risk_per_trade_pct,
            )

        # Mark-to-market öz sermaye (Sharpe / drawdown hesaplaması için)
        unrealized = 0.0
        if position is not None:
            unrealized = (close - position["entry_price"]) * position["shares"]
        equity_curve.append(capital + unrealized)
        equity_index.append(ts)

    # Açık kalan pozisyon varsa dönem sonunda kapat (mark-to-market realize et)
    if position is not None:
        last_row = df.iloc[-1]
        exit_price = float(last_row["close"])
        exit_fee = position["shares"] * exit_price * (fee_bps / 10_000.0)
        gross_pnl = (exit_price - position["entry_price"]) * position["shares"]
        pnl = gross_pnl - position["entry_fee"] - exit_fee
        pnl_pct = pnl / (position["entry_price"] * position["shares"])
        capital += pnl
        trades.append(
            Trade(
                entry_time=position["entry_time"],
                entry_price=position["entry_price"],
                exit_time=df.index[-1],
                exit_price=exit_price,
                exit_reason="period_end",
                shares=position["shares"],
                pnl=pnl,
                pnl_pct=pnl_pct,
            )
        )
        equity_curve[-1] = capital

    equity_series = pd.Series(equity_curve, index=equity_index, name="equity")
    trades_df = pd.DataFrame([t.__dict__ for t in trades])

    metrics = compute_metrics_from_results(equity_series, trades_df, initial_capital, periods_per_year)
    return {"trades": trades_df, "equity_curve": equity_series, "metrics": metrics}


def compute_metrics_from_results(
    equity_curve: pd.Series,
    trades_df: pd.DataFrame,
    initial_capital: float,
    periods_per_year: int,
) -> dict:
    if equity_curve.empty:
        return {
            "total_return_pct": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate": 0.0,
            "num_trades": 0,
            "avg_pnl_pct": 0.0,
            "profit_factor": 0.0,
        }

    period_returns = equity_curve.pct_change().dropna()
    total_return_pct = (equity_curve.iloc[-1] / initial_capital - 1.0) * 100

    if period_returns.std(ddof=0) > 1e-12:
        sharpe = (period_returns.mean() / period_returns.std(ddof=0)) * np.sqrt(periods_per_year)
    else:
        sharpe = 0.0

    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown_pct = drawdown.min() * 100

    if not trades_df.empty:
        win_rate = float((trades_df["pnl"] > 0).mean())
        avg_pnl_pct = float(trades_df["pnl_pct"].mean())
        gross_profit = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum())
        gross_loss = float(-trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 1e-9 else float("inf") if gross_profit > 0 else 0.0
    else:
        win_rate = 0.0
        avg_pnl_pct = 0.0
        profit_factor = 0.0

    return {
        "total_return_pct": float(total_return_pct),
        "sharpe_ratio": float(sharpe),
        "max_drawdown_pct": float(max_drawdown_pct),
        "win_rate": win_rate,
        "num_trades": int(len(trades_df)),
        "avg_pnl_pct": avg_pnl_pct,
        "profit_factor": profit_factor,
    }


def compute_financial_metrics(fold_df: pd.DataFrame, **kwargs) -> dict:
    """src.model.WalkForwardEngine(financial_metrics_fn=...) için ince sarmalayıcı.

    Her walk-forward fold'unun out-of-sample sinyalleriyle mini bir backtest
    çalıştırıp sadece finansal metrikleri (Sharpe/Drawdown/WinRate) döndürür.
    """
    result = run_backtest(fold_df, **kwargs)
    return result["metrics"]
