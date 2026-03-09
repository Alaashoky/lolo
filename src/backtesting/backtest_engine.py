"""
Backtesting Engine.

Simulates trade execution on historical OHLCV data using signal logic
from the AI ensemble or rule-based strategies.

Features:
- Walk-forward validation
- Slippage and spread simulation
- Commission calculations
- Trailing stops / take profits
- Daily performance tracking
- Full trade log
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("forex_bot.backtesting.engine")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Trade:
    """Represents a single completed trade."""

    id: str
    pair: str
    direction: str          # "BUY" or "SELL"
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    lot_size: float = 0.01
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    profit_loss: float = 0.0
    profit_pips: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    reason: str = ""        # "Take Profit", "Stop Loss", "Signal Reverse", "End of Data"
    is_open: bool = True


@dataclass
class DailyStats:
    """Per-day performance snapshot."""

    date: str
    equity: float
    daily_pnl: float
    trades_opened: int = 0
    trades_closed: int = 0
    win_count: int = 0
    loss_count: int = 0


# ---------------------------------------------------------------------------
# Backtesting Engine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """
    Event-driven backtesting engine that replays historical candles.

    Args:
        initial_balance:       Starting account balance in USD.
        risk_per_trade:        Risk percentage per trade (e.g. 2.0 = 2 %).
        max_concurrent_trades: Maximum number of open positions at once.
        slippage_pips:         Fixed slippage in pips applied on every order.
        spread_pips:           Fixed spread in pips added to every entry.
        commission_percent:    Round-trip commission as % of trade value.
        pip_value:             USD value of 1 pip for 1 standard lot.
        lot_size:              Fixed lot size (overrides risk-based sizing
                               when ``use_fixed_lot=True``).
        use_fixed_lot:         Use fixed lot size rather than risk-based sizing.
        trailing_stop_pips:    Enable trailing stop (0 = disabled).
    """

    def __init__(
        self,
        initial_balance: float = 10_000.0,
        risk_per_trade: float = 2.0,
        max_concurrent_trades: int = 3,
        slippage_pips: float = 3.0,
        spread_pips: float = 2.0,
        commission_percent: float = 0.01,
        pip_value: float = 10.0,
        lot_size: float = 0.1,
        use_fixed_lot: bool = True,
        trailing_stop_pips: float = 0.0,
    ) -> None:
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.max_concurrent_trades = max_concurrent_trades
        self.slippage_pips = slippage_pips
        self.spread_pips = spread_pips
        self.commission_percent = commission_percent
        self.pip_value = pip_value
        self.lot_size = lot_size
        self.use_fixed_lot = use_fixed_lot
        self.trailing_stop_pips = trailing_stop_pips

        # State
        self.balance: float = initial_balance
        self.equity: float = initial_balance
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.daily_stats: List[DailyStats] = []

        self._current_day: Optional[str] = None
        self._day_open_equity: float = initial_balance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all state to initial values."""
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.open_trades = []
        self.closed_trades = []
        self.equity_curve = []
        self.daily_stats = []
        self._current_day = None
        self._day_open_equity = self.initial_balance

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        pair: str = "UNKNOWN",
        pip_multiplier: float = 0.0001,
    ) -> Dict:
        """
        Replay candles and execute trades based on *signals*.

        Args:
            df:             OHLCV DataFrame with DatetimeIndex.
            signals:        Series of strings (``BUY`` / ``SELL`` / ``HOLD``)
                            indexed identically to *df*.
            pair:           Trading pair name used in trade records.
            pip_multiplier: Size of 1 pip in price units (0.0001 for most
                            forex pairs, 0.01 for JPY pairs).

        Returns:
            Dictionary with summary statistics and trade list.
        """
        self.reset()

        if df.empty or signals.empty:
            logger.warning("Empty data or signals passed to BacktestEngine.run().")
            return self._summary(pair)

        df = df.copy()
        signals = signals.reindex(df.index, fill_value="HOLD")

        spread_price = self.spread_pips * pip_multiplier
        slip_price = self.slippage_pips * pip_multiplier

        for i, (ts, row) in enumerate(df.iterrows()):
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])
            signal = str(signals.iloc[i]).upper()

            # Track daily stats
            self._update_daily_tracking(ts, close)

            # 1. Update open positions (check SL/TP/trailing)
            self._update_open_positions(ts, high, low, close, pip_multiplier)

            # 2. Execute new signal
            if signal in ("BUY", "SELL") and len(self.open_trades) < self.max_concurrent_trades:
                # Avoid opening a trade in the same direction already open
                same_dir = [t for t in self.open_trades if t.direction == signal and t.pair == pair]
                if not same_dir:
                    self._open_trade(ts, signal, close, pair, spread_price, slip_price, pip_multiplier)

            # 3. Record equity curve
            unrealised = self._unrealised_pnl(close, pip_multiplier)
            self.equity = self.balance + unrealised
            self.equity_curve.append((ts, self.equity))

        # Close any remaining open trades at last close price
        last_ts = df.index[-1]
        last_close = float(df["close"].iloc[-1])
        for trade in list(self.open_trades):
            self._close_trade(trade, last_ts, last_close, pip_multiplier, reason="End of Data")

        return self._summary(pair)

    def run_walkforward(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        pair: str = "UNKNOWN",
        n_splits: int = 5,
        pip_multiplier: float = 0.0001,
    ) -> List[Dict]:
        """
        Walk-forward validation: split data into *n_splits* folds and
        run a separate backtest on each out-of-sample window.

        Returns:
            List of summary dicts, one per fold.
        """
        n = len(df)
        fold_size = n // n_splits
        results = []

        for fold in range(n_splits):
            start = fold * fold_size
            end = start + fold_size if fold < n_splits - 1 else n
            fold_df = df.iloc[start:end]
            fold_signals = signals.iloc[start:end]

            self.reset()
            summary = self.run(fold_df, fold_signals, pair=pair, pip_multiplier=pip_multiplier)
            summary["fold"] = fold + 1
            results.append(summary)
            logger.debug("Walk-forward fold %d/%d: %.2f%% return", fold + 1, n_splits, summary.get("total_return_pct", 0))

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _open_trade(
        self,
        ts: datetime,
        direction: str,
        close: float,
        pair: str,
        spread_price: float,
        slip_price: float,
        pip_multiplier: float,
    ) -> None:
        """Create and open a new trade."""
        if direction == "BUY":
            entry_price = close + spread_price + slip_price
        else:
            entry_price = close - slip_price

        # Risk-based lot sizing
        if self.use_fixed_lot:
            lot = self.lot_size
        else:
            risk_amount = self.balance * (self.risk_per_trade / 100.0)
            sl_distance_pips = 20  # default 20-pip SL
            lot = risk_amount / (sl_distance_pips * self.pip_value)
            lot = max(0.01, min(lot, 10.0))

        # Stop loss and take profit (simple ATR-based approximation)
        sl_pips = 25
        tp_pips = 50
        if direction == "BUY":
            sl = entry_price - sl_pips * pip_multiplier
            tp = entry_price + tp_pips * pip_multiplier
        else:
            sl = entry_price + sl_pips * pip_multiplier
            tp = entry_price - tp_pips * pip_multiplier

        # Commission
        commission = lot * 100_000 * entry_price * (self.commission_percent / 100.0)

        trade = Trade(
            id=str(uuid.uuid4())[:8],
            pair=pair,
            direction=direction,
            entry_time=ts,
            entry_price=entry_price,
            lot_size=lot,
            stop_loss=sl,
            take_profit=tp,
            commission=commission,
            slippage=slip_price * 100,
        )
        self.open_trades.append(trade)

        # Deduct commission immediately
        self.balance -= commission

        logger.debug(
            "Opened %s %s @ %.5f  SL=%.5f  TP=%.5f  lot=%.2f",
            direction, pair, entry_price, sl, tp, lot,
        )

    def _update_open_positions(
        self,
        ts: datetime,
        high: float,
        low: float,
        close: float,
        pip_multiplier: float,
    ) -> None:
        """Check SL/TP for each open trade and close if triggered."""
        for trade in list(self.open_trades):
            if trade.direction == "BUY":
                # Update trailing stop
                if self.trailing_stop_pips > 0 and trade.stop_loss is not None:
                    new_sl = close - self.trailing_stop_pips * pip_multiplier
                    trade.stop_loss = max(trade.stop_loss, new_sl)

                if trade.stop_loss is not None and low <= trade.stop_loss:
                    self._close_trade(trade, ts, trade.stop_loss, pip_multiplier, reason="Stop Loss")
                elif trade.take_profit is not None and high >= trade.take_profit:
                    self._close_trade(trade, ts, trade.take_profit, pip_multiplier, reason="Take Profit")

            else:  # SELL
                if self.trailing_stop_pips > 0 and trade.stop_loss is not None:
                    new_sl = close + self.trailing_stop_pips * pip_multiplier
                    trade.stop_loss = min(trade.stop_loss, new_sl)

                if trade.stop_loss is not None and high >= trade.stop_loss:
                    self._close_trade(trade, ts, trade.stop_loss, pip_multiplier, reason="Stop Loss")
                elif trade.take_profit is not None and low <= trade.take_profit:
                    self._close_trade(trade, ts, trade.take_profit, pip_multiplier, reason="Take Profit")

    def _close_trade(
        self,
        trade: Trade,
        ts: datetime,
        exit_price: float,
        pip_multiplier: float,
        reason: str = "",
    ) -> None:
        """Close a trade and record P&L."""
        trade.exit_time = ts
        trade.exit_price = exit_price
        trade.reason = reason
        trade.is_open = False

        if trade.direction == "BUY":
            pip_diff = (exit_price - trade.entry_price) / pip_multiplier
        else:
            pip_diff = (trade.entry_price - exit_price) / pip_multiplier

        trade.profit_pips = pip_diff
        trade.profit_loss = pip_diff * self.pip_value * trade.lot_size

        self.balance += trade.profit_loss
        self.open_trades.remove(trade)
        self.closed_trades.append(trade)

        logger.debug(
            "Closed %s %s @ %.5f  P&L=%.2f  reason=%s",
            trade.direction, trade.pair, exit_price, trade.profit_loss, reason,
        )

    def _unrealised_pnl(self, current_price: float, pip_multiplier: float) -> float:
        pnl = 0.0
        for t in self.open_trades:
            if t.direction == "BUY":
                pnl += (current_price - t.entry_price) / pip_multiplier * self.pip_value * t.lot_size
            else:
                pnl += (t.entry_price - current_price) / pip_multiplier * self.pip_value * t.lot_size
        return pnl

    def _update_daily_tracking(self, ts: datetime, close: float) -> None:
        """Track per-day stats."""
        day = str(ts.date()) if hasattr(ts, "date") else str(ts)[:10]
        if day != self._current_day:
            if self._current_day is not None:
                self.daily_stats.append(
                    DailyStats(
                        date=self._current_day,
                        equity=self.equity,
                        daily_pnl=self.equity - self._day_open_equity,
                    )
                )
            self._current_day = day
            self._day_open_equity = self.equity

    def _summary(self, pair: str) -> Dict:
        """Compile a summary dictionary from closed trades."""
        trades = self.closed_trades
        n_trades = len(trades)
        n_wins = sum(1 for t in trades if t.profit_loss > 0)
        n_losses = n_trades - n_wins
        total_pnl = sum(t.profit_loss for t in trades)
        gross_profit = sum(t.profit_loss for t in trades if t.profit_loss > 0)
        gross_loss = abs(sum(t.profit_loss for t in trades if t.profit_loss < 0))

        win_rate = n_wins / n_trades if n_trades > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        total_return_pct = (self.balance - self.initial_balance) / self.initial_balance * 100

        equity_values = [e for _, e in self.equity_curve]
        max_drawdown, max_drawdown_pct = _compute_max_drawdown(equity_values, self.initial_balance)

        return {
            "pair": pair,
            "initial_balance": self.initial_balance,
            "final_balance": self.balance,
            "total_return_pct": round(total_return_pct, 2),
            "total_pnl": round(total_pnl, 2),
            "total_trades": n_trades,
            "winning_trades": n_wins,
            "losing_trades": n_losses,
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else None,
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "max_drawdown": round(max_drawdown, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "trades": [_trade_to_dict(t) for t in trades],
            "equity_curve": [(str(ts), eq) for ts, eq in self.equity_curve],
        }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _compute_max_drawdown(
    equity_values: List[float], initial: float
) -> Tuple[float, float]:
    """Compute maximum drawdown (absolute and percentage)."""
    if not equity_values:
        return 0.0, 0.0

    peak = initial
    max_dd = 0.0
    for eq in equity_values:
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd

    max_dd_pct = (max_dd / peak * 100) if peak > 0 else 0.0
    return max_dd, max_dd_pct


def _trade_to_dict(trade: Trade) -> Dict:
    return {
        "id": trade.id,
        "pair": trade.pair,
        "direction": trade.direction,
        "entry_time": str(trade.entry_time),
        "entry_price": round(trade.entry_price, 5),
        "exit_time": str(trade.exit_time) if trade.exit_time else None,
        "exit_price": round(trade.exit_price, 5) if trade.exit_price else None,
        "lot_size": trade.lot_size,
        "stop_loss": round(trade.stop_loss, 5) if trade.stop_loss else None,
        "take_profit": round(trade.take_profit, 5) if trade.take_profit else None,
        "profit_loss": round(trade.profit_loss, 2),
        "profit_pips": round(trade.profit_pips, 1),
        "commission": round(trade.commission, 4),
        "reason": trade.reason,
    }
