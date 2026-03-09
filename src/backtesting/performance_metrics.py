"""
Performance Metrics Calculator.

Computes a comprehensive set of risk-adjusted and trade-level metrics
from a backtest result dictionary or a series of equity values.

Metrics:
- Total Return (%)
- Annualised Return (%)
- CAGR
- Win Rate
- Profit Factor
- Sharpe Ratio (annualised)
- Sortino Ratio (annualised)
- Calmar Ratio
- Max Drawdown (%)
- Recovery Factor
- Average Win / Average Loss
- Risk-Reward Ratio
- Consecutive wins / losses
- Number of trades
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("forex_bot.backtesting.metrics")

# Annualisation factor for hourly data
_TRADING_HOURS_PER_YEAR = 8_760  # 365 * 24


class PerformanceMetrics:
    """
    Compute and expose comprehensive backtest performance metrics.

    Args:
        risk_free_rate: Annual risk-free rate used in Sharpe/Sortino (default: 2 %).
    """

    def __init__(self, risk_free_rate: float = 0.02) -> None:
        self.risk_free_rate = risk_free_rate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, summary: Dict) -> Dict:
        """
        Compute all metrics from a backtest summary dict produced by
        ``BacktestEngine._summary()``.

        Returns:
            Extended dict with all performance metrics added.
        """
        metrics: Dict = {}
        trades = summary.get("trades", [])
        equity_curve = summary.get("equity_curve", [])
        initial = summary.get("initial_balance", 10_000.0)
        final = summary.get("final_balance", initial)

        equity_values = [eq for _, eq in equity_curve]

        # Basic P&L
        metrics.update(self._basic_metrics(initial, final, trades))

        # Return series
        returns = self._equity_to_returns(equity_values)
        metrics.update(self._risk_metrics(returns, initial, equity_values))

        # Trade-level stats
        metrics.update(self._trade_metrics(trades))

        # Combine with original summary (metrics override duplicate keys)
        result = {**summary, **metrics}
        return result

    def compute_from_equity(
        self,
        equity_values: List[float],
        initial_balance: float = 10_000.0,
    ) -> Dict:
        """Compute metrics from a plain equity curve."""
        returns = self._equity_to_returns(equity_values)
        metrics: Dict = {
            "initial_balance": initial_balance,
            "final_balance": equity_values[-1] if equity_values else initial_balance,
        }
        metrics.update(self._risk_metrics(returns, initial_balance, equity_values))
        return metrics

    # ------------------------------------------------------------------
    # Metric groups
    # ------------------------------------------------------------------

    def _basic_metrics(
        self, initial: float, final: float, trades: List[Dict]
    ) -> Dict:
        total_return_pct = (final - initial) / initial * 100 if initial > 0 else 0.0
        n_trades = len(trades)
        n_wins = sum(1 for t in trades if t.get("profit_loss", 0) > 0)
        n_losses = n_trades - n_wins

        gross_profit = sum(t["profit_loss"] for t in trades if t.get("profit_loss", 0) > 0)
        gross_loss = abs(sum(t["profit_loss"] for t in trades if t.get("profit_loss", 0) < 0))

        win_rate = n_wins / n_trades if n_trades > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

        avg_win = gross_profit / n_wins if n_wins > 0 else 0.0
        avg_loss = gross_loss / n_losses if n_losses > 0 else 0.0
        risk_reward = avg_win / avg_loss if avg_loss > 0 else None

        return {
            "total_return_pct": round(total_return_pct, 2),
            "total_trades": n_trades,
            "winning_trades": n_wins,
            "losing_trades": n_losses,
            "win_rate": round(win_rate * 100, 2),
            "profit_factor": round(profit_factor, 4) if profit_factor is not None else None,
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "risk_reward_ratio": round(risk_reward, 2) if risk_reward is not None else None,
        }

    def _risk_metrics(
        self,
        returns: np.ndarray,
        initial: float,
        equity_values: List[float],
    ) -> Dict:
        """Sharpe, Sortino, Calmar, Max DD, CAGR, etc."""
        if len(equity_values) < 2:
            return {}

        # Annualised return
        n_bars = len(equity_values)
        years = n_bars / _TRADING_HOURS_PER_YEAR
        final = equity_values[-1]

        if years > 0 and initial > 0:
            cagr = (final / initial) ** (1 / years) - 1
        else:
            cagr = 0.0

        # Max drawdown
        max_dd, max_dd_pct = _max_drawdown(equity_values)

        # Sharpe ratio
        sharpe = self._sharpe(returns)

        # Sortino ratio
        sortino = self._sortino(returns)

        # Calmar ratio (annualised return / max drawdown)
        calmar = (cagr / (max_dd_pct / 100)) if max_dd_pct > 0 else None

        # Recovery factor (net profit / max drawdown)
        net_profit = final - initial
        recovery = net_profit / max_dd if max_dd > 0 else None

        return {
            "cagr_pct": round(cagr * 100, 2),
            "annualised_return_pct": round(cagr * 100, 2),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "calmar_ratio": round(calmar, 4) if calmar is not None else None,
            "max_drawdown": round(max_dd, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "recovery_factor": round(recovery, 4) if recovery is not None else None,
        }

    def _trade_metrics(self, trades: List[Dict]) -> Dict:
        """Consecutive wins/losses, average hold time, etc."""
        if not trades:
            return {}

        pnls = [t.get("profit_loss", 0) for t in trades]

        max_consec_wins = _max_consecutive(pnls, positive=True)
        max_consec_losses = _max_consecutive(pnls, positive=False)
        avg_pnl = float(np.mean(pnls))
        std_pnl = float(np.std(pnls))

        return {
            "max_consecutive_wins": max_consec_wins,
            "max_consecutive_losses": max_consec_losses,
            "avg_pnl_per_trade": round(avg_pnl, 2),
            "std_pnl_per_trade": round(std_pnl, 2),
        }

    # ------------------------------------------------------------------
    # Ratio calculators
    # ------------------------------------------------------------------

    def _sharpe(self, returns: np.ndarray) -> float:
        """Annualised Sharpe Ratio."""
        if len(returns) < 2:
            return 0.0
        mean_r = float(np.mean(returns))
        std_r = float(np.std(returns, ddof=1))
        if std_r == 0:
            return 0.0
        rf_hourly = self.risk_free_rate / _TRADING_HOURS_PER_YEAR
        excess = mean_r - rf_hourly
        return float(excess / std_r * math.sqrt(_TRADING_HOURS_PER_YEAR))

    def _sortino(self, returns: np.ndarray) -> float:
        """Annualised Sortino Ratio (downside deviation only)."""
        if len(returns) < 2:
            return 0.0
        rf_hourly = self.risk_free_rate / _TRADING_HOURS_PER_YEAR
        excess = returns - rf_hourly
        downside = excess[excess < 0]
        if len(downside) == 0:
            return float("inf")
        downside_std = float(np.std(downside, ddof=1))
        if downside_std == 0:
            return 0.0
        mean_excess = float(np.mean(excess))
        return float(mean_excess / downside_std * math.sqrt(_TRADING_HOURS_PER_YEAR))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _equity_to_returns(equity_values: List[float]) -> np.ndarray:
        """Convert equity curve to per-bar percentage returns."""
        if len(equity_values) < 2:
            return np.array([], dtype=float)
        arr = np.array(equity_values, dtype=float)
        return np.diff(arr) / (arr[:-1] + 1e-10)


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------


def _max_drawdown(equity_values: List[float]) -> Tuple[float, float]:
    """Return (max_dd_absolute, max_dd_percent)."""
    if not equity_values:
        return 0.0, 0.0
    peak = equity_values[0]
    max_dd = 0.0
    for eq in equity_values:
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd
    max_dd_pct = (max_dd / peak * 100) if peak > 0 else 0.0
    return float(max_dd), float(max_dd_pct)


def _max_consecutive(pnls: List[float], positive: bool) -> int:
    """Count the longest run of wins (positive=True) or losses."""
    max_run = 0
    current = 0
    for pnl in pnls:
        if (pnl > 0) == positive:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run
