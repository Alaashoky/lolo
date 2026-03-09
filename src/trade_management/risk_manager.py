"""
Risk Manager.

Calculates position sizes, stop loss / take profit levels,
and monitors account drawdown.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("forex_bot.risk_manager")


class RiskManager:
    """
    Handles all risk-related calculations for the trading bot.

    Supports:
    - Position sizing based on fixed % risk per trade
    - Stop loss and take profit calculation
    - Daily drawdown monitoring
    - Maximum concurrent trade enforcement
    """

    def __init__(self, config: dict) -> None:
        """
        Args:
            config: risk_management.json content.
        """
        account = config.get("account", {})
        risk = config.get("risk_management", {})
        sl_cfg = config.get("stop_loss", {})

        self.account_balance: float = account.get("balance", 10_000.0)
        self.leverage: int = account.get("leverage", 1)

        self.risk_per_trade: float = risk.get("risk_per_trade", 2.0)
        self.max_daily_loss: float = risk.get("max_daily_loss", 50.0)
        self.max_concurrent_trades: int = risk.get("max_concurrent_trades", 3)

        self.use_trailing_stop: bool = sl_cfg.get("use_trailing_stop", True)
        self.trailing_stop_pips: float = sl_cfg.get("trailing_stop_pips", 10.0)

        self._daily_loss: float = 0.0
        self._active_trade_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        pip_value: float = 10.0,
    ) -> float:
        """
        Calculate lot size using the fixed-risk model.

        Args:
            entry_price: Proposed entry price.
            stop_loss_price: Proposed stop loss price.
            pip_value: Value of one pip in account currency (default $10 for
                       standard lot on major pairs).

        Returns:
            Position size in lots (rounded to 2 decimal places).
        """
        risk_amount = self.account_balance * (self.risk_per_trade / 100.0)
        stop_distance = abs(entry_price - stop_loss_price)

        if stop_distance == 0:
            logger.warning("Stop distance is zero – returning 0.01 lot minimum.")
            return 0.01

        # Convert stop distance to pips (assumes 4-decimal pair: 1 pip = 0.0001)
        pips = stop_distance / 0.0001
        lot_size = risk_amount / (pips * pip_value)
        lot_size = max(0.01, round(lot_size, 2))

        logger.debug(
            "Position size calculated",
            extra={
                "risk_amount": risk_amount,
                "stop_pips": round(pips, 1),
                "lot_size": lot_size,
            },
        )
        return lot_size

    def calculate_stop_loss(
        self,
        entry_price: float,
        signal: str,
        atr: Optional[float] = None,
        multiplier: float = 1.5,
    ) -> float:
        """
        Calculate stop loss price.

        Uses ATR-based stop when ATR is provided, otherwise falls back
        to trailing_stop_pips.

        Args:
            entry_price: Trade entry price.
            signal: 'BUY' or 'SELL'.
            atr: Current ATR value (optional).
            multiplier: ATR multiplier for stop distance.

        Returns:
            Stop loss price.
        """
        if atr and atr > 0:
            stop_distance = atr * multiplier
        else:
            stop_distance = self.trailing_stop_pips * 0.0001

        if signal == "BUY":
            return round(entry_price - stop_distance, 5)
        return round(entry_price + stop_distance, 5)

    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss_price: float,
        signal: str,
        rr_ratio: float = 2.0,
    ) -> float:
        """
        Calculate take profit using a risk/reward ratio.

        Args:
            entry_price: Trade entry price.
            stop_loss_price: Stop loss price.
            signal: 'BUY' or 'SELL'.
            rr_ratio: Desired risk-to-reward ratio.

        Returns:
            Take profit price.
        """
        risk = abs(entry_price - stop_loss_price)
        reward = risk * rr_ratio

        if signal == "BUY":
            return round(entry_price + reward, 5)
        return round(entry_price - reward, 5)

    def is_trade_allowed(self) -> bool:
        """
        Check whether opening a new trade is allowed given current
        drawdown and concurrent trade limits.

        Returns:
            True if a new trade can be opened.
        """
        if self._active_trade_count >= self.max_concurrent_trades:
            logger.info("Max concurrent trades reached (%d).", self.max_concurrent_trades)
            return False

        daily_loss_pct = (self._daily_loss / self.account_balance) * 100.0
        if daily_loss_pct >= self.max_daily_loss:
            logger.warning(
                "Daily loss limit reached: %.2f%% (max %.2f%%).",
                daily_loss_pct,
                self.max_daily_loss,
            )
            return False

        return True

    def record_trade_open(self) -> None:
        """Increment the active trade counter."""
        self._active_trade_count += 1

    def record_trade_close(self, pnl: float) -> None:
        """
        Decrement the active trade counter and update daily P&L.

        Args:
            pnl: Realised P&L of the closed trade (negative = loss).
        """
        self._active_trade_count = max(0, self._active_trade_count - 1)
        if pnl < 0:
            self._daily_loss += abs(pnl)

    def reset_daily_stats(self) -> None:
        """Reset daily loss tracking (call at start of each trading day)."""
        self._daily_loss = 0.0
        logger.info("Daily risk stats reset.")

    def update_balance(self, new_balance: float) -> None:
        """Update the account balance (e.g., after receiving broker data)."""
        self.account_balance = new_balance
