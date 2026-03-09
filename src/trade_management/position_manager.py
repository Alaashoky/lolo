"""
Position Manager.

Opens, closes, and modifies positions; handles trailing stops
and partial profit-taking logic.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("forex_bot.position_manager")


class Position:
    """Represents a single open trade position."""

    def __init__(
        self,
        pair: str,
        direction: str,
        entry_price: float,
        lot_size: float,
        stop_loss: float,
        take_profit: float,
    ) -> None:
        self.id: str = str(uuid.uuid4())[:8]
        self.pair = pair
        self.direction = direction  # 'BUY' or 'SELL'
        self.entry_price = entry_price
        self.lot_size = lot_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.open_time: datetime = datetime.now(timezone.utc)
        self.close_price: Optional[float] = None
        self.close_time: Optional[datetime] = None
        self.pnl: float = 0.0
        self.is_open: bool = True
        self.partial_closed: bool = False

    def unrealised_pnl(self, current_price: float, pip_value: float = 10.0) -> float:
        """Calculate unrealised P&L in account currency."""
        if self.direction == "BUY":
            pips = (current_price - self.entry_price) / 0.0001
        else:
            pips = (self.entry_price - current_price) / 0.0001
        return round(pips * pip_value * self.lot_size, 2)

    def __repr__(self) -> str:
        return (
            f"<Position {self.id} {self.pair} {self.direction} "
            f"@ {self.entry_price} SL={self.stop_loss} TP={self.take_profit}>"
        )


class PositionManager:
    """
    Manages the lifecycle of trading positions.

    Features:
    - Open and close positions
    - Trailing stop updates
    - Partial profit taking at configurable R multiples
    - Aggregate P&L tracking
    """

    def __init__(self, risk_config: dict) -> None:
        """
        Args:
            risk_config: risk_management.json content.
        """
        sl_cfg = risk_config.get("stop_loss", {})
        self.use_trailing_stop: bool = sl_cfg.get("use_trailing_stop", True)
        self.trailing_stop_pips: float = sl_cfg.get("trailing_stop_pips", 10.0)
        self.trailing_step: float = self.trailing_stop_pips * 0.0001

        self.positions: dict[str, Position] = {}
        self.closed_positions: list[Position] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open_position(
        self,
        pair: str,
        direction: str,
        entry_price: float,
        lot_size: float,
        stop_loss: float,
        take_profit: float,
    ) -> Position:
        """
        Create and register a new position.

        Args:
            pair: Currency pair (e.g., 'EURUSD').
            direction: 'BUY' or 'SELL'.
            entry_price: Fill price.
            lot_size: Position size in lots.
            stop_loss: Initial stop loss price.
            take_profit: Take profit price.

        Returns:
            The newly created Position object.
        """
        pos = Position(pair, direction, entry_price, lot_size, stop_loss, take_profit)
        self.positions[pos.id] = pos
        logger.info("Position opened", extra={"position": str(pos)})
        return pos

    def close_position(self, position_id: str, close_price: float) -> Optional[Position]:
        """
        Close an open position and move it to the closed list.

        Args:
            position_id: Unique position ID.
            close_price: Closing fill price.

        Returns:
            The closed Position, or None if not found.
        """
        pos = self.positions.pop(position_id, None)
        if pos is None:
            logger.warning("Position %s not found.", position_id)
            return None

        pos.close_price = close_price
        pos.close_time = datetime.now(timezone.utc)
        pos.is_open = False

        if pos.direction == "BUY":
            pips = (close_price - pos.entry_price) / 0.0001
        else:
            pips = (pos.entry_price - close_price) / 0.0001

        pos.pnl = round(pips * 10.0 * pos.lot_size, 2)
        self.closed_positions.append(pos)

        logger.info(
            "Position closed",
            extra={"position": str(pos), "pnl": pos.pnl},
        )
        return pos

    def update_trailing_stops(self, current_prices: dict[str, float]) -> None:
        """
        Move stop losses forward for profitable positions.

        Args:
            current_prices: Dict mapping pair symbols to current bid prices.
        """
        if not self.use_trailing_stop:
            return

        for pos in list(self.positions.values()):
            price = current_prices.get(pos.pair)
            if price is None:
                continue

            if pos.direction == "BUY":
                new_sl = price - self.trailing_step
                if new_sl > pos.stop_loss:
                    pos.stop_loss = round(new_sl, 5)
                    logger.debug("Trailing stop updated for %s: %.5f", pos.id, pos.stop_loss)

            elif pos.direction == "SELL":
                new_sl = price + self.trailing_step
                if new_sl < pos.stop_loss:
                    pos.stop_loss = round(new_sl, 5)
                    logger.debug("Trailing stop updated for %s: %.5f", pos.id, pos.stop_loss)

    def take_partial_profit(
        self,
        position_id: str,
        current_price: float,
        partial_ratio: float = 0.5,
        trigger_r: float = 1.0,
    ) -> bool:
        """
        Close a portion of a position when it reaches *trigger_r* multiples
        of the initial risk.

        Args:
            position_id: Position to check.
            current_price: Current market price.
            partial_ratio: Fraction of position to close (default 0.5 = 50%).
            trigger_r: R multiple at which partial close is triggered.

        Returns:
            True if partial close was executed.
        """
        pos = self.positions.get(position_id)
        if pos is None or pos.partial_closed:
            return False

        initial_risk = abs(pos.entry_price - pos.stop_loss)
        if initial_risk == 0:
            return False

        if pos.direction == "BUY":
            profit_r = (current_price - pos.entry_price) / initial_risk
        else:
            profit_r = (pos.entry_price - current_price) / initial_risk

        if profit_r >= trigger_r:
            pos.lot_size = round(pos.lot_size * (1 - partial_ratio), 2)
            pos.partial_closed = True
            logger.info(
                "Partial profit taken on %s at %.5f (R=%.2f)",
                pos.id,
                current_price,
                profit_r,
            )
            return True

        return False

    def check_stop_take_profit(self, current_prices: dict[str, float]) -> list[Position]:
        """
        Check all open positions for SL/TP hits.

        Args:
            current_prices: Dict mapping pair → current price.

        Returns:
            List of positions that were closed.
        """
        closed = []
        for pos_id, pos in list(self.positions.items()):
            price = current_prices.get(pos.pair)
            if price is None:
                continue

            hit_sl = (pos.direction == "BUY" and price <= pos.stop_loss) or (
                pos.direction == "SELL" and price >= pos.stop_loss
            )
            hit_tp = (pos.direction == "BUY" and price >= pos.take_profit) or (
                pos.direction == "SELL" and price <= pos.take_profit
            )

            if hit_sl or hit_tp:
                close_price = pos.stop_loss if hit_sl else pos.take_profit
                closed_pos = self.close_position(pos_id, close_price)
                if closed_pos:
                    closed.append(closed_pos)

        return closed

    @property
    def open_count(self) -> int:
        """Number of currently open positions."""
        return len(self.positions)

    def get_all_positions(self) -> list[Position]:
        """Return a list of all open Position objects."""
        return list(self.positions.values())

    def total_realised_pnl(self) -> float:
        """Sum of P&L across all closed positions."""
        return round(sum(p.pnl for p in self.closed_positions), 2)
