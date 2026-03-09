"""
Trade Database.

SQLite-backed storage for trades, performance metrics, and
historical market data using SQLAlchemy.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger("forex_bot.database")


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


class TradeRecord(Base):
    """Persisted representation of a completed trade."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    position_id = Column(String(16), nullable=False, index=True)
    pair = Column(String(16), nullable=False)
    direction = Column(String(4), nullable=False)
    entry_price = Column(Float, nullable=False)
    close_price = Column(Float)
    lot_size = Column(Float, nullable=False)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    pnl = Column(Float, default=0.0)
    open_time = Column(DateTime(timezone=True))
    close_time = Column(DateTime(timezone=True))
    strategy = Column(String(64))
    notes = Column(String(256))


class PerformanceSnapshot(Base):
    """Daily performance snapshot."""

    __tablename__ = "performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_date = Column(DateTime(timezone=True), nullable=False)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)


class MarketDataRecord(Base):
    """Cached OHLCV candle data."""

    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pair = Column(String(16), nullable=False, index=True)
    timeframe = Column(String(8), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)


# ---------------------------------------------------------------------------
# Database manager
# ---------------------------------------------------------------------------


class Database:
    """
    SQLite database manager for the trading bot.

    Provides methods for logging trades, storing performance snapshots,
    and caching market data.
    """

    def __init__(self, db_path: str = "data/forex_bot.db") -> None:
        """
        Args:
            db_path: File path for the SQLite database.
        """
        import os
        dir_path = os.path.dirname(db_path) or "."
        os.makedirs(dir_path, exist_ok=True)
        connection_string = f"sqlite:///{db_path}"
        self._engine = create_engine(connection_string, echo=False)
        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)
        logger.info("Database initialised at %s", db_path)

    # ------------------------------------------------------------------
    # Trade logging
    # ------------------------------------------------------------------

    def log_trade_open(
        self,
        position_id: str,
        pair: str,
        direction: str,
        entry_price: float,
        lot_size: float,
        stop_loss: float,
        take_profit: float,
        strategy: str = "",
    ) -> None:
        """Insert an open trade record."""
        with Session(self._engine) as session:
            record = TradeRecord(
                position_id=position_id,
                pair=pair,
                direction=direction,
                entry_price=entry_price,
                lot_size=lot_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                open_time=datetime.now(timezone.utc),
                strategy=strategy,
            )
            session.add(record)
            session.commit()
            logger.debug("Trade open logged: %s %s", direction, pair)

    def log_trade_close(
        self,
        position_id: str,
        close_price: float,
        pnl: float,
        notes: str = "",
    ) -> None:
        """Update an existing trade record with close details."""
        with Session(self._engine) as session:
            record = (
                session.query(TradeRecord)
                .filter_by(position_id=position_id)
                .order_by(TradeRecord.id.desc())
                .first()
            )
            if record:
                record.close_price = close_price
                record.pnl = pnl
                record.close_time = datetime.now(timezone.utc)
                record.notes = notes
                session.commit()
                logger.debug("Trade close logged: %s pnl=%.2f", position_id, pnl)
            else:
                logger.warning("Trade record not found for position_id=%s", position_id)

    # ------------------------------------------------------------------
    # Performance tracking
    # ------------------------------------------------------------------

    def save_performance_snapshot(
        self,
        total_trades: int,
        winning_trades: int,
        losing_trades: int,
        total_pnl: float,
        win_rate: float,
        max_drawdown: float,
    ) -> None:
        """Save a daily performance snapshot."""
        with Session(self._engine) as session:
            snap = PerformanceSnapshot(
                snapshot_date=datetime.now(timezone.utc),
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                total_pnl=total_pnl,
                win_rate=win_rate,
                max_drawdown=max_drawdown,
            )
            session.add(snap)
            session.commit()

    def get_trade_history(self, limit: int = 100) -> list[dict]:
        """
        Return recent closed trades as a list of dicts.

        Args:
            limit: Maximum number of records to return.
        """
        with Session(self._engine) as session:
            records = (
                session.query(TradeRecord)
                .filter(TradeRecord.close_time.isnot(None))
                .order_by(TradeRecord.close_time.desc())
                .limit(limit)
                .all()
            )
            return [self._trade_to_dict(r) for r in records]

    def get_performance_history(self, limit: int = 30) -> list[dict]:
        """Return recent performance snapshots."""
        with Session(self._engine) as session:
            records = (
                session.query(PerformanceSnapshot)
                .order_by(PerformanceSnapshot.snapshot_date.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "date": r.snapshot_date,
                    "total_trades": r.total_trades,
                    "winning_trades": r.winning_trades,
                    "losing_trades": r.losing_trades,
                    "total_pnl": r.total_pnl,
                    "win_rate": r.win_rate,
                    "max_drawdown": r.max_drawdown,
                }
                for r in records
            ]

    # ------------------------------------------------------------------
    # Market data caching
    # ------------------------------------------------------------------

    def cache_candles(self, pair: str, timeframe: str, df) -> None:
        """Store OHLCV candles to the database for later retrieval."""
        import pandas as pd

        with Session(self._engine) as session:
            for ts, row in df.iterrows():
                record = MarketDataRecord(
                    pair=pair,
                    timeframe=timeframe,
                    timestamp=ts if hasattr(ts, "tzinfo") else pd.Timestamp(ts, tz="UTC"),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume", 0)),
                )
                session.merge(record)
            session.commit()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _trade_to_dict(record: TradeRecord) -> dict:
        return {
            "position_id": record.position_id,
            "pair": record.pair,
            "direction": record.direction,
            "entry_price": record.entry_price,
            "close_price": record.close_price,
            "lot_size": record.lot_size,
            "stop_loss": record.stop_loss,
            "take_profit": record.take_profit,
            "pnl": record.pnl,
            "open_time": record.open_time,
            "close_time": record.close_time,
            "strategy": record.strategy,
        }
