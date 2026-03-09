"""
Main Bot Controller.

Orchestrates all strategies, filters, trade management, and data
handling for the SE Forex Trading Bot.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

from src.strategies.smc_strategy import SMCStrategy
from src.strategies.ict_strategy import ICTStrategy
from src.strategies.price_action_strategy import PriceActionStrategy
from src.strategies.indicators_strategy import IndicatorsStrategy
from src.filters.trend_filter import TrendFilter
from src.filters.news_filter import NewsFilter
from src.trade_management.position_manager import PositionManager
from src.trade_management.risk_manager import RiskManager
from src.data.market_data import MarketData
from src.data.database import Database
from src.utils.logger import setup_logger, log_trade_event
from src.ai.prediction import PredictionEngine

logger = logging.getLogger("forex_bot")


class ForexBot:
    """
    SE Forex Trading Bot.

    Initialises all components, runs the main trading loop, and
    coordinates strategy consensus, filter validation, trade
    execution, and performance logging.
    """

    CONFIG_DIR = "config"

    def __init__(self) -> None:
        setup_logger("forex_bot")
        logger.info("Initialising SE Forex Trading Bot…")

        self._settings = self._load_json("settings.json")
        self._strategies_cfg = self._load_json("strategies.json")
        self._risk_cfg = self._load_json("risk_management.json")

        # Trading settings
        trading = self._settings.get("trading", {})
        self.pairs: list[str] = trading.get(
            "enabled_pairs", ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "XAUUSD"]
        )
        self.timeframe: str = trading.get("timeframe", "H1")

        # Consensus threshold
        consensus_cfg = self._strategies_cfg.get("consensus", {})
        self.min_agreement: float = consensus_cfg.get("min_agreement", 0.5)

        # Initialise strategies
        strat_cfg = self._strategies_cfg.get("strategies", {})
        self.smc = SMCStrategy(strat_cfg.get("smc", {}))
        self.ict = ICTStrategy(strat_cfg.get("ict", {}))
        self.price_action = PriceActionStrategy(strat_cfg.get("price_action", {}))
        self.indicators = IndicatorsStrategy(strat_cfg.get("indicators", {}))

        self._strategies = {
            "smc": self.smc,
            "ict": self.ict,
            "price_action": self.price_action,
            "indicators": self.indicators,
        }

        # Filters
        filters_cfg = {**self._settings.get("filters", {}), **self._settings.get("data", {})}
        self.trend_filter = TrendFilter(filters_cfg)
        self.news_filter = NewsFilter(filters_cfg)

        # Trade management
        self.risk_manager = RiskManager(self._risk_cfg)
        self.position_manager = PositionManager(self._risk_cfg)

        # Data / persistence
        data_cfg = self._settings.get("data", {})
        self.market_data = MarketData(data_cfg)
        self.db = Database()

        # AI Ensemble
        ai_cfg = self._settings.get("ai", {})
        self._ai_enabled: bool = ai_cfg.get("enabled", False)
        self._ai_engine: Optional[PredictionEngine] = None
        self._ai_ensemble_weight: float = ai_cfg.get("ensemble_weight", 0.4)
        if self._ai_enabled:
            try:
                self._ai_engine = PredictionEngine(self._settings)
                logger.info("AI Ensemble initialised (ensemble_weight=%.2f).", self._ai_ensemble_weight)
            except Exception as exc:
                logger.warning("AI Ensemble could not be initialised: %s", exc)
                self._ai_enabled = False

        self._running = False
        logger.info("Bot initialised. Pairs: %s", ", ".join(self.pairs))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, loop_interval: int = 60) -> None:
        """
        Start the main trading loop.

        Args:
            loop_interval: Seconds between each analysis cycle (default 60).
        """
        self._running = True
        logger.info("Bot started. Loop interval: %ds", loop_interval)

        try:
            while self._running:
                self._reset_daily_stats_if_needed()
                self._run_cycle()
                time.sleep(loop_interval)
        except KeyboardInterrupt:
            logger.info("Bot stopped by user.")
        finally:
            self._running = False
            self._save_performance_snapshot()

    def stop(self) -> None:
        """Signal the bot to stop after the current cycle."""
        self._running = False
        logger.info("Stop signal received.")

    # ------------------------------------------------------------------
    # Trading cycle
    # ------------------------------------------------------------------

    def _run_cycle(self) -> None:
        """Execute one full analysis + execution cycle for all pairs."""
        current_prices: dict[str, float] = {}

        for pair in self.pairs:
            try:
                df = self.market_data.get_candles(pair, self.timeframe)
                if not self.market_data.validate_dataframe(df):
                    logger.warning("Invalid market data for %s – skipping.", pair)
                    continue

                price = float(df["close"].iloc[-1])
                current_prices[pair] = price

                # Run consensus
                signal, confidence = self._consensus(pair, df)
                if signal == "NEUTRAL":
                    continue

                # Apply filters
                if not self._passes_filters(pair, signal, df):
                    logger.debug("%s signal filtered out for %s.", signal, pair)
                    continue

                # Risk gate
                if not self.risk_manager.is_trade_allowed():
                    break

                # Execute trade
                self._execute_trade(pair, signal, price, df)

            except Exception as exc:
                logger.exception("Error processing %s: %s", pair, exc)

        # Update trailing stops and check SL/TP hits
        if current_prices:
            self.position_manager.update_trailing_stops(current_prices)
            closed = self.position_manager.check_stop_take_profit(current_prices)
            for pos in closed:
                self.risk_manager.record_trade_close(pos.pnl)
                self.db.log_trade_close(pos.id, pos.close_price or 0.0, pos.pnl)
                log_trade_event(logger, "CLOSE", {
                    "position_id": pos.id,
                    "pair": pos.pair,
                    "pnl": pos.pnl,
                })

    # ------------------------------------------------------------------
    # Consensus engine
    # ------------------------------------------------------------------

    def _consensus(self, pair: str, df) -> tuple[str, float]:
        """
        Run all enabled strategies and AI ensemble, then aggregate signals.

        The AI ensemble prediction is blended with the traditional strategy
        consensus using the configured ``ensemble_weight``.

        Returns:
            Tuple of (signal, confidence) where signal is 'BUY', 'SELL',
            or 'NEUTRAL'.
        """
        strat_cfg = self._strategies_cfg.get("strategies", {})
        votes: dict[str, float] = {"BUY": 0.0, "SELL": 0.0}

        # Traditional strategies
        trad_weight = 1.0 - self._ai_ensemble_weight if self._ai_enabled else 1.0

        for name, strategy in self._strategies.items():
            cfg = strat_cfg.get(name, {})
            if not cfg.get("enabled", True):
                continue
            weight = cfg.get("weight", 0.25) * trad_weight
            try:
                result = strategy.analyze(df)
            except Exception as exc:
                logger.warning("Strategy %s error for %s: %s", name, pair, exc)
                continue

            sig = result.get("signal", "NEUTRAL")
            conf = result.get("confidence", 0.0)
            if sig in votes:
                votes[sig] += weight * conf

        # AI ensemble contribution
        if self._ai_enabled and self._ai_engine is not None:
            try:
                ai_signal, ai_conf = self._ai_engine.predict(df, pair=pair)
                if ai_signal in votes:
                    votes[ai_signal] += self._ai_ensemble_weight * ai_conf
                    logger.debug("AI signal for %s: %s (%.3f)", pair, ai_signal, ai_conf)
            except Exception as exc:
                logger.warning("AI ensemble error for %s: %s", pair, exc)

        total = sum(votes.values())
        if total == 0:
            return "NEUTRAL", 0.0

        best_signal = max(votes, key=votes.get)
        best_score = votes[best_signal] / total

        if best_score >= self.min_agreement:
            return best_signal, best_score
        return "NEUTRAL", best_score

    # ------------------------------------------------------------------
    # Filter application
    # ------------------------------------------------------------------

    def _passes_filters(self, pair: str, signal: str, df) -> bool:
        """Return True if the signal passes all enabled filters."""
        filters_cfg = self._settings.get("filters", {})

        if filters_cfg.get("trend_filter_enabled", True):
            if not self.trend_filter.is_trade_allowed(df, signal):
                return False

        if filters_cfg.get("news_filter_enabled", True):
            if not self.news_filter.is_trade_allowed(pair):
                return False

        return True

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def _execute_trade(
        self, pair: str, signal: str, entry_price: float, df
    ) -> None:
        """Calculate risk parameters and open a position."""
        # Compute ATR for stop loss sizing
        atr = self._compute_atr(df)

        stop_loss = self.risk_manager.calculate_stop_loss(entry_price, signal, atr)
        take_profit = self.risk_manager.calculate_take_profit(
            entry_price, stop_loss, signal
        )
        lot_size = self.risk_manager.calculate_position_size(entry_price, stop_loss)

        pos = self.position_manager.open_position(
            pair=pair,
            direction=signal,
            entry_price=entry_price,
            lot_size=lot_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        self.risk_manager.record_trade_open()
        self.db.log_trade_open(
            position_id=pos.id,
            pair=pair,
            direction=signal,
            entry_price=entry_price,
            lot_size=lot_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        log_trade_event(logger, "OPEN", {
            "position_id": pos.id,
            "pair": pair,
            "direction": signal,
            "entry": entry_price,
            "sl": stop_loss,
            "tp": take_profit,
            "lots": lot_size,
        })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_atr(self, df, period: int = 14) -> Optional[float]:
        """Compute the latest ATR value from the DataFrame."""
        try:
            import numpy as np
            import pandas as pd

            high_low = df["high"] - df["low"]
            high_close = (df["high"] - df["close"].shift()).abs()
            low_close = (df["low"] - df["close"].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(period).mean().iloc[-1]
            return float(atr) if not np.isnan(atr) else None
        except Exception:
            return None

    def _reset_daily_stats_if_needed(self) -> None:
        """Reset daily risk stats at the start of each new UTC day."""
        now = datetime.now(timezone.utc)
        if not hasattr(self, "_last_reset_day") or self._last_reset_day != now.date():
            self.risk_manager.reset_daily_stats()
            self._last_reset_day = now.date()

    def _save_performance_snapshot(self) -> None:
        """Persist a performance snapshot to the database."""
        closed = self.position_manager.closed_positions
        if not closed:
            return

        total = len(closed)
        wins = sum(1 for p in closed if p.pnl > 0)
        losses = total - wins
        total_pnl = sum(p.pnl for p in closed)
        win_rate = (wins / total * 100) if total > 0 else 0.0

        # Simple max drawdown calculation
        running = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in closed:
            running += p.pnl
            if running > peak:
                peak = running
            dd = peak - running
            if dd > max_dd:
                max_dd = dd

        self.db.save_performance_snapshot(total, wins, losses, total_pnl, win_rate, max_dd)
        logger.info(
            "Performance snapshot saved: trades=%d win_rate=%.1f%% pnl=%.2f",
            total, win_rate, total_pnl,
        )

    def _load_json(self, filename: str) -> dict:
        path = os.path.join(self.CONFIG_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError:
            logger.warning("Config file not found: %s – using defaults.", path)
            return {}
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON in %s: %s – using defaults.", path, exc)
            return {}
