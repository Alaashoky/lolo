"""
Prediction Engine – real-time and batch predictions.

Provides a clean interface between the AI ensemble and the trading bot.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from src.ai.config import DEFAULT_CONFIG
from src.ai.ensemble import AIEnsemble
from src.ai.meta_learner import MetaLearner

logger = logging.getLogger("forex_bot.ai.prediction")


class PredictionEngine:
    """
    Wraps the AIEnsemble + MetaLearner and exposes a simple
    ``predict`` method compatible with the bot's consensus interface.

    Usage::

        engine = PredictionEngine(config)
        signal, confidence = engine.predict(df, pair="EURUSD")
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or DEFAULT_CONFIG
        self._cfg = cfg.get("ai", DEFAULT_CONFIG["ai"])
        self._ensemble = AIEnsemble(cfg)
        self._meta_learner = MetaLearner(cfg)
        self._confidence_threshold: float = self._cfg.get("confidence_threshold", 0.55)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        df: pd.DataFrame,
        pair: str = "UNKNOWN",
        headlines: Optional[list[str]] = None,
    ) -> tuple[str, float]:
        """
        Generate a trading signal for a single currency pair.

        Args:
            df: OHLCV DataFrame (most recent bars last).
            pair: Pair identifier for logging.
            headlines: Optional news headlines for the sentiment model.

        Returns:
            Tuple of (signal, confidence) where signal ∈ {'BUY', 'SELL', 'NEUTRAL'}.
        """
        try:
            # Ensemble vote
            ens_signal, ens_conf, details = self._ensemble.predict(df, headlines=headlines)

            # Meta-learner refinement
            final_signal, final_conf = self._meta_learner.decide(
                ens_signal, ens_conf, details, self._confidence_threshold
            )

            logger.debug(
                "AI prediction for %s: ensemble=%s(%.3f) → final=%s(%.3f)",
                pair, ens_signal, ens_conf, final_signal, final_conf,
            )
            return final_signal, final_conf

        except Exception as exc:
            logger.warning("PredictionEngine.predict error for %s: %s", pair, exc)
            return "NEUTRAL", 0.0

    def batch_predict(
        self, df: pd.DataFrame, window_size: int = 100
    ) -> list[tuple[str, float]]:
        """
        Run sliding-window predictions across the entire DataFrame.

        Args:
            df: Full OHLCV DataFrame.
            window_size: Number of bars per prediction window.

        Returns:
            List of (signal, confidence) tuples aligned with df rows
            from index window_size onward.
        """
        results = []
        for i in range(window_size, len(df)):
            window = df.iloc[i - window_size + 1 : i + 1]
            signal, conf = self.predict(window)
            results.append((signal, conf))
        return results

    # ------------------------------------------------------------------
    # Property access
    # ------------------------------------------------------------------

    @property
    def ensemble(self) -> AIEnsemble:
        return self._ensemble

    @property
    def meta_learner(self) -> MetaLearner:
        return self._meta_learner

    def is_ready(self) -> bool:
        """Return True if at least one model is trained."""
        return self._ensemble.is_any_model_trained()
