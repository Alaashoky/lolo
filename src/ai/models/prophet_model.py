"""
Prophet Model for time-series forecasting.

Uses Facebook's Prophet for additive decomposition with:
- Trend + seasonality + holiday effects
- Uncertainty intervals
- Changepoint detection
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("forex_bot.ai.prophet")

try:
    from prophet import Prophet  # type: ignore

    _PROPHET_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PROPHET_AVAILABLE = False
    logger.warning("prophet not installed – ProphetModel will run in fallback mode.")


class ProphetModel:
    """
    Prophet-based directional price forecasting model.

    Generates a BUY / HOLD / SELL signal by comparing the predicted
    future price against the current price.

    Falls back to neutral predictions when Prophet is unavailable.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or {}
        self.changepoint_prior_scale: float = cfg.get("changepoint_prior_scale", 0.05)
        self.seasonality_prior_scale: float = cfg.get("seasonality_prior_scale", 10.0)
        self.forecast_horizon: int = cfg.get("forecast_horizon", 5)
        self._model: Optional[object] = None
        self._is_trained: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame) -> dict:
        """
        Fit the Prophet model on a time-series DataFrame.

        Args:
            df: DataFrame with 'ds' (datetime) and 'y' (close price) columns,
                or a generic OHLCV DataFrame whose index is a DatetimeIndex.

        Returns:
            Empty dict (Prophet does not expose training metrics).
        """
        if not _PROPHET_AVAILABLE:
            logger.warning("ProphetModel.train: Prophet unavailable – skipping.")
            return {}

        prophet_df = self._prepare_dataframe(df)
        if prophet_df is None or len(prophet_df) < 10:
            logger.warning("ProphetModel.train: insufficient data.")
            return {}

        self._model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            daily_seasonality=False,
            weekly_seasonality=True,
        )
        self._model.fit(prophet_df)
        self._is_trained = True
        logger.info("ProphetModel training complete. n_samples=%d", len(prophet_df))
        return {}

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate directional forecasts.

        Returns:
            Tuple of (predicted_classes, probabilities) where the last
            element corresponds to the most recent bar.
        """
        n = len(df)
        neutral_probs = np.full((n, 3), 1 / 3, dtype=np.float32)
        neutral_classes = np.ones(n, dtype=int)

        if not _PROPHET_AVAILABLE or not self._is_trained or self._model is None:
            return neutral_classes, neutral_probs

        try:
            future = self._model.make_future_dataframe(periods=self.forecast_horizon, freq="h")
            forecast = self._model.predict(future)

            current_price = df["close"].iloc[-1]
            future_price = forecast["yhat"].iloc[-1]

            pct_change = (future_price - current_price) / (current_price + 1e-10)
            signal = self._pct_to_signal(pct_change)
            probs = self._signal_to_probs(signal, abs(pct_change))

            # Broadcast single prediction across all rows
            probs_arr = np.tile(probs, (n, 1)).astype(np.float32)
            classes_arr = np.full(n, signal, dtype=int)
            return classes_arr, probs_arr
        except Exception as exc:
            logger.warning("ProphetModel.predict error: %s", exc)
            return neutral_classes, neutral_probs

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path + ".pkl", "wb") as fh:
            pickle.dump({"model": self._model, "is_trained": self._is_trained}, fh)

    def load(self, path: str) -> None:
        pkl_path = path + ".pkl"
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as fh:
                data = pickle.load(fh)
            self._model = data.get("model")
            self._is_trained = data.get("is_trained", False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_dataframe(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Convert OHLCV or generic DataFrame to Prophet's ds/y format."""
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        if "ds" in df.columns and "y" in df.columns:
            return df[["ds", "y"]].dropna()

        if "close" not in df.columns:
            return None

        if isinstance(df.index, pd.DatetimeIndex):
            ds = df.index
        else:
            ds = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq="h")

        return pd.DataFrame({"ds": ds, "y": df["close"].values}).dropna()

    @staticmethod
    def _pct_to_signal(pct_change: float) -> int:
        """Convert percentage price change to signal label."""
        if pct_change > 0.001:
            return 2  # BUY
        if pct_change < -0.001:
            return 0  # SELL
        return 1  # HOLD

    @staticmethod
    def _signal_to_probs(signal: int, magnitude: float) -> list[float]:
        """Build a soft probability vector from a signal and its magnitude."""
        confidence = min(0.9, 0.5 + magnitude * 100)
        base = (1 - confidence) / 2
        probs = [base, base, base]
        probs[signal] = confidence
        return probs
