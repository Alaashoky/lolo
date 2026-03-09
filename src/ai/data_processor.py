"""
Data Processor for the AI Ensemble System.

Handles loading, normalisation, missing-value imputation, and
train/validation/test splitting of market data.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # type: ignore

logger = logging.getLogger("forex_bot.ai.data_processor")


class DataProcessor:
    """
    Prepares raw OHLCV data for model training and inference.

    Responsibilities:
    - Missing value imputation
    - Outlier removal
    - Feature scaling (MinMax or Standard)
    - Train / validation / test splitting
    - Sequence creation for time-series models
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or {}
        training = cfg.get("training", {})
        self.train_ratio: float = training.get("train_ratio", 0.80)
        self.val_ratio: float = training.get("val_ratio", 0.10)
        self.test_ratio: float = training.get("test_ratio", 0.10)
        self.min_samples: int = training.get("min_samples", 500)

        self._price_scaler: Optional[MinMaxScaler] = None
        self._feature_scaler: Optional[StandardScaler] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare a raw OHLCV DataFrame.

        Args:
            df: Raw DataFrame with at least open/high/low/close columns.

        Returns:
            Cleaned DataFrame with normalised values.
        """
        df = df.copy()
        df = self._ensure_columns(df)
        df = self._handle_missing_values(df)
        df = self._remove_outliers(df)
        return df

    def split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets (temporal split).

        Args:
            df: Processed DataFrame.

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        n = len(df)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        logger.debug(
            "Split sizes – train: %d, val: %d, test: %d", len(train_df), len(val_df), len(test_df)
        )
        return train_df, val_df, test_df

    def scale_prices(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        MinMax-scale OHLC price columns in-place.

        Args:
            df: DataFrame with price columns.
            fit: If True, fit a new scaler; otherwise use existing one.

        Returns:
            DataFrame with scaled price columns.
        """
        price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
        if not price_cols:
            return df

        df = df.copy()
        if fit or self._price_scaler is None:
            self._price_scaler = MinMaxScaler(feature_range=(0, 1))
            df[price_cols] = self._price_scaler.fit_transform(df[price_cols])
        else:
            df[price_cols] = self._price_scaler.transform(df[price_cols])
        return df

    def scale_features(
        self, df: pd.DataFrame, feature_cols: list[str], fit: bool = True
    ) -> pd.DataFrame:
        """
        Standard-scale feature columns in-place.

        Args:
            df: DataFrame with feature columns.
            feature_cols: List of column names to scale.
            fit: If True, fit a new scaler; otherwise use existing one.

        Returns:
            DataFrame with scaled feature columns.
        """
        cols = [c for c in feature_cols if c in df.columns]
        if not cols:
            return df

        df = df.copy()
        if fit or self._feature_scaler is None:
            self._feature_scaler = StandardScaler()
            df[cols] = self._feature_scaler.fit_transform(df[cols])
        else:
            df[cols] = self._feature_scaler.transform(df[cols])
        return df

    def create_sequences(
        self, data: np.ndarray, lookback: int = 60
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create overlapping input/target sequences for time-series models.

        Args:
            data: 2-D array of shape (n_samples, n_features). The last
                  column is treated as the target.
            lookback: Number of time steps in each input window.

        Returns:
            Tuple of (X, y) with shapes (n, lookback, n_features-1) and (n,).
        """
        X, y = [], []
        features = data[:, :-1]
        target = data[:, -1]
        for i in range(lookback, len(data)):
            X.append(features[i - lookback : i])
            y.append(target[i])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def inverse_scale_prices(self, scaled: np.ndarray) -> np.ndarray:
        """Invert price scaling for interpretable output."""
        if self._price_scaler is None:
            return scaled
        # Reshape for inverse transform (scaler expects 2-D)
        reshaped = scaled.reshape(-1, 1)
        dummy = np.zeros((len(reshaped), self._price_scaler.n_features_in_))
        dummy[:, 0] = reshaped[:, 0]
        inverted = self._price_scaler.inverse_transform(dummy)
        return inverted[:, 0]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Normalise column names to lower-case."""
        df.columns = [c.lower() for c in df.columns]
        return df

    @staticmethod
    def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill then back-fill missing values."""
        df = df.ffill().bfill()
        df = df.dropna()
        return df

    @staticmethod
    def _remove_outliers(df: pd.DataFrame, zscore_threshold: float = 5.0) -> pd.DataFrame:
        """
        Remove rows where any numeric column is more than
        `zscore_threshold` standard deviations from the mean.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if numeric_cols.empty:
            return df

        zscores = (df[numeric_cols] - df[numeric_cols].mean()) / (
            df[numeric_cols].std() + 1e-10
        )
        mask = (zscores.abs() <= zscore_threshold).all(axis=1)
        removed = (~mask).sum()
        if removed:
            logger.debug("Removed %d outlier rows.", removed)
        return df[mask].reset_index(drop=True)
