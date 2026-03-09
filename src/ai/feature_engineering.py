"""
Feature Engineering for the AI Ensemble System.

Extracts technical indicators and derived features from OHLCV data.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("forex_bot.ai.feature_engineering")


class FeatureEngineer:
    """
    Derives model-ready features from raw OHLCV data.

    Feature groups:
    - Technical indicators: RSI, MACD, Bollinger Bands, ATR, Stochastic
    - Price pattern features: candle body, wicks, highs/lows
    - Volume features: volume MA, relative volume
    - Time-based features: hour, day-of-week, month
    - Volatility features: daily range, ATR ratio
    - Momentum features: ROC, momentum
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("feature_engineering", {})
        self.rsi_period: int = cfg.get("rsi_period", 14)
        self.macd_fast: int = cfg.get("macd_fast", 12)
        self.macd_slow: int = cfg.get("macd_slow", 26)
        self.macd_signal: int = cfg.get("macd_signal", 9)
        self.bb_period: int = cfg.get("bb_period", 20)
        self.bb_std: float = cfg.get("bb_std", 2.0)
        self.atr_period: int = cfg.get("atr_period", 14)
        self.volume_ma_period: int = cfg.get("volume_ma_period", 20)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute and append all feature columns to a copy of *df*.

        Args:
            df: DataFrame with columns open, high, low, close (and
                optionally volume).  Index may be datetime or integer.

        Returns:
            DataFrame with original columns plus derived features.
            Rows with NaN values (warm-up period) are dropped.
        """
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        df = self._add_technical_indicators(df)
        df = self._add_price_pattern_features(df)
        df = self._add_volume_features(df)
        df = self._add_time_features(df)
        df = self._add_volatility_features(df)
        df = self._add_momentum_features(df)

        before = len(df)
        df = df.dropna().reset_index(drop=True)
        logger.debug("Feature extraction: %d → %d rows (dropped %d NaN rows).", before, len(df), before - len(df))
        return df

    def feature_names(self) -> list[str]:
        """Return the list of engineered feature column names."""
        return [
            # Technical indicators
            "rsi",
            "macd",
            "macd_signal_line",
            "macd_hist",
            "bb_upper",
            "bb_lower",
            "bb_width",
            "bb_pct_b",
            "atr",
            "stoch_k",
            "stoch_d",
            "ema_20",
            "ema_50",
            "sma_200",
            # Price pattern
            "candle_body",
            "upper_wick",
            "lower_wick",
            "body_ratio",
            "high_20",
            "low_20",
            "pct_from_high",
            "pct_from_low",
            # Volume
            "volume_ma",
            "relative_volume",
            # Time
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            # Volatility
            "daily_range",
            "atr_ratio",
            "close_std_20",
            # Momentum
            "roc_5",
            "roc_10",
            "momentum_10",
        ]

    # ------------------------------------------------------------------
    # Feature groups
    # ------------------------------------------------------------------

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI, MACD, Bollinger Bands, ATR, Stochastic, EMAs."""
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # RSI
        df["rsi"] = self._rsi(close, self.rsi_period)

        # MACD
        ema_fast = close.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.macd_slow, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal_line"] = df["macd"].ewm(span=self.macd_signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal_line"]

        # Bollinger Bands
        sma_bb = close.rolling(self.bb_period).mean()
        std_bb = close.rolling(self.bb_period).std()
        df["bb_upper"] = sma_bb + self.bb_std * std_bb
        df["bb_lower"] = sma_bb - self.bb_std * std_bb
        df["bb_width"] = df["bb_upper"] - df["bb_lower"]
        df["bb_pct_b"] = (close - df["bb_lower"]) / (df["bb_width"] + 1e-10)

        # ATR
        df["atr"] = self._atr(high, low, close, self.atr_period)

        # Stochastic Oscillator
        low_n = low.rolling(14).min()
        high_n = high.rolling(14).max()
        df["stoch_k"] = 100 * (close - low_n) / (high_n - low_n + 1e-10)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        # Moving averages
        df["ema_20"] = close.ewm(span=20, adjust=False).mean()
        df["ema_50"] = close.ewm(span=50, adjust=False).mean()
        df["sma_200"] = close.rolling(200).mean()

        return df

    def _add_price_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Candle body, wicks, recent high/low distances."""
        df["candle_body"] = (df["close"] - df["open"]).abs()
        df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
        candle_range = df["high"] - df["low"]
        df["body_ratio"] = df["candle_body"] / (candle_range + 1e-10)

        df["high_20"] = df["high"].rolling(20).max()
        df["low_20"] = df["low"].rolling(20).min()
        df["pct_from_high"] = (df["close"] - df["high_20"]) / (df["high_20"] + 1e-10)
        df["pct_from_low"] = (df["close"] - df["low_20"]) / (df["low_20"] + 1e-10)
        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume moving average and relative volume."""
        if "volume" in df.columns:
            vol = df["volume"].clip(lower=0)
            df["volume_ma"] = vol.rolling(self.volume_ma_period).mean()
            df["relative_volume"] = vol / (df["volume_ma"] + 1e-10)
        else:
            df["volume_ma"] = 0.0
            df["relative_volume"] = 1.0
        return df

    @staticmethod
    def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """Cyclically-encoded hour and day-of-week."""
        if isinstance(df.index, pd.DatetimeIndex):
            hour = df.index.hour
            dow = df.index.dayofweek
        else:
            hour = pd.Series(np.zeros(len(df)))
            dow = pd.Series(np.zeros(len(df)))

        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Daily range, ATR ratio, and rolling standard deviation."""
        df["daily_range"] = df["high"] - df["low"]
        df["atr_ratio"] = df["atr"] / (df["close"] + 1e-10)
        df["close_std_20"] = df["close"].rolling(20).std()
        return df

    @staticmethod
    def _add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
        """Rate of change and momentum."""
        df["roc_5"] = df["close"].pct_change(5)
        df["roc_10"] = df["close"].pct_change(10)
        df["momentum_10"] = df["close"] - df["close"].shift(10)
        return df

    # ------------------------------------------------------------------
    # Indicator helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        tr = pd.concat(
            [
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()
