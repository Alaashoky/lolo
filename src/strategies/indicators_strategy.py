"""
Technical Indicators Strategy.

Uses RSI, Stochastic Oscillator, Bollinger Bands, and ATR to
generate trading signals without relying on TA-Lib so the module
works in environments where TA-Lib is not compiled.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional


class IndicatorsStrategy:
    """
    Technical indicators strategy.

    Generates BUY / SELL / NEUTRAL signals based on:
    - RSI (Relative Strength Index)
    - Stochastic Oscillator (%K / %D)
    - Bollinger Bands position
    - ATR-based volatility filter
    """

    def __init__(self, config: dict) -> None:
        """
        Args:
            config: Strategy configuration dict from strategies.json.
        """
        self.enabled: bool = config.get("enabled", True)
        self.weight: float = config.get("weight", 0.25)
        self.rsi_period: int = config.get("rsi_period", 14)
        self.rsi_overbought: float = config.get("rsi_overbought", 70.0)
        self.rsi_oversold: float = config.get("rsi_oversold", 30.0)
        self.stoch_period: int = config.get("stoch_period", 14)
        self.bb_period: int = config.get("bb_period", 20)
        self.bb_std: float = config.get("bb_std", 2.0)
        self.atr_period: int = config.get("atr_period", 14)
        self.min_atr_multiplier: float = config.get("min_atr_multiplier", 0.5)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Run technical indicator analysis on OHLCV data.

        Args:
            df: DataFrame with columns [open, high, low, close, volume].

        Returns:
            dict with signal, confidence, and details.
        """
        min_rows = max(self.rsi_period, self.bb_period, self.stoch_period) + 5
        if not self.enabled or len(df) < min_rows:
            return {"signal": "NEUTRAL", "confidence": 0.0, "details": {}}

        rsi_signal = self._rsi_signal(df)
        stoch_signal = self._stochastic_signal(df)
        bb_signal = self._bollinger_signal(df)
        atr_ok = self._atr_filter(df)

        if not atr_ok:
            return {
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "details": {"reason": "low_volatility"},
            }

        signal, confidence = self._build_signal(rsi_signal, stoch_signal, bb_signal)

        rsi_val = self._compute_rsi(df["close"])
        return {
            "signal": signal,
            "confidence": confidence,
            "details": {
                "rsi": round(rsi_val, 2),
                "rsi_signal": rsi_signal,
                "stoch_signal": stoch_signal,
                "bb_signal": bb_signal,
            },
        }

    # ------------------------------------------------------------------
    # Indicator computations
    # ------------------------------------------------------------------

    def _compute_rsi(self, closes: pd.Series) -> float:
        """Compute the current RSI value."""
        delta = closes.diff().dropna()
        gain = delta.clip(lower=0).rolling(self.rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(self.rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not rsi.empty else 50.0

    def _rsi_signal(self, df: pd.DataFrame) -> Optional[str]:
        """Return 'bullish' if RSI is oversold, 'bearish' if overbought."""
        rsi = self._compute_rsi(df["close"])
        if rsi <= self.rsi_oversold:
            return "bullish"
        if rsi >= self.rsi_overbought:
            return "bearish"
        return None

    def _stochastic_signal(self, df: pd.DataFrame) -> Optional[str]:
        """
        Compute %K and %D of the Stochastic Oscillator.
        Returns 'bullish' if both below 20, 'bearish' if both above 80.
        """
        period = self.stoch_period
        low_min = df["low"].rolling(period).min()
        high_max = df["high"].rolling(period).max()
        k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
        d = k.rolling(3).mean()

        k_val = float(k.iloc[-1]) if not k.empty else 50.0
        d_val = float(d.iloc[-1]) if not d.empty else 50.0

        if k_val < 20 and d_val < 20:
            return "bullish"
        if k_val > 80 and d_val > 80:
            return "bearish"
        return None

    def _bollinger_signal(self, df: pd.DataFrame) -> Optional[str]:
        """
        Bollinger Bands signal.
        'bullish' if close is below the lower band,
        'bearish' if close is above the upper band.
        """
        closes = df["close"]
        sma = closes.rolling(self.bb_period).mean()
        std = closes.rolling(self.bb_period).std()
        upper = sma + self.bb_std * std
        lower = sma - self.bb_std * std

        last_close = closes.iloc[-1]
        last_upper = upper.iloc[-1]
        last_lower = lower.iloc[-1]

        if last_close < last_lower:
            return "bullish"
        if last_close > last_upper:
            return "bearish"
        return None

    def _atr_filter(self, df: pd.DataFrame) -> bool:
        """
        Return True if ATR is above the minimum threshold,
        indicating sufficient market volatility to trade.
        """
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean().iloc[-1]
        avg_close = df["close"].iloc[-self.atr_period :].mean()
        return bool(atr > avg_close * self.min_atr_multiplier * 0.001)

    def _build_signal(
        self,
        rsi: Optional[str],
        stoch: Optional[str],
        bb: Optional[str],
    ) -> tuple[str, float]:
        """Combine indicator signals into a final signal."""
        scores = {"bullish": 0, "bearish": 0}
        total = 0

        for sig in [rsi, stoch, bb]:
            if sig in scores:
                scores[sig] += 1
                total += 1

        if total == 0:
            return "NEUTRAL", 0.0

        bull = scores["bullish"] / total
        bear = scores["bearish"] / total

        if bull > bear and bull >= 0.5:
            return "BUY", bull
        if bear > bull and bear >= 0.5:
            return "SELL", bear
        return "NEUTRAL", max(bull, bear)
