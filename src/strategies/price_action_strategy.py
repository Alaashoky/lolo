"""
Price Action Strategy.

Detects classic candlestick patterns (pin bars, engulfing candles)
and key support/resistance levels to generate trading signals.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional


class PriceActionStrategy:
    """
    Price action strategy based on candlestick patterns and S/R levels.

    Signals are derived from:
    - Pin bar (hammer / shooting star) detection
    - Engulfing candle patterns
    - Dynamic support / resistance proximity
    """

    def __init__(self, config: dict) -> None:
        """
        Args:
            config: Strategy configuration dict from strategies.json.
        """
        self.enabled: bool = config.get("enabled", True)
        self.weight: float = config.get("weight", 0.25)
        self.pin_bar_ratio: float = config.get("pin_bar_ratio", 2.5)
        self.sr_lookback: int = config.get("sr_lookback", 50)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Run price action analysis on OHLCV data.

        Args:
            df: DataFrame with columns [open, high, low, close, volume].

        Returns:
            dict with signal, confidence, and details.
        """
        if not self.enabled or len(df) < 5:
            return {"signal": "NEUTRAL", "confidence": 0.0, "details": {}}

        pin = self._detect_pin_bar(df)
        engulf = self._detect_engulfing(df)
        sr = self._check_support_resistance(df)

        signal, confidence = self._build_signal(pin, engulf, sr)

        return {
            "signal": signal,
            "confidence": confidence,
            "details": {"pin_bar": pin, "engulfing": engulf, "sr_level": sr},
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_pin_bar(self, df: pd.DataFrame) -> Optional[str]:
        """
        Detect a pin bar (hammer or shooting star) on the last candle.

        A bullish pin bar has a long lower wick (≥ pin_bar_ratio × body)
        and small upper wick.
        A bearish pin bar has a long upper wick and small lower wick.
        """
        last = df.iloc[-1]
        body = abs(last["close"] - last["open"])
        total_range = last["high"] - last["low"]

        if total_range == 0 or body == 0:
            return None

        lower_wick = min(last["open"], last["close"]) - last["low"]
        upper_wick = last["high"] - max(last["open"], last["close"])

        if lower_wick >= self.pin_bar_ratio * body and upper_wick <= body:
            return "bullish"
        if upper_wick >= self.pin_bar_ratio * body and lower_wick <= body:
            return "bearish"
        return None

    def _detect_engulfing(self, df: pd.DataFrame) -> Optional[str]:
        """
        Detect a bullish or bearish engulfing pattern on the last two candles.
        """
        if len(df) < 2:
            return None

        prev = df.iloc[-2]
        curr = df.iloc[-1]

        prev_body_high = max(prev["open"], prev["close"])
        prev_body_low = min(prev["open"], prev["close"])
        curr_body_high = max(curr["open"], curr["close"])
        curr_body_low = min(curr["open"], curr["close"])

        # Bullish engulfing: previous candle bearish, current candle bullish
        # and fully engulfs previous body
        if (
            prev["close"] < prev["open"]
            and curr["close"] > curr["open"]
            and curr_body_low < prev_body_low
            and curr_body_high > prev_body_high
        ):
            return "bullish"

        # Bearish engulfing: previous candle bullish, current candle bearish
        if (
            prev["close"] > prev["open"]
            and curr["close"] < curr["open"]
            and curr_body_low < prev_body_low
            and curr_body_high > prev_body_high
        ):
            return "bearish"

        return None

    def _check_support_resistance(self, df: pd.DataFrame) -> Optional[str]:
        """
        Check whether price is near a key support or resistance level.

        Uses local swing highs/lows from the last *sr_lookback* bars.
        """
        lookback = min(self.sr_lookback, len(df) - 1)
        history = df.iloc[-lookback - 1 : -1]
        last_close = df["close"].iloc[-1]

        resistance = history["high"].max()
        support = history["low"].min()
        mid = (resistance + support) / 2
        tolerance = (resistance - support) * 0.05  # 5% band

        if abs(last_close - support) <= tolerance:
            return "bullish"
        if abs(last_close - resistance) <= tolerance:
            return "bearish"
        return None

    def _build_signal(
        self,
        pin: Optional[str],
        engulf: Optional[str],
        sr: Optional[str],
    ) -> tuple[str, float]:
        """Combine pattern signals into a final signal with confidence."""
        scores = {"bullish": 0, "bearish": 0}
        total = 0

        for sig in [pin, engulf, sr]:
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
