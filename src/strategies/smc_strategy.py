"""
Smart Money Concepts (SMC) Strategy.

Detects Break of Structure (BOS), Fair Value Gaps (FVG),
and Order Blocks to generate trading signals.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional


class SMCStrategy:
    """
    Smart Money Concepts strategy.

    Generates BUY / SELL / NEUTRAL signals based on:
    - Break of Structure (BOS)
    - Fair Value Gap (FVG)
    - Order Block recognition
    """

    def __init__(self, config: dict) -> None:
        """
        Args:
            config: Strategy configuration dict from strategies.json.
        """
        self.enabled: bool = config.get("enabled", True)
        self.bos_threshold: float = config.get("bos_threshold", 0.002)
        self.weight: float = config.get("weight", 0.25)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Run SMC analysis on OHLCV data.

        Args:
            df: DataFrame with columns [open, high, low, close, volume].
                Must have at least 20 rows.

        Returns:
            dict with keys:
              - signal (str): 'BUY', 'SELL', or 'NEUTRAL'
              - confidence (float): 0.0 – 1.0
              - details (dict): intermediate analysis results
        """
        if not self.enabled or len(df) < 20:
            return {"signal": "NEUTRAL", "confidence": 0.0, "details": {}}

        bos = self._detect_bos(df)
        fvg = self._detect_fvg(df)
        ob = self._detect_order_block(df)

        signal, confidence = self._build_signal(bos, fvg, ob)

        return {
            "signal": signal,
            "confidence": confidence,
            "details": {"bos": bos, "fvg": fvg, "order_block": ob},
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_bos(self, df: pd.DataFrame) -> Optional[str]:
        """
        Identify a Break of Structure.

        A bullish BOS occurs when the latest close exceeds the previous
        swing high by more than *bos_threshold*.
        A bearish BOS occurs when the latest close falls below the previous
        swing low by more than *bos_threshold*.
        """
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        # Use last 10 bars to find swing high/low
        window = min(10, len(df) - 1)
        swing_high = highs[-window - 1 : -1].max()
        swing_low = lows[-window - 1 : -1].min()
        last_close = closes[-1]

        if last_close > swing_high * (1 + self.bos_threshold):
            return "bullish"
        if last_close < swing_low * (1 - self.bos_threshold):
            return "bearish"
        return None

    def _detect_fvg(self, df: pd.DataFrame) -> Optional[str]:
        """
        Detect a Fair Value Gap in the last three candles.

        Bullish FVG: candle[-3].high < candle[-1].low
        Bearish FVG: candle[-3].low  > candle[-1].high
        """
        if len(df) < 3:
            return None

        c1_high = df["high"].iloc[-3]
        c3_low = df["low"].iloc[-1]
        c1_low = df["low"].iloc[-3]
        c3_high = df["high"].iloc[-1]

        if c1_high < c3_low:
            return "bullish"
        if c1_low > c3_high:
            return "bearish"
        return None

    def _detect_order_block(self, df: pd.DataFrame) -> Optional[str]:
        """
        Identify the most recent order block.

        A bullish order block is the last bearish candle before a strong
        bullish move. A bearish order block is the last bullish candle
        before a strong bearish move.
        """
        if len(df) < 5:
            return None

        closes = df["close"].values
        opens = df["open"].values

        # Check last 5 candles for order block pattern
        for i in range(-5, -1):
            is_bearish_candle = closes[i] < opens[i]
            strong_bullish_move = closes[i + 1] > opens[i] * (1 + self.bos_threshold)
            if is_bearish_candle and strong_bullish_move:
                return "bullish"

            is_bullish_candle = closes[i] > opens[i]
            strong_bearish_move = closes[i + 1] < opens[i] * (1 - self.bos_threshold)
            if is_bullish_candle and strong_bearish_move:
                return "bearish"

        return None

    def _build_signal(
        self,
        bos: Optional[str],
        fvg: Optional[str],
        ob: Optional[str],
    ) -> tuple[str, float]:
        """Combine individual signals into a final signal with confidence."""
        scores = {"bullish": 0, "bearish": 0}
        total = 0

        for signal in [bos, fvg, ob]:
            if signal in scores:
                scores[signal] += 1
                total += 1

        if total == 0:
            return "NEUTRAL", 0.0

        bull_score = scores["bullish"] / total
        bear_score = scores["bearish"] / total

        if bull_score > bear_score and bull_score >= 0.5:
            return "BUY", bull_score
        if bear_score > bull_score and bear_score >= 0.5:
            return "SELL", bear_score
        return "NEUTRAL", max(bull_score, bear_score)
