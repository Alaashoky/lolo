"""
ICT (Inner Circle Trader) Liquidity Strategy.

Detects liquidity sweeps and analyses market structure to
generate directional trading signals.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional


class ICTStrategy:
    """
    ICT-based strategy.

    Generates BUY / SELL / NEUTRAL signals based on:
    - Liquidity sweep detection (stop hunts above/below key levels)
    - Market structure (higher highs / lower lows)
    """

    def __init__(self, config: dict) -> None:
        """
        Args:
            config: Strategy configuration dict from strategies.json.
        """
        self.enabled: bool = config.get("enabled", True)
        self.weight: float = config.get("weight", 0.25)
        self.sweep_threshold: float = config.get("sweep_threshold", 0.001)
        self.structure_lookback: int = config.get("structure_lookback", 20)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Run ICT analysis on OHLCV data.

        Args:
            df: DataFrame with columns [open, high, low, close, volume].

        Returns:
            dict with signal, confidence, and details.
        """
        if not self.enabled or len(df) < self.structure_lookback:
            return {"signal": "NEUTRAL", "confidence": 0.0, "details": {}}

        sweep = self._detect_liquidity_sweep(df)
        structure = self._analyze_market_structure(df)

        signal, confidence = self._build_signal(sweep, structure)

        return {
            "signal": signal,
            "confidence": confidence,
            "details": {"liquidity_sweep": sweep, "market_structure": structure},
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_liquidity_sweep(self, df: pd.DataFrame) -> Optional[str]:
        """
        Detect a liquidity sweep (stop hunt) in recent price action.

        A bullish sweep occurs when price wicks below a recent swing low
        and then closes back above it (false break down → long).
        A bearish sweep occurs when price wicks above a recent swing high
        and then closes back below it (false break up → short).
        """
        lookback = min(self.structure_lookback, len(df) - 1)
        recent = df.iloc[-lookback - 1 : -1]

        swing_high = recent["high"].max()
        swing_low = recent["low"].min()

        last_bar = df.iloc[-1]

        # Wick below swing low then close above it → bullish sweep
        if last_bar["low"] < swing_low and last_bar["close"] > swing_low:
            return "bullish"

        # Wick above swing high then close below it → bearish sweep
        if last_bar["high"] > swing_high and last_bar["close"] < swing_high:
            return "bearish"

        return None

    def _analyze_market_structure(self, df: pd.DataFrame) -> Optional[str]:
        """
        Determine market structure by comparing the last two swing
        highs and lows.

        Bullish structure: higher highs AND higher lows.
        Bearish structure: lower highs AND lower lows.
        """
        if len(df) < 4:
            return None

        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values

        mid = len(closes) // 2

        first_half_high = highs[:mid].max()
        second_half_high = highs[mid:].max()
        first_half_low = lows[:mid].min()
        second_half_low = lows[mid:].min()

        hh = second_half_high > first_half_high
        hl = second_half_low > first_half_low
        lh = second_half_high < first_half_high
        ll = second_half_low < first_half_low

        if hh and hl:
            return "bullish"
        if lh and ll:
            return "bearish"
        return None

    def _build_signal(
        self,
        sweep: Optional[str],
        structure: Optional[str],
    ) -> tuple[str, float]:
        """Combine sweep and structure into a final signal."""
        scores = {"bullish": 0, "bearish": 0}
        total = 0

        for sig in [sweep, structure]:
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
