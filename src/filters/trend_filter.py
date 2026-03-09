"""
Trend Filter.

Uses moving averages and price structure analysis to determine the
prevailing trend and decide whether trading is permitted.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional


class TrendFilter:
    """
    Moving average based trend filter.

    Returns True (trading allowed) when price is aligned with the
    detected trend direction requested by the calling strategy signal.
    """

    def __init__(self, config: dict) -> None:
        """
        Args:
            config: Settings dict; supports 'fast_ma', 'slow_ma', 'trend_ma' keys.
        """
        self.enabled: bool = config.get("trend_filter_enabled", True)
        self.fast_ma: int = config.get("fast_ma", 20)
        self.slow_ma: int = config.get("slow_ma", 50)
        self.trend_ma: int = config.get("trend_ma", 200)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_trade_allowed(self, df: pd.DataFrame, signal: str) -> bool:
        """
        Check whether the given signal is aligned with the current trend.

        Args:
            df: OHLCV DataFrame.
            signal: 'BUY' or 'SELL' from a strategy.

        Returns:
            True if the signal is aligned with the trend (or filter is
            disabled), False otherwise.
        """
        if not self.enabled:
            return True

        trend = self._get_trend(df)
        if trend is None:
            return True  # no trend detected → allow trading

        if signal == "BUY" and trend == "bullish":
            return True
        if signal == "SELL" and trend == "bearish":
            return True
        return False

    def get_trend(self, df: pd.DataFrame) -> Optional[str]:
        """Return the current trend as 'bullish', 'bearish', or None."""
        return self._get_trend(df)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_trend(self, df: pd.DataFrame) -> Optional[str]:
        """
        Determine trend using a multi-MA approach.

        Uses fast / slow / trend MAs when sufficient data exists,
        falls back to fewer MAs for shorter series.
        """
        closes = df["close"]
        n = len(closes)

        if n >= self.trend_ma:
            fast = closes.rolling(self.fast_ma).mean().iloc[-1]
            slow = closes.rolling(self.slow_ma).mean().iloc[-1]
            trend = closes.rolling(self.trend_ma).mean().iloc[-1]
            last = closes.iloc[-1]

            if last > trend and fast > slow:
                return "bullish"
            if last < trend and fast < slow:
                return "bearish"
            return None

        if n >= self.slow_ma:
            fast = closes.rolling(self.fast_ma).mean().iloc[-1]
            slow = closes.rolling(self.slow_ma).mean().iloc[-1]
            if fast > slow:
                return "bullish"
            if fast < slow:
                return "bearish"
            return None

        if n >= self.fast_ma:
            fast_ma = closes.rolling(self.fast_ma).mean().iloc[-1]
            last = closes.iloc[-1]
            return "bullish" if last > fast_ma else "bearish"

        return None

    def _price_structure_trend(self, df: pd.DataFrame) -> Optional[str]:
        """
        Secondary validation using price structure (HH/HL or LL/LH).
        """
        if len(df) < 4:
            return None

        highs = df["high"].values
        lows = df["low"].values
        mid = len(highs) // 2

        hh = highs[mid:].max() > highs[:mid].max()
        hl = lows[mid:].min() > lows[:mid].min()
        lh = highs[mid:].max() < highs[:mid].max()
        ll = lows[mid:].min() < lows[:mid].min()

        if hh and hl:
            return "bullish"
        if lh and ll:
            return "bearish"
        return None
