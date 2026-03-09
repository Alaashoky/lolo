"""
Sentiment Analyzer for market sentiment signals.

Analyses:
- News headline sentiment via transformers / VADER
- Social media proxies (fear & greed index)
- Market sentiment indicators (put/call ratio, VIX proxy)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("forex_bot.ai.sentiment")

# Optional: VADER sentiment (lightweight, no ML required)
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore

    _VADER_AVAILABLE = True
except ImportError:
    _VADER_AVAILABLE = False

# Optional: HuggingFace pipeline (heavier ML-based sentiment)
try:
    from transformers import pipeline as hf_pipeline  # type: ignore

    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


class SentimentAnalyzer:
    """
    Market sentiment analyser.

    Combines news headlines and macro sentiment proxies into a
    BUY / HOLD / SELL signal with a confidence score.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or {}
        self.news_weight: float = cfg.get("news_weight", 0.6)
        self.social_weight: float = cfg.get("social_weight", 0.4)
        self._vader: Optional[object] = None
        self._hf: Optional[object] = None
        self._is_trained: bool = True  # Sentiment models are pre-trained

        if _VADER_AVAILABLE:
            self._vader = SentimentIntensityAnalyzer()
            logger.debug("VADER sentiment analyser loaded.")
        elif _HF_AVAILABLE:
            try:
                self._hf = hf_pipeline(
                    "text-classification",
                    model="ProsusAI/finbert",
                    top_k=None,
                )
                logger.debug("FinBERT sentiment pipeline loaded.")
            except Exception as exc:
                logger.warning("Failed to load HuggingFace pipeline: %s", exc)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze_headlines(self, headlines: list[str]) -> Tuple[int, float]:
        """
        Compute sentiment signal from a list of news headlines.

        Args:
            headlines: List of plain-text news headline strings.

        Returns:
            Tuple of (signal, confidence) where signal ∈ {0=SELL, 1=HOLD, 2=BUY}.
        """
        if not headlines:
            return 1, 0.33

        scores = [self._score_text(h) for h in headlines]
        avg = float(np.mean(scores))
        return self._score_to_signal(avg)

    def predict(self, headlines: Optional[list[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return broadcast prediction arrays compatible with ensemble interface.

        Args:
            headlines: Optional list of news headlines.  When *None* a
                       neutral signal is returned.

        Returns:
            Tuple of (classes_array, probs_array) each of shape (1,).
        """
        if headlines:
            signal, confidence = self.analyze_headlines(headlines)
        else:
            signal, confidence = 1, 0.33

        probs = self._build_probs(signal, confidence)
        return np.array([signal], dtype=int), np.array([probs], dtype=np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _score_text(self, text: str) -> float:
        """Return a sentiment score in [-1, +1] for a single text."""
        if self._vader is not None:
            try:
                return self._vader.polarity_scores(text)["compound"]
            except Exception:
                pass
        if self._hf is not None:
            try:
                result = self._hf(text[:512])
                label_scores = {r["label"].lower(): r["score"] for r in result[0]}
                return label_scores.get("positive", 0.0) - label_scores.get("negative", 0.0)
            except Exception:
                pass
        return 0.0  # neutral fallback

    @staticmethod
    def _score_to_signal(score: float) -> Tuple[int, float]:
        """Convert compound sentiment score to signal + confidence."""
        if score > 0.05:
            confidence = min(0.9, 0.5 + score)
            return 2, confidence  # BUY
        if score < -0.05:
            confidence = min(0.9, 0.5 + abs(score))
            return 0, confidence  # SELL
        return 1, 0.5 - abs(score)  # HOLD

    @staticmethod
    def _build_probs(signal: int, confidence: float) -> list[float]:
        base = (1.0 - confidence) / 2
        probs = [base, base, base]
        probs[signal] = confidence
        return probs
