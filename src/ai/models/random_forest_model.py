"""
Random Forest Model for ensemble voting.

Features:
- 200+ decision trees
- Out-of-bag error estimation
- Feature importance analysis
- Voting mechanism for signal generation
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("forex_bot.ai.random_forest")

try:
    from sklearn.ensemble import RandomForestClassifier  # type: ignore

    _SKL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKL_AVAILABLE = False
    logger.warning("scikit-learn not installed – RandomForestModel will run in fallback mode.")


class RandomForestModel:
    """
    Random Forest classifier for trading signal prediction.

    Falls back to uniform predictions when scikit-learn is unavailable.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or {}
        self.n_estimators: int = cfg.get("n_estimators", 200)
        self.max_depth: Optional[int] = cfg.get("max_depth", 10)
        self.min_samples_split: int = cfg.get("min_samples_split", 5)
        self._model: Optional[RandomForestClassifier] = None  # type: ignore
        self._is_trained: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Train the Random Forest classifier.

        Args:
            X: Shape (n_samples, n_features).
            y: Integer labels (0=SELL, 1=HOLD, 2=BUY).

        Returns:
            Dict with oob_score and feature_importances.
        """
        if not _SKL_AVAILABLE:
            logger.warning("RandomForestModel.train: scikit-learn unavailable – skipping.")
            return {}

        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            oob_score=True,
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(X, y)
        self._is_trained = True

        oob = getattr(self._model, "oob_score_", None)
        logger.info("RandomForestModel training complete. OOB score=%.4f", oob or 0.0)
        return {
            "oob_score": oob,
            "feature_importances": self._model.feature_importances_,
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (predicted_classes, probabilities)."""
        if not _SKL_AVAILABLE or not self._is_trained or self._model is None:
            n = len(X)
            probs = np.full((n, 3), 1 / 3, dtype=np.float32)
            return np.ones(n, dtype=int), probs

        probs = self._model.predict_proba(X).astype(np.float32)
        if probs.shape[1] < 3:
            pad = np.zeros((len(probs), 3 - probs.shape[1]), dtype=np.float32)
            probs = np.hstack([probs, pad])
        classes = np.argmax(probs, axis=1)
        return classes, probs

    def feature_importance(self) -> Optional[np.ndarray]:
        """Return feature importance scores."""
        if self._model is not None and self._is_trained:
            return self._model.feature_importances_
        return None

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
