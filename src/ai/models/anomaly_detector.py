"""
Anomaly Detector for unusual market movement detection.

Uses Isolation Forest to identify statistical outliers in:
- Price spikes
- Volume anomalies
- Volatility bursts
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("forex_bot.ai.anomaly")

try:
    from sklearn.ensemble import IsolationForest  # type: ignore

    _SKL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKL_AVAILABLE = False
    logger.warning("scikit-learn not installed – AnomalyDetector will run in fallback mode.")


class AnomalyDetector:
    """
    Isolation Forest-based market anomaly detector.

    Anomalous bars are flagged and the ensemble receives a HOLD signal
    with low confidence, preventing trades during erratic conditions.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or {}
        self.contamination: float = cfg.get("contamination", 0.05)
        self.n_estimators: int = cfg.get("n_estimators", 100)
        self._model: Optional[object] = None
        self._is_trained: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X: np.ndarray) -> dict:
        """
        Fit the Isolation Forest on feature data.

        Args:
            X: Shape (n_samples, n_features).

        Returns:
            Dict with anomaly_rate (fraction of training rows flagged).
        """
        if not _SKL_AVAILABLE:
            logger.warning("AnomalyDetector.train: scikit-learn unavailable – skipping.")
            return {}

        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1,
        )
        labels = self._model.fit_predict(X)
        anomaly_rate = float((labels == -1).mean())
        self._is_trained = True
        logger.info("AnomalyDetector training complete. Anomaly rate=%.2f%%", anomaly_rate * 100)
        return {"anomaly_rate": anomaly_rate}

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in new data.

        Returns:
            Tuple of (classes, probs) where:
              - classes = 1 (HOLD) for anomalies, else -1 (no anomaly)
              - probs[i] = [0.1, 0.8, 0.1] for anomaly, else uniform
        """
        n = len(X)
        if not _SKL_AVAILABLE or not self._is_trained or self._model is None:
            return np.zeros(n, dtype=int), np.full((n, 3), 1 / 3, dtype=np.float32)

        raw = self._model.predict(X)  # +1 = normal, -1 = anomaly
        scores = self._model.score_samples(X)  # Lower = more anomalous

        probs = np.full((n, 3), 1 / 3, dtype=np.float32)
        classes = np.zeros(n, dtype=int)

        for i, (label, score) in enumerate(zip(raw, scores)):
            if label == -1:
                # Anomaly → strong HOLD signal
                probs[i] = [0.1, 0.8, 0.1]
                classes[i] = 1  # HOLD
            else:
                # Normal → weak directional bias via anomaly score
                confidence = min(0.6, 0.33 + abs(score) * 0.1)
                probs[i] = [1 / 3, 1 / 3, 1 / 3]
                classes[i] = 1  # neutral

        return classes, probs

    def is_anomalous(self, X: np.ndarray) -> np.ndarray:
        """Return boolean array – True where a bar is anomalous."""
        if not _SKL_AVAILABLE or not self._is_trained or self._model is None:
            return np.zeros(len(X), dtype=bool)
        return self._model.predict(X) == -1

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
