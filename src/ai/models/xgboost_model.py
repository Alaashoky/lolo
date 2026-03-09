"""
XGBoost Classifier for BUY / SELL / HOLD signal generation.

Features:
- Ternary classification (0=SELL, 1=HOLD, 2=BUY)
- Feature importance ranking
- Probability calibration
- Hyperparameter support
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("forex_bot.ai.xgboost")

try:
    import xgboost as xgb  # type: ignore
    from sklearn.calibration import CalibratedClassifierCV  # type: ignore

    _XGB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _XGB_AVAILABLE = False
    logger.warning("XGBoost not installed – XGBoostModel will run in fallback mode.")


class XGBoostModel:
    """
    XGBoost-based trading signal classifier.

    Falls back to uniform predictions when XGBoost is unavailable.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or {}
        self.n_estimators: int = cfg.get("n_estimators", 200)
        self.max_depth: int = cfg.get("max_depth", 6)
        self.learning_rate: float = cfg.get("learning_rate", 0.05)
        self.subsample: float = cfg.get("subsample", 0.8)
        self._model: Optional[object] = None
        self._feature_importances: Optional[np.ndarray] = None
        self._is_trained: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Train the XGBoost classifier.

        Args:
            X: Shape (n_samples, n_features).
            y: Integer labels (0=SELL, 1=HOLD, 2=BUY).

        Returns:
            Dict with feature_importances key.
        """
        if not _XGB_AVAILABLE:
            logger.warning("XGBoostModel.train: XGBoost unavailable – skipping.")
            return {}

        base = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
        self._model = CalibratedClassifierCV(base, cv=3, method="isotonic")
        self._model.fit(X, y)

        # Feature importances extracted from the calibrated model's base estimator
        try:
            fitted_base = self._model.calibrated_classifiers_[0].estimator
            self._feature_importances = fitted_base.feature_importances_
        except (AttributeError, IndexError):
            self._feature_importances = None

        self._is_trained = True
        logger.info("XGBoostModel training complete. n_samples=%d", len(X))
        return {"feature_importances": self._feature_importances}

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (predicted_classes, probabilities)."""
        if not _XGB_AVAILABLE or not self._is_trained or self._model is None:
            n = len(X)
            probs = np.full((n, 3), 1 / 3, dtype=np.float32)
            return np.ones(n, dtype=int), probs

        probs = self._model.predict_proba(X).astype(np.float32)
        # Ensure 3 columns (SELL, HOLD, BUY)
        if probs.shape[1] < 3:
            pad = np.zeros((len(probs), 3 - probs.shape[1]), dtype=np.float32)
            probs = np.hstack([probs, pad])
        classes = np.argmax(probs, axis=1)
        return classes, probs

    def feature_importance(self) -> Optional[np.ndarray]:
        """Return feature importance scores, or None if not available."""
        return self._feature_importances

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path + ".pkl", "wb") as fh:
            pickle.dump(
                {"model": self._model, "is_trained": self._is_trained,
                 "feature_importances": self._feature_importances},
                fh,
            )

    def load(self, path: str) -> None:
        pkl_path = path + ".pkl"
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as fh:
                data = pickle.load(fh)
            self._model = data.get("model")
            self._is_trained = data.get("is_trained", False)
            self._feature_importances = data.get("feature_importances")
