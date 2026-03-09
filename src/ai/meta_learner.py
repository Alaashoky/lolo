"""
Meta-Learner for final trading signal decision.

The meta-learner:
- Sits on top of the 8-model ensemble
- Learns which models perform best in different market conditions
- Adapts model weights based on recent prediction accuracy
- Applies confidence threshold filtering
- Produces the final BUY / SELL / NEUTRAL signal
"""

from __future__ import annotations

import logging
import os
import pickle
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger("forex_bot.ai.meta_learner")

try:
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.preprocessing import StandardScaler as _Scaler  # type: ignore

    _SKL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKL_AVAILABLE = False


class MetaLearner:
    """
    Stacking meta-learner that combines ensemble model outputs into a
    final trading signal.

    Input features: concatenated probability vectors from all 8 models
    (24 features = 8 models × 3 classes).

    Output: BUY / SELL / NEUTRAL with confidence score.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("meta_learner", {})
        self._lookback: int = cfg.get("lookback_performance", 100)
        self._min_samples: int = cfg.get("min_samples", 20)
        self._adaptation_rate: float = cfg.get("adaptation_rate", 0.1)

        self._meta_model: Optional[LogisticRegression] = None  # type: ignore[type-arg]
        self._scaler: Optional[object] = None
        self._is_trained: bool = False

        # Rolling performance history for adaptive weighting
        self._performance_history: deque = deque(maxlen=self._lookback)

    @property
    def min_samples(self) -> int:
        """Minimum number of samples required to train the meta-learner."""
        return self._min_samples

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, model_probs: np.ndarray, y: np.ndarray) -> dict:
        """
        Train the meta-learner on stacked model probability outputs.

        Args:
            model_probs: Shape (n_samples, n_models * 3) – concatenated
                         probability vectors from all base models.
            y: Integer ground-truth labels (0=SELL, 1=HOLD, 2=BUY).

        Returns:
            Dict with training accuracy.
        """
        if not _SKL_AVAILABLE:
            logger.warning("MetaLearner.train: scikit-learn unavailable – skipping.")
            return {}

        if len(model_probs) < self._min_samples:
            logger.warning("MetaLearner: insufficient samples (%d < %d).", len(model_probs), self._min_samples)
            return {}

        self._scaler = _Scaler()
        X_scaled = self._scaler.fit_transform(model_probs)

        self._meta_model = LogisticRegression(
            max_iter=1000, multi_class="multinomial", random_state=42
        )
        self._meta_model.fit(X_scaled, y)
        self._is_trained = True

        train_acc = float(self._meta_model.score(X_scaled, y))
        logger.info("MetaLearner training complete. Train acc=%.4f", train_acc)
        return {"train_accuracy": train_acc}

    # ------------------------------------------------------------------
    # Prediction / decision
    # ------------------------------------------------------------------

    def decide(
        self,
        ensemble_signal: str,
        ensemble_confidence: float,
        model_details: dict,
        confidence_threshold: float = 0.55,
    ) -> tuple[str, float]:
        """
        Apply meta-level decision logic to the ensemble output.

        When the meta-learner is trained it overrides the raw ensemble
        vote; otherwise it enforces the confidence threshold only.

        Args:
            ensemble_signal: Raw signal from the ensemble ('BUY'/'SELL'/'NEUTRAL').
            ensemble_confidence: Ensemble confidence score [0, 1].
            model_details: Per-model prediction details dict.
            confidence_threshold: Minimum confidence to act.

        Returns:
            Tuple of (final_signal, final_confidence).
        """
        if self._is_trained and model_details:
            meta_probs = self._build_meta_input(model_details)
            if meta_probs is not None:
                signal, conf = self._meta_predict(meta_probs)
                if conf >= confidence_threshold:
                    return signal, conf
                return "NEUTRAL", conf

        # Fallback: pass through ensemble decision
        if ensemble_confidence >= confidence_threshold and ensemble_signal != "NEUTRAL":
            return ensemble_signal, ensemble_confidence
        return "NEUTRAL", ensemble_confidence

    def record_outcome(self, predicted_signal: str, actual_outcome: str) -> None:
        """
        Record a trade outcome for adaptive weight updates.

        Args:
            predicted_signal: The signal that was acted upon.
            actual_outcome: The actual market direction ('BUY'/'SELL'/'NEUTRAL').
        """
        correct = predicted_signal == actual_outcome
        self._performance_history.append(int(correct))

    @property
    def recent_accuracy(self) -> float:
        """Compute rolling accuracy over the performance history window."""
        if not self._performance_history:
            return 0.5
        return float(np.mean(list(self._performance_history)))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path + ".pkl", "wb") as fh:
            pickle.dump(
                {
                    "meta_model": self._meta_model,
                    "scaler": self._scaler,
                    "is_trained": self._is_trained,
                    "performance_history": list(self._performance_history),
                },
                fh,
            )

    def load(self, path: str) -> None:
        pkl_path = path + ".pkl"
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as fh:
                data = pickle.load(fh)
            self._meta_model = data.get("meta_model")
            self._scaler = data.get("scaler")
            self._is_trained = data.get("is_trained", False)
            self._performance_history = deque(
                data.get("performance_history", []), maxlen=self._lookback
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_meta_input(model_details: dict) -> Optional[np.ndarray]:
        """Flatten per-model probabilities into a single feature vector."""
        probs_list = []
        for detail in model_details.values():
            probs = detail.get("probs", [1 / 3, 1 / 3, 1 / 3])
            probs_list.extend(probs[:3])
        if not probs_list:
            return None
        return np.array(probs_list, dtype=np.float32).reshape(1, -1)

    def _meta_predict(self, X: np.ndarray) -> tuple[str, float]:
        """Run the trained meta-model on a (1, n_features) input."""
        from src.ai.config import SIGNAL_LABELS
        from sklearn.linear_model import LogisticRegression as _LR  # type: ignore

        if self._scaler is not None:
            X = self._scaler.transform(X)
        if not isinstance(self._meta_model, _LR):
            return "NEUTRAL", 0.5
        probs = self._meta_model.predict_proba(X)[0]
        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        signal = SIGNAL_LABELS.get(best_idx, "HOLD")
        if signal == "HOLD":
            signal = "NEUTRAL"
        return signal, confidence
