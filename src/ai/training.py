"""
Training System for the AI Ensemble.

Handles:
- Data preparation for each model type
- Sequential training pipeline
- Cross-validation
- Model persistence
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from src.ai.config import DEFAULT_CONFIG, SIGNAL_TO_INT
from src.ai.data_processor import DataProcessor
from src.ai.feature_engineering import FeatureEngineer
from src.ai.ensemble import AIEnsemble
from src.ai.meta_learner import MetaLearner

logger = logging.getLogger("forex_bot.ai.training")


class TrainingPipeline:
    """
    End-to-end training pipeline for the AI ensemble.

    Steps:
    1. Process and split the raw OHLCV data
    2. Engineer features
    3. Generate labels (next-bar direction)
    4. Train each base model
    5. Train the meta-learner on stacked outputs
    6. Save all models to disk
    """

    def __init__(
        self,
        ensemble: AIEnsemble,
        meta_learner: MetaLearner,
        config: Optional[dict] = None,
    ) -> None:
        cfg = config or DEFAULT_CONFIG
        ai_cfg = cfg.get("ai", DEFAULT_CONFIG["ai"])
        self._cfg = ai_cfg
        self._ensemble = ensemble
        self._meta_learner = meta_learner
        self._models_dir: str = ai_cfg.get("models_dir", "models/ai")
        self._lookback: int = ai_cfg.get("lookback_window", 60)
        self._min_samples: int = ai_cfg.get("training", {}).get("min_samples", 500)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_all(self, df: pd.DataFrame, pair: str = "UNKNOWN") -> dict:
        """
        Train every base model and the meta-learner.

        Args:
            df: Raw OHLCV DataFrame (should have at least min_samples rows).
            pair: Currency pair name used as a subfolder for saved models.

        Returns:
            Dict with per-model training results.
        """
        processor = self._ensemble.data_processor
        engineer = self._ensemble.feature_engineer

        if len(df) < self._min_samples:
            logger.warning(
                "TrainingPipeline: only %d rows for %s (min %d) – skipping.",
                len(df), pair, self._min_samples,
            )
            return {}

        # 1. Process data
        df_clean = processor.process(df)
        train_df, val_df, _ = processor.split(df_clean)

        # 2. Feature engineering
        train_feat = engineer.extract(train_df)
        feature_cols = [c for c in engineer.feature_names() if c in train_feat.columns]

        X_flat = train_feat[feature_cols].values.astype(np.float32)

        # 3. Generate labels: 1-bar-ahead direction
        y = self._make_labels(train_feat)

        # Align lengths
        min_len = min(len(X_flat), len(y))
        X_flat = X_flat[:min_len]
        y = y[:min_len]

        # 4. Build sequence tensors for LSTM/Transformer/CNN
        X_seq = self._build_sequences(X_flat)
        y_seq = y[self._lookback:]  # corresponding labels
        min_seq = min(len(X_seq), len(y_seq))
        X_seq = X_seq[:min_seq]
        y_seq = y_seq[:min_seq]

        results: dict = {}

        # 5. Train each model
        models = self._ensemble.models
        model_dispatch = {
            "lstm":          lambda: models["lstm"].train(X_seq, y_seq),
            "transformer":   lambda: models["transformer"].train(X_seq, y_seq),
            "cnn":           lambda: models["cnn"].train(X_seq, y_seq),
            "xgboost":       lambda: models["xgboost"].train(X_flat, y),
            "random_forest": lambda: models["random_forest"].train(X_flat, y),
            "prophet":       lambda: models["prophet"].train(train_df),
            "anomaly":       lambda: models["anomaly"].train(X_flat),
            "sentiment":     lambda: {},  # pre-trained; no training needed
        }

        for name, train_fn in model_dispatch.items():
            try:
                res = train_fn()
                results[name] = res
                logger.info("Trained model: %s", name)
            except Exception as exc:
                logger.warning("Failed to train %s: %s", name, exc)
                results[name] = {"error": str(exc)}

        # 6. Train meta-learner using validation set
        self._train_meta_learner(val_df, engineer, feature_cols, results, pair)

        # 7. Save all models
        self.save_all(pair)

        return results

    def save_all(self, pair: str = "default") -> None:
        """Persist all models to disk."""
        base = os.path.join(self._models_dir, pair)
        os.makedirs(base, exist_ok=True)

        for name, model in self._ensemble.models.items():
            path = os.path.join(base, name)
            try:
                model.save(path)  # type: ignore[attr-defined]
            except Exception as exc:
                logger.warning("Could not save model %s: %s", name, exc)

        meta_path = os.path.join(base, "meta_learner")
        try:
            self._meta_learner.save(meta_path)
        except Exception as exc:
            logger.warning("Could not save meta-learner: %s", exc)

    def load_all(self, pair: str = "default") -> None:
        """Load all persisted models from disk."""
        base = os.path.join(self._models_dir, pair)

        for name, model in self._ensemble.models.items():
            path = os.path.join(base, name)
            try:
                model.load(path)  # type: ignore[attr-defined]
            except Exception as exc:
                logger.debug("Could not load model %s: %s", name, exc)

        meta_path = os.path.join(base, "meta_learner")
        try:
            self._meta_learner.load(meta_path)
        except Exception as exc:
            logger.debug("Could not load meta-learner: %s", exc)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_meta_learner(
        self,
        val_df: pd.DataFrame,
        engineer: FeatureEngineer,
        feature_cols: list[str],
        results: dict,
        pair: str,
    ) -> None:
        """Collect stacked probabilities on the validation set and train the meta-learner."""
        if len(val_df) < 30:
            logger.debug("Skipping meta-learner training – insufficient validation data.")
            return

        try:
            val_feat = engineer.extract(val_df)
            available = [c for c in feature_cols if c in val_feat.columns]
            X_val_flat = val_feat[available].values.astype(np.float32)
            y_val = self._make_labels(val_feat)
            min_len = min(len(X_val_flat), len(y_val))
            X_val_flat = X_val_flat[:min_len]
            y_val = y_val[:min_len]

            models = self._ensemble.models
            all_probs: list[np.ndarray] = []

            for name in ["xgboost", "random_forest", "anomaly"]:
                model = models[name]
                if getattr(model, "_is_trained", False):
                    _, p = model.predict(X_val_flat)  # type: ignore[attr-defined]
                    all_probs.append(p)

            if not all_probs:
                return

            stacked = np.hstack(all_probs)
            if len(stacked) >= self._meta_learner.min_samples:
                self._meta_learner.train(stacked, y_val)
        except Exception as exc:
            logger.warning("Meta-learner training failed: %s", exc)

    @staticmethod
    def _make_labels(df: pd.DataFrame, threshold: float = 0.0005) -> np.ndarray:
        """
        Generate next-bar directional labels.

        Returns:
            Integer array (0=SELL, 1=HOLD, 2=BUY).
        """
        close = df["close"].values
        returns = np.diff(close) / (close[:-1] + 1e-10)
        labels = np.where(returns > threshold, 2,
                 np.where(returns < -threshold, 0, 1))
        return labels.astype(int)

    def _build_sequences(self, X: np.ndarray) -> np.ndarray:
        """Build (n, lookback, n_features) tensor."""
        if len(X) <= self._lookback:
            return np.empty((0, self._lookback, X.shape[1]), dtype=np.float32)
        seqs = [X[i - self._lookback : i] for i in range(self._lookback, len(X))]
        return np.array(seqs, dtype=np.float32)
