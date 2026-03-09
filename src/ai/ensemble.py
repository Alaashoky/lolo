"""
AI Ensemble System.

Aggregates predictions from all 8 AI models using a weighted voting
mechanism to produce a consensus signal with a confidence score.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd

from src.ai.config import DEFAULT_CONFIG, SIGNAL_LABELS
from src.ai.data_processor import DataProcessor
from src.ai.feature_engineering import FeatureEngineer
from src.ai.models.lstm_model import LSTMModel
from src.ai.models.transformer_model import TransformerModel
from src.ai.models.xgboost_model import XGBoostModel
from src.ai.models.random_forest_model import RandomForestModel
from src.ai.models.cnn_model import CNNModel
from src.ai.models.prophet_model import ProphetModel
from src.ai.models.sentiment_analyzer import SentimentAnalyzer
from src.ai.models.anomaly_detector import AnomalyDetector

logger = logging.getLogger("forex_bot.ai.ensemble")


class AIEnsemble:
    """
    8-model AI ensemble for trading signal generation.

    Models:
    1. LSTM         – price forecasting
    2. Transformer  – sequential pattern recognition
    3. XGBoost      – BUY/SELL/HOLD classification
    4. Random Forest– ensemble voting
    5. CNN          – candlestick pattern detection
    6. Prophet      – time-series forecasting
    7. Sentiment    – news/market sentiment
    8. Anomaly      – unusual movement detection

    Usage::

        ensemble = AIEnsemble(config)
        signal, confidence, details = ensemble.predict(df)
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or DEFAULT_CONFIG
        ai_cfg = cfg.get("ai", DEFAULT_CONFIG["ai"])
        models_cfg = ai_cfg.get("models", {})

        self._cfg = ai_cfg
        self._lookback: int = ai_cfg.get("lookback_window", 60)
        self._confidence_threshold: float = ai_cfg.get("confidence_threshold", 0.55)

        # Data utilities
        self._processor = DataProcessor(ai_cfg)
        self._engineer = FeatureEngineer(ai_cfg)

        # Initialise models
        self._models: dict[str, object] = {}
        self._weights: dict[str, float] = {}
        self._enabled: dict[str, bool] = {}

        model_specs = [
            ("lstm",          LSTMModel,         models_cfg.get("lstm", {})),
            ("transformer",   TransformerModel,  models_cfg.get("transformer", {})),
            ("xgboost",       XGBoostModel,       models_cfg.get("xgboost", {})),
            ("random_forest", RandomForestModel, models_cfg.get("random_forest", {})),
            ("cnn",           CNNModel,          models_cfg.get("cnn", {})),
            ("prophet",       ProphetModel,      models_cfg.get("prophet", {})),
            ("sentiment",     SentimentAnalyzer, models_cfg.get("sentiment", {})),
            ("anomaly",       AnomalyDetector,   models_cfg.get("anomaly", {})),
        ]

        for name, ModelClass, mcfg in model_specs:
            self._models[name] = ModelClass(mcfg)
            self._weights[name] = mcfg.get("weight", 0.125)
            self._enabled[name] = mcfg.get("enabled", True)

        logger.info("AIEnsemble initialised with %d models.", len(self._models))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        df: pd.DataFrame,
        headlines: Optional[list[str]] = None,
    ) -> tuple[str, float, dict]:
        """
        Run all enabled models and return the ensemble decision.

        Args:
            df: OHLCV DataFrame (at least lookback+1 rows).
            headlines: Optional list of news headlines for the sentiment model.

        Returns:
            Tuple of (signal, confidence, details) where:
              - signal: 'BUY', 'SELL', or 'NEUTRAL'
              - confidence: float in [0, 1]
              - details: per-model predictions for transparency
        """
        if len(df) < self._lookback + 1:
            logger.warning("Insufficient data (%d rows) for ensemble prediction.", len(df))
            return "NEUTRAL", 0.0, {}

        # Prepare features
        features_df = self._engineer.extract(df)
        feature_cols = self._engineer.feature_names()
        available_cols = [c for c in feature_cols if c in features_df.columns]

        if len(features_df) == 0:
            logger.warning("Feature extraction produced 0 rows – insufficient lookback data.")
            return "NEUTRAL", 0.0, {}

        # Build flat feature matrix (for non-sequential models)
        X_flat = features_df[available_cols].values.astype(np.float32)

        # Build sequence matrix (for LSTM/Transformer/CNN)
        X_seq = self._build_sequences(X_flat)

        # Collect predictions in parallel
        votes = np.zeros(3, dtype=np.float64)  # [SELL, HOLD, BUY]
        details: dict[str, dict] = {}

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            for name, model in self._models.items():
                if not self._enabled.get(name, True):
                    continue
                futures[executor.submit(
                    self._run_model, name, model, X_flat, X_seq, df, features_df, headlines
                )] = name

            for future in as_completed(futures):
                name = futures[future]
                try:
                    signal_idx, confidence, probs = future.result()
                    weight = self._weights.get(name, 0.125)
                    votes += np.array(probs[-1]) * weight
                    details[name] = {
                        "signal": SIGNAL_LABELS.get(signal_idx, "HOLD"),
                        "confidence": round(float(confidence), 4),
                        "probs": [round(float(p), 4) for p in probs[-1]],
                        "weight": weight,
                    }
                except Exception as exc:
                    logger.warning("Model %s prediction failed: %s", name, exc)

        # Normalise votes
        total = votes.sum()
        if total < 1e-9:
            return "NEUTRAL", 0.0, details

        votes /= total
        best_idx = int(np.argmax(votes))
        best_conf = float(votes[best_idx])
        signal_str = SIGNAL_LABELS.get(best_idx, "HOLD")

        # Map HOLD to NEUTRAL for the bot interface
        if signal_str == "HOLD" or best_conf < self._confidence_threshold:
            return "NEUTRAL", best_conf, details

        return signal_str, best_conf, details

    def is_any_model_trained(self) -> bool:
        """Return True if at least one model has been trained."""
        for model in self._models.values():
            if getattr(model, "_is_trained", False):
                return True
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_model(
        self,
        name: str,
        model: object,
        X_flat: np.ndarray,
        X_seq: np.ndarray,
        df: pd.DataFrame,
        features_df: pd.DataFrame,
        headlines: Optional[list[str]],
    ) -> tuple[int, float, np.ndarray]:
        """Dispatch prediction to the appropriate model interface."""
        if name == "prophet":
            classes, probs = model.predict(df)  # type: ignore[attr-defined]
        elif name == "sentiment":
            classes, probs = model.predict(headlines)  # type: ignore[attr-defined]
        elif name == "anomaly":
            if len(X_flat) == 0:
                probs = np.full((1, 3), 1 / 3, dtype=np.float32)
                classes = np.ones(1, dtype=int)
            else:
                classes, probs = model.predict(X_flat)  # type: ignore[attr-defined]
        elif name in ("lstm", "transformer", "cnn"):
            if len(X_seq) == 0:
                probs = np.full((1, 3), 1 / 3, dtype=np.float32)
                classes = np.ones(1, dtype=int)
            else:
                classes, probs = model.predict(X_seq)  # type: ignore[attr-defined]
        else:
            if len(X_flat) == 0:
                probs = np.full((1, 3), 1 / 3, dtype=np.float32)
                classes = np.ones(1, dtype=int)
            else:
                classes, probs = model.predict(X_flat)  # type: ignore[attr-defined]

        last_class = int(classes[-1])
        last_conf = float(probs[-1, last_class])
        return last_class, last_conf, probs

    def _build_sequences(self, X_flat: np.ndarray) -> np.ndarray:
        """Build (n, lookback, n_features) sequence array."""
        if len(X_flat) < self._lookback:
            return np.empty((0, self._lookback, X_flat.shape[1]), dtype=np.float32)

        seqs = []
        for i in range(self._lookback, len(X_flat)):
            seqs.append(X_flat[i - self._lookback : i])
        return np.array(seqs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Model access
    # ------------------------------------------------------------------

    @property
    def models(self) -> dict[str, object]:
        """Expose internal model dict (for training/evaluation)."""
        return self._models

    @property
    def feature_engineer(self) -> FeatureEngineer:
        return self._engineer

    @property
    def data_processor(self) -> DataProcessor:
        return self._processor

    @property
    def lookback(self) -> int:
        """Lookback window length used for sequence models."""
        return self._lookback

    @property
    def confidence_threshold(self) -> float:
        """Minimum confidence required to emit a BUY/SELL signal."""
        return self._confidence_threshold
