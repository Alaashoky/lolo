"""
LSTM Model for price forecasting.

Architecture:
- 2-3 stacked LSTM layers with dropout
- Bidirectional processing
- Simple attention mechanism
- Input: lookback time steps × n_features
- Output: directional signal probability (BUY / HOLD / SELL)
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("forex_bot.ai.lstm")

# ---------------------------------------------------------------------------
# Optional TensorFlow import
# ---------------------------------------------------------------------------
try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore

    _TF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TF_AVAILABLE = False
    logger.warning("TensorFlow not installed – LSTMModel will run in fallback mode.")


class LSTMModel:
    """
    Bidirectional LSTM with attention for trading signal classification.

    When TensorFlow is unavailable the model returns a neutral prediction
    so the ensemble can still function without it.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or {}
        self.units: list[int] = cfg.get("units", [128, 64, 32])
        self.dropout: float = cfg.get("dropout", 0.2)
        self.epochs: int = cfg.get("epochs", 50)
        self.batch_size: int = cfg.get("batch_size", 32)
        self.lookback: int = cfg.get("lookback", 60)
        self._model: Optional[object] = None
        self._is_trained: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Fit the LSTM on sequence data.

        Args:
            X: Shape (n_samples, lookback, n_features).
            y: Integer class labels (0=SELL, 1=HOLD, 2=BUY).

        Returns:
            Training history metrics.
        """
        if not _TF_AVAILABLE:
            logger.warning("LSTMModel.train: TensorFlow unavailable – skipping.")
            return {}

        n_classes = len(np.unique(y))
        n_features = X.shape[2]

        self._model = self._build(n_features, n_classes)
        y_cat = keras.utils.to_categorical(y, num_classes=3)

        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
        ]

        history = self._model.fit(
            X, y_cat,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=0,
        )
        self._is_trained = True
        logger.info("LSTMModel training complete. Final val_acc=%.4f",
                    history.history.get("val_accuracy", [0])[-1])
        return history.history

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict class probabilities.

        Returns:
            Tuple of (predicted_classes, probabilities) each of shape (n,).
        """
        if not _TF_AVAILABLE or not self._is_trained or self._model is None:
            n = len(X)
            probs = np.full((n, 3), 1 / 3, dtype=np.float32)
            return np.ones(n, dtype=int), probs

        probs = self._model.predict(X, verbose=0)
        classes = np.argmax(probs, axis=1)
        return classes, probs

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights and config to *path*."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        if _TF_AVAILABLE and self._is_trained and self._model is not None:
            self._model.save(path + ".keras")
        meta = {"is_trained": self._is_trained, "units": self.units,
                "dropout": self.dropout, "lookback": self.lookback}
        with open(path + ".meta.pkl", "wb") as fh:
            pickle.dump(meta, fh)

    def load(self, path: str) -> None:
        """Load model weights from *path*."""
        meta_path = path + ".meta.pkl"
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as fh:
                meta = pickle.load(fh)
            self._is_trained = meta.get("is_trained", False)
        keras_path = path + ".keras"
        if _TF_AVAILABLE and os.path.exists(keras_path):
            self._model = keras.models.load_model(keras_path)

    # ------------------------------------------------------------------
    # Model builder
    # ------------------------------------------------------------------

    def _build(self, n_features: int, n_classes: int):
        """Construct a bidirectional LSTM with attention."""
        inputs = keras.Input(shape=(self.lookback, n_features))
        x = inputs

        for i, units in enumerate(self.units[:-1]):
            x = keras.layers.Bidirectional(
                keras.layers.LSTM(units, return_sequences=True, dropout=self.dropout)
            )(x)

        # Final LSTM layer
        x = keras.layers.Bidirectional(
            keras.layers.LSTM(self.units[-1], return_sequences=True, dropout=self.dropout)
        )(x)

        # Attention
        attention = keras.layers.Dense(1, activation="tanh")(x)
        attention = keras.layers.Flatten()(attention)
        attention = keras.layers.Activation("softmax")(attention)
        attention = keras.layers.RepeatVector(self.units[-1] * 2)(attention)
        attention = keras.layers.Permute([2, 1])(attention)
        x = keras.layers.Multiply()([x, attention])
        x = keras.layers.Lambda(lambda v: tf.reduce_sum(v, axis=1))(x)

        x = keras.layers.Dense(64, activation="relu")(x)
        x = keras.layers.Dropout(self.dropout)(x)
        outputs = keras.layers.Dense(n_classes, activation="softmax")(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model
