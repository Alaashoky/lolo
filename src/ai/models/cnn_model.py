"""
CNN Model for candlestick pattern detection.

Architecture:
- 1-D convolutional layers applied to OHLCV time-series windows
- Pooling layers for dimensionality reduction
- Dense classification head (BUY / HOLD / SELL)
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("forex_bot.ai.cnn")

try:
    from tensorflow import keras  # type: ignore

    _TF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TF_AVAILABLE = False
    logger.warning("TensorFlow not installed – CNNModel will run in fallback mode.")


class CNNModel:
    """
    1-D CNN for candlestick pattern classification.

    Falls back to uniform predictions when TensorFlow is unavailable.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or {}
        self.filters: list[int] = cfg.get("filters", [32, 64])
        self.kernel_size: int = cfg.get("kernel_size", 3)
        self.dropout: float = cfg.get("dropout", 0.25)
        self.epochs: int = cfg.get("epochs", 50)
        self.batch_size: int = cfg.get("batch_size", 32)
        self._model: Optional[object] = None
        self._is_trained: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Fit the CNN on sequence data.

        Args:
            X: Shape (n_samples, seq_len, n_features).
            y: Integer labels (0=SELL, 1=HOLD, 2=BUY).
        """
        if not _TF_AVAILABLE:
            logger.warning("CNNModel.train: TensorFlow unavailable – skipping.")
            return {}

        n_features = X.shape[2]
        seq_len = X.shape[1]
        self._model = self._build(seq_len, n_features)
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
        logger.info("CNNModel training complete. Final val_acc=%.4f",
                    history.history.get("val_accuracy", [0])[-1])
        return history.history

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (predicted_classes, probabilities)."""
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
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        if _TF_AVAILABLE and self._is_trained and self._model is not None:
            self._model.save(path + ".keras")
        with open(path + ".meta.pkl", "wb") as fh:
            pickle.dump({"is_trained": self._is_trained}, fh)

    def load(self, path: str) -> None:
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

    def _build(self, seq_len: int, n_features: int):
        """Construct 1-D CNN classifier."""
        inputs = keras.Input(shape=(seq_len, n_features))
        x = inputs
        for f in self.filters:
            x = keras.layers.Conv1D(f, self.kernel_size, activation="relu", padding="same")(x)
            x = keras.layers.MaxPooling1D(pool_size=2)(x)
            x = keras.layers.Dropout(self.dropout)(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(64, activation="relu")(x)
        x = keras.layers.Dropout(self.dropout)(x)
        outputs = keras.layers.Dense(3, activation="softmax")(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model
