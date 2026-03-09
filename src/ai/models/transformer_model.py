"""
Transformer Model for sequential pattern recognition.

Architecture:
- Positional encoding
- Multi-head self-attention blocks
- Feed-forward sub-layers
- Layer normalisation & dropout
- Classification head (BUY / HOLD / SELL)
"""

from __future__ import annotations

import logging
import math
import os
import pickle
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("forex_bot.ai.transformer")

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.optim as optim  # type: ignore
    from torch.utils.data import DataLoader, TensorDataset  # type: ignore

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed – TransformerModel will run in fallback mode.")


# ---------------------------------------------------------------------------
# PyTorch model definition
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class _PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = x + self.pe[:, : x.size(1)]
            return self.dropout(x)

    class _TransformerClassifier(nn.Module):
        def __init__(
            self,
            n_features: int,
            d_model: int = 64,
            num_heads: int = 4,
            num_layers: int = 2,
            n_classes: int = 3,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.input_proj = nn.Linear(n_features, d_model)
            self.pos_enc = _PositionalEncoding(d_model, dropout=dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.classifier = nn.Sequential(
                nn.Linear(d_model, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, n_classes),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.input_proj(x)
            x = self.pos_enc(x)
            x = self.encoder(x)
            x = x.mean(dim=1)
            return self.classifier(x)


class TransformerModel:
    """
    PyTorch Transformer for trading signal classification.

    Falls back to uniform predictions when PyTorch is not available.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or {}
        self.d_model: int = cfg.get("d_model", 64)
        self.num_heads: int = cfg.get("num_heads", 4)
        self.num_layers: int = cfg.get("num_layers", 2)
        self.dropout: float = cfg.get("dropout", 0.1)
        self.epochs: int = cfg.get("epochs", 50)
        self.batch_size: int = cfg.get("batch_size", 32)
        self._model: Optional[object] = None
        self._is_trained: bool = False
        self._device = "cpu"
        if _TORCH_AVAILABLE:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Fit the transformer on sequence data.

        Args:
            X: Shape (n_samples, seq_len, n_features).
            y: Integer class labels (0=SELL, 1=HOLD, 2=BUY).
        """
        if not _TORCH_AVAILABLE:
            logger.warning("TransformerModel.train: PyTorch unavailable – skipping.")
            return {}

        n_features = X.shape[2]
        self._model = _TransformerClassifier(
            n_features=n_features,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self._device)

        X_t = torch.tensor(X, dtype=torch.float32).to(self._device)
        y_t = torch.tensor(y, dtype=torch.long).to(self._device)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self._model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

        history: dict = {"loss": [], "accuracy": []}
        for epoch in range(self.epochs):
            self._model.train()
            total_loss, correct, total = 0.0, 0, 0
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = self._model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(xb)
                correct += (preds.argmax(1) == yb).sum().item()
                total += len(xb)
            epoch_loss = total_loss / total
            epoch_acc = correct / total
            history["loss"].append(epoch_loss)
            history["accuracy"].append(epoch_acc)
            scheduler.step(epoch_loss)

        self._is_trained = True
        logger.info("TransformerModel training complete. Final acc=%.4f", history["accuracy"][-1])
        return history

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (predicted_classes, probabilities)."""
        if not _TORCH_AVAILABLE or not self._is_trained or self._model is None:
            n = len(X)
            probs = np.full((n, 3), 1 / 3, dtype=np.float32)
            return np.ones(n, dtype=int), probs

        self._model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self._device)
            logits = self._model(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        classes = np.argmax(probs, axis=1)
        return classes, probs

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        if _TORCH_AVAILABLE and self._is_trained and self._model is not None:
            torch.save(self._model.state_dict(), path + ".pt")
        meta = {
            "is_trained": self._is_trained,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
        }
        with open(path + ".meta.pkl", "wb") as fh:
            pickle.dump(meta, fh)

    def load(self, path: str) -> None:
        meta_path = path + ".meta.pkl"
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as fh:
                meta = pickle.load(fh)
            self._is_trained = meta.get("is_trained", False)
