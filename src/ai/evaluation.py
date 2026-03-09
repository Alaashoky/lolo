"""
Evaluation System for the AI Ensemble.

Metrics:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC (macro one-vs-rest)
- Confusion matrix
- Per-model contribution analysis
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("forex_bot.ai.evaluation")

try:
    from sklearn.metrics import (  # type: ignore
        accuracy_score,
        classification_report,
        confusion_matrix,
        roc_auc_score,
    )
    from sklearn.preprocessing import label_binarize  # type: ignore

    _SKL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKL_AVAILABLE = False


class ModelEvaluator:
    """
    Comprehensive evaluation suite for the AI ensemble and individual models.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_probs: Optional[np.ndarray] = None
    ) -> dict:
        """
        Compute a full set of classification metrics.

        Args:
            y_true: Ground-truth integer labels (0=SELL, 1=HOLD, 2=BUY).
            y_pred: Predicted integer labels.
            y_probs: Optional (n, 3) probability matrix for ROC-AUC.

        Returns:
            Dict with accuracy, precision, recall, f1, roc_auc, confusion_matrix.
        """
        if not _SKL_AVAILABLE:
            logger.warning("ModelEvaluator: scikit-learn unavailable.")
            return {}

        results: dict = {}
        results["accuracy"] = float(accuracy_score(y_true, y_pred))

        report = classification_report(
            y_true, y_pred, target_names=["SELL", "HOLD", "BUY"], output_dict=True, zero_division=0
        )
        results["classification_report"] = report
        results["precision_macro"] = report.get("macro avg", {}).get("precision", 0.0)
        results["recall_macro"] = report.get("macro avg", {}).get("recall", 0.0)
        results["f1_macro"] = report.get("macro avg", {}).get("f1-score", 0.0)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        results["confusion_matrix"] = cm.tolist()

        if y_probs is not None and y_probs.shape[1] == 3:
            try:
                y_bin = label_binarize(y_true, classes=[0, 1, 2])
                results["roc_auc"] = float(
                    roc_auc_score(y_bin, y_probs, multi_class="ovr", average="macro")
                )
            except Exception as exc:
                logger.debug("ROC-AUC computation failed: %s", exc)
                results["roc_auc"] = None

        return results

    def evaluate_ensemble(
        self,
        df_test: pd.DataFrame,
        ensemble,
        labels: np.ndarray,
    ) -> dict:
        """
        Evaluate the full ensemble on a held-out test set.

        Args:
            df_test: OHLCV test DataFrame.
            ensemble: AIEnsemble instance.
            labels: True labels aligned with df_test rows.

        Returns:
            Evaluation metrics dict.
        """
        from src.ai.config import SIGNAL_TO_INT

        predictions = []
        all_probs = []
        lookback = ensemble._lookback

        for i in range(lookback, len(df_test)):
            window = df_test.iloc[i - lookback : i + 1]
            signal, conf, _ = ensemble.predict(window)
            signal_int = SIGNAL_TO_INT.get(signal, 1)
            predictions.append(signal_int)
            # Build dummy 3-class prob row
            probs = [0.0, 0.0, 0.0]
            probs[signal_int] = conf
            remaining = 1.0 - conf
            for j in range(3):
                if j != signal_int:
                    probs[j] = remaining / 2
            all_probs.append(probs)

        if not predictions:
            return {}

        y_pred = np.array(predictions)
        y_true = labels[lookback : lookback + len(predictions)]
        y_probs = np.array(all_probs)

        return self.evaluate(y_true, y_pred, y_probs)

    def compare_models(self, results: dict[str, dict]) -> pd.DataFrame:
        """
        Build a comparison DataFrame from per-model evaluation results.

        Args:
            results: Mapping of model_name → evaluate() output dict.

        Returns:
            DataFrame with one row per model.
        """
        rows = []
        for model_name, metrics in results.items():
            rows.append({
                "model": model_name,
                "accuracy": metrics.get("accuracy", 0.0),
                "precision": metrics.get("precision_macro", 0.0),
                "recall": metrics.get("recall_macro", 0.0),
                "f1": metrics.get("f1_macro", 0.0),
                "roc_auc": metrics.get("roc_auc", None),
            })
        return pd.DataFrame(rows).set_index("model")

    def format_report(self, metrics: dict) -> str:
        """Return a human-readable string summary of evaluation metrics."""
        lines = [
            "=" * 50,
            "  AI Ensemble Evaluation Report",
            "=" * 50,
            f"  Accuracy  : {metrics.get('accuracy', 0.0):.4f}",
            f"  Precision : {metrics.get('precision_macro', 0.0):.4f}",
            f"  Recall    : {metrics.get('recall_macro', 0.0):.4f}",
            f"  F1 (macro): {metrics.get('f1_macro', 0.0):.4f}",
            f"  ROC-AUC   : {metrics.get('roc_auc', 'N/A')}",
            "=" * 50,
        ]
        cm = metrics.get("confusion_matrix")
        if cm:
            lines.append("  Confusion Matrix (SELL / HOLD / BUY):")
            for row in cm:
                lines.append("    " + "  ".join(f"{v:5d}" for v in row))
        return "\n".join(lines)
