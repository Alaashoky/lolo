#!/usr/bin/env python3
"""
evaluate_performance.py – Compare all 8 AI models and the ensemble.

Loads pre-trained models and evaluates them on held-out test data,
printing a comparison table and saving a model_comparison chart.

Usage:
    python scripts/evaluate_performance.py
    python scripts/evaluate_performance.py --pairs EURUSD,GBPUSD --years 2
    python scripts/evaluate_performance.py --models-dir results/models
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtesting.data_downloader import DataDownloader
from src.backtesting.visualization import Visualizer
from src.ai.ensemble import AIEnsemble
from src.ai.data_processor import DataProcessor
from src.ai.feature_engineering import FeatureEngineer
from src.ai.training import TrainingPipeline
from src.ai.meta_learner import MetaLearner
from src.utils.logger import setup_logger

import numpy as np
import pandas as pd

logger = logging.getLogger("forex_bot.scripts.evaluate_performance")

MODEL_NAMES = [
    "lstm", "transformer", "xgboost", "random_forest",
    "cnn", "prophet", "sentiment", "anomaly",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and compare all AI model performances."
    )
    parser.add_argument(
        "--pairs", default="EURUSD,GBPUSD",
        help="Comma-separated pair list.",
    )
    parser.add_argument(
        "--years", type=int, default=2,
        help="Years of data to evaluate on (default: 2).",
    )
    parser.add_argument(
        "--timeframe", default="H1",
        choices=["M1", "M5", "M15", "M30", "H1", "H4", "D1"],
    )
    parser.add_argument(
        "--models-dir", default="results/models",
        help="Directory containing trained model files.",
    )
    parser.add_argument(
        "--data-dir", default="data/historical",
        help="Directory for cached OHLCV CSV files.",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Output directory for evaluation results.",
    )
    return parser.parse_args()


def _make_labels(df: pd.DataFrame, threshold: float = 0.0005) -> np.ndarray:
    """Directional labels: 0=SELL, 1=HOLD, 2=BUY."""
    close = df["close"].values
    returns = np.diff(close) / (close[:-1] + 1e-10)
    return np.where(returns > threshold, 2,
           np.where(returns < -threshold, 0, 1)).astype(int)


def _evaluate_model(model, X_flat, X_seq, y, model_name: str) -> Dict:
    """Run prediction and compute accuracy for a single model."""
    try:
        seq_models = {"lstm", "transformer", "cnn"}
        if model_name in seq_models:
            if len(X_seq) == 0:
                return {"accuracy": 0.0, "win_rate": 0.0, "samples": 0}
            preds, _ = model.predict(X_seq)
            y_eval = y[len(y) - len(preds):]
        else:
            if len(X_flat) == 0:
                return {"accuracy": 0.0, "win_rate": 0.0, "samples": 0}
            preds, _ = model.predict(X_flat)
            y_eval = y[len(y) - len(preds):]

        min_len = min(len(preds), len(y_eval))
        preds = preds[:min_len]
        y_eval = y_eval[:min_len]

        accuracy = float(np.mean(preds == y_eval)) * 100
        # Win rate: fraction of BUY/SELL signals that match correct direction
        direction_mask = (y_eval != 1)
        direction_acc = float(np.mean(preds[direction_mask] == y_eval[direction_mask])) * 100 \
            if direction_mask.sum() > 0 else 0.0

        return {
            "accuracy": round(accuracy, 2),
            "win_rate": round(direction_acc, 2),
            "samples": min_len,
        }
    except Exception as exc:
        logger.debug("Evaluation error for %s: %s", model_name, exc)
        return {"accuracy": 0.0, "win_rate": 0.0, "samples": 0, "error": str(exc)}


def main() -> None:
    setup_logger("forex_bot")
    args = parse_args()

    pairs = [p.strip().upper() for p in args.pairs.split(",")]

    downloader = DataDownloader(data_dir=args.data_dir)
    data_map = downloader.download_multiple(
        pairs=pairs, years=args.years, timeframe=args.timeframe,
    )

    settings: dict = {"ai": {"models_dir": args.models_dir}}
    visualizer = Visualizer(output_dir=os.path.join(args.output_dir, "charts"))

    # Aggregate metrics across pairs
    aggregated: Dict[str, List] = {name: [] for name in MODEL_NAMES}
    aggregated["ensemble"] = []

    for pair, df in data_map.items():
        if df.empty:
            logger.warning("No data for %s – skipping.", pair)
            continue

        logger.info("Evaluating %s (%d rows)…", pair, len(df))

        ensemble = AIEnsemble(settings)
        meta = MetaLearner(settings)
        pipeline = TrainingPipeline(ensemble, meta, settings)
        pipeline.load_all(pair)

        # Prepare test data (last 15%)
        processor = ensemble.data_processor
        engineer = ensemble.feature_engineer

        df_clean = processor.process(df)
        _, _, test_df = processor.split(df_clean)

        if len(test_df) < 60:
            logger.warning("Insufficient test data for %s.", pair)
            continue

        test_feat = engineer.extract(test_df)
        feature_cols = [c for c in engineer.feature_names() if c in test_feat.columns]
        X_flat = test_feat[feature_cols].values.astype(np.float32)
        y = _make_labels(test_feat)

        # Build sequence tensor
        lookback = 60
        if len(X_flat) > lookback:
            seqs = [X_flat[i - lookback:i] for i in range(lookback, len(X_flat))]
            X_seq = np.array(seqs, dtype=np.float32)
        else:
            X_seq = np.empty((0, lookback, X_flat.shape[1]), dtype=np.float32)

        # Evaluate each model
        for name in MODEL_NAMES:
            model = ensemble.models.get(name)
            if model is None:
                continue
            result = _evaluate_model(model, X_flat, X_seq, y, name)
            result["pair"] = pair
            aggregated[name].append(result)

        # Evaluate ensemble (via predict on windowed data)
        ens_preds, ens_labels = [], []
        for i in range(lookback, min(len(test_df), lookback + 500)):  # cap for speed
            window = test_df.iloc[max(0, i - lookback):i + 1]
            try:
                signal, conf, _ = ensemble.predict(window)
                label = int(y[i]) if i < len(y) else 1
                pred_int = {"BUY": 2, "HOLD": 1, "SELL": 0}.get(signal, 1)
                ens_preds.append(pred_int)
                ens_labels.append(label)
            except Exception:
                pass

        if ens_preds:
            ens_acc = float(np.mean(np.array(ens_preds) == np.array(ens_labels))) * 100
            aggregated["ensemble"].append({"accuracy": round(ens_acc, 2), "pair": pair})

    # Compute mean metrics per model
    summary: Dict[str, Dict] = {}
    for name, results in aggregated.items():
        if results:
            acc = float(np.mean([r["accuracy"] for r in results]))
            wr = float(np.mean([r.get("win_rate", 0) for r in results]))
            summary[name] = {"accuracy": round(acc, 2), "win_rate": round(wr, 2)}
        else:
            summary[name] = {"accuracy": 0.0, "win_rate": 0.0}

    # Print comparison table
    _print_table(summary)

    # Save JSON
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "model_evaluation.json")
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Evaluation results saved to %s", json_path)

    # Generate comparison chart
    chart_path = visualizer.plot_model_comparison(summary)
    if chart_path:
        print(f"\nModel comparison chart: {chart_path}")


def _print_table(summary: Dict[str, Dict]) -> None:
    print(f"\n{'='*55}")
    print(f"  {'Model':<22} {'Accuracy':>10} {'Win Rate':>10}")
    print(f"{'='*55}")
    for name, m in sorted(summary.items(), key=lambda x: -x[1].get("accuracy", 0)):
        acc = m.get("accuracy", 0)
        wr = m.get("win_rate", 0)
        print(f"  {name:<22} {acc:>9.2f}% {wr:>9.2f}%")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
