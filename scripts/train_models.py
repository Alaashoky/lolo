#!/usr/bin/env python3
"""
train_models.py – Train all 8 AI models on multi-year historical data.

Usage:
    python scripts/train_models.py --pairs EURUSD,GBPUSD,XAUUSD --years 5
    python scripts/train_models.py --pairs all --years 3 --models lstm,xgboost
    python scripts/train_models.py --config data/config_training.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtesting.data_downloader import DataDownloader
from src.ai.ensemble import AIEnsemble
from src.ai.meta_learner import MetaLearner
from src.ai.training import TrainingPipeline
from src.utils.logger import setup_logger

logger = logging.getLogger("forex_bot.scripts.train_models")

DEFAULT_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "XAUUSD"]
ALL_PAIRS = DEFAULT_PAIRS + ["BTCUSD", "ETHUSD"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train all AI models on historical OHLCV data."
    )
    parser.add_argument(
        "--pairs",
        default="EURUSD,GBPUSD,XAUUSD",
        help="Comma-separated pair list, or 'all'.",
    )
    parser.add_argument(
        "--years", type=int, default=5,
        help="Years of history to train on (default: 5).",
    )
    parser.add_argument(
        "--timeframe", default="H1",
        choices=["M1", "M5", "M15", "M30", "H1", "H4", "D1"],
        help="Candle timeframe (default: H1).",
    )
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated list of models to train, or 'all'.",
    )
    parser.add_argument(
        "--models-dir", default="results/models",
        help="Output directory for saved model files.",
    )
    parser.add_argument(
        "--data-dir", default="data/historical",
        help="Directory containing or to store cached CSV data.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a JSON config file (data/config_training.json).",
    )
    parser.add_argument(
        "--force-refresh", action="store_true",
        help="Re-download data even if cache exists.",
    )
    return parser.parse_args()


def _load_config(path: Optional[str]) -> dict:
    """Load a JSON config file or return empty dict."""
    if path and os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {}


def main() -> None:
    setup_logger("forex_bot")
    args = parse_args()
    cfg = _load_config(args.config)

    # Resolve pairs
    if args.pairs.lower() == "all":
        pairs = ALL_PAIRS
    else:
        pairs = [p.strip().upper() for p in args.pairs.split(",")]

    # Merge config overrides
    training_cfg = cfg.get("training", {})
    years = training_cfg.get("years", args.years)
    timeframe = training_cfg.get("timeframe", args.timeframe)

    logger.info(
        "Training on %d pairs: %s  |  years=%d  timeframe=%s",
        len(pairs), ", ".join(pairs), years, timeframe,
    )

    # 1. Download data
    downloader = DataDownloader(data_dir=args.data_dir)
    data_map = downloader.download_multiple(
        pairs=pairs,
        years=years,
        timeframe=timeframe,
        force_refresh=args.force_refresh,
    )

    # Build settings dict for AI ensemble
    settings: dict = {"ai": {"models_dir": args.models_dir}}
    if cfg.get("models"):
        settings["ai"]["models"] = cfg["models"]

    # 2. Train each pair
    overall_results: Dict[str, Dict] = {}
    t_start = time.time()

    for pair, df in data_map.items():
        if df.empty:
            logger.warning("Skipping %s – no data available.", pair)
            continue

        logger.info("Training models for %s (%d rows)…", pair, len(df))
        pair_start = time.time()

        try:
            ensemble = AIEnsemble(settings)
            meta = MetaLearner(settings)
            pipeline = TrainingPipeline(ensemble, meta, settings)
            results = pipeline.train_all(df, pair=pair)
            overall_results[pair] = results
        except Exception as exc:
            logger.error("Training failed for %s: %s", pair, exc)
            overall_results[pair] = {"error": str(exc)}
            continue

        elapsed = time.time() - pair_start
        _print_pair_results(pair, results, elapsed)

    total_elapsed = time.time() - t_start

    # 3. Save summary
    os.makedirs("results", exist_ok=True)
    summary_path = "results/training_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(overall_results, fh, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Training complete in {total_elapsed:.1f}s")
    print(f"Summary saved to: {summary_path}")
    print(f"Models saved to:  {args.models_dir}/")
    print(f"{'='*60}\n")


def _print_pair_results(pair: str, results: Dict, elapsed: float) -> None:
    print(f"\n{'='*50}")
    print(f"  {pair}  ({elapsed:.1f}s)")
    print(f"{'='*50}")
    for model, res in results.items():
        if isinstance(res, dict) and "error" in res:
            print(f"  ✗  {model:<20}  ERROR: {res['error']}")
        else:
            print(f"  ✓  {model:<20}  {res}")


if __name__ == "__main__":
    main()
