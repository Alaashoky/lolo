#!/usr/bin/env python3
"""
prepare_data.py – Download and validate historical data for training.

Usage:
    python scripts/prepare_data.py --pairs EURUSD,GBPUSD,XAUUSD --years 5
    python scripts/prepare_data.py --pairs all --years 3 --timeframe H1
    python scripts/prepare_data.py --force-refresh
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

# Allow running from project root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtesting.data_downloader import DataDownloader
from src.utils.logger import setup_logger

logger = logging.getLogger("forex_bot.scripts.prepare_data")

DEFAULT_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD",
    "XAUUSD", "BTCUSD", "ETHUSD",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and validate historical OHLCV data for model training."
    )
    parser.add_argument(
        "--pairs",
        default="EURUSD,GBPUSD,USDJPY,AUDUSD,XAUUSD",
        help="Comma-separated pair list, or 'all' for all supported pairs.",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Number of years of history to download (default: 5).",
    )
    parser.add_argument(
        "--timeframe",
        default="H1",
        choices=["M1", "M5", "M15", "M30", "H1", "H4", "D1"],
        help="Candle timeframe (default: H1).",
    )
    parser.add_argument(
        "--data-dir",
        default="data/historical",
        help="Directory to save downloaded CSV files.",
    )
    parser.add_argument(
        "--source",
        default="yfinance",
        choices=["yfinance", "ccxt"],
        help="Primary data source.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-download even if cache exists.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logger("forex_bot")
    args = parse_args()

    # Resolve pair list
    if args.pairs.lower() == "all":
        pairs = DEFAULT_PAIRS
    else:
        pairs = [p.strip().upper() for p in args.pairs.split(",")]

    logger.info(
        "Preparing data for %d pair(s): %s  |  years=%d  timeframe=%s",
        len(pairs), ", ".join(pairs), args.years, args.timeframe,
    )

    downloader = DataDownloader(data_dir=args.data_dir, source=args.source)

    data_map = downloader.download_multiple(
        pairs=pairs,
        years=args.years,
        timeframe=args.timeframe,
        force_refresh=args.force_refresh,
    )

    # Validate and print summary
    print("\n" + "=" * 60)
    print(f"{'Pair':<10} {'Rows':>8} {'Start':<14} {'End':<14} {'Issues'}")
    print("=" * 60)

    report = {}
    total_rows = 0

    for pair, df in data_map.items():
        validation = downloader.validate(df, pair=pair)
        report[pair] = validation
        rows = int(validation["rows"])
        total_rows += rows
        start = str(validation.get("start", ""))[:10]
        end = str(validation.get("end", ""))[:10]
        issues = ", ".join(validation.get("issues", [])) or "OK"  # type: ignore[arg-type]
        print(f"{pair:<10} {rows:>8,} {start:<14} {end:<14} {issues}")

    print("=" * 60)
    print(f"{'TOTAL':<10} {total_rows:>8,}")
    print()

    # Save validation report
    os.makedirs(args.data_dir, exist_ok=True)
    report_path = os.path.join(args.data_dir, "validation_report.json")
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    logger.info("Validation report saved to %s", report_path)


if __name__ == "__main__":
    main()
