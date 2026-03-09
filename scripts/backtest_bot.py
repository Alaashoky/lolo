#!/usr/bin/env python3
"""
backtest_bot.py – Simulate the full bot on historical data.

Runs the AI ensemble over 5 years of OHLCV data, tracks all trades,
computes performance metrics, and generates an HTML report.

Usage:
    python scripts/backtest_bot.py --pairs EURUSD,GBPUSD --years 5
    python scripts/backtest_bot.py --pairs all --years 3 --balance 10000
    python scripts/backtest_bot.py --config data/config_training.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtesting.data_downloader import DataDownloader
from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.performance_metrics import PerformanceMetrics
from src.backtesting.visualization import Visualizer
from src.backtesting.report_generator import ReportGenerator
from src.ai.ensemble import AIEnsemble
from src.ai.data_processor import DataProcessor
from src.ai.feature_engineering import FeatureEngineer
from src.utils.logger import setup_logger

import pandas as pd
import numpy as np

logger = logging.getLogger("forex_bot.scripts.backtest_bot")

DEFAULT_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "XAUUSD"]

# Pip multipliers for common pairs
_PIP_MULTIPLIERS: Dict[str, float] = {
    "USDJPY": 0.01,
    "XAUUSD": 0.1,
    "BTCUSD": 1.0,
    "ETHUSD": 0.1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest the full AI trading bot on historical data."
    )
    parser.add_argument(
        "--pairs", default="EURUSD,GBPUSD",
        help="Comma-separated pair list, or 'all'.",
    )
    parser.add_argument(
        "--years", type=int, default=5,
        help="Years of history to backtest (default: 5).",
    )
    parser.add_argument(
        "--timeframe", default="H1",
        choices=["M1", "M5", "M15", "M30", "H1", "H4", "D1"],
    )
    parser.add_argument(
        "--balance", type=float, default=10_000.0,
        help="Starting account balance in USD (default: 10000).",
    )
    parser.add_argument(
        "--risk", type=float, default=2.0,
        help="Risk per trade as percentage of balance (default: 2.0).",
    )
    parser.add_argument(
        "--slippage", type=float, default=3.0,
        help="Slippage in pips (default: 3).",
    )
    parser.add_argument(
        "--spread", type=float, default=2.0,
        help="Spread in pips (default: 2).",
    )
    parser.add_argument(
        "--models-dir", default="results/models",
        help="Directory of trained model files.",
    )
    parser.add_argument(
        "--data-dir", default="data/historical",
        help="Directory for cached OHLCV CSV files.",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Output directory for reports and charts.",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to a JSON config file.",
    )
    parser.add_argument(
        "--walk-forward", action="store_true",
        help="Run walk-forward validation instead of a single backtest.",
    )
    parser.add_argument(
        "--force-refresh", action="store_true",
        help="Re-download data even if cache exists.",
    )
    return parser.parse_args()


def _load_config(path) -> dict:
    if path and os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {}


def _generate_signals(df: pd.DataFrame, ensemble: AIEnsemble) -> pd.Series:
    """
    Generate BUY/SELL/HOLD signals for every candle in *df*.

    For efficiency the ensemble is called on rolling windows.
    Falls back to a simple moving-average crossover when the AI
    ensemble is not trained.
    """
    signals = []

    # Try AI ensemble first
    lookback = ensemble.lookback
    data_processor = ensemble.data_processor
    feature_engineer = ensemble.feature_engineer

    use_ai = False
    # Check if any base model is trained
    for model in ensemble.models.values():
        if getattr(model, "_is_trained", False):
            use_ai = True
            break

    if use_ai:
        logger.info("Generating signals with AI ensemble…")
        for i in range(len(df)):
            if i < lookback:
                signals.append("HOLD")
                continue
            window = df.iloc[i - lookback : i + 1]
            try:
                signal, confidence, _ = ensemble.predict(window)
                signals.append(signal if confidence >= ensemble.confidence_threshold else "HOLD")
            except Exception:
                signals.append("HOLD")
    else:
        logger.info("AI models not trained – using MA crossover fallback signals.")
        close = df["close"]
        ma_fast = close.rolling(20).mean()
        ma_slow = close.rolling(50).mean()
        for i in range(len(df)):
            if pd.isna(ma_fast.iloc[i]) or pd.isna(ma_slow.iloc[i]):
                signals.append("HOLD")
            elif ma_fast.iloc[i] > ma_slow.iloc[i]:
                signals.append("BUY")
            elif ma_fast.iloc[i] < ma_slow.iloc[i]:
                signals.append("SELL")
            else:
                signals.append("HOLD")

    return pd.Series(signals, index=df.index)


def main() -> None:
    setup_logger("forex_bot")
    args = parse_args()
    cfg = _load_config(args.config)

    # Resolve pairs
    bt_cfg = cfg.get("backtesting", {})
    if args.pairs.lower() == "all":
        pairs = DEFAULT_PAIRS
    else:
        pairs = [p.strip().upper() for p in args.pairs.split(",")]

    balance = bt_cfg.get("initial_balance", args.balance)
    risk = bt_cfg.get("risk_per_trade", args.risk)
    slippage = bt_cfg.get("slippage_pips", args.slippage)
    spread = bt_cfg.get("spread_pips", args.spread)
    commission = bt_cfg.get("commission_percent", 0.01)

    logger.info(
        "Backtesting %d pairs: %s  |  balance=$%.0f  years=%d",
        len(pairs), ", ".join(pairs), balance, args.years,
    )

    # Components
    downloader = DataDownloader(data_dir=args.data_dir)
    engine = BacktestEngine(
        initial_balance=balance,
        risk_per_trade=risk,
        slippage_pips=slippage,
        spread_pips=spread,
        commission_percent=commission,
    )
    metrics_calc = PerformanceMetrics()
    visualizer = Visualizer(output_dir=os.path.join(args.output_dir, "charts"))
    reporter = ReportGenerator(
        output_dir=args.output_dir,
        chart_dir=os.path.join(args.output_dir, "charts"),
    )

    # Build AI settings
    settings: dict = {"ai": {"models_dir": args.models_dir}}
    ensemble = AIEnsemble(settings)

    # Try to load pre-trained models
    from src.ai.meta_learner import MetaLearner
    from src.ai.training import TrainingPipeline
    meta = MetaLearner(settings)
    pipeline = TrainingPipeline(ensemble, meta, settings)

    # Aggregate results across all pairs
    all_summaries: Dict[str, Dict] = {}
    combined_trades: List[Dict] = []
    t_start = time.time()

    for pair in pairs:
        logger.info("Backtesting %s…", pair)

        # Download data
        data_map = downloader.download_multiple(
            [pair], years=args.years, timeframe=args.timeframe,
            force_refresh=args.force_refresh,
        )
        df = data_map.get(pair, pd.DataFrame())

        if df.empty:
            logger.warning("No data for %s – skipping.", pair)
            continue

        # Try to load trained models for this pair
        try:
            pipeline.load_all(pair)
        except Exception:
            pass  # models not trained yet, use fallback signals

        # Generate signals
        pip_mult = _PIP_MULTIPLIERS.get(pair, 0.0001)
        signals = _generate_signals(df, ensemble)

        # Run backtest
        if args.walk_forward:
            fold_results = engine.run_walkforward(
                df, signals, pair=pair, pip_multiplier=pip_mult
            )
            summary = fold_results[-1]  # use last fold as representative
        else:
            engine.reset()
            summary = engine.run(df, signals, pair=pair, pip_multiplier=pip_mult)

        # Enrich with performance metrics
        summary = metrics_calc.compute(summary)
        all_summaries[pair] = summary
        combined_trades.extend(summary.get("trades", []))

        _print_summary(pair, summary)

    # Build combined summary
    combined = _merge_summaries(all_summaries, balance)
    combined["trades"] = combined_trades

    # Save JSON
    json_path = reporter.save_json(combined, "backtest_results.json")

    # Visualise
    chart_paths = visualizer.plot_all(combined)

    # Generate HTML report
    report_path = reporter.generate(combined, chart_paths=chart_paths)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Backtesting complete in {elapsed:.1f}s")
    print(f"Pairs tested:  {', '.join(pairs)}")
    print(f"JSON results:  {json_path}")
    print(f"HTML report:   {report_path}")
    print(f"Charts:        {os.path.join(args.output_dir, 'charts')}/")
    print(f"{'='*60}\n")


def _print_summary(pair: str, s: Dict) -> None:
    print(f"\n  {pair}")
    print(f"  Return:       {s.get('total_return_pct', 0):+.2f}%")
    print(f"  Trades:       {s.get('total_trades', 0)}")
    print(f"  Win Rate:     {s.get('win_rate', 0):.1f}%")
    print(f"  Profit Factor:{s.get('profit_factor') or 'N/A'}")
    print(f"  Sharpe:       {s.get('sharpe_ratio', 0):.2f}")
    print(f"  Max DD:       {s.get('max_drawdown_pct', 0):.2f}%")


def _merge_summaries(summaries: Dict[str, Dict], initial_balance: float) -> Dict:
    """Aggregate per-pair summaries into a combined portfolio summary."""
    if not summaries:
        return {"pair": "Portfolio", "initial_balance": initial_balance,
                "final_balance": initial_balance, "trades": []}

    total_pnl = sum(s.get("total_pnl", 0) for s in summaries.values())
    total_trades = sum(s.get("total_trades", 0) for s in summaries.values())
    winning = sum(s.get("winning_trades", 0) for s in summaries.values())
    losing = sum(s.get("losing_trades", 0) for s in summaries.values())
    final = initial_balance + total_pnl

    return {
        "pair": "Portfolio",
        "initial_balance": initial_balance,
        "final_balance": final,
        "total_pnl": total_pnl,
        "total_return_pct": (final - initial_balance) / initial_balance * 100,
        "total_trades": total_trades,
        "winning_trades": winning,
        "losing_trades": losing,
        "win_rate": winning / total_trades * 100 if total_trades > 0 else 0,
        "pairs": list(summaries.keys()),
        "per_pair": {k: {
            "total_return_pct": v.get("total_return_pct"),
            "total_trades": v.get("total_trades"),
            "win_rate": v.get("win_rate"),
            "sharpe_ratio": v.get("sharpe_ratio"),
            "max_drawdown_pct": v.get("max_drawdown_pct"),
        } for k, v in summaries.items()},
    }


if __name__ == "__main__":
    main()
