"""
Backtesting package for the SE Forex Trading Bot.

Provides tools for:
- Downloading historical OHLCV data
- Running backtests on historical data
- Computing performance metrics
- Generating visualizations and HTML reports
"""

from src.backtesting.data_downloader import DataDownloader
from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.performance_metrics import PerformanceMetrics
from src.backtesting.visualization import Visualizer
from src.backtesting.report_generator import ReportGenerator

__all__ = [
    "DataDownloader",
    "BacktestEngine",
    "PerformanceMetrics",
    "Visualizer",
    "ReportGenerator",
]
