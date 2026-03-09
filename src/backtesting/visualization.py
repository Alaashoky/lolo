"""
Visualization module for backtesting results.

Generates the following chart types:
- Equity curve
- Drawdown chart (underwater plot)
- Monthly returns heatmap
- Win/Loss P&L distribution
- Daily P&L histogram
- Model performance comparison bar chart
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("forex_bot.backtesting.visualization")

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.gridspec import GridSpec
    _MPL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MPL_AVAILABLE = False
    logger.warning("matplotlib not installed – charts will be skipped.")


class Visualizer:
    """
    Creates and saves performance charts for a backtest result.

    Args:
        output_dir: Directory where PNG files are saved.
        style:      Matplotlib style (default: ``seaborn-v0_8-darkgrid``).
        dpi:        Resolution of saved images.
    """

    def __init__(
        self,
        output_dir: str = "results/charts",
        style: str = "seaborn-v0_8-darkgrid",
        dpi: int = 120,
    ) -> None:
        self._output_dir = output_dir
        self._style = style
        self._dpi = dpi
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot_all(self, summary: Dict) -> Dict[str, str]:
        """
        Generate all standard charts and save them to *output_dir*.

        Returns:
            Dictionary mapping chart name → file path.
        """
        if not _MPL_AVAILABLE:
            logger.warning("matplotlib not available – skipping all charts.")
            return {}

        paths: Dict[str, str] = {}
        equity_curve = summary.get("equity_curve", [])
        trades = summary.get("trades", [])
        pair = summary.get("pair", "Unknown")

        if equity_curve:
            paths["equity_curve"] = self.plot_equity_curve(equity_curve, pair=pair)
            paths["drawdown"] = self.plot_drawdown(equity_curve, pair=pair)
            paths["monthly_returns"] = self.plot_monthly_returns(equity_curve, pair=pair)

        if trades:
            paths["win_loss_distribution"] = self.plot_win_loss_distribution(trades, pair=pair)
            paths["daily_pnl"] = self.plot_daily_pnl(trades, pair=pair)

        return paths

    def plot_equity_curve(
        self,
        equity_curve: List[Tuple],
        pair: str = "",
        filename: str = "equity_curve.png",
    ) -> str:
        """Equity curve with balance line and buy-and-hold benchmark."""
        with _style_context(self._style):
            fig, ax = plt.subplots(figsize=(14, 5))

            timestamps = [t for t, _ in equity_curve]
            values = [v for _, v in equity_curve]
            ts = pd.to_datetime(timestamps)

            ax.plot(ts, values, linewidth=1.5, color="#2196F3", label="Strategy Equity")
            ax.axhline(values[0], linestyle="--", color="#9E9E9E", linewidth=0.8, label="Initial Balance")

            ax.set_title(f"Equity Curve – {pair}", fontsize=14, fontweight="bold")
            ax.set_xlabel("Date")
            ax.set_ylabel("Account Balance (USD)")
            ax.legend()
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
            fig.tight_layout()

            path = os.path.join(self._output_dir, filename)
            fig.savefig(path, dpi=self._dpi)
            plt.close(fig)
            logger.debug("Saved equity curve to %s", path)
            return path

    def plot_drawdown(
        self,
        equity_curve: List[Tuple],
        pair: str = "",
        filename: str = "drawdown.png",
    ) -> str:
        """Underwater (peak-to-trough) drawdown chart."""
        values = np.array([v for _, v in equity_curve], dtype=float)
        peak = np.maximum.accumulate(values)
        drawdown_pct = (values - peak) / (peak + 1e-10) * 100
        timestamps = pd.to_datetime([t for t, _ in equity_curve])

        with _style_context(self._style):
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.fill_between(timestamps, drawdown_pct, 0, color="#F44336", alpha=0.6)
            ax.plot(timestamps, drawdown_pct, color="#B71C1C", linewidth=0.8)
            ax.set_title(f"Drawdown – {pair}", fontsize=14, fontweight="bold")
            ax.set_xlabel("Date")
            ax.set_ylabel("Drawdown (%)")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
            fig.tight_layout()

            path = os.path.join(self._output_dir, filename)
            fig.savefig(path, dpi=self._dpi)
            plt.close(fig)
            return path

    def plot_monthly_returns(
        self,
        equity_curve: List[Tuple],
        pair: str = "",
        filename: str = "monthly_returns.png",
    ) -> str:
        """Monthly returns heatmap (year × month)."""
        if not equity_curve:
            return ""

        ts = pd.to_datetime([t for t, _ in equity_curve])
        vals = [v for _, v in equity_curve]
        equity_series = pd.Series(vals, index=ts)

        # Resample to month-end
        monthly = equity_series.resample("ME").last()
        monthly_returns = monthly.pct_change() * 100
        monthly_returns = monthly_returns.dropna()

        if monthly_returns.empty:
            return ""

        df_pivot = monthly_returns.to_frame("return")
        df_pivot["year"] = df_pivot.index.year
        df_pivot["month"] = df_pivot.index.month
        heatmap_data = df_pivot.pivot(index="year", columns="month", values="return")

        month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"]

        with _style_context(self._style):
            fig, ax = plt.subplots(figsize=(14, max(4, len(heatmap_data) * 0.7)))

            vmax = max(abs(heatmap_data.values[~np.isnan(heatmap_data.values)].max()),
                       abs(heatmap_data.values[~np.isnan(heatmap_data.values)].min()),
                       1)

            cax = ax.imshow(
                heatmap_data.values,
                cmap="RdYlGn",
                aspect="auto",
                vmin=-vmax,
                vmax=vmax,
            )
            fig.colorbar(cax, ax=ax, format="%.1f%%")

            ax.set_xticks(range(len(heatmap_data.columns)))
            ax.set_xticklabels([month_labels[m - 1] for m in heatmap_data.columns])
            ax.set_yticks(range(len(heatmap_data.index)))
            ax.set_yticklabels(heatmap_data.index)

            # Annotate cells
            for r, year in enumerate(heatmap_data.index):
                for c, month in enumerate(heatmap_data.columns):
                    val = heatmap_data.loc[year, month]
                    if not np.isnan(val):
                        ax.text(c, r, f"{val:.1f}%", ha="center", va="center",
                                fontsize=7, color="black")

            ax.set_title(f"Monthly Returns (%) – {pair}", fontsize=14, fontweight="bold")
            fig.tight_layout()

            path = os.path.join(self._output_dir, filename)
            fig.savefig(path, dpi=self._dpi)
            plt.close(fig)
            return path

    def plot_win_loss_distribution(
        self,
        trades: List[Dict],
        pair: str = "",
        filename: str = "win_loss_distribution.png",
    ) -> str:
        """Histogram of P&L per trade split by wins and losses."""
        pnls = [t.get("profit_loss", 0) for t in trades]
        if not pnls:
            return ""

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        with _style_context(self._style):
            fig, ax = plt.subplots(figsize=(10, 5))
            bins = 30
            if wins:
                ax.hist(wins, bins=bins, color="#4CAF50", alpha=0.7, label=f"Wins ({len(wins)})")
            if losses:
                ax.hist(losses, bins=bins, color="#F44336", alpha=0.7, label=f"Losses ({len(losses)})")
            ax.axvline(0, color="black", linewidth=1)
            ax.set_title(f"Win/Loss Distribution – {pair}", fontsize=14, fontweight="bold")
            ax.set_xlabel("P&L (USD)")
            ax.set_ylabel("Frequency")
            ax.legend()
            fig.tight_layout()

            path = os.path.join(self._output_dir, filename)
            fig.savefig(path, dpi=self._dpi)
            plt.close(fig)
            return path

    def plot_daily_pnl(
        self,
        trades: List[Dict],
        pair: str = "",
        filename: str = "daily_pnl.png",
    ) -> str:
        """Daily P&L bar chart."""
        if not trades:
            return ""

        # Aggregate by exit date
        rows = []
        for t in trades:
            exit_time = t.get("exit_time")
            pnl = t.get("profit_loss", 0)
            if exit_time:
                date = str(exit_time)[:10]
                rows.append({"date": date, "pnl": pnl})

        if not rows:
            return ""

        df = pd.DataFrame(rows).groupby("date")["pnl"].sum().reset_index()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        with _style_context(self._style):
            fig, ax = plt.subplots(figsize=(14, 4))
            colors = ["#4CAF50" if p > 0 else "#F44336" for p in df["pnl"]]
            ax.bar(df["date"], df["pnl"], color=colors, width=0.8)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(f"Daily P&L – {pair}", fontsize=14, fontweight="bold")
            ax.set_xlabel("Date")
            ax.set_ylabel("P&L (USD)")
            fig.tight_layout()

            path = os.path.join(self._output_dir, filename)
            fig.savefig(path, dpi=self._dpi)
            plt.close(fig)
            return path

    def plot_model_comparison(
        self,
        model_metrics: Dict[str, Dict],
        filename: str = "model_comparison.png",
    ) -> str:
        """
        Bar chart comparing win rate and accuracy across models.

        Args:
            model_metrics: Dict mapping model name → dict with keys
                           ``win_rate``, ``accuracy``, etc.
        """
        if not model_metrics:
            return ""

        names = list(model_metrics.keys())
        win_rates = [model_metrics[n].get("win_rate", 0) for n in names]
        accuracies = [model_metrics[n].get("accuracy", 0) for n in names]

        x = np.arange(len(names))
        width = 0.35

        with _style_context(self._style):
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.bar(x - width / 2, win_rates, width, label="Win Rate (%)", color="#2196F3")
            ax.bar(x + width / 2, accuracies, width, label="Accuracy (%)", color="#FF9800")
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=20, ha="right")
            ax.set_ylabel("Percentage (%)")
            ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
            ax.legend()
            fig.tight_layout()

            path = os.path.join(self._output_dir, filename)
            fig.savefig(path, dpi=self._dpi)
            plt.close(fig)
            return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _style_context:
    """Context manager for matplotlib styles with graceful fallback."""

    def __init__(self, style: str) -> None:
        self._style = style

    def __enter__(self) -> None:
        try:
            plt.style.use(self._style)
        except Exception:
            pass  # Use default style if not available

    def __exit__(self, *_: object) -> None:
        plt.rcdefaults()
