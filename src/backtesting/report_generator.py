"""
Report Generator.

Produces a self-contained HTML performance report from a backtest
summary dictionary using Jinja2 templates.

The report includes:
- Summary statistics table
- Equity curve and drawdown chart images
- Monthly returns heatmap
- Win/Loss distribution
- Trade list (paginated)
- Risk metrics
- Model contribution analysis
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger("forex_bot.backtesting.report_generator")

# ---------------------------------------------------------------------------
# Jinja2 template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Backtest Performance Report – {{ pair }}</title>
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: #eaeaea; margin: 0; padding: 0; }
  .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
  h1 { color: #4fc3f7; border-bottom: 2px solid #4fc3f7; padding-bottom: 8px; }
  h2 { color: #81d4fa; margin-top: 32px; }
  .summary-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 16px; margin: 16px 0; }
  .stat-card { background: #16213e; border-radius: 8px; padding: 16px; text-align: center; border: 1px solid #0f3460; }
  .stat-card .label { font-size: 12px; color: #90caf9; text-transform: uppercase; letter-spacing: 0.5px; }
  .stat-card .value { font-size: 24px; font-weight: bold; margin-top: 8px; }
  .positive { color: #66bb6a; }
  .negative { color: #ef5350; }
  .neutral  { color: #ffa726; }
  img { max-width: 100%; border-radius: 8px; margin: 12px 0; border: 1px solid #0f3460; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }
  th { background: #0f3460; padding: 10px 8px; text-align: left; color: #90caf9; }
  td { padding: 8px; border-bottom: 1px solid #0f3460; }
  tr:hover td { background: #16213e; }
  .win { color: #66bb6a; }
  .loss { color: #ef5350; }
  .footer { margin-top: 40px; text-align: center; color: #616161; font-size: 12px; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }
  .badge-buy  { background: #1b5e20; color: #a5d6a7; }
  .badge-sell { background: #b71c1c; color: #ef9a9a; }
</style>
</head>
<body>
<div class="container">
  <h1>📊 Backtest Performance Report</h1>
  <p style="color:#90caf9">Pair: <strong>{{ pair }}</strong> &nbsp;|&nbsp;
     Generated: <strong>{{ generated_at }}</strong></p>

  <h2>Summary Statistics</h2>
  <div class="summary-grid">
    {% for stat in stats %}
    <div class="stat-card">
      <div class="label">{{ stat.label }}</div>
      <div class="value {{ stat.cls }}">{{ stat.value }}</div>
    </div>
    {% endfor %}
  </div>

  {% if chart_equity %}
  <h2>Equity Curve</h2>
  <img src="{{ chart_equity }}" alt="Equity Curve">
  {% endif %}

  {% if chart_drawdown %}
  <h2>Drawdown</h2>
  <img src="{{ chart_drawdown }}" alt="Drawdown Chart">
  {% endif %}

  {% if chart_monthly %}
  <h2>Monthly Returns Heatmap</h2>
  <img src="{{ chart_monthly }}" alt="Monthly Returns">
  {% endif %}

  {% if chart_pnl_dist %}
  <h2>Win/Loss Distribution</h2>
  <img src="{{ chart_pnl_dist }}" alt="Win/Loss Distribution">
  {% endif %}

  {% if chart_daily %}
  <h2>Daily P&amp;L</h2>
  <img src="{{ chart_daily }}" alt="Daily P&L">
  {% endif %}

  {% if chart_model_cmp %}
  <h2>Model Comparison</h2>
  <img src="{{ chart_model_cmp }}" alt="Model Comparison">
  {% endif %}

  <h2>Trade List <small style="font-size:13px;color:#9e9e9e">(last {{ trades|length }} trades)</small></h2>
  <table>
    <thead>
      <tr>
        <th>#</th><th>Pair</th><th>Direction</th>
        <th>Entry Time</th><th>Entry Price</th>
        <th>Exit Time</th><th>Exit Price</th>
        <th>Lot</th><th>P&amp;L (USD)</th><th>Pips</th><th>Reason</th>
      </tr>
    </thead>
    <tbody>
    {% for t in trades %}
    <tr>
      <td>{{ loop.index }}</td>
      <td>{{ t.pair }}</td>
      <td><span class="badge badge-{{ t.direction|lower }}">{{ t.direction }}</span></td>
      <td>{{ t.entry_time }}</td>
      <td>{{ "%.5f"|format(t.entry_price) }}</td>
      <td>{{ t.exit_time or "–" }}</td>
      <td>{% if t.exit_price %}{{ "%.5f"|format(t.exit_price) }}{% else %}–{% endif %}</td>
      <td>{{ t.lot_size }}</td>
      <td class="{% if t.profit_loss > 0 %}win{% else %}loss{% endif %}">
        {{ "%.2f"|format(t.profit_loss) }}
      </td>
      <td class="{% if t.profit_pips > 0 %}win{% else %}loss{% endif %}">
        {{ "%.1f"|format(t.profit_pips) }}
      </td>
      <td>{{ t.reason }}</td>
    </tr>
    {% endfor %}
    </tbody>
  </table>

  <h2>Risk Metrics</h2>
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>
    {% for row in risk_rows %}
    <tr><td>{{ row.label }}</td><td>{{ row.value }}</td></tr>
    {% endfor %}
    </tbody>
  </table>

  <div class="footer">
    SE Forex Trading Bot – Backtesting Report &copy; {{ year }}
  </div>
</div>
</body>
</html>
"""


class ReportGenerator:
    """
    Generate an HTML backtest performance report.

    Args:
        output_dir:    Directory where the report is saved.
        chart_dir:     Directory containing chart PNG files (used for
                       relative <img> paths in the HTML).
        max_trade_rows: Maximum number of trades to display in the
                        trade table (most recent first, capped).
    """

    def __init__(
        self,
        output_dir: str = "results",
        chart_dir: str = "results/charts",
        max_trade_rows: int = 200,
    ) -> None:
        self._output_dir = output_dir
        self._chart_dir = chart_dir
        self._max_trade_rows = max_trade_rows
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        summary: Dict,
        chart_paths: Optional[Dict[str, str]] = None,
        filename: str = "performance_report.html",
    ) -> str:
        """
        Render and save the HTML report.

        Args:
            summary:     Backtest summary dict (from BacktestEngine /
                         PerformanceMetrics).
            chart_paths: Dict mapping chart name → file path.
            filename:    Output filename inside *output_dir*.

        Returns:
            Absolute path to the generated report.
        """
        chart_paths = chart_paths or {}
        try:
            from jinja2 import Template  # type: ignore
            template = Template(_HTML_TEMPLATE)
        except ImportError:
            logger.warning("jinja2 not installed – generating plain JSON report instead.")
            return self._generate_json(summary, filename.replace(".html", ".json"))

        pair = summary.get("pair", "Unknown")
        stats = self._build_stats(summary)
        risk_rows = self._build_risk_rows(summary)
        trades = summary.get("trades", [])[-self._max_trade_rows:]

        # Make chart paths relative or absolute as appropriate
        def _path(key: str) -> str:
            p = chart_paths.get(key, "")
            if p and os.path.isabs(p):
                # Make relative to output_dir for portability
                try:
                    return os.path.relpath(p, self._output_dir)
                except ValueError:
                    return p
            return p

        html = template.render(
            pair=pair,
            generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            year=datetime.utcnow().year,
            stats=stats,
            trades=trades,
            risk_rows=risk_rows,
            chart_equity=_path("equity_curve"),
            chart_drawdown=_path("drawdown"),
            chart_monthly=_path("monthly_returns"),
            chart_pnl_dist=_path("win_loss_distribution"),
            chart_daily=_path("daily_pnl"),
            chart_model_cmp=_path("model_comparison"),
        )

        out_path = os.path.join(self._output_dir, filename)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(html)

        logger.info("Performance report saved to %s", out_path)
        return out_path

    def save_json(self, summary: Dict, filename: str = "backtest_results.json") -> str:
        """
        Serialise the full backtest summary to JSON.

        Returns:
            Path to the saved JSON file.
        """
        path = os.path.join(self._output_dir, filename)
        # Truncate equity_curve for readability (sample every N points)
        out = dict(summary)
        eq = out.get("equity_curve", [])
        if len(eq) > 2000:
            step = len(eq) // 2000
            out["equity_curve"] = eq[::step]

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=str)

        logger.info("Backtest results saved to %s", path)
        return path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_json(self, summary: Dict, filename: str) -> str:
        return self.save_json(summary, filename)

    def _build_stats(self, s: Dict) -> List[Dict]:
        """Build stat-card data for the HTML summary grid."""
        def _pct_cls(v: Optional[float]) -> str:
            if v is None:
                return "neutral"
            return "positive" if v >= 0 else "negative"

        total_ret = s.get("total_return_pct")
        cagr = s.get("cagr_pct") or s.get("annualised_return_pct")
        wr = s.get("win_rate")
        pf = s.get("profit_factor")
        sharpe = s.get("sharpe_ratio")
        sortino = s.get("sortino_ratio")
        max_dd = s.get("max_drawdown_pct")
        n_trades = s.get("total_trades", 0)

        def fmt(v: object, suffix: str = "") -> str:
            if v is None:
                return "N/A"
            if isinstance(v, float):
                return f"{v:.2f}{suffix}"
            return f"{v}{suffix}"

        return [
            {"label": "Total Return",     "value": fmt(total_ret, "%"), "cls": _pct_cls(total_ret)},
            {"label": "CAGR",             "value": fmt(cagr, "%"),      "cls": _pct_cls(cagr)},
            {"label": "Win Rate",         "value": fmt(wr, "%"),        "cls": "positive" if (wr or 0) >= 50 else "negative"},
            {"label": "Profit Factor",    "value": fmt(pf),             "cls": "positive" if (pf or 0) >= 1 else "negative"},
            {"label": "Sharpe Ratio",     "value": fmt(sharpe),         "cls": "positive" if (sharpe or 0) >= 1 else "neutral"},
            {"label": "Sortino Ratio",    "value": fmt(sortino),        "cls": "positive" if (sortino or 0) >= 1 else "neutral"},
            {"label": "Max Drawdown",     "value": fmt(max_dd, "%"),    "cls": "negative" if (max_dd or 0) > 5 else "neutral"},
            {"label": "Total Trades",     "value": str(n_trades),       "cls": "neutral"},
        ]

    def _build_risk_rows(self, s: Dict) -> List[Dict]:
        """Build rows for the risk metrics table."""
        keys = [
            ("Initial Balance",       "initial_balance",       "${:.2f}"),
            ("Final Balance",         "final_balance",         "${:.2f}"),
            ("Total P&L",             "total_pnl",             "${:.2f}"),
            ("Total Return",          "total_return_pct",      "{:.2f}%"),
            ("CAGR",                  "cagr_pct",              "{:.2f}%"),
            ("Win Rate",              "win_rate",              "{:.2f}%"),
            ("Profit Factor",         "profit_factor",         "{:.4f}"),
            ("Sharpe Ratio",          "sharpe_ratio",          "{:.4f}"),
            ("Sortino Ratio",         "sortino_ratio",         "{:.4f}"),
            ("Calmar Ratio",          "calmar_ratio",          "{:.4f}"),
            ("Max Drawdown",          "max_drawdown",          "${:.2f}"),
            ("Max Drawdown %",        "max_drawdown_pct",      "{:.2f}%"),
            ("Recovery Factor",       "recovery_factor",       "{:.4f}"),
            ("Avg Win",               "avg_win",               "${:.2f}"),
            ("Avg Loss",              "avg_loss",              "${:.2f}"),
            ("Risk-Reward Ratio",     "risk_reward_ratio",     "{:.2f}"),
            ("Consecutive Wins",      "max_consecutive_wins",  "{}"),
            ("Consecutive Losses",    "max_consecutive_losses","{}"),
        ]
        rows = []
        for label, key, fmt in keys:
            val = s.get(key)
            if val is not None:
                try:
                    rows.append({"label": label, "value": fmt.format(val)})
                except (TypeError, ValueError):
                    rows.append({"label": label, "value": str(val)})
        return rows
