"""
Dark Mode Dashboard.

Tkinter-based GUI that displays real-time prices, open positions,
trade history, and performance metrics with a dark colour scheme.
"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from tkinter import ttk
from datetime import datetime
from typing import Optional

logger = logging.getLogger("forex_bot.dashboard")

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

DARK_BG = "#1e1e2e"
PANEL_BG = "#2a2a3e"
HEADER_BG = "#313153"
ACCENT = "#7c6af7"
TEXT_FG = "#cdd6f4"
TEXT_DIM = "#6c7086"
GREEN = "#a6e3a1"
RED = "#f38ba8"
YELLOW = "#f9e2af"
FONT_MONO = ("Courier New", 10)
FONT_LABEL = ("Segoe UI", 10)
FONT_HEADER = ("Segoe UI", 12, "bold")
FONT_TITLE = ("Segoe UI", 14, "bold")


class Dashboard:
    """
    Dark mode trading dashboard built with Tkinter.

    Run ``dashboard.start()`` in the main thread.  The bot can push
    updates from a background thread via the ``update_*`` methods which
    are thread-safe (they schedule work via ``after()``).
    """

    def __init__(self, title: str = "SE Forex Trading Bot") -> None:
        self._title = title
        self._root: Optional[tk.Tk] = None
        self._running = False

        # Data stores (updated by the bot thread)
        self._prices: dict[str, float] = {}
        self._positions: list[dict] = []
        self._history: list[dict] = []
        self._metrics: dict = {}

        # Tkinter widget refs filled during _build_ui()
        self._price_vars: dict[str, tk.StringVar] = {}
        self._positions_tree: Optional[ttk.Treeview] = None
        self._history_tree: Optional[ttk.Treeview] = None
        self._metrics_labels: dict[str, tk.StringVar] = {}

    # ------------------------------------------------------------------
    # Public API (called from bot thread)
    # ------------------------------------------------------------------

    def update_prices(self, prices: dict[str, float]) -> None:
        """Thread-safe price update."""
        self._prices.update(prices)
        if self._root:
            self._root.after(0, self._refresh_prices)

    def update_positions(self, positions: list[dict]) -> None:
        """Thread-safe positions update."""
        self._positions = list(positions)
        if self._root:
            self._root.after(0, self._refresh_positions)

    def update_history(self, history: list[dict]) -> None:
        """Thread-safe trade history update."""
        self._history = list(history)
        if self._root:
            self._root.after(0, self._refresh_history)

    def update_metrics(self, metrics: dict) -> None:
        """Thread-safe performance metrics update."""
        self._metrics.update(metrics)
        if self._root:
            self._root.after(0, self._refresh_metrics)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Build the GUI and enter the Tkinter main loop.

        This method **blocks** until the window is closed.
        Call it from the main thread; push data from background threads
        using the ``update_*`` methods.
        """
        self._root = tk.Tk()
        self._root.title(self._title)
        self._root.configure(bg=DARK_BG)
        self._root.geometry("1200x750")
        self._root.minsize(900, 600)

        self._style_ttk()
        self._build_ui()
        self._running = True

        logger.info("Dashboard started.")
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._root.mainloop()

    def stop(self) -> None:
        """Signal the dashboard to close."""
        if self._root:
            self._root.after(0, self._on_close)

    # ------------------------------------------------------------------
    # TTK style
    # ------------------------------------------------------------------

    def _style_ttk(self) -> None:
        style = ttk.Style(self._root)
        style.theme_use("clam")

        style.configure(".", background=DARK_BG, foreground=TEXT_FG, font=FONT_LABEL)
        style.configure("TFrame", background=DARK_BG)
        style.configure("Panel.TFrame", background=PANEL_BG)
        style.configure(
            "Treeview",
            background=PANEL_BG,
            foreground=TEXT_FG,
            fieldbackground=PANEL_BG,
            rowheight=22,
        )
        style.configure(
            "Treeview.Heading",
            background=HEADER_BG,
            foreground=TEXT_FG,
            font=FONT_LABEL,
        )
        style.map("Treeview", background=[("selected", ACCENT)])
        style.configure(
            "TLabel",
            background=DARK_BG,
            foreground=TEXT_FG,
            font=FONT_LABEL,
        )
        style.configure(
            "Header.TLabel",
            background=DARK_BG,
            foreground=ACCENT,
            font=FONT_HEADER,
        )
        style.configure(
            "Title.TLabel",
            background=DARK_BG,
            foreground=TEXT_FG,
            font=FONT_TITLE,
        )
        style.configure(
            "TNotebook",
            background=DARK_BG,
            tabmargins=[2, 5, 2, 0],
        )
        style.configure(
            "TNotebook.Tab",
            background=PANEL_BG,
            foreground=TEXT_DIM,
            padding=[10, 4],
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", HEADER_BG)],
            foreground=[("selected", TEXT_FG)],
        )

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = self._root

        # Title bar
        title_bar = ttk.Frame(root)
        title_bar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(8, 4))
        ttk.Label(title_bar, text="⚡ SE Forex Trading Bot", style="Title.TLabel").pack(side=tk.LEFT)
        self._clock_var = tk.StringVar(value="")
        ttk.Label(title_bar, textvariable=self._clock_var, style="Header.TLabel").pack(side=tk.RIGHT)
        self._tick_clock()

        # Separator
        sep = tk.Frame(root, height=1, bg=ACCENT)
        sep.pack(fill=tk.X, padx=10, pady=2)

        # Main content area
        content = ttk.Frame(root)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        # Left column: prices + metrics
        left = ttk.Frame(content)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        self._build_prices_panel(left)
        self._build_metrics_panel(left)

        # Right column: tabbed positions + history
        right = ttk.Frame(content)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_tabs(right)

    def _build_prices_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.Frame(parent, style="Panel.TFrame")
        frame.pack(fill=tk.X, pady=(0, 8))
        frame.configure(padding=8)

        ttk.Label(frame, text="Live Prices", style="Header.TLabel").pack(anchor=tk.W)

        pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "XAUUSD"]
        for pair in pairs:
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=f"{pair}:", width=9, font=FONT_MONO, foreground=YELLOW).pack(side=tk.LEFT)
            var = tk.StringVar(value="-.-----")
            self._price_vars[pair] = var
            ttk.Label(row, textvariable=var, font=FONT_MONO, foreground=TEXT_FG, width=10).pack(side=tk.LEFT)

    def _build_metrics_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.Frame(parent, style="Panel.TFrame")
        frame.pack(fill=tk.X)
        frame.configure(padding=8)

        ttk.Label(frame, text="Performance", style="Header.TLabel").pack(anchor=tk.W)

        metrics_def = [
            ("win_rate", "Win Rate"),
            ("total_pnl", "Total P&L"),
            ("total_trades", "Total Trades"),
            ("open_positions", "Open Positions"),
            ("daily_pnl", "Daily P&L"),
        ]
        for key, label in metrics_def:
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=f"{label}:", width=16).pack(side=tk.LEFT)
            var = tk.StringVar(value="—")
            self._metrics_labels[key] = var
            ttk.Label(row, textvariable=var, font=FONT_MONO, foreground=ACCENT, width=12).pack(side=tk.LEFT)

    def _build_tabs(self, parent: ttk.Frame) -> None:
        nb = ttk.Notebook(parent)
        nb.pack(fill=tk.BOTH, expand=True)

        pos_frame = ttk.Frame(nb)
        nb.add(pos_frame, text=" Open Positions ")
        self._positions_tree = self._make_treeview(
            pos_frame,
            columns=["ID", "Pair", "Dir", "Entry", "SL", "TP", "Lots", "Unrealised P&L"],
            widths=[60, 70, 40, 80, 80, 80, 50, 100],
        )

        hist_frame = ttk.Frame(nb)
        nb.add(hist_frame, text=" Trade History ")
        self._history_tree = self._make_treeview(
            hist_frame,
            columns=["Pair", "Dir", "Entry", "Close", "Lots", "P&L", "Strategy", "Closed At"],
            widths=[70, 40, 80, 80, 50, 80, 90, 140],
        )

    @staticmethod
    def _make_treeview(
        parent: ttk.Frame, columns: list[str], widths: list[int]
    ) -> ttk.Treeview:
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)

        tree = ttk.Treeview(frame, columns=columns, show="headings", selectmode="browse")
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)

        for col, w in zip(columns, widths):
            tree.heading(col, text=col)
            tree.column(col, width=w, anchor=tk.CENTER)

        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        return tree

    # ------------------------------------------------------------------
    # Refresh callbacks (run on main thread via after())
    # ------------------------------------------------------------------

    def _refresh_prices(self) -> None:
        for pair, price in self._prices.items():
            if pair in self._price_vars:
                self._price_vars[pair].set(f"{price:.5f}")

    def _refresh_positions(self) -> None:
        tree = self._positions_tree
        if tree is None:
            return
        for row in tree.get_children():
            tree.delete(row)
        for pos in self._positions:
            pnl = pos.get("unrealised_pnl", 0.0)
            colour = "green_row" if pnl >= 0 else "red_row"
            tree.insert(
                "",
                tk.END,
                values=(
                    pos.get("id", ""),
                    pos.get("pair", ""),
                    pos.get("direction", ""),
                    f"{pos.get('entry_price', 0):.5f}",
                    f"{pos.get('stop_loss', 0):.5f}",
                    f"{pos.get('take_profit', 0):.5f}",
                    f"{pos.get('lot_size', 0):.2f}",
                    f"{pnl:+.2f}",
                ),
                tags=(colour,),
            )
        tree.tag_configure("green_row", foreground=GREEN)
        tree.tag_configure("red_row", foreground=RED)

    def _refresh_history(self) -> None:
        tree = self._history_tree
        if tree is None:
            return
        for row in tree.get_children():
            tree.delete(row)
        for trade in self._history:
            pnl = trade.get("pnl", 0.0)
            colour = "green_row" if pnl >= 0 else "red_row"
            closed_at = trade.get("close_time", "")
            if hasattr(closed_at, "strftime"):
                closed_at = closed_at.strftime("%Y-%m-%d %H:%M")
            tree.insert(
                "",
                tk.END,
                values=(
                    trade.get("pair", ""),
                    trade.get("direction", ""),
                    f"{trade.get('entry_price', 0):.5f}",
                    f"{trade.get('close_price', 0):.5f}",
                    f"{trade.get('lot_size', 0):.2f}",
                    f"{pnl:+.2f}",
                    trade.get("strategy", ""),
                    closed_at,
                ),
                tags=(colour,),
            )
        tree.tag_configure("green_row", foreground=GREEN)
        tree.tag_configure("red_row", foreground=RED)

    def _refresh_metrics(self) -> None:
        mapping = {
            "win_rate": lambda v: f"{v:.1f}%",
            "total_pnl": lambda v: f"${v:+.2f}",
            "total_trades": lambda v: str(int(v)),
            "open_positions": lambda v: str(int(v)),
            "daily_pnl": lambda v: f"${v:+.2f}",
        }
        for key, fmt in mapping.items():
            if key in self._metrics and key in self._metrics_labels:
                self._metrics_labels[key].set(fmt(self._metrics[key]))

    # ------------------------------------------------------------------
    # Clock
    # ------------------------------------------------------------------

    def _tick_clock(self) -> None:
        if self._root:
            self._clock_var.set(datetime.utcnow().strftime("UTC  %Y-%m-%d  %H:%M:%S"))
            self._root.after(1000, self._tick_clock)

    # ------------------------------------------------------------------
    # Close handler
    # ------------------------------------------------------------------

    def _on_close(self) -> None:
        self._running = False
        if self._root:
            self._root.destroy()
        logger.info("Dashboard closed.")
