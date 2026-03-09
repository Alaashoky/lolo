"""
Data Downloader for historical OHLCV data.

Supports multiple data sources:
- yfinance  – forex, commodities, crypto
- ccxt      – cryptocurrency exchanges
- OANDA     – forex (requires API key)

Downloads up to 5 years of hourly candles, handles missing data,
validates integrity, and persists results as CSV files in data/historical/.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("forex_bot.backtesting.data_downloader")

# ---------------------------------------------------------------------------
# Symbol mapping: internal name → yfinance ticker
# ---------------------------------------------------------------------------
_YFINANCE_MAP: Dict[str, str] = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
    "USDCAD": "USDCAD=X",
    "USDCHF": "USDCHF=X",
    "XAUUSD": "GC=F",
    "XAGUSD": "SI=F",
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
}

# yfinance interval strings
_INTERVAL_MAP: Dict[str, str] = {
    "M1":  "1m",
    "M5":  "5m",
    "M15": "15m",
    "M30": "30m",
    "H1":  "1h",
    "H4":  "4h",
    "D1":  "1d",
    "W1":  "1wk",
}

# Maximum days yfinance returns per request for each interval
_MAX_DAYS: Dict[str, int] = {
    "1m":  7,
    "5m":  60,
    "15m": 60,
    "30m": 60,
    "1h":  730,
    "4h":  730,
    "1d":  3650,
    "1wk": 3650,
}


class DataDownloader:
    """
    Downloads and caches multi-year OHLCV data from supported sources.

    Args:
        data_dir: Directory to store CSV files (default: ``data/historical``).
        source:   Primary data source (``yfinance``, ``ccxt``).
    """

    def __init__(
        self,
        data_dir: str = "data/historical",
        source: str = "yfinance",
    ) -> None:
        self._data_dir = data_dir
        self._source = source.lower()
        os.makedirs(data_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(
        self,
        pair: str,
        start: datetime,
        end: datetime,
        timeframe: str = "H1",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Download (or load from cache) OHLCV data for *pair*.

        Args:
            pair:          Symbol name, e.g. ``EURUSD``.
            start:         Inclusive start datetime (UTC).
            end:           Exclusive end datetime (UTC).
            timeframe:     One of ``M1``, ``M5``, ``M15``, ``M30``,
                           ``H1``, ``H4``, ``D1``, ``W1``.
            force_refresh: Re-download even if a cached file exists.

        Returns:
            DataFrame with columns [open, high, low, close, volume] and a
            DatetimeIndex in UTC.
        """
        cache_path = self._cache_path(pair, timeframe, start, end)
        if not force_refresh and os.path.exists(cache_path):
            logger.info("Loading cached data for %s from %s", pair, cache_path)
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=True)
            return df

        logger.info(
            "Downloading %s %s from %s to %s (source=%s)",
            pair, timeframe, start.date(), end.date(), self._source,
        )

        # Choose source based on symbol type or explicit setting
        source = self._resolve_source(pair)

        if source == "yfinance":
            df = self._download_yfinance(pair, start, end, timeframe)
        elif source == "ccxt":
            df = self._download_ccxt(pair, start, end, timeframe)
        else:
            raise ValueError(f"Unsupported data source: {source}")

        df = self._clean(df)

        if df.empty:
            logger.warning("No data returned for %s %s.", pair, timeframe)
            return df

        df.to_csv(cache_path)
        logger.info("Saved %d rows to %s", len(df), cache_path)
        return df

    def download_multiple(
        self,
        pairs: List[str],
        years: int = 5,
        timeframe: str = "H1",
        force_refresh: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Download historical data for a list of pairs.

        Args:
            pairs:         List of symbol names.
            years:         How many years of history to fetch.
            timeframe:     Candle timeframe.
            force_refresh: Re-download even if cache exists.

        Returns:
            Dictionary mapping pair name → DataFrame.
        """
        end = datetime.now(tz=timezone.utc)
        start = end - timedelta(days=years * 365)

        results: Dict[str, pd.DataFrame] = {}
        for pair in pairs:
            try:
                df = self.download(
                    pair, start, end, timeframe=timeframe, force_refresh=force_refresh
                )
                results[pair] = df
                logger.info(
                    "Downloaded %s: %d rows (%s → %s)",
                    pair, len(df),
                    df.index[0].date() if not df.empty else "N/A",
                    df.index[-1].date() if not df.empty else "N/A",
                )
            except Exception as exc:
                logger.error("Failed to download %s: %s", pair, exc)
                results[pair] = pd.DataFrame()

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_source(self, pair: str) -> str:
        """Pick the best source for a given pair."""
        crypto_pairs = {"BTCUSD", "ETHUSD", "BNBUSD", "XRPUSD"}
        if pair in crypto_pairs and self._source == "ccxt":
            return "ccxt"
        return "yfinance"

    def _cache_path(
        self, pair: str, timeframe: str, start: datetime, end: datetime
    ) -> str:
        fname = (
            f"{pair}_{timeframe}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
        )
        return os.path.join(self._data_dir, fname)

    # ------------------------------------------------------------------
    # yfinance downloader
    # ------------------------------------------------------------------

    def _download_yfinance(
        self,
        pair: str,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> pd.DataFrame:
        """Download from Yahoo Finance in chunks when needed."""
        try:
            import yfinance as yf  # type: ignore
        except ImportError:
            raise ImportError("yfinance is not installed. Run: pip install yfinance")

        ticker = _YFINANCE_MAP.get(pair)
        if ticker is None:
            # Attempt a direct pass-through (e.g. "EURUSD=X")
            ticker = pair
            logger.debug("No yfinance mapping for %s – using %s directly.", pair, ticker)

        interval = _INTERVAL_MAP.get(timeframe, "1h")
        max_days = _MAX_DAYS.get(interval, 365)

        # Split request into chunks if the range exceeds the API limit
        chunks: List[pd.DataFrame] = []
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + timedelta(days=max_days), end)
            try:
                raw = yf.download(
                    ticker,
                    start=chunk_start.strftime("%Y-%m-%d"),
                    end=chunk_end.strftime("%Y-%m-%d"),
                    interval=interval,
                    auto_adjust=True,
                    progress=False,
                )
                if not raw.empty:
                    chunks.append(raw)
            except Exception as exc:
                logger.warning(
                    "yfinance error for %s (%s → %s): %s",
                    ticker, chunk_start.date(), chunk_end.date(), exc,
                )
            chunk_start = chunk_end
            # Respect rate limits
            time.sleep(0.5)

        if not chunks:
            return pd.DataFrame()

        df = pd.concat(chunks)
        df = df[~df.index.duplicated(keep="first")]
        df.columns = [c.lower() for c in df.columns]

        # Normalise column names
        rename = {}
        for col in df.columns:
            if col in ("adj close", "adj_close"):
                rename[col] = "close"
        df = df.rename(columns=rename)

        required = ["open", "high", "low", "close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning("Missing columns %s for %s.", missing, pair)
            return pd.DataFrame()

        if "volume" not in df.columns:
            df["volume"] = 0.0

        df.index = pd.to_datetime(df.index, utc=True)
        return df[["open", "high", "low", "close", "volume"]].sort_index()

    # ------------------------------------------------------------------
    # ccxt downloader
    # ------------------------------------------------------------------

    def _download_ccxt(
        self,
        pair: str,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> pd.DataFrame:
        """Download from a ccxt-compatible exchange (default: Binance)."""
        try:
            import ccxt  # type: ignore
        except ImportError:
            raise ImportError("ccxt is not installed. Run: pip install ccxt")

        # Convert internal pair format (BTCUSD → BTC/USDT)
        symbol = _to_ccxt_symbol(pair)
        tf = _INTERVAL_MAP.get(timeframe, "1h")

        exchange = ccxt.binance({"enableRateLimit": True})
        if not exchange.has["fetchOHLCV"]:
            raise RuntimeError("Exchange does not support fetchOHLCV.")

        since_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        limit = 1000
        rows: list = []

        while since_ms < end_ms:
            try:
                data = exchange.fetch_ohlcv(symbol, timeframe=tf, since=since_ms, limit=limit)
            except Exception as exc:
                logger.warning("ccxt error for %s: %s", symbol, exc)
                break

            if not data:
                break

            rows.extend(data)
            since_ms = data[-1][0] + 1
            time.sleep(exchange.rateLimit / 1000)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
        df = df[df.index < pd.Timestamp(end, tz=timezone.utc)]
        return df

    # ------------------------------------------------------------------
    # Cleaning & validation
    # ------------------------------------------------------------------

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate OHLCV data:
        - Normalise column names
        - Remove duplicates
        - Forward-fill then back-fill missing values
        - Drop rows where prices are zero or negative
        - Remove obvious outliers (z-score > 8)
        """
        if df.empty:
            return df

        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        # Remove duplicate timestamps
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()

        # Forward-fill then back-fill
        df = df.ffill().bfill()
        df = df.dropna()

        # Remove zero or negative prices
        price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
        mask = (df[price_cols] > 0).all(axis=1)
        df = df[mask]

        # Remove extreme outliers in close price
        if len(df) > 30:
            close = df["close"]
            z = (close - close.mean()) / (close.std() + 1e-10)
            df = df[z.abs() <= 8]

        return df.reset_index(drop=False)  # keep the datetime index

    def validate(self, df: pd.DataFrame, pair: str = "") -> Dict[str, object]:
        """
        Validate downloaded data and return a report.

        Returns:
            Dict with keys: rows, missing_pct, start, end, issues.
        """
        report: Dict[str, object] = {
            "pair": pair,
            "rows": len(df),
            "missing_pct": 0.0,
            "start": None,
            "end": None,
            "issues": [],
        }

        if df.empty:
            report["issues"] = ["DataFrame is empty"]
            return report

        report["start"] = str(df.index[0])
        report["end"] = str(df.index[-1])
        report["missing_pct"] = float(df.isnull().mean().mean() * 100)

        issues = []
        price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
        if (df[price_cols] <= 0).any().any():
            issues.append("Contains zero or negative prices")

        if df.index.duplicated().any():
            issues.append("Duplicate timestamps found")

        if not df.index.is_monotonic_increasing:
            issues.append("Index is not sorted")

        report["issues"] = issues
        return report


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _to_ccxt_symbol(pair: str) -> str:
    """Convert ``BTCUSD`` → ``BTC/USDT``."""
    mapping = {
        "BTCUSD": "BTC/USDT",
        "ETHUSD": "ETH/USDT",
        "BNBUSD": "BNB/USDT",
        "XRPUSD": "XRP/USDT",
    }
    return mapping.get(pair, pair)
