"""
Market Data Handler.

Fetches real-time and historical OHLCV candle data from multiple
broker/exchange sources (OANDA, yfinance, ccxt) with validation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger("forex_bot.market_data")


class MarketData:
    """
    Unified market data interface.

    Supports three data sources:
    - ``oanda``: OANDA REST API v20 (requires ``oanda_api_key`` and
      ``oanda_account_id`` in config)
    - ``yfinance``: Yahoo Finance (no API key required, sandbox or live)
    - ``ccxt``: Generic crypto/forex exchange via ccxt

    In sandbox mode the class generates synthetic OHLCV data for
    testing purposes.
    """

    REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]

    def __init__(self, config: dict) -> None:
        """
        Args:
            config: ``data`` section from settings.json, optionally
                    merged with broker credentials from environment.
        """
        self.broker: str = config.get("broker", "yfinance").lower()
        self.sandbox: bool = config.get("sandbox_mode", True)
        self.oanda_api_key: Optional[str] = config.get("oanda_api_key")
        self.oanda_account_id: Optional[str] = config.get("oanda_account_id")
        self._price_cache: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_candles(
        self,
        pair: str,
        timeframe: str = "H1",
        count: int = 200,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candle data.

        Args:
            pair: Instrument symbol, e.g. 'EURUSD' or 'EUR_USD'.
            timeframe: Candle timeframe ('M1', 'M5', 'M15', 'H1', 'H4', 'D1').
            count: Number of candles to fetch.

        Returns:
            DataFrame with columns [open, high, low, close, volume] and a
            DatetimeIndex. Returns an empty DataFrame on failure.
        """
        if self.sandbox:
            return self._generate_synthetic_data(pair, count)

        try:
            if self.broker == "oanda":
                return self._fetch_oanda(pair, timeframe, count)
            if self.broker == "yfinance":
                return self._fetch_yfinance(pair, timeframe, count)
            if self.broker == "ccxt":
                return self._fetch_ccxt(pair, timeframe, count)
        except Exception as exc:
            logger.error("Failed to fetch candles for %s: %s", pair, exc)

        return pd.DataFrame(columns=self.REQUIRED_COLUMNS)

    def get_current_price(self, pair: str) -> Optional[float]:
        """
        Return the latest ask/mid price for *pair*.

        In sandbox mode returns the last synthetic close price.

        Args:
            pair: Instrument symbol.

        Returns:
            Current price as float, or None on failure.
        """
        if self.sandbox:
            price = self._price_cache.get(pair)
            if price is None:
                df = self._generate_synthetic_data(pair, 5)
                price = float(df["close"].iloc[-1]) if not df.empty else None
                if price:
                    self._price_cache[pair] = price
            return price

        try:
            df = self.get_candles(pair, "M1", 2)
            if not df.empty:
                return float(df["close"].iloc[-1])
        except Exception as exc:
            logger.warning("Could not get current price for %s: %s", pair, exc)

        return None

    def update_price_cache(self, prices: dict[str, float]) -> None:
        """Manually update the price cache (e.g., from broker stream)."""
        self._price_cache.update(prices)

    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Check that a DataFrame has the required OHLCV columns and no NaNs
        in the last row.

        Returns:
            True if valid, False otherwise.
        """
        if df is None or df.empty:
            return False
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                return False
        if df[self.REQUIRED_COLUMNS].iloc[-1].isnull().any():
            return False
        return True

    # ------------------------------------------------------------------
    # Private: data source adapters
    # ------------------------------------------------------------------

    def _fetch_yfinance(
        self, pair: str, timeframe: str, count: int
    ) -> pd.DataFrame:
        """Fetch data using yfinance."""
        import yfinance as yf  # lazy import

        tf_map = {
            "M1": "1m", "M5": "5m", "M15": "15m",
            "H1": "1h", "H4": "4h", "D1": "1d",
        }
        interval = tf_map.get(timeframe, "1h")

        # yfinance uses 'EURUSD=X' notation for FX pairs
        symbol = pair.upper()
        if "=" not in symbol and "/" not in symbol and len(symbol) == 6:
            symbol = f"{symbol}=X"

        ticker = yf.Ticker(symbol)
        period = "60d" if timeframe in ("H1", "H4", "D1") else "5d"
        df = ticker.history(period=period, interval=interval)
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={"stock splits": "splits"})
        df = df[["open", "high", "low", "close", "volume"]].tail(count)
        df.index = pd.to_datetime(df.index, utc=True)
        return df

    def _fetch_oanda(
        self, pair: str, timeframe: str, count: int
    ) -> pd.DataFrame:
        """Fetch data from OANDA v20 REST API."""
        import v20  # type: ignore

        tf_map = {
            "M1": "M1", "M5": "M5", "M15": "M15",
            "H1": "H1", "H4": "H4", "D1": "D",
        }
        granularity = tf_map.get(timeframe, "H1")
        instrument = pair.replace("/", "_")

        ctx = v20.Context("api-fxtrade.oanda.com", token=self.oanda_api_key)
        resp = ctx.instrument.candles(
            instrument,
            count=count,
            granularity=granularity,
            price="M",
        )
        candles = resp.get("candles", [])
        records = []
        for c in candles:
            mid = c.mid
            records.append(
                {
                    "time": pd.Timestamp(c.time),
                    "open": float(mid.o),
                    "high": float(mid.h),
                    "low": float(mid.l),
                    "close": float(mid.c),
                    "volume": int(c.volume),
                }
            )
        df = pd.DataFrame(records).set_index("time")
        df.index = pd.to_datetime(df.index, utc=True)
        return df

    def _fetch_ccxt(
        self, pair: str, timeframe: str, count: int
    ) -> pd.DataFrame:
        """Fetch data via ccxt (generic exchange)."""
        import ccxt  # type: ignore

        tf_map = {
            "M1": "1m", "M5": "5m", "M15": "15m",
            "H1": "1h", "H4": "4h", "D1": "1d",
        }
        interval = tf_map.get(timeframe, "1h")
        exchange = ccxt.oanda()
        symbol = pair[:3] + "/" + pair[3:]
        ohlcv = exchange.fetch_ohlcv(symbol, interval, limit=count)
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("time").drop(columns=["timestamp"])
        return df

    # ------------------------------------------------------------------
    # Synthetic data generator (sandbox / testing)
    # ------------------------------------------------------------------

    def _generate_synthetic_data(self, pair: str, count: int) -> pd.DataFrame:
        """
        Generate a random-walk OHLCV series for sandbox testing.

        The seed is derived from the pair name so the same pair always
        produces a consistent series within a Python session.
        """
        rng = np.random.default_rng(seed=sum(ord(c) for c in pair))

        base_prices = {
            "EURUSD": 1.08, "GBPUSD": 1.26, "USDJPY": 149.0,
            "AUDUSD": 0.65, "XAUUSD": 2000.0, "USDCAD": 1.35,
            "USDCHF": 0.90, "NZDUSD": 0.60,
        }
        base = base_prices.get(pair.upper().replace("_", "").replace("/", ""), 1.10)

        returns = rng.normal(0, 0.0005, count)
        closes = base * np.cumprod(1 + returns)

        noise = rng.uniform(0.0001, 0.0005, count)
        highs = closes + noise
        lows = closes - noise
        opens = np.roll(closes, 1)
        opens[0] = base
        volumes = rng.integers(100, 5000, count).astype(float)

        index = pd.date_range(
            end=datetime.now(timezone.utc), periods=count, freq="h"
        )
        df = pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
            index=index,
        )
        return df
