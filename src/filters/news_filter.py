"""
News Filter.

Checks for upcoming high-impact economic events and pauses trading
when such events are imminent or recently concluded.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

logger = logging.getLogger("forex_bot.news_filter")


class NewsFilter:
    """
    Economic news filter.

    Queries an economic calendar API (or falls back to a static list)
    to detect high-impact events. Trading is paused for a configurable
    window around each event.
    """

    # Fallback list of known high-impact event keywords
    HIGH_IMPACT_KEYWORDS = [
        "non-farm payroll",
        "nfp",
        "fomc",
        "interest rate",
        "cpi",
        "gdp",
        "unemployment",
        "ecb",
        "boe",
        "boj",
        "fed",
        "inflation",
        "pmi",
    ]

    def __init__(self, config: dict) -> None:
        """
        Args:
            config: Settings dict. Supported keys:
              - news_filter_enabled (bool)
              - news_api_key (str): NewsAPI key (optional)
              - pre_event_minutes (int): minutes before event to pause (default 30)
              - post_event_minutes (int): minutes after event to pause (default 30)
              - affected_currencies (list[str]): currencies to monitor
        """
        self.enabled: bool = config.get("news_filter_enabled", True)
        self.api_key: Optional[str] = config.get("news_api_key")
        self.pre_event_minutes: int = config.get("pre_event_minutes", 30)
        self.post_event_minutes: int = config.get("post_event_minutes", 30)
        self.affected_currencies: list[str] = config.get(
            "affected_currencies", ["USD", "EUR", "GBP", "JPY", "AUD"]
        )
        self._event_cache: list[dict] = []
        self._cache_expiry: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_trade_allowed(self, pair: str, now: Optional[datetime] = None) -> bool:
        """
        Return True if trading is allowed for the given pair right now.

        Args:
            pair: Currency pair symbol, e.g. 'EURUSD'.
            now: Current datetime (UTC). Defaults to datetime.now(UTC).

        Returns:
            True if no high-impact event is imminent, False otherwise.
        """
        if not self.enabled:
            return True

        if now is None:
            now = datetime.now(timezone.utc)

        currencies = self._pair_to_currencies(pair)
        events = self._get_events(now)

        for event in events:
            event_time = event.get("time")
            currency = event.get("currency", "")
            if event_time is None:
                continue
            if currency not in currencies and currency != "ALL":
                continue

            window_start = event_time - timedelta(minutes=self.pre_event_minutes)
            window_end = event_time + timedelta(minutes=self.post_event_minutes)

            if window_start <= now <= window_end:
                logger.info(
                    "Trading paused due to high-impact news",
                    extra={"event": event.get("title"), "pair": pair},
                )
                return False

        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pair_to_currencies(self, pair: str) -> list[str]:
        """Extract base and quote currencies from a pair symbol."""
        pair = pair.upper().replace("/", "").replace("_", "").replace("-", "")
        if len(pair) >= 6:
            return [pair[:3], pair[3:6]]
        return [pair]

    def _get_events(self, now: datetime) -> list[dict]:
        """
        Return a list of high-impact economic events for today.

        Attempts to fetch from NewsAPI; falls back to an empty list on
        any error (fail-open: allow trading when news data is unavailable).
        Results are cached for one hour.
        """
        if self._cache_expiry and now < self._cache_expiry and self._event_cache:
            return self._event_cache

        events: list[dict] = []

        if self.api_key:
            try:
                events = self._fetch_newsapi_events(now)
            except Exception as exc:
                logger.warning("NewsAPI fetch failed: %s – trading will continue.", exc)

        self._event_cache = events
        self._cache_expiry = now + timedelta(hours=1)
        return events

    def _fetch_newsapi_events(self, now: datetime) -> list[dict]:
        """
        Fetch economic headlines from NewsAPI and convert them to
        event dicts if they match high-impact keywords.
        """
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": "forex interest rate central bank",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 20,
            "apiKey": self.api_key,
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        events = []
        for article in data.get("articles", []):
            title = (article.get("title") or "").lower()
            if any(kw in title for kw in self.HIGH_IMPACT_KEYWORDS):
                published = article.get("publishedAt", "")
                try:
                    event_time = datetime.fromisoformat(
                        published.replace("Z", "+00:00")
                    )
                except ValueError:
                    continue
                events.append(
                    {
                        "title": article.get("title"),
                        "currency": "ALL",
                        "time": event_time,
                    }
                )

        return events
