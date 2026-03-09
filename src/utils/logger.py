"""
Logging utility for the SE Forex Trading Bot.
Provides structured logging to both file and console.
"""

import logging
import os
from datetime import datetime
from pythonjsonlogger import jsonlogger


def setup_logger(name: str = "forex_bot", log_level: str = "INFO") -> logging.Logger:
    """
    Set up and return a configured logger instance.

    Args:
        name: Logger name.
        log_level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    log_filename = os.path.join("logs", f"forex_bot_{datetime.now().strftime('%Y%m%d')}.log")

    # File handler with JSON formatting
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(level)
    json_formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    file_handler.setFormatter(json_formatter)

    # Console handler with human-readable formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def log_trade_event(logger: logging.Logger, event_type: str, details: dict) -> None:
    """
    Log a structured trade event.

    Args:
        logger: Logger instance.
        event_type: Type of trade event (e.g., 'OPEN', 'CLOSE', 'MODIFY').
        details: Dictionary with event details.
    """
    logger.info("Trade event", extra={"event_type": event_type, **details})
