"""
SE Forex Trading Bot – Entry Point.

Run this file to start the bot:

    python main.py

The bot reads configuration from the ``config/`` directory and
begins the main trading loop. Press Ctrl+C to stop gracefully.
"""

from src.bot import ForexBot

if __name__ == "__main__":
    bot = ForexBot()
    bot.run()
