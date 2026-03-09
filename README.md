# SE Forex Trading Bot

A fully functional AI-powered Forex trading bot that combines four independent strategy modules with a multi-layer filter system, comprehensive risk management, and a dark-mode live dashboard.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [How to Run](#how-to-run)
6. [Strategies](#strategies)
7. [Filters](#filters)
8. [Risk Management](#risk-management)
9. [Dashboard](#dashboard)
10. [Database & Logging](#database--logging)
11. [API Setup](#api-setup)
12. [Backtesting](#backtesting)
13. [Troubleshooting](#troubleshooting)

---

## Project Overview

SE Forex Trading Bot uses an **AI Smart Consensus** engine that aggregates signals from four independent trading strategies. A trade is only executed when a weighted majority of strategies agree on direction, and all active filters pass.

**Key features:**

- Smart consensus across 4 strategies (SMC, ICT, Price Action, Indicators)
- Multi-filter system: Trend Filter + Economic News Filter
- Advanced trade management with trailing stops and partial profit-taking
- Fixed-risk position sizing (configurable % risk per trade)
- Daily drawdown monitoring and max concurrent trade limits
- Dark-mode Tkinter dashboard with real-time updates
- SQLite database for trade logging and performance analytics
- Sandbox mode for safe testing without a live broker
- Support for Forex majors, Gold (XAUUSD), and crypto pairs

---

## Project Structure

```
lolo/
├── main.py                          # Entry point
├── requirements.txt                 # Python dependencies
├── config/
│   ├── settings.json                # Main trading settings
│   ├── strategies.json              # Strategy parameters & weights
│   └── risk_management.json        # Risk management settings
├── src/
│   ├── bot.py                       # Main bot controller
│   ├── strategies/
│   │   ├── smc_strategy.py          # Smart Money Concepts
│   │   ├── ict_strategy.py          # ICT Liquidity Strategy
│   │   ├── price_action_strategy.py # Price Action patterns
│   │   └── indicators_strategy.py  # Technical indicators
│   ├── filters/
│   │   ├── trend_filter.py          # Moving average trend filter
│   │   └── news_filter.py           # Economic news filter
│   ├── trade_management/
│   │   ├── position_manager.py      # Position lifecycle management
│   │   └── risk_manager.py         # Risk & position sizing
│   ├── data/
│   │   ├── market_data.py           # Market data fetcher
│   │   └── database.py             # SQLite trade database
│   ├── ui/
│   │   └── dashboard.py            # Dark mode Tkinter dashboard
│   └── utils/
│       └── logger.py               # Structured JSON + console logger
├── data/
│   └── forex_bot.db                # SQLite database (auto-created)
└── logs/
    └── forex_bot_YYYYMMDD.log      # Daily log files (auto-created)
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) OANDA or other broker API credentials

### Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/Alaashoky/lolo.git
   cd lolo
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux / macOS
   venv\Scripts\activate.bat       # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   > **Note:** `ta-lib` requires native C libraries. On Ubuntu/Debian:
   > ```bash
   > sudo apt-get install libta-lib-dev
   > ```
   > On macOS with Homebrew:
   > ```bash
   > brew install ta-lib
   > ```
   > On Windows, download pre-compiled wheels from
   > https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

4. **Set up environment variables** (optional, for live trading)

   Create a `.env` file in the project root:

   ```dotenv
   OANDA_API_KEY=your_oanda_api_key
   OANDA_ACCOUNT_ID=your_account_id
   NEWS_API_KEY=your_newsapi_key
   ```

---

## Configuration

### `config/settings.json`

Controls which pairs to trade, the timeframe, and which filters are active.

```json
{
  "trading": {
    "enabled_pairs": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "XAUUSD"],
    "timeframe": "H1",
    "session": "all"
  },
  "filters": {
    "trend_filter_enabled": true,
    "news_filter_enabled": true
  },
  "data": {
    "broker": "oanda",
    "sandbox_mode": true
  }
}
```

| Field | Description |
|---|---|
| `enabled_pairs` | List of instrument symbols to trade |
| `timeframe` | Candle timeframe: `M1`, `M5`, `M15`, `H1`, `H4`, `D1` |
| `broker` | Data source: `oanda`, `yfinance`, or `ccxt` |
| `sandbox_mode` | `true` = use synthetic data (safe for testing) |

### `config/strategies.json`

Controls which strategies are active and their consensus weights.
All weights should sum to 1.0.

```json
{
  "strategies": {
    "smc":          { "enabled": true, "bos_threshold": 0.002, "weight": 0.25 },
    "ict":          { "enabled": true, "weight": 0.25 },
    "price_action": { "enabled": true, "weight": 0.25 },
    "indicators":   { "enabled": true, "weight": 0.25 }
  },
  "consensus": {
    "min_agreement": 0.5
  }
}
```

`min_agreement` is the minimum weighted-consensus score (0–1) required to execute a trade.

### `config/risk_management.json`

```json
{
  "account": {
    "balance": 10000,
    "leverage": 1
  },
  "risk_management": {
    "risk_per_trade": 2.0,
    "max_daily_loss": 50,
    "max_concurrent_trades": 3
  },
  "stop_loss": {
    "use_trailing_stop": true,
    "trailing_stop_pips": 10
  }
}
```

| Field | Description |
|---|---|
| `balance` | Starting account balance in account currency |
| `risk_per_trade` | Maximum % of account to risk on a single trade |
| `max_daily_loss` | Maximum % daily loss before trading stops |
| `max_concurrent_trades` | Maximum simultaneously open positions |
| `use_trailing_stop` | Enable/disable trailing stop loss |
| `trailing_stop_pips` | Trailing stop distance in pips |

---

## How to Run

### Sandbox mode (default – no broker required)

```bash
python main.py
```

The bot will use synthetic price data and will not connect to any broker.

### Live trading with OANDA

1. Edit `config/settings.json`:
   ```json
   {
     "data": {
       "broker": "oanda",
       "sandbox_mode": false
     }
   }
   ```
2. Add your OANDA credentials to `.env`
3. Run:
   ```bash
   python main.py
   ```

### Live trading with Yahoo Finance (free, limited)

```json
{
  "data": {
    "broker": "yfinance",
    "sandbox_mode": false
  }
}
```

---

## Strategies

### 1. Smart Money Concepts (SMC) – `src/strategies/smc_strategy.py`

Identifies institutional order flow through three signals:

- **Break of Structure (BOS):** Detects when price breaks above/below recent swing high/low by a configurable threshold.
- **Fair Value Gap (FVG):** Identifies 3-candle patterns where a price gap exists between candle 1 and candle 3.
- **Order Block:** Finds the last opposing candle before a strong directional move.

Parameters: `bos_threshold` (default 0.002 = 0.2%)

### 2. ICT Liquidity Strategy – `src/strategies/ict_strategy.py`

Based on Inner Circle Trader concepts:

- **Liquidity Sweep:** Detects false breaks above swing highs / below swing lows (stop hunts) that reverse.
- **Market Structure:** Analyses higher highs / higher lows (bullish) vs lower highs / lower lows (bearish) using the last N bars.

Parameters: `structure_lookback` (default 20)

### 3. Price Action – `src/strategies/price_action_strategy.py`

Classic candlestick and S/R patterns:

- **Pin Bar:** Hammer (bullish) and shooting star (bearish) detection using wick-to-body ratios.
- **Engulfing:** Bullish and bearish engulfing candle patterns.
- **Support/Resistance:** Dynamic levels derived from historical swing highs/lows.

Parameters: `pin_bar_ratio` (default 2.5), `sr_lookback` (default 50)

### 4. Technical Indicators – `src/strategies/indicators_strategy.py`

Pure NumPy/Pandas implementation (no TA-Lib dependency for core signals):

- **RSI (14):** Oversold < 30 = BUY, Overbought > 70 = SELL.
- **Stochastic Oscillator:** K and D both < 20 = BUY, both > 80 = SELL.
- **Bollinger Bands (20, 2σ):** Price below lower band = BUY, above upper band = SELL.
- **ATR filter:** Skips signals in low-volatility conditions.

---

## Filters

### Trend Filter – `src/filters/trend_filter.py`

Uses a multi-MA cascade to determine trend direction:

1. If 200+ bars available: fast MA (20) > slow MA (50) > trend MA (200) = bullish.
2. If 50–199 bars: fast MA > slow MA = bullish.
3. Fallback: price > fast MA = bullish.

Only trades aligned with the trend direction are allowed through.

### News Filter – `src/filters/news_filter.py`

Prevents trading during high-impact economic events:

- Checks for news events ± 30 minutes around each event (configurable).
- When a `NEWS_API_KEY` is provided, fetches real headlines for forex/macro keywords.
- Without an API key, fails open (trading continues) rather than blocking all trades.
- Results are cached for 1 hour to minimise API calls.

---

## Risk Management

The risk system (`src/trade_management/risk_manager.py`) enforces three layers of protection:

### 1. Position Sizing

Uses the **fixed fractional** model:

```
lot_size = (account_balance × risk_per_trade%) / (stop_pips × pip_value)
```

### 2. Stop Loss & Take Profit

- **ATR-based stop:** `entry ± (ATR × 1.5)` when ATR data is available.
- **Pip-based fallback:** `entry ± (trailing_stop_pips × 0.0001)`.
- **Take profit:** `entry ± (risk × rr_ratio)` — default 2:1 risk/reward.

### 3. Drawdown Limits

- `max_concurrent_trades`: hard cap on simultaneous positions.
- `max_daily_loss`: trading halts for the day once the threshold is reached.

---

## Dashboard

The dark-mode dashboard (`src/ui/dashboard.py`) is built with Tkinter and runs in the main thread while the bot runs in a background thread.

**Features:**

- Live price ticker for all configured pairs
- Open positions table with unrealised P&L (green/red colour coding)
- Trade history table with closed trade details
- Performance metrics panel: win rate, total P&L, daily P&L, open positions
- UTC clock

To run with the dashboard, replace `main.py` content with:

```python
import threading
from src.bot import ForexBot
from src.ui.dashboard import Dashboard

if __name__ == "__main__":
    bot = ForexBot()
    dash = Dashboard()

    bot_thread = threading.Thread(target=bot.run, daemon=True)
    bot_thread.start()

    dash.start()  # blocks until window closed
    bot.stop()
```

---

## Database & Logging

### Database (`data/forex_bot.db`)

SQLite database with three tables:

| Table | Contents |
|---|---|
| `trades` | Every opened and closed trade with full details |
| `performance` | Daily performance snapshots |
| `market_data` | Cached OHLCV candle data |

### Logs (`logs/forex_bot_YYYYMMDD.log`)

Structured JSON logs for machine parsing + formatted console output. Log events include trade opens/closes, filter decisions, errors, and performance snapshots.

---

## API Setup

### OANDA

1. Create a free practice account at https://www.oanda.com/
2. Generate an API key from *My Account → Manage API Access*
3. Add to `.env`:
   ```
   OANDA_API_KEY=your-practice-api-key
   OANDA_ACCOUNT_ID=your-account-id
   ```

### NewsAPI (optional)

1. Register at https://newsapi.org/ (free tier available)
2. Add to `.env`:
   ```
   NEWS_API_KEY=your-newsapi-key
   ```
   Then reference it in `settings.json`:
   ```json
   { "filters": { "news_filter_enabled": true, "news_api_key": "your-key" } }
   ```

### Yahoo Finance

No API key required. Set `"broker": "yfinance"` in `settings.json`. Note that Yahoo Finance data has a 15-minute delay and limited history for some forex pairs.

---

## Backtesting

To backtest a strategy using historical data:

```python
import pandas as pd
from src.strategies.smc_strategy import SMCStrategy
from src.strategies.indicators_strategy import IndicatorsStrategy
from src.data.market_data import MarketData

# Load historical data
md = MarketData({"broker": "yfinance", "sandbox_mode": False})
df = md.get_candles("EURUSD", "H1", 500)

# Run strategy on rolling windows
smc = SMCStrategy({"enabled": True, "bos_threshold": 0.002, "weight": 0.25})
results = []
for i in range(50, len(df)):
    window = df.iloc[:i]
    result = smc.analyze(window)
    results.append(result["signal"])

print(pd.Series(results).value_counts())
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'talib'`

Install system dependencies before `pip install ta-lib`:

```bash
# Ubuntu/Debian
sudo apt-get install libta-lib0-dev

# macOS
brew install ta-lib
```

Note: The indicators strategy uses a pure NumPy/Pandas implementation so `ta-lib` is not required for core functionality.

### `sqlite3.OperationalError: unable to open database file`

Ensure the `data/` directory is writable:

```bash
mkdir -p data
```

### Bot shows only NEUTRAL signals in sandbox mode

This is expected with the default synthetic random-walk data and all four strategies enabled with a `min_agreement` of 0.5. Lower `min_agreement` to `0.3` in `strategies.json` to see more signals during testing.

### Dashboard window does not appear

Tkinter requires a display. On headless servers, install:

```bash
sudo apt-get install python3-tk
export DISPLAY=:0
```

Or use a virtual framebuffer:

```bash
sudo apt-get install xvfb
Xvfb :99 -screen 0 1920x1080x24 &
export DISPLAY=:99
python main.py
```

### High-impact news filter is blocking all trades

If `NEWS_API_KEY` is not set, the news filter fails open (allows trading). If it is set and blocking unexpectedly, check the `pre_event_minutes` / `post_event_minutes` values in `settings.json` and reduce them.

---

## License

This project is provided for educational purposes. Always test thoroughly in a sandbox environment before risking real capital. Forex trading involves significant risk of loss.