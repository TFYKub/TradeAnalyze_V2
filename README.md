# TradeAnalyze V3.0

**Institutional‑grade trading system** combining futures signals, options analysis, portfolio optimisation, and risk management — designed for systematic trading of equities, crypto, and futures.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Features

- **Unified Signal Pipeline**  
  Combines 19+ engines: Markov Regime, Market Structure, RSI Divergence, Bayesian Reliability, ML Forecast, Conviction Scoring, and more.

- **Institutional Options Analysis**  
  Full suite: chain fetching (yfinance + Deribit), Black‑Scholes Greeks, volatility surface, expected move, 14 strategies, EV/POP/Kelly ranking.

- **Multi‑Asset Support**  
  Equities, crypto (BTC, ETH, SOL, etc.), and futures‑like instruments.

- **Portfolio Optimisation**  
  Risk Parity and Hierarchical Risk Parity (HRP) for multi‑asset allocation; single‑asset mode provides volatility and drawdown estimates.

- **Robust Monitoring**  
  Health server (Flask), Prometheus metrics, watchdog (stall detection), daily summary reports.

- **Production‑Ready**  
  SQLite persistence, batching for Google Sheets, graceful fallbacks, retries, and LINE/Telegram alerts.

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/TradeAnalyze_V2.git
cd TradeAnalyze_V2
2. Set Up Environment
Create a virtual environment (optional but recommended):

bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
3. Install Dependencies
bash
pip install -r requirements.txt
4. Configure Environment Variables
Create a .env file in the project root:

env
SHEET_ID=your_google_sheet_id
GOOGLE_CREDENTIALS={"type": "service_account", ...}   # JSON string or path
LINE_TOKEN=your_line_notify_token
TELEGRAM_BOT_TOKEN=your_telegram_bot_token   # optional
TELEGRAM_CHAT_ID=your_telegram_chat_id       # optional
5. Set Up Google Sheets
Create a Google Sheet and share it with your service account email.

Add a worksheet named SYMBOL_CONFIG with columns: symbol, group, asset_type.

Fill with your symbols (e.g., AAPL, BTC-USD, ETH-USD), group LINE, and stock or crypto.

6. Run the System
bash
python main.py
The system will start processing symbols, generate signals, and send notifications.

Configuration
All thresholds are centralised in config/thresholds.py.
Feature flags (V3 phases, parallel execution) are in config/config.py.