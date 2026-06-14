TradeAnalyze V.3 – Institutional Trading Bot
Futures + Options | Markov Regime | ML Forecast | Conviction Scoring | Portfolio Optimiser

TradeAnalyze V.3 is a fully automated trading system that analyses daily OHLCV data for stocks, ETFs and crypto assets, generates directional futures signals (LONG / SHORT / NO TRADE) and recommends option strategies using a multi‑engine institutional pipeline. All signals are logged to Google Sheets and sent to Telegram / LINE with hedge‑fund‑style reports.

🚀 Key Features
Feature	Description
Markov Regime Detection	Gaussian HMM with 5 states (STRONG_BULL … STRONG_BEAR)
Ensemble Regime v2	7‑component: Markov, Trend, Volatility, Macro, Liquidity, Breadth, Cross‑Asset
Liquidity Regime	DXY, VIX, Yields, TLT → RISK_ON / RISK_OFF / CRISIS / RECOVERY
Market Breadth	Crypto (BTC.D, TOTAL3, SSR) + Equity (A/D, %>200dma, NH/NL)
Derivatives Flow	Funding rate, Open Interest, Long/Short ratio, Liquidation clusters
Conviction Engine	8‑signal weighted score → FULL / NORMAL / HALF / NO TRADE
ML Forecast	XGBoost / LightGBM → 5d/10d/20d expected returns & probabilities
Walk‑Forward Validation	Rolling OOS performance & adaptive ensemble weights (V3)
Regime‑Switching Monte Carlo	GBM with regime‑dependent volatility (V3)
Portfolio Optimiser	Risk Parity & Hierarchical Risk Parity (HRP)
Option Strategy Selector	14 strategies – EV / POP / Kelly ranked, approved by AI score
Advanced Greeks	Delta, Gamma, Vega, Theta, Vanna, Charm, Vomma, Veta, Speed
LINE / Telegram Alerts	Institutional‑grade rate‑limited alerts with monthly quota detection
Google Sheets Logging	TradeSignals, Options, Option_Chain, InstitutionalSignals, etc.
📦 Quick Start
Clone & install

bash
git clone https://github.com/TFYKub/TradeAnalyze_V2.git
cd TradeAnalyze_V2
python -m venv venv_v2 && source venv_v2/bin/activate
pip install -r requirements.txt
Set environment variables (.env file or shell)

ini
LINE_TOKEN=your_line_token
SHEET_ID=your_google_sheet_id
GOOGLE_CREDENTIALS='{"type":"service_account",...}'
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
Configure symbols in Google Sheet SYMBOL_CONFIG (columns: symbol, group, asset_type)

Enable V3 in config/config.py

python
USE_V3 = True
Run

bash
python main.py
📊 Outputs
Telegram / LINE – executive decision, trade plan, option strategy, Monte Carlo, institutional dashboards.

Google Sheets – TradeSignals, Options, Option_Chain, InstitutionalSignals, RegimeDashboard, FlowSnapshot.

🧪 V3 New Phases (optional, toggle in config.py)
Phase	Description
1	Walk‑forward validation + adaptive ensemble
2	Quantile forecast (LightGBM, 5/10/20d)
3	Regime‑switching Monte Carlo
4	Portfolio optimiser (risk parity, min var, max Sharpe, Kelly)
5	Crypto flow V2 (OI delta, funding momentum, liquidations, whale flow)
6	Dealer Greeks (gamma exposure, vanna, charm, pin‑risk)
7	Stress testing (2008, Covid, 2022, vol shock, liquidity crisis)
🔧 Configuration
All thresholds (MIN_RR, MIN_AI_SCORE, MAX_REGIME_CONFIDENCE, etc.) are centralised in config/thresholds.py.
Notification channels: Telegram primary, LINE fallback (automatic monthly quota detection).

🤝 Contributing
Run tests before submitting PRs:

bash
pytest tests/
📄 License
Proprietary – all rights reserved.

Built with yfinance, hmmlearn, xgboost, lightgbm, scipy, gspread.