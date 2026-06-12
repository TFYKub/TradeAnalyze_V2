Core Principle
Additive & backward‑compatible – original pipeline (FuturesOrchestrator) untouched.

New orchestrator FuturesOrchestrator_v2 extends the 19‑step pipeline with 10 institutional engines.

New result object FuturesResult_v2 contains all original fields + new institutional fields.

Rollback: change one import.

New Engines (Phases 12–21)
Phase	Engine	Purpose
12	Liquidity Regime	DXY, VIX, yields, TLT → RISK_ON/OFF/CRISIS/RECOVERY
13	Flow Engine	Funding rate, OI, L/S ratio, liquidation cascade
14	Market Breadth	Crypto + equity breadth (A/D, % above 200DMA)
15	Regime Persistence	Markov sojourn time, half‑life, exit probability
16	Bayesian Reliability	Regime‑conditional weights (prevents RSI dominance)
17	Forecast Engine	XGBoost/LightGBM 5d/10d/20d return predictions
18	Conviction Engine	8‑signal aggregator → position size gate (NO TRADE / HALF / NORMAL / FULL)
19	Advanced Greeks	Vanna, Charm, Vomma, Veta, Speed – full options card
20	Portfolio Optimizer	Risk Parity + Hierarchical Risk Parity (HRP)
21	Cross‑Asset Engine	BTC vs SPY/QQQ/GLD/DXY/TLT – decoupling detection
Integration Points
Step 2b – Liquidity Regime

Step 3 – Regime Ensemble v2 (includes liquidity, breadth, cross‑asset)

Step 3b – Market Breadth

Step 3c – Cross‑Asset Engine

Step 3d – Regime Persistence

Step 13b – Flow Engine

Step 14 – Bayesian Reliability v2

Step 14b – Forecast Engine

Step 14c – Conviction Engine

Step 16b – Portfolio Optimizer (single‑asset mode)

Report – Institutional dashboards appended to daily report

New Outputs (examples)
conviction_tier (FULL SIZE / NORMAL SIZE / HALF SIZE / NO TRADE)

liquidity_regime, flow_regime, breadth_regime, persistence_label

forecast_direction, forecast_20d_return

cross_asset_regime, btc_rs_score, btc_beta_spy

portfolio_vol, portfolio_sharpe, portfolio_drawdown

New Google Sheets (auto‑created)
InstitutionalSignals – full snapshot of all 50+ new fields

RegimeDashboard – daily regime + conviction + persistence

FlowSnapshot – derivatives flow history

New LINE Alerts v2
Conviction‑filtered (min_conviction tunable)

Automatic alerts for CRISIS liquidity and squeezes

Dependencies Added
scipy (HRP clustering)

xgboost / lightgbm (optional, fallback to statistical forecast)

scikit-learn (feature scaling)

Safety Design
Every engine wrapped in _safe() → returns None on failure

All field reads use getattr(result, field, default)

No single engine crash stops the pipeline

Migration Steps (8 phases)
Install dependencies (pip install -r requirements_v2.txt)

Run unit tests (pytest tests/ -v)

Create 3 new worksheets (one‑time script)

Dry run (both orchestrators, v2 writes sheets only)

Enable v2 LINE alerts (start with min_conviction=65)

Cut over to v2 orchestrator (change one import)

Calibrate conviction thresholds (after 1 week)

Forecast model warm‑up (automatic after 120+ bars)

Rollback
Change import back to FuturesOrchestrator – original pipeline resumes.

Updated README.md
Replace your existing README.md with the content below.

markdown
# TradeAnalyze v2.0 — Institutional Quant Platform

> Built on TradeAnalyze v1.4 – fully backward compatible.  
> Adds 10 institutional‑grade engines as a second analytical layer.

[![Tests](https://img.shields.io/badge/tests-175%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()

---

## What’s New in v2.0

**Original v1.4 (preserved unchanged):**  
Markov Regime · Regime Ensemble · Volatility Regime · Market Structure · Bayesian Engine · AI Score · Monte Carlo · Kelly Sizing · Volume Profile · Anchored VWAP · Options Layer · Portfolio Layer · Google Sheets · LINE Alerts

**New institutional layer (v2.0 – Phases 12–21):**

| Phase | Engine | File |
|-------|--------|------|
| 12 | Liquidity Regime | `engines/liquidity_regime.py` |
| 13 | Flow Engine | `crypto/flow_engine.py` |
| 14 | Market Breadth | `engines/market_breadth.py` |
| 15 | Regime Persistence | `engines/regime_persistence.py` |
| 16 | Bayesian Reliability | `engines/bayesian_reliability.py` |
| 17 | Forecast Engine | `engines/forecast_engine.py` |
| 18 | Conviction Engine | `engines/conviction_engine.py` |
| 19 | Advanced Greeks | `options/greeks_advanced.py` |
| 20 | Portfolio Optimizer | `portfolio/optimizer.py` |
| 21 | Cross‑Asset Engine | `engines/cross_asset_engine.py` |

---

## Quick Start

```bash
# 1. Install (includes v2 dependencies)
python -m venv venv_v2
source venv_v2/bin/activate   # Windows: venv_v2\Scripts\activate
pip install -r requirements_v2.txt

# 2. Configure environment (copy .env.example → .env)
LINE_TOKEN=...
SHEET_ID=...
GOOGLE_CREDENTIALS=...   # JSON string or file path

# 3. Run tests
pytest tests/ -v

# 4. Run the v2 pipeline (dry run first)
python main.py
Key New Outputs (from FuturesResult_v2)
python
print(result.conviction_tier)      # FULL SIZE / NORMAL SIZE / HALF SIZE / NO TRADE
print(result.conviction_score)     # 0–100
print(result.liquidity_regime)     # RISK_ON / RISK_OFF / CRISIS / RECOVERY
print(result.flow_regime)          # BULLISH_FLOW / SHORT_SQUEEZE / LONG_SQUEEZE
print(result.forecast_direction)   # BULLISH / BEARISH / NEUTRAL
print(result.cross_asset_regime)   # RISK_ON_ALIGNED / DECOUPLED_BULL / ...
All original fields (regime, ai_score, entry, stop_loss, etc.) remain unchanged.

New Google Sheets (auto‑created)
InstitutionalSignals – full institutional snapshot per run

RegimeDashboard – daily regime history

FlowSnapshot – derivatives flow history

LINE Alerts v2
python
from alerts.line_alert_v2 import send_institutional_alert

send_institutional_alert(
    symbol="BTC", price=65000, result_v2=result,
    min_conviction=60.0,   # adjust after calibration
)
Alerts fire on:

Approved trades with conviction ≥ threshold

Short/long squeezes

CRISIS liquidity regime

Architecture Overview
text
OHLCV DataFrame
      │
      ├─► Markov HMM       (32%) ─────────────────────────────┐
      ├─► Liquidity Engine (10%) [Phase 12]                    │
      ├─► Market Breadth   (6%)  [Phase 14]                    ├─► Regime Ensemble v2
      ├─► Cross-Asset      (4%)  [Phase 21] ──────────────────┘   (regime + confidence)
      │
      ├─► Regime Persistence      [Phase 15]  (sojourn time)
      ├─► Flow Engine             [Phase 13]  (funding + OI + L/S)
      ├─► Bayesian Reliability    [Phase 16]  (prevents RSI dominance)
      ├─► Forecast Engine         [Phase 17]  (XGBoost/LightGBM)
      └─► Conviction Engine       [Phase 18]  (8‑signal → position size)
Migration from v1.4
See MIGRATION.md for the full 8‑phase deployment guide.

TL;DR – cut‑over in one line:

python
# OLD
# from core.futures_orchestrator import FuturesOrchestrator
# result = FuturesOrchestrator().run(symbol, df)

# NEW
from core.futures_orchestrator_v2 import FuturesOrchestrator_v2
result = FuturesOrchestrator_v2().run(symbol, df)
Rollback: change the import back – original pipeline untouched.

Environment Variables (unchanged from v1.4)
Variable	Description
LINE_TOKEN	LINE Channel Access Token
SHEET_ID	Google Sheets document ID
GOOGLE_CREDENTIALS	Service account JSON (string or file path)
No new API keys required – all macro data from yfinance and OKX public endpoints.

Running Tests
bash
pytest tests/ -v                    # 175 tests, no network
pytest tests/ -m unit               # fast unit tests
pytest tests/ -m integration        # mocked integration tests
File Structure (v2 additions highlighted)
text
TradeAnalyze_V1.4/
├── engines/
│   ├── liquidity_regime.py         ← NEW
│   ├── market_breadth.py           ← NEW
│   ├── regime_persistence.py       ← NEW
│   ├── bayesian_reliability.py     ← NEW
│   ├── forecast_engine.py          ← NEW
│   ├── conviction_engine.py        ← NEW
│   ├── cross_asset_engine.py       ← NEW
│   ├── regime_ensemble_v2.py       ← NEW
│   └── ... (original engines unchanged)
├── crypto/
│   ├── flow_engine.py              ← NEW
│   └── ... (original unchanged)
├── options/
│   ├── greeks_advanced.py          ← NEW
│   └── ... (original unchanged)
├── portfolio/
│   ├── optimizer.py                ← NEW
│   └── ... (original unchanged)
├── core/
│   ├── futures_orchestrator_v2.py  ← NEW
│   └── futures_orchestrator.py     (unchanged)
├── report/
│   ├── dashboards.py               ← NEW
│   └── daily_report.py             (unchanged)
├── reports/
│   └── sheet_writer_v2.py          ← NEW
├── alerts/
│   └── line_alert_v2.py            ← NEW
├── tests/                          (all v2 tests added)
├── requirements_v2.txt
├── MIGRATION.md
├── ARCHITECTURE.md
└── README.md                       (this file)
Dependencies Added for v2
scipy (HRP clustering)

xgboost / lightgbm (optional – falls back to statistical forecast)

scikit-learn (feature scaling)

All original dependencies remain (yfinance, pandas, numpy, hmmlearn, gspread, …).

License & Support
This is the institutional upgrade of TradeAnalyze – built for production use.
For deployment details, see MIGRATION.md and ARCHITECTURE.md.
