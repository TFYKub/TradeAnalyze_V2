TradeAnalyze V.3 – Architecture Summary
1. High‑Level Overview
TradeAnalyze V.3 is a modular, event‑driven trading system that processes daily OHLCV data for each symbol independently. It consists of four layers:

Data Layer – fetches market data, option chains, and macro instruments.

Analysis Pipeline – runs 20+ independent engines (regime, structure, liquidity, flow, forecast, conviction, etc.).

Decision Layer – applies gates (AI score, EV, RR, Kelly, MC) and produces a final trade decision.

Output Layer – sends notifications and writes to Google Sheets.

All engine calls are wrapped in _safe() – a failure in any component never crashes the pipeline.

2. Core Modules & Responsibilities
Module	Responsibility
main.py	Entry point. Loads symbols, loops over them, calls orchestrator.
core/futures_orchestrator_v2.py	V2 pipeline (19 steps + 10 institutional steps). Returns FuturesResult_v2.
core/futures_orchestrator_v3.py	Extends V2 with signal tracking, adaptive ensemble, regime‑switching MC.
regime/markov.py	Gaussian HMM with 5 states. Returns transition matrix and regime probabilities.
engines/regime_ensemble_v2.py	Combines Markov, trend, volatility, macro, liquidity, breadth, cross‑asset into a single regime.
engines/liquidity_regime.py	Global liquidity (VIX, DXY, yields, TLT) → RISK_ON / OFF / CRISIS / RECOVERY.
engines/market_breadth.py	Crypto + equity breadth → STRONG_BULL … STRONG_BEAR.
crypto/flow_engine.py	Aggregates funding, OI, L/S ratio, liquidation clusters → flow score & regime.
engines/conviction_engine.py	8‑signal weighted average → tier & Kelly multiplier.
engines/forecast_engine.py	ML forecast (XGBoost/LightGBM) for 5d/10d/20d returns & probabilities.
portfolio/optimizer.py	Risk parity & HRP for multi‑asset portfolios.
options/options_orchestrator.py	Volatility → expected move → POP (MC) → strategy selection (rules + composite) → approval gate.
options/greeks_advanced.py	Black‑Scholes with second‑order Greeks (vanna, charm, vomma, veta, speed).
alerts/notification_manager.py	Telegram primary, LINE fallback (rate‑limited, monthly quota detection).
reports/sheet_writer_v2.py	Writes institutional dashboards to Google Sheets.
reports/execution_report_v3.py	Phone‑optimised unified report (futures + options).
3. Data Flow (Simplified)
text
Symbol list → get_market_data (yfinance)
       ↓
FuturesOrchestrator_v2/v3.run()
  ├─ Indicators (EMA, RSI, ATR)
  ├─ Markov + Calibration
  ├─ Liquidity Regime (Phase 12)
  ├─ Market Breadth (Phase 14)
  ├─ Cross‑Asset Engine (Phase 21)
  ├─ Regime Ensemble v2
  ├─ Regime Persistence (Phase 15)
  ├─ Volatility Regime
  ├─ Structure / S/R / Entry
  ├─ Institutional Stop
  ├─ AI Score & Trade Quality
  ├─ Flow Engine (Phase 13)
  ├─ Bayesian v2 (Phase 16)
  ├─ Forecast Engine (Phase 17)
  ├─ Conviction Engine (Phase 18)
  ├─ Monte Carlo & Consistency
  ├─ Position Sizing (Kelly)
  ├─ Portfolio Optimizer (Phase 20)
  ├─ Final Decision (7 gates)
  ├─ Volume Profile + AVWAP
  └─ Report + Dashboards
       ↓
OptionsOrchestrator
  ├─ Volatility engine (IV, HV, IV rank)
  ├─ Expected move
  ├─ Monte Carlo POP for 14 strategies
  ├─ Rule engine + composite ranking
  ├─ Strike selection from live chain
  └─ Approval: EV>0, POP≥55, AI≥60, score≥60
       ↓
Output: Telegram / LINE alerts + Google Sheets
4. V3 Specific Components
Phase	Module	Description
1	walk_forward.py + adaptive_ensemble.py	Rolling OOS validation, dynamic engine weights (SQLite)
2	quantile_forecast.py	LightGBM quantile regression (5/10/20d)
3	regime_switching_mc.py	GBM with regime‑dependent volatility
4	portfolio_optimizer_v3.py	Risk parity, min variance, max Sharpe, Kelly portfolio
5	crypto_flow_v2.py	OI delta, funding momentum, liquidation pressure, whale flow
6	dealer_greeks.py	Gamma exposure, vanna, charm, pin‑risk
7	stress_testing.py	Historical & custom scenario stress tests
5. Configuration & Environment
Variable	Purpose
SHEET_ID	Google Sheet ID for logging
GOOGLE_CREDENTIALS	JSON string or path to service account key
LINE_TOKEN	LINE Messaging API channel access token
TELEGRAM_BOT_TOKEN	Telegram bot token
TELEGRAM_CHAT_ID	Telegram chat ID
USE_V3	Boolean – enable V3 phases (default False)
V3_PHASES	Dict – enable/disable individual V3 sub‑phases
All thresholds (MIN_RR, MIN_AI_SCORE, etc.) are in config/thresholds.py.

6. Error Handling & Resilience
External calls (yfinance, Google Sheets, LINE API, Deribit) are wrapped in retries.

Every engine call inside the orchestrator is wrapped in _safe() – returns default on exception, logs error, continues.

LINE quota detection – if API responds with "monthly limit exceeded", LINE_DISABLED is set globally.

Google Sheets failures are logged but do not stop the trading loop.

7. Testing
Run with pytest tests/. Test files cover phases 12–21 using mock fixtures (no live network calls).
Key test files: test_bayesian_forecast_conviction.py (16‑18), test_options_portfolio_crossasset.py (19‑21), test_flow_breadth_persistence.py (13‑15), test_liquidity_regime.py (12), test_integration_smoke.py (end‑to‑end wiring).

8. Extending the System
Adding a new engine:

Create a file in engines/ with a function returning a dataclass.

Add a step in futures_orchestrator_v2.py inside run() (wrapped in _safe()).

Extend FuturesResult_v2 with new fields.

Update sheet_writer_v2.py and line_alert_v2.py to display/log the new data.

Write unit tests.

Adding a new option strategy:

Add payoff logic to options/strategy_models.py.

Add builder function to options/strategy_definitions.py.

Register in STRATEGY_BUILDERS and rule engine (strategy_rules.py).

Update selection_engine.py to include it in candidate lists.

9. Version History
V1 – original pipeline (Markov, structure, AI score, basic stop/TP).

V2 – added phases 12–21 (liquidity, breadth, flow, persistence, forecast, conviction, portfolio, cross‑asset).

V3 – added walk‑forward, quantile forecast, regime‑switching MC, portfolio optimiser V3, crypto flow V2, dealer Greeks, stress testing.

For full details, see the source code and inline comments.