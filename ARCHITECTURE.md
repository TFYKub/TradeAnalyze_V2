Architecture Overview
See ARCHITECTURE.md for a detailed breakdown of modules, data flow, and extension points.

Monitoring
Health endpoint: http://localhost:8080/health

Prometheus metrics: http://localhost:8080/metrics

Watchdog: runs via cron (see scripts/watchdog_check.py)

Testing
Run the test suite:

bash
pytest tests/
License
MIT

Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Support
For questions or issues, please open a GitHub issue.

text

---

# ARCHITECTURE.md

```markdown
# TradeAnalyze V3.0 – Architecture

## Overview

TradeAnalyze is a modular, event‑driven trading system that processes symbols one by one, producing a comprehensive institutional report for each. The architecture is designed for **reliability**, **extensibility**, and **transparency** – every decision is traceable to its constituent signals.

The system is structured as a **pipeline** with clear separation of concerns:

1. **Data Acquisition** – fetches OHLCV and option chains.
2. **Regime & Structure** – identifies market state and trend.
3. **Entry & Risk** – computes stop losses, position sizing, Monte Carlo.
4. **Scoring & Decision** – aggregates signals into AI Score, Trade Quality, Conviction.
5. **Options** – full institutional options analysis.
6. **Persistence & Reporting** – writes to SQLite, Google Sheets, and sends alerts.

---

## High‑Level Data Flow
┌─────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ SYMBOL │─────▶│ Orchestrator │─────▶│ Engines │
│ (from GS) │ │ (V2 or V3) │ │ (Regime, Flow, │
└─────────────┘ └─────────────────┘ │ Conviction, │
│ Forecast, …) │
└─────────────────┘
│
▼
┌─────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Google │◀─────│ Reports │◀─────│ Risk & │
│ Sheets │ │ (Sheet writers,│ │ Options │
│ SQLite │ │ formatters) │ │ Analysis │
└─────────────┘ └─────────────────┘ └─────────────────┘

text

---

## Core Modules

### 1. `core/` – Orchestrators & State

- **`futures_orchestrator_v2.py`** – Main pipeline: 19 steps (regime → structure → entry → scoring → risk → decision). Returns `FuturesResult_v2`.
- **`futures_orchestrator_v3.py`** – Extends V2 with V3 enhancements (walk‑forward, quantile forecast, regime‑switching MC, portfolio optimisation). Adds `TradeStateV3`.
- **`trade_state.py`** – Holds all V3‑specific state.

### 2. `engines/` – Analysis Engines

| Module | Purpose |
|--------|---------|
| `conviction_engine.py` | Aggregates 8 signals into a single conviction score & tier. |
| `forecast_engine.py` | ML (XGBoost/LightGBM) for 5/10/20‑day returns. |
| `liquidity_regime.py` | Macro regime (VIX, DXY, yields, TLT vol). |
| `market_breadth.py` | Crypto + equity breadth composite. |
| `cross_asset_engine.py` | BTC vs SPY/QQQ/DXY/GLD/TLT – relative strength & decoupling. |
| `regime_persistence.py` | Markov‑based regime sojourn time & exit probability. |
| `bayesian_reliability.py` | Regime‑conditional reliability weighting for Bayesian signals. |
| `regime_ensemble_v2.py` | 7‑component ensemble (Markov, trend, vol, macro, liquidity, breadth, cross‑asset). |
| `volatility_regime.py` | Classifies LOW/NORMAL/HIGH/PANIC vol. |
| `trade_quality.py` | Grades trade A+ … REJECT. |
| `signal_tracker.py` | Stores predictions for walk‑forward evaluation. |
| `adaptive_ensemble.py` | Dynamically re‑weights engines based on historical accuracy. |

### 3. `options/` – Full Options Suite

| Module | Purpose |
|--------|---------|
| `options_orchestrator.py` | Coordinates volatility, expected move, probability, strategy selection, approval. |
| `volatility_engine.py` | IV, IV Rank, HV20, HV60, ATR14, IV/HV ratio. |
| `expected_move_engine.py` | 1σ / 1.5σ / 2σ bands. |
| `probability_engine.py` | Monte Carlo (10k paths) for strategy POPs. |
| `selection_engine.py` | Rule‑based + composite scoring → top 3 strategies. |
| `strike_selector.py` | Picks strikes from live chain (or estimated). |
| `strategy_models.py` | Payoff calculations for 14 strategies. |
| `greeks_engine.py` | Black‑Scholes Greeks (delta, gamma, theta, vega, rho). |
| `vol_surface.py` | Put/call skew, smile, term structure. |
| `iv_rank.py` | IV Rank & Percentile. |

### 4. `crypto/` – Crypto Derivatives

- **`funding_rate.py`** – Fetch from Deribit/OKX, classify crowded longs/shorts.
- **`open_interest.py`** – Price × OI divergence (confirmation, continuation, weak rally, capitulation).
- **`liquidation_engine.py`** – Estimates long/short liquidation clusters and cascade risk.
- **`flow_engine.py`** – Aggregates funding, OI, L/S ratio, and liquidation into a directional flow score.

### 5. `portfolio/` – Optimisation & Risk

- **`optimizer.py`** – Risk Parity and Hierarchical Risk Parity (HRP). Single‑asset mode returns vol and drawdown.
- **`correlation_engine.py`** – (Currently unused in production; kept for future integration.)
- **`risk_budget.py`** – (Not used; can be removed.)

### 6. `risk/` – Stop & Position Sizing

- **`stop_engine.py`** – 4‑type institutional stop (ATR, structure, swing, volatility) with selection.
- **`stop_loss_engine.py`** – Basic SL/TP with swing + ATR.
- **`position_sizing.py`** – Kelly / half‑Kelly with regime risk multiplier.
- **`consistency_checker.py`** – Monte Carlo + signal consistency.

### 7. `simulation/` – Risk Metrics

- **`monte_carlo.py`** – GBM with cached vol parameters; returns P(profit), P(stop), P(target), VaR, CVaR.
- **`portfolio_risk.py`** – Historical VaR, CVaR, max drawdown, Sharpe, Sortino, Calmar.

### 8. `data/` – Acquisition

- **`market_data.py`** – yfinance OHLCV with technical indicators.
- **`option_chain.py`** – yfinance for stocks, Deribit for crypto.

### 9. `reports/` – Output

- **`sheet_writer_v2.py`** – Batched Google Sheets writer for institutional signals, regime dashboard, flow snapshot.
- **`sheet_writer.py`** – Legacy (TradeSignals, Options).
- **`option_chain_writer.py`** – Writes enriched option chain to Google Sheets.
- **`options_sheet_writer.py`** – Writes Options_Analysis sheet.
- **`execution_report_v3.py`** – Phone‑optimised execution report.
- **`dashboards.py`** – Institutional dashboards (phases 12–21) appended to daily report.
- **`formatter.py`** – (Legacy; not used anymore.)
- **`options_formatter.py`** – (Legacy; not used anymore.)

### 10. `persistence/` – SQLite

- **`trade_persistence.py`** – Active trades, closed trades, engine signals. Thread‑local connections.

### 11. `alerts/` – Notifications

- **`telegram_alert.py`** – Primary channel with retries and backoff.
- **`line_alert_v2.py`** – Fallback with rate limiting, monthly quota detection, and message splitting.
- **`notification_manager.py`** – Multi‑channel dispatcher.

### 12. `monitoring/` – Observability

- **`health_server.py`** – Flask server with `/health`, `/metrics` (Prometheus), `/ready`.
- **`metrics.py`** – Convenience functions for updating Prometheus gauges.
- **`watchdog.py`** – Checks timestamp file; sends alert on stall.
- **`daily_summary.py`** – Aggregates daily trades and sends report.

### 13. `analytics/` – Performance Attribution

- **`performance_attribution.py`** – Brinson‑style attribution per engine (Markov, Trend, Options, Risk, Volatility). Tracks rolling Sharpe, Sortino, win rate.

### 14. `config/` – Configuration

- **`config.py`** – Feature flags (`USE_V3`, `USE_PARALLEL`, `MAX_WORKERS`, V3 phase toggles).
- **`thresholds.py`** – All trading thresholds (MIN_RR, MIN_AI_SCORE, MAX_KELLY, etc.).
- **`logging_config.py`** – JSON logging with rotation.

### 15. `utils/` – Utilities

- **`retry.py`** – Retry decorator with exponential backoff.
- **`sheets_auth.py`** – Google Sheets authentication (gspread).
- **`symbol_loader.py`** – Loads symbols from Google Sheets.
- **`batch_writer.py`** – Batches rows per sheet and writes once per run.

---

## Data Contracts

Most modules return **frozen dataclasses** with clear fields. Key examples:

- **`FuturesResult_v2`** – All signals (regime, decision, entry, stop, TP, scores, risk metrics).
- **`EnsembleRegimeResult`** – Final regime, confidence, component breakdown.
- **`ConvictionResult`** – Score, tier, Kelly multiplier, component scores.
- **`ForecastResult`** – 5/10/20 day returns, probabilities, confidence.
- **`LiquidityRegimeResult`** – Macro regime, risk multiplier, VIX/DXY/Yield scores.
- **`MarketBreadthResult`** – Crypto + equity breadth composite.
- **`CrossAssetResult`** – BTC relative strength, beta to SPY/QQQ, decoupling detection.
- **`PortfolioOptimizationResult`** – Weights, risk contributions, volatility, Sharpe.
- **`OptionsRecommendation`** – Primary strategy, top 3, volatility, expected move, approval.
- **`MonteCarloResult`** – P(profit), P(stop), P(target), VaR, CVaR.

---

## Error Handling Philosophy

Every engine is wrapped in `_safe()` inside the orchestrator, so a failure in one component **never crashes the entire pipeline**. Fallbacks provide sensible defaults (e.g., neutral scores, zero values).  
External API calls (yfinance, Deribit, OKX) are **retried with exponential backoff** and fallback to alternative sources.

---

## Extension Points

To add a new engine:
1. Create a new module in `engines/` returning a frozen dataclass.
2. Import it in the orchestrator and call it within the `run()` method.
3. Add the result fields to `FuturesResult_v2` (and optional `TradeStateV3`).
4. Update sheet writers and formatters if needed.

All new engines should be safe (handle `None` inputs) and follow the existing naming conventions.

---

## Database Schema

### `active_trades`
- Stores open positions with entry snapshot (JSON).

### `trade_history`
- Closed trades with P&L, exit time.

### `engine_signals`
- Tracks every prediction for walk‑forward evaluation.

---

## Monitoring & Health

- **Health server** – runs on port 8080 (daemon thread).
- **Prometheus metrics** – trade counts, win rate, portfolio vol/drawdown, API latency.
- **Watchdog** – checks timestamp file; sends alert if bot stalls > 1 hour.
- **Daily summary** – sends a performance report if trades were closed today.

---

## Testing

Tests reside in `tests/` and cover all V2/V3 phases (12–21). They use synthetic data and mock objects. Run with `pytest tests/`.

---

## Contributing Guidelines

1. Follow the existing code style (PEP 8 with 4‑space indentation, descriptive variable names).
2. Add docstrings for all public functions and classes.
3. Write unit tests for new functionality.
4. Ensure all tests pass before submitting a PR.

---

## License

MIT – see [LICENSE](LICENSE) for details.