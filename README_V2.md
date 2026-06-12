# TradeAnalyze v2.0 — Institutional Quant Platform

> Upgrade from Advanced Retail Quant → Institutional Quant Platform
> Built on top of TradeAnalyze v1.4 — fully backward compatible

[![Tests](https://img.shields.io/badge/tests-175%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()

---

## What Was Upgraded

TradeAnalyze v1.4 was a production retail quant system. Version 2.0 adds 10 institutional-grade engines as a second analytical layer — **without changing a single line of existing code.**

**Original system (v1.4) — preserved unchanged:**
Markov Regime · Regime Ensemble · Volatility Regime · Market Structure · Bayesian Engine · AI Score · Monte Carlo · Kelly Sizing · Volume Profile · Anchored VWAP · Options Layer · Portfolio Layer · Google Sheets · LINE Alerts

**New institutional layer (v2.0):**

| Phase | Engine | File | Purpose |
|-------|--------|------|---------|
| 12 | Liquidity Regime | `engines/liquidity_regime.py` | Global macro liquidity — DXY / VIX / Yields / TLT |
| 13 | Flow Engine | `crypto/flow_engine.py` | Funding + OI + Long/Short + Liquidation clusters |
| 14 | Market Breadth | `engines/market_breadth.py` | Crypto + equity participation breadth |
| 15 | Regime Persistence | `engines/regime_persistence.py` | Markov sojourn time — how long regime will last |
| 16 | Bayesian Reliability | `engines/bayesian_reliability.py` | Regime-conditional weighting — stops RSI from overriding |
| 17 | Forecast Engine | `engines/forecast_engine.py` | XGBoost/LightGBM 5d/10d/20d return prediction |
| 18 | Conviction Engine | `engines/conviction_engine.py` | 8-signal aggregator → position size gate |
| 19 | Advanced Greeks | `options/greeks_advanced.py` | Vanna · Charm · Vomma · Veta · Speed + full options card |
| 20 | Portfolio Optimizer | `portfolio/optimizer.py` | Risk Parity + Hierarchical Risk Parity (HRP) |
| 21 | Cross-Asset Engine | `engines/cross_asset_engine.py` | BTC vs SPY/QQQ/GLD/DXY/TLT correlation + regime |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements_v2.txt
```

Minimum: `numpy pandas scipy yfinance requests`
Optional (ML): `xgboost lightgbm` — falls back to statistical baseline if missing

### 2. Configure environment

```bash
cp .env.example .env
# Set LINE_TOKEN, SHEET_ID, GOOGLE_CREDS_FILE
```

### 3. Run tests

```bash
pytest tests/ -v
# 175 passed
```

### 4. Use the new orchestrator

```python
from core.futures_orchestrator_v2 import FuturesOrchestrator_v2
import yfinance as yf

df     = yf.download("BTC-USD", period="1y", interval="1d", auto_adjust=True)
result = FuturesOrchestrator_v2().run("BTC", df)

print(result.final_decision)      # LONG / SHORT / WAIT
print(result.conviction_tier)     # FULL SIZE / NORMAL SIZE / HALF SIZE / NO TRADE
print(result.conviction_score)    # 0–100
print(result.liquidity_regime)    # RISK_ON / RISK_OFF / CRISIS / RECOVERY
print(result.flow_regime)         # BULLISH_FLOW / SHORT_SQUEEZE / ...
print(result.forecast_direction)  # BULLISH / BEARISH / NEUTRAL
print(result.report_text)         # full report + institutional dashboards
```

---

## File Structure

```
TradeAnalyze_V1.4/
│
├── engines/
│   ├── liquidity_regime.py         Phase 12  ← NEW
│   ├── market_breadth.py           Phase 14  ← NEW
│   ├── regime_persistence.py       Phase 15  ← NEW
│   ├── bayesian_reliability.py     Phase 16  ← NEW
│   ├── forecast_engine.py          Phase 17  ← NEW
│   ├── conviction_engine.py        Phase 18  ← NEW
│   ├── cross_asset_engine.py       Phase 21  ← NEW
│   ├── regime_ensemble_v2.py       Integration ← NEW (v1 unchanged)
│   ├── regime_ensemble.py          ORIGINAL — unchanged
│   ├── bayesian_engine.py          ORIGINAL — unchanged
│   └── ... (all other originals)   ORIGINAL — unchanged
│
├── crypto/
│   ├── flow_engine.py              Phase 13  ← NEW
│   ├── funding_rate.py             ORIGINAL — unchanged
│   ├── open_interest.py            ORIGINAL — unchanged
│   └── liquidation_engine.py       ORIGINAL — unchanged
│
├── options/
│   ├── greeks_advanced.py          Phase 19  ← NEW
│   └── ... (all originals)         ORIGINAL — unchanged
│
├── portfolio/
│   ├── optimizer.py                Phase 20  ← NEW
│   └── ... (all originals)         ORIGINAL — unchanged
│
├── core/
│   ├── futures_orchestrator_v2.py  Integration ← NEW
│   └── futures_orchestrator.py     ORIGINAL — unchanged
│
├── report/
│   ├── dashboards.py               Integration ← NEW
│   └── daily_report.py             ORIGINAL — unchanged
│
├── reports/
│   └── sheet_writer_v2.py          Integration ← NEW (v1 unchanged)
│
├── alerts/
│   ├── line_alert_v2.py            Integration ← NEW (v1 unchanged)
│   └── line_alert.py               ORIGINAL — unchanged
│
├── tests/
│   ├── conftest.py                 shared fixtures
│   ├── test_liquidity_regime.py    Phase 12 — 28 tests
│   ├── test_flow_breadth_persistence.py   Phases 13–15 — 42 tests
│   ├── test_bayesian_forecast_conviction.py  Phases 16–18 — 48 tests
│   ├── test_options_portfolio_crossasset.py  Phases 19–21 — 42 tests
│   └── test_integration_smoke.py   end-to-end mocked — 15 tests
│
├── requirements_v2.txt
├── pytest.ini
├── .env.example
├── MIGRATION.md                    8-phase deployment guide
└── ARCHITECTURE.md                 data contracts + flow diagram
```

---

## Architecture

```
OHLCV DataFrame
      │
      ├─► Markov HMM       (32%) ─────────────────────────────┐
      ├─► Liquidity Engine (10%) [Phase 12]                    │
      │   DXY · VIX · Yields · TLT vol                        ├─► Regime Ensemble v2
      ├─► Market Breadth   (6%)  [Phase 14]                    │   → regime + confidence
      │   BTC.D · TOTAL3 · A/D · % above 200DMA              │
      ├─► Cross-Asset      (4%)  [Phase 21] ──────────────────┘
      │   BTC vs SPY/QQQ/GLD/DXY/TLT
      │
      ├─► Regime Persistence      [Phase 15]
      │   Sojourn time · half-life · exit probability
      │
      ├─► Flow Engine             [Phase 13]
      │   Funding · OI Delta · L/S Ratio · Liquidation Clusters
      │
      ├─► Bayesian Reliability    [Phase 16]
      │   RSI in STRONG_BEAR gets 0.15 reliability weight
      │   Prevents single-indicator dominance
      │
      ├─► Forecast Engine         [Phase 17]
      │   XGBoost/LightGBM · 15 features · walk-forward
      │   → 5d / 10d / 20d return + probability_up
      │
      └─► Conviction Engine       [Phase 18]
          8 signals · 0–100 score
          ┌──────────────────────────────────────┐
          │ 90–100 → FULL SIZE    (1.00× Kelly)  │
          │ 70–89  → NORMAL SIZE  (0.75× Kelly)  │
          │ 50–69  → HALF SIZE    (0.50× Kelly)  │
          │  < 50  → NO TRADE                    │
          └──────────────────────────────────────┘
```

---

## Conviction Engine — Signal Weights

| Signal | Weight | Source |
|--------|--------|--------|
| Regime quality | 25% | Ensemble regime + confidence |
| Trend alignment | 15% | EMA + market structure |
| Flow | 15% | Funding + OI + L/S ratio |
| Structure clarity | 10% | BOS/CHoCH + swing quality |
| Volatility | 10% | Vol regime suitability |
| Breadth | 10% | Crypto + equity participation |
| Liquidity | 10% | Global macro environment |
| Forecast | 5% | ML return direction |

**Persistence discount:** If regime is `EXHAUSTED`, conviction −15 points.
**Crisis override:** `CRISIS` liquidity → all sizing capped at 0.25× regardless of conviction.

---

## New Output Fields

```python
result.liquidity_regime       # "RISK_ON" | "RISK_OFF" | "CRISIS" | "RECOVERY"
result.liquidity_score        # 0–100
result.liquidity_risk_mult    # 0.25–1.25

result.flow_regime            # "BULLISH_FLOW" | "SHORT_SQUEEZE" | "LONG_SQUEEZE" | ...
result.flow_score             # 0–100
result.flow_direction         # "BULLISH" | "BEARISH" | "NEUTRAL"

result.breadth_regime         # "STRONG_BULL" | "BULL" | "NEUTRAL" | "BEAR" | "STRONG_BEAR"
result.breadth_score          # 0–100

result.persistence_label      # "ESTABLISHED" | "MATURING" | "FRESH" | "EXHAUSTED"
result.remaining_days         # estimated days left in current regime
result.exit_prob_7d           # % probability of regime change within 7 days

result.forecast_direction     # "BULLISH" | "BEARISH" | "NEUTRAL"
result.forecast_20d_return    # expected % return over 20 days
result.forecast_confidence    # 0–100

result.conviction_score       # 0–100
result.conviction_tier        # "FULL SIZE" | "NORMAL SIZE" | "HALF SIZE" | "NO TRADE"
result.conviction_kelly_mult  # 0.0 | 0.50 | 0.75 | 1.00

result.portfolio_vol          # annualised portfolio volatility %
result.portfolio_drawdown     # max drawdown estimate %
result.portfolio_sharpe       # Sharpe ratio

result.cross_asset_regime     # "RISK_ON_ALIGNED" | "DECOUPLED_BULL" | "TRANSITION" | ...
result.btc_rs_score           # BTC relative strength vs asset basket 0–100
result.btc_beta_spy           # BTC rolling beta to SPY
```

---

## Google Sheets — New Worksheets

Auto-created on first run. Original sheets untouched.

| Worksheet | Contents |
|-----------|----------|
| `InstitutionalSignals` | Full snapshot — all 50+ new fields per run |
| `RegimeDashboard` | Daily regime + conviction + persistence history |
| `FlowSnapshot` | Derivatives flow — funding / OI / L/S / cascade |

```python
from reports.sheet_writer_v2 import write_all_institutional

write_all_institutional(
    symbol="BTC", price=65000.0, result_v2=result,
    liquidity=liquidity_result, flow=flow_result,
    breadth=breadth_result, conviction=conviction_result,
)
```

---

## LINE Alerts v2

Conviction-filtered alerts — no noise from low-quality signals.

```python
from alerts.line_alert_v2 import send_institutional_alert

send_institutional_alert(
    symbol="BTC", price=65000.0,
    result_v2=result,
    liquidity=liquidity_result,
    flow=flow_result,
    conviction=conviction_result,
    min_conviction=60.0,    # tune as needed
)
```

Alert fires when:
- Approved trade with `conviction_score ≥ min_conviction`
- `flow_regime in (SHORT_SQUEEZE, LONG_SQUEEZE)` — always alerts
- `liquidity_regime == CRISIS` — always alerts with crisis broadcast

---

## Advanced Options Greeks (Phase 19)

```python
from options.greeks_advanced import build_options_recommendation

rec = build_options_recommendation(
    strategy="SELL_PUT", strike=48000, dte=30,
    direction="SHORT", price=50000, iv=0.70,
    premium=500, max_profit=500, max_loss=4500,
    breakeven=47500, option_type="put",
)

# Full institutional card
print(rec.summary_line)
# SELL_PUT | K=48000.00 | 30DTE | POP=72% | EV=+185.20 |
# δ=-0.285 θ=-12.4500 ν=0.0821 | Kelly=0.041

# Second-order Greeks (new)
rec.vanna   # delta sensitivity to IV — hedging risk
rec.charm   # delta decay per day — monitoring overnight
rec.vomma   # vega convexity — vol-of-vol exposure
```

---

## Portfolio Optimizer (Phase 20)

```python
from portfolio.optimizer import compute_portfolio_optimization
import yfinance as yf

prices = {t: yf.download(f"{t}-USD", period="1y")["Close"]
          for t in ["BTC", "ETH", "SOL", "BNB"]}

result = compute_portfolio_optimization(prices, method="hrp")
# method="risk_parity" | "hrp" | "auto"

print(result.recommended_weights)   # {BTC: 0.38, ETH: 0.28, ...}
print(result.risk_contributions)    # % of total vol per asset
print(result.portfolio_volatility)  # annualised %
print(result.diversification_ratio) # weighted avg vol / portfolio vol
```

---

## Safety Design

Every new engine follows this contract:

- Wrapped in `_safe()` in orchestrator — failure returns `None`, never raises
- All field reads use `getattr(result, field, default)` — `None` safe
- Dashboard sections show `"Data unavailable"` if engine returned `None`
- LINE alerts skip unavailable blocks gracefully
- Sheets write `""` for missing values

**No single engine failure can crash the pipeline or block a trade signal.**

---

## Running Tests

```bash
# All 175 tests — no API keys, no network needed
pytest tests/ -v

# By phase
pytest tests/test_liquidity_regime.py -v           # Phase 12
pytest tests/test_flow_breadth_persistence.py -v   # Phases 13–15
pytest tests/test_bayesian_forecast_conviction.py -v  # Phases 16–18
pytest tests/test_options_portfolio_crossasset.py -v  # Phases 19–21
pytest tests/test_integration_smoke.py -v          # end-to-end mocked

# By marker
pytest tests/ -m unit         # pure unit (no I/O)
pytest tests/ -m integration  # mocked integration
```

---

## Migration from v1.4

See [`MIGRATION.md`](MIGRATION.md) for the full 8-phase deployment guide with rollback procedure.

**TL;DR (3 steps):**

```python
# 1. Install
# pip install -r requirements_v2.txt

# 2. Replace orchestrator
from core.futures_orchestrator_v2 import FuturesOrchestrator_v2
result = FuturesOrchestrator_v2().run(symbol, df)

# 3. Add new outputs
from reports.sheet_writer_v2 import write_all_institutional
write_all_institutional(symbol, price, result)

from alerts.line_alert_v2 import send_institutional_alert
send_institutional_alert(symbol, price, result, min_conviction=60.0)
```

**Rollback:** change one import back to `FuturesOrchestrator`. Original system untouched.

---

## New Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `scipy` | ≥1.11 | HRP hierarchical clustering |
| `xgboost` | ≥2.0 | Forecast Engine (optional) |
| `lightgbm` | ≥4.3 | Forecast Engine fallback (optional) |
| `scikit-learn` | ≥1.4 | Feature scaling (optional) |

All other packages already in v1.4 requirements.

---

## Documents

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — Full data contracts for all 10 new engines, execution flow, weight tables
- [`MIGRATION.md`](MIGRATION.md) — 8-phase deployment plan, rollback, known limitations
- [`.env.example`](.env.example) — All environment variables with descriptions

---

*TradeAnalyze v2.0 — Institutional Quant Platform*
