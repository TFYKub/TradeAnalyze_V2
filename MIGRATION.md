# TradeAnalyze Institutional Upgrade — Migration Plan
# Phases 12–21  |  v1.4 → v2.0

---

## Overview

This document describes how to safely migrate from the existing
TradeAnalyze v1.4 (Advanced Retail Quant) to the upgraded v2.0
(Institutional Quant Platform) without breaking any live alerts,
existing Google Sheets output, or running pipelines.

**Core principle:** Everything in v2 is additive. The original pipeline
(`FuturesOrchestrator`) remains untouched. The new orchestrator
(`FuturesOrchestrator_v2`) is a separate class that wraps and extends it.

---

## Phase 1 — Dependency Installation  (Day 1)

Install new packages in a staging environment first:

```bash
# Create isolated environment (recommended)
python -m venv venv_v2
source venv_v2/bin/activate

# Install all dependencies
pip install -r requirements_v2.txt

# Verify ML libraries
python -c "import xgboost; print('XGBoost OK:', xgboost.__version__)"
python -c "import lightgbm; print('LightGBM OK:', lightgbm.__version__)"
python -c "import scipy; print('SciPy OK:', scipy.__version__)"
```

**If XGBoost/LightGBM fail to install** (e.g. ARM build issues):
The `ForecastEngine` degrades gracefully to a statistical fallback.
You can proceed without them — `model_used` will show `"statistical_fallback"`.

---

## Phase 2 — Run Unit Tests  (Day 1)

```bash
cd /path/to/TradeAnalyze_V1.4
pytest tests/ -v --tb=short 2>&1 | tee test_results.txt
```

**Expected:** All tests in `tests/test_liquidity_regime.py`,
`test_flow_breadth_persistence.py`, `test_bayesian_forecast_conviction.py`,
`test_options_portfolio_crossasset.py` pass without network calls.

**Integration tests** (`test_integration_smoke.py`) use mocks — no API keys needed.

If any test fails related to imports, check that the new engine files are
present in the correct directories:

```bash
ls engines/liquidity_regime.py    # Phase 12
ls crypto/flow_engine.py          # Phase 13
ls engines/market_breadth.py      # Phase 14
ls engines/regime_persistence.py  # Phase 15
ls engines/bayesian_reliability.py # Phase 16
ls engines/forecast_engine.py     # Phase 17
ls engines/conviction_engine.py   # Phase 18
ls options/greeks_advanced.py     # Phase 19
ls portfolio/optimizer.py         # Phase 20
ls engines/cross_asset_engine.py  # Phase 21
```

---

## Phase 3 — Google Sheets Setup  (Day 1–2)

New worksheets must be created in the existing Google Sheet.
They are **auto-created** on first write by `_ensure_headers()`, but
pre-creating them avoids a rare race condition on first run.

**Add 3 new worksheets** to the Google Sheet (`SHEET_ID` in config):

| Worksheet Name        | Purpose                              |
|-----------------------|--------------------------------------|
| `InstitutionalSignals`| Full per-run snapshot of all engines |
| `RegimeDashboard`     | Compact daily regime history         |
| `FlowSnapshot`        | Derivatives flow history             |

Existing sheets (`TradeSignals`, `Options`, `MarketData`) are unchanged.

```python
# One-time setup script (run manually):
from utils.sheets_auth import get_sheets_client
from config.config import SHEET_ID

gc = get_sheets_client()
sh = gc.open_by_key(SHEET_ID)
for name in ["InstitutionalSignals", "RegimeDashboard", "FlowSnapshot"]:
    try:
        sh.add_worksheet(title=name, rows=5000, cols=60)
        print(f"Created: {name}")
    except Exception:
        print(f"Already exists: {name}")
```

---

## Phase 4 — Parallel Dry Run  (Days 2–4)

Run both orchestrators in parallel — v1 controls live alerts, v2 logs only.

```python
# In your main pipeline script, add alongside existing run():
from core.futures_orchestrator_v2 import FuturesOrchestrator_v2
from reports.sheet_writer_v2 import write_all_institutional

orch_v2 = FuturesOrchestrator_v2()

# Dry-run: log to sheets but don't send LINE alerts yet
result_v2 = orch_v2.run(symbol, df)
write_all_institutional(
    symbol=symbol, price=price, result_v2=result_v2,
    liquidity=None,   # pass actual engine results when available
    flow=None,
    ...
)
```

Monitor `InstitutionalSignals` sheet for 2–3 trading days.
Check:
- No crashes (pipeline completes for all symbols)
- `ConvictionTier` column looks reasonable (not all NO_TRADE or all FULL SIZE)
- `LiquidityRegime` matches current market conditions
- `FlowRegime` matches what you see on exchange dashboards manually

---

## Phase 5 — Enable LINE Alerts v2  (Day 4–5)

After dry-run validation, enable institutional LINE alerts.

```python
from alerts.line_alert_v2 import send_institutional_alert

# Replace or supplement existing LINE send in your pipeline:
send_institutional_alert(
    symbol=symbol, price=price,
    result_v2=result_v2,
    liquidity=liquidity_result,
    flow=flow_result,
    breadth=breadth_result,
    persistence=persistence_result,
    forecast=forecast_result,
    conviction=conviction_result,
    cross_asset=cross_asset_result,
    min_conviction=60.0,    # only alert on ≥60 conviction (tune as needed)
)
```

**Recommended settings for first week:**
- `min_conviction=65.0` — reduces noise while system calibrates
- Keep original `send_line_message()` calls for core signals as backup
- Tune down to `50.0` after 1 week of validation

---

## Phase 6 — Cut Over to Orchestrator v2  (Day 5–7)

Once LINE alerts are validated and sheets are clean:

```python
# OLD (keep as fallback):
# from core.futures_orchestrator import FuturesOrchestrator
# result = FuturesOrchestrator().run(symbol, df)

# NEW (drop-in replacement):
from core.futures_orchestrator_v2 import FuturesOrchestrator_v2
result = FuturesOrchestrator_v2().run(symbol, df)
```

The `FuturesResult_v2` returned by the new orchestrator contains all
original fields plus new institutional fields. Downstream code that reads
`result.regime`, `result.ai_score`, etc. is unaffected.

---

## Phase 7 — Conviction Calibration  (Week 2)

After 1 week of live data, calibrate conviction thresholds:

1. Export `InstitutionalSignals` sheet to CSV
2. Check distribution of `ConvictionScore` column
3. If most trades cluster at HALF SIZE (50–69): consider lowering `min_conviction`
4. If too many FULL SIZE: increase regime confidence thresholds in `conviction_engine.py`

**Bayesian Reliability tuning** (Phase 16):
Check `test_bayesian_forecast_conviction.py` — if RSI signals are still
overriding regime in practice, lower `RSI_RELIABILITY["BEAR"]` further
in `engines/bayesian_reliability.py`.

---

## Phase 8 — Forecast Model Warm-Up  (Week 2–4)

The `ForecastEngine` trains on historical data in-memory each run.
It requires `MIN_BARS = 120` days to produce ML forecasts.

- First 4 months of live data: statistical fallback (expected)
- After 120+ trading days: XGBoost/LightGBM models activate
- Watch `ModelUsed` column in `InstitutionalSignals` sheet

No action needed — fully automatic.

---

## Rollback Procedure

If any phase causes issues, rollback is one line change:

```python
# Rollback: switch back to original orchestrator
from core.futures_orchestrator import FuturesOrchestrator
result = FuturesOrchestrator().run(symbol, df)
```

Original sheets (`TradeSignals`, `Options`, `MarketData`) are untouched.
Original LINE alerts are untouched.
New sheets can be archived without affecting anything.

---

## Environment Variables (no changes required)

Existing env vars are unchanged:

```bash
LINE_TOKEN=...        # unchanged
SHEET_ID=...          # unchanged — same Google Sheet
GOOGLE_CREDS_FILE=... # unchanged
```

No new API keys are required for the institutional upgrade.
- DXY, VIX, TLT, equity ETFs → yfinance (free, no key)
- OKX L/S ratio → public endpoint (no auth)
- All regime/forecast computation → local/in-process

---

## Known Limitations & Mitigations

| Limitation | Impact | Mitigation |
|---|---|---|
| yfinance rate limits during market hours | DXY/VIX fetch may occasionally fail | `compute_liquidity_regime()` returns cached fallback |
| OKX API unavailable | L/S ratio missing → flow score uses 3 components | `_score_long_short(None)` returns neutral 50 |
| XGBoost not installed | No ML forecast | Statistical fallback activated automatically |
| < 120 bars of history | Forecast uses statistical method | `model_used = "statistical_fallback"` in output |
| Google Sheets quota | New writes (3 extra sheets) count against quota | Batch writes; consider `write_all_institutional` once per symbol |
