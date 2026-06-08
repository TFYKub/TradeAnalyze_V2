# TradeAnalyze — Institutional Trading Engine

Automated daily trading-signal engine for retail and institutional-style investors.

## Architecture (11 Phases)

```
Phase 1  — Logic Fixes          config/thresholds.py, risk/stop_engine.py,
                                 market_structure/structure_consistency.py,
                                 regime/markov_calibration.py,
                                 risk/consistency_checker.py

Phase 2  — Regime Ensemble      engines/regime_ensemble.py
Phase 3  — Volatility Regime    engines/volatility_regime.py
Phase 4  — Trade Quality        engines/trade_quality.py
Phase 5  — Options Layer        options/iv_rank.py, options/vol_surface.py,
                                 options/strategy_selector.py
Phase 6  — Futures Layer        engines/volume_profile.py, engines/anchored_vwap.py
Phase 7  — Crypto Layer         crypto/funding_rate.py, crypto/open_interest.py,
                                 crypto/liquidation_engine.py
Phase 8  — Portfolio Layer      portfolio/correlation_engine.py, portfolio/risk_budget.py
Phase 9  — Bayesian Engine      engines/bayesian_engine.py
Phase 10 — Walk-Forward         research/walk_forward.py
Phase 11 — Performance Attrib   analytics/performance_attribution.py
```

## Directory Structure

```
config/          Env vars, logging, thresholds (centralised)
core/            FuturesOrchestrator (19-step) + OptionsOrchestrator
data/            Market data, option chain fetcher
engines/         Regime ensemble, vol regime, trade quality, Bayesian,
                 volume profile, anchored VWAP
indicators/      EMA, RSI, ATR
market_structure/ Swing detector, structure break, S/R, consistency
options/         IV rank, vol surface, strategy engine, selection
regime/          Markov HMM, calibration
risk/            Stop engine (4 types), position sizing, consistency checker
signals/         Trend filter, divergence, entry engine, final decision
simulation/      Monte Carlo (10k), portfolio risk
ai/              AI scoring engine (5-component)
crypto/          Funding rate, open interest, liquidation zones
portfolio/       Correlation matrix, risk budget
analytics/       Performance attribution tracker
research/        Walk-forward validation
report/          Full institutional report builder
reports/         Google Sheets writers (TradeSignals, Options, Option_Chain)
alerts/          LINE Messaging API broadcast
```

## Required env vars

| Variable             | Description                          |
|----------------------|--------------------------------------|
| `SHEET_ID`           | Google Sheets document ID            |
| `GOOGLE_CREDENTIALS` | Service account JSON (stringified)   |
| `LINE_TOKEN`         | LINE Channel Access Token            |

## SYMBOL_CONFIG sheet columns

| symbol | group | asset_type |
|--------|-------|-----------|
| AAPL   | LINE  | stock     |
| BTC    | LINE  | crypto    |

## Running

```bash
pip install -r requirements.txt
export SHEET_ID=...
export GOOGLE_CREDENTIALS='{"type":"service_account",...}'
export LINE_TOKEN=...
python main.py
```

## Trade Decision Gates (7)

| Gate | Threshold         |
|------|-------------------|
| Regime Confidence | ≥ 60% |
| AI Score          | ≥ 70  |
| Expected Value    | > 0   |
| Kelly Fraction    | > 0   |
| MC P(Profit)      | ≥ 60% |
| Risk Reward       | ≥ 1.5 |
| EMA + Structure   | aligned |

## GitHub Actions

Runs daily at 07:00 Bangkok time (UTC 00:00) via `.github/workflows/daily.yml`.
