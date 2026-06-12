# TradeAnalyze v2.0 — Institutional Architecture & Data Contracts
# Phases 12–21 Complete Reference

---

## System Overview

```
                    ┌─────────────────────────────────────────────────────┐
                    │          FuturesOrchestrator_v2  (29 steps)         │
                    │                                                       │
  ┌──────────┐      │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
  │ OHLCV df │─────▶│  │ Markov   │  │Liquidity │  │  Breadth Engine  │  │
  └──────────┘      │  │ Engine   │  │  Engine  │  │  (crypto+equity) │  │
                    │  └────┬─────┘  └────┬─────┘  └────────┬─────────┘  │
                    │       │             │                   │             │
                    │       └─────────────┴──────────────────┘             │
                    │                        │                              │
                    │              ┌─────────▼──────────┐                  │
                    │              │  RegimeEnsemble v2  │                  │
                    │              │  (7-component)      │                  │
                    │              └─────────┬──────────┘                  │
                    │                        │                              │
                    │       ┌────────────────┼────────────────────┐        │
                    │       │                │                     │        │
                    │  ┌────▼────┐  ┌───────▼──────┐  ┌─────────▼──────┐ │
                    │  │Persist  │  │ CrossAsset   │  │   FlowEngine   │ │
                    │  │ Engine  │  │   Engine     │  │(fund+OI+LS+liq)│ │
                    │  └────┬────┘  └───────┬──────┘  └─────────┬──────┘ │
                    │       │               │                    │         │
                    │       └───────────────┼────────────────────┘         │
                    │                       │                               │
                    │              ┌────────▼───────────┐                  │
                    │              │   Bayesian v2      │                  │
                    │              │ (reliability gated)│                  │
                    │              └────────┬───────────┘                  │
                    │                       │                               │
                    │              ┌────────▼───────────┐                  │
                    │              │  Forecast Engine   │                  │
                    │              │  XGBoost/LGBM      │                  │
                    │              └────────┬───────────┘                  │
                    │                       │                               │
                    │              ┌────────▼───────────┐                  │
                    │              │  Conviction Engine │                  │
                    │              │  (8-dim aggregator)│                  │
                    │              └────────┬───────────┘                  │
                    │                       │                               │
                    │       ┌───────────────┼───────────────────┐          │
                    │       │               │                   │          │
                    │  ┌────▼────┐  ┌──────▼──────┐  ┌────────▼──────┐  │
                    │  │ Options │  │  Portfolio  │  │    Report     │  │
                    │  │ Adv Grk │  │  Optimizer  │  │  Dashboards   │  │
                    │  └─────────┘  └─────────────┘  └───────────────┘  │
                    └─────────────────────────────────────────────────────┘
                                              │
                              ┌───────────────┼─────────────────┐
                              │               │                 │
                        ┌─────▼────┐  ┌──────▼──────┐  ┌──────▼──────┐
                        │  Google  │  │    LINE     │  │  FuturesR.  │
                        │  Sheets  │  │  Alert v2   │  │    v2       │
                        │   v2     │  └─────────────┘  └─────────────┘
                        └──────────┘
```

---

## Execution Flow  (29 steps)

```
Steps 1–11   : Original pipeline (indicators, Markov, structure, stops, TP)
Step  2b     : Liquidity Regime Engine          [Phase 12]
Step  3      : Regime Ensemble v2               [Phase 2+12+14+21]
Step  3b     : Market Breadth Engine            [Phase 14]
Step  3c     : Cross-Asset Engine               [Phase 21]
Step  3d     : Regime Persistence Engine        [Phase 15]
Steps 12–13  : AI Score + Trade Quality         [original]
Step  13b    : Flow Engine                      [Phase 13]
Step  14     : Bayesian Reliability v2          [Phase 16]
Step  14b    : Forecast Engine                  [Phase 17]
Step  14c    : Conviction Engine                [Phase 18]
Steps 15–16  : Monte Carlo + Position Sizing    [original]
Step  16b    : Portfolio Optimizer              [Phase 20]
Steps 17–18  : Final Decision + Options         [original + Phase 19]
Step  19     : Report + Institutional Dashboards[extended]
```

---

## Data Contracts

### Phase 12 — LiquidityRegimeResult
```python
@dataclass(frozen=True)
class LiquidityRegimeResult:
    liquidity_regime:  str    # RISK_ON | RISK_OFF | CRISIS | RECOVERY
    confidence:        float  # 0–100
    score:             float  # 0–100  composite liquidity score
    risk_multiplier:   float  # 0.25–1.25  applied to Kelly sizing
    vix_level:         float  # last VIX close
    vix_regime:        str    # CALM | ELEVATED | STRESS | CRISIS
    dxy_trend:         str    # STRENGTHENING | WEAKENING | STABLE
    yield_trend:       str    # RISING | FALLING | STABLE
    tlt_vol_pct:       float  # TLT 20d HV as MOVE proxy
    component_scores:  dict   # {vix, dxy, yield, tlt_vol} → float
    interpretation:    str
    data_source:       str    # live | cached | fallback
```
**Weights:** VIX 40% | DXY 25% | Yield 20% | TLT Vol 15%

---

### Phase 13 — FlowEngineResult
```python
@dataclass(frozen=True)
class FlowEngineResult:
    symbol:           str
    flow_score:       float  # 0–100
    flow_direction:   str    # BULLISH | BEARISH | NEUTRAL
    flow_confidence:  float  # 0–100
    flow_regime:      str    # BULLISH_FLOW | BEARISH_FLOW | SHORT_SQUEEZE | LONG_SQUEEZE | NEUTRAL
    funding_score:    float  # component score 0–100
    oi_score:         float
    ls_score:         float
    liq_score:        float
    funding_rate_pct: float
    funding_regime:   str
    oi_signal:        str    # CONFIRMATION | CONTINUATION | WEAK_RALLY | CAPITULATION
    ls_ratio:         float  # long/short ratio
    cascade_risk:     str    # HIGH | MODERATE | LOW
    interpretation:   str
    component_detail: dict
```
**Weights:** Funding 35% | OI 30% | L/S 25% | Liquidation 10%

---

### Phase 14 — MarketBreadthResult
```python
@dataclass(frozen=True)
class MarketBreadthResult:
    breadth_score:      float  # 0–100
    breadth_regime:     str    # STRONG_BULL | BULL | NEUTRAL | BEAR | STRONG_BEAR
    breadth_confidence: float  # 0–100
    crypto_breadth:     CryptoBreadthResult
    equity_breadth:     EquityBreadthResult
    component_scores:   dict
    interpretation:     str
    data_quality:       str    # FULL | PARTIAL | FALLBACK

class CryptoBreadthResult:
    btc_dominance_est:    float  # % (proxy)
    btc_dom_trend:        str
    total3_change_pct:    float  # 20d alt basket change
    total3_trend:         str
    ssr_proxy:            float  # stablecoin ratio proxy
    crypto_breadth_score: float

class EquityBreadthResult:
    advance_decline_ratio: float
    pct_above_200dma:      float
    new_high_count:        int
    new_low_count:         int
    equity_breadth_score:  float
```
**Composite weights:** Equity 60% | Crypto 40%

---

### Phase 15 — RegimePersistenceResult
```python
@dataclass(frozen=True)
class RegimePersistenceResult:
    regime:                  str
    self_transition_prob:    float  # P(i→i) from Markov matrix
    expected_duration_days:  float  # E[T] = 1/(1-p_self)
    regime_half_life_days:   float  # ln(0.5)/ln(p_self)
    remaining_duration_days: float  # probabilistic estimate
    exit_prob_7d:            float  # P(exit within 7d) %
    exit_prob_14d:           float
    persistence_score:       float  # 0–100
    persistence_label:       str    # ESTABLISHED | MATURING | FRESH | EXHAUSTED
    most_likely_next:        str
    next_regime_probs:       dict
    interpretation:          str
```

---

### Phase 16 — BayesianResult (v2, same contract as v1)
Drop-in: `compute_bayesian_analysis_v2()` returns same `BayesianResult`.
**Change:** All likelihoods multiplied by regime-conditional reliability weight.
**Effect:** RSI oversold in STRONG_BEAR has `reliability=0.15` → posterior dampened.

---

### Phase 17 — ForecastResult
```python
@dataclass(frozen=True)
class ForecastResult:
    expected_return_5d:   float  # % expected return
    expected_return_10d:  float
    expected_return_20d:  float
    probability_up_5d:    float  # 0–1
    probability_up_10d:   float
    probability_up_20d:   float
    forecast_confidence:  float  # 0–100 (model quality proxy)
    forecast_direction:   str    # BULLISH | BEARISH | NEUTRAL
    feature_importances:  dict   # {feature: importance}
    model_used:           str    # xgboost | lightgbm | statistical_fallback
    horizon_forecasts:    dict   # {5d: {return_pct, prob_up}, ...}
    interpretation:       str
```
**Features (15):** rsi14, atr_pct, volume_ratio, momentum_20, momentum_5,
close_to_ema200, close_to_ema50, hv5, hv20, vov, regime_bull_prob,
regime_bear_prob, regime_confidence, funding_rate, liquidity_score,
breadth_score, oi_trend

---

### Phase 18 — ConvictionResult
```python
@dataclass(frozen=True)
class ConvictionResult:
    conviction_score:       float  # 0–100
    conviction_tier:        str    # FULL SIZE | NORMAL SIZE | HALF SIZE | NO TRADE
    kelly_multiplier:       float  # 0.0 | 0.50 | 0.75 | 1.00
    trade_allowed:          bool
    component_scores:       dict   # {8 signals → score}
    component_weights:      dict
    weighted_scores:        dict
    weakest_signal:         str
    strongest_signal:       str
    alignment_count:        int    # how many of 8 signals align
    regime_persistence_ok:  bool
    interpretation:         str
```
**Weights:** Regime 25% | Trend 15% | Flow 15% | Structure 10% |
Volatility 10% | Breadth 10% | Liquidity 10% | Forecast 5%

**Tiers:**
- 90–100 → FULL SIZE   (1.00× Kelly)
- 70–89  → NORMAL SIZE (0.75× Kelly)
- 50–69  → HALF SIZE   (0.50× Kelly)
- < 50   → NO TRADE    (0.00× Kelly)

---

### Phase 19 — AdvancedGreeks + InstitutionalOptionsRecommendation
```python
@dataclass(frozen=True)
class AdvancedGreeks:
    delta: float   # dV/dS
    gamma: float   # d²V/dS²
    theta: float   # dV/dt (per day)
    vega:  float   # dV/dσ (per 1% IV)
    rho:   float   # dV/dr
    vanna: float   # d²V/dSdσ  (delta per 1% IV change)
    charm: float   # d²V/dSdt  (delta decay per day)
    vomma: float   # d²V/dσ²   (vega convexity)
    veta:  float   # d²V/dσdt  (vega decay per day)
    speed: float   # d³V/dS³   (gamma change per $)

@dataclass(frozen=True)
class InstitutionalOptionsRecommendation:
    strategy, strike, dte, pop, expected_value,
    delta, theta, vega, kelly_size,
    gamma, vanna, charm, vomma,
    expected_move, prob_50, max_profit, max_loss,
    breakeven, iv_used, summary_line
```

---

### Phase 20 — PortfolioOptimizationResult
```python
@dataclass(frozen=True)
class PortfolioOptimizationResult:
    method_used:           str    # risk_parity | hrp | equal_weight | single_asset
    recommended_weights:   dict   # {symbol: float}  sum=1.0
    risk_contributions:    dict   # {symbol: % of total vol}
    portfolio_volatility:  float  # annualised %
    portfolio_sharpe:      float
    portfolio_drawdown:    float  # parametric max DD estimate %
    diversification_ratio: float  # weighted_avg_vol / portfolio_vol
    correlation_matrix:    dict
    volatilities:          dict
    expected_returns:      dict
    n_assets:              int
    interpretation:        str
```
**Methods:**
- Risk Parity: weight ∝ 1/σᵢ, iterative convergence
- HRP: Ward linkage → quasi-diagonalisation → recursive bisection

---

### Phase 21 — CrossAssetResult
```python
@dataclass(frozen=True)
class CrossAssetResult:
    cross_asset_regime:       str    # RISK_ON_ALIGNED | RISK_OFF_ALIGNED | DECOUPLED_BULL | DECOUPLED_BEAR | TRANSITION
    relative_strength_score:  float  # 0–100 BTC vs basket
    regime_confidence:        float
    pair_analyses:            dict   # {symbol: PairAnalysis}
    rolling_correlations:     dict   # {symbol: 30d corr}
    decoupling_detected:      bool
    decoupling_direction:     str    # BULL | BEAR | NONE
    btc_beta_to_spy:          float
    btc_beta_to_qqq:          float
    dominant_driver:          str    # which asset BTC is most correlated with
    interpretation:           str

class PairAnalysis:
    pair, btc_rs_score, rolling_corr, expected_corr,
    corr_deviation, btc_return_20d, pair_return_20d,
    outperforming, signal
```
**RS Score weights:** QQQ 30% | SPY 25% | GLD 20% | DXY 15% | TLT 10%

---

## Regime Ensemble v2 — Component Weights

| Component     | Weight | Source              |
|---------------|--------|---------------------|
| Markov HMM    | 32%    | regime/markov.py    |
| Trend (EMA)   | 20%    | ema + structure     |
| Volatility    | 16%    | volatility_regime   |
| Macro (RSI)   | 12%    | signal_engine       |
| **Liquidity** | **10%**| **Phase 12 (new)**  |
| **Breadth**   | **6%** | **Phase 14 (new)**  |
| **CrossAsset**| **4%** | **Phase 21 (new)**  |

**CRISIS override:** If `liquidity_regime == CRISIS`, position_size_mult
is capped at 0.25 and `trade_permission = NO_TRADE` regardless of ensemble.

---

## FuturesResult_v2 — New Fields vs Original

All original `FuturesResult` fields are preserved.
New fields added (all optional with safe defaults):

| Field                  | Phase | Default     |
|------------------------|-------|-------------|
| liquidity_regime       | 12    | "RISK_ON"   |
| liquidity_score        | 12    | 55.0        |
| liquidity_risk_mult    | 12    | 1.0         |
| flow_regime            | 13    | "NEUTRAL"   |
| flow_score             | 13    | 50.0        |
| flow_direction         | 13    | "NEUTRAL"   |
| breadth_regime         | 14    | "NEUTRAL"   |
| breadth_score          | 14    | 50.0        |
| persistence_label      | 15    | "ESTABLISHED"|
| remaining_days         | 15    | 10.0        |
| exit_prob_7d           | 15    | 0.0         |
| forecast_direction     | 17    | "NEUTRAL"   |
| forecast_20d_return    | 17    | 0.0         |
| forecast_confidence    | 17    | 50.0        |
| conviction_score       | 18    | 50.0        |
| conviction_tier        | 18    | "HALF SIZE" |
| conviction_kelly_mult  | 18    | 0.5         |
| portfolio_vol          | 20    | 0.0         |
| portfolio_drawdown     | 20    | 0.0         |
| portfolio_sharpe       | 20    | 0.0         |
| cross_asset_regime     | 21    | "TRANSITION"|
| btc_rs_score           | 21    | 50.0        |
| btc_beta_spy           | 21    | 1.0         |

---

## Safe Failure Modes

Every new engine follows this contract:
1. Wrapped in `_safe()` in orchestrator_v2 → failure returns `None`
2. All consumers call `getattr(result, field, default)` → `None` safe
3. Report/dashboard sections show "Data unavailable" if `None`
4. LINE alerts skip unavailable sections gracefully
5. Sheets write empty string `""` for missing values

No single engine failure can crash the pipeline or block a trade signal.
