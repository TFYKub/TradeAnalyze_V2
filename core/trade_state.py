"""
Extended TradeState – Single Source of Truth for V3
Adds all new fields from phases 1-7.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np

@dataclass
class TradeStateV3:
    # ----- Phase 1: Walk‑Forward & Adaptive Ensemble -----
    walk_forward_result: Optional[Dict] = None          # {engine: {sharpe, hit_rate}}
    ensemble_weights: Optional[Dict[str, float]] = None # dynamic weights
    signal_records: List[Dict] = field(default_factory=list)

    # ----- Phase 2: Quantile Forecast -----
    quantile_forecast_5d: Optional[Dict] = None
    quantile_forecast_10d: Optional[Dict] = None
    quantile_forecast_20d: Optional[Dict] = None
    forecast_confidence_interval: Optional[tuple] = None

    # ----- Phase 3: Regime‑Switching Monte Carlo -----
    regime_mc_result: Optional[Dict] = None   # {prob_profit, prob_stop, var_95, cvar_95, regime_weights}
    dynamic_var: Optional[float] = None
    dynamic_cvar: Optional[float] = None

    # ----- Phase 4: Portfolio Optimizer -----
    portfolio_allocation: Optional[Dict] = None  # {symbol: weight}
    portfolio_risk_parity_weights: Optional[Dict] = None
    portfolio_min_var_weights: Optional[Dict] = None
    portfolio_max_sharpe_weights: Optional[Dict] = None
    portfolio_kelly_weights: Optional[Dict] = None
    portfolio_volatility: Optional[float] = None
    portfolio_sharpe: Optional[float] = None

    # ----- Phase 5: Crypto Flow V2 -----
    crypto_flow_v2: Optional[Dict] = None   # {flow_score, oi_delta, funding_momentum, liquidation_pressure, whale_flow_ratio}

    # ----- Phase 6: Dealer Greeks -----
    dealer_greeks: Optional[Dict] = None    # {total_gamma, gamma_concentration, pin_risk_strike, dealer_vanna, dealer_charm}

    # ----- Phase 7: Stress Testing -----
    stress_test_results: Optional[List[Dict]] = None  # list of scenario results

    # ----- Metadata -----
    timestamp: datetime = field(default_factory=datetime.now)
    version: str = "3.0"