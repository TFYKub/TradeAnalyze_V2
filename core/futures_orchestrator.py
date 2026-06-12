# core/futures_orchestrator.py
"""
Original Futures Orchestrator (v1) – kept only for the FuturesResult dataclass.
The v1 orchestrator class is dead; v2 uses FuturesOrchestrator_v2.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class FuturesResult:
    """Original FuturesResult dataclass – required by options_orchestrator.py."""
    symbol:              str
    price:               float
    runtime:             float
    report_text:         str
    final_decision:      str
    ai_score:            float
    trade_grade:         str
    trade_quality_score: float
    regime:              str
    regime_conf:         float
    vol_regime:          str
    entry:               float
    stop_loss:           float
    stop_reason:         str
    tp1:                 Optional[float]
    tp2:                 Optional[float]
    rr:                  float
    risk_pct:            float
    mc_profit_prob:      float
    kelly:               float
    ev:                  float
    sharpe:              float
    approved:            bool
    consistency_ok:      bool
    bayesian_bull:       float
    bayesian_bear:       float