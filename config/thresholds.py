"""
Institutional Trading Thresholds — Centralised Config
=======================================================
All min/max thresholds used across the system.
Change here → applies everywhere.
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class TradingThresholds:
    # Risk Reward
    MIN_RR:              float = 1.5
    MIN_RR_PREFERRED:    float = 2.0
    MIN_RR_EXCELLENT:    float = 3.0

    # AI Score
    MIN_AI_SCORE:        float = 70.0
    MIN_AI_PREFERRED:    float = 80.0

    # Expected Value
    MIN_EV:              float = 0.0
    MIN_EV_PREFERRED:    float = 0.5

    # Regime
    MIN_REGIME_CONFIDENCE: float = 60.0
    MAX_REGIME_CONFIDENCE: float = 95.0   # cap to avoid 100%

    # Monte Carlo
    MIN_MC_PROFIT_PROB:  float = 60.0

    # Kelly
    MAX_KELLY:           float = 0.25
    MAX_POSITION_RISK:   float = 0.02     # 2% of account

    # Trade Quality
    MIN_TRADE_QUALITY:   str   = "B"      # A+/A/B/C/REJECT
    MIN_TRADE_SCORE:     float = 60.0

    # Volatility
    ATR_MULTIPLIER:      float = 1.0
    ATR_MULTIPLIER_HIGH_VOL: float = 1.5
    ATR_MULTIPLIER_LOW_VOL:  float = 0.75

    # Structure
    MIN_STRUCTURE_SCORE: float = 40.0
    MAX_STRUCTURE_CONFLICT_PENALTY: float = 30.0


THRESHOLDS = TradingThresholds()
