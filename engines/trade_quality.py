"""
Institutional Trade Quality Engine  (Phase 4)
===============================================
Grades trades: A+ | A | B | C | REJECT

Composite quality score 0–100 from 7 inputs:
  Regime     (20%) — STRONG_BULL/BEAR=100, BULL/BEAR=80, RANGE=50
  Trend      (15%) — EMA alignment strength
  Structure  (20%) — HH-HL or LL-LH clarity
  EV         (15%) — Expected Value ≥ 0 and ideally > 1R
  RR         (15%) — Risk Reward ≥ 1.5, ideally ≥ 3
  Volume     (10%) — Volume confirmation vs average
  Volatility (5%)  — Vol regime (normal = best for directional)

Grade mapping:
  90–100 → A+
  80–89  → A
  65–79  → B
  50–64  → C
  < 50   → REJECT
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

WEIGHTS = {
    "regime":     0.20,
    "trend":      0.15,
    "structure":  0.20,
    "ev":         0.15,
    "rr":         0.15,
    "volume":     0.10,
    "volatility": 0.05,
}


@dataclass(frozen=True)
class TradeQualityResult:
    grade:           str      # A+ | A | B | C | REJECT
    score:           float    # 0–100
    component_scores: dict[str, float]
    trade_allowed:   bool
    grade_reason:    str


def _regime_score(regime: str, confidence: float) -> float:
    base = {
        "STRONG_BULL": 100, "STRONG_BEAR": 100,
        "BULL":         80, "BEAR":         80,
        "RANGE":        50, "CORRECTION":   40,
    }.get(regime, 40)
    return round(base * min(1.0, confidence / 100), 1)


def _trend_score(ema_alignment: float) -> float:
    return min(100.0, max(0.0, float(ema_alignment)))


def _structure_score(structure_trend: str, structure_clarity: float) -> float:
    base = {
        "BULLISH": 100, "BEARISH": 100, "MIXED": 50, "UNKNOWN": 20
    }.get(structure_trend, 20)
    return round((base + structure_clarity) / 2, 1)


def _ev_score(ev: float) -> float:
    if ev >= 2.0:    return 100.0
    if ev >= 1.0:    return 80.0
    if ev >= 0.5:    return 65.0
    if ev >= 0.0:    return 50.0
    return 0.0


def _rr_score(rr: float) -> float:
    if rr >= 4.0: return 100.0
    if rr >= 3.0: return 85.0
    if rr >= 2.0: return 70.0
    if rr >= 1.5: return 55.0
    return 0.0


def _volume_score(vol_ratio: float) -> float:
    """vol_ratio = current_volume / avg_volume_20."""
    if vol_ratio >= 2.0: return 100.0
    if vol_ratio >= 1.5: return 85.0
    if vol_ratio >= 1.0: return 70.0
    if vol_ratio >= 0.7: return 50.0
    return 30.0


def _vol_regime_score(vol_regime: str) -> float:
    return {
        "NORMAL_VOL": 100.0,
        "LOW_VOL":     75.0,
        "HIGH_VOL":    50.0,
        "PANIC_VOL":   10.0,
    }.get(vol_regime, 60.0)


def _grade(score: float) -> tuple[str, bool]:
    if score >= 90: return "A+",     True
    if score >= 80: return "A",      True
    if score >= 65: return "B",      True
    if score >= 50: return "C",      True
    return "REJECT", False


def compute_trade_quality(
    regime:           str,
    regime_confidence: float,
    ema_alignment:    float,      # 0–100 from EMAResult
    structure_trend:  str,
    structure_clarity: float,     # 0–100
    ev:               float,      # Expected Value in R
    rr:               float,      # Risk/Reward ratio
    volume_ratio:     float,      # current / avg (1.0 = average)
    vol_regime:       str,        # LOW_VOL | NORMAL_VOL | HIGH_VOL | PANIC_VOL
) -> TradeQualityResult:
    """
    Compute institutional trade quality grade.
    """

    scores = {
        "regime":     _regime_score(regime, regime_confidence),
        "trend":      _trend_score(ema_alignment),
        "structure":  _structure_score(structure_trend, structure_clarity),
        "ev":         _ev_score(ev),
        "rr":         _rr_score(rr),
        "volume":     _volume_score(volume_ratio),
        "volatility": _vol_regime_score(vol_regime),
    }

    total = sum(scores[k] * WEIGHTS[k] for k in scores)
    total = round(total, 1)

    grade, allowed = _grade(total)

    weak = [k for k, v in scores.items() if v < 50]
    if weak:
        reason = f"Grade {grade} ({total:.0f}/100) — weak: {', '.join(weak)}"
    else:
        reason = f"Grade {grade} ({total:.0f}/100) — all components strong"

    logger.info("[trade_quality] %s score=%.1f  scores=%s", grade, total, scores)

    return TradeQualityResult(
        grade            = grade,
        score            = total,
        component_scores = scores,
        trade_allowed    = allowed,
        grade_reason     = reason,
    )
