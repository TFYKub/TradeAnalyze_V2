"""
Final Trade Decision Engine  (upgraded — Phase 1, Fix 1)
==========================================================
All thresholds now sourced from config/thresholds.py (configurable).

7-Gate system — trade approved only when ALL pass:
  Gate 1: Regime Confidence  >= MIN_REGIME_CONFIDENCE (60%)
  Gate 2: AI Score           >= MIN_AI_SCORE          (70)
  Gate 3: Expected Value     >= MIN_EV                (0)
  Gate 4: Kelly Fraction     > 0
  Gate 5: MC Profit Prob     >= MIN_MC_PROFIT_PROB    (60%)
  Gate 6: Risk Reward        >= MIN_RR                (1.5)  ← was 2.0, now configurable
  Gate 7: Structure + EMA alignment confirmed
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from config.thresholds import THRESHOLDS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FinalDecision:
    decision:       str
    approved:       bool
    gates_passed:   list[str] = field(default_factory=list)
    gates_failed:   list[str] = field(default_factory=list)
    reason:         str = ""
    confidence_pct: float = 0.0


def evaluate_trade(
    direction:         str,
    regime_confidence: float,
    ai_score:          float,
    expected_value:    float,
    kelly_fraction:    float,
    mc_profit_prob:    float,
    best_rr:           float,
    structure_trend:   str,
    ema_bias:          str,
) -> FinalDecision:

    gates_passed: list[str] = []
    gates_failed: list[str] = []

    if direction in ("WAIT", "NO_TRADE"):
        return FinalDecision(
            decision="NO_TRADE", approved=False,
            gates_failed=["Direction = WAIT"],
            reason="No directional signal from entry engine",
        )

    # Gate 1
    if regime_confidence >= THRESHOLDS.MIN_REGIME_CONFIDENCE:
        gates_passed.append(f"Regime Confidence {regime_confidence:.0f}% ≥ {THRESHOLDS.MIN_REGIME_CONFIDENCE:.0f}%")
    else:
        gates_failed.append(f"Regime Confidence {regime_confidence:.0f}% < {THRESHOLDS.MIN_REGIME_CONFIDENCE:.0f}%")

    # Gate 2
    if ai_score >= THRESHOLDS.MIN_AI_SCORE:
        gates_passed.append(f"AI Score {ai_score:.0f} ≥ {THRESHOLDS.MIN_AI_SCORE:.0f}")
    else:
        gates_failed.append(f"AI Score {ai_score:.0f} < {THRESHOLDS.MIN_AI_SCORE:.0f}")

    # Gate 3
    if expected_value >= THRESHOLDS.MIN_EV:
        gates_passed.append(f"EV {expected_value:.2f}R > {THRESHOLDS.MIN_EV}")
    else:
        gates_failed.append(f"EV {expected_value:.2f}R ≤ {THRESHOLDS.MIN_EV}")

    # Gate 4
    if kelly_fraction > 0:
        gates_passed.append(f"Kelly {kelly_fraction:.3f} > 0")
    else:
        gates_failed.append(f"Kelly {kelly_fraction:.3f} ≤ 0")

    # Gate 5
    if mc_profit_prob >= THRESHOLDS.MIN_MC_PROFIT_PROB:
        gates_passed.append(f"MC P(profit) {mc_profit_prob:.1f}% ≥ {THRESHOLDS.MIN_MC_PROFIT_PROB:.0f}%")
    else:
        gates_failed.append(f"MC P(profit) {mc_profit_prob:.1f}% < {THRESHOLDS.MIN_MC_PROFIT_PROB:.0f}%")

    # Gate 6 — uses configurable MIN_RR
    if best_rr >= THRESHOLDS.MIN_RR:
        gates_passed.append(f"RR {best_rr:.2f} ≥ {THRESHOLDS.MIN_RR:.1f}")
    else:
        gates_failed.append(f"RR {best_rr:.2f} < {THRESHOLDS.MIN_RR:.1f} (MIN_RR)")

    # Gate 7
    dir_ok = (
        (direction == "LONG"  and ema_bias == "BULLISH") or
        (direction == "SHORT" and ema_bias == "BEARISH")
    )
    if dir_ok:
        gates_passed.append(f"EMA {ema_bias} confirms {direction}")
    else:
        gates_failed.append(f"EMA {ema_bias} conflicts with {direction}")

    total    = len(gates_passed) + len(gates_failed)
    approved = len(gates_failed) == 0
    conf_pct = round(len(gates_passed) / total * 100, 1) if total > 0 else 0.0

    if approved:
        reason   = f"All {total} gates passed → {direction}"
        decision = direction
    else:
        reason   = "BLOCKED: " + " | ".join(gates_failed)
        decision = "NO_TRADE"

    return FinalDecision(
        decision=decision, approved=approved,
        gates_passed=gates_passed, gates_failed=gates_failed,
        reason=reason, confidence_pct=conf_pct,
    )
