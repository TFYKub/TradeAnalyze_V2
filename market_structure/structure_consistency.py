"""
Structure Consistency Engine  (Phase 1, Fix 3)
================================================
Detects conflicts between market structure signals and reduces
confidence when contradictory signals appear.

Conflict examples:
  • HH-HL (bullish)  +  Bearish BOS  → contradiction
  • LL-LH (bearish)  +  Bullish BOS  → contradiction
  • RSI divergence opposite to EMA bias

Output: StructureConsistencyResult
  structure_confidence : 0–100 (reduced when conflicts present)
  conflict_detected    : bool
  conflict_reason      : str
  confidence_penalty   : float  (subtracted from regime confidence)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StructureConsistencyResult:
    structure_confidence: float          # 0–100
    conflict_detected:    bool
    conflict_reason:      str
    confidence_penalty:   float          # deduct from overall confidence
    consistency_grade:    str            # STRONG | MODERATE | WEAK | CONFLICT
    details:              tuple[str, ...] = field(default_factory=tuple)


def check_structure_consistency(
    structure_trend:   str,    # "BULLISH" | "BEARISH" | "MIXED" | "UNKNOWN"
    bos_bullish:       bool,   # Break of swing high
    bos_bearish:       bool,   # Break of swing low
    ema_bias:          str,    # "BULLISH" | "BEARISH"
    divergence_kind:   str,    # "BULLISH" | "BEARISH" | "NONE"
    regime:            str,    # Markov regime
    structure_score:   float,  # raw 0–100 from structure_break.py
) -> StructureConsistencyResult:
    """
    Cross-validate structure signals for internal consistency.

    Deductions:
      HH-HL + Bearish BOS         → -25
      LL-LH + Bullish BOS         → -25
      Structure trend ≠ EMA bias  → -15
      Divergence opposes trend    → -10
      Structure = UNKNOWN         → -20
      Regime vs structure mismatch→ -10
    """

    conflicts: list[str] = []
    penalty   = 0.0

    # ── BOS contradicts structure ─────────────────────────────────────────────
    if structure_trend == "BULLISH" and bos_bearish:
        conflicts.append("HH-HL structure but Bearish BOS — potential trend reversal")
        penalty += 25
    if structure_trend == "BEARISH" and bos_bullish:
        conflicts.append("LL-LH structure but Bullish BOS — potential reversal forming")
        penalty += 25

    # ── Structure vs EMA mismatch ─────────────────────────────────────────────
    if structure_trend == "BULLISH" and ema_bias == "BEARISH":
        conflicts.append("Bullish structure but EMA Bearish — structure lagging EMA")
        penalty += 15
    if structure_trend == "BEARISH" and ema_bias == "BULLISH":
        conflicts.append("Bearish structure but EMA Bullish — EMA possibly topping")
        penalty += 15

    # ── RSI divergence opposing structure ────────────────────────────────────
    if structure_trend == "BULLISH" and divergence_kind == "BEARISH":
        conflicts.append("Bullish structure with Bearish RSI divergence — caution")
        penalty += 10
    if structure_trend == "BEARISH" and divergence_kind == "BULLISH":
        conflicts.append("Bearish structure with Bullish RSI divergence — possible base")
        penalty += 10

    # ── Unknown / mixed structure ─────────────────────────────────────────────
    if structure_trend in ("UNKNOWN", "MIXED"):
        conflicts.append(f"Structure is {structure_trend} — reduced clarity")
        penalty += 20

    # ── Regime vs structure mismatch ─────────────────────────────────────────
    regime_bull  = regime in ("STRONG_BULL", "BULL")
    regime_bear  = regime in ("STRONG_BEAR", "BEAR", "CORRECTION")
    if regime_bull and structure_trend == "BEARISH":
        conflicts.append("Bull regime but Bearish structure — watch for regime change")
        penalty += 10
    if regime_bear and structure_trend == "BULLISH":
        conflicts.append("Bear regime but Bullish structure — watch for reversal")
        penalty += 10

    # ── Final calculation ─────────────────────────────────────────────────────
    from config.thresholds import THRESHOLDS
    penalty = min(penalty, THRESHOLDS.MAX_STRUCTURE_CONFLICT_PENALTY + 20)  # cap at 50
    base_confidence   = structure_score
    final_confidence  = max(5.0, base_confidence - penalty)

    conflict_detected = len(conflicts) > 0
    conflict_reason   = " | ".join(conflicts) if conflicts else "No conflicts"

    if final_confidence >= 75:
        grade = "STRONG"
    elif final_confidence >= 50:
        grade = "MODERATE"
    elif final_confidence >= 25:
        grade = "WEAK"
    else:
        grade = "CONFLICT"

    if conflict_detected:
        logger.info("[structure_consistency] %s penalty=%.0f conf=%.0f  %s",
                    grade, penalty, final_confidence, conflicts)

    return StructureConsistencyResult(
        structure_confidence = round(final_confidence, 1),
        conflict_detected    = conflict_detected,
        conflict_reason      = conflict_reason,
        confidence_penalty   = round(penalty, 1),
        consistency_grade    = grade,
        details              = tuple(conflicts),
    )
