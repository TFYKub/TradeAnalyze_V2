"""
Monte Carlo + Signal Consistency Checker  (Phase 1, Fix 5)
============================================================
Validates logical consistency between:
  • P(Profit) vs P(Target Hit)
  • Expected Return vs Expected Drawdown
  • EV sign vs POP
  • RR vs EV

Flags contradictory outputs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConsistencyResult:
    is_consistent:  bool
    warnings:       tuple[str, ...]
    errors:         tuple[str, ...]
    confidence_adj: float     # 0–1 multiplier to apply to final confidence
    details:        dict


def check_monte_carlo_consistency(
    prob_profit:     float,   # 0–100
    prob_target_hit: float,   # 0–100
    prob_stop_hit:   float,   # 0–100
    expected_return: float,   # %
    ev:              float,   # in R units
    rr:              float,   # risk/reward
    pop:             float,   # 0–100 from options engine
) -> ConsistencyResult:
    """
    Cross-validate simulation outputs for internal consistency.

    Returns ConsistencyResult with warnings/errors and a confidence adjustment.
    """

    warnings: list[str] = []
    errors:   list[str] = []

    # Rule 1: P(profit) >= P(target_hit) always
    # (reaching target is a subset of being profitable)
    if prob_target_hit > prob_profit + 5:   # +5 tolerance
        errors.append(
            f"P(target)={prob_target_hit:.1f}% > P(profit)={prob_profit:.1f}% — impossible: "
            "target hit ⊂ profit, so P(target) must be ≤ P(profit)"
        )

    # Rule 2: P(profit) + P(stop_hit) should roughly sum to ≤ 100%
    # (paths that neither profit nor hit stop end in the middle)
    if prob_profit + prob_stop_hit > 105:
        warnings.append(
            f"P(profit)={prob_profit:.1f}% + P(stop)={prob_stop_hit:.1f}% = "
            f"{prob_profit + prob_stop_hit:.1f}% > 100% — check simulation logic"
        )

    # Rule 3: Positive EV should correlate with P(profit) > 50%
    if ev > 0 and prob_profit < 40:
        warnings.append(
            f"EV={ev:.2f}R > 0 but P(profit)={prob_profit:.1f}% < 40% — "
            "high EV with low win rate implies extreme right tail, verify RR"
        )

    if ev <= 0 and prob_profit > 70:
        warnings.append(
            f"EV={ev:.2f}R ≤ 0 but P(profit)={prob_profit:.1f}% > 70% — "
            "high win rate with negative EV implies loss > win, check position sizing"
        )

    # Rule 4: Expected return sign should match EV sign
    if ev > 0 and expected_return < -2:
        warnings.append(
            f"EV={ev:.2f}R positive but expected_return={expected_return:.1f}% negative — "
            "check GBM drift vs historical win rate"
        )

    # Rule 5: POP should be in plausible range for RR
    # High RR → low POP is normal for debit spreads
    # Low RR  → high POP is normal for credit spreads
    if rr >= 3.0 and pop > 80:
        warnings.append(
            f"RR={rr:.1f} high but POP={pop:.1f}% also very high — "
            "verify strike selection; unlikely to have both simultaneously"
        )
    if rr < 1.0 and pop < 50:
        errors.append(
            f"RR={rr:.1f} < 1 and POP={pop:.1f}% < 50 → negative EV trade, must reject"
        )

    is_consistent = len(errors) == 0
    n_issues      = len(warnings) + len(errors) * 2
    confidence_adj = max(0.5, 1.0 - n_issues * 0.10)

    if not is_consistent:
        logger.warning("[consistency] %d errors: %s", len(errors), errors)
    if warnings:
        logger.info("[consistency] %d warnings: %s", len(warnings), warnings)

    return ConsistencyResult(
        is_consistent  = is_consistent,
        warnings       = tuple(warnings),
        errors         = tuple(errors),
        confidence_adj = round(confidence_adj, 3),
        details        = {
            "prob_profit":     prob_profit,
            "prob_target_hit": prob_target_hit,
            "prob_stop_hit":   prob_stop_hit,
            "expected_return": expected_return,
            "ev":              ev,
            "rr":              rr,
        },
    )
