"""
Regime Persistence Engine  (Phase 15)
=======================================
Uses the Markov transition matrix to estimate how long the current regime
has been active and how much longer it is expected to last.

Mathematics — Geometric Sojourn Time:
  For a Markov chain state i, the expected number of time steps spent
  in state i before transitioning is:
    E[T_i] = 1 / (1 - P(i → i))

  Where P(i → i) is the self-transition probability from the transition matrix.
  This gives the mean sojourn time in days (using daily data).

  Remaining duration estimate uses regime half-life:
    half_life = E[T_i] × ln(0.5) / ln(P(i → i))
              ≈ E[T_i] × 0.693

  Since we don't track regime start date across sessions, remaining duration
  is probabilistically estimated based on the distribution:
    P(remaining ≥ k) = P(i → i)^k

Outputs are reported to the dashboard and used by ConvictionEngine
to discount signals when a regime is near exhaustion.

Report example:
  Current Regime : BEAR
  Expected Duration: 18 days
  Elapsed (est.)   : ~11 days
  Remaining (est.) : ~7 days
  Half Life        : 11 days
  Exit Probability : 38% (within 7 days)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegimePersistenceResult:
    regime:                  str
    self_transition_prob:    float     # P(i → i) from Markov matrix
    expected_duration_days:  float     # mean sojourn time
    regime_half_life_days:   float     # median time until exit
    remaining_duration_days: float     # estimated days left (probabilistic)
    exit_prob_7d:            float     # probability of exit within 7 days
    exit_prob_14d:           float     # probability of exit within 14 days
    persistence_score:       float     # 0–100: 100 = very stable, 0 = about to exit
    persistence_label:       str       # EXHAUSTED | MATURING | ESTABLISHED | FRESH
    most_likely_next:        str       # most likely next regime
    next_regime_probs:       dict[str, float]
    interpretation:          str


# ── Core Mathematics ──────────────────────────────────────────────────────────
def _expected_duration(p_self: float) -> float:
    """E[T] = 1 / (1 - p_self). Floor at 1 day, cap at 365."""
    if p_self >= 1.0:
        return 365.0
    if p_self <= 0.0:
        return 1.0
    return min(365.0, max(1.0, 1.0 / (1.0 - p_self)))


def _half_life(p_self: float) -> float:
    """
    Median sojourn time: k such that P(T > k) = 0.5.
    P(T > k) = p_self^k → k = ln(0.5) / ln(p_self)
    """
    if p_self <= 0.0:
        return 0.5
    if p_self >= 1.0:
        return 365.0
    try:
        return max(0.5, math.log(0.5) / math.log(p_self))
    except (ValueError, ZeroDivisionError):
        return _expected_duration(p_self) * 0.693


def _exit_prob_within(p_self: float, days: int) -> float:
    """P(exit within d days) = 1 - p_self^days."""
    if p_self <= 0.0:
        return 1.0
    if p_self >= 1.0:
        return 0.0
    return round(1.0 - p_self ** days, 4)


def _remaining_duration(expected: float, half_life: float) -> float:
    """
    Without knowing elapsed time, estimate remaining duration
    from the midpoint perspective:
      - If at expected/2, remaining ≈ expected/2
      - Capped at expected duration
    We use half-life as the central estimate of remaining time.
    """
    # Conditional expected remaining time assuming we are somewhere in the distribution:
    # E[remaining | T > 0] = E[T] for memoryless geometric (Markov property)
    # In practice we discount slightly for regimes that look mature
    return round(min(expected, max(0.5, half_life)), 1)


def _persistence_score(p_self: float, exit_prob_7d: float) -> float:
    """
    0–100 score where 100 = rock-solid, 0 = about to flip.
    Combines self-transition probability with short-term exit probability.
    """
    base    = p_self * 100.0
    exit_penalty = exit_prob_7d * 30.0     # high 7d exit prob penalises
    return round(max(0.0, min(100.0, base - exit_penalty)), 1)


def _persistence_label(score: float) -> str:
    if score >= 80: return "ESTABLISHED"
    if score >= 60: return "MATURING"
    if score >= 40: return "FRESH"
    return "EXHAUSTED"


# ── Main Entry ─────────────────────────────────────────────────────────────────
def compute_regime_persistence(
    current_regime:    str,
    transition_matrix: dict[str, dict[str, float]],
) -> RegimePersistenceResult:
    """
    Compute regime persistence metrics from Markov transition matrix.

    Parameters
    ----------
    current_regime    : e.g. "BEAR"
    transition_matrix : from MarkovRegimeEngine.detect() — dict[from → dict[to → prob]]

    Returns
    -------
    RegimePersistenceResult — always returns, never raises
    """
    # ── Fallback defaults ─────────────────────────────────────────────────────
    if not transition_matrix or current_regime not in transition_matrix:
        logger.warning(
            "[persistence] No transition matrix for regime=%s — using default probs",
            current_regime,
        )
        # Typical self-transition for crypto regimes ≈ 0.85
        fallback_p_self = 0.85
        exp_dur  = _expected_duration(fallback_p_self)
        half_lif = _half_life(fallback_p_self)
        remain   = _remaining_duration(exp_dur, half_lif)
        exit_7d  = _exit_prob_within(fallback_p_self, 7)
        exit_14d = _exit_prob_within(fallback_p_self, 14)
        score    = _persistence_score(fallback_p_self, exit_7d)

        return RegimePersistenceResult(
            regime                  = current_regime,
            self_transition_prob    = fallback_p_self,
            expected_duration_days  = round(exp_dur,  1),
            regime_half_life_days   = round(half_lif, 1),
            remaining_duration_days = remain,
            exit_prob_7d            = round(exit_7d  * 100, 1),
            exit_prob_14d           = round(exit_14d * 100, 1),
            persistence_score       = score,
            persistence_label       = _persistence_label(score),
            most_likely_next        = "UNKNOWN",
            next_regime_probs       = {},
            interpretation          = f"[Fallback] {current_regime}: ~{remain:.0f}d remaining (est.)",
        )

    row = transition_matrix[current_regime]
    p_self  = float(row.get(current_regime, 0.80))
    p_self  = max(0.01, min(0.99, p_self))    # clamp to avoid math edge cases

    # ── Core calculations ────────────────────────────────────────────────────
    exp_dur   = round(_expected_duration(p_self), 1)
    half_lif  = round(_half_life(p_self),         1)
    remain    = _remaining_duration(exp_dur, half_lif)
    exit_7d   = _exit_prob_within(p_self, 7)
    exit_14d  = _exit_prob_within(p_self, 14)
    score     = _persistence_score(p_self, exit_7d)
    label     = _persistence_label(score)

    # ── Next regime probabilities ────────────────────────────────────────────
    next_probs = {
        regime: round(prob, 4)
        for regime, prob in row.items()
        if regime != current_regime
    }
    # Normalise next-regime probs to sum to 1 (conditional on exit)
    total_exit = sum(next_probs.values())
    if total_exit > 0:
        next_probs = {k: round(v / total_exit, 4) for k, v in next_probs.items()}

    most_likely_next = max(next_probs, key=next_probs.get) if next_probs else "UNKNOWN"

    # ── Interpretation ────────────────────────────────────────────────────────
    urgency = ""
    if exit_7d > 0.50:
        urgency = " ⚠️ EXIT likely within 7d"
    elif exit_7d > 0.30:
        urgency = " 🟡 Moderate exit probability"

    interp = (
        f"{current_regime} [{label}]: E={exp_dur:.0f}d | Remaining≈{remain:.0f}d | "
        f"Half-life={half_lif:.0f}d | Exit(7d)={exit_7d*100:.0f}% | "
        f"Next→{most_likely_next}{urgency}"
    )

    logger.info(
        "[persistence] %s p_self=%.3f E=%.1fd half_life=%.1fd "
        "exit_7d=%.0f%% exit_14d=%.0f%% score=%.0f label=%s next=%s",
        current_regime, p_self, exp_dur, half_lif,
        exit_7d * 100, exit_14d * 100, score, label, most_likely_next,
    )

    return RegimePersistenceResult(
        regime                  = current_regime,
        self_transition_prob    = round(p_self, 4),
        expected_duration_days  = exp_dur,
        regime_half_life_days   = half_lif,
        remaining_duration_days = remain,
        exit_prob_7d            = round(exit_7d  * 100, 1),
        exit_prob_14d           = round(exit_14d * 100, 1),
        persistence_score       = score,
        persistence_label       = label,
        most_likely_next        = most_likely_next,
        next_regime_probs       = next_probs,
        interpretation          = interp,
    )
