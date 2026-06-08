"""
Markov Calibration Engine  (Phase 1, Fix 4)
=============================================
Fixes 100% probability issue via:
  1. Softmax temperature scaling
  2. Laplace smoothing (add ε to all states)
  3. Confidence capping at MAX_REGIME_CONFIDENCE (95%)
  4. Calibration score based on entropy spread

Applied to RegimeResult.regime_probs_all before downstream use.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

TEMPERATURE  = 1.5    # > 1 → softer (less extreme) probabilities
LAPLACE_EPS  = 0.02   # add 2% to each state before normalising
MAX_CONF     = 95.0   # hard cap on confidence
MIN_CONF     = 5.0    # floor


@dataclass(frozen=True)
class CalibrationResult:
    raw_probs:         dict[str, float]   # before calibration
    calibrated_probs:  dict[str, float]   # after softmax + smoothing
    top_regime:        str
    calibrated_conf:   float              # 0–100
    calibration_score: float              # 0–100 (100 = well-spread)
    was_clipped:       bool               # True if confidence was capped


def _softmax(probs: dict[str, float], temperature: float) -> dict[str, float]:
    """Temperature-scaled softmax to reduce overconfidence."""
    keys   = list(probs.keys())
    logits = [math.log(max(v, 1e-9)) / temperature for v in probs.values()]
    max_l  = max(logits)
    exp    = [math.exp(l - max_l) for l in logits]
    total  = sum(exp)
    return {k: e / total for k, e in zip(keys, exp)}


def _laplace_smooth(probs: dict[str, float], eps: float) -> dict[str, float]:
    """Add small ε to every state then renormalise."""
    smoothed = {k: v + eps for k, v in probs.items()}
    total    = sum(smoothed.values())
    return {k: v / total for k, v in smoothed.items()}


def _entropy_score(probs: dict[str, float]) -> float:
    """
    Calibration score based on entropy.
    High entropy (uniform) = 0, Low entropy (one state dominates) = 100.
    We want moderate spread, so: score = 1 - (entropy / max_entropy).
    """
    n = len(probs)
    if n <= 1:
        return 100.0
    entropy     = -sum(p * math.log(p + 1e-9) for p in probs.values())
    max_entropy = math.log(n)
    clarity     = 1.0 - (entropy / max_entropy)
    return round(clarity * 100, 1)


def calibrate_regime_probs(
    raw_probs:   dict[str, float],
    temperature: float = TEMPERATURE,
    eps:         float = LAPLACE_EPS,
) -> CalibrationResult:
    """
    Apply softmax + Laplace smoothing + confidence cap.

    Parameters
    ----------
    raw_probs   : regime probability dict from HMM posterior
    temperature : softmax temperature (higher = softer)
    eps         : Laplace smoothing epsilon

    Returns
    -------
    CalibrationResult
    """
    from config.thresholds import THRESHOLDS

    # Step 1: Laplace smooth
    smoothed = _laplace_smooth(raw_probs, eps)

    # Step 2: Softmax with temperature
    calibrated = _softmax(smoothed, temperature)

    # Step 3: Re-normalise to ensure sum = 1.0
    total = sum(calibrated.values())
    calibrated = {k: v / total for k, v in calibrated.items()}

    # Top regime and confidence
    top_regime   = max(calibrated, key=calibrated.get)
    raw_conf     = calibrated[top_regime] * 100
    was_clipped  = raw_conf > THRESHOLDS.MAX_REGIME_CONFIDENCE
    final_conf   = min(THRESHOLDS.MAX_REGIME_CONFIDENCE,
                       max(MIN_CONF, raw_conf))

    cal_score = _entropy_score(calibrated)

    logger.debug(
        "[markov_cal] top=%s raw_conf=%.1f → cal_conf=%.1f clipped=%s score=%.1f",
        top_regime, raw_conf, final_conf, was_clipped, cal_score
    )

    return CalibrationResult(
        raw_probs         = {k: round(v, 4) for k, v in raw_probs.items()},
        calibrated_probs  = {k: round(v, 4) for k, v in calibrated.items()},
        top_regime        = top_regime,
        calibrated_conf   = round(final_conf, 1),
        calibration_score = cal_score,
        was_clipped       = was_clipped,
    )
