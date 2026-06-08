"""
Institutional Stop Loss Engine  (Phase 1, Fix 2)
==================================================
Generates 4 stop types and selects the best:

  1. ATR Stop        — price ± ATR × multiplier
  2. Structure Stop  — below/above last swing low/high
  3. Swing Stop      — below/above 2nd-last swing (wider)
  4. Volatility Stop — ATR-based with vol regime multiplier

Selection rule:
  LONG:  selected = max(atr_stop, structure_stop, swing_stop, vol_stop)
         (highest stop = least risk, tightest below entry)
  SHORT: selected = min(atr_stop, structure_stop, swing_stop, vol_stop)
         (lowest stop = tightest above entry)

Output: InstitutionalStopResult
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Multipliers
ATR_MULT_NORMAL   = 1.0
ATR_MULT_HIGH_VOL = 1.5
ATR_MULT_LOW_VOL  = 0.75
MIN_STOP_PCT      = 0.005    # minimum stop distance = 0.5% of price


@dataclass(frozen=True)
class InstitutionalStopResult:
    direction:       str
    entry:           float

    atr_stop:        float
    structure_stop:  float
    swing_stop:      float
    volatility_stop: float
    selected_stop:   float
    stop_reason:     str

    risk:            float   # $ distance: entry → selected_stop
    risk_pct:        float   # risk as % of entry price

    min_tp_for_2rr:  float   # entry ± 2× risk (minimum acceptable TP)
    min_tp_for_3rr:  float   # entry ± 3× risk

    # Validity
    stop_valid:      bool
    invalid_reason:  str


def compute_institutional_stop(
    direction:    str,
    entry:        float,
    atr:          float,
    swing_low:    float | None = None,
    swing_high:   float | None = None,
    swing_low_2:  float | None = None,   # 2nd-last swing low (wider)
    swing_high_2: float | None = None,   # 2nd-last swing high
    vol_regime:   str = "NORMAL",        # LOW | NORMAL | HIGH | PANIC
) -> InstitutionalStopResult:
    """
    Calculate all stop types and select the institutional best.

    Parameters
    ----------
    direction    : "LONG" | "SHORT"
    entry        : entry price
    atr          : ATR-14 value
    swing_low    : most recent swing low price
    swing_high   : most recent swing high price
    swing_low_2  : second-to-last swing low
    swing_high_2 : second-to-last swing high
    vol_regime   : volatility regime for multiplier selection
    """

    # ATR multiplier by vol regime
    mult = {
        "LOW":    ATR_MULT_LOW_VOL,
        "NORMAL": ATR_MULT_NORMAL,
        "HIGH":   ATR_MULT_HIGH_VOL,
        "PANIC":  ATR_MULT_HIGH_VOL * 1.3,
    }.get(vol_regime.upper(), ATR_MULT_NORMAL)

    min_stop_dist = entry * MIN_STOP_PCT

    if direction == "LONG":
        # 1. ATR Stop: entry - ATR × 1.0
        atr_stop = entry - atr * ATR_MULT_NORMAL

        # 2. Structure Stop: below most recent swing low - ATR buffer
        structure_stop = (swing_low - atr * 0.5) if swing_low else entry - atr * 1.5

        # 3. Swing Stop: below 2nd swing low (wider, safer)
        swing_stop = (swing_low_2 - atr * 0.25) if swing_low_2 else structure_stop - atr * 0.5

        # 4. Volatility Stop: vol-adjusted
        vol_stop = entry - atr * mult

        # Selected: maximum of all stops (closest to entry = tightest, best risk)
        # But not so close it's below min stop distance
        candidates = [atr_stop, structure_stop, swing_stop, vol_stop]
        valid_candidates = [s for s in candidates if entry - s >= min_stop_dist]
        selected = max(valid_candidates) if valid_candidates else entry - min_stop_dist

        risk = entry - selected
        min_tp_2rr = entry + risk * 2
        min_tp_3rr = entry + risk * 3

        # Stop reason
        stop_reason = _stop_reason_long(
            selected, atr_stop, structure_stop, swing_stop, vol_stop
        )

    elif direction == "SHORT":
        atr_stop        = entry + atr * ATR_MULT_NORMAL
        structure_stop  = (swing_high + atr * 0.5) if swing_high else entry + atr * 1.5
        swing_stop      = (swing_high_2 + atr * 0.25) if swing_high_2 else structure_stop + atr * 0.5
        vol_stop        = entry + atr * mult

        candidates = [atr_stop, structure_stop, swing_stop, vol_stop]
        valid_candidates = [s for s in candidates if s - entry >= min_stop_dist]
        selected = min(valid_candidates) if valid_candidates else entry + min_stop_dist

        risk = selected - entry
        min_tp_2rr = entry - risk * 2
        min_tp_3rr = entry - risk * 3

        stop_reason = _stop_reason_short(
            selected, atr_stop, structure_stop, swing_stop, vol_stop
        )

    else:
        return InstitutionalStopResult(
            direction="WAIT", entry=entry,
            atr_stop=entry, structure_stop=entry, swing_stop=entry, volatility_stop=entry,
            selected_stop=entry, stop_reason="No direction",
            risk=0, risk_pct=0,
            min_tp_for_2rr=entry, min_tp_for_3rr=entry,
            stop_valid=False, invalid_reason="Direction is WAIT",
        )

    risk_pct    = risk / entry * 100 if entry > 0 else 0
    stop_valid  = risk > 0
    invalid_reason = "" if stop_valid else "Zero risk distance"

    logger.info(
        "[stop_engine] %s entry=%.4f stop=%.4f risk=%.4f (%.2f%%) reason=%s",
        direction, entry, selected, risk, risk_pct, stop_reason,
    )

    return InstitutionalStopResult(
        direction       = direction,
        entry           = round(entry, 4),
        atr_stop        = round(atr_stop, 4),
        structure_stop  = round(structure_stop, 4),
        swing_stop      = round(swing_stop, 4),
        volatility_stop = round(vol_stop, 4),
        selected_stop   = round(selected, 4),
        stop_reason     = stop_reason,
        risk            = round(risk, 4),
        risk_pct        = round(risk_pct, 3),
        min_tp_for_2rr  = round(min_tp_2rr, 4),
        min_tp_for_3rr  = round(min_tp_3rr, 4),
        stop_valid      = stop_valid,
        invalid_reason  = invalid_reason,
    )


def _stop_reason_long(selected, atr, structure, swing, vol) -> str:
    closest = max([atr, structure, swing, vol])
    if abs(selected - closest) < 0.001:
        if abs(selected - structure) < 0.001:
            return "Structure Stop (swing low - ATR buffer)"
        if abs(selected - atr) < 0.001:
            return "ATR Stop (entry - 1× ATR)"
        if abs(selected - swing) < 0.001:
            return "Swing Stop (2nd swing low)"
        return "Volatility Stop (vol-adjusted ATR)"
    return "ATR Stop (default)"


def _stop_reason_short(selected, atr, structure, swing, vol) -> str:
    closest = min([atr, structure, swing, vol])
    if abs(selected - structure) < 0.001:
        return "Structure Stop (swing high + ATR buffer)"
    if abs(selected - atr) < 0.001:
        return "ATR Stop (entry + 1× ATR)"
    if abs(selected - swing) < 0.001:
        return "Swing Stop (2nd swing high)"
    return "Volatility Stop (vol-adjusted ATR)"
