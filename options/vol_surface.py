"""
Volatility Surface Engine  (Phase 5)
=======================================
Estimates put/call skew, smile, and term structure
from the enriched option chain rows.

Put Skew    = avg IV(OTM puts)  - avg IV(ATM calls)
Call Skew   = avg IV(OTM calls) - avg IV(ATM calls)
Smile       = (put_skew + call_skew) / 2  (positive = wings expensive)
Term Struct = IV(30D) - IV(60D)  (positive = near-term fear)

Vol Surface Score: 0–100
  High score = vol surface is informative (large skew signals)
"""
from __future__ import annotations
import logging
import statistics
from dataclasses import dataclass

logger = logging.getLogger(__name__)

ATM_DELTA_RANGE = (0.40, 0.60)   # treat delta 0.40–0.60 as ATM
OTM_PUT_DELTA   = (-0.35, -0.15) # OTM put delta range
OTM_CALL_DELTA  = (0.15, 0.35)   # OTM call delta range


@dataclass(frozen=True)
class VolSurfaceResult:
    put_skew:        float    # OTM put IV premium over ATM
    call_skew:       float    # OTM call IV premium over ATM
    smile:           float    # avg wing premium
    term_structure:  float    # IV(30D) - IV(60D)
    atm_iv_30d:      float
    atm_iv_60d:      float
    vol_surface_score: float  # 0–100
    interpretation:  str
    skew_signal:     str      # FEAR | GREED | NEUTRAL | BALANCED


def _mean_iv(rows: list[dict]) -> float:
    ivs = [float(r["iv"]) for r in rows if r.get("iv", 0) > 0]
    return statistics.mean(ivs) if ivs else 0.0


def compute_vol_surface(enriched_chain: list[dict]) -> VolSurfaceResult:
    """
    Compute skew, smile, and term structure from enriched option rows.

    Parameters
    ----------
    enriched_chain : list of rows from greeks_pipeline (with delta, iv, dte_bucket)
    """
    if not enriched_chain:
        return VolSurfaceResult(
            put_skew=0, call_skew=0, smile=0, term_structure=0,
            atm_iv_30d=0, atm_iv_60d=0, vol_surface_score=0,
            interpretation="No chain data", skew_signal="NEUTRAL",
        )

    def _filter(opt_type, delta_range, dte_bucket=None):
        rows = [r for r in enriched_chain
                if r.get("option_type") == opt_type
                and r.get("delta") is not None
                and delta_range[0] <= abs(float(r["delta"])) <= delta_range[1]]
        if dte_bucket:
            rows = [r for r in rows if r.get("dte_bucket") == dte_bucket]
        return rows

    # ATM (30D)
    atm_calls_30 = _filter("call", ATM_DELTA_RANGE, 30)
    atm_iv_30    = _mean_iv(atm_calls_30)

    # ATM (60D)
    atm_calls_60 = _filter("call", ATM_DELTA_RANGE, 60)
    atm_iv_60    = _mean_iv(atm_calls_60)
    if not atm_iv_60 and not atm_calls_60:
        atm_iv_60 = atm_iv_30   # fallback

    # OTM puts/calls
    otm_puts  = _filter("put",  OTM_PUT_DELTA)
    otm_calls = _filter("call", OTM_CALL_DELTA)
    otm_put_iv  = _mean_iv(otm_puts)
    otm_call_iv = _mean_iv(otm_calls)

    atm_ref = atm_iv_30 or 0.01
    put_skew  = round(otm_put_iv  - atm_ref, 4) if otm_put_iv  else 0.0
    call_skew = round(otm_call_iv - atm_ref, 4) if otm_call_iv else 0.0
    smile     = round((abs(put_skew) + abs(call_skew)) / 2, 4)
    term_struct = round(atm_iv_30 - atm_iv_60, 4) if atm_iv_60 else 0.0

    # Score: how informative the surface is
    score = min(100.0, (abs(put_skew) + abs(call_skew) + abs(term_struct)) / atm_ref * 200)

    # Skew signal
    if put_skew > 0.03:
        skew_sig = "FEAR"
        interp   = f"Put skew {put_skew:+.3f} → market buying downside protection (bearish fear)"
    elif call_skew > 0.03:
        skew_sig = "GREED"
        interp   = f"Call skew {call_skew:+.3f} → market chasing upside (bullish greed)"
    elif abs(put_skew - call_skew) < 0.01:
        skew_sig = "BALANCED"
        interp   = "Symmetric smile — balanced put/call demand"
    else:
        skew_sig = "NEUTRAL"
        interp   = f"Term structure {term_struct:+.3f}, skew near zero"

    if term_struct > 0.02:
        interp += " | Contango term structure (near-term fear)"
    elif term_struct < -0.02:
        interp += " | Backwardation (longer-term concern)"

    return VolSurfaceResult(
        put_skew=put_skew, call_skew=call_skew, smile=round(smile, 4),
        term_structure=term_struct, atm_iv_30d=round(atm_iv_30, 4),
        atm_iv_60d=round(atm_iv_60, 4), vol_surface_score=round(score, 1),
        interpretation=interp, skew_signal=skew_sig,
    )
