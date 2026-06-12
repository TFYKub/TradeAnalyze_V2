"""
Conviction Engine  (Phase 18)
===============================
Aggregates signals from all upstream engines into a single, calibrated
conviction score (0–100) that drives position sizing and trade filtering.

This is the final arbiter before a trade is approved.

Input signals (8 dimensions):
  Regime      (25%) — quality of the detected market regime
  Trend       (15%) — EMA + structure alignment
  Structure   (10%) — market structure clarity (HH-HL / LL-LH)
  Volatility  (10%) — vol regime suitability for the trade direction
  Flow        (15%) — derivatives flow alignment (funding, OI, L/S)
  Breadth     (10%) — market-wide participation
  Liquidity   (10%) — global macro liquidity environment
  Forecast    ( 5%) — ML forecast direction alignment

Conviction Tiers:
  90–100 → FULL SIZE    (1.0× Kelly)
  70–89  → NORMAL SIZE  (0.75× Kelly)
  50–69  → HALF SIZE    (0.50× Kelly)
  < 50   → NO TRADE     (skip entirely)

Position Sizing Integration:
  kelly_multiplier = conviction_sizing_mult × base_kelly
  This is the final sizing gate — overrides all earlier signals.

Design Principle:
  Conviction Engine never generates signals — it only aggregates.
  All component engines run independently; conviction only reads outputs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── Conviction Tiers ──────────────────────────────────────────────────────────
TIER_FULL   = 90.0
TIER_NORMAL = 70.0
TIER_HALF   = 50.0

TIER_LABELS = {
    "FULL SIZE":   (TIER_FULL,   1.00),
    "NORMAL SIZE": (TIER_NORMAL, 0.75),
    "HALF SIZE":   (TIER_HALF,   0.50),
    "NO TRADE":    (0.0,         0.00),
}


@dataclass(frozen=True)
class ConvictionResult:
    conviction_score:      float          # 0–100
    conviction_tier:       str            # FULL SIZE | NORMAL SIZE | HALF SIZE | NO TRADE
    kelly_multiplier:      float          # 0.0 | 0.50 | 0.75 | 1.00
    trade_allowed:         bool

    # Component breakdown
    component_scores:      dict[str, float]   # {signal_name: raw_score}
    component_weights:     dict[str, float]   # {signal_name: weight}
    weighted_scores:       dict[str, float]   # {signal_name: score × weight}

    # Diagnostics
    weakest_signal:        str            # which component is dragging score down
    strongest_signal:      str
    alignment_count:       int            # how many components agree on direction
    regime_persistence_ok: bool           # True if regime not about to exit

    interpretation:        str


# ── Component Score Converters ────────────────────────────────────────────────
def _regime_to_score(regime: str, regime_confidence: float) -> float:
    """Convert regime + confidence to 0–100 conviction component."""
    base = {
        "STRONG_BULL": 100.0,
        "BULL":         80.0,
        "RANGE":        40.0,   # range = low directional conviction
        "BEAR":         80.0,
        "STRONG_BEAR": 100.0,
    }.get(regime, 40.0)
    return round(base * min(1.0, regime_confidence / 100.0), 1)


def _trend_to_score(ema_alignment: float, structure_trend: str) -> float:
    """EMA alignment + structure clarity."""
    struct_bonus = {
        "BULLISH": 10.0, "BEARISH": 10.0,
        "MIXED": -10.0,  "UNKNOWN": -20.0,
    }.get(structure_trend, 0.0)
    return round(max(0.0, min(100.0, ema_alignment + struct_bonus)), 1)


def _structure_to_score(structure_clarity: float, structure_trend: str) -> float:
    base = {"BULLISH": 90.0, "BEARISH": 90.0, "MIXED": 45.0, "UNKNOWN": 20.0}.get(
        structure_trend, 20.0)
    return round((base * 0.5 + structure_clarity * 0.5), 1)


def _volatility_to_score(vol_regime: str, direction: str) -> float:
    """
    Normal vol is ideal for directional trades.
    High/panic vol discounts conviction.
    """
    return {
        "NORMAL_VOL": 90.0,
        "LOW_VOL":    75.0,   # low vol OK, may lack momentum
        "HIGH_VOL":   45.0,   # high vol = wide stops, risky
        "PANIC_VOL":  10.0,   # panic = no directional trades
    }.get(vol_regime, 60.0)


def _flow_to_score(
    flow_score: float,
    flow_direction: str,
    trade_direction: str,
    flow_regime: str,
) -> float:
    """
    Flow aligned with trade direction → high score.
    Flow opposed → penalise. NEUTRAL flow → neutral score.
    """
    if flow_regime in ("SHORT_SQUEEZE", "LONG_SQUEEZE"):
        # Squeeze regimes boost conviction heavily if aligned
        if (flow_regime == "SHORT_SQUEEZE" and trade_direction == "LONG") or \
           (flow_regime == "LONG_SQUEEZE"  and trade_direction == "SHORT"):
            return 95.0
        return 20.0   # wrong side of squeeze

    if flow_direction == "NEUTRAL":
        return 50.0

    # Alignment check
    flow_bull = flow_direction == "BULLISH"
    trade_long = trade_direction == "LONG"
    if flow_bull == trade_long:
        return round(50.0 + (flow_score - 50.0) * 1.0, 1)    # aligned: use raw flow score
    else:
        return round(50.0 - (flow_score - 50.0) * 0.8, 1)    # opposed: invert


def _breadth_to_score(breadth_score: float, breadth_regime: str) -> float:
    """Breadth score directly maps (already 0–100)."""
    return round(max(0.0, min(100.0, breadth_score)), 1)


def _liquidity_to_score(
    liquidity_score: float,
    liquidity_regime: str,
    risk_multiplier: float,
) -> float:
    """
    CRISIS → 0, RISK_OFF → 30, RECOVERY → 65, RISK_ON → 90+
    """
    base = {
        "RISK_ON":   min(100.0, 70.0 + (liquidity_score - 70.0) * 1.5),
        "RECOVERY":  60.0,
        "RISK_OFF":  35.0,
        "CRISIS":    5.0,
    }.get(liquidity_regime, liquidity_score)
    return round(max(0.0, min(100.0, base)), 1)


def _forecast_to_score(
    forecast_direction: str,
    probability_up_20d: float,
    trade_direction: str,
    forecast_confidence: float,
) -> float:
    """
    Forecast aligned with trade direction → boost.
    High forecast confidence amplifies.
    """
    conf_weight = forecast_confidence / 100.0

    trade_long = trade_direction == "LONG"
    fc_bull    = forecast_direction == "BULLISH"

    if forecast_direction == "NEUTRAL":
        base = 50.0
    elif fc_bull == trade_long:
        # Aligned
        base = 50.0 + (probability_up_20d - 0.50) * 100.0
    else:
        # Opposed
        base = 50.0 - (probability_up_20d - 0.50) * 100.0

    base = max(0.0, min(100.0, base))
    # Discount for low-confidence forecasts
    return round(50.0 + (base - 50.0) * conf_weight, 1)


# ── Tier Classification ────────────────────────────────────────────────────────
def _classify_tier(score: float) -> tuple[str, float, bool]:
    """Returns (tier_label, kelly_multiplier, trade_allowed)."""
    if score >= TIER_FULL:
        return "FULL SIZE",   1.00, True
    if score >= TIER_NORMAL:
        return "NORMAL SIZE", 0.75, True
    if score >= TIER_HALF:
        return "HALF SIZE",   0.50, True
    return "NO TRADE",        0.00, False


# ── Main Entry ─────────────────────────────────────────────────────────────────
def compute_conviction(
    # Regime
    regime:             str,
    regime_confidence:  float,
    # Trend
    ema_alignment:      float,
    structure_trend:    str,
    structure_clarity:  float,
    # Volatility
    vol_regime:         str,
    # Direction
    trade_direction:    str,        # "LONG" | "SHORT" | "WAIT"
    # Flow (optional — defaults to neutral if unavailable)
    flow_score:         float = 50.0,
    flow_direction:     str   = "NEUTRAL",
    flow_regime:        str   = "NEUTRAL",
    # Breadth (optional)
    breadth_score:      float = 50.0,
    breadth_regime:     str   = "NEUTRAL",
    # Liquidity (optional)
    liquidity_score:    float = 55.0,
    liquidity_regime:   str   = "RISK_ON",
    risk_multiplier:    float = 1.0,
    # Forecast (optional)
    forecast_direction: str   = "NEUTRAL",
    probability_up_20d: float = 0.50,
    forecast_confidence:float = 50.0,
    # Persistence (optional)
    persistence_score:  float = 70.0,
    persistence_label:  str   = "ESTABLISHED",
) -> ConvictionResult:
    """
    Compute final conviction score aggregating all upstream signals.

    Parameters
    ----------
    All parameters are optional except the core regime/trend/vol/direction.
    Missing values default to neutral — the engine degrades gracefully.

    Returns
    -------
    ConvictionResult — always returns, never raises
    """
    # ── Early exit for WAIT direction ────────────────────────────────────────
    if trade_direction == "WAIT":
        return ConvictionResult(
            conviction_score   = 0.0,
            conviction_tier    = "NO TRADE",
            kelly_multiplier   = 0.0,
            trade_allowed      = False,
            component_scores   = {},
            component_weights  = {},
            weighted_scores    = {},
            weakest_signal     = "direction",
            strongest_signal   = "n/a",
            alignment_count    = 0,
            regime_persistence_ok = True,
            interpretation     = "NO TRADE — direction=WAIT",
        )

    # ── Component scores ─────────────────────────────────────────────────────
    WEIGHTS = {
        "regime":     0.25,
        "trend":      0.15,
        "structure":  0.10,
        "volatility": 0.10,
        "flow":       0.15,
        "breadth":    0.10,
        "liquidity":  0.10,
        "forecast":   0.05,
    }

    scores = {
        "regime":     _regime_to_score(regime, regime_confidence),
        "trend":      _trend_to_score(ema_alignment, structure_trend),
        "structure":  _structure_to_score(structure_clarity, structure_trend),
        "volatility": _volatility_to_score(vol_regime, trade_direction),
        "flow":       _flow_to_score(flow_score, flow_direction, trade_direction, flow_regime),
        "breadth":    _breadth_to_score(breadth_score, breadth_regime),
        "liquidity":  _liquidity_to_score(liquidity_score, liquidity_regime, risk_multiplier),
        "forecast":   _forecast_to_score(
                          forecast_direction, probability_up_20d,
                          trade_direction, forecast_confidence
                      ),
    }

    weighted = {k: round(scores[k] * WEIGHTS[k], 2) for k in scores}
    composite = round(sum(weighted.values()), 1)

    # ── Persistence discount ─────────────────────────────────────────────────
    # If regime is EXHAUSTED (about to transition), discount conviction by up to 15 pts
    persistence_discount = 0.0
    persistence_ok = True
    if persistence_label == "EXHAUSTED":
        persistence_discount = 15.0
        persistence_ok = False
    elif persistence_label == "MATURING" and persistence_score < 50:
        persistence_discount = 7.0

    composite = round(max(0.0, composite - persistence_discount), 1)

    # ── Tier classification ──────────────────────────────────────────────────
    tier, kelly_mult, trade_allowed = _classify_tier(composite)

    # ── Diagnostics ─────────────────────────────────────────────────────────
    weakest   = min(scores, key=scores.get)
    strongest = max(scores, key=scores.get)

    # Count aligned signals (score > 55 = bullish bias, < 45 = bearish bias)
    if trade_direction == "LONG":
        aligned = sum(1 for s in scores.values() if s > 55.0)
    elif trade_direction == "SHORT":
        aligned = sum(1 for s in scores.values() if s > 55.0)   # same logic — we inverted in scorers
    else:
        aligned = 0

    # ── Interpretation ────────────────────────────────────────────────────────
    tier_emoji = {
        "FULL SIZE":   "🏆",
        "NORMAL SIZE": "✅",
        "HALF SIZE":   "⚡",
        "NO TRADE":    "⏸️",
    }.get(tier, "❓")

    disc_note = f" [-{persistence_discount:.0f} persistence discount]" if persistence_discount > 0 else ""
    interp = (
        f"{tier_emoji} {tier} — conviction={composite:.0f}/100{disc_note} | "
        f"Kelly×{kelly_mult:.2f} | "
        f"Weak: {weakest}({scores[weakest]:.0f}) | "
        f"Strong: {strongest}({scores[strongest]:.0f}) | "
        f"Aligned: {aligned}/8"
    )

    logger.info(
        "[conviction] score=%.1f tier=%s kelly=%.2f aligned=%d/8 "
        "weak=%s(%s) persistence=%s",
        composite, tier, kelly_mult, aligned,
        weakest, persistence_label, persistence_label,
    )

    return ConvictionResult(
        conviction_score      = composite,
        conviction_tier       = tier,
        kelly_multiplier      = kelly_mult,
        trade_allowed         = trade_allowed,
        component_scores      = scores,
        component_weights     = WEIGHTS,
        weighted_scores       = weighted,
        weakest_signal        = weakest,
        strongest_signal      = strongest,
        alignment_count       = aligned,
        regime_persistence_ok = persistence_ok,
        interpretation        = interp,
    )
