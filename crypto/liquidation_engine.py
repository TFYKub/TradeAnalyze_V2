"""
Liquidation Engine  (Phase 7)
================================
Estimates liquidation zones from current price + funding data.

Long liquidations cluster BELOW current price at leverage multiples.
Short liquidations cluster ABOVE current price at leverage multiples.

Typical leverage tiers (retail crypto):
  5×  → liquidation at ±20% from entry
  10× → ±10%
  20× → ±5%
  50× → ±2%
  100×→ ±1%

Liquidity clusters = price levels where large stop orders accumulate
  (equal highs/lows, round numbers, swing points).
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

LEVERAGE_TIERS = [5, 10, 20, 50, 100]


@dataclass(frozen=True)
class LiquidationZone:
    price:         float
    direction:     str       # "LONG_LIQ" | "SHORT_LIQ"
    leverage:      int       # leverage tier
    estimated_size: str      # "SMALL" | "MEDIUM" | "LARGE"
    distance_pct:  float


@dataclass(frozen=True)
class LiquidationResult:
    symbol:              str
    current_price:       float
    long_liq_zones:      tuple[LiquidationZone, ...]
    short_liq_zones:     tuple[LiquidationZone, ...]
    liquidity_clusters:  tuple[float, ...]   # from swing highs/lows
    nearest_long_liq:    float | None
    nearest_short_liq:   float | None
    cascade_risk:        str     # "HIGH" | "MODERATE" | "LOW"
    interpretation:      str


def _est_size(leverage: int) -> str:
    if leverage >= 50: return "LARGE"
    if leverage >= 20: return "MEDIUM"
    return "SMALL"


def compute_liquidation_zones(
    symbol:          str,
    price:           float,
    swing_highs:     list[float],
    swing_lows:      list[float],
    crowded_long:    bool = False,
    crowded_short:   bool = False,
) -> LiquidationResult:
    """
    Estimate liquidation zones based on price levels and leverage tiers.

    Parameters
    ----------
    symbol       : ticker
    price        : current spot price
    swing_highs  : recent swing high prices (for liquidity clusters)
    swing_lows   : recent swing low prices
    crowded_long : from FundingRateResult
    crowded_short: from FundingRateResult
    """
    long_zones: list[LiquidationZone] = []
    short_zones: list[LiquidationZone] = []

    for lev in LEVERAGE_TIERS:
        margin_pct = 1.0 / lev
        # Long liquidation: price drops by 1/leverage (maintenance margin ≈ 0.5%)
        long_liq_price  = round(price * (1 - margin_pct * 0.90), 4)
        short_liq_price = round(price * (1 + margin_pct * 0.90), 4)

        # Inflate estimated size if crowded
        size = _est_size(lev)
        if crowded_long  and lev in (20, 50, 100): size = "LARGE"
        if crowded_short and lev in (20, 50, 100): size = "LARGE"

        long_zones.append(LiquidationZone(
            price=long_liq_price, direction="LONG_LIQ", leverage=lev,
            estimated_size=size, distance_pct=round(margin_pct * 90, 2)
        ))
        short_zones.append(LiquidationZone(
            price=short_liq_price, direction="SHORT_LIQ", leverage=lev,
            estimated_size=size, distance_pct=round(margin_pct * 90, 2)
        ))

    # Liquidity clusters: equal highs / equal lows / round numbers
    clusters: list[float] = []
    for h in swing_highs[-5:]:
        clusters.append(round(h, 0))
    for l in swing_lows[-5:]:
        clusters.append(round(l, 0))
    # Round number clusters near price
    for mult in [0.90, 0.95, 1.05, 1.10]:
        rn = round(price * mult / 1000) * 1000
        if rn > 0:
            clusters.append(float(rn))
    clusters = sorted(set(clusters))

    # Nearest zones
    long_below   = sorted([z for z in long_zones  if z.price < price], key=lambda z: -z.price)
    short_above  = sorted([z for z in short_zones if z.price > price], key=lambda z:  z.price)
    nearest_long  = long_below[0].price  if long_below  else None
    nearest_short = short_above[0].price if short_above else None

    # Cascade risk: HIGH if crowded + large zones near current price
    if (crowded_long or crowded_short) and any(z.estimated_size == "LARGE" for z in long_zones + short_zones):
        cascade = "HIGH"
    elif crowded_long or crowded_short:
        cascade = "MODERATE"
    else:
        cascade = "LOW"

    interp = (
        f"{'⚠️ HIGH cascade risk' if cascade == 'HIGH' else cascade + ' cascade risk'}: "
        f"{'Crowded longs may cascade below ' + str(nearest_long) if crowded_long else ''}"
        f"{'Crowded shorts may cascade above ' + str(nearest_short) if crowded_short else ''}"
        f"{'Normal liquidation distribution' if not crowded_long and not crowded_short else ''}"
    ).strip()

    logger.info("[liq_engine] %s cascade=%s nearest_long=%.2f nearest_short=%.2f",
                symbol, cascade, nearest_long or 0, nearest_short or 0)

    return LiquidationResult(
        symbol             = symbol,
        current_price      = price,
        long_liq_zones     = tuple(long_zones),
        short_liq_zones    = tuple(short_zones),
        liquidity_clusters = tuple(clusters[:10]),
        nearest_long_liq   = nearest_long,
        nearest_short_liq  = nearest_short,
        cascade_risk       = cascade,
        interpretation     = interp,
    )
