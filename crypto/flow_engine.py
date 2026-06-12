"""
Flow Engine  (Phase 13)
========================
Aggregates all crypto derivatives flow signals into a single directional
flow score and regime classification.

Sources combined:
  • Funding Rate          — crowded long/short detection
  • Open Interest + Delta — new money entering vs. covering
  • Long/Short Ratio      — OKX taker top-account ratio
  • Liquidation Clusters  — cascade risk proximity

Flow Score (0–100):
  0–25   → BEARISH_FLOW  (strong selling / shorts entering)
  26–40  → BEARISH_FLOW  (moderate)
  41–59  → NEUTRAL
  60–74  → BULLISH_FLOW  (moderate)
  75–100 → BULLISH_FLOW  (strong)

Special regime overrides:
  SHORT_SQUEEZE : crowded short + rising OI + price breakout
  LONG_SQUEEZE  : crowded long  + rising OI + price breakdown

Output: FlowEngineResult
  flow_score      : float  0–100
  flow_direction  : str    BULLISH | BEARISH | NEUTRAL
  flow_confidence : float  0–100
  flow_regime     : str    BULLISH_FLOW | BEARISH_FLOW | SHORT_SQUEEZE | LONG_SQUEEZE | NEUTRAL
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger(__name__)

OKX_BASE  = "https://www.okx.com/api/v5/rubik"
TIMEOUT   = 10

# ── Data Contracts ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class LongShortResult:
    symbol:         str
    long_ratio:     float          # 0–1 fraction of longs
    short_ratio:    float          # 0–1 fraction of shorts
    ls_ratio:       float          # long / short (>1 = more longs)
    source:         str
    interpretation: str


@dataclass(frozen=True)
class FlowEngineResult:
    symbol:           str
    flow_score:       float          # 0–100
    flow_direction:   str            # BULLISH | BEARISH | NEUTRAL
    flow_confidence:  float          # 0–100
    flow_regime:      str            # BULLISH_FLOW | BEARISH_FLOW | SHORT_SQUEEZE | LONG_SQUEEZE | NEUTRAL

    # Component breakdown
    funding_score:    float          # 0–100 component
    oi_score:         float          # 0–100 component
    ls_score:         float          # 0–100 component
    liq_score:        float          # 0–100 component

    # Raw inputs summary
    funding_rate_pct: float
    funding_regime:   str
    oi_signal:        str
    ls_ratio:         float
    cascade_risk:     str

    interpretation:   str
    component_detail: dict[str, str]


# ── Long / Short Ratio Fetcher ──────────────────────────────────────────────────
def _fetch_long_short_ratio(symbol: str) -> Optional[LongShortResult]:
    """
    Fetch top-account long/short ratio from OKX.
    Endpoint: /api/v5/rubik/stat/contracts/long-short-account-ratio
    """
    currency = symbol.upper().split("-")[0].split("/")[0]
    try:
        r = requests.get(
            f"{OKX_BASE}/stat/contracts/long-short-account-ratio",
            params={"ccy": currency, "period": "5m"},
            timeout=TIMEOUT,
            headers={"User-Agent": "TradeAnalyze/1.4"},
        )
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            return None

        # Most recent record
        rec = data[0]
        ls_ratio = float(rec.get("longShortRatio", 1.0))
        long_r   = ls_ratio / (1 + ls_ratio)
        short_r  = 1 - long_r

        interp = (
            f"L/S={ls_ratio:.2f} — {'Longs dominant' if ls_ratio > 1.2 else 'Shorts dominant' if ls_ratio < 0.8 else 'Balanced'}"
        )

        logger.debug("[flow] OKX L/S ratio %s = %.3f", symbol, ls_ratio)
        return LongShortResult(
            symbol=symbol, long_ratio=round(long_r, 4),
            short_ratio=round(short_r, 4), ls_ratio=round(ls_ratio, 3),
            source="okx", interpretation=interp,
        )
    except Exception as exc:
        logger.debug("[flow] L/S fetch failed for %s: %s", symbol, exc)
        return None


# ── Component Scorers ──────────────────────────────────────────────────────────
def _score_funding(funding_regime: str, funding_rate_pct: float) -> float:
    """
    Convert funding regime → flow score component.
    CROWDED_LONG → bearish flow (score < 40)
    CROWDED_SHORT → bullish flow (score > 60)
    NEUTRAL → 50
    """
    base = {
        "CROWDED_LONG":  20.0,   # extreme longs crowded → bearish contrarian
        "HIGH_LONG":     35.0,   # elevated longs → mild bearish pressure
        "NEUTRAL":       50.0,
        "HIGH_SHORT":    65.0,   # elevated shorts → mild bullish pressure
        "CROWDED_SHORT": 80.0,   # extreme shorts → bullish contrarian
    }.get(funding_regime, 50.0)

    # Fine-tune by magnitude: extreme rates push further
    magnitude_adj = min(20.0, abs(funding_rate_pct) * 100)
    if funding_rate_pct > 0:
        base = max(10.0, base - magnitude_adj * 0.2)
    elif funding_rate_pct < 0:
        base = min(90.0, base + magnitude_adj * 0.2)

    return round(min(100.0, max(0.0, base)), 1)


def _score_oi(oi_signal: str, oi_trend: str) -> float:
    """
    Convert price × OI signal → directional score.
    CONFIRMATION (price↑ + OI↑)     → bullish → high score
    CONTINUATION (price↓ + OI↑)     → bearish → low score
    WEAK_RALLY   (price↑ + OI↓)     → unsustained → slightly bullish but low conviction
    CAPITULATION (price↓ + OI↓)     → potential bottom → neutral to bullish
    """
    return {
        "CONFIRMATION": 78.0,
        "WEAK_RALLY":   58.0,
        "NEUTRAL":      50.0,
        "UNKNOWN":      50.0,
        "CAPITULATION": 52.0,   # potential reversal; treat neutral with slight bullish bias
        "CONTINUATION": 22.0,
    }.get(oi_signal, 50.0)


def _score_long_short(ls: Optional[LongShortResult]) -> float:
    """
    Convert L/S ratio → contrarian flow score.
    Extreme longs crowded (L/S > 2.0) → bearish
    Extreme shorts crowded (L/S < 0.5) → bullish
    """
    if ls is None:
        return 50.0

    ratio = ls.ls_ratio
    if ratio >= 2.0:
        return 20.0   # very crowded longs → bearish
    if ratio >= 1.5:
        return 35.0
    if ratio >= 1.2:
        return 42.0
    if ratio <= 0.5:
        return 80.0   # very crowded shorts → bullish
    if ratio <= 0.7:
        return 65.0
    if ratio <= 0.9:
        return 56.0
    return 50.0


def _score_liquidation(cascade_risk: str, nearest_long_liq: Optional[float],
                       nearest_short_liq: Optional[float], price: float) -> float:
    """
    Cascade risk score — HIGH cascade risk near current price = dangerous for longs.
    """
    if cascade_risk == "HIGH":
        return 35.0   # liquidation cascade → bearish pressure
    if cascade_risk == "MODERATE":
        return 45.0
    return 55.0   # LOW cascade risk → slight bullish bias


# ── Regime Classification ──────────────────────────────────────────────────────
def _classify_flow(
    composite:       float,
    funding_regime:  str,
    oi_signal:       str,
    cascade_risk:    str,
    ls:              Optional[LongShortResult],
) -> tuple[str, str, float]:
    """
    Returns (flow_regime, flow_direction, confidence).
    """
    # Detect squeeze conditions
    crowded_short = funding_regime == "CROWDED_SHORT"
    crowded_long  = funding_regime == "CROWDED_LONG"
    oi_building   = oi_signal in ("CONFIRMATION", "CONTINUATION")

    if crowded_short and oi_building and composite > 65:
        regime    = "SHORT_SQUEEZE"
        direction = "BULLISH"
        conf      = min(90.0, 65.0 + (composite - 65.0) * 0.8)
        return regime, direction, round(conf, 1)

    if crowded_long and oi_building and composite < 35:
        regime    = "LONG_SQUEEZE"
        direction = "BEARISH"
        conf      = min(90.0, 65.0 + (35.0 - composite) * 0.8)
        return regime, direction, round(conf, 1)

    # Standard classification
    if composite >= 68.0:
        regime    = "BULLISH_FLOW"
        direction = "BULLISH"
        conf      = min(90.0, 55.0 + (composite - 68.0) * 1.2)
    elif composite >= 55.0:
        regime    = "BULLISH_FLOW"
        direction = "BULLISH"
        conf      = 45.0 + (composite - 55.0) * 0.8
    elif composite <= 32.0:
        regime    = "BEARISH_FLOW"
        direction = "BEARISH"
        conf      = min(90.0, 55.0 + (32.0 - composite) * 1.2)
    elif composite <= 45.0:
        regime    = "BEARISH_FLOW"
        direction = "BEARISH"
        conf      = 45.0 + (45.0 - composite) * 0.8
    else:
        regime    = "NEUTRAL"
        direction = "NEUTRAL"
        conf      = 40.0 + (10.0 - abs(composite - 50.0)) * 1.5

    return regime, direction, round(min(90.0, conf), 1)


# ── Main Entry ──────────────────────────────────────────────────────────────────
def compute_flow_engine(
    symbol:           str,
    price:            float,
    funding_result=None,     # FundingRateResult from crypto/funding_rate.py
    oi_result=None,          # OpenInterestResult from crypto/open_interest.py
    liq_result=None,         # LiquidationResult from crypto/liquidation_engine.py
) -> FlowEngineResult:
    """
    Compute unified derivatives flow score.

    Parameters
    ----------
    symbol         : ticker (e.g. "BTC")
    price          : current spot price
    funding_result : FundingRateResult (optional — will show neutral if None)
    oi_result      : OpenInterestResult (optional)
    liq_result     : LiquidationResult (optional)

    Returns
    -------
    FlowEngineResult — always returns, never raises
    """
    # ── Extract component values safely ──────────────────────────────────────
    funding_regime   = getattr(funding_result, "funding_regime",   "NEUTRAL")
    funding_rate_pct = getattr(funding_result, "funding_rate_pct", 0.0)
    oi_signal        = getattr(oi_result,      "price_oi_signal",  "UNKNOWN")
    oi_trend         = getattr(oi_result,      "oi_trend",         "STABLE")
    cascade_risk     = getattr(liq_result,     "cascade_risk",     "LOW")
    nearest_long_liq = getattr(liq_result,     "nearest_long_liq", None)
    nearest_short_liq= getattr(liq_result,     "nearest_short_liq",None)

    # ── Fetch Long/Short ratio ────────────────────────────────────────────────
    ls_result = _fetch_long_short_ratio(symbol)
    ls_ratio  = ls_result.ls_ratio if ls_result else 1.0

    # ── Component scores ─────────────────────────────────────────────────────
    WEIGHTS = {
        "funding": 0.35,
        "oi":      0.30,
        "ls":      0.25,
        "liq":     0.10,
    }

    f_score  = _score_funding(funding_regime, funding_rate_pct)
    oi_score = _score_oi(oi_signal, oi_trend)
    ls_score = _score_long_short(ls_result)
    liq_scr  = _score_liquidation(cascade_risk, nearest_long_liq, nearest_short_liq, price)

    composite = (
        f_score  * WEIGHTS["funding"]
        + oi_score * WEIGHTS["oi"]
        + ls_score * WEIGHTS["ls"]
        + liq_scr  * WEIGHTS["liq"]
    )
    composite = round(composite, 1)

    # ── Regime & direction ────────────────────────────────────────────────────
    flow_regime, flow_direction, flow_confidence = _classify_flow(
        composite, funding_regime, oi_signal, cascade_risk, ls_result
    )

    # ── Interpretation ────────────────────────────────────────────────────────
    regime_emoji = {
        "BULLISH_FLOW":  "🟢",
        "BEARISH_FLOW":  "🔴",
        "SHORT_SQUEEZE": "🚀",
        "LONG_SQUEEZE":  "💥",
        "NEUTRAL":       "⚪",
    }.get(flow_regime, "❓")

    interp = (
        f"{regime_emoji} {flow_regime} (score={composite:.0f}/100) — "
        f"Funding={funding_regime} | OI={oi_signal} | L/S={ls_ratio:.2f} | Cascade={cascade_risk}"
    )

    detail = {
        "funding": f"{funding_regime} ({funding_rate_pct:+.4f}%) → score={f_score:.0f}",
        "oi":      f"{oi_signal} trend={oi_trend} → score={oi_score:.0f}",
        "ls":      f"L/S={ls_ratio:.2f} → score={ls_score:.0f}",
        "liq":     f"Cascade={cascade_risk} → score={liq_scr:.0f}",
    }

    logger.info(
        "[flow_engine] %s regime=%s dir=%s score=%.1f conf=%.1f "
        "funding=%s oi=%s L/S=%.2f cascade=%s",
        symbol, flow_regime, flow_direction, composite, flow_confidence,
        funding_regime, oi_signal, ls_ratio, cascade_risk,
    )

    return FlowEngineResult(
        symbol           = symbol,
        flow_score       = composite,
        flow_direction   = flow_direction,
        flow_confidence  = flow_confidence,
        flow_regime      = flow_regime,
        funding_score    = f_score,
        oi_score         = oi_score,
        ls_score         = ls_score,
        liq_score        = liq_scr,
        funding_rate_pct = round(funding_rate_pct, 5),
        funding_regime   = funding_regime,
        oi_signal        = oi_signal,
        ls_ratio         = round(ls_ratio, 3),
        cascade_risk     = cascade_risk,
        interpretation   = interp,
        component_detail = detail,
    )
