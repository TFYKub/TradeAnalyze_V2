"""
Liquidity Regime Engine  (Phase 12)
=====================================
Global liquidity model using macro instruments to classify the current
liquidity environment and produce a risk multiplier for position sizing.

Instruments:
  DXY    (^DXY)   — Dollar strength: rising = RISK_OFF
  US10Y  (^TNX)   — Treasury yield: rapid rise = RISK_OFF, falling = RISK_ON
  VIX    (^VIX)   — Equity fear gauge: >35 = CRISIS, >25 = RISK_OFF
  TLT             — Bond proxy for MOVE-like volatility; TLT vol ≈ bond stress
  M2     (optional via FRED) — Broad money supply growth

Liquidity Score (0–100):
  0–25   → CRISIS      (extreme tightening / flight to safety)
  26–45  → RISK_OFF    (tightening / defensive)
  46–70  → NEUTRAL     (transitional — maps to RECOVERY or RISK_OFF depending on trend)
  71–100 → RISK_ON     (easing / expanding / risk appetite)

Regimes:
  RISK_ON    — liquidity expanding, risk appetite high
  RISK_OFF   — liquidity tightening, defensive positioning
  CRISIS     — systemic stress (VIX > 35, DXY surge, yields spiking)
  RECOVERY   — coming off CRISIS, liquidity improving

Risk Multiplier:
  RISK_ON    → 1.0–1.2  (allow larger positions)
  RISK_OFF   → 0.6–0.8  (reduce size)
  CRISIS     → 0.25     (minimal / no new positions)
  RECOVERY   → 0.75     (cautious re-entry)

Output: LiquidityRegimeResult
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
VIX_CRISIS     = 35.0
VIX_RISK_OFF   = 25.0
VIX_ELEVATED   = 20.0
DXY_SURGE_PCT  = 2.5    # 10-day DXY rise % that signals stress
YIELD_SURGE_PCT= 15.0   # 10-day yield rise % (e.g. 3.5% → 4.0% = +14%)
LOOKBACK_DAYS  = 20     # days for trend calculation
FETCH_PERIOD   = "6mo"


@dataclass(frozen=True)
class LiquidityRegimeResult:
    liquidity_regime:    str            # RISK_ON | RISK_OFF | CRISIS | RECOVERY
    confidence:          float          # 0–100
    score:               float          # raw liquidity score 0–100
    risk_multiplier:     float          # applied to position sizing
    vix_level:           float
    vix_regime:          str            # CALM | ELEVATED | STRESS | CRISIS
    dxy_trend:           str            # STRENGTHENING | WEAKENING | STABLE
    yield_trend:         str            # RISING | FALLING | STABLE
    tlt_vol_pct:         float          # TLT 20-day HV as proxy for bond stress
    component_scores:    dict[str, float]
    interpretation:      str
    data_source:         str            # "live" | "cached" | "fallback"


# ── Data Fetchers ──────────────────────────────────────────────────────────────
def _fetch_instrument(ticker: str, period: str = FETCH_PERIOD) -> Optional[pd.DataFrame]:
    """Fetch OHLCV from yfinance with safe fallback."""
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df is None or df.empty or len(df) < 20:
            return None
        return df
    except Exception as exc:
        logger.warning("[liquidity] fetch %s failed: %s", ticker, exc)
        return None


# ── Component Scorers ──────────────────────────────────────────────────────────
def _score_vix(df: Optional[pd.DataFrame]) -> tuple[float, float, str]:
    """
    Returns (score_0_100, vix_level, vix_regime).
    High VIX → low score (RISK_OFF).
    """
    if df is None or df.empty:
        return 50.0, 20.0, "UNKNOWN"

    vix = float(df["Close"].iloc[-1])

    if vix >= VIX_CRISIS:
        score  = max(0.0, 10.0 - (vix - VIX_CRISIS) * 0.5)
        regime = "CRISIS"
    elif vix >= VIX_RISK_OFF:
        score  = 10.0 + (VIX_CRISIS - vix) / (VIX_CRISIS - VIX_RISK_OFF) * 30.0
        regime = "STRESS"
    elif vix >= VIX_ELEVATED:
        score  = 40.0 + (VIX_RISK_OFF - vix) / (VIX_RISK_OFF - VIX_ELEVATED) * 25.0
        regime = "ELEVATED"
    else:
        score  = 65.0 + max(0.0, (VIX_ELEVATED - vix) / VIX_ELEVATED * 35.0)
        regime = "CALM"

    return round(min(100.0, max(0.0, score)), 1), round(vix, 2), regime


def _score_dxy(df: Optional[pd.DataFrame]) -> tuple[float, str]:
    """
    Returns (score_0_100, trend).
    Rising DXY → dollar strengthens → RISK_OFF (lower score).
    """
    if df is None or df.empty:
        return 50.0, "UNKNOWN"

    close = df["Close"]
    if len(close) < LOOKBACK_DAYS:
        return 50.0, "INSUFFICIENT_DATA"

    current   = float(close.iloc[-1])
    prior     = float(close.iloc[-LOOKBACK_DAYS])
    change_pct = (current - prior) / prior * 100

    # Momentum: 5-day vs 20-day EMA spread
    ema5  = float(close.ewm(span=5,  adjust=False).mean().iloc[-1])
    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    spread_pct = (ema5 - ema20) / ema20 * 100

    if change_pct >= DXY_SURGE_PCT:
        score = 15.0
        trend = "STRENGTHENING"
    elif change_pct >= 1.0:
        score = 35.0
        trend = "STRENGTHENING"
    elif change_pct <= -DXY_SURGE_PCT:
        score = 85.0
        trend = "WEAKENING"
    elif change_pct <= -1.0:
        score = 65.0
        trend = "WEAKENING"
    else:
        score = 50.0
        trend = "STABLE"

    # Adjust by current momentum
    score += max(-15.0, min(15.0, -spread_pct * 3))
    return round(min(100.0, max(0.0, score)), 1), trend


def _score_yield(df: Optional[pd.DataFrame]) -> tuple[float, str]:
    """
    Returns (score_0_100, yield_trend).
    Rapidly rising yields → RISK_OFF.
    Falling yields during stress → may signal RECOVERY.
    """
    if df is None or df.empty:
        return 50.0, "UNKNOWN"

    close = df["Close"]
    if len(close) < LOOKBACK_DAYS:
        return 50.0, "INSUFFICIENT_DATA"

    current = float(close.iloc[-1])
    prior   = float(close.iloc[-LOOKBACK_DAYS])

    if prior <= 0.1:
        return 50.0, "INSUFFICIENT_DATA"

    change_pct = (current - prior) / prior * 100

    if change_pct >= YIELD_SURGE_PCT:
        score = 20.0
        trend = "RISING"
    elif change_pct >= 5.0:
        score = 38.0
        trend = "RISING"
    elif change_pct <= -YIELD_SURGE_PCT:
        # Falling yields in crisis = flight to safety (RISK_OFF) or accommodative (RISK_ON)
        # We score moderate since it's ambiguous
        score = 60.0
        trend = "FALLING"
    elif change_pct <= -5.0:
        score = 62.0
        trend = "FALLING"
    else:
        score = 50.0
        trend = "STABLE"

    return round(min(100.0, max(0.0, score)), 1), trend


def _score_tlt_vol(df: Optional[pd.DataFrame]) -> float:
    """
    TLT 20-day historical volatility as MOVE proxy.
    Returns score 0–100 (high vol = low score = RISK_OFF).
    """
    if df is None or df.empty:
        return 50.0

    log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    if len(log_ret) < 20:
        return 50.0

    hv20 = float(log_ret.rolling(20).std().iloc[-1] * math.sqrt(252) * 100)

    # TLT HV: < 10% = calm bonds, > 25% = bond stress
    if hv20 >= 25.0:
        score = 15.0
    elif hv20 >= 18.0:
        score = 35.0
    elif hv20 >= 12.0:
        score = 55.0
    else:
        score = 78.0

    return round(score, 1), round(hv20, 2)


# ── Regime Classifier ──────────────────────────────────────────────────────────
def _classify_regime(
    score: float,
    vix_regime: str,
    prev_regime: Optional[str] = None,
) -> tuple[str, float, float]:
    """
    Returns (regime, confidence, risk_multiplier).
    Uses score + VIX override for CRISIS detection.
    """
    # Hard CRISIS override
    if vix_regime == "CRISIS":
        return "CRISIS", 92.0, 0.25

    if score >= 72.0:
        regime      = "RISK_ON"
        confidence  = min(95.0, 60.0 + (score - 72.0) * 1.5)
        risk_mult   = round(min(1.25, 0.9 + (score - 72.0) * 0.015), 2)
    elif score >= 55.0:
        # RISK_ON borderline — check if recovering from stress
        if prev_regime in ("CRISIS", "RISK_OFF"):
            regime     = "RECOVERY"
            confidence = 60.0 + (score - 55.0) * 0.8
            risk_mult  = 0.75
        else:
            regime     = "RISK_ON"
            confidence = 55.0 + (score - 55.0) * 0.5
            risk_mult  = 0.90
    elif score >= 35.0:
        regime     = "RISK_OFF"
        confidence = 55.0 + (45.0 - score) * 0.8
        risk_mult  = round(max(0.55, 0.85 - (55.0 - score) * 0.015), 2)
    else:
        regime     = "CRISIS"
        confidence = min(95.0, 70.0 + (35.0 - score) * 1.0)
        risk_mult  = 0.25

    return regime, round(min(95.0, confidence), 1), round(risk_mult, 2)


# ── Main Entry ─────────────────────────────────────────────────────────────────
def compute_liquidity_regime(
    prev_regime: Optional[str] = None,
) -> LiquidityRegimeResult:
    """
    Compute global liquidity regime from macro instruments.

    Parameters
    ----------
    prev_regime : last known regime for RECOVERY detection

    Returns
    -------
    LiquidityRegimeResult — always returns a value, never raises
    """
    source = "live"

    # ── Fetch instruments ─────────────────────────────────────────────────────
    df_vix   = _fetch_instrument("^VIX")
    df_dxy   = _fetch_instrument("DX-Y.NYB")  # DXY
    df_tnt   = _fetch_instrument("^TNX")      # US 10-Year yield
    df_tlt   = _fetch_instrument("TLT")       # iShares 20Y Bond (MOVE proxy)

    # Check how many feeds returned
    feeds_ok = sum(df is not None for df in [df_vix, df_dxy, df_tnt, df_tlt])
    if feeds_ok == 0:
        logger.warning("[liquidity] All feeds failed — using full fallback")
        source = "fallback"
        return LiquidityRegimeResult(
            liquidity_regime="RISK_ON", confidence=40.0, score=55.0,
            risk_multiplier=0.85, vix_level=20.0, vix_regime="UNKNOWN",
            dxy_trend="UNKNOWN", yield_trend="UNKNOWN", tlt_vol_pct=12.0,
            component_scores={"vix": 50, "dxy": 50, "yield": 50, "tlt_vol": 50},
            interpretation="Liquidity data unavailable — neutral fallback applied",
            data_source="fallback",
        )

    if feeds_ok < 3:
        source = "cached"

    # ── Component scores ─────────────────────────────────────────────────────
    vix_score, vix_level, vix_regime = _score_vix(df_vix)
    dxy_score, dxy_trend             = _score_dxy(df_dxy)
    yield_score, yield_trend         = _score_yield(df_tnt)
    tlt_result                       = _score_tlt_vol(df_tlt)

    if isinstance(tlt_result, tuple):
        tlt_score, tlt_hv = tlt_result
    else:
        tlt_score, tlt_hv = tlt_result, 12.0

    # ── Weighted composite score ──────────────────────────────────────────────
    # VIX is the primary fear gauge → highest weight
    WEIGHTS = {"vix": 0.40, "dxy": 0.25, "yield": 0.20, "tlt_vol": 0.15}
    composite = (
        vix_score   * WEIGHTS["vix"]
        + dxy_score   * WEIGHTS["dxy"]
        + yield_score * WEIGHTS["yield"]
        + tlt_score   * WEIGHTS["tlt_vol"]
    )
    composite = round(composite, 1)

    # ── Regime classification ─────────────────────────────────────────────────
    regime, confidence, risk_mult = _classify_regime(composite, vix_regime, prev_regime)

    # ── Interpretation string ─────────────────────────────────────────────────
    regime_emoji = {
        "RISK_ON":   "✅",
        "RISK_OFF":  "⚠️",
        "CRISIS":    "🚨",
        "RECOVERY":  "🔄",
    }.get(regime, "❓")

    interp = (
        f"{regime_emoji} {regime} (score={composite:.0f}/100 | VIX={vix_level:.1f} "
        f"| DXY={dxy_trend} | Yields={yield_trend} | BondVol={tlt_hv:.1f}%)"
    )

    component_scores = {
        "vix":     vix_score,
        "dxy":     dxy_score,
        "yield":   yield_score,
        "tlt_vol": tlt_score,
    }

    logger.info(
        "[liquidity] regime=%s conf=%.1f score=%.1f mult=%.2f "
        "VIX=%.1f DXY=%s Yield=%s source=%s",
        regime, confidence, composite, risk_mult,
        vix_level, dxy_trend, yield_trend, source,
    )

    return LiquidityRegimeResult(
        liquidity_regime  = regime,
        confidence        = confidence,
        score             = composite,
        risk_multiplier   = risk_mult,
        vix_level         = vix_level,
        vix_regime        = vix_regime,
        dxy_trend         = dxy_trend,
        yield_trend       = yield_trend,
        tlt_vol_pct       = tlt_hv,
        component_scores  = component_scores,
        interpretation    = interp,
        data_source       = source,
    )
