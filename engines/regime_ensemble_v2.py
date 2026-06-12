"""
Regime Ensemble Engine  v2  (Phases 2 + 12 + 14 + 21 integration)
===================================================================
Extends the existing 4-component ensemble with 3 new macro-level inputs:

  Original (weights re-normalised):
    Markov HMM     (32%) — statistical, data-driven
    Trend Regime   (20%) — EMA-based directional
    Volatility     (16%) — vol-driven classification
    Macro          (12%) — RSI + momentum proxy

  NEW additions (Phase 12 / 14 / 21):
    Liquidity      (10%) — Global liquidity regime (DXY/VIX/Yields)
    Breadth        ( 6%) — Market breadth (crypto + equity)
    Cross-Asset    ( 4%) — BTC vs SPY/QQQ/DXY/GLD/TLT alignment

Total weight: 32+20+16+12+10+6+4 = 100%

Usage:
  from engines.regime_ensemble_v2 import compute_ensemble_regime_v2

  Replace step 3 in futures_orchestrator_v2.py:
    ensemble = compute_ensemble_regime_v2(df, raw_probs,
                                           liquidity_result, breadth_result, cross_asset_result)

Backward compatible: if new inputs are None, falls back to original 4-component weights.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from engines.regime_ensemble import (
    EnsembleRegimeResult,
    REGIMES,
    REGIME_TO_IDX,
    _trend_regime,
    _volatility_regime_probs,
    _macro_regime_probs,
    _normalise,
    _permission,
)

logger = logging.getLogger(__name__)

# ── Weights v2 ─────────────────────────────────────────────────────────────────
WEIGHTS_V2 = {
    "markov":      0.32,
    "trend":       0.20,
    "volatility":  0.16,
    "macro":       0.12,
    "liquidity":   0.10,   # NEW Phase 12
    "breadth":     0.06,   # NEW Phase 14
    "cross_asset": 0.04,   # NEW Phase 21
}

# Fallback when new signals are unavailable (collapses to original weights)
WEIGHTS_FALLBACK = {
    "markov":     0.40,
    "trend":      0.25,
    "volatility": 0.20,
    "macro":      0.15,
}


# ── Liquidity → Regime Probs ───────────────────────────────────────────────────
def _liquidity_to_probs(liquidity_result) -> dict[str, float]:
    """
    Convert LiquidityRegimeResult to regime probability vector.
    RISK_ON  → BULL/STRONG_BULL elevated
    RISK_OFF → BEAR/RANGE elevated
    CRISIS   → STRONG_BEAR elevated
    """
    if liquidity_result is None:
        return {r: 1/len(REGIMES) for r in REGIMES}

    regime = getattr(liquidity_result, "liquidity_regime", "RISK_ON")
    score  = getattr(liquidity_result, "score", 55.0)

    probs = {r: 0.02 for r in REGIMES}

    if regime == "RISK_ON":
        probs["STRONG_BULL"] = 0.35 + (score - 70) * 0.01 if score > 70 else 0.25
        probs["BULL"]        = 0.40
        probs["RANGE"]       = 0.15
    elif regime == "RECOVERY":
        probs["BULL"]        = 0.40
        probs["RANGE"]       = 0.30
        probs["STRONG_BULL"] = 0.15
    elif regime == "RISK_OFF":
        probs["BEAR"]        = 0.45
        probs["RANGE"]       = 0.30
        probs["STRONG_BEAR"] = 0.10
    elif regime == "CRISIS":
        probs["STRONG_BEAR"] = 0.60
        probs["BEAR"]        = 0.25
    else:
        probs["RANGE"]       = 0.40

    return _normalise(probs)


# ── Breadth → Regime Probs ─────────────────────────────────────────────────────
def _breadth_to_probs(breadth_result) -> dict[str, float]:
    """
    Convert MarketBreadthResult to regime probability vector.
    STRONG_BULL breadth → BULL/STRONG_BULL elevated.
    """
    if breadth_result is None:
        return {r: 1/len(REGIMES) for r in REGIMES}

    regime = getattr(breadth_result, "breadth_regime", "NEUTRAL")
    score  = getattr(breadth_result, "breadth_score", 50.0)

    probs = {r: 0.02 for r in REGIMES}

    if regime == "STRONG_BULL":
        probs["STRONG_BULL"] = 0.45; probs["BULL"] = 0.35
    elif regime == "BULL":
        probs["BULL"] = 0.50; probs["STRONG_BULL"] = 0.20; probs["RANGE"] = 0.15
    elif regime == "NEUTRAL":
        probs["RANGE"] = 0.45; probs["BULL"] = 0.25; probs["BEAR"] = 0.20
    elif regime == "BEAR":
        probs["BEAR"] = 0.50; probs["RANGE"] = 0.20; probs["STRONG_BEAR"] = 0.15
    elif regime == "STRONG_BEAR":
        probs["STRONG_BEAR"] = 0.45; probs["BEAR"] = 0.35

    return _normalise(probs)


# ── Cross-Asset → Regime Probs ─────────────────────────────────────────────────
def _cross_asset_to_probs(cross_asset_result) -> dict[str, float]:
    """
    Convert CrossAssetResult to regime probability vector.
    RISK_ON_ALIGNED → BULL/STRONG_BULL boosted.
    DECOUPLED_BULL  → regime less certain but bullish bias.
    """
    if cross_asset_result is None:
        return {r: 1/len(REGIMES) for r in REGIMES}

    regime = getattr(cross_asset_result, "cross_asset_regime", "TRANSITION")
    rs     = getattr(cross_asset_result, "relative_strength_score", 50.0)

    probs = {r: 0.02 for r in REGIMES}

    if regime == "RISK_ON_ALIGNED":
        probs["BULL"]        = 0.45; probs["STRONG_BULL"] = 0.30; probs["RANGE"] = 0.15
    elif regime == "DECOUPLED_BULL":
        probs["BULL"]        = 0.40; probs["STRONG_BULL"] = 0.25; probs["RANGE"] = 0.20
    elif regime == "TRANSITION":
        probs["RANGE"]       = 0.45; probs["BULL"] = 0.25; probs["BEAR"] = 0.20
    elif regime == "RISK_OFF_ALIGNED":
        probs["BEAR"]        = 0.45; probs["RANGE"] = 0.25; probs["STRONG_BEAR"] = 0.15
    elif regime == "DECOUPLED_BEAR":
        probs["BEAR"]        = 0.40; probs["STRONG_BEAR"] = 0.20; probs["RANGE"] = 0.25

    return _normalise(probs)


# ── Main Entry ─────────────────────────────────────────────────────────────────
def compute_ensemble_regime_v2(
    df:                pd.DataFrame,
    markov_probs:      dict[str, float],
    liquidity_result   = None,    # LiquidityRegimeResult or None
    breadth_result     = None,    # MarketBreadthResult or None
    cross_asset_result = None,    # CrossAssetResult or None
) -> EnsembleRegimeResult:
    """
    Extended 7-component ensemble regime engine.

    Parameters
    ----------
    df                 : OHLCV DataFrame
    markov_probs       : from MarkovRegimeEngine
    liquidity_result   : from compute_liquidity_regime() — optional
    breadth_result     : from compute_market_breadth()   — optional
    cross_asset_result : from compute_cross_asset()      — optional

    Returns
    -------
    EnsembleRegimeResult (identical contract to v1)
    """
    new_signals_available = any(x is not None for x in
                                [liquidity_result, breadth_result, cross_asset_result])

    # ── Component probabilities ───────────────────────────────────────────────
    trend_probs          = _trend_regime(df)
    vol_probs, vol_label = _volatility_regime_probs(df)
    macro_probs          = _macro_regime_probs(df)
    markov_norm          = _normalise({r: markov_probs.get(r, 0.02) for r in REGIMES})

    # New components
    liq_probs  = _liquidity_to_probs(liquidity_result)
    br_probs   = _breadth_to_probs(breadth_result)
    ca_probs   = _cross_asset_to_probs(cross_asset_result)

    # ── Select weights ────────────────────────────────────────────────────────
    if new_signals_available:
        W = WEIGHTS_V2.copy()
        # If some new signals are missing, redistribute their weights to markov
        missing_w = 0.0
        if liquidity_result is None:
            missing_w += W.pop("liquidity", 0)
        if breadth_result is None:
            missing_w += W.pop("breadth", 0)
        if cross_asset_result is None:
            missing_w += W.pop("cross_asset", 0)
        if missing_w > 0:
            # Distribute missing weight to markov (most reliable)
            W["markov"] = W.get("markov", 0.32) + missing_w

        # Re-normalise weights
        total_w = sum(W.values())
        W = {k: v / total_w for k, v in W.items()}
    else:
        W = WEIGHTS_FALLBACK

    # ── Weighted ensemble ─────────────────────────────────────────────────────
    ensemble: dict[str, float] = {r: 0.0 for r in REGIMES}
    component_map = {
        "markov":      markov_norm,
        "trend":       trend_probs,
        "volatility":  vol_probs,
        "macro":       macro_probs,
        "liquidity":   liq_probs,
        "breadth":     br_probs,
        "cross_asset": ca_probs,
    }

    for comp, w in W.items():
        probs = component_map[comp]
        for r in REGIMES:
            ensemble[r] += probs[r] * w

    ensemble = _normalise(ensemble)

    # ── Winning regime & confidence ───────────────────────────────────────────
    regime     = max(ensemble, key=ensemble.get)
    confidence = round(ensemble[regime] * 100, 1)

    from config.thresholds import THRESHOLDS
    confidence = min(confidence, THRESHOLDS.MAX_REGIME_CONFIDENCE)

    # Entropy-based clarity score
    entropy     = -sum(p * math.log(p + 1e-9) for p in ensemble.values())
    max_entropy = math.log(len(REGIMES))
    clarity     = round((1 - entropy / max_entropy) * 100, 1)

    perm, size_mult = _permission(regime, confidence)

    # ── Liquidity override: CRISIS liquidity overrides position sizing ────────
    if liquidity_result is not None:
        liq_regime  = getattr(liquidity_result, "liquidity_regime", "RISK_ON")
        risk_mult   = getattr(liquidity_result, "risk_multiplier",  1.0)
        if liq_regime == "CRISIS":
            size_mult = min(size_mult, 0.25)
            if perm not in ("NO_TRADE",):
                perm = "NO_TRADE"
                logger.warning("[ensemble_v2] CRISIS liquidity → position sizing set to 0.25")
        elif liq_regime == "RISK_OFF":
            size_mult = min(size_mult, risk_mult)

    logger.info(
        "[ensemble_v2] regime=%s conf=%.1f%% clarity=%.1f perm=%s size=%.2f "
        "new_signals=%s",
        regime, confidence, clarity, perm, size_mult, new_signals_available,
    )

    # ── Build component scores dict (extended) ────────────────────────────────
    component_scores = {
        "markov":     max(markov_norm, key=markov_norm.get),
        "trend":      max(trend_probs, key=trend_probs.get),
        "volatility": max(vol_probs,   key=vol_probs.get),
        "macro":      max(macro_probs, key=macro_probs.get),
    }
    if liquidity_result:
        component_scores["liquidity"]   = getattr(liquidity_result, "liquidity_regime", "N/A")
    if breadth_result:
        component_scores["breadth"]     = getattr(breadth_result,   "breadth_regime",   "N/A")
    if cross_asset_result:
        component_scores["cross_asset"] = getattr(cross_asset_result,"cross_asset_regime","N/A")

    return EnsembleRegimeResult(
        regime             = regime,
        confidence         = confidence,
        weighted_probs     = {r: round(v, 4) for r, v in ensemble.items()},
        component_scores   = component_scores,
        ensemble_score     = clarity,
        trade_permission   = perm,
        position_size_mult = size_mult,
    )
