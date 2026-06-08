"""
Regime Ensemble Engine  (Phase 2)
===================================
Combines 4 regime signals into a weighted ensemble:

  1. Markov HMM (40%)        — statistical, data-driven
  2. Trend Regime (25%)      — EMA-based directional
  3. Volatility Regime (20%) — vol-driven classification
  4. Macro Regime (15%)      — RSI + momentum macro proxy

Output: EnsembleRegimeResult
  regime           : STRONG_BULL | BULL | RANGE | BEAR | STRONG_BEAR
  confidence       : 0–100
  component_scores : breakdown per engine
  weighted_probs   : {regime: float}  (sum to 1.0)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REGIMES = ["STRONG_BULL", "BULL", "RANGE", "BEAR", "STRONG_BEAR"]
REGIME_TO_IDX = {r: i for i, r in enumerate(REGIMES)}

# Ensemble weights (sum = 1.0)
WEIGHTS = {
    "markov":     0.40,
    "trend":      0.25,
    "volatility": 0.20,
    "macro":      0.15,
}


@dataclass(frozen=True)
class EnsembleRegimeResult:
    regime:           str           # winning regime
    confidence:       float         # 0–100
    weighted_probs:   dict[str, float]  # all 5 regimes, sum=1
    component_scores: dict[str, str]    # {engine: regime_assigned}
    ensemble_score:   float         # 0–100 composite clarity
    trade_permission: str           # LONG_ONLY | SHORT_ONLY | BOTH | NO_TRADE
    position_size_mult: float


# ──────────────────────────────────────────────────────────────────────────────
# COMPONENT CLASSIFIERS
# ──────────────────────────────────────────────────────────────────────────────
def _trend_regime(df: pd.DataFrame) -> dict[str, float]:
    """EMA-based trend regime probabilities."""
    close  = df["Close"]
    ema20  = close.ewm(span=20, adjust=False).mean()
    ema50  = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    last  = float(close.iloc[-1])
    e20   = float(ema20.iloc[-1])
    e50   = float(ema50.iloc[-1])
    e200  = float(ema200.iloc[-1])

    # Spread of EMA20 from EMA50 (trend strength)
    spread = (e20 - e50) / e50 * 100 if e50 > 0 else 0

    probs = {r: 0.02 for r in REGIMES}  # base 2%

    if last > e20 > e50 > e200:
        probs["STRONG_BULL"] = 0.70 + min(0.20, abs(spread) * 0.02)
        probs["BULL"]        = 0.20
    elif last > e20 > e50:
        probs["BULL"]        = 0.65
        probs["STRONG_BULL"] = 0.20
    elif last < e20 < e50 < e200:
        probs["STRONG_BEAR"] = 0.70 + min(0.20, abs(spread) * 0.02)
        probs["BEAR"]        = 0.20
    elif last < e20 < e50:
        probs["BEAR"]        = 0.65
        probs["STRONG_BEAR"] = 0.20
    else:
        probs["RANGE"] = 0.65

    return _normalise(probs)


def _volatility_regime_probs(df: pd.DataFrame) -> tuple[dict[str, float], str]:
    """Vol-based regime classification."""
    log_ret  = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    hv20     = float(log_ret.rolling(20).std().iloc[-1] * math.sqrt(252))

    # ATR pct of price
    hl   = df["High"] - df["Low"]
    hc   = (df["High"] - df["Close"].shift()).abs()
    lc   = (df["Low"]  - df["Close"].shift()).abs()
    tr   = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr  = float(tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1])
    atr_pct = atr / float(df["Close"].iloc[-1]) * 100

    # Low vol → trend-following / Bull more likely
    # High vol → range / bear more likely
    probs = {r: 0.02 for r in REGIMES}

    if atr_pct < 1.0:   # LOW_VOL
        probs["BULL"]  = 0.40; probs["STRONG_BULL"] = 0.25; probs["RANGE"] = 0.25
        vol_label = "LOW"
    elif atr_pct < 2.5:  # NORMAL
        probs["BULL"]  = 0.30; probs["RANGE"] = 0.35; probs["BEAR"] = 0.20
        vol_label = "NORMAL"
    elif atr_pct < 4.0:  # HIGH
        probs["RANGE"] = 0.30; probs["BEAR"] = 0.35; probs["STRONG_BEAR"] = 0.20
        vol_label = "HIGH"
    else:               # PANIC
        probs["STRONG_BEAR"] = 0.50; probs["BEAR"] = 0.30
        vol_label = "PANIC"

    return _normalise(probs), vol_label


def _macro_regime_probs(df: pd.DataFrame) -> dict[str, float]:
    """RSI + momentum macro proxy regime."""
    log_ret  = np.log(df["Close"] / df["Close"].shift(1)).dropna()

    # RSI approximation
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = float((100 - 100 / (1 + rs)).iloc[-1])

    # 20-day momentum
    mom20 = float((df["Close"].iloc[-1] / df["Close"].iloc[-21] - 1) * 100
                  if len(df) > 21 else 0)

    probs = {r: 0.02 for r in REGIMES}

    if rsi > 65 and mom20 > 5:
        probs["STRONG_BULL"] = 0.55; probs["BULL"] = 0.30
    elif rsi > 55 and mom20 > 0:
        probs["BULL"] = 0.55; probs["RANGE"] = 0.25
    elif 40 <= rsi <= 55 and abs(mom20) < 3:
        probs["RANGE"] = 0.60; probs["BULL"] = 0.20
    elif rsi < 45 and mom20 < 0:
        probs["BEAR"] = 0.55; probs["RANGE"] = 0.25
    elif rsi < 30 and mom20 < -5:
        probs["STRONG_BEAR"] = 0.55; probs["BEAR"] = 0.30
    else:
        probs["RANGE"] = 0.50

    return _normalise(probs)


def _normalise(probs: dict[str, float]) -> dict[str, float]:
    """Normalise to sum=1 with softmax smoothing."""
    total = sum(probs.values())
    if total <= 0:
        return {r: 1/len(REGIMES) for r in REGIMES}
    return {r: probs[r] / total for r in REGIMES}


# ──────────────────────────────────────────────────────────────────────────────
# ENSEMBLE
# ──────────────────────────────────────────────────────────────────────────────
def _permission(regime: str, conf: float) -> tuple[str, float]:
    if conf < 60:
        return "NO_TRADE", 0.5
    m = {
        "STRONG_BULL": ("LONG_ONLY",  1.00),
        "BULL":        ("LONG_ONLY",  0.75),
        "RANGE":       ("BOTH",       0.50),
        "BEAR":        ("SHORT_ONLY", 0.75),
        "STRONG_BEAR": ("SHORT_ONLY", 1.00),
    }
    return m.get(regime, ("NO_TRADE", 0.5))


def compute_ensemble_regime(
    df:              pd.DataFrame,
    markov_probs:    dict[str, float],   # from MarkovRegimeEngine
) -> EnsembleRegimeResult:
    """
    Combine Markov HMM + trend + volatility + macro into ensemble regime.

    Parameters
    ----------
    df           : daily OHLCV DataFrame
    markov_probs : regime_probs_all from MarkovRegimeEngine.detect()
    """

    # Component regimes
    trend_probs              = _trend_regime(df)
    vol_probs, vol_label     = _volatility_regime_probs(df)
    macro_probs              = _macro_regime_probs(df)

    # Ensure all regimes present in markov_probs
    markov_norm = _normalise({r: markov_probs.get(r, 0.02) for r in REGIMES})

    # Weighted ensemble
    ensemble: dict[str, float] = {r: 0.0 for r in REGIMES}
    for r in REGIMES:
        ensemble[r] = (
            markov_norm[r] * WEIGHTS["markov"]
            + trend_probs[r] * WEIGHTS["trend"]
            + vol_probs[r]   * WEIGHTS["volatility"]
            + macro_probs[r] * WEIGHTS["macro"]
        )

    ensemble = _normalise(ensemble)

    # Winning regime
    regime     = max(ensemble, key=ensemble.get)
    confidence = round(ensemble[regime] * 100, 1)

    # Confidence cap at 95% (Phase 1, Fix 4)
    from config.thresholds import THRESHOLDS
    confidence = min(confidence, THRESHOLDS.MAX_REGIME_CONFIDENCE)

    # Ensemble clarity: entropy-based score
    entropy      = -sum(p * math.log(p + 1e-9) for p in ensemble.values())
    max_entropy  = math.log(len(REGIMES))
    clarity      = round((1 - entropy / max_entropy) * 100, 1)

    perm, size_mult = _permission(regime, confidence)

    logger.info(
        "[ensemble] regime=%s conf=%.1f%% clarity=%.1f  "
        "markov=%s trend=%s vol=%s macro=%s",
        regime, confidence, clarity,
        max(markov_norm, key=markov_norm.get),
        max(trend_probs, key=trend_probs.get),
        max(vol_probs, key=vol_probs.get),
        max(macro_probs, key=macro_probs.get),
    )

    return EnsembleRegimeResult(
        regime          = regime,
        confidence      = confidence,
        weighted_probs  = {r: round(v, 4) for r, v in ensemble.items()},
        component_scores= {
            "markov":     max(markov_norm, key=markov_norm.get),
            "trend":      max(trend_probs, key=trend_probs.get),
            "volatility": max(vol_probs,   key=vol_probs.get),
            "macro":      max(macro_probs, key=macro_probs.get),
        },
        ensemble_score  = clarity,
        trade_permission= perm,
        position_size_mult = size_mult,
    )
