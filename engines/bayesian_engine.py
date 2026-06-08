"""
Bayesian Probability Engine  (Phase 9)
========================================
Converts indicator signals into posterior probabilities.

Instead of: "RSI Oversold"
Output:     "P(Bounce | RSI < 30) = 67%"

Uses Bayesian updating:
  P(A|B) = P(B|A) × P(A) / P(B)

Priors based on empirical backtesting averages across equity markets.
These are approximations — calibrate with own historical data.

Supported signals:
  • RSI extremes       — oversold / overbought
  • Regime transitions — probability of reversal
  • Trend alignment    — confluence strengthens probability
  • Volatility events  — vol expansion / contraction
  • Structure breaks   — BOS probability leading to trend continuation
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BayesianSignal:
    signal_name:   str
    condition:     str
    prior:         float    # P(outcome)
    likelihood:    float    # P(signal | outcome)
    posterior:     float    # P(outcome | signal) — computed
    confidence:    str      # HIGH | MEDIUM | LOW
    description:   str


@dataclass(frozen=True)
class BayesianResult:
    signals:          tuple[BayesianSignal, ...]
    composite_bull_prob: float   # weighted average bullish probability
    composite_bear_prob: float
    net_edge:         float      # bull_prob - bear_prob
    interpretation:   str


# ── Empirical priors (approximate) ───────────────────────────────────────────
# P(upward move in next 10 days) across equity markets ≈ 0.55 (slight bull bias)
BASE_BULL_PRIOR = 0.55
BASE_BEAR_PRIOR = 0.45


def _posterior(prior: float, likelihood_given_signal: float) -> float:
    """P(outcome | signal) via Bayes — simplified flat P(signal) = 0.5."""
    p_signal = 0.5   # assume signal fires ~50% of the time
    return min(0.99, max(0.01, prior * likelihood_given_signal / p_signal))


def _confidence(posterior: float) -> str:
    if posterior >= 0.70:  return "HIGH"
    if posterior >= 0.55:  return "MEDIUM"
    return "LOW"


def compute_rsi_bayesian(rsi: float) -> BayesianSignal:
    """P(Bounce | RSI level)"""
    if rsi < 20:
        prior      = BASE_BULL_PRIOR
        likelihood = 0.82   # very strong oversold → high bounce probability
        cond       = f"RSI < 20 (extreme oversold)"
    elif rsi < 30:
        prior      = BASE_BULL_PRIOR
        likelihood = 0.72
        cond       = f"RSI < 30 (oversold)"
    elif rsi < 40:
        prior      = BASE_BULL_PRIOR
        likelihood = 0.60
        cond       = f"RSI < 40 (weak)"
    elif rsi > 80:
        prior      = BASE_BEAR_PRIOR
        likelihood = 0.80
        cond       = f"RSI > 80 (extreme overbought)"
    elif rsi > 70:
        prior      = BASE_BEAR_PRIOR
        likelihood = 0.68
        cond       = f"RSI > 70 (overbought)"
    elif rsi > 60:
        prior      = BASE_BEAR_PRIOR
        likelihood = 0.55
        cond       = f"RSI > 60 (elevated)"
    else:
        prior      = 0.50
        likelihood = 0.50
        cond       = f"RSI {rsi:.0f} (neutral)"

    post = _posterior(prior, likelihood)
    direction = "Bullish" if rsi < 50 else "Bearish"

    return BayesianSignal(
        signal_name = "RSI",
        condition   = cond,
        prior       = round(prior, 3),
        likelihood  = round(likelihood, 3),
        posterior   = round(post, 3),
        confidence  = _confidence(post),
        description = f"P({direction} | {cond}) = {post*100:.0f}%",
    )


def compute_regime_bayesian(regime: str, confidence: float) -> BayesianSignal:
    """P(Trend continues | Regime detected)"""
    cont_prob = {
        "STRONG_BULL": 0.78,
        "BULL":        0.68,
        "RANGE":       0.55,
        "BEAR":        0.68,
        "STRONG_BEAR": 0.78,
        "CORRECTION":  0.40,
    }.get(regime, 0.55)

    prior      = BASE_BULL_PRIOR if "BULL" in regime else BASE_BEAR_PRIOR
    likelihood = cont_prob * (confidence / 100)
    post       = _posterior(prior, likelihood)
    direction  = "Bullish" if "BULL" in regime else "Bearish" if "BEAR" in regime else "Neutral"

    return BayesianSignal(
        signal_name = "Regime",
        condition   = f"{regime} (conf={confidence:.0f}%)",
        prior       = round(prior, 3),
        likelihood  = round(likelihood, 3),
        posterior   = round(post, 3),
        confidence  = _confidence(post),
        description = f"P({direction} trend continues | {regime}) = {post*100:.0f}%",
    )


def compute_trend_bayesian(ema_alignment: float, structure_trend: str) -> BayesianSignal:
    """P(Continuation | EMA + Structure alignment)"""
    align_score  = ema_alignment / 100
    struct_bonus = 0.15 if structure_trend in ("BULLISH", "BEARISH") else 0.0

    prior      = 0.55
    likelihood = min(0.90, align_score * 0.80 + struct_bonus)
    post       = _posterior(prior, likelihood)
    direction  = "Bullish" if structure_trend == "BULLISH" else "Bearish" if structure_trend == "BEARISH" else "Neutral"

    return BayesianSignal(
        signal_name = "Trend",
        condition   = f"EMA strength={ema_alignment:.0f} + Structure={structure_trend}",
        prior       = round(prior, 3),
        likelihood  = round(likelihood, 3),
        posterior   = round(post, 3),
        confidence  = _confidence(post),
        description = f"P({direction} continuation | EMA+Structure) = {post*100:.0f}%",
    )


def compute_volatility_bayesian(vol_regime: str, atr_pct: float) -> BayesianSignal:
    """P(Mean reversion | Vol regime) — high vol → expect mean reversion"""
    mr_prob = {
        "LOW_VOL":    0.35,   # low vol → trend continuation more likely
        "NORMAL_VOL": 0.50,
        "HIGH_VOL":   0.65,   # high vol → mean reversion more likely
        "PANIC_VOL":  0.75,   # panic → extreme mean reversion
    }.get(vol_regime, 0.50)

    prior      = 0.50
    likelihood = mr_prob
    post       = _posterior(prior, likelihood)

    return BayesianSignal(
        signal_name = "Volatility",
        condition   = f"{vol_regime} (ATR={atr_pct:.1f}%)",
        prior       = round(prior, 3),
        likelihood  = round(likelihood, 3),
        posterior   = round(post, 3),
        confidence  = _confidence(post),
        description = f"P(Mean Reversion | {vol_regime}) = {post*100:.0f}%",
    )


def compute_bayesian_analysis(
    rsi:             float,
    regime:          str,
    regime_confidence: float,
    ema_alignment:   float,
    structure_trend: str,
    vol_regime:      str,
    atr_pct:         float,
) -> BayesianResult:
    """
    Run all Bayesian signal computations and compute composite probabilities.
    """

    rsi_sig    = compute_rsi_bayesian(rsi)
    regime_sig = compute_regime_bayesian(regime, regime_confidence)
    trend_sig  = compute_trend_bayesian(ema_alignment, structure_trend)
    vol_sig    = compute_volatility_bayesian(vol_regime, atr_pct)

    signals = (rsi_sig, regime_sig, trend_sig, vol_sig)

    # Weighted composite
    w = [0.25, 0.35, 0.25, 0.15]
    is_bull = {
        "STRONG_BULL": True, "BULL": True,
        "RANGE": None, "BEAR": False, "STRONG_BEAR": False,
        "CORRECTION": False,
    }.get(regime)

    bull_prob = 0.0
    bear_prob = 0.0

    for sig, wi in zip(signals, w):
        p = sig.posterior
        if "Bull" in sig.description or is_bull is True:
            bull_prob += p * wi
        elif "Bear" in sig.description or is_bull is False:
            bear_prob += p * wi
        else:
            bull_prob += 0.5 * wi
            bear_prob += 0.5 * wi

    bull_prob = round(min(0.99, max(0.01, bull_prob)), 3)
    bear_prob = round(min(0.99, max(0.01, bear_prob)), 3)
    net_edge  = round(bull_prob - bear_prob, 3)

    if net_edge > 0.15:
        interp = f"Strong bullish edge: P(Bull)={bull_prob*100:.0f}% vs P(Bear)={bear_prob*100:.0f}%"
    elif net_edge < -0.15:
        interp = f"Strong bearish edge: P(Bear)={bear_prob*100:.0f}% vs P(Bull)={bull_prob*100:.0f}%"
    elif abs(net_edge) < 0.05:
        interp = f"No clear edge: P(Bull)≈P(Bear) ≈ {bull_prob*100:.0f}%"
    else:
        interp = f"Moderate edge: net={net_edge*100:+.0f}%"

    return BayesianResult(
        signals              = signals,
        composite_bull_prob  = bull_prob,
        composite_bear_prob  = bear_prob,
        net_edge             = net_edge,
        interpretation       = interp,
    )
