"""
Bayesian Reliability Weighting  (Phase 16)
============================================
Extends the Bayesian Engine to prevent single-indicator dominance by
applying regime-conditional reliability weights before posterior calculation.

Problem Solved:
  RSI oversold in a STRONG_BEAR regime should have LOW reliability —
  the market can stay oversold for days. But RSI oversold in a BULL regime
  (during a healthy pullback) has HIGH reliability.

  Without weighting: RSI oversold always produces ~90% posterior → overrides regime signal.
  With weighting: posterior = likelihood × reliability_weight — regime context now gates it.

Reliability Tables (empirically calibrated, updatable):
  Each indicator has a per-regime reliability coefficient (0.0–1.0).
  The coefficient scales the likelihood before Bayesian update:
    posterior ∝ prior × (likelihood × reliability)

New Function:
  compute_bayesian_analysis_v2(...)
    — drop-in replacement for compute_bayesian_analysis()
    — adds `regime` parameter to apply reliability filtering
    — preserves all existing BayesianResult / BayesianSignal contracts

Integration:
  futures_orchestrator.py step 14 → replace compute_bayesian_analysis with v2
  All downstream consumers (report, scoring) unchanged — same output types.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from engines.bayesian_engine import (
    BayesianResult,
    BayesianSignal,
    _posterior,
    _confidence,
    BASE_BULL_PRIOR,
    BASE_BEAR_PRIOR,
    compute_rsi_bayesian,
    compute_regime_bayesian,
    compute_trend_bayesian,
    compute_volatility_bayesian,
)

logger = logging.getLogger(__name__)


# ── Reliability Tables ─────────────────────────────────────────────────────────
# Format: {indicator: {regime: reliability_coefficient}}
# 1.0 = full reliability (use likelihood as-is)
# 0.1 = near-ignore (indicator is noise in this regime)

RSI_RELIABILITY: dict[str, float] = {
    "STRONG_BULL": 1.00,   # overbought valid; oversold = rare pullback signal
    "BULL":        0.85,
    "RANGE":       0.70,   # RSI works in ranges
    "BEAR":        0.40,   # oversold can stay oversold for weeks
    "STRONG_BEAR": 0.15,   # RSI oversold = meaningless in strong bear
}

TREND_RELIABILITY: dict[str, float] = {
    "STRONG_BULL": 1.00,
    "BULL":        0.90,
    "RANGE":       0.50,   # EMA alignment noisy in ranges
    "BEAR":        0.85,
    "STRONG_BEAR": 1.00,
}

VOLATILITY_RELIABILITY: dict[str, float] = {
    "STRONG_BULL": 0.70,
    "BULL":        0.75,
    "RANGE":       0.90,   # vol signals most useful in range/chop
    "BEAR":        0.85,
    "STRONG_BEAR": 0.60,   # panic vol regime = mean reversion less reliable
}

REGIME_RELIABILITY: dict[str, float] = {
    "STRONG_BULL": 1.00,   # regime signal always highly reliable (it IS the regime)
    "BULL":        1.00,
    "RANGE":       0.80,   # regime unclear in range → slight discount
    "BEAR":        1.00,
    "STRONG_BEAR": 1.00,
}


# ── Weighted Posterior ────────────────────────────────────────────────────────
def _weighted_posterior(prior: float, likelihood: float, reliability: float) -> float:
    """
    Apply reliability weighting before Bayesian update.
    Effective likelihood = likelihood × reliability
    posterior ∝ prior × effective_likelihood / p_signal
    """
    effective_likelihood = likelihood * reliability
    return min(0.99, max(0.01, _posterior(prior, effective_likelihood)))


# ── Reliability-Weighted Signal Computers ─────────────────────────────────────
def compute_rsi_bayesian_weighted(rsi: float, regime: str) -> BayesianSignal:
    """RSI Bayesian signal with regime-conditional reliability weighting."""
    reliability = RSI_RELIABILITY.get(regime, 0.70)
    base_signal = compute_rsi_bayesian(rsi)

    # Re-compute posterior with reliability adjustment
    weighted_post = _weighted_posterior(
        base_signal.prior,
        base_signal.likelihood,
        reliability,
    )

    return BayesianSignal(
        signal_name  = base_signal.signal_name,
        condition    = base_signal.condition,
        prior        = base_signal.prior,
        likelihood   = round(base_signal.likelihood * reliability, 3),
        posterior    = round(weighted_post, 3),
        confidence   = _confidence(weighted_post),
        description  = (
            f"{base_signal.description} [reliability={reliability:.0%} in {regime}]"
        ),
    )


def compute_regime_bayesian_weighted(
    regime: str, confidence: float
) -> BayesianSignal:
    """Regime continuation signal — always high reliability (regime IS the prior)."""
    reliability = REGIME_RELIABILITY.get(regime, 0.90)
    from engines.bayesian_engine import compute_regime_bayesian
    base_signal = compute_regime_bayesian(regime, confidence)

    weighted_post = _weighted_posterior(
        base_signal.prior,
        base_signal.likelihood,
        reliability,
    )
    return BayesianSignal(
        signal_name  = base_signal.signal_name,
        condition    = base_signal.condition,
        prior        = base_signal.prior,
        likelihood   = round(base_signal.likelihood * reliability, 3),
        posterior    = round(weighted_post, 3),
        confidence   = _confidence(weighted_post),
        description  = base_signal.description,
    )


def compute_trend_bayesian_weighted(
    ema_alignment: float, structure_trend: str, regime: str
) -> BayesianSignal:
    """EMA + structure trend signal with regime reliability."""
    reliability = TREND_RELIABILITY.get(regime, 0.75)
    base_signal = compute_trend_bayesian(ema_alignment, structure_trend)

    weighted_post = _weighted_posterior(
        base_signal.prior,
        base_signal.likelihood,
        reliability,
    )
    return BayesianSignal(
        signal_name  = base_signal.signal_name,
        condition    = base_signal.condition,
        prior        = base_signal.prior,
        likelihood   = round(base_signal.likelihood * reliability, 3),
        posterior    = round(weighted_post, 3),
        confidence   = _confidence(weighted_post),
        description  = (
            f"{base_signal.description} [rel={reliability:.0%}]"
        ),
    )


def compute_volatility_bayesian_weighted(
    vol_regime: str, atr_pct: float, regime: str
) -> BayesianSignal:
    """Volatility signal with regime reliability."""
    reliability = VOLATILITY_RELIABILITY.get(regime, 0.75)
    base_signal = compute_volatility_bayesian(vol_regime, atr_pct)

    weighted_post = _weighted_posterior(
        base_signal.prior,
        base_signal.likelihood,
        reliability,
    )
    return BayesianSignal(
        signal_name  = base_signal.signal_name,
        condition    = base_signal.condition,
        prior        = base_signal.prior,
        likelihood   = round(base_signal.likelihood * reliability, 3),
        posterior    = round(weighted_post, 3),
        confidence   = _confidence(weighted_post),
        description  = base_signal.description,
    )


# ── Reliability Audit ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ReliabilityAudit:
    """Shows which signals were dampened and by how much."""
    rsi_reliability:        float
    trend_reliability:      float
    vol_reliability:        float
    regime_reliability:     float
    dominant_signal:        str     # which signal has highest weighted posterior
    dominance_prevented:    bool    # True if weighting changed the dominant signal
    regime_applied:         str


def _audit_reliability(
    regime: str,
    weighted_signals: tuple[BayesianSignal, ...],
    unweighted_signals: tuple[BayesianSignal, ...],
) -> ReliabilityAudit:
    """Compare weighted vs unweighted to detect dominance prevention."""
    names = [s.signal_name for s in weighted_signals]
    w_posteriors  = [s.posterior for s in weighted_signals]
    uw_posteriors = [s.posterior for s in unweighted_signals]

    dominant_w  = names[w_posteriors.index(max(w_posteriors))]
    dominant_uw = names[uw_posteriors.index(max(uw_posteriors))]

    return ReliabilityAudit(
        rsi_reliability    = RSI_RELIABILITY.get(regime, 0.70),
        trend_reliability  = TREND_RELIABILITY.get(regime, 0.75),
        vol_reliability    = VOLATILITY_RELIABILITY.get(regime, 0.75),
        regime_reliability = REGIME_RELIABILITY.get(regime, 0.90),
        dominant_signal    = dominant_w,
        dominance_prevented= (dominant_w != dominant_uw),
        regime_applied     = regime,
    )


# ── Main Entry — Drop-in Replacement ──────────────────────────────────────────
def compute_bayesian_analysis_v2(
    rsi:               float,
    regime:            str,
    regime_confidence: float,
    ema_alignment:     float,
    structure_trend:   str,
    vol_regime:        str,
    atr_pct:           float,
) -> BayesianResult:
    """
    Reliability-weighted Bayesian analysis.
    Drop-in replacement for compute_bayesian_analysis() with regime gating.

    Parameters — identical to compute_bayesian_analysis().
    Returns     — BayesianResult (identical contract).
    """
    # ── Reliability-weighted signals ─────────────────────────────────────────
    rsi_sig    = compute_rsi_bayesian_weighted(rsi, regime)
    regime_sig = compute_regime_bayesian_weighted(regime, regime_confidence)
    trend_sig  = compute_trend_bayesian_weighted(ema_alignment, structure_trend, regime)
    vol_sig    = compute_volatility_bayesian_weighted(vol_regime, atr_pct, regime)

    signals = (rsi_sig, regime_sig, trend_sig, vol_sig)

    # ── Composite probabilities (same logic as original) ─────────────────────
    w = [0.25, 0.35, 0.25, 0.15]
    is_bull = {
        "STRONG_BULL": True, "BULL": True,
        "RANGE": None, "BEAR": False, "STRONG_BEAR": False,
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
        interp = (
            f"Strong bullish edge [reliability-weighted]: "
            f"P(Bull)={bull_prob*100:.0f}% vs P(Bear)={bear_prob*100:.0f}%"
        )
    elif net_edge < -0.15:
        interp = (
            f"Strong bearish edge [reliability-weighted]: "
            f"P(Bear)={bear_prob*100:.0f}% vs P(Bull)={bull_prob*100:.0f}%"
        )
    elif abs(net_edge) < 0.05:
        interp = f"No clear edge: P(Bull)≈P(Bear)≈{bull_prob*100:.0f}%"
    else:
        interp = f"Moderate edge: net={net_edge*100:+.0f}%"

    # Log if RSI was dampened (the key use case this phase solves)
    rsi_rel = RSI_RELIABILITY.get(regime, 0.70)
    if rsi_rel < 0.50:
        logger.debug(
            "[bayes_v2] RSI reliability=%.0f%% in %s — RSI posterior dampened from raw signal",
            rsi_rel * 100, regime,
        )

    return BayesianResult(
        signals              = signals,
        composite_bull_prob  = bull_prob,
        composite_bear_prob  = bear_prob,
        net_edge             = net_edge,
        interpretation       = interp,
    )
