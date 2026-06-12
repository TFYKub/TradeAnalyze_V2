"""
Institutional Options Engine  (Phase 19)
==========================================
Upgrades the options layer to institutional-grade with:

1. Advanced Greeks Engine — adds second-order and cross-Greeks:
     vanna  : dDelta/dVol  = change in delta per 1% IV move
     charm  : dDelta/dTime = delta decay per day (delta bleed)
     vomma  : dVega/dVol   = vega convexity (vol sensitivity of vega)
     veta   : dVega/dTime  = vega decay per day
     speed  : dGamma/dS    = how gamma changes with price

2. Probability Engine Upgrade:
     POP    (Probability of Profit) — MC with drift
     P50    (Probability of 50% max profit) — for premium selling
     Expected Move (1σ, 1SD) — EM = price × IV × √(DTE/365)

3. Strategy Evaluation:
     Expected Value (EV) per strategy
     Kelly Fraction — optimal position sizing
     Sharpe Estimate — EV / (std of outcome)

4. Full Recommendation Format:
     Every options recommendation includes:
     Strategy | Strike | DTE | POP | EV | Delta | Theta | Vega | Kelly Size

All functions are additive — they call existing greeks_engine.py and extend it.
Backward compatible: existing callers are unaffected.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)

RISK_FREE_RATE = 0.05
MIN_IV         = 0.001
MIN_T          = 1 / 365


# ── Advanced Greeks Dataclass ──────────────────────────────────────────────────
@dataclass(frozen=True)
class AdvancedGreeks:
    """Complete institutional Greeks set — first and second order."""
    # First order (from existing greeks_engine.py)
    delta:   float
    gamma:   float
    theta:   float
    vega:    float
    rho:     float
    # Second order / cross Greeks
    vanna:   float    # dDelta/dVol  — delta sensitivity to IV changes
    charm:   float    # dDelta/dTime — delta decay per calendar day
    vomma:   float    # dVega/dVol   — vega convexity
    veta:    float    # dVega/dTime  — vega decay per day
    speed:   float    # dGamma/dS    — how gamma changes with price move


@dataclass(frozen=True)
class ProbabilityMetrics:
    """Enhanced probability metrics for options strategies."""
    pop:            float    # Probability of Profit (%)
    p50:            float    # Probability of 50% max profit (for credit spreads)
    expected_move:  float    # 1σ expected move in $ (price × IV × √(DTE/365))
    expected_move_pct: float # expected move as % of current price
    prob_above_be:  float    # P(price > breakeven) %
    prob_below_be:  float    # P(price < breakeven) %
    upper_1sd:      float    # spot + expected_move
    lower_1sd:      float    # spot − expected_move
    upper_2sd:      float    # spot + 2 × expected_move
    lower_2sd:      float    # spot − 2 × expected_move


@dataclass(frozen=True)
class StrategyEvaluation:
    """EV, Kelly, and Sharpe for an options strategy."""
    strategy_name:   str
    strike:          float         # primary strike (or width for spreads)
    dte:             int
    direction:       str           # LONG | SHORT

    # Greeks snapshot
    greeks:          AdvancedGreeks
    probabilities:   ProbabilityMetrics

    # P&L structure
    max_profit:      float         # $ or premium received
    max_loss:        float         # $ or premium paid (positive = loss)
    breakeven:       float         # price at expiry for breakeven

    # Evaluation metrics
    pop:             float         # Probability of Profit (%)
    ev:              float         # Expected Value ($)
    kelly_fraction:  float         # optimal bet size (0–0.25 capped)
    sharpe_estimate: float         # EV / outcome_std_proxy

    recommendation:  str           # BUY | SELL | SKIP
    recommendation_reason: str


@dataclass(frozen=True)
class InstitutionalOptionsRecommendation:
    """
    Full institutional options recommendation.
    Every field required by the spec:
      Strategy | Strike | DTE | POP | Expected Value | Delta | Theta | Vega | Kelly Size
    """
    strategy:        str
    strike:          float
    dte:             int
    pop:             float          # %
    expected_value:  float          # $ EV
    delta:           float
    theta:           float          # daily $ decay
    vega:            float          # $ per 1% IV move
    kelly_size:      float          # fraction of account

    # Extended
    gamma:           float
    vanna:           float
    charm:           float
    vomma:           float
    expected_move:   float          # 1σ $ move
    prob_50:         float          # P50 %
    max_profit:      float
    max_loss:        float
    breakeven:       float
    iv_used:         float          # IV used in calculation

    summary_line:    str            # one-line trade card for report


# ── Black-Scholes with Full Greeks ────────────────────────────────────────────
def compute_advanced_greeks(
    S: float,
    K: float,
    T: float,          # years to expiry
    sigma: float,      # annual IV
    r: float = RISK_FREE_RATE,
    option_type: str = "call",
) -> AdvancedGreeks:
    """
    Compute complete institutional Greeks including second-order / cross Greeks.

    Parameters
    ----------
    S           : spot price
    K           : strike price
    T           : years to expiry (e.g. 30/365)
    sigma       : implied volatility (annual, e.g. 0.30 = 30%)
    r           : risk-free rate
    option_type : "call" | "put"

    Returns
    -------
    AdvancedGreeks — returns zeros if inputs are degenerate
    """
    if any(v <= 0 for v in (S, K, sigma)):
        return AdvancedGreeks(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    T     = max(T, MIN_T)
    sigma = max(sigma, MIN_IV)
    sign  = 1 if option_type == "call" else -1

    try:
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(sign * d1)
        cdf_d2 = norm.cdf(sign * d2)
        disc   = math.exp(-r * T)

        # ── First order ──────────────────────────────────────────────────────
        delta = sign * cdf_d1
        gamma = pdf_d1 / (S * sigma * sqrt_T)
        theta = (-(S * pdf_d1 * sigma) / (2 * sqrt_T)
                 - sign * r * K * disc * cdf_d2) / 365
        vega  = S * pdf_d1 * sqrt_T / 100
        rho   = sign * K * T * disc * cdf_d2 / 100

        # ── Second order / cross Greeks ───────────────────────────────────────
        # Vanna: dDelta/dVol = dVega/dS
        # Vanna = -pdf(d1) × d2 / sigma
        vanna = -(pdf_d1 * d2) / sigma / 100   # per 1% IV move

        # Charm: dDelta/dTime (delta bleed per day)
        # Charm = -pdf(d1) × (2rT - d2×sigma×sqrt_T) / (2T×sigma×sqrt_T)
        if T > MIN_T:
            charm_num = 2 * r * T - d2 * sigma * sqrt_T
            charm_den = 2 * T * sigma * sqrt_T
            charm = sign * pdf_d1 * charm_num / charm_den / 365
        else:
            charm = 0.0

        # Vomma (Volga): dVega/dVol = Vega × d1 × d2 / sigma
        # Represents convexity of vega w.r.t. vol
        vomma = vega * d1 * d2 / sigma   # $ per (1% IV)² move

        # Veta: dVega/dTime (vega decay per day)
        # Veta = -Vega × [r×d1/(sigma×sqrt_T) - (1+d1×d2)/(2T)] / 365
        if T > MIN_T:
            veta_factor = r * d1 / (sigma * sqrt_T) - (1 + d1 * d2) / (2 * T)
            veta = -vega * veta_factor / 365
        else:
            veta = 0.0

        # Speed: dGamma/dS = -Gamma × (d1 / (sigma × sqrt_T) + 1) / S
        speed = -gamma * (d1 / (sigma * sqrt_T) + 1) / S

        return AdvancedGreeks(
            delta  = round(float(delta),  5),
            gamma  = round(float(gamma),  6),
            theta  = round(float(theta),  5),
            vega   = round(float(vega),   5),
            rho    = round(float(rho),    5),
            vanna  = round(float(vanna),  6),
            charm  = round(float(charm),  6),
            vomma  = round(float(vomma),  5),
            veta   = round(float(veta),   5),
            speed  = round(float(speed),  7),
        )

    except Exception as exc:
        logger.debug("[greeks_adv] error S=%s K=%s T=%s sigma=%s: %s", S, K, T, sigma, exc)
        return AdvancedGreeks(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)


# ── Probability Metrics ────────────────────────────────────────────────────────
def compute_probability_metrics(
    price:      float,
    iv:         float,      # annual
    dte:        int,        # calendar days to expiry
    breakeven:  float,
    direction:  str,        # "LONG" | "SHORT"
    n_paths:    int = 5_000,
) -> ProbabilityMetrics:
    """
    Compute POP, P50, and expected move using closed-form + Monte Carlo hybrid.
    """
    T      = max(dte / 365.0, MIN_T)
    em     = price * iv * math.sqrt(T)                    # 1σ expected move $
    em_pct = em / price * 100 if price > 0 else 0.0

    upper_1sd = round(price + em, 4)
    lower_1sd = round(price - em, 4)
    upper_2sd = round(price + 2 * em, 4)
    lower_2sd = round(price - 2 * em, 4)

    # GBM simulation for prob metrics
    try:
        rng      = np.random.default_rng(42)
        drift    = -0.5 * iv ** 2 * T
        diffuse  = iv * math.sqrt(T) * rng.standard_normal(n_paths)
        terminal = price * np.exp(drift + diffuse)

        if direction == "LONG":
            pop = float((terminal > breakeven).mean() * 100)
        else:
            pop = float((terminal < breakeven).mean() * 100)

        # P50: for credit strategies, P(close enough to take 50% of max profit)
        # Approximation: P(terminal stays within 50% of max move)
        p50 = float((
            (terminal > lower_1sd * 0.8) & (terminal < upper_1sd * 1.2)
        ).mean() * 100)

        prob_above = float((terminal > breakeven).mean() * 100)
        prob_below = float((terminal < breakeven).mean() * 100)

    except Exception:
        pop        = 50.0
        p50        = 60.0
        prob_above = 50.0
        prob_below = 50.0

    return ProbabilityMetrics(
        pop              = round(pop,        1),
        p50              = round(p50,        1),
        expected_move    = round(em,         4),
        expected_move_pct= round(em_pct,     2),
        prob_above_be    = round(prob_above, 1),
        prob_below_be    = round(prob_below, 1),
        upper_1sd        = upper_1sd,
        lower_1sd        = lower_1sd,
        upper_2sd        = upper_2sd,
        lower_2sd        = lower_2sd,
    )


# ── Strategy EV + Kelly + Sharpe ──────────────────────────────────────────────
def compute_strategy_evaluation(
    strategy_name: str,
    strike:        float,
    dte:           int,
    direction:     str,        # "LONG" | "SHORT"
    price:         float,      # current spot
    iv:            float,
    premium:       float,      # option price (credit received or debit paid)
    max_profit:    float,      # $ maximum profit
    max_loss:      float,      # $ maximum loss (positive number)
    breakeven:     float,
) -> tuple[float, float, float]:
    """
    Returns (ev, kelly_fraction, sharpe_estimate).

    ev              = E[profit] using POP-weighted payoff
    kelly_fraction  = max(0, (pop×win/loss - loss) / win), capped at 0.25
    sharpe_estimate = ev / outcome_std
    """
    pop_decimal = compute_probability_metrics(price, iv, dte, breakeven, direction).pop / 100.0
    w = pop_decimal
    l = 1.0 - w

    win_amt  = max_profit if max_profit > 0 else premium * 2
    loss_amt = max_loss   if max_loss  > 0 else premium * 3

    ev = w * win_amt - l * loss_amt

    if loss_amt > 0:
        r_ratio = win_amt / loss_amt
        kelly   = max(0.0, min(0.25, (w * r_ratio - l) / r_ratio))
    else:
        kelly = 0.0

    # Sharpe: EV / std of binary outcome
    outcome_std = math.sqrt(w * l) * (win_amt + loss_amt)
    sharpe      = ev / outcome_std if outcome_std > 0 else 0.0

    return round(ev, 2), round(kelly, 4), round(sharpe, 3)


# ── Full Recommendation Builder ────────────────────────────────────────────────
def build_options_recommendation(
    strategy:    str,
    strike:      float,
    dte:         int,
    direction:   str,          # "LONG" | "SHORT"
    price:       float,
    iv:          float,
    premium:     float,
    max_profit:  float,
    max_loss:    float,
    breakeven:   float,
    option_type: str = "call",
    r:           float = RISK_FREE_RATE,
) -> InstitutionalOptionsRecommendation:
    """
    Build a complete institutional options recommendation card.

    All inputs are the same as existing strategy_models.StrategySetup fields.
    """
    T = max(dte / 365.0, MIN_T)

    # ── Advanced Greeks ───────────────────────────────────────────────────────
    greeks = compute_advanced_greeks(price, strike, T, iv, r, option_type)

    # ── Probability metrics ──────────────────────────────────────────────────
    probs = compute_probability_metrics(price, iv, dte, breakeven, direction)

    # ── EV / Kelly / Sharpe ──────────────────────────────────────────────────
    ev, kelly, sharpe = compute_strategy_evaluation(
        strategy, strike, dte, direction, price, iv,
        premium, max_profit, max_loss, breakeven,
    )

    # ── Summary line (for report) ────────────────────────────────────────────
    summary = (
        f"{strategy} | K={strike:.2f} | {dte}DTE | "
        f"POP={probs.pop:.0f}% | EV=${ev:+.2f} | "
        f"δ={greeks.delta:+.3f} θ={greeks.theta:+.4f} ν={greeks.vega:.4f} | "
        f"Kelly={kelly:.3f}"
    )

    logger.debug("[options_inst] %s", summary)

    return InstitutionalOptionsRecommendation(
        strategy       = strategy,
        strike         = strike,
        dte            = dte,
        pop            = probs.pop,
        expected_value = ev,
        delta          = greeks.delta,
        theta          = greeks.theta,
        vega           = greeks.vega,
        kelly_size     = kelly,
        gamma          = greeks.gamma,
        vanna          = greeks.vanna,
        charm          = greeks.charm,
        vomma          = greeks.vomma,
        expected_move  = probs.expected_move,
        prob_50        = probs.p50,
        max_profit     = max_profit,
        max_loss       = max_loss,
        breakeven      = breakeven,
        iv_used        = round(iv, 4),
        summary_line   = summary,
    )


# ── Batch Recommendation Ranker ───────────────────────────────────────────────
def rank_recommendations(
    recommendations: list[InstitutionalOptionsRecommendation],
) -> list[InstitutionalOptionsRecommendation]:
    """
    Rank recommendations by composite score: EV × POP × Kelly.
    Higher = better. Returns sorted descending.
    """
    def _score(r: InstitutionalOptionsRecommendation) -> float:
        ev_norm    = min(1.0, max(0.0, (r.expected_value + 500) / 1000))
        pop_norm   = r.pop / 100.0
        kelly_norm = r.kelly_size / 0.25
        return ev_norm * 0.40 + pop_norm * 0.40 + kelly_norm * 0.20

    return sorted(recommendations, key=_score, reverse=True)
