"""
Risk Budget + Exposure Engine  (Phase 8)
==========================================
Calculates portfolio heat, risk budget usage, and exposure metrics.

Portfolio Heat  = sum of active position risks as % of total capital
Risk Budget Used = portfolio_heat / max_risk_budget
Remaining Capacity = max_budget - used

Exposure:
  Net Exposure   = long_exposure - short_exposure
  Gross Exposure = long_exposure + short_exposure
  Delta Exposure = sum(position_delta × notional)
  Beta Exposure  = sum(position_beta × weight)
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

MAX_PORTFOLIO_RISK = 0.06     # 6% total portfolio heat max
MAX_SINGLE_POSITION = 0.02    # 2% per position


@dataclass(frozen=True)
class PositionRisk:
    symbol:         str
    direction:      str      # LONG | SHORT
    risk_pct:       float    # % of capital at risk
    delta:          float    # option delta or 1.0 for stocks
    beta:           float    # vs SPY, default 1.0
    notional:       float    # position notional value


@dataclass(frozen=True)
class RiskBudgetResult:
    portfolio_heat:       float    # % of capital currently at risk
    risk_budget_used_pct: float    # 0–100% of MAX_PORTFOLIO_RISK used
    remaining_capacity:   float    # remaining risk % available
    at_capacity:          bool     # True if heat >= 80% of budget
    over_capacity:        bool     # True if heat > budget

    # Exposure
    net_exposure:   float    # long - short (% of capital)
    gross_exposure: float    # long + short
    delta_exposure: float    # sum(delta × weight)
    beta_exposure:  float    # sum(beta × weight)

    # Recommendations
    new_position_allowed: bool
    max_new_risk_pct:     float    # max risk % for next trade
    recommendation:       str
    active_positions:     int


def compute_risk_budget(
    positions:      list[PositionRisk],
    total_capital:  float = 100_000,
) -> RiskBudgetResult:
    """
    Compute portfolio risk budget usage and exposure metrics.

    Parameters
    ----------
    positions     : list of active positions with risk/delta/beta
    total_capital : total account equity (for notional calcs)
    """
    if not positions:
        return RiskBudgetResult(
            portfolio_heat=0, risk_budget_used_pct=0,
            remaining_capacity=MAX_PORTFOLIO_RISK*100,
            at_capacity=False, over_capacity=False,
            net_exposure=0, gross_exposure=0,
            delta_exposure=0, beta_exposure=0,
            new_position_allowed=True,
            max_new_risk_pct=MAX_SINGLE_POSITION*100,
            recommendation="Portfolio empty — full risk budget available",
            active_positions=0,
        )

    longs  = [p for p in positions if p.direction == "LONG"]
    shorts = [p for p in positions if p.direction == "SHORT"]

    portfolio_heat  = sum(p.risk_pct for p in positions)
    budget_used_pct = round(portfolio_heat / MAX_PORTFOLIO_RISK * 100, 1)
    remaining       = max(0.0, MAX_PORTFOLIO_RISK * 100 - portfolio_heat * 100)

    at_cap  = portfolio_heat >= MAX_PORTFOLIO_RISK * 0.80
    over_cap= portfolio_heat >= MAX_PORTFOLIO_RISK

    # Exposure
    total_notional = sum(p.notional for p in positions) or total_capital
    long_exp  = sum(p.notional for p in longs)  / total_capital * 100
    short_exp = sum(p.notional for p in shorts) / total_capital * 100
    net_exp   = long_exp - short_exp
    gross_exp = long_exp + short_exp

    delta_exp = sum(p.delta * (p.notional / total_capital) for p in positions)
    beta_exp  = sum(p.beta  * (p.notional / total_capital) for p in positions)

    # New position sizing
    new_allowed    = not over_cap
    max_new_risk   = max(0.0, min(MAX_SINGLE_POSITION, MAX_PORTFOLIO_RISK - portfolio_heat))

    if over_cap:
        rec = f"⚠️ Over risk budget ({portfolio_heat*100:.1f}% > {MAX_PORTFOLIO_RISK*100:.0f}%) — reduce positions"
    elif at_cap:
        rec = f"🟡 Near capacity ({portfolio_heat*100:.1f}%) — limit new positions to {max_new_risk*100:.2f}%"
    else:
        rec = f"✅ {remaining:.1f}% risk budget remaining — new positions allowed up to {max_new_risk*100:.2f}%"

    logger.info("[risk_budget] heat=%.2f%% used=%.0f%% remaining=%.2f%% positions=%d",
                portfolio_heat*100, budget_used_pct, remaining, len(positions))

    return RiskBudgetResult(
        portfolio_heat       = round(portfolio_heat * 100, 3),
        risk_budget_used_pct = budget_used_pct,
        remaining_capacity   = round(remaining, 3),
        at_capacity          = at_cap,
        over_capacity        = over_cap,
        net_exposure         = round(net_exp, 2),
        gross_exposure       = round(gross_exp, 2),
        delta_exposure       = round(delta_exp, 4),
        beta_exposure        = round(beta_exp, 4),
        new_position_allowed = new_allowed,
        max_new_risk_pct     = round(max_new_risk * 100, 3),
        recommendation       = rec,
        active_positions     = len(positions),
    )
