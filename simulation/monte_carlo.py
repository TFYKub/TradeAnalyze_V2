"""
Monte Carlo Simulation Engine – with caching
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.thresholds import THRESHOLDS

logger = logging.getLogger(__name__)

N_SIMULATIONS  = 10_000
DEFAULT_HORIZON = 20
RISK_FREE_DAILY = 0.05 / 252

# Cache for volatility parameters per symbol (24h TTL)
_MC_CACHE = {}  # symbol -> (mu, sigma, timestamp)


def _get_vol_params(close_series: pd.Series, symbol: str, ttl_seconds: int = 86400):
    now = time.time()
    if symbol in _MC_CACHE:
        mu, sigma, ts = _MC_CACHE[symbol]
        if now - ts < ttl_seconds:
            return mu, sigma
    log_returns = np.log(close_series / close_series.shift(1)).dropna()
    mu = float(log_returns.mean())
    sigma = float(log_returns.std())
    if sigma <= 0:
        sigma = 0.01
    _MC_CACHE[symbol] = (mu, sigma, now)
    return mu, sigma


@dataclass(frozen=True)
class MonteCarloResult:
    simulations:          int
    horizon:              int
    prob_profit:          float
    prob_stop_hit:        float
    prob_target_hit:      float
    expected_return_pct:  float
    expected_drawdown_pct: float
    ci_95_low:            float
    ci_95_high:           float
    var_95:               float
    cvar_95:              float
    sharpe_simulated:     float
    sortino_simulated:    float


def run_monte_carlo(
    close_series: pd.Series,
    entry:        float,
    stop_loss:    float,
    target:       float,
    horizon:      int   = DEFAULT_HORIZON,
    simulations:  int   = N_SIMULATIONS,
    direction:    str   = "LONG",
    symbol:       str   = "default",
) -> MonteCarloResult:
    # Use cached mu, sigma
    mu, sigma = _get_vol_params(close_series, symbol)
    drift = mu - 0.5 * sigma ** 2

    rng    = np.random.default_rng(42)
    shocks = rng.normal(drift, sigma, size=(simulations, horizon))
    log_paths = np.cumsum(shocks, axis=1)
    price_paths = entry * np.exp(log_paths)
    final_prices = price_paths[:, -1]

    if THRESHOLDS.MODEL_TRANSACTION_COSTS:
        cost_pct = THRESHOLDS.COST_STOCK_PCT * 2
    else:
        cost_pct = 0.0

    final_returns_pct = (final_prices - entry) / entry * 100 - cost_pct * 100

    if direction.upper() == "LONG":
        prob_profit = float((final_prices > entry).mean() * 100)
        stop_touched = (price_paths.min(axis=1) <= stop_loss)
        target_touched = (price_paths.max(axis=1) >= target)
    else:
        prob_profit = float((final_prices < entry).mean() * 100)
        stop_touched = (price_paths.max(axis=1) >= stop_loss)
        target_touched = (price_paths.min(axis=1) <= target)

    prob_stop_hit = float(stop_touched.mean() * 100)
    prob_target_hit = float(target_touched.mean() * 100)
    expected_return = float(final_returns_pct.mean())

    running_max = np.maximum.accumulate(price_paths, axis=1)
    drawdown_per_path = ((price_paths - running_max) / running_max).min(axis=1)
    expected_dd = float(drawdown_per_path.mean() * 100)

    ci_low = float(np.percentile(final_returns_pct, 2.5))
    ci_high = float(np.percentile(final_returns_pct, 97.5))
    var_95 = float(-np.percentile(final_returns_pct, 5))
    tail = final_returns_pct[final_returns_pct <= np.percentile(final_returns_pct, 5)]
    cvar_95 = float(-tail.mean()) if len(tail) > 0 else var_95

    daily_path_returns = np.diff(np.log(price_paths), axis=1)
    mean_daily = daily_path_returns.mean(axis=1)
    std_daily = daily_path_returns.std(axis=1)
    excess_daily = mean_daily - RISK_FREE_DAILY
    sharpe_per_path = np.where(std_daily > 0, excess_daily / std_daily * np.sqrt(252), 0)
    sharpe_sim = float(sharpe_per_path.mean())

    downside = np.where(daily_path_returns < 0, daily_path_returns, 0)
    downside_std = downside.std(axis=1)
    sortino_per_path = np.where(
        downside_std > 0,
        excess_daily / downside_std * np.sqrt(252),
        0
    )
    sortino_sim = float(sortino_per_path.mean())

    logger.info(
        "MC[%d paths, %dd, %s]  P(profit)=%.1f%%  P(SL)=%.1f%%  P(TP)=%.1f%%  "
        "E[ret]=%.2f%%  VaR95=%.2f%%  cost=%.3f%% (cached params)",
        simulations, horizon, direction,
        prob_profit, prob_stop_hit, prob_target_hit,
        expected_return, var_95, cost_pct * 100,
    )

    return MonteCarloResult(
        simulations=simulations,
        horizon=horizon,
        prob_profit=round(prob_profit, 1),
        prob_stop_hit=round(prob_stop_hit, 1),
        prob_target_hit=round(prob_target_hit, 1),
        expected_return_pct=round(expected_return, 2),
        expected_drawdown_pct=round(expected_dd, 2),
        ci_95_low=round(ci_low, 2),
        ci_95_high=round(ci_high, 2),
        var_95=round(var_95, 2),
        cvar_95=round(cvar_95, 2),
        sharpe_simulated=round(sharpe_sim, 3),
        sortino_simulated=round(sortino_sim, 3),
    )