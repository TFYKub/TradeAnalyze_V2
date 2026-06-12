"""
Portfolio Optimizer  (Phase 20)
=================================
Institutional portfolio construction using two complementary methods:

1. Risk Parity
   Each asset contributes equally to total portfolio volatility.
   Weight ∝ 1/σᵢ (inverse volatility), iteratively refined.
   - Avoids concentration in low-vol assets dominating notional
   - Produces robust, diversified allocation

2. Hierarchical Risk Parity (HRP)  [López de Prado, 2016]
   Step 1 — Tree Clustering: hierarchical cluster assets by correlation distance
   Step 2 — Quasi-Diagonalisation: reorder covariance matrix
   Step 3 — Recursive Bisection: allocate weights top-down
   - Robust to estimation error in correlation matrices
   - Outperforms mean-variance in out-of-sample tests

Outputs:
  recommended_weights   : dict[symbol: float]  (sum to 1.0)
  risk_contributions    : dict[symbol: float]  (% of total portfolio vol)
  portfolio_volatility  : float (annualised %)
  portfolio_drawdown    : float (max drawdown estimate %)
  method_used           : "risk_parity" | "hrp" | "equal_weight"
  diversification_ratio : float (weighted avg vol / portfolio vol)

Integration:
  portfolio/optimizer.py is called by FuturesOrchestrator for multi-asset portfolios
  Single-asset analysis uses the volatility and drawdown outputs only
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)

MIN_HISTORY   = 60     # minimum bars per asset
RISK_FREE_PCT = 5.0    # annualised %


@dataclass(frozen=True)
class PortfolioOptimizationResult:
    """Output contract for the Portfolio Optimizer."""
    method_used:           str              # "risk_parity" | "hrp" | "equal_weight"
    recommended_weights:   dict[str, float] # {symbol: weight}, sum=1.0
    risk_contributions:    dict[str, float] # {symbol: % of total vol}
    portfolio_volatility:  float            # annualised %
    portfolio_sharpe:      float            # (return - rf) / vol
    portfolio_drawdown:    float            # max drawdown estimate %
    diversification_ratio: float            # weighted avg vol / portfolio vol
    correlation_matrix:    dict[str, dict[str, float]]  # {sym: {sym: corr}}
    volatilities:          dict[str, float] # {symbol: annualised vol %}
    expected_returns:      dict[str, float] # {symbol: annualised return %}
    n_assets:              int
    interpretation:        str


# ── Covariance / Return Estimation ────────────────────────────────────────────
def _compute_statistics(
    prices: dict[str, pd.Series],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Returns (cov_matrix, exp_returns, vols) from dict of price series.
    Uses exponentially weighted covariance for recency bias.
    """
    # Align series to common index
    price_df  = pd.DataFrame(prices).dropna()
    returns   = np.log(price_df / price_df.shift(1)).dropna()

    if len(returns) < MIN_HISTORY:
        raise ValueError(f"Insufficient history: {len(returns)} bars < {MIN_HISTORY}")

    # Exponentially weighted covariance (halflife = 60 days)
    cov_ann = returns.ewm(halflife=60, adjust=True).cov().iloc[-len(prices):] * 252

    # Annualised returns (EWM mean of log returns × 252)
    mu_ann  = returns.ewm(halflife=60, adjust=True).mean().iloc[-1] * 252

    # Annualised vols
    vols    = np.sqrt(np.diag(cov_ann.values)) * 100     # in %

    return cov_ann, mu_ann, pd.Series(vols, index=price_df.columns)


# ── Method 1: Risk Parity ─────────────────────────────────────────────────────
def _risk_parity_weights(
    cov: pd.DataFrame,
    max_iter: int = 200,
    tol: float    = 1e-8,
) -> pd.Series:
    """
    Compute risk parity weights via iterative algorithm.
    Converges to weights where each asset contributes equally to total vol.
    """
    n    = len(cov.columns)
    w    = pd.Series(np.ones(n) / n, index=cov.columns)

    sigma = np.sqrt(np.diag(cov.values))
    sigma = np.where(sigma < 1e-9, 1e-9, sigma)

    for iteration in range(max_iter):
        w_prev = w.copy()

        # Risk contributions: RC_i = w_i × (Σw)_i / sqrt(w'Σw)
        Sw     = cov.values @ w.values
        port_vol = math.sqrt(float(w.values @ Sw))
        if port_vol < 1e-12:
            break
        rc = w.values * Sw / port_vol

        # Update: target equal risk contribution = port_vol / n
        target_rc = port_vol / n
        # Naïve update toward equal risk
        w_new = w.values * (target_rc / (rc + 1e-12))
        w_new = np.maximum(w_new, 1e-8)   # enforce non-negativity
        w_new /= w_new.sum()
        w = pd.Series(w_new, index=cov.columns)

        if np.max(np.abs(w.values - w_prev.values)) < tol:
            logger.debug("[rp] converged at iteration %d", iteration)
            break

    return w / w.sum()


# ── Method 2: Hierarchical Risk Parity ────────────────────────────────────────
def _correlation_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """Convert correlation matrix to distance matrix: d = sqrt(0.5(1-r))."""
    dist = np.sqrt(0.5 * (1.0 - corr.values))
    return pd.DataFrame(dist, index=corr.index, columns=corr.columns)


def _get_quasi_diagonal(link, n: int) -> list[int]:
    """
    Reorder assets via quasi-diagonalisation from the linkage tree.
    Returns ordered list of original asset indices.
    """
    root = to_tree(link)

    def _recurse(node) -> list[int]:
        if node.is_leaf():
            return [node.id]
        return _recurse(node.left) + _recurse(node.right)

    return _recurse(root)


def _recursive_bisection(
    cov:       pd.DataFrame,
    sorted_idx: list[int],
) -> pd.Series:
    """
    Allocate weights by recursive bisection of the sorted covariance matrix.
    Each cluster gets weight proportional to inverse cluster variance.
    """
    assets = list(cov.columns)
    sorted_assets = [assets[i] for i in sorted_idx]
    weights = pd.Series(1.0, index=sorted_assets)

    clusters = [sorted_assets]

    while clusters:
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid   = len(cluster) // 2
            left  = cluster[:mid]
            right = cluster[mid:]

            # Variance of each sub-cluster (equal-weight within cluster)
            def _cluster_var(items):
                sub_w   = pd.Series(1.0 / len(items), index=items)
                sub_cov = cov.loc[items, items]
                return float(sub_w @ sub_cov @ sub_w)

            var_l = _cluster_var(left)
            var_r = _cluster_var(right)

            # Allocate inversely proportional to variance
            total = var_l + var_r + 1e-12
            alpha_l = 1.0 - var_l / total

            for item in left:
                weights[item] *= alpha_l
            for item in right:
                weights[item] *= (1.0 - alpha_l)

            new_clusters += [left, right]

        clusters = [c for c in new_clusters if len(c) > 1]

    return weights / weights.sum()


def _hrp_weights(cov: pd.DataFrame, corr: pd.DataFrame) -> pd.Series:
    """Full HRP pipeline."""
    dist      = _correlation_distance(corr)
    dist_sq   = squareform(dist.values, checks=False)
    link      = linkage(dist_sq, method="ward")
    n         = len(cov.columns)
    sort_idx  = _get_quasi_diagonal(link, n)
    return _recursive_bisection(cov, sort_idx)


# ── Risk Contribution Calculation ─────────────────────────────────────────────
def _compute_risk_contributions(
    weights: pd.Series,
    cov: pd.DataFrame,
) -> tuple[float, dict[str, float]]:
    """
    Returns (portfolio_vol_pct, {symbol: risk_contribution_pct}).
    """
    Sw       = cov.values @ weights.values
    port_var = float(weights.values @ Sw)
    port_vol = math.sqrt(max(port_var, 0.0)) * 100    # annualised %

    if port_vol < 1e-6:
        rc = {sym: round(100.0 / len(weights), 2) for sym in weights.index}
        return port_vol, rc

    rc_abs  = weights.values * Sw
    rc_pct  = rc_abs / (port_var + 1e-12) * 100
    rc_dict = {sym: round(float(rc_pct[i]), 2) for i, sym in enumerate(weights.index)}

    return round(port_vol, 2), rc_dict


# ── Max Drawdown Estimate ──────────────────────────────────────────────────────
def _estimate_drawdown(port_vol: float, port_ret: float) -> float:
    """
    Parametric max drawdown estimate using Gaussian assumption:
    E[MaxDD] ≈ σ × sqrt(2 × ln(T/Δt)) for T = 252 days
    This is an approximation; actual DD depends on path.
    """
    T    = 252
    try:
        factor = math.sqrt(2 * math.log(T))
        return round(port_vol * factor, 2)
    except Exception:
        return round(port_vol * 2.5, 2)


# ── Main Entry ─────────────────────────────────────────────────────────────────
def compute_portfolio_optimization(
    prices:     dict[str, pd.Series],   # {symbol: Close price Series}
    method:     str = "auto",           # "risk_parity" | "hrp" | "equal_weight" | "auto"
    min_weight: float = 0.02,           # minimum per-asset weight (floor)
    max_weight: float = 0.40,           # maximum per-asset weight (ceiling)
) -> PortfolioOptimizationResult:
    """
    Compute optimal portfolio weights using Risk Parity or HRP.

    Parameters
    ----------
    prices     : dict mapping symbol to Close price series (aligned dates recommended)
    method     : "auto" selects HRP for ≥5 assets, risk_parity otherwise
    min_weight : floor on any single asset weight
    max_weight : ceiling on any single asset weight

    Returns
    -------
    PortfolioOptimizationResult — always returns, never raises
    """
    symbols = list(prices.keys())
    n       = len(symbols)

    # ── Fallback for single asset or too few assets ───────────────────────────
    if n < 2:
        sym = symbols[0] if symbols else "UNKNOWN"
        price_series = prices.get(sym, pd.Series([1.0]))
        log_ret  = np.log(price_series / price_series.shift(1)).dropna()
        vol_pct  = float(log_ret.std() * math.sqrt(252) * 100) if len(log_ret) > 10 else 20.0
        dd_est   = _estimate_drawdown(vol_pct, 0.0)
        return PortfolioOptimizationResult(
            method_used           = "single_asset",
            recommended_weights   = {sym: 1.0},
            risk_contributions    = {sym: 100.0},
            portfolio_volatility  = round(vol_pct, 2),
            portfolio_sharpe      = 0.0,
            portfolio_drawdown    = dd_est,
            diversification_ratio = 1.0,
            correlation_matrix    = {sym: {sym: 1.0}},
            volatilities          = {sym: round(vol_pct, 2)},
            expected_returns      = {sym: 0.0},
            n_assets              = 1,
            interpretation        = f"Single asset — vol={vol_pct:.1f}% DD≈{dd_est:.1f}%",
        )

    # ── Compute statistics ────────────────────────────────────────────────────
    try:
        cov_ann, mu_ann, vols = _compute_statistics(prices)
    except Exception as exc:
        logger.warning("[optimizer] statistics failed: %s — using equal weight", exc)
        eq_w = {s: round(1.0 / n, 4) for s in symbols}
        return PortfolioOptimizationResult(
            method_used           = "equal_weight",
            recommended_weights   = eq_w,
            risk_contributions    = {s: round(100.0 / n, 2) for s in symbols},
            portfolio_volatility  = 0.0,
            portfolio_sharpe      = 0.0,
            portfolio_drawdown    = 0.0,
            diversification_ratio = 1.0,
            correlation_matrix    = {},
            volatilities          = {s: 0.0 for s in symbols},
            expected_returns      = {s: 0.0 for s in symbols},
            n_assets              = n,
            interpretation        = f"Equal weight (statistics unavailable): {n} assets",
        )

    # ── Correlation matrix ────────────────────────────────────────────────────
    sigma_diag   = np.sqrt(np.diag(cov_ann.values))
    sigma_outer  = np.outer(sigma_diag, sigma_diag) + 1e-12
    corr_matrix  = cov_ann.values / sigma_outer
    corr_df      = pd.DataFrame(corr_matrix, index=cov_ann.index, columns=cov_ann.columns)
    corr_dict    = {
        row: {col: round(float(corr_df.loc[row, col]), 4) for col in corr_df.columns}
        for row in corr_df.index
    }

    # ── Select method ────────────────────────────────────────────────────────
    if method == "auto":
        method = "hrp" if n >= 5 else "risk_parity"

    weights_series: pd.Series
    actual_method = method

    try:
        if method == "hrp":
            weights_series = _hrp_weights(cov_ann, corr_df)
        elif method == "risk_parity":
            weights_series = _risk_parity_weights(cov_ann)
        else:
            weights_series = pd.Series(np.ones(n) / n, index=cov_ann.columns)
            actual_method  = "equal_weight"
    except Exception as exc:
        logger.warning("[optimizer] %s failed: %s — fallback to equal weight", method, exc)
        weights_series = pd.Series(np.ones(n) / n, index=cov_ann.columns)
        actual_method  = "equal_weight"

    # ── Apply weight constraints ─────────────────────────────────────────────
    w = weights_series.clip(lower=min_weight, upper=max_weight)
    w = w / w.sum()    # re-normalise

    # ── Risk contributions ────────────────────────────────────────────────────
    port_vol, rc_dict = _compute_risk_contributions(w, cov_ann)

    # ── Portfolio metrics ─────────────────────────────────────────────────────
    port_ret_ann = float(w.values @ mu_ann.reindex(w.index).fillna(0).values) * 100
    port_sharpe  = round((port_ret_ann - RISK_FREE_PCT) / (port_vol + 1e-6), 3) if port_vol > 0 else 0.0
    port_dd      = _estimate_drawdown(port_vol, port_ret_ann)

    # Diversification ratio = weighted avg vol / portfolio vol
    avg_vol = float(w.values @ vols.reindex(w.index).fillna(0).values)
    div_ratio = round(avg_vol / (port_vol + 1e-6), 3) if port_vol > 0 else 1.0

    rec_weights = {sym: round(float(w[sym]), 4) for sym in w.index}
    vols_dict   = {sym: round(float(vols.get(sym, 0.0)), 2) for sym in symbols}
    ret_dict    = {sym: round(float(mu_ann.get(sym, 0.0) * 100), 2) for sym in symbols}

    interp = (
        f"[{actual_method}] {n} assets | "
        f"Port Vol={port_vol:.1f}% | Sharpe={port_sharpe:.2f} | "
        f"MaxDD≈{port_dd:.1f}% | DivRatio={div_ratio:.2f} | "
        f"Weights: {', '.join(f'{s}={v:.0%}' for s, v in rec_weights.items())}"
    )

    logger.info(
        "[optimizer] method=%s n=%d vol=%.1f%% sharpe=%.2f dd≈%.1f%% div=%.2f",
        actual_method, n, port_vol, port_sharpe, port_dd, div_ratio,
    )

    return PortfolioOptimizationResult(
        method_used           = actual_method,
        recommended_weights   = rec_weights,
        risk_contributions    = rc_dict,
        portfolio_volatility  = port_vol,
        portfolio_sharpe      = port_sharpe,
        portfolio_drawdown    = port_dd,
        diversification_ratio = div_ratio,
        correlation_matrix    = corr_dict,
        volatilities          = vols_dict,
        expected_returns      = ret_dict,
        n_assets              = n,
        interpretation        = interp,
    )
