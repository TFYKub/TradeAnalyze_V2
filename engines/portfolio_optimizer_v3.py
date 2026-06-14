"""
Portfolio Optimizer V3 – Risk Parity, MinVar, Max Sharpe, Kelly
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class PortfolioOptimizationResultV3:
    method: str
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe: float
    diversification_ratio: float

class PortfolioOptimizerV3:
    def __init__(self, returns_df: pd.DataFrame, risk_free_rate: float = 0.05):
        self.returns = returns_df
        self.cov = returns_df.cov() * 252
        self.mean = returns_df.mean() * 252
        self.rf = risk_free_rate
        self.assets = returns_df.columns.tolist()

    def _result(self, weights_array, method) -> PortfolioOptimizationResultV3:
        weights = dict(zip(self.assets, weights_array))
        port_ret = np.sum(weights_array * self.mean)
        port_vol = np.sqrt(weights_array @ self.cov @ weights_array)
        sharpe = (port_ret - self.rf) / port_vol if port_vol > 0 else 0
        avg_vol = np.sum(weights_array * np.sqrt(np.diag(self.cov)))
        div_ratio = avg_vol / port_vol if port_vol > 0 else 1.0
        return PortfolioOptimizationResultV3(
            method=method, weights=weights,
            expected_return=port_ret, volatility=port_vol,
            sharpe=sharpe, diversification_ratio=div_ratio
        )

    def risk_parity(self) -> PortfolioOptimizationResultV3:
        n = len(self.assets)
        def objective(w):
            port_vol = np.sqrt(w @ self.cov @ w)
            rc = w * (self.cov @ w) / port_vol
            return np.sum((rc - port_vol/n)**2)
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(n)]
        w0 = np.ones(n)/n
        res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=cons)
        return self._result(res.x, "RiskParity")

    def min_variance(self) -> PortfolioOptimizationResultV3:
        n = len(self.assets)
        def objective(w):
            return w @ self.cov @ w
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(n)]
        w0 = np.ones(n)/n
        res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=cons)
        return self._result(res.x, "MinVariance")

    def max_sharpe(self) -> PortfolioOptimizationResultV3:
        n = len(self.assets)
        def neg_sharpe(w):
            ret = np.sum(w * self.mean)
            vol = np.sqrt(w @ self.cov @ w)
            return -(ret - self.rf) / vol if vol > 0 else 0
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(n)]
        w0 = np.ones(n)/n
        res = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=cons)
        return self._result(res.x, "MaxSharpe")

    def kelly_portfolio(self) -> PortfolioOptimizationResultV3:
        # Simplified: Kelly = (mean - rf) / variance
        variance = np.diag(self.cov)
        kelly_weights = np.maximum(0, (self.mean - self.rf) / variance)
        kelly_weights /= kelly_weights.sum()
        return self._result(kelly_weights, "Kelly")