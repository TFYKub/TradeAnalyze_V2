"""
Regime‑Switching Monte Carlo (GBM with regime‑dependent volatility)
"""
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class RegimeMCResult:
    prob_profit: float
    prob_stop_hit: float
    expected_return: float
    var_95: float
    cvar_95: float
    regime_weights: Dict[str, float]

class RegimeSwitchingMC:
    def __init__(self, n_sims: int = 10000, horizon: int = 20):
        self.n_sims = n_sims
        self.horizon = horizon

    def run(self, current_price: float, entry: float, stop: float,
            regime_probs: Dict[str, float], regime_vols: Dict[str, float],
            drift: float = 0.0) -> RegimeMCResult:
        """
        regime_probs: e.g. {'BULL':0.6, 'BEAR':0.2, 'RANGE':0.2}
        regime_vols: {'BULL':0.3, 'BEAR':0.5, 'RANGE':0.2}  annualised
        """
        dt = 1/252
        all_prices = []
        for regime, prob in regime_probs.items():
            n = int(self.n_sims * prob)
            vol = regime_vols.get(regime, 0.3)
            if n <= 0: continue
            # simulate paths
            shocks = np.random.normal(0, vol * np.sqrt(dt), (n, self.horizon))
            log_returns = np.cumsum(shocks, axis=1)
            paths = current_price * np.exp(log_returns)
            all_prices.append(paths)
        if not all_prices:
            return RegimeMCResult(0,0,0,0,0,{})
        prices = np.vstack(all_prices)
        final = prices[:, -1]
        # profit: final > entry (for LONG) – we assume direction is LONG
        prob_profit = (final > entry).mean()
        # stop hit (any price <= stop)
        stop_hit = (prices.min(axis=1) <= stop).mean()
        exp_return = (final.mean() / current_price - 1) * 100
        var_95 = -np.percentile((final/current_price - 1), 5) * 100
        tail = (final/current_price - 1)[(final/current_price - 1) <= np.percentile((final/current_price - 1), 5)]
        cvar_95 = -np.mean(tail) * 100 if len(tail) > 0 else var_95
        return RegimeMCResult(
            prob_profit=prob_profit*100,
            prob_stop_hit=stop_hit*100,
            expected_return=exp_return,
            var_95=var_95,
            cvar_95=cvar_95,
            regime_weights=regime_probs
        )