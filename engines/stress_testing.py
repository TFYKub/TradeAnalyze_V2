"""
Stress Testing Framework – Historical and custom scenarios
"""
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class StressTestResult:
    scenario_name: str
    portfolio_loss_pct: float
    max_drawdown: float
    var_99_stress: float
    margin_call_prob: float
    survival_score: float   # 0-100

class StressTestEngine:
    SCENARIOS = {
        "2008_Crash": {"SPY": -0.38, "BTC": -0.60, "QQQ": -0.42, "GLD": 0.05, "TLT": 0.10},
        "2020_Covid": {"SPY": -0.34, "BTC": -0.50, "QQQ": -0.30, "GLD": -0.01, "TLT": 0.08},
        "2022_Bear": {"SPY": -0.25, "BTC": -0.64, "QQQ": -0.33, "GLD": -0.02, "TLT": -0.15},
        "Vol_Shock": {"vol_multiplier": 2.0},  # double volatility
        "Liquidity_Crisis": {"spread_multiplier": 5.0, "slippage": 0.10}
    }

    def run(self, portfolio_weights: Dict[str, float], current_prices: Dict[str, float],
            volatility: float) -> List[StressTestResult]:
        results = []
        for name, shock in self.SCENARIOS.items():
            if "vol_multiplier" in shock:
                # simplistic: VaR scales with vol
                var_shock = volatility * shock["vol_multiplier"] / volatility
                loss = var_shock * 2.33  # rough 99% VaR
                results.append(StressTestResult(
                    scenario_name=name, portfolio_loss_pct=loss, max_drawdown=loss,
                    var_99_stress=loss, margin_call_prob=0.5 if loss > 20 else 0.1,
                    survival_score=max(0, 100 - loss)
                ))
                continue
            # Apply asset shocks
            new_port_value = 0
            old_port_value = 0
            for asset, weight in portfolio_weights.items():
                shock_val = shock.get(asset, -0.2)  # default -20%
                old_price = current_prices.get(asset, 100)
                new_price = old_price * (1 + shock_val)
                old_port_value += weight * old_price
                new_port_value += weight * new_price
            loss_pct = (1 - new_port_value / old_port_value) * 100 if old_port_value > 0 else 0
            results.append(StressTestResult(
                scenario_name=name, portfolio_loss_pct=loss_pct, max_drawdown=loss_pct,
                var_99_stress=loss_pct, margin_call_prob=min(1.0, loss_pct / 30),
                survival_score=max(0, 100 - loss_pct)
            ))
        return results