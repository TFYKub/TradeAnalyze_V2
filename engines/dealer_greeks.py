"""
Dealer Gamma Exposure, Vanna, Charm, Pin‑Risk
"""
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class DealerGreeksResult:
    total_gamma: float
    gamma_concentration: Dict[float, float]
    dealer_vanna: float
    dealer_charm: float
    pin_risk_strike: Optional[float]
    pin_risk_severity: float
    interpretation: str

class DealerGreeksEngine:
    def __init__(self, enriched_chain: List[dict], spot: float):
        self.chain = enriched_chain
        self.spot = spot

    def compute(self) -> DealerGreeksResult:
        total_gamma = 0.0
        gamma_by_strike = {}
        total_vanna = 0.0
        total_charm = 0.0
        for row in self.chain:
            if not row.get('open_interest') or row['open_interest'] == 0:
                continue
            gamma = row.get('gamma', 0)
            vanna = row.get('vanna', 0)
            charm = row.get('charm', 0)
            oi = row['open_interest']
            # Dealer gamma = - (option gamma * OI) because dealers are short options
            dealer_gamma = -gamma * oi
            total_gamma += dealer_gamma
            strike = row['strike']
            gamma_by_strike[strike] = gamma_by_strike.get(strike, 0) + dealer_gamma
            total_vanna += -vanna * oi
            total_charm += -charm * oi

        # Pin risk: strike with highest absolute gamma concentration
        if gamma_by_strike:
            pin_strike = max(gamma_by_strike, key=lambda k: abs(gamma_by_strike[k]))
            severity = min(100, abs(gamma_by_strike[pin_strike]) / (abs(total_gamma) + 1e-6) * 100)
        else:
            pin_strike = None
            severity = 0

        return DealerGreeksResult(
            total_gamma=total_gamma,
            gamma_concentration=gamma_by_strike,
            dealer_vanna=total_vanna,
            dealer_charm=total_charm,
            pin_risk_strike=pin_strike,
            pin_risk_severity=severity,
            interpretation=f"Total gamma {total_gamma:.0f}, pin risk at {pin_strike} severity {severity:.0f}%"
        )