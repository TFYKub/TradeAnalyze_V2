"""
Crypto Flow Engine V2 – OI delta, funding momentum, liquidation pressure, whale flow
"""
import requests
from typing import Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class CryptoFlowV2Result:
    flow_score: float
    flow_regime: str   # BULLISH, BEARISH, SQUEEZE, NEUTRAL
    oi_delta: float
    funding_momentum: float
    liquidation_pressure: float
    whale_flow_ratio: float
    interpretation: str

class CryptoFlowV2:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.base = symbol.split('-')[0]  # BTC, ETH

    def fetch_oi(self) -> float:
        # Deribit example
        try:
            resp = requests.get(f"https://www.deribit.com/api/v2/public/get_open_interest?currency={self.base}")
            return resp.json()['result']['open_interest']
        except: return 0

    def fetch_oi_delta(self) -> float:
        # store previous OI in a cache
        # dummy implementation
        return 0.0

    def fetch_funding_rate(self) -> float:
        # returns current 8h rate
        try:
            resp = requests.get(f"https://www.deribit.com/api/v2/public/ticker?instrument_name={self.base}_PERPETUAL")
            return resp.json()['result']['current_funding']
        except: return 0

    def fetch_funding_momentum(self) -> float:
        # (current - previous) / interval
        return 0.0

    def fetch_liquidations(self) -> float:
        # total liquidations / volume
        return 0.0

    def fetch_whale_flow(self) -> float:
        # large taker orders / total volume
        return 0.0

    def compute(self) -> CryptoFlowV2Result:
        oi_delta = self.fetch_oi_delta()
        funding_mom = self.fetch_funding_momentum()
        liq_pressure = self.fetch_liquidations()
        whale = self.fetch_whale_flow()
        # Combine into score (0-100)
        score = 50 + (oi_delta * 10) + (funding_mom * 100) - (liq_pressure * 50) + (whale * 20)
        score = max(0, min(100, score))
        if score > 70:
            regime = "BULLISH"
        elif score < 30:
            regime = "BEARISH"
        else:
            regime = "NEUTRAL"
        return CryptoFlowV2Result(
            flow_score=score, flow_regime=regime,
            oi_delta=oi_delta, funding_momentum=funding_mom,
            liquidation_pressure=liq_pressure, whale_flow_ratio=whale,
            interpretation=f"Flow score {score:.0f} – {regime}"
        )