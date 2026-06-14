"""
Futures Orchestrator V3 – Extends V2 with all new phases
"""
import pandas as pd
from datetime import datetime
from core.futures_orchestrator_v2 import FuturesOrchestrator_v2, FuturesResult_v2
from core.trade_state import TradeStateV3
from engines.signal_tracker import SignalDatabase, SignalRecord
from engines.adaptive_ensemble import AdaptiveEnsemble
from engines.regime_switching_mc import RegimeSwitchingMC
from config.config import V3_PHASES

class FuturesOrchestrator_v3(FuturesOrchestrator_v2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.signal_db = SignalDatabase()
        self.adaptive_ensemble = AdaptiveEnsemble(self.signal_db) if V3_PHASES.get("walk_forward", True) else None
        self.regime_mc = RegimeSwitchingMC() if V3_PHASES.get("regime_switching_mc", True) else None

    def run(self, symbol: str, df: pd.DataFrame) -> FuturesResult_v2:
        # Run V2 pipeline
        result_v2 = super().run(symbol, df)

        # Create V3 state
        state = TradeStateV3()
        current_price = float(df['Close'].iloc[-1])

        # Phase 1: Signal tracking
        record = SignalRecord(
            timestamp=datetime.now(),
            symbol=symbol,
            engine="ensemble",
            predicted_direction=result_v2.final_decision,
            confidence=result_v2.ai_score / 100.0
        )
        self.signal_db.insert(record)

        # Phase 3: Regime-switching MC (if trade is active)
        if self.regime_mc and result_v2.final_decision in ("LONG", "SHORT"):
            # Get regime probabilities (simplified)
            regime_probs = {"BULL": 0.6, "BEAR": 0.2, "RANGE": 0.2}
            regime_vols = {"BULL": 0.3, "BEAR": 0.5, "RANGE": 0.2}
            mc_result = self.regime_mc.run(
                current_price, result_v2.entry, result_v2.stop_loss,
                regime_probs, regime_vols
            )
            state.regime_mc_result = {
                "prob_profit": mc_result.prob_profit,
                "prob_stop_hit": mc_result.prob_stop_hit,
                "expected_return": mc_result.expected_return,
                "var_95": mc_result.var_95,
                "cvar_95": mc_result.cvar_95,
                "regime_weights": mc_result.regime_weights
            }

        # Attach state to result_v2 (as a custom attribute)
        result_v2.v3_state = state
        return result_v2