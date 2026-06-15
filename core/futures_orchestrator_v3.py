# core/futures_orchestrator_v3.py
"""
Futures Orchestrator V3 – Extends V2 with all new phases + State Persistence
"""
import pandas as pd
from datetime import datetime, timezone
from core.futures_orchestrator_v2 import FuturesOrchestrator_v2, FuturesResult_v2
from core.trade_state import TradeStateV3
from engines.signal_tracker import SignalDatabase, SignalRecord
from engines.adaptive_ensemble import AdaptiveEnsemble
from engines.regime_switching_mc import RegimeSwitchingMC
from config.config import V3_PHASES
from persistence.trade_persistence import get_persistence, ActiveTrade
import json


class FuturesOrchestrator_v3(FuturesOrchestrator_v2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.signal_db = SignalDatabase()
        self.adaptive_ensemble = AdaptiveEnsemble(self.signal_db) if V3_PHASES.get("walk_forward", True) else None
        self.regime_mc = RegimeSwitchingMC() if V3_PHASES.get("regime_switching_mc", True) else None
        self.persistence = get_persistence()

    def run(self, symbol: str, df: pd.DataFrame) -> FuturesResult_v2:
        # Check if there is already an active trade for this symbol (reconciliation)
        if self.persistence.has_active_trade(symbol):
            logger.info("[%s] Active trade already exists – skipping new entry (position reconciliation)", symbol)
            # Return a dummy result that indicates NO_TRADE (or we could load existing state)
            # For simplicity, we return a NO_TRADE result.
            # In production, you might want to load the existing trade state.
            # Here we create a minimal result with NO_TRADE.
            from core.futures_orchestrator_v2 import FuturesResult_v2
            price = float(df["Close"].iloc[-1])
            dummy = FuturesResult_v2(
                symbol=symbol, price=price, runtime=0.0,
                report_text=f"Active trade already open for {symbol} – no new trade",
                final_decision="NO_TRADE", ai_score=0.0, trade_grade="N/A",
                trade_quality_score=0.0, regime="UNKNOWN", regime_conf=0.0,
                vol_regime="NORMAL_VOL", entry=price, stop_loss=price,
                stop_reason="Position reconciliation", tp1=None, tp2=None,
                rr=0.0, risk_pct=0.0, mc_profit_prob=0.0, kelly=0.0,
                ev=0.0, sharpe=0.0, approved=False, consistency_ok=False,
                bayesian_bull=0.5, bayesian_bear=0.5,
                # v2 extra fields (defaults)
                liquidity_regime="RISK_ON", liquidity_score=55.0, liquidity_risk_mult=1.0,
                flow_regime="NEUTRAL", flow_score=50.0, flow_direction="NEUTRAL",
                breadth_regime="NEUTRAL", breadth_score=50.0,
                persistence_label="ESTABLISHED", remaining_days=10.0, exit_prob_7d=0.0,
                forecast_direction="NEUTRAL", forecast_20d_return=0.0, forecast_confidence=50.0,
                conviction_score=50.0, conviction_tier="HALF SIZE", conviction_kelly_mult=0.5,
                portfolio_vol=0.0, portfolio_drawdown=0.0, portfolio_sharpe=0.0,
                cross_asset_regime="TRANSITION", btc_rs_score=50.0, btc_beta_spy=1.0,
            )
            return dummy

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

        # ----- PERSISTENCE: Save active trade if approved -----
        if result_v2.approved and result_v2.final_decision in ("LONG", "SHORT"):
            # Build snapshot of key state at entry
            snapshot = {
                "regime": result_v2.regime,
                "regime_conf": result_v2.regime_conf,
                "ai_score": result_v2.ai_score,
                "trade_grade": result_v2.trade_grade,
                "conviction_score": result_v2.conviction_score,
                "liquidity_regime": result_v2.liquidity_regime,
                "flow_regime": result_v2.flow_regime,
                "breadth_regime": result_v2.breadth_regime,
            }
            trade_id = f"{symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            active = ActiveTrade(
                symbol=symbol,
                direction=result_v2.final_decision,
                entry_price=result_v2.entry,
                stop_loss=result_v2.stop_loss,
                tp1=result_v2.tp1,
                tp2=result_v2.tp2,
                position_size=result_v2.risk_pct,
                entry_time=datetime.now(timezone.utc).isoformat(),
                entry_snapshot=json.dumps(snapshot),
                trade_id=trade_id,
                last_updated=datetime.now(timezone.utc).isoformat(),
            )
            self.persistence.save_active_trade(active)

        # Attach state to result_v2 (as a custom attribute)
        result_v2.v3_state = state
        return result_v2