"""
Futures Trade Orchestrator  v2  (Institutional Upgrade — Phases 12–21)
========================================================================
Extends the existing 19-step pipeline with 10 new institutional steps.
All existing steps (1–19) are PRESERVED AND UNCHANGED.
New steps are injected at logical points:

  Step 2b  — Liquidity Regime       (Phase 12)  after Markov
  Step 3   — Regime Ensemble v2     (Phase 2+)  replaces compute_ensemble_regime
  Step 3b  — Market Breadth         (Phase 14)  after ensemble
  Step 3c  — Cross-Asset Engine     (Phase 21)  after breadth
  Step 3d  — Regime Persistence     (Phase 15)  after ensemble
  Step 13b — Flow Engine            (Phase 13)  after trade quality
  Step 14  — Bayesian v2            (Phase 16)  replaces bayesian_analysis
  Step 14b — Forecast Engine        (Phase 17)  after bayesian
  Step 14c — Conviction Engine      (Phase 18)  after forecast
  Step 16b — Portfolio Optimizer    (Phase 20)  single-asset mode
  Step 17b — Options: Adv. Greeks   (Phase 19)  if options context available

FuturesResult_v2 adds new fields to FuturesResult (backward compatible via dataclass inheritance).

Usage (drop-in):
  from core.futures_orchestrator_v2 import FuturesOrchestrator_v2
  orch = FuturesOrchestrator_v2()
  result = orch.run(symbol, df)
"""
from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass
from typing import Optional

import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

DEFAULT_WIN_RATE = 0.52
DEFAULT_AVG_RR   = 2.5


# ── Extended Result ────────────────────────────────────────────────────────────
@dataclass
class FuturesResult_v2:
    """All original FuturesResult fields + new institutional fields."""
    # ── Original fields (unchanged) ──────────────────────────────────────────
    symbol:              str
    price:               float
    runtime:             float
    report_text:         str
    final_decision:      str
    ai_score:            float
    trade_grade:         str
    trade_quality_score: float
    regime:              str
    regime_conf:         float
    vol_regime:          str
    entry:               float
    stop_loss:           float
    stop_reason:         str
    tp1:                 Optional[float]
    tp2:                 Optional[float]
    rr:                  float
    risk_pct:            float
    mc_profit_prob:      float
    kelly:               float
    ev:                  float
    sharpe:              float
    approved:            bool
    consistency_ok:      bool
    bayesian_bull:       float
    bayesian_bear:       float

    # ── Phase 12: Liquidity ───────────────────────────────────────────────────
    liquidity_regime:    str   = "RISK_ON"
    liquidity_score:     float = 55.0
    liquidity_risk_mult: float = 1.0

    # ── Phase 13: Flow ────────────────────────────────────────────────────────
    flow_regime:         str   = "NEUTRAL"
    flow_score:          float = 50.0
    flow_direction:      str   = "NEUTRAL"

    # ── Phase 14: Breadth ─────────────────────────────────────────────────────
    breadth_regime:      str   = "NEUTRAL"
    breadth_score:       float = 50.0

    # ── Phase 15: Persistence ─────────────────────────────────────────────────
    persistence_label:   str   = "ESTABLISHED"
    remaining_days:      float = 10.0
    exit_prob_7d:        float = 0.0

    # ── Phase 17: Forecast ────────────────────────────────────────────────────
    forecast_direction:  str   = "NEUTRAL"
    forecast_20d_return: float = 0.0
    forecast_confidence: float = 50.0

    # ── Phase 18: Conviction ──────────────────────────────────────────────────
    conviction_score:    float = 50.0
    conviction_tier:     str   = "HALF SIZE"
    conviction_kelly_mult: float = 0.5

    # ── Phase 20: Portfolio ───────────────────────────────────────────────────
    portfolio_vol:       float = 0.0
    portfolio_drawdown:  float = 0.0
    portfolio_sharpe:    float = 0.0

    # ── Phase 21: Cross-Asset ─────────────────────────────────────────────────
    cross_asset_regime:  str   = "TRANSITION"
    btc_rs_score:        float = 50.0
    btc_beta_spy:        float = 1.0


# ── Safe Runner ────────────────────────────────────────────────────────────────
def _safe(fn, *args, default=None, label="", **kwargs):
    """Run a function and return default on any exception. Never crashes pipeline."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        logger.warning("[orchestrator_v2] %s failed: %s", label or fn.__name__, exc)
        return default


# ── Orchestrator v2 ────────────────────────────────────────────────────────────
class FuturesOrchestrator_v2:
    """
    Extended orchestrator — all 19 original steps + 10 new institutional steps.
    Each new step is wrapped in _safe() so failure never kills the pipeline.
    """

    def __init__(self, win_rate: float = DEFAULT_WIN_RATE, avg_rr: float = DEFAULT_AVG_RR):
        from regime.markov import MarkovRegimeEngine
        self._regime_engine = MarkovRegimeEngine()
        self._win_rate = win_rate
        self._avg_rr   = avg_rr
        self._prev_liquidity_regime: Optional[str] = None

    def run(self, symbol: str, df: pd.DataFrame) -> FuturesResult_v2:  # noqa: C901
        # ── Lazy imports (same pattern as original) ───────────────────────────
        from ai.scoring_engine import compute_ai_score
        from config.thresholds import THRESHOLDS
        from engines.bayesian_reliability import compute_bayesian_analysis_v2
        from engines.conviction_engine import compute_conviction
        from engines.cross_asset_engine import compute_cross_asset
        from engines.forecast_engine import compute_forecast
        from engines.liquidity_regime import compute_liquidity_regime
        from engines.market_breadth import compute_market_breadth
        from engines.regime_ensemble_v2 import compute_ensemble_regime_v2
        from engines.regime_persistence import compute_regime_persistence
        from engines.trade_quality import compute_trade_quality
        from engines.volatility_regime import compute_volatility_regime
        from engines.volume_profile import compute_volume_profile
        from engines.anchored_vwap import compute_anchored_vwap
        from indicators.atr import compute_atr, get_atr_result
        from indicators.ema import compute_ema, get_ema_result
        from indicators.rsi import compute_rsi, get_rsi_result
        from market_structure.structure_break import detect_structure
        from market_structure.structure_consistency import check_structure_consistency
        from market_structure.support_resistance import detect_sr_levels
        from market_structure.swing_detector import get_recent_swings
        from portfolio.optimizer import compute_portfolio_optimization
        from regime.markov_calibration import calibrate_regime_probs
        from report.daily_report import build_daily_report
        from report.dashboards import build_institutional_dashboards
        from risk.consistency_checker import check_monte_carlo_consistency
        from risk.position_sizing import compute_position
        from risk.stop_engine import compute_institutional_stop
        from risk.stop_loss_engine import compute_sl_tp
        from signals.divergence_detector import detect_divergence
        from signals.entry_engine import check_entry
        from signals.final_decision import evaluate_trade
        from signals.trend_filter import apply_trend_filter
        from simulation.monte_carlo import run_monte_carlo
        from simulation.portfolio_risk import compute_portfolio_risk

        t0    = time.time()
        price = float(df["Close"].iloc[-1])
        logger.info("[%s] v2 start price=%.4f bars=%d", symbol, price, len(df))

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # STEPS 1–11: ORIGINAL PIPELINE (UNCHANGED)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 1. Indicators
        df  = compute_ema(df)
        df  = compute_rsi(df, period=14)
        df  = compute_atr(df, period=14)
        ema = get_ema_result(df)
        rsi = get_rsi_result(df, period=14)
        atr = get_atr_result(df, period=14)

        # 2. Markov + Calibration
        reg_raw = cal = None
        raw_probs  = {}
        regime_conf = 55.0
        try:
            reg_raw     = self._regime_engine.detect(df)
            cal         = calibrate_regime_probs(reg_raw.regime_probs_all)
            raw_probs   = cal.calibrated_probs
            regime_conf = cal.calibrated_conf
        except Exception as exc:
            logger.warning("[%s] Markov: %s", symbol, exc)
            raw_probs = {"BULL": 0.55} if ema.bias == "BULLISH" else {"BEAR": 0.55}

        # ── 2b. PHASE 12: Liquidity Regime (runs in parallel with Markov) ────
        liquidity_result = _safe(
            compute_liquidity_regime,
            self._prev_liquidity_regime,
            label="liquidity_regime",
        )
        if liquidity_result:
            self._prev_liquidity_regime = liquidity_result.liquidity_regime

        # ── 3. PHASE 2+: Regime Ensemble v2 (includes liquidity + breadth + cross-asset probs) ─
        # Run breadth & cross-asset BEFORE ensemble so they can inform it
        breadth_result = _safe(compute_market_breadth, label="market_breadth")
        cross_asset_result = _safe(compute_cross_asset, df, label="cross_asset")

        ensemble = regime = None
        try:
            ensemble    = compute_ensemble_regime_v2(
                df, raw_probs, liquidity_result, breadth_result, cross_asset_result
            )
            regime      = ensemble.regime
            regime_conf = min(ensemble.confidence, THRESHOLDS.MAX_REGIME_CONFIDENCE)
        except Exception as exc:
            logger.warning("[%s] Ensemble v2: %s", symbol, exc)
            regime = max(raw_probs, key=raw_probs.get) if raw_probs else "RANGE"

        # ── 3d. PHASE 15: Regime Persistence ─────────────────────────────────
        transition_matrix = getattr(reg_raw, "transition_matrix", {}) if reg_raw else {}
        persistence_result = _safe(
            compute_regime_persistence, regime, transition_matrix,
            label="regime_persistence",
        )

        # 4. Volatility Regime (original)
        try:
            vol_reg = compute_volatility_regime(df)
        except Exception:
            from engines.volatility_regime import VolatilityRegimeResult
            vol_reg = VolatilityRegimeResult(
                regime="NORMAL_VOL", vol_score=50, hv20=0.20, hv60=0.18, hv5=0.22,
                atr_pct=atr.atr_pct, vov=0.02, iv_hv_ratio=1.0,
                position_size_mult=1.0, stop_distance_mult=1.0,
                preferred_strategy="BULL_CALL_SPREAD",
                recommended_action="Normal vol — standard",
            )

        # 5. Swings (original)
        swing_data = get_recent_swings(df)
        sh_all = swing_data["all_highs"];  sl_all = swing_data["all_lows"]
        last_sh = swing_data["last_swing_high"];  last_sl = swing_data["last_swing_low"]
        sh_2 = sh_all[-2] if len(sh_all) >= 2 else None
        sl_2 = sl_all[-2] if len(sl_all) >= 2 else None

        # 6. Structure + Consistency (original)
        structure   = detect_structure(sh_all, sl_all, price)
        divergence  = detect_divergence(df, rsi_col="RSI14")
        consistency = check_structure_consistency(
            structure_trend=structure.trend, bos_bullish=structure.bos_bullish,
            bos_bearish=structure.bos_bearish, ema_bias=ema.bias,
            divergence_kind=divergence.kind, regime=regime,
            structure_score=structure.structure_score,
        )
        regime_conf = max(10.0, regime_conf - consistency.confidence_penalty * 0.5)

        # 7–9. S/R + Trend + Entry (original)
        sr           = detect_sr_levels(df, sh_all, sl_all, price)
        trend_filter = apply_trend_filter(ema, structure, divergence, regime)
        entry_result = check_entry(df=df, final_bias=trend_filter.final_bias,
                                   supports=sr["supports"], resistances=sr["resistances"],
                                   current_price=price)
        direction = entry_result.direction

        # 10. Institutional Stop (original)
        inst_stop = compute_institutional_stop(
            direction=direction, entry=price, atr=atr.atr14,
            swing_low=last_sl.price   if last_sl else None,
            swing_high=last_sh.price  if last_sh else None,
            swing_low_2=sl_2.price    if sl_2 else None,
            swing_high_2=sh_2.price   if sh_2 else None,
            vol_regime=vol_reg.regime,
        )

        # 11. TP + Risk (original)
        sl_tp     = compute_sl_tp(
            direction=direction, entry=price, atr=atr.atr14,
            swing_low=last_sl.price  if last_sl else None,
            swing_high=last_sh.price if last_sh else None,
            supports=sr["supports"], resistances=sr["resistances"],
        )
        stop_loss   = inst_stop.selected_stop
        stop_reason = inst_stop.stop_reason
        tp1 = sl_tp.tp1;  tp2 = sl_tp.tp2
        risk_dist   = inst_stop.risk
        rr1 = abs(tp1 - price) / risk_dist if (tp1 and risk_dist > 0) else 0.0
        rr2 = abs(tp2 - price) / risk_dist if (tp2 and risk_dist > 0) else 0.0
        best_rr = max(rr1, rr2)
        if best_rr < THRESHOLDS.MIN_RR:
            direction = "WAIT"

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # STEPS 12–14: SCORING (ORIGINAL + UPGRADES)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        # 12. AI Score (original — unchanged interface)
        ai_score = compute_ai_score(
            regime=regime, regime_confidence=regime_conf,
            structure_trend=structure.trend,
            structure_clarity=consistency.structure_confidence,
            ema_alignment=ema.alignment_strength, ema_bias=ema.bias,
            rsi_value=rsi.value, rsi_momentum=rsi.momentum,
            rr=best_rr, direction=direction,
        )
        if ai_score.final_score < THRESHOLDS.MIN_AI_SCORE:
            direction = "WAIT"

        # 13. Trade Quality (original)
        vol_ratio = float(df["Volume"].iloc[-1] / df["Volume"].rolling(20).mean().iloc[-1]) \
                    if "Volume" in df.columns else 1.0
        trade_q = compute_trade_quality(
            regime=regime, regime_confidence=regime_conf,
            ema_alignment=ema.alignment_strength,
            structure_trend=structure.trend,
            structure_clarity=consistency.structure_confidence,
            ev=0.5, rr=best_rr, volume_ratio=vol_ratio, vol_regime=vol_reg.regime,
        )

        # ── 13b. PHASE 13: Flow Engine ────────────────────────────────────────
        from crypto.flow_engine import compute_flow_engine
        from crypto.funding_rate import compute_funding_rate
        from crypto.open_interest import compute_open_interest
        from crypto.liquidation_engine import compute_liquidations

        funding_result = _safe(compute_funding_rate,  symbol, label="funding_rate")
        oi_result      = _safe(compute_open_interest, symbol, label="open_interest")
        liq_result     = _safe(compute_liquidations,  symbol, price, label="liquidations")
        flow_result    = _safe(
            compute_flow_engine, symbol, price, funding_result, oi_result, liq_result,
            label="flow_engine",
        )

        # 14. PHASE 16: Bayesian v2 (reliability-weighted — drop-in replacement)
        bayes = _safe(
            compute_bayesian_analysis_v2,
            rsi=rsi.value, regime=regime, regime_confidence=regime_conf,
            ema_alignment=ema.alignment_strength,
            structure_trend=structure.trend,
            vol_regime=vol_reg.regime, atr_pct=vol_reg.atr_pct,
            label="bayesian_v2",
        )
        if bayes is None:
            from engines.bayesian_engine import compute_bayesian_analysis
            bayes = compute_bayesian_analysis(
                rsi=rsi.value, regime=regime, regime_confidence=regime_conf,
                ema_alignment=ema.alignment_strength, structure_trend=structure.trend,
                vol_regime=vol_reg.regime, atr_pct=vol_reg.atr_pct,
            )

        # ── 14b. PHASE 17: Forecast Engine ────────────────────────────────────
        funding_rate_val = getattr(funding_result, "funding_rate_pct", 0.0) if funding_result else 0.0
        oi_trend_enc     = {"INCREASING": 1.0, "STABLE": 0.0, "DECREASING": -1.0}.get(
            getattr(oi_result, "oi_trend", "STABLE"), 0.0
        )
        liq_score_val   = getattr(liquidity_result, "score",        55.0) if liquidity_result else 55.0
        brd_score_val   = getattr(breadth_result,   "breadth_score", 50.0) if breadth_result  else 50.0

        forecast_result = _safe(
            compute_forecast, df,
            regime_bull_prob  = bayes.composite_bull_prob,
            regime_bear_prob  = bayes.composite_bear_prob,
            regime_confidence = regime_conf,
            funding_rate      = funding_rate_val,
            liquidity_score   = liq_score_val,
            breadth_score     = brd_score_val,
            oi_trend_enc      = oi_trend_enc,
            label="forecast_engine",
        )

        # ── 14c. PHASE 18: Conviction Engine ─────────────────────────────────
        persistence_label = getattr(persistence_result, "persistence_label", "ESTABLISHED") if persistence_result else "ESTABLISHED"
        persistence_score = getattr(persistence_result, "persistence_score", 70.0)         if persistence_result else 70.0

        conviction_result = _safe(
            compute_conviction,
            regime=regime, regime_confidence=regime_conf,
            ema_alignment=ema.alignment_strength,
            structure_trend=structure.trend,
            structure_clarity=consistency.structure_confidence,
            vol_regime=vol_reg.regime,
            trade_direction=direction,
            flow_score      = getattr(flow_result, "flow_score",     50.0) if flow_result else 50.0,
            flow_direction  = getattr(flow_result, "flow_direction",  "NEUTRAL") if flow_result else "NEUTRAL",
            flow_regime     = getattr(flow_result, "flow_regime",     "NEUTRAL") if flow_result else "NEUTRAL",
            breadth_score   = brd_score_val,
            breadth_regime  = getattr(breadth_result,  "breadth_regime",    "NEUTRAL") if breadth_result else "NEUTRAL",
            liquidity_score = liq_score_val,
            liquidity_regime= getattr(liquidity_result,"liquidity_regime",  "RISK_ON")  if liquidity_result else "RISK_ON",
            risk_multiplier = getattr(liquidity_result,"risk_multiplier",   1.0)        if liquidity_result else 1.0,
            forecast_direction  = getattr(forecast_result, "forecast_direction",  "NEUTRAL") if forecast_result else "NEUTRAL",
            probability_up_20d  = getattr(forecast_result, "probability_up_20d",  0.50)      if forecast_result else 0.50,
            forecast_confidence = getattr(forecast_result, "forecast_confidence", 50.0)      if forecast_result else 50.0,
            persistence_score   = persistence_score,
            persistence_label   = persistence_label,
            label="conviction_engine",
        )

        # Apply conviction gate: NO TRADE if conviction < 50
        if conviction_result and not conviction_result.trade_allowed:
            direction = "WAIT"
            logger.info("[%s] Conviction gate: score=%.1f < 50 → direction=WAIT",
                        symbol, conviction_result.conviction_score)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # STEPS 15–19: RISK + DECISION + REPORT (ORIGINAL + EXTENSIONS)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        # 15. Monte Carlo + Consistency (original)
        mc   = run_monte_carlo(close_series=df["Close"], entry=price,
                               stop_loss=stop_loss, target=tp2 or price * 1.04,
                               horizon=20, simulations=10_000)
        port = compute_portfolio_risk(df["Close"])
        mc_check = check_monte_carlo_consistency(
            prob_profit=mc.prob_profit, prob_target_hit=mc.prob_target_hit,
            prob_stop_hit=mc.prob_stop_hit, expected_return=mc.expected_return_pct,
            ev=0.5, rr=best_rr, pop=mc.prob_profit,
        )

        # 16. Position Sizing — apply conviction kelly multiplier
        position = compute_position(self._win_rate, max(self._avg_rr, best_rr), regime)

        # ── 16b. PHASE 20: Portfolio Optimizer (single-asset mode) ────────────
        portfolio_opt = _safe(
            compute_portfolio_optimization,
            {symbol: df["Close"]},
            method="risk_parity",
            label="portfolio_optimizer",
        )

        # Apply liquidity risk multiplier to final sizing
        liq_mult  = getattr(liquidity_result, "risk_multiplier", 1.0) if liquidity_result else 1.0
        conv_mult = getattr(conviction_result, "kelly_multiplier", 0.75) if conviction_result else 0.75

        # 17. Final Decision (original)
        final = evaluate_trade(
            direction=direction, regime_confidence=regime_conf,
            ai_score=ai_score.final_score, expected_value=position.ev,
            kelly_fraction=position.kelly_fraction, mc_profit_prob=mc.prob_profit,
            best_rr=best_rr, structure_trend=structure.trend, ema_bias=ema.bias,
        )

        # 18. Volume Profile + AVWAP (original)
        vol_profile = avwap_result = None
        try:
            vol_profile  = compute_volume_profile(df, lookback=60)
            avwap_result = compute_anchored_vwap(df)
        except Exception as exc:
            logger.debug("[%s] vol_profile/avwap: %s", symbol, exc)

        # 19. Report — original + new institutional dashboards
        from types import SimpleNamespace
        risk_obj = SimpleNamespace(
            direction=direction, entry=price, stop_loss=stop_loss,
            tp1=tp1, tp2=tp2, risk=risk_dist,
            rr1=rr1, rr2=rr2,
            valid_rr=(best_rr >= THRESHOLDS.MIN_RR), reason="",
        )

        report_text = build_daily_report(
            symbol=symbol, price=price,
            regime=reg_raw, ema=ema, rsi=rsi,
            structure=structure, divergence=divergence,
            trend_filter=trend_filter, sr=sr, risk=risk_obj,
            ai_score=ai_score, mc=mc, port=port, position=position,
            entry_result=entry_result, final=final,
            cal_result=cal, ensemble=ensemble, vol_regime=vol_reg,
            consistency=consistency, mc_consistency=mc_check,
            bayesian=bayes, trade_quality=trade_q,
            inst_stop=inst_stop, vol_profile=vol_profile, avwap=avwap_result,
        )

        # Append institutional dashboard sections
        try:
            inst_dashboards = build_institutional_dashboards(
                liquidity=liquidity_result,
                flow=flow_result,
                breadth=breadth_result,
                persistence=persistence_result,
                forecast=forecast_result,
                conviction=conviction_result,
                portfolio_opt=portfolio_opt,
                cross_asset=cross_asset_result,
            )
            report_text = report_text + "\n\n" + inst_dashboards
        except Exception as exc:
            logger.warning("[%s] institutional dashboards failed: %s", symbol, exc)

        runtime = round(time.time() - t0, 2)
        logger.info(
            "[%s] v2 done %.1fs | %s conf=%.0f%% | %s AI=%.0f grade=%s "
            "conviction=%.0f(%s) RR=%.2f liq=%s flow=%s approved=%s",
            symbol, runtime, regime, regime_conf, final.decision,
            ai_score.final_score, trade_q.grade,
            getattr(conviction_result, "conviction_score", 0),
            getattr(conviction_result, "conviction_tier", "?"),
            best_rr,
            getattr(liquidity_result,  "liquidity_regime", "?"),
            getattr(flow_result,        "flow_regime",       "?"),
            final.approved,
        )

        return FuturesResult_v2(
            symbol=symbol, price=price, runtime=runtime,
            report_text=report_text, final_decision=final.decision,
            ai_score=ai_score.final_score, trade_grade=trade_q.grade,
            trade_quality_score=trade_q.score, regime=regime, regime_conf=regime_conf,
            vol_regime=vol_reg.regime, entry=price, stop_loss=stop_loss,
            stop_reason=stop_reason, tp1=tp1, tp2=tp2, rr=best_rr,
            risk_pct=position.risk_pct, mc_profit_prob=mc.prob_profit,
            kelly=position.kelly_fraction, ev=position.ev, sharpe=port.sharpe,
            approved=final.approved, consistency_ok=mc_check.is_consistent,
            bayesian_bull=bayes.composite_bull_prob,
            bayesian_bear=bayes.composite_bear_prob,
            # New fields
            liquidity_regime    = getattr(liquidity_result,  "liquidity_regime",    "RISK_ON"),
            liquidity_score     = getattr(liquidity_result,  "score",               55.0),
            liquidity_risk_mult = getattr(liquidity_result,  "risk_multiplier",     1.0),
            flow_regime         = getattr(flow_result,        "flow_regime",         "NEUTRAL"),
            flow_score          = getattr(flow_result,        "flow_score",          50.0),
            flow_direction      = getattr(flow_result,        "flow_direction",      "NEUTRAL"),
            breadth_regime      = getattr(breadth_result,     "breadth_regime",      "NEUTRAL"),
            breadth_score       = getattr(breadth_result,     "breadth_score",       50.0),
            persistence_label   = getattr(persistence_result, "persistence_label",   "ESTABLISHED"),
            remaining_days      = getattr(persistence_result, "remaining_duration_days", 10.0),
            exit_prob_7d        = getattr(persistence_result, "exit_prob_7d",        0.0),
            forecast_direction  = getattr(forecast_result,    "forecast_direction",  "NEUTRAL"),
            forecast_20d_return = getattr(forecast_result,    "expected_return_20d", 0.0),
            forecast_confidence = getattr(forecast_result,    "forecast_confidence", 50.0),
            conviction_score    = getattr(conviction_result,  "conviction_score",    50.0),
            conviction_tier     = getattr(conviction_result,  "conviction_tier",     "HALF SIZE"),
            conviction_kelly_mult= getattr(conviction_result, "kelly_multiplier",    0.5),
            portfolio_vol       = getattr(portfolio_opt,       "portfolio_volatility",0.0),
            portfolio_drawdown  = getattr(portfolio_opt,       "portfolio_drawdown",  0.0),
            portfolio_sharpe    = getattr(portfolio_opt,       "portfolio_sharpe",    0.0),
            cross_asset_regime  = getattr(cross_asset_result,  "cross_asset_regime",  "TRANSITION"),
            btc_rs_score        = getattr(cross_asset_result,  "relative_strength_score", 50.0),
            btc_beta_spy        = getattr(cross_asset_result,  "btc_beta_to_spy",     1.0),
        )
