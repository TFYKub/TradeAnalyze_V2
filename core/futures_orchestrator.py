"""
Futures Trade Orchestrator — Institutional Grade  (v2)
=======================================================
Full 19-step pipeline — all 11 phases integrated.
"""
from __future__ import annotations
import logging, time, warnings
from dataclasses import dataclass
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

DEFAULT_WIN_RATE = 0.52
DEFAULT_AVG_RR   = 2.5


@dataclass
class FuturesResult:
    symbol:             str
    price:              float
    runtime:            float
    report_text:        str
    final_decision:     str
    ai_score:           float
    trade_grade:        str
    trade_quality_score: float
    regime:             str
    regime_conf:        float
    vol_regime:         str
    entry:              float
    stop_loss:          float
    stop_reason:        str
    tp1:                float | None
    tp2:                float | None
    rr:                 float
    risk_pct:           float
    mc_profit_prob:     float
    kelly:              float
    ev:                 float
    sharpe:             float
    approved:           bool
    consistency_ok:     bool
    bayesian_bull:      float
    bayesian_bear:      float


class FuturesOrchestrator:

    def __init__(self, win_rate: float = DEFAULT_WIN_RATE, avg_rr: float = DEFAULT_AVG_RR):
        from regime.markov import MarkovRegimeEngine
        self._regime_engine = MarkovRegimeEngine()
        self._win_rate = win_rate
        self._avg_rr   = avg_rr

    def run(self, symbol: str, df: pd.DataFrame) -> FuturesResult:
        from ai.scoring_engine import compute_ai_score
        from config.thresholds import THRESHOLDS
        from engines.bayesian_engine import compute_bayesian_analysis
        from engines.regime_ensemble import compute_ensemble_regime
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
        from regime.markov_calibration import calibrate_regime_probs
        from report.daily_report import build_daily_report
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
        logger.info("[%s] start price=%.4f bars=%d", symbol, price, len(df))

        # ── 1. Indicators ──────────────────────────────────────────────────────
        df  = compute_ema(df)
        df  = compute_rsi(df, period=14)
        df  = compute_atr(df, period=14)
        ema = get_ema_result(df)
        rsi = get_rsi_result(df, period=14)
        atr = get_atr_result(df, period=14)

        # ── 2. Markov + Calibration ────────────────────────────────────────────
        reg_raw = cal = None
        raw_probs  = {}
        regime_conf = 55.0
        try:
            reg_raw = self._regime_engine.detect(df)
            cal     = calibrate_regime_probs(reg_raw.regime_probs_all)
            raw_probs   = cal.calibrated_probs
            regime_conf = cal.calibrated_conf
        except Exception as exc:
            logger.warning("[%s] Markov: %s", symbol, exc)
            raw_probs = {"BULL": 0.55} if ema.bias == "BULLISH" else {"BEAR": 0.55}

        # ── 3. Regime Ensemble ─────────────────────────────────────────────────
        ensemble = regime = None
        try:
            ensemble    = compute_ensemble_regime(df, raw_probs)
            regime      = ensemble.regime
            regime_conf = min(ensemble.confidence, THRESHOLDS.MAX_REGIME_CONFIDENCE)
        except Exception as exc:
            logger.warning("[%s] Ensemble: %s", symbol, exc)
            regime = max(raw_probs, key=raw_probs.get) if raw_probs else "RANGE"

        # ── 4. Volatility Regime ───────────────────────────────────────────────
        try:
            vol_reg = compute_volatility_regime(df)
        except Exception as exc:
            logger.debug("[%s] VolRegime: %s", symbol, exc)
            from engines.volatility_regime import VolatilityRegimeResult
            vol_reg = VolatilityRegimeResult(
                regime="NORMAL_VOL", vol_score=50, hv20=0.20, hv60=0.18, hv5=0.22,
                atr_pct=atr.atr_pct, vov=0.02, iv_hv_ratio=1.0,
                position_size_mult=1.0, stop_distance_mult=1.0,
                preferred_strategy="BULL_CALL_SPREAD",
                recommended_action="Normal vol — standard",
            )

        # ── 5. Swings ──────────────────────────────────────────────────────────
        swing_data = get_recent_swings(df)
        sh_all = swing_data["all_highs"];  sl_all = swing_data["all_lows"]
        last_sh = swing_data["last_swing_high"];  last_sl = swing_data["last_swing_low"]
        sh_2 = sh_all[-2] if len(sh_all) >= 2 else None
        sl_2 = sl_all[-2] if len(sl_all) >= 2 else None

        # ── 6. Structure + Consistency ─────────────────────────────────────────
        structure  = detect_structure(sh_all, sl_all, price)
        divergence = detect_divergence(df, rsi_col="RSI14")
        consistency = check_structure_consistency(
            structure_trend=structure.trend, bos_bullish=structure.bos_bullish,
            bos_bearish=structure.bos_bearish, ema_bias=ema.bias,
            divergence_kind=divergence.kind, regime=regime,
            structure_score=structure.structure_score,
        )
        regime_conf = max(10.0, regime_conf - consistency.confidence_penalty * 0.5)

        # ── 7. S/R ────────────────────────────────────────────────────────────
        sr = detect_sr_levels(df, sh_all, sl_all, price)

        # ── 8-9. Trend + Entry ────────────────────────────────────────────────
        trend_filter = apply_trend_filter(ema, structure, divergence, regime)
        entry_result = check_entry(df=df, final_bias=trend_filter.final_bias,
                                   supports=sr["supports"], resistances=sr["resistances"],
                                   current_price=price)
        direction = entry_result.direction

        # ── 10. Institutional Stop ────────────────────────────────────────────
        inst_stop = compute_institutional_stop(
            direction=direction, entry=price, atr=atr.atr14,
            swing_low=last_sl.price if last_sl else None,
            swing_high=last_sh.price if last_sh else None,
            swing_low_2=sl_2.price  if sl_2 else None,
            swing_high_2=sh_2.price if sh_2 else None,
            vol_regime=vol_reg.regime,
        )

        # ── 11. TP from S/R ───────────────────────────────────────────────────
        sl_tp = compute_sl_tp(
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

        # Phase 1 Fix #1: RR gate
        if best_rr < THRESHOLDS.MIN_RR:
            direction = "WAIT"

        # ── 12. AI Score ──────────────────────────────────────────────────────
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

        # ── 13. Trade Quality ─────────────────────────────────────────────────
        vol_ratio = float(df["Volume"].iloc[-1] / df["Volume"].rolling(20).mean().iloc[-1]) \
                    if "Volume" in df.columns else 1.0
        trade_q = compute_trade_quality(
            regime=regime, regime_confidence=regime_conf,
            ema_alignment=ema.alignment_strength,
            structure_trend=structure.trend,
            structure_clarity=consistency.structure_confidence,
            ev=0.5, rr=best_rr, volume_ratio=vol_ratio, vol_regime=vol_reg.regime,
        )

        # ── 14. Bayesian ──────────────────────────────────────────────────────
        bayes = compute_bayesian_analysis(
            rsi=rsi.value, regime=regime, regime_confidence=regime_conf,
            ema_alignment=ema.alignment_strength,
            structure_trend=structure.trend,
            vol_regime=vol_reg.regime, atr_pct=vol_reg.atr_pct,
        )

        # ── 15. Monte Carlo + Consistency ─────────────────────────────────────
        mc   = run_monte_carlo(close_series=df["Close"], entry=price,
                               stop_loss=stop_loss, target=tp2 or price * 1.04,
                               horizon=20, simulations=10_000)
        port = compute_portfolio_risk(df["Close"])
        mc_check = check_monte_carlo_consistency(
            prob_profit=mc.prob_profit, prob_target_hit=mc.prob_target_hit,
            prob_stop_hit=mc.prob_stop_hit, expected_return=mc.expected_return_pct,
            ev=0.5, rr=best_rr, pop=mc.prob_profit,
        )

        # ── 16. Position ──────────────────────────────────────────────────────
        position = compute_position(self._win_rate, max(self._avg_rr, best_rr), regime)

        # ── 17. Final Decision ────────────────────────────────────────────────
        final = evaluate_trade(
            direction=direction, regime_confidence=regime_conf,
            ai_score=ai_score.final_score, expected_value=position.ev,
            kelly_fraction=position.kelly_fraction, mc_profit_prob=mc.prob_profit,
            best_rr=best_rr, structure_trend=structure.trend, ema_bias=ema.bias,
        )

        # ── 18. Volume Profile + AVWAP ────────────────────────────────────────
        vol_profile = avwap_result = None
        try:
            vol_profile  = compute_volume_profile(df, lookback=60)
            avwap_result = compute_anchored_vwap(df)
        except Exception as exc:
            logger.debug("[%s] vol_profile/avwap: %s", symbol, exc)

        # ── 19. Report ────────────────────────────────────────────────────────
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

        runtime = round(time.time() - t0, 2)
        logger.info(
            "[%s] done %.1fs | %s conf=%.0f%% | %s AI=%.0f grade=%s RR=%.2f | approved=%s",
            symbol, runtime, regime, regime_conf, final.decision,
            ai_score.final_score, trade_q.grade, best_rr, final.approved,
        )

        return FuturesResult(
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
        )
