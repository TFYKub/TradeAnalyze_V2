"""
Institutional Daily Report  (v2 — all 11 phases)
==================================================
Sections:
  1.  Header + Price
  2.  Markov Regime Dashboard  (calibrated probs + transition matrix)
  3.  Regime Ensemble          (4-component weighted)
  4.  Volatility Regime        (vol score + adjustments)
  5.  Market Structure Analysis (with consistency check)
  6.  Bayesian Probabilities
  7.  Key S/R Levels + Volume Profile + AVWAP
  8.  Trade Plan               (institutional stop + 4 stop types)
  9.  Trade Quality Grade
  10. AI Score Breakdown
  11. Position Sizing (Kelly/EV)
  12. Monte Carlo + Portfolio Risk
  13. Final Institutional Dashboard
"""
from __future__ import annotations
import math
from datetime import datetime
from typing import Any

# ── Helpers ───────────────────────────────────────────────────────────────────
def _f(x: Any, dec: int = 2) -> str:
    if x is None: return "N/A"
    try:
        v = float(x)
        return "N/A" if (math.isnan(v) or math.isinf(v)) else f"{v:.{dec}f}"
    except: return str(x)

def _pct(x: Any) -> str:
    return "N/A" if x is None else f"{float(x):.1f}%"

def _s(x: Any) -> str:
    return "N/A" if (x is None or x == "") else str(x)

def _diff(entry: float, target: float) -> str:
    try:
        d = (float(target) - float(entry)) / float(entry) * 100
        return f"({'+' if d >= 0 else ''}{d:.1f}%)"
    except: return ""

def _bar(pct: float, w: int = 10) -> str:
    n = max(0, min(w, round(pct / 100 * w)))
    return "█" * n + "░" * (w - n)

_RE  = {"STRONG_BULL":"🚀","BULL":"📈","RANGE":"↔️","CORRECTION":"⚠️",
        "BEAR":"📉","STRONG_BEAR":"🔻"}
_DE  = {"LONG":"🟢","SHORT":"🔴","NO_TRADE":"⏸️","WAIT":"⏸️"}
_GE  = {"A+":"🏆","A":"🥇","B":"🥈","C":"🥉","REJECT":"❌"}
_VE  = {"HIGH_VOL":"🔥","NORMAL_VOL":"✅","LOW_VOL":"💤","PANIC_VOL":"🚨"}

SEP  = "━" * 28
SEP2 = "─" * 26


# ── Section 1: Header ─────────────────────────────────────────────────────────
def _hdr(symbol, price):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return [SEP, f"🐱 TradeAnalyze  |  {symbol}  |  {now}", SEP,
            f"💰 Price : {_f(price)}", ""]


# ── Section 2: Markov Regime ──────────────────────────────────────────────────
def _section_markov(regime_result, cal_result=None):
    lines = [SEP, "📊 MARKOV REGIME DASHBOARD", SEP]
    if regime_result is None:
        lines.append("  (Regime engine unavailable)")
        return lines
    reg  = regime_result.current_regime
    prob = regime_result.regime_probability
    conf = regime_result.confidence
    nxt  = regime_result.expected_next_regime
    perm = regime_result.trade_permission
    probs= regime_result.regime_probs_all
    tm   = regime_result.transition_matrix

    if cal_result:
        conf = cal_result.calibrated_conf
        probs = cal_result.calibrated_probs
        lines.append(f"  Calibrated : softmax T=1.5  score={cal_result.calibration_score:.0f}/100")

    lines += [
        f"  Regime     : {_RE.get(reg,'❓')} {reg}",
        f"  Probability: {_pct(prob*100)}",
        f"  Confidence : {_pct(conf)}",
        f"  Next Regime: {_RE.get(nxt,'❓')} {nxt}",
        f"  Permission : {perm}", "",
        "  All Regimes ──────────────────",
    ]
    for r, p in sorted(probs.items(), key=lambda x: -x[1]):
        lines.append(f"    {r:<12} {_bar(p*100,8)}  {_pct(p*100)}")

    if reg in tm:
        lines += ["", f"  Transitions from {reg} ────────"]
        for to_r, p in sorted(tm[reg].items(), key=lambda x: -x[1])[:4]:
            lines.append(f"    → {to_r:<12}  {_pct(p*100)}")

    fs = regime_result.feature_snapshot
    if fs:
        lines += ["", "  Features ──────────────────────",
                  f"    DailyRet  : {_f(fs.get('daily_return'),3)}%",
                  f"    RollingVol: {_f(fs.get('rolling_vol_20'),1)}% ann.",
                  f"    EMAMomentum:{_f(fs.get('momentum_score'),2)}%",
                  f"    RSI Norm  : {_f(fs.get('rsi_normalised'),3)}"]
    return lines


# ── Section 3: Regime Ensemble ────────────────────────────────────────────────
def _section_ensemble(ens):
    if ens is None:
        return []
    lines = [SEP, "🧩 REGIME ENSEMBLE (4-Component)", SEP,
             f"  Ensemble   : {_RE.get(ens.regime,'❓')} {ens.regime}",
             f"  Confidence : {_pct(ens.confidence)}",
             f"  Clarity    : {_f(ens.ensemble_score)}/100",
             f"  Permission : {ens.trade_permission}",
             f"  Pos Mult   : {ens.position_size_mult:.0%}", "",
             "  Components ────────────────────"]
    for eng, r in ens.component_scores.items():
        lines.append(f"    {eng:<10} → {_RE.get(r,'❓')} {r}")
    lines += ["", "  Weighted Probs ─────────────────"]
    for r, p in sorted(ens.weighted_probs.items(), key=lambda x: -x[1]):
        lines.append(f"    {r:<12} {_bar(p*100,8)}  {_pct(p*100)}")
    return lines


# ── Section 4: Volatility Regime ──────────────────────────────────────────────
def _section_vol_regime(vr):
    if vr is None: return []
    return [
        SEP, f"⚡ VOLATILITY REGIME  {_VE.get(vr.regime,'')} {vr.regime}", SEP,
        f"  Vol Score  : {_f(vr.vol_score)}/100",
        f"  HV20       : {_pct(vr.hv20*100)}  HV5: {_pct(vr.hv5*100)}",
        f"  ATR%       : {_f(vr.atr_pct)}%  VoV: {_f(vr.vov,4)}",
        f"  IV/HV      : {_f(vr.iv_hv_ratio,2)}",
        f"  Pos Mult   : {vr.position_size_mult:.0%}  Stop Mult: {vr.stop_distance_mult:.1f}×",
        f"  Strategy   : {vr.preferred_strategy}",
        f"  Action     : {vr.recommended_action}",
    ]


# ── Section 5: Market Structure ───────────────────────────────────────────────
def _section_structure(ema, rsi, structure, divergence, trend_filter, consistency=None):
    lines = [SEP, "📐 MARKET STRUCTURE ANALYSIS", SEP,
             f"  Trend      : {_s(ema.bias)}  EMA12={_f(ema.ema12)}  EMA26={_f(ema.ema26)}",
             f"  EMA Spread : {_f(ema.spread_pct,3)}%  Strength={_f(ema.alignment_strength)}/100",
             f"  Structure  : {_s(structure.pattern)}  ({_s(structure.trend)})",
             f"  Clarity    : {_f(structure.structure_score)}/100",
             f"  BOS Bull   : {structure.bos_bullish}  BOS Bear: {structure.bos_bearish}",
             f"  RSI        : {_f(rsi.value)}  Zone: {_s(rsi.zone)}  Mom: {_s(rsi.momentum)}",
             f"  Divergence : {_s(divergence.kind)}  detected={divergence.detected}",
             f"  Final Bias : {_s(trend_filter.final_bias)}",
             f"  Reason     : {_s(trend_filter.reason)[:60]}",
    ]
    if consistency:
        flag = "⚠️" if consistency.conflict_detected else "✅"
        lines += ["",
                  f"  {flag} Structure Consistency: {consistency.consistency_grade}",
                  f"  Conf Adj   : {_f(consistency.structure_confidence)}/100",
                  f"  Penalty    : -{_f(consistency.confidence_penalty)}"]
        if consistency.conflict_detected:
            lines.append(f"  Conflict   : {consistency.conflict_reason[:70]}")
    return lines


# ── Section 6: Bayesian ───────────────────────────────────────────────────────
def _section_bayesian(bayes):
    if bayes is None: return []
    lines = [SEP, "🎯 BAYESIAN PROBABILITY ENGINE", SEP,
             f"  Bull Edge  : {_pct(bayes.composite_bull_prob*100)}",
             f"  Bear Edge  : {_pct(bayes.composite_bear_prob*100)}",
             f"  Net Edge   : {bayes.net_edge*100:+.1f}%",
             f"  Interpret  : {bayes.interpretation}", ""]
    for sig in bayes.signals:
        lines.append(f"  {sig.signal_name:<10} {sig.description[:60]}")
    return lines


# ── Section 7: S/R + Volume Profile + AVWAP ──────────────────────────────────
def _section_sr(sr, vol_profile=None, avwap=None):
    lines = [SEP, "🏔️  KEY LEVELS", SEP, "  Resistance ──────────────────"]
    for i, lvl in enumerate(sr.get("resistances", [])[:3], 1):
        lines.append(f"  R{i}: {_f(lvl.price)}  dist={_f(lvl.distance_pct,2)}%  "
                     f"touches={lvl.touch_count}  score={_f(lvl.strength_score)}")
    lines.append("  Support ────────────────────────")
    for i, lvl in enumerate(sr.get("supports", [])[:3], 1):
        lines.append(f"  S{i}: {_f(lvl.price)}  dist={_f(lvl.distance_pct,2)}%  "
                     f"touches={lvl.touch_count}  score={_f(lvl.strength_score)}")

    if vol_profile:
        lines += ["",
                  f"  Vol Profile ─────────────────────",
                  f"  POC      : {_f(vol_profile.poc)}",
                  f"  VA High  : {_f(vol_profile.va_high)}",
                  f"  VA Low   : {_f(vol_profile.va_low)}",
                  f"  Inst Bias: {vol_profile.institutional_bias}",
                  f"  HVN      : {', '.join(_f(h) for h in vol_profile.hvn_levels[:3])}",
        ]
    if avwap:
        lines += ["",
                  f"  AVWAP ───────────────────────────",
                  f"  Monthly  : {_f(avwap.monthly_vwap)}  ({avwap.monthly_dist_pct:+.1f}%)",
                  f"  Quarterly: {_f(avwap.quarterly_vwap)}  ({avwap.quarterly_dist_pct:+.1f}%)",
                  f"  Yearly   : {_f(avwap.yearly_vwap)}  ({avwap.yearly_dist_pct:+.1f}%)",
                  f"  Trend    : {avwap.avwap_trend}  Above: {avwap.above_count}/4",
        ]
    return lines


# ── Section 8: Trade Plan ─────────────────────────────────────────────────────
def _section_trade_plan(risk, ai_score, entry_result, inst_stop=None):
    d     = risk.direction
    emoji = _DE.get(d, "❓")
    rr    = max(risk.rr1, risk.rr2)
    lines = [SEP, "📋 TRADE PLAN", SEP,
             f"  Direction  : {emoji} {d}",
             f"  Entry      : {_f(risk.entry)}",
             f"  Stop Loss  : {_f(risk.stop_loss)}  {_diff(risk.entry, risk.stop_loss)}",
             f"  TP1        : {_f(risk.tp1)}  {_diff(risk.entry, risk.tp1)}",
             f"  TP2        : {_f(risk.tp2)}  {_diff(risk.entry, risk.tp2)}",
             f"  RR         : {_f(rr, 2)}  Valid(≥1.5): {risk.valid_rr}",
    ]
    if inst_stop:
        lines += ["",
                  f"  Stop Engine ─────────────────────",
                  f"  ATR Stop   : {_f(inst_stop.atr_stop)}",
                  f"  Struct Stop: {_f(inst_stop.structure_stop)}",
                  f"  Swing Stop : {_f(inst_stop.swing_stop)}",
                  f"  Vol Stop   : {_f(inst_stop.volatility_stop)}",
                  f"  Selected   : {_f(inst_stop.selected_stop)}",
                  f"  Reason     : {inst_stop.stop_reason}",
                  f"  Risk       : {_f(inst_stop.risk)}  ({_f(inst_stop.risk_pct,2)}%)",
                  f"  MinTP(2R)  : {_f(inst_stop.min_tp_for_2rr)}",
        ]
    lines += ["",
              f"  AI Score   : {_f(ai_score.final_score)}/100",
              f"    Regime(30%): {_f(ai_score.regime_score)}",
              f"    Struct(25%): {_f(ai_score.structure_score)}",
              f"    Trend (20%): {_f(ai_score.trend_score)}",
              f"    Moment(15%): {_f(ai_score.momentum_score)}",
              f"    RR    (10%): {_f(ai_score.rr_score)}",
              f"  Trigger    : {_s(entry_result.reason)[:60]}",
    ]
    return lines


# ── Section 9: Trade Quality ──────────────────────────────────────────────────
def _section_quality(tq):
    if tq is None: return []
    g     = tq.grade
    emoji = _GE.get(g, "❓")
    lines = [SEP, f"⭐ TRADE QUALITY  {emoji} {g}  ({_f(tq.score)}/100)", SEP]
    for k, v in tq.component_scores.items():
        bar = _bar(v, 8)
        lines.append(f"  {k:<10} {bar}  {_f(v)}")
    lines.append(f"  Allowed    : {tq.trade_allowed}")
    lines.append(f"  Reason     : {tq.grade_reason[:60]}")
    return lines


# ── Section 10: Position Sizing ───────────────────────────────────────────────
def _section_position(position):
    return [SEP, "💹 POSITION SIZING", SEP,
            f"  Win Rate   : {_pct(position.win_rate*100)}",
            f"  Avg RR     : {_f(position.avg_rr, 2)}",
            f"  EV         : {_f(position.ev, 3)}R",
            f"  Full Kelly : {_f(position.kelly_fraction, 4)}",
            f"  Half Kelly : {_f(position.half_kelly, 4)}",
            f"  Regime Mult: {position.regime_mult:.0%}",
            f"  Risk %     : {_pct(position.risk_pct*100)}",
            f"  Kelly OK   : {position.kelly_valid}",
    ]


# ── Section 11: Monte Carlo + Portfolio ──────────────────────────────────────
def _section_simulation(mc, port, consistency=None):
    lines = [SEP, "🎲 RISK & SIMULATION", SEP,
             f"  MC ({mc.simulations:,} paths, {mc.horizon}d)",
             f"  P(Profit)  : {_bar(mc.prob_profit)} {_pct(mc.prob_profit)}",
             f"  P(Stop Hit): {_bar(mc.prob_stop_hit)} {_pct(mc.prob_stop_hit)}",
             f"  P(Target)  : {_bar(mc.prob_target_hit)} {_pct(mc.prob_target_hit)}",
             f"  Exp Return : {_f(mc.expected_return_pct)}%",
             f"  Exp DD     : {_f(mc.expected_drawdown_pct)}%",
             f"  95% CI     : [{_f(mc.ci_95_low)}%, {_f(mc.ci_95_high)}%]",
             f"  VaR(95%)   : {_f(mc.var_95)}%",
             f"  CVaR(95%)  : {_f(mc.cvar_95)}%",
    ]
    if consistency:
        flag = "✅" if consistency.is_consistent else "⚠️"
        lines.append(f"  {flag} MC Consistency adj: ×{consistency.confidence_adj}")
        if consistency.errors:
            lines.append(f"  Error: {consistency.errors[0][:60]}")
    lines += ["",
              f"  Portfolio Risk (Historical)",
              f"  VaR 95%    : {_f(port.var_95)}%",
              f"  CVaR 95%   : {_f(port.cvar_95)}%",
              f"  MaxDD      : {_f(port.max_drawdown)}%",
              f"  Vol (Ann)  : {_f(port.volatility_annual)}%",
              f"  Sharpe     : {_f(port.sharpe, 3)}",
              f"  Sortino    : {_f(port.sortino, 3)}",
              f"  Calmar     : {_f(port.calmar, 3)}",
    ]
    return lines


# ── Section 12: Final Dashboard ───────────────────────────────────────────────
def _section_final(symbol, price, final, ai_score, regime, regime_conf,
                   mc, port, position, risk, tq=None, vol_regime=None, bayes=None):
    emoji = _DE.get(final.decision, "❓")
    conf_bar = _bar(final.confidence_pct)
    rr    = max(risk.rr1, risk.rr2)

    lines = [SEP,
             f"🏛️  INSTITUTIONAL TRADE DASHBOARD — {symbol}",
             SEP,
             f"  Price      : {_f(price)}",
             f"  Regime     : {_RE.get(regime,'❓')} {regime}  (conf {regime_conf:.0f}%)",
    ]
    if vol_regime:
        lines.append(f"  Vol Regime : {_VE.get(vol_regime.regime,'')} {vol_regime.regime}")
    if tq:
        lines.append(f"  Trade Grade: {_GE.get(tq.grade,'❓')} {tq.grade}  ({tq.score:.0f}/100)")
    if bayes:
        lines.append(f"  Bayesian   : Bull={_pct(bayes.composite_bull_prob*100)}  Bear={_pct(bayes.composite_bear_prob*100)}")
    lines += [
        f"  AI Score   : {_f(ai_score.final_score)}/100",
        f"  EV         : {_f(position.ev, 3)}R",
        f"  Kelly      : {_f(position.kelly_fraction, 4)}",
        f"  MC P(Profit): {_pct(mc.prob_profit)}",
        f"  VaR 95%    : {_f(port.var_95)}%",
        f"  Sharpe     : {_f(port.sharpe, 3)}",
        f"  Sortino    : {_f(port.sortino, 3)}",
        SEP2,
        f"  Direction  : {emoji} {risk.direction}",
        f"  Entry      : {_f(risk.entry)}",
        f"  Stop Loss  : {_f(risk.stop_loss)}  {_diff(risk.entry, risk.stop_loss)}",
        f"  TP1        : {_f(risk.tp1)}  {_diff(risk.entry, risk.tp1)}",
        f"  TP2        : {_f(risk.tp2)}  {_diff(risk.entry, risk.tp2)}",
        f"  Risk Reward: {_f(rr, 2)}",
        f"  Position   : {_pct(position.risk_pct*100)} of account",
        SEP2,
        f"  Gates      : {conf_bar}  {_f(final.confidence_pct)}%",
        f"  Passed     : {len(final.gates_passed)}/{len(final.gates_passed)+len(final.gates_failed)}",
        "",
        SEP,
        f"  FINAL DECISION : {emoji} {final.decision}",
        SEP,
        f"  {final.reason[:80]}",
    ]
    if final.gates_failed:
        lines.append(f"  Blocked    : {final.gates_failed[0]}")
    return lines


# ── MAIN BUILDER ──────────────────────────────────────────────────────────────
def build_daily_report(
    symbol:       str,
    price:        float,
    regime,       # RegimeResult or None
    ema,          # EMAResult
    rsi,          # RSIResult
    structure,    # StructureResult
    divergence,   # DivergenceResult
    trend_filter, # TrendFilterResult
    sr:           dict,
    risk,         # RiskResult
    ai_score,     # AIScoreResult
    mc,           # MonteCarloResult
    port,         # PortfolioRiskResult
    position,     # PositionResult
    entry_result, # EntryResult
    final,        # FinalDecision
    # New Phase additions (all optional for backward compat)
    cal_result         = None,   # CalibrationResult
    ensemble           = None,   # EnsembleRegimeResult
    vol_regime         = None,   # VolatilityRegimeResult
    consistency        = None,   # StructureConsistencyResult
    mc_consistency     = None,   # ConsistencyResult
    bayesian           = None,   # BayesianResult
    trade_quality      = None,   # TradeQualityResult
    inst_stop          = None,   # InstitutionalStopResult
    vol_profile        = None,   # VolumeProfileResult
    avwap              = None,   # AVWAPResult
) -> str:

    sections = (
        _hdr(symbol, price)
        + _section_markov(regime, cal_result)
        + _section_ensemble(ensemble)
        + _section_vol_regime(vol_regime)
        + _section_structure(ema, rsi, structure, divergence, trend_filter, consistency)
        + _section_bayesian(bayesian)
        + _section_sr(sr, vol_profile, avwap)
        + _section_trade_plan(risk, ai_score, entry_result, inst_stop)
        + _section_quality(trade_quality)
        + _section_position(position)
        + _section_simulation(mc, port, mc_consistency)
        + _section_final(symbol, price, final, ai_score,
                         ensemble.regime if ensemble else (regime.current_regime if regime else "RANGE"),
                         ensemble.confidence if ensemble else (regime.confidence if regime else 50),
                         mc, port, position, risk, trade_quality, vol_regime, bayesian)
    )

    return "\n".join(sections)
