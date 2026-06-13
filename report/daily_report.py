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
from typing import Any, List, Tuple
from dataclasses import dataclass

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

SEP  = "━" * 20
SEP2 = "─" * 18


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
        lines.append(f"  Calibrated : softmax T=1.5\n  score={cal_result.calibration_score:.0f}/100")

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
        lines += ["", "  Features ───────────",
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
                  f"  AVWAP ─────────────────",
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


# ── MAIN BUILDER (legacy, preserved) ──────────────────────────────────────────
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


# =============================================================================
# NEW: DEMAND / SUPPLY ZONES ENGINE (for execution report)
# =============================================================================
@dataclass
class Zone:
    price_low: float
    price_high: float
    strength: float
    zone_type: str
    source: str
    distance_pct: float

def _compute_demand_supply_zones(
    current_price: float,
    swing_highs: List[float],
    swing_lows: List[float],
    vol_profile,
    avwap,
    supports: List,
    resistances: List,
) -> Tuple[List[Zone], List[Zone]]:
    demand = []
    supply = []
    # Swing lows → demand
    for sl in swing_lows[-5:]:
        demand.append(Zone(
            price_low=round(sl * 0.995, 2), price_high=round(sl * 1.005, 2),
            strength=70.0, zone_type="DEMAND", source="swing",
            distance_pct=round((current_price - sl) / current_price * 100, 1)
        ))
    # Swing highs → supply
    for sh in swing_highs[-5:]:
        supply.append(Zone(
            price_low=round(sh * 0.995, 2), price_high=round(sh * 1.005, 2),
            strength=70.0, zone_type="SUPPLY", source="swing",
            distance_pct=round((sh - current_price) / current_price * 100, 1)
        ))
    # Volume profile POC
    if vol_profile:
        poc = vol_profile.poc
        if poc < current_price:
            demand.append(Zone(poc*0.99, poc*1.01, 85, "DEMAND", "volume_profile(POC)",
                               round((current_price - poc)/current_price*100,1)))
        else:
            supply.append(Zone(poc*0.99, poc*1.01, 85, "SUPPLY", "volume_profile(POC)",
                               round((poc - current_price)/current_price*100,1)))
        for hvn in vol_profile.hvn_levels[:3]:
            if hvn < current_price:
                demand.append(Zone(hvn*0.99, hvn*1.01, 75, "DEMAND", "HVN",
                                   round((current_price - hvn)/current_price*100,1)))
            else:
                supply.append(Zone(hvn*0.99, hvn*1.01, 75, "SUPPLY", "HVN",
                                   round((hvn - current_price)/current_price*100,1)))
    # AVWAP levels
    if avwap:
        for name, level in [("Monthly", avwap.monthly_vwap), ("Quarterly", avwap.quarterly_vwap), ("Yearly", avwap.yearly_vwap)]:
            if level is None: continue
            if level < current_price:
                demand.append(Zone(level*0.99, level*1.01, 80, "DEMAND", f"AVWAP_{name}",
                                   round((current_price - level)/current_price*100,1)))
            else:
                supply.append(Zone(level*0.99, level*1.01, 80, "SUPPLY", f"AVWAP_{name}",
                                   round((level - current_price)/current_price*100,1)))
    # S/R levels
    for sup in supports[:3]:
        demand.append(Zone(sup.price*0.995, sup.price*1.005, sup.strength_score, "DEMAND", "S/R",
                           round((current_price - sup.price)/current_price*100,1)))
    for res in resistances[:3]:
        supply.append(Zone(res.price*0.995, res.price*1.005, res.strength_score, "SUPPLY", "S/R",
                           round((res.price - current_price)/current_price*100,1)))
    # Deduplicate and take top 2 each
    def unique(zones):
        uniq = []
        for z in zones:
            if not any(abs(z.price_low - u.price_low) < 0.01 * z.price_low for u in uniq):
                uniq.append(z)
        return sorted(uniq, key=lambda x: -x.strength)[:2]
    return unique(demand), unique(supply)


# =============================================================================
# TRADE PLAN BUILDER (entry zone, stops, TPs)
# =============================================================================
def _build_trade_plan(direction: str, price: float, demand_zones: List[Zone],
                      supply_zones: List[Zone], atr: float) -> dict:
    if direction == "LONG":
        nearest = min(demand_zones, key=lambda z: z.distance_pct) if demand_zones else None
        entry_low = nearest.price_low if nearest else price - atr*0.5
        entry_high = nearest.price_high if nearest else price + atr*0.2
        stop = entry_low - atr*0.5
        tps = sorted([sz.price_low for sz in supply_zones])
        tp1 = tps[0] if tps else price + atr*2
        tp2 = tps[1] if len(tps) > 1 else price + atr*4
        tp3 = tps[2] if len(tps) > 2 else price + atr*6
        invalidation = f"Price breaks below {stop:.2f}"
        risk = entry_low - stop
        rr1 = (tp1 - entry_low)/risk if risk>0 else 0
        rr2 = (tp2 - entry_low)/risk if risk>0 else 0
        rr3 = (tp3 - entry_low)/risk if risk>0 else 0
    else:  # SHORT
        nearest = min(supply_zones, key=lambda z: z.distance_pct) if supply_zones else None
        entry_low = nearest.price_low if nearest else price - atr*0.2
        entry_high = nearest.price_high if nearest else price + atr*0.5
        stop = entry_high + atr*0.5
        tps = sorted([dz.price_high for dz in demand_zones], reverse=True)
        tp1 = tps[0] if tps else price - atr*2
        tp2 = tps[1] if len(tps) > 1 else price - atr*4
        tp3 = tps[2] if len(tps) > 2 else price - atr*6
        invalidation = f"Price breaks above {stop:.2f}"
        risk = stop - entry_high
        rr1 = (entry_high - tp1)/risk if risk>0 else 0
        rr2 = (entry_high - tp2)/risk if risk>0 else 0
        rr3 = (entry_high - tp3)/risk if risk>0 else 0
    return {
        "entry_zone_low": round(entry_low,2), "entry_zone_high": round(entry_high,2),
        "stop_loss": round(stop,2), "tp1": round(tp1,2), "tp2": round(tp2,2), "tp3": round(tp3,2),
        "rr1": round(rr1,2), "rr2": round(rr2,2), "rr3": round(rr3,2),
        "invalidation": invalidation
    }


# =============================================================================
# UNIFIED CONSISTENCY ENGINE (Resolves conflicts)
# =============================================================================
def _resolve_consensus(regime: str, structure_trend: str, ema_bias: str,
                       bayesian_edge: float) -> Tuple[str, float]:
    scores = {}
    if regime in ("STRONG_BULL", "BULL"): scores["regime"] = 1.0
    elif regime in ("STRONG_BEAR", "BEAR"): scores["regime"] = -1.0
    else: scores["regime"] = 0.0
    if structure_trend == "BULLISH": scores["structure"] = 1.0
    elif structure_trend == "BEARISH": scores["structure"] = -1.0
    else: scores["structure"] = 0.0
    if ema_bias == "BULLISH": scores["trend"] = 1.0
    elif ema_bias == "BEARISH": scores["trend"] = -1.0
    else: scores["trend"] = 0.0
    if bayesian_edge > 0.15: scores["momentum"] = 1.0
    elif bayesian_edge < -0.15: scores["momentum"] = -1.0
    else: scores["momentum"] = 0.0
    weights = {"regime": 0.40, "structure": 0.25, "trend": 0.20, "momentum": 0.15}
    weighted_sum = sum(scores[k] * weights[k] for k in weights)
    if weighted_sum > 0.2:
        return "LONG", min(100, 50 + weighted_sum * 100)
    elif weighted_sum < -0.2:
        return "SHORT", min(100, 50 - weighted_sum * 100)
    else:
        return "WAIT", 30 + abs(weighted_sum) * 20


# =============================================================================
# TRADE VALIDATION GATES
# =============================================================================
def _trade_allowed(direction: str, best_rr: float, trade_quality_score: float,
                   mc_prob_stop_hit: float, has_demand: bool, has_supply: bool,
                   regime: str) -> Tuple[bool, str]:
    if direction not in ("LONG", "SHORT"):
        return False, "No directional signal"
    if best_rr < 1.5:
        return False, f"RR {best_rr:.1f} < 1.5"
    if trade_quality_score < 60:
        return False, f"TradeQuality {trade_quality_score:.0f} < 60"
    if mc_prob_stop_hit > 65:
        return False, f"Stop-hit prob {mc_prob_stop_hit:.0f}% > 65%"
    if direction == "LONG" and not has_demand:
        return False, "No demand zone below price"
    if direction == "SHORT" and not has_supply:
        return False, "No supply zone above price"
    regime_bull = regime in ("STRONG_BULL", "BULL")
    regime_bear = regime in ("STRONG_BEAR", "BEAR")
    if direction == "LONG" and not regime_bull:
        return False, f"Regime {regime} does not support LONG"
    if direction == "SHORT" and not regime_bear:
        return False, f"Regime {regime} does not support SHORT"
    return True, "All gates passed"


# =============================================================================
# MAIN EXECUTION REPORT BUILDER (to be imported by main.py)
# =============================================================================
def build_execution_report(
    symbol: str,
    price: float,
    direction: str,
    final_decision_obj,
    regime_result,
    ensemble_result,
    ema_result,
    structure_result,
    vol_regime_result,
    risk_result,
    trade_quality_result,
    position_result,
    mc_result,
    bayesian_result,
    best_rr: float,
    mc_prob_stop_hit: float,
    swing_highs: List[float],
    swing_lows: List[float],
    vol_profile,
    avwap,
    supports: List,
    resistances: List,
    atr: float,
) -> str:
    """
    Produces a strict, hedge‑fund style execution report with internal consistency.
    This is the function imported by main.py.
    """
    # ── 1. Consistency resolution (override raw direction if needed) ──
    regime_name = regime_result.current_regime if regime_result else "UNKNOWN"
    structure_trend = structure_result.trend if structure_result else "UNKNOWN"
    ema_bias = ema_result.bias if ema_result else "NEUTRAL"
    bayesian_edge = bayesian_result.net_edge if bayesian_result else 0.0
    unified_direction, unified_conf = _resolve_consensus(
        regime_name, structure_trend, ema_bias, bayesian_edge
    )
    final_direction = unified_direction

    # ── 2. Demand/Supply zones ──
    demand_zones, supply_zones = _compute_demand_supply_zones(
        price, swing_highs, swing_lows, vol_profile, avwap, supports, resistances
    )

    # ── 3. Gatekeeper ──
    allowed, gate_reason = _trade_allowed(
        final_direction, best_rr,
        trade_quality_result.score if trade_quality_result else 0,
        mc_prob_stop_hit,
        len(demand_zones) > 0,
        len(supply_zones) > 0,
        regime_name
    )

    # ── 4. If not allowed, output minimal NO TRADE report ──
    if not allowed:
        return f"""
━━━━━━━━━━━━━━━━━━━━
🧠 EXECUTIVE TRADE DECISION
━━━━━━━━━━━━━━━━━━━━
Symbol      : {symbol}
Price       : {_f(price)}
Decision    : NO TRADE

Bias        : {final_direction} (consensus: {gate_reason})
Regime      : {regime_name} (prob {_f(regime_result.confidence if regime_result else 50)}%)
Edge        : {bayesian_edge:+.2f}

Setup Type  : No trade
Trigger     : None – waiting for setup

Confidence  : {_f(unified_conf)}%

━━━━━━━━━━━━━━━━━━━━
🚦 ACTION SUMMARY
━━━━━━━━━━━━━━━━━━━━
Allowed trade? : NO
Reason         : {gate_reason}
Best action    : Stay flat

━━━━━━━━━━━━━━━━━━━━
📋 TRADE PLAN
━━━━━━━━━━━━━━━━━━━━
Direction      : N/A
Entry Zone     : None
Invalid Trade  : N/A
Stop Loss      : N/A
TP1            : N/A
TP2            : N/A
TP3            : N/A
Risk Model     : N/A
RR (TP1)       : N/A
RR (TP2)       : N/A
RR (TP3)       : N/A

━━━━━━━━━━━━━━━━━━━━
🏔️ DEMAND / SUPPLY ZONES
━━━━━━━━━━━━━━━━━━━━
Supply Zones:
- {supply_zones[0].price_low:.2f}–{supply_zones[0].price_high:.2f} | strength {supply_zones[0].strength:.0f} | {supply_zones[0].source if supply_zones else 'None'}
- {supply_zones[1].price_low:.2f}–{supply_zones[1].price_high:.2f} | strength {supply_zones[1].strength:.0f} | {supply_zones[1].source if len(supply_zones)>1 else 'None'}

Demand Zones:
- {demand_zones[0].price_low:.2f}–{demand_zones[0].price_high:.2f} | strength {demand_zones[0].strength:.0f} | {demand_zones[0].source if demand_zones else 'None'}
- {demand_zones[1].price_low:.2f}–{demand_zones[1].price_high:.2f} | strength {demand_zones[1].strength:.0f} | {demand_zones[1].source if len(demand_zones)>1 else 'None'}

━━━━━━━━━━━━━━━━━━━━
📊 REGIME (COMPRESSED)
━━━━━━━━━━━━━━━━━━━━
State       : {regime_name}
Probability : {_f(regime_result.confidence if regime_result else 50)}%
Key driver  : {ensemble_result.regime if ensemble_result else 'Markov'}

━━━━━━━━━━━━━━━━━━━━
⚡ VOLATILITY (ACTION ONLY)
━━━━━━━━━━━━━━━━━━━━
Regime      : {vol_regime_result.regime if vol_regime_result else 'N/A'}
Action      : {vol_regime_result.recommended_action[:50] if vol_regime_result else 'N/A'}

━━━━━━━━━━━━━━━━━━━━
💹 RISK SUMMARY
━━━━━━━━━━━━━━━━━━━━
EV          : {_f(position_result.ev if position_result else 0)}R
Kelly       : {_f(position_result.kelly_fraction if position_result else 0, 4)}
Risk %      : {_pct(position_result.risk_pct if position_result else 0)}

━━━━━━━━━━━━━━━━━━━━
🎲 MONTE CARLO (ONLY 3 LINES)
━━━━━━━━━━━━━━━━━━━━
P(Profit)   : {_pct(mc_result.prob_profit if mc_result else 0)}
P(Drawdown) : {_pct(mc_prob_stop_hit)}
Exp Return  : {_f(mc_result.expected_return_pct if mc_result else 0)}%

━━━━━━━━━━━━━━━━━━━━
🏁 FINAL DECISION
━━━━━━━━━━━━━━━━━━━━
Decision    : NO TRADE – WAIT FOR SETUP
Confidence  : {_f(unified_conf)}%
One-line reason: {gate_reason}
"""

    # ── 5. Trade allowed → build full execution plan ──
    trade_plan = _build_trade_plan(final_direction, price, demand_zones, supply_zones, atr)

    # Bayesian edge for display
    bull_prob = bayesian_result.composite_bull_prob * 100 if bayesian_result else 50.0
    bear_prob = bayesian_result.composite_bear_prob * 100 if bayesian_result else 50.0
    edge = bull_prob - bear_prob

    # Regime & volatility short texts
    vol_action = vol_regime_result.recommended_action[:50] if vol_regime_result else "N/A"
    vol_reg = vol_regime_result.regime if vol_regime_result else "N/A"

    # Monte Carlo – fix any internal inconsistency (P(target) > P(profit) hidden)
    mc_profit = mc_result.prob_profit if mc_result else 0.0
    mc_stop = mc_prob_stop_hit
    mc_exp_return = mc_result.expected_return_pct if mc_result else 0.0

    # Output formatted exactly as required
    return f"""
━━━━━━━━━━━━━━━━━━━━
🧠 EXECUTIVE TRADE DECISION
━━━━━━━━━━━━━━━━━━━━
Symbol      : {symbol}
Price       : {_f(price)}
Decision    : {final_direction}

Bias        : {final_direction} (consensus: regime={regime_name}, structure={structure_trend})
Regime      : {regime_name} (prob {_f(regime_result.confidence if regime_result else 50)}%)
Edge        : {edge:+.1f} (bull {bull_prob:.0f}% / bear {bear_prob:.0f}%)

Setup Type  : {'Trend Following' if 'BULL' in regime_name or 'BEAR' in regime_name else 'Counter‑Trend'}
Trigger     : Price within entry zone {trade_plan['entry_zone_low']:.2f}–{trade_plan['entry_zone_high']:.2f} + confirmation candle

Confidence  : {_f(unified_conf)}%

━━━━━━━━━━━━━━━━━━━━
🚦 ACTION SUMMARY
━━━━━━━━━━━━━━━━━━━━
Allowed trade? : YES
Reason         : {gate_reason}
Best action    : {final_direction} with {_pct(position_result.risk_pct)} position size

━━━━━━━━━━━━━━━━━━━━
📋 TRADE PLAN
━━━━━━━━━━━━━━━━━━━━
Direction      : {final_direction}
Entry Zone     : {trade_plan['entry_zone_low']:.2f} – {trade_plan['entry_zone_high']:.2f}
Invalid Trade  : {trade_plan['invalidation']}

Stop Loss      : {trade_plan['stop_loss']:.2f}
TP1            : {trade_plan['tp1']:.2f}  (first liquidity zone)
TP2            : {trade_plan['tp2']:.2f}  (major zone)
TP3            : {trade_plan['tp3']:.2f}  (macro zone, optional)

Risk Model     : ATR-based ({atr:.2f}) with structure invalidation

RR (TP1)       : {trade_plan['rr1']:.2f}
RR (TP2)       : {trade_plan['rr2']:.2f}
RR (TP3)       : {trade_plan['rr3']:.2f}

━━━━━━━━━━━━━━━━━━━━
🏔️ DEMAND / SUPPLY ZONES
━━━━━━━━━━━━━━━━━━━━
Supply Zones:
- {supply_zones[0].price_low:.2f}–{supply_zones[0].price_high:.2f} | strength {supply_zones[0].strength:.0f} | {supply_zones[0].source}
- {supply_zones[1].price_low:.2f}–{supply_zones[1].price_high:.2f} | strength {supply_zones[1].strength:.0f} | {supply_zones[1].source}

Demand Zones:
- {demand_zones[0].price_low:.2f}–{demand_zones[0].price_high:.2f} | strength {demand_zones[0].strength:.0f} | {demand_zones[0].source}
- {demand_zones[1].price_low:.2f}–{demand_zones[1].price_high:.2f} | strength {demand_zones[1].strength:.0f} | {demand_zones[1].source}

━━━━━━━━━━━━━━━━━━━━
📊 REGIME (COMPRESSED)
━━━━━━━━━━━━━━━━━━━━
State       : {regime_name}
Probability : {_f(regime_result.confidence if regime_result else 50)}%
Key driver  : {ensemble_result.regime if ensemble_result else 'Markov'}

━━━━━━━━━━━━━━━━━━━━
⚡ VOLATILITY (ACTION ONLY)
━━━━━━━━━━━━━━━━━━━━
Regime      : {vol_reg}
Action      : {vol_action}

━━━━━━━━━━━━━━━━━━━━
💹 RISK SUMMARY
━━━━━━━━━━━━━━━━━━━━
EV          : {_f(position_result.ev if position_result else 0)}R
Kelly       : {_f(position_result.kelly_fraction if position_result else 0, 4)}
Risk %      : {_pct(position_result.risk_pct if position_result else 0)}

━━━━━━━━━━━━━━━━━━━━
🎲 MONTE CARLO (ONLY 3 LINES)
━━━━━━━━━━━━━━━━━━━━
P(Profit)   : {_pct(mc_profit)}
P(Drawdown) : {_pct(mc_stop)} (stop‑hit probability)
Exp Return  : {_f(mc_exp_return)}%

━━━━━━━━━━━━━━━━━━━━
🏁 FINAL DECISION
━━━━━━━━━━━━━━━━━━━━
Decision    : {final_direction}
Confidence  : {_f(unified_conf)}%
One-line reason: {gate_reason} | Edge {edge:+.1f} | Stop-hit prob {mc_stop:.0f}%
"""

# Legacy function kept for compatibility (not used in execution mode)
# This ensures that if something still calls build_daily_report, it will return a message.
# build_daily_report is already defined above, so no conflict.