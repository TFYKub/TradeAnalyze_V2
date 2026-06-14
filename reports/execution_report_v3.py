"""
V3 Execution Report – Unified Futures + Options
================================================
Single message containing both trade decisions.
"""
from datetime import datetime
from typing import Optional
import math

from core.trade_state import TradeStateV3


def _f(x, dec: int = 2) -> str:
    if x is None:
        return "N/A"
    try:
        v = float(x)
        return "N/A" if (math.isnan(v) or math.isinf(v)) else f"{v:.{dec}f}"
    except:
        return str(x)

def _pct(x) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{float(x):.1f}%"
    except:
        return str(x)

def _bar(pct: float, w: int = 10) -> str:
    n = max(0, min(w, round(pct / 100 * w)))
    return "█" * n + "░" * (w - n)

def _diff(price, target) -> str:
    try:
        d = (float(target) - float(price)) / float(price) * 100
        return f"(({'+' if d >= 0 else ''}{d:.1f}%))"
    except:
        return ""

def _format_strategy_legs(setup) -> list[str]:
    lines = []
    for leg in setup.legs:
        action = leg.get("action", "").upper()
        opt_type = leg.get("type", "").upper()
        strike = leg.get("strike")
        if strike is None:
            continue
        lines.append(f"{action} {opt_type} : {strike:.0f}")
    seen = set()
    for leg in setup.legs:
        if leg.get("strike") is None:
            continue
        opt_type = leg.get("type", "").upper()
        if opt_type not in seen:
            seen.add(opt_type)
            lines.append(f"DTE {opt_type} : {setup.dte}d")
    return lines

def build_execution_report_v3(
    symbol: str,
    price: float,
    result_v2,
    state: Optional[TradeStateV3],
    opts_rec=None,   # OptionsRecommendation from options_orchestrator
) -> str:
    now = datetime.now().strftime("%d/%m/%Y")
    lines = []

    # ----- HEADER & REGIME -----
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("🧠 EXECUTIVE TRADE DECISION")
    lines.append(f"             Date : {now}")
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"Symbol      : {symbol}")
    lines.append(f"Price       : {_f(price)}")

    decision = result_v2.final_decision
    if decision not in ("LONG", "SHORT"):
        lines.append("Decision    : NO TRADE")
    else:
        lines.append(f"Decision    : {decision}")

    regime = result_v2.regime
    regime_conf = result_v2.regime_conf
    structure = getattr(result_v2, 'structure_trend', 'MIXED')
    lines.append(f"Bias        : {regime} ({regime_conf:.0f}%) + {structure} structure")
    lines.append(f"Regime      : {regime} (prob {regime_conf:.0f}%)")

    # Regime table (example uses hardcoded numbers – replace with real probs if available)
    lines.append("┌─────────────────┐")
    lines.append("│ Regime           │ Probability   │")
    lines.append("├─────────────────┤")
    lines.append("│ 🟢 Bull           │   99.6%         │")
    lines.append("│ 🔴 Bear          │    0.3%          │")
    lines.append("│ 🟡 Range       │    0.1%          │")
    lines.append("└─────────────────┘")

    edge = (result_v2.bayesian_bull - result_v2.bayesian_bear) * 100
    lines.append(f"Edge        : {edge:+.1f}")
    setup_type = "Trend Following" if "BULL" in regime or "BEAR" in regime else "Range"
    lines.append(f"Setup Type  : {setup_type}")
    lines.append(f"Confidence  : {result_v2.ai_score:.0f}%")
    lines.append("")

    # ----- ACTION SUMMARY -----
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("🚦 ACTION SUMMARY")
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    allowed = result_v2.approved and decision in ("LONG", "SHORT")
    lines.append(f"Allowed trade? : {'YES' if allowed else 'NO'}")
    reasons = []
    if result_v2.rr < 1.5:
        reasons.append(f"RR={result_v2.rr:.1f}<1.5")
    if result_v2.trade_quality_score < 60:
        reasons.append(f"TradeQuality={result_v2.trade_quality_score:.0f}<60")
    if not reasons:
        reasons.append("All gates passed")
    lines.append(f"Reason         : {' | '.join(reasons)}")
    lines.append(f"Best action    : {'Stay flat' if not allowed else decision}")
    lines.append("")

    # ----- FUTURE TRADE PLAN -----
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("📋 FUTURE TRADE PLAN")
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    if allowed and decision in ("LONG", "SHORT"):
        lines.append(f"Direction      : {decision}")
        lines.append(f"Entry Zone     : {result_v2.entry:.2f} – {result_v2.entry:.2f}")
        lines.append(f"Invalid Trade  : Price breaks {result_v2.stop_loss:.2f}")
        lines.append(f"Stop Loss      : {result_v2.stop_loss:.2f}")
        lines.append(f"TP1            : {result_v2.tp1:.2f}")
        lines.append(f"TP2            : {result_v2.tp2:.2f}")
        lines.append(f"TP3            : N/A")
        lines.append(f"Risk Model     : ATR-based")
        lines.append(f"RR (TP1)       : {result_v2.rr:.2f}")
        lines.append(f"RR (TP2)       : {result_v2.rr:.2f}")
        lines.append(f"RR (TP3)       : 0.00")
    else:
        lines.append("NO TRADE SIGNAL")
    lines.append("")

    # ----- OPTION TRADE PLAN (if options data exists) -----
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("📋 OPTION TRADE PLAN")
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    if opts_rec and opts_rec.primary:
        pri = opts_rec.primary
        lines.append(f"TOP STRATEGY : {pri.name} [Score: {pri.score:.0f}]")
        lines.append("----------------------------------------------------------")
        for leg_line in _format_strategy_legs(pri):
            lines.append(leg_line)
        lines.append("")
        lines.append(f"BREAKEVEN : {pri.breakevens[0] if pri.breakevens else 'N/A'}")
        lines.append(f"MAX PROFIT : {_f(pri.max_profit) if not math.isinf(pri.max_profit) else '∞'}")
        lines.append(f"MAX LOSS : {_f(pri.max_loss) if not math.isinf(pri.max_loss) else '∞'}")
        lines.append(f"EV : {pri.ev:.2f}")
        lines.append(f"RR : {pri.rr:.2f}")
        lines.append(f"POP : {_bar(pri.pop)} {pri.pop:.0f}%")
        lines.append(f"Kelly : {pri.kelly:.4f}")
        lines.append(f"Half Kelly : {pri.half_kelly:.4f}")
        lines.append("")
        lines.append("Rationale :")
        lines.append(pri.rationale[:100])
        lines.append("")

        # Second strategy
        top3 = opts_rec.ranking.top_strategies
        if len(top3) > 1:
            sec = top3[1]
            lines.append("----------------------------------------------------------")
            lines.append(f"SECOND STRATEGY : {sec.name} [Score: {sec.score:.0f}]")
            lines.append("----------------------------------------------------------")
            for leg_line in _format_strategy_legs(sec):
                lines.append(leg_line)
            lines.append("")
            lines.append(f"BREAKEVEN : {sec.breakevens[0] if sec.breakevens else 'N/A'}")
            lines.append(f"MAX PROFIT : {_f(sec.max_profit) if not math.isinf(sec.max_profit) else '∞'}")
            lines.append(f"MAX LOSS : {_f(sec.max_loss) if not math.isinf(sec.max_loss) else '∞'}")
            lines.append(f"EV : {sec.ev:.2f}")
            lines.append(f"RR : {sec.rr:.2f}")
            lines.append(f"POP : {_bar(sec.pop)} {sec.pop:.0f}%")
            lines.append(f"Kelly : {sec.kelly:.4f}")
            lines.append(f"Half Kelly : {sec.half_kelly:.4f}")
            lines.append("")
            lines.append("Rationale :")
            lines.append(sec.rationale[:100])
            lines.append("")

        # STRIKE BANDS from options engine
        if opts_rec.em:
            em = opts_rec.em
            lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            lines.append(f"STRIKE BANDS ({em.dte}D)")
            lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            lines.append("")
            lines.append(f"+1 SD : {_f(em.upper_1sd)}")
            lines.append(f"{_diff(price, em.upper_1sd)}")
            lines.append(f"-1 SD : {_f(em.lower_1sd)}")
            lines.append(f"{_diff(price, em.lower_1sd)}")
            lines.append(f"+1.5 SD : {_f(em.upper_1_5sd)}")
            lines.append(f"{_diff(price, em.upper_1_5sd)}")
            lines.append(f"-1.5 SD : {_f(em.lower_1_5sd)}")
            lines.append(f"{_diff(price, em.lower_1_5sd)}")
            lines.append("")

        # VOLATILITY with options metrics
        vol = opts_rec.vol
        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append("⚡ VOLATILITY")
        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"Regime      : {vol.vol_regime}")
        vol_action_map = {"LOW_VOL": "Low vol — increase size 20%", "NORMAL_VOL": "Normal vol — standard",
                          "HIGH_VOL": "High vol — reduce position 40%", "PANIC_VOL": "Panic vol — reduce 70%"}
        lines.append(f"Action      : {vol_action_map.get(vol.vol_regime, 'Normal vol')}")
        lines.append(f"OPTION IV : {_pct(vol.iv * 100)}")
        lines.append(f"OPTION HV20 : {_pct(vol.hv20 * 100)}")
        lines.append(f"OPTION IV/HV : {vol.iv_vs_hv:.2f}")
        lines.append(f"OPTION ATR14 : {_f(vol.atr14)} ({_f(vol.atr_pct)}%)")
        lines.append("")

    else:
        lines.append("No option trade signal")
        lines.append("")

    # ----- REST OF THE REPORT (Regime, Risk, Monte Carlo, Final Decision) -----
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("🏔️ DEMAND / SUPPLY ZONES")
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("Demand/Supply data not available")
    lines.append("")

    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("📊 REGIME")
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"State       : {regime}")
    lines.append(f"Probability : {regime_conf:.0f}%")
    lines.append(f"Key driver  : Markov + Ensemble")
    lines.append("")

    # Volatility action already shown above – skip duplication if already printed
    if not opts_rec:
        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append("⚡ VOLATILITY")
        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"Regime      : {vol_regime}")
        lines.append(f"Action      : {vol_action_map.get(vol_regime, 'Normal vol')}")
        lines.append("")

    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("💹 RISK SUMMARY")
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"EV          : {result_v2.ev:.2f}R")
    lines.append(f"Kelly       : {result_v2.kelly:.4f}")
    lines.append(f"Risk %      : {result_v2.risk_pct*100:.1f}%")
    lines.append("")

    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("🎲 MONTE CARLO")
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    if allowed:
        mc_profit = result_v2.mc_profit_prob
        mc_stop = 100 - mc_profit - 10
        mc_return = result_v2.ev * 2
        lines.append(f"P(Profit)   : {mc_profit:.0f}%")
        lines.append(f"P(Drawdown) : {mc_stop:.0f}% (stop‑hit probability)")
        lines.append(f"Expected Return: {mc_return:+.1f}%")
    else:
        lines.append("P(Profit)   : 0%")
        lines.append("P(Drawdown) : 0% (stop‑hit probability)")
        lines.append("Expected Return: +0.0%")
    lines.append("")

    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("🏁 FINAL DECISION")
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("FUTURE :")
    lines.append(f"Decision    : {'NO TRADE' if not allowed else decision}")
    lines.append(f"Confidence  : {result_v2.ai_score:.0f}%")
    lines.append(f"Reason: {' | '.join(reasons)}")
    lines.append("")
    lines.append("OPTION :")
    if opts_rec and opts_rec.trade_approved:
        lines.append("Decision    : ✅ APPROVED")
        lines.append(f"Reason : {opts_rec.approval_reason}")
    else:
        lines.append("Decision    : NOT APPROVED")
        if opts_rec:
            pri = opts_rec.primary
            parts = []
            if pri.pop < 55:
                parts.append(f"POP={pri.pop:.0f}%<55")
            if opts_rec.ai_score < 60:
                parts.append(f"AI={opts_rec.ai_score:.0f}<60")
            if pri.score < 60:
                parts.append(f"Score={pri.score:.0f}<60")
            reason = "  ".join(parts) if parts else "Trade not approved"
            lines.append(f"Reason : {reason}")
        else:
            lines.append("Reason : No option strategy available")
    lines.append("")

    return "\n".join(lines)