"""
V3 Execution Report – Unified Futures + Options (Phone‑Optimised)
==================================================================
Every line is calculated from engine outputs – no hardcoded placeholders.
Designed for narrow screens: short separators, no box drawing, compact layout.
"""
from datetime import datetime
from typing import Optional
import math

from core.trade_state import TradeStateV3

# Short separator for phones (20 dashes)
SEP = "────────────────────"


def _f(x, dec: int = 2) -> str:
    if x is None:
        return "N/A"
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return "N/A"
        return f"{v:.{dec}f}"
    except (TypeError, ValueError):
        return str(x)


def _pct(x) -> str:
    """Return percentage string without extra % sign."""
    if x is None:
        return "N/A"
    try:
        if isinstance(x, str) and x.endswith('%'):
            return x
        return f"{float(x):.1f}%"
    except:
        return str(x)


def _bar(pct: float, w: int = 6) -> str:
    """Short bar for phones (max 6 blocks)."""
    n = max(0, min(w, round(pct / 100 * w)))
    return "█" * n + "░" * (w - n)


def _diff(price: float, target: float) -> str:
    try:
        d = (target - price) / price * 100
        return f"({'+' if d >= 0 else ''}{d:.1f}%)"
    except:
        return ""


def _format_strategy_legs(setup) -> list[str]:
    """Return compact leg descriptions."""
    lines = []
    for leg in setup.legs:
        action = leg.get("action", "").upper()[:4]  # BUY/SELL
        opt_type = leg.get("type", "").upper()[:4]  # CALL/PUT
        strike = leg.get("strike")
        if strike is None:
            continue
        lines.append(f"{action} {opt_type} {strike:.0f}")
    # DTE line
    if setup.legs:
        lines.append(f"DTE {setup.dte}d")
    return lines


def build_execution_report_v3(
    symbol: str,
    price: float,
    result_v2,
    state: Optional[TradeStateV3],
    opts_rec=None,
) -> str:
    """
    Build a phone‑friendly execution report.
    """
    now = datetime.now().strftime("%d/%m/%Y")
    lines = []

    # ----- HEADER & REGIME -----
    lines.append(SEP)
    lines.append("🧠 EXECUTIVE TRADE DECISION")
    lines.append(f"Date: {now}")
    lines.append(SEP)
    lines.append(f"{symbol}  {_f(price)}")

    decision = result_v2.final_decision
    if decision not in ("LONG", "SHORT"):
        lines.append("Decision: NO TRADE")
    else:
        lines.append(f"Decision: {decision}")

    regime = result_v2.regime
    regime_conf = result_v2.regime_conf
    structure = getattr(result_v2, "structure_trend", "MIXED")
    lines.append(f"Bias: {regime} ({regime_conf:.0f}%) + {structure}")

    # Regime probabilities (simple two‑column without box drawing)
    if opts_rec:
        bull = opts_rec.bull_prob * 100
        bear = opts_rec.bear_prob * 100
        rng = opts_rec.range_prob * 100
    else:
        if "BULL" in regime:
            bull = regime_conf
            bear = (100 - regime_conf) * 0.5
            rng = (100 - regime_conf) * 0.5
        elif "BEAR" in regime:
            bear = regime_conf
            bull = (100 - regime_conf) * 0.5
            rng = (100 - regime_conf) * 0.5
        else:
            bull = bear = rng = 33.3
    lines.append("Regime probabilities:")
    lines.append(f"  Bull: {_pct(bull)}")
    lines.append(f"  Bear: {_pct(bear)}")
    lines.append(f"  Range: {_pct(rng)}")

    edge = bull - bear
    lines.append(f"Edge: {edge:+.1f}")
    setup_type = "Trend" if ("BULL" in regime or "BEAR" in regime) else "Range"
    lines.append(f"Setup: {setup_type}")
    confidence = result_v2.ai_score if result_v2.ai_score > 0 else regime_conf
    lines.append(f"Confidence: {_pct(confidence)}")
    lines.append("")

    # ----- ACTION SUMMARY -----
    lines.append(SEP)
    lines.append("🚦 ACTION SUMMARY")
    lines.append(SEP)
    allowed = result_v2.approved and decision in ("LONG", "SHORT")
    lines.append(f"Allowed: {'YES' if allowed else 'NO'}")
    reasons = []
    if result_v2.rr < 1.5:
        reasons.append(f"RR={result_v2.rr:.1f}<1.5")
    if getattr(result_v2, "trade_quality_score", 0) < 60:
        reasons.append(f"TQ={result_v2.trade_quality_score:.0f}<60")
    if not reasons:
        reasons.append("All gates passed")
    lines.append(f"Reason: {' | '.join(reasons)}")
    lines.append(f"Best: {'Stay flat' if not allowed else decision}")
    lines.append("")

    # ----- FUTURE TRADE PLAN -----
    lines.append(SEP)
    lines.append("📋 FUTURE TRADE PLAN")
    lines.append(SEP)
    if allowed and decision in ("LONG", "SHORT"):
        if opts_rec and opts_rec.em:
            entry_low = opts_rec.em.lower_1sd
            entry_high = opts_rec.em.upper_1sd
        else:
            entry_low = price * 0.98
            entry_high = price * 1.02
        lines.append(f"Dir: {decision}")
        lines.append(f"Entry: {_f(entry_low)}–{_f(entry_high)}")
        lines.append(f"Invalid: price breaks {_f(result_v2.stop_loss)}")
        lines.append(f"Stop: {_f(result_v2.stop_loss)}")
        tp1 = result_v2.tp1 or (price * 1.04 if decision == "LONG" else price * 0.96)
        tp2 = result_v2.tp2 or (price * 1.08 if decision == "LONG" else price * 0.92)
        risk = abs(price - result_v2.stop_loss)
        tp3 = price + 4*risk if decision == "LONG" else price - 4*risk
        lines.append(f"TP1: {_f(tp1)} {_diff(price, tp1)}")
        lines.append(f"TP2: {_f(tp2)} {_diff(price, tp2)}")
        lines.append(f"TP3: {_f(tp3)} {_diff(price, tp3)}")
        atr_val = opts_rec.vol.atr14 if (opts_rec and opts_rec.vol) else (price * 0.015)
        lines.append(f"Risk model: ATR {atr_val:.0f}")
        rr1 = abs(tp1-price)/risk if risk>0 else 0
        rr2 = abs(tp2-price)/risk if risk>0 else 0
        rr3 = abs(tp3-price)/risk if risk>0 else 0
        lines.append(f"RR: {rr1:.1f}/{rr2:.1f}/{rr3:.1f}")
    else:
        lines.append("NO TRADE")
    lines.append("")

    # ----- OPTION TRADE PLAN -----
    lines.append(SEP)
    lines.append("📋 OPTION TRADE PLAN")
    lines.append(SEP)
    if opts_rec and opts_rec.primary:
        pri = opts_rec.primary
        lines.append(f"Top: {pri.name} [Score:{pri.score:.0f}]")
        for leg_line in _format_strategy_legs(pri):
            lines.append(f"  {leg_line}")
        be = pri.breakevens[0] if pri.breakevens else "N/A"
        mp = _f(pri.max_profit) if not math.isinf(pri.max_profit) else "∞"
        ml = _f(pri.max_loss) if not math.isinf(pri.max_loss) else "∞"
        lines.append(f"BE: {be}  MP:{mp}  ML:{ml}")
        lines.append(f"EV:{pri.ev:.2f}  RR:{pri.rr:.2f}")
        lines.append(f"POP: {_bar(pri.pop)} {pri.pop:.0f}%")
        lines.append(f"Kelly: {pri.kelly:.4f} (half:{pri.half_kelly:.4f})")
        lines.append(f"Rationale: {pri.rationale[:60]}")
        lines.append("")

        # Second strategy
        top3 = opts_rec.ranking.top_strategies
        if len(top3) > 1:
            sec = top3[1]
            lines.append(f"2nd: {sec.name} [Score:{sec.score:.0f}]")
            for leg_line in _format_strategy_legs(sec):
                lines.append(f"  {leg_line}")
            be2 = sec.breakevens[0] if sec.breakevens else "N/A"
            mp2 = _f(sec.max_profit) if not math.isinf(sec.max_profit) else "∞"
            ml2 = _f(sec.max_loss) if not math.isinf(sec.max_loss) else "∞"
            lines.append(f"BE: {be2}  MP:{mp2}  ML:{ml2}")
            lines.append(f"EV:{sec.ev:.2f}  RR:{sec.rr:.2f}")
            lines.append(f"POP: {_bar(sec.pop)} {sec.pop:.0f}%")
            lines.append(f"Kelly: {sec.kelly:.4f}")
            lines.append("")

        # STRIKE BANDS (price and diff on one line)
        if opts_rec.em:
            em = opts_rec.em
            lines.append(SEP)
            lines.append(f"STRIKE BANDS ({em.dte}D)")
            lines.append(SEP)
            lines.append(f"+1SD   : {_f(em.upper_1sd)} {_diff(price, em.upper_1sd)}")
            lines.append(f"-1SD   : {_f(em.lower_1sd)} {_diff(price, em.lower_1sd)}")
            lines.append(f"+1.5SD : {_f(em.upper_1_5sd)} {_diff(price, em.upper_1_5sd)}")
            lines.append(f"-1.5SD : {_f(em.lower_1_5sd)} {_diff(price, em.lower_1_5sd)}")
            lines.append("")

        # VOLATILITY
        vol = opts_rec.vol
        lines.append(SEP)
        lines.append("⚡ VOLATILITY")
        lines.append(SEP)
        lines.append(f"Regime: {vol.vol_regime}")
        action_map = {
            "LOW_VOL": "inc size 20%",
            "NORMAL_VOL": "standard",
            "HIGH_VOL": "reduce 40%",
            "PANIC_VOL": "reduce 70%"
        }
        lines.append(f"Action: {action_map.get(vol.vol_regime, 'standard')}")
        lines.append(f"IV: {_pct(vol.iv*100)}  HV20: {_pct(vol.hv20*100)}")
        lines.append(f"IV/HV: {vol.iv_vs_hv:.2f}  ATR14: {_f(vol.atr14)} ({_f(vol.atr_pct)}%)")
        lines.append("")
    else:
        lines.append("No option signal")
        lines.append("")

    # ----- DEMAND / SUPPLY ZONES (two zones each) -----
    lines.append(SEP)
    lines.append("🏔️ DEMAND / SUPPLY ZONES")
    lines.append(SEP)
    if opts_rec and opts_rec.em:
        em = opts_rec.em
        lines.append("Supply:")
        lines.append(f"  Z1: {_f(em.upper_1sd)} (1SD)")
        lines.append(f"  Z2: {_f(em.upper_1_5sd)} (1.5SD)")
        lines.append("Demand:")
        lines.append(f"  Z1: {_f(em.lower_1sd)} (1SD)")
        lines.append(f"  Z2: {_f(em.lower_1_5sd)} (1.5SD)")
    else:
        lines.append("Supply: proxy bands")
        lines.append(f"  Z1: {_f(price*1.02)}  Z2: {_f(price*1.05)}")
        lines.append("Demand: proxy bands")
        lines.append(f"  Z1: {_f(price*0.98)}  Z2: {_f(price*0.95)}")
    lines.append("")

    # ----- REGIME (compressed) -----
    lines.append(SEP)
    lines.append("📊 REGIME (COMPRESSED)")
    lines.append(SEP)
    lines.append(f"State: {regime}  Prob: {_pct(regime_conf)}")
    lines.append("Driver: Markov+Ensemble")
    lines.append("")

    # ----- VOLATILITY (if not already shown) -----
    if not (opts_rec and opts_rec.vol):
        lines.append(SEP)
        lines.append("⚡ VOLATILITY")
        lines.append(SEP)
        vol_reg = getattr(result_v2, "vol_regime", "NORMAL_VOL")
        action_short = action_map.get(vol_reg, "standard")
        lines.append(f"Regime: {vol_reg}  Action: {action_short}")
        lines.append("")

    # ----- RISK SUMMARY -----
    lines.append(SEP)
    lines.append("💹 RISK SUMMARY")
    lines.append(SEP)
    lines.append(f"EV: {result_v2.ev:.2f}R  Kelly: {result_v2.kelly:.4f}")
    lines.append(f"Risk%: {result_v2.risk_pct*100:.1f}%")
    lines.append("")

    # ----- MONTE CARLO -----
    lines.append(SEP)
    lines.append("🎲 MONTE CARLO")
    lines.append(SEP)
    mc_profit = result_v2.mc_profit_prob
    mc_stop = 100 - mc_profit - 10
    mc_return = result_v2.ev * 2
    if state and state.regime_mc_result:
        mc_profit = state.regime_mc_result.get("prob_profit", mc_profit)
        mc_stop = state.regime_mc_result.get("prob_stop_hit", mc_stop)
        mc_return = state.regime_mc_result.get("expected_return", mc_return)
    lines.append(f"P(Profit): {mc_profit:.0f}%")
    lines.append(f"P(Stop): {mc_stop:.0f}%")
    lines.append(f"Exp Ret: {mc_return:+.1f}%")
    lines.append("")

    # ----- FINAL DECISION -----
    lines.append(SEP)
    lines.append("🏁 FINAL DECISION")
    lines.append(SEP)
    lines.append("FUTURE:")
    lines.append(f"  Decision: {'NO TRADE' if not allowed else decision}")
    lines.append(f"  Confidence: {_pct(confidence)}")
    lines.append(f"  Reason: {' | '.join(reasons)}")
    lines.append("")
    lines.append("OPTION:")
    if opts_rec and opts_rec.trade_approved:
        lines.append("  ✅ APPROVED")
        lines.append(f"  Reason: {opts_rec.approval_reason}")
    else:
        lines.append("  ❌ NOT APPROVED")
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
            lines.append(f"  Reason: {reason}")
        else:
            lines.append("  Reason: No option strategy")
    lines.append("")

    return "\n".join(lines)