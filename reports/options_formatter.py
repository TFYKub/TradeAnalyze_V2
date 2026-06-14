"""
Options LINE Message Formatter – Institutional Execution Style
================================================================
Formats OptionsRecommendation into a clean, legible report suitable
for LINE notifications. Only output formatting is changed; all
calculations (volatility, POP, EV, Kelly, selection, approval) stay
exactly as in the existing engine.
"""
from __future__ import annotations
import math
from datetime import datetime

# ---------- Helper functions (kept from original) ----------
def _f(x, dec: int = 2) -> str:
    if x is None:
        return "N/A"
    try:
        v = float(x)
        return "N/A" if (math.isnan(v) or math.isinf(v)) else f"{v:.{dec}f}"
    except (TypeError, ValueError):
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

def _s(x) -> str:
    return "N/A" if (x is None or x == "") else str(x)

def _diff(price, target) -> str:
    try:
        d = (float(target) - float(price)) / float(price) * 100
        return f"({'+' if d >= 0 else ''}{d:.1f}%)"
    except:
        return ""

# ---------- Leg display helper ----------
def _format_strategy_legs(setup) -> list[str]:
    """
    Returns a list of lines describing the active legs of a strategy.
    Only legs with a non‑None strike are shown.
    """
    lines = []
    for leg in setup.legs:
        action = leg.get("action", "").upper()
        opt_type = leg.get("type", "").upper()
        strike = leg.get("strike")
        if strike is None:
            continue
        # Format: "BUY CALL : 65000" or "SELL PUT : 60000"
        lines.append(f"{action} {opt_type} : {strike:.0f}")
    return lines

def _format_strategy_dte(setup) -> list[str]:
    """
    Returns a list of lines with DTE information for each leg that has a DTE.
    For simplicity, we assume the same DTE for all legs (stored in setup.dte).
    """
    dte = setup.dte
    lines = []
    # We output one DTE line per leg type? The spec shows separate lines.
    # To keep it clean, we output a single line per leg that exists.
    for leg in setup.legs:
        opt_type = leg.get("type", "").upper()
        if leg.get("strike") is not None:
            lines.append(f"DTE {opt_type} : {dte}d")
    return lines

# ---------- Main formatting function ----------
def format_options_message(rec) -> str:
    """
    Returns a fully formatted LINE message for options recommendations.
    rec: OptionsRecommendation from options_orchestrator.py
    """
    now = datetime.now().strftime("%d/%m/%Y")
    lines = []

    # ==================================================
    # HEADER
    # ==================================================
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"🐱 OPTIONS ANALYSIS | {rec.symbol} | {now}")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("")
    lines.append(f"Price : {_f(rec.price)}")
    lines.append("")

    # ==================================================
    # SUMMARY
    # ==================================================
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("SUMMARY")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("")
    lines.append(f"REGIME : {rec.regime} (conf {rec.regime_conf:.0f}%)")
    lines.append("")
    lines.append(f"Bull : {_pct(rec.bull_prob*100)}")
    lines.append(f"Bear : {_pct(rec.bear_prob*100)}")
    lines.append(f"Range : {_pct(rec.range_prob*100)}")
    lines.append("")
    lines.append(f"IV Rank : {_pct(rec.vol.iv_rank)}")
    lines.append("")
    lines.append("Expected Move :")
    lines.append(f"±{_f(rec.em.expected_move)}")
    lines.append(f"({_f(rec.em.expected_move_pct)}%)")
    lines.append(f"[{rec.em.dte}D]")
    lines.append("")

    # ==================================================
    # TOP STRATEGY
    # ==================================================
    primary = rec.primary
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"TOP STRATEGY : {primary.name} [Score: {primary.score:.0f}]")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # Legs (only active)
    leg_lines = _format_strategy_legs(primary)
    for line in leg_lines:
        lines.append(line)
    # DTE lines (only for legs that exist)
    dte_lines = _format_strategy_dte(primary)
    for line in dte_lines:
        lines.append(line)

    lines.append("")
    lines.append(f"BREAKEVEN : {primary.breakevens[0] if primary.breakevens else 'N/A'}")
    lines.append(f"MAX PROFIT : {_f(primary.max_profit) if not math.isinf(primary.max_profit) else '∞'}")
    lines.append(f"MAX LOSS : {_f(primary.max_loss) if not math.isinf(primary.max_loss) else '∞'}")
    lines.append(f"EV : {primary.ev:.2f}")
    lines.append(f"RR : {primary.rr:.2f}")
    pop_bar = _bar(primary.pop)
    lines.append(f"POP : {pop_bar} {primary.pop:.0f}%")
    lines.append(f"Kelly : {primary.kelly:.4f}")
    lines.append(f"Half Kelly : {primary.half_kelly:.4f}")
    lines.append("")
    lines.append("Rationale :")
    lines.append(primary.rationale[:100])
    lines.append("")

    # ==================================================
    # SECOND STRATEGY (if exists)
    # ==================================================
    top3 = rec.ranking.top_strategies
    if len(top3) > 1:
        second = top3[1]
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"SECOND STRATEGY : {second.name} [Score: {second.score:.0f}]")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        leg_lines2 = _format_strategy_legs(second)
        for line in leg_lines2:
            lines.append(line)
        dte_lines2 = _format_strategy_dte(second)
        for line in dte_lines2:
            lines.append(line)
        lines.append("")
        lines.append(f"BREAKEVEN : {second.breakevens[0] if second.breakevens else 'N/A'}")
        lines.append(f"MAX PROFIT : {_f(second.max_profit) if not math.isinf(second.max_profit) else '∞'}")
        lines.append(f"MAX LOSS : {_f(second.max_loss) if not math.isinf(second.max_loss) else '∞'}")
        lines.append(f"EV : {second.ev:.2f}")
        lines.append(f"RR : {second.rr:.2f}")
        pop_bar2 = _bar(second.pop)
        lines.append(f"POP : {pop_bar2} {second.pop:.0f}%")
        lines.append(f"Kelly : {second.kelly:.4f}")
        lines.append(f"Half Kelly : {second.half_kelly:.4f}")
        lines.append("")
        lines.append("Rationale :")
        lines.append(second.rationale[:100])
        lines.append("")

    # ==================================================
    # VOLATILITY
    # ==================================================
    vol = rec.vol
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("VOLATILITY")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("")
    lines.append(f"IV : {_pct(vol.iv * 100)}")
    lines.append(f"HV20 : {_pct(vol.hv20 * 100)}")
    lines.append(f"IV/HV : {vol.iv_vs_hv:.2f}")
    lines.append(f"ATR14 : {_f(vol.atr14)}")
    lines.append(f"({_f(vol.atr_pct)}%)")
    lines.append(f"Vol Regime : {vol.vol_regime}")
    lines.append("")

    # ==================================================
    # STRIKE BANDS
    # ==================================================
    em = rec.em
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"STRIKE BANDS ({em.dte}D)")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("")
    lines.append(f"Price : {_f(rec.price)}")
    lines.append("")
    plus1_pct = _diff(rec.price, em.upper_1sd)
    lines.append(f"+1 SD : {_f(em.upper_1sd)}")
    lines.append(f"({plus1_pct})")
    minus1_pct = _diff(rec.price, em.lower_1sd)
    lines.append(f"-1 SD : {_f(em.lower_1sd)}")
    lines.append(f"({minus1_pct})")
    plus15_pct = _diff(rec.price, em.upper_1_5sd)
    lines.append(f"+1.5 SD : {_f(em.upper_1_5sd)}")
    lines.append(f"({plus15_pct})")
    minus15_pct = _diff(rec.price, em.lower_1_5sd)
    lines.append(f"-1.5 SD : {_f(em.lower_1_5sd)}")
    lines.append(f"({minus15_pct})")
    lines.append("")

    # ==================================================
    # FINAL DECISION
    # ==================================================
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("🏁 FINAL DECISION")
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("")
    if rec.trade_approved:
        lines.append("✅ APPROVED")
        lines.append("")
        lines.append(f"Strategy : {primary.name}")
        lines.append(f"POP : {primary.pop:.0f}%")
        lines.append(f"AI Score : {rec.ai_score:.0f}")
        lines.append("")
        lines.append("Reason :")
        lines.append(rec.approval_reason)
    else:
        lines.append("⏸️ NOT APPROVED")
        lines.append("")
        # Build a concise reason
        parts = []
        if primary.pop < 55:
            parts.append(f"POP={primary.pop:.0f}%<55")
        if rec.ai_score < 60:
            parts.append(f"AI={rec.ai_score:.0f}<60")
        if primary.ev <= 0:
            parts.append(f"EV={primary.ev:.2f}≤0")
        if primary.score < 60:
            parts.append(f"Score={primary.score:.0f}<60")
        reason = "  ".join(parts) if parts else rec.approval_reason
        lines.append("Reason :")
        lines.append(reason)

    lines.append("")
    return "\n".join(lines)