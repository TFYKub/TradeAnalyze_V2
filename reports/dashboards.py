"""
Institutional Dashboards  (Report Extension — Phases 12–21)
=============================================================
Generates formatted text dashboard sections for each new engine.
Appended to the existing daily_report output by futures_orchestrator_v2.

Sections:
  1. Liquidity Dashboard        (Phase 12)
  2. Flow Dashboard             (Phase 13)
  3. Breadth Dashboard          (Phase 14)
  4. Regime Persistence         (Phase 15)
  5. Forecast Dashboard         (Phase 17)
  6. Conviction Dashboard       (Phase 18)
  7. Portfolio Risk Dashboard   (Phase 20)
  8. Cross Asset Dashboard      (Phase 21)

Each function returns a str block.
build_institutional_dashboards() concatenates all available sections.
All functions are safe — return "N/A" section if result is None.
"""
from __future__ import annotations

from typing import Optional

SEP  = "─" * 60
SEP2 = "═" * 60
NL   = "\n"


# ── Helpers ────────────────────────────────────────────────────────────────────
def _bar(score: float, width: int = 20) -> str:
    """Visual progress bar: score 0–100 → ████░░░░ style."""
    filled = max(0, min(width, int(score / 100 * width)))
    return "█" * filled + "░" * (width - filled)


def _pct(v: float, decimals: int = 1) -> str:
    return f"{v:.{decimals}f}%"


def _sign(v: float) -> str:
    return f"{v:+.2f}"


# ── 1. Liquidity Dashboard ─────────────────────────────────────────────────────
def _liquidity_dashboard(liq) -> str:
    if liq is None:
        return f"  [Liquidity]  Data unavailable\n"

    regime  = liq.liquidity_regime
    score   = liq.score
    conf    = liq.confidence
    mult    = liq.risk_multiplier
    vix     = liq.vix_level
    dxy_t   = liq.dxy_trend
    yld_t   = liq.yield_trend
    tlt_v   = liq.tlt_vol_pct
    bar     = _bar(score)
    comps   = liq.component_scores

    emoji = {"RISK_ON": "✅", "RISK_OFF": "⚠️", "CRISIS": "🚨", "RECOVERY": "🔄"}.get(regime, "❓")

    return (
        f"  ┌── 🌊 LIQUIDITY REGIME ──────────────────────────────────┐\n"
        f"  │  Regime     : {emoji} {regime:<20}  Conf: {conf:.0f}%\n"
        f"  │  Score      : [{bar}] {score:.0f}/100\n"
        f"  │  Risk Mult  : ×{mult:.2f}   VIX: {vix:.1f}  TLT-Vol: {tlt_v:.1f}%\n"
        f"  │  DXY Trend  : {dxy_t:<12}  Yield Trend: {yld_t}\n"
        f"  │  Components : VIX={comps.get('vix',0):.0f} DXY={comps.get('dxy',0):.0f} "
        f"Yield={comps.get('yield',0):.0f} TLT={comps.get('tlt_vol',0):.0f}\n"
        f"  └────────────────────────────────────────────────────────┘\n"
    )


# ── 2. Flow Dashboard ──────────────────────────────────────────────────────────
def _flow_dashboard(flow) -> str:
    if flow is None:
        return f"  [Flow]  Data unavailable\n"

    regime  = flow.flow_regime
    score   = flow.flow_score
    conf    = flow.flow_confidence
    direc   = flow.flow_direction
    fr_pct  = flow.funding_rate_pct
    ls_r    = flow.ls_ratio
    cascade = flow.cascade_risk
    bar     = _bar(score)

    emoji = {
        "BULLISH_FLOW": "🟢", "BEARISH_FLOW": "🔴",
        "SHORT_SQUEEZE": "🚀", "LONG_SQUEEZE": "💥", "NEUTRAL": "⚪"
    }.get(regime, "❓")

    return (
        f"  ┌── 🌀 DERIVATIVES FLOW ──────────────────────────────────┐\n"
        f"  │  Regime     : {emoji} {regime:<22}  Conf: {conf:.0f}%\n"
        f"  │  Direction  : {direc:<12}  Score: [{bar}] {score:.0f}/100\n"
        f"  │  Funding    : {fr_pct:+.4f}%    L/S Ratio: {ls_r:.2f}\n"
        f"  │  OI Signal  : {flow.oi_signal:<15}  Cascade: {cascade}\n"
        f"  │  Comp Scores: Fund={flow.funding_score:.0f} OI={flow.oi_score:.0f} "
        f"LS={flow.ls_score:.0f} Liq={flow.liq_score:.0f}\n"
        f"  └────────────────────────────────────────────────────────┘\n"
    )


# ── 3. Breadth Dashboard ───────────────────────────────────────────────────────
def _breadth_dashboard(br) -> str:
    if br is None:
        return f"  [Breadth]  Data unavailable\n"

    regime  = br.breadth_regime
    score   = br.breadth_score
    conf    = br.breadth_confidence
    bar     = _bar(score)
    cb      = br.crypto_breadth
    eb      = br.equity_breadth

    emoji = {
        "STRONG_BULL": "🚀", "BULL": "📈",
        "NEUTRAL": "↔️", "BEAR": "📉", "STRONG_BEAR": "🔻"
    }.get(regime, "❓")

    return (
        f"  ┌── 📊 MARKET BREADTH ────────────────────────────────────┐\n"
        f"  │  Regime     : {emoji} {regime:<22}  Conf: {conf:.0f}%\n"
        f"  │  Score      : [{bar}] {score:.0f}/100\n"
        f"  │  ── Crypto ──\n"
        f"  │  BTC.D Est  : {cb.btc_dominance_est:.1f}%  ({cb.btc_dom_trend})\n"
        f"  │  Altcoins   : {cb.total3_change_pct:+.1f}% 20d  ({cb.total3_trend})\n"
        f"  │  ── Equity ──\n"
        f"  │  A/D Ratio  : {eb.advance_decline_ratio:.2f}   "
        f"Above 200DMA: {eb.pct_above_200dma:.0f}%\n"
        f"  │  New Highs  : {eb.new_high_count}  New Lows: {eb.new_low_count}  "
        f"Quality: {br.data_quality}\n"
        f"  └────────────────────────────────────────────────────────┘\n"
    )


# ── 4. Regime Persistence Dashboard ───────────────────────────────────────────
def _persistence_dashboard(per) -> str:
    if per is None:
        return f"  [Persistence]  Data unavailable\n"

    label    = per.persistence_label
    score    = per.persistence_score
    exp_d    = per.expected_duration_days
    remain   = per.remaining_duration_days
    hl       = per.regime_half_life_days
    ex7d     = per.exit_prob_7d
    ex14d    = per.exit_prob_14d
    nxt      = per.most_likely_next
    p_self   = per.self_transition_prob
    bar      = _bar(score)

    urgency  = "⚠️ HIGH" if ex7d > 50 else "🟡 MOD" if ex7d > 30 else "🟢 LOW"

    return (
        f"  ┌── ⏱️  REGIME PERSISTENCE ──────────────────────────────┐\n"
        f"  │  Regime     : {per.regime}  [{label}]\n"
        f"  │  Stability  : [{bar}] {score:.0f}/100\n"
        f"  │  E[Duration]: {exp_d:.0f}d  Half-life: {hl:.0f}d  P(self): {p_self:.3f}\n"
        f"  │  Remaining  : ~{remain:.0f}d  Exit 7d: {ex7d:.0f}%  Exit 14d: {ex14d:.0f}%\n"
        f"  │  Exit Risk  : {urgency}   Next→ {nxt}\n"
        f"  └────────────────────────────────────────────────────────┘\n"
    )


# ── 5. Forecast Dashboard ──────────────────────────────────────────────────────
def _forecast_dashboard(fc) -> str:
    if fc is None:
        return f"  [Forecast]  Data unavailable\n"

    direc = fc.forecast_direction
    conf  = fc.forecast_confidence
    model = fc.model_used
    bar   = _bar(conf)
    hf    = fc.horizon_forecasts

    r5   = hf.get("5d",  {})
    r10  = hf.get("10d", {})
    r20  = hf.get("20d", {})

    emoji = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "⚪"}.get(direc, "❓")

    def _row(label, d):
        ret  = d.get("return_pct", 0.0)
        pup  = d.get("prob_up", 0.5)
        pbar = _bar(pup * 100, 10)
        return f"  │  {label:<5}: {ret:+6.1f}%  P(↑)={pup:.0%}  [{pbar}]\n"

    # Top features
    top_feat = sorted(fc.feature_importances.items(), key=lambda x: -x[1])[:3] if fc.feature_importances else []
    feat_str  = "  ".join(f"{k}={v:.3f}" for k, v in top_feat) if top_feat else "N/A"

    return (
        f"  ┌── 🔮 ML FORECAST ENGINE ({model}) ─────────────────────┐\n"
        f"  │  Direction  : {emoji} {direc:<12}  Confidence: [{bar}] {conf:.0f}%\n"
        f"  │  ── Horizon Forecasts ──\n"
        + _row("5d", r5) + _row("10d", r10) + _row("20d", r20) +
        f"  │  Top Features: {feat_str}\n"
        f"  └────────────────────────────────────────────────────────┘\n"
    )


# ── 6. Conviction Dashboard ────────────────────────────────────────────────────
def _conviction_dashboard(conv) -> str:
    if conv is None:
        return f"  [Conviction]  Data unavailable\n"

    score    = conv.conviction_score
    tier     = conv.conviction_tier
    kelly    = conv.kelly_multiplier
    weak     = conv.weakest_signal
    strong   = conv.strongest_signal
    aligned  = conv.alignment_count
    bar      = _bar(score)

    tier_emoji = {
        "FULL SIZE": "🏆", "NORMAL SIZE": "✅",
        "HALF SIZE": "⚡", "NO TRADE": "⏸️"
    }.get(tier, "❓")

    comp_lines = ""
    for name, s in conv.component_scores.items():
        mini_bar = _bar(s, 10)
        comp_lines += f"  │    {name:<12}: [{mini_bar}] {s:.0f}\n"

    return (
        f"  ┌── 🎯 CONVICTION ENGINE ────────────────────────────────┐\n"
        f"  │  Score      : [{bar}] {score:.0f}/100\n"
        f"  │  Tier       : {tier_emoji} {tier:<16}  Kelly×{kelly:.2f}\n"
        f"  │  Aligned    : {aligned}/8 signals  |  Weak: {weak}  Strong: {strong}\n"
        f"  │  ── Component Breakdown ──\n"
        + comp_lines +
        f"  └────────────────────────────────────────────────────────┘\n"
    )


# ── 7. Portfolio Risk Dashboard ────────────────────────────────────────────────
def _portfolio_dashboard(opt) -> str:
    if opt is None:
        return f"  [Portfolio]  Data unavailable\n"

    method   = opt.method_used
    vol      = opt.portfolio_volatility
    sharpe   = opt.portfolio_sharpe
    dd       = opt.portfolio_drawdown
    div      = opt.diversification_ratio
    n        = opt.n_assets

    weight_lines = ""
    for sym, w in opt.recommended_weights.items():
        rc       = opt.risk_contributions.get(sym, 0)
        vol_s    = opt.volatilities.get(sym, 0)
        ret_s    = opt.expected_returns.get(sym, 0)
        mini_bar = _bar(w * 100, 12)
        weight_lines += (
            f"  │    {sym:<8}: [{mini_bar}] {w:.1%}  "
            f"Risk%={rc:.0f}%  Vol={vol_s:.0f}%  E[R]={ret_s:+.0f}%\n"
        )

    return (
        f"  ┌── 💼 PORTFOLIO OPTIMIZER ({method}) ────────────────────┐\n"
        f"  │  Assets     : {n}   Port Vol: {vol:.1f}%   Sharpe: {sharpe:.2f}\n"
        f"  │  Max DD est : {dd:.1f}%   Div Ratio: {div:.2f}\n"
        f"  │  ── Recommended Weights ──\n"
        + weight_lines +
        f"  └────────────────────────────────────────────────────────┘\n"
    )


# ── 8. Cross Asset Dashboard ───────────────────────────────────────────────────
def _cross_asset_dashboard(ca) -> str:
    if ca is None:
        return f"  [Cross-Asset]  Data unavailable\n"

    regime   = ca.cross_asset_regime
    rs       = ca.relative_strength_score
    conf     = ca.regime_confidence
    decouple = ca.decoupling_detected
    dec_dir  = ca.decoupling_direction
    beta_spy = ca.btc_beta_to_spy
    beta_qqq = ca.btc_beta_to_qqq
    driver   = ca.dominant_driver
    bar      = _bar(rs)

    emoji = {
        "RISK_ON_ALIGNED": "🟢", "RISK_OFF_ALIGNED": "🔴",
        "DECOUPLED_BULL": "🚀", "DECOUPLED_BEAR": "⚠️", "TRANSITION": "↔️"
    }.get(regime, "❓")

    dec_str = f"⚡ DECOUPLED ({dec_dir})" if decouple else "Correlated"

    pair_lines = ""
    for name, pa in ca.pair_analyses.items():
        corr_bar  = _bar((pa.rolling_corr + 1) / 2 * 100, 8)
        out_emoji = "↑" if pa.outperforming else "↓"
        pair_lines += (
            f"  │    BTC/{name:<4}: Corr={pa.rolling_corr:+.2f} [{corr_bar}]  "
            f"BTC={pa.btc_return_20d:+.1f}%  {out_emoji}  {pa.signal}\n"
        )

    return (
        f"  ┌── 🌐 CROSS ASSET ENGINE ────────────────────────────────┐\n"
        f"  │  Regime     : {emoji} {regime:<28}  Conf: {conf:.0f}%\n"
        f"  │  Rel Strength: [{bar}] {rs:.0f}/100  {dec_str}\n"
        f"  │  β_SPY={beta_spy:.2f}  β_QQQ={beta_qqq:.2f}  Dominant Driver: {driver}\n"
        f"  │  ── Pair Analysis (20d) ──\n"
        + pair_lines +
        f"  └────────────────────────────────────────────────────────┘\n"
    )


# ── Master Builder ─────────────────────────────────────────────────────────────
def build_institutional_dashboards(
    liquidity    = None,
    flow         = None,
    breadth      = None,
    persistence  = None,
    forecast     = None,
    conviction   = None,
    portfolio_opt= None,
    cross_asset  = None,
) -> str:
    """
    Build all institutional dashboard sections as a single formatted string.
    Appended to existing report_text by futures_orchestrator_v2.

    All parameters optional — unavailable sections display graceful placeholder.
    """
    sections = [
        f"\n{SEP2}",
        f"  ⚡ INSTITUTIONAL ANALYSIS — PHASES 12–21",
        SEP2,
        "",
        _liquidity_dashboard(liquidity),
        _flow_dashboard(flow),
        _breadth_dashboard(breadth),
        _persistence_dashboard(persistence),
        _forecast_dashboard(forecast),
        _conviction_dashboard(conviction),
        _portfolio_dashboard(portfolio_opt),
        _cross_asset_dashboard(cross_asset),
        SEP,
    ]
    return NL.join(sections)
