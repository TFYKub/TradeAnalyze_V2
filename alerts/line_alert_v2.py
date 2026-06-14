"""
LINE Alert System v2 – Production‑Grade with Rate Limiting & Retries
====================================================================
Extends the original send_line_message() with:
  • Token bucket rate limiter (100 requests/minute)
  • Exponential backoff with jitter on 429 errors
  • Respects Retry-After header
  • Global lock to serialise requests
  • Detailed header logging for error diagnosis
  • Monthly quota detection and permanent disable
All formatting functions are unchanged from the original V2.
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from threading import Lock
from typing import Optional

import requests

# Import global LINE_DISABLED flag from config
from config.config import LINE_TOKEN as CFG_LINE_TOKEN, LINE_DISABLED as GLOBAL_LINE_DISABLED

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Helper: safe attribute getter
# ----------------------------------------------------------------------
def _g(obj, *attrs, default="—"):
    for attr in attrs:
        v = getattr(obj, attr, None)
        if v is not None:
            return v
    return default

# ----------------------------------------------------------------------
# Global runtime flag for LINE disabled (exported)
# ----------------------------------------------------------------------
LINE_DISABLED = GLOBAL_LINE_DISABLED

# ----------------------------------------------------------------------
# Rate limiting: token bucket (100 requests per minute – safer limit)
# ----------------------------------------------------------------------
_RATE_LIMIT_PER_MINUTE = 100
_TOKEN_BUCKET = {
    "tokens": _RATE_LIMIT_PER_MINUTE,
    "last_refill": time.time(),
}
_send_lock = Lock()

def _refill_bucket() -> None:
    now = time.time()
    elapsed = now - _TOKEN_BUCKET["last_refill"]
    tokens_to_add = elapsed * (_RATE_LIMIT_PER_MINUTE / 60.0)
    if tokens_to_add > 0:
        _TOKEN_BUCKET["tokens"] = min(_RATE_LIMIT_PER_MINUTE,
                                      _TOKEN_BUCKET["tokens"] + tokens_to_add)
        _TOKEN_BUCKET["last_refill"] = now
        logger.debug("[line_v2] Token bucket refilled: now %.1f tokens", _TOKEN_BUCKET["tokens"])

def _consume_token() -> bool:
    _refill_bucket()
    if _TOKEN_BUCKET["tokens"] >= 1:
        _TOKEN_BUCKET["tokens"] -= 1
        logger.debug("[line_v2] Token consumed: %.1f tokens remaining", _TOKEN_BUCKET["tokens"])
        return True
    logger.debug("[line_v2] No tokens available, waiting...")
    return False

# ----------------------------------------------------------------------
# Low‑level sender with exponential backoff and jitter
# ----------------------------------------------------------------------
LINE_TOKEN = CFG_LINE_TOKEN
_LINE_BROADCAST_URL = "https://api.line.me/v2/bot/message/broadcast"
LINE_CHAR_LIMIT = 4500
MAX_RETRIES = 10

def _log_response_headers(resp: requests.Response) -> None:
    """Log all response headers for debugging."""
    headers_to_log = [
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
        "Retry-After",
    ]
    logger.warning("[line_v2] Response status: %d %s", resp.status_code, resp.reason)
    for header in headers_to_log:
        if header in resp.headers:
            logger.warning("[line_v2] Header %s: %s", header, resp.headers[header])
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("[line_v2] Full response headers: %s", dict(resp.headers))

def send_line_message(msg: str) -> bool:
    """Broadcast a text message via LINE Messaging API with rate limiting and smart retries."""
    global LINE_DISABLED
    if LINE_DISABLED:
        logger.info("[LINE] Notifications disabled (monthly quota exceeded)")
        return False

    if not LINE_TOKEN:
        logger.warning("LINE_TOKEN not set — skipping LINE notification")
        return False

    with _send_lock:
        # Wait for a token
        while not _consume_token():
            time.sleep(0.1)

        headers = {
            "Authorization": f"Bearer {LINE_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {"messages": [{"type": "text", "text": msg[:LINE_CHAR_LIMIT]}]}

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.post(_LINE_BROADCAST_URL, headers=headers, json=payload, timeout=30)

                if resp.status_code == 200:
                    logger.info("[line_v2] message sent (%d chars)", len(msg))
                    return True

                if resp.status_code == 429:
                    _log_response_headers(resp)
                    error_str = ""
                    try:
                        error_body = resp.json()
                        error_str = json.dumps(error_body).lower()
                        logger.warning("[line_v2] 429 error body: %s", error_str[:200])
                    except:
                        error_str = resp.text.lower()
                        logger.warning("[line_v2] 429 error body (text): %s", resp.text[:200])

                    # Monthly quota detection
                    if "monthly" in error_str and "limit" in error_str:
                        logger.warning("[LINE] Monthly quota exceeded – disabling LINE for this runtime")
                        LINE_DISABLED = True
                        import config.config as cfg
                        cfg.LINE_DISABLED = True
                        return False

                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait = int(retry_after)
                        except ValueError:
                            wait = min(60, 2 ** attempt)
                    else:
                        wait = min(60, 2 ** attempt)
                    wait += random.uniform(0, 3)
                    logger.warning("[line_v2] 429 Too Many Requests (attempt %d/%d), waiting %.1fs", attempt, MAX_RETRIES, wait)
                    time.sleep(wait)
                    continue

                # Other errors
                _log_response_headers(resp)
                try:
                    error_body = resp.json()
                    logger.error("[line_v2] HTTP %d error body: %s", resp.status_code, json.dumps(error_body)[:200])
                except:
                    logger.error("[line_v2] HTTP %d error body (text): %s", resp.status_code, resp.text[:200])
                return False

            except requests.Timeout:
                logger.error("[line_v2] Request timeout (attempt %d/%d)", attempt, MAX_RETRIES)
                if attempt == MAX_RETRIES:
                    return False
                time.sleep(min(60, 2 ** attempt))
                continue
            except Exception as exc:
                logger.error("[line_v2] send failed: %s", exc)
                return False

        logger.error("[line_v2] send failed after %d retries", MAX_RETRIES)
        return False

# ----------------------------------------------------------------------
# Emoji maps and block formatters (unchanged from original V2)
# ----------------------------------------------------------------------
_REGIME_EMOJI = {
    "STRONG_BULL": "🚀", "BULL": "📈",
    "RANGE": "↔️",
    "BEAR": "📉",       "STRONG_BEAR": "🔻",
}
_LIQ_EMOJI = {
    "RISK_ON": "✅", "RECOVERY": "🔄", "RISK_OFF": "⚠️", "CRISIS": "🚨",
}
_FLOW_EMOJI = {
    "BULLISH_FLOW": "🟢", "BEARISH_FLOW": "🔴",
    "SHORT_SQUEEZE": "🚀", "LONG_SQUEEZE": "💥", "NEUTRAL": "⚪",
}
_BR_EMOJI = {
    "STRONG_BULL": "🚀", "BULL": "📈",
    "NEUTRAL": "➡️", "BEAR": "📉", "STRONG_BEAR": "🔻",
}
_TIER_EMOJI = {
    "FULL SIZE": "🏆", "NORMAL SIZE": "✅",
    "HALF SIZE": "⚡", "NO TRADE": "⏸️",
}
_DECISION_EMOJI = {
    "LONG": "🟢 LONG", "SHORT": "🔴 SHORT",
    "WAIT": "⏸️ WAIT", "NO_TRADE": "⛔ NO TRADE",
}
_FC_EMOJI = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "⚪"}
_CA_EMOJI = {
    "RISK_ON_ALIGNED": "🟢", "RISK_OFF_ALIGNED": "🔴",
    "DECOUPLED_BULL": "🚀", "DECOUPLED_BEAR": "⚠️", "TRANSITION": "↔️",
}
_PERSIST_EMOJI = {
    "ESTABLISHED": "🟢", "MATURING": "🟡",
    "FRESH": "🔵", "EXHAUSTED": "🔴",
}

def _fmt_header(symbol: str, price: float, r) -> str:
    decision = _g(r, "final_decision", default="—")
    grade    = _g(r, "trade_grade",    default="—")
    ai       = _g(r, "ai_score",       default=0.0)
    approved = _g(r, "approved",       default=False)
    dec_str  = _DECISION_EMOJI.get(decision, f"❓ {decision}")
    appr_str = "✅ APPROVED" if approved else "❌ REJECTED"
    return (
        f"{'━'*28}\n"
        f"⚡ TradeAnalyze — Institutional\n"
        f"{'━'*28}\n"
        f"📌 {symbol}  ${price:,.4f}\n"
        f"{dec_str}  Grade: {grade}  AI: {ai:.0f}\n"
        f"{appr_str}\n"
    )

def _fmt_regime_block(r, persistence) -> str:
    regime   = _g(r, "regime",       default="—")
    conf     = _g(r, "regime_conf",  default=0.0)
    vol_reg  = _g(r, "vol_regime",   default="—")
    cv_score = _g(r, "conviction_score", default=0.0)
    cv_tier  = _g(r, "conviction_tier",  default="—")
    cv_kelly = _g(r, "conviction_kelly_mult", default=0.0)

    pe_label  = _g(persistence, "persistence_label",       default=_g(r, "persistence_label", "—"))
    pe_remain = _g(persistence, "remaining_duration_days", default=_g(r, "remaining_days", 0.0))
    pe_ex7    = _g(persistence, "exit_prob_7d",            default=_g(r, "exit_prob_7d", 0.0))
    pe_next   = _g(persistence, "most_likely_next",        default="—")

    r_emoji  = _REGIME_EMOJI.get(regime,   "❓")
    t_emoji  = _TIER_EMOJI.get(cv_tier,    "❓")
    p_emoji  = _PERSIST_EMOJI.get(pe_label, "❓")

    return (
        f"\n📊 REGIME & CONVICTION\n"
        f"  {r_emoji} {regime} ({conf:.0f}%)  Vol: {vol_reg}\n"
        f"  {t_emoji} {cv_tier}  Score: {cv_score:.0f}/100  Kelly×{cv_kelly:.2f}\n"
        f"  {p_emoji} Persist: {pe_label}  ~{pe_remain:.0f}d left  "
        f"Exit7d: {pe_ex7:.0f}%  Next→{pe_next}\n"
    )

def _fmt_macro_block(liquidity, breadth, cross_asset) -> str:
    liq_r    = _g(liquidity, "liquidity_regime", default="—")
    liq_s    = _g(liquidity, "score",            default=0.0)
    liq_mult = _g(liquidity, "risk_multiplier",  default=1.0)
    vix      = _g(liquidity, "vix_level",        default=0.0)
    dxy_t    = _g(liquidity, "dxy_trend",        default="—")
    l_emoji  = _LIQ_EMOJI.get(liq_r, "❓")

    br_r   = _g(breadth, "breadth_regime", default="—")
    br_s   = _g(breadth, "breadth_score",  default=0.0)
    eb     = getattr(breadth, "equity_breadth", None)
    ad     = _g(eb, "advance_decline_ratio", default=0.0)
    p200   = _g(eb, "pct_above_200dma",      default=0.0)
    b_emoji= _BR_EMOJI.get(br_r, "❓")

    ca_r   = _g(cross_asset, "cross_asset_regime",       default="—")
    rs     = _g(cross_asset, "relative_strength_score",  default=0.0)
    b_spy  = _g(cross_asset, "btc_beta_to_spy",          default=0.0)
    decouple=_g(cross_asset, "decoupling_detected",      default=False)
    c_emoji= _CA_EMOJI.get(ca_r, "❓")
    dec_str= "⚡DECOUPLED" if decouple else ""

    return (
        f"\n🌍 MACRO ENVIRONMENT\n"
        f"  {l_emoji} Liq: {liq_r} ({liq_s:.0f}/100)  ×{liq_mult:.2f}  "
        f"VIX:{vix:.1f}  DXY:{dxy_t}\n"
        f"  {b_emoji} Breadth: {br_r} ({br_s:.0f}/100)  "
        f"A/D:{ad:.2f}  >200d:{p200:.0f}%\n"
        f"  {c_emoji} Cross-Asset: {ca_r}  RS:{rs:.0f}  β_SPY:{b_spy:.2f} {dec_str}\n"
    )

def _fmt_flow_block(flow) -> str:
    if flow is None:
        return "\n🌀 FLOW: Data unavailable\n"
    fl_r  = flow.flow_regime
    fl_s  = flow.flow_score
    fl_d  = flow.flow_direction
    fl_c  = flow.flow_confidence
    fr    = flow.funding_rate_pct
    ls    = flow.ls_ratio
    oi    = flow.oi_signal
    casc  = flow.cascade_risk
    f_emoji = _FLOW_EMOJI.get(fl_r, "❓")
    return (
        f"\n🌀 DERIVATIVES FLOW\n"
        f"  {f_emoji} {fl_r} ({fl_s:.0f}/100  {fl_d}  {fl_c:.0f}%)\n"
        f"  Funding: {fr:+.4f}%   L/S: {ls:.2f}\n"
        f"  OI: {oi}   Cascade: {casc}\n"
    )

def _fmt_forecast_block(forecast) -> str:
    if forecast is None:
        return "\n🔮 FORECAST: Data unavailable\n"
    fc_d   = forecast.forecast_direction
    fc_c   = forecast.forecast_confidence
    model  = forecast.model_used
    hf     = forecast.horizon_forecasts
    r5     = hf.get("5d",  {})
    r10    = hf.get("10d", {})
    r20    = hf.get("20d", {})
    f_emoji= _FC_EMOJI.get(fc_d, "❓")
    top_feat = sorted(
        forecast.feature_importances.items(), key=lambda x: -x[1]
    )[:2] if forecast.feature_importances else []
    feat_str = "  ".join(f"{k}" for k, _ in top_feat) if top_feat else ""
    return (
        f"\n🔮 ML FORECAST  [{model}]\n"
        f"  {f_emoji} {fc_d}  Conf: {fc_c:.0f}%\n"
        f"  5d:  {r5.get('return_pct',0):+.1f}%  P(↑):{r5.get('prob_up',0.5):.0%}\n"
        f"  10d: {r10.get('return_pct',0):+.1f}%  P(↑):{r10.get('prob_up',0.5):.0%}\n"
        f"  20d: {r20.get('return_pct',0):+.1f}%  P(↑):{r20.get('prob_up',0.5):.0%}\n"
        + (f"  Drivers: {feat_str}\n" if feat_str else "")
    )

def _fmt_trade_setup(r) -> str:
    entry = _g(r, "entry",     default=0.0)
    sl    = _g(r, "stop_loss", default=0.0)
    tp1   = _g(r, "tp1",       default=None)
    tp2   = _g(r, "tp2",       default=None)
    rr    = _g(r, "rr",        default=0.0)
    kelly = _g(r, "kelly",     default=0.0)
    mc    = _g(r, "mc_profit_prob", default=0.0)
    ev    = _g(r, "ev",        default=0.0)
    tp1_str = f"${tp1:,.4f}" if tp1 else "—"
    tp2_str = f"${tp2:,.4f}" if tp2 else "—"
    return (
        f"\n💰 TRADE SETUP\n"
        f"  Entry: ${entry:,.4f}   SL: ${sl:,.4f}\n"
        f"  TP1:   {tp1_str}   TP2: {tp2_str}\n"
        f"  RR: {rr:.2f}x   Kelly: {kelly:.3f}   MC: {mc:.0%}\n"
        f"  EV: {ev:.2f}\n"
    )

def _fmt_footer(r) -> str:
    runtime = _g(r, "runtime", default=0.0)
    return (
        f"\n{'─'*28}\n"
        f"⏱ {runtime:.1f}s  |  TradeAnalyze v2\n"
    )

def format_institutional_alert(
    symbol:      str,
    price:       float,
    result_v2,
    liquidity  = None,
    flow       = None,
    breadth    = None,
    persistence= None,
    forecast   = None,
    conviction = None,
    cross_asset= None,
) -> str:
    blocks = [
        _fmt_header(symbol, price, result_v2),
        _fmt_regime_block(result_v2, persistence),
        _fmt_macro_block(liquidity, breadth, cross_asset),
        _fmt_flow_block(flow),
        _fmt_forecast_block(forecast),
        _fmt_trade_setup(result_v2),
        _fmt_footer(result_v2),
    ]
    msg = "".join(blocks)
    if len(msg) > LINE_CHAR_LIMIT:
        msg = msg[:LINE_CHAR_LIMIT - 20] + "\n...[truncated]"
    return msg

def _should_alert(result_v2, conviction=None, min_conviction: float = 50.0) -> bool:
    decision = _g(result_v2, "final_decision", default="WAIT")
    cv_score = _g(conviction, "conviction_score",
                  default=_g(result_v2, "conviction_score", default=0.0))
    approved = _g(result_v2, "approved", default=False)
    liq = _g(result_v2, "liquidity_regime", default="RISK_ON")
    if liq == "CRISIS":
        return True
    if approved and cv_score >= min_conviction:
        return True
    flow_r = _g(result_v2, "flow_regime", default="NEUTRAL")
    if flow_r in ("SHORT_SQUEEZE", "LONG_SQUEEZE"):
        return True
    return False

def send_institutional_alert(
    symbol:         str,
    price:          float,
    result_v2,
    liquidity     = None,
    flow          = None,
    breadth       = None,
    persistence   = None,
    forecast      = None,
    conviction    = None,
    cross_asset   = None,
    force:          bool = False,
    min_conviction: float = 50.0,
) -> bool:
    if not force and not _should_alert(result_v2, conviction, min_conviction):
        logger.info(
            "[line_v2] %s alert filtered (decision=%s conviction=%.0f)",
            symbol,
            _g(result_v2, "final_decision", default="?"),
            _g(conviction, "conviction_score",
               default=_g(result_v2, "conviction_score", default=0.0)),
        )
        return False
    try:
        msg = format_institutional_alert(
            symbol, price, result_v2,
            liquidity, flow, breadth, persistence,
            forecast, conviction, cross_asset,
        )
        sent = send_line_message(msg)
        if sent:
            logger.info("[line_v2] %s institutional alert sent", symbol)
        return sent
    except Exception as exc:
        logger.error("[line_v2] format/send failed for %s: %s", symbol, exc)
        return False

def send_crisis_alert(liquidity) -> bool:
    if liquidity is None:
        return False
    regime   = _g(liquidity, "liquidity_regime", default="")
    vix      = _g(liquidity, "vix_level",        default=0.0)
    score    = _g(liquidity, "score",             default=0.0)
    dxy_t    = _g(liquidity, "dxy_trend",         default="—")
    yield_t  = _g(liquidity, "yield_trend",       default="—")
    if regime != "CRISIS":
        return False
    msg = (
        f"🚨 CRISIS ALERT — TradeAnalyze v2\n"
        f"{'━'*28}\n"
        f"Liquidity Regime: CRISIS\n"
        f"Score: {score:.0f}/100   VIX: {vix:.1f}\n"
        f"DXY: {dxy_t}   Yields: {yield_t}\n"
        f"\n⛔ All position sizing reduced to 0.25×\n"
        f"⛔ No new directional trades recommended\n"
        f"{'─'*28}\n"
        f"TradeAnalyze v2 — Institutional Risk Monitor"
    )
    return send_line_message(msg)