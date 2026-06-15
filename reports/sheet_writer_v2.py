"""
Google Sheets Writer v2 – Batched version
"""
import logging
import math
from datetime import datetime
from typing import Optional

from config.config import SHEET_ID
from utils.sheets_auth import get_sheets_client
from utils.batch_writer import get_batch_writer

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _safe(v) -> str | float | int:
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, dict):
        return " | ".join(f"{k}:{round(val,2)}" for k, val in list(v.items())[:5])
    if isinstance(v, list):
        return " | ".join(str(i) for i in v[:5])
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return ""
        return round(v, 4)
    return v


def _ensure_headers(ws, headers: list[str]) -> None:
    try:
        first_row = ws.row_values(1)
        if not first_row or first_row[0] != headers[0]:
            ws.insert_row(headers, index=1)
    except Exception as exc:
        logger.debug("[sheet_v2] header check: %s", exc)


_INST_HEADERS = [
    "Timestamp", "Symbol", "Price",
    "Regime", "RegimeConf", "VolRegime", "FinalDecision",
    "AIScore", "TradeGrade", "Approved",
    "LiquidityRegime", "LiquidityScore", "LiquidityRiskMult",
    "VIXLevel", "DXYTrend", "YieldTrend",
    "FlowRegime", "FlowScore", "FlowDirection",
    "FundingRate", "LSRatio", "OISignal", "CascadeRisk",
    "BreadthRegime", "BreadthScore", "BreadthConf",
    "BTCDomEst", "TOTAL3Change", "ADRatio", "PctAbove200DMA",
    "PersistenceLabel", "PersistenceScore",
    "ExpectedDuration", "RemainingDays", "HalfLifeDays",
    "ExitProb7d", "ExitProb14d", "MostLikelyNext",
    "ForecastDirection", "ForecastConf", "ModelUsed",
    "Return5d", "ProbUp5d",
    "Return10d", "ProbUp10d",
    "Return20d", "ProbUp20d",
    "ConvictionScore", "ConvictionTier", "KellyMult",
    "TradeAllowed", "WeakestSignal", "StrongestSignal", "AlignedCount",
    "PortfolioVol", "PortfolioSharpe", "PortfolioDD", "DivRatio",
    "CrossAssetRegime", "CrossAssetConf", "RSScore",
    "BtcBetaSPY", "BtcBetaQQQ", "DecouplingDetected",
    "CorrQQQ", "CorrSPY", "CorrDXY", "CorrGLD",
]


def log_institutional_signals(
    symbol:    str,
    price:     float,
    result_v2,
    liquidity  = None,
    flow       = None,
    breadth    = None,
    persistence= None,
    forecast   = None,
    conviction = None,
    portfolio  = None,
    cross_asset= None,
) -> None:
    r = result_v2

    liq_regime  = _safe(getattr(liquidity, "liquidity_regime",  getattr(r, "liquidity_regime",  "")))
    liq_score   = _safe(getattr(liquidity, "score",             getattr(r, "liquidity_score",   "")))
    liq_mult    = _safe(getattr(liquidity, "risk_multiplier",   getattr(r, "liquidity_risk_mult","")))
    vix_level   = _safe(getattr(liquidity, "vix_level",   ""))
    dxy_trend   = _safe(getattr(liquidity, "dxy_trend",   ""))
    yield_trend = _safe(getattr(liquidity, "yield_trend", ""))

    fl_regime  = _safe(getattr(flow, "flow_regime",    getattr(r, "flow_regime",    "")))
    fl_score   = _safe(getattr(flow, "flow_score",     getattr(r, "flow_score",     "")))
    fl_dir     = _safe(getattr(flow, "flow_direction", getattr(r, "flow_direction", "")))
    fl_fr      = _safe(getattr(flow, "funding_rate_pct", ""))
    fl_ls      = _safe(getattr(flow, "ls_ratio",         ""))
    fl_oi      = _safe(getattr(flow, "oi_signal",        ""))
    fl_casc    = _safe(getattr(flow, "cascade_risk",     ""))

    br_regime  = _safe(getattr(breadth, "breadth_regime",     getattr(r, "breadth_regime",  "")))
    br_score   = _safe(getattr(breadth, "breadth_score",      getattr(r, "breadth_score",   "")))
    br_conf    = _safe(getattr(breadth, "breadth_confidence", ""))
    cb         = getattr(breadth, "crypto_breadth", None)
    eb         = getattr(breadth, "equity_breadth", None)
    br_btcd    = _safe(getattr(cb, "btc_dominance_est",    ""))
    br_t3      = _safe(getattr(cb, "total3_change_pct",   ""))
    br_ad      = _safe(getattr(eb, "advance_decline_ratio",""))
    br_200     = _safe(getattr(eb, "pct_above_200dma",    ""))

    pe_label   = _safe(getattr(persistence, "persistence_label",       getattr(r, "persistence_label", "")))
    pe_score   = _safe(getattr(persistence, "persistence_score",       ""))
    pe_exp     = _safe(getattr(persistence, "expected_duration_days",  ""))
    pe_remain  = _safe(getattr(persistence, "remaining_duration_days", getattr(r, "remaining_days", "")))
    pe_hl      = _safe(getattr(persistence, "regime_half_life_days",   ""))
    pe_ex7     = _safe(getattr(persistence, "exit_prob_7d",            getattr(r, "exit_prob_7d", "")))
    pe_ex14    = _safe(getattr(persistence, "exit_prob_14d",           ""))
    pe_next    = _safe(getattr(persistence, "most_likely_next",        ""))

    fc_dir     = _safe(getattr(forecast, "forecast_direction",  getattr(r, "forecast_direction",  "")))
    fc_conf    = _safe(getattr(forecast, "forecast_confidence", getattr(r, "forecast_confidence", "")))
    fc_model   = _safe(getattr(forecast, "model_used",          ""))
    hf         = getattr(forecast, "horizon_forecasts", {})
    fc_r5      = _safe(hf.get("5d",  {}).get("return_pct", ""))
    fc_p5      = _safe(hf.get("5d",  {}).get("prob_up",    ""))
    fc_r10     = _safe(hf.get("10d", {}).get("return_pct", ""))
    fc_p10     = _safe(hf.get("10d", {}).get("prob_up",    ""))
    fc_r20     = _safe(hf.get("20d", {}).get("return_pct", getattr(r, "forecast_20d_return", "")))
    fc_p20     = _safe(hf.get("20d", {}).get("prob_up",    getattr(forecast, "probability_up_20d", "")))

    cv_score   = _safe(getattr(conviction, "conviction_score",   getattr(r, "conviction_score",   "")))
    cv_tier    = _safe(getattr(conviction, "conviction_tier",    getattr(r, "conviction_tier",    "")))
    cv_kelly   = _safe(getattr(conviction, "kelly_multiplier",   getattr(r, "conviction_kelly_mult","")))
    cv_allow   = _safe(getattr(conviction, "trade_allowed",      ""))
    cv_weak    = _safe(getattr(conviction, "weakest_signal",     ""))
    cv_strong  = _safe(getattr(conviction, "strongest_signal",   ""))
    cv_aligned = _safe(getattr(conviction, "alignment_count",    ""))

    po_vol     = _safe(getattr(portfolio, "portfolio_volatility",  getattr(r, "portfolio_vol",      "")))
    po_sharpe  = _safe(getattr(portfolio, "portfolio_sharpe",      getattr(r, "portfolio_sharpe",   "")))
    po_dd      = _safe(getattr(portfolio, "portfolio_drawdown",    getattr(r, "portfolio_drawdown", "")))
    po_div     = _safe(getattr(portfolio, "diversification_ratio", ""))

    ca_regime  = _safe(getattr(cross_asset, "cross_asset_regime",       getattr(r, "cross_asset_regime","" )))
    ca_conf    = _safe(getattr(cross_asset, "regime_confidence",         ""))
    ca_rs      = _safe(getattr(cross_asset, "relative_strength_score",  getattr(r, "btc_rs_score",    "")))
    ca_bspy    = _safe(getattr(cross_asset, "btc_beta_to_spy",           getattr(r, "btc_beta_spy",    "")))
    ca_bqqq    = _safe(getattr(cross_asset, "btc_beta_to_qqq",           ""))
    ca_dec     = _safe(getattr(cross_asset, "decoupling_detected",       ""))
    corrs      = getattr(cross_asset, "rolling_correlations", {})
    ca_cqqq    = _safe(corrs.get("QQQ", ""))
    ca_cspy    = _safe(corrs.get("SPY", ""))
    ca_cdxy    = _safe(corrs.get("DXY", ""))
    ca_cgld    = _safe(corrs.get("GLD", ""))

    row = [
        _now(), symbol, _safe(price),
        _safe(getattr(r, "regime", "")),
        _safe(getattr(r, "regime_conf", "")),
        _safe(getattr(r, "vol_regime", "")),
        _safe(getattr(r, "final_decision", "")),
        _safe(getattr(r, "ai_score", "")),
        _safe(getattr(r, "trade_grade", "")),
        _safe(getattr(r, "approved", "")),
        liq_regime, liq_score, liq_mult, vix_level, dxy_trend, yield_trend,
        fl_regime, fl_score, fl_dir, fl_fr, fl_ls, fl_oi, fl_casc,
        br_regime, br_score, br_conf, br_btcd, br_t3, br_ad, br_200,
        pe_label, pe_score, pe_exp, pe_remain, pe_hl, pe_ex7, pe_ex14, pe_next,
        fc_dir, fc_conf, fc_model, fc_r5, fc_p5, fc_r10, fc_p10, fc_r20, fc_p20,
        cv_score, cv_tier, cv_kelly, cv_allow, cv_weak, cv_strong, cv_aligned,
        po_vol, po_sharpe, po_dd, po_div,
        ca_regime, ca_conf, ca_rs, ca_bspy, ca_bqqq, ca_dec,
        ca_cqqq, ca_cspy, ca_cdxy, ca_cgld,
    ]

    try:
        get_batch_writer().add_row("InstitutionalSignals", row, _INST_HEADERS)
        logger.info(f"[sheet_v2] InstitutionalSignals ← {symbol} (batched)")
    except Exception as exc:
        logger.error(f"[sheet_v2] InstitutionalSignals add failed ({symbol}): {exc}")


_REGIME_HEADERS = [
    "Timestamp", "Symbol",
    "Regime", "RegimeConf", "PersistenceLabel", "RemainingDays", "ExitProb7d",
    "LiquidityRegime", "LiquidityScore",
    "BreadthRegime", "BreadthScore",
    "CrossAssetRegime", "RSScore",
    "ConvictionScore", "ConvictionTier",
]


def log_regime_dashboard(symbol: str, result_v2) -> None:
    r = result_v2
    row = [
        _now(), symbol,
        _safe(getattr(r, "regime",              "")),
        _safe(getattr(r, "regime_conf",         "")),
        _safe(getattr(r, "persistence_label",   "")),
        _safe(getattr(r, "remaining_days",      "")),
        _safe(getattr(r, "exit_prob_7d",        "")),
        _safe(getattr(r, "liquidity_regime",    "")),
        _safe(getattr(r, "liquidity_score",     "")),
        _safe(getattr(r, "breadth_regime",      "")),
        _safe(getattr(r, "breadth_score",       "")),
        _safe(getattr(r, "cross_asset_regime",  "")),
        _safe(getattr(r, "btc_rs_score",        "")),
        _safe(getattr(r, "conviction_score",    "")),
        _safe(getattr(r, "conviction_tier",     "")),
    ]
    try:
        get_batch_writer().add_row("RegimeDashboard", row, _REGIME_HEADERS)
        logger.info(f"[sheet_v2] RegimeDashboard ← {symbol} (batched)")
    except Exception as exc:
        logger.error(f"[sheet_v2] RegimeDashboard add failed ({symbol}): {exc}")


_FLOW_HEADERS = [
    "Timestamp", "Symbol", "Price",
    "FlowRegime", "FlowScore", "FlowDirection", "FlowConf",
    "FundingRate", "FundingRegime", "LSRatio",
    "OISignal", "CascadeRisk",
    "FundingScore", "OIScore", "LSScore", "LiqScore",
]


def log_flow_snapshot(symbol: str, price: float, flow) -> None:
    if flow is None:
        return
    row = [
        _now(), symbol, _safe(price),
        _safe(flow.flow_regime),    _safe(flow.flow_score),
        _safe(flow.flow_direction), _safe(flow.flow_confidence),
        _safe(flow.funding_rate_pct), _safe(flow.funding_regime),
        _safe(flow.ls_ratio),
        _safe(flow.oi_signal),    _safe(flow.cascade_risk),
        _safe(flow.funding_score),_safe(flow.oi_score),
        _safe(flow.ls_score),     _safe(flow.liq_score),
    ]
    try:
        get_batch_writer().add_row("FlowSnapshot", row, _FLOW_HEADERS)
        logger.info(f"[sheet_v2] FlowSnapshot ← {symbol} (batched)")
    except Exception as exc:
        logger.error(f"[sheet_v2] FlowSnapshot add failed ({symbol}): {exc}")


def write_all_institutional(
    symbol:      str,
    price:       float,
    result_v2,
    liquidity  = None,
    flow       = None,
    breadth    = None,
    persistence= None,
    forecast   = None,
    conviction = None,
    portfolio  = None,
    cross_asset= None,
) -> None:
    try:
        log_institutional_signals(
            symbol, price, result_v2,
            liquidity, flow, breadth, persistence,
            forecast, conviction, portfolio, cross_asset,
        )
    except Exception as exc:
        logger.error(f"[sheet_v2] InstitutionalSignals failed: {exc}")

    try:
        log_regime_dashboard(symbol, result_v2)
    except Exception as exc:
        logger.error(f"[sheet_v2] RegimeDashboard failed: {exc}")

    try:
        log_flow_snapshot(symbol, price, flow)
    except Exception as exc:
        logger.error(f"[sheet_v2] FlowSnapshot failed: {exc}")