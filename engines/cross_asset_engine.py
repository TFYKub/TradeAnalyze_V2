"""
Cross Asset Engine  (Phase 21)
================================
Analyses BTC's relationship with macro and risk assets to:
  1. Confirm / challenge the current regime
  2. Detect decoupling events (BTC moving independently of risk assets)
  3. Generate a relative strength score vs. each pair
  4. Output a cross-asset regime classification

Asset Pairs Analysed:
  BTC vs GLD    — safe-haven comparison (positive corr = risk-on for both)
  BTC vs SPY    — equity risk correlation (BTC = risk-on asset)
  BTC vs QQQ    — tech/growth factor correlation (strongest historical link)
  BTC vs DXY    — dollar strength inverse (BTC falls when DXY rises)
  BTC vs TLT    — bond proxy; BTC often inversely correlated to duration

Cross-Asset Regime:
  RISK_ON_ALIGNED   — BTC rising with SPY/QQQ, DXY weak, bonds rising
  RISK_OFF_ALIGNED  — BTC falling with SPY/QQQ, DXY strong
  DECOUPLED_BULL    — BTC rising while risk-off in equities (crypto-specific demand)
  DECOUPLED_BEAR    — BTC falling while equities hold up (crypto-specific sell)
  TRANSITION        — mixed signals, no dominant cross-asset pattern

Relative Strength Score (0–100):
  100 = BTC strongly outperforming all comparators
  0   = BTC strongly underperforming all comparators
  50  = inline / neutral

Integration points:
  → Regime Ensemble (weight 8%, new component)
  → Forecast Engine (feature: cross_asset_regime_enc)
  → Conviction Engine (cross-asset alignment bonus/penalty)
  → Report dashboards (Cross Asset section)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FETCH_PERIOD  = "6mo"
CORR_WINDOW   = 30    # rolling correlation window (days)
RS_WINDOW     = 20    # relative strength lookback (days)

# Asset ticker map
TICKERS = {
    "BTC":  "BTC-USD",
    "GLD":  "GLD",       # Gold ETF
    "SPY":  "SPY",       # S&P 500
    "QQQ":  "QQQ",       # NASDAQ 100
    "DXY":  "DX-Y.NYB",  # Dollar Index
    "TLT":  "TLT",       # 20Y Treasury (bond proxy)
}

# Expected correlations in RISK_ON regime (positive = same direction, negative = inverse)
EXPECTED_CORRELATIONS_RISK_ON = {
    "GLD":  0.20,    # mild positive — both are alternative assets
    "SPY":  0.60,    # strong positive — both risk-on
    "QQQ":  0.65,    # strongest — tech/growth factor
    "DXY": -0.45,    # negative — DXY strength = BTC weakness
    "TLT":  0.15,    # mild positive in RISK_ON (rates low = bonds up = BTC up)
}


@dataclass(frozen=True)
class PairAnalysis:
    """Analysis of BTC vs one comparator asset."""
    pair:              str          # e.g. "BTC/QQQ"
    btc_rs_score:      float        # 0–100 relative strength of BTC vs this pair
    rolling_corr:      float        # 30-day rolling correlation
    expected_corr:     float        # historical expected correlation
    corr_deviation:    float        # actual − expected (magnitude = decoupling signal)
    btc_return_20d:    float        # BTC 20-day return %
    pair_return_20d:   float        # comparator 20-day return %
    outperforming:     bool         # BTC > comparator on risk-adjusted basis
    signal:            str          # ALIGNED | DECOUPLED | INVERSE


@dataclass(frozen=True)
class CrossAssetResult:
    cross_asset_regime:    str              # RISK_ON_ALIGNED | RISK_OFF_ALIGNED | DECOUPLED_BULL | DECOUPLED_BEAR | TRANSITION
    relative_strength_score: float         # 0–100
    regime_confidence:     float           # 0–100
    pair_analyses:         dict[str, PairAnalysis]
    rolling_correlations:  dict[str, float]  # {pair_name: 30d corr}
    decoupling_detected:   bool
    decoupling_direction:  str              # BULL | BEAR | NONE
    btc_beta_to_spy:       float            # BTC beta to SPY (rolling 30d)
    btc_beta_to_qqq:       float
    dominant_driver:       str              # which asset BTC is most correlated with currently
    interpretation:        str


# ── Data Fetch ─────────────────────────────────────────────────────────────────
def _fetch(ticker: str) -> Optional[pd.Series]:
    """Returns Close price series or None on failure."""
    try:
        import yfinance as yf
        df = yf.download(ticker, period=FETCH_PERIOD, interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df is None or len(df) < 40:
            return None
        return df["Close"].dropna()
    except Exception as exc:
        logger.debug("[cross_asset] fetch %s: %s", ticker, exc)
        return None


# ── Rolling Correlation & Beta ────────────────────────────────────────────────
def _rolling_corr(btc_ret: pd.Series, asset_ret: pd.Series, window: int = CORR_WINDOW) -> float:
    """30-day rolling correlation between BTC and asset returns."""
    combined = pd.DataFrame({"btc": btc_ret, "asset": asset_ret}).dropna()
    if len(combined) < window:
        return 0.0
    return round(float(combined["btc"].rolling(window).corr(combined["asset"]).iloc[-1]), 3)


def _rolling_beta(btc_ret: pd.Series, asset_ret: pd.Series, window: int = CORR_WINDOW) -> float:
    """
    Rolling beta: β = Cov(BTC, Asset) / Var(Asset).
    BTC β to SPY > 1 means BTC amplifies equity moves.
    """
    combined = pd.DataFrame({"btc": btc_ret, "asset": asset_ret}).dropna()
    if len(combined) < window:
        return 1.0
    recent   = combined.tail(window)
    cov_mat  = recent.cov()
    var_asset = float(cov_mat.loc["asset", "asset"])
    if var_asset < 1e-12:
        return 1.0
    return round(float(cov_mat.loc["btc", "asset"]) / var_asset, 3)


# ── Pair Analysis ─────────────────────────────────────────────────────────────
def _analyse_pair(
    pair_name:  str,
    btc_close:  pd.Series,
    asset_close: pd.Series,
) -> PairAnalysis:
    """Compute BTC vs single comparator analysis."""
    # Align series
    combined = pd.DataFrame({"btc": btc_close, "asset": asset_close}).dropna()
    if len(combined) < RS_WINDOW + 5:
        return PairAnalysis(
            pair=f"BTC/{pair_name}", btc_rs_score=50.0,
            rolling_corr=0.0, expected_corr=EXPECTED_CORRELATIONS_RISK_ON.get(pair_name, 0.0),
            corr_deviation=0.0, btc_return_20d=0.0, pair_return_20d=0.0,
            outperforming=False, signal="UNKNOWN",
        )

    btc_ret   = np.log(combined["btc"]   / combined["btc"].shift(1)).dropna()
    asset_ret = np.log(combined["asset"] / combined["asset"].shift(1)).dropna()

    # 20-day return
    btc_ret20  = float((combined["btc"].iloc[-1]   / combined["btc"].iloc[-min(RS_WINDOW, len(combined)-1)]   - 1) * 100)
    asset_ret20= float((combined["asset"].iloc[-1] / combined["asset"].iloc[-min(RS_WINDOW, len(combined)-1)] - 1) * 100)

    # Rolling correlation
    corr         = _rolling_corr(btc_ret, asset_ret)
    expected_corr= EXPECTED_CORRELATIONS_RISK_ON.get(pair_name, 0.0)
    corr_dev     = round(corr - expected_corr, 3)    # positive = more correlated than expected

    # Relative strength: risk-adjusted return comparison
    btc_vol20   = float(btc_ret.rolling(RS_WINDOW).std().iloc[-1]  * math.sqrt(252) * 100) if len(btc_ret) >= RS_WINDOW else 20.0
    asset_vol20 = float(asset_ret.rolling(RS_WINDOW).std().iloc[-1]* math.sqrt(252) * 100) if len(asset_ret) >= RS_WINDOW else 10.0

    # Sharpe-like RS: (BTC_ret / BTC_vol) vs (Asset_ret / Asset_vol)
    btc_sharpe   = btc_ret20   / (btc_vol20   + 1e-6)
    asset_sharpe = asset_ret20 / (asset_vol20 + 1e-6)

    # For DXY: inverse relationship expected, so flip sign
    if pair_name == "DXY":
        asset_sharpe = -asset_sharpe

    rs_diff  = btc_sharpe - asset_sharpe
    # Map to 0–100 (rs_diff in range -3 to +3 typically)
    rs_score = round(min(100.0, max(0.0, 50.0 + rs_diff * 15.0)), 1)

    outperforming = rs_score > 52.0

    # Signal: ALIGNED | DECOUPLED | INVERSE
    if pair_name == "DXY":
        # DXY inverse relationship
        if corr < -0.20:
            signal = "ALIGNED"     # BTC falling when DXY rises = normal
        elif corr > 0.20:
            signal = "INVERSE"     # BTC rising with DXY = unusual
        else:
            signal = "DECOUPLED"
    else:
        if abs(corr - expected_corr) < 0.25:
            signal = "ALIGNED"
        elif (corr > 0 and expected_corr > 0) or (corr < 0 and expected_corr < 0):
            signal = "ALIGNED"    # direction same, magnitude different
        elif abs(corr_dev) > 0.50:
            signal = "DECOUPLED"
        else:
            signal = "TRANSITION"

    return PairAnalysis(
        pair            = f"BTC/{pair_name}",
        btc_rs_score    = rs_score,
        rolling_corr    = corr,
        expected_corr   = expected_corr,
        corr_deviation  = corr_dev,
        btc_return_20d  = round(btc_ret20,   2),
        pair_return_20d = round(asset_ret20,  2),
        outperforming   = outperforming,
        signal          = signal,
    )


# ── Regime Classification ──────────────────────────────────────────────────────
def _classify_cross_asset_regime(
    pairs:       dict[str, PairAnalysis],
    btc_ret20:   float,
) -> tuple[str, float, bool, str]:
    """
    Returns (regime, confidence, decoupling_detected, decoupling_direction).
    """
    spy_pa  = pairs.get("SPY")
    qqq_pa  = pairs.get("QQQ")
    dxy_pa  = pairs.get("DXY")
    gld_pa  = pairs.get("GLD")

    # Count aligned vs decoupled
    aligned_count   = sum(1 for p in pairs.values() if p.signal == "ALIGNED")
    decoupled_count = sum(1 for p in pairs.values() if p.signal == "DECOUPLED")

    btc_rising = btc_ret20 > 2.0
    btc_falling= btc_ret20 < -2.0

    spy_rising  = (spy_pa.pair_return_20d > 1.0)  if spy_pa  else True
    spy_falling = (spy_pa.pair_return_20d < -1.0) if spy_pa  else False
    dxy_rising  = (dxy_pa.pair_return_20d > 0.5)  if dxy_pa  else False

    decoupling = decoupled_count >= 2

    # Decoupling direction
    if decoupling:
        if btc_rising and spy_falling:
            dec_dir = "BULL"
        elif btc_falling and spy_rising:
            dec_dir = "BEAR"
        else:
            dec_dir = "NONE"
    else:
        dec_dir = "NONE"

    # Regime classification
    if decoupling and dec_dir == "BULL":
        regime = "DECOUPLED_BULL"
        conf   = 55.0 + decoupled_count * 5.0
    elif decoupling and dec_dir == "BEAR":
        regime = "DECOUPLED_BEAR"
        conf   = 55.0 + decoupled_count * 5.0
    elif btc_rising and spy_rising and not dxy_rising and aligned_count >= 3:
        regime = "RISK_ON_ALIGNED"
        conf   = 60.0 + aligned_count * 5.0
    elif btc_falling and (spy_falling or dxy_rising) and aligned_count >= 2:
        regime = "RISK_OFF_ALIGNED"
        conf   = 55.0 + aligned_count * 5.0
    else:
        regime = "TRANSITION"
        conf   = 40.0 + aligned_count * 3.0

    return regime, round(min(90.0, conf), 1), decoupling, dec_dir


# ── Main Entry ─────────────────────────────────────────────────────────────────
def compute_cross_asset(btc_df: Optional[pd.DataFrame] = None) -> CrossAssetResult:
    """
    Compute cross-asset regime and relative strength for BTC.

    Parameters
    ----------
    btc_df : optional pre-fetched BTC OHLCV DataFrame (uses yfinance if None)

    Returns
    -------
    CrossAssetResult — always returns, never raises
    """
    # ── Fetch BTC ─────────────────────────────────────────────────────────────
    if btc_df is not None and len(btc_df) >= 40:
        btc_close = btc_df["Close"].dropna()
    else:
        btc_close = _fetch(TICKERS["BTC"])

    if btc_close is None:
        logger.warning("[cross_asset] BTC data unavailable — returning neutral fallback")
        return CrossAssetResult(
            cross_asset_regime    = "TRANSITION",
            relative_strength_score = 50.0,
            regime_confidence     = 30.0,
            pair_analyses         = {},
            rolling_correlations  = {},
            decoupling_detected   = False,
            decoupling_direction  = "NONE",
            btc_beta_to_spy       = 1.0,
            btc_beta_to_qqq       = 1.5,
            dominant_driver       = "UNKNOWN",
            interpretation        = "Cross-asset data unavailable — neutral applied",
        )

    btc_ret = np.log(btc_close / btc_close.shift(1)).dropna()
    btc_ret20 = float((btc_close.iloc[-1] / btc_close.iloc[-min(RS_WINDOW, len(btc_close)-1)] - 1) * 100)

    # ── Fetch & analyse each comparator ──────────────────────────────────────
    pair_names = ["GLD", "SPY", "QQQ", "DXY", "TLT"]
    pairs: dict[str, PairAnalysis] = {}
    corrs: dict[str, float]        = {}
    available = 0

    for name in pair_names:
        asset_close = _fetch(TICKERS[name])
        if asset_close is None:
            logger.debug("[cross_asset] %s unavailable", name)
            continue
        pa = _analyse_pair(name, btc_close, asset_close)
        pairs[name] = pa
        corrs[name] = pa.rolling_corr
        available  += 1

    if available == 0:
        return CrossAssetResult(
            cross_asset_regime    = "TRANSITION",
            relative_strength_score = 50.0,
            regime_confidence     = 30.0,
            pair_analyses         = {},
            rolling_correlations  = {},
            decoupling_detected   = False,
            decoupling_direction  = "NONE",
            btc_beta_to_spy       = 1.0,
            btc_beta_to_qqq       = 1.5,
            dominant_driver       = "UNKNOWN",
            interpretation        = "All comparator assets unavailable",
        )

    # ── Beta calculations ─────────────────────────────────────────────────────
    beta_spy = 1.0
    beta_qqq = 1.5
    if "SPY" in pairs:
        spy_close = _fetch(TICKERS["SPY"])
        if spy_close is not None:
            spy_ret  = np.log(spy_close / spy_close.shift(1)).dropna()
            beta_spy = _rolling_beta(btc_ret, spy_ret)
    if "QQQ" in pairs:
        qqq_close = _fetch(TICKERS["QQQ"])
        if qqq_close is not None:
            qqq_ret  = np.log(qqq_close / qqq_close.shift(1)).dropna()
            beta_qqq = _rolling_beta(btc_ret, qqq_ret)

    # ── Composite relative strength score ─────────────────────────────────────
    # Weighted by importance — QQQ and SPY matter most
    rs_weights = {"QQQ": 0.30, "SPY": 0.25, "GLD": 0.20, "DXY": 0.15, "TLT": 0.10}
    total_w    = sum(rs_weights[k] for k in rs_weights if k in pairs)
    if total_w > 0:
        rs_composite = sum(
            pairs[k].btc_rs_score * rs_weights[k]
            for k in rs_weights if k in pairs
        ) / total_w
    else:
        rs_composite = 50.0
    rs_composite = round(rs_composite, 1)

    # ── Dominant driver (highest |correlation|) ───────────────────────────────
    if corrs:
        dominant_driver = max(corrs, key=lambda k: abs(corrs[k]))
    else:
        dominant_driver = "UNKNOWN"

    # ── Cross-asset regime ────────────────────────────────────────────────────
    regime, conf, decoupling, dec_dir = _classify_cross_asset_regime(pairs, btc_ret20)

    # ── Interpretation ────────────────────────────────────────────────────────
    regime_emoji = {
        "RISK_ON_ALIGNED":  "🟢",
        "RISK_OFF_ALIGNED": "🔴",
        "DECOUPLED_BULL":   "🚀",
        "DECOUPLED_BEAR":   "⚠️",
        "TRANSITION":       "↔️",
    }.get(regime, "❓")

    corr_str = " | ".join(
        f"{k}={corrs[k]:+.2f}" for k in ["QQQ", "SPY", "DXY", "GLD", "TLT"] if k in corrs
    )

    interp = (
        f"{regime_emoji} {regime} (conf={conf:.0f}% | RS={rs_composite:.0f}/100) — "
        f"BTC 20d={btc_ret20:+.1f}% | β_SPY={beta_spy:.2f} β_QQQ={beta_qqq:.2f} | "
        f"Corr: {corr_str}"
    )

    logger.info(
        "[cross_asset] %s conf=%.1f RS=%.1f btc_20d=%+.1f%% "
        "beta_spy=%.2f decouple=%s dominant=%s",
        regime, conf, rs_composite, btc_ret20,
        beta_spy, decoupling, dominant_driver,
    )

    return CrossAssetResult(
        cross_asset_regime      = regime,
        relative_strength_score = rs_composite,
        regime_confidence       = conf,
        pair_analyses           = {k: v for k, v in pairs.items()},
        rolling_correlations    = {k: round(v, 3) for k, v in corrs.items()},
        decoupling_detected     = decoupling,
        decoupling_direction    = dec_dir,
        btc_beta_to_spy         = round(beta_spy, 3),
        btc_beta_to_qqq         = round(beta_qqq, 3),
        dominant_driver         = dominant_driver,
        interpretation          = interp,
    )
