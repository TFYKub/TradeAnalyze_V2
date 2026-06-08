"""
Volatility Regime Engine  (Phase 3)
=====================================
Classifies: LOW_VOL | NORMAL_VOL | HIGH_VOL | PANIC_VOL

Using:
  • Historical Volatility (HV20, HV60)
  • ATR as % of price
  • Realised Vol (5-day)
  • Volatility of Volatility (VoV — std of rolling HV)

Output: VolatilityRegimeResult
  regime            : LOW_VOL | NORMAL_VOL | HIGH_VOL | PANIC_VOL
  vol_score         : 0–100
  vol_multiplier    : position size / stop multiplier
  recommended_action: how to adjust position + stop

Modifies:
  • Position size (smaller in HIGH/PANIC)
  • Stop distance (wider in HIGH/PANIC)
  • Strategy selection (options strategy changes with vol regime)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


@dataclass(frozen=True)
class VolatilityRegimeResult:
    regime:              str      # LOW_VOL | NORMAL_VOL | HIGH_VOL | PANIC_VOL
    vol_score:           float    # 0–100 (100 = extreme vol)
    hv20:                float    # 20-day historical vol (annualised)
    hv60:                float    # 60-day historical vol
    hv5:                 float    # 5-day realised vol (current condition)
    atr_pct:             float    # ATR/price %
    vov:                 float    # volatility of volatility
    iv_hv_ratio:         float    # IV/HV (> 1 = options expensive)

    # Adjustments
    position_size_mult:  float    # multiply base position size by this
    stop_distance_mult:  float    # multiply base stop distance by this

    # Recommendations
    preferred_strategy:  str      # hint for options strategy
    recommended_action:  str      # human-readable instruction


def _hv(close: pd.Series, window: int) -> float:
    log_ret = np.log(close / close.shift(1)).dropna()
    if len(log_ret) < window:
        return float(log_ret.std() * math.sqrt(TRADING_DAYS))
    return float(log_ret.iloc[-window:].std() * math.sqrt(TRADING_DAYS))


def _atr_pct(df: pd.DataFrame) -> float:
    hl  = df["High"] - df["Low"]
    hc  = (df["High"] - df["Close"].shift()).abs()
    lc  = (df["Low"]  - df["Close"].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = float(tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1])
    price = float(df["Close"].iloc[-1])
    return atr / price * 100 if price > 0 else 0


def _vov(close: pd.Series, window: int = 20) -> float:
    """Volatility of Volatility — std of rolling 20-day HV."""
    log_ret = np.log(close / close.shift(1)).dropna()
    roll_hv = log_ret.rolling(window).std() * math.sqrt(TRADING_DAYS)
    if len(roll_hv.dropna()) < 5:
        return 0.0
    return float(roll_hv.dropna().std())


def compute_volatility_regime(
    df:      pd.DataFrame,
    iv:      float | None = None,   # current IV from option chain (optional)
) -> VolatilityRegimeResult:
    """
    Classify current volatility regime and return adjustment multipliers.

    Parameters
    ----------
    df : daily OHLCV DataFrame (≥ 60 bars recommended)
    iv : annualised IV from option chain (0–1 scale, optional)
    """

    close    = df["Close"]
    hv5      = round(_hv(close, 5),  4)
    hv20     = round(_hv(close, 20), 4)
    hv60     = round(_hv(close, 60), 4)
    atr_pct  = round(_atr_pct(df), 3)
    vov_val  = round(_vov(close, 20), 4)

    # IV/HV ratio
    iv_hv = round(iv / hv20, 3) if (iv and hv20 > 0) else 1.0

    # Vol score (0–100): weighted by short-term / ATR / VoV
    hv_score   = min(100, hv20 * 200)     # 0.50 (50% HV) → 100
    atr_score  = min(100, atr_pct * 25)   # 4% ATR → 100
    vov_score  = min(100, vov_val * 500)  # 0.20 VoV → 100
    vol_score  = round(hv_score * 0.40 + atr_score * 0.40 + vov_score * 0.20, 1)

    # Regime classification
    if vol_score >= 80 or atr_pct >= 4.0:
        regime             = "PANIC_VOL"
        pos_mult           = 0.30
        stop_mult          = 2.0
        preferred_strategy = "LONG_STRADDLE or IRON_CONDOR (post-spike)"
        action             = "⚠️ Panic vol — reduce position 70%, widen stops 2×"
    elif vol_score >= 55 or atr_pct >= 2.5:
        regime             = "HIGH_VOL"
        pos_mult           = 0.60
        stop_mult          = 1.5
        preferred_strategy = "IRON_CONDOR or SHORT_STRANGLE (sell premium)"
        action             = "High vol — reduce position 40%, widen stops 1.5×"
    elif vol_score >= 25 or atr_pct >= 1.0:
        regime             = "NORMAL_VOL"
        pos_mult           = 1.00
        stop_mult          = 1.0
        preferred_strategy = "BULL_CALL_SPREAD or PUT_DEBIT_SPREAD"
        action             = "Normal vol — standard position and stop"
    else:
        regime             = "LOW_VOL"
        pos_mult           = 1.20   # can slightly increase size
        stop_mult          = 0.75   # tighter stop in low vol
        preferred_strategy = "LONG_CALL or LONG_PUT (cheap premium)"
        action             = "Low vol — can increase size 20%, tighten stops 0.75×"

    logger.info(
        "[vol_regime] %s score=%.0f hv20=%.1f%% hv5=%.1f%% atr=%.2f%% vov=%.3f",
        regime, vol_score, hv20*100, hv5*100, atr_pct, vov_val
    )

    return VolatilityRegimeResult(
        regime              = regime,
        vol_score           = vol_score,
        hv20                = hv20,
        hv60                = hv60,
        hv5                 = hv5,
        atr_pct             = atr_pct,
        vov                 = vov_val,
        iv_hv_ratio         = iv_hv,
        position_size_mult  = pos_mult,
        stop_distance_mult  = stop_mult,
        preferred_strategy  = preferred_strategy,
        recommended_action  = action,
    )
