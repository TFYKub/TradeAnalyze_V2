"""
IV Rank Engine  (Phase 5)
==========================
IV Rank      = (IV_current - IV_52w_low)  / (IV_52w_high - IV_52w_low) × 100
IV Percentile = % of days in past 252d where IV < IV_current

Both use a rolling HV20 proxy when real IV history is unavailable.
"""
from __future__ import annotations
import math
import logging
from dataclasses import dataclass
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
LOOKBACK = 252
TRADING_DAYS = 252


@dataclass(frozen=True)
class IVRankResult:
    iv_current:    float    # annualised (0–1 scale)
    iv_rank:       float    # 0–100
    iv_percentile: float    # 0–100
    iv_52w_high:   float
    iv_52w_low:    float
    iv_environment: str     # HIGH_IV | NORMAL_IV | LOW_IV
    signal:        str      # BUY_VOL | SELL_VOL | NEUTRAL


def compute_iv_rank(
    df: pd.DataFrame,
    current_iv: float | None = None,
) -> IVRankResult:
    """
    Compute IV Rank and Percentile.

    Parameters
    ----------
    df         : daily OHLCV DataFrame (≥ 60 bars)
    current_iv : real IV from option chain (0–1 scale). If None, use HV proxy.
    """
    log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()

    # Rolling 20-day HV as IV proxy
    roll_hv = (log_ret.rolling(20).std() * math.sqrt(TRADING_DAYS)).dropna()
    window  = min(LOOKBACK, len(roll_hv))
    hist    = roll_hv.iloc[-window:]

    iv = (current_iv if (current_iv and current_iv > 0.005)
          else round(float(roll_hv.iloc[-1]) * 1.15, 4))  # HV × 1.15 premium

    iv_hi  = float(hist.max())
    iv_lo  = float(hist.min())
    iv_rank = round((iv - iv_lo) / (iv_hi - iv_lo) * 100, 1) if iv_hi > iv_lo else 50.0
    iv_rank = max(0.0, min(100.0, iv_rank))

    iv_pct  = round(float((hist < iv).mean() * 100), 1)

    if iv_rank >= 65:
        env, sig = "HIGH_IV", "SELL_VOL"
    elif iv_rank <= 30:
        env, sig = "LOW_IV",  "BUY_VOL"
    else:
        env, sig = "NORMAL_IV", "NEUTRAL"

    return IVRankResult(
        iv_current=round(iv, 4), iv_rank=iv_rank, iv_percentile=iv_pct,
        iv_52w_high=round(iv_hi, 4), iv_52w_low=round(iv_lo, 4),
        iv_environment=env, signal=sig,
    )
