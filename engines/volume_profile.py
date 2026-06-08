"""
Volume Profile Engine  (Phase 6)
==================================
Calculates: POC | HVN | LVN | Value Area High | Value Area Low

Algorithm: TPO (Time-Price-Opportunity) histogram
  1. Bin price range into N buckets
  2. Sum volume in each bucket
  3. POC = bucket with max volume
  4. Value Area = 70% of total volume around POC
  5. HVN = buckets with > 1.5× average volume
  6. LVN = buckets with < 0.5× average volume
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
N_BINS    = 50      # price buckets
VA_TARGET = 0.70    # 70% of volume = Value Area


@dataclass(frozen=True)
class VolumeLevelResult:
    price:         float
    volume:        float
    volume_pct:    float    # % of total volume at this level
    label:         str      # POC | HVN | LVN | VA_HIGH | VA_LOW

@dataclass(frozen=True)
class VolumeProfileResult:
    poc:           float    # Point of Control
    va_high:       float    # Value Area High
    va_low:        float    # Value Area Low
    hvn_levels:    tuple[float, ...]    # High Volume Nodes
    lvn_levels:    tuple[float, ...]    # Low Volume Nodes
    total_volume:  float
    price_range:   tuple[float, float]
    institutional_bias: str    # "BULLISH" | "BEARISH" | "NEUTRAL"
    support_from_poc:   float  # distance from current price to POC
    resistance_from_poc: float

def compute_volume_profile(
    df: pd.DataFrame,
    lookback: int = 60,
) -> VolumeProfileResult:
    """
    Compute volume profile over last *lookback* bars.

    Parameters
    ----------
    df       : OHLCV DataFrame
    lookback : number of bars to include
    """
    data   = df.iloc[-lookback:].copy()
    close  = float(data["Close"].iloc[-1])
    hi     = float(data["High"].max())
    lo     = float(data["Low"].min())

    if hi <= lo:
        hi = lo * 1.01

    bins   = np.linspace(lo, hi, N_BINS + 1)
    bucket_volume = np.zeros(N_BINS)

    for _, row in data.iterrows():
        bar_lo, bar_hi = float(row["Low"]), float(row["High"])
        vol = float(row["Volume"])
        for i in range(N_BINS):
            overlap_lo = max(bar_lo, bins[i])
            overlap_hi = min(bar_hi, bins[i + 1])
            if overlap_hi > overlap_lo:
                bucket_vol = vol * (overlap_hi - overlap_lo) / (bar_hi - bar_lo + 1e-9)
                bucket_volume[i] += bucket_vol

    bucket_mid   = (bins[:-1] + bins[1:]) / 2
    total_volume = float(bucket_volume.sum())

    if total_volume <= 0:
        # Fallback when volume data is missing
        poc = close
        return VolumeProfileResult(
            poc=poc, va_high=poc*1.02, va_low=poc*0.98,
            hvn_levels=(poc,), lvn_levels=(),
            total_volume=0, price_range=(lo, hi),
            institutional_bias="NEUTRAL",
            support_from_poc=0, resistance_from_poc=0,
        )

    # POC
    poc_idx = int(np.argmax(bucket_volume))
    poc     = round(float(bucket_mid[poc_idx]), 4)

    # Value Area (expand from POC until 70% of volume)
    va_lo_idx = poc_idx
    va_hi_idx = poc_idx
    va_volume = float(bucket_volume[poc_idx])

    while va_volume < VA_TARGET * total_volume:
        add_lo = float(bucket_volume[va_lo_idx - 1]) if va_lo_idx > 0 else 0
        add_hi = float(bucket_volume[va_hi_idx + 1]) if va_hi_idx < N_BINS - 1 else 0
        if add_lo > add_hi and va_lo_idx > 0:
            va_lo_idx -= 1; va_volume += add_lo
        elif va_hi_idx < N_BINS - 1:
            va_hi_idx += 1; va_volume += add_hi
        else:
            break

    va_high = round(float(bucket_mid[va_hi_idx]), 4)
    va_low  = round(float(bucket_mid[va_lo_idx]), 4)

    # HVN (> 1.5× average) and LVN (< 0.5× average)
    avg_vol = total_volume / N_BINS
    hvn = tuple(round(float(bucket_mid[i]), 4) for i in range(N_BINS)
                if bucket_volume[i] >= avg_vol * 1.5)
    lvn = tuple(round(float(bucket_mid[i]), 4) for i in range(N_BINS)
                if 0 < bucket_volume[i] <= avg_vol * 0.5)

    # Institutional bias: where is price relative to POC + VA?
    if close > va_high:
        bias = "BULLISH"   # price above value area — buyers in control
    elif close < va_low:
        bias = "BEARISH"   # price below value area — sellers in control
    else:
        bias = "NEUTRAL"

    support_from_poc    = round(close - poc, 4) if close >= poc else 0
    resistance_from_poc = round(poc - close, 4) if close < poc else 0

    logger.info("[vol_profile] POC=%.2f VA=[%.2f-%.2f] bias=%s HVN=%d LVN=%d",
                poc, va_low, va_high, bias, len(hvn), len(lvn))

    return VolumeProfileResult(
        poc=poc, va_high=va_high, va_low=va_low,
        hvn_levels=hvn[:5], lvn_levels=lvn[:5],
        total_volume=round(total_volume, 0),
        price_range=(round(lo, 4), round(hi, 4)),
        institutional_bias=bias,
        support_from_poc=support_from_poc,
        resistance_from_poc=resistance_from_poc,
    )
