"""
Volume Profile Engine – Vectorised with numpy.histogram
"""
from __future__ import annotations
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
N_BINS    = 50
VA_TARGET = 0.70


@dataclass(frozen=True)
class VolumeProfileResult:
    poc:           float
    va_high:       float
    va_low:        float
    hvn_levels:    tuple[float, ...]
    lvn_levels:    tuple[float, ...]
    total_volume:  float
    price_range:   tuple[float, float]
    institutional_bias: str
    support_from_poc:   float
    resistance_from_poc: float


def compute_volume_profile(
    df: pd.DataFrame,
    lookback: int = 60,
) -> VolumeProfileResult:
    data = df.iloc[-lookback:].copy()
    close = float(data["Close"].iloc[-1])
    hi = float(data["High"].max())
    lo = float(data["Low"].min())

    if hi <= lo:
        hi = lo * 1.01

    bins = np.linspace(lo, hi, N_BINS + 1)
    bucket_volume = np.zeros(N_BINS)
    bucket_mid = (bins[:-1] + bins[1:]) / 2

    # Vectorised volume distribution using weighted histogram per bar
    # Instead of nested loops, we iterate over bars (still necessary because of overlap calculation)
    # but inner loop is minimised by using `searchsorted` to find bin range.
    for _, row in data.iterrows():
        bar_lo = row["Low"]
        bar_hi = row["High"]
        vol = row["Volume"]
        bar_range = bar_hi - bar_lo
        if bar_range <= 0:
            continue

        # Find indices of bins that overlap with this bar
        left_idx = np.searchsorted(bins, bar_lo, side='right') - 1
        right_idx = np.searchsorted(bins, bar_hi, side='left')
        left_idx = max(0, left_idx)
        right_idx = min(N_BINS, right_idx)

        for i in range(left_idx, right_idx):
            overlap_lo = max(bar_lo, bins[i])
            overlap_hi = min(bar_hi, bins[i+1])
            overlap = overlap_hi - overlap_lo
            if overlap > 0:
                bucket_volume[i] += vol * overlap / bar_range

    total_volume = float(bucket_volume.sum())

    if total_volume <= 0:
        return VolumeProfileResult(
            poc=close, va_high=close*1.02, va_low=close*0.98,
            hvn_levels=(close,), lvn_levels=(),
            total_volume=0, price_range=(lo, hi),
            institutional_bias="NEUTRAL",
            support_from_poc=0, resistance_from_poc=0,
        )

    poc_idx = int(np.argmax(bucket_volume))
    poc = round(float(bucket_mid[poc_idx]), 4)

    # Value Area expansion (still requires loop because expansion order depends on volume)
    va_lo_idx = poc_idx
    va_hi_idx = poc_idx
    va_volume = float(bucket_volume[poc_idx])

    while va_volume < VA_TARGET * total_volume:
        add_lo = float(bucket_volume[va_lo_idx - 1]) if va_lo_idx > 0 else 0
        add_hi = float(bucket_volume[va_hi_idx + 1]) if va_hi_idx < N_BINS - 1 else 0
        if add_lo > add_hi and va_lo_idx > 0:
            va_lo_idx -= 1
            va_volume += add_lo
        elif va_hi_idx < N_BINS - 1:
            va_hi_idx += 1
            va_volume += add_hi
        else:
            break

    va_high = round(float(bucket_mid[va_hi_idx]), 4)
    va_low = round(float(bucket_mid[va_lo_idx]), 4)

    avg_vol = total_volume / N_BINS
    hvn = tuple(round(float(bucket_mid[i]), 4) for i in range(N_BINS)
                if bucket_volume[i] >= avg_vol * 1.5)
    lvn = tuple(round(float(bucket_mid[i]), 4) for i in range(N_BINS)
                if 0 < bucket_volume[i] <= avg_vol * 0.5)

    if close > va_high:
        bias = "BULLISH"
    elif close < va_low:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    support_from_poc = round(close - poc, 4) if close >= poc else 0
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