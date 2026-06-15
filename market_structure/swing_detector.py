"""
Swing High / Swing Low Detector – Vectorised version
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

logger = logging.getLogger(__name__)

DEFAULT_WINDOW = 5


@dataclass(frozen=True)
class SwingPoint:
    index:  int
    date:   str
    price:  float
    kind:   str


def detect_swings(df: pd.DataFrame, window: int = DEFAULT_WINDOW) -> tuple[list[SwingPoint], list[SwingPoint]]:
    """
    Detect swing highs and lows using scipy.signal.argrelextrema.
    Much faster than manual loop for large data.
    """
    high_arr = df["High"].to_numpy()
    low_arr = df["Low"].to_numpy()
    dates = df.index.astype(str).tolist()

    # Find local maxima (swing highs)
    swing_high_idx = argrelextrema(high_arr, np.greater, order=window)[0]
    swing_low_idx = argrelextrema(low_arr, np.less, order=window)[0]

    highs = [
        SwingPoint(index=int(i), date=dates[i], price=float(high_arr[i]), kind="HIGH")
        for i in swing_high_idx
    ]
    lows = [
        SwingPoint(index=int(i), date=dates[i], price=float(low_arr[i]), kind="LOW")
        for i in swing_low_idx
    ]

    return highs, lows


def get_recent_swings(
    df: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
    n_recent: int = 5,
) -> dict:
    highs, lows = detect_swings(df, window)
    return {
        "recent_highs":    highs[-n_recent:],
        "recent_lows":     lows[-n_recent:],
        "last_swing_high": highs[-1] if highs else None,
        "last_swing_low":  lows[-1] if lows else None,
        "all_highs":       highs,
        "all_lows":        lows,
    }