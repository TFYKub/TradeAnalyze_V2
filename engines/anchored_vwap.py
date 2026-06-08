"""
Anchored VWAP Engine  (Phase 6)
=================================
Calculates AVWAP anchored to:
  • Start of current Month
  • Start of current Quarter
  • Start of current Year
  • 52-week high/low dates (event AVWAP)

AVWAP = cumsum(Price × Volume) / cumsum(Volume)
  where Price = typical price (H+L+C)/3
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class AVWAPResult:
    monthly_vwap:   float
    quarterly_vwap: float
    yearly_vwap:    float
    event_vwap:     float | None   # anchored to 52w high or low

    # Distances from current price
    monthly_dist_pct:   float
    quarterly_dist_pct: float
    yearly_dist_pct:    float

    # Trend interpretation
    avwap_trend:    str    # "BULLISH" | "BEARISH" | "MIXED" | "NEUTRAL"
    above_count:    int    # how many AVWAPs is price above?
    vwap_support:   float | None   # closest AVWAP below price
    vwap_resistance:float | None   # closest AVWAP above price


def _vwap_from(df: pd.DataFrame, anchor_date: pd.Timestamp) -> float:
    """Compute AVWAP from anchor_date to last bar."""
    subset = df[df.index >= anchor_date]
    if subset.empty:
        return float(df["Close"].iloc[-1])
    tp  = (subset["High"] + subset["Low"] + subset["Close"]) / 3
    vol = subset["Volume"]
    denom = float(vol.sum())
    if denom <= 0:
        return float(tp.iloc[-1])
    return float((tp * vol).sum() / denom)


def compute_anchored_vwap(df: pd.DataFrame) -> AVWAPResult:
    """
    Compute monthly / quarterly / yearly / event AVWAPs.

    Parameters
    ----------
    df : daily OHLCV DataFrame with DatetimeIndex
    """
    df = df.copy()
    df.index = pd.DatetimeIndex(df.index)

    last_date  = df.index[-1]
    last_price = float(df["Close"].iloc[-1])

    # Anchor dates
    month_start = pd.Timestamp(last_date.year, last_date.month, 1)
    q_month     = ((last_date.month - 1) // 3) * 3 + 1
    quarter_start = pd.Timestamp(last_date.year, q_month, 1)
    year_start  = pd.Timestamp(last_date.year, 1, 1)

    m_vwap = round(_vwap_from(df, month_start), 4)
    q_vwap = round(_vwap_from(df, quarter_start), 4)
    y_vwap = round(_vwap_from(df, year_start), 4)

    # Event AVWAP: anchor to 52-week high or low
    year_data = df.iloc[-252:] if len(df) >= 252 else df
    high_idx  = year_data["High"].idxmax()
    low_idx   = year_data["Low"].idxmin()
    # Anchor to whichever extreme is more recent
    if abs((last_date - high_idx).days) < abs((last_date - low_idx).days):
        event_anchor = high_idx
    else:
        event_anchor = low_idx
    e_vwap = round(_vwap_from(df, event_anchor), 4)

    # Distances
    def dist_pct(vwap): return round((last_price - vwap) / vwap * 100, 2) if vwap > 0 else 0

    m_dist = dist_pct(m_vwap)
    q_dist = dist_pct(q_vwap)
    y_dist = dist_pct(y_vwap)

    avwaps = [m_vwap, q_vwap, y_vwap, e_vwap]
    above  = sum(1 for v in avwaps if last_price > v)

    if above >= 3:
        trend = "BULLISH"
    elif above <= 1:
        trend = "BEARISH"
    elif above == 2:
        trend = "MIXED"
    else:
        trend = "NEUTRAL"

    below_avwaps  = sorted([v for v in avwaps if v <= last_price], reverse=True)
    above_avwaps  = sorted([v for v in avwaps if v > last_price])
    support      = below_avwaps[0]  if below_avwaps else None
    resistance   = above_avwaps[0] if above_avwaps else None

    logger.info("[avwap] monthly=%.2f quarterly=%.2f yearly=%.2f trend=%s above=%d",
                m_vwap, q_vwap, y_vwap, trend, above)

    return AVWAPResult(
        monthly_vwap=m_vwap, quarterly_vwap=q_vwap,
        yearly_vwap=y_vwap, event_vwap=e_vwap,
        monthly_dist_pct=m_dist, quarterly_dist_pct=q_dist, yearly_dist_pct=y_dist,
        avwap_trend=trend, above_count=above,
        vwap_support=support, vwap_resistance=resistance,
    )
