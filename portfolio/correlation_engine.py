"""
Portfolio Correlation Engine  (Phase 8)
=========================================
Builds rolling 60-day correlation matrix across:
  BTC, SPY, QQQ, VIX, DXY, GLD, GC=F, CL=F, TLT

Outputs:
  correlation_matrix   : dict[str, dict[str, float]]
  diversification_score: 0–100 (100 = perfectly uncorrelated)
  high_corr_pairs      : pairs with |corr| > 0.70
  low_corr_pairs       : pairs with |corr| < 0.30  (diversifiers)
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

UNIVERSE = ["BTC-USD", "SPY", "QQQ", "^VIX", "DX-Y.NYB", "GLD", "GC=F", "CL=F", "TLT"]
HIGH_CORR_THRESHOLD = 0.70
LOW_CORR_THRESHOLD  = 0.30
LOOKBACK_DAYS       = 60


@dataclass(frozen=True)
class CorrelationResult:
    correlation_matrix:   dict[str, dict[str, float]]
    diversification_score: float             # 0–100
    high_corr_pairs:      tuple[tuple[str, str, float], ...]
    low_corr_pairs:       tuple[tuple[str, str, float], ...]
    avg_correlation:      float
    symbols_used:         tuple[str, ...]


def compute_correlations(
    prices_dict: dict[str, pd.Series],   # {symbol: daily close series}
    lookback:    int = LOOKBACK_DAYS,
) -> CorrelationResult:
    """
    Compute rolling correlation matrix from price series.

    Parameters
    ----------
    prices_dict : {symbol: pd.Series of daily close prices}
    lookback    : number of bars for correlation window
    """
    valid = {k: v.dropna() for k, v in prices_dict.items() if len(v.dropna()) >= lookback}
    if len(valid) < 2:
        return CorrelationResult(
            correlation_matrix={}, diversification_score=50.0,
            high_corr_pairs=(), low_corr_pairs=(), avg_correlation=0.0,
            symbols_used=tuple(valid.keys()),
        )

    # Align on common dates, compute log returns
    df = pd.DataFrame({k: np.log(v / v.shift(1)) for k, v in valid.items()}).dropna()
    df = df.iloc[-lookback:]

    if df.empty or len(df) < 10:
        return CorrelationResult(
            correlation_matrix={}, diversification_score=50.0,
            high_corr_pairs=(), low_corr_pairs=(), avg_correlation=0.0,
            symbols_used=tuple(valid.keys()),
        )

    corr_df = df.corr()
    symbols = list(corr_df.columns)

    # Build dict matrix
    corr_matrix = {
        s: {t: round(float(corr_df.loc[s, t]), 3) for t in symbols}
        for s in symbols
    }

    # Pair analysis
    high_corr: list[tuple[str, str, float]] = []
    low_corr:  list[tuple[str, str, float]] = []
    all_corr:  list[float] = []

    for i, s1 in enumerate(symbols):
        for s2 in symbols[i+1:]:
            c = corr_matrix[s1][s2]
            all_corr.append(abs(c))
            if abs(c) >= HIGH_CORR_THRESHOLD:
                high_corr.append((s1, s2, c))
            if abs(c) <= LOW_CORR_THRESHOLD:
                low_corr.append((s1, s2, c))

    avg_corr = round(float(np.mean(all_corr)), 3) if all_corr else 0.0

    # Diversification score: 100 = all uncorrelated, 0 = all perfectly correlated
    div_score = round((1 - avg_corr) * 100, 1)

    high_corr.sort(key=lambda x: -abs(x[2]))
    low_corr.sort(key=lambda x: abs(x[2]))

    logger.info("[correlation] %d symbols  avg_corr=%.2f  div_score=%.0f  high=%d low=%d",
                len(symbols), avg_corr, div_score, len(high_corr), len(low_corr))

    return CorrelationResult(
        correlation_matrix   = corr_matrix,
        diversification_score= div_score,
        high_corr_pairs      = tuple(high_corr[:5]),
        low_corr_pairs       = tuple(low_corr[:5]),
        avg_correlation      = avg_corr,
        symbols_used         = tuple(symbols),
    )
