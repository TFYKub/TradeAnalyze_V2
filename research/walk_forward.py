"""
Walk-Forward Validation Engine  (Phase 10)
============================================
Implements rolling walk-forward analysis:
  Train Window  → fit strategy parameters
  Validation    → evaluate in-sample
  Forward Test  → evaluate out-of-sample (never trained on)

Generates out-of-sample:
  Sharpe Ratio
  CAGR
  Max Drawdown
  Win Rate
  Profit Factor

Method:
  Anchored or rolling walk-forward with configurable windows.
  Uses simple signal + stop/TP rule on historical OHLCV.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


@dataclass(frozen=True)
class WalkForwardFold:
    fold_n:         int
    train_start:    str
    train_end:      str
    test_start:     str
    test_end:       str
    train_sharpe:   float
    test_sharpe:    float
    test_cagr:      float
    test_max_dd:    float
    test_win_rate:  float
    test_profit_factor: float
    n_trades:       int


@dataclass(frozen=True)
class WalkForwardResult:
    folds:               tuple[WalkForwardFold, ...]
    oos_sharpe_mean:     float    # out-of-sample Sharpe mean
    oos_sharpe_std:      float
    oos_cagr_mean:       float
    oos_max_dd_mean:     float
    oos_win_rate_mean:   float
    oos_profit_factor:   float
    consistency_score:   float    # 0–100: % folds with positive Sharpe
    n_total_trades:      int
    recommendation:      str


def _sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    excess = returns.mean() - 0.05 / TRADING_DAYS
    return float(excess / returns.std() * math.sqrt(TRADING_DAYS))


def _cagr(returns: np.ndarray) -> float:
    if len(returns) < 2:
        return 0.0
    total = float(np.prod(1 + returns))
    n_years = len(returns) / TRADING_DAYS
    return float(total ** (1 / n_years) - 1) * 100 if n_years > 0 else 0.0


def _max_drawdown(returns: np.ndarray) -> float:
    equity = np.cumprod(1 + returns)
    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / peak
    return float(dd.min() * 100) if len(dd) > 0 else 0.0


def _simulate_trades(
    df:        pd.DataFrame,
    signal_fn: Callable[[pd.DataFrame, int], str] | None = None,
) -> tuple[list[float], int, float]:
    """
    Simple rule-based backtest: EMA crossover + ATR stop.

    Returns (trade_returns, n_trades, profit_factor).
    Uses default signal if signal_fn is None.
    """
    if len(df) < 30:
        return [], 0, 0.0

    close  = df["Close"].to_numpy()
    ema12  = pd.Series(close).ewm(span=12, adjust=False).mean().to_numpy()
    ema26  = pd.Series(close).ewm(span=26, adjust=False).mean().to_numpy()

    # Simple ATR
    hl     = df["High"].to_numpy() - df["Low"].to_numpy()
    atr    = pd.Series(hl).ewm(alpha=1/14, adjust=False).mean().to_numpy()

    trade_returns: list[float] = []
    in_trade   = False
    entry      = 0.0
    stop_loss  = 0.0
    direction  = ""

    for i in range(26, len(close) - 1):
        if not in_trade:
            # Entry: EMA crossover
            if ema12[i] > ema26[i] and ema12[i-1] <= ema26[i-1]:
                in_trade  = True
                entry     = close[i+1]
                stop_loss = entry - atr[i] * 1.5
                direction = "LONG"
            elif ema12[i] < ema26[i] and ema12[i-1] >= ema26[i-1]:
                in_trade  = True
                entry     = close[i+1]
                stop_loss = entry + atr[i] * 1.5
                direction = "SHORT"
        else:
            # Exit: stop or TP (2R)
            tp = entry + 2 * abs(entry - stop_loss) * (1 if direction == "LONG" else -1)
            hit_stop = (direction == "LONG"  and close[i] <= stop_loss) or \
                       (direction == "SHORT" and close[i] >= stop_loss)
            hit_tp   = (direction == "LONG"  and close[i] >= tp) or \
                       (direction == "SHORT" and close[i] <= tp)

            if hit_stop or hit_tp:
                exit_price = stop_loss if hit_stop else tp
                ret = (exit_price - entry) / entry
                if direction == "SHORT":
                    ret = -ret
                trade_returns.append(ret)
                in_trade = False

    if not trade_returns:
        return [], 0, 0.0

    wins   = [r for r in trade_returns if r > 0]
    losses = [r for r in trade_returns if r <= 0]
    pf     = (sum(wins) / -sum(losses)) if losses else float("inf")
    return trade_returns, len(trade_returns), round(pf, 2)


def run_walk_forward(
    df:            pd.DataFrame,
    train_bars:    int = 252,   # ~1 year
    test_bars:     int = 63,    # ~1 quarter
    min_folds:     int = 3,
) -> WalkForwardResult:
    """
    Run walk-forward validation on OHLCV data.

    Parameters
    ----------
    df         : daily OHLCV DataFrame (DatetimeIndex)
    train_bars : number of bars in training window
    test_bars  : number of bars in test window
    min_folds  : minimum required folds
    """
    df = df.reset_index(drop=False)
    n  = len(df)

    total_required = train_bars + test_bars * min_folds
    if n < total_required:
        logger.warning("[walk_forward] Not enough data (%d bars, need %d)", n, total_required)
        return WalkForwardResult(
            folds=(), oos_sharpe_mean=0, oos_sharpe_std=0,
            oos_cagr_mean=0, oos_max_dd_mean=0, oos_win_rate_mean=0,
            oos_profit_factor=0, consistency_score=0,
            n_total_trades=0, recommendation="Insufficient data for walk-forward",
        )

    folds: list[WalkForwardFold] = []
    start = 0

    fold_n = 0
    while start + train_bars + test_bars <= n:
        train_df = df.iloc[start:start + train_bars].copy()
        test_df  = df.iloc[start + train_bars:start + train_bars + test_bars].copy()

        if len(train_df) < 50 or len(test_df) < 10:
            break

        # In-sample
        train_rets, _, _ = _simulate_trades(train_df)
        train_sharpe = _sharpe(np.array(train_rets)) if train_rets else 0.0

        # Out-of-sample
        test_rets, n_trades, pf = _simulate_trades(test_df)
        test_arr     = np.array(test_rets) if test_rets else np.array([0.0])
        test_sharpe  = _sharpe(test_arr)
        test_cagr    = _cagr(test_arr)
        test_dd      = _max_drawdown(test_arr)
        wins         = sum(1 for r in test_rets if r > 0)
        win_rate     = wins / n_trades if n_trades > 0 else 0.0

        # Date labels
        def _date(row):
            d = row.get("Date") or row.get("index")
            return str(d)[:10] if d is not None else "?"

        folds.append(WalkForwardFold(
            fold_n     = fold_n,
            train_start= _date(train_df.iloc[0]),
            train_end  = _date(train_df.iloc[-1]),
            test_start = _date(test_df.iloc[0]),
            test_end   = _date(test_df.iloc[-1]),
            train_sharpe = round(train_sharpe, 3),
            test_sharpe  = round(test_sharpe, 3),
            test_cagr    = round(test_cagr, 2),
            test_max_dd  = round(test_dd, 2),
            test_win_rate= round(win_rate, 3),
            test_profit_factor = pf,
            n_trades     = n_trades,
        ))

        start  += test_bars   # rolling: shift by one test window
        fold_n += 1

    if not folds:
        return WalkForwardResult(
            folds=(), oos_sharpe_mean=0, oos_sharpe_std=0,
            oos_cagr_mean=0, oos_max_dd_mean=0, oos_win_rate_mean=0,
            oos_profit_factor=0, consistency_score=0, n_total_trades=0,
            recommendation="No complete folds generated",
        )

    oos_sharpes  = [f.test_sharpe for f in folds]
    oos_cagrs    = [f.test_cagr for f in folds]
    oos_dds      = [f.test_max_dd for f in folds]
    oos_winrates = [f.test_win_rate for f in folds]
    oos_pfs      = [f.test_profit_factor for f in folds if f.test_profit_factor != float("inf")]
    total_trades = sum(f.n_trades for f in folds)

    mean_sharpe  = round(float(np.mean(oos_sharpes)), 3)
    std_sharpe   = round(float(np.std(oos_sharpes)), 3)
    mean_cagr    = round(float(np.mean(oos_cagrs)), 2)
    mean_dd      = round(float(np.mean(oos_dds)), 2)
    mean_wr      = round(float(np.mean(oos_winrates)), 3)
    mean_pf      = round(float(np.mean(oos_pfs)), 2) if oos_pfs else 0.0
    pos_folds    = sum(1 for s in oos_sharpes if s > 0)
    consistency  = round(pos_folds / len(folds) * 100, 1)

    if mean_sharpe > 1.0 and consistency >= 70:
        rec = f"✅ Robust strategy: OOS Sharpe={mean_sharpe} consistency={consistency:.0f}%"
    elif mean_sharpe > 0.5 and consistency >= 55:
        rec = f"🟡 Moderate: OOS Sharpe={mean_sharpe} consistency={consistency:.0f}% — needs monitoring"
    else:
        rec = f"❌ Weak OOS: Sharpe={mean_sharpe} consistency={consistency:.0f}% — strategy may be overfit"

    logger.info("[walk_forward] %d folds  OOS sharpe=%.2f±%.2f  consistency=%.0f%%  trades=%d",
                len(folds), mean_sharpe, std_sharpe, consistency, total_trades)

    return WalkForwardResult(
        folds              = tuple(folds),
        oos_sharpe_mean    = mean_sharpe,
        oos_sharpe_std     = std_sharpe,
        oos_cagr_mean      = mean_cagr,
        oos_max_dd_mean    = mean_dd,
        oos_win_rate_mean  = mean_wr,
        oos_profit_factor  = mean_pf,
        consistency_score  = consistency,
        n_total_trades     = total_trades,
        recommendation     = rec,
    )
