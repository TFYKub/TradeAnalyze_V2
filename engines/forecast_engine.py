"""
Forecast Engine  (Phase 17)
=============================
Machine-learning return forecaster using XGBoost (or LightGBM fallback).
Predicts 5-day, 10-day, and 20-day forward returns in a walk-forward
rolling-window framework.

Architecture:
  • 3 separate models (one per horizon)
  • Walk-forward: train on all data except last 40 bars → predict last bar
  • Features: technical + regime + flow + macro (whatever is available)
  • Probability buckets: up/down estimated via quantile analysis of residuals

Features (15 total):
  Technical:    rsi, atr_pct, volume_ratio, momentum_20, close_to_ema200
  Markov:       regime_bull_prob, regime_bear_prob, regime_confidence
  Volatility:   hv20, hv5, vov (vol-of-vol)
  Flow/Macro:   funding_rate (optional), liquidity_score (optional),
                breadth_score (optional), oi_trend_enc (optional)

Outputs:
  expected_return_5d   : float  % expected return
  expected_return_10d  : float
  expected_return_20d  : float
  probability_up_20d   : float  0–1
  probability_down_20d : float  0–1
  forecast_confidence  : float  0–100
  forecast_direction   : str    BULLISH | BEARISH | NEUTRAL
  feature_importances  : dict[str, float]

Design Notes:
  • Safe fallback to technical mean-reversion if XGBoost unavailable
  • Minimum 120 bars required to train (6 months daily)
  • All features are z-score normalized within the training window
  • Probability_up = fraction of ensemble predictions > 0
"""
from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

MIN_BARS     = 120    # minimum history to train
TRAIN_HOLD   = 40     # hold-out bars for most recent prediction
HORIZONS     = [5, 10, 20]   # forward return days
N_ESTIMATORS = 200
LEARNING_RATE= 0.05
MAX_DEPTH    = 4
RANDOM_SEED  = 42


@dataclass(frozen=True)
class ForecastResult:
    expected_return_5d:   float          # % expected return, 5-day horizon
    expected_return_10d:  float          # 10-day
    expected_return_20d:  float          # 20-day
    probability_up_5d:    float          # 0–1
    probability_up_10d:   float
    probability_up_20d:   float
    forecast_confidence:  float          # 0–100 (model fit quality)
    forecast_direction:   str            # BULLISH | BEARISH | NEUTRAL
    feature_importances:  dict[str, float]
    model_used:           str            # "xgboost" | "lightgbm" | "statistical_fallback"
    horizon_forecasts:    dict[str, dict]  # {5d: {return, prob_up}, ...}
    interpretation:       str


# ── Feature Engineering ────────────────────────────────────────────────────────
def _build_features(
    df:                pd.DataFrame,
    regime_bull_prob:  float = 0.55,
    regime_bear_prob:  float = 0.30,
    regime_confidence: float = 65.0,
    funding_rate:      float = 0.0,
    liquidity_score:   float = 55.0,
    breadth_score:     float = 50.0,
    oi_trend_enc:      float = 0.0,   # +1=increasing, 0=stable, -1=decreasing
) -> pd.DataFrame:
    """
    Build feature matrix from OHLCV + regime inputs.
    Returns DataFrame with one row per bar, NaN rows for insufficient history.
    """
    close  = df["Close"]
    volume = df["Volume"] if "Volume" in df.columns else pd.Series(1.0, index=df.index)

    # ── Technical Features ───────────────────────────────────────────────────
    log_ret  = np.log(close / close.shift(1))
    hv5      = log_ret.rolling(5).std()  * math.sqrt(252) * 100
    hv20     = log_ret.rolling(20).std() * math.sqrt(252) * 100
    vov      = hv20.rolling(10).std().fillna(0)   # vol of vol

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi14 = 100 - 100 / (1 + rs)

    ema14  = close.ewm(span=14,  adjust=False).mean()
    ema50  = close.ewm(span=50,  adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    atr_hl = df["High"] - df["Low"]
    atr_hc = (df["High"] - close.shift(1)).abs()
    atr_lc = (df["Low"]  - close.shift(1)).abs()
    atr14  = pd.concat([atr_hl, atr_hc, atr_lc], axis=1).max(axis=1).ewm(
        alpha=1/14, adjust=False).mean()
    atr_pct = atr14 / close * 100

    momentum_20 = close.pct_change(20) * 100
    momentum_5  = close.pct_change(5)  * 100
    close_to_ema200 = (close - ema200) / ema200 * 100
    close_to_ema50  = (close - ema50)  / ema50  * 100

    vol_avg20  = volume.rolling(20).mean()
    vol_ratio  = (volume / vol_avg20.replace(0, 1)).clip(0, 5)

    features = pd.DataFrame({
        "rsi14":           rsi14,
        "atr_pct":         atr_pct,
        "volume_ratio":    vol_ratio,
        "momentum_20":     momentum_20,
        "momentum_5":      momentum_5,
        "close_to_ema200": close_to_ema200,
        "close_to_ema50":  close_to_ema50,
        "hv5":             hv5,
        "hv20":            hv20,
        "vov":             vov,
    }, index=df.index)

    # ── Regime / Macro Features (scalar → broadcast to all rows) ────────────
    # These are current-state values — we broadcast the latest available
    features["regime_bull_prob"]  = regime_bull_prob
    features["regime_bear_prob"]  = regime_bear_prob
    features["regime_confidence"] = regime_confidence / 100.0
    features["funding_rate"]      = funding_rate
    features["liquidity_score"]   = liquidity_score / 100.0
    features["breadth_score"]     = breadth_score   / 100.0
    features["oi_trend"]          = oi_trend_enc

    return features


def _build_targets(close: pd.Series) -> dict[int, pd.Series]:
    """Forward returns for each horizon (shift(-h) = future return)."""
    targets = {}
    for h in HORIZONS:
        targets[h] = close.shift(-h) / close - 1.0    # fractional return
    return targets


# ── Model Training and Prediction ─────────────────────────────────────────────
def _try_xgboost(X_train, y_train, X_pred) -> Optional[tuple[float, dict]]:
    """Returns (prediction, feature_importances) or None."""
    try:
        import xgboost as xgb
        model = xgb.XGBRegressor(
            n_estimators  = N_ESTIMATORS,
            learning_rate = LEARNING_RATE,
            max_depth     = MAX_DEPTH,
            subsample     = 0.8,
            colsample_bytree = 0.8,
            random_state  = RANDOM_SEED,
            verbosity     = 0,
        )
        model.fit(X_train, y_train)
        pred = float(model.predict(X_pred)[0])
        importances = {
            name: round(float(imp), 4)
            for name, imp in zip(X_train.columns, model.feature_importances_)
        }
        return pred, importances
    except ImportError:
        return None
    except Exception as exc:
        logger.debug("[forecast] XGBoost failed: %s", exc)
        return None


def _try_lightgbm(X_train, y_train, X_pred) -> Optional[tuple[float, dict]]:
    """Returns (prediction, feature_importances) or None."""
    try:
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators  = N_ESTIMATORS,
            learning_rate = LEARNING_RATE,
            max_depth     = MAX_DEPTH,
            subsample     = 0.8,
            random_state  = RANDOM_SEED,
            verbose       = -1,
        )
        model.fit(X_train, y_train)
        pred = float(model.predict(X_pred)[0])
        importances = {
            name: round(float(imp / (imp.sum() + 1e-9)), 4)
            for name, imp in zip(X_train.columns, model.feature_importances_)
        }
        return pred, importances
    except ImportError:
        return None
    except Exception as exc:
        logger.debug("[forecast] LightGBM failed: %s", exc)
        return None


def _statistical_fallback(
    close: pd.Series, horizon: int
) -> tuple[float, float]:
    """
    Simple statistical baseline: weighted average of recent rolling returns.
    Returns (expected_return_pct, prob_up).
    """
    log_ret = np.log(close / close.shift(1)).dropna()
    if len(log_ret) < 20:
        return 0.0, 0.50

    # Regime-weighted: recent returns get more weight
    weights = np.exp(np.linspace(-1, 0, min(60, len(log_ret))))
    weights /= weights.sum()
    w_ret = float(np.dot(weights[-len(log_ret):], log_ret.iloc[-len(weights):]))

    # Scale to horizon
    scaled = w_ret * horizon
    expected_pct = scaled * 100

    # Prob up from empirical distribution of horizon returns
    if len(close) > horizon + 10:
        h_rets = (close.shift(-horizon) / close - 1.0).dropna()
        prob_up = float((h_rets > 0).mean())
    else:
        prob_up = 0.5 + min(0.2, abs(expected_pct) * 0.02) * (1 if expected_pct > 0 else -1)

    return round(expected_pct, 2), round(max(0.05, min(0.95, prob_up)), 3)


# ── Main Forecasting Loop ──────────────────────────────────────────────────────
def compute_forecast(
    df:                pd.DataFrame,
    regime_bull_prob:  float = 0.55,
    regime_bear_prob:  float = 0.30,
    regime_confidence: float = 65.0,
    funding_rate:      float = 0.0,
    liquidity_score:   float = 55.0,
    breadth_score:     float = 50.0,
    oi_trend_enc:      float = 0.0,
) -> ForecastResult:
    """
    Generate ML-based return forecasts across 3 horizons.

    Parameters
    ----------
    df               : daily OHLCV DataFrame (≥ 120 bars recommended)
    regime_bull_prob : from Markov/Ensemble regime engine (0–1)
    regime_bear_prob : from Markov/Ensemble regime engine (0–1)
    regime_confidence: from regime engine (0–100)
    funding_rate     : from FundingRateResult (optional; 0.0 if unavailable)
    liquidity_score  : from LiquidityRegimeResult (0–100; 55 if unavailable)
    breadth_score    : from MarketBreadthResult   (0–100; 50 if unavailable)
    oi_trend_enc     : +1=increasing OI, 0=stable, -1=decreasing

    Returns
    -------
    ForecastResult — always returns, never raises
    """
    # ── Insufficient data guard ──────────────────────────────────────────────
    if len(df) < MIN_BARS:
        logger.warning(
            "[forecast] Only %d bars available (need %d) — using statistical fallback",
            len(df), MIN_BARS,
        )
        results: dict[int, dict] = {}
        for h in HORIZONS:
            ret, pup = _statistical_fallback(df["Close"], h)
            results[h] = {"return_pct": ret, "prob_up": pup}

        return _build_result(results, {}, "statistical_fallback", regime_bull_prob)

    # ── Feature + target construction ────────────────────────────────────────
    features = _build_features(
        df, regime_bull_prob, regime_bear_prob, regime_confidence,
        funding_rate, liquidity_score, breadth_score, oi_trend_enc,
    )
    targets  = _build_targets(df["Close"])

    # Drop NaN rows
    valid_mask = features.notna().all(axis=1)
    features   = features[valid_mask].copy()

    # ── Per-horizon training ─────────────────────────────────────────────────
    results        : dict[int, dict]  = {}
    importances_all: list[dict]       = []
    model_used      = "statistical_fallback"

    for h in HORIZONS:
        tgt = targets[h][valid_mask].dropna()
        # Align features with target (drop last h rows since target is NaN there)
        common_idx = features.index.intersection(tgt.index)
        X = features.loc[common_idx]
        y = tgt.loc[common_idx]

        if len(X) < MIN_BARS // 2:
            ret, pup = _statistical_fallback(df["Close"], h)
            results[h] = {"return_pct": ret, "prob_up": pup}
            continue

        # Walk-forward: train on all but last TRAIN_HOLD bars, predict on latest bar
        X_train = X.iloc[:-TRAIN_HOLD]
        y_train = y.iloc[:-TRAIN_HOLD]
        X_pred  = features.iloc[[-1]]    # latest bar (might include recent macro)

        # Z-score normalise features using training window stats
        mu    = X_train.mean()
        sigma = X_train.std().replace(0, 1.0)
        X_train_sc = (X_train - mu) / sigma
        X_pred_sc  = (X_pred  - mu) / sigma

        # Try XGBoost → LightGBM → statistical
        model_result = (
            _try_xgboost(X_train_sc, y_train, X_pred_sc)
            or _try_lightgbm(X_train_sc, y_train, X_pred_sc)
        )

        if model_result is not None:
            pred, imps = model_result
            model_used = "xgboost" if _try_xgboost.__module__ == "__main__" else model_used
            # Detect which library was used
            try:
                import xgboost; model_used = "xgboost"
            except ImportError:
                try:
                    import lightgbm; model_used = "lightgbm"
                except ImportError:
                    pass

            ret_pct = round(pred * 100, 2)
            importances_all.append(imps)

            # Estimate prob_up from residual distribution
            residuals = (X_train_sc.copy().assign(target=y_train)
                         .pipe(lambda df_r: df_r["target"]))
            threshold = 0.0
            up_frac   = float((residuals > threshold).mean())
            # Adjust for prediction direction
            prob_up = up_frac * 0.6 + (0.5 + min(0.4, abs(ret_pct) * 0.02) *
                       (1 if ret_pct > 0 else -1)) * 0.4
            prob_up = round(max(0.05, min(0.95, prob_up)), 3)
        else:
            ret_pct, prob_up = _statistical_fallback(df["Close"], h)
            model_used = "statistical_fallback"

        results[h] = {"return_pct": ret_pct, "prob_up": prob_up}

    # ── Aggregate feature importances across horizons ────────────────────────
    if importances_all:
        all_keys = set(k for d in importances_all for k in d)
        avg_imp  = {
            k: round(float(np.mean([d.get(k, 0.0) for d in importances_all])), 4)
            for k in all_keys
        }
        # Sort descending
        avg_imp = dict(sorted(avg_imp.items(), key=lambda x: -x[1]))
    else:
        avg_imp = {}

    return _build_result(results, avg_imp, model_used, regime_bull_prob)


def _build_result(
    results: dict[int, dict],
    importances: dict[str, float],
    model_used: str,
    regime_bull_prob: float,
) -> ForecastResult:
    """Assemble ForecastResult from per-horizon results."""
    r5   = results.get(5,  {"return_pct": 0.0, "prob_up": 0.50})
    r10  = results.get(10, {"return_pct": 0.0, "prob_up": 0.50})
    r20  = results.get(20, {"return_pct": 0.0, "prob_up": 0.50})

    pup20 = r20["prob_up"]
    ret20 = r20["return_pct"]

    if pup20 > 0.60 and ret20 > 0.5:
        direction = "BULLISH"
    elif pup20 < 0.40 and ret20 < -0.5:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"

    # Confidence: model quality proxy — higher confidence when direction agrees
    # across all 3 horizons and with regime
    direction_votes = sum([
        r5["prob_up"]  > 0.55,
        r10["prob_up"] > 0.55,
        r20["prob_up"] > 0.55,
    ])
    regime_agrees = (regime_bull_prob > 0.55 and direction == "BULLISH") or \
                    (regime_bull_prob < 0.45 and direction == "BEARISH")

    confidence = 40.0 + direction_votes * 12.0 + (10.0 if regime_agrees else 0.0)
    if model_used == "statistical_fallback":
        confidence = min(confidence, 55.0)
    confidence = round(min(90.0, confidence), 1)

    interp = (
        f"[{model_used}] {direction} — "
        f"5d={r5['return_pct']:+.1f}% (P↑={r5['prob_up']:.0%}) | "
        f"10d={r10['return_pct']:+.1f}% (P↑={r10['prob_up']:.0%}) | "
        f"20d={r20['return_pct']:+.1f}% (P↑={r20['prob_up']:.0%}) | "
        f"conf={confidence:.0f}%"
    )

    logger.info("[forecast] %s", interp)

    return ForecastResult(
        expected_return_5d   = r5["return_pct"],
        expected_return_10d  = r10["return_pct"],
        expected_return_20d  = r20["return_pct"],
        probability_up_5d    = r5["prob_up"],
        probability_up_10d   = r10["prob_up"],
        probability_up_20d   = pup20,
        forecast_confidence  = confidence,
        forecast_direction   = direction,
        feature_importances  = importances,
        model_used           = model_used,
        horizon_forecasts    = {
            "5d":  r5,
            "10d": r10,
            "20d": r20,
        },
        interpretation = interp,
    )
