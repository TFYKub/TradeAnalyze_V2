"""
tests/conftest.py
==================
Shared pytest fixtures for the TradeAnalyze v2 test suite.

Fixtures defined here are auto-available to all test files
without explicit import — pytest injects them by name.

Fixture Catalogue:
  ohlcv_200          — 200-bar synthetic BTC OHLCV DataFrame
  ohlcv_short        — 30-bar DataFrame (below MIN_BARS threshold)
  btc_price_series   — 252-bar Close price Series
  mock_liquidity_on  — LiquidityRegimeResult (RISK_ON)
  mock_liquidity_crisis — LiquidityRegimeResult (CRISIS)
  mock_flow_bullish  — FlowEngineResult (BULLISH_FLOW)
  mock_flow_squeeze  — FlowEngineResult (SHORT_SQUEEZE)
  mock_breadth_bull  — MarketBreadthResult (BULL)
  mock_persistence   — RegimePersistenceResult (MATURING)
  mock_forecast_bull — ForecastResult (BULLISH)
  mock_conviction_normal — ConvictionResult (NORMAL SIZE)
  mock_cross_asset   — CrossAssetResult (RISK_ON_ALIGNED)
  mock_portfolio_single — PortfolioOptimizationResult (single asset)
  transition_matrix  — Realistic Markov transition matrix for BTC
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest


# ── OHLCV Factories ────────────────────────────────────────────────────────────
def _make_ohlcv(n: int, seed: int = 42, drift: float = 0.0005,
                vol: float = 0.02, start: float = 50_000.0) -> pd.DataFrame:
    rng    = np.random.default_rng(seed)
    rets   = rng.normal(drift, vol, n)
    closes = start * np.cumprod(1 + rets)
    highs  = closes * (1 + rng.uniform(0.001, 0.012, n))
    lows   = closes * (1 - rng.uniform(0.001, 0.012, n))
    opens  = closes * (1 + rng.normal(0, 0.003, n))
    vols   = rng.integers(500_000, 8_000_000, n).astype(float)
    return pd.DataFrame({
        "Open":  opens,
        "High":  highs,
        "Low":   lows,
        "Close": closes,
        "Volume": vols,
    })


@pytest.fixture(scope="session")
def ohlcv_200() -> pd.DataFrame:
    """200-bar BTC OHLCV — sufficient for all engine computations."""
    return _make_ohlcv(200)


@pytest.fixture(scope="session")
def ohlcv_short() -> pd.DataFrame:
    """30-bar OHLCV — below MIN_BARS, triggers fallback paths."""
    return _make_ohlcv(30)


@pytest.fixture(scope="session")
def btc_price_series(ohlcv_200) -> pd.Series:
    """252-bar Close series for portfolio and cross-asset tests."""
    return ohlcv_200["Close"].copy()


# ── Liquidity Fixtures ─────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def mock_liquidity_on():
    from engines.liquidity_regime import LiquidityRegimeResult
    return LiquidityRegimeResult(
        liquidity_regime="RISK_ON", confidence=75.0, score=72.0,
        risk_multiplier=1.0, vix_level=15.5, vix_regime="CALM",
        dxy_trend="STABLE", yield_trend="STABLE", tlt_vol_pct=9.8,
        component_scores={"vix": 78, "dxy": 68, "yield": 65, "tlt_vol": 72},
        interpretation="RISK_ON — mock fixture",
        data_source="fallback",
    )


@pytest.fixture(scope="session")
def mock_liquidity_crisis():
    from engines.liquidity_regime import LiquidityRegimeResult
    return LiquidityRegimeResult(
        liquidity_regime="CRISIS", confidence=88.0, score=18.0,
        risk_multiplier=0.25, vix_level=42.0, vix_regime="CRISIS",
        dxy_trend="STRENGTHENING", yield_trend="RISING", tlt_vol_pct=28.5,
        component_scores={"vix": 8, "dxy": 12, "yield": 18, "tlt_vol": 10},
        interpretation="🚨 CRISIS — mock fixture",
        data_source="fallback",
    )


# ── Flow Fixtures ──────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def mock_flow_bullish():
    from crypto.flow_engine import FlowEngineResult
    return FlowEngineResult(
        symbol="BTC", flow_score=68.0, flow_direction="BULLISH",
        flow_confidence=72.0, flow_regime="BULLISH_FLOW",
        funding_score=65.0, oi_score=75.0, ls_score=62.0, liq_score=60.0,
        funding_rate_pct=0.0003, funding_regime="NEUTRAL",
        oi_signal="CONFIRMATION", ls_ratio=1.12, cascade_risk="LOW",
        interpretation="🟢 BULLISH_FLOW — mock fixture",
        component_detail={},
    )


@pytest.fixture(scope="session")
def mock_flow_squeeze():
    from crypto.flow_engine import FlowEngineResult
    return FlowEngineResult(
        symbol="BTC", flow_score=82.0, flow_direction="BULLISH",
        flow_confidence=88.0, flow_regime="SHORT_SQUEEZE",
        funding_score=85.0, oi_score=80.0, ls_score=78.0, liq_score=70.0,
        funding_rate_pct=-0.012, funding_regime="CROWDED_SHORT",
        oi_signal="CONFIRMATION", ls_ratio=0.45, cascade_risk="LOW",
        interpretation="🚀 SHORT_SQUEEZE — mock fixture",
        component_detail={},
    )


# ── Breadth Fixtures ───────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def mock_breadth_bull():
    from engines.market_breadth import (
        MarketBreadthResult, CryptoBreadthResult, EquityBreadthResult,
    )
    cb = CryptoBreadthResult(
        btc_dominance_est=50.5, btc_dom_trend="STABLE",
        total3_change_pct=8.2, total3_trend="RISING",
        ssr_proxy=65.0, crypto_breadth_score=68.0,
    )
    eb = EquityBreadthResult(
        advance_decline_ratio=1.55, pct_above_200dma=68.0,
        new_high_count=6, new_low_count=1, equity_breadth_score=72.0,
    )
    return MarketBreadthResult(
        breadth_score=70.5, breadth_regime="BULL", breadth_confidence=74.0,
        crypto_breadth=cb, equity_breadth=eb,
        component_scores={
            "crypto_breadth": 68.0, "equity_breadth": 72.0,
            "btc_dominance": 52.0, "total3_momentum": 62.0,
            "advance_decline": 63.0, "pct_above_200dma": 68.0,
        },
        interpretation="📈 BULL breadth — mock fixture",
        data_quality="FULL",
    )


# ── Persistence Fixtures ───────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def mock_persistence():
    from engines.regime_persistence import RegimePersistenceResult
    return RegimePersistenceResult(
        regime="BULL", self_transition_prob=0.88,
        expected_duration_days=8.3, regime_half_life_days=5.4,
        remaining_duration_days=5.0,
        exit_prob_7d=40.0, exit_prob_14d=62.0,
        persistence_score=72.0, persistence_label="MATURING",
        most_likely_next="RANGE",
        next_regime_probs={"RANGE": 0.65, "BEAR": 0.35},
        interpretation="BULL [MATURING]: ~5d remaining — mock fixture",
    )


# ── Forecast Fixtures ──────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def mock_forecast_bull():
    from engines.forecast_engine import ForecastResult
    return ForecastResult(
        expected_return_5d=2.8, expected_return_10d=4.5, expected_return_20d=7.2,
        probability_up_5d=0.66, probability_up_10d=0.64, probability_up_20d=0.69,
        forecast_confidence=74.0, forecast_direction="BULLISH",
        feature_importances={
            "rsi14": 0.18, "momentum_20": 0.15,
            "regime_bull_prob": 0.14, "hv20": 0.11,
        },
        model_used="xgboost",
        horizon_forecasts={
            "5d":  {"return_pct": 2.8,  "prob_up": 0.66},
            "10d": {"return_pct": 4.5,  "prob_up": 0.64},
            "20d": {"return_pct": 7.2,  "prob_up": 0.69},
        },
        interpretation="🟢 BULLISH [xgboost] — mock fixture",
    )


# ── Conviction Fixtures ────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def mock_conviction_normal():
    from engines.conviction_engine import ConvictionResult
    return ConvictionResult(
        conviction_score=78.0, conviction_tier="NORMAL SIZE",
        kelly_multiplier=0.75, trade_allowed=True,
        component_scores={
            "regime": 80.0, "trend": 75.0, "structure": 72.0,
            "volatility": 88.0, "flow": 70.0, "breadth": 68.0,
            "liquidity": 74.0, "forecast": 70.0,
        },
        component_weights={
            "regime": 0.25, "trend": 0.15, "structure": 0.10,
            "volatility": 0.10, "flow": 0.15, "breadth": 0.10,
            "liquidity": 0.10, "forecast": 0.05,
        },
        weighted_scores={
            "regime": 20.0, "trend": 11.25, "structure": 7.2,
            "volatility": 8.8, "flow": 10.5, "breadth": 6.8,
            "liquidity": 7.4, "forecast": 3.5,
        },
        weakest_signal="breadth", strongest_signal="volatility",
        alignment_count=7, regime_persistence_ok=True,
        interpretation="✅ NORMAL SIZE — conviction=78/100 — mock fixture",
    )


# ── Cross-Asset Fixtures ───────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def mock_cross_asset():
    from engines.cross_asset_engine import CrossAssetResult
    return CrossAssetResult(
        cross_asset_regime="RISK_ON_ALIGNED", relative_strength_score=66.0,
        regime_confidence=74.0, pair_analyses={},
        rolling_correlations={
            "QQQ": 0.74, "SPY": 0.66, "DXY": -0.42,
            "GLD": 0.22, "TLT": 0.18,
        },
        decoupling_detected=False, decoupling_direction="NONE",
        btc_beta_to_spy=1.38, btc_beta_to_qqq=1.52,
        dominant_driver="QQQ",
        interpretation="🟢 RISK_ON_ALIGNED — mock fixture",
    )


# ── Portfolio Fixtures ─────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def mock_portfolio_single():
    from portfolio.optimizer import PortfolioOptimizationResult
    return PortfolioOptimizationResult(
        method_used="single_asset",
        recommended_weights={"BTC": 1.0},
        risk_contributions={"BTC": 100.0},
        portfolio_volatility=68.5,
        portfolio_sharpe=0.92,
        portfolio_drawdown=26.8,
        diversification_ratio=1.0,
        correlation_matrix={"BTC": {"BTC": 1.0}},
        volatilities={"BTC": 68.5},
        expected_returns={"BTC": 62.0},
        n_assets=1,
        interpretation="single_asset — BTC=100% — mock fixture",
    )


# ── Transition Matrix ──────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def transition_matrix() -> dict:
    """
    Realistic BTC regime transition matrix (calibrated from historical data).
    Rows: from-state.  Columns: to-state.  Each row sums to 1.0.
    """
    return {
        "STRONG_BULL": {
            "STRONG_BULL": 0.72, "BULL": 0.20,
            "RANGE": 0.06, "BEAR": 0.01, "STRONG_BEAR": 0.01,
        },
        "BULL": {
            "STRONG_BULL": 0.15, "BULL": 0.68,
            "RANGE": 0.12, "BEAR": 0.04, "STRONG_BEAR": 0.01,
        },
        "RANGE": {
            "STRONG_BULL": 0.08, "BULL": 0.18,
            "RANGE": 0.52, "BEAR": 0.17, "STRONG_BEAR": 0.05,
        },
        "BEAR": {
            "STRONG_BULL": 0.01, "BULL": 0.05,
            "RANGE": 0.18, "BEAR": 0.65, "STRONG_BEAR": 0.11,
        },
        "STRONG_BEAR": {
            "STRONG_BULL": 0.01, "BULL": 0.02,
            "RANGE": 0.08, "BEAR": 0.22, "STRONG_BEAR": 0.67,
        },
    }
