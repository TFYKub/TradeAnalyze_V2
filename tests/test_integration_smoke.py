"""
Integration Smoke Tests — Phases 12–21
=======================================
Tests the wiring between engines without live network calls.
All external I/O (yfinance, OKX API, LINE, Google Sheets) is mocked.

Scope:
  • LINE alert formatter — no network
  • Sheet writer row builder — no network
  • Dashboard text builder — no network
  • Regime ensemble v2 — pure maths, no network
  • Regime persistence math end-to-end
  • Conviction end-to-end with all signals
"""
from __future__ import annotations

import math
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch


# ── Shared Fixtures ────────────────────────────────────────────────────────────
def _ohlcv(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng    = np.random.default_rng(seed)
    closes = 50000 * np.cumprod(1 + rng.normal(0.001, 0.02, n))
    return pd.DataFrame({
        "Open":   closes * 0.999,
        "High":   closes * 1.010,
        "Low":    closes * 0.990,
        "Close":  closes,
        "Volume": rng.integers(1_000_000, 9_000_000, n).astype(float),
    })


def _mock_liquidity(regime="RISK_ON"):
    from engines.liquidity_regime import LiquidityRegimeResult
    return LiquidityRegimeResult(
        liquidity_regime=regime, confidence=72.0, score=68.0,
        risk_multiplier=1.0, vix_level=16.0, vix_regime="CALM",
        dxy_trend="STABLE", yield_trend="STABLE", tlt_vol_pct=10.5,
        component_scores={"vix": 72, "dxy": 65, "yield": 62, "tlt_vol": 70},
        interpretation="mock RISK_ON", data_source="fallback",
    )


def _mock_flow():
    from crypto.flow_engine import FlowEngineResult
    return FlowEngineResult(
        symbol="BTC", flow_score=62.0, flow_direction="BULLISH",
        flow_confidence=68.0, flow_regime="BULLISH_FLOW",
        funding_score=65.0, oi_score=70.0, ls_score=58.0, liq_score=60.0,
        funding_rate_pct=0.0005, funding_regime="NEUTRAL",
        oi_signal="CONFIRMATION", ls_ratio=1.15, cascade_risk="LOW",
        interpretation="mock bullish flow", component_detail={},
    )


def _mock_breadth():
    from engines.market_breadth import (
        MarketBreadthResult, CryptoBreadthResult, EquityBreadthResult,
    )
    cb = CryptoBreadthResult(
        btc_dominance_est=52.0, btc_dom_trend="STABLE",
        total3_change_pct=5.0, total3_trend="RISING",
        ssr_proxy=60.0, crypto_breadth_score=65.0,
    )
    eb = EquityBreadthResult(
        advance_decline_ratio=1.4, pct_above_200dma=62.0,
        new_high_count=5, new_low_count=1, equity_breadth_score=68.0,
    )
    return MarketBreadthResult(
        breadth_score=67.0, breadth_regime="BULL", breadth_confidence=72.0,
        crypto_breadth=cb, equity_breadth=eb,
        component_scores={}, interpretation="mock bull breadth", data_quality="FULL",
    )


def _mock_persistence():
    from engines.regime_persistence import RegimePersistenceResult
    return RegimePersistenceResult(
        regime="BULL", self_transition_prob=0.88,
        expected_duration_days=8.3, regime_half_life_days=5.4,
        remaining_duration_days=5.0,
        exit_prob_7d=40.0, exit_prob_14d=62.0,
        persistence_score=72.0, persistence_label="MATURING",
        most_likely_next="RANGE", next_regime_probs={"RANGE": 0.6, "BEAR": 0.4},
        interpretation="mock persistence",
    )


def _mock_forecast():
    from engines.forecast_engine import ForecastResult
    return ForecastResult(
        expected_return_5d=2.5,  expected_return_10d=4.0, expected_return_20d=7.0,
        probability_up_5d=0.65, probability_up_10d=0.63, probability_up_20d=0.68,
        forecast_confidence=72.0, forecast_direction="BULLISH",
        feature_importances={"rsi14": 0.18, "momentum_20": 0.15, "regime_bull_prob": 0.14},
        model_used="xgboost",
        horizon_forecasts={
            "5d":  {"return_pct": 2.5,  "prob_up": 0.65},
            "10d": {"return_pct": 4.0,  "prob_up": 0.63},
            "20d": {"return_pct": 7.0,  "prob_up": 0.68},
        },
        interpretation="mock bullish forecast",
    )


def _mock_conviction():
    from engines.conviction_engine import ConvictionResult
    return ConvictionResult(
        conviction_score=78.0, conviction_tier="NORMAL SIZE",
        kelly_multiplier=0.75, trade_allowed=True,
        component_scores={
            "regime": 80.0, "trend": 75.0, "structure": 70.0,
            "volatility": 85.0, "flow": 72.0, "breadth": 68.0,
            "liquidity": 75.0, "forecast": 70.0,
        },
        component_weights={
            "regime": 0.25, "trend": 0.15, "structure": 0.10,
            "volatility": 0.10, "flow": 0.15, "breadth": 0.10,
            "liquidity": 0.10, "forecast": 0.05,
        },
        weighted_scores={
            "regime": 20.0, "trend": 11.25, "structure": 7.0,
            "volatility": 8.5, "flow": 10.8, "breadth": 6.8,
            "liquidity": 7.5, "forecast": 3.5,
        },
        weakest_signal="breadth", strongest_signal="volatility",
        alignment_count=6, regime_persistence_ok=True,
        interpretation="mock NORMAL SIZE",
    )


def _mock_cross_asset():
    from engines.cross_asset_engine import CrossAssetResult
    return CrossAssetResult(
        cross_asset_regime="RISK_ON_ALIGNED", relative_strength_score=65.0,
        regime_confidence=72.0, pair_analyses={},
        rolling_correlations={"QQQ": 0.72, "SPY": 0.65, "DXY": -0.40, "GLD": 0.20},
        decoupling_detected=False, decoupling_direction="NONE",
        btc_beta_to_spy=1.35, btc_beta_to_qqq=1.50,
        dominant_driver="QQQ", interpretation="mock risk-on aligned",
    )


def _mock_portfolio():
    from portfolio.optimizer import PortfolioOptimizationResult
    return PortfolioOptimizationResult(
        method_used="risk_parity", recommended_weights={"BTC": 1.0},
        risk_contributions={"BTC": 100.0}, portfolio_volatility=72.0,
        portfolio_sharpe=0.85, portfolio_drawdown=28.0,
        diversification_ratio=1.0, correlation_matrix={"BTC": {"BTC": 1.0}},
        volatilities={"BTC": 72.0}, expected_returns={"BTC": 60.0},
        n_assets=1, interpretation="mock single asset",
    )


# ── LINE Alert v2 ──────────────────────────────────────────────────────────────
class TestLineAlertV2:
    def _mock_result(self):
        r = MagicMock()
        r.final_decision     = "LONG"
        r.trade_grade        = "A"
        r.ai_score           = 82.0
        r.approved           = True
        r.regime             = "BULL"
        r.regime_conf        = 80.0
        r.vol_regime         = "NORMAL_VOL"
        r.conviction_score   = 78.0
        r.conviction_tier    = "NORMAL SIZE"
        r.conviction_kelly_mult = 0.75
        r.persistence_label  = "MATURING"
        r.remaining_days     = 5.0
        r.exit_prob_7d       = 40.0
        r.entry              = 50000.0
        r.stop_loss          = 48000.0
        r.tp1                = 52000.0
        r.tp2                = 54000.0
        r.rr                 = 2.0
        r.kelly              = 0.06
        r.mc_profit_prob     = 0.62
        r.ev                 = 0.45
        r.runtime            = 4.2
        r.liquidity_regime   = "RISK_ON"
        r.flow_regime        = "BULLISH_FLOW"
        r.cross_asset_regime = "RISK_ON_ALIGNED"
        r.btc_rs_score       = 65.0
        r.btc_beta_spy       = 1.35
        return r

    def test_format_returns_string(self):
        from alerts.line_alert_v2 import format_institutional_alert
        msg = format_institutional_alert(
            "BTC", 50000.0, self._mock_result(),
            liquidity=_mock_liquidity(), flow=_mock_flow(),
            breadth=_mock_breadth(), persistence=_mock_persistence(),
            forecast=_mock_forecast(), conviction=_mock_conviction(),
            cross_asset=_mock_cross_asset(),
        )
        assert isinstance(msg, str)
        assert len(msg) > 100

    def test_message_under_char_limit(self):
        from alerts.line_alert_v2 import format_institutional_alert, LINE_CHAR_LIMIT
        msg = format_institutional_alert("BTC", 50000.0, self._mock_result())
        assert len(msg) <= LINE_CHAR_LIMIT

    def test_message_contains_symbol(self):
        from alerts.line_alert_v2 import format_institutional_alert
        msg = format_institutional_alert("BTC", 50000.0, self._mock_result())
        assert "BTC" in msg

    def test_message_contains_decision(self):
        from alerts.line_alert_v2 import format_institutional_alert
        msg = format_institutional_alert("BTC", 50000.0, self._mock_result())
        assert "LONG" in msg

    def test_none_inputs_dont_crash(self):
        from alerts.line_alert_v2 import format_institutional_alert
        msg = format_institutional_alert(
            "BTC", 50000.0, self._mock_result(),
            liquidity=None, flow=None, breadth=None,
            persistence=None, forecast=None, conviction=None,
        )
        assert isinstance(msg, str)

    def test_should_alert_approved_trade(self):
        from alerts.line_alert_v2 import _should_alert
        r = self._mock_result()
        assert _should_alert(r, _mock_conviction(), min_conviction=50.0) is True

    def test_should_alert_crisis_always(self):
        from alerts.line_alert_v2 import _should_alert
        r = MagicMock()
        r.final_decision   = "WAIT"
        r.approved         = False
        r.liquidity_regime = "CRISIS"
        r.flow_regime      = "NEUTRAL"
        assert _should_alert(r, min_conviction=90.0) is True

    def test_should_not_alert_low_conviction_wait(self):
        from alerts.line_alert_v2 import _should_alert
        r = MagicMock()
        r.final_decision   = "WAIT"
        r.approved         = False
        r.liquidity_regime = "RISK_ON"
        r.flow_regime      = "NEUTRAL"
        low_cv = MagicMock(); low_cv.conviction_score = 30.0
        assert _should_alert(r, low_cv, min_conviction=50.0) is False

    def test_squeeze_triggers_alert(self):
        from alerts.line_alert_v2 import _should_alert
        r = MagicMock()
        r.final_decision   = "WAIT"
        r.approved         = False
        r.liquidity_regime = "RISK_ON"
        r.flow_regime      = "SHORT_SQUEEZE"
        assert _should_alert(r) is True

    @patch("alerts.line_alert_v2.send_line_message", return_value=True)
    def test_send_does_not_call_line_if_filtered(self, mock_send):
        from alerts.line_alert_v2 import send_institutional_alert
        r = MagicMock()
        r.final_decision   = "WAIT"
        r.approved         = False
        r.liquidity_regime = "RISK_ON"
        r.flow_regime      = "NEUTRAL"
        low_cv = MagicMock(); low_cv.conviction_score = 20.0
        result = send_institutional_alert("BTC", 50000.0, r, conviction=low_cv)
        assert result is False
        mock_send.assert_not_called()


# ── Dashboard Text Builder ─────────────────────────────────────────────────────
class TestDashboards:
    def test_build_returns_string(self):
        from report.dashboards import build_institutional_dashboards
        output = build_institutional_dashboards(
            liquidity=_mock_liquidity(), flow=_mock_flow(),
            breadth=_mock_breadth(), persistence=_mock_persistence(),
            forecast=_mock_forecast(), conviction=_mock_conviction(),
            portfolio_opt=_mock_portfolio(), cross_asset=_mock_cross_asset(),
        )
        assert isinstance(output, str)
        assert len(output) > 200

    def test_all_sections_present(self):
        from report.dashboards import build_institutional_dashboards
        output = build_institutional_dashboards(
            liquidity=_mock_liquidity(), flow=_mock_flow(),
            breadth=_mock_breadth(), persistence=_mock_persistence(),
            forecast=_mock_forecast(), conviction=_mock_conviction(),
            portfolio_opt=_mock_portfolio(), cross_asset=_mock_cross_asset(),
        )
        assert "LIQUIDITY" in output
        assert "FLOW" in output
        assert "BREADTH" in output
        assert "PERSISTENCE" in output
        assert "FORECAST" in output
        assert "CONVICTION" in output
        assert "PORTFOLIO" in output
        assert "CROSS ASSET" in output

    def test_all_none_doesnt_crash(self):
        from report.dashboards import build_institutional_dashboards
        output = build_institutional_dashboards()
        assert isinstance(output, str)


# ── Conviction End-to-End ──────────────────────────────────────────────────────
class TestConvictionEndToEnd:
    """End-to-end conviction using real engine with mocked upstream inputs."""

    def test_full_bull_confluence_reaches_normal_size(self):
        from engines.conviction_engine import compute_conviction
        result = compute_conviction(
            regime="BULL", regime_confidence=82.0,
            ema_alignment=80.0, structure_trend="BULLISH",
            structure_clarity=75.0, vol_regime="NORMAL_VOL",
            trade_direction="LONG",
            flow_score=68.0, flow_direction="BULLISH", flow_regime="BULLISH_FLOW",
            breadth_score=70.0, breadth_regime="BULL",
            liquidity_score=72.0, liquidity_regime="RISK_ON", risk_multiplier=1.0,
            forecast_direction="BULLISH", probability_up_20d=0.66,
            forecast_confidence=70.0, persistence_score=72.0,
            persistence_label="MATURING",
        )
        assert result.conviction_tier in ("NORMAL SIZE", "FULL SIZE")
        assert result.trade_allowed is True
        assert result.kelly_multiplier >= 0.75

    def test_crisis_environment_blocks_trade(self):
        from engines.conviction_engine import compute_conviction
        result = compute_conviction(
            regime="BULL", regime_confidence=70.0,
            ema_alignment=60.0, structure_trend="BULLISH",
            structure_clarity=60.0, vol_regime="PANIC_VOL",
            trade_direction="LONG",
            liquidity_score=15.0, liquidity_regime="CRISIS", risk_multiplier=0.25,
        )
        assert result.conviction_tier in ("HALF SIZE", "NO TRADE")
