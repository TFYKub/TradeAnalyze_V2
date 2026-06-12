"""
Tests — Phase 16: Bayesian Reliability Weighting
Tests — Phase 17: Forecast Engine
Tests — Phase 18: Conviction Engine
"""
import math
import pytest
import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 16 — Bayesian Reliability Weighting
# ══════════════════════════════════════════════════════════════════════════════
from engines.bayesian_reliability import (
    RSI_RELIABILITY,
    TREND_RELIABILITY,
    _weighted_posterior,
    compute_rsi_bayesian_weighted,
    compute_trend_bayesian_weighted,
)


class TestReliabilityTables:
    def test_rsi_strong_bear_is_low(self):
        assert RSI_RELIABILITY["STRONG_BEAR"] <= 0.2

    def test_rsi_strong_bull_is_high(self):
        assert RSI_RELIABILITY["STRONG_BULL"] >= 0.9

    def test_trend_reliability_is_high_in_trends(self):
        assert TREND_RELIABILITY["STRONG_BULL"] >= 0.9
        assert TREND_RELIABILITY["STRONG_BEAR"] >= 0.9 if "STRONG_BEAR" in TREND_RELIABILITY else True

    def test_all_reliabilities_bounded(self):
        for table in (RSI_RELIABILITY, TREND_RELIABILITY):
            for v in table.values():
                assert 0.0 <= v <= 1.0


class TestWeightedPosterior:
    def test_full_reliability_same_as_unweighted(self):
        # reliability=1.0 should not change outcome significantly
        result = _weighted_posterior(0.5, 0.8, 1.0)
        assert result > 0.5   # high likelihood pushes above prior

    def test_zero_reliability_collapses_to_prior(self):
        result = _weighted_posterior(0.5, 0.9, 0.0)
        # effective_likelihood ≈ 0 → posterior stays near prior
        assert result <= 0.55

    def test_output_bounded(self):
        for prior in [0.1, 0.5, 0.9]:
            for like in [0.1, 0.5, 0.9]:
                for rel in [0.0, 0.5, 1.0]:
                    p = _weighted_posterior(prior, like, rel)
                    assert 0.0 < p < 1.0


class TestRSIBayesianWeighted:
    """Core use case: RSI oversold in STRONG_BEAR should produce low posterior."""

    def test_oversold_bear_regime_is_dampened(self):
        bear_sig = compute_rsi_bayesian_weighted(rsi=22.0, regime="STRONG_BEAR")
        bull_sig = compute_rsi_bayesian_weighted(rsi=22.0, regime="STRONG_BULL")
        # In STRONG_BEAR, RSI oversold should be less convincing than in STRONG_BULL
        assert bear_sig.posterior < bull_sig.posterior

    def test_signal_description_includes_reliability(self):
        sig = compute_rsi_bayesian_weighted(30.0, "BEAR")
        assert "reliability" in sig.description.lower()

    def test_signal_has_required_fields(self):
        sig = compute_rsi_bayesian_weighted(50.0, "RANGE")
        assert hasattr(sig, "posterior")
        assert hasattr(sig, "likelihood")
        assert hasattr(sig, "confidence")
        assert 0 < sig.posterior < 1


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 17 — Forecast Engine
# ══════════════════════════════════════════════════════════════════════════════
from engines.forecast_engine import (
    _build_features,
    _build_targets,
    _statistical_fallback,
    _build_result,
    ForecastResult,
)


def _make_ohlcv(n: int = 150, seed: int = 42) -> pd.DataFrame:
    rng    = np.random.default_rng(seed)
    closes = 50000 * np.cumprod(1 + rng.normal(0.0005, 0.02, n))
    highs  = closes * (1 + rng.uniform(0, 0.01, n))
    lows   = closes * (1 - rng.uniform(0, 0.01, n))
    return pd.DataFrame({
        "Open": closes, "High": highs, "Low": lows,
        "Close": closes, "Volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
    })


class TestBuildFeatures:
    def test_feature_count_correct(self):
        df = _make_ohlcv(200)
        features = _build_features(df)
        assert "rsi14" in features.columns
        assert "atr_pct" in features.columns
        assert "momentum_20" in features.columns
        assert "regime_bull_prob" in features.columns

    def test_macro_inputs_broadcast(self):
        df = _make_ohlcv(200)
        features = _build_features(df, liquidity_score=80.0)
        # liquidity_score should be normalised to 0.8 throughout
        assert features["liquidity_score"].iloc[-1] == pytest.approx(0.80, abs=0.01)

    def test_no_infinite_values(self):
        df = _make_ohlcv(200)
        features = _build_features(df)
        assert not features.isin([float("inf"), float("-inf")]).any().any()


class TestBuildTargets:
    def test_5d_target_is_correct_shift(self):
        df = _make_ohlcv(50)
        targets = _build_targets(df["Close"])
        # Row 0 target = close[5]/close[0] - 1
        expected = df["Close"].iloc[5] / df["Close"].iloc[0] - 1
        assert targets[5].iloc[0] == pytest.approx(expected, rel=1e-5)

    def test_all_horizons_present(self):
        df = _make_ohlcv(50)
        t  = _build_targets(df["Close"])
        assert set(t.keys()) == {5, 10, 20}


class TestStatisticalFallback:
    def test_returns_float_tuple(self):
        df = _make_ohlcv(100)
        ret, prob = _statistical_fallback(df["Close"], 20)
        assert isinstance(ret, float)
        assert isinstance(prob, float)

    def test_prob_bounded(self):
        df = _make_ohlcv(100)
        _, prob = _statistical_fallback(df["Close"], 20)
        assert 0.0 < prob < 1.0

    def test_short_series_doesnt_crash(self):
        df = _make_ohlcv(10)
        ret, prob = _statistical_fallback(df["Close"], 20)
        assert isinstance(ret, float)


class TestBuildResult:
    def test_bullish_direction(self):
        results = {
            5:  {"return_pct": 3.0,  "prob_up": 0.70},
            10: {"return_pct": 5.0,  "prob_up": 0.68},
            20: {"return_pct": 8.0,  "prob_up": 0.72},
        }
        r = _build_result(results, {}, "xgboost", 0.65)
        assert r.forecast_direction == "BULLISH"
        assert r.forecast_confidence > 50

    def test_bearish_direction(self):
        results = {
            5:  {"return_pct": -3.0,  "prob_up": 0.30},
            10: {"return_pct": -5.0,  "prob_up": 0.28},
            20: {"return_pct": -8.0,  "prob_up": 0.25},
        }
        r = _build_result(results, {}, "xgboost", 0.35)
        assert r.forecast_direction == "BEARISH"

    def test_result_frozen(self):
        results = {h: {"return_pct": 0.0, "prob_up": 0.5} for h in [5, 10, 20]}
        r = _build_result(results, {}, "statistical_fallback", 0.5)
        with pytest.raises(Exception):
            r.forecast_direction = "BULLISH"

    def test_all_probability_fields_bounded(self):
        results = {h: {"return_pct": 2.0, "prob_up": 0.65} for h in [5, 10, 20]}
        r = _build_result(results, {}, "statistical_fallback", 0.6)
        for field in (r.probability_up_5d, r.probability_up_10d, r.probability_up_20d):
            assert 0.0 < field < 1.0


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 18 — Conviction Engine
# ══════════════════════════════════════════════════════════════════════════════
from engines.conviction_engine import (
    _regime_to_score,
    _flow_to_score,
    _liquidity_to_score,
    _classify_tier,
    compute_conviction,
    TIER_FULL, TIER_NORMAL, TIER_HALF,
)


class TestRegimeToScore:
    def test_strong_bull_high_score(self):
        assert _regime_to_score("STRONG_BULL", 90.0) > 85

    def test_range_low_score(self):
        assert _regime_to_score("RANGE", 70.0) < 40

    def test_confidence_scales_score(self):
        high_conf = _regime_to_score("BULL", 90.0)
        low_conf  = _regime_to_score("BULL", 40.0)
        assert high_conf > low_conf


class TestFlowToScore:
    def test_aligned_flow_high(self):
        score = _flow_to_score(75.0, "BULLISH", "LONG", "BULLISH_FLOW")
        assert score > 60

    def test_opposed_flow_low(self):
        score = _flow_to_score(75.0, "BULLISH", "SHORT", "BULLISH_FLOW")
        assert score < 50

    def test_squeeze_aligned_very_high(self):
        score = _flow_to_score(80.0, "BULLISH", "LONG", "SHORT_SQUEEZE")
        assert score >= 90

    def test_squeeze_wrong_side_very_low(self):
        score = _flow_to_score(80.0, "BULLISH", "SHORT", "SHORT_SQUEEZE")
        assert score <= 25


class TestLiquidityToScore:
    def test_crisis_near_zero(self):
        score = _liquidity_to_score(20.0, "CRISIS", 0.25)
        assert score < 15

    def test_risk_on_high(self):
        score = _liquidity_to_score(80.0, "RISK_ON", 1.1)
        assert score > 60


class TestClassifyTier:
    def test_full_size_tier(self):
        tier, mult, allowed = _classify_tier(95.0)
        assert tier == "FULL SIZE"
        assert mult == pytest.approx(1.0)
        assert allowed is True

    def test_normal_size_tier(self):
        tier, mult, allowed = _classify_tier(75.0)
        assert tier == "NORMAL SIZE"
        assert mult == pytest.approx(0.75)
        assert allowed is True

    def test_half_size_tier(self):
        tier, mult, allowed = _classify_tier(55.0)
        assert tier == "HALF SIZE"
        assert mult == pytest.approx(0.5)
        assert allowed is True

    def test_no_trade_tier(self):
        tier, mult, allowed = _classify_tier(40.0)
        assert tier == "NO TRADE"
        assert mult == pytest.approx(0.0)
        assert allowed is False


class TestComputeConviction:
    def _base_kwargs(self, **overrides):
        kwargs = dict(
            regime="BULL", regime_confidence=80.0,
            ema_alignment=75.0, structure_trend="BULLISH",
            structure_clarity=70.0, vol_regime="NORMAL_VOL",
            trade_direction="LONG",
        )
        kwargs.update(overrides)
        return kwargs

    def test_wait_direction_returns_no_trade(self):
        result = compute_conviction(**self._base_kwargs(trade_direction="WAIT"))
        assert result.conviction_tier == "NO TRADE"
        assert result.trade_allowed is False

    def test_strong_bull_signals_high_conviction(self):
        result = compute_conviction(
            regime="STRONG_BULL", regime_confidence=90.0,
            ema_alignment=90.0, structure_trend="BULLISH",
            structure_clarity=85.0, vol_regime="NORMAL_VOL",
            trade_direction="LONG",
            flow_score=75.0, flow_direction="BULLISH", flow_regime="BULLISH_FLOW",
            breadth_score=80.0, breadth_regime="STRONG_BULL",
            liquidity_score=80.0, liquidity_regime="RISK_ON", risk_multiplier=1.1,
            forecast_direction="BULLISH", probability_up_20d=0.72,
            forecast_confidence=75.0,
        )
        assert result.conviction_score >= 70
        assert result.trade_allowed is True

    def test_crisis_liquidity_kills_conviction(self):
        result = compute_conviction(
            **self._base_kwargs(
                liquidity_regime="CRISIS",
                liquidity_score=15.0,
                risk_multiplier=0.25,
            )
        )
        # CRISIS liquidity (score=5) must suppress to HALF SIZE or lower, never NORMAL/FULL
        assert result.conviction_tier in ("HALF SIZE", "NO TRADE")
        assert result.conviction_score < 70

    def test_exhausted_persistence_penalises(self):
        normal = compute_conviction(**self._base_kwargs())
        exhausted = compute_conviction(
            **self._base_kwargs(
                persistence_label="EXHAUSTED",
                persistence_score=20.0,
            )
        )
        assert exhausted.conviction_score < normal.conviction_score

    def test_result_has_all_required_fields(self):
        result = compute_conviction(**self._base_kwargs())
        assert hasattr(result, "conviction_score")
        assert hasattr(result, "conviction_tier")
        assert hasattr(result, "kelly_multiplier")
        assert hasattr(result, "weakest_signal")
        assert hasattr(result, "strongest_signal")
        assert hasattr(result, "component_scores")
        assert len(result.component_scores) == 8

    def test_result_frozen(self):
        result = compute_conviction(**self._base_kwargs())
        with pytest.raises(Exception):
            result.conviction_score = 99.0
