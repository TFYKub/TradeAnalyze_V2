"""
Tests — Phase 12: Liquidity Regime Engine
==========================================
covers: score_vix, score_dxy, score_yield, classify_regime, compute_liquidity_regime fallback
"""
import math
import pytest
import pandas as pd
import numpy as np

from engines.liquidity_regime import (
    _score_vix,
    _score_dxy,
    _score_yield,
    _score_tlt_vol,
    _classify_regime,
    LiquidityRegimeResult,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────
def _make_close(values: list[float], col: str = "Close") -> pd.DataFrame:
    return pd.DataFrame({col: values})


def _make_ohlcv(closes: list[float]) -> pd.DataFrame:
    df = pd.DataFrame({
        "Open":  closes, "High": [c * 1.01 for c in closes],
        "Low":   [c * 0.99 for c in closes], "Close": closes,
        "Volume": [1_000_000] * len(closes),
    })
    return df


# ── VIX Scorer ─────────────────────────────────────────────────────────────────
class TestScoreVIX:
    def test_crisis_vix_gives_low_score(self):
        df = _make_close([40.0])
        score, level, regime = _score_vix(df)
        assert score < 15
        assert regime == "CRISIS"
        assert level == pytest.approx(40.0, abs=0.1)

    def test_calm_vix_gives_high_score(self):
        df = _make_close([12.0])
        score, level, regime = _score_vix(df)
        assert score > 65
        assert regime == "CALM"

    def test_elevated_vix_gives_mid_score(self):
        df = _make_close([22.0])
        score, level, regime = _score_vix(df)
        assert 40 <= score <= 68
        assert regime == "ELEVATED"

    def test_stress_vix(self):
        df = _make_close([28.0])
        score, level, regime = _score_vix(df)
        assert regime == "STRESS"
        assert score < 50

    def test_none_input_returns_neutral(self):
        score, level, regime = _score_vix(None)
        assert score == pytest.approx(50.0)
        assert regime == "UNKNOWN"

    def test_score_bounded_0_100(self):
        for vix_val in [5, 15, 25, 40, 80]:
            df = _make_close([float(vix_val)])
            score, _, _ = _score_vix(df)
            assert 0 <= score <= 100


# ── DXY Scorer ─────────────────────────────────────────────────────────────────
class TestScoreDXY:
    def _make_dxy(self, start: float, end: float, n: int = 25) -> pd.DataFrame:
        closes = np.linspace(start, end, n).tolist()
        return _make_close(closes)

    def test_surging_dxy_gives_low_score(self):
        df = self._make_dxy(100.0, 103.0, 25)
        score, trend = _score_dxy(df)
        assert score <= 40
        assert trend == "STRENGTHENING"

    def test_weakening_dxy_gives_high_score(self):
        df = self._make_dxy(105.0, 102.0, 25)
        score, trend = _score_dxy(df)
        assert score >= 55
        assert trend == "WEAKENING"

    def test_stable_dxy_gives_neutral(self):
        closes = [100.0 + 0.05 * math.sin(i) for i in range(25)]
        df = _make_close(closes)
        score, trend = _score_dxy(df)
        assert 35 <= score <= 65

    def test_none_returns_neutral(self):
        score, trend = _score_dxy(None)
        assert score == pytest.approx(50.0)
        assert trend == "UNKNOWN"


# ── Yield Scorer ───────────────────────────────────────────────────────────────
class TestScoreYield:
    def test_surging_yields_give_low_score(self):
        # Need old values at least LOOKBACK_DAYS=20 bars back:
        # 25 bars of low yield, then 5 bars of high → iloc[-20] still low
        df = _make_close([3.5] * 25 + [4.1] * 5)
        score, trend = _score_yield(df)
        assert score <= 40
        assert trend == "RISING"

    def test_falling_yields_give_higher_score(self):
        # 25 bars of high yield, then 5 bars of low → iloc[-20] still high
        df = _make_close([4.5] * 25 + [3.8] * 5)
        score, trend = _score_yield(df)
        assert score >= 55
        assert trend == "FALLING"

    def test_none_returns_neutral(self):
        score, trend = _score_yield(None)
        assert score == pytest.approx(50.0)


# ── Regime Classifier ──────────────────────────────────────────────────────────
class TestClassifyRegime:
    def test_high_score_risk_on(self):
        regime, conf, mult = _classify_regime(85.0, "CALM")
        assert regime == "RISK_ON"
        assert conf > 60
        assert mult >= 0.9

    def test_crisis_vix_overrides(self):
        regime, conf, mult = _classify_regime(70.0, "CRISIS")
        assert regime == "CRISIS"
        assert mult == pytest.approx(0.25)

    def test_low_score_crisis(self):
        regime, conf, mult = _classify_regime(20.0, "STRESS")
        assert regime == "CRISIS"
        assert mult == pytest.approx(0.25)

    def test_risk_off_range(self):
        regime, conf, mult = _classify_regime(38.0, "STRESS")
        assert regime == "RISK_OFF"
        assert mult < 0.9

    def test_recovery_from_crisis(self):
        regime, conf, mult = _classify_regime(60.0, "CALM", prev_regime="CRISIS")
        assert regime == "RECOVERY"
        assert mult == pytest.approx(0.75)

    def test_multiplier_bounded(self):
        for score in [0, 25, 50, 75, 100]:
            regime, conf, mult = _classify_regime(float(score), "CALM")
            assert 0.0 <= mult <= 1.3
            assert 0.0 <= conf <= 100.0


# ── Fallback Result ────────────────────────────────────────────────────────────
class TestLiquidityFallbackResult:
    """Verifies the dataclass contract is stable even with default/fallback values."""

    def test_result_fields_present(self):
        result = LiquidityRegimeResult(
            liquidity_regime="RISK_ON", confidence=60.0, score=62.0,
            risk_multiplier=1.0, vix_level=18.0, vix_regime="CALM",
            dxy_trend="STABLE", yield_trend="STABLE", tlt_vol_pct=12.0,
            component_scores={"vix": 70, "dxy": 60, "yield": 55, "tlt_vol": 65},
            interpretation="Test result",
            data_source="fallback",
        )
        assert result.liquidity_regime == "RISK_ON"
        assert result.risk_multiplier == pytest.approx(1.0)
        assert isinstance(result.component_scores, dict)

    def test_result_is_frozen(self):
        result = LiquidityRegimeResult(
            liquidity_regime="RISK_ON", confidence=60.0, score=62.0,
            risk_multiplier=1.0, vix_level=18.0, vix_regime="CALM",
            dxy_trend="STABLE", yield_trend="STABLE", tlt_vol_pct=12.0,
            component_scores={}, interpretation="Test", data_source="fallback",
        )
        with pytest.raises(Exception):  # frozen dataclass raises on assignment
            result.liquidity_regime = "CRISIS"
