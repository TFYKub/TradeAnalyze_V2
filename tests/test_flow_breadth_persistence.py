"""
Tests — Phase 13: Flow Engine
Tests — Phase 14: Market Breadth Engine
Tests — Phase 15: Regime Persistence Engine
"""
import pytest

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 13 — Flow Engine
# ══════════════════════════════════════════════════════════════════════════════
from crypto.flow_engine import (
    _score_funding,
    _score_oi,
    _score_long_short,
    _score_liquidation,
    _classify_flow,
    FlowEngineResult,
    LongShortResult,
)


class TestScoreFunding:
    def test_crowded_long_is_bearish(self):
        score = _score_funding("CROWDED_LONG", 0.05)
        assert score < 30

    def test_crowded_short_is_bullish(self):
        score = _score_funding("CROWDED_SHORT", -0.05)
        assert score > 70

    def test_neutral_is_midpoint(self):
        score = _score_funding("NEUTRAL", 0.0)
        assert 40 <= score <= 60

    def test_score_bounded(self):
        for regime in ("CROWDED_LONG", "HIGH_LONG", "NEUTRAL", "HIGH_SHORT", "CROWDED_SHORT"):
            s = _score_funding(regime, 0.01)
            assert 0 <= s <= 100


class TestScoreOI:
    def test_confirmation_bullish(self):
        assert _score_oi("CONFIRMATION", "INCREASING") > 65

    def test_continuation_bearish(self):
        assert _score_oi("CONTINUATION", "INCREASING") < 35

    def test_capitulation_near_neutral(self):
        s = _score_oi("CAPITULATION", "DECREASING")
        assert 40 <= s <= 65

    def test_unknown_returns_neutral(self):
        assert _score_oi("UNKNOWN", "STABLE") == pytest.approx(50.0)


class TestScoreLongShort:
    def _make_ls(self, ratio: float) -> LongShortResult:
        long_r = ratio / (1 + ratio)
        return LongShortResult(
            symbol="BTC", long_ratio=round(long_r, 4),
            short_ratio=round(1 - long_r, 4), ls_ratio=ratio,
            source="test", interpretation="test",
        )

    def test_very_crowded_longs_bearish(self):
        assert _score_long_short(self._make_ls(2.5)) < 25

    def test_very_crowded_shorts_bullish(self):
        assert _score_long_short(self._make_ls(0.4)) > 75

    def test_balanced_neutral(self):
        s = _score_long_short(self._make_ls(1.0))
        assert 45 <= s <= 55

    def test_none_returns_50(self):
        assert _score_long_short(None) == pytest.approx(50.0)


class TestClassifyFlow:
    def test_squeeze_detection_short(self):
        regime, direction, conf = _classify_flow(
            composite=75.0,
            funding_regime="CROWDED_SHORT",
            oi_signal="CONFIRMATION",
            cascade_risk="LOW",
            ls=None,
        )
        assert regime == "SHORT_SQUEEZE"
        assert direction == "BULLISH"
        assert conf >= 60

    def test_squeeze_detection_long(self):
        regime, direction, conf = _classify_flow(
            composite=25.0,
            funding_regime="CROWDED_LONG",
            oi_signal="CONFIRMATION",
            cascade_risk="LOW",
            ls=None,
        )
        assert regime == "LONG_SQUEEZE"
        assert direction == "BEARISH"

    def test_bullish_flow(self):
        regime, direction, _ = _classify_flow(75.0, "HIGH_SHORT", "CONFIRMATION", "LOW", None)
        assert regime == "BULLISH_FLOW"
        assert direction == "BULLISH"

    def test_bearish_flow(self):
        regime, direction, _ = _classify_flow(25.0, "HIGH_LONG", "CONTINUATION", "HIGH", None)
        assert regime == "BEARISH_FLOW"
        assert direction == "BEARISH"

    def test_neutral_flow(self):
        regime, direction, _ = _classify_flow(50.0, "NEUTRAL", "UNKNOWN", "LOW", None)
        assert regime == "NEUTRAL"
        assert direction == "NEUTRAL"


class TestFlowEngineResult:
    def test_dataclass_frozen(self):
        r = FlowEngineResult(
            symbol="BTC", flow_score=60.0, flow_direction="BULLISH",
            flow_confidence=70.0, flow_regime="BULLISH_FLOW",
            funding_score=65.0, oi_score=70.0, ls_score=55.0, liq_score=60.0,
            funding_rate_pct=0.001, funding_regime="NEUTRAL",
            oi_signal="CONFIRMATION", ls_ratio=1.1, cascade_risk="LOW",
            interpretation="test", component_detail={},
        )
        with pytest.raises(Exception):
            r.flow_score = 99.0


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 14 — Market Breadth Engine
# ══════════════════════════════════════════════════════════════════════════════
from engines.market_breadth import (
    _classify_breadth,
    MarketBreadthResult,
    CryptoBreadthResult,
    EquityBreadthResult,
)


class TestClassifyBreadth:
    @pytest.mark.parametrize("score,expected_regime", [
        (90.0, "STRONG_BULL"),
        (70.0, "BULL"),
        (50.0, "NEUTRAL"),
        (35.0, "BEAR"),
        (15.0, "STRONG_BEAR"),
    ])
    def test_regime_thresholds(self, score, expected_regime):
        regime, conf = _classify_breadth(score)
        assert regime == expected_regime
        assert 0 < conf <= 100

    def test_confidence_increases_with_extremity(self):
        _, conf_neutral = _classify_breadth(50.0)
        _, conf_bull    = _classify_breadth(90.0)
        assert conf_bull > conf_neutral

    def test_all_scores_bounded(self):
        for score in range(0, 101, 5):
            regime, conf = _classify_breadth(float(score))
            assert regime in ("STRONG_BULL", "BULL", "NEUTRAL", "BEAR", "STRONG_BEAR")
            assert 0 < conf <= 100


class TestMarketBreadthResult:
    def _make_result(self, score: float, regime: str) -> MarketBreadthResult:
        cb = CryptoBreadthResult(
            btc_dominance_est=52.0, btc_dom_trend="STABLE",
            total3_change_pct=0.0, total3_trend="STABLE",
            ssr_proxy=50.0, crypto_breadth_score=score,
        )
        eb = EquityBreadthResult(
            advance_decline_ratio=1.0, pct_above_200dma=50.0,
            new_high_count=3, new_low_count=3, equity_breadth_score=score,
        )
        return MarketBreadthResult(
            breadth_score=score, breadth_regime=regime,
            breadth_confidence=70.0, crypto_breadth=cb, equity_breadth=eb,
            component_scores={}, interpretation="test", data_quality="FULL",
        )

    def test_strong_bull_result(self):
        r = self._make_result(85.0, "STRONG_BULL")
        assert r.breadth_regime == "STRONG_BULL"
        assert r.breadth_score > 80

    def test_result_frozen(self):
        r = self._make_result(50.0, "NEUTRAL")
        with pytest.raises(Exception):
            r.breadth_score = 99.0


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 15 — Regime Persistence Engine
# ══════════════════════════════════════════════════════════════════════════════
from engines.regime_persistence import (
    _expected_duration,
    _half_life,
    _exit_prob_within,
    _persistence_score,
    _persistence_label,
    compute_regime_persistence,
)


class TestExpectedDuration:
    def test_high_self_prob_long_duration(self):
        assert _expected_duration(0.95) == pytest.approx(20.0, abs=0.1)

    def test_low_self_prob_short_duration(self):
        assert _expected_duration(0.5) == pytest.approx(2.0, abs=0.1)

    def test_prob_zero_returns_one(self):
        assert _expected_duration(0.0) == pytest.approx(1.0)

    def test_prob_one_returns_max(self):
        assert _expected_duration(1.0) == pytest.approx(365.0)

    def test_always_positive(self):
        for p in [0.01, 0.5, 0.8, 0.9, 0.95, 0.99]:
            assert _expected_duration(p) > 0


class TestHalfLife:
    def test_half_life_less_than_expected(self):
        for p in [0.7, 0.8, 0.9, 0.95]:
            assert _half_life(p) < _expected_duration(p)

    def test_half_life_positive(self):
        for p in [0.5, 0.8, 0.9]:
            assert _half_life(p) > 0

    def test_high_persistence_long_half_life(self):
        assert _half_life(0.95) > _half_life(0.7)


class TestExitProbWithin:
    def test_certain_exit_immediately(self):
        assert _exit_prob_within(0.0, 1) == pytest.approx(1.0)

    def test_never_exit_if_p1(self):
        assert _exit_prob_within(1.0, 7) == pytest.approx(0.0)

    def test_prob_increases_with_days(self):
        p = 0.85
        assert _exit_prob_within(p, 14) > _exit_prob_within(p, 7)

    def test_prob_bounded_0_1(self):
        for d in [1, 3, 7, 14, 30]:
            assert 0.0 <= _exit_prob_within(0.85, d) <= 1.0


class TestPersistenceLabel:
    @pytest.mark.parametrize("score,label", [
        (90.0, "ESTABLISHED"),
        (65.0, "MATURING"),
        (45.0, "FRESH"),
        (20.0, "EXHAUSTED"),
    ])
    def test_labels(self, score, label):
        assert _persistence_label(score) == label


class TestComputeRegimePersistence:
    def _make_matrix(self, p_self: float = 0.85) -> dict:
        p_other = (1 - p_self) / 2
        return {
            "BULL": {
                "BULL":  p_self,
                "RANGE": p_other,
                "BEAR":  p_other,
            }
        }

    def test_with_valid_matrix(self):
        result = compute_regime_persistence("BULL", self._make_matrix(0.9))
        assert result.regime == "BULL"
        assert result.expected_duration_days == pytest.approx(10.0, abs=0.5)
        assert result.self_transition_prob == pytest.approx(0.9, abs=0.01)
        assert result.persistence_score >= 0

    def test_with_empty_matrix_uses_fallback(self):
        result = compute_regime_persistence("BEAR", {})
        assert result.regime == "BEAR"
        assert result.expected_duration_days > 0
        assert result.persistence_label in ("ESTABLISHED", "MATURING", "FRESH", "EXHAUSTED")

    def test_exit_probs_logical(self):
        result = compute_regime_persistence("BULL", self._make_matrix(0.85))
        assert result.exit_prob_7d <= result.exit_prob_14d

    def test_most_likely_next_not_self(self):
        result = compute_regime_persistence("BULL", self._make_matrix(0.85))
        assert result.most_likely_next != "BULL"

    def test_result_frozen(self):
        result = compute_regime_persistence("BULL", self._make_matrix())
        with pytest.raises(Exception):
            result.regime = "BEAR"
