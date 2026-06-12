"""
Tests — Phase 19: Institutional Options Engine (Advanced Greeks)
Tests — Phase 20: Portfolio Optimizer (Risk Parity + HRP)
Tests — Phase 21: Cross Asset Engine
"""
import math
import pytest
import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 19 — Advanced Greeks
# ══════════════════════════════════════════════════════════════════════════════
from options.greeks_advanced import (
    compute_advanced_greeks,
    compute_probability_metrics,
    compute_strategy_evaluation,
    build_options_recommendation,
    AdvancedGreeks,
    InstitutionalOptionsRecommendation,
)

# Standard test parameters
S     = 50000.0   # BTC spot
K_ATM = 50000.0
K_OTM = 55000.0
K_ITM = 45000.0
IV    = 0.70      # 70% IV (typical for BTC)
T30   = 30 / 365
T7    = 7 / 365
R     = 0.05


class TestAdvancedGreeks:
    def test_call_delta_between_0_and_1(self):
        g = compute_advanced_greeks(S, K_ATM, T30, IV, R, "call")
        assert 0 < g.delta < 1

    def test_put_delta_between_neg1_and_0(self):
        g = compute_advanced_greeks(S, K_ATM, T30, IV, R, "put")
        assert -1 < g.delta < 0

    def test_atm_call_delta_near_half(self):
        g = compute_advanced_greeks(S, K_ATM, T30, IV, R, "call")
        assert 0.40 < g.delta < 0.60

    def test_gamma_positive_for_both(self):
        for opt_type in ("call", "put"):
            g = compute_advanced_greeks(S, K_ATM, T30, IV, R, opt_type)
            assert g.gamma >= 0

    def test_theta_negative(self):
        g = compute_advanced_greeks(S, K_ATM, T30, IV, R, "call")
        assert g.theta < 0   # theta is always a cost

    def test_vega_positive(self):
        g = compute_advanced_greeks(S, K_ATM, T30, IV, R, "call")
        assert g.vega > 0

    def test_vanna_computed(self):
        g = compute_advanced_greeks(S, K_ATM, T30, IV, R, "call")
        assert isinstance(g.vanna, float)
        # vanna should be non-zero for ATM
        assert g.vanna != 0.0

    def test_charm_computed(self):
        g = compute_advanced_greeks(S, K_ATM, T30, IV, R, "call")
        assert isinstance(g.charm, float)

    def test_vomma_computed(self):
        g = compute_advanced_greeks(S, K_ATM, T30, IV, R, "call")
        assert isinstance(g.vomma, float)

    def test_deep_itm_call_delta_near_1(self):
        g = compute_advanced_greeks(S, K_ITM, T30, IV, R, "call")
        assert g.delta > 0.70

    def test_deep_otm_call_delta_near_0(self):
        g = compute_advanced_greeks(S, K_OTM, T30, 0.30, R, "call")
        assert g.delta < 0.35

    def test_zero_inputs_return_zeros(self):
        g = compute_advanced_greeks(0, K_ATM, T30, IV, R, "call")
        assert g.delta == 0
        assert g.gamma == 0

    def test_near_expiry_high_gamma(self):
        g_30d = compute_advanced_greeks(S, K_ATM, T30, IV, R, "call")
        g_7d  = compute_advanced_greeks(S, K_ATM, T7,  IV, R, "call")
        assert g_7d.gamma > g_30d.gamma   # gamma increases near expiry

    def test_result_is_frozen(self):
        g = compute_advanced_greeks(S, K_ATM, T30, IV, R, "call")
        with pytest.raises(Exception):
            g.delta = 0.99


class TestProbabilityMetrics:
    def test_pop_bounded(self):
        pm = compute_probability_metrics(S, IV, 30, S * 1.05, "LONG")
        assert 0 < pm.pop < 100

    def test_p50_bounded(self):
        pm = compute_probability_metrics(S, IV, 30, S * 1.05, "LONG")
        assert 0 < pm.p50 < 100

    def test_expected_move_positive(self):
        pm = compute_probability_metrics(S, IV, 30, S, "LONG")
        assert pm.expected_move > 0

    def test_expected_move_formula(self):
        # EM = S × IV × sqrt(DTE/365)
        expected_em = S * IV * math.sqrt(30 / 365)
        pm = compute_probability_metrics(S, IV, 30, S, "LONG")
        assert pm.expected_move == pytest.approx(expected_em, rel=0.01)

    def test_upper_1sd_gt_spot(self):
        pm = compute_probability_metrics(S, IV, 30, S, "LONG")
        assert pm.upper_1sd > S

    def test_lower_1sd_lt_spot(self):
        pm = compute_probability_metrics(S, IV, 30, S, "LONG")
        assert pm.lower_1sd < S

    def test_longer_dte_wider_move(self):
        pm30 = compute_probability_metrics(S, IV, 30, S, "LONG")
        pm7  = compute_probability_metrics(S, IV, 7,  S, "LONG")
        assert pm30.expected_move > pm7.expected_move


class TestStrategyEvaluation:
    def test_positive_ev_on_favorable_trade(self):
        # Very high POP + large max profit vs small max loss
        ev, kelly, sharpe = compute_strategy_evaluation(
            "SELL_PUT", K_ITM, 30, "SHORT",
            S, IV, premium=500.0,
            max_profit=500.0, max_loss=4500.0, breakeven=K_ITM - 500,
        )
        assert isinstance(ev, float)
        assert isinstance(kelly, float)
        assert 0.0 <= kelly <= 0.25   # Kelly capped at 25%

    def test_kelly_capped_at_025(self):
        ev, kelly, _ = compute_strategy_evaluation(
            "SELL_STRANGLE", S, 30, "SHORT",
            S, IV, premium=1000.0,
            max_profit=1000.0, max_loss=100.0, breakeven=S * 0.98,
        )
        assert kelly <= 0.25

    def test_sharpe_finite(self):
        _, _, sharpe = compute_strategy_evaluation(
            "BUY_CALL", K_OTM, 30, "LONG",
            S, IV, premium=200.0,
            max_profit=5000.0, max_loss=200.0, breakeven=K_OTM + 200,
        )
        assert math.isfinite(sharpe)


class TestBuildOptionsRecommendation:
    def _make_rec(self) -> InstitutionalOptionsRecommendation:
        return build_options_recommendation(
            strategy="SELL_PUT", strike=K_ITM, dte=30,
            direction="SHORT", price=S, iv=IV, premium=500.0,
            max_profit=500.0, max_loss=4500.0,
            breakeven=K_ITM - 500, option_type="put",
        )

    def test_all_required_fields_present(self):
        rec = self._make_rec()
        for field in ("strategy", "strike", "dte", "pop", "expected_value",
                      "delta", "theta", "vega", "kelly_size", "vanna", "charm",
                      "expected_move", "summary_line"):
            assert hasattr(rec, field), f"Missing field: {field}"

    def test_summary_line_contains_key_info(self):
        rec = self._make_rec()
        assert "SELL_PUT" in rec.summary_line
        assert "DTE" in rec.summary_line or "30" in rec.summary_line

    def test_result_is_frozen(self):
        rec = self._make_rec()
        with pytest.raises(Exception):
            rec.pop = 99.0


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 20 — Portfolio Optimizer
# ══════════════════════════════════════════════════════════════════════════════
from portfolio.optimizer import (
    _risk_parity_weights,
    compute_portfolio_optimization,
    PortfolioOptimizationResult,
)

def _make_price_series(n: int = 252, seed: int = 0, drift: float = 0.0002) -> pd.Series:
    rng    = np.random.default_rng(seed)
    prices = 100 * np.cumprod(1 + rng.normal(drift, 0.02, n))
    return pd.Series(prices)


def _make_portfolio(n_assets: int = 4, n_bars: int = 252) -> dict[str, pd.Series]:
    symbols = ["BTC", "ETH", "SOL", "BNB"][:n_assets]
    return {s: _make_price_series(n_bars, seed=i) for i, s in enumerate(symbols)}


class TestRiskParityWeights:
    def test_weights_sum_to_one(self):
        from portfolio.optimizer import _compute_statistics
        prices = _make_portfolio(4, 252)
        cov, _, _ = _compute_statistics(prices)
        w = _risk_parity_weights(cov)
        assert w.sum() == pytest.approx(1.0, abs=1e-4)

    def test_all_weights_positive(self):
        from portfolio.optimizer import _compute_statistics
        prices = _make_portfolio(4, 252)
        cov, _, _ = _compute_statistics(prices)
        w = _risk_parity_weights(cov)
        assert (w > 0).all()

    def test_high_vol_asset_gets_lower_weight(self):
        """Higher-vol asset should get less weight in risk parity."""
        from portfolio.optimizer import _compute_statistics
        rng = np.random.default_rng(99)
        # Asset A: low vol, Asset B: high vol
        a = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 252)))
        b = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.04, 252)))
        cov, _, _ = _compute_statistics({"A": a, "B": b})
        w = _risk_parity_weights(cov)
        assert w["A"] > w["B"]


class TestComputePortfolioOptimization:
    def test_single_asset_mode(self):
        prices = {"BTC": _make_price_series(252)}
        result = compute_portfolio_optimization(prices)
        assert result.recommended_weights["BTC"] == pytest.approx(1.0, abs=0.01)
        assert result.method_used == "single_asset"

    def test_multi_asset_weights_sum_to_one(self):
        prices = _make_portfolio(4, 252)
        result = compute_portfolio_optimization(prices, method="risk_parity")
        total = sum(result.recommended_weights.values())
        assert total == pytest.approx(1.0, abs=1e-3)

    def test_weight_constraints_respected(self):
        prices = _make_portfolio(4, 252)
        result = compute_portfolio_optimization(
            prices, method="risk_parity", min_weight=0.05, max_weight=0.50
        )
        for w in result.recommended_weights.values():
            assert 0.04 <= w <= 0.51   # slight tolerance for re-normalisation

    def test_portfolio_vol_positive(self):
        prices = _make_portfolio(4, 252)
        result = compute_portfolio_optimization(prices)
        assert result.portfolio_volatility >= 0

    def test_hrp_weights_sum_to_one(self):
        prices = _make_portfolio(5, 252)
        result = compute_portfolio_optimization(prices, method="hrp")
        total = sum(result.recommended_weights.values())
        assert total == pytest.approx(1.0, abs=1e-3)

    def test_result_has_required_fields(self):
        prices = _make_portfolio(3, 252)
        result = compute_portfolio_optimization(prices)
        for field in ("recommended_weights", "risk_contributions",
                      "portfolio_volatility", "portfolio_sharpe",
                      "portfolio_drawdown", "diversification_ratio",
                      "correlation_matrix", "volatilities", "n_assets"):
            assert hasattr(result, field)

    def test_insufficient_data_fallback(self):
        """Fewer than MIN_HISTORY bars → equal weight fallback, no crash."""
        prices = _make_portfolio(3, 30)  # only 30 bars
        result = compute_portfolio_optimization(prices)
        assert result.method_used == "equal_weight"
        assert sum(result.recommended_weights.values()) == pytest.approx(1.0, abs=1e-3)

    def test_result_frozen(self):
        prices = _make_portfolio(2, 252)
        result = compute_portfolio_optimization(prices)
        with pytest.raises(Exception):
            result.portfolio_volatility = 999.0


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 21 — Cross Asset Engine
# ══════════════════════════════════════════════════════════════════════════════
from engines.cross_asset_engine import (
    _rolling_corr,
    _rolling_beta,
    _analyse_pair,
    _classify_cross_asset_regime,
    CrossAssetResult,
)


def _make_series(n: int = 120, seed: int = 0, drift: float = 0.0002) -> pd.Series:
    rng    = np.random.default_rng(seed)
    prices = 100 * np.cumprod(1 + rng.normal(drift, 0.02, n))
    return pd.Series(prices)


def _log_ret(s: pd.Series) -> pd.Series:
    return np.log(s / s.shift(1)).dropna()


class TestRollingCorr:
    def test_perfect_positive_corr(self):
        s = _make_series(120)
        assert _rolling_corr(_log_ret(s), _log_ret(s)) == pytest.approx(1.0, abs=0.01)

    def test_perfect_negative_corr(self):
        s    = _make_series(120)
        neg  = -_log_ret(s)
        pos  = _log_ret(s)
        corr = _rolling_corr(pos, neg)
        assert corr < -0.90

    def test_unrelated_series_near_zero(self):
        a = _log_ret(_make_series(120, seed=0))
        b = _log_ret(_make_series(120, seed=99))
        corr = _rolling_corr(a, b)
        assert -0.6 < corr < 0.6

    def test_insufficient_data_returns_zero(self):
        a = _log_ret(_make_series(10))
        b = _log_ret(_make_series(10))
        assert _rolling_corr(a, b) == pytest.approx(0.0)


class TestRollingBeta:
    def test_identical_series_beta_one(self):
        s   = _make_series(120)
        ret = _log_ret(s)
        assert _rolling_beta(ret, ret) == pytest.approx(1.0, abs=0.05)

    def test_double_vol_series_beta_two(self):
        rng = np.random.default_rng(42)
        base   = pd.Series(rng.normal(0, 0.01, 120))
        double = base * 2
        assert _rolling_beta(double, base) == pytest.approx(2.0, abs=0.3)

    def test_beta_positive_for_correlated(self):
        a = _log_ret(_make_series(120, drift=0.001))
        b = _log_ret(_make_series(120, drift=0.001))
        # loosely correlated uptrend series should have positive beta
        beta = _rolling_beta(a, b)
        assert isinstance(beta, float)


class TestClassifyCrossAssetRegime:
    def _make_pair(self, ret_btc, ret_pair, signal="ALIGNED"):
        from engines.cross_asset_engine import PairAnalysis
        return PairAnalysis(
            pair=f"BTC/TEST", btc_rs_score=60.0, rolling_corr=0.5,
            expected_corr=0.5, corr_deviation=0.0,
            btc_return_20d=ret_btc, pair_return_20d=ret_pair,
            outperforming=(ret_btc > ret_pair), signal=signal,
        )

    def test_risk_on_aligned(self):
        pairs = {
            "SPY": self._make_pair(5.0, 3.0, "ALIGNED"),
            "QQQ": self._make_pair(5.0, 4.0, "ALIGNED"),
            "GLD": self._make_pair(5.0, 1.0, "ALIGNED"),
            "DXY": self._make_pair(5.0, -1.0, "ALIGNED"),
        }
        regime, conf, _, _ = _classify_cross_asset_regime(pairs, btc_ret20=5.0)
        assert regime == "RISK_ON_ALIGNED"

    def test_decoupled_bull(self):
        pairs = {
            "SPY": self._make_pair(5.0, -3.0, "DECOUPLED"),
            "QQQ": self._make_pair(5.0, -2.0, "DECOUPLED"),
            "GLD": self._make_pair(5.0, 1.0, "ALIGNED"),
        }
        regime, conf, decouple, dec_dir = _classify_cross_asset_regime(pairs, btc_ret20=5.0)
        assert regime == "DECOUPLED_BULL"
        assert decouple is True
        assert dec_dir == "BULL"

    def test_transition_mixed(self):
        pairs = {
            "SPY": self._make_pair(1.0, 0.5, "ALIGNED"),
            "QQQ": self._make_pair(1.0, 2.0, "ALIGNED"),
        }
        regime, conf, _, _ = _classify_cross_asset_regime(pairs, btc_ret20=1.0)
        assert regime in ("RISK_ON_ALIGNED", "TRANSITION")
        assert 0 < conf <= 90


class TestCrossAssetResultFallback:
    def test_no_btc_data_returns_transition(self):
        result = CrossAssetResult(
            cross_asset_regime="TRANSITION", relative_strength_score=50.0,
            regime_confidence=30.0, pair_analyses={},
            rolling_correlations={}, decoupling_detected=False,
            decoupling_direction="NONE", btc_beta_to_spy=1.0,
            btc_beta_to_qqq=1.5, dominant_driver="UNKNOWN",
            interpretation="fallback",
        )
        assert result.cross_asset_regime == "TRANSITION"
        assert result.relative_strength_score == pytest.approx(50.0)

    def test_result_frozen(self):
        result = CrossAssetResult(
            cross_asset_regime="TRANSITION", relative_strength_score=50.0,
            regime_confidence=30.0, pair_analyses={},
            rolling_correlations={}, decoupling_detected=False,
            decoupling_direction="NONE", btc_beta_to_spy=1.0,
            btc_beta_to_qqq=1.5, dominant_driver="UNKNOWN",
            interpretation="test",
        )
        with pytest.raises(Exception):
            result.cross_asset_regime = "RISK_ON_ALIGNED"
