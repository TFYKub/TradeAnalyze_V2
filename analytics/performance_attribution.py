"""
Performance Attribution Engine  (Phase 11)
============================================
Tracks contribution to P&L from each engine signal:
  Markov Regime, Trend, Options, Risk, Volatility

Attribution method: Brinson-Hood-Beebower style
  Each signal's contribution = signal_return × signal_weight

For a closed trade, decompose the P&L into:
  1. What % came from correct regime identification?
  2. What % from trend alignment?
  3. What % from options timing?
  4. What % from proper risk sizing?
  5. What % from vol regime awareness?

Also tracks rolling statistics: Sharpe, Sortino, Calmar per engine.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

ENGINES = ["markov", "trend", "options", "risk", "volatility"]

# Attribution weights (how much each engine influenced the trade decision)
# These are configurable priors — update with actual backtest attribution
DEFAULT_WEIGHTS = {
    "markov":     0.30,
    "trend":      0.25,
    "options":    0.20,
    "risk":       0.15,
    "volatility": 0.10,
}

TRADING_DAYS = 252


@dataclass
class TradeRecord:
    """Single completed trade for attribution tracking."""
    symbol:        str
    direction:     str
    entry:         float
    exit:          float
    pnl_pct:       float              # actual P&L %
    trade_date:    str
    # Signal scores at entry (0–100 each)
    markov_score:  float = 50.0
    trend_score:   float = 50.0
    options_score: float = 50.0
    risk_score:    float = 50.0
    vol_score:     float = 50.0
    regime:        str = ""
    ai_score:      float = 0.0
    rr:            float = 0.0


@dataclass(frozen=True)
class EngineAttribution:
    engine:             str
    total_contribution: float    # sum of attributed P&L
    contribution_pct:   float    # % of total P&L attributed
    avg_score:          float    # average signal score
    sharpe:             float
    sortino:            float
    win_rate:           float
    n_trades:           int


@dataclass(frozen=True)
class PerformanceAttributionResult:
    total_pnl:          float
    total_return_pct:   float
    win_rate:           float
    n_trades:           int
    sharpe:             float
    sortino:            float
    calmar:             float
    max_drawdown:       float
    engine_attributions: dict[str, EngineAttribution]
    top_contributing_engine: str
    weakest_engine:     str
    attribution_summary: str


class PerformanceTracker:
    """
    Maintains a rolling log of completed trades and computes attribution.
    Instantiate once and call add_trade() for each closed trade.
    """

    def __init__(self):
        self._trades: list[TradeRecord] = []

    def add_trade(self, trade: TradeRecord) -> None:
        self._trades.append(trade)
        logger.info("[attribution] trade added: %s %s pnl=%.2f%%",
                    trade.symbol, trade.direction, trade.pnl_pct)

    @property
    def n_trades(self) -> int:
        return len(self._trades)

    def compute_attribution(self) -> PerformanceAttributionResult | None:
        """Compute full attribution across all recorded trades."""
        if not self._trades:
            return None

        pnls    = np.array([t.pnl_pct for t in self._trades])
        total   = float(pnls.sum())
        wins    = int((pnls > 0).sum())
        wr      = wins / len(pnls)

        # Portfolio risk metrics
        daily_returns = pnls / 100
        sharpe   = _sharpe(daily_returns)
        sortino  = _sortino(daily_returns)
        max_dd   = _max_drawdown(daily_returns)
        calmar   = (float(np.mean(daily_returns) * TRADING_DAYS * 100) / abs(max_dd)
                    if max_dd != 0 else 0.0)

        # Engine attributions
        engine_results: dict[str, EngineAttribution] = {}

        for engine in ENGINES:
            score_key = f"{engine}_score"
            scores    = np.array([getattr(t, score_key, 50.0) for t in self._trades])
            weight    = DEFAULT_WEIGHTS[engine]

            # Attribution = trade P&L × (engine_score / 100) × weight
            attributed = pnls * (scores / 100) * weight
            total_attr = float(attributed.sum())
            attr_pct   = total_attr / total * 100 if total != 0 else 0

            attr_wins  = int((attributed > 0).sum())
            attr_wr    = attr_wins / len(attributed)
            attr_sharpe= _sharpe(attributed / 100)
            attr_sort  = _sortino(attributed / 100)

            engine_results[engine] = EngineAttribution(
                engine             = engine,
                total_contribution = round(total_attr, 3),
                contribution_pct   = round(attr_pct, 1),
                avg_score          = round(float(scores.mean()), 1),
                sharpe             = round(attr_sharpe, 3),
                sortino            = round(attr_sort, 3),
                win_rate           = round(attr_wr, 3),
                n_trades           = len(self._trades),
            )

        top     = max(engine_results, key=lambda e: engine_results[e].total_contribution)
        weakest = min(engine_results, key=lambda e: engine_results[e].total_contribution)

        summary = (
            f"{len(self._trades)} trades | Return={total:+.2f}% | "
            f"WR={wr*100:.0f}% | Sharpe={sharpe:.2f} | MaxDD={max_dd:.1f}% | "
            f"Best engine: {top} | Weakest: {weakest}"
        )

        return PerformanceAttributionResult(
            total_pnl            = round(total, 3),
            total_return_pct     = round(total, 2),
            win_rate             = round(wr, 3),
            n_trades             = len(self._trades),
            sharpe               = round(sharpe, 3),
            sortino              = round(sortino, 3),
            calmar               = round(calmar, 3),
            max_drawdown         = round(max_dd, 2),
            engine_attributions  = engine_results,
            top_contributing_engine = top,
            weakest_engine       = weakest,
            attribution_summary  = summary,
        )


# ── Stats helpers ─────────────────────────────────────────────────────────────
def _sharpe(r: np.ndarray) -> float:
    if len(r) < 2 or r.std() == 0: return 0.0
    return float((r.mean() - 0.05 / TRADING_DAYS) / r.std() * math.sqrt(TRADING_DAYS))


def _sortino(r: np.ndarray) -> float:
    down = r[r < 0]
    if len(down) < 2 or down.std() == 0: return 0.0
    return float((r.mean() - 0.05 / TRADING_DAYS) / down.std() * math.sqrt(TRADING_DAYS))


def _max_drawdown(r: np.ndarray) -> float:
    eq  = np.cumprod(1 + r)
    pk  = np.maximum.accumulate(eq)
    dd  = (eq - pk) / pk
    return float(dd.min() * 100) if len(dd) > 0 else 0.0


# Singleton tracker — import and use across sessions
TRACKER = PerformanceTracker()
