"""
Prometheus Metrics Helper – Phase 5.3
======================================
Provides convenient functions to update metrics from anywhere in the code.
All functions are safe even if prometheus_client is not installed.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

# Import from health_server – those functions already handle missing prometheus
from monitoring.health_server import (
    record_trade as _record_trade,
    record_api_latency as _record_api_latency,
    update_win_rate as _update_win_rate,
    update_portfolio_metrics as _update_portfolio_metrics,
)


def record_trade(symbol: str, direction: str, pnl_pct: float):
    """Record a closed trade for metrics."""
    _record_trade(symbol, direction, pnl_pct)


def update_win_rate(win_rate: float):
    """Update rolling win rate gauge."""
    _update_win_rate(win_rate)


def update_portfolio_metrics(volatility: float, drawdown: float):
    """Update portfolio volatility and drawdown gauges."""
    _update_portfolio_metrics(volatility, drawdown)


@contextmanager
def measure_api_latency(endpoint: str):
    """Context manager to measure API call latency."""
    start = time.time()
    yield
    duration = time.time() - start
    _record_api_latency(endpoint, duration)