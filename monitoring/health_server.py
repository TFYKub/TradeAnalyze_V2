"""
HTTP Health & Metrics Server – Phase 5.1 & 5.3
================================================
Provides /health and /metrics endpoints via Flask.
Runs in a background daemon thread so it doesn't block main processing.

If Flask or prometheus_client are not installed, the server will not start,
but the bot will continue to run (metrics will be no-ops).
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Try to import Flask and Prometheus; fallback to dummy implementations
FLASK_AVAILABLE = False
PROMETHEUS_AVAILABLE = False

try:
    import prometheus_client
    from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    prometheus_client = None
    logger.warning("prometheus_client not installed – metrics will be disabled")

try:
    from flask import Flask, Response, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    Flask = None
    logger.warning("Flask not installed – health server will not start")

# ── Prometheus Metrics (dummy if not available) ──────────────────────────────
if PROMETHEUS_AVAILABLE:
    TRADE_COUNT = Counter(
        "tradeanalyze_trades_total",
        "Total number of trades executed",
        ["symbol", "direction", "result"]
    )
    WIN_RATE = Gauge("tradeanalyze_win_rate", "Rolling win rate over last 100 trades")
    PORTFOLIO_DD = Gauge("tradeanalyze_max_drawdown_pct", "Current portfolio max drawdown %")
    PORTFOLIO_VOL = Gauge("tradeanalyze_volatility_annual_pct", "Portfolio annualised volatility %")
    API_LATENCY = Histogram(
        "tradeanalyze_api_latency_seconds",
        "Latency of external API calls",
        ["endpoint"],
        buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
    )
    BOT_STATUS = Gauge("tradeanalyze_bot_status", "Bot health status: 1 = running, 0 = stalled/crashed")
    LAST_RUN_TIMESTAMP = Gauge("tradeanalyze_last_run_timestamp", "Unix timestamp of last successful run")
    OPEN_TRADES = Gauge("tradeanalyze_open_trades", "Number of currently open positions")
else:
    # Dummy implementations (no-ops)
    class _DummyCounter:
        def labels(self, *args, **kwargs):
            return self
        def inc(self, *args, **kwargs):
            pass
    class _DummyGauge:
        def set(self, *args, **kwargs):
            pass
    class _DummyHistogram:
        def labels(self, *args, **kwargs):
            return self
        def observe(self, *args, **kwargs):
            pass

    TRADE_COUNT = _DummyCounter()
    WIN_RATE = _DummyGauge()
    PORTFOLIO_DD = _DummyGauge()
    PORTFOLIO_VOL = _DummyGauge()
    API_LATENCY = _DummyHistogram()
    BOT_STATUS = _DummyGauge()
    LAST_RUN_TIMESTAMP = _DummyGauge()
    OPEN_TRADES = _DummyGauge()

# ── Health data ───────────────────────────────────────────────────────────────
_health_data = {
    "last_run_timestamp": None,
    "status": "starting",
    "open_positions": 0,
    "version": "2.0",
    "symbols_processed": 0,
    "errors": 0,
}
_health_lock = threading.Lock()


def update_health(
    status: str = None,
    open_positions: int = None,
    symbols_processed: int = None,
    errors: int = None,
):
    """Update the health data (thread‑safe)."""
    with _health_lock:
        if status is not None:
            _health_data["status"] = status
            if PROMETHEUS_AVAILABLE:
                BOT_STATUS.set(1 if status == "running" else 0)
        if open_positions is not None:
            _health_data["open_positions"] = open_positions
            if PROMETHEUS_AVAILABLE:
                OPEN_TRADES.set(open_positions)
        if symbols_processed is not None:
            _health_data["symbols_processed"] = symbols_processed
        if errors is not None:
            _health_data["errors"] = errors
        if _health_data["last_run_timestamp"] is None:
            _health_data["last_run_timestamp"] = time.time()
            if PROMETHEUS_AVAILABLE:
                LAST_RUN_TIMESTAMP.set(_health_data["last_run_timestamp"])


def record_trade(symbol: str, direction: str, pnl_pct: float):
    """Record a closed trade for metrics."""
    if PROMETHEUS_AVAILABLE:
        result = "win" if pnl_pct > 0 else "loss"
        TRADE_COUNT.labels(symbol=symbol, direction=direction, result=result).inc()


def record_api_latency(endpoint: str, duration: float):
    """Record external API call latency."""
    if PROMETHEUS_AVAILABLE:
        API_LATENCY.labels(endpoint=endpoint).observe(duration)


def update_win_rate(win_rate: float):
    """Update the rolling win rate gauge."""
    if PROMETHEUS_AVAILABLE:
        WIN_RATE.set(win_rate)


def update_portfolio_metrics(volatility: float, drawdown: float):
    """Update portfolio volatility and drawdown gauges."""
    if PROMETHEUS_AVAILABLE:
        PORTFOLIO_VOL.set(volatility)
        PORTFOLIO_DD.set(drawdown)


def _create_flask_app():
    """Create Flask app with health and metrics endpoints (only if Flask available)."""
    if not FLASK_AVAILABLE:
        return None
    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        with _health_lock:
            data = _health_data.copy()
        data["timestamp"] = datetime.now().isoformat()
        return jsonify(data)

    @app.route("/metrics", methods=["GET"])
    def metrics():
        if PROMETHEUS_AVAILABLE:
            return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)
        else:
            return "Prometheus client not installed", 503

    @app.route("/ready", methods=["GET"])
    def ready():
        with _health_lock:
            if _health_data["status"] == "running":
                return "OK", 200
            return "NOT READY", 503

    return app


def start_health_server(port: int = 8080) -> Optional[threading.Thread]:
    """Start the health server in a background daemon thread, if Flask available."""
    if not FLASK_AVAILABLE:
        logger.info("Flask not installed – health server not started")
        return None

    app = _create_flask_app()
    if app is None:
        return None

    from threading import Thread
    server_thread = Thread(
        target=lambda: app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False),
        daemon=True
    )
    server_thread.start()
    logger.info(f"Health server started on port {port}")
    return server_thread