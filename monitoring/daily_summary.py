"""
Daily Summary Report – Phase 5.4
=================================
Aggregates trade history from the last 24 hours and sends a summary to Telegram.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Optional

from persistence.trade_persistence import get_persistence
from analytics.performance_attribution import TRACKER, PerformanceAttributionResult
from alerts.notification_manager import send_notification

logger = logging.getLogger(__name__)


def get_daily_summary() -> Optional[str]:
    """
    Generate a formatted summary of today's trading activity.
    Returns None if no trades were closed today.
    """
    persistence = get_persistence()
    # Load recent history (last 100 trades, then filter by date)
    history = persistence.load_recent_history(limit=200)
    now = datetime.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    today_trades = [
        t for t in history
        if datetime.fromisoformat(t.exit_time) >= today_start
    ]

    if not today_trades:
        return None

    total_pnl = sum(t.pnl_pct for t in today_trades)
    wins = [t for t in today_trades if t.pnl_pct > 0]
    losses = [t for t in today_trades if t.pnl_pct <= 0]
    win_rate = len(wins) / len(today_trades) * 100 if today_trades else 0
    avg_win = sum(t.pnl_pct for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.pnl_pct for t in losses) / len(losses) if losses else 0
    best = max(today_trades, key=lambda t: t.pnl_pct) if today_trades else None
    worst = min(today_trades, key=lambda t: t.pnl_pct) if today_trades else None

    # Compute overall performance metrics from tracker
    attribution = TRACKER.compute_attribution()
    sharpe = attribution.sharpe if attribution else 0.0
    max_dd = attribution.max_drawdown if attribution else 0.0

    summary = (
        f"📊 DAILY TRADING SUMMARY – {now.strftime('%Y-%m-%d')}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Trades closed: {len(today_trades)}\n"
        f"Total P&L: {total_pnl:+.2f}%\n"
        f"Win rate: {win_rate:.1f}%\n"
        f"Avg win: {avg_win:+.2f}% | Avg loss: {avg_loss:+.2f}%\n"
        f"Best trade: {best.pnl_pct:+.2f}% ({best.symbol} {best.direction})\n"
        f"Worst trade: {worst.pnl_pct:+.2f}% ({worst.symbol} {worst.direction})\n"
        f"Portfolio Sharpe: {sharpe:.2f}\n"
        f"Max drawdown: {max_dd:.1f}%\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    return summary


def send_daily_summary():
    """Generate and send the daily summary report if there were trades."""
    summary = get_daily_summary()
    if summary:
        send_notification(summary)
        logger.info("[daily_summary] Report sent")
    else:
        logger.info("[daily_summary] No trades today – skipping report")