"""
Watchdog – Phase 5.5
=====================
Monitors bot health via timestamp file. If the bot hasn't run in more than
`STALL_THRESHOLD_SECONDS`, send an alert to Telegram/LINE.
"""
from __future__ import annotations

import os
import time
import logging
from datetime import datetime

from alerts.notification_manager import send_notification

logger = logging.getLogger(__name__)

HEALTH_FILE = "logs/.last_run"
STALL_THRESHOLD_SECONDS = 3600  # 1 hour – adjust as needed


def update_last_run():
    """Update the timestamp file after a successful run."""
    try:
        with open(HEALTH_FILE, "w") as f:
            f.write(str(time.time()))
        logger.info("[watchdog] Updated last run timestamp")
    except Exception as e:
        logger.error(f"[watchdog] Failed to write timestamp: {e}")


def get_last_run() -> float:
    """Read last run timestamp from file, or return 0 if file missing."""
    try:
        with open(HEALTH_FILE, "r") as f:
            return float(f.read().strip())
    except (FileNotFoundError, ValueError):
        return 0.0


def check_and_alert() -> bool:
    """
    Check if bot has stalled. Returns True if healthy, False if stalled.
    Sends an alert if stalled.
    """
    last_run = get_last_run()
    if last_run == 0:
        logger.info("[watchdog] No previous run found – assuming first run")
        return True

    elapsed = time.time() - last_run
    if elapsed > STALL_THRESHOLD_SECONDS:
        msg = (
            f"🚨 BOT STALLED ALERT\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Last successful run: {datetime.fromtimestamp(last_run).isoformat()}\n"
            f"Elapsed: {elapsed / 60:.1f} minutes\n"
            f"Threshold: {STALL_THRESHOLD_SECONDS / 60:.0f} minutes\n"
            f"Action: Check bot logs and restart if necessary."
        )
        send_notification(msg)
        logger.warning(f"[watchdog] Bot stalled – elapsed {elapsed:.0f}s")
        return False

    logger.debug(f"[watchdog] Bot healthy – last run {elapsed:.0f}s ago")
    return True


# If called directly as a script, run the check
if __name__ == "__main__":
    # Configure basic logging for standalone execution
    logging.basicConfig(level=logging.INFO)
    check_and_alert()