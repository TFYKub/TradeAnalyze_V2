"""
Telegram Alert Module – Primary Notification Channel
======================================================
Sends messages via Telegram Bot API with retries and exponential backoff.
Never raises uncaught exceptions – returns True/False only.
"""
import logging
import time
import random
import requests
from typing import Optional

from config.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
INITIAL_DELAY = 1.0
MAX_DELAY = 60.0

_TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

def send_telegram_message(message: str) -> bool:
    """
    Send a message via Telegram Bot API.
    Returns True if successful, False otherwise.
    Never raises exceptions.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("[Telegram] Missing credentials – notifications disabled")
        return False

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message[:4096],
        # "parse_mode": "HTML",   # disabled to avoid HTML parsing errors
    }

    delay = INITIAL_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.debug("[Telegram] Sending message (attempt %d/%d)", attempt, MAX_RETRIES)
            resp = requests.post(_TELEGRAM_API_URL, json=payload, timeout=30)

            if resp.status_code == 200:
                logger.info("[Telegram] Message sent successfully")
                return True

            # Rate limit (429) – retry with backoff
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = int(retry_after)
                    except ValueError:
                        wait = delay
                else:
                    wait = delay
                wait += random.uniform(0, 2)
                logger.warning("[Telegram] Rate limit (429), waiting %.1fs", wait)
                time.sleep(wait)
                delay = min(delay * 2, MAX_DELAY)
                continue

            # Other errors – log and stop retrying
            logger.error("[Telegram] HTTP %d: %s", resp.status_code, resp.text[:200])
            return False

        except requests.Timeout:
            logger.warning("[Telegram] Timeout (attempt %d/%d)", attempt, MAX_RETRIES)
            if attempt == MAX_RETRIES:
                return False
            time.sleep(delay)
            delay = min(delay * 2, MAX_DELAY)
            continue
        except Exception as exc:
            logger.error("[Telegram] Unexpected error: %s", exc)
            return False

    logger.error("[Telegram] Failed after %d retries", MAX_RETRIES)
    return False