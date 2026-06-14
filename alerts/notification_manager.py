"""
Notification Manager – Multi-Channel Alert System
===================================================
Primary: Telegram
Secondary: LINE (with monthly quota detection)
Never raises exceptions – analysis continues even if all channels fail.
"""
import logging
from alerts.telegram_alert import send_telegram_message
from alerts.line_alert_v2 import send_line_message, LINE_DISABLED
from alerts.line_alert_v2 import send_line_message, LINE_DISABLED

logger = logging.getLogger(__name__)

def send_notification(message: str) -> bool:
    """
    Send notification via primary (Telegram) then fallback to LINE.
    Returns True if at least one channel succeeded, False otherwise.
    Never raises exceptions.
    """
    try:
        # 1. Try Telegram
        logger.debug("[Notification Manager] Attempting Telegram...")
        if send_telegram_message(message):
            logger.info("[Notification Manager] Notification sent via Telegram")
            return True
        else:
            logger.warning("[Notification Manager] Telegram failed, falling back to LINE")
    except Exception as e:
        logger.exception("[Notification Manager] Telegram raised exception: %s", e)

    try:
        # 2. Try LINE (if not disabled)
        if not LINE_DISABLED:
            logger.debug("[Notification Manager] Attempting LINE...")
            if send_line_message(message):
                logger.info("[Notification Manager] Notification sent via LINE")
                return True
            else:
                logger.warning("[Notification Manager] LINE failed")
        else:
            logger.info("[Notification Manager] LINE is disabled (monthly quota)")
    except Exception as e:
        logger.exception("[Notification Manager] LINE raised exception: %s", e)

    logger.error("[Notification Manager] All notification channels failed")
    return False