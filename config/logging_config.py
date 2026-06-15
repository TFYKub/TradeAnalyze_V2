# config/logging_config.py – with JSON structured logging and rotation
import logging
import logging.handlers
import json
import os
import time
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ── JSON Formatter ───────────────────────────────────────────────────────────
class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        # Add extra fields (if any)
        if hasattr(record, "extra"):
            log_entry.update(record.extra)
        return json.dumps(log_entry)

# ── Rotating File Handler (10 MB per file, keep 5 backups) ───────────────────
file_handler = logging.handlers.RotatingFileHandler(
    f"{LOG_DIR}/trade_analyze.log",
    maxBytes=10_485_760,  # 10 MB
    backupCount=5,
    encoding="utf-8"
)
file_handler.setFormatter(JSONFormatter())

# ── Console handler (still human-readable for debugging) ─────────────────────
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
)

# ── Configure root logger ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)

logger = logging.getLogger("TradeAnalyze")