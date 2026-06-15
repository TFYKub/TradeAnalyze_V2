# config/config.py
import os
import json
import tempfile
from dotenv import load_dotenv

load_dotenv()

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_TOKEN")
SHEET_ID                  = os.getenv("SHEET_ID")
GOOGLE_CREDENTIALS_RAW    = os.getenv("GOOGLE_CREDENTIALS")

if not SHEET_ID:
    raise EnvironmentError("Missing SHEET_ID in environment variables")
if not GOOGLE_CREDENTIALS_RAW:
    raise EnvironmentError("Missing GOOGLE_CREDENTIALS in environment variables")

if GOOGLE_CREDENTIALS_RAW.strip().startswith('{'):
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    tmp.write(GOOGLE_CREDENTIALS_RAW)
    tmp.close()
    GOOGLE_CREDENTIALS_PATH = tmp.name
else:
    GOOGLE_CREDENTIALS_PATH = GOOGLE_CREDENTIALS_RAW

GOOGLE_CREDENTIALS = GOOGLE_CREDENTIALS_PATH

TIMEZONE = "Asia/Bangkok"

# ========== V3 Feature Flags ==========
USE_V3 = True

V3_PHASES = {
    "walk_forward": True,
    "quantile_forecast": True,
    "regime_switching_mc": True,
    "portfolio_optimizer": True,
    "crypto_flow_v2": True,
    "dealer_greeks": True,
    "stress_testing": True,
}

# ========== Performance & Scaling Flags ==========
USE_PARALLEL = True          # enable parallel symbol processing (ThreadPool)
MAX_WORKERS = 4              # number of concurrent workers

# ========== Notification Configuration ==========
LINE_TOKEN = os.getenv("LINE_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

LINE_DISABLED = False

print("\n[CONFIG] Notification channel status:")
if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
    print("  ✅ Telegram enabled")
else:
    print("  ⚠️ Telegram disabled (missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID)")

if LINE_TOKEN:
    print("  ✅ LINE enabled")
else:
    print("  ⚠️ LINE disabled (missing LINE_TOKEN)")
print("")