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

# If GOOGLE_CREDENTIALS_RAW is a JSON string (starts with '{'), write to temp file.
if GOOGLE_CREDENTIALS_RAW.strip().startswith('{'):
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    tmp.write(GOOGLE_CREDENTIALS_RAW)
    tmp.close()
    GOOGLE_CREDENTIALS_PATH = tmp.name
else:
    GOOGLE_CREDENTIALS_PATH = GOOGLE_CREDENTIALS_RAW

GOOGLE_CREDENTIALS = GOOGLE_CREDENTIALS_PATH   # for compatibility with existing code

TIMEZONE = "Asia/Bangkok"