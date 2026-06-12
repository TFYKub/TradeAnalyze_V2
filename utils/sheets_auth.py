"""
Shared Google Sheets authentication helper.
Single source of truth — imported by symbol_loader and sheet_writer.
Uses google-auth (modern) instead of oauth2client.
"""

import json
import os

import gspread
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]


def get_sheets_client() -> gspread.Client:
    """Authenticate and return an authorised gspread client."""

    creds_json = os.getenv("GOOGLE_CREDENTIALS")

    if not creds_json:
        raise EnvironmentError("Missing GOOGLE_CREDENTIALS environment variable")

    # If it's a JSON string (from GitHub secret or .env with JSON), parse it.
    # Otherwise treat it as a file path.
    if creds_json.strip().startswith('{'):
        creds_dict = json.loads(creds_json)
    else:
        # It's a file path
        with open(creds_json, 'r') as f:
            creds_dict = json.load(f)

    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    return gspread.authorize(creds)