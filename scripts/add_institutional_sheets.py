# scripts/add_institutional_sheets.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import SHEET_ID
from utils.sheets_auth import get_sheets_client

def main():
    print(f"Connecting to sheet: {SHEET_ID}")
    gc = get_sheets_client()
    sh = gc.open_by_key(SHEET_ID)
    new_sheets = ["InstitutionalSignals", "RegimeDashboard", "FlowSnapshot"]
    for name in new_sheets:
        try:
            sh.worksheet(name)
            print(f"✅ Worksheet '{name}' already exists.")
        except Exception:
            try:
                sh.add_worksheet(title=name, rows=5000, cols=60)
                print(f"✅ Created worksheet: {name}")
            except Exception as e:
                print(f"❌ Failed to create '{name}': {e}")

if __name__ == "__main__":
    main()