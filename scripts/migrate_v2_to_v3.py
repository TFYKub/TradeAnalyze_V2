"""
Migration script: creates V3 sheets and initialises SQLite database.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reports.sheet_writer_v3 import create_v3_sheets
from engines.signal_tracker import SignalDatabase

def main():
    print("Creating V3 Google Sheets...")
    create_v3_sheets()
    print("Initialising signal database...")
    db = SignalDatabase()
    print("Migration complete. You can now set USE_V3=True in config.py")

if __name__ == "__main__":
    main()