"""
Signal Tracking Database – SQLite
Stores every signal prediction for future walk‑forward analysis.
"""
import sqlite3
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

@dataclass
class SignalRecord:
    timestamp: datetime
    symbol: str
    engine: str
    predicted_direction: str   # "LONG", "SHORT", "NEUTRAL"
    confidence: float
    actual_return_5d: Optional[float] = None
    actual_direction: Optional[str] = None

class SignalDatabase:
    def __init__(self, db_path: str = "signals.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                engine TEXT,
                predicted_direction TEXT,
                confidence REAL,
                actual_return_5d REAL,
                actual_direction TEXT,
                recorded_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_engine_time ON signals(engine, timestamp)")

    def insert(self, record: SignalRecord):
        self.conn.execute("""
            INSERT INTO signals (timestamp, symbol, engine, predicted_direction, confidence, actual_return_5d, actual_direction)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (record.timestamp.isoformat(), record.symbol, record.engine, record.predicted_direction,
              record.confidence, record.actual_return_5d, record.actual_direction))
        self.conn.commit()

    def update_actuals(self, symbol: str, engine: str, timestamp: datetime, actual_return: float):
        actual_dir = "LONG" if actual_return > 0.01 else "SHORT" if actual_return < -0.01 else "NEUTRAL"
        self.conn.execute("""
            UPDATE signals SET actual_return_5d = ?, actual_direction = ?
            WHERE symbol = ? AND engine = ? AND timestamp = ?
        """, (actual_return, actual_dir, symbol, engine, timestamp.isoformat()))
        self.conn.commit()