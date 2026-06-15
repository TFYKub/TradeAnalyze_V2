"""
Trade Persistence Layer – SQLite with thread‑local connections.
"""
from __future__ import annotations

import sqlite3
import json
import logging
import threading
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DB_PATH = "trade_analyze.db"


@dataclass
class ActiveTrade:
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    tp1: float
    tp2: float
    position_size: float
    entry_time: str
    entry_snapshot: str
    trade_id: str
    last_updated: str


@dataclass
class ClosedTrade:
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    exit_time: str
    pnl_pct: float
    trade_id: str
    entry_snapshot: str


class TradePersistence:
    """Singleton with thread‑local SQLite connections."""
    _instance = None
    _lock = threading.Lock()
    _thread_local = threading.local()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_db()
        return cls._instance

    def _init_db(self):
        """Create tables if they don't exist (main thread only)."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS active_trades (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                tp1 REAL,
                tp2 REAL,
                position_size REAL NOT NULL,
                entry_time TEXT NOT NULL,
                entry_snapshot TEXT NOT NULL,
                last_updated TEXT NOT NULL
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_active_symbol ON active_trades(symbol)")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_history (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                exit_time TEXT NOT NULL,
                pnl_pct REAL NOT NULL,
                entry_snapshot TEXT NOT NULL
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_symbol ON trade_history(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_time ON trade_history(exit_time)")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS engine_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                engine TEXT NOT NULL,
                predicted_direction TEXT NOT NULL,
                confidence REAL NOT NULL,
                actual_return_5d REAL,
                actual_direction TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_signal_engine ON engine_signals(engine, timestamp)")

        conn.commit()

    def _get_conn(self):
        if not hasattr(self._thread_local, "conn"):
            self._thread_local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            self._thread_local.conn.row_factory = sqlite3.Row
        return self._thread_local.conn

    @contextmanager
    def _cursor(self):
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    def save_active_trade(self, trade: ActiveTrade) -> None:
        with self._cursor() as cur:
            cur.execute("""
                INSERT OR REPLACE INTO active_trades
                (trade_id, symbol, direction, entry_price, stop_loss, tp1, tp2,
                 position_size, entry_time, entry_snapshot, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.trade_id, trade.symbol, trade.direction, trade.entry_price,
                trade.stop_loss, trade.tp1, trade.tp2, trade.position_size,
                trade.entry_time, trade.entry_snapshot, trade.last_updated
            ))
        logger.info("[persistence] Saved active trade %s", trade.trade_id)

    def delete_active_trade(self, trade_id: str) -> None:
        with self._cursor() as cur:
            cur.execute("DELETE FROM active_trades WHERE trade_id = ?", (trade_id,))
        logger.info("[persistence] Deleted active trade %s", trade_id)

    def load_active_trades(self) -> List[ActiveTrade]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM active_trades")
            rows = cur.fetchall()
        trades = []
        for row in rows:
            trades.append(ActiveTrade(
                symbol=row["symbol"],
                direction=row["direction"],
                entry_price=row["entry_price"],
                stop_loss=row["stop_loss"],
                tp1=row["tp1"],
                tp2=row["tp2"],
                position_size=row["position_size"],
                entry_time=row["entry_time"],
                entry_snapshot=row["entry_snapshot"],
                trade_id=row["trade_id"],
                last_updated=row["last_updated"],
            ))
        return trades

    def has_active_trade(self, symbol: str) -> bool:
        with self._cursor() as cur:
            cur.execute("SELECT 1 FROM active_trades WHERE symbol = ? LIMIT 1", (symbol,))
            return cur.fetchone() is not None

    def save_closed_trade(self, closed: ClosedTrade) -> None:
        with self._cursor() as cur:
            cur.execute("""
                INSERT OR REPLACE INTO trade_history
                (trade_id, symbol, direction, entry_price, exit_price, exit_time,
                 pnl_pct, entry_snapshot)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                closed.trade_id, closed.symbol, closed.direction, closed.entry_price,
                closed.exit_price, closed.exit_time, closed.pnl_pct, closed.entry_snapshot
            ))
        logger.info("[persistence] Saved closed trade %s pnl=%.2f%%", closed.trade_id, closed.pnl_pct)

    def load_recent_history(self, limit: int = 100) -> List[ClosedTrade]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM trade_history ORDER BY exit_time DESC LIMIT ?", (limit,))
            rows = cur.fetchall()
        return [
            ClosedTrade(
                symbol=r["symbol"], direction=r["direction"],
                entry_price=r["entry_price"], exit_price=r["exit_price"],
                exit_time=r["exit_time"], pnl_pct=r["pnl_pct"],
                trade_id=r["trade_id"], entry_snapshot=r["entry_snapshot"]
            ) for r in rows
        ]

    def insert_signal(self, timestamp: str, symbol: str, engine: str,
                      predicted_direction: str, confidence: float) -> None:
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO engine_signals
                (timestamp, symbol, engine, predicted_direction, confidence)
                VALUES (?, ?, ?, ?, ?)
            """, (timestamp, symbol, engine, predicted_direction, confidence))

    def update_signal_actual(self, timestamp: str, symbol: str, engine: str,
                             actual_return: float) -> None:
        actual_dir = "LONG" if actual_return > 0.01 else "SHORT" if actual_return < -0.01 else "NEUTRAL"
        with self._cursor() as cur:
            cur.execute("""
                UPDATE engine_signals
                SET actual_return_5d = ?, actual_direction = ?
                WHERE timestamp = ? AND symbol = ? AND engine = ?
            """, (actual_return, actual_dir, timestamp, symbol, engine))

    def clear_all_active_trades(self) -> None:
        with self._cursor() as cur:
            cur.execute("DELETE FROM active_trades")
        logger.warning("[persistence] Cleared all active trades")

    def close(self):
        if hasattr(self._thread_local, "conn"):
            self._thread_local.conn.close()
            del self._thread_local.conn


_persistence = None


def get_persistence() -> TradePersistence:
    global _persistence
    if _persistence is None:
        _persistence = TradePersistence()
    return _persistence