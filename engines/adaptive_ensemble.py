# engines/adaptive_ensemble.py
from typing import Dict, List
from datetime import datetime, timedelta
from engines.signal_tracker import SignalDatabase

class AdaptiveEnsemble:
    def __init__(self, signal_db: SignalDatabase):
        self.db = signal_db

    def get_weights(self, current_date: datetime, engines: List[str]) -> Dict[str, float]:
        perf = {}
        for eng in engines:
            row = self.db.conn.execute("""
                SELECT COUNT(*), AVG(confidence) FROM signals
                WHERE engine = ? AND timestamp > ? AND actual_direction IS NOT NULL
            """, (eng, (current_date - timedelta(days=252)).isoformat())).fetchone()
            if row and row[0] > 10:
                perf[eng] = row[1] or 0.5
            else:
                perf[eng] = 0.5
        total = sum(perf.values())
        if total == 0:
            return {e: 1/len(engines) for e in engines}
        return {e: perf[e]/total for e in engines}