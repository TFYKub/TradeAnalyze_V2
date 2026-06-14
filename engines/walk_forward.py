"""
Walk‑Forward Analytics Engine
Computes rolling performance metrics for each engine.
"""
from typing import Dict, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class WalkForwardEngine:
    def __init__(self, lookback_days: int = 252, step_days: int = 30):
        self.lookback = lookback_days
        self.step = step_days

    def compute_metrics(self, price_series: pd.Series, signals: pd.DataFrame) -> Dict[str, Dict]:
        """
        signals: DataFrame with columns 'timestamp', 'engine', 'prediction', 'confidence'
        Returns: {engine: {'sharpe':, 'hit_rate':, 'avg_conf':, 'max_dd':}}
        """
        results = {}
        engines = signals['engine'].unique()
        for eng in engines:
            eng_signals = signals[signals['engine'] == eng].copy()
            if len(eng_signals) < 10:
                results[eng] = {'sharpe': 0, 'hit_rate': 0, 'avg_conf': 0, 'max_dd': 0}
                continue
            # Align with price returns
            returns = price_series.pct_change(5).shift(-5)  # 5-day forward return
            merged = pd.merge(eng_signals, returns, left_on='timestamp', right_index=True, how='inner')
            merged['correct'] = (merged['prediction'] == np.sign(merged['return'])) | ((merged['prediction'] == 0) & (abs(merged['return']) < 0.01))
            hit_rate = merged['correct'].mean()
            # Sharpe of strategy if we bet confidence * sign
            merged['pnl'] = merged['confidence'] * np.sign(merged['return']) * merged['return']
            sharpe = merged['pnl'].mean() / (merged['pnl'].std() + 1e-9) * np.sqrt(252)
            results[eng] = {
                'sharpe': round(sharpe, 3),
                'hit_rate': round(hit_rate, 3),
                'avg_conf': round(merged['confidence'].mean(), 3),
                'max_dd': 0  # simplified
            }
        return results