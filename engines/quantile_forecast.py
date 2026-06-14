"""
Quantile Forecast Engine using LightGBM quantile regression.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
import lightgbm as lgb

@dataclass
class QuantileForecastResult:
    horizon: int
    quantiles: Dict[float, float]   # {0.05: , 0.5:, 0.95:}
    mean: float
    std: float
    var_95: float
    cvar_95: float
    confidence_interval: Tuple[float, float]

class QuantileForecastEngine:
    def __init__(self, horizon: int = 20, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]):
        self.horizon = horizon
        self.quantiles = quantiles
        self.models = {}

    def train(self, X: pd.DataFrame, y: pd.Series):
        """X: features; y: forward return"""
        for q in self.quantiles:
            self.models[q] = lgb.LGBMRegressor(objective='quantile', alpha=q, n_estimators=100, verbose=-1)
            self.models[q].fit(X, y)

    def predict(self, X: pd.DataFrame) -> QuantileForecastResult:
        preds = {q: self.models[q].predict(X)[0] for q in self.quantiles}
        values = np.array(list(preds.values()))
        mean = float(np.mean(values))
        std = float(np.std(values))
        var_95 = -float(np.percentile(values, 5))
        tail = values[values <= np.percentile(values, 5)]
        cvar_95 = -float(np.mean(tail)) if len(tail) > 0 else var_95
        return QuantileForecastResult(
            horizon=self.horizon,
            quantiles=preds,
            mean=mean,
            std=std,
            var_95=var_95,
            cvar_95=cvar_95,
            confidence_interval=(preds[0.05], preds[0.95])
        )