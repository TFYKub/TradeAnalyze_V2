"""
V3 Google Sheets Writer – creates and populates new institutional sheets.
"""
import logging
from datetime import datetime
from config.config import SHEET_ID
from utils.sheets_auth import get_sheets_client

logger = logging.getLogger(__name__)

def create_v3_sheets():
    """Call this once during migration to create all V3 worksheets."""
    new_sheets = [
        "EnginePerformance",
        "QuantileForecast",
        "RegimeMC",
        "PortfolioAllocation",
        "CryptoFlowV2",
        "DealerGreeks",
        "StressTests"
    ]
    gc = get_sheets_client()
    sh = gc.open_by_key(SHEET_ID)
    for name in new_sheets:
        try:
            sh.worksheet(name)
            logger.info(f"Worksheet {name} already exists")
        except:
            sh.add_worksheet(title=name, rows=5000, cols=30)
            logger.info(f"Created worksheet {name}")

def log_quantile_forecast(symbol: str, horizon: int, quantiles: dict, mean: float, var_95: float):
    gc = get_sheets_client()
    ws = gc.open_by_key(SHEET_ID).worksheet("QuantileForecast")
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        symbol, horizon,
        quantiles.get(0.05, 0), quantiles.get(0.25, 0), quantiles.get(0.5, 0),
        quantiles.get(0.75, 0), quantiles.get(0.95, 0),
        mean, var_95
    ]
    ws.append_row(row, value_input_option="USER_ENTERED")

def log_regime_mc(symbol: str, mc_result: dict):
    gc = get_sheets_client()
    ws = gc.open_by_key(SHEET_ID).worksheet("RegimeMC")
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        symbol,
        mc_result.get('prob_profit', 0),
        mc_result.get('prob_stop_hit', 0),
        mc_result.get('expected_return', 0),
        mc_result.get('var_95', 0),
        mc_result.get('cvar_95', 0),
        str(mc_result.get('regime_weights', {}))
    ]
    ws.append_row(row, value_input_option="USER_ENTERED")

# Similar functions for other sheets...