"""
Option Chain Sheet Writer – Batched version
"""
import logging
import math
from datetime import datetime

from config.config import SHEET_ID
from utils.sheets_auth import get_sheets_client
from utils.batch_writer import get_batch_writer

logger = logging.getLogger(__name__)

SHEET_NAME = "Option_Chain"

HEADERS = [
    "Timestamp", "Symbol", "Source",
    "Expiry", "DTE", "DTE_Bucket", "Type",
    "Strike", "Bid", "Ask", "Mid", "Last",
    "IV", "Volume", "OI", "ITM",
    "Delta", "Gamma", "Theta", "Vega", "Rho",
    "Moneyness", "High_Gamma",
    "Theta_Category", "Vega_Category", "Direction_Bias",
]

_SYMBOL_COL_IDX = 1


def _safe(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return ""
        return round(v, 6)
    return v


def _row_from(ts: str, symbol: str, r: dict) -> list:
    return [
        ts,
        symbol,
        _safe(r.get("source")),
        _safe(r.get("expiry")),
        _safe(r.get("dte")),
        _safe(r.get("dte_bucket")),
        _safe(r.get("option_type")),
        _safe(r.get("strike")),
        _safe(r.get("bid")),
        _safe(r.get("ask")),
        _safe(r.get("mid")),
        _safe(r.get("last")),
        _safe(r.get("iv")),
        _safe(r.get("volume")),
        _safe(r.get("open_interest")),
        _safe(r.get("in_the_money")),
        _safe(r.get("delta")),
        _safe(r.get("gamma")),
        _safe(r.get("theta")),
        _safe(r.get("vega")),
        _safe(r.get("rho")),
        _safe(r.get("moneyness")),
        _safe(r.get("high_gamma")),
        _safe(r.get("theta_category")),
        _safe(r.get("vega_category")),
        _safe(r.get("direction_bias")),
    ]


def _ensure_headers(ws) -> None:
    try:
        first = ws.row_values(1)
        if not first or first[0] != "Timestamp":
            ws.insert_row(HEADERS, index=1)
            logger.info("[option_chain_writer] Header row inserted")
    except Exception as exc:
        logger.warning(f"[option_chain_writer] header check failed: {exc}")


def write_option_chain(symbol: str, enriched_rows: list[dict]) -> int:
    if not enriched_rows:
        logger.info(f"[option_chain_writer] {symbol}: no rows to write")
        return 0

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        for r in enriched_rows:
            row = _row_from(ts, symbol, r)
            get_batch_writer().add_row(SHEET_NAME, row, HEADERS)
        logger.info(f"[option_chain_writer] {symbol}: added {len(enriched_rows)} rows to batch")
        return len(enriched_rows)
    except Exception as exc:
        logger.error(f"[option_chain_writer] {symbol}: add failed — {exc}")
        return 0


def clear_symbol_rows(symbol: str) -> int:
    gc = get_sheets_client()
    ws = gc.open_by_key(SHEET_ID).worksheet(SHEET_NAME)

    try:
        all_values = ws.get_all_values()
    except Exception as exc:
        logger.warning(f"[option_chain_writer] clear: read failed — {exc}")
        return 0

    to_delete = [
        i + 1
        for i, row in enumerate(all_values)
        if i > 0
        and len(row) > _SYMBOL_COL_IDX
        and row[_SYMBOL_COL_IDX] == symbol
    ]

    if not to_delete:
        return 0

    deleted = 0
    for row_idx in reversed(to_delete):
        try:
            ws.delete_rows(row_idx)
            deleted += 1
        except Exception as exc:
            logger.debug(f"[option_chain_writer] delete row {row_idx}: {exc}")

    logger.info(f"[option_chain_writer] {symbol}: cleared {deleted} stale rows")
    return deleted


def overwrite_all_symbols(all_enriched: dict[str, list[dict]]) -> int:
    if not all_enriched:
        return 0

    gc = get_sheets_client()
    ws = gc.open_by_key(SHEET_ID).worksheet(SHEET_NAME)
    _ensure_headers(ws)

    try:
        last_row = ws.row_count
        if last_row > 1:
            ws.delete_rows(2, last_row)
    except Exception as exc:
        logger.warning(f"[option_chain_writer] bulk clear failed: {exc}")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    all_rows: list[list] = []

    for symbol, enriched_rows in all_enriched.items():
        for r in enriched_rows:
            all_rows.append(_row_from(ts, symbol, r))

    if not all_rows:
        return 0

    ws.append_rows(all_rows, value_input_option="USER_ENTERED")
    logger.info(f"[option_chain_writer] bulk write: {len(all_rows)} total rows")
    return len(all_rows)