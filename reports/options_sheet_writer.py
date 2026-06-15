# reports/options_sheet_writer.py – Batched version
import logging
import math
from datetime import datetime

from config.config import SHEET_ID
from utils.sheets_auth import get_sheets_client
from utils.batch_writer import get_batch_writer

logger = logging.getLogger(__name__)

SHEET_NAME = "Options_Analysis"
HEADERS = [
    "Timestamp", "Symbol", "Regime", "RegimeConf", "BullProb", "BearProb",
    "IV", "IVRank", "IVPct", "HV20", "ATR14", "IVSource",
    "ExpMove", "ExpMovePct", "DTE", "Rule",
    "Strategy", "Score", "POP", "EV", "RR", "Kelly", "HalfKelly",
    "Strikes", "MaxProfit", "MaxLoss", "Breakevens", "Rationale",
    "Alt1", "Alt1Score", "Alt2", "Alt2Score",
    "Approved", "AIScore",
]


def _safe(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, list):
        return str(v)
    if isinstance(v, float):
        return "" if (math.isnan(v) or math.isinf(v)) else round(v, 5)
    return v


def _ensure_worksheet(ws_name: str):
    gc = get_sheets_client()
    sh = gc.open_by_key(SHEET_ID)
    try:
        ws = sh.worksheet(ws_name)
    except Exception:
        ws = sh.add_worksheet(title=ws_name, rows=5000, cols=len(HEADERS))
        logger.info(f"[options_sheet_writer] Created worksheet '{ws_name}'")
        ws.update('A1', [HEADERS], value_input_option='USER_ENTERED')
    return ws


def _ensure_headers(ws):
    try:
        first_row = ws.row_values(1)
        if not first_row or first_row[0] != HEADERS[0]:
            ws.insert_row(HEADERS, index=1)
            logger.info(f"[options_sheet_writer] Headers inserted for {SHEET_NAME}")
    except Exception as exc:
        logger.warning(f"[options_sheet_writer] Header check failed: {exc}")


def write_options_analysis(rec) -> None:
    pri = rec.primary
    vol = rec.vol
    em = rec.em
    ranking = rec.ranking
    top3 = ranking.top_strategies
    alt1 = top3[1] if len(top3) > 1 else None
    alt2 = top3[2] if len(top3) > 2 else None

    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        _safe(rec.symbol), _safe(rec.regime), _safe(round(rec.regime_conf, 1)),
        _safe(round(rec.bull_prob, 3)), _safe(round(rec.bear_prob, 3)),
        _safe(round(vol.iv, 4)), _safe(round(vol.iv_rank, 1)), _safe(round(vol.iv_percentile, 1)),
        _safe(round(vol.hv20, 4)), _safe(round(vol.atr14, 4)), _safe(vol.source),
        _safe(round(em.expected_move, 2)), _safe(round(em.expected_move_pct, 2)), _safe(em.dte),
        _safe(ranking.rule_triggered),
        _safe(pri.name), _safe(round(pri.score, 1)), _safe(round(pri.pop, 1)),
        _safe(round(pri.ev, 2)), _safe(round(pri.rr, 2)),
        _safe(round(pri.kelly, 4)), _safe(round(pri.half_kelly, 4)),
        _safe(pri.strike_summary), _safe(pri.max_profit), _safe(pri.max_loss),
        _safe(str(pri.breakevens)), _safe(pri.rationale[:80]),
        _safe(alt1.name if alt1 else ""), _safe(round(alt1.score, 1) if alt1 else ""),
        _safe(alt2.name if alt2 else ""), _safe(round(alt2.score, 1) if alt2 else ""),
        _safe(rec.trade_approved), _safe(round(rec.ai_score, 1)),
    ]

    try:
        get_batch_writer().add_row(SHEET_NAME, row, HEADERS)
        logger.info(f"Options_Analysis ← {rec.symbol} (batched)")
    except Exception as exc:
        logger.error(f"Options_Analysis add failed ({rec.symbol}): {exc}")