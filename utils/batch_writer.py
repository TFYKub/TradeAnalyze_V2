"""
Batch Google Sheets Writer – Collects rows per sheet and writes once per run.
"""
import logging
from typing import Dict, List, Any

from utils.sheets_auth import get_sheets_client
from config.config import SHEET_ID

logger = logging.getLogger(__name__)


class BatchWriter:
    """Singleton batch writer for Google Sheets."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._buffers: Dict[str, List[List[Any]]] = {}
        self._headers: Dict[str, List[str]] = {}
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = get_sheets_client()
        return self._client

    def add_row(self, sheet_name: str, row: List[Any], headers: List[str]):
        """Add a row to the buffer for the given sheet."""
        if sheet_name not in self._buffers:
            self._buffers[sheet_name] = []
            self._headers[sheet_name] = headers
        self._buffers[sheet_name].append(row)

    def flush(self):
        """Write all buffered rows to Google Sheets."""
        if not self._buffers:
            logger.info("[BatchWriter] No buffered rows to flush.")
            return

        gc = self._get_client()
        sh = gc.open_by_key(SHEET_ID)

        for sheet_name, rows in self._buffers.items():
            try:
                ws = sh.worksheet(sheet_name)
                # Ensure headers
                first_row = ws.row_values(1)
                if not first_row or first_row[0] != self._headers[sheet_name][0]:
                    ws.insert_row(self._headers[sheet_name], index=1)
                # Append rows in one batch
                if rows:
                    ws.append_rows(rows, value_input_option="USER_ENTERED")
                    logger.info(f"[BatchWriter] Flushed {len(rows)} rows to {sheet_name}")
            except Exception as e:
                logger.error(f"[BatchWriter] Failed to flush {sheet_name}: {e}")

        # Clear buffers after flush
        self._buffers.clear()
        self._headers.clear()

    def clear(self):
        """Clear buffers without writing."""
        self._buffers.clear()
        self._headers.clear()


# Global singleton instance
_batch_writer = None


def get_batch_writer() -> BatchWriter:
    global _batch_writer
    if _batch_writer is None:
        _batch_writer = BatchWriter()
    return _batch_writer