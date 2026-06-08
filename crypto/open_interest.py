"""
Open Interest Engine  (Phase 7)
=================================
Fetches OI from Deribit / OKX and interprets price × OI divergence.

Price ↑ + OI ↑ → New longs entering    → trend CONFIRMATION (bullish)
Price ↓ + OI ↑ → New shorts entering   → trend CONTINUATION (bearish)
Price ↑ + OI ↓ → Short covering rally  → WEAK / unsustained move
Price ↓ + OI ↓ → Long liquidation      → CAPITULATION (may be near bottom)
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
import requests

logger = logging.getLogger(__name__)

DERIBIT_BASE = "https://www.deribit.com/api/v2/public"
OKX_BASE     = "https://www.okx.com/api/v5/public"
TIMEOUT      = 10


@dataclass(frozen=True)
class OpenInterestResult:
    symbol:         str
    open_interest:  float          # in USD or contracts
    oi_change_pct:  float | None   # 24h change %
    source:         str
    oi_trend:       str            # INCREASING | DECREASING | STABLE
    price_oi_signal: str           # CONFIRMATION | CONTINUATION | WEAK_RALLY | CAPITULATION | UNKNOWN
    signal_strength: str           # STRONG | MODERATE | WEAK
    interpretation: str


def _fetch_deribit_oi(symbol: str) -> dict | None:
    currency   = symbol.upper().split("-")[0].split("/")[0]
    instrument = f"{currency}-PERPETUAL"
    try:
        r = requests.get(f"{DERIBIT_BASE}/ticker",
                         params={"instrument_name": instrument},
                         timeout=TIMEOUT, headers={"User-Agent": "TradeAnalyze/1.0"})
        r.raise_for_status()
        d   = r.json().get("result", {})
        oi  = d.get("open_interest")
        if oi is not None:
            return {"oi": float(oi), "source": "deribit"}
    except Exception as exc:
        logger.debug("Deribit OI failed: %s", exc)
    return None


def _fetch_okx_oi(symbol: str) -> dict | None:
    currency = symbol.upper().split("-")[0].split("/")[0]
    inst_id  = f"{currency}-USDT-SWAP"
    try:
        r = requests.get(f"{OKX_BASE}/open-interest",
                         params={"instId": inst_id},
                         timeout=TIMEOUT, headers={"User-Agent": "TradeAnalyze/1.0"})
        r.raise_for_status()
        items = r.json().get("data", [])
        if items:
            oi = float(items[0].get("oi", 0) or items[0].get("oiCcy", 0))
            return {"oi": oi, "source": "okx"}
    except Exception as exc:
        logger.debug("OKX OI failed: %s", exc)
    return None


def _classify_price_oi(
    price_change: float, oi_change: float
) -> tuple[str, str, str]:
    """
    Returns (signal, strength, interpretation).
    price_change and oi_change are % values.
    """
    if price_change > 0 and oi_change > 0:
        sig   = "CONFIRMATION"
        strg  = "STRONG" if price_change > 2 and oi_change > 5 else "MODERATE"
        interp = f"Price↑({price_change:+.1f}%) + OI↑({oi_change:+.1f}%) → New longs entering — bullish confirmation"
    elif price_change < 0 and oi_change > 0:
        sig   = "CONTINUATION"
        strg  = "STRONG" if abs(price_change) > 2 and oi_change > 5 else "MODERATE"
        interp = f"Price↓({price_change:+.1f}%) + OI↑({oi_change:+.1f}%) → New shorts entering — bearish continuation"
    elif price_change > 0 and oi_change < 0:
        sig   = "WEAK_RALLY"
        strg  = "WEAK"
        interp = f"Price↑({price_change:+.1f}%) + OI↓({oi_change:+.1f}%) → Short covering rally — unsustained move"
    elif price_change < 0 and oi_change < 0:
        sig   = "CAPITULATION"
        strg  = "STRONG" if abs(price_change) > 3 and abs(oi_change) > 5 else "MODERATE"
        interp = f"Price↓({price_change:+.1f}%) + OI↓({oi_change:+.1f}%) → Long liquidation — potential capitulation"
    else:
        sig   = "UNKNOWN"
        strg  = "WEAK"
        interp = f"Price({price_change:+.1f}%) + OI({oi_change:+.1f}%) — insufficient signal"

    return sig, strg, interp


def fetch_open_interest(
    symbol:       str,
    price_change: float = 0.0,   # 24h price change %
) -> OpenInterestResult:
    """
    Fetch OI and interpret price × OI divergence.

    Parameters
    ----------
    symbol       : e.g. "BTC", "ETH"
    price_change : 24h price change % (pass from market data)
    """
    data = _fetch_deribit_oi(symbol) or _fetch_okx_oi(symbol)

    if data is None:
        data = {"oi": 0.0, "source": "unavailable"}

    oi       = float(data["oi"])
    # OI change: unavailable without historical snapshot, default None
    oi_change = None

    oi_trend = "STABLE"
    signal   = "UNKNOWN"
    strength = "WEAK"
    interp   = "OI data unavailable — cannot classify"

    if oi_change is not None:
        oi_trend = "INCREASING" if oi_change > 2 else "DECREASING" if oi_change < -2 else "STABLE"
        signal, strength, interp = _classify_price_oi(price_change, oi_change)
    elif oi > 0 and price_change != 0:
        # Use price change alone as proxy
        interp = f"OI={oi:,.0f} (no 24h delta). Price change: {price_change:+.1f}%"

    logger.info("[oi] %s oi=%.0f trend=%s signal=%s source=%s",
                symbol, oi, oi_trend, signal, data["source"])

    return OpenInterestResult(
        symbol          = symbol,
        open_interest   = round(oi, 2),
        oi_change_pct   = oi_change,
        source          = data["source"],
        oi_trend        = oi_trend,
        price_oi_signal = signal,
        signal_strength = strength,
        interpretation  = interp,
    )
