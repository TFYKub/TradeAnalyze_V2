"""
Crypto Funding Rate Engine  (Phase 7)
=======================================
Fetches perpetual funding rates from Deribit and OKX public APIs.
Interprets funding regime to detect crowded longs/shorts.

Funding Rate Logic:
  Positive rate → longs pay shorts → market biased LONG (crowded longs)
  Negative rate → shorts pay longs → market biased SHORT (crowded shorts)

Extreme positive (> +0.10%)  → CROWDED_LONG  → contrarian SHORT signal
Extreme negative (< -0.05%)  → CROWDED_SHORT → contrarian LONG signal
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

DERIBIT_BASE = "https://www.deribit.com/api/v2/public"
OKX_BASE     = "https://www.okx.com/api/v5/public"
TIMEOUT      = 10

# Thresholds
CROWDED_LONG_THRESHOLD  =  0.0010   # +0.10% per 8h = very crowded longs
CROWDED_SHORT_THRESHOLD = -0.0005   # -0.05% per 8h = crowded shorts
NORMAL_THRESHOLD        =  0.0003   # ±0.03% = normal range


@dataclass(frozen=True)
class FundingRateResult:
    symbol:          str
    funding_rate:    float           # current 8h funding rate (decimal)
    funding_rate_pct: float          # percentage
    predicted_rate:  float | None    # next period predicted rate
    source:          str             # "deribit" | "okx" | "unavailable"
    funding_regime:  str             # CROWDED_LONG | HIGH_LONG | NEUTRAL | HIGH_SHORT | CROWDED_SHORT
    crowded_long:    bool
    crowded_short:   bool
    contrarian_signal: str           # "SHORT_BIAS" | "LONG_BIAS" | "NEUTRAL"
    interpretation:  str


def _classify_funding(rate: float) -> tuple[str, bool, bool, str]:
    if rate >= CROWDED_LONG_THRESHOLD:
        return "CROWDED_LONG", True, False, "SHORT_BIAS"
    if rate >= NORMAL_THRESHOLD:
        return "HIGH_LONG", False, False, "NEUTRAL"
    if rate <= CROWDED_SHORT_THRESHOLD:
        return "CROWDED_SHORT", False, True, "LONG_BIAS"
    if rate <= -NORMAL_THRESHOLD:
        return "HIGH_SHORT", False, False, "NEUTRAL"
    return "NEUTRAL", False, False, "NEUTRAL"


def _fetch_deribit_funding(symbol: str) -> dict | None:
    """Fetch from Deribit perpetual ticker."""
    currency = symbol.upper().split("-")[0].split("/")[0]
    instrument = f"{currency}-PERPETUAL"
    try:
        r = requests.get(
            f"{DERIBIT_BASE}/ticker",
            params={"instrument_name": instrument},
            timeout=TIMEOUT,
            headers={"User-Agent": "TradeAnalyze/1.0"},
        )
        r.raise_for_status()
        data = r.json().get("result", {})
        rate = data.get("current_funding")
        pred = data.get("interest_1h")
        if rate is not None:
            return {"rate": float(rate), "predicted": float(pred) if pred else None, "source": "deribit"}
    except Exception as exc:
        logger.debug("Deribit funding fetch failed: %s", exc)
    return None


def _fetch_okx_funding(symbol: str) -> dict | None:
    """Fetch from OKX perpetual swap."""
    currency = symbol.upper().split("-")[0].split("/")[0]
    inst_id  = f"{currency}-USDT-SWAP"
    try:
        r = requests.get(
            f"{OKX_BASE}/funding-rate",
            params={"instId": inst_id},
            timeout=TIMEOUT,
            headers={"User-Agent": "TradeAnalyze/1.0"},
        )
        r.raise_for_status()
        items = r.json().get("data", [])
        if items:
            rate = float(items[0].get("fundingRate", 0))
            pred = items[0].get("nextFundingRate")
            return {"rate": rate, "predicted": float(pred) if pred else None, "source": "okx"}
    except Exception as exc:
        logger.debug("OKX funding fetch failed: %s", exc)
    return None


def fetch_funding_rate(symbol: str) -> FundingRateResult:
    """
    Fetch funding rate from Deribit → OKX → fallback(0).

    Parameters
    ----------
    symbol : e.g. "BTC", "BTC-USD", "ETH"
    """
    data = _fetch_deribit_funding(symbol) or _fetch_okx_funding(symbol)

    if data is None:
        logger.warning("[funding] No data for %s — using zero fallback", symbol)
        data = {"rate": 0.0, "predicted": None, "source": "unavailable"}

    rate    = float(data["rate"])
    regime, crowded_long, crowded_short, contrarian = _classify_funding(rate)

    interp_map = {
        "CROWDED_LONG":  f"⚠️ Crowded longs ({rate*100:+.4f}%) → contrarian SHORT bias",
        "HIGH_LONG":     f"Elevated longs ({rate*100:+.4f}%) → slight bearish pressure",
        "NEUTRAL":       f"Neutral funding ({rate*100:+.4f}%) → no crowding signal",
        "HIGH_SHORT":    f"Elevated shorts ({rate*100:+.4f}%) → slight bullish pressure",
        "CROWDED_SHORT": f"⚠️ Crowded shorts ({rate*100:+.4f}%) → contrarian LONG bias",
    }

    logger.info("[funding] %s rate=%+.5f regime=%s source=%s",
                symbol, rate, regime, data["source"])

    return FundingRateResult(
        symbol           = symbol,
        funding_rate     = round(rate, 6),
        funding_rate_pct = round(rate * 100, 5),
        predicted_rate   = data.get("predicted"),
        source           = data["source"],
        funding_regime   = regime,
        crowded_long     = crowded_long,
        crowded_short    = crowded_short,
        contrarian_signal= contrarian,
        interpretation   = interp_map.get(regime, ""),
    )
