"""
Market Breadth Engine  (Phase 14)
===================================
Measures broad market participation to confirm or challenge individual-asset signals.
Weak breadth in a rising asset = fragile move; strong breadth = healthy trend.

Crypto Breadth (3 signals):
  BTC Dominance      — BTC.D: rising dom = altcoins selling (RISK_OFF for alts)
  TOTAL3             — Total crypto minus BTC+ETH market cap proxy
  Stablecoin Supply Ratio (SSR) — high SSR = dry powder low = bearish

Equity Breadth (3 signals):
  Advance/Decline Ratio   — NYSE A/D using sector ETF basket
  New High / New Low      — 52-week extremes via ETF basket
  % Stocks Above 200 DMA  — breadth health indicator

Composite:
  breadth_score    : 0–100
  breadth_regime   : STRONG_BULL | BULL | NEUTRAL | BEAR | STRONG_BEAR
  breadth_confidence: 0–100

Integration:
  Regime Ensemble weights breadth at 10%
  High breadth boosts regime confidence; low breadth penalises it
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FETCH_PERIOD  = "1y"
LOOKBACK_DAYS = 20

# Equity breadth basket — liquid sector ETFs covering broad market
SECTOR_ETFS = [
    "XLK",   # Technology
    "XLF",   # Financials
    "XLE",   # Energy
    "XLV",   # Health Care
    "XLC",   # Communication
    "XLI",   # Industrials
    "XLY",   # Consumer Discretionary
    "XLP",   # Consumer Staples
    "XLU",   # Utilities
    "XLRE",  # Real Estate
]

# Crypto proxies available on yfinance
BTC_TICKER    = "BTC-USD"
ETH_TICKER    = "ETH-USD"
TOTAL3_PROXY  = ["SOL-USD", "BNB-USD", "ADA-USD", "AVAX-USD", "DOT-USD"]


@dataclass(frozen=True)
class CryptoBreadthResult:
    btc_dominance_est:   float          # estimated % (proxy)
    btc_dom_trend:       str            # RISING | FALLING | STABLE
    total3_change_pct:   float          # 20d change in altcoin basket
    total3_trend:        str
    ssr_proxy:           float          # stablecoin proxy score (USDT market cap proxy)
    crypto_breadth_score: float         # 0–100


@dataclass(frozen=True)
class EquityBreadthResult:
    advance_decline_ratio: float        # >1 = more advancers
    pct_above_200dma:      float        # % of ETF basket above 200 DMA
    new_high_count:        int          # ETFs at/near 52-week high
    new_low_count:         int          # ETFs at/near 52-week low
    equity_breadth_score:  float        # 0–100


@dataclass(frozen=True)
class MarketBreadthResult:
    breadth_score:       float          # 0–100 composite
    breadth_regime:      str            # STRONG_BULL | BULL | NEUTRAL | BEAR | STRONG_BEAR
    breadth_confidence:  float          # 0–100
    crypto_breadth:      CryptoBreadthResult
    equity_breadth:      EquityBreadthResult
    component_scores:    dict[str, float]
    interpretation:      str
    data_quality:        str            # FULL | PARTIAL | FALLBACK


# ── Fetch Helpers ──────────────────────────────────────────────────────────────
def _fetch(ticker: str, period: str = FETCH_PERIOD) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df if (df is not None and len(df) >= 30) else None
    except Exception as exc:
        logger.debug("[breadth] fetch %s: %s", ticker, exc)
        return None


def _fetch_multi(tickers: list[str]) -> dict[str, pd.DataFrame]:
    try:
        import yfinance as yf
        raw = yf.download(tickers, period=FETCH_PERIOD, interval="1d",
                          auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            return {}
        result = {}
        if isinstance(raw.columns, pd.MultiIndex):
            for t in tickers:
                try:
                    df = raw.xs(t, axis=1, level=1).dropna(how="all")
                    if len(df) >= 30:
                        result[t] = df
                except Exception:
                    pass
        else:
            # Single ticker returned flat
            result[tickers[0]] = raw
        return result
    except Exception as exc:
        logger.debug("[breadth] multi-fetch failed: %s", exc)
        return {}


# ── Crypto Breadth ─────────────────────────────────────────────────────────────
def _compute_crypto_breadth() -> CryptoBreadthResult:
    """
    Estimate crypto market breadth without direct CoinGecko dependency.
    Uses yfinance BTC, ETH, and TOTAL3 proxy tickers.
    """
    df_btc = _fetch(BTC_TICKER)
    alt_dfs = {t: _fetch(t) for t in TOTAL3_PROXY}
    alt_dfs = {k: v for k, v in alt_dfs.items() if v is not None}

    # ── BTC Dominance proxy ──────────────────────────────────────────────────
    btc_dom_est  = 52.0    # fallback estimate
    btc_dom_trend = "STABLE"
    if df_btc is not None and alt_dfs:
        # Approximate dominance via market cap proxy using price × rough supply
        # We instead measure BTC's momentum vs altcoin basket
        btc_20d  = float(df_btc["Close"].pct_change(20).iloc[-1] * 100) if len(df_btc) > 20 else 0.0
        alt_rets = []
        for df in alt_dfs.values():
            if len(df) > 20:
                r = float(df["Close"].pct_change(20).iloc[-1] * 100)
                alt_rets.append(r)
        if alt_rets:
            avg_alt = float(np.mean(alt_rets))
            dom_delta = btc_20d - avg_alt   # positive = BTC outperforming = dom rising
            btc_dom_est = 52.0 + dom_delta * 0.5  # rough proxy
            btc_dom_est = max(30.0, min(75.0, btc_dom_est))
            if dom_delta > 5:
                btc_dom_trend = "RISING"
            elif dom_delta < -5:
                btc_dom_trend = "FALLING"

    # ── TOTAL3 (altcoin momentum) ────────────────────────────────────────────
    total3_change = 0.0
    total3_trend  = "STABLE"
    if alt_dfs:
        recent_rets = []
        for df in alt_dfs.values():
            if len(df) > LOOKBACK_DAYS:
                r = float(df["Close"].pct_change(LOOKBACK_DAYS).iloc[-1] * 100)
                recent_rets.append(r)
        if recent_rets:
            total3_change = float(np.mean(recent_rets))
            total3_trend  = "RISING" if total3_change > 3 else "FALLING" if total3_change < -3 else "STABLE"

    # ── SSR Proxy — ratio of USDT (stablecoin) market cap vs BTC ────────────
    # Without direct on-chain data, use BTC volatility as inverse proxy:
    # High BTC vol = market uncertain = stablecoins in demand = bearish
    ssr_proxy = 50.0
    if df_btc is not None and len(df_btc) > 20:
        log_ret = np.log(df_btc["Close"] / df_btc["Close"].shift(1)).dropna()
        hv20    = float(log_ret.rolling(20).std().iloc[-1] * math.sqrt(252) * 100)
        # High vol = high SSR (dry powder used) = bearish
        ssr_proxy = max(10.0, min(90.0, 80.0 - hv20 * 0.8))

    # ── Crypto breadth score ─────────────────────────────────────────────────
    # BTC dominance: rising dom = bad for alts but crypto market may be rotating to BTC
    dom_score   = 50.0 - (btc_dom_est - 52.0) * 0.5    # rising dom → lower score for alts
    total3_score = 50.0 + total3_change * 1.5
    total3_score = max(0.0, min(100.0, total3_score))
    ssr_score    = ssr_proxy

    crypto_score = (dom_score * 0.35 + total3_score * 0.40 + ssr_score * 0.25)
    crypto_score = round(max(0.0, min(100.0, crypto_score)), 1)

    logger.debug(
        "[breadth:crypto] dom_est=%.1f dom_trend=%s total3=%+.1f%% ssr_proxy=%.1f score=%.1f",
        btc_dom_est, btc_dom_trend, total3_change, ssr_proxy, crypto_score,
    )

    return CryptoBreadthResult(
        btc_dominance_est    = round(btc_dom_est, 1),
        btc_dom_trend        = btc_dom_trend,
        total3_change_pct    = round(total3_change, 2),
        total3_trend         = total3_trend,
        ssr_proxy            = round(ssr_proxy, 1),
        crypto_breadth_score = crypto_score,
    )


# ── Equity Breadth ─────────────────────────────────────────────────────────────
def _compute_equity_breadth() -> EquityBreadthResult:
    """
    Measure equity market breadth using sector ETF basket.
    """
    etf_data = _fetch_multi(SECTOR_ETFS)

    if not etf_data:
        logger.warning("[breadth:equity] No ETF data — using neutral fallback")
        return EquityBreadthResult(
            advance_decline_ratio=1.0, pct_above_200dma=50.0,
            new_high_count=3, new_low_count=3,
            equity_breadth_score=50.0,
        )

    advances = 0
    declines = 0
    above_200 = 0
    new_highs = 0
    new_lows  = 0
    total     = len(etf_data)

    for ticker, df in etf_data.items():
        try:
            close   = df["Close"]
            last    = float(close.iloc[-1])
            prev    = float(close.iloc[-2]) if len(close) > 1 else last

            # A/D ratio
            if last > prev:
                advances += 1
            else:
                declines += 1

            # % above 200 DMA
            if len(close) >= 200:
                ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])
                if last > ema200:
                    above_200 += 1

            # New highs / lows (within 2% of 52-week high/low)
            if len(close) >= 252:
                high52 = float(close.rolling(252).max().iloc[-1])
                low52  = float(close.rolling(252).min().iloc[-1])
                if last >= high52 * 0.98:
                    new_highs += 1
                if last <= low52 * 1.02:
                    new_lows += 1
        except Exception:
            total -= 1   # exclude broken data from denominator

    if total == 0:
        total = 1

    ad_ratio       = round((advances + 0.5) / (declines + 0.5), 3)
    pct_above200   = round(above_200 / total * 100, 1)
    nh_score       = round(new_highs / total * 100, 1)
    nl_score       = round(new_lows  / total * 100, 1)

    # Score components (0–100)
    ad_score  = min(100.0, max(0.0, 50.0 + (ad_ratio - 1.0) * 25.0))
    pct_score = pct_above200
    nh_nl_score = max(0.0, min(100.0, 50.0 + (nh_score - nl_score) * 1.5))

    equity_score = round(
        ad_score    * 0.40
        + pct_score   * 0.40
        + nh_nl_score * 0.20,
        1,
    )

    logger.debug(
        "[breadth:equity] A/D=%.2f above200=%.1f%% NH=%d NL=%d score=%.1f",
        ad_ratio, pct_above200, new_highs, new_lows, equity_score,
    )

    return EquityBreadthResult(
        advance_decline_ratio = ad_ratio,
        pct_above_200dma      = pct_above200,
        new_high_count        = new_highs,
        new_low_count         = new_lows,
        equity_breadth_score  = equity_score,
    )


# ── Regime Classification ──────────────────────────────────────────────────────
def _classify_breadth(score: float) -> tuple[str, float]:
    if score >= 80.0:
        return "STRONG_BULL", round(min(92.0, 65.0 + (score - 80.0) * 1.5), 1)
    if score >= 62.0:
        return "BULL",        round(55.0 + (score - 62.0) * 0.8, 1)
    if score >= 42.0:
        return "NEUTRAL",     round(40.0 + (10.0 - abs(score - 52.0)) * 1.5, 1)
    if score >= 25.0:
        return "BEAR",        round(55.0 + (42.0 - score) * 0.8, 1)
    return "STRONG_BEAR",     round(min(92.0, 65.0 + (25.0 - score) * 1.5), 1)


# ── Main Entry ─────────────────────────────────────────────────────────────────
def compute_market_breadth() -> MarketBreadthResult:
    """
    Compute global market breadth (crypto + equity).

    Returns MarketBreadthResult — always returns, never raises.
    """
    crypto_breadth = CryptoBreadthResult(
        btc_dominance_est=52.0, btc_dom_trend="STABLE",
        total3_change_pct=0.0, total3_trend="STABLE",
        ssr_proxy=50.0, crypto_breadth_score=50.0,
    )
    equity_breadth = EquityBreadthResult(
        advance_decline_ratio=1.0, pct_above_200dma=50.0,
        new_high_count=3, new_low_count=3, equity_breadth_score=50.0,
    )
    data_quality = "FULL"

    try:
        crypto_breadth = _compute_crypto_breadth()
    except Exception as exc:
        logger.warning("[breadth] crypto breadth failed: %s", exc)
        data_quality = "PARTIAL"

    try:
        equity_breadth = _compute_equity_breadth()
    except Exception as exc:
        logger.warning("[breadth] equity breadth failed: %s", exc)
        data_quality = "PARTIAL" if data_quality == "FULL" else "FALLBACK"

    # ── Composite score ──────────────────────────────────────────────────────
    crypto_score = crypto_breadth.crypto_breadth_score
    equity_score = equity_breadth.equity_breadth_score

    # Weight equity breadth slightly higher (more liquid, more data)
    composite = round(crypto_score * 0.40 + equity_score * 0.60, 1)

    regime, confidence = _classify_breadth(composite)

    regime_emoji = {
        "STRONG_BULL": "🚀", "BULL": "📈",
        "NEUTRAL": "↔️",
        "BEAR": "📉",   "STRONG_BEAR": "🔻",
    }.get(regime, "❓")

    interp = (
        f"{regime_emoji} {regime} breadth (score={composite:.0f}/100) — "
        f"Crypto={crypto_score:.0f} | Equity={equity_score:.0f} | "
        f"BTC.D trend={crypto_breadth.btc_dom_trend} | "
        f"A/D={equity_breadth.advance_decline_ratio:.2f} | "
        f"Above200={equity_breadth.pct_above_200dma:.0f}%"
    )

    component_scores = {
        "crypto_breadth": crypto_score,
        "equity_breadth": equity_score,
        "btc_dominance":  round(50.0 - (crypto_breadth.btc_dominance_est - 52.0) * 0.5, 1),
        "total3_momentum":round(50.0 + crypto_breadth.total3_change_pct * 1.5, 1),
        "advance_decline":round(min(100.0, max(0.0, 50.0 + (equity_breadth.advance_decline_ratio - 1.0) * 25.0)), 1),
        "pct_above_200dma":equity_breadth.pct_above_200dma,
    }

    logger.info(
        "[breadth] regime=%s score=%.1f conf=%.1f crypto=%.1f equity=%.1f quality=%s",
        regime, composite, confidence, crypto_score, equity_score, data_quality,
    )

    return MarketBreadthResult(
        breadth_score      = composite,
        breadth_regime     = regime,
        breadth_confidence = confidence,
        crypto_breadth     = crypto_breadth,
        equity_breadth     = equity_breadth,
        component_scores   = component_scores,
        interpretation     = interp,
        data_quality       = data_quality,
    )
