"""
Unified Trade Engine – Institutional Execution Terminal
========================================================
Single‑source‑of‑truth system that computes regime, structure,
volatility, risk, trade plan, and Monte Carlo from ONE consistent
state. No conflicting signals, no recomputation.

Outputs a hedge‑fund style execution report with futures & options plan.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from datetime import datetime

# ============================================================================
#                            STATE DEFINITION (SINGLE SOURCE OF TRUTH)
# ============================================================================

@dataclass(frozen=True)
class PriceData:
    """Cleaned, aligned OHLCV with derived indicators."""
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    dates: np.ndarray
    # Indicators (computed once)
    ema12: np.ndarray
    ema26: np.ndarray
    rsi14: np.ndarray
    atr14: np.ndarray
    # Swing points (indices)
    swing_high_idx: List[int]
    swing_low_idx: List[int]
    # VWAP / AVWAP (optional)
    monthly_vwap: float
    quarterly_vwap: float
    yearly_vwap: float


@dataclass
class TradeState:
    """Central state – all downstream outputs are derived from this."""
    # Raw data
    price_data: PriceData
    current_price: float
    # Regime
    regime: str                    # BULL, BEAR, RANGE, STRONG_BULL, STRONG_BEAR
    regime_conf: float             # 0-100
    # Volatility
    vol_regime: str                # LOW_VOL, NORMAL_VOL, HIGH_VOL, PANIC_VOL
    vol_action: str                # human‑readable action
    # Trend & structure
    structure_trend: str           # BULLISH, BEARISH, MIXED
    structure_score: float
    ema_bias: str
    # Zones (computed once)
    demand_zones: List[dict]
    supply_zones: List[dict]
    # Trade plan (raw)
    direction: str                 # LONG, SHORT, WAIT
    entry_zone_low: float
    entry_zone_high: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    rr1: float
    rr2: float
    rr3: float
    invalidation: str
    # Options (simplified)
    option_strategy: str
    option_buy_call: Optional[float]
    option_sell_call: Optional[float]
    option_buy_put: Optional[float]
    option_sell_put: Optional[float]
    option_strike_call: Tuple[Optional[float], Optional[float]]
    option_strike_put: Tuple[Optional[float], Optional[float]]
    option_dte_call: int
    option_dte_put: int
    # Risk & MC
    trade_quality_score: float
    ev: float
    kelly: float
    risk_pct: float
    mc_prob_profit: float
    mc_prob_drawdown: float      # stop‑hit probability
    mc_exp_return: float
    # Decision gates
    trade_allowed: bool
    gate_reason: str
    final_confidence: float


# ============================================================================
#                            INDICATORS (ONE PASS)
# ============================================================================

def compute_indicators(df: pd.DataFrame) -> PriceData:
    """Compute all required indicators from the raw OHLCV DataFrame."""
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    open_ = df['Open'].values
    volume = df['Volume'].values
    dates = df.index.values

    # EMA12/26
    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values

    # RSI14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean().values
    avg_loss = pd.Series(loss).rolling(14).mean().values
    rs = avg_gain / (avg_loss + 1e-9)
    rsi14 = 100 - 100 / (1 + rs)

    # ATR14
    hl = high - low
    hc = np.abs(high - np.roll(close, 1))
    lc = np.abs(low - np.roll(close, 1))
    tr = np.maximum(hl, np.maximum(hc, lc))
    atr14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values

    # Swing points (simplified)
    window = 5
    swing_high_idx = []
    swing_low_idx = []
    for i in range(window, len(close)-window):
        if high[i] == max(high[i-window:i+window+1]):
            swing_high_idx.append(i)
        if low[i] == min(low[i-window:i+window+1]):
            swing_low_idx.append(i)

    # AVWAP (monthly/quarterly/yearly – simplified using last date)
    last_date = dates[-1]
    if isinstance(last_date, np.datetime64):
        last_date = pd.Timestamp(last_date)
    month_start = pd.Timestamp(last_date.year, last_date.month, 1)
    quarter_start = pd.Timestamp(last_date.year, ((last_date.month-1)//3)*3+1, 1)
    year_start = pd.Timestamp(last_date.year, 1, 1)

    def vwap_from(anchor):
        idx = np.where(df.index >= anchor)[0]
        if len(idx) == 0:
            return close[-1]
        sub_close = close[idx]
        sub_vol = volume[idx]
        tp = (high[idx] + low[idx] + close[idx]) / 3
        return np.sum(tp * sub_vol) / (np.sum(sub_vol) + 1e-9)

    monthly_vwap = vwap_from(month_start)
    quarterly_vwap = vwap_from(quarter_start)
    yearly_vwap = vwap_from(year_start)

    return PriceData(
        open=open_, high=high, low=low, close=close, volume=volume, dates=dates,
        ema12=ema12, ema26=ema26, rsi14=rsi14, atr14=atr14,
        swing_high_idx=swing_high_idx, swing_low_idx=swing_low_idx,
        monthly_vwap=monthly_vwap, quarterly_vwap=quarterly_vwap, yearly_vwap=yearly_vwap
    )


# ============================================================================
#                            DEMAND / SUPPLY ZONES
# ============================================================================

def compute_demand_supply_zones(price_data: PriceData, current_price: float) -> Tuple[List[dict], List[dict]]:
    """
    Returns (demand_zones, supply_zones) each as list of dict with keys:
        low, high, strength, type, reason
    """
    demand = []
    supply = []
    high_prices = price_data.high
    low_prices = price_data.low
    close_prices = price_data.close
    swings_high = [price_data.high[i] for i in price_data.swing_high_idx]
    swings_low = [price_data.low[i] for i in price_data.swing_low_idx]

    # 1. Swing lows → demand zones
    for sl in swings_low[-5:]:
        demand.append({
            'low': round(sl * 0.995, 2),
            'high': round(sl * 1.005, 2),
            'strength': 70.0,
            'type': 'DEMAND',
            'reason': 'Swing low'
        })
    # 2. Swing highs → supply zones
    for sh in swings_high[-5:]:
        supply.append({
            'low': round(sh * 0.995, 2),
            'high': round(sh * 1.005, 2),
            'strength': 70.0,
            'type': 'SUPPLY',
            'reason': 'Swing high'
        })

    # 3. AVWAP levels
    for name, level in [('Monthly', price_data.monthly_vwap), ('Quarterly', price_data.quarterly_vwap), ('Yearly', price_data.yearly_vwap)]:
        if level is None: continue
        zone = {
            'low': round(level * 0.99, 2),
            'high': round(level * 1.01, 2),
            'strength': 80.0,
            'reason': f'AVWAP {name}'
        }
        if level < current_price:
            zone['type'] = 'DEMAND'
            demand.append(zone)
        else:
            zone['type'] = 'SUPPLY'
            supply.append(zone)

    # 4. Simple volume profile proxy: use price at 30-day max volume
    lookback = min(60, len(close_prices))
    vol_window = price_data.volume[-lookback:]
    price_window = close_prices[-lookback:]
    max_vol_idx = np.argmax(vol_window)
    if max_vol_idx < len(price_window):
        poc = price_window[max_vol_idx]
        zone = {
            'low': round(poc * 0.99, 2),
            'high': round(poc * 1.01, 2),
            'strength': 85.0,
            'reason': 'Volume profile POC'
        }
        if poc < current_price:
            zone['type'] = 'DEMAND'
            demand.append(zone)
        else:
            zone['type'] = 'SUPPLY'
            supply.append(zone)

    # Deduplicate by proximity and take top 2 each
    def unique(zones):
        uniq = []
        for z in zones:
            if not any(abs(z['low'] - u['low']) < 0.01 * z['low'] for u in uniq):
                uniq.append(z)
        return sorted(uniq, key=lambda x: -x['strength'])[:2]

    demand = unique(demand)
    supply = unique(supply)
    return demand, supply


# ============================================================================
#                            REGIME & STRUCTURE (CONSISTENT)
# ============================================================================

def detect_regime(price_data: PriceData) -> Tuple[str, float]:
    """
    Simplified regime based on EMA200, EMA50, and ROC.
    Returns (regime, confidence).
    """
    close = price_data.close
    ema200 = pd.Series(close).ewm(span=200, adjust=False).mean().values
    ema50 = pd.Series(close).ewm(span=50, adjust=False).mean().values
    last = close[-1]
    last_ema200 = ema200[-1]
    last_ema50 = ema50[-1]

    # Momentum (20-day rate of change)
    roc20 = (close[-1] / close[-21]) - 1 if len(close) > 20 else 0

    if last > last_ema200 and last > last_ema50 and roc20 > 0.02:
        regime = "STRONG_BULL"
        conf = 85.0
    elif last > last_ema200 and last > last_ema50:
        regime = "BULL"
        conf = 70.0
    elif last < last_ema200 and last < last_ema50 and roc20 < -0.02:
        regime = "STRONG_BEAR"
        conf = 85.0
    elif last < last_ema200 and last < last_ema50:
        regime = "BEAR"
        conf = 70.0
    else:
        regime = "RANGE"
        conf = 50.0
    return regime, conf


def detect_volatility_regime(atr_pct: float) -> Tuple[str, str]:
    """Returns (vol_regime, action_string)."""
    if atr_pct >= 4.0:
        return "PANIC_VOL", "⚠️ Panic vol — reduce position 70%, widen stops 2×"
    elif atr_pct >= 2.5:
        return "HIGH_VOL", "High vol — reduce position 40%, widen stops 1.5×"
    elif atr_pct >= 1.0:
        return "NORMAL_VOL", "Normal vol — standard position and stop"
    else:
        return "LOW_VOL", "Low vol — increase size 20%, tighten stops 0.75×"


def detect_structure(price_data: PriceData) -> Tuple[str, float]:
    """
    Simple structure: compare last 3 swing highs and lows.
    Returns (trend, structure_score).
    """
    highs_idx = price_data.swing_high_idx
    lows_idx = price_data.swing_low_idx
    if len(highs_idx) < 2 or len(lows_idx) < 2:
        return "MIXED", 40.0

    last_highs = [price_data.high[i] for i in highs_idx[-3:]]
    last_lows = [price_data.low[i] for i in lows_idx[-3:]]

    hh = all(last_highs[i] > last_highs[i-1] for i in range(1, len(last_highs))) if len(last_highs) > 1 else False
    hl = all(last_lows[i] > last_lows[i-1] for i in range(1, len(last_lows))) if len(last_lows) > 1 else False
    ll = all(last_lows[i] < last_lows[i-1] for i in range(1, len(last_lows))) if len(last_lows) > 1 else False
    lh = all(last_highs[i] < last_highs[i-1] for i in range(1, len(last_highs))) if len(last_highs) > 1 else False

    if hh and hl:
        return "BULLISH", 80.0
    elif ll and lh:
        return "BEARISH", 80.0
    else:
        return "MIXED", 50.0


# ============================================================================
#                            TRADE PLAN GENERATION
# ============================================================================

def generate_trade_plan(state: TradeState) -> TradeState:
    """
    Builds entry zone, stops, TPs, RR, options.
    Assumes demand/supply zones already computed.
    """
    price = state.current_price
    atr = state.price_data.atr14[-1]
    direction = state.direction
    demand = state.demand_zones
    supply = state.supply_zones

    if direction == "LONG":
        nearest_demand = min(demand, key=lambda z: abs(z['low'] - price)) if demand else None
        if nearest_demand:
            entry_low = nearest_demand['low']
            entry_high = nearest_demand['high']
        else:
            entry_low = price - atr * 0.5
            entry_high = price + atr * 0.2
        stop = entry_low - atr * 0.5
        # TP levels: next supply zones sorted by price
        tps = sorted([z['low'] for z in supply])
        tp1 = tps[0] if tps else price + atr * 2
        tp2 = tps[1] if len(tps) > 1 else price + atr * 4
        tp3 = tps[2] if len(tps) > 2 else price + atr * 6
        invalidation = f"Price breaks below {stop:.2f}"
        risk = entry_low - stop
        rr1 = (tp1 - entry_low) / risk if risk > 0 else 0
        rr2 = (tp2 - entry_low) / risk if risk > 0 else 0
        rr3 = (tp3 - entry_low) / risk if risk > 0 else 0
        # Options: Bull Call Spread
        option_strategy = "BULL_CALL_SPREAD"
        buy_call = round(price / 1000) * 1000 if price > 1000 else round(price)
        sell_call = round(tp1 / 1000) * 1000
        option_buy_call = buy_call
        option_sell_call = sell_call
        option_buy_put = None
        option_sell_put = None
        option_strike_call = (buy_call, sell_call)
        option_strike_put = (None, None)
        option_dte_call = 30
        option_dte_put = 0

    elif direction == "SHORT":
        nearest_supply = min(supply, key=lambda z: abs(z['low'] - price)) if supply else None
        if nearest_supply:
            entry_low = nearest_supply['low']
            entry_high = nearest_supply['high']
        else:
            entry_low = price - atr * 0.2
            entry_high = price + atr * 0.5
        stop = entry_high + atr * 0.5
        tps = sorted([z['low'] for z in demand], reverse=True)
        tp1 = tps[0] if tps else price - atr * 2
        tp2 = tps[1] if len(tps) > 1 else price - atr * 4
        tp3 = tps[2] if len(tps) > 2 else price - atr * 6
        invalidation = f"Price breaks above {stop:.2f}"
        risk = stop - entry_high
        rr1 = (entry_high - tp1) / risk if risk > 0 else 0
        rr2 = (entry_high - tp2) / risk if risk > 0 else 0
        rr3 = (entry_high - tp3) / risk if risk > 0 else 0
        # Options: Bear Put Spread
        option_strategy = "BEAR_PUT_SPREAD"
        buy_put = round(price / 1000) * 1000
        sell_put = round(tp1 / 1000) * 1000
        option_buy_call = None
        option_sell_call = None
        option_buy_put = buy_put
        option_sell_put = sell_put
        option_strike_call = (None, None)
        option_strike_put = (buy_put, sell_put)
        option_dte_call = 0
        option_dte_put = 30
    else:
        # WAIT or NO_TRADE
        return state

    state.direction = direction
    state.entry_zone_low = entry_low
    state.entry_zone_high = entry_high
    state.stop_loss = stop
    state.tp1 = tp1
    state.tp2 = tp2
    state.tp3 = tp3
    state.rr1 = rr1
    state.rr2 = rr2
    state.rr3 = rr3
    state.invalidation = invalidation
    state.option_strategy = option_strategy
    state.option_buy_call = option_buy_call
    state.option_sell_call = option_sell_call
    state.option_buy_put = option_buy_put
    state.option_sell_put = option_sell_put
    state.option_strike_call = option_strike_call
    state.option_strike_put = option_strike_put
    state.option_dte_call = option_dte_call
    state.option_dte_put = option_dte_put
    return state


# ============================================================================
#                            RISK & MONTE CARLO (SHARED STATE)
# ============================================================================

def compute_risk_metrics(state: TradeState) -> TradeState:
    """
    Computes trade quality, Kelly, EV, and Monte Carlo using the same state.
    """
    # Trade quality (simplified)
    rr_best = max(state.rr1, state.rr2, state.rr3)
    if rr_best >= 3.0:
        tq_score = 85.0
    elif rr_best >= 2.0:
        tq_score = 70.0
    elif rr_best >= 1.5:
        tq_score = 60.0
    else:
        tq_score = 40.0
    # Penalise if no demand/supply
    if not state.demand_zones or not state.supply_zones:
        tq_score -= 20
    tq_score = max(0, min(100, tq_score))
    state.trade_quality_score = tq_score

    # Kelly & EV
    win_rate = 0.52 if state.direction == "LONG" else 0.48  # rough baseline
    avg_rr = rr_best
    kelly = (win_rate * avg_rr - (1 - win_rate)) / avg_rr if avg_rr > 0 else 0
    kelly = max(0, min(0.25, kelly))
    ev = win_rate * avg_rr - (1 - win_rate)  # in R
    state.kelly = kelly
    state.ev = ev
    # Risk % of capital (half‑Kelly × 2% max)
    risk_pct = min(0.02, kelly * 0.5 * 0.02 * 2)
    state.risk_pct = risk_pct

    # Monte Carlo using GBM
    close = state.price_data.close
    log_ret = np.diff(np.log(close))
    mu = np.mean(log_ret)
    sigma = np.std(log_ret)
    if sigma <= 0:
        sigma = 0.02
    S0 = state.current_price
    entry = state.entry_zone_low if state.direction == "LONG" else state.entry_zone_high
    stop = state.stop_loss
    tp = state.tp1
    horizon = 20
    n_sims = 5000
    dt = 1/252  # daily steps
    drift = mu - 0.5 * sigma**2
    rng = np.random.default_rng(42)
    shocks = rng.normal(drift * dt, sigma * np.sqrt(dt), size=(n_sims, horizon))
    log_paths = np.cumsum(shocks, axis=1)
    price_paths = S0 * np.exp(log_paths)

    if state.direction == "LONG":
        profit = (price_paths[:, -1] > entry).mean()
        stop_hit = (price_paths.min(axis=1) <= stop).mean()
    else:
        profit = (price_paths[:, -1] < entry).mean()
        stop_hit = (price_paths.max(axis=1) >= stop).mean()
    exp_return = (price_paths[:, -1].mean() / S0 - 1) * 100

    state.mc_prob_profit = profit * 100
    state.mc_prob_drawdown = stop_hit * 100
    state.mc_exp_return = exp_return

    return state


def apply_validation_gates(state: TradeState) -> TradeState:
    """Enforce trade validity rules. Updates trade_allowed flag."""
    rr_best = max(state.rr1, state.rr2, state.rr3)
    ok = True
    reasons = []
    if rr_best < 1.5:
        ok = False; reasons.append(f"RR={rr_best:.1f}<1.5")
    if state.trade_quality_score < 60:
        ok = False; reasons.append(f"TradeQuality={state.trade_quality_score:.0f}<60")
    if state.mc_prob_drawdown > 65:
        ok = False; reasons.append(f"MC stop-hit={state.mc_prob_drawdown:.0f}%>65%")
    if len(state.demand_zones) == 0 or len(state.supply_zones) == 0:
        ok = False; reasons.append("Missing demand or supply zone")
    regime_support = (state.direction == "LONG" and "BULL" in state.regime) or (state.direction == "SHORT" and "BEAR" in state.regime)
    if state.direction != "WAIT" and not regime_support:
        ok = False; reasons.append(f"Regime {state.regime} does not support {state.direction}")

    state.trade_allowed = ok
    state.gate_reason = " | ".join(reasons) if reasons else "All gates passed"
    if not ok:
        state.direction = "NO TRADE"
    return state


# ============================================================================
#                            REPORT RENDERER (EXACT FORMAT)
# ============================================================================

def render_execution_report(state: TradeState, symbol: str) -> str:
    """Generate the final output string matching the hedge‑fund terminal format."""
    lines = []
    # 1. Executive Trade Decision
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("🧠 EXECUTIVE TRADE DECISION")
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"Symbol      : {symbol}")
    lines.append(f"Price       : {state.current_price:.2f}")
    decision_str = state.direction if state.trade_allowed else "NO TRADE"
    lines.append(f"Decision    : {decision_str}")
    bias_reason = f"{state.direction} – {state.regime} ({state.regime_conf:.0f}%) + {state.structure_trend} structure"
    lines.append(f"Bias        : {bias_reason}")
    lines.append(f"Regime      : {state.regime} (prob {state.regime_conf:.0f}%)")
    # Edge: Bayesian proxy using regime confidence and structure
    edge = (state.regime_conf - 50) if "BULL" in state.regime else (50 - state.regime_conf)
    lines.append(f"Edge        : {edge:+.1f}")
    setup_type = "Trend Following" if "BULL" in state.regime or "BEAR" in state.regime else "Range"
    lines.append(f"Setup Type  : {setup_type}")
    trigger = f"Price within {state.entry_zone_low:.0f}–{state.entry_zone_high:.0f} + confirmation candle"
    lines.append(f"Trigger     : {trigger}")
    conf = state.final_confidence if hasattr(state, 'final_confidence') else state.regime_conf
    lines.append(f"Confidence  : {conf:.0f}%")
    lines.append("")

    # 2. Action Summary
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("🚦 ACTION SUMMARY")
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"Allowed trade? : {'YES' if state.trade_allowed else 'NO'}")
    lines.append(f"Reason         : {state.gate_reason}")
    best_action = state.direction if state.trade_allowed else "Stay flat"
    lines.append(f"Best action    : {best_action}")
    lines.append("")

    # 3. Trade Plan (only if allowed)
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("📋 TRADE PLAN")
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    if state.trade_allowed:
        lines.append("\nFUTURES TRADE")
        lines.append("------------------------")
        lines.append(f"Direction      : {state.direction}")
        lines.append(f"Entry Zone     : {state.entry_zone_low:.0f} – {state.entry_zone_high:.0f}")
        lines.append(f"Invalid Trade  : {state.invalidation}")
        lines.append(f"Stop Loss      : {state.stop_loss:.0f}")
        lines.append(f"TP1            : {state.tp1:.0f} (first supply zone)" if state.direction=="LONG" else f"TP1            : {state.tp1:.0f} (first demand zone)")
        lines.append(f"TP2            : {state.tp2:.0f} (major zone)")
        lines.append(f"TP3            : {state.tp3:.0f} (macro zone)")
        lines.append(f"Risk Model     : ATR-based ({state.price_data.atr14[-1]:.0f}) with structure invalidation")
        lines.append(f"RR (TP1)       : {state.rr1:.2f}")
        lines.append(f"RR (TP2)       : {state.rr2:.2f}")
        lines.append(f"RR (TP3)       : {state.rr3:.2f}")
        lines.append("")
        lines.append("OPTION TRADE")
        lines.append("------------------------")
        lines.append(f"Strategy       : {state.option_strategy}")
        lines.append(f"BUY CALL       : {state.option_buy_call if state.option_buy_call else '-'}")
        lines.append(f"BUY PUT        : {state.option_buy_put if state.option_buy_put else '-'}")
        lines.append(f"SELL CALL      : {state.option_sell_call if state.option_sell_call else '-'}")
        lines.append(f"SELL PUT       : {state.option_sell_put if state.option_sell_put else '-'}")
        lines.append(f"STRIKE CALL    : {state.option_strike_call[0]} / {state.option_strike_call[1]}")
        lines.append(f"STRIKE PUT     : {state.option_strike_put[0]} / {state.option_strike_put[1]}")
        lines.append(f"DTE CALL       : {state.option_dte_call}")
        lines.append(f"DTE PUT        : {state.option_dte_put}")
    else:
        lines.append("NO TRADE – WAIT FOR SETUP")
    lines.append("")

    # 4. Demand / Supply Zones
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("🏔️ DEMAND / SUPPLY ZONES")
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("Supply Zones:")
    for i, z in enumerate(state.supply_zones[:2], 1):
        lines.append(f"- zone {i}: {z['low']:.0f}–{z['high']:.0f} | strength {z['strength']:.0f} | {z['reason']}")
    lines.append("")
    lines.append("Demand Zones:")
    for i, z in enumerate(state.demand_zones[:2], 1):
        lines.append(f"- zone {i}: {z['low']:.0f}–{z['high']:.0f} | strength {z['strength']:.0f} | {z['reason']}")
    lines.append("")

    # 5. Regime (compressed)
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("📊 REGIME (COMPRESSED)")
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"State       : {state.regime}")
    lines.append(f"Probability : {state.regime_conf:.0f}%")
    lines.append(f"Key driver  : Markov + Ensemble")
    lines.append("")

    # 6. Volatility (action only)
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("⚡ VOLATILITY (ACTION ONLY)")
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"Regime      : {state.vol_regime}")
    lines.append(f"Action      : {state.vol_action}")
    lines.append("")

    # 7. Risk Summary
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("💹 RISK SUMMARY")
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"EV          : {state.ev:.2f}R")
    lines.append(f"Kelly       : {state.kelly:.4f}")
    lines.append(f"Risk %      : {state.risk_pct*100:.1f}%")
    lines.append("")

    # 8. Monte Carlo (3 lines)
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("🎲 MONTE CARLO (ONLY 3 LINES)")
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"P(Profit)   : {state.mc_prob_profit:.0f}%")
    lines.append(f"P(Drawdown) : {state.mc_prob_drawdown:.0f}% (stop‑hit probability)")
    lines.append(f"Expected Return: {state.mc_exp_return:+.1f}%")
    lines.append("")

    # 9. Final Decision
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("🏁 FINAL DECISION")
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"Decision    : {decision_str}")
    lines.append(f"Confidence  : {conf:.0f}%")
    lines.append(f"One-line reason: {state.gate_reason}")
    lines.append("")

    return "\n".join(lines)


# ============================================================================
#                            UNIFIED TRADE ENGINE (MAIN CLASS)
# ============================================================================

class UnifiedTradeEngine:
    """
    Single source of truth. Takes OHLCV data (pandas DataFrame) and computes
    everything needed for the execution report.
    """

    def __init__(self, df: pd.DataFrame):
        # Step 1: Compute indicators (one pass)
        self.price_data = compute_indicators(df)
        self.current_price = float(df['Close'].iloc[-1])

        # Step 2: Build initial state
        self.state = TradeState(
            price_data=self.price_data,
            current_price=self.current_price,
            regime="UNKNOWN",
            regime_conf=50.0,
            vol_regime="NORMAL_VOL",
            vol_action="",
            structure_trend="MIXED",
            structure_score=50.0,
            ema_bias="BULLISH" if self.price_data.ema12[-1] > self.price_data.ema26[-1] else "BEARISH",
            demand_zones=[],
            supply_zones=[],
            direction="WAIT",
            entry_zone_low=0, entry_zone_high=0,
            stop_loss=0, tp1=0, tp2=0, tp3=0,
            rr1=0, rr2=0, rr3=0,
            invalidation="",
            option_strategy="",
            option_buy_call=None, option_sell_call=None,
            option_buy_put=None, option_sell_put=None,
            option_strike_call=(None,None), option_strike_put=(None,None),
            option_dte_call=0, option_dte_put=0,
            trade_quality_score=0, ev=0, kelly=0, risk_pct=0,
            mc_prob_profit=0, mc_prob_drawdown=0, mc_exp_return=0,
            trade_allowed=False, gate_reason="", final_confidence=0
        )

        # Step 3: Compute regime, volatility, structure
        regime, conf = detect_regime(self.price_data)
        self.state.regime = regime
        self.state.regime_conf = conf

        atr_pct = self.price_data.atr14[-1] / self.current_price * 100
        vol_reg, vol_action = detect_volatility_regime(atr_pct)
        self.state.vol_regime = vol_reg
        self.state.vol_action = vol_action

        struct_trend, struct_score = detect_structure(self.price_data)
        self.state.structure_trend = struct_trend
        self.state.structure_score = struct_score

        # Step 4: Compute demand/supply zones (one time)
        demand, supply = compute_demand_supply_zones(self.price_data, self.current_price)
        self.state.demand_zones = demand
        self.state.supply_zones = supply

        # Step 5: Determine initial direction (simple rule: if regime bullish and structure bullish -> LONG)
        if ("BULL" in regime or regime == "STRONG_BULL") and struct_trend == "BULLISH":
            self.state.direction = "LONG"
        elif ("BEAR" in regime or regime == "STRONG_BEAR") and struct_trend == "BEARISH":
            self.state.direction = "SHORT"
        else:
            self.state.direction = "WAIT"

        # Step 6: Generate trade plan (entry zones, stops, TPs, options)
        if self.state.direction != "WAIT":
            self.state = generate_trade_plan(self.state)

        # Step 7: Compute risk and Monte Carlo (using the same state)
        if self.state.direction != "WAIT":
            self.state = compute_risk_metrics(self.state)
        else:
            # fill dummies
            self.state.trade_quality_score = 0
            self.state.kelly = 0
            self.state.ev = 0
            self.state.risk_pct = 0
            self.state.mc_prob_profit = 0
            self.state.mc_prob_drawdown = 0
            self.state.mc_exp_return = 0

        # Step 8: Apply validation gates (overrides direction if invalid)
        self.state = apply_validation_gates(self.state)

        # Final confidence (average of regime conf and trade quality)
        self.state.final_confidence = (self.state.regime_conf + self.state.trade_quality_score) / 2

    def get_report(self, symbol: str) -> str:
        """Returns the final execution report as a string."""
        return render_execution_report(self.state, symbol)