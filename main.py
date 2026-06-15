# main.py – full file (as in your codebase, verified)
"""
TradeAnalyze — Main Entry Point  (v2 — full institutional)
=============================================================
Uses FuturesOrchestrator_v2/v3 exclusively.
All LINE messages go through line_alert_v2 (which exports send_line_message).
"""
import time
import traceback
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

from config.config_validator import validate
from config.logging_config import logger
from alerts.line_alert_v2 import send_institutional_alert
from alerts.notification_manager import send_notification
from data.market_data import get_market_data
from data.option_chain import fetch_option_chain
from engines.greeks_pipeline import enrich_with_greeks
from options.options_orchestrator import run_options_analysis
from options.iv_rank import compute_iv_rank
from options.vol_surface import compute_vol_surface
from regime.markov import MarkovRegimeEngine
from indicators.ema import compute_ema
from indicators.rsi import compute_rsi
from indicators.atr import compute_atr
from reports.options_formatter import format_options_message
from reports.options_sheet_writer import write_options_analysis
from reports.option_chain_writer import clear_symbol_rows, write_option_chain
from reports.sheet_writer import log_trade_signals
from reports.sheet_writer_v2 import write_all_institutional
from utils.symbol_loader import load_symbols_with_type
from config.thresholds import THRESHOLDS
from persistence.trade_persistence import get_persistence

from config.config import USE_V3

if USE_V3:
    from core.futures_orchestrator_v3 import FuturesOrchestrator_v3
    from reports.execution_report_v3 import build_execution_report_v3
    OrchestratorClass = FuturesOrchestrator_v3
    report_builder = build_execution_report_v3
else:
    from core.futures_orchestrator_v2 import FuturesOrchestrator_v2
    OrchestratorClass = FuturesOrchestrator_v2
    report_builder = None

MIN_CONVICTION_ALERT = 60.0

# ---------- Helper: warn about zero transaction costs ----------
def _warn_transaction_costs():
    """Log a warning if transaction costs are not being modelled."""
    if THRESHOLDS.MODEL_TRANSACTION_COSTS:
        logger.info(
            "Transaction cost model ENABLED: stocks %.3f%% (round-trip), crypto %.3f%%",
            THRESHOLDS.COST_STOCK_PCT * 200,
            THRESHOLDS.COST_CRYPTO_PCT * 200,
        )
    else:
        logger.warning("Transaction costs are DISABLED in config/thresholds.py. "
                       "For realistic backtesting, set MODEL_TRANSACTION_COSTS=True.")

# ---------- Helper: load open positions from database ----------
def load_open_positions():
    """Recover open positions from database and notify."""
    persistence = get_persistence()
    active = persistence.load_active_trades()
    if active:
        logger.info("Recovered %d open positions from database:", len(active))
        for trade in active:
            logger.info("  %s %s entry=%.2f size=%.2f%%", trade.symbol, trade.direction,
                        trade.entry_price, trade.position_size)
            # Optionally send a recovery alert
            send_notification(f"[RECOVERY] Open position: {trade.symbol} {trade.direction} "
                              f"entry={trade.entry_price:.2f} risk={trade.position_size:.1f}%")
    else:
        logger.info("No open positions found in database.")
    return active

# ---------- Helper to build trade signal dict ----------
def _build_trade_signal_dict(symbol, futures, asset_type):
    return {
        "symbol": symbol, "regime": futures.regime, "price": futures.price,
        "position": futures.final_decision, "entry": futures.entry,
        "sl": futures.stop_loss, "tp1": futures.tp1, "tp2": futures.tp2,
        "risk": abs(futures.entry - futures.stop_loss),
        "holding_days": 0, "active": futures.approved,
        "ai_score": futures.ai_score, "rr": futures.rr,
        "greek_conviction": futures.trade_grade,
        "conviction_reasons": [futures.stop_reason],
        "greek_strategy_hint": futures.vol_regime,
        "iv_rank_proxy": None, "iv_environment": futures.vol_regime,
        "put_call_delta_skew": None, "dominant_dte": None,
        "near_term_risk": False, "avg_iv": None, "pc_oi_ratio": None,
        "avg_gamma": None, "fast_decay_pct": None, "asset_type": asset_type,
    }

# ---------- Crypto extras ----------
def _run_crypto_extras(symbol: str, price: float) -> str:
    lines = ["", "━"*28, "🔐 CRYPTO INSTITUTIONAL DATA", "━"*28]
    try:
        from crypto.funding_rate import fetch_funding_rate
        fr = fetch_funding_rate(symbol)
        lines += [
            f"  Funding Rate: {fr.funding_rate_pct:+.5f}%",
            f"  Regime      : {fr.funding_regime}",
            f"  Signal      : {fr.contrarian_signal}",
            f"  ⚠️  {fr.interpretation}" if fr.crowded_long or fr.crowded_short else f"  {fr.interpretation}",
        ]
    except Exception as exc:
        lines.append(f"  Funding Rate: unavailable ({exc})")

    try:
        from crypto.open_interest import fetch_open_interest
        oi = fetch_open_interest(symbol, price_change=0.0)
        lines += [
            f"  OI          : {oi.open_interest:,.0f}",
            f"  OI Trend    : {oi.oi_trend}",
            f"  Price×OI    : {oi.price_oi_signal}  ({oi.signal_strength})",
        ]
    except Exception as exc:
        lines.append(f"  Open Interest: unavailable ({exc})")

    return "\n".join(lines)

# ---------- Main trading engine ----------
def run_trading_engine() -> None:
    # Log transaction cost assumptions (Task 1.6)
    _warn_transaction_costs()

    # Recover open positions (Task 2.4)
    open_positions = load_open_positions()

    validate()
    orchestrator = OrchestratorClass(win_rate=0.52, avg_rr=2.5)
    regime_engine = MarkovRegimeEngine()
    success = fail = 0

    symbol_list = load_symbols_with_type("LINE")
    if not symbol_list:
        print("❌ No symbols found in SYMBOL_CONFIG (group=LINE)")
        return

    print(f"\n🚀 ===== TRADING ENGINE START =====")
    print(f"📊 Symbols: {len(symbol_list)}")
    if open_positions:
        print(f"🔄 Recovered {len(open_positions)} open positions from database")

    for item in symbol_list:
        symbol = item["symbol"]
        asset_type = item["asset_type"]
        print(f"\n{'━'*44}")
        print(f"📊 {symbol}  ({asset_type})")

        # Skip if symbol already has an open position (prevent duplicate)
        persistence = get_persistence()
        if persistence.has_active_trade(symbol):
            print(f"  ⏸️  Skipping {symbol} – active trade already exists (reconciliation)")
            logger.info("[%s] Skipped due to existing active trade", symbol)
            continue

        try:
            df = get_market_data(symbol)
            if df is None or df.empty:
                print(f"  ❌ No market data"); fail += 1; continue

            price = float(df["Close"].iloc[-1])
            print(f"  ⚙️  Futures analysis...")
            futures = orchestrator.run(symbol, df)

            dec_e = {"LONG":"🟢","SHORT":"🔴","NO_TRADE":"⏸️"}.get(futures.final_decision,"❓")
            print(f"  {dec_e} {futures.final_decision}  "
                  f"Regime={futures.regime}({futures.regime_conf:.0f}%)  "
                  f"Grade={futures.trade_grade}  AI={futures.ai_score:.0f}  "
                  f"RR={futures.rr:.2f}  MC={futures.mc_profit_prob:.0f}%")

            # Write sheets
            sig_dict = _build_trade_signal_dict(symbol, futures, asset_type)
            log_trade_signals(symbol, [sig_dict], [{"bull":0,"bear":0,"sideway":0}])
            write_all_institutional(
                symbol=symbol, price=price, result_v2=futures,
                liquidity=getattr(futures, 'liquidity_result', None),
                flow=getattr(futures, 'flow_result', None),
                breadth=getattr(futures, 'breadth_result', None),
                persistence=getattr(futures, 'persistence_result', None),
                forecast=getattr(futures, 'forecast_result', None),
                conviction=getattr(futures, 'conviction_result', None),
                cross_asset=getattr(futures, 'cross_asset_result', None),
            )

            # Option Chain & Options Analysis
            print(f"  ⚙️  Option chain...")
            enriched_chain = []
            opts_rec = None
            try:
                raw_chain = fetch_option_chain(symbol, price, asset_type=asset_type)
                enriched_chain = enrich_with_greeks(raw_chain, spot=price)
                if enriched_chain:
                    clear_symbol_rows(symbol)
                    n = write_option_chain(symbol, enriched_chain)
                    print(f"  📋 Option_Chain: {n} rows ✅")
                    df_ind = compute_ema(compute_rsi(compute_atr(df.copy())))
                    chain_iv = next((float(r["iv"]) for r in enriched_chain
                                     if r.get("option_type") == "call" and r.get("iv", 0) > 0), None)
                    iv_rank_result = compute_iv_rank(df_ind, current_iv=chain_iv)
                    iv_surface = compute_vol_surface(enriched_chain)
                    print(f"  📐 IV Rank={iv_rank_result.iv_rank:.0f}  {iv_rank_result.signal}  "
                          f"Skew={iv_surface.skew_signal}")
                else:
                    print(f"  ⚠️  Option chain: no data")
            except Exception as exc:
                logger.warning("[%s] Option chain: %s", symbol, exc)
                print(f"  ⚠️  Option chain: {exc}")

            print(f"  ⚙️  Options analysis...")
            try:
                df_ind = compute_ema(compute_rsi(compute_atr(df.copy())))
                try:
                    reg_result = regime_engine.detect(df_ind)
                    regime_probs = reg_result.regime_probs_all
                except Exception:
                    regime_probs = {futures.regime: 0.65}

                opts_rec = run_options_analysis(
                    symbol=symbol, price=price, df=df_ind,
                    regime=futures.regime, regime_conf=futures.regime_conf,
                    regime_probs=regime_probs, ai_score=futures.ai_score,
                    enriched_chain=enriched_chain,
                )
                write_options_analysis(opts_rec)
                print(f"  📊 Options: {opts_rec.primary.name}  "
                      f"score={opts_rec.primary.score:.0f}  "
                      f"EV={opts_rec.primary.ev:.1f}  "
                      f"POP={opts_rec.primary.pop:.0f}%  "
                      f"{'✅' if opts_rec.trade_approved else '⏸️'}")
            except Exception as exc:
                logger.error("[%s] Options analysis: %s", symbol, exc)
                print(f"  ⚠️  Options: {exc}")

            # Crypto extras
            if asset_type == "crypto":
                try:
                    crypto_msg = _run_crypto_extras(symbol, price)
                    send_notification(crypto_msg)
                    print(f"  🔐 Crypto data sent")
                except Exception as exc:
                    print(f"  ⚠️  Crypto extras: {exc}")

            # Unified execution report (futures + options)
            if USE_V3:
                v3_state = getattr(futures, 'v3_state', None)
                decision_report = report_builder(symbol, price, futures, v3_state, opts_rec)
            else:
                decision_report = futures.report_text

            msg = decision_report[:4490] + "\n…" if len(decision_report) > 4500 else decision_report
            send_notification(msg)
            print(f"  📱 Unified execution report → Notification sent")

            # Institutional alert
            send_institutional_alert(
                symbol=symbol, price=price, result_v2=futures,
                liquidity=getattr(futures, 'liquidity_result', None),
                flow=getattr(futures, 'flow_result', None),
                breadth=getattr(futures, 'breadth_result', None),
                persistence=getattr(futures, 'persistence_result', None),
                forecast=getattr(futures, 'forecast_result', None),
                conviction=getattr(futures, 'conviction_result', None),
                cross_asset=getattr(futures, 'cross_asset_result', None),
                min_conviction=MIN_CONVICTION_ALERT,
            )

            success += 1
            print(f"  ⏱  {futures.runtime:.1f}s")
            time.sleep(1.5)

        except Exception:
            fail += 1
            logger.error("[%s] UNHANDLED:\n%s", symbol, traceback.format_exc())
            print(f"  ❌ ERROR:\n{traceback.format_exc()}")

    print(f"\n{'━'*44}")
    print(f"🏁 DONE  ✅ {success}  ❌ {fail}")
    logger.info("Engine done — success=%d fail=%d", success, fail)

if __name__ == "__main__":
    try:
        # Set timezone to UTC for all datetime operations
        import os
        os.environ["TZ"] = "UTC"
        time.tzset()
        run_trading_engine()
    except Exception:
        logger.critical("GLOBAL ERROR:\n%s", traceback.format_exc())
        print(f"GLOBAL ERROR:\n{traceback.format_exc()}")