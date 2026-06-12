"""
TradeAnalyze — Main Entry Point  (v2 integration – dry run)
=============================================================
Runs both v1 (live) and v2 (dry-run) pipelines.
v2 writes institutional sheets but does NOT send LINE alerts.
After validation, switch to v2-only.
"""
import time
import traceback
import warnings
warnings.filterwarnings("ignore")

from config.config_validator import validate
from config.logging_config import logger
from alerts.line_alert import send_line_message
from core.futures_orchestrator import FuturesOrchestrator
from core.futures_orchestrator_v2 import FuturesOrchestrator_v2   # NEW
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
from reports.sheet_writer_v2 import write_all_institutional   # NEW
from utils.symbol_loader import load_symbols_with_type

# Flag to control v2 LINE alerts – set to False during dry run
V2_SEND_ALERTS = False   # Change to True after validation

def _build_trade_signal_dict(symbol, futures, asset_type):
    """Build signal dict for TradeSignals sheet."""
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

def _run_crypto_extras(symbol: str, price: float) -> str:
    """Fetch crypto-specific data and return a LINE-ready summary string."""
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
        price_change = 0.0
        oi = fetch_open_interest(symbol, price_change=price_change)
        lines += [
            f"  OI          : {oi.open_interest:,.0f}",
            f"  OI Trend    : {oi.oi_trend}",
            f"  Price×OI    : {oi.price_oi_signal}  ({oi.signal_strength})",
        ]
    except Exception as exc:
        lines.append(f"  Open Interest: unavailable ({exc})")

    return "\n".join(lines)

def run_trading_engine() -> None:
    validate()

    futures_orch_v1 = FuturesOrchestrator(win_rate=0.52, avg_rr=2.5)
    futures_orch_v2 = FuturesOrchestrator_v2(win_rate=0.52, avg_rr=2.5)   # NEW
    regime_engine = MarkovRegimeEngine()
    success = fail = 0

    symbol_list = load_symbols_with_type("LINE")
    if not symbol_list:
        print("❌ No symbols found in SYMBOL_CONFIG (group=LINE)")
        return

    print(f"\n🚀 ===== TRADING ENGINE START =====")
    print(f"📊 Symbols: {len(symbol_list)}")

    for item in symbol_list:
        symbol     = item["symbol"]
        asset_type = item["asset_type"]

        print(f"\n{'━'*44}")
        print(f"📊 {symbol}  ({asset_type})")

        try:
            df = get_market_data(symbol)
            if df is None or df.empty:
                print(f"  ❌ No market data"); fail += 1; continue

            price = float(df["Close"].iloc[-1])

            # ── A) v1 Futures Pipeline (live alerts) ─────────────────────────
            print(f"  ⚙️  Futures analysis (v1)...")
            futures_v1 = futures_orch_v1.run(symbol, df)

            dec_e = {"LONG":"🟢","SHORT":"🔴","NO_TRADE":"⏸️"}.get(futures_v1.final_decision,"❓")
            print(f"  {dec_e} {futures_v1.final_decision}  "
                  f"Regime={futures_v1.regime}({futures_v1.regime_conf:.0f}%)  "
                  f"Grade={futures_v1.trade_grade}  AI={futures_v1.ai_score:.0f}  "
                  f"RR={futures_v1.rr:.2f}  MC={futures_v1.mc_profit_prob:.0f}%")

            # Write TradeSignals sheet (v1)
            sig_dict = _build_trade_signal_dict(symbol, futures_v1, asset_type)
            log_trade_signals(symbol, [sig_dict], [{"bull":0,"bear":0,"sideway":0}])

            # Send v1 LINE alerts (live)
            msg_v1 = futures_v1.report_text[:4490] + "\n…" if len(futures_v1.report_text) > 4500 else futures_v1.report_text
            send_line_message(msg_v1)
            print(f"  📱 Futures report (v1) → LINE ✅")

            # ── B) v2 Futures Pipeline (dry run – sheets only, no alerts) ────
            print(f"  ⚙️  Futures analysis (v2 dry run)...")
            futures_v2 = futures_orch_v2.run(symbol, df)

            # Write institutional sheets (new)
            write_all_institutional(
                symbol=symbol, price=price, result_v2=futures_v2,
                liquidity=getattr(futures_v2, 'liquidity_result', None),
                flow=getattr(futures_v2, 'flow_result', None),
                breadth=getattr(futures_v2, 'breadth_result', None),
                persistence=getattr(futures_v2, 'persistence_result', None),
                forecast=getattr(futures_v2, 'forecast_result', None),
                conviction=getattr(futures_v2, 'conviction_result', None),
                cross_asset=getattr(futures_v2, 'cross_asset_result', None),
            )

            # Optionally send v2 alerts (disabled during dry run)
            if V2_SEND_ALERTS:
                from alerts.line_alert_v2 import send_institutional_alert
                send_institutional_alert(
                    symbol=symbol, price=price, result_v2=futures_v2,
                    liquidity=getattr(futures_v2, 'liquidity_result', None),
                    flow=getattr(futures_v2, 'flow_result', None),
                    breadth=getattr(futures_v2, 'breadth_result', None),
                    persistence=getattr(futures_v2, 'persistence_result', None),
                    forecast=getattr(futures_v2, 'forecast_result', None),
                    conviction=getattr(futures_v2, 'conviction_result', None),
                    cross_asset=getattr(futures_v2, 'cross_asset_result', None),
                    min_conviction=65.0,
                )
            else:
                print(f"  📋 v2 institutional sheets written (alerts disabled)")

            # ── C) Option Chain + IV Rank + Vol Surface (unchanged) ──────────
            print(f"  ⚙️  Option chain...")
            enriched_chain = []
            iv_rank_result = iv_surface = None
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

            # ── D) Options Analysis (unchanged) ──────────────────────────────
            print(f"  ⚙️  Options analysis...")
            try:
                df_ind = compute_ema(compute_rsi(compute_atr(df.copy())))
                try:
                    reg_result = regime_engine.detect(df_ind)
                    regime_probs = reg_result.regime_probs_all
                except Exception:
                    regime_probs = {futures_v1.regime: 0.65}

                opts_rec = run_options_analysis(
                    symbol=symbol, price=price, df=df_ind,
                    regime=futures_v1.regime, regime_conf=futures_v1.regime_conf,
                    regime_probs=regime_probs, ai_score=futures_v1.ai_score,
                    enriched_chain=enriched_chain,
                )
                write_options_analysis(opts_rec)
                opts_msg = format_options_message(opts_rec)
                if len(opts_msg) > 4500: opts_msg = opts_msg[:4490] + "\n…"
                send_line_message(opts_msg)
                print(f"  📊 Options: {opts_rec.primary.name}  "
                      f"score={opts_rec.primary.score:.0f}  "
                      f"EV={opts_rec.primary.ev:.1f}  "
                      f"POP={opts_rec.primary.pop:.0f}%  "
                      f"{'✅' if opts_rec.trade_approved else '⏸️'}")
            except Exception as exc:
                logger.error("[%s] Options analysis: %s", symbol, exc)
                print(f"  ⚠️  Options: {exc}")

            # ── E) Crypto extras (unchanged) ─────────────────────────────────
            if asset_type == "crypto":
                try:
                    crypto_msg = _run_crypto_extras(symbol, price)
                    send_line_message(crypto_msg)
                    print(f"  🔐 Crypto data sent")
                except Exception as exc:
                    print(f"  ⚠️  Crypto extras: {exc}")

            success += 1
            print(f"  ⏱  {futures_v1.runtime:.1f}s")
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
        run_trading_engine()
    except Exception:
        logger.critical("GLOBAL ERROR:\n%s", traceback.format_exc())
        print(f"GLOBAL ERROR:\n{traceback.format_exc()}")