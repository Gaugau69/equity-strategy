"""
ibkr_executor.py
----------------
Paper trading executor — deux étapes séparées pour éviter les conflits asyncio :

  Étape 1 (--generate) : génère les signaux, sauvegarde dans orders_pending.json
  Étape 2 (--execute)  : lit orders_pending.json et passe les ordres via IBKR
  Étape 0 (--dry-run)  : fait les deux sans passer d'ordres

Usage
-----
    python ibkr_executor.py --dry-run --force     # test complet sans ordres
    python ibkr_executor.py --generate --force    # génère signaux seulement
    python ibkr_executor.py --execute             # passe les ordres
    python ibkr_executor.py --force               # run complet (lundi seulement)
"""

from __future__ import annotations
import sys, argparse, json, time, warnings, logging

# Fix Unicode pour terminal Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from pathlib import Path
from datetime import datetime, date
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

LOG_DIR = Path("outputs/paper_trading")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "executor.log"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

TWS_HOST      = "127.0.0.1"
TWS_PORT      = 7497
TWS_CLIENT_ID = 10
CAPITAL       = 100_000
REBAL_DAY     = 0   # Monday
PRICES_CSV    = "data/prices_pit.csv"  # donnees historiques pour signaux
PRICES_FRESH  = "data/prices.csv"      # donnees fraiches pour prix execution
ORDERS_FILE   = LOG_DIR / "orders_pending.json"


def step1_generate(capital: float) -> dict:
    from strategy.factors   import compute_factors, cross_sectional_zscore
    from strategy.signals   import generate_signals
    from strategy.portfolio import build_portfolio
    from strategy           import config

    log.info("[1/2] Loading prices...")
    prices = pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True)
    prices = prices.sort_index().tail(400)
    prices = prices.ffill(limit=5).dropna(axis=1, thresh=int(len(prices) * 0.9))
    log.info(f"      {prices.shape[1]} tickers x {len(prices)} days")

    log.info("[1/2] Computing factors & signals...")
    market  = prices.mean(axis=1)
    fraw    = compute_factors(prices, market)
    fnorm   = cross_sectional_zscore(fraw)
    fwd     = prices.pct_change(config.REBAL_FREQ).shift(-config.REBAL_FREQ)
    signals = generate_signals(fnorm, fwd,
                               train_window=config.TRAIN_WINDOW,
                               rebal_freq=config.REBAL_FREQ,
                               alpha=config.RIDGE_ALPHA)
    if signals.empty:
        log.error("No signals generated")
        return {}

    latest = signals.index[-1]
    log.info(f"      Signal date: {latest.date()}")

    daily  = prices.pct_change()
    mkt_r  = market.pct_change()
    window = min(252, len(daily) - 5)
    cov    = daily.iloc[-window:].subtract(daily.iloc[-window:].mean()).multiply(
             mkt_r.iloc[-window:] - mkt_r.iloc[-window:].mean(), axis=0).mean()
    betas  = (cov / mkt_r.iloc[-window:].var()).fillna(1.0)

    t_idx   = prices.index.get_loc(latest)
    cov_mat = daily.iloc[max(0, t_idx-60):t_idx].cov().fillna(0).values * 252
    sig     = signals.loc[latest].reindex(prices.columns)
    weights = build_portfolio(sig, betas, cov_mat, list(prices.columns),
                              top_n=config.TOP_N, max_weight=config.MAX_WEIGHT)
    weights = weights[weights.abs() > 1e-5]

    log.info(f"      {(weights>0).sum()} long, {(weights<0).sum()} short")

    # Use fresh prices for order sizing if available
    import os
    if os.path.exists("data/prices.csv"):
        try:
            fresh = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
            latest_prices = fresh.iloc[-1]
            log.info(f"      Using fresh prices from {fresh.index[-1].date()} for order sizing")
        except Exception:
            latest_prices = prices.iloc[-1]
    else:
        latest_prices = prices.iloc[-1]
    orders = []
    for ticker, w in weights.items():
        price = latest_prices.get(ticker, None)
        if price is None or np.isnan(float(price)) or price <= 0:
            continue
        qty = int(abs(w * capital) / price)
        if qty == 0:
            continue
        orders.append({
            "ticker":       ticker,
            "action":       "BUY" if w > 0 else "SELL",
            "quantity":     qty,
            "price_est":    round(float(price), 2),
            "weight":       round(float(w), 4),
            "dollar_value": round(float(w * capital), 2),
        })

    result = {
        "generated_at": datetime.now().isoformat(),
        "signal_date":  str(latest.date()),
        "n_long":       int((weights > 0).sum()),
        "n_short":      int((weights < 0).sum()),
        "capital":      capital,
        "orders":       orders,
    }
    with open(ORDERS_FILE, "w") as f:
        json.dump(result, f, indent=2)

    log.info(f"      {len(orders)} orders saved to {ORDERS_FILE}")
    return result


def step2_execute(dry_run: bool = False) -> None:
    if not ORDERS_FILE.exists():
        log.error(f"No orders file: {ORDERS_FILE}. Run --generate first.")
        return

    with open(ORDERS_FILE) as f:
        data = json.load(f)

    orders = data["orders"]
    log.info(f"[2/2] {'[DRY RUN] ' if dry_run else ''}Executing {len(orders)} orders")
    log.info(f"      Signal: {data['signal_date']}  |  {data['n_long']}L / {data['n_short']}S")

    log.info(f"\n  {'Action':<6} {'Qty':>6} {'Ticker':<8} {'Price':>8} {'Weight':>8} {'$Value':>10}")
    log.info(f"  {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for o in orders:
        log.info(f"  {o['action']:<6} {o['quantity']:>6} {o['ticker']:<8} "
                 f"${o['price_est']:>7.2f} {o['weight']:>+8.4f} ${o['dollar_value']:>9.0f}")

    if dry_run:
        log.info("\n  [DRY RUN] Orders shown — nothing sent to IBKR")
        return

    from ib_insync import IB, Stock, LimitOrder, util
    util.logToConsole(50)
    ib = IB()
    log.info(f"\n  Connecting to TWS {TWS_HOST}:{TWS_PORT}...")
    ib.connect(TWS_HOST, TWS_PORT, clientId=TWS_CLIENT_ID, timeout=10)
    log.info(f"  Connected: {ib.managedAccounts()}")

    filled = []
    for o in orders:
        try:
            contract  = Stock(o["ticker"].replace("-", " "), "SMART", "USD")
            qualified = ib.qualifyContracts(contract)
            if not qualified:
                log.warning(f"  Cannot qualify {o['ticker']} — skip")
                continue
            limit_px = round(o["price_est"] * (1.002 if o["action"] == "BUY" else 0.998), 2)
            trade    = ib.placeOrder(qualified[0], LimitOrder(o["action"], o["quantity"], limit_px))
            filled.append(trade)
            log.info(f"  Placed: {o['action']} {o['quantity']} {o['ticker']} @ ${limit_px}")
            time.sleep(0.05)
        except Exception as e:
            log.warning(f"  Order failed {o['ticker']}: {e}")

    if filled:
        ib.sleep(15)
    ib.disconnect()
    log.info(f"  {len(filled)} orders placed")

    summary = pd.DataFrame([{
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "signal_date": data["signal_date"],
        "n_orders": len(orders),
        "dry_run": dry_run,
    }])
    log_path = LOG_DIR / "rebalance_log.csv"
    if log_path.exists():
        summary = pd.concat([pd.read_csv(log_path), summary], ignore_index=True)
    summary.to_csv(log_path, index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run",  action="store_true")
    p.add_argument("--force",    action="store_true")
    p.add_argument("--generate", action="store_true")
    p.add_argument("--execute",  action="store_true")
    p.add_argument("--capital",  type=float, default=CAPITAL)
    args = p.parse_args()

    log.info("=" * 60)
    log.info(" IBKR Paper Trading Executor")
    log.info(f" Mode: {'DRY RUN' if args.dry_run else 'LIVE PAPER'}")
    log.info(f" Capital: ${args.capital:,.0f}")
    log.info("=" * 60)

    today = date.today()
    if not args.force and not args.execute and today.weekday() != REBAL_DAY:
        log.info(f"Today is {today.strftime('%A')} — not Monday. Use --force.")
        return

    if args.execute:
        step2_execute(dry_run=args.dry_run)
    elif args.generate:
        step1_generate(args.capital)
    else:
        result = step1_generate(args.capital)
        if result:
            step2_execute(dry_run=args.dry_run)

    log.info("\nDone. OK")


if __name__ == "__main__":
    main()