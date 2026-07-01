"""
run_live.py
-----------
Weekly paper-trading runner (Interactive Brokers).

Run every Monday morning before market open. The script:
  1. Downloads ~2.5 years of daily prices via yfinance
  2. Computes factors, generates the rolling Ridge signal
  3. Builds target portfolio weights via Markowitz
  4. Fetches current positions from IBKR (paper account)
  5. Computes the trade delta and submits MOO orders

Prerequisites
-------------
  pip install ib_insync yfinance
  TWS or IB Gateway must be running with paper account selected
  (Socket port 7497 for TWS, 4002 for IB Gateway)

Usage
-----
  python run_live.py                     # dry run (print orders, no submission)
  python run_live.py --submit            # submit orders to IBKR
  python run_live.py --account-value 50000  # override account size ($50k)
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf

from strategy.factors  import compute_factors, cross_sectional_zscore
from strategy.signals  import generate_signals
from strategy.portfolio import build_portfolio
from strategy           import config

# ─────────────────────────────────────────────────────────────────────────────
# Universe (same as run_real_data.py curated fallback)
# ─────────────────────────────────────────────────────────────────────────────

UNIVERSE = [
    "AAPL","MSFT","AMZN","NVDA","GOOGL","META","JPM","JNJ","V","PG",
    "XOM","LLY","MA","AVGO","HD","CVX","MRK","ABBV","COST","PEP","KO","WMT",
    "CSCO","TMO","MCD","ACN","BAC","ABT","CRM","NFLX","LIN","AMD","ADBE","TXN",
    "NKE","NEE","PM","ORCL","HON","RTX","AMGN","QCOM","LOW","GS","INTU","CAT",
    "AXP","SYK","BKNG","GILD","ADP","NOW","TJX","REGN","VRTX","ZTS","CB","C",
    "CME","CVS","MO","EOG","SO","DUK","PGR","BSX","MU","SCHW","ITW","NOC",
    "ETN","AON","SHW","BDX","ISRG","CSX","WM","CL","FDX","NSC","ECL","MCO",
    "ORLY","PSA","CTAS","ICE","KLAC","LRCX","SNPS","CDNS","PANW","ADI","AMAT",
]

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────

LOOKBACK_YEARS  = 3       # years of history to download (needs > TRAIN_WINDOW)
IBKR_HOST       = "127.0.0.1"
IBKR_PORT       = 7497    # TWS paper: 7497 | IB Gateway paper: 4002
IBKR_CLIENT_ID  = 10
MIN_TRADE_BPS   = 50      # ignore rebalances below 0.50% of NAV per leg
DEFAULT_ACCOUNT = 100_000 # fallback account value if IBKR query fails ($)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Data
# ─────────────────────────────────────────────────────────────────────────────

def fetch_prices() -> tuple[pd.DataFrame, pd.Series]:
    end   = date.today()
    start = end - timedelta(days=int(LOOKBACK_YEARS * 365.25))
    print(f"[1/5] Downloading prices ({start} → {end})…")

    raw = yf.download(UNIVERSE, start=str(start), end=str(end),
                      auto_adjust=True, progress=False)
    prices = raw["Close"].sort_index(axis=1)

    # Basic cleaning
    prices = prices.loc[:, prices.notna().mean() >= 0.90]
    prices = prices.ffill(limit=5).dropna(how="any", axis=0)
    print(f"      {len(prices.columns)} stocks × {len(prices)} days")

    spy = yf.download("SPY", start=str(start), end=str(end),
                      auto_adjust=True, progress=False)["Close"].squeeze()
    market = spy.reindex(prices.index).ffill()

    return prices, market


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Signal
# ─────────────────────────────────────────────────────────────────────────────

def compute_target_weights(prices: pd.DataFrame, market: pd.Series) -> pd.Series:
    print("[2/5] Computing factors…")
    fraw  = compute_factors(prices, market)
    fnorm = cross_sectional_zscore(fraw)

    fwd   = prices.pct_change(config.REBAL_FREQ).shift(-config.REBAL_FREQ)
    fwd_m = market.pct_change(config.REBAL_FREQ).shift(-config.REBAL_FREQ)

    print("[3/5] Generating signals (rolling Ridge)…")
    signals = generate_signals(fnorm, fwd,
                                train_window       = config.TRAIN_WINDOW,
                                rebal_freq         = config.REBAL_FREQ,
                                alpha              = config.RIDGE_ALPHA,
                                market_fwd_returns = fwd_m)

    # Use the most recent signal date
    last_signal = signals.iloc[-1]
    print(f"      Signal date: {signals.index[-1].date()}")

    # Estimate betas (last 252 days)
    dr  = prices.pct_change().dropna()
    mr  = market.pct_change().dropna().reindex(dr.index)
    w   = min(252, len(dr) - 5)
    cov = (dr.iloc[-w:].subtract(dr.iloc[-w:].mean())
             .multiply(mr.iloc[-w:] - mr.iloc[-w:].mean(), axis=0).mean())
    betas = (cov / mr.iloc[-w:].var()).reindex(prices.columns).fillna(1.0)

    n_stocks = len(prices.columns)
    top_n    = max(10, min(config.TOP_N, n_stocks // 8))

    # Latest covariance (Ledoit-Wolf)
    from sklearn.covariance import LedoitWolf
    cov_slice = dr.iloc[-config.COV_LOOKBACK:].dropna()
    lw        = LedoitWolf().fit(cov_slice.values)
    cov_mat   = lw.covariance_ * 252

    print("[4/5] Building portfolio…")
    weights = build_portfolio(
        last_signal,
        betas,
        cov_mat,
        list(prices.columns),
        top_n      = top_n,
        lambda_reg = config.LAMBDA_REG,
    )

    # Print summary
    longs  = weights[weights > 0].sort_values(ascending=False)
    shorts = weights[weights < 0].sort_values()
    print(f"      Longs  ({len(longs)}): {', '.join(longs.index[:5].tolist())}…")
    print(f"      Shorts ({len(shorts)}): {', '.join(shorts.index[:5].tolist())}…")
    print(f"      Beta exposure: {(weights * betas).sum():+.3f}")
    print(f"      Gross leverage: {weights.abs().sum():.2f}x")

    return weights


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — IBKR connection
# ─────────────────────────────────────────────────────────────────────────────

def get_ibkr_state(account_value_override: float | None = None):
    """
    Connect to IBKR, fetch current paper positions and account equity.
    Returns (current_weights: pd.Series, account_value: float).
    """
    try:
        from ib_insync import IB, Stock, util
        util.logToConsole("CRITICAL")
    except ImportError:
        print("  ib_insync not installed — run: pip install ib_insync")
        sys.exit(1)

    ib = IB()
    try:
        ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID, timeout=10)
    except Exception as e:
        print(f"  Cannot connect to IBKR ({e})")
        print("  Make sure TWS / IB Gateway is running with paper account.")
        sys.exit(1)

    # Account value
    if account_value_override:
        nav = account_value_override
    else:
        vals = ib.accountValues()
        # Try USD first, then any base currency (accounts denominated in EUR, GBP, etc.)
        nav_usd  = [v for v in vals if v.tag == "NetLiquidation" and v.currency == "USD"]
        nav_base = [v for v in vals if v.tag == "NetLiquidation" and v.currency not in ("", "BASE", "USD")]
        if nav_usd:
            nav = float(nav_usd[0].value)
        elif nav_base:
            nav = float(nav_base[0].value)
        else:
            nav = DEFAULT_ACCOUNT
    print(f"      Account NAV: ${nav:,.0f}")

    # Current positions
    positions = {p.contract.symbol: p.position for p in ib.positions()
                 if p.contract.secType == "STK"}

    # Convert share counts to weight fractions
    current_weights = {}
    if positions:
        tickers = list(positions.keys())
        try:
            snap = yf.download(tickers, period="1d", auto_adjust=True,
                               progress=False)["Close"].iloc[-1]
        except Exception:
            snap = pd.Series(dtype=float)
        for tk, shares in positions.items():
            price = float(snap.get(tk, 0))
            current_weights[tk] = shares * price / nav if price else 0.0

    ib.disconnect()
    return pd.Series(current_weights, dtype=float), nav


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Orders
# ─────────────────────────────────────────────────────────────────────────────

def compute_orders(target_weights: pd.Series,
                   current_weights: pd.Series,
                   nav: float) -> pd.DataFrame:
    """
    Compute share-level orders from weight delta.
    Returns DataFrame with columns [ticker, side, shares, delta_weight].
    """
    # Align on union of tickers
    all_tickers = target_weights.index.union(current_weights.index)
    target  = target_weights.reindex(all_tickers).fillna(0.0)
    current = current_weights.reindex(all_tickers).fillna(0.0)
    delta   = target - current

    # Fetch last prices for tickers we want to trade
    trade_mask = delta.abs() > MIN_TRADE_BPS / 10_000
    if not trade_mask.any():
        return pd.DataFrame(columns=["ticker", "side", "shares", "delta_weight"])

    tickers_to_trade = delta[trade_mask].index.tolist()
    try:
        prices_snap = yf.download(tickers_to_trade, period="2d",
                                  auto_adjust=True, progress=False)["Close"].iloc[-1]
    except Exception:
        prices_snap = pd.Series(dtype=float)

    orders = []
    for tk in tickers_to_trade:
        dw    = delta[tk]
        price = float(prices_snap.get(tk, 0))
        if price < 1.0:
            continue
        raw_shares = dw * nav / price
        shares     = int(abs(raw_shares))
        if shares == 0:
            continue
        orders.append({
            "ticker":       tk,
            "side":         "BUY" if dw > 0 else "SELL",
            "shares":       shares,
            "delta_weight": round(dw * 100, 2),
        })

    return pd.DataFrame(orders)


def submit_orders(orders_df: pd.DataFrame, nav: float) -> None:
    """Submit MOO (Market on Open) orders via IBKR."""
    if orders_df.empty:
        print("  No orders to submit.")
        return

    try:
        from ib_insync import IB, Stock, Order, util
        util.logToConsole("CRITICAL")
    except ImportError:
        print("  ib_insync not installed.")
        return

    ib = IB()
    ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID + 1, timeout=10)

    submitted = 0
    for _, row in orders_df.iterrows():
        contract = Stock(row["ticker"], "SMART", "USD")
        order    = Order(
            action          = row["side"],
            totalQuantity   = row["shares"],
            orderType       = "MOO",   # Market on Open
            tif             = "OPG",   # At the opening
            outsideRth      = False,
        )
        trade = ib.placeOrder(contract, order)
        ib.sleep(0.1)
        print(f"      {row['side']:4s} {row['shares']:5d} {row['ticker']:<6}  "
              f"(Δw={row['delta_weight']:+.2f}%)  → orderId={trade.order.orderId}")
        submitted += 1

    ib.disconnect()
    print(f"      {submitted} orders submitted.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--submit", action="store_true",
                   help="Actually submit orders to IBKR (default: dry run)")
    p.add_argument("--account-value", type=float, default=None,
                   help="Override account NAV in USD")
    p.add_argument("--skip-ibkr", action="store_true",
                   help="Skip IBKR connection, use zero current positions")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print(f"  Live paper-trading run — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # ── 1-4: Signal pipeline ──────────────────────────────────────────────────
    prices, market = fetch_prices()
    target_weights = compute_target_weights(prices, market)

    # ── 5: IBKR state ─────────────────────────────────────────────────────────
    if args.skip_ibkr:
        print("[5/5] Skipping IBKR (--skip-ibkr flag)")
        current_weights = pd.Series(dtype=float)
        nav = args.account_value or DEFAULT_ACCOUNT
    else:
        print("[5/5] Fetching IBKR positions…")
        current_weights, nav = get_ibkr_state(args.account_value)

    # ── Orders ────────────────────────────────────────────────────────────────
    orders = compute_orders(target_weights, current_weights, nav)

    print("\n════════════════════════════════════════════")
    print(f"  TARGET PORTFOLIO  (NAV=${nav:,.0f})")
    print("════════════════════════════════════════════")
    if orders.empty:
        print("  No trades — portfolio is within tolerance.")
    else:
        print(f"  {'Ticker':<8} {'Side':<5} {'Shares':>6}  {'Δ weight':>8}")
        print("  " + "─" * 34)
        for _, row in orders.iterrows():
            flag = "  ← NEW" if row["ticker"] not in current_weights.index else ""
            print(f"  {row['ticker']:<8} {row['side']:<5} {row['shares']:>6}  "
                  f"{row['delta_weight']:>+7.2f}%{flag}")
        print(f"\n  Total legs: {len(orders)} trades")

    if args.submit:
        print("\n  Submitting MOO orders…")
        submit_orders(orders, nav)
    else:
        print("\n  DRY RUN — rerun with --submit to place orders.")

    # Save target weights for next week's delta computation
    out = Path("outputs/live")
    out.mkdir(parents=True, exist_ok=True)
    target_weights.to_csv(out / "target_weights_latest.csv", header=["weight"])
    print(f"\n  Weights saved → {out}/target_weights_latest.csv")
    print("Done. ✓\n")


if __name__ == "__main__":
    main()
