"""
run_strategy.py
---------------
Command-line entry point.

Usage
-----
    python run_strategy.py                      # default: simulated data
    python run_strategy.py --csv data/prices.csv
    python run_strategy.py --tc 5 --top_n 50 --lambda 3.0
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from strategy.data_loader import simulate_universe, load_from_csv
from strategy.factors     import compute_factors, cross_sectional_zscore
from strategy.signals     import generate_signals
from strategy.backtest    import run_backtest
from strategy.analytics   import print_report, plot_results
from strategy import config


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Systematic Equity Market-Neutral Strategy"
    )
    p.add_argument("--csv",    type=str,   default=None,
                   help="Path to CSV price file (date index, ticker columns)")
    p.add_argument("--tc",     type=float, default=config.TRANSACTION_COST_BPS,
                   help="Transaction cost in bps (default: 10)")
    p.add_argument("--top_n",  type=int,   default=config.TOP_N,
                   help="Long/short leg size (default: 100)")
    p.add_argument("--lam",    type=float, default=config.LAMBDA_REG,
                   help="Markowitz regularisation λ (default: 5.0)")
    p.add_argument("--no_plot",action="store_true",
                   help="Skip chart output")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── 1. Data ───────────────────────────────────────────────────────────
    print("\n[1/5] Loading data…")
    if args.csv:
        prices     = load_from_csv(args.csv)
        true_betas = pd.Series(1.0, index=prices.columns).values
        print(f"      Loaded {prices.shape[1]} stocks from {args.csv}")
    else:
        prices, true_betas = simulate_universe(
            n_stocks   = config.N_STOCKS,
            n_days     = config.N_DAYS,
            start_date = config.START_DATE,
            seed       = config.RANDOM_SEED,
        )
        print(f"      Simulated {prices.shape[1]} stocks × {len(prices)} days")

    market_prices = prices.mean(axis=1)

    # ── 2. Factors ────────────────────────────────────────────────────────
    print("\n[2/5] Computing factors…")
    factors_raw  = compute_factors(prices, market_prices)
    factors_norm = cross_sectional_zscore(factors_raw)
    print(f"      Panel: {factors_norm.shape}  |  factors: {list(factors_norm.columns.get_level_values(0).unique())}")

    # ── 3. Forward returns & signals ──────────────────────────────────────
    print("\n[3/5] Generating signals (rolling Ridge)…")
    fwd_returns = prices.pct_change(config.REBAL_FREQ).shift(-config.REBAL_FREQ)
    fwd_market  = market_prices.pct_change(config.REBAL_FREQ).shift(-config.REBAL_FREQ)
    signals     = generate_signals(
        factors_norm, fwd_returns,
        train_window       = config.TRAIN_WINDOW,
        rebal_freq         = config.REBAL_FREQ,
        alpha              = config.RIDGE_ALPHA,
        market_fwd_returns = fwd_market,
    )
    print(f"      {len(signals)} rebalancing dates generated")

    # ── 4. Backtest ───────────────────────────────────────────────────────
    print("\n[4/5] Running backtest…")
    result = run_backtest(
        prices,
        signals,
        true_betas,
        transaction_cost_bps = args.tc,
        top_n                = args.top_n,
        lambda_reg           = args.lam,
    )

    # ── 5. Output ─────────────────────────────────────────────────────────
    print("\n[5/5] Results")
    print_report(result.metrics)

    # Save NAV
    out_dir = config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    result.nav.to_csv(out_dir / "nav_series.csv", header=True)
    print(f"  NAV saved → {out_dir / 'nav_series.csv'}")

    # Chart
    if not args.no_plot:
        fig = plot_results(result)
        chart_path = out_dir / "backtest_dashboard.png"
        fig.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        print(f"  Chart saved → {chart_path}")

    print("\nDone. ✓\n")


if __name__ == "__main__":
    main()