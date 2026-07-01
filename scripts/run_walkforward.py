"""
run_walkforward.py
------------------
Amélioration 2 : Walk-forward out-of-sample validation.

Principe
--------
Au lieu de backtest sur toute la période (risque d'overfitting sur
les hyper-paramètres), on découpe en N fenêtres :

  |<── train ──>|<─ test ─>|
                |<── train ──>|<─ test ─>|
                              |<── train ──>|<─ test ─>|

Chaque fenêtre de test est strictement out-of-sample :
- Les hyper-paramètres (TOP_N, RIDGE_ALPHA, LAMBDA_REG) sont figés
- Seul le modèle Ridge se réentraîne sur chaque fenêtre de train
- Les résultats out-of-sample sont concaténés pour former le NAV final

Paramètres
----------
  TRAIN_YEARS : 2     années d'entraînement par fenêtre
  TEST_YEARS  : 1     année de test out-of-sample
  STEP_YEARS  : 1     pas de glissement entre fenêtres

Avec 6 ans de données (2018-2024) :
  Fenêtre 1 : train 2018-2019  →  test 2020
  Fenêtre 2 : train 2019-2020  →  test 2021
  Fenêtre 3 : train 2020-2021  →  test 2022
  Fenêtre 4 : train 2021-2022  →  test 2023
  Fenêtre 5 : train 2022-2023  →  test 2024  (si assez de données)
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import argparse
from strategy.data_loader   import simulate_universe, load_from_csv
from strategy.factors       import compute_factors, cross_sectional_zscore
from strategy.signals       import generate_signals
from strategy.backtest      import run_backtest
from strategy.analytics     import compute_metrics, print_report, drawdown_series
from strategy               import config

OUT_DIR = Path("outputs/walkforward")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Walk-forward parameters ───────────────────────────────────────────────────
TRAIN_YEARS = 2
TEST_YEARS  = 1
STEP_YEARS  = 1
BDAYS_YEAR  = 252


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward engine
# ─────────────────────────────────────────────────────────────────────────────

def run_walkforward(prices: pd.DataFrame, betas: np.ndarray) -> dict:
    """
    Execute walk-forward validation and return per-window + aggregate results.
    """
    market  = prices.mean(axis=1)
    dates   = prices.index
    n_days  = len(dates)

    train_days = TRAIN_YEARS * BDAYS_YEAR
    test_days  = TEST_YEARS  * BDAYS_YEAR
    step_days  = STEP_YEARS  * BDAYS_YEAR

    windows = []
    start   = 0
    while start + train_days + test_days <= n_days:
        windows.append({
            "train_start": start,
            "train_end":   start + train_days,
            "test_start":  start + train_days,
            "test_end":    min(start + train_days + test_days, n_days),
        })
        start += step_days

    print(f"\n  Walk-forward windows: {len(windows)}")
    for i, w in enumerate(windows):
        ts = dates[w["train_start"]].date()
        te = dates[w["train_end"]-1].date()
        os_s = dates[w["test_start"]].date()
        os_e = dates[w["test_end"]-1].date()
        print(f"  Window {i+1}: train [{ts} → {te}]  test [{os_s} → {os_e}]")

    # ── Run each window ───────────────────────────────────────────────────────
    window_results = []
    oos_returns    = []

    for i, w in enumerate(windows):
        print(f"\n{'─'*55}")
        print(f"  Window {i+1}/{len(windows)}")
        print(f"{'─'*55}")

        # Slice data for this window (train + test)
        prices_w = prices.iloc[w["train_start"] : w["test_end"]]
        market_w = market.iloc[w["train_start"] : w["test_end"]]

        # Compute factors on full window (no look-ahead: factors only use past data)
        print("  Computing factors...")
        fraw_w  = compute_factors(prices_w, market_w)
        fnorm_w = cross_sectional_zscore(fraw_w)
        fwd_w   = prices_w.pct_change(config.REBAL_FREQ).shift(-config.REBAL_FREQ)
        fwd_mkt_w = market_w.pct_change(config.REBAL_FREQ).shift(-config.REBAL_FREQ)

        # Generate signals — training window only sees train data internally
        # (rolling Ridge uses config.TRAIN_WINDOW weeks, which stays within train period)
        print("  Generating signals...")
        signals_w = generate_signals(
            fnorm_w, fwd_w,
            train_window       = config.TRAIN_WINDOW,
            rebal_freq         = config.REBAL_FREQ,
            alpha              = config.RIDGE_ALPHA,
            market_fwd_returns = fwd_mkt_w,
        )

        if len(signals_w) == 0:
            print("  No signals generated, skipping window")
            continue

        # ── In-sample backtest (train period) ─────────────────────────────
        train_end_date  = dates[w["train_end"] - 1]
        test_start_date = dates[w["test_start"]]
        test_end_date   = dates[w["test_end"] - 1]

        is_signals  = signals_w[signals_w.index <= train_end_date]
        oos_signals = signals_w[signals_w.index >= test_start_date]

        if len(is_signals) > 5:
            is_result = run_backtest(
                prices_w[prices_w.index <= train_end_date],
                is_signals,
                betas,
                transaction_cost_bps = config.TRANSACTION_COST_BPS,
                top_n    = config.TOP_N,
                lambda_reg = config.LAMBDA_REG,
            )
        else:
            is_result = None

        # ── Out-of-sample backtest (test period) ──────────────────────────
        if len(oos_signals) > 2:
            oos_result = run_backtest(
                prices_w[prices_w.index >= test_start_date],
                oos_signals,
                betas,
                transaction_cost_bps = config.TRANSACTION_COST_BPS,
                top_n    = config.TOP_N,
                lambda_reg = config.LAMBDA_REG,
            )
            oos_returns.append(oos_result.returns)
        else:
            oos_result = None
            print("  Not enough OOS signals")

        window_results.append({
            "window":     i + 1,
            "train_start": dates[w["train_start"]].date(),
            "train_end":   train_end_date.date(),
            "test_start":  test_start_date.date(),
            "test_end":    test_end_date.date(),
            "is_metrics":  is_result.metrics  if is_result  else {},
            "oos_metrics": oos_result.metrics if oos_result else {},
            "is_nav":      is_result.nav      if is_result  else None,
            "oos_nav":     oos_result.nav     if oos_result else None,
        })

        # Print window summary
        if oos_result:
            m = oos_result.metrics
            print(f"\n  OOS Sharpe : {m['Sharpe Ratio']:+.3f}  |  "
                  f"Return : {m['Annualised Return (%)']:+.1f}%  |  "
                  f"MaxDD : {m['Max Drawdown (%)']:.1f}%")

    # ── Concatenate OOS returns into full OOS NAV ─────────────────────────────
    if not oos_returns:
        raise RuntimeError("No out-of-sample results generated")

    all_oos_ret = pd.concat(oos_returns).sort_index()
    # Remove duplicate dates at window boundaries
    all_oos_ret = all_oos_ret[~all_oos_ret.index.duplicated(keep="first")]
    oos_nav     = pd.Series(
        np.cumprod(1 + all_oos_ret.values),
        index = all_oos_ret.index,
        name  = "OOS NAV",
    )
    oos_metrics = compute_metrics(all_oos_ret, oos_nav)

    return {
        "windows":     window_results,
        "oos_returns": all_oos_ret,
        "oos_nav":     oos_nav,
        "oos_metrics": oos_metrics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_walkforward_report(results: dict) -> None:
    windows = results["windows"]

    print("\n" + "═" * 65)
    print("  WALK-FORWARD REPORT — WINDOW SUMMARY")
    print("═" * 65)
    print(f"  {'Win':<4} {'Test period':<24} {'OOS Sharpe':>11} {'OOS Ret%':>9} {'OOS DD%':>8}")
    print(f"  {'─'*4} {'─'*24} {'─'*11} {'─'*9} {'─'*8}")

    for w in windows:
        m = w["oos_metrics"]
        if not m:
            print(f"  {w['window']:<4} {str(w['test_start'])+' → '+str(w['test_end']):<24} {'N/A':>11}")
            continue
        sr  = m.get("Sharpe Ratio", float("nan"))
        ret = m.get("Annualised Return (%)", float("nan"))
        dd  = m.get("Max Drawdown (%)", float("nan"))
        flag = " ✓" if sr > 0 else " ✗"
        print(f"  {w['window']:<4} {str(w['test_start'])+' → '+str(w['test_end']):<24} "
              f"{sr:>+10.3f}{flag} {ret:>+8.1f}% {dd:>+7.1f}%")

    print("═" * 65)
    m = results["oos_metrics"]
    print(f"\n  AGGREGATE OOS PERFORMANCE")
    print(f"  {'─'*40}")
    for k, v in m.items():
        print(f"  {k:<28}: {v:>10}")
    print(f"  {'═'*40}")

    # IS vs OOS comparison
    is_sharpes  = [w["is_metrics"].get("Sharpe Ratio", None)  for w in windows if w["is_metrics"]]
    oos_sharpes = [w["oos_metrics"].get("Sharpe Ratio", None) for w in windows if w["oos_metrics"]]
    if is_sharpes and oos_sharpes:
        print(f"\n  IS  mean Sharpe : {np.mean(is_sharpes):+.3f}")
        print(f"  OOS mean Sharpe : {np.mean(oos_sharpes):+.3f}")
        ratio = np.mean(oos_sharpes) / np.mean(is_sharpes) if np.mean(is_sharpes) != 0 else 0
        print(f"  OOS/IS ratio    :  {ratio:.2f}  (>0.5 = acceptable, >0.7 = good)")


def plot_walkforward(results: dict) -> plt.Figure:
    fig = plt.figure(figsize=(14, 10), facecolor="#0d1117")
    gs  = gridspec.GridSpec(3, 2, hspace=0.5, wspace=0.35, figure=fig)

    # ── 1. Full OOS NAV ───────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    nav = results["oos_nav"]
    ax1.plot(nav.index, nav.values, color="#00d4ff", lw=1.5, label="OOS NAV")
    ax1.axhline(1.0, color="#555", lw=0.8, ls="--")
    ax1.fill_between(nav.index, 1, nav.values,
                     where=(nav.values >= 1), color="#00d4ff", alpha=0.08)
    ax1.fill_between(nav.index, 1, nav.values,
                     where=(nav.values < 1),  color="#ff4d4d",  alpha=0.12)

    # Shade test windows
    colors = ["#1a3a2a", "#1a2a3a", "#2a1a3a", "#3a2a1a", "#1a3a3a"]
    for i, w in enumerate(results["windows"]):
        if w["oos_nav"] is not None:
            ax1.axvspan(w["oos_nav"].index[0], w["oos_nav"].index[-1],
                        alpha=0.15, color=colors[i % len(colors)],
                        label=f"W{w['window']}")
    ax1.set_title("Out-of-Sample NAV (concatenated)", color="white", fontsize=11)
    ax1.set_ylabel("NAV", color="#aaa")
    ax1.legend(fontsize=7, facecolor="#1a1f2e", labelcolor="white", ncol=6)

    # ── 2. OOS Drawdown ───────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    dd  = drawdown_series(nav) * 100
    ax2.fill_between(dd.index, 0, dd.values, color="#ff4d4d", alpha=0.6)
    ax2.set_title("OOS Drawdown (%)", color="white", fontsize=10)
    ax2.set_ylabel("%", color="#aaa")

    # ── 3. Per-window Sharpe (IS vs OOS) ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ws  = [w for w in results["windows"] if w["is_metrics"] and w["oos_metrics"]]
    x   = np.arange(len(ws))
    is_sr  = [w["is_metrics"].get("Sharpe Ratio", 0)  for w in ws]
    oos_sr = [w["oos_metrics"].get("Sharpe Ratio", 0) for w in ws]
    ax3.bar(x - 0.2, is_sr,  0.35, label="In-sample",     color="#f5a623", alpha=0.8)
    ax3.bar(x + 0.2, oos_sr, 0.35, label="Out-of-sample", color="#00d4ff", alpha=0.8)
    ax3.axhline(0, color="#555", lw=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"W{w['window']}" for w in ws], color="#888", fontsize=8)
    ax3.set_title("Sharpe: In-sample vs OOS", color="white", fontsize=10)
    ax3.legend(fontsize=8, facecolor="#1a1f2e", labelcolor="white")

    # ── 4. OOS Rolling Sharpe ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    oos_ret = results["oos_returns"]
    roll_sr = (oos_ret.rolling(63).mean() / oos_ret.rolling(63).std()) * np.sqrt(252)
    ax4.plot(roll_sr.index, roll_sr.values, color="#f5a623", lw=1.2)
    ax4.axhline(0, color="#555", lw=0.8, ls="--")
    ax4.set_title("Rolling 63d Sharpe (OOS)", color="white", fontsize=10)

    # ── 5. Monthly OOS returns heatmap ────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    monthly = oos_ret.resample("ME").apply(lambda x: (1+x).prod()-1) * 100
    ax5.bar(monthly.index, monthly.values,
            color=["#4ade80" if x > 0 else "#ff4d4d" for x in monthly.values],
            alpha=0.8, width=20)
    ax5.axhline(0, color="#555", lw=0.8)
    ax5.set_title("Monthly OOS Returns (%)", color="white", fontsize=10)

    # Style all axes
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#888", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2f3e")

    fig.suptitle(
        f"Walk-Forward Validation  |  {TRAIN_YEARS}y train / {TEST_YEARS}y test",
        color="white", fontsize=13, y=0.98,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print(" Amélioration 2 — Walk-Forward Out-of-Sample")
    print(f" {TRAIN_YEARS}y train  /  {TEST_YEARS}y test  /  {STEP_YEARS}y step")
    print("=" * 55)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n[1/3] Simulating universe...")
    prices, betas = simulate_universe(
        n_stocks=config.N_STOCKS, n_days=config.N_DAYS,
        start_date=config.START_DATE, seed=config.RANDOM_SEED,
    )
    print(f"      {prices.shape[1]} stocks × {len(prices)} days  "
          f"({prices.index[0].date()} → {prices.index[-1].date()})")

    # ── Walk-forward ──────────────────────────────────────────────────────────
    # Estimate betas if not provided (real data)
    if betas is None:
        daily = prices.pct_change()
        market = prices.mean(axis=1)
        mkt_r = market.pct_change()
        window = min(252, len(daily) - 5)
        r_s = daily.iloc[-window:]
        r_m = mkt_r.iloc[-window:]
        cov  = r_s.subtract(r_s.mean()).multiply(r_m - r_m.mean(), axis=0).mean()
        var  = r_m.var()
        betas = (cov / var).fillna(1.0).values
        print(f"      Betas estimated: mean={betas.mean():.2f}  range=[{betas.min():.2f}, {betas.max():.2f}]")

    print("\n[2/3] Running walk-forward validation...")
    results = run_walkforward(prices, betas)

    # ── Report & plots ────────────────────────────────────────────────────────
    print("\n[3/3] Generating report...")
    print_walkforward_report(results)

    fig = plot_walkforward(results)
    fig.savefig(OUT_DIR / "walkforward_dashboard.png",
                dpi=150, bbox_inches="tight", facecolor="#0d1117")

    # Save OOS returns
    results["oos_nav"].to_csv(OUT_DIR / "oos_nav.csv")
    results["oos_returns"].to_csv(OUT_DIR / "oos_returns.csv")

    # Save window summary
    summary = []
    for w in results["windows"]:
        row = {"window": w["window"],
               "test_start": w["test_start"], "test_end": w["test_end"]}
        row.update({f"oos_{k}": v for k, v in w["oos_metrics"].items()})
        summary.append(row)
    pd.DataFrame(summary).to_csv(OUT_DIR / "window_summary.csv", index=False)

    print(f"\n  Outputs saved → {OUT_DIR}/")
    print("Done. ✓\n")


if __name__ == "__main__":
    main()
