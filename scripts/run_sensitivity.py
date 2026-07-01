"""
run_sensitivity.py
------------------
Amélioration 3 : Sensitivity analysis des hyper-paramètres.

Teste systématiquement l'impact de chaque paramètre clé sur
les métriques de performance, en faisant varier un paramètre
à la fois (ceteris paribus).

Paramètres testés
-----------------
  TOP_N        : taille du livre long/short  [25, 50, 75, 100, 125]
  RIDGE_ALPHA  : régularisation Ridge        [5, 15, 25, 50, 100]
  REBAL_FREQ   : fréquence de rebalancement  [5, 10, 21] jours
  TC_BPS       : coûts de transaction        [5, 10, 20, 30] bps
  TRAIN_WINDOW : fenêtre d'entraînement      [26, 52, 78] semaines

Output
------
  outputs/sensitivity/sensitivity_report.csv
  outputs/sensitivity/sensitivity_dashboard.png
"""

from __future__ import annotations
import sys, warnings, itertools
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from strategy.data_loader  import simulate_universe
from strategy.factors      import compute_factors, cross_sectional_zscore
from strategy.signals      import generate_signals
from strategy.backtest     import run_backtest
from strategy.analytics    import compute_metrics
from strategy              import config

OUT_DIR = Path("outputs/sensitivity")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Baseline (current config) ─────────────────────────────────────────────────
BASELINE = {
    "TOP_N":        config.TOP_N,
    "RIDGE_ALPHA":  config.RIDGE_ALPHA,
    "REBAL_FREQ":   config.REBAL_FREQ,
    "TC_BPS":       config.TRANSACTION_COST_BPS,
    "TRAIN_WINDOW": config.TRAIN_WINDOW,
    "LAMBDA_REG":   config.LAMBDA_REG,
}

# ── Parameter grid (one dimension at a time) ─────────────────────────────────
PARAM_GRID = {
    "TOP_N":        [25, 50, 75, 100, 125],
    "RIDGE_ALPHA":  [5.0, 15.0, 25.0, 50.0, 100.0],
    "REBAL_FREQ":   [5, 10, 21],
    "TC_BPS":       [5.0, 10.0, 20.0, 30.0],
    "TRAIN_WINDOW": [26, 52, 78],
    "LAMBDA_REG":   [5.0, 10.0, 25.0, 50.0, 100.0],
}

KEY_METRICS = ["Sharpe Ratio", "Annualised Return (%)", "Max Drawdown (%)", "Calmar Ratio"]


# ─────────────────────────────────────────────────────────────────────────────
# Single run
# ─────────────────────────────────────────────────────────────────────────────

def single_run(prices, betas, market, params: dict) -> dict:
    """Run full pipeline with given params and return metrics."""
    fwd     = prices.pct_change(params["REBAL_FREQ"]).shift(-params["REBAL_FREQ"])
    fwd_mkt = market.pct_change(params["REBAL_FREQ"]).shift(-params["REBAL_FREQ"])

    # Recompute factors only if needed (always here for simplicity)
    fraw  = compute_factors(prices, market)
    fnorm = cross_sectional_zscore(fraw)

    try:
        signals = generate_signals(
            fnorm, fwd,
            train_window       = params["TRAIN_WINDOW"],
            rebal_freq         = params["REBAL_FREQ"],
            alpha              = params["RIDGE_ALPHA"],
            market_fwd_returns = fwd_mkt,
        )
        if len(signals) < 5:
            return {}

        result = run_backtest(
            prices, signals, betas,
            transaction_cost_bps = params["TC_BPS"],
            top_n      = params["TOP_N"],
            lambda_reg = params["LAMBDA_REG"],
        )
        return result.metrics
    except Exception as e:
        print(f"    run failed: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Sensitivity sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_sensitivity(prices, betas, market) -> pd.DataFrame:
    records = []

    print(f"\n  Baseline params: {BASELINE}")
    print(f"  Total runs: {sum(len(v) for v in PARAM_GRID.values())} + 1 baseline\n")

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("  [baseline]", end=" ", flush=True)
    base_metrics = single_run(prices, betas, market, BASELINE)
    if base_metrics:
        rec = {"param": "baseline", "value": "baseline", **base_metrics}
        records.append(rec)
        print(f"Sharpe={base_metrics.get('Sharpe Ratio', 'N/A'):.3f}")

    # ── Sweep each parameter ──────────────────────────────────────────────────
    for param_name, values in PARAM_GRID.items():
        print(f"\n  Sweeping {param_name}: {values}")
        for val in values:
            # Skip if this is the baseline value
            if val == BASELINE[param_name]:
                print(f"    {param_name}={val} → (baseline, skipping)")
                continue

            params = {**BASELINE, param_name: val}
            print(f"    {param_name}={val}", end=" ", flush=True)
            metrics = single_run(prices, betas, market, params)
            if metrics:
                rec = {"param": param_name, "value": val, **metrics}
                records.append(rec)
                sr = metrics.get("Sharpe Ratio", float("nan"))
                ret = metrics.get("Annualised Return (%)", float("nan"))
                print(f"→ Sharpe={sr:+.3f}  Ret={ret:+.1f}%")
            else:
                print("→ failed")

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def print_sensitivity_report(df: pd.DataFrame, base_metrics: dict) -> None:
    base_sr = base_metrics.get("Sharpe Ratio", 0)

    print("\n" + "═" * 65)
    print("  SENSITIVITY ANALYSIS REPORT")
    print("═" * 65)

    for param_name in PARAM_GRID:
        sub = df[df["param"] == param_name].copy()
        if sub.empty:
            continue

        # Add baseline row
        base_row = pd.DataFrame([{"param": param_name, "value": BASELINE[param_name],
                                   **base_metrics}])
        sub = pd.concat([sub, base_row], ignore_index=True)
        sub["value_num"] = pd.to_numeric(sub["value"], errors="coerce")
        sub = sub.sort_values("value_num")

        print(f"\n  {param_name}  (baseline={BASELINE[param_name]})")
        print(f"  {'Value':<14} {'Sharpe':>8} {'Return%':>9} {'MaxDD%':>8} {'vs Base':>8}")
        print(f"  {'─'*14} {'─'*8} {'─'*9} {'─'*8} {'─'*8}")

        for _, row in sub.iterrows():
            sr  = row.get("Sharpe Ratio", float("nan"))
            ret = row.get("Annualised Return (%)", float("nan"))
            dd  = row.get("Max Drawdown (%)", float("nan"))
            delta = sr - base_sr
            flag  = " ●" if row["value"] == BASELINE[param_name] else ""
            print(f"  {str(row['value']):<14} {sr:>+8.3f} {ret:>+8.1f}% {dd:>+7.1f}%  {delta:>+7.3f}{flag}")

    print("\n  ● = baseline value")
    print("═" * 65)

    # Robustness summary
    all_sharpes = df[df["param"] != "baseline"]["Sharpe Ratio"].dropna()
    print(f"\n  ROBUSTNESS SUMMARY")
    print(f"  Baseline Sharpe     : {base_sr:+.3f}")
    print(f"  Sharpe range        : [{all_sharpes.min():+.3f}, {all_sharpes.max():+.3f}]")
    print(f"  % positive Sharpe   : {(all_sharpes > 0).mean()*100:.0f}%")
    print(f"  Sharpe std across grid : {all_sharpes.std():.3f}")
    robust = all_sharpes.std() < 0.3
    print(f"  Verdict: {'ROBUST ✓' if robust else 'SENSITIVE ✗'} (std < 0.3 = robust)")


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_sensitivity(df: pd.DataFrame, base_metrics: dict) -> plt.Figure:
    params  = list(PARAM_GRID.keys())
    n_params = len(params)
    fig, axes = plt.subplots(1, n_params, figsize=(16, 5), facecolor="#0d1117")
    base_sr = base_metrics.get("Sharpe Ratio", 0)

    for ax, param_name in zip(axes, params):
        sub = df[df["param"] == param_name].copy()
        base_row = pd.DataFrame([{"param": param_name, "value": BASELINE[param_name],
                                   **base_metrics}])
        sub = pd.concat([sub, base_row], ignore_index=True)
        sub["value_num"] = pd.to_numeric(sub["value"], errors="coerce")
        sub = sub.sort_values("value_num")

        values  = sub["value_num"].tolist()
        sharpes = sub["Sharpe Ratio"].tolist()
        colors  = ["#f5a623" if v == BASELINE[param_name] else
                   "#4ade80" if s > 0 else "#f87171"
                   for v, s in zip(values, sharpes)]

        ax.bar(range(len(values)), sharpes, color=colors, alpha=0.85, width=0.6)
        ax.axhline(0,       color="#555", lw=0.8, ls="--")
        ax.axhline(base_sr, color="#f5a623", lw=1.0, ls=":", alpha=0.6)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([str(v) for v in values],
                           color="#888", fontsize=8, rotation=30)
        ax.set_title(param_name, color="white", fontsize=10, pad=6)
        ax.set_ylabel("Sharpe" if param_name == params[0] else "", color="#aaa", fontsize=9)
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#888", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2f3e")

    fig.suptitle("Sensitivity Analysis — Sharpe Ratio vs Parameter Value\n"
                 "(orange = baseline, dotted line = baseline Sharpe)",
                 color="white", fontsize=11, y=1.02)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print(" Amélioration 3 — Sensitivity Analysis")
    print("=" * 55)

    print("\n[1/3] Simulating universe...")
    prices, betas = simulate_universe(
        n_stocks   = config.N_STOCKS,
        n_days     = config.N_DAYS,
        start_date = config.START_DATE,
        seed       = config.RANDOM_SEED,
    )
    market = prices.mean(axis=1)
    print(f"      {prices.shape[1]} stocks × {len(prices)} days")

    print("\n[2/3] Running sensitivity sweep...")
    df = run_sensitivity(prices, betas, market)
    df.to_csv(OUT_DIR / "sensitivity_results.csv", index=False)

    base_row = df[df["param"] == "baseline"]
    base_metrics = base_row.iloc[0].to_dict() if not base_row.empty else {}

    print("\n[3/3] Generating report...")
    print_sensitivity_report(df, base_metrics)

    fig = plot_sensitivity(df, base_metrics)
    fig.savefig(OUT_DIR / "sensitivity_dashboard.png",
                dpi=150, bbox_inches="tight", facecolor="#0d1117")

    print(f"\n  Outputs saved → {OUT_DIR}/")
    print("Done. ✓\n")


if __name__ == "__main__":
    main()