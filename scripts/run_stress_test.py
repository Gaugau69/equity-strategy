"""
run_stress_test.py
------------------
Amélioration 4 : Stress test sur régimes de marché.

Teste la stratégie sur 5 régimes distincts simulés avec
des caractéristiques de marché réalistes :

  1. Bull market       : tendance haussière forte, faible vol
  2. Bear market       : tendance baissière, vol élevée
  3. Crise (Covid)     : crash brutal -30% puis rebond rapide
  4. Stagflation       : marché flat, rotation sectorielle forte
  5. High volatility   : vol très élevée, pas de tendance claire

Pour chaque régime, on mesure :
  - Sharpe, Return, MaxDD, Hit Rate
  - Factor IC (le signal tient-il dans ce régime ?)
  - Comportement vs marché (corrélation au SPY simulé)
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

from strategy.factors  import compute_factors, cross_sectional_zscore
from strategy.signals  import generate_signals, _panel_to_3d, _composite
from strategy.backtest import run_backtest
from strategy.analytics import compute_metrics, drawdown_series
from strategy import config
from scipy.stats import spearmanr

OUT_DIR = Path("outputs/stress_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_STOCKS = 500   # must be >> 2*TOP_N to avoid market proxy bias
N_DAYS   = 756   # ~3 years per regime


# ─────────────────────────────────────────────────────────────────────────────
# Regime simulators
# ─────────────────────────────────────────────────────────────────────────────

def simulate_regime(regime: str, n_stocks: int = N_STOCKS,
                    n_days: int = N_DAYS, seed: int = 42):
    """
    Simulate prices under a specific market regime.

    Each regime changes:
      - Market drift and volatility
      - Factor premium strength
      - Idiosyncratic noise level
      - Correlation structure
    """
    np.random.seed(seed)
    dates   = pd.bdate_range("2020-01-01", periods=n_days)
    tickers = [f"STK{i:03d}" for i in range(n_stocks)]

    betas     = np.random.uniform(0.5, 1.8, n_stocks)
    true_vols = np.random.uniform(0.010, 0.025, n_stocks)

    def cs_z(x):
        mu, sd = x.mean(), x.std()
        return (x - mu) / sd if sd > 1e-8 else np.zeros_like(x)

    vol_z  = cs_z(true_vols)
    beta_z = cs_z(betas)

    # ── Regime parameters ─────────────────────────────────────────────────
    regimes = {
        "Bull market": {
            "mkt_drift": 0.0006,   # ~15% annual
            "mkt_vol":   0.007,    # low vol
            "idio_scale": 1.0,
            "lambda_mom":  0.00012,
            "lambda_vol": -0.00006,
            "lambda_beta":-0.00004,
            "description": "Strong uptrend, low volatility, momentum works well",
        },
        "Bear market": {
            "mkt_drift": -0.0005,  # ~-12% annual
            "mkt_vol":   0.018,    # high vol
            "idio_scale": 1.5,
            "lambda_mom": -0.00008, # momentum reversal in bear
            "lambda_vol": -0.00010, # low-vol premium stronger
            "lambda_beta":-0.00008,
            "description": "Downtrend, high vol, momentum reversal",
        },
        "Covid crash": {
            "mkt_drift": 0.0001,   # flat overall (crash + recovery)
            "mkt_vol":   0.025,    # extreme vol
            "idio_scale": 2.5,
            "lambda_mom": -0.00005, # momentum destroyed
            "lambda_vol": -0.00015, # low-vol premium very strong
            "lambda_beta":-0.00010,
            "description": "Crash -30% then V-shaped recovery, extreme vol",
            "crash": True,
        },
        "Stagflation": {
            "mkt_drift": 0.00005,  # ~1% annual
            "mkt_vol":   0.012,
            "idio_scale": 1.2,
            "lambda_mom":  0.00004, # weak momentum
            "lambda_vol": -0.00008,
            "lambda_beta":-0.00006,
            "description": "Flat market, sector rotation, weak signals",
        },
        "High volatility": {
            "mkt_drift": 0.0002,
            "mkt_vol":   0.022,    # very high vol
            "idio_scale": 2.0,
            "lambda_mom":  0.00006,
            "lambda_vol": -0.00012,
            "lambda_beta":-0.00008,
            "description": "No clear trend, very high idiosyncratic vol",
        },
    }

    p = regimes[regime]

    # ── Simulate market returns ────────────────────────────────────────────
    market_ret = np.random.normal(p["mkt_drift"], p["mkt_vol"], n_days)

    # Covid crash: sharp drop in days 50-70, recovery days 70-150
    if p.get("crash"):
        market_ret[50:70]  += -0.025   # crash phase
        market_ret[70:150] +=  0.008   # recovery phase

    # ── Simulate stock returns ────────────────────────────────────────────
    all_ret    = np.zeros((n_days, n_stocks))
    price_hist = np.ones((n_days, n_stocks)) * 100.0
    mom_window = 21

    for t in range(n_days):
        idio  = np.random.normal(0, true_vols * p["idio_scale"], n_stocks)
        alpha = p["lambda_vol"] * vol_z + p["lambda_beta"] * beta_z

        if t >= mom_window + 1:
            past_ret = all_ret[t - mom_window : t].sum(axis=0)
            mom_z    = cs_z(past_ret)
            alpha    += p["lambda_mom"] * mom_z

        r = alpha + betas * market_ret[t] + idio
        all_ret[t] = r
        if t == 0:
            price_hist[t] = 100.0 * np.exp(r)
        else:
            price_hist[t] = price_hist[t-1] * np.exp(r)

    prices = pd.DataFrame(price_hist, index=dates, columns=tickers)

    # Use the pure market factor series as proxy (independent of stock selection)
    # This avoids the artefact where market = mean(portfolio stocks)
    market = pd.Series(np.exp(np.cumsum(market_ret)) * 100,
                       index=dates, name="market")

    return prices, betas, market, p["description"]


# ─────────────────────────────────────────────────────────────────────────────
# Run one regime
# ─────────────────────────────────────────────────────────────────────────────

def run_regime(regime_name: str, seed: int = 42) -> dict:
    print(f"\n{'─'*55}")
    print(f"  {regime_name}")
    print(f"{'─'*55}")

    prices, betas, market, desc = simulate_regime(regime_name, seed=seed)
    print(f"  {desc}")

    # Factors & signals
    fraw   = compute_factors(prices, market)
    fnorm  = cross_sectional_zscore(fraw)
    fwd    = prices.pct_change(config.REBAL_FREQ).shift(-config.REBAL_FREQ)

    # Measure factor IC before running backtest
    factor_dict, _, tickers = _panel_to_3d(fnorm)
    dates_test = fnorm.index[130::5][:30]
    ics = []
    for d in dates_test:
        if d not in fwd.index: continue
        comp  = _composite(factor_dict, tickers, d)
        r     = fwd.loc[d].reindex(tickers)
        valid = comp.notna() & r.notna()
        if valid.sum() < 20: continue
        ic, _ = spearmanr(comp[valid], r[valid])
        ics.append(ic)
    mean_ic = np.mean(ics) if ics else float("nan")

    # Generate signals & backtest
    try:
        signals = generate_signals(
            fnorm, fwd,
            train_window = config.TRAIN_WINDOW,
            rebal_freq   = config.REBAL_FREQ,
            alpha        = config.RIDGE_ALPHA,
        )
        if len(signals) < 5:
            raise ValueError("Too few signals")

        result = run_backtest(
            prices, signals, betas,
            transaction_cost_bps = config.TRANSACTION_COST_BPS,
            top_n      = config.TOP_N,
            lambda_reg = config.LAMBDA_REG,
        )
        metrics = result.metrics
        nav     = result.nav
        returns = result.returns

        # Market return for comparison
        mkt_ret   = market.pct_change().dropna()
        mkt_total = (market.iloc[-1] / market.iloc[0] - 1) * 100
        mkt_ann   = ((1 + mkt_ret.mean()) ** 252 - 1) * 100

        # Correlation with market
        common = returns.index.intersection(mkt_ret.index)
        corr   = returns[common].corr(mkt_ret[common]) if len(common) > 10 else float("nan")

        print(f"  IC={mean_ic:+.4f}  Sharpe={metrics['Sharpe Ratio']:+.3f}  "
              f"Ret={metrics['Annualised Return (%)']:+.1f}%  "
              f"DD={metrics['Max Drawdown (%)']:.1f}%  "
              f"mkt_corr={corr:+.3f}")

        return {
            "regime":      regime_name,
            "description": desc,
            "metrics":     metrics,
            "nav":         nav,
            "returns":     returns,
            "mean_ic":     mean_ic,
            "mkt_corr":    corr,
            "mkt_total":   mkt_total,
            "mkt_ann":     mkt_ann,
        }

    except Exception as e:
        print(f"  FAILED: {e}")
        return {"regime": regime_name, "metrics": {}, "nav": None}


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def print_stress_report(results: list[dict]) -> None:
    print("\n" + "═" * 72)
    print("  STRESS TEST REPORT")
    print("═" * 72)
    print(f"  {'Regime':<22} {'Sharpe':>7} {'Ret%':>7} {'MaxDD%':>8} "
          f"{'IC':>7} {'MktCorr':>8} {'vs Mkt':>8}")
    print(f"  {'─'*22} {'─'*7} {'─'*7} {'─'*8} {'─'*7} {'─'*8} {'─'*8}")

    for r in results:
        if not r.get("metrics"):
            print(f"  {r['regime']:<22} {'FAILED':>7}")
            continue
        m      = r["metrics"]
        sr     = m.get("Sharpe Ratio", float("nan"))
        ret    = m.get("Annualised Return (%)", float("nan"))
        dd     = m.get("Max Drawdown (%)", float("nan"))
        ic     = r.get("mean_ic", float("nan"))
        corr   = r.get("mkt_corr", float("nan"))
        mkt    = r.get("mkt_ann", float("nan"))
        alpha  = ret - mkt
        flag   = " ✓" if sr > 0 else " ✗"
        print(f"  {r['regime']:<22} {sr:>+7.3f}{flag} {ret:>+6.1f}% {dd:>+7.1f}% "
              f"{ic:>+7.4f} {corr:>+8.3f} {alpha:>+7.1f}%")

    print("═" * 72)

    sharpes = [r["metrics"].get("Sharpe Ratio", float("nan"))
               for r in results if r.get("metrics")]
    sharpes = [s for s in sharpes if not np.isnan(s)]
    corrs   = [r.get("mkt_corr", float("nan"))
               for r in results if r.get("mkt_corr") is not None]
    corrs   = [c for c in corrs if not np.isnan(c)]

    print(f"\n  ROBUSTNESS ACROSS REGIMES")
    print(f"  Sharpe range     : [{min(sharpes):+.3f}, {max(sharpes):+.3f}]")
    print(f"  % positive Sharpe: {sum(s>0 for s in sharpes)/len(sharpes)*100:.0f}%")
    print(f"  Worst regime     : {results[sharpes.index(min(sharpes))]['regime']}")
    print(f"  Best regime      : {results[sharpes.index(max(sharpes))]['regime']}")
    print(f"  Avg mkt corr     : {np.mean(corrs):+.3f}  (target: |corr| < 0.1)")
    neutral = abs(np.mean(corrs)) < 0.15
    print(f"  Market neutral   : {'YES ✓' if neutral else 'NO ✗'}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_stress(results: list[dict]) -> plt.Figure:
    valid = [r for r in results if r.get("nav") is not None]

    fig = plt.figure(figsize=(16, 10), facecolor="#0d1117")
    gs  = gridspec.GridSpec(3, 3, hspace=0.55, wspace=0.35, figure=fig)

    colors = ["#00d4ff", "#f87171", "#fb923c", "#a78bfa", "#4ade80"]

    # ── 1. NAV per regime ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    for r, col in zip(valid, colors):
        nav = r["nav"]
        ax1.plot(nav.index, nav.values, color=col, lw=1.3,
                 label=f"{r['regime']}  (SR={r['metrics'].get('Sharpe Ratio',0):+.2f})")
    ax1.axhline(1.0, color="#555", lw=0.8, ls="--")
    ax1.set_title("NAV by regime", color="white", fontsize=11)
    ax1.legend(fontsize=8, facecolor="#1a1f2e", labelcolor="white", ncol=3)

    # ── 2. Sharpe bar ────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    names   = [r["regime"].replace(" ", "\n") for r in valid]
    sharpes = [r["metrics"].get("Sharpe Ratio", 0) for r in valid]
    bar_cols = ["#4ade80" if s > 0 else "#f87171" for s in sharpes]
    ax2.barh(range(len(names)), sharpes, color=bar_cols, alpha=0.85)
    ax2.axvline(0, color="#555", lw=0.8)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, color="#888", fontsize=8)
    ax2.set_title("Sharpe ratio", color="white", fontsize=10)

    # ── 3. IC by regime ───────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ics = [r.get("mean_ic", 0) for r in valid]
    ic_cols = ["#4ade80" if ic > 0 else "#f87171" for ic in ics]
    ax3.barh(range(len(names)), ics, color=ic_cols, alpha=0.85)
    ax3.axvline(0, color="#555", lw=0.8)
    ax3.set_yticks(range(len(names)))
    ax3.set_yticklabels(names, color="#888", fontsize=8)
    ax3.set_title("Signal IC", color="white", fontsize=10)

    # ── 4. Market correlation ────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    corrs = [r.get("mkt_corr", 0) for r in valid]
    corr_cols = ["#f87171" if abs(c) > 0.15 else "#4ade80" for c in corrs]
    ax4.barh(range(len(names)), corrs, color=corr_cols, alpha=0.85)
    ax4.axvline(0,     color="#555",  lw=0.8)
    ax4.axvline(0.15,  color="#f87171", lw=0.8, ls="--", alpha=0.5)
    ax4.axvline(-0.15, color="#f87171", lw=0.8, ls="--", alpha=0.5)
    ax4.set_yticks(range(len(names)))
    ax4.set_yticklabels(names, color="#888", fontsize=8)
    ax4.set_title("Mkt correlation", color="white", fontsize=10)

    # ── 5-9. Drawdown per regime ──────────────────────────────────────────────
    for i, (r, col) in enumerate(zip(valid, colors)):
        row = 2
        col_idx = i % 3
        if i >= 3:
            row = 2
            col_idx = i - 3 + (3 - len(valid) % 3 if len(valid) % 3 != 0 else 0)
        ax = fig.add_subplot(gs[2, i % 3]) if i < 3 else None
        if ax is None:
            continue
        dd = drawdown_series(r["nav"]) * 100
        ax.fill_between(dd.index, 0, dd.values, color=col, alpha=0.5)
        ax.set_title(r["regime"], color="white", fontsize=9)
        ax.set_ylabel("DD %", color="#aaa", fontsize=8)

    for ax in fig.get_axes():
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#888", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2f3e")

    fig.suptitle("Stress Test — Performance Across Market Regimes",
                 color="white", fontsize=13, y=1.01)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print(" Amélioration 4 — Stress Test Régimes de Marché")
    print("=" * 55)

    REGIMES = [
        "Bull market",
        "Bear market",
        "Covid crash",
        "Stagflation",
        "High volatility",
    ]

    print(f"\n  Testing {len(REGIMES)} regimes...")
    results = []
    for regime in REGIMES:
        r = run_regime(regime)
        results.append(r)

    print("\n\n[Report]")
    print_stress_report(results)

    # Save metrics table
    rows = []
    for r in results:
        if r.get("metrics"):
            row = {"regime": r["regime"], "IC": r.get("mean_ic"),
                   "mkt_corr": r.get("mkt_corr"), **r["metrics"]}
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT_DIR / "stress_metrics.csv", index=False)

    # Plot
    fig = plot_stress(results)
    fig.savefig(OUT_DIR / "stress_dashboard.png",
                dpi=150, bbox_inches="tight", facecolor="#0d1117")

    print(f"\n  Outputs saved → {OUT_DIR}/")
    print("Done. ✓\n")


if __name__ == "__main__":
    main()