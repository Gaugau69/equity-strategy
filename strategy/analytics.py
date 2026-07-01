"""
analytics.py
------------
Performance analytics: metrics, drawdown, and matplotlib charts.

Public API
----------
compute_metrics(returns, nav)  → dict of scalar KPIs
plot_results(result)           → matplotlib Figure
print_report(metrics)          → formatted console output
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

ANN = 252   # trading days per year


def compute_metrics(returns: pd.Series, nav: pd.Series) -> dict[str, float]:
    """
    Compute standard quantitative finance performance metrics.

    Parameters
    ----------
    returns : daily strategy returns (net of transaction costs)
    nav     : cumulative NAV series (starts at 1.0)

    Returns
    -------
    dict of labelled scalar metrics
    """
    mu  = returns.mean() * ANN
    vol = returns.std()  * np.sqrt(ANN)
    sr  = mu / vol if vol > 0 else 0.0

    rolling_max = nav.cummax()
    dd          = (nav - rolling_max) / rolling_max
    max_dd      = dd.min()

    calmar    = mu / abs(max_dd) if max_dd != 0 else 0.0
    hit_rate  = (returns > 0).mean()
    skew      = returns.skew()
    kurt      = returns.kurt()
    total_ret = nav.iloc[-1] - 1

    return {
        "Total Return (%)":     round(total_ret * 100, 2),
        "Annualised Return (%)":round(mu   * 100, 2),
        "Annualised Vol (%)":   round(vol  * 100, 2),
        "Sharpe Ratio":         round(sr,          3),
        "Max Drawdown (%)":     round(max_dd * 100, 2),
        "Calmar Ratio":         round(calmar,       3),
        "Hit Rate (%)":         round(hit_rate * 100, 2),
        "Skewness":             round(skew,          3),
        "Excess Kurtosis":      round(kurt,          3),
        "Total Trading Days":   len(returns),
    }


def drawdown_series(nav: pd.Series) -> pd.Series:
    """Return the drawdown series from peak."""
    return (nav - nav.cummax()) / nav.cummax()


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(result) -> plt.Figure:
    """
    Generate a 4-panel performance dashboard.

    Panels
    ------
    1. Cumulative NAV
    2. Drawdown
    3. Rolling 63-day Sharpe ratio
    4. Rebalancing turnover

    Parameters
    ----------
    result : BacktestResult dataclass

    Returns
    -------
    matplotlib Figure
    """
    fig = plt.figure(figsize=(14, 10), facecolor="#0d1117")
    gs  = gridspec.GridSpec(4, 1, hspace=0.45, figure=fig)

    axes = [fig.add_subplot(gs[i]) for i in range(4)]
    _style_axes(axes)

    nav      = result.nav
    returns  = result.returns
    dd       = drawdown_series(nav)
    turnover = result.turnover

    # ── 1. NAV ─────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(nav.index, nav.values, color="#00d4ff", lw=1.5, label="Strategy NAV")
    ax.axhline(1.0, color="#555", lw=0.8, ls="--")
    ax.fill_between(nav.index, 1, nav.values,
                    where=(nav.values >= 1), color="#00d4ff", alpha=0.08)
    ax.fill_between(nav.index, 1, nav.values,
                    where=(nav.values < 1),  color="#ff4d4d",  alpha=0.12)
    ax.set_title("Cumulative NAV", color="white", fontsize=11, pad=6)
    ax.set_ylabel("NAV", color="#aaa")
    ax.legend(fontsize=8, facecolor="#1a1f2e", labelcolor="white")

    # ── 2. Drawdown ────────────────────────────────────────────────────────
    ax = axes[1]
    ax.fill_between(dd.index, 0, dd.values * 100, color="#ff4d4d", alpha=0.6)
    ax.set_title("Drawdown (%)", color="white", fontsize=11, pad=6)
    ax.set_ylabel("%", color="#aaa")

    # ── 3. Rolling Sharpe (63-day) ─────────────────────────────────────────
    ax = axes[2]
    roll_std = returns.rolling(63).std().replace(0, np.nan)
    roll_sr  = (returns.rolling(63).mean() / roll_std) * np.sqrt(ANN)
    ax.plot(roll_sr.index, roll_sr.values, color="#f5a623", lw=1.2)
    ax.axhline(0, color="#555", lw=0.8, ls="--")
    ax.set_title("Rolling 63-day Sharpe Ratio", color="white", fontsize=11, pad=6)
    ax.set_ylabel("Sharpe", color="#aaa")

    # ── 4. Turnover ────────────────────────────────────────────────────────
    ax = axes[3]
    ax.bar(turnover.index, turnover.values * 100,
           width=4, color="#7b61ff", alpha=0.7)
    ax.set_title("Rebalancing Turnover (%)", color="white", fontsize=11, pad=6)
    ax.set_ylabel("%", color="#aaa")

    fig.suptitle(
        "Systematic Equity Market-Neutral Strategy — Backtest",
        color="white", fontsize=13, y=0.98,
    )

    return fig


def _style_axes(axes: list[plt.Axes]) -> None:
    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#888", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2f3e")


# ─────────────────────────────────────────────────────────────────────────────
# Console report
# ─────────────────────────────────────────────────────────────────────────────

def print_report(metrics: dict[str, float]) -> None:
    """Print a formatted performance summary to stdout."""
    width = 44
    print("\n" + "═" * width)
    print(" PERFORMANCE SUMMARY".center(width))
    print("═" * width)
    for k, v in metrics.items():
        print(f"  {k:<28}{v:>10}")
    print("═" * width + "\n")
