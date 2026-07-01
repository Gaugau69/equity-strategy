"""Tests for performance metrics and analytics."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strategy.analytics import compute_metrics, drawdown_series


def _flat_nav(n: int = 252) -> tuple[pd.Series, pd.Series]:
    dates   = pd.bdate_range("2020-01-01", periods=n)
    returns = pd.Series(0.0, index=dates)
    nav     = pd.Series(1.0, index=dates)
    return returns, nav


def _positive_nav(n: int = 252) -> tuple[pd.Series, pd.Series]:
    dates   = pd.bdate_range("2020-01-01", periods=n)
    returns = pd.Series(0.001, index=dates)
    nav     = pd.Series(np.cumprod(1 + returns.values), index=dates)
    return returns, nav


def test_metrics_keys():
    returns, nav = _positive_nav()
    m = compute_metrics(returns, nav)
    expected = {
        "Total Return (%)", "Annualised Return (%)", "Annualised Vol (%)",
        "Sharpe Ratio", "Max Drawdown (%)", "Calmar Ratio",
        "Hit Rate (%)", "Skewness", "Excess Kurtosis", "Total Trading Days",
    }
    assert set(m.keys()) == expected


def test_flat_portfolio_sharpe_zero():
    returns, nav = _flat_nav()
    m = compute_metrics(returns, nav)
    assert m["Sharpe Ratio"] == 0.0
    assert m["Max Drawdown (%)"] == 0.0
    assert m["Total Return (%)"] == 0.0


def test_positive_return_metrics():
    returns, nav = _positive_nav()
    m = compute_metrics(returns, nav)
    assert m["Annualised Return (%)"] > 0
    assert m["Sharpe Ratio"] > 0
    assert m["Max Drawdown (%)"] == 0.0  # monotone NAV has no drawdown
    assert m["Hit Rate (%)"] == 100.0


def test_drawdown_series_non_positive():
    _, nav = _positive_nav()
    dd = drawdown_series(nav)
    assert (dd <= 0).all(), "Drawdown must always be ≤ 0"
    assert dd.iloc[0] == 0.0, "Drawdown starts at 0 for a new high"


def test_drawdown_series_crash():
    dates = pd.bdate_range("2020-01-01", periods=10)
    nav   = pd.Series([1.0, 1.1, 1.2, 1.0, 0.9, 0.95, 1.1, 1.3, 1.2, 1.4], index=dates)
    dd    = drawdown_series(nav)
    # Max drawdown should be at index where nav=0.9 relative to peak 1.2
    worst_idx = dd.idxmin()
    assert abs(dd[worst_idx] - (0.9 - 1.2) / 1.2) < 1e-10


def test_total_return_consistent_with_nav():
    returns, nav = _positive_nav(252)
    m = compute_metrics(returns, nav)
    expected_total = round((nav.iloc[-1] - 1) * 100, 2)  # metric is rounded to 2dp
    assert abs(m["Total Return (%)"] - expected_total) < 1e-6
