"""Tests for portfolio construction invariants."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strategy.portfolio import build_portfolio
from strategy.config import TOP_N, MAX_WEIGHT, LAMBDA_REG


N = 100   # small universe for fast tests


def _make_inputs(seed: int = 0):
    rng = np.random.default_rng(seed)
    tickers = [f"S{i:03d}" for i in range(N)]
    signals = pd.Series(rng.standard_normal(N), index=tickers)
    betas   = pd.Series(rng.uniform(0.5, 1.8, N), index=tickers)
    cov     = np.eye(N) * 0.04   # diagonal 20%-vol cov
    return signals, betas, cov, tickers


def test_net_exposure_bounded():
    """
    After beta neutralisation the legs are asymmetric in dollar size.
    The net should remain small (bounded by the beta differential), not zero.
    """
    signals, betas, cov, tickers = _make_inputs()
    w = build_portfolio(signals, betas, cov, tickers, top_n=20)
    net = abs(w.sum())
    assert net < 0.5, f"Net dollar exposure too large: {net:.4f}"


def test_gross_exposure():
    signals, betas, cov, tickers = _make_inputs()
    w = build_portfolio(signals, betas, cov, tickers, top_n=20)
    gross = w.abs().sum()
    assert abs(gross - 1.0) < 1e-4, f"Gross exposure should be ~1.0, got {gross:.4f}"


def test_weight_cap():
    signals, betas, cov, tickers = _make_inputs()
    w = build_portfolio(signals, betas, cov, tickers, top_n=20, max_weight=MAX_WEIGHT)
    assert w.abs().max() <= MAX_WEIGHT + 1e-6, (
        f"Weight cap violated: max |w| = {w.abs().max():.4f} > {MAX_WEIGHT}"
    )


def test_long_short_structure():
    """Top signals must be long, bottom signals must be short."""
    signals, betas, cov, tickers = _make_inputs()
    top_n = 10
    w = build_portfolio(signals, betas, cov, tickers, top_n=top_n)
    long_tickers  = set(signals.nlargest(top_n).index)
    short_tickers = set(signals.nsmallest(top_n).index)
    assert all(w[t] >= -1e-6 for t in long_tickers),  "Long leg has negative weights"
    assert all(w[t] <=  1e-6 for t in short_tickers), "Short leg has positive weights"


def test_insufficient_signals_returns_zeros():
    signals = pd.Series({"A": 1.0, "B": 2.0})  # fewer than 2*top_n
    betas   = pd.Series({"A": 1.0, "B": 1.0})
    cov     = np.eye(2) * 0.04
    w = build_portfolio(signals, betas, cov, ["A", "B"], top_n=5)
    assert (w == 0.0).all(), "Should return all-zero weights when universe too small"


def test_nan_signals_handled():
    signals, betas, cov, tickers = _make_inputs(seed=1)
    signals.iloc[:10] = np.nan   # inject NaNs
    w = build_portfolio(signals, betas, cov, tickers, top_n=15)
    assert np.isfinite(w.values).all(), "Weights contain NaN/Inf with NaN signals"


def test_beta_neutrality_approximate():
    """Portfolio beta should be close to zero."""
    rng = np.random.default_rng(42)
    tickers = [f"S{i:03d}" for i in range(N)]
    signals = pd.Series(rng.standard_normal(N), index=tickers)
    betas   = pd.Series(np.linspace(0.5, 1.8, N), index=tickers)
    cov     = np.eye(N) * 0.04
    w = build_portfolio(signals, betas, cov, tickers, top_n=20, lambda_reg=LAMBDA_REG)
    port_beta = float(betas @ w)
    assert abs(port_beta) < 0.15, f"Portfolio beta too large: {port_beta:.4f}"
