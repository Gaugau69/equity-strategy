"""Tests for factor construction and normalisation."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strategy.factors import (
    compute_factors,
    cross_sectional_zscore,
    orthogonalize_factors,
    _rolling_beta,
)


def _make_prices(n_days: int = 300, n_stocks: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates   = pd.bdate_range("2020-01-01", periods=n_days)
    tickers = [f"S{i:02d}" for i in range(n_stocks)]
    ret     = rng.normal(0.0003, 0.015, (n_days, n_stocks))
    prices  = 100.0 * np.exp(ret.cumsum(axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


def test_compute_factors_shape():
    prices = _make_prices()
    market = prices.mean(axis=1)
    panel  = compute_factors(prices, market)
    factor_names = panel.columns.get_level_values(0).unique().tolist()
    assert set(factor_names) == {"MOM_1M", "MOM_3M", "MOM_12M", "VOL_1M", "BETA"}
    assert panel.shape[0] == len(prices)
    assert panel.shape[1] == 5 * prices.shape[1]


def test_rolling_beta_shape():
    prices = _make_prices()
    ret    = prices.pct_change()
    mkt    = prices.mean(axis=1).pct_change()
    betas  = _rolling_beta(ret, mkt, window=60)
    assert betas.shape == ret.shape
    # Beta should be finite after warm-up
    assert np.isfinite(betas.iloc[65:].values).all()


def test_zscore_cross_sectional_mean_zero():
    prices = _make_prices()
    market = prices.mean(axis=1)
    panel  = compute_factors(prices, market)
    norm   = cross_sectional_zscore(panel)
    for fname in norm.columns.get_level_values(0).unique():
        # Clipping at ±3σ can shift the mean slightly when the distribution is skewed;
        # tolerance of 0.05 is sufficient to catch genuine normalisation failures.
        daily_means = norm[fname].mean(axis=1).dropna()
        assert (daily_means.abs() < 0.05).all(), (
            f"Cross-sectional mean of {fname} too far from 0, got max {daily_means.abs().max():.3f}"
        )


def test_zscore_bounded():
    prices = _make_prices()
    market = prices.mean(axis=1)
    panel  = compute_factors(prices, market)
    norm   = cross_sectional_zscore(panel)
    assert norm.max().max() <=  3.0 + 1e-6, "Z-scores not clipped at +3"
    assert norm.min().min() >= -3.0 - 1e-6, "Z-scores not clipped at -3"


def test_orthogonalize_reduces_correlation():
    """
    With deliberately correlated factor data, orthogonalization should
    reduce average off-diagonal cross-factor correlations.
    """
    rng  = np.random.default_rng(99)
    T, N = 400, 80
    dates   = pd.bdate_range("2020-01-01", periods=T)
    tickers = [f"S{i:02d}" for i in range(N)]

    # Shared factor drives correlations across factor columns
    common  = rng.standard_normal((T, N))
    noise   = rng.standard_normal((T, N))

    factor_names = ["MOM_1M", "MOM_3M", "MOM_12M", "VOL_1M", "BETA"]
    frames: dict = {}
    for i, fname in enumerate(factor_names):
        # Each factor = 0.8 * common + 0.6 * noise → strong cross-factor correlation
        frames[fname] = pd.DataFrame(
            0.8 * common + 0.6 * rng.standard_normal((T, N)),
            index=dates, columns=tickers,
        )

    panel = pd.concat(frames, axis=1)
    orth  = orthogonalize_factors(panel)

    def avg_offdiag_corr(p):
        corrs = []
        for date in p.index[50:100]:
            vals  = np.column_stack([p[f].loc[date].values for f in factor_names])
            valid = np.all(np.isfinite(vals), axis=1)
            if valid.sum() < 10:
                continue
            C   = np.corrcoef(vals[valid].T)
            n   = len(factor_names)
            off = [abs(C[i, j]) for i in range(n) for j in range(n) if i != j]
            corrs.append(np.mean(off))
        return np.mean(corrs) if corrs else 0.0

    before = avg_offdiag_corr(panel)
    after  = avg_offdiag_corr(orth)
    assert after < before, (
        f"Orthogonalization should reduce cross-factor correlations "
        f"(before={before:.3f}, after={after:.3f})"
    )
