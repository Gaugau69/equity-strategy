"""
factors.py
----------
Cross-sectional factor construction and normalisation.

Factors implemented
-------------------
MOM_1M   : 1-month price momentum  (21-day return)
MOM_3M   : 3-month price momentum  (63-day return)
MOM_12M  : 12-month momentum, skip last month  (231-day return, lagged 21d)
VOL_1M   : 1-month realised volatility (21-day std of daily returns)
BETA     : 60-day rolling OLS market beta
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    MOM_SHORT_WINDOW,
    MOM_MID_WINDOW,
    MOM_LONG_WINDOW,
    MOM_SKIP_WINDOW,
    VOL_WINDOW,
    BETA_WINDOW,
)


# ─────────────────────────────────────────────────────────────────────────────
# Raw factor computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_factors(
    prices: pd.DataFrame,
    market_prices: pd.Series,
) -> pd.DataFrame:
    """
    Compute raw factor values for every stock on every date.

    Parameters
    ----------
    prices        : DataFrame (dates × tickers) of adjusted close prices
    market_prices : Series (dates,) used as the market return proxy

    Returns
    -------
    panel : DataFrame (dates × tickers) with MultiIndex columns
            (factor_name, ticker)
    """
    ret     = prices.pct_change()
    mkt_ret = market_prices.pct_change()

    factors: dict[str, pd.DataFrame] = {}

    # ── Momentum ──────────────────────────────────────────────────────────────
    factors["MOM_1M"]  = prices.pct_change(MOM_SHORT_WINDOW)
    factors["MOM_3M"]  = prices.pct_change(MOM_MID_WINDOW)
    factors["MOM_12M"] = prices.shift(MOM_SKIP_WINDOW).pct_change(MOM_LONG_WINDOW)

    # ── Volatility ────────────────────────────────────────────────────────────
    factors["VOL_1M"]  = ret.rolling(VOL_WINDOW).std()

    # ── Rolling Beta ──────────────────────────────────────────────────────────
    betas = _rolling_beta(ret, mkt_ret, window=BETA_WINDOW)
    factors["BETA"] = betas

    panel = pd.concat(factors, axis=1)
    return panel


def _rolling_beta(
    stock_ret: pd.DataFrame,
    mkt_ret: pd.Series,
    window: int,
) -> pd.DataFrame:
    """
    Compute rolling OLS beta (stock vs. market) using the covariance formula:

        β_i = Cov(r_i, r_mkt) / Var(r_mkt)

    Parameters
    ----------
    stock_ret : DataFrame of daily stock returns
    mkt_ret   : Series of daily market returns
    window    : lookback window in days

    Returns
    -------
    betas : DataFrame aligned with stock_ret
    """
    roll_cov = stock_ret.rolling(window).cov(mkt_ret)
    roll_var = mkt_ret.rolling(window).var()
    betas = roll_cov.div(roll_var, axis=0)
    # Where variance is zero or missing, fall back to beta = 1.0
    betas = betas.where(roll_var > 0, 1.0)
    return betas


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────

def cross_sectional_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise each factor cross-sectionally at every date:

        1. Subtract cross-sectional mean and divide by cross-sectional std
        2. Winsorise at ±3 σ  (clip after z-scoring)

    Parameters
    ----------
    panel : MultiIndex-column DataFrame as returned by `compute_factors`

    Returns
    -------
    normalised panel with same shape
    """
    out = panel.copy()

    factor_names = out.columns.get_level_values(0).unique()

    for fname in factor_names:
        s  = out[fname]
        mu = s.mean(axis=1)
        sd = s.std(axis=1)

        z = (
            s.subtract(mu, axis=0)
             .divide(sd.replace(0, np.nan), axis=0)
             .clip(-3, 3)
        )
        out[fname] = z

    return out


def orthogonalize_factors(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Remove cross-factor linear dependencies by cross-sectional residualisation.

    For each factor f, regress it on all other factors cross-sectionally at
    every date and replace it with the residual.  This reduces the
    multicollinearity that degrades Ridge regression coefficients when BETA
    and VOL_1M are correlated.

    Parameters
    ----------
    panel : normalised MultiIndex-column DataFrame (output of cross_sectional_zscore)

    Returns
    -------
    orthogonalised panel with same shape
    """
    factor_names = panel.columns.get_level_values(0).unique().tolist()
    if len(factor_names) < 2:
        return panel

    out = panel.copy()

    for fname in factor_names:
        others = [f for f in factor_names if f != fname]
        y_df   = panel[fname].values.astype(float)        # (T, N)
        X_mats = [panel[f].values.astype(float) for f in others]  # list of (T, N)
        T, N   = y_df.shape
        K      = len(others)

        resid = y_df.copy()
        for t in range(T):
            y = y_df[t]
            # Design matrix: intercept + other factors (columns = observations)
            X = np.column_stack([np.ones(N)] + [X_mats[k][t] for k in range(K)])
            valid = np.isfinite(y) & np.all(np.isfinite(X[:, 1:]), axis=1)
            if valid.sum() < K + 5:
                continue
            try:
                beta, _, _, _ = np.linalg.lstsq(X[valid], y[valid], rcond=None)
                resid[t] = y - X @ beta
            except np.linalg.LinAlgError:
                pass

        out[fname] = pd.DataFrame(
            resid,
            index=panel.index,
            columns=panel[fname].columns,
        )

    return out