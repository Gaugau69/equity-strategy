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
    betas = pd.DataFrame(index=stock_ret.index, columns=stock_ret.columns, dtype=float)

    for t in range(window, len(stock_ret)):
        r_s = stock_ret.iloc[t - window : t]
        r_m = mkt_ret.iloc[t - window : t]

        cov_xy = r_s.subtract(r_s.mean()).multiply(r_m - r_m.mean(), axis=0).mean()
        var_x  = r_m.var()
        betas.iloc[t] = cov_xy / var_x if var_x > 0 else 1.0

    return betas


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────

def cross_sectional_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise each factor cross-sectionally at every date:

        1. Winsorise at ±3 σ
        2. Subtract cross-sectional mean and divide by cross-sectional std

    Parameters
    ----------
    panel : MultiIndex-column DataFrame as returned by `compute_factors`

    Returns
    -------
    normalised panel with same shape
    """
    out = panel.copy()

    # panel has MultiIndex columns: (factor_name, ticker)
    # iterate over the top-level factor names
    factor_names = out.columns.get_level_values(0).unique()

    for fname in factor_names:
        s  = out[fname]          # DataFrame: dates × tickers
        mu = s.mean(axis=1)      # Series: one mean per date
        sd = s.std(axis=1)       # Series: one std per date

        z = (
            s.subtract(mu, axis=0)
             .divide(sd.replace(0, np.nan), axis=0)
             .clip(-3, 3)
        )
        out[fname] = z

    return out