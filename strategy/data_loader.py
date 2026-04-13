"""
data_loader.py
--------------
Data ingestion layer.

simulate_universe injects realistic factor premia calibrated to
produce strategy metrics consistent with real-world quant equity:

  Target metrics (post-cost):
    Sharpe ratio     ~  0.8 – 1.5
    Annual return    ~  5  – 15 %
    Max drawdown     ~ -5  – -20%
    IC (Spearman)    ~  0.03 – 0.06

Factor premia are weak relative to idiosyncratic noise, replicating
the low signal-to-noise ratio of real equity markets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


def simulate_universe(
    n_stocks: int = 500,
    n_days: int = 1500,
    start_date: str = "2018-01-01",
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Simulate daily close prices with calibrated, realistic factor premia.

    Data-generating process
    -----------------------
    r_i,t = α_i,t  +  β_i * r_mkt,t  +  ε_i,t

    α_i,t = λ_mom  * z_mom_i,t-1
           + λ_vol  * z_vol_i       (static characteristic)
           + λ_beta * z_beta_i      (static characteristic)
           + η_i,t                  (time-varying alpha noise)

    Calibration targets
    -------------------
    - IC (Spearman rank corr of signal vs fwd return) ≈ 0.04
    - Annual factor spread (top decile - bottom decile) ≈ 4-8 %
    - Idiosyncratic vol dominates: SNR << 1 per stock per week
    """
    np.random.seed(seed)
    dates   = pd.bdate_range(start_date, periods=n_days)
    tickers = [f"STK{i:03d}" for i in range(n_stocks)]

    # ── Static stock characteristics ──────────────────────────────────────
    betas     = np.random.uniform(0.5, 1.8, n_stocks)
    true_vols = np.random.uniform(0.010, 0.028, n_stocks)   # daily idio vol

    # ── Market factor ──────────────────────────────────────────────────────
    # Realistic equity market: ~8% annual return, ~16% vol
    market_ret = np.random.normal(0.0003, 0.010, n_days)

    # ── Pre-compute static z-scores ────────────────────────────────────────
    def cs_z(x: np.ndarray) -> np.ndarray:
        mu, sd = x.mean(), x.std()
        return (x - mu) / sd if sd > 1e-8 else np.zeros_like(x)

    vol_z  = cs_z(true_vols)
    beta_z = cs_z(betas)

    # ── Factor loading calibration ─────────────────────────────────────────
    # Each loading gives annualised top-vs-bottom spread ≈ loading * 2σ * √252
    # Targeting ~2-3% annual spread per factor → daily loading ≈ 0.00008-0.00015
    LAMBDA_MOM  =  0.00010   # momentum  (positive: winners outperform)
    LAMBDA_VOL  = -0.00008   # low-vol   (negative: low-vol outperforms)
    LAMBDA_BETA = -0.00006   # low-beta  (negative: low-beta outperforms)

    # ── Alpha noise: each stock has a persistent slow-moving alpha component
    # This creates realistic factor timing noise
    alpha_noise_vol = 0.00015   # daily std of time-varying alpha noise
    alpha_noise = np.random.normal(0, alpha_noise_vol, (n_days, n_stocks))
    # AR(1) smoothing to make it persistent (weekly autocorrelation ~0.7)
    ar_coef = 0.85
    for t in range(1, n_days):
        alpha_noise[t] = ar_coef * alpha_noise[t-1] + np.sqrt(1 - ar_coef**2) * alpha_noise[t]

    # ── Simulate returns ───────────────────────────────────────────────────
    all_ret = np.zeros((n_days, n_stocks))
    prices  = np.ones(n_stocks) * 100.0
    price_hist = np.zeros((n_days, n_stocks))

    mom_window = 21   # 1-month momentum lookback

    for t in range(n_days):
        # Idiosyncratic shock — dominates alpha (realistic SNR)
        idio = np.random.normal(0, true_vols, n_stocks)

        # Factor alpha
        alpha = LAMBDA_VOL * vol_z + LAMBDA_BETA * beta_z + alpha_noise[t]

        # Momentum alpha (only after enough history)
        if t >= mom_window + 1:
            past_ret_sum = all_ret[t - mom_window : t].sum(axis=0)
            mom_z = cs_z(past_ret_sum)
            alpha = alpha + LAMBDA_MOM * mom_z

        r = alpha + betas * market_ret[t] + idio
        all_ret[t] = r
        prices = prices * np.exp(r)
        price_hist[t] = prices

    df_prices = pd.DataFrame(price_hist, index=dates, columns=tickers)
    return df_prices, betas


def load_from_csv(path: str | Path) -> pd.DataFrame:
    """Load prices from a CSV with date index and ticker columns."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df.astype(float)


def load_from_yfinance(
    tickers: list[str],
    start: str = "2018-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance (requires yfinance)."""
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("pip install yfinance") from e
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    return raw["Close"].dropna(how="all")