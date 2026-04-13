"""
backtest.py
-----------
Full backtesting engine with tqdm progress bar.

For each rebalancing date the engine:
  1. Estimates a rolling covariance matrix.
  2. Calls `build_portfolio` to get new weights.
  3. Computes daily P&L over the holding period.
  4. Deducts one-way transaction costs at each rebalance.

All results are returned in a structured `BacktestResult` dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from .portfolio import build_portfolio
from .analytics import compute_metrics
from .config import TRANSACTION_COST_BPS, TOP_N, LAMBDA_REG, COV_LOOKBACK


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    nav             : pd.Series
    returns         : pd.Series
    weights_history : dict[pd.Timestamp, pd.Series]
    metrics         : dict[str, float]
    turnover        : pd.Series


# ─────────────────────────────────────────────────────────────────────────────
# Backtest engine
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    betas_true: np.ndarray,
    transaction_cost_bps: float = TRANSACTION_COST_BPS,
    top_n: int = TOP_N,
    lambda_reg: float = LAMBDA_REG,
    cov_lookback: int = COV_LOOKBACK,
) -> BacktestResult:
    """
    Simulate the strategy over the full backtest period.

    Parameters
    ----------
    prices               : DataFrame (dates × tickers) of close prices
    signals              : DataFrame (rebal_dates × tickers) of alpha scores
    betas_true           : array (n_stocks,) of market betas
    transaction_cost_bps : one-way cost in basis points
    top_n                : long / short leg size
    lambda_reg           : Markowitz regularisation
    cov_lookback         : rolling covariance window in days

    Returns
    -------
    BacktestResult dataclass
    """
    tickers       = list(prices.columns)
    tc            = transaction_cost_bps / 10_000
    daily_returns = prices.pct_change()
    beta_series   = pd.Series(betas_true, index=tickers)

    port_records    : list[dict]                        = []
    weights_history : dict[pd.Timestamp, pd.Series]    = {}
    turnover_records: dict[pd.Timestamp, float]         = {}
    prev_weights = pd.Series(0.0, index=tickers)

    rebal_dates = signals.index
    total       = len(rebal_dates) - 1   # last date has no next period

    # ── Progress bar ──────────────────────────────────────────────────────────
    if HAS_TQDM:
        iterator = tqdm(
            enumerate(rebal_dates),
            total=total,
            desc="  Backtesting",
            unit="rebal",
            ncols=80,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
    else:
        # Fallback: plain print every 10 %
        iterator = enumerate(rebal_dates)

    last_pct = -1   # for fallback progress

    for i, t_date in iterator:
        if i + 1 >= len(rebal_dates):
            break

        # ── Fallback progress (no tqdm) ────────────────────────────────────
        if not HAS_TQDM:
            pct = int(100 * i / total)
            if pct % 10 == 0 and pct != last_pct:
                bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                print(f"  Backtesting  [{bar}] {pct:3d}%  ({i}/{total})", flush=True)
                last_pct = pct

        next_date = rebal_dates[i + 1]

        # ── Rolling covariance ─────────────────────────────────────────────
        t_idx      = prices.index.get_loc(t_date)
        cov_slice  = daily_returns.iloc[max(0, t_idx - cov_lookback) : t_idx]
        cov_matrix = cov_slice.cov().fillna(0).values * 252

        # ── Portfolio weights ──────────────────────────────────────────────
        sig_t       = signals.loc[t_date].reindex(tickers)
        new_weights = build_portfolio(
            sig_t,
            beta_series,
            cov_matrix,
            tickers,
            top_n=top_n,
            lambda_reg=lambda_reg,
        )

        # ── Turnover ───────────────────────────────────────────────────────
        to = (new_weights - prev_weights).abs().sum()
        turnover_records[t_date] = to

        # ── Daily P&L ─────────────────────────────────────────────────────
        mask     = (prices.index > t_date) & (prices.index <= next_date)
        hold_idx = prices.index[mask]

        for j, d in enumerate(hold_idx):
            gross_pnl    = (new_weights * daily_returns.loc[d]).sum()
            tc_deduction = (to * tc) if j == 0 else 0.0
            port_records.append({"date": d, "return": gross_pnl - tc_deduction})

        weights_history[t_date] = new_weights
        prev_weights = new_weights.copy()

    # ── Final print for fallback ───────────────────────────────────────────
    if not HAS_TQDM:
        bar = "█" * 20
        print(f"  Backtesting  [{bar}] 100%  ({total}/{total})", flush=True)

    # ── Assemble series ────────────────────────────────────────────────────
    ret_df  = pd.DataFrame(port_records).set_index("date")
    ret_ser = ret_df["return"]
    nav     = pd.Series(
        np.cumprod(1 + ret_ser.values),
        index=ret_ser.index,
        name="NAV",
    )
    turnover_ser = pd.Series(turnover_records, name="Turnover")

    return BacktestResult(
        nav             = nav,
        returns         = ret_ser,
        weights_history = weights_history,
        metrics         = compute_metrics(ret_ser, nav),
        turnover        = turnover_ser,
    )