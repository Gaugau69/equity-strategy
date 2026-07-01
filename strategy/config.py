"""
config.py
---------
Central configuration for the Systematic Equity Market-Neutral Strategy.
All hyper-parameters and paths live here — edit this file only.
"""

from pathlib import Path

# ── Project paths ─────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parents[1]
DATA_DIR    = ROOT_DIR / "data"
OUTPUT_DIR  = ROOT_DIR / "outputs"

# ── Universe / simulation ─────────────────────────────────────────────────────
N_STOCKS    : int   = 500
N_DAYS      : int   = 1500
START_DATE  : str   = "2018-01-01"
RANDOM_SEED : int   = 42

# ── Factor construction ───────────────────────────────────────────────────────
MOM_SHORT_WINDOW  : int = 21    # 1-month momentum (days)
MOM_MID_WINDOW    : int = 63    # 3-month momentum (days)
MOM_LONG_WINDOW   : int = 231   # 12-month momentum, skip-1-month (days)
MOM_SKIP_WINDOW   : int = 21    # skip period for long momentum
VOL_WINDOW        : int = 21    # realised volatility window (days)
BETA_WINDOW       : int = 60    # rolling beta estimation window (days)
REV_WINDOW        : int = 5     # short-term reversal window (days)
HIGH52W_WINDOW    : int = 252   # 52-week high lookback (trading days)
RVOL_WINDOW       : int = 20    # relative volume normalization window (days)

# ── Signal generation (rolling Ridge) ────────────────────────────────────────
TRAIN_WINDOW  : int   = 52    # training window in rebalancing periods
REBAL_FREQ    : int   = 10    # rebalancing frequency in business days (≈ biweekly)
RIDGE_ALPHA   : float = 100.0  # Ridge regularisation strength

# ── Portfolio construction ────────────────────────────────────────────────────
TOP_N        : int   = 100   # stocks in long leg = stocks in short leg
LAMBDA_REG   : float = 10.0  # Markowitz regularisation (higher = more risk aversion)
MAX_WEIGHT   : float = 0.04  # maximum absolute weight per stock

# ── Backtesting ───────────────────────────────────────────────────────────────
TRANSACTION_COST_BPS   : float = 10.0  # one-way spread/impact cost in bps
SHORT_BORROW_COST_BPS  : float = 50.0  # annual short-borrow fee in bps (~50bps easy-to-borrow)
COV_LOOKBACK           : int   = 126   # days for rolling covariance (needs > n_stocks for Ledoit-Wolf)

# ── Signal generation ─────────────────────────────────────────────────────────
EWM_HALFLIFE : int = 52  # Ridge sample-weight half-life in rebalancing periods

# ── Volatility targeting ───────────────────────────────────────────────────────
VOL_TARGET   : float = 0.05  # annualised portfolio vol target (5 %)
VOL_LOOKBACK : int   = 10    # days of past portfolio returns used to estimate realized vol