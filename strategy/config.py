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

# ── Signal generation (rolling Ridge) ────────────────────────────────────────
TRAIN_WINDOW  : int   = 52    # training window in weeks
REBAL_FREQ    : int   = 5     # rebalancing frequency in business days (≈ weekly)
RIDGE_ALPHA   : float = 25.0   # Ridge regularisation strength

# ── Portfolio construction ────────────────────────────────────────────────────
TOP_N        : int   = 75    # stocks in long leg = stocks in short leg
LAMBDA_REG   : float = 1.0    # Markowitz regularisation strength
MAX_WEIGHT   : float = 0.04   # maximum absolute weight per stock

# ── Backtesting ───────────────────────────────────────────────────────────────
TRANSACTION_COST_BPS : float = 10.0   # realistic one-way cost   # one-way transaction cost in basis points
COV_LOOKBACK         : int   = 60     # days for rolling covariance estimation