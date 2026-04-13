"""
strategy
--------
Systematic Equity Market-Neutral Strategy package.

Modules
-------
config      : hyper-parameters and paths
data_loader : price data ingestion (simulation / CSV / yfinance)
factors     : cross-sectional factor construction & normalisation
signals     : rolling Ridge regression signal generation
portfolio   : beta-neutral Markowitz portfolio optimisation
backtest    : full simulation engine
analytics   : performance metrics, plots, console report
"""

from .config      import *            # noqa: F401,F403
from .data_loader import simulate_universe, load_from_csv, load_from_yfinance
from .factors     import compute_factors, cross_sectional_zscore
from .signals     import generate_signals
from .portfolio   import build_portfolio
from .backtest    import run_backtest, BacktestResult
from .analytics   import compute_metrics, plot_results, print_report, drawdown_series

__all__ = [
    "simulate_universe", "load_from_csv", "load_from_yfinance",
    "compute_factors", "cross_sectional_zscore",
    "generate_signals",
    "build_portfolio",
    "run_backtest", "BacktestResult",
    "compute_metrics", "plot_results", "print_report", "drawdown_series",
]
