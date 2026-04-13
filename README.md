# Systematic Equity Market-Neutral Strategy

ML-driven long/short equity strategy on ~500 US stocks.

## Project Structure

```
strategy/
├── strategy/               ← Python package (importable modules)
│   ├── __init__.py
│   ├── config.py           ← All hyper-parameters in one place
│   ├── data_loader.py      ← Simulate / CSV / yfinance data
│   ├── factors.py          ← MOM, VOL, BETA factor construction
│   ├── signals.py          ← Rolling Ridge regression
│   ├── portfolio.py        ← Beta-neutral Markowitz optimisation
│   ├── backtest.py         ← Full simulation engine
│   └── analytics.py        ← Metrics, charts, console report
│
├── notebooks/
│   └── strategy_walkthrough.ipynb   ← Step-by-step interactive notebook
│
├── outputs/                ← Generated files (nav, charts, metrics)
├── data/                   ← Drop CSV price files here
│
├── run_strategy.py         ← CLI entry point
└── requirements.txt
```

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run with simulated data
python run_strategy.py

# Run with real CSV prices
python run_strategy.py --csv data/prices.csv

# Custom parameters
python run_strategy.py --tc 5 --top_n 50 --lam 3.0

# Launch notebook
jupyter notebook notebooks/strategy_walkthrough.ipynb
```

## Strategy Overview

| Component | Detail |
|-----------|--------|
| Universe | ~500 US stocks (simulated or real) |
| Factors | MOM_1M, MOM_3M, MOM_12M, VOL_1M, BETA |
| Normalisation | Cross-sectional z-score, winsorised ±3σ |
| Model | Rolling Ridge regression (52-week train window) |
| Rebalancing | Weekly (every 5 business days) |
| Portfolio | Dollar-neutral + Beta-neutral, regularised Markowitz |
| Transaction cost | 10 bps one-way (configurable) |
| Metrics | Sharpe, Calmar, Max DD, Hit Rate, Skew, Kurtosis |

## Using Real Data

```python
# In data_loader.py or directly in the notebook:
from strategy.data_loader import load_from_yfinance

prices = load_from_yfinance(["AAPL", "MSFT", ...], start="2018-01-01")
```

## Key Configuration (config.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_STOCKS` | 500 | Universe size |
| `TRAIN_WINDOW` | 52 | Rolling train window (weeks) |
| `RIDGE_ALPHA` | 1.0 | Ridge regularisation |
| `TOP_N` | 100 | Long/short leg size |
| `LAMBDA_REG` | 5.0 | Markowitz λ |
| `MAX_WEIGHT` | 0.05 | Max position weight |
| `TRANSACTION_COST_BPS` | 10 | One-way TC in bps |
