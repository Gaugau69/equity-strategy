"""
run_real_data.py
----------------
Amélioration 1 : vraies données S&P 500.

Modes disponibles (choisis selon ton environnement) :
  --source yfinance   : télécharge via yfinance (réseau direct nécessaire)
  --source csv        : charge depuis un CSV local déjà téléchargé
  --source stooq      : télécharge via stooq.com (alternative si yfinance bloqué)

Obtenir les données manuellement si tout est bloqué
-----------------------------------------------------
Option A — stooq (gratuit, pas de compte requis) :
  Le script tente automatiquement pandas_datareader avec stooq.

Option B — fichier CSV local :
  1. Va sur https://finance.yahoo.com/quote/AAPL/history
  2. Ou télécharge depuis https://stooq.com/q/d/l/?s=aapl.us&i=d
  3. Place un fichier prices.csv dans data/ avec colonnes : date, AAPL, MSFT, ...
  4. Lance : python run_real_data.py --source csv --csv data/prices.csv

Option C — yfinance avec proxy désactivé :
  set HTTPS_PROXY=
  set HTTP_PROXY=
  python run_real_data.py --source yfinance

Usage
-----
  python run_real_data.py                        # auto-détecte la source
  python run_real_data.py --source stooq         # stooq.com
  python run_real_data.py --source csv --csv data/prices.csv
"""

from __future__ import annotations
import sys, warnings, argparse
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from strategy.factors  import compute_factors, cross_sectional_zscore
from strategy.signals  import generate_signals
from strategy.backtest import run_backtest
from strategy.analytics import print_report, plot_results
from strategy import config

OUT_DIR = Path("outputs/real")
OUT_DIR.mkdir(parents=True, exist_ok=True)
START   = "2018-01-01"
END     = "2024-12-31"

_SP500_FALLBACK = [
    "AAPL","MSFT","AMZN","NVDA","GOOGL","GOOG","META","BRK-B","TSLA","UNH",
    "XOM","LLY","JPM","JNJ","V","PG","MA","AVGO","HD","CVX","MRK","ABBV",
    "COST","PEP","KO","WMT","CSCO","TMO","MCD","ACN","BAC","ABT","CRM","NFLX",
    "LIN","DHR","AMD","ADBE","TXN","NKE","NEE","PM","ORCL","HON","RTX","MS",
    "AMGN","QCOM","UPS","IBM","LOW","GS","INTU","CAT","ELV","SPGI","BLK","DE",
    "AXP","PLD","MDLZ","SYK","BKNG","T","GILD","ADP","CI","NOW","TJX","REGN",
    "VRTX","ZTS","CB","C","TMUS","CME","USB","CVS","MO","EOG","SO","DUK","PGR",
    "BSX","MU","SCHW","ITW","NOC","ETN","AON","MMC","APD","GE","SHW","BDX",
    "ISRG","CSX","WM","CL","MMM","PNC","FCX","EMR","FDX","NSC","HUM","ECL",
    "EW","MCO","ORLY","PSA","CTAS","ICE","KLAC","LRCX","SNPS","CDNS","FTNT",
    "PANW","MCHP","ADI","AMAT","AIG","MET","PRU","AFL","TRV","ALL","CNC","HCA",
    "DVN","OXY","HAL","SLB","PSX","VLO","MPC","COP","APA","HES","WFC","COF",
    "DFS","SYF","AMP","LHX","GD","LMT","BA","TDG","CARR","OTIS","ROP","AME",
    "PH","ROK","DOV","XYL","GWW","FAST","SWK","IR","A","KEYS","IDXX","MTD",
    "WAT","TER","SPG","AMT","CCI","EQIX","DLR","O","WELL","VTR","MAA","EQR",
    "AVB","RSG","WCN","CLX","CHD","EL","KMB","TSN","GIS","K","KHC","MKC",
    "HSY","STZ","MCK","DGX","LH","IQV","F","GM","HOG","CMI","PCAR","URI",
    "UNP","WAB","LSTR","CHRW","XPO","JBHT","SAIA","ODFL","KNX",
    "VMC","MLM","OC","IP","PKG","SEE","AEP","EXC","D","SRE","PEG","FE",
    "ES","ETR","EIX","PPL","CNP","NI","AEE","WEC","DTE","CMS",
]


def _get_sp500_tickers() -> list[str]:
    """Fetch current S&P 500 tickers from Wikipedia; fall back to curated list."""
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            attrs={"id": "constituents"},
        )
        tickers = (
            tables[0]["Symbol"]
            .astype(str)
            .str.strip()
            .str.replace(".", "-", regex=False)
            .tolist()
        )
        if len(tickers) > 400:
            print(f"      Fetched {len(tickers)} tickers from Wikipedia")
            return tickers
    except Exception as e:
        print(f"      Wikipedia fetch failed ({e}) — using curated fallback list")
    return list(dict.fromkeys(_SP500_FALLBACK))


SP500_TICKERS = _get_sp500_tickers()


# ─────────────────────────────────────────────────────────────────────────────
# Data sources
# ─────────────────────────────────────────────────────────────────────────────

def download_yfinance(tickers: list[str]) -> pd.DataFrame:
    """Download via yfinance (requires direct internet access)."""
    import yfinance as yf
    print("      Downloading via yfinance...")
    frames = []
    batch  = 50
    for i in range(0, len(tickers), batch):
        b   = tickers[i:i+batch]
        pct = int(100 * i / len(tickers))
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"      [{bar}] {pct:3d}%", end="\r")
        try:
            raw   = yf.download(b, start=START, end=END,
                                auto_adjust=True, progress=False)
            close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
            frames.append(close)
        except Exception:
            pass
    print(f"      [{'█'*20}] 100%")
    return pd.concat(frames, axis=1)


def download_stooq(tickers: list[str]) -> pd.DataFrame:
    """Download via stooq.com using pandas_datareader (often works through proxies)."""
    try:
        import pandas_datareader as pdr
    except ImportError:
        print("      Installing pandas_datareader...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install",
                       "pandas_datareader", "-q"])
        import pandas_datareader as pdr

    print("      Downloading via stooq.com...")
    frames = {}
    total  = len(tickers)
    for i, ticker in enumerate(tickers):
        pct = int(100 * i / total)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"      [{bar}] {pct:3d}%  {ticker:<8}", end="\r")
        try:
            stooq_ticker = ticker.replace("-", ".").lower() + ".us"
            df = pdr.DataReader(stooq_ticker, "stooq",
                                start=START, end=END)
            if "Close" in df.columns and len(df) > 100:
                frames[ticker] = df["Close"].sort_index()
        except Exception:
            pass
    print(f"      [{'█'*20}] 100%  ")
    if not frames:
        raise RuntimeError("stooq download returned 0 tickers")
    return pd.DataFrame(frames)


def load_csv(path: str) -> pd.DataFrame:
    """Load prices from a local CSV file."""
    print(f"      Loading from {path}...")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df.astype(float)


def auto_download(tickers: list[str], source: str) -> pd.DataFrame:
    """Try sources in order until one works."""
    if source == "csv":
        raise ValueError("Use --csv path/to/file.csv with --source csv")

    if source in ("yfinance", "auto"):
        try:
            prices = download_yfinance(tickers)
            if prices.shape[1] > 20:
                return prices
            print("      yfinance returned <20 tickers, trying stooq...")
        except Exception as e:
            print(f"      yfinance failed: {e}")

    if source in ("stooq", "auto"):
        try:
            prices = download_stooq(tickers)
            if prices.shape[1] > 20:
                return prices
        except Exception as e:
            print(f"      stooq failed: {e}")

    raise RuntimeError(
        "\n\n  ✗ All download sources failed (likely proxy/firewall).\n"
        "  Solutions:\n"
        "    1. Disable your proxy:  set HTTPS_PROXY=  &&  set HTTP_PROXY=\n"
        "       Then retry: python run_real_data.py\n\n"
        "    2. Download manually and use CSV mode:\n"
        "       - Go to https://stooq.com and download daily OHLCV for your tickers\n"
        "       - Or use https://finance.yahoo.com/quote/AAPL/history\n"
        "       - Save as data/prices.csv (columns: date, AAPL, MSFT, ...)\n"
        "       - Run: python run_real_data.py --source csv --csv data/prices.csv\n\n"
        "    3. Use a VPN or hotspot and retry.\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Universe cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_universe(prices: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[3/6] Cleaning universe...")
    n0 = prices.shape[1]

    # Remove duplicated columns
    prices = prices.loc[:, ~prices.columns.duplicated()]

    # Require ≥ 90% coverage
    prices = prices.loc[:, prices.notna().mean() >= 0.90]
    print(f"      After coverage filter: {prices.shape[1]} / {n0}")

    # Forward-fill short gaps (max 5 days)
    prices = prices.ffill(limit=5)

    # Drop penny stocks
    prices = prices.loc[:, prices.median() >= 5.0]
    print(f"      After price filter:    {prices.shape[1]}")

    # Drop flat stocks (zero std = suspended)
    prices = prices.loc[:, prices.pct_change().std() > 1e-5]
    print(f"      After activity filter: {prices.shape[1]}")

    # Need at least 50 stocks to run the strategy
    if prices.shape[1] < 50:
        raise RuntimeError(
            f"Only {prices.shape[1]} stocks after cleaning — not enough to run strategy.\n"
            "Try downloading more tickers or relaxing the filters."
        )

    print(f"      Final universe: {prices.shape[1]} stocks × {len(prices)} days  "
          f"({prices.index[0].date()} → {prices.index[-1].date()})")
    return prices


# ─────────────────────────────────────────────────────────────────────────────
# Beta estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_betas(prices: pd.DataFrame, source: str) -> tuple[np.ndarray, pd.Series]:
    print(f"\n[4/6] Estimating market betas...")

    # Download SPY as market proxy
    market = None
    if source != "csv":
        try:
            if source in ("yfinance", "auto"):
                import yfinance as yf
                spy = yf.download("SPY", start=START, end=END,
                                  auto_adjust=True, progress=False)["Close"]
                if isinstance(spy, pd.DataFrame):
                    spy = spy.squeeze()
                market = spy.reindex(prices.index).ffill()
            elif source == "stooq":
                import pandas_datareader as pdr
                spy    = pdr.DataReader("spy.us", "stooq", start=START, end=END)["Close"]
                market = spy.sort_index().reindex(prices.index).ffill()
        except Exception:
            pass

    # Fallback: equal-weight index
    if market is None or float(market.isna().mean().mean() if isinstance(market, pd.DataFrame) else market.isna().mean()) > 0.5:
        print("      SPY unavailable — using equal-weight index as market proxy")
        market = prices.mean(axis=1)

    market.name = "market"

    # Vectorised rolling beta (last 252 days)
    ret_s   = prices.pct_change()
    ret_m   = market.pct_change().reindex(ret_s.index)
    window  = min(252, len(ret_s) - 5)
    r_s     = ret_s.iloc[-window:]
    r_m     = ret_m.iloc[-window:]
    cov_xm  = r_s.subtract(r_s.mean()).multiply(r_m - r_m.mean(), axis=0).mean()
    var_m   = r_m.var()
    betas   = (cov_xm / var_m).reindex(prices.columns).fillna(1.0).values

    print(f"      Beta range: [{betas.min():.2f}, {betas.max():.2f}]  "
          f"mean={betas.mean():.2f}  median={np.median(betas):.2f}")
    return betas, market


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(prices, betas, market):
    print(f"\n[5/6] Running strategy pipeline...")

    print("      Computing factors...")
    fraw   = compute_factors(prices, market)
    fnorm  = cross_sectional_zscore(fraw)
    print(f"      Factor panel: {fnorm.shape}")

    fwd     = prices.pct_change(config.REBAL_FREQ).shift(-config.REBAL_FREQ)
    fwd_mkt = market.pct_change(config.REBAL_FREQ).shift(-config.REBAL_FREQ)

    # Cap top_n at ~12% of universe (each leg should be a concentrated decile)
    n_stocks = len(prices.columns)
    top_n    = max(10, min(config.TOP_N, n_stocks // 8))
    print(f"      Universe: {n_stocks} stocks → top_n={top_n} per leg")

    print("      Generating signals (rolling Ridge)...")
    signals = generate_signals(fnorm, fwd,
                               train_window=config.TRAIN_WINDOW,
                               rebal_freq=config.REBAL_FREQ,
                               alpha=config.RIDGE_ALPHA,
                               market_fwd_returns=fwd_mkt)
    print(f"      {len(signals)} rebalancing dates")

    print("      Running backtest...")
    result = run_backtest(prices, signals, betas,
                          transaction_cost_bps=config.TRANSACTION_COST_BPS,
                          top_n=top_n, lambda_reg=config.LAMBDA_REG)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(result):
    print(f"\n[6/6] Saving outputs → {OUT_DIR}/")
    result.nav.to_csv(OUT_DIR / "nav_real.csv")
    result.returns.to_csv(OUT_DIR / "returns_real.csv")
    pd.DataFrame([result.metrics]).to_csv(OUT_DIR / "metrics_real.csv", index=False)
    result.returns.resample("ME").apply(
        lambda x: (1 + x).prod() - 1
    ).to_csv(OUT_DIR / "monthly_returns_real.csv")
    fig = plot_results(result)
    fig.savefig(OUT_DIR / "dashboard_real.png", dpi=150,
                bbox_inches="tight", facecolor="#0d1117")
    print_report(result.metrics)
    print(f"\n  Outputs saved to {OUT_DIR}/")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="auto",
                   choices=["auto","yfinance","stooq","csv"],
                   help="Data source (default: auto)")
    p.add_argument("--csv", default=None,
                   help="Path to CSV prices file (required with --source csv)")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print(" Amélioration 1 — Vraies données S&P 500")
    print(f" Source: {args.source}")
    print("=" * 60)

    # ── 1. Tickers ─────────────────────────────────────────────────────────
    print(f"\n[1/6] Universe: {len(SP500_TICKERS)} S&P 500 tickers")

    # ── 2. Download ─────────────────────────────────────────────────────────
    print(f"\n[2/6] Downloading prices ({START} → {END})...")
    if args.source == "csv":
        if not args.csv:
            print("  ERROR: --csv <path> required with --source csv")
            sys.exit(1)
        prices = load_csv(args.csv)
    else:
        prices = auto_download(SP500_TICKERS, args.source)

    # ── 3-6. Clean → Beta → Pipeline → Save ────────────────────────────────
    prices       = clean_universe(prices)
    betas, mkt   = estimate_betas(prices, args.source)
    result       = run_pipeline(prices, betas, mkt)
    save_outputs(result)

    print("\nDone. ✓\n")


if __name__ == "__main__":
    main()