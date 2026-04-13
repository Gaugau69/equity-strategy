"""
data_collection.py  —  S&P 500 price download via Yahoo Finance CSV direct + cache incremental.
N'utilise PAS yfinance/curl_cffi — contourne le bug SSL Windows avec accents dans le chemin.

Cache incremental : ne re-telecharge que les nouveaux jours depuis la derniere mise a jour.
Premier run : ~10 minutes. Runs suivants : ~30 secondes.
"""
import os
import sys
import ssl
import certifi

os.environ["SSL_CERT_FILE"]      = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
ssl._create_default_https_context = ssl.create_default_context

# Fix Unicode pour terminal Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path
from io import StringIO
from datetime import date, datetime, timezone, timedelta
import urllib.request
import time
import json
import pandas as pd
import numpy as np

# ── Parametres ────────────────────────────────────────────────────────────────
START = "2015-01-01"

DATA_DIR  = Path("data")
CACHE_DIR = DATA_DIR / "_raw_cache"
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

MAX_MISSING_RATIO = 0.10
MAX_FFILL_GAP     = 5
RETURN_CLIP_LOWER = -0.30
RETURN_CLIP_UPPER = 0.30
PRINT_EVERY       = 25
SLEEP_BETWEEN     = 0.3

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

_SSL_CTX = ssl.create_default_context(cafile=certifi.where())
HEADERS  = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def fetch_url(url: str) -> str:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as r:
        return r.read().decode("utf-8", errors="ignore")


# ── Tickers S&P 500 ───────────────────────────────────────────────────────────
def load_sp500_tickers() -> list[str]:
    html   = fetch_url(WIKI_SP500_URL)
    tables = pd.read_html(StringIO(html))
    df     = tables[0].copy()
    if "Symbol" not in df.columns:
        raise ValueError("Column 'Symbol' not found.")
    tickers = df["Symbol"].astype(str).str.strip().tolist()
    tickers = [t.replace(".", "-") for t in tickers]
    return sorted(set(tickers))


# ── Download incremental ──────────────────────────────────────────────────────
def _date_to_ts(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _get_last_cached_date(ticker: str) -> str | None:
    """Retourne la derniere date en cache pour ce ticker, ou None si pas de cache."""
    cache_path = CACHE_DIR / f"{ticker}.csv"
    if not cache_path.exists():
        return None
    try:
        s = pd.read_csv(cache_path, index_col=0, parse_dates=True).squeeze("columns")
        if s.empty:
            return None
        return s.index.max().strftime("%Y-%m-%d")
    except Exception:
        return None


def download_one_ticker(ticker: str, today: str) -> pd.Series:
    """
    Telecharge les donnees pour un ticker.
    - Si cache existe et a jour : retourne le cache directement
    - Si cache existe mais perime : telecharge seulement les nouveaux jours et merge
    - Si pas de cache : telecharge tout depuis START
    """
    cache_path    = CACHE_DIR / f"{ticker}.csv"
    last_cached   = _get_last_cached_date(ticker)

    # Cache a jour (derniere date = aujourd'hui ou hier)
    if last_cached is not None:
        last_dt = datetime.strptime(last_cached, "%Y-%m-%d").date()
        today_dt = datetime.strptime(today, "%Y-%m-%d").date()
        # Si on a les donnees jusqu'a hier ou aujourd'hui, pas besoin de re-telecharger
        if (today_dt - last_dt).days <= 3:
            s = pd.read_csv(cache_path, index_col=0, parse_dates=True).squeeze("columns")
            s.name = ticker
            return s

    # Determine la plage de dates a telecharger
    if last_cached is not None:
        # Telecharge depuis le lendemain de la derniere date en cache
        start_dl = (datetime.strptime(last_cached, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        start_dl = START

    t1  = _date_to_ts(start_dl)
    t2  = _date_to_ts(today)
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?interval=1d&period1={t1}&period2={t2}&events=history"
    )

    raw  = fetch_url(url)
    data = json.loads(raw)

    result = data.get("chart", {}).get("result")
    if not result:
        error = data.get("chart", {}).get("error", {})
        # Si pas de nouvelles donnees mais cache existe, retourne le cache
        if last_cached is not None:
            s = pd.read_csv(cache_path, index_col=0, parse_dates=True).squeeze("columns")
            s.name = ticker
            return s
        raise ValueError(f"No data for {ticker}: {error}")

    result     = result[0]
    timestamps = result.get("timestamp", [])
    closes     = result["indicators"]["quote"][0].get("close", [])

    if not timestamps or not closes:
        if last_cached is not None:
            s = pd.read_csv(cache_path, index_col=0, parse_dates=True).squeeze("columns")
            s.name = ticker
            return s
        raise ValueError(f"Empty series for {ticker}")

    dates    = [datetime.utcfromtimestamp(ts).date() for ts in timestamps]
    new_data = pd.Series(closes, index=pd.to_datetime(dates), name=ticker, dtype=float)
    new_data = new_data.dropna().sort_index()

    # Merge avec le cache existant si disponible
    if last_cached is not None and cache_path.exists():
        old_data = pd.read_csv(cache_path, index_col=0, parse_dates=True).squeeze("columns")
        old_data.name = ticker
        close = pd.concat([old_data, new_data])
        close = close[~close.index.duplicated(keep="last")].sort_index()
    else:
        close = new_data

    if close.empty:
        raise ValueError(f"Empty close series for {ticker}")

    close.to_csv(cache_path)
    return close


def download_prices(tickers: list[str], today: str) -> pd.DataFrame:
    series_list, failed = [], []
    total    = len(tickers)
    cached_n = sum(1 for t in tickers if _get_last_cached_date(t) is not None)
    print(f"Downloading {total} tickers ({cached_n} cached, {total - cached_n} fresh)...")

    for i, ticker in enumerate(tickers, 1):
        was_cached    = _get_last_cached_date(ticker)
        last_dt       = datetime.strptime(was_cached, "%Y-%m-%d").date() if was_cached else None
        today_dt      = datetime.strptime(today, "%Y-%m-%d").date()
        fully_cached  = last_dt is not None and (today_dt - last_dt).days <= 3

        try:
            s = download_one_ticker(ticker, today)
            series_list.append(s)
        except Exception as e:
            failed.append((ticker, str(e)))
            if len(failed) <= 5:
                print(f"  ERROR {ticker}: {e}")

        if not fully_cached:
            time.sleep(SLEEP_BETWEEN)

        if i % PRINT_EVERY == 0 or i == total:
            print(f"  {i}/{total} | ok={len(series_list)} | fail={len(failed)}")

    if not series_list:
        raise RuntimeError("No price series downloaded.")

    prices = pd.concat(series_list, axis=1).sort_index()
    print(f"\nDownload done. ok={len(series_list)}, fail={len(failed)}")
    if failed:
        print("Failures:", [(t, e[:60]) for t, e in failed[:10]])
    return prices


# ── Nettoyage ─────────────────────────────────────────────────────────────────
def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="first")]
    prices = prices.dropna(axis=1, how="all")
    prices = prices.where(prices > 0)
    daily_chg = prices.pct_change(fill_method=None).abs()
    prices    = prices.where(daily_chg < 0.90)
    missing   = prices.isna().mean()
    prices    = prices[missing[missing <= MAX_MISSING_RATIO].index]
    if prices.shape[1] == 0:
        raise ValueError("No column left after missing-value filter.")
    prices = prices.ffill(limit=MAX_FFILL_GAP)
    prices = prices.dropna(axis=1, how="all")
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change(fill_method=None)
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.clip(lower=RETURN_CLIP_LOWER, upper=RETURN_CLIP_UPPER)
    return returns


def quality_report(prices: pd.DataFrame, returns: pd.DataFrame) -> None:
    print("\n=== DATA QUALITY REPORT ===")
    print(f"Prices : {prices.shape}  |  Returns : {returns.shape}")
    print(f"Date range : {prices.index[0].date()} -> {prices.index[-1].date()}")
    print("Top-10 missing (prices):")
    print(prices.isna().mean().sort_values(ascending=False).head(10).to_string())
    stacked = returns.stack()
    print(f"\nReturn stats (n={len(stacked):,}):")
    print(stacked.describe(percentiles=[0.01, 0.05, 0.95, 0.99]).to_string())


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    today   = date.today().strftime("%Y-%m-%d")
    tickers = load_sp500_tickers()
    print(f"S&P 500 universe: {len(tickers)} tickers")
    print(f"Updating to: {today}")

    prices_raw = download_prices(tickers, today)
    prices     = clean_prices(prices_raw)
    returns    = compute_returns(prices)

    prices.to_csv(DATA_DIR / "prices.csv")
    returns.to_csv(DATA_DIR / "returns.csv")

    quality_report(prices, returns)
    print("\nSaved: prices.csv, returns.csv")


if __name__ == "__main__":
    main()