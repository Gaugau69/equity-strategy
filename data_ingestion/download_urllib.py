"""
download_urllib.py
------------------
Télécharge les données S&P 500 en utilisant uniquement urllib
(bibliothèque standard Python — pas de curl, pas de yfinance).

Compatible avec les proxies SSL inspection d'entreprise.

Usage :
    python download_urllib.py
"""

import os
import ssl
import time
import random
import urllib.request
import urllib.error
from io import StringIO
from pathlib import Path

import pandas as pd

# ── Désactiver SSL complètement (proxy corporate) ─────────────────────────────
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode    = ssl.CERT_NONE

os.makedirs("data", exist_ok=True)

TICKERS = [
    "AAPL","MSFT","AMZN","NVDA","GOOGL","META","TSLA","UNH","XOM","LLY",
    "JPM","JNJ","V","PG","MA","AVGO","HD","CVX","MRK","ABBV","COST","PEP",
    "KO","WMT","CSCO","TMO","MCD","ACN","BAC","ABT","CRM","NFLX","LIN",
    "DHR","AMD","ADBE","TXN","NKE","NEE","PM","ORCL","HON","RTX","MS",
    "AMGN","QCOM","UPS","IBM","LOW","GS","INTU","CAT","SPGI","BLK",
    "DE","AXP","SYK","BKNG","GILD","ADP","CI","TJX","REGN","VRTX",
    "ZTS","CB","C","TMUS","CME","USB","CVS","MO","EOG","SO","DUK",
    "PGR","BSX","MU","SCHW","ITW","NOC","ETN","GE","SHW","BDX","ISRG",
    "CSX","WM","CL","MMM","PNC","EMR","FDX","NSC","HUM","ECL",
]

# Dates en timestamp Unix
P1 = 1514764800  # 2018-01-01
P2 = 1735689600  # 2024-12-31

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_ticker(ticker: str) -> pd.Series | None:
    """Télécharge un ticker via urllib (pas de curl)."""
    url = (
        f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
        f"?period1={P1}&period2={P2}&interval=1d&events=history"
    )
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
            content = resp.read().decode("utf-8")
        if "Date" not in content or len(content) < 100:
            return None
        df = pd.read_csv(StringIO(content), parse_dates=["Date"], index_col="Date")
        if "Close" not in df.columns or len(df) < 100:
            return None
        s = df["Close"].replace("null", float("nan")).astype(float).dropna()
        return s.rename(ticker) if len(s) > 100 else None
    except urllib.error.HTTPError as e:
        if e.code == 401:
            # Yahoo requires login — try query2
            return fetch_ticker_query2(ticker)
        return None
    except Exception:
        return None


def fetch_ticker_query2(ticker: str) -> pd.Series | None:
    """Fallback sur query2.finance.yahoo.com."""
    url = (
        f"https://query2.finance.yahoo.com/v7/finance/download/{ticker}"
        f"?period1={P1}&period2={P2}&interval=1d&events=history"
    )
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
            content = resp.read().decode("utf-8")
        if "Date" not in content or len(content) < 100:
            return None
        df = pd.read_csv(StringIO(content), parse_dates=["Date"], index_col="Date")
        if "Close" not in df.columns:
            return None
        s = df["Close"].replace("null", float("nan")).astype(float).dropna()
        return s.rename(ticker) if len(s) > 100 else None
    except Exception:
        return None


def main():
    print("=" * 55)
    print(" Download S&P 500 — urllib (sans curl)")
    print("=" * 55)

    results  = {}
    failed   = []
    total    = len(TICKERS)

    for i, ticker in enumerate(TICKERS):
        pct = int(100 * i / total)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        ok  = len(results)
        print(f"  [{bar}] {pct:3d}%  {ticker:<6}  ✓{ok}", end="\r")

        s = fetch_ticker(ticker)
        if s is not None:
            results[ticker] = s
        else:
            failed.append(ticker)

        time.sleep(random.uniform(0.2, 0.5))

    print(f"  [{'█'*20}] 100%  Done              ")
    print(f"\n  ✓ Téléchargés : {len(results)} / {total}")
    if failed:
        print(f"  ✗ Échoués    : {len(failed)}  ({', '.join(failed[:8])}{'...' if len(failed)>8 else ''})")

    if len(results) < 30:
        print("\n  ✗ Pas assez de données.")
        print("  Ton réseau bloque complètement les connexions HTTPS externes.")
        print("  Solution définitive : hotspot mobile ou VPN.")
        return

    prices = pd.DataFrame(results)
    prices.index = pd.to_datetime(prices.index)
    if hasattr(prices.index, "tz") and prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices.index.name = "date"
    prices.sort_index(inplace=True)
    prices.to_csv("data/prices.csv")

    print(f"\n  ✓ Sauvegardé : data/prices.csv")
    print(f"  {prices.shape[1]} tickers × {len(prices)} jours")
    print(f"  {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"\n  Lance maintenant :")
    print(f"  python run_real_data.py --source csv --csv data/prices.csv")


if __name__ == "__main__":
    main()
