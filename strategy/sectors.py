"""
sectors.py
----------
GICS sector mapping for the trading universe.
Fetched from yfinance once and cached to data/sector_map.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

_CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "sector_map.csv"

_NORMALIZE = {
    "Technology":              "IT",
    "Information Technology":  "IT",
    "Health Care":             "HC",
    "Healthcare":              "HC",
    "Financials":              "FIN",
    "Financial Services":      "FIN",
    "Consumer Discretionary":  "CD",
    "Communication Services":  "COMM",
    "Industrials":             "IND",
    "Consumer Staples":        "CS",
    "Energy":                  "EN",
    "Utilities":               "UT",
    "Real Estate":             "RE",
    "Materials":               "MAT",
}


def get_sector_map(tickers: list[str], refresh: bool = False) -> dict[str, str]:
    """
    Return {ticker: sector_code} for all tickers.
    Reads from CSV cache; only fetches tickers that are missing.
    """
    cached: dict[str, str] = {}
    if _CACHE_PATH.exists() and not refresh:
        cached = pd.read_csv(_CACHE_PATH, index_col=0)["sector"].to_dict()

    missing = [t for t in tickers if t not in cached]
    if missing:
        print(f"      Fetching sectors for {len(missing)} tickers…", flush=True)
        for tk in missing:
            try:
                raw = yf.Ticker(tk).info.get("sector", "") or ""
                cached[tk] = _NORMALIZE.get(raw, raw) if raw else "Unknown"
            except Exception:
                cached[tk] = "Unknown"

        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        (pd.Series(cached)
           .rename_axis("ticker")
           .rename("sector")
           .to_csv(_CACHE_PATH, header=True))

    return {t: cached.get(t, "Unknown") for t in tickers}
