"""
edgar.py
--------
SEC EDGAR XBRL fundamentals pipeline.

Downloads quarterly filings (10-Q / 10-K) from the SEC's free EDGAR API
and builds point-in-time factor panels for:

  EPS_MOM   : EPS vs same quarter 1 year ago (earnings acceleration)
  ASSET_GRW : YoY asset growth — lower is better (quality / no dilution)

Point-in-time integrity: values are made available on the `filed` date of each
10-Q/10-K, NOT on the fiscal period end date.  This prevents look-ahead bias.

EDGAR API limits: 10 requests/second. A User-Agent header is required.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

_CACHE_DIR  = Path(__file__).resolve().parents[1] / "data" / "edgar"
_CIK_CACHE  = _CACHE_DIR / "cik_map.json"
_UA         = "equity-strategy-research contact@example.com"   # required by SEC


# ─────────────────────────────────────────────────────────────────────────────
# CIK mapping
# ─────────────────────────────────────────────────────────────────────────────

def _get_cik_map() -> dict[str, str]:
    """Return {ticker: zero-padded-10-digit-CIK}. Cached after first download."""
    if _CIK_CACHE.exists():
        return json.loads(_CIK_CACHE.read_text())

    print("      Downloading SEC CIK map…", flush=True)
    resp = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers={"User-Agent": _UA},
        timeout=30,
    )
    resp.raise_for_status()
    raw  = resp.json()
    cmap = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in raw.values()}
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _CIK_CACHE.write_text(json.dumps(cmap))
    return cmap


# ─────────────────────────────────────────────────────────────────────────────
# Per-ticker fundamentals
# ─────────────────────────────────────────────────────────────────────────────

# XBRL concepts to try, in order of preference
_EPS_CONCEPTS   = ["EarningsPerShareBasic", "EarningsPerShareDiluted"]
_ASSET_CONCEPTS = ["Assets"]

def _fetch_company_facts(cik: str) -> dict | None:
    """Download full XBRL company facts for one CIK. Returns None on failure."""
    cache = _CACHE_DIR / "facts" / f"{cik}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    try:
        url  = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        resp = requests.get(url, headers={"User-Agent": _UA}, timeout=30)
        if resp.status_code != 200:
            return None
        data = resp.json()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(resp.text)
        return data
    except Exception:
        return None


def _extract_concept(facts: dict, concepts: list[str]) -> pd.DataFrame | None:
    """
    Extract quarterly values for the first matching concept.
    Returns DataFrame with columns [end, filed, val], sorted by filed date.
    Only 10-Q and 10-K filings; first filing per fiscal period (point-in-time).
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for concept in concepts:
        if concept not in us_gaap:
            continue
        units = us_gaap[concept].get("units", {})
        # EPS is "USD/shares"; balance-sheet items are "USD"; ratios are "pure"
        rows  = (units.get("USD/shares") or units.get("USD")
                 or units.get("pure") or units.get("shares") or [])
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df = df[df["form"].isin(["10-Q", "10-K"])].copy()
        if df.empty:
            continue
        df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
        df["end"]   = pd.to_datetime(df["end"],   errors="coerce")
        df = df.dropna(subset=["filed", "end", "val"])
        # Keep only the first filing for each fiscal quarter end
        df = df.sort_values("filed").drop_duplicates(subset=["end"], keep="first")
        df = df.sort_values("filed").reset_index(drop=True)
        return df[["end", "filed", "val"]]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def download_fundamentals(tickers: list[str]) -> dict[str, dict]:
    """
    Download + cache quarterly EPS and Assets for each ticker.

    Returns {ticker: {"eps": DataFrame, "assets": DataFrame}}
    where each DataFrame has columns [end, filed, val].
    """
    cik_map = _get_cik_map()
    result  = {}
    missing = [t for t in tickers if t not in result]

    n = len(missing)
    print(f"      Downloading EDGAR fundamentals for {n} tickers…", flush=True)
    for i, tk in enumerate(missing):
        cik = cik_map.get(tk)
        if cik is None:
            result[tk] = {"eps": None, "assets": None}
            continue

        facts = _fetch_company_facts(cik)
        if facts is None:
            result[tk] = {"eps": None, "assets": None}
        else:
            result[tk] = {
                "eps":    _extract_concept(facts, _EPS_CONCEPTS),
                "assets": _extract_concept(facts, _ASSET_CONCEPTS),
            }

        # Respect SEC rate limit (10 req/s) — skip delay for cache hits
        cache = _CACHE_DIR / "facts" / f"{cik}.json"
        if not cache.exists():
            time.sleep(0.12)

        if (i + 1) % 20 == 0:
            print(f"        {i+1}/{n}…", flush=True)

    return result


def build_fundamental_factors(
    fundamentals: dict[str, dict],
    dates: pd.DatetimeIndex,
    tickers: list[str],
) -> pd.DataFrame:
    """
    Build a factor panel DataFrame with MultiIndex columns (factor, ticker).

    Factors
    -------
    EPS_MOM   : (EPS_q - EPS_q4) / |EPS_q4|   — YoY EPS growth
                Positive = accelerating earnings.  Uses filing date (point-in-time).
    ASSET_GRW : (Assets_q - Assets_q4) / Assets_q4   — YoY asset growth
                Lower is better (quality signal).
    """
    eps_panel   = pd.DataFrame(np.nan, index=dates, columns=tickers)
    asset_panel = pd.DataFrame(np.nan, index=dates, columns=tickers)

    for tk in tickers:
        info = fundamentals.get(tk, {})

        # ── EPS momentum ─────────────────────────────────────────────────────
        eps_df = info.get("eps")
        if eps_df is not None and len(eps_df) >= 5:
            eps_df = eps_df.set_index("filed").sort_index()
            eps_df["end"] = pd.to_datetime(eps_df["end"])
            for date in dates:
                # Latest EPS filed on or before this date
                avail = eps_df[eps_df.index <= date]
                if len(avail) < 5:
                    continue
                latest_row  = avail.iloc[-1]
                latest_end  = latest_row["end"]
                latest_eps  = latest_row["val"]
                # Same quarter 1 year ago (filed and ending ~1 year before)
                prior = avail[avail["end"] <= latest_end - pd.Timedelta(days=300)]
                if prior.empty:
                    continue
                # Pick the quarter whose end date is closest to 1 year ago
                prior_sorted = prior.iloc[
                    (prior["end"] - (latest_end - pd.Timedelta(days=365))).abs().argsort()
                ]
                prior_eps = prior_sorted.iloc[0]["val"]
                if abs(prior_eps) > 1e-8:
                    eps_panel.loc[date, tk] = (latest_eps - prior_eps) / abs(prior_eps)

        # ── Asset growth ──────────────────────────────────────────────────────
        ast_df = info.get("assets")
        if ast_df is not None and len(ast_df) >= 5:
            ast_df = ast_df.set_index("filed").sort_index()
            ast_df["end"] = pd.to_datetime(ast_df["end"])
            for date in dates:
                avail = ast_df[ast_df.index <= date]
                if len(avail) < 5:
                    continue
                latest_row   = avail.iloc[-1]
                latest_end   = latest_row["end"]
                latest_assets = latest_row["val"]
                prior = avail[avail["end"] <= latest_end - pd.Timedelta(days=300)]
                if prior.empty:
                    continue
                prior_sorted = prior.iloc[
                    (prior["end"] - (latest_end - pd.Timedelta(days=365))).abs().argsort()
                ]
                prior_assets = prior_sorted.iloc[0]["val"]
                if prior_assets > 1e-4:
                    asset_panel.loc[date, tk] = (
                        (latest_assets - prior_assets) / prior_assets
                    )

    # Combine into MultiIndex panel
    eps_panel.columns   = pd.MultiIndex.from_product([["EPS_MOM"],   tickers])
    asset_panel.columns = pd.MultiIndex.from_product([["ASSET_GRW"], tickers])
    return pd.concat([eps_panel, asset_panel], axis=1)
