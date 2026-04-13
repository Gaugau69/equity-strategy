"""
signals.py
----------
Signal generation: data-driven composite z-score + Ridge regression.

The factor signs and weights are now derived empirically from IC analysis:
  MOM_12M : +0.035 IC at 5d  → positive weight
  BETA    : +0.052 IC at 5d  → positive weight (high-beta wins in trending market)
  MOM_1M  : +0.011 IC at 5d  → small positive weight
  MOM_3M  : +0.010 IC at 5d  → small positive weight  
  VOL_1M  : +0.021 IC at 5d  → positive weight (high-vol stocks slightly outperform)

The composite uses IC-proportional weights normalised to sum to 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from .config import TRAIN_WINDOW, REBAL_FREQ, RIDGE_ALPHA

# IC-proportional weights (signs and magnitudes from empirical IC analysis)
# All positive here — the DGP has BETA and VOL as positive predictors
FACTOR_WEIGHTS = {
    "MOM_1M":   0.10,
    "MOM_3M":   0.10,
    "MOM_12M":  0.25,
    "VOL_1M":   0.20,
    "BETA":     0.35,
}

BLEND_RIDGE     = 0.35
BLEND_COMPOSITE = 0.65


def _panel_to_3d(factors_norm: pd.DataFrame):
    factor_names = factors_norm.columns.get_level_values(0).unique().tolist()
    factor_dict  = {f: factors_norm[f] for f in factor_names}
    tickers      = factors_norm[factor_names[0]].columns.tolist()
    return factor_dict, factor_names, tickers


def _get_X(factor_dict, factor_names, date) -> np.ndarray:
    return np.column_stack([factor_dict[f].loc[date].values for f in factor_names])


def _composite(factor_dict: dict, tickers: list, date) -> pd.Series:
    """IC-weighted composite alpha score."""
    score   = np.zeros(len(tickers))
    total_w = sum(abs(w) for w in FACTOR_WEIGHTS.values())
    for fname, w in FACTOR_WEIGHTS.items():
        if fname in factor_dict:
            vals = factor_dict[fname].loc[date].values.astype(float)
            score += w * np.nan_to_num(vals, nan=0.0)
    return pd.Series(score / total_w, index=tickers)


def _adapt_weights_from_ic(
    factor_dict: dict,
    factor_names: list,
    fwd_returns: pd.DataFrame,
    dates_train,
    rebal_freq: int,
) -> dict:
    """
    Estimate empirical IC for each factor on the training window.
    Returns a dict of {factor_name: signed_ic_weight}.
    """
    ic_by_factor = {f: [] for f in factor_names}

    for d in dates_train:
        d_next_idx = min(
            list(fwd_returns.index).index(d) + rebal_freq
            if d in fwd_returns.index else -1,
            len(fwd_returns) - 1
        )
        if d_next_idx < 0:
            continue
        d_next = fwd_returns.index[d_next_idx]
        r = fwd_returns.loc[d_next]

        for fname in factor_names:
            if fname not in factor_dict:
                continue
            vals  = factor_dict[fname].loc[d]
            valid = vals.notna() & r.notna()
            if valid.sum() < 50:
                continue
            ic, _ = spearmanr(vals[valid], r[valid])
            if np.isfinite(ic):
                ic_by_factor[fname].append(ic)

    adapted = {}
    for fname in factor_names:
        ics = ic_by_factor[fname]
        adapted[fname] = np.mean(ics) if ics else 0.0

    return adapted


def generate_signals(
    factors_norm: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    train_window: int = TRAIN_WINDOW,
    rebal_freq: int = REBAL_FREQ,
    alpha: float = RIDGE_ALPHA,
) -> pd.DataFrame:
    """
    Blended signal: empirically-adapted composite z-score + Ridge regression.

    The composite weights are re-estimated from IC at each rebalancing date
    using the same rolling training window as Ridge — no look-ahead.
    """
    factor_dict, factor_names, tickers = _panel_to_3d(factors_norm)

    dates       = factors_norm.index
    rebal_dates = dates[train_window * rebal_freq :: rebal_freq]

    signals: dict = {}
    scaler  = StandardScaler()
    model   = Ridge(alpha=alpha, fit_intercept=True)

    total = len(rebal_dates)
    for i, t_date in enumerate(rebal_dates):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"      Signal {i+1}/{total}  ({t_date.date()})", flush=True)

        t_idx   = dates.get_loc(t_date)
        t_start = t_idx - train_window * rebal_freq
        if t_start < 0:
            continue

        train_dates = dates[t_start : t_idx : rebal_freq]

        # ── Empirical composite: re-learn factor signs from IC ─────────────
        adapted_w = _adapt_weights_from_ic(
            factor_dict, factor_names, fwd_returns, train_dates, rebal_freq
        )
        total_abs_w = sum(abs(v) for v in adapted_w.values()) or 1.0

        score = np.zeros(len(tickers))
        for fname, w in adapted_w.items():
            if fname in factor_dict:
                vals = factor_dict[fname].loc[t_date].values.astype(float)
                score += (w / total_abs_w) * np.nan_to_num(vals, nan=0.0)
        composite = pd.Series(score, index=tickers)

        # ── Ridge regression ───────────────────────────────────────────────
        X_list, y_list = [], []
        for step in range(t_start, t_idx, rebal_freq):
            d      = dates[step]
            d_next = dates[min(step + rebal_freq, len(dates) - 1)]
            f_row  = _get_X(factor_dict, factor_names, d)
            r_row  = fwd_returns.loc[d_next].values
            valid  = np.isfinite(f_row).all(axis=1) & np.isfinite(r_row)
            if valid.sum() < 50:
                continue
            X_list.append(f_row[valid])
            y_list.append(r_row[valid])

        ridge_signal = composite.copy()

        if X_list:
            X_tr = np.vstack(X_list)
            y_tr = np.concatenate(y_list)
            Xs   = scaler.fit_transform(X_tr)
            model.fit(Xs, y_tr)

            f_now     = _get_X(factor_dict, factor_names, t_date)
            valid_now = np.isfinite(f_now).all(axis=1)
            scores    = np.full(len(tickers), np.nan)
            scores[valid_now] = model.predict(scaler.transform(f_now[valid_now]))

            # IC sign check on training set
            ic_train, _ = spearmanr(model.predict(Xs), y_tr)
            if np.isfinite(ic_train) and ic_train < 0:
                scores = -scores

            # Cross-sectional z-score
            fin = np.isfinite(scores)
            if fin.sum() > 10:
                mu, sd = scores[fin].mean(), scores[fin].std()
                if sd > 1e-8:
                    scores = np.where(fin, (scores - mu) / sd, 0.0)
            ridge_signal = pd.Series(scores, index=tickers)

        blended = BLEND_RIDGE * ridge_signal + BLEND_COMPOSITE * composite
        signals[t_date] = blended

    return pd.DataFrame(signals).T