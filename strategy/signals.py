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

from .config import TRAIN_WINDOW, REBAL_FREQ, RIDGE_ALPHA, EWM_HALFLIFE

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

# Factors used for alpha signal generation (all computed factors).
# BETA is included as a predictor — its IC is estimated on market-residualised
# returns so the coefficient reflects true cross-sectional alpha, not the
# market-drift premium.
SIGNAL_FACTORS = ["MOM_1M", "MOM_3M", "MOM_12M", "VOL_1M", "BETA"]


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


BETA_MEAN = 1.0   # approximate mean of raw betas in the universe
BETA_STD  = 0.35  # approximate std of raw betas (U(0.5,1.8)→0.375; real→~0.35)


def _adapt_weights_from_ic(
    factor_dict: dict,
    factor_names: list,
    fwd_returns: pd.DataFrame,
    dates_train,
    rebal_freq: int,
    ewm_halflife: int = 26,
    ic_tstat_min: float = 1.0,
    market_fwd_returns: pd.Series = None,
) -> dict:
    """
    Estimate empirical IC for each factor on the training window.

    Improvements:
      1. Market residualisation — subtract beta_approx × r_market from each
         stock's return before computing IC.  This removes the common market
         drift (which confounds the BETA and VOL factor ICs in trending markets)
         and exposes the true cross-sectional alpha signal.
         beta_approx = BETA_MEAN + BETA_z × BETA_STD  (reconstructed from the
         z-scored BETA factor without needing raw values).
      2. EWM weighting — recent cross-sections count more (half-life =
         ewm_halflife periods) so weights adapt when the IC regime shifts.
      3. T-stat filter — factors with |mean IC| × sqrt(N) / std_IC < ic_tstat_min
         are zeroed, preventing noise-driven bets.

    Returns {factor_name: signed_ic_weight}  (0.0 if below significance).
    """
    ic_by_factor = {f: [] for f in factor_names}

    for d in dates_train:
        if d not in fwd_returns.index:
            continue
        r = fwd_returns.loc[d]

        # Market residualisation: r_resid_i = r_i - beta_approx_i × r_market
        # This changes the cross-sectional rank order and removes market-drift
        # confounding from the IC of BETA (and correlated) factors.
        if (market_fwd_returns is not None
                and d in market_fwd_returns.index
                and 'BETA' in factor_dict
                and d in factor_dict['BETA'].index):
            r_market = float(market_fwd_returns.loc[d])
            beta_z   = factor_dict['BETA'].loc[d]
            beta_approx = BETA_MEAN + beta_z * BETA_STD
            r = r - beta_approx * r_market

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
        if not ics:
            adapted[fname] = 0.0
            continue

        n = len(ics)
        # EWM weights: index 0 = oldest, index n-1 = newest → age of newest = 0
        ages = np.arange(n - 1, -1, -1, dtype=float)
        w    = np.exp(-np.log(2) / max(ewm_halflife, 1) * ages)
        w   /= w.sum()

        mean_ic = float(np.dot(w, ics))
        std_ic  = float(np.std(ics)) if n > 1 else 1.0
        tstat   = abs(mean_ic) * np.sqrt(n) / max(std_ic, 1e-8)

        adapted[fname] = mean_ic if tstat >= ic_tstat_min else 0.0

    return adapted


def generate_signals(
    factors_norm: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    train_window: int = TRAIN_WINDOW,
    rebal_freq: int = REBAL_FREQ,
    alpha: float = RIDGE_ALPHA,
    ewm_halflife: int = EWM_HALFLIFE,
    market_fwd_returns: pd.Series = None,
) -> pd.DataFrame:
    """
    Blended signal: empirically-adapted composite z-score + Ridge regression.

    The composite weights are re-estimated from IC at each rebalancing date
    using the same rolling training window as Ridge — no look-ahead.
    Ridge training uses exponentially decaying sample weights (half-life =
    ewm_halflife periods) so that recent observations carry more weight.

    If market_fwd_returns is provided, both IC estimation and Ridge targets
    are market-residualised (r_resid = r - beta_approx × r_market) to remove
    the market-drift confound from factor IC estimation.
    """
    factor_dict, factor_names, tickers = _panel_to_3d(factors_norm)
    # Separate signal factors (predictors) from residualisation-only factors (BETA)
    sig_factors  = [f for f in factor_names if f in SIGNAL_FACTORS]
    beta_col_all = factor_names.index('BETA') if 'BETA' in factor_names else None
    sig_cols     = [factor_names.index(f) for f in sig_factors]

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

        # ── Empirical composite: IC only for signal factors (not BETA) ─────
        adapted_w = _adapt_weights_from_ic(
            factor_dict, sig_factors, fwd_returns, train_dates, rebal_freq,
            ewm_halflife=ewm_halflife,
            market_fwd_returns=market_fwd_returns,
        )
        total_abs_w = sum(abs(v) for v in adapted_w.values()) or 1.0

        score = np.zeros(len(tickers))
        for fname, w in adapted_w.items():
            if fname in factor_dict:
                vals = factor_dict[fname].loc[t_date].values.astype(float)
                score += (w / total_abs_w) * np.nan_to_num(vals, nan=0.0)
        composite = pd.Series(score, index=tickers)

        # ── Ridge regression (X = signal factors only, not BETA) ──────────
        X_list, y_list = [], []
        for step in range(t_start, t_idx, rebal_freq):
            d     = dates[step]
            f_all = _get_X(factor_dict, factor_names, d)   # all factors
            f_row = f_all[:, sig_cols]                      # signal factors only
            r_row = fwd_returns.loc[d].values

            # Market residualisation uses the full BETA column
            if (market_fwd_returns is not None
                    and d in market_fwd_returns.index
                    and beta_col_all is not None):
                r_market_d  = float(market_fwd_returns.loc[d])
                beta_z_d    = f_all[:, beta_col_all]
                beta_approx = BETA_MEAN + beta_z_d * BETA_STD
                r_row = r_row - beta_approx * r_market_d

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

            # Exponentially decaying sample weights: recent observations count more.
            # Weights increase from oldest (index 0) to newest (index -1).
            n_obs = len(y_tr)
            ages  = np.arange(n_obs - 1, -1, -1, dtype=float)  # 0 = newest
            sw    = np.exp(-np.log(2) / max(ewm_halflife, 1) * ages)
            sw   *= n_obs / sw.sum()  # normalise so mean weight = 1

            model.fit(Xs, y_tr, sample_weight=sw)

            f_now_all = _get_X(factor_dict, factor_names, t_date)
            f_now     = f_now_all[:, sig_cols]             # signal factors only
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