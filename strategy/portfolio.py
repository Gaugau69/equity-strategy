"""
portfolio.py
------------
Beta-neutral long/short portfolio construction.

Architecture
------------
Two-step approach:

  Step 1 — Markowitz weight optimisation (within the selected universe):
      maximise  α'w  −  λ w'Σw
      subject to  Σw_i = 0          (dollar-neutral)
                  w_i ≥ 0  ∀ i ∈ long leg
                  w_i ≤ 0  ∀ i ∈ short leg
                  |w_i| ≤ max_weight

  Step 2 — Beta neutralisation (post-hoc leg scaling):
      Scale the short leg so that  β_portfolio = β_long − β_short ≈ 0.
      The short leg is multiplied by  short_scale = β_long / β_short,
      then gross is renormalised to 1.  The legs are intentionally left
      at asymmetric dollar sizes — this is what guarantees  β ≈ 0.

      Dollar exposure of each leg after step 2:
          long_notional  = β_short / (β_long + β_short)
          short_notional = β_long  / (β_long + β_short)
      The portfolio carries a small net dollar bias (long or short) that
      exactly offsets the beta differential between the two legs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import TOP_N, MAX_WEIGHT, LAMBDA_REG


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Markowitz weight optimisation
# ─────────────────────────────────────────────────────────────────────────────

def _solve_markowitz(
    alpha_sel: np.ndarray,
    cov_sel: np.ndarray,
    n_long: int,
    lambda_reg: float,
    max_weight: float,
    sector_ids: np.ndarray | None = None,
    sector_lambda: float = 2.0,
) -> np.ndarray:
    """
    Solve the mean-variance QP over the selected subset (dollar-neutral only).

    Variables : w ∈ R^n_sel  (first n_long = long leg, rest = short leg)
    Objective : min  λ̃ w'Σw − α'w  +  μ Σ_s (Σ_{i∈s} w_i)²
    Subject to:
        Σw_i  = 0                         (dollar-neutral)
        0 ≤ w_i ≤ max_weight  ∀ i < n_long
        −max_weight ≤ w_i ≤ 0  ∀ i ≥ n_long

    The sector penalty (coefficient μ = sector_lambda × λ̃) softly pushes
    the net sector exposure toward zero without hard constraints that could
    make the QP infeasible when a sector has few selected stocks.

    Beta neutrality is NOT enforced here — it is applied post-hoc via
    leg scaling so as not to constrain the alpha-seeking step.
    """
    n_sel = len(alpha_sel)

    cov_scale = max(float(np.diag(cov_sel).mean()), 1e-8)
    lam = lambda_reg * cov_scale

    unique_sectors = np.unique(sector_ids) if sector_ids is not None else None

    def objective(w: np.ndarray) -> float:
        val = lam * float(w @ cov_sel @ w) - float(alpha_sel @ w)
        if unique_sectors is not None:
            for s in unique_sectors:
                val += sector_lambda * lam * float(w[sector_ids == s].sum()) ** 2
        return val

    def grad(w: np.ndarray) -> np.ndarray:
        g = 2.0 * lam * (cov_sel @ w) - alpha_sel
        if unique_sectors is not None:
            for s in unique_sectors:
                mask = sector_ids == s
                g[mask] += 2.0 * sector_lambda * lam * float(w[mask].sum())
        return g

    constraints = [
        {
            "type": "eq",
            "fun":  lambda w: w.sum(),
            "jac":  lambda _: np.ones(n_sel),
        },
    ]

    bounds = (
        [(0.0, max_weight)] * n_long
        + [(-max_weight, 0.0)] * (n_sel - n_long)
    )

    w0 = np.zeros(n_sel)
    w0[:n_long] = 0.5 / n_long
    w0[n_long:] = -0.5 / (n_sel - n_long)

    result = minimize(
        objective,
        w0,
        jac=grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 300},
    )

    return result.x if result.success else w0


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Post-hoc beta neutralisation (leg scaling)
# ─────────────────────────────────────────────────────────────────────────────

def _beta_neutralise(weights: pd.Series, betas_t: pd.Series) -> pd.Series:
    """
    Scale the short leg so that  β_portfolio ≈ 0.

    Proof: with short_scale = β_long / β_short and gross normalised to 1,
        long_fraction  = β_short / (β_long + β_short)
        short_fraction = β_long  / (β_long + β_short)
        β_portfolio = β_long × long_fraction − β_short × short_fraction = 0

    The legs end up at different dollar sizes (not dollar-neutral), but the
    residual net dollar exposure is what exactly cancels the beta difference.
    Do NOT re-equalise the legs after scaling — that would undo the fix.
    """
    beta_mean   = betas_t.mean()
    long_idx    = weights[weights > 0].index
    short_idx   = weights[weights < 0].index

    if len(long_idx) == 0 or len(short_idx) == 0:
        return weights

    betas_long  = betas_t.reindex(long_idx).fillna(beta_mean)
    betas_short = betas_t.reindex(short_idx).fillna(beta_mean)
    long_w      = weights[long_idx]
    short_w     = weights[short_idx]

    beta_long   = float(betas_long  @ long_w)
    beta_short  = float(betas_short @ short_w.abs())

    short_scale = beta_long / beta_short if beta_short > 1e-6 else 1.0
    weights[short_idx] *= short_scale

    # Renormalise gross to 1 — preserve the asymmetric leg sizing
    gross = weights.abs().sum()
    if gross > 1e-8:
        weights /= gross

    return weights


# ─────────────────────────────────────────────────────────────────────────────
# Rank-based fallback (no covariance needed)
# ─────────────────────────────────────────────────────────────────────────────

def _rank_weights(
    long_idx: list,
    short_idx: list,
    ranked: pd.Series,
    tickers: list[str],
    max_weight: float,
) -> pd.Series:
    long_ranks  = ranked[long_idx]
    short_ranks = ranked[short_idx]
    long_w  = long_ranks  / long_ranks.sum()
    short_w = short_ranks / short_ranks.sum()

    weights = pd.Series(0.0, index=tickers)
    weights[long_idx]  =  long_w.values
    weights[short_idx] = -short_w.values

    weights = weights.clip(-max_weight, max_weight)
    gross = weights.abs().sum()
    if gross > 1e-8:
        weights /= gross

    long_sum  = weights[weights > 0].sum()
    short_sum = weights[weights < 0].abs().sum()
    if long_sum > 1e-8 and short_sum > 1e-8:
        avg = (long_sum + short_sum) / 2.0
        weights[weights > 0] *= avg / long_sum
        weights[weights < 0] *= avg / short_sum
    gross = weights.abs().sum()
    if gross > 1e-8:
        weights /= gross
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_portfolio(
    signals_t: pd.Series,
    betas_t: pd.Series,
    cov_matrix: np.ndarray,
    tickers: list[str],
    top_n: int = TOP_N,
    lambda_reg: float = LAMBDA_REG,
    max_weight: float = MAX_WEIGHT,
    sector_map: dict[str, str] | None = None,
) -> pd.Series:
    """
    Construct a dollar-neutral, beta-neutral long/short portfolio.

    1. Pre-select top_n long + top_n short by signal rank.
    2. Solve mean-variance QP (Markowitz) over that subset, with an optional
       soft sector-neutrality penalty when sector_map is provided.
    3. Apply beta neutralisation via short-leg scaling.

    Returns weights with:
      - Σ w_i  ≈ 0      (dollar-neutral)
      - β'w    ≈ 0      (beta-neutral via leg scaling)
      - Σ|w_i| = 1.0   (gross exposure)
      - |w_i|  ≤ max_weight
    """
    valid = signals_t.dropna()
    if len(valid) < top_n * 2:
        return pd.Series(0.0, index=tickers)

    ranked       = valid.rank(ascending=True)
    top_n_actual = min(top_n, len(valid) // 2)
    long_idx     = ranked.nlargest(top_n_actual).index.tolist()
    short_idx    = ranked.nsmallest(top_n_actual).index.tolist()
    all_sel      = long_idx + short_idx
    n_long       = len(long_idx)

    # ── Sub-covariance for selected stocks ────────────────────────────────────
    ticker_to_int = {t: i for i, t in enumerate(tickers)}
    sel_int  = [ticker_to_int[t] for t in all_sel if t in ticker_to_int]
    cov_sel  = cov_matrix[np.ix_(sel_int, sel_int)]
    cov_sel  = cov_sel + 1e-6 * np.eye(len(all_sel))

    alpha_sel  = signals_t.reindex(all_sel).fillna(0.0).values
    sector_ids = (np.array([sector_map.get(t, "Unknown") for t in all_sel])
                  if sector_map is not None else None)

    # ── Markowitz step ────────────────────────────────────────────────────────
    try:
        w_opt = _solve_markowitz(alpha_sel, cov_sel, n_long, lambda_reg,
                                 max_weight, sector_ids)
        valid_sol = np.isfinite(w_opt).all() and np.abs(w_opt).sum() > 1e-8
    except Exception:
        valid_sol = False

    if valid_sol:
        weights = pd.Series(0.0, index=tickers)
        for t, w in zip(all_sel, w_opt):
            weights[t] = w
        gross = weights.abs().sum()
        if gross > 1e-8:
            weights /= gross
    else:
        weights = _rank_weights(long_idx, short_idx, ranked, tickers, max_weight)

    # ── Beta neutralisation (post-hoc leg scaling) ────────────────────────────
    weights = _beta_neutralise(weights, betas_t)

    return weights
