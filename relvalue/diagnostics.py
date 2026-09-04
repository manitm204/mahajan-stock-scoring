"""Pair-relationship diagnostics.

Computed on a FORMATION window only (PIT) for eligibility/selection, and on
the full sample only for the descriptive candidate inventory (clearly labeled
as such — never used for trading decisions).

Distinctions maintained (see program brief): return correlation vs price-level
fit (R² of log-level OLS) vs cointegration (EG/ADF on the residual) vs hedge
ratio stability (rolling-beta dispersion) vs mean reversion speed (AR(1)
half-life, zero crossings) vs long-memory persistence (variance-ratio Hurst).
High correlation or R² alone is candidate generation, not evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint


def half_life(spread: pd.Series) -> float:
    s0, s1 = spread.shift(1).iloc[1:], spread.iloc[1:]
    var = s0.var()
    if not var or np.isnan(var):
        return float("inf")
    b = ((s1 - s0) * (s0 - s0.mean())).mean() / var
    if b >= 0 or b <= -1:
        return float("inf")
    return float(-np.log(2) / np.log(1 + b))


def crossings(spread: pd.Series) -> int:
    sgn = np.sign((spread - spread.mean()).to_numpy())
    sgn = sgn[sgn != 0]
    return int((np.diff(sgn) != 0).sum())


def hurst_vr(spread: pd.Series, lags=(2, 4, 8, 16, 32)) -> float:
    """Variance-ratio Hurst exponent of the spread (<0.5 = mean-reverting)."""
    x = spread.dropna().to_numpy()
    if len(x) < 4 * max(lags):
        return float("nan")
    taus = [np.std(x[l:] - x[:-l]) for l in lags]
    if any(t <= 0 for t in taus):
        return float("nan")
    return float(np.polyfit(np.log(lags), np.log(taus), 1)[0])


def ols_hedge(la: pd.Series, lb: pd.Series) -> tuple[float, float, pd.Series]:
    """OLS log(A) = alpha + beta*log(B); returns (alpha, beta, residual)."""
    x = np.vstack([np.ones(len(lb)), lb.to_numpy()]).T
    coef, *_ = np.linalg.lstsq(x, la.to_numpy(), rcond=None)
    resid = la - (coef[0] + coef[1] * lb)
    return float(coef[0]), float(coef[1]), resid


def pair_diagnostics(pa: pd.Series, pb: pd.Series,
                     full_tests: bool = True) -> dict:
    """Diagnostic battery on two aligned price series (one window)."""
    df = pd.concat([pa, pb], axis=1, keys=["a", "b"]).dropna()
    if len(df) < 60:
        return {}
    la, lb = np.log(df["a"]), np.log(df["b"])
    ra, rb = df["a"].pct_change().iloc[1:], df["b"].pct_change().iloc[1:]
    corr = float(ra.corr(rb))
    roll = ra.rolling(63).corr(rb).dropna()
    alpha, beta, resid = ols_hedge(la, lb)
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((la - la.mean()) ** 2).sum())
    r2_level = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rb_beta = _rolling_beta(la, lb, 126)
    out = {
        "n_days": len(df),
        "ret_corr": corr,
        "ret_r2": corr ** 2,
        "roll_corr_mean": float(roll.mean()) if len(roll) else float("nan"),
        "roll_corr_std": float(roll.std()) if len(roll) else float("nan"),
        "level_r2": r2_level,
        "beta": beta,
        "alpha": alpha,
        "beta_roll_std": rb_beta,
        "resid_sigma": float(resid.std()),
        "half_life": half_life(resid),
        "crossings": crossings(resid),
        "hurst": hurst_vr(resid),
    }
    if full_tests:
        try:
            out["adf_p"] = float(adfuller(resid.to_numpy(), maxlag=int(len(resid) ** (1 / 3)),
                                          autolag=None)[1])
        except Exception:
            out["adf_p"] = float("nan")
        try:
            out["eg_p"] = float(coint(la.to_numpy(), lb.to_numpy())[1])
        except Exception:
            out["eg_p"] = float("nan")
    return out


def _rolling_beta(la: pd.Series, lb: pd.Series, window: int) -> float:
    if len(la) < window * 2:
        return float("nan")
    cov = la.rolling(window).cov(lb)
    var = lb.rolling(window).var()
    rb = (cov / var).dropna()
    return float(rb.std()) if len(rb) else float("nan")
