"""Bucket statistics with overlap-aware significance.

The forward window (21 trading days) is ~4x longer than the sampling step
(1 week), so consecutive observations share most of their return path. Naive
t-tests on this data are badly oversized. Two corrections are reported:

* Newey-West HAC t-stat on the group dummy (lag 4 weeks = the overlap length).
* Circular block bootstrap CI/p-value (mean block 8 weeks), which also survives
  the fat tails and volatility clustering that HAC standard errors assume away.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

NW_LAG = 4
BLOCK = 8
N_BOOT = 5000
SEED = 20260725


def _describe(x: np.ndarray) -> dict:
    if len(x) == 0:
        # a degenerate split (e.g. SPY's beta to itself is always 1.0, so
        # "beta >= 1.25" never fires) — report it rather than crashing
        return {"n": 0, "mean": np.nan, "median": np.nan, "std": np.nan,
                "hit_rate": np.nan, "p05": np.nan, "p95": np.nan}
    return {
        "n": len(x),
        "mean": np.mean(x),
        "median": np.median(x),
        "std": np.std(x, ddof=1) if len(x) > 1 else np.nan,
        "hit_rate": float(np.mean(x > 0)),
        "p05": np.percentile(x, 5),
        "p95": np.percentile(x, 95),
    }


def _newey_west_t(mask: np.ndarray, y: np.ndarray) -> float:
    """HAC t-stat on the ON-dummy from y ~ 1 + dummy."""
    X = sm.add_constant(mask.astype(float))
    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAG})
    return float(fit.tvalues[1])


def _block_bootstrap(mask: np.ndarray, y: np.ndarray, stat) -> np.ndarray:
    """Circular block bootstrap of ``stat(on, off)``, preserving time ordering.

    Blocks are drawn from the *joint* (mask, y) series, so the on/off split
    counts move with each replicate — that is deliberate: the uncertainty in
    how often the signal fires is part of the uncertainty in its effect.
    """
    rng = np.random.default_rng(SEED)
    n = len(y)
    n_blocks = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, n, size=(N_BOOT, n_blocks))
    offsets = np.arange(BLOCK)
    idx = (starts[:, :, None] + offsets[None, None, :]).reshape(N_BOOT, -1)[:, :n] % n
    out = np.full(N_BOOT, np.nan)
    for b in range(N_BOOT):
        m, v = mask[idx[b]], y[idx[b]]
        if m.sum() < 2 or (~m).sum() < 2:
            continue
        out[b] = stat(v[m], v[~m])
    return out[~np.isnan(out)]


def compare(mask: pd.Series, fwd: pd.Series, label: str, on_name: str,
            off_name: str) -> dict:
    """Compare forward returns when ``mask`` is True vs False."""
    ok = mask.notna() & fwd.notna()
    m = mask[ok].astype(bool).to_numpy()
    y = fwd[ok].to_numpy()
    on, off = _describe(y[m]), _describe(y[~m])

    res = {"signal": label, "on_label": on_name, "off_label": off_name}
    res.update({f"on_{k}": v for k, v in on.items()})
    res.update({f"off_{k}": v for k, v in off.items()})
    res["mean_diff"] = on["mean"] - off["mean"]
    res["median_diff"] = on["median"] - off["median"]

    if on["n"] >= 10 and off["n"] >= 10:
        res["nw_t"] = _newey_west_t(m, y)
        boot = _block_bootstrap(m, y, lambda a, b: a.mean() - b.mean())
        res["boot_lo"], res["boot_hi"] = np.percentile(boot, [2.5, 97.5])
        # two-sided bootstrap p: how much of the distribution sits past zero
        frac = np.mean(boot <= 0)
        res["boot_p"] = float(min(1.0, 2 * min(frac, 1 - frac)))
    else:
        res["nw_t"] = res["boot_lo"] = res["boot_hi"] = res["boot_p"] = np.nan
    return res


def rank_ic(sig: pd.Series, fwd: pd.Series) -> dict:
    """Spearman correlation between the signal level and next-month return.

    A threshold test sees one cut point and throws away the shape; this asks
    whether the whole signal ranks the forward month at all.

    The bootstrap ranks once and then resamples blocks of the *ranks*, taking
    Pearson on each replicate. That is the standard fast equivalent of a
    Spearman bootstrap (Spearman IS Pearson-on-ranks) and it vectorises, which
    matters at 5000 replicates x dozens of slices.
    """
    ok = sig.notna() & fwd.notna()
    x = sig[ok].rank().to_numpy(dtype=float)
    y = fwd[ok].rank().to_numpy(dtype=float)
    n = len(y)
    if n < 20:
        return {"n": int(n), "rho": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "p": float("nan")}
    rho = float(np.corrcoef(x, y)[0, 1])

    rng = np.random.default_rng(SEED)
    n_blocks = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, n, size=(N_BOOT, n_blocks))
    idx = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]
           ).reshape(N_BOOT, -1)[:, :n] % n
    xb, yb = x[idx], y[idx]
    xc = xb - xb.mean(axis=1, keepdims=True)
    yc = yb - yb.mean(axis=1, keepdims=True)
    denom = np.sqrt((xc ** 2).sum(1) * (yc ** 2).sum(1))
    boot = np.divide((xc * yc).sum(1), denom, out=np.full(N_BOOT, np.nan),
                     where=denom > 0)
    boot = boot[~np.isnan(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    frac = np.mean(boot <= 0)
    return {"n": int(n), "rho": rho, "lo": float(lo), "hi": float(hi),
            "p": float(min(1.0, 2 * min(frac, 1 - frac)))}


def table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)
