"""PIT pair discovery per formation window, one function per family.

All selection statistics use ONLY prices within [formation_start,
formation_end]; membership is members_as_of(test_start). Full-sample
diagnostics exist only in the descriptive inventory (run_relvalue_study.py
discover), never for trading eligibility.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import ALL_ETFS, SECTOR_ETF, SHARE_CLASS_PAIRS
from .diagnostics import crossings, half_life, ols_hedge, pair_diagnostics
from .signals import PairSpec

MIN_COVERAGE = 0.95


def _formation_slice(tr: pd.DataFrame, tickers: list[str],
                     f_start: str, f_end: str) -> pd.DataFrame:
    cols = [t for t in dict.fromkeys(tickers) if t in tr.columns]
    fpx = tr.loc[(tr.index >= f_start) & (tr.index <= f_end), cols]
    fpx = fpx.dropna(how="all")             # holiday-row gotcha
    if fpx.empty:
        return fpx
    ok = fpx.columns[(fpx.notna().mean() >= MIN_COVERAGE)
                     & fpx.iloc[0].notna() & fpx.iloc[-1].notna()]
    return fpx[ok].ffill()


def _resid_spec(fpx: pd.DataFrame, a: str, b: str, family: str,
                sector: str, diag: dict | None = None) -> PairSpec:
    la, lb = np.log(fpx[a]), np.log(fpx[b])
    alpha, beta, resid = ols_hedge(la, lb)
    return PairSpec(a, b, family, sector, alpha=alpha, beta=beta,
                    mu=float(resid.mean()), sigma=float(resid.std()),
                    diag=diag or {})


def coint_pairs(tr: pd.DataFrame, members: list[str], sectors: pd.Series,
                f_start: str, f_end: str, top_corr: int = 200,
                max_pairs: int = 20, pmax: float = 0.05,
                hl_range: tuple[float, float] = (5.0, 60.0),
                min_crossings: int = 6) -> list[PairSpec]:
    """F1: same-sector pairs prefiltered by return correlation, then required
    to pass Engle-Granger (p ≤ pmax) with an investable residual half-life.
    Ranked by EG p-value; one appearance per ticker."""
    from statsmodels.tsa.stattools import coint as eg_test
    fpx = _formation_slice(tr, members, f_start, f_end)
    if fpx.empty:
        return []
    rets = fpx.pct_change(fill_method=None).iloc[1:].fillna(0.0)
    sec = sectors.reindex(fpx.columns).fillna("Unknown")
    cands: list[tuple[float, str, str, str]] = []
    for s, group in sec.groupby(sec):
        names = sorted(group.index)
        if len(names) < 2:
            continue
        c = np.corrcoef(rets[names].to_numpy(), rowvar=False)
        iu = np.triu_indices(len(names), 1)
        for i, j in zip(*iu):
            cands.append((float(c[i, j]), names[i], names[j], s))
    cands.sort(reverse=True)
    out: list[tuple[float, PairSpec]] = []
    for corr, a, b, s in cands[:top_corr]:
        la, lb = np.log(fpx[a]), np.log(fpx[b])
        try:
            p = float(eg_test(la.to_numpy(), lb.to_numpy())[1])
        except Exception:
            continue
        if p > pmax:
            continue
        _, _, resid = ols_hedge(la, lb)
        hl = half_life(resid)
        if not (hl_range[0] <= hl <= hl_range[1]):
            continue
        if crossings(resid) < min_crossings:
            continue
        spec = _resid_spec(fpx, a, b, "coint", s,
                           {"eg_p": p, "half_life": hl, "ret_corr": corr})
        out.append((p, spec))
    out.sort(key=lambda t: t[0])
    used: set[str] = set()
    picked = []
    for _, spec in out:
        if spec.a in used or spec.b in used or spec.sigma <= 0:
            continue
        picked.append(spec)
        used.update((spec.a, spec.b))
        if len(picked) >= max_pairs:
            break
    return picked


def stock_etf_pairs(tr: pd.DataFrame, members: list[str], sectors: pd.Series,
                    f_start: str, f_end: str, max_pairs: int = 30,
                    adf_max: float = 0.10,
                    hl_range: tuple[float, float] = (5.0, 60.0),
                    min_beta: float = 0.5) -> list[PairSpec]:
    """F2: each member vs its own sector ETF (residual of log stock on log
    ETF). Requires a mean-reverting residual (ADF ≤ adf_max, half-life in
    range). ETF legs may repeat; stock legs are unique."""
    from statsmodels.tsa.stattools import adfuller
    etfs = sorted(set(SECTOR_ETF.values()))
    fpx = _formation_slice(tr, list(members) + etfs, f_start, f_end)
    if fpx.empty:
        return []
    scored: list[tuple[float, PairSpec]] = []
    for t in members:
        s = sectors.get(t, "Unknown")
        etf = SECTOR_ETF.get(s)
        if t not in fpx.columns or etf not in fpx.columns:
            continue
        la, lb = np.log(fpx[t]), np.log(fpx[etf])
        alpha, beta, resid = ols_hedge(la, lb)
        if abs(beta) < min_beta:
            continue
        hl = half_life(resid)
        if not (hl_range[0] <= hl <= hl_range[1]):
            continue
        try:
            p = float(adfuller(resid.to_numpy(),
                               maxlag=int(len(resid) ** (1 / 3)),
                               autolag=None)[1])
        except Exception:
            continue
        if p > adf_max:
            continue
        spec = PairSpec(t, etf, "setf", s, alpha=alpha, beta=beta,
                        mu=float(resid.mean()), sigma=float(resid.std()),
                        diag={"adf_p": p, "half_life": hl})
        scored.append((p, spec))
    scored.sort(key=lambda x: x[0])
    return [s for _, s in scored[:max_pairs]]


def etf_etf_pairs(tr: pd.DataFrame, f_start: str, f_end: str,
                  max_pairs: int = 10, pmax: float = 0.10,
                  hl_range: tuple[float, float] = (5.0, 90.0)) -> list[PairSpec]:
    """F3: all ETF×ETF combinations, Engle-Granger filtered (no same-ticker
    cap — the ETF set is small and pairs are economically distinct)."""
    from statsmodels.tsa.stattools import coint as eg_test
    fpx = _formation_slice(tr, ALL_ETFS, f_start, f_end)
    out: list[tuple[float, PairSpec]] = []
    cols = list(fpx.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            la, lb = np.log(fpx[a]), np.log(fpx[b])
            try:
                p = float(eg_test(la.to_numpy(), lb.to_numpy())[1])
            except Exception:
                continue
            if p > pmax:
                continue
            _, _, resid = ols_hedge(la, lb)
            hl = half_life(resid)
            if not (hl_range[0] <= hl <= hl_range[1]):
                continue
            out.append((p, _resid_spec(fpx, a, b, "etf", "ETF",
                                       {"eg_p": p, "half_life": hl})))
    out.sort(key=lambda t: t[0])
    return [s for _, s in out[:max_pairs]]


def share_class_pairs(tr: pd.DataFrame, f_start: str, f_end: str) -> list[PairSpec]:
    """F4: dual share classes — log-spread spec (beta pinned to 1)."""
    out = []
    for a, b in SHARE_CLASS_PAIRS:
        fpx = _formation_slice(tr, [a, b], f_start, f_end)
        if len(fpx.columns) < 2:
            continue
        ls = np.log(fpx[a]) - np.log(fpx[b])
        sigma = float(ls.std())
        if sigma <= 0:
            continue
        out.append(PairSpec(a, b, "dual", "Dual", alpha=0.0, beta=1.0,
                            mu=float(ls.mean()), sigma=sigma))
    return out


def ssd_pairs_specs(px_raw: pd.DataFrame, tr: pd.DataFrame, members, sectors,
                    f_start: str, f_end: str, n: int = 20) -> list[PairSpec]:
    """F5 base: reuse the validated SSD formation (pairtrading.form_pairs) on
    the CLEAN TR panel, emitted as PairSpecs (distance-signal convention:
    sigma in normalized-price units, anchored at formation end)."""
    from pairtrading.study import form_pairs
    pairs = form_pairs(tr, members, sectors, f_start, f_end, n_candidates=n)
    return [PairSpec(p.a, p.b, "ssd", p.sector, sigma=p.sigma,
                     diag={"ssd": p.ssd, "ret_corr": p.corr}) for p in pairs]


def inventory_diagnostics(tr: pd.DataFrame, specs: list[PairSpec],
                          start: str, end: str) -> pd.DataFrame:
    """Descriptive battery over [start, end] for the candidate inventory."""
    rows = []
    for s in specs:
        if s.a not in tr.columns or s.b not in tr.columns:
            continue
        win = tr.loc[(tr.index >= start) & (tr.index <= end)]
        d = pair_diagnostics(win[s.a], win[s.b])
        if d:
            rows.append({"a": s.a, "b": s.b, "family": s.family,
                         "sector": s.sector, **d})
    return pd.DataFrame(rows)
