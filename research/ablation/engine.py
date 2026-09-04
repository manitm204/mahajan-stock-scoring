"""Config -> costed equity curve for one construction variant.

Pipeline per rebalance date: composite (VIX-tilted or not) -> exclusion screen ->
selection (with optional hysteresis) -> weighting scheme -> sector overlay ->
renormalize -> realize the period return on the delisting-realized price matrix.
Costs: ``cost_bps`` per side applied to traded volume vs the *drifted* previous
book (so both rebalance trades and forced delisting turnover are priced; the first
period pays full entry cost). All metrics are net of these costs.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from backtesting.data_loader import QQQ, SPY
from research.walkforward.portfolio import _cagr, performance_metrics

ERA_SPLIT = "2021-12-31"     # regime robustness guard (era1 vs era2)
SEARCH_END = "2023-12-31"    # 2024+ holdout — NOTE: consulted by report_full_grid's
                             # finalist pick (2026-07), so it is a *discounted* check,
                             # not a pristine gate; forward months are the clean gate
SOFTMAX_T = 1.0          # on within-book z-scores; ~3.6x weight at +1.28 sigma vs median
POSITION_CAP = 0.05      # for the capped cap-weight variant
ERC_MAX_NAMES = 200      # above this, ERC falls back to inverse-vol


@dataclass(frozen=True)
class AblationConfig:
    name: str
    top_pct: float = 0.25
    exclusion: str = "none"       # none | any_p10 | two_p10 | any_p5 | soft_p10
    weighting: str = "ew"         # ew|cap|cap5|ewcap|rank_lin|rank_sqrt|score_vol|inv_vol|erc|softmax
    sector: str = "none"          # none | neutral | equal | cap_match | bands5
    vix_tilt: bool = False
    hold_months: int = 1
    exit_pct: float | None = None  # hysteresis: hold until rank drops below this breadth
    cost_bps: float = 10.0         # one-way transaction cost per side
    exec_lag_days: int = 0         # trade N trading days after the signal close (T+N)
    final_cap: float | None = None  # hard per-name ceiling re-applied AFTER the sector
                                    # overlay (capped within each sector so sector
                                    # totals stay matched); None = no post-overlay cap


# --------------------------------------------------------------------------- #
# Exclusion
# --------------------------------------------------------------------------- #
def apply_exclusion(score: pd.Series, d: str, parent_ranks: dict, mode: str) -> pd.Series:
    if mode == "none":
        return score
    ranks = pd.DataFrame({p: per.get(d) for p, per in parent_ranks.items()
                          if per.get(d) is not None}).reindex(score.index)
    if mode == "any_p5":
        return score[(ranks < 5).sum(axis=1) == 0]
    if mode == "any_p10":
        return score[(ranks < 10).sum(axis=1) == 0]
    if mode == "two_p10":
        return score[(ranks < 10).sum(axis=1) <= 1]
    if mode == "soft_p10":   # demote instead of drop: -10 composite points if flagged
        return score - 10.0 * ((ranks < 10).sum(axis=1) > 0)
    raise ValueError(f"unknown exclusion: {mode!r}")


# --------------------------------------------------------------------------- #
# Selection (optional hysteresis)
# --------------------------------------------------------------------------- #
def select_book(score: pd.Series, top_pct: float, exit_pct: float | None,
                prev_names: list) -> list:
    s = score.dropna().sort_values(ascending=False)
    if s.empty:
        return []
    k = max(1, int(round(len(s) * top_pct)))
    if exit_pct is None or not prev_names:
        return s.index[:k].tolist()
    exit_k = max(k, int(round(len(s) * exit_pct)))
    survivors = [n for n in prev_names if n in set(s.index[:exit_k])]
    fill = [n for n in s.index if n not in set(survivors)]
    return (survivors + fill)[:max(k, len(survivors))] if len(survivors) < k \
        else survivors


# --------------------------------------------------------------------------- #
# Weighting schemes
# --------------------------------------------------------------------------- #
def _fill_median(x: pd.Series) -> pd.Series:
    med = x.median()
    return x.fillna(med if med == med else 1.0)


def _cap_weights(caps: pd.Series) -> pd.Series:
    c = _fill_median(caps)
    return c / c.sum()


def _waterfill_cap(w: pd.Series, cap: float) -> pd.Series:
    """Clip positions at ``cap`` and redistribute the excess to uncapped names."""
    if len(w) * cap < 1.0:      # infeasible: cap can't hold the book, degrade to EW
        return pd.Series(1.0 / len(w), index=w.index)
    for _ in range(20):
        over = w > cap
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w = w.clip(upper=cap)
        under = ~over
        w[under] += excess * w[under] / w[under].sum()
    return w / w.sum()


def cap_within_sectors(w: pd.Series, sectors: pd.Series, cap: float) -> pd.Series:
    """Enforce a per-name ceiling ``cap`` while preserving each sector's total weight.

    Runs the waterfill cap independently inside every sector, redistributing a name's
    excess only to its sector peers — so a book that was sector-matched by the overlay
    stays matched, but no single position exceeds ``cap`` (unless a sector's own target
    is too big to hold under the cap given its member count, in which case that sector
    degrades to equal-weight internally). ``w`` is assumed to sum to 1."""
    sec = sectors.reindex(w.index).fillna("Unknown")
    out = w.copy()
    for s in sec.unique():
        idx = sec.index[sec == s]
        grp = w.loc[idx]
        tot = float(grp.sum())
        if tot <= 0:
            continue
        capped = _waterfill_cap(grp / tot, cap / tot)   # cap in within-sector space
        out.loc[idx] = capped * tot                      # restore the sector total
    return out


def _erc_weights(names: list, matrix: pd.DataFrame, d: str) -> pd.Series:
    pos = matrix.index.get_loc(d)
    win = matrix.iloc[max(0, pos - 252):pos + 1].reindex(columns=names) \
        .pct_change(fill_method=None).iloc[1:]
    win = win.loc[:, win.notna().sum() >= 60].fillna(0.0)
    if win.shape[1] < 2:
        return pd.Series(1.0 / len(names), index=names)
    cov = win.cov().values
    cov = 0.7 * cov + 0.3 * np.diag(np.diag(cov))          # shrink toward diagonal
    n = cov.shape[0]
    w = 1.0 / np.sqrt(np.maximum(np.diag(cov), 1e-12))     # inverse-vol start
    w /= w.sum()
    for _ in range(200):
        rc = w * (cov @ w)
        adj = np.sqrt(rc.mean() / np.maximum(rc, 1e-16))
        w = np.maximum(w * adj, 0.0)
        w /= w.sum()
    out = pd.Series(w, index=win.columns).reindex(names)
    return _fill_median(out) / _fill_median(out).sum()


def base_weights(cfg: AblationConfig, names: list, score: pd.Series, d: str,
                 data) -> pd.Series:
    ew = pd.Series(1.0 / len(names), index=names)
    if cfg.weighting == "ew":
        return ew
    if cfg.weighting in ("cap", "cap5", "ewcap"):
        cw = _cap_weights(data.caps.loc[d].reindex(names))
        if cfg.weighting == "cap":
            return cw
        if cfg.weighting == "cap5":
            return _waterfill_cap(cw, POSITION_CAP)
        return 0.5 * ew + 0.5 * cw
    if cfg.weighting in ("rank_lin", "rank_sqrt"):
        rk = score.reindex(names).rank()                   # worst=1 .. best=k
        w = rk if cfg.weighting == "rank_lin" else np.sqrt(rk)
        return w / w.sum()
    if cfg.weighting in ("score_vol", "inv_vol"):
        vol = _fill_median(data.vol.loc[d].reindex(names)).clip(lower=1e-4)
        w = (score.reindex(names).clip(lower=1.0) / vol
             if cfg.weighting == "score_vol" else 1.0 / vol)
        return w / w.sum()
    if cfg.weighting == "erc":
        if len(names) > ERC_MAX_NAMES:                     # cost guard: fall back
            vol = _fill_median(data.vol.loc[d].reindex(names)).clip(lower=1e-4)
            return (1.0 / vol) / (1.0 / vol).sum()
        return _erc_weights(names, data.matrix, d)
    if cfg.weighting == "softmax":
        s = score.reindex(names)
        z = (s - s.mean()) / (s.std(ddof=0) or 1.0)
        w = np.exp(z / SOFTMAX_T)
        return w / w.sum()
    raise ValueError(f"unknown weighting: {cfg.weighting!r}")


# --------------------------------------------------------------------------- #
# Sector overlay
# --------------------------------------------------------------------------- #
def _qqq_members() -> list:
    """Current QQQ constituents, cached at cache/qqq_members.json (refresh via
    scripts/pit2020_qqq_sector.fetch_members). Today's membership — a proxy for
    'allocated like QQQ', not a historical reconstruction."""
    import json
    path = Path(__file__).resolve().parents[2] / "cache" / "qqq_members.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run scripts/pit2020_qqq_sector.fetch_members()")
    return json.loads(path.read_text())


def sector_overlay(w: pd.Series, cfg: AblationConfig, universe_names: list, d: str,
                   data) -> pd.Series:
    if cfg.sector == "none" or w.empty:
        return w
    sec = data.sectors.reindex(w.index).fillna("Unknown")
    uni_sec = data.sectors.reindex(universe_names).fillna("Unknown")
    present = sec.unique().tolist()

    if cfg.sector == "neutral":            # universe share by name count
        tgt = uni_sec.value_counts(normalize=True)
    elif cfg.sector == "equal":            # equal across sectors present in the book
        tgt = pd.Series(1.0 / len(present), index=present)
    elif cfg.sector in ("cap_match", "bands5"):   # cap-weighted universe (SPY proxy)
        ucaps = _fill_median(data.caps.loc[d].reindex(universe_names))
        tgt = ucaps.groupby(uni_sec).sum()
        tgt = tgt / tgt.sum()
    elif cfg.sector == "blend_match":      # 50/50 SPY-proxy / QQQ-proxy targets
        # User-ratified allocation policy 2026-08-07 (output/crowding/
        # incremental_weights/REPORT.md addendum 2026-08-07c): PIT-cap sector
        # shares of the universe (SPY proxy) blended equally with those of the
        # current QQQ constituents.
        ucaps = _fill_median(data.caps.loc[d].reindex(universe_names))
        spy_tgt = ucaps.groupby(uni_sec).sum()
        spy_tgt = spy_tgt / spy_tgt.sum()
        mcaps = data.caps.loc[d].reindex(
            [t for t in _qqq_members() if t in data.caps.columns]).dropna()
        qqq_tgt = mcaps.groupby(data.sectors.reindex(mcaps.index)
                                .fillna("Unknown")).sum()
        qqq_tgt = qqq_tgt / qqq_tgt.sum()
        idx = spy_tgt.index.union(qqq_tgt.index)
        tgt = (0.5 * spy_tgt.reindex(idx).fillna(0.0)
               + 0.5 * qqq_tgt.reindex(idx).fillna(0.0))
    else:
        raise ValueError(f"unknown sector mode: {cfg.sector!r}")

    tgt = tgt.reindex(present).fillna(0.0)
    if tgt.sum() <= 0:
        return w
    tgt = tgt / tgt.sum()

    cur = w.groupby(sec).sum().reindex(present).fillna(0.0)
    if cfg.sector == "bands5":             # only clamp sectors outside target +/- 5pp
        tgt = cur.clip(lower=tgt - 0.05, upper=tgt + 0.05)
        tgt = tgt / tgt.sum()

    scale = (tgt / cur.replace(0.0, np.nan)).fillna(0.0)
    out = w * sec.map(scale)
    return out / out.sum() if out.sum() > 0 else w


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def _alpha_on(p: pd.Series, b: pd.Series, ppy: float) -> float:
    """Annualized CAPM alpha (intercept) of ``p`` on ``b`` over their overlap."""
    df = pd.concat([p.rename("p"), b.rename("b")], axis=1).dropna()
    if len(df) < 3:
        return float("nan")
    x, y = df["b"].values, df["p"].values
    var = float(np.var(x, ddof=1))
    if var <= 0:
        return float("nan")
    beta = float(np.cov(y, x, ddof=1)[0, 1] / var)
    return float((y.mean() - beta * x.mean()) * ppy)


def alpha_tstat(p: pd.Series, b: pd.Series) -> float:
    df = pd.concat([p.rename("p"), b.rename("b")], axis=1).dropna()
    n = len(df)
    if n < 3:
        return float("nan")
    x, y = df["b"].values, df["p"].values
    beta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    a = y.mean() - beta * x.mean()
    resid = y - a - beta * x
    s2 = float((resid ** 2).sum() / (n - 2))
    se = np.sqrt(s2 * (1.0 / n + x.mean() ** 2 / ((x - x.mean()) ** 2).sum()))
    return float(a / se) if se > 0 else float("nan")


def _lagged_prices(matrix: pd.DataFrame, form: list, lag: int) -> pd.DataFrame:
    """Price rows ``lag`` trading days after each formation date, re-labelled with the
    formation dates so the simulation loop is lag-agnostic. lag=0 is a no-op."""
    if lag == 0:
        return matrix.loc[form]
    n = len(matrix.index)
    pos = [min(matrix.index.get_loc(f) + lag, n - 1) for f in form]
    px = matrix.iloc[pos].copy()
    px.index = form
    return px


def simulate_config(data, cfg: AblationConfig,
                    scores: dict | None = None,
                    keep_series: bool = False) -> dict:
    """``scores`` overrides the composite ({date: Series}) so alternative signals
    (score-change, interactions) reuse the identical costed pipeline.
    ``keep_series`` attaches the net/benchmark period-return series (keys prefixed
    ``_``) for downstream diagnostics — strip before writing rows to CSV."""
    if scores is None:
        scores = data.tilt_scores() if cfg.vix_tilt else data.run.pooled_scores
    form = data.rebal_dates[::cfg.hold_months]
    px = _lagged_prices(data.matrix, form, cfg.exec_lag_days)
    net, gross_r, turn, volume = {}, {}, {}, {}
    n_names, eff_n = [], []
    prev_w = pd.Series(dtype=float)
    prev_names: list = []

    for i in range(len(form) - 1):
        d, nxt = form[i], form[i + 1]
        sc = apply_exclusion(scores[d].dropna(), d, data.parent_ranks, cfg.exclusion)
        names = select_book(sc, cfg.top_pct, cfg.exit_pct, prev_names)
        if not names:
            continue
        w = base_weights(cfg, names, sc, d, data)
        w = sector_overlay(w, cfg, scores[d].dropna().index.tolist(), d, data)
        w = w / w.sum()
        if cfg.final_cap is not None:
            w = cap_within_sectors(w, data.sectors, cfg.final_cap)

        ret = (px.loc[nxt].reindex(w.index)
               / px.loc[d].reindex(w.index) - 1.0)
        ok = ret.notna()                    # only never-listed names can be NaN now
        w = w[ok] / w[ok].sum()
        ret = ret[ok]
        g = float((w * ret).sum())

        traded = float((w - prev_w.reindex(w.index.union(prev_w.index))
                        .fillna(0.0)).abs().sum())
        cost = traded * cfg.cost_bps / 1e4
        gross_r[nxt] = g
        net[nxt] = g - cost
        volume[nxt] = traded
        turn[d] = 0.5 * traded
        n_names.append(len(w))
        eff_n.append(1.0 / float((w ** 2).sum()))
        drift = w * (1.0 + ret)
        prev_w = drift / drift.sum()
        prev_names = names

    pr = pd.Series(net, dtype=float).sort_index()
    spy_all = px[SPY]
    spy = (spy_all / spy_all.shift(1) - 1.0).dropna()
    spy = spy.reindex(pr.index)
    if QQQ in px.columns:
        qqq_all = px[QQQ]
        qqq = (qqq_all / qqq_all.shift(1) - 1.0).dropna().reindex(pr.index)
    else:                    # small test fixtures may not carry QQQ
        qqq = None

    m = performance_metrics(pr, cfg.hold_months, pd.Series(turn).sort_index(),
                            {SPY: spy, QQQ: qqq})
    ppy = 12.0 / cfg.hold_months
    idx = pr.index.astype(str)
    dd = m.get("max_drawdown")
    row = {
        "config": cfg.name, "top_pct": cfg.top_pct, "exclusion": cfg.exclusion,
        "weighting": cfg.weighting, "sector": cfg.sector, "vix": cfg.vix_tilt,
        "hold_m": cfg.hold_months, "exit_pct": cfg.exit_pct,
        "net_cagr": m["cagr"], "gross_cagr": _cagr(pd.Series(gross_r).sort_index(), ppy),
        "sharpe": m["sharpe"], "sortino": m["sortino"], "max_dd": dd,
        "calmar": m["cagr"] / abs(dd) if dd and dd == dd and dd != 0 else float("nan"),
        "beta": m.get("spy_beta"), "alpha": m.get("spy_alpha"),
        "alpha_t": alpha_tstat(pr, spy), "ir": m.get("spy_ir"),
        "ex_spy": m.get("spy_excess_cagr"),
        "beta_qqq": m.get("qqq_beta"), "alpha_qqq": m.get("qqq_alpha"),
        "ex_qqq": m.get("qqq_excess_cagr"),
        "ex_spy_e1": _cagr(pr[idx <= ERA_SPLIT], ppy) - _cagr(spy[idx <= ERA_SPLIT], ppy),
        "ex_spy_e2": _cagr(pr[idx > ERA_SPLIT], ppy) - _cagr(spy[idx > ERA_SPLIT], ppy),
        "alpha_search": _alpha_on(pr[idx <= SEARCH_END], spy[idx <= SEARCH_END], ppy),
        "alpha_holdout": _alpha_on(pr[idx > SEARCH_END], spy[idx > SEARCH_END], ppy),
        "turnover": m["avg_turnover"], "avg_names": float(np.mean(n_names)),
        "eff_n": float(np.mean(eff_n)),
    }
    if keep_series:
        row["_returns"] = pr
        row["_gross"] = pd.Series(gross_r, dtype=float).sort_index()
        row["_spy"] = spy
        row["_qqq"] = qqq
    return row


def benchmark_row(data, ticker: str = SPY, hold_months: int = 1) -> dict:
    """Buy-and-hold benchmark (SPY/QQQ/...) on the same grid, for the report footer."""
    form = data.rebal_dates[::hold_months]
    all_px = data.matrix[ticker].reindex(form)
    r = (all_px / all_px.shift(1) - 1.0).dropna()
    spy_all = data.matrix[SPY].reindex(form)
    spy = (spy_all / spy_all.shift(1) - 1.0).dropna().reindex(r.index)
    m = performance_metrics(r, hold_months, None,
                            {SPY: spy} if ticker != SPY else None)
    idx = r.index.astype(str)
    ppy = 12.0 / hold_months
    vs_spy = ticker != SPY
    return {"config": ticker, "net_cagr": m["cagr"], "gross_cagr": m["cagr"],
            "sharpe": m["sharpe"], "sortino": m["sortino"],
            "max_dd": m["max_drawdown"],
            "calmar": m["cagr"] / abs(m["max_drawdown"]),
            "beta": m.get("spy_beta") if vs_spy else 1.0,
            "alpha": m.get("spy_alpha") if vs_spy else 0.0,
            "alpha_t": alpha_tstat(r, spy) if vs_spy else float("nan"),
            "ir": m.get("spy_ir") if vs_spy else float("nan"),
            "ex_spy": m.get("spy_excess_cagr") if vs_spy else 0.0,
            "ex_spy_e1": (_cagr(r[idx <= ERA_SPLIT], ppy)
                          - _cagr(spy[idx <= ERA_SPLIT], ppy)) if vs_spy else 0.0,
            "ex_spy_e2": (_cagr(r[idx > ERA_SPLIT], ppy)
                          - _cagr(spy[idx > ERA_SPLIT], ppy)) if vs_spy else 0.0,
            "alpha_search": (_alpha_on(r[idx <= SEARCH_END], spy[idx <= SEARCH_END],
                                       ppy) if vs_spy else 0.0),
            "alpha_holdout": (_alpha_on(r[idx > SEARCH_END], spy[idx > SEARCH_END],
                                        ppy) if vs_spy else 0.0),
            "turnover": 0.0,
            "avg_names": 1.0, "eff_n": 1.0, "top_pct": float("nan"),
            "exclusion": "", "weighting": "", "sector": "", "vix": False,
            "hold_m": hold_months, "exit_pct": None}


def spy_row(data, hold_months: int = 1) -> dict:
    """SPY buy-and-hold on the same grid, for the report footer."""
    return benchmark_row(data, SPY, hold_months)
