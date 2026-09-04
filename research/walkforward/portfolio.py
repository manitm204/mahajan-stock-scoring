"""A small, self-contained out-of-sample portfolio simulator.

Given the frozen composite scores per rebalance date and the adjusted-close price
matrix, it forms top-percentile portfolios and realises the returns they earn between
rebalances. It is deliberately *not* the production backtester
(:mod:`backtesting.engine`): that engine is driven by config-time weights, risk overlays
and holdout logic that would muddy a clean OOS read. Here the only inputs are the frozen
scores + prices, so the equity curve reflects the factor edge and nothing else.

Supports the constructions the study compares:

* **top_pct** ∈ {0.10, 0.20, 0.30} — long the top decile/quintile/tercile by score;
* **mode** ``equal`` (1/N) or ``sector_neutral`` (each GICS sector weighted to its share
  of the scored universe, names equal-weighted within sector — neutralises sector bets);
* **short** — dollar-neutral long-top / short-bottom overlay (Q5);
* **hold_months** ∈ {1, 3, 6, 12} — non-overlapping holding period (Q4).

Metrics: CAGR, annualised Sharpe, annualised volatility, max drawdown (via
:func:`backtesting.metrics.max_drawdown`), average turnover and period hit rate, with a
SPY benchmark on the identical rebalance grid.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtesting.data_loader import BENCHMARKS, SPY
from backtesting.metrics import max_drawdown
from research.forward_returns import realize_delistings


@dataclass
class SimResult:
    label: str
    period_returns: pd.Series          # index = period-end date, one realised return each
    equity: pd.Series                  # compounded, starts at 1.0
    turnover: pd.Series                # per-rebalance one-way turnover
    bench_period_returns: dict         # {benchmark: per-period return Series}
    bench_equity: dict                 # {benchmark: compounded equity Series}
    metrics: dict = field(default_factory=dict)

    # -- back-compat accessors (older callers/tests read the SPY leg directly) --
    @property
    def spy_period_returns(self) -> pd.Series:
        return self.bench_period_returns.get(SPY, pd.Series(dtype=float))

    @property
    def spy_equity(self) -> pd.Series:
        return self.bench_equity.get(SPY, pd.Series(dtype=float))


# --------------------------------------------------------------------------- #
# Weighting
# --------------------------------------------------------------------------- #
def _equal_weights(names: list[str]) -> pd.Series:
    if not names:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(names), index=names)


def sector_neutral_weights(names: list[str], universe_names: list[str],
                           sectors: pd.Series) -> pd.Series:
    """Weight ``names`` so each sector's aggregate weight equals its share of
    ``universe_names`` (the scored universe), names equal-weighted within sector.

    Sectors present in the universe but with no selected name have their target weight
    redistributed proportionally across the sectors that *do* hold selected names, so the
    book always sums to 1 and carries no unintended sector tilt.
    """
    if not names:
        return pd.Series(dtype=float)
    sec = sectors.reindex(universe_names).fillna("Unknown")
    uni_share = sec.value_counts(normalize=True)          # target sector weights
    sel_sec = sectors.reindex(names).fillna("Unknown")
    present = sel_sec.unique().tolist()
    # Renormalise the universe shares onto the sectors we can actually fill.
    fillable = uni_share.reindex(present).fillna(0.0)
    if fillable.sum() <= 0:
        return _equal_weights(names)
    fillable = fillable / fillable.sum()
    w = pd.Series(0.0, index=names)
    for s in present:
        members = sel_sec.index[sel_sec == s].tolist()
        if members:
            w.loc[members] = fillable[s] / len(members)
    total = w.sum()
    return w / total if total > 0 else _equal_weights(names)


def _select(score: pd.Series, top_pct: float, top: bool) -> list[str]:
    s = score.dropna()
    if s.empty:
        return []
    k = max(1, int(round(len(s) * top_pct)))
    ordered = s.sort_values(ascending=not top)   # top=True → highest first
    return ordered.index[:k].tolist()


def _leg_weights(names: list[str], universe_names: list[str], sectors: pd.Series,
                 mode: str) -> pd.Series:
    if mode == "equal":
        return _equal_weights(names)
    if mode == "sector_neutral":
        return sector_neutral_weights(names, universe_names, sectors)
    raise ValueError(f"unknown weighting mode: {mode!r}")


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def _period_return(weights: pd.Series, px_now: pd.Series, px_next: pd.Series) -> float:
    if weights.empty:
        return np.nan
    ret = (px_next.reindex(weights.index) / px_now.reindex(weights.index)) - 1.0
    ok = ret.notna()
    if not ok.any():
        return np.nan
    w = weights[ok]
    w = w / w.sum()                     # renormalise over names that actually priced
    return float((w * ret[ok]).sum())


def _turnover(prev: pd.Series, cur: pd.Series) -> float:
    idx = prev.index.union(cur.index)
    return float(0.5 * (cur.reindex(idx).fillna(0.0) - prev.reindex(idx).fillna(0.0)).abs().sum())


def simulate(
    scores: dict[str, pd.Series],
    price_matrix: pd.DataFrame,
    sectors: pd.Series,
    *,
    top_pct: float = 0.20,
    mode: str = "equal",
    short: bool = False,
    hold_months: int = 1,
    label: str | None = None,
) -> SimResult:
    """Form a portfolio at each holding-grid rebalance and realise its returns.

    ``scores`` is ``{date: Series[ticker → composite]}``; only dates present in
    ``price_matrix`` are usable. ``hold_months`` subsamples the rebalance grid to
    non-overlapping form dates. Returns a :class:`SimResult` with equity, per-period
    returns, turnover and metrics.
    """
    label = label or f"top{int(top_pct*100)}_{mode}{'_LS' if short else ''}_{hold_months}m"
    # Realize delistings at their last print (held as cash) rather than letting
    # _period_return renormalize the survivors — otherwise terminal losses vanish.
    price_matrix = realize_delistings(price_matrix)
    dates = [d for d in sorted(scores) if d in price_matrix.index]
    form_dates = dates[::hold_months] if hold_months > 1 else dates

    per_ret: dict[str, float] = {}
    bench_ret: dict[str, dict[str, float]] = {b: {} for b in BENCHMARKS}
    turn: dict[str, float] = {}
    prev_book = pd.Series(dtype=float)

    for i, d in enumerate(form_dates):
        # Hold until the next form date (or the last available rebalance for the tail).
        nxt = form_dates[i + 1] if i + 1 < len(form_dates) else None
        if nxt is None:
            break
        px_now, px_next = price_matrix.loc[d], price_matrix.loc[nxt]
        score = scores[d]
        universe_names = score.dropna().index.tolist()

        longs = _select(score, top_pct, top=True)
        lw = _leg_weights(longs, universe_names, sectors, mode)
        lr = _period_return(lw, px_now, px_next)

        if short:
            long_set = set(longs)
            shorts = [t for t in _select(score, top_pct, top=False) if t not in long_set]
            sw = _leg_weights(shorts, universe_names, sectors, mode)
            sr = _period_return(sw, px_now, px_next)
            per_ret[nxt] = np.nan if (np.isnan(lr) or np.isnan(sr)) else lr - sr
            # Net book (long − short) for turnover accounting; subtract aligns on the
            # index union so the result always carries unique labels even if the legs
            # ever brush (they are disjoint by construction after the exclusion above).
            book = lw.subtract(sw, fill_value=0.0)
        else:
            per_ret[nxt] = lr
            book = lw

        turn[d] = _turnover(prev_book, book)
        prev_book = book
        for b in BENCHMARKS:
            if b in px_now.index and b in px_next.index and px_now[b] > 0:
                bench_ret[b][nxt] = float(px_next[b] / px_now[b] - 1.0)

    pr = pd.Series(per_ret).sort_index()
    benches = {b: pd.Series(r).sort_index() for b, r in bench_ret.items()}
    res = SimResult(
        label=label,
        period_returns=pr,
        equity=_equity(pr),
        turnover=pd.Series(turn).sort_index(),
        bench_period_returns=benches,
        bench_equity={b: _equity(s) for b, s in benches.items()},
    )
    res.metrics = performance_metrics(pr, hold_months, res.turnover, benches)
    return res


def _equity(returns: pd.Series) -> pd.Series:
    r = returns.dropna()
    if r.empty:
        return pd.Series(dtype=float)
    return (1.0 + r).cumprod()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _sortino(r: pd.Series, ppy: float) -> float:
    """Annualised Sortino: mean / downside-deviation (target 0). Downside deviation uses
    the RMS of the negative returns, per-period, annualised by √ppy."""
    downside = r[r < 0]
    if downside.empty:
        return float("inf") if r.mean() > 0 else float("nan")
    dd = float(np.sqrt((downside ** 2).mean()))
    return float(r.mean() / dd * np.sqrt(ppy)) if dd > 0 else float("nan")


def _cagr(r: pd.Series, ppy: float) -> float:
    r = r.dropna()
    if r.empty:
        return float("nan")
    eq = (1.0 + r).cumprod()
    years = len(r) / ppy
    return float(eq.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")


def benchmark_stats(returns: pd.Series, bench: pd.Series, ppy: float) -> dict:
    """Portfolio-vs-benchmark stats on the aligned per-period returns.

    Returns ``{cagr, excess_cagr, alpha (annualised CAPM intercept), beta,
    info_ratio, tracking_error, rel_max_drawdown}``. ``beta`` = cov(p,b)/var(b);
    ``alpha`` = (mean_p − beta·mean_b)·ppy; ``info_ratio`` = mean(p−b)/std(p−b)·√ppy;
    ``rel_max_drawdown`` is the worst drawdown of the portfolio/benchmark equity ratio
    (deepest relative underperformance)."""
    df = pd.concat([returns.rename("p"), bench.rename("b")], axis=1).dropna()
    out = {k: float("nan") for k in
           ("bench_cagr", "excess_cagr", "alpha", "beta", "info_ratio",
            "tracking_error", "rel_max_drawdown")}
    if df.empty:
        return out
    p, b = df["p"], df["b"]
    out["bench_cagr"] = _cagr(b, ppy)
    out["excess_cagr"] = _cagr(p, ppy) - out["bench_cagr"] \
        if out["bench_cagr"] == out["bench_cagr"] else float("nan")
    var_b = float(b.var(ddof=1)) if len(b) > 1 else float("nan")
    if var_b and var_b > 0:
        beta = float(((p - p.mean()) * (b - b.mean())).sum() / ((b - b.mean()) ** 2).sum())
        out["beta"] = beta
        out["alpha"] = float((p.mean() - beta * b.mean()) * ppy)
    active = p - b
    te = float(active.std(ddof=1)) if len(active) > 1 else float("nan")
    out["tracking_error"] = te * np.sqrt(ppy) if te == te else float("nan")
    out["info_ratio"] = float(active.mean() / te * np.sqrt(ppy)) if te and te > 0 \
        else float("nan")
    rel_eq = (1.0 + p).cumprod() / (1.0 + b).cumprod()
    out["rel_max_drawdown"] = float(max_drawdown(rel_eq))
    return out


def performance_metrics(returns: pd.Series, hold_months: int,
                        turnover: pd.Series | None = None,
                        benchmarks: dict | None = None) -> dict:
    """CAGR / total return / Sharpe / Sortino / vol / max-DD / turnover / hit rate for a
    period-return series, plus per-benchmark excess/alpha/beta/IR/relative-DD.

    Annualisation uses ``periods_per_year = 12 / hold_months``. ``benchmarks`` maps a name
    (``SPY``/``QQQ``) to its aligned per-period return series; its stats are flattened into
    the result as ``<bench>_cagr``, ``<bench>_alpha``, ``<bench>_beta``, ``<bench>_ir`` …
    SPY's excess CAGR is also exposed as ``spy_cagr``/``excess_cagr`` for back-compat."""
    r = returns.dropna()
    ppy = 12.0 / hold_months
    out: dict = {"n_periods": int(len(r))}
    if r.empty:
        out.update({k: float("nan") for k in
                    ("cagr", "sharpe", "sortino", "ann_vol", "max_drawdown",
                     "avg_turnover", "hit_rate", "total_return", "spy_cagr",
                     "excess_cagr")})
        return out
    equity = (1.0 + r).cumprod()
    years = len(r) / ppy
    std = float(r.std(ddof=1)) if len(r) > 1 else float("nan")
    out.update({
        "total_return": float(equity.iloc[-1] - 1.0),
        "cagr": _cagr(r, ppy),
        "sharpe": float(r.mean() / std * np.sqrt(ppy)) if std and std > 0 else float("nan"),
        "sortino": _sortino(r, ppy),
        "ann_vol": std * np.sqrt(ppy) if std == std else float("nan"),
        "max_drawdown": float(max_drawdown(equity)),
        "hit_rate": float((r > 0).mean()),
        "avg_turnover": float(turnover.dropna().mean()) if turnover is not None
        and not turnover.dropna().empty else float("nan"),
    })
    benchmarks = benchmarks or {}
    for name, bench in benchmarks.items():
        if bench is None or bench.dropna().empty:
            continue
        bs = benchmark_stats(r, bench, ppy)
        key = name.lower()
        out[f"{key}_cagr"] = bs["bench_cagr"]
        out[f"{key}_excess_cagr"] = bs["excess_cagr"]
        out[f"{key}_alpha"] = bs["alpha"]
        out[f"{key}_beta"] = bs["beta"]
        out[f"{key}_ir"] = bs["info_ratio"]
        out[f"{key}_te"] = bs["tracking_error"]
        out[f"{key}_rel_max_drawdown"] = bs["rel_max_drawdown"]
    # Back-compat aliases (older report code / tests read the SPY leg by these names).
    out["spy_cagr"] = out.get("spy_cagr", float("nan"))
    out["excess_cagr"] = out.get("spy_excess_cagr", float("nan"))
    return out
