"""Precompute the equity-curve / metrics comparison for the dashboard's
Strategy Lab page (user request 2026-09-06): two loop-engineering research
strategies vs SPY/QQQ.

  v3_loopeng_cap9_minhold1wk_quartile -- research/loop_engineering/harness.py's
    managed-book engine (research/strategies/engine.py::simulate_managed_book)
    with StrategyConfig(k=10, trail_pct=0.1, cap_months=9, rank_floor_pct=0.75,
    min_hold_months=0.25) -- see manual_extension_2026-09-06.json step 2.

  21_loopeng_book4_hold4_evict3 -- the autoresearch momentum-eviction sleeve
    logic (research/autoresearch/candidate.py::_momentum_evict /
    generate_targets), replayed here with BOOK_SIZE=4, HOLD_MONTHS=4,
    REFRESH_N=3 instead of the live champion's BOOK_SIZE=11/HOLD_MONTHS=4/
    REFRESH_N=1 (this strategy never corresponded to a committed candidate.py
    state -- see research/autoresearch/session_log.md).

Both series are computed on the SAME production EQEFF composite scores and
price panel (research/strategies/engine.py::load_data), and both are
benchmarked against the same same-day-execution SPY/QQQ monthly returns
(scripts/run_strategy_sweep.py's convention) so the comparison table and
equity-curve chart use one consistent benchmark definition throughout. This
differs by construction from research/autoresearch/evaluate.py's own
T+1-lagged benchmark (used to score the live champion) by at most one
trading day per period -- immaterial for a dashboard overview, but not the
number to cite in research writeups.

Usage: python scripts/generate_strategy_lab_comparison.py
Writes: output/loop_engineering_v3/strategy_lab_comparison.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backtesting.data_loader import SPY, QQQ                       # noqa: E402
from research.autoresearch.evaluate import (                        # noqa: E402
    compute_portfolio_returns, targets_to_weight_matrix,
)
from research.strategies.engine import _top_k, load_data, simulate_managed_book  # noqa: E402
from research.strategies.generic_rule import StrategyConfig, build_rule  # noqa: E402
from research.walkforward.portfolio import performance_metrics      # noqa: E402
from scripts.run_strategy_sweep import bench_returns, monthly_returns  # noqa: E402

OUT = REPO / "output" / "loop_engineering_v3" / "strategy_lab_comparison.json"

# -- book4_hold4_evict3's own params (the live champion in candidate.py is
# BOOK_SIZE=11/HOLD_MONTHS=4/REFRESH_N=1 -- this replay uses #21's params) -- #
BOOK4_SIZE = 4
BOOK4_HOLD_MONTHS = 4
BOOK4_SLEEVE_COUNT = 3
BOOK4_REFRESH_N = 3
BOOK4_STEP = max(BOOK4_HOLD_MONTHS // BOOK4_SLEEVE_COUNT, 1)


def _fill_to_k(preferred, exclude, full_scores, need):
    out = [t for t in preferred if t not in exclude][:need]
    if len(out) < need:
        seen = set(exclude) | set(out)
        remaining = full_scores[~full_scores.index.isin(seen)]
        out += _top_k(remaining, need - len(out))
    return out


def _momentum_evict(scores, held, prev_scores, k, refresh_n):
    s = scores.dropna()
    if s.empty:
        return []
    if not held:
        return _top_k(s, k)
    held_scored = [t for t in held if t in s.index]
    if prev_scores is not None:
        decline = {t: prev_scores.get(t, s[t]) - s[t] for t in held_scored}
        worst_first = sorted(held_scored, key=lambda t: -decline[t])
    else:
        worst_first = list(reversed(_top_k(s[s.index.isin(held_scored)], len(held_scored))))
    to_drop = set(worst_first[:refresh_n])
    keep = [t for t in held_scored if t not in to_drop]
    need = k - len(keep)
    return keep + _fill_to_k(_top_k(s, len(s)), keep, s, need)


def _book4_targets(comp: dict, dates: list) -> pd.DataFrame:
    """Replay of candidate.py::generate_targets, parameterized for #21
    (BOOK_SIZE=4, HOLD_MONTHS=4, REFRESH_N=3) instead of the live champion's
    module-level constants."""
    k = BOOK4_SIZE
    sleeve_holdings = [[] for _ in range(BOOK4_SLEEVE_COUNT)]
    weight_per_name = (1.0 / BOOK4_SLEEVE_COUNT) / k
    rows = []
    for i, d in enumerate(dates):
        for j in range(BOOK4_SLEEVE_COUNT):
            offset = j * BOOK4_STEP
            if i >= offset and (i - offset) % BOOK4_HOLD_MONTHS == 0:
                scores = comp.get(d)
                prev_scores = comp.get(dates[i - BOOK4_HOLD_MONTHS]) if i >= BOOK4_HOLD_MONTHS else None
                if scores is not None:
                    sleeve_holdings[j] = _momentum_evict(
                        scores, sleeve_holdings[j], prev_scores, k, BOOK4_REFRESH_N)
        agg = {}
        for holdings in sleeve_holdings:
            for t in holdings:
                agg[t] = agg.get(t, 0.0) + weight_per_name
        for t, w in agg.items():
            rows.append({"date": d, "ticker": t, "weight": w})
    return pd.DataFrame(rows, columns=["date", "ticker", "weight"])


def _metrics(pr: pd.Series, spy: pd.Series, qqq: pd.Series) -> dict:
    m = performance_metrics(pr, hold_months=1, benchmarks={SPY: spy, QQQ: qqq})
    aligned_spy = pd.concat([pr.rename("p"), spy.rename("b")], axis=1).dropna()
    aligned_qqq = pd.concat([pr.rename("p"), qqq.rename("b")], axis=1).dropna()
    r2_spy = float(aligned_spy["p"].corr(aligned_spy["b"]) ** 2) if len(aligned_spy) > 1 else float("nan")
    r2_qqq = float(aligned_qqq["p"].corr(aligned_qqq["b"]) ** 2) if len(aligned_qqq) > 1 else float("nan")
    return {
        "cagr": m["cagr"],
        "sharpe": m["sharpe"],
        "beta_vs_spy": m.get("spy_beta"),
        "max_dd": m["max_drawdown"],
        "alpha_vs_spy": m.get("spy_alpha"),
        "r2_vs_spy": r2_spy,
        "r2_vs_qqq": r2_qqq,
    }


def main() -> None:
    print("loading composite scores + price matrix ...", flush=True)
    data = load_data()
    rebal = data.rebal_dates
    spy = bench_returns(data.matrix, SPY, rebal)
    qqq = bench_returns(data.matrix, QQQ, rebal)

    print("simulating v3_loopeng_cap9_minhold1wk_quartile ...", flush=True)
    cfg = StrategyConfig(k=10, trail_pct=0.1, cap_months=9,
                         rank_floor_pct=0.75, min_hold_months=0.25)
    nav_v3 = simulate_managed_book(
        data, build_rule(cfg), k=cfg.k, rank_entry_k=cfg.entry_rank_k,
        max_per_sector=cfg.max_per_sector, sector_map=data.sector,
        min_value_pct=cfg.min_value_pct)
    pr_v3 = monthly_returns(nav_v3, rebal)

    print("simulating 21_loopeng_book4_hold4_evict3 ...", flush=True)
    targets = _book4_targets(data.comp, rebal)
    weights = targets_to_weight_matrix(targets, rebal)
    pr_book4, _turnover = compute_portfolio_returns(weights, data.matrix, rebal)

    print("computing metrics + equity curves ...", flush=True)
    metrics = {
        "v3_loopeng_cap9_minhold1wk_quartile": _metrics(pr_v3, spy, qqq),
        "21_loopeng_book4_hold4_evict3": _metrics(pr_book4, spy, qqq),
        "SPY": _metrics(spy, spy, qqq),
        "QQQ": _metrics(qqq, spy, qqq),
    }

    all_dates = sorted(set(pr_v3.index) | set(pr_book4.index) | set(spy.index) | set(qqq.index))
    curves = pd.DataFrame(index=all_dates)
    curves["v3_loopeng_cap9_minhold1wk_quartile"] = pr_v3.reindex(all_dates)
    curves["21_loopeng_book4_hold4_evict3"] = pr_book4.reindex(all_dates)
    curves["SPY"] = spy.reindex(all_dates)
    curves["QQQ"] = qqq.reindex(all_dates)
    equity = (1.0 + curves.fillna(0.0)).cumprod()

    out = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "dates": [d if isinstance(d, str) else str(d) for d in equity.index],
        "equity": {col: equity[col].tolist() for col in equity.columns},
        "metrics": metrics,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        json.dump(out, fh, default=lambda x: None if x != x else x, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
