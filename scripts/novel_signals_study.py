"""Novel signal families study (Phase C) — see output/novel_signals/PREREGISTRATION.md.

Ten pre-registered configs, construction locked to the finalist's choices
(top 25% / cap5 / cap_match, VIX tilt OFF, monthly, 10 bps/side):

  F1  Δ-composite:   k ∈ {3,6} months, Δ-weight ∈ {0.3, 0.5, 1.0}
  F2  interactions:  min/product of (value, momentum) and (quality, revisions)

All rows share a common start (max k trimmed) so Sharpe/alpha are comparable with
the level-composite reference run under identical construction. Selection uses
SEARCH-period columns only (≤2023-12-31); holdout columns are written but not
consulted (Phase D reads them once, discounted).

Output: output/novel_signals/results.csv + RESULTS.md
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pandas as pd

REPO = Path("/home/manit/Desktop/fun_projects/mahajan_hedge_fund")
sys.path.insert(0, str(REPO))

from run_walkforward import PANEL_START, PRICE_END, _load_panel, get_db  # noqa: E402
from research.ablation import load_ablation_data  # noqa: E402
from research.ablation.engine import AblationConfig, simulate_config  # noqa: E402

OUT = REPO / "output/novel_signals"
MAX_K = 6
CONSTRUCTION = dict(top_pct=0.25, exclusion="none", weighting="cap5",
                    sector="cap_match", vix_tilt=False)


def delta_scores(level: dict, dates: list, k: int) -> dict:
    """Cross-sectional percentile of the k-month change in composite percentile."""
    out = {}
    for i, d in enumerate(dates):
        if i < k:
            continue
        prev, cur = level.get(dates[i - k]), level.get(d)
        if prev is None or cur is None:
            continue
        common = cur.index.intersection(prev.index)
        if len(common) < 50:
            continue
        out[d] = (cur[common] - prev[common]).rank(pct=True) * 100.0
    return out


def blend_scores(level: dict, delta: dict, a: float) -> dict:
    """a = weight on Δ; (1-a) on the level composite. Index = Δ's (needs history)."""
    out = {}
    for d, dl in delta.items():
        lv = level[d].reindex(dl.index)
        out[d] = (1.0 - a) * lv + a * dl
    return out


def pair_scores(p1: dict, p2: dict, how: str) -> dict:
    out = {}
    for d in sorted(set(p1) & set(p2)):
        df = pd.concat([p1[d].rename("a"), p2[d].rename("b")], axis=1).dropna()
        if len(df) < 50:
            continue
        s = df.min(axis=1) if how == "min" else df["a"] * df["b"] / 100.0
        out[d] = s
    return out


def main() -> None:
    panel = _load_panel(False)
    with get_db() as db:
        data = load_ablation_data(panel, db, PANEL_START, PRICE_END,
                                  splits="rolling5y")
    OUT.mkdir(parents=True, exist_ok=True)

    level = data.run.pooled_scores
    dates = data.rebal_dates
    common_dates = dates[MAX_K:]                 # shared period for fair comparison
    pr = data.parent_ranks

    signals: dict[str, dict] = {}
    for k in (3, 6):
        dl = delta_scores(level, dates, k)
        for a in (0.3, 0.5, 1.0):
            signals[f"F1 delta k={k} w={a}"] = blend_scores(level, dl, a)
    for how in ("min", "prod"):
        signals[f"F2 val^mom {how}"] = pair_scores(pr["value"], pr["momentum"], how)
        signals[f"F2 qual^rev {how}"] = pair_scores(pr["quality"], pr["revisions"],
                                                    how)
    signals["REF level composite"] = {d: level[d] for d in common_dates}

    rows = []
    for name, sig in signals.items():
        run_data = copy.copy(data)
        run_data.rebal_dates = [d for d in common_dates if d in sig]
        cfg = AblationConfig(name=name, **CONSTRUCTION)
        row = simulate_config(run_data, cfg, scores=sig)
        rows.append(row)
        print(f"  {name:26s} search_a {row['alpha_search']*100:+5.2f}%  "
              f"sharpe {row['sharpe']:.2f}  ex_e1 {row['ex_spy_e1']*100:+5.2f}% "
              f"(holdout written, not read)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results.csv", index=False)

    ref = df[df["config"] == "REF level composite"].iloc[0]
    # search-period sub-era split: era1 = 2017-2021, era2 = 2022-2023. The engine's
    # ex_spy_e2 covers 2022-2026 (includes holdout years); recompute-free proxy is not
    # available from the row, so the promotion check on sub-era 2022-23 is done in
    # Phase D from stored return series; here we apply bars 1 and 3 and report.
    df["bar1_search_alpha_t"] = (df["alpha_search"] > 0) & (df["alpha_t"] >= 1.5)
    df["bar3_sharpe_ge_ref"] = df["sharpe"] >= float(ref["sharpe"])

    with (OUT / "RESULTS.md").open("w") as fh:
        fh.write("# Novel signals — results (selection on SEARCH columns only)\n\n"
                 f"Common period {common_dates[0]} -> {common_dates[-1]}, "
                 "construction locked (top25/cap5/cap_match, no tilt), 10 bps/side. "
                 "Bars per PREREGISTRATION.md; holdout column not consulted here.\n\n")
        cols = ["config", "net_cagr", "sharpe", "sortino", "max_dd", "alpha_search",
                "alpha_t", "ex_spy_e1", "ex_spy_e2", "turnover", "avg_names",
                "bar1_search_alpha_t", "bar3_sharpe_ge_ref"]
        fh.write(df[cols].to_markdown(index=False, floatfmt=".3f"))
        fh.write("\n\nNote: alpha_t is full-period (engine limitation); the binding "
                 "bar-1 t-stat on the search window alone is recomputed in Phase D "
                 "for any row that passes here.\n")
    print(f"\nwrote {OUT}/results.csv + RESULTS.md", flush=True)


if __name__ == "__main__":
    main()
