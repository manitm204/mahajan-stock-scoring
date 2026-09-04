"""Evaluate the 8 selected parent factors like sub-factors.

Reads the expanded-library CandidatePanel + the parent-selection weights, composites
each parent from its selected subs, and runs the standard sub-factor scorecard on
the resulting parent panel: IC (3M/6M pooled), IC-IR, Q5-Q1 spread, hit rate,
monotonicity, coverage. Also computes the parent-vs-parent correlation matrix and
evaluates an equal-weight 8-parent composite. Read-only — nothing written to the
live DB or production factor set.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=FutureWarning,
                        module="research.subfactor_expansion")

from backtesting import data_loader as dl
from data.db import get_db
from research import HORIZON_MONTHS, compute_forward_returns
from research.panel import ScorePanel
from research.subfactor_expansion.panel import (cache_key as exp_cache_key,
                                                load_cached_panel as exp_load)
from research.subset_selection import (correlation_matrix, inventory,
                                       subfactor_performance)

EXP_START = "2023-01-01"
EXP_END = "2026-06-30"
CACHE = Path("cache/subfactor_expansion") / exp_cache_key(EXP_START, EXP_END, "monthly")
SELECTIONS = Path("output/parent_selection_expansion/selections.csv")
OUT_DIR = Path("output/parent_eval")

STAT_HORIZONS = ("3M", "6M")
PARENT_ORDER = ["momentum", "value", "quality", "growth", "revisions",
                "institutional", "insider", "short"]


def _adapt(cand) -> ScorePanel:
    parent_keys = [p for p, subs in cand.candidates_by_parent.items() if subs]
    return ScorePanel(
        rebal_dates=list(cand.rebal_dates), scores=cand.scores,
        parent_keys=parent_keys,
        sub_by_parent={p: list(cand.candidates_by_parent[p]) for p in parent_keys},
        universe=list(cand.universe),
    )


def _load_weights() -> dict[str, dict[str, float]]:
    """Parse selections.csv 'weights' column → {parent: {sub: weight, ...}}."""
    sel = pd.read_csv(SELECTIONS)
    out: dict[str, dict[str, float]] = {}
    for _, r in sel.iterrows():
        wmap: dict[str, float] = {}
        for tok in str(r["weights"]).split(","):
            tok = tok.strip()
            if "=" not in tok:
                continue
            name, w = tok.split("=", 1)
            wmap[name.strip()] = float(w.strip())
        # Renormalize (safety — CSV weights already sum to 1.0 but be strict).
        s = sum(wmap.values())
        if s > 0:
            wmap = {k: v / s for k, v in wmap.items()}
        out[r["parent"]] = wmap
    return out


def _composite_row(mat: np.ndarray, w: np.ndarray,
                   coverage_aware: bool = True) -> np.ndarray:
    """Per-row weighted mean of ``mat`` with weights ``w``.

    ``coverage_aware=True`` skips NaN cells and renormalizes across the *available*
    columns per row. ``coverage_aware=False`` requires every column present — any
    NaN in the row → NaN result (the "strict" all-parents-must-be-present variant).
    """
    if not coverage_aware:
        any_missing = np.isnan(mat).any(axis=1)
        with np.errstate(invalid="ignore"):
            wsum = w.sum()
            comp = np.where(wsum > 0, np.nansum(mat * w[None, :], axis=1) / wsum, np.nan)
        return np.where(any_missing, np.nan, comp)
    mask = ~np.isnan(mat)
    wrow = mask * w[None, :]
    wsum = wrow.sum(axis=1)
    arr0 = np.where(mask, mat, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        comp = (arr0 * wrow).sum(axis=1) / wsum
    return np.where(wsum > 0, comp, np.nan)


def _build_parent_panel(sub_panel: ScorePanel,
                        weights_by_parent: dict[str, dict[str, float]]) -> ScorePanel:
    """One 'sub' per parent = weighted composite of its selected sub-factors."""
    parents = [p for p in PARENT_ORDER if p in weights_by_parent]
    scores: dict[str, pd.DataFrame] = {}
    for d in sub_panel.rebal_dates:
        frame = sub_panel.scores.get(d)
        if frame is None:
            continue
        cols: dict[str, pd.Series] = {}
        for parent in parents:
            wmap = weights_by_parent[parent]
            present = [s for s in wmap if s in frame.columns]
            if not present:
                cols[parent] = pd.Series(np.nan, index=frame.index)
                continue
            sub_mat = frame[present].to_numpy(dtype=float)
            w = np.array([wmap[s] for s in present], dtype=float)
            cols[parent] = pd.Series(_composite_row(sub_mat, w), index=frame.index)
        scores[d] = pd.DataFrame(cols).reindex(sub_panel.universe)
    return ScorePanel(
        rebal_dates=list(scores.keys()), scores=scores,
        parent_keys=parents,
        sub_by_parent={p: [p] for p in parents},
        universe=list(sub_panel.universe),
    )


def _attach_composite(panel: ScorePanel, name: str, parents: list[str],
                      weights: np.ndarray, coverage_aware: bool) -> None:
    """Add a named composite column to every rebalance frame."""
    for d in panel.rebal_dates:
        frame = panel.scores[d]
        mat = frame[parents].to_numpy(dtype=float)
        frame[name] = _composite_row(mat, weights, coverage_aware=coverage_aware)
    if name not in panel.parent_keys:
        panel.parent_keys.append(name)
    panel.sub_by_parent[name] = [name]


def _ic_ir_weights(parent_score: pd.DataFrame, cap: float = 0.25) -> dict[str, float]:
    """Build an IC+IR-weighted, capped, renormalized parent-weight vector.

    Combined score = 0.5 · (mean_ic / max_ic⁺) + 0.5 · (IR / max_ir⁺), each floored at 0
    (so a parent with a non-positive IC or IR gets zero share from that half). Weights
    ∝ combined score, water-filled to ``cap`` per parent, renormalized to sum to 1.
    Any parent with combined score ≤ 0 receives 0 weight — dilutive parents are
    excluded, not floored to a token share."""
    ic = parent_score.set_index("parent")["mean_ic_3m6m"].clip(lower=0.0)
    ir = parent_score.set_index("parent")["information_ratio"].clip(lower=0.0)
    ic_norm = ic / ic.max() if ic.max() > 0 else ic * 0.0
    ir_norm = ir / ir.max() if ir.max() > 0 else ir * 0.0
    combined = 0.5 * ic_norm + 0.5 * ir_norm
    raw = combined / combined.sum()
    w = raw.to_dict()
    # Water-fill cap: any parent > cap → cap, redistribute excess proportionally to under-cap.
    for _ in range(20):
        over = [p for p, x in w.items() if x > cap + 1e-12]
        if not over:
            break
        excess = sum(w[p] - cap for p in over)
        for p in over:
            w[p] = cap
        under = [p for p, x in w.items() if x < cap - 1e-12 and x > 0]
        pool = sum(w[p] for p in under)
        if not under or pool <= 1e-12:
            break
        for p in under:
            w[p] += excess * w[p] / pool
    tot = sum(w.values())
    return {p: w[p] / tot for p in w}


def _fwd() -> dict[str, dict[str, pd.Series]]:
    with get_db() as db:
        matrix = dl.load_price_matrix(db, db.universe_tickers(), EXP_START, EXP_END)
    return matrix


def _fmt_row(row: pd.Series, cols: list[str]) -> str:
    def _one(v):
        if pd.isna(v):
            return "—"
        return f"{v:+.4f}" if isinstance(v, (int, float, np.floating)) else str(v)
    return "| " + " | ".join(_one(row[c]) for c in cols) + " |"


def _md_table(df: pd.DataFrame, cols: list[str], label_col: str,
              fmt: dict[str, str] | None = None) -> str:
    fmt = fmt or {}
    header = "| " + " | ".join([label_col] + cols) + " |"
    sep = "| " + " | ".join(["---"] * (len(cols) + 1)) + " |"
    lines = [header, sep]
    for _, r in df.iterrows():
        cells = [str(r[label_col])]
        for c in cols:
            v = r[c]
            if pd.isna(v):
                cells.append("—")
                continue
            spec = fmt.get(c, "+.4f")
            cells.append(f"{v:{spec}}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    if not CACHE.exists():
        raise SystemExit(f"missing candidate panel cache: {CACHE}\n"
                         "run: python run_subfactor_expansion.py --build")
    if not SELECTIONS.exists():
        raise SystemExit(f"missing selections: {SELECTIONS}\n"
                         "run: python run_parent_selection.py --source expansion")

    print(f"loading candidate panel: {CACHE}")
    cand = exp_load(CACHE)
    sub_panel = _adapt(cand)
    weights = _load_weights()
    print(f"parents to evaluate: {list(weights)}")

    parent_panel = _build_parent_panel(sub_panel, weights)

    print("computing forward returns…")
    price_matrix = _fwd()
    fwd_by_h = compute_forward_returns(price_matrix, parent_panel.rebal_dates,
                                       HORIZON_MONTHS)

    parents_all = [p for p in parent_panel.parent_keys]  # snapshot: 8 parents only

    # --- Per-parent scorecard (unchanged from previous run) ---------------
    print("running per-parent scorecard (stat_horizons=3M/6M)…")
    perf = subfactor_performance(parent_panel, fwd_by_h, stat_horizons=STAT_HORIZONS)
    inv = inventory(parent_panel)
    perf["mean_ic_3m6m"] = perf[["ic_3M", "ic_6M"]].mean(axis=1)
    scored = (perf.drop(columns=["parent"], errors="ignore")
                  .rename(columns={"sub_factor": "parent"})
                  .merge(inv[["sub_factor", "coverage"]].rename(
                      columns={"sub_factor": "parent"}), on="parent"))
    keep = ["parent", "mean_ic_3m6m", "information_ratio", "spread_q5_q1",
            "hit_rate", "monotonicity", "coverage", "ic_1M", "ic_3M", "ic_6M",
            "ic_12M", "n_periods"]
    scored = scored[keep].copy()
    parent_score = scored[scored["parent"].isin(parents_all)].reset_index(drop=True)
    parent_score["parent"] = pd.Categorical(parent_score["parent"],
                                            categories=PARENT_ORDER, ordered=True)
    parent_score = parent_score.sort_values("parent").reset_index(drop=True)

    # --- Composite variants ----------------------------------------------
    def _ew(subset: list[str]) -> np.ndarray:
        return np.ones(len(subset), dtype=float)

    variants: list[tuple[str, list[str], np.ndarray, bool]] = []
    variants.append(("V1_EW_all_strict",
                     parents_all, _ew(parents_all), False))
    variants.append(("V2_EW_no_quality",
                     [p for p in parents_all if p != "quality"],
                     _ew([p for p in parents_all if p != "quality"]), False))
    v3_subset = [p for p in parents_all if p not in ("quality", "insider")]
    variants.append(("V3_EW_no_quality_insider",
                     v3_subset, _ew(v3_subset), False))
    icir_weights = _ic_ir_weights(parent_score, cap=0.25)
    v4_subset = [p for p in parents_all if icir_weights.get(p, 0.0) > 0]
    v4_w = np.array([icir_weights[p] for p in v4_subset], dtype=float)
    variants.append(("V4_ICIR_weighted_cap25", v4_subset, v4_w, False))
    variants.append(("V5_EW_coverage_aware",
                     parents_all, _ew(parents_all), True))

    print(f"attaching {len(variants)} composite columns…")
    for name, subset, w, cov in variants:
        _attach_composite(parent_panel, name, subset, w, coverage_aware=cov)

    print("running composite scorecard…")
    perf_all = subfactor_performance(parent_panel, fwd_by_h, stat_horizons=STAT_HORIZONS)
    inv_all = inventory(parent_panel)
    perf_all["mean_ic_3m6m"] = perf_all[["ic_3M", "ic_6M"]].mean(axis=1)
    scored_all = (perf_all.drop(columns=["parent"], errors="ignore")
                          .rename(columns={"sub_factor": "parent"})
                          .merge(inv_all[["sub_factor", "coverage"]].rename(
                              columns={"sub_factor": "parent"}), on="parent"))
    comp_names = [n for n, _, _, _ in variants]
    comp_score = scored_all[scored_all["parent"].isin(comp_names)].copy()
    comp_score["parent"] = pd.Categorical(comp_score["parent"],
                                          categories=comp_names, ordered=True)
    comp_score = comp_score.sort_values("parent").reset_index(drop=True)

    # --- Δ vs baseline (V1) ----------------------------------------------
    baseline = comp_score[comp_score["parent"] == "V1_EW_all_strict"].iloc[0]
    comp_score["Δ_ic"] = comp_score["mean_ic_3m6m"] - baseline["mean_ic_3m6m"]
    comp_score["Δ_ir"] = comp_score["information_ratio"] - baseline["information_ratio"]
    comp_score["Δ_spread"] = comp_score["spread_q5_q1"] - baseline["spread_q5_q1"]
    comp_score["Δ_hit"] = comp_score["hit_rate"] - baseline["hit_rate"]
    comp_score["Δ_cov"] = comp_score["coverage"] - baseline["coverage"]

    # --- Parent × parent correlation --------------------------------------
    corr = correlation_matrix(parent_panel, subs=parents_all)

    # --- persist ----------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scored_all.to_csv(OUT_DIR / "scorecard.csv", index=False)
    corr.to_csv(OUT_DIR / "correlation.csv")

    weight_rows: list[dict] = []
    for name, subset, w, cov in variants:
        wnorm = w / w.sum()
        for p in parents_all:
            weight_rows.append({"composite": name, "parent": p,
                                "weight": float(dict(zip(subset, wnorm)).get(p, 0.0)),
                                "coverage_aware": cov})
    pd.DataFrame(weight_rows).to_csv(OUT_DIR / "composite_weights.csv", index=False)

    # --- console print ----------------------------------------------------
    show = ["mean_ic_3m6m", "information_ratio", "spread_q5_q1", "hit_rate",
            "monotonicity", "coverage"]
    print("\n" + "=" * 100)
    print(" PER-PARENT SCORECARD (3M/6M pooled)")
    print("=" * 100)
    print(parent_score[["parent"] + show].to_string(index=False,
          float_format=lambda v: f"{v:+.4f}"))
    print("\n" + "=" * 100)
    print(" COMPOSITE VARIANTS")
    print("=" * 100)
    print(comp_score[["parent"] + show + ["Δ_ic", "Δ_ir", "Δ_spread",
                                          "Δ_hit", "Δ_cov"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print("\n WEIGHTS PER VARIANT (row-normalized):")
    for name, subset, w, cov in variants:
        wnorm = w / w.sum()
        wtxt = ", ".join(f"{p}={wi:.3f}" for p, wi in zip(subset, wnorm))
        print(f"  {name:26s} [cov={cov}] → {wtxt}")

    # Rank composite variants by each metric
    print("\n RANKINGS across composites:")
    for label, col in {"IC (3M/6M)": "mean_ic_3m6m",
                       "IC IR": "information_ratio",
                       "Q5-Q1": "spread_q5_q1",
                       "Hit Rate": "hit_rate"}.items():
        r = comp_score[["parent", col]].sort_values(col, ascending=False).reset_index(drop=True)
        r.index += 1
        print(f"\n  by {label}:")
        print(r.to_string())

    print(f"\nWrote: {OUT_DIR}/scorecard.csv, correlation.csv, composite_weights.csv")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
