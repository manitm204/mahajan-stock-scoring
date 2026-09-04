"""Stage A/B/C config grids and the progress-printing runner.

Stage A (OFAT): vary one family at a time around the neutral reference
(top25%, EW, no exclusion, no sector constraint, no VIX, monthly). Anchors:
whole-universe EW and SPY. Stage B: cross the best levels of the interacting
families (breadth x weighting x sector) with VIX on/off. Stage C: frequency,
hysteresis and cost sensitivity on the finalist.

Winner rule (pre-registered): positive net alpha vs SPY full-period AND
non-negative excess vs SPY in both eras (split 2021-12-31).
"""
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .engine import AblationConfig, simulate_config, spy_row

REFERENCE = AblationConfig(name="REF top25_ew", top_pct=0.25)

_COLS = ["family", "config", "top_pct", "exclusion", "weighting", "sector", "vix",
         "hold_m", "exit_pct", "net_cagr", "gross_cagr", "sharpe", "sortino",
         "max_dd", "calmar", "beta", "alpha", "alpha_t", "ir", "ex_spy",
         "ex_spy_e1", "ex_spy_e2", "alpha_search", "alpha_holdout",
         "turnover", "avg_names", "eff_n"]

BREADTHS = [0.10, 0.25, 0.50, 0.75, 1.00]
EXCLUSIONS = ["none", "any_p10", "two_p10", "any_p5", "soft_p10"]
WEIGHTINGS = ["ew", "cap", "cap5", "ewcap", "rank_lin", "rank_sqrt", "score_vol",
              "inv_vol", "erc", "softmax"]
SECTORS = ["none", "neutral", "equal", "cap_match", "bands5"]


# --------------------------------------------------------------------------- #
# Grids
# --------------------------------------------------------------------------- #
def stage_a_configs() -> list[tuple[str, AblationConfig]]:
    cfgs: list[tuple[str, AblationConfig]] = [
        ("anchor", replace(REFERENCE, name="top100_ew (universe)", top_pct=1.00)),
        ("reference", REFERENCE),
    ]
    for pct in (0.10, 0.50, 0.75):
        cfgs.append(("breadth", replace(REFERENCE, name=f"top{int(pct*100)}_ew",
                                        top_pct=pct)))
    for ex in ("any_p10", "two_p10", "any_p5", "soft_p10"):
        cfgs.append(("exclusion", replace(REFERENCE, name=f"top25_ew excl:{ex}",
                                          exclusion=ex)))
    for wt in ("cap", "cap5", "ewcap", "rank_lin", "rank_sqrt", "score_vol",
               "inv_vol", "erc", "softmax"):
        cfgs.append(("weighting", replace(REFERENCE, name=f"top25_{wt}",
                                          weighting=wt)))
    for sc in ("neutral", "equal", "cap_match", "bands5"):
        cfgs.append(("sector", replace(REFERENCE, name=f"top25_ew sec:{sc}",
                                       sector=sc)))
    cfgs.append(("vix", replace(REFERENCE, name="top25_ew vixtilt", vix_tilt=True)))
    return cfgs


def _top_levels(df: pd.DataFrame, family: str, col, k: int) -> list:
    """Best-k levels of one family by net alpha; the reference row competes as the
    family's default level."""
    pool = df[df["family"].isin([family, "reference", "anchor"])]
    ranked = pool.sort_values("alpha", ascending=False)
    seen: list = []
    for _, r in ranked.iterrows():
        v = col(r)
        if v not in seen:
            seen.append(v)
        if len(seen) == k:
            break
    return seen


def full_grid_configs() -> list[tuple[str, AblationConfig]]:
    """Every construction combination: breadth x exclusion x weighting x sector x VIX
    = 5*5*10*5*2 = 2,500 configs."""
    cfgs = []
    for pct in BREADTHS:
        for ex in EXCLUSIONS:
            for wt in WEIGHTINGS:
                for sc in SECTORS:
                    for vx in (False, True):
                        name = (f"top{int(pct*100)}_{wt}_{ex}_{sc}"
                                f"{'_vix' if vx else ''}")
                        cfgs.append(("grid", AblationConfig(
                            name=name, top_pct=pct, exclusion=ex, weighting=wt,
                            sector=sc, vix_tilt=vx)))
    return cfgs


def stage_b_configs(stage_a: pd.DataFrame) -> list[tuple[str, AblationConfig]]:
    breadths = _top_levels(stage_a, "breadth", lambda r: float(r["top_pct"]), 2)
    weights = _top_levels(stage_a, "weighting", lambda r: r["weighting"] or "ew", 3)
    sectors = _top_levels(stage_a, "sector", lambda r: r["sector"] or "none", 2)
    print(f"  [stage B] crossing breadth={breadths} x weighting={weights} "
          f"x sector={sectors} x vix on/off", flush=True)
    cfgs = []
    for pct in breadths:
        for wt in weights:
            for sc in sectors:
                for vx in (False, True):
                    name = (f"top{int(pct*100)}_{wt}_sec:{sc}"
                            f"{'_vix' if vx else ''}")
                    cfgs.append(("cross", AblationConfig(
                        name=name, top_pct=pct, weighting=wt, sector=sc,
                        vix_tilt=vx)))
    return cfgs


def pick_winner(df: pd.DataFrame) -> pd.Series:
    """Apply the pre-registered rule; fall back (loudly) to best alpha if nothing
    qualifies."""
    ok = df[(df["config"] != "SPY") & (df["alpha"] > 0)
            & (df["ex_spy_e1"] >= 0) & (df["ex_spy_e2"] >= 0)]
    if len(ok):
        return ok.sort_values("alpha", ascending=False).iloc[0]
    print("  [!] NO config passes the pre-registered rule "
          "(alpha>0 AND both eras >=0) — falling back to best alpha, "
          "which does NOT count as a win.", flush=True)
    cand = df[df["config"] != "SPY"]
    return cand.sort_values("alpha", ascending=False).iloc[0]


def stage_c_configs(row: pd.Series) -> list[tuple[str, AblationConfig]]:
    base = AblationConfig(
        name="finalist", top_pct=float(row["top_pct"]),
        exclusion=row["exclusion"] or "none", weighting=row["weighting"] or "ew",
        sector=row["sector"] or "none", vix_tilt=bool(row["vix"]))
    cfgs = []
    for hm in (1, 3, 6):
        for exit_pct in (None, min(1.0, base.top_pct + 0.10)):
            tag = f"hold{hm}m" + (f"_exit{int(exit_pct*100)}" if exit_pct else "")
            cfgs.append(("final", replace(base, name=f"finalist {tag}",
                                          hold_months=hm, exit_pct=exit_pct)))
    for bps in (0.0, 20.0):
        cfgs.append(("cost", replace(base, name=f"finalist cost{int(bps)}bps",
                                     cost_bps=bps)))
    return cfgs


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def _fmt_row(r: dict) -> str:
    return (f"net {r['net_cagr']*100:+6.2f}%  Shp {r['sharpe']:5.2f}  "
            f"a {r['alpha']*100:+5.2f}% (t {r['alpha_t']:+4.1f})  "
            f"exSPY {r['ex_spy']*100:+5.2f}%  "
            f"[e1 {r['ex_spy_e1']*100:+5.2f} | e2 {r['ex_spy_e2']*100:+5.2f}]  "
            f"n {r['avg_names']:.0f}/effN {r['eff_n']:.0f}  to {r['turnover']:.2f}")


def run_stage(data, configs: list, csv_path: str | Path, label: str,
              progress_every: int = 1) -> pd.DataFrame:
    """Run every config, writing the CSV incrementally (safe to tail / safe to kill).

    ``progress_every`` throttles both console prints and CSV checkpoints for large
    grids: a per-config line is printed every ``progress_every`` configs (plus a
    running best-by-search-alpha marker), and the CSV is flushed at the same cadence.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    t_start = time.time()
    best = None
    print(f"\n=== Stage {label}: {len(configs)} configs "
          f"(progress every {progress_every}) ===", flush=True)
    for i, (family, cfg) in enumerate(configs, 1):
        t0 = time.time()
        row = simulate_config(data, cfg)
        row["family"] = family
        rows.append(row)
        a_s = row.get("alpha_search")
        if a_s == a_s and (best is None or a_s > best[0]):
            best = (a_s, row["config"])
        if i % progress_every == 0 or i == len(configs):
            pd.DataFrame(rows).reindex(columns=_COLS).to_csv(csv_path, index=False)
            eta = (time.time() - t_start) / i * (len(configs) - i)
            print(f"[{i:4d}/{len(configs)}] {cfg.name:34s} {_fmt_row(row)} "
                  f"| best_search {best[1]} a={best[0]*100:+.2f}% "
                  f"| ETA {eta/60:4.1f}m", flush=True)
    srow = spy_row(data)
    srow["family"] = "benchmark"
    rows.append(srow)
    df = pd.DataFrame(rows).reindex(columns=_COLS)
    df.to_csv(csv_path, index=False)

    print(f"\n--- Stage {label} done in {(time.time()-t_start)/60:.1f} min — "
          f"top 8 by net alpha (SPY last) ---", flush=True)
    show = df[df["config"] != "SPY"].sort_values("alpha", ascending=False).head(8)
    for _, r in pd.concat([show, df[df["config"] == "SPY"]]).iterrows():
        print(f"  {r['config']:32s} {_fmt_row(r)}", flush=True)
    print(f"\nsaved -> {csv_path}", flush=True)
    return df


def report_full_grid(df: pd.DataFrame) -> pd.Series | None:
    """Apply the pre-registered strict-OOS promotion rule to the full grid and print
    the verdict. Returns the finalist row (highest holdout alpha among promoted) or
    None if nothing is promoted."""
    g = df[df["config"] != "SPY"].copy()
    n = len(g)
    beat_full = (g["alpha"] > 0).mean()
    beat_holdout = (g["alpha_holdout"] > 0).mean()
    promoted = g[(g["alpha_search"] > 0) & (g["alpha_holdout"] > 0)
                 & (g["ex_spy_e1"] >= 0) & (g["ex_spy_e2"] >= 0)]

    print("\n" + "=" * 78)
    print("FULL-GRID VERDICT (pre-registered strict-OOS rule)")
    print("=" * 78)
    print(f"  configs evaluated              : {n}")
    print(f"  beat SPY full-period (alpha>0) : {beat_full:6.1%}  "
          f"({int(beat_full*n)} configs)")
    print(f"  beat SPY in HOLDOUT (2024-26)  : {beat_holdout:6.1%}  "
          f"(untouched during ranking)")
    print(f"  PROMOTED (all 4 conditions)    : {len(promoted)}  "
          f"[search a>0 AND holdout a>0 AND both eras >=0]", flush=True)

    if promoted.empty:
        print("\n  >>> NO config passes. The construction space does not contain a "
              "robust\n      SPY-beating book on this universe. Honest next step: "
              "universe expansion.")
        # Show what the best in-sample config gives up out-of-sample (the overfit gap).
        top = g.sort_values("alpha_search", ascending=False).head(10)
        print("\n  overfit gap — top 10 by SEARCH alpha, then their HOLDOUT alpha:")
        for _, r in top.iterrows():
            print(f"    {r['config']:34s} search a {r['alpha_search']*100:+5.2f}%  "
                  f"-> holdout a {r['alpha_holdout']*100:+5.2f}%  "
                  f"(full {r['alpha']*100:+5.2f}%, t {r['alpha_t']:+.1f})", flush=True)
        return None

    fin = promoted.sort_values("alpha_holdout", ascending=False).iloc[0]
    print(f"\n  FINALIST (highest holdout alpha): {fin['config']}")
    print("\n  promoted configs (ranked by holdout alpha):")
    for _, r in promoted.sort_values("alpha_holdout", ascending=False).head(15).iterrows():
        print(f"    {r['config']:34s} search a {r['alpha_search']*100:+5.2f}%  "
              f"holdout a {r['alpha_holdout']*100:+5.2f}%  full {r['alpha']*100:+5.2f}% "
              f"(t {r['alpha_t']:+.1f})  Shp {r['sharpe']:.2f}", flush=True)
    return fin
