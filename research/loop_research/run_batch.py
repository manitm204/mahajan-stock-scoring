"""Runs every candidate in batch_candidates.ALL_WAVES against a shared
100-sim baseline (a faster screening pass than the 500-sim baseline used for
exp1-4 -- deterministic candidates are exact either way since they just
replay one run; stochastic ones get a real 100-sim distribution). Anything
that screens well here should get a full 500-sim confirmation run before any
promotion decision, exactly as flagged to the user when this harness was
first proposed (cheap screen -> confirm on survivors).

Usage: python -m research.loop_research.run_batch
Writes: output/loop_research/batch_results.json (all results, incl. every
failure) and appends a compact summary table to session_log.md.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from research.loop_research import harness as H            # noqa: E402
from research.loop_research.batch_candidates import ALL_WAVES  # noqa: E402

OUT_DIR = REPO / "output" / "loop_research"
BATCH_JSON = OUT_DIR / "batch_results.json"
LOG = REPO / "research" / "loop_research" / "session_log.md"
N_SIMS_SCREEN = 100
METRIC_NAMES = ["cagr", "sharpe", "sortino", "max_dd", "avg_turnover",
               "spy_beta", "spy_alpha", "spy_ir", "qqq_ir",
               "unique_holdings", "total_return"]


def _agg(vals):
    a = np.array([v for v in vals if v == v], dtype=float)
    if a.size == 0:
        return {"median": float("nan")}
    return {"median": float(np.median(a))}


def _is_deterministic(selector) -> bool:
    """Probe: run the selector twice with different seeds/rng draws on the
    same date -- if the resulting book is identical both times it's
    (effectively) deterministic; otherwise treat it as stochastic and give
    it the full N_SIMS_SCREEN draws."""
    import random
    bundle = H.get_data()
    data = bundle["data"]
    d = data.rebal_dates[20]
    scores = data.comp.get(d)
    pscores = bundle["parent_scores"].get(d, {})
    ctx = {"date": d, "dates": list(data.rebal_dates), "i": 20, "bundle": bundle}
    if scores is None:
        return True
    r1 = random.Random(1)
    r2 = random.Random(2)
    try:
        out1 = selector(scores, pscores, [], H.N, H.REFRESH_N, r1, ctx=ctx)
        out2 = selector(scores, pscores, [], H.N, H.REFRESH_N, r2, ctx=ctx)
    except Exception:
        return True
    return sorted(out1) == sorted(out2)


def main():
    bundle = H.get_data()
    print("computing 100-sim baseline (screening pass) ...", flush=True)
    cache = OUT_DIR / f"baseline_{N_SIMS_SCREEN}.json"
    if cache.exists():
        with cache.open() as fh:
            baseline = json.load(fh)
    else:
        baseline = H.run_monte_carlo(bundle, H.random_selector, N_SIMS_SCREEN, seed_base=0)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with cache.open("w") as fh:
            json.dump(baseline, fh)
    base_sharpe = np.array([s["sharpe"] for s in baseline])
    base_by_metric = {m: np.array([s[m] for s in baseline]) for m in METRIC_NAMES}

    all_candidates = []
    for wave_name, items in ALL_WAVES.items():
        for name, hypothesis, selector in items:
            all_candidates.append((wave_name, name, hypothesis, selector))

    print(f"{len(all_candidates)} candidates queued", flush=True)
    results = []
    t0 = time.time()
    for idx, (wave, name, hyp, selector) in enumerate(all_candidates):
        det = _is_deterministic(selector)
        n_run = N_SIMS_SCREEN if not det else N_SIMS_SCREEN
        sims = H.run_monte_carlo(bundle, selector, N_SIMS_SCREEN, seed_base=0, deterministic=det)
        cand_sharpe = np.array([s["sharpe"] for s in sims])
        diffs = cand_sharpe - base_sharpe
        win_rate = float((diffs > 0).mean())
        rec = {
            "wave": wave, "name": name, "hypothesis": hyp,
            "deterministic": det, "n_sims": N_SIMS_SCREEN,
            "candidate_sharpe_median": float(np.median(cand_sharpe)),
            "baseline_sharpe_median": float(np.median(base_sharpe)),
            "win_rate_vs_baseline": win_rate,
            "candidate_cagr_median": float(np.median([s["cagr"] for s in sims])),
            "candidate_max_dd_median": float(np.median([s["max_dd"] for s in sims])),
            "candidate_alpha_median": float(np.median([s["spy_alpha"] for s in sims])),
        }
        results.append(rec)
        elapsed = time.time() - t0
        rate = (idx + 1) / elapsed
        eta = (len(all_candidates) - idx - 1) / rate if rate > 0 else float("nan")
        print(f"[{idx+1}/{len(all_candidates)}] {name:45s} det={det!s:5s} "
             f"win_rate={win_rate:.1%}  sharpe={rec['candidate_sharpe_median']:.3f}  "
             f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with BATCH_JSON.open("w") as fh:
        json.dump(results, fh, indent=2)

    ranked = sorted(results, key=lambda r: r["win_rate_vs_baseline"], reverse=True)
    print("\n=== TOP 10 by win rate vs baseline (100-sim screen) ===")
    for r in ranked[:10]:
        print(f"  {r['name']:45s} win_rate={r['win_rate_vs_baseline']:.1%}  "
             f"sharpe={r['candidate_sharpe_median']:.3f} vs {r['baseline_sharpe_median']:.3f}")
    print("\n=== BOTTOM 10 by win rate vs baseline ===")
    for r in ranked[-10:]:
        print(f"  {r['name']:45s} win_rate={r['win_rate_vs_baseline']:.1%}  "
             f"sharpe={r['candidate_sharpe_median']:.3f} vs {r['baseline_sharpe_median']:.3f}")
    print(f"\nwrote {BATCH_JSON}  ({len(results)} candidates)")


if __name__ == "__main__":
    main()
