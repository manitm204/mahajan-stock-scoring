"""Drives one experiment: run a candidate selector's Monte Carlo distribution
paired seed-for-seed against the champion (random baseline), compare, and
append a structured record to the session log / results JSON. See
harness.py for the immutable mechanics; this file only orchestrates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research.loop_research import harness as H

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "output" / "loop_research"
LOG = REPO / "research" / "loop_research" / "session_log.md"
RESULTS_JSON = OUT_DIR / "results.json"

N_SIMS = 500
METRIC_NAMES = ["cagr", "sharpe", "sortino", "max_dd", "avg_turnover",
               "spy_beta", "spy_alpha", "spy_ir", "qqq_ir",
               "unique_holdings", "total_return"]


def _agg(vals):
    a = np.array([v for v in vals if v == v], dtype=float)
    if a.size == 0:
        return {"median": float("nan"), "mean": float("nan"), "p25": float("nan"), "p75": float("nan")}
    return {"median": float(np.median(a)), "mean": float(a.mean()),
           "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75))}


def get_baseline(bundle, n_sims=N_SIMS, seed_base=0):
    cache = OUT_DIR / f"baseline_{n_sims}.json"
    if cache.exists():
        with cache.open() as fh:
            return json.load(fh)
    sims = H.run_monte_carlo(bundle, H.random_selector, n_sims, seed_base=seed_base)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with cache.open("w") as fh:
        json.dump(sims, fh)
    return sims


def run_experiment(name, hypothesis, selector, deterministic, bundle=None,
                   n_sims=N_SIMS, seed_base=0, params=None, decision_primary="sharpe"):
    bundle = bundle or H.get_data()
    baseline = get_baseline(bundle, n_sims, seed_base)
    candidate = H.run_monte_carlo(bundle, selector, n_sims, seed_base=seed_base,
                                  deterministic=deterministic)

    base_by_metric = {m: [s[m] for s in baseline] for m in METRIC_NAMES}
    cand_by_metric = {m: [s[m] for s in candidate] for m in METRIC_NAMES}

    diffs = np.array(cand_by_metric[decision_primary]) - np.array(base_by_metric[decision_primary])
    diffs = diffs[diffs == diffs]
    win_rate = float((diffs > 0).mean()) if diffs.size else float("nan")
    ci = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))) if diffs.size else (float("nan"),) * 2

    record = {
        "name": name,
        "hypothesis": hypothesis,
        "params": params or {},
        "deterministic": deterministic,
        "n_sims": n_sims,
        "decision_primary": decision_primary,
        "baseline": {m: _agg(base_by_metric[m]) for m in METRIC_NAMES},
        "candidate": {m: _agg(cand_by_metric[m]) for m in METRIC_NAMES},
        "paired_diff_primary": {"median": float(np.median(diffs)) if diffs.size else float("nan"),
                                "ci95": ci, "win_rate_vs_baseline": win_rate},
    }
    _append_result(record)
    _append_log(record)
    return record


def _append_result(record):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = []
    if RESULTS_JSON.exists():
        with RESULTS_JSON.open() as fh:
            existing = json.load(fh)
    existing.append(record)
    with RESULTS_JSON.open("w") as fh:
        json.dump(existing, fh, indent=2)


def _append_log(r):
    b, c, d = r["baseline"], r["candidate"], r["paired_diff_primary"]
    lines = [
        f"\n## {r['name']}",
        f"**Hypothesis:** {r['hypothesis']}",
        f"**Params:** {r['params']}  |  deterministic={r['deterministic']}  |  n_sims={r['n_sims']}",
        "",
        "| metric | baseline median | candidate median | paired diff (median) |",
        "|---|---|---|---|",
    ]
    for m in METRIC_NAMES:
        lines.append(f"| {m} | {b[m]['median']:.4f} | {c[m]['median']:.4f} | "
                     f"{c[m]['median'] - b[m]['median']:+.4f} |")
    lines += [
        "",
        f"**{r['decision_primary']} win rate vs baseline:** {d['win_rate_vs_baseline']:.1%}  "
        f"(95% CI on paired diff: [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}])",
        "",
    ]
    with LOG.open("a") as fh:
        fh.write("\n".join(lines) + "\n")
