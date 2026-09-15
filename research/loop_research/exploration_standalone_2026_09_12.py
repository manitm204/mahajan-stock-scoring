"""Standalone top-20-pool strategies (user request 2026-09-12): built
directly from the base top-20-by-composite-rank pool -- NOT an add-on to
the insider_revisions_min_top20 champion, no insider/revisions signal
involved at all -- then compared against that champion as the baseline to
beat. Same one-idea-at-a-time discipline as the prior batch.

Usage: python -m research.loop_research.exploration_standalone_2026_09_12 <idea_key>
Writes: output/loop_research/promotion_<name>.json (full evaluate_promotion
report) and output/loop_research/experiments/<name>.json (compact summary).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from research.loop_research.exploration_batch_2026_09_11 import (   # noqa: E402
    make_selector, make_perturbed_selector,
)
from research.loop_research.exploration_batch_100 import (          # noqa: E402
    metric_insider_revisions_min,
)
from research.loop_research.promotion import (                      # noqa: E402
    evaluate_promotion, build_summary, print_report, OUT_DIR,
)

CHAMPION_SELECTOR = make_selector(
    "insider_revisions_min_top20", metric_insider_revisions_min, ascending=False)

VAL_REVERSION_LOOKBACK_MONTHS = 60  # ~5 years


# ---------------------------------------------------------------------------
# Idea 1: valuation mean-reversion, built purely from the top-20 pool's own
# price/valuation data -- no insider or revisions signal involved.
# `val_earnings_yield` is the closest available proxy to "1/PE" in the
# panel (a sector-percentile earnings-yield score, not a raw PE ratio, so
# this is a percentile-rank version of the user's "trailing 5yr PE minus
# current PE" idea -- buy names whose valuation percentile has cheapened
# the most vs. their own trailing-5-year average, i.e. the biggest POSITIVE
# gap between current earnings-yield percentile and its own 5-year average
# (current cheaper than its own history = biggest gap).
def metric_valuation_reversion(scores, parent_scores, pool, ctx):
    frame_by_date = ctx["bundle"]["subfactor_frames"]
    i, dates = ctx["i"], ctx["dates"]
    j = max(i - VAL_REVERSION_LOOKBACK_MONTHS + 1, 0)
    if j == i:
        return None
    cur_frame = frame_by_date.get(ctx["date"])
    if cur_frame is None or "val_earnings_yield" not in cur_frame.columns:
        return None
    cur = cur_frame["val_earnings_yield"].reindex(pool)

    hist_frames = []
    for dd in dates[j:i]:  # history strictly BEFORE the current date
        f = frame_by_date.get(dd)
        if f is not None and "val_earnings_yield" in f.columns:
            hist_frames.append(f["val_earnings_yield"].reindex(pool))
    if len(hist_frames) < 12:  # need at least a year of history to trust an average
        return None
    hist_avg = pd.concat(hist_frames, axis=1).mean(axis=1)

    return cur - hist_avg


IDEAS = {
    "valuation_reversion": (
        "valuation_reversion_top20",
        "standalone, base-pool strategy (no insider/revisions): within the "
        "top-20-by-composite pool, buy whichever names' earnings-yield "
        "percentile has risen the most above its own trailing 5-year "
        "average (i.e. currently cheap relative to their own valuation "
        "history) -- a percentile-based proxy for the user's 'trailing "
        "5yr PE minus current PE' mean-reversion idea",
        metric_valuation_reversion, False,
    ),
}


def run_experiment(key: str, n_sims: int = 500):
    name, hypothesis, metric_fn, ascending = IDEAS[key]
    candidate_selector = make_selector(name, metric_fn, ascending)

    def make_pert(noise_mult):
        return make_perturbed_selector(metric_fn, ascending, noise_mult)

    print(f"=== {name} ===\nhypothesis: {hypothesis}\n"
          f"baseline: insider_revisions_min_top20 (current champion)\n")
    report = evaluate_promotion(
        candidate_selector, baseline_selector=CHAMPION_SELECTOR,
        make_perturbed_selector=make_pert, n_sims=n_sims,
        deterministic_candidate=True,
    )
    print_report(report)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"promotion_{name}.json"
    with out_path.open("w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(f"\nwrote {out_path}")

    summary = build_summary(report, name, hypothesis)
    summary["baseline"] = "insider_revisions_min_top20"
    exp_dir = OUT_DIR / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    summary_path = exp_dir / f"{name}.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"wrote {summary_path}")
    return report, summary


if __name__ == "__main__":
    key = sys.argv[1]
    run_experiment(key)
