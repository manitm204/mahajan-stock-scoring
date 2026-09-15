"""100 more standalone top-20-pool strategies (user request 2026-09-12,
continuing from exploration_standalone_2026_09_12.py's single proof-of-
understanding run). Every idea here is built directly from the base
top-20-by-composite-rank pool's own data (parent scores, the 90-ish raw
subfactor columns, or price history) -- NOT an add-on to the champion, and
most have nothing to do with insider/revisions at all, per the user's
explicit instruction. Each is tested one at a time against the current
champion `insider_revisions_min_top20` as the baseline to beat, using the
same `evaluate_promotion` 8-gate + perturbation protocol as every prior
candidate in this project.

Performance note: since ALL 100 ideas share the exact same baseline, the
baseline's Monte Carlo paths are computed ONCE up front
(`promotion.compute_baseline_paths`) and reused via the new
`baseline_precomputed=` argument on `evaluate_promotion` (added 2026-09-12
specifically to make a batch like this practical -- previously every single
`evaluate_promotion` call redundantly re-simulated the baseline 500 times).

Crash-resilience (user-requested): every idea's full report is written to
`output/loop_research/promotion_<name>.json` and its compact summary to
`output/loop_research/experiments/<name>.json` IMMEDIATELY after that idea
finishes, and a running line is appended to
`output/loop_research/batch100_manifest.jsonl`. Re-running this script
SKIPS any idea whose experiment JSON already exists, so an interrupted run
resumes exactly where it left off rather than restarting.

Usage: python -m research.loop_research.exploration_batch_2026_09_12_100
"""
from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from research.loop_research import harness as H                      # noqa: E402
from research.loop_research.exploration_batch_2026_09_11 import (    # noqa: E402
    make_selector, make_perturbed_selector,
)
from research.loop_research.exploration_batch_100 import (           # noqa: E402
    metric_insider_revisions_min,
)
from research.loop_research.promotion import (                       # noqa: E402
    evaluate_promotion, build_summary, compute_baseline_paths, OUT_DIR,
)

CHAMPION_SELECTOR = make_selector(
    "insider_revisions_min_top20", metric_insider_revisions_min, ascending=False)

PARENTS = ["momentum", "value", "quality", "growth", "revisions",
          "institutional", "insider", "short"]
PREFIX = {"momentum": "mom_", "value": "val_", "quality": "qual_",
         "growth": "grw_", "revisions": "rev_", "institutional": "inst_",
         "insider": "ins_", "short": "si_"}

N_SIMS = 500
MANIFEST = OUT_DIR / "batch100_manifest.jsonl"
EXP_DIR = OUT_DIR / "experiments"


def _sub_frame(ctx):
    return ctx["bundle"]["subfactor_frames"].get(ctx["date"])


# --- Family A: parent-pair MIN (28) -- is the floor/AND logic that made
# insider+revisions work general across ANY pair of parents, or specific? ---
def make_parent_min(p1, p2):
    def fn(scores, parent_scores, pool, ctx):
        a, b = parent_scores.get(p1), parent_scores.get(p2)
        if a is None or b is None:
            return None
        return pd.concat([a, b], axis=1).min(axis=1).reindex(pool)
    return fn


# --- Family B: individual subfactor LEVEL rankings (32 = 8 parents x 4) ---
def make_subfactor_level(col):
    def fn(scores, parent_scores, pool, ctx):
        frame = _sub_frame(ctx)
        if frame is None or col not in frame.columns:
            return None
        return frame[col].reindex(pool)
    return fn


SUBFACTOR_LEVEL_PICKS = {
    "momentum": ["mom_52w_high_prox", "mom_vol_adjusted", "mom_consistency_60d", "mom_max_drawdown_252d"],
    "value": ["val_book_to_price", "val_fcf_yield_clean", "val_ev_ebitda_inv", "val_shareholder_yield"],
    "quality": ["qual_roic", "qual_altman_z", "qual_piotroski_f", "qual_fcf_margin"],
    "growth": ["grw_revenue_yoy", "grw_earnings_yoy", "grw_fcf_growth_smoothed", "grw_margin_expansion_1y"],
    "revisions": ["rev_pt_target_upside_30d", "rev_grade_diffusion", "rev_agreement_90d", "rev_forward_eps_revision_90d"],
    "institutional": ["inst_high_conviction", "inst_net_flow_dollars", "inst_concentration_pct", "inst_multi_fund_open"],
    "insider": ["ins_net_dollar_flow", "ins_buy_sell_ratio", "ins_large_buy_dollars", "ins_purchase_frequency_180d"],
    "short": ["si_squeeze_setup_flag", "si_short_covering_signal", "si_days_to_cover", "si_short_pct_float_pctile_252d"],
}


# --- Family C/D: valuation mean-reversion (current vs own trailing avg) ---
def make_valuation_reversion(col, lookback_months):
    def fn(scores, parent_scores, pool, ctx):
        frame_by_date = ctx["bundle"]["subfactor_frames"]
        i, dates = ctx["i"], ctx["dates"]
        j = max(i - lookback_months + 1, 0)
        if j == i:
            return None
        cur_frame = frame_by_date.get(ctx["date"])
        if cur_frame is None or col not in cur_frame.columns:
            return None
        cur = cur_frame[col].reindex(pool)
        hist = [frame_by_date[dd][col].reindex(pool) for dd in dates[j:i]
               if dd in frame_by_date and col in frame_by_date[dd].columns]
        if len(hist) < 12:
            return None
        return cur - pd.concat(hist, axis=1).mean(axis=1)
    return fn


VAL_REVERSION_60M_COLS = ["val_book_to_price", "val_fcf_yield_clean", "val_ev_ebitda_inv",
                         "val_ev_fcf_inv", "val_ev_revenue_inv", "val_dividend_yield",
                         "val_buyback_yield"]


# --- Family E: parent-level reversion (composite + 4 parents, 60mo) ---
def make_parent_reversion(parent_name, lookback_months):
    def fn(scores, parent_scores, pool, ctx):
        data = ctx["bundle"]["data"]
        i, dates = ctx["i"], ctx["dates"]
        j = max(i - lookback_months + 1, 0)
        if j == i:
            return None
        if parent_name == "__composite__":
            cur = data.comp.get(ctx["date"])
            hist = [data.comp[dd] for dd in dates[j:i] if dd in data.comp]
        else:
            cur_p = ctx["bundle"]["parent_scores"].get(ctx["date"], {})
            cur = cur_p.get(parent_name)
            hist = [ctx["bundle"]["parent_scores"][dd].get(parent_name) for dd in dates[j:i]
                   if dd in ctx["bundle"]["parent_scores"]]
            hist = [h for h in hist if h is not None]
        if cur is None or len(hist) < 12:
            return None
        cur = cur.reindex(pool)
        hist_avg = pd.concat([h.reindex(pool) for h in hist], axis=1).mean(axis=1)
        return (cur - hist_avg).reindex(pool)
    return fn


# --- Family F: price mean-reversion, current vs own trailing 5y average ---
def metric_price_reversion(scores, parent_scores, pool, ctx):
    matrix = ctx["bundle"]["data"].matrix
    d = ctx["date"]
    if d not in matrix.index:
        return None
    idx = matrix.index.get_loc(d)
    start = max(idx - 1260, 0)
    if idx - start < 500:
        return None
    cols = [t for t in pool if t in matrix.columns]
    window = matrix.iloc[start:idx + 1][cols]
    return (window.iloc[-1] / window.mean() - 1.0)


# --- Family G: distance from 5-year (1260d) high/low ---
def make_dist_from_high_5y():
    def fn(scores, parent_scores, pool, ctx):
        matrix = ctx["bundle"]["data"].matrix
        d = ctx["date"]
        if d not in matrix.index:
            return None
        idx = matrix.index.get_loc(d)
        start = max(idx - 1260, 0)
        if idx - start < 500:
            return None
        cols = [t for t in pool if t in matrix.columns]
        window = matrix.iloc[start:idx + 1][cols]
        return window.iloc[-1] / window.max() - 1.0
    return fn


def make_dist_from_low_5y():
    def fn(scores, parent_scores, pool, ctx):
        matrix = ctx["bundle"]["data"].matrix
        d = ctx["date"]
        if d not in matrix.index:
            return None
        idx = matrix.index.get_loc(d)
        start = max(idx - 1260, 0)
        if idx - start < 500:
            return None
        cols = [t for t in pool if t in matrix.columns]
        window = matrix.iloc[start:idx + 1][cols]
        return window.iloc[-1] / window.min() - 1.0
    return fn


# --- Family H: per-parent subfactor breadth (count of that parent's own
# subfactors above the day's cross-sectional median, one idea per parent) ---
def make_parent_breadth(parent_name):
    prefix = PREFIX[parent_name]
    def fn(scores, parent_scores, pool, ctx):
        frame = _sub_frame(ctx)
        if frame is None:
            return None
        cols = [c for c in frame.columns if c.startswith(prefix)]
        if not cols:
            return None
        sub = frame[cols]
        med = sub.median()
        above = (sub >= med).sum(axis=1)
        return above.reindex(pool)
    return fn


# --- Family I: rank-sum of ALL raw subfactor columns (not parent ranks) ---
def metric_subfactor_rank_sum(scores, parent_scores, pool, ctx):
    frame = _sub_frame(ctx)
    if frame is None:
        return None
    ranks = frame.rank(pct=True)
    return ranks.sum(axis=1).reindex(pool)


# --- Family J/K: min / max across ALL 8 parents ---
def metric_all_parents_min(scores, parent_scores, pool, ctx):
    if len(parent_scores) < 8:
        return None
    df = pd.DataFrame({p: s for p, s in parent_scores.items()})
    return df.min(axis=1).reindex(pool)


def metric_all_parents_max(scores, parent_scores, pool, ctx):
    if len(parent_scores) < 8:
        return None
    df = pd.DataFrame({p: s for p, s in parent_scores.items()})
    return df.max(axis=1).reindex(pool)


# --- Family L/M: cross-sectional dispersion across the 8 parent scores ---
def metric_parent_dispersion(scores, parent_scores, pool, ctx):
    if len(parent_scores) < 8:
        return None
    df = pd.DataFrame({p: s for p, s in parent_scores.items()})
    return df.std(axis=1).reindex(pool)


# --- Family N/O: hard-threshold gates (as opposed to continuous floors) ---
def metric_quality_gated_revisions(scores, parent_scores, pool, ctx):
    q, r = parent_scores.get("quality"), parent_scores.get("revisions")
    if q is None or r is None:
        return None
    q, r = q.reindex(pool), r.reindex(pool)
    med = q.median()
    return r.where(q >= med, -1e9)


def metric_quality_gated_insider(scores, parent_scores, pool, ctx):
    q, ins = parent_scores.get("quality"), parent_scores.get("insider")
    if q is None or ins is None:
        return None
    q, ins = q.reindex(pool), ins.reindex(pool)
    med = q.median()
    return ins.where(q >= med, -1e9)


def metric_growth_gated_momentum(scores, parent_scores, pool, ctx):
    g, m = parent_scores.get("growth"), parent_scores.get("momentum")
    if g is None or m is None:
        return None
    g, m = g.reindex(pool), m.reindex(pool)
    med = g.median()
    return m.where(g >= med, -1e9)


def metric_momentum_gated_quality(scores, parent_scores, pool, ctx):
    m, q = parent_scores.get("momentum"), parent_scores.get("quality")
    if m is None or q is None:
        return None
    m, q = m.reindex(pool), q.reindex(pool)
    med = m.median()
    return q.where(m >= med, -1e9)


# --- Family P: 3-way AVERAGE (not min) of the same 3 parents already known
# to matter, as a contrast to the champion's strict AND/min logic ---
def metric_insider_revisions_institutional_avg(scores, parent_scores, pool, ctx):
    a = parent_scores.get("insider")
    b = parent_scores.get("revisions")
    c = parent_scores.get("institutional")
    if a is None or b is None or c is None:
        return None
    return pd.concat([a, b, c], axis=1).mean(axis=1).reindex(pool)


# --- Family Q: literal random pick from the top-20 pool (deterministic per
# seed via a hash, so it plays nicely with deterministic_candidate=True) ---
def make_random_metric(seed):
    def fn(scores, parent_scores, pool, ctx):
        d = str(ctx["date"])
        return pd.Series({t: (hash((t, d, seed)) % 10_000_000) / 10_000_000.0 for t in pool})
    return fn


def generate_ideas():
    ideas = []

    # A: parent-pair MIN, all 28 pairs
    for p1, p2 in combinations(PARENTS, 2):
        name = f"parentmin_{p1}_{p2}_top20"
        hyp = f"floor/AND logic generalized: min({p1},{p2}) -- both must be strong, testing if this mechanism (which works for insider+revisions) is general or specific"
        ideas.append((name, hyp, make_parent_min(p1, p2), False))

    # B: subfactor-level rankings, 8 parents x 4
    for parent, cols in SUBFACTOR_LEVEL_PICKS.items():
        for col in cols:
            name = f"sublevel_{col}_top20"
            hyp = f"rank directly by the raw {parent} subfactor '{col}' (subfactor-level, not the parent aggregate)"
            ideas.append((name, hyp, make_subfactor_level(col), False))

    # C: valuation reversion at 60mo, 7 cols (val_earnings_yield_60mo already tested standalone)
    for col in VAL_REVERSION_60M_COLS:
        name = f"valrev60m_{col}_top20"
        hyp = f"mean-reversion: current {col} percentile minus its own trailing 5-year average"
        ideas.append((name, hyp, make_valuation_reversion(col, 60), False))

    # D: earnings-yield reversion at alternate lookbacks (12/24/36mo)
    for m in (12, 24, 36):
        name = f"valrev{m}m_val_earnings_yield_top20"
        hyp = f"mean-reversion: current earnings-yield percentile minus its own trailing {m}-month average (shorter lookback than the already-tested 60-month version)"
        ideas.append((name, hyp, make_valuation_reversion("val_earnings_yield", m), False))

    # E: parent-level reversion at 60mo (composite + 4 parents) + revisions at 12mo
    for parent in ["__composite__", "momentum", "value", "quality", "growth"]:
        label = "composite" if parent == "__composite__" else parent
        name = f"parentrev60m_{label}_top20"
        hyp = f"mean-reversion at the PARENT level: current {label} score/percentile minus its own trailing 5-year average (contrarian on {label} itself)"
        ideas.append((name, hyp, make_parent_reversion(parent, 60), False))
    ideas.append(("parentrev12m_revisions_top20",
                 "mean-reversion on the revisions parent at a short 12-month lookback (contrarian: buy names whose revisions have gotten relatively worse vs their own recent average)",
                 make_parent_reversion("revisions", 12), False))

    # F: price mean-reversion
    ideas.append(("price_reversion_5y_top20",
                 "classic long-term reversal: current price furthest BELOW its own trailing 5-year average",
                 metric_price_reversion, True))

    # G: distance from 5y high/low
    ideas.append(("near_5y_high_top20", "closest to the trailing 5-year price high (long-horizon trend)", make_dist_from_high_5y(), False))
    ideas.append(("near_5y_low_top20", "closest to the trailing 5-year price low (deep long-horizon value/oversold)", make_dist_from_low_5y(), True))

    # H: per-parent subfactor breadth
    for parent in PARENTS:
        name = f"breadth_{parent}_top20"
        hyp = f"count of {parent}'s OWN subfactors above that day's cross-sectional median (within-family breadth/conviction, not a single subfactor)"
        ideas.append((name, hyp, make_parent_breadth(parent), False))

    # I: subfactor rank-sum
    ideas.append(("subfactor_rank_sum_top20",
                 "sum of percentile ranks across ALL ~90 raw subfactor columns (finer-grained than the parent-rank-sum already tested)",
                 metric_subfactor_rank_sum, False))

    # J/K: min/max across all 8 parents
    ideas.append(("all_parents_min_top20", "ALL 8 parents must be simultaneously strong (min across all 8) -- the most extreme version of the floor/AND logic", metric_all_parents_min, False))
    ideas.append(("all_parents_max_top20", "the single BEST of the 8 parents is enough (max across all 8) -- the most extreme OR/negative-control contrast", metric_all_parents_max, False))

    # L/M: dispersion
    ideas.append(("low_parent_dispersion_top20", "most BALANCED profile: lowest cross-sectional std across the 8 parent scores (no glaring weakness)", metric_parent_dispersion, True))
    ideas.append(("high_parent_dispersion_top20", "most BARBELL profile: highest cross-sectional std across the 8 parent scores (one dominant strength, contrarian to the balanced idea)", metric_parent_dispersion, False))

    # N/O/R/S: hard-threshold gates
    ideas.append(("quality_gated_revisions_top20", "hard threshold (not continuous floor): revisions score counts only if quality is above the day's median, else disqualified", metric_quality_gated_revisions, False))
    ideas.append(("quality_gated_insider_top20", "hard threshold: insider score counts only if quality is above the day's median, else disqualified", metric_quality_gated_insider, False))
    ideas.append(("growth_gated_momentum_top20", "hard threshold: momentum score counts only if growth is above the day's median, else disqualified", metric_growth_gated_momentum, False))
    ideas.append(("momentum_gated_quality_top20", "hard threshold: quality score counts only if momentum is above the day's median, else disqualified", metric_momentum_gated_quality, False))

    # P: 3-way average contrast
    ideas.append(("insider_revisions_institutional_avg_top20", "3-way AVERAGE (not min) of insider/revisions/institutional -- contrast to the champion's strict AND/min logic using the same parents", metric_insider_revisions_institutional_avg, False))

    # Q: literal random picks, 3 seeds
    for seed in (1, 2, 3):
        name = f"random_pick_seed{seed}_top20"
        hyp = f"literal random pick from the top-20 pool (fixed seed {seed}) -- pure null-strategy sanity check against the champion"
        ideas.append((name, hyp, make_random_metric(seed), False))

    return ideas


def run_batch():
    ideas = generate_ideas()
    print(f"{len(ideas)} ideas queued\n", flush=True)

    bundle = H.get_data()
    print("computing champion baseline (500 sims, ONCE, reused for every idea) ...", flush=True)
    t0 = time.time()
    base_raw, base_metrics = compute_baseline_paths(CHAMPION_SELECTOR, bundle=bundle, n_sims=N_SIMS)
    print(f"baseline ready in {time.time()-t0:.0f}s\n", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    t_batch0 = time.time()
    n_done = 0
    for idx, (name, hypothesis, metric_fn, ascending) in enumerate(ideas):
        summary_path = EXP_DIR / f"{name}.json"
        if summary_path.exists():
            print(f"[{idx+1}/{len(ideas)}] {name}: already done, skipping", flush=True)
            continue

        candidate_selector = make_selector(name, metric_fn, ascending)

        def make_pert(noise_mult, _mf=metric_fn, _asc=ascending):
            return make_perturbed_selector(_mf, _asc, noise_mult)

        t0 = time.time()
        report = evaluate_promotion(
            candidate_selector, baseline_selector=CHAMPION_SELECTOR,
            make_perturbed_selector=make_pert, n_sims=N_SIMS,
            deterministic_candidate=True, bundle=bundle,
            baseline_precomputed=(base_raw, base_metrics),
        )
        elapsed = time.time() - t0

        out_path = OUT_DIR / f"promotion_{name}.json"
        with out_path.open("w") as fh:
            json.dump(report, fh, indent=2, default=float)

        summary = build_summary(report, name, hypothesis)
        summary["baseline"] = "insider_revisions_min_top20"
        with summary_path.open("w") as fh:
            json.dump(summary, fh, indent=2, default=float)

        with MANIFEST.open("a") as fh:
            fh.write(json.dumps({
                "name": name, "decision": report["decision"],
                "sharpe_improve": report["sharpe_improve"],
                "win_rate": report["win_rate"],
                "perturbation_win_rate": report["perturbation_win_rate"],
                "n_years_positive": report["n_years_positive"],
                "n_years": report["n_years"],
                "best_year_removed_negative": report["remainder_after_best_year"] < 0,
            }, default=float) + "\n")

        n_done += 1
        total_elapsed = time.time() - t_batch0
        rate = n_done / total_elapsed
        remaining = len(ideas) - idx - 1
        eta_min = (remaining / rate) / 60 if rate > 0 else float("nan")
        print(f"[{idx+1}/{len(ideas)}] {name:45s} decision={report['decision']:9s} "
             f"sharpe_improve={report['sharpe_improve']:+.4f}  win_rate={report['win_rate']:.1%}  "
             f"pert_win={report['perturbation_win_rate']!s:>6}  ({elapsed:.0f}s, ETA {eta_min:.0f}m)",
             flush=True)

    print(f"\ndone. {n_done} new ideas tested this run, {len(ideas)} total in the batch.")


if __name__ == "__main__":
    run_batch()
