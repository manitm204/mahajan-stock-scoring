"""Parent-factor construction by *transparent rank-and-diversify* selection.

A deliberately simple, explainable alternative to the combo-enumeration engine in
:mod:`research.parent_construction`. For every parent bucket it:

1. measures each sub-factor on five standard reads — mean IC, IC-IR, Q5-Q1 spread,
   hit rate, coverage. Every read that depends on a horizon shares the **3M/6M**
   horizon so the ranking is internally consistent: mean IC (the metric used to *pick*)
   is the mean of the 3M and 6M ICs, and IC-IR / spread / hit / monotonicity are pooled
   over the 3M and 6M forward-return series too. The 1M and 12M ICs are still reported
   but deliberately excluded from selection (1M is noisy turnover, 12M overlaps
   heavily); coverage is horizon-free; all reads are point-in-time;
2. turns each metric into a within-bucket percentile rank and blends them into one
   Sub-factor Score = 0.50*IC + 0.25*IC-IR + 0.15*spread + 0.05*hit + 0.05*coverage;
3. sorts subs by that score and builds the parent greedily — take the top sub, then
   add the next-best only if it is genuinely diversifying (cross-sectional Spearman
   R² below 0.60 against *every* already-selected sub), stopping at three subs, when
   no candidate is both diversifying and positive-IC, or when the next candidate's
   mean IC turns non-positive;
4. weights the chosen subs in proportion to their standalone mean IC, caps any single
   sub at 50% and renormalises.

The aim is not to maximise one number but to produce parent factors that are
predictive, consistent, diversified and interpretable — built from two or three
complementary signals rather than one dominant sub. Read-only: it reads the captured
:class:`ScorePanel` and never touches production scoring or weights.

A parent whose leading sub has a non-positive selection (3M/6M-mean) IC is flagged, but
horizon-aware rather than binary: ``HORIZON_SPECIFIC`` (positive at some horizon but
negative on the selection mean), ``POSSIBLE_SIGN_INVERSION`` (strongly negative across
*every* horizon — the score direction is probably backwards), or plain
``NO_POSITIVE_IC`` (flat / weakly negative everywhere). The sign-inversion test still
scans all four horizons, so 1M/12M inform the flag even though they never pick.

Honesty note: unlike the combo-enumeration study these metrics are full-sample (no
held-out tail). The framework trades that out-of-sample check for transparency, so
treat marginal IC differences between subs with care.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .panel import ScorePanel
from .subset_selection import correlation_matrix, inventory, subfactor_performance

# --- selection rules (all fixed, so the construction is fully reproducible) ---
RANK_WEIGHTS = {"mean_ic": 0.50, "ic_ir": 0.25, "spread_q5_q1": 0.15,
                "hit_rate": 0.05, "coverage": 0.05}     # sums to 1.00
RANK_COLS = {"mean_ic": "ic_rank", "ic_ir": "ic_ir_rank",
             "spread_q5_q1": "spread_rank", "hit_rate": "hit_rank",
             "coverage": "cov_rank"}
METRIC_COLS = ["mean_ic", "ic_ir", "spread_q5_q1", "hit_rate", "coverage"]
HORIZON_COLS = ["ic_1M", "ic_3M", "ic_6M", "ic_12M"]   # all four are reported
SELECT_HORIZONS = ["ic_3M", "ic_6M"]                   # …but only these two pick
# The horizon(s) the *distributional* stats (IC-IR, Q5-Q1 spread, hit, monotonicity)
# are pooled over, kept consistent with the 3M/6M pick metric. Derived from
# SELECT_HORIZONS (strip the ``ic_`` prefix) so the two never drift apart.
STAT_HORIZONS = tuple(h.replace("ic_", "") for h in SELECT_HORIZONS)   # ("3M", "6M")
R2_MAX = 0.60             # add a sub only if R² < this vs every selected sub
MAX_SUBS = 3              # cap the parent at three complementary subs
SINGLE_CAP = 0.50         # no single sub may exceed this share of the parent
SIGN_INV_IC = -0.02       # multi-horizon mean IC ≤ this + negative at (nearly) every
                          # horizon => the direction is probably inverted, not "no signal"
MIN_NAMES = 20


def _f(v) -> str:
    return "—" if v is None or (isinstance(v, float) and pd.isna(v)) else f"{v:+.3f}"


# ---------------------------------------------------------------------------
# 1-2. Per-sub metrics + within-bucket percentile ranks
# ---------------------------------------------------------------------------
def parent_metrics(perf: pd.DataFrame, inv: pd.DataFrame) -> pd.DataFrame:
    """Per-sub scorecard reduced to the ranking metrics (+ per-horizon IC + parent tag).

    Reuses the existing panel readers: :func:`subfactor_performance` supplies the
    per-horizon ICs plus the IC-IR, Q5-Q1 spread, hit rate and monotonicity pooled over
    its ``stat_horizons`` (3M/6M here, set in :func:`run_selection`); :func:`inventory`
    supplies coverage. ``mean_ic`` — the metric used to rank, weight and gate selection —
    is the mean of the **3M and 6M** ICs only: 1M is dominated by turnover noise and 12M
    is heavily overlapping, so both are reported (kept as ``ic_1M``/``ic_12M``) but held
    out of the pick. Falls back to whatever selection horizons are present, then to the
    full set, then to the 1M ``ic_spearman``.
    """
    df = perf.merge(inv[["sub_factor", "coverage"]], on="sub_factor", how="left")
    df = df.rename(columns={"information_ratio": "ic_ir"})
    sel = [c for c in SELECT_HORIZONS if c in df.columns]
    have = sel or [c for c in HORIZON_COLS if c in df.columns]
    df["mean_ic"] = df[have].mean(axis=1) if have else df.get("ic_spearman")
    return df


def horizon_flag(row: pd.Series) -> str:
    """Classify a sub-factor's usability from its horizon IC profile.

    ``""`` positive selection (3M/6M) mean (usable); ``HORIZON_SPECIFIC`` positive at
    some horizon but negative on the selection mean; ``POSSIBLE_SIGN_INVERSION`` strongly
    negative at (nearly) every horizon; ``NO_POSITIVE_IC`` flat / weakly negative
    everywhere. The horizon scan still uses all four ICs, so 1M/12M inform the flag."""
    hs = [row[c] for c in HORIZON_COLS if c in row.index and pd.notna(row[c])]
    multi = row.get("mean_ic")
    if not hs or pd.isna(multi):
        return "NO_DATA"
    if multi > 0:
        return ""
    if any(x > 0 for x in hs):
        return "HORIZON_SPECIFIC"
    if sum(x < 0 for x in hs) >= max(3, len(hs) - 1) and multi <= SIGN_INV_IC:
        return "POSSIBLE_SIGN_INVERSION"
    return "NO_POSITIVE_IC"


def _pct_rank(s: pd.Series) -> pd.Series:
    """Within-bucket percentile rank (higher metric -> higher rank; NaN -> worst)."""
    s = s.fillna(s.min()) if s.notna().any() else s.fillna(0.0)
    return s.rank(pct=True, method="average")


def rank_subfactors(sub_df: pd.DataFrame) -> pd.DataFrame:
    """Add the five percentile ranks + the blended Sub-factor Score, sorted best-first."""
    df = sub_df.copy().reset_index(drop=True)
    for m, rc in RANK_COLS.items():
        df[rc] = _pct_rank(df[m])
    df["subfactor_score"] = sum(RANK_WEIGHTS[m] * df[RANK_COLS[m]] for m in RANK_WEIGHTS)
    df["horizon_flag"] = df.apply(horizon_flag, axis=1)
    return df.sort_values("subfactor_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Greedy diversification via cross-sectional Spearman R²
# ---------------------------------------------------------------------------
def _r2(corr: pd.DataFrame, a: str, b: str) -> float:
    if a in corr.index and b in corr.columns and pd.notna(corr.loc[a, b]):
        return float(corr.loc[a, b]) ** 2
    return np.nan


def select_subfactors(ranked: pd.DataFrame, corr: pd.DataFrame, *,
                      r2_max: float = R2_MAX, max_subs: int = MAX_SUBS
                      ) -> tuple[list[str], list[dict], str]:
    """Build the parent: top sub first, then only diversifying (R²<``r2_max``),
    positive-IC subs, up to ``max_subs``. Returns (selected, decisions, stop_reason)."""
    order = list(ranked["sub_factor"])
    ic = dict(zip(ranked["sub_factor"], ranked["mean_ic"]))
    selected = [order[0]]
    decisions = [{"sub_factor": order[0], "status": "selected",
                  "reason": "highest sub-factor score in the bucket — always taken",
                  "max_r2": np.nan, "r2_partner": None}]
    considered = {order[0]}
    stop: str | None = None

    for c in order[1:]:
        if len(selected) >= max_subs:
            stop = f"reached the {max_subs}-sub cap"
            break
        if not (ic[c] > 0):                       # NaN or non-positive mean IC
            decisions.append({"sub_factor": c, "status": "rejected_negative_ic",
                              "reason": f"mean IC {_f(ic[c])} ≤ 0 — stop",
                              "max_r2": np.nan, "r2_partner": None})
            considered.add(c)
            stop = "next candidate's mean IC ≤ 0"
            break
        r2s = {s: _r2(corr, c, s) for s in selected}
        finite = {s: v for s, v in r2s.items() if pd.notna(v)}
        max_r2 = max(finite.values()) if finite else np.nan
        partner = max(finite, key=finite.get) if finite else None
        if pd.notna(max_r2) and max_r2 >= r2_max:
            decisions.append({"sub_factor": c, "status": "rejected_redundant",
                              "reason": f"R²={max_r2:.2f} ≥ {r2_max:.2f} vs `{partner}`",
                              "max_r2": max_r2, "r2_partner": partner})
            considered.add(c)
            continue
        selected.append(c)
        decisions.append({"sub_factor": c, "status": "selected",
                          "reason": (f"diversifying (max R²={max_r2:.2f} vs `{partner}`) "
                                     f"and mean IC {_f(ic[c])} > 0"),
                          "max_r2": max_r2, "r2_partner": partner})
        considered.add(c)

    if stop is None:
        stop = ("no remaining candidate passed the R² filter"
                if len(selected) < max_subs and len(considered) < len(order)
                else "all diversifying candidates added")
    for c in order:
        if c not in considered:
            decisions.append({"sub_factor": c, "status": "not_selected",
                              "reason": "selection stopped before this candidate was reached",
                              "max_r2": np.nan, "r2_partner": None})
    return selected, decisions, stop


# ---------------------------------------------------------------------------
# 4. Weight ∝ mean IC, cap 50%, renormalise
# ---------------------------------------------------------------------------
def parent_weights(selected: list[str], ic_map: dict[str, float], *,
                   cap: float = SINGLE_CAP) -> tuple[dict[str, float], bool]:
    """Weights proportional to (positive) mean IC, water-filled to a per-sub ``cap``.

    Returns (weights, no_positive_ic). If no selected sub has positive IC the parent
    has no usable long signal as constructed — weights fall back to equal and the flag
    is raised so the report can surface it."""
    raw = {s: max(float(ic_map[s]) if pd.notna(ic_map[s]) else 0.0, 0.0) for s in selected}
    tot = sum(raw.values())
    if tot <= 1e-12:
        return {s: 1.0 / len(selected) for s in selected}, True
    w = {s: raw[s] / tot for s in selected}
    for _ in range(20):
        over = [s for s in w if w[s] > cap + 1e-12]
        if not over:
            break
        excess = sum(w[s] - cap for s in over)
        for s in over:
            w[s] = cap
        under = [s for s in w if w[s] < cap - 1e-12]
        pool = sum(w[s] for s in under)
        if not under or pool <= 1e-12:
            break
        for s in under:
            w[s] += excess * w[s] / pool
    tot = sum(w.values())
    return {s: w[s] / tot for s in w}, False


def formula(weights: dict[str, float]) -> str:
    """Human-readable parent formula, e.g. ``0.50*grw_earnings_yoy + 0.30*...``."""
    items = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    return " + ".join(f"{w:.2f}*{s}" for s, w in items)


def r2_matrix(selected: list[str], corr: pd.DataFrame) -> pd.DataFrame:
    """R² (squared avg cross-sectional Spearman corr) among the selected subs."""
    return corr.reindex(index=selected, columns=selected).astype(float).pow(2)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
@dataclass
class ParentSelection:
    parent: str
    ranking: pd.DataFrame          # every sub: metrics + ranks + score + status
    selected: list[str]
    weights: dict[str, float]
    r2: pd.DataFrame               # R² matrix among selected
    decisions: pd.DataFrame        # per-sub include/reject reason
    metrics: pd.DataFrame          # the five metrics + weight for selected subs
    formula: str
    stop_reason: str
    no_positive_ic: bool           # leading sub has non-positive multi-horizon mean IC
    signal_flag: str               # "" | HORIZON_SPECIFIC | POSSIBLE_SIGN_INVERSION | NO_POSITIVE_IC


def build_parent(parent: str, sub_df: pd.DataFrame, corr: pd.DataFrame) -> ParentSelection:
    """Rank -> diversify-select -> weight, for one parent bucket."""
    ranked = rank_subfactors(sub_df)
    selected, decisions, stop = select_subfactors(ranked, corr)
    ic_map = dict(zip(ranked["sub_factor"], ranked["mean_ic"]))
    weights, no_pos = parent_weights(selected, ic_map)
    dec_df = pd.DataFrame(decisions)
    status = dec_df.set_index("sub_factor")["status"]
    ranked = ranked.assign(status=ranked["sub_factor"].map(status))
    metrics = (ranked.set_index("sub_factor").loc[selected, METRIC_COLS]
               .assign(weight=[weights[s] for s in selected]))
    # Parent-level signal read from the leading sub's horizon profile.
    signal_flag = horizon_flag(ranked.set_index("sub_factor").loc[selected[0]])
    return ParentSelection(parent, ranked, selected, weights,
                           r2_matrix(selected, corr), dec_df, metrics,
                           formula(weights), stop, no_pos, signal_flag)


def run_selection(panel: ScorePanel, fwd_by_h: dict[str, dict[str, pd.Series]],
                  parents: list[str] | None = None, *,
                  min_names: int = MIN_NAMES) -> list[ParentSelection]:
    """Run the whole study — compute per-sub metrics + correlations once, then build
    every parent from its own sub-factors.

    The distributional stats (IC-IR / spread / hit / monotonicity) are pooled over
    ``STAT_HORIZONS`` (3M & 6M) so every ranking metric shares the 3M/6M horizon of the
    ``mean_ic`` pick — instead of the legacy 1M read. Falls back to 1M only if neither
    3M nor 6M forward-return set is present in ``fwd_by_h``."""
    stat_h = tuple(h for h in STAT_HORIZONS if h in fwd_by_h) or ("1M",)
    perf = subfactor_performance(panel, fwd_by_h, min_names=min_names, stat_horizons=stat_h)
    inv = inventory(panel)
    corr = correlation_matrix(panel, min_names=min_names)
    merged = parent_metrics(perf, inv)
    parents = parents or [p for p in panel.parent_keys if panel.sub_by_parent.get(p)]
    results: list[ParentSelection] = []
    for parent in parents:
        sub_df = merged[merged["parent"] == parent]
        if not sub_df.empty:
            results.append(build_parent(parent, sub_df, corr))
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
_SHOW = ["sub_factor", "ic_1M", "ic_3M", "ic_6M", "ic_12M", "mean_ic", "ic_ir",
         "spread_q5_q1", "hit_rate", "coverage", "subfactor_score", "status",
         "horizon_flag"]

# Score direction declared in factors/insider.py + factors/institutional.py — every
# insider/institutional sub is higher_is_better=True (buying / accumulation assumed
# bullish). Used only to annotate the sign audit.
AUDITED_DIRECTION = {
    "ins_net_dollar_flow": True, "ins_buy_sell_ratio": True, "ins_high_conviction_buy": True,
    # research-library insider subs (expansion source): all buying/accumulation = bullish
    "ins_purchase_frequency_180d": True, "ins_cluster_buyers_180d": True,
    "ins_ceo_cfo_buy_dollars": True, "ins_sell_pressure_inv": False, "ins_net_flow_90d": True,
    "ins_officer_buy_ratio": True, "ins_no_selling_flag": True,
    "ins_buy_dollar_volume": True, "ins_buy_count": True,
    "ins_large_buy_dollars": True, "ins_large_buy_count": True,
    "inst_fund_count": True, "inst_net_share_change": True, "inst_new_positions": True,
    "inst_multi_fund_open": True, "inst_high_conviction": True,
}
AUDIT_PARENTS = ("insider", "institutional")
AUDIT_COLS = ["sub_factor", "ic_1M", "ic_3M", "ic_6M", "ic_12M", "mean_ic",
              "spread_q5_q1", "hit_rate", "coverage", "higher_is_better", "horizon_flag"]

_FLAG_NOTE = {
    "POSSIBLE_SIGN_INVERSION":
        "**POSSIBLE_SIGN_INVERSION** — the leading sub is negative at (nearly) every "
        "horizon and strongly so on the multi-horizon mean. The score direction is "
        "probably backwards; inspect the buys-vs-sells / dollar-flow sign and the P/S "
        "ingest mapping before using it.",
    "HORIZON_SPECIFIC":
        "**HORIZON_SPECIFIC** — the leading sub is positive at some horizon but negative "
        "on the multi-horizon mean, so it is not a clean long signal (it may work only at "
        "a specific horizon).",
    "NO_POSITIVE_IC":
        "**NO_POSITIVE_IC** — the leading sub is flat / weakly negative at every horizon: "
        "no usable long signal as constructed (weights fell back to equal).",
    "NO_DATA": "**NO_DATA** — insufficient horizon IC to judge this parent.",
}


def _fmt(v) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    try:
        if pd.isna(v):
            return "—"
    except (TypeError, ValueError):
        pass
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return str(int(v)) if float(v).is_integer() else f"{v:.4f}"
    return "" if v is None else str(v)


def _md_table(df: pd.DataFrame, cols: list[str] | None = None,
              index_label: str | None = None) -> str:
    cols = cols or list(df.columns)
    header = ([index_label] if index_label else []) + list(cols)
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    for idx, r in df.iterrows():
        cells = ([f"`{idx}`"] if index_label else []) + [_fmt(r[c]) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _parent_section(res: ParentSelection) -> list[str]:
    L = [f"\n## {res.parent}\n"]
    wtxt = ", ".join(f"`{s}` {res.weights[s]:.0%}" for s in res.selected)
    L.append(f"**Selected ({len(res.selected)}):** {wtxt}")
    L.append(f"\n**Formula:** `{res.formula}`\n")
    if res.signal_flag:
        L.append("> ⚠ " + _FLAG_NOTE.get(res.signal_flag, res.signal_flag) + "\n")
    L.append(f"_Selection stopped: {res.stop_reason}._\n")

    L.append("**Sub-factor ranking (within bucket):**\n")
    L.append(_md_table(res.ranking, _SHOW))
    L.append("\n**Selected metrics & weights:**\n")
    L.append(_md_table(res.metrics.reset_index(),
                       ["sub_factor", "mean_ic", "ic_ir", "spread_q5_q1",
                        "hit_rate", "coverage", "weight"]))
    L.append("\n**R² matrix (selected):**\n")
    L.append(_md_table(res.r2.round(3), index_label="R²"))

    L.append("\n**Why each decision:**")
    dec = res.decisions.set_index("sub_factor")
    for sub in res.ranking["sub_factor"]:
        row = dec.loc[sub]
        L.append(f"- `{sub}` [**{row['status']}**]: {row['reason']}")
    L.append("")
    return L


# Which forward-return horizon feeds each metric, and why. Documented explicitly in the
# report so the selection is auditable end-to-end.
_HORIZON_DOC = [
    ("mean IC (pick metric)", "3M & 6M mean", "primary signal-strength read; 1M too "
     "noisy/turnover-driven, 12M overlaps too heavily to add independent information"),
    ("IC-IR", "3M & 6M pooled", "IC series pooled across 3M and 6M, then mean/std — "
     "consistency of the *same* horizon the pick uses, not a 1M read"),
    ("Q5-Q1 spread", "3M & 6M avg profile", "quintile profile averaged over the 3M and "
     "6M periods, then top-minus-bottom"),
    ("hit rate", "3M & 6M pooled", "share of pooled 3M/6M IC observations that are > 0"),
    ("monotonicity", "3M & 6M avg profile", "computed off the same averaged 3M/6M "
     "quintile profile as the spread"),
    ("coverage", "horizon-free", "fraction of the universe with a non-missing score — "
     "a name-availability read, independent of any forward-return horizon"),
    ("ic_1M / ic_12M", "reported only", "shown for context and for the sign-inversion "
     "flag scan, but never enter the ranking, weighting or stop rules"),
]


def _horizon_doc_section() -> list[str]:
    """A table stating exactly which forward-return horizon each ranking metric uses."""
    df = pd.DataFrame(_HORIZON_DOC, columns=["metric", "horizon", "why"])
    return ["**Horizon used by each metric** (all point-in-time):\n",
            _md_table(df, ["metric", "horizon", "why"]), ""]


def write_report(out_dir: Path, results: list[ParentSelection], meta: dict) -> None:
    L = ["# Parent-Factor Construction — Rank & Diversify\n"]
    L.append(
        "A simple, transparent parent build. Each sub-factor is scored on five reads "
        "(mean IC, IC-IR, Q5-Q1 spread, hit rate, coverage), each read is turned into a "
        "within-bucket percentile rank, and the ranks are blended into one **Sub-factor "
        "Score** = 0.50·IC + 0.25·IC-IR + 0.15·spread + 0.05·hit + 0.05·coverage (IC here "
        "is the 3M/6M-mean). The "
        "parent takes the top sub, then adds the next-best only if it is genuinely "
        f"diversifying (cross-sectional Spearman **R² < {R2_MAX:.2f}** vs every selected "
        f"sub), up to **{MAX_SUBS}** subs — stopping early when no candidate is both "
        "diversifying and positive-IC. Selected subs are weighted ∝ mean IC, capped at "
        f"**{SINGLE_CAP:.0%}**, renormalised. Read-only; nothing is applied to production.\n")
    L.append(
        "**Mean IC — the metric that picks — is the mean of the 3M and 6M ICs only.** The "
        "1M and 12M ICs are still reported (columns `ic_1M`/`ic_12M`) but excluded from "
        "selection: 1M is dominated by turnover noise and 12M overlaps heavily. **Every "
        "other ranking metric shares that 3M/6M horizon** — IC-IR, Q5-Q1 spread, hit rate "
        "and monotonicity are all pooled over the 3M and 6M forward-return series, so no "
        "single metric silently reintroduces the 1M read. A parent whose leading sub has a "
        "non-positive 3M/6M-mean is flagged `HORIZON_SPECIFIC` (positive at some horizon), "
        "`POSSIBLE_SIGN_INVERSION` (strongly negative at every horizon), or `NO_POSITIVE_IC` "
        "(flat) — the flag scan still reads all four horizons.\n")
    L += _horizon_doc_section()
    L.append(
        f"- **Window:** {meta['start']} → {meta['end']} ({meta['freq']}), "
        f"{meta['n_rebalances']} rebalances, {meta['n_universe']} names.\n"
        "> ⚠ Metrics are full-sample (no held-out tail); this selector trades the OOS "
        "check for transparency, so treat marginal IC gaps between subs with care.\n")

    L.append("## Selected construction per parent\n")
    rows = [{"parent": r.parent, "n_subs": len(r.selected), "formula": r.formula,
             "selected": ", ".join(r.selected), "flag": r.signal_flag} for r in results]
    L.append(_md_table(pd.DataFrame(rows),
                       ["parent", "n_subs", "formula", "selected", "flag"]))

    L += _audit_section(results)
    for res in results:
        L += _parent_section(res)
    (out_dir / "REPORT.md").write_text("\n".join(L))


def _audit_section(results: list[ParentSelection]) -> list[str]:
    """Per-horizon IC audit for the sparse-signal buckets (insider, institutional):
    where a single-horizon read and the multi-horizon mean disagree, and where the
    declared score direction is likely inverted."""
    audited = [r for r in results if r.parent in AUDIT_PARENTS]
    if not audited:
        return []
    L = ["\n## Horizon audit — insider & institutional\n",
         "Why the 3M grouped scorecard and the selector could disagree: the scorecard "
         "displays **3M** IC, the selector ranks on the **mean of 3M and 6M**. Both use "
         "the same panel and the same `HORIZON_MONTHS` = {1M,3M,6M,12M}; the 1M and 12M "
         "columns are shown but do not enter the pick. Every sub below is declared "
         "`higher_is_better=True` (buying / accumulation assumed bullish).\n"]
    for res in audited:
        df = res.ranking.copy()
        df["higher_is_better"] = df["sub_factor"].map(AUDITED_DIRECTION)
        L.append(f"\n**{res.parent}:**\n")
        L.append(_md_table(df, [c for c in AUDIT_COLS if c in df.columns]))
        inv = df[df["horizon_flag"] == "POSSIBLE_SIGN_INVERSION"]["sub_factor"].tolist()
        hs = df[df["horizon_flag"] == "HORIZON_SPECIFIC"]["sub_factor"].tolist()
        if inv:
            L.append(f"\n- 🚩 **Sign-inversion suspects** (negative at ~every horizon, "
                     f"declared higher-is-better): {', '.join(f'`{s}`' for s in inv)}.")
        if hs:
            L.append(f"- **Horizon-specific** (positive at some horizon, negative on the "
                     f"multi-horizon mean): {', '.join(f'`{s}`' for s in hs)}.")
        L.append("")
    return L
