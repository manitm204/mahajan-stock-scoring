"""Emit the human-readable REPORT.md for the subfactor expansion pipeline.

Consumes the validation frames produced by :mod:`validation` and writes a
single Markdown document to ``output/subfactor_expansion/REPORT.md``. Reports
are strictly derived — no scoring / IC logic lives here.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .library import iter_parents
from .panel import CandidatePanel


# Prior production counts (lean sub-factor set — output/factor_research/REPORT.md).
_OLD_COUNTS = {
    "momentum": 4, "value": 4, "quality": 2, "growth": 3,
    "revisions": 5, "institutional": 5, "insider": 3, "short": 3,
}


def write_report(
    panel: CandidatePanel,
    validation: dict[str, pd.DataFrame],
    out_path: Path,
    ic_horizon: str = "3M",
    new_endpoints: list[str] | None = None,
    backfill_status: dict[str, str] | None = None,
) -> Path:
    """Assemble the report and write it. Returns the output path."""
    coverage = validation["coverage"]
    ic = validation["ic"]
    summary = validation["summary"]
    sign = validation["sign"]

    lines: list[str] = []
    lines.append("# Subfactor Expansion — Validation Report")
    lines.append("")
    lines.append(f"Rebalance dates: **{len(panel.rebal_dates)}** "
                 f"({panel.rebal_dates[0]} → {panel.rebal_dates[-1]})")
    lines.append(f"Universe: **{len(panel.universe)}** tickers")
    lines.append(f"Focus IC horizon: **{ic_horizon}**")
    lines.append("")

    # ---- 1. Old vs new counts -------------------------------------------------
    lines.append("## 1. Subfactor counts by parent")
    lines.append("")
    lines.append("| Parent | Old (production lean) | New candidates | Δ |")
    lines.append("|---|---:|---:|---:|")
    for parent in iter_parents():
        new = len(panel.candidates_by_parent.get(parent, []))
        old = _OLD_COUNTS.get(parent, 0)
        lines.append(f"| {parent} | {old} | {new} | +{new - old} |")
    lines.append(f"| **total** | **{sum(_OLD_COUNTS.values())}** | "
                 f"**{sum(len(v) for v in panel.candidates_by_parent.values())}** | "
                 f"**+{sum(len(v) for v in panel.candidates_by_parent.values()) - sum(_OLD_COUNTS.values())}** |")
    lines.append("")

    # ---- 2. New endpoints + backfill status ----------------------------------
    lines.append("## 2. Raw-data endpoints added")
    if not new_endpoints:
        lines.append("- (none — every candidate is derivable from local tables)")
    else:
        for ep in new_endpoints:
            lines.append(f"- {ep}")
    if backfill_status:
        lines.append("")
        lines.append("### Backfill status")
        for table, status in backfill_status.items():
            lines.append(f"- **{table}**: {status}")
    lines.append("")

    # ---- 3. Top / bottom by IC at the focus horizon --------------------------
    ic_focus = ic[ic["horizon"] == ic_horizon].sort_values("mean_ic", ascending=False)
    lines.append(f"## 3. Top / bottom candidates by mean IC ({ic_horizon})")
    lines.append("")
    lines.append("### Top 15")
    lines.append(_ic_table(ic_focus.head(15)))
    lines.append("")
    lines.append("### Bottom 15")
    lines.append(_ic_table(ic_focus.tail(15).sort_values("mean_ic")))
    lines.append("")

    # ---- 4. Per-parent view ---------------------------------------------------
    lines.append(f"## 4. Per-parent candidates ({ic_horizon} IC)")
    lines.append("")
    for parent in iter_parents():
        sub = summary[summary["parent"] == parent].sort_values("mean_ic", ascending=False)
        lines.append(f"### {parent}")
        lines.append(_parent_table(sub))
        lines.append("")

    # ---- 5. Sign sanity flags -------------------------------------------------
    if not sign.empty:
        suspects = sign[
            (sign["sign_agrees"] == False)  # noqa: E712 - explicit bool
            & sign["raw_corr_mean"].notna()
        ].sort_values("raw_corr_mean")
        lines.append("## 5. Suspicious sign inversions")
        if suspects.empty:
            lines.append("None flagged — every candidate's raw ↔ forward-return "
                         "correlation sign matches its declared direction.")
        else:
            lines.append("Candidates where the raw correlation to forward returns "
                         f"disagrees with the declared direction (horizon {ic_horizon}).")
            lines.append("")
            lines.append("| Candidate | Parent | Declared | Raw corr | Hit rate |")
            lines.append("|---|---|---|---:|---:|")
            for _, r in suspects.iterrows():
                lines.append(
                    f"| {r['candidate']} | {r['parent']} | "
                    f"{'higher' if r['declared_higher_is_better'] else 'lower'} | "
                    f"{r['raw_corr_mean']:+.3f} | {r['raw_corr_hit_rate']:.0%} |")
        lines.append("")

    # ---- 6. Low-coverage flags -----------------------------------------------
    low_cov = coverage[coverage["coverage"] < 0.30].sort_values("coverage")
    lines.append("## 6. Low-coverage candidates (< 30%)")
    if low_cov.empty:
        lines.append("None — every candidate ranks at least 30% of the universe.")
    else:
        lines.append("| Candidate | Parent | Coverage | Neutral-50 share |")
        lines.append("|---|---|---:|---:|")
        for _, r in low_cov.iterrows():
            lines.append(f"| {r['candidate']} | {r['parent']} | "
                         f"{r['coverage']:.0%} | {r['neutral_50_share']:.0%} |")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    return out_path


def _ic_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no data)"
    rows = ["| Candidate | Parent | Mean IC | IR | Hit rate | Stability | N |",
            "|---|---|---:|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        rows.append(
            f"| {r['candidate']} | {r['parent']} | "
            f"{_fmt(r.get('mean_ic'))} | {_fmt(r.get('information_ratio'))} | "
            f"{_pct(r.get('hit_rate'))} | {_pct(r.get('stability'))} | "
            f"{int(r.get('n_periods') or 0)} |")
    return "\n".join(rows)


def _parent_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no candidates)"
    rows = ["| Candidate | Coverage | Mean IC | IR | Q5-Q1 | Monoton. | Year-stab |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        rows.append(
            f"| {r['candidate']} | {_pct(r.get('coverage'))} | "
            f"{_fmt(r.get('mean_ic'))} | {_fmt(r.get('information_ratio'))} | "
            f"{_fmt(r.get('spread_q5_q1'))} | {_fmt(r.get('monotonicity'))} | "
            f"{_pct(r.get('year_stability'))} |")
    return "\n".join(rows)


def _fmt(v) -> str:  # noqa: ANN001
    if v is None or (isinstance(v, float) and (pd.isna(v))):
        return "—"
    return f"{v:+.3f}" if isinstance(v, (int, float)) else str(v)


def _pct(v) -> str:  # noqa: ANN001
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{v:.0%}"
