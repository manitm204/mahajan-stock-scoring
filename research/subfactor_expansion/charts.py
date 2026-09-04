"""Analysis charts for the expanded 76-candidate library.

Three deliverables:

* ``ic_by_subfactor.png``      — horizontal bar of mean IC per candidate,
                                 coloured by parent.
* ``scorecard.png``            — two-panel: mean IC + Q5-Q1 spread.
* ``correlation_heatmap.png``  — pairwise cross-sectional Spearman corr,
                                 ordered by parent so clusters are visible.

Reads only from ``CandidatePanel`` (cached) + the validation ``summary_<h>.csv``.
Kept separate from :mod:`research.subfactor_expansion.report` because charting
pulls in matplotlib and we want the markdown path to stay lightweight.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .library import iter_parents
from .panel import CandidatePanel


_PARENT_ORDER = ["momentum", "value", "quality", "growth", "revisions",
                 "short", "insider", "institutional"]


def _parent_colors(parents: list[str]) -> dict[str, tuple]:
    uniq = [p for p in _PARENT_ORDER if p in parents] + \
           [p for p in sorted(set(parents)) if p not in _PARENT_ORDER]
    return {p: c for p, c in zip(uniq, plt.cm.tab10(np.linspace(0, 1, max(len(uniq), 1))))}


# Primary IC metric: the 3M/6M mean when the validation summary carries it (the metric
# the parent selector picks on), else the legacy focus-horizon mean_ic. Charts read the
# ``mean_ic`` column throughout, so ``write_all`` copies the chosen metric into it and
# passes a truthful axis label via ``ic_label``.
def _resolve_ic(summary: pd.DataFrame, horizon: str) -> tuple[pd.DataFrame, str]:
    df = summary.copy()
    if "mean_ic_3m6m" in df.columns and df["mean_ic_3m6m"].notna().any():
        df["mean_ic"] = df["mean_ic_3m6m"]
        return df, "3M/6M mean"
    return df, horizon


def _style_selection(ax, order: list[str], bars, selected: set[str] | None) -> None:
    """Dim the bars/labels of *rejected* candidates and star + bold the *selected* ones.

    ``order`` is the candidate name per bar (bottom-to-top). No-op when ``selected`` is
    None so the plain (selection-agnostic) charts are unchanged."""
    if selected is None:
        return
    for bar, name in zip(bars, order):
        bar.set_alpha(1.0 if name in selected else 0.32)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([("★ " if n in selected else "   ") + n for n in order],
                       fontsize=7)
    for tick, name in zip(ax.get_yticklabels(), order):
        if name in selected:
            tick.set_fontweight("bold")
            tick.set_color("black")
        else:
            tick.set_color("0.55")


def chart_ic_by_subfactor(summary: pd.DataFrame, path: Path, horizon: str,
                          ic_label: str | None = None) -> Path:
    """Horizontal bar of mean IC per candidate, coloured by parent."""
    df = summary.dropna(subset=["mean_ic"]).sort_values("mean_ic")
    if df.empty:
        return path
    colors = _parent_colors(list(df["parent"]))
    fig, ax = plt.subplots(figsize=(11, max(6, 0.28 * len(df))))
    ax.barh(df["candidate"], df["mean_ic"],
            color=[colors[p] for p in df["parent"]])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(f"Mean Spearman IC per candidate ({ic_label or horizon}) — "
                 f"{len(df)}-candidate library")
    ax.set_xlabel("Mean IC")
    ax.grid(True, axis="x", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[p]) for p in colors]
    ax.legend(handles, list(colors), loc="lower right", fontsize=8, title="parent")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def chart_ic_by_parent_group(summary: pd.DataFrame, path: Path,
                             horizon: str, ic_label: str | None = None,
                             selected: set[str] | None = None) -> Path:
    """Grouped bar chart: candidates stacked by parent, one colour per parent,
    within-parent order sorted by IC. Right margin lists per-parent positive
    vs negative counts so you can see which buckets are working. When ``selected``
    is given, the parent-selected candidates are starred/bold and the rest dimmed."""
    df = summary.dropna(subset=["mean_ic"]).copy()
    if df.empty:
        return path

    colors = _parent_colors(list(df["parent"]))
    ordered_parents = [p for p in _PARENT_ORDER if p in colors] + \
                      [p for p in colors if p not in _PARENT_ORDER]

    rows: list[pd.Series] = []
    parent_stats: list[tuple[str, int, int, float]] = []  # parent, +, -, mean
    for parent in ordered_parents:
        sub = df[df["parent"] == parent].sort_values("mean_ic", ascending=True)
        rows.append(sub)
        n_pos = int((sub["mean_ic"] > 0).sum())
        n_neg = int((sub["mean_ic"] < 0).sum())
        parent_stats.append((parent, n_pos, n_neg, float(sub["mean_ic"].mean())))
    plot_df = pd.concat(rows, ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, max(6, 0.28 * len(plot_df))))
    bars = ax.barh(plot_df["candidate"], plot_df["mean_ic"],
                   color=[colors[p] for p in plot_df["parent"]],
                   edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="black", lw=0.8)
    _style_selection(ax, list(plot_df["candidate"]), bars, selected)

    # Divider lines between parent blocks.
    cursor = 0
    counts_by_parent: dict[str, tuple[int, int, int, float]] = {}
    for parent, n_pos, n_neg, mean_ic in parent_stats:
        n = int((plot_df["parent"] == parent).sum())
        if n == 0:
            continue
        if cursor > 0:
            ax.axhline(cursor - 0.5, color="black", lw=0.6, alpha=0.6)
        counts_by_parent[parent] = (cursor, cursor + n - 1, n_pos, n_neg,
                                    mean_ic)
        cursor += n

    sel_note = " — ★ bold = selected for parent, dimmed = rejected" if selected else ""
    ax.set_title(
        f"Mean Spearman IC per candidate ({ic_label or horizon}), grouped by parent — "
        f"one colour per bucket{sel_note}")
    ax.set_xlabel("Mean IC")
    ax.grid(True, axis="x", alpha=0.3)
    ax.margins(y=0.01)

    # Reserve room on the right so the annotations don't get clipped.
    x_min, x_max = ax.get_xlim()
    ax.set_xlim(x_min, x_max + 0.55 * (x_max - x_min))

    # Parent header + counts on the right, outside the bar area.
    label_x = x_max + 0.03 * (x_max - x_min)
    for parent, (bot, top, n_pos, n_neg, mean_ic) in counts_by_parent.items():
        mid = (bot + top) / 2
        ax.text(label_x, mid,
                f"{parent}\n+{n_pos} / −{n_neg}\navg {mean_ic:+.3f}",
                ha="left", va="center", fontsize=10, fontweight="bold",
                color=colors[parent])

    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_ic_and_r2_by_parent(
    summary: pd.DataFrame,
    corr: pd.DataFrame,
    panel: CandidatePanel,
    path: Path,
    horizon: str,
    ic_threshold: float = 0.01,
    ic_label: str | None = None,
    selected: set[str] | None = None,
) -> Path:
    """Two-panel grouped chart.

    * Left  — mean IC per candidate (green dashed at the threshold).
    * Right — R² of each candidate vs. the top-IC candidate in the same parent
              (a diversification lens — high R² means "redundant with the leader").

    Parent header annotations include ``N with IC>threshold`` so it is easy to
    see which buckets carry real signal beyond noise.
    """
    df = summary.dropna(subset=["mean_ic"]).copy()
    if df.empty:
        return path

    colors = _parent_colors(list(df["parent"]))
    ordered_parents = [p for p in _PARENT_ORDER if p in colors] + \
                      [p for p in colors if p not in _PARENT_ORDER]

    # Pick each parent's top-IC candidate (breaks ties by name for stability).
    top_by_parent: dict[str, str] = {}
    for parent in ordered_parents:
        sub = df[df["parent"] == parent].sort_values(
            ["mean_ic", "candidate"], ascending=[False, True])
        if not sub.empty:
            top_by_parent[parent] = sub.iloc[0]["candidate"]

    def _r2(cand: str, parent: str) -> float:
        top = top_by_parent.get(parent)
        if top is None or top not in corr.columns or cand not in corr.columns:
            return float("nan")
        c = corr.loc[cand, top]
        return float(c ** 2) if pd.notna(c) else float("nan")

    df["r2_vs_top"] = [
        _r2(c, p) for c, p in zip(df["candidate"], df["parent"])
    ]
    df["is_top"] = [
        c == top_by_parent.get(p) for c, p in zip(df["candidate"], df["parent"])
    ]

    # Within each parent, sort ascending by IC so the leader sits at the top.
    rows: list[pd.DataFrame] = []
    parent_stats: list[tuple[str, int, int, int, float]] = []
    for parent in ordered_parents:
        sub = df[df["parent"] == parent].sort_values("mean_ic", ascending=True)
        if sub.empty:
            continue
        rows.append(sub)
        n_strong = int((sub["mean_ic"] > ic_threshold).sum())
        n_pos = int((sub["mean_ic"] > 0).sum())
        parent_stats.append(
            (parent, n_strong, n_pos, len(sub), float(sub["mean_ic"].mean()))
        )
    plot_df = pd.concat(rows, ignore_index=True)

    fig, (ax_ic, ax_r2) = plt.subplots(
        1, 2, figsize=(17, max(6, 0.28 * len(plot_df))), sharey=True,
        gridspec_kw={"width_ratios": [1.4, 1.0]},
    )

    bar_colors = [colors[p] for p in plot_df["parent"]]
    edge = ["black" if t else "none" for t in plot_df["is_top"]]
    order = list(plot_df["candidate"])

    bars_ic = ax_ic.barh(plot_df["candidate"], plot_df["mean_ic"], color=bar_colors,
                         edgecolor=edge, linewidth=1.1)
    ax_ic.axvline(0, color="black", lw=0.8)
    ax_ic.axvline(ic_threshold, color="green", lw=0.8, ls="--", alpha=0.7,
                  label=f"IC = {ic_threshold:+.2f}")
    leader_note = "black outline = parent leader"
    sel_note = "; ★ bold = selected, dimmed = rejected" if selected else ""
    ax_ic.set_title(f"Mean Spearman IC ({ic_label or horizon}) — {leader_note}{sel_note}")
    ax_ic.set_xlabel("Mean IC")
    ax_ic.grid(True, axis="x", alpha=0.3)
    ax_ic.legend(loc="lower right", fontsize=8)

    bars_r2 = ax_r2.barh(plot_df["candidate"], plot_df["r2_vs_top"].fillna(0.0),
                        color=bar_colors, edgecolor=edge, linewidth=1.1)
    _style_selection(ax_ic, order, bars_ic, selected)
    if selected is not None:                     # dim rejected bars on the R² panel too
        for bar, name in zip(bars_r2, order):
            bar.set_alpha(1.0 if name in selected else 0.32)
    ax_r2.set_title("R² vs top-IC candidate in same parent (0 = orthogonal, 1 = duplicate)")
    ax_r2.set_xlabel("R² (Spearman corr²)")
    ax_r2.set_xlim(0, 1.05)
    ax_r2.axvline(0.5, color="black", lw=0.6, ls=":", alpha=0.5)
    ax_r2.grid(True, axis="x", alpha=0.3)

    # Divider lines between parent blocks (both axes).
    cursor = 0
    counts_by_parent: dict[str, tuple[int, int, int, int, int, float]] = {}
    for parent, n_strong, n_pos, n, mean_ic in parent_stats:
        if cursor > 0:
            for ax in (ax_ic, ax_r2):
                ax.axhline(cursor - 0.5, color="black", lw=0.6, alpha=0.6)
        counts_by_parent[parent] = (cursor, cursor + n - 1, n_strong, n_pos,
                                    n, mean_ic)
        cursor += n

    # Reserve right-side room and stamp per-parent labels on the R² axis.
    x_min, x_max = ax_r2.get_xlim()
    ax_r2.set_xlim(x_min, x_max + 0.60 * (x_max - x_min))
    label_x = x_max + 0.05 * (x_max - x_min)
    for parent, (bot, top, n_strong, n_pos, n, mean_ic) in counts_by_parent.items():
        mid = (bot + top) / 2
        ax_r2.text(
            label_x, mid,
            f"{parent}\n{n_strong}/{n} with IC>{ic_threshold:+.2f}\n"
            f"+{n_pos} / −{n - n_pos}, avg {mean_ic:+.3f}",
            ha="left", va="center", fontsize=9, fontweight="bold",
            color=colors[parent],
        )

    fig.suptitle(
        f"Grouped subfactor scorecard — IC and diversification vs. parent leader "
        f"(horizon {horizon})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_scorecard(summary: pd.DataFrame, path: Path, horizon: str,
                    ic_label: str | None = None) -> Path:
    """Two panels: mean IC + Q5-Q1 spread per candidate. The IC panel uses the 3M/6M
    mean when available (``ic_label``); the spread stays at the focus horizon."""
    df = summary.dropna(subset=["mean_ic"]).sort_values("mean_ic")
    if df.empty:
        return path
    colors = _parent_colors(list(df["parent"]))
    fig, axes = plt.subplots(1, 2, figsize=(15, max(6, 0.28 * len(df))), sharey=True)
    panels = [("mean_ic", f"Mean Spearman IC ({ic_label or horizon})"),
              ("spread_q5_q1", f"Q5-Q1 spread ({horizon}, avg period)")]
    for ax, (col, title) in zip(axes, panels):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        ax.barh(df["candidate"], df[col].fillna(0.0),
                color=[colors[p] for p in df["parent"]])
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(title)
        ax.grid(True, axis="x", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[p]) for p in colors]
    axes[1].legend(handles, list(colors), loc="lower right", fontsize=8, title="parent")
    fig.suptitle(f"Subfactor scorecard — {len(df)}-candidate library",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def correlation_matrix(panel: CandidatePanel, min_names: int = 40) -> pd.DataFrame:
    """Per-date Spearman corr among all candidates, then averaged across dates."""
    subs = panel.all_candidates
    mats: list[np.ndarray] = []
    for d in panel.rebal_dates:
        frame = panel.scores.get(d)
        if frame is None or frame.empty:
            continue
        usable = [c for c in subs if c in frame.columns and frame[c].nunique() >= 2]
        if len(usable) < 2:
            continue
        sub_frame = frame[usable]
        if sub_frame.dropna(how="any").shape[0] < min_names:
            # Use pairwise-available observations rather than requiring all cols dense.
            pass
        m = sub_frame.corr(method="spearman").reindex(index=subs, columns=subs)
        mats.append(m.to_numpy(dtype=float))
    if not mats:
        return pd.DataFrame(index=subs, columns=subs, dtype=float)
    avg = np.nanmean(np.stack(mats), axis=0)
    return pd.DataFrame(avg, index=subs, columns=subs)


def _order_by_parent(panel: CandidatePanel) -> list[str]:
    order: list[str] = []
    for parent in iter_parents():
        order.extend(panel.candidates_by_parent.get(parent, []))
    return order


def chart_correlation(corr: pd.DataFrame, panel: CandidatePanel, path: Path) -> Path:
    """Correlation heatmap, candidates grouped by parent with visible dividers."""
    order = [s for s in _order_by_parent(panel) if s in corr.columns]
    if len(order) < 2:
        return path
    data = corr.reindex(index=order, columns=order).to_numpy(dtype=float)
    n = len(order)
    fig, ax = plt.subplots(figsize=(0.24 * n + 3, 0.24 * n + 3))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1)

    parents_by_pos = [panel.parent_of(s) for s in order]
    boundaries: list[int] = []
    prev = None
    for i, p in enumerate(parents_by_pos):
        if p != prev:
            boundaries.append(i)
            prev = p
    for b in boundaries[1:]:
        ax.axhline(b - 0.5, color="black", lw=0.6)
        ax.axvline(b - 0.5, color="black", lw=0.6)

    # Parent labels centered on each block.
    block_edges = boundaries + [n]
    for i in range(len(boundaries)):
        mid = (block_edges[i] + block_edges[i + 1] - 1) / 2
        ax.text(mid, -1.2, parents_by_pos[block_edges[i]], ha="center",
                va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(range(n))
    ax.set_xticklabels(order, rotation=90, fontsize=6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(order, fontsize=6)
    cbar = plt.colorbar(im, ax=ax, fraction=0.025)
    cbar.set_label("avg cross-sectional Spearman corr")
    ax.set_title("Candidate correlation — parent blocks separated by black lines")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def redundant_pairs(corr: pd.DataFrame, panel: CandidatePanel,
                    threshold: float = 0.85) -> pd.DataFrame:
    """Sub-factor pairs with |avg cross-sectional corr| above ``threshold``."""
    subs = list(corr.columns)
    rows: list[dict] = []
    for i, a in enumerate(subs):
        for b in subs[i + 1:]:
            c = corr.loc[a, b]
            if pd.notna(c) and abs(c) >= threshold:
                rows.append({
                    "a": a, "b": b,
                    "parent_a": panel.parent_of(a),
                    "parent_b": panel.parent_of(b),
                    "corr": float(c),
                })
    if not rows:
        return pd.DataFrame(columns=["a", "b", "parent_a", "parent_b", "corr"])
    df = pd.DataFrame(rows)
    return df.sort_values("corr", key=lambda s: s.abs(),
                          ascending=False).reset_index(drop=True)


def write_all(
    panel: CandidatePanel,
    summary: pd.DataFrame,
    out_dir: Path,
    horizon: str = "3M",
    corr_threshold: float = 0.85,
    selected: set[str] | None = None,
) -> dict[str, Path]:
    """Emit ic_by_subfactor.png, scorecard.png, correlation_heatmap.png +
    the redundant-pairs CSV. Returns the paths written.

    Charts plot the **3M/6M mean IC** whenever the summary carries ``mean_ic_3m6m``
    (the metric the parent selector picks on); otherwise the focus-horizon ``mean_ic``.
    Pass ``selected`` (the union of parent-selected candidate names, e.g. from
    ``run_parent_selection.py --source expansion``) to star/bold selected subfactors and
    dim the rejected ones in the grouped charts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plot, ic_label = _resolve_ic(summary, horizon)
    ic_path = chart_ic_by_subfactor(plot, out_dir / "ic_by_subfactor.png", horizon, ic_label)
    grp_path = chart_ic_by_parent_group(
        plot, out_dir / "ic_by_parent_group.png", horizon, ic_label, selected)
    sc_path = chart_scorecard(plot, out_dir / "scorecard.png", horizon, ic_label)
    corr = correlation_matrix(panel)
    corr.to_csv(out_dir / "correlation_matrix.csv")
    ic_r2_path = chart_ic_and_r2_by_parent(
        plot, corr, panel, out_dir / "ic_and_r2_by_parent.png", horizon,
        ic_label=ic_label, selected=selected)
    heat_path = chart_correlation(corr, panel, out_dir / "correlation_heatmap.png")
    pairs = redundant_pairs(corr, panel, corr_threshold)
    pairs.to_csv(out_dir / "redundant_pairs.csv", index=False)
    return {
        "ic_by_subfactor": ic_path,
        "ic_by_parent_group": grp_path,
        "ic_and_r2_by_parent": ic_r2_path,
        "scorecard": sc_path,
        "correlation_heatmap": heat_path,
        "correlation_matrix": out_dir / "correlation_matrix.csv",
        "redundant_pairs": out_dir / "redundant_pairs.csv",
    }
