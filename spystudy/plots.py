"""Charts for the SPY signal study.

Palette and mark rules follow the house data-viz reference: two categorical
slots (blue = signal ON, green = signal OFF, validated for CVD separation),
hairline solid grid, thin marks, legend always present, no dual axes.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.use("Agg")

ON = "#2a78d6"       # categorical slot 1
OFF = "#008300"      # categorical slot 2
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "text.color": INK,
    "axes.labelcolor": INK2,
    "axes.edgecolor": BASELINE,
    "axes.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "grid.linestyle": "-",
})


def _clean(ax, xgrid=False, ygrid=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.set_axisbelow(True)
    ax.grid(axis="y" if ygrid else "x", visible=ygrid or xgrid)
    if xgrid:
        ax.grid(axis="x", visible=True)
    if not ygrid:
        ax.grid(axis="y", visible=False)


def _pct(x, _=None):
    return f"{x * 100:.0f}%"


def _pp(x, _=None):
    """Percentage points, 1dp — 0dp collapses narrow ranges into duplicate ticks."""
    return f"{x * 100:+.1f}pp".replace("-", "−")


def effect_summary(res: pd.DataFrame, out: Path, title: str):
    """Headline: mean forward-return difference (ON − OFF) with bootstrap CI."""
    d = res.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    y = np.arange(len(d))
    sig = (d["boot_lo"] > 0) | (d["boot_hi"] < 0)
    colors = [ON if s else MUTED for s in sig]

    ax.axvline(0, color=BASELINE, lw=1.0, zorder=1)
    for i, row in d.iterrows():
        ax.plot([row["boot_lo"], row["boot_hi"]], [y[i], y[i]],
                color=colors[i], lw=2.0, solid_capstyle="round", zorder=2)
    ax.scatter(d["mean_diff"], y, s=46, color=colors, zorder=3,
               edgecolor=SURFACE, linewidth=2)

    for i, row in d.iterrows():
        txt = f"{row['mean_diff'] * 100:+.2f}pp   n={int(row['on_n'])}"
        ax.annotate(txt.replace("-", "−"),
                    (max(row["boot_hi"], row["mean_diff"]) + 0.0016, y[i]),
                    va="center", ha="left", fontsize=8, color=INK2)

    ax.set_yticks(y)
    ax.set_yticklabels(d["on_label"], fontsize=9, color=INK)
    ax.set_xlabel("Next-month total return, signal ON minus signal OFF")
    ax.xaxis.set_major_formatter(_pp)
    ax.set_xlim(d["boot_lo"].min() - 0.004, d["boot_hi"].max() + 0.022)
    _clean(ax, xgrid=True, ygrid=False)
    ax.set_title(title, loc="left", pad=14, color=INK, fontweight="bold")
    fig.text(0.012, 0.022,
             "Blue = 95% bootstrap CI excludes zero. Intervals are circular "
             "block-bootstrap (8-week blocks), which account for the "
             "overlapping forward windows.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(out, dpi=170)
    plt.close(fig)


def box_grid(weekly: pd.DataFrame, masks: dict, res: pd.DataFrame, out: Path,
             title: str):
    """Paired box plots, signal ON vs OFF, for every bucket."""
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    fwd = weekly["fwd_1m"]
    labels, ticks = [], []
    for i, (_, row) in enumerate(res.iterrows()):
        m = masks[row["signal"]]
        ok = (m.notna() & fwd.notna()).to_numpy()
        flag = m.to_numpy(dtype=float) > 0
        y = fwd.to_numpy()
        on, offv = y[ok & flag], y[ok & ~flag]
        for j, (vals, c) in enumerate(((on, ON), (offv, OFF))):
            bp = ax.boxplot([vals], positions=[i * 3 + j], widths=0.85,
                            showfliers=False, patch_artist=True,
                            medianprops=dict(color=SURFACE, lw=1.6),
                            whiskerprops=dict(color=c, lw=1.0),
                            capprops=dict(color=c, lw=1.0),
                            boxprops=dict(facecolor=c, edgecolor=c, lw=0))
            for b in bp["boxes"]:
                b.set_alpha(0.85)
        ax.scatter([i * 3, i * 3 + 1], [on.mean(), offv.mean()],
                   marker="D", s=16, color=SURFACE, edgecolor=INK, lw=0.9,
                   zorder=4)
        ticks.append(i * 3 + 0.5)
        labels.append(row["short"])

    ax.axhline(0, color=BASELINE, lw=1.0)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=8, color=INK)
    ax.set_ylabel("Next-month total return")
    ax.yaxis.set_major_formatter(_pct)
    _clean(ax)
    handles = [plt.Line2D([], [], color=ON, lw=6, label="Signal ON"),
               plt.Line2D([], [], color=OFF, lw=6, label="Signal OFF"),
               plt.Line2D([], [], marker="D", color=INK, lw=0, mfc=SURFACE,
                          ms=5, label="mean")]
    ax.legend(handles=handles, ncol=3, loc="upper right")
    ax.set_title(title, loc="left", pad=12, color=INK, fontweight="bold")
    ax.annotate("Boxes span the interquartile range; whiskers 1.5×IQR; "
                "outliers omitted. White line = median.",
                (0, -0.16), xycoords="axes fraction", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def hist_grid(weekly: pd.DataFrame, masks: dict, res: pd.DataFrame, out: Path,
              title: str):
    """Overlapping densities of next-month return, ON vs OFF, per bucket."""
    n = len(res)
    ncol, nrow = 4, int(np.ceil(n / 4))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.9 * nrow), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    fwd = weekly["fwd_1m"]
    bins = np.linspace(-0.16, 0.16, 33)

    for k, (_, row) in enumerate(res.iterrows()):
        ax = axes[k]
        m = masks[row["signal"]]
        ok = (m.notna() & fwd.notna()).to_numpy()
        flag = m.to_numpy(dtype=float) > 0
        y = fwd.to_numpy()
        on, offv = y[ok & flag], y[ok & ~flag]
        ax.hist(offv, bins=bins, density=True, color=OFF, alpha=0.45, lw=0)
        ax.hist(on, bins=bins, density=True, color=ON, alpha=0.55, lw=0)
        ax.axvline(np.mean(offv), color=OFF, lw=1.6)
        ax.axvline(np.mean(on), color=ON, lw=1.6)
        ax.set_title(f"{row['on_label']}\n"
                     f"ON   n={int(row['on_n'])} · mean {row['on_mean'] * 100:+.2f}%\n"
                     f"OFF  n={int(row['off_n'])} · mean {row['off_mean'] * 100:+.2f}%",
                     loc="left", fontsize=8.2, color=INK)
        ax.xaxis.set_major_formatter(_pct)
        ax.set_yticks([])
        _clean(ax, ygrid=False)
        ax.spines["left"].set_visible(False)

    for ax in axes[n:]:
        ax.set_visible(False)
    # sharex hides tick labels on all but the last row; the last row is short,
    # so re-expose them on whichever axis is last in each column
    for col in range(ncol):
        last = max(i for i in range(col, n, ncol)) if col < n else None
        if last is not None:
            axes[last].tick_params(labelbottom=True)
    handles = [plt.Line2D([], [], color=ON, lw=6, label="Signal ON"),
               plt.Line2D([], [], color=OFF, lw=6, label="Signal OFF"),
               plt.Line2D([], [], color=INK, lw=1.6, label="group mean")]
    axes[0].legend(handles=handles, ncol=1, loc="upper left", fontsize=7.6)
    fig.suptitle(title, x=0.008, ha="left", fontsize=12, fontweight="bold",
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out, dpi=170)
    plt.close(fig)


def mean_median_bars(res: pd.DataFrame, out: Path, title: str):
    """Mean and median next-month return side by side, ON vs OFF."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    x = np.arange(len(res))
    for ax, stat in zip(axes, ("mean", "median")):
        ax.bar(x - 0.21, res[f"on_{stat}"], width=0.40, color=ON,
               label="Signal ON")
        ax.bar(x + 0.21, res[f"off_{stat}"], width=0.40, color=OFF,
               label="Signal OFF")
        ax.axhline(0, color=BASELINE, lw=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(res["short"], fontsize=7.8, color=INK)
        ax.set_title(f"{stat.capitalize()} next-month total return", loc="left",
                     color=INK)
        ax.yaxis.set_major_formatter(lambda v, p: f"{v * 100:.1f}%")
        _clean(ax)
    axes[0].legend(ncol=2, loc="upper right")
    fig.suptitle(title, x=0.006, ha="left", fontsize=12, fontweight="bold",
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=170)
    plt.close(fig)


def quintile_grid(weekly: pd.DataFrame, cols: list[tuple[str, str]], out: Path,
                  title: str):
    """Mean next-month return by quintile of each continuous signal.

    A threshold test only sees one cut point; this shows whether the relation is
    monotone or whether the chosen threshold sits on noise.
    """
    fwd = weekly["fwd_1m"]
    ncol = 3
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.5 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for k, (col, label) in enumerate(cols):
        ax = axes[k]
        ok = weekly[col].notna() & fwd.notna()
        q = pd.qcut(weekly.loc[ok, col], 5, labels=[f"Q{i}" for i in range(1, 6)])
        g = fwd[ok].groupby(q, observed=True)
        mean, med, n = g.mean(), g.median(), g.size()
        ax.bar(np.arange(5) - 0.20, mean.values, width=0.38, color=ON,
               label="mean")
        ax.bar(np.arange(5) + 0.20, med.values, width=0.38, color=OFF,
               label="median")
        ax.axhline(0, color=BASELINE, lw=1.0)
        ax.set_xticks(range(5))
        ax.set_xticklabels([f"{lbl}\nn={v}" for lbl, v in zip(mean.index, n.values)],
                           fontsize=7.8, color=INK)
        ax.set_title(f"{label}   (Q1 = lowest)", loc="left", fontsize=9,
                     color=INK)
        ax.yaxis.set_major_formatter(lambda v, p: f"{v * 100:.1f}%")
        ax.margins(y=0.18)
        _clean(ax)
    for ax in axes[len(cols):]:
        ax.set_visible(False)
    axes[0].legend(ncol=2, loc="upper right")
    fig.suptitle(title, x=0.006, ha="left", fontsize=12, fontweight="bold",
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=170)
    plt.close(fig)


def rank_ic_chart(full: pd.DataFrame, eras: pd.DataFrame, out: Path, title: str):
    """Spearman IC over the whole sample, and split by era."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    d = full.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(d))
    sig = (d["lo"] > 0) | (d["hi"] < 0)
    colors = [ON if s else MUTED for s in sig]
    ax = axes[0]
    ax.axvline(0, color=BASELINE, lw=1.0)
    for i, row in d.iterrows():
        ax.plot([row["lo"], row["hi"]], [y[i], y[i]], color=colors[i], lw=2.0,
                solid_capstyle="round")
    ax.scatter(d["rho"], y, s=44, color=colors, edgecolor=SURFACE, lw=2, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"], fontsize=8.8, color=INK)
    ax.set_xlabel("Spearman rank IC vs next-month return")
    ax.set_title("Full sample, 2014–2026", loc="left", color=INK)
    _clean(ax, xgrid=True, ygrid=False)

    ax = axes[1]
    e = eras.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(e))
    ax.axvline(0, color=BASELINE, lw=1.0)
    ax.scatter(e["rho_early"], y + 0.16, s=44, color=ON, edgecolor=SURFACE,
               lw=1.6, label="2014 – mid-2020", zorder=3)
    ax.scatter(e["rho_late"], y - 0.16, s=44, color=OFF, edgecolor=SURFACE,
               lw=1.6, label="mid-2020 – 2026", zorder=3)
    for i, row in e.iterrows():
        ax.plot([row["rho_early"], row["rho_late"]], [y[i] + 0.16, y[i] - 0.16],
                color=BASELINE, lw=1.0, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(e["label"], fontsize=8.8, color=INK)
    ax.set_xlabel("Spearman rank IC, by era")
    ax.set_title("Does it hold in both halves?", loc="left", color=INK)
    ax.set_ylim(-1.25, len(e) - 0.45)  # clear a band under the last row
    ax.legend(loc="lower right", ncol=2)
    _clean(ax, xgrid=True, ygrid=False)

    fig.suptitle(title, x=0.006, ha="left", fontsize=12, fontweight="bold",
                 color=INK)
    fig.text(0.006, 0.02, "Blue = 95% block-bootstrap CI excludes zero. A signal "
             "whose two era dots sit on opposite sides of zero did not hold up.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(out, dpi=170)
    plt.close(fig)


def redundancy(corr: pd.DataFrame, out: Path, title: str):
    """Diverging heatmap of the signal-vs-signal rank correlations."""
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "div", ["#e34948", "#f0efec", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    im = ax.imshow(corr.to_numpy(), cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=32, ha="right", fontsize=8.5,
                       color=INK)
    ax.set_yticklabels(corr.index, fontsize=8.5, color=INK)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}".replace("-", "−"), ha="center", va="center",
                    fontsize=8.5, color=INK if abs(v) < 0.55 else SURFACE)
    ax.set_xticks(np.arange(len(corr) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(corr) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.03)
    cb.set_label("Spearman correlation", fontsize=8.5, color=INK2)
    cb.outline.set_visible(False)
    ax.set_title(title, loc="left", pad=12, color=INK, fontweight="bold",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def sector_attribution(by_sector: pd.DataFrame, loo: pd.DataFrame,
                       low: pd.Series, names: dict, out: Path, title: str):
    """Which sectors sit lowest, and which one actually drags the aggregate."""
    lowm = low.reindex(by_sector.index).fillna(False).to_numpy().astype(bool)
    tab = pd.DataFrame({
        "low": by_sector[lowm].mean(),
        "high": by_sector[~lowm].mean(),
        "loo": loo[lowm].mean(),
    }).sort_values("low")
    y = np.arange(len(tab))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    ax = axes[0]
    ax.barh(y + 0.20, tab["low"], height=0.38, color=ON, label="corr < 0.45")
    ax.barh(y - 0.20, tab["high"], height=0.38, color=OFF, label="corr ≥ 0.45")
    ax.set_yticks(y)
    ax.set_yticklabels([names.get(t, t) for t in tab.index], fontsize=8.5,
                       color=INK)
    ax.set_xlabel("Mean correlation of this sector to the other sectors,\n"
                  "in market-wide low- vs high-correlation weeks")
    ax.set_title("Every sector de-correlates together", loc="left", color=INK)
    ax.set_xlim(0, tab["high"].max() * 1.30)  # clear a column for the legend
    ax.legend(loc="lower right", title=None)
    _clean(ax, xgrid=True, ygrid=False)

    ax = axes[1]
    # one series: the side of zero already carries the sign, so one hue
    ax.barh(y, tab["loo"] * 100, height=0.62, color=ON)
    ax.axvline(0, color=BASELINE, lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([names.get(t, t) for t in tab.index], fontsize=8.5,
                       color=INK)
    ax.set_xlabel("Change in market-wide correlation if this sector is dropped (pp)")
    ax.set_title("Utilities and Energy sit apart — but always, not just in "
                 "low-corr weeks", loc="left", fontsize=9.5, color=INK)
    _clean(ax, xgrid=True, ygrid=False)

    fig.suptitle(title, x=0.006, ha="left", fontsize=12, fontweight="bold",
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=170)
    plt.close(fig)


def sector_effect(tab: pd.DataFrame, names: dict, out: Path, title: str):
    """Per-sector: SPY next-month return when THAT sector decouples."""
    d = tab.sort_values("diff").reset_index(drop=True)
    y = np.arange(len(d))[::-1]
    sig = (d["boot_lo"] > 0) | (d["boot_hi"] < 0)
    colors = [ON if s else MUTED for s in sig]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.axvline(0, color=BASELINE, lw=1.0)
    for i, row in d.iterrows():
        ax.plot([row["boot_lo"], row["boot_hi"]], [y[i], y[i]], color=colors[i],
                lw=2.0, solid_capstyle="round")
    ax.scatter(d["diff"], y, s=44, color=colors, edgecolor=SURFACE, linewidth=2,
               zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([names.get(t, t) for t in d["sector"]], fontsize=9,
                       color=INK)
    ax.set_xlabel("SPY next-month return when this sector is in its bottom "
                  "tercile of correlation-to-peers, minus the rest")
    ax.xaxis.set_major_formatter(_pp)
    _clean(ax, xgrid=True, ygrid=False)
    ax.set_title(title, loc="left", pad=12, color=INK, fontweight="bold")
    fig.text(0.012, 0.025,
             "Blue = 95% bootstrap CI excludes zero. Every sector leans the "
             "same way, which is what a broad regime effect looks like.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=170)
    plt.close(fig)
