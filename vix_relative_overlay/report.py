"""Aggregation tables, dashboard and report writing for the overlay study."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import HORIZON_MONTHS, benchmark_stats, perf_metrics
from .windows import BLOCKS, block_of

FOCUS_TOP = 0.20
FOCUS_MODE = "equal"


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _groupings(frame: pd.DataFrame, level: str) -> list[tuple[str, pd.DataFrame]]:
    if level == "full":
        return [("full", frame)]
    if level == "block":
        out = []
        for name, _, _ in BLOCKS:
            sub = frame[frame["window"].map(block_of) == name]
            if not sub.empty:
                out.append((name, sub))
        return out
    if level == "window":
        return [(w, g) for w, g in frame.groupby("window", sort=True)]
    raise ValueError(level)


def performance_table(sims: dict[tuple[str, float, str], pd.DataFrame],
                      level: str) -> pd.DataFrame:
    """Per (variant, top %, mode, grouping): the full metric set on net returns,
    gross CAGR alongside, benchmark stats vs SPY."""
    rows: list[dict] = []
    for (variant, top_pct, mode), df in sims.items():
        for name, sub in _groupings(df, level):
            net = sub.set_index("realize_date")["net"]
            gross = sub.set_index("realize_date")["gross"]
            spy = sub.set_index("realize_date")["spy"]
            turn = sub.set_index("realize_date")["turnover"]
            m = perf_metrics(net, turn)
            b = benchmark_stats(net, spy)
            rows.append({
                "variant": variant, "top_pct": top_pct, "mode": mode,
                "grouping": name, **m,
                "gross_cagr": perf_metrics(gross)["cagr"] if not gross.dropna().empty
                else np.nan,
                "spy_cagr": b["bench_cagr"], "spy_excess_cagr": b["excess_cagr"],
                "spy_alpha": b["alpha"], "spy_beta": b["beta"],
                "spy_ir": b["info_ratio"], "spy_te": b["tracking_error"],
                "spy_rel_max_drawdown": b["rel_max_drawdown"],
            })
    return pd.DataFrame(rows)


def ic_summary(ics: dict[str, pd.DataFrame], level: str) -> pd.DataFrame:
    """Per (variant, grouping, horizon): pooled composite IC + Q5-Q1 spread."""
    rows: list[dict] = []
    for variant, df in ics.items():
        for name, sub in _groupings(df, level):
            for h, g in sub.groupby("horizon"):
                arr = g["ic"].dropna().to_numpy(dtype=float)
                spreads = g["spread"].dropna().to_numpy(dtype=float)
                if arr.size == 0:
                    continue
                std = float(arr.std(ddof=1)) if arr.size > 1 else np.nan
                mean = float(arr.mean())
                months = HORIZON_MONTHS[h]
                rows.append({
                    "variant": variant, "grouping": name, "horizon": h,
                    "n_periods": int(arr.size), "mean_ic": mean,
                    "ic_ir": mean / std if std and std > 1e-9 else np.nan,
                    "hit_rate": float((arr > 0).mean()),
                    "mean_spread": float(spreads.mean()) if spreads.size else np.nan,
                    "spread_ann": (float(spreads.mean()) * 12.0 / months
                                   if spreads.size else np.nan),
                })
    return pd.DataFrame(rows)


def overlap_summary(overlaps: dict[tuple[str, float], pd.DataFrame],
                    level: str) -> pd.DataFrame:
    rows: list[dict] = []
    for (variant, top_pct), df in overlaps.items():
        if df.empty:
            continue
        for name, sub in _groupings(df, level):
            rows.append({
                "variant": variant, "top_pct": top_pct, "grouping": name,
                "n_rebals": int(len(sub)),
                "mean_overlap": float(sub["overlap"].mean()),
                "mean_n_enter": float(sub["n_enter"].mean()),
                "mean_n_exit": float(sub["n_exit"].mean()),
                "max_n_enter": int(sub["n_enter"].max()),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Dashboard — one compact row per test window
# --------------------------------------------------------------------------- #
def _window_return(sim: pd.DataFrame, window: str, col: str) -> float:
    r = sim.loc[sim["window"] == window, col].dropna()
    return float((1.0 + r).prod() - 1.0) if not r.empty else np.nan


def build_dashboard(frozen: dict, decisions: dict[str, list],
                    sims: dict[tuple[str, float, str], pd.DataFrame],
                    ) -> pd.DataFrame:
    """Per window: VIX context, overlay strength, weight movement, and window
    performance of every variant vs baseline and SPY (top-20 % equal, net)."""
    dec_by = {v: pd.DataFrame([{
        "window": d.window, "date": d.date, "vix": d.vix, "pct": d.vix_pctile,
        "strength": d.strength, "l1": d.l1_vs_baseline, "n": d.sample_n,
    } for d in ds]) for v, ds in decisions.items()}
    base_sim = sims[("baseline", FOCUS_TOP, FOCUS_MODE)]

    rows: list[dict] = []
    for label, fw in frozen.items():
        p = dec_by["pctile"]
        p = p[p["window"] == label]
        row = {
            "window": label,
            "n_rebals": len(fw.test_rebals),
            "vix_first": round(float(p["vix"].iloc[0]), 1) if not p.empty else np.nan,
            "vix_min": round(float(p["vix"].min()), 1) if not p.empty else np.nan,
            "vix_max": round(float(p["vix"].max()), 1) if not p.empty else np.nan,
            "train_vix_median": round(float(fw.vix_train.median()), 1),
            "pctile_first": round(float(p["pct"].iloc[0]), 0) if not p.empty else np.nan,
            "pctile_mean": round(float(p["pct"].mean()), 0) if not p.empty else np.nan,
        }
        for v in dec_by:
            if v == "baseline":
                continue
            d = dec_by[v]
            d = d[d["window"] == label]
            short = {"fixed_bucket": "fixed", "pctile": "pctile",
                     "pctile_strong": "strong"}.get(v, v)
            row[f"{short}_strength_mean"] = round(float(d["strength"].mean()), 3) \
                if not d.empty else np.nan
            row[f"{short}_wL1_mean"] = round(float(d["l1"].mean()), 3) \
                if not d.empty else np.nan
        row["ret_baseline"] = _window_return(base_sim, label, "net")
        for v in ("fixed_bucket", "pctile", "pctile_strong"):
            row[f"ret_{v}"] = _window_return(sims[(v, FOCUS_TOP, FOCUS_MODE)],
                                             label, "net")
        row["ret_spy"] = _window_return(base_sim, label, "spy")
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
VARIANT_COLORS = {"baseline": "steelblue", "fixed_bucket": "darkorange",
                  "pctile": "forestgreen", "pctile_strong": "crimson"}
VARIANT_LABELS = {"baseline": "Baseline (rolling-5Y)",
                  "fixed_bucket": "Fixed VIX buckets (30% tilt)",
                  "pctile": "VIX percentile ladder (10/20/30%)",
                  "pctile_strong": "Percentile ladder ×2 (20/40/60%)"}


def plot_equity(sims: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13, 6))
    base = sims[("baseline", FOCUS_TOP, FOCUS_MODE)]
    for v, color in VARIANT_COLORS.items():
        df = sims[(v, FOCUS_TOP, FOCUS_MODE)]
        r = df.set_index("realize_date")["net"].dropna()
        eq = (1.0 + r).cumprod()
        x = pd.to_datetime(eq.index)
        ax.plot(x, eq.values, color=color, linewidth=1.6, label=VARIANT_LABELS[v])
    spy = base.set_index("realize_date")["spy"].dropna()
    eq = (1.0 + spy).cumprod()
    ax.plot(pd.to_datetime(eq.index), eq.values, color="gray", linewidth=1.4,
            linestyle="--", label="SPY")
    ax.set_yscale("log")
    ax.set_ylabel("Equity (log, net of costs)")
    ax.set_title(f"VIX-relative overlay study — top {int(FOCUS_TOP*100)}% "
                 f"{FOCUS_MODE}, monthly, net")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dashboard(dash: pd.DataFrame, decisions: dict[str, list],
                   out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = pd.DataFrame([{"date": d.date, "vix": d.vix, "pct": d.vix_pctile,
                       "strength": d.strength, "l1": d.l1_vs_baseline}
                      for d in decisions["pctile"]])
    x = pd.to_datetime(p["date"])
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=False)

    ax = axes[0]
    ax.plot(x, p["vix"], color="black", linewidth=1.2, label="VIX at rebalance")
    ax2 = ax.twinx()
    ax2.fill_between(x, p["pct"], 50, color="forestgreen", alpha=0.25,
                     label="VIX percentile vs training window")
    ax2.axhspan(30, 70, color="gray", alpha=0.12)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("training percentile")
    ax.set_ylabel("VIX")
    ax.set_title("Rebalance-date VIX and its training-relative percentile "
                 "(gray band = no-tilt zone)")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    ax.bar(x, p["strength"], width=18, color="forestgreen", alpha=0.8,
           label="pctile overlay strength")
    ax.plot(x, p["l1"], color="crimson", linewidth=1.2,
            label="parent-weight shift (0.5·Σ|Δw|)")
    ax.set_ylabel("strength / weight L1")
    ax.set_title("Overlay strength and how far it moved the parent weights")
    ax.legend(fontsize=8)

    ax = axes[2]
    idx = np.arange(len(dash))
    width = 0.25
    for i, (v, color) in enumerate([("fixed_bucket", "darkorange"),
                                    ("pctile", "forestgreen"),
                                    ("pctile_strong", "crimson")]):
        delta = dash[f"ret_{v}"] - dash["ret_baseline"]
        ax.bar(idx + (i - 1) * width, delta, width, color=color,
               label=f"{VARIANT_LABELS[v]} − baseline")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(idx)
    ax.set_xticklabels(dash["window"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Δ 6-month net return")
    ax.set_title("Per-window net return: overlay minus baseline "
                 f"(top {int(FOCUS_TOP*100)}% equal)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #
def _f(v, fmt=".3f") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:{fmt}}"


def _md(df: pd.DataFrame, fmts: dict[str, str]) -> str:
    lines = ["| " + " | ".join(str(c) for c in df.columns) + " |",
             "|" + "|".join("---" for _ in df.columns) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            fmt = fmts.get(c)
            cells.append(_f(v, fmt) if fmt else
                         (str(v) if not isinstance(v, float) else _f(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


PERF_FMTS = {"total_return": "+.1%", "cagr": "+.1%", "gross_cagr": "+.1%",
             "sharpe": ".2f", "sortino": ".2f", "ann_vol": ".1%",
             "max_drawdown": ".1%", "hit_rate": ".0%", "avg_turnover": ".2f",
             "spy_cagr": "+.1%", "spy_excess_cagr": "+.1%", "spy_alpha": "+.1%",
             "spy_beta": ".2f", "spy_ir": ".2f", "spy_te": ".1%",
             "spy_rel_max_drawdown": ".1%"}
IC_FMTS = {"mean_ic": "+.3f", "ic_ir": ".2f", "hit_rate": ".0%",
           "mean_spread": "+.2%", "spread_ann": "+.1%"}
DASH_FMTS = {"pctile_first": ".0f", "pctile_mean": ".0f",
             "ret_baseline": "+.1%", "ret_fixed_bucket": "+.1%",
             "ret_pctile": "+.1%", "ret_pctile_strong": "+.1%", "ret_spy": "+.1%"}


def _verdict(full_perf: pd.DataFrame, full_ic: pd.DataFrame,
             dash: pd.DataFrame) -> list[str]:
    base = full_perf[(full_perf["variant"] == "baseline")
                     & (full_perf["top_pct"] == FOCUS_TOP)
                     & (full_perf["mode"] == FOCUS_MODE)].iloc[0]
    base_ic6 = full_ic[(full_ic["variant"] == "baseline")
                       & (full_ic["horizon"] == "6M")]["mean_ic"]
    base_ic6 = float(base_ic6.iloc[0]) if not base_ic6.empty else np.nan
    lines: list[str] = []
    any_robust = False
    for v in ("fixed_bucket", "pctile", "pctile_strong"):
        row = full_perf[(full_perf["variant"] == v)
                        & (full_perf["top_pct"] == FOCUS_TOP)
                        & (full_perf["mode"] == FOCUS_MODE)]
        if row.empty:
            continue
        row = row.iloc[0]
        ic6 = full_ic[(full_ic["variant"] == v) & (full_ic["horizon"] == "6M")]["mean_ic"]
        ic6 = float(ic6.iloc[0]) if not ic6.empty else np.nan
        d_sharpe = row["sharpe"] - base["sharpe"]
        d_ir = row["spy_ir"] - base["spy_ir"]
        d_ic = ic6 - base_ic6
        deltas = (dash[f"ret_{v}"] - dash["ret_baseline"]).dropna()
        active = deltas[deltas.abs() > 1e-9]
        win_rate = float((active > 0).mean()) if not active.empty else np.nan
        robust = (d_sharpe > 0 and d_ir > 0 and d_ic >= -0.002
                  and (win_rate != win_rate or win_rate >= 0.5))
        any_robust = any_robust or robust
        lines.append(
            f"- **{VARIANT_LABELS[v]}** — ΔSharpe {_f(d_sharpe, '+.2f')}, "
            f"ΔSPY-IR {_f(d_ir, '+.2f')}, ΔIC(6M) {_f(d_ic, '+.3f')}, "
            f"active-window win rate {_f(win_rate, '.0%')} "
            f"({int(len(active))} windows with a live tilt) → "
            + ("**improves** the baseline on this sample."
               if robust else "**does not robustly improve** the baseline."))
    head = ("**Verdict: the VIX-relative parent tilt adds a robust improvement "
            "on this sample.**" if any_robust else
            "**Verdict: no robust improvement — the rolling-5Y baseline stands.**")
    return [head, ""] + lines


def write_reports(out_dir: Path, *, frozen: dict, decisions: dict,
                  sims: dict, ics: dict, overlaps: dict, dash: pd.DataFrame,
                  cost_bps: float, fidelity_diff: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    perf = {lvl: performance_table(sims, lvl) for lvl in ("full", "block", "window")}
    ic = {lvl: ic_summary(ics, lvl) for lvl in ("full", "block", "window")}
    ov = {lvl: overlap_summary(overlaps, lvl) for lvl in ("full", "block")}

    for lvl in ("full", "block", "window"):
        perf[lvl].to_csv(out_dir / f"performance_{lvl}.csv", index=False)
        ic[lvl].to_csv(out_dir / f"ic_{lvl}.csv", index=False)
    for lvl in ("full", "block"):
        ov[lvl].to_csv(out_dir / f"overlap_{lvl}.csv", index=False)
    dash.to_csv(out_dir / "dashboard.csv", index=False)

    dec_rows = [{
        "variant": v, "window": d.window, "date": d.date, "vix": d.vix,
        "vix_pctile": d.vix_pctile, "strength": d.strength,
        "sample_key": d.sample_key, "sample_n": d.sample_n,
        "weight_l1_vs_baseline": d.l1_vs_baseline,
        **{f"w_{p}": w for p, w in sorted(d.weights.items())},
    } for v, ds in decisions.items() for d in ds]
    pd.DataFrame(dec_rows).to_csv(out_dir / "rebalance_log.csv", index=False)

    pw_rows = [{"window": lbl, "n_train_rebals": len(fw.train_rebals),
                "train_start": fw.train_start, "test_start": fw.test_start,
                **{f"w_{p}": w for p, w in sorted(fw.parent_weights.items())}}
               for lbl, fw in frozen.items()]
    pd.DataFrame(pw_rows).to_csv(out_dir / "baseline_parent_weights.csv", index=False)

    plot_equity(sims, out_dir / "equity_curves.png")
    plot_dashboard(dash, decisions, out_dir / "dashboard.png")

    # ---- REPORT.md ----
    focus = lambda df: df[(df.get("top_pct", FOCUS_TOP) == FOCUS_TOP)
                          & (df.get("mode", FOCUS_MODE) == FOCUS_MODE)] \
        if "top_pct" in df.columns else df
    lines: list[str] = [
        "# VIX-Relative Parent-Weight Overlay — clean-room walk-forward study",
        "",
        f"{len(frozen)} semiannual OOS windows ({list(frozen)[0]} → {list(frozen)[-1]}), "
        "rolling-5Y baseline rebuilt and frozen per window; overlays adjust parent "
        "weights only, monthly, from training-data VIX statistics. "
        f"Costs: {cost_bps:.0f} bps one-way on turnover. All returns net unless noted.",
        "",
        f"_Composite fidelity vs the research implementation: max abs diff "
        f"{fidelity_diff:.2e} (exact)._",
        "",
        "## Verdict",
        "",
        *_verdict(perf["full"], ic["full"][ic["full"]["grouping"] == "full"], dash),
        "",
        "---",
        "",
        f"## Full period — top {int(FOCUS_TOP*100)}% {FOCUS_MODE} (net)",
        "",
    ]
    cols = ["variant", "n_periods", "total_return", "cagr", "gross_cagr", "sharpe",
            "sortino", "ann_vol", "max_drawdown", "avg_turnover", "spy_cagr",
            "spy_excess_cagr", "spy_alpha", "spy_beta", "spy_ir",
            "spy_rel_max_drawdown"]
    lines.append(_md(focus(perf["full"])[cols], PERF_FMTS))

    lines += ["", "### All portfolio configurations (full period)", ""]
    cols_all = ["variant", "top_pct", "mode", "cagr", "sharpe", "max_drawdown",
                "spy_excess_cagr", "spy_ir", "avg_turnover"]
    lines.append(_md(perf["full"][cols_all], PERF_FMTS))

    lines += ["", "## Composite IC and Q5-Q1 (pooled OOS)", ""]
    ic_full = ic["full"][ic["full"]["grouping"] == "full"]
    lines.append(_md(ic_full[["variant", "horizon", "n_periods", "mean_ic", "ic_ir",
                              "hit_rate", "mean_spread", "spread_ann"]], IC_FMTS))

    lines += ["", "## 3-year blocks — top 20% equal (net)", ""]
    blk = focus(perf["block"])
    lines.append(_md(blk[["variant", "grouping", "cagr", "sharpe", "max_drawdown",
                          "spy_excess_cagr", "spy_ir"]], PERF_FMTS))

    lines += ["", "## Holdings impact of the overlay (full period)", ""]
    if not ov["full"].empty:
        lines.append(_md(ov["full"][["variant", "top_pct", "mean_overlap",
                                     "mean_n_enter", "mean_n_exit", "max_n_enter"]],
                         {"mean_overlap": ".1%", "mean_n_enter": ".1f",
                          "mean_n_exit": ".1f"}))

    lines += ["", "## Per-window dashboard", "",
              "_One row per 6-month OOS window; returns are the window's compounded "
              f"net top-{int(FOCUS_TOP*100)}% equal returns._", ""]
    lines.append(_md(dash, DASH_FMTS))

    lines += [
        "", "---", "",
        "## Method guarantees",
        "",
        "- Training rebalances stop 6 months before each test window so no "
        "selection or regime forward-return crosses the boundary (asserted).",
        "- The training price matrix is truncated at the boundary; the VIX "
        "distribution ends the day before the test window (asserted).",
        "- Regime utilities: 0.5·IC-rank + 0.25·IR-rank + 0.25·Q5Q1-rank on "
        "same-regime training rebalances; shrunk toward the full-window utility "
        "by n/(n+12); <4 observations → baseline weights; parent cap 25%.",
        "- Sub-factor selection is never changed by VIX; only parent weights move.",
        "- Baseline model per window produced by the existing V4 selection chain "
        "(`research.walkforward.selection.select_config`); everything else in "
        "`vix_relative_overlay/` is an independent implementation.",
        "",
        "| File | Contents |",
        "|------|----------|",
        "| `performance_{full,block,window}.csv` | net metrics per variant/config |",
        "| `ic_{full,block,window}.csv` | composite IC + Q5-Q1 |",
        "| `overlap_{full,block}.csv` | holdings overlap / names entering-leaving |",
        "| `dashboard.csv` / `dashboard.png` | per-window VIX, strength, Δweights, returns |",
        "| `rebalance_log.csv` | every rebalance decision incl. adjusted weights |",
        "| `baseline_parent_weights.csv` | frozen rolling-5Y weights per window |",
        "| `equity_curves.png` | net equity, variants vs SPY |",
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines))
