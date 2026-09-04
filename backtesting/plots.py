"""Matplotlib visuals for a backtest run (no seaborn, headless Agg backend).

Each function renders one figure to ``output_dir`` and returns its path;
:func:`generate_all` produces the full required set. Figures degrade gracefully:
if a series is empty (e.g. shorts in a long-only run) it is simply skipped rather
than raising, so a partial run still yields the charts it can.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import hit_rates

_SERIES_STYLE = {
    "long_basket": ("Long basket", "#1f77b4"),
    "short_basket": ("Short basket", "#d62728"),
    "long_short": ("Long/Short", "#2ca02c"),
    "spy": ("SPY", "#7f7f7f"),
}


def _dt(index) -> pd.DatetimeIndex:
    return pd.to_datetime(pd.Index(index))


def equity_curve(result, output_dir: Path) -> Path:
    eq = result.equity_curve.copy()
    eq.index = _dt(eq.index)
    fig, ax = plt.subplots(figsize=(11, 6))
    for col, (label, color) in _SERIES_STYLE.items():
        if col in eq and eq[col].notna().sum() > 1:
            ax.plot(eq.index, eq[col], label=label, color=color, linewidth=1.6)
    ax.set_title(f"Equity Curve — {result.spec.label}")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("Date")
    ax.axhline(1.0, color="black", linewidth=0.6, alpha=0.4)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, output_dir / "equity_curve.png")


def drawdowns(result, output_dir: Path) -> Path:
    eq = result.equity_curve.copy()
    eq.index = _dt(eq.index)
    fig, ax = plt.subplots(figsize=(11, 5))
    for col in (result.primary_series, "spy"):
        s = eq[col].dropna() if col in eq else pd.Series(dtype=float)
        if len(s) > 1:
            dd = (s / s.cummax() - 1.0) * 100.0
            label, color = _SERIES_STYLE[col]
            ax.fill_between(dd.index, dd.values, 0, alpha=0.3, color=color)
            ax.plot(dd.index, dd.values, label=label, color=color, linewidth=1.2)
    ax.set_title(f"Drawdown — {result.spec.label}")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, output_dir / "drawdowns.png")


def rolling_returns(result, output_dir: Path) -> Path:
    eq = result.equity_curve.copy()
    eq.index = _dt(eq.index)
    fig, ax = plt.subplots(figsize=(11, 5))
    for col in (result.primary_series, "spy"):
        s = eq[col].dropna() if col in eq else pd.Series(dtype=float)
        if len(s) > 3:
            monthly = s.resample("ME").last()
            roll = (monthly / monthly.shift(3) - 1.0).dropna() * 100.0
            if not roll.empty:
                label, color = _SERIES_STYLE[col]
                ax.plot(roll.index, roll.values, label=label, color=color, linewidth=1.4)
    ax.set_title(f"Rolling 3-Month Returns — {result.spec.label}")
    ax.set_ylabel("3-Month Return (%)")
    ax.set_xlabel("Date")
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, output_dir / "rolling_returns.png")


def long_forward_distribution(result, output_dir: Path) -> Path:
    fwd = result.forward_returns
    longs = fwd[fwd["side"] == "long"]["stock_fwd"].dropna() if not fwd.empty else pd.Series(dtype=float)
    return _distribution(
        longs, output_dir / "long_forward_return_distribution.png",
        "Long Basket — Forward Return Distribution", "#1f77b4")


def short_forward_distribution(result, output_dir: Path) -> Path:
    fwd = result.forward_returns
    shorts = fwd[fwd["side"] == "short"]["stock_fwd"].dropna() if not fwd.empty else pd.Series(dtype=float)
    return _distribution(
        shorts, output_dir / "short_forward_return_distribution.png",
        "Short Basket — Underlying Forward Return Distribution (want < 0)", "#d62728")


def hit_rate_chart(result, output_dir: Path) -> Path:
    rates = hit_rates(result.forward_returns)
    labels = [
        "Long picks\npositive",
        "Short picks\nnegative",
        "Long picks\nbeat SPY",
        "Short picks\nunder SPY",
    ]
    keys = ["long_hit_rate", "short_hit_rate", "long_vs_spy_hit_rate", "short_underperf_hit_rate"]
    vals = [(rates[k] or np.nan) * 100.0 for k in keys]
    colors = ["#1f77b4", "#d62728", "#1f77b4", "#d62728"]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, vals, color=colors, alpha=0.85)
    ax.axhline(50, color="black", linewidth=1.0, linestyle="--", alpha=0.6, label="50% (coin flip)")
    ax.set_ylabel("Hit Rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Hit Rates — {result.spec.label}")
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.1f}%",
                    ha="center", va="bottom", fontsize=10)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, output_dir / "hit_rates.png")


def factor_quintile_chart(result, output_dir: Path) -> Path:
    q = result.factor_quintiles
    fig, ax = plt.subplots(figsize=(9, 6))
    cols = [f"q{i}_forward_return" for i in range(1, 6)]
    if not q.empty and all(c in q.columns for c in cols):
        means = [q[c].mean() * 100.0 for c in cols]
        colors = ["#d62728", "#ff7f0e", "#bcbd22", "#17becf", "#2ca02c"]
        bars = ax.bar(["Q1\n(lowest)", "Q2", "Q3", "Q4", "Q5\n(highest)"], means, color=colors, alpha=0.85)
        for bar, v in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}%",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
        spread = q["q5_minus_q1_spread"].mean() * 100.0
        ax.set_title(f"Average Forward Return by Composite Quintile\n"
                     f"Q5-Q1 spread = {spread:.2f}% per period — {result.spec.label}")
    else:
        ax.set_title("Factor quintile returns (insufficient data)")
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_ylabel("Avg Forward Return per Period (%)")
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, output_dir / "factor_quintile_returns.png")


def factor_weights_chart(result, output_dir: Path) -> Path:
    wl = result.weights_log
    fig, ax = plt.subplots(figsize=(11, 6))
    wcols = [c for c in wl.columns if c.endswith("_w")] if not wl.empty else []
    if wcols and "rebalance_date" in wl.columns:
        x = pd.to_datetime(wl["rebalance_date"])
        data = wl[wcols].fillna(0.0)
        labels = [c[:-2] for c in wcols]
        ax.stackplot(x, *[data[c].values for c in wcols], labels=labels, alpha=0.85)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Composite weight")
        ax.set_xlabel("Rebalance date")
        ax.legend(loc="upper center", ncol=4, fontsize=8, framealpha=0.9)
        applied = set(wl.get("applied", pd.Series(dtype=object)))
        mode = "walk-forward IC" if any(str(a).startswith("ic") or "warmup" in str(a)
                                        for a in applied) else "static / regime"
        ax.set_title(f"Composite Factor Weights Over Time ({mode}) — {result.spec.label}")
    else:
        ax.set_title("Factor weights (no data)")
    ax.grid(True, alpha=0.3)
    return _save(fig, output_dir / "factor_weights.png")


def generate_all(result, output_dir: str | Path) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [
        equity_curve(result, out),
        drawdowns(result, out),
        rolling_returns(result, out),
        long_forward_distribution(result, out),
        short_forward_distribution(result, out),
        hit_rate_chart(result, out),
        factor_quintile_chart(result, out),
        factor_weights_chart(result, out),
    ]
    return [p for p in paths if p is not None]


# ---------------------------------------------------------------------------
def _distribution(values: pd.Series, path: Path, title: str, color: str) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))
    if len(values) >= 2:
        pct = values * 100.0
        ax.hist(pct, bins=40, color=color, alpha=0.8, edgecolor="white")
        mean = float(pct.mean())
        ax.axvline(0, color="black", linewidth=1.0, alpha=0.6)
        ax.axvline(mean, color="black", linewidth=1.4, linestyle="--",
                   label=f"mean = {mean:.2f}%")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xlabel("Forward Return (%)")
    ax.set_ylabel("Count (name-periods)")
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, path)


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
