"""One-off chart: Sharpe / IC(6M) / max-drawdown across the 6 regime-aware study
variants (B, B+2%floor, C, C+2%floor, 24M, 24M+2%floor), grouped by evidence mode
(5Y long-run / VIX-weighted / 24-month) with paired no-floor vs 2%-floor bars.
Reads output/regime_aware/walkforward_comparison.csv, writes
output/regime_aware/variant_comparison_bars.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT_DIR = Path("output/regime_aware")
MODES = [("B", "5Y (long-run)"), ("C", "VIX-weighted"), ("24M", "24-month")]
METRICS = [("sharpe", "Sharpe"), ("ic_6m", "IC (6M)"), ("max_drawdown", "Max Drawdown")]


def main() -> None:
    df = pd.read_csv(OUT_DIR / "walkforward_comparison.csv")
    means = df.groupby("variant")[[m for m, _ in METRICS]].mean()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    x = range(len(MODES))
    width = 0.35
    for ax, (metric, label) in zip(axes, METRICS):
        no_floor = [means.loc[key, metric] for key, _ in MODES]
        floor = [means.loc[f"{key}+2%floor", metric] for key, _ in MODES]
        ax.bar([i - width / 2 for i in x], no_floor, width, label="No floor", color="#4C72B0")
        ax.bar([i + width / 2 for i in x], floor, width, label="2% floor", color="#DD8452")
        ax.set_xticks(list(x))
        ax.set_xticklabels([name for _, name in MODES])
        ax.set_title(label)
        ax.axhline(0, color="black", linewidth=0.6)
        for i, v in enumerate(no_floor):
            ax.text(i - width / 2, v, f"{v:.3f}" if abs(v) < 1 else f"{v:.2f}",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
        for i, v in enumerate(floor):
            ax.text(i + width / 2, v, f"{v:.3f}" if abs(v) < 1 else f"{v:.2f}",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Regime-Aware Study: Full-Period Mean by Evidence Mode x Floor")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "variant_comparison_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'variant_comparison_bars.png'}")


if __name__ == "__main__":
    main()
