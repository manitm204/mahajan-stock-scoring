"""Block bootstrap over calendar time (user request 2026-09-10) -- the real
significance number for the champion's per-period diff series.

The one-sample t-test used in champion_period_breakdown.py (t=2.00, p=0.049)
assumes the 77 monthly diff observations are independent. They aren't: the
staggered-sleeve book holds names for 4 months and reviews sleeves on a
staggered monthly cadence, so adjacent months share holdings and returns are
serially correlated. A plain t-test on autocorrelated data understates the
true standard error and overstates significance.

Moving-block bootstrap fix: resample contiguous BLOCKS of months (not
individual months) with replacement, preserving the local autocorrelation
structure within each block. Build the bootstrap sampling distribution of
the mean monthly diff, then read off a real 95% CI and two-sided p-value
(fraction of the bootstrap distribution on the far side of zero, doubled).
Reports several block lengths (1, 3, 4, 6, 12 months) since the choice of L
trades off bias (too short: still autocorrelated) vs variance (too long:
few effective blocks) -- L=4 (the sleeve hold length) is the principled
default, but should agree in sign/rough magnitude with its neighbors.

Usage: python -m research.loop_research.block_bootstrap
Reads: output/loop_research/champion_period_diff_series.csv
Writes: output/loop_research/block_bootstrap.csv
        output/loop_research/block_bootstrap.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

IN_CSV = REPO / "output" / "loop_research" / "champion_period_diff_series.csv"
OUT_DIR = REPO / "output" / "loop_research"
BLUE = "#2a78d6"
MAGENTA = "#e87ba4"
GRAY = "#8a8f98"

BLOCK_LENGTHS = [1, 3, 4, 6, 12]
N_BOOT = 10000


def moving_block_bootstrap(x: np.ndarray, block_len: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    n = len(x)
    block_len = min(block_len, n)
    n_blocks_needed = int(np.ceil(n / block_len))
    max_start = n - block_len
    starts = rng.integers(0, max_start + 1, size=(n_boot, n_blocks_needed))
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        sample = np.concatenate([x[s:s + block_len] for s in starts[b]])[:n]
        boot_means[b] = sample.mean()
    return boot_means


def bootstrap_table(x: np.ndarray, block_lengths=BLOCK_LENGTHS, n_boot: int = N_BOOT,
                    seed: int = 0):
    """Generic entry point (any candidate-minus-baseline monthly diff series,
    not just the champion's) -- used directly by promotion.py's
    evaluate_promotion() output (report['diff_by_date'])."""
    x = np.asarray(x)
    observed_mean = float(x.mean())
    rng = np.random.default_rng(seed)

    rows = []
    boot_dists = {}
    for L in block_lengths:
        boot = moving_block_bootstrap(x, L, n_boot, rng)
        boot_dists[L] = boot
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p_two_sided = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
        p_two_sided = min(p_two_sided, 1.0)
        rows.append({
            "block_len_months": L,
            "observed_mean_monthly_diff": observed_mean,
            "boot_se": float(boot.std(ddof=1)),
            "ci_lo_95": float(lo),
            "ci_hi_95": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0),
            "p_value_two_sided": float(p_two_sided),
        })
    table = pd.DataFrame(rows)
    return table, boot_dists, x


def run(seed: int = 0) -> pd.DataFrame:
    df = pd.read_csv(IN_CSV)
    x = df["diff"].to_numpy()
    return bootstrap_table(x, seed=seed)


def _plot(table: pd.DataFrame, boot_dists: dict, x: np.ndarray):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))

    ax = axes[0]
    L_default = 4
    boot = boot_dists[L_default]
    ax.hist(boot, bins=50, color=BLUE, alpha=0.85)
    ax.axvline(0.0, color=GRAY, linestyle="--", linewidth=1.5, label="no edge (0)")
    obs = table.loc[table["block_len_months"] == L_default, "observed_mean_monthly_diff"].iloc[0]
    ax.axvline(obs, color=MAGENTA, linewidth=2, label=f"observed mean = {obs:+.4f}")
    row = table[table["block_len_months"] == L_default].iloc[0]
    ax.set_title(f"Block bootstrap (L={L_default} months), p={row['p_value_two_sided']:.3f}")
    ax.set_xlabel("bootstrap mean monthly diff")
    ax.legend(fontsize=8.5)

    ax2 = axes[1]
    ax2.errorbar(table["block_len_months"], table["observed_mean_monthly_diff"],
                yerr=[table["observed_mean_monthly_diff"] - table["ci_lo_95"],
                      table["ci_hi_95"] - table["observed_mean_monthly_diff"]],
                fmt="o-", color=MAGENTA, ecolor=BLUE, elinewidth=2, capsize=4)
    ax2.axhline(0.0, color=GRAY, linestyle="--", linewidth=1.25)
    ax2.set_xlabel("block length (months)")
    ax2.set_ylabel("mean monthly diff, 95% bootstrap CI")
    ax2.set_title("Sensitivity to block length")

    fig.suptitle("Block bootstrap over calendar time: champion vs random-baseline diff", fontsize=12)
    fig.tight_layout()
    out_png = OUT_DIR / "block_bootstrap.png"
    fig.savefig(out_png, dpi=140)
    print(f"wrote {out_png}")


def main():
    table, boot_dists, x = run()
    table.to_csv(OUT_DIR / "block_bootstrap.csv", index=False)
    print(f"wrote {OUT_DIR / 'block_bootstrap.csv'}")
    print(table.to_string(index=False))
    _plot(table, boot_dists, x)


if __name__ == "__main__":
    main()
