"""Studies 2+3 — Target-magnitude calibration and nonlinearity.

Question: when an analyst predicts +10% / +20% / +40%, do bigger predicted
gains actually produce bigger realized returns?

Event-level design (NOT the next-analyst-quote resolution used by the firm
skill score): every price-target event is resolved against the adj_close
matrix at fixed 1/3/6/12-month horizons from the event date. Raw, SPY-adjusted
and sector-ETF-adjusted returns. Inference uses month-cluster bootstrap
(events within a calendar month resampled together) because overlapping
forward windows make naive per-event SEs badly overstated.

Run:  python -m research.analyst_deep_dive.study23_calibration
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.analyst_deep_dive.common import (
    OUT_DIR, CHART_DIR, HORIZON_MONTHS, SECTOR_ETF, load_events,
    load_price_matrix, sector_map, month_cluster_bootstrap_slope, nearest_idx,
)

HORIZONS = ["1M", "3M", "6M", "12M"]
ECON_BUCKETS = [(-np.inf, 0.0, "<0%"), (0.0, 0.05, "0-5%"), (0.05, 0.10, "5-10%"),
                (0.10, 0.20, "10-20%"), (0.20, 0.40, "20-40%"), (0.40, np.inf, ">40%")]
PCT_GROUPS = [(0.0, 0.10, "bottom10"), (0.10, 0.25, "p10-25"), (0.25, 0.50, "p25-50"),
              (0.50, 0.75, "p50-75"), (0.75, 0.90, "p75-90"), (0.90, 1.0, "top10"),
              (0.95, 1.0, "top5"), (0.75, 0.95, "p75-95")]


def pava_isotonic(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators isotonic fit of y on x (increasing)."""
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    # start from 50 equal-count bins to keep it stable/cheap
    n_bins = 50
    idx = np.linspace(0, len(xs), n_bins + 1).astype(int)
    bx = np.array([xs[a:b].mean() for a, b in zip(idx[:-1], idx[1:]) if b > a])
    by = np.array([ys[a:b].mean() for a, b in zip(idx[:-1], idx[1:]) if b > a])
    bw = np.array([b - a for a, b in zip(idx[:-1], idx[1:]) if b > a]).astype(float)
    # PAVA
    vals = by.copy()
    wts = bw.copy()
    blocks = [[i] for i in range(len(vals))]
    i = 0
    while i < len(vals) - 1:
        if vals[i] > vals[i + 1] + 1e-12:
            new_w = wts[i] + wts[i + 1]
            new_v = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / new_w
            vals = np.concatenate([vals[:i], [new_v], vals[i + 2:]])
            wts = np.concatenate([wts[:i], [new_w], wts[i + 2:]])
            blocks = blocks[:i] + [blocks[i] + blocks[i + 1]] + blocks[i + 2:]
            i = max(i - 1, 0)
        else:
            i += 1
    fit_x, fit_y = [], []
    for bi, v in zip(blocks, vals):
        for j in bi:
            fit_x.append(bx[j])
            fit_y.append(v)
    return np.array(fit_x), np.array(fit_y)


def build_event_returns() -> pd.DataFrame:
    events = load_events(clean=True)
    tickers = sorted(events["ticker"].unique())
    etfs = sorted(set(SECTOR_ETF.values())) + ["SPY"]
    matrix = load_price_matrix(start="2021-01-01")
    matrix = matrix.ffill()          # delisting realization, same as battery
    dates_idx = pd.DatetimeIndex(pd.to_datetime(matrix.index))
    smap = sector_map(normalized=True)

    events = events[events["ticker"].isin(matrix.columns)].copy()
    events["sector"] = events["ticker"].map(smap)
    events["etf"] = events["sector"].map(SECTOR_ETF)

    rows = []
    by_ticker = {t: matrix[t].values for t in set(events["ticker"]) | set(etfs)
                 if t in matrix.columns}
    date_strs = matrix.index.tolist()

    def px(tkr, pos):
        arr = by_ticker.get(tkr)
        if arr is None:
            return np.nan
        return arr[pos]

    pos_cache: dict[str, int | None] = {}

    def entry_pos(dstr):
        if dstr in pos_cache:
            return pos_cache[dstr]
        ts = pd.Timestamp(dstr)
        pos = dates_idx.searchsorted(ts)
        if pos >= len(dates_idx) or (dates_idx[pos] - ts).days > 5:
            pos_cache[dstr] = None
        else:
            pos_cache[dstr] = pos
        return pos_cache[dstr]

    horizon_pos_cache: dict[tuple[str, str], int | None] = {}

    def horizon_pos(dstr, h):
        key = (dstr, h)
        if key in horizon_pos_cache:
            return horizon_pos_cache[key]
        target = pd.Timestamp(dstr) + pd.DateOffset(months=HORIZON_MONTHS[h])
        best = nearest_idx(dates_idx, target, max_gap_days=15)
        horizon_pos_cache[key] = None if best is None else dates_idx.get_loc(best)
        return horizon_pos_cache[key]

    for r in events.itertuples(index=False):
        e = entry_pos(r.date)
        if e is None:
            continue
        p0 = px(r.ticker, e)
        if not np.isfinite(p0) or p0 <= 0:
            continue
        row = {"ticker": r.ticker, "date": r.date, "firm": r.analyst_company,
               "sector": r.sector, "upside": r.upside, "log_upside": r.log_upside}
        keep = False
        for h in HORIZONS:
            hp = horizon_pos(r.date, h)
            if hp is None or hp <= e:
                row[f"ret_{h}"] = np.nan
                row[f"mkt_{h}"] = np.nan
                row[f"sec_{h}"] = np.nan
                row[f"attain_{h}"] = np.nan
                continue
            p1 = px(r.ticker, hp)
            if not np.isfinite(p1):
                row[f"ret_{h}"] = np.nan
                row[f"mkt_{h}"] = np.nan
                row[f"sec_{h}"] = np.nan
                row[f"attain_{h}"] = np.nan
                continue
            ret = p1 / p0 - 1.0
            row[f"ret_{h}"] = ret
            spy0, spy1 = px("SPY", e), px("SPY", hp)
            row[f"mkt_{h}"] = ret - (spy1 / spy0 - 1.0) if np.isfinite(spy0) and np.isfinite(spy1) else np.nan
            if isinstance(r.etf, str):
                e0, e1 = px(r.etf, e), px(r.etf, hp)
                row[f"sec_{h}"] = ret - (e1 / e0 - 1.0) if np.isfinite(e0) and np.isfinite(e1) else np.nan
            else:
                row[f"sec_{h}"] = np.nan
            path = by_ticker[r.ticker][e:hp + 1]
            row[f"attain_{h}"] = bool(np.nanmax(path) / p0 - 1.0 >= r.upside)
            keep = True
        if keep:
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    cache_f = OUT_DIR / "event_returns.parquet"
    if cache_f.exists():
        ev = pd.read_parquet(cache_f)
    else:
        ev = build_event_returns()
        ev.to_parquet(cache_f)
    print(f"events with resolved horizons: {len(ev)}, {ev['date'].min()}..{ev['date'].max()}")

    # ------------------------------------------------------------------
    # Regressions per horizon (raw + market-adjusted + sector-adjusted)
    # ------------------------------------------------------------------
    reg_rows = []
    for h in HORIZONS:
        for retcol, label in [(f"ret_{h}", "raw"), (f"mkt_{h}", "mkt_adj"), (f"sec_{h}", "sec_adj")]:
            d = ev[["date", "upside", retcol]].dropna()
            if len(d) < 500:
                continue
            res = month_cluster_bootstrap_slope(d, "upside", retcol)
            pear = d["upside"].corr(d[retcol])
            spear = d["upside"].corr(d[retcol], method="spearman")
            from scipy import stats as st
            p_boot = 2 * (1 - st.norm.cdf(abs(res["t_boot"])))
            reg_rows.append({
                "horizon": h, "returns": label, "alpha": res["alpha"], "beta": res["beta"],
                "boot_se": res["boot_se"], "t_stat": res["t_boot"], "p_value": p_boot,
                "ci90_lo": res["ci_lo"], "ci90_hi": res["ci_hi"],
                "pearson_r": pear, "r2": pear ** 2, "spearman": spear,
                "n_obs": res["n"], "n_months": res["n_months"],
                "date_min": d["date"].min(), "date_max": d["date"].max(),
            })
    reg = pd.DataFrame(reg_rows)
    reg.to_csv(OUT_DIR / "study2_regressions.csv", index=False)
    print("\n=== regressions (ActualReturn ~ ExpectedUpside) ===")
    print(reg.round(4).to_string(index=False))

    # log-upside variant, raw returns only
    log_rows = []
    for h in HORIZONS:
        d = ev[["date", "log_upside", f"ret_{h}"]].dropna()
        res = month_cluster_bootstrap_slope(d, "log_upside", f"ret_{h}")
        log_rows.append({"horizon": h, **{k: res[k] for k in ["alpha", "beta", "boot_se", "t_boot", "n"]}})
    pd.DataFrame(log_rows).to_csv(OUT_DIR / "study2_regressions_log.csv", index=False)

    # ------------------------------------------------------------------
    # Economic buckets (study 3)
    # ------------------------------------------------------------------
    bucket_rows = []
    for lo, hi, name in ECON_BUCKETS:
        sel = ev[(ev["upside"] > lo) & (ev["upside"] <= hi)]
        for h in HORIZONS:
            r = sel[f"ret_{h}"].dropna()
            if len(r) < 30:
                continue
            bucket_rows.append({
                "bucket": name, "horizon": h, "n": len(r),
                "mean_upside": float(sel["upside"].mean()),
                "mean_ret": float(r.mean()), "median_ret": float(r.median()),
                "mean_mkt_adj": float(sel[f"mkt_{h}"].dropna().mean()),
                "mean_sec_adj": float(sel[f"sec_{h}"].dropna().mean()),
                "pct_positive": float((r > 0).mean()),
                "pct_attained": float(sel[f"attain_{h}"].dropna().mean()),
                "mean_forecast_err": float((sel["upside"] - sel[f"ret_{h}"]).dropna().mean()),
                "median_forecast_err": float((sel["upside"] - sel[f"ret_{h}"]).dropna().median()),
            })
    bdf = pd.DataFrame(bucket_rows)
    bdf.to_csv(OUT_DIR / "study3_econ_buckets.csv", index=False)
    print("\n=== economic buckets (12M) ===")
    print(bdf[bdf["horizon"] == "12M"].round(4).to_string(index=False))

    # ------------------------------------------------------------------
    # Percentile groups — percentiles WITHIN event-month (cross-sectional)
    # ------------------------------------------------------------------
    ev["month"] = ev["date"].str[:7]
    ev["pctile"] = ev.groupby("month")["upside"].rank(pct=True)
    pct_rows = []
    for lo, hi, name in PCT_GROUPS:
        sel = ev[(ev["pctile"] > lo) & (ev["pctile"] <= hi)]
        for h in HORIZONS:
            r = sel[f"ret_{h}"].dropna()
            if len(r) < 30:
                continue
            pct_rows.append({
                "group": name, "horizon": h, "n": len(r),
                "mean_ret": float(r.mean()), "median_ret": float(r.median()),
                "mean_mkt_adj": float(sel[f"mkt_{h}"].dropna().mean()),
                "mean_sec_adj": float(sel[f"sec_{h}"].dropna().mean()),
                "pct_positive": float((r > 0).mean()),
            })
    pdf = pd.DataFrame(pct_rows)
    pdf.to_csv(OUT_DIR / "study3_percentile_groups.csv", index=False)

    # headline comparisons
    def grp(name, h, col="mean_sec_adj"):
        row = pdf[(pdf["group"] == name) & (pdf["horizon"] == h)]
        return float(row[col].iloc[0]) if len(row) else np.nan

    comparisons = {}
    for h in ["3M", "12M"]:
        full = ev[f"sec_{h}"].dropna().mean()
        comparisons[h] = {
            "top10_minus_full": grp("top10", h) - full,
            "top10_minus_mid50": grp("top10", h) - (grp("p25-50", h) + grp("p50-75", h)) / 2,
            "top5_minus_p75_95": grp("top5", h) - grp("p75-95", h),
            "extreme_gt40_minus_moderate_10_20": (
                float(bdf[(bdf["bucket"] == ">40%") & (bdf["horizon"] == h)]["mean_sec_adj"].iloc[0])
                - float(bdf[(bdf["bucket"] == "10-20%") & (bdf["horizon"] == h)]["mean_sec_adj"].iloc[0])),
        }
    with open(OUT_DIR / "study3_comparisons.json", "w") as f:
        json.dump(comparisons, f, indent=2)
    print("\n=== headline comparisons (sector-adjusted) ===")
    print(json.dumps(comparisons, indent=2))

    # isotonic (PAVA) fit for 3M and 12M
    iso_out = {}
    for h in ["3M", "12M"]:
        d = ev[["upside", f"ret_{h}"]].dropna()
        fx, fy = pava_isotonic(d["upside"].values, d[f"ret_{h}"].values)
        iso_out[h] = {"x": fx.tolist(), "y": fy.tolist()}
    with open(OUT_DIR / "study3_isotonic.json", "w") as f:
        json.dump(iso_out, f)

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def eq_bins(d, xcol, ycol, n_bins=25):
        d = d[[xcol, ycol]].dropna().sort_values(xcol)
        idx = np.linspace(0, len(d), n_bins + 1).astype(int)
        bx = [d[xcol].iloc[a:b].mean() for a, b in zip(idx[:-1], idx[1:]) if b > a]
        by = [d[ycol].iloc[a:b].mean() for a, b in zip(idx[:-1], idx[1:]) if b > a]
        bm = [d[ycol].iloc[a:b].median() for a, b in zip(idx[:-1], idx[1:]) if b > a]
        return np.array(bx), np.array(by), np.array(bm)

    # 1. scatter + binned + regression per horizon
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, h in zip(axes.flat, HORIZONS):
        d = ev[["date", "upside", f"ret_{h}"]].dropna()
        ax.scatter(d["upside"], d[f"ret_{h}"], s=3, alpha=0.08, color="#888")
        bx, by, _ = eq_bins(d, "upside", f"ret_{h}")
        ax.plot(bx, by, "o-", color="#2a78d6", lw=2, ms=5, label="equal-count bin mean")
        row = reg[(reg["horizon"] == h) & (reg["returns"] == "raw")].iloc[0]
        xs = np.linspace(d["upside"].quantile(0.01), d["upside"].quantile(0.99), 50)
        ax.plot(xs, row["alpha"] + row["beta"] * xs, "--", color="#e34948", lw=1.5,
                label=f"OLS β={row['beta']:.3f} (t={row['t_stat']:.1f})")
        ax.fill_between(xs, row["alpha"] + row["ci90_lo"] * xs,
                        row["alpha"] + row["ci90_hi"] * xs, color="#e34948", alpha=0.12)
        ax.set_xlim(-0.3, 1.0)
        ax.set_ylim(-0.6, 1.0)
        ax.axhline(0, color="#ccc", lw=0.7)
        ax.set_title(f"{h}: realized vs expected upside")
        ax.set_xlabel("expected upside at issue")
        ax.set_ylabel("realized return")
        ax.legend(fontsize=8)
    fig.suptitle("Study 2 — target-magnitude calibration (raw returns)", y=1.0)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "calibration_scatter_binned.png", dpi=110)
    plt.close(fig)

    # 2. median realized by economic bucket (12M + 3M)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, h in zip(axes, ["3M", "12M"]):
        sub = bdf[bdf["horizon"] == h]
        ax.bar(sub["bucket"], sub["median_ret"], color="#2a78d6", label="median realized")
        ax.plot(sub["bucket"], sub["mean_upside"], "o--", color="#e34948", label="mean promised")
        ax.axhline(0, color="#ccc", lw=0.7)
        ax.set_title(f"{h} horizon")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Median realized return vs promised upside, by bucket")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "bucket_median_vs_promised.png", dpi=110)
    plt.close(fig)

    # 3. mean sector-adjusted return by percentile group
    fig, ax = plt.subplots(figsize=(9, 4.5))
    order = ["bottom10", "p10-25", "p25-50", "p50-75", "p75-90", "top10", "top5"]
    for h, color in [("3M", "#2a78d6"), ("12M", "#1baf7a")]:
        vals = [grp(g, h) for g in order]
        ax.plot(order, vals, "o-", label=h, color=color)
    ax.axhline(0, color="#ccc", lw=0.7)
    ax.set_title("Mean sector-adjusted return by within-month upside percentile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHART_DIR / "percentile_sector_adj.png", dpi=110)
    plt.close(fig)

    # 4. sample counts + attainment
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sub = bdf[bdf["horizon"] == "12M"]
    axes[0].bar(sub["bucket"], sub["n"], color="#4a3aa7")
    axes[0].set_title("Sample count by bucket (12M-resolved)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].bar(sub["bucket"], sub["pct_attained"], color="#eda100")
    axes[1].set_title("Share touching the target within 12M")
    axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "bucket_counts_attainment.png", dpi=110)
    plt.close(fig)

    # 5. isotonic overlay
    fig, ax = plt.subplots(figsize=(8, 5))
    for h, color in [("3M", "#2a78d6"), ("12M", "#1baf7a")]:
        fx, fy = np.array(iso_out[h]["x"]), np.array(iso_out[h]["y"])
        ax.plot(fx, fy, "-", color=color, lw=2, label=f"isotonic {h}")
    ax.axhline(0, color="#ccc", lw=0.7)
    ax.set_xlim(-0.25, 1.0)
    ax.set_xlabel("expected upside")
    ax.set_ylabel("isotonic-fit realized return")
    ax.set_title("Isotonic (monotone) fit — realized return vs expected upside")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHART_DIR / "isotonic_fit.png", dpi=110)
    plt.close(fig)

    print(f"\ncharts written to {CHART_DIR}")


if __name__ == "__main__":
    main()
