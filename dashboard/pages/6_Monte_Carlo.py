"""Monte Carlo — does random selection among tied names matter?

Precomputed by scripts/monte_carlo_random_book.py and
scripts/monte_carlo_managed_book.py (research artifacts, not wired into the
live production composite/scoring pipeline -- see
docs/monte_carlo_book_construction.md). This page only reads those cached
JSONs; it does not recompute the simulations live.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if sys.path[:1] != [_ROOT]:
    if _ROOT in sys.path:
        sys.path.remove(_ROOT)
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components import empty_state, page_header, sidebar

st.set_page_config(page_title="Monte Carlo", layout="wide", page_icon="🎲")
sidebar()

ROOT = Path(__file__).resolve().parents[2]

CONFIGS = {
    "sleeves": {
        "label": "Staggered sleeves — n=5 / hold=4mo / refresh=5 / 4 sleeves",
        "path": ROOT / "output" / "monte_carlo_random_book" / "n5_hold4_refresh5_sleeves4_sims2500.json",
        "color": "#2563eb",
        "blurb": (
            "4 staggered 5-name sleeves, each reviewed every 4 months, evicting "
            "5 names at random from that sleeve and refilling from the composite "
            "score==100 pool (instead of the deterministic worst-momentum-decline "
            "eviction rule)."
        ),
    },
}
BENCH_COLOR = {"SPY": "#64748b", "QQQ": "#94a3b8"}


@st.cache_data(ttl=3600, show_spinner=False)
def load_config(path_str: str) -> dict | None:
    path = Path(path_str)
    if not path.exists():
        return None
    with path.open() as fh:
        return json.load(fh)


page_header("Monte Carlo", "Random-entry simulations vs SPY and QQQ")

any_loaded = False
for key, meta in CONFIGS.items():
    data = load_config(str(meta["path"]))
    if data is None:
        continue
    any_loaded = True

    st.markdown(f"### {meta['label']}")
    st.caption(meta["blurb"])

    dates = data["dates"]
    p10 = np.array(data["p10_curve"]) * 1000.0
    p90 = np.array(data["p90_curve"]) * 1000.0
    median = np.array(data["median_curve"]) * 1000.0
    spy = np.array(data["spy_curve"]) * 1000.0
    qqq = np.array(data["qqq_curve"]) * 1000.0

    BAND_COLOR = "rgba(96, 165, 250, 0.28)"   # light blue
    MEDIAN_COLOR = "#1e3a8a"                  # dark blue
    LINE_COLORS = {"SPY": "#ea580c", "QQQ": "#059669"}   # distinct, dashed

    fig = go.Figure()
    fig.add_scatter(x=dates, y=p90, mode="lines", line=dict(width=0),
                    showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=dates, y=p10, mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor=BAND_COLOR,
                    name="10th–90th percentile", hoverinfo="skip")
    fig.add_scatter(x=dates, y=median, mode="lines",
                    line=dict(color=MEDIAN_COLOR, width=3.5), name="Median")
    fig.add_scatter(x=dates, y=spy, mode="lines",
                    line=dict(color=LINE_COLORS["SPY"], width=1.75, dash="dash"),
                    name="SPY")
    fig.add_scatter(x=dates, y=qqq, mode="lines",
                    line=dict(color=LINE_COLORS["QQQ"], width=1.75, dash="dash"),
                    name="QQQ")

    end_markers = [(median[-1], MEDIAN_COLOR), (spy[-1], LINE_COLORS["SPY"]),
                  (qqq[-1], LINE_COLORS["QQQ"])]
    for y_end, color in end_markers:
        fig.add_annotation(x=dates[-1], y=y_end, text=f"${y_end:,.0f}",
                           showarrow=False, xanchor="left", xshift=8,
                           font=dict(color=color, size=12))

    fig.update_layout(
        height=420, margin=dict(l=10, r=70, t=10, b=10),
        yaxis_title="Portfolio Value ($1,000 invested)",
        yaxis=dict(tickprefix="$", gridcolor="rgba(148, 163, 184, 0.15)"),
        xaxis=dict(gridcolor="rgba(148, 163, 184, 0.15)"),
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True, key=f"{key}_chart")

    params = data["params"]
    summary = data["summary"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sims", summary["n_sims"])
    m2.metric("Median final return", f"{summary['median_final_return']:.2f}x")
    m3.metric("% sims beating SPY", f"{summary['pct_sims_beating_spy']:.0%}")
    m4.metric("% sims beating QQQ", f"{summary['pct_sims_beating_qqq']:.0%}")

    rows = []
    for name, label in [("portfolio", meta["label"]), ("spy", "SPY"), ("qqq", "QQQ")]:
        s = data["stats"][name]
        rows.append({
            "": label, "CAGR": s["cagr"], "Sharpe": s["sharpe"],
            "Sortino": s["sortino"], "Max DD": s["max_dd"],
            "Beta (vs SPY)": s["beta"], "Alpha (vs SPY, ann.)": s["alpha"],
        })
    table = pd.DataFrame(rows).set_index("")
    st.dataframe(
        table.style.format({
            "CAGR": "{:+.1%}", "Sharpe": "{:.2f}", "Sortino": "{:.2f}",
            "Max DD": "{:.1%}", "Beta (vs SPY)": "{:.2f}",
            "Alpha (vs SPY, ann.)": "{:+.1%}",
        }),
        use_container_width=True, height=42 + 35 * len(table))
    st.caption(f"Params: `{json.dumps(params)}`. Portfolio stats are the mean "
              f"across all {summary['n_sims']} random-draw sims.")
    st.divider()

# --- Market-neutral variants (each long is hedged with a short) ---------- #
# Same random long book as above, but every long name is paired with a short
# leg so portfolio beta collapses toward 0. "% beating SPY/QQQ" is not a
# meaningful bar for a hedged book, so these show "% sims positive" + beta
# instead (and read from results.json, which lacks pct_sims_beating_qqq).
MN_CONFIGS = {
    "market_neutral": {
        "label": "Market-neutral — short the sector ETF",
        "path": ROOT / "output" / "monte_carlo_market_neutral" / "results.json",
        "portfolio_label": "Long − sector ETF",
        "blurb": (
            "Same staggered random long book, but each long name is paired with "
            "a short of its GICS sector SPDR ETF at 1x the long weight "
            "(dollar-neutral per name), driving portfolio beta to ~0. Strips out "
            "market return, leaving the stock-specific residual."
        ),
    },
    "short_worst_in_sector": {
        "label": "Market-neutral — short a weak same-sector name",
        "path": ROOT / "output" / "monte_carlo_short_worst_in_sector" / "results.json",
        "portfolio_label": "Long − weak same-sector short",
        "blurb": (
            "Each long name is hedged by shorting one randomly-chosen stock from "
            "the 3 lowest composite-scored names in its GICS sector (short "
            "persisted while the long is held). Harvests a within-sector "
            "good-minus-bad spread at near-zero beta, at the cost of single-name "
            "short risk."
        ),
    },
    "short_midlow_in_sector": {
        "label": "Market-neutral — short a mildly-weak same-sector name (20–40th pctile)",
        "path": ROOT / "output" / "monte_carlo_short_midlow_in_sector" / "results.json",
        "portfolio_label": "Long − 20–40th pctile same-sector short",
        "blurb": (
            "Each long name is hedged by shorting one randomly-chosen stock from "
            "the 20th–40th within-sector composite-score percentile band of its "
            "GICS sector (short persisted while the long is held). Avoids shorting "
            "the absolute bottom names — which are the most prone to violent "
            "mean-reversion / short squeezes — while still capturing a "
            "within-sector good-minus-weak spread at near-zero beta."
        ),
    },
}
MN_BAND_COLOR = "rgba(248, 113, 113, 0.22)"   # light red — distinguishes hedged books
MN_MEDIAN_COLOR = "#991b1b"
MN_LINE_COLORS = {"SPY": "#ea580c", "QQQ": "#059669"}

for key, meta in MN_CONFIGS.items():
    data = load_config(str(meta["path"]))
    if data is None:
        continue
    any_loaded = True

    st.markdown(f"### {meta['label']}")
    st.caption(meta["blurb"])

    dates = data["dates"]
    p10 = np.array(data["p10_curve"]) * 1000.0
    p90 = np.array(data["p90_curve"]) * 1000.0
    median = np.array(data["median_curve"]) * 1000.0
    spy = np.array(data["spy_curve"]) * 1000.0
    qqq = np.array(data["qqq_curve"]) * 1000.0

    fig = go.Figure()
    fig.add_scatter(x=dates, y=p90, mode="lines", line=dict(width=0),
                    showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=dates, y=p10, mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor=MN_BAND_COLOR,
                    name="10th–90th percentile", hoverinfo="skip")
    fig.add_scatter(x=dates, y=median, mode="lines",
                    line=dict(color=MN_MEDIAN_COLOR, width=3.5), name="Median")
    fig.add_scatter(x=dates, y=spy, mode="lines",
                    line=dict(color=MN_LINE_COLORS["SPY"], width=1.75, dash="dash"),
                    name="SPY")
    fig.add_scatter(x=dates, y=qqq, mode="lines",
                    line=dict(color=MN_LINE_COLORS["QQQ"], width=1.75, dash="dash"),
                    name="QQQ")

    for y_end, color in [(median[-1], MN_MEDIAN_COLOR), (spy[-1], MN_LINE_COLORS["SPY"]),
                         (qqq[-1], MN_LINE_COLORS["QQQ"])]:
        fig.add_annotation(x=dates[-1], y=y_end, text=f"${y_end:,.0f}",
                           showarrow=False, xanchor="left", xshift=8,
                           font=dict(color=color, size=12))

    fig.update_layout(
        height=420, margin=dict(l=10, r=70, t=10, b=10),
        yaxis_title="Portfolio Value ($1,000 invested)",
        yaxis=dict(tickprefix="$", gridcolor="rgba(148, 163, 184, 0.15)"),
        xaxis=dict(gridcolor="rgba(148, 163, 184, 0.15)"),
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True, key=f"{key}_chart")

    summary = data["summary"]
    port = data["stats"]["portfolio"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sims", summary["n_sims"])
    m2.metric("Median final return", f"{summary['median_final_return']:+.1%}")
    m3.metric("% sims positive", f"{summary['pct_sims_positive']:.0%}")
    m4.metric("Beta (vs SPY)", f"{port['beta']:+.2f}")

    rows = []
    for name, label in [("portfolio", meta["portfolio_label"]), ("spy", "SPY"), ("qqq", "QQQ")]:
        s = data["stats"][name]
        rows.append({
            "": label, "CAGR": s["cagr"], "Sharpe": s["sharpe"],
            "Sortino": s["sortino"], "Max DD": s["max_dd"],
            "Beta (vs SPY)": s.get("beta"), "Alpha (vs SPY, ann.)": s.get("alpha"),
        })
    table = pd.DataFrame(rows).set_index("")
    st.dataframe(
        table.style.format({
            "CAGR": "{:+.1%}", "Sharpe": "{:.2f}", "Sortino": "{:.2f}",
            "Max DD": "{:.1%}", "Beta (vs SPY)": "{:.2f}",
            "Alpha (vs SPY, ann.)": "{:+.1%}",
        }),
        use_container_width=True, height=42 + 35 * len(table))
    st.caption(f"Params: `{json.dumps(data['params'])}`. Portfolio stats are the "
              f"mean across all {summary['n_sims']} random-draw sims.")
    st.divider()

INSIDER_REV_PATH = ROOT / "output" / "loop_research" / "insider_revisions_min_perturbation.json"
insider_rev_data = load_config(str(INSIDER_REV_PATH))
if insider_rev_data is not None:
    any_loaded = True
    st.markdown("### Top-20-pool champion — min(insider, revisions), 0.5x noise")
    st.caption(
        "Within the top-20-by-composite-score pool, rank by min(insider parent "
        "score, revisions parent score) descending — both signals must "
        "independently clear a bar (floor/AND logic), not a sum or average. "
        f"Gaussian noise ({insider_rev_data['noise_mult']}x the date's "
        "cross-sectional std) is added to the ranking score before each sim's "
        "top-5 selection so the band reflects genuine ranking-sensitivity, not "
        "just a single deterministic path. Research finding only "
        "(research/loop_research champion.md, promoted 2026-09-12), not wired "
        "into production selection.")

    dates_r = insider_rev_data["dates"]
    p10_r = np.array(insider_rev_data["p10_curve"]) * 1000.0
    p90_r = np.array(insider_rev_data["p90_curve"]) * 1000.0
    median_r = np.array(insider_rev_data["median_curve"]) * 1000.0
    spy_r = np.array(insider_rev_data["spy_curve"]) * 1000.0
    qqq_r = np.array(insider_rev_data["qqq_curve"]) * 1000.0

    fig_r = go.Figure()
    fig_r.add_scatter(x=dates_r, y=p90_r, mode="lines", line=dict(width=0),
                      showlegend=False, hoverinfo="skip")
    fig_r.add_scatter(x=dates_r, y=p10_r, mode="lines", line=dict(width=0),
                      fill="tonexty", fillcolor=BAND_COLOR,
                      name="10th–90th percentile", hoverinfo="skip")
    fig_r.add_scatter(x=dates_r, y=median_r, mode="lines",
                      line=dict(color=MEDIAN_COLOR, width=3.5), name="Median")
    fig_r.add_scatter(x=dates_r, y=spy_r, mode="lines",
                      line=dict(color=LINE_COLORS["SPY"], width=1.75, dash="dash"),
                      name="SPY")
    fig_r.add_scatter(x=dates_r, y=qqq_r, mode="lines",
                      line=dict(color=LINE_COLORS["QQQ"], width=1.75, dash="dash"),
                      name="QQQ")

    for y_end, color in [(median_r[-1], MEDIAN_COLOR), (spy_r[-1], LINE_COLORS["SPY"]),
                         (qqq_r[-1], LINE_COLORS["QQQ"])]:
        fig_r.add_annotation(x=dates_r[-1], y=y_end, text=f"${y_end:,.0f}",
                             showarrow=False, xanchor="left", xshift=8,
                             font=dict(color=color, size=12))

    fig_r.update_layout(
        height=420, margin=dict(l=10, r=70, t=10, b=10),
        yaxis_title="Portfolio Value ($1,000 invested)",
        yaxis=dict(tickprefix="$", gridcolor="rgba(148, 163, 184, 0.15)"),
        xaxis=dict(gridcolor="rgba(148, 163, 184, 0.15)"),
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig_r, use_container_width=True, key="insider_rev_chart")

    summary_r = insider_rev_data["summary"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sims", summary_r["n_sims"])
    m2.metric("Median final return", f"{summary_r['median_final_return']:.2f}x")
    m3.metric("% sims beating SPY", f"{summary_r['pct_sims_beating_spy']:.0%}")
    m4.metric("% sims beating QQQ", f"{summary_r['pct_sims_beating_qqq']:.0%}")

    rows_r = []
    for name, label in [("portfolio", "min(insider, revisions)"), ("spy", "SPY"), ("qqq", "QQQ")]:
        s = insider_rev_data["stats"][name]
        rows_r.append({
            "": label, "CAGR": s["cagr"], "Sharpe": s["sharpe"],
            "Sortino": s["sortino"], "Max DD": s["max_dd"],
            "Beta (vs SPY)": s.get("beta"), "Alpha (vs SPY, ann.)": s.get("alpha"),
        })
    table_r = pd.DataFrame(rows_r).set_index("")
    st.dataframe(
        table_r.style.format({
            "CAGR": "{:+.1%}", "Sharpe": "{:.2f}", "Sortino": "{:.2f}",
            "Max DD": "{:.1%}", "Beta (vs SPY)": "{:.2f}",
            "Alpha (vs SPY, ann.)": "{:+.1%}",
        }),
        use_container_width=True, height=42 + 35 * len(table_r))
    st.caption(f"Params: `{json.dumps(insider_rev_data['params'])}`. Portfolio "
              f"stats are the mean across all {summary_r['n_sims']} noise sims.")
    st.divider()

HIST_FREQS = {
    "weekly": "reset to equal weight on the first session of every week",
    "monthly": "reset to equal weight at each month-end inside the hold",
}
for freq, freq_desc in HIST_FREQS.items():
    hist_data = load_config(str(ROOT / "output" / "monte_carlo_random_book"
                                / f"sleeve_rebalance_hist_{freq}_n5_hold4_sleeves4_sims500.json"))
    if hist_data is None:
        continue
    Freq = freq.capitalize()
    any_loaded = True
    hs = hist_data["summary"]
    st.markdown(f"### {Freq} rebalance vs drift — one result per sleeve hold")
    st.caption(
        f"Every 4-month sleeve hold across all sims (same random books as the top "
        f"section). For each hold, the sleeve's 5 names are run two ways: **drift** "
        f"(equal-weight at entry, then left alone) and **{freq}** ({freq_desc}, "
        f"10 bps on each reset). Each bar "
        f"counts holds by their {freq} − drift return over the 4 months. Right of 0 = "
        f"{freq} rebalancing helped that sleeve (scripts/sleeve_rebalance_histogram.py).")

    HIST_COLOR = "#2563eb"
    grid = "rgba(148, 163, 184, 0.15)"
    diffs = np.array(hist_data["diffs"]) * 100
    lo, hi = np.percentile(diffs, [0.5, 99.5])
    fig_h = go.Figure()
    fig_h.add_histogram(x=diffs[(diffs >= lo) & (diffs <= hi)], nbinsx=80,
                        marker=dict(color=HIST_COLOR, line=dict(color="white", width=1)),
                        name="Sleeve holds",
                        hovertemplate="%{x}%: %{y:,} holds<extra></extra>")
    fig_h.add_vline(x=0, line=dict(color="#64748b", width=1.5, dash="dash"),
                    annotation_text="0 (no difference)", annotation_position="top left",
                    annotation_font_color="#64748b")
    fig_h.add_vline(x=hs["mean_diff"] * 100, line=dict(color="#0f172a", width=2),
                    annotation_text=f"mean {hs['mean_diff']:+.3%}",
                    annotation_position="top right", annotation_font_color="#0f172a")
    fig_h.update_layout(
        height=380, margin=dict(l=10, r=10, t=30, b=10), bargap=0, showlegend=False,
        xaxis=dict(title=f"{Freq} − drift return over the 4-month hold (%)",
                   ticksuffix="%", gridcolor=grid),
        yaxis=dict(title="Number of sleeve holds", gridcolor=grid),
        plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_h, use_container_width=True, key=f"sleeve_hist_{freq}")
    st.caption(f"Middle 99% shown ({hs['n_sleeve_periods']:,} holds total); the "
              "mean line uses all of them.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Mean {freq} − drift", f"{hs['mean_diff']:+.3%}")
    m2.metric(f"Median {freq} − drift", f"{hs['median_diff']:+.3%}")
    m3.metric(f"Holds where {freq} won", f"{hs['pct_positive']:.0%}")
    m4.metric("t-stat (non-overlapping holds)", f"{hs['nonoverlap_t_stat']:.2f}",
              help=f"Holds starting 4 months apart (n={hs['nonoverlap_n']} periods, "
                   "averaged across sims), so periods don't share returns. |t| < 2 "
                   "means the mean isn't distinguishable from 0.")

    # each start month averaged across sims: is it a few lucky periods?
    by_start = pd.Series(hist_data["by_start"]) * 100
    fig_s = go.Figure()
    fig_s.add_bar(x=by_start.index, y=by_start.values,
                  marker=dict(color=np.where(by_start.values >= 0, HIST_COLOR, "#b45309")),
                  hovertemplate="%{x}: %{y:+.2f}%<extra></extra>")
    fig_s.update_layout(
        height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, bargap=0.15,
        yaxis=dict(title=f"Avg {freq} − drift (%)", ticksuffix="%", gridcolor=grid),
        xaxis=dict(title="Hold start month", gridcolor=grid),
        plot_bgcolor="rgba(0,0,0,0)")
    st.markdown("**By hold start month** (averaged across sims)")
    st.plotly_chart(fig_s, use_container_width=True, key=f"sleeve_by_start_{freq}")
    st.caption(f"{Freq} beat drift in {hs['by_start_pct_positive']:.0%} of the "
              f"{hs['n_start_dates']} start months. Blue = {freq} helped, amber = drift "
              "did better. All sims share one 2019–2026 price history, so this chart, "
              "not the sim count, is the real sample size.")
    st.divider()

TAX_PATH = ROOT / "output" / "monte_carlo_random_book" / "tax_hold4_vs_hold12_sims500.json"
tax_data = load_config(str(TAX_PATH))
if tax_data is not None:
    any_loaded = True
    tp = tax_data["params"]
    s4, s12 = tax_data["strategies"]["hold4"], tax_data["strategies"]["hold12"]
    sh = tax_data["strategies"]["hold12_harvest"]
    hv = tax_data["harvest_vs_hold12"]
    sm, smh = tax_data["strategies"]["mix12"], tax_data["strategies"]["mix12_harvest"]
    mv = tax_data["mix_vs"]
    bspy, bqqq = tax_data["bench"]["spy"], tax_data["bench"]["qqq"]
    START = 1000.0

    st.markdown("### Taxes included — 4-month hold vs 1-year hold")
    st.caption(
        f"Taxable account, federal rates {tp['st_rate']:.0%} short-term / "
        f"{tp['lt_rate']:.0%} long-term, {tp['n_sims']} sims of the same 5-name / "
        "4-sleeve random book. Each sleeve lets its names drift between reviews. Every "
        "sleeve is funded on day one (its first period is shortened to set up the "
        "stagger), so no strategy sits in cash at the start. "
        "**4-month hold**: every sale is under a year, so gains are taxed short-term. "
        "**1-year hold**: each sleeve holds 12 months and waits until its positions "
        "have been held more than a year before selling (a few days' delay), so "
        "gains are long-term. Gains and losses net each calendar year (losses carry "
        "forward) and tax is paid the next January. At the end everything is sold "
        "and the remaining gains are taxed by holding period, so after-tax values are "
        "what you would walk away with. SPY/QQQ are buy-and-hold, taxed once at "
        "long-term rates when sold at the end (scripts/monte_carlo_tax.py).")

    rows = [("4-month hold", s4["pre"]["median_final"], s4["post"]["median_final"]),
            ("1-year hold", s12["pre"]["median_final"], s12["post"]["median_final"]),
            ("1-year + loss harvest", sh["pre"]["median_final"], sh["post"]["median_final"]),
            ("12-sleeve mix", sm["pre"]["median_final"], sm["post"]["median_final"]),
            ("12-sleeve mix + harvest", smh["pre"]["median_final"], smh["post"]["median_final"]),
            ("SPY", bspy["final_pre"], bspy["final_post"]),
            ("QQQ", bqqq["final_pre"], bqqq["final_post"])]
    lost4 = (s4["pre"]["median_final"] - s4["post"]["median_final"]) * START
    lost12 = (s12["pre"]["median_final"] - s12["post"]["median_final"]) * START
    drag4 = s4["pre"]["stats"]["cagr"] - s4["post"]["stats"]["cagr"]
    drag12 = s12["pre"]["stats"]["cagr"] - s12["post"]["stats"]["cagr"]
    st.markdown(
        f"**How much the 4-month strategy loses to taxes:** \\${START:,.0f} grows to "
        f"\\${s4['pre']['median_final'] * START:,.0f} before tax but only "
        f"\\${s4['post']['median_final'] * START:,.0f} after: taxes cost "
        f"\\${lost4:,.0f}, or {lost4 / (s4['pre']['median_final'] * START):.0%} of the "
        f"ending value (\\${s4['median_tax_paid'] * START:,.0f} paid to the IRS, the "
        f"rest is growth that money would have earned). CAGR drops from {s4['pre']['stats']['cagr']:.1%} to "
        f"{s4['post']['stats']['cagr']:.1%} ({drag4 * 100:.1f} points a year), because "
        f"{s4['median_st_share']:.0%} of its gains are short-term at "
        f"{tp['st_rate']:.0%}.\n\n"
        f"**Holding a year instead:** taxes cost only \\${lost12:,.0f} "
        f"({drag12 * 100:.1f} points a year), since "
        f"{1 - s12['median_st_share']:.0%} of the gains are long-term. But the 1-year "
        f"hold also earns less before tax ({s12['pre']['stats']['cagr']:.1%} vs "
        f"{s4['pre']['stats']['cagr']:.1%}), because the picks are refreshed less "
        f"often. After tax it ends at \\${s12['post']['median_final'] * START:,.0f} vs "
        f"\\${s4['post']['median_final'] * START:,.0f} for the 4-month hold.\n\n"
        f"**1-year + loss harvesting:** same 1-year book, but any position that is "
        f"below cost the day before its 1-year mark is sold then (a short-term loss), "
        f"while winners still sell after the mark (long-term gain). It harvests "
        f"\\${sh['median_harvested'] * START:,.0f} of losses and trims taxes paid from "
        f"\\${s12['median_tax_paid'] * START:,.0f} to "
        f"\\${sh['median_tax_paid'] * START:,.0f}, but ends "
        f"\\${-hv['mean_final_diff'] * START:,.0f} *lower* on average after tax and "
        f"beats the plain 1-year hold in only {hv['pct_sims_harvest_better']:.0%} of "
        f"sims. Under US netting rules a short-term loss only beats a long-term one when "
        f"it offsets short-term gains taxed at {tp['st_rate']:.0%}, and the 1-year "
        f"strategy has almost none, so both kinds of loss end up offsetting long-term "
        f"gains at {tp['lt_rate']:.0%}. Meanwhile the sold losers' cash sits idle for a "
        f"few days, and a loser re-picked within 30 days loses its deduction to the "
        f"wash-sale rule.\n\n"
        f"**12-sleeve mix:** 12 sleeves of 5 names, 1/12 of the money each. Sleeves "
        f"1, 3, …, 11 hold 4 months (short-term gains); sleeves 2, 4, …, 12 hold a year. "
        f"With loss harvesting in the 1-year sleeves, their short-term losses offset the "
        f"4-month sleeves' gains taxed at {tp['st_rate']:.0%}, so the harvesting finally "
        f"pays: it adds \\${mv['mix12']['mean_final_diff'] * START:,.0f} over the same "
        f"mix without harvesting and wins in {mv['mix12']['pct_sims_better']:.0%} of "
        f"sims. But the mix as a whole ends "
        f"\\${-mv['hold12']['mean_final_diff'] * START:,.0f} below the plain 1-year "
        f"hold on average (better in only {mv['hold12']['pct_sims_better']:.0%} of sims). "
        f"Half its money is in 4-month sleeves, so {smh['median_st_share']:.0%} of its "
        f"gains are still short-term, and the \\${smh['median_harvested'] * START:,.0f} "
        f"of harvested losses covers only a small part of them.")

    PRE_COLOR, POST_COLOR = "#94a3b8", "#2563eb"
    grid = "rgba(148, 163, 184, 0.15)"
    labels = [r[0] for r in rows]
    fig_t = go.Figure()
    for tag, idx, color in (("Before tax", 1, PRE_COLOR), ("After tax", 2, POST_COLOR)):
        vals = [r[idx] * START for r in rows]
        fig_t.add_bar(x=labels, y=vals, name=tag, marker=dict(color=color),
                      text=[f"${v:,.0f}" for v in vals], textposition="outside",
                      hovertemplate=f"%{{x}} {tag.lower()}: $%{{y:,.0f}}<extra></extra>")
    fig_t.update_layout(
        barmode="group", bargap=0.3, bargroupgap=0.08, height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title=f"Median ending value (${START:,.0f} invested)", tickprefix="$",
                   gridcolor=grid),
        plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_t, use_container_width=True, key="tax_bars")

    dates_t = tax_data["dates"]
    fig_c = go.Figure()
    for name, curve, color in (("SPY", bspy["curve_post"], BENCH_COLOR["SPY"]),
                               ("QQQ", bqqq["curve_post"], BENCH_COLOR["QQQ"])):
        fig_c.add_scatter(x=dates_t, y=np.array(curve) * START, mode="lines", name=name,
                          line=dict(color=color, width=1.5, dash="dash"),
                          hovertemplate=f"{name}: $%{{y:,.0f}}<extra></extra>")
    for name, sd, color in (("4-month hold", s4, "#db2777"), ("1-year hold", s12, POST_COLOR)):
        fig_c.add_scatter(x=dates_t, y=np.array(sd["post"]["median_curve"]) * START,
                          mode="lines", name=f"{name} (median)",
                          line=dict(color=color, width=2),
                          hovertemplate=f"{name}: $%{{y:,.0f}}<extra></extra>")
    fig_c.update_layout(
        height=400, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
        yaxis=dict(title=f"After-tax value (${START:,.0f} invested)", tickprefix="$",
                   gridcolor=grid),
        xaxis=dict(gridcolor=grid), plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.1))
    st.markdown("**After-tax growth** (median across sims)")
    st.plotly_chart(fig_c, use_container_width=True, key="tax_curves")
    st.caption("Taxes come out each January, so the lines step down then. The drop at "
              "the very end is the final cash-out tax on positions still held.")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("1-year beats 4-month after tax",
              f"{tax_data['pct_sims_hold12_beats_hold4_after_tax']:.0%} of sims")
    m2.metric("1-year beats SPY after tax",
              f"{tax_data['pct_sims_hold12_beats_spy_after_tax']:.0%} of sims")
    m3.metric("4-month beats SPY after tax",
              f"{tax_data['pct_sims_hold4_beats_spy_after_tax']:.0%} of sims")
    m4.metric("Harvest beats plain 1-year after tax",
              f"{hv['pct_sims_harvest_better']:.0%} of sims")
    m5.metric("Mix + harvest beats plain 1-year after tax",
              f"{mv['hold12']['pct_sims_better']:.0%} of sims")

    trows = []
    for name, sd in (("4-month hold", s4), ("1-year hold", s12),
                     ("1-year + loss harvest", sh), ("12-sleeve mix", sm),
                     ("12-sleeve mix + harvest", smh)):
        trows.append({"": name, "CAGR before tax": sd["pre"]["stats"]["cagr"],
                      "CAGR after tax": sd["post"]["stats"]["cagr"],
                      "Tax drag (pts/yr)": (sd["pre"]["stats"]["cagr"]
                                            - sd["post"]["stats"]["cagr"]) * 100,
                      "Taxes paid": sd["median_tax_paid"] * START,
                      "Gains short-term": sd["median_st_share"],
                      "Sharpe after tax": sd["post"]["stats"]["sharpe"],
                      "Max DD after tax": sd["post"]["stats"]["max_dd"]})
    for name, b in (("SPY", bspy), ("QQQ", bqqq)):
        trows.append({"": name, "CAGR before tax": b["stats_pre"]["cagr"],
                      "CAGR after tax": b["stats_post"]["cagr"],
                      "Tax drag (pts/yr)": (b["stats_pre"]["cagr"]
                                            - b["stats_post"]["cagr"]) * 100,
                      "Taxes paid": b["tax_paid"] * START, "Gains short-term": 0.0,
                      "Sharpe after tax": b["stats_post"]["sharpe"],
                      "Max DD after tax": b["stats_post"]["max_dd"]})
    ttable = pd.DataFrame(trows).set_index("")
    st.dataframe(
        ttable.style.format({
            "CAGR before tax": "{:+.1%}", "CAGR after tax": "{:+.1%}",
            "Tax drag (pts/yr)": "{:.1f}", "Taxes paid": "${:,.0f}",
            "Gains short-term": "{:.0%}", "Sharpe after tax": "{:.2f}",
            "Max DD after tax": "{:.1%}",
        }),
        use_container_width=True, height=42 + 35 * len(ttable))
    st.caption(
        f"Medians across sims; {tp['years']:.1f} years; {tp['cost_bps']:.0f} bps per "
        "trade. Wash-sale rule applied (30 days either side). The 1-year hold's few "
        "short-term gains all come from the final cash-out "
        "(positions bought in the last year); before that it realizes none. Dividends are "
        "treated as reinvested and not taxed yearly, which flatters every line slightly, "
        "buy-and-hold most. The backtest starts 2020-01-31, a month before the COVID "
        "crash; like every section on this page, all strategies are fully invested "
        "from day one and take the full crash, the same as SPY.")
    st.divider()

if not any_loaded:
    empty_state(
        "No Monte Carlo results yet.",
        "Run `python scripts/monte_carlo_random_book.py --n 5 --hold 2 --refresh 5 "
        "--sims 500` and `python scripts/monte_carlo_managed_book.py --mode random "
        "--k 5 --sims 200` first — see docs/monte_carlo_book_construction.md.")
    st.stop()

st.caption("See docs/monte_carlo_book_construction.md for the full writeup, "
          "other saved configs, and caveats. Not wired into the live "
          "production composite/scoring pipeline.")
