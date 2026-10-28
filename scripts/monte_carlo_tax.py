"""After-tax Monte Carlo: the 5-name / 4-sleeve random book held 4 months vs
held 12 months (plus a loss-harvesting variant), in a taxable account.

Same random selection as monte_carlo_random_book.py (seed s = seed s there),
refresh=5, 4 staggered sleeves. Each sleeve is its own sub-account that
splits its value equally across its names at each review and then drifts
(no in-hold rebalancing -- per sleeve_rebalance_histogram.py it adds nothing).
A name re-drawn into the same sleeve is kept (trimmed/topped up to equal
weight) rather than sold and rebought.

Tax model (US federal, taxable account):
  * lot-level: each buy opens a lot (date, basis); sells draw pro-rata from
    the name's lots in that sleeve
  * a lot's gain is long-term only if sold MORE than 1 year after purchase
    (sale date > purchase date + 1 calendar year), else short-term
  * --hold 12 waits each sleeve's trade until every lot it holds clears that
    line (first trading day after the anniversary -- a few days' delay), so
    all its gains are long-term; --hold 4 trades on schedule (all short-term)
  * every sleeve is funded on day one; its first period is shortened so later
    reviews land on its staggered month (no idle cash while the book ramps up)
  * mix12: 12 sleeves; odd-numbered sleeves hold 4 months (short-term), even-
    numbered hold 12 months. mix12_harvest also harvests losers in the
    12-month sleeves, so those short-term losses can offset the 4-month
    sleeves' short-term gains (all sleeves net together each year)
  * hold12_harvest: same 1-year book, but on the trading day before each lot's
    1-year line, any lot below cost (judged on the prior close) is sold, so the
    loss is short-term; winners sell after the line as before. The cash waits
    in the sleeve for its next review
  * realized gains net within each calendar year (losses carry forward,
    offsetting short-term first); tax is paid on the first trading day of the
    next year out of each sleeve's cash pro-rata to sleeve value (cash may go
    briefly negative; no forced sales, no interest)
  * at the end the whole book is liquidated and remaining gains taxed by lot
    age (mark-to-liquidation), so after-tax numbers are walk-away values
  * SPY/QQQ: buy-and-hold, single lot, all long-term at liquidation
Approximations: adj_close = total return, so dividends are deferred rather
than taxed yearly (slightly favorable to every line, most to buy-and-hold);
wash sales applied (30 days either side; disallowed loss moves into the
replacement lot's basis). Costs: COST_BPS per side.

Usage: python scripts/monte_carlo_tax.py [--sims 500] [--seed 0]
Writes: output/monte_carlo_random_book/tax_hold4_vs_hold12_sims500.json
"""
from __future__ import annotations

import argparse
import heapq
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backtesting.data_loader import SPY, QQQ                                       # noqa: E402
from research.autoresearch.evaluate import (                                        # noqa: E402
    COST_BPS, EXEC_LAG_DAYS, _snap_forward, bench_returns,
)
from research.strategies.engine import load_data                                   # noqa: E402
from research.walkforward.portfolio import performance_metrics                     # noqa: E402
from scripts.monte_carlo_random_book import _random_book                          # noqa: E402

OUT_DIR = REPO / "output" / "monte_carlo_random_book"
ST_RATE, LT_RATE = 0.32, 0.15
BOOK, REFRESH = 5, 5
MAX_WAIT_DAYS = 10     # a 1-year sleeve waits at most this many sessions for the line


def _four(hold: int) -> list:
    """4 sleeves of one hold length, staggered evenly: [(offset months, hold)]."""
    step = max(hold // 4, 1)
    return [(k * step, hold) for k in range(4)]


# sleeves 1,3,..,11 (offsets 0,2,..,10) hold 4 months; sleeves 2,4,..,12
# (offsets 1,3,..,11) hold 12 months
MIX12 = [(k, 4 if k % 2 == 0 else 12) for k in range(12)]

# (key, sleeve specs, harvest losers in the 12-month sleeves)
CONFIGS = (("hold4", _four(4), False), ("hold12", _four(12), False),
           ("hold12_harvest", _four(12), True),
           ("mix12", MIX12, False), ("mix12_harvest", MIX12, True))


def _staggered_schedule(comp: dict, dates: list, rng: random.Random,
                        specs: list) -> list:
    """Every sleeve is filled on the first date (fully invested from day one);
    sleeve (offset, hold) then reviews at month `offset` (or `hold` if offset
    is 0) and every `hold` months after, so a shortened first period sets up
    the stagger. Returns per date (sleeve holdings, set of sleeves reviewed)."""
    holdings = [[] for _ in specs]
    out = []
    for i, d in enumerate(dates):
        reviewed = set()
        scores = comp.get(d)
        for j, (offset, hold) in enumerate(specs):
            first = offset if offset > 0 else hold
            due = i == 0 or (i >= first and (i - first) % hold == 0)
            if due and scores is not None:
                holdings[j] = _random_book(scores, holdings[j], BOOK, REFRESH, rng)
                reviewed.add(j)
        out.append(([list(h) for h in holdings], reviewed))
    return out


def _tax_due(st: float, lt: float, carry: float, st_rate: float, lt_rate: float):
    """(tax, new carry, taxed st gain, taxed lt gain) for one year's net realized.
    Carried-forward losses are applied as short-term (simplification); a net
    loss in one bucket offsets the other bucket's gain."""
    st += carry
    net = st + lt
    if net <= 0:
        return 0.0, net, 0.0, 0.0
    if st < 0:
        st_g, lt_g = 0.0, net
    elif lt < 0:
        st_g, lt_g = net, 0.0
    else:
        st_g, lt_g = st, lt
    return st_rate * st_g + lt_rate * lt_g, 0.0, st_g, lt_g


def _simulate(schedule: list, px: np.ndarray, col: dict, exec_pos: list,
              lt_pos: np.ndarray, wash_lo: np.ndarray, wash_hi: np.ndarray,
              year_pos: set, st_rate: float, lt_rate: float, wait_lt: list,
              harvest: list) -> dict:
    """Per-sleeve flags. wait_lt[j]: delay sleeve j's review trade until its
    lots clear the 1-year line, if that is at most MAX_WAIT_DAYS away (a
    shortened first period trades on schedule). harvest[j]: the trading day
    before each of sleeve j's lots reaches the line, sell it if it is below
    cost (decided on the prior close), so the loss is short-term; winners
    stay and sell after the line as usual."""
    cost = COST_BPS / 1e4
    S = len(schedule[0][0])
    lots = [[] for _ in range(S)]          # per sleeve: [col, value, basis, buy_pos]
    cash = np.full(S, 1.0 / S)
    yr = {"st": 0.0, "lt": 0.0}
    carry = 0.0
    tot = {"tax": 0.0, "st_gain": 0.0, "lt_gain": 0.0, "harvested": 0.0, "wash": 0.0}
    pending_wash: dict[int, list] = {}     # col -> [[last pos of 30-day window, loss, bucket]]
    nav_out = np.zeros(len(exec_pos))
    monthly_at = {p: i for i, p in enumerate(exec_pos)}
    heap = [(p, 0, -1, None) for p in sorted(set(exec_pos) | year_pos)]
    heapq.heapify(heap)
    last = exec_pos[0]

    def nav_sleeve(j):
        return sum(l[1] for l in lots[j]) + cash[j]

    def book(c, gain, p, bucket, selling):
        """Record a realized gain/loss, applying the wash-sale rule: a loss is
        disallowed (added to the replacement lot's basis) if the same name was
        bought within 30 days before the sale, or is bought within 30 days after."""
        if gain >= 0:
            yr[bucket] += gain
            return
        loss = -gain
        repl = [l for jj in range(S) for l in lots[jj]
                if l[0] == c and l[3] >= wash_lo[p] and l[1] > 1e-12
                and not any(l is x for x in selling)]
        if repl:
            max(repl, key=lambda l: l[3])[2] += loss
            tot["wash"] += loss
            return
        yr[bucket] -= loss
        pending_wash.setdefault(c, []).append([wash_hi[p], loss, bucket])

    def on_buy(c, lot, p):
        keep = []
        for w in pending_wash.get(c, []):
            if p <= w[0]:                  # rebought inside the window: undo the loss
                yr[w[2]] += w[1]
                lot[2] += w[1]
                tot["wash"] += w[1]
            else:
                keep.append(w)
        pending_wash[c] = keep

    def realize(j, c, amount, p):
        """Sell `amount` of name c from sleeve j pro-rata across its lots."""
        mine = [l for l in lots[j] if l[0] == c]
        held = sum(l[1] for l in mine)
        frac = min(amount / held, 1.0) if held > 0 else 0.0
        for l in mine:
            take, b = l[1] * frac, l[2] * frac
            l[1] -= take
            l[2] -= b
            book(c, take - b, p, "lt" if p >= lt_pos[l[3]] else "st", mine)
        lots[j] = [l for l in lots[j] if l[1] > 1e-12]
        cash[j] += held * frac * (1 - cost)

    def trade(j, new_cols, p):
        V = nav_sleeve(j)
        k = len(new_cols)
        tgt = V / k if k else 0.0
        cur = {}
        for l in lots[j]:
            cur[l[0]] = cur.get(l[0], 0.0) + l[1]
        for c, v in list(cur.items()):
            excess = v - (tgt if c in new_cols else 0.0)
            if excess > 1e-12:
                realize(j, c, excess, p)
        buys = {c: tgt - cur.get(c, 0.0) for c in new_cols}
        buys = {c: a for c, a in buys.items() if a > 1e-12}
        need = sum(buys.values())
        s = min(1.0, cash[j] / need) if need > 0 else 0.0
        for c, a in buys.items():
            lot = [c, a * s * (1 - cost), a * s * (1 - cost), p]
            lots[j].append(lot)
            on_buy(c, lot, p)
            if harvest[j]:
                heapq.heappush(heap, (int(lt_pos[p]) - 1, 2, j, p))
        cash[j] -= need * s

    def harvest_losers(j, b, p):
        for l in [l for l in lots[j] if l[3] == b]:
            r = px[p - 1, l[0]] / px[p, l[0]]
            prev_val = l[1] * r if np.isfinite(r) else l[1]
            if prev_val < l[2]:
                gain = l[1] - l[2]
                cash[j] += l[1] * (1 - cost)
                lots[j] = [x for x in lots[j] if x is not l]
                tot["harvested"] += max(-gain, 0.0)
                book(l[0], gain, p, "st", [l])

    while heap:
        p, kind, j, payload = heapq.heappop(heap)
        if p != last:                                   # drift every lot to p
            for jj in range(S):
                for l in lots[jj]:
                    r = px[p, l[0]] / px[last, l[0]]
                    if np.isfinite(r):
                        l[1] *= r
            last = p
        if kind == 1:                                   # queued sleeve trade
            trade(j, payload, p)
            continue
        if kind == 2:                                   # pre-anniversary loss harvest
            if p < exec_pos[-1]:
                harvest_losers(j, payload, p)
            continue
        if p in year_pos and p != exec_pos[0]:          # pay last year's tax
            tax, carry, st_g, lt_g = _tax_due(yr["st"], yr["lt"], carry, st_rate, lt_rate)
            yr["st"] = yr["lt"] = 0.0
            tot["tax"] += tax
            tot["st_gain"] += st_g
            tot["lt_gain"] += lt_g
            if tax:
                navs = np.array([nav_sleeve(jj) for jj in range(S)])
                cash -= tax * navs / navs.sum()
        m = monthly_at.get(p)
        if m is None:
            continue
        nav_out[m] = sum(nav_sleeve(jj) for jj in range(S))
        if m == len(exec_pos) - 1:
            break
        holdings, reviewed = schedule[m]
        for jj in sorted(reviewed):
            new_cols = [col[t] for t in holdings[jj]]
            td = p
            if wait_lt[jj] and lots[jj]:
                line = int(max(lt_pos[l[3]] for l in lots[jj]))
                if line - p <= MAX_WAIT_DAYS:
                    td = max(p, line)
            heapq.heappush(heap, (td, 1, jj, new_cols))

    # terminal liquidation at the last exec date, taxed with this year's realized
    p_end = exec_pos[-1]
    st_before = tot["st_gain"] + max(yr["st"], 0.0)     # ST realized before liquidation
    for jj in range(S):
        for l in lots[jj]:
            yr["lt" if p_end >= lt_pos[l[3]] else "st"] += l[1] - l[2]
    tax, _, st_g, lt_g = _tax_due(yr["st"], yr["lt"], carry, st_rate, lt_rate)
    tot["tax"] += tax
    tot["st_gain"] += st_g
    tot["lt_gain"] += lt_g
    final = nav_out[-1] - tax
    return {"nav": nav_out, "final": final, "liq_year_tax": tax,
            "st_gain_pre_liq": st_before, **tot}


def _stats(pr: pd.Series, spy: pd.Series) -> dict:
    m = performance_metrics(pr, hold_months=1, benchmarks={"SPY": spy})
    return {"cagr": m["cagr"], "sharpe": m["sharpe"], "sortino": m["sortino"],
            "max_dd": m["max_drawdown"], "beta": m.get("spy_beta"),
            "alpha": m.get("spy_alpha")}


def _curve_series(nav: np.ndarray, final: float, dates: list) -> pd.Series:
    """Monthly returns with the walk-away (after liquidation tax) value as the
    last point."""
    v = nav.copy()
    v[-1] = final
    return pd.Series(v[1:] / v[:-1] - 1.0, index=dates)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("loading composite scores + price matrix ...", flush=True)
    data = load_data()
    rebal = list(data.rebal_dates)
    index = data.matrix.index
    ts = pd.DatetimeIndex(pd.to_datetime(index))
    all_dates = rebal[:-1]
    spy = bench_returns(data.matrix, SPY, rebal)
    qqq = bench_returns(data.matrix, QQQ, rebal)

    pool = sorted({t for s in data.comp.values() for t in s.dropna().index}
                  & set(data.matrix.columns))
    col = {t: i for i, t in enumerate(pool)}
    px = data.matrix[pool].to_numpy(dtype=float)
    exec_pos = [min(index.get_loc(_snap_forward(index, d)) + EXEC_LAG_DAYS, len(index) - 1)
                for d in rebal]
    # first position whose date is MORE than 1 calendar year after position p
    lt_pos = ts.searchsorted(ts + pd.DateOffset(years=1), side="right")
    # 30-calendar-day wash-sale window around each position
    wash_lo = ts.searchsorted(ts - pd.Timedelta(days=30), side="left")
    wash_hi = ts.searchsorted(ts + pd.Timedelta(days=30), side="right") - 1
    year_pos = {p for p in range(exec_pos[0] + 1, exec_pos[-1] + 1)
                if ts[p].year != ts[p - 1].year}

    years = (ts[exec_pos[-1]] - ts[exec_pos[0]]).days / 365.25
    results = {}
    for key, specs, hv in CONFIGS:
        wait_lt = [h >= 12 for _, h in specs]
        harvest = [h >= 12 and hv for _, h in specs]
        res = {"pre": {"curves": [], "stats": {k: [] for k in
                                               ("cagr", "sharpe", "sortino", "max_dd", "beta", "alpha")}},
               "post": {"curves": [], "stats": {k: [] for k in
                                                ("cagr", "sharpe", "sortino", "max_dd", "beta", "alpha")}},
               "tax_paid": [], "st_share": [], "final_pre": [], "final_post": [],
               "liq_year_tax": [], "st_gain_pre_liq": [], "harvested": [], "wash": []}
        for s in range(args.sims):
            sched = _staggered_schedule(data.comp, rebal, random.Random(args.seed + s), specs)
            for tag, (st_r, lt_r) in (("pre", (0.0, 0.0)), ("post", (ST_RATE, LT_RATE))):
                out = _simulate(sched, px, col, exec_pos, lt_pos, wash_lo, wash_hi,
                                year_pos, st_r, lt_r, wait_lt, harvest)
                pr = _curve_series(out["nav"], out["final"], all_dates)
                for k, v in _stats(pr, spy).items():
                    res[tag]["stats"][k].append(v)
                res[tag]["curves"].append((1.0 + pr).cumprod().values)
                res[f"final_{tag}"].append(out["final"])
                if tag == "post":
                    res["tax_paid"].append(out["tax"])
                    res["liq_year_tax"].append(out["liq_year_tax"])
                    res["st_gain_pre_liq"].append(out["st_gain_pre_liq"])
                    res["harvested"].append(out["harvested"])
                    res["wash"].append(out["wash"])
                    g = out["st_gain"] + out["lt_gain"]
                    res["st_share"].append(out["st_gain"] / g if g > 0 else 0.0)
            if (s + 1) % 50 == 0:
                print(f"  {key}: sim {s + 1}/{args.sims}", flush=True)
        results[key] = res

    spy_r = spy.reindex(all_dates).fillna(0.0)
    qqq_r = qqq.reindex(all_dates).fillna(0.0)
    bench = {}
    for name, r in (("spy", spy_r), ("qqq", qqq_r)):
        curve = (1.0 + r).cumprod().values
        final_post = curve[-1] - LT_RATE * max(curve[-1] - 1.0, 0.0)
        post = curve.copy()
        post[-1] = final_post
        pr_post = pd.Series(np.diff(np.r_[1.0, post]) / np.r_[1.0, post][:-1], index=all_dates)
        bench[name] = {"curve_pre": curve.tolist(), "curve_post": post.tolist(),
                       "final_pre": float(curve[-1]), "final_post": float(final_post),
                       "tax_paid": float(curve[-1] - final_post),
                       "stats_pre": _stats(r, spy_r), "stats_post": _stats(pr_post, spy_r)}

    out = {"generated_at": pd.Timestamp.utcnow().isoformat(),
           "params": {"book_size": BOOK, "refresh_n": REFRESH,
                      "sleeve_specs": {k: sp for k, sp, _ in CONFIGS},
                      "configs": [c[0] for c in CONFIGS], "n_sims": args.sims, "seed": args.seed,
                      "st_rate": ST_RATE, "lt_rate": LT_RATE, "cost_bps": COST_BPS,
                      "years": years},
           "dates": [str(d) for d in all_dates], "strategies": {}, "bench": bench}
    print(f"\n=== {years:.2f} years, $1 start; medians across {args.sims} sims ===")
    for key, res in results.items():
        entry = {}
        for tag in ("pre", "post"):
            c = np.array(res[tag]["curves"])
            entry[tag] = {"median_curve": np.median(c, axis=0).tolist(),
                          "p10_curve": np.percentile(c, 10, axis=0).tolist(),
                          "p90_curve": np.percentile(c, 90, axis=0).tolist(),
                          "stats": {k: float(np.median(v)) for k, v in res[tag]["stats"].items()},
                          "median_final": float(np.median(res[f"final_{tag}"]))}
        entry["median_tax_paid"] = float(np.median(res["tax_paid"]))
        entry["median_st_share"] = float(np.median(res["st_share"]))
        entry["median_liq_year_tax"] = float(np.median(res["liq_year_tax"]))
        entry["max_st_gain_pre_liq"] = float(np.max(res["st_gain_pre_liq"]))
        entry["median_harvested"] = float(np.median(res["harvested"]))
        entry["median_wash_disallowed"] = float(np.median(res["wash"]))
        entry["sim_final_post"] = res["final_post"]
        out["strategies"][key] = entry
        pre, post = entry["pre"], entry["post"]
        print(f"  {key:<14} pre-tax ${pre['median_final']:.2f} (cagr {pre['stats']['cagr']:.2%})  "
              f"after-tax ${post['median_final']:.2f} (cagr {post['stats']['cagr']:.2%}, "
              f"sharpe {post['stats']['sharpe']:.2f})  tax ${entry['median_tax_paid']:.2f}  "
              f"ST share {entry['median_st_share']:.0%}  final-year+liquidation tax "
              f"${entry['median_liq_year_tax']:.2f}  max ST gain before liquidation "
              f"{entry['max_st_gain_pre_liq']:.3f}  harvested ${entry['median_harvested']:.3f}  "
              f"wash-disallowed ${entry['median_wash_disallowed']:.3f}")
    for name, b in bench.items():
        print(f"  {name.upper():<7} pre-tax ${b['final_pre']:.2f}  after-tax ${b['final_post']:.2f} "
              f"(cagr {b['stats_post']['cagr']:.2%})")
    f4 = np.array(results["hold4"]["final_post"])
    f12 = np.array(results["hold12"]["final_post"])
    out["pct_sims_hold12_beats_hold4_after_tax"] = float((f12 > f4).mean())
    out["pct_sims_hold12_beats_spy_after_tax"] = float((f12 > bench["spy"]["final_post"]).mean())
    out["pct_sims_hold4_beats_spy_after_tax"] = float((f4 > bench["spy"]["final_post"]).mean())
    print(f"  hold12 > hold4 after tax in {out['pct_sims_hold12_beats_hold4_after_tax']:.0%} of sims; "
          f"beats SPY after tax: hold12 {out['pct_sims_hold12_beats_spy_after_tax']:.0%}, "
          f"hold4 {out['pct_sims_hold4_beats_spy_after_tax']:.0%}")
    fh_ = np.array(results["hold12_harvest"]["final_post"])
    diff = fh_ - f12
    out["harvest_vs_hold12"] = {
        "mean_final_diff": float(diff.mean()), "median_final_diff": float(np.median(diff)),
        "pct_sims_harvest_better": float((diff > 0).mean()),
        "mean_cagr_diff": float(np.mean(np.array(results["hold12_harvest"]["post"]["stats"]["cagr"])
                                        - np.array(results["hold12"]["post"]["stats"]["cagr"]))),
        "pct_sims_beats_spy": float((fh_ > bench["spy"]["final_post"]).mean())}

    def paired(a: str, b: str) -> dict:
        fa, fb = np.array(results[a]["final_post"]), np.array(results[b]["final_post"])
        ca = np.array(results[a]["post"]["stats"]["cagr"])
        cb = np.array(results[b]["post"]["stats"]["cagr"])
        d = fa - fb
        return {"mean_final_diff": float(d.mean()), "pct_sims_better": float((d > 0).mean()),
                "mean_cagr_diff": float((ca - cb).mean())}

    out["mix_vs"] = {b: paired("mix12_harvest", b)
                     for b in ("mix12", "hold12", "hold12_harvest", "hold4")}
    out["pct_sims_mix12_harvest_beats_spy_after_tax"] = float(
        (np.array(results["mix12_harvest"]["final_post"]) > bench["spy"]["final_post"]).mean())
    for b, v in out["mix_vs"].items():
        print(f"  mix12_harvest - {b}: mean ${v['mean_final_diff']:+.4f} per $1, better in "
              f"{v['pct_sims_better']:.0%} of sims, CAGR {v['mean_cagr_diff']:+.2%}")
    print(f"  harvest - hold12 after tax: mean ${diff.mean():+.4f} per $1, better in "
          f"{(diff > 0).mean():.0%} of sims, CAGR {out['harvest_vs_hold12']['mean_cagr_diff']:+.2%}")

    out_path = OUT_DIR / f"tax_hold4_vs_hold12_sims{args.sims}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(out, fh, default=lambda x: None if x != x else x)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
