"""Build the LIVE stock book from the latest scoring run, per the locked spec
(output/ablation/PORTFOLIO_SUMMARY.md): top-25% by composite -> cap5 weights
(PIT free-float market cap, 5% single-name cap) -> cap_match sector overlay
(book sector weights matched to the cap-weighted scored universe). The VIX
tilt is already reflected in the live composite (engine weights; neutral band
15-23 is a no-op anyway).

Usage: python scripts/finalist_toolkit/build_live_book.py [dollars]
Writes output/ablation/live_book_<date>.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.db import get_db  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
POSITION_CAP = 0.05
FINAL_CAP = 0.10   # post-overlay per-name cap, redistributed within sector
TOP_PCT = 0.25
SHARES_LAG_DAYS = 10


def waterfill_cap(w: pd.Series, cap: float) -> pd.Series:
    w = w / w.sum()
    for _ in range(50):
        over = w[w > cap + 1e-12]
        if over.empty:
            break
        under = w.index[w < cap - 1e-12]
        excess = float((over - cap).sum())
        w.loc[over.index] = cap
        if len(under) == 0 or w.loc[under].sum() <= 0:
            break
        w.loc[under] += excess * w.loc[under] / w.loc[under].sum()
    return w / w.sum()


def main(dollars: float) -> None:
    uni = pd.read_csv(ROOT / "output" / "scored_universe_latest.csv")
    uni = uni.dropna(subset=["composite_score"])

    with get_db() as db:
        px = db.query_df(
            "SELECT ticker, date, close FROM daily_prices "
            "WHERE date = (SELECT MAX(date) FROM daily_prices)")
        asof = px["date"].iloc[0]
        cutoff = (pd.Timestamp(asof) - pd.Timedelta(days=SHARES_LAG_DAYS)).strftime("%Y-%m-%d")
        fl = db.query_df(
            "SELECT ticker, float_shares FROM short_interest s "
            "WHERE float_shares > 0 AND date = (SELECT MAX(date) FROM short_interest "
            "  WHERE ticker = s.ticker AND float_shares > 0 AND date <= ?)",
            (cutoff,))

    price = px.set_index("ticker")["close"]
    floats = fl.set_index("ticker")["float_shares"]
    uni = uni.set_index("ticker")
    caps = (floats * price).reindex(uni.index)
    cov = caps.notna().mean()
    if cov < 0.9:
        raise RuntimeError(f"PIT cap coverage only {cov:.0%} of scored universe")
    caps = caps.fillna(caps.median())
    sectors = uni["sector"].fillna("Unknown")

    # top 25% by composite; hold Alphabet via GOOGL only (GOOG stays in the
    # universe for the sector cap_match targets, just isn't bought)
    k = int(round(len(uni) * TOP_PCT))
    book = uni["composite_score"].sort_values(ascending=False).head(k).index
    book = book.drop("GOOG", errors="ignore")

    # cap5 base weights
    w = waterfill_cap(caps.reindex(book), POSITION_CAP)

    # cap_match sector overlay vs the full scored universe
    tgt = caps.groupby(sectors).sum()
    tgt = tgt / tgt.sum()
    cur = w.groupby(sectors.reindex(book)).sum()
    scale = (tgt.reindex(cur.index) / cur).fillna(0.0)
    w = w * sectors.reindex(book).map(scale)
    w = w / w.sum()

    # final within-sector cap: clamp any name to FINAL_CAP, redistribute the
    # excess to same-sector names so sector totals stay cap_matched
    sec = sectors.reindex(book)
    for s in sec.unique():
        grp = w.loc[sec.index[sec == s]]
        tot = float(grp.sum())
        if tot > 0:
            w.loc[grp.index] = waterfill_cap(grp / tot, FINAL_CAP / tot) * tot

    out = pd.DataFrame({
        "sector": sectors.reindex(book),
        "composite_score": uni["composite_score"].reindex(book),
        "weight_pct": (w * 100).round(3),
        "dollars": (w * dollars).round(2),
        "price": price.reindex(book).round(2),
    })
    out["shares"] = (out["dollars"] / out["price"]).round(4)

    # 0.01-share order grid: floor to the grid, then spend the leftover cash in
    # 0.01-share steps, always on the name furthest below its dollar target
    step = 0.01
    tgt_d = out["dollars"]
    sh = (tgt_d / out["price"] / step).astype(int) * step
    cash = dollars - float((sh * out["price"]).sum())
    while True:
        deficit = tgt_d - sh * out["price"]
        cost = out["price"] * step
        # abs-error reduction from buying one more 0.01-share step
        gain = cost.where(deficit >= cost, 2 * deficit - cost)
        cand = gain[(cost <= cash) & (gain > 0)]
        if cand.empty:
            break
        t = cand.idxmax()
        sh.loc[t] += step
        cash -= float(out.loc[t, "price"]) * step
    out["shares_2dp"] = sh.round(2)
    out["dollars_2dp"] = (sh * out["price"]).round(2)

    out = out.sort_values("weight_pct", ascending=False)
    dest = ROOT / "output" / "ablation" / f"live_book_{asof}.csv"
    out.to_csv(dest, index_label="ticker")

    print(f"as-of {asof}  names={len(out)}  ${dollars:,.0f}  "
          f"effN={1.0 / float((w ** 2).sum()):.1f}  max w={w.max():.2%}")
    print(f"wrote {dest}")

    held = out[out["shares_2dp"] > 0]
    drift = (out["dollars_2dp"] - out["dollars"]).abs().sum() / dollars
    print(f"\n0.01-share grid: {len(held)} names held (of {len(out)}), "
          f"invested ${held['dollars_2dp'].sum():,.2f}, cash left ${cash:,.2f}, "
          f"total abs drift {drift:.1%} of book")
    dropped = out[out["shares_2dp"] == 0]
    print(f"dropped (target too small for 0.01 sh): {len(dropped)} names, "
          f"${dropped['dollars'].sum():,.2f} of targets")
    print("\nsector weights (book vs cap_match target):")
    cmp = pd.DataFrame({"book": w.groupby(sectors.reindex(book)).sum(),
                        "target": tgt}).fillna(0.0)
    print((cmp * 100).round(1).sort_values("target", ascending=False))


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 5000.0)
