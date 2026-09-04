"""Layer the frozen VIX literature tilt onto the clean rolling-5y composite and
re-run the depth curve vs SPY. Baseline (no tilt) vs vix-tilt, side by side."""
import pandas as pd

from run_walkforward import _load_panel, get_db, PANEL_START, PRICE_END
from research.walkforward.runner import run_splits
from research.walkforward.splits import resolve_splits
from research.walkforward import portfolio as pf
from research.walkforward.portfolio import performance_metrics
from research.walkforward.compose import build_parent_panel
from research.walkforward.selection import slice_panel
from factors.composite import NEUTRAL, _normalize_parents
from factors.utils import sector_percentile
from factors.vix_tilt import apply_vix_tilt, band_multipliers
from backtesting import data_loader as dl

panel = _load_panel(False)
with get_db() as db:
    matrix = dl.load_price_matrix(db, panel.universe, PANEL_START, PRICE_END)
    sectors = dl.global_sectors(db)
    vixdf = db.query_df("SELECT date, close FROM daily_prices WHERE ticker='VIX' ORDER BY date")
vix = pd.Series(vixdf["close"].values, index=vixdf["date"].astype(str))

def vix_spot(d):
    s = vix[vix.index <= d]
    return float(s.iloc[-1]) if len(s) else None

run = run_splits(panel, resolve_splits("rolling5y"), matrix, sectors)

def composite_vixtilt(panel, splits_data, sectors, mode_tilt=True,
                      norm_mode="zscore", target_std=20.0, min_obs=5):
    out = {}
    for sd in splits_data:
        cfg = sd["config"]
        test_rebals = list(sd["scores"].keys())
        ppanel = build_parent_panel(slice_panel(panel, test_rebals), cfg.sub_weights)
        for d in test_rebals:
            frame = ppanel.scores.get(d)
            if frame is None:
                continue
            pw = apply_vix_tilt(cfg.parent_weights, vix_spot(d))[0] if mode_tilt \
                else dict(cfg.parent_weights)
            cols = [p for p in pw if p in frame.columns]
            blend = _normalize_parents(frame[cols], norm_mode, target_std)
            comp = pd.Series(0.0, index=frame.index); used = 0.0
            for k in cols:
                comp += pw[k] * blend[k].fillna(NEUTRAL); used += pw[k]
            if used > 0:
                comp /= used
            secs = sectors.reindex(frame.index).fillna("Unknown")
            out[d] = sector_percentile(comp, secs, higher_is_better=True, min_obs=min_obs)
    return out

base_scores = run.pooled_scores                       # no-tilt (matches earlier runs)
tilt_scores = composite_vixtilt(panel, run.splits_data, sectors, mode_tilt=True)

# --- sanity: does the tilt actually move weights? pick highest & lowest VIX test date ---
test_dates = sorted(base_scores)
hi = max(test_dates, key=lambda d: vix_spot(d) or 0)
lo = min(test_dates, key=lambda d: vix_spot(d) or 999)
# find the split config covering each date
def cfg_for(d):
    for sd in run.splits_data:
        if d in sd["scores"]:
            return sd["config"]
    return None
for tag, d in (("HIGH-VIX", hi), ("LOW-VIX", lo)):
    c = cfg_for(d); v = vix_spot(d); mm, vm = band_multipliers(v)
    b = c.parent_weights; t = apply_vix_tilt(b, v)[0]
    def g(w, p): return w.get(p, 0.0)
    print(f"[{tag}] {d} VIX={v:.1f} mom×{mm:.2f} val×{vm:.2f}  "
          f"mom {g(b,'momentum'):.2f}->{g(t,'momentum'):.2f} | "
          f"qual {g(b,'quality'):.2f}->{g(t,'quality'):.2f} | "
          f"val {g(b,'value'):.2f}->{g(t,'value'):.2f}")

def sweep(scores, mode):
    rows = []
    spy = None
    for pct in [round(0.1*i, 1) for i in range(1, 11)]:
        sim = pf.simulate(scores, matrix, sectors, top_pct=pct, mode=mode)
        m = sim.metrics
        rows.append((f"top{int(pct*100)}%", m["cagr"], m["sharpe"], m["sortino"], m["max_drawdown"]))
        if spy is None:
            sm = performance_metrics(sim.spy_period_returns, hold_months=1)
            spy = ("SPY", sm["cagr"], sm["sharpe"], sm["sortino"], sm["max_drawdown"])
    return rows, spy

for mode in ("sector_neutral",):
    base_rows, spy = sweep(base_scores, mode)
    tilt_rows, _ = sweep(tilt_scores, mode)
    print(f"\n=== depth curve ({mode}) : baseline vs VIX-tilt ===")
    print(f"{'book':>8} | {'CAGR base':>9} {'CAGR tilt':>9} | {'Shp base':>8} {'Shp tilt':>8} | "
          f"{'Srt base':>8} {'Srt tilt':>8} | {'DD base':>7} {'DD tilt':>7}")
    print("-"*92)
    for b, t in zip(base_rows, tilt_rows):
        print(f"{b[0]:>8} | {b[1]*100:+8.2f}% {t[1]*100:+8.2f}% | {b[2]:8.2f} {t[2]:8.2f} | "
              f"{b[3]:8.2f} {t[3]:8.2f} | {b[4]*100:+6.1f}% {t[4]*100:+6.1f}%")
    print("-"*92)
    print(f"{'SPY':>8} | {spy[1]*100:+8.2f}% {'':>9} | {spy[2]:8.2f} {'':>8} | "
          f"{spy[3]:8.2f} {'':>8} | {spy[4]*100:+6.1f}%")
