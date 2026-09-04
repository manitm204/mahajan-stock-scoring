"""Trade-level stats for the two production option structures (explain request)."""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/manit/Desktop/fun_projects/mahajan_hedge_fund")
import numpy as np, pandas as pd
import options_zoo as oz
from ivr_sweep import get_env, run_flex, CS, IC

env = get_env()
tdays, spy, vixd = env["tdays"], env["spy"], env["vixd"]
td_dt = pd.to_datetime(tdays)
vser = pd.Series(vixd)
lo, hi = vser.rolling(252).min(), vser.rolling(252).max()
ivr = ((vser - lo) / (hi - lo).replace(0, np.nan) * 100).fillna(50).values
i0 = max(252, int(np.searchsorted(np.array(tdays), "2017-01-01")))

def stats(trades, lbl):
    n = len(trades); win = np.mean([t["win"] for t in trades])
    roi = np.array([t["roi"] for t in trades])
    cr = np.mean([t["credit"] for t in trades]); ba = np.mean([t["basis"] for t in trades])
    dd = np.mean([t["days"] for t in trades])
    print(f"{lbl:26}{n:>5}{win*100:>7.0f}%{roi.mean()*100:>8.1f}%{roi.min()*100:>8.1f}%"
          f"{roi.max()*100:>8.1f}%{cr:>8.2f}{ba:>8.2f}{dd:>7.0f}")

# CS always-on; IC only fires when IVR>=30
cs_t, _ = run_flex(lambda v: (*CS, "cs"), tdays, td_dt, spy, vixd, ivr, i0, return_trades=True) \
    if "return_trades" in run_flex.__code__.co_varnames else (None, None)

# run_flex may not expose trades; call run_strategy directly instead
cs_trades, _ = oz.run_strategy(oz.b_callspread, 30, None, None, None, tdays, td_dt, spy, vixd, ivr, i0)
ic_trades, _ = oz.run_strategy(oz.b_condor, 45, 0.5, None, 30, tdays, td_dt, spy, vixd, ivr, i0)

print(f"\n{'structure':26}{'n':>5}{'win%':>7}{'avgROI':>8}{'worst':>8}{'best':>8}"
      f"{'credit':>8}{'risk$':>8}{'days':>7}   (ROI = on max-loss basis)")
stats(cs_trades, "Call spread (always on)")
stats(ic_trades, "Iron condor (IVR>=30)")

# show a few example entries: spot, short-call strike, credit
print("\nExample call-spread entries (first 5 after 2020):")
print(f"{'date':>12}{'SPY':>8}{'VIX':>6}{'shortK':>8}{'longK':>8}{'credit':>8}{'width':>7}")
c = 0
for i in range(i0, len(tdays)-2):
    if tdays[i] < "2020-01-01": continue
    S0, a = spy[i], vixd[i]/100.0
    K = oz.strike_at("c", S0, a, 30/365, 0.04, 0.15)
    V0 = oz.val([(-1,"c",K),(1,"c",K+5)], S0, a, 30/365, 0.04)
    print(f"{tdays[i]:>12}{S0:>8.1f}{vixd[i]:>6.1f}{K:>8.1f}{K+5:>8.1f}{-V0:>8.2f}{5.0:>7.1f}")
    c += 1
    if c >= 5: break
