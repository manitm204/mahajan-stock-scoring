"""Adversarial validation battery (Phase D of the 2026-07-18 gap-filling program).

For the locked finalist and any Phase C survivor (bars per PREREGISTRATION.md):
  - bootstrap CIs (paired block bootstrap) for Sharpe/CAGR/alpha, full + search window
  - search-window-only alpha t-stat (the binding pre-registration bar)
  - sub-era splits incl. the 2022-23 search sub-era and stress windows
  - drop-best-year / drop-best-3-months fragility
  - rolling 36-month alpha summary
  - parameter-neighborhood stability (breadth/weighting/sector neighbors)
  - ONE discounted holdout read (2024+), disclosed as previously-consulted

Output: output/validation/VALIDATION.md (+ neighbors.csv)
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

REPO = Path("/home/manit/Desktop/fun_projects/mahajan_hedge_fund")
sys.path.insert(0, str(REPO))

from run_walkforward import PANEL_START, PRICE_END, _load_panel, get_db  # noqa: E402
from research.ablation import load_ablation_data  # noqa: E402
from research.ablation.diagnostics import (bootstrap_ci, drop_best,  # noqa: E402
                                           rolling_alpha)
from research.ablation.engine import (SEARCH_END, AblationConfig,  # noqa: E402
                                      alpha_tstat, simulate_config)

OUT = REPO / "output/validation"
FINALIST = AblationConfig("finalist", top_pct=0.25, weighting="cap5",
                          sector="cap_match", vix_tilt=True)
NEIGHBORS = [
    replace(FINALIST, name="top20", top_pct=0.20),
    replace(FINALIST, name="top30", top_pct=0.30),
    replace(FINALIST, name="w=cap", weighting="cap"),
    replace(FINALIST, name="w=ewcap", weighting="ewcap"),
    replace(FINALIST, name="sec=none", sector="none"),
    replace(FINALIST, name="notilt", vix_tilt=False),
]
STRESS = {"2018Q4": ("2018-09-30", "2018-12-31"),
          "covid": ("2020-01-31", "2020-04-30"),
          "2022bear": ("2021-12-31", "2022-12-31"),
          "2022-23 sub-era": ("2021-12-31", "2023-12-31")}


def battery(name: str, row: dict, ppy: float = 12.0) -> str:
    p, spy = row["_returns"], row["_spy"]
    idx = p.index.astype(str)
    p_s, b_s = p[idx <= SEARCH_END], spy[idx <= SEARCH_END]
    boot_full = bootstrap_ci(p, spy, ppy)
    boot_srch = bootstrap_ci(p_s, b_s, ppy)
    frag = drop_best(p, spy, ppy)
    ra = rolling_alpha(p, spy, ppy, 36)
    lines = [f"### {name}", ""]
    lines.append(f"- search-window alpha t = **{alpha_tstat(p_s, b_s):.2f}** "
                 f"(full-period t = {alpha_tstat(p, spy):.2f})")
    lines.append(f"- bootstrap 90% CI (search): Sharpe {boot_srch['sharpe_ci90'][0]:.2f}"
                 f"..{boot_srch['sharpe_ci90'][1]:.2f}, alpha "
                 f"{boot_srch['alpha_ci90'][0]*100:+.1f}%.."
                 f"{boot_srch['alpha_ci90'][1]*100:+.1f}%  "
                 f"(P[alpha<0] = {boot_srch['p_alpha_neg']:.0%})")
    lines.append(f"- bootstrap 90% CI (full): alpha "
                 f"{boot_full['alpha_ci90'][0]*100:+.1f}%.."
                 f"{boot_full['alpha_ci90'][1]*100:+.1f}%  "
                 f"(P[alpha<0] = {boot_full['p_alpha_neg']:.0%})")
    lines.append(f"- fragility: excess vs SPY {frag['excess_full']*100:+.1f}%/yr full; "
                 f"without best year ({frag['best_year']}) "
                 f"{frag['excess_wo_best_year']*100:+.1f}%; without best 3 months "
                 f"{frag['excess_wo_best3_periods']*100:+.1f}%")
    lines.append(f"- rolling 36m alpha: positive {float((ra > 0).mean()):.0%} of "
                 f"windows, min {ra.min()*100:+.1f}%, last {ra.iloc[-1]*100:+.1f}%")
    for label, (a, b) in STRESS.items():
        m = (idx > a) & (idx <= b)
        if m.sum() >= 2:
            ex = float((p[m] - spy[m]).mean() * 12)
            lines.append(f"- stress {label}: excess vs SPY {ex*100:+.1f}%/yr "
                         f"({int(m.sum())} months)")
    hold = row.get("alpha_holdout")
    lines.append(f"- **discounted holdout (2024+) alpha: {hold*100:+.2f}%** — "
                 "disclosed: this window was consulted by the prior construction "
                 "search; treat as corroboration, not proof")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    panel = _load_panel(False)
    with get_db() as db:
        data = load_ablation_data(panel, db, PANEL_START, PRICE_END,
                                  splits="rolling5y")
    OUT.mkdir(parents=True, exist_ok=True)

    candidates = {"finalist (top25/cap5/capmatch/vix)":
                  simulate_config(data, FINALIST, keep_series=True)}

    # Phase C outcome (2026-07-18): zero of the 10 pre-registered novel-signal
    # configs passed the bars (output/novel_signals/RESULTS.md), so the battery
    # runs on the finalist alone.

    print("[neighbors] parameter stability sweep ...", flush=True)
    nrows = [simulate_config(data, cfg) for cfg in [FINALIST] + NEIGHBORS]
    ndf = pd.DataFrame(nrows)
    ndf.to_csv(OUT / "neighbors.csv", index=False)

    with (OUT / "VALIDATION.md").open("w") as fh:
        fh.write("# Validation battery (2026-07-18)\n\nMonthly grid, net of 10 "
                 "bps/side, vs SPY. Bootstrap = paired circular block bootstrap "
                 "(6-month blocks, 2000 draws).\n\n")
        for name, row in candidates.items():
            fh.write(battery(name, row))
        fh.write("\n## Parameter neighborhood (stability)\n\n")
        cols = ["config", "net_cagr", "sharpe", "alpha", "alpha_t", "alpha_search",
                "alpha_holdout", "ex_spy_e1", "ex_spy_e2", "turnover"]
        fh.write(ndf[cols].to_markdown(index=False, floatfmt=".3f"))
        spread = ndf["alpha_search"].max() - ndf["alpha_search"].min()
        fh.write(f"\n\nSearch-alpha spread across neighbors: {spread*100:.2f}pp; "
                 f"all-positive: {bool((ndf['alpha_search'] > 0).all())}\n")
    print(f"wrote {OUT}/VALIDATION.md + neighbors.csv", flush=True)


if __name__ == "__main__":
    main()
