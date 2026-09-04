"""Loser-screen walk-forward study (research-only).

Question: is the composite worth more as a *loser-avoider* than a winner-picker?
The walk-forward validation (output/walkforward/RECOMMENDATION.md) found hump-shaped
OOS quantiles — the bottom of the ranking is reliably bad even where the top is not —
and the only persistently positive parents (short, quality) are "avoid bad companies"
signals. So instead of holding the top decile, hold a broad equal-weight book and
*exclude* the model's worst-ranked names.

Reuses the vixtilt study machinery unchanged: 19 semiannual PIT windows 2017-H1→
2026-H1, frozen rolling-5y baselines from cache/vixtilt/ (select_config sub/parent
weights), production-faithful composite, monthly equal-weight rebalance, 10bps/side.
The composite is identical across variants — only the book-construction rule differs.

PRE-REGISTERED 2026-07-14 (before the first run; no retuning after):
  Primary variant: ``screen20`` — broad book minus the bottom 20% by composite.
  Reference:       ``broad``    — every scored name, same EW/monthly/cost rules.
  Pass bar (both required):
    (a) full-period net Sharpe(screen20) > net Sharpe(broad);
    (b) screen20 net total return beats broad in >= 10 of the 19 windows.
  Context only (not the bar): screen10/screen30 (depth sensitivity), bottom20
  (the excluded names as a book — diagnostic that Q1 actually underperforms),
  top20 (the incumbent construction), SPY comparisons, drawdown stats.
  The clean comparison is screened-vs-broad (identical construction); vs-SPY is
  confounded by the equal-weight small-cap tilt and is reported as context.

FOLLOW-UP (v2) PRE-REGISTERED 2026-07-14 (before the first v2 run; no retuning after):
  Confirmatory — cap weighting. ``cap_screen20`` (cap-weighted broad book minus the
  bottom 20% of names by composite, built on the market-cap-covered subset) vs
  ``cap_broad`` (cap-weighted, same subset). This removes the equal-weight
  small/mid-cap tilt and asks the product question directly: does "the index minus
  the model's avoid-list" beat "the index"? The 20% depth is carried over from the
  v1 primary, NOT re-chosen.
  Pass bar (both required):
    (a) full-period net Sharpe(cap_screen20) > net Sharpe(cap_broad);
    (b) cap_screen20 net total return beats cap_broad in >= 10 of the 19 windows.
  Market caps are point-in-time: price(d) x implied shares from the latest annual
  report available >= 90 days before d (production ``implied_shares`` logic —
  reported shares preferred, else net_income / diluted EPS with guards).
  Exploratory (no bar, mapping only): depth curve screen10..70 in both weightings.
  A-priori expectation: interior maximum, declining toward the top-book limit
  (dropping 80% == the top20 book, Sharpe 0.660 in v1). No production depth will
  be chosen from the argmax of this curve; it exists to describe the dose-response.

BLEND (v3) PRE-REGISTERED 2026-07-14 (before the first v3 run; no retuning after):
  Motivation: the yearly breakdown showed cap-vs-EW is a regime bet (cap's full-
  period edge is 2023-24 mega-cap concentration; EW won 6/10 years incl. 2022).
  A 50/50 blend hedges that bet rather than making it.
  Book: on the mcap-covered subset, weight = 0.5 x equal-weight + 0.5 x cap
  weight (each leg normalised over the same names before summing).
  Confirmatory: ``mix_screen20`` vs ``mix_broad``, same bar as v1/v2:
    (a) full-period net Sharpe(mix_screen20) > net Sharpe(mix_broad);
    (b) >= 10 of 19 window wins on net total return.
  Context (no bar): mix_screen10/30 curve points; yearly mix-vs-EW-vs-cap
  behaviour (does the blend damp both 2022-style and 2023/24-style regret);
  cap_broad and mix_broad tracking stats vs SPY.

PARENT-VETO (v4) PRE-REGISTERED 2026-07-14 (before the first v4 run; no retuning
after). This is the FOURTH pass over the same 110 months — hold any pass more
loosely than v1-v3.
  Motivation: the composite is an average, so one terrible parent can hide
  behind mid-pack strength elsewhere; the walk-forward OOS study found quality
  and short the only parents with positive OOS IC, giving vetoes on those two
  independent prior support. Also the practical goal: a smaller book than
  screen25 without winner-picking (which the tilt tests proved harmful).
  Veto rule (fixed): start from the ratified base book (mix weighting on the
  mcap-covered subset, minus the bottom 25% by composite = mix_screen25).
  For each veto parent, compute the per-date percentile rank of the parent
  score across ALL scored names that date (pandas rank(pct=True), average tie
  method); a survivor is vetoed if its rank is <= veto_pct for ANY veto parent
  (union). NaN parent values never veto; a parent with fewer than 10 distinct
  values that date is skipped (degenerate: e.g. short before 2018, revisions
  before 2019).
  Primary variant: ``vqs20`` — veto parents {quality, short}, veto_pct 0.20.
  Reference:       ``mix_screen25`` (identical construction, no veto).
  Pass bar (ALL three required):
    (a) full-period net Sharpe(vqs20) > net Sharpe(mix_screen25);
    (b) vqs20 net total return beats mix_screen25 in >= 10 of the 19 windows;
    (c) full-period net Sharpe(vqs20) >= the 90th percentile of 200
        random-characteristic null books (each draw: one fixed uniform u per
        ticker, numpy default_rng seed 20260714; each date drop the same
        NUMBER of names from mix_screen25 as vqs20 dropped that date, lowest
        u first, mix-weighted). The null preserves book size and month-to-
        month persistence but carries no information — it is what "fewer
        names by luck" looks like.
  A-priori expectations: the vetoed-names complement book (``vqs20_cut``)
  underperforms mix_screen25; the all-parent veto concentrates harder but is
  diluted by the OOS-dead parents.
  Exploratory (no bar, mapping only): vqs10; all-parent veto at 10/20;
  single-parent vetoes at 20 (which parent's bottom quintile actually hurts);
  the vqs20_cut mechanism check. No production veto will be chosen from the
  argmax of these cells without a fresh pre-registration.

NEW-CANDIDATE VETOES (v5) PRE-REGISTERED 2026-07-14 (before the first v5 run;
no retuning after). FIFTH pass over the same 110 months; 5 primary cells, so
at the null ~0.5 false passes are expected at the 10% level — a single
marginal pass is weak evidence, coherence across cells is the real read.
  Motivation: literature-supported "bad company" flags NOT derived from the
  existing eight parents, computable from data already in the DB
  (docs/strategy_current_and_roadmap.md, roadmap item 1).
  Base book: vpos6_20 (the ratified working spec). Each candidate enters as
  ONE additional veto column at the same 20% percentile rule (union with the
  base's six parent vetoes; NaN never vetoes; <10 distinct values that date
  = candidate skipped that date).
  Candidate definitions (fixed; all oriented as GOODNESS scores so the
  bottom 20% = the flagged names):
    net_issuance: implied shares = net_income / eps_diluted (|eps| >= 0.01,
      shares in (1e6, 1e11)) from the latest annual report available at d
      (fiscal_date + 90d lag) and the prior annual report (fiscal gap 300 to
      430 days, also available at d); score = -(sh_latest/sh_prior - 1).
    asset_growth: same two reports, total_assets > 0 both years;
      score = -(ta_latest/ta_prior - 1).
    accruals (Sloan, cash-flow form): latest report only, total_assets > 0;
      score = -(net_income - operating_cash_flow) / total_assets.
    idio_vol: trailing 252 trading days of daily returns ending AT the
      formation date (inclusive — same convention as composite scoring),
      per-name regression on SPY (pairwise-complete, min 126 obs),
      score = -annualised residual std.
    max5: mean of the 5 highest daily returns in the trailing 21 trading
      days ending at d (min 15 obs); score = -max5 (lottery flag).
  Primary cells: vp6_<cand>20 for each of the five candidates.
  Pass bar per candidate (ALL three required, judged independently):
    (a) full-period net Sharpe(vp6_<cand>20) > net Sharpe(vpos6_20);
    (b) >= 10 of 19 window wins vs vpos6_20 on net total return;
    (c) full-period net Sharpe >= the 90th percentile of 200
        random-characteristic null draws sized to that book
        (ref vpos6_20; same null construction as v4; numpy default_rng
        seeds 20260715 + cell index 0..4).
  Exploratory (no bar): solo_<cand>20 (each candidate as the ONLY veto on
  plain mix_screen25 — standalone strength diagnostic); vp6_allnew20 (base
  + all five). No post-hoc candidate combination will be adopted from this
  battery without a fresh pre-registration.
  A-priori expectation: headroom is small — the base book is already twice-
  screened (~267 names), so most candidates should overlap the existing
  vetoes and fail (b) or (c); net_issuance and idio_vol are the strongest
  priors.
"""
