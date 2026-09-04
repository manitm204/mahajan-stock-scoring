# Local-LLM Filing Pilot — Design & Pre-Registration

**Status: RUN & CLOSED 2026-07-15 — VERDICT: FAIL (all three checks).**
239/239 filings scored by gpt-oss-120b, zero parse failures; exclusions
2.3% / 3.5% (inside the 10% guard). The worst-decile tier OUTPERFORMED the
rest of the book in BOTH cohorts (2024: +10.7% vs +7.4%; 2025: +15.3% vs
+1.7%); pooled worst-dummy t = **+1.28** (wrong sign); actual drag −0.084 vs
null q90 +0.074 (beats only 8% of random tiers). Per the pre-registered
decision rule: the local/free-LLM avoid-list idea is SHELVED; the overlay
concept remains testable only forward with the production LLM. Consistent
with the v5 finding that penalizing scary-looking large caps is harmful —
the LLM's "worst accounting/balance-sheet narrative" names carried a risk
premium instead of dying. Artifacts: output/llm_pilot/{results.csv,
verdict.csv,pilot.png}. (Drafted 2026-07-15.)

**Amendment 2026-07-15, before any filing was scored:** model switched from
local Llama 3.1 8B to **Llama 3.3 70B (`llama-3.3-70b-versatile`) via the Groq
free API tier** — same documented December 2023 training cutoff, far stronger
model, and the free-tier rate limits (1,000 req/day, 12k tokens/min) run the
full two-cohort study in ~5 h vs 20–35 h locally. Both cohorts retained (the
1-cohort cut considered for the local route is unnecessary). Filings are
public SEC documents, so nothing confidential leaves the machine. The
cutoff-verification probe was run BEFORE this amendment was finalized and
**PASSED**: the model answered the pre-cutoff control (SVB collapse, March
2023) correctly and claimed ignorance on all five post-cutoff events,
including one from January 2024, weeks after the cutoff
(`output/llm_pilot/cutoff_probe.json`). Everything else below — cohorts,
tier definition, bar, seeds, decision rule — is unchanged.

**Amendment 2 (2026-07-15, before any retained scoring):** Groq's free tier
turned out to carry an undocumented ~100k-token/day cap → ~16 filings/day →
~2 weeks. Model/host switched to **gpt-oss-120b via the Cerebras free API**
(the only Dec-2023-or-earlier-cutoff model still served there was none;
gpt-oss-120b has a **documented June 2024 cutoff**, which sits exactly at
cohort A's formation date). Because the slack is zero, a STRICTER boundary
probe was run and **PASSED** (`output/llm_pilot/cutoff_probe_gptoss.json`):
the model knows May 2024 (Red Lobster bankruptcy) and early-June 2024 (S&P
level) but claims ignorance of July 19 2024 (CrowdStrike outage — week 3 of
cohort A's forward window), Aug 5 2024 (yen unwind), Nov 2024 (election),
Sep 2024 FOMC, and Jan 2025 (DeepSeek/Nvidia). MD&A cap trimmed 20k → 8k
chars, sized to Cerebras' observed limits (5 req/min, 150 req/hour, 1M
tokens/day) with ~14% retry margin; the 150/hour cap sets the ~1.7h runtime. The 21 filings scored by Llama 3.3
70B on Groq were archived (not mixed in); ALL filings are scored uniformly
by gpt-oss-120b. Cohorts, tier definition, bar, seeds, and the decision rule
remain unchanged. Residual risk, stated: cohort A's protection now rests on
the empirical probe at a zero-slack boundary; cohort B (formation 2025-06)
retains a full year of slack.

## Question

Can a free, local LLM reading point-in-time SEC filings identify names inside
the vpos6_20 book that subsequently underperform? If yes, an LLM avoid-list is
worth running as a live forward ledger. If no, we conclude only that *this
model* can't do it (a weak local model failing does not kill the overlay
concept — the production paid-LLM overlay is a separate, stronger instrument).

## The look-ahead problem and the protection

A 2026-trained LLM already knows what happened to these companies. Any test
where the model's training cutoff overlaps the forward-return window is
worthless — the model can "detect" a blowup by remembering it.

**Protection: the model's training cutoff must predate the portfolio formation
date.** The model may know a company's pre-cutoff reputation (a human analyst
does too); it must not know anything from the forward window.

- **Model: Llama 3.1 8B Instruct, Q4_K_M quantization** (via ollama /
  llama.cpp). Meta documents the training cutoff as **December 2023**.
  Backup if it can't follow the JSON schema: Qwen2.5-7B-Instruct (cutoff also
  2023).
- **Cutoff verification step (before any filing is scored):** probe the model
  with 5 post-cutoff event questions (2024 election result, 2024–2025 market
  events, specific 2024 earnings outcomes). Log the answers. If it answers any
  correctly, the cutoff claim is wrong — stop and pick another model.

## Cohorts

Two cohorts, both formed after the model cutoff:

| Cohort | Formation (book as of) | Filings used | Forward window |
|---|---|---|---|
| A | 2024-06-30 rebal date | latest 10-K MD&A filed ≤ formation | 2024-07 → 2025-06 |
| B | 2025-06-30 rebal date | latest 10-K MD&A filed ≤ formation | 2025-07 → 2026-06 |

- Book = the vpos6_20 members (~267 names) at that rebal date, from the same
  PIT panel as the loser-screen study (survivorship-free; delisted names keep
  their filings on EDGAR and their prices in the deep-history matrix).
- Filing = most recent 10-K with an extractable Item 7 MD&A filed on or before
  the formation date; fallback most recent 10-Q Item 2. Names with neither are
  excluded and counted.
- The two cohorts overlap heavily in membership (~200 common names with
  different filings and different forward windows). They are NOT independent
  samples; pooled statistics are reported but the both-cohorts-directional
  check below is the guard.

## Inputs and prompts

- Production prompts **verbatim**: `FILING_SYSTEM` + `FILING_USER_TEMPLATE`
  from `analysis/prompts.py`, production MD&A extraction
  (`analysis/data_access.extract_mdna_section`) and edge-preserving truncation.
- One deviation from production, declared: MD&A truncated to **20,000 chars**
  (vs production 45,000) for CPU prompt-processing budget. Same limit for
  every name — no cherry-picking.
- The model sees ONLY: ticker, sector, form type, filing date, MD&A text.
  No prices, no factor scores, no dates beyond the filing date.

## Tier construction (fixed before running)

Per name: `llm_score = mean(earnings_quality_score, balance_sheet_score,
100 − accounting_risk)`. Names where the model returns unparseable JSON after
one retry are excluded and counted (if exclusions exceed 10% of a cohort, the
model failed the pilot on execution grounds — report and stop).

**Worst tier = bottom 10% of the cohort by `llm_score`** (~27 names per
cohort). No confidence weighting, no red-flag counting, no alternative cuts —
one pre-registered definition.

## Pre-registered bar

Outcome per name: 12-month forward total return (monthly compounding from the
deep-history price matrix; delisted names carry their return to the last print,
then flat). All three checks must pass:

1. **Direction, both cohorts:** worst-tier equal-weight forward return is
   below the rest-of-book equal-weight return in cohort A AND in cohort B.
2. **Pooled significance with quant control:** pooled across cohorts,
   regression `fwd_return ~ composite_percentile + worst_tier_dummy`
   (composite percentile as of formation), dummy coefficient negative with
   **t ≤ −2.0**. This is the guard against the LLM merely re-discovering the
   composite.
3. **Size-matched null:** 200 random draws (seed **20260716**) of the same
   tier size per cohort; the actual worst-tier drag (rest-minus-tier spread,
   pooled) must exceed the **90th percentile** of null drags.

Decision rule, declared now:
- **All 3 pass** → start the live forward avoid-list ledger AND budget time
  for a stronger open model / fuller analyzer stack.
- **Only check 1 passes** → forward ledger only, no further backtest passes.
- **Check 1 fails** → shelve the local-LLM idea; the concept remains testable
  only forward with the production LLM.

No second cut of the tier definition, no third cohort, no model swap after
seeing results (a JSON-compliance swap per above must happen before any
returns are looked at).

## A-priori expectation (honesty clause)

Expected outcome: **fail or directional-only.** An 8B quantized model is far
below the production analyzer's quality, MD&A prose in large caps is heavily
lawyered, and the loser-screen program showed large-cap exclusion signals are
mostly exhausted (v5: five candidates, zero passes). The pilot is worth ~2
days of machine time because a pass would unlock a genuinely orthogonal,
free, PIT-safe signal source.

## Data & compute plan

1. **EDGAR backfill** (needed — warehouse holds 2026 filings only): windowed
   variant of `data/sec_data.py` pulling each cohort name's latest 10-K (and
   10-Q fallback) as of the formation dates. ≈ 500–700 filings, well inside
   EDGAR's 10 req/s limit; append-only into the existing `sec_filings` table.
   Verify: MD&A extractable for ≥ 90% of each cohort's book.
2. **Model runtime** (no GPU, ~7 GB free RAM): 8B Q4 ≈ 5 GB resident;
   ~5k-token prompts ≈ 2–4 min per filing on 14 CPU cores → **≈ 20–35 h for
   ~540 filings**. Runner caches one JSON per (ticker, filing, model) so it is
   resume-safe and runs as overnight batches.
3. **Analysis script**: tiers, three checks, null draws, verdict CSV + chart
   under `output/llm_pilot/`.

## Phase 2 — production-overlay replication (pre-registered 2026-07-16,
## before the first overlay call)

The filing-analyzer pilot FAILED its avoid-list bar; exploration found an
inverted (contrarian) accounting-risk pattern (t=+2.00, one nominal hit in
~25 cuts — candidate only). Phase 2 tests what production ACTUALLY runs: the
categorical research overlay.

Design: production OVERLAY_SYSTEM + OVERLAY_USER_TEMPLATE + validate_overlay
verbatim; same gpt-oss-120b / Cerebras (probe already passed). Quant block
rebuilt PIT from the clean window caches (composite percentile, 8 parent
scores, 70/40 driver bars, PIT price returns; valuation ratios omitted —
not PIT-available). filing_analysis = the phase-1 JSONs; earnings/risk/
insider = null (production-tolerated; declared). All 239 covered names,
both cohorts, LONG signal direction.

Pre-registered read:
  * Primary: {REVIEW, AVOID_RED_FLAG} underperform {PASS, WATCHLIST} on
    cohort-demeaned 12M forward return (production intent), reported
    two-sided.
  * Secondary: {WEAKENS, CONTRADICTS} vs {CONFIRMS} on quant_signal_review.
  * Guard: any comparison side with <15 pooled names is reported
    descriptively, NO verdict (the model sets its own label distribution).
  * No tier re-cuts after seeing results; category definitions are the
    production enums.

**Phase 2 RESULT (2026-07-16, run CLOSED):** 236/239 scored (3 lost to the
daily token quota, alphabetical tail, declared exclusions). Labels:
WATCHLIST 178, PASS 56, AVOID_RED_FLAG 2, REVIEW 1; CONFIRMS 41, MIXED 147,
WEAKENS 49, CONTRADICTS 0. **Primary comparison tripped the <15 guard as
the model almost never flags (3 names) — descriptive only** (the 2 AVOID
names returned −20.5pp demeaned, the 1 REVIEW +30pp; anecdotes).
**Secondary: WEAKENS beat CONFIRMS by +3.5pp (t=+0.53) — wrong direction,
insignificant.** Unregistered but striking: PASS (n=56) −4.7pp demeaned,
Sharpe 0.14 vs WATCHLIST (n=178) +1.5pp, Sharpe 0.65 — the names the model
liked best did worst, the same contrarian lean as phase 1. Conclusion: with
filing-only evidence, the production overlay's judgments show NO edge in
the intended direction on this book; the alarm tiers (REVIEW/AVOID) are
essentially never exercised by MD&A prose alone, so any production AVOID
value must come from the other three analyzers (earnings calls, risk-factor
diffs, insider) — untested here. Artifacts: output/llm_pilot/
{overlay_results.csv,overlay_charts.png}.

## Known limitations (stated up front)

- Company-reputation prior: the model knows these firms as of Dec 2023.
  Acceptable (analyst-equivalent); outcome knowledge is what's excluded.
- Cohort A filings filed in late 2023 may literally be in the training set.
  Harmless for this test: the binding constraint is cutoff < formation, and
  the forward window starts 6+ months after cutoff.
- Single-analyzer pilot ≠ production overlay (no transcripts, no risk-factor
  diff, no insider context, no quant block). A fail here does not indict the
  production overlay.
- Two overlapping cohorts = limited power. That is what "pilot" means.
