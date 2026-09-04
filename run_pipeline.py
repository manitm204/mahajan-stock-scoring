"""Mahajan Hedge Fund - end-to-end orchestrator.

Runs Layers 1-3 in order, stopping the chain on the first failure. The
sequence:

    1. Layer 1 (data)        - python run_data.py
    2. Layer 2 (scoring)     - python run_scoring.py
    3. Layer 3 (analysis)    - python run_analysis.py --full-run --top-n 10
                                  --quant-weight 0.75 --qual-weight 0.25

Portfolio construction / execution layers were retired in the 2026-08-08
frontend rework (model book is now computed read-only by the dashboard);
launch the dashboard afterwards with ``python run_dashboard.py``.

Each step is a subprocess so argparse state stays clean and a non-zero
exit code in any layer halts the chain with a clear "step N failed"
message. Use ``--skip-*`` flags to bypass layers (useful when the data
warehouse is already fresh and you just want to re-score, etc.).

Examples::

    # Full pipeline with the defaults you asked for
    python run_pipeline.py

    # Already-fresh data; re-run from scoring onward
    python run_pipeline.py --skip-data

    # Pump the budget and the LLM top-N
    python run_pipeline.py --top-n 15 --budget 2.0
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

# Resolve relative to this file so the pipeline can run from any cwd.
ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mahajan Hedge Fund - run Layers 1-3 end-to-end",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Per-layer skip flags ------------------------------------------------
    p.add_argument("--skip-data",      action="store_true",
                   help="Skip Layer 1 (assume cache/mahajan.db is fresh)")
    p.add_argument("--skip-scoring",   action="store_true",
                   help="Skip Layer 2 (composite_scores already current)")
    p.add_argument("--skip-analysis",  action="store_true",
                   help="Skip Layer 3 (no Claude review this run)")

    # Layer 3 knobs (the top-N LLM coverage cap) --------------------------
    p.add_argument("--top-n", type=int, default=10,
                   help="Layer 3 LLM coverage: top-N long + top-N short")
    p.add_argument("--budget", type=float, default=2.50,
                   help="Layer 3 hard $ budget for Claude calls")
    p.add_argument("--no-cache", action="store_true",
                   help="Layer 3: bypass the analysis cache")

    # Convenience -------------------------------------------------------
    p.add_argument("--dry-run", action="store_true",
                   help="Print every command without running it")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------
def _banner(step: int, title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n  STEP {step} - {title}\n{bar}", flush=True)


def _run(step: int, title: str, argv: Sequence[str], *, dry_run: bool) -> None:
    _banner(step, title)
    cmd = [sys.executable, *argv]
    pretty = " ".join(shlex.quote(c) for c in cmd)
    print(f"$ {pretty}", flush=True)
    if dry_run:
        return
    started = time.time()
    proc = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - started
    if proc.returncode != 0:
        print(
            f"\n!! STEP {step} ({title}) failed with exit code "
            f"{proc.returncode} after {elapsed:.1f}s -- halting pipeline.",
            flush=True,
        )
        sys.exit(proc.returncode)
    print(f"  step done in {elapsed:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    started = time.time()
    print("Mahajan Hedge Fund - pipeline starting")
    print(f"  LLM coverage   : top {args.top_n} long + top {args.top_n} short")
    print(f"  budget cap     : ${args.budget:.2f}")
    print(f"  dry-run        : {args.dry_run}")

    step = 0

    # --- Layer 1 -------------------------------------------------------
    if not args.skip_data:
        step += 1
        _run(step, "Layer 1 - data ingestion",
             ["run_data.py"], dry_run=args.dry_run)

    # --- Layer 2 -------------------------------------------------------
    if not args.skip_scoring:
        step += 1
        _run(step, "Layer 2 - composite scoring",
             ["run_scoring.py"], dry_run=args.dry_run)

    # --- Layer 3 -------------------------------------------------------
    if not args.skip_analysis:
        step += 1
        argv = [
            "run_analysis.py",
            "--full-run",
            "--top-n", str(args.top_n),
            "--budget", str(args.budget),
        ]
        if args.no_cache:
            argv.append("--no-cache")
        _run(step, "Layer 3 - qualitative review + persist research_overlays",
             argv, dry_run=args.dry_run)

    elapsed = time.time() - started
    print("\n" + "=" * 78)
    print(f"  PIPELINE COMPLETE in {elapsed:.1f}s")
    print("=" * 78)
    print()
    print("Next step:")
    print()
    print("  # launch the read-only dashboard:")
    print("  python run_dashboard.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
