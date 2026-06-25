#!/usr/bin/env python3
"""Summarize per-stage RAG cost from the cost_logger JSONL logs.

Reads data/cost_logs/cost_*.jsonl (written by peatlearn/rag/cost_logger.py) and
prints the average per-query cost broken out by stage (embed / rerank / generate
/ verify), with each stage's share of the total. This is the measurement the
LLM council asked for: is retrieval actually a meaningful fraction of cost?

Usage:
    python scripts/eval/cost_report.py                 # all logs
    python scripts/eval/cost_report.py --date 20260620 # one day
    python scripts/eval/cost_report.py --last 100      # last N queries
"""

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "data" / "cost_logs"


def load_records(date: str | None) -> list[dict]:
    pattern = f"cost_{date}.jsonl" if date else "cost_*.jsonl"
    records: list[dict] = []
    for path in sorted(glob.glob(str(LOG_DIR / pattern))):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-stage RAG cost report")
    ap.add_argument("--date", help="YYYYMMDD — only this day's log")
    ap.add_argument("--last", type=int, help="Only the last N queries")
    args = ap.parse_args()

    records = load_records(args.date)
    if args.last:
        records = records[-args.last:]

    if not records:
        print(f"\n  No cost records found in {LOG_DIR}\n"
              f"  Run some queries through the RAG first (eval harness or live traffic).\n")
        return 1

    n = len(records)
    stage_cost: dict[str, float] = defaultdict(float)
    stage_calls: dict[str, int] = defaultdict(int)
    stage_seen: dict[str, int] = defaultdict(int)  # queries where stage fired
    stage_model: dict[str, str] = {}
    total_cost = 0.0
    total_elapsed = 0.0

    for r in records:
        total_cost += r.get("total_usd", 0.0)
        total_elapsed += r.get("elapsed_s", 0.0)
        for name, s in r.get("stages", {}).items():
            stage_cost[name] += s.get("cost_usd", 0.0)
            stage_calls[name] += s.get("calls", 0)
            stage_seen[name] += 1
            if s.get("model"):
                stage_model[name] = s["model"]

    avg_total = total_cost / n
    order = ["embed", "rerank", "generate", "verify"]
    stages = order + [s for s in stage_cost if s not in order]

    print("\n" + "=" * 68)
    print("            RAG PER-STAGE COST REPORT")
    print("=" * 68)
    print(f"\n  Queries analyzed : {n}")
    print(f"  Avg cost/query   : ${avg_total:.5f}")
    print(f"  Avg latency      : {total_elapsed / n:.2f}s")
    print(f"  Projected /1k    : ${avg_total * 1000:.2f}")
    print(f"\n  {'stage':<10} {'avg $/query':>12} {'share':>8} {'model':<22} {'fired':>7}")
    print("  " + "-" * 64)
    retrieval_share = 0.0
    for name in stages:
        if name not in stage_cost:
            continue
        avg = stage_cost[name] / n
        share = (stage_cost[name] / total_cost * 100) if total_cost else 0.0
        if name in ("embed", "rerank"):
            retrieval_share += share
        print(f"  {name:<10} {avg:>12.6f} {share:>7.1f}% "
              f"{stage_model.get(name, '?'):<22} {stage_seen[name]:>5}/{n}")
    print("  " + "-" * 64)
    print(f"  {'TOTAL':<10} {avg_total:>12.6f} {'100.0%':>8}")

    gen_verify_share = sum(
        stage_cost.get(s, 0.0) for s in ("generate", "verify")
    ) / total_cost * 100 if total_cost else 0.0

    print(f"\n  Retrieval (embed+rerank) share : {retrieval_share:.1f}%")
    print(f"  Generation+verify share        : {gen_verify_share:.1f}%")
    print("\n  Council gate: a router optimizes ONLY the retrieval share above.")
    if retrieval_share < 25:
        print(f"  -> Retrieval is {retrieval_share:.0f}% of cost. The router's best-case")
        print( "     saving is small; the cost case for it is weak. (As predicted.)")
    else:
        print(f"  -> Retrieval is {retrieval_share:.0f}% of cost — larger than expected;")
        print( "     a router experiment may be worth prototyping against the harness.")
    print("\n" + "=" * 68 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
