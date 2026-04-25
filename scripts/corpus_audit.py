#!/usr/bin/env python3
"""
Corpus Audit — Analyze all 552 processed files for quality issues.

Reports:
  1. Parser coverage: how many RAY PEAT chunks have vs lack CONTEXT
  2. Duplicates: exact-match and near-duplicate detection
  3. Garbage: chunks under 25 tokens (filler like "Yeah.", "Mm-hmm.")
  4. Content type breakdown: transcript vs article vs newsletter vs email
  5. Token distribution: histogram of chunk sizes

Compares the buggy parser (split by \\n\\n) vs the smart parser (marker scanning)
to quantify exactly how many contexts are recovered.

Usage:
    python scripts/corpus_audit.py
    python scripts/corpus_audit.py --metadata data/embeddings/vectors/metadata_20250728_221826.json
"""

import json
import re
import sys
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "ai_cleaned"
ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Parsers ─────────────────────────────────────────────────────────────────

def parse_chunks_smart(content: str) -> list[dict]:
    """Parse RAY PEAT / CONTEXT pairs using marker scanning (NOT split by \\n\\n).

    This is the FIXED parser that correctly pairs markers regardless of
    whitespace between them — the root cause of 15,093 lost contexts.
    """
    chunks = []

    # Find all marker positions in order
    markers = []
    for m in re.finditer(r'\*\*RAY PEAT:\*\*', content):
        markers.append(('RP', m.start(), m.end()))
    for m in re.finditer(r'\*\*CONTEXT:\*\*', content):
        markers.append(('CTX', m.start(), m.end()))

    # Sort by position in file
    markers.sort(key=lambda x: x[1])

    # Walk markers and pair them
    i = 0
    while i < len(markers):
        tag, start, end = markers[i]

        if tag == 'RP':
            # Extract RP text: from end of marker to next marker (or EOF)
            next_pos = markers[i + 1][1] if i + 1 < len(markers) else len(content)
            rp_text = content[end:next_pos].strip()

            # Look for adjacent CONTEXT (before or after)
            ctx_text = ""

            # Check if NEXT marker is CONTEXT
            if i + 1 < len(markers) and markers[i + 1][0] == 'CTX':
                ctx_end = markers[i + 1][2]
                ctx_next = markers[i + 2][1] if i + 2 < len(markers) else len(content)
                ctx_text = content[ctx_end:ctx_next].strip()

            # Check if PREVIOUS marker was CONTEXT (context-first pattern)
            elif i > 0 and markers[i - 1][0] == 'CTX':
                ctx_end = markers[i - 1][2]
                ctx_text = content[ctx_end:start].strip()

            tokens = len(rp_text.split())
            chunks.append({
                "ray_peat_response": rp_text,
                "context": ctx_text,
                "tokens": tokens,
                "has_context": bool(ctx_text),
            })

        i += 1

    return chunks


def parse_chunks_buggy(content: str) -> list[dict]:
    """Original buggy parser (split by \\n\\n) — for comparison."""
    chunks = []
    sections = content.split('\n\n')

    for section in sections:
        section = section.strip()
        if not section:
            continue

        ray_peat_match = re.search(
            r'\*\*RAY PEAT:\*\*\s*(.*?)(?=\*\*CONTEXT:\*\*|$)', section, re.DOTALL
        )
        context_match = re.search(
            r'\*\*CONTEXT:\*\*\s*(.*?)(?=\*\*RAY PEAT:\*\*|$)', section, re.DOTALL
        )

        if ray_peat_match:
            rp_text = ray_peat_match.group(1).strip()
            ctx_text = context_match.group(1).strip() if context_match else ""
            tokens = len(rp_text.split())
            chunks.append({
                "ray_peat_response": rp_text,
                "context": ctx_text,
                "tokens": tokens,
                "has_context": bool(ctx_text),
            })

    return chunks


# ── Helpers ──────────────────────────────────────────────────────────────────

def classify_source(file_path: Path) -> str:
    """Classify a processed file by its source content type."""
    parts = file_path.relative_to(PROCESSED_DIR).parts

    if "01_Audio_Transcripts" in parts:
        return "transcript"
    elif "02_Publications" in parts:
        return "article"
    elif "03_Chronological_Content" in parts:
        return "newsletter"
    elif "04_Health_Topics" in parts:
        return "article"
    elif "05_Academic_Documents" in parts:
        return "academic"
    elif "06_Email_Communications" in parts:
        return "email"
    elif "07_Special_Collections" in parts:
        return "special"
    elif "08_Newslatters" in parts:
        return "newsletter"
    elif "09_Miscellaneous" in parts:
        return "misc"
    else:
        return "unknown"


def fingerprint(text: str) -> str:
    """Normalize text for dedup: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def bar_chart(count: int, total: int, width: int = 30) -> str:
    filled = int(round(count / total * width)) if total else 0
    return "#" * filled


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Corpus quality audit")
    parser.add_argument("--metadata", default=None,
                        help="Optional: metadata JSON from old embedding run for comparison")
    args = parser.parse_args()

    processed_files = sorted(PROCESSED_DIR.rglob("*_processed.txt"))
    print(f"Found {len(processed_files)} processed files\n")

    # Accumulators
    all_chunks_smart = []
    all_chunks_buggy = []
    type_counts = Counter()
    type_chunks = defaultdict(int)
    type_context_hits = defaultdict(int)
    file_stats = []

    for fp in processed_files:
        content = fp.read_text(encoding="utf-8", errors="replace")
        source_type = classify_source(fp)
        type_counts[source_type] += 1

        smart = parse_chunks_smart(content)
        buggy = parse_chunks_buggy(content)

        for c in smart:
            c["source_file"] = str(fp.relative_to(PROCESSED_DIR))
            c["source_type"] = source_type

        all_chunks_smart.extend(smart)
        all_chunks_buggy.extend(buggy)

        type_chunks[source_type] += len(smart)
        type_context_hits[source_type] += sum(1 for c in smart if c["has_context"])

        file_stats.append({
            "file": str(fp.relative_to(PROCESSED_DIR)),
            "type": source_type,
            "chunks_smart": len(smart),
            "chunks_buggy": len(buggy),
            "ctx_smart": sum(1 for c in smart if c["has_context"]),
            "ctx_buggy": sum(1 for c in buggy if c["has_context"]),
        })

    # ── Report ───────────────────────────────────────────────────────────────
    W = 70
    print("=" * W)
    print("  CORPUS QUALITY AUDIT REPORT")
    print("=" * W)

    # 1. File type breakdown
    total_chunks = len(all_chunks_smart)
    total_ctx = sum(1 for c in all_chunks_smart if c["has_context"])

    print(f"\n  ## Content Type Breakdown\n")
    print(f"  {'Type':<15} {'Files':>6} {'Chunks':>8} {'w/ Context':>12} {'Coverage':>10}")
    print(f"  {'-'*15} {'-'*6} {'-'*8} {'-'*12} {'-'*10}")
    for t in sorted(type_counts.keys(), key=lambda x: -type_chunks[x]):
        total = type_chunks[t]
        with_ctx = type_context_hits[t]
        pct = (with_ctx / total * 100) if total else 0
        print(f"  {t:<15} {type_counts[t]:>6} {total:>8} {with_ctx:>12} {pct:>9.1f}%")

    print(f"\n  {'TOTAL':<15} {len(processed_files):>6} {total_chunks:>8} {total_ctx:>12} {total_ctx/total_chunks*100:>9.1f}%")

    # 2. Parser comparison
    buggy_ctx = sum(1 for c in all_chunks_buggy if c["has_context"])
    buggy_total = len(all_chunks_buggy)
    print(f"\n  ## Parser Comparison (buggy vs smart)\n")
    print(f"  Buggy (split \\n\\n):  {buggy_total:>7,} chunks, {buggy_ctx:>6,} with context ({buggy_ctx/max(buggy_total,1)*100:.1f}%)")
    print(f"  Smart (markers):     {total_chunks:>7,} chunks, {total_ctx:>6,} with context ({total_ctx/total_chunks*100:.1f}%)")
    print(f"  Contexts recovered:  +{total_ctx - buggy_ctx:,}")
    print(f"  Chunks difference:   {total_chunks - buggy_total:+,}")

    # 3. Duplicate detection
    print(f"\n  ## Duplicate Detection\n")
    seen = defaultdict(list)
    for i, c in enumerate(all_chunks_smart):
        fp = fingerprint(c["ray_peat_response"])
        if len(fp) > 20:  # skip tiny fragments
            seen[fp].append(i)

    dup_groups = {k: v for k, v in seen.items() if len(v) > 1}
    dup_count = sum(len(v) - 1 for v in dup_groups.values())
    print(f"  Unique text fingerprints: {len(seen):,}")
    print(f"  Duplicate groups:         {len(dup_groups):,}")
    print(f"  Total duplicates:         {dup_count:,} ({dup_count/total_chunks*100:.1f}% of corpus)")

    # Worst offenders
    worst = sorted(dup_groups.items(), key=lambda x: len(x[1]), reverse=True)[:5]
    if worst:
        print(f"\n  Top 5 duplicate groups:")
        for fp_text, indices in worst:
            preview = fp_text[:80].encode("ascii", "replace").decode()
            files = set()
            for idx in indices[:5]:
                sf = all_chunks_smart[idx]["source_file"]
                files.add(sf.encode("ascii", "replace").decode())
            print(f"    {len(indices):>4}x  \"{preview}...\"")
            print(f"          from: {', '.join(sorted(files))}")

    # 4. Garbage detection
    print(f"\n  ## Garbage / Filler Chunks\n")
    garbage = [c for c in all_chunks_smart if c["tokens"] < 25]
    filler = [c for c in all_chunks_smart if c["tokens"] < 10]
    print(f"  Under 25 tokens: {len(garbage):,} ({len(garbage)/total_chunks*100:.1f}%)")
    print(f"  Under 10 tokens: {len(filler):,} ({len(filler)/total_chunks*100:.1f}%)")

    # Breakdown by type
    print(f"\n  Garbage by content type:")
    for t in sorted(type_counts.keys()):
        type_garbage = sum(1 for c in all_chunks_smart if c["source_type"] == t and c["tokens"] < 25)
        type_total = type_chunks[t]
        if type_total > 0:
            print(f"    {t:<15} {type_garbage:>5} / {type_total:>6} ({type_garbage/type_total*100:.1f}%)")

    if filler:
        print(f"\n  Sample filler (under 10 tokens):")
        for c in filler[:10]:
            print(f"    [{c['tokens']:>2}t] \"{c['ray_peat_response'][:60]}\"")

    # 5. Token distribution
    print(f"\n  ## Token Distribution\n")
    tokens_list = [c["tokens"] for c in all_chunks_smart]
    buckets = [
        ("1-10", 1, 10),
        ("11-25", 11, 25),
        ("26-50", 26, 50),
        ("51-100", 51, 100),
        ("101-200", 101, 200),
        ("201-500", 201, 500),
        ("501+", 501, 999999),
    ]
    for label, lo, hi in buckets:
        count = sum(1 for t in tokens_list if lo <= t <= hi)
        b = bar_chart(count, total_chunks, 30)
        print(f"  {label:>10}: {count:>6} ({count/total_chunks*100:>5.1f}%) {b}")

    avg_tokens = sum(tokens_list) / len(tokens_list) if tokens_list else 0
    sorted_tokens = sorted(tokens_list)
    median_tokens = sorted_tokens[len(sorted_tokens) // 2] if sorted_tokens else 0
    print(f"\n  Mean: {avg_tokens:.0f} tokens | Median: {median_tokens} tokens")
    print(f"  Min: {min(tokens_list)} | Max: {max(tokens_list)}")

    # 6. After-cleanup projection
    print(f"\n  ## Projected Corpus After Cleanup\n")
    clean_count = total_chunks - dup_count - len(garbage)
    print(f"  Current chunks:     {total_chunks:,}")
    print(f"  - Duplicates:       -{dup_count:,}")
    print(f"  - Garbage (<25t):   -{len(garbage):,}")
    print(f"  = Clean corpus:     ~{clean_count:,} chunks")
    print(f"  Context coverage:   {total_ctx/total_chunks*100:.1f}% (smart parser)")

    # 7. Optional: compare with old metadata
    if args.metadata:
        meta_path = Path(args.metadata)
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            meta_total = len(meta)
            meta_ctx = sum(1 for m in meta if m.get("context", "").strip())
            meta_empty = meta_total - meta_ctx
            print(f"\n  ## Comparison with Pinecone metadata ({meta_path.name})\n")
            print(f"  Pinecone vectors:  {meta_total:,}")
            print(f"  With context:      {meta_ctx:,} ({meta_ctx/meta_total*100:.1f}%)")
            print(f"  Empty context:     {meta_empty:,} ({meta_empty/meta_total*100:.1f}%)")
            print(f"  Smart parser fix:  {total_ctx:,} ({total_ctx/total_chunks*100:.1f}%)")
            print(f"  Improvement:       +{total_ctx - meta_ctx:,} contexts recovered")

    # Save report JSON
    report = {
        "timestamp": datetime.now().isoformat(),
        "files": len(processed_files),
        "total_chunks_smart": total_chunks,
        "total_chunks_buggy": buggy_total,
        "context_coverage_smart_pct": round(total_ctx / total_chunks * 100, 1),
        "context_coverage_buggy_pct": round(buggy_ctx / max(buggy_total, 1) * 100, 1),
        "contexts_recovered": total_ctx - buggy_ctx,
        "duplicates": dup_count,
        "duplicate_groups": len(dup_groups),
        "garbage_under_25t": len(garbage),
        "garbage_under_10t": len(filler),
        "clean_projected": clean_count,
        "type_breakdown": {
            t: {"files": type_counts[t], "chunks": type_chunks[t],
                "with_context": type_context_hits[t]}
            for t in type_counts
        },
        "token_stats": {
            "mean": round(avg_tokens, 1),
            "median": median_tokens,
            "min": min(tokens_list),
            "max": max(tokens_list),
        },
    }
    report_path = ARTIFACTS_DIR / "corpus_audit.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Report saved: {report_path.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * W)
    return report


if __name__ == "__main__":
    main()
