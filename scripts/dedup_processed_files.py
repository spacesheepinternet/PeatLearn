#!/usr/bin/env python3
"""
Deduplicate bloated processed files in data/processed/ai_cleaned/.

The AI cleaning step sometimes repeats the same RAY PEAT quote hundreds
of times with different CONTEXT labels.  This script keeps only the first
occurrence of each unique quote block.

Usage:
    python scripts/dedup_processed_files.py              # dry-run (default)
    python scripts/dedup_processed_files.py --apply       # overwrite files
    python scripts/dedup_processed_files.py --threshold 150  # custom bloat %
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

PROCESSED_DIR = Path("data/processed/ai_cleaned")
RAW_DIR = Path("data/raw/raw_data")


def parse_blocks(text: str) -> list[tuple[str, str]]:
    """Parse file into (ray_peat_text, context_text) pairs.

    Returns a list of (quote, context) tuples.  Anything before the first
    **RAY PEAT:** marker is returned as a single block with context=''.
    """
    # Split on the RAY PEAT marker, keeping it
    parts = re.split(r"(?=\*\*RAY PEAT:\*\*)", text)
    blocks: list[tuple[str, str]] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Try to separate quote from context
        m = re.match(
            r"\*\*RAY PEAT:\*\*\s*(.*?)(?:\n\s*\n\s*\*\*CONTEXT:\*\*\s*(.*?))?$",
            part,
            re.DOTALL,
        )
        if m:
            quote = m.group(1).strip()
            context = (m.group(2) or "").strip()
            blocks.append((quote, context))
        else:
            # Non-standard block — keep as-is
            blocks.append((part, ""))
    return blocks


def dedup_blocks(blocks: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Remove duplicate quote blocks, keeping first occurrence."""
    seen: set[str] = set()
    result: list[tuple[str, str]] = []
    for quote, context in blocks:
        if quote in seen:
            continue
        seen.add(quote)
        result.append((quote, context))
    return result


def reconstruct(blocks: list[tuple[str, str]]) -> str:
    """Rebuild file text from deduplicated blocks."""
    parts: list[str] = []
    for quote, context in blocks:
        if quote.startswith("**RAY PEAT:**") or not quote:
            # Already has marker (non-standard block)
            parts.append(quote)
        else:
            parts.append(f"**RAY PEAT:** {quote}")
        if context:
            parts.append(f"\n\n**CONTEXT:** {context}")
        parts.append("")  # blank line separator
    return "\n\n".join(parts).strip() + "\n"


def find_bloated_files(threshold_pct: int = 200) -> list[tuple[Path, Path, int]]:
    """Find processed files that are significantly larger than their raw source."""
    bloated: list[tuple[Path, Path, int]] = []
    for proc_path in PROCESSED_DIR.rglob("*_processed.txt"):
        rel = proc_path.relative_to(PROCESSED_DIR)
        # Reconstruct raw path: remove _processed suffix
        raw_name = str(rel).replace("_processed.txt", ".txt")
        raw_path = RAW_DIR / raw_name
        if not raw_path.exists():
            continue
        raw_size = raw_path.stat().st_size
        proc_size = proc_path.stat().st_size
        if raw_size == 0:
            continue
        ratio = proc_size * 100 // raw_size
        if ratio > threshold_pct:
            bloated.append((proc_path, raw_path, ratio))
    bloated.sort(key=lambda x: -x[2])
    return bloated


def main():
    parser = argparse.ArgumentParser(description="Deduplicate bloated processed files")
    parser.add_argument("--apply", action="store_true", help="Actually overwrite files (default: dry-run)")
    parser.add_argument("--threshold", type=int, default=200, help="Bloat threshold %% (default: 200)")
    args = parser.parse_args()

    bloated = find_bloated_files(args.threshold)
    if not bloated:
        print("No bloated files found.")
        return

    print(f"Found {len(bloated)} bloated files (>{args.threshold}% of raw size):\n")

    total_before = 0
    total_after = 0
    files_fixed = 0

    for proc_path, raw_path, ratio in bloated:
        text = proc_path.read_text(encoding="utf-8")
        blocks = parse_blocks(text)
        deduped = dedup_blocks(blocks)
        removed = len(blocks) - len(deduped)

        if removed == 0:
            print(f"  SKIP {ratio:4d}% | {proc_path.name} (no duplicate blocks found)")
            continue

        new_text = reconstruct(deduped)
        before_size = len(text.encode("utf-8"))
        after_size = len(new_text.encode("utf-8"))
        total_before += before_size
        total_after += after_size
        files_fixed += 1

        print(
            f"  {'FIX ' if args.apply else 'WOULD FIX'} {ratio:4d}% | "
            f"{proc_path.name} | "
            f"{len(blocks)} -> {len(deduped)} blocks ({removed} dupes removed) | "
            f"{before_size:,} -> {after_size:,} bytes"
        )

        if args.apply:
            proc_path.write_text(new_text, encoding="utf-8")

    print(f"\n{'Applied' if args.apply else 'Dry-run'}: {files_fixed} files, "
          f"{total_before:,} -> {total_after:,} bytes "
          f"({(total_before - total_after):,} bytes saved)")

    if not args.apply and files_fixed > 0:
        print("\nRun with --apply to overwrite files.")


if __name__ == "__main__":
    main()
