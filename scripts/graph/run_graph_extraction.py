"""
Batch knowledge graph extractor.

Walks data/processed/ai_cleaned/, calls extract_graph_triples.py on every
*_processed.txt file, stores results in the SQLite DB, and writes a progress log.

Usage:
  python scripts/graph/run_graph_extraction.py
  python scripts/graph/run_graph_extraction.py --limit 20   # first 20 docs only (for testing)
  python scripts/graph/run_graph_extraction.py --resume     # skip docs already in DB
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.extract_graph_triples import extract
from scripts.store_graph_triples import store, DB_PATH
from scripts.init_graph_db import init_db

CORPUS_ROOT = Path("data/processed/ai_cleaned")
LOG_PATH = Path("data/knowledge_graph/extraction_log.jsonl")


def already_processed_docs(db_path: Path) -> set[str]:
    if not db_path.exists():
        return set()
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("SELECT DISTINCT source_doc FROM triples")
    docs = {row[0] for row in cur.fetchall()}
    con.close()
    return docs


def main():
    parser = argparse.ArgumentParser(description="Batch knowledge graph extraction")
    parser.add_argument("--limit", type=int, default=None, help="Max number of docs to process")
    parser.add_argument("--resume", action="store_true", help="Skip docs already in DB")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--files", nargs="+", help="Specific files to process (overrides corpus walk)")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[error] GEMINI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    db_path = Path(args.db)
    init_db(db_path)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    done_docs = already_processed_docs(db_path) if args.resume else set()

    if args.files:
        files = [Path(f) for f in args.files]
    else:
        files = sorted(CORPUS_ROOT.rglob("*_processed.txt"))
    print(f"Found {len(files)} processed files in {CORPUS_ROOT}")

    if args.resume and done_docs:
        print(f"Resuming — {len(done_docs)} docs already in DB, skipping them")

    processed = errors = total_triples = 0

    for f in files:
        if args.limit and processed >= args.limit:
            break

        doc_name = f.stem.replace("_processed", "")
        if doc_name in done_docs:
            continue

        safe_name = doc_name.encode("ascii", errors="replace").decode("ascii")
        print(f"\n[{processed+1}] {safe_name}")
        t0 = time.time()
        try:
            triples = extract(str(f), api_key)
            inserted, skipped, rejected = store(triples, db_path)
            elapsed = time.time() - t0
            total_triples += inserted
            print(f"    {len(triples)} extracted, {inserted} stored, {skipped} skipped, {rejected} rejected ({elapsed:.1f}s)")

            LOG_PATH.open("a").write(json.dumps({
                "doc": doc_name,
                "file": str(f),
                "extracted": len(triples),
                "inserted": inserted,
                "elapsed_s": round(elapsed, 1),
            }) + "\n")

        except Exception as e:
            errors += 1
            print(f"    [error] {e}", file=sys.stderr)
            LOG_PATH.open("a").write(json.dumps({
                "doc": doc_name,
                "file": str(f),
                "error": str(e),
            }) + "\n")

        processed += 1
        # Brief pause between calls to avoid rate limiting
        time.sleep(0.5)

    print(f"\nDone. {processed} docs processed, {errors} errors, {total_triples} triples stored.")
    print(f"DB: {db_path.resolve()}")
    print(f"Log: {LOG_PATH.resolve()}")


if __name__ == "__main__":
    main()
