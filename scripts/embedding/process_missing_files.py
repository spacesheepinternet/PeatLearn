"""Process the 17 raw files that the original pipeline silently skipped.

Cause: preprocessing/optimized_pipeline.py extension whitelist excluded .html,
and the run was scoped to data/raw/raw_data/ (not data/raw/), so .pdf files
under data/raw/new_content_2026/ were never discovered either.

This script:
  1. Strips HTML to clean text (bs4) / extracts PDF text (pdfplumber)
  2. Runs the existing EnhancedSignalProcessor on the cleaned input
  3. Saves output to data/processed/ai_cleaned/<relative_path>/<stem>_processed.txt

After running this, regenerate corpus_v3 and embed only the new IDs:
    python scripts/embedding/build_corpus_v3.py
    python scripts/embedding/embed_and_upload_v3.py --resume
"""
from __future__ import annotations

import re
import sys
import argparse
import logging
from pathlib import Path

from bs4 import BeautifulSoup
import pdfplumber

# The processor's internal `import rules_based_cleaners as rbc` is a sibling import.
# Add that dir to sys.path so the import resolves.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "preprocessing" / "cleaning"))
from unified_signal_processor_v2 import EnhancedSignalProcessor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("process_missing")

RAW_ROOT = _REPO_ROOT / "data" / "raw"
PROCESSED_ROOT = _REPO_ROOT / "data" / "processed" / "ai_cleaned"
STAGING_ROOT = _REPO_ROOT / "data" / "_missing_files_staging"

# (raw path relative to data/raw, output subdir under data/processed/ai_cleaned)
MISSING_FILES: list[tuple[str, str]] = [
    # 09_Miscellaneous — entire directory dropped by pipeline (HTML extension excluded)
    ("raw_data/09_Miscellaneous/A Renowned Nutritional Counselor Offers His Thoughts About Thyroid Disease.html", "09_Miscellaneous"),
    ("raw_data/09_Miscellaneous/Estrogen - Age Stress Hormone..html", "09_Miscellaneous"),
    ("raw_data/09_Miscellaneous/Negation.html", "09_Miscellaneous"),
    ("raw_data/09_Miscellaneous/On culture, government, and social class.html", "09_Miscellaneous"),
    ("raw_data/09_Miscellaneous/Ray Peat's Brain Part II- An Index Of Terms & Ideas.html", "09_Miscellaneous"),
    ("raw_data/09_Miscellaneous/Ray Peat's Brain- Building a Foundation for Better Understanding.html", "09_Miscellaneous"),
    ("raw_data/09_Miscellaneous/Signs & Symptoms That Respond To Progesterone.html", "09_Miscellaneous"),
    ("raw_data/09_Miscellaneous/Thyroid.html", "09_Miscellaneous"),
    ("raw_data/09_Miscellaneous/Welcome.html", "09_Miscellaneous"),
    ("raw_data/09_Miscellaneous/When Western Medicine Isn’t Working—Different Insights From A Leader In Health.html", "09_Miscellaneous"),
    # 02_Publications/Articles — sibling subdir skipped (HTML extension excluded)
    ("raw_data/02_Publications/Articles/An Interview With Dr. Raymond Peat Part I & II - by Karen Mcc et Matt Labosco, Greg Waitt, Wayde Curran, and Mariam.html", "02_Publications/Articles"),
    ("raw_data/02_Publications/Articles/An Interview With Dr. Raymond Peat Part II- Mind & Body.html", "02_Publications/Articles"),
    ("raw_data/02_Publications/Articles/An Interview With Dr. Raymond Peat- Negation - by Karen Mcc.html", "02_Publications/Articles"),
    ("raw_data/02_Publications/Articles/An Interview With Dr. Raymond Peat- Organizing the Panic - by Karen Mcc et Wayde Curran, Eti Csiga and Tyler Derosier.html", "02_Publications/Articles"),
    ("raw_data/02_Publications/Articles/An Interview With Dr. Raymond Peat.html", "02_Publications/Articles"),
    ("raw_data/02_Publications/Articles/Organizing the Panic - An Interview with Dr. Ray Peat.html", "02_Publications/Articles"),
    # new_content_2026 — outside the scanned input dir
    ("new_content_2026/generative-energy-restoring-the-wholeness-of-life.pdf", "02_Publications/Books"),
]


def strip_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript", "form"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_pdf(path: Path) -> str:
    parts: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                parts.append(t)
    text = "\n\n".join(parts)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def stage_one(raw_rel: str, output_subdir: str) -> Path | None:
    raw_path = RAW_ROOT / raw_rel
    if not raw_path.exists():
        log.warning(f"MISSING raw file: {raw_path}")
        return None

    suffix = raw_path.suffix.lower()
    if suffix == ".html":
        text = strip_html(raw_path.read_text(encoding="utf-8", errors="ignore"))
    elif suffix == ".pdf":
        text = extract_pdf(raw_path)
    else:
        log.warning(f"unsupported extension {suffix} for {raw_path}")
        return None

    if len(text) < 200:
        log.warning(f"extracted text too short ({len(text)} chars) for {raw_path.name}")
        return None

    staged_dir = STAGING_ROOT / output_subdir
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged_path = staged_dir / (raw_path.stem + ".txt")
    staged_path.write_text(text, encoding="utf-8")
    log.info(f"staged {raw_path.name} -> {staged_path.relative_to(_REPO_ROOT)} ({len(text):,} chars)")
    return staged_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage-only", action="store_true", help="extract HTML/PDF to staging dir, do not call AI cleaner")
    ap.add_argument("--limit", type=int, default=None, help="process only first N files (debugging)")
    ap.add_argument("--skip-pdf", action="store_true", help="skip the Generative Energy book PDF")
    args = ap.parse_args()

    targets = MISSING_FILES
    if args.skip_pdf:
        targets = [t for t in targets if not t[0].endswith(".pdf")]
    if args.limit:
        targets = targets[: args.limit]

    log.info(f"processing {len(targets)} missing files")

    staged: list[tuple[Path, str]] = []
    for raw_rel, output_subdir in targets:
        p = stage_one(raw_rel, output_subdir)
        if p:
            staged.append((p, output_subdir))

    log.info(f"staged {len(staged)}/{len(targets)} files to {STAGING_ROOT.relative_to(_REPO_ROOT)}")
    if args.stage_only or not staged:
        return 0

    log.info("running AI cleaner on staged files...")
    processor = EnhancedSignalProcessor()
    # Bug in unified_signal_processor_v2.py: code references self.ai_model but
    # constructor only sets self.client. Patch so the gating checks pass.
    processor.ai_model = processor.client

    successes = 0
    failures = 0
    for staged_path, output_subdir in staged:
        out_dir = PROCESSED_ROOT / output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{staged_path.stem}_processed.txt"

        if out_path.exists():
            log.info(f"already processed: {out_path.relative_to(_REPO_ROOT)} (skip)")
            continue

        result = processor.process_file(staged_path)
        if result.success and result.processed_content:
            out_path.write_text(result.processed_content, encoding="utf-8")
            successes += 1
            log.info(
                f"  -> wrote {out_path.relative_to(_REPO_ROOT)} "
                f"({len(result.processed_content):,} chars, "
                f"signal {result.signal_ratio_before:.2f} -> {result.signal_ratio_after:.2f}, "
                f"~${result.estimated_cost:.4f})"
            )
        else:
            failures += 1
            log.error(f"FAILED: {staged_path.name} ({result.error_message})")

    log.info(f"done: {successes} succeeded, {failures} failed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
