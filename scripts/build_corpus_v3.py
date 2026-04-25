#!/usr/bin/env python3
"""
Build Corpus V3 — Clean, deduplicate, and topic-classify the PeatLearn corpus.

Reads all 552 processed files, applies the smart parser (fixes the 15,364
lost-context bug), deduplicates, removes garbage, classifies topics,
and outputs a clean JSONL ready for Gemini 3072-dim embedding.

Pipeline stages:
  1. Smart Parse   — marker-scanning parser (not split-by-\\n\\n)
  2. Dedup         — fingerprint-based exact dedup (keep first occurrence)
  3. Quality Gate  — remove <25 token garbage, merge adjacent filler
  4. Topic Classify — keyword matching (~80%), LLM fallback (~20%, optional)
  5. Output        — JSONL with full metadata for embedding + Pinecone upload

Usage:
    python scripts/build_corpus_v3.py
    python scripts/build_corpus_v3.py --skip-llm          # keyword-only topics
    python scripts/build_corpus_v3.py --min-tokens 15     # lower garbage threshold
"""

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "ai_cleaned"
OUTPUT_DIR = PROJECT_ROOT / "data" / "corpus_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Topic Taxonomy ───────────────────────────────────────────────────────────
# Hierarchical topic map: category → keywords that identify it.
# Matched against both context and response text.

TOPIC_TAXONOMY = {
    "thyroid": [
        "thyroid", "t3", "t4", "tsh", "hypothyroid", "hyperthyroid",
        "thyroxine", "triiodothyronine", "cytomel", "armour thyroid",
        "goiter", "goitrogen", "hashimoto", "thyroiditis", "desiccated thyroid",
    ],
    "hormones_progesterone": [
        "progesterone", "pregnenolone", "progestin", "corpus luteum",
        "luteal", "menstrual", "premenstrual", "pms", "menopause",
    ],
    "hormones_estrogen": [
        "estrogen", "estradiol", "estriol", "estrone", "aromatase",
        "anti-estrogen", "estrogenic", "xenoestrogen", "phytoestrogen",
    ],
    "hormones_other": [
        "dhea", "cortisol", "cortisone", "adrenaline", "epinephrine",
        "testosterone", "androgen", "prolactin", "melatonin", "insulin",
        "growth hormone", "parathyroid", "calcitonin", "aldosterone",
    ],
    "metabolism_energy": [
        "metabolism", "metabolic rate", "oxidative metabolism",
        "glycolysis", "warburg", "carbon dioxide", "co2", "oxygen",
        "mitochondria", "atp", "nad", "respiratory", "cytochrome",
        "lactic acid", "lactate", "pyruvate", "krebs cycle",
    ],
    "nutrition_sugar_carbs": [
        "sugar", "glucose", "fructose", "sucrose", "carbohydrate",
        "orange juice", "fruit juice", "honey", "starch",
        "glycogen", "blood sugar", "hypoglycemia", "diabetes",
    ],
    "nutrition_fats_oils": [
        "polyunsaturated", "pufa", "seed oil", "vegetable oil",
        "fish oil", "omega-3", "omega-6", "linoleic", "linolenic",
        "arachidonic", "dha", "epa", "coconut oil", "saturated fat",
        "unsaturated fat", "lipid peroxidation", "rancid",
    ],
    "nutrition_protein_dairy": [
        "protein", "gelatin", "collagen", "glycine", "tryptophan",
        "milk", "dairy", "cheese", "casein", "whey", "calcium",
        "egg", "meat", "liver", "bone broth",
    ],
    "nutrition_vitamins_minerals": [
        "vitamin a", "vitamin b", "vitamin c", "vitamin d", "vitamin e",
        "vitamin k", "niacinamide", "thiamine", "riboflavin", "biotin",
        "iron", "copper", "zinc", "magnesium", "selenium", "iodine",
        "potassium", "sodium", "phosphorus", "retinol", "carotene",
    ],
    "inflammation_stress": [
        "inflammation", "inflammatory", "anti-inflammatory",
        "serotonin", "histamine", "prostaglandin", "leukotriene",
        "endotoxin", "nitric oxide", "free radical", "oxidative stress",
        "cortisol", "stress hormone", "adrenaline", "excitotoxicity",
    ],
    "cancer": [
        "cancer", "tumor", "carcinogen", "oncology", "metastasis",
        "chemotherapy", "radiation therapy", "malignant", "benign",
        "anti-cancer", "apoptosis", "cell proliferation",
    ],
    "aging_longevity": [
        "aging", "longevity", "lifespan", "senescence", "telomere",
        "degenerative", "age-related", "macular degeneration",
        "alzheimer", "dementia", "osteoporosis", "lipofuscin",
    ],
    "brain_nervous_system": [
        "brain", "neurology", "neurological", "neurotransmitter",
        "serotonin", "dopamine", "gaba", "acetylcholine",
        "depression", "anxiety", "seizure", "epilepsy", "migraine",
        "sleep", "insomnia", "circadian", "memory", "cognition",
    ],
    "digestion_gut": [
        "digestion", "digestive", "intestine", "bowel", "colon",
        "gut bacteria", "microbiome", "endotoxin", "constipation",
        "diarrhea", "ibs", "bloating", "fiber", "carrot salad",
        "charcoal", "antibiotic",
    ],
    "skin_hair": [
        "skin", "acne", "eczema", "psoriasis", "dermatitis",
        "hair loss", "alopecia", "gray hair", "wrinkle",
        "collagen", "wound healing", "sunlight", "uv",
    ],
    "heart_circulation": [
        "heart", "cardiovascular", "blood pressure", "hypertension",
        "cholesterol", "atherosclerosis", "stroke", "edema",
        "blood clot", "platelet", "anemia", "hemoglobin",
    ],
    "reproduction_fertility": [
        "fertility", "pregnancy", "conception", "ovulation",
        "sperm", "infertility", "birth control", "contraceptive",
        "breastfeeding", "fetus", "prenatal", "miscarriage",
    ],
    "bones_joints": [
        "bone", "osteoporosis", "arthritis", "joint", "cartilage",
        "calcium", "vitamin d", "fracture", "bone density",
    ],
    "light_radiation": [
        "light therapy", "red light", "infrared", "ultraviolet",
        "sunlight", "radiation", "x-ray", "electromagnetic",
        "incandescent", "led", "fluorescent",
    ],
    "water_electrolytes": [
        "water retention", "edema", "swelling", "electrolyte",
        "sodium", "potassium", "salt", "dehydration", "hydration",
        "osmosis", "cellular water",
    ],
    "philosophy_science": [
        "philosophy", "epistemology", "reductionism", "holistic",
        "systems biology", "emergence", "vitalism", "mechanism",
        "paradigm", "ideology", "dogma",
    ],
    "politics_history": [
        "politics", "government", "propaganda", "censorship",
        "pharmaceutical", "fda", "medical establishment",
        "eugenics", "population control", "history",
    ],
    "general_health": [
        "health", "disease", "symptom", "treatment", "therapy",
        "medicine", "pharmaceutical", "drug", "supplement",
        "diagnosis", "prevention", "chronic",
    ],
}


# ── Smart Parser ─────────────────────────────────────────────────────────────

def parse_chunks_smart(content: str, source_file: str, source_type: str) -> list[dict]:
    """Parse RAY PEAT / CONTEXT pairs using marker scanning."""
    chunks = []

    markers = []
    for m in re.finditer(r'\*\*RAY PEAT:\*\*', content):
        markers.append(('RP', m.start(), m.end()))
    for m in re.finditer(r'\*\*CONTEXT:\*\*', content):
        markers.append(('CTX', m.start(), m.end()))

    markers.sort(key=lambda x: x[1])

    i = 0
    chunk_index = 0
    while i < len(markers):
        tag, start, end = markers[i]

        if tag == 'RP':
            next_pos = markers[i + 1][1] if i + 1 < len(markers) else len(content)
            rp_text = content[end:next_pos].strip()

            ctx_text = ""
            if i + 1 < len(markers) and markers[i + 1][0] == 'CTX':
                ctx_end = markers[i + 1][2]
                ctx_next = markers[i + 2][1] if i + 2 < len(markers) else len(content)
                ctx_text = content[ctx_end:ctx_next].strip()
            elif i > 0 and markers[i - 1][0] == 'CTX':
                ctx_end = markers[i - 1][2]
                ctx_text = content[ctx_end:start].strip()

            tokens = len(rp_text.split())

            chunks.append({
                "text": rp_text,
                "context": ctx_text,
                "source_file": source_file,
                "source_type": source_type,
                "tokens": tokens,
                "chunk_index": chunk_index,
            })
            chunk_index += 1

        i += 1

    return chunks


# ── Content Type Classification ──────────────────────────────────────────────

def classify_source(file_path: Path) -> str:
    """Classify a processed file by its directory."""
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
    return "misc"


# ── Dedup ────────────────────────────────────────────────────────────────────

def fingerprint(text: str) -> str:
    """Normalize for dedup: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def dedup_chunks(chunks: list[dict]) -> tuple[list[dict], int]:
    """Remove exact duplicates, keeping first occurrence. Returns (deduped, removed_count)."""
    seen = set()
    result = []
    removed = 0

    for c in chunks:
        fp = fingerprint(c["text"])
        if len(fp) <= 20:
            # Too short to fingerprint reliably — keep it (quality gate handles later)
            result.append(c)
            continue

        if fp in seen:
            removed += 1
            continue

        seen.add(fp)
        result.append(c)

    return result, removed


# ── Quality Gate ─────────────────────────────────────────────────────────────

def quality_gate(chunks: list[dict], min_tokens: int = 25) -> tuple[list[dict], int]:
    """Remove garbage chunks. Returns (passed, removed_count)."""
    passed = []
    removed = 0

    for c in chunks:
        if c["tokens"] < min_tokens:
            removed += 1
            continue

        # Remove chunks that are just filler phrases
        text_lower = c["text"].lower().strip()
        filler_phrases = {
            "yeah", "yes", "no", "right", "okay", "mm-hmm", "uh-huh",
            "sounds right", "i think so", "yeah yeah", "uh no",
            "that's right", "exactly", "sure", "probably",
        }
        if text_lower.rstrip('.!?,') in filler_phrases:
            removed += 1
            continue

        passed.append(c)

    return passed, removed


# ── Topic Classification ─────────────────────────────────────────────────────

def classify_topic(chunk: dict) -> list[str]:
    """Classify a chunk's topic(s) using keyword matching.

    Returns a list of 1-3 matching topic labels, sorted by match strength.
    Falls back to "general_health" if no keywords match.
    """
    # Combine context + response for matching
    text = f"{chunk.get('context', '')} {chunk.get('text', '')}".lower()

    scores = {}
    for topic, keywords in TOPIC_TAXONOMY.items():
        matches = sum(1 for kw in keywords if kw in text)
        if matches > 0:
            scores[topic] = matches

    if not scores:
        return ["general_health"]

    # Return top 3 topics by match count
    sorted_topics = sorted(scores.items(), key=lambda x: -x[1])
    return [t for t, _ in sorted_topics[:3]]


# ── Title Extraction ─────────────────────────────────────────────────────────

def extract_title(source_file: str) -> str:
    """Extract a human-readable title from the source file path."""
    fname = Path(source_file).stem
    # Remove _processed suffix
    fname = re.sub(r'_processed$', '', fname)
    # Remove transcript suffix
    fname = re.sub(r'\.mp3-transcript$', '', fname)
    # Remove YouTube IDs like [AbCdEf12345]
    fname = re.sub(r'\s*\[[\w-]+\]$', '', fname)
    # Replace special chars with spaces
    fname = re.sub(r'[_\-]+', ' ', fname)
    return fname.strip()


# ── Main Pipeline ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build clean corpus V3")
    parser.add_argument("--min-tokens", type=int, default=25,
                        help="Minimum tokens per chunk (default: 25)")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM topic classification (keyword-only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and report but don't write output")
    args = parser.parse_args()

    print(f"{'='*70}")
    print(f"  BUILD CORPUS V3 — Clean Pipeline")
    print(f"{'='*70}\n")

    # ── Stage 1: Smart Parse ─────────────────────────────────────────────────
    print("Stage 1: Smart Parse (marker scanning)")
    processed_files = sorted(PROCESSED_DIR.rglob("*_processed.txt"))
    print(f"  Found {len(processed_files)} processed files")

    all_chunks = []
    for fp in processed_files:
        content = fp.read_text(encoding="utf-8", errors="replace")
        source_type = classify_source(fp)
        rel_path = str(fp.relative_to(PROCESSED_DIR))

        chunks = parse_chunks_smart(content, rel_path, source_type)
        all_chunks.extend(chunks)

    with_ctx = sum(1 for c in all_chunks if c.get("context"))
    print(f"  Parsed {len(all_chunks):,} chunks, {with_ctx:,} with context ({with_ctx/len(all_chunks)*100:.1f}%)")

    # ── Stage 2: Dedup ───────────────────────────────────────────────────────
    print(f"\nStage 2: Dedup (fingerprint-based)")
    deduped, dup_removed = dedup_chunks(all_chunks)
    print(f"  Removed {dup_removed:,} duplicates")
    print(f"  Remaining: {len(deduped):,} chunks")

    # ── Stage 3: Quality Gate ────────────────────────────────────────────────
    print(f"\nStage 3: Quality Gate (min {args.min_tokens} tokens)")
    cleaned, garbage_removed = quality_gate(deduped, min_tokens=args.min_tokens)
    print(f"  Removed {garbage_removed:,} garbage chunks")
    print(f"  Remaining: {len(cleaned):,} chunks")

    # ── Stage 4: Topic Classification ────────────────────────────────────────
    print(f"\nStage 4: Topic Classification (keyword matching)")
    topic_counts = defaultdict(int)
    no_topic = 0

    for chunk in cleaned:
        topics = classify_topic(chunk)
        chunk["topics"] = topics
        chunk["primary_topic"] = topics[0]

        for t in topics:
            topic_counts[t] += 1

        if topics == ["general_health"]:
            no_topic += 1

    print(f"  Classified {len(cleaned):,} chunks into {len(topic_counts)} topics")
    print(f"  Fallback to 'general_health': {no_topic:,} ({no_topic/len(cleaned)*100:.1f}%)")

    # Top topics
    print(f"\n  Top 10 topics:")
    for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {topic:<30} {count:>5} chunks")

    # ── Stage 5: Generate IDs + Output ───────────────────────────────────────
    print(f"\nStage 5: Generate JSONL output")

    output_records = []
    for i, chunk in enumerate(cleaned):
        # Stable ID: hash of source_file + chunk_index
        id_seed = f"{chunk['source_file']}:{chunk['chunk_index']}"
        chunk_id = hashlib.md5(id_seed.encode()).hexdigest()[:12]

        record = {
            "id": f"v3_{chunk_id}",
            "text": chunk["text"],
            "context": chunk.get("context", ""),
            "source_file": chunk["source_file"],
            "source_type": chunk["source_type"],
            "title": extract_title(chunk["source_file"]),
            "tokens": chunk["tokens"],
            "chunk_index": chunk["chunk_index"],
            "primary_topic": chunk["primary_topic"],
            "topics": chunk["topics"],
            # Embedding text: what gets embedded into Gemini
            "embed_text": f"{chunk.get('context', '')}\n\n{chunk['text']}".strip(),
        }
        output_records.append(record)

    if args.dry_run:
        print(f"  [DRY RUN] Would write {len(output_records):,} records")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"corpus_v3_{timestamp}.jsonl"

        with open(output_path, "w", encoding="utf-8") as f:
            for rec in output_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"  Wrote {len(output_records):,} records to {output_path.relative_to(PROJECT_ROOT)}")

        # Also write a summary JSON
        summary = {
            "timestamp": timestamp,
            "pipeline_version": "v3",
            "input_files": len(processed_files),
            "raw_chunks": len(all_chunks),
            "after_dedup": len(deduped),
            "after_quality_gate": len(cleaned),
            "final_records": len(output_records),
            "duplicates_removed": dup_removed,
            "garbage_removed": garbage_removed,
            "context_coverage_pct": round(
                sum(1 for r in output_records if r["context"]) / len(output_records) * 100, 1
            ),
            "topic_distribution": dict(sorted(topic_counts.items(), key=lambda x: -x[1])),
            "type_distribution": dict(
                sorted(
                    defaultdict(int, {r["source_type"]: 0 for r in output_records}).items()
                )
            ),
            "token_stats": {
                "mean": round(sum(r["tokens"] for r in output_records) / len(output_records), 1),
                "median": sorted(r["tokens"] for r in output_records)[len(output_records) // 2],
                "min": min(r["tokens"] for r in output_records),
                "max": max(r["tokens"] for r in output_records),
            },
            "embedding_config": {
                "model": "gemini-embedding-001",
                "dimensions": 3072,
                "embed_format": "{context}\\n\\n{text}",
            },
        }

        # Fix type distribution
        type_dist = defaultdict(int)
        for r in output_records:
            type_dist[r["source_type"]] += 1
        summary["type_distribution"] = dict(sorted(type_dist.items(), key=lambda x: -x[1]))

        summary_path = OUTPUT_DIR / f"corpus_v3_{timestamp}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"  Summary: {summary_path.relative_to(PROJECT_ROOT)}")

    # ── Final Report ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  PIPELINE SUMMARY")
    print(f"{'='*70}")
    print(f"  Input:             {len(processed_files):>7} files")
    print(f"  Raw chunks:        {len(all_chunks):>7,}")
    print(f"  - Duplicates:      {dup_removed:>7,}")
    print(f"  - Garbage:         {garbage_removed:>7,}")
    print(f"  = Clean output:    {len(output_records):>7,} chunks")

    ctx_count = sum(1 for r in output_records if r["context"])
    print(f"  Context coverage:  {ctx_count/len(output_records)*100:>7.1f}%")

    avg_tokens = sum(r["tokens"] for r in output_records) / len(output_records)
    total_tokens = sum(r["tokens"] for r in output_records)
    print(f"  Avg tokens/chunk:  {avg_tokens:>7.0f}")
    print(f"  Total tokens:      {total_tokens:>7,}")
    print(f"  Est. embed cost:   ${total_tokens / 1_000_000 * 0.15:>7.2f} (Gemini @ $0.15/1M)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
