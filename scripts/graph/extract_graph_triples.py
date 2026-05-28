"""
Knowledge graph triple extractor.

Reads a processed corpus document, sends it to Gemini, and extracts
relationship triples in the form:
  (subject) → [relationship] → (object)

Each triple keeps the verbatim Peat sentence it came from.

Usage:
  python scripts/graph/extract_graph_triples.py --file "data/processed/ai_cleaned/09_Miscellaneous/Thyroid_processed.txt"
  python scripts/graph/extract_graph_triples.py --file "..." --out triples.json
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types as genai_types

# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You extract factual relationships from Ray Peat's writing.

Your job is to read the text and find sentences where Peat says that one thing
affects, causes, blocks, promotes, inhibits, depletes, converts to, or otherwise
acts on another thing.

STRICT RULES:
1. Only extract what Peat explicitly states. Do not infer, guess, or connect ideas
   across sentences on your own.
2. Copy the relationship verb EXACTLY as Peat wrote it. If he said "may reduce",
   write "may reduce" — not "reduces". If he said "tends to inhibit", write
   "tends to inhibit". The hedging words matter.
3. The verbatim_quote field must be the exact sentence from the text — word for word.
   Never paraphrase or summarize it.
   CRITICAL: Before writing a triple, re-read the verbatim_quote and confirm that
   the relationship you are writing is actually stated in that sentence. If you
   cannot point to the exact words in the quote that express the relationship, skip
   the triple. Do not infer the relationship from surrounding context.
4. If a claim has a condition attached ("when X is low", "in the context of stress",
   "in some cases"), put that condition in the conditional field.
5. ENTITY NAMES MUST BE SHORT AND CONSISTENT. Use the shortest clear name.
   They are nodes in a graph — they must be reusable across many documents.
   Maximum 3 words. If you need more than 3 words to name something, you are
   describing a process, not naming an entity — skip the triple.

   CORRECT → WRONG:
   "estrogen" → not "estrogenic hormones" or "excess estrogen levels"
   "thyroid hormone" → not "the secretion of hormone by the thyroid gland"
   "beans" → not "diets containing beans" (Rule 8 handles the list expansion)
   "broccoli" → not "undercooked broccoli" (put "undercooked" in conditional)
   "liver" → not "the liver's ability to process estrogen"
   "progesterone" → not "a progesterone deficiency"
   "T3" → not "the active T3 hormone" or "t4 to the active t3"
   "intestine" → not "anything that cleans up your intestine"
   "blood glucose" → not "blood sugar and metabolism stabilization"

   NEVER use a full clause or sentence fragment as an entity name.
   NEVER start an entity name with "anything that", "the ability to",
   "the process of", "a diet containing", or similar descriptive phrases.
6. Never use null for subject or object. If the subject or object of a sentence
   is unclear or too complex to name simply, skip that triple entirely.
7. Do not extract vague philosophical statements. Only extract concrete biological
   or physiological relationships.
8. When one sentence lists multiple items doing the same thing (e.g. "beans,
   lentils, and nuts cause hypothyroidism"), create one triple per item — but
   use the simple item name ("beans", "lentils", "nuts"), not phrases like
   "diets containing beans".
9. The relationship field must be a verb or verb phrase in present tense. Write
   "promotes" not "promoting", "inhibits" not "inhibiting". Copy hedging words
   but keep the verb in present tense. NEVER use a noun or adjective as the
   relationship — words like "problem", "central thing", "part of", "mechanism"
   are not valid relationships. If you cannot express the relationship as a verb,
   skip the triple.
10. Never use vague words as the object or subject: "things", "systems", "nature",
    "quality", "role", "effect", "properties", "ability", "capacity", "processes",
    "everything", "something", "it". If the subject or object is one of these vague
    words, rephrase to name the concrete thing — or skip the triple.
11. Only extract relationships about biology, physiology, nutrition, hormones,
    metabolism, health conditions, or specific substances and their effects on
    the body. Skip ANY sentence about history, politics, war, culture, philosophy,
    economics, religion, media, or social commentary — even if Peat discusses it.
    If you are unsure whether a sentence is about biology, skip it.
12. Skip sentences that are phrased as questions (ending in "?"). Questions are not
    statements of fact, even if they sound like they might be true.
13. Only extract Peat's OWN views and claims. Skip any sentence where Peat is
    reporting, narrating, or arguing against what someone else believes.

    SIGNAL WORDS that mean the claim is NOT Peat's — skip the whole sentence:
    "mainstream", "they say", "they claim", "they believe", "he believed",
    "it was thought", "everyone published", "everything published", "suddenly
    reversing", "they were seeing", "was discovered to", "according to",
    "the idea that", "the theory that", "the claim that", "people think",
    "researchers say", "the literature says", "it is claimed", "as they say".

    CONTRAST WORDS that introduce Peat's actual view after he describes the
    wrong one — extract only what comes AFTER these: "but", "however",
    "in reality", "actually", "I think", "I believe", "I've found",
    "it turns out", "my view", "in my experience".

    SELF-TEST before writing any triple: Ask "Is this something Ray Peat
    would personally recommend or endorse?" If the answer is no or unclear,
    skip the triple.

CANONICAL RELATIONSHIP VOCABULARY:
You MUST use one of the following 20 canonical verbs as the relationship field.
Do not invent new verbs. Pick the closest match.

  inhibits       — blocks, suppresses, shuts off, turns off, stops, prevents activity of
  reduces        — lowers, decreases, diminishes, drops
  increases      — raises, elevates, boosts, upregulates
  causes         — leads to, results in, triggers, produces (a condition)
  promotes       — stimulates, activates, encourages, drives, upregulates activity of
  protects against — defends against, prevents, guards against
  worsens        — aggravates, exacerbates, accelerates (a condition)
  improves       — restores, corrects, normalizes, resolves (a condition)
  produces       — secretes, generates, synthesizes, releases (a substance)
  depletes       — uses up, exhausts, drains
  converts to    — is converted to, transforms into, becomes
  interferes with — disrupts, impairs, degrades function of
  requires       — depends on, needs, is essential for
  supports       — maintains, sustains, enables
  is associated with — correlates with, accompanies (use only when Peat states this explicitly)
  treats         — used for, therapeutic for, helps with (a condition)
  is toxic to    — damages, poisons, harms
  binds to       — attaches to, chelates, sequesters
  competes with  — antagonizes, opposes (at the same receptor or pathway)
  is converted from — comes from, derived from, made from

If the relationship verb in the sentence does not map cleanly to any of these 20,
skip the triple — do not improvise a new verb.

ENTITY CATEGORIES (things that can be subjects or objects):
- Hormones: estrogen, progesterone, cortisol, thyroid hormone, T3, T4, serotonin,
  dopamine, adrenaline, insulin, DHEA, pregnenolone, aldosterone, prolactin
- Nutrients and supplements: magnesium, calcium, vitamin D, vitamin K, vitamin A,
  vitamin B12, zinc, selenium, copper, aspirin, caffeine, niacinamide
- Foods and dietary things: unsaturated fats, saturated fats, sugar, glucose,
  fructose, gelatin, milk, orange juice, coconut oil, PUFA, butter, starch
- Biological molecules: nitric oxide, carbon dioxide, prostaglandins, free fatty
  acids, lactic acid, ATP, histamine, glutamate, adenosine
- Body systems and organs: mitochondria, liver, thyroid gland, adrenal glands,
  brain, pituitary, ovaries
- Conditions: hypothyroidism, hyperthyroidism, cancer, aging, inflammation,
  stress, dementia, Alzheimer's, insomnia, estrogen dominance
- Biological processes: oxidative phosphorylation, glycolysis, mitochondrial
  respiration, lipid peroxidation, cell division, energy production

OUTPUT FORMAT:
Return a JSON array. Each item must have exactly these fields:
{
  "subject": "short entity name (1-4 words, reusable across documents)",
  "relationship": "present-tense verb phrase, hedging preserved",
  "object": "short entity name (1-4 words, reusable across documents)",
  "verbatim_quote": "the exact sentence from the text, word for word",
  "conditional": "any condition Peat attached, or null if none",
  "claim_strength": "explicit" or "implied"
}

claim_strength is "explicit" when Peat directly states the relationship.
claim_strength is "implied" only when the relationship is very clearly intended
but not stated in a single clean sentence.

If you find no valid relationships in the text, return an empty array: []

Return only the JSON array. No explanation, no preamble, no markdown code blocks.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_peat_text(file_path: str) -> str:
    """Read a processed file and return only the RAY PEAT: sections."""
    text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    # Pull out just Peat's words (skip CONTEXT lines)
    chunks = re.findall(r"\*\*RAY PEAT:\*\*\s*(.*?)(?=\*\*CONTEXT:\*\*|\*\*RAY PEAT:\*\*|$)",
                        text, re.DOTALL)
    if not chunks:
        # File may not have the RAY PEAT: markers — return full text
        return text.strip()
    return "\n\n---\n\n".join(c.strip() for c in chunks if c.strip())


def call_gemini(prompt_text: str, api_key: str, retries: int = 3) -> str:
    client = genai.Client(api_key=api_key)
    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt_text,
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.1,
                    max_output_tokens=16384,
                ),
            )
            return response.text.strip()
        except Exception as e:
            if attempt == retries:
                raise
            wait = 10 * attempt  # 10s, 20s
            print(f"[warn] Gemini error (attempt {attempt}/{retries}): {e} — retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)


def clean_json(raw: str) -> str:
    """Strip markdown code fences if Gemini added them."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_MAX_CHARS_PER_CALL = 8_000  # ~2k tokens of input; dense docs can produce 100+ triples per batch


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Split text on --- separators into batches under max_chars each."""
    parts = [p.strip() for p in text.split("---") if p.strip()]
    batches, current = [], []
    current_len = 0
    for part in parts:
        if current_len + len(part) > max_chars and current:
            batches.append("\n\n---\n\n".join(current))
            current, current_len = [], 0
        current.append(part)
        current_len += len(part)
    if current:
        batches.append("\n\n---\n\n".join(current))
    return batches


def extract(file_path: str, api_key: str) -> list[dict]:
    peat_text = extract_peat_text(file_path)
    if not peat_text:
        print(f"[warn] No text found in {file_path}", file=sys.stderr)
        return []

    doc_name = Path(file_path).stem.replace("_processed", "")
    batches = _chunk_text(peat_text, _MAX_CHARS_PER_CALL)
    print(f"[info] {doc_name}: {len(batches)} batch(es)", file=sys.stderr)

    all_triples = []
    for i, batch in enumerate(batches, 1):
        prompt = f"Document: {doc_name} (part {i}/{len(batches)})\n\n{batch}"
        raw = call_gemini(prompt, api_key)
        cleaned = clean_json(raw)
        try:
            triples = json.loads(cleaned)
            all_triples.extend(triples)
            print(f"[info]   batch {i}: {len(triples)} triples", file=sys.stderr)
        except json.JSONDecodeError as e:
            print(f"[error] JSON parse failed on batch {i}: {e}", file=sys.stderr)
            print(f"Raw output (first 500 chars):\n{raw[:500]}", file=sys.stderr)

    # Attach source document to every triple
    for t in all_triples:
        t["source_doc"] = doc_name
        t["source_file"] = str(file_path)

    return all_triples


def main():
    parser = argparse.ArgumentParser(description="Extract knowledge graph triples from a corpus document")
    parser.add_argument("--file", required=True, help="Path to a _processed.txt file")
    parser.add_argument("--out", default=None, help="Write JSON output to this file (default: print to stdout)")
    args = parser.parse_args()

    # Load .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[error] GEMINI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    if not Path(args.file).exists():
        print(f"[error] File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    triples = extract(args.file, api_key)

    if args.out:
        Path(args.out).write_text(json.dumps(triples, indent=2), encoding="utf-8")
        print(f"Wrote {len(triples)} triples to {args.out}")
    else:
        print(json.dumps(triples, indent=2))
        print(f"\n--- {len(triples)} triples extracted ---", file=sys.stderr)


if __name__ == "__main__":
    main()
