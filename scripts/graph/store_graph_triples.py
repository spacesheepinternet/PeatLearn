"""
Store extracted triples into the knowledge graph SQLite database.

Reads a JSON file (output of extract_graph_triples.py) and inserts every triple.
Skips duplicates: same subject + relationship + object + source_doc.

Usage:
  python scripts/graph/store_graph_triples.py --file triples.json
  python scripts/graph/store_graph_triples.py --file triples.json --db data/knowledge_graph/triples.db
"""

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

from scripts.init_graph_db import DB_PATH, init_db


_DATE_RE = re.compile(r"(\d{4})(?:[_-](\d{2}))?")

# ---------------------------------------------------------------------------
# Entity alias map
# Keys: lowercase (post-canonicalise). Values: display-ready canonical name.
# Purpose: collapse synonyms so "seed oils" and "PUFAs" become one node,
# allowing the 3-doc filter to accumulate them across documents.
# ---------------------------------------------------------------------------
ENTITY_ALIASES: dict[str, str] = {
    # ── PUFAs / unsaturated fats ──────────────────────────────────────────
    "seed oil": "PUFA",
    "seed oils": "PUFA",
    "vegetable oil": "PUFA",
    "vegetable oils": "PUFA",
    "polyunsaturated fat": "PUFA",
    "polyunsaturated fats": "PUFA",
    "polyunsaturated fatty acid": "PUFA",
    "polyunsaturated fatty acids": "PUFA",
    "unsaturated fat": "PUFA",
    "unsaturated fats": "PUFA",
    "linoleic acid": "PUFA",
    "arachidonic acid": "PUFA",
    "corn oil": "PUFA",
    "soybean oil": "PUFA",
    "canola oil": "PUFA",
    "fish oil": "PUFA",
    "omega-6": "PUFA",
    "omega-3": "PUFA",
    "pufa": "PUFA",
    "pufas": "PUFA",

    # ── Thyroid hormones ──────────────────────────────────────────────────
    "thyroid hormones": "thyroid hormone",
    "t3": "T3",
    "triiodothyronine": "T3",
    "t4": "T4",
    "thyroxin": "T4",
    "thyroxine": "T4",
    "levothyroxine": "T4",
    "reverse t3": "reverse T3",
    "reverse-t3": "reverse T3",
    "rt3": "reverse T3",

    # ── Estrogen ──────────────────────────────────────────────────────────
    "estrogens": "estrogen",
    "estrogenic hormones": "estrogen",
    "estradiol": "estrogen",
    "estrone": "estrogen",

    # ── Cortisol ──────────────────────────────────────────────────────────
    "hydrocortisone": "cortisol",
    "glucocorticoid": "cortisol",
    "glucocorticoids": "cortisol",

    # ── Adrenaline ────────────────────────────────────────────────────────
    "adrenalin": "adrenaline",
    "epinephrine": "adrenaline",

    # ── Noradrenaline ─────────────────────────────────────────────────────
    "noradrenalin": "noradrenaline",
    "norepinephrine": "noradrenaline",

    # ── Serotonin ─────────────────────────────────────────────────────────
    "5-ht": "serotonin",
    "5-hydroxytryptamine": "serotonin",

    # ── CO2 ───────────────────────────────────────────────────────────────
    "carbon dioxide": "CO2",
    "co2": "CO2",

    # ── ATP ───────────────────────────────────────────────────────────────
    "adenosine triphosphate": "ATP",
    "atp": "ATP",

    # ── Free fatty acids ──────────────────────────────────────────────────
    "ffa": "free fatty acids",
    "ffas": "free fatty acids",
    "unesterified fatty acids": "free fatty acids",

    # ── Lactic acid ───────────────────────────────────────────────────────
    "lactate": "lactic acid",

    # ── Aspirin ───────────────────────────────────────────────────────────
    "acetylsalicylic acid": "aspirin",

    # ── Vitamins ──────────────────────────────────────────────────────────
    "vitamin a": "vitamin A",
    "retinol": "vitamin A",
    "retinoic acid": "vitamin A",
    "vitamin d": "vitamin D",
    "vitamin d3": "vitamin D",
    "cholecalciferol": "vitamin D",
    "calcitriol": "vitamin D",
    "vitamin e": "vitamin E",
    "tocopherol": "vitamin E",
    "tocopherols": "vitamin E",
    "vitamin k": "vitamin K",
    "vitamin k2": "vitamin K2",
    "menaquinone": "vitamin K2",
    "vitamin c": "vitamin C",
    "ascorbic acid": "vitamin C",

    # ── DHEA ──────────────────────────────────────────────────────────────
    "dhea": "DHEA",
    "dehydroepiandrosterone": "DHEA",

    # ── Prostaglandins ────────────────────────────────────────────────────
    "prostaglandin": "prostaglandins",

    # ── Mitochondria ──────────────────────────────────────────────────────
    "mitochondrion": "mitochondria",

    # ── Blood glucose ─────────────────────────────────────────────────────
    "blood sugar": "blood glucose",
    "blood-sugar": "blood glucose",

    # ── Inflammation ──────────────────────────────────────────────────────
    "inflammatory response": "inflammation",
    "inflammatory process": "inflammation",
    "inflammatory processes": "inflammation",
}


def _canonicalise(name: str) -> str:
    """Lowercase and normalise whitespace. Keeps multi-word concepts intact."""
    return " ".join(name.strip().lower().split())


def _apply_alias(name: str) -> str:
    """After canonicalise, map synonyms to a single display-ready canonical name."""
    return ENTITY_ALIASES.get(name, name)


# ---------------------------------------------------------------------------
# Optional scispaCy / UMLS normalisation
# Requires:
#   pip install scispacy
#   pip install https://s3-us-west-2.amazonaws.com/ai2-s3-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
# The UMLS linker downloads ~2.5 GB on first use (cached afterward).
# If not installed, this step is silently skipped.
# ---------------------------------------------------------------------------
_scispacy_attempted = False      # init runs at most once per process
_scispacy_generator = None       # CandidateGenerator, or None if unavailable

_UMLS_MIN_SCORE = 0.85           # ignore matches below this confidence


def _init_scispacy() -> bool:
    global _scispacy_attempted, _scispacy_generator
    if _scispacy_attempted:
        return _scispacy_generator is not None
    _scispacy_attempted = True
    try:
        from scispacy.candidate_generation import CandidateGenerator  # noqa: PLC0415
        _scispacy_generator = CandidateGenerator(name="umls")
        print("[info] scispaCy UMLS KB loaded", file=sys.stderr)
        return True
    except Exception as e:
        print(f"[info] scispaCy not available ({type(e).__name__}: {e}) — UMLS normalisation skipped",
              file=sys.stderr)
        return False


def _scispacy_normalise(name: str) -> str:
    """Look up name in the UMLS KB via CandidateGenerator.

    Returns the UMLS preferred name (lowercased) if score >= _UMLS_MIN_SCORE,
    otherwise returns the original name unchanged.

    Note: UMLS resolves spelling variants and drug-name synonyms
    (e.g. thyroxin → levothyroxine, triiodothyronine → liothyronine) but does
    NOT collapse class hierarchies (estradiol stays estradiol, not estrogen).
    The manual ENTITY_ALIASES dict handles class-level grouping afterward.
    """
    if not _init_scispacy():
        return name
    try:
        candidates = _scispacy_generator([name], 1)
        if candidates and candidates[0]:
            top = candidates[0][0]
            score = top.similarities[0] if top.similarities else 0.0
            if score >= _UMLS_MIN_SCORE:
                concept_id = top.concept_id
                canonical = _scispacy_generator.kb.cui_to_entity[concept_id].canonical_name
                return canonical.lower()
    except Exception:
        pass
    return name


def _normalise_entity(name: str) -> str:
    """Full entity normalisation pipeline:
      raw string → canonicalise → UMLS lookup → manual alias map
    """
    canonical = _canonicalise(name)
    umls_name = _scispacy_normalise(canonical)
    # Try alias map on UMLS result first; if no hit, try the pre-UMLS form
    result = _apply_alias(umls_name)
    if result == umls_name and umls_name != canonical:
        result = _apply_alias(canonical)
    return result


# ---------------------------------------------------------------------------
# Relationship normaliser
# Maps the infinite variety of Gemini verb phrases → 20 canonical verbs.
# Applied at store time so the graph accumulates edges across documents.
# ---------------------------------------------------------------------------
_REL_PATTERNS: list[tuple[list[str], str]] = [
    # inhibits
    (["inhibit", "suppress", "block", "shut off", "turn off", "stops", "stop ",
      "wipe out", "can be suppressed", "can totally", "interfere", "impair",
      "disrupt", "degrades", "down-regulat"], "inhibits"),
    # reduces
    (["reduc", "lower", "decreas", "diminish", "drop", "will lower", "will decrease",
      "will sometimes lower", "lessen"], "reduces"),
    # increases
    (["increas", "rais", "elevat", "boost", "upregulat", "will increase",
      "is likely to increase", "can increase"], "increases"),
    # causes
    (["caus", "leads to", "result in", "trigger", "is the consequence of",
      "gives rise to", "produces (a condition"], "causes"),
    # promotes
    (["promot", "stimulat", "activat", "driv", "encouraging", "accelerat",
      "keeps stimulating"], "promotes"),
    # protects against
    (["protect", "defend", "guard", "are very protective", "is protective"], "protects against"),
    # worsens
    (["worsen", "exacerbat", "aggravat"], "worsens"),
    # improves
    (["improv", "restor", "correct", "normaliz", "resolv", "can correct",
      "directly correct"], "improves"),
    # produces
    (["produc", "secret", "generat", "synthesiz", "releas", "can secrete",
      "can produce"], "produces"),
    # depletes
    (["deplet", "exhaust", "drain", "uses up"], "depletes"),
    # converts to
    (["convert", "transform", "becomes (a substance", "converted to",
      "is likely to convert", "is permitted to form"], "converts to"),
    # requires
    (["requir", "depends on", "needs", "essential for", "is essential",
      "is necessary", "are main factors"], "requires"),
    # supports
    (["support", "maintain", "sustain", "enabl", "allows", "permits",
      "strongly influence"], "supports"),
    # treats
    (["treat", "therapeutic", "used for", "helps with", "can control",
      "can be controlled", "is used to"], "treats"),
    # is toxic to
    (["toxic", "poison", "harm", "damage", "can have a directly toxic"], "is toxic to"),
    # binds to
    (["bind", "attach", "chelat", "sequest"], "binds to"),
    # competes with
    (["compet", "antagoniz", "oppos"], "competes with"),
    # is converted from
    (["derived from", "comes from", "made from", "is converted from"], "is converted from"),
    # is associated with
    (["associat", "correlat", "accompanies", "involved in", "involves"], "is associated with"),
    # leads to  (distinct from causes — a downstream consequence, not direct cause)
    (["leads to", "lead to"], "leads to"),
    # inhibits (extra catch)
    (["stop ", "stops ", "shut", "turn off", "kill", "doesn't have",
      "prevent ", "prevents ", "can break"], "inhibits"),
    # improves (extra catch)
    (["recover", "clearing up", "can clear", "regressed", "shrink"], "improves"),
    # reduces (extra catch)
    (["will sometimes lower", "rise" ], "reduces"),
]


def _normalise_relationship(rel: str) -> str:
    """Map a free-form Gemini verb phrase to one of the 20 canonical verbs."""
    rel_lower = rel.lower().strip()
    for patterns, canonical in _REL_PATTERNS:
        if any(p in rel_lower for p in patterns):
            return canonical
    return rel  # keep as-is if nothing matches — will show up in audits


def _infer_doc_date(source_doc: str) -> str | None:
    """Best-effort date extraction from document name, e.g. 'newsletter_2004_01' → '2004-01'."""
    m = _DATE_RE.search(source_doc)
    if not m:
        return None
    year = m.group(1)
    month = m.group(2)
    return f"{year}-{month}" if month else year


def _quote_contains_entity(quote: str, entity: str) -> bool:
    """Check that at least one meaningful word from the entity appears in the quote."""
    quote_lower = quote.lower()
    # Strip common stop words that don't anchor the entity
    stop = {"the", "a", "an", "of", "in", "to", "and", "or", "is", "are", "was", "were",
            "for", "with", "by", "from", "its", "their", "this", "that", "it"}
    words = [w for w in re.findall(r"[a-z0-9]+", entity.lower()) if w not in stop and len(w) > 2]
    if not words:
        return True  # Can't validate single stop-word entities, let them through
    return any(w in quote_lower for w in words)


def _is_valid(subject: str, relationship: str, obj: str, verbatim: str) -> bool:
    """Return False for triples that fail the quote-grounding check."""
    # At least one key word from both subject AND object must appear in the quote
    return (
        _quote_contains_entity(verbatim, subject)
        and _quote_contains_entity(verbatim, obj)
    )


def store(triples: list[dict], db_path: Path = DB_PATH) -> tuple[int, int, int]:
    """Insert triples, skip duplicates and ungrounded triples. Returns (inserted, skipped, rejected)."""
    init_db(db_path)
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    inserted = skipped = rejected = 0
    for t in triples:
        subject_raw = _canonicalise(t.get("subject") or "")
        relationship = (t.get("relationship") or "").strip()
        obj_raw = _canonicalise(t.get("object") or "")

        verbatim = (t.get("verbatim_quote") or "").strip()
        source_doc = (t.get("source_doc") or "").strip()

        if not (subject_raw and relationship and obj_raw and verbatim and source_doc):
            skipped += 1
            continue

        # Validate against original names (before aliasing) so "unsaturated fats"
        # is checked against the quote, not the alias "PUFA"
        if not _is_valid(subject_raw, relationship, obj_raw, verbatim):
            rejected += 1
            continue

        # Apply full normalisation pipeline after grounding check passes
        subject = _normalise_entity(subject_raw)
        obj = _normalise_entity(obj_raw)
        relationship = _normalise_relationship(relationship)

        # Deduplicate on (subject, relationship, object, source_doc)
        cur.execute(
            "SELECT 1 FROM triples WHERE subject=? AND relationship=? AND object=? AND source_doc=?",
            (subject, relationship, obj, source_doc),
        )
        if cur.fetchone():
            skipped += 1
            continue

        cur.execute(
            """INSERT INTO triples
               (subject, relationship, object, verbatim_quote, conditional,
                claim_strength, source_doc, source_file, doc_date)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                subject,
                relationship,
                obj,
                verbatim,
                t.get("conditional"),
                t.get("claim_strength", "explicit"),
                source_doc,
                t.get("source_file", ""),
                _infer_doc_date(source_doc),
            ),
        )
        inserted += 1

    con.commit()
    con.close()
    return inserted, skipped, rejected


def main():
    parser = argparse.ArgumentParser(description="Store extracted triples into the graph DB")
    parser.add_argument("--file", required=True, help="JSON file from extract_graph_triples.py")
    parser.add_argument("--db", default=str(DB_PATH), help="Path to SQLite DB")
    args = parser.parse_args()

    db_path = Path(args.db)
    triples = json.loads(Path(args.file).read_text(encoding="utf-8"))
    inserted, skipped, rejected = store(triples, db_path)
    print(f"Done. Inserted: {inserted}, Skipped (dup/invalid): {skipped}, Rejected (ungrounded): {rejected}")


if __name__ == "__main__":
    main()
