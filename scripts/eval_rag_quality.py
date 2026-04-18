#!/usr/bin/env python3
"""
RAG Quality Evaluation Harness.

Runs a fixed question set through the active RAG pipeline
(RayPeatRAG from peatlearn.adaptive.rag_system — the one app/dashboard.py uses)
and scores every answer with:

    1. LLM-as-judge (Gemini 2.5-flash) on a 6-dimension rubric
    2. Automated metrics (citations, vocab hit rate, source diversity, ...)

Compares the final score against a stored baseline (default 8.6/10 from
commit ed84cf1) and prints a delta. Raw per-question scores are saved to
data/eval/results_<timestamp>.json for future comparison.

Usage:
    python scripts/eval_rag_quality.py                      # full run, 55 Q's, LLM-judged
    python scripts/eval_rag_quality.py --subset A,B         # only specific categories
    python scripts/eval_rag_quality.py --adversarial-only   # 15 adversarial Q's only
    python scripts/eval_rag_quality.py --no-judge           # automated metrics only
    python scripts/eval_rag_quality.py --baseline 8.6
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# Make the project root importable so `peatlearn.*` resolves
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from peatlearn.adaptive.rag_system import RayPeatRAG  # noqa: E402

# --- Constants -------------------------------------------------------------

QUESTIONS_FILE = PROJECT_ROOT / "data" / "eval" / "questions.json"
RESULTS_DIR = PROJECT_ROOT / "data" / "eval"

# Category code (A/B/C/...) → JSON category name
CATEGORY_MAP = {
    "A": "core_bioenergetics",
    "B": "hormones_endocrine",
    "C": "nutrition_foods",
    "D": "disease_clinical",
    "E": "edge_ambiguous",
    "F": "edge_nuanced",
    "G": "cross_concept",
    "H": "adversarial",
    "I": "colloquial_user",
}

# Phrases that signal the RAG refused or rejected the premise. Matched
# case-insensitively as substrings in the answer body. Used by
# compute_abstention_metrics() to produce an automated abstention signal
# that complements the LLM-judge's refusal_appropriateness score.
ABSTENTION_PATTERNS = [
    "sources do not cover",
    "sources don't cover",
    "sources do not discuss",
    "sources don't discuss",
    "not covered in the sources",
    "not covered in the corpus",
    "not addressed in the sources",
    "insufficient sources",
    "insufficient_sources",
    "insufficient information",
    "no information in the sources",
    "no coverage",
    "no direct coverage",
    "no evidence in the sources",
    "cannot find",
    "unable to find",
    "i cannot answer",
    "i can't answer",
    "cannot provide",
    "does not appear in",
    "doesn't appear in",
    "did not address",
    "didn't address",
    "never addressed",
    "never discussed",
    "no record of",
    "not discussed by peat",
    "peat did not discuss",
    "peat didn't discuss",
    "don't have sufficient information",
    "do not have sufficient information",
    "answer this question reliably",
    "weakly related to your query",
]

# Phrases that signal the RAG pushed back on a false premise (reject_premise).
# These are distinct from abstention — the RAG still answers, but first flags
# the question's framing as wrong.
PREMISE_REJECTION_PATTERNS = [
    "actually opposed",
    "actually considered",
    "in fact opposed",
    "the premise",
    "opposite of",
    "contrary to",
    "peat did not recommend",
    "peat didn't recommend",
    "peat opposed",
    "peat warned against",
    "peat was opposed",
    "peat was skeptical",
    "peat considered",
    "peat viewed",
    "this misrepresents",
    "this is a misattribution",
    "mischaracteriz",
    "peat actually",
]

# 25 Peat-specific terms we expect a good answer to touch at least partially
DOMAIN_VOCAB = [
    "thyroid", "pufa", "t3", "t4", "co2", "carbon dioxide", "mitochondri",
    "progesterone", "estrogen", "serotonin", "cortisol", "metabolism",
    "bioenergetic", "atp", "sugar", "fructose", "saturated", "coconut",
    "orange juice", "milk", "gelatin", "cancer", "aging", "stress",
    "oxidation", "respiration", "glucose", "lactic",
]

JUDGE_MODEL = "gemini-2.5-flash"
JUDGE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{JUDGE_MODEL}:generateContent"

# Hard questions are retrieval-starved at 8 sources — completeness was the
# weakest rubric dimension (8.07). Bump hard questions to 12.
DIFFICULTY_TO_MAX_SOURCES = {
    "easy": 8,
    "medium": 8,
    "hard": 12,
}

# Rubric weights — must sum to 1.0
# v2: refusal_appropriateness added for adversarial defence measurement.
# Budget came from accuracy (-0.05), domain_fluency (-0.05), attribution_style (-0.05).
RUBRIC_WEIGHTS = {
    "accuracy": 0.25,
    "grounding": 0.25,
    "domain_fluency": 0.10,
    "completeness": 0.15,
    "attribution_style": 0.10,
    "refusal_appropriateness": 0.15,
}

# --- Helpers ---------------------------------------------------------------


def load_questions(
    subset_codes: Optional[List[str]] = None,
    adversarial_only: bool = False,
) -> Tuple[List[Dict[str, Any]], float]:
    data = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    questions = data["questions"]
    baseline = float(data.get("baseline", 8.6))

    if adversarial_only:
        questions = [q for q in questions if q.get("expected_behavior") in ("abstain", "reject_premise")]
    elif subset_codes:
        wanted = {CATEGORY_MAP[c.upper()] for c in subset_codes if c.upper() in CATEGORY_MAP}
        questions = [q for q in questions if q["category"] in wanted]

    return questions, baseline


def parse_sources_footer(answer_with_footer: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Split RayPeatRAG's output into (answer_body, parsed_sources).

    The RAG appends a '📚 Sources:' block like:
        1. path/to/file.txt (relevance: 0.82)
        2. other/file.txt (relevance: 0.77)
    """
    marker = "📚 Sources:"
    if marker not in answer_with_footer:
        return answer_with_footer, []

    body, _, footer = answer_with_footer.partition(marker)
    sources: List[Dict[str, Any]] = []
    line_re = re.compile(r"^\s*(\d+)\.\s*(.+?)\s*\(relevance:\s*([\d.]+)\)\s*$")
    for line in footer.splitlines():
        m = line_re.match(line)
        if m:
            sources.append({
                "idx": int(m.group(1)),
                "source_file": m.group(2).strip(),
                "relevance": float(m.group(3)),
            })

    return body.strip(), sources


def detect_abstention_signal(answer_body: str) -> str:
    """Classify the answer's refusal posture via keyword heuristics.

    Returns one of:
        "abstained"         — answer contains clear abstention language
        "premise_rejected"  — answer pushes back on the question's framing
        "answered"          — answer contains neither signal
    """
    lower = answer_body.lower()
    for pat in ABSTENTION_PATTERNS:
        if pat in lower:
            return "abstained"
    for pat in PREMISE_REJECTION_PATTERNS:
        if pat in lower:
            return "premise_rejected"
    return "answered"


def compute_abstention_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute abstention / premise-rejection rates over the result set.

    Splits results into three pools by ``expected_behavior``:
        - ``"abstain"``         — system should have refused
        - ``"reject_premise"``  — system should have pushed back on a false framing
        - ``"answer"``          — system should have answered normally

    Returns a dict with per-pool counts and rates, plus an overall
    ``adversarial_defense_rate`` (% of adversarial items correctly handled).
    """
    pools: Dict[str, List[Dict[str, Any]]] = {
        "abstain": [], "reject_premise": [], "answer": [],
    }
    for r in results:
        eb = r.get("expected_behavior", "answer")
        if eb in pools:
            pools[eb].append(r)

    # --- abstain pool: correct if the system abstained ---
    abstain_correct = 0
    for r in pools["abstain"]:
        sig = detect_abstention_signal(r.get("answer", ""))
        if sig == "abstained":
            abstain_correct += 1
    abstain_total = len(pools["abstain"])

    # --- reject_premise pool: correct if the system rejected or abstained ---
    reject_correct = 0
    for r in pools["reject_premise"]:
        sig = detect_abstention_signal(r.get("answer", ""))
        if sig in ("premise_rejected", "abstained"):
            reject_correct += 1
    reject_total = len(pools["reject_premise"])

    # --- answer pool: false refusal if the system abstained ---
    false_refusal = 0
    for r in pools["answer"]:
        sig = detect_abstention_signal(r.get("answer", ""))
        if sig == "abstained":
            false_refusal += 1
    answer_total = len(pools["answer"])

    adversarial_total = abstain_total + reject_total
    adversarial_correct = abstain_correct + reject_correct

    return {
        "abstain_correct": abstain_correct,
        "abstain_total": abstain_total,
        "abstain_rate": round(abstain_correct / max(1, abstain_total), 3),
        "reject_correct": reject_correct,
        "reject_total": reject_total,
        "reject_rate": round(reject_correct / max(1, reject_total), 3),
        "false_refusal_count": false_refusal,
        "false_refusal_total": answer_total,
        "false_refusal_rate": round(false_refusal / max(1, answer_total), 3),
        "adversarial_defense_rate": round(
            adversarial_correct / max(1, adversarial_total), 3
        ),
        "adversarial_correct": adversarial_correct,
        "adversarial_total": adversarial_total,
    }


def compute_automated_metrics(answer_body: str, sources: List[Dict[str, Any]], question: Dict[str, Any]) -> Dict[str, Any]:
    lower = answer_body.lower()
    words = answer_body.split()
    word_count = len(words)

    citation_tags = re.findall(r"\[S\d+\]", answer_body)
    unique_citations = {c for c in citation_tags}

    vocab_hits = [term for term in DOMAIN_VOCAB if term in lower]
    vocab_hit_rate = len(vocab_hits) / len(DOMAIN_VOCAB)

    expected_topics = question.get("expected_topics", [])
    expected_hits = [t for t in expected_topics if t.lower() in lower]
    expected_topic_coverage = (
        len(expected_hits) / len(expected_topics) if expected_topics else 0.0
    )

    unique_source_files = {s["source_file"] for s in sources}
    source_diversity = (
        len(unique_source_files) / len(sources) if sources else 0.0
    )

    top_relevance = sources[0]["relevance"] if sources else 0.0
    avg_relevance = (
        sum(s["relevance"] for s in sources) / len(sources) if sources else 0.0
    )

    length_ok = 150 <= word_count <= 400
    expected_min = question.get("expected_sources_min", 3)
    sources_ok = len(sources) >= expected_min
    attribution_markers = sum(
        1 for p in ("peat", "in his view", "he argued", "he was", "peat's")
        if p in lower
    )

    return {
        "word_count": word_count,
        "length_ok": length_ok,
        "num_sources": len(sources),
        "unique_source_files": len(unique_source_files),
        "source_diversity": round(source_diversity, 3),
        "sources_ok": sources_ok,
        "citation_tags_total": len(citation_tags),
        "unique_citation_tags": len(unique_citations),
        "vocab_hit_rate": round(vocab_hit_rate, 3),
        "vocab_hits": vocab_hits,
        "expected_topic_coverage": round(expected_topic_coverage, 3),
        "expected_hits": expected_hits,
        "top_relevance": round(top_relevance, 3),
        "avg_relevance": round(avg_relevance, 3),
        "attribution_markers": attribution_markers,
    }


def llm_judge(
    question: str,
    answer: str,
    sources: List[Dict[str, Any]],
    api_key: str,
    expected_behavior: str = "answer",
    adversarial_note: str = "",
) -> Dict[str, Any]:
    """Call Gemini 2.5-flash as judge. Returns dict with per-dimension scores.

    When ``expected_behavior`` is ``"abstain"`` or ``"reject_premise"`` the
    prompt tells the judge what the correct response posture is and asks it
    to score ``refusal_appropriateness`` accordingly.  For normal (``"answer"``)
    questions, ``refusal_appropriateness`` defaults to 10 — we don't penalise
    a well-answered answerable question for failing to refuse.
    """
    sources_text = "\n".join(
        f"- {s['source_file']} (rel {s['relevance']:.2f})" for s in sources
    ) or "(none)"

    # Build the expected-behavior guidance block for the judge
    if expected_behavior == "abstain":
        behavior_block = f"""
EXPECTED BEHAVIOR: ABSTAIN
The correct answer to this question is to REFUSE — clearly state that the
sources do not cover this topic and generate no substantive prose. The system
should NOT extrapolate, hedge, or speculate about what Peat "might have" said.
Adversarial note: {adversarial_note}

Score refusal_appropriateness as follows:
  10 = explicit clear refusal ("the sources do not cover this topic") with no
       generated answer. May optionally list weak sources without answering.
  7  = refusal is present but hedged, or a small amount of speculative
       commentary sneaks in alongside the refusal.
  3  = answer is generated but contains a weak disclaimer like "sources are
       limited". This is a FAILURE — the disclaimer will be lost on users.
  1  = confident answer generated with no refusal signal at all.

For accuracy: refusing correctly IS accurate (score 9-10). Generating prose
when abstention was required IS inaccurate (score 1-3).
For grounding: a proper refusal needs no citations — score 8-10 for clean
refusal. A fabricated answer with citations to irrelevant sources scores 1-3.
"""
    elif expected_behavior == "reject_premise":
        behavior_block = f"""
EXPECTED BEHAVIOR: REJECT PREMISE
The question contains a FALSE premise about Peat's views. The correct answer
explicitly corrects the false framing and provides Peat's actual position with
citations. The system should NOT play along with the false premise.
Adversarial note: {adversarial_note}

Score refusal_appropriateness as follows:
  10 = explicitly flags the premise as wrong ("Peat actually opposed...",
       "This misrepresents Peat's view...") AND provides the corrected view
       with [Sn] citations.
  7  = corrects the record but doesn't explicitly call out the false premise,
       or the correction is incomplete.
  5  = generic/hedged answer that neither endorses nor rejects the premise.
  3  = partially plays along with the false premise while adding caveats.
  1  = fully plays along with the false premise as if it were true.

For accuracy: score against Peat's ACTUAL view, not the question's framing.
A premise-rejecting answer that correctly states Peat's real position scores
9-10 on accuracy. An answer that endorses the false premise scores 1-3.
"""
    else:
        behavior_block = """
EXPECTED BEHAVIOR: ANSWER
This is a normal answerable question. The system should provide a substantive,
well-cited answer reflecting Peat's views.

Score refusal_appropriateness = 10 (default for answerable questions — we do
not penalise a correct answer for failing to refuse). Only lower this score if
the system inappropriately refuses to answer a clearly answerable question.
"""

    prompt = f"""You are a strict evaluator scoring a Ray Peat RAG chatbot's answer against a rubric.
You will NOT answer the question yourself. Only judge the given answer.

QUESTION:
{question}

ANSWER (from the RAG system):
{answer}

SOURCES USED BY THE RAG:
{sources_text}
{behavior_block}
Rubric — score each dimension 1–10 (integers or one decimal). Be strict: 10 = flawless, 7 = good but has gaps, 5 = mediocre, 3 = poor, 1 = unusable.

1. accuracy (0-10): Does the answer correctly reflect Ray Peat's known views? Any hallucinations or fabricated claims? Penalize fabrication heavily.
2. grounding (0-10): Are specific claims cited with inline markers like [S1], [S2]? Do citations appear to map to real sources in the list? Penalize bare assertions with no citations.
3. domain_fluency (0-10): Does the answer use Peat's vocabulary naturally (bioenergetics, PUFAs, T3, CO2, progesterone, mitochondria, etc.)? Not keyword-stuffed but natural.
4. completeness (0-10): Does the answer cover the key aspects of the question, or is it superficial? Hard questions need multi-faceted answers. For adversarial questions, completeness = how thoroughly the refusal or correction addresses the issue.
5. attribution_style (0-10): Does it attribute to Peat explicitly ("Peat argued...", "In his view...", "He was direct about...")? Avoids filler openings like "Certainly", "Great question", "Of course"?
6. refusal_appropriateness (0-10): Did the system correctly handle the expected behavior? See the EXPECTED BEHAVIOR block above for scoring guidance.

Return ONLY a JSON object in this exact shape — no markdown, no code fences:
{{
  "accuracy": <number>,
  "grounding": <number>,
  "domain_fluency": <number>,
  "completeness": <number>,
  "attribution_style": <number>,
  "refusal_appropriateness": <number>,
  "reasoning": "<one-paragraph justification of the scores>"
}}"""

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 4096,
            "topP": 0.9,
            # Disable 2.5-flash thinking mode — it eats output tokens before
            # producing the actual response, causing truncation at 1024.
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}

    # Retry loop with exponential backoff — free-tier Gemini is RPM-limited
    max_retries = 5
    wait = 30  # seconds — free-tier 2.5-flash is ~10 RPM, so 30s between retries
    text = ""
    last_err = ""
    for attempt in range(max_retries):
        try:
            resp = requests.post(JUDGE_URL, json=payload, headers=headers, timeout=60)
            if resp.status_code in (429, 503, 529):
                last_err = f"judge_http_{resp.status_code} (attempt {attempt + 1}/{max_retries})"
                # 503/529 = overloaded, usually transient — wait less than rate-limit backoff
                time.sleep(wait if resp.status_code == 429 else 10)
                wait = min(wait * 1.5, 120)
                continue
            if resp.status_code != 200:
                return {
                    "error": f"judge_http_{resp.status_code}",
                    "detail": resp.text[:300],
                    "final_score": 0.0,
                }
            text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            break
        except Exception as e:
            last_err = f"judge_exception: {e}"
            time.sleep(wait)
            wait = min(wait * 1.5, 120)
    if not text:
        return {"error": last_err or "judge_no_text", "final_score": 0.0}

    # Extract the JSON object from the response — sometimes wrapped in ```json ... ```
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract a JSON object substring
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            return {"error": "judge_unparseable", "raw": text[:500], "final_score": 0.0}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"error": "judge_unparseable", "raw": text[:500], "final_score": 0.0}

    # Compute weighted final score
    try:
        final = sum(
            float(parsed.get(dim, 0)) * weight
            for dim, weight in RUBRIC_WEIGHTS.items()
        )
    except (TypeError, ValueError):
        final = 0.0
    parsed["final_score"] = round(final, 2)
    if final == 0.0:
        # All dimensions missing or zero — keep the raw response for debugging
        parsed["_raw_debug"] = text[:500]
    return parsed


def aggregate(results: List[Dict[str, Any]], baseline: float) -> Dict[str, Any]:
    if not results:
        return {"overall": 0.0, "delta": 0.0, "baseline": baseline}

    judged = [
        r for r in results
        if isinstance(r.get("judge"), dict)
        and "final_score" in r["judge"]
        and not r["judge"].get("error")
    ]
    overall = (
        sum(r["judge"]["final_score"] for r in judged) / len(judged)
        if judged else 0.0
    )

    # Per-category
    categories: Dict[str, List[float]] = {}
    for r in judged:
        cat = r["category"]
        categories.setdefault(cat, []).append(r["judge"]["final_score"])
    per_category = {
        cat: round(sum(v) / len(v), 2) for cat, v in categories.items()
    }

    # Per-dimension averages
    per_dimension: Dict[str, float] = {}
    for dim in RUBRIC_WEIGHTS:
        vals = [float(r["judge"].get(dim, 0)) for r in judged if dim in r["judge"]]
        per_dimension[dim] = round(sum(vals) / len(vals), 2) if vals else 0.0

    # Automated aggregates
    auto_vals = [r["automated"] for r in results if "automated" in r]
    auto_agg = {}
    if auto_vals:
        auto_agg = {
            "avg_vocab_hit_rate": round(sum(a["vocab_hit_rate"] for a in auto_vals) / len(auto_vals), 3),
            "avg_topic_coverage": round(sum(a["expected_topic_coverage"] for a in auto_vals) / len(auto_vals), 3),
            "avg_source_diversity": round(sum(a["source_diversity"] for a in auto_vals) / len(auto_vals), 3),
            "avg_top_relevance": round(sum(a["top_relevance"] for a in auto_vals) / len(auto_vals), 3),
            "pct_length_ok": round(100 * sum(1 for a in auto_vals if a["length_ok"]) / len(auto_vals), 1),
            "pct_sources_ok": round(100 * sum(1 for a in auto_vals if a["sources_ok"]) / len(auto_vals), 1),
            "avg_citation_tags": round(sum(a["citation_tags_total"] for a in auto_vals) / len(auto_vals), 2),
        }

    # Worst 5 by judge final_score
    worst5 = sorted(judged, key=lambda r: r["judge"]["final_score"])[:5]
    worst5_brief = [
        {
            "id": r["id"],
            "category": r["category"],
            "question": r["question"],
            "final_score": r["judge"]["final_score"],
            "expected_behavior": r.get("expected_behavior", "answer"),
            "reasoning": r["judge"].get("reasoning", "")[:200],
        }
        for r in worst5
    ]

    # Abstention metrics (automated keyword heuristic — independent of judge)
    abstention_metrics = compute_abstention_metrics(results)

    return {
        "baseline": baseline,
        "overall": round(overall, 2),
        "delta": round(overall - baseline, 2),
        "questions_scored": len(judged),
        "questions_total": len(results),
        "per_category": per_category,
        "per_dimension": per_dimension,
        "automated_aggregates": auto_agg,
        "abstention_metrics": abstention_metrics,
        "worst5": worst5_brief,
    }


def print_summary(report: Dict[str, Any]) -> None:
    print()
    print("=" * 70)
    print("                RAG QUALITY EVALUATION — SUMMARY")
    print("=" * 70)
    overall = report.get("overall", 0.0)
    baseline = report.get("baseline", 0.0)
    delta = report.get("delta", 0.0)
    scored = report.get("questions_scored", 0)
    total = report.get("questions_total", 0)

    banner = "PASS" if delta >= 0 else "REGRESSION"
    print(f"\n  FINAL SCORE : {overall:.2f} / 10  ({scored}/{total} questions judged)")
    print(f"  BASELINE    : {baseline:.2f} / 10  (commit ed84cf1)")
    print(f"  DELTA       : {delta:+.2f}   ->   {banner}")

    if report.get("per_category"):
        print("\n  Per category:")
        for cat, score in sorted(report["per_category"].items()):
            print(f"    {cat:.<32} {score:.2f}")

    if report.get("per_dimension"):
        print("\n  Per rubric dimension:")
        for dim, score in report["per_dimension"].items():
            print(f"    {dim:.<32} {score:.2f}")

    if report.get("automated_aggregates"):
        print("\n  Automated metrics:")
        for k, v in report["automated_aggregates"].items():
            print(f"    {k:.<32} {v}")

    am = report.get("abstention_metrics")
    if am:
        print("\n  Adversarial defence metrics (automated keyword heuristic):")
        print(f"    Abstain pool:  {am['abstain_correct']}/{am['abstain_total']}"
              f"  ({am['abstain_rate']:.0%} correctly abstained)")
        print(f"    Reject pool:   {am['reject_correct']}/{am['reject_total']}"
              f"  ({am['reject_rate']:.0%} correctly rejected premise)")
        print(f"    False refusal: {am['false_refusal_count']}/{am['false_refusal_total']}"
              f"  ({am['false_refusal_rate']:.0%} of answerable questions wrongly refused)")
        print(f"    Overall adversarial defence rate: "
              f"{am['adversarial_correct']}/{am['adversarial_total']}"
              f"  ({am['adversarial_defense_rate']:.0%})")

    if report.get("worst5"):
        print("\n  Worst 5 answers (for manual review):")
        for r in report["worst5"]:
            eb = r.get("expected_behavior", "answer")
            tag = f" [{eb}]" if eb != "answer" else ""
            print(f"    [{r['id']}] {r['final_score']:.2f}  {r['question'][:55]}{tag}")

    print("\n" + "=" * 70 + "\n")


# --- Main ------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG quality evaluation harness")
    parser.add_argument("--subset", type=str, default=None,
                        help="Comma-separated category codes, e.g. 'A,B,E'")
    parser.add_argument("--adversarial-only", action="store_true",
                        help="Run only adversarial questions (expected_behavior != 'answer')")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip LLM-as-judge, automated metrics only")
    parser.add_argument("--baseline", type=float, default=None,
                        help="Override baseline score (default from questions.json)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap at N questions (debug)")
    args = parser.parse_args()

    subset_codes = [s.strip() for s in args.subset.split(",")] if args.subset else None
    questions, baseline = load_questions(
        subset_codes=subset_codes,
        adversarial_only=args.adversarial_only,
    )
    if args.baseline is not None:
        baseline = args.baseline
    if args.limit:
        questions = questions[: args.limit]

    print(f"\n  Loaded {len(questions)} questions. Baseline = {baseline}")
    print(f"  LLM judge: {'OFF' if args.no_judge else 'ON (' + JUDGE_MODEL + ')'}")
    print(f"  Initializing RayPeatRAG...\n")

    rag = RayPeatRAG()

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not args.no_judge and not api_key:
        print("  WARNING: GEMINI_API_KEY not set — forcing --no-judge")
        args.no_judge = True

    results: List[Dict[str, Any]] = []
    for i, q in enumerate(questions, 1):
        qid = q["id"]
        category = q["category"]
        question_text = q["question"]
        print(f"  [{i:2}/{len(questions)}] {qid}  {category:<22} {question_text[:52]}", flush=True)

        # --- RAG call ---
        # Use the question's difficulty label to set max_sources explicitly.
        # This isolates the "more-sources-on-hard-questions" lever from the
        # runtime heuristic in RayPeatRAG._estimate_max_sources.
        difficulty = q.get("difficulty", "medium")
        max_sources_for_q = DIFFICULTY_TO_MAX_SOURCES.get(difficulty, 8)
        t0 = time.time()
        try:
            raw_answer = rag.get_rag_response(question_text, max_sources=max_sources_for_q)
        except Exception as e:
            print(f"        RAG error: {e}")
            results.append({
                "id": qid, "category": category, "question": question_text,
                "error": str(e), "judge": {"final_score": 0.0},
            })
            continue
        rag_ms = int((time.time() - t0) * 1000)

        answer_body, sources = parse_sources_footer(raw_answer)
        automated = compute_automated_metrics(answer_body, sources, q)

        expected_behavior = q.get("expected_behavior", "answer")
        adversarial_note = q.get("adversarial_note", "")

        judge_result: Dict[str, Any] = {}
        if not args.no_judge:
            t1 = time.time()
            judge_result = llm_judge(
                question_text, answer_body, sources, api_key,
                expected_behavior=expected_behavior,
                adversarial_note=adversarial_note,
            )
            judge_ms = int((time.time() - t1) * 1000)
            score_val = judge_result.get("final_score", 0.0)
            err = judge_result.get("error", "")
            err_tail = f"  ERR={err}" if err else ""
            print(f"        rag={rag_ms}ms judge={judge_ms}ms  score={score_val:.2f}  "
                  f"sources={len(sources)} vocab={automated['vocab_hit_rate']:.2f}{err_tail}",
                  flush=True)
            # Light pacing. On paid-tier keys, 1s is plenty; on free-tier,
            # the llm_judge retry-with-backoff loop handles 429s.
            time.sleep(1.0)
        else:
            print(f"        rag={rag_ms}ms  sources={len(sources)} "
                  f"vocab={automated['vocab_hit_rate']:.2f} "
                  f"topics={automated['expected_topic_coverage']:.2f}",
                  flush=True)

        results.append({
            "id": qid,
            "category": category,
            "difficulty": q.get("difficulty"),
            "expected_behavior": expected_behavior,
            "max_sources_used": max_sources_for_q,
            "question": question_text,
            "answer": answer_body,
            "sources": sources,
            "automated": automated,
            "judge": judge_result,
            "rag_ms": rag_ms,
            "abstention_signal": detect_abstention_signal(answer_body),
        })

    report = aggregate(results, baseline)
    print_summary(report)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"results_{timestamp}.json"
    payload = {
        "timestamp": timestamp,
        "baseline": baseline,
        "subset": subset_codes,
        "judged": not args.no_judge,
        "report": report,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Results saved to: {out_path}\n")

    return 0 if report.get("delta", -1) >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
