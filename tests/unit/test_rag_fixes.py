"""
Unit tests for the 2026-09-19 fixes.

  - verifier: sees the same per-source window as the generator (was 500 chars
    of Peat's words vs the generator's 1200, so true claims got stripped)
  - rag_system: reranks against the resolved follow-up query, not the raw text
  - eval harness: parses negative relevance scores, recognises the pipeline's
    own refusal / premise-rejection / error replies

All tests run fully offline — network calls are monkeypatched.
"""

import importlib.util
from pathlib import Path

import pytest

from peatlearn.rag import verifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_eval_module():
    spec = importlib.util.spec_from_file_location(
        "eval_rag_quality", PROJECT_ROOT / "scripts" / "eval" / "eval_rag_quality.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Verifier window
# --------------------------------------------------------------------------- #

class _FakeResp:
    status_code = 200

    def json(self):
        return {"candidates": [{"content": {"parts": [{"text": '{"claims": []}'}]}}]}


def test_verifier_sees_text_past_500_chars(monkeypatch):
    """A fact at char ~900 of a source must reach the verifier prompt."""
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["prompt"] = json["contents"][0]["parts"][0]["text"]
        return _FakeResp()

    monkeypatch.setattr(verifier.requests, "post", fake_post)
    text = "x" * 900 + " MARKER_LATE_FACT " + "y" * 100
    verifier.verify_claims(
        "Peat said something [S1].",
        [{"source_file": "a.txt", "context": "ctx", "ray_peat_response": text}],
        api_key="fake-key",
    )
    assert "MARKER_LATE_FACT" in captured["prompt"]


def test_verifier_window_matches_generator():
    """Guard against the two windows drifting apart again."""
    assert verifier.SOURCE_TEXT_CHARS >= 1200
    assert verifier.SOURCE_CONTEXT_CHARS >= 400


# --------------------------------------------------------------------------- #
# Reranker receives the resolved follow-up query
# --------------------------------------------------------------------------- #

class _FakeIndex:
    def query(self, vector=None, top_k=None, include_metadata=None, filter=None):
        return {"matches": [{
            "id": "c1", "score": 0.8,
            "metadata": {"source_file": "f.txt", "context": "c", "ray_peat_response": "r"},
        }]}


class _FakeSearch:
    index = _FakeIndex()

    def embed_query(self, q):
        return [0.1] * 8


def test_rerank_uses_resolved_query(monkeypatch):
    from peatlearn.adaptive.rag_system import RayPeatRAG
    from peatlearn.rag import (
        query_contextualizer, domain_guard, temporal_guard, reranker, confidence,
    )

    seen = {}
    monkeypatch.setattr(query_contextualizer, "contextualize",
                        lambda q, h, k: "how much progesterone should I take")
    monkeypatch.setattr(domain_guard, "check_domain", lambda q, api_key=None: None)
    monkeypatch.setattr(temporal_guard, "check_temporal", lambda q: None)

    def fake_rerank(query, candidates):
        seen["query"] = query
        return [dict(c, rerank_score=-9.0) for c in candidates]

    monkeypatch.setattr(reranker, "rerank", fake_rerank)

    class _Abstain:
        tier = "ABSTAIN"
        reasons = ["test"]

    # Force ABSTAIN so the test stops before any LLM call.
    monkeypatch.setattr(confidence, "score_retrieval", lambda *a, **k: _Abstain())

    rag = RayPeatRAG(search_engine=_FakeSearch())
    rag.api_key = "fake-key"
    rag._get_rag_response_sync(
        "how much?", chat_history=[{"role": "user", "content": "progesterone?"}]
    )
    assert seen["query"] == "how much progesterone should I take"


# --------------------------------------------------------------------------- #
# Eval harness parsing
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def ev():
    return _load_eval_module()


def test_sources_footer_accepts_negative_relevance(ev):
    raw = (
        "Answer text.\n\n📚 Sources:\n"
        "1. a.txt (relevance: 2.31)\n"
        "2. b.txt (relevance: -0.33)\n"
        "3. c.txt (relevance: -1.39)\n"
    )
    _, sources = ev.parse_sources_footer(raw)
    assert [s["relevance"] for s in sources] == [2.31, -0.33, -1.39]


@pytest.mark.parametrize("answer", [
    "I couldn't find relevant information on that topic in Ray Peat's work.",
    "I'm focused on Dr. Ray Peat's work ... which falls outside that domain.",
    "it references a topic that emerged after Ray Peat's death in October 2022.",
    "so I can't reliably identify which specific studies, papers, or experiments",
    "I don't have sufficient information in Ray Peat's corpus to answer this",
])
def test_pipeline_refusals_detected(ev, answer):
    assert ev.detect_abstention_signal(answer) == "abstained"


@pytest.mark.parametrize("answer", [
    "That premise is incorrect—Ray Peat consistently argued the opposite [S4].",
    "Ray Peat does not recommend a strict ketogenic diet [S1].",
    "In fact, he explicitly warned against it [S1, S8].",
    "he did not recommend them for cooking [S2].",
])
def test_premise_rejections_detected(ev, answer):
    assert ev.detect_abstention_signal(answer) == "premise_rejected"


def test_error_reply_is_not_a_refusal(ev):
    answer = 'Sorry, I ran into a technical issue processing your question: "x"'
    assert ev.detect_abstention_signal(answer) == "error"


def test_false_refusal_counted_on_answer_pool(ev):
    results = [
        {"id": "F2", "expected_behavior": "answer",
         "answer": "I couldn't find relevant information on that topic in Ray Peat's work."},
        {"id": "A1", "expected_behavior": "answer", "answer": "Peat argued thyroid [S1]."},
    ]
    m = ev.compute_abstention_metrics(results)
    assert m["false_refusal_count"] == 1
    assert m["error_count"] == 0
