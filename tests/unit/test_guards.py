"""
Unit tests for the pre-retrieval guards.

  - conversational_guard: greetings / thanks / farewells / meta questions
  - domain_guard: out-of-domain (non-health) questions

These freeze the behaviour verified by hand when the guards were added: a bare
"hi" or an off-topic question must never reach retrieval and come back as a
fabricated Peat-persona reply.

All tests run fully offline. The domain guard's LLM fallback tier is
monkeypatched so the lexical fast-paths and wiring can be checked with no API
key and no network.
"""

import pytest

from peatlearn.rag.conversational_guard import check_conversational
from peatlearn.rag import domain_guard
from peatlearn.rag.domain_guard import check_domain


# --------------------------------------------------------------------------- #
# Conversational guard
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("query", [
    "hi", "Hi!", "hello", "hello there", "HEY", "hey there", "yo",
    "good morning", "sup", "howdy",
    "thanks", "thank you", "thank you so much", "ty", "cheers",
    "bye", "goodbye", "see you later", "take care",
    "who are you?", "what are you", "what can you do", "what is this",
    "help", "introduce yourself",
    "   ", "!!!",            # empty / punctuation-only -> greeting prompt
])
def test_conversational_intercepted(query):
    """Pure greetings / small-talk / meta must get a canned reply (not None)."""
    assert check_conversational(query) is not None


@pytest.mark.parametrize("query", [
    "hi, what does Peat say about thyroid?",
    "help me understand estrogen",
    "thanks to estrogen what happens",
    "what about polyunsaturated fats",
    "is sugar good for energy",
    "who was the researcher Peat cited",
])
def test_conversational_passes_through(query):
    """Real questions that merely open with a greeting word must pass through."""
    assert check_conversational(query) is None


def test_conversational_reply_has_no_citations_or_footer():
    """Greeting replies must not carry [S#] citations or a confidence footer."""
    reply = check_conversational("hi")
    assert "[S" not in reply
    assert "Confidence:" not in reply


# --------------------------------------------------------------------------- #
# Domain guard — lexical fast-paths (no API call)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("query", [
    "what does Peat say about thyroid?",
    "is coffee bad for me",
    "how does estrogen affect metabolism",
    "what foods give energy",
    "progesterone and stress",
    "is the keto diet good",
    "is light therapy from a therapist helpful",   # must not trip on 'api'/'therapy'
])
def test_domain_in_domain_allowed(query):
    """Health/Peat vocab -> allowed (None) via the in-domain fast-path, no API."""
    assert check_domain(query, api_key="") is None


@pytest.mark.parametrize("query", [
    "write me python code to sort a list",
    "what is the best stock to buy",
    "who won the world cup",
    "what is the capital of France",
    "solve for x in this equation",
    "recommend a netflix movie",
    "how do I invest in bitcoin",
])
def test_domain_out_of_domain_flagged(query):
    """Clear non-health vocab -> flagged (reason string) via fast-path, no API."""
    assert check_domain(query, api_key="") is not None


def test_domain_in_domain_wins_over_out_of_domain():
    """A query mentioning both health and non-health vocab is allowed."""
    assert check_domain("is the keto diet bad for my python code", api_key="") is None


def test_domain_fails_open_without_key():
    """Ambiguous query + no API key -> fail open (None), downstream gate applies."""
    # 'always cold' hits neither lexical list; with no key the LLM tier is skipped.
    assert check_domain("why am I always cold", api_key="") is None


# --------------------------------------------------------------------------- #
# Domain guard — LLM fallback tier (monkeypatched, offline)
# --------------------------------------------------------------------------- #

def test_domain_llm_fallback_flags_out(monkeypatch):
    """Ambiguous query the classifier judges OUT -> flagged."""
    monkeypatch.setattr(domain_guard, "_classify_with_llm", lambda q, k: False)
    reason = check_domain("how do I change a tire", api_key="fake-key")
    assert reason is not None


def test_domain_llm_fallback_allows_in(monkeypatch):
    """Ambiguous query the classifier judges IN -> allowed."""
    monkeypatch.setattr(domain_guard, "_classify_with_llm", lambda q, k: True)
    assert check_domain("I feel exhausted every afternoon", api_key="fake-key") is None


def test_domain_llm_undetermined_allows(monkeypatch):
    """Classifier returns None (undetermined) -> fail open, allowed."""
    monkeypatch.setattr(domain_guard, "_classify_with_llm", lambda q, k: None)
    assert check_domain("some genuinely ambiguous phrase", api_key="fake-key") is None
