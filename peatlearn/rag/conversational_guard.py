"""
Conversational guard for greetings, small-talk, and meta questions.

The RAG pipeline assumes every input is a substantive question about Ray Peat's
work. A bare "hi", "thanks", or "who are you?" has no answer in the corpus, but
retrieval will still return the nearest passages, the reranker will score one
high enough for a HIGH tier, and the LLM will confabulate a Peat-persona reply
with fabricated [S#] citations. For a health-critical, zero-hallucination app
that is a serious failure mode.

This module intercepts those inputs *before* retrieval runs and returns a plain,
honest reply with no sources and no confidence footer. It is intentionally
high-precision: it only fires on inputs that are *entirely* conversational
filler, so a real question that happens to open with "hi" (e.g. "hi, what does
Peat say about thyroid?") still flows through to the full pipeline.

Matching is exact-phrase on a normalized form (lowercased, punctuation and
surrounding whitespace stripped). Extend the phrase sets as new patterns show up
in real usage — do NOT loosen to substring matching, which would swallow real
questions.
"""

import re
from typing import Optional

# What the assistant is — reused across replies so the framing stays consistent.
_WHAT_I_AM = (
    "I'm a question-answering assistant grounded in Dr. Ray Peat's work — his "
    "interviews, articles, newsletters, and email correspondence on bioenergetic "
    "health, nutrition, hormones, and metabolism. I only answer from what Peat "
    "actually said, and I'll tell you when a topic isn't covered rather than guess."
)

_GREETING_REPLY = (
    "Hello! " + _WHAT_I_AM + "\n\n"
    "Ask me something specific — for example, \"What did Peat think about "
    "polyunsaturated fats?\", \"How does thyroid affect metabolism?\", or "
    "\"What foods did Peat recommend for energy?\""
)

_THANKS_REPLY = (
    "You're welcome! Ask me anything else about Ray Peat's views on health, "
    "nutrition, hormones, or metabolism and I'll ground the answer in his work."
)

_FAREWELL_REPLY = (
    "Take care! Come back anytime you have a question about Ray Peat's work."
)

_CAPABILITY_REPLY = (
    _WHAT_I_AM + "\n\n"
    "Good things to ask about: thyroid and metabolism, progesterone and estrogen, "
    "polyunsaturated vs. saturated fats, sugar and energy, calcium and PTH, stress "
    "hormones, and the specific foods Peat recommended. Ask a concrete question and "
    "I'll answer from his actual words."
)

# Exact-phrase sets (normalized). Each maps to one of the replies above.
_GREETINGS = frozenset({
    "hi", "hii", "hiii", "hello", "helo", "hey", "heya", "hiya", "yo",
    "hi there", "hii there", "hey there", "hello there", "well hello",
    "greetings", "sup", "wassup", "whats up", "what's up", "howdy",
    "good morning", "good afternoon", "good evening", "good day",
    "morning", "gm", "hello again", "hi again", "hey again", "hi bot",
    "hello bot", "hey bot", "knock knock", "anyone there", "you there",
    "are you there",
})

_THANKS = frozenset({
    "thanks", "thank you", "thank u", "thx", "thnx", "ty", "tysm",
    "thank you so much", "thanks a lot", "thanks so much", "many thanks",
    "appreciate it", "i appreciate it", "much appreciated", "cheers",
    "thank you very much", "thanks very much", "great thanks", "ok thanks",
    "okay thanks", "perfect thanks", "thank you for the help",
})

_FAREWELLS = frozenset({
    "bye", "byebye", "bye bye", "goodbye", "good bye", "see you",
    "see ya", "see you later", "cya", "later", "take care", "good night",
    "goodnight", "gn", "im done", "i'm done", "that's all", "thats all",
    "thats it", "that's it", "nothing else",
})

_CAPABILITY = frozenset({
    "who are you", "what are you", "what is this", "whats this",
    "what's this", "what is peatlearn", "what can you do", "what do you do",
    "what can i ask", "what can i ask you", "what should i ask",
    "how does this work", "how do you work", "what do you know",
    "help", "help me", "what is this app", "what is this site",
    "what is this bot", "introduce yourself", "tell me about yourself",
    "what are you for", "what's your purpose", "whats your purpose",
})


def _normalize(query: str) -> str:
    """Lowercase, strip surrounding punctuation/whitespace, collapse inner spaces."""
    q = (query or "").lower().strip()
    # Drop surrounding punctuation (!, ?, ., …) but keep apostrophes inside words.
    q = re.sub(r"^[\s\W_]+|[\s\W_]+$", "", q, flags=re.UNICODE)
    q = re.sub(r"\s+", " ", q)
    return q


def check_conversational(query: str) -> Optional[str]:
    """Return a canned reply if the query is pure greeting/small-talk/meta.

    Args:
        query: The raw user query.

    Returns:
        A plain reply string (no sources, no confidence footer) when the input is
        entirely conversational filler; otherwise None, meaning the query should
        proceed through the normal RAG pipeline.
    """
    norm = _normalize(query)
    if not norm:
        # Empty / punctuation-only input — treat as a greeting prompt.
        return _GREETING_REPLY
    if norm in _GREETINGS:
        return _GREETING_REPLY
    if norm in _THANKS:
        return _THANKS_REPLY
    if norm in _FAREWELLS:
        return _FAREWELL_REPLY
    if norm in _CAPABILITY:
        return _CAPABILITY_REPLY
    return None
