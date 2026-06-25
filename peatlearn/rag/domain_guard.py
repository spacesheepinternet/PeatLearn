"""
Domain guard — flag questions unrelated to Ray Peat's health/bioenergetic corpus.

The RAG pipeline assumes every input is a health/bioenergetic question. An
out-of-domain query ("best stock to buy?", "write me Python code", "who won the
world cup?") has no answer in Peat's corpus, but retrieval will still surface the
nearest passages and the LLM will confabulate a Peat-flavoured reply. For a
zero-hallucination health app, those must be refused before retrieval.

Hybrid classifier (cheap first, LLM only when needed):

    1. In-domain lexical fast-path  — query contains clear health/Peat vocab
                                       (clinical OR colloquial) -> ALLOW, no API.
    2. Out-of-domain lexical fast-path — query contains clear non-health vocab
                                       (code/finance/sports/...) AND no in-domain
                                       vocab -> FLAG, no API.
    3. Ambiguous middle             — neither list hit (e.g. colloquial health
                                       like "why am I always cold?") -> one cheap
                                       gemini-2.5-flash-lite classify call.

Matching uses word boundaries (so "api" doesn't match "therapist"). In-domain
always wins over out-of-domain, so a question that mentions both ("is the keto
diet bad for my code") is allowed. Anything explicitly about Ray Peat is treated
as in-domain regardless of subject.

Fails OPEN: if the classify call errors or no API key is set, the query is
allowed through — the downstream confidence/ABSTAIN gate is still in place, so a
guard outage degrades to current behaviour rather than refusing everything.
"""

import logging
import os
import re
from typing import Optional

import requests

logger = logging.getLogger(__name__)

CLASSIFY_MODEL = "gemini-2.5-flash-lite"
CLASSIFY_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{CLASSIFY_MODEL}:generateContent"
)

# --- In-domain vocabulary (health / nutrition / hormones / Peat) --------------
# Both clinical and colloquial terms. Any hit short-circuits to ALLOW.
_IN_DOMAIN = [
    # Peat himself — questions about his views are always in-domain
    r"peat", r"bioenergetic", r"ray\s+peat",
    # Hormones / endocrine
    r"thyroid", r"hypothyroid", r"hyperthyroid", r"hashimoto", r"t3", r"t4",
    r"tsh", r"hormone[s]?", r"endocrine", r"progesterone", r"estrogen",
    r"oestrogen", r"estrogenic", r"cortisol", r"adrenaline", r"adrenal",
    r"testosterone", r"dhea", r"pregnenolone", r"prolactin", r"serotonin",
    r"dopamine", r"melatonin", r"insulin", r"glucagon", r"pth", r"parathyroid",
    r"libido", r"fertility", r"menopause", r"pms", r"period[s]?", r"menstrual",
    # Metabolism / energy / physiology
    r"metabolism", r"metabolic", r"mitochondria", r"atp", r"oxidative",
    r"respiration", r"co2", r"carbon\s+dioxide", r"lactate", r"lactic",
    r"glucose", r"glycogen", r"blood\s+sugar", r"fatigue", r"energy",
    r"temperature", r"pulse", r"metabolic\s+rate", r"endotoxin", r"endotoxemia",
    # Nutrition / food
    r"nutrition", r"diet[s]?", r"food[s]?", r"eat", r"eating", r"sugar",
    r"fructose", r"glucose", r"carb[s]?", r"carbohydrate[s]?", r"protein",
    r"\bfat[s]?\b", r"pufa", r"polyunsaturated", r"saturated\s+fat",
    r"coconut\s+oil", r"olive\s+oil", r"seed\s+oil[s]?", r"milk", r"dairy",
    r"cheese", r"gelatin", r"collagen", r"calcium", r"magnesium", r"potassium",
    r"vitamin[s]?", r"mineral[s]?", r"\bsalt\b", r"sodium", r"caffeine",
    r"coffee", r"aspirin", r"niacinamide", r"niacin", r"\bgut\b", r"digestion",
    r"digestive", r"fiber", r"fibre", r"starch", r"fruit[s]?", r"juice",
    r"\bliver\b", r"supplement[s]?", r"\bdose\b", r"dosage", r"deficiency",
    r"calorie[s]?", r"fasting", r"keto", r"carnivore",
    # Health / body / disease
    r"health[y]?", r"disease[s]?", r"illness", r"symptom[s]?", r"inflammation",
    r"inflammatory", r"stress", r"aging", r"ageing", r"cancer", r"tumou?r",
    r"diabetes", r"obesity", r"weight", r"sleep", r"mood", r"anxiety",
    r"depress", r"headache[s]?", r"migraine", r"bloating", r"hair\s+loss",
    r"\bskin\b", r"immune", r"immunity", r"\bcell[s]?\b", r"tissue[s]?",
    r"\bblood\b", r"\bbody\b", r"biology", r"biological", r"physiology",
    r"physiological", r"medical", r"medicine", r"\bdrug[s]?\b", r"\borgan[s]?\b",
    r"nervous\s+system", r"brain", r"\bheart\b", r"kidney", r"bone[s]?",
    r"muscle[s]?", r"\bfever\b", r"infection", r"virus", r"bacteria",
    # Light / environment (Peat themes)
    r"red\s+light", r"light\s+therapy", r"\buv\b", r"sunlight", r"circadian",
]

# --- Out-of-domain vocabulary (clearly unrelated to health) -------------------
# Only flags when NO in-domain term is present.
_OUT_DOMAIN = [
    # Programming / tech
    r"python", r"javascript", r"\bjava\b", r"\bc\+\+", r"\bcode\b", r"coding",
    r"\bprogram\b", r"programming", r"\bapi\b", r"debug", r"\bregex\b",
    r"\bsql\b", r"\bhtml\b", r"\bcss\b", r"compile", r"\bgit\b", r"github",
    r"\balgorithm[s]?\b", r"software", r"\bapp\b", r"website", r"server",
    r"database", r"function\s+that", r"write\s+a\s+script", r"stack\s+trace",
    # Finance / markets
    r"stock[s]?", r"invest", r"investing", r"investment", r"crypto",
    r"bitcoin", r"ethereum", r"\bnft[s]?\b", r"portfolio", r"\bmarket\b",
    r"trading", r"mortgage", r"\btax(es)?\b", r"salary", r"\bbank\b",
    r"interest\s+rate",
    # Sports
    r"football", r"soccer", r"basketball", r"baseball", r"\bnba\b", r"\bnfl\b",
    r"world\s+cup", r"olympic", r"tournament", r"playoff", r"touchdown",
    # Entertainment / celebrity
    r"\bmovie[s]?\b", r"\bfilm[s]?\b", r"\bactor\b", r"actress", r"celebrity",
    r"\bsong[s]?\b", r"lyrics", r"netflix", r"video\s+game[s]?", r"\bxbox\b",
    r"playstation", r"\banime\b",
    # Geography / politics / general trivia
    r"capital\s+of", r"president\s+of", r"\belection[s]?\b", r"politic",
    r"\bwar\b", r"\bstartup\b", r"\bweather\b", r"\bforecast\b",
    r"translate", r"translation", r"\bspanish\b", r"\bfrench\b",
    # Math / homework
    r"\bequation\b", r"\bintegral\b", r"derivative", r"calculus", r"geometry",
    r"algebra", r"\bmatrix\b", r"\bsolve\s+for\b",
]

_IN_RE = re.compile(r"\b(?:" + "|".join(_IN_DOMAIN) + r")", re.IGNORECASE)
_OUT_RE = re.compile(r"\b(?:" + "|".join(_OUT_DOMAIN) + r")", re.IGNORECASE)

_CLASSIFY_PROMPT = """You are a domain gatekeeper for a question-answering system about Dr. Ray Peat's bioenergetic view of human health.

IN-DOMAIN (answer IN): human health, the body, nutrition, diet, foods, hormones (thyroid, estrogen, progesterone, cortisol, etc.), metabolism, energy, physiology, biology, disease, symptoms, aging, stress, sleep, mood, supplements, medicine, and plain-language "how do I feel better / why do I feel X" questions. Even colloquial health questions like "why am I always cold?" or "is coffee bad for me?" are IN. ANY question explicitly about what Ray Peat thought, said, or wrote — on any subject — is IN.

OUT-OF-DOMAIN (answer OUT): programming/code, math homework, finance/investing, sports, celebrities/entertainment, current events/politics, geography/trivia, language translation, or any request with no connection to human health or biology.

Question: {query}

Reply with exactly one word: IN or OUT."""


def _classify_with_llm(query: str, api_key: str) -> Optional[bool]:
    """Return True if in-domain, False if out-of-domain, None if undetermined."""
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": _CLASSIFY_PROMPT.format(query=query)}]}
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 8,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    try:
        resp = requests.post(CLASSIFY_URL, json=payload, headers=headers, timeout=15)
        if resp.status_code != 200:
            logger.warning(f"Domain classifier API error {resp.status_code}")
            return None
        _j = resp.json()
        text = _j["candidates"][0]["content"]["parts"][0]["text"].strip().upper()
        # Cost accounting (best-effort, never breaks the gate)
        try:
            from peatlearn.rag import cost_logger as _cl
            _cl.record_gemini("domain_classify", CLASSIFY_MODEL, _j.get("usageMetadata"))
        except Exception:
            pass
        if "OUT" in text:
            return False
        if "IN" in text:
            return True
        return None
    except Exception as e:
        logger.warning(f"Domain classifier call failed: {e}")
        return None


def check_domain(query: str, api_key: Optional[str] = None) -> Optional[str]:
    """Decide whether a query is in Ray Peat's health domain.

    Args:
        query: The raw user query.
        api_key: Gemini API key for the fallback classify call. Falls back to
            GEMINI_API_KEY env var. If absent, the guard fails open (allows).

    Returns:
        None if the query is in-domain (proceed with RAG), or a short reason
        string when the query is out-of-domain (caller should refuse).
    """
    q = (query or "").strip()
    if not q:
        return None  # empty handled upstream by the conversational guard

    # 1. In-domain vocab wins outright.
    if _IN_RE.search(q):
        return None

    # 2. Clear out-of-domain vocab, no in-domain vocab -> flag without an API call.
    if _OUT_RE.search(q):
        return "Out-of-domain: question is unrelated to Ray Peat's health/bioenergetic corpus"

    # 3. Ambiguous — ask the cheap classifier.
    api_key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return None  # fail open — downstream confidence gate still applies

    verdict = _classify_with_llm(q, api_key)
    if verdict is False:
        return "Out-of-domain: question is unrelated to Ray Peat's health/bioenergetic corpus"
    # True or None (undetermined) -> allow through
    return None
