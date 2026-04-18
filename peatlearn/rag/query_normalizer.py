"""Query vocabulary normalization for Ray Peat RAG.

Maps colloquial health terms to Ray Peat's actual corpus vocabulary so that
embedding, HyDE, and cross-encoder reranking all operate on terms that exist
in the corpus. Without this, a query like "carbs" embeds far from Peat's
"carbohydrates / glucose / sucrose / fructose" and retrieval fails silently.

Usage:
    from peatlearn.rag.query_normalizer import normalize_query
    search_query = normalize_query(user_query)
"""

import logging
import re

_log = logging.getLogger(__name__)

# Colloquial term/phrase → Peat corpus vocabulary.
# Multi-word phrases are checked first (longest match wins).
# Expansion terms are APPENDED to the query, not substituted, so the
# original intent is preserved while the embedding gets richer signal.
_EXPANSIONS: dict[str, str] = {
    # --- nutrition / macros ---
    "carbs":                "carbohydrates glucose sucrose fructose sugar glycogen",
    "carb":                 "carbohydrates glucose sucrose fructose sugar glycogen",
    "protein powder":       "protein amino acids gelatin collagen whey",
    "fiber":                "fiber cellulose intestinal bacteria digestion",
    # --- fats ---
    "seed oils":            "polyunsaturated fatty acids PUFA linoleic acid unsaturated",
    "vegetable oils":       "polyunsaturated fatty acids PUFA linoleic acid unsaturated",
    "fish oil":             "fish oil DHA EPA omega-3 polyunsaturated",
    "omega 3":              "omega-3 DHA EPA fish oil polyunsaturated fatty acids",
    "trans fats":           "trans fat hydrogenated margarine PUFA",
    # --- metabolism ---
    "metabolic health":     "oxidative metabolism thyroid metabolic rate cellular respiration",
    "metabolism":           "oxidative metabolism thyroid metabolic rate cellular respiration ATP",
    "fat burning":          "fatty acid oxidation lipolysis metabolism thyroid",
    "weight loss":          "weight metabolism thyroid metabolic rate obesity",
    "low energy":           "energy metabolism thyroid T3 mitochondria ATP cellular respiration",
    "slow metabolism":      "hypothyroidism thyroid T3 T4 metabolic rate",
    # --- gut ---
    "gut health":           "intestinal bacteria endotoxin gut flora digestion serotonin",
    "bloating":             "bloating intestinal bacteria endotoxin digestion",
    "leaky gut":            "intestinal permeability endotoxin bacteria gut",
    "probiotics":           "bacteria intestinal flora carrot fiber antibiotic",
    # --- hormones ---
    "estrogen":             "estrogen aromatase progesterone estradiol",
    "hormones":             "hormones estrogen progesterone thyroid cortisol pregnenolone",
    "birth control":        "oral contraceptives estrogen progesterone",
    "testosterone":         "testosterone DHT androgen progesterone",
    # --- mental health ---
    "depression":           "depression serotonin learned helplessness PUFA stress",
    "anxiety":              "anxiety serotonin adrenaline cortisol GABA magnesium",
    "brain fog":            "brain serotonin hypothyroidism PUFA glucose",
    "brain health":         "brain serotonin PUFA cholesterol progesterone",
    # --- stress ---
    "stress":               "stress cortisol adrenaline serotonin endorphin",
    "stress response":      "cortisol adrenaline serotonin aldosterone stress",
    "adrenal fatigue":      "adrenal cortisol adrenaline stress exhaustion",
    # --- lifestyle ---
    "sleep":                "sleep melatonin serotonin cortisol tryptophan",
    "red light":            "red light therapy photobiomodulation infrared mitochondria",
    "sunlight":             "light red infrared vitamin D",
    "exercise":             "exercise lactic acid stress cortisol metabolism",
    "keto":                 "ketogenic low-carb fatty acid oxidation glucose starvation",
    "ketogenic":            "ketogenic low-carb fatty acid oxidation glucose starvation",
    "intermittent fasting": "fasting stress cortisol adrenaline metabolism",
    "fasting":              "fasting stress cortisol adrenaline metabolism",
    # --- appearance / body ---
    "skin health":          "skin estrogen PUFA collagen",
    "hair loss":            "hair loss thyroid DHT estrogen stress cortisol",
    "acne":                 "acne estrogen PUFA skin hormones",
    "aging":                "aging oxidative metabolism PUFA cellular respiration progesterone",
    "anti aging":           "aging oxidative metabolism PUFA progesterone pregnenolone",
    # --- specific conditions ---
    "inflammation":         "prostaglandins eicosanoids arachidonic acid PUFA inflammation",
    "cancer":               "cancer estrogen PUFA oxidative metabolism respiration",
    "diabetes":             "diabetes glucose insulin metabolism thyroid",
    "heart health":         "heart cardiovascular PUFA calcium magnesium",
    "cholesterol":          "cholesterol thyroid steroids pregnenolone progesterone",
    "bone health":          "bone calcium vitamin D phosphate parathyroid",
    "fertility":            "fertility progesterone estrogen thyroid pregnenolone",
    "muscle":               "muscle protein metabolism thyroid anabolic",
    # --- supplements (common lay names) ---
    "vitamin d":            "vitamin D calcium cholecalciferol",
    "magnesium":            "magnesium calcium mineral stress",
    "aspirin":              "aspirin anti-inflammatory prostaglandins",
    "coconut oil":          "coconut oil saturated fat thyroid",
}

# Sort phrases by length (longest first) so multi-word matches take priority.
_PHRASES_SORTED = sorted(_EXPANSIONS.keys(), key=len, reverse=True)


def normalize_query(query: str) -> str:
    """Expand colloquial terms in *query* to Ray Peat's corpus vocabulary.

    Returns the original query with matched expansion terms appended in
    parentheses.  If nothing matches, returns the query unchanged.

    Examples
    --------
    >>> normalize_query("explain how carbs affect metabolic health")
    'explain how carbs affect metabolic health (carbohydrates glucose sucrose fructose sugar glycogen oxidative metabolism thyroid metabolic rate cellular respiration)'
    """
    q_lower = query.lower()
    matched_terms: list[str] = []
    matched_phrases: set[str] = set()  # track which phrases already matched

    for phrase in _PHRASES_SORTED:
        # Word-boundary match so "carb" doesn't match inside "carbohydrates"
        pattern = r'\b' + re.escape(phrase) + r'\b'
        if re.search(pattern, q_lower):
            # Skip if a longer phrase already covered this one
            # (e.g. "seed oils" matched → skip "oils" if it were in the dict)
            if any(phrase in longer and longer != phrase for longer in matched_phrases):
                continue
            matched_phrases.add(phrase)
            matched_terms.append(_EXPANSIONS[phrase])

    if not matched_terms:
        return query

    # Deduplicate expansion tokens across all matched phrases
    seen: set[str] = set()
    deduped: list[str] = []
    for block in matched_terms:
        for token in block.split():
            tok_lower = token.lower()
            if tok_lower not in seen:
                seen.add(tok_lower)
                deduped.append(token)

    expanded = f"{query} ({' '.join(deduped)})"
    _log.info(f"Query expanded: \"{query}\" → +{len(deduped)} terms")
    return expanded
