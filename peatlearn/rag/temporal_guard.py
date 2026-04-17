"""
Temporal guard for post-2022 queries.

Ray Peat died in October 2022. Any query about topics that emerged or became
mainstream after his death cannot be grounded in his corpus — regardless of
what the retrieval system seems to find. This module flags such queries for
automatic ABSTAIN routing.

The wordlist is intentionally narrow (high-precision, not high-recall). It
targets specific product names, drug names, and protocol names that are
unambiguously post-Peat. General terms like "vaccine" or "exercise" are
NOT included because Peat did discuss those topics generally.

Maintain and extend the wordlist as new post-2022 health trends emerge.
"""

import logging
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Post-2022 terms — specific enough to avoid false positives on general topics.
# Each entry is lowercased for case-insensitive matching.
# Format: (pattern, reason)
_POST_2022_TERMS: list[Tuple[str, str]] = [
    # GLP-1 agonists (mainstream post-2022)
    ("ozempic", "Ozempic (semaglutide) became mainstream for weight loss post-2022"),
    ("semaglutide", "Semaglutide became mainstream post-2022"),
    ("wegovy", "Wegovy (semaglutide brand) launched post-2021"),
    ("mounjaro", "Mounjaro (tirzepatide) approved 2022"),
    ("tirzepatide", "Tirzepatide approved 2022"),
    ("glp-1 agonist", "GLP-1 agonists became mainstream metabolic intervention post-2022"),
    ("glp1 agonist", "GLP-1 agonists became mainstream metabolic intervention post-2022"),

    # NAD+ supplement brands (Sinclair-era, 2019+, but Peat never addressed)
    ("nicotinamide riboside", "NR supplementation trend post-dates Peat's active work"),
    ("nicotinamide mononucleotide", "NMN supplementation trend post-dates Peat's active work"),
    (" nmn ", "NMN supplementation trend post-dates Peat's active work"),
    (" nr supplement", "NR supplementation trend post-dates Peat's active work"),

    # Zone 2 as a named protocol
    ("zone 2 cardio", "Zone 2 cardio as a named protocol popularized 2022+"),
    ("zone 2 training", "Zone 2 training protocol popularized 2022+"),
    ("zone two cardio", "Zone 2 cardio protocol popularized 2022+"),

    # mRNA vaccine specifics (Peat may have had private views but no published position)
    ("mrna vaccine", "mRNA vaccines — no published Peat position"),
    ("mrna covid", "mRNA COVID vaccines — no published Peat position"),
    ("pfizer vaccine", "Specific vaccine brands — no published Peat position"),
    ("moderna vaccine", "Specific vaccine brands — no published Peat position"),

    # CGM biohacking
    ("continuous glucose monitor", "CGM biohacking trend post-dates Peat's engagement"),
    (" cgm ", "CGM devices — Peat never addressed these specifically"),
    ("levels health", "Levels Health CGM company — post-2020"),
    ("nutrisense", "Nutrisense CGM company — post-2020"),

    # Post-2022 supplement/drug trends
    ("rapamycin", "Rapamycin for longevity — post-dates Peat's engagement"),
    ("bryan johnson", "Bryan Johnson's 'Blueprint' protocol is post-2022"),
    ("blueprint protocol", "Bryan Johnson's 'Blueprint' protocol is post-2022"),
]


def check_temporal(query: str) -> Optional[str]:
    """Check if a query references post-2022 terminology.

    Args:
        query: The user's question text.

    Returns:
        A reason string if the query is flagged (should route to ABSTAIN),
        or None if the query is fine.
    """
    q_lower = f" {query.lower()} "  # pad with spaces for word-boundary matching
    for term, reason in _POST_2022_TERMS:
        if term in q_lower:
            logger.info(f"Temporal guard triggered: '{term}' in query — {reason}")
            return reason
    return None
