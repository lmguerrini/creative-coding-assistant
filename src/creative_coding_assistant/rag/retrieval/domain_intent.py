"""Lightweight query-domain intent detection for retrieval postprocessing."""

from __future__ import annotations

import re
from dataclasses import dataclass

from creative_coding_assistant.contracts import CreativeCodingDomain

_WHITESPACE_PATTERN = re.compile(r"\s+")
_THREE_JS_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bthree(?:\.js|js|\s+js)\b"), 3),
)
_REACT_THREE_FIBER_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\breact\s+three\s+fiber\b"), 3),
    (re.compile(r"@react-three/fiber"), 3),
    (re.compile(r"\br3f\b"), 3),
    (re.compile(r"\buseframe\b"), 2),
)
_P5_JS_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bp5(?:\.js|js)?\b"), 3),
)
_GLSL_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bglsl\b"), 3),
    (re.compile(r"\bfragment\s+shader\b"), 2),
    (re.compile(r"\bvertex\s+shader\b"), 2),
    (re.compile(r"\bshader\b"), 1),
)
_INTENT_PATTERNS: tuple[
    tuple[CreativeCodingDomain, tuple[tuple[re.Pattern[str], int], ...]],
    ...,
] = (
    (CreativeCodingDomain.THREE_JS, _THREE_JS_PATTERNS),
    (CreativeCodingDomain.REACT_THREE_FIBER, _REACT_THREE_FIBER_PATTERNS),
    (CreativeCodingDomain.P5_JS, _P5_JS_PATTERNS),
    (CreativeCodingDomain.GLSL, _GLSL_PATTERNS),
)
_RELATED_DOMAIN_FALLBACKS: dict[
    CreativeCodingDomain,
    tuple[CreativeCodingDomain, ...],
] = {
    CreativeCodingDomain.THREE_JS: (CreativeCodingDomain.REACT_THREE_FIBER,),
    CreativeCodingDomain.REACT_THREE_FIBER: (CreativeCodingDomain.THREE_JS,),
}


@dataclass(frozen=True)
class DomainIntent:
    primary_domain: CreativeCodingDomain
    allowed_domains: tuple[CreativeCodingDomain, ...]


def detect_domain_intent(query: str) -> DomainIntent | None:
    """Return a conservative dominant-domain intent when one clearly exists."""

    normalized_query = _normalize_query(query)
    if not normalized_query:
        return None

    scores = {
        domain: _score_domain(normalized_query, patterns)
        for domain, patterns in _INTENT_PATTERNS
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    primary_domain, primary_score = ranked[0]
    secondary_score = ranked[1][1]

    if primary_score <= 0 or primary_score <= secondary_score:
        return None

    related_domains = _RELATED_DOMAIN_FALLBACKS.get(primary_domain, ())
    return DomainIntent(
        primary_domain=primary_domain,
        allowed_domains=(primary_domain, *related_domains),
    )


def _normalize_query(value: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", value.strip().lower())


def _score_domain(
    query: str,
    patterns: tuple[tuple[re.Pattern[str], int], ...],
) -> int:
    return sum(weight for pattern, weight in patterns if pattern.search(query))
