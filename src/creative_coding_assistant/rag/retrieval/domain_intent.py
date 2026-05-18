"""Lightweight query-domain intent detection for retrieval postprocessing."""

from __future__ import annotations

import re
from collections.abc import Sequence
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
)
_PROCESSING_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bprocessing\.org\b"), 3),
    (re.compile(r"\bprocessing\s+(?:sketch|code|java|reference|api)\b"), 3),
    (re.compile(r"\bpde\s+(?:sketch|file|code)\b"), 2),
    (re.compile(r"\.pde\b"), 2),
)
_CANVAS_2D_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bcanvasrenderingcontext2d\b"), 3),
    (re.compile(r"\bcanvas\s*2d\b"), 3),
    (re.compile(r"\b2d\s+canvas\b"), 3),
    (re.compile(r"\bgetcontext\([\"']2d[\"']\)"), 2),
)
_WEBGPU_WGSL_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bwebgpu\b"), 3),
    (re.compile(r"\bwgsl\b"), 3),
    (re.compile(r"\bnavigator\.gpu\b"), 3),
    (re.compile(r"\bgpucanvascontext\b"), 2),
    (re.compile(r"\bgpudevice\b"), 2),
)
_GSAP_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bgsap\b"), 3),
    (re.compile(r"\bgreensock\b"), 3),
    (re.compile(r"\bgsap\.(?:to|from|fromto|timeline|set)\b"), 3),
)
_TONE_JS_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\btone\.js\b"), 3),
    (re.compile(r"\btonejs\b"), 3),
    (re.compile(r"\btone\.(?:synth|transport|sequence|player|start)\b"), 3),
    (re.compile(r"\bnew\s+tone\."), 3),
)
_PIXI_JS_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bpixi\.js\b"), 3),
    (re.compile(r"\bpixijs\b"), 3),
    (re.compile(r"\bpixi\.(?:application|graphics|sprite|container|assets)\b"), 3),
)
_MATTER_JS_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bmatter\.js\b"), 3),
    (re.compile(r"\bmatterjs\b"), 3),
    (
        re.compile(
            r"\bmatter\.(?:engine|world|bodies|body|runner|composite|constraint)\b"
        ),
        3,
    ),
)
_RAPIER_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"@dimforge/rapier(?:2d|3d)?"), 3),
    (re.compile(r"\brapier\.rs\b"), 3),
    (
        re.compile(
            r"\brapier\s+(?:physics|rigid\s+bodies|colliders?|world|js|"
            r"javascript|2d|3d)\b"
        ),
        3,
    ),
    (re.compile(r"\brapier(?:2d|3d)\b"), 3),
)
_HYDRA_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bhydra(?:-synth|\s+synth|\s+video\s+synth)\b"), 3),
    (re.compile(r"\bhydra\.ojack\b"), 3),
    (re.compile(r"\bhydra\s+(?:osc|src|modulate|live\s+coding|sketch)\b"), 3),
)
_SHADERTOY_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bshadertoy\b"), 3),
    (re.compile(r"\bmainimage\s*\("), 3),
    (re.compile(r"\bfragcoord\b"), 2),
    (re.compile(r"\bi(?:time|resolution|mouse|channel0)\b"), 2),
)
_INTENT_PATTERNS: tuple[
    tuple[CreativeCodingDomain, tuple[tuple[re.Pattern[str], int], ...]],
    ...,
] = (
    (CreativeCodingDomain.THREE_JS, _THREE_JS_PATTERNS),
    (CreativeCodingDomain.REACT_THREE_FIBER, _REACT_THREE_FIBER_PATTERNS),
    (CreativeCodingDomain.P5_JS, _P5_JS_PATTERNS),
    (CreativeCodingDomain.GLSL, _GLSL_PATTERNS),
    (CreativeCodingDomain.PROCESSING, _PROCESSING_PATTERNS),
    (CreativeCodingDomain.CANVAS_2D, _CANVAS_2D_PATTERNS),
    (CreativeCodingDomain.WEBGPU_WGSL, _WEBGPU_WGSL_PATTERNS),
    (CreativeCodingDomain.GSAP, _GSAP_PATTERNS),
    (CreativeCodingDomain.TONE_JS, _TONE_JS_PATTERNS),
    (CreativeCodingDomain.PIXI_JS, _PIXI_JS_PATTERNS),
    (CreativeCodingDomain.MATTER_JS, _MATTER_JS_PATTERNS),
    (CreativeCodingDomain.RAPIER, _RAPIER_PATTERNS),
    (CreativeCodingDomain.HYDRA, _HYDRA_PATTERNS),
    (CreativeCodingDomain.SHADERTOY, _SHADERTOY_PATTERNS),
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


def detect_explicit_query_domains(query: str) -> tuple[CreativeCodingDomain, ...]:
    """Return all explicitly named query domains in stable enum order."""

    normalized_query = _normalize_query(query)
    if not normalized_query:
        return ()

    scores = [
        (domain, _score_domain(normalized_query, patterns))
        for domain, patterns in _INTENT_PATTERNS
    ]
    return tuple(domain for domain, score in scores if score > 0)


def resolve_effective_query_domains(
    *,
    query: str,
    selected_domains: Sequence[CreativeCodingDomain],
) -> tuple[CreativeCodingDomain, ...]:
    """Prefer explicit query domains and otherwise preserve selected domains."""

    explicit_domains = detect_explicit_query_domains(query)
    if explicit_domains:
        return explicit_domains

    normalized: list[CreativeCodingDomain] = []
    for domain in selected_domains:
        if domain not in normalized:
            normalized.append(domain)
    return tuple(normalized)


def _normalize_query(value: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", value.strip().lower())


def _score_domain(
    query: str,
    patterns: tuple[tuple[re.Pattern[str], int], ...],
) -> int:
    return sum(weight for pattern, weight in patterns if pattern.search(query))
