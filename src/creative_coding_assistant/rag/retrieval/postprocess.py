"""Post-processing helpers for official KB retrieval results."""

from __future__ import annotations

import re
from collections.abc import Sequence
from difflib import SequenceMatcher

from creative_coding_assistant.rag.retrieval.domain_intent import (
    detect_domain_intent,
    detect_explicit_query_domains,
)
from creative_coding_assistant.rag.retrieval.models import KnowledgeBaseSearchResult

_GENERIC_EXAMPLE_PHRASES: tuple[str, ...] = (
    "select an example from the sidebar",
    (
        "skip to main content menu reference tutorials examples contribute "
        "community about start coding donate"
    ),
    "three.js examples three.js examples",
)
_GENERIC_DOC_INDEX_PHRASES: tuple[str, ...] = (
    "three.js docs",
    "docs manual",
    "animationaction",
    "buffergeometry",
    "object3d",
    "uniformsgroup",
)
_GENERIC_MANUAL_PHRASES: tuple[str, ...] = (
    "three.js manual",
    "docs manual",
    "en fr ru",
    "中文",
    "日本語",
)
_HARD_FILTERED_SOURCE_IDS = frozenset(
    {
        "three_examples",
        "three_docs",
        "three_manual",
    }
)
_DEDUP_TEXT_PREFIX_LENGTH = 280
_DEDUP_SIMILARITY_THRESHOLD = 0.80
_INLINE_TYPE_ANNOTATION_PATTERN = re.compile(
    r"\b[a-z_][a-z0-9_]*\s*:\s*[^,)=\]}]+"
)
_ANGLE_BRACKET_PATTERN = re.compile(r"<[^>]+>")
_NON_ALPHANUMERIC_PATTERN = re.compile(r"[^a-z0-9]+")


def select_retrieval_results(
    results: Sequence[KnowledgeBaseSearchResult],
    *,
    limit: int,
    query: str | None = None,
) -> tuple[KnowledgeBaseSearchResult, ...]:
    """Filter low-value retrieval hits while preserving a safe fallback."""

    domain_candidates = _apply_domain_intent_filter(results, query=query)
    source_filtered = tuple(
        result
        for result in domain_candidates
        if result.source_id not in _HARD_FILTERED_SOURCE_IDS
    )
    candidates = source_filtered or domain_candidates

    filtered = tuple(result for result in candidates if not _is_low_value_chunk(result))
    filtered_candidates = filtered or candidates

    deduplicated = _deduplicate_results(filtered_candidates)
    return deduplicated[:limit]


def _is_low_value_chunk(result: KnowledgeBaseSearchResult) -> bool:
    normalized_text = _normalize_text(result.text)
    normalized_title = _normalize_text(result.document_title)

    if _contains_any(normalized_text, _GENERIC_EXAMPLE_PHRASES):
        return True

    if (
        result.source_id == "three_docs"
        and normalized_title == "three.js docs"
        and _contains_any(normalized_text, _GENERIC_DOC_INDEX_PHRASES)
    ):
        return True

    if (
        result.source_id == "three_manual"
        and normalized_title == "three.js manual"
        and _contains_any(normalized_text, _GENERIC_MANUAL_PHRASES)
    ):
        return True

    return False


def _contains_any(text: str, phrases: Sequence[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _deduplicate_results(
    results: Sequence[KnowledgeBaseSearchResult],
) -> tuple[KnowledgeBaseSearchResult, ...]:
    kept: list[KnowledgeBaseSearchResult] = []
    previews: dict[str, list[str]] = {}

    for result in results:
        preview = _normalized_preview(result.text)
        seen_previews = previews.get(result.source_id, [])
        if any(
            _is_near_duplicate(preview, seen_preview)
            for seen_preview in seen_previews
        ):
            continue

        kept.append(result)
        previews.setdefault(result.source_id, []).append(preview)

    return tuple(kept)


def _apply_domain_intent_filter(
    results: Sequence[KnowledgeBaseSearchResult],
    *,
    query: str | None,
) -> tuple[KnowledgeBaseSearchResult, ...]:
    if query is None:
        return tuple(results)

    explicit_domains = detect_explicit_query_domains(query)
    if len(explicit_domains) > 1:
        narrowed = tuple(
            result for result in results if result.domain in explicit_domains
        )
        return narrowed or tuple(results)

    intent = detect_domain_intent(query)
    if intent is None:
        return tuple(results)

    primary_results = tuple(
        result for result in results if result.domain == intent.primary_domain
    )
    if primary_results:
        return primary_results

    narrowed = tuple(
        result for result in results if result.domain in intent.allowed_domains
    )
    return narrowed or tuple(results)


def _is_near_duplicate(left: str, right: str) -> bool:
    if left == right:
        return True
    return SequenceMatcher(a=left, b=right).ratio() >= _DEDUP_SIMILARITY_THRESHOLD


def _normalized_preview(value: str) -> str:
    return _normalize_dedup_text(value)[:_DEDUP_TEXT_PREFIX_LENGTH]


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def _normalize_dedup_text(value: str) -> str:
    normalized = _normalize_text(value)
    without_types = _INLINE_TYPE_ANNOTATION_PATTERN.sub(" ", normalized)
    without_tags = _ANGLE_BRACKET_PATTERN.sub(" ", without_types)
    alphanumeric = _NON_ALPHANUMERIC_PATTERN.sub(" ", without_tags)
    return " ".join(alphanumeric.split())
