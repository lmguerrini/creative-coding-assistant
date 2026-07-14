"""Post-processing helpers for official KB retrieval results."""

from __future__ import annotations

import re
from collections.abc import Sequence
from difflib import SequenceMatcher

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.rag.retrieval.domain_intent import (
    detect_domain_intent,
    detect_explicit_query_domains,
)
from creative_coding_assistant.rag.retrieval.models import KnowledgeBaseSearchResult

_GENERIC_EXAMPLE_PHRASES: tuple[str, ...] = (
    "select an example from the sidebar",
    ("skip to main content menu reference tutorials examples contribute community about start coding donate"),
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
_NON_ACTIONABLE_DOC_GAP_PHRASES: tuple[str, ...] = (
    "all the details of how to write",
    "too much for these articles",
    "unfortunately undocumented",
    "read through the examples or the code",
)
_INDEX_ONLY_SOURCE_IDS = frozenset(
    {
        "tone_js_docs",
        "three_examples",
        "three_docs",
    }
)
_DEDUP_TEXT_PREFIX_LENGTH = 280
_DEDUP_SIMILARITY_THRESHOLD = 0.80
_INLINE_TYPE_ANNOTATION_PATTERN = re.compile(r"\b[a-z_][a-z0-9_]*\s*:\s*[^,)=\]}]+")
_ANGLE_BRACKET_PATTERN = re.compile(r"<[^>]+>")
_NON_ALPHANUMERIC_PATTERN = re.compile(r"[^a-z0-9]+")
_MAX_PRIMARY_CHUNKS_PER_SOURCE = 2


def select_retrieval_results(
    results: Sequence[KnowledgeBaseSearchResult],
    *,
    limit: int,
    query: str | None = None,
    requested_domains: Sequence[CreativeCodingDomain] = (),
) -> tuple[KnowledgeBaseSearchResult, ...]:
    """Filter low-value retrieval hits without presenting junk as evidence."""

    domain_candidates = _apply_domain_intent_filter(
        results,
        query=query,
        requested_domains=requested_domains,
    )
    source_filtered = tuple(result for result in domain_candidates if result.source_id not in _INDEX_ONLY_SOURCE_IDS)
    # Judge broad manual pages by their chunks. The Three.js manual contains two
    # landing shells followed by substantive guidance, so excluding its entire
    # source would discard useful evidence.
    filtered = tuple(result for result in source_filtered if not _is_low_value_chunk(result))
    deduplicated = _deduplicate_results(filtered)
    return _select_domain_diverse_results(
        deduplicated,
        limit=limit,
        requested_domains=requested_domains,
    )


def _is_low_value_chunk(result: KnowledgeBaseSearchResult) -> bool:
    normalized_text = _normalize_text(result.text)
    normalized_title = _normalize_text(result.document_title)

    if _is_heading_only_chunk(result, normalized_text=normalized_text):
        return True

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

    if sum(phrase in normalized_text for phrase in _NON_ACTIONABLE_DOC_GAP_PHRASES) >= 2:
        return True

    return False


def _is_heading_only_chunk(
    result: KnowledgeBaseSearchResult,
    *,
    normalized_text: str,
) -> bool:
    if not normalized_text:
        return True

    normalized_titles = {
        _normalize_text(result.document_title),
        _normalize_text(result.registry_title),
    }
    return any(
        normalized_text == title or title.startswith(f"{normalized_text} ") for title in normalized_titles if title
    )


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
        if any(_is_near_duplicate(preview, seen_preview) for seen_preview in seen_previews):
            continue

        kept.append(result)
        previews.setdefault(result.source_id, []).append(preview)

    return tuple(kept)


def _apply_domain_intent_filter(
    results: Sequence[KnowledgeBaseSearchResult],
    *,
    query: str | None,
    requested_domains: Sequence[CreativeCodingDomain],
) -> tuple[KnowledgeBaseSearchResult, ...]:
    if len(tuple(dict.fromkeys(requested_domains))) > 1:
        return tuple(results)

    if query is None:
        return tuple(results)

    explicit_domains = detect_explicit_query_domains(query)
    if len(explicit_domains) > 1:
        narrowed = tuple(result for result in results if result.domain in explicit_domains)
        return narrowed or tuple(results)

    intent = detect_domain_intent(query)
    if intent is None:
        return tuple(results)

    primary_results = tuple(result for result in results if result.domain == intent.primary_domain)
    if primary_results:
        return primary_results

    narrowed = tuple(result for result in results if result.domain in intent.allowed_domains)
    return narrowed or tuple(results)


def _select_domain_diverse_results(
    results: Sequence[KnowledgeBaseSearchResult],
    *,
    limit: int,
    requested_domains: Sequence[CreativeCodingDomain],
) -> tuple[KnowledgeBaseSearchResult, ...]:
    requested = tuple(dict.fromkeys(requested_domains))
    if limit < 2:
        return tuple(results[:limit])

    requested_set = set(requested)
    selected: list[KnowledgeBaseSearchResult] = []
    covered_domains: set[CreativeCodingDomain] = set()

    if len(requested) > 1:
        for result in results:
            if result.domain not in requested_set or result.domain in covered_domains:
                continue
            selected.append(result)
            covered_domains.add(result.domain)
            if len(selected) == limit or covered_domains == requested_set:
                break

    selected_ids = {_result_identity(result) for result in selected}
    source_counts: dict[str, int] = {}
    for result in selected:
        source_counts[result.source_id] = source_counts.get(result.source_id, 0) + 1

    # Fill open slots with distinct sources first. This keeps repeated chunks
    # from consuming a small context window before another independent source is
    # considered.
    for result in results:
        if len(selected) == limit:
            break
        identity = _result_identity(result)
        if identity in selected_ids:
            continue
        if source_counts.get(result.source_id, 0) > 0:
            continue
        selected.append(result)
        selected_ids.add(identity)
        source_counts[result.source_id] = source_counts.get(result.source_id, 0) + 1

    # Once every available source has had an opportunity, allow a second chunk
    # from a source to preserve useful related details.
    for result in results:
        if len(selected) == limit:
            break
        identity = _result_identity(result)
        if identity in selected_ids:
            continue
        if source_counts.get(result.source_id, 0) >= _MAX_PRIMARY_CHUNKS_PER_SOURCE:
            continue
        selected.append(result)
        selected_ids.add(identity)
        source_counts[result.source_id] = source_counts.get(result.source_id, 0) + 1

    for result in results:
        if len(selected) == limit:
            break
        identity = _result_identity(result)
        if identity in selected_ids:
            continue
        selected.append(result)
        selected_ids.add(identity)

    return tuple(selected)


def _result_identity(result: KnowledgeBaseSearchResult) -> tuple[str, int, str]:
    return (result.record_id, result.chunk_index, result.chunk_hash)


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
