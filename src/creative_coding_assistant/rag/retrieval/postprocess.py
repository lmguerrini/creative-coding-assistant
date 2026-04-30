"""Post-processing helpers for official KB retrieval results."""

from __future__ import annotations

import re
from collections.abc import Sequence
from difflib import SequenceMatcher

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
_DEDUP_TEXT_PREFIX_LENGTH = 280
_DEDUP_SIMILARITY_THRESHOLD = 0.80
_MAX_RESULTS_PER_SOURCE = 2
_INLINE_TYPE_ANNOTATION_PATTERN = re.compile(
    r"\b[a-z_][a-z0-9_]*\s*:\s*[^,)=\]}]+"
)
_ANGLE_BRACKET_PATTERN = re.compile(r"<[^>]+>")
_NON_ALPHANUMERIC_PATTERN = re.compile(r"[^a-z0-9]+")


def select_retrieval_results(
    results: Sequence[KnowledgeBaseSearchResult],
    *,
    limit: int,
) -> tuple[KnowledgeBaseSearchResult, ...]:
    """Filter low-value retrieval hits while preserving a safe fallback."""

    filtered = tuple(result for result in results if not _is_low_value_chunk(result))
    if filtered:
        deduplicated = _deduplicate_results(filtered)
        return _apply_source_diversity(deduplicated, limit=limit)
    return tuple(results[:limit])


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


def _apply_source_diversity(
    results: Sequence[KnowledgeBaseSearchResult],
    *,
    limit: int,
) -> tuple[KnowledgeBaseSearchResult, ...]:
    kept, _ = _take_source_limited_results(results, limit=limit)
    return kept


def _take_source_limited_results(
    results: Sequence[KnowledgeBaseSearchResult],
    *,
    limit: int,
    existing_counts: dict[str, int] | None = None,
) -> tuple[tuple[KnowledgeBaseSearchResult, ...], dict[str, int]]:
    kept: list[KnowledgeBaseSearchResult] = []
    counts = dict(existing_counts or {})

    for result in results:
        source_count = counts.get(result.source_id, 0)
        if source_count >= _MAX_RESULTS_PER_SOURCE:
            continue

        kept.append(result)
        counts[result.source_id] = source_count + 1
        if len(kept) >= limit:
            break

    return tuple(kept), counts


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
