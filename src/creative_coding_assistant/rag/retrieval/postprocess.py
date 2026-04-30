"""Post-processing helpers for official KB retrieval results."""

from __future__ import annotations

from collections.abc import Sequence

from creative_coding_assistant.rag.retrieval.models import KnowledgeBaseSearchResult

_GENERIC_EXAMPLE_PHRASES: tuple[str, ...] = (
    "select an example from the sidebar",
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


def select_retrieval_results(
    results: Sequence[KnowledgeBaseSearchResult],
    *,
    limit: int,
) -> tuple[KnowledgeBaseSearchResult, ...]:
    """Filter low-value retrieval hits while preserving a safe fallback."""

    filtered = tuple(result for result in results if not _is_low_value_chunk(result))
    if filtered:
        return filtered[:limit]
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


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().split())
