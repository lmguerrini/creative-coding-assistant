"""Chroma metadata filter translation for KB retrieval."""

from __future__ import annotations

from creative_coding_assistant.rag.retrieval.models import KnowledgeBaseRetrievalFilter


def build_kb_where_filter(
    filters: KnowledgeBaseRetrievalFilter,
) -> dict[str, object] | None:
    conditions: list[dict[str, object]] = []

    if filters.domain is not None:
        conditions.append({"domain": filters.domain.value})
    if filters.source_id is not None:
        conditions.append({"source_id": filters.source_id})
    if filters.source_type is not None:
        conditions.append({"source_type": filters.source_type.value})
    if filters.publisher is not None:
        conditions.append({"publisher": filters.publisher})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}
