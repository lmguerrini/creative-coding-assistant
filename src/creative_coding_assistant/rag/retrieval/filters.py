"""Chroma metadata filter translation for KB retrieval."""

from __future__ import annotations

from creative_coding_assistant.rag.retrieval.models import KnowledgeBaseRetrievalFilter


def build_kb_where_filter(
    filters: KnowledgeBaseRetrievalFilter,
) -> dict[str, object] | None:
    conditions: list[dict[str, object]] = []

    domain_condition = _build_domain_condition(filters)
    if domain_condition is not None:
        conditions.append(domain_condition)
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


def _build_domain_condition(
    filters: KnowledgeBaseRetrievalFilter,
) -> dict[str, object] | None:
    if not filters.domains:
        return None
    if len(filters.domains) == 1:
        return {"domain": filters.domains[0].value}
    return {
        "$or": [{"domain": domain.value} for domain in filters.domains],
    }
