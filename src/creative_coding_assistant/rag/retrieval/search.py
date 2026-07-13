"""Semantic retrieval over indexed official knowledge-base chunks."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from loguru import logger

from creative_coding_assistant.rag.retrieval.embedder import QueryEmbedder
from creative_coding_assistant.rag.retrieval.filters import build_kb_where_filter
from creative_coding_assistant.rag.retrieval.models import (
    KnowledgeBaseRetrievalRequest,
    KnowledgeBaseRetrievalResponse,
    KnowledgeBaseSearchResult,
)
from creative_coding_assistant.rag.retrieval.postprocess import (
    select_retrieval_results,
)
from creative_coding_assistant.rag.sources import OfficialSourceType
from creative_coding_assistant.vectorstore import (
    ChromaCollection,
    ChromaRepository,
    QueryMatchRecord,
    get_collection_definition,
)

# Keep bounded headroom for post-query removal of heading-only and index chunks.
_RETRIEVAL_CANDIDATE_MULTIPLIER = 4
_MAX_RETRIEVAL_CANDIDATES = 20


class KnowledgeBaseRetriever:
    """Search the indexed official knowledge base without orchestration concerns."""

    def __init__(self, *, client: Any, embedder: QueryEmbedder) -> None:
        definition = get_collection_definition(ChromaCollection.KB_OFFICIAL_DOCS)
        self._repository = ChromaRepository(client=client, definition=definition)
        self._embedder = embedder

    def search(
        self,
        request: KnowledgeBaseRetrievalRequest,
    ) -> KnowledgeBaseRetrievalResponse:
        query_embedding = self._embed_query(request.query)
        matches = self._query_candidates(
            request=request,
            query_embedding=query_embedding,
        )
        raw_results = tuple(self._build_result(match) for match in matches)
        results = select_retrieval_results(
            raw_results,
            limit=request.limit,
            query=request.query,
            requested_domains=request.filters.domains,
        )
        logger.info(
            "Retrieved {} KB chunk(s) for query '{}' after filtering {} raw match(es)",
            len(results),
            request.query,
            len(raw_results),
        )
        return KnowledgeBaseRetrievalResponse(request=request, results=results)

    def _query_candidates(
        self,
        *,
        request: KnowledgeBaseRetrievalRequest,
        query_embedding: list[float],
    ) -> tuple[QueryMatchRecord, ...]:
        requested_domains = tuple(dict.fromkeys(request.filters.domains))
        candidate_limit = self._candidate_limit(request.limit)
        if len(requested_domains) < 2:
            return self._repository.query(
                embedding=query_embedding,
                limit=candidate_limit,
                where=build_kb_where_filter(request.filters),
            )

        # A single nearest-neighbour query can fill the candidate pool with one
        # strong domain before another requested domain is considered. Query each
        # requested domain with the same embedding, then merge by distance. The
        # total candidate budget stays approximately equal to the single-query
        # budget while every explicitly requested domain gets a fair opportunity.
        per_domain_limit = max(
            1,
            math.ceil(candidate_limit / len(requested_domains)),
        )
        matches: list[QueryMatchRecord] = []
        for domain in requested_domains:
            domain_filters = request.filters.model_copy(
                update={"domain": domain, "domains": (domain,)}
            )
            matches.extend(
                self._repository.query(
                    embedding=query_embedding,
                    limit=per_domain_limit,
                    where=build_kb_where_filter(domain_filters),
                )
            )
        return _merge_matches_by_distance(matches)

    def _embed_query(self, query: str) -> list[float]:
        embedding = self._embedder.embed_query(query)
        if not embedding:
            raise ValueError("Retrieval query embedding must not be empty.")
        return embedding

    def _build_result(self, match: QueryMatchRecord) -> KnowledgeBaseSearchResult:
        metadata = match.metadata
        collection = metadata.get("collection")
        record_kind = metadata.get("record_kind")
        if collection != "kb_official_docs" or record_kind != "official_doc_chunk":
            raise ValueError("KB retrieval received a non-official-doc match.")

        distance = float(match.distance)
        return KnowledgeBaseSearchResult(
            record_id=match.id,
            source_id=str(metadata["source_id"]),
            domain=str(metadata["domain"]),
            source_type=OfficialSourceType(str(metadata["source_type"])),
            publisher=str(metadata["publisher"]),
            registry_title=str(metadata["registry_title"]),
            document_title=str(metadata["document_title"]),
            source_url=str(metadata["source_url"]),
            resolved_url=str(metadata.get("resolved_url"))
            if metadata.get("resolved_url") is not None
            else None,
            chunk_index=int(metadata["chunk_index"]),
            text=match.document or "",
            char_count=int(metadata["char_count"]),
            content_hash=str(metadata["content_hash"]),
            chunk_hash=str(metadata["chunk_hash"]),
            distance=distance,
            score=1.0 / (1.0 + distance),
        )

    def _candidate_limit(self, limit: int) -> int:
        return min(limit * _RETRIEVAL_CANDIDATE_MULTIPLIER, _MAX_RETRIEVAL_CANDIDATES)


def _merge_matches_by_distance(
    matches: Sequence[QueryMatchRecord],
) -> tuple[QueryMatchRecord, ...]:
    best_by_id: dict[str, QueryMatchRecord] = {}
    for match in matches:
        existing = best_by_id.get(match.id)
        if existing is None or match.distance < existing.distance:
            best_by_id[match.id] = match
    return tuple(sorted(best_by_id.values(), key=lambda match: match.distance))
