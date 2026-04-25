"""Semantic retrieval over indexed official knowledge-base chunks."""

from __future__ import annotations

from typing import Any

from loguru import logger

from creative_coding_assistant.rag.retrieval.embedder import QueryEmbedder
from creative_coding_assistant.rag.retrieval.filters import build_kb_where_filter
from creative_coding_assistant.rag.retrieval.models import (
    KnowledgeBaseRetrievalRequest,
    KnowledgeBaseRetrievalResponse,
    KnowledgeBaseSearchResult,
)
from creative_coding_assistant.rag.sources import OfficialSourceType
from creative_coding_assistant.vectorstore import (
    ChromaCollection,
    ChromaRepository,
    QueryMatchRecord,
    get_collection_definition,
)


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
        where = build_kb_where_filter(request.filters)
        matches = self._repository.query(
            embedding=query_embedding,
            limit=request.limit,
            where=where,
        )
        results = tuple(self._build_result(match) for match in matches)
        logger.info(
            "Retrieved {} KB chunk(s) for query '{}'",
            len(results),
            request.query,
        )
        return KnowledgeBaseRetrievalResponse(request=request, results=results)

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
