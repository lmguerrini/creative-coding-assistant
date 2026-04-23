"""Official knowledge-base retrieval contracts and implementation."""

from creative_coding_assistant.rag.retrieval.models import (
    KnowledgeBaseRetrievalFilter,
    KnowledgeBaseRetrievalRequest,
    KnowledgeBaseRetrievalResponse,
    KnowledgeBaseSearchResult,
)
from creative_coding_assistant.rag.retrieval.search import (
    KnowledgeBaseRetriever,
    QueryEmbedder,
)

__all__ = [
    "KnowledgeBaseRetrievalFilter",
    "KnowledgeBaseRetrievalRequest",
    "KnowledgeBaseRetrievalResponse",
    "KnowledgeBaseSearchResult",
    "KnowledgeBaseRetriever",
    "QueryEmbedder",
]
