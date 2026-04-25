"""Official knowledge-base retrieval contracts and implementation."""

from creative_coding_assistant.rag.retrieval.embedder import QueryEmbedder
from creative_coding_assistant.rag.retrieval.factory import build_query_embedder
from creative_coding_assistant.rag.retrieval.models import (
    KnowledgeBaseRetrievalFilter,
    KnowledgeBaseRetrievalRequest,
    KnowledgeBaseRetrievalResponse,
    KnowledgeBaseSearchResult,
)
from creative_coding_assistant.rag.retrieval.openai_embedder import OpenAIQueryEmbedder
from creative_coding_assistant.rag.retrieval.search import (
    KnowledgeBaseRetriever,
)

__all__ = [
    "build_query_embedder",
    "KnowledgeBaseRetrievalFilter",
    "KnowledgeBaseRetrievalRequest",
    "KnowledgeBaseRetrievalResponse",
    "KnowledgeBaseSearchResult",
    "KnowledgeBaseRetriever",
    "OpenAIQueryEmbedder",
    "QueryEmbedder",
]
