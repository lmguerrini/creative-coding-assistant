"""Official knowledge-base sync contracts and implementation boundaries."""

from creative_coding_assistant.rag.sync.chunking import (
    ChunkingPolicy,
    OfficialSourceChunker,
)
from creative_coding_assistant.rag.sync.embedding import (
    ChunkEmbedder,
    OpenAIChunkEmbedder,
)
from creative_coding_assistant.rag.sync.factory import build_chunk_embedder
from creative_coding_assistant.rag.sync.fetcher import (
    OfficialSourceFetcher,
    SourceTransport,
    TransportResponse,
    UrllibSourceTransport,
    default_sync_request,
)
from creative_coding_assistant.rag.sync.indexing import OfficialKnowledgeBaseIndexer
from creative_coding_assistant.rag.sync.models import (
    FetchedSourceDocument,
    NormalizedSourceDocument,
    OfficialSourceChunk,
    OfficialSourceSyncRequest,
    SourceContentFormat,
)
from creative_coding_assistant.rag.sync.normalize import OfficialSourceNormalizer

__all__ = [
    "build_chunk_embedder",
    "ChunkEmbedder",
    "ChunkingPolicy",
    "FetchedSourceDocument",
    "OfficialKnowledgeBaseIndexer",
    "NormalizedSourceDocument",
    "OfficialSourceChunk",
    "OfficialSourceChunker",
    "OpenAIChunkEmbedder",
    "OfficialSourceFetcher",
    "OfficialSourceSyncRequest",
    "OfficialSourceNormalizer",
    "SourceTransport",
    "SourceContentFormat",
    "TransportResponse",
    "UrllibSourceTransport",
    "default_sync_request",
]
