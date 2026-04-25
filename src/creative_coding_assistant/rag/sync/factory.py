"""Factory helpers for official KB chunk embedders."""

from __future__ import annotations

from loguru import logger

from creative_coding_assistant.core import Settings
from creative_coding_assistant.rag.sync.embedding import (
    ChunkEmbedder,
    OpenAIChunkEmbedder,
)


def build_chunk_embedder(settings: Settings) -> ChunkEmbedder | None:
    """Build the configured KB chunk embedder when embedding config is available."""

    if not settings.has_openai_embedding_config:
        logger.info("Chunk embedder not configured; KB indexing remains explicit.")
        return None

    return OpenAIChunkEmbedder(settings=settings)
