"""Factory helpers for retrieval query embedders."""

from __future__ import annotations

from loguru import logger

from creative_coding_assistant.core import Settings
from creative_coding_assistant.rag.retrieval.embedder import QueryEmbedder
from creative_coding_assistant.rag.retrieval.openai_embedder import (
    OpenAIQueryEmbedder,
)


def build_query_embedder(settings: Settings) -> QueryEmbedder | None:
    """Build the configured query embedder when embedding config is available."""

    if not settings.has_openai_embedding_config:
        logger.info("Query embedder not configured; retrieval remains optional.")
        return None

    return OpenAIQueryEmbedder(settings=settings)
