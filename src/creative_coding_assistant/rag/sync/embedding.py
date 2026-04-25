"""Chunk embedding boundaries for official KB indexing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from creative_coding_assistant.core import Settings, load_settings
from creative_coding_assistant.rag.embeddings import OpenAIEmbeddingClient
from creative_coding_assistant.rag.sync.models import OfficialSourceChunk


class ChunkEmbedder(Protocol):
    def embed_chunks(
        self,
        chunks: Sequence[OfficialSourceChunk],
    ) -> tuple[list[float], ...]:
        """Return one embedding vector per KB chunk."""


class OpenAIChunkEmbedder(ChunkEmbedder):
    """Embed official KB chunks through the OpenAI embeddings API."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        model: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        resolved_settings = settings or load_settings()
        self._embedder = OpenAIEmbeddingClient(
            settings=resolved_settings,
            model=model,
            api_key=api_key,
            client=client,
        )

    def embed_chunks(
        self,
        chunks: Sequence[OfficialSourceChunk],
    ) -> tuple[list[float], ...]:
        if not chunks:
            return ()
        return self._embedder.embed_texts(tuple(chunk.text for chunk in chunks))
