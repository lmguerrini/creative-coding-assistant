"""OpenAI-backed query embedder for KB retrieval."""

from __future__ import annotations

from typing import Any

from creative_coding_assistant.core import Settings, load_settings
from creative_coding_assistant.rag.embeddings import OpenAIEmbeddingClient
from creative_coding_assistant.rag.retrieval.embedder import QueryEmbedder


class OpenAIQueryEmbedder(QueryEmbedder):
    """Translate retrieval queries into OpenAI embedding requests."""

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

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Query embedding text must not be empty.")
        return self._embedder.embed_texts((text,))[0]
