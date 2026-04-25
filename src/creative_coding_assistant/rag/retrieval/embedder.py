"""Query embedding boundary contracts for KB retrieval."""

from __future__ import annotations

from typing import Protocol


class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> list[float]:
        """Return an embedding vector for a retrieval query."""
