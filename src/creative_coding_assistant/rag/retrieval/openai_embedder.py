"""OpenAI-backed query embedder for KB retrieval."""

from __future__ import annotations

from typing import Any

from loguru import logger

from creative_coding_assistant.core import Settings, load_settings
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
        self._settings = settings or load_settings()
        self._model = model or self._settings.openai_embedding_model
        self._api_key = api_key
        self._client = client

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Query embedding text must not be empty.")

        client = self._client or _build_openai_client(api_key=self._api_key)
        logger.info(
            "Dispatching query embedding request to OpenAI with {} chars",
            len(text),
        )
        response = client.embeddings.create(
            model=self._model,
            input=text,
        )
        return _extract_embedding(response)


def _build_openai_client(*, api_key: str | None) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("The OpenAI SDK is not installed.") from exc

    settings = load_settings()
    resolved_api_key = api_key or settings.get_openai_api_key()
    if not resolved_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=resolved_api_key)


def _extract_embedding(response: Any) -> list[float]:
    data = _read_field(response, "data", ()) or ()
    if not data:
        raise ValueError("OpenAI embedding response did not include embedding data.")

    embedding = _read_field(data[0], "embedding", ()) or ()
    values = [float(value) for value in embedding]
    if not values:
        raise ValueError("OpenAI embedding response did not include an embedding.")
    return values


def _read_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)
