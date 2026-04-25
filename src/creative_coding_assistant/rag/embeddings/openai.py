"""Shared OpenAI embeddings client for retrieval and indexing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from loguru import logger

from creative_coding_assistant.core import Settings, load_settings


class OpenAIEmbeddingClient:
    """Dispatch text embedding requests through the OpenAI embeddings API."""

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

    def embed_texts(self, texts: Sequence[str]) -> tuple[list[float], ...]:
        normalized_texts = _normalize_texts(texts)
        if not normalized_texts:
            return ()

        client = self._client or _build_openai_client(api_key=self._api_key)
        logger.info(
            "Dispatching {} embedding input(s) to OpenAI",
            len(normalized_texts),
        )
        response = client.embeddings.create(
            model=self._model,
            input=list(normalized_texts),
        )
        return _extract_embeddings(response)


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


def _normalize_texts(texts: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for index, text in enumerate(texts):
        cleaned = str(text).strip()
        if not cleaned:
            raise ValueError(
                f"Embedding text at index {index} must not be empty."
            )
        normalized.append(cleaned)
    return tuple(normalized)


def _extract_embeddings(response: Any) -> tuple[list[float], ...]:
    data = _read_field(response, "data", ()) or ()
    if not data:
        raise ValueError("OpenAI embedding response did not include embedding data.")

    embeddings: list[list[float]] = []
    for item in data:
        embedding = _read_field(item, "embedding", ()) or ()
        values = [float(value) for value in embedding]
        if not values:
            raise ValueError("OpenAI embedding response did not include an embedding.")
        embeddings.append(values)
    return tuple(embeddings)


def _read_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)
