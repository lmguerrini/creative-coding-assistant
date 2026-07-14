"""Chroma client helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.telemetry.product import ProductTelemetryClient, ProductTelemetryEvent
from overrides import override

from creative_coding_assistant.core.config import Settings
from creative_coding_assistant.vectorstore.collections import (
    CollectionDefinition,
    collection_definitions,
)


class NoopChromaTelemetryClient(ProductTelemetryClient):
    """Local product telemetry sink that never opens an external boundary."""

    @override
    def capture(self, event: ProductTelemetryEvent) -> None:
        del event


def create_chroma_client(settings: Settings | None = None, *, path: Path | None = None):
    """Create a persistent Chroma client and ensure its directory exists."""

    resolved_path = path or (settings or Settings()).chroma_persist_dir
    resolved_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(resolved_path),
        settings=ChromaSettings(
            anonymized_telemetry=False,
            chroma_product_telemetry_impl=("creative_coding_assistant.vectorstore.client.NoopChromaTelemetryClient"),
            chroma_telemetry_impl=("creative_coding_assistant.vectorstore.client.NoopChromaTelemetryClient"),
        ),
    )


def ensure_collection(client: Any, definition: CollectionDefinition):
    """Create or return a Chroma collection with responsibility metadata."""

    return client.get_or_create_collection(
        name=definition.name.value,
        metadata={
            "responsibility": definition.responsibility,
            "schema_version": 1,
        },
    )


def ensure_project_collections(client: Any) -> dict[str, Any]:
    """Ensure all configured project collections exist."""

    return {definition.name.value: ensure_collection(client, definition) for definition in collection_definitions()}
