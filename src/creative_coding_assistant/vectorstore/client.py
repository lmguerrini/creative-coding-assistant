"""Chroma client helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from creative_coding_assistant.core.config import Settings
from creative_coding_assistant.vectorstore.collections import (
    CollectionDefinition,
    collection_definitions,
)


def create_chroma_client(settings: Settings | None = None, *, path: Path | None = None):
    """Create a persistent Chroma client and ensure its directory exists."""

    resolved_path = path or (settings or Settings()).chroma_persist_dir
    resolved_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(resolved_path))


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

    return {
        definition.name.value: ensure_collection(client, definition)
        for definition in collection_definitions()
    }
