"""Chroma collection names and responsibilities."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ChromaCollection(StrEnum):
    KB_OFFICIAL_DOCS = "kb_official_docs"
    CONVERSATION_TURNS = "conversation_turns"
    CONVERSATION_SUMMARIES = "conversation_summaries"
    PROJECT_MEMORY = "project_memory"
    EVAL_TRACES = "eval_traces"
    PREVIEW_ARTIFACTS_INDEX = "preview_artifacts_index"


class CollectionDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: ChromaCollection
    responsibility: str = Field(min_length=1)
    primary_metadata: tuple[str, ...] = Field(default_factory=tuple)


COLLECTION_DEFINITIONS: tuple[CollectionDefinition, ...] = (
    CollectionDefinition(
        name=ChromaCollection.KB_OFFICIAL_DOCS,
        responsibility="Official documentation chunks and sync state.",
        primary_metadata=("domain", "source_id", "source_url", "content_hash"),
    ),
    CollectionDefinition(
        name=ChromaCollection.CONVERSATION_TURNS,
        responsibility="Durable user and assistant turns.",
        primary_metadata=("conversation_id", "project_id", "mode", "domain"),
    ),
    CollectionDefinition(
        name=ChromaCollection.CONVERSATION_SUMMARIES,
        responsibility="Running summaries for long conversations.",
        primary_metadata=("conversation_id", "project_id", "turn_count"),
    ),
    CollectionDefinition(
        name=ChromaCollection.PROJECT_MEMORY,
        responsibility="Stable project preferences, goals, and decisions.",
        primary_metadata=("project_id", "domain", "memory_kind"),
    ),
    CollectionDefinition(
        name=ChromaCollection.EVAL_TRACES,
        responsibility="Session traces and live evaluation observations.",
        primary_metadata=("trace_id", "conversation_id", "route", "mode"),
    ),
    CollectionDefinition(
        name=ChromaCollection.PREVIEW_ARTIFACTS_INDEX,
        responsibility="Preview artifact manifests and capture metadata.",
        primary_metadata=("artifact_id", "project_id", "domain", "content_hash"),
    ),
)


def collection_names() -> tuple[str, ...]:
    """Return all persistent Chroma collection names."""

    return tuple(collection.value for collection in ChromaCollection)


def collection_definitions() -> tuple[CollectionDefinition, ...]:
    """Return immutable collection definitions."""

    return COLLECTION_DEFINITIONS


def get_collection_definition(name: ChromaCollection) -> CollectionDefinition:
    """Return the definition for a configured collection."""

    for definition in COLLECTION_DEFINITIONS:
        if definition.name == name:
            return definition
    raise ValueError(f"Unknown Chroma collection: {name}")
