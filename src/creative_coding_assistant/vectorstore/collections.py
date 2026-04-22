"""Chroma collection names separated by concern."""

from __future__ import annotations

from enum import Enum


class ChromaCollection(str, Enum):
    KB_OFFICIAL_DOCS = "kb_official_docs"
    CONVERSATION_TURNS = "conversation_turns"
    CONVERSATION_SUMMARIES = "conversation_summaries"
    PROJECT_MEMORY = "project_memory"
    EVAL_TRACES = "eval_traces"
    PREVIEW_ARTIFACTS_INDEX = "preview_artifacts_index"


def collection_names() -> tuple[str, ...]:
    """Return all persistent Chroma collections for bootstrap validation."""

    return tuple(collection.value for collection in ChromaCollection)
