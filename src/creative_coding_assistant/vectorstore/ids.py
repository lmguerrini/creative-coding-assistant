"""Stable ID conventions for Chroma records."""

from __future__ import annotations

import hashlib
from enum import StrEnum

from creative_coding_assistant.vectorstore.collections import ChromaCollection


class VectorRecordKind(StrEnum):
    OFFICIAL_DOC_CHUNK = "official_doc_chunk"
    CONVERSATION_TURN = "conversation_turn"
    CONVERSATION_SUMMARY = "conversation_summary"
    PROJECT_MEMORY = "project_memory"
    EVAL_TRACE = "eval_trace"
    PREVIEW_ARTIFACT = "preview_artifact"


def make_record_id(
    *,
    collection: ChromaCollection,
    record_kind: VectorRecordKind,
    parts: tuple[str, ...],
    schema_version: int = 1,
) -> str:
    """Build a deterministic Chroma ID from stable semantic parts."""

    cleaned_parts = tuple(part.strip() for part in parts if part.strip())
    if not cleaned_parts:
        raise ValueError("At least one stable ID part is required.")
    if schema_version < 1:
        raise ValueError("Schema version must be at least 1.")

    digest_input = "\x1f".join(cleaned_parts).encode("utf-8")
    digest = hashlib.sha256(digest_input).hexdigest()[:24]
    return f"{collection.value}:{record_kind.value}:v{schema_version}:{digest}"
