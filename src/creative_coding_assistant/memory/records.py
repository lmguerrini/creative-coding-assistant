"""Conversions between memory schemas and vector records."""

from __future__ import annotations

from datetime import datetime

from creative_coding_assistant.contracts import AssistantMode, CreativeCodingDomain
from creative_coding_assistant.memory.schemas import (
    ConversationRole,
    ConversationSummaryRecord,
    ConversationSummaryWrite,
    ConversationTurnRecord,
    ConversationTurnWrite,
    ProjectMemoryKind,
    ProjectMemoryRecord,
    ProjectMemoryWrite,
)
from creative_coding_assistant.vectorstore import (
    ChromaCollection,
    StoredVectorRecord,
    VectorRecord,
    VectorRecordKind,
)
from creative_coding_assistant.vectorstore.metadata import ChromaRecordMetadata


def turn_to_vector_record(record_id: str, turn: ConversationTurnWrite) -> VectorRecord:
    return VectorRecord(
        id=record_id,
        document=turn.content,
        metadata=ChromaRecordMetadata(
            collection=ChromaCollection.CONVERSATION_TURNS,
            record_kind=VectorRecordKind.CONVERSATION_TURN,
            source_id=record_id,
            domain=turn.domain,
            mode=turn.mode,
            conversation_id=turn.conversation_id,
            project_id=turn.project_id,
            extras={
                "created_at": turn.created_at.isoformat(),
                "role": turn.role.value,
                "turn_index": turn.turn_index,
            },
        ),
        embedding=turn.embedding,
    )


def summary_to_vector_record(
    record_id: str,
    summary: ConversationSummaryWrite,
) -> VectorRecord:
    return VectorRecord(
        id=record_id,
        document=summary.content,
        metadata=ChromaRecordMetadata(
            collection=ChromaCollection.CONVERSATION_SUMMARIES,
            record_kind=VectorRecordKind.CONVERSATION_SUMMARY,
            source_id=record_id,
            domain=summary.domain,
            conversation_id=summary.conversation_id,
            project_id=summary.project_id,
            extras={
                "covered_turn_count": summary.covered_turn_count,
                "created_at": summary.created_at.isoformat(),
            },
        ),
        embedding=summary.embedding,
    )


def project_memory_to_vector_record(
    record_id: str,
    memory: ProjectMemoryWrite,
) -> VectorRecord:
    return VectorRecord(
        id=record_id,
        document=memory.content,
        metadata=ChromaRecordMetadata(
            collection=ChromaCollection.PROJECT_MEMORY,
            record_kind=VectorRecordKind.PROJECT_MEMORY,
            source_id=record_id,
            domain=memory.domain,
            project_id=memory.project_id,
            extras={
                "created_at": memory.created_at.isoformat(),
                "memory_kind": memory.memory_kind.value,
                "source": memory.source,
            },
        ),
        embedding=memory.embedding,
    )


def turn_from_stored_record(record: StoredVectorRecord) -> ConversationTurnRecord:
    metadata = record.metadata
    return ConversationTurnRecord(
        content=_document(record),
        created_at=_metadata_datetime(metadata, "created_at"),
        project_id=_optional_text(metadata, "project_id"),
        domain=_optional_domain(metadata),
        conversation_id=_required_text(metadata, "conversation_id"),
        turn_index=_metadata_int(metadata, "turn_index"),
        role=ConversationRole(_required_text(metadata, "role")),
        mode=_optional_mode(metadata),
    )


def summary_from_stored_record(record: StoredVectorRecord) -> ConversationSummaryRecord:
    metadata = record.metadata
    return ConversationSummaryRecord(
        content=_document(record),
        created_at=_metadata_datetime(metadata, "created_at"),
        project_id=_optional_text(metadata, "project_id"),
        domain=_optional_domain(metadata),
        conversation_id=_required_text(metadata, "conversation_id"),
        covered_turn_count=_metadata_int(metadata, "covered_turn_count"),
    )


def project_memory_from_stored_record(
    record: StoredVectorRecord,
) -> ProjectMemoryRecord:
    metadata = record.metadata
    return ProjectMemoryRecord(
        content=_document(record),
        created_at=_metadata_datetime(metadata, "created_at"),
        project_id=_required_text(metadata, "project_id"),
        domain=_optional_domain(metadata),
        memory_kind=ProjectMemoryKind(_required_text(metadata, "memory_kind")),
        source=_required_text(metadata, "source"),
    )


def _document(record: StoredVectorRecord) -> str:
    if not record.document:
        raise ValueError(f"Memory record is missing document content: {record.id}")
    return record.document


def _required_text(metadata: dict[str, object], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Memory metadata is missing required text key: {key}")
    return value


def _optional_text(metadata: dict[str, object], key: str) -> str | None:
    value = metadata.get(key)
    return value if isinstance(value, str) and value.strip() else None


def _metadata_int(metadata: dict[str, object], key: str) -> int:
    value = metadata.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Memory metadata is missing required integer key: {key}")
    return value


def _metadata_datetime(metadata: dict[str, object], key: str) -> datetime:
    return datetime.fromisoformat(_required_text(metadata, key))


def _optional_domain(metadata: dict[str, object]) -> CreativeCodingDomain | None:
    value = _optional_text(metadata, "domain")
    return CreativeCodingDomain(value) if value is not None else None


def _optional_mode(metadata: dict[str, object]) -> AssistantMode | None:
    value = _optional_text(metadata, "mode")
    return AssistantMode(value) if value is not None else None
