"""Chroma-backed memory repositories."""

from __future__ import annotations

from typing import Any

from creative_coding_assistant.memory.records import (
    project_memory_from_stored_record,
    project_memory_to_vector_record,
    summary_from_stored_record,
    summary_to_vector_record,
    turn_from_stored_record,
    turn_to_vector_record,
)
from creative_coding_assistant.memory.schemas import (
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
    ChromaRepository,
    VectorRecordKind,
    get_collection_definition,
    make_record_id,
)


class ConversationTurnRepository:
    def __init__(self, *, client: Any) -> None:
        self._repository = _build_repository(
            client,
            ChromaCollection.CONVERSATION_TURNS,
        )

    def upsert(self, turn: ConversationTurnWrite) -> str:
        record_id = make_record_id(
            collection=ChromaCollection.CONVERSATION_TURNS,
            record_kind=VectorRecordKind.CONVERSATION_TURN,
            parts=(turn.conversation_id, str(turn.turn_index), turn.role.value),
        )
        self._repository.upsert([turn_to_vector_record(record_id, turn)])
        return record_id

    def list_recent(
        self,
        *,
        conversation_id: str,
        limit: int,
    ) -> tuple[ConversationTurnRecord, ...]:
        if limit < 1:
            raise ValueError("Recent turn limit must be at least 1.")
        stored_records = self._repository.list(
            where={"conversation_id": conversation_id}
        )
        turns = sorted(
            (turn_from_stored_record(record) for record in stored_records),
            key=lambda turn: turn.turn_index,
        )
        return tuple(turns[-limit:])


class ConversationSummaryRepository:
    def __init__(self, *, client: Any) -> None:
        self._repository = _build_repository(
            client,
            ChromaCollection.CONVERSATION_SUMMARIES,
        )

    def upsert(self, summary: ConversationSummaryWrite) -> str:
        record_id = make_record_id(
            collection=ChromaCollection.CONVERSATION_SUMMARIES,
            record_kind=VectorRecordKind.CONVERSATION_SUMMARY,
            parts=(summary.conversation_id, str(summary.covered_turn_count)),
        )
        self._repository.upsert([summary_to_vector_record(record_id, summary)])
        return record_id

    def get_latest(
        self,
        *,
        conversation_id: str,
    ) -> ConversationSummaryRecord | None:
        stored_records = self._repository.list(
            where={"conversation_id": conversation_id}
        )
        if not stored_records:
            return None
        summaries = sorted(
            (summary_from_stored_record(record) for record in stored_records),
            key=lambda summary: summary.covered_turn_count,
        )
        return summaries[-1]


class ProjectMemoryRepository:
    def __init__(self, *, client: Any) -> None:
        self._repository = _build_repository(client, ChromaCollection.PROJECT_MEMORY)

    def upsert(self, memory: ProjectMemoryWrite) -> str:
        record_id = make_record_id(
            collection=ChromaCollection.PROJECT_MEMORY,
            record_kind=VectorRecordKind.PROJECT_MEMORY,
            parts=(
                memory.project_id,
                memory.memory_kind.value,
                memory.source,
                memory.content,
            ),
        )
        self._repository.upsert([project_memory_to_vector_record(record_id, memory)])
        return record_id

    def list(
        self,
        *,
        project_id: str,
        memory_kind: ProjectMemoryKind | None = None,
    ) -> tuple[ProjectMemoryRecord, ...]:
        where: dict[str, object] = {"project_id": project_id}
        if memory_kind is not None:
            where = {
                "$and": [
                    {"project_id": project_id},
                    {"memory_kind": memory_kind.value},
                ]
            }
        memories = [
            project_memory_from_stored_record(record)
            for record in self._repository.list(where)
        ]
        return tuple(sorted(memories, key=lambda memory: memory.created_at))


def _build_repository(client: Any, collection: ChromaCollection) -> ChromaRepository:
    return ChromaRepository(
        client=client,
        definition=get_collection_definition(collection),
    )
