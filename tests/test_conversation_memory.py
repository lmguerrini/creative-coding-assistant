import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from creative_coding_assistant.contracts import AssistantMode, CreativeCodingDomain
from creative_coding_assistant.memory import (
    ConversationRole,
    ConversationSummaryRepository,
    ConversationSummaryWrite,
    ConversationTurnRepository,
    ConversationTurnWrite,
    ProjectMemoryKind,
    ProjectMemoryRepository,
    ProjectMemoryWrite,
)
from creative_coding_assistant.vectorstore import create_chroma_client


class ConversationMemoryTests(unittest.TestCase):
    def test_turn_repository_lists_recent_turns_in_chronological_order(self) -> None:
        with _memory_client() as client:
            repository = ConversationTurnRepository(client=client)
            for turn_index in (2, 0, 1):
                repository.upsert(
                    _turn(
                        turn_index=turn_index,
                        role=ConversationRole.USER,
                        content=f"Turn {turn_index}",
                    )
                )

            recent_turns = repository.list_recent(
                conversation_id="conversation-1",
                limit=2,
            )

            self.assertEqual([turn.turn_index for turn in recent_turns], [1, 2])
            self.assertEqual(
                [turn.content for turn in recent_turns],
                ["Turn 1", "Turn 2"],
            )

    def test_summary_repository_returns_latest_summary(self) -> None:
        with _memory_client() as client:
            repository = ConversationSummaryRepository(client=client)
            repository.upsert(_summary(covered_turn_count=2, content="Early summary"))
            repository.upsert(_summary(covered_turn_count=5, content="Latest summary"))

            latest = repository.get_latest(conversation_id="conversation-1")

            self.assertIsNotNone(latest)
            self.assertEqual(latest.covered_turn_count, 5)
            self.assertEqual(latest.content, "Latest summary")

    def test_project_memory_repository_filters_by_kind(self) -> None:
        with _memory_client() as client:
            repository = ProjectMemoryRepository(client=client)
            repository.upsert(
                _project_memory(
                    memory_kind=ProjectMemoryKind.GOAL,
                    content="Build reactive particle studies.",
                    offset_minutes=0,
                )
            )
            repository.upsert(
                _project_memory(
                    memory_kind=ProjectMemoryKind.STYLE,
                    content="Prefer restrained palettes.",
                    offset_minutes=1,
                )
            )
            repository.upsert(
                _project_memory(
                    project_id="project-2",
                    memory_kind=ProjectMemoryKind.STYLE,
                    content="Use neon colors.",
                    offset_minutes=2,
                )
            )

            project_memories = repository.list(project_id="project-1")
            style_memories = repository.list(
                project_id="project-1",
                memory_kind=ProjectMemoryKind.STYLE,
            )

            self.assertEqual(len(project_memories), 2)
            self.assertEqual(project_memories[0].memory_kind, ProjectMemoryKind.GOAL)
            self.assertEqual(len(style_memories), 1)
            self.assertEqual(style_memories[0].content, "Prefer restrained palettes.")

    def test_memory_records_require_timezone_aware_timestamps(self) -> None:
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            ConversationTurnWrite(
                conversation_id="conversation-1",
                turn_index=0,
                role=ConversationRole.USER,
                content="Hello",
                created_at=datetime(2026, 1, 1, 12, 0),
                embedding=[0.1, 0.2, 0.3],
            )


def _turn(
    *,
    turn_index: int,
    role: ConversationRole,
    content: str,
) -> ConversationTurnWrite:
    return ConversationTurnWrite(
        conversation_id="conversation-1",
        turn_index=turn_index,
        role=role,
        content=content,
        created_at=_time(turn_index),
        project_id="project-1",
        domain=CreativeCodingDomain.THREE_JS,
        mode=AssistantMode.GENERATE,
        embedding=[0.1 + turn_index, 0.2, 0.3],
    )


def _summary(*, covered_turn_count: int, content: str) -> ConversationSummaryWrite:
    return ConversationSummaryWrite(
        conversation_id="conversation-1",
        covered_turn_count=covered_turn_count,
        content=content,
        created_at=_time(covered_turn_count),
        project_id="project-1",
        domain=CreativeCodingDomain.THREE_JS,
        embedding=[0.4 + covered_turn_count, 0.5, 0.6],
    )


def _project_memory(
    *,
    memory_kind: ProjectMemoryKind,
    content: str,
    offset_minutes: int,
    project_id: str = "project-1",
) -> ProjectMemoryWrite:
    return ProjectMemoryWrite(
        project_id=project_id,
        memory_kind=memory_kind,
        content=content,
        created_at=_time(offset_minutes),
        domain=CreativeCodingDomain.P5_JS,
        source="user",
        embedding=[0.7 + offset_minutes, 0.8, 0.9],
    )


def _time(offset_minutes: int) -> datetime:
    return datetime(2026, 1, 1, 12, 0, tzinfo=UTC) + timedelta(
        minutes=offset_minutes
    )


class _memory_client:
    def __enter__(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        return create_chroma_client(path=Path(self._temp_dir.name) / "chroma")

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
