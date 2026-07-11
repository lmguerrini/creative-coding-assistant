"""Service boundary for local workspace session persistence."""

from __future__ import annotations

from creative_coding_assistant.core import Settings, load_settings
from creative_coding_assistant.workspace.contracts import WorkspaceSessionRecord
from creative_coding_assistant.workspace.repository import (
    SQLiteWorkspaceSessionRepository,
)


class WorkspaceSessionPersistenceService:
    """Save and restore the default local workspace session."""

    def __init__(self, repository: SQLiteWorkspaceSessionRepository) -> None:
        self._repository = repository

    def get_session(
        self,
        *,
        user_id: str,
        session_id: str,
    ) -> WorkspaceSessionRecord | None:
        return self._repository.get(user_id=user_id, session_id=session_id)

    def save_session(self, record: WorkspaceSessionRecord) -> WorkspaceSessionRecord:
        return self._repository.upsert(record)

    def list_sessions(self, *, user_id: str) -> tuple[WorkspaceSessionRecord, ...]:
        return self._repository.list_for_user(user_id=user_id)

    def delete_session(self, *, user_id: str, session_id: str) -> bool:
        return self._repository.delete(user_id=user_id, session_id=session_id)


def build_workspace_session_persistence_service(
    settings: Settings | None = None,
) -> WorkspaceSessionPersistenceService:
    """Build the default SQLite-backed local workspace persistence service."""

    resolved_settings = settings or load_settings()
    return WorkspaceSessionPersistenceService(
        repository=SQLiteWorkspaceSessionRepository(
            resolved_settings.workspace_session_db_path
        )
    )
