"""SQLite repository for single-user workspace sessions."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from creative_coding_assistant.workspace.contracts import WorkspaceSessionRecord


class SQLiteWorkspaceSessionRepository:
    """Persist workspace session records in a small local SQLite database."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)

    def get(
        self,
        *,
        user_id: str,
        session_id: str,
    ) -> WorkspaceSessionRecord | None:
        self._initialize()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json
                FROM workspace_sessions
                WHERE user_id = ? AND session_id = ?
                """,
                (user_id, session_id),
            ).fetchone()

        if row is None:
            return None

        return WorkspaceSessionRecord.model_validate_json(str(row["payload_json"]))

    def upsert(self, record: WorkspaceSessionRecord) -> WorkspaceSessionRecord:
        self._initialize()
        existing = self.get(
            user_id=record.user_id,
            session_id=record.session_id,
        )
        stamped = record.with_timestamps(
            created_at=existing.created_at if existing is not None else None
        )
        payload_json = stamped.model_dump_json(by_alias=True)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO workspace_sessions (
                    user_id,
                    session_id,
                    project_id,
                    title,
                    payload_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, session_id) DO UPDATE SET
                    project_id = excluded.project_id,
                    title = excluded.title,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (
                    stamped.user_id,
                    stamped.session_id,
                    stamped.project_id,
                    stamped.title,
                    payload_json,
                    stamped.created_at.isoformat()
                    if stamped.created_at is not None
                    else "",
                    stamped.updated_at.isoformat()
                    if stamped.updated_at is not None
                    else "",
                ),
            )
        return stamped

    def list_for_user(self, *, user_id: str) -> tuple[WorkspaceSessionRecord, ...]:
        """List a profile's sessions newest first without crossing user boundaries."""

        self._initialize()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json
                FROM workspace_sessions
                WHERE user_id = ?
                ORDER BY updated_at DESC, session_id ASC
                """,
                (user_id,),
            ).fetchall()
        return tuple(
            WorkspaceSessionRecord.model_validate_json(str(row["payload_json"]))
            for row in rows
        )

    def delete(self, *, user_id: str, session_id: str) -> bool:
        """Delete one explicitly selected local session record."""

        self._initialize()
        with self._connect() as connection:
            result = connection.execute(
                """
                DELETE FROM workspace_sessions
                WHERE user_id = ? AND session_id = ?
                """,
                (user_id, session_id),
            )
        return result.rowcount > 0

    def _initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS workspace_sessions (
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, session_id)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_workspace_sessions_updated_at
                ON workspace_sessions(updated_at)
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        return connection
