"""Multi-session persistence contracts for the V9.7 session sidebar."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from creative_coding_assistant.api.workspace_sessions import WorkspaceSessionApplication
from creative_coding_assistant.workspace import WorkspaceSessionPersistenceService
from creative_coding_assistant.workspace.contracts import WorkspaceSessionRecord
from creative_coding_assistant.workspace.repository import SQLiteWorkspaceSessionRepository


class V97SessionLifecycleTests(unittest.TestCase):
    def test_repository_lists_newest_sessions_and_keeps_profiles_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = SQLiteWorkspaceSessionRepository(Path(directory) / "sessions.sqlite3")
            service = WorkspaceSessionPersistenceService(repository)
            first = service.save_session(
                WorkspaceSessionRecord(
                    userId="profile-a",
                    sessionId="first",
                    projectId="project-a",
                    title="First",
                )
            )
            second = service.save_session(
                WorkspaceSessionRecord(
                    userId="profile-a",
                    sessionId="second",
                    projectId="project-a",
                    title="Second",
                )
            )
            service.save_session(
                WorkspaceSessionRecord(
                    userId="profile-b",
                    sessionId="private",
                    projectId="project-b",
                    title="Private",
                )
            )

            listed = service.list_sessions(user_id="profile-a")

            self.assertEqual({record.session_id for record in listed}, {"first", "second"})
            self.assertNotIn("private", {record.session_id for record in listed})
            self.assertTrue(service.delete_session(user_id="profile-a", session_id="first"))
            self.assertIsNone(service.get_session(user_id="profile-a", session_id="first"))
            self.assertIsNotNone(service.get_session(user_id="profile-a", session_id="second"))
            self.assertIsNotNone(service.get_session(user_id="profile-b", session_id="private"))
            self.assertIsNotNone(first.updated_at)
            self.assertIsNotNone(second.updated_at)

    def test_http_list_and_delete_require_explicit_session_identity(self) -> None:
        service = _SessionService()
        app = WorkspaceSessionApplication(service=service)
        list_headers: dict[str, object] = {}
        delete_headers: dict[str, object] = {}

        list_body = b"".join(
            app(
                {
                    "PATH_INFO": "/api/workspace/session",
                    "QUERY_STRING": "list=true&userId=profile-a",
                    "REQUEST_METHOD": "GET",
                },
                _capture_response(list_headers),
            )
        )
        delete_body = b"".join(
            app(
                {
                    "PATH_INFO": "/api/workspace/session",
                    "QUERY_STRING": "userId=profile-a&sessionId=one",
                    "REQUEST_METHOD": "DELETE",
                },
                _capture_response(delete_headers),
            )
        )

        self.assertEqual(list_headers["status"], "200 OK")
        self.assertEqual(json.loads(list_body)["sessions"][0]["sessionId"], "one")
        self.assertEqual(delete_headers["status"], "204 No Content")
        self.assertEqual(delete_body, b"")
        self.assertEqual(service.deleted, [("profile-a", "one")])


class _SessionService:
    def __init__(self) -> None:
        self.deleted: list[tuple[str, str]] = []

    def list_sessions(self, *, user_id: str):
        if user_id != "profile-a":
            return ()
        return (
            WorkspaceSessionRecord(
                userId="profile-a",
                sessionId="one",
                projectId="project-a",
                title="One",
            ),
        )

    def delete_session(self, *, user_id: str, session_id: str) -> bool:
        self.deleted.append((user_id, session_id))
        return (user_id, session_id) == ("profile-a", "one")


def _capture_response(target: dict[str, object]):
    def start_response(status, headers, exc_info=None):
        del exc_info
        target["status"] = status
        target["headers"] = headers

    return start_response
