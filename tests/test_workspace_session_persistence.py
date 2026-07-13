import io
import json
import tempfile
import unittest
from pathlib import Path

from creative_coding_assistant.api import WorkspaceSessionApplication
from creative_coding_assistant.workspace import (
    DEFAULT_LOCAL_SESSION_ID,
    DEFAULT_LOCAL_USER_ID,
    SQLiteWorkspaceSessionRepository,
    WorkspaceSessionPersistenceService,
    WorkspaceSessionRecord,
)


class WorkspaceSessionPersistenceTests(unittest.TestCase):
    def test_record_accepts_frontend_session_shape(self) -> None:
        record = WorkspaceSessionRecord.model_validate(_session_payload())

        self.assertEqual(record.user_id, DEFAULT_LOCAL_USER_ID)
        self.assertEqual(record.session_id, DEFAULT_LOCAL_SESSION_ID)
        self.assertEqual(record.active_inspector_tab, "Code")
        self.assertTrue(record.preview_open)
        self.assertEqual(record.layout.inspector_width, 440)
        self.assertEqual(record.layout.density, "compact")
        self.assertEqual(record.preferences.theme, "codex")
        self.assertFalse(record.preferences.auto_open_preview)
        self.assertEqual(record.messages[0].content, "Keep this chat.")
        self.assertEqual(record.artifacts[0].id, "source-sketch")
        self.assertEqual(record.artifacts[0].content, "function setup() {}")

    def test_record_accepts_current_frontend_v4_session_shape(self) -> None:
        record = WorkspaceSessionRecord.model_validate(
            _session_payload(
                schema_version=4,
                active_inspector_tab="Preview",
                theme="blueprint",
                preview_height=520,
            )
        )

        self.assertEqual(record.schema_version, 4)
        self.assertEqual(record.active_inspector_tab, "Preview")
        self.assertEqual(record.preferences.theme, "blueprint")
        self.assertEqual(record.layout.preview_height, 520)

    def test_record_accepts_codex_white_and_typography_preferences(self) -> None:
        payload = _session_payload(theme="codex_white")
        preferences = payload["preferences"]
        assert isinstance(preferences, dict)
        preferences.update(
            {
                "headingFontSize": "large",
                "labelFontSize": "small",
            }
        )

        record = WorkspaceSessionRecord.model_validate(payload)

        self.assertEqual(record.preferences.theme, "codex_white")
        self.assertEqual(record.preferences.heading_font_size, "large")
        self.assertEqual(record.preferences.label_font_size, "small")

    def test_record_preserves_terminal_product_outcome(self) -> None:
        record = WorkspaceSessionRecord.model_validate(
            _session_payload(product_outcome=True)
        )

        assert record.workflow is not None
        assert record.workflow.product_outcome is not None
        self.assertEqual(record.workflow.product_outcome.product_outcome, "PARTIAL")
        self.assertEqual(record.workflow.product_outcome.preview_status, "UNAVAILABLE")

    def test_sqlite_repository_round_trips_session_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = SQLiteWorkspaceSessionRepository(
                Path(temp_dir) / "workspace.sqlite3"
            )
            saved = repository.upsert(
                WorkspaceSessionRecord.model_validate(_session_payload())
            )

            restored = repository.get(
                user_id=DEFAULT_LOCAL_USER_ID,
                session_id=DEFAULT_LOCAL_SESSION_ID,
            )

        self.assertIsNotNone(saved.created_at)
        self.assertIsNotNone(saved.updated_at)
        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertEqual(restored.title, "Persisted sketch session")
        self.assertEqual(restored.messages[-1].role, "assistant")
        self.assertEqual(restored.preview.artifact_name, "preview-request.json")

    def test_wsgi_endpoint_saves_and_restores_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = _app_for_temp_db(Path(temp_dir))
            save_status: dict[str, object] = {}
            payload = json.dumps(
                _session_payload(
                    schema_version=4,
                    active_inspector_tab="Preview",
                    theme="blueprint",
                    preview_height=520,
                    product_outcome=True,
                )
            ).encode("utf-8")

            save_body = b"".join(
                app(
                    {
                        "PATH_INFO": "/api/workspace/session",
                        "REQUEST_METHOD": "POST",
                        "CONTENT_LENGTH": str(len(payload)),
                        "wsgi.input": io.BytesIO(payload),
                    },
                    _capture_start_response(save_status),
                )
            )
            get_status: dict[str, object] = {}
            get_body = b"".join(
                app(
                    {
                        "PATH_INFO": "/api/workspace/session",
                        "QUERY_STRING": (
                            "userId=local-user&sessionId=local-nextjs-session"
                        ),
                        "REQUEST_METHOD": "GET",
                    },
                    _capture_start_response(get_status),
                )
            )

        self.assertEqual(save_status["status"], "200 OK")
        self.assertEqual(json.loads(save_body)["schemaVersion"], 4)
        self.assertEqual(json.loads(save_body)["activeInspectorTab"], "Preview")
        self.assertEqual(get_status["status"], "200 OK")
        restored = json.loads(get_body)
        self.assertEqual(restored["title"], "Persisted sketch session")
        self.assertEqual(restored["layout"]["previewHeight"], 520)
        self.assertEqual(restored["preferences"]["theme"], "blueprint")
        self.assertEqual(restored["messages"][0]["content"], "Keep this chat.")
        self.assertEqual(
            restored["workflow"]["productOutcome"]["product_outcome"],
            "PARTIAL",
        )

    def test_wsgi_endpoint_updates_existing_session_with_put(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = _app_for_temp_db(Path(temp_dir))
            create_status: dict[str, object] = {}
            update_status: dict[str, object] = {}
            get_status: dict[str, object] = {}
            create_payload = json.dumps(_session_payload()).encode("utf-8")
            update_payload = json.dumps(
                _session_payload(
                    schema_version=4,
                    active_inspector_tab="Preview",
                    theme="blueprint",
                    preview_height=520,
                )
                | {"title": "Updated sketch session"}
            ).encode("utf-8")

            b"".join(
                app(
                    {
                        "PATH_INFO": "/api/workspace/session",
                        "REQUEST_METHOD": "POST",
                        "CONTENT_LENGTH": str(len(create_payload)),
                        "wsgi.input": io.BytesIO(create_payload),
                    },
                    _capture_start_response(create_status),
                )
            )
            update_body = b"".join(
                app(
                    {
                        "PATH_INFO": "/api/workspace/session",
                        "REQUEST_METHOD": "PUT",
                        "CONTENT_LENGTH": str(len(update_payload)),
                        "wsgi.input": io.BytesIO(update_payload),
                    },
                    _capture_start_response(update_status),
                )
            )
            get_body = b"".join(
                app(
                    {
                        "PATH_INFO": "/api/workspace/session",
                        "QUERY_STRING": (
                            "userId=local-user&sessionId=local-nextjs-session"
                        ),
                        "REQUEST_METHOD": "GET",
                    },
                    _capture_start_response(get_status),
                )
            )

        updated = json.loads(update_body)
        restored = json.loads(get_body)
        self.assertEqual(create_status["status"], "200 OK")
        self.assertEqual(update_status["status"], "200 OK")
        self.assertEqual(get_status["status"], "200 OK")
        self.assertEqual(updated["title"], "Updated sketch session")
        self.assertEqual(updated["schemaVersion"], 4)
        self.assertEqual(restored["title"], "Updated sketch session")
        self.assertEqual(restored["layout"]["previewHeight"], 520)
        self.assertEqual(restored["preferences"]["theme"], "blueprint")

    def test_wsgi_endpoint_returns_404_for_missing_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = _app_for_temp_db(Path(temp_dir))
            status_headers: dict[str, object] = {}

            body = b"".join(
                app(
                    {
                        "PATH_INFO": "/api/workspace/session",
                        "REQUEST_METHOD": "GET",
                    },
                    _capture_start_response(status_headers),
                )
            )

        self.assertEqual(status_headers["status"], "404 Not Found")
        self.assertEqual(json.loads(body)["error"], "session_not_found")

    def test_wsgi_endpoint_returns_empty_result_for_expected_first_run_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = _app_for_temp_db(Path(temp_dir))
            status_headers: dict[str, object] = {}

            body = b"".join(
                app(
                    {
                        "PATH_INFO": "/api/workspace/session",
                        "QUERY_STRING": "missingSession=empty",
                        "REQUEST_METHOD": "GET",
                    },
                    _capture_start_response(status_headers),
                )
            )

        self.assertEqual(status_headers["status"], "204 No Content")
        self.assertEqual(body, b"")

    def test_wsgi_endpoint_rejects_invalid_session_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = _app_for_temp_db(Path(temp_dir))
            status_headers: dict[str, object] = {}
            payload = json.dumps({"sessionId": ""}).encode("utf-8")

            body = b"".join(
                app(
                    {
                        "PATH_INFO": "/api/workspace/session",
                        "REQUEST_METHOD": "POST",
                        "CONTENT_LENGTH": str(len(payload)),
                        "wsgi.input": io.BytesIO(payload),
                    },
                    _capture_start_response(status_headers),
                )
            )

        self.assertEqual(status_headers["status"], "400 Bad Request")
        self.assertEqual(json.loads(body)["error"], "invalid_session")


def _session_payload(
    *,
    schema_version: int = 3,
    active_inspector_tab: str = "Code",
    theme: str = "codex",
    preview_height: int = 240,
    product_outcome: bool = False,
) -> dict[str, object]:
    return {
        "schemaVersion": schema_version,
        "userId": DEFAULT_LOCAL_USER_ID,
        "sessionId": DEFAULT_LOCAL_SESSION_ID,
        "projectId": "local-nextjs-workspace",
        "title": "Persisted sketch session",
        "activeArtifactId": "source-sketch",
        "activeInspectorTab": active_inspector_tab,
        "previewOpen": True,
        "previewArtifactId": "preview-manifest",
        "layout": {
            "density": "compact",
            "inspectorCollapsed": False,
            "inspectorWidth": 440,
            "previewHeight": preview_height,
        },
        "preferences": {
            "theme": theme,
            "autoOpenPreview": False,
            "showDebugPanels": True,
        },
        "workspace": {
            "name": "Persisted sketch session",
            "focus": "Audio-reactive projection field",
        },
        "messages": [
            {
                "role": "user",
                "time": "10:15",
                "content": "Keep this chat.",
            },
            {
                "role": "assistant",
                "time": "10:16",
                "content": "Restored workspace state.",
            },
        ],
        "workflow": {
            "status": "Complete",
            "currentNode": "finalization",
            "currentStep": "Finalization",
            "steps": [
                {
                    "nodeId": "intake",
                    "displayLabel": "Intake",
                    "state": "complete",
                    "detail": "Request received.",
                }
            ],
            **(
                {
                    "productOutcome": {
                        "orchestration_status": "COMPLETED",
                        "provider_status": "COMPLETED",
                        "generation_status": "COMPLETED",
                        "deliverable_status": "USABLE",
                        "artifact_extraction_status": "EXTRACTED",
                        "artifact_runnability": "UNSUPPORTED",
                        "preview_status": "UNAVAILABLE",
                        "runtime_health": "NOT_AVAILABLE",
                        "product_outcome": "PARTIAL",
                        "summary": "A usable artifact was produced, but live preview is unavailable.",
                        "recovery_action": "Open Code to use the artifact.",
                    }
                }
                if product_outcome
                else {}
            ),
        },
        "artifacts": [
            {
                "id": "source-sketch",
                "title": "webgpu-particle-field.ts",
                "type": "code",
                "language": "TypeScript",
                "status": "Draft",
                "summary": "Primary sketch artifact.",
                "content": "function setup() {}",
                "actions": ["Open", "Preview"],
            }
        ],
        "preview": {
            "available": True,
            "active": True,
            "collapsed": False,
            "state": "ready",
            "title": "Preview available",
            "targetId": "browser_sandbox",
            "target": "Browser sandbox",
            "status": "Preview open",
            "artifactName": "preview-request.json",
            "sourceArtifactId": "source-sketch",
            "sourceArtifactName": "webgpu-particle-field.ts",
            "outputArtifactName": "preview-request.json",
            "summary": "Preview state.",
            "renderer": "preview.noop",
            "trigger": "Preview preview-request.json",
            "version": "v1",
        },
        "snapshot": {
            "workspace": {
                "name": "Persisted sketch session",
                "focus": "Audio-reactive projection field",
            }
        },
    }


def _app_for_temp_db(temp_dir: Path) -> WorkspaceSessionApplication:
    repository = SQLiteWorkspaceSessionRepository(temp_dir / "workspace.sqlite3")
    return WorkspaceSessionApplication(
        service=WorkspaceSessionPersistenceService(repository=repository)
    )


def _capture_start_response(target: dict[str, object]):
    def start_response(
        status: str,
        headers: list[tuple[str, str]],
        exc_info: object | None = None,
    ) -> None:
        del headers
        del exc_info
        target["status"] = status

    return start_response


if __name__ == "__main__":
    unittest.main()
