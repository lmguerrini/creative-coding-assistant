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

    def test_schema_v3_preserves_nested_evaluation_benchmark_aliases(self) -> None:
        payload = _session_payload(schema_version=3)
        preferences = payload["preferences"]
        assert isinstance(preferences, dict)
        benchmark = _evaluation_benchmark_payload()
        preferences["evaluationHistory"] = [
            _evaluation_history_payload(benchmark=benchmark)
        ]

        record = WorkspaceSessionRecord.model_validate(payload)
        serialized = record.model_dump(mode="json", by_alias=True)
        restored = WorkspaceSessionRecord.model_validate_json(
            record.model_dump_json(by_alias=True)
        )

        history = record.preferences.evaluation_history
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].run_id, "eval-run-current-product-001")
        self.assertEqual(history[0].benchmark, benchmark)
        serialized_preferences = serialized["preferences"]
        assert isinstance(serialized_preferences, dict)
        serialized_history = serialized_preferences["evaluationHistory"]
        assert isinstance(serialized_history, list)
        self.assertEqual(serialized_history[0]["benchmark"], benchmark)
        self.assertEqual(
            restored.preferences.evaluation_history[0].benchmark,
            benchmark,
        )

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

    def test_sqlite_repository_round_trips_nested_evaluation_benchmark(self) -> None:
        payload = _session_payload(schema_version=3)
        preferences = payload["preferences"]
        assert isinstance(preferences, dict)
        benchmark = _evaluation_benchmark_payload()
        preferences["evaluationHistory"] = [
            _evaluation_history_payload(benchmark=benchmark)
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            repository = SQLiteWorkspaceSessionRepository(
                Path(temp_dir) / "workspace.sqlite3"
            )
            repository.upsert(WorkspaceSessionRecord.model_validate(payload))
            restored = repository.get(
                user_id=DEFAULT_LOCAL_USER_ID,
                session_id=DEFAULT_LOCAL_SESSION_ID,
            )

        self.assertIsNotNone(restored)
        assert restored is not None
        history = restored.preferences.evaluation_history
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].benchmark, benchmark)

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


def _evaluation_history_payload(
    *,
    benchmark: dict[str, object],
) -> dict[str, object]:
    return {
        "id": "evaluation-history-current-product-001",
        "runId": "eval-run-current-product-001",
        "datasetId": "current_product",
        "metrics": [
            "context_precision",
            "faithfulness",
            "answer_relevancy",
            "context_relevancy",
            "context_recall",
        ],
        "status": "completed",
        "detail": "Current-product retrieval benchmark completed.",
        "evaluatedAt": "2026-07-14T10:00:00Z",
        "resultRows": 7,
        "metricFailures": 0,
        "dryRun": False,
        "providerCallsAllowed": True,
        "benchmark": benchmark,
    }


def _evaluation_benchmark_payload() -> dict[str, object]:
    return {
        "schemaVersion": 3,
        "id": "benchmark-current-product-001",
        "datasetVersion": "current-product-retrieval.v1",
        "datasetFingerprint": "sha256:dataset",
        "promptVersion": "current-product-prompt.v1",
        "scope": "rag",
        "selectedCaseIds": ["framework_selection_hydra_p5"],
        "startedAt": "2026-07-14T09:58:00Z",
        "completedAt": "2026-07-14T10:00:00Z",
        "durationMs": 120000,
        "executionMode": "provider_assisted",
        "environmentStatus": "ready",
        "statusLabel": "Demo Ready",
        "measuredScore": 0.9,
        "targetThreshold": 0.85,
        "evidenceCompleteness": 1.0,
        "caseCoverage": 1.0,
        "executedCases": 7,
        "selectedCases": 7,
        "counts": {
            "pass": 7,
            "partial": 0,
            "fail": 0,
            "blocked": 0,
            "missing": 0,
            "notRun": 0,
        },
        "categoryResults": [
            {
                "category": "rag",
                "status": "pass",
                "score": 0.9,
            }
        ],
        "caseResults": [
            {
                "caseId": "framework_selection_hydra_p5",
                "title": "Hydra and p5.js framework selection",
                "status": "pass",
                "categories": ["rag"],
                "metrics": [
                    {
                        "id": "faithfulness",
                        "status": "pass",
                        "score": 0.91,
                    }
                ],
            }
        ],
        "recommendations": [],
        "missingMetricIds": [],
        "provider": "OpenAI",
        "model": "gpt-5-mini",
        "workflow": "multi_agent",
        "totalTokens": 8000,
        "estimatedCost": None,
        "currency": "USD",
        "ragas": {
            "state": "completed",
            "runId": "eval-run-current-product-001",
            "metricScores": {
                "context_precision": 0.92,
                "faithfulness": 0.91,
                "answer_relevancy": 0.89,
                "context_relevancy": 0.88,
                "context_recall": 0.9,
            },
            "caseRows": [
                {
                    "sampleId": "framework_selection_hydra_p5",
                    "sourceIds": ["hydra_docs", "p5_reference"],
                }
            ],
        },
        "scoreOrigin": "current_product",
        "benchmarkVersion": "current-product-retrieval.v1",
        "retrievalFingerprint": "sha256:retrieval",
        "promptFingerprint": "sha256:prompt",
        "generationFingerprint": "sha256:generation",
        "generationModel": "gpt-5-mini",
        "evaluator": "gpt-4o-mini",
        "embeddingModel": "text-embedding-3-small",
        "timestamp": "2026-07-14T10:00:00Z",
        "runId": "eval-run-current-product-001",
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
