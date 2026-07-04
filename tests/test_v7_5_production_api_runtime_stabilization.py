import io
import json
import unittest

from creative_coding_assistant.api import (
    CHROMA_REQUIREMENT,
    STREAM_CONTRACT_VERSION,
    WORKSPACE_SESSION_CONTRACT_VERSION,
    AssistantStreamingApplication,
    BackendDevApplication,
    DependencyHealthReport,
    HealthCheckApplication,
    WorkspaceSessionApplication,
    build_api_telemetry_event,
    build_dependency_health_report,
    build_health_payload,
    build_release_checklist,
    create_backend_dev_app,
    validate_production_configuration,
)
from creative_coding_assistant.api.contracts import (
    ERROR_CONTRACT_VERSION,
    HEALTH_CONTRACT_VERSION,
    STREAM_CONTRACT_HEADER,
    WORKSPACE_SESSION_CONTRACT_HEADER,
)
from creative_coding_assistant.api.dev_server import run_backend_dev_server
from creative_coding_assistant.core import Settings


class V75ProductionApiRuntimeStabilizationTests(unittest.TestCase):
    def test_backend_route_manifest_includes_stable_api_and_health_paths(self) -> None:
        app = create_backend_dev_app()

        self.assertEqual(
            app.available_paths,
            (
                "/api/assistant/stream",
                "/api/workspace/session",
                "/api/health",
                "/api/health/live",
                "/api/health/ready",
            ),
        )

    def test_streaming_endpoint_exposes_contract_version_and_error_shape(self) -> None:
        app = AssistantStreamingApplication(service_factory=_failing_stream_factory)
        status_headers: dict[str, object] = {}
        payload = json.dumps({"query": "Generate a field."}).encode("utf-8")

        body = b"".join(
            app(
                {
                    "PATH_INFO": "/api/assistant/stream",
                    "REQUEST_METHOD": "POST",
                    "CONTENT_LENGTH": str(len(payload)),
                    "wsgi.input": io.BytesIO(payload),
                },
                _capture_start_response(status_headers),
            )
        )

        event = json.loads(body.decode("utf-8").splitlines()[0])
        self.assertEqual(status_headers["status"], "200 OK")
        self.assertIn(
            (STREAM_CONTRACT_HEADER, STREAM_CONTRACT_VERSION),
            status_headers["headers"],
        )
        self.assertEqual(event["event_type"], "error")
        self.assertEqual(event["payload"]["code"], "assistant_stream_failed")
        self.assertEqual(
            event["payload"]["contract_version"],
            STREAM_CONTRACT_VERSION,
        )

    def test_invalid_stream_payload_uses_versioned_error_contract(self) -> None:
        app = AssistantStreamingApplication(service=_NoopStreamService())
        status_headers: dict[str, object] = {}
        payload = json.dumps({"query": ""}).encode("utf-8")

        body = b"".join(
            app(
                {
                    "PATH_INFO": "/api/assistant/stream",
                    "REQUEST_METHOD": "POST",
                    "CONTENT_LENGTH": str(len(payload)),
                    "wsgi.input": io.BytesIO(payload),
                },
                _capture_start_response(status_headers),
            )
        )

        response = json.loads(body)
        self.assertEqual(status_headers["status"], "400 Bad Request")
        self.assertEqual(response["error"], "invalid_request")
        self.assertEqual(response["contractVersion"], ERROR_CONTRACT_VERSION)
        self.assertIn("requestId", response)

    def test_workspace_missing_and_invalid_payload_keep_backward_compatibility(self):
        app = WorkspaceSessionApplication(service=_EmptyWorkspaceService())
        missing_status: dict[str, object] = {}
        invalid_status: dict[str, object] = {}
        payload = json.dumps({"sessionId": ""}).encode("utf-8")

        missing_body = b"".join(
            app(
                {
                    "PATH_INFO": "/api/workspace/session",
                    "REQUEST_METHOD": "GET",
                },
                _capture_start_response(missing_status),
            )
        )
        invalid_body = b"".join(
            app(
                {
                    "PATH_INFO": "/api/workspace/session",
                    "REQUEST_METHOD": "POST",
                    "CONTENT_LENGTH": str(len(payload)),
                    "wsgi.input": io.BytesIO(payload),
                },
                _capture_start_response(invalid_status),
            )
        )

        missing = json.loads(missing_body)
        invalid = json.loads(invalid_body)
        self.assertEqual(missing_status["status"], "404 Not Found")
        self.assertEqual(missing["error"], "session_not_found")
        self.assertEqual(missing["contractVersion"], ERROR_CONTRACT_VERSION)
        self.assertIn(
            (WORKSPACE_SESSION_CONTRACT_HEADER, WORKSPACE_SESSION_CONTRACT_VERSION),
            missing_status["headers"],
        )
        self.assertEqual(invalid_status["status"], "400 Bad Request")
        self.assertEqual(invalid["error"], "invalid_session")

    def test_workspace_service_failure_recovers_as_503(self) -> None:
        app = WorkspaceSessionApplication(service=_FailingWorkspaceService())
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

        response = json.loads(body)
        self.assertEqual(status_headers["status"], "503 Service Unavailable")
        self.assertEqual(response["error"], "workspace_session_unavailable")
        self.assertTrue(response["recoverable"])

    def test_health_endpoints_report_readiness_and_contracts(self) -> None:
        settings = Settings(_env_file=None)
        app = HealthCheckApplication(settings_factory=lambda: settings)
        status_headers: dict[str, object] = {}

        body = b"".join(
            app(
                {
                    "PATH_INFO": "/api/health/ready",
                    "REQUEST_METHOD": "GET",
                },
                _capture_start_response(status_headers),
            )
        )

        response = json.loads(body)
        self.assertEqual(status_headers["status"], "200 OK")
        self.assertEqual(response["contractVersion"], HEALTH_CONTRACT_VERSION)
        self.assertEqual(response["checks"]["assistantStream"], STREAM_CONTRACT_VERSION)
        self.assertEqual(
            build_health_payload(settings=settings)["checks"]["workspaceSession"],
            WORKSPACE_SESSION_CONTRACT_VERSION,
        )

    def test_production_configuration_validation_guards_missing_credentials(self):
        report = validate_production_configuration(
            Settings(_env_file=None, environment="production", openai_api_key=None)
        )

        self.assertEqual(report.status, "guarded")
        self.assertIn(
            "OPENAI_API_KEY or CCA_OPENAI_API_KEY is required in production.",
            report.warnings,
        )

    def test_chroma_dependency_health_uses_safe_pre_1_requirement(self) -> None:
        safe = build_dependency_health_report(chromadb_version="0.6.3")
        vulnerable = build_dependency_health_report(chromadb_version="1.5.9")

        self.assertEqual(CHROMA_REQUIREMENT, ">=0.6.3,<1.0.0")
        self.assertIsInstance(safe, DependencyHealthReport)
        self.assertEqual(safe.status, "ready")
        self.assertEqual(vulnerable.status, "guarded")
        self.assertIn("CVE-2026-45829", vulnerable.vulnerability_ids)

    def test_telemetry_event_and_release_checklist_cover_v75_contracts(self):
        event = build_api_telemetry_event(
            event_kind="error",
            route="/api/workspace/session",
            status_code=404,
            request_id="request-1",
            error_code="session_not_found",
            recoverable=True,
        )
        checklist = build_release_checklist(
            dependency_health=build_dependency_health_report(
                chromadb_version="0.6.3",
            )
        )

        self.assertEqual(event.contract_version, "api.v1")
        self.assertEqual(checklist.roadmap_item_count, 22)
        self.assertTrue(checklist.roadmap_coverage_complete)
        self.assertTrue(checklist.remote_ci_verification_required)

    def test_dev_server_refuses_production_without_explicit_override(self) -> None:
        settings = Settings(_env_file=None, environment="production")

        with self.assertRaisesRegex(RuntimeError, "development server"):
            run_backend_dev_server(
                settings=settings,
                app=BackendDevApplication(mounts=()),
            )


def _capture_start_response(target: dict[str, object]):
    def start_response(
        status: str,
        headers: list[tuple[str, str]],
        exc_info: object | None = None,
    ) -> None:
        del exc_info
        target["status"] = status
        target["headers"] = headers

    return start_response


def _failing_stream_factory():
    raise RuntimeError("stream service unavailable")


class _NoopStreamService:
    def stream(self, request):
        del request
        return iter(())


class _EmptyWorkspaceService:
    def get_session(self, *, user_id: str, session_id: str):
        del user_id
        del session_id
        return None

    def save_session(self, record):
        return record


class _FailingWorkspaceService:
    def get_session(self, *, user_id: str, session_id: str):
        del user_id
        del session_id
        raise RuntimeError("database unavailable")

    def save_session(self, record):
        del record
        raise RuntimeError("database unavailable")


if __name__ == "__main__":
    unittest.main()
