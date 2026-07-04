import json
import unittest

from creative_coding_assistant.api import (
    AssistantStreamingApplication,
    HealthCheckApplication,
    build_cors_policy_report,
    build_dependency_health_report,
    build_deployment_readiness_checklist,
    create_backend_dev_app,
    validate_production_configuration,
)
from creative_coding_assistant.core import Settings


class V77ProductionDeploymentFoundationTests(unittest.TestCase):
    def test_local_cors_preserves_wildcard_for_smoke_compatibility(self) -> None:
        settings = Settings(_env_file=None, environment="local")
        app = AssistantStreamingApplication(settings_factory=lambda: settings)
        status_headers: dict[str, object] = {}

        body = b"".join(
            app(
                {
                    "PATH_INFO": "/api/assistant/stream",
                    "REQUEST_METHOD": "OPTIONS",
                    "HTTP_ORIGIN": "http://localhost:3000",
                },
                _capture_start_response(status_headers),
            )
        )

        headers = dict(status_headers["headers"])
        self.assertEqual(status_headers["status"], "204 No Content")
        self.assertEqual(body, b"")
        self.assertEqual(headers["Access-Control-Allow-Origin"], "*")
        self.assertEqual(build_cors_policy_report(settings).status, "ready")

    def test_production_cors_echoes_explicit_allowed_origin(self) -> None:
        settings = Settings(
            _env_file=None,
            environment="production",
            cors_allowed_origins=("https://app.example",),
            openai_api_key="sk-test",
        )
        app = HealthCheckApplication(settings_factory=lambda: settings)
        status_headers: dict[str, object] = {}

        body = b"".join(
            app(
                {
                    "PATH_INFO": "/api/health/ready",
                    "REQUEST_METHOD": "GET",
                    "HTTP_ORIGIN": "https://app.example",
                },
                _capture_start_response(status_headers),
            )
        )

        headers = dict(status_headers["headers"])
        response = json.loads(body)
        self.assertEqual(status_headers["status"], "200 OK")
        self.assertEqual(response["status"], "ready")
        self.assertEqual(headers["Access-Control-Allow-Origin"], "https://app.example")
        self.assertEqual(headers["Vary"], "Origin")

    def test_production_cors_omits_disallowed_origin(self) -> None:
        settings = Settings(
            _env_file=None,
            environment="production",
            cors_allowed_origins=("https://app.example",),
            openai_api_key="sk-test",
        )
        app = HealthCheckApplication(settings_factory=lambda: settings)
        status_headers: dict[str, object] = {}

        b"".join(
            app(
                {
                    "PATH_INFO": "/api/health/live",
                    "REQUEST_METHOD": "GET",
                    "HTTP_ORIGIN": "https://other.example",
                },
                _capture_start_response(status_headers),
            )
        )

        headers = dict(status_headers["headers"])
        self.assertEqual(status_headers["status"], "200 OK")
        self.assertNotIn("Access-Control-Allow-Origin", headers)

    def test_production_wildcard_cors_is_guarded_and_not_granted(self) -> None:
        settings = Settings(
            _env_file=None,
            environment="production",
            cors_allowed_origins=("*",),
            openai_api_key="sk-test",
        )
        app = HealthCheckApplication(settings_factory=lambda: settings)
        status_headers: dict[str, object] = {}

        body = b"".join(
            app(
                {
                    "PATH_INFO": "/api/health/ready",
                    "REQUEST_METHOD": "GET",
                    "HTTP_ORIGIN": "https://app.example",
                },
                _capture_start_response(status_headers),
            )
        )

        headers = dict(status_headers["headers"])
        response = json.loads(body)
        config_report = validate_production_configuration(settings)
        self.assertEqual(status_headers["status"], "503 Service Unavailable")
        self.assertEqual(response["checks"]["configuration"], "guarded")
        self.assertNotIn("Access-Control-Allow-Origin", headers)
        self.assertIn(
            "CCA_CORS_ALLOWED_ORIGINS must not include '*' in production.",
            config_report.warnings,
        )

    def test_backend_not_found_uses_production_cors_policy(self) -> None:
        settings = Settings(
            _env_file=None,
            environment="production",
            cors_allowed_origins=("https://app.example",),
            openai_api_key="sk-test",
        )
        app = create_backend_dev_app(settings_factory=lambda: settings)
        status_headers: dict[str, object] = {}

        body = b"".join(
            app(
                {
                    "PATH_INFO": "/api/missing",
                    "REQUEST_METHOD": "GET",
                    "HTTP_ORIGIN": "https://app.example",
                },
                _capture_start_response(status_headers),
            )
        )

        headers = dict(status_headers["headers"])
        response = json.loads(body)
        self.assertEqual(status_headers["status"], "404 Not Found")
        self.assertEqual(headers["Access-Control-Allow-Origin"], "https://app.example")
        self.assertEqual(response["error"], "not_found")

    def test_deployment_readiness_checklist_covers_v77_without_behavior_changes(self):
        settings = Settings(
            _env_file=None,
            environment="production",
            cors_allowed_origins=("https://app.example",),
            openai_api_key="sk-test",
        )
        checklist = build_deployment_readiness_checklist(
            configuration=validate_production_configuration(settings),
            dependency_health=build_dependency_health_report(chromadb_version="0.6.3"),
            cors_policy=build_cors_policy_report(settings),
        )

        self.assertEqual(checklist.version, "v7.7")
        self.assertEqual(checklist.deployment_item_count, 12)
        self.assertTrue(checklist.deployment_coverage_complete)
        self.assertFalse(checklist.creative_behavior_changed)
        self.assertFalse(checklist.provider_routing_changed)
        self.assertEqual(checklist.guarded_items, ())

    def test_production_wsgi_entrypoint_is_callable(self) -> None:
        from creative_coding_assistant.api.wsgi import application

        self.assertTrue(callable(application))


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


if __name__ == "__main__":
    unittest.main()
