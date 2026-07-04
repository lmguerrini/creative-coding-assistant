"""Health and readiness endpoints for the backend bridge."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from http import HTTPStatus
from typing import Any

from creative_coding_assistant.api.contracts import (
    HEALTH_CONTRACT_HEADER,
    HEALTH_CONTRACT_VERSION,
    STREAM_CONTRACT_VERSION,
    WORKSPACE_SESSION_CONTRACT_VERSION,
    StartResponse,
    error_response,
    json_response,
    request_id_from_environ,
)
from creative_coding_assistant.api.cors import resolve_cors_allow_origin
from creative_coding_assistant.api.production import (
    ProductionConfigurationReport,
    validate_production_configuration,
)
from creative_coding_assistant.core.config import Settings, load_settings

DEFAULT_HEALTH_PATH = "/api/health"
DEFAULT_LIVENESS_PATH = "/api/health/live"
DEFAULT_READINESS_PATH = "/api/health/ready"
HEALTH_METHODS = "GET, OPTIONS"


class HealthCheckApplication:
    """Small WSGI app that exposes liveness and readiness contracts."""

    def __init__(
        self,
        *,
        settings_factory: Callable[[], Settings] = load_settings,
        paths: tuple[str, str, str] = (
            DEFAULT_HEALTH_PATH,
            DEFAULT_LIVENESS_PATH,
            DEFAULT_READINESS_PATH,
        ),
    ) -> None:
        self._settings_factory = settings_factory
        self._paths = paths

    def __call__(
        self,
        environ: dict[str, Any],
        start_response: StartResponse,
    ) -> Iterable[bytes]:
        request_id = request_id_from_environ(environ)
        path = str(environ.get("PATH_INFO", ""))
        method = str(environ.get("REQUEST_METHOD", "GET")).upper()
        settings = self._settings_factory()
        allow_origin = resolve_cors_allow_origin(environ, settings=settings)

        if path not in self._paths:
            return error_response(
                start_response,
                HTTPStatus.NOT_FOUND,
                error="not_found",
                message="Health route was not found.",
                request_id=request_id,
                allow_methods=HEALTH_METHODS,
                allow_origin=allow_origin,
                details={"available_paths": list(self._paths)},
            )

        if method == "OPTIONS":
            from creative_coding_assistant.api.contracts import empty_response

            return empty_response(
                start_response,
                HTTPStatus.NO_CONTENT,
                request_id=request_id,
                allow_methods=HEALTH_METHODS,
                allow_origin=allow_origin,
            )

        if method != "GET":
            return error_response(
                start_response,
                HTTPStatus.METHOD_NOT_ALLOWED,
                error="method_not_allowed",
                message="Health endpoints accept GET and OPTIONS.",
                request_id=request_id,
                allow_methods=HEALTH_METHODS,
                allow_origin=allow_origin,
                details={"allowed_methods": ["GET", "OPTIONS"]},
                extra_headers=[("Allow", HEALTH_METHODS)],
            )

        config_report = validate_production_configuration(settings=settings)
        payload = build_health_payload(
            path=path,
            settings=settings,
            configuration=config_report,
        )
        status = (
            HTTPStatus.OK
            if path != DEFAULT_READINESS_PATH or config_report.status == "ready"
            else HTTPStatus.SERVICE_UNAVAILABLE
        )
        return json_response(
            start_response,
            status,
            payload,
            request_id=request_id,
            allow_methods=HEALTH_METHODS,
            allow_origin=allow_origin,
            extra_headers=[(HEALTH_CONTRACT_HEADER, HEALTH_CONTRACT_VERSION)],
        )


def create_health_check_app(
    *,
    settings_factory: Callable[[], Settings] = load_settings,
) -> HealthCheckApplication:
    """Create the WSGI health-check app."""

    return HealthCheckApplication(settings_factory=settings_factory)


def build_health_payload(
    *,
    path: str = DEFAULT_HEALTH_PATH,
    settings: Settings | None = None,
    configuration: ProductionConfigurationReport | None = None,
) -> dict[str, object]:
    """Build a deterministic health payload without touching external services."""

    resolved_settings = settings or load_settings()
    config_report = configuration or validate_production_configuration(
        settings=resolved_settings
    )
    readiness = "ready" if config_report.status == "ready" else "guarded"
    return {
        "status": "ok" if path != DEFAULT_READINESS_PATH else readiness,
        "service": resolved_settings.app_name,
        "environment": resolved_settings.environment,
        "contractVersion": HEALTH_CONTRACT_VERSION,
        "checks": {
            "liveness": "ok",
            "configuration": config_report.status,
            "assistantStream": STREAM_CONTRACT_VERSION,
            "workspaceSession": WORKSPACE_SESSION_CONTRACT_VERSION,
        },
    }
