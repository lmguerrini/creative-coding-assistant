"""Local WSGI dev entrypoint for browser-facing backend bridge apps."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any
from wsgiref.simple_server import make_server

from creative_coding_assistant.api.contracts import (
    StartResponse,
    error_response,
    request_id_from_environ,
)
from creative_coding_assistant.api.cors import resolve_cors_allow_origin
from creative_coding_assistant.api.health import (
    DEFAULT_HEALTH_PATH,
    DEFAULT_LIVENESS_PATH,
    DEFAULT_READINESS_PATH,
    create_health_check_app,
)
from creative_coding_assistant.api.streaming import (
    DEFAULT_STREAM_PATH,
    create_assistant_streaming_app,
)
from creative_coding_assistant.api.workspace_sessions import (
    DEFAULT_WORKSPACE_SESSION_PATH,
    create_workspace_session_app,
)
from creative_coding_assistant.core.config import Settings, load_settings
from creative_coding_assistant.core.logging import configure_logging

WsgiApplication = Callable[[dict[str, Any], StartResponse], Iterable[bytes]]

DEFAULT_DEV_HOST = "127.0.0.1"
DEFAULT_DEV_PORT = 8000
DEFAULT_BACKEND_ROUTE_METHODS = "GET, POST, PUT, OPTIONS"


@dataclass(frozen=True)
class MountedWsgiApp:
    path: str
    app: WsgiApplication


class BackendDevApplication:
    """Exact-path dispatcher for the local Next.js bridge WSGI apps."""

    def __init__(
        self,
        mounts: tuple[MountedWsgiApp, ...],
        *,
        settings_factory: Callable[[], Settings] = load_settings,
    ) -> None:
        self._mounts = mounts
        self._settings_factory = settings_factory

    def __call__(
        self,
        environ: dict[str, Any],
        start_response: StartResponse,
    ) -> Iterable[bytes]:
        path = str(environ.get("PATH_INFO", ""))
        for mount in self._mounts:
            if path == mount.path:
                return mount.app(environ, start_response)
        return error_response(
            start_response,
            HTTPStatus.NOT_FOUND,
            error="not_found",
            message="Backend bridge route was not found.",
            request_id=request_id_from_environ(environ),
            allow_methods=DEFAULT_BACKEND_ROUTE_METHODS,
            allow_origin=resolve_cors_allow_origin(
                environ,
                settings=self._settings_factory(),
            ),
            details={"available_paths": [mount.path for mount in self._mounts]},
        )

    @property
    def available_paths(self) -> tuple[str, ...]:
        """Return mounted paths in dispatcher order."""

        return tuple(mount.path for mount in self._mounts)


def create_backend_dev_app(
    *,
    stream_app: WsgiApplication | None = None,
    workspace_app: WsgiApplication | None = None,
    settings_factory: Callable[[], Settings] = load_settings,
) -> BackendDevApplication:
    """Create the local dispatcher for the assistant and workspace WSGI apps."""

    return BackendDevApplication(
        mounts=(
            MountedWsgiApp(
                path=DEFAULT_STREAM_PATH,
                app=stream_app
                or create_assistant_streaming_app(settings_factory=settings_factory),
            ),
            MountedWsgiApp(
                path=DEFAULT_WORKSPACE_SESSION_PATH,
                app=workspace_app
                or create_workspace_session_app(settings_factory=settings_factory),
            ),
            MountedWsgiApp(
                path=DEFAULT_HEALTH_PATH,
                app=create_health_check_app(settings_factory=settings_factory),
            ),
            MountedWsgiApp(
                path=DEFAULT_LIVENESS_PATH,
                app=create_health_check_app(settings_factory=settings_factory),
            ),
            MountedWsgiApp(
                path=DEFAULT_READINESS_PATH,
                app=create_health_check_app(settings_factory=settings_factory),
            ),
        ),
        settings_factory=settings_factory,
    )


def run_backend_dev_server(
    *,
    host: str = DEFAULT_DEV_HOST,
    port: int = DEFAULT_DEV_PORT,
    app: WsgiApplication | None = None,
    settings: Settings | None = None,
    allow_production_dev_server: bool = False,
) -> None:
    """Run the local backend bridge dispatcher with wsgiref."""

    resolved_settings = settings or load_settings()
    _validate_dev_server_environment(
        resolved_settings,
        allow_production_dev_server=allow_production_dev_server,
    )
    dev_app = app or create_backend_dev_app(settings_factory=lambda: resolved_settings)
    with make_server(host, port, dev_app) as server:
        print(f"Creative Coding Assistant backend bridge listening on {host}:{port}")
        server.serve_forever()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the local Creative Coding Assistant backend bridge.",
    )
    parser.add_argument("--host", default=DEFAULT_DEV_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_DEV_PORT)
    parser.add_argument(
        "--allow-production-dev-server",
        action="store_true",
        help="Allow the local wsgiref bridge when CCA_ENVIRONMENT=production.",
    )
    args = parser.parse_args(argv)

    settings = load_settings()
    configure_logging(
        settings.log_level,
        structured=settings.structured_logging,
    )
    run_backend_dev_server(
        host=args.host,
        port=args.port,
        settings=settings,
        allow_production_dev_server=args.allow_production_dev_server,
    )
    return 0


def _validate_dev_server_environment(
    settings: Settings,
    *,
    allow_production_dev_server: bool,
) -> None:
    if settings.environment.strip().lower() == "production" and (
        not allow_production_dev_server
    ):
        raise RuntimeError(
            "The local wsgiref backend bridge is a development server. "
            "Use a production WSGI/ASGI host, or pass "
            "--allow-production-dev-server for an explicit operator override."
        )


if __name__ == "__main__":
    raise SystemExit(main())
