"""Local WSGI dev entrypoint for browser-facing backend bridge apps."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any
from wsgiref.simple_server import make_server

from creative_coding_assistant.api.streaming import (
    DEFAULT_STREAM_PATH,
    create_assistant_streaming_app,
)
from creative_coding_assistant.api.workspace_sessions import (
    DEFAULT_WORKSPACE_SESSION_PATH,
    create_workspace_session_app,
)

StartResponse = Callable[
    [str, list[tuple[str, str]], Any | None],
    Callable[[bytes], object] | None,
]
WsgiApplication = Callable[[dict[str, Any], StartResponse], Iterable[bytes]]

DEFAULT_DEV_HOST = "127.0.0.1"
DEFAULT_DEV_PORT = 8000


@dataclass(frozen=True)
class MountedWsgiApp:
    path: str
    app: WsgiApplication


class BackendDevApplication:
    """Exact-path dispatcher for the local Next.js bridge WSGI apps."""

    def __init__(self, mounts: tuple[MountedWsgiApp, ...]) -> None:
        self._mounts = mounts

    def __call__(
        self,
        environ: dict[str, Any],
        start_response: StartResponse,
    ) -> Iterable[bytes]:
        path = str(environ.get("PATH_INFO", ""))
        for mount in self._mounts:
            if path == mount.path:
                return mount.app(environ, start_response)
        return _json_response(
            start_response,
            "404 Not Found",
            {
                "error": "not_found",
                "available_paths": [mount.path for mount in self._mounts],
            },
        )


def create_backend_dev_app(
    *,
    stream_app: WsgiApplication | None = None,
    workspace_app: WsgiApplication | None = None,
) -> BackendDevApplication:
    """Create the local dispatcher for the assistant and workspace WSGI apps."""

    return BackendDevApplication(
        mounts=(
            MountedWsgiApp(
                path=DEFAULT_STREAM_PATH,
                app=stream_app or create_assistant_streaming_app(),
            ),
            MountedWsgiApp(
                path=DEFAULT_WORKSPACE_SESSION_PATH,
                app=workspace_app or create_workspace_session_app(),
            ),
        )
    )


def run_backend_dev_server(
    *,
    host: str = DEFAULT_DEV_HOST,
    port: int = DEFAULT_DEV_PORT,
    app: WsgiApplication | None = None,
) -> None:
    """Run the local backend bridge dispatcher with wsgiref."""

    dev_app = app or create_backend_dev_app()
    with make_server(host, port, dev_app) as server:
        print(f"Creative Coding Assistant backend bridge listening on {host}:{port}")
        server.serve_forever()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the local Creative Coding Assistant backend bridge.",
    )
    parser.add_argument("--host", default=DEFAULT_DEV_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_DEV_PORT)
    args = parser.parse_args(argv)

    run_backend_dev_server(host=args.host, port=args.port)
    return 0


def _json_response(
    start_response: StartResponse,
    status: str,
    payload: dict[str, Any],
) -> Iterable[bytes]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    start_response(
        status,
        [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(body))),
            ("Access-Control-Allow-Origin", "*"),
            ("Access-Control-Allow-Headers", "Content-Type"),
            ("Access-Control-Allow-Methods", "GET, POST, PUT, OPTIONS"),
        ],
        None,
    )
    return (body,)


if __name__ == "__main__":
    raise SystemExit(main())
