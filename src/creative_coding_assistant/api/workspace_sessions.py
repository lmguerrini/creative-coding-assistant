"""Minimal HTTP API for local workspace session persistence."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any
from urllib.parse import parse_qs

from pydantic import ValidationError

from creative_coding_assistant.workspace import (
    DEFAULT_LOCAL_SESSION_ID,
    DEFAULT_LOCAL_USER_ID,
    WorkspaceSessionPersistenceService,
    WorkspaceSessionRecord,
    build_workspace_session_persistence_service,
)

StartResponse = Callable[
    [str, list[tuple[str, str]], Any | None],
    Callable[[bytes], object] | None,
]

DEFAULT_WORKSPACE_SESSION_PATH = "/api/workspace/session"
MAX_REQUEST_BYTES = 256 * 1024


class WorkspaceSessionApplication:
    """Small WSGI app exposing local workspace session save/restore."""

    def __init__(
        self,
        *,
        service: WorkspaceSessionPersistenceService | None = None,
        service_factory: Callable[
            [], WorkspaceSessionPersistenceService
        ] = build_workspace_session_persistence_service,
        path: str = DEFAULT_WORKSPACE_SESSION_PATH,
    ) -> None:
        self._service = service
        self._service_factory = service_factory
        self._path = path

    def __call__(
        self,
        environ: dict[str, Any],
        start_response: StartResponse,
    ) -> Iterable[bytes]:
        path = str(environ.get("PATH_INFO", ""))
        method = str(environ.get("REQUEST_METHOD", "GET")).upper()

        if path != self._path:
            return _json_response(
                start_response,
                "404 Not Found",
                {"error": "not_found"},
            )

        if method == "OPTIONS":
            return _empty_response(start_response, "204 No Content")

        if method == "GET":
            return self._handle_get(environ, start_response)

        if method in {"POST", "PUT"}:
            return self._handle_save(environ, start_response)

        return _json_response(
            start_response,
            "405 Method Not Allowed",
            {"error": "method_not_allowed"},
            extra_headers=[("Allow", "GET, POST, PUT, OPTIONS")],
        )

    def _handle_get(
        self,
        environ: dict[str, Any],
        start_response: StartResponse,
    ) -> Iterable[bytes]:
        query = parse_qs(str(environ.get("QUERY_STRING", "")))
        user_id = _first_query_value(query, "userId", DEFAULT_LOCAL_USER_ID)
        session_id = _first_query_value(
            query,
            "sessionId",
            DEFAULT_LOCAL_SESSION_ID,
        )
        record = self._service_instance().get_session(
            user_id=user_id,
            session_id=session_id,
        )

        if record is None:
            return _json_response(
                start_response,
                "404 Not Found",
                {
                    "error": "session_not_found",
                    "userId": user_id,
                    "sessionId": session_id,
                },
            )

        return _json_response(
            start_response,
            "200 OK",
            record.model_dump(mode="json", by_alias=True),
        )

    def _handle_save(
        self,
        environ: dict[str, Any],
        start_response: StartResponse,
    ) -> Iterable[bytes]:
        try:
            payload = _read_json_body(environ)
            record = WorkspaceSessionRecord.model_validate(payload)
        except (ValueError, ValidationError) as exc:
            return _json_response(
                start_response,
                "400 Bad Request",
                {
                    "error": "invalid_session",
                    "message": str(exc),
                },
            )

        saved = self._service_instance().save_session(record)
        return _json_response(
            start_response,
            "200 OK",
            saved.model_dump(mode="json", by_alias=True),
        )

    def _service_instance(self) -> WorkspaceSessionPersistenceService:
        if self._service is not None:
            return self._service
        return self._service_factory()


def create_workspace_session_app(
    *,
    service: WorkspaceSessionPersistenceService | None = None,
    service_factory: Callable[
        [], WorkspaceSessionPersistenceService
    ] = build_workspace_session_persistence_service,
) -> WorkspaceSessionApplication:
    """Create the WSGI application used by the Next.js persistence bridge."""

    return WorkspaceSessionApplication(
        service=service,
        service_factory=service_factory,
    )


def _first_query_value(
    query: dict[str, list[str]],
    key: str,
    fallback: str,
) -> str:
    value = query.get(key, [fallback])[0].strip()
    return value or fallback


def _read_json_body(environ: dict[str, Any]) -> dict[str, Any]:
    raw_length = str(environ.get("CONTENT_LENGTH", "") or "0")
    try:
        content_length = int(raw_length)
    except ValueError as exc:
        raise ValueError("Invalid Content-Length header.") from exc

    if content_length <= 0:
        raise ValueError("Request body is required.")

    if content_length > MAX_REQUEST_BYTES:
        raise ValueError("Request body is too large.")

    body = environ["wsgi.input"].read(content_length)
    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    return payload


def _json_response(
    start_response: StartResponse,
    status: str,
    payload: dict[str, Any],
    *,
    extra_headers: list[tuple[str, str]] | None = None,
) -> Iterable[bytes]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    start_response(
        status,
        [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(body))),
            *_cors_headers(),
            *(extra_headers or []),
        ],
        None,
    )
    return (body,)


def _empty_response(
    start_response: StartResponse,
    status: str,
) -> Iterable[bytes]:
    start_response(
        status,
        [
            ("Content-Length", "0"),
            *_cors_headers(),
        ],
        None,
    )
    return ()


def _cors_headers() -> list[tuple[str, str]]:
    return [
        ("Access-Control-Allow-Origin", "*"),
        ("Access-Control-Allow-Headers", "Content-Type"),
        ("Access-Control-Allow-Methods", "GET, POST, PUT, OPTIONS"),
    ]
