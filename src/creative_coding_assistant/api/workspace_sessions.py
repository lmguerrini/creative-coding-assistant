"""Minimal HTTP API for local workspace session persistence."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs

from pydantic import ValidationError

from creative_coding_assistant.api.contracts import (
    WORKSPACE_SESSION_CONTRACT_HEADER,
    WORKSPACE_SESSION_CONTRACT_VERSION,
    ApiRequestBodyError,
    StartResponse,
    empty_response,
    error_response,
    json_response,
    read_json_body,
    request_id_from_environ,
)
from creative_coding_assistant.workspace import (
    DEFAULT_LOCAL_SESSION_ID,
    DEFAULT_LOCAL_USER_ID,
    WorkspaceSessionPersistenceService,
    WorkspaceSessionRecord,
    build_workspace_session_persistence_service,
)

DEFAULT_WORKSPACE_SESSION_PATH = "/api/workspace/session"
MAX_REQUEST_BYTES = 256 * 1024
WORKSPACE_SESSION_METHODS = "GET, POST, PUT, OPTIONS"


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
        request_id = request_id_from_environ(environ)
        path = str(environ.get("PATH_INFO", ""))
        method = str(environ.get("REQUEST_METHOD", "GET")).upper()

        if path != self._path:
            return error_response(
                start_response,
                HTTPStatus.NOT_FOUND,
                error="not_found",
                message="Workspace session route was not found.",
                request_id=request_id,
                allow_methods=WORKSPACE_SESSION_METHODS,
                details={"available_paths": [self._path]},
            )

        if method == "OPTIONS":
            return empty_response(
                start_response,
                HTTPStatus.NO_CONTENT,
                request_id=request_id,
                allow_methods=WORKSPACE_SESSION_METHODS,
                extra_headers=[
                    (
                        WORKSPACE_SESSION_CONTRACT_HEADER,
                        WORKSPACE_SESSION_CONTRACT_VERSION,
                    )
                ],
            )

        if method == "GET":
            return self._handle_get(environ, start_response, request_id=request_id)

        if method in {"POST", "PUT"}:
            return self._handle_save(environ, start_response, request_id=request_id)

        return error_response(
            start_response,
            HTTPStatus.METHOD_NOT_ALLOWED,
            error="method_not_allowed",
            message="Workspace session accepts GET, POST, PUT, and OPTIONS.",
            request_id=request_id,
            allow_methods=WORKSPACE_SESSION_METHODS,
            details={"allowed_methods": ["GET", "POST", "PUT", "OPTIONS"]},
            extra_headers=[
                ("Allow", WORKSPACE_SESSION_METHODS),
                (
                    WORKSPACE_SESSION_CONTRACT_HEADER,
                    WORKSPACE_SESSION_CONTRACT_VERSION,
                ),
            ],
        )

    def _handle_get(
        self,
        environ: dict[str, Any],
        start_response: StartResponse,
        *,
        request_id: str,
    ) -> Iterable[bytes]:
        query = parse_qs(str(environ.get("QUERY_STRING", "")))
        user_id = _first_query_value(query, "userId", DEFAULT_LOCAL_USER_ID)
        session_id = _first_query_value(
            query,
            "sessionId",
            DEFAULT_LOCAL_SESSION_ID,
        )
        try:
            record = self._service_instance().get_session(
                user_id=user_id,
                session_id=session_id,
            )
        except Exception:
            return _workspace_unavailable_response(
                start_response,
                request_id=request_id,
            )

        if record is None:
            return error_response(
                start_response,
                HTTPStatus.NOT_FOUND,
                error="session_not_found",
                message="Workspace session was not found.",
                request_id=request_id,
                allow_methods=WORKSPACE_SESSION_METHODS,
                details={"userId": user_id, "sessionId": session_id},
                extra_headers=[
                    (
                        WORKSPACE_SESSION_CONTRACT_HEADER,
                        WORKSPACE_SESSION_CONTRACT_VERSION,
                    )
                ],
            )

        return json_response(
            start_response,
            HTTPStatus.OK,
            record.model_dump(mode="json", by_alias=True),
            request_id=request_id,
            allow_methods=WORKSPACE_SESSION_METHODS,
            extra_headers=[
                (
                    WORKSPACE_SESSION_CONTRACT_HEADER,
                    WORKSPACE_SESSION_CONTRACT_VERSION,
                )
            ],
        )

    def _handle_save(
        self,
        environ: dict[str, Any],
        start_response: StartResponse,
        *,
        request_id: str,
    ) -> Iterable[bytes]:
        try:
            payload = read_json_body(environ, max_bytes=MAX_REQUEST_BYTES)
            record = WorkspaceSessionRecord.model_validate(payload)
        except ApiRequestBodyError as exc:
            return error_response(
                start_response,
                exc.status,
                error=exc.code,
                message=exc.message,
                request_id=request_id,
                allow_methods=WORKSPACE_SESSION_METHODS,
                extra_headers=[
                    (
                        WORKSPACE_SESSION_CONTRACT_HEADER,
                        WORKSPACE_SESSION_CONTRACT_VERSION,
                    )
                ],
            )
        except ValidationError as exc:
            return error_response(
                start_response,
                HTTPStatus.BAD_REQUEST,
                error="invalid_session",
                message=str(exc),
                request_id=request_id,
                allow_methods=WORKSPACE_SESSION_METHODS,
                extra_headers=[
                    (
                        WORKSPACE_SESSION_CONTRACT_HEADER,
                        WORKSPACE_SESSION_CONTRACT_VERSION,
                    )
                ],
            )

        try:
            saved = self._service_instance().save_session(record)
        except Exception:
            return _workspace_unavailable_response(
                start_response,
                request_id=request_id,
            )

        return json_response(
            start_response,
            HTTPStatus.OK,
            saved.model_dump(mode="json", by_alias=True),
            request_id=request_id,
            allow_methods=WORKSPACE_SESSION_METHODS,
            extra_headers=[
                (
                    WORKSPACE_SESSION_CONTRACT_HEADER,
                    WORKSPACE_SESSION_CONTRACT_VERSION,
                )
            ],
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


def _workspace_unavailable_response(
    start_response: StartResponse,
    *,
    request_id: str,
) -> Iterable[bytes]:
    return error_response(
        start_response,
        HTTPStatus.SERVICE_UNAVAILABLE,
        error="workspace_session_unavailable",
        message="Workspace session persistence is temporarily unavailable.",
        request_id=request_id,
        allow_methods=WORKSPACE_SESSION_METHODS,
        extra_headers=[
            (
                WORKSPACE_SESSION_CONTRACT_HEADER,
                WORKSPACE_SESSION_CONTRACT_VERSION,
            )
        ],
    )
