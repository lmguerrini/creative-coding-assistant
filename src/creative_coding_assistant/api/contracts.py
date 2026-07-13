"""Stable HTTP API response contracts for browser-facing runtime bridges."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from http import HTTPStatus
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

API_CONTRACT_VERSION = "api.v1"
ERROR_CONTRACT_VERSION = "api-error.v1"
STREAM_CONTRACT_VERSION = "assistant-stream.v1"
WORKSPACE_SESSION_CONTRACT_VERSION = "workspace-session.v1"
HEALTH_CONTRACT_VERSION = "health.v1"
DEFAULT_CORS_ALLOW_ORIGIN: str | None = None
REQUEST_ID_HEADER = "X-Request-Id"
MAX_REQUEST_ID_LENGTH = 128
API_CONTRACT_HEADER = "X-CCA-API-Contract-Version"
ERROR_CONTRACT_HEADER = "X-CCA-Error-Contract-Version"
STREAM_CONTRACT_HEADER = "X-CCA-Stream-Contract-Version"
WORKSPACE_SESSION_CONTRACT_HEADER = "X-CCA-Workspace-Session-Contract-Version"
HEALTH_CONTRACT_HEADER = "X-CCA-Health-Contract-Version"
_JSON_SEPARATORS = (",", ":")

StartResponse = Callable[
    [str, list[tuple[str, str]], Any | None],
    Callable[[bytes], object] | None,
]


class ApiErrorResponse(BaseModel):
    """Versioned API error body that preserves the legacy ``error`` field."""

    model_config = ConfigDict(
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    error: str = Field(min_length=1)
    message: str = Field(min_length=1)
    status: int = Field(ge=400, le=599)
    request_id: str = Field(alias="requestId", min_length=1)
    contract_version: str = Field(
        default=ERROR_CONTRACT_VERSION,
        alias="contractVersion",
    )
    recoverable: bool = True
    details: dict[str, Any] = Field(default_factory=dict)


class ApiRequestBodyError(ValueError):
    """Typed request-body parsing error that maps directly to an HTTP status."""

    def __init__(self, *, code: str, message: str, status: HTTPStatus) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def request_id_from_environ(environ: dict[str, Any]) -> str:
    """Resolve or create a stable request id for API responses."""

    incoming = str(environ.get("HTTP_X_REQUEST_ID", "")).strip()
    if (
        incoming
        and len(incoming) <= MAX_REQUEST_ID_LENGTH
        and all(" " <= character <= "~" for character in incoming)
    ):
        return incoming
    return str(uuid4())


def cors_headers(
    *,
    allow_methods: str,
    allow_origin: str | None = DEFAULT_CORS_ALLOW_ORIGIN,
) -> list[tuple[str, str]]:
    """Return CORS headers shared by all browser-facing WSGI apps."""

    headers = [
        ("Access-Control-Allow-Headers", "Content-Type, X-Request-Id"),
        ("Access-Control-Allow-Methods", allow_methods),
    ]
    if allow_origin is not None:
        headers.insert(0, ("Access-Control-Allow-Origin", allow_origin))
        if allow_origin != "*":
            headers.append(("Vary", "Origin"))
    return headers


def json_response(
    start_response: StartResponse,
    status: HTTPStatus | str,
    payload: dict[str, Any],
    *,
    request_id: str,
    allow_methods: str,
    allow_origin: str | None = DEFAULT_CORS_ALLOW_ORIGIN,
    extra_headers: list[tuple[str, str]] | None = None,
) -> Iterable[bytes]:
    """Emit a compact JSON response with stable API headers."""

    body = json.dumps(payload, separators=_JSON_SEPARATORS).encode("utf-8")
    start_response(
        _status_text(status),
        [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(body))),
            (REQUEST_ID_HEADER, request_id),
            (API_CONTRACT_HEADER, API_CONTRACT_VERSION),
            *cors_headers(allow_methods=allow_methods, allow_origin=allow_origin),
            *(extra_headers or []),
        ],
        None,
    )
    return (body,)


def empty_response(
    start_response: StartResponse,
    status: HTTPStatus | str,
    *,
    request_id: str,
    allow_methods: str,
    allow_origin: str | None = DEFAULT_CORS_ALLOW_ORIGIN,
    extra_headers: list[tuple[str, str]] | None = None,
) -> Iterable[bytes]:
    """Emit an empty response with stable API headers."""

    start_response(
        _status_text(status),
        [
            ("Content-Length", "0"),
            (REQUEST_ID_HEADER, request_id),
            (API_CONTRACT_HEADER, API_CONTRACT_VERSION),
            *cors_headers(allow_methods=allow_methods, allow_origin=allow_origin),
            *(extra_headers or []),
        ],
        None,
    )
    return ()


def error_response(
    start_response: StartResponse,
    status: HTTPStatus,
    *,
    error: str,
    message: str,
    request_id: str,
    allow_methods: str,
    allow_origin: str | None = DEFAULT_CORS_ALLOW_ORIGIN,
    recoverable: bool = True,
    details: dict[str, Any] | None = None,
    extra_headers: list[tuple[str, str]] | None = None,
) -> Iterable[bytes]:
    """Emit a versioned API error while keeping legacy top-level details."""

    error_details = details or {}
    payload = ApiErrorResponse(
        error=error,
        message=message,
        status=status.value,
        request_id=request_id,
        recoverable=recoverable,
        details=error_details,
    ).model_dump(mode="json", by_alias=True)
    payload.update(error_details)
    return json_response(
        start_response,
        status,
        payload,
        request_id=request_id,
        allow_methods=allow_methods,
        allow_origin=allow_origin,
        extra_headers=[
            (ERROR_CONTRACT_HEADER, ERROR_CONTRACT_VERSION),
            *(extra_headers or []),
        ],
    )


def read_json_body(environ: dict[str, Any], *, max_bytes: int) -> dict[str, Any]:
    """Read and validate a JSON request body from WSGI environ."""

    raw_length = str(environ.get("CONTENT_LENGTH", "") or "0")
    try:
        content_length = int(raw_length)
    except ValueError as exc:
        raise ApiRequestBodyError(
            code="invalid_content_length",
            message="Invalid Content-Length header.",
            status=HTTPStatus.BAD_REQUEST,
        ) from exc

    if content_length <= 0:
        raise ApiRequestBodyError(
            code="request_body_required",
            message="Request body is required.",
            status=HTTPStatus.BAD_REQUEST,
        )

    if content_length > max_bytes:
        raise ApiRequestBodyError(
            code="request_body_too_large",
            message="Request body is too large.",
            status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        )

    try:
        body = environ["wsgi.input"].read(content_length)
        payload = json.loads(body.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ApiRequestBodyError(
            code="invalid_json_encoding",
            message="Request body must be UTF-8 encoded JSON.",
            status=HTTPStatus.BAD_REQUEST,
        ) from exc
    except json.JSONDecodeError as exc:
        raise ApiRequestBodyError(
            code="invalid_json",
            message="Request body must be valid JSON.",
            status=HTTPStatus.BAD_REQUEST,
        ) from exc

    if not isinstance(payload, dict):
        raise ApiRequestBodyError(
            code="invalid_json_object",
            message="Request body must be a JSON object.",
            status=HTTPStatus.BAD_REQUEST,
        )
    return payload


def status_code(status: str) -> int:
    """Extract the numeric code from a WSGI status string."""

    return int(status.split(" ", maxsplit=1)[0])


def _status_text(status: HTTPStatus | str) -> str:
    if isinstance(status, HTTPStatus):
        return f"{status.value} {status.phrase}"
    return status
