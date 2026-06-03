"""Minimal NDJSON streaming API for the Next.js assistant bridge."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from creative_coding_assistant.app import build_assistant_service
from creative_coding_assistant.contracts import (
    MAX_IMAGE_REFERENCE_COUNT,
    AssistantArtifactRefinement,
    AssistantImageReference,
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
    StreamEvent,
    StreamEventType,
)
from creative_coding_assistant.orchestration import AssistantService

StartResponse = Callable[
    [str, list[tuple[str, str]], Any | None],
    Callable[[bytes], object] | None,
]

DEFAULT_STREAM_PATH = "/api/assistant/stream"
DEFAULT_CONVERSATION_ID = "local-nextjs-session"
DEFAULT_PROJECT_ID = "local-nextjs-workspace"
MAX_REQUEST_BYTES = 8 * 1024 * 1024


class AssistantStreamRequest(BaseModel):
    """Browser-facing request shape accepted by the streaming endpoint."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    query: str = Field(min_length=1)
    conversation_id: str = Field(
        default=DEFAULT_CONVERSATION_ID,
        alias="conversationId",
    )
    project_id: str = Field(default=DEFAULT_PROJECT_ID, alias="projectId")
    domain: CreativeCodingDomain | None = None
    domains: tuple[CreativeCodingDomain, ...] = Field(default_factory=tuple)
    mode: AssistantMode = AssistantMode.GENERATE
    attachments: tuple[AssistantImageReference, ...] = Field(default_factory=tuple)
    artifact_refinement: AssistantArtifactRefinement | None = Field(
        default=None,
        alias="artifactRefinement",
    )

    @field_validator("attachments")
    @classmethod
    def validate_attachment_count(
        cls,
        value: tuple[AssistantImageReference, ...],
    ) -> tuple[AssistantImageReference, ...]:
        if len(value) > MAX_IMAGE_REFERENCE_COUNT:
            raise ValueError(
                f"Attach up to {MAX_IMAGE_REFERENCE_COUNT} image references."
            )
        return value

    def to_assistant_request(self) -> AssistantRequest:
        """Convert the HTTP request into the stable backend service contract."""

        return AssistantRequest(
            query=self.query,
            conversation_id=self.conversation_id,
            project_id=self.project_id,
            domain=self.domain,
            domains=self.domains,
            mode=self.mode,
            attachments=self.attachments,
            artifact_refinement=self.artifact_refinement,
        )


def serialize_stream_event(event: StreamEvent) -> str:
    """Serialize one backend event as one NDJSON line."""

    return json.dumps(event.model_dump(mode="json"), separators=(",", ":")) + "\n"


def iter_assistant_stream_ndjson(
    *,
    request: AssistantStreamRequest,
    service: AssistantService,
) -> Iterator[str]:
    """Yield assistant service stream events as NDJSON lines."""

    next_sequence = 0
    try:
        for event in service.stream(request.to_assistant_request()):
            next_sequence = event.sequence + 1
            yield serialize_stream_event(event)
    except Exception:
        error_event = StreamEvent(
            event_type=StreamEventType.ERROR,
            sequence=next_sequence,
            payload={
                "code": "assistant_stream_failed",
                "message": "Assistant stream failed before completion.",
                "category": "stream",
                "subsystem": "assistant_stream",
                "recoverable": True,
                "suggested_action": "Retry the request from the client.",
                "retry_label": "Send prompt again",
            },
        )
        yield serialize_stream_event(error_event)


class AssistantStreamingApplication:
    """Small WSGI app exposing the assistant NDJSON stream endpoint."""

    def __init__(
        self,
        *,
        service: AssistantService | None = None,
        service_factory: Callable[[], AssistantService] = build_assistant_service,
        path: str = DEFAULT_STREAM_PATH,
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

        if method != "POST":
            return _json_response(
                start_response,
                "405 Method Not Allowed",
                {"error": "method_not_allowed"},
                extra_headers=[("Allow", "POST, OPTIONS")],
            )

        try:
            payload = _read_json_body(environ)
            request = AssistantStreamRequest.model_validate(payload)
        except (ValueError, ValidationError) as exc:
            return _json_response(
                start_response,
                "400 Bad Request",
                {
                    "error": "invalid_request",
                    "message": str(exc),
                },
            )

        start_response(
            "200 OK",
            [
                ("Content-Type", "application/x-ndjson; charset=utf-8"),
                ("Cache-Control", "no-cache, no-transform"),
                ("X-Accel-Buffering", "no"),
                *_cors_headers(),
            ],
            None,
        )
        service = (
            self._service if self._service is not None else self._service_factory()
        )
        return (
            line.encode("utf-8")
            for line in iter_assistant_stream_ndjson(
                request=request,
                service=service,
            )
        )


def create_assistant_streaming_app(
    *,
    service: AssistantService | None = None,
    service_factory: Callable[[], AssistantService] = build_assistant_service,
) -> AssistantStreamingApplication:
    """Create the WSGI application used by the first Next.js bridge."""

    return AssistantStreamingApplication(
        service=service,
        service_factory=service_factory,
    )


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
    decoded = body.decode("utf-8")
    payload = json.loads(decoded)
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
        ("Access-Control-Allow-Methods", "POST, OPTIONS"),
    ]
