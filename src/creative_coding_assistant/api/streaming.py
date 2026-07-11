"""Minimal NDJSON streaming API for the Next.js assistant bridge."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from http import HTTPStatus
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from creative_coding_assistant.api.contracts import (
    STREAM_CONTRACT_HEADER,
    STREAM_CONTRACT_VERSION,
    ApiRequestBodyError,
    StartResponse,
    cors_headers,
    empty_response,
    error_response,
    read_json_body,
    request_id_from_environ,
)
from creative_coding_assistant.api.cors import resolve_cors_allow_origin
from creative_coding_assistant.app import build_assistant_service
from creative_coding_assistant.contracts import (
    MAX_IMAGE_REFERENCE_COUNT,
    AssistantArtifactRefinement,
    AssistantImageReference,
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
    GenerationControls,
    PersonalizationContext,
    StreamEvent,
    StreamEventType,
    WorkflowExecutionMode,
)
from creative_coding_assistant.core.config import Settings, load_settings
from creative_coding_assistant.orchestration import AssistantService
from creative_coding_assistant.security import assess_user_request_safety

DEFAULT_STREAM_PATH = "/api/assistant/stream"
DEFAULT_CONVERSATION_ID = "local-nextjs-session"
DEFAULT_PROJECT_ID = "local-nextjs-workspace"
MAX_REQUEST_BYTES = 8 * 1024 * 1024
_JSON_SEPARATORS = (",", ":")
STREAM_METHODS = "POST, OPTIONS"


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
    workflow_mode: WorkflowExecutionMode = Field(
        default=WorkflowExecutionMode.AUTO,
        alias="workflowMode",
    )
    generation_controls: GenerationControls = Field(
        default_factory=GenerationControls,
        alias="generationControls",
    )
    personalization_context: PersonalizationContext = Field(
        default_factory=PersonalizationContext,
        alias="personalizationContext",
    )
    attachments: tuple[AssistantImageReference, ...] = Field(default_factory=tuple)
    artifact_refinement: AssistantArtifactRefinement | None = Field(
        default=None,
        alias="artifactRefinement",
    )
    clarification_response: str | None = Field(
        default=None,
        alias="clarificationResponse",
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
            workflow_mode=self.workflow_mode,
            generation_controls=self.generation_controls,
            personalization_context=self.personalization_context,
            attachments=self.attachments,
            artifact_refinement=self.artifact_refinement,
            clarification_response=self.clarification_response,
        )


def serialize_stream_event(event: StreamEvent) -> str:
    """Serialize one backend event as one NDJSON line."""

    return _serialize_ndjson_payload(event.model_dump(mode="json"))


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
        yield serialize_stream_event(
            _assistant_stream_failed_event(sequence=next_sequence)
        )


class AssistantStreamingApplication:
    """Small WSGI app exposing the assistant NDJSON stream endpoint."""

    def __init__(
        self,
        *,
        service: AssistantService | None = None,
        service_factory: Callable[[], AssistantService] = build_assistant_service,
        settings_factory: Callable[[], Settings] = load_settings,
        path: str = DEFAULT_STREAM_PATH,
    ) -> None:
        self._service = service
        self._service_factory = service_factory
        self._settings_factory = settings_factory
        self._path = path

    def __call__(
        self,
        environ: dict[str, Any],
        start_response: StartResponse,
    ) -> Iterable[bytes]:
        request_id = request_id_from_environ(environ)
        path = str(environ.get("PATH_INFO", ""))
        method = str(environ.get("REQUEST_METHOD", "GET")).upper()
        allow_origin = resolve_cors_allow_origin(
            environ,
            settings=self._settings_factory(),
        )

        if path != self._path:
            return error_response(
                start_response,
                HTTPStatus.NOT_FOUND,
                error="not_found",
                message="Assistant stream route was not found.",
                request_id=request_id,
                allow_methods=STREAM_METHODS,
                allow_origin=allow_origin,
                details={"available_paths": [self._path]},
            )

        if method == "OPTIONS":
            return empty_response(
                start_response,
                HTTPStatus.NO_CONTENT,
                request_id=request_id,
                allow_methods=STREAM_METHODS,
                allow_origin=allow_origin,
                extra_headers=[(STREAM_CONTRACT_HEADER, STREAM_CONTRACT_VERSION)],
            )

        if method != "POST":
            return error_response(
                start_response,
                HTTPStatus.METHOD_NOT_ALLOWED,
                error="method_not_allowed",
                message="Assistant stream accepts POST and OPTIONS.",
                request_id=request_id,
                allow_methods=STREAM_METHODS,
                allow_origin=allow_origin,
                details={"allowed_methods": ["POST", "OPTIONS"]},
                extra_headers=[
                    ("Allow", STREAM_METHODS),
                    (STREAM_CONTRACT_HEADER, STREAM_CONTRACT_VERSION),
                ],
            )

        try:
            payload = read_json_body(environ, max_bytes=MAX_REQUEST_BYTES)
            request = AssistantStreamRequest.model_validate(payload)
        except ApiRequestBodyError as exc:
            return error_response(
                start_response,
                exc.status,
                error=exc.code,
                message=exc.message,
                request_id=request_id,
                allow_methods=STREAM_METHODS,
                allow_origin=allow_origin,
                extra_headers=[(STREAM_CONTRACT_HEADER, STREAM_CONTRACT_VERSION)],
            )
        except ValidationError as exc:
            return error_response(
                start_response,
                HTTPStatus.BAD_REQUEST,
                error="invalid_request",
                message=str(exc),
                request_id=request_id,
                allow_methods=STREAM_METHODS,
                allow_origin=allow_origin,
                extra_headers=[(STREAM_CONTRACT_HEADER, STREAM_CONTRACT_VERSION)],
            )

        safety = assess_user_request_safety(request.query)
        if not safety.allowed:
            return error_response(
                start_response,
                HTTPStatus.BAD_REQUEST,
                error=safety.code or "unsafe_request",
                message=safety.message
                or "This request is outside the supported safety boundary.",
                request_id=request_id,
                allow_methods=STREAM_METHODS,
                allow_origin=allow_origin,
                recoverable=True,
                extra_headers=[(STREAM_CONTRACT_HEADER, STREAM_CONTRACT_VERSION)],
            )

        start_response(
            "200 OK",
            [
                ("Content-Type", "application/x-ndjson; charset=utf-8"),
                ("Cache-Control", "no-cache, no-transform"),
                ("X-Accel-Buffering", "no"),
                ("X-Request-Id", request_id),
                (STREAM_CONTRACT_HEADER, STREAM_CONTRACT_VERSION),
                *cors_headers(
                    allow_methods=STREAM_METHODS,
                    allow_origin=allow_origin,
                ),
            ],
            None,
        )
        return _stream_response_bytes(
            request=request,
            service=self._service,
            service_factory=self._service_factory,
        )


def create_assistant_streaming_app(
    *,
    service: AssistantService | None = None,
    service_factory: Callable[[], AssistantService] = build_assistant_service,
    settings_factory: Callable[[], Settings] = load_settings,
) -> AssistantStreamingApplication:
    """Create the WSGI application used by the first Next.js bridge."""

    return AssistantStreamingApplication(
        service=service,
        service_factory=service_factory,
        settings_factory=settings_factory,
    )


def _serialize_ndjson_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=_JSON_SEPARATORS) + "\n"


def _assistant_stream_failed_event(*, sequence: int) -> StreamEvent:
    return StreamEvent(
        event_type=StreamEventType.ERROR,
        sequence=sequence,
        payload={
            "code": "assistant_stream_failed",
            "message": "Assistant stream failed before completion.",
            "category": "stream",
            "subsystem": "assistant_stream",
            "recoverable": True,
            "suggested_action": "Retry the request from the client.",
            "retry_label": "Send prompt again",
            "contract_version": STREAM_CONTRACT_VERSION,
        },
    )


def _stream_response_bytes(
    *,
    request: AssistantStreamRequest,
    service: AssistantService | None,
    service_factory: Callable[[], AssistantService],
) -> Iterable[bytes]:
    try:
        resolved_service = service if service is not None else service_factory()
        for line in iter_assistant_stream_ndjson(
            request=request,
            service=resolved_service,
        ):
            yield line.encode("utf-8")
    except Exception:
        yield serialize_stream_event(_assistant_stream_failed_event(sequence=0)).encode(
            "utf-8"
        )
