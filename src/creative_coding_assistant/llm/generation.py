"""Provider-agnostic generation contracts built from rendered prompts."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import Protocol, Self

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.prompt_templates import (
    RenderedPromptResponse,
)
from creative_coding_assistant.orchestration.routing import RouteDecision, RouteName


class GenerationMessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    CONTEXT = "context"
    ASSISTANT = "assistant"


class GenerationMessageName(StrEnum):
    SYSTEM = "system"
    USER = "user"
    MEMORY = "memory"
    RETRIEVAL = "retrieval"


class GenerationFinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    CANCELLED = "cancelled"
    ERROR = "error"


class GenerationEventType(StrEnum):
    DELTA = "delta"
    COMPLETED = "completed"
    ERROR = "error"


class GenerationMessage(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: GenerationMessageRole
    name: GenerationMessageName
    content: str = Field(min_length=1)


class GenerationRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    route: RouteName
    rendered_prompt: RenderedPromptResponse
    stream: bool = True

    @model_validator(mode="after")
    def validate_route_alignment(self) -> Self:
        if self.rendered_prompt.request.route != self.route:
            raise ValueError("Rendered prompt route must match the generation route.")
        return self


class GenerationInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    request: GenerationRequest
    messages: tuple[GenerationMessage, ...] = Field(min_length=1)


class GenerationDelta(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    index: int = Field(ge=0)
    role: GenerationMessageRole = GenerationMessageRole.ASSISTANT
    content: str = Field(min_length=1)


class GeneratedOutput(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: GenerationMessageRole = GenerationMessageRole.ASSISTANT
    content: str = Field(min_length=1)
    finish_reason: GenerationFinishReason


class GenerationError(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class GenerationResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    request: GenerationInput
    output: GeneratedOutput


class GenerationStreamEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_type: GenerationEventType
    delta: GenerationDelta | None = None
    response: GenerationResponse | None = None
    error: GenerationError | None = None

    @model_validator(mode="after")
    def validate_payload_alignment(self) -> Self:
        if self.event_type is GenerationEventType.DELTA:
            invalid_delta_payload = (
                self.delta is None
                or self.response is not None
                or self.error is not None
            )
            if invalid_delta_payload:
                raise ValueError(
                    "Delta generation events require only a delta payload."
                )
        elif self.event_type is GenerationEventType.COMPLETED:
            invalid_response_payload = (
                self.response is None
                or self.delta is not None
                or self.error is not None
            )
            if invalid_response_payload:
                raise ValueError(
                    "Completed generation events require only a response payload."
                )
        elif self.event_type is GenerationEventType.ERROR:
            invalid_error_payload = (
                self.error is None
                or self.delta is not None
                or self.response is not None
            )
            if invalid_error_payload:
                raise ValueError(
                    "Error generation events require only an error payload."
                )
        return self


class GenerationInputBuilder(Protocol):
    def build(
        self,
        request: GenerationRequest,
    ) -> GenerationInput:
        """Transform rendered prompt sections into provider-ready messages."""


class GenerationProvider(Protocol):
    def stream(
        self,
        request: GenerationInput,
    ) -> Iterable[GenerationStreamEvent]:
        """Stream provider-agnostic generation events for one request."""


class RenderedPromptGenerationBuilder:
    """Convert rendered prompt sections into provider-neutral messages."""

    def build(
        self,
        request: GenerationRequest,
    ) -> GenerationInput:
        messages = tuple(
            GenerationMessage(
                role=GenerationMessageRole(section.role.value),
                name=GenerationMessageName(section.name.value),
                content=section.content,
            )
            for section in request.rendered_prompt.sections
        )
        generation_input = GenerationInput(request=request, messages=messages)
        logger.info(
            "Built generation input with {} message(s) for route '{}'",
            len(generation_input.messages),
            request.route.value,
        )
        return generation_input


def build_generation_request(
    *,
    route_decision: RouteDecision | RouteName,
    rendered_prompt: RenderedPromptResponse,
    stream: bool = True,
) -> GenerationRequest:
    route = (
        route_decision
        if isinstance(route_decision, RouteName)
        else route_decision.route
    )
    return GenerationRequest(
        route=route,
        rendered_prompt=rendered_prompt,
        stream=stream,
    )
