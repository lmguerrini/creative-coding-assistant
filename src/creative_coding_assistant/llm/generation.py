"""Provider-agnostic generation contracts built from rendered prompts."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import Literal, Protocol, Self

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from creative_coding_assistant.contracts import (
    MAX_IMAGE_REFERENCE_BYTES,
    MAX_IMAGE_REFERENCE_COUNT,
    GenerationControls,
)
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


class GenerationTokenUsage(BaseModel):
    """Provider-neutral token accounting for one generation response."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)


class GenerationMessage(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: GenerationMessageRole
    name: GenerationMessageName
    content: str = Field(min_length=1)


class GenerationImageInput(BaseModel):
    """Validated image bytes held behind a redacted provider boundary."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    mime_type: str = Field(min_length=1)
    size_bytes: int = Field(gt=0, le=MAX_IMAGE_REFERENCE_BYTES)
    data_url: SecretStr
    detail: Literal["auto", "low", "high"] = "auto"


class GenerationRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    route: RouteName
    rendered_prompt: RenderedPromptResponse
    stream: bool = True
    generation_controls: GenerationControls = Field(default_factory=GenerationControls)

    @model_validator(mode="after")
    def validate_route_alignment(self) -> Self:
        if self.rendered_prompt.request.route != self.route:
            raise ValueError("Rendered prompt route must match the generation route.")
        return self


class GenerationInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    request: GenerationRequest
    messages: tuple[GenerationMessage, ...] = Field(min_length=1)
    image_inputs: tuple[GenerationImageInput, ...] = Field(
        default_factory=tuple,
        max_length=MAX_IMAGE_REFERENCE_COUNT,
    )

    @model_validator(mode="after")
    def validate_image_input_alignment(self) -> Self:
        user_message_count = sum(
            message.name is GenerationMessageName.USER for message in self.messages
        )
        if self.image_inputs and user_message_count != 1:
            raise ValueError(
                "Generation image inputs require exactly one user message."
            )
        return self


class GenerationDelta(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)
    role: GenerationMessageRole = GenerationMessageRole.ASSISTANT
    content: str = Field(min_length=1)
    provider: str | None = Field(default=None, min_length=1)
    model: str | None = Field(default=None, min_length=1)


class GeneratedOutput(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: GenerationMessageRole = GenerationMessageRole.ASSISTANT
    content: str = Field(min_length=1)
    finish_reason: GenerationFinishReason
    provider: str | None = Field(default=None, min_length=1)
    model: str | None = Field(default=None, min_length=1)
    response_id: str | None = Field(default=None, min_length=1)
    usage: GenerationTokenUsage | None = None


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
        attachments = (
            request.rendered_prompt.request.prompt_input.request.assistant_request.attachments
        )
        image_inputs = tuple(
            GenerationImageInput(
                id=attachment.id,
                name=attachment.name,
                mime_type=attachment.mime_type,
                size_bytes=attachment.size_bytes,
                data_url=attachment.data_url,
            )
            for attachment in attachments
            if attachment.data_url is not None
        )
        generation_input = GenerationInput(
            request=request,
            messages=messages,
            image_inputs=image_inputs,
        )
        logger.info(
            "Built generation input with {} message(s) and {} image input(s) "
            "for route '{}'",
            len(generation_input.messages),
            len(generation_input.image_inputs),
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
        generation_controls=(
            rendered_prompt.request.prompt_input.request.assistant_request.generation_controls
        ),
    )
