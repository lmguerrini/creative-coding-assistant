"""Orchestration boundary for provider-neutral generation input preparation."""

from __future__ import annotations

from typing import Protocol, Self

from loguru import logger
from pydantic import BaseModel, ConfigDict, model_validator

from creative_coding_assistant.llm.generation import (
    GenerationInput,
    GenerationInputBuilder,
    RenderedPromptGenerationBuilder,
    build_generation_request,
)
from creative_coding_assistant.orchestration.prompt_templates import (
    RenderedPromptResponse,
)
from creative_coding_assistant.orchestration.routing import RouteDecision, RouteName


class ProviderGenerationRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    route: RouteName
    rendered_prompt: RenderedPromptResponse
    stream: bool = True

    @model_validator(mode="after")
    def validate_route_alignment(self) -> Self:
        if self.rendered_prompt.request.route != self.route:
            raise ValueError(
                "Rendered prompt route must match the provider boundary route."
            )
        return self


class ProviderGenerationGateway(Protocol):
    def prepare_generation(
        self,
        request: ProviderGenerationRequest,
    ) -> GenerationInput:
        """Prepare provider-neutral generation input without model execution."""


class LlmGenerationAdapter:
    """Adapt rendered prompts into llm generation inputs for later providers."""

    def __init__(
        self,
        generation_input_builder: GenerationInputBuilder | None = None,
    ) -> None:
        self._generation_input_builder = (
            generation_input_builder or RenderedPromptGenerationBuilder()
        )

    def prepare_generation(
        self,
        request: ProviderGenerationRequest,
    ) -> GenerationInput:
        generation_request = build_generation_request(
            route_decision=request.route,
            rendered_prompt=request.rendered_prompt,
            stream=request.stream,
        )
        generation_input = self._generation_input_builder.build(generation_request)
        logger.info(
            "Prepared provider generation input with {} message(s) for route '{}'",
            len(generation_input.messages),
            request.route.value,
        )
        return generation_input


def build_provider_generation_request(
    *,
    route_decision: RouteDecision | RouteName,
    rendered_prompt: RenderedPromptResponse,
    stream: bool = True,
) -> ProviderGenerationRequest:
    route = (
        route_decision
        if isinstance(route_decision, RouteName)
        else route_decision.route
    )
    return ProviderGenerationRequest(
        route=route,
        rendered_prompt=rendered_prompt,
        stream=stream,
    )
