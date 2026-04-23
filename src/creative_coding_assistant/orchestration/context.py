"""Orchestration-facing context assembly contracts and adapters."""

from __future__ import annotations

from typing import Protocol, Self

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.memory import MemoryContextResponse
from creative_coding_assistant.orchestration.retrieval import RetrievalContextResponse
from creative_coding_assistant.orchestration.routing import RouteDecision, RouteName


class AssembledContextRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    route: RouteName
    memory_context: MemoryContextResponse | None = None
    retrieval_context: RetrievalContextResponse | None = None

    @model_validator(mode="after")
    def validate_route_alignment(self) -> Self:
        if self.memory_context is None and self.retrieval_context is None:
            raise ValueError("Assembled context requires at least one context source.")
        if (
            self.memory_context is not None
            and self.memory_context.request.route != self.route
        ):
            raise ValueError("Memory context route must match the assembled route.")
        if (
            self.retrieval_context is not None
            and self.retrieval_context.request.route != self.route
        ):
            raise ValueError("Retrieval context route must match the assembled route.")
        return self


class AssembledContextSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    recent_turn_count: int = Field(ge=0)
    has_running_summary: bool
    project_memory_count: int = Field(ge=0)
    retrieval_chunk_count: int = Field(ge=0)


class AssembledContextResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    request: AssembledContextRequest
    summary: AssembledContextSummary
    memory_context: MemoryContextResponse | None = None
    retrieval_context: RetrievalContextResponse | None = None


class ContextAssembler(Protocol):
    def assemble(
        self,
        request: AssembledContextRequest,
    ) -> AssembledContextResponse:
        """Compose memory and retrieval outputs into one orchestration context."""


class OrchestrationContextAssembler:
    """Compose independent subsystem contexts without formatting prompts."""

    def assemble(
        self,
        request: AssembledContextRequest,
    ) -> AssembledContextResponse:
        memory_context = request.memory_context
        retrieval_context = request.retrieval_context
        summary = AssembledContextSummary(
            recent_turn_count=(
                len(memory_context.recent_turns) if memory_context is not None else 0
            ),
            has_running_summary=(
                memory_context.running_summary is not None
                if memory_context is not None
                else False
            ),
            project_memory_count=(
                len(memory_context.project_memories)
                if memory_context is not None
                else 0
            ),
            retrieval_chunk_count=(
                len(retrieval_context.chunks) if retrieval_context is not None else 0
            ),
        )
        logger.info(
            "Assembled orchestration context with {} recent turn(s), {} project "
            "memory item(s), and {} retrieval chunk(s)",
            summary.recent_turn_count,
            summary.project_memory_count,
            summary.retrieval_chunk_count,
        )
        return AssembledContextResponse(
            request=request,
            summary=summary,
            memory_context=memory_context,
            retrieval_context=retrieval_context,
        )


def build_assembled_context_request(
    *,
    route_decision: RouteDecision,
    memory_context: MemoryContextResponse | None,
    retrieval_context: RetrievalContextResponse | None,
) -> AssembledContextRequest | None:
    if memory_context is None and retrieval_context is None:
        return None
    return AssembledContextRequest(
        route=route_decision.route,
        memory_context=memory_context,
        retrieval_context=retrieval_context,
    )
