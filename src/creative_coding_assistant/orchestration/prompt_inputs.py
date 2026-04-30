"""Prompt-input contracts and transformation boundaries."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, Self

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.memory import ConversationRole, ProjectMemoryKind
from creative_coding_assistant.orchestration.context import AssembledContextResponse
from creative_coding_assistant.orchestration.routing import (
    DomainSelectionShape,
    RouteDecision,
    RouteName,
)
from creative_coding_assistant.rag.sources import OfficialSourceType


class PromptUserInput(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    query: str = Field(min_length=1)
    mode: AssistantMode
    domain: CreativeCodingDomain | None = None
    domains: tuple[CreativeCodingDomain, ...] = Field(default_factory=tuple)
    domain_selection: DomainSelectionShape = DomainSelectionShape.NONE

    @field_validator("domains", mode="before")
    @classmethod
    def normalize_domains(
        cls,
        value: Sequence[CreativeCodingDomain | str] | CreativeCodingDomain | str | None,
    ) -> tuple[CreativeCodingDomain, ...]:
        if value is None:
            return ()
        if isinstance(value, CreativeCodingDomain):
            return (value,)
        if isinstance(value, str):
            return (CreativeCodingDomain(value.strip()),)

        normalized: list[CreativeCodingDomain] = []
        for item in value:
            domain = (
                item
                if isinstance(item, CreativeCodingDomain)
                else CreativeCodingDomain(str(item).strip())
            )
            if domain not in normalized:
                normalized.append(domain)
        return tuple(normalized)

    @model_validator(mode="before")
    @classmethod
    def populate_legacy_domain_fields(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value

        normalized = dict(value)
        domain = normalized.get("domain")
        domains = normalized.get("domains")

        if domain is not None and not domains:
            normalized["domains"] = (domain,)

        return normalized

    @model_validator(mode="after")
    def validate_domain_alignment(self) -> PromptUserInput:
        if self.domain is None and len(self.domains) == 1:
            object.__setattr__(self, "domain", self.domains[0])

        if self.domain is not None and self.domain not in self.domains:
            raise ValueError(
                "Prompt user input domain must be included in domains "
                "when both are provided."
            )

        object.__setattr__(
            self,
            "domain_selection",
            _selection_shape_for_domains(self.domains),
        )
        return self


class PromptConversationTurnInput(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    turn_index: int = Field(ge=0)
    role: ConversationRole
    content: str = Field(min_length=1)


class PromptRunningSummaryInput(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    content: str = Field(min_length=1)
    covered_turn_count: int = Field(ge=1)


class PromptProjectMemoryInput(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    memory_kind: ProjectMemoryKind
    content: str = Field(min_length=1)
    source: str = Field(min_length=1)


class PromptMemoryInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    recent_turns: tuple[PromptConversationTurnInput, ...] = Field(default_factory=tuple)
    running_summary: PromptRunningSummaryInput | None = None
    project_memories: tuple[PromptProjectMemoryInput, ...] = Field(
        default_factory=tuple
    )


class PromptKnowledgeChunkInput(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    source_id: str = Field(min_length=1)
    domain: CreativeCodingDomain
    source_type: OfficialSourceType
    publisher: str = Field(min_length=1)
    registry_title: str = Field(min_length=1)
    document_title: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    excerpt: str = Field(min_length=1)
    score: float = Field(ge=0, le=1)


class PromptRetrievalInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunks: tuple[PromptKnowledgeChunkInput, ...] = Field(default_factory=tuple)


class PromptInputRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    route: RouteName
    assistant_request: AssistantRequest
    assembled_context: AssembledContextResponse | None = None

    @model_validator(mode="after")
    def validate_route_alignment(self) -> Self:
        if (
            self.assembled_context is not None
            and self.assembled_context.request.route != self.route
        ):
            raise ValueError(
                "Assembled context route must match the prompt-input route."
            )
        return self


class PromptInputResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    request: PromptInputRequest
    user_input: PromptUserInput
    memory_input: PromptMemoryInput | None = None
    retrieval_input: PromptRetrievalInput | None = None


class PromptInputBuilder(Protocol):
    def build(
        self,
        request: PromptInputRequest,
    ) -> PromptInputResponse:
        """Return structured prompt-ready inputs without rendering prompt text."""


class StructuredPromptInputBuilder:
    """Transform assembled context into prompt-ready structured inputs."""

    def build(
        self,
        request: PromptInputRequest,
    ) -> PromptInputResponse:
        assembled_context = request.assembled_context
        memory_input = None
        retrieval_input = None

        if (
            assembled_context is not None
            and assembled_context.memory_context is not None
        ):
            memory_context = assembled_context.memory_context
            memory_input = PromptMemoryInput(
                recent_turns=tuple(
                    PromptConversationTurnInput(
                        turn_index=turn.turn_index,
                        role=turn.role,
                        content=turn.content,
                    )
                    for turn in memory_context.recent_turns
                ),
                running_summary=(
                    PromptRunningSummaryInput(
                        content=memory_context.running_summary.content,
                        covered_turn_count=(
                            memory_context.running_summary.covered_turn_count
                        ),
                    )
                    if memory_context.running_summary is not None
                    else None
                ),
                project_memories=tuple(
                    PromptProjectMemoryInput(
                        memory_kind=memory.memory_kind,
                        content=memory.content,
                        source=memory.source,
                    )
                    for memory in memory_context.project_memories
                ),
            )

        if (
            assembled_context is not None
            and assembled_context.retrieval_context is not None
        ):
            retrieval_context = assembled_context.retrieval_context
            retrieval_input = PromptRetrievalInput(
                chunks=tuple(
                    PromptKnowledgeChunkInput(
                        source_id=chunk.source_id,
                        domain=chunk.domain,
                        source_type=chunk.source_type,
                        publisher=chunk.publisher,
                        registry_title=chunk.registry_title,
                        document_title=chunk.document_title,
                        source_url=chunk.source_url,
                        excerpt=chunk.excerpt,
                        score=chunk.score,
                    )
                    for chunk in retrieval_context.chunks
                )
            )

        prompt_input = PromptInputResponse(
            request=request,
            user_input=PromptUserInput(
                query=request.assistant_request.query,
                mode=request.assistant_request.mode,
                domain=request.assistant_request.domain,
                domains=request.assistant_request.domains,
            ),
            memory_input=memory_input,
            retrieval_input=retrieval_input,
        )
        logger.info(
            "Built prompt inputs with memory={} and retrieval={}",
            memory_input is not None,
            retrieval_input is not None,
        )
        return prompt_input


def build_prompt_input_request(
    *,
    assistant_request: AssistantRequest,
    route_decision: RouteDecision,
    assembled_context: AssembledContextResponse | None,
) -> PromptInputRequest:
    return PromptInputRequest(
        route=route_decision.route,
        assistant_request=assistant_request,
        assembled_context=assembled_context,
    )


def _selection_shape_for_domains(
    domains: tuple[CreativeCodingDomain, ...],
) -> DomainSelectionShape:
    if not domains:
        return DomainSelectionShape.NONE
    if len(domains) == 1:
        return DomainSelectionShape.SINGLE
    return DomainSelectionShape.MULTI
