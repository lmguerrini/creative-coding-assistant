"""Orchestration-facing memory boundary contracts and adapters."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Protocol

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import AssistantMode, AssistantRequest
from creative_coding_assistant.memory import (
    ConversationRole,
    ConversationSummaryRecord,
    ConversationSummaryRepository,
    ConversationTurnRecord,
    ConversationTurnRepository,
    ProjectMemoryRecord,
    ProjectMemoryRepository,
)
from creative_coding_assistant.memory.schemas import ProjectMemoryKind
from creative_coding_assistant.orchestration.routing import (
    RouteCapability,
    RouteDecision,
    RouteName,
)

DEFAULT_RECENT_TURN_LIMIT = 6


class MemoryContextSource(StrEnum):
    CHROMA_MEMORY = "chroma_memory"


class RecentConversationTurn(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    turn_index: int = Field(ge=0)
    role: ConversationRole
    content: str = Field(min_length=1)
    created_at: datetime
    mode: AssistantMode | None = None


class ConversationSummaryContext(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    content: str = Field(min_length=1)
    created_at: datetime
    covered_turn_count: int = Field(ge=1)


class ProjectMemoryContext(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    content: str = Field(min_length=1)
    created_at: datetime
    memory_kind: ProjectMemoryKind
    source: str = Field(min_length=1)


class MemoryContextRequest(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    route: RouteName
    conversation_id: str | None = None
    project_id: str | None = None
    recent_turn_limit: int = Field(default=DEFAULT_RECENT_TURN_LIMIT, ge=1, le=20)
    include_running_summary: bool = True
    include_project_memory: bool = True


class MemoryContextResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    request: MemoryContextRequest
    source: MemoryContextSource = MemoryContextSource.CHROMA_MEMORY
    recent_turns: tuple[RecentConversationTurn, ...] = Field(default_factory=tuple)
    running_summary: ConversationSummaryContext | None = None
    project_memories: tuple[ProjectMemoryContext, ...] = Field(default_factory=tuple)


class MemoryGateway(Protocol):
    def retrieve_context(
        self,
        request: MemoryContextRequest,
    ) -> MemoryContextResponse:
        """Return orchestration-facing memory context for a single request."""


class ChromaMemoryAdapter:
    """Adapt memory repositories into orchestration-facing context results."""

    def __init__(
        self,
        *,
        turn_repository: ConversationTurnRepository,
        summary_repository: ConversationSummaryRepository,
        project_memory_repository: ProjectMemoryRepository,
    ) -> None:
        self._turn_repository = turn_repository
        self._summary_repository = summary_repository
        self._project_memory_repository = project_memory_repository

    def retrieve_context(
        self,
        request: MemoryContextRequest,
    ) -> MemoryContextResponse:
        recent_turns = self._load_recent_turns(request)
        running_summary = self._load_running_summary(request)
        project_memories = self._load_project_memories(request)
        logger.info(
            "Built orchestration memory context with {} turn(s), summary={}, "
            "{} project memory item(s)",
            len(recent_turns),
            running_summary is not None,
            len(project_memories),
        )
        return MemoryContextResponse(
            request=request,
            recent_turns=recent_turns,
            running_summary=running_summary,
            project_memories=project_memories,
        )

    def _load_recent_turns(
        self,
        request: MemoryContextRequest,
    ) -> tuple[RecentConversationTurn, ...]:
        if request.conversation_id is None:
            return ()
        turns = self._turn_repository.list_recent(
            conversation_id=request.conversation_id,
            limit=request.recent_turn_limit,
        )
        return tuple(_recent_turn_from_record(turn) for turn in turns)

    def _load_running_summary(
        self,
        request: MemoryContextRequest,
    ) -> ConversationSummaryContext | None:
        if not request.include_running_summary or request.conversation_id is None:
            return None
        summary = self._summary_repository.get_latest(
            conversation_id=request.conversation_id
        )
        if summary is None:
            return None
        return _summary_context_from_record(summary)

    def _load_project_memories(
        self,
        request: MemoryContextRequest,
    ) -> tuple[ProjectMemoryContext, ...]:
        if not request.include_project_memory or request.project_id is None:
            return ()
        memories = self._project_memory_repository.list(project_id=request.project_id)
        return tuple(_project_memory_context_from_record(memory) for memory in memories)


def build_memory_context_request(
    assistant_request: AssistantRequest,
    route_decision: RouteDecision,
    *,
    recent_turn_limit: int = DEFAULT_RECENT_TURN_LIMIT,
) -> MemoryContextRequest | None:
    if RouteCapability.MEMORY_CONTEXT not in route_decision.capabilities:
        return None
    if (
        assistant_request.conversation_id is None
        and assistant_request.project_id is None
    ):
        return None

    return MemoryContextRequest(
        route=route_decision.route,
        conversation_id=assistant_request.conversation_id,
        project_id=assistant_request.project_id,
        recent_turn_limit=recent_turn_limit,
        include_running_summary=assistant_request.conversation_id is not None,
        include_project_memory=assistant_request.project_id is not None,
    )


def _recent_turn_from_record(record: ConversationTurnRecord) -> RecentConversationTurn:
    return RecentConversationTurn(
        turn_index=record.turn_index,
        role=record.role,
        content=record.content,
        created_at=record.created_at,
        mode=record.mode,
    )


def _summary_context_from_record(
    record: ConversationSummaryRecord,
) -> ConversationSummaryContext:
    return ConversationSummaryContext(
        content=record.content,
        created_at=record.created_at,
        covered_turn_count=record.covered_turn_count,
    )


def _project_memory_context_from_record(
    record: ProjectMemoryRecord,
) -> ProjectMemoryContext:
    return ProjectMemoryContext(
        content=record.content,
        created_at=record.created_at,
        memory_kind=record.memory_kind,
        source=record.source,
    )
