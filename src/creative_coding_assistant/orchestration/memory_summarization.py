"""V5.1 memory summarization contracts for existing memory context."""

from __future__ import annotations

import re
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.memory import (
    MemoryContextResponse,
    ProjectMemoryContext,
    RecentConversationTurn,
)

MemorySummarySourceKind = Literal[
    "recent_turns",
    "running_summary",
    "project_memory",
]
MemorySummarizationStatus = Literal["unchanged", "summarized"]
MemorySummarizationPressure = Literal["low", "medium", "high"]

MEMORY_SUMMARY_SEGMENT_SERIALIZATION_VERSION = "memory_summary_segment.v1"
MEMORY_SUMMARIZATION_RESULT_SERIALIZATION_VERSION = "memory_summarization_result.v1"
MEMORY_SUMMARIZATION_AUTHORITY_BOUNDARY = (
    "Memory summarization produces a separate summary artifact from existing "
    "memory context only; it does not write memory, replace running summaries, "
    "mutate conversation turns, mutate project memory, query memory storage, "
    "route providers or models, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "memory_storage_write",
    "running_summary_replacement",
    "conversation_turn_mutation",
    "project_memory_mutation",
    "memory_query_execution",
    "context_routing",
    "provider_or_model_routing",
    "persistent_storage_write",
    "generated_output_modification",
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


class MemorySummarySegment(BaseModel):
    """One summary segment derived from existing memory context."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    segment_id: str = Field(min_length=1, max_length=180)
    source_kind: MemorySummarySourceKind
    source_id: str = Field(min_length=1, max_length=180)
    turn_indices: tuple[int, ...] = Field(default_factory=tuple, max_length=40)
    project_memory_kind: str | None = Field(default=None, max_length=80)
    original_text: str = Field(min_length=1, max_length=120_000)
    summary_text: str = Field(min_length=1, max_length=120_000)
    original_token_estimate: int = Field(ge=1, le=240_000)
    summary_token_estimate: int = Field(ge=1, le=240_000)
    saved_tokens: int = Field(ge=0, le=240_000)
    summarization_status: MemorySummarizationStatus
    summarization_pressure: MemorySummarizationPressure
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    memory_summarization_implemented: Literal[True] = True
    memory_storage_write_implemented: Literal[False] = False
    running_summary_replacement_implemented: Literal[False] = False
    conversation_turn_mutation_implemented: Literal[False] = False
    project_memory_mutation_implemented: Literal[False] = False
    memory_query_execution_implemented: Literal[False] = False
    context_routing_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["memory_summary_segment.v1"] = (
        MEMORY_SUMMARY_SEGMENT_SERIALIZATION_VERSION
    )
    summarization_only: Literal[True] = True

    @model_validator(mode="after")
    def _segment_matches_summary(self) -> Self:
        if self.summary_token_estimate > self.original_token_estimate:
            raise ValueError(
                "summary_token_estimate must not exceed original_token_estimate"
            )
        if self.saved_tokens != (
            self.original_token_estimate - self.summary_token_estimate
        ):
            raise ValueError("saved_tokens must match token estimate delta")
        expected_status = "summarized" if self.saved_tokens > 0 else "unchanged"
        if self.summarization_status != expected_status:
            raise ValueError("summarization_status must match saved tokens")
        if self.summarization_status == "unchanged" and (
            self.summary_text != self.original_text
        ):
            raise ValueError("unchanged segments must preserve original text")
        return self


class MemorySummarizationResult(BaseModel):
    """Bounded V5.1 memory summary artifact for existing memory context."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["memory_summarizer"] = "memory_summarizer"
    serialization_version: Literal["memory_summarization_result.v1"] = (
        MEMORY_SUMMARIZATION_RESULT_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=MEMORY_SUMMARIZATION_AUTHORITY_BOUNDARY,
        max_length=1200,
    )
    source_route: str = Field(min_length=1, max_length=80)
    source_recent_turn_count: int = Field(ge=0, le=10_000)
    source_project_memory_count: int = Field(ge=0, le=10_000)
    source_has_running_summary: bool
    segments: tuple[MemorySummarySegment, ...] = Field(min_length=1, max_length=40)
    segment_ids: tuple[str, ...] = Field(min_length=1, max_length=40)
    target_token_budget: int = Field(ge=1, le=240_000)
    original_total_tokens: int = Field(ge=1, le=240_000)
    summary_total_tokens: int = Field(ge=1, le=240_000)
    saved_total_tokens: int = Field(ge=0, le=240_000)
    within_budget: bool
    summarization_pressure: MemorySummarizationPressure
    summary_text: str = Field(min_length=1, max_length=240_000)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    memory_summarization_implemented: Literal[True] = True
    memory_storage_write_implemented: Literal[False] = False
    running_summary_replacement_implemented: Literal[False] = False
    conversation_turn_mutation_implemented: Literal[False] = False
    project_memory_mutation_implemented: Literal[False] = False
    memory_query_execution_implemented: Literal[False] = False
    context_routing_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    summarization_only: Literal[True] = True

    @model_validator(mode="after")
    def _result_matches_segments(self) -> Self:
        derived_segment_ids = tuple(segment.segment_id for segment in self.segments)
        if len(set(derived_segment_ids)) != len(derived_segment_ids):
            raise ValueError("segment_ids must be unique")
        if self.segment_ids != derived_segment_ids:
            raise ValueError("segment_ids must match segments")
        original_total = sum(segment.original_token_estimate for segment in self.segments)
        summary_total = sum(segment.summary_token_estimate for segment in self.segments)
        saved_total = sum(segment.saved_tokens for segment in self.segments)
        if self.original_total_tokens != original_total:
            raise ValueError("original_total_tokens must match segments")
        if self.summary_total_tokens != summary_total:
            raise ValueError("summary_total_tokens must match segments")
        if self.saved_total_tokens != saved_total:
            raise ValueError("saved_total_tokens must match segments")
        if self.saved_total_tokens != (
            self.original_total_tokens - self.summary_total_tokens
        ):
            raise ValueError("saved_total_tokens must match token delta")
        if self.within_budget != (self.summary_total_tokens <= self.target_token_budget):
            raise ValueError("within_budget must match summary token total")
        if self.summary_text != _join_summary_segments(self.segments):
            raise ValueError("summary_text must match segments")
        return self


def summarize_memory_context(
    memory_context: MemoryContextResponse,
    *,
    target_token_budget: int = 1_200,
) -> MemorySummarizationResult:
    """Summarize existing memory context without storage writes."""

    if target_token_budget <= 0:
        raise ValueError("target_token_budget must be positive")
    segments = _input_segments(memory_context)
    if not segments:
        raise ValueError("memory summarization requires at least one memory source")

    original_total = sum(_estimate_tokens(segment.original_text) for segment in segments)
    budgets = _segment_token_budgets(
        segments,
        target_token_budget=target_token_budget,
        original_total=original_total,
    )
    summarized_segments = tuple(
        _summarize_segment(segment, budgets[segment.segment_id])
        for segment in segments
    )
    summary_total = sum(
        segment.summary_token_estimate for segment in summarized_segments
    )
    saved_total = sum(segment.saved_tokens for segment in summarized_segments)

    return MemorySummarizationResult(
        source_route=memory_context.request.route.value,
        source_recent_turn_count=len(memory_context.recent_turns),
        source_project_memory_count=len(memory_context.project_memories),
        source_has_running_summary=memory_context.running_summary is not None,
        segments=summarized_segments,
        segment_ids=tuple(segment.segment_id for segment in summarized_segments),
        target_token_budget=target_token_budget,
        original_total_tokens=original_total,
        summary_total_tokens=summary_total,
        saved_total_tokens=saved_total,
        within_budget=summary_total <= target_token_budget,
        summarization_pressure=_summarization_pressure(
            original_total=original_total,
            summary_total=summary_total,
            target_token_budget=target_token_budget,
        ),
        summary_text=_join_summary_segments(summarized_segments),
        advisory_actions=_result_actions(saved_total),
    )


def memory_summary_segment_by_id(
    segment_id: str,
    result: MemorySummarizationResult | None = None,
) -> MemorySummarySegment | None:
    """Return one summary segment without reading or writing memory storage."""

    source_result = result or summarize_memory_context(_placeholder_memory_context())
    for segment in source_result.segments:
        if segment.segment_id == segment_id:
            return segment
    return None


def memory_summary_segments_for_kind(
    source_kind: MemorySummarySourceKind,
    result: MemorySummarizationResult | None = None,
) -> tuple[MemorySummarySegment, ...]:
    """Return summary segments by source kind without mutating memory."""

    source_result = result or summarize_memory_context(_placeholder_memory_context())
    return tuple(
        segment for segment in source_result.segments if segment.source_kind == source_kind
    )


class _InputSegment(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    segment_id: str
    source_kind: MemorySummarySourceKind
    source_id: str
    turn_indices: tuple[int, ...] = ()
    project_memory_kind: str | None = None
    original_text: str


def _input_segments(memory_context: MemoryContextResponse) -> tuple[_InputSegment, ...]:
    segments: list[_InputSegment] = []
    if memory_context.recent_turns:
        segments.append(_recent_turns_segment(memory_context.recent_turns))
    if memory_context.running_summary is not None:
        segments.append(
            _InputSegment(
                segment_id="memory_summary::running_summary",
                source_kind="running_summary",
                source_id="memory_context.running_summary",
                original_text=memory_context.running_summary.content,
            )
        )
    segments.extend(
        _project_memory_segment(index, memory)
        for index, memory in enumerate(memory_context.project_memories)
    )
    return tuple(segments)


def _recent_turns_segment(
    turns: tuple[RecentConversationTurn, ...],
) -> _InputSegment:
    return _InputSegment(
        segment_id="memory_summary::recent_turns",
        source_kind="recent_turns",
        source_id="memory_context.recent_turns",
        turn_indices=tuple(turn.turn_index for turn in turns),
        original_text="\n".join(
            f"{turn.turn_index}:{turn.role.value}: {turn.content}" for turn in turns
        ),
    )


def _project_memory_segment(
    index: int,
    memory: ProjectMemoryContext,
) -> _InputSegment:
    return _InputSegment(
        segment_id=f"memory_summary::project_memory::{index}",
        source_kind="project_memory",
        source_id=f"memory_context.project_memories.{index}",
        project_memory_kind=memory.memory_kind.value,
        original_text=memory.content,
    )


def _summarize_segment(
    segment: _InputSegment,
    target_tokens: int,
) -> MemorySummarySegment:
    original = segment.original_text
    original_tokens = _estimate_tokens(original)
    summary = (
        original
        if original_tokens <= target_tokens
        else _summarize_text_to_budget(original, target_tokens)
    )
    summary_tokens = _estimate_tokens(summary)
    if summary_tokens >= original_tokens:
        summary = original
        summary_tokens = original_tokens
    saved_tokens = original_tokens - summary_tokens

    return MemorySummarySegment(
        segment_id=segment.segment_id,
        source_kind=segment.source_kind,
        source_id=segment.source_id,
        turn_indices=segment.turn_indices,
        project_memory_kind=segment.project_memory_kind,
        original_text=original,
        summary_text=summary,
        original_token_estimate=original_tokens,
        summary_token_estimate=summary_tokens,
        saved_tokens=saved_tokens,
        summarization_status="summarized" if saved_tokens else "unchanged",
        summarization_pressure=_segment_pressure(
            original_tokens=original_tokens,
            summary_tokens=summary_tokens,
            target_tokens=target_tokens,
        ),
        evidence=(
            f"source_kind:{segment.source_kind}",
            f"source_id:{segment.source_id}",
            f"target_tokens:{target_tokens}",
            f"summary_tokens:{summary_tokens}",
        ),
        advisory_actions=_segment_actions(saved_tokens),
    )


def _summarize_text_to_budget(text: str, target_tokens: int) -> str:
    normalized_lines = tuple(
        " ".join(line.strip().split())
        for line in text.splitlines()
        if line.strip()
    )
    normalized = "\n".join(normalized_lines) if normalized_lines else " ".join(text.split())
    if _estimate_tokens(normalized) <= target_tokens:
        return normalized

    marker = "[summarized: memory detail omitted]"
    character_budget = max(24, target_tokens * 4 - len(marker) - 1)
    sentences = tuple(
        sentence.strip()
        for sentence in _SENTENCE_BOUNDARY.split(normalized)
        if sentence.strip()
    )
    selected: list[str] = []
    used = 0
    for sentence in sentences:
        separator = "\n" if selected else ""
        next_length = used + len(separator) + len(sentence)
        if next_length > character_budget:
            break
        selected.append(sentence)
        used = next_length

    if not selected:
        selected_text = normalized[:character_budget].rstrip()
    else:
        selected_text = "\n".join(selected).rstrip()
    return f"{selected_text}\n{marker}".strip()


def _segment_token_budgets(
    segments: tuple[_InputSegment, ...],
    *,
    target_token_budget: int,
    original_total: int,
) -> dict[str, int]:
    if original_total <= target_token_budget:
        return {
            segment.segment_id: _estimate_tokens(segment.original_text)
            for segment in segments
        }

    remaining = target_token_budget
    budgets: dict[str, int] = {}
    for index, segment in enumerate(segments):
        original_tokens = _estimate_tokens(segment.original_text)
        if index == len(segments) - 1:
            budget = max(1, remaining)
        else:
            proportional = max(1, target_token_budget * original_tokens // original_total)
            budget = min(original_tokens, proportional)
        budgets[segment.segment_id] = max(1, budget)
        remaining = max(0, remaining - budgets[segment.segment_id])
    return budgets


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _segment_pressure(
    *,
    original_tokens: int,
    summary_tokens: int,
    target_tokens: int,
) -> MemorySummarizationPressure:
    if summary_tokens > target_tokens:
        return "high"
    if summary_tokens < original_tokens:
        return "medium"
    return "low"


def _summarization_pressure(
    *,
    original_total: int,
    summary_total: int,
    target_token_budget: int,
) -> MemorySummarizationPressure:
    if summary_total > target_token_budget:
        return "high"
    if summary_total < original_total:
        return "medium"
    return "low"


def _segment_actions(saved_tokens: int) -> tuple[str, ...]:
    actions = [
        "Produce memory summary as a separate artifact.",
        "Preserve source memory context for auditability.",
    ]
    if saved_tokens:
        actions.append("Record summary savings without writing memory storage.")
    return tuple(actions)


def _result_actions(saved_total: int) -> tuple[str, ...]:
    actions = [
        "Expose memory summaries only through this summarization result.",
        "Preserve memory storage, provider routing, and output boundaries.",
    ]
    if saved_total:
        actions.append("Use summary artifact only when explicitly selected.")
    else:
        actions.append("Keep original memory context because it fits the budget.")
    return tuple(actions)


def _join_summary_segments(
    segments: tuple[MemorySummarySegment, ...],
) -> str:
    return "\n\n".join(
        f"[memory:{segment.source_kind}:{segment.source_id}]\n{segment.summary_text}"
        for segment in segments
    )


def _placeholder_memory_context() -> MemoryContextResponse:
    from datetime import UTC, datetime

    from creative_coding_assistant.contracts import AssistantMode
    from creative_coding_assistant.memory import ConversationRole
    from creative_coding_assistant.orchestration.memory import MemoryContextRequest
    from creative_coding_assistant.orchestration.routing import RouteName

    return MemoryContextResponse(
        request=MemoryContextRequest(route=RouteName.EXPLAIN),
        recent_turns=(
            RecentConversationTurn(
                turn_index=0,
                role=ConversationRole.USER,
                content="Memory summarization placeholder.",
                created_at=datetime(2026, 6, 28, tzinfo=UTC),
                mode=AssistantMode.EXPLAIN,
            ),
        ),
    )
