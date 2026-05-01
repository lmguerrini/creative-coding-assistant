"""Prompt-input contracts and transformation boundaries."""

from __future__ import annotations

import re
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
from creative_coding_assistant.orchestration.memory import RecentConversationTurn
from creative_coding_assistant.orchestration.routing import (
    DomainSelectionShape,
    RouteDecision,
    RouteName,
)
from creative_coding_assistant.rag.retrieval.domain_intent import (
    detect_explicit_query_domains,
    resolve_effective_query_domains,
)
from creative_coding_assistant.rag.sources import OfficialSourceType


class PromptUserInput(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    query: str = Field(min_length=1)
    mode: AssistantMode
    domain: CreativeCodingDomain | None = None
    domains: tuple[CreativeCodingDomain, ...] = Field(default_factory=tuple)
    ui_selected_domains: tuple[CreativeCodingDomain, ...] = Field(
        default_factory=tuple
    )
    detected_domains: tuple[CreativeCodingDomain, ...] = Field(default_factory=tuple)
    effective_domains: tuple[CreativeCodingDomain, ...] = Field(default_factory=tuple)
    domain_selection: DomainSelectionShape = DomainSelectionShape.NONE
    is_follow_up: bool = False

    @field_validator(
        "domains",
        "ui_selected_domains",
        "detected_domains",
        "effective_domains",
        mode="before",
    )
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
        ui_selected_domains = normalized.get("ui_selected_domains")
        effective_domains = normalized.get("effective_domains")

        if domains and not ui_selected_domains:
            normalized["ui_selected_domains"] = domains

        if domains and not effective_domains:
            normalized["effective_domains"] = domains

        if effective_domains and not domains:
            normalized["domains"] = effective_domains

        if domain is not None and not normalized.get("domains"):
            normalized["domains"] = (domain,)
        if domain is not None and not normalized.get("ui_selected_domains"):
            normalized["ui_selected_domains"] = (domain,)
        if domain is not None and not normalized.get("effective_domains"):
            normalized["effective_domains"] = (domain,)

        return normalized

    @model_validator(mode="after")
    def validate_domain_alignment(self) -> PromptUserInput:
        if not self.domains and self.effective_domains:
            object.__setattr__(self, "domains", self.effective_domains)

        if not self.effective_domains and self.domains:
            object.__setattr__(self, "effective_domains", self.domains)

        if not self.ui_selected_domains and self.domains:
            object.__setattr__(self, "ui_selected_domains", self.domains)

        if self.domain is None and len(self.effective_domains) == 1:
            object.__setattr__(self, "domain", self.effective_domains[0])

        if self.domain is not None and self.domain not in self.effective_domains:
            raise ValueError(
                "Prompt user input domain must be included in domains "
                "when both are provided."
            )

        object.__setattr__(
            self,
            "domain_selection",
            _selection_shape_for_domains(self.effective_domains),
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
        ui_selected_domains = request.assistant_request.domains
        detected_domains = detect_explicit_query_domains(
            request.assistant_request.query
        )
        effective_domains = resolve_effective_query_domains(
            query=request.assistant_request.query,
            selected_domains=ui_selected_domains,
        )
        is_follow_up = _looks_like_follow_up_query(request.assistant_request.query)
        memory_input = None
        retrieval_input = None

        if (
            assembled_context is not None
            and assembled_context.memory_context is not None
        ):
            memory_context = assembled_context.memory_context
            memory_input = PromptMemoryInput(
                recent_turns=_build_prompt_recent_turns(
                    query=request.assistant_request.query,
                    recent_turns=memory_context.recent_turns,
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
                domain=(
                    effective_domains[0]
                    if len(effective_domains) == 1
                    else None
                ),
                domains=effective_domains,
                ui_selected_domains=ui_selected_domains,
                detected_domains=detected_domains,
                effective_domains=effective_domains,
                is_follow_up=is_follow_up,
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


_FOLLOW_UP_PHRASE_PATTERNS = (
    re.compile(r"\buse the previous\b"),
    re.compile(r"\bsame code\b"),
    re.compile(r"\bprevious code\b"),
    re.compile(r"\bcontinue from the previous\b"),
    re.compile(r"\bcontinue from previous\b"),
    re.compile(r"\bmodify it\b"),
    re.compile(r"\bmake it\b"),
    re.compile(r"\bchange it\b"),
)
_FOLLOW_UP_COMMAND_PATTERN = re.compile(
    r"^(?:now\s+)?(?:continue|modify|change|make|add|remove|convert)\b"
)
_FENCED_CODE_BLOCK_PATTERN = re.compile(
    r"```(?P<language>[^\n`]*)\n(?P<code>.*?)```",
    re.DOTALL,
)
_MAX_USER_TURN_CHARS = 220
_MAX_ASSISTANT_TEXT_CHARS = 280
_MAX_CODE_BLOCK_CHARS = 700
_CODE_BLOCK_HEAD_CHARS = 380
_CODE_BLOCK_TAIL_CHARS = 280


def _looks_like_follow_up_query(query: str) -> bool:
    normalized = _collapse_whitespace(query).lower()
    if _FOLLOW_UP_COMMAND_PATTERN.search(normalized):
        return True
    return any(
        pattern.search(normalized) is not None
        for pattern in _FOLLOW_UP_PHRASE_PATTERNS
    )


def _build_prompt_recent_turns(
    *,
    query: str,
    recent_turns: tuple[RecentConversationTurn, ...],
) -> tuple[PromptConversationTurnInput, ...]:
    if not _looks_like_follow_up_query(query):
        return ()

    selected_turns = _select_recent_turn_pair(recent_turns)
    return tuple(
        PromptConversationTurnInput(
            turn_index=turn.turn_index,
            role=turn.role,
            content=_compact_turn_content(turn),
        )
        for turn in selected_turns
    )


def _select_recent_turn_pair(
    recent_turns: tuple[RecentConversationTurn, ...],
) -> tuple[RecentConversationTurn, ...]:
    if not recent_turns:
        return ()

    assistant_index = None
    for index in range(len(recent_turns) - 1, -1, -1):
        if recent_turns[index].role is ConversationRole.ASSISTANT:
            assistant_index = index
            break

    if assistant_index is None:
        return recent_turns[-1:]

    for index in range(assistant_index - 1, -1, -1):
        if recent_turns[index].role is ConversationRole.USER:
            return recent_turns[index : assistant_index + 1]

    return recent_turns[assistant_index : assistant_index + 1]


def _compact_turn_content(turn: RecentConversationTurn) -> str:
    if turn.role is ConversationRole.USER:
        return _truncate_text(_collapse_whitespace(turn.content), _MAX_USER_TURN_CHARS)
    return _compact_assistant_turn_content(turn.content)


def _compact_assistant_turn_content(content: str) -> str:
    code_match = _select_primary_code_block(content)
    if code_match is None:
        return _truncate_text(_collapse_whitespace(content), _MAX_ASSISTANT_TEXT_CHARS)

    prose = _truncate_text(
        _collapse_whitespace(_FENCED_CODE_BLOCK_PATTERN.sub(" ", content)),
        _MAX_ASSISTANT_TEXT_CHARS,
    )
    language = code_match.group("language").strip()
    code = code_match.group("code").strip()
    compact_code = _truncate_code_excerpt(code)
    if language:
        code_fence = f"```{language}\n{compact_code}\n```"
    else:
        code_fence = f"```\n{compact_code}\n```"

    if prose:
        return f"{prose}\nRelevant code excerpt:\n{code_fence}"
    return f"Relevant code excerpt:\n{code_fence}"


def _select_primary_code_block(content: str) -> re.Match[str] | None:
    matches = list(_FENCED_CODE_BLOCK_PATTERN.finditer(content))
    if not matches:
        return None
    return max(matches, key=lambda match: len(match.group("code").strip()))


def _collapse_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _truncate_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    truncated = value[: limit - 1].rstrip()
    return f"{truncated}..."


def _truncate_code_excerpt(code: str) -> str:
    if len(code) <= _MAX_CODE_BLOCK_CHARS:
        return code

    head = code[:_CODE_BLOCK_HEAD_CHARS].rstrip()
    tail = code[-_CODE_BLOCK_TAIL_CHARS :].lstrip()
    return f"{head}\n...\n{tail}"
