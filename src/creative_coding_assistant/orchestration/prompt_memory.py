"""Prompt memory compaction and session summary helpers."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.memory import ConversationRole
from creative_coding_assistant.orchestration.memory import RecentConversationTurn
from creative_coding_assistant.rag.retrieval.domain_intent import (
    detect_explicit_query_domains,
)


class PromptConversationTurnInput(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    turn_index: int = Field(ge=0)
    role: ConversationRole
    content: str = Field(min_length=1)


class PromptSessionMemorySummaryInput(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    summary: str = Field(min_length=1)
    detected_domain: CreativeCodingDomain | None = None
    code_present: bool = False


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
_MAX_SESSION_MEMORY_SUMMARIES = 5
_CODE_SIGNAL_PATTERNS = (
    re.compile(r"```"),
    re.compile(r"\bfunction\s+\w+\s*\("),
    re.compile(r"\b(?:const|let|var)\s+\w+"),
    re.compile(r"\bcreateCanvas\s*\("),
    re.compile(r"\bellipse\s*\("),
    re.compile(r"\brect\s*\("),
    re.compile(r"\bbackground\s*\("),
    re.compile(r"\bTHREE\."),
    re.compile(r"\bgl_FragColor\b"),
    re.compile(r"@react-three/fiber"),
)
_ACTION_PATTERNS = (
    ("convert", re.compile(r"\bconvert\b")),
    ("debug", re.compile(r"\b(?:debug|fix|error|broken|issue)\b")),
    ("explain", re.compile(r"\b(?:explain|why|how|what)\b")),
    ("create", re.compile(r"\b(?:create|build|generate|make|sketch)\b")),
)
_TOPIC_PATTERNS = (
    ("bouncing ball", re.compile(r"\bbouncing ball\b")),
    ("rotating cube", re.compile(r"\brotating cube\b")),
    ("orbitcontrols setup", re.compile(r"\borbitcontrols\b")),
    ("shader", re.compile(r"\bshader\b")),
    ("sketch", re.compile(r"\bsketch\b")),
)


def looks_like_follow_up_query(query: str) -> bool:
    normalized = _collapse_whitespace(query).lower()
    if _FOLLOW_UP_COMMAND_PATTERN.search(normalized):
        return True
    return any(
        pattern.search(normalized) is not None
        for pattern in _FOLLOW_UP_PHRASE_PATTERNS
    )


def build_prompt_recent_turns(
    *,
    query: str,
    recent_turns: tuple[RecentConversationTurn, ...],
) -> tuple[PromptConversationTurnInput, ...]:
    if not looks_like_follow_up_query(query):
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


def build_session_memory_summaries(
    recent_turns: tuple[RecentConversationTurn, ...],
) -> tuple[PromptSessionMemorySummaryInput, ...]:
    summaries: list[PromptSessionMemorySummaryInput] = []
    fallback_domain: CreativeCodingDomain | None = None

    for turn in recent_turns:
        detected_domain = _detect_summary_domain(
            turn.content,
            fallback_domain=fallback_domain,
        )
        code_present = _has_code_signal(turn.content)
        summary = _summarize_turn(
            turn,
            detected_domain=detected_domain,
            code_present=code_present,
        )
        if summary is None:
            continue

        summaries.append(
            PromptSessionMemorySummaryInput(
                summary=summary,
                detected_domain=detected_domain,
                code_present=code_present,
            )
        )
        if detected_domain is not None:
            fallback_domain = detected_domain

    return tuple(summaries[-_MAX_SESSION_MEMORY_SUMMARIES:])


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


def _detect_summary_domain(
    content: str,
    *,
    fallback_domain: CreativeCodingDomain | None,
) -> CreativeCodingDomain | None:
    detected_domains = detect_explicit_query_domains(content)
    if detected_domains:
        return detected_domains[0]
    return fallback_domain


def _has_code_signal(content: str) -> bool:
    return any(pattern.search(content) is not None for pattern in _CODE_SIGNAL_PATTERNS)


def _summarize_turn(
    turn: RecentConversationTurn,
    *,
    detected_domain: CreativeCodingDomain | None,
    code_present: bool,
) -> str | None:
    action = _detect_summary_action(turn.content)
    topic = _detect_summary_topic(turn.content)
    domain_label = _domain_summary_label(detected_domain)

    if turn.role is ConversationRole.USER:
        if action == "convert" and domain_label is not None:
            return f"User asked to convert the project to {domain_label}."
        if action == "debug":
            return _compose_summary(
                prefix="User asked to debug",
                topic=topic,
                domain_label=domain_label,
            )
        if action == "explain":
            if topic is not None:
                return _compose_summary(
                    prefix="User asked for",
                    topic=f"{topic} explanation",
                    domain_label=domain_label,
                )
            return _compose_summary(
                prefix="User asked for",
                topic="explanation",
                domain_label=domain_label,
            )
        if action == "create":
            return _compose_summary(
                prefix="User requested",
                topic=topic or "project work",
                domain_label=domain_label,
            )
        return _compose_summary(
            prefix="User continued",
            topic=topic or "the project",
            domain_label=domain_label,
        )

    if code_present:
        return _compose_summary(
            prefix="Assistant generated",
            topic=topic or _default_generated_topic(detected_domain),
            domain_label=domain_label,
        )
    if action == "debug":
        return _compose_summary(
            prefix="Assistant discussed debugging",
            topic=topic,
            domain_label=domain_label,
        )
    if action == "explain":
        return _compose_summary(
            prefix="Assistant explained",
            topic=topic or "the concept",
            domain_label=domain_label,
        )
    return _compose_summary(
        prefix="Assistant responded with",
        topic=topic or "implementation guidance",
        domain_label=domain_label,
    )


def _detect_summary_action(content: str) -> str | None:
    normalized = content.lower()
    for label, pattern in _ACTION_PATTERNS:
        if pattern.search(normalized) is not None:
            return label
    return None


def _detect_summary_topic(content: str) -> str | None:
    normalized = content.lower()
    for label, pattern in _TOPIC_PATTERNS:
        if pattern.search(normalized) is not None:
            return label
    return None


def _compose_summary(
    *,
    prefix: str,
    topic: str | None,
    domain_label: str | None,
) -> str:
    if topic is not None and domain_label is not None:
        return f"{prefix} {domain_label} {topic}."
    if topic is not None:
        return f"{prefix} {topic}."
    if domain_label is not None:
        return f"{prefix} {domain_label} work."
    return f"{prefix} the current task."


def _domain_summary_label(domain: CreativeCodingDomain | None) -> str | None:
    if domain is None:
        return None
    if domain is CreativeCodingDomain.THREE_JS:
        return "Three.js"
    if domain is CreativeCodingDomain.REACT_THREE_FIBER:
        return "React Three Fiber"
    if domain is CreativeCodingDomain.P5_JS:
        return "p5.js"
    return "GLSL"


def _default_generated_topic(domain: CreativeCodingDomain | None) -> str:
    if domain is CreativeCodingDomain.P5_JS:
        return "sketch"
    if domain is CreativeCodingDomain.GLSL:
        return "shader code"
    if domain is CreativeCodingDomain.REACT_THREE_FIBER:
        return "component code"
    if domain is CreativeCodingDomain.THREE_JS:
        return "scene code"
    return "code"
