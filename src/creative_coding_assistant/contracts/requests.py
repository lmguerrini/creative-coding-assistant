"""Assistant request and response contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from creative_coding_assistant.contracts.events import StreamEvent


class CreativeCodingDomain(str, Enum):
    THREE_JS = "three_js"
    REACT_THREE_FIBER = "react_three_fiber"
    P5_JS = "p5_js"
    GLSL = "glsl"


class AssistantMode(str, Enum):
    GENERATE = "generate"
    EXPLAIN = "explain"
    DEBUG = "debug"
    DESIGN = "design"
    REVIEW = "review"
    PREVIEW = "preview"


@dataclass(frozen=True)
class AssistantRequest:
    query: str
    conversation_id: str | None = None
    project_id: str | None = None
    domain: CreativeCodingDomain | None = None
    mode: AssistantMode = AssistantMode.GENERATE

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("Assistant request query must not be empty.")


@dataclass(frozen=True)
class AssistantResponse:
    answer: str
    events: tuple[StreamEvent, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.answer.strip():
            raise ValueError("Assistant response answer must not be empty.")
