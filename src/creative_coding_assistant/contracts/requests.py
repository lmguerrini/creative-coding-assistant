"""Assistant request and response contracts."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from creative_coding_assistant.contracts.events import StreamEvent


class CreativeCodingDomain(StrEnum):
    THREE_JS = "three_js"
    REACT_THREE_FIBER = "react_three_fiber"
    P5_JS = "p5_js"
    GLSL = "glsl"


class AssistantMode(StrEnum):
    GENERATE = "generate"
    EXPLAIN = "explain"
    DEBUG = "debug"
    DESIGN = "design"
    REVIEW = "review"
    PREVIEW = "preview"


class AssistantRequest(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    query: str = Field(min_length=1)
    conversation_id: str | None = None
    project_id: str | None = None
    domain: CreativeCodingDomain | None = None
    mode: AssistantMode = AssistantMode.GENERATE

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Assistant request query must not be empty.")
        return value


class AssistantResponse(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    answer: str = Field(min_length=1)
    events: tuple[StreamEvent, ...] = Field(default_factory=tuple)
