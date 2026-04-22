"""Pydantic schemas for durable memory records."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from creative_coding_assistant.contracts import AssistantMode, CreativeCodingDomain


class ConversationRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class ProjectMemoryKind(StrEnum):
    GOAL = "goal"
    PREFERENCE = "preference"
    CONSTRAINT = "constraint"
    DECISION = "decision"
    STYLE = "style"
    TECHNICAL_NOTE = "technical_note"


class MemoryRecordBase(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    content: str = Field(min_length=1)
    created_at: datetime
    project_id: str | None = None
    domain: CreativeCodingDomain | None = None

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Memory timestamps must be timezone-aware.")
        return value


class EmbeddedMemoryRecordBase(MemoryRecordBase):
    embedding: list[float] = Field(min_length=1)


class ConversationTurnRecord(MemoryRecordBase):
    conversation_id: str = Field(min_length=1)
    turn_index: int = Field(ge=0)
    role: ConversationRole
    mode: AssistantMode | None = None


class ConversationTurnWrite(ConversationTurnRecord):
    embedding: list[float] = Field(min_length=1)


class ConversationSummaryRecord(MemoryRecordBase):
    conversation_id: str = Field(min_length=1)
    covered_turn_count: int = Field(ge=1)


class ConversationSummaryWrite(ConversationSummaryRecord):
    embedding: list[float] = Field(min_length=1)


class ProjectMemoryRecord(MemoryRecordBase):
    project_id: str = Field(min_length=1)
    memory_kind: ProjectMemoryKind
    source: str = Field(default="explicit", min_length=1)


class ProjectMemoryWrite(ProjectMemoryRecord):
    embedding: list[float] = Field(min_length=1)
