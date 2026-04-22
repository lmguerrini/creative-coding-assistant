"""Streaming event contracts shared by all clients."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StreamEventType(StrEnum):
    STATUS = "status"
    MEMORY = "memory"
    RETRIEVAL = "retrieval"
    TOOL_START = "tool_start"
    TOOL_RESULT = "tool_result"
    TOKEN_DELTA = "token_delta"
    PREVIEW_ARTIFACT = "preview_artifact"
    EVAL_UPDATE = "eval_update"
    FINAL = "final"
    ERROR = "error"


class StreamEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_type: StreamEventType
    sequence: int = Field(ge=0)
    payload: dict[str, Any] = Field(default_factory=dict)
