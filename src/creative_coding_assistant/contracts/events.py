"""Streaming event contracts shared by all clients."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StreamEventType(str, Enum):
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


@dataclass(frozen=True)
class StreamEvent:
    event_type: StreamEventType
    sequence: int
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("Stream event sequence must be zero or greater.")
