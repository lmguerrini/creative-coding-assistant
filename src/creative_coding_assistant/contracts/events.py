"""Streaming event contracts shared by all clients."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StreamEventType(StrEnum):
    STATUS = "status"
    MEMORY = "memory"
    RETRIEVAL = "retrieval"
    CONTEXT = "context"
    PROMPT_INPUT = "prompt_input"
    PLANNING = "planning"
    PROMPT_RENDERED = "prompt_rendered"
    GENERATION_INPUT = "generation_input"
    TOOL_START = "tool_start"
    TOOL_RESULT = "tool_result"
    TOKEN_DELTA = "token_delta"
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    NODE_FAILED = "node_failed"
    REVIEW_PASSED = "review_passed"
    REVIEW_FAILED = "review_failed"
    REFINEMENT_REQUESTED = "refinement_requested"
    REFINEMENT_COMPLETED = "refinement_completed"
    RETRY_STARTED = "retry_started"
    RETRY_COMPLETED = "retry_completed"
    ARTIFACT_EXTRACTED = "artifact_extracted"
    ARTIFACT_CRITIQUE = "artifact_critique"
    PREVIEW_ARTIFACT = "preview_artifact"
    EVAL_UPDATE = "eval_update"
    FINAL = "final"
    ERROR = "error"


class StreamEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_type: StreamEventType
    sequence: int = Field(ge=0)
    payload: dict[str, Any] = Field(default_factory=dict)
