"""Typed tool contracts and registry foundation for future workflow nodes."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from enum import StrEnum
from typing import Any, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.orchestration.routing import RouteCapability

_TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(?:[.-][a-z0-9_]+)*$")


class ToolStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


TERMINAL_TOOL_STATUSES: tuple[ToolStatus, ...] = (
    ToolStatus.SUCCEEDED,
    ToolStatus.FAILED,
    ToolStatus.SKIPPED,
)

_ALLOWED_TOOL_STATUS_TRANSITIONS: dict[ToolStatus, tuple[ToolStatus, ...]] = {
    ToolStatus.PENDING: (
        ToolStatus.RUNNING,
        ToolStatus.SUCCEEDED,
        ToolStatus.FAILED,
        ToolStatus.SKIPPED,
    ),
    ToolStatus.RUNNING: (
        ToolStatus.SUCCEEDED,
        ToolStatus.FAILED,
    ),
    ToolStatus.SUCCEEDED: (),
    ToolStatus.FAILED: (),
    ToolStatus.SKIPPED: (),
}


class ToolIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    name: str = Field(min_length=1)

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, value: object) -> str:
        return str(value).strip().lower()

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not _TOOL_NAME_PATTERN.fullmatch(value):
            raise ValueError(
                "Tool names must be lowercase identifiers with optional '.', '-', "
                "or '_' separators."
            )
        return value


class ToolMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    identity: ToolIdentity
    display_name: str = Field(min_length=1)
    description: str | None = None
    version: str | None = None
    required_capabilities: tuple[RouteCapability, ...] = (
        RouteCapability.TOOL_USE,
    )
    tags: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self.identity.name

    @field_validator("required_capabilities", mode="before")
    @classmethod
    def normalize_capabilities(
        cls,
        value: Sequence[RouteCapability | str] | RouteCapability | str | None,
    ) -> tuple[RouteCapability, ...]:
        if value is None:
            return (RouteCapability.TOOL_USE,)
        if isinstance(value, RouteCapability):
            return (value,)
        if isinstance(value, str):
            return (RouteCapability(str(value).strip()),)

        normalized: list[RouteCapability] = []
        for item in value:
            capability = (
                item
                if isinstance(item, RouteCapability)
                else RouteCapability(str(item).strip())
            )
            if capability not in normalized:
                normalized.append(capability)
        return tuple(normalized)

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_tags(
        cls,
        value: Sequence[str] | str | None,
    ) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            values = [value]
        else:
            values = list(value)

        normalized: list[str] = []
        for item in values:
            tag = str(item).strip().lower()
            if tag and tag not in normalized:
                normalized.append(tag)
        return tuple(normalized)


class ToolRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    request_id: str = Field(min_length=1)
    tool: ToolMetadata
    arguments: dict[str, Any] = Field(default_factory=dict)

    @property
    def tool_name(self) -> str:
        return self.tool.name


class ToolError(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    request: ToolRequest
    status: ToolStatus
    output: dict[str, Any] = Field(default_factory=dict)
    error: ToolError | None = None

    @property
    def tool_name(self) -> str:
        return self.request.tool_name

    @model_validator(mode="after")
    def validate_terminal_result(self) -> Self:
        if self.status not in TERMINAL_TOOL_STATUSES:
            raise ValueError("Tool results must use a terminal tool status.")

        if self.status is ToolStatus.FAILED:
            if self.error is None:
                raise ValueError("Failed tool results require an error payload.")
            if self.output:
                raise ValueError("Failed tool results cannot include output.")
            return self

        if self.error is not None:
            raise ValueError(
                "Successful or skipped tool results cannot include an error payload."
            )

        if self.status is ToolStatus.SKIPPED and self.output:
            raise ValueError("Skipped tool results cannot include output.")

        return self

    @classmethod
    def succeeded(
        cls,
        *,
        request: ToolRequest,
        output: dict[str, Any] | None = None,
    ) -> ToolResult:
        return cls(
            request=request,
            status=ToolStatus.SUCCEEDED,
            output=output or {},
        )

    @classmethod
    def failed(
        cls,
        *,
        request: ToolRequest,
        code: str,
        message: str,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> ToolResult:
        return cls(
            request=request,
            status=ToolStatus.FAILED,
            error=ToolError(
                code=code,
                message=message,
                retryable=retryable,
                details=details or {},
            ),
        )

    @classmethod
    def skipped(
        cls,
        *,
        request: ToolRequest,
    ) -> ToolResult:
        return cls(
            request=request,
            status=ToolStatus.SKIPPED,
        )


def get_allowed_tool_status_transitions(
    status: ToolStatus,
) -> tuple[ToolStatus, ...]:
    return _ALLOWED_TOOL_STATUS_TRANSITIONS[status]


def can_transition_tool_status(
    current: ToolStatus,
    next_status: ToolStatus,
) -> bool:
    return next_status in get_allowed_tool_status_transitions(current)


def validate_tool_status_transition(
    current: ToolStatus,
    next_status: ToolStatus,
) -> None:
    if not can_transition_tool_status(current, next_status):
        raise ValueError(
            f"Invalid tool status transition: {current.value} -> "
            f"{next_status.value}."
        )


class AssistantTool(Protocol):
    @property
    def metadata(self) -> ToolMetadata:
        """Expose stable metadata for one registered tool."""

    def invoke(self, request: ToolRequest) -> ToolResult:
        """Execute one tool request and return a typed terminal result."""


class DuplicateToolRegistrationError(ValueError):
    """Raised when a registry receives two tools with the same identity."""


class ToolNotRegisteredError(LookupError):
    """Raised when a registry lookup references an unknown tool."""


class ToolRegistry:
    """Small in-memory registry for future workflow-owned tools."""

    def __init__(self, tools: Iterable[AssistantTool] = ()) -> None:
        self._tools: dict[str, AssistantTool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: AssistantTool) -> None:
        name = tool.metadata.name
        if name in self._tools:
            raise DuplicateToolRegistrationError(
                f"Tool already registered: {name}"
            )
        self._tools[name] = tool

    def is_registered(self, tool: ToolIdentity | str) -> bool:
        return self._resolve_name(tool) in self._tools

    def get(self, tool: ToolIdentity | str) -> AssistantTool:
        name = self._resolve_name(tool)
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotRegisteredError(f"Unknown tool: {name}") from exc

    def list_metadata(self) -> tuple[ToolMetadata, ...]:
        return tuple(tool.metadata for tool in self._tools.values())

    def invoke(self, request: ToolRequest) -> ToolResult:
        tool = self.get(request.tool.identity)
        result = tool.invoke(request)
        if result.request != request:
            raise ValueError("Tool results must reference the original request.")
        return result

    @staticmethod
    def _resolve_name(tool: ToolIdentity | str) -> str:
        if isinstance(tool, ToolIdentity):
            return tool.name
        return ToolIdentity(name=tool).name
