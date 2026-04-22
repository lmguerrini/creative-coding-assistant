"""Explicit route selection for assistant requests."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)


class RouteName(StrEnum):
    GENERATE = "generate"
    EXPLAIN = "explain"
    DEBUG = "debug"
    DESIGN = "design"
    REVIEW = "review"
    PREVIEW = "preview"


class RouteCapability(StrEnum):
    MEMORY_CONTEXT = "memory_context"
    OFFICIAL_DOCS = "official_docs"
    TOOL_USE = "tool_use"
    PREVIEW_ARTIFACTS = "preview_artifacts"
    LIVE_EVALUATION = "live_evaluation"


class RouteDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    route: RouteName
    mode: AssistantMode
    domain: CreativeCodingDomain | None = None
    capabilities: tuple[RouteCapability, ...] = Field(default_factory=tuple)


MODE_ROUTE_MAP: dict[AssistantMode, tuple[RouteName, tuple[RouteCapability, ...]]] = {
    AssistantMode.GENERATE: (
        RouteName.GENERATE,
        (
            RouteCapability.MEMORY_CONTEXT,
            RouteCapability.OFFICIAL_DOCS,
            RouteCapability.TOOL_USE,
            RouteCapability.LIVE_EVALUATION,
        ),
    ),
    AssistantMode.EXPLAIN: (
        RouteName.EXPLAIN,
        (
            RouteCapability.MEMORY_CONTEXT,
            RouteCapability.OFFICIAL_DOCS,
            RouteCapability.LIVE_EVALUATION,
        ),
    ),
    AssistantMode.DEBUG: (
        RouteName.DEBUG,
        (
            RouteCapability.MEMORY_CONTEXT,
            RouteCapability.OFFICIAL_DOCS,
            RouteCapability.TOOL_USE,
            RouteCapability.LIVE_EVALUATION,
        ),
    ),
    AssistantMode.DESIGN: (
        RouteName.DESIGN,
        (
            RouteCapability.MEMORY_CONTEXT,
            RouteCapability.OFFICIAL_DOCS,
            RouteCapability.LIVE_EVALUATION,
        ),
    ),
    AssistantMode.REVIEW: (
        RouteName.REVIEW,
        (
            RouteCapability.MEMORY_CONTEXT,
            RouteCapability.OFFICIAL_DOCS,
            RouteCapability.TOOL_USE,
            RouteCapability.LIVE_EVALUATION,
        ),
    ),
    AssistantMode.PREVIEW: (
        RouteName.PREVIEW,
        (
            RouteCapability.MEMORY_CONTEXT,
            RouteCapability.OFFICIAL_DOCS,
            RouteCapability.TOOL_USE,
            RouteCapability.PREVIEW_ARTIFACTS,
            RouteCapability.LIVE_EVALUATION,
        ),
    ),
}


def route_request(request: AssistantRequest) -> RouteDecision:
    """Select an explicit route from request mode and optional domain."""

    route_name, capabilities = MODE_ROUTE_MAP[request.mode]
    return RouteDecision(
        route=route_name,
        mode=request.mode,
        domain=request.domain,
        capabilities=capabilities,
    )
