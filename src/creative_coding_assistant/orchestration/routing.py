"""Explicit route selection for assistant requests."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


class DomainSelectionShape(StrEnum):
    NONE = "none"
    SINGLE = "single"
    MULTI = "multi"


class RouteDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    route: RouteName
    mode: AssistantMode
    domain: CreativeCodingDomain | None = None
    domains: tuple[CreativeCodingDomain, ...] = Field(default_factory=tuple)
    domain_selection: DomainSelectionShape = DomainSelectionShape.NONE
    capabilities: tuple[RouteCapability, ...] = Field(default_factory=tuple)

    @field_validator("domains", mode="before")
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

        if domain is not None and not domains:
            normalized["domains"] = (domain,)

        return normalized

    @model_validator(mode="after")
    def validate_domain_selection(self) -> RouteDecision:
        if self.domain is None and len(self.domains) == 1:
            object.__setattr__(self, "domain", self.domains[0])

        if self.domain is not None and self.domain not in self.domains:
            raise ValueError(
                "Route decision domain must be included in domains "
                "when both are provided."
            )

        object.__setattr__(
            self,
            "domain_selection",
            _selection_shape_for_domains(self.domains),
        )
        return self


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
        domains=request.domains,
        capabilities=capabilities,
    )


def _selection_shape_for_domains(
    domains: tuple[CreativeCodingDomain, ...],
) -> DomainSelectionShape:
    if not domains:
        return DomainSelectionShape.NONE
    if len(domains) == 1:
        return DomainSelectionShape.SINGLE
    return DomainSelectionShape.MULTI
