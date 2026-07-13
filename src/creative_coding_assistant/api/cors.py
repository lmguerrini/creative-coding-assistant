"""Environment-aware CORS policy helpers for browser-facing WSGI apps."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.core.config import (
    DEFAULT_LOCAL_CORS_ORIGINS,
    Settings,
    load_settings,
)

CorsPolicyStatus = Literal["ready", "guarded"]

LOCAL_CORS_WILDCARD = "*"
PRODUCTION_ENVIRONMENT = "production"


class CorsPolicyReport(BaseModel):
    """Resolved CORS posture without mutating process configuration."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    status: CorsPolicyStatus
    environment: str = Field(min_length=1)
    allowed_origins: tuple[str, ...]
    wildcard_allowed: bool
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _status_matches_warnings(self) -> CorsPolicyReport:
        expected = "guarded" if self.warnings else "ready"
        if self.status != expected:
            raise ValueError("status must match CORS warnings")
        if self.environment == PRODUCTION_ENVIRONMENT and self.wildcard_allowed:
            raise ValueError("production CORS policy cannot allow wildcard origins")
        return self


def build_cors_policy_report(settings: Settings | None = None) -> CorsPolicyReport:
    """Build the effective CORS policy for the current environment."""

    resolved = settings or load_settings()
    environment = resolved.environment.strip().lower()
    configured_origins = _normalized_origins(resolved.cors_allowed_origins)
    warnings: list[str] = []

    if environment == PRODUCTION_ENVIRONMENT:
        explicit_origins = tuple(
            origin for origin in configured_origins if origin != LOCAL_CORS_WILDCARD
        )
        if LOCAL_CORS_WILDCARD in configured_origins:
            warnings.append(
                "CCA_CORS_ALLOWED_ORIGINS must not include '*' in production."
            )
        if not explicit_origins:
            warnings.append(
                "CCA_CORS_ALLOWED_ORIGINS must define at least one explicit production origin."
            )
        return CorsPolicyReport(
            status="guarded" if warnings else "ready",
            environment=environment,
            allowed_origins=explicit_origins,
            wildcard_allowed=False,
            warnings=tuple(warnings),
        )

    return CorsPolicyReport(
        status="ready",
        environment=environment,
        allowed_origins=configured_origins,
        wildcard_allowed=LOCAL_CORS_WILDCARD in configured_origins,
    )


def resolve_cors_allow_origin(
    environ: dict[str, object],
    *,
    settings: Settings | None = None,
) -> str | None:
    """Resolve the response Access-Control-Allow-Origin value for one request."""

    policy = build_cors_policy_report(settings=settings)
    request_origin = str(environ.get("HTTP_ORIGIN", "")).strip()

    if policy.wildcard_allowed:
        return LOCAL_CORS_WILDCARD

    if request_origin in policy.allowed_origins:
        return request_origin

    return None


def _normalized_origins(origins: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(origin.strip() for origin in origins if origin.strip())
    return normalized or DEFAULT_LOCAL_CORS_ORIGINS
