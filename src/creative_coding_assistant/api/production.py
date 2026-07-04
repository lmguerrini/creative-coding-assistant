"""Production readiness reports for the browser-facing runtime API."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.api.contracts import (
    API_CONTRACT_VERSION,
    ERROR_CONTRACT_VERSION,
    HEALTH_CONTRACT_VERSION,
    STREAM_CONTRACT_VERSION,
    WORKSPACE_SESSION_CONTRACT_VERSION,
)
from creative_coding_assistant.api.cors import (
    CorsPolicyReport,
    build_cors_policy_report,
)
from creative_coding_assistant.core.config import Settings, load_settings

ProductionReadinessStatus = Literal["ready", "guarded"]
DependencyStatus = Literal["ready", "guarded", "missing"]
TelemetryEventKind = Literal[
    "request",
    "error",
    "health",
    "configuration",
    "dependency",
]

CHROMA_REQUIREMENT = ">=0.6.3,<1.0.0"
CHROMA_MINIMUM_VERSION = "0.6.3"
CHROMA_UNSAFE_MAJOR_FLOOR = "1.0.0"
CHROMA_VULNERABILITY_IDS = ("CVE-2026-45829", "GHSA-f4j7-r4q5-qw2c")

V7_5_ROADMAP_ITEMS = (
    "API Contract Audit",
    "Backend Route Stabilization",
    "Streaming Contract Versioning",
    "Workspace Session Contract Stabilization",
    "Error Response Contract Stabilization",
    "Dev/Prod Server Boundary Audit",
    "Deployment Config Hardening",
    "Production Readiness Smoke Test",
    "Full-project Ruff Remediation Sprint",
    "Workspace Session 404/400 Resolution",
    "Chroma Dependency Upgrade",
    "Dependency Health Review",
    "Production Configuration Validation",
    "API Backward Compatibility",
    "Workspace Recovery",
    "Graceful Failure Recovery",
    "Health Check Endpoints",
    "Telemetry Readiness",
    "Observability Layer",
    "Production Logging Contracts",
    "Configuration Migration",
    "Release Checklist Generator",
)

V7_7_DEPLOYMENT_READINESS_ITEMS = (
    "Dockerfile",
    "Optional docker-compose",
    "Production WSGI server command",
    "Environment-aware CORS policy",
    "Production deployment documentation",
    "Health-check deployment guidance",
    "Basic production runtime checklist",
    "CI coverage reporting",
    "CI security and dependency scan",
    "Chroma dependency posture verification",
    "Production configuration validation",
    "Release and deployment readiness checklist",
)


class ProductionConfigurationReport(BaseModel):
    """Configuration validation report with no environment mutation."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    status: ProductionReadinessStatus
    environment: str = Field(min_length=1)
    required_keys: tuple[str, ...]
    present_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    contract_versions: dict[str, str]

    @model_validator(mode="after")
    def _report_matches_keys(self) -> ProductionConfigurationReport:
        expected_missing = tuple(
            key for key in self.required_keys if key not in self.present_keys
        )
        if self.missing_keys != expected_missing:
            raise ValueError("missing_keys must match required and present keys")
        expected_status = "guarded" if self.missing_keys or self.warnings else "ready"
        if self.status != expected_status:
            raise ValueError("status must match missing keys and warnings")
        return self


class DependencyHealthReport(BaseModel):
    """Dependency health posture for runtime-critical packages."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    status: DependencyStatus
    chromadb_requirement: str = CHROMA_REQUIREMENT
    chromadb_installed_version: str | None
    vulnerability_ids: tuple[str, ...] = CHROMA_VULNERABILITY_IDS
    notes: tuple[str, ...]


class ApiTelemetryEvent(BaseModel):
    """Structured telemetry-ready event; construction does not emit externally."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    event_kind: TelemetryEventKind
    route: str = Field(min_length=1)
    status_code: int = Field(ge=100, le=599)
    request_id: str = Field(min_length=1)
    contract_version: str = API_CONTRACT_VERSION
    error_code: str | None = None
    recoverable: bool | None = None


class ReleaseChecklistItem(BaseModel):
    """One generated release-readiness checklist entry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    roadmap_item: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    required: bool = True
    status: ProductionReadinessStatus


class ReleaseChecklist(BaseModel):
    """Generated V7.5 release checklist with complete roadmap coverage."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    version: Literal["v7.5"] = "v7.5"
    roadmap_item_count: int = Field(ge=22, le=22)
    checklist_item_count: int = Field(ge=22, le=22)
    roadmap_coverage_complete: bool
    remote_ci_verification_required: bool = True
    items: tuple[ReleaseChecklistItem, ...] = Field(min_length=22, max_length=22)
    guarded_items: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _checklist_covers_roadmap(self) -> ReleaseChecklist:
        item_names = tuple(item.roadmap_item for item in self.items)
        if item_names != V7_5_ROADMAP_ITEMS:
            raise ValueError("items must match V7.5 roadmap order")
        if self.roadmap_item_count != len(V7_5_ROADMAP_ITEMS):
            raise ValueError("roadmap_item_count must match V7.5 roadmap")
        if self.checklist_item_count != len(self.items):
            raise ValueError("checklist_item_count must match items")
        guarded = tuple(
            item.roadmap_item for item in self.items if item.status == "guarded"
        )
        if self.guarded_items != guarded:
            raise ValueError("guarded_items must match item statuses")
        if self.roadmap_coverage_complete is not True:
            raise ValueError("roadmap_coverage_complete must be true")
        return self


class DeploymentReadinessChecklistItem(BaseModel):
    """One V7.7 deployment-readiness checklist entry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    deployment_item: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    required: bool = True
    status: ProductionReadinessStatus


class DeploymentReadinessChecklist(BaseModel):
    """Generated V7.7 release/deployment readiness checklist."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    version: Literal["v7.7"] = "v7.7"
    deployment_item_count: int = Field(ge=12, le=12)
    checklist_item_count: int = Field(ge=12, le=12)
    deployment_coverage_complete: bool
    creative_behavior_changed: Literal[False] = False
    provider_routing_changed: Literal[False] = False
    items: tuple[DeploymentReadinessChecklistItem, ...] = Field(
        min_length=12,
        max_length=12,
    )
    guarded_items: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _checklist_covers_deployment_items(self) -> DeploymentReadinessChecklist:
        item_names = tuple(item.deployment_item for item in self.items)
        if item_names != V7_7_DEPLOYMENT_READINESS_ITEMS:
            raise ValueError("items must match V7.7 deployment readiness order")
        if self.deployment_item_count != len(V7_7_DEPLOYMENT_READINESS_ITEMS):
            raise ValueError("deployment_item_count must match V7.7 items")
        if self.checklist_item_count != len(self.items):
            raise ValueError("checklist_item_count must match items")
        guarded = tuple(
            item.deployment_item for item in self.items if item.status == "guarded"
        )
        if self.guarded_items != guarded:
            raise ValueError("guarded_items must match item statuses")
        if self.deployment_coverage_complete is not True:
            raise ValueError("deployment_coverage_complete must be true")
        return self


@dataclass(frozen=True)
class ConfigurationMigration:
    """Mapping from a legacy environment alias to the current setting name."""

    legacy_key: str
    current_key: str
    description: str


CONFIGURATION_MIGRATIONS = (
    ConfigurationMigration(
        legacy_key="CCA_WORKSPACE_DB_PATH",
        current_key="CCA_WORKSPACE_SESSION_DB_PATH",
        description="Workspace session persistence path was made explicit.",
    ),
    ConfigurationMigration(
        legacy_key="CCA_API_ENVIRONMENT",
        current_key="CCA_ENVIRONMENT",
        description="API environment now uses the shared runtime environment key.",
    ),
)


def validate_production_configuration(
    settings: Settings | None = None,
) -> ProductionConfigurationReport:
    """Validate production-relevant settings without creating resources."""

    resolved = settings or load_settings()
    required_keys = (
        "CCA_ENVIRONMENT",
        "CCA_LOG_LEVEL",
        "CCA_CORS_ALLOWED_ORIGINS",
        "CCA_CHROMA_PERSIST_DIR",
        "CCA_ARTIFACT_DIR",
        "CCA_WORKSPACE_SESSION_DB_PATH",
    )
    present_keys = tuple(
        key
        for key, value in (
            ("CCA_ENVIRONMENT", resolved.environment),
            ("CCA_LOG_LEVEL", resolved.log_level),
            ("CCA_CORS_ALLOWED_ORIGINS", ",".join(resolved.cors_allowed_origins)),
            ("CCA_CHROMA_PERSIST_DIR", str(resolved.chroma_persist_dir)),
            ("CCA_ARTIFACT_DIR", str(resolved.artifact_dir)),
            (
                "CCA_WORKSPACE_SESSION_DB_PATH",
                str(resolved.workspace_session_db_path),
            ),
        )
        if value
    )
    warnings = _configuration_warnings(resolved)
    return ProductionConfigurationReport(
        status="guarded"
        if tuple(key for key in required_keys if key not in present_keys) or warnings
        else "ready",
        environment=resolved.environment,
        required_keys=required_keys,
        present_keys=present_keys,
        missing_keys=tuple(key for key in required_keys if key not in present_keys),
        warnings=warnings,
        contract_versions={
            "api": API_CONTRACT_VERSION,
            "error": ERROR_CONTRACT_VERSION,
            "health": HEALTH_CONTRACT_VERSION,
            "stream": STREAM_CONTRACT_VERSION,
            "workspaceSession": WORKSPACE_SESSION_CONTRACT_VERSION,
        },
    )


def build_dependency_health_report(
    *,
    chromadb_version: str | None = None,
) -> DependencyHealthReport:
    """Build dependency health posture for Chroma without importing Chroma."""

    installed = (
        chromadb_version
        if chromadb_version is not None
        else _package_version("chromadb")
    )
    if installed is None:
        return DependencyHealthReport(
            status="missing",
            chromadb_installed_version=None,
            notes=("chromadb is not installed in the active environment.",),
        )

    if not _version_at_least(installed, CHROMA_MINIMUM_VERSION):
        return DependencyHealthReport(
            status="guarded",
            chromadb_installed_version=installed,
            notes=(
                f"chromadb must satisfy {CHROMA_REQUIREMENT}.",
                "Upgrade fresh installs to the safe pre-1.0 line.",
            ),
        )

    if _version_at_least(installed, CHROMA_UNSAFE_MAJOR_FLOOR):
        return DependencyHealthReport(
            status="guarded",
            chromadb_installed_version=installed,
            notes=(
                "PyPI vulnerability metadata reports an unfixed ChromaDB "
                "server code-injection issue for 1.0.0 and later.",
                f"Fresh installs are constrained to {CHROMA_REQUIREMENT}.",
            ),
        )

    return DependencyHealthReport(
        status="ready",
        chromadb_installed_version=installed,
        notes=(f"chromadb satisfies {CHROMA_REQUIREMENT}.",),
    )


def build_api_telemetry_event(
    *,
    event_kind: TelemetryEventKind,
    route: str,
    status_code: int,
    request_id: str,
    error_code: str | None = None,
    recoverable: bool | None = None,
) -> ApiTelemetryEvent:
    """Build a telemetry-ready API event for logging or future exporters."""

    return ApiTelemetryEvent(
        event_kind=event_kind,
        route=route,
        status_code=status_code,
        request_id=request_id,
        error_code=error_code,
        recoverable=recoverable,
    )


def log_api_telemetry_event(event: ApiTelemetryEvent) -> None:
    """Log a structured API event without sending external telemetry."""

    logger.bind(**event.model_dump(mode="json")).info("api_telemetry_event")


def build_release_checklist(
    *,
    configuration: ProductionConfigurationReport | None = None,
    dependency_health: DependencyHealthReport | None = None,
) -> ReleaseChecklist:
    """Generate the V7.5 release checklist from readiness reports."""

    config = configuration or validate_production_configuration()
    dependencies = dependency_health or build_dependency_health_report()
    guarded_inputs = {
        "Deployment Config Hardening": config.status,
        "Production Configuration Validation": config.status,
        "Chroma Dependency Upgrade": dependencies.status,
        "Dependency Health Review": dependencies.status,
    }
    items = tuple(
        ReleaseChecklistItem(
            roadmap_item=item,
            evidence=_release_checklist_evidence(item),
            status="guarded"
            if guarded_inputs.get(item) in {"guarded", "missing"}
            else "ready",
        )
        for item in V7_5_ROADMAP_ITEMS
    )
    return ReleaseChecklist(
        roadmap_item_count=len(V7_5_ROADMAP_ITEMS),
        checklist_item_count=len(items),
        roadmap_coverage_complete=True,
        items=items,
        guarded_items=tuple(
            item.roadmap_item for item in items if item.status == "guarded"
        ),
    )


def build_deployment_readiness_checklist(
    *,
    configuration: ProductionConfigurationReport | None = None,
    dependency_health: DependencyHealthReport | None = None,
    cors_policy: CorsPolicyReport | None = None,
) -> DeploymentReadinessChecklist:
    """Generate the V7.7 deployment readiness checklist."""

    config = configuration or validate_production_configuration()
    dependencies = dependency_health or build_dependency_health_report()
    cors = cors_policy or build_cors_policy_report()
    guarded_inputs = {
        "Environment-aware CORS policy": cors.status,
        "Chroma dependency posture verification": dependencies.status,
        "Production configuration validation": config.status,
    }
    items = tuple(
        DeploymentReadinessChecklistItem(
            deployment_item=item,
            evidence=_deployment_readiness_evidence(item),
            status="guarded"
            if guarded_inputs.get(item) in {"guarded", "missing"}
            else "ready",
        )
        for item in V7_7_DEPLOYMENT_READINESS_ITEMS
    )
    return DeploymentReadinessChecklist(
        deployment_item_count=len(V7_7_DEPLOYMENT_READINESS_ITEMS),
        checklist_item_count=len(items),
        deployment_coverage_complete=True,
        items=items,
        guarded_items=tuple(
            item.deployment_item for item in items if item.status == "guarded"
        ),
    )


def _configuration_warnings(settings: Settings) -> tuple[str, ...]:
    warnings: list[str] = []
    environment = settings.environment.strip().lower()
    if environment == "production" and not settings.has_openai_api_key:
        warnings.append(
            "OPENAI_API_KEY or CCA_OPENAI_API_KEY is required in production."
        )
    if settings.log_level not in {"TRACE", "DEBUG", "INFO", "WARNING", "ERROR"}:
        warnings.append("CCA_LOG_LEVEL must be a standard Loguru level.")
    warnings.extend(build_cors_policy_report(settings=settings).warnings)
    return tuple(warnings)


def _package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _version_at_least(value: str, floor: str) -> bool:
    return _version_parts(value) >= _version_parts(floor)


def _version_parts(value: str) -> tuple[int, ...]:
    release = value.split("+", maxsplit=1)[0].split("-", maxsplit=1)[0]
    return tuple(int(part) for part in release.split(".") if part.isdigit())


def _release_checklist_evidence(item: str) -> str:
    evidence = {
        "API Contract Audit": "Shared api.contracts models and focused tests.",
        "Backend Route Stabilization": "Backend route manifest and dispatcher tests.",
        "Streaming Contract Versioning": "assistant-stream.v1 response headers.",
        "Workspace Session Contract Stabilization": "workspace-session.v1 headers.",
        "Error Response Contract Stabilization": "api-error.v1 payload tests.",
        "Dev/Prod Server Boundary Audit": "production dev-server guard tests.",
        "Deployment Config Hardening": "Settings and configuration report.",
        "Production Readiness Smoke Test": "health/readiness endpoint tests.",
        "Full-project Ruff Remediation Sprint": "ruff check src tests scripts.",
        "Workspace Session 404/400 Resolution": "typed missing/invalid responses.",
        "Chroma Dependency Upgrade": "pyproject chromadb constraint.",
        "Dependency Health Review": "dependency health report.",
        "Production Configuration Validation": "production configuration report.",
        "API Backward Compatibility": "legacy error and endpoint shape tests.",
        "Workspace Recovery": "workspace service-failure recovery tests.",
        "Graceful Failure Recovery": "stream service-failure recovery tests.",
        "Health Check Endpoints": "/api/health live and ready routes.",
        "Telemetry Readiness": "ApiTelemetryEvent contract.",
        "Observability Layer": "structured API telemetry logging helper.",
        "Production Logging Contracts": "configurable structured logging.",
        "Configuration Migration": "CONFIGURATION_MIGRATIONS aliases.",
        "Release Checklist Generator": "build_release_checklist coverage tests.",
    }
    return evidence[item]


def _deployment_readiness_evidence(item: str) -> str:
    evidence = {
        "Dockerfile": "Root Dockerfile builds the backend WSGI runtime image.",
        "Optional docker-compose": (
            "docker-compose.yml runs the backend with a data volume."
        ),
        "Production WSGI server command": (
            "Gunicorn serves creative_coding_assistant.api.wsgi:application."
        ),
        "Environment-aware CORS policy": (
            "CORS resolves from CCA_ENVIRONMENT and CCA_CORS_ALLOWED_ORIGINS."
        ),
        "Production deployment documentation": (
            "docs/PRODUCTION_DEPLOYMENT.md documents local and production execution."
        ),
        "Health-check deployment guidance": (
            "Docker and docs use /api/health/live and /api/health/ready."
        ),
        "Basic production runtime checklist": (
            "Runtime checklist covers env, storage, logs, probes, and rollback."
        ),
        "CI coverage reporting": (
            "CI runs pytest with coverage XML and terminal reporting."
        ),
        "CI security and dependency scan": (
            "CI runs pip-audit against installed backend dependencies."
        ),
        "Chroma dependency posture verification": (
            "Chroma remains constrained to the safe pre-1.0 line."
        ),
        "Production configuration validation": (
            "validate_production_configuration reports guarded production gaps."
        ),
        "Release and deployment readiness checklist": (
            "build_deployment_readiness_checklist covers all V7.7 items."
        ),
    }
    return evidence[item]
