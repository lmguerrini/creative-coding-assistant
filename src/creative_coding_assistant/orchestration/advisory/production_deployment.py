"""V5.6 production deployment readiness metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.api.dev_server import DEFAULT_DEV_HOST, DEFAULT_DEV_PORT
from creative_coding_assistant.api.streaming import DEFAULT_STREAM_PATH
from creative_coding_assistant.api.workspace_sessions import (
    DEFAULT_WORKSPACE_SESSION_PATH,
)
from creative_coding_assistant.orchestration.production_release_packaging import (
    ProductionPackagingPlan,
    build_production_packaging_plan,
)

DeploymentSurfaceId = Literal[
    "backend_runtime_entrypoint",
    "frontend_runtime_entrypoint",
    "environment_configuration",
    "runtime_data_paths",
    "external_deployment_manifest",
]
DeploymentReadinessStatus = Literal["ready", "guarded"]

PRODUCTION_DEPLOYMENT_RECORD_SERIALIZATION_VERSION = "production_deployment_record.v1"
PRODUCTION_DEPLOYMENT_PLAN_SERIALIZATION_VERSION = "production_deployment_plan.v1"
PRODUCTION_DEPLOYMENT_AUTHORITY_BOUNDARY = (
    "V5.6 production deployment metadata reviews backend, frontend, "
    "environment, runtime data, and external deployment assumptions from "
    "repository files only; it does not deploy services, start servers, "
    "install dependencies, build packages, create containers, provision "
    "providers, mutate environment variables, write runtime data, execute "
    "workflows, call providers, merge, push, tag, or apply Runtime Evolution."
)

_REQUIRED_SURFACES: tuple[DeploymentSurfaceId, ...] = (
    "backend_runtime_entrypoint",
    "frontend_runtime_entrypoint",
    "environment_configuration",
    "runtime_data_paths",
    "external_deployment_manifest",
)
_EXTERNAL_MANIFESTS = (
    "Dockerfile",
    "docker-compose.yml",
    "vercel.json",
    "render.yaml",
    "Procfile",
)
_RUNTIME_DATA_PATHS = (
    "data/chroma",
    "data/artifacts",
    "data/eval",
)
_REQUIRED_FRONTEND_SCRIPTS = ("build", "start")
_BLOCKED_RUNTIME_BEHAVIORS = (
    "deployment_execution",
    "server_start",
    "dependency_installation",
    "package_build_execution",
    "container_image_build",
    "provider_provisioning",
    "environment_variable_mutation",
    "runtime_data_write",
    "workflow_execution",
    "provider_execution",
    "merge_push_tag_operation",
    "runtime_evolution_application",
)


class ProductionDeploymentRecord(BaseModel):
    """One deployment readiness surface."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    record_id: str = Field(min_length=1, max_length=180)
    surface_id: DeploymentSurfaceId
    status: DeploymentReadinessStatus
    source_refs: tuple[str, ...] = Field(min_length=1, max_length=16)
    required_items: tuple[str, ...] = Field(min_length=1, max_length=24)
    present_items: tuple[str, ...] = Field(default_factory=tuple, max_length=24)
    missing_items: tuple[str, ...] = Field(default_factory=tuple, max_length=24)
    deployment_notes: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    deployment_record_implemented: Literal[True] = True
    deployment_execution_implemented: Literal[False] = False
    server_start_implemented: Literal[False] = False
    dependency_installation_implemented: Literal[False] = False
    package_build_executed: Literal[False] = False
    container_image_build_implemented: Literal[False] = False
    provider_provisioning_implemented: Literal[False] = False
    environment_variable_mutation_implemented: Literal[False] = False
    runtime_data_write_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    merge_push_tag_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["production_deployment_record.v1"] = (
        PRODUCTION_DEPLOYMENT_RECORD_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _record_matches_items(self) -> Self:
        if self.record_id != f"production_deployment::{self.surface_id}":
            raise ValueError("record_id must match surface_id")
        if self.missing_items != tuple(
            item for item in self.required_items if item not in self.present_items
        ):
            raise ValueError("missing_items must match required and present items")
        if self.status != ("guarded" if self.missing_items else "ready"):
            raise ValueError("status must match missing items")
        return self


class ProductionDeploymentPlan(BaseModel):
    """Deployment readiness posture for local demo and external assumptions."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["production_deployment"] = "production_deployment"
    serialization_version: Literal["production_deployment_plan.v1"] = (
        PRODUCTION_DEPLOYMENT_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=PRODUCTION_DEPLOYMENT_AUTHORITY_BOUNDARY,
        max_length=1800,
    )
    source_packaging_serialization_version: str = Field(min_length=1, max_length=120)
    backend_host: str = Field(min_length=1, max_length=120)
    backend_port: int = Field(ge=1, le=65535)
    backend_paths: tuple[str, ...] = Field(min_length=2, max_length=2)
    frontend_scripts: tuple[str, ...] = Field(min_length=2, max_length=2)
    external_manifest_paths: tuple[str, ...] = Field(min_length=5, max_length=5)
    runtime_data_paths: tuple[str, ...] = Field(min_length=3, max_length=3)
    records: tuple[ProductionDeploymentRecord, ...] = Field(min_length=5, max_length=5)
    record_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    surface_ids: tuple[DeploymentSurfaceId, ...] = Field(min_length=5, max_length=5)
    ready_record_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    guarded_record_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    record_count: int = Field(ge=5, le=5)
    deployment_status: DeploymentReadinessStatus
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    deployment_metadata_implemented: Literal[True] = True
    local_backend_entrypoint_documented: Literal[True] = True
    local_frontend_entrypoint_documented: Literal[True] = True
    environment_configuration_documented: Literal[True] = True
    runtime_data_paths_documented: Literal[True] = True
    external_deployment_assumptions_documented: Literal[True] = True
    deployment_execution_implemented: Literal[False] = False
    server_start_implemented: Literal[False] = False
    dependency_installation_implemented: Literal[False] = False
    package_build_executed: Literal[False] = False
    container_image_build_implemented: Literal[False] = False
    provider_provisioning_implemented: Literal[False] = False
    environment_variable_mutation_implemented: Literal[False] = False
    runtime_data_write_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    merge_push_tag_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_records(self) -> Self:
        if self.record_ids != tuple(record.record_id for record in self.records):
            raise ValueError("record_ids must match records")
        if self.surface_ids != tuple(record.surface_id for record in self.records):
            raise ValueError("surface_ids must match records")
        if self.surface_ids != _REQUIRED_SURFACES:
            raise ValueError("surface_ids must cover required deployment surfaces")
        if self.ready_record_ids != _record_ids_for_status(self.records, "ready"):
            raise ValueError("ready_record_ids must match records")
        if self.guarded_record_ids != _record_ids_for_status(self.records, "guarded"):
            raise ValueError("guarded_record_ids must match records")
        if self.record_count != len(self.records):
            raise ValueError("record_count must match records")
        if self.deployment_status != _plan_status(self.records):
            raise ValueError("deployment_status must match records")
        return self


def build_production_deployment_plan(
    project_root: str | Path | None = None,
    *,
    packaging: ProductionPackagingPlan | None = None,
) -> ProductionDeploymentPlan:
    """Build deployment readiness metadata without deploying or starting servers."""

    root = Path(project_root or ".").resolve()
    packaging_source = packaging or build_production_packaging_plan(root)
    package_json = json.loads(
        (root / "clients/nextjs/package.json").read_text(encoding="utf-8")
    )
    scripts = package_json.get("scripts", {})
    records = _records(root=root, packaging=packaging_source, scripts=scripts)
    return ProductionDeploymentPlan(
        source_packaging_serialization_version=packaging_source.serialization_version,
        backend_host=DEFAULT_DEV_HOST,
        backend_port=DEFAULT_DEV_PORT,
        backend_paths=(DEFAULT_STREAM_PATH, DEFAULT_WORKSPACE_SESSION_PATH),
        frontend_scripts=_REQUIRED_FRONTEND_SCRIPTS,
        external_manifest_paths=_EXTERNAL_MANIFESTS,
        runtime_data_paths=_RUNTIME_DATA_PATHS,
        records=records,
        record_ids=tuple(record.record_id for record in records),
        surface_ids=tuple(record.surface_id for record in records),
        ready_record_ids=_record_ids_for_status(records, "ready"),
        guarded_record_ids=_record_ids_for_status(records, "guarded"),
        record_count=len(records),
        deployment_status=_plan_status(records),
    )


def production_deployment_record_by_surface(
    surface_id: DeploymentSurfaceId | str,
    plan: ProductionDeploymentPlan | None = None,
) -> ProductionDeploymentRecord | None:
    """Return one deployment record by surface id."""

    normalized = str(surface_id).strip()
    source_plan = plan or build_production_deployment_plan()
    for record in source_plan.records:
        if record.surface_id == normalized:
            return record
    return None


def production_deployment_records_for_status(
    status: DeploymentReadinessStatus,
    plan: ProductionDeploymentPlan | None = None,
) -> tuple[ProductionDeploymentRecord, ...]:
    """Return deployment records by readiness status."""

    source_plan = plan or build_production_deployment_plan()
    return tuple(record for record in source_plan.records if record.status == status)


def _records(
    *,
    root: Path,
    packaging: ProductionPackagingPlan,
    scripts: object,
) -> tuple[ProductionDeploymentRecord, ...]:
    frontend_scripts = scripts if isinstance(scripts, dict) else {}
    return (
        _record(
            surface_id="backend_runtime_entrypoint",
            source_refs=("src/creative_coding_assistant/api/dev_server.py",),
            required_items=(
                "dev_server_module",
                DEFAULT_STREAM_PATH,
                DEFAULT_WORKSPACE_SESSION_PATH,
            ),
            present_items=(
                "dev_server_module",
                DEFAULT_STREAM_PATH,
                DEFAULT_WORKSPACE_SESSION_PATH,
            ),
            deployment_notes=(
                f"Local backend bridge defaults to {DEFAULT_DEV_HOST}:{DEFAULT_DEV_PORT}.",
                "Production hosting must explicitly choose host, port, and WSGI/ASGI wrapper.",
            ),
        ),
        _record(
            surface_id="frontend_runtime_entrypoint",
            source_refs=(
                "clients/nextjs/package.json",
                "clients/nextjs/next.config.mjs",
            ),
            required_items=_REQUIRED_FRONTEND_SCRIPTS,
            present_items=tuple(
                script
                for script in _REQUIRED_FRONTEND_SCRIPTS
                if script in frontend_scripts
            ),
            deployment_notes=(
                "Next.js build and start scripts are present for operator-run deployment validation.",
                "No automatic npm install, build, or start is introduced.",
            ),
        ),
        _record(
            surface_id="environment_configuration",
            source_refs=(".env.example", packaging.role),
            required_items=(
                "OPENAI_API_KEY",
                "CCA_DEFAULT_GENERATION_PROVIDER",
                "CCA_CHROMA_PERSIST_DIR",
                "CCA_ARTIFACT_DIR",
            ),
            present_items=tuple(
                key
                for key in (
                    "OPENAI_API_KEY",
                    "CCA_DEFAULT_GENERATION_PROVIDER",
                    "CCA_CHROMA_PERSIST_DIR",
                    "CCA_ARTIFACT_DIR",
                )
                if key in packaging.environment_variable_keys
            ),
            deployment_notes=(
                "Provider credentials and runtime paths are explicit environment configuration.",
                "Missing production configuration should fail safely before provider execution.",
            ),
        ),
        _record(
            surface_id="runtime_data_paths",
            source_refs=_RUNTIME_DATA_PATHS,
            required_items=_RUNTIME_DATA_PATHS,
            present_items=tuple(
                path for path in _RUNTIME_DATA_PATHS if (root / path).exists()
            ),
            deployment_notes=(
                "Chroma, artifact, and eval data directories exist as local runtime paths.",
                "Deployment must provision durable storage explicitly when needed.",
            ),
        ),
        _record(
            surface_id="external_deployment_manifest",
            source_refs=_EXTERNAL_MANIFESTS,
            required_items=_EXTERNAL_MANIFESTS,
            present_items=tuple(
                path for path in _EXTERNAL_MANIFESTS if (root / path).exists()
            ),
            deployment_notes=(
                "No external deployment manifest is present in the repository.",
                "V5.6 keeps deployment assumptions explicit instead of adding automatic deployment.",
            ),
        ),
    )


def _record(
    *,
    surface_id: DeploymentSurfaceId,
    source_refs: tuple[str, ...],
    required_items: tuple[str, ...],
    present_items: tuple[str, ...],
    deployment_notes: tuple[str, ...],
) -> ProductionDeploymentRecord:
    missing = tuple(item for item in required_items if item not in present_items)
    return ProductionDeploymentRecord(
        record_id=f"production_deployment::{surface_id}",
        surface_id=surface_id,
        status="guarded" if missing else "ready",
        source_refs=source_refs,
        required_items=required_items,
        present_items=present_items,
        missing_items=missing,
        deployment_notes=deployment_notes,
    )


def _record_ids_for_status(
    records: tuple[ProductionDeploymentRecord, ...],
    status: DeploymentReadinessStatus,
) -> tuple[str, ...]:
    return tuple(record.record_id for record in records if record.status == status)


def _plan_status(
    records: tuple[ProductionDeploymentRecord, ...],
) -> DeploymentReadinessStatus:
    if any(record.status == "guarded" for record in records):
        return "guarded"
    return "ready"
