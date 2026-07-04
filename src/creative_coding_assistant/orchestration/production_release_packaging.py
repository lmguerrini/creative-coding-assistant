"""V5.6 production release packaging readiness metadata."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

PackagingSurfaceId = Literal[
    "python_package_metadata",
    "frontend_package_metadata",
    "environment_template",
    "runtime_data_placeholders",
    "release_scripts",
]
PackagingSurfaceStatus = Literal["ready", "guarded"]

PRODUCTION_PACKAGING_RECORD_SERIALIZATION_VERSION = (
    "production_release_packaging_record.v1"
)
PRODUCTION_PACKAGING_PLAN_SERIALIZATION_VERSION = "production_release_packaging_plan.v1"
PRODUCTION_PACKAGING_AUTHORITY_BOUNDARY = (
    "V5.6 production release packaging metadata validates package, frontend, "
    "environment, runtime placeholder, and script readiness from repository "
    "files only; it does not install dependencies, run package builds, create "
    "archives, build containers, provision providers, mutate environment "
    "variables, write runtime data, execute workflows, call providers, merge, "
    "push, tag, or apply Runtime Evolution."
)

_REQUIRED_SURFACES: tuple[PackagingSurfaceId, ...] = (
    "python_package_metadata",
    "frontend_package_metadata",
    "environment_template",
    "runtime_data_placeholders",
    "release_scripts",
)
_REQUIRED_ENV_KEYS = (
    "OPENAI_API_KEY",
    "CCA_OPENAI_API_KEY",
    "CCA_OPENAI_MODEL",
    "CCA_OPENAI_EMBEDDING_MODEL",
    "CCA_DEFAULT_GENERATION_PROVIDER",
    "CCA_DEFAULT_DOMAIN",
    "CCA_DEFAULT_MODE",
    "CCA_CHROMA_PERSIST_DIR",
    "CCA_ARTIFACT_DIR",
    "CCA_EVAL_DATA_PATH",
    "CCA_EVAL_RAGAS_RESULTS_PATH",
    "CCA_LOG_LEVEL",
)
_REQUIRED_FRONTEND_SCRIPTS = ("dev", "build", "start", "typecheck", "test")
_REQUIRED_RUNTIME_PLACEHOLDERS = (
    "data/chroma/.gitkeep",
    "data/artifacts/.gitkeep",
    "data/eval/.gitkeep",
)
_REQUIRED_SCRIPT_PATHS = (
    "scripts/sync_official_kb.py",
    "scripts/eval_live_sessions.py",
    "scripts/run_eval_latest.sh",
    "scripts/README.md",
)
_PACKAGING_COMMANDS = (
    "python -m build",
    "npm --prefix clients/nextjs run build",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "dependency_installation",
    "package_build_execution",
    "archive_creation",
    "container_image_build",
    "provider_provisioning",
    "environment_variable_mutation",
    "api_key_assumption",
    "runtime_data_write",
    "workflow_execution",
    "provider_execution",
    "telemetry_emission",
    "generated_output_modification",
    "merge_push_tag_operation",
    "runtime_evolution_application",
)


class ProductionPackagingRecord(BaseModel):
    """One repository packaging readiness record."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    record_id: str = Field(min_length=1, max_length=180)
    surface_id: PackagingSurfaceId
    status: PackagingSurfaceStatus
    source_paths: tuple[str, ...] = Field(min_length=1, max_length=12)
    required_items: tuple[str, ...] = Field(min_length=1, max_length=24)
    present_items: tuple[str, ...] = Field(default_factory=tuple, max_length=24)
    missing_items: tuple[str, ...] = Field(default_factory=tuple, max_length=24)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=10)
    packaging_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=18,
    )
    packaging_record_implemented: Literal[True] = True
    dependency_installation_implemented: Literal[False] = False
    package_build_executed: Literal[False] = False
    archive_creation_implemented: Literal[False] = False
    container_image_build_implemented: Literal[False] = False
    provider_provisioning_implemented: Literal[False] = False
    environment_variable_mutation_implemented: Literal[False] = False
    runtime_data_write_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    merge_push_tag_implemented: Literal[False] = False
    serialization_version: Literal["production_release_packaging_record.v1"] = (
        PRODUCTION_PACKAGING_RECORD_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _record_matches_items(self) -> Self:
        if self.record_id != f"production_packaging::{self.surface_id}":
            raise ValueError("record_id must match surface_id")
        if self.missing_items != tuple(
            item for item in self.required_items if item not in self.present_items
        ):
            raise ValueError("missing_items must match required and present items")
        if self.status != ("guarded" if self.missing_items else "ready"):
            raise ValueError("status must match missing items")
        return self


class ProductionPackagingPlan(BaseModel):
    """Production release packaging posture over repository metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["production_release_packaging"] = "production_release_packaging"
    serialization_version: Literal["production_release_packaging_plan.v1"] = (
        PRODUCTION_PACKAGING_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=PRODUCTION_PACKAGING_AUTHORITY_BOUNDARY,
        max_length=1800,
    )
    project_root: str = Field(min_length=1, max_length=600)
    python_package_name: str = Field(min_length=1, max_length=120)
    python_package_version: str = Field(min_length=1, max_length=80)
    python_requires: str = Field(min_length=1, max_length=80)
    frontend_package_name: str = Field(min_length=1, max_length=160)
    frontend_package_version: str = Field(min_length=1, max_length=80)
    frontend_private_package: bool
    packaging_commands: tuple[str, ...] = Field(min_length=2, max_length=2)
    environment_variable_keys: tuple[str, ...] = Field(min_length=1, max_length=40)
    runtime_placeholder_paths: tuple[str, ...] = Field(min_length=3, max_length=3)
    records: tuple[ProductionPackagingRecord, ...] = Field(min_length=5, max_length=5)
    record_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    surface_ids: tuple[PackagingSurfaceId, ...] = Field(min_length=5, max_length=5)
    ready_record_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    guarded_record_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    record_count: int = Field(ge=5, le=5)
    packaging_status: PackagingSurfaceStatus
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=18,
    )
    packaging_metadata_implemented: Literal[True] = True
    python_package_review_implemented: Literal[True] = True
    frontend_package_review_implemented: Literal[True] = True
    environment_template_review_implemented: Literal[True] = True
    runtime_placeholder_review_implemented: Literal[True] = True
    release_script_review_implemented: Literal[True] = True
    dependency_installation_implemented: Literal[False] = False
    package_build_executed: Literal[False] = False
    archive_creation_implemented: Literal[False] = False
    container_image_build_implemented: Literal[False] = False
    provider_provisioning_implemented: Literal[False] = False
    environment_variable_mutation_implemented: Literal[False] = False
    runtime_data_write_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    merge_push_tag_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_records(self) -> Self:
        if self.record_ids != tuple(record.record_id for record in self.records):
            raise ValueError("record_ids must match records")
        if self.surface_ids != tuple(record.surface_id for record in self.records):
            raise ValueError("surface_ids must match records")
        if self.surface_ids != _REQUIRED_SURFACES:
            raise ValueError("surface_ids must cover required packaging surfaces")
        if self.ready_record_ids != _record_ids_for_status(self.records, "ready"):
            raise ValueError("ready_record_ids must match records")
        if self.guarded_record_ids != _record_ids_for_status(self.records, "guarded"):
            raise ValueError("guarded_record_ids must match records")
        if self.record_count != len(self.records):
            raise ValueError("record_count must match records")
        if self.packaging_status != _plan_status(self.records):
            raise ValueError("packaging_status must match records")
        if self.packaging_commands != _PACKAGING_COMMANDS:
            raise ValueError("packaging_commands must remain documented only")
        return self


def build_production_packaging_plan(
    project_root: str | Path | None = None,
) -> ProductionPackagingPlan:
    """Inspect repository packaging metadata without installing or building."""

    root = Path(project_root or ".").resolve()
    pyproject_path = root / "pyproject.toml"
    package_path = root / "clients/nextjs/package.json"
    env_path = root / ".env.example"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    package_json = json.loads(package_path.read_text(encoding="utf-8"))
    env_keys = _env_keys(env_path)
    project = pyproject["project"]
    records = _records(
        root=root,
        pyproject=pyproject,
        package_json=package_json,
        env_keys=env_keys,
    )
    return ProductionPackagingPlan(
        project_root=str(root),
        python_package_name=str(project["name"]),
        python_package_version=str(project["version"]),
        python_requires=str(project["requires-python"]),
        frontend_package_name=str(package_json["name"]),
        frontend_package_version=str(package_json["version"]),
        frontend_private_package=bool(package_json.get("private")),
        packaging_commands=_PACKAGING_COMMANDS,
        environment_variable_keys=env_keys,
        runtime_placeholder_paths=_REQUIRED_RUNTIME_PLACEHOLDERS,
        records=records,
        record_ids=tuple(record.record_id for record in records),
        surface_ids=tuple(record.surface_id for record in records),
        ready_record_ids=_record_ids_for_status(records, "ready"),
        guarded_record_ids=_record_ids_for_status(records, "guarded"),
        record_count=len(records),
        packaging_status=_plan_status(records),
    )


def production_packaging_record_by_surface(
    surface_id: PackagingSurfaceId | str,
    plan: ProductionPackagingPlan | None = None,
) -> ProductionPackagingRecord | None:
    """Return one packaging record by surface id."""

    normalized = str(surface_id).strip()
    source_plan = plan or build_production_packaging_plan()
    for record in source_plan.records:
        if record.surface_id == normalized:
            return record
    return None


def production_packaging_records_for_status(
    status: PackagingSurfaceStatus,
    plan: ProductionPackagingPlan | None = None,
) -> tuple[ProductionPackagingRecord, ...]:
    """Return packaging records by status."""

    source_plan = plan or build_production_packaging_plan()
    return tuple(record for record in source_plan.records if record.status == status)


def _records(
    *,
    root: Path,
    pyproject: dict[str, object],
    package_json: dict[str, object],
    env_keys: tuple[str, ...],
) -> tuple[ProductionPackagingRecord, ...]:
    project = pyproject.get("project", {})
    build_system = pyproject.get("build-system", {})
    tool = pyproject.get("tool", {})
    scripts = package_json.get("scripts", {})
    return (
        _record(
            surface_id="python_package_metadata",
            source_paths=("pyproject.toml",),
            required_items=(
                "build-system.requires",
                "build-system.build-backend",
                "project.name",
                "project.version",
                "project.dependencies",
                "tool.setuptools.packages.find",
            ),
            present_items=_present_items(
                (
                    ("build-system.requires", _has_key(build_system, "requires")),
                    (
                        "build-system.build-backend",
                        _has_key(build_system, "build-backend"),
                    ),
                    ("project.name", _has_key(project, "name")),
                    ("project.version", _has_key(project, "version")),
                    ("project.dependencies", _has_key(project, "dependencies")),
                    (
                        "tool.setuptools.packages.find",
                        "setuptools" in tool
                        and "packages" in tool["setuptools"]
                        and "find" in tool["setuptools"]["packages"],
                    ),
                )
            ),
            evidence=(
                f"python_package:{project.get('name', 'missing')}",
                f"python_version:{project.get('version', 'missing')}",
                f"requires_python:{project.get('requires-python', 'missing')}",
            ),
            packaging_actions=(
                "Package with setuptools metadata already declared.",
                "Keep dependency installation operator-controlled.",
            ),
        ),
        _record(
            surface_id="frontend_package_metadata",
            source_paths=("clients/nextjs/package.json",),
            required_items=_REQUIRED_FRONTEND_SCRIPTS,
            present_items=tuple(
                script for script in _REQUIRED_FRONTEND_SCRIPTS if script in scripts
            ),
            evidence=(
                f"frontend_package:{package_json.get('name', 'missing')}",
                f"frontend_version:{package_json.get('version', 'missing')}",
                f"private:{bool(package_json.get('private'))}",
            ),
            packaging_actions=(
                "Use declared Next.js scripts for build and start validation.",
                "Do not add automatic npm install or deployment hooks.",
            ),
        ),
        _record(
            surface_id="environment_template",
            source_paths=(".env.example",),
            required_items=_REQUIRED_ENV_KEYS,
            present_items=tuple(key for key in _REQUIRED_ENV_KEYS if key in env_keys),
            evidence=(
                f"env_keys:{len(env_keys)}",
                "provider_key_template:OPENAI_API_KEY",
                "local_paths_template:data",
            ),
            packaging_actions=(
                "Keep API key and provider configuration explicit.",
                "Fail safely when required production configuration is absent.",
            ),
        ),
        _record(
            surface_id="runtime_data_placeholders",
            source_paths=_REQUIRED_RUNTIME_PLACEHOLDERS,
            required_items=_REQUIRED_RUNTIME_PLACEHOLDERS,
            present_items=tuple(
                path
                for path in _REQUIRED_RUNTIME_PLACEHOLDERS
                if (root / path).exists()
            ),
            evidence=(
                "chroma_placeholder:data/chroma/.gitkeep",
                "artifact_placeholder:data/artifacts/.gitkeep",
                "eval_placeholder:data/eval/.gitkeep",
            ),
            packaging_actions=(
                "Keep runtime data directories explicit and empty by default.",
                "Do not write packaged runtime data during readiness checks.",
            ),
        ),
        _record(
            surface_id="release_scripts",
            source_paths=_REQUIRED_SCRIPT_PATHS,
            required_items=_REQUIRED_SCRIPT_PATHS,
            present_items=tuple(
                path for path in _REQUIRED_SCRIPT_PATHS if (root / path).exists()
            ),
            evidence=(
                "kb_sync_script:scripts/sync_official_kb.py",
                "eval_script:scripts/eval_live_sessions.py",
                "latest_eval_script:scripts/run_eval_latest.sh",
            ),
            packaging_actions=(
                "Document script availability without running release automation.",
                "Keep merge, push, and tag outside Codex-controlled packaging.",
            ),
        ),
    )


def _record(
    *,
    surface_id: PackagingSurfaceId,
    source_paths: tuple[str, ...],
    required_items: tuple[str, ...],
    present_items: tuple[str, ...],
    evidence: tuple[str, ...],
    packaging_actions: tuple[str, ...],
) -> ProductionPackagingRecord:
    missing = tuple(item for item in required_items if item not in present_items)
    return ProductionPackagingRecord(
        record_id=f"production_packaging::{surface_id}",
        surface_id=surface_id,
        status="guarded" if missing else "ready",
        source_paths=source_paths,
        required_items=required_items,
        present_items=present_items,
        missing_items=missing,
        evidence=evidence,
        packaging_actions=packaging_actions,
    )


def _env_keys(path: Path) -> tuple[str, ...]:
    keys: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        if key and key not in keys:
            keys.append(key)
    return tuple(keys)


def _present_items(items: tuple[tuple[str, bool], ...]) -> tuple[str, ...]:
    return tuple(item for item, present in items if present)


def _has_key(source: object, key: str) -> bool:
    return isinstance(source, dict) and key in source and bool(source[key])


def _record_ids_for_status(
    records: tuple[ProductionPackagingRecord, ...],
    status: PackagingSurfaceStatus,
) -> tuple[str, ...]:
    return tuple(record.record_id for record in records if record.status == status)


def _plan_status(
    records: tuple[ProductionPackagingRecord, ...],
) -> PackagingSurfaceStatus:
    if any(record.status == "guarded" for record in records):
        return "guarded"
    return "ready"
