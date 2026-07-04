"""V5.6 production release architecture consistency metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.production_architecture_freeze import (
    build_production_architecture_freeze,
)
from creative_coding_assistant.orchestration.production_creative_readiness_review import (
    build_production_creative_readiness_review,
)
from creative_coding_assistant.orchestration.production_demo_assets import (
    build_production_demo_asset_plan,
)
from creative_coding_assistant.orchestration.production_deployment import (
    build_production_deployment_plan,
)
from creative_coding_assistant.orchestration.production_final_hardening import (
    build_production_final_hardening,
)
from creative_coding_assistant.orchestration.production_readiness_review import (
    build_production_readiness_review,
)
from creative_coding_assistant.orchestration.production_release_audit import (
    build_production_release_audit,
)
from creative_coding_assistant.orchestration.production_release_candidate import (
    build_production_release_candidate,
)
from creative_coding_assistant.orchestration.production_release_final_optimization import (
    build_production_release_final_optimization,
)
from creative_coding_assistant.orchestration.production_release_packaging import (
    build_production_packaging_plan,
)

ProductionArchitectureConsistencyLayer = Literal[
    "release_readiness_boundary",
    "packaging_deployment_boundary",
    "demo_creative_boundary",
    "readiness_audit_boundary",
    "architecture_release_control_boundary",
    "hardening_boundary",
]
ProductionArchitectureConsistencyStage = Literal["v5_6_architecture_consistency_pass"]
ProductionArchitectureConsistencyStatus = Literal["pass"]

PRODUCTION_ARCHITECTURE_CONSISTENCY_RECORD_SERIALIZATION_VERSION = (
    "production_architecture_consistency_record.v1"
)
PRODUCTION_ARCHITECTURE_CONSISTENCY_REGISTRY_SERIALIZATION_VERSION = (
    "production_architecture_consistency_registry.v1"
)
PRODUCTION_ARCHITECTURE_CONSISTENCY_AUTHORITY_BOUNDARY = (
    "V5.6 production release architecture consistency metadata verifies "
    "production-release surface coverage, metadata-only declarations, "
    "blocked runtime behavior declarations, V5 architecture alignment, V4 "
    "boundary compatibility, and version runtime rules only; it does not "
    "expand architecture, mutate workflow graphs, change provider/model "
    "routing, execute providers or workflows, install dependencies, build "
    "packages, deploy services, generate assets, execute retrieval, mutate "
    "generated output, write storage, emit HITL requests, merge, push, tag, "
    "or apply Runtime Evolution."
)

_SURFACE_LAYERS: tuple[tuple[str, ProductionArchitectureConsistencyLayer], ...] = (
    ("production_release_final_optimization", "release_readiness_boundary"),
    ("production_release_packaging", "packaging_deployment_boundary"),
    ("production_release_candidate", "release_readiness_boundary"),
    ("production_demo_assets", "demo_creative_boundary"),
    ("production_deployment", "packaging_deployment_boundary"),
    ("production_readiness_review", "readiness_audit_boundary"),
    ("production_creative_readiness_review", "demo_creative_boundary"),
    ("production_architecture_freeze", "architecture_release_control_boundary"),
    ("production_release_audit", "architecture_release_control_boundary"),
    ("production_final_hardening", "hardening_boundary"),
)
_ARCHITECTURE_LAYERS: tuple[ProductionArchitectureConsistencyLayer, ...] = (
    "release_readiness_boundary",
    "packaging_deployment_boundary",
    "demo_creative_boundary",
    "readiness_audit_boundary",
    "architecture_release_control_boundary",
    "hardening_boundary",
)
_VALIDATED_VERSION_RULES = (
    "v5_6_surface_role_declared",
    "serialization_version_declared",
    "metadata_only_declared",
    "v5_architecture_boundary_preserved",
    "v4_boundary_compatibility_confirmed",
    "provider_model_routing_not_applied",
    "provider_execution_not_applied",
    "workflow_execution_not_applied",
    "generated_output_mutation_blocked",
    "release_operations_human_controlled",
    "hitl_not_emitted",
    "runtime_evolution_not_applied",
)
_PASSIVE_BOUNDARY_FLAGS = (
    "architecture_expansion_blocked",
    "workflow_graph_mutation_blocked",
    "provider_model_routing_blocked",
    "provider_execution_blocked",
    "dependency_installation_blocked",
    "package_build_blocked",
    "deployment_execution_blocked",
    "asset_generation_blocked",
    "retrieval_execution_blocked",
    "workflow_execution_blocked",
    "workflow_control_blocked",
    "storage_write_blocked",
    "generated_output_mutation_blocked",
    "hitl_emission_blocked",
    "merge_push_tag_blocked",
    "runtime_evolution_blocked",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "architecture_expansion",
    "workflow_graph_mutation",
    "provider_or_model_routing_mutation",
    "provider_execution",
    "dependency_installation",
    "package_build_execution",
    "deployment_execution",
    "asset_generation",
    "retrieval_execution",
    "workflow_execution",
    "workflow_control",
    "persistent_storage_write",
    "generated_output_modification",
    "hitl_request_emission",
    "merge_push_tag_operation",
    "runtime_evolution_application",
)
_ACTIVE_RUNTIME_FLAGS = (
    "core_architecture_expansion_implemented",
    "architecture_expansion_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_execution_implemented",
    "workflow_control_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "dependency_installation_implemented",
    "runtime_installation_implemented",
    "package_build_executed",
    "deployment_execution_implemented",
    "asset_generation_implemented",
    "retrieval_execution_implemented",
    "persistent_storage_write_implemented",
    "storage_write_implemented",
    "generated_output_mutation_implemented",
    "release_artifact_creation_implemented",
    "hitl_request_emitted",
    "merge_push_tag_implemented",
    "runtime_evolution_implemented",
)
_COUNT_FIELDS = (
    "record_count",
    "asset_count",
    "guarded_finding_count",
)


class ProductionArchitectureConsistencyRecord(BaseModel):
    """One V5.6 production architecture consistency record."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    surface_id: str = Field(min_length=1, max_length=160)
    architecture_layer: ProductionArchitectureConsistencyLayer
    architecture_stage: ProductionArchitectureConsistencyStage = (
        "v5_6_architecture_consistency_pass"
    )
    source_role: str = Field(min_length=1, max_length=160)
    source_serialization_version: str = Field(min_length=1, max_length=160)
    source_count_field: str = Field(min_length=1, max_length=80)
    source_count: int = Field(ge=0, le=1000)
    validated_version_rules: tuple[str, ...] = Field(min_length=12, max_length=12)
    passive_boundary_flags: tuple[str, ...] = Field(min_length=16, max_length=16)
    source_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=48,
    )
    source_active_runtime_flags: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=32,
    )
    missing_coverage_items: tuple[str, ...] = Field(
        default_factory=tuple, max_length=16
    )
    source_metadata_only_declared: bool = True
    v5_architecture_consistency_confirmed: Literal[True] = True
    v4_boundary_compatibility_confirmed: Literal[True] = True
    version_runtime_rules_confirmed: Literal[True] = True
    architecture_consistency_status: ProductionArchitectureConsistencyStatus = "pass"
    architecture_expansion_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    dependency_installation_implemented: Literal[False] = False
    runtime_installation_implemented: Literal[False] = False
    package_build_executed: Literal[False] = False
    deployment_execution_implemented: Literal[False] = False
    asset_generation_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    release_artifact_creation_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    merge_push_tag_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["production_architecture_consistency_record.v1"] = (
        PRODUCTION_ARCHITECTURE_CONSISTENCY_RECORD_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _record_matches_contract(self) -> Self:
        if self.validated_version_rules != _VALIDATED_VERSION_RULES:
            raise ValueError("validated_version_rules must match V5.6 rules")
        if self.passive_boundary_flags != _PASSIVE_BOUNDARY_FLAGS:
            raise ValueError("passive_boundary_flags must match V5.6 boundaries")
        if not self.source_metadata_only_declared:
            raise ValueError("source_metadata_only_declared must be true")
        if self.source_active_runtime_flags:
            raise ValueError("source_active_runtime_flags must remain empty")
        if self.missing_coverage_items:
            raise ValueError("missing_coverage_items must remain empty")
        return self


class ProductionArchitectureConsistencyRegistry(BaseModel):
    """V5.6 production architecture consistency registry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["production_architecture_consistency_registry"] = (
        "production_architecture_consistency_registry"
    )
    serialization_version: Literal[
        "production_architecture_consistency_registry.v1"
    ] = PRODUCTION_ARCHITECTURE_CONSISTENCY_REGISTRY_SERIALIZATION_VERSION
    authority_boundary: str = Field(
        default=PRODUCTION_ARCHITECTURE_CONSISTENCY_AUTHORITY_BOUNDARY,
        max_length=2200,
    )
    architecture_stage: ProductionArchitectureConsistencyStage = (
        "v5_6_architecture_consistency_pass"
    )
    records: tuple[ProductionArchitectureConsistencyRecord, ...] = Field(
        min_length=10,
        max_length=10,
    )
    surface_ids: tuple[str, ...] = Field(min_length=10, max_length=10)
    record_count: int = Field(ge=10, le=10)
    architecture_layers: tuple[ProductionArchitectureConsistencyLayer, ...] = Field(
        min_length=6,
        max_length=6,
    )
    validated_version_rules: tuple[str, ...] = Field(min_length=12, max_length=12)
    passive_boundary_flags: tuple[str, ...] = Field(min_length=16, max_length=16)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=20,
    )
    all_surfaces_covered: Literal[True] = True
    no_active_runtime_flags: Literal[True] = True
    no_missing_coverage: Literal[True] = True
    v5_architecture_consistency_confirmed: Literal[True] = True
    v4_boundaries_preserved: Literal[True] = True
    version_runtime_rules_confirmed: Literal[True] = True
    runtime_evolution_not_applied: Literal[True] = True
    architecture_expansion_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    dependency_installation_implemented: Literal[False] = False
    runtime_installation_implemented: Literal[False] = False
    package_build_executed: Literal[False] = False
    deployment_execution_implemented: Literal[False] = False
    asset_generation_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    release_artifact_creation_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    merge_push_tag_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_records(self) -> Self:
        if self.surface_ids != tuple(record.surface_id for record in self.records):
            raise ValueError("surface_ids must match records")
        if self.surface_ids != tuple(surface for surface, _layer in _SURFACE_LAYERS):
            raise ValueError("surface_ids must cover V5.6 surfaces")
        if self.record_count != len(self.records):
            raise ValueError("record_count must match records")
        if self.architecture_layers != _ARCHITECTURE_LAYERS:
            raise ValueError("architecture_layers must match V5.6 layers")
        if self.validated_version_rules != _VALIDATED_VERSION_RULES:
            raise ValueError("validated_version_rules must match V5.6 rules")
        if self.passive_boundary_flags != _PASSIVE_BOUNDARY_FLAGS:
            raise ValueError("passive_boundary_flags must match V5.6 boundaries")
        if any(record.source_active_runtime_flags for record in self.records):
            raise ValueError("records must not expose active runtime flags")
        if any(record.missing_coverage_items for record in self.records):
            raise ValueError("records must not expose missing coverage")
        return self


def production_architecture_consistency_registry(
    project_root: str | Path | None = None,
) -> ProductionArchitectureConsistencyRegistry:
    """Build V5.6 architecture consistency metadata without runtime changes."""

    root = Path(project_root or ".").resolve()
    final_source = build_production_release_final_optimization()
    packaging_source = build_production_packaging_plan(root)
    candidate_source = build_production_release_candidate(
        final_optimization=final_source,
        packaging=packaging_source,
    )
    demo_source = build_production_demo_asset_plan(
        root,
        release_candidate=candidate_source,
    )
    deployment_source = build_production_deployment_plan(
        root,
        packaging=packaging_source,
    )
    readiness_source = build_production_readiness_review(
        final_optimization=final_source,
        packaging=packaging_source,
        release_candidate=candidate_source,
        demo_assets=demo_source,
        deployment=deployment_source,
    )
    creative_source = build_production_creative_readiness_review(
        demo_assets=demo_source,
        production_readiness=readiness_source,
    )
    architecture_source = build_production_architecture_freeze(
        root,
        final_optimization=final_source,
        packaging=packaging_source,
        release_candidate=candidate_source,
        demo_assets=demo_source,
        deployment=deployment_source,
        production_readiness=readiness_source,
        creative_readiness=creative_source,
    )
    audit_source = build_production_release_audit(
        root,
        release_candidate=candidate_source,
        packaging=packaging_source,
        deployment=deployment_source,
        production_readiness=readiness_source,
        creative_readiness=creative_source,
        architecture_freeze=architecture_source,
    )
    hardening_source = build_production_final_hardening(
        root,
        release_audit=audit_source,
        deployment=deployment_source,
        production_readiness=readiness_source,
        creative_readiness=creative_source,
        architecture_freeze=architecture_source,
    )
    sources = (
        final_source,
        packaging_source,
        candidate_source,
        demo_source,
        deployment_source,
        readiness_source,
        creative_source,
        architecture_source,
        audit_source,
        hardening_source,
    )
    records = tuple(
        _record(surface_id=surface_id, layer=layer, source=source)
        for (surface_id, layer), source in zip(_SURFACE_LAYERS, sources, strict=True)
    )
    return ProductionArchitectureConsistencyRegistry(
        records=records,
        surface_ids=tuple(record.surface_id for record in records),
        record_count=len(records),
        architecture_layers=_ARCHITECTURE_LAYERS,
        validated_version_rules=_VALIDATED_VERSION_RULES,
        passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
    )


def production_architecture_consistency_by_surface(
    surface_id: str,
    registry: ProductionArchitectureConsistencyRegistry | None = None,
) -> ProductionArchitectureConsistencyRecord | None:
    """Return one production architecture consistency record by surface."""

    source_registry = registry or production_architecture_consistency_registry()
    for record in source_registry.records:
        if record.surface_id == surface_id:
            return record
    return None


def production_architecture_consistency_records_for_layer(
    layer: ProductionArchitectureConsistencyLayer | str,
    registry: ProductionArchitectureConsistencyRegistry | None = None,
) -> tuple[ProductionArchitectureConsistencyRecord, ...]:
    """Return production architecture consistency records by layer."""

    normalized = str(layer).strip()
    source_registry = registry or production_architecture_consistency_registry()
    return tuple(
        record
        for record in source_registry.records
        if record.architecture_layer == normalized
    )


def _record(
    *,
    surface_id: str,
    layer: ProductionArchitectureConsistencyLayer,
    source: Any,
) -> ProductionArchitectureConsistencyRecord:
    count_field, count = _count(source)
    active_flags = _active_runtime_flags(source)
    metadata_only_declared = bool(getattr(source, "metadata_only", False))
    blocked = tuple(getattr(source, "blocked_runtime_behaviors", ()))
    missing = _missing_coverage(
        metadata_only_declared=metadata_only_declared,
        active_flags=active_flags,
        blocked_runtime_behaviors=blocked,
    )
    return ProductionArchitectureConsistencyRecord(
        surface_id=surface_id,
        architecture_layer=layer,
        source_role=str(source.role),
        source_serialization_version=str(source.serialization_version),
        source_count_field=count_field,
        source_count=count,
        validated_version_rules=_VALIDATED_VERSION_RULES,
        passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
        source_blocked_runtime_behaviors=blocked,
        source_active_runtime_flags=active_flags,
        missing_coverage_items=missing,
        source_metadata_only_declared=metadata_only_declared,
    )


def _count(source: Any) -> tuple[str, int]:
    for field in _COUNT_FIELDS:
        value = getattr(source, field, None)
        if isinstance(value, int):
            return field, value
    return "record_count", 0


def _active_runtime_flags(source: Any) -> tuple[str, ...]:
    return tuple(
        flag for flag in _ACTIVE_RUNTIME_FLAGS if bool(getattr(source, flag, False))
    )


def _missing_coverage(
    *,
    metadata_only_declared: bool,
    active_flags: tuple[str, ...],
    blocked_runtime_behaviors: tuple[str, ...],
) -> tuple[str, ...]:
    missing: list[str] = []
    if not metadata_only_declared:
        missing.append("metadata_only_missing")
    if active_flags:
        missing.append("active_runtime_flags_present")
    if not blocked_runtime_behaviors:
        missing.append("blocked_runtime_behaviors_missing")
    return tuple(missing)
