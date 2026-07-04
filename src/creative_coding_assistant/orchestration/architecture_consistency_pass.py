"""Passive V4.6 architecture consistency pass metadata."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_collaboration_audit import (
    agent_collaboration_audit_registry,
)
from creative_coding_assistant.orchestration.agent_contract_audit import (
    agent_contract_audit_registry,
)
from creative_coding_assistant.orchestration.agent_cost_tracking_foundation import (
    agent_cost_tracking_foundation_registry,
)
from creative_coding_assistant.orchestration.agent_determinism_audit import (
    agent_determinism_audit_registry,
)
from creative_coding_assistant.orchestration.agent_explainability_audit import (
    agent_explainability_audit_registry,
)
from creative_coding_assistant.orchestration.agent_performance_tracking_foundation import (
    agent_performance_tracking_foundation_registry,
)
from creative_coding_assistant.orchestration.agent_registry_audit import (
    agent_registry_audit_registry,
)
from creative_coding_assistant.orchestration.agent_reliability_audit import (
    agent_reliability_audit_registry,
)
from creative_coding_assistant.orchestration.agent_telemetry_foundation import (
    agent_telemetry_foundation_registry,
)
from creative_coding_assistant.orchestration.blackboard_audit import (
    blackboard_audit_registry,
)
from creative_coding_assistant.orchestration.creative_diversity_audit import (
    creative_diversity_audit_registry,
)
from creative_coding_assistant.orchestration.engine_contract_consistency import (
    engine_contract_consistency_registry,
)
from creative_coding_assistant.orchestration.escalation_policy_audit import (
    escalation_policy_audit_registry,
)
from creative_coding_assistant.orchestration.hybrid_workflow_audit import (
    hybrid_workflow_audit_registry,
)
from creative_coding_assistant.orchestration.shared_context_audit import (
    shared_context_audit_registry,
)

ArchitectureConsistencyLayer = Literal[
    "agent_contract_boundary",
    "registry_boundary",
    "shared_context_boundary",
    "collaboration_boundary",
    "hybrid_workflow_boundary",
    "hardening_quality_boundary",
    "foundation_observability_boundary",
    "engine_contract_boundary",
]
ArchitectureConsistencyStage = Literal["v4_6_architecture_consistency_pass"]
ArchitectureConsistencyStatus = Literal["pass"]

ARCHITECTURE_CONSISTENCY_RECORD_SERIALIZATION_VERSION = (
    "architecture_consistency_record.v1"
)
ARCHITECTURE_CONSISTENCY_PASS_REGISTRY_SERIALIZATION_VERSION = (
    "architecture_consistency_pass_registry.v1"
)
ARCHITECTURE_CONSISTENCY_PASS_AUTHORITY_BOUNDARY = (
    "V4.6 architecture consistency pass metadata checks passive hardening "
    "registry serialization, role identity, metadata-only declarations, "
    "source counts, architecture documentation references, blocked runtime "
    "behavior declarations, and active flag absence only; it does not rewrite "
    "architecture documents, mutate workflow graphs, generate prompts, select "
    "providers or models, select runtimes, invoke agents, execute artifacts, "
    "trigger retries, write memory or storage, or modify generated output."
)

_ARCHITECTURE_DOC_REFS = (
    "README.md",
    "docs/PROJECT_CONTEXT.md",
    "docs/IMPLEMENTATION_ROADMAP.md",
    "docs/ARCHITECTURE_DECISIONS.md",
    "architecture/workflow_graph.md",
    "architecture/engine_matrix.md",
)
_VALIDATED_ARCHITECTURE_SURFACES = (
    "registry_role",
    "serialization_version",
    "metadata_only",
    "authority_boundary",
    "source_count",
    "architecture_doc_refs",
    "blocked_runtime_behaviors",
    "active_runtime_flags",
)
_PASSIVE_BOUNDARY_FLAGS = (
    "architecture_doc_rewrite_blocked",
    "workflow_graph_mutation_blocked",
    "prompt_generation_blocked",
    "provider_model_routing_blocked",
    "runtime_selection_blocked",
    "agent_invocation_blocked",
    "artifact_execution_blocked",
    "retry_triggering_blocked",
    "storage_write_blocked",
    "generated_output_mutation_blocked",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "architecture_doc_rewrite",
    "workflow_graph_mutation",
    "prompt_generation",
    "provider_or_model_routing",
    "runtime_selection",
    "agent_invocation",
    "artifact_execution",
    "retry_or_refinement_triggering",
    "memory_or_storage_write",
    "generated_output_modification",
)
_CONSISTENCY_FINDINGS = (
    "registry_role_confirmed",
    "serialization_version_confirmed",
    "metadata_only_declaration_confirmed",
    "authority_boundary_confirmed",
    "source_count_confirmed",
    "architecture_doc_refs_confirmed",
    "runtime_behavior_blocks_confirmed",
    "active_runtime_flags_absent",
)

_COUNT_FIELDS = (
    "audit_count",
    "profile_count",
    "family_count",
    "metadata_count",
    "contract_count",
    "stage_count",
)


class ArchitectureConsistencyRecord(BaseModel):
    """One passive architecture consistency record for a hardening registry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    source_registry_id: str = Field(min_length=1, max_length=120)
    architecture_layer: ArchitectureConsistencyLayer
    architecture_stage: ArchitectureConsistencyStage = (
        "v4_6_architecture_consistency_pass"
    )
    source_role: str = Field(min_length=1, max_length=120)
    source_serialization_version: str = Field(min_length=1, max_length=120)
    source_count_field: str = Field(min_length=1, max_length=80)
    source_count: int = Field(ge=1, le=64)
    architecture_doc_refs: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=32,
    )
    source_active_runtime_flags: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=24,
    )
    validated_architecture_surfaces: tuple[str, ...] = Field(
        min_length=8,
        max_length=8,
    )
    passive_boundary_flags: tuple[str, ...] = Field(min_length=10, max_length=10)
    consistency_findings: tuple[str, ...] = Field(min_length=8, max_length=8)
    missing_coverage_items: tuple[str, ...] = Field(
        default_factory=tuple, max_length=20
    )
    source_metadata_only_declared: Literal[True] = True
    architecture_consistency_status: ArchitectureConsistencyStatus = "pass"
    architecture_doc_rewrite_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    prompt_generation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    artifact_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["architecture_consistency_record.v1"] = (
        ARCHITECTURE_CONSISTENCY_RECORD_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class ArchitectureConsistencyPassRegistry(BaseModel):
    """Stable passive V4.6 architecture consistency pass registry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["architecture_consistency_pass_registry"] = (
        "architecture_consistency_pass_registry"
    )
    serialization_version: Literal["architecture_consistency_pass_registry.v1"] = (
        ARCHITECTURE_CONSISTENCY_PASS_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=ARCHITECTURE_CONSISTENCY_PASS_AUTHORITY_BOUNDARY,
        max_length=1200,
    )
    architecture_stage: ArchitectureConsistencyStage = (
        "v4_6_architecture_consistency_pass"
    )
    records: tuple[ArchitectureConsistencyRecord, ...] = Field(
        min_length=15,
        max_length=15,
    )
    source_registry_ids: tuple[str, ...] = Field(min_length=15, max_length=15)
    record_count: int = Field(ge=15, le=15)
    architecture_layers: tuple[ArchitectureConsistencyLayer, ...] = Field(
        min_length=8,
        max_length=8,
    )
    architecture_doc_refs: tuple[str, ...] = Field(min_length=6, max_length=6)
    validated_architecture_surfaces: tuple[str, ...] = Field(
        min_length=8,
        max_length=8,
    )
    passive_boundary_flags: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    all_sources_covered: Literal[True] = True
    architecture_docs_referenced: Literal[True] = True
    no_active_runtime_flags: Literal[True] = True
    no_missing_coverage: Literal[True] = True
    architecture_doc_rewrite_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    prompt_generation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    artifact_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_records(self) -> Self:
        derived_source_ids = tuple(record.source_registry_id for record in self.records)
        if len(set(derived_source_ids)) != len(derived_source_ids):
            raise ValueError("source_registry_ids must be unique")
        if self.source_registry_ids != derived_source_ids:
            raise ValueError("source_registry_ids must match records")
        if self.record_count != len(self.records):
            raise ValueError("record_count must match records")
        if self.source_registry_ids != _source_registry_ids():
            raise ValueError("source_registry_ids must match hardening source order")
        if self.architecture_layers != _architecture_layers():
            raise ValueError("architecture_layers must match source layer order")
        if self.architecture_doc_refs != _ARCHITECTURE_DOC_REFS:
            raise ValueError("architecture_doc_refs must match references")
        if self.validated_architecture_surfaces != _VALIDATED_ARCHITECTURE_SURFACES:
            raise ValueError("validated_architecture_surfaces must match registry")
        if self.passive_boundary_flags != _PASSIVE_BOUNDARY_FLAGS:
            raise ValueError("passive_boundary_flags must match registry")
        if any(
            (
                self.architecture_doc_rewrite_implemented,
                self.workflow_graph_mutation_implemented,
                self.prompt_generation_implemented,
                self.provider_model_routing_implemented,
                self.runtime_selection_implemented,
                self.agent_invocation_implemented,
                self.artifact_execution_implemented,
                self.retry_triggering_implemented,
                self.storage_write_implemented,
                self.generated_output_mutation_implemented,
            )
        ):
            raise ValueError("architecture consistency pass must remain passive")

        known_layers = set(self.architecture_layers)
        for record in self.records:
            if record.architecture_stage != self.architecture_stage:
                raise ValueError("architecture_stage must match registry")
            if record.architecture_layer not in known_layers:
                raise ValueError("architecture_layer must be known")
            if record.architecture_doc_refs != self.architecture_doc_refs:
                raise ValueError("architecture_doc_refs must match registry")
            if record.validated_architecture_surfaces != (
                self.validated_architecture_surfaces
            ):
                raise ValueError("validated_architecture_surfaces must match registry")
            if record.passive_boundary_flags != self.passive_boundary_flags:
                raise ValueError("passive_boundary_flags must match registry")
            if record.source_active_runtime_flags:
                raise ValueError("records must not contain active runtime flags")
            if record.missing_coverage_items:
                raise ValueError("records must not contain missing coverage")
        return self


def architecture_consistency_pass_registry() -> ArchitectureConsistencyPassRegistry:
    """Return passive V4.6 architecture consistency pass metadata."""

    return ARCHITECTURE_CONSISTENCY_PASS_REGISTRY


def architecture_consistency_record_by_source_registry(
    source_registry_id: str,
    registry: ArchitectureConsistencyPassRegistry | None = None,
) -> ArchitectureConsistencyRecord | None:
    """Return one architecture consistency record by source registry id."""

    source_registry = registry or ARCHITECTURE_CONSISTENCY_PASS_REGISTRY
    normalized_source_id = str(source_registry_id).strip()
    for record in source_registry.records:
        if record.source_registry_id == normalized_source_id:
            return record
    return None


def architecture_consistency_records_for_layer(
    architecture_layer: str,
    registry: ArchitectureConsistencyPassRegistry | None = None,
) -> tuple[ArchitectureConsistencyRecord, ...]:
    """Return passive architecture consistency records for one layer."""

    source_registry = registry or ARCHITECTURE_CONSISTENCY_PASS_REGISTRY
    normalized_layer = str(architecture_layer).strip()
    return tuple(
        record
        for record in source_registry.records
        if record.architecture_layer == normalized_layer
    )


_SOURCE_SPECS: tuple[
    tuple[str, ArchitectureConsistencyLayer, Callable[[], Any]],
    ...,
] = (
    (
        "agent_contract_audit_registry",
        "agent_contract_boundary",
        agent_contract_audit_registry,
    ),
    (
        "escalation_policy_audit_registry",
        "hybrid_workflow_boundary",
        escalation_policy_audit_registry,
    ),
    (
        "hybrid_workflow_audit_registry",
        "hybrid_workflow_boundary",
        hybrid_workflow_audit_registry,
    ),
    (
        "agent_registry_audit_registry",
        "registry_boundary",
        agent_registry_audit_registry,
    ),
    (
        "blackboard_audit_registry",
        "shared_context_boundary",
        blackboard_audit_registry,
    ),
    (
        "shared_context_audit_registry",
        "shared_context_boundary",
        shared_context_audit_registry,
    ),
    (
        "agent_collaboration_audit_registry",
        "collaboration_boundary",
        agent_collaboration_audit_registry,
    ),
    (
        "creative_diversity_audit_registry",
        "hybrid_workflow_boundary",
        creative_diversity_audit_registry,
    ),
    (
        "agent_explainability_audit_registry",
        "hardening_quality_boundary",
        agent_explainability_audit_registry,
    ),
    (
        "agent_reliability_audit_registry",
        "hardening_quality_boundary",
        agent_reliability_audit_registry,
    ),
    (
        "agent_determinism_audit_registry",
        "hardening_quality_boundary",
        agent_determinism_audit_registry,
    ),
    (
        "agent_telemetry_foundation_registry",
        "foundation_observability_boundary",
        agent_telemetry_foundation_registry,
    ),
    (
        "agent_cost_tracking_foundation_registry",
        "foundation_observability_boundary",
        agent_cost_tracking_foundation_registry,
    ),
    (
        "agent_performance_tracking_foundation_registry",
        "foundation_observability_boundary",
        agent_performance_tracking_foundation_registry,
    ),
    (
        "engine_contract_consistency_registry",
        "engine_contract_boundary",
        engine_contract_consistency_registry,
    ),
)


def _source_registry_ids() -> tuple[str, ...]:
    return tuple(source_id for source_id, _layer, _provider in _SOURCE_SPECS)


def _architecture_layers() -> tuple[ArchitectureConsistencyLayer, ...]:
    return tuple(dict.fromkeys(layer for _source_id, layer, _provider in _SOURCE_SPECS))


def _source_count(registry: Any) -> tuple[str, int]:
    for count_field in _COUNT_FIELDS:
        value = getattr(registry, count_field, None)
        if isinstance(value, int):
            return count_field, value
    for collection_field in (
        "records",
        "audit_records",
        "profiles",
        "families",
        "contracts",
    ):
        value = getattr(registry, collection_field, None)
        if value is not None:
            return collection_field, len(value)
    raise ValueError(f"missing source count for {registry!r}")


def _active_runtime_flags(registry: Any) -> tuple[str, ...]:
    dumped = registry.model_dump(mode="python")
    active_flags: list[str] = []
    for field_name, value in dumped.items():
        if not isinstance(value, bool) or not value:
            continue
        if field_name.endswith(
            (
                "_implemented",
                "_changed",
                "_performed",
                "_enabled",
            )
        ):
            active_flags.append(field_name)
    return tuple(active_flags)


def _missing_coverage_items(
    *,
    source_registry_id: str,
    registry: Any,
    source_count: int,
    active_flags: tuple[str, ...],
) -> tuple[str, ...]:
    missing: list[str] = []
    if getattr(registry, "role", "") != source_registry_id:
        missing.append("source_registry_role_mismatch")
    serialization_version = getattr(registry, "serialization_version", "")
    if not serialization_version.endswith(".v1"):
        missing.append("serialization_version_missing")
    if not getattr(registry, "metadata_only", False):
        missing.append("metadata_only_declaration_missing")
    if not getattr(registry, "authority_boundary", ""):
        missing.append("authority_boundary_missing")
    if source_count < 1:
        missing.append("source_count_missing")
    if not getattr(registry, "blocked_runtime_behaviors", ()):
        missing.append("blocked_runtime_behaviors_missing")
    if active_flags:
        missing.append("active_runtime_flags_present")
    return tuple(missing)


def _record(
    source_registry_id: str,
    architecture_layer: ArchitectureConsistencyLayer,
    provider: Callable[[], Any],
) -> ArchitectureConsistencyRecord:
    registry = provider()
    count_field, source_count = _source_count(registry)
    active_flags = _active_runtime_flags(registry)
    return ArchitectureConsistencyRecord(
        source_registry_id=source_registry_id,
        architecture_layer=architecture_layer,
        source_role=registry.role,
        source_serialization_version=registry.serialization_version,
        source_count_field=count_field,
        source_count=source_count,
        architecture_doc_refs=_ARCHITECTURE_DOC_REFS,
        source_blocked_runtime_behaviors=registry.blocked_runtime_behaviors,
        source_active_runtime_flags=active_flags,
        validated_architecture_surfaces=_VALIDATED_ARCHITECTURE_SURFACES,
        passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
        consistency_findings=_CONSISTENCY_FINDINGS,
        missing_coverage_items=_missing_coverage_items(
            source_registry_id=source_registry_id,
            registry=registry,
            source_count=source_count,
            active_flags=active_flags,
        ),
        source_metadata_only_declared=registry.metadata_only,
    )


ARCHITECTURE_CONSISTENCY_RECORDS = tuple(
    _record(source_id, layer, provider) for source_id, layer, provider in _SOURCE_SPECS
)
ARCHITECTURE_CONSISTENCY_PASS_REGISTRY = ArchitectureConsistencyPassRegistry(
    records=ARCHITECTURE_CONSISTENCY_RECORDS,
    source_registry_ids=tuple(
        record.source_registry_id for record in ARCHITECTURE_CONSISTENCY_RECORDS
    ),
    record_count=len(ARCHITECTURE_CONSISTENCY_RECORDS),
    architecture_layers=_architecture_layers(),
    architecture_doc_refs=_ARCHITECTURE_DOC_REFS,
    validated_architecture_surfaces=_VALIDATED_ARCHITECTURE_SURFACES,
    passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
)
