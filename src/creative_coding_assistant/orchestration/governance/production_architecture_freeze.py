"""V5.6 production architecture freeze metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.production_creative_readiness_review import (
    ProductionCreativeReadinessReview,
    build_production_creative_readiness_review,
)
from creative_coding_assistant.orchestration.production_demo_assets import (
    ProductionDemoAssetPlan,
    build_production_demo_asset_plan,
)
from creative_coding_assistant.orchestration.production_deployment import (
    ProductionDeploymentPlan,
    build_production_deployment_plan,
)
from creative_coding_assistant.orchestration.production_readiness_review import (
    ProductionReadinessReview,
    build_production_readiness_review,
)
from creative_coding_assistant.orchestration.production_release_candidate import (
    ProductionReleaseCandidatePlan,
    build_production_release_candidate,
)
from creative_coding_assistant.orchestration.production_release_final_optimization import (
    ProductionReleaseFinalOptimizationPlan,
    build_production_release_final_optimization,
)
from creative_coding_assistant.orchestration.production_release_packaging import (
    ProductionPackagingPlan,
    build_production_packaging_plan,
)

ProductionArchitectureFreezeDomain = Literal[
    "runtime_topology_freeze",
    "v5_6_metadata_surface_freeze",
    "provider_model_routing_freeze",
    "deployment_release_operations_freeze",
    "generated_output_boundary_freeze",
    "runtime_evolution_gate_freeze",
]
ProductionArchitectureFreezeStatus = Literal["frozen", "blocked"]

PRODUCTION_ARCHITECTURE_FREEZE_RECORD_SERIALIZATION_VERSION = (
    "production_architecture_freeze_record.v1"
)
PRODUCTION_ARCHITECTURE_FREEZE_SERIALIZATION_VERSION = (
    "production_architecture_freeze.v1"
)
PRODUCTION_ARCHITECTURE_FREEZE_AUTHORITY_BOUNDARY = (
    "V5.6 production architecture freeze metadata declares the current V5.6 "
    "release architecture frozen for production audit by composing existing "
    "readiness metadata and documented architecture boundaries only; it does "
    "not introduce new core architecture, mutate workflow graphs, change "
    "provider or model routing, execute providers, install runtimes, run "
    "package builds, deploy services, generate assets, execute retrieval, "
    "modify generated output, write storage, create release artifacts, emit "
    "HITL requests, merge, push, tag, or apply Runtime Evolution."
)

_SOURCE_SURFACE_IDS = (
    "production_release_final_optimization",
    "production_release_packaging",
    "production_release_candidate",
    "production_demo_assets",
    "production_deployment",
    "production_readiness_review",
    "production_creative_readiness_review",
)
_ARCHITECTURE_DOC_REFS = (
    "README.md",
    "docs/PROJECT_CONTEXT.md",
    "docs/IMPLEMENTATION_ROADMAP.md",
    "docs/ARCHITECTURE_DECISIONS.md",
)
_REQUIRED_DOMAINS: tuple[ProductionArchitectureFreezeDomain, ...] = (
    "runtime_topology_freeze",
    "v5_6_metadata_surface_freeze",
    "provider_model_routing_freeze",
    "deployment_release_operations_freeze",
    "generated_output_boundary_freeze",
    "runtime_evolution_gate_freeze",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "core_architecture_expansion",
    "workflow_graph_mutation",
    "workflow_execution",
    "workflow_control",
    "provider_or_model_routing_mutation",
    "provider_execution",
    "runtime_installation",
    "package_build_execution",
    "deployment_execution",
    "asset_generation",
    "retrieval_execution",
    "generated_output_modification",
    "persistent_storage_write",
    "release_artifact_creation",
    "hitl_request_emission",
    "merge_push_tag_operation",
    "runtime_evolution_application",
)


class ProductionArchitectureFreezeRecord(BaseModel):
    """One V5.6 production architecture freeze record."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    record_id: str = Field(min_length=1, max_length=180)
    freeze_domain: ProductionArchitectureFreezeDomain
    freeze_status: ProductionArchitectureFreezeStatus
    source_surface_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_serialization_versions: tuple[str, ...] = Field(min_length=1, max_length=8)
    architecture_doc_refs: tuple[str, ...] = Field(min_length=4, max_length=4)
    freeze_assertions: tuple[str, ...] = Field(min_length=1, max_length=12)
    guarded_assumptions: tuple[str, ...] = Field(default_factory=tuple, max_length=24)
    prohibited_changes: tuple[str, ...] = Field(min_length=1, max_length=24)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=12)
    architecture_change_required: Literal[False] = False
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    architecture_freeze_record_implemented: Literal[True] = True
    core_architecture_expansion_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    runtime_installation_implemented: Literal[False] = False
    package_build_executed: Literal[False] = False
    deployment_execution_implemented: Literal[False] = False
    asset_generation_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    release_artifact_creation_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    merge_push_tag_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["production_architecture_freeze_record.v1"] = (
        PRODUCTION_ARCHITECTURE_FREEZE_RECORD_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _record_matches_contract(self) -> Self:
        if self.record_id != f"production_architecture_freeze::{self.freeze_domain}":
            raise ValueError("record_id must match freeze_domain")
        if len(self.source_surface_ids) != len(self.source_serialization_versions):
            raise ValueError("source ids and serialization versions must align")
        if self.architecture_doc_refs != _ARCHITECTURE_DOC_REFS:
            raise ValueError("architecture_doc_refs must match freeze docs")
        if self.freeze_status != "frozen":
            raise ValueError("freeze_status must remain frozen")
        return self


class ProductionArchitectureFreeze(BaseModel):
    """V5.6 production architecture freeze declaration."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["production_architecture_freeze"] = "production_architecture_freeze"
    serialization_version: Literal["production_architecture_freeze.v1"] = (
        PRODUCTION_ARCHITECTURE_FREEZE_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=PRODUCTION_ARCHITECTURE_FREEZE_AUTHORITY_BOUNDARY,
        max_length=2200,
    )
    version: Literal["V5.6.0"] = "V5.6.0"
    target_branch: Literal["feature/production-release"] = "feature/production-release"
    target_tag: Literal["v5.6.0"] = "v5.6.0"
    source_surface_ids: tuple[str, ...] = Field(min_length=7, max_length=7)
    source_serialization_versions: tuple[str, ...] = Field(min_length=7, max_length=7)
    architecture_doc_refs: tuple[str, ...] = Field(min_length=4, max_length=4)
    missing_architecture_doc_refs: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    records: tuple[ProductionArchitectureFreezeRecord, ...] = Field(
        min_length=6,
        max_length=6,
    )
    record_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    freeze_domains: tuple[ProductionArchitectureFreezeDomain, ...] = Field(
        min_length=6,
        max_length=6,
    )
    frozen_record_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    record_count: int = Field(ge=6, le=6)
    guarded_assumption_count: int = Field(ge=0, le=100)
    prohibited_change_count: int = Field(ge=0, le=200)
    architecture_freeze_status: ProductionArchitectureFreezeStatus
    release_audit_can_proceed: Literal[True] = True
    runtime_evolution_hitl_required: Literal[True] = True
    no_architecture_expansion_required: Literal[True] = True
    architecture_freeze_implemented: Literal[True] = True
    source_surfaces_frozen: Literal[True] = True
    architecture_docs_referenced: Literal[True] = True
    guarded_assumptions_documented: Literal[True] = True
    core_architecture_expansion_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    runtime_installation_implemented: Literal[False] = False
    package_build_executed: Literal[False] = False
    deployment_execution_implemented: Literal[False] = False
    asset_generation_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    release_artifact_creation_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    merge_push_tag_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _freeze_matches_records(self) -> Self:
        if self.source_surface_ids != _SOURCE_SURFACE_IDS:
            raise ValueError("source_surface_ids must match V5.6 freeze surfaces")
        if len(self.source_surface_ids) != len(self.source_serialization_versions):
            raise ValueError("source ids and serialization versions must align")
        if self.architecture_doc_refs != _ARCHITECTURE_DOC_REFS:
            raise ValueError("architecture_doc_refs must match freeze docs")
        if self.record_ids != tuple(record.record_id for record in self.records):
            raise ValueError("record_ids must match records")
        if self.freeze_domains != tuple(
            record.freeze_domain for record in self.records
        ):
            raise ValueError("freeze_domains must match records")
        if self.freeze_domains != _REQUIRED_DOMAINS:
            raise ValueError("freeze_domains must cover required domains")
        if self.frozen_record_ids != tuple(record.record_id for record in self.records):
            raise ValueError("frozen_record_ids must match records")
        if self.record_count != len(self.records):
            raise ValueError("record_count must match records")
        if self.guarded_assumption_count != sum(
            len(record.guarded_assumptions) for record in self.records
        ):
            raise ValueError("guarded_assumption_count must match records")
        if self.prohibited_change_count != sum(
            len(record.prohibited_changes) for record in self.records
        ):
            raise ValueError("prohibited_change_count must match records")
        if self.architecture_freeze_status != "frozen":
            raise ValueError("architecture_freeze_status must remain frozen")
        return self


def build_production_architecture_freeze(
    project_root: str | Path | None = None,
    *,
    final_optimization: ProductionReleaseFinalOptimizationPlan | None = None,
    packaging: ProductionPackagingPlan | None = None,
    release_candidate: ProductionReleaseCandidatePlan | None = None,
    demo_assets: ProductionDemoAssetPlan | None = None,
    deployment: ProductionDeploymentPlan | None = None,
    production_readiness: ProductionReadinessReview | None = None,
    creative_readiness: ProductionCreativeReadinessReview | None = None,
) -> ProductionArchitectureFreeze:
    """Build architecture freeze metadata without changing runtime architecture."""

    root = Path(project_root or ".").resolve()
    final_source = final_optimization or build_production_release_final_optimization()
    packaging_source = packaging or build_production_packaging_plan(root)
    candidate_source = release_candidate or build_production_release_candidate(
        final_optimization=final_source,
        packaging=packaging_source,
    )
    demo_source = demo_assets or build_production_demo_asset_plan(
        root,
        release_candidate=candidate_source,
    )
    deployment_source = deployment or build_production_deployment_plan(
        root,
        packaging=packaging_source,
    )
    readiness_source = production_readiness or build_production_readiness_review(
        final_optimization=final_source,
        packaging=packaging_source,
        release_candidate=candidate_source,
        demo_assets=demo_source,
        deployment=deployment_source,
    )
    creative_source = creative_readiness or build_production_creative_readiness_review(
        demo_assets=demo_source,
        production_readiness=readiness_source,
    )
    source_versions = (
        final_source.serialization_version,
        packaging_source.serialization_version,
        candidate_source.serialization_version,
        demo_source.serialization_version,
        deployment_source.serialization_version,
        readiness_source.serialization_version,
        creative_source.serialization_version,
    )
    missing_docs = _missing_doc_refs(root)
    guarded_assumptions = _guarded_assumptions(
        final_source=final_source,
        candidate_source=candidate_source,
        deployment_source=deployment_source,
        readiness_source=readiness_source,
        creative_source=creative_source,
        missing_docs=missing_docs,
    )
    records = _records(
        source_versions=source_versions,
        final_source=final_source,
        packaging_source=packaging_source,
        candidate_source=candidate_source,
        demo_source=demo_source,
        deployment_source=deployment_source,
        readiness_source=readiness_source,
        creative_source=creative_source,
        guarded_assumptions=guarded_assumptions,
    )
    return ProductionArchitectureFreeze(
        source_surface_ids=_SOURCE_SURFACE_IDS,
        source_serialization_versions=source_versions,
        architecture_doc_refs=_ARCHITECTURE_DOC_REFS,
        missing_architecture_doc_refs=missing_docs,
        records=records,
        record_ids=tuple(record.record_id for record in records),
        freeze_domains=tuple(record.freeze_domain for record in records),
        frozen_record_ids=tuple(record.record_id for record in records),
        record_count=len(records),
        guarded_assumption_count=sum(
            len(record.guarded_assumptions) for record in records
        ),
        prohibited_change_count=sum(
            len(record.prohibited_changes) for record in records
        ),
        architecture_freeze_status="frozen",
    )


def production_architecture_freeze_record_by_domain(
    freeze_domain: ProductionArchitectureFreezeDomain | str,
    freeze: ProductionArchitectureFreeze | None = None,
) -> ProductionArchitectureFreezeRecord | None:
    """Return one architecture freeze record by domain."""

    normalized = str(freeze_domain).strip()
    source_freeze = freeze or build_production_architecture_freeze()
    for record in source_freeze.records:
        if record.freeze_domain == normalized:
            return record
    return None


def production_architecture_freeze_records_for_status(
    status: ProductionArchitectureFreezeStatus,
    freeze: ProductionArchitectureFreeze | None = None,
) -> tuple[ProductionArchitectureFreezeRecord, ...]:
    """Return architecture freeze records by status."""

    source_freeze = freeze or build_production_architecture_freeze()
    return tuple(
        record for record in source_freeze.records if record.freeze_status == status
    )


def _records(
    *,
    source_versions: tuple[str, ...],
    final_source: ProductionReleaseFinalOptimizationPlan,
    packaging_source: ProductionPackagingPlan,
    candidate_source: ProductionReleaseCandidatePlan,
    demo_source: ProductionDemoAssetPlan,
    deployment_source: ProductionDeploymentPlan,
    readiness_source: ProductionReadinessReview,
    creative_source: ProductionCreativeReadinessReview,
    guarded_assumptions: tuple[str, ...],
) -> tuple[ProductionArchitectureFreezeRecord, ...]:
    return (
        _record(
            domain="runtime_topology_freeze",
            source_ids=(final_source.role, readiness_source.role),
            source_versions=(
                final_source.serialization_version,
                readiness_source.serialization_version,
            ),
            assertions=(
                "compact_langgraph_runtime_preserved",
                "v5_6_adds_readiness_metadata_only",
                "no_new_runtime_nodes_required",
            ),
            guarded=(),
            prohibited=(
                "workflow_graph_mutation",
                "workflow_execution",
                "workflow_control",
                "graph_compilation",
            ),
            evidence=(
                f"demo_workflow_steps:{len(final_source.demo_workflow_steps)}",
                f"readiness_records:{readiness_source.record_count}",
                f"source_surfaces:{len(_SOURCE_SURFACE_IDS)}",
            ),
        ),
        _record(
            domain="v5_6_metadata_surface_freeze",
            source_ids=_SOURCE_SURFACE_IDS,
            source_versions=source_versions,
            assertions=(
                "final_optimization_surface_frozen",
                "packaging_surface_frozen",
                "release_candidate_surface_frozen",
                "demo_asset_surface_frozen",
                "deployment_surface_frozen",
                "readiness_review_surfaces_frozen",
            ),
            guarded=guarded_assumptions,
            prohibited=(
                "new_core_architecture_surface",
                "new_runtime_node",
                "new_storage_backend",
                "new_provider_runtime",
            ),
            evidence=(
                f"metadata_surfaces:{len(_SOURCE_SURFACE_IDS)}",
                f"packaging_records:{packaging_source.record_count}",
                f"creative_readiness_records:{creative_source.record_count}",
            ),
        ),
        _record(
            domain="provider_model_routing_freeze",
            source_ids=(
                final_source.role,
                candidate_source.role,
                readiness_source.role,
            ),
            source_versions=(
                final_source.serialization_version,
                candidate_source.serialization_version,
                readiness_source.serialization_version,
            ),
            assertions=(
                "provider_model_routing_not_mutated",
                "manual_assisted_auto_boundaries_preserved",
                "missing_configuration_is_guarded_metadata",
            ),
            guarded=_unique(
                (
                    *final_source.unavailable_reason_codes,
                    *final_source.required_hitl_gates,
                )
            ),
            prohibited=(
                "provider_or_model_routing_mutation",
                "provider_execution",
                "automatic_provider_switching",
                "automatic_model_download",
                "provider_provisioning",
            ),
            evidence=(
                f"provider_ids:{','.join(final_source.provider_ids)}",
                f"execution_mode:{final_source.selected_execution_mode_id}",
                f"candidate_status:{candidate_source.release_candidate_status}",
            ),
        ),
        _record(
            domain="deployment_release_operations_freeze",
            source_ids=(
                packaging_source.role,
                deployment_source.role,
                candidate_source.role,
            ),
            source_versions=(
                packaging_source.serialization_version,
                deployment_source.serialization_version,
                candidate_source.serialization_version,
            ),
            assertions=(
                "package_metadata_inspected_without_build",
                "deployment_assumptions_documented_without_deploy",
                "release_operations_remain_human_controlled",
            ),
            guarded=tuple(deployment_source.guarded_record_ids),
            prohibited=(
                "package_build_execution",
                "dependency_installation",
                "deployment_execution",
                "container_image_build",
                "release_artifact_creation",
                "merge_push_tag_operation",
            ),
            evidence=(
                f"package_records:{packaging_source.record_count}",
                f"deployment_status:{deployment_source.deployment_status}",
                f"release_candidate:{candidate_source.release_candidate_id}",
            ),
        ),
        _record(
            domain="generated_output_boundary_freeze",
            source_ids=(demo_source.role, creative_source.role, readiness_source.role),
            source_versions=(
                demo_source.serialization_version,
                creative_source.serialization_version,
                readiness_source.serialization_version,
            ),
            assertions=(
                "demo_assets_are_inventory_only",
                "creative_readiness_does_not_evaluate_output",
                "generated_output_mutation_boundary_preserved",
            ),
            guarded=tuple(creative_source.guarded_record_ids),
            prohibited=(
                "asset_generation",
                "retrieval_execution",
                "preview_rendering_execution",
                "artifact_mutation",
                "generated_output_modification",
                "prompt_mutation",
            ),
            evidence=(
                f"demo_asset_status:{demo_source.demo_asset_status}",
                f"creative_readiness_status:{creative_source.creative_readiness_status}",
                f"preview_media_paths:{len(demo_source.preview_media_paths)}",
            ),
        ),
        _record(
            domain="runtime_evolution_gate_freeze",
            source_ids=(
                candidate_source.role,
                readiness_source.role,
                creative_source.role,
            ),
            source_versions=(
                candidate_source.serialization_version,
                readiness_source.serialization_version,
                creative_source.serialization_version,
            ),
            assertions=(
                "runtime_evolution_not_applied",
                "runtime_evolution_requires_hitl_gate",
                "release_audit_precedes_runtime_evolution_review",
            ),
            guarded=guarded_assumptions,
            prohibited=(
                "runtime_evolution_application",
                "automatic_runtime_policy_change",
                "automatic_architecture_change",
                "hitl_request_emission",
            ),
            evidence=(
                f"required_pre_release_checks:{len(candidate_source.required_pre_release_checks)}",
                f"production_readiness_status:{readiness_source.production_readiness_status}",
                f"creative_readiness_status:{creative_source.creative_readiness_status}",
            ),
        ),
    )


def _record(
    *,
    domain: ProductionArchitectureFreezeDomain,
    source_ids: tuple[str, ...],
    source_versions: tuple[str, ...],
    assertions: tuple[str, ...],
    guarded: tuple[str, ...],
    prohibited: tuple[str, ...],
    evidence: tuple[str, ...],
) -> ProductionArchitectureFreezeRecord:
    return ProductionArchitectureFreezeRecord(
        record_id=f"production_architecture_freeze::{domain}",
        freeze_domain=domain,
        freeze_status="frozen",
        source_surface_ids=source_ids,
        source_serialization_versions=source_versions,
        architecture_doc_refs=_ARCHITECTURE_DOC_REFS,
        freeze_assertions=_unique(assertions),
        guarded_assumptions=_unique(guarded),
        prohibited_changes=_unique(prohibited),
        evidence=evidence,
    )


def _guarded_assumptions(
    *,
    final_source: ProductionReleaseFinalOptimizationPlan,
    candidate_source: ProductionReleaseCandidatePlan,
    deployment_source: ProductionDeploymentPlan,
    readiness_source: ProductionReadinessReview,
    creative_source: ProductionCreativeReadinessReview,
    missing_docs: tuple[str, ...],
) -> tuple[str, ...]:
    return _unique(
        (
            *final_source.unavailable_reason_codes,
            *final_source.required_hitl_gates,
            *candidate_source.guarded_record_ids,
            *deployment_source.guarded_record_ids,
            *readiness_source.guarded_record_ids,
            *creative_source.guarded_record_ids,
            *(f"missing_architecture_doc:{doc}" for doc in missing_docs),
        )
    )


def _missing_doc_refs(root: Path) -> tuple[str, ...]:
    return tuple(ref for ref in _ARCHITECTURE_DOC_REFS if not (root / ref).exists())


def _unique(values: tuple[str, ...]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if normalized and normalized not in result:
            result.append(normalized)
    return tuple(result)
