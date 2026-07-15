"""Production release audit metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.production_architecture_freeze import (
    ProductionArchitectureFreeze,
    build_production_architecture_freeze,
)
from creative_coding_assistant.orchestration.production_creative_readiness_review import (
    ProductionCreativeReadinessReview,
    build_production_creative_readiness_review,
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
from creative_coding_assistant.orchestration.production_release_packaging import (
    ProductionPackagingPlan,
    build_production_packaging_plan,
)

ProductionReleaseAuditArea = Literal[
    "validation_gate_audit",
    "release_candidate_audit",
    "production_readiness_audit",
    "creative_readiness_audit",
    "deployment_packaging_audit",
    "architecture_freeze_audit",
    "release_control_audit",
]
ProductionReleaseAuditStatus = Literal["pass", "guarded", "blocked"]
ProductionReleaseAuditOutcome = Literal["pass_with_guarded_assumptions", "blocked"]

PRODUCTION_RELEASE_AUDIT_RECORD_SERIALIZATION_VERSION = (
    "production_release_audit_record.v1"
)
PRODUCTION_RELEASE_AUDIT_SERIALIZATION_VERSION = "production_release_audit.v1"
PRODUCTION_RELEASE_AUDIT_AUTHORITY_BOUNDARY = (
    "Production release audit metadata reviews release candidate, "
    "packaging, deployment, production readiness, creative readiness, "
    "architecture freeze, validation, and release-control posture for "
    "inspection only; it does not create release artifacts, run package "
    "builds, install dependencies, deploy services, mutate provider/model "
    "routing, execute providers or workflows, generate assets, execute "
    "retrieval, mutate generated output, write storage, emit HITL requests, "
    "merge, push, tag, or apply Runtime Evolution."
)

_SOURCE_SURFACE_IDS = (
    "production_release_candidate",
    "production_release_packaging",
    "production_deployment",
    "production_readiness_review",
    "production_creative_readiness_review",
    "production_architecture_freeze",
)
_REQUIRED_AREAS: tuple[ProductionReleaseAuditArea, ...] = (
    "validation_gate_audit",
    "release_candidate_audit",
    "production_readiness_audit",
    "creative_readiness_audit",
    "deployment_packaging_audit",
    "architecture_freeze_audit",
    "release_control_audit",
)
_PENDING_RELEASE_GATES = (
    "codex_engineering_audit_pending",
    "runtime_failure_path_audit_pending",
    "final_validation_pending",
    "cumulative_local_app_smoke_test_pending",
    "capability_acceptance_test_pending",
    "runtime_evolution_review_pending",
    "merge_push_tag_gate_pending",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "release_artifact_creation",
    "package_build_execution",
    "dependency_installation",
    "deployment_execution",
    "provider_or_model_routing_mutation",
    "provider_execution",
    "workflow_execution",
    "workflow_control",
    "asset_generation",
    "retrieval_execution",
    "generated_output_modification",
    "persistent_storage_write",
    "hitl_request_emission",
    "merge_push_tag_operation",
    "runtime_evolution_application",
)


class ProductionReleaseAuditRecord(BaseModel):
    """One production release audit record."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    audit_id: str = Field(min_length=1, max_length=180)
    audit_area: ProductionReleaseAuditArea
    audit_status: ProductionReleaseAuditStatus
    source_surface_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_serialization_versions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=12)
    pass_findings: tuple[str, ...] = Field(min_length=1, max_length=24)
    guarded_findings: tuple[str, ...] = Field(default_factory=tuple, max_length=48)
    blocking_findings: tuple[str, ...] = Field(default_factory=tuple, max_length=24)
    required_followups: tuple[str, ...] = Field(default_factory=tuple, max_length=24)
    release_blocker: bool
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=20,
    )
    release_audit_record_implemented: Literal[True] = True
    release_artifact_creation_implemented: Literal[False] = False
    package_build_executed: Literal[False] = False
    dependency_installation_implemented: Literal[False] = False
    deployment_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    asset_generation_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    merge_push_tag_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["production_release_audit_record.v1"] = (
        PRODUCTION_RELEASE_AUDIT_RECORD_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _record_matches_contract(self) -> Self:
        if self.audit_id != f"production_release_audit::{self.audit_area}":
            raise ValueError("audit_id must match audit_area")
        if len(self.source_surface_ids) != len(self.source_serialization_versions):
            raise ValueError("source ids and serialization versions must align")
        if self.audit_status != _status_for_findings(
            self.guarded_findings,
            self.blocking_findings,
        ):
            raise ValueError("audit_status must match findings")
        if self.release_blocker != bool(self.blocking_findings):
            raise ValueError("release_blocker must match blocking findings")
        return self


class ProductionReleaseAudit(BaseModel):
    """Production release audit aggregate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["production_release_audit"] = "production_release_audit"
    serialization_version: Literal["production_release_audit.v1"] = (
        PRODUCTION_RELEASE_AUDIT_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=PRODUCTION_RELEASE_AUDIT_AUTHORITY_BOUNDARY,
        max_length=2200,
    )
    version: Literal["V5.6.0"] = "V5.6.0"
    target_branch: Literal["feature/production-release"] = "feature/production-release"
    target_tag: Literal["v5.6.0"] = "v5.6.0"
    source_surface_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_serialization_versions: tuple[str, ...] = Field(min_length=6, max_length=6)
    pending_release_gate_ids: tuple[str, ...] = Field(min_length=7, max_length=7)
    records: tuple[ProductionReleaseAuditRecord, ...] = Field(
        min_length=7,
        max_length=7,
    )
    audit_ids: tuple[str, ...] = Field(min_length=7, max_length=7)
    audit_areas: tuple[ProductionReleaseAuditArea, ...] = Field(
        min_length=7,
        max_length=7,
    )
    pass_audit_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=7)
    guarded_audit_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=7)
    blocked_audit_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=7)
    record_count: int = Field(ge=7, le=7)
    pass_finding_count: int = Field(ge=0, le=200)
    guarded_finding_count: int = Field(ge=0, le=200)
    blocking_finding_count: int = Field(ge=0, le=100)
    release_blocker_count: int = Field(ge=0, le=7)
    release_audit_status: ProductionReleaseAuditStatus
    release_audit_outcome: ProductionReleaseAuditOutcome
    release_audit_can_proceed_to_hardening: Literal[True] = True
    release_audit_implemented: Literal[True] = True
    validation_gate_audit_implemented: Literal[True] = True
    release_candidate_audit_implemented: Literal[True] = True
    production_readiness_audit_implemented: Literal[True] = True
    creative_readiness_audit_implemented: Literal[True] = True
    deployment_packaging_audit_implemented: Literal[True] = True
    architecture_freeze_audit_implemented: Literal[True] = True
    release_control_audit_implemented: Literal[True] = True
    release_artifact_creation_implemented: Literal[False] = False
    package_build_executed: Literal[False] = False
    dependency_installation_implemented: Literal[False] = False
    deployment_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    asset_generation_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    merge_push_tag_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=20,
    )
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _audit_matches_records(self) -> Self:
        if self.source_surface_ids != _SOURCE_SURFACE_IDS:
            raise ValueError("source_surface_ids must match release audit sources")
        if len(self.source_surface_ids) != len(self.source_serialization_versions):
            raise ValueError("source ids and serialization versions must align")
        if self.pending_release_gate_ids != _PENDING_RELEASE_GATES:
            raise ValueError("pending_release_gate_ids must match required gates")
        if self.audit_ids != tuple(record.audit_id for record in self.records):
            raise ValueError("audit_ids must match records")
        if self.audit_areas != tuple(record.audit_area for record in self.records):
            raise ValueError("audit_areas must match records")
        if self.audit_areas != _REQUIRED_AREAS:
            raise ValueError("audit_areas must cover required audit areas")
        if self.pass_audit_ids != _audit_ids_for_status(self.records, "pass"):
            raise ValueError("pass_audit_ids must match records")
        if self.guarded_audit_ids != _audit_ids_for_status(self.records, "guarded"):
            raise ValueError("guarded_audit_ids must match records")
        if self.blocked_audit_ids != _audit_ids_for_status(self.records, "blocked"):
            raise ValueError("blocked_audit_ids must match records")
        if self.record_count != len(self.records):
            raise ValueError("record_count must match records")
        if self.pass_finding_count != sum(
            len(record.pass_findings) for record in self.records
        ):
            raise ValueError("pass_finding_count must match records")
        if self.guarded_finding_count != sum(
            len(record.guarded_findings) for record in self.records
        ):
            raise ValueError("guarded_finding_count must match records")
        if self.blocking_finding_count != sum(
            len(record.blocking_findings) for record in self.records
        ):
            raise ValueError("blocking_finding_count must match records")
        if self.release_blocker_count != sum(
            1 for record in self.records if record.release_blocker
        ):
            raise ValueError("release_blocker_count must match records")
        if self.release_audit_status != _audit_status(self.records):
            raise ValueError("release_audit_status must match records")
        if self.release_audit_outcome != _audit_outcome(self.records):
            raise ValueError("release_audit_outcome must match records")
        return self


def build_production_release_audit(
    project_root: str | Path | None = None,
    *,
    release_candidate: ProductionReleaseCandidatePlan | None = None,
    packaging: ProductionPackagingPlan | None = None,
    deployment: ProductionDeploymentPlan | None = None,
    production_readiness: ProductionReadinessReview | None = None,
    creative_readiness: ProductionCreativeReadinessReview | None = None,
    architecture_freeze: ProductionArchitectureFreeze | None = None,
) -> ProductionReleaseAudit:
    """Build release audit metadata without release operations."""

    root = Path(project_root or ".").resolve()
    packaging_source = packaging or build_production_packaging_plan(root)
    candidate_source = release_candidate or build_production_release_candidate(
        packaging=packaging_source,
    )
    deployment_source = deployment or build_production_deployment_plan(
        root,
        packaging=packaging_source,
    )
    readiness_source = production_readiness or build_production_readiness_review(
        packaging=packaging_source,
        release_candidate=candidate_source,
        deployment=deployment_source,
    )
    creative_source = creative_readiness or build_production_creative_readiness_review(
        production_readiness=readiness_source,
    )
    architecture_source = architecture_freeze or build_production_architecture_freeze(
        root,
        packaging=packaging_source,
        release_candidate=candidate_source,
        deployment=deployment_source,
        production_readiness=readiness_source,
        creative_readiness=creative_source,
    )
    records = _records(
        candidate_source=candidate_source,
        packaging_source=packaging_source,
        deployment_source=deployment_source,
        readiness_source=readiness_source,
        creative_source=creative_source,
        architecture_source=architecture_source,
    )
    return ProductionReleaseAudit(
        source_surface_ids=_SOURCE_SURFACE_IDS,
        source_serialization_versions=(
            candidate_source.serialization_version,
            packaging_source.serialization_version,
            deployment_source.serialization_version,
            readiness_source.serialization_version,
            creative_source.serialization_version,
            architecture_source.serialization_version,
        ),
        pending_release_gate_ids=_PENDING_RELEASE_GATES,
        records=records,
        audit_ids=tuple(record.audit_id for record in records),
        audit_areas=tuple(record.audit_area for record in records),
        pass_audit_ids=_audit_ids_for_status(records, "pass"),
        guarded_audit_ids=_audit_ids_for_status(records, "guarded"),
        blocked_audit_ids=_audit_ids_for_status(records, "blocked"),
        record_count=len(records),
        pass_finding_count=sum(len(record.pass_findings) for record in records),
        guarded_finding_count=sum(len(record.guarded_findings) for record in records),
        blocking_finding_count=sum(len(record.blocking_findings) for record in records),
        release_blocker_count=sum(1 for record in records if record.release_blocker),
        release_audit_status=_audit_status(records),
        release_audit_outcome=_audit_outcome(records),
    )


def production_release_audit_record_by_area(
    audit_area: ProductionReleaseAuditArea | str,
    audit: ProductionReleaseAudit | None = None,
) -> ProductionReleaseAuditRecord | None:
    """Return one release audit record by area."""

    normalized = str(audit_area).strip()
    source_audit = audit or build_production_release_audit()
    for record in source_audit.records:
        if record.audit_area == normalized:
            return record
    return None


def production_release_audit_records_for_status(
    status: ProductionReleaseAuditStatus,
    audit: ProductionReleaseAudit | None = None,
) -> tuple[ProductionReleaseAuditRecord, ...]:
    """Return release audit records by status."""

    source_audit = audit or build_production_release_audit()
    return tuple(
        record for record in source_audit.records if record.audit_status == status
    )


def _records(
    *,
    candidate_source: ProductionReleaseCandidatePlan,
    packaging_source: ProductionPackagingPlan,
    deployment_source: ProductionDeploymentPlan,
    readiness_source: ProductionReadinessReview,
    creative_source: ProductionCreativeReadinessReview,
    architecture_source: ProductionArchitectureFreeze,
) -> tuple[ProductionReleaseAuditRecord, ...]:
    return (
        _record(
            area="validation_gate_audit",
            source_ids=(candidate_source.role,),
            source_versions=(candidate_source.serialization_version,),
            evidence=(
                f"required_pre_release_checks:{len(candidate_source.required_pre_release_checks)}",
                "task_1_full_validation:passed",
                "task_10_architecture_freeze:passed",
            ),
            pass_findings=(
                "baseline_validation_recorded",
                "architecture_freeze_completed",
            ),
            guarded_findings=_PENDING_RELEASE_GATES,
            blocking_findings=(),
            followups=_PENDING_RELEASE_GATES,
        ),
        _record(
            area="release_candidate_audit",
            source_ids=(candidate_source.role,),
            source_versions=(candidate_source.serialization_version,),
            evidence=(
                f"release_candidate_id:{candidate_source.release_candidate_id}",
                f"release_candidate_status:{candidate_source.release_candidate_status}",
                f"guarded_records:{len(candidate_source.guarded_record_ids)}",
            ),
            pass_findings=(
                "release_candidate_metadata_present",
                "release_controls_linked",
            ),
            guarded_findings=tuple(candidate_source.guarded_record_ids),
            blocking_findings=(),
            followups=("keep_release_candidate_guardrails_visible",),
        ),
        _record(
            area="production_readiness_audit",
            source_ids=(readiness_source.role,),
            source_versions=(readiness_source.serialization_version,),
            evidence=(
                f"production_readiness_status:{readiness_source.production_readiness_status}",
                f"readiness_records:{readiness_source.record_count}",
                f"blocking_findings:{readiness_source.blocking_finding_count}",
            ),
            pass_findings=(
                "production_readiness_review_present",
                "no_production_readiness_blockers",
            ),
            guarded_findings=tuple(readiness_source.guarded_record_ids),
            blocking_findings=tuple(readiness_source.blocked_record_ids),
            followups=("resolve_configuration_and_deployment_assumptions",),
        ),
        _record(
            area="creative_readiness_audit",
            source_ids=(creative_source.role,),
            source_versions=(creative_source.serialization_version,),
            evidence=(
                f"creative_readiness_status:{creative_source.creative_readiness_status}",
                f"creative_readiness_records:{creative_source.record_count}",
                f"blocking_findings:{creative_source.blocking_finding_count}",
            ),
            pass_findings=(
                "creative_readiness_review_present",
                "creative_demo_materials_available",
            ),
            guarded_findings=tuple(creative_source.guarded_record_ids),
            blocking_findings=tuple(creative_source.blocked_record_ids),
            followups=("keep_creative_analytics_guarded_as_metadata",),
        ),
        _record(
            area="deployment_packaging_audit",
            source_ids=(packaging_source.role, deployment_source.role),
            source_versions=(
                packaging_source.serialization_version,
                deployment_source.serialization_version,
            ),
            evidence=(
                f"packaging_status:{packaging_source.packaging_status}",
                f"deployment_status:{deployment_source.deployment_status}",
                f"external_manifests:{len(deployment_source.external_manifest_paths)}",
            ),
            pass_findings=(
                "packaging_metadata_present",
                "local_deployment_entrypoints_documented",
            ),
            guarded_findings=(
                *packaging_source.guarded_record_ids,
                *deployment_source.guarded_record_ids,
            ),
            blocking_findings=(),
            followups=("document_external_deployment_target_before_hosting",),
        ),
        _record(
            area="architecture_freeze_audit",
            source_ids=(architecture_source.role,),
            source_versions=(architecture_source.serialization_version,),
            evidence=(
                f"architecture_freeze_status:{architecture_source.architecture_freeze_status}",
                f"freeze_records:{architecture_source.record_count}",
                f"missing_docs:{len(architecture_source.missing_architecture_doc_refs)}",
            ),
            pass_findings=(
                "architecture_freeze_present",
                "no_architecture_expansion_required",
                "runtime_evolution_hitl_required",
            ),
            guarded_findings=_freeze_guarded_assumptions(architecture_source),
            blocking_findings=(),
            followups=("preserve_architecture_freeze_until_runtime_evolution_review",),
        ),
        _record(
            area="release_control_audit",
            source_ids=(candidate_source.role, architecture_source.role),
            source_versions=(
                candidate_source.serialization_version,
                architecture_source.serialization_version,
            ),
            evidence=(
                "merge_push_tag:human_controlled",
                "release_artifacts:not_created",
                "runtime_evolution:not_applied",
            ),
            pass_findings=(
                "merge_push_tag_gate_preserved",
                "release_artifact_creation_blocked",
                "runtime_evolution_review_deferred_to_gate",
            ),
            guarded_findings=_PENDING_RELEASE_GATES,
            blocking_findings=(),
            followups=(
                "complete final validation, smoke, acceptance, metrics, history, and human release gates",
            ),
        ),
    )


def _record(
    *,
    area: ProductionReleaseAuditArea,
    source_ids: tuple[str, ...],
    source_versions: tuple[str, ...],
    evidence: tuple[str, ...],
    pass_findings: tuple[str, ...],
    guarded_findings: tuple[str, ...],
    blocking_findings: tuple[str, ...],
    followups: tuple[str, ...],
) -> ProductionReleaseAuditRecord:
    return ProductionReleaseAuditRecord(
        audit_id=f"production_release_audit::{area}",
        audit_area=area,
        audit_status=_status_for_findings(guarded_findings, blocking_findings),
        source_surface_ids=source_ids,
        source_serialization_versions=source_versions,
        evidence=evidence,
        pass_findings=_unique(pass_findings),
        guarded_findings=_unique(guarded_findings),
        blocking_findings=_unique(blocking_findings),
        required_followups=_unique(followups),
        release_blocker=bool(blocking_findings),
    )


def _freeze_guarded_assumptions(
    architecture_source: ProductionArchitectureFreeze,
) -> tuple[str, ...]:
    return _unique(
        tuple(
            assumption
            for record in architecture_source.records
            for assumption in record.guarded_assumptions
        )
    )


def _status_for_findings(
    guarded_findings: tuple[str, ...],
    blocking_findings: tuple[str, ...],
) -> ProductionReleaseAuditStatus:
    if blocking_findings:
        return "blocked"
    if guarded_findings:
        return "guarded"
    return "pass"


def _audit_ids_for_status(
    records: tuple[ProductionReleaseAuditRecord, ...],
    status: ProductionReleaseAuditStatus,
) -> tuple[str, ...]:
    return tuple(record.audit_id for record in records if record.audit_status == status)


def _audit_status(
    records: tuple[ProductionReleaseAuditRecord, ...],
) -> ProductionReleaseAuditStatus:
    if any(record.audit_status == "blocked" for record in records):
        return "blocked"
    if any(record.audit_status == "guarded" for record in records):
        return "guarded"
    return "pass"


def _audit_outcome(
    records: tuple[ProductionReleaseAuditRecord, ...],
) -> ProductionReleaseAuditOutcome:
    if any(record.release_blocker for record in records):
        return "blocked"
    return "pass_with_guarded_assumptions"


def _unique(values: tuple[str, ...]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if normalized and normalized not in result:
            result.append(normalized)
    return tuple(result)
