"""V5.6 final production hardening metadata."""

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
from creative_coding_assistant.orchestration.production_release_audit import (
    ProductionReleaseAudit,
    build_production_release_audit,
)

ProductionHardeningArea = Literal[
    "configuration_hardening",
    "deployment_hardening",
    "release_gate_hardening",
    "creative_demo_hardening",
    "architecture_boundary_hardening",
    "failure_path_hardening",
]
ProductionHardeningStatus = Literal["ready", "guarded", "blocked"]

PRODUCTION_HARDENING_RECORD_SERIALIZATION_VERSION = (
    "production_hardening_record.v1"
)
PRODUCTION_FINAL_HARDENING_SERIALIZATION_VERSION = (
    "production_final_hardening.v1"
)
PRODUCTION_FINAL_HARDENING_AUTHORITY_BOUNDARY = (
    "V5.6 final production hardening metadata records release-hardening "
    "actions for guarded assumptions and remaining gates only; it does not "
    "mutate configuration, provision providers, install dependencies or "
    "runtimes, build packages, deploy services, create release artifacts, "
    "change provider/model routing, execute providers or workflows, generate "
    "assets, execute retrieval, mutate generated output, write storage, emit "
    "HITL requests, merge, push, tag, or apply Runtime Evolution."
)

_REQUIRED_AREAS: tuple[ProductionHardeningArea, ...] = (
    "configuration_hardening",
    "deployment_hardening",
    "release_gate_hardening",
    "creative_demo_hardening",
    "architecture_boundary_hardening",
    "failure_path_hardening",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "configuration_mutation",
    "provider_provisioning",
    "dependency_installation",
    "runtime_installation",
    "package_build_execution",
    "deployment_execution",
    "release_artifact_creation",
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


class ProductionHardeningRecord(BaseModel):
    """One final production hardening record."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    hardening_id: str = Field(min_length=1, max_length=180)
    hardening_area: ProductionHardeningArea
    hardening_status: ProductionHardeningStatus
    source_surface_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_serialization_versions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=12)
    hardening_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    guarded_findings: tuple[str, ...] = Field(default_factory=tuple, max_length=48)
    blocking_findings: tuple[str, ...] = Field(default_factory=tuple, max_length=24)
    release_blocker: bool
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    hardening_record_implemented: Literal[True] = True
    hardening_action_execution_implemented: Literal[False] = False
    configuration_mutation_implemented: Literal[False] = False
    provider_provisioning_implemented: Literal[False] = False
    dependency_installation_implemented: Literal[False] = False
    runtime_installation_implemented: Literal[False] = False
    package_build_executed: Literal[False] = False
    deployment_execution_implemented: Literal[False] = False
    release_artifact_creation_implemented: Literal[False] = False
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
    serialization_version: Literal["production_hardening_record.v1"] = (
        PRODUCTION_HARDENING_RECORD_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _record_matches_contract(self) -> Self:
        if self.hardening_id != f"production_hardening::{self.hardening_area}":
            raise ValueError("hardening_id must match hardening_area")
        if len(self.source_surface_ids) != len(self.source_serialization_versions):
            raise ValueError("source ids and serialization versions must align")
        if self.hardening_status != _status_for_findings(
            self.guarded_findings,
            self.blocking_findings,
        ):
            raise ValueError("hardening_status must match findings")
        if self.release_blocker != bool(self.blocking_findings):
            raise ValueError("release_blocker must match blocking findings")
        return self


class ProductionFinalHardening(BaseModel):
    """Final production hardening posture for V5.6."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["production_final_hardening"] = "production_final_hardening"
    serialization_version: Literal["production_final_hardening.v1"] = (
        PRODUCTION_FINAL_HARDENING_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=PRODUCTION_FINAL_HARDENING_AUTHORITY_BOUNDARY,
        max_length=2200,
    )
    source_release_audit_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_deployment_serialization_version: str = Field(min_length=1, max_length=120)
    source_production_readiness_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_creative_readiness_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_architecture_freeze_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    records: tuple[ProductionHardeningRecord, ...] = Field(
        min_length=6,
        max_length=6,
    )
    hardening_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    hardening_areas: tuple[ProductionHardeningArea, ...] = Field(
        min_length=6,
        max_length=6,
    )
    ready_hardening_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    guarded_hardening_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    blocked_hardening_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    record_count: int = Field(ge=6, le=6)
    hardening_action_count: int = Field(ge=0, le=100)
    guarded_finding_count: int = Field(ge=0, le=200)
    blocking_finding_count: int = Field(ge=0, le=100)
    release_blocker_count: int = Field(ge=0, le=6)
    final_hardening_status: ProductionHardeningStatus
    final_hardening_outcome: Literal["guarded_ready_for_consistency_pass", "blocked"]
    can_proceed_to_architecture_consistency_pass: Literal[True] = True
    final_hardening_implemented: Literal[True] = True
    configuration_hardening_implemented: Literal[True] = True
    deployment_hardening_implemented: Literal[True] = True
    release_gate_hardening_implemented: Literal[True] = True
    creative_demo_hardening_implemented: Literal[True] = True
    architecture_boundary_hardening_implemented: Literal[True] = True
    failure_path_hardening_implemented: Literal[True] = True
    hardening_action_execution_implemented: Literal[False] = False
    configuration_mutation_implemented: Literal[False] = False
    provider_provisioning_implemented: Literal[False] = False
    dependency_installation_implemented: Literal[False] = False
    runtime_installation_implemented: Literal[False] = False
    package_build_executed: Literal[False] = False
    deployment_execution_implemented: Literal[False] = False
    release_artifact_creation_implemented: Literal[False] = False
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
        max_length=24,
    )
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _hardening_matches_records(self) -> Self:
        if self.hardening_ids != tuple(record.hardening_id for record in self.records):
            raise ValueError("hardening_ids must match records")
        if self.hardening_areas != tuple(
            record.hardening_area for record in self.records
        ):
            raise ValueError("hardening_areas must match records")
        if self.hardening_areas != _REQUIRED_AREAS:
            raise ValueError("hardening_areas must cover required hardening areas")
        if self.ready_hardening_ids != _hardening_ids_for_status(self.records, "ready"):
            raise ValueError("ready_hardening_ids must match records")
        if self.guarded_hardening_ids != _hardening_ids_for_status(
            self.records,
            "guarded",
        ):
            raise ValueError("guarded_hardening_ids must match records")
        if self.blocked_hardening_ids != _hardening_ids_for_status(
            self.records,
            "blocked",
        ):
            raise ValueError("blocked_hardening_ids must match records")
        if self.record_count != len(self.records):
            raise ValueError("record_count must match records")
        if self.hardening_action_count != sum(
            len(record.hardening_actions) for record in self.records
        ):
            raise ValueError("hardening_action_count must match records")
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
        if self.final_hardening_status != _hardening_status(self.records):
            raise ValueError("final_hardening_status must match records")
        if self.final_hardening_outcome != _hardening_outcome(self.records):
            raise ValueError("final_hardening_outcome must match records")
        return self


def build_production_final_hardening(
    project_root: str | Path | None = None,
    *,
    release_audit: ProductionReleaseAudit | None = None,
    deployment: ProductionDeploymentPlan | None = None,
    production_readiness: ProductionReadinessReview | None = None,
    creative_readiness: ProductionCreativeReadinessReview | None = None,
    architecture_freeze: ProductionArchitectureFreeze | None = None,
) -> ProductionFinalHardening:
    """Build final production hardening metadata without applying actions."""

    root = Path(project_root or ".").resolve()
    deployment_source = deployment or build_production_deployment_plan(root)
    readiness_source = production_readiness or build_production_readiness_review(
        deployment=deployment_source,
    )
    creative_source = creative_readiness or build_production_creative_readiness_review(
        production_readiness=readiness_source,
    )
    architecture_source = architecture_freeze or build_production_architecture_freeze(
        root,
        deployment=deployment_source,
        production_readiness=readiness_source,
        creative_readiness=creative_source,
    )
    audit_source = release_audit or build_production_release_audit(
        root,
        deployment=deployment_source,
        production_readiness=readiness_source,
        creative_readiness=creative_source,
        architecture_freeze=architecture_source,
    )
    records = _records(
        audit_source=audit_source,
        deployment_source=deployment_source,
        readiness_source=readiness_source,
        creative_source=creative_source,
        architecture_source=architecture_source,
    )
    return ProductionFinalHardening(
        source_release_audit_serialization_version=audit_source.serialization_version,
        source_deployment_serialization_version=deployment_source.serialization_version,
        source_production_readiness_serialization_version=(
            readiness_source.serialization_version
        ),
        source_creative_readiness_serialization_version=(
            creative_source.serialization_version
        ),
        source_architecture_freeze_serialization_version=(
            architecture_source.serialization_version
        ),
        records=records,
        hardening_ids=tuple(record.hardening_id for record in records),
        hardening_areas=tuple(record.hardening_area for record in records),
        ready_hardening_ids=_hardening_ids_for_status(records, "ready"),
        guarded_hardening_ids=_hardening_ids_for_status(records, "guarded"),
        blocked_hardening_ids=_hardening_ids_for_status(records, "blocked"),
        record_count=len(records),
        hardening_action_count=sum(len(record.hardening_actions) for record in records),
        guarded_finding_count=sum(len(record.guarded_findings) for record in records),
        blocking_finding_count=sum(len(record.blocking_findings) for record in records),
        release_blocker_count=sum(1 for record in records if record.release_blocker),
        final_hardening_status=_hardening_status(records),
        final_hardening_outcome=_hardening_outcome(records),
    )


def production_hardening_record_by_area(
    hardening_area: ProductionHardeningArea | str,
    hardening: ProductionFinalHardening | None = None,
) -> ProductionHardeningRecord | None:
    """Return one hardening record by area."""

    normalized = str(hardening_area).strip()
    source_hardening = hardening or build_production_final_hardening()
    for record in source_hardening.records:
        if record.hardening_area == normalized:
            return record
    return None


def production_hardening_records_for_status(
    status: ProductionHardeningStatus,
    hardening: ProductionFinalHardening | None = None,
) -> tuple[ProductionHardeningRecord, ...]:
    """Return hardening records by status."""

    source_hardening = hardening or build_production_final_hardening()
    return tuple(
        record for record in source_hardening.records if record.hardening_status == status
    )


def _records(
    *,
    audit_source: ProductionReleaseAudit,
    deployment_source: ProductionDeploymentPlan,
    readiness_source: ProductionReadinessReview,
    creative_source: ProductionCreativeReadinessReview,
    architecture_source: ProductionArchitectureFreeze,
) -> tuple[ProductionHardeningRecord, ...]:
    return (
        _record(
            area="configuration_hardening",
            source_ids=(readiness_source.role, audit_source.role),
            source_versions=(
                readiness_source.serialization_version,
                audit_source.serialization_version,
            ),
            evidence=(
                f"production_readiness_status:{readiness_source.production_readiness_status}",
                f"guarded_readiness_records:{len(readiness_source.guarded_record_ids)}",
                "configuration_mutation:not_applied",
            ),
            actions=(
                "Keep missing provider credentials surfaced as guarded metadata.",
                "Require operator configuration before live provider execution.",
            ),
            guarded=_unique(
                (
                    *readiness_source.guarded_record_ids,
                    *audit_source.guarded_audit_ids,
                )
            ),
            blocking=tuple(readiness_source.blocked_record_ids),
        ),
        _record(
            area="deployment_hardening",
            source_ids=(deployment_source.role, audit_source.role),
            source_versions=(
                deployment_source.serialization_version,
                audit_source.serialization_version,
            ),
            evidence=(
                f"deployment_status:{deployment_source.deployment_status}",
                f"deployment_guarded_records:{len(deployment_source.guarded_record_ids)}",
                "external_deployment_manifest:guarded_assumption",
            ),
            actions=(
                "Keep local demo deployment separate from external deployment automation.",
                "Document an external deployment target before production hosting.",
            ),
            guarded=tuple(deployment_source.guarded_record_ids),
            blocking=(),
        ),
        _record(
            area="release_gate_hardening",
            source_ids=(audit_source.role,),
            source_versions=(audit_source.serialization_version,),
            evidence=(
                f"release_audit_status:{audit_source.release_audit_status}",
                f"pending_release_gates:{len(audit_source.pending_release_gate_ids)}",
                f"release_blockers:{audit_source.release_blocker_count}",
            ),
            actions=(
                "Preserve pending engineering audit, failure audit, validation, smoke, acceptance, metrics, history, Runtime Evolution, and release gates.",
                "Do not merge, push, tag, or emit HITL during hardening metadata.",
            ),
            guarded=audit_source.pending_release_gate_ids,
            blocking=tuple(audit_source.blocked_audit_ids),
        ),
        _record(
            area="creative_demo_hardening",
            source_ids=(creative_source.role,),
            source_versions=(creative_source.serialization_version,),
            evidence=(
                f"creative_readiness_status:{creative_source.creative_readiness_status}",
                f"creative_guarded_records:{len(creative_source.guarded_record_ids)}",
                "generated_output_evaluation:not_applied",
            ),
            actions=(
                "Use prepared demo assets and explanation cues without regenerating assets.",
                "Keep creative analytics passive and detached from output evaluation.",
            ),
            guarded=tuple(creative_source.guarded_record_ids),
            blocking=tuple(creative_source.blocked_record_ids),
        ),
        _record(
            area="architecture_boundary_hardening",
            source_ids=(architecture_source.role,),
            source_versions=(architecture_source.serialization_version,),
            evidence=(
                f"architecture_freeze_status:{architecture_source.architecture_freeze_status}",
                f"freeze_records:{architecture_source.record_count}",
                f"missing_docs:{len(architecture_source.missing_architecture_doc_refs)}",
            ),
            actions=(
                "Keep the V5.6 architecture freeze intact through consistency pass.",
                "Defer Runtime Evolution decisions to the explicit review gate.",
            ),
            guarded=_architecture_guarded_assumptions(architecture_source),
            blocking=(),
        ),
        _record(
            area="failure_path_hardening",
            source_ids=(readiness_source.role, audit_source.role),
            source_versions=(
                readiness_source.serialization_version,
                audit_source.serialization_version,
            ),
            evidence=(
                f"failure_review_records:{readiness_source.record_count}",
                f"audit_guarded_findings:{audit_source.guarded_finding_count}",
                "failure_paths:deterministic_metadata",
            ),
            actions=(
                "Carry provider unavailable, API-key missing, deployment assumption, and pending-gate paths into failure audit.",
                "Keep failure hardening deterministic and metadata-only.",
            ),
            guarded=_unique(
                (
                    *readiness_source.guarded_record_ids,
                    *audit_source.guarded_audit_ids,
                )
            ),
            blocking=tuple(readiness_source.blocked_record_ids),
        ),
    )


def _record(
    *,
    area: ProductionHardeningArea,
    source_ids: tuple[str, ...],
    source_versions: tuple[str, ...],
    evidence: tuple[str, ...],
    actions: tuple[str, ...],
    guarded: tuple[str, ...],
    blocking: tuple[str, ...],
) -> ProductionHardeningRecord:
    return ProductionHardeningRecord(
        hardening_id=f"production_hardening::{area}",
        hardening_area=area,
        hardening_status=_status_for_findings(guarded, blocking),
        source_surface_ids=source_ids,
        source_serialization_versions=source_versions,
        evidence=evidence,
        hardening_actions=actions,
        guarded_findings=_unique(guarded),
        blocking_findings=_unique(blocking),
        release_blocker=bool(blocking),
    )


def _architecture_guarded_assumptions(
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
) -> ProductionHardeningStatus:
    if blocking_findings:
        return "blocked"
    if guarded_findings:
        return "guarded"
    return "ready"


def _hardening_ids_for_status(
    records: tuple[ProductionHardeningRecord, ...],
    status: ProductionHardeningStatus,
) -> tuple[str, ...]:
    return tuple(record.hardening_id for record in records if record.hardening_status == status)


def _hardening_status(
    records: tuple[ProductionHardeningRecord, ...],
) -> ProductionHardeningStatus:
    if any(record.hardening_status == "blocked" for record in records):
        return "blocked"
    if any(record.hardening_status == "guarded" for record in records):
        return "guarded"
    return "ready"


def _hardening_outcome(
    records: tuple[ProductionHardeningRecord, ...],
) -> Literal["guarded_ready_for_consistency_pass", "blocked"]:
    if any(record.release_blocker for record in records):
        return "blocked"
    return "guarded_ready_for_consistency_pass"


def _unique(values: tuple[str, ...]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if normalized and normalized not in result:
            result.append(normalized)
    return tuple(result)
