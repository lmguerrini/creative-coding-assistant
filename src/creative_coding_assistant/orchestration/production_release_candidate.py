"""V5.6 production release candidate readiness metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.adaptive_execution_policy_engine import (
    AdaptiveExecutionAvailabilityContext,
)
from creative_coding_assistant.orchestration.production_release_final_optimization import (
    ProductionReleaseFinalOptimizationPlan,
    build_production_release_final_optimization,
)
from creative_coding_assistant.orchestration.production_release_packaging import (
    ProductionPackagingPlan,
    build_production_packaging_plan,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
)

ReleaseCandidateSurfaceId = Literal[
    "baseline_validation",
    "final_optimization_readiness",
    "packaging_readiness",
    "production_safety_boundaries",
    "release_operation_controls",
]
ReleaseCandidateStatus = Literal["ready", "guarded"]

PRODUCTION_RELEASE_CANDIDATE_RECORD_SERIALIZATION_VERSION = (
    "production_release_candidate_record.v1"
)
PRODUCTION_RELEASE_CANDIDATE_PLAN_SERIALIZATION_VERSION = (
    "production_release_candidate_plan.v1"
)
PRODUCTION_RELEASE_CANDIDATE_AUTHORITY_BOUNDARY = (
    "V5.6 production release candidate metadata composes validation, final "
    "optimization, packaging, safety, and release-control posture for "
    "inspection only; it does not create release artifacts, build packages, "
    "deploy services, tag releases, merge branches, push refs, execute "
    "providers, change provider or model routing, install runtimes, emit HITL "
    "requests, mutate generated output, or apply Runtime Evolution."
)

_REQUIRED_SURFACES: tuple[ReleaseCandidateSurfaceId, ...] = (
    "baseline_validation",
    "final_optimization_readiness",
    "packaging_readiness",
    "production_safety_boundaries",
    "release_operation_controls",
)
_COMPLETED_CAPABILITY_VERSIONS = ("V5.1.0", "V5.2.0", "V5.3.0", "V5.4.0", "V5.5.0")
_REQUIRED_PRE_RELEASE_CHECKS = (
    "full_validation",
    "final_optimization",
    "packaging",
    "codex_engineering_audit",
    "runtime_failure_path_audit",
    "local_app_smoke_test",
    "merge_push_tag_gate",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "release_artifact_creation",
    "package_build_execution",
    "deployment_execution",
    "merge_operation",
    "push_operation",
    "tag_operation",
    "provider_or_model_routing_mutation",
    "provider_execution",
    "runtime_installation",
    "automatic_model_download",
    "hitl_request_emission",
    "workflow_execution",
    "generated_output_modification",
    "runtime_evolution_application",
)


class ProductionReleaseCandidateRecord(BaseModel):
    """One release-candidate readiness record."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    record_id: str = Field(min_length=1, max_length=180)
    surface_id: ReleaseCandidateSurfaceId
    status: ReleaseCandidateStatus
    source_surface_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_serialization_versions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=10)
    guarded_reason_codes: tuple[str, ...] = Field(default_factory=tuple, max_length=32)
    required_followups: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    release_blocker: bool
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=18,
    )
    release_candidate_record_implemented: Literal[True] = True
    release_artifact_creation_implemented: Literal[False] = False
    package_build_executed: Literal[False] = False
    deployment_execution_implemented: Literal[False] = False
    merge_operation_implemented: Literal[False] = False
    push_operation_implemented: Literal[False] = False
    tag_operation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    runtime_installation_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["production_release_candidate_record.v1"] = (
        PRODUCTION_RELEASE_CANDIDATE_RECORD_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _record_matches_surface(self) -> Self:
        if self.record_id != f"production_release_candidate::{self.surface_id}":
            raise ValueError("record_id must match surface_id")
        if len(self.source_surface_ids) != len(self.source_serialization_versions):
            raise ValueError("source ids and serialization versions must align")
        if self.status != ("guarded" if self.guarded_reason_codes else "ready"):
            raise ValueError("status must match guarded reasons")
        if self.release_blocker and self.status != "guarded":
            raise ValueError("release blockers must be guarded")
        return self


class ProductionReleaseCandidatePlan(BaseModel):
    """V5.6 release candidate readiness posture."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["production_release_candidate"] = "production_release_candidate"
    serialization_version: Literal["production_release_candidate_plan.v1"] = (
        PRODUCTION_RELEASE_CANDIDATE_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=PRODUCTION_RELEASE_CANDIDATE_AUTHORITY_BOUNDARY,
        max_length=2000,
    )
    version: Literal["V5.6.0"] = "V5.6.0"
    release_candidate_id: Literal["v5.6.0-rc.1"] = "v5.6.0-rc.1"
    target_branch: Literal["feature/production-release"] = "feature/production-release"
    target_tag: Literal["v5.6.0"] = "v5.6.0"
    route_name: RouteName
    task_type: TaskRoutingType
    requested_execution_mode_id: ExecutionModeId
    selected_execution_mode_id: ExecutionModeId
    source_final_optimization_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_packaging_serialization_version: str = Field(min_length=1, max_length=120)
    completed_capability_versions: tuple[str, ...] = Field(min_length=5, max_length=5)
    required_pre_release_checks: tuple[str, ...] = Field(min_length=7, max_length=7)
    records: tuple[ProductionReleaseCandidateRecord, ...] = Field(
        min_length=5,
        max_length=5,
    )
    record_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    surface_ids: tuple[ReleaseCandidateSurfaceId, ...] = Field(
        min_length=5,
        max_length=5,
    )
    ready_record_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    guarded_record_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    release_blocker_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    record_count: int = Field(ge=5, le=5)
    release_candidate_status: ReleaseCandidateStatus
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=18,
    )
    release_candidate_metadata_implemented: Literal[True] = True
    validation_gate_recorded: Literal[True] = True
    final_optimization_linked: Literal[True] = True
    packaging_linked: Literal[True] = True
    safety_boundaries_linked: Literal[True] = True
    release_controls_linked: Literal[True] = True
    release_artifact_creation_implemented: Literal[False] = False
    package_build_executed: Literal[False] = False
    deployment_execution_implemented: Literal[False] = False
    merge_operation_implemented: Literal[False] = False
    push_operation_implemented: Literal[False] = False
    tag_operation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    runtime_installation_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_records(self) -> Self:
        if self.record_ids != tuple(record.record_id for record in self.records):
            raise ValueError("record_ids must match records")
        if self.surface_ids != tuple(record.surface_id for record in self.records):
            raise ValueError("surface_ids must match records")
        if self.surface_ids != _REQUIRED_SURFACES:
            raise ValueError("surface_ids must cover required RC surfaces")
        if self.ready_record_ids != _record_ids_for_status(self.records, "ready"):
            raise ValueError("ready_record_ids must match records")
        if self.guarded_record_ids != _record_ids_for_status(self.records, "guarded"):
            raise ValueError("guarded_record_ids must match records")
        if self.release_blocker_ids != tuple(
            record.record_id for record in self.records if record.release_blocker
        ):
            raise ValueError("release_blocker_ids must match records")
        if self.record_count != len(self.records):
            raise ValueError("record_count must match records")
        if self.release_candidate_status != _plan_status(self.records):
            raise ValueError("release_candidate_status must match records")
        if self.completed_capability_versions != _COMPLETED_CAPABILITY_VERSIONS:
            raise ValueError("completed_capability_versions must cover V5.1-V5.5")
        if self.required_pre_release_checks != _REQUIRED_PRE_RELEASE_CHECKS:
            raise ValueError("required_pre_release_checks must match V5.6 gates")
        return self


def build_production_release_candidate(
    *,
    task_type: TaskRoutingType | str = "creative_coding",
    route: RouteName | str = RouteName.GENERATE,
    execution_mode_id: ExecutionModeId | str = "assisted_mode",
    availability_context: AdaptiveExecutionAvailabilityContext | None = None,
    final_optimization: ProductionReleaseFinalOptimizationPlan | None = None,
    packaging: ProductionPackagingPlan | None = None,
) -> ProductionReleaseCandidatePlan:
    """Build release candidate readiness metadata without release operations."""

    final_source = final_optimization or build_production_release_final_optimization(
        task_type=task_type,
        route=route,
        execution_mode_id=execution_mode_id,
        availability_context=availability_context,
    )
    packaging_source = packaging or build_production_packaging_plan()
    guarded_reasons = _unique(
        (*final_source.unavailable_reason_codes, *final_source.required_hitl_gates)
    )
    records = _records(final_source=final_source, packaging_source=packaging_source)
    return ProductionReleaseCandidatePlan(
        route_name=final_source.route_name,
        task_type=final_source.task_type,
        requested_execution_mode_id=final_source.requested_execution_mode_id,
        selected_execution_mode_id=final_source.selected_execution_mode_id,
        source_final_optimization_serialization_version=final_source.serialization_version,
        source_packaging_serialization_version=packaging_source.serialization_version,
        completed_capability_versions=_COMPLETED_CAPABILITY_VERSIONS,
        required_pre_release_checks=_REQUIRED_PRE_RELEASE_CHECKS,
        records=records,
        record_ids=tuple(record.record_id for record in records),
        surface_ids=tuple(record.surface_id for record in records),
        ready_record_ids=_record_ids_for_status(records, "ready"),
        guarded_record_ids=_record_ids_for_status(records, "guarded"),
        release_blocker_ids=tuple(
            record.record_id for record in records if record.release_blocker
        ),
        record_count=len(records),
        release_candidate_status="guarded" if guarded_reasons else _plan_status(records),
    )


def release_candidate_record_by_surface(
    surface_id: ReleaseCandidateSurfaceId | str,
    plan: ProductionReleaseCandidatePlan | None = None,
) -> ProductionReleaseCandidateRecord | None:
    """Return one release candidate record by surface id."""

    normalized = str(surface_id).strip()
    source_plan = plan or build_production_release_candidate()
    for record in source_plan.records:
        if record.surface_id == normalized:
            return record
    return None


def release_candidate_records_for_status(
    status: ReleaseCandidateStatus,
    plan: ProductionReleaseCandidatePlan | None = None,
) -> tuple[ProductionReleaseCandidateRecord, ...]:
    """Return release candidate records by status."""

    source_plan = plan or build_production_release_candidate()
    return tuple(record for record in source_plan.records if record.status == status)


def _records(
    *,
    final_source: ProductionReleaseFinalOptimizationPlan,
    packaging_source: ProductionPackagingPlan,
) -> tuple[ProductionReleaseCandidateRecord, ...]:
    guarded_reasons = _unique(
        (*final_source.unavailable_reason_codes, *final_source.required_hitl_gates)
    )
    return (
        _record(
            surface_id="baseline_validation",
            source_ids=("runtime_task_1_full_validation",),
            source_versions=("runtime_progress.v5",),
            evidence=(
                "baseline_validation:passed_before_release_candidate",
                "full_pytest:required_by_runtime",
                "frontend_validation:required_by_runtime",
            ),
            guarded_reason_codes=(),
            required_followups=("repeat_final_validation_before_acceptance",),
            release_blocker=False,
        ),
        _record(
            surface_id="final_optimization_readiness",
            source_ids=(final_source.role,),
            source_versions=(final_source.serialization_version,),
            evidence=(
                f"optimization_status:{final_source.production_optimization_status}",
                f"optimization_records:{final_source.record_count}",
                f"guarded_records:{final_source.guarded_record_count}",
            ),
            guarded_reason_codes=guarded_reasons,
            required_followups=(
                "keep_configuration_and_hitl_guards_visible",
                "resolve guarded items before production operation",
            )
            if guarded_reasons
            else (),
            release_blocker=False,
        ),
        _record(
            surface_id="packaging_readiness",
            source_ids=(packaging_source.role,),
            source_versions=(packaging_source.serialization_version,),
            evidence=(
                f"packaging_status:{packaging_source.packaging_status}",
                f"packaging_records:{packaging_source.record_count}",
                f"packaging_guarded:{len(packaging_source.guarded_record_ids)}",
            ),
            guarded_reason_codes=tuple(packaging_source.guarded_record_ids),
            required_followups=(
                "resolve_missing_packaging_metadata",
            )
            if packaging_source.guarded_record_ids
            else (),
            release_blocker=bool(packaging_source.guarded_record_ids),
        ),
        _record(
            surface_id="production_safety_boundaries",
            source_ids=(final_source.role, packaging_source.role),
            source_versions=(
                final_source.serialization_version,
                packaging_source.serialization_version,
            ),
            evidence=(
                f"execution_mode:{final_source.selected_execution_mode_id}",
                f"blocked_runtime_behaviors:{len(final_source.blocked_runtime_behaviors)}",
                "manual_human_gates:preserved",
            ),
            guarded_reason_codes=guarded_reasons,
            required_followups=(
                "maintain_manual_assisted_auto_mode_boundaries",
                "keep unsafe automatic execution disabled",
            ),
            release_blocker=False,
        ),
        _record(
            surface_id="release_operation_controls",
            source_ids=("version_runtime_rules",),
            source_versions=("v5_runtime.v2",),
            evidence=(
                "merge_push_tag:human_controlled",
                "runtime_evolution:HITL_only",
                "release_candidate:metadata_only",
            ),
            guarded_reason_codes=(),
            required_followups=(
                "stop_before_merge_push_tag",
                "do_not_create_release_tag_from_codex",
            ),
            release_blocker=False,
        ),
    )


def _record(
    *,
    surface_id: ReleaseCandidateSurfaceId,
    source_ids: tuple[str, ...],
    source_versions: tuple[str, ...],
    evidence: tuple[str, ...],
    guarded_reason_codes: tuple[str, ...],
    required_followups: tuple[str, ...],
    release_blocker: bool,
) -> ProductionReleaseCandidateRecord:
    return ProductionReleaseCandidateRecord(
        record_id=f"production_release_candidate::{surface_id}",
        surface_id=surface_id,
        status="guarded" if guarded_reason_codes else "ready",
        source_surface_ids=source_ids,
        source_serialization_versions=source_versions,
        evidence=evidence,
        guarded_reason_codes=guarded_reason_codes,
        required_followups=required_followups,
        release_blocker=release_blocker,
    )


def _record_ids_for_status(
    records: tuple[ProductionReleaseCandidateRecord, ...],
    status: ReleaseCandidateStatus,
) -> tuple[str, ...]:
    return tuple(record.record_id for record in records if record.status == status)


def _plan_status(
    records: tuple[ProductionReleaseCandidateRecord, ...],
) -> ReleaseCandidateStatus:
    if any(record.status == "guarded" for record in records):
        return "guarded"
    return "ready"


def _unique(values: tuple[str, ...]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if normalized and normalized not in result:
            result.append(normalized)
    return tuple(result)
