"""Production readiness metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.production_demo_assets import (
    ProductionDemoAssetPlan,
    build_production_demo_asset_plan,
)
from creative_coding_assistant.orchestration.production_deployment import (
    ProductionDeploymentPlan,
    build_production_deployment_plan,
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

ProductionReadinessArea = Literal[
    "configuration_readiness",
    "safety_readiness",
    "ux_explainability_readiness",
    "deployment_readiness",
    "failure_determinism_readiness",
    "mvp_demo_readiness",
]
ProductionReadinessStatus = Literal["ready", "guarded", "blocked"]

PRODUCTION_READINESS_RECORD_SERIALIZATION_VERSION = "production_readiness_record.v1"
PRODUCTION_READINESS_REVIEW_SERIALIZATION_VERSION = "production_readiness_review.v1"
PRODUCTION_READINESS_REVIEW_AUTHORITY_BOUNDARY = (
    "Production readiness metadata aggregates configuration, "
    "safety, UX explainability, deployment, deterministic failure, and MVP demo "
    "posture for inspection only; it does not change configuration, provision "
    "providers, install runtimes, execute providers, deploy services, emit "
    "HITL requests, mutate workflows, generate assets, or write storage."
)

_REQUIRED_AREAS: tuple[ProductionReadinessArea, ...] = (
    "configuration_readiness",
    "safety_readiness",
    "ux_explainability_readiness",
    "deployment_readiness",
    "failure_determinism_readiness",
    "mvp_demo_readiness",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "configuration_mutation",
    "provider_provisioning",
    "runtime_installation",
    "provider_execution",
    "deployment_execution",
    "hitl_request_emission",
    "workflow_execution",
    "workflow_control",
    "asset_generation",
    "persistent_storage_write",
)


class ProductionReadinessRecord(BaseModel):
    """One production readiness review finding."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    record_id: str = Field(min_length=1, max_length=180)
    area: ProductionReadinessArea
    status: ProductionReadinessStatus
    source_surface_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=12)
    guarded_findings: tuple[str, ...] = Field(default_factory=tuple, max_length=24)
    blocking_findings: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    recommended_operator_actions: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    readiness_record_implemented: Literal[True] = True
    configuration_mutation_implemented: Literal[False] = False
    provider_provisioning_implemented: Literal[False] = False
    runtime_installation_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    deployment_execution_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    asset_generation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    merge_push_tag_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["production_readiness_record.v1"] = (
        PRODUCTION_READINESS_RECORD_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _record_matches_status(self) -> Self:
        if self.record_id != f"production_readiness::{self.area}":
            raise ValueError("record_id must match area")
        expected = _status_for_findings(self.guarded_findings, self.blocking_findings)
        if self.status != expected:
            raise ValueError("status must match findings")
        return self


class ProductionReadinessReview(BaseModel):
    """Aggregate production readiness metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["production_readiness_review"] = "production_readiness_review"
    serialization_version: Literal["production_readiness_review.v1"] = (
        PRODUCTION_READINESS_REVIEW_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=PRODUCTION_READINESS_REVIEW_AUTHORITY_BOUNDARY,
        max_length=1800,
    )
    source_final_optimization_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_packaging_serialization_version: str = Field(min_length=1, max_length=120)
    source_release_candidate_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_demo_assets_serialization_version: str = Field(min_length=1, max_length=120)
    source_deployment_serialization_version: str = Field(min_length=1, max_length=120)
    records: tuple[ProductionReadinessRecord, ...] = Field(min_length=6, max_length=6)
    record_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    areas: tuple[ProductionReadinessArea, ...] = Field(min_length=6, max_length=6)
    ready_record_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    guarded_record_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    blocked_record_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    record_count: int = Field(ge=6, le=6)
    guarded_finding_count: int = Field(ge=0, le=100)
    blocking_finding_count: int = Field(ge=0, le=100)
    production_readiness_status: ProductionReadinessStatus
    mvp_readiness_statement: str = Field(min_length=1, max_length=400)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    production_readiness_review_implemented: Literal[True] = True
    configuration_review_implemented: Literal[True] = True
    safety_review_implemented: Literal[True] = True
    ux_explainability_review_implemented: Literal[True] = True
    deployment_review_implemented: Literal[True] = True
    failure_review_implemented: Literal[True] = True
    mvp_demo_review_implemented: Literal[True] = True
    configuration_mutation_implemented: Literal[False] = False
    provider_provisioning_implemented: Literal[False] = False
    runtime_installation_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    deployment_execution_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    asset_generation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    merge_push_tag_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _review_matches_records(self) -> Self:
        if self.record_ids != tuple(record.record_id for record in self.records):
            raise ValueError("record_ids must match records")
        if self.areas != tuple(record.area for record in self.records):
            raise ValueError("areas must match records")
        if self.areas != _REQUIRED_AREAS:
            raise ValueError("areas must cover required readiness areas")
        if self.ready_record_ids != _record_ids_for_status(self.records, "ready"):
            raise ValueError("ready_record_ids must match records")
        if self.guarded_record_ids != _record_ids_for_status(self.records, "guarded"):
            raise ValueError("guarded_record_ids must match records")
        if self.blocked_record_ids != _record_ids_for_status(self.records, "blocked"):
            raise ValueError("blocked_record_ids must match records")
        if self.record_count != len(self.records):
            raise ValueError("record_count must match records")
        if self.guarded_finding_count != sum(
            len(record.guarded_findings) for record in self.records
        ):
            raise ValueError("guarded_finding_count must match records")
        if self.blocking_finding_count != sum(
            len(record.blocking_findings) for record in self.records
        ):
            raise ValueError("blocking_finding_count must match records")
        if self.production_readiness_status != _review_status(self.records):
            raise ValueError("production_readiness_status must match records")
        return self


def build_production_readiness_review(
    *,
    final_optimization: ProductionReleaseFinalOptimizationPlan | None = None,
    packaging: ProductionPackagingPlan | None = None,
    release_candidate: ProductionReleaseCandidatePlan | None = None,
    demo_assets: ProductionDemoAssetPlan | None = None,
    deployment: ProductionDeploymentPlan | None = None,
) -> ProductionReadinessReview:
    """Build production readiness review metadata without changing runtime state."""

    final_source = final_optimization or build_production_release_final_optimization()
    packaging_source = packaging or build_production_packaging_plan()
    candidate_source = release_candidate or build_production_release_candidate(
        final_optimization=final_source,
        packaging=packaging_source,
    )
    demo_source = demo_assets or build_production_demo_asset_plan(
        release_candidate=candidate_source,
    )
    deployment_source = deployment or build_production_deployment_plan(
        packaging=packaging_source,
    )
    records = _records(
        final_source=final_source,
        packaging_source=packaging_source,
        candidate_source=candidate_source,
        demo_source=demo_source,
        deployment_source=deployment_source,
    )
    return ProductionReadinessReview(
        source_final_optimization_serialization_version=final_source.serialization_version,
        source_packaging_serialization_version=packaging_source.serialization_version,
        source_release_candidate_serialization_version=candidate_source.serialization_version,
        source_demo_assets_serialization_version=demo_source.serialization_version,
        source_deployment_serialization_version=deployment_source.serialization_version,
        records=records,
        record_ids=tuple(record.record_id for record in records),
        areas=tuple(record.area for record in records),
        ready_record_ids=_record_ids_for_status(records, "ready"),
        guarded_record_ids=_record_ids_for_status(records, "guarded"),
        blocked_record_ids=_record_ids_for_status(records, "blocked"),
        record_count=len(records),
        guarded_finding_count=sum(len(record.guarded_findings) for record in records),
        blocking_finding_count=sum(len(record.blocking_findings) for record in records),
        production_readiness_status=_review_status(records),
        mvp_readiness_statement=(
            "Ready for local product validation with explicit user configuration "
            "and deployment assumptions; not yet an automated external deployment."
        ),
    )


def production_readiness_record_by_area(
    area: ProductionReadinessArea | str,
    review: ProductionReadinessReview | None = None,
) -> ProductionReadinessRecord | None:
    """Return one readiness record by area."""

    normalized = str(area).strip()
    source_review = review or build_production_readiness_review()
    for record in source_review.records:
        if record.area == normalized:
            return record
    return None


def production_readiness_records_for_status(
    status: ProductionReadinessStatus,
    review: ProductionReadinessReview | None = None,
) -> tuple[ProductionReadinessRecord, ...]:
    """Return readiness records by status."""

    source_review = review or build_production_readiness_review()
    return tuple(record for record in source_review.records if record.status == status)


def _records(
    *,
    final_source: ProductionReleaseFinalOptimizationPlan,
    packaging_source: ProductionPackagingPlan,
    candidate_source: ProductionReleaseCandidatePlan,
    demo_source: ProductionDemoAssetPlan,
    deployment_source: ProductionDeploymentPlan,
) -> tuple[ProductionReadinessRecord, ...]:
    configuration_findings = _unique(
        (*final_source.unavailable_reason_codes, *final_source.required_hitl_gates)
    )
    deployment_findings = tuple(deployment_source.guarded_record_ids)
    return (
        _record(
            area="configuration_readiness",
            source_ids=(final_source.role, packaging_source.role),
            evidence=(
                f"provider_ids:{','.join(final_source.provider_ids)}",
                f"env_keys:{len(packaging_source.environment_variable_keys)}",
                f"unavailable_reasons:{len(final_source.unavailable_reason_codes)}",
            ),
            guarded_findings=configuration_findings,
            blocking_findings=(),
            actions=(
                "Configure provider credentials before live provider execution.",
                "Keep missing configuration diagnostics fail-safe and visible.",
            ),
        ),
        _record(
            area="safety_readiness",
            source_ids=(final_source.role, candidate_source.role),
            evidence=(
                f"execution_mode:{final_source.selected_execution_mode_id}",
                f"candidate_status:{candidate_source.release_candidate_status}",
                "manual_assisted_auto_boundaries:documented",
            ),
            guarded_findings=tuple(candidate_source.guarded_record_ids),
            blocking_findings=(),
            actions=(
                "Preserve Manual, Assisted, and Auto mode HITL boundaries.",
                "Do not turn guarded release candidate findings into automatic execution.",
            ),
        ),
        _record(
            area="ux_explainability_readiness",
            source_ids=(final_source.role, demo_source.role),
            evidence=(
                f"explanation_fields:{len(final_source.explanation_fields)}",
                f"talking_points:{len(demo_source.explanation_talking_points)}",
                f"demo_steps:{len(demo_source.demo_workflow_steps)}",
            ),
            guarded_findings=(),
            blocking_findings=(),
            actions=(
                "Show selected provider, model, mode, strategy, estimates, fallback, and escalation reason.",
            ),
        ),
        _record(
            area="deployment_readiness",
            source_ids=(deployment_source.role,),
            evidence=(
                f"deployment_status:{deployment_source.deployment_status}",
                f"guarded_records:{len(deployment_source.guarded_record_ids)}",
                "external_manifest:explicit_guarded_assumption",
            ),
            guarded_findings=deployment_findings,
            blocking_findings=(),
            actions=(
                "Document external deployment target before production hosting.",
                "Keep local demo deployment separate from external deployment automation.",
            ),
        ),
        _record(
            area="failure_determinism_readiness",
            source_ids=(final_source.role, deployment_source.role),
            evidence=(
                "provider_unavailable:deterministic_metadata",
                "api_key_missing:guarded_metadata",
                "local_runtime_unavailable:manual_boundary",
            ),
            guarded_findings=configuration_findings,
            blocking_findings=(),
            actions=(
                "Keep provider unavailable, API key missing, local runtime, fallback, and HITL paths deterministic.",
            ),
        ),
        _record(
            area="mvp_demo_readiness",
            source_ids=(candidate_source.role, demo_source.role),
            evidence=(
                f"demo_asset_status:{demo_source.demo_asset_status}",
                f"release_candidate_status:{candidate_source.release_candidate_status}",
                "local_product_demo:ready_with_user_configuration",
            ),
            guarded_findings=tuple(candidate_source.guarded_record_ids),
            blocking_findings=(),
            actions=(
                "Use existing demo assets for local product validation.",
                "State external deployment and provider configuration assumptions explicitly.",
            ),
        ),
    )


def _record(
    *,
    area: ProductionReadinessArea,
    source_ids: tuple[str, ...],
    evidence: tuple[str, ...],
    guarded_findings: tuple[str, ...],
    blocking_findings: tuple[str, ...],
    actions: tuple[str, ...],
) -> ProductionReadinessRecord:
    return ProductionReadinessRecord(
        record_id=f"production_readiness::{area}",
        area=area,
        status=_status_for_findings(guarded_findings, blocking_findings),
        source_surface_ids=source_ids,
        evidence=evidence,
        guarded_findings=guarded_findings,
        blocking_findings=blocking_findings,
        recommended_operator_actions=actions,
    )


def _status_for_findings(
    guarded_findings: tuple[str, ...],
    blocking_findings: tuple[str, ...],
) -> ProductionReadinessStatus:
    if blocking_findings:
        return "blocked"
    if guarded_findings:
        return "guarded"
    return "ready"


def _record_ids_for_status(
    records: tuple[ProductionReadinessRecord, ...],
    status: ProductionReadinessStatus,
) -> tuple[str, ...]:
    return tuple(record.record_id for record in records if record.status == status)


def _review_status(
    records: tuple[ProductionReadinessRecord, ...],
) -> ProductionReadinessStatus:
    if any(record.status == "blocked" for record in records):
        return "blocked"
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
