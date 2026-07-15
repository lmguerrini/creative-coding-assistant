"""Production creative readiness metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.creative_analytics import (
    CreativeAnalytics,
    CreativeAnalyticsPanel,
    build_creative_analytics,
)
from creative_coding_assistant.orchestration.production_demo_assets import (
    ProductionDemoAssetPlan,
    ProductionDemoAssetRecord,
    build_production_demo_asset_plan,
)
from creative_coding_assistant.orchestration.production_readiness_review import (
    ProductionReadinessRecord,
    ProductionReadinessReview,
    build_production_readiness_review,
)

ProductionCreativeReadinessArea = Literal[
    "creative_prompt_readiness",
    "visual_preview_readiness",
    "retrieval_context_readiness",
    "creative_quality_readiness",
    "creative_diversity_consistency_readiness",
    "creative_workflow_explainability_readiness",
]
ProductionCreativeReadinessStatus = Literal["ready", "guarded", "blocked"]

PRODUCTION_CREATIVE_READINESS_RECORD_SERIALIZATION_VERSION = (
    "production_creative_readiness_record.v1"
)
PRODUCTION_CREATIVE_READINESS_REVIEW_SERIALIZATION_VERSION = (
    "production_creative_readiness_review.v1"
)
PRODUCTION_CREATIVE_READINESS_AUTHORITY_BOUNDARY = (
    "Production creative readiness metadata aggregates existing "
    "demo assets, creative analytics, and production readiness records for "
    "product validation only; it does not evaluate generated output, "
    "collect creative metrics, execute creative scoring, generate variants or "
    "assets, execute retrieval, mutate prompts or artifacts, modify generated "
    "output, route providers or models, execute providers or workflows, write "
    "memory or storage, deploy, or emit HITL requests."
)

_SOURCE_SURFACES = (
    "production_demo_assets",
    "creative_analytics",
    "production_readiness_review",
)
_REQUIRED_AREAS: tuple[ProductionCreativeReadinessArea, ...] = (
    "creative_prompt_readiness",
    "visual_preview_readiness",
    "retrieval_context_readiness",
    "creative_quality_readiness",
    "creative_diversity_consistency_readiness",
    "creative_workflow_explainability_readiness",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "generated_output_evaluation",
    "creative_metric_collection",
    "creative_scoring_execution",
    "variant_generation",
    "asset_generation",
    "retrieval_execution",
    "prompt_mutation",
    "artifact_selection",
    "artifact_mutation",
    "generated_output_modification",
    "provider_or_model_routing",
    "provider_execution",
    "workflow_execution",
    "workflow_control",
    "preview_rendering_execution",
    "human_input_request_emission",
    "memory_write",
    "persistent_storage_write",
    "deployment_execution",
)


class ProductionCreativeReadinessRecord(BaseModel):
    """One production creative readiness record."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    record_id: str = Field(min_length=1, max_length=180)
    area: ProductionCreativeReadinessArea
    status: ProductionCreativeReadinessStatus
    source_surface_ids: tuple[str, ...] = Field(min_length=1, max_length=6)
    source_serialization_versions: tuple[str, ...] = Field(min_length=1, max_length=6)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=12)
    ready_signals: tuple[str, ...] = Field(min_length=1, max_length=24)
    guarded_findings: tuple[str, ...] = Field(default_factory=tuple, max_length=24)
    blocking_findings: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    operator_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    creative_readiness_record_implemented: Literal[True] = True
    generated_output_evaluation_implemented: Literal[False] = False
    creative_metric_collection_implemented: Literal[False] = False
    creative_scoring_execution_implemented: Literal[False] = False
    variant_generation_implemented: Literal[False] = False
    asset_generation_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    artifact_selection_implemented: Literal[False] = False
    artifact_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    preview_rendering_execution_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    deployment_execution_implemented: Literal[False] = False
    merge_push_tag_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["production_creative_readiness_record.v1"] = (
        PRODUCTION_CREATIVE_READINESS_RECORD_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _record_matches_contract(self) -> Self:
        if self.record_id != f"production_creative_readiness::{self.area}":
            raise ValueError("record_id must match area")
        if len(self.source_surface_ids) != len(self.source_serialization_versions):
            raise ValueError("source ids and serialization versions must align")
        if self.status != _status_for_findings(
            self.guarded_findings,
            self.blocking_findings,
        ):
            raise ValueError("status must match findings")
        return self


class ProductionCreativeReadinessReview(BaseModel):
    """Aggregate production creative readiness metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["production_creative_readiness_review"] = (
        "production_creative_readiness_review"
    )
    serialization_version: Literal["production_creative_readiness_review.v1"] = (
        PRODUCTION_CREATIVE_READINESS_REVIEW_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=PRODUCTION_CREATIVE_READINESS_AUTHORITY_BOUNDARY,
        max_length=2200,
    )
    source_demo_assets_serialization_version: str = Field(min_length=1, max_length=120)
    source_creative_analytics_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_production_readiness_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_surfaces: tuple[str, ...] = Field(min_length=3, max_length=3)
    records: tuple[ProductionCreativeReadinessRecord, ...] = Field(
        min_length=6,
        max_length=6,
    )
    record_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    areas: tuple[ProductionCreativeReadinessArea, ...] = Field(
        min_length=6,
        max_length=6,
    )
    ready_record_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    guarded_record_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    blocked_record_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    record_count: int = Field(ge=6, le=6)
    ready_signal_count: int = Field(ge=0, le=200)
    guarded_finding_count: int = Field(ge=0, le=100)
    blocking_finding_count: int = Field(ge=0, le=100)
    creative_readiness_status: ProductionCreativeReadinessStatus
    capstone_creative_readiness_statement: str = Field(min_length=1, max_length=500)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    creative_readiness_review_implemented: Literal[True] = True
    prompt_readiness_review_implemented: Literal[True] = True
    preview_readiness_review_implemented: Literal[True] = True
    retrieval_context_review_implemented: Literal[True] = True
    creative_quality_review_implemented: Literal[True] = True
    creative_diversity_consistency_review_implemented: Literal[True] = True
    workflow_explainability_review_implemented: Literal[True] = True
    generated_output_evaluation_implemented: Literal[False] = False
    creative_metric_collection_implemented: Literal[False] = False
    creative_scoring_execution_implemented: Literal[False] = False
    variant_generation_implemented: Literal[False] = False
    asset_generation_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    artifact_selection_implemented: Literal[False] = False
    artifact_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    preview_rendering_execution_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    deployment_execution_implemented: Literal[False] = False
    merge_push_tag_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _review_matches_records(self) -> Self:
        if self.source_surfaces != _SOURCE_SURFACES:
            raise ValueError("source_surfaces must match creative readiness sources")
        if self.record_ids != tuple(record.record_id for record in self.records):
            raise ValueError("record_ids must match records")
        if self.areas != tuple(record.area for record in self.records):
            raise ValueError("areas must match records")
        if self.areas != _REQUIRED_AREAS:
            raise ValueError("areas must cover required creative readiness areas")
        if self.ready_record_ids != _record_ids_for_status(self.records, "ready"):
            raise ValueError("ready_record_ids must match records")
        if self.guarded_record_ids != _record_ids_for_status(self.records, "guarded"):
            raise ValueError("guarded_record_ids must match records")
        if self.blocked_record_ids != _record_ids_for_status(self.records, "blocked"):
            raise ValueError("blocked_record_ids must match records")
        if self.record_count != len(self.records):
            raise ValueError("record_count must match records")
        if self.ready_signal_count != sum(
            len(record.ready_signals) for record in self.records
        ):
            raise ValueError("ready_signal_count must match records")
        if self.guarded_finding_count != sum(
            len(record.guarded_findings) for record in self.records
        ):
            raise ValueError("guarded_finding_count must match records")
        if self.blocking_finding_count != sum(
            len(record.blocking_findings) for record in self.records
        ):
            raise ValueError("blocking_finding_count must match records")
        if self.creative_readiness_status != _review_status(self.records):
            raise ValueError("creative_readiness_status must match records")
        return self


def build_production_creative_readiness_review(
    *,
    demo_assets: ProductionDemoAssetPlan | None = None,
    creative_analytics: CreativeAnalytics | None = None,
    production_readiness: ProductionReadinessReview | None = None,
) -> ProductionCreativeReadinessReview:
    """Build production creative readiness metadata without evaluating output."""

    demo_source = demo_assets or build_production_demo_asset_plan()
    analytics_source = creative_analytics or build_creative_analytics()
    readiness_source = production_readiness or build_production_readiness_review(
        demo_assets=demo_source,
    )
    records = _records(
        demo_source=demo_source,
        analytics_source=analytics_source,
        readiness_source=readiness_source,
    )
    return ProductionCreativeReadinessReview(
        source_demo_assets_serialization_version=demo_source.serialization_version,
        source_creative_analytics_serialization_version=(
            analytics_source.serialization_version
        ),
        source_production_readiness_serialization_version=(
            readiness_source.serialization_version
        ),
        source_surfaces=_SOURCE_SURFACES,
        records=records,
        record_ids=tuple(record.record_id for record in records),
        areas=tuple(record.area for record in records),
        ready_record_ids=_record_ids_for_status(records, "ready"),
        guarded_record_ids=_record_ids_for_status(records, "guarded"),
        blocked_record_ids=_record_ids_for_status(records, "blocked"),
        record_count=len(records),
        ready_signal_count=sum(len(record.ready_signals) for record in records),
        guarded_finding_count=sum(len(record.guarded_findings) for record in records),
        blocking_finding_count=sum(len(record.blocking_findings) for record in records),
        creative_readiness_status=_review_status(records),
        capstone_creative_readiness_statement=(
            "Creative demo materials are available for local product validation: "
            "preview media, retrieval scenarios, workflow narrative, and "
            "explanation cues; creative analytics remain guarded metadata, not "
            "generated-output evaluation."
        ),
    )


def production_creative_readiness_record_by_area(
    area: ProductionCreativeReadinessArea | str,
    review: ProductionCreativeReadinessReview | None = None,
) -> ProductionCreativeReadinessRecord | None:
    """Return one production creative readiness record by area."""

    normalized = str(area).strip()
    source_review = review or build_production_creative_readiness_review()
    for record in source_review.records:
        if record.area == normalized:
            return record
    return None


def production_creative_readiness_records_for_status(
    status: ProductionCreativeReadinessStatus,
    review: ProductionCreativeReadinessReview | None = None,
) -> tuple[ProductionCreativeReadinessRecord, ...]:
    """Return production creative readiness records by status."""

    source_review = review or build_production_creative_readiness_review()
    return tuple(record for record in source_review.records if record.status == status)


def _records(
    *,
    demo_source: ProductionDemoAssetPlan,
    analytics_source: CreativeAnalytics,
    readiness_source: ProductionReadinessReview,
) -> tuple[ProductionCreativeReadinessRecord, ...]:
    preview = _demo_asset("preview_media", demo_source)
    retrieval = _demo_asset("retrieval_scenario_pack", demo_source)
    quality = _analytics_panel(
        "creative_analytics::quality_readiness", analytics_source
    )
    diversity = _analytics_panel(
        "creative_analytics::diversity_readiness",
        analytics_source,
    )
    consistency = _analytics_panel(
        "creative_analytics::consistency_readiness",
        analytics_source,
    )
    ux = _production_readiness_record("ux_explainability_readiness", readiness_source)
    return (
        _prompt_record(demo_source),
        _preview_record(demo_source, preview),
        _retrieval_record(demo_source, retrieval),
        _quality_record(analytics_source, quality),
        _diversity_consistency_record(analytics_source, diversity, consistency),
        _workflow_explainability_record(demo_source, readiness_source, ux),
    )


def _prompt_record(
    demo_source: ProductionDemoAssetPlan,
) -> ProductionCreativeReadinessRecord:
    prompt = demo_source.demo_prompt
    findings = _findings_for_missing_terms(
        prompt,
        ("Three.js", "audio-reactive", "explanation"),
    )
    return _record(
        area="creative_prompt_readiness",
        source_ids=(demo_source.role,),
        source_versions=(demo_source.serialization_version,),
        evidence=(
            f"demo_prompt_chars:{len(prompt)}",
            f"demo_workflow_steps:{len(demo_source.demo_workflow_steps)}",
            f"explanation_talking_points:{len(demo_source.explanation_talking_points)}",
        ),
        ready_signals=(
            "creative_coding_prompt",
            "three_js_prompt",
            "audio_reactive_direction",
            "explanation_requirements",
        ),
        guarded_findings=findings,
        blocking_findings=(),
        actions=(
            "Use the prepared prompt as the creative-coding brief.",
            "Keep prompt changes explicit and user-controlled.",
        ),
    )


def _preview_record(
    demo_source: ProductionDemoAssetPlan,
    preview: ProductionDemoAssetRecord | None,
) -> ProductionCreativeReadinessRecord:
    guarded = _asset_guarded_findings(preview)
    present_items = preview.present_items if preview is not None else ()
    return _record(
        area="visual_preview_readiness",
        source_ids=(demo_source.role,),
        source_versions=(demo_source.serialization_version,),
        evidence=(
            f"preview_media_paths:{len(demo_source.preview_media_paths)}",
            f"present_preview_items:{len(present_items)}",
            f"demo_asset_status:{demo_source.demo_asset_status}",
        ),
        ready_signals=tuple(demo_source.preview_media_paths),
        guarded_findings=guarded,
        blocking_findings=(),
        actions=(
            "Use existing preview media as fallback product evidence.",
            "Do not render or overwrite preview media during readiness review.",
        ),
    )


def _retrieval_record(
    demo_source: ProductionDemoAssetPlan,
    retrieval: ProductionDemoAssetRecord | None,
) -> ProductionCreativeReadinessRecord:
    guarded = _asset_guarded_findings(retrieval)
    if not demo_source.retrieval_scenario_ids:
        guarded = (*guarded, "missing_retrieval_scenarios")
    return _record(
        area="retrieval_context_readiness",
        source_ids=(demo_source.role,),
        source_versions=(demo_source.serialization_version,),
        evidence=(
            f"retrieval_pack:{demo_source.source_retrieval_demo_pack_id}",
            f"retrieval_scenarios:{len(demo_source.retrieval_scenario_ids)}",
            f"retrieval_asset_status:{retrieval.status if retrieval else 'missing'}",
        ),
        ready_signals=(
            demo_source.source_retrieval_demo_pack_id,
            *demo_source.retrieval_scenario_ids,
        ),
        guarded_findings=guarded,
        blocking_findings=(),
        actions=(
            "Use existing retrieval demo scenarios as creative context.",
            "Keep retrieval execution disabled during readiness review.",
        ),
    )


def _quality_record(
    analytics_source: CreativeAnalytics,
    quality: CreativeAnalyticsPanel | None,
) -> ProductionCreativeReadinessRecord:
    guarded = _panel_guarded_findings(quality)
    return _record(
        area="creative_quality_readiness",
        source_ids=(analytics_source.role,),
        source_versions=(analytics_source.serialization_version,),
        evidence=(
            f"creative_analytics_status:{analytics_source.creative_analytics_status}",
            f"quality_panel_status:{quality.status if quality else 'missing'}",
            f"creative_signals:{quality.creative_signal_count if quality else 0}",
        ),
        ready_signals=(
            "quality_readiness_panel_linked",
            f"guardrail_signals:{quality.guardrail_signal_count if quality else 0}",
        ),
        guarded_findings=guarded,
        blocking_findings=(),
        actions=(
            "Expose quality readiness as passive metadata only.",
            "Do not evaluate generated output or execute creative scoring.",
        ),
    )


def _diversity_consistency_record(
    analytics_source: CreativeAnalytics,
    diversity: CreativeAnalyticsPanel | None,
    consistency: CreativeAnalyticsPanel | None,
) -> ProductionCreativeReadinessRecord:
    guarded = _unique(
        (
            *_panel_guarded_findings(diversity),
            *_panel_guarded_findings(consistency),
        )
    )
    return _record(
        area="creative_diversity_consistency_readiness",
        source_ids=(analytics_source.role,),
        source_versions=(analytics_source.serialization_version,),
        evidence=(
            f"diversity_panel_status:{diversity.status if diversity else 'missing'}",
            f"consistency_panel_status:{consistency.status if consistency else 'missing'}",
            f"analytics_guarded_panels:{len(analytics_source.guarded_panel_ids)}",
        ),
        ready_signals=(
            "diversity_readiness_panel_linked",
            "consistency_readiness_panel_linked",
        ),
        guarded_findings=guarded,
        blocking_findings=(),
        actions=(
            "Use diversity and consistency readiness as review metadata.",
            "Do not generate variants or validate generated-output consistency.",
        ),
    )


def _workflow_explainability_record(
    demo_source: ProductionDemoAssetPlan,
    readiness_source: ProductionReadinessReview,
    ux: ProductionReadinessRecord | None,
) -> ProductionCreativeReadinessRecord:
    guarded: tuple[str, ...] = ()
    if ux is None:
        guarded = ("missing_ux_explainability_readiness_record",)
    elif ux.status != "ready":
        guarded = tuple(ux.guarded_findings) or (ux.record_id,)
    if len(demo_source.explanation_talking_points) < 9:
        guarded = (*guarded, "incomplete_explanation_talking_points")
    return _record(
        area="creative_workflow_explainability_readiness",
        source_ids=(demo_source.role, readiness_source.role),
        source_versions=(
            demo_source.serialization_version,
            readiness_source.serialization_version,
        ),
        evidence=(
            f"workflow_steps:{len(demo_source.demo_workflow_steps)}",
            f"explanation_talking_points:{len(demo_source.explanation_talking_points)}",
            f"ux_readiness_status:{ux.status if ux else 'missing'}",
        ),
        ready_signals=(
            *demo_source.demo_workflow_steps,
            *demo_source.explanation_talking_points,
        ),
        guarded_findings=_unique(guarded),
        blocking_findings=(),
        actions=(
            "Walk through the creative workflow narrative with visible explanations.",
            "Keep workflow execution and HITL emission outside readiness metadata.",
        ),
    )


def _record(
    *,
    area: ProductionCreativeReadinessArea,
    source_ids: tuple[str, ...],
    source_versions: tuple[str, ...],
    evidence: tuple[str, ...],
    ready_signals: tuple[str, ...],
    guarded_findings: tuple[str, ...],
    blocking_findings: tuple[str, ...],
    actions: tuple[str, ...],
) -> ProductionCreativeReadinessRecord:
    return ProductionCreativeReadinessRecord(
        record_id=f"production_creative_readiness::{area}",
        area=area,
        status=_status_for_findings(guarded_findings, blocking_findings),
        source_surface_ids=source_ids,
        source_serialization_versions=source_versions,
        evidence=evidence,
        ready_signals=_unique(ready_signals),
        guarded_findings=_unique(guarded_findings),
        blocking_findings=_unique(blocking_findings),
        operator_actions=actions,
    )


def _demo_asset(
    asset_kind: str,
    demo_source: ProductionDemoAssetPlan,
) -> ProductionDemoAssetRecord | None:
    for record in demo_source.records:
        if record.asset_kind == asset_kind:
            return record
    return None


def _analytics_panel(
    panel_id: str,
    analytics_source: CreativeAnalytics,
) -> CreativeAnalyticsPanel | None:
    for panel in analytics_source.panels:
        if panel.panel_id == panel_id:
            return panel
    return None


def _production_readiness_record(
    area: str,
    readiness_source: ProductionReadinessReview,
) -> ProductionReadinessRecord | None:
    for record in readiness_source.records:
        if record.area == area:
            return record
    return None


def _asset_guarded_findings(
    record: ProductionDemoAssetRecord | None,
) -> tuple[str, ...]:
    if record is None:
        return ("missing_demo_asset_record",)
    if record.status == "ready":
        return ()
    return tuple(f"missing_demo_asset_item:{item}" for item in record.missing_items)


def _panel_guarded_findings(
    panel: CreativeAnalyticsPanel | None,
) -> tuple[str, ...]:
    if panel is None:
        return ("missing_creative_analytics_panel",)
    if panel.status == "ready":
        return ()
    return (panel.panel_id,)


def _findings_for_missing_terms(text: str, terms: tuple[str, ...]) -> tuple[str, ...]:
    normalized = text.casefold()
    return tuple(
        f"missing_prompt_term:{term}"
        for term in terms
        if term.casefold() not in normalized
    )


def _status_for_findings(
    guarded_findings: tuple[str, ...],
    blocking_findings: tuple[str, ...],
) -> ProductionCreativeReadinessStatus:
    if blocking_findings:
        return "blocked"
    if guarded_findings:
        return "guarded"
    return "ready"


def _record_ids_for_status(
    records: tuple[ProductionCreativeReadinessRecord, ...],
    status: ProductionCreativeReadinessStatus,
) -> tuple[str, ...]:
    return tuple(record.record_id for record in records if record.status == status)


def _review_status(
    records: tuple[ProductionCreativeReadinessRecord, ...],
) -> ProductionCreativeReadinessStatus:
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
