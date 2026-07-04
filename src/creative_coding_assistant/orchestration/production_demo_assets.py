"""V5.6 production demo asset readiness metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.eval import (
    RetrievalDemoPack,
    build_capstone_retrieval_demo_pack,
)
from creative_coding_assistant.orchestration.production_release_candidate import (
    ProductionReleaseCandidatePlan,
    build_production_release_candidate,
)

DemoAssetKind = Literal[
    "preview_media",
    "demo_prompt",
    "retrieval_scenario_pack",
    "workflow_narrative",
    "explanation_talking_points",
]
DemoAssetStatus = Literal["ready", "guarded"]

PRODUCTION_DEMO_ASSET_RECORD_SERIALIZATION_VERSION = "production_demo_asset_record.v1"
PRODUCTION_DEMO_ASSET_PLAN_SERIALIZATION_VERSION = "production_demo_asset_plan.v1"
PRODUCTION_DEMO_ASSET_AUTHORITY_BOUNDARY = (
    "V5.6 production demo asset metadata inventories existing preview media, "
    "demo prompts, retrieval scenarios, workflow narrative steps, and "
    "explanation talking points for operator preparation only; it does not "
    "generate assets, execute retrieval, run generation, render previews, "
    "write project bundles, mutate artifacts, call providers, change routing, "
    "deploy, merge, push, tag, or apply Runtime Evolution."
)

_REQUIRED_ASSET_KINDS: tuple[DemoAssetKind, ...] = (
    "preview_media",
    "demo_prompt",
    "retrieval_scenario_pack",
    "workflow_narrative",
    "explanation_talking_points",
)
_PREVIEW_MEDIA_PATHS = (
    "assets/preview_current.png",
    "assets/preview_v1.png",
    "assets/preview_v2.png",
)
_DEMO_PROMPT = (
    "Create a luminous audio-reactive Three.js scene for a capstone demo: "
    "concentric geometry, subtle bloom, FFT-driven motion accents, and a clear "
    "explanation of provider choice, execution mode, estimates, fallback, and "
    "escalation boundaries."
)
_DEMO_WORKFLOW_STEPS = (
    "Task",
    "Routing Intelligence",
    "Adaptive Execution Policy",
    "Execution Simulation",
    "Generation",
    "Artifact",
    "Explanation",
    "Final Output",
)
_EXPLANATION_TALKING_POINTS = (
    "selected provider",
    "selected model",
    "execution mode",
    "execution strategy",
    "quality estimate",
    "cost estimate",
    "latency estimate",
    "fallback strategy",
    "escalation reason",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "asset_generation",
    "retrieval_execution",
    "provider_execution",
    "preview_rendering_execution",
    "project_bundle_write",
    "artifact_mutation",
    "provider_or_model_routing_mutation",
    "workflow_execution",
    "workflow_control",
    "deployment_execution",
    "merge_push_tag_operation",
    "runtime_evolution_application",
)


class ProductionDemoAssetRecord(BaseModel):
    """One production demo asset readiness record."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    asset_id: str = Field(min_length=1, max_length=180)
    asset_kind: DemoAssetKind
    status: DemoAssetStatus
    source_refs: tuple[str, ...] = Field(min_length=1, max_length=16)
    required_items: tuple[str, ...] = Field(min_length=1, max_length=24)
    present_items: tuple[str, ...] = Field(default_factory=tuple, max_length=24)
    missing_items: tuple[str, ...] = Field(default_factory=tuple, max_length=24)
    operator_notes: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    demo_asset_record_implemented: Literal[True] = True
    asset_generation_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    preview_rendering_execution_implemented: Literal[False] = False
    project_bundle_write_implemented: Literal[False] = False
    artifact_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    deployment_execution_implemented: Literal[False] = False
    merge_push_tag_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["production_demo_asset_record.v1"] = (
        PRODUCTION_DEMO_ASSET_RECORD_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _record_matches_items(self) -> Self:
        if self.asset_id != f"production_demo_asset::{self.asset_kind}":
            raise ValueError("asset_id must match asset_kind")
        if self.missing_items != tuple(
            item for item in self.required_items if item not in self.present_items
        ):
            raise ValueError("missing_items must match required and present items")
        if self.status != ("guarded" if self.missing_items else "ready"):
            raise ValueError("status must match missing items")
        return self


class ProductionDemoAssetPlan(BaseModel):
    """Production demo asset plan over existing assets and demo metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["production_demo_assets"] = "production_demo_assets"
    serialization_version: Literal["production_demo_asset_plan.v1"] = (
        PRODUCTION_DEMO_ASSET_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=PRODUCTION_DEMO_ASSET_AUTHORITY_BOUNDARY,
        max_length=1800,
    )
    source_release_candidate_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_retrieval_demo_pack_id: str = Field(min_length=1, max_length=180)
    demo_prompt: str = Field(min_length=1, max_length=900)
    demo_workflow_steps: tuple[str, ...] = Field(min_length=8, max_length=8)
    explanation_talking_points: tuple[str, ...] = Field(min_length=9, max_length=9)
    preview_media_paths: tuple[str, ...] = Field(min_length=3, max_length=3)
    retrieval_scenario_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    records: tuple[ProductionDemoAssetRecord, ...] = Field(min_length=5, max_length=5)
    asset_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    asset_kinds: tuple[DemoAssetKind, ...] = Field(min_length=5, max_length=5)
    ready_asset_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    guarded_asset_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    asset_count: int = Field(ge=5, le=5)
    demo_asset_status: DemoAssetStatus
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    demo_asset_metadata_implemented: Literal[True] = True
    preview_media_inventory_implemented: Literal[True] = True
    demo_prompt_implemented: Literal[True] = True
    retrieval_demo_pack_linked: Literal[True] = True
    workflow_narrative_implemented: Literal[True] = True
    explanation_talking_points_implemented: Literal[True] = True
    asset_generation_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    preview_rendering_execution_implemented: Literal[False] = False
    project_bundle_write_implemented: Literal[False] = False
    artifact_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    deployment_execution_implemented: Literal[False] = False
    merge_push_tag_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_records(self) -> Self:
        if self.asset_ids != tuple(record.asset_id for record in self.records):
            raise ValueError("asset_ids must match records")
        if self.asset_kinds != tuple(record.asset_kind for record in self.records):
            raise ValueError("asset_kinds must match records")
        if self.asset_kinds != _REQUIRED_ASSET_KINDS:
            raise ValueError("asset_kinds must cover required demo asset kinds")
        if self.ready_asset_ids != _asset_ids_for_status(self.records, "ready"):
            raise ValueError("ready_asset_ids must match records")
        if self.guarded_asset_ids != _asset_ids_for_status(self.records, "guarded"):
            raise ValueError("guarded_asset_ids must match records")
        if self.asset_count != len(self.records):
            raise ValueError("asset_count must match records")
        if self.demo_asset_status != _plan_status(self.records):
            raise ValueError("demo_asset_status must match records")
        if self.demo_workflow_steps != _DEMO_WORKFLOW_STEPS:
            raise ValueError("demo_workflow_steps must match release demo flow")
        if self.explanation_talking_points != _EXPLANATION_TALKING_POINTS:
            raise ValueError("explanation_talking_points must match UX review")
        return self


def build_production_demo_asset_plan(
    project_root: str | Path | None = None,
    *,
    release_candidate: ProductionReleaseCandidatePlan | None = None,
    retrieval_demo_pack: RetrievalDemoPack | None = None,
) -> ProductionDemoAssetPlan:
    """Build demo asset metadata without running the demo."""

    root = Path(project_root or ".").resolve()
    release_source = release_candidate or build_production_release_candidate()
    retrieval_source = retrieval_demo_pack or build_capstone_retrieval_demo_pack()
    records = _records(root=root, retrieval_demo_pack=retrieval_source)
    return ProductionDemoAssetPlan(
        source_release_candidate_serialization_version=(
            release_source.serialization_version
        ),
        source_retrieval_demo_pack_id=retrieval_source.pack_id,
        demo_prompt=_DEMO_PROMPT,
        demo_workflow_steps=_DEMO_WORKFLOW_STEPS,
        explanation_talking_points=_EXPLANATION_TALKING_POINTS,
        preview_media_paths=_PREVIEW_MEDIA_PATHS,
        retrieval_scenario_ids=tuple(
            scenario.demo_id for scenario in retrieval_source.scenarios
        ),
        records=records,
        asset_ids=tuple(record.asset_id for record in records),
        asset_kinds=tuple(record.asset_kind for record in records),
        ready_asset_ids=_asset_ids_for_status(records, "ready"),
        guarded_asset_ids=_asset_ids_for_status(records, "guarded"),
        asset_count=len(records),
        demo_asset_status=_plan_status(records),
    )


def production_demo_asset_by_kind(
    asset_kind: DemoAssetKind | str,
    plan: ProductionDemoAssetPlan | None = None,
) -> ProductionDemoAssetRecord | None:
    """Return one demo asset record by kind."""

    normalized = str(asset_kind).strip()
    source_plan = plan or build_production_demo_asset_plan()
    for record in source_plan.records:
        if record.asset_kind == normalized:
            return record
    return None


def production_demo_assets_for_status(
    status: DemoAssetStatus,
    plan: ProductionDemoAssetPlan | None = None,
) -> tuple[ProductionDemoAssetRecord, ...]:
    """Return demo asset records by readiness status."""

    source_plan = plan or build_production_demo_asset_plan()
    return tuple(record for record in source_plan.records if record.status == status)


def _records(
    *,
    root: Path,
    retrieval_demo_pack: RetrievalDemoPack,
) -> tuple[ProductionDemoAssetRecord, ...]:
    scenario_ids = tuple(scenario.demo_id for scenario in retrieval_demo_pack.scenarios)
    return (
        _record(
            asset_kind="preview_media",
            source_refs=_PREVIEW_MEDIA_PATHS,
            required_items=_PREVIEW_MEDIA_PATHS,
            present_items=tuple(
                path for path in _PREVIEW_MEDIA_PATHS if (root / path).exists()
            ),
            operator_notes=(
                "Use existing workstation preview screenshots as visual context.",
                "Do not regenerate preview media during the demo asset check.",
            ),
        ),
        _record(
            asset_kind="demo_prompt",
            source_refs=("v5_6_demo_prompt",),
            required_items=(
                "capstone_prompt",
                "provider_explanation",
                "fallback_explanation",
            ),
            present_items=(
                "capstone_prompt",
                "provider_explanation",
                "fallback_explanation",
            ),
            operator_notes=(
                "Start from the prepared capstone creative-coding prompt.",
                "Keep provider, model, mode, estimate, fallback, and escalation explanation visible.",
            ),
        ),
        _record(
            asset_kind="retrieval_scenario_pack",
            source_refs=(retrieval_demo_pack.pack_id,),
            required_items=scenario_ids,
            present_items=scenario_ids,
            operator_notes=(
                f"Use {len(scenario_ids)} existing capstone retrieval scenarios.",
                "Do not execute retrieval during asset inventory.",
            ),
        ),
        _record(
            asset_kind="workflow_narrative",
            source_refs=("v5_6_end_to_end_demo_flow",),
            required_items=_DEMO_WORKFLOW_STEPS,
            present_items=_DEMO_WORKFLOW_STEPS,
            operator_notes=(
                "Narrate task to routing to policy to simulation to generation to artifact to explanation.",
                "Keep workflow execution separate from the metadata inventory.",
            ),
        ),
        _record(
            asset_kind="explanation_talking_points",
            source_refs=("production_ux_review_fields",),
            required_items=_EXPLANATION_TALKING_POINTS,
            present_items=_EXPLANATION_TALKING_POINTS,
            operator_notes=(
                "Explain selected provider, model, mode, strategy, estimates, fallback, and escalation.",
                "Make manual/HITL boundaries explicit during the demo.",
            ),
        ),
    )


def _record(
    *,
    asset_kind: DemoAssetKind,
    source_refs: tuple[str, ...],
    required_items: tuple[str, ...],
    present_items: tuple[str, ...],
    operator_notes: tuple[str, ...],
) -> ProductionDemoAssetRecord:
    missing = tuple(item for item in required_items if item not in present_items)
    return ProductionDemoAssetRecord(
        asset_id=f"production_demo_asset::{asset_kind}",
        asset_kind=asset_kind,
        status="guarded" if missing else "ready",
        source_refs=source_refs,
        required_items=required_items,
        present_items=present_items,
        missing_items=missing,
        operator_notes=operator_notes,
    )


def _asset_ids_for_status(
    records: tuple[ProductionDemoAssetRecord, ...],
    status: DemoAssetStatus,
) -> tuple[str, ...]:
    return tuple(record.asset_id for record in records if record.status == status)


def _plan_status(records: tuple[ProductionDemoAssetRecord, ...]) -> DemoAssetStatus:
    if any(record.status == "guarded" for record in records):
        return "guarded"
    return "ready"
