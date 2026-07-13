"""V8.8 capstone demo and showcase readiness metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.eval import RetrievalDemoPack, build_capstone_retrieval_demo_pack
from creative_coding_assistant.orchestration.advisory.production_creative_readiness_review import (
    ProductionCreativeReadinessReview,
    build_production_creative_readiness_review,
)
from creative_coding_assistant.orchestration.advisory.production_demo_assets import (
    ProductionDemoAssetPlan,
    build_production_demo_asset_plan,
)

DemoShowcaseCoverageStatus = Literal["prepared", "supported_existing", "guarded"]
DemoShowcaseMetricStatus = Literal["ready", "guarded", "manual"]
DemoShowcaseChecklistStatus = Literal["ready", "manual", "blocked_until_hitl"]
DemoShowcaseFallbackTrigger = Literal[
    "provider_failure",
    "retrieval_unavailable",
    "preview_unavailable",
    "network_unavailable",
    "time_overrun",
]
CapstoneCaseId = Literal[
    "case_1_rag_knowledge_assistant",
    "case_2_bounded_agent_automation",
    "case_3_source_grounded_search",
    "case_5_ai_coding_assistant",
    "case_6_advanced_llm_tools",
]

DEMO_SHOWCASE_PLAN_SERIALIZATION_VERSION = "demo_showcase_plan.v1"
DEMO_SHOWCASE_RECORD_SERIALIZATION_VERSION = "demo_showcase_record.v1"
DEMO_SHOWCASE_AUTHORITY_BOUNDARY = (
    "V8.8 demo showcase metadata prepares capstone demo prompts, flows, case "
    "alignment, metrics summaries, fallback scripts, and showcase checklists "
    "over existing product surfaces only; it does not execute providers, run "
    "retrieval, render previews, route models, mutate prompts or artifacts, "
    "write persistent storage, call external DCC or MCP tools, implement "
    "future autonomous or immersive execution platforms, deploy, merge, push, tag, freeze the release, or "
    "start the V8 Grand Review."
)

_ROADMAP_ITEMS = (
    "Demo Mode",
    "Golden Demo Flows",
    "Capstone Case Alignment",
    "Internal Preview Showcase",
    "Demo Prompt Library",
    "Evaluation Dashboard",
    "README Finalization",
    "Showcase Upload Preparation",
    "SCR Presentation Support",
    "SMART Presentation Support",
    "Ethical AI Summary",
    "Demo Fallback Mode",
    "Presentation Polish",
    "Manual Demo Checklist",
    "Demo Reliability Validation",
    "Golden Demo Dataset",
    "Offline Demo Fallback",
    "Provider Failure Recovery",
    "Demo Metrics Dashboard",
)
_REQUIRED_CASE_IDS: tuple[CapstoneCaseId, ...] = (
    "case_1_rag_knowledge_assistant",
    "case_2_bounded_agent_automation",
    "case_3_source_grounded_search",
    "case_5_ai_coding_assistant",
    "case_6_advanced_llm_tools",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "provider_execution",
    "retrieval_execution",
    "preview_rendering_execution",
    "provider_or_model_routing_mutation",
    "workflow_control",
    "artifact_mutation",
    "persistent_storage_write",
    "external_dcc_execution",
    "mcp_tool_execution",
    "holomind_implementation",
    "holoiverse_implementation",
    "deployment_execution",
    "merge_push_tag_operation",
    "version_freeze",
    "grand_review_start",
)


class DemoShowcaseCoverageRecord(BaseModel):
    """One V8.8 roadmap item mapped to a truthful demo/showcase surface."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    roadmap_item: str = Field(min_length=1, max_length=120)
    status: DemoShowcaseCoverageStatus
    implementation_surface: str = Field(min_length=1, max_length=240)
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=8)
    claim_boundary: str = Field(min_length=1, max_length=520)
    hitl_review_required: bool = True
    serialization_version: Literal["demo_showcase_record.v1"] = DEMO_SHOWCASE_RECORD_SERIALIZATION_VERSION


class CapstoneCaseAlignment(BaseModel):
    """Demo claim posture for one Capstone case."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    case_id: CapstoneCaseId
    case_label: str = Field(min_length=1, max_length=160)
    alignment_status: Literal["primary", "guarded_support"]
    demo_claim: str = Field(min_length=1, max_length=520)
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=8)
    boundary: str = Field(min_length=1, max_length=520)


class DemoPromptRecord(BaseModel):
    """One prompt in the V8.8 demo prompt library."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    prompt_id: str = Field(pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    title: str = Field(min_length=1, max_length=160)
    capstone_cases: tuple[CapstoneCaseId, ...] = Field(min_length=1, max_length=5)
    prompt_text: str = Field(min_length=1, max_length=1400)
    expected_demo_value: str = Field(min_length=1, max_length=420)
    fallback_notes: tuple[str, ...] = Field(min_length=1, max_length=6)


class GoldenDemoFlow(BaseModel):
    """One timed golden demo flow for operator rehearsal."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    flow_id: str = Field(pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    title: str = Field(min_length=1, max_length=180)
    duration_seconds: int = Field(ge=30, le=600)
    primary_prompt_id: str = Field(pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    capstone_cases: tuple[CapstoneCaseId, ...] = Field(min_length=1, max_length=5)
    operator_steps: tuple[str, ...] = Field(min_length=3, max_length=12)
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=8)
    fallback_trigger: DemoShowcaseFallbackTrigger


class DemoMetricSummary(BaseModel):
    """One demo metric or readiness signal exposed without collecting live metrics."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    metric_id: str = Field(pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    label: str = Field(min_length=1, max_length=160)
    status: DemoShowcaseMetricStatus
    value: str = Field(min_length=1, max_length=220)
    source_refs: tuple[str, ...] = Field(min_length=1, max_length=8)
    interpretation: str = Field(min_length=1, max_length=520)
    live_metric_collection_implemented: Literal[False] = False


class DemoFallbackPlan(BaseModel):
    """Operator fallback for a known demo risk."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    trigger: DemoShowcaseFallbackTrigger
    fallback_mode: str = Field(min_length=1, max_length=160)
    operator_action: str = Field(min_length=1, max_length=520)
    audience_framing: str = Field(min_length=1, max_length=520)
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=8)


class DemoChecklistItem(BaseModel):
    """Manual demo, reliability, or showcase-upload checklist item."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    item_id: str = Field(pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    category: Literal[
        "manual_demo",
        "reliability",
        "showcase_upload",
        "presentation",
        "ethics",
    ]
    status: DemoShowcaseChecklistStatus
    action: str = Field(min_length=1, max_length=520)
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=8)


class DemoPresentationSegment(BaseModel):
    """Timed presentation segment for the 10-minute demo and 5-minute Q&A."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    segment_id: str = Field(pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    phase: Literal["ten_minute_demo", "five_minute_qa"]
    duration_seconds: int = Field(ge=30, le=300)
    purpose: str = Field(min_length=1, max_length=260)
    talking_points: tuple[str, ...] = Field(min_length=1, max_length=8)


class DemoShowcasePlan(BaseModel):
    """V8.8 capstone demo and showcase readiness plan over existing surfaces."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["demo_showcase_experience"] = "demo_showcase_experience"
    serialization_version: Literal["demo_showcase_plan.v1"] = DEMO_SHOWCASE_PLAN_SERIALIZATION_VERSION
    authority_boundary: str = Field(default=DEMO_SHOWCASE_AUTHORITY_BOUNDARY, max_length=1800)
    source_demo_assets_serialization_version: str = Field(min_length=1, max_length=120)
    source_creative_readiness_serialization_version: str = Field(min_length=1, max_length=120)
    source_retrieval_demo_pack_id: str = Field(min_length=1, max_length=180)
    coverage_records: tuple[DemoShowcaseCoverageRecord, ...] = Field(min_length=19, max_length=19)
    coverage_items: tuple[str, ...] = Field(min_length=19, max_length=19)
    capstone_case_alignments: tuple[CapstoneCaseAlignment, ...] = Field(min_length=5, max_length=5)
    golden_demo_flows: tuple[GoldenDemoFlow, ...] = Field(min_length=4, max_length=8)
    demo_prompt_library: tuple[DemoPromptRecord, ...] = Field(min_length=5, max_length=10)
    demo_metrics: tuple[DemoMetricSummary, ...] = Field(min_length=5, max_length=10)
    fallback_plans: tuple[DemoFallbackPlan, ...] = Field(min_length=5, max_length=5)
    checklist_items: tuple[DemoChecklistItem, ...] = Field(min_length=10, max_length=18)
    presentation_segments: tuple[DemoPresentationSegment, ...] = Field(min_length=7, max_length=10)
    documentation_refs: tuple[str, ...] = Field(min_length=5, max_length=12)
    dataset_refs: tuple[str, ...] = Field(min_length=1, max_length=6)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=20,
    )
    demo_mode_prepared: Literal[True] = True
    golden_demo_flows_prepared: Literal[True] = True
    capstone_case_alignment_prepared: Literal[True] = True
    internal_preview_showcase_prepared: Literal[True] = True
    demo_prompt_library_prepared: Literal[True] = True
    evaluation_summary_prepared: Literal[True] = True
    readme_finalization_draft_prepared: Literal[True] = True
    showcase_upload_prepared: Literal[True] = True
    ethical_ai_summary_prepared: Literal[True] = True
    fallback_mode_prepared: Literal[True] = True
    demo_reliability_validation_prepared: Literal[True] = True
    metadata_only: Literal[True] = True
    provider_execution_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    preview_rendering_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    artifact_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    external_dcc_execution_implemented: Literal[False] = False
    mcp_tool_execution_implemented: Literal[False] = False
    holomind_implemented: Literal[False] = False
    holoiverse_implemented: Literal[False] = False
    merge_push_tag_implemented: Literal[False] = False
    version_freeze_implemented: Literal[False] = False
    grand_review_started: Literal[False] = False

    @model_validator(mode="after")
    def _plan_matches_records(self) -> Self:
        if self.coverage_items != tuple(record.roadmap_item for record in self.coverage_records):
            raise ValueError("coverage_items must match coverage records")
        if self.coverage_items != _ROADMAP_ITEMS:
            raise ValueError("coverage_items must cover the V8.8 roadmap")
        if tuple(alignment.case_id for alignment in self.capstone_case_alignments) != _REQUIRED_CASE_IDS:
            raise ValueError("capstone case alignment must cover required cases")
        if len({prompt.prompt_id for prompt in self.demo_prompt_library}) != len(self.demo_prompt_library):
            raise ValueError("demo prompt ids must be unique")
        prompt_ids = {prompt.prompt_id for prompt in self.demo_prompt_library}
        for flow in self.golden_demo_flows:
            if flow.primary_prompt_id not in prompt_ids:
                raise ValueError("golden demo flows must reference prompt library ids")
        if tuple(plan.trigger for plan in self.fallback_plans) != (
            "provider_failure",
            "retrieval_unavailable",
            "preview_unavailable",
            "network_unavailable",
            "time_overrun",
        ):
            raise ValueError("fallback plans must cover required demo triggers")
        demo_seconds = sum(
            segment.duration_seconds for segment in self.presentation_segments if segment.phase == "ten_minute_demo"
        )
        qa_seconds = sum(
            segment.duration_seconds for segment in self.presentation_segments if segment.phase == "five_minute_qa"
        )
        if demo_seconds != 600:
            raise ValueError("ten-minute demo segments must total 600 seconds")
        if qa_seconds != 300:
            raise ValueError("five-minute Q&A segments must total 300 seconds")
        return self


def build_demo_showcase_plan(
    project_root: str | Path | None = None,
    *,
    demo_assets: ProductionDemoAssetPlan | None = None,
    creative_readiness: ProductionCreativeReadinessReview | None = None,
    retrieval_demo_pack: RetrievalDemoPack | None = None,
) -> DemoShowcasePlan:
    """Build the V8.8 demo/showcase readiness plan without running the demo."""

    root = Path(project_root or ".").resolve()
    demo_source = demo_assets or build_production_demo_asset_plan(root)
    creative_source = creative_readiness or build_production_creative_readiness_review()
    retrieval_source = retrieval_demo_pack or build_capstone_retrieval_demo_pack()
    prompts = _prompt_library(demo_source=demo_source)
    return DemoShowcasePlan(
        source_demo_assets_serialization_version=demo_source.serialization_version,
        source_creative_readiness_serialization_version=creative_source.serialization_version,
        source_retrieval_demo_pack_id=retrieval_source.pack_id,
        coverage_records=_coverage_records(),
        coverage_items=_ROADMAP_ITEMS,
        capstone_case_alignments=_case_alignments(),
        golden_demo_flows=_golden_demo_flows(prompts=prompts),
        demo_prompt_library=prompts,
        demo_metrics=_demo_metrics(
            demo_assets=demo_source,
            creative_readiness=creative_source,
            retrieval_demo_pack=retrieval_source,
        ),
        fallback_plans=_fallback_plans(),
        checklist_items=_checklist_items(),
        presentation_segments=_presentation_segments(),
        documentation_refs=(
            "README.md",
            "docs/CAPSTONE_DEMO_SHOWCASE.md",
            "docs/CAPSTONE_EVALUATION_ETHICS.md",
            "demo/README.md",
            "demo/demo_prompt_library.md",
            "demo/manual_demo_checklist.md",
            "demo/showcase_upload_preparation.md",
            "architecture/demo_showcase_experience.md",
        ),
        dataset_refs=("demo/golden_demo_dataset.json",),
    )


def demo_showcase_coverage_by_item(
    roadmap_item: str,
    plan: DemoShowcasePlan | None = None,
) -> DemoShowcaseCoverageRecord | None:
    """Return one V8.8 coverage record by roadmap item."""

    source_plan = plan or build_demo_showcase_plan()
    return next((record for record in source_plan.coverage_records if record.roadmap_item == roadmap_item), None)


def demo_showcase_flow_by_id(
    flow_id: str,
    plan: DemoShowcasePlan | None = None,
) -> GoldenDemoFlow | None:
    """Return one golden demo flow by id."""

    source_plan = plan or build_demo_showcase_plan()
    return next((flow for flow in source_plan.golden_demo_flows if flow.flow_id == flow_id), None)


def demo_showcase_prompt_by_id(
    prompt_id: str,
    plan: DemoShowcasePlan | None = None,
) -> DemoPromptRecord | None:
    """Return one demo prompt by id."""

    source_plan = plan or build_demo_showcase_plan()
    return next((prompt for prompt in source_plan.demo_prompt_library if prompt.prompt_id == prompt_id), None)


def demo_showcase_fallback_by_trigger(
    trigger: DemoShowcaseFallbackTrigger | str,
    plan: DemoShowcasePlan | None = None,
) -> DemoFallbackPlan | None:
    """Return one demo fallback plan by trigger."""

    source_plan = plan or build_demo_showcase_plan()
    normalized = str(trigger).strip()
    return next((fallback for fallback in source_plan.fallback_plans if fallback.trigger == normalized), None)


def _coverage_records() -> tuple[DemoShowcaseCoverageRecord, ...]:
    rows: tuple[tuple[str, DemoShowcaseCoverageStatus, str, tuple[str, ...], str], ...] = (
        (
            "Demo Mode",
            "prepared",
            "operator-facing demo script and golden dataset",
            ("docs/CAPSTONE_DEMO_SHOWCASE.md", "demo/golden_demo_dataset.json"),
            "Prepared as manual demo mode, not a new runtime UI mode or automatic workflow controller.",
        ),
        (
            "Golden Demo Flows",
            "prepared",
            "timed golden flow records and demo docs",
            ("demo/README.md", "demo/golden_demo_dataset.json"),
            "Flows are rehearsal scripts over existing product behavior; they do not execute the assistant.",
        ),
        (
            "Capstone Case Alignment",
            "prepared",
            "case alignment matrix for Cases 1, 2, 3, 5, and 6",
            ("docs/CAPSTONE_DEMO_SHOWCASE.md",),
            "Cases 2 and 3 are guarded claims only where existing bounded workflow/search behavior supports them.",
        ),
        (
            "Internal Preview Showcase",
            "supported_existing",
            "existing preview screenshots and preview surfaces",
            ("assets/preview_current.png", "src/creative_coding_assistant/preview/contracts.py"),
            (
                "Preview is showcased through existing internal preview artifacts; "
                "V8.8 does not render or repair previews."
            ),
        ),
        (
            "Demo Prompt Library",
            "prepared",
            "prompt library records and markdown prompt bank",
            ("demo/demo_prompt_library.md",),
            "Prompts are operator assets; they do not mutate product prompts or provider selection.",
        ),
        (
            "Evaluation Dashboard",
            "supported_existing",
            "RAGAs docs, live-session eval CLI, and readiness metrics summary",
            ("docs/eval_pipeline.md", "scripts/eval_live_sessions.py"),
            "Summary references supported manual evaluation; V8.8 does not collect live metrics.",
        ),
        (
            "README Finalization",
            "prepared",
            "README capstone positioning and claim boundaries",
            ("README.md",),
            "README updates are a finalization draft before HITL and Grand Review.",
        ),
        (
            "Showcase Upload Preparation",
            "prepared",
            "showcase upload checklist and manifest",
            ("demo/showcase_upload_preparation.md",),
            "Upload preparation is a checklist only; V8.8 does not upload or publish artifacts.",
        ),
        (
            "SCR Presentation Support",
            "prepared",
            "situation, challenge, response presentation structure",
            ("docs/CAPSTONE_DEMO_SHOWCASE.md",),
            "Presentation support is scripted guidance, not generated slides.",
        ),
        (
            "SMART Presentation Support",
            "prepared",
            "specific, measurable, achievable, relevant, time-boxed demo plan",
            ("docs/CAPSTONE_DEMO_SHOWCASE.md",),
            "SMART support is planning guidance for the speaker.",
        ),
        (
            "Ethical AI Summary",
            "prepared",
            "ethical AI and limitations summary",
            ("docs/CAPSTONE_EVALUATION_ETHICS.md",),
            "Ethics summary is explicit disclosure; it does not add safety enforcement behavior.",
        ),
        (
            "Demo Fallback Mode",
            "prepared",
            "provider, retrieval, preview, network, and timing fallback runbook",
            ("demo/manual_demo_checklist.md", "docs/CAPSTONE_DEMO_SHOWCASE.md"),
            "Fallback mode is an operator runbook, not automatic failover logic.",
        ),
        (
            "Presentation Polish",
            "prepared",
            "10-minute demo and 5-minute Q&A script",
            ("docs/CAPSTONE_DEMO_SHOWCASE.md",),
            "Polish is documentation only; V8.8 does not create final slide assets.",
        ),
        (
            "Manual Demo Checklist",
            "prepared",
            "manual rehearsal checklist",
            ("demo/manual_demo_checklist.md",),
            "Checklist requires human rehearsal before final showcase.",
        ),
        (
            "Demo Reliability Validation",
            "prepared",
            "manual reliability checks and fallback criteria",
            ("demo/manual_demo_checklist.md",),
            "Reliability validation is manual and focused; V8.8 does not run long CI.",
        ),
        (
            "Golden Demo Dataset",
            "prepared",
            "tracked golden demo dataset",
            ("demo/golden_demo_dataset.json",),
            "Dataset is a deterministic demo aid, not recorded live evaluation output.",
        ),
        (
            "Offline Demo Fallback",
            "prepared",
            "offline narrative, screenshots, prompts, and dataset",
            ("demo/golden_demo_dataset.json", "assets/preview_current.png"),
            "Offline fallback is manual; it does not simulate live provider success.",
        ),
        (
            "Provider Failure Recovery",
            "prepared",
            "provider failure audience framing and recovery steps",
            ("demo/manual_demo_checklist.md",),
            "Recovery is a scripted operator transition, not automatic provider failover.",
        ),
        (
            "Demo Metrics Dashboard",
            "supported_existing",
            "read-only metrics summary over existing demo/eval readiness surfaces",
            ("docs/CAPSTONE_EVALUATION_ETHICS.md",),
            "Metrics are summarized from existing surfaces and manual eval docs; no live dashboard is added.",
        ),
    )
    return tuple(
        DemoShowcaseCoverageRecord(
            roadmap_item=item,
            status=status,
            implementation_surface=surface,
            evidence_refs=refs,
            claim_boundary=boundary,
        )
        for item, status, surface, refs, boundary in rows
    )


def _case_alignments() -> tuple[CapstoneCaseAlignment, ...]:
    return (
        CapstoneCaseAlignment(
            case_id="case_1_rag_knowledge_assistant",
            case_label="Case 1: RAG-powered knowledge assistant",
            alignment_status="primary",
            demo_claim=(
                "CCA retrieves and frames creative-coding knowledge for runtime, "
                "shader, and audio-visual guidance."
            ),
            evidence_refs=("src/creative_coding_assistant/eval/retrieval_demo_pack.py", "docs/eval_pipeline.md"),
            boundary="The demo should claim source-grounded creative guidance, not complete coverage of all documents.",
        ),
        CapstoneCaseAlignment(
            case_id="case_2_bounded_agent_automation",
            case_label="Case 2: Agent automation",
            alignment_status="guarded_support",
            demo_claim="CCA can explain bounded workflow stages and advisory routing/validation metadata.",
            evidence_refs=("architecture/workflow_graph.md", "README.md"),
            boundary="Do not claim autonomous agent swarms or unattended workflow control.",
        ),
        CapstoneCaseAlignment(
            case_id="case_3_source_grounded_search",
            case_label="Case 3: Smart document search",
            alignment_status="guarded_support",
            demo_claim="CCA demonstrates registered-source KB search for creative-coding references.",
            evidence_refs=("src/creative_coding_assistant/eval/retrieval_demo_pack.py", "docs/sync.md"),
            boundary="Do not claim generic document search outside the registered KB and indexed local sources.",
        ),
        CapstoneCaseAlignment(
            case_id="case_5_ai_coding_assistant",
            case_label="Case 5: AI coding assistant for creative coding",
            alignment_status="primary",
            demo_claim=(
                "CCA translates creative intent into code-oriented browser "
                "runtime guidance and artifact planning."
            ),
            evidence_refs=("README.md", "assets/preview_current.png"),
            boundary="Keep the claim to creative coding assistance, not fully autonomous production delivery.",
        ),
        CapstoneCaseAlignment(
            case_id="case_6_advanced_llm_tools",
            case_label="Case 6: Advanced LLM tools",
            alignment_status="primary",
            demo_claim=(
                "CCA combines LangGraph orchestration, Chroma-backed retrieval, "
                "preview surfaces, and manual eval tooling."
            ),
            evidence_refs=(
                "pyproject.toml",
                "docs/eval_pipeline.md",
                "clients/nextjs/public/preview-sandbox.html",
            ),
            boundary=(
                "Do not claim live DCC/MCP integrations, autonomous immersive "
                "execution, or unsupported preview runtimes."
            ),
        ),
    )


def _prompt_library(*, demo_source: ProductionDemoAssetPlan) -> tuple[DemoPromptRecord, ...]:
    return (
        DemoPromptRecord(
            prompt_id="luminous_audio_reactive_three_scene",
            title="Luminous audio-reactive Three.js capstone scene",
            capstone_cases=(
                "case_1_rag_knowledge_assistant",
                "case_5_ai_coding_assistant",
                "case_6_advanced_llm_tools",
            ),
            prompt_text=demo_source.demo_prompt,
            expected_demo_value="Primary golden flow for prompt to retrieval to artifact to preview to explanation.",
            fallback_notes=(
                "Use assets/preview_current.png if live preview is unavailable.",
                "Use the golden dataset to narrate provider, fallback, and evaluation boundaries.",
            ),
        ),
        DemoPromptRecord(
            prompt_id="concept_to_visual_translation",
            title="Concept-to-visual browser system",
            capstone_cases=("case_1_rag_knowledge_assistant", "case_5_ai_coding_assistant"),
            prompt_text=(
                "Translate the concept of threshold, recursion, and return into a practical browser visual system "
                "with geometry, motion, color, runtime choice, interaction, implementation constraints, and clear "
                "authority boundaries."
            ),
            expected_demo_value="Shows guarded concept-to-visual translation without authority claims.",
            fallback_notes=("Use the retrieval demo scenario with the same id.", "Keep the explanation practical."),
        ),
        DemoPromptRecord(
            prompt_id="creative_debugging_silent_audio",
            title="Debug silent browser audio",
            capstone_cases=(
                "case_1_rag_knowledge_assistant",
                "case_3_source_grounded_search",
                "case_5_ai_coding_assistant",
            ),
            prompt_text=(
                "My browser audio-reactive sketch stays silent until a click and the analyser looks flat. "
                "What should I check first, and how should I explain the browser-audio constraints?"
            ),
            expected_demo_value="Shows source-grounded debugging and ethical transparency about browser constraints.",
            fallback_notes=("Use as a Q&A recovery prompt if the main demo overruns.",),
        ),
        DemoPromptRecord(
            prompt_id="shader_post_fx_pipeline",
            title="Glow-heavy shader and post-processing pipeline",
            capstone_cases=("case_1_rag_knowledge_assistant", "case_5_ai_coding_assistant"),
            prompt_text=(
                "How do I build a glow-heavy kaleidoscopic browser visual with shaders or post-processing passes? "
                "Give source-grounded runtime tradeoffs, failure risks, and fallback implementation options."
            ),
            expected_demo_value="Demonstrates retrieval-backed technical comparison for creative coding.",
            fallback_notes=("Use docs/eval_pipeline.md if asked about retrieval quality evidence.",),
        ),
        DemoPromptRecord(
            prompt_id="offline_showcase_walkthrough",
            title="Offline showcase walkthrough",
            capstone_cases=(
                "case_2_bounded_agent_automation",
                "case_5_ai_coding_assistant",
                "case_6_advanced_llm_tools",
            ),
            prompt_text=(
                "Walk through the prepared Creative Coding Assistant demo offline: problem, solution, data, "
                "evaluation, ethical considerations, fallback plan, challenges, and next steps."
            ),
            expected_demo_value=(
                "Keeps the presentation coherent if provider, retrieval, "
                "preview, or network access fails."
            ),
            fallback_notes=(
                "Use when any live system dependency is unavailable.",
                "Do not imply the offline flow is live.",
            ),
        ),
        DemoPromptRecord(
            prompt_id="p5_generative_morphogenesis_sketch",
            title="p5.js generative morphogenesis sketch",
            capstone_cases=("case_1_rag_knowledge_assistant", "case_5_ai_coding_assistant"),
            prompt_text=(
                "Create a p5.js generative morphogenesis sketch using reaction diffusion, cellular automata, "
                "L-systems, flow fields, particle systems, differential growth, diffusion-limited aggregation, "
                "branching, self-organization, and emergent form. Keep it browser-safe, explain interaction "
                "controls, and cite retrieval/source boundaries conservatively."
            ),
            expected_demo_value="Covers p5.js and morphogenesis as a bounded browser creative-coding path.",
            fallback_notes=("Use this when the primary Three.js path needs a simpler canvas-based fallback.",),
        ),
        DemoPromptRecord(
            prompt_id="hydra_feedback_texture_chain",
            title="Hydra and GLSL feedback texture chain",
            capstone_cases=("case_1_rag_knowledge_assistant", "case_5_ai_coding_assistant"),
            prompt_text=(
                "Design a Hydra-style feedback texture chain and a GLSL fragment-shader fallback for a luminous "
                "kaleidoscopic scene. Include oscillator layers, modulation, feedback, moire-like pattern motion, "
                "and output routing. Explain what is actually supported in-browser, what should be treated as "
                "pseudocode or adaptation guidance, and how to recover if a runtime is unavailable."
            ),
            expected_demo_value="Covers Hydra-if-supported and GLSL with explicit runtime fallback boundaries.",
            fallback_notes=("Use only as bounded runtime guidance if Hydra execution is not available.",),
        ),
        DemoPromptRecord(
            prompt_id="geometry_morphogenesis_visual_system",
            title="Geometry and morphogenesis visual system",
            capstone_cases=("case_1_rag_knowledge_assistant", "case_5_ai_coding_assistant"),
            prompt_text=(
                "Design a geometry and morphogenesis visual system for the browser. Combine radial structures, "
                "recursive growth, reaction diffusion, diffusion-limited aggregation, branching, flow fields, and "
                "particle trails. Include runtime selection, preview strategy, source boundaries, and a graceful "
                "fallback plan."
            ),
            expected_demo_value=(
                "Covers generative structures and emergent form without unsupported authority claims."
            ),
            fallback_notes=("Use for generative-systems Q&A or output-quality review.",),
        ),
        DemoPromptRecord(
            prompt_id="installation_immersive_scene_planning",
            title="Installation and immersive scene planning",
            capstone_cases=(
                "case_2_bounded_agent_automation",
                "case_5_ai_coding_assistant",
                "case_6_advanced_llm_tools",
            ),
            prompt_text=(
                "Plan a browser-based installation or immersive scene for a gallery demo. Include concept, "
                "geometry, audience movement, runtimes, retrieval needs, preview plan, artifact package, "
                "evaluation checks, fallback route, and handoff boundaries."
            ),
            expected_demo_value="Covers bounded planning metadata and handoff guidance.",
            fallback_notes=("Use for senior-reviewer questions about system direction and next steps.",),
        ),
    )


def _golden_demo_flows(*, prompts: tuple[DemoPromptRecord, ...]) -> tuple[GoldenDemoFlow, ...]:
    prompt_ids = {prompt.prompt_id for prompt in prompts}
    if "luminous_audio_reactive_three_scene" not in prompt_ids:
        raise ValueError("primary prompt is required")
    return (
        GoldenDemoFlow(
            flow_id="primary_creative_coding_flow",
            title="Prompt to grounded creative-coding artifact",
            duration_seconds=420,
            primary_prompt_id="luminous_audio_reactive_three_scene",
            capstone_cases=(
                "case_1_rag_knowledge_assistant",
                "case_5_ai_coding_assistant",
                "case_6_advanced_llm_tools",
            ),
            operator_steps=(
                "State the problem and target user.",
                "Run or narrate the golden creative-coding prompt.",
                "Show retrieval grounding and runtime/artifact planning.",
                "Show the preview surface or prepared screenshot.",
                "Explain critique, refinement, fallback, and HITL boundaries.",
            ),
            evidence_refs=("README.md", "demo/golden_demo_dataset.json", "assets/preview_current.png"),
            fallback_trigger="provider_failure",
        ),
        GoldenDemoFlow(
            flow_id="case_alignment_flow",
            title="Capstone case alignment walkthrough",
            duration_seconds=90,
            primary_prompt_id="offline_showcase_walkthrough",
            capstone_cases=_REQUIRED_CASE_IDS,
            operator_steps=(
                "Map the demo to Cases 5, 1, and 6 as primary alignment.",
                "Describe Case 2 as bounded workflow explanation only.",
                "Describe Case 3 as registered-source KB search only.",
            ),
            evidence_refs=("docs/CAPSTONE_DEMO_SHOWCASE.md",),
            fallback_trigger="time_overrun",
        ),
        GoldenDemoFlow(
            flow_id="evaluation_ethics_flow",
            title="Evaluation and ethical AI summary",
            duration_seconds=60,
            primary_prompt_id="shader_post_fx_pipeline",
            capstone_cases=("case_1_rag_knowledge_assistant", "case_6_advanced_llm_tools"),
            operator_steps=(
                "Show retrieval evaluation workflow and known metric boundaries.",
                "Explain no overclaiming, source grounding, cost boundaries, and provider fallback.",
                "Name the known limitations before Q&A.",
            ),
            evidence_refs=("docs/CAPSTONE_EVALUATION_ETHICS.md", "docs/eval_pipeline.md"),
            fallback_trigger="retrieval_unavailable",
        ),
        GoldenDemoFlow(
            flow_id="offline_fallback_flow",
            title="Offline demo fallback",
            duration_seconds=30,
            primary_prompt_id="offline_showcase_walkthrough",
            capstone_cases=("case_5_ai_coding_assistant", "case_6_advanced_llm_tools"),
            operator_steps=(
                "Switch to the golden dataset and screenshots.",
                "Say which live dependency failed.",
                "Continue with the same problem, solution, evaluation, and limitations narrative.",
            ),
            evidence_refs=("demo/golden_demo_dataset.json", "assets/preview_current.png"),
            fallback_trigger="network_unavailable",
        ),
        GoldenDemoFlow(
            flow_id="p5_generative_morphogenesis_flow",
            title="p5.js generative morphogenesis fallback flow",
            duration_seconds=120,
            primary_prompt_id="p5_generative_morphogenesis_sketch",
            capstone_cases=("case_1_rag_knowledge_assistant", "case_5_ai_coding_assistant"),
            operator_steps=(
                "Switch to a canvas-first p5.js prompt.",
                "Show how morphogenesis techniques become practical code guidance.",
                "State that preview/demo assets are fallback evidence, not live generated proof.",
            ),
            evidence_refs=("demo/demo_prompt_library.md", "demo/golden_demo_dataset.json"),
            fallback_trigger="preview_unavailable",
        ),
        GoldenDemoFlow(
            flow_id="hydra_glsl_runtime_flow",
            title="Hydra-if-supported and GLSL runtime fallback flow",
            duration_seconds=120,
            primary_prompt_id="hydra_feedback_texture_chain",
            capstone_cases=("case_1_rag_knowledge_assistant", "case_5_ai_coding_assistant"),
            operator_steps=(
                "Use the runtime prompt to discuss Hydra support honestly.",
                "Show GLSL as the stable browser-shader fallback.",
                "Explain unsupported runtime handling without claiming execution.",
            ),
            evidence_refs=("demo/demo_prompt_library.md", "docs/eval_pipeline.md"),
            fallback_trigger="preview_unavailable",
        ),
        GoldenDemoFlow(
            flow_id="geometry_morphogenesis_flow",
            title="Geometry and morphogenesis visual-system flow",
            duration_seconds=150,
            primary_prompt_id="geometry_morphogenesis_visual_system",
            capstone_cases=("case_1_rag_knowledge_assistant", "case_5_ai_coding_assistant"),
            operator_steps=(
                "Use the geometry/morphogenesis prompt to frame rule-based form.",
                "Show reaction diffusion, flow fields, and particle trails as implementation choices.",
                "Name source, runtime, and performance boundaries.",
            ),
            evidence_refs=("demo/demo_prompt_library.md", "docs/CAPSTONE_EVALUATION_ETHICS.md"),
            fallback_trigger="time_overrun",
        ),
        GoldenDemoFlow(
            flow_id="installation_immersive_scene_planning_flow",
            title="Installation and immersive scene planning flow",
            duration_seconds=180,
            primary_prompt_id="installation_immersive_scene_planning",
            capstone_cases=(
                "case_2_bounded_agent_automation",
                "case_5_ai_coding_assistant",
                "case_6_advanced_llm_tools",
            ),
            operator_steps=(
                "Show concept, runtime, retrieval, preview, artifact, evaluation, and fallback handoff points.",
                "Repeat no external DCC/MCP execution, autonomous delivery, public deployment, or freeze claims.",
                "Use the integrated Demo Mode path as the primary presenter frame.",
            ),
            evidence_refs=("README.md", "demo/golden_demo_dataset.json"),
            fallback_trigger="provider_failure",
        ),
    )


def _demo_metrics(
    *,
    demo_assets: ProductionDemoAssetPlan,
    creative_readiness: ProductionCreativeReadinessReview,
    retrieval_demo_pack: RetrievalDemoPack,
) -> tuple[DemoMetricSummary, ...]:
    return (
        DemoMetricSummary(
            metric_id="retrieval_demo_scenarios",
            label="Retrieval demo scenarios",
            status="ready",
            value=str(len(retrieval_demo_pack.scenarios)),
            source_refs=("src/creative_coding_assistant/eval/retrieval_demo_pack.py",),
            interpretation="Registered scenario count for source-grounded creative-coding retrieval demos.",
        ),
        DemoMetricSummary(
            metric_id="demo_asset_readiness",
            label="Demo asset readiness",
            status=demo_assets.demo_asset_status,
            value=f"{len(demo_assets.ready_asset_ids)}/{demo_assets.asset_count} ready",
            source_refs=("src/creative_coding_assistant/orchestration/advisory/production_demo_assets.py",),
            interpretation="Existing V5.6 demo asset inventory is used as V8.8 evidence.",
        ),
        DemoMetricSummary(
            metric_id="creative_readiness_status",
            label="Creative readiness review",
            status=creative_readiness.creative_readiness_status,
            value=creative_readiness.creative_readiness_status,
            source_refs=(
                "src/creative_coding_assistant/orchestration/advisory/production_creative_readiness_review.py",
            ),
            interpretation="Guarded status is acceptable because generated-output scoring remains manual.",
        ),
        DemoMetricSummary(
            metric_id="preview_media_inventory",
            label="Preview media inventory",
            status="ready" if demo_assets.preview_media_paths else "guarded",
            value=str(len(demo_assets.preview_media_paths)),
            source_refs=demo_assets.preview_media_paths,
            interpretation="Prepared screenshots support fallback showcase when live preview is unavailable.",
        ),
        DemoMetricSummary(
            metric_id="ragas_context_precision_workflow",
            label="RAGAs context precision workflow",
            status="manual",
            value="supported_manual_eval",
            source_refs=("docs/eval_pipeline.md", "scripts/eval_live_sessions.py"),
            interpretation="RAGAs context precision is supported for recorded samples but is not rerun by V8.8.",
        ),
    )


def _fallback_plans() -> tuple[DemoFallbackPlan, ...]:
    return (
        DemoFallbackPlan(
            trigger="provider_failure",
            fallback_mode="Golden dataset plus prepared screenshots",
            operator_action=(
                "State the provider failure and continue with the prevalidated "
                "prompt, dataset, and preview image."
            ),
            audience_framing=(
                "The fallback demonstrates demo reliability without pretending "
                "a provider response happened."
            ),
            evidence_refs=("demo/golden_demo_dataset.json", "assets/preview_current.png"),
        ),
        DemoFallbackPlan(
            trigger="retrieval_unavailable",
            fallback_mode="Source-grounding narrative",
            operator_action=(
                "Show the retrieval demo pack, scenario ids, and eval workflow "
                "without running retrieval live."
            ),
            audience_framing=(
                "The system supports source-grounded retrieval, while this "
                "fallback avoids unverifiable live claims."
            ),
            evidence_refs=("src/creative_coding_assistant/eval/retrieval_demo_pack.py", "docs/eval_pipeline.md"),
        ),
        DemoFallbackPlan(
            trigger="preview_unavailable",
            fallback_mode="Static internal preview showcase",
            operator_action="Use assets/preview_current.png and explain the preview boundary.",
            audience_framing=(
                "The screenshot is a prepared visual reference, not a claim "
                "that the preview just rendered live."
            ),
            evidence_refs=("assets/preview_current.png", "docs/CAPSTONE_DEMO_SHOWCASE.md"),
        ),
        DemoFallbackPlan(
            trigger="network_unavailable",
            fallback_mode="Offline narrative and Q&A mode",
            operator_action=(
                "Use docs, dataset, and screenshots; skip live provider, "
                "retrieval, and preview operations."
            ),
            audience_framing="Offline mode preserves the capstone explanation and makes infrastructure risk explicit.",
            evidence_refs=("demo/README.md", "demo/golden_demo_dataset.json"),
        ),
        DemoFallbackPlan(
            trigger="time_overrun",
            fallback_mode="Short-path demo script",
            operator_action="Jump from primary flow to evaluation/ethics summary and Q&A.",
            audience_framing=(
                "The shortened path preserves the required problem, solution, "
                "data, evaluation, and next steps."
            ),
            evidence_refs=("docs/CAPSTONE_DEMO_SHOWCASE.md",),
        ),
    )


def _checklist_items() -> tuple[DemoChecklistItem, ...]:
    return (
        DemoChecklistItem(
            item_id="confirm_branch",
            category="manual_demo",
            status="ready",
            action="Confirm the demo branch is feature/demo-showcase and no main-branch work is happening.",
            evidence_refs=("demo/manual_demo_checklist.md",),
        ),
        DemoChecklistItem(
            item_id="rehearse_primary_flow",
            category="manual_demo",
            status="manual",
            action="Rehearse the primary creative-coding flow end to end with a 7-minute target.",
            evidence_refs=("docs/CAPSTONE_DEMO_SHOWCASE.md",),
        ),
        DemoChecklistItem(
            item_id="verify_preview_assets",
            category="reliability",
            status="ready",
            action="Confirm assets/preview_current.png, preview_v1.png, and preview_v2.png are available.",
            evidence_refs=("assets/preview_current.png", "assets/preview_v1.png", "assets/preview_v2.png"),
        ),
        DemoChecklistItem(
            item_id="verify_prompt_library",
            category="reliability",
            status="ready",
            action="Keep the demo prompt library open for provider or timing fallback.",
            evidence_refs=("demo/demo_prompt_library.md",),
        ),
        DemoChecklistItem(
            item_id="prepare_eval_summary",
            category="presentation",
            status="ready",
            action="Use the evaluation summary to distinguish supported metrics from manual evidence.",
            evidence_refs=("docs/CAPSTONE_EVALUATION_ETHICS.md", "docs/eval_pipeline.md"),
        ),
        DemoChecklistItem(
            item_id="prepare_ethics_summary",
            category="ethics",
            status="ready",
            action="State source grounding, cost, privacy, provider, and limitation boundaries explicitly.",
            evidence_refs=("docs/CAPSTONE_EVALUATION_ETHICS.md",),
        ),
        DemoChecklistItem(
            item_id="prepare_showcase_upload",
            category="showcase_upload",
            status="manual",
            action="Collect README link, demo docs, screenshots, evaluation summary, and known limitations.",
            evidence_refs=("demo/showcase_upload_preparation.md",),
        ),
        DemoChecklistItem(
            item_id="check_no_overclaims",
            category="presentation",
            status="ready",
            action=(
                "Avoid live DCC/MCP, autonomous execution platform, autonomous "
                "agent swarm, and generic search claims."
            ),
            evidence_refs=("docs/CAPSTONE_DEMO_SHOWCASE.md", "README.md"),
        ),
        DemoChecklistItem(
            item_id="run_fallback_rehearsal",
            category="reliability",
            status="manual",
            action="Practice provider, retrieval, preview, network, and time-overrun fallback transitions.",
            evidence_refs=("demo/manual_demo_checklist.md",),
        ),
        DemoChecklistItem(
            item_id="hitl_before_public_showcase",
            category="showcase_upload",
            status="blocked_until_hitl",
            action="Stop for HITL review before merge, push, tag, freeze, or public showcase upload.",
            evidence_refs=(
                "docs/CAPSTONE_DEMO_SHOWCASE.md",
                "demo/showcase_upload_preparation.md",
            ),
        ),
    )


def _presentation_segments() -> tuple[DemoPresentationSegment, ...]:
    return (
        DemoPresentationSegment(
            segment_id="problem_purpose",
            phase="ten_minute_demo",
            duration_seconds=75,
            purpose="Explain project purpose, target user, and creative-coding problem.",
            talking_points=(
                "Creative intent is hard to translate into working browser visuals.",
                "CCA focuses on creative coding.",
            ),
        ),
        DemoPresentationSegment(
            segment_id="solution_architecture",
            phase="ten_minute_demo",
            duration_seconds=90,
            purpose="Explain the solution architecture at demo depth.",
            talking_points=(
                "LangGraph backend",
                "Chroma-backed KB",
                "Next.js workstation",
                "preview and eval surfaces",
            ),
        ),
        DemoPresentationSegment(
            segment_id="primary_live_flow",
            phase="ten_minute_demo",
            duration_seconds=255,
            purpose="Run or narrate the golden prompt-to-artifact flow.",
            talking_points=("prompt", "retrieval", "planning", "artifact", "preview", "critique/refinement"),
        ),
        DemoPresentationSegment(
            segment_id="evaluation_ethics",
            phase="ten_minute_demo",
            duration_seconds=90,
            purpose="Show evaluation evidence and ethical AI boundaries.",
            talking_points=("manual RAGAs workflow", "readiness metrics", "source grounding", "limitations"),
        ),
        DemoPresentationSegment(
            segment_id="challenges_next_steps",
            phase="ten_minute_demo",
            duration_seconds=90,
            purpose="Close with challenges, fallback, and next steps.",
            talking_points=("demo reliability", "known limitations", "Grand Review next", "post-capstone hardening"),
        ),
        DemoPresentationSegment(
            segment_id="qa_data_evaluation",
            phase="five_minute_qa",
            duration_seconds=120,
            purpose="Answer data, retrieval, and evaluation questions.",
            talking_points=("registered sources", "local eval records", "context precision boundary"),
        ),
        DemoPresentationSegment(
            segment_id="qa_architecture_limits",
            phase="five_minute_qa",
            duration_seconds=120,
            purpose="Answer architecture, provider, and limitation questions.",
            talking_points=("provider fallback", "no live DCC/MCP", "no autonomous execution platform", "manual HITL"),
        ),
        DemoPresentationSegment(
            segment_id="qa_next_steps",
            phase="five_minute_qa",
            duration_seconds=60,
            purpose="Answer next-step and showcase-readiness questions.",
            talking_points=("V8 Grand Review", "manual rehearsal", "public claim hardening"),
        ),
    )
