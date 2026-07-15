"""Product demo experience and fallback-readiness metadata."""

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
    "Demo experience metadata describes prompts, product flows, scenario "
    "coverage, evaluation summaries, fallbacks, and validation checklists over "
    "existing product surfaces only; it does not execute providers, run "
    "retrieval, render previews, route models, mutate prompts or artifacts, "
    "write persistent storage, call external tools, deploy, or control release "
    "operations."
)

_COVERAGE_ITEMS = (
    "Demo Mode",
    "Primary Product Flows",
    "Scenario Coverage",
    "Preview Fallbacks",
    "Demo Prompt Library",
    "Evaluation Summary",
    "Product Boundary Summary",
    "Artifact Availability Check",
    "Workflow Context",
    "Validation Criteria",
    "Ethical AI Summary",
    "Fallback Guidance",
    "Product Explanation",
    "Manual Validation Checklist",
    "Demo Reliability Validation",
    "Deterministic Demo Dataset",
    "Offline Fallback",
    "Provider Failure Recovery",
    "Evaluation Metrics Summary",
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
)


class DemoShowcaseCoverageRecord(BaseModel):
    """One coverage item mapped to a truthful product surface."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    roadmap_item: str = Field(min_length=1, max_length=120)
    status: DemoShowcaseCoverageStatus
    implementation_surface: str = Field(min_length=1, max_length=240)
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=8)
    claim_boundary: str = Field(min_length=1, max_length=520)
    hitl_review_required: bool = True
    serialization_version: Literal["demo_showcase_record.v1"] = DEMO_SHOWCASE_RECORD_SERIALIZATION_VERSION


class CapstoneCaseAlignment(BaseModel):
    """Compatibility record for one product capability scenario."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    case_id: CapstoneCaseId
    case_label: str = Field(min_length=1, max_length=160)
    alignment_status: Literal["primary", "guarded_support"]
    demo_claim: str = Field(min_length=1, max_length=520)
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=8)
    boundary: str = Field(min_length=1, max_length=520)


class DemoPromptRecord(BaseModel):
    """One prompt in the product demo library."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    prompt_id: str = Field(pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    title: str = Field(min_length=1, max_length=160)
    capstone_cases: tuple[CapstoneCaseId, ...] = Field(min_length=1, max_length=5)
    prompt_text: str = Field(min_length=1, max_length=1400)
    expected_demo_value: str = Field(min_length=1, max_length=420)
    fallback_notes: tuple[str, ...] = Field(min_length=1, max_length=6)


class GoldenDemoFlow(BaseModel):
    """One bounded product demo flow."""

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
    """Fallback guidance for a known demo risk."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    trigger: DemoShowcaseFallbackTrigger
    fallback_mode: str = Field(min_length=1, max_length=160)
    operator_action: str = Field(min_length=1, max_length=520)
    audience_framing: str = Field(min_length=1, max_length=520)
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=8)


class DemoChecklistItem(BaseModel):
    """Manual product-validation checklist item."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    item_id: str = Field(pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    category: Literal[
        "manual_demo",
        "reliability",
        "artifact_readiness",
        "product_guidance",
        "ethics",
    ]
    status: DemoShowcaseChecklistStatus
    action: str = Field(min_length=1, max_length=520)
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=8)


class DemoPresentationSegment(BaseModel):
    """Product-validation section retained under the compatibility schema."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    segment_id: str = Field(pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    phase: Literal["product_walkthrough", "validation_notes"]
    duration_seconds: int = Field(ge=30, le=300)
    purpose: str = Field(min_length=1, max_length=260)
    talking_points: tuple[str, ...] = Field(min_length=1, max_length=8)


class DemoShowcasePlan(BaseModel):
    """Demo experience readiness plan over existing product surfaces."""

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
    showcase_upload_prepared: Literal[False] = False
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
        if self.coverage_items != _COVERAGE_ITEMS:
            raise ValueError("coverage_items must cover required product surfaces")
        if tuple(alignment.case_id for alignment in self.capstone_case_alignments) != _REQUIRED_CASE_IDS:
            raise ValueError("scenario alignment must cover required compatibility ids")
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
        phases = {segment.phase for segment in self.presentation_segments}
        if phases != {"product_walkthrough", "validation_notes"}:
            raise ValueError("presentation_segments must cover product walkthrough and validation notes")
        return self


def build_demo_showcase_plan(
    project_root: str | Path | None = None,
    *,
    demo_assets: ProductionDemoAssetPlan | None = None,
    creative_readiness: ProductionCreativeReadinessReview | None = None,
    retrieval_demo_pack: RetrievalDemoPack | None = None,
) -> DemoShowcasePlan:
    """Build demo experience metadata without running the demo."""

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
        coverage_items=_COVERAGE_ITEMS,
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
            "demo/README.md",
            "demo/demo_prompt_library.md",
            "demo/manual_demo_checklist.md",
            "docs/eval_pipeline.md",
            "architecture/demo_showcase_experience.md",
        ),
        dataset_refs=("demo/golden_demo_dataset.json",),
    )


def demo_showcase_coverage_by_item(
    roadmap_item: str,
    plan: DemoShowcasePlan | None = None,
) -> DemoShowcaseCoverageRecord | None:
    """Return one coverage record by its compatibility item name."""

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
            "product flow metadata and deterministic dataset",
            ("demo/README.md", "demo/golden_demo_dataset.json"),
            "Prepared as inspectable guidance, not an automatic workflow controller.",
        ),
        (
            "Primary Product Flows",
            "prepared",
            "bounded flow records and product documentation",
            ("demo/README.md", "demo/golden_demo_dataset.json"),
            "Flows describe existing product behavior and do not execute the assistant.",
        ),
        (
            "Scenario Coverage",
            "prepared",
            "capability coverage across retrieval, workflow, search, coding, and tooling",
            ("README.md", "src/creative_coding_assistant/eval/retrieval_demo_pack.py"),
            "Workflow and search claims stay within existing bounded product behavior.",
        ),
        (
            "Preview Fallbacks",
            "supported_existing",
            "archived preview references and product preview surfaces",
            (
                "assets/screenshots-archive/preview_current.png",
                "src/creative_coding_assistant/preview/contracts.py",
            ),
            "Archived images are fallback references; this metadata does not render or repair previews.",
        ),
        (
            "Demo Prompt Library",
            "prepared",
            "prompt library records and markdown prompt bank",
            ("demo/demo_prompt_library.md",),
            "Prompt records do not mutate product prompts or provider selection.",
        ),
        (
            "Evaluation Summary",
            "supported_existing",
            "evaluation documentation, CLI, and readiness metrics summary",
            ("docs/eval_pipeline.md", "scripts/eval_live_sessions.py"),
            "The summary references supported evaluation paths and does not collect live metrics.",
        ),
        (
            "Product Boundary Summary",
            "prepared",
            "README product capabilities and claim boundaries",
            ("README.md",),
            "The README remains the primary public product boundary.",
        ),
        (
            "Artifact Availability Check",
            "prepared",
            "tracked demo references and archived fallback images",
            ("demo/README.md", "assets/screenshots-archive/preview_current.png"),
            "Availability checks inspect tracked references and do not publish artifacts.",
        ),
        (
            "Workflow Context",
            "prepared",
            "bounded workflow stages and execution boundaries",
            ("architecture/workflow_graph.md", "README.md"),
            "Workflow context explains existing behavior without controlling execution.",
        ),
        (
            "Validation Criteria",
            "prepared",
            "source, runtime, fallback, and evaluation checks",
            ("demo/manual_demo_checklist.md", "docs/eval_pipeline.md"),
            "Validation criteria are guidance and do not alter runtime state.",
        ),
        (
            "Ethical AI Summary",
            "prepared",
            "ethical AI and limitations summary",
            ("README.md", "docs/eval_pipeline.md"),
            "Ethics summary is explicit disclosure; it does not add safety enforcement behavior.",
        ),
        (
            "Fallback Guidance",
            "prepared",
            "provider, retrieval, preview, network, and shortened-path guidance",
            ("demo/manual_demo_checklist.md", "demo/README.md"),
            "Fallback guidance is descriptive and does not implement automatic failover.",
        ),
        (
            "Product Explanation",
            "prepared",
            "purpose, architecture, product flow, evaluation, and limitations",
            ("README.md", "architecture/demo_showcase_experience.md"),
            "Explanation metadata does not create media or alter product output.",
        ),
        (
            "Manual Validation Checklist",
            "prepared",
            "manual product-validation checks",
            ("demo/manual_demo_checklist.md",),
            "The checklist keeps validation steps explicit and human-controlled.",
        ),
        (
            "Demo Reliability Validation",
            "prepared",
            "manual reliability checks and fallback criteria",
            ("demo/manual_demo_checklist.md",),
            "Reliability validation is manual and does not run the full test suite.",
        ),
        (
            "Deterministic Demo Dataset",
            "prepared",
            "tracked deterministic demo dataset",
            ("demo/golden_demo_dataset.json",),
            "The dataset is a deterministic fixture, not recorded live evaluation output.",
        ),
        (
            "Offline Fallback",
            "prepared",
            "offline documentation, screenshots, prompts, and dataset",
            (
                "demo/golden_demo_dataset.json",
                "assets/screenshots-archive/preview_current.png",
            ),
            "Offline fallback is manual; it does not simulate live provider success.",
        ),
        (
            "Provider Failure Recovery",
            "prepared",
            "provider failure explanation and recovery steps",
            ("demo/manual_demo_checklist.md",),
            "Recovery guidance does not implement automatic provider failover.",
        ),
        (
            "Evaluation Metrics Summary",
            "supported_existing",
            "read-only metrics summary over existing evaluation surfaces",
            ("docs/eval_pipeline.md",),
            "Metrics are summarized from existing evidence; no live collection is added.",
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
            case_label="Source-grounded knowledge assistance",
            alignment_status="primary",
            demo_claim=(
                "CCA retrieves and frames creative-coding knowledge for runtime, shader, and audio-visual guidance."
            ),
            evidence_refs=("src/creative_coding_assistant/eval/retrieval_demo_pack.py", "docs/eval_pipeline.md"),
            boundary="Claim source-grounded creative guidance, not complete coverage of all documents.",
        ),
        CapstoneCaseAlignment(
            case_id="case_2_bounded_agent_automation",
            case_label="Bounded workflow guidance",
            alignment_status="guarded_support",
            demo_claim="CCA can explain bounded workflow stages and advisory routing/validation metadata.",
            evidence_refs=("architecture/workflow_graph.md", "README.md"),
            boundary="Do not claim autonomous agent swarms or unattended workflow control.",
        ),
        CapstoneCaseAlignment(
            case_id="case_3_source_grounded_search",
            case_label="Registered-source search",
            alignment_status="guarded_support",
            demo_claim="CCA demonstrates registered-source KB search for creative-coding references.",
            evidence_refs=("src/creative_coding_assistant/eval/retrieval_demo_pack.py", "docs/sync.md"),
            boundary="Do not claim generic document search outside the registered KB and indexed local sources.",
        ),
        CapstoneCaseAlignment(
            case_id="case_5_ai_coding_assistant",
            case_label="Creative-coding assistance",
            alignment_status="primary",
            demo_claim=(
                "CCA translates creative intent into code-oriented browser runtime guidance and artifact planning."
            ),
            evidence_refs=(
                "README.md",
                "assets/screenshots-archive/preview_current.png",
            ),
            boundary="Keep the claim to creative coding assistance, not fully autonomous production delivery.",
        ),
        CapstoneCaseAlignment(
            case_id="case_6_advanced_llm_tools",
            case_label="Bounded orchestration and evaluation tools",
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
            title="Luminous audio-reactive Three.js scene",
            capstone_cases=(
                "case_1_rag_knowledge_assistant",
                "case_5_ai_coding_assistant",
                "case_6_advanced_llm_tools",
            ),
            prompt_text=demo_source.demo_prompt,
            expected_demo_value="Primary golden flow for prompt to retrieval to artifact to preview to explanation.",
            fallback_notes=(
                "Use the archived fallback preview if live preview is unavailable.",
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
            fallback_notes=("Use as a recovery prompt when the primary flow is unavailable.",),
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
            title="Offline product walkthrough",
            capstone_cases=(
                "case_2_bounded_agent_automation",
                "case_5_ai_coding_assistant",
                "case_6_advanced_llm_tools",
            ),
            prompt_text=(
                "Explain the Creative Coding Assistant offline using its problem, solution, data, "
                "evaluation, ethical considerations, fallback plan, limitations, and next steps."
            ),
            expected_demo_value=(
                "Keeps the product explanation coherent if provider, retrieval, preview, or network access fails."
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
            fallback_notes=("Use for generative-systems or output-quality guidance.",),
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
            fallback_notes=("Use when explaining system direction and bounded next steps.",),
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
                "Identify the problem and target user.",
                "Use the prepared creative-coding prompt.",
                "Show retrieval grounding and runtime/artifact planning.",
                "Use the preview surface or archived fallback image.",
                "Explain critique, refinement, fallback, and HITL boundaries.",
            ),
            evidence_refs=(
                "README.md",
                "demo/golden_demo_dataset.json",
                "assets/screenshots-archive/preview_current.png",
            ),
            fallback_trigger="provider_failure",
        ),
        GoldenDemoFlow(
            flow_id="case_alignment_flow",
            title="Product scenario coverage",
            duration_seconds=90,
            primary_prompt_id="offline_showcase_walkthrough",
            capstone_cases=_REQUIRED_CASE_IDS,
            operator_steps=(
                "Connect the product flow to creative coding, retrieval, and evaluation.",
                "Keep workflow claims limited to bounded application stages.",
                "Keep search claims limited to registered and indexed sources.",
            ),
            evidence_refs=("README.md", "architecture/workflow_graph.md"),
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
                "Include known limitations in the product explanation.",
            ),
            evidence_refs=("README.md", "docs/eval_pipeline.md"),
            fallback_trigger="retrieval_unavailable",
        ),
        GoldenDemoFlow(
            flow_id="offline_fallback_flow",
            title="Offline product fallback",
            duration_seconds=30,
            primary_prompt_id="offline_showcase_walkthrough",
            capstone_cases=("case_5_ai_coding_assistant", "case_6_advanced_llm_tools"),
            operator_steps=(
                "Use the deterministic dataset and archived screenshots.",
                "Identify the unavailable live dependency.",
                "Continue with the same problem, solution, evaluation, and limitations.",
            ),
            evidence_refs=(
                "demo/golden_demo_dataset.json",
                "assets/screenshots-archive/preview_current.png",
            ),
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
            evidence_refs=("demo/demo_prompt_library.md", "README.md"),
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
                "Repeat no external tool execution, autonomous delivery, or public deployment claims.",
                "Use the integrated Demo Mode path as the primary product entry point.",
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
            interpretation="The existing demo asset inventory supplies current readiness evidence.",
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
            interpretation="Archived screenshots support fallback guidance when live preview is unavailable.",
        ),
        DemoMetricSummary(
            metric_id="ragas_context_precision_workflow",
            label="RAGAs context precision workflow",
            status="manual",
            value="supported_manual_eval",
            source_refs=("docs/eval_pipeline.md", "scripts/eval_live_sessions.py"),
            interpretation="Context precision is supported for approved evaluation samples but is not rerun here.",
        ),
    )


def _fallback_plans() -> tuple[DemoFallbackPlan, ...]:
    return (
        DemoFallbackPlan(
            trigger="provider_failure",
            fallback_mode="Golden dataset plus prepared screenshots",
            operator_action=(
                "Identify the provider failure and continue with the validated prompt, dataset, and preview image."
            ),
            audience_framing=(
                "The fallback demonstrates product reliability without implying a provider response happened."
            ),
            evidence_refs=(
                "demo/golden_demo_dataset.json",
                "assets/screenshots-archive/preview_current.png",
            ),
        ),
        DemoFallbackPlan(
            trigger="retrieval_unavailable",
            fallback_mode="Source-grounding narrative",
            operator_action=(
                "Show the retrieval demo pack, scenario ids, and eval workflow without running retrieval live."
            ),
            audience_framing=(
                "The system supports source-grounded retrieval, while this fallback avoids unverifiable live claims."
            ),
            evidence_refs=("src/creative_coding_assistant/eval/retrieval_demo_pack.py", "docs/eval_pipeline.md"),
        ),
        DemoFallbackPlan(
            trigger="preview_unavailable",
            fallback_mode="Archived preview reference",
            operator_action=(
                "Load assets/screenshots-archive/preview_current.png and label it "
                "as an archived fallback rather than a live render."
            ),
            audience_framing=(
                "The screenshot is a prepared visual reference, not a claim that the preview just rendered live."
            ),
            evidence_refs=(
                "assets/screenshots-archive/preview_current.png",
                "demo/README.md",
            ),
        ),
        DemoFallbackPlan(
            trigger="network_unavailable",
            fallback_mode="Offline product walkthrough",
            operator_action=(
                "Use docs, dataset, and screenshots; skip live provider, retrieval, and preview operations."
            ),
            audience_framing="Offline mode preserves the product explanation and makes infrastructure risk explicit.",
            evidence_refs=("demo/README.md", "demo/golden_demo_dataset.json"),
        ),
        DemoFallbackPlan(
            trigger="time_overrun",
            fallback_mode="Essential product path",
            operator_action="Use the primary flow followed by the evaluation and limitations summary.",
            audience_framing=(
                "The shortened path preserves the required problem, solution, data, evaluation, and next steps."
            ),
            evidence_refs=("demo/README.md",),
        ),
    )


def _checklist_items() -> tuple[DemoChecklistItem, ...]:
    return (
        DemoChecklistItem(
            item_id="confirm_product_state",
            category="manual_demo",
            status="ready",
            action="Confirm the local product and deterministic fixtures are available.",
            evidence_refs=("demo/manual_demo_checklist.md",),
        ),
        DemoChecklistItem(
            item_id="validate_primary_flow",
            category="manual_demo",
            status="manual",
            action="Validate the primary creative-coding flow and its product boundaries.",
            evidence_refs=("demo/README.md",),
        ),
        DemoChecklistItem(
            item_id="verify_preview_assets",
            category="reliability",
            status="ready",
            action="Confirm all archived fallback preview images are available.",
            evidence_refs=(
                "assets/screenshots-archive/preview_current.png",
                "assets/screenshots-archive/preview_v1.png",
                "assets/screenshots-archive/preview_v2.png",
            ),
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
            category="product_guidance",
            status="ready",
            action="Use the evaluation summary to distinguish supported metrics from manual evidence.",
            evidence_refs=("README.md", "docs/eval_pipeline.md"),
        ),
        DemoChecklistItem(
            item_id="prepare_ethics_summary",
            category="ethics",
            status="ready",
            action="State source grounding, cost, privacy, provider, and limitation boundaries explicitly.",
            evidence_refs=("README.md",),
        ),
        DemoChecklistItem(
            item_id="verify_artifact_refs",
            category="artifact_readiness",
            status="manual",
            action="Confirm that documented prompts, screenshots, and evaluation references exist.",
            evidence_refs=("demo/README.md",),
        ),
        DemoChecklistItem(
            item_id="check_no_overclaims",
            category="product_guidance",
            status="ready",
            action=(
                "Avoid live DCC/MCP, autonomous execution platform, autonomous agent swarm, and generic search claims."
            ),
            evidence_refs=("README.md",),
        ),
        DemoChecklistItem(
            item_id="validate_fallbacks",
            category="reliability",
            status="manual",
            action="Validate provider, retrieval, preview, network, and shortened-path fallbacks.",
            evidence_refs=("demo/manual_demo_checklist.md",),
        ),
        DemoChecklistItem(
            item_id="confirm_manual_boundaries",
            category="artifact_readiness",
            status="blocked_until_hitl",
            action="Require explicit human approval before any external publishing action.",
            evidence_refs=("README.md", "demo/README.md"),
        ),
    )


def _presentation_segments() -> tuple[DemoPresentationSegment, ...]:
    return (
        DemoPresentationSegment(
            segment_id="product_purpose",
            phase="product_walkthrough",
            duration_seconds=60,
            purpose="Summarize product purpose, target user, and creative-coding problem.",
            talking_points=(
                "Creative intent is hard to translate into working browser visuals.",
                "CCA focuses on creative coding.",
            ),
        ),
        DemoPresentationSegment(
            segment_id="system_boundaries",
            phase="product_walkthrough",
            duration_seconds=60,
            purpose="Summarize the implemented architecture and its boundaries.",
            talking_points=(
                "LangGraph backend",
                "Chroma-backed KB",
                "Next.js workstation",
                "preview and eval surfaces",
            ),
        ),
        DemoPresentationSegment(
            segment_id="primary_product_flow",
            phase="product_walkthrough",
            duration_seconds=120,
            purpose="Describe the validated prompt-to-artifact product flow.",
            talking_points=("prompt", "retrieval", "planning", "artifact", "preview", "critique/refinement"),
        ),
        DemoPresentationSegment(
            segment_id="evaluation_boundaries",
            phase="product_walkthrough",
            duration_seconds=60,
            purpose="Summarize evaluation evidence and ethical AI boundaries.",
            talking_points=("manual RAGAs workflow", "readiness metrics", "source grounding", "limitations"),
        ),
        DemoPresentationSegment(
            segment_id="limitations_next_steps",
            phase="product_walkthrough",
            duration_seconds=60,
            purpose="Record known limitations, fallbacks, and bounded next steps.",
            talking_points=("product reliability", "known limitations", "fallbacks", "future validation"),
        ),
        DemoPresentationSegment(
            segment_id="data_retrieval_notes",
            phase="validation_notes",
            duration_seconds=60,
            purpose="Record data, retrieval, and evaluation boundaries.",
            talking_points=("registered sources", "local eval records", "context precision boundary"),
        ),
        DemoPresentationSegment(
            segment_id="runtime_limitations",
            phase="validation_notes",
            duration_seconds=60,
            purpose="Record architecture, provider, and runtime limitations.",
            talking_points=("provider fallback", "no live DCC/MCP", "no autonomous execution platform", "manual HITL"),
        ),
        DemoPresentationSegment(
            segment_id="product_followups",
            phase="validation_notes",
            duration_seconds=60,
            purpose="Record product follow-ups and claim-boundary checks.",
            talking_points=("manual validation", "known limitations", "public claim boundaries"),
        ),
    )
