"""Typed contracts for the V8.7 HoloGenesis Creative Operating System."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

V8_7_CAPABILITY_ID = "v8_7_hologenesis_core"
V8_7_HOLOGENESIS_SCOPE = (
    "Unify V8.1 creative knowledge, V8.2 symbolic translation, V8.3 geometry, "
    "V8.4 architecture, V8.5 narrative, and V8.6 immersive composition into "
    "bounded creative operating-system reports, project planning, curatorial "
    "reasoning, recommendation, readiness scoring, and project bundle generation."
)
V8_7_AUTHORITY_BOUNDARY = (
    "V8.7 HoloGenesis Creative Operating System is a deterministic report and "
    "planning layer. It composes existing V8.1-V8.6 engines without live external "
    "DCC execution, MCP tool execution, provider or model routing, workflow "
    "control, storage writes, frontend expansion, HoloMind implementation, "
    "HOLOiVERSE implementation, or V8.8 demo showcase behavior."
)


class HoloGenesisGraphKind(StrEnum):
    SYMBOLIC = "symbolic"
    SACRED_KNOWLEDGE = "sacred_knowledge"
    GEOMETRY = "geometry"
    NARRATIVE = "narrative"
    INSTALLATION = "installation"


class HoloGenesisRoadmapClassification(StrEnum):
    IMPLEMENTED_REPORT_BEHAVIOR = "implemented_report_behavior"
    REUSED_EXISTING_REPORT = "reused_existing_report"
    EXPORT_PLANNING_ONLY = "export_planning_only"
    ADVISORY_ONLY = "advisory_only"
    FUTURE_HOOK_ONLY = "future_hook_only"
    OUT_OF_SCOPE_UNSUPPORTED = "out_of_scope_unsupported"
    MISSING = "missing"


class HoloGenesisConfidenceBand(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    GUARDED = "guarded"


class HoloGenesisValidationSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    HITL_REQUIRED = "hitl_required"


class HoloGenesisReadinessBand(StrEnum):
    READY_FOR_REVIEW = "ready_for_review"
    PARTIAL = "partial"
    GUARDED = "guarded"


class HoloGenesisGraphNode(BaseModel):
    """One unified graph node sourced from existing V8 reports."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    node_id: str = Field(min_length=1, max_length=180)
    label: str = Field(min_length=1, max_length=220)
    source_engine_ids: tuple[str, ...] = Field(min_length=1, max_length=10)
    summary: str = Field(min_length=1, max_length=760)
    recommendations: tuple[str, ...] = Field(min_length=1, max_length=8)
    confidence_signal: float = Field(ge=0, le=1)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)


class HoloGenesisGraphEdge(BaseModel):
    """Relationship inside a unified graph."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    edge_id: str = Field(min_length=1, max_length=180)
    from_node_id: str = Field(min_length=1, max_length=180)
    to_node_id: str = Field(min_length=1, max_length=180)
    relationship: str = Field(min_length=1, max_length=240)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def _edge_not_self_referential(self) -> Self:
        if self.from_node_id == self.to_node_id:
            raise ValueError("HoloGenesis graph edges cannot point to themselves")
        return self


class HoloGenesisUnifiedGraph(BaseModel):
    """One V8.7 unified graph projection."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    graph_id: str = Field(min_length=1, max_length=160)
    kind: HoloGenesisGraphKind
    source_engine_ids: tuple[str, ...] = Field(min_length=1, max_length=10)
    nodes: tuple[HoloGenesisGraphNode, ...] = Field(min_length=1, max_length=18)
    edges: tuple[HoloGenesisGraphEdge, ...] = Field(default_factory=tuple, max_length=24)
    synthesis_summary: str = Field(min_length=1, max_length=860)
    report_only: Literal[True] = True
    graph_storage_write_implemented: Literal[False] = False
    graph_execution_implemented: Literal[False] = False

    @model_validator(mode="after")
    def _edges_reference_nodes(self) -> Self:
        node_ids = {node.node_id for node in self.nodes}
        for edge in self.edges:
            if edge.from_node_id not in node_ids or edge.to_node_id not in node_ids:
                raise ValueError("HoloGenesis graph edges must reference known nodes")
        return self


class HoloGenesisBlackboardEntry(BaseModel):
    """Report-only creative blackboard entry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    entry_id: str = Field(min_length=1, max_length=180)
    channel: Literal[
        "knowledge_unification",
        "symbolic_geometry",
        "narrative_installation",
        "curatorial_quality",
        "delivery_bundle",
    ]
    summary: str = Field(min_length=1, max_length=620)
    source_graph_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    decision_pressure: Literal["low", "medium", "high"]
    recommended_action: str = Field(min_length=1, max_length=520)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=10)
    persisted: Literal[False] = False


class HoloGenesisScheduleStep(BaseModel):
    """Symbolic schedule step for planning order only."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    step_id: str = Field(min_length=1, max_length=150)
    sequence: int = Field(ge=1, le=12)
    focus: str = Field(min_length=1, max_length=240)
    source_entry_ids: tuple[str, ...] = Field(min_length=1, max_length=6)
    rationale: str = Field(min_length=1, max_length=620)
    output_contract: str = Field(min_length=1, max_length=420)
    execution_implemented: Literal[False] = False


class HoloGenesisPlannerStage(BaseModel):
    """Creative planner stage produced by V8.7."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    stage_id: str = Field(min_length=1, max_length=160)
    title: str = Field(min_length=1, max_length=220)
    objective: str = Field(min_length=1, max_length=620)
    recommended_outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    dependencies: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    confidence_signal: float = Field(ge=0, le=1)


class HoloGenesisRouteRecommendation(BaseModel):
    """Advisory creative route, not provider/model routing."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    route_id: str = Field(min_length=1, max_length=160)
    route_type: Literal[
        "browser_internal",
        "curatorial_review",
        "external_export_planning",
        "research_followup",
    ]
    recommendation: str = Field(min_length=1, max_length=620)
    rationale: str = Field(min_length=1, max_length=620)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=1, max_length=10)


class HoloGenesisDecision(BaseModel):
    """Explainable artistic or curatorial decision."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    decision_id: str = Field(min_length=1, max_length=180)
    decision: str = Field(min_length=1, max_length=720)
    rationale: str = Field(min_length=1, max_length=860)
    source_graph_kinds: tuple[HoloGenesisGraphKind, ...] = Field(min_length=1, max_length=5)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=12)
    confidence_signal: float = Field(ge=0, le=1)


class HoloGenesisCuratorialAssessment(BaseModel):
    """Curatorial reasoning, validation, and explainability assessment."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    assessment_id: str = Field(min_length=1, max_length=180)
    engine: Literal[
        "curatorial_intelligence",
        "curatorial_reasoning",
        "curatorial_validation",
        "curatorial_explainability",
        "mystical_consistency",
        "symbolic_explainability",
        "aesthetic_evaluation",
    ]
    status: Literal["pass", "guarded", "review_required"]
    summary: str = Field(min_length=1, max_length=760)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=12)
    confidence_signal: float = Field(ge=0, le=1)
    bounded_framing: str = Field(min_length=1, max_length=520)


class HoloGenesisReadinessScore(BaseModel):
    """Bounded readiness score for installation and exhibition planning."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    score_id: str = Field(min_length=1, max_length=160)
    label: Literal[
        "installation_quality",
        "museum_readiness",
        "international_exhibition_readiness",
        "project_bundle_readiness",
    ]
    score: int = Field(ge=0, le=100)
    band: HoloGenesisReadinessBand
    rationale: str = Field(min_length=1, max_length=720)
    review_notes: tuple[str, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def _band_matches_score(self) -> Self:
        if self.band != hologenesis_readiness_band(self.score):
            raise ValueError("readiness band must match score")
        return self


class HoloGenesisExternalIntegrationAudit(BaseModel):
    """Reality check for external creative-tool integrations."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    integration_id: str = Field(min_length=1, max_length=120)
    label: Literal["Unity", "Unreal", "TouchDesigner", "Blender", "Houdini", "MCP Creative Tool Layer"]
    classification: HoloGenesisRoadmapClassification
    supported_behavior: str = Field(min_length=1, max_length=620)
    unsupported_behavior: str = Field(min_length=1, max_length=620)
    export_planning_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    live_execution_supported: Literal[False] = False


class HoloGenesisProjectBundle(BaseModel):
    """Generated project bundle outline, not filesystem output."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    project_title: str = Field(min_length=1, max_length=180)
    project_summary: str = Field(min_length=1, max_length=820)
    architecture_outline: tuple[str, ...] = Field(min_length=1, max_length=10)
    portfolio_outline: tuple[str, ...] = Field(min_length=1, max_length=10)
    readme_outline: tuple[str, ...] = Field(min_length=1, max_length=10)
    capstone_outputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    pipeline_steps: tuple[str, ...] = Field(min_length=1, max_length=10)
    workflow_recommendations: tuple[str, ...] = Field(min_length=1, max_length=10)
    tool_recommendations: tuple[str, ...] = Field(min_length=1, max_length=10)
    tech_stack_recommendations: tuple[str, ...] = Field(min_length=1, max_length=10)
    tradeoff_analysis: tuple[str, ...] = Field(min_length=1, max_length=10)
    research_mode_plan: tuple[str, ...] = Field(min_length=1, max_length=10)
    reference_discovery_queries: tuple[str, ...] = Field(min_length=1, max_length=10)
    bundle_file_writes_implemented: Literal[False] = False


class HoloGenesisValidationFinding(BaseModel):
    """Deterministic validation finding for V8.7."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    finding_id: str = Field(min_length=1, max_length=180)
    severity: HoloGenesisValidationSeverity
    summary: str = Field(min_length=1, max_length=620)
    action: str = Field(min_length=1, max_length=620)


class HoloGenesisRoadmapItemAssessment(BaseModel):
    """Reality-check classification for one V8.7 roadmap item."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    item: str = Field(min_length=1)
    classification: HoloGenesisRoadmapClassification
    rationale: str = Field(min_length=1, max_length=760)
    action_required_before_hitl: bool = False
    hitl_required: bool = False


class HoloGenesisConfidence(BaseModel):
    """Confidence posture for a V8.7 HoloGenesis report."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    score: float = Field(ge=0, le=1)
    band: HoloGenesisConfidenceBand
    graph_count: int = Field(ge=0)
    decision_count: int = Field(ge=0)
    readiness_score_count: int = Field(ge=0)
    reused_engine_ids: tuple[str, ...] = Field(min_length=1, max_length=18)
    caveats: tuple[str, ...] = Field(default_factory=tuple, max_length=12)

    @model_validator(mode="after")
    def _band_matches_score(self) -> Self:
        if self.band != hologenesis_confidence_band(self.score, guarded=bool(self.caveats)):
            raise ValueError("confidence band must match score and caveat posture")
        return self


class HoloGenesisReport(BaseModel):
    """Top-level V8.7 HoloGenesis Creative Operating System report."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    capability_id: Literal["v8_7_hologenesis_core"] = V8_7_CAPABILITY_ID
    os_scope: str = Field(default=V8_7_HOLOGENESIS_SCOPE, min_length=1)
    authority_boundary: str = Field(default=V8_7_AUTHORITY_BOUNDARY, min_length=1)
    source_query: str = Field(min_length=1, max_length=920)
    reused_surface_ids: tuple[str, ...] = Field(min_length=1, max_length=32)
    composition_audit_summary: tuple[str, ...] = Field(min_length=6, max_length=14)
    unified_graphs: tuple[HoloGenesisUnifiedGraph, ...] = Field(min_length=5, max_length=5)
    blackboard_entries: tuple[HoloGenesisBlackboardEntry, ...] = Field(min_length=5, max_length=8)
    symbolic_schedule: tuple[HoloGenesisScheduleStep, ...] = Field(min_length=4, max_length=8)
    creative_plan: tuple[HoloGenesisPlannerStage, ...] = Field(min_length=4, max_length=8)
    route_recommendations: tuple[HoloGenesisRouteRecommendation, ...] = Field(min_length=4, max_length=8)
    artistic_decisions: tuple[HoloGenesisDecision, ...] = Field(min_length=4, max_length=10)
    curatorial_assessments: tuple[HoloGenesisCuratorialAssessment, ...] = Field(min_length=7, max_length=10)
    readiness_scores: tuple[HoloGenesisReadinessScore, ...] = Field(min_length=4, max_length=6)
    external_integration_audit: tuple[HoloGenesisExternalIntegrationAudit, ...] = Field(min_length=6, max_length=6)
    project_bundle: HoloGenesisProjectBundle
    validation_findings: tuple[HoloGenesisValidationFinding, ...] = Field(min_length=1, max_length=14)
    confidence: HoloGenesisConfidence
    roadmap_assessment: tuple[HoloGenesisRoadmapItemAssessment, ...] = Field(min_length=1, max_length=52)
    implemented_report_items: tuple[str, ...] = Field(default_factory=tuple)
    reused_existing_report_items: tuple[str, ...] = Field(default_factory=tuple)
    export_planning_only_items: tuple[str, ...] = Field(default_factory=tuple)
    advisory_only_items: tuple[str, ...] = Field(default_factory=tuple)
    future_hook_only_items: tuple[str, ...] = Field(default_factory=tuple)
    out_of_scope_unsupported_items: tuple[str, ...] = Field(default_factory=tuple)
    missing_items: tuple[str, ...] = Field(default_factory=tuple)
    unsupported_claim_risks: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    hitl_questions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    unified_graphs_implemented: Literal[True] = True
    creative_blackboard_implemented: Literal[True] = True
    symbolic_scheduler_implemented: Literal[True] = True
    creative_planner_implemented: Literal[True] = True
    creative_router_implemented: Literal[True] = True
    artistic_decision_engine_implemented: Literal[True] = True
    curatorial_engines_implemented: Literal[True] = True
    installation_simulation_report_implemented: Literal[True] = True
    project_bundle_generator_implemented: Literal[True] = True
    external_integration_export_planning_implemented: Literal[True] = True
    external_dcc_execution_implemented: Literal[False] = False
    mcp_tool_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    storage_write_implemented: Literal[False] = False
    frontend_ui_implemented: Literal[False] = False
    holomind_implemented: Literal[False] = False
    holoiverse_implemented: Literal[False] = False
    v8_8_demo_showcase_started: Literal[False] = False

    @model_validator(mode="after")
    def _report_matches_contract(self) -> Self:
        graph_kinds = {graph.kind for graph in self.unified_graphs}
        if graph_kinds != set(HoloGenesisGraphKind):
            raise ValueError("V8.7 must include the five required unified graph kinds")
        if len({entry.entry_id for entry in self.blackboard_entries}) != len(self.blackboard_entries):
            raise ValueError("blackboard entry ids must be unique")
        if len({step.step_id for step in self.symbolic_schedule}) != len(self.symbolic_schedule):
            raise ValueError("schedule step ids must be unique")
        if tuple(step.sequence for step in self.symbolic_schedule) != tuple(
            sorted(step.sequence for step in self.symbolic_schedule)
        ):
            raise ValueError("symbolic schedule must be ordered by sequence")
        if len({stage.stage_id for stage in self.creative_plan}) != len(self.creative_plan):
            raise ValueError("planner stage ids must be unique")
        if len({route.route_id for route in self.route_recommendations}) != len(self.route_recommendations):
            raise ValueError("route recommendation ids must be unique")
        if len({decision.decision_id for decision in self.artistic_decisions}) != len(self.artistic_decisions):
            raise ValueError("artistic decision ids must be unique")
        if len({item.integration_id for item in self.external_integration_audit}) != len(
            self.external_integration_audit
        ):
            raise ValueError("external integration audit ids must be unique")
        classified = hologenesis_items_by_classification(self.roadmap_assessment)
        if self.implemented_report_items != classified[HoloGenesisRoadmapClassification.IMPLEMENTED_REPORT_BEHAVIOR]:
            raise ValueError("implemented report items must match roadmap assessment")
        if self.reused_existing_report_items != classified[HoloGenesisRoadmapClassification.REUSED_EXISTING_REPORT]:
            raise ValueError("reused report items must match roadmap assessment")
        if self.export_planning_only_items != classified[HoloGenesisRoadmapClassification.EXPORT_PLANNING_ONLY]:
            raise ValueError("export-planning items must match roadmap assessment")
        if self.advisory_only_items != classified[HoloGenesisRoadmapClassification.ADVISORY_ONLY]:
            raise ValueError("advisory items must match roadmap assessment")
        if self.future_hook_only_items != classified[HoloGenesisRoadmapClassification.FUTURE_HOOK_ONLY]:
            raise ValueError("future hook items must match roadmap assessment")
        if self.out_of_scope_unsupported_items != classified[HoloGenesisRoadmapClassification.OUT_OF_SCOPE_UNSUPPORTED]:
            raise ValueError("unsupported items must match roadmap assessment")
        if self.missing_items != classified[HoloGenesisRoadmapClassification.MISSING]:
            raise ValueError("missing items must match roadmap assessment")
        return self


def hologenesis_items_by_classification(
    assessments: Sequence[HoloGenesisRoadmapItemAssessment],
) -> dict[HoloGenesisRoadmapClassification, tuple[str, ...]]:
    return {
        classification: tuple(item.item for item in assessments if item.classification == classification)
        for classification in HoloGenesisRoadmapClassification
    }


def hologenesis_confidence_band(
    score: float,
    *,
    guarded: bool,
) -> HoloGenesisConfidenceBand:
    if guarded:
        return HoloGenesisConfidenceBand.GUARDED
    if score >= 0.75:
        return HoloGenesisConfidenceBand.HIGH
    if score >= 0.5:
        return HoloGenesisConfidenceBand.MEDIUM
    return HoloGenesisConfidenceBand.LOW


def hologenesis_readiness_band(score: int) -> HoloGenesisReadinessBand:
    if score >= 72:
        return HoloGenesisReadinessBand.READY_FOR_REVIEW
    if score >= 50:
        return HoloGenesisReadinessBand.PARTIAL
    return HoloGenesisReadinessBand.GUARDED


__all__ = [
    "HoloGenesisBlackboardEntry",
    "HoloGenesisConfidence",
    "HoloGenesisConfidenceBand",
    "HoloGenesisCuratorialAssessment",
    "HoloGenesisDecision",
    "HoloGenesisExternalIntegrationAudit",
    "HoloGenesisGraphEdge",
    "HoloGenesisGraphKind",
    "HoloGenesisGraphNode",
    "HoloGenesisPlannerStage",
    "HoloGenesisProjectBundle",
    "HoloGenesisReadinessBand",
    "HoloGenesisReadinessScore",
    "HoloGenesisReport",
    "HoloGenesisRoadmapClassification",
    "HoloGenesisRoadmapItemAssessment",
    "HoloGenesisRouteRecommendation",
    "HoloGenesisScheduleStep",
    "HoloGenesisUnifiedGraph",
    "HoloGenesisValidationFinding",
    "HoloGenesisValidationSeverity",
    "V8_7_AUTHORITY_BOUNDARY",
    "V8_7_CAPABILITY_ID",
    "V8_7_HOLOGENESIS_SCOPE",
    "hologenesis_confidence_band",
    "hologenesis_items_by_classification",
    "hologenesis_readiness_band",
]
