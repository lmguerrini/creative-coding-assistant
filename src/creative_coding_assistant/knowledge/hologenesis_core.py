"""V8.7 bounded HoloGenesis Creative Operating System builder."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from creative_coding_assistant.contracts import AssistantRequest, CreativeCodingDomain
from creative_coding_assistant.knowledge.creative_distillation import (
    CreativeKnowledgeDistillationReport,
    build_v8_1_creative_knowledge_distillation,
)
from creative_coding_assistant.knowledge.hologenesis_core_catalog import (
    ROADMAP_CLASSIFICATION_ROWS,
)
from creative_coding_assistant.knowledge.hologenesis_core_contracts import (
    HoloGenesisConfidence,
    HoloGenesisReport,
    HoloGenesisRoadmapClassification,
    HoloGenesisRoadmapItemAssessment,
    hologenesis_confidence_band,
    hologenesis_items_by_classification,
)
from creative_coding_assistant.knowledge.hologenesis_core_delivery import (
    build_hologenesis_project_bundle,
    build_hologenesis_validation_findings,
    hologenesis_composition_audit_summary,
)
from creative_coding_assistant.knowledge.hologenesis_core_guidance import (
    build_hologenesis_artistic_decisions,
    build_hologenesis_blackboard_entries,
    build_hologenesis_creative_plan,
    build_hologenesis_curatorial_assessments,
    build_hologenesis_external_integration_audit,
    build_hologenesis_readiness_scores,
    build_hologenesis_route_recommendations,
    build_hologenesis_symbolic_schedule,
    build_hologenesis_unified_graphs,
)
from creative_coding_assistant.knowledge.immersive_audiovisual_composer import (
    build_v8_6_immersive_audiovisual_composer,
)
from creative_coding_assistant.knowledge.immersive_audiovisual_composer_contracts import (
    ImmersiveAudiovisualComposerReport,
)
from creative_coding_assistant.knowledge.mythopoetic_narrative import (
    build_v8_5_mythopoetic_narrative_engine,
)
from creative_coding_assistant.knowledge.mythopoetic_narrative_contracts import (
    MythopoeticNarrativeReport,
)
from creative_coding_assistant.knowledge.sacred_architecture import (
    build_v8_4_sacred_architecture_engine,
)
from creative_coding_assistant.knowledge.sacred_architecture_contracts import (
    SacredArchitectureReport,
)
from creative_coding_assistant.knowledge.sacred_geometry import (
    build_v8_3_sacred_geometry_engine,
)
from creative_coding_assistant.knowledge.sacred_geometry_contracts import SacredGeometryReport
from creative_coding_assistant.knowledge.symbolic_translation import (
    SymbolicTranslationReport,
    build_v8_2_symbolic_translation_engine,
)
from creative_coding_assistant.orchestration.creative_planning import (
    derive_creative_execution_plan,
)
from creative_coding_assistant.orchestration.creative_translation import (
    CreativeTranslation,
    derive_creative_translation,
)


def build_v8_7_hologenesis_core(
    query: str,
    *,
    domains: Sequence[CreativeCodingDomain] = (),
    creative_translation: CreativeTranslation | None = None,
    v8_1_distillation: CreativeKnowledgeDistillationReport | None = None,
    v8_2_symbolic_translation: SymbolicTranslationReport | None = None,
    v8_3_sacred_geometry: SacredGeometryReport | None = None,
    v8_4_sacred_architecture: SacredArchitectureReport | None = None,
    v8_5_mythopoetic_narrative: MythopoeticNarrativeReport | None = None,
    v8_6_immersive_composer: ImmersiveAudiovisualComposerReport | None = None,
) -> HoloGenesisReport:
    """Build a bounded V8.7 HoloGenesis report without runtime mutation."""

    translation = creative_translation or derive_creative_translation(query, domains=domains)
    distillation = v8_1_distillation or build_v8_1_creative_knowledge_distillation()
    symbolic = v8_2_symbolic_translation or build_v8_2_symbolic_translation_engine(
        query,
        domains=domains,
        creative_translation=translation,
        v8_1_distillation=distillation,
    )
    geometry = v8_3_sacred_geometry or build_v8_3_sacred_geometry_engine(
        query,
        domains=domains,
        creative_translation=translation,
        v8_1_distillation=distillation,
        v8_2_symbolic_translation=symbolic,
    )
    architecture = v8_4_sacred_architecture or build_v8_4_sacred_architecture_engine(
        query,
        domains=domains,
        creative_translation=translation,
        v8_1_distillation=distillation,
        v8_2_symbolic_translation=symbolic,
        v8_3_sacred_geometry=geometry,
    )
    narrative = v8_5_mythopoetic_narrative or build_v8_5_mythopoetic_narrative_engine(
        query,
        domains=domains,
        creative_translation=translation,
        v8_1_distillation=distillation,
        v8_2_symbolic_translation=symbolic,
        v8_3_sacred_geometry=geometry,
        v8_4_sacred_architecture=architecture,
    )
    composer = v8_6_immersive_composer or build_v8_6_immersive_audiovisual_composer(
        query,
        domains=domains,
        creative_translation=translation,
        v8_1_distillation=distillation,
        v8_2_symbolic_translation=symbolic,
        v8_3_sacred_geometry=geometry,
        v8_4_sacred_architecture=architecture,
        v8_5_mythopoetic_narrative=narrative,
    )
    request = AssistantRequest(query=query, domains=domains)
    execution_plan = derive_creative_execution_plan(
        request=request,
        route_decision=None,
        creative_translation=translation,
        retrieval_chunk_count=distillation.kb_reality.indexed_chunk_count,
    )
    unsupported_claim_risks = hologenesis_unsupported_claim_risks(query)
    graphs = build_hologenesis_unified_graphs(
        distillation=distillation,
        symbolic=symbolic,
        geometry=geometry,
        architecture=architecture,
        narrative=narrative,
        composer=composer,
    )
    blackboard = build_hologenesis_blackboard_entries(graphs)
    schedule = build_hologenesis_symbolic_schedule(blackboard)
    creative_plan = build_hologenesis_creative_plan(
        execution_plan=execution_plan,
        composer=composer,
    )
    routes = build_hologenesis_route_recommendations(
        domains=domains,
        execution_plan=execution_plan,
    )
    decisions = build_hologenesis_artistic_decisions(graphs=graphs, composer=composer)
    curatorial = build_hologenesis_curatorial_assessments(
        symbolic=symbolic,
        geometry=geometry,
        architecture=architecture,
        narrative=narrative,
        composer=composer,
        unsupported_claim_risks=unsupported_claim_risks,
    )
    readiness = build_hologenesis_readiness_scores(
        architecture=architecture,
        composer=composer,
        unsupported_claim_risks=unsupported_claim_risks,
    )
    external_audit = build_hologenesis_external_integration_audit()
    bundle = build_hologenesis_project_bundle(
        source_query=query,
        domains=domains,
        translation=translation,
        execution_plan=execution_plan,
        geometry=geometry,
        architecture=architecture,
        narrative=narrative,
        composer=composer,
    )
    roadmap = hologenesis_core_roadmap_assessment()
    classified = hologenesis_items_by_classification(roadmap)
    reused_surface_ids = _reused_surface_ids(
        symbolic=symbolic,
        geometry=geometry,
        architecture=architecture,
        narrative=narrative,
        composer=composer,
    )
    return HoloGenesisReport(
        source_query=_clip(query, 920),
        reused_surface_ids=reused_surface_ids,
        composition_audit_summary=hologenesis_composition_audit_summary(),
        unified_graphs=graphs,
        blackboard_entries=blackboard,
        symbolic_schedule=schedule,
        creative_plan=creative_plan,
        route_recommendations=routes,
        artistic_decisions=decisions,
        curatorial_assessments=curatorial,
        readiness_scores=readiness,
        external_integration_audit=external_audit,
        project_bundle=bundle,
        validation_findings=build_hologenesis_validation_findings(unsupported_claim_risks),
        confidence=_build_confidence(
            distillation=distillation,
            symbolic=symbolic,
            geometry=geometry,
            architecture=architecture,
            narrative=narrative,
            composer=composer,
            graph_count=len(graphs),
            decision_count=len(decisions),
            readiness_score_count=len(readiness),
            reused_surface_ids=reused_surface_ids,
            unsupported_claim_risks=unsupported_claim_risks,
        ),
        roadmap_assessment=roadmap,
        implemented_report_items=classified[HoloGenesisRoadmapClassification.IMPLEMENTED_REPORT_BEHAVIOR],
        reused_existing_report_items=classified[HoloGenesisRoadmapClassification.REUSED_EXISTING_REPORT],
        export_planning_only_items=classified[HoloGenesisRoadmapClassification.EXPORT_PLANNING_ONLY],
        advisory_only_items=classified[HoloGenesisRoadmapClassification.ADVISORY_ONLY],
        future_hook_only_items=classified[HoloGenesisRoadmapClassification.FUTURE_HOOK_ONLY],
        out_of_scope_unsupported_items=classified[HoloGenesisRoadmapClassification.OUT_OF_SCOPE_UNSUPPORTED],
        missing_items=classified[HoloGenesisRoadmapClassification.MISSING],
        unsupported_claim_risks=unsupported_claim_risks,
        hitl_questions=hologenesis_hitl_questions(unsupported_claim_risks),
    )


def hologenesis_core_prompt_lines(report: HoloGenesisReport) -> tuple[str, ...]:
    """Render compact provider-independent V8.7 HoloGenesis guidance."""

    lines = [
        f"HoloGenesis boundary: {report.authority_boundary}",
        f"HoloGenesis confidence: {report.confidence.band.value} {report.confidence.score:.2f}.",
    ]
    lines.extend(f"Composition audit: {item}" for item in report.composition_audit_summary)
    for graph in report.unified_graphs:
        lines.append(f"Unified graph: {graph.kind.value}; {graph.synthesis_summary}")
        for node in graph.nodes[:3]:
            lines.append(f"Graph node: {node.node_id}; {node.label}; {node.summary}")
    for entry in report.blackboard_entries:
        lines.append(f"Creative blackboard: {entry.channel}; {entry.summary}; action: {entry.recommended_action}")
    for step in report.symbolic_schedule:
        lines.append(f"Symbolic schedule: {step.sequence}; {step.focus}; output: {step.output_contract}")
    for stage in report.creative_plan:
        lines.append(f"Creative planner: {stage.title}; {stage.objective}")
    for route in report.route_recommendations:
        lines.append(f"Creative route: {route.route_type}; {route.recommendation}")
    for decision in report.artistic_decisions[:6]:
        lines.append(f"Artistic decision: {decision.decision}; {decision.rationale}")
    for assessment in report.curatorial_assessments:
        lines.append(f"Curatorial assessment: {assessment.engine}; {assessment.status}; {assessment.summary}")
    for score in report.readiness_scores:
        lines.append(f"Readiness score: {score.label}; {score.score}; {score.band.value}; {score.rationale}")
    for audit in report.external_integration_audit:
        lines.append(
            "External integration audit: "
            f"{audit.label}; {audit.classification.value}; {audit.supported_behavior}; "
            f"unsupported: {audit.unsupported_behavior}"
        )
    bundle = report.project_bundle
    lines.append(f"Creative project bundle: {bundle.project_title}; {bundle.project_summary}")
    lines.extend(f"Project architecture: {item}" for item in bundle.architecture_outline[:5])
    lines.extend(f"README outline: {item}" for item in bundle.readme_outline[:5])
    lines.extend(f"Capstone output: {item}" for item in bundle.capstone_outputs[:5])
    lines.extend(f"Research mode: {item}" for item in bundle.research_mode_plan[:5])
    lines.extend(f"Reference discovery query: {item}" for item in bundle.reference_discovery_queries[:5])
    lines.extend(
        f"HoloGenesis validation: {item.severity.value}; {item.summary}" for item in report.validation_findings
    )
    lines.extend(f"Unsupported HoloGenesis claim risk: {item}" for item in report.unsupported_claim_risks)
    lines.extend(f"HITL HoloGenesis question: {item}" for item in report.hitl_questions)
    return tuple(lines[:180])


def hologenesis_core_roadmap_assessment() -> tuple[HoloGenesisRoadmapItemAssessment, ...]:
    """Return the V8.7 roadmap reality-check assessment."""

    return tuple(
        HoloGenesisRoadmapItemAssessment(
            item=item,
            classification=HoloGenesisRoadmapClassification(classification),
            rationale=rationale,
            action_required_before_hitl=action_required,
            hitl_required=hitl_required,
        )
        for item, (classification, rationale, action_required, hitl_required) in ROADMAP_CLASSIFICATION_ROWS.items()
    )


def hologenesis_unsupported_claim_risks(query: str) -> tuple[str, ...]:
    """Detect unsupported HoloGenesis/DCC/authority claim risks in request text."""

    normalized = " ".join(query.lower().split())
    risks: list[str] = []
    if "holomind" in normalized:
        risks.append("HoloMind is requested or implied, but V8.7 only exposes explicit future hooks.")
    if "holoiverse" in normalized:
        risks.append("HOLOiVERSE is requested or implied, but V8.7 does not implement it.")
    external_terms = ("unity", "unreal", "touchdesigner", "blender", "houdini", "dcc")
    execution_terms = ("automate", "call", "control", "execute", "live", "run", "spawn", "write")
    if any(term in normalized for term in external_terms) and any(term in normalized for term in execution_terms):
        risks.append("Live external DCC execution is requested or implied, but V8.7 only supports export planning.")
    if "mcp" in normalized and any(term in normalized for term in execution_terms):
        risks.append(
            "Live MCP creative tool execution is requested or implied, but V8.7 only supports tool-layer planning."
        )
    authority_terms = ("certify", "guarantee", "prove", "ritual efficacy", "authoritative")
    if any(term in normalized for term in authority_terms):
        risks.append("Authority/certification language is requested or implied, but V8.7 readiness remains advisory.")
    return tuple(_dedupe(risks))


def hologenesis_hitl_questions(unsupported_claim_risks: Sequence[str]) -> tuple[str, ...]:
    if not unsupported_claim_risks:
        return ()
    return (
        "Should unsupported live DCC/MCP, HoloMind, HOLOiVERSE, or certification "
        "language be removed before public use?",
        "Should external-tool handoff stay as manual export-planning notes for this HITL review?",
    )


def _build_confidence(
    *,
    distillation: CreativeKnowledgeDistillationReport,
    symbolic: SymbolicTranslationReport,
    geometry: SacredGeometryReport,
    architecture: SacredArchitectureReport,
    narrative: MythopoeticNarrativeReport,
    composer: ImmersiveAudiovisualComposerReport,
    graph_count: int,
    decision_count: int,
    readiness_score_count: int,
    reused_surface_ids: tuple[str, ...],
    unsupported_claim_risks: Sequence[str],
) -> HoloGenesisConfidence:
    distillation_score = _average(record.confidence.score for record in distillation.records[:8])
    score = _average(
        (
            distillation_score,
            symbolic.confidence.score,
            geometry.confidence.score,
            architecture.confidence.score,
            narrative.confidence.score,
            composer.confidence.score,
        )
    )
    if unsupported_claim_risks:
        score = min(score, 0.66)
    caveats = tuple(unsupported_claim_risks[:8])
    return HoloGenesisConfidence(
        score=round(score, 3),
        band=hologenesis_confidence_band(score, guarded=bool(caveats)),
        graph_count=graph_count,
        decision_count=decision_count,
        readiness_score_count=readiness_score_count,
        reused_engine_ids=reused_surface_ids[:18],
        caveats=caveats,
    )


def _reused_surface_ids(
    *,
    symbolic: SymbolicTranslationReport,
    geometry: SacredGeometryReport,
    architecture: SacredArchitectureReport,
    narrative: MythopoeticNarrativeReport,
    composer: ImmersiveAudiovisualComposerReport,
) -> tuple[str, ...]:
    return _dedupe(
        (
            "v3_creative_translation",
            "v3_creative_planning",
            "v8_1_creative_knowledge_distillation",
            "v8_2_symbolic_translation_engine",
            "v8_3_sacred_geometry_engine",
            "v8_4_sacred_architecture_engine",
            "v8_5_mythopoetic_engine",
            "v8_6_immersive_composer",
            *symbolic.reused_surface_ids,
            *geometry.reused_surface_ids,
            *architecture.reused_surface_ids,
            *narrative.reused_surface_ids,
            *composer.reused_surface_ids,
        )
    )[:32]


def _average(values: Iterable[float]) -> float:
    collected = tuple(values)
    if not collected:
        return 0.0
    return sum(collected) / len(collected)


def _clip(value: str, limit: int) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "."


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        cleaned = " ".join(str(value).split())
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return tuple(result)


__all__ = [
    "build_v8_7_hologenesis_core",
    "hologenesis_core_prompt_lines",
    "hologenesis_core_roadmap_assessment",
    "hologenesis_hitl_questions",
    "hologenesis_unsupported_claim_risks",
]
