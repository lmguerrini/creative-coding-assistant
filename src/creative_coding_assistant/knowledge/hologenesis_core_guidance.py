"""Guidance builders for the V8.7 HoloGenesis Creative Operating System."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.knowledge.creative_distillation import CreativeKnowledgeDistillationReport
from creative_coding_assistant.knowledge.hologenesis_core_catalog import EXTERNAL_INTEGRATION_ROWS
from creative_coding_assistant.knowledge.hologenesis_core_contracts import (
    HoloGenesisBlackboardEntry,
    HoloGenesisCuratorialAssessment,
    HoloGenesisDecision,
    HoloGenesisExternalIntegrationAudit,
    HoloGenesisGraphEdge,
    HoloGenesisGraphKind,
    HoloGenesisGraphNode,
    HoloGenesisPlannerStage,
    HoloGenesisReadinessScore,
    HoloGenesisRoadmapClassification,
    HoloGenesisRouteRecommendation,
    HoloGenesisScheduleStep,
    HoloGenesisUnifiedGraph,
    hologenesis_readiness_band,
)
from creative_coding_assistant.knowledge.immersive_audiovisual_composer_contracts import (
    ImmersiveAudiovisualComposerReport,
)
from creative_coding_assistant.knowledge.mythopoetic_narrative_contracts import (
    MythopoeticNarrativeReport,
)
from creative_coding_assistant.knowledge.sacred_architecture_contracts import (
    SacredArchitectureReport,
)
from creative_coding_assistant.knowledge.sacred_geometry_contracts import SacredGeometryReport
from creative_coding_assistant.knowledge.symbolic_translation import SymbolicTranslationReport
from creative_coding_assistant.orchestration.creative_planning import CreativeExecutionPlan


def build_hologenesis_unified_graphs(
    *,
    distillation: CreativeKnowledgeDistillationReport,
    symbolic: SymbolicTranslationReport,
    geometry: SacredGeometryReport,
    architecture: SacredArchitectureReport,
    narrative: MythopoeticNarrativeReport,
    composer: ImmersiveAudiovisualComposerReport,
) -> tuple[HoloGenesisUnifiedGraph, ...]:
    """Build the five required V8.7 graph projections."""

    symbolic_nodes = tuple(
        HoloGenesisGraphNode(
            node_id=f"symbolic::{mapping.motif_id}",
            label=mapping.motif_label,
            source_engine_ids=("v8_2_symbolic_translation_engine",),
            summary=mapping.creative_coding_intent,
            recommendations=_dedupe(
                (*mapping.visual_guidance[:2], *mapping.motion_guidance[:2], *mapping.audio_guidance[:1])
            ),
            confidence_signal=mapping.confidence_score,
            evidence=mapping.evidence[:10],
        )
        for mapping in symbolic.motif_mappings[:6]
    )
    sacred_nodes = tuple(
        HoloGenesisGraphNode(
            node_id=f"sacred_knowledge::{record.record_id}",
            label=record.title,
            source_engine_ids=("v8_1_creative_knowledge_distillation",),
            summary=record.summary,
            recommendations=_dedupe((*record.workflow_steps[:2], *record.technique_tags[:2], *record.pattern_tags[:2])),
            confidence_signal=record.confidence.score,
            evidence=tuple(item.note for item in record.provenance[:6]),
        )
        for record in distillation.records[:6]
    )
    geometry_nodes = tuple(
        HoloGenesisGraphNode(
            node_id=f"geometry::{pattern.pattern_id}",
            label=pattern.label,
            source_engine_ids=("v8_3_sacred_geometry_engine", "v8_6_immersive_composer"),
            summary=pattern.creative_intent,
            recommendations=_dedupe(
                (
                    *pattern.structure_guidance[:2],
                    *pattern.algorithm_recommendations[:2],
                    *pattern.motion_mappings[:1],
                    *pattern.audio_harmonic_mappings[:1],
                )
            ),
            confidence_signal=pattern.confidence_score,
            evidence=pattern.evidence[:10],
        )
        for pattern in geometry.pattern_guidance[:6]
    )
    narrative_nodes = tuple(
        HoloGenesisGraphNode(
            node_id=f"narrative::{scene.scene_id}",
            label=scene.title,
            source_engine_ids=("v8_5_mythopoetic_engine", "v8_6_immersive_composer"),
            summary=scene.narrative_function,
            recommendations=_dedupe(
                (
                    _first(scene.visual_guidance),
                    _first(scene.motion_guidance),
                    _first(scene.audio_guidance),
                    _first(scene.spatial_guidance),
                )
            ),
            confidence_signal=narrative.confidence.score,
            evidence=scene.evidence[:10],
        )
        for scene in narrative.scene_sequence[:6]
    )
    installation_nodes = tuple(
        HoloGenesisGraphNode(
            node_id=f"installation::{node.node_id}",
            label=node.label,
            source_engine_ids=("v8_4_sacred_architecture_engine", "v8_6_immersive_composer"),
            summary=node.dramaturgical_function,
            recommendations=_dedupe(
                (
                    node.visual_language,
                    node.sacred_lighting,
                    node.spatial_audio_role,
                    node.audience_function,
                )
            ),
            confidence_signal=composer.confidence.score,
            evidence=node.evidence[:10],
        )
        for node in composer.scene_graph[:6]
    )
    return (
        _graph(
            kind=HoloGenesisGraphKind.SYMBOLIC,
            source_engine_ids=("v8_2_symbolic_translation_engine", "v8_5_mythopoetic_engine"),
            nodes=symbolic_nodes,
            summary="Unifies symbolic motif, interpretation boundary, and symbolic narrative signals.",
        ),
        _graph(
            kind=HoloGenesisGraphKind.SACRED_KNOWLEDGE,
            source_engine_ids=("v8_1_creative_knowledge_distillation", "v8_2_symbolic_translation_engine"),
            nodes=sacred_nodes,
            summary="Unifies distilled creative knowledge with bounded sacred-knowledge provenance and confidence.",
        ),
        _graph(
            kind=HoloGenesisGraphKind.GEOMETRY,
            source_engine_ids=("v8_3_sacred_geometry_engine", "v8_6_immersive_composer"),
            nodes=geometry_nodes,
            summary="Unifies geometry, harmonic proportion, motion, light, and audio mapping guidance.",
        ),
        _graph(
            kind=HoloGenesisGraphKind.NARRATIVE,
            source_engine_ids=("v8_5_mythopoetic_engine", "v8_6_immersive_composer"),
            nodes=narrative_nodes,
            summary="Unifies mythopoetic scene sequence, audience communication, and temporal dramaturgy.",
        ),
        _graph(
            kind=HoloGenesisGraphKind.INSTALLATION,
            source_engine_ids=("v8_4_sacred_architecture_engine", "v8_6_immersive_composer"),
            nodes=installation_nodes,
            summary=(
                "Unifies spatial topology, installation flow, audience journey, "
                "and immersive scene graph guidance."
            ),
        ),
    )


def build_hologenesis_blackboard_entries(
    graphs: Sequence[HoloGenesisUnifiedGraph],
) -> tuple[HoloGenesisBlackboardEntry, ...]:
    graph_ids = {graph.kind: graph.graph_id for graph in graphs}
    return (
        HoloGenesisBlackboardEntry(
            entry_id="blackboard::knowledge_unification",
            channel="knowledge_unification",
            summary="Use V8.1 provenance and confidence as the root evidence layer for the project plan.",
            source_graph_ids=(graph_ids[HoloGenesisGraphKind.SACRED_KNOWLEDGE],),
            decision_pressure="medium",
            recommended_action="Keep source confidence visible in every public-facing project claim.",
            evidence=("V8.1 records are reused as bounded knowledge evidence.",),
        ),
        HoloGenesisBlackboardEntry(
            entry_id="blackboard::symbolic_geometry",
            channel="symbolic_geometry",
            summary=(
                "Bind symbolic motifs to geometry and visual/audiovisual mappings "
                "before generating delivery outputs."
            ),
            source_graph_ids=(graph_ids[HoloGenesisGraphKind.SYMBOLIC], graph_ids[HoloGenesisGraphKind.GEOMETRY]),
            decision_pressure="medium",
            recommended_action="Prefer motifs with explicit V8.2 and V8.3 evidence over unsupported esoteric claims.",
            evidence=("V8.2 motif mappings and V8.3 geometry patterns are both present.",),
        ),
        HoloGenesisBlackboardEntry(
            entry_id="blackboard::narrative_installation",
            channel="narrative_installation",
            summary="Connect narrative scenes to spatial installation nodes and audience journey functions.",
            source_graph_ids=(graph_ids[HoloGenesisGraphKind.NARRATIVE], graph_ids[HoloGenesisGraphKind.INSTALLATION]),
            decision_pressure="high",
            recommended_action=(
                "Review scene order, thresholds, and audience path before treating "
                "the plan as exhibit-ready."
            ),
            evidence=("V8.5 scenes and V8.6 scene graph nodes define the project flow.",),
        ),
        HoloGenesisBlackboardEntry(
            entry_id="blackboard::curatorial_quality",
            channel="curatorial_quality",
            summary="Use curatorial validation, aesthetic evaluation, and readiness scoring as HITL review inputs.",
            source_graph_ids=tuple(graph.graph_id for graph in graphs),
            decision_pressure="high",
            recommended_action="Treat readiness scores as review guidance, not as certification.",
            evidence=("V8.7 readiness scoring remains bounded report behavior.",),
        ),
        HoloGenesisBlackboardEntry(
            entry_id="blackboard::delivery_bundle",
            channel="delivery_bundle",
            summary=(
                "Generate project architecture, README, portfolio, capstone, and "
                "external export-planning outlines."
            ),
            source_graph_ids=tuple(graph.graph_id for graph in graphs),
            decision_pressure="medium",
            recommended_action=(
                "Keep generated bundle artifacts as outlines until a human accepts "
                "file writes or external production."
            ),
            evidence=("V8.7 does not write bundle files or execute external tools.",),
        ),
    )


def build_hologenesis_symbolic_schedule(
    entries: Sequence[HoloGenesisBlackboardEntry],
) -> tuple[HoloGenesisScheduleStep, ...]:
    entry_ids = tuple(entry.entry_id for entry in entries)
    return (
        HoloGenesisScheduleStep(
            step_id="schedule::01_unify_evidence",
            sequence=1,
            focus="Unify knowledge, symbolic, geometry, architecture, narrative, and audiovisual evidence.",
            source_entry_ids=entry_ids[:2],
            rationale="Evidence must be composed before curatorial or delivery claims can be honest.",
            output_contract="Unified graph nodes, confidence signals, and blackboard entries.",
        ),
        HoloGenesisScheduleStep(
            step_id="schedule::02_compose_project",
            sequence=2,
            focus="Compose installation narrative, audience path, visual language, and spatial audio plan.",
            source_entry_ids=entry_ids[1:3],
            rationale="The project should be planned as a coherent installation before bundle generation.",
            output_contract="Creative planner stages, artistic decisions, and route recommendations.",
        ),
        HoloGenesisScheduleStep(
            step_id="schedule::03_curate_validate",
            sequence=3,
            focus="Run bounded curatorial reasoning, aesthetic evaluation, and readiness scoring.",
            source_entry_ids=entry_ids[2:4],
            rationale="Curatorial validation separates review-ready plans from unsupported claims.",
            output_contract="Curatorial assessments, validation findings, and readiness scores.",
        ),
        HoloGenesisScheduleStep(
            step_id="schedule::04_generate_bundle",
            sequence=4,
            focus="Generate project architecture, portfolio, README, capstone, research, and export-planning outlines.",
            source_entry_ids=entry_ids[3:],
            rationale=(
                "Delivery artifacts should be generated only after evidence, "
                "composition, and validation are visible."
            ),
            output_contract="Typed project bundle and external integration audit.",
        ),
    )


def build_hologenesis_creative_plan(
    *,
    execution_plan: CreativeExecutionPlan,
    composer: ImmersiveAudiovisualComposerReport,
) -> tuple[HoloGenesisPlannerStage, ...]:
    runtime = execution_plan.recommended_runtime or "browser-internal preview runtime"
    return (
        HoloGenesisPlannerStage(
            stage_id="plan::unified_brief",
            title="Unified creative brief",
            objective="Convert V8.1-V8.6 evidence into one bounded project brief.",
            recommended_outputs=("curatorial thesis", "source provenance summary", "claim boundary notes"),
            dependencies=("v8_1_creative_knowledge_distillation", "v8_2_symbolic_translation_engine"),
            confidence_signal=0.82,
        ),
        HoloGenesisPlannerStage(
            stage_id="plan::installation_architecture",
            title="Installation architecture",
            objective="Shape spatial topology, scene graph, audience path, and audiovisual timing.",
            recommended_outputs=("installation flow", "scene graph map", "audience journey notes"),
            dependencies=("v8_4_sacred_architecture_engine", "v8_6_immersive_composer"),
            confidence_signal=composer.confidence.score,
        ),
        HoloGenesisPlannerStage(
            stage_id="plan::internal_preview",
            title="Internal preview plan",
            objective=f"Use {runtime} guidance for browser-internal validation before any external handoff.",
            recommended_outputs=("preview target notes", "runtime constraints", "fallback path"),
            dependencies=("v8_6_immersive_composer",),
            confidence_signal=0.76 if execution_plan.runtime_available else 0.58,
        ),
        HoloGenesisPlannerStage(
            stage_id="plan::curatorial_review",
            title="Curatorial review",
            objective="Review symbolic consistency, aesthetic quality, museum readiness, and exhibition constraints.",
            recommended_outputs=("readiness scores", "HITL review questions", "risk notes"),
            dependencies=("plan::unified_brief", "plan::installation_architecture"),
            confidence_signal=0.74,
        ),
        HoloGenesisPlannerStage(
            stage_id="plan::project_bundle",
            title="Project bundle",
            objective="Produce architecture, README, portfolio, capstone, research, and export-planning outlines.",
            recommended_outputs=(
                "README outline",
                "portfolio narrative",
                "capstone output plan",
                "external export checklist",
            ),
            dependencies=("plan::curatorial_review",),
            confidence_signal=0.78,
        ),
    )


def build_hologenesis_route_recommendations(
    *,
    domains: Sequence[CreativeCodingDomain],
    execution_plan: CreativeExecutionPlan,
) -> tuple[HoloGenesisRouteRecommendation, ...]:
    runtime = execution_plan.recommended_runtime or "browser-internal preview foundation"
    domain_values = {domain.value for domain in domains}
    external_domains = domain_values.intersection(
        {"unity", "unreal", "touchdesigner", "houdini", "blender_geometry_nodes", "blender_python_api"}
    )
    return (
        HoloGenesisRouteRecommendation(
            route_id="route::browser_internal_preview",
            route_type="browser_internal",
            recommendation=f"Use {runtime} for implemented preview and validation where available.",
            rationale="V8.6 audited browser preview foundations are reusable; V8.7 does not add preview runtimes.",
            blocked_runtime_behaviors=("provider_model_routing", "preview_runtime_mutation", "artifact_execution"),
        ),
        HoloGenesisRouteRecommendation(
            route_id="route::curatorial_review",
            route_type="curatorial_review",
            recommendation=(
                "Review curatorial thesis, symbolic consistency, aesthetic quality, "
                "and readiness scores before public claims."
            ),
            rationale=(
                "V8.7 produces review-ready reports, but human judgment remains "
                "the authority for exhibition claims."
            ),
            blocked_runtime_behaviors=("certification", "institutional_approval", "automatic_hitl_execution"),
        ),
        HoloGenesisRouteRecommendation(
            route_id="route::external_export_planning",
            route_type="external_export_planning",
            recommendation=_external_route_recommendation(external_domains),
            rationale="External DCC and MCP layers are audited as handoff planning only.",
            blocked_runtime_behaviors=("external_dcc_execution", "mcp_tool_execution", "file_export"),
        ),
        HoloGenesisRouteRecommendation(
            route_id="route::research_followup",
            route_type="research_followup",
            recommendation=(
                "Use creative research mode to plan follow-up questions and "
                "reference discovery terms without browsing or fetching."
            ),
            rationale=(
                "V8.7 can frame research and references, but live discovery "
                "belongs to explicit future or user-approved flows."
            ),
            blocked_runtime_behaviors=("web_browsing", "paper_download", "source_registry_mutation"),
        ),
    )


def build_hologenesis_artistic_decisions(
    *,
    graphs: Sequence[HoloGenesisUnifiedGraph],
    composer: ImmersiveAudiovisualComposerReport,
) -> tuple[HoloGenesisDecision, ...]:
    graph_kinds = tuple(graph.kind for graph in graphs)
    composer_decisions = composer.artistic_decisions[:2]
    decisions = [
        HoloGenesisDecision(
            decision_id="decision::unify_evidence_first",
            decision="Ground the project in V8.1 provenance before making symbolic or curatorial claims.",
            rationale=(
                "Knowledge provenance is the strongest guard against overclaiming "
                "sacred, mystical, or exhibition-readiness behavior."
            ),
            source_graph_kinds=(HoloGenesisGraphKind.SACRED_KNOWLEDGE, HoloGenesisGraphKind.SYMBOLIC),
            evidence=("V8.1 records and V8.2 interpretation boundaries are present.",),
            confidence_signal=0.84,
        ),
        HoloGenesisDecision(
            decision_id="decision::compose_installation_flow",
            decision="Treat narrative scenes and immersive scene graph nodes as the primary project spine.",
            rationale="V8.5 and V8.6 provide ordered scene and audience-flow evidence that can drive bundle structure.",
            source_graph_kinds=(HoloGenesisGraphKind.NARRATIVE, HoloGenesisGraphKind.INSTALLATION),
            evidence=("V8.5 scene sequence and V8.6 scene graph are available.",),
            confidence_signal=composer.confidence.score,
        ),
        HoloGenesisDecision(
            decision_id="decision::keep_external_tools_manual",
            decision="Frame Unity, Unreal, TouchDesigner, Blender, Houdini, and MCP only as export-planning handoffs.",
            rationale=(
                "No callable live integration path exists in the current codebase, "
                "so executing or exporting would overclaim."
            ),
            source_graph_kinds=graph_kinds[:5],
            evidence=("External integration audit classifies all external layers as export-planning-only.",),
            confidence_signal=0.91,
        ),
        HoloGenesisDecision(
            decision_id="decision::bundle_as_outline",
            decision="Generate architecture, README, portfolio, capstone, and research outputs as typed outlines only.",
            rationale=(
                "This provides Capstone value without writing storage or "
                "pretending automated publication exists."
            ),
            source_graph_kinds=graph_kinds[:5],
            evidence=("Project bundle file writes are explicitly false.",),
            confidence_signal=0.86,
        ),
    ]
    for item in composer_decisions:
        decisions.append(
            HoloGenesisDecision(
                decision_id=f"decision::composer::{item.decision_id}",
                decision=item.decision,
                rationale=item.rationale,
                source_graph_kinds=tuple(
                    kind
                    for kind in (
                        HoloGenesisGraphKind.SYMBOLIC,
                        HoloGenesisGraphKind.GEOMETRY,
                        HoloGenesisGraphKind.INSTALLATION,
                    )
                    if kind in graph_kinds
                ),
                evidence=item.evidence[:10],
                confidence_signal=composer.confidence.score,
            )
        )
    return tuple(decisions[:6])


def build_hologenesis_curatorial_assessments(
    *,
    symbolic: SymbolicTranslationReport,
    geometry: SacredGeometryReport,
    architecture: SacredArchitectureReport,
    narrative: MythopoeticNarrativeReport,
    composer: ImmersiveAudiovisualComposerReport,
    unsupported_claim_risks: Sequence[str],
) -> tuple[HoloGenesisCuratorialAssessment, ...]:
    risk_status = "review_required" if unsupported_claim_risks else "pass"
    bounded = (
        "Bounded creative interpretation only; no religious, mystical, institutional, or psychological authority claim."
    )
    return (
        HoloGenesisCuratorialAssessment(
            assessment_id="curatorial::intelligence",
            engine="curatorial_intelligence",
            status="pass",
            summary="The project has enough cross-domain evidence to support a curatorial concept review.",
            evidence=(symbolic.capability_id, geometry.capability_id, narrative.capability_id),
            confidence_signal=_average(
                (symbolic.confidence.score, geometry.confidence.score, narrative.confidence.score)
            ),
            bounded_framing=bounded,
        ),
        HoloGenesisCuratorialAssessment(
            assessment_id="curatorial::reasoning",
            engine="curatorial_reasoning",
            status="pass",
            summary=(
                "Symbolic motifs, geometry patterns, spatial architecture, "
                "narrative scenes, and audiovisual plans are traceably linked."
            ),
            evidence=(architecture.capability_id, composer.capability_id),
            confidence_signal=_average((architecture.confidence.score, composer.confidence.score)),
            bounded_framing=bounded,
        ),
        HoloGenesisCuratorialAssessment(
            assessment_id="curatorial::validation",
            engine="curatorial_validation",
            status=risk_status,
            summary=(
                "Validation separates implemented report behavior from unsupported "
                "execution, DCC, and HoloMind claims."
            ),
            evidence=tuple(unsupported_claim_risks[:6]) or ("No unsupported live integration claim detected.",),
            confidence_signal=0.62 if unsupported_claim_risks else 0.82,
            bounded_framing=bounded,
        ),
        HoloGenesisCuratorialAssessment(
            assessment_id="curatorial::explainability",
            engine="curatorial_explainability",
            status="pass",
            summary="Decisions and bundle sections carry graph, provenance, confidence, and readiness evidence.",
            evidence=("Unified graph ids, blackboard entries, and readiness scores are included.",),
            confidence_signal=0.84,
            bounded_framing=bounded,
        ),
        HoloGenesisCuratorialAssessment(
            assessment_id="curatorial::mystical_consistency",
            engine="mystical_consistency",
            status="guarded" if unsupported_claim_risks else "pass",
            summary=(
                "Mystical language is handled as creative framing and consistency "
                "review, not truth, proof, or ritual efficacy."
            ),
            evidence=symbolic.interpretation_boundaries[:6],
            confidence_signal=0.56 if unsupported_claim_risks else 0.78,
            bounded_framing=bounded,
        ),
        HoloGenesisCuratorialAssessment(
            assessment_id="curatorial::symbolic_explainability",
            engine="symbolic_explainability",
            status="pass",
            summary="Symbolic choices are explainable through V8.2 motif mappings and V8.5 narrative symbol nodes.",
            evidence=tuple(mapping.motif_id for mapping in symbolic.motif_mappings[:6]),
            confidence_signal=symbolic.confidence.score,
            bounded_framing=bounded,
        ),
        HoloGenesisCuratorialAssessment(
            assessment_id="curatorial::aesthetic_evaluation",
            engine="aesthetic_evaluation",
            status="pass",
            summary=(
                "Aesthetic quality is evaluated through V8.6 visual language, "
                "lighting, color, audio, and audience journey plans."
            ),
            evidence=composer.composition_audit_summary[:6],
            confidence_signal=composer.confidence.score,
            bounded_framing=bounded,
        ),
    )


def build_hologenesis_readiness_scores(
    *,
    architecture: SacredArchitectureReport,
    composer: ImmersiveAudiovisualComposerReport,
    unsupported_claim_risks: Sequence[str],
) -> tuple[HoloGenesisReadinessScore, ...]:
    risk_penalty = 14 if unsupported_claim_risks else 0
    installation = _clamp_score(
        58 + len(composer.scene_graph) * 5 + len(architecture.pattern_guidance) * 2 - risk_penalty
    )
    museum = _clamp_score(54 + len(composer.artistic_decisions) * 4 + len(composer.preview_audit) - risk_penalty)
    international = _clamp_score(museum - 8)
    bundle = _clamp_score(76 - (8 if unsupported_claim_risks else 0))
    return (
        _readiness(
            score_id="readiness::installation_quality",
            label="installation_quality",
            score=installation,
            rationale="Score reflects scene graph depth, architectural pattern coverage, and claim-risk posture.",
            notes=("Qualitative planning score only.", "No physical simulation or safety certification is executed."),
        ),
        _readiness(
            score_id="readiness::museum",
            label="museum_readiness",
            score=museum,
            rationale=(
                "Score reflects explainable decisions, preview audit evidence, "
                "and curatorial validation posture."
            ),
            notes=("Advisory museum readiness only.", "Requires human curatorial and production review."),
        ),
        _readiness(
            score_id="readiness::international",
            label="international_exhibition_readiness",
            score=international,
            rationale=(
                "Score discounts museum readiness for translation, rights, "
                "shipping, access, and venue constraints."
            ),
            notes=("No legal, customs, safety, insurance, or institutional certification is provided.",),
        ),
        _readiness(
            score_id="readiness::bundle",
            label="project_bundle_readiness",
            score=bundle,
            rationale=(
                "Score reflects availability of typed architecture, README, "
                "portfolio, capstone, pipeline, and research outlines."
            ),
            notes=("Bundle generation is in-memory report behavior.", "No files are written by V8.7."),
        ),
    )


def build_hologenesis_external_integration_audit() -> tuple[HoloGenesisExternalIntegrationAudit, ...]:
    return tuple(
        HoloGenesisExternalIntegrationAudit(
            integration_id=f"external::{integration_id}",
            label=label,
            classification=HoloGenesisRoadmapClassification.EXPORT_PLANNING_ONLY,
            supported_behavior=supported,
            unsupported_behavior=unsupported,
            export_planning_notes=notes,
        )
        for integration_id, (label, supported, unsupported, notes) in EXTERNAL_INTEGRATION_ROWS.items()
    )


def _graph(
    *,
    kind: HoloGenesisGraphKind,
    source_engine_ids: tuple[str, ...],
    nodes: tuple[HoloGenesisGraphNode, ...],
    summary: str,
) -> HoloGenesisUnifiedGraph:
    edges = tuple(
        HoloGenesisGraphEdge(
            edge_id=f"{kind.value}::edge::{index}",
            from_node_id=nodes[index - 1].node_id,
            to_node_id=nodes[index].node_id,
            relationship="sequenced_composition",
            evidence=(f"{nodes[index - 1].label} composes into {nodes[index].label}.",),
        )
        for index in range(1, len(nodes))
    )
    return HoloGenesisUnifiedGraph(
        graph_id=f"hologenesis::{kind.value}",
        kind=kind,
        source_engine_ids=source_engine_ids,
        nodes=nodes,
        edges=edges,
        synthesis_summary=summary,
    )


def _readiness(
    *,
    score_id: str,
    label: str,
    score: int,
    rationale: str,
    notes: tuple[str, ...],
) -> HoloGenesisReadinessScore:
    return HoloGenesisReadinessScore(
        score_id=score_id,
        label=label,
        score=score,
        band=hologenesis_readiness_band(score),
        rationale=rationale,
        review_notes=notes,
    )


def _external_route_recommendation(external_domains: set[str]) -> str:
    if external_domains:
        labels = ", ".join(sorted(external_domains))
        return f"Treat requested external domains ({labels}) as manual export-planning targets only."
    return "Include external DCC/MCP handoff notes only when useful for manual downstream production."


def _average(values: Iterable[float]) -> float:
    collected = tuple(values)
    if not collected:
        return 0.0
    return round(sum(collected) / len(collected), 3)


def _clamp_score(value: int) -> int:
    return max(0, min(100, value))


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        cleaned = " ".join(str(value).split())
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return tuple(result)


def _first(values: Sequence[str]) -> str:
    return values[0] if values else ""
