"""Bounded Artifact Dependency Graph for V3.3 workflows."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import AssistantRequest
from creative_coding_assistant.orchestration._metadata_utils import _clip, _dedupe
from creative_coding_assistant.orchestration.artifact_planner import ArtifactPlan
from creative_coding_assistant.orchestration.audio_visual_scene import (
    AudioVisualSceneProfile,
)
from creative_coding_assistant.orchestration.creative_composition import (
    CreativeCompositionPlan,
)
from creative_coding_assistant.orchestration.creative_constraint_priorities import (
    CreativeConstraintPrioritization,
)
from creative_coding_assistant.orchestration.creative_constraints import (
    CreativeConstraintSolution,
)
from creative_coding_assistant.orchestration.creative_hierarchy import (
    CreativeHierarchyPlan,
)
from creative_coding_assistant.orchestration.creative_intent import (
    CreativeIntentDecomposition,
)
from creative_coding_assistant.orchestration.creative_planning import (
    CreativeExecutionPlan,
)
from creative_coding_assistant.orchestration.creative_quality_prediction import (
    CreativeQualityPrediction,
)
from creative_coding_assistant.orchestration.creative_strategy import (
    CreativeStrategyProfile,
)
from creative_coding_assistant.orchestration.creative_technique import (
    CreativeTechniqueProfile,
)
from creative_coding_assistant.orchestration.creative_tradeoffs import (
    CreativeTradeoffProfile,
)
from creative_coding_assistant.orchestration.creative_translation import (
    CreativeTranslation,
)
from creative_coding_assistant.orchestration.cross_modality import (
    CrossModalityCompositionProfile,
)
from creative_coding_assistant.orchestration.emotional_consistency import (
    EmotionalConsistencyProfile,
)
from creative_coding_assistant.orchestration.generative_structure import (
    GenerativeStructureBlueprint,
)
from creative_coding_assistant.orchestration.procedural_structure import (
    ProceduralStructurePlan,
)
from creative_coding_assistant.orchestration.routing import RouteDecision
from creative_coding_assistant.orchestration.runtime_capabilities import (
    RuntimeCapabilityProfile,
)
from creative_coding_assistant.orchestration.semantic_motif import SemanticMotifSystem
from creative_coding_assistant.orchestration.symbolic_narrative import (
    SymbolicNarrativePlan,
)

DependencyNodeType = Literal[
    "planned_artifact",
    "required_component",
    "runtime_requirement",
    "creative_metadata",
    "generative_metadata",
    "output_structure",
    "prompt_guidance",
    "downstream_consumer",
]
DependencyNodeStatus = Literal["available", "inferred", "missing"]
DependencyRelationship = Literal[
    "requires",
    "informs",
    "blocks",
    "soft_informs",
    "feeds_prompt",
    "consumed_by",
]
DependencyStrength = Literal["required", "optional", "blocking", "soft"]

ARTIFACT_DEPENDENCY_GRAPH_AUTHORITY_BOUNDARY = (
    "The Artifact Dependency Graph structures inspectable metadata about "
    "planned artifact dependencies only; it does not implement Runtime "
    "Compatibility Engine, Multi-Artifact Strategy, Artifact Critic, Artifact "
    "Refiner, V4 multi-agent behavior, V5 execution optimization, runtime "
    "execution, provider routing, model routing, preview behavior, or runtime "
    "repair."
)


class ArtifactDependencyNode(BaseModel):
    """One inspectable node in the artifact dependency map."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    node_id: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=140)
    node_type: DependencyNodeType
    status: DependencyNodeStatus
    summary: str = Field(min_length=1, max_length=360)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=6)


class ArtifactDependencyEdge(BaseModel):
    """One bounded dependency edge between graph nodes."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    source_node_id: str = Field(min_length=1, max_length=80)
    target_node_id: str = Field(min_length=1, max_length=80)
    relationship: DependencyRelationship
    strength: DependencyStrength
    rationale: str = Field(min_length=1, max_length=300)


class ArtifactDependencyGraph(BaseModel):
    """Inspectable metadata-only dependency graph for planned artifacts."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["artifact_dependency_graph"] = "artifact_dependency_graph"
    primary_artifact_node_id: str = Field(min_length=1, max_length=80)
    artifact_nodes: tuple[ArtifactDependencyNode, ...] = Field(
        min_length=1,
        max_length=16,
    )
    dependency_edges: tuple[ArtifactDependencyEdge, ...] = Field(
        default_factory=tuple,
        max_length=32,
    )
    required_upstream_metadata: tuple[str, ...] = Field(min_length=1, max_length=10)
    optional_upstream_metadata: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=18,
    )
    blocking_dependencies: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    soft_dependencies: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    runtime_facing_dependencies: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=10,
    )
    prompt_facing_dependencies: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=10,
    )
    downstream_consumers: tuple[str, ...] = Field(min_length=1, max_length=8)
    missing_dependency_risks: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=10,
    )
    dependency_conflicts: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    hitl_questions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(
        default=ARTIFACT_DEPENDENCY_GRAPH_AUTHORITY_BOUNDARY,
        max_length=720,
    )
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=12)


def derive_artifact_dependency_graph(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    artifact_plan: ArtifactPlan | None,
    creative_translation: CreativeTranslation | None = None,
    creative_intent: CreativeIntentDecomposition | None = None,
    creative_hierarchy: CreativeHierarchyPlan | None = None,
    creative_plan: CreativeExecutionPlan | None = None,
    creative_constraints: CreativeConstraintSolution | None = None,
    creative_constraint_priorities: CreativeConstraintPrioritization | None = None,
    creative_strategy: CreativeStrategyProfile | None = None,
    creative_techniques: CreativeTechniqueProfile | None = None,
    runtime_capabilities: RuntimeCapabilityProfile | None = None,
    creative_tradeoffs: CreativeTradeoffProfile | None = None,
    creative_quality_prediction: CreativeQualityPrediction | None = None,
    symbolic_narrative: SymbolicNarrativePlan | None = None,
    creative_composition: CreativeCompositionPlan | None = None,
    procedural_structure: ProceduralStructurePlan | None = None,
    generative_structure: GenerativeStructureBlueprint | None = None,
    semantic_motif: SemanticMotifSystem | None = None,
    emotional_consistency: EmotionalConsistencyProfile | None = None,
    cross_modality: CrossModalityCompositionProfile | None = None,
    audio_visual_scene: AudioVisualSceneProfile | None = None,
) -> ArtifactDependencyGraph:
    """Derive artifact dependency metadata without changing execution behavior."""

    nodes = _artifact_nodes(
        request=request,
        route_decision=route_decision,
        artifact_plan=artifact_plan,
    )
    edges = _dependency_edges(nodes, artifact_plan)
    optional_upstream = _optional_upstream_metadata(
        route_decision=route_decision,
        creative_translation=creative_translation,
        creative_intent=creative_intent,
        creative_hierarchy=creative_hierarchy,
        creative_plan=creative_plan,
        creative_constraints=creative_constraints,
        creative_constraint_priorities=creative_constraint_priorities,
        creative_strategy=creative_strategy,
        creative_techniques=creative_techniques,
        runtime_capabilities=runtime_capabilities,
        creative_tradeoffs=creative_tradeoffs,
        creative_quality_prediction=creative_quality_prediction,
        symbolic_narrative=symbolic_narrative,
        creative_composition=creative_composition,
        procedural_structure=procedural_structure,
        generative_structure=generative_structure,
        semantic_motif=semantic_motif,
        emotional_consistency=emotional_consistency,
        cross_modality=cross_modality,
        audio_visual_scene=audio_visual_scene,
    )
    missing_risks = _missing_dependency_risks(
        route_decision=route_decision,
        artifact_plan=artifact_plan,
        runtime_capabilities=runtime_capabilities,
        generative_structure=generative_structure,
    )
    conflicts = _dependency_conflicts(
        artifact_plan=artifact_plan,
        creative_constraints=creative_constraints,
        creative_tradeoffs=creative_tradeoffs,
    )
    blocking = _blocking_dependencies(artifact_plan, missing_risks)
    soft = _soft_dependencies(optional_upstream, artifact_plan)
    guidance = _prompt_guidance(blocking)
    return ArtifactDependencyGraph(
        primary_artifact_node_id="primary_artifact",
        artifact_nodes=tuple(nodes),
        dependency_edges=tuple(edges),
        required_upstream_metadata=_required_upstream_metadata(artifact_plan),
        optional_upstream_metadata=optional_upstream,
        blocking_dependencies=blocking,
        soft_dependencies=soft,
        runtime_facing_dependencies=_runtime_facing_dependencies(
            artifact_plan=artifact_plan,
            creative_plan=creative_plan,
            runtime_capabilities=runtime_capabilities,
        ),
        prompt_facing_dependencies=_prompt_facing_dependencies(artifact_plan, guidance),
        downstream_consumers=(
            "prompt_renderer",
            "creative_assistant_director",
            "creative_reasoning_engine",
            "workflow_serialization",
            "final_payload",
            "nextjs_stream_hydration",
        ),
        missing_dependency_risks=missing_risks,
        dependency_conflicts=conflicts,
        hitl_questions=_hitl_questions(
            artifact_plan=artifact_plan,
            blocking_dependencies=blocking,
            missing_dependency_risks=missing_risks,
            dependency_conflicts=conflicts,
        ),
        prompt_guidance=guidance,
        evidence=_evidence(
            request=request,
            route_decision=route_decision,
            artifact_plan=artifact_plan,
            optional_upstream=optional_upstream,
        ),
    )


def artifact_dependency_graph_prompt_lines(
    graph: ArtifactDependencyGraph,
) -> tuple[str, ...]:
    """Render dependency graph metadata as compact prompt guidance."""

    lines = [
        f"Authority boundary: {graph.authority_boundary}",
        f"Primary artifact node: {graph.primary_artifact_node_id}.",
    ]
    lines.extend(
        (
            "Artifact dependency node: "
            f"{node.node_id} ({node.node_type}, {node.status}) - {node.summary}"
        )
        for node in graph.artifact_nodes
    )
    lines.extend(
        (
            "Artifact dependency edge: "
            f"{edge.source_node_id} -> {edge.target_node_id} "
            f"({edge.relationship}, {edge.strength}) - {edge.rationale}"
        )
        for edge in graph.dependency_edges
    )
    lines.extend(
        f"Required upstream metadata: {item}"
        for item in graph.required_upstream_metadata
    )
    lines.extend(
        f"Optional upstream metadata: {item}"
        for item in graph.optional_upstream_metadata
    )
    lines.extend(
        f"Blocking artifact dependency: {item}"
        for item in graph.blocking_dependencies
    )
    lines.extend(
        f"Soft artifact dependency: {item}" for item in graph.soft_dependencies
    )
    lines.extend(
        f"Runtime-facing artifact dependency: {item}"
        for item in graph.runtime_facing_dependencies
    )
    lines.extend(
        f"Prompt-facing artifact dependency: {item}"
        for item in graph.prompt_facing_dependencies
    )
    lines.extend(
        f"Downstream artifact dependency consumer: {item}"
        for item in graph.downstream_consumers
    )
    lines.extend(
        f"Missing artifact dependency risk: {item}"
        for item in graph.missing_dependency_risks
    )
    lines.extend(
        f"Artifact dependency conflict: {item}"
        for item in graph.dependency_conflicts
    )
    lines.extend(
        f"HITL artifact dependency question: {item}"
        for item in graph.hitl_questions
    )
    lines.extend(
        f"Artifact dependency guidance: {item}" for item in graph.prompt_guidance
    )
    return tuple(lines[:56])


def _artifact_nodes(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    artifact_plan: ArtifactPlan | None,
) -> list[ArtifactDependencyNode]:
    nodes = [
        ArtifactDependencyNode(
            node_id="primary_artifact",
            label="Primary planned artifact",
            node_type="planned_artifact",
            status="available" if artifact_plan is not None else "inferred",
            summary=_primary_artifact_summary(request, route_decision, artifact_plan),
            evidence=_primary_artifact_evidence(route_decision, artifact_plan),
        )
    ]
    if artifact_plan is None:
        nodes.append(
            ArtifactDependencyNode(
                node_id="prompt_guidance",
                label="Fallback prompt dependency guidance",
                node_type="prompt_guidance",
                status="inferred",
                summary="Use the request and route as the only dependency inputs.",
                evidence=("Artifact Plan unavailable.",),
            )
        )
        return nodes

    nodes.extend(
        (
            _node(
                "required_components",
                "Required artifact components",
                "required_component",
                artifact_plan.required_components,
            ),
            _node(
                "runtime_requirements",
                "Runtime-facing requirements",
                "runtime_requirement",
                artifact_plan.runtime_requirements,
            ),
            _node(
                "creative_inputs",
                "Creative metadata inputs",
                "creative_metadata",
                artifact_plan.creative_dependencies,
            ),
            _node(
                "generative_inputs",
                "Generative metadata inputs",
                "generative_metadata",
                artifact_plan.generative_dependencies,
            ),
            _node(
                "output_structure",
                "Expected output structure",
                "output_structure",
                artifact_plan.expected_output_structure,
            ),
            _node(
                "prompt_guidance",
                "Prompt-facing guidance",
                "prompt_guidance",
                artifact_plan.prompt_guidance,
            ),
            ArtifactDependencyNode(
                node_id="downstream_consumers",
                label="Downstream workflow consumers",
                node_type="downstream_consumer",
                status="available",
                summary=(
                    "Prompt rendering, Director, Reasoning, serialization, "
                    "final payload, and stream hydration consume this graph."
                ),
                evidence=("Metadata-only downstream consumption.",),
            ),
        )
    )
    return nodes


def _dependency_edges(
    nodes: list[ArtifactDependencyNode],
    artifact_plan: ArtifactPlan | None,
) -> list[ArtifactDependencyEdge]:
    node_ids = {node.node_id for node in nodes}
    edges: list[ArtifactDependencyEdge] = []
    for source, relationship, strength, rationale in (
        (
            "required_components",
            "requires",
            "required",
            "The primary artifact must satisfy its required components.",
        ),
        (
            "runtime_requirements",
            "requires",
            "required",
            "Runtime-facing requirements constrain artifact implementation notes.",
        ),
        (
            "creative_inputs",
            "informs",
            "required",
            "Creative metadata shapes artifact intent and hierarchy.",
        ),
        (
            "generative_inputs",
            "informs",
            "required",
            "Generative metadata shapes procedural modules and motifs.",
        ),
        (
            "output_structure",
            "requires",
            "required",
            "Expected output structure determines how the artifact is returned.",
        ),
        (
            "prompt_guidance",
            "feeds_prompt",
            "soft",
            "Prompt guidance informs generation without changing execution.",
        ),
    ):
        if source in node_ids:
            edges.append(
                ArtifactDependencyEdge(
                    source_node_id=source,
                    target_node_id="primary_artifact",
                    relationship=relationship,
                    strength=strength,
                    rationale=rationale,
                )
            )
    if "downstream_consumers" in node_ids:
        edges.append(
            ArtifactDependencyEdge(
                source_node_id="primary_artifact",
                target_node_id="downstream_consumers",
                relationship="consumed_by",
                strength="soft",
                rationale=(
                    "Downstream workflow consumers inspect dependency metadata "
                    "without changing artifact execution."
                ),
            )
        )
    if artifact_plan is None:
        edges.append(
            ArtifactDependencyEdge(
                source_node_id="prompt_guidance",
                target_node_id="primary_artifact",
                relationship="soft_informs",
                strength="soft",
                rationale="Fallback guidance is advisory when Artifact Plan is absent.",
            )
        )
    return edges


def _required_upstream_metadata(
    artifact_plan: ArtifactPlan | None,
) -> tuple[str, ...]:
    status = "available" if artifact_plan is not None else "missing"
    return ("assistant_request:available", f"artifact_plan:{status}")


def _optional_upstream_metadata(
    *,
    route_decision: RouteDecision | None,
    creative_translation: CreativeTranslation | None,
    creative_intent: CreativeIntentDecomposition | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
    creative_plan: CreativeExecutionPlan | None,
    creative_constraints: CreativeConstraintSolution | None,
    creative_constraint_priorities: CreativeConstraintPrioritization | None,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
    creative_tradeoffs: CreativeTradeoffProfile | None,
    creative_quality_prediction: CreativeQualityPrediction | None,
    symbolic_narrative: SymbolicNarrativePlan | None,
    creative_composition: CreativeCompositionPlan | None,
    procedural_structure: ProceduralStructurePlan | None,
    generative_structure: GenerativeStructureBlueprint | None,
    semantic_motif: SemanticMotifSystem | None,
    emotional_consistency: EmotionalConsistencyProfile | None,
    cross_modality: CrossModalityCompositionProfile | None,
    audio_visual_scene: AudioVisualSceneProfile | None,
) -> tuple[str, ...]:
    items: list[str] = []
    for name, value in (
        ("route_decision", route_decision),
        ("creative_translation", creative_translation),
        ("creative_intent", creative_intent),
        ("creative_hierarchy", creative_hierarchy),
        ("creative_plan", creative_plan),
        ("creative_constraints", creative_constraints),
        ("creative_constraint_priorities", creative_constraint_priorities),
        ("creative_strategy", creative_strategy),
        ("creative_techniques", creative_techniques),
        ("runtime_capabilities", runtime_capabilities),
        ("creative_tradeoffs", creative_tradeoffs),
        ("creative_quality_prediction", creative_quality_prediction),
        ("symbolic_narrative", symbolic_narrative),
        ("creative_composition", creative_composition),
        ("procedural_structure", procedural_structure),
        ("generative_structure", generative_structure),
        ("semantic_motif", semantic_motif),
        ("emotional_consistency", emotional_consistency),
        ("cross_modality", cross_modality),
        ("audio_visual_scene", audio_visual_scene),
    ):
        if value is not None:
            items.append(f"{name}:available")
    return tuple(items[:18])


def _blocking_dependencies(
    artifact_plan: ArtifactPlan | None,
    missing_dependency_risks: tuple[str, ...],
) -> tuple[str, ...]:
    if artifact_plan is None:
        return (
            "Artifact Plan metadata is required before full artifact dependency "
            "mapping can be trusted.",
        )
    return tuple(
        risk
        for risk in missing_dependency_risks
        if "required" in risk.lower() or "missing" in risk.lower()
    )[:8]


def _soft_dependencies(
    optional_upstream_metadata: tuple[str, ...],
    artifact_plan: ArtifactPlan | None,
) -> tuple[str, ...]:
    dependencies = [
        f"Use {item.removesuffix(':available')} as non-blocking context."
        for item in optional_upstream_metadata[:8]
    ]
    if artifact_plan is not None:
        dependencies.extend(artifact_plan.prompt_guidance[:2])
    return _dedupe(dependencies)[:10]


def _runtime_facing_dependencies(
    *,
    artifact_plan: ArtifactPlan | None,
    creative_plan: CreativeExecutionPlan | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
) -> tuple[str, ...]:
    dependencies: list[str] = []
    if artifact_plan is not None:
        dependencies.extend(artifact_plan.runtime_requirements)
    if creative_plan is not None and creative_plan.recommended_runtime is not None:
        dependencies.append(
            f"Execution plan runtime hint: {creative_plan.recommended_runtime}."
        )
    if runtime_capabilities is not None:
        dependencies.append(
            "Inspected runtime candidates remain advisory: "
            + ", ".join(runtime_capabilities.likely_candidates)
            + "."
        )
    return _dedupe(dependencies)[:10]


def _prompt_facing_dependencies(
    artifact_plan: ArtifactPlan | None,
    graph_guidance: tuple[str, ...],
) -> tuple[str, ...]:
    dependencies = list(graph_guidance)
    if artifact_plan is not None:
        dependencies.extend(artifact_plan.prompt_guidance[:3])
        dependencies.extend(artifact_plan.expected_output_structure[:3])
    return _dedupe(dependencies)[:10]


def _missing_dependency_risks(
    *,
    route_decision: RouteDecision | None,
    artifact_plan: ArtifactPlan | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
    generative_structure: GenerativeStructureBlueprint | None,
) -> tuple[str, ...]:
    risks: list[str] = []
    if artifact_plan is None:
        risks.append(
            "Artifact Plan is missing; dependency graph falls back to request scope."
        )
    else:
        risks.extend(
            f"Artifact Plan missing information: {item}"
            for item in artifact_plan.missing_information[:4]
        )
        if artifact_plan.generative_dependencies and generative_structure is None:
            risks.append(
                "Generative dependencies are present without a generative "
                "structure blueprint."
            )
        if artifact_plan.runtime_requirements and runtime_capabilities is None:
            risks.append(
                "Runtime-facing dependencies are present without runtime "
                "capability metadata."
            )
    if route_decision is None:
        risks.append("Route metadata is unavailable for dependency interpretation.")
    return _dedupe(risks)[:10]


def _dependency_conflicts(
    *,
    artifact_plan: ArtifactPlan | None,
    creative_constraints: CreativeConstraintSolution | None,
    creative_tradeoffs: CreativeTradeoffProfile | None,
) -> tuple[str, ...]:
    conflicts: list[str] = []
    if creative_constraints is not None:
        conflicts.extend(creative_constraints.conflicts[:3])
    if creative_tradeoffs is not None:
        conflicts.extend(
            f"{item.source_axis} vs {item.target_axis}: {item.summary}"
            for item in creative_tradeoffs.primary_tradeoffs[:3]
        )
    if artifact_plan is not None:
        if (
            artifact_plan.artifact_family == "audiovisual_scene"
            and not any("audio" in item.lower() for item in artifact_plan.evidence)
        ):
            conflicts.append(
                "Audio-visual artifact dependency may exceed visual-only "
                "runtime support."
            )
        if (
            artifact_plan.artifact_type == "design_spec"
            and artifact_plan.runtime_requirements
        ):
            conflicts.append(
                "Design artifact has runtime-facing dependencies that should "
                "remain advisory."
            )
    return _dedupe(conflicts)[:10]


def _hitl_questions(
    *,
    artifact_plan: ArtifactPlan | None,
    blocking_dependencies: tuple[str, ...],
    missing_dependency_risks: tuple[str, ...],
    dependency_conflicts: tuple[str, ...],
) -> tuple[str, ...]:
    questions: list[str] = []
    if artifact_plan is not None:
        questions.extend(artifact_plan.hitl_questions[:3])
    questions.extend(
        f"Should we resolve this blocking artifact dependency: {item}"
        for item in blocking_dependencies[:2]
    )
    questions.extend(
        f"Should we resolve this missing artifact dependency risk: {item}"
        for item in missing_dependency_risks[:2]
    )
    questions.extend(
        f"Should this artifact dependency conflict constrain generation: {item}"
        for item in dependency_conflicts[:2]
    )
    return _dedupe(questions)[:8]


def _prompt_guidance(
    blocking_dependencies: tuple[str, ...],
) -> tuple[str, ...]:
    guidance = [
        "Use the Artifact Dependency Graph as dependency metadata only.",
        (
            "Satisfy required and blocking artifact dependencies before optional "
            "or soft dependencies."
        ),
        (
            "Keep runtime-facing dependencies advisory; do not auto-select "
            "runtimes, providers, models, or preview behavior."
        ),
        (
            "Keep prompt-facing dependencies visible in artifact labels, code "
            "comments, or concise setup notes."
        ),
        (
            "Do not treat dependency graph metadata as multi-artifact strategy, "
            "artifact critique, artifact refinement, runtime repair, or execution."
        ),
    ]
    if blocking_dependencies:
        guidance.append(
            "Surface blocking dependency questions before expanding scope."
        )
    return tuple(guidance[:8])


def _evidence(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    artifact_plan: ArtifactPlan | None,
    optional_upstream: tuple[str, ...],
) -> tuple[str, ...]:
    evidence = [f"Mode: {request.mode.value}."]
    if route_decision is not None:
        evidence.append(f"Route: {route_decision.route.value}.")
        if route_decision.domains:
            evidence.append(
                "Domains: "
                + ", ".join(domain.value for domain in route_decision.domains)
                + "."
            )
    if artifact_plan is not None:
        evidence.append(
            "Artifact plan: "
            f"{artifact_plan.artifact_type}; {artifact_plan.artifact_family}."
        )
        evidence.append(
            f"Required artifact components: {len(artifact_plan.required_components)}."
        )
    evidence.append(f"Available optional upstream metadata: {len(optional_upstream)}.")
    return _dedupe(evidence)[:12]


def _node(
    node_id: str,
    label: str,
    node_type: DependencyNodeType,
    values: tuple[str, ...],
) -> ArtifactDependencyNode:
    status: DependencyNodeStatus = "available" if values else "missing"
    summary = "; ".join(values[:3]) if values else f"No {label.lower()} available."
    return ArtifactDependencyNode(
        node_id=node_id,
        label=label,
        node_type=node_type,
        status=status,
        summary=summary,
        evidence=tuple(values[:4]),
    )


def _primary_artifact_summary(
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    artifact_plan: ArtifactPlan | None,
) -> str:
    if artifact_plan is not None:
        return (
            f"{artifact_plan.artifact_type}/{artifact_plan.artifact_family}: "
            f"{artifact_plan.primary_artifact_intent}"
        )
    route = route_decision.route.value if route_decision is not None else "inferred"
    return (
        f"Request-level artifact inferred for route {route}: "
        f"{_clip(request.query, 300)}"
    )


def _primary_artifact_evidence(
    route_decision: RouteDecision | None,
    artifact_plan: ArtifactPlan | None,
) -> tuple[str, ...]:
    evidence: list[str] = []
    if route_decision is not None:
        evidence.append(f"Route: {route_decision.route.value}.")
    if artifact_plan is not None:
        evidence.extend(artifact_plan.evidence[:4])
    return tuple(evidence)
