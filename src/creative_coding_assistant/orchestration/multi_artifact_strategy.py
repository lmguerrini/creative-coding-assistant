"""Bounded Multi-Artifact Strategy for V3.3 workflows."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import AssistantRequest
from creative_coding_assistant.orchestration._metadata_utils import _dedupe
from creative_coding_assistant.orchestration.artifact_capability_matrix import (
    ArtifactCapabilityMatrix,
)
from creative_coding_assistant.orchestration.artifact_dependency_graph import (
    ArtifactDependencyGraph,
)
from creative_coding_assistant.orchestration.artifact_planner import (
    ArtifactFamily,
    ArtifactPlan,
    ArtifactType,
)
from creative_coding_assistant.orchestration.creative_constraints import (
    CreativeConstraintSolution,
)
from creative_coding_assistant.orchestration.creative_planning import (
    CreativeExecutionPlan,
)
from creative_coding_assistant.orchestration.creative_tradeoffs import (
    CreativeTradeoffProfile,
)
from creative_coding_assistant.orchestration.routing import RouteDecision
from creative_coding_assistant.orchestration.runtime_capabilities import (
    RuntimeCapabilityId,
    RuntimeCapabilityProfile,
)
from creative_coding_assistant.orchestration.runtime_compatibility import (
    RuntimeCompatibilityProfile,
)

ArtifactStrategyRole = Literal["primary", "supporting", "optional"]
ArtifactStrategyPriority = Literal["critical", "high", "medium", "low"]
ArtifactStrategyAction = Literal["produce", "separate", "document", "handoff"]
ArtifactStrategyCombinationMode = Literal[
    "primary_with_supporting_sections",
    "separated_parallel_sections",
    "defer_combination",
]

MULTI_ARTIFACT_STRATEGY_AUTHORITY_BOUNDARY = (
    "The Multi-Artifact Strategy plans artifact ordering, grouping, "
    "separation, combination, dependencies, handoffs, runtime-aware notes, "
    "capability-aware notes, risks, missing information, HITL questions, "
    "and prompt guidance as inspectable metadata only; it does not generate "
    "artifacts, critique artifacts, refine artifacts, merge artifacts, "
    "implement Artifact Export Intelligence, execute runtimes, auto-select "
    "runtimes, route providers or models, change preview behavior, implement "
    "V4 multi-agent behavior, or implement V5 execution optimization."
)


class ArtifactStrategyArtifact(BaseModel):
    """One planned artifact role in a multi-artifact response strategy."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    artifact_id: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=140)
    role: ArtifactStrategyRole
    artifact_type: ArtifactType
    artifact_family: ArtifactFamily
    priority: ArtifactStrategyPriority
    purpose: str = Field(min_length=1, max_length=300)
    runtime_targets: tuple[RuntimeCapabilityId, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    capability_targets: tuple[RuntimeCapabilityId, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    depends_on: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    handoff_points: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=8)


class ArtifactStrategySequenceStep(BaseModel):
    """One ordered step in the artifact production strategy."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    step_id: str = Field(min_length=1, max_length=80)
    order: int = Field(ge=1, le=12)
    artifact_id: str = Field(min_length=1, max_length=80)
    action: ArtifactStrategyAction
    rationale: str = Field(min_length=1, max_length=300)
    depends_on: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=6)


class ArtifactStrategyPriorityEntry(BaseModel):
    """Priority rationale for one artifact role."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    artifact_id: str = Field(min_length=1, max_length=80)
    priority: ArtifactStrategyPriority
    rationale: str = Field(min_length=1, max_length=300)


class ArtifactStrategyGroup(BaseModel):
    """Grouping metadata for related artifact roles."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    group_id: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=140)
    artifact_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    grouping_rationale: str = Field(min_length=1, max_length=300)
    separation_rationale: str = Field(min_length=1, max_length=300)


class MultiArtifactStrategy(BaseModel):
    """Inspectable metadata-only strategy for multi-artifact workflows."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["multi_artifact_strategy"] = "multi_artifact_strategy"
    artifact_strategy_summary: str = Field(min_length=1, max_length=420)
    primary_artifact: ArtifactStrategyArtifact
    supporting_artifacts: tuple[ArtifactStrategyArtifact, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    artifact_sequence: tuple[ArtifactStrategySequenceStep, ...] = Field(
        min_length=1,
        max_length=8,
    )
    artifact_priority: tuple[ArtifactStrategyPriorityEntry, ...] = Field(
        min_length=1,
        max_length=8,
    )
    artifact_grouping: tuple[ArtifactStrategyGroup, ...] = Field(
        min_length=1,
        max_length=4,
    )
    artifact_separation_strategy: tuple[str, ...] = Field(
        min_length=1,
        max_length=8,
    )
    artifact_combination_strategy: tuple[str, ...] = Field(
        min_length=1,
        max_length=8,
    )
    artifact_dependency_order: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=10,
    )
    artifact_handoff_points: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=10,
    )
    runtime_aware_artifact_strategy: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=10,
    )
    capability_aware_artifact_strategy: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=10,
    )
    combination_mode: ArtifactStrategyCombinationMode
    risk_areas: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    missing_information: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    hitl_questions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(
        default=MULTI_ARTIFACT_STRATEGY_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=12)


def derive_multi_artifact_strategy(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_capabilities: RuntimeCapabilityProfile | None = None,
    runtime_compatibility: RuntimeCompatibilityProfile | None = None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None = None,
    creative_plan: CreativeExecutionPlan | None = None,
    creative_constraints: CreativeConstraintSolution | None = None,
    creative_tradeoffs: CreativeTradeoffProfile | None = None,
) -> MultiArtifactStrategy:
    """Derive multi-artifact strategy metadata without generating artifacts."""

    primary = _primary_artifact(
        artifact_plan=artifact_plan,
        runtime_compatibility=runtime_compatibility,
        artifact_capability_matrix=artifact_capability_matrix,
    )
    supporting = _supporting_artifacts(
        artifact_plan=artifact_plan,
        artifact_dependency_graph=artifact_dependency_graph,
        runtime_compatibility=runtime_compatibility,
        artifact_capability_matrix=artifact_capability_matrix,
        creative_tradeoffs=creative_tradeoffs,
    )
    artifacts = (primary, *supporting)
    sequence = _artifact_sequence(artifacts)
    priorities = _artifact_priorities(artifacts)
    groups = _artifact_groups(primary, supporting)
    risks = _risk_areas(
        artifact_plan=artifact_plan,
        artifact_dependency_graph=artifact_dependency_graph,
        runtime_compatibility=runtime_compatibility,
        artifact_capability_matrix=artifact_capability_matrix,
        creative_constraints=creative_constraints,
        creative_tradeoffs=creative_tradeoffs,
    )
    missing = _missing_information(
        route_decision=route_decision,
        artifact_plan=artifact_plan,
        artifact_dependency_graph=artifact_dependency_graph,
        runtime_compatibility=runtime_compatibility,
        artifact_capability_matrix=artifact_capability_matrix,
        creative_plan=creative_plan,
    )
    return MultiArtifactStrategy(
        artifact_strategy_summary=_strategy_summary(primary, supporting),
        primary_artifact=primary,
        supporting_artifacts=supporting,
        artifact_sequence=sequence,
        artifact_priority=priorities,
        artifact_grouping=groups,
        artifact_separation_strategy=_separation_strategy(primary, supporting),
        artifact_combination_strategy=_combination_strategy(supporting),
        artifact_dependency_order=_dependency_order(
            artifact_dependency_graph,
            sequence,
        ),
        artifact_handoff_points=_handoff_points(sequence),
        runtime_aware_artifact_strategy=_runtime_strategy(
            runtime_capabilities=runtime_capabilities,
            runtime_compatibility=runtime_compatibility,
            creative_plan=creative_plan,
        ),
        capability_aware_artifact_strategy=_capability_strategy(
            artifact_capability_matrix,
        ),
        combination_mode=_combination_mode(supporting),
        risk_areas=risks,
        missing_information=missing,
        hitl_questions=_hitl_questions(
            missing=missing,
            risks=risks,
            supporting=supporting,
        ),
        prompt_guidance=_prompt_guidance(supporting),
        evidence=_evidence(
            request=request,
            route_decision=route_decision,
            artifact_plan=artifact_plan,
            artifact_dependency_graph=artifact_dependency_graph,
            runtime_compatibility=runtime_compatibility,
            artifact_capability_matrix=artifact_capability_matrix,
            supporting=supporting,
        ),
    )


def multi_artifact_strategy_prompt_lines(
    strategy: MultiArtifactStrategy,
) -> tuple[str, ...]:
    """Render multi-artifact strategy metadata as compact prompt guidance."""

    lines = [
        f"Authority boundary: {strategy.authority_boundary}",
        f"Strategy summary: {strategy.artifact_strategy_summary}",
        f"Primary artifact: {strategy.primary_artifact.title}.",
        f"Combination mode: {strategy.combination_mode}.",
        (
            "Supporting artifacts: "
            + _artifact_titles(strategy.supporting_artifacts)
            + "."
        ),
    ]
    lines.extend(
        f"Artifact sequence: {step.order}. {step.artifact_id} "
        f"{step.action}; {step.rationale}"
        for step in strategy.artifact_sequence
    )
    lines.extend(
        f"Artifact priority: {item.artifact_id} {item.priority}; {item.rationale}"
        for item in strategy.artifact_priority
    )
    lines.extend(
        f"Artifact group: {group.label}; {group.grouping_rationale}"
        for group in strategy.artifact_grouping
    )
    lines.extend(
        f"Artifact separation strategy: {item}"
        for item in strategy.artifact_separation_strategy
    )
    lines.extend(
        f"Artifact combination strategy: {item}"
        for item in strategy.artifact_combination_strategy
    )
    lines.extend(
        f"Artifact dependency order: {item}"
        for item in strategy.artifact_dependency_order
    )
    lines.extend(
        f"Artifact handoff point: {item}" for item in strategy.artifact_handoff_points
    )
    lines.extend(
        f"Runtime-aware artifact strategy: {item}"
        for item in strategy.runtime_aware_artifact_strategy
    )
    lines.extend(
        f"Capability-aware artifact strategy: {item}"
        for item in strategy.capability_aware_artifact_strategy
    )
    lines.extend(f"Multi-artifact risk: {item}" for item in strategy.risk_areas)
    lines.extend(
        f"Missing multi-artifact information: {item}"
        for item in strategy.missing_information
    )
    lines.extend(
        f"HITL multi-artifact question: {item}" for item in strategy.hitl_questions
    )
    lines.extend(
        f"Multi-artifact guidance: {item}" for item in strategy.prompt_guidance
    )
    return tuple(lines[:64])


def _primary_artifact(
    *,
    artifact_plan: ArtifactPlan | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
) -> ArtifactStrategyArtifact:
    artifact_type: ArtifactType = (
        artifact_plan.artifact_type if artifact_plan is not None else "explanation"
    )
    artifact_family: ArtifactFamily = (
        artifact_plan.artifact_family
        if artifact_plan is not None
        else "creative_coding_response"
    )
    return ArtifactStrategyArtifact(
        artifact_id="primary_artifact",
        title=_primary_title(artifact_plan),
        role="primary",
        artifact_type=artifact_type,
        artifact_family=artifact_family,
        priority="critical",
        purpose=(
            "Deliver the main requested creative-coding output before any "
            "supporting explanation or metadata."
        ),
        runtime_targets=_runtime_targets(runtime_compatibility),
        capability_targets=_capability_targets(artifact_capability_matrix),
        depends_on=(),
        handoff_points=(
            "Primary output hands off to supporting metadata only after the "
            "main artifact shape is clear.",
        ),
        evidence=_dedupe(
            (
                f"Artifact type: {artifact_type}.",
                f"Artifact family: {artifact_family}.",
            )
        ),
    )


def _supporting_artifacts(
    *,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    creative_tradeoffs: CreativeTradeoffProfile | None,
) -> tuple[ArtifactStrategyArtifact, ...]:
    supporting: list[ArtifactStrategyArtifact] = []
    family = (
        artifact_plan.artifact_family
        if artifact_plan is not None
        else "creative_coding_response"
    )
    if artifact_dependency_graph is not None:
        supporting.append(
            _support_artifact(
                artifact_id="dependency_notes",
                title="Dependency and output-structure notes",
                family=family,
                priority="high",
                purpose=(
                    "Keep required components, runtime-facing dependencies, "
                    "prompt-facing dependencies, and downstream assumptions "
                    "inspectable after the primary artifact."
                ),
                handoff=(
                    "Hand off from primary artifact to dependency notes when "
                    "the response needs to explain ordering or required sections."
                ),
                evidence=artifact_dependency_graph.prompt_guidance[:2],
            )
        )
    if runtime_compatibility is not None:
        supporting.append(
            _support_artifact(
                artifact_id="runtime_notes",
                title="Runtime compatibility notes",
                family=family,
                priority="medium",
                purpose=(
                    "Surface compatible, partial, and unsupported runtime "
                    "metadata without changing runtime selection or execution."
                ),
                handoff=(
                    "Hand off to runtime notes after the primary artifact "
                    "and dependency notes establish artifact shape."
                ),
                runtime_targets=runtime_compatibility.preferred_runtimes,
                evidence=runtime_compatibility.prompt_guidance[:2],
            )
        )
    if artifact_capability_matrix is not None:
        supporting.append(
            _support_artifact(
                artifact_id="capability_notes",
                title="Artifact capability notes",
                family=family,
                priority="medium",
                purpose=(
                    "Explain target strengths, weaknesses, fit dimensions, "
                    "and unsupported or risky capabilities as planning "
                    "metadata only."
                ),
                handoff=(
                    "Hand off to capability notes when target fit or target "
                    "limitations need explicit caveats."
                ),
                runtime_targets=artifact_capability_matrix.strongest_targets,
                evidence=artifact_capability_matrix.prompt_guidance[:2],
            )
        )
    if creative_tradeoffs is not None and creative_tradeoffs.primary_tradeoffs:
        supporting.append(
            _support_artifact(
                artifact_id="tradeoff_notes",
                title="Implementation trade-off notes",
                family=family,
                priority="low",
                purpose=(
                    "Keep trade-off consequences visible without producing "
                    "extra artifacts or expanding scope."
                ),
                handoff=(
                    "Hand off to trade-off notes only when complexity, "
                    "performance, or fidelity caveats materially affect the "
                    "primary artifact."
                ),
                evidence=tuple(
                    item.summary for item in creative_tradeoffs.primary_tradeoffs[:2]
                ),
            )
        )
    return tuple(supporting[:6])


def _support_artifact(
    *,
    artifact_id: str,
    title: str,
    family: ArtifactFamily,
    priority: ArtifactStrategyPriority,
    purpose: str,
    handoff: str,
    runtime_targets: tuple[RuntimeCapabilityId, ...] = (),
    evidence: tuple[str, ...] = (),
) -> ArtifactStrategyArtifact:
    return ArtifactStrategyArtifact(
        artifact_id=artifact_id,
        title=title,
        role="supporting",
        artifact_type="explanation",
        artifact_family=family,
        priority=priority,
        purpose=purpose,
        runtime_targets=runtime_targets[:4],
        capability_targets=runtime_targets[:4],
        depends_on=("primary_artifact",),
        handoff_points=(handoff,),
        evidence=_dedupe(evidence),
    )


def _artifact_sequence(
    artifacts: tuple[ArtifactStrategyArtifact, ...],
) -> tuple[ArtifactStrategySequenceStep, ...]:
    steps: list[ArtifactStrategySequenceStep] = []
    for index, artifact in enumerate(artifacts, start=1):
        action: ArtifactStrategyAction = (
            "produce" if artifact.role == "primary" else "document"
        )
        steps.append(
            ArtifactStrategySequenceStep(
                step_id=f"step_{index}_{artifact.artifact_id}",
                order=index,
                artifact_id=artifact.artifact_id,
                action=action,
                rationale=_sequence_rationale(artifact),
                depends_on=artifact.depends_on,
                prompt_guidance=_sequence_guidance(artifact),
            )
        )
    return tuple(steps[:8])


def _artifact_priorities(
    artifacts: tuple[ArtifactStrategyArtifact, ...],
) -> tuple[ArtifactStrategyPriorityEntry, ...]:
    return tuple(
        ArtifactStrategyPriorityEntry(
            artifact_id=artifact.artifact_id,
            priority=artifact.priority,
            rationale=(
                "Primary artifact leads the response."
                if artifact.role == "primary"
                else f"{artifact.title} supports the primary artifact."
            ),
        )
        for artifact in artifacts[:8]
    )


def _artifact_groups(
    primary: ArtifactStrategyArtifact,
    supporting: tuple[ArtifactStrategyArtifact, ...],
) -> tuple[ArtifactStrategyGroup, ...]:
    groups = [
        ArtifactStrategyGroup(
            group_id="primary_output_group",
            label="Primary output",
            artifact_ids=(primary.artifact_id,),
            grouping_rationale="Keep the main requested artifact isolated and first.",
            separation_rationale=(
                "Do not bury the primary artifact inside explanatory metadata."
            ),
        )
    ]
    if supporting:
        groups.append(
            ArtifactStrategyGroup(
                group_id="supporting_metadata_group",
                label="Supporting metadata",
                artifact_ids=tuple(item.artifact_id for item in supporting),
                grouping_rationale=(
                    "Keep dependency, runtime, capability, and trade-off notes "
                    "together as support sections."
                ),
                separation_rationale=(
                    "Supporting metadata should not be merged into executable "
                    "artifact code or treated as separate generated outputs."
                ),
            )
        )
    return tuple(groups)


def _separation_strategy(
    primary: ArtifactStrategyArtifact,
    supporting: tuple[ArtifactStrategyArtifact, ...],
) -> tuple[str, ...]:
    strategy = [
        f"Lead with {primary.title} as the only primary artifact.",
        "Use explicit labels or section boundaries for each supporting artifact role.",
        "Keep metadata notes outside runnable code blocks unless explicitly requested.",
    ]
    if supporting:
        strategy.append(
            "Place supporting artifacts after the primary artifact in priority order."
        )
    return tuple(strategy[:8])


def _combination_strategy(
    supporting: tuple[ArtifactStrategyArtifact, ...],
) -> tuple[str, ...]:
    strategy = [
        "Combine artifacts in one response only as separated sections.",
        "Do not merge runtime notes, capability notes, or dependency notes into code.",
        "Do not create extra files or artifact variants unless the user requests them.",
    ]
    if supporting:
        strategy.append(
            "Use supporting artifacts to explain constraints and handoffs, not to "
            "expand implementation scope."
        )
    else:
        strategy.append(
            "No supporting artifacts are required; keep response single-artifact."
        )
    return tuple(strategy[:8])


def _dependency_order(
    graph: ArtifactDependencyGraph | None,
    sequence: tuple[ArtifactStrategySequenceStep, ...],
) -> tuple[str, ...]:
    order = [
        f"{step.order}. {step.artifact_id}"
        + (f" after {', '.join(step.depends_on)}" if step.depends_on else "")
        for step in sequence
    ]
    if graph is not None:
        order.extend(
            f"{edge.source_node_id} {edge.relationship} {edge.target_node_id}"
            for edge in graph.dependency_edges[:4]
        )
    return _dedupe(order)[:10]


def _handoff_points(
    sequence: tuple[ArtifactStrategySequenceStep, ...],
) -> tuple[str, ...]:
    handoffs: list[str] = []
    for step in sequence:
        if step.depends_on:
            handoffs.append(
                f"{', '.join(step.depends_on)} -> {step.artifact_id}: "
                f"{step.rationale}"
            )
    if not handoffs:
        handoffs.append("No supporting artifact handoff is required.")
    return _dedupe(handoffs)[:10]


def _runtime_strategy(
    *,
    runtime_capabilities: RuntimeCapabilityProfile | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    creative_plan: CreativeExecutionPlan | None,
) -> tuple[str, ...]:
    strategy: list[str] = []
    if creative_plan is not None and creative_plan.recommended_runtime is not None:
        strategy.append(
            "Preserve existing planning runtime hint: "
            f"{creative_plan.recommended_runtime}."
        )
    if runtime_capabilities is not None:
        strategy.append(
            "Runtime candidates are non-binding: "
            + ", ".join(runtime_capabilities.likely_candidates)
            + "."
        )
    if runtime_compatibility is not None:
        strategy.append(
            "Preferred compatible runtimes are metadata only: "
            + ", ".join(runtime_compatibility.preferred_runtimes)
            + "."
        )
        if runtime_compatibility.unsupported_runtimes:
            strategy.append(
                "Unsupported runtimes should be caveated, not used as targets: "
                + ", ".join(runtime_compatibility.unsupported_runtimes[:3])
                + "."
            )
    strategy.append(
        "Do not use multi-artifact strategy to auto-select or execute runtimes."
    )
    return _dedupe(strategy)[:10]


def _capability_strategy(
    matrix: ArtifactCapabilityMatrix | None,
) -> tuple[str, ...]:
    if matrix is None:
        return ("Capability-aware artifact strategy is inferred from defaults.",)
    strategy = [
        "Strongest target capabilities are metadata only: "
        + ", ".join(matrix.strongest_targets)
        + ".",
        "Use capability notes to decide what needs explanation, not what to execute.",
    ]
    strategy.extend(matrix.target_strengths[:2])
    strategy.extend(matrix.target_weaknesses[:2])
    strategy.extend(matrix.unsupported_or_risky_capabilities[:2])
    return _dedupe(strategy)[:10]


def _risk_areas(
    *,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    creative_constraints: CreativeConstraintSolution | None,
    creative_tradeoffs: CreativeTradeoffProfile | None,
) -> tuple[str, ...]:
    risks: list[str] = []
    if artifact_plan is not None:
        risks.extend(artifact_plan.implementation_risks[:3])
    if artifact_dependency_graph is not None:
        risks.extend(artifact_dependency_graph.dependency_conflicts[:3])
        risks.extend(artifact_dependency_graph.missing_dependency_risks[:2])
    if runtime_compatibility is not None:
        risks.extend(runtime_compatibility.implementation_risks[:3])
    if artifact_capability_matrix is not None:
        risks.extend(artifact_capability_matrix.capability_risks[:3])
    if creative_constraints is not None:
        risks.extend(creative_constraints.conflicts[:2])
    if creative_tradeoffs is not None:
        risks.extend(creative_tradeoffs.complexity_risks[:2])
        risks.extend(creative_tradeoffs.runtime_risks[:2])
    risks.append("Do not let supporting artifacts expand implementation scope.")
    return _dedupe(risks)[:10]


def _missing_information(
    *,
    route_decision: RouteDecision | None,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    creative_plan: CreativeExecutionPlan | None,
) -> tuple[str, ...]:
    missing: list[str] = []
    if route_decision is None or not route_decision.domains:
        missing.append("Route/domain metadata is inferred or unavailable.")
    if artifact_plan is None:
        missing.append("Artifact Plan metadata is unavailable.")
    else:
        missing.extend(artifact_plan.missing_information[:3])
    if artifact_dependency_graph is None:
        missing.append("Artifact Dependency Graph metadata is unavailable.")
    elif not artifact_dependency_graph.downstream_consumers:
        missing.append("Artifact downstream consumers are unavailable.")
    if runtime_compatibility is None:
        missing.append("Runtime Compatibility Engine metadata is unavailable.")
    if artifact_capability_matrix is None:
        missing.append("Artifact Capability Matrix metadata is unavailable.")
    if creative_plan is None:
        missing.append("Creative plan metadata is unavailable.")
    return _dedupe(missing)[:10]


def _hitl_questions(
    *,
    missing: tuple[str, ...],
    risks: tuple[str, ...],
    supporting: tuple[ArtifactStrategyArtifact, ...],
) -> tuple[str, ...]:
    questions = [
        f"Should we resolve this missing multi-artifact input: {item}"
        for item in missing[:3]
    ]
    questions.extend(
        f"Should this multi-artifact risk constrain response structure: {item}"
        for item in risks[:3]
    )
    if len(supporting) > 2:
        questions.append(
            "Should supporting artifacts be reduced to only dependency "
            "and runtime notes?"
        )
    return tuple(questions[:8])


def _prompt_guidance(
    supporting: tuple[ArtifactStrategyArtifact, ...],
) -> tuple[str, ...]:
    guidance = [
        "Use Multi-Artifact Strategy output as response-structure metadata only.",
        "Lead with the primary artifact before supporting metadata sections.",
        (
            "Do not generate extra artifacts, merge artifacts, execute runtimes, "
            "auto-select runtimes, route providers, or change preview behavior "
            "from this strategy."
        ),
    ]
    if supporting:
        guidance.append(
            "Render supporting artifacts as clearly separated explanatory sections."
        )
    else:
        guidance.append("Keep response single-artifact unless the user asks otherwise.")
    return tuple(guidance[:8])


def _evidence(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    supporting: tuple[ArtifactStrategyArtifact, ...],
) -> tuple[str, ...]:
    evidence = [f"Mode: {request.mode.value}."]
    if route_decision is not None:
        evidence.append(f"Route selected: {route_decision.route.value}.")
    if artifact_plan is not None:
        evidence.append(
            f"Artifact: {artifact_plan.artifact_type}; {artifact_plan.artifact_family}."
        )
    if artifact_dependency_graph is not None:
        evidence.append(
            "Dependency graph: "
            f"{len(artifact_dependency_graph.artifact_nodes)} nodes; "
            f"{len(artifact_dependency_graph.dependency_edges)} edges."
        )
    if runtime_compatibility is not None:
        evidence.append(
            "Runtime compatibility preferred: "
            + ", ".join(runtime_compatibility.preferred_runtimes)
            + "."
        )
    if artifact_capability_matrix is not None:
        evidence.append(
            "Capability targets: "
            + ", ".join(artifact_capability_matrix.strongest_targets)
            + "."
        )
    evidence.append(f"Supporting artifact count: {len(supporting)}.")
    return _dedupe(evidence)[:12]


def _strategy_summary(
    primary: ArtifactStrategyArtifact,
    supporting: tuple[ArtifactStrategyArtifact, ...],
) -> str:
    if supporting:
        return (
            f"Produce {primary.title} first, then separate "
            f"{len(supporting)} supporting metadata section(s) for dependencies, "
            "runtime/capability context, and trade-offs when relevant."
        )
    return f"Produce {primary.title} as a single primary artifact."


def _primary_title(artifact_plan: ArtifactPlan | None) -> str:
    if artifact_plan is None:
        return "Primary creative-coding response"
    return f"Primary {artifact_plan.artifact_family.replace('_', ' ')}"


def _runtime_targets(
    runtime_compatibility: RuntimeCompatibilityProfile | None,
) -> tuple[RuntimeCapabilityId, ...]:
    if runtime_compatibility is None:
        return ()
    return runtime_compatibility.preferred_runtimes[:4]


def _capability_targets(
    matrix: ArtifactCapabilityMatrix | None,
) -> tuple[RuntimeCapabilityId, ...]:
    if matrix is None:
        return ()
    return matrix.strongest_targets[:4]


def _combination_mode(
    supporting: tuple[ArtifactStrategyArtifact, ...],
) -> ArtifactStrategyCombinationMode:
    if not supporting:
        return "defer_combination"
    if len(supporting) <= 2:
        return "primary_with_supporting_sections"
    return "separated_parallel_sections"


def _sequence_rationale(artifact: ArtifactStrategyArtifact) -> str:
    if artifact.role == "primary":
        return "Primary artifact must establish the requested output before support."
    return f"{artifact.title} should support, not replace, the primary artifact."


def _sequence_guidance(
    artifact: ArtifactStrategyArtifact,
) -> tuple[str, ...]:
    if artifact.role == "primary":
        return ("Lead with the primary artifact output.",)
    return (
        f"Render {artifact.title} as a separated supporting section.",
        "Do not treat supporting metadata as a generated artifact variant.",
    )


def _artifact_titles(artifacts: tuple[ArtifactStrategyArtifact, ...]) -> str:
    if not artifacts:
        return "none"
    return ", ".join(artifact.title for artifact in artifacts)
