"""Shared Artifact Intelligence engine contracts for V3.3 metadata."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ArtifactEngineCategory = Literal["artifact_intelligence"]
ArtifactEngineCacheability = Literal[
    "deterministic_per_request",
    "deterministic_with_upstream_metadata",
]
ArtifactEngineParallelizationSupport = Literal[
    "requires_ordered_upstream_metadata",
    "parallel_after_required_inputs",
]

ARTIFACT_ENGINE_CONTRACT_SERIALIZATION_VERSION = "artifact_engine_contract.v1"
ARTIFACT_ENGINE_CONTRACT_REGISTRY_SERIALIZATION_VERSION = (
    "artifact_engine_contract_registry.v1"
)
ARTIFACT_ENGINE_CONTRACT_REGISTRY_AUTHORITY_BOUNDARY = (
    "Artifact Intelligence engine contracts describe metadata interfaces, "
    "dependencies, signals, cost, latency, and future hooks only; they do not "
    "change planning behavior, prompts, workflow ordering, routing, runtime "
    "selection, execution, retries, previews, artifacts, or generated output."
)


class ArtifactEngineCostMetadata(BaseModel):
    """Static estimated cost metadata for an Artifact Intelligence engine."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    relative_cost: Literal["low", "medium"]
    external_provider_calls: bool = False
    cost_basis: str = Field(min_length=1, max_length=260)
    cache_sensitivity: str = Field(min_length=1, max_length=260)


class ArtifactEngineLatencyMetadata(BaseModel):
    """Static estimated latency metadata for an Artifact Intelligence engine."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    relative_latency: Literal["low", "medium"]
    latency_basis: str = Field(min_length=1, max_length=260)
    blocking_inputs: tuple[str, ...] = Field(default_factory=tuple, max_length=16)


class ArtifactIntelligenceEngineContract(BaseModel):
    """Common metadata contract exposed by every Artifact Intelligence engine."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    engine_id: str = Field(min_length=1, max_length=80)
    engine_name: str = Field(min_length=1, max_length=140)
    engine_version: str = Field(min_length=1, max_length=24)
    engine_category: ArtifactEngineCategory = "artifact_intelligence"
    authority_boundary: str = Field(min_length=1, max_length=900)
    required_inputs: tuple[str, ...] = Field(default_factory=tuple, max_length=16)
    optional_inputs: tuple[str, ...] = Field(default_factory=tuple, max_length=24)
    produced_metadata: tuple[str, ...] = Field(min_length=1, max_length=18)
    produced_signals: tuple[str, ...] = Field(min_length=1, max_length=18)
    confidence_signals: tuple[str, ...] = Field(min_length=1, max_length=12)
    ambiguity_signals: tuple[str, ...] = Field(min_length=1, max_length=12)
    risk_signals: tuple[str, ...] = Field(min_length=1, max_length=12)
    escalation_candidates: tuple[str, ...] = Field(min_length=1, max_length=12)
    downstream_dependencies: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=16,
    )
    upstream_dependencies: tuple[str, ...] = Field(default_factory=tuple, max_length=16)
    cacheability: ArtifactEngineCacheability
    parallelization_support: ArtifactEngineParallelizationSupport
    estimated_cost_metadata: ArtifactEngineCostMetadata
    estimated_latency_metadata: ArtifactEngineLatencyMetadata
    serialization_version: Literal["artifact_engine_contract.v1"] = (
        ARTIFACT_ENGINE_CONTRACT_SERIALIZATION_VERSION
    )
    future_agent_hooks: tuple[str, ...] = Field(min_length=1, max_length=12)
    future_execution_hooks: tuple[str, ...] = Field(min_length=1, max_length=12)


class ArtifactIntelligenceEngineContractRegistry(BaseModel):
    """Stable registry of Artifact Intelligence engine metadata contracts."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["artifact_intelligence_engine_contract_registry"] = (
        "artifact_intelligence_engine_contract_registry"
    )
    engine_category: ArtifactEngineCategory = "artifact_intelligence"
    serialization_version: Literal[
        "artifact_engine_contract_registry.v1"
    ] = ARTIFACT_ENGINE_CONTRACT_REGISTRY_SERIALIZATION_VERSION
    authority_boundary: str = Field(
        default=ARTIFACT_ENGINE_CONTRACT_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    engine_contracts: tuple[ArtifactIntelligenceEngineContract, ...] = Field(
        min_length=10,
        max_length=10,
    )
    engine_ids: tuple[str, ...] = Field(min_length=10, max_length=10)
    contract_count: int = Field(ge=10, le=10)
    future_agent_consumers: tuple[str, ...] = Field(min_length=1, max_length=8)


def artifact_intelligence_engine_contracts() -> (
    ArtifactIntelligenceEngineContractRegistry
):
    """Return the static Artifact Intelligence contract registry."""

    return ARTIFACT_INTELLIGENCE_ENGINE_CONTRACT_REGISTRY


def artifact_intelligence_engine_contract_by_id(
    engine_id: str,
) -> ArtifactIntelligenceEngineContract | None:
    """Return one engine contract by id without changing engine behavior."""

    for contract in ARTIFACT_INTELLIGENCE_ENGINE_CONTRACTS:
        if contract.engine_id == engine_id:
            return contract
    return None


def _contract(
    *,
    engine_id: str,
    engine_name: str,
    authority_boundary: str,
    required_inputs: tuple[str, ...],
    optional_inputs: tuple[str, ...] = (),
    produced_metadata: tuple[str, ...],
    produced_signals: tuple[str, ...],
    confidence_signals: tuple[str, ...],
    ambiguity_signals: tuple[str, ...],
    risk_signals: tuple[str, ...],
    escalation_candidates: tuple[str, ...],
    downstream_dependencies: tuple[str, ...],
    upstream_dependencies: tuple[str, ...],
    cacheability: ArtifactEngineCacheability = "deterministic_with_upstream_metadata",
    parallelization_support: ArtifactEngineParallelizationSupport = (
        "requires_ordered_upstream_metadata"
    ),
    relative_cost: Literal["low", "medium"] = "low",
    relative_latency: Literal["low", "medium"] = "low",
    future_execution_hooks: tuple[str, ...] = (
        "v5_execution_optimization_readiness",
        "v6_learning_engine_feedback",
    ),
) -> ArtifactIntelligenceEngineContract:
    return ArtifactIntelligenceEngineContract(
        engine_id=engine_id,
        engine_name=engine_name,
        engine_version="v3.3",
        authority_boundary=authority_boundary,
        required_inputs=required_inputs,
        optional_inputs=optional_inputs,
        produced_metadata=produced_metadata,
        produced_signals=produced_signals,
        confidence_signals=confidence_signals,
        ambiguity_signals=ambiguity_signals,
        risk_signals=risk_signals,
        escalation_candidates=escalation_candidates,
        downstream_dependencies=downstream_dependencies,
        upstream_dependencies=upstream_dependencies,
        cacheability=cacheability,
        parallelization_support=parallelization_support,
        estimated_cost_metadata=ArtifactEngineCostMetadata(
            relative_cost=relative_cost,
            cost_basis=(
                "Deterministic metadata derivation from in-memory workflow "
                "state; no provider or runtime execution calls."
            ),
            cache_sensitivity=(
                "Cache key must include request, route, and declared upstream "
                "metadata inputs for this engine."
            ),
        ),
        estimated_latency_metadata=ArtifactEngineLatencyMetadata(
            relative_latency=relative_latency,
            latency_basis=(
                "Bounded local metadata construction with no network, runtime, "
                "artifact execution, or file export work."
            ),
            blocking_inputs=required_inputs,
        ),
        future_agent_hooks=(
            "v4_planner_agent_contract",
            "v4_artifact_agent_contract",
            "v4_runtime_agent_contract",
            "v4_agent_router_contract",
            "adaptive_multi_agent_escalation_contract",
        ),
        future_execution_hooks=future_execution_hooks,
    )


ARTIFACT_PLANNER_ENGINE_CONTRACT = _contract(
    engine_id="artifact_planner",
    engine_name="Artifact Planner",
    authority_boundary=(
        "Structures intended artifact shape, dependencies, requirements, risks, "
        "missing information, HITL questions, and prompt guidance as metadata "
        "only; it does not select, critique, refine, execute, route, preview, "
        "repair, or implement future multi-agent behavior."
    ),
    required_inputs=("assistant_request", "route_decision"),
    optional_inputs=(
        "creative_translation",
        "creative_intent",
        "creative_hierarchy",
        "creative_plan",
        "creative_constraints",
        "creative_constraint_priorities",
        "creative_strategy",
        "creative_techniques",
        "runtime_capabilities",
        "creative_tradeoffs",
        "creative_quality_prediction",
        "symbolic_narrative",
        "creative_composition",
        "procedural_structure",
        "generative_structure",
        "semantic_motif",
        "emotional_consistency",
        "cross_modality",
        "audio_visual_scene",
    ),
    produced_metadata=("ArtifactPlan",),
    produced_signals=(
        "primary_artifact_intent",
        "artifact_type",
        "artifact_family",
        "required_components",
        "runtime_requirements",
        "expected_output_structure",
        "missing_information",
        "evidence",
        "prompt_guidance",
    ),
    confidence_signals=("evidence", "required_components", "missing_information"),
    ambiguity_signals=("missing_information", "hitl_questions"),
    risk_signals=("implementation_risks",),
    escalation_candidates=("missing_information", "hitl_questions"),
    downstream_dependencies=(
        "artifact_dependency_graph",
        "runtime_compatibility_engine",
        "artifact_capability_matrix",
        "multi_artifact_strategy",
        "artifact_critic",
        "artifact_refiner",
        "artifact_intelligence_synthesis",
        "artifact_merge_planner",
        "artifact_export_intelligence",
    ),
    upstream_dependencies=("creative_intelligence_metadata", "runtime_capabilities"),
    cacheability="deterministic_per_request",
)

ARTIFACT_DEPENDENCY_GRAPH_ENGINE_CONTRACT = _contract(
    engine_id="artifact_dependency_graph",
    engine_name="Artifact Dependency Graph",
    authority_boundary=(
        "Structures artifact dependency nodes, edges, upstream metadata, "
        "downstream consumers, conflicts, risks, HITL questions, and prompt "
        "guidance only; it does not execute, route, preview, repair, or "
        "implement downstream engines."
    ),
    required_inputs=("assistant_request", "route_decision", "artifact_plan"),
    optional_inputs=ARTIFACT_PLANNER_ENGINE_CONTRACT.optional_inputs,
    produced_metadata=("ArtifactDependencyGraph",),
    produced_signals=(
        "artifact_nodes",
        "dependency_edges",
        "required_upstream_metadata",
        "optional_upstream_metadata",
        "downstream_consumers",
        "prompt_facing_dependencies",
        "evidence",
    ),
    confidence_signals=("artifact_nodes", "dependency_edges", "evidence"),
    ambiguity_signals=("missing_dependency_risks", "hitl_questions"),
    risk_signals=("blocking_dependencies", "dependency_conflicts"),
    escalation_candidates=("blocking_dependencies", "hitl_questions"),
    downstream_dependencies=(
        "runtime_compatibility_engine",
        "artifact_capability_matrix",
        "multi_artifact_strategy",
        "artifact_critic",
        "artifact_refiner",
        "artifact_intelligence_synthesis",
        "artifact_merge_planner",
        "artifact_export_intelligence",
    ),
    upstream_dependencies=("artifact_planner",),
)

RUNTIME_COMPATIBILITY_ENGINE_CONTRACT = _contract(
    engine_id="runtime_compatibility_engine",
    engine_name="Runtime Compatibility Engine",
    authority_boundary=(
        "Evaluates runtime compatibility, requirements, limitations, "
        "portability, interoperability, risks, HITL questions, and prompt "
        "guidance as metadata only; it does not execute, auto-select runtimes, "
        "route providers, choose renderers, preview, repair, or optimize."
    ),
    required_inputs=(
        "assistant_request",
        "route_decision",
        "artifact_plan",
        "artifact_dependency_graph",
    ),
    optional_inputs=(
        "runtime_capabilities",
        "creative_plan",
        "creative_constraints",
        "creative_tradeoffs",
    ),
    produced_metadata=("RuntimeCompatibilityProfile",),
    produced_signals=(
        "compatible_runtimes",
        "unsupported_runtimes",
        "preferred_runtimes",
        "runtime_confidence",
        "compatibility_assessments",
        "portability",
        "interoperability",
    ),
    confidence_signals=("runtime_confidence", "compatibility_assessments"),
    ambiguity_signals=("missing_runtime_information", "hitl_questions"),
    risk_signals=("implementation_risks", "unsupported_runtimes"),
    escalation_candidates=("missing_runtime_information", "hitl_questions"),
    downstream_dependencies=(
        "artifact_capability_matrix",
        "multi_artifact_strategy",
        "artifact_critic",
        "artifact_refiner",
        "artifact_intelligence_synthesis",
        "artifact_merge_planner",
        "artifact_export_intelligence",
    ),
    upstream_dependencies=("artifact_planner", "artifact_dependency_graph"),
)

ARTIFACT_CAPABILITY_MATRIX_ENGINE_CONTRACT = _contract(
    engine_id="artifact_capability_matrix",
    engine_name="Artifact Capability Matrix",
    authority_boundary=(
        "Maps planned artifact needs against runtime and capability fit as "
        "metadata only; it does not select runtimes, execute artifacts, route "
        "providers, modify previews, merge artifacts, or trigger future agents."
    ),
    required_inputs=(
        "assistant_request",
        "route_decision",
        "artifact_plan",
        "artifact_dependency_graph",
        "runtime_compatibility",
    ),
    optional_inputs=(
        "runtime_capabilities",
        "creative_plan",
        "creative_constraints",
        "creative_strategy",
        "creative_techniques",
        "creative_tradeoffs",
    ),
    produced_metadata=("ArtifactCapabilityMatrix",),
    produced_signals=(
        "capability_profiles",
        "strongest_targets",
        "weakest_targets",
        "export_fit",
        "portability_fit",
        "interoperability_fit",
        "capability_confidence",
    ),
    confidence_signals=("capability_confidence", "capability_profiles"),
    ambiguity_signals=("unsupported_or_risky_capabilities", "hitl_questions"),
    risk_signals=("capability_risks", "target_weaknesses"),
    escalation_candidates=("unsupported_or_risky_capabilities", "hitl_questions"),
    downstream_dependencies=(
        "multi_artifact_strategy",
        "artifact_critic",
        "artifact_refiner",
        "artifact_intelligence_synthesis",
        "artifact_merge_planner",
        "artifact_export_intelligence",
    ),
    upstream_dependencies=(
        "artifact_planner",
        "artifact_dependency_graph",
        "runtime_compatibility_engine",
    ),
)

MULTI_ARTIFACT_STRATEGY_ENGINE_CONTRACT = _contract(
    engine_id="multi_artifact_strategy",
    engine_name="Multi-Artifact Strategy",
    authority_boundary=(
        "Plans primary/supporting artifact relationships, sequencing, "
        "priorities, groups, handoffs, risks, HITL questions, and prompt "
        "guidance as metadata only; it does not generate, execute, merge, "
        "export, route, preview, or trigger workflows."
    ),
    required_inputs=(
        "assistant_request",
        "route_decision",
        "artifact_plan",
        "artifact_dependency_graph",
        "runtime_compatibility",
        "artifact_capability_matrix",
    ),
    optional_inputs=(
        "runtime_capabilities",
        "creative_plan",
        "creative_constraints",
        "creative_tradeoffs",
    ),
    produced_metadata=("MultiArtifactStrategy",),
    produced_signals=(
        "primary_artifact",
        "supporting_artifacts",
        "artifact_sequence",
        "artifact_priorities",
        "artifact_groups",
        "artifact_handoff_points",
        "evidence",
    ),
    confidence_signals=("artifact_sequence", "evidence"),
    ambiguity_signals=("hitl_questions", "missing_artifact_information"),
    risk_signals=("coordination_risks", "handoff_risks"),
    escalation_candidates=("hitl_questions", "coordination_risks"),
    downstream_dependencies=(
        "artifact_critic",
        "artifact_refiner",
        "artifact_intelligence_synthesis",
        "artifact_merge_planner",
        "artifact_export_intelligence",
    ),
    upstream_dependencies=(
        "artifact_planner",
        "artifact_dependency_graph",
        "runtime_compatibility_engine",
        "artifact_capability_matrix",
    ),
)

ARTIFACT_CRITIC_ENGINE_CONTRACT = _contract(
    engine_id="artifact_critic",
    engine_name="Artifact Critic",
    authority_boundary=(
        "Critiques pre-generation artifact metadata for strengths, weaknesses, "
        "runtime concerns, risk assessment, HITL questions, and prompt guidance "
        "only; it does not critique generated artifacts, refine output, execute, "
        "route, preview, merge, export, or retry."
    ),
    required_inputs=(
        "assistant_request",
        "route_decision",
        "artifact_plan",
        "artifact_dependency_graph",
        "runtime_compatibility",
        "artifact_capability_matrix",
        "multi_artifact_strategy",
    ),
    produced_metadata=("ArtifactCriticProfile",),
    produced_signals=(
        "strengths",
        "weaknesses",
        "runtime_concerns",
        "risk_assessment",
        "critique_confidence",
        "prompt_guidance",
    ),
    confidence_signals=("critique_confidence", "strengths"),
    ambiguity_signals=("hitl_questions", "weaknesses"),
    risk_signals=("risk_assessment", "runtime_concerns", "weaknesses"),
    escalation_candidates=("hitl_questions", "risk_assessment"),
    downstream_dependencies=(
        "artifact_refiner",
        "artifact_intelligence_synthesis",
        "artifact_merge_planner",
        "artifact_export_intelligence",
    ),
    upstream_dependencies=(
        "artifact_planner",
        "artifact_dependency_graph",
        "runtime_compatibility_engine",
        "artifact_capability_matrix",
        "multi_artifact_strategy",
    ),
)

ARTIFACT_REFINER_ENGINE_CONTRACT = _contract(
    engine_id="artifact_refiner",
    engine_name="Artifact Refiner",
    authority_boundary=(
        "Recommends metadata-level refinement focus, improvements, risk "
        "reductions, trade-offs, HITL questions, and prompt guidance only; it "
        "does not modify artifacts, run refinement passes, execute, route, "
        "preview, merge, export, or retry."
    ),
    required_inputs=(
        "assistant_request",
        "route_decision",
        "artifact_plan",
        "artifact_dependency_graph",
        "runtime_compatibility",
        "artifact_capability_matrix",
        "multi_artifact_strategy",
        "artifact_critic",
    ),
    produced_metadata=("ArtifactRefinerProfile",),
    produced_signals=(
        "refinement_focus",
        "recommended_improvements",
        "risk_reductions",
        "tradeoff_notes",
        "refinement_confidence",
        "prompt_guidance",
    ),
    confidence_signals=("refinement_confidence", "recommended_improvements"),
    ambiguity_signals=("hitl_questions", "tradeoff_notes"),
    risk_signals=("risk_reductions", "tradeoff_notes"),
    escalation_candidates=("hitl_questions", "tradeoff_notes"),
    downstream_dependencies=(
        "artifact_intelligence_synthesis",
        "artifact_merge_planner",
        "artifact_export_intelligence",
    ),
    upstream_dependencies=(
        "artifact_planner",
        "artifact_dependency_graph",
        "runtime_compatibility_engine",
        "artifact_capability_matrix",
        "multi_artifact_strategy",
        "artifact_critic",
    ),
)

ARTIFACT_INTELLIGENCE_SYNTHESIS_ENGINE_CONTRACT = _contract(
    engine_id="artifact_intelligence_synthesis",
    engine_name="Artifact Intelligence Synthesis",
    authority_boundary=(
        "Synthesizes artifact planning, dependency, runtime, capability, "
        "strategy, critic, and refiner metadata into implementation-readiness "
        "guidance only; it does not generate, choose runtimes, route, preview, "
        "merge, export, execute, or trigger future systems."
    ),
    required_inputs=(
        "assistant_request",
        "route_decision",
        "artifact_plan",
        "artifact_dependency_graph",
        "runtime_compatibility",
        "artifact_capability_matrix",
        "multi_artifact_strategy",
        "artifact_critic",
        "artifact_refiner",
    ),
    produced_metadata=("ArtifactIntelligenceSynthesisProfile",),
    produced_signals=(
        "synthesis_confidence",
        "implementation_readiness",
        "implementation_risk",
        "recommended_artifact_path",
        "major_risks",
        "coordination_notes",
        "prompt_guidance",
    ),
    confidence_signals=("synthesis_confidence", "implementation_readiness"),
    ambiguity_signals=("hitl_questions", "coordination_notes"),
    risk_signals=("implementation_risk", "major_risks"),
    escalation_candidates=("hitl_questions", "implementation_risk"),
    downstream_dependencies=("artifact_merge_planner", "artifact_export_intelligence"),
    upstream_dependencies=(
        "artifact_planner",
        "artifact_dependency_graph",
        "runtime_compatibility_engine",
        "artifact_capability_matrix",
        "multi_artifact_strategy",
        "artifact_critic",
        "artifact_refiner",
    ),
)

ARTIFACT_MERGE_PLANNER_ENGINE_CONTRACT = _contract(
    engine_id="artifact_merge_planner",
    engine_name="Artifact Merge Planner",
    authority_boundary=(
        "Plans artifact merge strategy, boundaries, join points, integration "
        "order, risks, rejected merge paths, HITL questions, and prompt "
        "guidance as metadata only; it does not merge, modify, execute, route, "
        "preview, export, retry, or escalate."
    ),
    required_inputs=(
        "assistant_request",
        "route_decision",
        "artifact_plan",
        "artifact_dependency_graph",
        "runtime_compatibility",
        "artifact_capability_matrix",
        "multi_artifact_strategy",
        "artifact_critic",
        "artifact_refiner",
        "artifact_intelligence_synthesis",
    ),
    produced_metadata=("ArtifactMergePlannerProfile",),
    produced_signals=(
        "merge_confidence",
        "merge_strategy",
        "recommended_merge_path",
        "artifact_boundaries",
        "artifact_join_points",
        "integration_order",
        "rejected_merge_paths",
    ),
    confidence_signals=("merge_confidence", "artifact_join_points"),
    ambiguity_signals=("hitl_questions", "artifact_separation_points"),
    risk_signals=("composition_risks", "runtime_merge_risks"),
    escalation_candidates=("hitl_questions", "rejected_merge_paths"),
    downstream_dependencies=("artifact_export_intelligence",),
    upstream_dependencies=(
        "artifact_planner",
        "artifact_dependency_graph",
        "runtime_compatibility_engine",
        "artifact_capability_matrix",
        "multi_artifact_strategy",
        "artifact_critic",
        "artifact_refiner",
        "artifact_intelligence_synthesis",
    ),
)

ARTIFACT_EXPORT_INTELLIGENCE_ENGINE_CONTRACT = _contract(
    engine_id="artifact_export_intelligence",
    engine_name="Artifact Export Intelligence",
    authority_boundary=(
        "Recommends export targets, formats, requirements, constraints, risks, "
        "package notes, documentation needs, handoffs, rejected export paths, "
        "HITL questions, and prompt guidance as metadata only; it does not "
        "export, write, package, modify, execute, select runtimes, route, "
        "preview, deploy, retry, or trigger workflows."
    ),
    required_inputs=(
        "assistant_request",
        "route_decision",
        "artifact_plan",
        "artifact_dependency_graph",
        "runtime_compatibility",
        "artifact_capability_matrix",
        "multi_artifact_strategy",
        "artifact_critic",
        "artifact_refiner",
        "artifact_intelligence_synthesis",
        "artifact_merge_planner",
    ),
    produced_metadata=("ArtifactExportIntelligenceProfile",),
    produced_signals=(
        "export_confidence",
        "export_readiness",
        "export_targets",
        "preferred_export_target",
        "export_format_recommendations",
        "downstream_tool_handoffs",
        "rejected_export_paths",
    ),
    confidence_signals=("export_confidence", "export_readiness"),
    ambiguity_signals=("hitl_questions", "export_constraints"),
    risk_signals=("export_risks", "rejected_export_paths"),
    escalation_candidates=("hitl_questions", "export_readiness"),
    downstream_dependencies=(),
    upstream_dependencies=(
        "artifact_planner",
        "artifact_dependency_graph",
        "runtime_compatibility_engine",
        "artifact_capability_matrix",
        "multi_artifact_strategy",
        "artifact_critic",
        "artifact_refiner",
        "artifact_intelligence_synthesis",
        "artifact_merge_planner",
    ),
    future_execution_hooks=(
        "v5_execution_optimization_readiness",
        "v6_learning_engine_feedback",
        "v6_blueprint_export_readiness",
    ),
)

ARTIFACT_INTELLIGENCE_ENGINE_CONTRACTS = (
    ARTIFACT_PLANNER_ENGINE_CONTRACT,
    ARTIFACT_DEPENDENCY_GRAPH_ENGINE_CONTRACT,
    RUNTIME_COMPATIBILITY_ENGINE_CONTRACT,
    ARTIFACT_CAPABILITY_MATRIX_ENGINE_CONTRACT,
    MULTI_ARTIFACT_STRATEGY_ENGINE_CONTRACT,
    ARTIFACT_CRITIC_ENGINE_CONTRACT,
    ARTIFACT_REFINER_ENGINE_CONTRACT,
    ARTIFACT_INTELLIGENCE_SYNTHESIS_ENGINE_CONTRACT,
    ARTIFACT_MERGE_PLANNER_ENGINE_CONTRACT,
    ARTIFACT_EXPORT_INTELLIGENCE_ENGINE_CONTRACT,
)

ARTIFACT_INTELLIGENCE_ENGINE_CONTRACT_REGISTRY = (
    ArtifactIntelligenceEngineContractRegistry(
        engine_contracts=ARTIFACT_INTELLIGENCE_ENGINE_CONTRACTS,
        engine_ids=tuple(
            contract.engine_id for contract in ARTIFACT_INTELLIGENCE_ENGINE_CONTRACTS
        ),
        contract_count=len(ARTIFACT_INTELLIGENCE_ENGINE_CONTRACTS),
        future_agent_consumers=(
            "v4_planner_agent",
            "v4_artifact_agent",
            "v4_runtime_agent",
            "v4_agent_router",
            "adaptive_multi_agent_escalation",
            "v5_execution_optimization_engine",
            "v6_learning_engine",
        ),
    )
)
