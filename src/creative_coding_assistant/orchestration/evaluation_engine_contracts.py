"""Shared Creative Evaluation engine contracts for V3.4 metadata."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EvaluationEngineCategory = Literal["creative_evaluation"]
EvaluationEngineCacheability = Literal[
    "deterministic_per_request",
    "deterministic_with_upstream_metadata",
]
EvaluationEngineParallelizationSupport = Literal[
    "requires_ordered_upstream_metadata",
    "parallel_after_required_inputs",
]

EVALUATION_ENGINE_CONTRACT_SERIALIZATION_VERSION = "evaluation_engine_contract.v1"
EVALUATION_ENGINE_CONTRACT_REGISTRY_SERIALIZATION_VERSION = (
    "evaluation_engine_contract_registry.v1"
)
EVALUATION_ENGINE_CONTRACT_REGISTRY_AUTHORITY_BOUNDARY = (
    "Creative Evaluation engine contracts describe metadata interfaces, "
    "dependencies, signals, evidence expectations, cost, latency, and future "
    "hooks only; they do not change evaluation logic, scoring, confidence, "
    "reflection, reports, prompts, workflow ordering, routing, runtime "
    "selection, execution, retries, previews, artifacts, or generated output."
)


class EvaluationEngineCostMetadata(BaseModel):
    """Static estimated cost metadata for a Creative Evaluation engine."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    relative_cost: Literal["low", "medium"]
    external_provider_calls: bool = False
    cost_basis: str = Field(min_length=1, max_length=260)
    cache_sensitivity: str = Field(min_length=1, max_length=260)


class EvaluationEngineLatencyMetadata(BaseModel):
    """Static estimated latency metadata for a Creative Evaluation engine."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    relative_latency: Literal["low", "medium"]
    latency_basis: str = Field(min_length=1, max_length=260)
    blocking_inputs: tuple[str, ...] = Field(default_factory=tuple, max_length=16)


class EvaluationEngineEvidenceContract(BaseModel):
    """Evidence expectations for one Creative Evaluation engine."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    evidence_sources: tuple[str, ...] = Field(min_length=1, max_length=16)
    evidence_payload_fields: tuple[str, ...] = Field(min_length=1, max_length=16)
    evidence_quality_signals: tuple[str, ...] = Field(min_length=1, max_length=16)
    missing_evidence_behavior: str = Field(min_length=1, max_length=260)


class EvaluationEngineContract(BaseModel):
    """Common metadata contract exposed by every Creative Evaluation engine."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    engine_id: str = Field(min_length=1, max_length=80)
    engine_name: str = Field(min_length=1, max_length=140)
    engine_version: str = Field(min_length=1, max_length=24)
    engine_category: EvaluationEngineCategory = "creative_evaluation"
    authority_boundary: str = Field(min_length=1, max_length=900)
    required_inputs: tuple[str, ...] = Field(default_factory=tuple, max_length=16)
    optional_inputs: tuple[str, ...] = Field(default_factory=tuple, max_length=32)
    produced_metadata: tuple[str, ...] = Field(min_length=1, max_length=18)
    produced_signals: tuple[str, ...] = Field(min_length=1, max_length=18)
    confidence_signals: tuple[str, ...] = Field(min_length=1, max_length=12)
    ambiguity_signals: tuple[str, ...] = Field(min_length=1, max_length=12)
    risk_signals: tuple[str, ...] = Field(min_length=1, max_length=12)
    downstream_dependencies: tuple[str, ...] = Field(default_factory=tuple, max_length=16)
    upstream_dependencies: tuple[str, ...] = Field(default_factory=tuple, max_length=16)
    evidence_contract: EvaluationEngineEvidenceContract
    cacheability: EvaluationEngineCacheability
    parallelization_support: EvaluationEngineParallelizationSupport
    estimated_cost_metadata: EvaluationEngineCostMetadata
    estimated_latency_metadata: EvaluationEngineLatencyMetadata
    serialization_version: Literal["evaluation_engine_contract.v1"] = (
        EVALUATION_ENGINE_CONTRACT_SERIALIZATION_VERSION
    )
    future_agent_hooks: tuple[str, ...] = Field(min_length=1, max_length=12)
    future_execution_hooks: tuple[str, ...] = Field(min_length=1, max_length=12)


class EvaluationEngineContractRegistry(BaseModel):
    """Stable registry of Creative Evaluation engine metadata contracts."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["evaluation_engine_contract_registry"] = (
        "evaluation_engine_contract_registry"
    )
    engine_category: EvaluationEngineCategory = "creative_evaluation"
    serialization_version: Literal[
        "evaluation_engine_contract_registry.v1"
    ] = EVALUATION_ENGINE_CONTRACT_REGISTRY_SERIALIZATION_VERSION
    authority_boundary: str = Field(
        default=EVALUATION_ENGINE_CONTRACT_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    engine_contracts: tuple[EvaluationEngineContract, ...] = Field(
        min_length=8,
        max_length=8,
    )
    engine_ids: tuple[str, ...] = Field(min_length=8, max_length=8)
    contract_count: int = Field(ge=8, le=8)
    future_agent_consumers: tuple[str, ...] = Field(min_length=1, max_length=8)


def evaluation_engine_contracts() -> EvaluationEngineContractRegistry:
    """Return the static Creative Evaluation contract registry."""

    return EVALUATION_ENGINE_CONTRACT_REGISTRY


def creative_evaluation_engine_contracts() -> EvaluationEngineContractRegistry:
    """Return the static Creative Evaluation contract registry."""

    return EVALUATION_ENGINE_CONTRACT_REGISTRY


def evaluation_engine_contract_by_id(
    engine_id: str,
) -> EvaluationEngineContract | None:
    """Return one engine contract by id without changing engine behavior."""

    for contract in EVALUATION_ENGINE_CONTRACTS:
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
    downstream_dependencies: tuple[str, ...],
    upstream_dependencies: tuple[str, ...],
    evidence_sources: tuple[str, ...],
    evidence_payload_fields: tuple[str, ...],
    evidence_quality_signals: tuple[str, ...],
    missing_evidence_behavior: str,
    cacheability: EvaluationEngineCacheability = (
        "deterministic_with_upstream_metadata"
    ),
    parallelization_support: EvaluationEngineParallelizationSupport = (
        "requires_ordered_upstream_metadata"
    ),
    relative_cost: Literal["low", "medium"] = "low",
    relative_latency: Literal["low", "medium"] = "low",
    future_execution_hooks: tuple[str, ...] = (
        "v5_execution_optimization_readiness",
        "v6_learning_engine_feedback",
    ),
) -> EvaluationEngineContract:
    return EvaluationEngineContract(
        engine_id=engine_id,
        engine_name=engine_name,
        engine_version="v3.4",
        authority_boundary=authority_boundary,
        required_inputs=required_inputs,
        optional_inputs=optional_inputs,
        produced_metadata=produced_metadata,
        produced_signals=produced_signals,
        confidence_signals=confidence_signals,
        ambiguity_signals=ambiguity_signals,
        risk_signals=risk_signals,
        downstream_dependencies=downstream_dependencies,
        upstream_dependencies=upstream_dependencies,
        evidence_contract=EvaluationEngineEvidenceContract(
            evidence_sources=evidence_sources,
            evidence_payload_fields=evidence_payload_fields,
            evidence_quality_signals=evidence_quality_signals,
            missing_evidence_behavior=missing_evidence_behavior,
        ),
        cacheability=cacheability,
        parallelization_support=parallelization_support,
        estimated_cost_metadata=EvaluationEngineCostMetadata(
            relative_cost=relative_cost,
            cost_basis=(
                "Deterministic metadata derivation from in-memory workflow "
                "state; no provider, runtime, preview, or execution calls."
            ),
            cache_sensitivity=(
                "Cache key must include request, route, generated response "
                "when required, and declared upstream evaluation metadata."
            ),
        ),
        estimated_latency_metadata=EvaluationEngineLatencyMetadata(
            relative_latency=relative_latency,
            latency_basis=(
                "Bounded local metadata construction with no network, runtime, "
                "preview, artifact execution, or file export work."
            ),
            blocking_inputs=required_inputs,
        ),
        future_agent_hooks=(
            "v3_5_inspector_contract",
            "v3_5_dashboard_contract",
            "v4_agentic_studio_contract",
            "adaptive_multi_agent_escalation_contract",
            "v5_execution_optimization_contract",
            "v6_learning_engine_contract",
        ),
        future_execution_hooks=future_execution_hooks,
    )


CREATIVE_CRITIC_ENGINE_CONTRACT = _contract(
    engine_id="creative_critic",
    engine_name="Creative Critic Engine",
    authority_boundary=(
        "Evaluates pre-generation creative metadata for strengths, weaknesses, "
        "quality scores, risks, missing information, unsupported assumptions, "
        "HITL questions, and prompt guidance only; it does not modify outputs, "
        "reject answers, route providers, select runtimes, preview, repair, "
        "retry, refine, or invoke future agents."
    ),
    required_inputs=("assistant_request", "route_decision"),
    optional_inputs=("creative_intelligence_metadata", "artifact_intelligence_metadata"),
    produced_metadata=("CreativeCriticProfile",),
    produced_signals=(
        "critique_summary",
        "creative_strengths",
        "creative_weaknesses",
        "quality_scores",
        "risk_assessment",
        "prompt_guidance",
    ),
    confidence_signals=("critic_confidence", "evidence", "quality_scores"),
    ambiguity_signals=("missing_information", "hitl_questions"),
    risk_signals=("risk_assessment", "unsupported_assumptions"),
    downstream_dependencies=(
        "self_evaluation",
        "creative_improvement_planner",
        "reflection_loop",
        "creative_confidence",
        "creative_score",
        "consistency_validation",
        "evaluation_reports",
    ),
    upstream_dependencies=("creative_intelligence_metadata", "artifact_intelligence_metadata"),
    evidence_sources=("assistant_request", "route_decision", "planning_metadata"),
    evidence_payload_fields=("evidence", "missing_information", "unsupported_assumptions"),
    evidence_quality_signals=("critic_confidence", "quality_scores", "risk_assessment"),
    missing_evidence_behavior=(
        "Surface missing information and unsupported assumptions as advisory "
        "metadata without changing workflow control."
    ),
    cacheability="deterministic_per_request",
)

SELF_EVALUATION_ENGINE_CONTRACT = _contract(
    engine_id="self_evaluation",
    engine_name="Self Evaluation Engine",
    authority_boundary=(
        "Assesses generated responses, artifacts, request alignment, intent, "
        "constraints, runtime fit, coherence, risks, gaps, and HITL questions "
        "as metadata only; it does not modify outputs, reject answers, route, "
        "select runtimes, change previews, retry, refine, or run reflection."
    ),
    required_inputs=("assistant_request", "route_decision", "generated_response"),
    optional_inputs=("creative_critic", "artifacts", "creative_intelligence_metadata"),
    produced_metadata=("SelfEvaluationProfile",),
    produced_signals=(
        "evaluation_summary",
        "request_alignment",
        "completeness_assessment",
        "ambiguity_assessment",
        "quality_gaps",
        "prompt_guidance",
    ),
    confidence_signals=("self_evaluation_confidence", "evidence"),
    ambiguity_signals=("ambiguity_assessment", "missing_information", "hitl_questions"),
    risk_signals=("hallucination_risk", "underdelivery_risk", "unsupported_assumptions"),
    downstream_dependencies=(
        "creative_improvement_planner",
        "reflection_loop",
        "creative_confidence",
        "creative_score",
        "consistency_validation",
        "evaluation_reports",
    ),
    upstream_dependencies=("creative_critic",),
    evidence_sources=("generated_response", "creative_critic", "artifacts"),
    evidence_payload_fields=("evidence", "quality_gaps", "unsupported_assumptions"),
    evidence_quality_signals=("self_evaluation_confidence", "request_alignment"),
    missing_evidence_behavior=(
        "Report missing information and ambiguity as advisory metadata without "
        "triggering retries, refinement, or output modification."
    ),
)

CREATIVE_IMPROVEMENT_PLANNER_ENGINE_CONTRACT = _contract(
    engine_id="creative_improvement_planner",
    engine_name="Creative Improvement Planner",
    authority_boundary=(
        "Plans metadata-level improvement priorities, high-impact "
        "opportunities, low-risk improvements, experimental candidates, "
        "trade-offs, future refinement candidates, and HITL questions only; "
        "it does not edit artifacts, retry, route, preview, select runtimes, "
        "or trigger workflow loops."
    ),
    required_inputs=("assistant_request", "route_decision", "creative_critic", "self_evaluation"),
    optional_inputs=("generated_response", "artifacts"),
    produced_metadata=("CreativeImprovementPlannerProfile",),
    produced_signals=(
        "improvement_summary",
        "improvement_priorities",
        "highest_impact_opportunities",
        "low_risk_improvements",
        "future_refinement_candidates",
        "prompt_guidance",
    ),
    confidence_signals=("confidence", "evidence", "improvement_priorities"),
    ambiguity_signals=("hitl_questions", "tradeoff_notes"),
    risk_signals=("risk_assessment", "experimental_candidates"),
    downstream_dependencies=(
        "reflection_loop",
        "creative_confidence",
        "creative_score",
        "consistency_validation",
        "evaluation_reports",
    ),
    upstream_dependencies=("creative_critic", "self_evaluation"),
    evidence_sources=("creative_critic", "self_evaluation", "generated_response"),
    evidence_payload_fields=("evidence", "improvement_priorities", "hitl_questions"),
    evidence_quality_signals=("confidence", "highest_impact_opportunities"),
    missing_evidence_behavior=(
        "Prefer lower-confidence advisory improvement metadata without "
        "triggering edits, retries, or workflow loops."
    ),
)

REFLECTION_LOOP_ENGINE_CONTRACT = _contract(
    engine_id="reflection_loop",
    engine_name="Reflection Loop Engine",
    authority_boundary=(
        "Estimates theoretical reflection value, unresolved questions, "
        "refinement candidates, stop conditions, and HITL recommendation as "
        "metadata only; it does not trigger refinement, provider calls, "
        "retries, routing, runtime changes, previews, workflow loops, artifact "
        "edits, or future agents."
    ),
    required_inputs=(
        "assistant_request",
        "route_decision",
        "creative_critic",
        "self_evaluation",
        "creative_improvement_planner",
    ),
    optional_inputs=("planning_metadata",),
    produced_metadata=("ReflectionLoopProfile",),
    produced_signals=(
        "reflection_summary",
        "reflection_required",
        "reflection_priority",
        "reflection_depth",
        "refinement_candidates",
        "prompt_guidance",
    ),
    confidence_signals=("reflection_confidence", "evidence"),
    ambiguity_signals=("unresolved_questions", "hitl_recommendation"),
    risk_signals=("stop_conditions", "refinement_candidates"),
    downstream_dependencies=(
        "creative_confidence",
        "creative_score",
        "consistency_validation",
        "evaluation_reports",
    ),
    upstream_dependencies=(
        "creative_critic",
        "self_evaluation",
        "creative_improvement_planner",
    ),
    evidence_sources=("creative_critic", "self_evaluation", "creative_improvement_planner"),
    evidence_payload_fields=("evidence", "unresolved_questions", "refinement_candidates"),
    evidence_quality_signals=("reflection_confidence", "reflection_priority"),
    missing_evidence_behavior=(
        "Lower reflection certainty and expose unresolved questions without "
        "triggering any loop or refinement behavior."
    ),
)

CREATIVE_CONFIDENCE_ENGINE_CONTRACT = _contract(
    engine_id="creative_confidence",
    engine_name="Creative Confidence Engine",
    authority_boundary=(
        "Summarizes confidence, component scores, uncertainty, limitations, "
        "strengths, weaknesses, reliability, execution readiness, HITL need, "
        "and escalation recommendation as metadata only; it does not change "
        "outputs, edit artifacts, refine, retry, route, select runtimes, call "
        "providers, alter previews, or invoke future agents."
    ),
    required_inputs=(
        "assistant_request",
        "route_decision",
        "creative_critic",
        "self_evaluation",
        "creative_improvement_planner",
        "reflection_loop",
    ),
    optional_inputs=("planning_metadata",),
    produced_metadata=("CreativeConfidenceProfile",),
    produced_signals=(
        "confidence_summary",
        "confidence_score",
        "confidence_level",
        "confidence_components",
        "confidence_uncertainties",
        "prompt_guidance",
    ),
    confidence_signals=("confidence_score", "confidence_components", "confidence_evidence"),
    ambiguity_signals=("confidence_uncertainties", "hitl_recommendation"),
    risk_signals=("confidence_limitations", "expected_output_reliability"),
    downstream_dependencies=("creative_score", "consistency_validation", "evaluation_reports"),
    upstream_dependencies=(
        "creative_critic",
        "self_evaluation",
        "creative_improvement_planner",
        "reflection_loop",
    ),
    evidence_sources=(
        "creative_critic",
        "self_evaluation",
        "creative_improvement_planner",
        "reflection_loop",
    ),
    evidence_payload_fields=("confidence_evidence", "confidence_components"),
    evidence_quality_signals=("confidence_score", "confidence_level"),
    missing_evidence_behavior=(
        "Expose uncertainty and HITL need as advisory metadata without changing "
        "execution or evaluation control."
    ),
)

CREATIVE_SCORE_ENGINE_CONTRACT = _contract(
    engine_id="creative_score",
    engine_name="Creative Score Engine",
    authority_boundary=(
        "Computes advisory creative score metadata, component breakdowns, "
        "calibration notes, explainability, penalties, strengths, weaknesses, "
        "rationale, evidence, and HITL recommendation only; it does not change "
        "outputs, edit artifacts, refine, retry, route, select runtimes, call "
        "providers, alter previews, or invoke optimization systems."
    ),
    required_inputs=(
        "assistant_request",
        "route_decision",
        "creative_critic",
        "self_evaluation",
        "creative_improvement_planner",
        "reflection_loop",
        "creative_confidence",
    ),
    optional_inputs=("planning_metadata",),
    produced_metadata=("CreativeScoreProfile",),
    produced_signals=(
        "overall_creative_score",
        "score_band",
        "score_breakdown",
        "score_components",
        "score_explainability",
        "prompt_guidance",
    ),
    confidence_signals=("confidence_weight", "score_evidence", "score_components"),
    ambiguity_signals=("hitl_recommendation", "uncertainty_penalty"),
    risk_signals=("risk_penalty", "negative_contributions", "weaknesses"),
    downstream_dependencies=("consistency_validation", "evaluation_reports"),
    upstream_dependencies=(
        "creative_critic",
        "self_evaluation",
        "creative_improvement_planner",
        "reflection_loop",
        "creative_confidence",
    ),
    evidence_sources=(
        "creative_critic",
        "self_evaluation",
        "creative_improvement_planner",
        "reflection_loop",
        "creative_confidence",
    ),
    evidence_payload_fields=("score_evidence", "score_components", "score_breakdown"),
    evidence_quality_signals=("overall_creative_score", "confidence_weight"),
    missing_evidence_behavior=(
        "Represent missing evidence as penalties or lower confidence metadata "
        "without changing generation, routing, or runtime behavior."
    ),
)

CONSISTENCY_VALIDATION_ENGINE_CONTRACT = _contract(
    engine_id="consistency_validation",
    engine_name="Consistency Validation Engine",
    authority_boundary=(
        "Validates existing V3.4 evaluation metadata for consistency, "
        "conflicts, contradiction level, ambiguity, unsupported conclusions, "
        "integrity, evidence, and HITL recommendation only; it does not change "
        "outputs, edit artifacts, refine, retry, route, select runtimes, alter "
        "previews, invoke future agents, or trigger reporting systems."
    ),
    required_inputs=(
        "assistant_request",
        "route_decision",
        "creative_critic",
        "self_evaluation",
        "creative_improvement_planner",
        "reflection_loop",
        "creative_confidence",
        "creative_score",
    ),
    optional_inputs=("planning_metadata", "creative_reasoning"),
    produced_metadata=("ConsistencyValidationProfile",),
    produced_signals=(
        "consistency_status",
        "consistency_summary",
        "detected_conflicts",
        "contradiction_level",
        "evaluation_integrity",
        "prompt_guidance",
    ),
    confidence_signals=("evaluation_integrity", "evidence"),
    ambiguity_signals=("ambiguity_level", "unsupported_conclusions", "hitl_recommendation"),
    risk_signals=("detected_conflicts", "contradiction_level"),
    downstream_dependencies=("evaluation_reports",),
    upstream_dependencies=(
        "creative_critic",
        "self_evaluation",
        "creative_improvement_planner",
        "reflection_loop",
        "creative_confidence",
        "creative_score",
    ),
    evidence_sources=(
        "creative_critic",
        "self_evaluation",
        "creative_improvement_planner",
        "reflection_loop",
        "creative_confidence",
        "creative_score",
    ),
    evidence_payload_fields=("evidence", "detected_conflicts", "unsupported_conclusions"),
    evidence_quality_signals=("evaluation_integrity", "contradiction_level"),
    missing_evidence_behavior=(
        "Mark checks as missing or fragile advisory metadata without triggering "
        "workflow changes, retries, or report actions."
    ),
)

EVALUATION_REPORTS_ENGINE_CONTRACT = _contract(
    engine_id="evaluation_reports",
    engine_name="Evaluation Reports",
    authority_boundary=(
        "Aggregates existing V3.4 evaluation metadata into report summaries, "
        "trace, provenance, explainability, dependencies, evidence, risks, "
        "recommendations, and HITL posture only; it does not modify outputs, "
        "regenerate responses, execute artifacts, trigger refinement, retry, "
        "route, select runtimes, alter previews, invoke future agents, or "
        "perform dashboard, inspector, optimization, or learning behavior."
    ),
    required_inputs=(
        "assistant_request",
        "route_decision",
        "creative_critic",
        "self_evaluation",
        "creative_improvement_planner",
        "reflection_loop",
        "creative_confidence",
        "creative_score",
        "consistency_validation",
    ),
    optional_inputs=("planning_metadata",),
    produced_metadata=("EvaluationReportProfile",),
    produced_signals=(
        "executive_summary",
        "quality_summary",
        "confidence_summary",
        "consistency_summary",
        "evaluation_trace",
        "evaluation_provenance",
    ),
    confidence_signals=("evaluation_provenance", "evaluation_dependencies"),
    ambiguity_signals=("evaluation_dependencies", "hitl_recommendation"),
    risk_signals=("risks", "weaknesses", "recommendations"),
    downstream_dependencies=(),
    upstream_dependencies=(
        "creative_critic",
        "self_evaluation",
        "creative_improvement_planner",
        "reflection_loop",
        "creative_confidence",
        "creative_score",
        "consistency_validation",
    ),
    evidence_sources=(
        "creative_critic",
        "self_evaluation",
        "creative_improvement_planner",
        "reflection_loop",
        "creative_confidence",
        "creative_score",
        "consistency_validation",
    ),
    evidence_payload_fields=("evaluation_trace", "evaluation_provenance", "evidence_chain"),
    evidence_quality_signals=("evaluation_dependencies", "evaluation_explainability"),
    missing_evidence_behavior=(
        "Record missing dependencies and lower report completeness as advisory "
        "metadata without triggering report actions or workflow changes."
    ),
    future_execution_hooks=(
        "v3_5_inspector_report_surface",
        "v3_5_dashboard_report_surface",
        "v5_execution_optimization_readiness",
        "v6_learning_engine_feedback",
    ),
)

EVALUATION_ENGINE_CONTRACTS = (
    CREATIVE_CRITIC_ENGINE_CONTRACT,
    SELF_EVALUATION_ENGINE_CONTRACT,
    CREATIVE_IMPROVEMENT_PLANNER_ENGINE_CONTRACT,
    REFLECTION_LOOP_ENGINE_CONTRACT,
    CREATIVE_CONFIDENCE_ENGINE_CONTRACT,
    CREATIVE_SCORE_ENGINE_CONTRACT,
    CONSISTENCY_VALIDATION_ENGINE_CONTRACT,
    EVALUATION_REPORTS_ENGINE_CONTRACT,
)

EVALUATION_ENGINE_CONTRACT_REGISTRY = EvaluationEngineContractRegistry(
    engine_contracts=EVALUATION_ENGINE_CONTRACTS,
    engine_ids=tuple(contract.engine_id for contract in EVALUATION_ENGINE_CONTRACTS),
    contract_count=len(EVALUATION_ENGINE_CONTRACTS),
    future_agent_consumers=(
        "v3_5_inspector",
        "v3_5_dashboard",
        "v4_agentic_studio",
        "adaptive_multi_agent_escalation",
        "v5_execution_optimization_engine",
        "v6_learning_engine",
    ),
)
