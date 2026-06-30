"""V6.1 advisory evaluation learning."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.adaptive_learning_engine import (
    AdaptiveLearningPlan,
    AdaptiveLearningSignal,
    evaluate_adaptive_learning_engine,
)
from creative_coding_assistant.orchestration.evaluation_engine_contracts import (
    EvaluationEngineContract,
    EvaluationEngineContractRegistry,
    evaluation_engine_contracts,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

EvaluationLearningPatternKind = Literal[
    "critic_contract_learning",
    "self_evaluation_contract_learning",
    "confidence_contract_learning",
    "report_guardrail_learning",
]
EvaluationLearningStatus = Literal["learnable", "review_required", "guarded"]
EvaluationLearningPriority = Literal["standard", "elevated", "critical", "guarded"]
EvaluationLearningPosture = Literal["learnable", "review_required", "guarded"]

EVALUATION_LEARNING_PATTERN_SERIALIZATION_VERSION = "evaluation_learning_pattern.v1"
EVALUATION_LEARNING_PLAN_SERIALIZATION_VERSION = "evaluation_learning_plan.v1"
EVALUATION_LEARNING_AUTHORITY_BOUNDARY = (
    "V6.1 evaluation learning derives evaluation patterns from read-only "
    "evaluation engine contract metadata and adaptive learning signals only; "
    "it does not run evaluations, evaluate generated output, mutate scores, "
    "change confidence, execute reflection loops, generate reports, change "
    "workflow order, route providers or models, execute providers, select "
    "runtimes, execute artifacts, emit HITL requests, control workflows, "
    "mutate workflow graphs, trigger retries or refinements, mutate prompts, "
    "write storage, modify generated output, or apply Runtime Evolution."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "evaluation_execution",
    "generated_output_evaluation",
    "score_mutation",
    "confidence_mutation",
    "reflection_loop_execution",
    "report_generation",
    "workflow_order_change",
    "provider_or_model_routing",
    "automatic_provider_switching",
    "automatic_model_switching",
    "provider_execution",
    "runtime_selection",
    "artifact_execution",
    "hitl_request_emission",
    "workflow_control",
    "workflow_graph_mutation",
    "workflow_execution",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)


class EvaluationLearningPattern(BaseModel):
    """One advisory evaluation learning pattern."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    pattern_id: str = Field(min_length=1, max_length=180)
    pattern_kind: EvaluationLearningPatternKind
    status: EvaluationLearningStatus
    priority: EvaluationLearningPriority
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    source_contract_registry_role: str = Field(min_length=1, max_length=120)
    source_engine_id: str = Field(min_length=1, max_length=120)
    source_engine_name: str = Field(min_length=1, max_length=160)
    source_learning_signal_id: str = Field(min_length=1, max_length=180)
    source_workflow_risk_factor_id: str = Field(min_length=1, max_length=180)
    required_input_count: int = Field(ge=0, le=16)
    optional_input_count: int = Field(ge=0, le=32)
    produced_signal_count: int = Field(ge=1, le=18)
    confidence_signal_count: int = Field(ge=1, le=12)
    ambiguity_signal_count: int = Field(ge=1, le=12)
    risk_signal_count: int = Field(ge=1, le=12)
    future_execution_hook_count: int = Field(ge=1, le=12)
    relative_cost: Literal["low", "medium"]
    relative_latency: Literal["low", "medium"]
    cacheability: str = Field(min_length=1, max_length=120)
    parallelization_support: str = Field(min_length=1, max_length=120)
    learning_priority_score: int = Field(ge=0, le=1_000)
    evaluation_learning_weight: int = Field(ge=0, le=240)
    evaluation_learning_score: int = Field(ge=0, le=1_000)
    hitl_required: bool
    evaluation_pattern_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    evaluation_summary: str = Field(min_length=1, max_length=360)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    evaluation_learning_implemented: Literal[True] = True
    evaluation_pattern_metadata_implemented: Literal[True] = True
    evaluation_contract_metadata_used: Literal[True] = True
    adaptive_learning_metadata_used: Literal[True] = True
    evaluation_execution_implemented: Literal[False] = False
    generated_output_evaluation_implemented: Literal[False] = False
    score_mutation_implemented: Literal[False] = False
    confidence_mutation_implemented: Literal[False] = False
    reflection_loop_execution_implemented: Literal[False] = False
    report_generation_implemented: Literal[False] = False
    workflow_order_change_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    artifact_execution_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["evaluation_learning_pattern.v1"] = (
        EVALUATION_LEARNING_PATTERN_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _pattern_matches_contract(self) -> Self:
        if self.pattern_id != f"evaluation_learning::{self.pattern_kind}":
            raise ValueError("pattern_id must match pattern_kind")
        if self.evaluation_learning_score != _evaluation_learning_score(
            required_input_count=self.required_input_count,
            optional_input_count=self.optional_input_count,
            produced_signal_count=self.produced_signal_count,
            confidence_signal_count=self.confidence_signal_count,
            ambiguity_signal_count=self.ambiguity_signal_count,
            risk_signal_count=self.risk_signal_count,
            future_execution_hook_count=self.future_execution_hook_count,
            relative_cost=self.relative_cost,
            relative_latency=self.relative_latency,
            learning_priority_score=self.learning_priority_score,
            evaluation_learning_weight=self.evaluation_learning_weight,
        ):
            raise ValueError("evaluation_learning_score must combine source scores")
        if self.priority != _evaluation_priority(
            self.evaluation_learning_score,
            self.status,
        ):
            raise ValueError("priority must match score and status")
        if self.status == "guarded" and not self.hitl_required:
            raise ValueError("guarded evaluation learning requires HITL posture")
        return self


class EvaluationLearningPlan(BaseModel):
    """Bounded V6.1 advisory evaluation learning plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["evaluation_learning"] = "evaluation_learning"
    serialization_version: Literal["evaluation_learning_plan.v1"] = (
        EVALUATION_LEARNING_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=EVALUATION_LEARNING_AUTHORITY_BOUNDARY,
        max_length=1700,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_contract_registry_role: str = Field(min_length=1, max_length=120)
    source_contract_registry_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_adaptive_learning_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    patterns: tuple[EvaluationLearningPattern, ...] = Field(min_length=4, max_length=4)
    pattern_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    learnable_pattern_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=4)
    review_required_pattern_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    guarded_pattern_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=4)
    standard_priority_pattern_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    elevated_priority_pattern_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    critical_priority_pattern_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    guarded_priority_pattern_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    hitl_required_pattern_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    applied_evaluation_pattern_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    pattern_count: int = Field(ge=4, le=4)
    review_required_pattern_count: int = Field(ge=0, le=4)
    guarded_pattern_count: int = Field(ge=0, le=4)
    hitl_required_pattern_count: int = Field(ge=0, le=4)
    highest_evaluation_learning_score: int = Field(ge=0, le=1_000)
    overall_evaluation_learning_score: int = Field(ge=0, le=1_000)
    overall_evaluation_learning_posture: EvaluationLearningPosture
    source_engine_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    evaluation_learning_implemented: Literal[True] = True
    evaluation_pattern_metadata_implemented: Literal[True] = True
    evaluation_contract_metadata_used: Literal[True] = True
    adaptive_learning_metadata_used: Literal[True] = True
    evaluation_execution_implemented: Literal[False] = False
    generated_output_evaluation_implemented: Literal[False] = False
    score_mutation_implemented: Literal[False] = False
    confidence_mutation_implemented: Literal[False] = False
    reflection_loop_execution_implemented: Literal[False] = False
    report_generation_implemented: Literal[False] = False
    workflow_order_change_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    artifact_execution_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_patterns(self) -> Self:
        derived_pattern_ids = tuple(pattern.pattern_id for pattern in self.patterns)
        if self.pattern_ids != derived_pattern_ids:
            raise ValueError("pattern_ids must match patterns")
        if self.review_required_pattern_ids != _pattern_ids_for_status(
            self.patterns,
            "review_required",
        ):
            raise ValueError("review_required_pattern_ids must match patterns")
        if self.guarded_pattern_ids != _pattern_ids_for_status(
            self.patterns,
            "guarded",
        ):
            raise ValueError("guarded_pattern_ids must match patterns")
        if self.critical_priority_pattern_ids != _pattern_ids_for_priority(
            self.patterns,
            "critical",
        ):
            raise ValueError("critical_priority_pattern_ids must match patterns")
        if self.guarded_priority_pattern_ids != _pattern_ids_for_priority(
            self.patterns,
            "guarded",
        ):
            raise ValueError("guarded_priority_pattern_ids must match patterns")
        if self.hitl_required_pattern_ids != tuple(
            pattern.pattern_id for pattern in self.patterns if pattern.hitl_required
        ):
            raise ValueError("hitl_required_pattern_ids must match patterns")
        if self.applied_evaluation_pattern_ids:
            raise ValueError("applied_evaluation_pattern_ids must remain empty")
        if self.pattern_count != len(self.patterns):
            raise ValueError("pattern_count must match patterns")
        if self.review_required_pattern_count != len(self.review_required_pattern_ids):
            raise ValueError("review_required_pattern_count must match patterns")
        if self.guarded_pattern_count != len(self.guarded_pattern_ids):
            raise ValueError("guarded_pattern_count must match patterns")
        if self.hitl_required_pattern_count != len(self.hitl_required_pattern_ids):
            raise ValueError("hitl_required_pattern_count must match patterns")
        if self.highest_evaluation_learning_score != max(
            pattern.evaluation_learning_score for pattern in self.patterns
        ):
            raise ValueError("highest_evaluation_learning_score must match patterns")
        if (
            self.overall_evaluation_learning_score
            != _overall_evaluation_learning_score(self.patterns)
        ):
            raise ValueError("overall_evaluation_learning_score must match patterns")
        if self.overall_evaluation_learning_posture != _overall_evaluation_posture(
            self.patterns,
        ):
            raise ValueError("overall_evaluation_learning_posture must match patterns")
        if self.source_engine_ids != tuple(
            pattern.source_engine_id for pattern in self.patterns
        ):
            raise ValueError("source_engine_ids must match patterns")
        for pattern in self.patterns:
            if pattern.route_name != self.route_name:
                raise ValueError("pattern route_name must match plan")
            if pattern.task_type != self.task_type:
                raise ValueError("pattern task_type must match plan")
        return self


def learn_evaluations(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    adaptive_learning: AdaptiveLearningPlan | None = None,
    evaluation_contracts: EvaluationEngineContractRegistry | None = None,
) -> EvaluationLearningPlan:
    """Derive evaluation learning patterns without running evaluations."""

    route_name = _resolve_route(route)
    normalized_task_type = str(task_type).strip()
    learning_plan = adaptive_learning or evaluate_adaptive_learning_engine(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=execution_mode_id,
    )
    contracts = evaluation_contracts or evaluation_engine_contracts()
    normalized_mode = str(
        execution_mode_id or learning_plan.signals[0].execution_mode_id
    )
    execution_modes = routing_execution_mode_registry()
    if normalized_mode not in execution_modes.execution_mode_ids:
        raise ValueError("execution_mode_id must be a known execution mode")
    patterns = _patterns(
        route_name=route_name,
        task_type=learning_plan.task_type,
        execution_mode_id=normalized_mode,  # type: ignore[arg-type]
        adaptive_learning=learning_plan,
        contracts=contracts,
    )
    return EvaluationLearningPlan(
        route_name=route_name,
        task_type=learning_plan.task_type,
        source_contract_registry_role=contracts.role,
        source_contract_registry_serialization_version=(
            contracts.serialization_version
        ),
        source_adaptive_learning_serialization_version=(
            learning_plan.serialization_version
        ),
        execution_mode_ids=execution_modes.execution_mode_ids,
        patterns=patterns,
        pattern_ids=tuple(pattern.pattern_id for pattern in patterns),
        learnable_pattern_ids=_pattern_ids_for_status(patterns, "learnable"),
        review_required_pattern_ids=_pattern_ids_for_status(
            patterns,
            "review_required",
        ),
        guarded_pattern_ids=_pattern_ids_for_status(patterns, "guarded"),
        standard_priority_pattern_ids=_pattern_ids_for_priority(patterns, "standard"),
        elevated_priority_pattern_ids=_pattern_ids_for_priority(patterns, "elevated"),
        critical_priority_pattern_ids=_pattern_ids_for_priority(patterns, "critical"),
        guarded_priority_pattern_ids=_pattern_ids_for_priority(patterns, "guarded"),
        hitl_required_pattern_ids=tuple(
            pattern.pattern_id for pattern in patterns if pattern.hitl_required
        ),
        applied_evaluation_pattern_ids=(),
        pattern_count=len(patterns),
        review_required_pattern_count=len(
            _pattern_ids_for_status(patterns, "review_required")
        ),
        guarded_pattern_count=len(_pattern_ids_for_status(patterns, "guarded")),
        hitl_required_pattern_count=sum(
            1 for pattern in patterns if pattern.hitl_required
        ),
        highest_evaluation_learning_score=max(
            pattern.evaluation_learning_score for pattern in patterns
        ),
        overall_evaluation_learning_score=_overall_evaluation_learning_score(patterns),
        overall_evaluation_learning_posture=_overall_evaluation_posture(patterns),
        source_engine_ids=tuple(pattern.source_engine_id for pattern in patterns),
        advisory_actions=_plan_actions(patterns),
    )


def evaluation_learning_pattern_by_id(
    pattern_id: str,
    plan: EvaluationLearningPlan | None = None,
) -> EvaluationLearningPattern | None:
    source_plan = plan or learn_evaluations()
    for pattern in source_plan.patterns:
        if pattern.pattern_id == pattern_id:
            return pattern
    return None


def evaluation_learning_patterns_for_status(
    status: EvaluationLearningStatus,
    plan: EvaluationLearningPlan | None = None,
) -> tuple[EvaluationLearningPattern, ...]:
    source_plan = plan or learn_evaluations()
    return tuple(
        pattern for pattern in source_plan.patterns if pattern.status == status
    )


def evaluation_learning_patterns_for_priority(
    priority: EvaluationLearningPriority,
    plan: EvaluationLearningPlan | None = None,
) -> tuple[EvaluationLearningPattern, ...]:
    source_plan = plan or learn_evaluations()
    return tuple(
        pattern for pattern in source_plan.patterns if pattern.priority == priority
    )


def _patterns(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    adaptive_learning: AdaptiveLearningPlan,
    contracts: EvaluationEngineContractRegistry,
) -> tuple[EvaluationLearningPattern, ...]:
    return (
        _pattern(
            kind="critic_contract_learning",
            contract=_required_contract("creative_critic", contracts),
            learning_signal_id="adaptive_learning::workflow_pattern_learning",
            weight=150,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
            contracts=contracts,
        ),
        _pattern(
            kind="self_evaluation_contract_learning",
            contract=_required_contract("self_evaluation", contracts),
            learning_signal_id="adaptive_learning::strategy_pattern_learning",
            weight=170,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
            contracts=contracts,
        ),
        _pattern(
            kind="confidence_contract_learning",
            contract=_required_contract("creative_confidence", contracts),
            learning_signal_id="adaptive_learning::governance_feedback_learning",
            weight=160,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
            contracts=contracts,
        ),
        _pattern(
            kind="report_guardrail_learning",
            contract=_required_contract("evaluation_reports", contracts),
            learning_signal_id="adaptive_learning::runtime_guardrail_learning",
            weight=210,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            adaptive_learning=adaptive_learning,
            contracts=contracts,
        ),
    )


def _pattern(
    *,
    kind: EvaluationLearningPatternKind,
    contract: EvaluationEngineContract,
    learning_signal_id: str,
    weight: int,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    adaptive_learning: AdaptiveLearningPlan,
    contracts: EvaluationEngineContractRegistry,
) -> EvaluationLearningPattern:
    learning_signal = _required_learning_signal(learning_signal_id, adaptive_learning)
    status = _pattern_status(kind=kind, learning_signal=learning_signal)
    score = _evaluation_learning_score(
        required_input_count=len(contract.required_inputs),
        optional_input_count=len(contract.optional_inputs),
        produced_signal_count=len(contract.produced_signals),
        confidence_signal_count=len(contract.confidence_signals),
        ambiguity_signal_count=len(contract.ambiguity_signals),
        risk_signal_count=len(contract.risk_signals),
        future_execution_hook_count=len(contract.future_execution_hooks),
        relative_cost=contract.estimated_cost_metadata.relative_cost,
        relative_latency=contract.estimated_latency_metadata.relative_latency,
        learning_priority_score=learning_signal.learning_priority_score,
        evaluation_learning_weight=weight,
    )
    return EvaluationLearningPattern(
        pattern_id=f"evaluation_learning::{kind}",
        pattern_kind=kind,
        status=status,
        priority=_evaluation_priority(score, status),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        source_contract_registry_role=contracts.role,
        source_engine_id=contract.engine_id,
        source_engine_name=contract.engine_name,
        source_learning_signal_id=learning_signal.signal_id,
        source_workflow_risk_factor_id=learning_signal.source_workflow_risk_factor_id,
        required_input_count=len(contract.required_inputs),
        optional_input_count=len(contract.optional_inputs),
        produced_signal_count=len(contract.produced_signals),
        confidence_signal_count=len(contract.confidence_signals),
        ambiguity_signal_count=len(contract.ambiguity_signals),
        risk_signal_count=len(contract.risk_signals),
        future_execution_hook_count=len(contract.future_execution_hooks),
        relative_cost=contract.estimated_cost_metadata.relative_cost,
        relative_latency=contract.estimated_latency_metadata.relative_latency,
        cacheability=contract.cacheability,
        parallelization_support=contract.parallelization_support,
        learning_priority_score=learning_signal.learning_priority_score,
        evaluation_learning_weight=weight,
        evaluation_learning_score=score,
        hitl_required=learning_signal.hitl_required,
        evaluation_pattern_tags=(contract.engine_id, kind.removesuffix("_learning")),
        evaluation_summary=_evaluation_summary(kind, status),
        advisory_actions=_pattern_actions(kind),
        evidence=(
            f"engine_id:{contract.engine_id}",
            f"required_inputs:{len(contract.required_inputs)}",
            f"produced_signals:{len(contract.produced_signals)}",
            f"confidence_signals:{len(contract.confidence_signals)}",
            f"risk_signals:{len(contract.risk_signals)}",
            f"learning_signal:{learning_signal.signal_id}",
            f"learning_priority_score:{learning_signal.learning_priority_score}",
        ),
    )


def _evaluation_learning_score(
    *,
    required_input_count: int,
    optional_input_count: int,
    produced_signal_count: int,
    confidence_signal_count: int,
    ambiguity_signal_count: int,
    risk_signal_count: int,
    future_execution_hook_count: int,
    relative_cost: Literal["low", "medium"],
    relative_latency: Literal["low", "medium"],
    learning_priority_score: int,
    evaluation_learning_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            produced_signal_count * 40
            + confidence_signal_count * 45
            + ambiguity_signal_count * 35
            + risk_signal_count * 35
            + future_execution_hook_count * 25
            + learning_priority_score // 3
            + evaluation_learning_weight
            - required_input_count * 20
            - optional_input_count * 8
            - _cost_penalty(relative_cost)
            - _latency_penalty(relative_latency),
        ),
    )


def _cost_penalty(value: Literal["low", "medium"]) -> int:
    return {"low": 0, "medium": 40}[value]


def _latency_penalty(value: Literal["low", "medium"]) -> int:
    return {"low": 0, "medium": 40}[value]


def _pattern_status(
    *,
    kind: EvaluationLearningPatternKind,
    learning_signal: AdaptiveLearningSignal,
) -> EvaluationLearningStatus:
    if learning_signal.status == "guardrail" or kind == "report_guardrail_learning":
        return "guarded"
    if learning_signal.hitl_required:
        return "review_required"
    return "learnable"


def _evaluation_priority(
    score: int,
    status: EvaluationLearningStatus,
) -> EvaluationLearningPriority:
    if status == "guarded":
        return "guarded"
    if score >= 840:
        return "critical"
    if score >= 620:
        return "elevated"
    return "standard"


def _pattern_ids_for_status(
    patterns: tuple[EvaluationLearningPattern, ...],
    status: EvaluationLearningStatus,
) -> tuple[str, ...]:
    return tuple(pattern.pattern_id for pattern in patterns if pattern.status == status)


def _pattern_ids_for_priority(
    patterns: tuple[EvaluationLearningPattern, ...],
    priority: EvaluationLearningPriority,
) -> tuple[str, ...]:
    return tuple(
        pattern.pattern_id for pattern in patterns if pattern.priority == priority
    )


def _overall_evaluation_learning_score(
    patterns: tuple[EvaluationLearningPattern, ...],
) -> int:
    return sum(pattern.evaluation_learning_score for pattern in patterns) // len(
        patterns
    )


def _overall_evaluation_posture(
    patterns: tuple[EvaluationLearningPattern, ...],
) -> EvaluationLearningPosture:
    if any(pattern.status == "guarded" for pattern in patterns):
        return "guarded"
    if any(pattern.hitl_required for pattern in patterns):
        return "review_required"
    return "learnable"


def _required_learning_signal(
    signal_id: str,
    plan: AdaptiveLearningPlan,
) -> AdaptiveLearningSignal:
    for signal in plan.signals:
        if signal.signal_id == signal_id:
            return signal
    raise ValueError("required evaluation learning adaptive metadata is missing")


def _required_contract(
    engine_id: str,
    registry: EvaluationEngineContractRegistry,
) -> EvaluationEngineContract:
    for contract in registry.engine_contracts:
        if contract.engine_id == engine_id:
            return contract
    raise ValueError("required evaluation contract metadata is missing")


def _evaluation_summary(
    kind: EvaluationLearningPatternKind,
    status: EvaluationLearningStatus,
) -> str:
    if status == "guarded":
        return f"Surface {kind} as guarded evaluation metadata without execution."
    if status == "review_required":
        return f"Surface {kind} for review before future evaluation learning behavior."
    return f"Surface {kind} as learnable evaluation metadata only."


def _pattern_actions(kind: EvaluationLearningPatternKind) -> tuple[str, ...]:
    return (
        f"Expose {kind} as derived evaluation learning metadata.",
        "Keep evaluation execution, generated-output evaluation, score and "
        "confidence mutation, reflection loops, reports, routing, workflow, "
        "storage, Runtime Evolution, and output mutation disabled.",
    )


def _plan_actions(patterns: tuple[EvaluationLearningPattern, ...]) -> tuple[str, ...]:
    actions = [
        "Expose evaluation learning patterns as advisory metadata only.",
        "Keep applied evaluation pattern ids empty.",
        "Preserve evaluation execution, generated-output evaluation, score, "
        "confidence, reflection, report, routing, workflow, storage, output, "
        "and Runtime Evolution boundaries.",
    ]
    if any(pattern.hitl_required for pattern in patterns):
        actions.append("Require review before any future evaluation learning behavior.")
    return tuple(actions)


def _resolve_route(route: RouteName | str) -> RouteName:
    if isinstance(route, RouteName):
        return route
    return RouteName(str(route))
