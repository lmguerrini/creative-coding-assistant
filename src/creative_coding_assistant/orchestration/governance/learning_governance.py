"""V6.1 advisory learning governance and safety."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.adaptive_learning_engine import (
    AdaptiveLearningPlan,
    evaluate_adaptive_learning_engine,
)
from creative_coding_assistant.orchestration.continuous_improvement_signals import (
    ContinuousImprovementSignalPlan,
    derive_continuous_improvement_signals,
)
from creative_coding_assistant.orchestration.failure_pattern_discovery import (
    FailurePatternDiscoveryPlan,
    discover_failure_patterns,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)
from creative_coding_assistant.orchestration.success_pattern_discovery import (
    SuccessPatternDiscoveryPlan,
    discover_success_patterns,
)

LearningGovernancePolicyKind = Literal[
    "learning_memory_boundary",
    "feedback_loop_boundary",
    "learning_policy_boundary",
    "hitl_governance_boundary",
    "safety_no_automation_boundary",
]
LearningGovernanceStatus = Literal["blocked", "review_required", "guarded"]
LearningGovernancePriority = Literal["standard", "elevated", "critical", "guarded"]
LearningGovernancePosture = Literal["blocked", "review_required", "guarded"]

LEARNING_GOVERNANCE_POLICY_SERIALIZATION_VERSION = "learning_governance_policy.v1"
LEARNING_GOVERNANCE_PLAN_SERIALIZATION_VERSION = "learning_governance_plan.v1"
LEARNING_GOVERNANCE_AUTHORITY_BOUNDARY = (
    "V6.1 learning governance describes learning memory, feedback loop, "
    "policy, HITL, explainability, and no-automation boundaries as advisory "
    "metadata only; it does not persist learning memory, apply feedback, "
    "update policies, enforce policies, emit HITL requests, request human "
    "input, change provider or model routing, execute providers, invoke "
    "agents, allocate resources, observe runtime outcomes, evaluate generated "
    "output, execute or control workflows, mutate workflow graphs, trigger "
    "retries or refinements, mutate prompts, write storage, modify generated "
    "output, or apply Runtime Evolution."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "learning_memory_persistence",
    "learning_feedback_application",
    "learning_policy_update",
    "learning_policy_enforcement",
    "hitl_request_emission",
    "human_input_request",
    "provider_or_model_routing",
    "automatic_provider_switching",
    "automatic_model_switching",
    "provider_execution",
    "agent_invocation",
    "resource_allocation",
    "runtime_outcome_observation",
    "generated_output_evaluation",
    "workflow_control",
    "workflow_graph_mutation",
    "workflow_execution",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)


class LearningGovernancePolicy(BaseModel):
    """One advisory V6.1 learning governance policy."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    policy_id: str = Field(min_length=1, max_length=180)
    policy_kind: LearningGovernancePolicyKind
    status: LearningGovernanceStatus
    priority: LearningGovernancePriority
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    source_plan_roles: tuple[str, ...] = Field(min_length=1, max_length=4)
    governed_surface: str = Field(min_length=1, max_length=140)
    review_requirement: str = Field(min_length=1, max_length=260)
    explainability_requirement: str = Field(min_length=1, max_length=260)
    no_automation_boundary: str = Field(min_length=1, max_length=360)
    source_signal_count: int = Field(ge=0, le=20)
    source_guarded_count: int = Field(ge=0, le=20)
    source_hitl_required_count: int = Field(ge=0, le=20)
    governance_weight: int = Field(ge=0, le=240)
    governance_score: int = Field(ge=0, le=1_000)
    hitl_required_before_application: bool
    governance_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    governance_summary: str = Field(min_length=1, max_length=360)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    learning_governance_implemented: Literal[True] = True
    governance_policy_metadata_implemented: Literal[True] = True
    learning_memory_persistence_implemented: Literal[False] = False
    learning_feedback_application_implemented: Literal[False] = False
    learning_policy_update_implemented: Literal[False] = False
    learning_policy_enforcement_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    resource_allocation_implemented: Literal[False] = False
    runtime_outcome_observation_implemented: Literal[False] = False
    generated_output_evaluation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["learning_governance_policy.v1"] = (
        LEARNING_GOVERNANCE_POLICY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _policy_matches_contract(self) -> Self:
        if self.policy_id != f"learning_governance::{self.policy_kind}":
            raise ValueError("policy_id must match policy_kind")
        if self.governance_score != _governance_score(
            source_signal_count=self.source_signal_count,
            source_guarded_count=self.source_guarded_count,
            source_hitl_required_count=self.source_hitl_required_count,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("governance_score must combine source counts")
        if self.priority != _governance_priority(self.governance_score, self.status):
            raise ValueError("priority must match score and status")
        if self.status == "guarded" and not self.hitl_required_before_application:
            raise ValueError("guarded governance policies require HITL posture")
        return self


class LearningGovernancePlan(BaseModel):
    """Bounded V6.1 advisory learning governance plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["learning_governance"] = "learning_governance"
    serialization_version: Literal["learning_governance_plan.v1"] = (
        LEARNING_GOVERNANCE_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=LEARNING_GOVERNANCE_AUTHORITY_BOUNDARY,
        max_length=1700,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_adaptive_learning_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_plan_roles: tuple[str, ...] = Field(min_length=4, max_length=4)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    policies: tuple[LearningGovernancePolicy, ...] = Field(
        min_length=5,
        max_length=5,
    )
    policy_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    blocked_policy_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    review_required_policy_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    guarded_policy_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    hitl_required_policy_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    applied_governance_policy_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    policy_count: int = Field(ge=5, le=5)
    blocked_policy_count: int = Field(ge=0, le=5)
    review_required_policy_count: int = Field(ge=0, le=5)
    guarded_policy_count: int = Field(ge=0, le=5)
    hitl_required_policy_count: int = Field(ge=0, le=5)
    highest_governance_score: int = Field(ge=0, le=1_000)
    overall_governance_score: int = Field(ge=0, le=1_000)
    overall_governance_posture: LearningGovernancePosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    learning_governance_implemented: Literal[True] = True
    governance_policy_metadata_implemented: Literal[True] = True
    learning_memory_persistence_implemented: Literal[False] = False
    learning_feedback_application_implemented: Literal[False] = False
    learning_policy_update_implemented: Literal[False] = False
    learning_policy_enforcement_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    resource_allocation_implemented: Literal[False] = False
    runtime_outcome_observation_implemented: Literal[False] = False
    generated_output_evaluation_implemented: Literal[False] = False
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
    def _plan_matches_policies(self) -> Self:
        derived_policy_ids = tuple(policy.policy_id for policy in self.policies)
        if self.policy_ids != derived_policy_ids:
            raise ValueError("policy_ids must match policies")
        if self.blocked_policy_ids != _policy_ids_for_status(self.policies, "blocked"):
            raise ValueError("blocked_policy_ids must match policies")
        if self.review_required_policy_ids != _policy_ids_for_status(
            self.policies,
            "review_required",
        ):
            raise ValueError("review_required_policy_ids must match policies")
        if self.guarded_policy_ids != _policy_ids_for_status(self.policies, "guarded"):
            raise ValueError("guarded_policy_ids must match policies")
        if self.hitl_required_policy_ids != tuple(
            policy.policy_id
            for policy in self.policies
            if policy.hitl_required_before_application
        ):
            raise ValueError("hitl_required_policy_ids must match policies")
        if self.applied_governance_policy_ids:
            raise ValueError("applied_governance_policy_ids must remain empty")
        if self.policy_count != len(self.policies):
            raise ValueError("policy_count must match policies")
        if self.highest_governance_score != max(
            policy.governance_score for policy in self.policies
        ):
            raise ValueError("highest_governance_score must match policies")
        if self.overall_governance_score != _overall_governance_score(self.policies):
            raise ValueError("overall_governance_score must match policies")
        if self.overall_governance_posture != _overall_governance_posture(
            self.policies,
        ):
            raise ValueError("overall_governance_posture must match policies")
        for policy in self.policies:
            if policy.route_name != self.route_name:
                raise ValueError("policy route_name must match plan")
            if policy.task_type != self.task_type:
                raise ValueError("policy task_type must match plan")
        return self


def build_learning_governance(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    adaptive_learning: AdaptiveLearningPlan | None = None,
    continuous_improvement: ContinuousImprovementSignalPlan | None = None,
    success_patterns: SuccessPatternDiscoveryPlan | None = None,
    failure_patterns: FailurePatternDiscoveryPlan | None = None,
) -> LearningGovernancePlan:
    """Build learning governance metadata without enforcing policies."""

    route_name = _resolve_route(route)
    normalized_task_type = str(task_type).strip()
    learning_plan = adaptive_learning or evaluate_adaptive_learning_engine(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=execution_mode_id,
    )
    improvement_plan = continuous_improvement or derive_continuous_improvement_signals(
        route=route_name,
        task_type=learning_plan.task_type,
        execution_mode_id=execution_mode_id,
        adaptive_learning=learning_plan,
    )
    success_plan = success_patterns or discover_success_patterns(
        route=route_name,
        task_type=learning_plan.task_type,
        execution_mode_id=execution_mode_id,
        adaptive_learning=learning_plan,
        continuous_improvement=improvement_plan,
    )
    failure_plan = failure_patterns or discover_failure_patterns(
        route=route_name,
        task_type=learning_plan.task_type,
        execution_mode_id=execution_mode_id,
        adaptive_learning=learning_plan,
        continuous_improvement=improvement_plan,
    )
    normalized_mode = str(
        execution_mode_id or learning_plan.signals[0].execution_mode_id
    )
    execution_modes = routing_execution_mode_registry()
    if normalized_mode not in execution_modes.execution_mode_ids:
        raise ValueError("execution_mode_id must be a known execution mode")
    policies = _policies(
        route_name=route_name,
        task_type=learning_plan.task_type,
        execution_mode_id=normalized_mode,  # type: ignore[arg-type]
        adaptive_learning=learning_plan,
        continuous_improvement=improvement_plan,
        success_patterns=success_plan,
        failure_patterns=failure_plan,
    )
    return LearningGovernancePlan(
        route_name=route_name,
        task_type=learning_plan.task_type,
        source_adaptive_learning_serialization_version=(
            learning_plan.serialization_version
        ),
        source_plan_roles=(
            improvement_plan.role,
            success_plan.role,
            failure_plan.role,
            learning_plan.role,
        ),
        execution_mode_ids=execution_modes.execution_mode_ids,
        policies=policies,
        policy_ids=tuple(policy.policy_id for policy in policies),
        blocked_policy_ids=_policy_ids_for_status(policies, "blocked"),
        review_required_policy_ids=_policy_ids_for_status(policies, "review_required"),
        guarded_policy_ids=_policy_ids_for_status(policies, "guarded"),
        hitl_required_policy_ids=tuple(
            policy.policy_id
            for policy in policies
            if policy.hitl_required_before_application
        ),
        applied_governance_policy_ids=(),
        policy_count=len(policies),
        blocked_policy_count=len(_policy_ids_for_status(policies, "blocked")),
        review_required_policy_count=len(
            _policy_ids_for_status(policies, "review_required")
        ),
        guarded_policy_count=len(_policy_ids_for_status(policies, "guarded")),
        hitl_required_policy_count=sum(
            1 for policy in policies if policy.hitl_required_before_application
        ),
        highest_governance_score=max(policy.governance_score for policy in policies),
        overall_governance_score=_overall_governance_score(policies),
        overall_governance_posture=_overall_governance_posture(policies),
        advisory_actions=_plan_actions(policies),
    )


def learning_governance_policy_by_id(
    policy_id: str,
    plan: LearningGovernancePlan | None = None,
) -> LearningGovernancePolicy | None:
    source_plan = plan or build_learning_governance()
    for policy in source_plan.policies:
        if policy.policy_id == policy_id:
            return policy
    return None


def learning_governance_policies_for_status(
    status: LearningGovernanceStatus,
    plan: LearningGovernancePlan | None = None,
) -> tuple[LearningGovernancePolicy, ...]:
    source_plan = plan or build_learning_governance()
    return tuple(policy for policy in source_plan.policies if policy.status == status)


def _policies(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    adaptive_learning: AdaptiveLearningPlan,
    continuous_improvement: ContinuousImprovementSignalPlan,
    success_patterns: SuccessPatternDiscoveryPlan,
    failure_patterns: FailurePatternDiscoveryPlan,
) -> tuple[LearningGovernancePolicy, ...]:
    source_roles = (
        continuous_improvement.role,
        success_patterns.role,
        failure_patterns.role,
        adaptive_learning.role,
    )
    signal_count = (
        continuous_improvement.signal_count
        + success_patterns.pattern_count
        + failure_patterns.pattern_count
    )
    guarded_count = (
        continuous_improvement.guarded_signal_count
        + success_patterns.guarded_pattern_count
        + failure_patterns.guarded_pattern_count
    )
    hitl_count = (
        continuous_improvement.hitl_required_signal_count
        + success_patterns.hitl_required_pattern_count
        + failure_patterns.hitl_required_pattern_count
    )
    return (
        _policy(
            kind="learning_memory_boundary",
            status="blocked",
            governed_surface="Learning Memory",
            review_requirement="Memory persistence remains unavailable in V6.1.",
            explainability_requirement="Explain that learning memory is metadata only.",
            no_automation_boundary="Do not persist memory or write learning storage.",
            weight=220,
            source_roles=source_roles,
            source_signal_count=signal_count,
            source_guarded_count=guarded_count,
            source_hitl_required_count=hitl_count,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
        ),
        _policy(
            kind="feedback_loop_boundary",
            status="review_required",
            governed_surface="Learning Feedback Loop",
            review_requirement="Feedback application requires future HITL scope.",
            explainability_requirement="Expose candidate feedback as advisory only.",
            no_automation_boundary="Do not apply feedback to strategies or policies.",
            weight=190,
            source_roles=source_roles,
            source_signal_count=signal_count,
            source_guarded_count=guarded_count,
            source_hitl_required_count=hitl_count,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
        ),
        _policy(
            kind="learning_policy_boundary",
            status="review_required",
            governed_surface="Learning Policies",
            review_requirement="Policy updates require explicit human approval.",
            explainability_requirement="Explain policy posture before future use.",
            no_automation_boundary="Do not update or enforce policies automatically.",
            weight=180,
            source_roles=source_roles,
            source_signal_count=signal_count,
            source_guarded_count=guarded_count,
            source_hitl_required_count=hitl_count,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
        ),
        _policy(
            kind="hitl_governance_boundary",
            status="guarded",
            governed_surface="Learning Governance HITL",
            review_requirement="HITL requests are represented, not emitted.",
            explainability_requirement="Expose why future automation needs review.",
            no_automation_boundary="Do not emit HITL or request human input.",
            weight=210,
            source_roles=source_roles,
            source_signal_count=signal_count,
            source_guarded_count=guarded_count,
            source_hitl_required_count=hitl_count,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
        ),
        _policy(
            kind="safety_no_automation_boundary",
            status="guarded",
            governed_surface="Learning Safety Policies",
            review_requirement="Automation remains blocked without future gate.",
            explainability_requirement="Explain blocked automation boundaries.",
            no_automation_boundary=(
                "Do not automate routing, execution, retries, or Runtime Evolution."
            ),
            weight=230,
            source_roles=source_roles,
            source_signal_count=signal_count,
            source_guarded_count=guarded_count,
            source_hitl_required_count=hitl_count,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
        ),
    )


def _policy(
    *,
    kind: LearningGovernancePolicyKind,
    status: LearningGovernanceStatus,
    governed_surface: str,
    review_requirement: str,
    explainability_requirement: str,
    no_automation_boundary: str,
    weight: int,
    source_roles: tuple[str, ...],
    source_signal_count: int,
    source_guarded_count: int,
    source_hitl_required_count: int,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
) -> LearningGovernancePolicy:
    score = _governance_score(
        source_signal_count=source_signal_count,
        source_guarded_count=source_guarded_count,
        source_hitl_required_count=source_hitl_required_count,
        governance_weight=weight,
    )
    return LearningGovernancePolicy(
        policy_id=f"learning_governance::{kind}",
        policy_kind=kind,
        status=status,
        priority=_governance_priority(score, status),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        source_plan_roles=source_roles,
        governed_surface=governed_surface,
        review_requirement=review_requirement,
        explainability_requirement=explainability_requirement,
        no_automation_boundary=no_automation_boundary,
        source_signal_count=source_signal_count,
        source_guarded_count=source_guarded_count,
        source_hitl_required_count=source_hitl_required_count,
        governance_weight=weight,
        governance_score=score,
        hitl_required_before_application=True,
        governance_tags=(kind, governed_surface.lower().replace(" ", "_")),
        governance_summary=_governance_summary(kind, status),
        advisory_actions=_policy_actions(kind),
        evidence=(
            f"source_signal_count:{source_signal_count}",
            f"source_guarded_count:{source_guarded_count}",
            f"source_hitl_required_count:{source_hitl_required_count}",
            f"governed_surface:{governed_surface}",
        ),
    )


def _governance_score(
    *,
    source_signal_count: int,
    source_guarded_count: int,
    source_hitl_required_count: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            source_signal_count * 20
            + source_guarded_count * 55
            + source_hitl_required_count * 35
            + governance_weight,
        ),
    )


def _governance_priority(
    score: int,
    status: LearningGovernanceStatus,
) -> LearningGovernancePriority:
    if status == "guarded":
        return "guarded"
    if score >= 840:
        return "critical"
    if score >= 620:
        return "elevated"
    return "standard"


def _policy_ids_for_status(
    policies: tuple[LearningGovernancePolicy, ...],
    status: LearningGovernanceStatus,
) -> tuple[str, ...]:
    return tuple(policy.policy_id for policy in policies if policy.status == status)


def _overall_governance_score(
    policies: tuple[LearningGovernancePolicy, ...],
) -> int:
    return sum(policy.governance_score for policy in policies) // len(policies)


def _overall_governance_posture(
    policies: tuple[LearningGovernancePolicy, ...],
) -> LearningGovernancePosture:
    if any(policy.status == "guarded" for policy in policies):
        return "guarded"
    if any(policy.status == "blocked" for policy in policies):
        return "blocked"
    return "review_required"


def _governance_summary(
    kind: LearningGovernancePolicyKind,
    status: LearningGovernanceStatus,
) -> str:
    if status == "guarded":
        return f"Surface {kind} as guarded governance metadata only."
    if status == "blocked":
        return f"Surface {kind} as blocked for V6.1 runtime behavior."
    return f"Surface {kind} for review before future learning behavior."


def _policy_actions(kind: LearningGovernancePolicyKind) -> tuple[str, ...]:
    return (
        f"Expose {kind} as advisory learning governance metadata.",
        "Keep memory persistence, feedback, policy update/enforcement, HITL "
        "emission, workflow control, storage, Runtime Evolution, and output "
        "mutation disabled.",
    )


def _plan_actions(
    policies: tuple[LearningGovernancePolicy, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose learning governance policies as advisory metadata only.",
        "Keep applied governance policy ids empty.",
        "Preserve memory, feedback, policy, HITL, routing, provider, workflow, "
        "storage, output, and Runtime Evolution boundaries.",
    ]
    if any(policy.hitl_required_before_application for policy in policies):
        actions.append("Require HITL before any future governance application.")
    return tuple(actions)


def _resolve_route(route: RouteName | str) -> RouteName:
    if isinstance(route, RouteName):
        return route
    return RouteName(str(route))
