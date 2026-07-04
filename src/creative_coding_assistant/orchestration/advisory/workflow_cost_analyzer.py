"""V5.1 workflow cost analyzer for bounded advisory estimates."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_cost_tracking_foundation import (
    AgentCostTrackingFoundationRegistry,
    agent_cost_tracking_foundation_registry,
)
from creative_coding_assistant.orchestration.artifact_engine_contracts import (
    ArtifactIntelligenceEngineContractRegistry,
    artifact_intelligence_engine_contracts,
)
from creative_coding_assistant.orchestration.creative_planning import (
    CreativeExecutionPlan,
)
from creative_coding_assistant.orchestration.execution_graph_analyzer import (
    ExecutionGraphAnalysis,
    analyze_assistant_execution_graph,
)
from creative_coding_assistant.orchestration.workflow_review import (
    MAX_WORKFLOW_REFINEMENT_COUNT,
)

WorkflowCostComponentKind = Literal[
    "workflow_node",
    "retry_reserve",
    "failure_reserve",
    "cost_source",
    "creative_plan",
]
WorkflowCostRelativeClass = Literal["none", "low", "medium", "high"]
WorkflowCostPressure = Literal["low", "medium", "high"]

WORKFLOW_COST_COMPONENT_SERIALIZATION_VERSION = "workflow_cost_component.v1"
WORKFLOW_COST_ANALYSIS_SERIALIZATION_VERSION = "workflow_cost_analysis.v1"
WORKFLOW_COST_ANALYZER_AUTHORITY_BOUNDARY = (
    "Workflow cost analysis derives bounded advisory token and relative-cost "
    "estimates from workflow graph topology, passive cost registries, and an "
    "optional creative execution plan only; it does not look up provider "
    "pricing, meter live usage, enforce budgets, route by cost, select "
    "providers or models, control workflow execution, trigger retries, mutate "
    "prompts, write storage, or modify generated output."
)

_SOURCE_COST_REGISTRIES = (
    "execution_graph_analysis",
    "agent_cost_tracking_foundation_registry",
    "artifact_intelligence_engine_contract_registry",
    "creative_execution_plan",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "provider_pricing_lookup",
    "live_usage_metering",
    "budget_enforcement",
    "cost_based_routing",
    "provider_or_model_routing",
    "workflow_control",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)
_NODE_TOKEN_ESTIMATES: dict[str, int] = {
    "intake": 80,
    "routing": 120,
    "memory": 180,
    "retrieval": 240,
    "context_assembly": 280,
    "prompt_input": 420,
    "planning": 900,
    "director": 380,
    "reasoning": 420,
    "prompt_rendering": 650,
    "generation": 2200,
    "artifact_extraction": 220,
    "preview_preparation": 180,
    "artifact_critique": 520,
    "review": 320,
    "refinement": 650,
    "finalization": 180,
    "failure": 160,
}
_RETRY_PATH_NODE_IDS = (
    "refinement",
    "generation",
    "artifact_extraction",
    "preview_preparation",
    "artifact_critique",
    "review",
)


class WorkflowCostComponent(BaseModel):
    """One bounded cost component in a workflow cost analysis."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    component_id: str = Field(min_length=1, max_length=160)
    component_kind: WorkflowCostComponentKind
    source_id: str = Field(min_length=1, max_length=120)
    relative_cost: WorkflowCostRelativeClass
    estimated_token_cost: int = Field(ge=0, le=120_000)
    cost_weight: int = Field(ge=0, le=10_000)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    pricing_lookup_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    cost_based_routing_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["workflow_cost_component.v1"] = (
        WORKFLOW_COST_COMPONENT_SERIALIZATION_VERSION
    )
    analysis_only: Literal[True] = True


class WorkflowCostAnalysis(BaseModel):
    """Bounded V5.1 advisory cost analysis for one workflow topology."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["workflow_cost_analyzer"] = "workflow_cost_analyzer"
    serialization_version: Literal["workflow_cost_analysis.v1"] = (
        WORKFLOW_COST_ANALYSIS_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=WORKFLOW_COST_ANALYZER_AUTHORITY_BOUNDARY,
        max_length=1200,
    )
    source_cost_registries: tuple[str, ...] = Field(min_length=4, max_length=4)
    source_graph_serialization_version: str = Field(min_length=1, max_length=80)
    components: tuple[WorkflowCostComponent, ...] = Field(min_length=1, max_length=80)
    component_ids: tuple[str, ...] = Field(min_length=1, max_length=80)
    node_count: int = Field(ge=1, le=40)
    component_count: int = Field(ge=1, le=80)
    critical_path_node_ids: tuple[str, ...] = Field(min_length=1, max_length=40)
    retry_path_node_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    critical_path_token_estimate: int = Field(ge=0, le=120_000)
    retry_iteration_token_estimate: int = Field(ge=0, le=120_000)
    retry_iteration_count: int = Field(ge=0, le=8)
    retry_token_reserve: int = Field(ge=0, le=120_000)
    failure_path_token_reserve: int = Field(ge=0, le=120_000)
    worst_case_token_estimate: int = Field(ge=0, le=240_000)
    creative_plan_token_estimate: int | None = Field(default=None, ge=500, le=12000)
    estimated_cost_pressure: WorkflowCostPressure
    cost_source_count: int = Field(ge=0, le=12)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    cost_analysis_implemented: Literal[True] = True
    pricing_lookup_implemented: Literal[False] = False
    live_usage_metering_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    cost_based_routing_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    analysis_only: Literal[True] = True

    @model_validator(mode="after")
    def _analysis_matches_components(self) -> Self:
        derived_component_ids = tuple(
            component.component_id for component in self.components
        )
        if len(set(derived_component_ids)) != len(derived_component_ids):
            raise ValueError("component_ids must be unique")
        if self.component_ids != derived_component_ids:
            raise ValueError("component_ids must match components")
        if self.component_count != len(self.components):
            raise ValueError("component_count must match components")
        if self.source_cost_registries != _SOURCE_COST_REGISTRIES:
            raise ValueError("source_cost_registries must match analyzer sources")

        workflow_node_tokens = {
            component.source_id: component.estimated_token_cost
            for component in self.components
            if component.component_kind == "workflow_node"
        }
        derived_critical = sum(
            workflow_node_tokens[node_id] for node_id in self.critical_path_node_ids
        )
        derived_retry_iteration = sum(
            workflow_node_tokens[node_id] for node_id in self.retry_path_node_ids
        )
        if self.critical_path_token_estimate != derived_critical:
            raise ValueError("critical_path_token_estimate must match node components")
        if self.retry_iteration_token_estimate != derived_retry_iteration:
            raise ValueError("retry_iteration_token_estimate must match retry path")
        if self.retry_token_reserve != (
            self.retry_iteration_token_estimate * self.retry_iteration_count
        ):
            raise ValueError("retry_token_reserve must match retry iteration budget")
        expected_worst_case = (
            self.critical_path_token_estimate
            + self.retry_token_reserve
            + self.failure_path_token_reserve
        )
        if self.worst_case_token_estimate != expected_worst_case:
            raise ValueError("worst_case_token_estimate must match reserves")
        if self.failure_path_token_reserve <= 0:
            raise ValueError("failure_path_token_reserve must be positive")
        return self


def analyze_workflow_cost(
    *,
    creative_plan: CreativeExecutionPlan | None = None,
    execution_graph: ExecutionGraphAnalysis | None = None,
    agent_cost_registry: AgentCostTrackingFoundationRegistry | None = None,
    artifact_engine_registry: ArtifactIntelligenceEngineContractRegistry | None = None,
) -> WorkflowCostAnalysis:
    """Return advisory workflow cost analysis without routing or enforcement."""

    graph = execution_graph or analyze_assistant_execution_graph()
    agent_costs = agent_cost_registry or agent_cost_tracking_foundation_registry()
    artifact_engines = (
        artifact_engine_registry or artifact_intelligence_engine_contracts()
    )
    components = _components(
        graph=graph,
        creative_plan=creative_plan,
        agent_costs=agent_costs,
        artifact_engines=artifact_engines,
    )
    workflow_node_tokens = {
        component.source_id: component.estimated_token_cost
        for component in components
        if component.component_kind == "workflow_node"
    }
    retry_iteration_count = _retry_iteration_count(creative_plan)
    retry_iteration_tokens = sum(
        workflow_node_tokens[node_id] for node_id in _RETRY_PATH_NODE_IDS
    )
    retry_reserve = retry_iteration_tokens * retry_iteration_count
    failure_reserve = workflow_node_tokens["failure"]
    critical_path_tokens = sum(
        workflow_node_tokens[node_id] for node_id in graph.critical_path_node_ids
    )
    worst_case_tokens = critical_path_tokens + retry_reserve + failure_reserve

    return WorkflowCostAnalysis(
        source_cost_registries=_SOURCE_COST_REGISTRIES,
        source_graph_serialization_version=graph.serialization_version,
        components=components,
        component_ids=tuple(component.component_id for component in components),
        node_count=graph.node_count,
        component_count=len(components),
        critical_path_node_ids=graph.critical_path_node_ids,
        retry_path_node_ids=_RETRY_PATH_NODE_IDS,
        critical_path_token_estimate=critical_path_tokens,
        retry_iteration_token_estimate=retry_iteration_tokens,
        retry_iteration_count=retry_iteration_count,
        retry_token_reserve=retry_reserve,
        failure_path_token_reserve=failure_reserve,
        worst_case_token_estimate=worst_case_tokens,
        creative_plan_token_estimate=(
            creative_plan.estimated_token_cost if creative_plan is not None else None
        ),
        estimated_cost_pressure=_cost_pressure(worst_case_tokens),
        cost_source_count=2,
        advisory_actions=_advisory_actions(worst_case_tokens, retry_iteration_count),
    )


def workflow_cost_component_by_id(
    component_id: str,
    analysis: WorkflowCostAnalysis | None = None,
) -> WorkflowCostComponent | None:
    """Return one cost component without executing workflow behavior."""

    source_analysis = analysis or analyze_workflow_cost()
    for component in source_analysis.components:
        if component.component_id == component_id:
            return component
    return None


def workflow_cost_components_for_kind(
    component_kind: WorkflowCostComponentKind,
    analysis: WorkflowCostAnalysis | None = None,
) -> tuple[WorkflowCostComponent, ...]:
    """Return cost components by kind without changing routing or budgets."""

    source_analysis = analysis or analyze_workflow_cost()
    return tuple(
        component
        for component in source_analysis.components
        if component.component_kind == component_kind
    )


def _components(
    *,
    graph: ExecutionGraphAnalysis,
    creative_plan: CreativeExecutionPlan | None,
    agent_costs: AgentCostTrackingFoundationRegistry,
    artifact_engines: ArtifactIntelligenceEngineContractRegistry,
) -> tuple[WorkflowCostComponent, ...]:
    components = [
        _workflow_node_component(node_id, creative_plan) for node_id in graph.node_order
    ]
    components.append(_retry_reserve_component(creative_plan))
    components.append(_failure_reserve_component())
    components.append(_agent_cost_source_component(agent_costs))
    components.append(_artifact_engine_cost_source_component(artifact_engines))
    if creative_plan is not None:
        components.append(_creative_plan_component(creative_plan))
    return tuple(components)


def _workflow_node_component(
    node_id: str,
    creative_plan: CreativeExecutionPlan | None,
) -> WorkflowCostComponent:
    estimate = _node_token_estimate(node_id, creative_plan)
    return WorkflowCostComponent(
        component_id=f"workflow_node::{node_id}",
        component_kind="workflow_node",
        source_id=node_id,
        relative_cost=_relative_cost_for_tokens(estimate),
        estimated_token_cost=estimate,
        cost_weight=max(1, estimate // 100),
        evidence=(
            "assistant_workflow_graph_topology",
            f"node:{node_id}",
        ),
        advisory_actions=_node_advisory_actions(node_id, estimate),
    )


def _retry_reserve_component(
    creative_plan: CreativeExecutionPlan | None,
) -> WorkflowCostComponent:
    retry_count = _retry_iteration_count(creative_plan)
    iteration_estimate = sum(
        _node_token_estimate(node_id, creative_plan) for node_id in _RETRY_PATH_NODE_IDS
    )
    return WorkflowCostComponent(
        component_id="reserve::retry_path",
        component_kind="retry_reserve",
        source_id="review_refinement_generation_loop",
        relative_cost=_relative_cost_for_tokens(iteration_estimate * retry_count),
        estimated_token_cost=iteration_estimate * retry_count,
        cost_weight=max(1, (iteration_estimate * retry_count) // 100),
        evidence=(
            "review_to_refinement_retry_edge",
            "refinement_to_generation_reentry_edge",
        ),
        advisory_actions=(
            "Keep retry reserve advisory until strategy selection.",
            "Do not trigger retries from cost analysis.",
        ),
    )


def _failure_reserve_component() -> WorkflowCostComponent:
    estimate = _NODE_TOKEN_ESTIMATES["failure"]
    return WorkflowCostComponent(
        component_id="reserve::failure_path",
        component_kind="failure_reserve",
        source_id="workflow_failure_path",
        relative_cost="low",
        estimated_token_cost=estimate,
        cost_weight=max(1, estimate // 100),
        evidence=("failure_terminal_path",),
        advisory_actions=("Preserve normalized failure response budget.",),
    )


def _agent_cost_source_component(
    registry: AgentCostTrackingFoundationRegistry,
) -> WorkflowCostComponent:
    weight = sum(_cost_class_weight(cost_class) for cost_class in registry.cost_classes)
    return WorkflowCostComponent(
        component_id="cost_source::agent_cost_tracking_foundation",
        component_kind="cost_source",
        source_id=registry.role,
        relative_cost=_relative_cost_for_weight(weight),
        estimated_token_cost=0,
        cost_weight=weight,
        evidence=(
            registry.serialization_version,
            f"profiles:{registry.profile_count}",
        ),
        advisory_actions=("Use passive agent cost metadata as a planning signal.",),
    )


def _artifact_engine_cost_source_component(
    registry: ArtifactIntelligenceEngineContractRegistry,
) -> WorkflowCostComponent:
    weight = sum(
        _cost_class_weight(contract.estimated_cost_metadata.relative_cost)
        for contract in registry.engine_contracts
    )
    return WorkflowCostComponent(
        component_id="cost_source::artifact_engine_contracts",
        component_kind="cost_source",
        source_id=registry.role,
        relative_cost=_relative_cost_for_weight(weight),
        estimated_token_cost=0,
        cost_weight=weight,
        evidence=(
            registry.serialization_version,
            f"contracts:{registry.contract_count}",
        ),
        advisory_actions=("Use engine cost metadata as a workflow planning signal.",),
    )


def _creative_plan_component(plan: CreativeExecutionPlan) -> WorkflowCostComponent:
    return WorkflowCostComponent(
        component_id="creative_plan::estimated_token_cost",
        component_kind="creative_plan",
        source_id="creative_execution_plan",
        relative_cost=_relative_cost_for_tokens(plan.estimated_token_cost),
        estimated_token_cost=plan.estimated_token_cost,
        cost_weight=max(1, plan.estimated_token_cost // 100),
        evidence=(
            f"candidate_count:{plan.candidate_count}",
            f"refinement_budget:{plan.refinement_budget}",
            f"complexity:{plan.expected_complexity}",
        ),
        advisory_actions=("Use plan token estimate for generation node analysis.",),
    )


def _node_token_estimate(
    node_id: str,
    creative_plan: CreativeExecutionPlan | None,
) -> int:
    if node_id == "generation" and creative_plan is not None:
        return creative_plan.estimated_token_cost
    return _NODE_TOKEN_ESTIMATES[node_id]


def _retry_iteration_count(creative_plan: CreativeExecutionPlan | None) -> int:
    if creative_plan is None:
        return MAX_WORKFLOW_REFINEMENT_COUNT
    return min(creative_plan.refinement_budget, MAX_WORKFLOW_REFINEMENT_COUNT)


def _relative_cost_for_tokens(tokens: int) -> WorkflowCostRelativeClass:
    if tokens == 0:
        return "none"
    if tokens < 2000:
        return "low"
    if tokens < 7000:
        return "medium"
    return "high"


def _relative_cost_for_weight(weight: int) -> WorkflowCostRelativeClass:
    if weight <= 0:
        return "none"
    if weight <= 8:
        return "low"
    if weight <= 16:
        return "medium"
    return "high"


def _cost_class_weight(cost_class: str) -> int:
    return {"none": 0, "low": 1, "medium": 2, "high": 3}.get(cost_class, 1)


def _cost_pressure(tokens: int) -> WorkflowCostPressure:
    if tokens < 8_000:
        return "low"
    if tokens < 18_000:
        return "medium"
    return "high"


def _node_advisory_actions(
    node_id: str,
    estimate: int,
) -> tuple[str, ...]:
    if node_id == "generation":
        return ("Track generation token estimate without selecting providers.",)
    if node_id == "retrieval":
        return ("Track retrieval context contribution before compression planning.",)
    if estimate >= 600:
        return ("Keep high local planning cost visible for later pruning decisions.",)
    return ("No immediate cost action; preserve workflow order.",)


def _advisory_actions(
    worst_case_tokens: int,
    retry_iteration_count: int,
) -> tuple[str, ...]:
    actions = [
        "Expose workflow cost pressure as advisory metadata only.",
        "Preserve explicit provider and model routing boundaries.",
    ]
    if retry_iteration_count > 0:
        actions.append("Account for bounded retry reserve before pruning decisions.")
    if worst_case_tokens >= 18_000:
        actions.append("Flag high advisory pressure for later strategy selection.")
    return tuple(actions)
