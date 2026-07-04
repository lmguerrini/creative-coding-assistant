"""V5.5 adaptive execution architecture consistency metadata."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.adaptive_cost_quality_optimizer import (
    optimize_adaptive_cost_quality,
)
from creative_coding_assistant.orchestration.adaptive_escalation_optimizer import (
    optimize_escalation_policy,
)
from creative_coding_assistant.orchestration.adaptive_execution_policy_engine import (
    evaluate_adaptive_execution_policy,
)
from creative_coding_assistant.orchestration.adaptive_execution_strategy_selection import (
    select_dynamic_execution_strategy,
)
from creative_coding_assistant.orchestration.adaptive_hybrid_workflow_optimizer import (
    optimize_hybrid_workflow,
)
from creative_coding_assistant.orchestration.adaptive_latency_optimizer import (
    optimize_adaptive_latency,
)
from creative_coding_assistant.orchestration.adaptive_policy_explainability import (
    explain_adaptive_policy,
)
from creative_coding_assistant.orchestration.agent_activation_optimizer import (
    optimize_agent_activation,
)
from creative_coding_assistant.orchestration.agent_diversity_optimizer import (
    optimize_agent_diversity,
)
from creative_coding_assistant.orchestration.creative_exploration_optimizer import (
    optimize_creative_exploration,
)
from creative_coding_assistant.orchestration.dynamic_agent_allocation import (
    allocate_dynamic_agents,
)
from creative_coding_assistant.orchestration.dynamic_resource_allocation import (
    allocate_dynamic_resources,
)
from creative_coding_assistant.orchestration.emergence_optimizer import (
    optimize_emergence,
)
from creative_coding_assistant.orchestration.execution_confidence_engine import (
    evaluate_execution_confidence,
)
from creative_coding_assistant.orchestration.reflection_budget_optimizer import (
    optimize_reflection_budget,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.workflow_risk_engine import (
    evaluate_workflow_risk,
)
from creative_coding_assistant.orchestration.workflow_self_tuning_policies import (
    plan_workflow_self_tuning_policies,
)

AdaptiveExecutionArchitectureLayer = Literal[
    "hybrid_workflow_boundary",
    "escalation_policy_boundary",
    "agent_resource_boundary",
    "cost_latency_boundary",
    "execution_strategy_boundary",
    "confidence_risk_boundary",
    "creative_adaptation_boundary",
    "explainability_boundary",
]
AdaptiveExecutionArchitectureStage = Literal["v5_5_architecture_consistency_pass"]
AdaptiveExecutionArchitectureStatus = Literal["pass"]

ADAPTIVE_EXECUTION_ARCHITECTURE_RECORD_SERIALIZATION_VERSION = (
    "adaptive_execution_architecture_consistency_record.v1"
)
ADAPTIVE_EXECUTION_ARCHITECTURE_REGISTRY_SERIALIZATION_VERSION = (
    "adaptive_execution_architecture_consistency_registry.v1"
)
ADAPTIVE_EXECUTION_ARCHITECTURE_AUTHORITY_BOUNDARY = (
    "V5.5 adaptive execution architecture consistency metadata checks "
    "advisory surface coverage, controlled adaptive policy application, "
    "serialization, route consistency, runtime boundaries, V4 compatibility, "
    "and version-level HITL/runtime evolution rules; it allows only the "
    "controlled V5.5 policy decision surface to apply allow/confirm/block "
    "policy semantics and does not change provider or model routing, execute "
    "providers, invoke agents, allocate resources, enforce budgets, emit HITL "
    "requests, control or execute workflows, mutate workflow graphs, trigger "
    "retries, mutate prompts, write storage, modify generated output, or "
    "apply Runtime Evolution."
)

_ROUTE_NAME = RouteName.GENERATE
_ARCHITECTURE_LAYERS: tuple[AdaptiveExecutionArchitectureLayer, ...] = (
    "hybrid_workflow_boundary",
    "escalation_policy_boundary",
    "agent_resource_boundary",
    "cost_latency_boundary",
    "execution_strategy_boundary",
    "confidence_risk_boundary",
    "creative_adaptation_boundary",
    "explainability_boundary",
)
_VALIDATED_VERSION_RULES = (
    "v5_5_surface_role_declared",
    "serialization_version_declared",
    "advisory_or_controlled_policy_declared",
    "controlled_policy_application_scoped",
    "provider_model_routing_not_applied",
    "provider_execution_not_applied",
    "agent_invocation_blocked",
    "workflow_control_blocked",
    "generated_output_mutation_blocked",
    "runtime_evolution_not_applied",
    "v4_boundary_compatibility_confirmed",
    "human_gate_not_emitted",
)
_PASSIVE_BOUNDARY_FLAGS = (
    "uncontrolled_policy_application_blocked",
    "strategy_application_blocked",
    "provider_model_routing_blocked",
    "provider_execution_blocked",
    "agent_invocation_blocked",
    "resource_allocation_blocked",
    "budget_enforcement_blocked",
    "hitl_emission_blocked",
    "workflow_control_blocked",
    "workflow_graph_mutation_blocked",
    "workflow_execution_blocked",
    "retry_triggering_blocked",
    "prompt_mutation_blocked",
    "storage_write_blocked",
    "generated_output_mutation_blocked",
    "runtime_evolution_blocked",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "uncontrolled_adaptive_policy_application",
    "strategy_application",
    "provider_or_model_routing",
    "provider_execution",
    "agent_invocation",
    "agent_activation",
    "resource_allocation",
    "runtime_resource_measurement",
    "budget_enforcement",
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
_ALLOWED_CONTROLLED_ACTIVE_FLAGS = (
    "policy_application_implemented",
    "execution_policy_application_implemented",
)
_ACTIVE_RUNTIME_FLAGS = (
    "policy_application_implemented",
    "execution_policy_application_implemented",
    "strategy_application_implemented",
    "routing_application_implemented",
    "risk_decision_application_implemented",
    "confidence_application_implemented",
    "self_tuning_application_implemented",
    "emergence_behavior_application_implemented",
    "agent_diversity_behavior_application_implemented",
    "reflection_budget_enforcement_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "automatic_provider_switching_implemented",
    "automatic_model_switching_implemented",
    "agent_invocation_implemented",
    "agent_activation_implemented",
    "agent_instantiation_implemented",
    "runtime_agent_allocation_implemented",
    "resource_allocation_implemented",
    "runtime_resource_measurement_implemented",
    "budget_enforcement_implemented",
    "hitl_request_emitted",
    "human_review_request_implemented",
    "workflow_control_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_execution_implemented",
    "graph_compilation_implemented",
    "retry_triggering_implemented",
    "refinement_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "storage_write_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
)
_COUNT_FIELD_CANDIDATES = (
    "candidate_count",
    "decision_count",
    "strategy_count",
    "allocation_count",
    "policy_count",
    "signal_count",
    "factor_count",
    "explanation_count",
)
_SURFACE_LAYERS: tuple[tuple[str, AdaptiveExecutionArchitectureLayer], ...] = (
    ("adaptive_hybrid_workflow_optimizer", "hybrid_workflow_boundary"),
    ("adaptive_escalation_optimizer", "escalation_policy_boundary"),
    ("agent_activation_optimizer", "agent_resource_boundary"),
    ("adaptive_cost_quality_optimizer", "cost_latency_boundary"),
    ("adaptive_latency_optimizer", "cost_latency_boundary"),
    ("adaptive_execution_strategy_selection", "execution_strategy_boundary"),
    ("adaptive_execution_policy_engine", "execution_strategy_boundary"),
    ("dynamic_agent_allocation", "agent_resource_boundary"),
    ("dynamic_resource_allocation", "agent_resource_boundary"),
    ("workflow_self_tuning_policies", "execution_strategy_boundary"),
    ("execution_confidence_engine", "confidence_risk_boundary"),
    ("workflow_risk_engine", "confidence_risk_boundary"),
    ("creative_exploration_optimizer", "creative_adaptation_boundary"),
    ("emergence_optimizer", "creative_adaptation_boundary"),
    ("agent_diversity_optimizer", "agent_resource_boundary"),
    ("reflection_budget_optimizer", "cost_latency_boundary"),
    ("adaptive_policy_explainability", "explainability_boundary"),
)


class AdaptiveExecutionArchitectureConsistencyRecord(BaseModel):
    """One V5.5 architecture consistency record."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    surface_id: str = Field(min_length=1, max_length=140)
    architecture_layer: AdaptiveExecutionArchitectureLayer
    architecture_stage: AdaptiveExecutionArchitectureStage = (
        "v5_5_architecture_consistency_pass"
    )
    source_role: str = Field(min_length=1, max_length=140)
    source_serialization_version: str = Field(min_length=1, max_length=140)
    source_route_name: RouteName | None = None
    source_count_field: str = Field(min_length=1, max_length=80)
    source_count: int = Field(ge=1, le=120)
    validated_version_rules: tuple[str, ...] = Field(min_length=12, max_length=12)
    passive_boundary_flags: tuple[str, ...] = Field(min_length=16, max_length=16)
    source_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=48,
    )
    source_active_runtime_flags: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=40,
    )
    missing_coverage_items: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=16,
    )
    source_advisory_only_declared: bool = True
    source_controlled_policy_declared: bool = False
    v5_architecture_consistency_confirmed: Literal[True] = True
    v4_boundary_compatibility_confirmed: Literal[True] = True
    version_runtime_rules_confirmed: Literal[True] = True
    architecture_consistency_status: AdaptiveExecutionArchitectureStatus = "pass"
    policy_application_implemented: bool = False
    execution_policy_application_implemented: bool = False
    strategy_application_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    resource_allocation_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    hitl_emission_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal[
        "adaptive_execution_architecture_consistency_record.v1"
    ] = ADAPTIVE_EXECUTION_ARCHITECTURE_RECORD_SERIALIZATION_VERSION
    advisory_only: Literal[True] = True


class AdaptiveExecutionArchitectureConsistencyRegistry(BaseModel):
    """V5.5 adaptive execution architecture consistency registry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["adaptive_execution_architecture_consistency_registry"] = (
        "adaptive_execution_architecture_consistency_registry"
    )
    serialization_version: Literal[
        "adaptive_execution_architecture_consistency_registry.v1"
    ] = ADAPTIVE_EXECUTION_ARCHITECTURE_REGISTRY_SERIALIZATION_VERSION
    authority_boundary: str = Field(
        default=ADAPTIVE_EXECUTION_ARCHITECTURE_AUTHORITY_BOUNDARY,
        max_length=2100,
    )
    architecture_stage: AdaptiveExecutionArchitectureStage = (
        "v5_5_architecture_consistency_pass"
    )
    route_name: RouteName = _ROUTE_NAME
    records: tuple[AdaptiveExecutionArchitectureConsistencyRecord, ...] = Field(
        min_length=17,
        max_length=17,
    )
    surface_ids: tuple[str, ...] = Field(min_length=17, max_length=17)
    record_count: int = Field(ge=17, le=17)
    architecture_layers: tuple[AdaptiveExecutionArchitectureLayer, ...] = Field(
        min_length=8,
        max_length=8,
    )
    validated_version_rules: tuple[str, ...] = Field(min_length=12, max_length=12)
    passive_boundary_flags: tuple[str, ...] = Field(min_length=16, max_length=16)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    all_surfaces_covered: Literal[True] = True
    route_consistency_confirmed: Literal[True] = True
    no_active_runtime_flags: Literal[False] = False
    controlled_active_runtime_flags_present: Literal[True] = True
    no_uncontrolled_runtime_flags: Literal[True] = True
    no_missing_coverage: Literal[True] = True
    v4_boundaries_preserved: Literal[True] = True
    runtime_evolution_not_applied: Literal[True] = True
    policy_application_implemented: Literal[True] = True
    execution_policy_application_implemented: Literal[True] = True
    strategy_application_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    resource_allocation_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    hitl_emission_implemented: Literal[False] = False
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
    def _registry_matches_records(self) -> Self:
        derived_surface_ids = tuple(record.surface_id for record in self.records)
        if len(set(derived_surface_ids)) != len(derived_surface_ids):
            raise ValueError("surface_ids must be unique")
        if self.surface_ids != derived_surface_ids:
            raise ValueError("surface_ids must match records")
        if self.surface_ids != _surface_ids():
            raise ValueError("surface_ids must match V5.5 surface order")
        if self.record_count != len(self.records):
            raise ValueError("record_count must match records")
        if self.architecture_layers != _ARCHITECTURE_LAYERS:
            raise ValueError("architecture_layers must match V5.5 layer order")
        if self.validated_version_rules != _VALIDATED_VERSION_RULES:
            raise ValueError("validated_version_rules must match registry")
        if self.passive_boundary_flags != _PASSIVE_BOUNDARY_FLAGS:
            raise ValueError("passive_boundary_flags must match registry")
        for record in self.records:
            if record.architecture_stage != self.architecture_stage:
                raise ValueError("architecture_stage must match registry")
            if record.validated_version_rules != self.validated_version_rules:
                raise ValueError("validated_version_rules must match registry")
            if record.passive_boundary_flags != self.passive_boundary_flags:
                raise ValueError("passive_boundary_flags must match registry")
            if record.source_route_name is not None and (
                record.source_route_name != self.route_name
            ):
                raise ValueError("source_route_name must match registry route")
            if record.source_active_runtime_flags and not _active_flags_allowed(record):
                raise ValueError(
                    "records must not contain uncontrolled active runtime flags"
                )
            if record.missing_coverage_items:
                raise ValueError("records must not contain missing coverage")
        return self


def adaptive_execution_architecture_consistency_registry() -> (
    AdaptiveExecutionArchitectureConsistencyRegistry
):
    """Return passive V5.5 adaptive execution architecture consistency metadata."""

    records = tuple(
        _record_from_source(surface_id, layer, source)
        for surface_id, layer, source in _source_specs()
    )
    return AdaptiveExecutionArchitectureConsistencyRegistry(
        records=records,
        surface_ids=tuple(record.surface_id for record in records),
        record_count=len(records),
        architecture_layers=_ARCHITECTURE_LAYERS,
        validated_version_rules=_VALIDATED_VERSION_RULES,
        passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
    )


def adaptive_execution_architecture_consistency_by_surface(
    surface_id: str,
    registry: AdaptiveExecutionArchitectureConsistencyRegistry | None = None,
) -> AdaptiveExecutionArchitectureConsistencyRecord | None:
    """Return one V5.5 architecture consistency record by surface id."""

    source_registry = registry or adaptive_execution_architecture_consistency_registry()
    normalized_surface_id = str(surface_id).strip()
    for record in source_registry.records:
        if record.surface_id == normalized_surface_id:
            return record
    return None


def adaptive_execution_architecture_consistency_records_for_layer(
    architecture_layer: str,
    registry: AdaptiveExecutionArchitectureConsistencyRegistry | None = None,
) -> tuple[AdaptiveExecutionArchitectureConsistencyRecord, ...]:
    """Return passive V5.5 architecture records for one layer."""

    source_registry = registry or adaptive_execution_architecture_consistency_registry()
    normalized_layer = str(architecture_layer).strip()
    return tuple(
        record
        for record in source_registry.records
        if record.architecture_layer == normalized_layer
    )


def _source_specs() -> tuple[tuple[str, AdaptiveExecutionArchitectureLayer, Any], ...]:
    sources = {
        "adaptive_hybrid_workflow_optimizer": optimize_hybrid_workflow(
            route=_ROUTE_NAME
        ),
        "adaptive_escalation_optimizer": optimize_escalation_policy(route=_ROUTE_NAME),
        "agent_activation_optimizer": optimize_agent_activation(route=_ROUTE_NAME),
        "adaptive_cost_quality_optimizer": optimize_adaptive_cost_quality(
            route=_ROUTE_NAME
        ),
        "adaptive_latency_optimizer": optimize_adaptive_latency(route=_ROUTE_NAME),
        "adaptive_execution_strategy_selection": select_dynamic_execution_strategy(
            route=_ROUTE_NAME
        ),
        "adaptive_execution_policy_engine": evaluate_adaptive_execution_policy(
            route=_ROUTE_NAME
        ),
        "dynamic_agent_allocation": allocate_dynamic_agents(route=_ROUTE_NAME),
        "dynamic_resource_allocation": allocate_dynamic_resources(route=_ROUTE_NAME),
        "workflow_self_tuning_policies": plan_workflow_self_tuning_policies(
            route=_ROUTE_NAME
        ),
        "execution_confidence_engine": evaluate_execution_confidence(route=_ROUTE_NAME),
        "workflow_risk_engine": evaluate_workflow_risk(route=_ROUTE_NAME),
        "creative_exploration_optimizer": optimize_creative_exploration(
            route=_ROUTE_NAME
        ),
        "emergence_optimizer": optimize_emergence(route=_ROUTE_NAME),
        "agent_diversity_optimizer": optimize_agent_diversity(route=_ROUTE_NAME),
        "reflection_budget_optimizer": optimize_reflection_budget(),
        "adaptive_policy_explainability": explain_adaptive_policy(route=_ROUTE_NAME),
    }
    return tuple(
        (surface_id, layer, sources[surface_id])
        for surface_id, layer in _SURFACE_LAYERS
    )


def _record_from_source(
    surface_id: str,
    layer: AdaptiveExecutionArchitectureLayer,
    source: Any,
) -> AdaptiveExecutionArchitectureConsistencyRecord:
    source_role = str(getattr(source, "role", surface_id))
    serialization_version = str(getattr(source, "serialization_version", ""))
    count_field, count = _source_count(source)
    blocked = tuple(getattr(source, "blocked_runtime_behaviors", ()))
    active_flags = _active_runtime_flags(source)
    missing = _missing_coverage(source, count_field=count_field, blocked=blocked)
    route_name = getattr(source, "route_name", None)
    return AdaptiveExecutionArchitectureConsistencyRecord(
        surface_id=surface_id,
        architecture_layer=layer,
        source_role=source_role,
        source_serialization_version=serialization_version,
        source_route_name=route_name if isinstance(route_name, RouteName) else None,
        source_count_field=count_field,
        source_count=count,
        validated_version_rules=_VALIDATED_VERSION_RULES,
        passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
        source_blocked_runtime_behaviors=blocked or _BLOCKED_RUNTIME_BEHAVIORS,
        source_active_runtime_flags=active_flags,
        missing_coverage_items=missing,
        source_advisory_only_declared=bool(getattr(source, "advisory_only", False)),
        source_controlled_policy_declared=bool(
            getattr(source, "controlled_policy_only", False)
        ),
        policy_application_implemented=bool(
            getattr(source, "policy_application_implemented", False)
        ),
        execution_policy_application_implemented=bool(
            getattr(source, "execution_policy_application_implemented", False)
        ),
    )


def _source_count(source: Any) -> tuple[str, int]:
    for field_name in _COUNT_FIELD_CANDIDATES:
        value = getattr(source, field_name, None)
        if isinstance(value, int) and value >= 1:
            return field_name, value
    records = getattr(source, "records", None)
    if isinstance(records, tuple) and records:
        return "record_count", len(records)
    raise ValueError("source count field is missing")


def _active_runtime_flags(source: Any) -> tuple[str, ...]:
    return tuple(
        flag_name
        for flag_name in _ACTIVE_RUNTIME_FLAGS
        if getattr(source, flag_name, False) is True
    )


def _missing_coverage(
    source: Any,
    *,
    count_field: str,
    blocked: tuple[str, ...],
) -> tuple[str, ...]:
    missing: list[str] = []
    if not getattr(source, "role", ""):
        missing.append("role_missing")
    if not getattr(source, "serialization_version", ""):
        missing.append("serialization_version_missing")
    if not (
        getattr(source, "advisory_only", False)
        or getattr(source, "controlled_policy_only", False)
    ):
        missing.append("advisory_or_controlled_policy_missing")
    if not count_field:
        missing.append("count_field_missing")
    if not blocked:
        missing.append("blocked_runtime_behaviors_missing")
    return tuple(missing)


def _active_flags_allowed(
    record: AdaptiveExecutionArchitectureConsistencyRecord,
) -> bool:
    return record.surface_id == "adaptive_execution_policy_engine" and set(
        record.source_active_runtime_flags
    ).issubset(set(_ALLOWED_CONTROLLED_ACTIVE_FLAGS))


def _surface_ids() -> tuple[str, ...]:
    return tuple(surface_id for surface_id, _ in _SURFACE_LAYERS)
