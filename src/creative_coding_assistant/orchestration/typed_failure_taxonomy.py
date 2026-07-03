"""V7.2 typed failure taxonomy contracts and registries."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.contracts import StreamEventType
from creative_coding_assistant.orchestration.execution_graph_analyzer import (
    ExecutionGraphAnalysis,
    analyze_assistant_execution_graph,
)
from creative_coding_assistant.orchestration.workflow_graph import (
    ASSISTANT_WORKFLOW_NODE_ORDER,
    assistant_workflow_model_payload_specs,
)
from creative_coding_assistant.orchestration.workflow_review import (
    MAX_WORKFLOW_REFINEMENT_COUNT,
)

TYPED_FAILURE_TAXONOMY_SERIALIZATION_VERSION = "typed_failure_taxonomy.v1"
FAILURE_TYPE_DEFINITION_SERIALIZATION_VERSION = "failure_type_definition.v1"
FAILURE_EVENT_CONTRACT_SERIALIZATION_VERSION = "failure_event_contract.v1"
FAILURE_TAXONOMY_VALIDATION_SERIALIZATION_VERSION = (
    "failure_taxonomy_validation.v1"
)

TYPED_FAILURE_TAXONOMY_ROADMAP_ITEMS = (
    "Failure Type Registry",
    "Node-Specific Failure Models",
    "Planning Sub-helper Failure Models",
    "Provider/Stream Failure Models",
    "Serialization Failure Models",
    "Workstation/Client Boundary Failure Models",
    "Failure Event Contract Stabilization",
    "Failure Recovery Invariants",
    "Failure Regression Suite",
    "Recovery Strategy Catalog",
    "Failure Explainability",
    "Failure Severity Classification",
    "Failure Analytics Contracts",
    "Failure Root Cause Classification",
    "Failure Reproducibility Engine",
    "Failure Ownership Mapping",
    "Failure Fix Recommendation Engine",
    "Failure Knowledge Base",
)
TYPED_FAILURE_BLOCKED_BEHAVIORS = (
    "live_failure_classification",
    "exception_interception",
    "failure_recovery_execution",
    "retry_triggering",
    "provider_model_routing",
    "provider_execution",
    "workflow_execution",
    "workflow_control",
    "workflow_graph_mutation",
    "stream_subscription",
    "persistent_storage_write",
    "runtime_evolution_application",
)

FailureDomain = Literal[
    "workflow_node",
    "planning_sub_helper",
    "provider_stream",
    "serialization",
    "workstation_client",
]
FailureSeverity = Literal[
    "notice",
    "recoverable",
    "degraded",
    "terminal",
    "guardrail",
]
FailureRootCause = Literal[
    "missing_dependency",
    "provider_error",
    "contract_violation",
    "serialization_error",
    "client_boundary_error",
    "workflow_exception",
    "quality_gate_failure",
    "configuration_gap",
]
FailureOwner = Literal[
    "runtime_graph",
    "planning_system",
    "provider_adapter",
    "serialization_boundary",
    "workstation_client",
    "quality_review",
    "observability",
]
RecoveryStrategyKind = Literal[
    "safe_terminal_answer",
    "retry_budget_review",
    "fallback_to_shell_answer",
    "request_clarification",
    "preserve_partial_artifacts",
    "surface_contract_error",
    "manual_follow_up",
]
FailureReproducibilityMode = Literal[
    "static_contract",
    "fixture_event",
    "synthetic_payload",
]
FailureRegressionScenarioKind = Literal[
    "unit_model_validation",
    "contract_lookup",
    "event_payload_shape",
    "boundary_invariant",
]

_DOMAIN_ORDER: tuple[FailureDomain, ...] = (
    "workflow_node",
    "planning_sub_helper",
    "provider_stream",
    "serialization",
    "workstation_client",
)
_SEVERITY_ORDER: tuple[FailureSeverity, ...] = (
    "notice",
    "recoverable",
    "degraded",
    "terminal",
    "guardrail",
)


class FailureTypeDefinition(BaseModel):
    """Stable definition for one V7.2 typed failure."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    failure_type_id: str = Field(min_length=1, max_length=180)
    failure_code: str = Field(min_length=1, max_length=120)
    domain: FailureDomain
    severity: FailureSeverity
    root_cause: FailureRootCause
    owner: FailureOwner
    recovery_strategy_id: str = Field(min_length=1, max_length=180)
    regression_scenario_id: str = Field(min_length=1, max_length=180)
    knowledge_base_entry_id: str = Field(min_length=1, max_length=180)
    fix_recommendation_id: str = Field(min_length=1, max_length=180)
    source_surface: str = Field(min_length=1, max_length=180)
    workflow_node_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=18)
    planning_helper_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=80)
    stream_event_types: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    serialization_surfaces: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    client_boundary_surfaces: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    event_codes: tuple[str, ...] = Field(min_length=1, max_length=12)
    analytics_dimensions: tuple[str, ...] = Field(min_length=1, max_length=12)
    reproducibility_mode: FailureReproducibilityMode
    explanation: str = Field(min_length=1, max_length=700)
    user_visible: bool
    retry_eligible: bool
    terminal: bool
    live_failure_classification_implemented: Literal[False] = False
    exception_interception_implemented: Literal[False] = False
    failure_recovery_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["failure_type_definition.v1"] = (
        FAILURE_TYPE_DEFINITION_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _definition_matches_identity(self) -> Self:
        expected_id = f"typed_failure::{self.domain}::{self.failure_code}"
        if self.failure_type_id != expected_id:
            raise ValueError("failure_type_id must match domain and code")
        if self.terminal != (self.severity == "terminal"):
            raise ValueError("terminal must match terminal severity")
        if self.terminal and self.retry_eligible:
            raise ValueError("terminal failure types cannot be retry eligible")
        if self.domain == "workflow_node" and not self.workflow_node_ids:
            raise ValueError("workflow node failures require workflow_node_ids")
        unknown_nodes = set(self.workflow_node_ids).difference(
            ASSISTANT_WORKFLOW_NODE_ORDER
        )
        if unknown_nodes:
            raise ValueError("workflow_node_ids must reference known nodes")
        if self.domain == "planning_sub_helper" and not self.planning_helper_ids:
            raise ValueError("planning failures require planning_helper_ids")
        known_helpers = set(_planning_helper_ids())
        unknown_helpers = set(self.planning_helper_ids).difference(known_helpers)
        if unknown_helpers:
            raise ValueError("planning_helper_ids must reference known helpers")
        if self.domain == "provider_stream" and not self.stream_event_types:
            raise ValueError("provider stream failures require stream_event_types")
        known_event_types = {event_type.value for event_type in StreamEventType}
        unknown_stream_events = set(self.stream_event_types).difference(
            known_event_types
        )
        if unknown_stream_events:
            raise ValueError("stream_event_types must reference known event types")
        if self.domain == "serialization" and not self.serialization_surfaces:
            raise ValueError("serialization failures require serialization_surfaces")
        if (
            self.domain == "workstation_client"
            and not self.client_boundary_surfaces
        ):
            raise ValueError(
                "workstation client failures require client_boundary_surfaces"
            )
        return self


class NodeSpecificFailureModel(BaseModel):
    """Typed failure coverage for one assistant workflow node."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    node_model_id: str = Field(min_length=1, max_length=180)
    node_id: str = Field(min_length=1, max_length=120)
    order_index: int = Field(ge=0)
    supported_failure_type_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    emitted_event_contract_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    workflow_failure_code_prefix: str = Field(min_length=1, max_length=120)
    state_input_keys: tuple[str, ...] = Field(default_factory=tuple, max_length=80)
    state_output_keys: tuple[str, ...] = Field(default_factory=tuple, max_length=80)
    failure_transition_target: Literal["failure"] = "failure"
    node_handler_invocation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _node_model_matches_workflow_order(self) -> Self:
        if self.node_model_id != f"node_failure_model::{self.node_id}":
            raise ValueError("node_model_id must match node_id")
        if self.order_index >= len(ASSISTANT_WORKFLOW_NODE_ORDER):
            raise ValueError("order_index must fit workflow order")
        if ASSISTANT_WORKFLOW_NODE_ORDER[self.order_index] != self.node_id:
            raise ValueError("node_id must match workflow order")
        if self.workflow_failure_code_prefix != f"workflow_{self.node_id}_":
            raise ValueError("workflow_failure_code_prefix must match node")
        return self


class PlanningSubHelperFailureModel(BaseModel):
    """Typed failure coverage for one planning metadata helper."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    helper_model_id: str = Field(min_length=1, max_length=180)
    helper_id: str = Field(min_length=1, max_length=120)
    payload_key: str = Field(min_length=1, max_length=120)
    availability_key: str | None = Field(default=None, max_length=120)
    supported_failure_type_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    upstream_node_id: Literal["planning"] = "planning"
    downstream_node_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    partial_planning_allowed: bool = True
    helper_invocation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _helper_model_matches_payload(self) -> Self:
        if self.helper_model_id != (
            f"planning_sub_helper_failure_model::{self.helper_id}"
        ):
            raise ValueError("helper_model_id must match helper_id")
        if self.helper_id != self.payload_key:
            raise ValueError("helper_id must match payload_key")
        return self


class ProviderStreamFailureModel(BaseModel):
    """Typed provider and stream boundary failure coverage."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    provider_stream_model_id: str = Field(min_length=1, max_length=180)
    surface_id: str = Field(min_length=1, max_length=120)
    stream_event_types: tuple[str, ...] = Field(min_length=1, max_length=12)
    supported_failure_type_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    fallback_strategy_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    provider_execution_implemented: Literal[False] = False
    stream_subscription_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _provider_stream_id_matches_surface(self) -> Self:
        if self.provider_stream_model_id != (
            f"provider_stream_failure_model::{self.surface_id}"
        ):
            raise ValueError("provider_stream_model_id must match surface_id")
        return self


class SerializationFailureModel(BaseModel):
    """Typed serialization boundary failure coverage."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    serialization_model_id: str = Field(min_length=1, max_length=180)
    surface_id: str = Field(min_length=1, max_length=120)
    serialized_contracts: tuple[str, ...] = Field(min_length=1, max_length=12)
    supported_failure_type_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    stable_payload_keys: tuple[str, ...] = Field(min_length=1, max_length=16)
    serialization_execution_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _serialization_id_matches_surface(self) -> Self:
        if self.serialization_model_id != (
            f"serialization_failure_model::{self.surface_id}"
        ):
            raise ValueError("serialization_model_id must match surface_id")
        return self


class WorkstationClientBoundaryFailureModel(BaseModel):
    """Typed workstation and client boundary failure coverage."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    boundary_model_id: str = Field(min_length=1, max_length=180)
    surface_id: str = Field(min_length=1, max_length=120)
    client_surface: str = Field(min_length=1, max_length=120)
    supported_failure_type_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    stable_payload_keys: tuple[str, ...] = Field(min_length=1, max_length=16)
    user_visible_recovery_id: str = Field(min_length=1, max_length=180)
    client_code_execution_implemented: Literal[False] = False
    server_request_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _boundary_id_matches_surface(self) -> Self:
        if self.boundary_model_id != (
            f"workstation_client_failure_model::{self.surface_id}"
        ):
            raise ValueError("boundary_model_id must match surface_id")
        return self


class FailureEventContract(BaseModel):
    """Stable event payload contract for typed failure surfaces."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    event_contract_id: str = Field(min_length=1, max_length=180)
    event_type: StreamEventType
    required_payload_keys: tuple[str, ...] = Field(min_length=1, max_length=16)
    optional_payload_keys: tuple[str, ...] = Field(default_factory=tuple, max_length=16)
    mapped_failure_type_ids: tuple[str, ...] = Field(min_length=1, max_length=20)
    stabilization_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    telemetry_emission_implemented: Literal[False] = False
    stream_subscription_implemented: Literal[False] = False
    serialization_version: Literal["failure_event_contract.v1"] = (
        FAILURE_EVENT_CONTRACT_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _event_contract_id_matches_type(self) -> Self:
        if self.event_contract_id != f"failure_event::{self.event_type.value}":
            raise ValueError("event_contract_id must match event_type")
        return self


class FailureRecoveryInvariant(BaseModel):
    """Invariant that must hold before future failure recovery is executable."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    invariant_id: str = Field(min_length=1, max_length=180)
    invariant_kind: str = Field(min_length=1, max_length=120)
    required_for_severities: tuple[FailureSeverity, ...] = Field(
        min_length=1,
        max_length=5,
    )
    enforced_failure_type_ids: tuple[str, ...] = Field(min_length=1, max_length=20)
    description: str = Field(min_length=1, max_length=600)
    recovery_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _invariant_id_matches_kind(self) -> Self:
        if self.invariant_id != f"failure_recovery_invariant::{self.invariant_kind}":
            raise ValueError("invariant_id must match invariant_kind")
        return self


class RecoveryStrategy(BaseModel):
    """Catalog entry for one bounded recovery strategy."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    strategy_id: str = Field(min_length=1, max_length=180)
    strategy_kind: RecoveryStrategyKind
    terminal: bool
    retry_allowed: bool
    max_retry_attempts: int = Field(ge=0, le=MAX_WORKFLOW_REFINEMENT_COUNT)
    fallback_response_required: bool
    invariant_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    description: str = Field(min_length=1, max_length=600)
    recovery_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _strategy_matches_kind(self) -> Self:
        if self.strategy_id != f"recovery_strategy::{self.strategy_kind}":
            raise ValueError("strategy_id must match strategy_kind")
        if self.terminal and self.retry_allowed:
            raise ValueError("terminal recovery strategies cannot allow retry")
        if not self.retry_allowed and self.max_retry_attempts != 0:
            raise ValueError("non-retry recovery strategies require zero attempts")
        return self


class FailureRegressionScenario(BaseModel):
    """Static regression scenario for one failure type."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    scenario_id: str = Field(min_length=1, max_length=180)
    scenario_kind: FailureRegressionScenarioKind
    failure_type_id: str = Field(min_length=1, max_length=180)
    fixture_id: str = Field(min_length=1, max_length=180)
    expected_event_contract_id: str = Field(min_length=1, max_length=180)
    expected_recovery_strategy_id: str = Field(min_length=1, max_length=180)
    expected_owner: FailureOwner
    reproducibility_key: str = Field(min_length=1, max_length=180)
    workflow_execution_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    stream_subscription_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _scenario_id_matches_failure_type(self) -> Self:
        if self.scenario_id != f"failure_regression::{self.failure_type_id}":
            raise ValueError("scenario_id must match failure_type_id")
        return self


class FailureAnalyticsContract(BaseModel):
    """Analytics dimensions for failure grouping without telemetry emission."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    analytics_contract_id: str = Field(min_length=1, max_length=180)
    domain: FailureDomain
    grouped_failure_type_ids: tuple[str, ...] = Field(min_length=1, max_length=20)
    dimensions: tuple[str, ...] = Field(min_length=1, max_length=12)
    metric_names: tuple[str, ...] = Field(min_length=1, max_length=12)
    severity_buckets: tuple[FailureSeverity, ...] = Field(min_length=1, max_length=5)
    telemetry_emission_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _analytics_id_matches_domain(self) -> Self:
        if self.analytics_contract_id != f"failure_analytics::{self.domain}":
            raise ValueError("analytics_contract_id must match domain")
        return self


class FailureRootCauseClassification(BaseModel):
    """Root-cause grouping for typed failure definitions."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    root_cause_id: str = Field(min_length=1, max_length=180)
    root_cause: FailureRootCause
    failure_type_ids: tuple[str, ...] = Field(min_length=1, max_length=20)
    evidence_keys: tuple[str, ...] = Field(min_length=1, max_length=12)
    default_owner: FailureOwner
    live_classification_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _root_cause_id_matches_root_cause(self) -> Self:
        if self.root_cause_id != f"failure_root_cause::{self.root_cause}":
            raise ValueError("root_cause_id must match root_cause")
        return self


class FailureReproducibilityRecord(BaseModel):
    """Deterministic repro metadata for one failure type."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    reproducibility_id: str = Field(min_length=1, max_length=180)
    failure_type_id: str = Field(min_length=1, max_length=180)
    reproducibility_mode: FailureReproducibilityMode
    fixture_key: str = Field(min_length=1, max_length=180)
    deterministic_fields: tuple[str, ...] = Field(min_length=1, max_length=16)
    reproduction_steps: tuple[str, ...] = Field(min_length=1, max_length=8)
    workflow_execution_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _repro_id_matches_failure_type(self) -> Self:
        if self.reproducibility_id != (
            f"failure_reproducibility::{self.failure_type_id}"
        ):
            raise ValueError("reproducibility_id must match failure_type_id")
        return self


class FailureOwnershipRecord(BaseModel):
    """Ownership mapping for typed failure classes."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    ownership_id: str = Field(min_length=1, max_length=180)
    owner: FailureOwner
    failure_type_ids: tuple[str, ...] = Field(min_length=1, max_length=20)
    owning_module: str = Field(min_length=1, max_length=180)
    escalation_policy: str = Field(min_length=1, max_length=600)
    hitl_request_emitted: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _ownership_id_matches_owner(self) -> Self:
        if self.ownership_id != f"failure_ownership::{self.owner}":
            raise ValueError("ownership_id must match owner")
        return self


class FailureFixRecommendation(BaseModel):
    """Fix recommendation metadata for one typed failure."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    recommendation_id: str = Field(min_length=1, max_length=180)
    failure_type_id: str = Field(min_length=1, max_length=180)
    owner: FailureOwner
    recommended_action: str = Field(min_length=1, max_length=700)
    validation_surface: tuple[str, ...] = Field(min_length=1, max_length=8)
    fix_requires_hitl_acceptance: bool
    fix_applied: Literal[False] = False
    workflow_mutation_implemented: Literal[False] = False
    storage_write_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _recommendation_id_matches_failure_type(self) -> Self:
        if self.recommendation_id != f"failure_fix::{self.failure_type_id}":
            raise ValueError("recommendation_id must match failure_type_id")
        return self


class FailureKnowledgeBaseEntry(BaseModel):
    """Knowledge-base entry for one typed failure definition."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    kb_entry_id: str = Field(min_length=1, max_length=180)
    failure_type_id: str = Field(min_length=1, max_length=180)
    title: str = Field(min_length=1, max_length=180)
    symptoms: tuple[str, ...] = Field(min_length=1, max_length=8)
    diagnosis: str = Field(min_length=1, max_length=700)
    recovery_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    linked_fix_recommendation_id: str = Field(min_length=1, max_length=180)
    persistent_storage_write_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _kb_id_matches_failure_type(self) -> Self:
        if self.kb_entry_id != f"failure_kb::{self.failure_type_id}":
            raise ValueError("kb_entry_id must match failure_type_id")
        return self


class TypedFailureTaxonomyRegistry(BaseModel):
    """Read-only V7.2 typed failure taxonomy registry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["typed_failure_taxonomy"] = "typed_failure_taxonomy"
    serialization_version: Literal["typed_failure_taxonomy.v1"] = (
        TYPED_FAILURE_TAXONOMY_SERIALIZATION_VERSION
    )
    source_graph: Literal["assistant_workflow_graph"] = "assistant_workflow_graph"
    source_graph_serialization_version: str = Field(min_length=1, max_length=120)
    source_workflow_node_order: tuple[str, ...] = Field(min_length=1, max_length=40)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=18, max_length=18)
    failure_types: tuple[FailureTypeDefinition, ...] = Field(
        min_length=1,
        max_length=40,
    )
    failure_type_ids: tuple[str, ...] = Field(min_length=1, max_length=40)
    node_failure_models: tuple[NodeSpecificFailureModel, ...] = Field(
        min_length=1,
        max_length=40,
    )
    planning_sub_helper_models: tuple[PlanningSubHelperFailureModel, ...] = Field(
        min_length=1,
        max_length=80,
    )
    provider_stream_models: tuple[ProviderStreamFailureModel, ...] = Field(
        min_length=1,
        max_length=12,
    )
    serialization_models: tuple[SerializationFailureModel, ...] = Field(
        min_length=1,
        max_length=12,
    )
    workstation_client_boundary_models: tuple[
        WorkstationClientBoundaryFailureModel,
        ...,
    ] = Field(min_length=1, max_length=12)
    event_contracts: tuple[FailureEventContract, ...] = Field(
        min_length=1,
        max_length=12,
    )
    recovery_invariants: tuple[FailureRecoveryInvariant, ...] = Field(
        min_length=1,
        max_length=12,
    )
    regression_scenarios: tuple[FailureRegressionScenario, ...] = Field(
        min_length=1,
        max_length=40,
    )
    recovery_strategies: tuple[RecoveryStrategy, ...] = Field(
        min_length=1,
        max_length=12,
    )
    analytics_contracts: tuple[FailureAnalyticsContract, ...] = Field(
        min_length=1,
        max_length=8,
    )
    root_cause_classifications: tuple[
        FailureRootCauseClassification,
        ...,
    ] = Field(min_length=1, max_length=12)
    reproducibility_records: tuple[FailureReproducibilityRecord, ...] = Field(
        min_length=1,
        max_length=40,
    )
    ownership_records: tuple[FailureOwnershipRecord, ...] = Field(
        min_length=1,
        max_length=8,
    )
    fix_recommendations: tuple[FailureFixRecommendation, ...] = Field(
        min_length=1,
        max_length=40,
    )
    knowledge_base_entries: tuple[FailureKnowledgeBaseEntry, ...] = Field(
        min_length=1,
        max_length=40,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    failure_type_count: int = Field(ge=1)
    roadmap_item_count: int = Field(ge=18, le=18)
    node_failure_model_count: int = Field(ge=1)
    planning_sub_helper_model_count: int = Field(ge=1)
    event_contract_count: int = Field(ge=1)
    regression_scenario_count: int = Field(ge=1)
    knowledge_base_entry_count: int = Field(ge=1)
    live_failure_classification_implemented: Literal[False] = False
    exception_interception_implemented: Literal[False] = False
    failure_recovery_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_links_are_complete(self) -> Self:
        if self.covered_roadmap_items != TYPED_FAILURE_TAXONOMY_ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V7.2 roadmap")
        if self.source_workflow_node_order != ASSISTANT_WORKFLOW_NODE_ORDER:
            raise ValueError("source_workflow_node_order must match workflow graph")
        if self.blocked_runtime_behaviors != TYPED_FAILURE_BLOCKED_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V7.2 boundary")
        if self.failure_type_ids != tuple(
            failure.failure_type_id for failure in self.failure_types
        ):
            raise ValueError("failure_type_ids must match failure_types")
        if len(set(self.failure_type_ids)) != len(self.failure_type_ids):
            raise ValueError("failure_type_ids must be unique")
        if self.failure_type_count != len(self.failure_types):
            raise ValueError("failure_type_count must match failure_types")
        if self.roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("roadmap_item_count must match covered roadmap")
        if self.node_failure_model_count != len(self.node_failure_models):
            raise ValueError("node_failure_model_count must match node models")
        if (
            self.planning_sub_helper_model_count
            != len(self.planning_sub_helper_models)
        ):
            raise ValueError("planning_sub_helper_model_count must match helpers")
        if self.event_contract_count != len(self.event_contracts):
            raise ValueError("event_contract_count must match event contracts")
        if self.regression_scenario_count != len(self.regression_scenarios):
            raise ValueError("regression_scenario_count must match scenarios")
        if self.knowledge_base_entry_count != len(self.knowledge_base_entries):
            raise ValueError("knowledge_base_entry_count must match entries")
        self._validate_registry_links()
        return self

    def _validate_registry_links(self) -> None:
        failure_ids = set(self.failure_type_ids)
        failure_by_id = {
            failure.failure_type_id: failure for failure in self.failure_types
        }
        strategy_ids = {strategy.strategy_id for strategy in self.recovery_strategies}
        scenario_ids = {scenario.scenario_id for scenario in self.regression_scenarios}
        kb_ids = {entry.kb_entry_id for entry in self.knowledge_base_entries}
        fix_ids = {
            recommendation.recommendation_id
            for recommendation in self.fix_recommendations
        }
        event_contract_ids = {
            contract.event_contract_id for contract in self.event_contracts
        }
        invariant_ids = {
            invariant.invariant_id for invariant in self.recovery_invariants
        }
        self._validate_unique_registry_ids()
        node_ids = tuple(model.node_id for model in self.node_failure_models)
        helper_ids = tuple(
            model.helper_id for model in self.planning_sub_helper_models
        )

        if node_ids != ASSISTANT_WORKFLOW_NODE_ORDER:
            raise ValueError("node_failure_models must follow workflow order")
        expected_helper_ids = tuple(
            spec.payload_key for spec in assistant_workflow_model_payload_specs()
        )
        if helper_ids != expected_helper_ids:
            raise ValueError("planning_sub_helper_models must match payload specs")

        for failure in self.failure_types:
            if failure.recovery_strategy_id not in strategy_ids:
                raise ValueError("failure recovery_strategy_id must be known")
            if failure.regression_scenario_id not in scenario_ids:
                raise ValueError("failure regression_scenario_id must be known")
            if failure.knowledge_base_entry_id not in kb_ids:
                raise ValueError("failure knowledge_base_entry_id must be known")
            if failure.fix_recommendation_id not in fix_ids:
                raise ValueError("failure fix_recommendation_id must be known")

        for contract in self.event_contracts:
            unknown = set(contract.mapped_failure_type_ids).difference(failure_ids)
            if unknown:
                raise ValueError("event mapped_failure_type_ids must be known")
        for invariant in self.recovery_invariants:
            unknown = set(invariant.enforced_failure_type_ids).difference(failure_ids)
            if unknown:
                raise ValueError("invariant enforced_failure_type_ids must be known")
        for strategy in self.recovery_strategies:
            unknown_invariants = set(strategy.invariant_ids).difference(
                invariant_ids
            )
            if unknown_invariants:
                raise ValueError("strategy invariant ids must be known")

        for collection in (
            self.node_failure_models,
            self.planning_sub_helper_models,
            self.provider_stream_models,
            self.serialization_models,
            self.workstation_client_boundary_models,
        ):
            for model in collection:
                unknown = set(model.supported_failure_type_ids).difference(
                    failure_ids
                )
                if unknown:
                    raise ValueError("model supported_failure_type_ids must be known")

        for model in self.node_failure_models:
            unknown_contracts = set(model.emitted_event_contract_ids).difference(
                event_contract_ids
            )
            if unknown_contracts:
                raise ValueError("node event contract ids must be known")
        for model in self.provider_stream_models:
            unknown = set(model.fallback_strategy_ids).difference(strategy_ids)
            if unknown:
                raise ValueError("provider fallback_strategy_ids must be known")
        for model in self.workstation_client_boundary_models:
            if model.user_visible_recovery_id not in strategy_ids:
                raise ValueError("client user_visible_recovery_id must be known")
        for scenario in self.regression_scenarios:
            if scenario.failure_type_id not in failure_ids:
                raise ValueError("scenario failure_type_id must be known")
            if scenario.expected_event_contract_id not in event_contract_ids:
                raise ValueError("scenario expected_event_contract_id must be known")
            if scenario.expected_recovery_strategy_id not in strategy_ids:
                raise ValueError("scenario expected_recovery_strategy_id must be known")
        for contract in self.analytics_contracts:
            unknown = set(contract.grouped_failure_type_ids).difference(failure_ids)
            if unknown:
                raise ValueError("analytics grouped_failure_type_ids must be known")
            for failure_id in contract.grouped_failure_type_ids:
                failure = failure_by_id[failure_id]
                if failure.domain != contract.domain:
                    raise ValueError("analytics domain must match grouped failures")
        for classification in self.root_cause_classifications:
            unknown = set(classification.failure_type_ids).difference(failure_ids)
            if unknown:
                raise ValueError("root cause failure_type_ids must be known")
            for failure_id in classification.failure_type_ids:
                failure = failure_by_id[failure_id]
                if failure.root_cause != classification.root_cause:
                    raise ValueError("root cause must match grouped failures")
        for record in self.reproducibility_records:
            if record.failure_type_id not in failure_ids:
                raise ValueError("reproducibility failure_type_id must be known")
        for ownership in self.ownership_records:
            unknown = set(ownership.failure_type_ids).difference(failure_ids)
            if unknown:
                raise ValueError("ownership failure_type_ids must be known")
            for failure_id in ownership.failure_type_ids:
                failure = failure_by_id[failure_id]
                if failure.owner != ownership.owner:
                    raise ValueError("ownership owner must match grouped failures")
        for recommendation in self.fix_recommendations:
            if recommendation.failure_type_id not in failure_ids:
                raise ValueError("fix failure_type_id must be known")
            failure = failure_by_id[recommendation.failure_type_id]
            if recommendation.owner != failure.owner:
                raise ValueError("fix owner must match failure owner")
        for entry in self.knowledge_base_entries:
            if entry.failure_type_id not in failure_ids:
                raise ValueError("knowledge failure_type_id must be known")
            if entry.linked_fix_recommendation_id not in fix_ids:
                raise ValueError(
                    "knowledge linked_fix_recommendation_id must be known"
                )

    def _validate_unique_registry_ids(self) -> None:
        failure_by_id = {
            failure.failure_type_id: failure for failure in self.failure_types
        }
        if len(failure_by_id) != len(self.failure_types):
            raise ValueError("failure_type_ids must be unique")
        _require_unique(
            (strategy.strategy_id for strategy in self.recovery_strategies),
            "recovery strategy ids must be unique",
        )
        _require_unique(
            (contract.event_contract_id for contract in self.event_contracts),
            "event contract ids must be unique",
        )
        _require_unique(
            (scenario.scenario_id for scenario in self.regression_scenarios),
            "regression scenario ids must be unique",
        )


class FailureTaxonomyValidationReport(BaseModel):
    """Validation report for V7.2 typed failure taxonomy completeness."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["failure_taxonomy_validation"] = "failure_taxonomy_validation"
    serialization_version: Literal["failure_taxonomy_validation.v1"] = (
        FAILURE_TAXONOMY_VALIDATION_SERIALIZATION_VERSION
    )
    validation_passed: bool
    checked_failure_type_count: int = Field(ge=0)
    checked_roadmap_item_count: int = Field(ge=0)
    checked_node_failure_model_count: int = Field(ge=0)
    missing_roadmap_items: tuple[str, ...] = Field(default_factory=tuple)
    orphaned_failure_type_ids: tuple[str, ...] = Field(default_factory=tuple)
    orphaned_strategy_ids: tuple[str, ...] = Field(default_factory=tuple)
    orphaned_event_contract_ids: tuple[str, ...] = Field(default_factory=tuple)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    live_failure_classification_implemented: Literal[False] = False
    recovery_execution_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    advisory_only: Literal[True] = True


def build_typed_failure_taxonomy_registry(
    *,
    execution_graph: ExecutionGraphAnalysis | None = None,
) -> TypedFailureTaxonomyRegistry:
    """Build V7.2 typed failure contracts without executing workflow code."""

    analysis = execution_graph or analyze_assistant_execution_graph()
    failure_types = _failure_types()
    event_contracts = _event_contracts(failure_types)
    recovery_invariants = _recovery_invariants(failure_types)
    recovery_strategies = _recovery_strategies(recovery_invariants)
    return TypedFailureTaxonomyRegistry(
        source_graph_serialization_version=analysis.serialization_version,
        source_workflow_node_order=analysis.node_order,
        covered_roadmap_items=TYPED_FAILURE_TAXONOMY_ROADMAP_ITEMS,
        failure_types=failure_types,
        failure_type_ids=tuple(failure.failure_type_id for failure in failure_types),
        node_failure_models=_node_failure_models(failure_types),
        planning_sub_helper_models=_planning_sub_helper_models(failure_types),
        provider_stream_models=_provider_stream_models(failure_types),
        serialization_models=_serialization_models(failure_types),
        workstation_client_boundary_models=_client_boundary_models(failure_types),
        event_contracts=event_contracts,
        recovery_invariants=recovery_invariants,
        regression_scenarios=_regression_scenarios(failure_types),
        recovery_strategies=recovery_strategies,
        analytics_contracts=_analytics_contracts(failure_types),
        root_cause_classifications=_root_cause_classifications(failure_types),
        reproducibility_records=_reproducibility_records(failure_types),
        ownership_records=_ownership_records(failure_types),
        fix_recommendations=_fix_recommendations(failure_types),
        knowledge_base_entries=_knowledge_base_entries(failure_types),
        blocked_runtime_behaviors=TYPED_FAILURE_BLOCKED_BEHAVIORS,
        failure_type_count=len(failure_types),
        roadmap_item_count=len(TYPED_FAILURE_TAXONOMY_ROADMAP_ITEMS),
        node_failure_model_count=len(ASSISTANT_WORKFLOW_NODE_ORDER),
        planning_sub_helper_model_count=len(assistant_workflow_model_payload_specs()),
        event_contract_count=len(event_contracts),
        regression_scenario_count=len(failure_types),
        knowledge_base_entry_count=len(failure_types),
    )


def failure_type_by_id(
    failure_type_id: str,
    registry: TypedFailureTaxonomyRegistry | None = None,
) -> FailureTypeDefinition | None:
    """Return one typed failure definition by stable id."""

    source_registry = registry or build_typed_failure_taxonomy_registry()
    for failure in source_registry.failure_types:
        if failure.failure_type_id == failure_type_id:
            return failure
    return None


def failure_types_for_domain(
    domain: FailureDomain,
    registry: TypedFailureTaxonomyRegistry | None = None,
) -> tuple[FailureTypeDefinition, ...]:
    """Return all failure definitions for one domain."""

    source_registry = registry or build_typed_failure_taxonomy_registry()
    return tuple(
        failure for failure in source_registry.failure_types if failure.domain == domain
    )


def failure_types_for_severity(
    severity: FailureSeverity,
    registry: TypedFailureTaxonomyRegistry | None = None,
) -> tuple[FailureTypeDefinition, ...]:
    """Return all failure definitions for one severity."""

    source_registry = registry or build_typed_failure_taxonomy_registry()
    return tuple(
        failure
        for failure in source_registry.failure_types
        if failure.severity == severity
    )


def failure_types_for_root_cause(
    root_cause: FailureRootCause,
    registry: TypedFailureTaxonomyRegistry | None = None,
) -> tuple[FailureTypeDefinition, ...]:
    """Return all failure definitions for one root cause."""

    source_registry = registry or build_typed_failure_taxonomy_registry()
    return tuple(
        failure
        for failure in source_registry.failure_types
        if failure.root_cause == root_cause
    )


def failure_types_for_owner(
    owner: FailureOwner,
    registry: TypedFailureTaxonomyRegistry | None = None,
) -> tuple[FailureTypeDefinition, ...]:
    """Return all failure definitions owned by one boundary."""

    source_registry = registry or build_typed_failure_taxonomy_registry()
    return tuple(
        failure for failure in source_registry.failure_types if failure.owner == owner
    )


def node_failure_model_by_id(
    node_id: str,
    registry: TypedFailureTaxonomyRegistry | None = None,
) -> NodeSpecificFailureModel | None:
    """Return node-specific failure coverage without invoking the node."""

    source_registry = registry or build_typed_failure_taxonomy_registry()
    for model in source_registry.node_failure_models:
        if model.node_id == node_id:
            return model
    return None


def recovery_strategy_by_id(
    strategy_id: str,
    registry: TypedFailureTaxonomyRegistry | None = None,
) -> RecoveryStrategy | None:
    """Return one recovery strategy catalog entry."""

    source_registry = registry or build_typed_failure_taxonomy_registry()
    for strategy in source_registry.recovery_strategies:
        if strategy.strategy_id == strategy_id:
            return strategy
    return None


def regression_scenario_by_id(
    scenario_id: str,
    registry: TypedFailureTaxonomyRegistry | None = None,
) -> FailureRegressionScenario | None:
    """Return one failure regression scenario."""

    source_registry = registry or build_typed_failure_taxonomy_registry()
    for scenario in source_registry.regression_scenarios:
        if scenario.scenario_id == scenario_id:
            return scenario
    return None


def explain_failure_type(
    failure_type_id: str,
    registry: TypedFailureTaxonomyRegistry | None = None,
) -> str | None:
    """Return the stable explanation for one typed failure."""

    failure = failure_type_by_id(failure_type_id, registry)
    return None if failure is None else failure.explanation


def validate_typed_failure_taxonomy(
    registry: TypedFailureTaxonomyRegistry | None = None,
) -> FailureTaxonomyValidationReport:
    """Validate coverage links for the V7.2 typed failure taxonomy."""

    source_registry = registry or build_typed_failure_taxonomy_registry()
    referenced_failure_ids = _referenced_failure_type_ids(source_registry)
    strategy_ids = {
        strategy.strategy_id for strategy in source_registry.recovery_strategies
    }
    referenced_strategy_ids = {
        failure.recovery_strategy_id for failure in source_registry.failure_types
    }
    event_contract_ids = {
        contract.event_contract_id for contract in source_registry.event_contracts
    }
    referenced_event_contract_ids = {
        model_event_id
        for model in source_registry.node_failure_models
        for model_event_id in model.emitted_event_contract_ids
    }
    missing_roadmap_items = tuple(
        item
        for item in TYPED_FAILURE_TAXONOMY_ROADMAP_ITEMS
        if item not in source_registry.covered_roadmap_items
    )
    orphaned_failure_type_ids = tuple(
        failure_id for failure_id in source_registry.failure_type_ids
        if failure_id not in referenced_failure_ids
    )
    orphaned_strategy_ids = tuple(
        strategy_id for strategy_id in strategy_ids
        if strategy_id not in referenced_strategy_ids
    )
    orphaned_event_contract_ids = tuple(
        event_id for event_id in event_contract_ids
        if event_id not in referenced_event_contract_ids
    )
    failures = (
        missing_roadmap_items
        or orphaned_failure_type_ids
        or orphaned_strategy_ids
        or orphaned_event_contract_ids
    )
    return FailureTaxonomyValidationReport(
        validation_passed=not failures,
        checked_failure_type_count=source_registry.failure_type_count,
        checked_roadmap_item_count=source_registry.roadmap_item_count,
        checked_node_failure_model_count=source_registry.node_failure_model_count,
        missing_roadmap_items=missing_roadmap_items,
        orphaned_failure_type_ids=orphaned_failure_type_ids,
        orphaned_strategy_ids=orphaned_strategy_ids,
        orphaned_event_contract_ids=orphaned_event_contract_ids,
        blocked_runtime_behaviors=TYPED_FAILURE_BLOCKED_BEHAVIORS,
    )


def _require_unique(values: Iterable[str], message: str) -> None:
    materialized = tuple(values)
    if len(set(materialized)) != len(materialized):
        raise ValueError(message)


def _failure_types() -> tuple[FailureTypeDefinition, ...]:
    specs = (
        _failure_spec(
            domain="workflow_node",
            code="node_exception",
            severity="terminal",
            root_cause="workflow_exception",
            owner="runtime_graph",
            strategy="safe_terminal_answer",
            surface="assistant_workflow_node",
            node_ids=ASSISTANT_WORKFLOW_NODE_ORDER,
            event_codes=("node_failed", "workflow_failed"),
            explanation=(
                "A workflow node raised an exception before completing its "
                "state transition, so the failure boundary must preserve the "
                "node id, code, message, and terminal answer contract."
            ),
            user_visible=True,
            retry_eligible=False,
            reproducibility_mode="fixture_event",
        ),
        _failure_spec(
            domain="workflow_node",
            code="review_quality_gate",
            severity="recoverable",
            root_cause="quality_gate_failure",
            owner="quality_review",
            strategy="retry_budget_review",
            surface="workflow_review",
            node_ids=("review", "refinement"),
            event_codes=("review_failed", "refinement_requested"),
            explanation=(
                "A review outcome requested refinement while the bounded retry "
                "budget remains available."
            ),
            user_visible=False,
            retry_eligible=True,
            reproducibility_mode="static_contract",
        ),
        _failure_spec(
            domain="workflow_node",
            code="missing_runtime_dependency",
            severity="degraded",
            root_cause="missing_dependency",
            owner="runtime_graph",
            strategy="fallback_to_shell_answer",
            surface="workflow_runtime_dependency",
            node_ids=("memory", "retrieval", "context_assembly", "generation"),
            event_codes=("context_assembly_unavailable", "shell_answer_completed"),
            explanation=(
                "A non-critical runtime dependency is unavailable and the "
                "assistant can preserve deterministic shell-answer fallback."
            ),
            user_visible=True,
            retry_eligible=False,
            reproducibility_mode="synthetic_payload",
        ),
        _failure_spec(
            domain="workflow_node",
            code="finalization_failure",
            severity="terminal",
            root_cause="workflow_exception",
            owner="runtime_graph",
            strategy="safe_terminal_answer",
            surface="workflow_finalization",
            node_ids=("finalization", "failure"),
            event_codes=("workflow_failed", "final_response"),
            explanation=(
                "Finalization could not produce the normal completed state; "
                "the failure node must emit a terminal answer instead."
            ),
            user_visible=True,
            retry_eligible=False,
            reproducibility_mode="fixture_event",
        ),
        _failure_spec(
            domain="planning_sub_helper",
            code="helper_unavailable",
            severity="recoverable",
            root_cause="missing_dependency",
            owner="planning_system",
            strategy="preserve_partial_artifacts",
            surface="planning_metadata_helper",
            helper_ids=_planning_helper_ids(),
            event_codes=("creative_plan_prepared", "planning_metadata_partial"),
            explanation=(
                "One planning sub-helper cannot provide metadata, but the "
                "planning node may preserve available sibling helper outputs."
            ),
            user_visible=False,
            retry_eligible=False,
            reproducibility_mode="static_contract",
        ),
        _failure_spec(
            domain="planning_sub_helper",
            code="planning_contract_violation",
            severity="degraded",
            root_cause="contract_violation",
            owner="planning_system",
            strategy="surface_contract_error",
            surface="planning_contract_model",
            helper_ids=_planning_helper_ids(),
            event_codes=("creative_plan_prepared", "node_failed"),
            explanation=(
                "Planning metadata failed a typed contract and must be surfaced "
                "as a contract error rather than silently mutating output."
            ),
            user_visible=True,
            retry_eligible=False,
            reproducibility_mode="synthetic_payload",
        ),
        _failure_spec(
            domain="provider_stream",
            code="provider_generation_error",
            severity="terminal",
            root_cause="provider_error",
            owner="provider_adapter",
            strategy="safe_terminal_answer",
            surface="generation_provider",
            stream_event_types=("generation_input", "token_delta", "error"),
            event_codes=("generation_failed", "workflow_generation_failed"),
            explanation=(
                "The generation provider returned an error before a usable "
                "assistant answer was available."
            ),
            user_visible=True,
            retry_eligible=False,
            reproducibility_mode="fixture_event",
        ),
        _failure_spec(
            domain="provider_stream",
            code="stream_interrupted",
            severity="degraded",
            root_cause="provider_error",
            owner="provider_adapter",
            strategy="fallback_to_shell_answer",
            surface="assistant_stream",
            stream_event_types=("token_delta", "error", "final"),
            event_codes=("assistant_stream_failed", "stream_interrupted"),
            explanation=(
                "A stream boundary ended early; downstream clients need a typed "
                "error shape and a deterministic fallback answer policy."
            ),
            user_visible=True,
            retry_eligible=False,
            reproducibility_mode="fixture_event",
        ),
        _failure_spec(
            domain="serialization",
            code="model_dump_failed",
            severity="terminal",
            root_cause="serialization_error",
            owner="serialization_boundary",
            strategy="surface_contract_error",
            surface="pydantic_model_dump",
            serialization_surfaces=("workflow_state", "stream_event", "artifact"),
            event_codes=("serialization_failed", "node_failed"),
            explanation=(
                "A contract object could not be serialized into the stable API "
                "or stream payload shape."
            ),
            user_visible=True,
            retry_eligible=False,
            reproducibility_mode="synthetic_payload",
        ),
        _failure_spec(
            domain="serialization",
            code="event_payload_invalid",
            severity="degraded",
            root_cause="serialization_error",
            owner="serialization_boundary",
            strategy="surface_contract_error",
            surface="stream_event_payload",
            serialization_surfaces=("node_failed", "error", "final"),
            event_codes=("payload_invalid", "error"),
            explanation=(
                "A stream event payload is missing required failure keys and "
                "must remain visible as a contract issue."
            ),
            user_visible=True,
            retry_eligible=False,
            reproducibility_mode="synthetic_payload",
        ),
        _failure_spec(
            domain="workstation_client",
            code="invalid_request_payload",
            severity="terminal",
            root_cause="client_boundary_error",
            owner="workstation_client",
            strategy="request_clarification",
            surface="assistant_request_contract",
            client_surfaces=("nextjs_stream_bridge", "streamlit_client"),
            event_codes=("assistant_request_invalid", "error"),
            explanation=(
                "A client submitted a request that cannot satisfy the assistant "
                "request contract."
            ),
            user_visible=True,
            retry_eligible=False,
            reproducibility_mode="synthetic_payload",
        ),
        _failure_spec(
            domain="workstation_client",
            code="stream_reducer_error",
            severity="degraded",
            root_cause="client_boundary_error",
            owner="workstation_client",
            strategy="manual_follow_up",
            surface="client_stream_reducer",
            client_surfaces=("streamlit_client", "nextjs_stream_bridge"),
            event_codes=("client_reducer_failed", "error"),
            explanation=(
                "A client-side stream reducer cannot apply a valid event and "
                "must preserve the last stable user-visible state."
            ),
            user_visible=True,
            retry_eligible=False,
            reproducibility_mode="fixture_event",
        ),
        _failure_spec(
            domain="workstation_client",
            code="missing_configuration",
            severity="guardrail",
            root_cause="configuration_gap",
            owner="observability",
            strategy="manual_follow_up",
            surface="local_configuration",
            client_surfaces=("dev_server", "provider_settings"),
            event_codes=("configuration_missing", "error"),
            explanation=(
                "A local configuration value needed by the user-facing surface "
                "is missing and should remain a guardrail record."
            ),
            user_visible=True,
            retry_eligible=False,
            reproducibility_mode="static_contract",
        ),
    )
    return tuple(FailureTypeDefinition(**spec) for spec in specs)


def _failure_spec(
    *,
    domain: FailureDomain,
    code: str,
    severity: FailureSeverity,
    root_cause: FailureRootCause,
    owner: FailureOwner,
    strategy: RecoveryStrategyKind,
    surface: str,
    event_codes: tuple[str, ...],
    explanation: str,
    user_visible: bool,
    retry_eligible: bool,
    reproducibility_mode: FailureReproducibilityMode,
    node_ids: tuple[str, ...] = (),
    helper_ids: tuple[str, ...] = (),
    stream_event_types: tuple[str, ...] = (),
    serialization_surfaces: tuple[str, ...] = (),
    client_surfaces: tuple[str, ...] = (),
) -> dict[str, object]:
    failure_type_id = f"typed_failure::{domain}::{code}"
    return {
        "failure_type_id": failure_type_id,
        "failure_code": code,
        "domain": domain,
        "severity": severity,
        "root_cause": root_cause,
        "owner": owner,
        "recovery_strategy_id": f"recovery_strategy::{strategy}",
        "regression_scenario_id": f"failure_regression::{failure_type_id}",
        "knowledge_base_entry_id": f"failure_kb::{failure_type_id}",
        "fix_recommendation_id": f"failure_fix::{failure_type_id}",
        "source_surface": surface,
        "workflow_node_ids": node_ids,
        "planning_helper_ids": helper_ids,
        "stream_event_types": stream_event_types,
        "serialization_surfaces": serialization_surfaces,
        "client_boundary_surfaces": client_surfaces,
        "event_codes": event_codes,
        "analytics_dimensions": (
            "domain",
            "severity",
            "root_cause",
            "owner",
            "source_surface",
        ),
        "reproducibility_mode": reproducibility_mode,
        "explanation": explanation,
        "user_visible": user_visible,
        "retry_eligible": retry_eligible,
        "terminal": severity == "terminal",
    }


def _node_failure_models(
    failure_types: tuple[FailureTypeDefinition, ...],
) -> tuple[NodeSpecificFailureModel, ...]:
    by_code = _failure_ids_by_code(failure_types)
    models: list[NodeSpecificFailureModel] = []
    for index, node_id in enumerate(ASSISTANT_WORKFLOW_NODE_ORDER):
        failure_ids = [by_code["node_exception"]]
        if node_id in {"memory", "retrieval", "context_assembly", "generation"}:
            failure_ids.append(by_code["missing_runtime_dependency"])
        if node_id == "planning":
            failure_ids.extend(
                (
                    by_code["helper_unavailable"],
                    by_code["planning_contract_violation"],
                )
            )
        if node_id == "generation":
            failure_ids.extend(
                (by_code["provider_generation_error"], by_code["stream_interrupted"])
            )
        if node_id in {"prompt_rendering", "finalization", "failure"}:
            failure_ids.append(by_code["event_payload_invalid"])
        if node_id in {"review", "refinement"}:
            failure_ids.append(by_code["review_quality_gate"])
        if node_id in {"finalization", "failure"}:
            failure_ids.append(by_code["finalization_failure"])
        if node_id in {"intake", "prompt_input"}:
            failure_ids.append(by_code["invalid_request_payload"])
        models.append(
            NodeSpecificFailureModel(
                node_model_id=f"node_failure_model::{node_id}",
                node_id=node_id,
                order_index=index,
                supported_failure_type_ids=tuple(dict.fromkeys(failure_ids)),
                emitted_event_contract_ids=_node_event_contract_ids(node_id),
                workflow_failure_code_prefix=f"workflow_{node_id}_",
                state_input_keys=_node_state_keys(node_id, "input"),
                state_output_keys=_node_state_keys(node_id, "output"),
            )
        )
    return tuple(models)


def _planning_sub_helper_models(
    failure_types: tuple[FailureTypeDefinition, ...],
) -> tuple[PlanningSubHelperFailureModel, ...]:
    by_code = _failure_ids_by_code(failure_types)
    models: list[PlanningSubHelperFailureModel] = []
    for spec in assistant_workflow_model_payload_specs():
        failure_ids = [by_code["helper_unavailable"]]
        if spec.payload_key in {
            "artifact_engine_contracts",
            "evaluation_engine_contracts",
            "runtime_compatibility",
            "consistency_validation",
        }:
            failure_ids.append(by_code["planning_contract_violation"])
        models.append(
            PlanningSubHelperFailureModel(
                helper_model_id=(
                    f"planning_sub_helper_failure_model::{spec.payload_key}"
                ),
                helper_id=spec.payload_key,
                payload_key=spec.payload_key,
                availability_key=spec.availability_key,
                supported_failure_type_ids=tuple(failure_ids),
                downstream_node_ids=("director", "reasoning", "prompt_rendering"),
            )
        )
    return tuple(models)


def _provider_stream_models(
    failure_types: tuple[FailureTypeDefinition, ...],
) -> tuple[ProviderStreamFailureModel, ...]:
    by_code = _failure_ids_by_code(failure_types)
    provider_failure_ids = (
        by_code["provider_generation_error"],
        by_code["stream_interrupted"],
    )
    return (
        ProviderStreamFailureModel(
            provider_stream_model_id=(
                "provider_stream_failure_model::openai_responses_stream"
            ),
            surface_id="openai_responses_stream",
            stream_event_types=("generation_input", "token_delta", "error"),
            supported_failure_type_ids=provider_failure_ids,
            fallback_strategy_ids=(
                "recovery_strategy::safe_terminal_answer",
                "recovery_strategy::fallback_to_shell_answer",
            ),
        ),
        ProviderStreamFailureModel(
            provider_stream_model_id=(
                "provider_stream_failure_model::assistant_ndjson_stream"
            ),
            surface_id="assistant_ndjson_stream",
            stream_event_types=("status", "token_delta", "error", "final"),
            supported_failure_type_ids=provider_failure_ids,
            fallback_strategy_ids=(
                "recovery_strategy::safe_terminal_answer",
                "recovery_strategy::fallback_to_shell_answer",
            ),
        ),
        ProviderStreamFailureModel(
            provider_stream_model_id=(
                "provider_stream_failure_model::workflow_generation_node"
            ),
            surface_id="workflow_generation_node",
            stream_event_types=("generation_input", "token_delta", "error"),
            supported_failure_type_ids=(
                by_code["provider_generation_error"],
                by_code["node_exception"],
            ),
            fallback_strategy_ids=("recovery_strategy::safe_terminal_answer",),
        ),
        ProviderStreamFailureModel(
            provider_stream_model_id=(
                "provider_stream_failure_model::client_stream_reducer"
            ),
            surface_id="client_stream_reducer",
            stream_event_types=("token_delta", "error", "final"),
            supported_failure_type_ids=(
                by_code["stream_interrupted"],
                by_code["stream_reducer_error"],
            ),
            fallback_strategy_ids=("recovery_strategy::manual_follow_up",),
        ),
    )


def _serialization_models(
    failure_types: tuple[FailureTypeDefinition, ...],
) -> tuple[SerializationFailureModel, ...]:
    by_code = _failure_ids_by_code(failure_types)
    failure_ids = (by_code["model_dump_failed"], by_code["event_payload_invalid"])
    return (
        _serialization_model("pydantic_model_dump", failure_ids),
        _serialization_model("ndjson_serialization", failure_ids),
        _serialization_model("stream_event_payload", failure_ids),
        _serialization_model("workflow_state_snapshot", failure_ids),
    )


def _serialization_model(
    surface_id: str,
    failure_ids: tuple[str, ...],
) -> SerializationFailureModel:
    return SerializationFailureModel(
        serialization_model_id=f"serialization_failure_model::{surface_id}",
        surface_id=surface_id,
        serialized_contracts=(
            "StreamEvent",
            "WorkflowFailureInfo",
            "AssistantWorkflowState",
        ),
        supported_failure_type_ids=failure_ids,
        stable_payload_keys=("code", "message", "node", "error_code", "answer"),
    )


def _client_boundary_models(
    failure_types: tuple[FailureTypeDefinition, ...],
) -> tuple[WorkstationClientBoundaryFailureModel, ...]:
    by_code = _failure_ids_by_code(failure_types)
    request_failure_ids = (
        by_code["invalid_request_payload"],
        by_code["missing_configuration"],
    )
    stream_failure_ids = (
        by_code["stream_reducer_error"],
        by_code["stream_interrupted"],
        by_code["event_payload_invalid"],
    )
    return (
        _client_boundary_model("nextjs_stream_bridge", "nextjs", stream_failure_ids),
        _client_boundary_model("streamlit_client", "streamlit", stream_failure_ids),
        _client_boundary_model(
            "workspace_session_api",
            "workspace_sessions",
            request_failure_ids,
        ),
        _client_boundary_model(
            "assistant_request_contract",
            "backend_api",
            request_failure_ids,
        ),
    )


def _client_boundary_model(
    surface_id: str,
    client_surface: str,
    failure_ids: tuple[str, ...],
) -> WorkstationClientBoundaryFailureModel:
    return WorkstationClientBoundaryFailureModel(
        boundary_model_id=f"workstation_client_failure_model::{surface_id}",
        surface_id=surface_id,
        client_surface=client_surface,
        supported_failure_type_ids=failure_ids,
        stable_payload_keys=("event_type", "sequence", "payload", "message"),
        user_visible_recovery_id="recovery_strategy::manual_follow_up",
    )


def _event_contracts(
    failure_types: tuple[FailureTypeDefinition, ...],
) -> tuple[FailureEventContract, ...]:
    by_code = _failure_ids_by_code(failure_types)
    all_failure_ids = tuple(failure.failure_type_id for failure in failure_types)
    return (
        FailureEventContract(
            event_contract_id="failure_event::error",
            event_type=StreamEventType.ERROR,
            required_payload_keys=("code", "message"),
            optional_payload_keys=("category", "subsystem", "node", "error_code"),
            mapped_failure_type_ids=all_failure_ids,
            stabilization_notes=(
                "Error events always preserve code and message.",
                "Typed taxonomy metadata remains advisory.",
            ),
        ),
        FailureEventContract(
            event_contract_id="failure_event::node_failed",
            event_type=StreamEventType.NODE_FAILED,
            required_payload_keys=("node", "error_code", "error_message"),
            optional_payload_keys=("transition_target", "decision_reason"),
            mapped_failure_type_ids=(
                by_code["node_exception"],
                by_code["planning_contract_violation"],
                by_code["model_dump_failed"],
            ),
            stabilization_notes=("Node failures preserve node and transition target.",),
        ),
        FailureEventContract(
            event_contract_id="failure_event::review_failed",
            event_type=StreamEventType.REVIEW_FAILED,
            required_payload_keys=("review_outcome", "review_reasons"),
            optional_payload_keys=("transition_target", "decision_reason"),
            mapped_failure_type_ids=(by_code["review_quality_gate"],),
            stabilization_notes=("Review failure remains bounded by retry budget.",),
        ),
        FailureEventContract(
            event_contract_id="failure_event::retry_completed",
            event_type=StreamEventType.RETRY_COMPLETED,
            required_payload_keys=("retry_status", "retry_count"),
            optional_payload_keys=("retry_reason", "transition_target"),
            mapped_failure_type_ids=(by_code["review_quality_gate"],),
            stabilization_notes=("Retry completion names exhausted versus passed.",),
        ),
        FailureEventContract(
            event_contract_id="failure_event::final",
            event_type=StreamEventType.FINAL,
            required_payload_keys=("answer",),
            optional_payload_keys=("route", "observability"),
            mapped_failure_type_ids=(
                by_code["finalization_failure"],
                by_code["provider_generation_error"],
                by_code["node_exception"],
            ),
            stabilization_notes=("Terminal failures still emit final answer text.",),
        ),
        FailureEventContract(
            event_contract_id="failure_event::status",
            event_type=StreamEventType.STATUS,
            required_payload_keys=("code", "message"),
            optional_payload_keys=("phase", "node", "details"),
            mapped_failure_type_ids=(
                by_code["missing_runtime_dependency"],
                by_code["missing_configuration"],
            ),
            stabilization_notes=("Status events surface degraded non-terminal paths.",),
        ),
    )


def _recovery_invariants(
    failure_types: tuple[FailureTypeDefinition, ...],
) -> tuple[FailureRecoveryInvariant, ...]:
    by_code = _failure_ids_by_code(failure_types)
    all_failure_ids = tuple(failure.failure_type_id for failure in failure_types)
    return (
        FailureRecoveryInvariant(
            invariant_id="failure_recovery_invariant::terminal_safe_answer",
            invariant_kind="terminal_safe_answer",
            required_for_severities=("terminal",),
            enforced_failure_type_ids=tuple(
                failure.failure_type_id
                for failure in failure_types
                if failure.severity == "terminal"
            ),
            description=(
                "Terminal failures must preserve a safe final answer and avoid "
                "hidden retries or workflow mutation."
            ),
        ),
        FailureRecoveryInvariant(
            invariant_id="failure_recovery_invariant::retry_budget_bound",
            invariant_kind="retry_budget_bound",
            required_for_severities=("recoverable",),
            enforced_failure_type_ids=(by_code["review_quality_gate"],),
            description=(
                "Recoverable review failures must not exceed "
                f"{MAX_WORKFLOW_REFINEMENT_COUNT} refinement attempts."
            ),
        ),
        FailureRecoveryInvariant(
            invariant_id="failure_recovery_invariant::provider_routing_stable",
            invariant_kind="provider_routing_stable",
            required_for_severities=("degraded", "terminal"),
            enforced_failure_type_ids=all_failure_ids,
            description=(
                "Typed failure metadata cannot change provider or model routing."
            ),
        ),
        FailureRecoveryInvariant(
            invariant_id="failure_recovery_invariant::event_payload_stable",
            invariant_kind="event_payload_stable",
            required_for_severities=_SEVERITY_ORDER,
            enforced_failure_type_ids=all_failure_ids,
            description=(
                "Failure events must retain code, message, node, and transition "
                "payload keys where applicable."
            ),
        ),
        FailureRecoveryInvariant(
            invariant_id="failure_recovery_invariant::client_boundary_visible",
            invariant_kind="client_boundary_visible",
            required_for_severities=("degraded", "guardrail", "terminal"),
            enforced_failure_type_ids=(
                by_code["invalid_request_payload"],
                by_code["stream_reducer_error"],
                by_code["missing_configuration"],
            ),
            description=(
                "Client boundary failures stay user visible and scoped to the "
                "request or stream event that caused them."
            ),
        ),
    )


def _recovery_strategies(
    invariants: tuple[FailureRecoveryInvariant, ...],
) -> tuple[RecoveryStrategy, ...]:
    invariant_ids = tuple(invariant.invariant_id for invariant in invariants)
    terminal_ids = (
        "failure_recovery_invariant::terminal_safe_answer",
        "failure_recovery_invariant::event_payload_stable",
    )
    stable_ids = (
        "failure_recovery_invariant::provider_routing_stable",
        "failure_recovery_invariant::event_payload_stable",
    )
    return (
        RecoveryStrategy(
            strategy_id="recovery_strategy::safe_terminal_answer",
            strategy_kind="safe_terminal_answer",
            terminal=True,
            retry_allowed=False,
            max_retry_attempts=0,
            fallback_response_required=True,
            invariant_ids=terminal_ids,
            description="Emit a deterministic terminal answer with code and message.",
        ),
        RecoveryStrategy(
            strategy_id="recovery_strategy::retry_budget_review",
            strategy_kind="retry_budget_review",
            terminal=False,
            retry_allowed=True,
            max_retry_attempts=MAX_WORKFLOW_REFINEMENT_COUNT,
            fallback_response_required=False,
            invariant_ids=(
                "failure_recovery_invariant::retry_budget_bound",
                "failure_recovery_invariant::event_payload_stable",
            ),
            description="Allow review refinement only within the bounded retry budget.",
        ),
        RecoveryStrategy(
            strategy_id="recovery_strategy::fallback_to_shell_answer",
            strategy_kind="fallback_to_shell_answer",
            terminal=False,
            retry_allowed=False,
            max_retry_attempts=0,
            fallback_response_required=True,
            invariant_ids=stable_ids,
            description="Use the existing deterministic shell answer fallback.",
        ),
        RecoveryStrategy(
            strategy_id="recovery_strategy::request_clarification",
            strategy_kind="request_clarification",
            terminal=True,
            retry_allowed=False,
            max_retry_attempts=0,
            fallback_response_required=True,
            invariant_ids=(
                "failure_recovery_invariant::client_boundary_visible",
                "failure_recovery_invariant::event_payload_stable",
            ),
            description="Surface a request-scoped correction path to the client.",
        ),
        RecoveryStrategy(
            strategy_id="recovery_strategy::preserve_partial_artifacts",
            strategy_kind="preserve_partial_artifacts",
            terminal=False,
            retry_allowed=False,
            max_retry_attempts=0,
            fallback_response_required=False,
            invariant_ids=stable_ids,
            description="Keep available planning metadata without mutating output.",
        ),
        RecoveryStrategy(
            strategy_id="recovery_strategy::surface_contract_error",
            strategy_kind="surface_contract_error",
            terminal=False,
            retry_allowed=False,
            max_retry_attempts=0,
            fallback_response_required=True,
            invariant_ids=stable_ids,
            description="Expose typed contract failure evidence for review.",
        ),
        RecoveryStrategy(
            strategy_id="recovery_strategy::manual_follow_up",
            strategy_kind="manual_follow_up",
            terminal=False,
            retry_allowed=False,
            max_retry_attempts=0,
            fallback_response_required=True,
            invariant_ids=invariant_ids,
            description="Record advisory remediation for human-controlled follow-up.",
        ),
    )


def _regression_scenarios(
    failure_types: tuple[FailureTypeDefinition, ...],
) -> tuple[FailureRegressionScenario, ...]:
    return tuple(
        FailureRegressionScenario(
            scenario_id=f"failure_regression::{failure.failure_type_id}",
            scenario_kind=_scenario_kind(failure),
            failure_type_id=failure.failure_type_id,
            fixture_id=f"fixture::{failure.failure_code}",
            expected_event_contract_id=_expected_event_contract_id(failure),
            expected_recovery_strategy_id=failure.recovery_strategy_id,
            expected_owner=failure.owner,
            reproducibility_key=f"{failure.domain}:{failure.failure_code}",
        )
        for failure in failure_types
    )


def _analytics_contracts(
    failure_types: tuple[FailureTypeDefinition, ...],
) -> tuple[FailureAnalyticsContract, ...]:
    by_domain: dict[FailureDomain, list[FailureTypeDefinition]] = defaultdict(list)
    for failure in failure_types:
        by_domain[failure.domain].append(failure)
    return tuple(
        FailureAnalyticsContract(
            analytics_contract_id=f"failure_analytics::{domain}",
            domain=domain,
            grouped_failure_type_ids=tuple(
                failure.failure_type_id for failure in by_domain[domain]
            ),
            dimensions=("domain", "severity", "root_cause", "owner"),
            metric_names=("failure_type_count", "terminal_count", "guardrail_count"),
            severity_buckets=tuple(
                severity
                for severity in _SEVERITY_ORDER
                if any(failure.severity == severity for failure in by_domain[domain])
            ),
        )
        for domain in _DOMAIN_ORDER
        if by_domain[domain]
    )


def _root_cause_classifications(
    failure_types: tuple[FailureTypeDefinition, ...],
) -> tuple[FailureRootCauseClassification, ...]:
    by_root: dict[FailureRootCause, list[FailureTypeDefinition]] = defaultdict(list)
    for failure in failure_types:
        by_root[failure.root_cause].append(failure)
    return tuple(
        FailureRootCauseClassification(
            root_cause_id=f"failure_root_cause::{root_cause}",
            root_cause=root_cause,
            failure_type_ids=tuple(
                failure.failure_type_id for failure in failures
            ),
            evidence_keys=("failure_code", "source_surface", "event_codes"),
            default_owner=failures[0].owner,
        )
        for root_cause, failures in sorted(by_root.items())
    )


def _reproducibility_records(
    failure_types: tuple[FailureTypeDefinition, ...],
) -> tuple[FailureReproducibilityRecord, ...]:
    return tuple(
        FailureReproducibilityRecord(
            reproducibility_id=(
                f"failure_reproducibility::{failure.failure_type_id}"
            ),
            failure_type_id=failure.failure_type_id,
            reproducibility_mode=failure.reproducibility_mode,
            fixture_key=f"fixture::{failure.failure_code}",
            deterministic_fields=(
                "failure_type_id",
                "failure_code",
                "domain",
                "severity",
                "root_cause",
            ),
            reproduction_steps=(
                "Load the static failure taxonomy registry.",
                f"Select {failure.failure_type_id}.",
                "Validate linked event, strategy, owner, and knowledge records.",
            ),
        )
        for failure in failure_types
    )


def _ownership_records(
    failure_types: tuple[FailureTypeDefinition, ...],
) -> tuple[FailureOwnershipRecord, ...]:
    by_owner: dict[FailureOwner, list[FailureTypeDefinition]] = defaultdict(list)
    for failure in failure_types:
        by_owner[failure.owner].append(failure)
    return tuple(
        FailureOwnershipRecord(
            ownership_id=f"failure_ownership::{owner}",
            owner=owner,
            failure_type_ids=tuple(
                failure.failure_type_id for failure in failures
            ),
            owning_module=_owner_module(owner),
            escalation_policy=(
                "Owner reviews typed failure evidence; Product Bug, HITL, "
                "Runtime Evolution, merge, push, and tag decisions remain "
                "human-controlled."
            ),
        )
        for owner, failures in sorted(by_owner.items())
    )


def _fix_recommendations(
    failure_types: tuple[FailureTypeDefinition, ...],
) -> tuple[FailureFixRecommendation, ...]:
    return tuple(
        FailureFixRecommendation(
            recommendation_id=f"failure_fix::{failure.failure_type_id}",
            failure_type_id=failure.failure_type_id,
            owner=failure.owner,
            recommended_action=_recommended_action(failure),
            validation_surface=(
                "tests/test_typed_failure_taxonomy.py",
                "git diff --check",
            ),
            fix_requires_hitl_acceptance=failure.user_visible,
        )
        for failure in failure_types
    )


def _knowledge_base_entries(
    failure_types: tuple[FailureTypeDefinition, ...],
) -> tuple[FailureKnowledgeBaseEntry, ...]:
    return tuple(
        FailureKnowledgeBaseEntry(
            kb_entry_id=f"failure_kb::{failure.failure_type_id}",
            failure_type_id=failure.failure_type_id,
            title=f"{failure.domain}: {failure.failure_code}",
            symptoms=(
                f"severity:{failure.severity}",
                f"root_cause:{failure.root_cause}",
                f"surface:{failure.source_surface}",
            ),
            diagnosis=failure.explanation,
            recovery_notes=(
                f"Use {failure.recovery_strategy_id}.",
                "Keep runtime behavior advisory until HITL-approved.",
            ),
            linked_fix_recommendation_id=f"failure_fix::{failure.failure_type_id}",
        )
        for failure in failure_types
    )


def _failure_ids_by_code(
    failure_types: tuple[FailureTypeDefinition, ...],
) -> dict[str, str]:
    return {failure.failure_code: failure.failure_type_id for failure in failure_types}


def _planning_helper_ids() -> tuple[str, ...]:
    return tuple(spec.payload_key for spec in assistant_workflow_model_payload_specs())


def _node_event_contract_ids(node_id: str) -> tuple[str, ...]:
    contract_ids = ["failure_event::node_failed", "failure_event::error"]
    if node_id in {"review", "refinement"}:
        contract_ids.extend(
            ("failure_event::review_failed", "failure_event::retry_completed")
        )
    if node_id in {"finalization", "failure", "generation"}:
        contract_ids.append("failure_event::final")
    if node_id in {"memory", "retrieval", "context_assembly"}:
        contract_ids.append("failure_event::status")
    return tuple(dict.fromkeys(contract_ids))


def _node_state_keys(
    node_id: str,
    direction: Literal["input", "output"],
) -> tuple[str, ...]:
    planning_outputs = _planning_helper_ids()
    state_io: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
        "intake": (("request",), ()),
        "routing": (("request",), ("route_decision",)),
        "memory": (("request", "route_decision"), ("memory_context",)),
        "retrieval": (("request", "route_decision"), ("retrieval_context",)),
        "context_assembly": (
            ("memory_context", "retrieval_context"),
            ("assembled_context",),
        ),
        "prompt_input": (("request", "assembled_context"), ("prompt_input",)),
        "planning": (("prompt_input",), planning_outputs),
        "director": (("creative_plan",), ("creative_director",)),
        "reasoning": (
            ("creative_plan", "creative_director"),
            ("creative_reasoning",),
        ),
        "prompt_rendering": (("prompt_input",), ("rendered_prompt",)),
        "generation": (("rendered_prompt", "route_decision"), ("final_answer",)),
        "artifact_extraction": (("final_answer",), ("artifacts",)),
        "preview_preparation": (("artifacts",), ("preview_results",)),
        "artifact_critique": (
            ("artifacts", "preview_results"),
            ("artifact_critique_summary",),
        ),
        "review": (("final_answer", "artifact_critique_summary"), ("review_result",)),
        "refinement": (("review_result",), ("refinement_count",)),
        "finalization": (("final_answer", "failure_info"), ("status", "final_answer")),
        "failure": (("failure_info",), ("status", "error_message")),
    }
    inputs, outputs = state_io[node_id]
    return inputs if direction == "input" else outputs


def _scenario_kind(
    failure: FailureTypeDefinition,
) -> FailureRegressionScenarioKind:
    if failure.domain == "serialization":
        return "event_payload_shape"
    if failure.domain in {"provider_stream", "workstation_client"}:
        return "boundary_invariant"
    if failure.root_cause == "contract_violation":
        return "unit_model_validation"
    return "contract_lookup"


def _expected_event_contract_id(failure: FailureTypeDefinition) -> str:
    if failure.failure_code == "review_quality_gate":
        return "failure_event::review_failed"
    if failure.failure_code in {"finalization_failure", "provider_generation_error"}:
        return "failure_event::final"
    if failure.failure_code in {"missing_runtime_dependency", "missing_configuration"}:
        return "failure_event::status"
    if failure.domain == "workflow_node":
        return "failure_event::node_failed"
    return "failure_event::error"


def _owner_module(owner: FailureOwner) -> str:
    return {
        "runtime_graph": "creative_coding_assistant.orchestration.workflow_graph",
        "planning_system": "creative_coding_assistant.orchestration.workflow_graph",
        "provider_adapter": "creative_coding_assistant.llm",
        "serialization_boundary": "creative_coding_assistant.contracts",
        "workstation_client": "creative_coding_assistant.clients",
        "quality_review": "creative_coding_assistant.orchestration.workflow_review",
        "observability": "creative_coding_assistant.analytics",
    }[owner]


def _recommended_action(failure: FailureTypeDefinition) -> str:
    if failure.terminal:
        return (
            "Preserve the terminal failure answer and add a focused regression "
            f"for {failure.failure_code}."
        )
    if failure.retry_eligible:
        return (
            "Validate retry budget invariants and keep retry execution bounded "
            "by workflow review policy."
        )
    return (
        "Keep the failure advisory and validate lookup, event, owner, and "
        f"knowledge-base links for {failure.failure_code}."
    )


def _referenced_failure_type_ids(
    registry: TypedFailureTaxonomyRegistry,
) -> set[str]:
    referenced: set[str] = set()
    for collection in (
        registry.node_failure_models,
        registry.planning_sub_helper_models,
        registry.provider_stream_models,
        registry.serialization_models,
        registry.workstation_client_boundary_models,
    ):
        for model in collection:
            referenced.update(model.supported_failure_type_ids)
    for contract in registry.event_contracts:
        referenced.update(contract.mapped_failure_type_ids)
    for scenario in registry.regression_scenarios:
        referenced.add(scenario.failure_type_id)
    for entry in registry.knowledge_base_entries:
        referenced.add(entry.failure_type_id)
    return referenced
