"""V7.1 runtime graph consolidation contracts and diagnostics."""

from __future__ import annotations

from collections import Counter
from typing import Literal, Self

from langgraph.graph import END, START
from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.execution_graph_analyzer import (
    ExecutionGraphAnalysis,
    ExecutionGraphEdge,
    analyze_assistant_execution_graph,
)
from creative_coding_assistant.orchestration.workflow import AssistantWorkflowState
from creative_coding_assistant.orchestration.workflow_graph import (
    ASSISTANT_WORKFLOW_NODE_ORDER,
    ASSISTANT_WORKFLOW_RECURSION_LIMIT,
    assistant_workflow_final_payload_keys,
    assistant_workflow_model_payload_specs,
    assistant_workflow_node_specs,
)

RUNTIME_GRAPH_CONSOLIDATION_SERIALIZATION_VERSION = (
    "runtime_graph_consolidation.v1"
)
WORKFLOW_CONTRACT_VALIDATION_SERIALIZATION_VERSION = (
    "workflow_contract_validation.v1"
)
GRAPH_INVARIANT_VERIFICATION_SERIALIZATION_VERSION = (
    "graph_invariant_verification.v1"
)
EXECUTION_TRACE_RECORDER_SERIALIZATION_VERSION = "execution_trace_record.v1"
EXECUTION_GRAPH_EXPLAINABILITY_SERIALIZATION_VERSION = (
    "execution_graph_explainability.v1"
)
GRAPH_DIFF_ENGINE_SERIALIZATION_VERSION = "graph_diff_engine.v1"
WORKFLOW_DETERMINISM_AUDIT_SERIALIZATION_VERSION = (
    "workflow_determinism_audit.v1"
)
GRAPH_PERFORMANCE_PROFILE_SERIALIZATION_VERSION = (
    "graph_performance_profile.v1"
)
EXECUTION_COST_PROFILE_SERIALIZATION_VERSION = "execution_cost_profile.v1"
EXECUTION_LATENCY_PROFILE_SERIALIZATION_VERSION = "execution_latency_profile.v1"
WORKFLOW_STATE_NORMALIZATION_SERIALIZATION_VERSION = (
    "workflow_state_normalization.v1"
)
EXECUTION_GRAPH_VISUALIZATION_SERIALIZATION_VERSION = (
    "execution_graph_visualization.v1"
)

RUNTIME_GRAPH_CONSOLIDATION_ROADMAP_ITEMS = (
    "Workflow Graph Audit",
    "LangGraph Node Decomposition Plan",
    "Creative Cognition Node Extraction",
    "Generative Design Node Extraction",
    "Artifact Intelligence Node Extraction",
    "Creative Evaluation Node Extraction",
    "Micro Error Path Design",
    "Subgraph Boundary Contracts",
    "Backward Compatibility Tests",
    "Workflow Node Handler Extraction",
    "Runtime Graph Module Split",
    "Unified Execution Graph Refactor",
    "Workflow State Normalization",
    "Execution Graph Visualization",
    "Graph Performance Profiling",
    "Workflow Contract Validator",
    "Graph Invariant Verification",
    "Execution Trace Recorder",
    "Execution Graph Explainability",
    "Graph Diff Engine",
    "Workflow Determinism Audit",
    "Execution Cost Profiling",
    "Execution Latency Profiling",
)
RUNTIME_GRAPH_BLOCKED_BEHAVIORS = (
    "provider_model_routing_change",
    "workflow_order_mutation",
    "langgraph_node_addition",
    "node_handler_invocation",
    "workflow_execution",
    "generated_output_mutation",
    "storage_mutation",
    "runtime_evolution_application",
    "hitl_decision_application",
)

RuntimeGraphSubgraphId = Literal[
    "workflow_foundation",
    "creative_cognition",
    "generative_design",
    "artifact_intelligence",
    "creative_evaluation",
    "failure_boundary",
]
RuntimeGraphModuleRole = Literal[
    "langgraph_adapter",
    "static_topology_analysis",
    "cognitive_execution_projection",
    "v7_consolidation_contracts",
    "workflow_state_contracts",
]

_SUBGRAPH_ORDER: tuple[RuntimeGraphSubgraphId, ...] = (
    "workflow_foundation",
    "creative_cognition",
    "generative_design",
    "artifact_intelligence",
    "creative_evaluation",
    "failure_boundary",
)
_PRIMARY_SUBGRAPH_BY_NODE: dict[str, RuntimeGraphSubgraphId] = {
    "intake": "workflow_foundation",
    "routing": "workflow_foundation",
    "memory": "workflow_foundation",
    "retrieval": "workflow_foundation",
    "context_assembly": "workflow_foundation",
    "prompt_input": "workflow_foundation",
    "planning": "creative_cognition",
    "director": "creative_cognition",
    "reasoning": "creative_cognition",
    "prompt_rendering": "workflow_foundation",
    "generation": "workflow_foundation",
    "artifact_extraction": "artifact_intelligence",
    "preview_preparation": "artifact_intelligence",
    "artifact_critique": "artifact_intelligence",
    "review": "creative_evaluation",
    "refinement": "creative_evaluation",
    "finalization": "workflow_foundation",
    "failure": "failure_boundary",
}
_OVERLAY_SUBGRAPHS_BY_NODE: dict[str, tuple[RuntimeGraphSubgraphId, ...]] = {
    "planning": (
        "creative_cognition",
        "generative_design",
        "artifact_intelligence",
        "creative_evaluation",
    ),
    "artifact_critique": ("artifact_intelligence", "creative_evaluation"),
    "review": ("creative_evaluation",),
    "refinement": ("creative_evaluation",),
    "failure": ("failure_boundary",),
}
_SUBGRAPH_ROADMAP_ITEM: dict[RuntimeGraphSubgraphId, str] = {
    "workflow_foundation": "Workflow Graph Audit",
    "creative_cognition": "Creative Cognition Node Extraction",
    "generative_design": "Generative Design Node Extraction",
    "artifact_intelligence": "Artifact Intelligence Node Extraction",
    "creative_evaluation": "Creative Evaluation Node Extraction",
    "failure_boundary": "Micro Error Path Design",
}
_STATE_IO_BY_NODE: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "intake": (("request",), ()),
    "routing": (("request",), ("route_decision",)),
    "memory": (("request", "route_decision"), ("memory_context",)),
    "retrieval": (("request", "route_decision"), ("retrieval_context",)),
    "context_assembly": (
        ("route_decision", "memory_context", "retrieval_context"),
        ("assembled_context",),
    ),
    "prompt_input": (
        ("request", "route_decision", "assembled_context"),
        ("prompt_input", "clarification"),
    ),
    "planning": (
        ("request", "route_decision", "prompt_input", "clarification"),
        (
            "creative_intent",
            "creative_hierarchy",
            "creative_strategy",
            "creative_techniques",
            "creative_plan",
            "creative_constraints",
            "creative_constraint_priorities",
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
            "artifact_plan",
            "artifact_dependency_graph",
            "runtime_compatibility",
            "artifact_capability_matrix",
            "multi_artifact_strategy",
            "artifact_critic",
            "artifact_refiner",
            "artifact_intelligence_synthesis",
            "artifact_merge_planner",
            "artifact_export_intelligence",
            "artifact_engine_contracts",
            "evaluation_engine_contracts",
            "creative_critic",
            "self_evaluation",
            "creative_improvement_planner",
            "reflection_loop",
            "creative_confidence",
            "creative_score",
            "consistency_validation",
            "evaluation_report",
        ),
    ),
    "director": (
        ("request", "route_decision", "prompt_input", "creative_plan"),
        ("creative_director",),
    ),
    "reasoning": (
        (
            "request",
            "route_decision",
            "prompt_input",
            "creative_plan",
            "creative_director",
        ),
        ("creative_reasoning",),
    ),
    "prompt_rendering": (("prompt_input",), ("rendered_prompt",)),
    "generation": (("rendered_prompt", "route_decision"), ("final_answer",)),
    "artifact_extraction": (("final_answer",), ("artifacts",)),
    "preview_preparation": (("artifacts",), ("preview_results",)),
    "artifact_critique": (
        ("artifacts", "preview_results", "artifact_plan"),
        ("artifact_critique_summary",),
    ),
    "review": (
        ("final_answer", "artifact_critique_summary", "refinement_count"),
        ("review_result",),
    ),
    "refinement": (
        ("review_result", "refinement_passes"),
        ("refinement_count", "refinement_passes"),
    ),
    "finalization": (
        ("final_answer", "clarification", "failure_info"),
        ("status", "final_answer"),
    ),
    "failure": (("failure_info",), ("status", "error_message")),
}
_EVENT_CODES_BY_NODE: dict[str, tuple[str, ...]] = {
    "intake": ("request_received",),
    "routing": ("route_selected",),
    "memory": ("memory_context_available", "memory_context_unavailable"),
    "retrieval": ("retrieval_context_available", "retrieval_context_unavailable"),
    "context_assembly": ("context_assembled", "context_assembly_unavailable"),
    "prompt_input": (
        "prompt_input_prepared",
        "prompt_input_unavailable",
        "clarification_required",
    ),
    "planning": ("creative_plan_prepared",),
    "director": ("creative_director_prepared",),
    "reasoning": ("creative_reasoning_prepared",),
    "prompt_rendering": ("prompt_rendered", "prompt_rendering_unavailable"),
    "generation": ("generation_completed", "shell_answer_completed"),
    "artifact_extraction": ("artifacts_extracted", "artifacts_unavailable"),
    "preview_preparation": ("preview_results_prepared", "preview_unavailable"),
    "artifact_critique": ("artifact_critique_completed",),
    "review": ("review_passed", "review_retry_requested", "review_failed"),
    "refinement": ("refinement_completed",),
    "finalization": ("final_response",),
    "failure": ("workflow_failed",),
}


class RuntimeGraphNodeContract(BaseModel):
    """Stable contract for one assistant workflow node."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    node_id: str = Field(min_length=1, max_length=120)
    order_index: int = Field(ge=0)
    primary_subgraph_id: RuntimeGraphSubgraphId
    subgraph_ids: tuple[RuntimeGraphSubgraphId, ...] = Field(
        min_length=1,
        max_length=6,
    )
    handler_reference: str = Field(min_length=1, max_length=220)
    state_inputs: tuple[str, ...] = Field(default_factory=tuple, max_length=80)
    state_outputs: tuple[str, ...] = Field(default_factory=tuple, max_length=80)
    emitted_event_codes: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    failure_transition_target: Literal["failure"] = "failure"
    handler_extraction_status: Literal["contract_extracted"] = (
        "contract_extracted"
    )
    deterministic_state_update: bool = True
    provider_model_routing_change_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    storage_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    backward_compatibility_required: Literal[True] = True
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _contract_matches_boundaries(self) -> Self:
        if self.primary_subgraph_id not in self.subgraph_ids:
            raise ValueError("primary_subgraph_id must be listed in subgraph_ids")
        if self.order_index >= len(ASSISTANT_WORKFLOW_NODE_ORDER):
            raise ValueError("order_index must be within workflow node order")
        if ASSISTANT_WORKFLOW_NODE_ORDER[self.order_index] != self.node_id:
            raise ValueError("node_id must match workflow order")
        if not self.handler_reference.endswith(f"._{self.node_id}_node"):
            raise ValueError("handler_reference must match node handler")
        return self


class RuntimeGraphSubgraphContract(BaseModel):
    """Boundary contract for a V7.1 runtime graph subgraph."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    subgraph_id: RuntimeGraphSubgraphId
    roadmap_item: str = Field(min_length=1, max_length=120)
    node_ids: tuple[str, ...] = Field(min_length=1, max_length=24)
    accepts_state_keys: tuple[str, ...] = Field(default_factory=tuple, max_length=80)
    emits_state_keys: tuple[str, ...] = Field(default_factory=tuple, max_length=80)
    boundary_summary: str = Field(min_length=1, max_length=600)
    owner_module: str = Field(min_length=1, max_length=180)
    failure_boundary_node_id: Literal["failure"] = "failure"
    subgraph_extraction_status: Literal["boundary_contract_extracted"] = (
        "boundary_contract_extracted"
    )
    workflow_graph_mutation_implemented: Literal[False] = False
    provider_model_routing_change_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    storage_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _subgraph_contract_matches_known_nodes(self) -> Self:
        unknown = set(self.node_ids).difference(ASSISTANT_WORKFLOW_NODE_ORDER)
        if unknown:
            raise ValueError("subgraph node_ids must reference known nodes")
        expected_roadmap_item = _SUBGRAPH_ROADMAP_ITEM[self.subgraph_id]
        if self.roadmap_item != expected_roadmap_item:
            raise ValueError("roadmap_item must match subgraph")
        return self


class RuntimeGraphDecompositionItem(BaseModel):
    """One decomposition unit in the LangGraph node split plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    decomposition_id: str = Field(min_length=1, max_length=160)
    roadmap_item: str = Field(min_length=1, max_length=120)
    source_node_ids: tuple[str, ...] = Field(min_length=1, max_length=24)
    target_subgraph_id: RuntimeGraphSubgraphId
    extraction_strategy: str = Field(min_length=1, max_length=500)
    behavior_change_required: Literal[False] = False
    validation_surface: tuple[str, ...] = Field(min_length=1, max_length=12)


class RuntimeGraphModuleSplit(BaseModel):
    """Runtime graph module responsibility boundary."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    module_name: str = Field(min_length=1, max_length=180)
    module_role: RuntimeGraphModuleRole
    responsibility: str = Field(min_length=1, max_length=500)
    owns_live_execution: bool = False
    exposes_public_contract: bool = True
    behavior_change_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False


class RuntimeGraphConsolidationPlan(BaseModel):
    """Read-only V7.1 consolidation plan for the assistant runtime graph."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["runtime_graph_consolidation"] = "runtime_graph_consolidation"
    serialization_version: Literal["runtime_graph_consolidation.v1"] = (
        RUNTIME_GRAPH_CONSOLIDATION_SERIALIZATION_VERSION
    )
    source_graph: Literal["assistant_workflow_graph"] = "assistant_workflow_graph"
    source_graph_serialization_version: str = Field(min_length=1, max_length=120)
    source_execution_graph_role: str = Field(min_length=1, max_length=120)
    source_node_order: tuple[str, ...] = Field(min_length=1, max_length=40)
    source_edge_ids: tuple[str, ...] = Field(min_length=1, max_length=140)
    node_contracts: tuple[RuntimeGraphNodeContract, ...] = Field(
        min_length=1,
        max_length=40,
    )
    subgraph_contracts: tuple[RuntimeGraphSubgraphContract, ...] = Field(
        min_length=6,
        max_length=6,
    )
    decomposition_items: tuple[RuntimeGraphDecompositionItem, ...] = Field(
        min_length=6,
        max_length=12,
    )
    module_split: tuple[RuntimeGraphModuleSplit, ...] = Field(
        min_length=5,
        max_length=5,
    )
    normalized_state_keys: tuple[str, ...] = Field(min_length=1, max_length=120)
    final_payload_keys: tuple[str, ...] = Field(min_length=1, max_length=80)
    runtime_payload_keys: tuple[str, ...] = Field(min_length=1, max_length=80)
    recursion_limit: int = Field(ge=1)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=23, max_length=23)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=9, max_length=9)
    behavior_change_implemented: Literal[False] = False
    provider_model_routing_change_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    storage_mutation_implemented: Literal[False] = False
    uncontrolled_workflow_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    v7_2_scope_started: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_runtime_graph(self) -> Self:
        node_ids = tuple(contract.node_id for contract in self.node_contracts)
        if node_ids != self.source_node_order:
            raise ValueError("node_contracts must match source_node_order")
        if self.source_node_order != ASSISTANT_WORKFLOW_NODE_ORDER:
            raise ValueError("source_node_order must match assistant workflow")
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("node ids must be unique")
        subgraph_ids = tuple(
            contract.subgraph_id for contract in self.subgraph_contracts
        )
        if subgraph_ids != _SUBGRAPH_ORDER:
            raise ValueError("subgraph_contracts must follow V7.1 subgraph order")
        covered_nodes = set()
        for contract in self.subgraph_contracts:
            covered_nodes.update(contract.node_ids)
        if covered_nodes != set(self.source_node_order):
            raise ValueError("subgraph_contracts must cover every workflow node")
        if self.covered_roadmap_items != RUNTIME_GRAPH_CONSOLIDATION_ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V7.1 roadmap")
        if self.blocked_runtime_behaviors != RUNTIME_GRAPH_BLOCKED_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V7.1 boundary")
        if len(set(self.final_payload_keys)) != len(self.final_payload_keys):
            raise ValueError("final_payload_keys must be unique")
        if len(set(self.runtime_payload_keys)) != len(self.runtime_payload_keys):
            raise ValueError("runtime_payload_keys must be unique")
        if not set(self.final_payload_keys).issubset(self.normalized_state_keys):
            raise ValueError("final payload keys must be normalized state keys")
        return self


class WorkflowContractValidationReport(BaseModel):
    """Validation output for V7.1 workflow graph contracts."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["workflow_contract_validator"] = "workflow_contract_validator"
    serialization_version: Literal["workflow_contract_validation.v1"] = (
        WORKFLOW_CONTRACT_VALIDATION_SERIALIZATION_VERSION
    )
    validation_passed: bool
    checked_node_contract_count: int = Field(ge=0)
    checked_subgraph_contract_count: int = Field(ge=0)
    checked_roadmap_item_count: int = Field(ge=0)
    missing_node_contract_ids: tuple[str, ...] = Field(default_factory=tuple)
    missing_subgraph_ids: tuple[str, ...] = Field(default_factory=tuple)
    compatibility_failures: tuple[str, ...] = Field(default_factory=tuple)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=9, max_length=9)
    workflow_execution_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False


class GraphInvariantVerificationReport(BaseModel):
    """Invariant checks for the runtime graph topology and contracts."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["graph_invariant_verification"] = "graph_invariant_verification"
    serialization_version: Literal["graph_invariant_verification.v1"] = (
        GRAPH_INVARIANT_VERIFICATION_SERIALIZATION_VERSION
    )
    invariant_status: Literal["pass", "fail"]
    checked_invariants: tuple[str, ...] = Field(min_length=1, max_length=20)
    failed_invariants: tuple[str, ...] = Field(default_factory=tuple)
    deterministic_node_order: bool
    unique_node_ids: bool
    unique_edge_ids: bool
    failure_path_reachable: bool
    bounded_retry_cycle_detected: bool
    terminal_nodes_stable: bool
    workflow_execution_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False


class ExecutionTraceStep(BaseModel):
    """One static trace step for the expected workflow contract."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    sequence_index: int = Field(ge=0)
    node_id: str = Field(min_length=1, max_length=120)
    entered_from_edge_id: str = Field(min_length=1, max_length=180)
    expected_event_surface: str = Field(min_length=1, max_length=120)
    trace_source: Literal["static_contract"] = "static_contract"


class ExecutionTraceRecord(BaseModel):
    """Read-only trace record derived from graph contracts."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["execution_trace_recorder"] = "execution_trace_recorder"
    serialization_version: Literal["execution_trace_record.v1"] = (
        EXECUTION_TRACE_RECORDER_SERIALIZATION_VERSION
    )
    trace_id: str = Field(min_length=1, max_length=160)
    trace_source: Literal["static_contract"] = "static_contract"
    trace_steps: tuple[ExecutionTraceStep, ...] = Field(min_length=1, max_length=40)
    node_ids: tuple[str, ...] = Field(min_length=1, max_length=40)
    edge_ids: tuple[str, ...] = Field(min_length=1, max_length=40)
    step_count: int = Field(ge=1)
    workflow_execution_implemented: Literal[False] = False
    telemetry_emission_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False

    @model_validator(mode="after")
    def _trace_counts_match(self) -> Self:
        if self.step_count != len(self.trace_steps):
            raise ValueError("step_count must match trace_steps")
        if self.node_ids != tuple(step.node_id for step in self.trace_steps):
            raise ValueError("node_ids must match trace steps")
        if self.edge_ids != tuple(
            step.entered_from_edge_id for step in self.trace_steps
        ):
            raise ValueError("edge_ids must match trace steps")
        return self


class ExecutionGraphExplanation(BaseModel):
    """Human-readable explanation of the consolidated runtime graph."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["execution_graph_explainability"] = (
        "execution_graph_explainability"
    )
    serialization_version: Literal["execution_graph_explainability.v1"] = (
        EXECUTION_GRAPH_EXPLAINABILITY_SERIALIZATION_VERSION
    )
    summary: str = Field(min_length=1, max_length=1000)
    node_explanations: tuple[str, ...] = Field(min_length=1, max_length=40)
    subgraph_explanations: tuple[str, ...] = Field(min_length=6, max_length=6)
    failure_boundary_explanation: str = Field(min_length=1, max_length=800)
    routing_change_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False


class GraphDiffReport(BaseModel):
    """Diff report between two runtime graph consolidation plans."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["graph_diff_engine"] = "graph_diff_engine"
    serialization_version: Literal["graph_diff_engine.v1"] = (
        GRAPH_DIFF_ENGINE_SERIALIZATION_VERSION
    )
    diff_status: Literal["no_change", "changed"]
    added_node_ids: tuple[str, ...] = Field(default_factory=tuple)
    removed_node_ids: tuple[str, ...] = Field(default_factory=tuple)
    reordered_node_ids: tuple[str, ...] = Field(default_factory=tuple)
    added_edge_ids: tuple[str, ...] = Field(default_factory=tuple)
    removed_edge_ids: tuple[str, ...] = Field(default_factory=tuple)
    changed_subgraph_ids: tuple[str, ...] = Field(default_factory=tuple)
    behavior_change_detected: bool
    provider_model_routing_change_detected: bool
    generated_output_mutation_detected: bool
    workflow_execution_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False


class WorkflowDeterminismAudit(BaseModel):
    """Determinism audit for the static runtime graph contract."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["workflow_determinism_audit"] = "workflow_determinism_audit"
    serialization_version: Literal["workflow_determinism_audit.v1"] = (
        WORKFLOW_DETERMINISM_AUDIT_SERIALIZATION_VERSION
    )
    deterministic: bool
    deterministic_node_order: bool
    deterministic_edge_ids: bool
    deterministic_payload_order: bool
    nondeterministic_surfaces: tuple[str, ...] = Field(default_factory=tuple)
    audit_notes: tuple[str, ...] = Field(min_length=1, max_length=12)
    workflow_execution_implemented: Literal[False] = False
    provider_model_routing_change_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False


class RuntimeGraphNodeProfile(BaseModel):
    """Static relative performance profile for one workflow node."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    node_id: str = Field(min_length=1, max_length=120)
    relative_cost_units: int = Field(ge=1, le=10)
    relative_latency_units: int = Field(ge=1, le=12)
    outgoing_edge_count: int = Field(ge=0)
    failure_edge_count: int = Field(ge=0)
    retry_edge_count: int = Field(ge=0)
    profiling_basis: Literal["static_topology"] = "static_topology"


class GraphPerformanceProfile(BaseModel):
    """Static graph performance posture for V7.1 visibility."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["graph_performance_profiling"] = "graph_performance_profiling"
    serialization_version: Literal["graph_performance_profile.v1"] = (
        GRAPH_PERFORMANCE_PROFILE_SERIALIZATION_VERSION
    )
    measurement_mode: Literal["static_relative"] = "static_relative"
    node_profiles: tuple[RuntimeGraphNodeProfile, ...] = Field(
        min_length=1,
        max_length=40,
    )
    total_relative_cost_units: int = Field(ge=1)
    total_relative_latency_units: int = Field(ge=1)
    branch_count: int = Field(ge=0)
    retry_edge_count: int = Field(ge=0)
    failure_edge_count: int = Field(ge=0)
    critical_path_node_count: int = Field(ge=1)
    live_timing_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    provider_model_routing_change_implemented: Literal[False] = False


class ExecutionCostProfile(BaseModel):
    """Static relative execution cost profile without pricing lookup."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["execution_cost_profiling"] = "execution_cost_profiling"
    serialization_version: Literal["execution_cost_profile.v1"] = (
        EXECUTION_COST_PROFILE_SERIALIZATION_VERSION
    )
    measurement_mode: Literal["static_relative"] = "static_relative"
    relative_total_cost_units: int = Field(ge=1)
    highest_relative_cost_node_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    pricing_lookup_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    provider_model_routing_change_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False


class ExecutionLatencyProfile(BaseModel):
    """Static relative execution latency profile without live timing."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["execution_latency_profiling"] = "execution_latency_profiling"
    serialization_version: Literal["execution_latency_profile.v1"] = (
        EXECUTION_LATENCY_PROFILE_SERIALIZATION_VERSION
    )
    measurement_mode: Literal["static_relative"] = "static_relative"
    relative_total_latency_units: int = Field(ge=1)
    highest_relative_latency_node_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=8,
    )
    live_timing_implemented: Literal[False] = False
    latency_budget_enforcement_implemented: Literal[False] = False
    provider_model_routing_change_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False


class WorkflowStateNormalizationReport(BaseModel):
    """Canonical state-key report for workflow state normalization."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["workflow_state_normalization"] = "workflow_state_normalization"
    serialization_version: Literal["workflow_state_normalization.v1"] = (
        WORKFLOW_STATE_NORMALIZATION_SERIALIZATION_VERSION
    )
    canonical_state_keys: tuple[str, ...] = Field(min_length=1, max_length=120)
    runtime_payload_keys: tuple[str, ...] = Field(min_length=1, max_length=80)
    final_payload_keys: tuple[str, ...] = Field(min_length=1, max_length=80)
    missing_runtime_payload_keys: tuple[str, ...] = Field(default_factory=tuple)
    missing_final_payload_keys: tuple[str, ...] = Field(default_factory=tuple)
    normalization_passed: bool
    state_mutation_implemented: Literal[False] = False
    storage_mutation_implemented: Literal[False] = False


class ExecutionGraphVisualization(BaseModel):
    """Mermaid visualization of the current runtime graph contract."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["execution_graph_visualization"] = "execution_graph_visualization"
    serialization_version: Literal["execution_graph_visualization.v1"] = (
        EXECUTION_GRAPH_VISUALIZATION_SERIALIZATION_VERSION
    )
    format: Literal["mermaid"] = "mermaid"
    diagram: str = Field(min_length=1, max_length=12000)
    node_count: int = Field(ge=1)
    edge_count: int = Field(ge=1)
    generated_from_static_contract: Literal[True] = True
    workflow_execution_implemented: Literal[False] = False


def build_runtime_graph_consolidation_plan(
    *,
    execution_graph: ExecutionGraphAnalysis | None = None,
) -> RuntimeGraphConsolidationPlan:
    """Build the V7.1 graph consolidation contract without executing the graph."""

    analysis = execution_graph or analyze_assistant_execution_graph()
    return RuntimeGraphConsolidationPlan(
        source_graph_serialization_version=analysis.serialization_version,
        source_execution_graph_role=analysis.role,
        source_node_order=analysis.node_order,
        source_edge_ids=tuple(edge.edge_id for edge in analysis.edges),
        node_contracts=_node_contracts(),
        subgraph_contracts=_subgraph_contracts(),
        decomposition_items=_decomposition_items(),
        module_split=_module_split(),
        normalized_state_keys=_canonical_state_keys(),
        final_payload_keys=assistant_workflow_final_payload_keys(),
        runtime_payload_keys=tuple(
            spec.payload_key for spec in assistant_workflow_model_payload_specs()
        ),
        recursion_limit=ASSISTANT_WORKFLOW_RECURSION_LIMIT,
        covered_roadmap_items=RUNTIME_GRAPH_CONSOLIDATION_ROADMAP_ITEMS,
        blocked_runtime_behaviors=RUNTIME_GRAPH_BLOCKED_BEHAVIORS,
    )


def runtime_graph_node_contract_by_id(
    node_id: str,
    plan: RuntimeGraphConsolidationPlan | None = None,
) -> RuntimeGraphNodeContract | None:
    """Return one V7.1 node contract without invoking the node handler."""

    source_plan = plan or build_runtime_graph_consolidation_plan()
    for contract in source_plan.node_contracts:
        if contract.node_id == node_id:
            return contract
    return None


def runtime_graph_node_contracts_for_subgraph(
    subgraph_id: RuntimeGraphSubgraphId,
    plan: RuntimeGraphConsolidationPlan | None = None,
) -> tuple[RuntimeGraphNodeContract, ...]:
    """Return node contracts covered by one subgraph boundary."""

    source_plan = plan or build_runtime_graph_consolidation_plan()
    return tuple(
        contract
        for contract in source_plan.node_contracts
        if subgraph_id in contract.subgraph_ids
    )


def validate_runtime_graph_contracts(
    plan: RuntimeGraphConsolidationPlan | None = None,
) -> WorkflowContractValidationReport:
    """Validate V7.1 runtime graph contract coverage."""

    source_plan = plan or build_runtime_graph_consolidation_plan()
    node_ids = {contract.node_id for contract in source_plan.node_contracts}
    subgraph_ids = {contract.subgraph_id for contract in source_plan.subgraph_contracts}
    missing_nodes = tuple(
        node_id for node_id in ASSISTANT_WORKFLOW_NODE_ORDER if node_id not in node_ids
    )
    missing_subgraphs = tuple(
        subgraph_id
        for subgraph_id in _SUBGRAPH_ORDER
        if subgraph_id not in subgraph_ids
    )
    compatibility_failures = _compatibility_failures(source_plan)
    return WorkflowContractValidationReport(
        validation_passed=not (
            missing_nodes or missing_subgraphs or compatibility_failures
        ),
        checked_node_contract_count=len(source_plan.node_contracts),
        checked_subgraph_contract_count=len(source_plan.subgraph_contracts),
        checked_roadmap_item_count=len(source_plan.covered_roadmap_items),
        missing_node_contract_ids=missing_nodes,
        missing_subgraph_ids=missing_subgraphs,
        compatibility_failures=compatibility_failures,
        blocked_runtime_behaviors=RUNTIME_GRAPH_BLOCKED_BEHAVIORS,
    )


def verify_runtime_graph_invariants(
    plan: RuntimeGraphConsolidationPlan | None = None,
    execution_graph: ExecutionGraphAnalysis | None = None,
) -> GraphInvariantVerificationReport:
    """Verify static graph invariants without compiling or streaming."""

    source_plan = plan or build_runtime_graph_consolidation_plan()
    analysis = execution_graph or analyze_assistant_execution_graph()
    node_ids = source_plan.source_node_order
    edge_ids = source_plan.source_edge_ids
    checks = {
        "deterministic_node_order": node_ids == ASSISTANT_WORKFLOW_NODE_ORDER,
        "unique_node_ids": len(set(node_ids)) == len(node_ids),
        "unique_edge_ids": len(set(edge_ids)) == len(edge_ids),
        "failure_path_reachable": analysis.failure_path_reachable,
        "bounded_retry_cycle_detected": analysis.bounded_retry_cycle_detected,
        "terminal_nodes_stable": analysis.terminal_node_ids
        == ("finalization", "failure"),
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    return GraphInvariantVerificationReport(
        invariant_status="fail" if failed else "pass",
        checked_invariants=tuple(checks),
        failed_invariants=failed,
        deterministic_node_order=checks["deterministic_node_order"],
        unique_node_ids=checks["unique_node_ids"],
        unique_edge_ids=checks["unique_edge_ids"],
        failure_path_reachable=checks["failure_path_reachable"],
        bounded_retry_cycle_detected=checks["bounded_retry_cycle_detected"],
        terminal_nodes_stable=checks["terminal_nodes_stable"],
    )


def record_runtime_graph_trace(
    plan: RuntimeGraphConsolidationPlan | None = None,
    execution_graph: ExecutionGraphAnalysis | None = None,
) -> ExecutionTraceRecord:
    """Record the static critical-path trace without running the workflow."""

    source_plan = plan or build_runtime_graph_consolidation_plan()
    analysis = execution_graph or analyze_assistant_execution_graph()
    critical_nodes = tuple(analysis.critical_path_node_ids)
    edge_lookup = _edge_lookup(analysis.edges)
    trace_steps: list[ExecutionTraceStep] = []
    previous = str(START)
    for index, node_id in enumerate(critical_nodes):
        edge_id = edge_lookup.get((previous, node_id), f"{previous}->{node_id}")
        contract = runtime_graph_node_contract_by_id(node_id, source_plan)
        event_surface = (
            contract.emitted_event_codes[0]
            if contract is not None and contract.emitted_event_codes
            else "node_lifecycle"
        )
        trace_steps.append(
            ExecutionTraceStep(
                sequence_index=index,
                node_id=node_id,
                entered_from_edge_id=edge_id,
                expected_event_surface=event_surface,
            )
        )
        previous = node_id
    return ExecutionTraceRecord(
        trace_id="runtime_graph_trace::critical_path",
        trace_steps=tuple(trace_steps),
        node_ids=tuple(step.node_id for step in trace_steps),
        edge_ids=tuple(step.entered_from_edge_id for step in trace_steps),
        step_count=len(trace_steps),
    )


def explain_runtime_graph(
    plan: RuntimeGraphConsolidationPlan | None = None,
) -> ExecutionGraphExplanation:
    """Return a static explanation for the consolidated execution graph."""

    source_plan = plan or build_runtime_graph_consolidation_plan()
    return ExecutionGraphExplanation(
        summary=(
            "V7.1 preserves the existing assistant LangGraph node order while "
            "making handler ownership, subgraph boundaries, state keys, failure "
            "edges, and profiling surfaces explicit."
        ),
        node_explanations=tuple(
            f"{contract.node_id}: {contract.primary_subgraph_id} owns "
            f"{len(contract.state_outputs)} state output contract(s)."
            for contract in source_plan.node_contracts
        ),
        subgraph_explanations=tuple(
            f"{contract.subgraph_id}: {contract.boundary_summary}"
            for contract in source_plan.subgraph_contracts
        ),
        failure_boundary_explanation=(
            "Every conditional runtime edge retains an explicit failure target; "
            "the terminal failure node remains the only failure boundary node."
        ),
    )


def diff_runtime_graphs(
    baseline: RuntimeGraphConsolidationPlan,
    candidate: RuntimeGraphConsolidationPlan,
) -> GraphDiffReport:
    """Compare two V7.1 graph contracts without mutating either graph."""

    added_nodes = tuple(
        node_id
        for node_id in candidate.source_node_order
        if node_id not in baseline.source_node_order
    )
    removed_nodes = tuple(
        node_id
        for node_id in baseline.source_node_order
        if node_id not in candidate.source_node_order
    )
    reordered_nodes = (
        ()
        if baseline.source_node_order == candidate.source_node_order
        else tuple(candidate.source_node_order)
    )
    added_edges = tuple(
        edge_id
        for edge_id in candidate.source_edge_ids
        if edge_id not in baseline.source_edge_ids
    )
    removed_edges = tuple(
        edge_id
        for edge_id in baseline.source_edge_ids
        if edge_id not in candidate.source_edge_ids
    )
    changed_subgraphs = tuple(
        candidate_contract.subgraph_id
        for baseline_contract, candidate_contract in zip(
            baseline.subgraph_contracts,
            candidate.subgraph_contracts,
            strict=False,
        )
        if baseline_contract != candidate_contract
    )
    changed = bool(
        added_nodes
        or removed_nodes
        or reordered_nodes
        or added_edges
        or removed_edges
        or changed_subgraphs
    )
    return GraphDiffReport(
        diff_status="changed" if changed else "no_change",
        added_node_ids=added_nodes,
        removed_node_ids=removed_nodes,
        reordered_node_ids=reordered_nodes,
        added_edge_ids=added_edges,
        removed_edge_ids=removed_edges,
        changed_subgraph_ids=changed_subgraphs,
        behavior_change_detected=changed,
        provider_model_routing_change_detected=(
            baseline.provider_model_routing_change_implemented
            != candidate.provider_model_routing_change_implemented
        ),
        generated_output_mutation_detected=(
            baseline.generated_output_mutation_implemented
            != candidate.generated_output_mutation_implemented
        ),
    )


def audit_runtime_graph_determinism(
    plan: RuntimeGraphConsolidationPlan | None = None,
) -> WorkflowDeterminismAudit:
    """Audit deterministic graph surfaces without executing the graph."""

    source_plan = plan or build_runtime_graph_consolidation_plan()
    deterministic_node_order = (
        source_plan.source_node_order == ASSISTANT_WORKFLOW_NODE_ORDER
    )
    deterministic_edge_ids = len(set(source_plan.source_edge_ids)) == len(
        source_plan.source_edge_ids
    )
    deterministic_payload_order = (
        source_plan.final_payload_keys == assistant_workflow_final_payload_keys()
    )
    nondeterministic = tuple(
        name
        for name, passed in (
            ("node_order", deterministic_node_order),
            ("edge_ids", deterministic_edge_ids),
            ("payload_order", deterministic_payload_order),
        )
        if not passed
    )
    return WorkflowDeterminismAudit(
        deterministic=not nondeterministic,
        deterministic_node_order=deterministic_node_order,
        deterministic_edge_ids=deterministic_edge_ids,
        deterministic_payload_order=deterministic_payload_order,
        nondeterministic_surfaces=nondeterministic,
        audit_notes=(
            "Node order is static and sourced from ASSISTANT_WORKFLOW_NODE_ORDER.",
            "Payload order is sourced from existing final-event payload keys.",
            "No live execution, provider routing, or Runtime Evolution is applied.",
        ),
    )


def profile_runtime_graph_performance(
    execution_graph: ExecutionGraphAnalysis | None = None,
) -> GraphPerformanceProfile:
    """Return static relative performance profile for graph visibility."""

    analysis = execution_graph or analyze_assistant_execution_graph()
    node_profiles = tuple(
        _node_profile(node_id, analysis) for node_id in analysis.node_order
    )
    return GraphPerformanceProfile(
        node_profiles=node_profiles,
        total_relative_cost_units=sum(
            profile.relative_cost_units for profile in node_profiles
        ),
        total_relative_latency_units=sum(
            profile.relative_latency_units for profile in node_profiles
        ),
        branch_count=analysis.branch_count,
        retry_edge_count=sum(
            1 for edge in analysis.edges if edge.edge_kind == "retry"
        ),
        failure_edge_count=analysis.failure_edge_count,
        critical_path_node_count=len(analysis.critical_path_node_ids),
    )


def profile_runtime_graph_cost(
    performance_profile: GraphPerformanceProfile | None = None,
) -> ExecutionCostProfile:
    """Project static relative cost posture without live metering."""

    profile = performance_profile or profile_runtime_graph_performance()
    highest = _highest_nodes(
        profile.node_profiles,
        key_name="relative_cost_units",
    )
    return ExecutionCostProfile(
        relative_total_cost_units=profile.total_relative_cost_units,
        highest_relative_cost_node_ids=highest,
    )


def profile_runtime_graph_latency(
    performance_profile: GraphPerformanceProfile | None = None,
) -> ExecutionLatencyProfile:
    """Project static relative latency posture without live timing."""

    profile = performance_profile or profile_runtime_graph_performance()
    highest = _highest_nodes(
        profile.node_profiles,
        key_name="relative_latency_units",
    )
    return ExecutionLatencyProfile(
        relative_total_latency_units=profile.total_relative_latency_units,
        highest_relative_latency_node_ids=highest,
    )


def normalize_runtime_workflow_state(
    plan: RuntimeGraphConsolidationPlan | None = None,
) -> WorkflowStateNormalizationReport:
    """Report canonical state key alignment without mutating state."""

    source_plan = plan or build_runtime_graph_consolidation_plan()
    canonical = _canonical_state_keys()
    runtime_missing = tuple(
        key for key in source_plan.runtime_payload_keys if key not in canonical
    )
    final_missing = tuple(
        key for key in source_plan.final_payload_keys if key not in canonical
    )
    return WorkflowStateNormalizationReport(
        canonical_state_keys=canonical,
        runtime_payload_keys=source_plan.runtime_payload_keys,
        final_payload_keys=source_plan.final_payload_keys,
        missing_runtime_payload_keys=runtime_missing,
        missing_final_payload_keys=final_missing,
        normalization_passed=not (runtime_missing or final_missing),
    )


def visualize_runtime_graph(
    execution_graph: ExecutionGraphAnalysis | None = None,
) -> ExecutionGraphVisualization:
    """Render the current runtime graph as Mermaid without running it."""

    analysis = execution_graph or analyze_assistant_execution_graph()
    lines = ["flowchart TD"]
    lines.append(f'  START["{START}"]')
    lines.append(f'  END["{END}"]')
    for node_id in analysis.node_order:
        lines.append(f'  {node_id}["{node_id}"]')
    for edge in analysis.edges:
        source = "START" if edge.source_node_id == str(START) else edge.source_node_id
        target = "END" if edge.target_node_id == str(END) else edge.target_node_id
        lines.append(f"  {source} -->|{edge.edge_kind}| {target}")
    return ExecutionGraphVisualization(
        diagram="\n".join(lines),
        node_count=analysis.node_count,
        edge_count=analysis.edge_count,
    )


def _node_contracts() -> tuple[RuntimeGraphNodeContract, ...]:
    specs = assistant_workflow_node_specs()
    handler_by_node = {spec.name: spec.handler for spec in specs}
    contracts: list[RuntimeGraphNodeContract] = []
    for index, node_id in enumerate(ASSISTANT_WORKFLOW_NODE_ORDER):
        state_inputs, state_outputs = _STATE_IO_BY_NODE[node_id]
        handler = handler_by_node[node_id]
        subgraphs = _OVERLAY_SUBGRAPHS_BY_NODE.get(
            node_id,
            (_PRIMARY_SUBGRAPH_BY_NODE[node_id],),
        )
        contracts.append(
            RuntimeGraphNodeContract(
                node_id=node_id,
                order_index=index,
                primary_subgraph_id=_PRIMARY_SUBGRAPH_BY_NODE[node_id],
                subgraph_ids=subgraphs,
                handler_reference=f"{handler.__module__}.{handler.__name__}",
                state_inputs=state_inputs,
                state_outputs=state_outputs,
                emitted_event_codes=_EVENT_CODES_BY_NODE[node_id],
            )
        )
    return tuple(contracts)


def _subgraph_contracts() -> tuple[RuntimeGraphSubgraphContract, ...]:
    node_contracts = _node_contracts()
    contracts: list[RuntimeGraphSubgraphContract] = []
    for subgraph_id in _SUBGRAPH_ORDER:
        members = tuple(
            contract.node_id
            for contract in node_contracts
            if subgraph_id in contract.subgraph_ids
        )
        inputs = _dedupe(
            key
            for contract in node_contracts
            if subgraph_id in contract.subgraph_ids
            for key in contract.state_inputs
        )
        outputs = _dedupe(
            key
            for contract in node_contracts
            if subgraph_id in contract.subgraph_ids
            for key in contract.state_outputs
        )
        contracts.append(
            RuntimeGraphSubgraphContract(
                subgraph_id=subgraph_id,
                roadmap_item=_SUBGRAPH_ROADMAP_ITEM[subgraph_id],
                node_ids=members,
                accepts_state_keys=inputs,
                emits_state_keys=outputs,
                boundary_summary=_subgraph_boundary_summary(subgraph_id, members),
                owner_module="creative_coding_assistant.orchestration.workflow_graph",
            )
        )
    return tuple(contracts)


def _decomposition_items() -> tuple[RuntimeGraphDecompositionItem, ...]:
    return (
        RuntimeGraphDecompositionItem(
            decomposition_id="langgraph_node_decomposition::workflow_foundation",
            roadmap_item="LangGraph Node Decomposition Plan",
            source_node_ids=(
                "intake",
                "routing",
                "memory",
                "retrieval",
                "context_assembly",
                "prompt_input",
                "prompt_rendering",
                "generation",
                "finalization",
            ),
            target_subgraph_id="workflow_foundation",
            extraction_strategy=(
                "Keep I/O, context, prompt rendering, generation, and "
                "final packaging in the live LangGraph adapter with explicit "
                "public topology contracts."
            ),
            validation_surface=("node_order", "event_sequence", "payload_keys"),
        ),
        RuntimeGraphDecompositionItem(
            decomposition_id="langgraph_node_decomposition::creative_cognition",
            roadmap_item="Creative Cognition Node Extraction",
            source_node_ids=("planning", "director", "reasoning"),
            target_subgraph_id="creative_cognition",
            extraction_strategy=(
                "Expose planning, director, and reasoning ownership as one "
                "cognitive subgraph while preserving existing emitted payloads."
            ),
            validation_surface=(
                "planning_payloads",
                "director_event",
                "reasoning_event",
            ),
        ),
        RuntimeGraphDecompositionItem(
            decomposition_id="langgraph_node_decomposition::generative_design",
            roadmap_item="Generative Design Node Extraction",
            source_node_ids=("planning",),
            target_subgraph_id="generative_design",
            extraction_strategy=(
                "Identify generative design derivations inside planning as a "
                "bounded extraction surface without adding LangGraph nodes."
            ),
            validation_surface=("generative_payloads", "state_normalization"),
        ),
        RuntimeGraphDecompositionItem(
            decomposition_id="langgraph_node_decomposition::artifact_intelligence",
            roadmap_item="Artifact Intelligence Node Extraction",
            source_node_ids=(
                "planning",
                "artifact_extraction",
                "preview_preparation",
                "artifact_critique",
            ),
            target_subgraph_id="artifact_intelligence",
            extraction_strategy=(
                "Separate artifact planning, extraction, preview, and critique "
                "contracts from the core workflow order."
            ),
            validation_surface=("artifact_payloads", "critique_events"),
        ),
        RuntimeGraphDecompositionItem(
            decomposition_id="langgraph_node_decomposition::creative_evaluation",
            roadmap_item="Creative Evaluation Node Extraction",
            source_node_ids=("planning", "artifact_critique", "review", "refinement"),
            target_subgraph_id="creative_evaluation",
            extraction_strategy=(
                "Make quality, self-evaluation, review, and bounded refinement "
                "contracts explicit while preserving retry behavior."
            ),
            validation_surface=(
                "review_edges",
                "refinement_retry",
                "evaluation_payloads",
            ),
        ),
        RuntimeGraphDecompositionItem(
            decomposition_id="langgraph_node_decomposition::failure_boundary",
            roadmap_item="Micro Error Path Design",
            source_node_ids=("failure",),
            target_subgraph_id="failure_boundary",
            extraction_strategy=(
                "Keep the terminal failure node and every conditional failure "
                "target explicit for local error-boundary review."
            ),
            validation_surface=("failure_edges", "terminal_nodes"),
        ),
    )


def _module_split() -> tuple[RuntimeGraphModuleSplit, ...]:
    return (
        RuntimeGraphModuleSplit(
            module_name="creative_coding_assistant.orchestration.workflow_graph",
            module_role="langgraph_adapter",
            responsibility=(
                "Owns live LangGraph construction, node handlers, transition "
                "selectors, and streaming integration."
            ),
            owns_live_execution=True,
        ),
        RuntimeGraphModuleSplit(
            module_name="creative_coding_assistant.orchestration.execution_graph_analyzer",
            module_role="static_topology_analysis",
            responsibility=(
                "Derives static node and edge topology without compiling or "
                "executing the graph."
            ),
        ),
        RuntimeGraphModuleSplit(
            module_name="creative_coding_assistant.orchestration.unified_execution_graph",
            module_role="cognitive_execution_projection",
            responsibility=(
                "Projects Cognitive OS execution metadata as read-only topology."
            ),
        ),
        RuntimeGraphModuleSplit(
            module_name=(
                "creative_coding_assistant.orchestration."
                "runtime_graph_consolidation"
            ),
            module_role="v7_consolidation_contracts",
            responsibility=(
                "Owns V7.1 graph contracts, validation, invariants, trace, "
                "explainability, diff, cost, latency, and visualization helpers."
            ),
        ),
        RuntimeGraphModuleSplit(
            module_name="creative_coding_assistant.orchestration.workflow",
            module_role="workflow_state_contracts",
            responsibility=(
                "Owns canonical AssistantWorkflowState fields and step order."
            ),
        ),
    )


def _compatibility_failures(plan: RuntimeGraphConsolidationPlan) -> tuple[str, ...]:
    failures: list[str] = []
    if plan.source_node_order != ASSISTANT_WORKFLOW_NODE_ORDER:
        failures.append("node_order_changed")
    if plan.recursion_limit != ASSISTANT_WORKFLOW_RECURSION_LIMIT:
        failures.append("recursion_limit_changed")
    if plan.final_payload_keys != assistant_workflow_final_payload_keys():
        failures.append("final_payload_order_changed")
    if plan.runtime_payload_keys != tuple(
        spec.payload_key for spec in assistant_workflow_model_payload_specs()
    ):
        failures.append("runtime_payload_order_changed")
    if any(
        (
            plan.behavior_change_implemented,
            plan.provider_model_routing_change_implemented,
            plan.generated_output_mutation_implemented,
            plan.storage_mutation_implemented,
            plan.uncontrolled_workflow_mutation_implemented,
            plan.runtime_evolution_implemented,
            plan.v7_2_scope_started,
        )
    ):
        failures.append("blocked_runtime_behavior_enabled")
    return tuple(failures)


def _canonical_state_keys() -> tuple[str, ...]:
    return tuple(AssistantWorkflowState.model_fields)


def _subgraph_boundary_summary(
    subgraph_id: RuntimeGraphSubgraphId,
    node_ids: tuple[str, ...],
) -> str:
    node_text = ", ".join(node_ids)
    summaries = {
        "workflow_foundation": (
            "Stable live workflow shell for request intake, routing, context, "
            "prompting, generation, and finalization."
        ),
        "creative_cognition": (
            "Creative planning, direction, and reasoning boundary with no "
            "provider/model routing changes."
        ),
        "generative_design": (
            "Generative structure, motif, composition, modality, and scene "
            "derivation boundary inside planning."
        ),
        "artifact_intelligence": (
            "Artifact planning, dependency, extraction, preview, and critique "
            "boundary."
        ),
        "creative_evaluation": (
            "Creative critique, self-evaluation, review, and bounded refinement "
            "boundary."
        ),
        "failure_boundary": (
            "Terminal failure contract and micro error path boundary."
        ),
    }
    return f"{summaries[subgraph_id]} Covered nodes: {node_text}."


def _edge_lookup(
    edges: tuple[ExecutionGraphEdge, ...],
) -> dict[tuple[str, str], str]:
    return {(edge.source_node_id, edge.target_node_id): edge.edge_id for edge in edges}


def _node_profile(
    node_id: str,
    analysis: ExecutionGraphAnalysis,
) -> RuntimeGraphNodeProfile:
    outgoing = tuple(edge for edge in analysis.edges if edge.source_node_id == node_id)
    return RuntimeGraphNodeProfile(
        node_id=node_id,
        relative_cost_units=_relative_cost_units(node_id),
        relative_latency_units=_relative_latency_units(node_id),
        outgoing_edge_count=len(outgoing),
        failure_edge_count=sum(1 for edge in outgoing if edge.edge_kind == "failure"),
        retry_edge_count=sum(1 for edge in outgoing if edge.edge_kind == "retry"),
    )


def _relative_cost_units(node_id: str) -> int:
    if node_id == "generation":
        return 5
    if node_id == "planning":
        return 4
    if node_id in {"retrieval", "artifact_critique", "review"}:
        return 2
    return 1


def _relative_latency_units(node_id: str) -> int:
    if node_id == "generation":
        return 8
    if node_id == "planning":
        return 5
    if node_id == "retrieval":
        return 3
    if node_id in {"artifact_critique", "review", "prompt_rendering"}:
        return 2
    return 1


def _highest_nodes(
    profiles: tuple[RuntimeGraphNodeProfile, ...],
    *,
    key_name: Literal["relative_cost_units", "relative_latency_units"],
) -> tuple[str, ...]:
    max_value = max(getattr(profile, key_name) for profile in profiles)
    return tuple(
        profile.node_id
        for profile in profiles
        if getattr(profile, key_name) == max_value
    )


def _dedupe(values: object) -> tuple[str, ...]:
    seen: Counter[str] = Counter()
    ordered: list[str] = []
    for value in values:
        key = str(value)
        if seen[key]:
            continue
        seen[key] += 1
        ordered.append(key)
    return tuple(ordered)
