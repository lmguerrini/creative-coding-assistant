"""V5.3 advisory workflow replay planning."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .execution_graph_analyzer import (
    ExecutionGraphAnalysis,
    analyze_assistant_execution_graph,
)
from .execution_profiling import ExecutionProfilingPlan, plan_execution_profiling
from .hybrid_studio import SessionReplayRegistry, session_replay_registry

WorkflowReplayKind = Literal[
    "topology_replay",
    "session_timeline_replay",
    "profiling_context_replay",
    "failure_path_replay",
    "storage_boundary_replay",
]
WorkflowReplayStatus = Literal[
    "replay_candidate",
    "failure_guardrail",
    "storage_guardrail",
]
WorkflowReplayPressure = Literal["low", "medium", "high", "guarded"]

WORKFLOW_REPLAY_CANDIDATE_SERIALIZATION_VERSION = "workflow_replay_candidate.v1"
WORKFLOW_REPLAY_PLAN_SERIALIZATION_VERSION = "workflow_replay_plan.v1"
WORKFLOW_REPLAY_AUTHORITY_BOUNDARY = (
    "Workflow replay planning derives advisory replay candidates from static "
    "workflow graph topology, passive session replay metadata, and advisory "
    "execution profiling metadata only; it does not replay runtime events, "
    "reconstruct live timelines, record sessions, capture snapshots, persist "
    "replay data, mutate workflow state, control workflow transitions, "
    "compile or execute workflow graphs, invoke agents or node handlers, "
    "route providers or models, trigger retries, mutate prompts, write "
    "storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "workflow_replay_execution",
    "runtime_event_replay",
    "timeline_reconstruction",
    "session_recording",
    "snapshot_capture",
    "execution_trace_reconstruction",
    "replay_persistence",
    "persistent_replay_storage",
    "workflow_state_mutation",
    "workflow_control",
    "workflow_graph_mutation",
    "langgraph_compilation",
    "workflow_execution",
    "agent_invocation",
    "node_handler_invocation",
    "provider_or_model_routing",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class WorkflowReplayCandidate(BaseModel):
    """One advisory V5.3 workflow replay candidate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    candidate_id: str = Field(min_length=1, max_length=180)
    replay_id: str = Field(min_length=1, max_length=120)
    replay_kind: WorkflowReplayKind
    status: WorkflowReplayStatus
    source_id: str = Field(min_length=1, max_length=180)
    source_serialization_version: str = Field(min_length=1, max_length=100)
    source_item_ids: tuple[str, ...] = Field(min_length=1, max_length=40)
    workflow_node_count: int = Field(ge=0, le=40)
    session_replay_profile_count: int = Field(ge=0, le=12)
    execution_profile_candidate_count: int = Field(ge=0, le=12)
    replay_context_count: int = Field(ge=0, le=200)
    advisory_replay_score: int = Field(ge=0, le=2_500)
    replay_pressure: WorkflowReplayPressure
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=22,
    )
    workflow_replay_planning_implemented: Literal[True] = True
    workflow_replay_execution_implemented: Literal[False] = False
    runtime_event_replay_implemented: Literal[False] = False
    timeline_reconstruction_implemented: Literal[False] = False
    session_recording_implemented: Literal[False] = False
    snapshot_capture_implemented: Literal[False] = False
    execution_trace_reconstruction_implemented: Literal[False] = False
    replay_persistence_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    workflow_state_mutation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["workflow_replay_candidate.v1"] = (
        WORKFLOW_REPLAY_CANDIDATE_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _candidate_matches_replay(self) -> Self:
        if self.candidate_id != f"workflow_replay::{self.replay_id}":
            raise ValueError("candidate_id must match replay_id")
        expected_score = _replay_score(
            status=self.status,
            workflow_node_count=self.workflow_node_count,
            session_replay_profile_count=self.session_replay_profile_count,
            execution_profile_candidate_count=(
                self.execution_profile_candidate_count
            ),
            replay_context_count=self.replay_context_count,
        )
        if self.advisory_replay_score != expected_score:
            raise ValueError("advisory_replay_score must match replay inputs")
        if self.replay_pressure != _replay_pressure(
            status=self.status,
            replay_score=self.advisory_replay_score,
        ):
            raise ValueError("replay_pressure must match replay inputs")
        if self.status == "replay_candidate" and self.advisory_replay_score <= 0:
            raise ValueError("replay candidates require advisory score")
        if self.status == "failure_guardrail" and self.workflow_node_count <= 0:
            raise ValueError("failure guardrails require workflow nodes")
        if self.status == "storage_guardrail" and self.replay_context_count != 0:
            raise ValueError("storage guardrails must not declare replay contexts")
        return self


class WorkflowReplayPlan(BaseModel):
    """Bounded V5.3 advisory workflow replay plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["workflow_replay_engine"] = "workflow_replay_engine"
    serialization_version: Literal["workflow_replay_plan.v1"] = (
        WORKFLOW_REPLAY_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=WORKFLOW_REPLAY_AUTHORITY_BOUNDARY,
        max_length=1700,
    )
    source_graph_serialization_version: str = Field(min_length=1, max_length=100)
    source_session_replay_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_execution_profiling_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_graph_node_count: int = Field(ge=1, le=40)
    source_session_replay_profile_count: int = Field(ge=1, le=12)
    source_execution_profile_candidate_count: int = Field(ge=1, le=12)
    failure_path_reachable: bool
    candidates: tuple[WorkflowReplayCandidate, ...] = Field(
        min_length=1,
        max_length=12,
    )
    candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    replay_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    failure_guardrail_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    storage_guardrail_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    candidate_count: int = Field(ge=1, le=12)
    replay_candidate_count: int = Field(ge=0, le=12)
    failure_guardrail_count: int = Field(ge=0, le=12)
    storage_guardrail_count: int = Field(ge=0, le=12)
    total_workflow_node_count: int = Field(ge=0, le=120)
    total_session_replay_profile_count: int = Field(ge=0, le=40)
    total_execution_profile_candidate_count: int = Field(ge=0, le=40)
    total_replay_context_count: int = Field(ge=0, le=200)
    highest_advisory_replay_score: int = Field(ge=0, le=2_500)
    total_advisory_replay_score: int = Field(ge=0, le=20_000)
    workflow_replay_pressure: WorkflowReplayPressure
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=22,
    )
    workflow_replay_planning_implemented: Literal[True] = True
    workflow_replay_execution_implemented: Literal[False] = False
    runtime_event_replay_implemented: Literal[False] = False
    timeline_reconstruction_implemented: Literal[False] = False
    session_recording_implemented: Literal[False] = False
    snapshot_capture_implemented: Literal[False] = False
    execution_trace_reconstruction_implemented: Literal[False] = False
    replay_persistence_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    workflow_state_mutation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_candidates(self) -> Self:
        derived_candidate_ids = tuple(
            candidate.candidate_id for candidate in self.candidates
        )
        if len(set(derived_candidate_ids)) != len(derived_candidate_ids):
            raise ValueError("candidate_ids must be unique")
        if self.candidate_ids != derived_candidate_ids:
            raise ValueError("candidate_ids must match candidates")
        if self.candidate_count != len(self.candidates):
            raise ValueError("candidate_count must match candidates")
        if self.replay_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "replay_candidate",
        ):
            raise ValueError("replay_candidate_ids must match candidates")
        if self.failure_guardrail_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "failure_guardrail",
        ):
            raise ValueError("failure_guardrail_candidate_ids must match candidates")
        if self.storage_guardrail_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "storage_guardrail",
        ):
            raise ValueError("storage_guardrail_candidate_ids must match candidates")
        if self.replay_candidate_count != len(self.replay_candidate_ids):
            raise ValueError("replay_candidate_count must match candidates")
        if self.failure_guardrail_count != len(self.failure_guardrail_candidate_ids):
            raise ValueError("failure_guardrail_count must match candidates")
        if self.storage_guardrail_count != len(self.storage_guardrail_candidate_ids):
            raise ValueError("storage_guardrail_count must match candidates")

        expected_node_count = sum(
            candidate.workflow_node_count for candidate in self.candidates
        )
        if self.total_workflow_node_count != expected_node_count:
            raise ValueError("total_workflow_node_count must match candidates")
        expected_session_count = sum(
            candidate.session_replay_profile_count for candidate in self.candidates
        )
        if self.total_session_replay_profile_count != expected_session_count:
            raise ValueError("total_session_replay_profile_count must match candidates")
        expected_profile_count = sum(
            candidate.execution_profile_candidate_count
            for candidate in self.candidates
        )
        if self.total_execution_profile_candidate_count != expected_profile_count:
            raise ValueError(
                "total_execution_profile_candidate_count must match candidates"
            )
        expected_context_count = sum(
            candidate.replay_context_count for candidate in self.candidates
        )
        if self.total_replay_context_count != expected_context_count:
            raise ValueError("total_replay_context_count must match candidates")
        expected_highest_score = max(
            candidate.advisory_replay_score for candidate in self.candidates
        )
        if self.highest_advisory_replay_score != expected_highest_score:
            raise ValueError("highest_advisory_replay_score must match candidates")
        expected_total_score = sum(
            candidate.advisory_replay_score for candidate in self.candidates
        )
        if self.total_advisory_replay_score != expected_total_score:
            raise ValueError("total_advisory_replay_score must match candidates")
        if self.workflow_replay_pressure != _plan_pressure(
            candidates=self.candidates,
            highest_score=self.highest_advisory_replay_score,
        ):
            raise ValueError("workflow_replay_pressure must match candidates")
        return self


def plan_workflow_replay(
    *,
    execution_graph: ExecutionGraphAnalysis | None = None,
    session_replay: SessionReplayRegistry | None = None,
    execution_profiling: ExecutionProfilingPlan | None = None,
) -> WorkflowReplayPlan:
    """Plan advisory workflow replay without replaying runtime events."""

    graph = execution_graph or analyze_assistant_execution_graph()
    session = session_replay or session_replay_registry()
    profiling = execution_profiling or plan_execution_profiling(
        execution_graph=graph
    )
    candidates = _candidates(graph=graph, session=session, profiling=profiling)
    highest_score = max(candidate.advisory_replay_score for candidate in candidates)

    return WorkflowReplayPlan(
        source_graph_serialization_version=graph.serialization_version,
        source_session_replay_serialization_version=session.serialization_version,
        source_execution_profiling_serialization_version=(
            profiling.serialization_version
        ),
        source_graph_node_count=graph.node_count,
        source_session_replay_profile_count=session.profile_count,
        source_execution_profile_candidate_count=profiling.candidate_count,
        failure_path_reachable=graph.failure_path_reachable,
        candidates=candidates,
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        replay_candidate_ids=_candidate_ids_for_status(
            candidates,
            "replay_candidate",
        ),
        failure_guardrail_candidate_ids=_candidate_ids_for_status(
            candidates,
            "failure_guardrail",
        ),
        storage_guardrail_candidate_ids=_candidate_ids_for_status(
            candidates,
            "storage_guardrail",
        ),
        candidate_count=len(candidates),
        replay_candidate_count=len(
            _candidate_ids_for_status(candidates, "replay_candidate")
        ),
        failure_guardrail_count=len(
            _candidate_ids_for_status(candidates, "failure_guardrail")
        ),
        storage_guardrail_count=len(
            _candidate_ids_for_status(candidates, "storage_guardrail")
        ),
        total_workflow_node_count=sum(
            candidate.workflow_node_count for candidate in candidates
        ),
        total_session_replay_profile_count=sum(
            candidate.session_replay_profile_count for candidate in candidates
        ),
        total_execution_profile_candidate_count=sum(
            candidate.execution_profile_candidate_count
            for candidate in candidates
        ),
        total_replay_context_count=sum(
            candidate.replay_context_count for candidate in candidates
        ),
        highest_advisory_replay_score=highest_score,
        total_advisory_replay_score=sum(
            candidate.advisory_replay_score for candidate in candidates
        ),
        workflow_replay_pressure=_plan_pressure(
            candidates=candidates,
            highest_score=highest_score,
        ),
        advisory_actions=_plan_actions(candidates),
    )


def workflow_replay_candidate_by_id(
    candidate_id: str,
    plan: WorkflowReplayPlan | None = None,
) -> WorkflowReplayCandidate | None:
    """Return one advisory workflow replay candidate without replaying."""

    source_plan = plan or plan_workflow_replay()
    for candidate in source_plan.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    return None


def workflow_replay_candidates_for_status(
    status: WorkflowReplayStatus,
    plan: WorkflowReplayPlan | None = None,
) -> tuple[WorkflowReplayCandidate, ...]:
    """Return workflow replay candidates by status without replay execution."""

    source_plan = plan or plan_workflow_replay()
    return tuple(
        candidate
        for candidate in source_plan.candidates
        if candidate.status == status
    )


def _candidates(
    *,
    graph: ExecutionGraphAnalysis,
    session: SessionReplayRegistry,
    profiling: ExecutionProfilingPlan,
) -> tuple[WorkflowReplayCandidate, ...]:
    return (
        _candidate(
            replay_id="topology_replay",
            kind="topology_replay",
            status="replay_candidate",
            source_id="execution_graph_analysis",
            source_serialization_version=graph.serialization_version,
            source_item_ids=graph.node_order,
            workflow_node_count=graph.node_count,
            session_replay_profile_count=0,
            execution_profile_candidate_count=0,
            replay_context_count=graph.edge_count,
            evidence=(
                f"workflow_nodes:{graph.node_count}",
                f"workflow_edges:{graph.edge_count}",
            ),
        ),
        _candidate(
            replay_id="session_timeline_replay",
            kind="session_timeline_replay",
            status="replay_candidate",
            source_id="session_replay_registry",
            source_serialization_version=session.serialization_version,
            source_item_ids=session.session_replay_profile_ids,
            workflow_node_count=0,
            session_replay_profile_count=session.profile_count,
            execution_profile_candidate_count=0,
            replay_context_count=len(session.replay_surface_refs),
            evidence=(
                f"session_replay_profiles:{session.profile_count}",
                f"replay_surfaces:{len(session.replay_surface_refs)}",
            ),
        ),
        _candidate(
            replay_id="profiling_context_replay",
            kind="profiling_context_replay",
            status="replay_candidate",
            source_id="execution_profiling_plan",
            source_serialization_version=profiling.serialization_version,
            source_item_ids=profiling.profile_candidate_ids,
            workflow_node_count=0,
            session_replay_profile_count=0,
            execution_profile_candidate_count=profiling.profile_candidate_count,
            replay_context_count=profiling.candidate_count,
            evidence=(
                f"profile_candidates:{profiling.profile_candidate_count}",
                f"profile_pressure:{profiling.execution_profile_pressure}",
            ),
        ),
        _candidate(
            replay_id="failure_path_replay",
            kind="failure_path_replay",
            status="failure_guardrail",
            source_id="execution_graph_analysis",
            source_serialization_version=graph.serialization_version,
            source_item_ids=graph.failure_entry_node_ids,
            workflow_node_count=len(graph.failure_entry_node_ids),
            session_replay_profile_count=0,
            execution_profile_candidate_count=0,
            replay_context_count=0,
            evidence=(
                f"failure_path_reachable:{graph.failure_path_reachable}",
                f"failure_edges:{graph.failure_edge_count}",
            ),
        ),
        _candidate(
            replay_id="storage_boundary_replay",
            kind="storage_boundary_replay",
            status="storage_guardrail",
            source_id="session_replay_registry",
            source_serialization_version=session.serialization_version,
            source_item_ids=session.source_registries,
            workflow_node_count=0,
            session_replay_profile_count=0,
            execution_profile_candidate_count=0,
            replay_context_count=0,
            evidence=(
                "replay_persistence:blocked",
                "persistent_replay_storage:blocked",
            ),
        ),
    )


def _candidate(
    *,
    replay_id: str,
    kind: WorkflowReplayKind,
    status: WorkflowReplayStatus,
    source_id: str,
    source_serialization_version: str,
    source_item_ids: tuple[str, ...],
    workflow_node_count: int,
    session_replay_profile_count: int,
    execution_profile_candidate_count: int,
    replay_context_count: int,
    evidence: tuple[str, ...],
) -> WorkflowReplayCandidate:
    score = _replay_score(
        status=status,
        workflow_node_count=workflow_node_count,
        session_replay_profile_count=session_replay_profile_count,
        execution_profile_candidate_count=execution_profile_candidate_count,
        replay_context_count=replay_context_count,
    )
    return WorkflowReplayCandidate(
        candidate_id=f"workflow_replay::{replay_id}",
        replay_id=replay_id,
        replay_kind=kind,
        status=status,
        source_id=source_id,
        source_serialization_version=source_serialization_version,
        source_item_ids=source_item_ids,
        workflow_node_count=workflow_node_count,
        session_replay_profile_count=session_replay_profile_count,
        execution_profile_candidate_count=execution_profile_candidate_count,
        replay_context_count=replay_context_count,
        advisory_replay_score=score,
        replay_pressure=_replay_pressure(status=status, replay_score=score),
        evidence=evidence,
        advisory_actions=_candidate_actions(status),
    )


def _candidate_ids_for_status(
    candidates: tuple[WorkflowReplayCandidate, ...],
    status: WorkflowReplayStatus,
) -> tuple[str, ...]:
    return tuple(
        candidate.candidate_id
        for candidate in candidates
        if candidate.status == status
    )


def _replay_score(
    *,
    status: WorkflowReplayStatus,
    workflow_node_count: int,
    session_replay_profile_count: int,
    execution_profile_candidate_count: int,
    replay_context_count: int,
) -> int:
    if status == "storage_guardrail":
        return 0
    score = (
        workflow_node_count * 35
        + session_replay_profile_count * 80
        + execution_profile_candidate_count * 60
        + replay_context_count * 25
    )
    if status == "failure_guardrail":
        score += 100
    return min(2_500, score)


def _replay_pressure(
    *,
    status: WorkflowReplayStatus,
    replay_score: int,
) -> WorkflowReplayPressure:
    if status in {"failure_guardrail", "storage_guardrail"}:
        return "guarded"
    if replay_score >= 800:
        return "high"
    if replay_score >= 300:
        return "medium"
    return "low"


def _plan_pressure(
    *,
    candidates: tuple[WorkflowReplayCandidate, ...],
    highest_score: int,
) -> WorkflowReplayPressure:
    if any(candidate.status != "replay_candidate" for candidate in candidates):
        return "guarded"
    if highest_score >= 800:
        return "high"
    if highest_score >= 300:
        return "medium"
    return "low"


def _candidate_actions(status: WorkflowReplayStatus) -> tuple[str, ...]:
    if status == "replay_candidate":
        return (
            "Expose workflow replay metadata for inspection only.",
            "Require explicit runtime authority before replay execution.",
        )
    if status == "failure_guardrail":
        return (
            "Preserve failure path visibility without replaying failure paths.",
        )
    return (
        "Keep replay persistence and replay storage disabled.",
    )


def _plan_actions(
    candidates: tuple[WorkflowReplayCandidate, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose workflow replay posture as advisory metadata only.",
        "Preserve replay, workflow, routing, storage, and output boundaries.",
    ]
    if _candidate_ids_for_status(candidates, "storage_guardrail"):
        actions.append("Keep workflow replay detached from persistent storage.")
    return tuple(actions)
