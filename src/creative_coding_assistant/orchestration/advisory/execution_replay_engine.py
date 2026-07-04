"""V5.3 advisory execution replay planning."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.execution_profiling import ExecutionProfilingPlan, plan_execution_profiling
from creative_coding_assistant.orchestration.hybrid_studio import ExecutionReplayRegistry, execution_replay_registry
from creative_coding_assistant.orchestration.workflow_replay_engine import WorkflowReplayPlan, plan_workflow_replay

ExecutionReplayCandidateKind = Literal[
    "execution_replay_context",
    "workflow_replay_context",
    "provider_selection_boundary",
    "cost_quality_boundary",
    "storage_boundary",
]
ExecutionReplayCandidateStatus = Literal[
    "replay_candidate",
    "provider_guardrail",
    "scoring_guardrail",
    "storage_guardrail",
]
ExecutionReplayPressure = Literal["low", "medium", "high", "guarded"]

EXECUTION_REPLAY_CANDIDATE_SERIALIZATION_VERSION = "execution_replay_candidate.v1"
EXECUTION_REPLAY_PLAN_SERIALIZATION_VERSION = "execution_replay_plan.v1"
EXECUTION_REPLAY_ENGINE_AUTHORITY_BOUNDARY = (
    "Execution replay planning derives advisory execution replay candidates "
    "from passive Studio execution replay metadata, advisory workflow replay "
    "metadata, and advisory execution profiling metadata only; it does not "
    "execute providers, reconstruct execution traces, replay runtime events, "
    "persist replay data, select providers or models, route providers or "
    "models, calculate cost or quality scores, control workflows, request "
    "human input, trigger retries, mutate prompts, write storage, or modify "
    "generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "execution_replay_execution",
    "provider_execution",
    "local_provider_execution",
    "cloud_provider_execution",
    "model_selection",
    "provider_or_model_routing",
    "execution_trace_reconstruction",
    "runtime_event_replay",
    "workflow_replay_execution",
    "replay_persistence",
    "persistent_replay_storage",
    "cost_scoring",
    "quality_scoring",
    "quality_evaluation",
    "workflow_control",
    "human_input_request",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class ExecutionReplayCandidate(BaseModel):
    """One advisory V5.3 execution replay candidate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    candidate_id: str = Field(min_length=1, max_length=180)
    replay_id: str = Field(min_length=1, max_length=120)
    replay_kind: ExecutionReplayCandidateKind
    status: ExecutionReplayCandidateStatus
    source_id: str = Field(min_length=1, max_length=180)
    source_serialization_version: str = Field(min_length=1, max_length=100)
    source_item_ids: tuple[str, ...] = Field(min_length=1, max_length=40)
    execution_replay_profile_count: int = Field(ge=0, le=40)
    workflow_replay_candidate_count: int = Field(ge=0, le=40)
    execution_profile_candidate_count: int = Field(ge=0, le=40)
    route_count: int = Field(ge=0, le=6)
    replay_context_count: int = Field(ge=0, le=200)
    advisory_replay_score: int = Field(ge=0, le=2_500)
    replay_pressure: ExecutionReplayPressure
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=22,
    )
    execution_replay_planning_implemented: Literal[True] = True
    execution_replay_execution_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    local_provider_execution_implemented: Literal[False] = False
    cloud_provider_execution_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    execution_trace_reconstruction_implemented: Literal[False] = False
    runtime_event_replay_implemented: Literal[False] = False
    workflow_replay_execution_implemented: Literal[False] = False
    replay_persistence_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    cost_scoring_implemented: Literal[False] = False
    quality_scoring_implemented: Literal[False] = False
    quality_evaluation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["execution_replay_candidate.v1"] = (
        EXECUTION_REPLAY_CANDIDATE_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _candidate_matches_replay(self) -> Self:
        if self.candidate_id != f"execution_replay::{self.replay_id}":
            raise ValueError("candidate_id must match replay_id")
        expected_score = _replay_score(
            status=self.status,
            execution_replay_profile_count=self.execution_replay_profile_count,
            workflow_replay_candidate_count=self.workflow_replay_candidate_count,
            execution_profile_candidate_count=(self.execution_profile_candidate_count),
            route_count=self.route_count,
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
        if self.status == "storage_guardrail" and self.replay_context_count != 0:
            raise ValueError("storage guardrails must not declare replay contexts")
        return self


class ExecutionReplayPlan(BaseModel):
    """Bounded V5.3 advisory execution replay plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["execution_replay_engine"] = "execution_replay_engine"
    serialization_version: Literal["execution_replay_plan.v1"] = (
        EXECUTION_REPLAY_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=EXECUTION_REPLAY_ENGINE_AUTHORITY_BOUNDARY,
        max_length=1700,
    )
    source_execution_replay_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_workflow_replay_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_execution_profiling_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_execution_replay_profile_count: int = Field(ge=1, le=40)
    source_workflow_replay_candidate_count: int = Field(ge=1, le=40)
    source_execution_profile_candidate_count: int = Field(ge=1, le=40)
    source_route_count: int = Field(ge=1, le=6)
    candidates: tuple[ExecutionReplayCandidate, ...] = Field(
        min_length=1,
        max_length=12,
    )
    candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    replay_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    provider_guardrail_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    scoring_guardrail_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    storage_guardrail_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    candidate_count: int = Field(ge=1, le=12)
    replay_candidate_count: int = Field(ge=0, le=12)
    provider_guardrail_count: int = Field(ge=0, le=12)
    scoring_guardrail_count: int = Field(ge=0, le=12)
    storage_guardrail_count: int = Field(ge=0, le=12)
    total_execution_replay_profile_count: int = Field(ge=0, le=80)
    total_workflow_replay_candidate_count: int = Field(ge=0, le=80)
    total_execution_profile_candidate_count: int = Field(ge=0, le=80)
    total_route_count: int = Field(ge=0, le=24)
    total_replay_context_count: int = Field(ge=0, le=400)
    highest_advisory_replay_score: int = Field(ge=0, le=2_500)
    total_advisory_replay_score: int = Field(ge=0, le=20_000)
    execution_replay_pressure: ExecutionReplayPressure
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=22,
    )
    execution_replay_planning_implemented: Literal[True] = True
    execution_replay_execution_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    local_provider_execution_implemented: Literal[False] = False
    cloud_provider_execution_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    execution_trace_reconstruction_implemented: Literal[False] = False
    runtime_event_replay_implemented: Literal[False] = False
    workflow_replay_execution_implemented: Literal[False] = False
    replay_persistence_implemented: Literal[False] = False
    persistent_replay_storage_implemented: Literal[False] = False
    cost_scoring_implemented: Literal[False] = False
    quality_scoring_implemented: Literal[False] = False
    quality_evaluation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
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
        if self.provider_guardrail_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "provider_guardrail",
        ):
            raise ValueError("provider_guardrail_candidate_ids must match candidates")
        if self.scoring_guardrail_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "scoring_guardrail",
        ):
            raise ValueError("scoring_guardrail_candidate_ids must match candidates")
        if self.storage_guardrail_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "storage_guardrail",
        ):
            raise ValueError("storage_guardrail_candidate_ids must match candidates")
        if self.replay_candidate_count != len(self.replay_candidate_ids):
            raise ValueError("replay_candidate_count must match candidates")
        if self.provider_guardrail_count != len(self.provider_guardrail_candidate_ids):
            raise ValueError("provider_guardrail_count must match candidates")
        if self.scoring_guardrail_count != len(self.scoring_guardrail_candidate_ids):
            raise ValueError("scoring_guardrail_count must match candidates")
        if self.storage_guardrail_count != len(self.storage_guardrail_candidate_ids):
            raise ValueError("storage_guardrail_count must match candidates")

        expected_execution_count = sum(
            candidate.execution_replay_profile_count for candidate in self.candidates
        )
        if self.total_execution_replay_profile_count != expected_execution_count:
            raise ValueError(
                "total_execution_replay_profile_count must match candidates"
            )
        expected_workflow_count = sum(
            candidate.workflow_replay_candidate_count for candidate in self.candidates
        )
        if self.total_workflow_replay_candidate_count != expected_workflow_count:
            raise ValueError(
                "total_workflow_replay_candidate_count must match candidates"
            )
        expected_profile_count = sum(
            candidate.execution_profile_candidate_count for candidate in self.candidates
        )
        if self.total_execution_profile_candidate_count != expected_profile_count:
            raise ValueError(
                "total_execution_profile_candidate_count must match candidates"
            )
        expected_routes = sum(candidate.route_count for candidate in self.candidates)
        if self.total_route_count != expected_routes:
            raise ValueError("total_route_count must match candidates")
        expected_contexts = sum(
            candidate.replay_context_count for candidate in self.candidates
        )
        if self.total_replay_context_count != expected_contexts:
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
        if self.execution_replay_pressure != _plan_pressure(
            candidates=self.candidates,
            highest_score=self.highest_advisory_replay_score,
        ):
            raise ValueError("execution_replay_pressure must match candidates")
        return self


def plan_execution_replay(
    *,
    execution_replay: ExecutionReplayRegistry | None = None,
    workflow_replay: WorkflowReplayPlan | None = None,
    execution_profiling: ExecutionProfilingPlan | None = None,
) -> ExecutionReplayPlan:
    """Plan advisory execution replay without replaying runtime events."""

    execution_registry = execution_replay or execution_replay_registry()
    workflow = workflow_replay or plan_workflow_replay()
    profiling = execution_profiling or plan_execution_profiling()
    candidates = _candidates(
        execution_registry=execution_registry,
        workflow=workflow,
        profiling=profiling,
    )
    highest_score = max(candidate.advisory_replay_score for candidate in candidates)

    return ExecutionReplayPlan(
        source_execution_replay_serialization_version=(
            execution_registry.serialization_version
        ),
        source_workflow_replay_serialization_version=workflow.serialization_version,
        source_execution_profiling_serialization_version=(
            profiling.serialization_version
        ),
        source_execution_replay_profile_count=execution_registry.profile_count,
        source_workflow_replay_candidate_count=workflow.candidate_count,
        source_execution_profile_candidate_count=profiling.candidate_count,
        source_route_count=len(execution_registry.route_names),
        candidates=candidates,
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        replay_candidate_ids=_candidate_ids_for_status(
            candidates,
            "replay_candidate",
        ),
        provider_guardrail_candidate_ids=_candidate_ids_for_status(
            candidates,
            "provider_guardrail",
        ),
        scoring_guardrail_candidate_ids=_candidate_ids_for_status(
            candidates,
            "scoring_guardrail",
        ),
        storage_guardrail_candidate_ids=_candidate_ids_for_status(
            candidates,
            "storage_guardrail",
        ),
        candidate_count=len(candidates),
        replay_candidate_count=len(
            _candidate_ids_for_status(candidates, "replay_candidate")
        ),
        provider_guardrail_count=len(
            _candidate_ids_for_status(candidates, "provider_guardrail")
        ),
        scoring_guardrail_count=len(
            _candidate_ids_for_status(candidates, "scoring_guardrail")
        ),
        storage_guardrail_count=len(
            _candidate_ids_for_status(candidates, "storage_guardrail")
        ),
        total_execution_replay_profile_count=sum(
            candidate.execution_replay_profile_count for candidate in candidates
        ),
        total_workflow_replay_candidate_count=sum(
            candidate.workflow_replay_candidate_count for candidate in candidates
        ),
        total_execution_profile_candidate_count=sum(
            candidate.execution_profile_candidate_count for candidate in candidates
        ),
        total_route_count=sum(candidate.route_count for candidate in candidates),
        total_replay_context_count=sum(
            candidate.replay_context_count for candidate in candidates
        ),
        highest_advisory_replay_score=highest_score,
        total_advisory_replay_score=sum(
            candidate.advisory_replay_score for candidate in candidates
        ),
        execution_replay_pressure=_plan_pressure(
            candidates=candidates,
            highest_score=highest_score,
        ),
        advisory_actions=_plan_actions(candidates),
    )


def execution_replay_candidate_by_id(
    candidate_id: str,
    plan: ExecutionReplayPlan | None = None,
) -> ExecutionReplayCandidate | None:
    """Return one advisory execution replay candidate without replaying."""

    source_plan = plan or plan_execution_replay()
    for candidate in source_plan.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    return None


def execution_replay_candidates_for_status(
    status: ExecutionReplayCandidateStatus,
    plan: ExecutionReplayPlan | None = None,
) -> tuple[ExecutionReplayCandidate, ...]:
    """Return execution replay candidates by status without replay execution."""

    source_plan = plan or plan_execution_replay()
    return tuple(
        candidate for candidate in source_plan.candidates if candidate.status == status
    )


def _candidates(
    *,
    execution_registry: ExecutionReplayRegistry,
    workflow: WorkflowReplayPlan,
    profiling: ExecutionProfilingPlan,
) -> tuple[ExecutionReplayCandidate, ...]:
    return (
        _candidate(
            replay_id="execution_replay_context",
            kind="execution_replay_context",
            status="replay_candidate",
            source_id="execution_replay_registry",
            source_serialization_version=execution_registry.serialization_version,
            source_item_ids=execution_registry.execution_replay_profile_ids,
            execution_replay_profile_count=execution_registry.profile_count,
            workflow_replay_candidate_count=0,
            execution_profile_candidate_count=0,
            route_count=len(execution_registry.route_names),
            replay_context_count=len(execution_registry.execution_replay_surface_refs),
            evidence=(
                f"execution_replay_profiles:{execution_registry.profile_count}",
                f"execution_replay_routes:{len(execution_registry.route_names)}",
            ),
        ),
        _candidate(
            replay_id="workflow_replay_context",
            kind="workflow_replay_context",
            status="replay_candidate",
            source_id="workflow_replay_plan",
            source_serialization_version=workflow.serialization_version,
            source_item_ids=workflow.replay_candidate_ids,
            execution_replay_profile_count=0,
            workflow_replay_candidate_count=workflow.replay_candidate_count,
            execution_profile_candidate_count=profiling.profile_candidate_count,
            route_count=0,
            replay_context_count=workflow.total_replay_context_count,
            evidence=(
                f"workflow_replay_candidates:{workflow.replay_candidate_count}",
                f"workflow_replay_pressure:{workflow.workflow_replay_pressure}",
            ),
        ),
        _candidate(
            replay_id="provider_selection_boundary",
            kind="provider_selection_boundary",
            status="provider_guardrail",
            source_id="execution_replay_registry",
            source_serialization_version=execution_registry.serialization_version,
            source_item_ids=execution_registry.provider_selection_profile_ids,
            execution_replay_profile_count=0,
            workflow_replay_candidate_count=0,
            execution_profile_candidate_count=0,
            route_count=0,
            replay_context_count=len(execution_registry.provider_selection_profile_ids),
            evidence=(
                "provider_execution:blocked",
                "provider_model_routing:blocked",
            ),
        ),
        _candidate(
            replay_id="cost_quality_boundary",
            kind="cost_quality_boundary",
            status="scoring_guardrail",
            source_id="execution_replay_registry",
            source_serialization_version=execution_registry.serialization_version,
            source_item_ids=(
                execution_registry.cost_profile_ids
                + execution_registry.quality_profile_ids
            ),
            execution_replay_profile_count=0,
            workflow_replay_candidate_count=0,
            execution_profile_candidate_count=0,
            route_count=0,
            replay_context_count=(
                len(execution_registry.cost_profile_ids)
                + len(execution_registry.quality_profile_ids)
            ),
            evidence=(
                "cost_scoring:blocked",
                "quality_evaluation:blocked",
            ),
        ),
        _candidate(
            replay_id="storage_boundary",
            kind="storage_boundary",
            status="storage_guardrail",
            source_id="workflow_replay_plan",
            source_serialization_version=workflow.serialization_version,
            source_item_ids=workflow.storage_guardrail_candidate_ids,
            execution_replay_profile_count=0,
            workflow_replay_candidate_count=0,
            execution_profile_candidate_count=0,
            route_count=0,
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
    kind: ExecutionReplayCandidateKind,
    status: ExecutionReplayCandidateStatus,
    source_id: str,
    source_serialization_version: str,
    source_item_ids: tuple[str, ...],
    execution_replay_profile_count: int,
    workflow_replay_candidate_count: int,
    execution_profile_candidate_count: int,
    route_count: int,
    replay_context_count: int,
    evidence: tuple[str, ...],
) -> ExecutionReplayCandidate:
    score = _replay_score(
        status=status,
        execution_replay_profile_count=execution_replay_profile_count,
        workflow_replay_candidate_count=workflow_replay_candidate_count,
        execution_profile_candidate_count=execution_profile_candidate_count,
        route_count=route_count,
        replay_context_count=replay_context_count,
    )
    return ExecutionReplayCandidate(
        candidate_id=f"execution_replay::{replay_id}",
        replay_id=replay_id,
        replay_kind=kind,
        status=status,
        source_id=source_id,
        source_serialization_version=source_serialization_version,
        source_item_ids=source_item_ids,
        execution_replay_profile_count=execution_replay_profile_count,
        workflow_replay_candidate_count=workflow_replay_candidate_count,
        execution_profile_candidate_count=execution_profile_candidate_count,
        route_count=route_count,
        replay_context_count=replay_context_count,
        advisory_replay_score=score,
        replay_pressure=_replay_pressure(status=status, replay_score=score),
        evidence=evidence,
        advisory_actions=_candidate_actions(status),
    )


def _candidate_ids_for_status(
    candidates: tuple[ExecutionReplayCandidate, ...],
    status: ExecutionReplayCandidateStatus,
) -> tuple[str, ...]:
    return tuple(
        candidate.candidate_id for candidate in candidates if candidate.status == status
    )


def _replay_score(
    *,
    status: ExecutionReplayCandidateStatus,
    execution_replay_profile_count: int,
    workflow_replay_candidate_count: int,
    execution_profile_candidate_count: int,
    route_count: int,
    replay_context_count: int,
) -> int:
    if status == "storage_guardrail":
        return 0
    score = (
        execution_replay_profile_count * 90
        + workflow_replay_candidate_count * 70
        + execution_profile_candidate_count * 60
        + route_count * 50
        + replay_context_count * 25
    )
    if status in {"provider_guardrail", "scoring_guardrail"}:
        score += 100
    return min(2_500, score)


def _replay_pressure(
    *,
    status: ExecutionReplayCandidateStatus,
    replay_score: int,
) -> ExecutionReplayPressure:
    if status != "replay_candidate":
        return "guarded"
    if replay_score >= 800:
        return "high"
    if replay_score >= 300:
        return "medium"
    return "low"


def _plan_pressure(
    *,
    candidates: tuple[ExecutionReplayCandidate, ...],
    highest_score: int,
) -> ExecutionReplayPressure:
    if any(candidate.status != "replay_candidate" for candidate in candidates):
        return "guarded"
    if highest_score >= 800:
        return "high"
    if highest_score >= 300:
        return "medium"
    return "low"


def _candidate_actions(
    status: ExecutionReplayCandidateStatus,
) -> tuple[str, ...]:
    if status == "replay_candidate":
        return (
            "Expose execution replay metadata for inspection only.",
            "Require explicit runtime authority before replay execution.",
        )
    if status == "provider_guardrail":
        return ("Preserve provider execution and routing boundaries.",)
    if status == "scoring_guardrail":
        return ("Keep cost scoring, quality scoring, and evaluation disabled.",)
    return ("Keep replay persistence and replay storage disabled.",)


def _plan_actions(
    candidates: tuple[ExecutionReplayCandidate, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose execution replay posture as advisory metadata only.",
        "Preserve replay, provider, scoring, workflow, storage, and output boundaries.",
    ]
    if _candidate_ids_for_status(candidates, "provider_guardrail"):
        actions.append("Keep execution replay detached from provider selection.")
    return tuple(actions)
