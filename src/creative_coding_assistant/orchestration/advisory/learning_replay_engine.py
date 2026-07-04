"""V6.1 advisory learning replay metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.adaptive_learning_engine import (
    AdaptiveLearningPlan,
    AdaptiveLearningSignal,
    evaluate_adaptive_learning_engine,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

LearningReplayScenarioKind = Literal[
    "workflow_pattern_replay",
    "strategy_pattern_replay",
    "routing_boundary_replay",
    "governance_feedback_replay",
    "runtime_guardrail_replay",
]
LearningReplayConfidence = Literal["high", "moderate", "low", "guarded"]
LearningReplayPosture = Literal["candidate", "review_required", "guarded"]

LEARNING_REPLAY_SCENARIO_SERIALIZATION_VERSION = "learning_replay_scenario.v1"
LEARNING_REPLAY_PLAN_SERIALIZATION_VERSION = "learning_replay_plan.v1"
LEARNING_REPLAY_AUTHORITY_BOUNDARY = (
    "V6.1 learning replay engine records replay scenario metadata for "
    "adaptive learning signals only; it does not execute learning replay, "
    "execute workflow replay, call providers, control workflows, write "
    "storage, mutate generated output, or apply Runtime Evolution."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "learning_replay_execution",
    "workflow_replay_execution",
    "provider_execution",
    "workflow_control",
    "workflow_graph_mutation",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)

_SCENARIO_MAP: tuple[
    tuple[LearningReplayScenarioKind, str, int],
    ...,
] = (
    ("workflow_pattern_replay", "adaptive_learning::workflow_pattern_learning", 220),
    ("strategy_pattern_replay", "adaptive_learning::strategy_pattern_learning", 180),
    ("routing_boundary_replay", "adaptive_learning::routing_boundary_learning", 240),
    (
        "governance_feedback_replay",
        "adaptive_learning::governance_feedback_learning",
        200,
    ),
    ("runtime_guardrail_replay", "adaptive_learning::runtime_guardrail_learning", 230),
)


class LearningReplayScenario(BaseModel):
    """One advisory replay scenario for a learning signal."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    scenario_id: str = Field(min_length=1, max_length=180)
    scenario_kind: LearningReplayScenarioKind
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    source_learning_signal_id: str = Field(min_length=1, max_length=180)
    source_learning_signal_kind: str = Field(min_length=1, max_length=120)
    source_learning_status: str = Field(min_length=1, max_length=80)
    learning_priority_score: int = Field(ge=0, le=1_000)
    replay_weight: int = Field(ge=0, le=260)
    replay_confidence_score: int = Field(ge=0, le=1_000)
    replay_confidence: LearningReplayConfidence
    expected_replay_insight: str = Field(min_length=1, max_length=360)
    replay_safety_boundary: str = Field(min_length=1, max_length=420)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=10,
    )
    learning_replay_engine_implemented: Literal[True] = True
    replay_metadata_implemented: Literal[True] = True
    adaptive_learning_metadata_used: Literal[True] = True
    replay_executed: Literal[False] = False
    workflow_replay_executed: Literal[False] = False
    provider_calls_executed: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["learning_replay_scenario.v1"] = (
        LEARNING_REPLAY_SCENARIO_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _scenario_matches_contract(self) -> Self:
        if self.scenario_id != f"learning_replay::{self.scenario_kind}":
            raise ValueError("scenario_id must match scenario_kind")
        if self.replay_confidence_score != _replay_confidence_score(
            learning_priority_score=self.learning_priority_score,
            replay_weight=self.replay_weight,
        ):
            raise ValueError("replay_confidence_score must combine source score")
        if self.replay_confidence != _replay_confidence(
            self.replay_confidence_score,
            self.source_learning_status,
        ):
            raise ValueError("replay_confidence must match score and status")
        return self


class LearningReplayPlan(BaseModel):
    """Bounded V6.1 advisory learning replay plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["learning_replay_engine"] = "learning_replay_engine"
    serialization_version: Literal["learning_replay_plan.v1"] = (
        LEARNING_REPLAY_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=LEARNING_REPLAY_AUTHORITY_BOUNDARY,
        max_length=1200,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_adaptive_learning_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    scenarios: tuple[LearningReplayScenario, ...] = Field(min_length=5, max_length=5)
    scenario_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    high_confidence_scenario_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    moderate_confidence_scenario_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    low_confidence_scenario_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    guarded_confidence_scenario_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    scenario_count: int = Field(ge=5, le=5)
    highest_replay_confidence_score: int = Field(ge=0, le=1_000)
    overall_replay_confidence_score: int = Field(ge=0, le=1_000)
    overall_replay_posture: LearningReplayPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=10,
    )
    learning_replay_engine_implemented: Literal[True] = True
    replay_metadata_implemented: Literal[True] = True
    adaptive_learning_metadata_used: Literal[True] = True
    replay_executed: Literal[False] = False
    workflow_replay_executed: Literal[False] = False
    provider_calls_executed: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_scenarios(self) -> Self:
        derived_ids = tuple(scenario.scenario_id for scenario in self.scenarios)
        if len(set(derived_ids)) != len(derived_ids):
            raise ValueError("scenario_ids must be unique")
        if self.scenario_ids != derived_ids:
            raise ValueError("scenario_ids must match scenarios")
        if self.scenario_count != len(self.scenarios):
            raise ValueError("scenario_count must match scenarios")
        if self.high_confidence_scenario_ids != _scenario_ids_for_confidence(
            self.scenarios,
            "high",
        ):
            raise ValueError("high_confidence_scenario_ids must match scenarios")
        if self.moderate_confidence_scenario_ids != _scenario_ids_for_confidence(
            self.scenarios,
            "moderate",
        ):
            raise ValueError("moderate_confidence_scenario_ids must match scenarios")
        if self.low_confidence_scenario_ids != _scenario_ids_for_confidence(
            self.scenarios,
            "low",
        ):
            raise ValueError("low_confidence_scenario_ids must match scenarios")
        if self.guarded_confidence_scenario_ids != _scenario_ids_for_confidence(
            self.scenarios,
            "guarded",
        ):
            raise ValueError("guarded_confidence_scenario_ids must match scenarios")
        if self.highest_replay_confidence_score != max(
            scenario.replay_confidence_score for scenario in self.scenarios
        ):
            raise ValueError("highest_replay_confidence_score must match scenarios")
        if self.overall_replay_confidence_score != _overall_replay_confidence_score(
            self.scenarios,
        ):
            raise ValueError("overall_replay_confidence_score must match scenarios")
        if self.overall_replay_posture != _overall_replay_posture(self.scenarios):
            raise ValueError("overall_replay_posture must match scenarios")
        for scenario in self.scenarios:
            if scenario.route_name != self.route_name:
                raise ValueError("scenario route_name must match plan")
            if scenario.task_type != self.task_type:
                raise ValueError("scenario task_type must match plan")
        return self


def build_learning_replay_engine(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    adaptive_learning: AdaptiveLearningPlan | None = None,
) -> LearningReplayPlan:
    """Build learning replay scenarios without executing replay."""

    route_name = _resolve_route(route)
    learning_plan = adaptive_learning or evaluate_adaptive_learning_engine(
        route=route_name,
        task_type=str(task_type).strip(),
        execution_mode_id=execution_mode_id,
    )
    normalized_mode = str(
        execution_mode_id or learning_plan.signals[0].execution_mode_id
    )
    execution_modes = routing_execution_mode_registry()
    if normalized_mode not in execution_modes.execution_mode_ids:
        raise ValueError("execution_mode_id must be a known execution mode")
    scenarios = _scenarios(
        route_name=route_name,
        task_type=learning_plan.task_type,
        execution_mode_id=normalized_mode,  # type: ignore[arg-type]
        adaptive_learning=learning_plan,
    )
    return LearningReplayPlan(
        route_name=route_name,
        task_type=learning_plan.task_type,
        source_adaptive_learning_serialization_version=(
            learning_plan.serialization_version
        ),
        execution_mode_ids=execution_modes.execution_mode_ids,
        scenarios=scenarios,
        scenario_ids=tuple(scenario.scenario_id for scenario in scenarios),
        high_confidence_scenario_ids=_scenario_ids_for_confidence(scenarios, "high"),
        moderate_confidence_scenario_ids=_scenario_ids_for_confidence(
            scenarios,
            "moderate",
        ),
        low_confidence_scenario_ids=_scenario_ids_for_confidence(scenarios, "low"),
        guarded_confidence_scenario_ids=_scenario_ids_for_confidence(
            scenarios,
            "guarded",
        ),
        scenario_count=len(scenarios),
        highest_replay_confidence_score=max(
            scenario.replay_confidence_score for scenario in scenarios
        ),
        overall_replay_confidence_score=_overall_replay_confidence_score(scenarios),
        overall_replay_posture=_overall_replay_posture(scenarios),
        advisory_actions=_plan_actions(scenarios),
    )


def learning_replay_scenario_by_id(
    scenario_id: str,
    plan: LearningReplayPlan | None = None,
) -> LearningReplayScenario | None:
    """Return one learning replay scenario without executing replay."""

    source_plan = plan or build_learning_replay_engine()
    normalized_id = str(scenario_id).strip()
    for scenario in source_plan.scenarios:
        if scenario.scenario_id == normalized_id:
            return scenario
    return None


def learning_replay_scenarios_for_confidence(
    replay_confidence: LearningReplayConfidence,
    plan: LearningReplayPlan | None = None,
) -> tuple[LearningReplayScenario, ...]:
    """Return learning replay scenarios by advisory confidence."""

    source_plan = plan or build_learning_replay_engine()
    return tuple(
        scenario
        for scenario in source_plan.scenarios
        if scenario.replay_confidence == replay_confidence
    )


def _scenarios(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    adaptive_learning: AdaptiveLearningPlan,
) -> tuple[LearningReplayScenario, ...]:
    return tuple(
        _scenario(
            kind=kind,
            signal=_required_learning_signal(signal_id, adaptive_learning),
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            replay_weight=weight,
        )
        for kind, signal_id, weight in _SCENARIO_MAP
    )


def _scenario(
    *,
    kind: LearningReplayScenarioKind,
    signal: AdaptiveLearningSignal,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    replay_weight: int,
) -> LearningReplayScenario:
    confidence_score = _replay_confidence_score(
        learning_priority_score=signal.learning_priority_score,
        replay_weight=replay_weight,
    )
    confidence = _replay_confidence(confidence_score, signal.status)
    return LearningReplayScenario(
        scenario_id=f"learning_replay::{kind}",
        scenario_kind=kind,
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        source_learning_signal_id=signal.signal_id,
        source_learning_signal_kind=signal.signal_kind,
        source_learning_status=signal.status,
        learning_priority_score=signal.learning_priority_score,
        replay_weight=replay_weight,
        replay_confidence_score=confidence_score,
        replay_confidence=confidence,
        expected_replay_insight=_expected_replay_insight(kind, confidence),
        replay_safety_boundary=_replay_safety_boundary(kind),
        advisory_actions=_scenario_actions(kind),
        evidence=(
            f"learning_signal:{signal.signal_id}",
            f"learning_priority_score:{signal.learning_priority_score}",
            f"learning_status:{signal.status}",
            f"replay_confidence:{confidence}",
        ),
    )


def _required_learning_signal(
    signal_id: str,
    plan: AdaptiveLearningPlan,
) -> AdaptiveLearningSignal:
    for signal in plan.signals:
        if signal.signal_id == signal_id:
            return signal
    raise ValueError("required learning signal metadata is missing")


def _replay_confidence_score(
    *,
    learning_priority_score: int,
    replay_weight: int,
) -> int:
    return min(
        1_000,
        max(0, learning_priority_score // 2 + replay_weight),
    )


def _replay_confidence(
    score: int,
    source_status: str,
) -> LearningReplayConfidence:
    if source_status == "guardrail":
        return "guarded"
    if score >= 700:
        return "high"
    if score >= 500:
        return "moderate"
    return "low"


def _scenario_ids_for_confidence(
    scenarios: tuple[LearningReplayScenario, ...],
    replay_confidence: LearningReplayConfidence,
) -> tuple[str, ...]:
    return tuple(
        scenario.scenario_id
        for scenario in scenarios
        if scenario.replay_confidence == replay_confidence
    )


def _overall_replay_confidence_score(
    scenarios: tuple[LearningReplayScenario, ...],
) -> int:
    return sum(scenario.replay_confidence_score for scenario in scenarios) // len(
        scenarios
    )


def _overall_replay_posture(
    scenarios: tuple[LearningReplayScenario, ...],
) -> LearningReplayPosture:
    if any(scenario.replay_confidence == "guarded" for scenario in scenarios):
        return "guarded"
    if any(scenario.replay_confidence == "low" for scenario in scenarios):
        return "review_required"
    return "candidate"


def _expected_replay_insight(
    kind: LearningReplayScenarioKind,
    confidence: LearningReplayConfidence,
) -> str:
    return (
        f"Inspect {kind} as {confidence} replay metadata to understand likely "
        "learning posture without running workflow replay."
    )


def _replay_safety_boundary(kind: LearningReplayScenarioKind) -> str:
    return (
        f"{kind} is metadata only: replay, workflow execution, provider calls, "
        "storage writes, generated-output mutation, and Runtime Evolution are "
        "blocked."
    )


def _scenario_actions(kind: LearningReplayScenarioKind) -> tuple[str, ...]:
    return (
        f"Expose {kind} for human review as replay metadata.",
        "Do not execute workflow replay or providers.",
        "Keep storage, output mutation, and Runtime Evolution disabled.",
    )


def _plan_actions(
    scenarios: tuple[LearningReplayScenario, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose learning replay scenarios as advisory metadata only.",
        "Keep replay execution, provider calls, storage writes, output mutation, "
        "and Runtime Evolution disabled.",
    ]
    if any(scenario.replay_confidence in {"low", "guarded"} for scenario in scenarios):
        actions.append("Require review before any future replay behavior.")
    return tuple(actions)


def _resolve_route(route: RouteName | str) -> RouteName:
    if isinstance(route, RouteName):
        return route
    return RouteName(str(route))
