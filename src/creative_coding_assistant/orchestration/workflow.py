"""Lightweight workflow state foundation for assistant orchestration."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from creative_coding_assistant.contracts import AssistantRequest
from creative_coding_assistant.orchestration.artifact_capability_matrix import (
    ArtifactCapabilityMatrix,
)
from creative_coding_assistant.orchestration.artifact_critic import (
    ArtifactCriticProfile,
)
from creative_coding_assistant.orchestration.artifact_critique import (
    ArtifactCritiqueSummary,
)
from creative_coding_assistant.orchestration.artifact_dependency_graph import (
    ArtifactDependencyGraph,
)
from creative_coding_assistant.orchestration.artifact_engine_contracts import (
    ArtifactIntelligenceEngineContractRegistry,
)
from creative_coding_assistant.orchestration.artifact_export_intelligence import (
    ArtifactExportIntelligenceProfile,
)
from creative_coding_assistant.orchestration.artifact_intelligence_synthesis import (
    ArtifactIntelligenceSynthesisProfile,
)
from creative_coding_assistant.orchestration.artifact_merge_planner import (
    ArtifactMergePlannerProfile,
)
from creative_coding_assistant.orchestration.artifact_planner import ArtifactPlan
from creative_coding_assistant.orchestration.artifact_refiner import (
    ArtifactRefinerProfile,
)
from creative_coding_assistant.orchestration.artifacts import (
    RefinementPassRecord,
    WorkflowArtifact,
)
from creative_coding_assistant.orchestration.audio_visual_scene import (
    AudioVisualSceneProfile,
)
from creative_coding_assistant.orchestration.clarification import ClarificationRequest
from creative_coding_assistant.orchestration.context import AssembledContextResponse
from creative_coding_assistant.orchestration.creative_composition import (
    CreativeCompositionPlan,
)
from creative_coding_assistant.orchestration.creative_confidence_engine import (
    CreativeConfidenceProfile,
)
from creative_coding_assistant.orchestration.creative_constraint_priorities import (
    CreativeConstraintPrioritization,
)
from creative_coding_assistant.orchestration.creative_constraints import (
    CreativeConstraintSolution,
)
from creative_coding_assistant.orchestration.creative_critic_engine import (
    CreativeCriticProfile,
)
from creative_coding_assistant.orchestration.creative_director import (
    CreativeAssistantDirectorBrief,
)
from creative_coding_assistant.orchestration.creative_hierarchy import (
    CreativeHierarchyPlan,
)
from creative_coding_assistant.orchestration.creative_improvement_planner import (
    CreativeImprovementPlannerProfile,
)
from creative_coding_assistant.orchestration.creative_intent import (
    CreativeIntentDecomposition,
)
from creative_coding_assistant.orchestration.creative_planning import (
    CreativeExecutionPlan,
)
from creative_coding_assistant.orchestration.creative_quality_prediction import (
    CreativeQualityPrediction,
)
from creative_coding_assistant.orchestration.creative_reasoning import (
    CreativeReasoningResult,
)
from creative_coding_assistant.orchestration.creative_score_engine import (
    CreativeScoreProfile,
)
from creative_coding_assistant.orchestration.creative_strategy import (
    CreativeStrategyProfile,
)
from creative_coding_assistant.orchestration.creative_technique import (
    CreativeTechniqueProfile,
)
from creative_coding_assistant.orchestration.creative_tradeoffs import (
    CreativeTradeoffProfile,
)
from creative_coding_assistant.orchestration.consistency_validation_engine import (
    ConsistencyValidationProfile,
)
from creative_coding_assistant.orchestration.cross_modality import (
    CrossModalityCompositionProfile,
)
from creative_coding_assistant.orchestration.emotional_consistency import (
    EmotionalConsistencyProfile,
)
from creative_coding_assistant.orchestration.generative_structure import (
    GenerativeStructureBlueprint,
)
from creative_coding_assistant.orchestration.memory import MemoryContextResponse
from creative_coding_assistant.orchestration.multi_artifact_strategy import (
    MultiArtifactStrategy,
)
from creative_coding_assistant.orchestration.procedural_structure import (
    ProceduralStructurePlan,
)
from creative_coding_assistant.orchestration.prompt_inputs import PromptInputResponse
from creative_coding_assistant.orchestration.prompt_templates import (
    RenderedPromptResponse,
)
from creative_coding_assistant.orchestration.retrieval import RetrievalContextResponse
from creative_coding_assistant.orchestration.reflection_loop_engine import (
    ReflectionLoopProfile,
)
from creative_coding_assistant.orchestration.routing import RouteDecision
from creative_coding_assistant.orchestration.runtime_capabilities import (
    RuntimeCapabilityProfile,
)
from creative_coding_assistant.orchestration.runtime_compatibility import (
    RuntimeCompatibilityProfile,
)
from creative_coding_assistant.orchestration.semantic_motif import SemanticMotifSystem
from creative_coding_assistant.orchestration.self_evaluation_engine import (
    SelfEvaluationProfile,
)
from creative_coding_assistant.orchestration.symbolic_narrative import (
    SymbolicNarrativePlan,
)
from creative_coding_assistant.orchestration.workflow_review import (
    WorkflowReviewResult,
)
from creative_coding_assistant.preview import PreviewResult


class WorkflowStep(StrEnum):
    INTAKE = "intake"
    ROUTING = "routing"
    MEMORY = "memory"
    RETRIEVAL = "retrieval"
    CONTEXT_ASSEMBLY = "context_assembly"
    PROMPT_INPUT = "prompt_input"
    PLANNING = "planning"
    DIRECTOR = "director"
    REASONING = "reasoning"
    PROMPT_RENDERING = "prompt_rendering"
    GENERATION = "generation"
    ARTIFACT_EXTRACTION = "artifact_extraction"
    PREVIEW_PREPARATION = "preview_preparation"
    ARTIFACT_CRITIQUE = "artifact_critique"
    REVIEW = "review"
    REFINEMENT = "refinement"
    FINALIZATION = "finalization"
    FAILURE = "failure"


class WorkflowStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


WORKFLOW_STEP_ORDER: tuple[WorkflowStep, ...] = (
    WorkflowStep.INTAKE,
    WorkflowStep.ROUTING,
    WorkflowStep.MEMORY,
    WorkflowStep.RETRIEVAL,
    WorkflowStep.CONTEXT_ASSEMBLY,
    WorkflowStep.PROMPT_INPUT,
    WorkflowStep.PLANNING,
    WorkflowStep.DIRECTOR,
    WorkflowStep.REASONING,
    WorkflowStep.PROMPT_RENDERING,
    WorkflowStep.GENERATION,
    WorkflowStep.ARTIFACT_EXTRACTION,
    WorkflowStep.PREVIEW_PREPARATION,
    WorkflowStep.ARTIFACT_CRITIQUE,
    WorkflowStep.REVIEW,
    WorkflowStep.REFINEMENT,
    WorkflowStep.FINALIZATION,
)


class WorkflowEventMetadata(BaseModel):
    """Small workflow snapshot that can be attached to future stream events."""

    model_config = ConfigDict(frozen=True)

    current_step: WorkflowStep | None = None
    status: WorkflowStatus
    completed_steps: tuple[WorkflowStep, ...] = ()
    skipped_steps: tuple[WorkflowStep, ...] = ()


class WorkflowFailureInfo(BaseModel):
    """Typed metadata for terminal workflow failures."""

    model_config = ConfigDict(frozen=True)

    step: WorkflowStep
    code: str
    message: str


class AssistantWorkflowState(BaseModel):
    """Explicit state for one assistant workflow run.

    The state intentionally mirrors the existing deterministic pipeline while
    remaining small enough to move through graph runtime nodes.
    """

    model_config = ConfigDict(frozen=True)

    request: AssistantRequest
    status: WorkflowStatus = WorkflowStatus.RUNNING
    current_step: WorkflowStep | None = None
    completed_steps: tuple[WorkflowStep, ...] = ()
    skipped_steps: tuple[WorkflowStep, ...] = ()
    route_decision: RouteDecision | None = None
    memory_context: MemoryContextResponse | None = None
    retrieval_context: RetrievalContextResponse | None = None
    assembled_context: AssembledContextResponse | None = None
    prompt_input: PromptInputResponse | None = None
    clarification: ClarificationRequest | None = None
    creative_intent: CreativeIntentDecomposition | None = None
    creative_hierarchy: CreativeHierarchyPlan | None = None
    creative_strategy: CreativeStrategyProfile | None = None
    creative_techniques: CreativeTechniqueProfile | None = None
    creative_plan: CreativeExecutionPlan | None = None
    creative_constraints: CreativeConstraintSolution | None = None
    creative_constraint_priorities: CreativeConstraintPrioritization | None = None
    runtime_capabilities: RuntimeCapabilityProfile | None = None
    creative_tradeoffs: CreativeTradeoffProfile | None = None
    creative_quality_prediction: CreativeQualityPrediction | None = None
    symbolic_narrative: SymbolicNarrativePlan | None = None
    creative_composition: CreativeCompositionPlan | None = None
    procedural_structure: ProceduralStructurePlan | None = None
    generative_structure: GenerativeStructureBlueprint | None = None
    semantic_motif: SemanticMotifSystem | None = None
    emotional_consistency: EmotionalConsistencyProfile | None = None
    cross_modality: CrossModalityCompositionProfile | None = None
    audio_visual_scene: AudioVisualSceneProfile | None = None
    artifact_plan: ArtifactPlan | None = None
    artifact_dependency_graph: ArtifactDependencyGraph | None = None
    runtime_compatibility: RuntimeCompatibilityProfile | None = None
    artifact_capability_matrix: ArtifactCapabilityMatrix | None = None
    multi_artifact_strategy: MultiArtifactStrategy | None = None
    artifact_critic: ArtifactCriticProfile | None = None
    artifact_refiner: ArtifactRefinerProfile | None = None
    artifact_intelligence_synthesis: (
        ArtifactIntelligenceSynthesisProfile | None
    ) = None
    artifact_merge_planner: ArtifactMergePlannerProfile | None = None
    artifact_export_intelligence: ArtifactExportIntelligenceProfile | None = None
    artifact_engine_contracts: ArtifactIntelligenceEngineContractRegistry | None = None
    creative_critic: CreativeCriticProfile | None = None
    self_evaluation: SelfEvaluationProfile | None = None
    creative_improvement_planner: CreativeImprovementPlannerProfile | None = None
    reflection_loop: ReflectionLoopProfile | None = None
    creative_confidence: CreativeConfidenceProfile | None = None
    creative_score: CreativeScoreProfile | None = None
    consistency_validation: ConsistencyValidationProfile | None = None
    creative_director: CreativeAssistantDirectorBrief | None = None
    creative_reasoning: CreativeReasoningResult | None = None
    rendered_prompt: RenderedPromptResponse | None = None
    artifacts: tuple[WorkflowArtifact, ...] = ()
    preview_results: tuple[PreviewResult, ...] = ()
    artifact_critique_summary: ArtifactCritiqueSummary | None = None
    review_result: WorkflowReviewResult | None = None
    refinement_count: int = 0
    refinement_passes: tuple[RefinementPassRecord, ...] = ()
    failure_info: WorkflowFailureInfo | None = None
    final_answer: str | None = None
    error_message: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            WorkflowStatus.COMPLETED,
            WorkflowStatus.FAILED,
        }

    def event_metadata(self) -> WorkflowEventMetadata:
        return WorkflowEventMetadata(
            current_step=self.current_step,
            status=self.status,
            completed_steps=self.completed_steps,
            skipped_steps=self.skipped_steps,
        )


def begin_assistant_workflow(request: AssistantRequest) -> AssistantWorkflowState:
    return AssistantWorkflowState(request=request)


def start_workflow_step(
    state: AssistantWorkflowState,
    step: WorkflowStep,
) -> AssistantWorkflowState:
    if state.is_terminal:
        raise ValueError("Cannot start a workflow step after terminal state.")
    if state.current_step is not None:
        raise ValueError("Cannot start a workflow step while another step is active.")
    if step in state.completed_steps or step in state.skipped_steps:
        raise ValueError(f"Workflow step already resolved: {step.value}")
    return state.model_copy(update={"current_step": step})


def restart_workflow_step(
    state: AssistantWorkflowState,
    step: WorkflowStep,
) -> AssistantWorkflowState:
    if state.is_terminal:
        raise ValueError("Cannot restart a workflow step after terminal state.")
    if state.current_step is not None:
        raise ValueError("Cannot restart a workflow step while another step is active.")
    return state.model_copy(update={"current_step": step})


def complete_workflow_step(
    state: AssistantWorkflowState,
    step: WorkflowStep,
    **updates: object,
) -> AssistantWorkflowState:
    _validate_active_step(state, step)
    return state.model_copy(
        update={
            "current_step": None,
            "completed_steps": _append_unique(state.completed_steps, step),
            "skipped_steps": _remove_step(state.skipped_steps, step),
            **updates,
        }
    )


def skip_workflow_step(
    state: AssistantWorkflowState,
    step: WorkflowStep,
) -> AssistantWorkflowState:
    _validate_active_step(state, step)
    return state.model_copy(
        update={
            "current_step": None,
            "completed_steps": _remove_step(state.completed_steps, step),
            "skipped_steps": _append_unique(state.skipped_steps, step),
        }
    )


def finish_workflow(
    state: AssistantWorkflowState,
    *,
    final_answer: str,
) -> AssistantWorkflowState:
    if state.current_step is not WorkflowStep.FINALIZATION:
        raise ValueError("Workflow must be in finalization before completion.")
    return complete_workflow_step(
        state,
        WorkflowStep.FINALIZATION,
        final_answer=final_answer,
        status=WorkflowStatus.COMPLETED,
    )


def fail_workflow(
    state: AssistantWorkflowState,
    *,
    error_message: str,
    failure_info: WorkflowFailureInfo | None = None,
    final_answer: str | None = None,
) -> AssistantWorkflowState:
    return state.model_copy(
        update={
            "status": WorkflowStatus.FAILED,
            "current_step": None,
            "error_message": error_message,
            "failure_info": failure_info,
            "final_answer": final_answer,
        }
    )


def next_workflow_step(step: WorkflowStep) -> WorkflowStep | None:
    index = WORKFLOW_STEP_ORDER.index(step)
    next_index = index + 1
    if next_index >= len(WORKFLOW_STEP_ORDER):
        return None
    return WORKFLOW_STEP_ORDER[next_index]


def _validate_active_step(
    state: AssistantWorkflowState,
    step: WorkflowStep,
) -> None:
    if state.current_step is not step:
        current = state.current_step.value if state.current_step is not None else None
        raise ValueError(
            f"Workflow step mismatch: expected {step.value}, current {current}."
        )


def _append_unique(
    steps: tuple[WorkflowStep, ...],
    step: WorkflowStep,
) -> tuple[WorkflowStep, ...]:
    if step in steps:
        return steps
    return (*steps, step)


def _remove_step(
    steps: tuple[WorkflowStep, ...],
    step: WorkflowStep,
) -> tuple[WorkflowStep, ...]:
    return tuple(item for item in steps if item is not step)
