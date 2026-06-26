"""LangGraph runtime for the assistant workflow."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from loguru import logger

from creative_coding_assistant.analytics import (
    LangSmithObservability,
    LangSmithRunMetadata,
)
from creative_coding_assistant.contracts import AssistantRequest, StreamEvent
from creative_coding_assistant.orchestration.artifact_capability_matrix import (
    derive_artifact_capability_matrix,
)
from creative_coding_assistant.orchestration.artifact_critic import (
    derive_artifact_critic_profile,
)
from creative_coding_assistant.orchestration.artifact_critique import (
    ArtifactCritiqueSummary,
    critique_workflow_artifacts,
)
from creative_coding_assistant.orchestration.artifact_dependency_graph import (
    derive_artifact_dependency_graph,
)
from creative_coding_assistant.orchestration.artifact_engine_contracts import (
    artifact_intelligence_engine_contracts,
)
from creative_coding_assistant.orchestration.artifact_export_intelligence import (
    derive_artifact_export_intelligence_profile,
)
from creative_coding_assistant.orchestration.artifact_intelligence_synthesis import (
    derive_artifact_intelligence_synthesis_profile,
)
from creative_coding_assistant.orchestration.artifact_merge_planner import (
    derive_artifact_merge_planner_profile,
)
from creative_coding_assistant.orchestration.artifact_planner import (
    derive_artifact_plan,
)
from creative_coding_assistant.orchestration.artifact_refiner import (
    derive_artifact_refiner_profile,
)
from creative_coding_assistant.orchestration.artifacts import (
    RefinementPassRecord,
    WorkflowArtifactCritique,
    extract_workflow_artifacts,
    prepare_workflow_preview_results,
)
from creative_coding_assistant.orchestration.audio_visual_scene import (
    derive_audio_visual_scene_profile,
)
from creative_coding_assistant.orchestration.clarification import ClarificationRequest
from creative_coding_assistant.orchestration.creative_composition import (
    derive_creative_composition_plan,
)
from creative_coding_assistant.orchestration.creative_confidence_engine import (
    derive_creative_confidence_profile,
)
from creative_coding_assistant.orchestration.creative_constraint_priorities import (
    derive_creative_constraint_priorities,
)
from creative_coding_assistant.orchestration.creative_constraints import (
    derive_creative_constraint_solution,
)
from creative_coding_assistant.orchestration.creative_critic_engine import (
    derive_creative_critic_profile,
)
from creative_coding_assistant.orchestration.creative_director import (
    CreativeAssistantDirectorBrief,
    derive_creative_assistant_director_brief,
)
from creative_coding_assistant.orchestration.creative_hierarchy import (
    derive_creative_hierarchy_plan,
)
from creative_coding_assistant.orchestration.creative_improvement_planner import (
    derive_creative_improvement_planner_profile,
)
from creative_coding_assistant.orchestration.creative_intent import (
    derive_creative_intent_decomposition,
)
from creative_coding_assistant.orchestration.creative_planning import (
    derive_creative_execution_plan,
)
from creative_coding_assistant.orchestration.creative_quality_prediction import (
    derive_creative_quality_prediction,
)
from creative_coding_assistant.orchestration.creative_reasoning import (
    CreativeReasoningResult,
    derive_creative_reasoning_result,
)
from creative_coding_assistant.orchestration.creative_strategy import (
    derive_creative_strategy_profile,
)
from creative_coding_assistant.orchestration.creative_technique import (
    derive_creative_technique_profile,
)
from creative_coding_assistant.orchestration.creative_tradeoffs import (
    derive_creative_tradeoff_profile,
)
from creative_coding_assistant.orchestration.cross_modality import (
    derive_cross_modality_composition_profile,
)
from creative_coding_assistant.orchestration.emotional_consistency import (
    derive_emotional_consistency_profile,
)
from creative_coding_assistant.orchestration.events import StreamEventBuilder
from creative_coding_assistant.orchestration.generative_structure import (
    derive_generative_structure_blueprint,
)
from creative_coding_assistant.orchestration.multi_artifact_strategy import (
    derive_multi_artifact_strategy,
)
from creative_coding_assistant.orchestration.procedural_structure import (
    derive_procedural_structure_plan,
)
from creative_coding_assistant.orchestration.prompt_templates import (
    RenderedPromptResponse,
    RenderedPromptRole,
    RenderedPromptSection,
    RenderedPromptSectionName,
)
from creative_coding_assistant.orchestration.refinement_passes import (
    attach_refinement_history,
    complete_latest_refinement_pass,
    plan_next_refinement_pass,
    select_refinement_source,
    start_refinement_pass_record,
)
from creative_coding_assistant.orchestration.reflection_loop_engine import (
    derive_reflection_loop_profile,
)
from creative_coding_assistant.orchestration.routing import RouteDecision
from creative_coding_assistant.orchestration.runtime_capabilities import (
    derive_runtime_capability_profile,
)
from creative_coding_assistant.orchestration.runtime_compatibility import (
    derive_runtime_compatibility_profile,
)
from creative_coding_assistant.orchestration.semantic_motif import (
    derive_semantic_motif_system,
)
from creative_coding_assistant.orchestration.self_evaluation_engine import (
    derive_self_evaluation_profile,
)
from creative_coding_assistant.orchestration.symbolic_narrative import (
    derive_symbolic_narrative_plan,
)
from creative_coding_assistant.orchestration.workflow import (
    AssistantWorkflowState,
    WorkflowFailureInfo,
    WorkflowStep,
    begin_assistant_workflow,
    complete_workflow_step,
    fail_workflow,
    finish_workflow,
    restart_workflow_step,
    skip_workflow_step,
    start_workflow_step,
)
from creative_coding_assistant.orchestration.workflow_review import (
    MAX_WORKFLOW_REFINEMENT_COUNT,
    WorkflowReviewOutcome,
    WorkflowReviewResult,
    review_assistant_answer,
)


class GenerationResultLike(Protocol):
    answer: str


class AssistantWorkflowGraphState(TypedDict, total=False):
    workflow_state: AssistantWorkflowState
    route_payload: dict[str, object]
    generation_result: GenerationResultLike | None
    pending_failure: WorkflowFailureInfo | None
    failure_event_emitted: bool


class AssistantWorkflowGraphContext(TypedDict):
    runtime: AssistantWorkflowRuntime


@dataclass(frozen=True)
class AssistantWorkflowRuntime:
    """Runtime services needed by graph nodes for one assistant turn."""

    event_builder: StreamEventBuilder
    observability: LangSmithObservability
    observability_run: LangSmithRunMetadata
    route_fn: Callable[[AssistantRequest], RouteDecision]
    stream_request_received: Callable[..., Iterator[object]]
    stream_route_selected: Callable[..., Iterator[object]]
    stream_memory_context: Callable[..., Iterator[object]]
    stream_retrieval_context: Callable[..., Iterator[object]]
    stream_assembled_context: Callable[..., Iterator[object]]
    stream_prompt_inputs: Callable[..., Iterator[object]]
    stream_rendered_prompt: Callable[..., Iterator[object]]
    stream_generation: Callable[..., Iterator[object]]
    build_shell_answer: Callable[[RouteDecision], str]


ASSISTANT_WORKFLOW_NODE_ORDER: tuple[str, ...] = (
    "intake",
    "routing",
    "memory",
    "retrieval",
    "context_assembly",
    "prompt_input",
    "planning",
    "director",
    "reasoning",
    "prompt_rendering",
    "generation",
    "artifact_extraction",
    "preview_preparation",
    "artifact_critique",
    "review",
    "refinement",
    "finalization",
    "failure",
)
ASSISTANT_WORKFLOW_RECURSION_LIMIT = 40


def build_initial_workflow_graph_state(
    request: AssistantRequest,
) -> AssistantWorkflowGraphState:
    return {"workflow_state": begin_assistant_workflow(request)}


def build_assistant_workflow_graph() -> Any:
    graph = StateGraph(
        AssistantWorkflowGraphState,
        context_schema=AssistantWorkflowGraphContext,
    )
    graph.add_node("intake", _intake_node)
    graph.add_node("routing", _routing_node)
    graph.add_node("memory", _memory_node)
    graph.add_node("retrieval", _retrieval_node)
    graph.add_node("context_assembly", _context_assembly_node)
    graph.add_node("prompt_input", _prompt_input_node)
    graph.add_node("planning", _planning_node)
    graph.add_node("director", _director_node)
    graph.add_node("reasoning", _reasoning_node)
    graph.add_node("prompt_rendering", _prompt_rendering_node)
    graph.add_node("generation", _generation_node)
    graph.add_node("artifact_extraction", _artifact_extraction_node)
    graph.add_node("preview_preparation", _preview_preparation_node)
    graph.add_node("artifact_critique", _artifact_critique_node)
    graph.add_node("review", _review_node)
    graph.add_node("refinement", _refinement_node)
    graph.add_node("finalization", _finalization_node)
    graph.add_node("failure", _failure_node)

    graph.add_edge(START, "intake")
    review_index = ASSISTANT_WORKFLOW_NODE_ORDER.index("review")
    for index in range(review_index):
        current_node = ASSISTANT_WORKFLOW_NODE_ORDER[index]
        next_node = ASSISTANT_WORKFLOW_NODE_ORDER[index + 1]
        if current_node == "prompt_input":
            graph.add_conditional_edges(
                current_node,
                _next_node_after_prompt_input,
                {
                    "planning": "planning",
                    "finalization": "finalization",
                    "failure": "failure",
                },
            )
            continue
        graph.add_conditional_edges(
            current_node,
            lambda state, next_node=next_node: _next_node_or_failure(state, next_node),
            {
                next_node: next_node,
                "failure": "failure",
            },
        )
    graph.add_conditional_edges(
        "review",
        _next_node_after_review,
        {
            "finalization": "finalization",
            "refinement": "refinement",
            "failure": "failure",
        },
    )
    graph.add_conditional_edges(
        "refinement",
        lambda state: _next_node_or_failure(state, "generation"),
        {
            "generation": "generation",
            "failure": "failure",
        },
    )
    graph.add_conditional_edges(
        "finalization",
        _next_node_after_finalization,
        {
            "end": END,
            "failure": "failure",
        },
    )
    graph.add_edge("failure", END)
    return graph.compile()


def stream_assistant_workflow_events(
    *,
    graph: Any,
    request: AssistantRequest,
    runtime: AssistantWorkflowRuntime,
) -> Iterator[StreamEvent]:
    initial_state = build_initial_workflow_graph_state(request)
    for item in graph.stream(
        initial_state,
        config={"recursion_limit": ASSISTANT_WORKFLOW_RECURSION_LIMIT},
        context={"runtime": runtime},
        stream_mode="custom",
    ):
        if isinstance(item, StreamEvent):
            yield item


def _intake_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.INTAKE,
    )
    try:
        _emit_streaming_step(
            runtime_context.stream_request_received(
                builder=runtime_context.event_builder,
                request=workflow_state.request,
                observability=runtime_context.observability,
                observability_run=runtime_context.observability_run,
            ),
            workflow_state=workflow_state,
        )
        return {
            "workflow_state": _complete_node(
                workflow_state,
                runtime_context,
                WorkflowStep.INTAKE,
                decision_reason="request_received",
            )
        }
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.INTAKE,
            exc=exc,
        )


def _routing_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.ROUTING,
    )
    try:
        decision = runtime_context.route_fn(workflow_state.request)
        route_payload = decision.model_dump(mode="json")
        _emit_streaming_step(
            runtime_context.stream_route_selected(
                builder=runtime_context.event_builder,
                decision=decision,
                route_payload=route_payload,
            ),
            workflow_state=workflow_state,
        )
        return {
            "workflow_state": _complete_node(
                workflow_state,
                runtime_context,
                WorkflowStep.ROUTING,
                decision_reason=f"route_selected:{decision.route.value}",
                route_decision=decision,
            ),
            "route_payload": route_payload,
        }
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.ROUTING,
            exc=exc,
        )


def _memory_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.MEMORY,
    )
    try:
        memory_context = _emit_streaming_step(
            runtime_context.stream_memory_context(
                builder=runtime_context.event_builder,
                request=workflow_state.request,
                decision=_route_decision(workflow_state),
            ),
            workflow_state=workflow_state,
        )
        if memory_context is None:
            return {
                "workflow_state": _skip_node(
                    workflow_state,
                    runtime_context,
                    WorkflowStep.MEMORY,
                    decision_reason="memory_context_unavailable",
                )
            }
        return {
            "workflow_state": _complete_node(
                workflow_state,
                runtime_context,
                WorkflowStep.MEMORY,
                decision_reason="memory_context_available",
                memory_context=memory_context,
            )
        }
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.MEMORY,
            exc=exc,
        )


def _retrieval_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.RETRIEVAL,
    )
    try:
        retrieval_context = _emit_streaming_step(
            runtime_context.stream_retrieval_context(
                builder=runtime_context.event_builder,
                request=workflow_state.request,
                decision=_route_decision(workflow_state),
                observability_run=runtime_context.observability_run,
            ),
            workflow_state=workflow_state,
        )
        if retrieval_context is None:
            return {
                "workflow_state": _skip_node(
                    workflow_state,
                    runtime_context,
                    WorkflowStep.RETRIEVAL,
                    decision_reason="retrieval_context_unavailable",
                )
            }
        return {
            "workflow_state": _complete_node(
                workflow_state,
                runtime_context,
                WorkflowStep.RETRIEVAL,
                decision_reason="retrieval_context_available",
                retrieval_context=retrieval_context,
            )
        }
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.RETRIEVAL,
            exc=exc,
        )


def _context_assembly_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.CONTEXT_ASSEMBLY,
    )
    try:
        assembled_context = _emit_streaming_step(
            runtime_context.stream_assembled_context(
                builder=runtime_context.event_builder,
                decision=_route_decision(workflow_state),
                memory_context=workflow_state.memory_context,
                retrieval_context=workflow_state.retrieval_context,
            ),
            workflow_state=workflow_state,
        )
        if assembled_context is None:
            return {
                "workflow_state": _skip_node(
                    workflow_state,
                    runtime_context,
                    WorkflowStep.CONTEXT_ASSEMBLY,
                    decision_reason="context_assembly_unavailable",
                )
            }
        return {
            "workflow_state": _complete_node(
                workflow_state,
                runtime_context,
                WorkflowStep.CONTEXT_ASSEMBLY,
                decision_reason="context_assembled",
                assembled_context=assembled_context,
            )
        }
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.CONTEXT_ASSEMBLY,
            exc=exc,
        )


def _prompt_input_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.PROMPT_INPUT,
    )
    try:
        prompt_input = _emit_streaming_step(
            runtime_context.stream_prompt_inputs(
                builder=runtime_context.event_builder,
                request=workflow_state.request,
                decision=_route_decision(workflow_state),
                assembled_context=workflow_state.assembled_context,
            ),
            workflow_state=workflow_state,
        )
        if prompt_input is None:
            return {
                "workflow_state": _skip_node(
                    workflow_state,
                    runtime_context,
                    WorkflowStep.PROMPT_INPUT,
                    decision_reason="prompt_input_unavailable",
                )
            }
        if prompt_input.clarification is not None:
            clarification_state = workflow_state.model_copy(
                update={
                    "prompt_input": prompt_input,
                    "clarification": prompt_input.clarification,
                }
            )
            _emit(
                runtime_context.event_builder.prompt_input(
                    code="clarification_required",
                    message="Clarification required before generation.",
                    clarification=prompt_input.clarification.model_dump(mode="json"),
                ),
                workflow_state=clarification_state,
                step=WorkflowStep.PROMPT_INPUT,
            )
            return {
                "workflow_state": _complete_node(
                    workflow_state,
                    runtime_context,
                    WorkflowStep.PROMPT_INPUT,
                    transition_target=WorkflowStep.FINALIZATION.value,
                    decision_reason="clarification_required",
                    prompt_input=prompt_input,
                    clarification=prompt_input.clarification,
                )
            }
        return {
            "workflow_state": _complete_node(
                workflow_state,
                runtime_context,
                WorkflowStep.PROMPT_INPUT,
                decision_reason="prompt_input_prepared",
                prompt_input=prompt_input,
            )
        }
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.PROMPT_INPUT,
            exc=exc,
        )


def _planning_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.PLANNING,
    )
    try:
        prompt_input = workflow_state.prompt_input
        if prompt_input is None:
            return {
                "workflow_state": _skip_node(
                    workflow_state,
                    runtime_context,
                    WorkflowStep.PLANNING,
                    decision_reason="prompt_input_unavailable_for_planning",
                )
            }

        retrieval_chunk_count = (
            len(prompt_input.retrieval_input.chunks)
            if prompt_input.retrieval_input is not None
            else 0
        )
        creative_intent = derive_creative_intent_decomposition(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_translation=prompt_input.creative_translation,
        )
        creative_hierarchy = derive_creative_hierarchy_plan(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_intent=creative_intent,
            creative_translation=prompt_input.creative_translation,
        )
        strategy = derive_creative_strategy_profile(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_translation=prompt_input.creative_translation,
        )
        techniques = derive_creative_technique_profile(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_translation=prompt_input.creative_translation,
            creative_strategy=strategy,
        )
        plan = derive_creative_execution_plan(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_translation=prompt_input.creative_translation,
            creative_strategy=strategy,
            creative_techniques=techniques,
            retrieval_chunk_count=retrieval_chunk_count,
        )
        constraints = derive_creative_constraint_solution(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_translation=prompt_input.creative_translation,
            creative_plan=plan,
            creative_strategy=strategy,
            creative_techniques=techniques,
            clarification=workflow_state.clarification,
            retrieval_chunk_count=retrieval_chunk_count,
        )
        runtime_capabilities = derive_runtime_capability_profile(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_translation=prompt_input.creative_translation,
            creative_strategy=strategy,
            creative_techniques=techniques,
            creative_plan=plan,
            creative_constraints=constraints,
        )
        tradeoffs = derive_creative_tradeoff_profile(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_translation=prompt_input.creative_translation,
            creative_strategy=strategy,
            creative_techniques=techniques,
            creative_plan=plan,
            creative_constraints=constraints,
            runtime_capabilities=runtime_capabilities,
        )
        constraint_priorities = derive_creative_constraint_priorities(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_translation=prompt_input.creative_translation,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_plan=plan,
            creative_constraints=constraints,
            creative_strategy=strategy,
            creative_techniques=techniques,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=tradeoffs,
        )
        quality_prediction = derive_creative_quality_prediction(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_translation=prompt_input.creative_translation,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_plan=plan,
            creative_constraints=constraints,
            creative_constraint_priorities=constraint_priorities,
            creative_strategy=strategy,
            creative_techniques=techniques,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=tradeoffs,
        )
        symbolic_narrative = derive_symbolic_narrative_plan(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_translation=prompt_input.creative_translation,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_plan=plan,
            creative_constraints=constraints,
            creative_constraint_priorities=constraint_priorities,
            creative_strategy=strategy,
            creative_techniques=techniques,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=tradeoffs,
            creative_quality_prediction=quality_prediction,
        )
        creative_composition = derive_creative_composition_plan(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_translation=prompt_input.creative_translation,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_plan=plan,
            creative_constraints=constraints,
            creative_constraint_priorities=constraint_priorities,
            creative_strategy=strategy,
            creative_techniques=techniques,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=tradeoffs,
            creative_quality_prediction=quality_prediction,
            symbolic_narrative=symbolic_narrative,
        )
        procedural_structure = derive_procedural_structure_plan(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_translation=prompt_input.creative_translation,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_plan=plan,
            creative_constraints=constraints,
            creative_constraint_priorities=constraint_priorities,
            creative_strategy=strategy,
            creative_techniques=techniques,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=tradeoffs,
            creative_quality_prediction=quality_prediction,
            symbolic_narrative=symbolic_narrative,
            creative_composition=creative_composition,
        )
        generative_structure = derive_generative_structure_blueprint(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_translation=prompt_input.creative_translation,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_plan=plan,
            creative_constraints=constraints,
            creative_constraint_priorities=constraint_priorities,
            creative_strategy=strategy,
            creative_techniques=techniques,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=tradeoffs,
            creative_quality_prediction=quality_prediction,
            symbolic_narrative=symbolic_narrative,
            creative_composition=creative_composition,
            procedural_structure=procedural_structure,
        )
        semantic_motif = derive_semantic_motif_system(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_translation=prompt_input.creative_translation,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_plan=plan,
            creative_constraints=constraints,
            creative_constraint_priorities=constraint_priorities,
            creative_strategy=strategy,
            creative_techniques=techniques,
            creative_tradeoffs=tradeoffs,
            creative_quality_prediction=quality_prediction,
            symbolic_narrative=symbolic_narrative,
            creative_composition=creative_composition,
            procedural_structure=procedural_structure,
            generative_structure=generative_structure,
        )
        emotional_consistency = derive_emotional_consistency_profile(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_translation=prompt_input.creative_translation,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_plan=plan,
            creative_constraints=constraints,
            creative_constraint_priorities=constraint_priorities,
            creative_strategy=strategy,
            creative_techniques=techniques,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=tradeoffs,
            creative_quality_prediction=quality_prediction,
            symbolic_narrative=symbolic_narrative,
            creative_composition=creative_composition,
            procedural_structure=procedural_structure,
            generative_structure=generative_structure,
            semantic_motif=semantic_motif,
        )
        cross_modality = derive_cross_modality_composition_profile(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_translation=prompt_input.creative_translation,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_plan=plan,
            creative_constraints=constraints,
            creative_constraint_priorities=constraint_priorities,
            creative_strategy=strategy,
            creative_techniques=techniques,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=tradeoffs,
            creative_quality_prediction=quality_prediction,
            symbolic_narrative=symbolic_narrative,
            creative_composition=creative_composition,
            procedural_structure=procedural_structure,
            generative_structure=generative_structure,
            semantic_motif=semantic_motif,
            emotional_consistency=emotional_consistency,
        )
        audio_visual_scene = derive_audio_visual_scene_profile(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_translation=prompt_input.creative_translation,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_plan=plan,
            creative_constraints=constraints,
            creative_constraint_priorities=constraint_priorities,
            creative_strategy=strategy,
            creative_techniques=techniques,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=tradeoffs,
            creative_quality_prediction=quality_prediction,
            symbolic_narrative=symbolic_narrative,
            creative_composition=creative_composition,
            procedural_structure=procedural_structure,
            generative_structure=generative_structure,
            semantic_motif=semantic_motif,
            emotional_consistency=emotional_consistency,
            cross_modality=cross_modality,
        )
        artifact_plan = derive_artifact_plan(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_translation=prompt_input.creative_translation,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_plan=plan,
            creative_constraints=constraints,
            creative_constraint_priorities=constraint_priorities,
            creative_strategy=strategy,
            creative_techniques=techniques,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=tradeoffs,
            creative_quality_prediction=quality_prediction,
            symbolic_narrative=symbolic_narrative,
            creative_composition=creative_composition,
            procedural_structure=procedural_structure,
            generative_structure=generative_structure,
            semantic_motif=semantic_motif,
            emotional_consistency=emotional_consistency,
            cross_modality=cross_modality,
            audio_visual_scene=audio_visual_scene,
        )
        artifact_dependency_graph = derive_artifact_dependency_graph(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            artifact_plan=artifact_plan,
            creative_translation=prompt_input.creative_translation,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_plan=plan,
            creative_constraints=constraints,
            creative_constraint_priorities=constraint_priorities,
            creative_strategy=strategy,
            creative_techniques=techniques,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=tradeoffs,
            creative_quality_prediction=quality_prediction,
            symbolic_narrative=symbolic_narrative,
            creative_composition=creative_composition,
            procedural_structure=procedural_structure,
            generative_structure=generative_structure,
            semantic_motif=semantic_motif,
            emotional_consistency=emotional_consistency,
            cross_modality=cross_modality,
            audio_visual_scene=audio_visual_scene,
        )
        runtime_compatibility = derive_runtime_compatibility_profile(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            artifact_plan=artifact_plan,
            artifact_dependency_graph=artifact_dependency_graph,
            runtime_capabilities=runtime_capabilities,
            creative_plan=plan,
            creative_constraints=constraints,
            creative_tradeoffs=tradeoffs,
        )
        artifact_capability_matrix = derive_artifact_capability_matrix(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            artifact_plan=artifact_plan,
            artifact_dependency_graph=artifact_dependency_graph,
            runtime_capabilities=runtime_capabilities,
            runtime_compatibility=runtime_compatibility,
            creative_plan=plan,
            creative_constraints=constraints,
            creative_strategy=strategy,
            creative_techniques=techniques,
            creative_tradeoffs=tradeoffs,
        )
        multi_artifact_strategy = derive_multi_artifact_strategy(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            artifact_plan=artifact_plan,
            artifact_dependency_graph=artifact_dependency_graph,
            runtime_capabilities=runtime_capabilities,
            runtime_compatibility=runtime_compatibility,
            artifact_capability_matrix=artifact_capability_matrix,
            creative_plan=plan,
            creative_constraints=constraints,
            creative_tradeoffs=tradeoffs,
        )
        artifact_critic = derive_artifact_critic_profile(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            artifact_plan=artifact_plan,
            artifact_dependency_graph=artifact_dependency_graph,
            runtime_compatibility=runtime_compatibility,
            artifact_capability_matrix=artifact_capability_matrix,
            multi_artifact_strategy=multi_artifact_strategy,
        )
        artifact_refiner = derive_artifact_refiner_profile(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            artifact_plan=artifact_plan,
            artifact_dependency_graph=artifact_dependency_graph,
            runtime_compatibility=runtime_compatibility,
            artifact_capability_matrix=artifact_capability_matrix,
            multi_artifact_strategy=multi_artifact_strategy,
            artifact_critic=artifact_critic,
        )
        artifact_intelligence_synthesis = (
            derive_artifact_intelligence_synthesis_profile(
                request=workflow_state.request,
                route_decision=workflow_state.route_decision,
                artifact_plan=artifact_plan,
                artifact_dependency_graph=artifact_dependency_graph,
                runtime_compatibility=runtime_compatibility,
                artifact_capability_matrix=artifact_capability_matrix,
                multi_artifact_strategy=multi_artifact_strategy,
                artifact_critic=artifact_critic,
                artifact_refiner=artifact_refiner,
            )
        )
        artifact_merge_planner = derive_artifact_merge_planner_profile(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            artifact_plan=artifact_plan,
            artifact_dependency_graph=artifact_dependency_graph,
            runtime_compatibility=runtime_compatibility,
            artifact_capability_matrix=artifact_capability_matrix,
            multi_artifact_strategy=multi_artifact_strategy,
            artifact_critic=artifact_critic,
            artifact_refiner=artifact_refiner,
            artifact_intelligence_synthesis=artifact_intelligence_synthesis,
        )
        artifact_export_intelligence = derive_artifact_export_intelligence_profile(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            artifact_plan=artifact_plan,
            artifact_dependency_graph=artifact_dependency_graph,
            runtime_compatibility=runtime_compatibility,
            artifact_capability_matrix=artifact_capability_matrix,
            multi_artifact_strategy=multi_artifact_strategy,
            artifact_critic=artifact_critic,
            artifact_refiner=artifact_refiner,
            artifact_intelligence_synthesis=artifact_intelligence_synthesis,
            artifact_merge_planner=artifact_merge_planner,
        )
        artifact_engine_contracts = artifact_intelligence_engine_contracts()
        creative_critic = derive_creative_critic_profile(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_translation=prompt_input.creative_translation,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_plan=plan,
            creative_constraints=constraints,
            creative_constraint_priorities=constraint_priorities,
            creative_strategy=strategy,
            creative_techniques=techniques,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=tradeoffs,
            creative_quality_prediction=quality_prediction,
            symbolic_narrative=symbolic_narrative,
            creative_composition=creative_composition,
            procedural_structure=procedural_structure,
            generative_structure=generative_structure,
            semantic_motif=semantic_motif,
            emotional_consistency=emotional_consistency,
            cross_modality=cross_modality,
            audio_visual_scene=audio_visual_scene,
            artifact_plan=artifact_plan,
            artifact_dependency_graph=artifact_dependency_graph,
            runtime_compatibility=runtime_compatibility,
            artifact_capability_matrix=artifact_capability_matrix,
            multi_artifact_strategy=multi_artifact_strategy,
            artifact_critic=artifact_critic,
            artifact_refiner=artifact_refiner,
            artifact_intelligence_synthesis=artifact_intelligence_synthesis,
            artifact_merge_planner=artifact_merge_planner,
            artifact_export_intelligence=artifact_export_intelligence,
        )
        self_evaluation = derive_self_evaluation_profile(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_translation=prompt_input.creative_translation,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_plan=plan,
            creative_constraints=constraints,
            creative_constraint_priorities=constraint_priorities,
            creative_strategy=strategy,
            creative_techniques=techniques,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=tradeoffs,
            creative_quality_prediction=quality_prediction,
            symbolic_narrative=symbolic_narrative,
            creative_composition=creative_composition,
            procedural_structure=procedural_structure,
            generative_structure=generative_structure,
            semantic_motif=semantic_motif,
            emotional_consistency=emotional_consistency,
            cross_modality=cross_modality,
            audio_visual_scene=audio_visual_scene,
            artifact_plan=artifact_plan,
            artifact_dependency_graph=artifact_dependency_graph,
            runtime_compatibility=runtime_compatibility,
            artifact_capability_matrix=artifact_capability_matrix,
            multi_artifact_strategy=multi_artifact_strategy,
            artifact_critic=artifact_critic,
            artifact_refiner=artifact_refiner,
            artifact_intelligence_synthesis=artifact_intelligence_synthesis,
            artifact_merge_planner=artifact_merge_planner,
            artifact_export_intelligence=artifact_export_intelligence,
            creative_critic=creative_critic,
        )
        creative_improvement_planner = (
            derive_creative_improvement_planner_profile(
                request=workflow_state.request,
                route_decision=workflow_state.route_decision,
                creative_critic=creative_critic,
                self_evaluation=self_evaluation,
            )
        )
        reflection_loop = derive_reflection_loop_profile(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_critic=creative_critic,
            self_evaluation=self_evaluation,
            creative_improvement_planner=creative_improvement_planner,
            planning_metadata=(
                creative_intent,
                creative_hierarchy,
                strategy,
                techniques,
                plan,
                constraints,
                constraint_priorities,
                runtime_capabilities,
                tradeoffs,
                quality_prediction,
                symbolic_narrative,
                creative_composition,
                procedural_structure,
                generative_structure,
                semantic_motif,
                emotional_consistency,
                cross_modality,
                audio_visual_scene,
                artifact_plan,
                artifact_dependency_graph,
                runtime_compatibility,
                artifact_capability_matrix,
                multi_artifact_strategy,
                artifact_critic,
                artifact_refiner,
                artifact_intelligence_synthesis,
                artifact_merge_planner,
                artifact_export_intelligence,
            ),
        )
        creative_confidence = derive_creative_confidence_profile(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_critic=creative_critic,
            self_evaluation=self_evaluation,
            creative_improvement_planner=creative_improvement_planner,
            reflection_loop=reflection_loop,
            planning_metadata=(
                creative_intent,
                creative_hierarchy,
                strategy,
                techniques,
                plan,
                constraints,
                constraint_priorities,
                runtime_capabilities,
                tradeoffs,
                quality_prediction,
                symbolic_narrative,
                creative_composition,
                procedural_structure,
                generative_structure,
                semantic_motif,
                emotional_consistency,
                cross_modality,
                audio_visual_scene,
                artifact_plan,
                artifact_dependency_graph,
                runtime_compatibility,
                artifact_capability_matrix,
                multi_artifact_strategy,
                artifact_critic,
                artifact_refiner,
                artifact_intelligence_synthesis,
                artifact_merge_planner,
                artifact_export_intelligence,
            ),
        )
        planned_prompt_input = prompt_input.model_copy(
            update={
                "creative_strategy": strategy,
                "creative_intent": creative_intent,
                "creative_hierarchy": creative_hierarchy,
                "creative_techniques": techniques,
                "creative_plan": plan,
                "creative_constraints": constraints,
                "creative_constraint_priorities": constraint_priorities,
                "runtime_capabilities": runtime_capabilities,
                "creative_tradeoffs": tradeoffs,
                "creative_quality_prediction": quality_prediction,
                "symbolic_narrative": symbolic_narrative,
                "creative_composition": creative_composition,
                "procedural_structure": procedural_structure,
                "generative_structure": generative_structure,
                "semantic_motif": semantic_motif,
                "emotional_consistency": emotional_consistency,
                "cross_modality": cross_modality,
                "audio_visual_scene": audio_visual_scene,
                "artifact_plan": artifact_plan,
                "artifact_dependency_graph": artifact_dependency_graph,
                "runtime_compatibility": runtime_compatibility,
                "artifact_capability_matrix": artifact_capability_matrix,
                "multi_artifact_strategy": multi_artifact_strategy,
                "artifact_critic": artifact_critic,
                "artifact_refiner": artifact_refiner,
                "artifact_intelligence_synthesis": (
                    artifact_intelligence_synthesis
                ),
                "artifact_merge_planner": artifact_merge_planner,
                "artifact_export_intelligence": artifact_export_intelligence,
                "artifact_engine_contracts": artifact_engine_contracts,
                "creative_critic": creative_critic,
                "self_evaluation": self_evaluation,
                "creative_improvement_planner": creative_improvement_planner,
                "reflection_loop": reflection_loop,
                "creative_confidence": creative_confidence,
            }
        )
        planned_state = workflow_state.model_copy(
            update={
                "creative_strategy": strategy,
                "creative_intent": creative_intent,
                "creative_hierarchy": creative_hierarchy,
                "creative_techniques": techniques,
                "creative_plan": plan,
                "creative_constraints": constraints,
                "creative_constraint_priorities": constraint_priorities,
                "runtime_capabilities": runtime_capabilities,
                "creative_tradeoffs": tradeoffs,
                "creative_quality_prediction": quality_prediction,
                "symbolic_narrative": symbolic_narrative,
                "creative_composition": creative_composition,
                "procedural_structure": procedural_structure,
                "generative_structure": generative_structure,
                "semantic_motif": semantic_motif,
                "emotional_consistency": emotional_consistency,
                "cross_modality": cross_modality,
                "audio_visual_scene": audio_visual_scene,
                "artifact_plan": artifact_plan,
                "artifact_dependency_graph": artifact_dependency_graph,
                "runtime_compatibility": runtime_compatibility,
                "artifact_capability_matrix": artifact_capability_matrix,
                "multi_artifact_strategy": multi_artifact_strategy,
                "artifact_critic": artifact_critic,
                "artifact_refiner": artifact_refiner,
                "artifact_intelligence_synthesis": (
                    artifact_intelligence_synthesis
                ),
                "artifact_merge_planner": artifact_merge_planner,
                "artifact_export_intelligence": artifact_export_intelligence,
                "artifact_engine_contracts": artifact_engine_contracts,
                "creative_critic": creative_critic,
                "self_evaluation": self_evaluation,
                "creative_improvement_planner": (
                    creative_improvement_planner
                ),
                "reflection_loop": reflection_loop,
                "creative_confidence": creative_confidence,
                "prompt_input": planned_prompt_input,
            }
        )
        _emit(
            runtime_context.event_builder.planning(
                code="creative_plan_prepared",
                message="Creative execution plan prepared.",
                creative_intent=creative_intent.model_dump(mode="json"),
                creative_hierarchy=creative_hierarchy.model_dump(mode="json"),
                creative_strategy=strategy.model_dump(mode="json"),
                creative_techniques=techniques.model_dump(mode="json"),
                creative_plan=plan.model_dump(mode="json"),
                creative_constraints=constraints.model_dump(mode="json"),
                creative_constraint_priorities=constraint_priorities.model_dump(
                    mode="json"
                ),
                runtime_capabilities=runtime_capabilities.model_dump(mode="json"),
                creative_tradeoffs=tradeoffs.model_dump(mode="json"),
                creative_quality_prediction=quality_prediction.model_dump(
                    mode="json"
                ),
                symbolic_narrative=symbolic_narrative.model_dump(mode="json"),
                creative_composition=creative_composition.model_dump(mode="json"),
                procedural_structure=procedural_structure.model_dump(mode="json"),
                generative_structure=generative_structure.model_dump(mode="json"),
                semantic_motif=semantic_motif.model_dump(mode="json"),
                emotional_consistency=emotional_consistency.model_dump(mode="json"),
                cross_modality=cross_modality.model_dump(mode="json"),
                audio_visual_scene=audio_visual_scene.model_dump(mode="json"),
                artifact_plan=artifact_plan.model_dump(mode="json"),
                artifact_dependency_graph=artifact_dependency_graph.model_dump(
                    mode="json"
                ),
                runtime_compatibility=runtime_compatibility.model_dump(mode="json"),
                artifact_capability_matrix=artifact_capability_matrix.model_dump(
                    mode="json"
                ),
                multi_artifact_strategy=multi_artifact_strategy.model_dump(
                    mode="json"
                ),
                artifact_critic=artifact_critic.model_dump(mode="json"),
                artifact_refiner=artifact_refiner.model_dump(mode="json"),
                artifact_intelligence_synthesis=(
                    artifact_intelligence_synthesis.model_dump(mode="json")
                ),
                artifact_merge_planner=artifact_merge_planner.model_dump(
                    mode="json"
                ),
                artifact_export_intelligence=(
                    artifact_export_intelligence.model_dump(mode="json")
                ),
                artifact_engine_contracts=artifact_engine_contracts.model_dump(
                    mode="json"
                ),
                creative_critic=creative_critic.model_dump(mode="json"),
                self_evaluation=self_evaluation.model_dump(mode="json"),
                creative_improvement_planner=(
                    creative_improvement_planner.model_dump(mode="json")
                ),
                reflection_loop=reflection_loop.model_dump(mode="json"),
                creative_confidence=creative_confidence.model_dump(mode="json"),
            ),
            workflow_state=planned_state,
            step=WorkflowStep.PLANNING,
        )
        return {
            "workflow_state": _complete_node(
                planned_state,
                runtime_context,
                WorkflowStep.PLANNING,
                decision_reason="creative_plan_prepared",
            )
        }
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.PLANNING,
            exc=exc,
        )


def _director_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.DIRECTOR,
    )
    try:
        if workflow_state.prompt_input is None or workflow_state.creative_plan is None:
            return {
                "workflow_state": _skip_node(
                    workflow_state,
                    runtime_context,
                    WorkflowStep.DIRECTOR,
                    decision_reason="director_inputs_unavailable",
                )
            }

        director = _derive_director_brief(workflow_state)
        directed_prompt_input = workflow_state.prompt_input.model_copy(
            update={"creative_director": director}
        )
        directed_state = workflow_state.model_copy(
            update={
                "creative_director": director,
                "prompt_input": directed_prompt_input,
            }
        )
        _emit(
            runtime_context.event_builder.planning(
                code="creative_director_prepared",
                message="Creative Assistant Director guidance prepared.",
                creative_director=director.model_dump(mode="json"),
            ),
            workflow_state=directed_state,
            step=WorkflowStep.DIRECTOR,
        )
        return {
            "workflow_state": _complete_node(
                directed_state,
                runtime_context,
                WorkflowStep.DIRECTOR,
                decision_reason="creative_director_prepared",
            )
        }
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.DIRECTOR,
            exc=exc,
        )


def _reasoning_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.REASONING,
    )
    try:
        if (
            workflow_state.prompt_input is None
            or workflow_state.creative_director is None
        ):
            return {
                "workflow_state": _skip_node(
                    workflow_state,
                    runtime_context,
                    WorkflowStep.REASONING,
                    decision_reason="reasoning_inputs_unavailable",
                )
            }

        reasoning = _derive_reasoning_result(workflow_state)
        reasoned_prompt_input = workflow_state.prompt_input.model_copy(
            update={"creative_reasoning": reasoning}
        )
        reasoned_state = workflow_state.model_copy(
            update={
                "creative_reasoning": reasoning,
                "prompt_input": reasoned_prompt_input,
            }
        )
        _emit(
            runtime_context.event_builder.planning(
                code="creative_reasoning_prepared",
                message="Creative Reasoning Engine synthesis prepared.",
                creative_reasoning=reasoning.model_dump(mode="json"),
            ),
            workflow_state=reasoned_state,
            step=WorkflowStep.REASONING,
        )
        return {
            "workflow_state": _complete_node(
                reasoned_state,
                runtime_context,
                WorkflowStep.REASONING,
                decision_reason="creative_reasoning_prepared",
            )
        }
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.REASONING,
            exc=exc,
        )


def _prompt_rendering_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.PROMPT_RENDERING,
    )
    try:
        rendered_prompt = _emit_streaming_step(
            runtime_context.stream_rendered_prompt(
                builder=runtime_context.event_builder,
                decision=_route_decision(workflow_state),
                prompt_inputs=workflow_state.prompt_input,
            ),
            workflow_state=workflow_state,
        )
        if rendered_prompt is None:
            return {
                "workflow_state": _skip_node(
                    workflow_state,
                    runtime_context,
                    WorkflowStep.PROMPT_RENDERING,
                    decision_reason="prompt_rendering_unavailable",
                )
            }
        return {
            "workflow_state": _complete_node(
                workflow_state,
                runtime_context,
                WorkflowStep.PROMPT_RENDERING,
                decision_reason="prompt_rendered",
                rendered_prompt=rendered_prompt,
            )
        }
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.PROMPT_RENDERING,
            exc=exc,
        )


def _generation_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.GENERATION,
        allow_reentry=True,
    )
    try:
        generation_result = _emit_streaming_step(
            runtime_context.stream_generation(
                builder=runtime_context.event_builder,
                decision=_route_decision(workflow_state),
                rendered_prompt=workflow_state.rendered_prompt,
            ),
            workflow_state=workflow_state,
        )
        if generation_result is None:
            return {
                "workflow_state": _skip_node(
                    workflow_state,
                    runtime_context,
                    WorkflowStep.GENERATION,
                    decision_reason="generation_unavailable",
                )
            }
        generation_failure = _failure_info_from_generation_result(generation_result)
        if generation_failure is not None:
            _emit_node_failed(
                runtime_context,
                workflow_state,
                WorkflowStep.GENERATION,
                generation_failure,
                decision_reason="generation_provider_failed",
            )
            return {
                "workflow_state": complete_workflow_step(
                    workflow_state,
                    WorkflowStep.GENERATION,
                    error_message=generation_failure.message,
                    failure_info=generation_failure,
                ),
                "pending_failure": generation_failure,
                "failure_event_emitted": True,
                "generation_result": None,
            }
        return {
            "workflow_state": _complete_node(
                workflow_state,
                runtime_context,
                WorkflowStep.GENERATION,
                decision_reason="generation_completed",
            ),
            "generation_result": generation_result,
        }
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.GENERATION,
            exc=exc,
            clear_generation_result=True,
        )


def _artifact_extraction_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.ARTIFACT_EXTRACTION,
        allow_reentry=True,
    )
    try:
        artifacts = extract_workflow_artifacts(
            _answer_for_review(
                state=state,
                workflow_state=workflow_state,
                runtime=runtime_context,
            ),
            request=workflow_state.request,
            route_decision=_route_decision(workflow_state),
            creative_translation=(
                workflow_state.prompt_input.creative_translation
                if workflow_state.prompt_input is not None
                else None
            ),
            creative_plan=workflow_state.creative_plan,
        )
        if not artifacts:
            return {
                "workflow_state": _skip_node(
                    workflow_state.model_copy(
                        update={"artifacts": (), "preview_results": ()}
                    ),
                    runtime_context,
                    WorkflowStep.ARTIFACT_EXTRACTION,
                    decision_reason="no_generated_artifacts",
                )
            }

        _emit(
            runtime_context.event_builder.artifact_extracted(
                artifacts=artifacts,
                code="artifact_extracted",
                message=(
                    f"Extracted {len(artifacts)} generated artifact"
                    f"{'s' if len(artifacts) != 1 else ''} from the answer."
                ),
            ),
            workflow_state=workflow_state,
            step=WorkflowStep.ARTIFACT_EXTRACTION,
        )
        return {
            "workflow_state": _complete_node(
                workflow_state,
                runtime_context,
                WorkflowStep.ARTIFACT_EXTRACTION,
                decision_reason="artifacts_extracted",
                artifacts=artifacts,
                preview_results=(),
            )
        }
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.ARTIFACT_EXTRACTION,
            exc=exc,
        )


def _preview_preparation_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.PREVIEW_PREPARATION,
        allow_reentry=True,
    )
    try:
        if not workflow_state.artifacts:
            return {
                "workflow_state": _skip_node(
                    workflow_state.model_copy(update={"preview_results": ()}),
                    runtime_context,
                    WorkflowStep.PREVIEW_PREPARATION,
                    decision_reason="no_artifacts_for_preview",
                )
            }

        preview_results = prepare_workflow_preview_results(
            workflow_state.artifacts,
            request=workflow_state.request,
            route_decision=_route_decision(workflow_state),
        )
        if not preview_results:
            return {
                "workflow_state": _skip_node(
                    workflow_state.model_copy(update={"preview_results": ()}),
                    runtime_context,
                    WorkflowStep.PREVIEW_PREPARATION,
                    decision_reason="no_previewable_artifacts",
                )
            }

        for result in preview_results:
            _emit(
                runtime_context.event_builder.preview_artifact(
                    result,
                    code="preview_artifact_prepared",
                    message=result.summary
                    or "Preview runtime metadata prepared for the artifact.",
                ),
                workflow_state=workflow_state,
                step=WorkflowStep.PREVIEW_PREPARATION,
            )
        return {
            "workflow_state": _complete_node(
                workflow_state,
                runtime_context,
                WorkflowStep.PREVIEW_PREPARATION,
                decision_reason="preview_metadata_prepared",
                preview_results=preview_results,
            )
        }
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.PREVIEW_PREPARATION,
            exc=exc,
        )


def _artifact_critique_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.ARTIFACT_CRITIQUE,
        allow_reentry=True,
    )
    try:
        if not workflow_state.artifacts:
            return {
                "workflow_state": _skip_node(
                    workflow_state.model_copy(
                        update={"artifact_critique_summary": None}
                    ),
                    runtime_context,
                    WorkflowStep.ARTIFACT_CRITIQUE,
                    decision_reason="no_artifacts_for_critique",
                )
            }

        _emit_artifact_critique_started(runtime_context, workflow_state)
        artifacts, critique_summary = critique_workflow_artifacts(
            workflow_state.artifacts,
            request=workflow_state.request,
            route_decision=_route_decision(workflow_state),
            preview_results=workflow_state.preview_results,
        )
        refinement_passes = workflow_state.refinement_passes
        if workflow_state.refinement_count > 0 and refinement_passes:
            refinement_passes = complete_latest_refinement_pass(
                pass_history=refinement_passes,
                result_artifact=select_refinement_source(artifacts),
                max_passes=MAX_WORKFLOW_REFINEMENT_COUNT,
            )
            artifacts = attach_refinement_history(artifacts, refinement_passes)
        for critique in critique_summary.critiques:
            _emit_artifact_scored(runtime_context, workflow_state, critique)
        _emit_artifact_recommendation(runtime_context, workflow_state, critique_summary)
        if critique_summary.refinement_required:
            _emit_artifact_refinement_requested(
                runtime_context,
                workflow_state,
                critique_summary,
            )
        _emit_artifact_critique_completed(
            runtime_context,
            workflow_state,
            critique_summary,
        )
        return {
            "workflow_state": _complete_node(
                workflow_state,
                runtime_context,
                WorkflowStep.ARTIFACT_CRITIQUE,
                decision_reason=(
                    "artifact_critique_requested_refinement"
                    if critique_summary.refinement_required
                    else "artifact_critique_completed"
                ),
                artifacts=artifacts,
                artifact_critique_summary=critique_summary,
                refinement_passes=refinement_passes,
            )
        }
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.ARTIFACT_CRITIQUE,
            exc=exc,
        )


def _review_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.REVIEW,
        allow_reentry=True,
    )
    try:
        previous_review_result = workflow_state.review_result
        review_result = review_assistant_answer(
            request=workflow_state.request,
            answer=_answer_for_review(
                state=state,
                workflow_state=workflow_state,
                runtime=runtime_context,
            ),
            refinement_count=workflow_state.refinement_count,
            artifact_critique_summary=workflow_state.artifact_critique_summary,
        )
        transition_target, decision_reason = _review_transition(
            review_result,
            workflow_state,
        )
        _emit_review_outcome(
            runtime_context,
            workflow_state,
            review_result,
            transition_target=transition_target,
            decision_reason=decision_reason,
        )
        if _review_requests_retry(review_result, workflow_state):
            _emit_refinement_requested(
                runtime_context,
                workflow_state,
                review_result,
            )
            _emit_retry_started(
                runtime_context,
                workflow_state,
                review_result,
            )
        elif workflow_state.refinement_count > 0:
            _emit_retry_completed(
                runtime_context,
                workflow_state,
                review_result,
                previous_review_result,
                transition_target=transition_target,
                decision_reason=decision_reason,
            )
        return {
            "workflow_state": _complete_node(
                workflow_state,
                runtime_context,
                WorkflowStep.REVIEW,
                transition_target=transition_target,
                decision_reason=decision_reason,
                review_result=review_result,
            )
        }
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.REVIEW,
            exc=exc,
        )


def _refinement_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.REFINEMENT,
        allow_reentry=True,
    )
    review_result = workflow_state.review_result
    try:
        if review_result is None:
            raise ValueError("Workflow review result is not available for refinement.")

        pass_record: RefinementPassRecord | None = None
        source_artifact = select_refinement_source(workflow_state.artifacts)
        if source_artifact is not None:
            decision = plan_next_refinement_pass(
                source_artifact=source_artifact,
                pass_history=workflow_state.refinement_passes,
                max_passes=MAX_WORKFLOW_REFINEMENT_COUNT,
            )
            if not decision.should_continue:
                raise ValueError(
                    "Workflow refinement node entered after stop condition."
                )
            pass_record = start_refinement_pass_record(
                source_artifact=source_artifact,
                decision=decision,
            )
        refined_prompt = _append_refinement_guidance(
            rendered_prompt=workflow_state.rendered_prompt,
            review_result=review_result,
            artifact_critique_summary=workflow_state.artifact_critique_summary,
            refinement_pass=pass_record,
        )
        retry_count = workflow_state.refinement_count + 1
        _emit_refinement_completed(
            runtime_context,
            workflow_state,
            review_result,
            retry_count=retry_count,
        )
        return {
            "workflow_state": _complete_node(
                workflow_state,
                runtime_context,
                WorkflowStep.REFINEMENT,
                transition_target=WorkflowStep.GENERATION.value,
                decision_reason="refinement_completed",
                rendered_prompt=refined_prompt,
                refinement_count=retry_count,
                refinement_passes=(
                    (*workflow_state.refinement_passes, pass_record)
                    if pass_record is not None
                    else workflow_state.refinement_passes
                ),
            ),
            "generation_result": None,
        }
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.REFINEMENT,
            exc=exc,
            clear_generation_result=True,
        )


def _finalization_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.FINALIZATION,
    )
    try:
        generation_result = state.get("generation_result")
        if generation_result is not None:
            answer = generation_result.answer
        elif workflow_state.clarification is not None:
            answer = _format_clarification_answer(workflow_state.clarification)
        else:
            answer = runtime_context.build_shell_answer(_route_decision(workflow_state))
        telemetry = (
            getattr(generation_result, "telemetry", None)
            if generation_result is not None
            else None
        )

        self_evaluation = _derive_self_evaluation_result(
            workflow_state,
            generated_response=answer,
        )
        evaluation_state = workflow_state.model_copy(
            update={"self_evaluation": self_evaluation}
        )
        creative_improvement_planner = (
            _derive_creative_improvement_planner_result(
                evaluation_state,
                generated_response=answer,
            )
        )
        reflection_loop = _derive_reflection_loop_result(
            evaluation_state.model_copy(
                update={
                    "creative_improvement_planner": (
                        creative_improvement_planner
                    )
                }
            )
        )
        creative_confidence = _derive_creative_confidence_result(
            evaluation_state.model_copy(
                update={
                    "creative_improvement_planner": (
                        creative_improvement_planner
                    ),
                    "reflection_loop": reflection_loop,
                }
            )
        )
        evaluated_prompt_input = (
            workflow_state.prompt_input.model_copy(
                update={
                    "self_evaluation": self_evaluation,
                    "creative_improvement_planner": (
                        creative_improvement_planner
                    ),
                    "reflection_loop": reflection_loop,
                    "creative_confidence": creative_confidence,
                }
            )
            if workflow_state.prompt_input is not None
            else None
        )
        evaluated_state = workflow_state.model_copy(
            update={
                "self_evaluation": self_evaluation,
                "creative_improvement_planner": (
                    creative_improvement_planner
                ),
                "reflection_loop": reflection_loop,
                "creative_confidence": creative_confidence,
                "prompt_input": evaluated_prompt_input,
            }
        )
        updated_director = _derive_director_brief(evaluated_state)
        directed_prompt_input = (
            evaluated_prompt_input.model_copy(
                update={"creative_director": updated_director}
            )
            if evaluated_prompt_input is not None
            else None
        )
        directed_state = evaluated_state.model_copy(
            update={
                "creative_director": updated_director,
                "prompt_input": directed_prompt_input,
            }
        )
        if workflow_state.creative_reasoning is not None:
            updated_reasoning = _derive_reasoning_result(directed_state)
            reasoned_prompt_input = (
                directed_prompt_input.model_copy(
                    update={"creative_reasoning": updated_reasoning}
                )
                if directed_prompt_input is not None
                else None
            )
            directed_state = directed_state.model_copy(
                update={
                    "creative_reasoning": updated_reasoning,
                    "prompt_input": reasoned_prompt_input,
                }
            )
        final_state = finish_workflow(directed_state, final_answer=answer)
        _emit_node_completed(
            runtime_context,
            final_state,
            WorkflowStep.FINALIZATION,
            transition_target="end",
            decision_reason="final_answer_emitted",
            resolution="completed",
        )
        _emit(
            runtime_context.event_builder.final(
                answer=answer,
                route=state["route_payload"],
                artifacts=[
                    artifact.model_dump(mode="json")
                    for artifact in final_state.artifacts
                ],
                artifact_critiques=[
                    critique.model_dump(mode="json")
                    for critique in (
                        final_state.artifact_critique_summary.critiques
                        if final_state.artifact_critique_summary is not None
                        else ()
                    )
                ],
                artifact_critique_summary=(
                    final_state.artifact_critique_summary.model_dump(mode="json")
                    if final_state.artifact_critique_summary is not None
                    else None
                ),
                preview_results=[
                    result.model_dump(mode="json")
                    for result in final_state.preview_results
                ],
                **_optional_event_payload(
                    "clarification",
                    (
                        final_state.clarification.model_dump(mode="json")
                        if final_state.clarification is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "creative_intent",
                    (
                        final_state.creative_intent.model_dump(mode="json")
                        if final_state.creative_intent is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "creative_hierarchy",
                    (
                        final_state.creative_hierarchy.model_dump(mode="json")
                        if final_state.creative_hierarchy is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "creative_strategy",
                    (
                        final_state.creative_strategy.model_dump(mode="json")
                        if final_state.creative_strategy is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "creative_techniques",
                    (
                        final_state.creative_techniques.model_dump(mode="json")
                        if final_state.creative_techniques is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "creative_plan",
                    (
                        final_state.creative_plan.model_dump(mode="json")
                        if final_state.creative_plan is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "creative_constraints",
                    (
                        final_state.creative_constraints.model_dump(mode="json")
                        if final_state.creative_constraints is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "creative_constraint_priorities",
                    (
                        final_state.creative_constraint_priorities.model_dump(
                            mode="json"
                        )
                        if final_state.creative_constraint_priorities is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "runtime_capabilities",
                    (
                        final_state.runtime_capabilities.model_dump(mode="json")
                        if final_state.runtime_capabilities is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "creative_tradeoffs",
                    (
                        final_state.creative_tradeoffs.model_dump(mode="json")
                        if final_state.creative_tradeoffs is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "creative_quality_prediction",
                    (
                        final_state.creative_quality_prediction.model_dump(
                            mode="json"
                        )
                        if final_state.creative_quality_prediction is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "symbolic_narrative",
                    (
                        final_state.symbolic_narrative.model_dump(mode="json")
                        if final_state.symbolic_narrative is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "creative_composition",
                    (
                        final_state.creative_composition.model_dump(mode="json")
                        if final_state.creative_composition is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "procedural_structure",
                    (
                        final_state.procedural_structure.model_dump(mode="json")
                        if final_state.procedural_structure is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "generative_structure",
                    (
                        final_state.generative_structure.model_dump(mode="json")
                        if final_state.generative_structure is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "semantic_motif",
                    (
                        final_state.semantic_motif.model_dump(mode="json")
                        if final_state.semantic_motif is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "emotional_consistency",
                    (
                        final_state.emotional_consistency.model_dump(mode="json")
                        if final_state.emotional_consistency is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "cross_modality",
                    (
                        final_state.cross_modality.model_dump(mode="json")
                        if final_state.cross_modality is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "audio_visual_scene",
                    (
                        final_state.audio_visual_scene.model_dump(mode="json")
                        if final_state.audio_visual_scene is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "artifact_plan",
                    (
                        final_state.artifact_plan.model_dump(mode="json")
                        if final_state.artifact_plan is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "artifact_dependency_graph",
                    (
                        final_state.artifact_dependency_graph.model_dump(mode="json")
                        if final_state.artifact_dependency_graph is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "runtime_compatibility",
                    (
                        final_state.runtime_compatibility.model_dump(mode="json")
                        if final_state.runtime_compatibility is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "artifact_capability_matrix",
                    (
                        final_state.artifact_capability_matrix.model_dump(mode="json")
                        if final_state.artifact_capability_matrix is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "multi_artifact_strategy",
                    (
                        final_state.multi_artifact_strategy.model_dump(mode="json")
                        if final_state.multi_artifact_strategy is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "artifact_critic",
                    (
                        final_state.artifact_critic.model_dump(mode="json")
                        if final_state.artifact_critic is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "artifact_refiner",
                    (
                        final_state.artifact_refiner.model_dump(mode="json")
                        if final_state.artifact_refiner is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "artifact_intelligence_synthesis",
                    (
                        final_state.artifact_intelligence_synthesis.model_dump(
                            mode="json"
                        )
                        if final_state.artifact_intelligence_synthesis is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "artifact_merge_planner",
                    (
                        final_state.artifact_merge_planner.model_dump(mode="json")
                        if final_state.artifact_merge_planner is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "artifact_export_intelligence",
                    (
                        final_state.artifact_export_intelligence.model_dump(
                            mode="json"
                        )
                        if final_state.artifact_export_intelligence is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "artifact_engine_contracts",
                    (
                        final_state.artifact_engine_contracts.model_dump(
                            mode="json"
                        )
                        if final_state.artifact_engine_contracts is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "creative_critic",
                    (
                        final_state.creative_critic.model_dump(mode="json")
                        if final_state.creative_critic is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "self_evaluation",
                    (
                        final_state.self_evaluation.model_dump(mode="json")
                        if final_state.self_evaluation is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "creative_improvement_planner",
                    (
                        final_state.creative_improvement_planner.model_dump(
                            mode="json"
                        )
                        if final_state.creative_improvement_planner is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "reflection_loop",
                    (
                        final_state.reflection_loop.model_dump(mode="json")
                        if final_state.reflection_loop is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "creative_confidence",
                    (
                        final_state.creative_confidence.model_dump(mode="json")
                        if final_state.creative_confidence is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "creative_director",
                    (
                        final_state.creative_director.model_dump(mode="json")
                        if final_state.creative_director is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "creative_reasoning",
                    (
                        final_state.creative_reasoning.model_dump(mode="json")
                        if final_state.creative_reasoning is not None
                        else None
                    ),
                ),
                **_optional_event_payload(
                    "observability",
                    runtime_context.observability.event_payload(
                        runtime_context.observability_run,
                        lineage={"stage": WorkflowStep.FINALIZATION.value},
                    ),
                ),
                **({"telemetry": telemetry} if telemetry is not None else {}),
            ),
            workflow_state=final_state,
            step=WorkflowStep.FINALIZATION,
            phase="completed",
        )
        return {"workflow_state": final_state}
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.FINALIZATION,
            exc=exc,
            clear_generation_result=True,
        )


def _failure_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.FAILURE,
    )
    failure_info = _pending_failure_info(state, workflow_state)
    if not state.get("failure_event_emitted", False):
        _emit(
            runtime_context.event_builder.error(
                code=failure_info.code,
                message=failure_info.message,
            ),
            workflow_state=workflow_state,
            step=WorkflowStep.FAILURE,
            phase="failed",
        )
    answer = _failure_answer(failure_info)
    final_state = fail_workflow(
        workflow_state,
        error_message=failure_info.message,
        failure_info=failure_info,
        final_answer=answer,
    )
    _emit_node_completed(
        runtime_context,
        final_state,
        WorkflowStep.FAILURE,
        transition_target="end",
        decision_reason="terminal_failure_answer_emitted",
        resolution="completed",
    )
    _emit(
        runtime_context.event_builder.final(
            answer=answer,
            route=state.get("route_payload"),
            **_optional_event_payload(
                "observability",
                runtime_context.observability.event_payload(
                    runtime_context.observability_run,
                    lineage={"stage": WorkflowStep.FAILURE.value},
                ),
            ),
        ),
        workflow_state=final_state,
        step=WorkflowStep.FAILURE,
        phase="failed",
    )
    return {
        "workflow_state": final_state,
        "pending_failure": None,
        "failure_event_emitted": True,
        "generation_result": None,
    }


def _next_node_after_review(state: AssistantWorkflowGraphState) -> str:
    if _has_pending_failure(state):
        return "failure"
    workflow_state = _workflow_state(state)
    review_result = workflow_state.review_result
    if review_result is None:
        raise ValueError("Workflow review result is not available.")
    if _review_requests_retry(review_result, workflow_state):
        return "refinement"
    return "finalization"


def _next_node_after_finalization(state: AssistantWorkflowGraphState) -> str:
    if _has_pending_failure(state):
        return "failure"
    return "end"


def _next_node_or_failure(
    state: AssistantWorkflowGraphState,
    next_node: str,
) -> str:
    if _has_pending_failure(state):
        return "failure"
    return next_node


def _next_node_after_prompt_input(state: AssistantWorkflowGraphState) -> str:
    if _has_pending_failure(state):
        return "failure"
    if _workflow_state(state).clarification is not None:
        return WorkflowStep.FINALIZATION.value
    return WorkflowStep.PLANNING.value


def _review_transition(
    review_result: WorkflowReviewResult,
    workflow_state: AssistantWorkflowState,
) -> tuple[str, str]:
    if _review_requests_retry(review_result, workflow_state):
        return WorkflowStep.REFINEMENT.value, "review_failed_retry_available"
    if review_result.outcome is WorkflowReviewOutcome.NEEDS_REFINEMENT:
        return WorkflowStep.FINALIZATION.value, "review_failed_retry_limit_reached"
    return WorkflowStep.FINALIZATION.value, "review_passed"


def _review_requests_retry(
    review_result: WorkflowReviewResult,
    workflow_state: AssistantWorkflowState,
) -> bool:
    source_artifact = select_refinement_source(workflow_state.artifacts)
    if source_artifact is None:
        return (
            review_result.outcome is WorkflowReviewOutcome.NEEDS_REFINEMENT
            and workflow_state.refinement_count < MAX_WORKFLOW_REFINEMENT_COUNT
        )
    decision = plan_next_refinement_pass(
        source_artifact=source_artifact,
        pass_history=workflow_state.refinement_passes,
        max_passes=MAX_WORKFLOW_REFINEMENT_COUNT,
    )
    return (
        review_result.outcome is WorkflowReviewOutcome.NEEDS_REFINEMENT
        and workflow_state.refinement_count < MAX_WORKFLOW_REFINEMENT_COUNT
        and decision.should_continue
    )


def _start_node(
    state: AssistantWorkflowState,
    runtime: AssistantWorkflowRuntime,
    step: WorkflowStep,
    *,
    allow_reentry: bool = False,
) -> AssistantWorkflowState:
    workflow_state = _start_graph_workflow_step(
        state,
        step,
        allow_reentry=allow_reentry,
    )
    _emit_node_started(runtime, workflow_state, step)
    return workflow_state


def _complete_node(
    workflow_state: AssistantWorkflowState,
    runtime: AssistantWorkflowRuntime,
    step: WorkflowStep,
    *,
    transition_target: str | None = None,
    decision_reason: str = "node_completed",
    resolution: str = "completed",
    **updates: object,
) -> AssistantWorkflowState:
    completed_state = complete_workflow_step(workflow_state, step, **updates)
    _emit_node_completed(
        runtime,
        completed_state,
        step,
        transition_target=transition_target,
        decision_reason=decision_reason,
        resolution=resolution,
    )
    return completed_state


def _skip_node(
    workflow_state: AssistantWorkflowState,
    runtime: AssistantWorkflowRuntime,
    step: WorkflowStep,
    *,
    transition_target: str | None = None,
    decision_reason: str = "node_skipped",
) -> AssistantWorkflowState:
    skipped_state = skip_workflow_step(workflow_state, step)
    _emit_node_completed(
        runtime,
        skipped_state,
        step,
        transition_target=transition_target,
        decision_reason=decision_reason,
        resolution="skipped",
    )
    return skipped_state


def _emit_node_started(
    runtime: AssistantWorkflowRuntime,
    workflow_state: AssistantWorkflowState,
    step: WorkflowStep,
) -> None:
    _emit(
        runtime.event_builder.node_started(
            node=step.value,
            node_label=_step_label(step),
            message=f"{_step_label(step)} started.",
            attempt_count=_node_attempt_count(workflow_state, step),
        ),
        workflow_state=workflow_state,
        step=step,
    )


def _emit_node_completed(
    runtime: AssistantWorkflowRuntime,
    workflow_state: AssistantWorkflowState,
    step: WorkflowStep,
    *,
    transition_target: str | None,
    decision_reason: str,
    resolution: str,
) -> None:
    target = transition_target or _default_transition_target(step)
    _emit(
        runtime.event_builder.node_completed(
            node=step.value,
            node_label=_step_label(step),
            message=f"{_step_label(step)} {resolution}.",
            resolution=resolution,
            **_transition_payload(step.value, target, decision_reason),
        ),
        workflow_state=workflow_state,
        step=step,
        phase="completed",
    )


def _emit_node_failed(
    runtime: AssistantWorkflowRuntime,
    workflow_state: AssistantWorkflowState,
    step: WorkflowStep,
    failure_info: WorkflowFailureInfo,
    *,
    transition_target: str = "failure",
    decision_reason: str = "node_failed",
) -> None:
    _emit(
        runtime.event_builder.node_failed(
            node=step.value,
            node_label=_step_label(step),
            message=f"{_step_label(step)} failed: {failure_info.message}",
            error_code=failure_info.code,
            error_message=failure_info.message,
            **_transition_payload(step.value, transition_target, decision_reason),
        ),
        workflow_state=workflow_state,
        step=step,
        phase="failed",
    )


def _emit_review_outcome(
    runtime: AssistantWorkflowRuntime,
    workflow_state: AssistantWorkflowState,
    review_result: WorkflowReviewResult,
    *,
    transition_target: str,
    decision_reason: str,
) -> None:
    payload = {
        "score": review_result.score,
        "rationale": review_result.rationale,
        "review": review_result.model_dump(mode="json"),
        "review_outcome": review_result.outcome.value,
        "review_reasons": list(review_result.reasons),
        "refinement_count": review_result.refinement_count,
        **_transition_payload(
            WorkflowStep.REVIEW.value,
            transition_target,
            decision_reason,
        ),
    }
    if review_result.passed:
        event = runtime.event_builder.review_passed(
            message=review_result.rationale,
            **payload,
        )
    else:
        event = runtime.event_builder.review_failed(
            message=review_result.rationale,
            **payload,
        )
    _emit(event, workflow_state=workflow_state, step=WorkflowStep.REVIEW)


def _emit_artifact_critique_started(
    runtime: AssistantWorkflowRuntime,
    workflow_state: AssistantWorkflowState,
) -> None:
    _emit(
        runtime.event_builder.artifact_critique(
            code="critique_started",
            message=(
                f"Critiquing {len(workflow_state.artifacts)} generated artifact"
                f"{'s' if len(workflow_state.artifacts) != 1 else ''}."
            ),
            artifact_count=len(workflow_state.artifacts),
        ),
        workflow_state=workflow_state,
        step=WorkflowStep.ARTIFACT_CRITIQUE,
    )


def _emit_artifact_scored(
    runtime: AssistantWorkflowRuntime,
    workflow_state: AssistantWorkflowState,
    critique: WorkflowArtifactCritique,
) -> None:
    _emit(
        runtime.event_builder.artifact_critique(
            code="artifact_scored",
            message=(
                f"Scored {critique.artifact_title} at "
                f"{critique.overall_score:.2f}."
            ),
            artifact_id=critique.artifact_id,
            artifact_title=critique.artifact_title,
            score=critique.overall_score,
            rank=critique.rank,
            passed=critique.passed,
            critique=critique.model_dump(mode="json"),
        ),
        workflow_state=workflow_state,
        step=WorkflowStep.ARTIFACT_CRITIQUE,
    )


def _emit_artifact_recommendation(
    runtime: AssistantWorkflowRuntime,
    workflow_state: AssistantWorkflowState,
    critique_summary: ArtifactCritiqueSummary,
) -> None:
    if not critique_summary.recommended_artifact_id:
        return
    _emit(
        runtime.event_builder.artifact_critique(
            code="artifact_selected_recommended",
            message=(
                f"Recommended {critique_summary.recommended_artifact_title} "
                "as the strongest artifact candidate."
            ),
            recommended_artifact_id=critique_summary.recommended_artifact_id,
            recommended_artifact_title=critique_summary.recommended_artifact_title,
            average_score=critique_summary.average_score,
            failed_artifact_count=critique_summary.failed_artifact_count,
            critique_summary=critique_summary.model_dump(mode="json"),
        ),
        workflow_state=workflow_state,
        step=WorkflowStep.ARTIFACT_CRITIQUE,
    )


def _emit_artifact_refinement_requested(
    runtime: AssistantWorkflowRuntime,
    workflow_state: AssistantWorkflowState,
    critique_summary: ArtifactCritiqueSummary,
) -> None:
    _emit(
        runtime.event_builder.artifact_critique(
            code="artifact_refinement_requested",
            message=(
                critique_summary.refinement_guidance
                or "Artifact critique requested refinement."
            ),
            recommended_artifact_id=critique_summary.recommended_artifact_id,
            refinement_reasons=list(critique_summary.refinement_reasons),
            refinement_guidance=critique_summary.refinement_guidance,
            critique_summary=critique_summary.model_dump(mode="json"),
            **_transition_payload(
                WorkflowStep.ARTIFACT_CRITIQUE.value,
                WorkflowStep.REVIEW.value,
                "artifact_critique_requested_refinement",
            ),
        ),
        workflow_state=workflow_state,
        step=WorkflowStep.ARTIFACT_CRITIQUE,
    )


def _emit_artifact_critique_completed(
    runtime: AssistantWorkflowRuntime,
    workflow_state: AssistantWorkflowState,
    critique_summary: ArtifactCritiqueSummary,
) -> None:
    _emit(
        runtime.event_builder.artifact_critique(
            code="critique_completed",
            message=(
                "Artifact critique completed; "
                f"recommended {critique_summary.recommended_artifact_title}."
            ),
            critique_summary=critique_summary.model_dump(mode="json"),
            recommended_artifact_id=critique_summary.recommended_artifact_id,
            average_score=critique_summary.average_score,
            refinement_required=critique_summary.refinement_required,
        ),
        workflow_state=workflow_state,
        step=WorkflowStep.ARTIFACT_CRITIQUE,
    )


def _emit_refinement_requested(
    runtime: AssistantWorkflowRuntime,
    workflow_state: AssistantWorkflowState,
    review_result: WorkflowReviewResult,
) -> None:
    retry_count = workflow_state.refinement_count + 1
    reason = _review_reason_text(review_result)
    _emit(
        runtime.event_builder.refinement_requested(
            message=f"Refinement requested for retry {retry_count}: {reason}.",
            retry_count=retry_count,
            retry_reason=reason,
            review=review_result.model_dump(mode="json"),
            **_transition_payload(
                WorkflowStep.REVIEW.value,
                WorkflowStep.REFINEMENT.value,
                "review_failed_retry_available",
            ),
        ),
        workflow_state=workflow_state,
        step=WorkflowStep.REVIEW,
    )


def _emit_refinement_completed(
    runtime: AssistantWorkflowRuntime,
    workflow_state: AssistantWorkflowState,
    review_result: WorkflowReviewResult,
    *,
    retry_count: int,
) -> None:
    reason = _review_reason_text(review_result)
    _emit(
        runtime.event_builder.refinement_completed(
            message=f"Refinement guidance prepared for retry {retry_count}.",
            retry_count=retry_count,
            retry_reason=reason,
            review=review_result.model_dump(mode="json"),
            **_transition_payload(
                WorkflowStep.REFINEMENT.value,
                WorkflowStep.GENERATION.value,
                "refinement_completed",
            ),
        ),
        workflow_state=workflow_state,
        step=WorkflowStep.REFINEMENT,
    )


def _emit_retry_started(
    runtime: AssistantWorkflowRuntime,
    workflow_state: AssistantWorkflowState,
    review_result: WorkflowReviewResult,
) -> None:
    retry_count = workflow_state.refinement_count + 1
    reason = _review_reason_text(review_result)
    _emit(
        runtime.event_builder.retry_started(
            message=f"Retry {retry_count} started: {reason}.",
            retry_count=retry_count,
            retry_reason=reason,
            review=review_result.model_dump(mode="json"),
            **_transition_payload(
                WorkflowStep.REVIEW.value,
                WorkflowStep.REFINEMENT.value,
                "review_failed_retry_available",
            ),
        ),
        workflow_state=workflow_state,
        step=WorkflowStep.REVIEW,
    )


def _emit_retry_completed(
    runtime: AssistantWorkflowRuntime,
    workflow_state: AssistantWorkflowState,
    review_result: WorkflowReviewResult,
    previous_review_result: WorkflowReviewResult | None,
    *,
    transition_target: str,
    decision_reason: str,
) -> None:
    retry_count = workflow_state.refinement_count
    reason = _review_reason_text(previous_review_result or review_result)
    status = "passed" if review_result.passed else "exhausted"
    _emit(
        runtime.event_builder.retry_completed(
            message=f"Retry {retry_count} {status}: {review_result.rationale}",
            retry_count=retry_count,
            retry_reason=reason,
            retry_status=status,
            review=review_result.model_dump(mode="json"),
            **_transition_payload(
                WorkflowStep.REVIEW.value,
                transition_target,
                decision_reason,
            ),
        ),
        workflow_state=workflow_state,
        step=WorkflowStep.REVIEW,
    )


def _start_graph_workflow_step(
    state: AssistantWorkflowState,
    step: WorkflowStep,
    *,
    allow_reentry: bool = False,
) -> AssistantWorkflowState:
    if allow_reentry and (step in state.completed_steps or step in state.skipped_steps):
        return restart_workflow_step(state, step)
    return start_workflow_step(state, step)


def _answer_for_review(
    *,
    state: AssistantWorkflowGraphState,
    workflow_state: AssistantWorkflowState,
    runtime: AssistantWorkflowRuntime,
) -> str:
    generation_result = state.get("generation_result")
    if generation_result is not None:
        return generation_result.answer
    return runtime.build_shell_answer(_route_decision(workflow_state))


def _derive_director_brief(
    workflow_state: AssistantWorkflowState,
) -> CreativeAssistantDirectorBrief:
    prompt_input = workflow_state.prompt_input
    return derive_creative_assistant_director_brief(
        request=workflow_state.request,
        route_decision=workflow_state.route_decision,
        creative_translation=(
            prompt_input.creative_translation if prompt_input is not None else None
        ),
        creative_intent=workflow_state.creative_intent,
        creative_hierarchy=workflow_state.creative_hierarchy,
        creative_strategy=workflow_state.creative_strategy,
        creative_techniques=workflow_state.creative_techniques,
        creative_plan=workflow_state.creative_plan,
        creative_constraints=workflow_state.creative_constraints,
        creative_constraint_priorities=workflow_state.creative_constraint_priorities,
        runtime_capabilities=workflow_state.runtime_capabilities,
        creative_tradeoffs=workflow_state.creative_tradeoffs,
        creative_quality_prediction=workflow_state.creative_quality_prediction,
        symbolic_narrative=workflow_state.symbolic_narrative,
        creative_composition=workflow_state.creative_composition,
        procedural_structure=workflow_state.procedural_structure,
        generative_structure=workflow_state.generative_structure,
        semantic_motif=workflow_state.semantic_motif,
        emotional_consistency=workflow_state.emotional_consistency,
        cross_modality=workflow_state.cross_modality,
        audio_visual_scene=workflow_state.audio_visual_scene,
        artifact_plan=workflow_state.artifact_plan,
        artifact_dependency_graph=workflow_state.artifact_dependency_graph,
        runtime_compatibility=workflow_state.runtime_compatibility,
        artifact_capability_matrix=workflow_state.artifact_capability_matrix,
        multi_artifact_strategy=workflow_state.multi_artifact_strategy,
        artifact_critic=workflow_state.artifact_critic,
        artifact_refiner=workflow_state.artifact_refiner,
        artifact_intelligence_synthesis=(
            workflow_state.artifact_intelligence_synthesis
        ),
        artifact_merge_planner=workflow_state.artifact_merge_planner,
        artifact_export_intelligence=workflow_state.artifact_export_intelligence,
        creative_critic=workflow_state.creative_critic,
        self_evaluation=workflow_state.self_evaluation,
        creative_improvement_planner=(
            workflow_state.creative_improvement_planner
        ),
        reflection_loop=workflow_state.reflection_loop,
        creative_confidence=workflow_state.creative_confidence,
        clarification=workflow_state.clarification,
        retrieval_chunk_count=(
            len(prompt_input.retrieval_input.chunks)
            if prompt_input is not None and prompt_input.retrieval_input is not None
            else 0
        ),
        artifact_critique_summary=workflow_state.artifact_critique_summary,
        review_result=workflow_state.review_result,
        refinement_count=workflow_state.refinement_count,
    )


def _derive_reasoning_result(
    workflow_state: AssistantWorkflowState,
) -> CreativeReasoningResult:
    prompt_input = workflow_state.prompt_input
    return derive_creative_reasoning_result(
        request=workflow_state.request,
        route_decision=workflow_state.route_decision,
        creative_translation=(
            prompt_input.creative_translation if prompt_input is not None else None
        ),
        creative_intent=workflow_state.creative_intent,
        creative_hierarchy=workflow_state.creative_hierarchy,
        creative_plan=workflow_state.creative_plan,
        creative_director=workflow_state.creative_director,
        creative_constraints=workflow_state.creative_constraints,
        creative_constraint_priorities=workflow_state.creative_constraint_priorities,
        creative_strategy=workflow_state.creative_strategy,
        creative_techniques=workflow_state.creative_techniques,
        runtime_capabilities=workflow_state.runtime_capabilities,
        creative_tradeoffs=workflow_state.creative_tradeoffs,
        creative_quality_prediction=workflow_state.creative_quality_prediction,
        symbolic_narrative=workflow_state.symbolic_narrative,
        creative_composition=workflow_state.creative_composition,
        procedural_structure=workflow_state.procedural_structure,
        generative_structure=workflow_state.generative_structure,
        semantic_motif=workflow_state.semantic_motif,
        emotional_consistency=workflow_state.emotional_consistency,
        cross_modality=workflow_state.cross_modality,
        audio_visual_scene=workflow_state.audio_visual_scene,
        artifact_plan=workflow_state.artifact_plan,
        artifact_dependency_graph=workflow_state.artifact_dependency_graph,
        runtime_compatibility=workflow_state.runtime_compatibility,
        artifact_capability_matrix=workflow_state.artifact_capability_matrix,
        multi_artifact_strategy=workflow_state.multi_artifact_strategy,
        artifact_critic=workflow_state.artifact_critic,
        artifact_refiner=workflow_state.artifact_refiner,
        artifact_intelligence_synthesis=(
            workflow_state.artifact_intelligence_synthesis
        ),
        artifact_merge_planner=workflow_state.artifact_merge_planner,
        artifact_export_intelligence=workflow_state.artifact_export_intelligence,
        creative_critic=workflow_state.creative_critic,
        self_evaluation=workflow_state.self_evaluation,
        creative_improvement_planner=(
            workflow_state.creative_improvement_planner
        ),
        reflection_loop=workflow_state.reflection_loop,
        creative_confidence=workflow_state.creative_confidence,
    )


def _derive_self_evaluation_result(
    workflow_state: AssistantWorkflowState,
    *,
    generated_response: str | None = None,
):
    prompt_input = workflow_state.prompt_input
    return derive_self_evaluation_profile(
        request=workflow_state.request,
        route_decision=workflow_state.route_decision,
        creative_translation=(
            prompt_input.creative_translation if prompt_input is not None else None
        ),
        creative_intent=workflow_state.creative_intent,
        creative_hierarchy=workflow_state.creative_hierarchy,
        creative_plan=workflow_state.creative_plan,
        creative_constraints=workflow_state.creative_constraints,
        creative_constraint_priorities=workflow_state.creative_constraint_priorities,
        creative_strategy=workflow_state.creative_strategy,
        creative_techniques=workflow_state.creative_techniques,
        runtime_capabilities=workflow_state.runtime_capabilities,
        creative_tradeoffs=workflow_state.creative_tradeoffs,
        creative_quality_prediction=workflow_state.creative_quality_prediction,
        symbolic_narrative=workflow_state.symbolic_narrative,
        creative_composition=workflow_state.creative_composition,
        procedural_structure=workflow_state.procedural_structure,
        generative_structure=workflow_state.generative_structure,
        semantic_motif=workflow_state.semantic_motif,
        emotional_consistency=workflow_state.emotional_consistency,
        cross_modality=workflow_state.cross_modality,
        audio_visual_scene=workflow_state.audio_visual_scene,
        artifact_plan=workflow_state.artifact_plan,
        artifact_dependency_graph=workflow_state.artifact_dependency_graph,
        runtime_compatibility=workflow_state.runtime_compatibility,
        artifact_capability_matrix=workflow_state.artifact_capability_matrix,
        multi_artifact_strategy=workflow_state.multi_artifact_strategy,
        artifact_critic=workflow_state.artifact_critic,
        artifact_refiner=workflow_state.artifact_refiner,
        artifact_intelligence_synthesis=(
            workflow_state.artifact_intelligence_synthesis
        ),
        artifact_merge_planner=workflow_state.artifact_merge_planner,
        artifact_export_intelligence=workflow_state.artifact_export_intelligence,
        creative_critic=workflow_state.creative_critic,
        generated_response=generated_response,
        artifacts=workflow_state.artifacts,
    )


def _derive_creative_improvement_planner_result(
    workflow_state: AssistantWorkflowState,
    *,
    generated_response: str | None = None,
):
    return derive_creative_improvement_planner_profile(
        request=workflow_state.request,
        route_decision=workflow_state.route_decision,
        creative_critic=workflow_state.creative_critic,
        self_evaluation=workflow_state.self_evaluation,
        generated_response=generated_response,
        artifacts=workflow_state.artifacts,
    )


def _derive_reflection_loop_result(workflow_state: AssistantWorkflowState):
    return derive_reflection_loop_profile(
        request=workflow_state.request,
        route_decision=workflow_state.route_decision,
        creative_critic=workflow_state.creative_critic,
        self_evaluation=workflow_state.self_evaluation,
        creative_improvement_planner=workflow_state.creative_improvement_planner,
        planning_metadata=_reflection_planning_metadata(workflow_state),
    )


def _derive_creative_confidence_result(workflow_state: AssistantWorkflowState):
    return derive_creative_confidence_profile(
        request=workflow_state.request,
        route_decision=workflow_state.route_decision,
        creative_critic=workflow_state.creative_critic,
        self_evaluation=workflow_state.self_evaluation,
        creative_improvement_planner=workflow_state.creative_improvement_planner,
        reflection_loop=workflow_state.reflection_loop,
        planning_metadata=_reflection_planning_metadata(workflow_state),
    )


def _reflection_planning_metadata(
    workflow_state: AssistantWorkflowState,
) -> tuple[object, ...]:
    return tuple(
        item
        for item in (
            workflow_state.creative_intent,
            workflow_state.creative_hierarchy,
            workflow_state.creative_strategy,
            workflow_state.creative_techniques,
            workflow_state.creative_plan,
            workflow_state.creative_constraints,
            workflow_state.creative_constraint_priorities,
            workflow_state.runtime_capabilities,
            workflow_state.creative_tradeoffs,
            workflow_state.creative_quality_prediction,
            workflow_state.symbolic_narrative,
            workflow_state.creative_composition,
            workflow_state.procedural_structure,
            workflow_state.generative_structure,
            workflow_state.semantic_motif,
            workflow_state.emotional_consistency,
            workflow_state.cross_modality,
            workflow_state.audio_visual_scene,
            workflow_state.artifact_plan,
            workflow_state.artifact_dependency_graph,
            workflow_state.runtime_compatibility,
            workflow_state.artifact_capability_matrix,
            workflow_state.multi_artifact_strategy,
            workflow_state.artifact_critic,
            workflow_state.artifact_refiner,
            workflow_state.artifact_intelligence_synthesis,
            workflow_state.artifact_merge_planner,
            workflow_state.artifact_export_intelligence,
        )
        if item is not None
    )


def _failure_info_from_generation_result(
    generation_result: object,
) -> WorkflowFailureInfo | None:
    error_code = getattr(generation_result, "error_code", None)
    error_message = getattr(generation_result, "error_message", None)
    if not error_code or not error_message:
        return None
    return WorkflowFailureInfo(
        step=WorkflowStep.GENERATION,
        code=str(error_code),
        message=str(error_message),
    )


def _append_refinement_guidance(
    *,
    rendered_prompt: RenderedPromptResponse | None,
    review_result: WorkflowReviewResult,
    artifact_critique_summary: ArtifactCritiqueSummary | None = None,
    refinement_pass: RefinementPassRecord | None = None,
) -> RenderedPromptResponse | None:
    if rendered_prompt is None:
        return None
    reasons = ", ".join(review_result.reasons) or "quality gate did not pass"
    artifact_guidance = (
        "\n- Artifact critique guidance: "
        f"{artifact_critique_summary.refinement_guidance}"
        if artifact_critique_summary
        and artifact_critique_summary.refinement_guidance
        else ""
    )
    pass_source = (
        refinement_pass.source_artifact_title
        or refinement_pass.source_artifact_id
        if refinement_pass is not None
        else None
    )
    pass_guidance = (
        "\n"
        f"- Refinement pass: {refinement_pass.pass_number}.\n"
        f"- Source artifact: {pass_source}.\n"
        f"- Pass objective: {refinement_pass.refinement_objective}"
        if refinement_pass is not None
        else ""
    )
    refinement_section = RenderedPromptSection(
        role=RenderedPromptRole.SYSTEM,
        name=RenderedPromptSectionName.SYSTEM,
        content=(
            "Refinement guidance:\n"
            "- Revise the previous answer before finalization.\n"
            f"- Address review issue(s): {reasons}.\n"
            "- Preserve the original user request and existing context."
            f"{artifact_guidance}"
            f"{pass_guidance}"
        ),
    )
    return rendered_prompt.model_copy(
        update={"sections": (*rendered_prompt.sections, refinement_section)}
    )


def _handle_workflow_exception(
    *,
    workflow_state: AssistantWorkflowState,
    runtime: AssistantWorkflowRuntime,
    step: WorkflowStep,
    exc: Exception,
    clear_generation_result: bool = False,
) -> AssistantWorkflowGraphState:
    failure_info = WorkflowFailureInfo(
        step=step,
        code=f"workflow_{step.value}_failed",
        message=str(exc) or f"{step.value} failed.",
    )
    logger.bind(
        step=step.value,
        error_code=failure_info.code,
        error_type=type(exc).__name__,
    ).exception(
        "assistant_workflow_step_failed: {}: {}",
        type(exc).__name__,
        exc,
    )
    _emit_node_failed(
        runtime,
        workflow_state,
        step,
        failure_info,
        decision_reason="node_exception",
    )
    _emit(
        runtime.event_builder.error(
            code=failure_info.code,
            message=failure_info.message,
        ),
        workflow_state=workflow_state,
        step=step,
        phase="failed",
    )
    update: AssistantWorkflowGraphState = {
        "workflow_state": workflow_state.model_copy(
            update={
                "current_step": None,
                "error_message": failure_info.message,
                "failure_info": failure_info,
            }
        ),
        "pending_failure": failure_info,
        "failure_event_emitted": True,
    }
    if clear_generation_result:
        update["generation_result"] = None
    return update


def _pending_failure_info(
    state: AssistantWorkflowGraphState,
    workflow_state: AssistantWorkflowState,
) -> WorkflowFailureInfo:
    pending_failure = state.get("pending_failure")
    if pending_failure is not None:
        return pending_failure
    if workflow_state.failure_info is not None:
        return workflow_state.failure_info
    raise ValueError("Workflow failure info is not available.")


def _failure_answer(failure_info: WorkflowFailureInfo) -> str:
    if failure_info.step is WorkflowStep.GENERATION:
        return f"Generation failed ({failure_info.code}): {failure_info.message}"
    return (
        "Workflow failed during "
        f"{failure_info.step.value} ({failure_info.code}): "
        f"{failure_info.message}"
    )


def _has_pending_failure(state: AssistantWorkflowGraphState) -> bool:
    return state.get("pending_failure") is not None


def _emit_streaming_step(
    step: Iterator[object],
    *,
    workflow_state: AssistantWorkflowState,
) -> object:
    while True:
        try:
            item = next(step)
        except StopIteration as exc:
            return exc.value
        if isinstance(item, StreamEvent):
            _emit(item, workflow_state=workflow_state)


def _emit(
    event: StreamEvent,
    *,
    workflow_state: AssistantWorkflowState | None = None,
    step: WorkflowStep | None = None,
    phase: str = "running",
) -> None:
    writer = get_stream_writer()
    if workflow_state is None:
        writer(event)
        return
    writer(
        event.model_copy(
            update={
                "payload": {
                    **event.payload,
                    "workflow": _serialize_workflow_runtime(
                        workflow_state=workflow_state,
                        step=step,
                        phase=phase,
                    ),
                }
            }
        )
    )


def _runtime(
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowRuntime:
    return runtime.context["runtime"]


def _workflow_state(
    state: AssistantWorkflowGraphState,
) -> AssistantWorkflowState:
    return state["workflow_state"]


def _route_decision(state: AssistantWorkflowState) -> RouteDecision:
    if state.route_decision is None:
        raise ValueError("Workflow route decision is not available.")
    return state.route_decision


def _serialize_workflow_runtime(
    *,
    workflow_state: AssistantWorkflowState,
    step: WorkflowStep | None,
    phase: str,
) -> dict[str, object]:
    runtime_step = step or workflow_state.current_step
    review_result = workflow_state.review_result
    clarification = workflow_state.clarification
    creative_strategy = workflow_state.creative_strategy
    creative_intent = workflow_state.creative_intent
    creative_hierarchy = workflow_state.creative_hierarchy
    creative_techniques = workflow_state.creative_techniques
    creative_plan = workflow_state.creative_plan
    creative_constraints = workflow_state.creative_constraints
    creative_constraint_priorities = workflow_state.creative_constraint_priorities
    runtime_capabilities = workflow_state.runtime_capabilities
    creative_tradeoffs = workflow_state.creative_tradeoffs
    creative_quality_prediction = workflow_state.creative_quality_prediction
    symbolic_narrative = workflow_state.symbolic_narrative
    creative_composition = workflow_state.creative_composition
    procedural_structure = workflow_state.procedural_structure
    generative_structure = workflow_state.generative_structure
    semantic_motif = workflow_state.semantic_motif
    emotional_consistency = workflow_state.emotional_consistency
    cross_modality = workflow_state.cross_modality
    audio_visual_scene = workflow_state.audio_visual_scene
    artifact_plan = workflow_state.artifact_plan
    artifact_dependency_graph = workflow_state.artifact_dependency_graph
    runtime_compatibility = workflow_state.runtime_compatibility
    artifact_capability_matrix = workflow_state.artifact_capability_matrix
    multi_artifact_strategy = workflow_state.multi_artifact_strategy
    artifact_critic = workflow_state.artifact_critic
    artifact_refiner = workflow_state.artifact_refiner
    artifact_intelligence_synthesis = (
        workflow_state.artifact_intelligence_synthesis
    )
    artifact_merge_planner = workflow_state.artifact_merge_planner
    artifact_export_intelligence = workflow_state.artifact_export_intelligence
    artifact_engine_contracts = workflow_state.artifact_engine_contracts
    creative_critic = workflow_state.creative_critic
    self_evaluation = workflow_state.self_evaluation
    creative_improvement_planner = workflow_state.creative_improvement_planner
    reflection_loop = workflow_state.reflection_loop
    creative_confidence = workflow_state.creative_confidence
    creative_director = workflow_state.creative_director
    creative_reasoning = workflow_state.creative_reasoning

    return {
        "step": runtime_step.value if runtime_step is not None else None,
        "phase": phase,
        "status": workflow_state.status.value,
        "current_step": (
            workflow_state.current_step.value
            if workflow_state.current_step is not None
            else None
        ),
        "completed_steps": [item.value for item in workflow_state.completed_steps],
        "skipped_steps": [item.value for item in workflow_state.skipped_steps],
        "refinement_count": workflow_state.refinement_count,
        "review_outcome": (
            review_result.outcome.value if review_result is not None else None
        ),
        "review_reasons": list(review_result.reasons) if review_result else [],
        "artifact_count": len(workflow_state.artifacts),
        "artifact_critique_count": (
            len(workflow_state.artifact_critique_summary.critiques)
            if workflow_state.artifact_critique_summary is not None
            else 0
        ),
        "recommended_artifact_id": (
            workflow_state.artifact_critique_summary.recommended_artifact_id
            if workflow_state.artifact_critique_summary is not None
            else None
        ),
        "preview_artifact_count": len(workflow_state.preview_results),
        "clarification_required": clarification is not None,
        "clarification_reason": (
            clarification.reason.value if clarification is not None else None
        ),
        "clarification_question_count": (
            len(clarification.questions) if clarification is not None else 0
        ),
        "clarification": (
            clarification.model_dump(mode="json")
            if clarification is not None
            else None
        ),
        "creative_plan": (
            creative_plan.model_dump(mode="json")
            if creative_plan is not None
            else None
        ),
        "planning_available": creative_plan is not None,
        "creative_intent": (
            creative_intent.model_dump(mode="json")
            if creative_intent is not None
            else None
        ),
        "intent_decomposer_available": creative_intent is not None,
        "creative_hierarchy": (
            creative_hierarchy.model_dump(mode="json")
            if creative_hierarchy is not None
            else None
        ),
        "hierarchy_planner_available": creative_hierarchy is not None,
        "creative_strategy": (
            creative_strategy.model_dump(mode="json")
            if creative_strategy is not None
            else None
        ),
        "strategy_available": creative_strategy is not None,
        "creative_techniques": (
            creative_techniques.model_dump(mode="json")
            if creative_techniques is not None
            else None
        ),
        "technique_selector_available": creative_techniques is not None,
        "creative_constraints": (
            creative_constraints.model_dump(mode="json")
            if creative_constraints is not None
            else None
        ),
        "constraint_solver_available": creative_constraints is not None,
        "creative_constraint_priorities": (
            creative_constraint_priorities.model_dump(mode="json")
            if creative_constraint_priorities is not None
            else None
        ),
        "constraint_prioritizer_available": (
            creative_constraint_priorities is not None
        ),
        "runtime_capabilities": (
            runtime_capabilities.model_dump(mode="json")
            if runtime_capabilities is not None
            else None
        ),
        "runtime_capability_reasoner_available": runtime_capabilities is not None,
        "creative_tradeoffs": (
            creative_tradeoffs.model_dump(mode="json")
            if creative_tradeoffs is not None
            else None
        ),
        "tradeoff_explorer_available": creative_tradeoffs is not None,
        "creative_quality_prediction": (
            creative_quality_prediction.model_dump(mode="json")
            if creative_quality_prediction is not None
            else None
        ),
        "quality_predictor_available": creative_quality_prediction is not None,
        "symbolic_narrative": (
            symbolic_narrative.model_dump(mode="json")
            if symbolic_narrative is not None
            else None
        ),
        "symbolic_narrative_available": symbolic_narrative is not None,
        "creative_composition": (
            creative_composition.model_dump(mode="json")
            if creative_composition is not None
            else None
        ),
        "creative_composition_available": creative_composition is not None,
        "procedural_structure": (
            procedural_structure.model_dump(mode="json")
            if procedural_structure is not None
            else None
        ),
        "procedural_structure_available": procedural_structure is not None,
        "generative_structure": (
            generative_structure.model_dump(mode="json")
            if generative_structure is not None
            else None
        ),
        "generative_structure_available": generative_structure is not None,
        "semantic_motif": (
            semantic_motif.model_dump(mode="json")
            if semantic_motif is not None
            else None
        ),
        "semantic_motif_available": semantic_motif is not None,
        "emotional_consistency": (
            emotional_consistency.model_dump(mode="json")
            if emotional_consistency is not None
            else None
        ),
        "emotional_consistency_available": emotional_consistency is not None,
        "cross_modality": (
            cross_modality.model_dump(mode="json")
            if cross_modality is not None
            else None
        ),
        "cross_modality_available": cross_modality is not None,
        "audio_visual_scene": (
            audio_visual_scene.model_dump(mode="json")
            if audio_visual_scene is not None
            else None
        ),
        "audio_visual_scene_available": audio_visual_scene is not None,
        "artifact_plan": (
            artifact_plan.model_dump(mode="json")
            if artifact_plan is not None
            else None
        ),
        "artifact_planner_available": artifact_plan is not None,
        "artifact_dependency_graph": (
            artifact_dependency_graph.model_dump(mode="json")
            if artifact_dependency_graph is not None
            else None
        ),
        "artifact_dependency_graph_available": artifact_dependency_graph is not None,
        "runtime_compatibility": (
            runtime_compatibility.model_dump(mode="json")
            if runtime_compatibility is not None
            else None
        ),
        "runtime_compatibility_available": runtime_compatibility is not None,
        "artifact_capability_matrix": (
            artifact_capability_matrix.model_dump(mode="json")
            if artifact_capability_matrix is not None
            else None
        ),
        "artifact_capability_matrix_available": (
            artifact_capability_matrix is not None
        ),
        "multi_artifact_strategy": (
            multi_artifact_strategy.model_dump(mode="json")
            if multi_artifact_strategy is not None
            else None
        ),
        "multi_artifact_strategy_available": multi_artifact_strategy is not None,
        "artifact_critic": (
            artifact_critic.model_dump(mode="json")
            if artifact_critic is not None
            else None
        ),
        "artifact_critic_available": artifact_critic is not None,
        "artifact_refiner": (
            artifact_refiner.model_dump(mode="json")
            if artifact_refiner is not None
            else None
        ),
        "artifact_refiner_available": artifact_refiner is not None,
        "artifact_intelligence_synthesis": (
            artifact_intelligence_synthesis.model_dump(mode="json")
            if artifact_intelligence_synthesis is not None
            else None
        ),
        "artifact_intelligence_synthesis_available": (
            artifact_intelligence_synthesis is not None
        ),
        "artifact_merge_planner": (
            artifact_merge_planner.model_dump(mode="json")
            if artifact_merge_planner is not None
            else None
        ),
        "artifact_merge_planner_available": artifact_merge_planner is not None,
        "artifact_export_intelligence": (
            artifact_export_intelligence.model_dump(mode="json")
            if artifact_export_intelligence is not None
            else None
        ),
        "artifact_export_intelligence_available": (
            artifact_export_intelligence is not None
        ),
        "artifact_engine_contracts": (
            artifact_engine_contracts.model_dump(mode="json")
            if artifact_engine_contracts is not None
            else None
        ),
        "artifact_engine_contracts_available": artifact_engine_contracts is not None,
        "creative_critic": (
            creative_critic.model_dump(mode="json")
            if creative_critic is not None
            else None
        ),
        "creative_critic_available": creative_critic is not None,
        "self_evaluation": (
            self_evaluation.model_dump(mode="json")
            if self_evaluation is not None
            else None
        ),
        "self_evaluation_available": self_evaluation is not None,
        "creative_improvement_planner": (
            creative_improvement_planner.model_dump(mode="json")
            if creative_improvement_planner is not None
            else None
        ),
        "creative_improvement_planner_available": (
            creative_improvement_planner is not None
        ),
        "reflection_loop": (
            reflection_loop.model_dump(mode="json")
            if reflection_loop is not None
            else None
        ),
        "reflection_loop_available": reflection_loop is not None,
        "creative_confidence": (
            creative_confidence.model_dump(mode="json")
            if creative_confidence is not None
            else None
        ),
        "creative_confidence_available": creative_confidence is not None,
        "creative_director": (
            creative_director.model_dump(mode="json")
            if creative_director is not None
            else None
        ),
        "director_available": creative_director is not None,
        "creative_reasoning": (
            creative_reasoning.model_dump(mode="json")
            if creative_reasoning is not None
            else None
        ),
        "creative_reasoning_available": creative_reasoning is not None,
        "image_reference_count": len(workflow_state.request.attachments),
        "image_references": [
            {
                "id": image.id,
                "name": image.name,
                "mime_type": image.mime_type,
                "size_bytes": image.size_bytes,
            }
            for image in workflow_state.request.attachments
        ],
    }


def _optional_event_payload(
    key: str,
    value: dict[str, object] | None,
) -> dict[str, dict[str, object]]:
    return {key: value} if value is not None else {}


def _format_clarification_answer(clarification: ClarificationRequest) -> str:
    lines = [
        "I need one quick clarification before generating.",
        "",
        clarification.summary,
        "",
    ]
    for index, question in enumerate(clarification.questions, start=1):
        lines.append(f"{index}. {question.prompt}")
        for option in question.suggested_options:
            lines.append(f"- {option}")
        if question.default_recommendation:
            lines.append(
                f"Default recommendation: {question.default_recommendation}"
            )
        lines.append("")
    lines.append("Reply with your choice and I will continue generation.")
    return "\n".join(lines).strip()


def _transition_payload(
    source: str,
    target: str,
    decision_reason: str,
) -> dict[str, object]:
    return {
        "transition_source": source,
        "transition_target": target,
        "decision_reason": decision_reason,
        "edge": {
            "source": source,
            "target": target,
            "decision_reason": decision_reason,
        },
    }


def _default_transition_target(step: WorkflowStep) -> str:
    if step is WorkflowStep.REFINEMENT:
        return WorkflowStep.GENERATION.value
    if step in {WorkflowStep.FINALIZATION, WorkflowStep.FAILURE}:
        return "end"

    try:
        next_index = ASSISTANT_WORKFLOW_NODE_ORDER.index(step.value) + 1
    except ValueError:
        return "end"

    if next_index >= len(ASSISTANT_WORKFLOW_NODE_ORDER):
        return "end"
    return ASSISTANT_WORKFLOW_NODE_ORDER[next_index]


def _step_label(step: WorkflowStep) -> str:
    return step.value.replace("_", " ").title()


def _node_attempt_count(
    workflow_state: AssistantWorkflowState,
    step: WorkflowStep,
) -> int:
    if step is WorkflowStep.GENERATION:
        return workflow_state.refinement_count + 1
    if step is WorkflowStep.REFINEMENT:
        return workflow_state.refinement_count + 1
    return 1


def _review_reason_text(review_result: WorkflowReviewResult) -> str:
    return ", ".join(review_result.reasons) or "quality gate passed"
