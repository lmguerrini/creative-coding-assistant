import unittest
from collections.abc import Iterator

from creative_coding_assistant.analytics import build_langsmith_observability
from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
    StreamEvent,
    StreamEventType,
)
from creative_coding_assistant.core import Settings
from creative_coding_assistant.orchestration import (
    ASSISTANT_WORKFLOW_NODE_ORDER,
    AssistantService,
    AssistantWorkflowRuntime,
    WorkflowFailureInfo,
    WorkflowReviewOutcome,
    WorkflowStatus,
    WorkflowStep,
    build_assistant_workflow_graph,
    build_initial_workflow_graph_state,
    stream_assistant_workflow_events,
)
from creative_coding_assistant.orchestration.events import StreamEventBuilder
from creative_coding_assistant.orchestration.routing import (
    RouteCapability,
    RouteDecision,
    RouteName,
)


class LangGraphWorkflowIntegrationTests(unittest.TestCase):
    def test_langgraph_node_order_matches_workflow_shape(self) -> None:
        self.assertEqual(
            ASSISTANT_WORKFLOW_NODE_ORDER,
            (
                "intake",
                "routing",
                "memory",
                "retrieval",
                "context_assembly",
                "prompt_input",
                "prompt_rendering",
                "generation",
                "review",
                "refinement",
                "finalization",
                "failure",
            ),
        )

    def test_graph_streams_existing_events_without_shape_changes(self) -> None:
        graph = build_assistant_workflow_graph()
        request = _request()
        runtime = _runtime()

        events = tuple(
            stream_assistant_workflow_events(
                graph=graph,
                request=request,
                runtime=runtime,
            )
        )

        self.assertEqual(
            [event.event_type for event in events],
            [
                StreamEventType.STATUS,
                StreamEventType.STATUS,
                StreamEventType.FINAL,
            ],
        )
        self.assertEqual([event.sequence for event in events], [0, 1, 2])
        self.assertEqual(events[0].payload["code"], "request_received")
        self.assertEqual(events[1].payload["code"], "route_selected")
        self.assertEqual(events[1].payload["route"]["route"], "generate")
        self.assertIn("generate route", events[2].payload["answer"])

    def test_graph_completes_workflow_state_after_generation(self) -> None:
        graph = build_assistant_workflow_graph()
        request = _request()
        final_state = graph.invoke(
            build_initial_workflow_graph_state(request),
            context={
                "runtime": _runtime(stream_generation=_stream_completed_generation)
            },
        )

        workflow_state = final_state["workflow_state"]
        self.assertEqual(workflow_state.status, WorkflowStatus.COMPLETED)
        self.assertEqual(workflow_state.final_answer, "Graph answer")
        self.assertEqual(
            workflow_state.completed_steps,
            (
                WorkflowStep.INTAKE,
                WorkflowStep.ROUTING,
                WorkflowStep.GENERATION,
                WorkflowStep.REVIEW,
                WorkflowStep.FINALIZATION,
            ),
        )
        self.assertEqual(
            workflow_state.skipped_steps,
            (
                WorkflowStep.MEMORY,
                WorkflowStep.RETRIEVAL,
                WorkflowStep.CONTEXT_ASSEMBLY,
                WorkflowStep.PROMPT_INPUT,
                WorkflowStep.PROMPT_RENDERING,
            ),
        )
        self.assertEqual(
            workflow_state.review_result.outcome,
            WorkflowReviewOutcome.PASS,
        )
        self.assertEqual(workflow_state.refinement_count, 0)

    def test_graph_streams_generation_custom_events_in_sequence(self) -> None:
        graph = build_assistant_workflow_graph()

        events = tuple(
            stream_assistant_workflow_events(
                graph=graph,
                request=_request(),
                runtime=_runtime(stream_generation=_stream_completed_generation),
            )
        )

        self.assertEqual(
            [event.event_type for event in events],
            [
                StreamEventType.STATUS,
                StreamEventType.STATUS,
                StreamEventType.GENERATION_INPUT,
                StreamEventType.TOKEN_DELTA,
                StreamEventType.TOKEN_DELTA,
                StreamEventType.FINAL,
            ],
        )
        self.assertEqual([event.sequence for event in events], list(range(6)))
        self.assertEqual(events[2].payload["code"], "generation_input_prepared")
        self.assertEqual(events[3].payload["text"], "Graph ")
        self.assertEqual(events[4].payload["text"], "answer")
        self.assertEqual(events[5].payload["answer"], "Graph answer")

    def test_stream_events_include_workflow_runtime_metadata(self) -> None:
        graph = build_assistant_workflow_graph()

        events = tuple(
            stream_assistant_workflow_events(
                graph=graph,
                request=_request(),
                runtime=_runtime(stream_generation=_stream_completed_generation),
            )
        )

        self.assertIn("emitted_at", events[0].payload)
        self.assertEqual(events[0].payload["workflow"]["step"], "intake")
        self.assertEqual(events[0].payload["workflow"]["phase"], "running")
        self.assertEqual(events[1].payload["workflow"]["step"], "routing")
        self.assertEqual(events[2].payload["workflow"]["step"], "generation")
        self.assertEqual(events[-1].payload["workflow"]["step"], "finalization")
        self.assertEqual(events[-1].payload["workflow"]["phase"], "completed")
        self.assertEqual(events[-1].payload["workflow"]["status"], "completed")
        self.assertEqual(
            events[-1].payload["workflow"]["completed_steps"],
            ["intake", "routing", "generation", "review", "finalization"],
        )
        self.assertEqual(
            events[-1].payload["workflow"]["skipped_steps"],
            [
                "memory",
                "retrieval",
                "context_assembly",
                "prompt_input",
                "prompt_rendering",
            ],
        )
        self.assertEqual(events[-1].payload["workflow"]["review_outcome"], "pass")

    def test_review_failure_runs_one_refinement_attempt(self) -> None:
        graph = build_assistant_workflow_graph()
        generation = _SequentialGeneration(
            "This answer does not include fenced code.",
            "```javascript\nconsole.log('refined');\n```",
        )

        final_state = graph.invoke(
            build_initial_workflow_graph_state(
                _request(query="Write code for a Three.js scene.")
            ),
            context={
                "runtime": _runtime(stream_generation=generation.stream)
            },
        )

        workflow_state = final_state["workflow_state"]
        self.assertEqual(generation.calls, 2)
        self.assertEqual(workflow_state.status, WorkflowStatus.COMPLETED)
        self.assertEqual(
            workflow_state.final_answer,
            "```javascript\nconsole.log('refined');\n```",
        )
        self.assertEqual(
            workflow_state.review_result.outcome,
            WorkflowReviewOutcome.PASS,
        )
        self.assertEqual(workflow_state.refinement_count, 1)
        self.assertIn(WorkflowStep.REFINEMENT, workflow_state.completed_steps)

    def test_review_refinement_is_bounded_to_one_attempt(self) -> None:
        graph = build_assistant_workflow_graph()
        generation = _SequentialGeneration(
            "Still no fenced code.",
            "Still no fenced code after retry.",
        )

        final_state = graph.invoke(
            build_initial_workflow_graph_state(
                _request(query="Write code for a Three.js scene.")
            ),
            context={
                "runtime": _runtime(stream_generation=generation.stream)
            },
        )

        workflow_state = final_state["workflow_state"]
        self.assertEqual(generation.calls, 2)
        self.assertEqual(workflow_state.status, WorkflowStatus.COMPLETED)
        self.assertEqual(
            workflow_state.final_answer,
            "Still no fenced code after retry.",
        )
        self.assertEqual(
            workflow_state.review_result.outcome,
            WorkflowReviewOutcome.NEEDS_REFINEMENT,
        )
        self.assertEqual(
            workflow_state.review_result.reasons,
            ("missing_code_block",),
        )
        self.assertEqual(workflow_state.refinement_count, 1)

    def test_refinement_stream_preserves_existing_event_shapes(self) -> None:
        graph = build_assistant_workflow_graph()
        generation = _SequentialGeneration(
            "This answer does not include fenced code.",
            "```javascript\nconsole.log('refined');\n```",
        )

        events = tuple(
            stream_assistant_workflow_events(
                graph=graph,
                request=_request(query="Write code for a Three.js scene."),
                runtime=_runtime(stream_generation=generation.stream),
            )
        )

        self.assertEqual(
            [event.event_type for event in events],
            [
                StreamEventType.STATUS,
                StreamEventType.STATUS,
                StreamEventType.GENERATION_INPUT,
                StreamEventType.TOKEN_DELTA,
                StreamEventType.GENERATION_INPUT,
                StreamEventType.TOKEN_DELTA,
                StreamEventType.FINAL,
            ],
        )
        self.assertEqual([event.sequence for event in events], list(range(7)))
        self.assertEqual(generation.calls, 2)
        self.assertEqual(
            events[-1].payload["answer"],
            "```javascript\nconsole.log('refined');\n```",
        )

    def test_graph_routes_routing_failures_to_terminal_failure_path(self) -> None:
        graph = build_assistant_workflow_graph()
        request = _request()

        events = tuple(
            stream_assistant_workflow_events(
                graph=graph,
                request=request,
                runtime=_runtime(route_fn=_failing_route),
            )
        )
        final_state = graph.invoke(
            build_initial_workflow_graph_state(request),
            context={"runtime": _runtime(route_fn=_failing_route)},
        )

        self.assertEqual(
            [event.event_type for event in events],
            [
                StreamEventType.STATUS,
                StreamEventType.ERROR,
                StreamEventType.FINAL,
            ],
        )
        self.assertEqual(events[1].payload["code"], "workflow_routing_failed")
        self.assertIn("route failed", events[2].payload["answer"])
        self.assertEqual(final_state["workflow_state"].status, WorkflowStatus.FAILED)
        self.assertEqual(
            final_state["workflow_state"].failure_info,
            WorkflowFailureInfo(
                step=WorkflowStep.ROUTING,
                code="workflow_routing_failed",
                message="route failed",
            ),
        )

    def test_graph_routes_provider_failures_to_terminal_failure_path(self) -> None:
        graph = build_assistant_workflow_graph()
        request = _request(mode=AssistantMode.EXPLAIN, query="Explain the scene setup.")

        events = tuple(
            stream_assistant_workflow_events(
                graph=graph,
                request=request,
                runtime=_runtime(stream_generation=_stream_failed_generation),
            )
        )
        final_state = graph.invoke(
            build_initial_workflow_graph_state(request),
            context={"runtime": _runtime(stream_generation=_stream_failed_generation)},
        )

        self.assertEqual(
            [event.event_type for event in events],
            [
                StreamEventType.STATUS,
                StreamEventType.STATUS,
                StreamEventType.GENERATION_INPUT,
                StreamEventType.ERROR,
                StreamEventType.FINAL,
            ],
        )
        self.assertEqual(events[3].payload["code"], "provider_unavailable")
        self.assertIn("Generation failed", events[4].payload["answer"])
        workflow_state = final_state["workflow_state"]
        self.assertEqual(workflow_state.status, WorkflowStatus.FAILED)
        self.assertEqual(
            workflow_state.failure_info,
            WorkflowFailureInfo(
                step=WorkflowStep.GENERATION,
                code="provider_unavailable",
                message="Provider unavailable.",
            ),
        )
        self.assertEqual(
            workflow_state.final_answer,
            "Generation failed (provider_unavailable): Provider unavailable.",
        )
        self.assertEqual(workflow_state.review_result, None)
        self.assertEqual(workflow_state.refinement_count, 0)

    def test_assistant_service_executes_via_compiled_graph(self) -> None:
        service = AssistantService()

        self.assertTrue(hasattr(service._workflow_graph, "stream"))
        events = tuple(service.stream(_request()))

        self.assertEqual(
            [event.event_type for event in events],
            [
                StreamEventType.STATUS,
                StreamEventType.STATUS,
                StreamEventType.FINAL,
            ],
        )


def _request(
    *,
    query: str = "Generate a Three.js scene.",
    mode: AssistantMode = AssistantMode.GENERATE,
) -> AssistantRequest:
    return AssistantRequest(
        query=query,
        domain=CreativeCodingDomain.THREE_JS,
        mode=mode,
    )


def _route_generate(request: AssistantRequest) -> RouteDecision:
    return RouteDecision(
        route=RouteName.GENERATE,
        mode=request.mode,
        domain=request.domain,
        capabilities=(RouteCapability.TOOL_USE,),
    )


def _failing_route(request: AssistantRequest) -> RouteDecision:
    del request
    raise RuntimeError("route failed")


def _runtime(
    *,
    route_fn=_route_generate,
    stream_generation=None,
) -> AssistantWorkflowRuntime:
    observability = build_langsmith_observability(Settings(_env_file=None))
    return AssistantWorkflowRuntime(
        event_builder=StreamEventBuilder(),
        observability=observability,
        observability_run=observability.assistant_run_context(_request()),
        route_fn=route_fn,
        stream_request_received=_stream_request_received,
        stream_route_selected=_stream_route_selected,
        stream_memory_context=_empty_streaming_step,
        stream_retrieval_context=_empty_streaming_step,
        stream_assembled_context=_empty_streaming_step,
        stream_prompt_inputs=_empty_streaming_step,
        stream_rendered_prompt=_empty_streaming_step,
        stream_generation=stream_generation or _empty_streaming_step,
        build_shell_answer=_shell_answer,
    )


def _stream_request_received(
    *,
    builder: StreamEventBuilder,
    **kwargs: object,
) -> Iterator[StreamEvent]:
    del kwargs
    yield builder.status(code="request_received", message="Request accepted.")


def _stream_route_selected(
    *,
    builder: StreamEventBuilder,
    decision: RouteDecision,
    route_payload: dict[str, object],
) -> Iterator[StreamEvent]:
    del decision
    yield builder.status(
        code="route_selected",
        message="Route selected.",
        route=route_payload,
    )


def _empty_streaming_step(**kwargs: object) -> Iterator[StreamEvent]:
    del kwargs
    if False:
        yield StreamEvent(
            event_type=StreamEventType.STATUS,
            sequence=0,
            payload={},
        )
    return None


def _stream_completed_generation(
    *,
    builder: StreamEventBuilder,
    **kwargs: object,
) -> Iterator[StreamEvent]:
    del kwargs
    yield builder.generation_input(
        code="generation_input_prepared",
        message="Provider generation input prepared.",
    )
    yield builder.token_delta("Graph ")
    yield builder.token_delta("answer")
    return _FakeGenerationResult(answer="Graph answer")


def _stream_failed_generation(
    *,
    builder: StreamEventBuilder,
    **kwargs: object,
) -> Iterator[StreamEvent]:
    del kwargs
    yield builder.generation_input(
        code="generation_input_prepared",
        message="Provider generation input prepared.",
    )
    yield builder.error(
        code="provider_unavailable",
        message="Provider unavailable.",
    )
    return _FakeGenerationResult(
        answer="Generation failed (provider_unavailable): Provider unavailable.",
        error_code="provider_unavailable",
        error_message="Provider unavailable.",
    )


def _shell_answer(decision: RouteDecision) -> str:
    return f"Shell answer for {decision.route.value} route."


class _FakeGenerationResult:
    def __init__(
        self,
        *,
        answer: str,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        self.answer = answer
        self.error_code = error_code
        self.error_message = error_message


class _SequentialGeneration:
    def __init__(self, *answers: str) -> None:
        self._answers = answers
        self.calls = 0

    def stream(
        self,
        *,
        builder: StreamEventBuilder,
        **kwargs: object,
    ) -> Iterator[StreamEvent]:
        del kwargs
        answer_index = min(self.calls, len(self._answers) - 1)
        answer = self._answers[answer_index]
        self.calls += 1
        yield builder.generation_input(
            code="generation_input_prepared",
            message="Provider generation input prepared.",
        )
        yield builder.token_delta(answer)
        return _FakeGenerationResult(answer=answer)


if __name__ == "__main__":
    unittest.main()
