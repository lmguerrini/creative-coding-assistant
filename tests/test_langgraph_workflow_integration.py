import unittest
from collections.abc import Iterator

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
    StreamEvent,
    StreamEventType,
)
from creative_coding_assistant.orchestration import (
    ASSISTANT_WORKFLOW_NODE_ORDER,
    AssistantService,
    AssistantWorkflowRuntime,
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
                "finalization",
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
                WorkflowStep.REVIEW,
            ),
        )

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

    def test_graph_propagates_node_failures_to_service_boundary(self) -> None:
        graph = build_assistant_workflow_graph()

        with self.assertRaisesRegex(RuntimeError, "route failed"):
            tuple(
                stream_assistant_workflow_events(
                    graph=graph,
                    request=_request(),
                    runtime=_runtime(route_fn=_failing_route),
                )
            )

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


def _request() -> AssistantRequest:
    return AssistantRequest(
        query="Generate a Three.js scene.",
        domain=CreativeCodingDomain.THREE_JS,
        mode=AssistantMode.GENERATE,
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
    return AssistantWorkflowRuntime(
        event_builder=StreamEventBuilder(),
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
    builder: StreamEventBuilder,
) -> Iterator[StreamEvent]:
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


def _shell_answer(decision: RouteDecision) -> str:
    return f"Shell answer for {decision.route.value} route."


class _FakeGenerationResult:
    def __init__(self, *, answer: str) -> None:
        self.answer = answer


if __name__ == "__main__":
    unittest.main()
