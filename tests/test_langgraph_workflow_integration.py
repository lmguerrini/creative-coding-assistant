import unittest
from collections.abc import Callable, Iterator, Sequence

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
    ASSISTANT_WORKFLOW_RECURSION_LIMIT,
    AssistantService,
    AssistantWorkflowRuntime,
    StructuredPromptInputBuilder,
    WorkflowFailureInfo,
    WorkflowReviewOutcome,
    WorkflowStatus,
    WorkflowStep,
    build_assistant_workflow_graph,
    build_initial_workflow_graph_state,
    build_prompt_input_request,
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
                "planning",
                "director",
                "prompt_rendering",
                "generation",
                "artifact_extraction",
                "preview_preparation",
                "artifact_critique",
                "review",
                "refinement",
                "finalization",
                "failure",
            ),
        )

    def test_graph_streams_node_lifecycle_events_with_legacy_statuses(self) -> None:
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

        event_types = _event_types(events)
        self.assertEqual(
            _node_payloads(events, StreamEventType.NODE_STARTED),
            [
                "intake",
                "routing",
                "memory",
                "retrieval",
                "context_assembly",
                "prompt_input",
                "planning",
                "director",
                "prompt_rendering",
                "generation",
                "artifact_extraction",
                "preview_preparation",
                "artifact_critique",
                "review",
                "finalization",
            ],
        )
        self.assertEqual(
            _node_payloads(events, StreamEventType.NODE_COMPLETED),
            [
                "intake",
                "routing",
                "memory",
                "retrieval",
                "context_assembly",
                "prompt_input",
                "planning",
                "director",
                "prompt_rendering",
                "generation",
                "artifact_extraction",
                "preview_preparation",
                "artifact_critique",
                "review",
                "finalization",
            ],
        )
        self.assertEqual([event.sequence for event in events], list(range(len(events))))
        self.assertEqual(event_types[-1], StreamEventType.FINAL)
        self.assertIn(StreamEventType.STATUS, event_types)
        self.assertIn(StreamEventType.REVIEW_PASSED, event_types)
        request_status = _first_event(
            events,
            StreamEventType.STATUS,
            "request_received",
        )
        routing_status = _first_event(events, StreamEventType.STATUS, "route_selected")
        review_passed = _first_event(events, StreamEventType.REVIEW_PASSED)
        final = events[-1]

        self.assertEqual(routing_status.payload["route"]["route"], "generate")
        self.assertEqual(review_passed.payload["score"], 1.0)
        self.assertEqual(
            review_passed.payload["rationale"],
            "Deterministic review passed without quality gate findings.",
        )
        self.assertEqual(review_passed.payload["transition_source"], "review")
        self.assertEqual(review_passed.payload["transition_target"], "finalization")
        self.assertEqual(review_passed.payload["decision_reason"], "review_passed")
        self.assertEqual(request_status.payload["workflow"]["step"], "intake")
        self.assertIn("generate route", final.payload["answer"])

    def test_graph_short_circuits_before_generation_when_clarification_required(
        self,
    ) -> None:
        graph = build_assistant_workflow_graph()
        request = AssistantRequest(
            query="Make something evocative about rain.",
            mode=AssistantMode.GENERATE,
        )
        runtime = _runtime(
            route_fn=_route_generate_without_domains,
            stream_prompt_inputs=_stream_prompt_inputs_with_builder,
            stream_generation=_unexpected_generation,
        )

        events = tuple(
            stream_assistant_workflow_events(
                graph=graph,
                request=request,
                runtime=runtime,
            )
        )
        final_event = events[-1]
        prompt_input_completion = _first_transition(
            events,
            StreamEventType.NODE_COMPLETED,
            source="prompt_input",
            target="finalization",
        )

        self.assertNotIn(StreamEventType.PROMPT_RENDERED, _event_types(events))
        self.assertNotIn(StreamEventType.GENERATION_INPUT, _event_types(events))
        self.assertEqual(
            prompt_input_completion.payload["decision_reason"],
            "clarification_required",
        )
        self.assertEqual(
            final_event.payload["clarification"]["reason"],
            "ambiguous_modality",
        )
        self.assertIn("I need one quick clarification", final_event.payload["answer"])
        self.assertEqual(
            final_event.payload["workflow"]["clarification_reason"],
            "ambiguous_modality",
        )
        self.assertIn(
            "prompt_input",
            final_event.payload["workflow"]["completed_steps"],
        )
        self.assertNotIn(
            "generation",
            final_event.payload["workflow"]["completed_steps"],
        )

    def test_graph_plans_between_prompt_input_and_prompt_rendering(self) -> None:
        graph = build_assistant_workflow_graph()
        request = _request(
            query="Generate a luminous p5.js mandala sketch.",
            domain=CreativeCodingDomain.P5_JS,
        )
        runtime = _runtime(
            stream_prompt_inputs=_stream_prompt_inputs_with_builder,
            stream_generation=_stream_completed_generation,
        )

        events = tuple(
            stream_assistant_workflow_events(
                graph=graph,
                request=request,
                runtime=runtime,
            )
        )

        planning_event = _first_event(
            events,
            StreamEventType.PLANNING,
            "creative_plan_prepared",
        )
        director_event = _first_event(
            events,
            StreamEventType.PLANNING,
            "creative_director_prepared",
        )
        final_event = events[-1]
        strategy = planning_event.payload["creative_strategy"]
        plan = planning_event.payload["creative_plan"]
        constraints = planning_event.payload["creative_constraints"]
        director = director_event.payload["creative_director"]

        self.assertEqual(planning_event.payload["workflow"]["step"], "planning")
        self.assertEqual(director_event.payload["workflow"]["step"], "director")
        self.assertEqual(strategy["role"], "creative_strategy_engine")
        self.assertTrue(planning_event.payload["workflow"]["strategy_available"])
        self.assertEqual(plan["output_modality"], "visual")
        self.assertEqual(plan["recommended_runtime"], "p5")
        self.assertEqual(constraints["role"], "creative_constraint_solver")
        self.assertEqual(constraints["runtime_fit"], "supported")
        self.assertTrue(
            planning_event.payload["workflow"]["constraint_solver_available"]
        )
        self.assertEqual(director["role"], "creative_assistant_director")
        self.assertEqual(director["runtime_direction"], "p5")
        self.assertEqual(
            final_event.payload["creative_strategy"]["role"],
            "creative_strategy_engine",
        )
        self.assertEqual(
            final_event.payload["creative_plan"]["recommended_runtime"],
            "p5",
        )
        self.assertEqual(
            final_event.payload["creative_constraints"]["recommended_runtime"],
            "p5",
        )
        self.assertEqual(
            final_event.payload["creative_director"]["runtime_direction"],
            "p5",
        )
        self.assertIn(
            "planning",
            final_event.payload["workflow"]["completed_steps"],
        )
        self.assertIn(
            "director",
            final_event.payload["workflow"]["completed_steps"],
        )
        _first_transition(
            events,
            StreamEventType.NODE_COMPLETED,
            source="planning",
            target="director",
        )
        _first_transition(
            events,
            StreamEventType.NODE_COMPLETED,
            source="director",
            target="prompt_rendering",
        )

    def test_graph_completes_workflow_state_after_generation(self) -> None:
        graph = build_assistant_workflow_graph()
        request = _request()
        final_state = graph.invoke(
            build_initial_workflow_graph_state(request),
            config={"recursion_limit": ASSISTANT_WORKFLOW_RECURSION_LIMIT},
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
                WorkflowStep.PLANNING,
                WorkflowStep.DIRECTOR,
                WorkflowStep.PROMPT_RENDERING,
                WorkflowStep.ARTIFACT_EXTRACTION,
                WorkflowStep.PREVIEW_PREPARATION,
                WorkflowStep.ARTIFACT_CRITIQUE,
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

        event_types = _event_types(events)
        _assert_subsequence(
            self,
            event_types,
            [
                StreamEventType.NODE_STARTED,
                StreamEventType.GENERATION_INPUT,
                StreamEventType.TOKEN_DELTA,
                StreamEventType.TOKEN_DELTA,
                StreamEventType.NODE_COMPLETED,
                StreamEventType.NODE_STARTED,
                StreamEventType.REVIEW_PASSED,
                StreamEventType.NODE_COMPLETED,
                StreamEventType.FINAL,
            ],
        )
        self.assertEqual([event.sequence for event in events], list(range(len(events))))
        generation_input = _first_event(events, StreamEventType.GENERATION_INPUT)
        token_events = _events_of_type(events, StreamEventType.TOKEN_DELTA)
        generation_completed = _first_transition(
            events,
            StreamEventType.NODE_COMPLETED,
            source="generation",
            target="artifact_extraction",
        )

        self.assertEqual(generation_input.payload["code"], "generation_input_prepared")
        self.assertEqual(
            [event.payload["text"] for event in token_events],
            ["Graph ", "answer"],
        )
        self.assertEqual(
            generation_completed.payload["decision_reason"],
            "generation_completed",
        )
        self.assertEqual(events[-1].payload["answer"], "Graph answer")

    def test_stream_events_include_workflow_runtime_metadata(self) -> None:
        graph = build_assistant_workflow_graph()

        events = tuple(
            stream_assistant_workflow_events(
                graph=graph,
                request=_request(),
                runtime=_runtime(stream_generation=_stream_completed_generation),
            )
        )

        intake_started = _first_transition(
            events,
            StreamEventType.NODE_STARTED,
            source="intake",
        )
        routing_started = _first_transition(
            events,
            StreamEventType.NODE_STARTED,
            source="routing",
        )
        generation_input = _first_event(events, StreamEventType.GENERATION_INPUT)
        final = events[-1]

        self.assertIn("emitted_at", intake_started.payload)
        self.assertEqual(intake_started.payload["workflow"]["step"], "intake")
        self.assertEqual(intake_started.payload["workflow"]["phase"], "running")
        self.assertEqual(routing_started.payload["workflow"]["step"], "routing")
        self.assertEqual(generation_input.payload["workflow"]["step"], "generation")
        self.assertEqual(events[-1].payload["workflow"]["step"], "finalization")
        self.assertEqual(final.payload["workflow"]["phase"], "completed")
        self.assertEqual(final.payload["workflow"]["status"], "completed")
        self.assertEqual(
            final.payload["workflow"]["completed_steps"],
            ["intake", "routing", "generation", "review", "finalization"],
        )
        self.assertEqual(
            final.payload["workflow"]["skipped_steps"],
            [
                "memory",
                "retrieval",
                "context_assembly",
                "prompt_input",
                "planning",
                "director",
                "prompt_rendering",
                "artifact_extraction",
                "preview_preparation",
                "artifact_critique",
            ],
        )
        self.assertEqual(final.payload["workflow"]["review_outcome"], "pass")
        self.assertEqual(final.payload["workflow"]["artifact_count"], 0)
        self.assertEqual(final.payload["workflow"]["artifact_critique_count"], 0)
        self.assertEqual(final.payload["workflow"]["preview_artifact_count"], 0)

    def test_review_failure_runs_one_refinement_attempt(self) -> None:
        graph = build_assistant_workflow_graph()
        generation = _SequentialGeneration(
            "This answer does not include fenced code.",
            "\n".join(
                [
                    "```javascript",
                    "const scene = new THREE.Scene();",
                    "const camera = new THREE.PerspectiveCamera(70, 1, 0.1, 100);",
                    "function animate() {",
                    "  scene.rotation.y += 0.01;",
                    "}",
                    "```",
                ]
            ),
        )

        final_state = graph.invoke(
            build_initial_workflow_graph_state(
                _request(query="Write code for a Three.js scene.")
            ),
            config={"recursion_limit": ASSISTANT_WORKFLOW_RECURSION_LIMIT},
            context={
                "runtime": _runtime(stream_generation=generation.stream)
            },
        )

        workflow_state = final_state["workflow_state"]
        self.assertEqual(generation.calls, 2)
        self.assertEqual(workflow_state.status, WorkflowStatus.COMPLETED)
        self.assertEqual(
            workflow_state.final_answer,
            "\n".join(
                [
                    "```javascript",
                    "const scene = new THREE.Scene();",
                    "const camera = new THREE.PerspectiveCamera(70, 1, 0.1, 100);",
                    "function animate() {",
                    "  scene.rotation.y += 0.01;",
                    "}",
                    "```",
                ]
            ),
        )
        self.assertEqual(
            workflow_state.review_result.outcome,
            WorkflowReviewOutcome.PASS,
        )
        self.assertEqual(workflow_state.refinement_count, 1)
        self.assertIn(WorkflowStep.REFINEMENT, workflow_state.completed_steps)

    def test_generation_artifacts_are_extracted_and_prepared_for_preview(self) -> None:
        graph = build_assistant_workflow_graph()
        answer = "\n".join(
            [
                "```javascript",
                "function setup() {",
                "  createCanvas(640, 360);",
                "}",
                "function draw() {",
                "  background(12);",
                "}",
                "```",
            ]
        )

        events = tuple(
            stream_assistant_workflow_events(
                graph=graph,
                request=_request(query="Write a p5.js sketch."),
                runtime=_runtime(stream_generation=_single_generation(answer)),
            )
        )
        final_state = graph.invoke(
            build_initial_workflow_graph_state(
                _request(query="Write a p5.js sketch.")
            ),
            config={"recursion_limit": ASSISTANT_WORKFLOW_RECURSION_LIMIT},
            context={"runtime": _runtime(stream_generation=_single_generation(answer))},
        )

        workflow_state = final_state["workflow_state"]
        artifact_event = next(
            event
            for event in events
            if event.event_type is StreamEventType.ARTIFACT_EXTRACTED
        )
        preview_event = next(
            event
            for event in events
            if event.event_type is StreamEventType.PREVIEW_ARTIFACT
        )

        self.assertEqual(workflow_state.artifacts[0].runtime, "p5")
        self.assertEqual(
            workflow_state.completed_steps,
            (
                WorkflowStep.INTAKE,
                WorkflowStep.ROUTING,
                WorkflowStep.GENERATION,
                WorkflowStep.ARTIFACT_EXTRACTION,
                WorkflowStep.PREVIEW_PREPARATION,
                WorkflowStep.ARTIFACT_CRITIQUE,
                WorkflowStep.REVIEW,
                WorkflowStep.FINALIZATION,
            ),
        )
        self.assertIsNotNone(workflow_state.artifact_critique_summary)
        self.assertEqual(
            workflow_state.artifact_critique_summary.recommended_artifact_id,
            workflow_state.artifacts[0].id,
        )
        self.assertTrue(workflow_state.artifacts[0].is_recommended)
        self.assertIsNotNone(workflow_state.artifacts[0].critique)
        self.assertIsNotNone(
            workflow_state.artifacts[0].critique.creative_evaluation
        )
        self.assertGreaterEqual(workflow_state.artifacts[0].quality_score, 0.68)
        critique_events = tuple(
            event
            for event in events
            if event.event_type is StreamEventType.ARTIFACT_CRITIQUE
        )
        self.assertEqual(
            [event.payload["code"] for event in critique_events],
            [
                "critique_started",
                "artifact_scored",
                "artifact_selected_recommended",
                "critique_completed",
            ],
        )
        self.assertEqual(
            artifact_event.payload["artifacts"][0]["title"],
            "generated-sketch-1.p5.js",
        )
        self.assertEqual(
            artifact_event.payload["workflow"]["step"],
            "artifact_extraction",
        )
        self.assertEqual(
            preview_event.payload["artifact_id"],
            workflow_state.artifacts[0].id,
        )
        self.assertEqual(preview_event.payload["status"], "succeeded")
        self.assertEqual(
            preview_event.payload["result"]["request"]["target"],
            "browser_sandbox",
        )
        self.assertEqual(events[-1].payload["artifacts"][0]["runtime"], "p5")
        self.assertIn(
            "creative_evaluation",
            events[-1].payload["artifacts"][0]["critique"],
        )
        self.assertEqual(
            events[-1].payload["preview_results"][0]["status"],
            "succeeded",
        )

    def test_planning_metadata_flows_into_generated_artifacts(self) -> None:
        graph = build_assistant_workflow_graph()
        answer = "\n".join(
            [
                "```javascript",
                "function setup() { createCanvas(640, 360); }",
                "function draw() { background(12); }",
                "```",
            ]
        )
        request = _request(
            query="Write a p5.js mandala sketch.",
            domain=CreativeCodingDomain.P5_JS,
        )

        final_state = graph.invoke(
            build_initial_workflow_graph_state(request),
            config={"recursion_limit": ASSISTANT_WORKFLOW_RECURSION_LIMIT},
            context={
                "runtime": _runtime(
                    stream_prompt_inputs=_stream_prompt_inputs_with_builder,
                    stream_generation=_single_generation(answer),
                )
            },
        )

        workflow_state = final_state["workflow_state"]
        self.assertIsNotNone(workflow_state.creative_plan)
        self.assertIsNotNone(workflow_state.artifacts[0].creative_plan)
        self.assertEqual(
            workflow_state.artifacts[0].creative_plan.recommended_runtime,
            "p5",
        )
        self.assertEqual(
            workflow_state.creative_plan,
            workflow_state.artifacts[0].creative_plan,
        )

    def test_generation_extracts_multiple_artifacts_with_selection_metadata(
        self,
    ) -> None:
        graph = build_assistant_workflow_graph()
        answer = "\n".join(
            [
                "Variant A:",
                "```python filename=shared-output.js",
                "palette = ['#0bf', '#111']",
                "```",
                "Variant B:",
                "```javascript filename=shared-output.js",
                "function setup() {",
                "  createCanvas(640, 360);",
                "}",
                "function draw() {",
                "  background(12);",
                "}",
                "```",
            ]
        )

        events = tuple(
            stream_assistant_workflow_events(
                graph=graph,
                request=_request(query="Generate two candidate sketches."),
                runtime=_runtime(stream_generation=_single_generation(answer)),
            )
        )
        final_state = graph.invoke(
            build_initial_workflow_graph_state(
                _request(query="Generate two candidate sketches.")
            ),
            config={"recursion_limit": ASSISTANT_WORKFLOW_RECURSION_LIMIT},
            context={"runtime": _runtime(stream_generation=_single_generation(answer))},
        )

        workflow_state = final_state["workflow_state"]
        artifact_event = next(
            event
            for event in events
            if event.event_type is StreamEventType.ARTIFACT_EXTRACTED
        )
        preview_events = tuple(
            event
            for event in events
            if event.event_type is StreamEventType.PREVIEW_ARTIFACT
        )

        self.assertEqual(len(workflow_state.artifacts), 2)
        self.assertEqual(
            [artifact.id for artifact in workflow_state.artifacts],
            ["shared-output.js", "shared-output.js-2"],
        )
        self.assertEqual(
            [artifact.source_order for artifact in workflow_state.artifacts],
            [1, 2],
        )
        self.assertEqual(
            [artifact.is_default for artifact in workflow_state.artifacts],
            [False, True],
        )
        self.assertEqual(
            [artifact.preview_eligible for artifact in workflow_state.artifacts],
            [False, True],
        )
        self.assertEqual(workflow_state.artifacts[0].domain, "three_js")
        self.assertEqual(artifact_event.payload["artifact_count"], 2)
        self.assertEqual(
            artifact_event.payload["artifacts"][1]["id"],
            "shared-output.js-2",
        )
        self.assertEqual(
            artifact_event.payload["artifacts"][1]["source_order"],
            2,
        )
        self.assertEqual(
            artifact_event.payload["artifacts"][0]["is_default"],
            False,
        )
        self.assertEqual(
            artifact_event.payload["artifacts"][1]["is_default"],
            True,
        )
        self.assertEqual(len(preview_events), 1)
        self.assertEqual(preview_events[0].payload["artifact_id"], "shared-output.js-2")
        self.assertEqual(events[-1].payload["workflow"]["artifact_count"], 2)
        self.assertEqual(
            events[-1].payload["workflow"]["artifact_critique_count"],
            2,
        )
        self.assertEqual(
            events[-1].payload["artifact_critique_summary"]["recommended_artifact_id"],
            workflow_state.artifact_critique_summary.recommended_artifact_id,
        )

    def test_review_refinement_is_bounded_to_default_pass_limit(self) -> None:
        graph = build_assistant_workflow_graph()
        generation = _SequentialGeneration(
            "Still no fenced code.",
            "Still no fenced code after retry.",
        )

        final_state = graph.invoke(
            build_initial_workflow_graph_state(
                _request(query="Write code for a Three.js scene.")
            ),
            config={"recursion_limit": ASSISTANT_WORKFLOW_RECURSION_LIMIT},
            context={
                "runtime": _runtime(stream_generation=generation.stream)
            },
        )

        workflow_state = final_state["workflow_state"]
        self.assertEqual(generation.calls, 3)
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
        self.assertEqual(workflow_state.refinement_count, 2)

    def test_refinement_stream_exposes_review_retry_and_preview_events(self) -> None:
        graph = build_assistant_workflow_graph()
        generation = _SequentialGeneration(
            "This answer does not include fenced code.",
            "\n".join(
                [
                    "```javascript",
                    "const scene = new THREE.Scene();",
                    "const camera = new THREE.PerspectiveCamera(70, 1, 0.1, 100);",
                    "function animate() {",
                    "  scene.rotation.y += 0.01;",
                    "}",
                    "```",
                ]
            ),
        )

        events = tuple(
            stream_assistant_workflow_events(
                graph=graph,
                request=_request(query="Write code for a Three.js scene."),
                runtime=_runtime(stream_generation=generation.stream),
            )
        )

        event_types = _event_types(events)
        _assert_subsequence(
            self,
            event_types,
            [
                StreamEventType.GENERATION_INPUT,
                StreamEventType.TOKEN_DELTA,
                StreamEventType.REVIEW_FAILED,
                StreamEventType.REFINEMENT_REQUESTED,
                StreamEventType.RETRY_STARTED,
                StreamEventType.NODE_COMPLETED,
                StreamEventType.REFINEMENT_COMPLETED,
                StreamEventType.GENERATION_INPUT,
                StreamEventType.TOKEN_DELTA,
                StreamEventType.ARTIFACT_EXTRACTED,
                StreamEventType.PREVIEW_ARTIFACT,
                StreamEventType.ARTIFACT_CRITIQUE,
                StreamEventType.REVIEW_PASSED,
                StreamEventType.RETRY_COMPLETED,
                StreamEventType.FINAL,
            ],
        )
        self.assertEqual([event.sequence for event in events], list(range(len(events))))
        self.assertEqual(generation.calls, 2)
        review_failed = _first_event(events, StreamEventType.REVIEW_FAILED)
        retry_started = _first_event(events, StreamEventType.RETRY_STARTED)
        retry_completed = _first_event(events, StreamEventType.RETRY_COMPLETED)
        refinement_completed = _first_event(
            events,
            StreamEventType.REFINEMENT_COMPLETED,
        )
        artifact_event = _first_event(events, StreamEventType.ARTIFACT_EXTRACTED)
        preview_event = _first_event(events, StreamEventType.PREVIEW_ARTIFACT)
        review_transition = _first_transition(
            events,
            StreamEventType.NODE_COMPLETED,
            source="review",
            target="refinement",
        )
        refinement_transition = _first_transition(
            events,
            StreamEventType.NODE_COMPLETED,
            source="refinement",
            target="generation",
        )

        self.assertEqual(review_failed.payload["review_outcome"], "needs_refinement")
        self.assertEqual(review_failed.payload["score"], 0.75)
        self.assertEqual(
            review_failed.payload["rationale"],
            "Deterministic review requested refinement: missing_code_block.",
        )
        self.assertEqual(retry_started.payload["retry_count"], 1)
        self.assertEqual(retry_started.payload["retry_reason"], "missing_code_block")
        self.assertEqual(retry_completed.payload["retry_count"], 1)
        self.assertEqual(retry_completed.payload["retry_status"], "passed")
        self.assertEqual(refinement_completed.payload["retry_count"], 1)
        self.assertEqual(
            refinement_completed.payload["retry_reason"],
            "missing_code_block",
        )
        self.assertEqual(
            review_transition.payload["decision_reason"],
            "review_failed_retry_available",
        )
        self.assertEqual(
            refinement_transition.payload["decision_reason"],
            "refinement_completed",
        )
        self.assertEqual(artifact_event.payload["code"], "artifact_extracted")
        self.assertEqual(preview_event.payload["code"], "preview_artifact_prepared")
        self.assertEqual(
            events[-1].payload["answer"],
            "\n".join(
                [
                    "```javascript",
                    "const scene = new THREE.Scene();",
                    "const camera = new THREE.PerspectiveCamera(70, 1, 0.1, 100);",
                    "function animate() {",
                    "  scene.rotation.y += 0.01;",
                    "}",
                    "```",
                ]
            ),
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
            config={"recursion_limit": ASSISTANT_WORKFLOW_RECURSION_LIMIT},
            context={"runtime": _runtime(route_fn=_failing_route)},
        )

        event_types = _event_types(events)
        _assert_subsequence(
            self,
            event_types,
            [
                StreamEventType.NODE_STARTED,
                StreamEventType.NODE_FAILED,
                StreamEventType.ERROR,
                StreamEventType.NODE_STARTED,
                StreamEventType.NODE_COMPLETED,
                StreamEventType.FINAL,
            ],
        )
        routing_failure = _first_transition(
            events,
            StreamEventType.NODE_FAILED,
            source="routing",
            target="failure",
        )
        error_event = _first_event(events, StreamEventType.ERROR)

        self.assertEqual(routing_failure.payload["decision_reason"], "node_exception")
        self.assertEqual(error_event.payload["code"], "workflow_routing_failed")
        self.assertIn("route failed", events[-1].payload["answer"])
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
            config={"recursion_limit": ASSISTANT_WORKFLOW_RECURSION_LIMIT},
            context={"runtime": _runtime(stream_generation=_stream_failed_generation)},
        )

        event_types = _event_types(events)
        _assert_subsequence(
            self,
            event_types,
            [
                StreamEventType.NODE_STARTED,
                StreamEventType.GENERATION_INPUT,
                StreamEventType.ERROR,
                StreamEventType.NODE_FAILED,
                StreamEventType.NODE_STARTED,
                StreamEventType.NODE_COMPLETED,
                StreamEventType.FINAL,
            ],
        )
        provider_error = _first_event(events, StreamEventType.ERROR)
        generation_failure = _first_transition(
            events,
            StreamEventType.NODE_FAILED,
            source="generation",
            target="failure",
        )

        self.assertEqual(provider_error.payload["code"], "provider_unavailable")
        self.assertEqual(
            generation_failure.payload["decision_reason"],
            "generation_provider_failed",
        )
        self.assertIn("Generation failed", events[-1].payload["answer"])
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

        event_types = _event_types(events)

        self.assertIn(StreamEventType.NODE_STARTED, event_types)
        self.assertIn(StreamEventType.NODE_COMPLETED, event_types)
        self.assertIn(StreamEventType.REVIEW_PASSED, event_types)
        self.assertEqual(events[-1].event_type, StreamEventType.FINAL)


def _event_types(events: Sequence[StreamEvent]) -> list[StreamEventType]:
    return [event.event_type for event in events]


def _events_of_type(
    events: Sequence[StreamEvent],
    event_type: StreamEventType,
) -> list[StreamEvent]:
    return [event for event in events if event.event_type is event_type]


def _node_payloads(
    events: Sequence[StreamEvent],
    event_type: StreamEventType,
) -> list[str]:
    return [
        str(event.payload["node"])
        for event in events
        if event.event_type is event_type and "node" in event.payload
    ]


def _first_event(
    events: Sequence[StreamEvent],
    event_type: StreamEventType,
    code: str | None = None,
) -> StreamEvent:
    for event in events:
        if event.event_type is not event_type:
            continue
        if code is not None and event.payload.get("code") != code:
            continue
        return event
    raise AssertionError(f"Missing event {event_type} with code {code!r}.")


def _first_transition(
    events: Sequence[StreamEvent],
    event_type: StreamEventType,
    *,
    source: str,
    target: str | None = None,
) -> StreamEvent:
    for event in events:
        payload_source = event.payload.get("transition_source") or event.payload.get(
            "node"
        )
        if event.event_type is not event_type or payload_source != source:
            continue
        if target is not None and event.payload.get("transition_target") != target:
            continue
        return event
    raise AssertionError(f"Missing transition {event_type} from {source} to {target}.")


def _assert_subsequence(
    testcase: unittest.TestCase,
    values: Sequence[StreamEventType],
    expected: Sequence[StreamEventType],
) -> None:
    start_index = 0
    for expected_value in expected:
        try:
            match_index = values.index(expected_value, start_index)
        except ValueError as exc:
            raise AssertionError(
                f"Missing {expected_value!r} after index {start_index} in {values!r}."
            ) from exc
        testcase.assertGreaterEqual(match_index, start_index)
        start_index = match_index + 1


def _request(
    *,
    query: str = "Generate a Three.js scene.",
    domain: CreativeCodingDomain = CreativeCodingDomain.THREE_JS,
    mode: AssistantMode = AssistantMode.GENERATE,
) -> AssistantRequest:
    return AssistantRequest(
        query=query,
        domain=domain,
        mode=mode,
    )


def _route_generate(request: AssistantRequest) -> RouteDecision:
    return RouteDecision(
        route=RouteName.GENERATE,
        mode=request.mode,
        domain=request.domain,
        capabilities=(RouteCapability.TOOL_USE,),
    )


def _route_generate_without_domains(request: AssistantRequest) -> RouteDecision:
    return RouteDecision(
        route=RouteName.GENERATE,
        mode=request.mode,
        capabilities=(RouteCapability.TOOL_USE,),
    )


def _failing_route(request: AssistantRequest) -> RouteDecision:
    del request
    raise RuntimeError("route failed")


def _runtime(
    *,
    route_fn=_route_generate,
    stream_prompt_inputs=None,
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
        stream_prompt_inputs=stream_prompt_inputs or _empty_streaming_step,
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


def _stream_prompt_inputs_with_builder(
    *,
    builder: StreamEventBuilder,
    request: AssistantRequest,
    decision: RouteDecision,
    assembled_context,
) -> Iterator[StreamEvent]:
    prompt_input_request = build_prompt_input_request(
        assistant_request=request,
        route_decision=decision,
        assembled_context=assembled_context,
    )
    prompt_input = StructuredPromptInputBuilder().build(prompt_input_request)
    yield builder.prompt_input(
        code="prompt_inputs_prepared",
        message="Prompt inputs prepared.",
        prompt_input=prompt_input.model_dump(mode="json"),
    )
    return prompt_input


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


def _unexpected_generation(**kwargs: object) -> Iterator[StreamEvent]:
    del kwargs
    raise AssertionError("Generation should not run while clarification is pending.")
    if False:
        yield StreamEvent(
            event_type=StreamEventType.STATUS,
            sequence=0,
            payload={},
        )


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


def _single_generation(answer: str) -> Callable[..., Iterator[StreamEvent]]:
    return _SequentialGeneration(answer).stream


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
