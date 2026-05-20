import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
    StreamEventType,
)
from creative_coding_assistant.orchestration import (
    WORKFLOW_STEP_ORDER,
    AssistantService,
    WorkflowFailureInfo,
    WorkflowStatus,
    WorkflowStep,
    begin_assistant_workflow,
    complete_workflow_step,
    fail_workflow,
    finish_workflow,
    next_workflow_step,
    restart_workflow_step,
    skip_workflow_step,
    start_workflow_step,
)
from creative_coding_assistant.orchestration.routing import (
    RouteCapability,
    RouteDecision,
    RouteName,
)


class WorkflowFoundationTests(unittest.TestCase):
    def test_workflow_state_tracks_typed_transitions(self) -> None:
        request = AssistantRequest(
            query="Create a Three.js scene.",
            domain=CreativeCodingDomain.THREE_JS,
            mode=AssistantMode.GENERATE,
        )
        decision = RouteDecision(
            route=RouteName.GENERATE,
            mode=AssistantMode.GENERATE,
            domain=CreativeCodingDomain.THREE_JS,
            capabilities=(RouteCapability.TOOL_USE,),
        )

        state = begin_assistant_workflow(request)
        self.assertEqual(state.status, WorkflowStatus.RUNNING)
        self.assertIsNone(state.current_step)
        self.assertFalse(state.is_terminal)

        state = start_workflow_step(state, WorkflowStep.INTAKE)
        self.assertEqual(state.current_step, WorkflowStep.INTAKE)
        state = complete_workflow_step(state, WorkflowStep.INTAKE)
        state = start_workflow_step(state, WorkflowStep.ROUTING)
        state = complete_workflow_step(
            state,
            WorkflowStep.ROUTING,
            route_decision=decision,
        )

        self.assertEqual(
            state.completed_steps,
            (WorkflowStep.INTAKE, WorkflowStep.ROUTING),
        )
        self.assertEqual(state.route_decision, decision)
        self.assertEqual(
            state.event_metadata().model_dump(mode="json"),
            {
                "current_step": None,
                "status": "running",
                "completed_steps": ["intake", "routing"],
                "skipped_steps": [],
            },
        )

    def test_workflow_state_tracks_skipped_future_steps(self) -> None:
        state = begin_assistant_workflow(AssistantRequest(query="Explain shaders."))
        state = start_workflow_step(state, WorkflowStep.REVIEW)
        state = skip_workflow_step(state, WorkflowStep.REVIEW)

        self.assertEqual(state.completed_steps, ())
        self.assertEqual(state.skipped_steps, (WorkflowStep.REVIEW,))
        self.assertIsNone(state.current_step)

    def test_workflow_rejects_out_of_order_completion(self) -> None:
        state = begin_assistant_workflow(AssistantRequest(query="Explain shaders."))
        state = start_workflow_step(state, WorkflowStep.INTAKE)

        with self.assertRaisesRegex(ValueError, "Workflow step mismatch"):
            complete_workflow_step(state, WorkflowStep.ROUTING)

    def test_workflow_finishes_only_from_finalization(self) -> None:
        state = begin_assistant_workflow(AssistantRequest(query="Explain shaders."))

        with self.assertRaisesRegex(ValueError, "finalization"):
            finish_workflow(state, final_answer="Done")

        state = start_workflow_step(state, WorkflowStep.FINALIZATION)
        state = finish_workflow(state, final_answer="Done")

        self.assertTrue(state.is_terminal)
        self.assertEqual(state.status, WorkflowStatus.COMPLETED)
        self.assertEqual(state.final_answer, "Done")
        self.assertIn(WorkflowStep.FINALIZATION, state.completed_steps)

    def test_workflow_can_enter_failed_terminal_state(self) -> None:
        state = begin_assistant_workflow(AssistantRequest(query="Explain shaders."))
        state = start_workflow_step(state, WorkflowStep.GENERATION)
        failure_info = WorkflowFailureInfo(
            step=WorkflowStep.GENERATION,
            code="provider_unavailable",
            message="Provider failed.",
        )

        failed_state = fail_workflow(
            state,
            error_message="Provider failed.",
            failure_info=failure_info,
            final_answer="Generation failed (provider_unavailable): Provider failed.",
        )

        self.assertTrue(failed_state.is_terminal)
        self.assertEqual(failed_state.status, WorkflowStatus.FAILED)
        self.assertIsNone(failed_state.current_step)
        self.assertEqual(failed_state.error_message, "Provider failed.")
        self.assertEqual(failed_state.failure_info, failure_info)
        self.assertIn("Generation failed", failed_state.final_answer)

    def test_workflow_step_order_is_explicit_for_future_graphs(self) -> None:
        self.assertEqual(WORKFLOW_STEP_ORDER[0], WorkflowStep.INTAKE)
        self.assertEqual(WORKFLOW_STEP_ORDER[-1], WorkflowStep.FINALIZATION)
        self.assertEqual(next_workflow_step(WorkflowStep.INTAKE), WorkflowStep.ROUTING)
        self.assertEqual(
            next_workflow_step(WorkflowStep.REVIEW),
            WorkflowStep.REFINEMENT,
        )
        self.assertEqual(
            next_workflow_step(WorkflowStep.REFINEMENT),
            WorkflowStep.FINALIZATION,
        )
        self.assertIsNone(next_workflow_step(WorkflowStep.FINALIZATION))

    def test_workflow_step_can_be_restarted_for_bounded_graph_retries(self) -> None:
        state = begin_assistant_workflow(AssistantRequest(query="Explain shaders."))
        state = start_workflow_step(state, WorkflowStep.GENERATION)
        state = complete_workflow_step(state, WorkflowStep.GENERATION)

        restarted_state = restart_workflow_step(state, WorkflowStep.GENERATION)

        self.assertEqual(restarted_state.current_step, WorkflowStep.GENERATION)
        self.assertEqual(restarted_state.completed_steps, (WorkflowStep.GENERATION,))

    def test_service_stream_event_sequence_remains_backward_compatible(self) -> None:
        service = AssistantService()
        request = AssistantRequest(
            query="Generate a Three.js particle field.",
            domain=CreativeCodingDomain.THREE_JS,
            mode=AssistantMode.GENERATE,
        )

        events = tuple(service.stream(request))

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
        self.assertIn("generate route", events[2].payload["answer"])


if __name__ == "__main__":
    unittest.main()
