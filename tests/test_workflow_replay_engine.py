import unittest

from creative_coding_assistant.orchestration import (
    WorkflowReplayPlan,
    analyze_assistant_execution_graph,
    plan_execution_profiling,
    plan_workflow_replay,
    session_replay_registry,
    workflow_replay_candidate_by_id,
    workflow_replay_candidates_for_status,
)

REQUIRED_WORKFLOW_REPLAY_FIELDS = {
    "candidate_id",
    "replay_id",
    "replay_kind",
    "status",
    "source_id",
    "source_serialization_version",
    "source_item_ids",
    "workflow_node_count",
    "session_replay_profile_count",
    "execution_profile_candidate_count",
    "replay_context_count",
    "advisory_replay_score",
    "replay_pressure",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "workflow_replay_planning_implemented",
    "workflow_replay_execution_implemented",
    "runtime_event_replay_implemented",
    "timeline_reconstruction_implemented",
    "session_recording_implemented",
    "snapshot_capture_implemented",
    "execution_trace_reconstruction_implemented",
    "replay_persistence_implemented",
    "persistent_replay_storage_implemented",
    "workflow_state_mutation_implemented",
    "workflow_control_implemented",
    "workflow_graph_mutation_implemented",
    "graph_compilation_implemented",
    "workflow_execution_implemented",
    "agent_invocation_implemented",
    "node_handler_invocation_implemented",
    "provider_model_routing_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}


class WorkflowReplayEngineTests(unittest.TestCase):
    def test_default_plan_derives_advisory_workflow_replay(self) -> None:
        graph = analyze_assistant_execution_graph()
        session = session_replay_registry()
        profiling = plan_execution_profiling(execution_graph=graph)
        plan = plan_workflow_replay(
            execution_graph=graph,
            session_replay=session,
            execution_profiling=profiling,
        )

        self.assertEqual(plan.role, "workflow_replay_engine")
        self.assertEqual(plan.serialization_version, "workflow_replay_plan.v1")
        self.assertEqual(
            plan.source_graph_serialization_version,
            graph.serialization_version,
        )
        self.assertEqual(
            plan.source_session_replay_serialization_version,
            session.serialization_version,
        )
        self.assertEqual(
            plan.source_execution_profiling_serialization_version,
            profiling.serialization_version,
        )
        self.assertEqual(plan.source_graph_node_count, graph.node_count)
        self.assertEqual(
            plan.source_session_replay_profile_count,
            session.profile_count,
        )
        self.assertEqual(
            plan.source_execution_profile_candidate_count,
            profiling.candidate_count,
        )
        self.assertTrue(plan.failure_path_reachable)
        self.assertEqual(plan.candidate_count, 5)
        self.assertEqual(
            plan.candidate_ids,
            (
                "workflow_replay::topology_replay",
                "workflow_replay::session_timeline_replay",
                "workflow_replay::profiling_context_replay",
                "workflow_replay::failure_path_replay",
                "workflow_replay::storage_boundary_replay",
            ),
        )
        self.assertEqual(plan.replay_candidate_count, 3)
        self.assertEqual(plan.failure_guardrail_count, 1)
        self.assertEqual(plan.storage_guardrail_count, 1)
        self.assertGreater(plan.total_workflow_node_count, 0)
        self.assertGreater(plan.total_session_replay_profile_count, 0)
        self.assertGreater(plan.total_execution_profile_candidate_count, 0)
        self.assertGreater(plan.total_replay_context_count, 0)
        self.assertGreater(plan.highest_advisory_replay_score, 0)
        self.assertGreater(plan.total_advisory_replay_score, 0)
        self.assertEqual(plan.workflow_replay_pressure, "guarded")
        self.assertIn("does not replay runtime events", plan.authority_boundary)
        self.assertTrue(plan.workflow_replay_planning_implemented)
        self.assertFalse(plan.workflow_replay_execution_implemented)
        self.assertFalse(plan.runtime_event_replay_implemented)
        self.assertFalse(plan.timeline_reconstruction_implemented)
        self.assertFalse(plan.session_recording_implemented)
        self.assertFalse(plan.snapshot_capture_implemented)
        self.assertFalse(plan.execution_trace_reconstruction_implemented)
        self.assertFalse(plan.replay_persistence_implemented)
        self.assertFalse(plan.persistent_replay_storage_implemented)
        self.assertFalse(plan.workflow_state_mutation_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.graph_compilation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.node_handler_invocation_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.advisory_only)

    def test_candidates_preserve_replay_execution_boundaries(self) -> None:
        plan = plan_workflow_replay()
        topology = workflow_replay_candidate_by_id(
            "workflow_replay::topology_replay",
            plan,
        )
        storage = workflow_replay_candidate_by_id(
            "workflow_replay::storage_boundary_replay",
            plan,
        )

        self.assertIsNotNone(topology)
        self.assertIsNotNone(storage)
        assert topology is not None
        assert storage is not None
        self.assertEqual(topology.status, "replay_candidate")
        self.assertGreater(topology.workflow_node_count, 0)
        self.assertEqual(storage.status, "storage_guardrail")
        self.assertEqual(storage.advisory_replay_score, 0)
        self.assertEqual(storage.replay_pressure, "guarded")

        for candidate in plan.candidates:
            self.assertEqual(
                set(candidate.model_dump(mode="json")),
                REQUIRED_WORKFLOW_REPLAY_FIELDS,
            )
            self.assertEqual(
                candidate.serialization_version,
                "workflow_replay_candidate.v1",
            )
            self.assertIn(
                "workflow_replay_execution",
                candidate.blocked_runtime_behaviors,
            )
            self.assertIn("runtime_event_replay", candidate.blocked_runtime_behaviors)
            self.assertTrue(candidate.workflow_replay_planning_implemented)
            self.assertFalse(candidate.workflow_replay_execution_implemented)
            self.assertFalse(candidate.runtime_event_replay_implemented)
            self.assertFalse(candidate.timeline_reconstruction_implemented)
            self.assertFalse(candidate.session_recording_implemented)
            self.assertFalse(candidate.snapshot_capture_implemented)
            self.assertFalse(candidate.execution_trace_reconstruction_implemented)
            self.assertFalse(candidate.replay_persistence_implemented)
            self.assertFalse(candidate.persistent_replay_storage_implemented)
            self.assertFalse(candidate.workflow_state_mutation_implemented)
            self.assertFalse(candidate.workflow_control_implemented)
            self.assertFalse(candidate.workflow_graph_mutation_implemented)
            self.assertFalse(candidate.graph_compilation_implemented)
            self.assertFalse(candidate.workflow_execution_implemented)
            self.assertFalse(candidate.agent_invocation_implemented)
            self.assertFalse(candidate.node_handler_invocation_implemented)
            self.assertFalse(candidate.provider_model_routing_implemented)
            self.assertFalse(candidate.retry_triggering_implemented)
            self.assertFalse(candidate.prompt_mutation_implemented)
            self.assertFalse(candidate.persistent_storage_write_implemented)
            self.assertFalse(candidate.generated_output_mutation_implemented)
            self.assertTrue(candidate.advisory_only)

    def test_lookup_helpers_are_stable_and_read_only(self) -> None:
        plan = plan_workflow_replay()
        replay = workflow_replay_candidates_for_status("replay_candidate", plan)
        failure = workflow_replay_candidates_for_status("failure_guardrail", plan)
        storage = workflow_replay_candidates_for_status("storage_guardrail", plan)
        missing = workflow_replay_candidate_by_id("missing", plan)

        self.assertIsNone(missing)
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in replay),
            plan.replay_candidate_ids,
        )
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in failure),
            plan.failure_guardrail_candidate_ids,
        )
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in storage),
            plan.storage_guardrail_candidate_ids,
        )

    def test_plan_rejects_mismatched_candidate_totals(self) -> None:
        plan = plan_workflow_replay()
        payload = plan.model_dump(mode="json")
        payload["candidate_ids"] = ("missing",) + tuple(payload["candidate_ids"][1:])

        with self.assertRaisesRegex(ValueError, "candidate_ids must match"):
            WorkflowReplayPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["total_replay_context_count"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "total_replay_context_count must match",
        ):
            WorkflowReplayPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["workflow_replay_pressure"] = "low"

        with self.assertRaisesRegex(ValueError, "workflow_replay_pressure must match"):
            WorkflowReplayPlan(**payload)

    def test_plan_does_not_declare_runtime_replay_terms(self) -> None:
        plan = plan_workflow_replay()
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for candidate in plan.candidates
                    for field in (
                        candidate.candidate_id,
                        candidate.source_id,
                        *candidate.evidence,
                        *candidate.advisory_actions,
                        *candidate.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "replay_runtime_event(",
            "reconstruct_timeline(",
            "record_session(",
            "capture_snapshot(",
            "persist_replay(",
            "mutate_workflow_state(",
            "control_workflow(",
            "compile_graph(",
            "execute_workflow(",
            "invoke_agent(",
            "route_provider(",
            "trigger_retry(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
