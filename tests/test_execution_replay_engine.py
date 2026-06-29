import unittest

from creative_coding_assistant.orchestration import (
    ExecutionReplayPlan,
    execution_replay_candidate_by_id,
    execution_replay_candidates_for_status,
    execution_replay_registry,
    plan_execution_profiling,
    plan_execution_replay,
    plan_workflow_replay,
)

REQUIRED_EXECUTION_REPLAY_FIELDS = {
    "candidate_id",
    "replay_id",
    "replay_kind",
    "status",
    "source_id",
    "source_serialization_version",
    "source_item_ids",
    "execution_replay_profile_count",
    "workflow_replay_candidate_count",
    "execution_profile_candidate_count",
    "route_count",
    "replay_context_count",
    "advisory_replay_score",
    "replay_pressure",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "execution_replay_planning_implemented",
    "execution_replay_execution_implemented",
    "provider_execution_implemented",
    "local_provider_execution_implemented",
    "cloud_provider_execution_implemented",
    "model_selection_implemented",
    "provider_model_routing_implemented",
    "execution_trace_reconstruction_implemented",
    "runtime_event_replay_implemented",
    "workflow_replay_execution_implemented",
    "replay_persistence_implemented",
    "persistent_replay_storage_implemented",
    "cost_scoring_implemented",
    "quality_scoring_implemented",
    "quality_evaluation_implemented",
    "workflow_control_implemented",
    "human_input_request_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}


class ExecutionReplayEngineTests(unittest.TestCase):
    def test_default_plan_derives_advisory_execution_replay(self) -> None:
        execution_registry = execution_replay_registry()
        workflow = plan_workflow_replay()
        profiling = plan_execution_profiling()
        plan = plan_execution_replay(
            execution_replay=execution_registry,
            workflow_replay=workflow,
            execution_profiling=profiling,
        )

        self.assertEqual(plan.role, "execution_replay_engine")
        self.assertEqual(plan.serialization_version, "execution_replay_plan.v1")
        self.assertEqual(
            plan.source_execution_replay_serialization_version,
            execution_registry.serialization_version,
        )
        self.assertEqual(
            plan.source_workflow_replay_serialization_version,
            workflow.serialization_version,
        )
        self.assertEqual(
            plan.source_execution_profiling_serialization_version,
            profiling.serialization_version,
        )
        self.assertEqual(
            plan.source_execution_replay_profile_count,
            execution_registry.profile_count,
        )
        self.assertEqual(
            plan.source_workflow_replay_candidate_count,
            workflow.candidate_count,
        )
        self.assertEqual(
            plan.source_execution_profile_candidate_count,
            profiling.candidate_count,
        )
        self.assertEqual(
            plan.source_route_count,
            len(execution_registry.route_names),
        )
        self.assertEqual(plan.candidate_count, 5)
        self.assertEqual(
            plan.candidate_ids,
            (
                "execution_replay::execution_replay_context",
                "execution_replay::workflow_replay_context",
                "execution_replay::provider_selection_boundary",
                "execution_replay::cost_quality_boundary",
                "execution_replay::storage_boundary",
            ),
        )
        self.assertEqual(plan.replay_candidate_count, 2)
        self.assertEqual(plan.provider_guardrail_count, 1)
        self.assertEqual(plan.scoring_guardrail_count, 1)
        self.assertEqual(plan.storage_guardrail_count, 1)
        self.assertGreater(plan.total_execution_replay_profile_count, 0)
        self.assertGreater(plan.total_workflow_replay_candidate_count, 0)
        self.assertGreater(plan.total_execution_profile_candidate_count, 0)
        self.assertGreater(plan.total_route_count, 0)
        self.assertGreater(plan.total_replay_context_count, 0)
        self.assertGreater(plan.highest_advisory_replay_score, 0)
        self.assertGreater(plan.total_advisory_replay_score, 0)
        self.assertEqual(plan.execution_replay_pressure, "guarded")
        self.assertIn("does not execute providers", plan.authority_boundary)
        self.assertTrue(plan.execution_replay_planning_implemented)
        self.assertFalse(plan.execution_replay_execution_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.local_provider_execution_implemented)
        self.assertFalse(plan.cloud_provider_execution_implemented)
        self.assertFalse(plan.model_selection_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.execution_trace_reconstruction_implemented)
        self.assertFalse(plan.runtime_event_replay_implemented)
        self.assertFalse(plan.workflow_replay_execution_implemented)
        self.assertFalse(plan.replay_persistence_implemented)
        self.assertFalse(plan.persistent_replay_storage_implemented)
        self.assertFalse(plan.cost_scoring_implemented)
        self.assertFalse(plan.quality_scoring_implemented)
        self.assertFalse(plan.quality_evaluation_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.human_input_request_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.advisory_only)

    def test_candidates_preserve_execution_replay_boundaries(self) -> None:
        plan = plan_execution_replay()
        replay = execution_replay_candidate_by_id(
            "execution_replay::execution_replay_context",
            plan,
        )
        provider = execution_replay_candidate_by_id(
            "execution_replay::provider_selection_boundary",
            plan,
        )
        storage = execution_replay_candidate_by_id(
            "execution_replay::storage_boundary",
            plan,
        )

        self.assertIsNotNone(replay)
        self.assertIsNotNone(provider)
        self.assertIsNotNone(storage)
        assert replay is not None
        assert provider is not None
        assert storage is not None
        self.assertEqual(replay.status, "replay_candidate")
        self.assertGreater(replay.execution_replay_profile_count, 0)
        self.assertEqual(provider.status, "provider_guardrail")
        self.assertEqual(provider.replay_pressure, "guarded")
        self.assertEqual(storage.status, "storage_guardrail")
        self.assertEqual(storage.advisory_replay_score, 0)

        for candidate in plan.candidates:
            self.assertEqual(
                set(candidate.model_dump(mode="json")),
                REQUIRED_EXECUTION_REPLAY_FIELDS,
            )
            self.assertEqual(
                candidate.serialization_version,
                "execution_replay_candidate.v1",
            )
            self.assertIn(
                "execution_replay_execution",
                candidate.blocked_runtime_behaviors,
            )
            self.assertIn("provider_execution", candidate.blocked_runtime_behaviors)
            self.assertTrue(candidate.execution_replay_planning_implemented)
            self.assertFalse(candidate.execution_replay_execution_implemented)
            self.assertFalse(candidate.provider_execution_implemented)
            self.assertFalse(candidate.local_provider_execution_implemented)
            self.assertFalse(candidate.cloud_provider_execution_implemented)
            self.assertFalse(candidate.model_selection_implemented)
            self.assertFalse(candidate.provider_model_routing_implemented)
            self.assertFalse(candidate.execution_trace_reconstruction_implemented)
            self.assertFalse(candidate.runtime_event_replay_implemented)
            self.assertFalse(candidate.workflow_replay_execution_implemented)
            self.assertFalse(candidate.replay_persistence_implemented)
            self.assertFalse(candidate.persistent_replay_storage_implemented)
            self.assertFalse(candidate.cost_scoring_implemented)
            self.assertFalse(candidate.quality_scoring_implemented)
            self.assertFalse(candidate.quality_evaluation_implemented)
            self.assertFalse(candidate.workflow_control_implemented)
            self.assertFalse(candidate.human_input_request_implemented)
            self.assertFalse(candidate.retry_triggering_implemented)
            self.assertFalse(candidate.prompt_mutation_implemented)
            self.assertFalse(candidate.persistent_storage_write_implemented)
            self.assertFalse(candidate.generated_output_mutation_implemented)
            self.assertTrue(candidate.advisory_only)

    def test_lookup_helpers_are_stable_and_read_only(self) -> None:
        plan = plan_execution_replay()
        replay = execution_replay_candidates_for_status("replay_candidate", plan)
        provider = execution_replay_candidates_for_status("provider_guardrail", plan)
        scoring = execution_replay_candidates_for_status("scoring_guardrail", plan)
        storage = execution_replay_candidates_for_status("storage_guardrail", plan)
        missing = execution_replay_candidate_by_id("missing", plan)

        self.assertIsNone(missing)
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in replay),
            plan.replay_candidate_ids,
        )
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in provider),
            plan.provider_guardrail_candidate_ids,
        )
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in scoring),
            plan.scoring_guardrail_candidate_ids,
        )
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in storage),
            plan.storage_guardrail_candidate_ids,
        )

    def test_plan_rejects_mismatched_candidate_totals(self) -> None:
        plan = plan_execution_replay()
        payload = plan.model_dump(mode="json")
        payload["candidate_ids"] = ("missing",) + tuple(payload["candidate_ids"][1:])

        with self.assertRaisesRegex(ValueError, "candidate_ids must match"):
            ExecutionReplayPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["total_replay_context_count"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "total_replay_context_count must match",
        ):
            ExecutionReplayPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["execution_replay_pressure"] = "low"

        with self.assertRaisesRegex(ValueError, "execution_replay_pressure must match"):
            ExecutionReplayPlan(**payload)

    def test_plan_does_not_declare_runtime_replay_terms(self) -> None:
        plan = plan_execution_replay()
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
            "execute_provider(",
            "execute_local_provider(",
            "execute_cloud_provider(",
            "select_model(",
            "route_provider(",
            "reconstruct_trace(",
            "replay_runtime_event(",
            "persist_replay(",
            "score_cost(",
            "score_quality(",
            "control_workflow(",
            "request_human_input(",
            "trigger_retry(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
