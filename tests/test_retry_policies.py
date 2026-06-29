import unittest

from creative_coding_assistant.orchestration import (
    RetryPolicyPlan,
    forecast_execution_cost,
    optimize_streaming,
    plan_retry_policies,
    plan_workflow_pruning,
    retry_policy_candidate_by_id,
    retry_policy_candidates_for_status,
)
from creative_coding_assistant.orchestration.workflow_review import (
    MAX_WORKFLOW_REFINEMENT_COUNT,
)

REQUIRED_RETRY_POLICY_FIELDS = {
    "candidate_id",
    "policy_id",
    "policy_kind",
    "status",
    "source_id",
    "source_serialization_version",
    "max_retry_attempts",
    "retry_budget_tokens",
    "advisory_retry_score",
    "failure_path_required",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "retry_policy_planning_implemented",
    "retry_triggering_implemented",
    "refinement_triggering_implemented",
    "workflow_control_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_order_mutation_implemented",
    "graph_compilation_implemented",
    "workflow_execution_implemented",
    "agent_invocation_implemented",
    "node_handler_invocation_implemented",
    "provider_model_routing_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}


class RetryPolicyTests(unittest.TestCase):
    def test_default_plan_derives_advisory_retry_policies(self) -> None:
        pruning = plan_workflow_pruning()
        forecast = forecast_execution_cost(pruning_plan=pruning)
        streaming = optimize_streaming()
        plan = plan_retry_policies(
            pruning_plan=pruning,
            cost_forecast=forecast,
            streaming_optimization=streaming,
        )

        self.assertEqual(plan.role, "retry_policy_planner")
        self.assertEqual(plan.serialization_version, "retry_policy_plan.v1")
        self.assertEqual(
            plan.source_pruning_serialization_version,
            pruning.serialization_version,
        )
        self.assertEqual(
            plan.source_cost_forecast_serialization_version,
            forecast.serialization_version,
        )
        self.assertEqual(
            plan.source_streaming_optimization_serialization_version,
            streaming.serialization_version,
        )
        self.assertEqual(
            plan.workflow_refinement_limit,
            MAX_WORKFLOW_REFINEMENT_COUNT,
        )
        self.assertTrue(plan.bounded_retry_cycle_detected)
        self.assertTrue(plan.failure_path_reachable)
        self.assertEqual(plan.candidate_count, 4)
        self.assertEqual(
            plan.candidate_ids,
            (
                "retry_policy::review_refinement",
                "retry_policy::generation_failure",
                "retry_policy::stream_failure_visibility",
                "retry_policy::cost_retry_reserve",
            ),
        )
        self.assertEqual(plan.bounded_retry_candidate_count, 1)
        self.assertEqual(plan.guardrail_candidate_count, 2)
        self.assertEqual(plan.review_only_candidate_count, 1)
        self.assertEqual(plan.max_retry_attempts, MAX_WORKFLOW_REFINEMENT_COUNT)
        self.assertGreater(plan.total_retry_budget_tokens, 0)
        self.assertEqual(
            plan.highest_advisory_retry_score,
            MAX_WORKFLOW_REFINEMENT_COUNT * 200,
        )
        self.assertEqual(plan.retry_policy_pressure, "high")
        self.assertIn("does not trigger retries", plan.authority_boundary)
        self.assertTrue(plan.retry_policy_planning_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.refinement_triggering_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_order_mutation_implemented)
        self.assertFalse(plan.graph_compilation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.node_handler_invocation_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.advisory_only)

    def test_candidates_preserve_retry_trigger_boundaries(self) -> None:
        plan = plan_retry_policies()
        review = retry_policy_candidate_by_id(
            "retry_policy::review_refinement",
            plan,
        )
        failure = retry_policy_candidate_by_id(
            "retry_policy::generation_failure",
            plan,
        )
        reserve = retry_policy_candidate_by_id(
            "retry_policy::cost_retry_reserve",
            plan,
        )

        self.assertIsNotNone(review)
        self.assertIsNotNone(failure)
        self.assertIsNotNone(reserve)
        assert review is not None
        assert failure is not None
        assert reserve is not None
        self.assertEqual(review.status, "bounded_retry_candidate")
        self.assertEqual(review.max_retry_attempts, MAX_WORKFLOW_REFINEMENT_COUNT)
        self.assertGreater(review.retry_budget_tokens, 0)
        self.assertEqual(failure.status, "guardrail")
        self.assertEqual(failure.max_retry_attempts, 0)
        self.assertEqual(reserve.status, "review_only")

        for candidate in plan.candidates:
            self.assertEqual(
                set(candidate.model_dump(mode="json")),
                REQUIRED_RETRY_POLICY_FIELDS,
            )
            self.assertEqual(
                candidate.serialization_version,
                "retry_policy_candidate.v1",
            )
            self.assertTrue(candidate.failure_path_required)
            self.assertIn("retry_triggering", candidate.blocked_runtime_behaviors)
            self.assertTrue(candidate.retry_policy_planning_implemented)
            self.assertFalse(candidate.retry_triggering_implemented)
            self.assertFalse(candidate.refinement_triggering_implemented)
            self.assertFalse(candidate.workflow_control_implemented)
            self.assertFalse(candidate.workflow_graph_mutation_implemented)
            self.assertFalse(candidate.workflow_order_mutation_implemented)
            self.assertFalse(candidate.graph_compilation_implemented)
            self.assertFalse(candidate.workflow_execution_implemented)
            self.assertFalse(candidate.agent_invocation_implemented)
            self.assertFalse(candidate.node_handler_invocation_implemented)
            self.assertFalse(candidate.provider_model_routing_implemented)
            self.assertFalse(candidate.prompt_mutation_implemented)
            self.assertFalse(candidate.persistent_storage_write_implemented)
            self.assertFalse(candidate.generated_output_mutation_implemented)
            self.assertTrue(candidate.advisory_only)

    def test_lookup_helpers_are_stable_and_read_only(self) -> None:
        plan = plan_retry_policies()
        bounded = retry_policy_candidates_for_status(
            "bounded_retry_candidate",
            plan,
        )
        guardrails = retry_policy_candidates_for_status("guardrail", plan)
        review_only = retry_policy_candidates_for_status("review_only", plan)
        missing = retry_policy_candidate_by_id("missing", plan)

        self.assertIsNone(missing)
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in bounded),
            plan.bounded_retry_candidate_ids,
        )
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in guardrails),
            plan.guardrail_candidate_ids,
        )
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in review_only),
            plan.review_only_candidate_ids,
        )

    def test_plan_rejects_mismatched_candidates_or_scores(self) -> None:
        plan = plan_retry_policies()
        payload = plan.model_dump(mode="json")
        payload["candidate_ids"] = ("missing",) + tuple(payload["candidate_ids"][1:])

        with self.assertRaisesRegex(ValueError, "candidate_ids must match"):
            RetryPolicyPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["total_retry_budget_tokens"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "total_retry_budget_tokens must match",
        ):
            RetryPolicyPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["retry_policy_pressure"] = "low"

        with self.assertRaisesRegex(ValueError, "retry_policy_pressure must match"):
            RetryPolicyPlan(**payload)

    def test_plan_does_not_declare_runtime_retry_terms(self) -> None:
        plan = plan_retry_policies()
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
            "trigger_retry(",
            "trigger_refinement(",
            "control_workflow(",
            "mutate_graph(",
            "compile_graph(",
            "execute_workflow(",
            "invoke_agent(",
            "invoke_node_handler(",
            "route_provider(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
