import unittest

from creative_coding_assistant.orchestration import (
    WorkflowPruningPlan,
    analyze_assistant_execution_graph,
    analyze_workflow_complexity,
    analyze_workflow_cost,
    plan_workflow_pruning,
    workflow_pruning_candidate_by_id,
    workflow_pruning_candidates_for_status,
)

REQUIRED_CANDIDATE_FIELDS = {
    "candidate_id",
    "candidate_kind",
    "source_id",
    "source_serialization_version",
    "status",
    "priority",
    "estimated_token_savings",
    "retained_token_cost",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "workflow_pruning_implemented",
    "workflow_node_removal_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_order_mutation_implemented",
    "execution_path_selection_implemented",
    "strategy_selection_implemented",
    "provider_model_routing_implemented",
    "workflow_control_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "planning_only",
}


class WorkflowPruningTests(unittest.TestCase):
    def test_default_plan_derives_advisory_pruning_candidates(self) -> None:
        graph = analyze_assistant_execution_graph()
        costs = analyze_workflow_cost(execution_graph=graph)
        complexity = analyze_workflow_complexity(
            execution_graph=graph,
            cost_analysis=costs,
        )
        plan = plan_workflow_pruning(
            execution_graph=graph,
            cost_analysis=costs,
            complexity_analysis=complexity,
        )

        self.assertEqual(plan.role, "workflow_pruning_planner")
        self.assertEqual(plan.serialization_version, "workflow_pruning_plan.v1")
        self.assertEqual(
            plan.source_graph_serialization_version,
            graph.serialization_version,
        )
        self.assertEqual(
            plan.source_cost_serialization_version,
            costs.serialization_version,
        )
        self.assertEqual(
            plan.source_complexity_serialization_version,
            complexity.serialization_version,
        )
        self.assertEqual(plan.candidate_count, len(plan.candidates))
        self.assertEqual(plan.estimated_token_savings, costs.retry_token_reserve)
        self.assertAlmostEqual(
            plan.savings_ratio,
            plan.estimated_token_savings / costs.worst_case_token_estimate,
        )
        self.assertEqual(plan.pruning_pressure, "high")
        self.assertIn("does not remove workflow nodes", plan.authority_boundary)
        self.assertTrue(plan.workflow_pruning_implemented)
        self.assertFalse(plan.workflow_node_removal_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_order_mutation_implemented)
        self.assertFalse(plan.execution_path_selection_implemented)
        self.assertFalse(plan.strategy_selection_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.planning_only)

    def test_candidates_cover_retained_review_and_prunable_surfaces(self) -> None:
        plan = plan_workflow_pruning()
        prunable = workflow_pruning_candidate_by_id(
            "workflow_pruning::reserve::retry_path",
            plan,
        )
        refinement = workflow_pruning_candidate_by_id(
            "workflow_pruning::node::refinement",
            plan,
        )
        failure_reserve = workflow_pruning_candidate_by_id(
            "workflow_pruning::reserve::failure_path",
            plan,
        )
        branching = workflow_pruning_candidate_by_id(
            "workflow_pruning::factor::branching",
            plan,
        )

        self.assertIsNotNone(prunable)
        self.assertIsNotNone(refinement)
        self.assertIsNotNone(failure_reserve)
        self.assertIsNotNone(branching)
        assert prunable is not None
        assert refinement is not None
        assert failure_reserve is not None
        assert branching is not None
        self.assertEqual(prunable.status, "prunable")
        self.assertGreater(prunable.estimated_token_savings, 0)
        self.assertEqual(refinement.status, "review")
        self.assertEqual(refinement.estimated_token_savings, 0)
        self.assertEqual(failure_reserve.status, "retain")
        self.assertEqual(branching.status, "review")

        for candidate in plan.candidates:
            self.assertEqual(
                set(candidate.model_dump(mode="json")),
                REQUIRED_CANDIDATE_FIELDS,
            )
            self.assertEqual(
                candidate.serialization_version,
                "workflow_pruning_candidate.v1",
            )
            self.assertIn(
                "workflow_graph_mutation",
                candidate.blocked_runtime_behaviors,
            )
            self.assertTrue(candidate.workflow_pruning_implemented)
            self.assertFalse(candidate.workflow_node_removal_implemented)
            self.assertFalse(candidate.workflow_graph_mutation_implemented)
            self.assertFalse(candidate.workflow_order_mutation_implemented)
            self.assertFalse(candidate.execution_path_selection_implemented)
            self.assertFalse(candidate.strategy_selection_implemented)
            self.assertFalse(candidate.provider_model_routing_implemented)
            self.assertFalse(candidate.workflow_control_implemented)
            self.assertFalse(candidate.retry_triggering_implemented)
            self.assertFalse(candidate.prompt_mutation_implemented)
            self.assertFalse(candidate.persistent_storage_write_implemented)
            self.assertFalse(candidate.generated_output_mutation_implemented)
            self.assertTrue(candidate.planning_only)

    def test_lookup_helpers_are_stable_and_read_only(self) -> None:
        plan = plan_workflow_pruning()
        prunable = workflow_pruning_candidates_for_status("prunable", plan)
        retained = workflow_pruning_candidates_for_status("retain", plan)
        review = workflow_pruning_candidates_for_status("review", plan)
        missing = workflow_pruning_candidate_by_id("missing", plan)

        self.assertIsNone(missing)
        self.assertEqual(len(prunable), plan.prunable_candidate_count)
        self.assertEqual(len(retained), plan.retained_candidate_count)
        self.assertEqual(len(review), plan.review_candidate_count)
        self.assertIs(
            prunable[0],
            workflow_pruning_candidate_by_id(prunable[0].candidate_id, plan),
        )

    def test_plan_rejects_mismatched_candidates_or_totals(self) -> None:
        plan = plan_workflow_pruning()
        payload = plan.model_dump(mode="json")
        payload["candidate_ids"] = ("missing",) + tuple(payload["candidate_ids"][1:])

        with self.assertRaisesRegex(ValueError, "candidate_ids must match"):
            WorkflowPruningPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["estimated_token_savings"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "estimated_token_savings must match",
        ):
            WorkflowPruningPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["pruning_pressure"] = "low"

        with self.assertRaisesRegex(ValueError, "pruning_pressure must match"):
            WorkflowPruningPlan(**payload)

    def test_plan_does_not_declare_runtime_mutation_terms(self) -> None:
        plan = plan_workflow_pruning()
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
            "remove_node(",
            "mutate_graph(",
            "select_execution_path(",
            "select_strategy(",
            "route_provider(",
            "trigger_retry(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
