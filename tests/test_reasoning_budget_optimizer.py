import unittest

from creative_coding_assistant.orchestration import (
    ReasoningBudgetOptimizationPlan,
    evaluate_budget_policies,
    optimize_reasoning_budget,
    performance_benchmark_scenario_by_id,
    plan_context_budget,
    plan_performance_benchmarking,
    predict_performance,
    reasoning_budget_recommendation_by_id,
    reasoning_budget_recommendations_for_status,
)

REQUIRED_REASONING_BUDGET_FIELDS = {
    "recommendation_id",
    "budget_id",
    "budget_kind",
    "status",
    "source_id",
    "source_serialization_version",
    "source_item_ids",
    "advisory_reasoning_tokens",
    "advisory_reserve_tokens",
    "advisory_pressure_score",
    "reasoning_budget_pressure",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "reasoning_budget_optimizer_planning_implemented",
    "reasoning_budget_enforcement_implemented",
    "runtime_reasoning_token_allocation_implemented",
    "budget_enforcement_implemented",
    "context_trimming_implemented",
    "prompt_compression_implemented",
    "memory_summarization_implemented",
    "hitl_request_emission_implemented",
    "provider_model_routing_implemented",
    "model_selection_implemented",
    "workflow_control_implemented",
    "workflow_timing_change_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_execution_implemented",
    "agent_invocation_implemented",
    "node_handler_invocation_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}


class ReasoningBudgetOptimizerTests(unittest.TestCase):
    def test_default_plan_derives_advisory_reasoning_budget(self) -> None:
        context = plan_context_budget()
        prediction = predict_performance()
        benchmarking = plan_performance_benchmarking(
            performance_prediction=prediction,
        )
        policies = evaluate_budget_policies()
        plan = optimize_reasoning_budget(
            context_budget=context,
            performance_prediction=prediction,
            performance_benchmarking=benchmarking,
            budget_policies=policies,
        )

        self.assertEqual(plan.role, "reasoning_budget_optimizer")
        self.assertEqual(
            plan.serialization_version,
            "reasoning_budget_optimization_plan.v1",
        )
        self.assertEqual(
            plan.source_context_budget_serialization_version,
            context.serialization_version,
        )
        self.assertEqual(
            plan.source_performance_prediction_serialization_version,
            prediction.serialization_version,
        )
        self.assertEqual(
            plan.source_performance_benchmarking_serialization_version,
            benchmarking.serialization_version,
        )
        self.assertEqual(
            plan.source_budget_policy_serialization_version,
            policies.serialization_version,
        )
        self.assertEqual(plan.recommendation_count, 4)
        self.assertEqual(
            plan.recommendation_ids,
            (
                "reasoning_budget::context_reasoning_allocation",
                "reasoning_budget::performance_reasoning_reserve",
                "reasoning_budget::benchmark_reasoning_reserve",
                "reasoning_budget::budget_policy_review",
            ),
        )
        self.assertEqual(plan.optimization_candidate_count, 2)
        self.assertEqual(plan.reserve_guardrail_count, 1)
        self.assertEqual(plan.review_guardrail_count, 1)
        self.assertGreater(plan.total_advisory_reasoning_tokens, 0)
        self.assertGreater(plan.total_advisory_reserve_tokens, 0)
        self.assertGreater(plan.highest_advisory_pressure_score, 0)
        self.assertGreater(plan.total_advisory_pressure_score, 0)
        self.assertEqual(plan.reasoning_budget_pressure, "guarded")
        self.assertIn("does not enforce budgets", plan.authority_boundary)
        self.assertTrue(plan.reasoning_budget_optimizer_planning_implemented)
        self.assertFalse(plan.reasoning_budget_enforcement_implemented)
        self.assertFalse(plan.runtime_reasoning_token_allocation_implemented)
        self.assertFalse(plan.budget_enforcement_implemented)
        self.assertFalse(plan.context_trimming_implemented)
        self.assertFalse(plan.prompt_compression_implemented)
        self.assertFalse(plan.memory_summarization_implemented)
        self.assertFalse(plan.hitl_request_emission_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.model_selection_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_timing_change_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.node_handler_invocation_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.advisory_only)

        reserve = performance_benchmark_scenario_by_id(
            "performance_benchmark::throughput_benchmark",
            benchmarking,
        )
        self.assertIsNotNone(reserve)

    def test_recommendations_preserve_budget_boundaries(self) -> None:
        plan = optimize_reasoning_budget()
        context = reasoning_budget_recommendation_by_id(
            "reasoning_budget::context_reasoning_allocation",
            plan,
        )
        review = reasoning_budget_recommendation_by_id(
            "reasoning_budget::budget_policy_review",
            plan,
        )

        self.assertIsNotNone(context)
        self.assertIsNotNone(review)
        assert context is not None
        assert review is not None
        self.assertEqual(context.status, "optimization_candidate")
        self.assertGreater(context.advisory_reasoning_tokens, 0)
        self.assertEqual(review.status, "review_guardrail")
        self.assertEqual(review.reasoning_budget_pressure, "guarded")

        for recommendation in plan.recommendations:
            self.assertEqual(
                set(recommendation.model_dump(mode="json")),
                REQUIRED_REASONING_BUDGET_FIELDS,
            )
            self.assertEqual(
                recommendation.serialization_version,
                "reasoning_budget_recommendation.v1",
            )
            self.assertIn(
                "reasoning_budget_enforcement",
                recommendation.blocked_runtime_behaviors,
            )
            self.assertTrue(
                recommendation.reasoning_budget_optimizer_planning_implemented,
            )
            self.assertFalse(recommendation.reasoning_budget_enforcement_implemented)
            self.assertFalse(
                recommendation.runtime_reasoning_token_allocation_implemented,
            )
            self.assertFalse(recommendation.budget_enforcement_implemented)
            self.assertFalse(recommendation.context_trimming_implemented)
            self.assertFalse(recommendation.prompt_compression_implemented)
            self.assertFalse(recommendation.memory_summarization_implemented)
            self.assertFalse(recommendation.hitl_request_emission_implemented)
            self.assertFalse(recommendation.provider_model_routing_implemented)
            self.assertFalse(recommendation.model_selection_implemented)
            self.assertFalse(recommendation.workflow_control_implemented)
            self.assertFalse(recommendation.workflow_timing_change_implemented)
            self.assertFalse(recommendation.workflow_graph_mutation_implemented)
            self.assertFalse(recommendation.workflow_execution_implemented)
            self.assertFalse(recommendation.agent_invocation_implemented)
            self.assertFalse(recommendation.node_handler_invocation_implemented)
            self.assertFalse(recommendation.retry_triggering_implemented)
            self.assertFalse(recommendation.prompt_mutation_implemented)
            self.assertFalse(recommendation.persistent_storage_write_implemented)
            self.assertFalse(recommendation.generated_output_mutation_implemented)
            self.assertTrue(recommendation.advisory_only)

    def test_lookup_helpers_are_stable_and_read_only(self) -> None:
        plan = optimize_reasoning_budget()
        optimization = reasoning_budget_recommendations_for_status(
            "optimization_candidate",
            plan,
        )
        reserve = reasoning_budget_recommendations_for_status(
            "reserve_guardrail",
            plan,
        )
        review = reasoning_budget_recommendations_for_status(
            "review_guardrail",
            plan,
        )
        missing = reasoning_budget_recommendation_by_id("missing", plan)

        self.assertIsNone(missing)
        self.assertEqual(
            tuple(recommendation.recommendation_id for recommendation in optimization),
            plan.optimization_candidate_ids,
        )
        self.assertEqual(
            tuple(recommendation.recommendation_id for recommendation in reserve),
            plan.reserve_guardrail_ids,
        )
        self.assertEqual(
            tuple(recommendation.recommendation_id for recommendation in review),
            plan.review_guardrail_ids,
        )

    def test_plan_rejects_mismatched_recommendation_totals(self) -> None:
        plan = optimize_reasoning_budget()
        payload = plan.model_dump(mode="json")
        payload["recommendation_ids"] = ("missing",) + tuple(
            payload["recommendation_ids"][1:]
        )

        with self.assertRaisesRegex(ValueError, "recommendation_ids must match"):
            ReasoningBudgetOptimizationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["total_advisory_reasoning_tokens"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "total_advisory_reasoning_tokens must match",
        ):
            ReasoningBudgetOptimizationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["reasoning_budget_pressure"] = "low"

        with self.assertRaisesRegex(
            ValueError,
            "reasoning_budget_pressure must match",
        ):
            ReasoningBudgetOptimizationPlan(**payload)

    def test_plan_does_not_declare_runtime_budget_terms(self) -> None:
        plan = optimize_reasoning_budget()
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for recommendation in plan.recommendations
                    for field in (
                        recommendation.recommendation_id,
                        recommendation.source_id,
                        *recommendation.evidence,
                        *recommendation.advisory_actions,
                        *recommendation.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "enforce_reasoning_budget(",
            "allocate_reasoning_tokens(",
            "enforce_budget(",
            "trim_context(",
            "compress_prompt(",
            "summarize_memory(",
            "emit_hitl_request(",
            "select_model(",
            "route_provider(",
            "control_workflow(",
            "execute_workflow(",
            "invoke_agent(",
            "trigger_retry(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
