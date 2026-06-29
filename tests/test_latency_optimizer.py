import unittest

from creative_coding_assistant.orchestration import (
    LatencyOptimizationPlan,
    agent_performance_tracking_foundation_registry,
    latency_optimization_candidate_by_id,
    latency_optimization_candidates_for_status,
    latency_threshold_routing_registry,
    optimize_latency,
    plan_parallel_scheduler,
)

REQUIRED_LATENCY_CANDIDATE_FIELDS = {
    "candidate_id",
    "source_schedule_candidate_id",
    "source_scheduling_group_id",
    "stage_id",
    "agent_ids",
    "status",
    "latency_band",
    "advisory_rank",
    "dependency_depth",
    "max_parallel_agents",
    "source_latency_classes",
    "source_latency_threshold_profile_ids",
    "source_latency_bands",
    "blocking_input_count",
    "advisory_latency_savings_score",
    "advisory_latency_pressure_score",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "latency_optimizer_implemented",
    "latency_measurement_implemented",
    "latency_threshold_evaluation_implemented",
    "latency_based_routing_implemented",
    "runtime_selection_implemented",
    "parallel_execution_implemented",
    "async_execution_implemented",
    "workflow_timing_change_implemented",
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


class LatencyOptimizerTests(unittest.TestCase):
    def test_default_plan_derives_advisory_latency_candidates(self) -> None:
        performance = agent_performance_tracking_foundation_registry()
        scheduler = plan_parallel_scheduler()
        thresholds = latency_threshold_routing_registry()
        plan = optimize_latency(
            performance_registry=performance,
            parallel_scheduler=scheduler,
            latency_thresholds=thresholds,
        )

        self.assertEqual(plan.role, "latency_optimizer")
        self.assertEqual(plan.serialization_version, "latency_optimization_plan.v1")
        self.assertEqual(
            plan.source_performance_tracking_serialization_version,
            performance.serialization_version,
        )
        self.assertEqual(
            plan.source_parallel_scheduler_serialization_version,
            scheduler.serialization_version,
        )
        self.assertEqual(
            plan.source_latency_threshold_serialization_version,
            thresholds.serialization_version,
        )
        self.assertEqual(
            plan.source_latency_threshold_profile_ids,
            thresholds.latency_threshold_profile_ids,
        )
        self.assertEqual(plan.source_latency_bands, thresholds.latency_bands)
        self.assertEqual(
            plan.latency_metadata_sources,
            thresholds.latency_metadata_sources,
        )
        self.assertEqual(plan.candidate_count, 6)
        self.assertEqual(
            plan.candidate_ids,
            (
                "latency_optimizer::foundational_context",
                "latency_optimizer::domain_context",
                "latency_optimizer::execution_context",
                "latency_optimizer::quality_review",
                "latency_optimizer::refinement_context",
                "latency_optimizer::final_synthesis",
            ),
        )
        self.assertEqual(plan.optimization_candidate_count, 4)
        self.assertEqual(plan.serial_guardrail_count, 2)
        self.assertEqual(plan.highest_advisory_latency_savings_score, 200)
        self.assertEqual(plan.total_advisory_latency_savings_score, 600)
        self.assertEqual(
            plan.highest_advisory_latency_pressure_score,
            max(
                candidate.advisory_latency_pressure_score
                for candidate in plan.candidates
            ),
        )
        self.assertEqual(
            plan.total_blocking_input_count,
            sum(candidate.blocking_input_count for candidate in plan.candidates),
        )
        self.assertEqual(plan.latency_optimization_pressure, "high")
        self.assertIn("does not measure latency", plan.authority_boundary)
        self.assertTrue(plan.latency_optimizer_implemented)
        self.assertFalse(plan.latency_measurement_implemented)
        self.assertFalse(plan.latency_threshold_evaluation_implemented)
        self.assertFalse(plan.latency_based_routing_implemented)
        self.assertFalse(plan.runtime_selection_implemented)
        self.assertFalse(plan.parallel_execution_implemented)
        self.assertFalse(plan.async_execution_implemented)
        self.assertFalse(plan.workflow_timing_change_implemented)
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

    def test_candidates_preserve_latency_and_execution_boundaries(self) -> None:
        plan = optimize_latency()
        domain = latency_optimization_candidate_by_id(
            "latency_optimizer::domain_context",
            plan,
        )
        refinement = latency_optimization_candidate_by_id(
            "latency_optimizer::refinement_context",
            plan,
        )

        self.assertIsNotNone(domain)
        self.assertIsNotNone(refinement)
        assert domain is not None
        assert refinement is not None
        self.assertEqual(domain.status, "optimization_candidate")
        self.assertEqual(domain.latency_band, "medium")
        self.assertEqual(domain.max_parallel_agents, 3)
        self.assertEqual(domain.advisory_latency_savings_score, 200)
        self.assertTrue(set(domain.source_latency_classes).issubset({"low"}))
        self.assertEqual(refinement.status, "serial_guardrail")
        self.assertEqual(refinement.latency_band, "guarded")
        self.assertEqual(refinement.advisory_latency_savings_score, 0)

        for candidate in plan.candidates:
            self.assertEqual(
                set(candidate.model_dump(mode="json")),
                REQUIRED_LATENCY_CANDIDATE_FIELDS,
            )
            self.assertEqual(
                candidate.serialization_version,
                "latency_optimization_candidate.v1",
            )
            self.assertEqual(
                candidate.source_latency_threshold_profile_ids,
                plan.source_latency_threshold_profile_ids,
            )
            self.assertEqual(candidate.source_latency_bands, plan.source_latency_bands)
            self.assertIn("latency_measurement", candidate.blocked_runtime_behaviors)
            self.assertIn("latency_based_routing", candidate.blocked_runtime_behaviors)
            self.assertTrue(candidate.latency_optimizer_implemented)
            self.assertFalse(candidate.latency_measurement_implemented)
            self.assertFalse(candidate.latency_threshold_evaluation_implemented)
            self.assertFalse(candidate.latency_based_routing_implemented)
            self.assertFalse(candidate.runtime_selection_implemented)
            self.assertFalse(candidate.parallel_execution_implemented)
            self.assertFalse(candidate.async_execution_implemented)
            self.assertFalse(candidate.workflow_timing_change_implemented)
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
        plan = optimize_latency()
        optimization_candidates = latency_optimization_candidates_for_status(
            "optimization_candidate",
            plan,
        )
        guardrails = latency_optimization_candidates_for_status(
            "serial_guardrail",
            plan,
        )
        missing = latency_optimization_candidate_by_id("missing", plan)

        self.assertIsNone(missing)
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in optimization_candidates),
            plan.optimization_candidate_ids,
        )
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in guardrails),
            plan.serial_guardrail_candidate_ids,
        )
        self.assertIs(
            optimization_candidates[0],
            latency_optimization_candidate_by_id(
                optimization_candidates[0].candidate_id,
                plan,
            ),
        )

    def test_plan_rejects_mismatched_candidates_or_scores(self) -> None:
        plan = optimize_latency()
        payload = plan.model_dump(mode="json")
        payload["candidate_ids"] = ("missing",) + tuple(payload["candidate_ids"][1:])

        with self.assertRaisesRegex(ValueError, "candidate_ids must match"):
            LatencyOptimizationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["total_advisory_latency_savings_score"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "total_advisory_latency_savings_score must match",
        ):
            LatencyOptimizationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["latency_optimization_pressure"] = "low"

        with self.assertRaisesRegex(
            ValueError,
            "latency_optimization_pressure must match",
        ):
            LatencyOptimizationPlan(**payload)

    def test_plan_does_not_declare_runtime_latency_terms(self) -> None:
        plan = optimize_latency()
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
                        candidate.source_schedule_candidate_id,
                        candidate.source_scheduling_group_id,
                        candidate.latency_band,
                        *candidate.evidence,
                        *candidate.advisory_actions,
                        *candidate.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "measure_latency(",
            "evaluate_threshold(",
            "route_by_latency(",
            "select_runtime(",
            "asyncio.gather(",
            "create_task(",
            "execute_parallel(",
            "change_workflow_timing(",
            "compile_graph(",
            "execute_workflow(",
            "invoke_agent(",
            "route_provider(",
            "trigger_retry(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
