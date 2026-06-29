import unittest

from creative_coding_assistant.orchestration import (
    LoadBalancerPlan,
    build_provider_capability_matrix,
    load_balance_candidate_by_id,
    load_balance_candidates_for_status,
    optimize_latency,
    plan_async_execution,
    plan_load_balancer,
    plan_parallel_scheduler,
    plan_retry_policies,
)

REQUIRED_LOAD_BALANCE_FIELDS = {
    "candidate_id",
    "balance_id",
    "balance_kind",
    "status",
    "source_id",
    "source_serialization_version",
    "source_candidate_ids",
    "advisory_load_units",
    "advisory_capacity_slots",
    "advisory_load_score",
    "load_pressure",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "load_balancer_planning_implemented",
    "load_balancing_runtime_implemented",
    "request_distribution_implemented",
    "traffic_shaping_implemented",
    "capacity_enforcement_implemented",
    "provider_selection_implemented",
    "automatic_model_selection_implemented",
    "provider_model_routing_implemented",
    "parallel_execution_implemented",
    "async_runtime_execution_implemented",
    "workflow_timing_change_implemented",
    "workflow_graph_mutation_implemented",
    "graph_compilation_implemented",
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


class LoadBalancerTests(unittest.TestCase):
    def test_default_plan_derives_advisory_load_candidates(self) -> None:
        scheduler = plan_parallel_scheduler()
        latency = optimize_latency(parallel_scheduler=scheduler)
        async_plan = plan_async_execution(
            parallel_scheduler=scheduler,
            latency_optimization=latency,
        )
        retry = plan_retry_policies()
        provider_matrix = build_provider_capability_matrix()
        plan = plan_load_balancer(
            async_execution=async_plan,
            latency_optimization=latency,
            retry_policy=retry,
            provider_capability=provider_matrix,
        )

        self.assertEqual(plan.role, "load_balancer")
        self.assertEqual(plan.serialization_version, "load_balancer_plan.v1")
        self.assertEqual(
            plan.source_async_execution_serialization_version,
            async_plan.serialization_version,
        )
        self.assertEqual(
            plan.source_latency_optimization_serialization_version,
            latency.serialization_version,
        )
        self.assertEqual(
            plan.source_retry_policy_serialization_version,
            retry.serialization_version,
        )
        self.assertEqual(
            plan.source_provider_capability_serialization_version,
            provider_matrix.serialization_version,
        )
        self.assertEqual(
            plan.provider_candidate_count,
            provider_matrix.provider_candidate_count,
        )
        self.assertEqual(plan.route_count, provider_matrix.route_count)
        self.assertEqual(plan.candidate_count, 4)
        self.assertEqual(
            plan.candidate_ids,
            (
                "load_balancer::async_slot_distribution",
                "load_balancer::latency_pressure_distribution",
                "load_balancer::retry_capacity_reserve",
                "load_balancer::provider_capacity_visibility",
            ),
        )
        self.assertEqual(plan.balancing_candidate_count, 2)
        self.assertEqual(plan.capacity_guardrail_count, 1)
        self.assertEqual(plan.routing_guardrail_count, 1)
        self.assertGreater(plan.total_advisory_load_units, 0)
        self.assertGreaterEqual(plan.max_advisory_capacity_slots, 2)
        self.assertGreater(plan.highest_advisory_load_score, 0)
        self.assertGreater(plan.total_advisory_load_score, 0)
        self.assertEqual(plan.load_balancing_pressure, "guarded")
        self.assertIn("does not distribute requests", plan.authority_boundary)
        self.assertTrue(plan.load_balancer_planning_implemented)
        self.assertFalse(plan.load_balancing_runtime_implemented)
        self.assertFalse(plan.request_distribution_implemented)
        self.assertFalse(plan.traffic_shaping_implemented)
        self.assertFalse(plan.capacity_enforcement_implemented)
        self.assertFalse(plan.provider_selection_implemented)
        self.assertFalse(plan.automatic_model_selection_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.parallel_execution_implemented)
        self.assertFalse(plan.async_runtime_execution_implemented)
        self.assertFalse(plan.workflow_timing_change_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.graph_compilation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.node_handler_invocation_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.advisory_only)

    def test_candidates_preserve_runtime_and_routing_boundaries(self) -> None:
        plan = plan_load_balancer()
        async_candidate = load_balance_candidate_by_id(
            "load_balancer::async_slot_distribution",
            plan,
        )
        provider_candidate = load_balance_candidate_by_id(
            "load_balancer::provider_capacity_visibility",
            plan,
        )

        self.assertIsNotNone(async_candidate)
        self.assertIsNotNone(provider_candidate)
        assert async_candidate is not None
        assert provider_candidate is not None
        self.assertEqual(async_candidate.status, "balancing_candidate")
        self.assertGreater(async_candidate.advisory_capacity_slots, 0)
        self.assertEqual(provider_candidate.status, "routing_guardrail")
        self.assertEqual(provider_candidate.advisory_capacity_slots, 0)
        self.assertEqual(provider_candidate.load_pressure, "guarded")

        for candidate in plan.candidates:
            self.assertEqual(
                set(candidate.model_dump(mode="json")),
                REQUIRED_LOAD_BALANCE_FIELDS,
            )
            self.assertEqual(
                candidate.serialization_version,
                "load_balance_candidate.v1",
            )
            self.assertIn(
                "load_balancing_runtime",
                candidate.blocked_runtime_behaviors,
            )
            self.assertIn(
                "provider_or_model_routing",
                candidate.blocked_runtime_behaviors,
            )
            self.assertTrue(candidate.load_balancer_planning_implemented)
            self.assertFalse(candidate.load_balancing_runtime_implemented)
            self.assertFalse(candidate.request_distribution_implemented)
            self.assertFalse(candidate.traffic_shaping_implemented)
            self.assertFalse(candidate.capacity_enforcement_implemented)
            self.assertFalse(candidate.provider_selection_implemented)
            self.assertFalse(candidate.automatic_model_selection_implemented)
            self.assertFalse(candidate.provider_model_routing_implemented)
            self.assertFalse(candidate.parallel_execution_implemented)
            self.assertFalse(candidate.async_runtime_execution_implemented)
            self.assertFalse(candidate.workflow_timing_change_implemented)
            self.assertFalse(candidate.workflow_graph_mutation_implemented)
            self.assertFalse(candidate.graph_compilation_implemented)
            self.assertFalse(candidate.workflow_execution_implemented)
            self.assertFalse(candidate.agent_invocation_implemented)
            self.assertFalse(candidate.node_handler_invocation_implemented)
            self.assertFalse(candidate.retry_triggering_implemented)
            self.assertFalse(candidate.prompt_mutation_implemented)
            self.assertFalse(candidate.persistent_storage_write_implemented)
            self.assertFalse(candidate.generated_output_mutation_implemented)
            self.assertTrue(candidate.advisory_only)

    def test_lookup_helpers_are_stable_and_read_only(self) -> None:
        plan = plan_load_balancer()
        balancing = load_balance_candidates_for_status(
            "balancing_candidate",
            plan,
        )
        capacity = load_balance_candidates_for_status("capacity_guardrail", plan)
        routing = load_balance_candidates_for_status("routing_guardrail", plan)
        missing = load_balance_candidate_by_id("missing", plan)

        self.assertIsNone(missing)
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in balancing),
            plan.balancing_candidate_ids,
        )
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in capacity),
            plan.capacity_guardrail_candidate_ids,
        )
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in routing),
            plan.routing_guardrail_candidate_ids,
        )

    def test_plan_rejects_mismatched_candidate_totals(self) -> None:
        plan = plan_load_balancer()
        payload = plan.model_dump(mode="json")
        payload["candidate_ids"] = ("missing",) + tuple(payload["candidate_ids"][1:])

        with self.assertRaisesRegex(ValueError, "candidate_ids must match"):
            LoadBalancerPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["total_advisory_load_units"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "total_advisory_load_units must match",
        ):
            LoadBalancerPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["load_balancing_pressure"] = "low"

        with self.assertRaisesRegex(ValueError, "load_balancing_pressure must match"):
            LoadBalancerPlan(**payload)

    def test_plan_does_not_declare_runtime_balancing_terms(self) -> None:
        plan = plan_load_balancer()
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
            "distribute_request(",
            "shape_traffic(",
            "balance_load(",
            "enforce_capacity(",
            "select_provider(",
            "select_model(",
            "route_provider(",
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
