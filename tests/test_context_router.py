import unittest

from creative_coding_assistant.orchestration import (
    ContextRoutingPlan,
    context_route_decision_by_id,
    context_route_decisions_for_lane,
    plan_context_budget,
    route_context_sources,
)

REQUIRED_CONTEXT_ROUTE_DECISION_FIELDS = {
    "decision_id",
    "source_allocation_id",
    "source_kind",
    "source_id",
    "lane",
    "disposition",
    "priority",
    "requested_tokens",
    "routed_tokens",
    "deferred_tokens",
    "pressure",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "context_routing_implemented",
    "context_trimming_implemented",
    "prompt_compression_implemented",
    "retrieval_compression_implemented",
    "memory_summarization_implemented",
    "provider_model_routing_implemented",
    "source_content_mutation_implemented",
    "prompt_mutation_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "routing_only",
}


class ContextRouterTests(unittest.TestCase):
    def test_default_route_plan_maps_budget_sources_to_lanes(self) -> None:
        budget = plan_context_budget()
        plan = route_context_sources(context_budget=budget)

        self.assertEqual(plan.role, "context_router")
        self.assertEqual(plan.serialization_version, "context_routing_plan.v1")
        self.assertEqual(
            plan.source_context_budget_serialization_version,
            "context_budget_plan.v1",
        )
        self.assertEqual(
            plan.decision_ids,
            (
                "route::user_request",
                "route::memory_recent_turns",
                "route::memory_summary",
                "route::project_memory",
                "route::retrieval_chunks",
                "route::creative_metadata",
                "route::workflow_overhead",
                "route::response_reserve",
            ),
        )
        self.assertEqual(
            plan.routed_lanes,
            (
                "primary_prompt",
                "conversation_memory",
                "project_memory",
                "retrieval_context",
                "planning_metadata",
                "response_budget",
            ),
        )
        self.assertEqual(plan.requested_context_tokens, budget.requested_context_tokens)
        self.assertEqual(plan.routed_context_tokens, budget.allocated_context_tokens)
        self.assertEqual(plan.deferred_context_tokens, 0)
        self.assertEqual(plan.response_reserved_tokens, budget.response_reserve_tokens)
        self.assertEqual(plan.context_budget_pressure, "low")
        self.assertEqual(plan.routing_pressure, "low")
        self.assertFalse(plan.has_deferred_context)
        self.assertIn("does not trim context", plan.authority_boundary)
        self.assertTrue(plan.context_routing_implemented)
        self.assertFalse(plan.context_trimming_implemented)
        self.assertFalse(plan.prompt_compression_implemented)
        self.assertFalse(plan.retrieval_compression_implemented)
        self.assertFalse(plan.memory_summarization_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.source_content_mutation_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.routing_only)

    def test_route_decisions_preserve_boundaries_and_defer_over_budget_tokens(
        self,
    ) -> None:
        budget = plan_context_budget(
            total_budget_tokens=500,
            response_reserve_tokens=100,
        )
        plan = route_context_sources(context_budget=budget)

        self.assertEqual(plan.context_budget_pressure, "high")
        self.assertEqual(plan.routing_pressure, "high")
        self.assertGreater(plan.over_budget_tokens, 0)
        self.assertGreater(plan.deferred_context_tokens, 0)
        self.assertTrue(plan.has_deferred_context)
        self.assertIn(
            "Flag deferred context for later scoped optimization tasks.",
            plan.advisory_actions,
        )

        for decision in plan.decisions:
            self.assertEqual(
                set(decision.model_dump(mode="json")),
                REQUIRED_CONTEXT_ROUTE_DECISION_FIELDS,
            )
            self.assertEqual(
                decision.serialization_version,
                "context_route_decision.v1",
            )
            self.assertLessEqual(decision.routed_tokens, decision.requested_tokens)
            self.assertEqual(
                decision.deferred_tokens,
                decision.requested_tokens - decision.routed_tokens,
            )
            self.assertIn("context_trimming", decision.blocked_runtime_behaviors)
            self.assertTrue(decision.context_routing_implemented)
            self.assertFalse(decision.context_trimming_implemented)
            self.assertFalse(decision.prompt_compression_implemented)
            self.assertFalse(decision.retrieval_compression_implemented)
            self.assertFalse(decision.memory_summarization_implemented)
            self.assertFalse(decision.provider_model_routing_implemented)
            self.assertFalse(decision.source_content_mutation_implemented)
            self.assertFalse(decision.prompt_mutation_implemented)
            self.assertFalse(decision.generated_output_mutation_implemented)
            self.assertTrue(decision.routing_only)

        workflow = context_route_decision_by_id("route::workflow_overhead", plan)
        response = context_route_decision_by_id("route::response_reserve", plan)
        self.assertIsNotNone(workflow)
        self.assertIsNotNone(response)
        assert workflow is not None
        assert response is not None
        self.assertEqual(workflow.lane, "planning_metadata")
        self.assertEqual(workflow.pressure, "high")
        self.assertGreater(workflow.deferred_tokens, 0)
        self.assertEqual(response.disposition, "reserve")
        self.assertEqual(response.lane, "response_budget")

    def test_lookup_helpers_are_stable_and_read_only(self) -> None:
        plan = route_context_sources()
        memory_routes = context_route_decisions_for_lane("conversation_memory", plan)
        missing = context_route_decision_by_id("missing", plan)

        self.assertIsNone(missing)
        self.assertEqual(len(memory_routes), 2)
        self.assertEqual(memory_routes[0].source_kind, "memory_recent_turns")
        self.assertEqual(memory_routes[1].source_kind, "memory_summary")
        self.assertIs(
            memory_routes[0],
            context_route_decision_by_id("route::memory_recent_turns", plan),
        )

    def test_plan_rejects_mismatched_decisions_or_totals(self) -> None:
        plan = route_context_sources()
        payload = plan.model_dump(mode="json")
        payload["decision_ids"] = ("missing",) + tuple(payload["decision_ids"][1:])

        with self.assertRaisesRegex(ValueError, "decision_ids must match"):
            ContextRoutingPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["deferred_context_tokens"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "deferred_context_tokens must match",
        ):
            ContextRoutingPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["routed_lanes"] = ("response_budget",)

        with self.assertRaisesRegex(ValueError, "routed_lanes must match"):
            ContextRoutingPlan(**payload)

    def test_plan_does_not_declare_content_or_provider_mutation_terms(self) -> None:
        plan = route_context_sources(
            context_budget=plan_context_budget(
                total_budget_tokens=500,
                response_reserve_tokens=100,
            )
        )
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for decision in plan.decisions
                    for field in (
                        decision.decision_id,
                        decision.source_allocation_id,
                        decision.source_kind,
                        decision.source_id,
                        decision.lane,
                        decision.disposition,
                        *decision.evidence,
                        *decision.advisory_actions,
                        *decision.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "trim_context(",
            "compress_prompt(",
            "compress_retrieval(",
            "summarize_memory(",
            "select_provider(",
            "route_provider(",
            "mutate_prompt(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
