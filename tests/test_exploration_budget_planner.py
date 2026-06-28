import unittest

from creative_coding_assistant.orchestration import (
    ExplorationBudgetPlan,
    analyze_creative_complexity,
    analyze_workflow_cost,
    exploration_budget_allocation_by_id,
    exploration_budget_allocations_for_topic,
    plan_context_budget,
    plan_exploration_budget,
)

REQUIRED_EXPLORATION_ALLOCATION_FIELDS = {
    "allocation_id",
    "topic_id",
    "source_budget_profile_id",
    "budget_posture",
    "priority",
    "requested_variants",
    "planned_variants",
    "max_advisory_variants",
    "requested_refinement_passes",
    "planned_refinement_passes",
    "max_advisory_refinement_passes",
    "pressure",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "budget_enforcement_implemented",
    "variant_generation_implemented",
    "refinement_triggering_implemented",
    "cost_routing_implemented",
    "context_routing_implemented",
    "provider_model_routing_implemented",
    "agent_invocation_implemented",
    "workflow_control_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "planning_only",
}


class ExplorationBudgetPlannerTests(unittest.TestCase):
    def test_default_plan_preserves_registry_exploration_capacity(self) -> None:
        plan = plan_exploration_budget()

        self.assertEqual(plan.role, "exploration_budget_planner")
        self.assertEqual(plan.serialization_version, "exploration_budget_plan.v1")
        self.assertEqual(
            plan.source_registry_serialization_version,
            "creative_exploration_budget_registry.v1",
        )
        self.assertEqual(
            plan.allocation_ids,
            (
                "exploration::planning_execution_fit",
                "exploration::style_aesthetic_alignment",
                "exploration::curation_refinement_need",
                "exploration::final_synthesis_readiness",
            ),
        )
        self.assertEqual(plan.total_requested_variants, 6)
        self.assertEqual(plan.total_planned_variants, 6)
        self.assertEqual(plan.total_requested_refinement_passes, 5)
        self.assertEqual(plan.total_planned_refinement_passes, 5)
        self.assertEqual(plan.exploration_pressure, "low")
        self.assertFalse(plan.budget_limited)
        self.assertIn("does not enforce budgets", plan.authority_boundary)
        self.assertTrue(plan.exploration_budget_planning_implemented)
        self.assertFalse(plan.budget_enforcement_implemented)
        self.assertFalse(plan.variant_generation_implemented)
        self.assertFalse(plan.refinement_triggering_implemented)
        self.assertFalse(plan.cost_routing_implemented)
        self.assertFalse(plan.context_routing_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.planning_only)

    def test_allocations_cover_topics_and_boundary_flags(self) -> None:
        plan = plan_exploration_budget(
            creative_complexity=analyze_creative_complexity(),
            workflow_cost=analyze_workflow_cost(),
            context_budget=plan_context_budget(
                total_budget_tokens=500,
                response_reserve_tokens=100,
            ),
        )

        self.assertEqual(plan.creative_complexity_level, "low")
        self.assertEqual(plan.workflow_cost_pressure, "medium")
        self.assertEqual(plan.context_budget_pressure, "high")
        self.assertGreater(plan.context_over_budget_tokens, 0)
        self.assertEqual(plan.exploration_pressure, "high")
        self.assertTrue(plan.budget_limited)
        self.assertLess(plan.total_planned_variants, plan.total_requested_variants)
        self.assertLess(
            plan.total_planned_refinement_passes,
            plan.total_requested_refinement_passes,
        )
        self.assertIn(
            "Flag reduced exploration capacity for later strategy selection.",
            plan.advisory_actions,
        )

        for allocation in plan.allocations:
            self.assertEqual(
                set(allocation.model_dump(mode="json")),
                REQUIRED_EXPLORATION_ALLOCATION_FIELDS,
            )
            self.assertEqual(
                allocation.serialization_version,
                "exploration_budget_allocation.v1",
            )
            self.assertLessEqual(
                allocation.planned_variants,
                allocation.requested_variants,
            )
            self.assertLessEqual(
                allocation.planned_refinement_passes,
                allocation.requested_refinement_passes,
            )
            self.assertIn("budget_enforcement", allocation.blocked_runtime_behaviors)
            self.assertFalse(allocation.budget_enforcement_implemented)
            self.assertFalse(allocation.variant_generation_implemented)
            self.assertFalse(allocation.refinement_triggering_implemented)
            self.assertFalse(allocation.cost_routing_implemented)
            self.assertFalse(allocation.context_routing_implemented)
            self.assertFalse(allocation.provider_model_routing_implemented)
            self.assertFalse(allocation.agent_invocation_implemented)
            self.assertFalse(allocation.workflow_control_implemented)
            self.assertFalse(allocation.retry_triggering_implemented)
            self.assertFalse(allocation.prompt_mutation_implemented)
            self.assertFalse(allocation.generated_output_mutation_implemented)
            self.assertTrue(allocation.planning_only)

        self.assertTrue(
            any(allocation.pressure == "high" for allocation in plan.allocations)
        )

    def test_lookup_helpers_are_stable_and_read_only(self) -> None:
        plan = plan_exploration_budget()
        style = exploration_budget_allocation_by_id(
            "exploration::style_aesthetic_alignment",
            plan,
        )
        style_topic = exploration_budget_allocations_for_topic(
            "style_aesthetic_alignment",
            plan,
        )
        missing = exploration_budget_allocation_by_id("missing", plan)

        self.assertIsNone(missing)
        self.assertIsNotNone(style)
        assert style is not None
        self.assertEqual(style.topic_id, "style_aesthetic_alignment")
        self.assertEqual(len(style_topic), 1)
        self.assertIs(style, style_topic[0])
        self.assertEqual(style.budget_posture, "broad")

    def test_plan_rejects_mismatched_allocations_or_totals(self) -> None:
        plan = plan_exploration_budget()
        payload = plan.model_dump(mode="json")
        payload["allocation_ids"] = ("missing",) + tuple(payload["allocation_ids"][1:])

        with self.assertRaisesRegex(ValueError, "allocation_ids must match"):
            ExplorationBudgetPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["total_planned_variants"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "total_planned_variants must match",
        ):
            ExplorationBudgetPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["max_total_refinement_passes"] = (
            plan.total_planned_refinement_passes - 1
        )

        with self.assertRaisesRegex(
            ValueError,
            "total_planned_refinement_passes must fit",
        ):
            ExplorationBudgetPlan(**payload)

    def test_plan_does_not_declare_active_exploration_terms(self) -> None:
        plan = plan_exploration_budget(
            workflow_cost=analyze_workflow_cost(),
            context_budget=plan_context_budget(
                total_budget_tokens=500,
                response_reserve_tokens=100,
            ),
        )
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for allocation in plan.allocations
                    for field in (
                        allocation.allocation_id,
                        allocation.topic_id,
                        allocation.source_budget_profile_id,
                        *allocation.evidence,
                        *allocation.advisory_actions,
                        *allocation.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "enforce_budget(",
            "generate_variant(",
            "trigger_refinement(",
            "route_by_cost(",
            "route_context(",
            "select_provider(",
            "invoke_agent(",
            "control_workflow(",
            "mutate_prompt(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
