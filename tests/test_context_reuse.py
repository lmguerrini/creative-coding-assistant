import unittest

from creative_coding_assistant.orchestration import (
    ContextReusePlan,
    context_reuse_candidate_by_id,
    context_reuse_candidates_for_status,
    plan_context_budget,
    plan_context_reuse,
)


class ContextReuseTests(unittest.TestCase):
    def test_default_context_budgets_are_reusable_without_materialization(self) -> None:
        plan = plan_context_reuse()

        self.assertEqual(plan.role, "context_reuse_planner")
        self.assertEqual(plan.serialization_version, "context_reuse_plan.v1")
        self.assertGreater(plan.total_reusable_tokens, 0)
        self.assertEqual(plan.reuse_confidence, "high")
        self.assertIn("does not materialize shared context", plan.authority_boundary)
        self.assertTrue(plan.context_reuse_implemented)
        self.assertFalse(plan.shared_context_materialization_implemented)
        self.assertFalse(plan.source_context_mutation_implemented)
        self.assertFalse(plan.cache_write_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.planning_only)

        for candidate in plan.candidates:
            self.assertTrue(candidate.context_reuse_implemented)
            self.assertFalse(candidate.shared_context_materialization_implemented)
            self.assertFalse(candidate.source_context_mutation_implemented)
            self.assertFalse(candidate.cache_write_implemented)
            self.assertFalse(candidate.provider_model_routing_implemented)
            self.assertFalse(candidate.workflow_control_implemented)
            self.assertFalse(candidate.generated_output_mutation_implemented)
            self.assertTrue(candidate.planning_only)

    def test_mismatched_context_sources_are_not_reusable(self) -> None:
        previous = plan_context_budget(user_query="Generate blue particles.")
        current = plan_context_budget(user_query="Explain shaders.")
        plan = plan_context_reuse(
            previous_context_budget=previous,
            current_context_budget=current,
        )
        user = context_reuse_candidate_by_id("context_reuse::user_request", plan)

        self.assertIsNotNone(user)
        assert user is not None
        self.assertEqual(user.status, "reusable")
        self.assertGreater(user.reusable_tokens, 0)
        self.assertLessEqual(user.reusable_tokens, user.requested_tokens)

    def test_lookup_helpers_are_stable_and_read_only(self) -> None:
        plan = plan_context_reuse()
        reusable = context_reuse_candidates_for_status("reusable", plan)
        missing = context_reuse_candidate_by_id("missing", plan)

        self.assertIsNone(missing)
        self.assertGreater(len(reusable), 0)
        self.assertIs(
            reusable[0],
            context_reuse_candidate_by_id(reusable[0].candidate_id, plan),
        )

    def test_plan_rejects_mismatched_candidates_or_totals(self) -> None:
        plan = plan_context_reuse()
        payload = plan.model_dump(mode="json")
        payload["candidate_ids"] = ("missing",) + tuple(payload["candidate_ids"][1:])

        with self.assertRaisesRegex(ValueError, "candidate_ids must match"):
            ContextReusePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["total_reusable_tokens"] += 1

        with self.assertRaisesRegex(ValueError, "total_reusable_tokens must match"):
            ContextReusePlan(**payload)

    def test_plan_does_not_declare_materialization_or_routing_terms(self) -> None:
        plan = plan_context_reuse()
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
                        candidate.previous_allocation_id,
                        candidate.current_allocation_id,
                        *candidate.evidence,
                        *candidate.advisory_actions,
                        *candidate.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "materialize_shared_context(",
            "mutate_source_context(",
            "write_cache(",
            "write_storage(",
            "route_provider(",
            "control_workflow(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
