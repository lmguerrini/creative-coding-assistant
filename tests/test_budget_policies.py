import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    BudgetPolicyPlan,
    budget_policies_for_posture,
    budget_policy_by_id,
    estimate_routing_cost,
    evaluate_budget_policies,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_BUDGET_POLICY_FIELDS = {
    "policy_id",
    "source_cost_scenario_id",
    "source_model_profile_id",
    "route_name",
    "soft_limit_units",
    "hard_limit_units",
    "estimated_max_cost_units",
    "budget_margin_units",
    "budget_posture",
    "status",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "budget_policy_implemented",
    "budget_enforcement_implemented",
    "hitl_request_implemented",
    "execution_blocking_implemented",
    "provider_model_routing_implemented",
    "model_selection_implemented",
    "provider_execution_implemented",
    "workflow_control_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}


class BudgetPolicyTests(unittest.TestCase):
    def test_default_budget_policies_use_cost_estimates(self) -> None:
        cost_plan = estimate_routing_cost(route=RouteName.GENERATE)
        plan = evaluate_budget_policies(cost_estimation=cost_plan)

        self.assertEqual(plan.role, "budget_policy_evaluator")
        self.assertEqual(plan.serialization_version, "budget_policy_plan.v1")
        self.assertEqual(
            plan.source_cost_estimation_serialization_version,
            cost_plan.serialization_version,
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(
            plan.recommended_policy_id,
            "budget_policy::creative_reasoning_model_profile",
        )
        self.assertEqual(plan.recommended_budget_posture, "review_recommended")
        self.assertEqual(plan.policy_count, 3)
        self.assertEqual(plan.review_recommended_count, 1)
        self.assertEqual(plan.over_budget_count, 0)
        self.assertIn("does not enforce budgets", plan.authority_boundary)
        self.assertTrue(plan.budget_policy_implemented)
        self.assertFalse(plan.budget_enforcement_implemented)
        self.assertFalse(plan.hitl_request_implemented)
        self.assertFalse(plan.execution_blocking_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.model_selection_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.advisory_only)

    def test_budget_policy_decisions_are_advisory_only(self) -> None:
        plan = evaluate_budget_policies(route=RouteName.REVIEW)

        for decision in plan.decisions:
            dumped = decision.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_BUDGET_POLICY_FIELDS)
            self.assertEqual(decision.serialization_version, "budget_policy_decision.v1")
            self.assertEqual(decision.route_name, RouteName.REVIEW)
            self.assertEqual(
                decision.budget_margin_units,
                decision.hard_limit_units - decision.estimated_max_cost_units,
            )
            self.assertIn("budget_enforcement", decision.blocked_runtime_behaviors)
            self.assertTrue(decision.budget_policy_implemented)
            self.assertFalse(decision.budget_enforcement_implemented)
            self.assertFalse(decision.hitl_request_implemented)
            self.assertFalse(decision.execution_blocking_implemented)
            self.assertFalse(decision.provider_model_routing_implemented)
            self.assertFalse(decision.model_selection_implemented)
            self.assertFalse(decision.provider_execution_implemented)
            self.assertFalse(decision.workflow_control_implemented)
            self.assertFalse(decision.retry_triggering_implemented)
            self.assertFalse(decision.prompt_mutation_implemented)
            self.assertFalse(decision.generated_output_mutation_implemented)
            self.assertTrue(decision.advisory_only)

    def test_lookup_helpers_return_policies_without_enforcing_budget(self) -> None:
        plan = evaluate_budget_policies(route=RouteName.GENERATE)
        recommended = budget_policy_by_id(
            "budget_policy::creative_reasoning_model_profile",
            plan,
        )
        review_policies = budget_policies_for_posture("review_recommended", plan)
        missing = budget_policy_by_id("missing", plan)

        self.assertIsNone(missing)
        self.assertIsNotNone(recommended)
        assert recommended is not None
        self.assertEqual(recommended.status, "recommended")
        self.assertIn(recommended, review_policies)

    def test_plan_rejects_mismatched_policies_or_recommendation(self) -> None:
        plan = evaluate_budget_policies(route=RouteName.GENERATE)
        payload = plan.model_dump(mode="json")
        payload["policy_ids"] = ("missing",) + tuple(payload["policy_ids"][1:])

        with self.assertRaisesRegex(ValueError, "policy_ids must match"):
            BudgetPolicyPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["recommended_policy_id"] = "missing"

        with self.assertRaisesRegex(ValueError, "recommended_policy_id must match"):
            BudgetPolicyPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["recommended_budget_posture"] = "within_budget"

        with self.assertRaisesRegex(
            ValueError,
            "recommended_budget_posture must match",
        ):
            BudgetPolicyPlan(**payload)

    def test_budget_policies_do_not_change_request_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Generate an intricate particle sketch.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = evaluate_budget_policies(route_decision=baseline_decision)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, baseline_decision.route)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_budget_policies_do_not_declare_runtime_application_terms(self) -> None:
        plan = evaluate_budget_policies(route=RouteName.REVIEW)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for decision in plan.decisions
                    for field in (
                        decision.policy_id,
                        decision.source_cost_scenario_id,
                        decision.source_model_profile_id,
                        *decision.evidence,
                        *decision.advisory_actions,
                        *decision.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "enforce_budget(",
            "request_hitl(",
            "block_execution(",
            "select_model(",
            "switch_model(",
            "route_provider(",
            "execute_provider(",
            "control_workflow(",
            "trigger_retry(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
