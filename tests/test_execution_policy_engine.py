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
    ExecutionPolicyPlan,
    evaluate_execution_policies,
    execution_policies_for_posture,
    execution_policy_by_id,
    recommend_runtime_execution,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_POLICY_FIELDS = {
    "policy_id",
    "source_runtime_recommendation_id",
    "source_hitl_budget_gate_id",
    "source_model_profile_id",
    "route_name",
    "runtime_posture",
    "gate_status",
    "execution_policy_posture",
    "policy_summary",
    "status",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "execution_policy_engine_implemented",
    "execution_policy_recommendation_implemented",
    "execution_policy_application_implemented",
    "runtime_recommendation_application_implemented",
    "hitl_request_emitted",
    "execution_blocking_implemented",
    "budget_enforcement_implemented",
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


class ExecutionPolicyEngineTests(unittest.TestCase):
    def test_default_execution_policies_use_runtime_recommendations(self) -> None:
        runtime_plan = recommend_runtime_execution(route=RouteName.GENERATE)
        plan = evaluate_execution_policies(runtime_recommendations=runtime_plan)

        self.assertEqual(plan.role, "execution_policy_engine")
        self.assertEqual(plan.serialization_version, "execution_policy_plan.v1")
        self.assertEqual(
            plan.source_runtime_recommendation_serialization_version,
            runtime_plan.serialization_version,
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(
            plan.recommended_policy_id,
            "execution_policy::creative_reasoning_model_profile",
        )
        self.assertEqual(
            plan.recommended_execution_policy_posture,
            "guarded_execution_policy",
        )
        self.assertEqual(plan.recommended_gate_status, "review_recommended")
        self.assertEqual(plan.policy_count, 3)
        self.assertEqual(plan.guarded_policy_count, 1)
        self.assertEqual(plan.manual_review_policy_count, 0)
        self.assertIn("does not apply execution policies", plan.authority_boundary)
        self.assertTrue(plan.execution_policy_engine_implemented)
        self.assertTrue(plan.execution_policy_recommendation_implemented)
        self.assertFalse(plan.execution_policy_application_implemented)
        self.assertFalse(plan.runtime_recommendation_application_implemented)
        self.assertFalse(plan.hitl_request_emitted)
        self.assertFalse(plan.execution_blocking_implemented)
        self.assertFalse(plan.budget_enforcement_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.model_selection_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.advisory_only)

    def test_execution_policy_decisions_are_advisory_only(self) -> None:
        plan = evaluate_execution_policies(route=RouteName.REVIEW)

        for policy in plan.policies:
            dumped = policy.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_EXECUTION_POLICY_FIELDS)
            self.assertEqual(
                policy.serialization_version,
                "execution_policy_decision.v1",
            )
            self.assertEqual(policy.route_name, RouteName.REVIEW)
            self.assertIn(
                "execution_policy_application",
                policy.blocked_runtime_behaviors,
            )
            self.assertTrue(policy.execution_policy_engine_implemented)
            self.assertTrue(policy.execution_policy_recommendation_implemented)
            self.assertFalse(policy.execution_policy_application_implemented)
            self.assertFalse(policy.runtime_recommendation_application_implemented)
            self.assertFalse(policy.hitl_request_emitted)
            self.assertFalse(policy.execution_blocking_implemented)
            self.assertFalse(policy.budget_enforcement_implemented)
            self.assertFalse(policy.provider_model_routing_implemented)
            self.assertFalse(policy.model_selection_implemented)
            self.assertFalse(policy.provider_execution_implemented)
            self.assertFalse(policy.workflow_control_implemented)
            self.assertFalse(policy.retry_triggering_implemented)
            self.assertFalse(policy.prompt_mutation_implemented)
            self.assertFalse(policy.generated_output_mutation_implemented)
            self.assertTrue(policy.advisory_only)

    def test_lookup_helpers_return_policies_without_application(self) -> None:
        plan = evaluate_execution_policies(route=RouteName.GENERATE)
        recommended = execution_policy_by_id(
            "execution_policy::creative_reasoning_model_profile",
            plan,
        )
        guarded = execution_policies_for_posture(
            "guarded_execution_policy",
            plan,
        )
        missing = execution_policy_by_id("missing", plan)

        self.assertIsNone(missing)
        self.assertIsNotNone(recommended)
        assert recommended is not None
        self.assertEqual(recommended.status, "recommended")
        self.assertIn(recommended, guarded)

    def test_plan_rejects_mismatched_policies_or_recommendation(self) -> None:
        plan = evaluate_execution_policies(route=RouteName.GENERATE)
        payload = plan.model_dump(mode="json")
        payload["policy_ids"] = ("missing",) + tuple(payload["policy_ids"][1:])

        with self.assertRaisesRegex(ValueError, "policy_ids must match"):
            ExecutionPolicyPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["recommended_policy_id"] = "missing"

        with self.assertRaisesRegex(ValueError, "recommended_policy_id must match"):
            ExecutionPolicyPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["recommended_execution_policy_posture"] = (
            "standard_execution_policy"
        )

        with self.assertRaisesRegex(
            ValueError,
            "recommended_execution_policy_posture must match",
        ):
            ExecutionPolicyPlan(**payload)

    def test_execution_policies_do_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Explain routing policy for a shader debug task.",
            mode=AssistantMode.DEBUG,
            domains=(CreativeCodingDomain.THREE_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = evaluate_execution_policies(route_decision=baseline_decision)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, baseline_decision.route)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_execution_policies_do_not_declare_application_terms(self) -> None:
        plan = evaluate_execution_policies(route=RouteName.REVIEW)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for policy in plan.policies
                    for field in (
                        policy.policy_id,
                        policy.source_runtime_recommendation_id,
                        policy.source_hitl_budget_gate_id,
                        policy.source_model_profile_id,
                        policy.policy_summary,
                        *policy.evidence,
                        *policy.advisory_actions,
                        *policy.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "apply_execution_policy(",
            "apply_runtime_recommendation(",
            "emit_hitl_request(",
            "request_hitl(",
            "block_execution(",
            "enforce_budget(",
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
