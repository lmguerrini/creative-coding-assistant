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
    HitlBudgetGatePlan,
    evaluate_budget_policies,
    evaluate_hitl_budget_gate,
    hitl_budget_gate_by_id,
    hitl_budget_gates_for_status,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_HITL_GATE_FIELDS = {
    "gate_id",
    "source_budget_policy_id",
    "source_model_profile_id",
    "route_name",
    "budget_posture",
    "gate_status",
    "operator_review_reason",
    "status",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "hitl_budget_gate_implemented",
    "hitl_request_emitted",
    "human_input_blocking_implemented",
    "budget_enforcement_implemented",
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


class HitlBudgetGateTests(unittest.TestCase):
    def test_default_hitl_budget_gate_uses_budget_policies(self) -> None:
        budget_plan = evaluate_budget_policies(route=RouteName.GENERATE)
        plan = evaluate_hitl_budget_gate(budget_policies=budget_plan)

        self.assertEqual(plan.role, "hitl_budget_gate")
        self.assertEqual(plan.serialization_version, "hitl_budget_gate_plan.v1")
        self.assertEqual(
            plan.source_budget_policy_serialization_version,
            budget_plan.serialization_version,
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(
            plan.recommended_gate_id,
            "hitl_budget_gate::creative_reasoning_model_profile",
        )
        self.assertEqual(plan.recommended_gate_status, "review_recommended")
        self.assertEqual(plan.gate_count, 3)
        self.assertEqual(plan.review_recommended_count, 1)
        self.assertEqual(plan.required_count, 0)
        self.assertIn("does not emit human input requests", plan.authority_boundary)
        self.assertTrue(plan.hitl_budget_gate_implemented)
        self.assertFalse(plan.hitl_request_emitted)
        self.assertFalse(plan.human_input_blocking_implemented)
        self.assertFalse(plan.budget_enforcement_implemented)
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

    def test_hitl_budget_gate_decisions_are_advisory_only(self) -> None:
        plan = evaluate_hitl_budget_gate(route=RouteName.REVIEW)

        for decision in plan.decisions:
            dumped = decision.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_HITL_GATE_FIELDS)
            self.assertEqual(
                decision.serialization_version,
                "hitl_budget_gate_decision.v1",
            )
            self.assertEqual(decision.route_name, RouteName.REVIEW)
            self.assertIn(
                "human_input_request_emission",
                decision.blocked_runtime_behaviors,
            )
            self.assertTrue(decision.hitl_budget_gate_implemented)
            self.assertFalse(decision.hitl_request_emitted)
            self.assertFalse(decision.human_input_blocking_implemented)
            self.assertFalse(decision.budget_enforcement_implemented)
            self.assertFalse(decision.execution_blocking_implemented)
            self.assertFalse(decision.provider_model_routing_implemented)
            self.assertFalse(decision.model_selection_implemented)
            self.assertFalse(decision.provider_execution_implemented)
            self.assertFalse(decision.workflow_control_implemented)
            self.assertFalse(decision.retry_triggering_implemented)
            self.assertFalse(decision.prompt_mutation_implemented)
            self.assertFalse(decision.generated_output_mutation_implemented)
            self.assertTrue(decision.advisory_only)

    def test_lookup_helpers_return_gates_without_emitting_hitl(self) -> None:
        plan = evaluate_hitl_budget_gate(route=RouteName.GENERATE)
        recommended = hitl_budget_gate_by_id(
            "hitl_budget_gate::creative_reasoning_model_profile",
            plan,
        )
        review_gates = hitl_budget_gates_for_status("review_recommended", plan)
        missing = hitl_budget_gate_by_id("missing", plan)

        self.assertIsNone(missing)
        self.assertIsNotNone(recommended)
        assert recommended is not None
        self.assertEqual(recommended.status, "recommended")
        self.assertIn(recommended, review_gates)

    def test_plan_rejects_mismatched_gates_or_recommendation(self) -> None:
        plan = evaluate_hitl_budget_gate(route=RouteName.GENERATE)
        payload = plan.model_dump(mode="json")
        payload["gate_ids"] = ("missing",) + tuple(payload["gate_ids"][1:])

        with self.assertRaisesRegex(ValueError, "gate_ids must match"):
            HitlBudgetGatePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["recommended_gate_id"] = "missing"

        with self.assertRaisesRegex(ValueError, "recommended_gate_id must match"):
            HitlBudgetGatePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["recommended_gate_status"] = "not_required"

        with self.assertRaisesRegex(ValueError, "recommended_gate_status must match"):
            HitlBudgetGatePlan(**payload)

    def test_hitl_budget_gate_does_not_change_request_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Generate a costly-looking fluid sketch.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = evaluate_hitl_budget_gate(route_decision=baseline_decision)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, baseline_decision.route)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_hitl_budget_gate_does_not_declare_runtime_application_terms(self) -> None:
        plan = evaluate_hitl_budget_gate(route=RouteName.REVIEW)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for decision in plan.decisions
                    for field in (
                        decision.gate_id,
                        decision.source_budget_policy_id,
                        decision.source_model_profile_id,
                        decision.operator_review_reason,
                        *decision.evidence,
                        *decision.advisory_actions,
                        *decision.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
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
