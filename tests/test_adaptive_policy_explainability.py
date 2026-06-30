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
    AdaptivePolicyExplainabilityPlan,
    adaptive_policy_explanation_by_id,
    adaptive_policy_explanations_for_status,
    explain_adaptive_policy,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_PROVIDER_IDS = ("openai", "anthropic", "gemini", "local")
REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
REQUIRED_HYBRID_DIRECTIONS = (
    "local_to_cloud",
    "cloud_to_local",
    "cloud_to_cloud",
    "local_to_local",
)
REQUIRED_EXPLANATION_FIELDS = {
    "explanation_id",
    "explanation_rank",
    "source_surface",
    "status",
    "route_name",
    "task_type",
    "source_serialization_version",
    "source_primary_record_id",
    "source_record_count",
    "source_policy_posture",
    "source_hitl_required",
    "source_guardrail_count",
    "source_signal_score",
    "evidence_weight",
    "hitl_explanation_weight",
    "guardrail_penalty",
    "policy_explainability_score",
    "explanation_summary",
    "referenced_advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "adaptive_policy_explainability_implemented",
    "adaptive_policy_explanation_metadata_implemented",
    "execution_policy_metadata_used",
    "dynamic_strategy_metadata_used",
    "escalation_metadata_used",
    "routing_explainability_metadata_used",
    "policy_application_implemented",
    "execution_policy_application_implemented",
    "strategy_application_implemented",
    "routing_application_implemented",
    "escalation_triggering_implemented",
    "hitl_request_emitted",
    "execution_blocking_implemented",
    "budget_enforcement_implemented",
    "provider_model_routing_implemented",
    "model_selection_implemented",
    "provider_execution_implemented",
    "agent_invocation_implemented",
    "workflow_control_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_execution_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "advisory_only",
}


class AdaptivePolicyExplainabilityTests(unittest.TestCase):
    def test_plan_combines_adaptive_policy_sources(self) -> None:
        plan = explain_adaptive_policy(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "adaptive_policy_explainability")
        self.assertEqual(
            plan.serialization_version,
            "adaptive_policy_explainability_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.source_execution_policy_serialization_version,
            "execution_policy_plan.v1",
        )
        self.assertEqual(
            plan.source_dynamic_strategy_serialization_version,
            "adaptive_execution_strategy_selection_plan.v1",
        )
        self.assertEqual(
            plan.source_escalation_optimization_serialization_version,
            "adaptive_escalation_optimization_plan.v1",
        )
        self.assertEqual(
            plan.source_routing_explainability_serialization_version,
            "routing_explainability_plan.v1",
        )
        self.assertEqual(plan.provider_ids, REQUIRED_PROVIDER_IDS)
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.hybrid_policy_directions, REQUIRED_HYBRID_DIRECTIONS)
        self.assertEqual(plan.explanation_count, 4)
        self.assertEqual(plan.source_surface_count, 4)
        self.assertEqual(
            plan.primary_explanation_id,
            "adaptive_policy::dynamic_execution_strategy",
        )
        self.assertEqual(plan.supporting_explanation_count, 2)
        self.assertEqual(plan.guardrail_explanation_count, 1)
        self.assertEqual(plan.hitl_required_explanation_count, 3)
        self.assertFalse(plan.applied_policy_explanation_ids)
        self.assertEqual(plan.highest_policy_explainability_score, 541)
        self.assertEqual(plan.policy_explainability_pressure, "guarded")
        self.assertIn("does not apply policies", plan.authority_boundary)
        self.assertTrue(plan.adaptive_policy_explainability_implemented)
        self.assertTrue(plan.adaptive_policy_explanation_metadata_implemented)
        self.assertTrue(plan.execution_policy_metadata_used)
        self.assertTrue(plan.dynamic_strategy_metadata_used)
        self.assertTrue(plan.escalation_metadata_used)
        self.assertTrue(plan.routing_explainability_metadata_used)
        self.assertFalse(plan.policy_application_implemented)
        self.assertFalse(plan.execution_policy_application_implemented)
        self.assertFalse(plan.strategy_application_implemented)
        self.assertFalse(plan.routing_application_implemented)
        self.assertFalse(plan.escalation_triggering_implemented)
        self.assertFalse(plan.hitl_request_emitted)
        self.assertFalse(plan.execution_blocking_implemented)
        self.assertFalse(plan.budget_enforcement_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.model_selection_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_explanations_score_policy_metadata_without_application(self) -> None:
        plan = explain_adaptive_policy(route="generate")

        for explanation in plan.explanations:
            dumped = explanation.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_EXPLANATION_FIELDS)
            self.assertEqual(
                explanation.serialization_version,
                "adaptive_policy_explanation_record.v1",
            )
            self.assertEqual(explanation.route_name, RouteName.GENERATE)
            self.assertEqual(
                explanation.explanation_id,
                f"adaptive_policy::{explanation.source_surface}",
            )
            if explanation.status == "primary":
                self.assertEqual(explanation.explanation_rank, 1)
            else:
                self.assertGreater(explanation.explanation_rank, 1)
            self.assertEqual(
                explanation.evidence_weight,
                min(240, explanation.source_record_count * 10),
            )
            self.assertEqual(
                explanation.hitl_explanation_weight,
                70 if explanation.source_hitl_required else 0,
            )
            expected_penalty = min(80, explanation.source_guardrail_count * 10)
            if explanation.status == "guardrail":
                expected_penalty += 90
            elif explanation.source_hitl_required:
                expected_penalty += 20
            self.assertEqual(explanation.guardrail_penalty, min(180, expected_penalty))
            self.assertEqual(
                explanation.policy_explainability_score,
                min(
                    600,
                    max(
                        0,
                        explanation.source_signal_score
                        + explanation.evidence_weight
                        + explanation.hitl_explanation_weight
                        - explanation.guardrail_penalty,
                    ),
                ),
            )
            self.assertIn("policy_application", explanation.blocked_runtime_behaviors)
            self.assertTrue(explanation.adaptive_policy_explainability_implemented)
            self.assertTrue(explanation.execution_policy_metadata_used)
            self.assertTrue(explanation.dynamic_strategy_metadata_used)
            self.assertTrue(explanation.escalation_metadata_used)
            self.assertTrue(explanation.routing_explainability_metadata_used)
            self.assertFalse(explanation.policy_application_implemented)
            self.assertFalse(explanation.execution_policy_application_implemented)
            self.assertFalse(explanation.strategy_application_implemented)
            self.assertFalse(explanation.routing_application_implemented)
            self.assertFalse(explanation.escalation_triggering_implemented)
            self.assertFalse(explanation.hitl_request_emitted)
            self.assertFalse(explanation.execution_blocking_implemented)
            self.assertFalse(explanation.budget_enforcement_implemented)
            self.assertFalse(explanation.provider_model_routing_implemented)
            self.assertFalse(explanation.model_selection_implemented)
            self.assertFalse(explanation.provider_execution_implemented)
            self.assertFalse(explanation.agent_invocation_implemented)
            self.assertFalse(explanation.workflow_control_implemented)
            self.assertFalse(explanation.workflow_graph_mutation_implemented)
            self.assertFalse(explanation.workflow_execution_implemented)
            self.assertFalse(explanation.retry_triggering_implemented)
            self.assertFalse(explanation.prompt_mutation_implemented)
            self.assertFalse(explanation.generated_output_mutation_implemented)
            self.assertTrue(explanation.advisory_only)

        dynamic = adaptive_policy_explanation_by_id(
            "adaptive_policy::dynamic_execution_strategy",
            plan,
        )
        supporting = adaptive_policy_explanations_for_status("supporting", plan)
        guardrail = adaptive_policy_explanations_for_status("guardrail", plan)
        self.assertIsNotNone(dynamic)
        assert dynamic is not None
        self.assertEqual(dynamic.status, "primary")
        self.assertEqual(dynamic.source_policy_posture, "balanced_hybrid_strategy")
        self.assertEqual(len(supporting), 2)
        self.assertEqual(len(guardrail), 1)
        self.assertTrue(all(item.source_hitl_required for item in guardrail))

    def test_plan_rejects_mismatched_policy_explainability_metadata(self) -> None:
        plan = explain_adaptive_policy()
        payload = plan.model_dump(mode="json")
        payload["explanation_ids"] = ("missing",) + tuple(
            payload["explanation_ids"][1:]
        )

        with self.assertRaisesRegex(ValueError, "explanation_ids must match"):
            AdaptivePolicyExplainabilityPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["primary_explanation_id"] = "missing"

        with self.assertRaisesRegex(
            ValueError,
            "primary_explanation_id must match",
        ):
            AdaptivePolicyExplainabilityPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["applied_policy_explanation_ids"] = (plan.explanation_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "applied_policy_explanation_ids must remain empty",
        ):
            AdaptivePolicyExplainabilityPlan(**payload)

    def test_explainability_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Explain adaptive policy for a shader workflow.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.THREE_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = explain_adaptive_policy(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_explainability_does_not_declare_policy_application_terms(self) -> None:
        plan = explain_adaptive_policy(route=RouteName.GENERATE)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for explanation in plan.explanations
                    for field in (
                        explanation.explanation_id,
                        explanation.source_surface,
                        explanation.source_primary_record_id,
                        explanation.source_policy_posture,
                        explanation.explanation_summary,
                        *explanation.referenced_advisory_actions,
                        *explanation.evidence,
                        *explanation.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "apply_policy(",
            "apply_execution_policy(",
            "apply_strategy(",
            "apply_routing(",
            "trigger_escalation(",
            "emit_hitl_request(",
            "block_execution(",
            "enforce_budget(",
            "select_model(",
            "switch_model(",
            "route_provider(",
            "execute_provider(",
            "invoke_agent(",
            "control_workflow(",
            "mutate_workflow_graph(",
            "execute_workflow(",
            "trigger_retry(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
