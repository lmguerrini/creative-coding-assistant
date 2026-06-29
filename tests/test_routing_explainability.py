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
    RoutingExplainabilityPlan,
    explain_routing_decision,
    route_request,
    routing_explanation_by_id,
    routing_explanations_for_source,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_ROUTING_EXPLANATION_FIELDS = {
    "explanation_id",
    "explanation_rank",
    "source_surface",
    "source_record_id",
    "route_name",
    "explanation_summary",
    "evidence",
    "referenced_advisory_actions",
    "blocked_runtime_behaviors",
    "status",
    "routing_explainability_implemented",
    "advisory_explanation_generation_implemented",
    "routing_application_implemented",
    "provider_model_routing_implemented",
    "model_selection_implemented",
    "configured_model_switching_implemented",
    "provider_execution_implemented",
    "hitl_request_emitted",
    "budget_enforcement_implemented",
    "workflow_control_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}


class RoutingExplainabilityTests(unittest.TestCase):
    def test_explainability_summarizes_advisory_routing_sources(self) -> None:
        plan = explain_routing_decision(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "routing_explainability")
        self.assertEqual(
            plan.serialization_version,
            "routing_explainability_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(
            plan.source_model_routing_serialization_version,
            "model_routing_plan.v1",
        )
        self.assertEqual(
            plan.source_model_recommendation_serialization_version,
            "model_recommendation_plan.v1",
        )
        self.assertEqual(plan.explanation_count, 6)
        self.assertEqual(plan.source_surface_count, 6)
        self.assertEqual(
            plan.source_surfaces,
            (
                "model_recommendation",
                "model_routing",
                "local_cloud_routing",
                "hybrid_routing",
                "quality_prediction",
                "cost_prediction",
            ),
        )
        self.assertEqual(
            plan.primary_explanation_id,
            "routing_explanation::model_recommendation",
        )
        self.assertEqual(
            plan.recommended_model_profile_id,
            "creative_reasoning_model_profile",
        )
        self.assertTrue(plan.route_consistency_confirmed)
        self.assertIn("does not select or switch", plan.authority_boundary)
        self.assertTrue(plan.routing_explainability_implemented)
        self.assertTrue(plan.advisory_explanation_generation_implemented)
        self.assertFalse(plan.routing_application_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.model_selection_implemented)
        self.assertFalse(plan.configured_model_switching_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.hitl_request_emitted)
        self.assertFalse(plan.budget_enforcement_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.advisory_only)

    def test_routing_explanation_records_are_advisory_only(self) -> None:
        plan = explain_routing_decision(route=RouteName.REVIEW)

        for explanation in plan.explanations:
            dumped = explanation.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_ROUTING_EXPLANATION_FIELDS)
            self.assertEqual(
                explanation.serialization_version,
                "routing_explanation_record.v1",
            )
            self.assertEqual(explanation.route_name, RouteName.REVIEW)
            self.assertIn(
                "provider_or_model_routing",
                explanation.blocked_runtime_behaviors,
            )
            self.assertTrue(explanation.routing_explainability_implemented)
            self.assertTrue(explanation.advisory_explanation_generation_implemented)
            self.assertFalse(explanation.routing_application_implemented)
            self.assertFalse(explanation.provider_model_routing_implemented)
            self.assertFalse(explanation.model_selection_implemented)
            self.assertFalse(explanation.configured_model_switching_implemented)
            self.assertFalse(explanation.provider_execution_implemented)
            self.assertFalse(explanation.hitl_request_emitted)
            self.assertFalse(explanation.budget_enforcement_implemented)
            self.assertFalse(explanation.workflow_control_implemented)
            self.assertFalse(explanation.retry_triggering_implemented)
            self.assertFalse(explanation.prompt_mutation_implemented)
            self.assertFalse(explanation.generated_output_mutation_implemented)
            self.assertTrue(explanation.advisory_only)

    def test_lookup_helpers_return_explanations_without_application(self) -> None:
        plan = explain_routing_decision(route=RouteName.GENERATE)
        primary = routing_explanation_by_id(
            "routing_explanation::model_recommendation",
            plan,
        )
        quality = routing_explanations_for_source("quality_prediction", plan)
        missing = routing_explanation_by_id("missing", plan)

        self.assertIsNone(missing)
        self.assertIsNotNone(primary)
        assert primary is not None
        self.assertEqual(primary.status, "primary")
        self.assertEqual(primary.source_surface, "model_recommendation")
        self.assertEqual(len(quality), 1)
        self.assertEqual(quality[0].source_surface, "quality_prediction")

    def test_plan_rejects_mismatched_explanations_or_primary(self) -> None:
        plan = explain_routing_decision(route=RouteName.GENERATE)
        payload = plan.model_dump(mode="json")
        payload["explanation_ids"] = ("missing",) + tuple(
            payload["explanation_ids"][1:]
        )

        with self.assertRaisesRegex(ValueError, "explanation_ids must match"):
            RoutingExplainabilityPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["source_surfaces"] = ("cost_prediction",) + tuple(
            payload["source_surfaces"][1:]
        )

        with self.assertRaisesRegex(ValueError, "source_surfaces must match"):
            RoutingExplainabilityPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["primary_explanation_id"] = "missing"

        with self.assertRaisesRegex(
            ValueError,
            "primary_explanation_id must match",
        ):
            RoutingExplainabilityPlan(**payload)

    def test_routing_explainability_does_not_change_routing_or_provider(self) -> None:
        request = AssistantRequest(
            query="Explain the routing choice for a p5.js generation.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = explain_routing_decision(route_decision=baseline_decision)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, baseline_decision.route)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_routing_explainability_does_not_declare_application_terms(self) -> None:
        plan = explain_routing_decision(route=RouteName.REVIEW)
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
                        explanation.source_record_id,
                        explanation.explanation_summary,
                        *explanation.evidence,
                        *explanation.referenced_advisory_actions,
                        *explanation.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "select_model(",
            "switch_model(",
            "route_provider(",
            "apply_routing(",
            "emit_hitl(",
            "enforce_budget(",
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
