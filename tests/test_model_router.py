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
    ModelRoutingPlan,
    model_profile_registry,
    model_route_candidate_by_id,
    model_route_candidates_for_status,
    provider_selection_registry,
    route_model_request,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_CANDIDATE_FIELDS = {
    "candidate_id",
    "rank",
    "source_model_profile_id",
    "model_profile_kind",
    "profile_name",
    "route_name",
    "route_applicability",
    "provider_candidate_ids",
    "capability_dimensions",
    "fit_score",
    "fit_band",
    "status",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "model_router_implemented",
    "model_route_recommendation_implemented",
    "model_selection_implemented",
    "automatic_model_selection_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "local_cloud_routing_implemented",
    "hybrid_routing_implemented",
    "cost_optimization_implemented",
    "quality_optimization_implemented",
    "budget_enforcement_implemented",
    "workflow_control_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "recommendation_only",
}


class ModelRouterTests(unittest.TestCase):
    def test_default_model_router_recommends_profile_for_generate_route(self) -> None:
        plan = route_model_request(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "model_router")
        self.assertEqual(plan.serialization_version, "model_routing_plan.v1")
        self.assertEqual(
            plan.source_model_profile_serialization_version,
            model_profile_registry().serialization_version,
        )
        self.assertEqual(
            plan.source_provider_selection_serialization_version,
            provider_selection_registry().serialization_version,
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(
            plan.source_model_profile_ids,
            (
                "creative_reasoning_model_profile",
                "code_assistance_model_profile",
                "fast_iteration_model_profile",
            ),
        )
        self.assertEqual(
            plan.recommended_candidate_id,
            "model_route::creative_reasoning_model_profile",
        )
        self.assertEqual(
            plan.recommended_model_profile_id,
            "creative_reasoning_model_profile",
        )
        self.assertEqual(plan.candidate_count, 3)
        self.assertEqual(plan.recommendation_confidence, "high")
        self.assertIn("does not select or switch", plan.authority_boundary)
        self.assertTrue(plan.model_router_implemented)
        self.assertTrue(plan.model_route_recommendation_implemented)
        self.assertFalse(plan.model_selection_implemented)
        self.assertFalse(plan.automatic_model_selection_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.local_cloud_routing_implemented)
        self.assertFalse(plan.hybrid_routing_implemented)
        self.assertFalse(plan.cost_optimization_implemented)
        self.assertFalse(plan.quality_optimization_implemented)
        self.assertFalse(plan.quality_prediction_implemented)
        self.assertFalse(plan.cost_prediction_implemented)
        self.assertFalse(plan.budget_enforcement_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.recommendation_only)

    def test_model_route_candidates_remain_advisory_and_source_aligned(self) -> None:
        plan = route_model_request(route=RouteName.GENERATE)
        known_provider_ids = set(provider_selection_registry().provider_candidate_ids)

        for candidate in plan.candidates:
            dumped = candidate.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_CANDIDATE_FIELDS)
            self.assertEqual(
                candidate.serialization_version, "model_route_candidate.v1"
            )
            self.assertEqual(candidate.route_name, RouteName.GENERATE)
            self.assertIn(RouteName.GENERATE, candidate.route_applicability)
            self.assertTrue(
                set(candidate.provider_candidate_ids).issubset(known_provider_ids)
            )
            self.assertIn(
                "provider_or_model_routing",
                candidate.blocked_runtime_behaviors,
            )
            self.assertTrue(candidate.model_router_implemented)
            self.assertTrue(candidate.model_route_recommendation_implemented)
            self.assertFalse(candidate.model_selection_implemented)
            self.assertFalse(candidate.automatic_model_selection_implemented)
            self.assertFalse(candidate.provider_model_routing_implemented)
            self.assertFalse(candidate.provider_execution_implemented)
            self.assertFalse(candidate.local_cloud_routing_implemented)
            self.assertFalse(candidate.hybrid_routing_implemented)
            self.assertFalse(candidate.cost_optimization_implemented)
            self.assertFalse(candidate.quality_optimization_implemented)
            self.assertFalse(candidate.budget_enforcement_implemented)
            self.assertFalse(candidate.workflow_control_implemented)
            self.assertFalse(candidate.retry_triggering_implemented)
            self.assertFalse(candidate.prompt_mutation_implemented)
            self.assertFalse(candidate.generated_output_mutation_implemented)
            self.assertTrue(candidate.recommendation_only)

    def test_review_route_prioritizes_evaluation_profile_and_lookup_helpers(
        self,
    ) -> None:
        plan = route_model_request(route="review")
        recommended = model_route_candidate_by_id(
            "model_route::evaluation_review_model_profile",
            plan,
        )
        missing = model_route_candidate_by_id("missing", plan)
        fallbacks = model_route_candidates_for_status("fallback", plan)

        self.assertIsNone(missing)
        self.assertIsNotNone(recommended)
        assert recommended is not None
        self.assertEqual(
            plan.recommended_model_profile_id,
            "evaluation_review_model_profile",
        )
        self.assertEqual(recommended.status, "recommended")
        self.assertEqual(recommended.rank, 1)
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in fallbacks),
            plan.fallback_candidate_ids,
        )

    def test_plan_rejects_mismatched_candidates_or_recommendation(self) -> None:
        plan = route_model_request(route=RouteName.GENERATE)
        payload = plan.model_dump(mode="json")
        payload["candidate_ids"] = ("missing",) + tuple(payload["candidate_ids"][1:])

        with self.assertRaisesRegex(ValueError, "candidate_ids must match"):
            ModelRoutingPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["recommended_candidate_id"] = "missing"

        with self.assertRaisesRegex(ValueError, "recommended_candidate_id must match"):
            ModelRoutingPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["recommendation_confidence"] = "low"

        with self.assertRaisesRegex(ValueError, "recommendation_confidence must match"):
            ModelRoutingPlan(**payload)

    def test_model_router_does_not_change_request_routing_or_provider_factory(
        self,
    ) -> None:
        request = AssistantRequest(
            query="Generate a p5.js particle field.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = route_model_request(route_decision=baseline_decision)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, baseline_decision.route)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_model_router_does_not_declare_runtime_application_terms(self) -> None:
        plan = route_model_request(route=RouteName.REVIEW)
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
                        candidate.source_model_profile_id,
                        candidate.model_profile_kind,
                        *candidate.evidence,
                        *candidate.advisory_actions,
                        *candidate.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "select_model(",
            "switch_model(",
            "route_provider(",
            "execute_provider(",
            "call_provider(",
            "route_local_cloud(",
            "route_hybrid(",
            "optimize_cost(",
            "optimize_quality(",
            "enforce_budget(",
            "control_workflow(",
            "trigger_retry(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
