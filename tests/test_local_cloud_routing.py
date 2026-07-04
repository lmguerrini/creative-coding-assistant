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
    LocalCloudRoutingPlan,
    cloud_model_registry,
    local_cloud_route_decision_by_id,
    local_cloud_route_decisions_for_lane,
    local_model_registry,
    route_local_vs_cloud,
    route_model_request,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_LOCAL_CLOUD_DECISION_FIELDS = {
    "decision_id",
    "source_model_route_candidate_id",
    "source_model_profile_id",
    "route_name",
    "local_surface_ids",
    "cloud_surface_ids",
    "provider_candidate_ids",
    "local_score",
    "cloud_score",
    "comparison_delta",
    "routing_lane",
    "routing_posture",
    "decision_status",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "local_cloud_routing_implemented",
    "local_model_discovery_implemented",
    "local_provider_execution_implemented",
    "cloud_provider_execution_implemented",
    "provider_model_routing_implemented",
    "model_selection_implemented",
    "model_switching_implemented",
    "hybrid_routing_implemented",
    "cost_optimization_implemented",
    "quality_optimization_implemented",
    "budget_enforcement_implemented",
    "workflow_control_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}


class LocalCloudRoutingTests(unittest.TestCase):
    def test_default_local_cloud_plan_compares_generate_route_metadata(self) -> None:
        model_plan = route_model_request(route=RouteName.GENERATE)
        plan = route_local_vs_cloud(model_routing=model_plan)

        self.assertEqual(plan.role, "local_cloud_router")
        self.assertEqual(plan.serialization_version, "local_cloud_routing_plan.v1")
        self.assertEqual(
            plan.source_model_routing_serialization_version,
            model_plan.serialization_version,
        )
        self.assertEqual(
            plan.source_local_model_serialization_version,
            local_model_registry().serialization_version,
        )
        self.assertEqual(
            plan.source_cloud_model_serialization_version,
            cloud_model_registry().serialization_version,
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(
            plan.recommended_decision_id,
            "local_cloud_route::creative_reasoning_model_profile",
        )
        self.assertEqual(plan.recommended_routing_lane, "balanced_candidate")
        self.assertEqual(plan.recommended_routing_posture, "balanced")
        self.assertEqual(plan.decision_count, 3)
        self.assertEqual(plan.local_candidate_count, 1)
        self.assertEqual(plan.cloud_candidate_count, 0)
        self.assertEqual(plan.balanced_candidate_count, 2)
        self.assertEqual(plan.routing_confidence, "low")
        self.assertIn(
            "does not discover installed local models", plan.authority_boundary
        )
        self.assertTrue(plan.local_cloud_routing_implemented)
        self.assertFalse(plan.local_model_discovery_implemented)
        self.assertFalse(plan.local_provider_execution_implemented)
        self.assertFalse(plan.cloud_provider_execution_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.model_selection_implemented)
        self.assertFalse(plan.model_switching_implemented)
        self.assertFalse(plan.hybrid_routing_implemented)
        self.assertFalse(plan.cost_optimization_implemented)
        self.assertFalse(plan.quality_optimization_implemented)
        self.assertFalse(plan.budget_enforcement_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.advisory_only)

    def test_local_cloud_decisions_are_advisory_and_source_aligned(self) -> None:
        plan = route_local_vs_cloud(route=RouteName.PREVIEW)
        known_local_ids = set(local_model_registry().surface_ids)
        known_cloud_ids = set(cloud_model_registry().surface_ids)

        for decision in plan.decisions:
            dumped = decision.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_LOCAL_CLOUD_DECISION_FIELDS)
            self.assertEqual(
                decision.serialization_version,
                "local_cloud_route_decision.v1",
            )
            self.assertEqual(decision.route_name, RouteName.PREVIEW)
            self.assertTrue(set(decision.local_surface_ids).issubset(known_local_ids))
            self.assertTrue(set(decision.cloud_surface_ids).issubset(known_cloud_ids))
            self.assertIn(
                "provider_or_model_routing",
                decision.blocked_runtime_behaviors,
            )
            self.assertTrue(decision.local_cloud_routing_implemented)
            self.assertFalse(decision.local_model_discovery_implemented)
            self.assertFalse(decision.local_provider_execution_implemented)
            self.assertFalse(decision.cloud_provider_execution_implemented)
            self.assertFalse(decision.provider_model_routing_implemented)
            self.assertFalse(decision.model_selection_implemented)
            self.assertFalse(decision.model_switching_implemented)
            self.assertFalse(decision.hybrid_routing_implemented)
            self.assertFalse(decision.cost_optimization_implemented)
            self.assertFalse(decision.quality_optimization_implemented)
            self.assertFalse(decision.budget_enforcement_implemented)
            self.assertFalse(decision.workflow_control_implemented)
            self.assertFalse(decision.retry_triggering_implemented)
            self.assertFalse(decision.prompt_mutation_implemented)
            self.assertFalse(decision.generated_output_mutation_implemented)
            self.assertTrue(decision.advisory_only)

    def test_review_route_prioritizes_cloud_posture_and_lookup_helpers(self) -> None:
        plan = route_local_vs_cloud(route=RouteName.REVIEW)
        recommended = local_cloud_route_decision_by_id(
            "local_cloud_route::evaluation_review_model_profile",
            plan,
        )
        cloud_decisions = local_cloud_route_decisions_for_lane("cloud_candidate", plan)
        missing = local_cloud_route_decision_by_id("missing", plan)

        self.assertIsNone(missing)
        self.assertIsNotNone(recommended)
        assert recommended is not None
        self.assertEqual(plan.recommended_routing_lane, "cloud_candidate")
        self.assertEqual(plan.recommended_routing_posture, "cloud_preferred")
        self.assertEqual(recommended.decision_status, "recommended")
        self.assertIn(recommended, cloud_decisions)

    def test_plan_rejects_mismatched_decisions_or_recommendation(self) -> None:
        plan = route_local_vs_cloud(route=RouteName.GENERATE)
        payload = plan.model_dump(mode="json")
        payload["decision_ids"] = ("missing",) + tuple(payload["decision_ids"][1:])

        with self.assertRaisesRegex(ValueError, "decision_ids must match"):
            LocalCloudRoutingPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["recommended_decision_id"] = "missing"

        with self.assertRaisesRegex(ValueError, "recommended_decision_id must match"):
            LocalCloudRoutingPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["routing_confidence"] = "high"

        with self.assertRaisesRegex(ValueError, "routing_confidence must match"):
            LocalCloudRoutingPlan(**payload)

    def test_local_cloud_router_does_not_change_request_routing_or_provider_factory(
        self,
    ) -> None:
        request = AssistantRequest(
            query="Review this Three.js shader for performance.",
            mode=AssistantMode.REVIEW,
            domains=(CreativeCodingDomain.THREE_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = route_local_vs_cloud(route_decision=baseline_decision)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, baseline_decision.route)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_local_cloud_router_does_not_declare_runtime_application_terms(
        self,
    ) -> None:
        plan = route_local_vs_cloud(route=RouteName.PREVIEW)
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
                        decision.source_model_route_candidate_id,
                        decision.source_model_profile_id,
                        *decision.local_surface_ids,
                        *decision.cloud_surface_ids,
                        *decision.evidence,
                        *decision.advisory_actions,
                        *decision.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "discover_local_models(",
            "select_model(",
            "switch_model(",
            "route_provider(",
            "execute_provider(",
            "call_provider(",
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
