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
    HybridRoutingPlan,
    hybrid_route_decision_by_id,
    hybrid_route_decisions_for_mode,
    route_hybrid_model_request,
    route_local_vs_cloud,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_HYBRID_DECISION_FIELDS = {
    "decision_id",
    "source_local_cloud_decision_id",
    "source_model_route_candidate_id",
    "source_model_profile_id",
    "route_name",
    "hybrid_mode",
    "source_routing_lane",
    "local_surface_ids",
    "cloud_surface_ids",
    "local_score",
    "cloud_score",
    "hybrid_score",
    "comparison_delta",
    "status",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "hybrid_routing_implemented",
    "hybrid_routing_application_implemented",
    "hybrid_workflow_execution_implemented",
    "provider_output_merging_implemented",
    "provider_model_routing_implemented",
    "model_selection_implemented",
    "model_switching_implemented",
    "local_provider_execution_implemented",
    "cloud_provider_execution_implemented",
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


class HybridRoutingTests(unittest.TestCase):
    def test_default_hybrid_plan_derives_from_local_cloud_routing(self) -> None:
        local_cloud_plan = route_local_vs_cloud(route=RouteName.GENERATE)
        plan = route_hybrid_model_request(local_cloud_routing=local_cloud_plan)

        self.assertEqual(plan.role, "hybrid_router")
        self.assertEqual(plan.serialization_version, "hybrid_routing_plan.v1")
        self.assertEqual(
            plan.source_local_cloud_serialization_version,
            local_cloud_plan.serialization_version,
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(
            plan.recommended_decision_id,
            "hybrid_route::creative_reasoning_model_profile",
        )
        self.assertEqual(plan.recommended_hybrid_mode, "balanced_dual")
        self.assertEqual(plan.decision_count, 3)
        self.assertEqual(plan.local_primary_count, 1)
        self.assertEqual(plan.cloud_primary_count, 0)
        self.assertEqual(plan.balanced_dual_count, 2)
        self.assertEqual(plan.routing_confidence, "medium")
        self.assertIn("does not execute hybrid workflows", plan.authority_boundary)
        self.assertTrue(plan.hybrid_routing_implemented)
        self.assertFalse(plan.hybrid_routing_application_implemented)
        self.assertFalse(plan.hybrid_workflow_execution_implemented)
        self.assertFalse(plan.provider_output_merging_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.model_selection_implemented)
        self.assertFalse(plan.model_switching_implemented)
        self.assertFalse(plan.local_provider_execution_implemented)
        self.assertFalse(plan.cloud_provider_execution_implemented)
        self.assertFalse(plan.cost_optimization_implemented)
        self.assertFalse(plan.quality_optimization_implemented)
        self.assertFalse(plan.budget_enforcement_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.advisory_only)

    def test_hybrid_decisions_are_advisory_and_source_aligned(self) -> None:
        local_cloud_plan = route_local_vs_cloud(route=RouteName.PREVIEW)
        plan = route_hybrid_model_request(local_cloud_routing=local_cloud_plan)
        source_decision_ids = set(local_cloud_plan.decision_ids)

        for decision in plan.decisions:
            dumped = decision.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_HYBRID_DECISION_FIELDS)
            self.assertEqual(decision.serialization_version, "hybrid_route_decision.v1")
            self.assertEqual(decision.route_name, RouteName.PREVIEW)
            self.assertIn(decision.source_local_cloud_decision_id, source_decision_ids)
            self.assertEqual(
                decision.hybrid_score, decision.local_score + decision.cloud_score
            )
            self.assertIn(
                "provider_or_model_routing", decision.blocked_runtime_behaviors
            )
            self.assertTrue(decision.hybrid_routing_implemented)
            self.assertFalse(decision.hybrid_routing_application_implemented)
            self.assertFalse(decision.hybrid_workflow_execution_implemented)
            self.assertFalse(decision.provider_output_merging_implemented)
            self.assertFalse(decision.provider_model_routing_implemented)
            self.assertFalse(decision.model_selection_implemented)
            self.assertFalse(decision.model_switching_implemented)
            self.assertFalse(decision.local_provider_execution_implemented)
            self.assertFalse(decision.cloud_provider_execution_implemented)
            self.assertFalse(decision.cost_optimization_implemented)
            self.assertFalse(decision.quality_optimization_implemented)
            self.assertFalse(decision.budget_enforcement_implemented)
            self.assertFalse(decision.workflow_control_implemented)
            self.assertFalse(decision.retry_triggering_implemented)
            self.assertFalse(decision.prompt_mutation_implemented)
            self.assertFalse(decision.generated_output_mutation_implemented)
            self.assertTrue(decision.advisory_only)

    def test_review_route_prioritizes_cloud_primary_hybrid_lookup(self) -> None:
        plan = route_hybrid_model_request(route=RouteName.REVIEW)
        recommended = hybrid_route_decision_by_id(
            "hybrid_route::evaluation_review_model_profile",
            plan,
        )
        cloud_primary = hybrid_route_decisions_for_mode("cloud_primary", plan)
        missing = hybrid_route_decision_by_id("missing", plan)

        self.assertIsNone(missing)
        self.assertIsNotNone(recommended)
        assert recommended is not None
        self.assertEqual(plan.recommended_hybrid_mode, "cloud_primary")
        self.assertEqual(recommended.status, "recommended")
        self.assertIn(recommended, cloud_primary)

    def test_plan_rejects_mismatched_decisions_or_recommendation(self) -> None:
        plan = route_hybrid_model_request(route=RouteName.GENERATE)
        payload = plan.model_dump(mode="json")
        payload["decision_ids"] = ("missing",) + tuple(payload["decision_ids"][1:])

        with self.assertRaisesRegex(ValueError, "decision_ids must match"):
            HybridRoutingPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["recommended_decision_id"] = "missing"

        with self.assertRaisesRegex(ValueError, "recommended_decision_id must match"):
            HybridRoutingPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["recommended_hybrid_mode"] = "cloud_primary"

        with self.assertRaisesRegex(ValueError, "recommended_hybrid_mode must match"):
            HybridRoutingPlan(**payload)

    def test_hybrid_router_does_not_change_request_routing_or_provider_factory(
        self,
    ) -> None:
        request = AssistantRequest(
            query="Preview this generated WebGL scene.",
            mode=AssistantMode.PREVIEW,
            domains=(CreativeCodingDomain.THREE_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = route_hybrid_model_request(route_decision=baseline_decision)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, baseline_decision.route)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_hybrid_router_does_not_declare_runtime_application_terms(self) -> None:
        plan = route_hybrid_model_request(route=RouteName.REVIEW)
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
                        decision.source_local_cloud_decision_id,
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
            "execute_hybrid_workflow(",
            "apply_hybrid_route(",
            "select_model(",
            "switch_model(",
            "route_provider(",
            "execute_provider(",
            "call_provider(",
            "merge_provider_outputs(",
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
