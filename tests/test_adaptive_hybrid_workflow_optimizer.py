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
    HybridWorkflowOptimizationPlan,
    hybrid_workflow_candidate_by_id,
    hybrid_workflow_candidates_requiring_hitl,
    optimize_hybrid_workflow,
    route_request,
)

REQUIRED_PROVIDER_IDS = ("openai", "anthropic", "gemini", "local")
REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
REQUIRED_HYBRID_DIRECTIONS = (
    "local_to_cloud",
    "cloud_to_local",
    "cloud_to_cloud",
    "local_to_local",
)
REQUIRED_CANDIDATE_FIELDS = {
    "candidate_id",
    "task_type",
    "capability_requirements",
    "policy_direction",
    "source_policy_id",
    "source_task_decision_id",
    "source_hybrid_route_decision_id",
    "execution_mode_id",
    "provider_sequence",
    "surface_sequence",
    "model_profile_sequence",
    "workflow_path_candidate_id",
    "status",
    "estimated_quality",
    "estimated_cost",
    "estimated_latency",
    "confidence_score",
    "risk_band",
    "unavailable_reason_codes",
    "hitl_required",
    "adaptive_score",
    "simulation",
    "fallback_reason_summary",
    "suggested_action",
    "tradeoffs",
    "evidence",
    "blocked_runtime_behaviors",
    "hybrid_workflow_optimizer_implemented",
    "adaptive_execution_intelligence_implemented",
    "strategy_recommendation_implemented",
    "execution_strategy_selection_implemented",
    "automatic_provider_switching_implemented",
    "automatic_model_switching_implemented",
    "automatic_model_download_implemented",
    "automatic_api_key_assumption_implemented",
    "provider_execution_implemented",
    "local_runtime_probe_implemented",
    "local_model_inventory_scan_implemented",
    "workflow_control_implemented",
    "hitl_request_emitted",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}


class AdaptiveHybridWorkflowOptimizerTests(unittest.TestCase):
    def test_plan_combines_path_and_routing_intelligence(self) -> None:
        plan = optimize_hybrid_workflow(task_type="creative_coding")

        self.assertEqual(plan.role, "hybrid_workflow_optimizer")
        self.assertEqual(
            plan.serialization_version,
            "adaptive_hybrid_workflow_optimization_plan.v1",
        )
        self.assertEqual(
            plan.source_path_optimization_serialization_version,
            "execution_path_optimization_plan.v1",
        )
        self.assertEqual(
            plan.source_hybrid_routing_serialization_version,
            "hybrid_routing_plan.v1",
        )
        self.assertEqual(
            plan.source_routing_intelligence_serialization_version,
            "model_routing_intelligence_registry.v1",
        )
        self.assertEqual(plan.provider_ids, REQUIRED_PROVIDER_IDS)
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.hybrid_policy_directions, REQUIRED_HYBRID_DIRECTIONS)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.decision_count, 4)
        self.assertEqual(plan.recommended_candidate_id, "hybrid_workflow::local_to_cloud")
        self.assertIsNone(plan.selected_candidate_id)
        self.assertIn("does not execute workflows", plan.authority_boundary)
        self.assertTrue(plan.hybrid_workflow_optimizer_implemented)
        self.assertTrue(plan.adaptive_execution_policy_metadata_implemented)
        self.assertTrue(plan.execution_simulation_estimates_implemented)
        self.assertTrue(plan.fallback_intelligence_implemented)
        self.assertFalse(plan.execution_strategy_selection_implemented)
        self.assertFalse(plan.automatic_provider_switching_implemented)
        self.assertFalse(plan.automatic_model_download_implemented)
        self.assertFalse(plan.automatic_api_key_assumption_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.local_runtime_probe_implemented)
        self.assertFalse(plan.local_model_inventory_scan_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.hitl_request_emitted)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.advisory_only)

    def test_candidates_cover_hybrid_policies_and_availability(self) -> None:
        plan = optimize_hybrid_workflow(task_type="creative_coding")

        self.assertEqual(
            tuple(candidate.policy_direction for candidate in plan.candidates),
            REQUIRED_HYBRID_DIRECTIONS,
        )
        self.assertGreaterEqual(plan.hitl_required_candidate_count, 1)
        for candidate in plan.candidates:
            dumped = candidate.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_CANDIDATE_FIELDS)
            self.assertEqual(
                candidate.serialization_version,
                "adaptive_hybrid_workflow_candidate.v1",
            )
            self.assertIn(candidate.execution_mode_id, REQUIRED_EXECUTION_MODES)
            self.assertEqual(
                len(candidate.surface_sequence),
                len(candidate.provider_sequence),
            )
            self.assertEqual(
                len(candidate.model_profile_sequence),
                len(candidate.provider_sequence),
            )
            self.assertIn("provider_or_model_routing_application", candidate.blocked_runtime_behaviors)
            self.assertTrue(candidate.hybrid_workflow_optimizer_implemented)
            self.assertTrue(candidate.adaptive_execution_intelligence_implemented)
            self.assertTrue(candidate.strategy_recommendation_implemented)
            self.assertFalse(candidate.execution_strategy_selection_implemented)
            self.assertFalse(candidate.automatic_provider_switching_implemented)
            self.assertFalse(candidate.automatic_model_switching_implemented)
            self.assertFalse(candidate.automatic_model_download_implemented)
            self.assertFalse(candidate.automatic_api_key_assumption_implemented)
            self.assertFalse(candidate.provider_execution_implemented)
            self.assertFalse(candidate.local_runtime_probe_implemented)
            self.assertFalse(candidate.local_model_inventory_scan_implemented)
            self.assertFalse(candidate.workflow_control_implemented)
            self.assertFalse(candidate.hitl_request_emitted)
            self.assertFalse(candidate.prompt_mutation_implemented)
            self.assertFalse(candidate.persistent_storage_write_implemented)
            self.assertFalse(candidate.generated_output_mutation_implemented)
            self.assertTrue(candidate.advisory_only)

        local_to_cloud = hybrid_workflow_candidate_by_id(
            "hybrid_workflow::local_to_cloud",
            plan,
        )
        cloud_to_cloud = hybrid_workflow_candidate_by_id(
            "hybrid_workflow::cloud_to_cloud",
            plan,
        )
        self.assertIsNotNone(local_to_cloud)
        self.assertIsNotNone(cloud_to_cloud)
        assert local_to_cloud is not None
        assert cloud_to_cloud is not None
        self.assertEqual(local_to_cloud.provider_sequence, ("local", "openai"))
        self.assertIn("missing_api_key", local_to_cloud.unavailable_reason_codes)
        self.assertIn("local_runtime_unavailable", local_to_cloud.unavailable_reason_codes)
        self.assertEqual(cloud_to_cloud.provider_sequence, ("openai", "anthropic", "gemini"))
        self.assertIn("provider_unsupported", cloud_to_cloud.unavailable_reason_codes)
        self.assertTrue(local_to_cloud.hitl_required)
        self.assertTrue(cloud_to_cloud.hitl_required)

    def test_simulation_and_fallback_are_pre_run_estimates(self) -> None:
        plan = optimize_hybrid_workflow(task_type="fast_draft", execution_mode_id="auto_mode")
        recommended = hybrid_workflow_candidate_by_id(plan.recommended_candidate_id, plan)
        hitl_candidates = hybrid_workflow_candidates_requiring_hitl(plan)

        self.assertIsNotNone(recommended)
        assert recommended is not None
        self.assertEqual(recommended.execution_mode_id, "auto_mode")
        self.assertGreaterEqual(len(hitl_candidates), 1)
        for candidate in plan.candidates:
            simulation = candidate.simulation
            self.assertEqual(
                simulation.serialization_version,
                "adaptive_hybrid_workflow_simulation.v1",
            )
            self.assertEqual(simulation.policy_direction, candidate.policy_direction)
            self.assertEqual(simulation.provider_sequence, candidate.provider_sequence)
            self.assertEqual(
                simulation.model_profile_sequence,
                candidate.model_profile_sequence,
            )
            self.assertEqual(
                simulation.workflow_path_candidate_id,
                candidate.workflow_path_candidate_id,
            )
            self.assertIn(
                simulation.estimated_quality,
                {"low", "medium", "high", "maximum"},
            )
            self.assertIn(simulation.estimated_cost, {"low", "medium", "high"})
            self.assertIn(simulation.estimated_latency, {"fast", "moderate", "slow"})
            self.assertFalse(simulation.execution_simulation_run)
            self.assertFalse(simulation.provider_call_performed)
            self.assertFalse(simulation.workflow_execution_implemented)
            self.assertFalse(simulation.provider_model_routing_implemented)
            self.assertTrue(simulation.metadata_only)

        self.assertEqual(
            plan.fallback.serialization_version,
            "adaptive_hybrid_workflow_fallback.v1",
        )
        self.assertEqual(plan.fallback.preferred_candidate_id, plan.recommended_candidate_id)
        self.assertIn(plan.fallback.fallback_candidate_id, plan.fallback_candidate_ids)
        self.assertTrue(plan.fallback.reason_summary)
        self.assertTrue(plan.fallback.suggested_action)
        self.assertTrue(plan.fallback.tradeoffs)
        self.assertTrue(plan.fallback.fallback_intelligence_implemented)
        self.assertFalse(plan.fallback.fallback_application_implemented)
        self.assertFalse(plan.fallback.provider_execution_implemented)
        self.assertFalse(plan.fallback.workflow_control_implemented)
        self.assertFalse(plan.fallback.hitl_request_emitted)
        self.assertFalse(plan.fallback.generated_output_mutation_implemented)

    def test_plan_rejects_mismatched_candidates_or_scores(self) -> None:
        plan = optimize_hybrid_workflow()
        payload = plan.model_dump(mode="json")
        payload["candidate_ids"] = ("missing",) + tuple(payload["candidate_ids"][1:])

        with self.assertRaisesRegex(ValueError, "candidate_ids must match"):
            HybridWorkflowOptimizationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["recommended_candidate_id"] = "missing"

        with self.assertRaisesRegex(ValueError, "recommended_candidate_id must match"):
            HybridWorkflowOptimizationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["highest_adaptive_score"] += 1

        with self.assertRaisesRegex(ValueError, "highest_adaptive_score must match"):
            HybridWorkflowOptimizationPlan(**payload)

    def test_optimizer_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Build a p5.js sketch with an iterative preview.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = optimize_hybrid_workflow(task_type="creative_coding")
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_optimizer_does_not_declare_runtime_application_terms(self) -> None:
        plan = optimize_hybrid_workflow(task_type="creative_coding")
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *(
                    field
                    for candidate in plan.candidates
                    for field in (
                        candidate.candidate_id,
                        candidate.source_policy_id,
                        candidate.source_task_decision_id,
                        candidate.source_hybrid_route_decision_id,
                        candidate.workflow_path_candidate_id,
                        *candidate.provider_sequence,
                        *candidate.surface_sequence,
                        *candidate.model_profile_sequence,
                        *candidate.fallback_reason_summary.split(),
                        *candidate.suggested_action.split(),
                        *candidate.tradeoffs,
                        *candidate.evidence,
                        *candidate.blocked_runtime_behaviors,
                    )
                ),
                plan.fallback.reason_summary,
                plan.fallback.suggested_action,
                *plan.fallback.tradeoffs,
                *plan.fallback.blocked_runtime_behaviors,
            )
        )

        for forbidden_term in (
            "execute_hybrid_workflow(",
            "select_execution_strategy(",
            "switch_provider(",
            "switch_model(",
            "download_model(",
            "assume_api_key(",
            "route_provider(",
            "execute_provider(",
            "call_provider(",
            "probe_local_runtime(",
            "scan_local_models(",
            "merge_provider_outputs(",
            "control_workflow(",
            "emit_hitl_request(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
