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
    EmergenceOptimizationPlan,
    emergence_candidate_by_id,
    emergence_candidates_for_status,
    optimize_emergence,
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
REQUIRED_CANDIDATE_FIELDS = {
    "candidate_id",
    "emergence_kind",
    "emergence_mode",
    "status",
    "route_name",
    "task_type",
    "execution_mode_id",
    "topic_id",
    "source_creative_exploration_candidate_id",
    "source_creative_analytics_panel_id",
    "source_workflow_risk_factor_id",
    "source_execution_confidence_signal_id",
    "provider_sequence",
    "model_profile_sequence",
    "hybrid_policy_direction",
    "unavailable_reason_codes",
    "exploration_status",
    "diversity_band",
    "workflow_risk_severity",
    "analytics_panel_status",
    "creative_signal_count",
    "guardrail_signal_count",
    "creative_exploration_score",
    "execution_confidence_score",
    "analytics_signal_score",
    "mode_weight",
    "guardrail_penalty",
    "emergence_potential_score",
    "recommended_emergence_path_count",
    "applied_emergence_path_count",
    "hitl_required",
    "emergence_summary",
    "fallback_summary",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "emergence_optimizer_implemented",
    "emergence_potential_metadata_implemented",
    "creative_exploration_metadata_used",
    "creative_analytics_metadata_used",
    "workflow_risk_metadata_used",
    "provider_intelligence_metadata_used",
    "availability_awareness_metadata_used",
    "manual_assisted_auto_mode_metadata_used",
    "hybrid_transition_metadata_used",
    "task_aware_category_metadata_used",
    "execution_simulation_metadata_used",
    "fallback_safety_metadata_used",
    "emergence_behavior_application_implemented",
    "emergent_variant_generation_implemented",
    "variant_generation_implemented",
    "variant_selection_implemented",
    "artifact_selection_implemented",
    "generated_output_evaluation_implemented",
    "creative_metric_collection_implemented",
    "refinement_triggering_implemented",
    "budget_enforcement_implemented",
    "provider_model_routing_implemented",
    "automatic_provider_switching_implemented",
    "automatic_model_switching_implemented",
    "provider_execution_implemented",
    "local_runtime_probe_implemented",
    "local_model_inventory_scan_implemented",
    "automatic_model_download_implemented",
    "agent_invocation_implemented",
    "resource_allocation_implemented",
    "hitl_request_emitted",
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


class EmergenceOptimizerTests(unittest.TestCase):
    def test_plan_combines_exploration_and_analytics_sources(self) -> None:
        plan = optimize_emergence(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "emergence_optimizer")
        self.assertEqual(
            plan.serialization_version,
            "emergence_optimization_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.source_creative_exploration_serialization_version,
            "creative_exploration_optimization_plan.v1",
        )
        self.assertEqual(
            plan.source_creative_analytics_serialization_version,
            "creative_analytics.v1",
        )
        self.assertEqual(plan.provider_ids, REQUIRED_PROVIDER_IDS)
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.hybrid_policy_directions, REQUIRED_HYBRID_DIRECTIONS)
        self.assertEqual(plan.candidate_count, 4)
        self.assertEqual(plan.recommended_candidate_count, 1)
        self.assertEqual(len(plan.bounded_candidate_ids), 1)
        self.assertEqual(plan.guardrail_candidate_count, 2)
        self.assertEqual(plan.hitl_required_candidate_count, 4)
        self.assertFalse(plan.applied_emergence_candidate_ids)
        self.assertEqual(plan.total_recommended_emergence_path_count, 2)
        self.assertEqual(plan.total_applied_emergence_path_count, 0)
        self.assertEqual(plan.highest_emergence_potential_score, 291)
        self.assertIn(
            "does not generate emergent variants",
            plan.authority_boundary,
        )
        self.assertTrue(plan.emergence_optimizer_implemented)
        self.assertTrue(plan.emergence_potential_metadata_implemented)
        self.assertTrue(plan.creative_exploration_metadata_used)
        self.assertTrue(plan.creative_analytics_metadata_used)
        self.assertTrue(plan.workflow_risk_metadata_used)
        self.assertTrue(plan.provider_intelligence_metadata_used)
        self.assertTrue(plan.availability_awareness_metadata_used)
        self.assertTrue(plan.manual_assisted_auto_mode_metadata_used)
        self.assertTrue(plan.hybrid_transition_metadata_used)
        self.assertTrue(plan.task_aware_category_metadata_used)
        self.assertTrue(plan.execution_simulation_metadata_used)
        self.assertTrue(plan.fallback_safety_metadata_used)
        self.assertFalse(plan.emergence_behavior_application_implemented)
        self.assertFalse(plan.emergent_variant_generation_implemented)
        self.assertFalse(plan.variant_generation_implemented)
        self.assertFalse(plan.variant_selection_implemented)
        self.assertFalse(plan.artifact_selection_implemented)
        self.assertFalse(plan.generated_output_evaluation_implemented)
        self.assertFalse(plan.creative_metric_collection_implemented)
        self.assertFalse(plan.refinement_triggering_implemented)
        self.assertFalse(plan.budget_enforcement_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.automatic_provider_switching_implemented)
        self.assertFalse(plan.automatic_model_switching_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.local_runtime_probe_implemented)
        self.assertFalse(plan.local_model_inventory_scan_implemented)
        self.assertFalse(plan.automatic_model_download_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.resource_allocation_implemented)
        self.assertFalse(plan.hitl_request_emitted)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_candidates_score_emergence_without_applying_behavior(self) -> None:
        plan = optimize_emergence(route="generate")

        for candidate in plan.candidates:
            dumped = candidate.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_CANDIDATE_FIELDS)
            self.assertEqual(
                candidate.serialization_version,
                "emergence_optimization_candidate.v1",
            )
            self.assertEqual(candidate.route_name, RouteName.GENERATE)
            self.assertEqual(
                candidate.candidate_id,
                f"emergence_optimizer::{candidate.emergence_kind}",
            )
            self.assertEqual(
                candidate.analytics_signal_score,
                min(
                    180,
                    max(
                        0,
                        candidate.creative_signal_count // 8
                        - candidate.guardrail_signal_count * 2
                        + 60,
                    ),
                ),
            )
            expected_penalty = candidate.guardrail_signal_count * 6
            if candidate.status == "guardrail":
                expected_penalty += 120
            elif candidate.analytics_panel_status == "guarded":
                expected_penalty += 40
            if candidate.workflow_risk_severity == "guarded":
                expected_penalty += 60
            elif candidate.workflow_risk_severity == "high":
                expected_penalty += 30
            self.assertEqual(candidate.guardrail_penalty, min(320, expected_penalty))
            self.assertEqual(
                candidate.emergence_potential_score,
                min(
                    500,
                    max(
                        0,
                        candidate.creative_exploration_score
                        + candidate.analytics_signal_score
                        + candidate.execution_confidence_score
                        + candidate.mode_weight
                        - candidate.guardrail_penalty,
                    ),
                ),
            )
            self.assertEqual(candidate.applied_emergence_path_count, 0)
            self.assertIn(
                "emergence_behavior_application",
                candidate.blocked_runtime_behaviors,
            )
            self.assertTrue(candidate.emergence_optimizer_implemented)
            self.assertTrue(candidate.creative_exploration_metadata_used)
            self.assertTrue(candidate.creative_analytics_metadata_used)
            self.assertTrue(candidate.workflow_risk_metadata_used)
            self.assertTrue(candidate.provider_intelligence_metadata_used)
            self.assertFalse(candidate.emergence_behavior_application_implemented)
            self.assertFalse(candidate.emergent_variant_generation_implemented)
            self.assertFalse(candidate.variant_generation_implemented)
            self.assertFalse(candidate.variant_selection_implemented)
            self.assertFalse(candidate.artifact_selection_implemented)
            self.assertFalse(candidate.generated_output_evaluation_implemented)
            self.assertFalse(candidate.creative_metric_collection_implemented)
            self.assertFalse(candidate.refinement_triggering_implemented)
            self.assertFalse(candidate.budget_enforcement_implemented)
            self.assertFalse(candidate.provider_model_routing_implemented)
            self.assertFalse(candidate.automatic_provider_switching_implemented)
            self.assertFalse(candidate.automatic_model_switching_implemented)
            self.assertFalse(candidate.provider_execution_implemented)
            self.assertFalse(candidate.local_runtime_probe_implemented)
            self.assertFalse(candidate.local_model_inventory_scan_implemented)
            self.assertFalse(candidate.automatic_model_download_implemented)
            self.assertFalse(candidate.agent_invocation_implemented)
            self.assertFalse(candidate.resource_allocation_implemented)
            self.assertFalse(candidate.hitl_request_emitted)
            self.assertFalse(candidate.workflow_control_implemented)
            self.assertFalse(candidate.workflow_graph_mutation_implemented)
            self.assertFalse(candidate.workflow_execution_implemented)
            self.assertFalse(candidate.retry_triggering_implemented)
            self.assertFalse(candidate.generated_output_mutation_implemented)
            self.assertTrue(candidate.hitl_required)
            self.assertTrue(candidate.advisory_only)

        recommended = emergence_candidate_by_id(
            "emergence_optimizer::aesthetic_emergence",
            plan,
        )
        guardrails = emergence_candidates_for_status("guardrail", plan)
        self.assertIsNotNone(recommended)
        assert recommended is not None
        self.assertEqual(recommended.status, "recommended")
        self.assertEqual(recommended.emergence_mode, "diversity_emergence")
        self.assertEqual(recommended.recommended_emergence_path_count, 1)
        self.assertEqual(len(guardrails), 2)
        self.assertTrue(
            all(
                candidate.recommended_emergence_path_count == 0
                for candidate in guardrails
            )
        )

    def test_plan_rejects_mismatched_emergence_metadata(self) -> None:
        plan = optimize_emergence()
        payload = plan.model_dump(mode="json")
        payload["candidate_ids"] = ("missing",) + tuple(payload["candidate_ids"][1:])

        with self.assertRaisesRegex(ValueError, "candidate_ids must match"):
            EmergenceOptimizationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["highest_emergence_potential_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "highest_emergence_potential_score must match",
        ):
            EmergenceOptimizationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["applied_emergence_candidate_ids"] = (plan.candidate_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "applied_emergence_candidate_ids must remain empty",
        ):
            EmergenceOptimizationPlan(**payload)

    def test_optimizer_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Optimize emergence posture for a visual study workflow.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = optimize_emergence(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_optimizer_does_not_declare_runtime_emergence_terms(self) -> None:
        plan = optimize_emergence(route=RouteName.GENERATE)
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
                        candidate.emergence_kind,
                        candidate.emergence_mode,
                        candidate.topic_id,
                        candidate.source_creative_exploration_candidate_id,
                        candidate.source_creative_analytics_panel_id,
                        candidate.source_workflow_risk_factor_id,
                        candidate.source_execution_confidence_signal_id,
                        *candidate.provider_sequence,
                        *candidate.model_profile_sequence,
                        candidate.emergence_summary,
                        candidate.fallback_summary,
                        *candidate.advisory_actions,
                        *candidate.evidence,
                        *candidate.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "apply_emergence(",
            "generate_emergent_variant(",
            "generate_variant(",
            "select_variant(",
            "select_artifact(",
            "evaluate_output(",
            "collect_creative_metric(",
            "trigger_refinement(",
            "enforce_budget(",
            "route_provider(",
            "switch_provider(",
            "switch_model(",
            "execute_provider(",
            "probe_local_runtime(",
            "scan_local_models(",
            "download_model(",
            "invoke_agent(",
            "allocate_resource(",
            "emit_hitl_request(",
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
