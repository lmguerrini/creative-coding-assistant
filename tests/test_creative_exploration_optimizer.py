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
    CreativeExplorationOptimizationPlan,
    creative_exploration_candidate_by_id,
    creative_exploration_candidates_for_status,
    optimize_creative_exploration,
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
    "topic_id",
    "strategy",
    "status",
    "route_name",
    "task_type",
    "execution_mode_id",
    "source_exploration_allocation_id",
    "source_diversity_prediction_id",
    "source_workflow_risk_factor_id",
    "source_execution_confidence_signal_id",
    "source_budget_profile_id",
    "provider_sequence",
    "model_profile_sequence",
    "hybrid_policy_direction",
    "unavailable_reason_codes",
    "budget_posture",
    "diversity_band",
    "exploration_pressure",
    "workflow_risk_severity",
    "priority",
    "requested_variants",
    "planned_variants",
    "recommended_advisory_variants",
    "applied_variant_count",
    "requested_refinement_passes",
    "planned_refinement_passes",
    "recommended_advisory_refinement_passes",
    "applied_refinement_pass_count",
    "diversity_readiness_score",
    "workflow_risk_score",
    "execution_confidence_score",
    "exploration_budget_score",
    "priority_weight",
    "risk_penalty",
    "creative_exploration_score",
    "hitl_required",
    "optimization_summary",
    "fallback_summary",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "creative_exploration_optimizer_implemented",
    "exploration_optimization_metadata_implemented",
    "exploration_budget_metadata_used",
    "creative_diversity_prediction_metadata_used",
    "workflow_risk_metadata_used",
    "provider_intelligence_metadata_used",
    "availability_awareness_metadata_used",
    "manual_assisted_auto_mode_metadata_used",
    "hybrid_transition_metadata_used",
    "task_aware_category_metadata_used",
    "execution_simulation_metadata_used",
    "fallback_safety_metadata_used",
    "variant_generation_implemented",
    "variant_selection_implemented",
    "artifact_selection_implemented",
    "refinement_triggering_implemented",
    "budget_enforcement_implemented",
    "cost_routing_implemented",
    "provider_model_routing_implemented",
    "automatic_provider_switching_implemented",
    "automatic_model_switching_implemented",
    "provider_execution_implemented",
    "local_runtime_probe_implemented",
    "local_model_inventory_scan_implemented",
    "automatic_model_download_implemented",
    "agent_invocation_implemented",
    "resource_allocation_implemented",
    "risk_mitigation_execution_implemented",
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


class CreativeExplorationOptimizerTests(unittest.TestCase):
    def test_plan_combines_budget_diversity_and_workflow_risk_sources(self) -> None:
        plan = optimize_creative_exploration(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "creative_exploration_optimizer")
        self.assertEqual(
            plan.serialization_version,
            "creative_exploration_optimization_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.source_exploration_budget_serialization_version,
            "exploration_budget_plan.v1",
        )
        self.assertEqual(
            plan.source_creative_diversity_prediction_serialization_version,
            "creative_diversity_prediction_plan.v1",
        )
        self.assertEqual(
            plan.source_workflow_risk_serialization_version,
            "workflow_risk_plan.v1",
        )
        self.assertEqual(plan.provider_ids, REQUIRED_PROVIDER_IDS)
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.hybrid_policy_directions, REQUIRED_HYBRID_DIRECTIONS)
        self.assertEqual(plan.candidate_count, 4)
        self.assertEqual(plan.recommended_candidate_count, 1)
        self.assertEqual(len(plan.bounded_candidate_ids), 1)
        self.assertEqual(plan.guardrail_candidate_count, 2)
        self.assertEqual(plan.hitl_required_candidate_count, 4)
        self.assertFalse(plan.applied_exploration_candidate_ids)
        self.assertEqual(plan.total_applied_variant_count, 0)
        self.assertEqual(plan.total_applied_refinement_pass_count, 0)
        self.assertEqual(plan.total_recommended_advisory_variants, 5)
        self.assertEqual(plan.total_recommended_advisory_refinement_passes, 2)
        self.assertIn("does not generate variants", plan.authority_boundary)
        self.assertTrue(plan.creative_exploration_optimizer_implemented)
        self.assertTrue(plan.exploration_optimization_metadata_implemented)
        self.assertTrue(plan.exploration_budget_metadata_used)
        self.assertTrue(plan.creative_diversity_prediction_metadata_used)
        self.assertTrue(plan.workflow_risk_metadata_used)
        self.assertTrue(plan.provider_intelligence_metadata_used)
        self.assertTrue(plan.availability_awareness_metadata_used)
        self.assertTrue(plan.manual_assisted_auto_mode_metadata_used)
        self.assertTrue(plan.hybrid_transition_metadata_used)
        self.assertTrue(plan.task_aware_category_metadata_used)
        self.assertTrue(plan.execution_simulation_metadata_used)
        self.assertTrue(plan.fallback_safety_metadata_used)
        self.assertFalse(plan.variant_generation_implemented)
        self.assertFalse(plan.variant_selection_implemented)
        self.assertFalse(plan.artifact_selection_implemented)
        self.assertFalse(plan.refinement_triggering_implemented)
        self.assertFalse(plan.budget_enforcement_implemented)
        self.assertFalse(plan.cost_routing_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.automatic_provider_switching_implemented)
        self.assertFalse(plan.automatic_model_switching_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.local_runtime_probe_implemented)
        self.assertFalse(plan.local_model_inventory_scan_implemented)
        self.assertFalse(plan.automatic_model_download_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.resource_allocation_implemented)
        self.assertFalse(plan.risk_mitigation_execution_implemented)
        self.assertFalse(plan.hitl_request_emitted)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_candidates_optimize_exploration_without_generation(self) -> None:
        plan = optimize_creative_exploration(route="generate")

        for candidate in plan.candidates:
            dumped = candidate.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_CANDIDATE_FIELDS)
            self.assertEqual(
                candidate.serialization_version,
                "creative_exploration_optimization_candidate.v1",
            )
            self.assertEqual(candidate.route_name, RouteName.GENERATE)
            self.assertEqual(
                candidate.candidate_id,
                f"creative_exploration_optimizer::{candidate.topic_id}",
            )
            self.assertEqual(
                candidate.exploration_budget_score,
                min(
                    260,
                    candidate.planned_variants * 35
                    + candidate.planned_refinement_passes * 30
                    + candidate.priority_weight,
                ),
            )
            expected_penalty = candidate.workflow_risk_score // 4
            if candidate.workflow_risk_severity == "guarded" or (
                candidate.status == "guardrail"
            ):
                expected_penalty += 80
            elif candidate.workflow_risk_severity == "high":
                expected_penalty += 40
            self.assertEqual(candidate.risk_penalty, min(360, expected_penalty))
            self.assertEqual(
                candidate.creative_exploration_score,
                min(
                    500,
                    max(
                        0,
                        candidate.diversity_readiness_score * 2
                        + candidate.exploration_budget_score
                        + candidate.execution_confidence_score
                        - candidate.risk_penalty,
                    ),
                ),
            )
            self.assertEqual(candidate.applied_variant_count, 0)
            self.assertEqual(candidate.applied_refinement_pass_count, 0)
            self.assertIn("variant_generation", candidate.blocked_runtime_behaviors)
            self.assertTrue(candidate.creative_exploration_optimizer_implemented)
            self.assertTrue(candidate.exploration_budget_metadata_used)
            self.assertTrue(candidate.creative_diversity_prediction_metadata_used)
            self.assertTrue(candidate.workflow_risk_metadata_used)
            self.assertTrue(candidate.provider_intelligence_metadata_used)
            self.assertFalse(candidate.variant_generation_implemented)
            self.assertFalse(candidate.variant_selection_implemented)
            self.assertFalse(candidate.artifact_selection_implemented)
            self.assertFalse(candidate.refinement_triggering_implemented)
            self.assertFalse(candidate.budget_enforcement_implemented)
            self.assertFalse(candidate.cost_routing_implemented)
            self.assertFalse(candidate.provider_model_routing_implemented)
            self.assertFalse(candidate.automatic_provider_switching_implemented)
            self.assertFalse(candidate.automatic_model_switching_implemented)
            self.assertFalse(candidate.provider_execution_implemented)
            self.assertFalse(candidate.local_runtime_probe_implemented)
            self.assertFalse(candidate.local_model_inventory_scan_implemented)
            self.assertFalse(candidate.automatic_model_download_implemented)
            self.assertFalse(candidate.agent_invocation_implemented)
            self.assertFalse(candidate.resource_allocation_implemented)
            self.assertFalse(candidate.risk_mitigation_execution_implemented)
            self.assertFalse(candidate.hitl_request_emitted)
            self.assertFalse(candidate.workflow_control_implemented)
            self.assertFalse(candidate.workflow_graph_mutation_implemented)
            self.assertFalse(candidate.workflow_execution_implemented)
            self.assertFalse(candidate.retry_triggering_implemented)
            self.assertFalse(candidate.generated_output_mutation_implemented)
            self.assertTrue(candidate.hitl_required)
            self.assertTrue(candidate.advisory_only)

        recommended = creative_exploration_candidate_by_id(
            "creative_exploration_optimizer::style_aesthetic_alignment",
            plan,
        )
        guardrails = creative_exploration_candidates_for_status("guardrail", plan)
        self.assertIsNotNone(recommended)
        assert recommended is not None
        self.assertEqual(recommended.status, "recommended")
        self.assertEqual(recommended.strategy, "diversity_priority")
        self.assertEqual(recommended.recommended_advisory_variants, 3)
        self.assertEqual(len(guardrails), 2)
        self.assertTrue(
            all(candidate.recommended_advisory_variants == 0 for candidate in guardrails)
        )

    def test_plan_rejects_mismatched_exploration_metadata(self) -> None:
        plan = optimize_creative_exploration()
        payload = plan.model_dump(mode="json")
        payload["candidate_ids"] = ("missing",) + tuple(payload["candidate_ids"][1:])

        with self.assertRaisesRegex(ValueError, "candidate_ids must match"):
            CreativeExplorationOptimizationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["highest_creative_exploration_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "highest_creative_exploration_score must match",
        ):
            CreativeExplorationOptimizationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["applied_exploration_candidate_ids"] = (plan.candidate_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "applied_exploration_candidate_ids must remain empty",
        ):
            CreativeExplorationOptimizationPlan(**payload)

    def test_optimizer_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Optimize creative exploration for a visual study workflow.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = optimize_creative_exploration(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_optimizer_does_not_declare_runtime_generation_terms(self) -> None:
        plan = optimize_creative_exploration(route=RouteName.GENERATE)
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
                        candidate.topic_id,
                        candidate.strategy,
                        candidate.source_exploration_allocation_id,
                        candidate.source_diversity_prediction_id,
                        candidate.source_workflow_risk_factor_id,
                        candidate.source_execution_confidence_signal_id,
                        candidate.source_budget_profile_id,
                        *candidate.provider_sequence,
                        *candidate.model_profile_sequence,
                        candidate.optimization_summary,
                        candidate.fallback_summary,
                        *candidate.advisory_actions,
                        *candidate.evidence,
                        *candidate.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "generate_variant(",
            "select_variant(",
            "select_artifact(",
            "trigger_refinement(",
            "enforce_budget(",
            "route_by_cost(",
            "route_provider(",
            "switch_provider(",
            "switch_model(",
            "execute_provider(",
            "probe_local_runtime(",
            "scan_local_models(",
            "download_model(",
            "invoke_agent(",
            "allocate_resource(",
            "execute_mitigation(",
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
