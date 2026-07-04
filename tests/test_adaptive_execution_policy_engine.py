import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.core import Settings
from creative_coding_assistant.orchestration import (
    adaptive_execution_availability_context,
    adaptive_execution_option_by_id,
    adaptive_execution_options_for_readiness,
    evaluate_adaptive_execution_policy,
    route_request,
    simulate_adaptive_execution,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_TASK_TYPES = (
    "coding",
    "reasoning",
    "creative_coding",
    "creative_writing",
    "long_context_reasoning",
    "multimodal_understanding",
    "image_understanding",
    "tool_use",
    "structured_output",
    "fast_draft",
    "low_cost_execution",
    "maximum_quality_execution",
)
REQUIRED_HYBRID_DIRECTIONS = (
    "local_to_cloud",
    "cloud_to_local",
    "cloud_to_cloud",
    "local_to_local",
)
REQUIRED_MANUAL_ACTIONS = (
    "local_model_download",
    "provider_provisioning",
    "runtime_installation",
    "runtime_evolution_review",
)


class AdaptiveExecutionPolicyEngineTests(unittest.TestCase):
    def test_policy_returns_actionable_decision_object(self) -> None:
        plan = evaluate_adaptive_execution_policy(
            task_type="creative_coding",
            execution_mode_id="assisted_mode",
        )

        self.assertEqual(plan.role, "adaptive_execution_policy_engine")
        self.assertEqual(
            plan.serialization_version, "adaptive_execution_policy_plan.v1"
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.candidate_count, 5)
        self.assertEqual(plan.simulation_count, 5)
        self.assertEqual(plan.fallback_decision_count, 5)
        self.assertEqual(plan.hybrid_policy_count, 4)
        self.assertEqual(plan.task_coverage_count, 12)
        self.assertTrue(plan.actionable_execution_decision_implemented)
        self.assertTrue(plan.policy_application_implemented)
        self.assertTrue(plan.execution_policy_application_implemented)
        self.assertTrue(plan.execution_simulation_implemented)
        self.assertTrue(plan.availability_aware_execution_implemented)
        self.assertTrue(plan.intelligent_fallback_engine_implemented)
        self.assertTrue(plan.adaptive_escalation_policy_implemented)
        self.assertTrue(plan.hybrid_execution_policy_implemented)
        self.assertTrue(plan.task_aware_execution_engine_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.automatic_provider_switching_implemented)
        self.assertFalse(plan.automatic_model_download_implemented)
        self.assertFalse(plan.automatic_api_key_assumption_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.controlled_policy_only)
        self.assertFalse(plan.advisory_only)

        decision = plan.selected_decision
        self.assertEqual(decision.requested_execution_mode_id, "assisted_mode")
        self.assertEqual(decision.selected_execution_mode_id, "assisted_mode")
        self.assertIsNotNone(decision.recommended_option_id)
        self.assertIsNotNone(decision.recommended_strategy_id)
        self.assertTrue(decision.required_hitl_gates)
        self.assertTrue(decision.execution_requires_user_confirmation)
        self.assertFalse(decision.execution_allowed_now)
        self.assertFalse(decision.provider_model_routing_implemented)
        self.assertFalse(decision.provider_execution_implemented)

    def test_execution_simulation_is_deterministic_and_offline(self) -> None:
        context = adaptive_execution_availability_context(
            configured_provider_ids=("openai",),
            safe_auto_risk_bands=("low", "medium"),
        )

        first = simulate_adaptive_execution(
            task_type="coding",
            execution_mode_id="auto_mode",
            availability_context=context,
        )
        second = simulate_adaptive_execution(
            task_type="coding",
            execution_mode_id="auto_mode",
            availability_context=context,
        )

        self.assertEqual(
            tuple(item.model_dump(mode="json") for item in first),
            tuple(item.model_dump(mode="json") for item in second),
        )
        for simulation in first:
            self.assertEqual(
                simulation.serialization_version,
                "adaptive_execution_simulation.v1",
            )
            self.assertFalse(simulation.provider_call_performed)
            self.assertFalse(simulation.network_call_performed)
            self.assertFalse(simulation.generation_execution_performed)
            self.assertFalse(simulation.workflow_execution_implemented)
            self.assertFalse(simulation.provider_model_routing_implemented)
            self.assertTrue(simulation.provider_model_path)
            self.assertIn(
                simulation.token_resource_posture,
                {
                    "low_token_pressure",
                    "moderate_token_pressure",
                    "high_token_pressure",
                },
            )

    def test_manual_mode_recommends_but_cannot_auto_select(self) -> None:
        context = adaptive_execution_availability_context(
            configured_provider_ids=("openai",),
        )
        plan = evaluate_adaptive_execution_policy(
            task_type="creative_writing",
            execution_mode_id="manual_mode",
            availability_context=context,
        )

        decision = plan.selected_decision
        self.assertEqual(decision.mode_transition, "manual_selection_required")
        self.assertIsNone(decision.selected_option_id)
        self.assertIsNone(decision.selected_strategy_id)
        self.assertEqual(decision.selected_provider_model_path, ())
        self.assertFalse(decision.execution_allowed_now)
        self.assertTrue(decision.execution_requires_user_confirmation)
        self.assertIn(
            "manual_provider_model_selection_required",
            decision.required_hitl_gates,
        )

    def test_assisted_mode_recommends_and_requires_confirmation(self) -> None:
        context = adaptive_execution_availability_context(
            configured_provider_ids=("openai",),
            safe_auto_risk_bands=("low", "medium"),
        )
        plan = evaluate_adaptive_execution_policy(
            task_type="coding",
            execution_mode_id="assisted_mode",
            availability_context=context,
        )

        decision = plan.selected_decision
        self.assertEqual(decision.mode_transition, "assisted_confirmation_required")
        self.assertIsNotNone(decision.selected_option_id)
        self.assertIsNotNone(decision.selected_strategy_id)
        self.assertTrue(decision.selected_provider_model_path)
        self.assertFalse(decision.execution_allowed_now)
        self.assertTrue(decision.execution_requires_user_confirmation)
        self.assertFalse(decision.execution_blocked)

    def test_auto_mode_selects_only_safe_available_path(self) -> None:
        context = adaptive_execution_availability_context(
            configured_provider_ids=("openai",),
            safe_auto_risk_bands=("low", "medium"),
        )
        plan = evaluate_adaptive_execution_policy(
            task_type="coding",
            execution_mode_id="auto_mode",
            availability_context=context,
        )

        decision = plan.selected_decision
        self.assertEqual(decision.mode_transition, "auto_selected")
        self.assertEqual(decision.selected_execution_mode_id, "auto_mode")
        self.assertEqual(
            decision.selected_option_id,
            "adaptive_execution_option::direct_task_provider",
        )
        self.assertTrue(decision.execution_allowed_now)
        self.assertFalse(decision.execution_requires_user_confirmation)
        self.assertFalse(decision.execution_blocked)
        self.assertEqual(decision.required_hitl_gates, ())
        self.assertEqual(decision.unavailable_reason_codes, ())
        for step in decision.selected_provider_model_path:
            self.assertTrue(step.immediate_execution_ready)
            self.assertFalse(step.provider_model_routing_implemented)
            self.assertFalse(step.provider_execution_implemented)

    def test_auto_mode_blocks_or_downgrades_when_hitl_is_required(self) -> None:
        plan = evaluate_adaptive_execution_policy(
            task_type="coding",
            execution_mode_id="auto_mode",
            availability_context=adaptive_execution_availability_context(),
        )

        decision = plan.selected_decision
        self.assertEqual(decision.mode_transition, "blocked")
        self.assertEqual(decision.selected_execution_mode_id, "assisted_mode")
        self.assertIsNone(decision.selected_option_id)
        self.assertFalse(decision.execution_allowed_now)
        self.assertTrue(decision.execution_requires_user_confirmation)
        self.assertTrue(decision.execution_blocked)
        self.assertIn("missing_api_key", decision.unavailable_reason_codes)
        self.assertIn("hitl_before_provider_provisioning", decision.required_hitl_gates)

    def test_unavailable_best_path_produces_fallback_and_suggested_action(self) -> None:
        plan = evaluate_adaptive_execution_policy(
            task_type="creative_coding",
            execution_mode_id="assisted_mode",
        )

        decision = plan.selected_decision
        self.assertTrue(decision.execution_blocked)
        self.assertTrue(decision.fallback.reason_summary)
        self.assertTrue(decision.fallback.tradeoff_summary)
        self.assertTrue(decision.fallback.execution_mode_impact)
        self.assertTrue(decision.suggested_action)
        self.assertIn("missing_api_key", decision.unavailable_reason_codes)
        self.assertIn("hitl_before_provider_provisioning", decision.required_hitl_gates)

    def test_missing_local_runtime_and_model_block_local_execution(self) -> None:
        plan = evaluate_adaptive_execution_policy(
            task_type="low_cost_execution",
            execution_mode_id="assisted_mode",
        )
        local_option = adaptive_execution_option_by_id(
            "adaptive_execution_option::local_to_local",
            plan,
        )

        self.assertIsNotNone(local_option)
        assert local_option is not None
        self.assertTrue(local_option.execution_blocked)
        self.assertIn(
            "local_runtime_unavailable", local_option.unavailable_reason_codes
        )
        self.assertIn(
            "local_model_not_installed", local_option.unavailable_reason_codes
        )
        self.assertIn(
            "insufficient_local_resources", local_option.unavailable_reason_codes
        )
        self.assertIn(
            "hitl_before_runtime_installation", local_option.required_hitl_gates
        )
        self.assertIn(
            "hitl_before_local_model_download", local_option.required_hitl_gates
        )
        for step in local_option.provider_model_path:
            if step.provider_id == "local":
                self.assertFalse(step.local_runtime_confirmed)
                self.assertFalse(step.local_model_confirmed)
                self.assertFalse(step.local_runtime_probe_implemented)
                self.assertFalse(step.local_model_inventory_scan_implemented)
                self.assertFalse(step.automatic_model_download_implemented)

    def test_intentionally_deferred_actions_are_manual_hitl_only(self) -> None:
        plan = evaluate_adaptive_execution_policy()

        self.assertEqual(
            tuple(action.action_kind for action in plan.manual_actions),
            REQUIRED_MANUAL_ACTIONS,
        )
        for action in plan.manual_actions:
            self.assertTrue(action.manual_action_required)
            self.assertTrue(action.hitl_required)
            self.assertTrue(action.execution_blocked_until_resolved)
            self.assertFalse(action.automatic_model_download_implemented)
            self.assertFalse(action.provider_provisioning_implemented)
            self.assertFalse(action.runtime_installation_implemented)
            self.assertFalse(action.runtime_evolution_implemented)
            self.assertFalse(action.provider_execution_implemented)

    def test_hybrid_policies_cover_required_directions_and_strategies(self) -> None:
        context = adaptive_execution_availability_context(
            configured_provider_ids=("openai",),
            supported_provider_ids=("openai", "local"),
            confirmed_local_runtime_kinds=("ollama",),
            installed_local_model_labels=("ollama chat model",),
            safe_auto_risk_bands=("low", "medium"),
        )
        plan = evaluate_adaptive_execution_policy(
            task_type="fast_draft",
            execution_mode_id="auto_mode",
            availability_context=context,
        )

        self.assertEqual(plan.hybrid_policy_directions, REQUIRED_HYBRID_DIRECTIONS)
        strategies = {
            strategy
            for policy in plan.hybrid_policies
            for strategy in policy.concrete_strategy_ids
        }
        self.assertIn("local_draft_to_cloud_final", strategies)
        self.assertIn("cloud_reasoning_to_local_variants", strategies)
        self.assertIn("cloud_provider_a_to_cloud_provider_b_fallback", strategies)
        self.assertIn("local_model_a_to_local_model_b_fallback", strategies)
        self.assertTrue(adaptive_execution_options_for_readiness("ready_now", plan))

    def test_task_aware_execution_covers_required_taxonomy(self) -> None:
        context = adaptive_execution_availability_context(
            configured_provider_ids=("openai",),
            supported_provider_ids=("openai", "local"),
            confirmed_local_runtime_kinds=("ollama",),
            installed_local_model_labels=("ollama chat model",),
            safe_auto_risk_bands=("low", "medium", "high"),
            safe_auto_cost_bands=("low", "medium", "high"),
            safe_auto_latency_bands=("fast", "moderate", "slow"),
        )

        for task_type in REQUIRED_TASK_TYPES:
            with self.subTest(task_type=task_type):
                plan = evaluate_adaptive_execution_policy(
                    task_type=task_type,
                    execution_mode_id="assisted_mode",
                    availability_context=context,
                )
                self.assertEqual(plan.task_types, REQUIRED_TASK_TYPES)
                self.assertEqual(plan.task_coverage_count, 12)
                self.assertEqual(plan.selected_decision.task_type, task_type)
                self.assertTrue(plan.execution_options)
                self.assertTrue(plan.selected_decision.recommended_strategy_id)

    def test_fallback_and_escalation_cover_required_failure_cases(self) -> None:
        plan = evaluate_adaptive_execution_policy(
            task_type="maximum_quality_execution",
            execution_mode_id="auto_mode",
        )

        fallback_reason_codes = {
            fallback.reason_code for fallback in plan.fallback_decisions
        }
        option_reasons = {
            reason
            for option in plan.execution_options
            for reason in option.unavailable_reason_codes
        }
        escalation_triggers = {rule.trigger_reason for rule in plan.escalation_rules}
        option_blocked_reasons = {
            reason
            for option in plan.execution_options
            for reason in option.blocked_reasons
        }
        self.assertIn("missing_api_key", option_reasons)
        self.assertIn("provider_unsupported", option_reasons)
        self.assertIn("local_runtime_unavailable", option_reasons)
        self.assertIn("local_model_not_installed", option_reasons)
        self.assertIn("insufficient_local_resources", option_reasons)
        self.assertIn("missing_modality_support", option_reasons)
        self.assertIn("high_cost_policy", escalation_triggers)
        self.assertIn("high_latency_policy", escalation_triggers)
        self.assertIn("low_expected_quality", escalation_triggers)
        self.assertIn("high_risk_execution_path", option_blocked_reasons)
        self.assertTrue(fallback_reason_codes)
        for fallback in plan.fallback_decisions:
            self.assertTrue(fallback.fallback_engine_implemented)
            self.assertFalse(fallback.provider_execution_implemented)
            self.assertFalse(fallback.workflow_execution_implemented)

    def test_provider_routing_output_and_workflow_boundaries_are_preserved(
        self,
    ) -> None:
        request = AssistantRequest(
            query="Build a p5.js sketch with a safe adaptive execution plan.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)

        plan = evaluate_adaptive_execution_policy(
            task_type="creative_coding",
            execution_mode_id="assisted_mode",
            settings=Settings(openai_api_key="sk-test-secret"),
        )
        next_decision = route_request(request)

        self.assertEqual(next_decision, baseline_decision)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)


if __name__ == "__main__":
    unittest.main()
