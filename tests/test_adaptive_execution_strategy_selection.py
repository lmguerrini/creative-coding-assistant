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
    AdaptiveExecutionStrategySelectionPlan,
    adaptive_execution_strategies_for_status,
    adaptive_execution_strategy_by_id,
    route_request,
    select_dynamic_execution_strategy,
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
    "strategy_id",
    "strategy_kind",
    "status",
    "route_name",
    "task_type",
    "execution_mode_id",
    "source_execution_strategy_id",
    "source_hybrid_workflow_candidate_id",
    "source_escalation_posture",
    "source_agent_activation_candidate_ids",
    "source_cost_quality_candidate_id",
    "source_latency_candidate_id",
    "policy_direction",
    "provider_sequence",
    "model_profile_sequence",
    "estimated_quality",
    "estimated_cost",
    "estimated_latency",
    "unavailable_reason_codes",
    "source_execution_strategy_score",
    "hybrid_adaptive_score",
    "cost_quality_score",
    "latency_score",
    "agent_activation_score",
    "escalation_pressure_score",
    "strategy_bias",
    "dynamic_strategy_score",
    "hitl_required",
    "fallback_strategy_id",
    "fallback_reason_summary",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "dynamic_execution_strategy_selection_implemented",
    "adaptive_execution_intelligence_implemented",
    "provider_intelligence_metadata_used",
    "availability_awareness_metadata_used",
    "execution_simulation_metadata_used",
    "fallback_intelligence_implemented",
    "strategy_application_implemented",
    "automatic_provider_switching_implemented",
    "automatic_model_switching_implemented",
    "automatic_model_download_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "local_runtime_probe_implemented",
    "local_model_inventory_scan_implemented",
    "agent_invocation_implemented",
    "hitl_request_emitted",
    "budget_enforcement_implemented",
    "workflow_control_implemented",
    "workflow_graph_mutation_implemented",
    "graph_compilation_implemented",
    "workflow_execution_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}


class AdaptiveExecutionStrategySelectionTests(unittest.TestCase):
    def test_plan_combines_dynamic_execution_strategy_sources(self) -> None:
        plan = select_dynamic_execution_strategy(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "dynamic_execution_strategy_selector")
        self.assertEqual(
            plan.serialization_version,
            "adaptive_execution_strategy_selection_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.source_execution_strategy_serialization_version,
            "execution_strategy_selection.v1",
        )
        self.assertEqual(
            plan.source_hybrid_workflow_serialization_version,
            "adaptive_hybrid_workflow_optimization_plan.v1",
        )
        self.assertEqual(
            plan.source_escalation_optimization_serialization_version,
            "adaptive_escalation_optimization_plan.v1",
        )
        self.assertEqual(
            plan.source_agent_activation_serialization_version,
            "agent_activation_optimization_plan.v1",
        )
        self.assertEqual(
            plan.source_adaptive_cost_quality_serialization_version,
            "adaptive_cost_quality_plan.v1",
        )
        self.assertEqual(
            plan.source_adaptive_latency_serialization_version,
            "adaptive_latency_plan.v1",
        )
        self.assertEqual(plan.provider_ids, REQUIRED_PROVIDER_IDS)
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.hybrid_policy_directions, REQUIRED_HYBRID_DIRECTIONS)
        self.assertEqual(plan.strategy_count, 4)
        self.assertEqual(plan.selected_strategy_count, 1)
        self.assertIsNone(plan.applied_strategy_id)
        self.assertIn(plan.selected_strategy_id, plan.strategy_ids)
        self.assertIn("does not apply strategies", plan.authority_boundary)
        self.assertTrue(plan.dynamic_execution_strategy_selection_implemented)
        self.assertTrue(plan.adaptive_execution_intelligence_implemented)
        self.assertTrue(plan.provider_intelligence_metadata_used)
        self.assertTrue(plan.availability_awareness_metadata_used)
        self.assertTrue(plan.execution_simulation_metadata_used)
        self.assertTrue(plan.fallback_intelligence_implemented)
        self.assertFalse(plan.strategy_application_implemented)
        self.assertFalse(plan.automatic_provider_switching_implemented)
        self.assertFalse(plan.automatic_model_switching_implemented)
        self.assertFalse(plan.automatic_model_download_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.local_runtime_probe_implemented)
        self.assertFalse(plan.local_model_inventory_scan_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.hitl_request_emitted)
        self.assertFalse(plan.budget_enforcement_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.advisory_only)

    def test_strategies_are_relative_metadata_selections(self) -> None:
        plan = select_dynamic_execution_strategy(route="generate")

        for strategy in plan.strategies:
            dumped = strategy.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_CANDIDATE_FIELDS)
            self.assertEqual(
                strategy.serialization_version,
                "adaptive_execution_strategy_candidate.v1",
            )
            self.assertEqual(strategy.route_name, RouteName.GENERATE)
            self.assertGreaterEqual(strategy.dynamic_strategy_score, 0)
            self.assertLessEqual(strategy.dynamic_strategy_score, 400)
            self.assertEqual(
                strategy.dynamic_strategy_score,
                min(
                    400,
                    max(
                        0,
                        strategy.source_execution_strategy_score
                        + strategy.hybrid_adaptive_score
                        + strategy.cost_quality_score // 2
                        + strategy.latency_score // 2
                        + strategy.agent_activation_score // 4
                        - strategy.escalation_pressure_score // 5
                        + strategy.strategy_bias,
                    ),
                ),
            )
            self.assertIn(
                "silent_provider_or_model_routing_change",
                strategy.blocked_runtime_behaviors,
            )
            self.assertTrue(strategy.dynamic_execution_strategy_selection_implemented)
            self.assertTrue(strategy.provider_intelligence_metadata_used)
            self.assertTrue(strategy.availability_awareness_metadata_used)
            self.assertTrue(strategy.execution_simulation_metadata_used)
            self.assertFalse(strategy.strategy_application_implemented)
            self.assertFalse(strategy.automatic_provider_switching_implemented)
            self.assertFalse(strategy.automatic_model_switching_implemented)
            self.assertFalse(strategy.automatic_model_download_implemented)
            self.assertFalse(strategy.provider_model_routing_implemented)
            self.assertFalse(strategy.provider_execution_implemented)
            self.assertFalse(strategy.local_runtime_probe_implemented)
            self.assertFalse(strategy.local_model_inventory_scan_implemented)
            self.assertFalse(strategy.agent_invocation_implemented)
            self.assertFalse(strategy.hitl_request_emitted)
            self.assertFalse(strategy.budget_enforcement_implemented)
            self.assertFalse(strategy.workflow_control_implemented)
            self.assertFalse(strategy.workflow_graph_mutation_implemented)
            self.assertFalse(strategy.workflow_execution_implemented)
            self.assertFalse(strategy.retry_triggering_implemented)
            self.assertFalse(strategy.generated_output_mutation_implemented)
            self.assertTrue(strategy.advisory_only)

        selected = adaptive_execution_strategy_by_id(plan.selected_strategy_id, plan)
        guardrails = adaptive_execution_strategies_for_status("guardrail", plan)
        fallbacks = adaptive_execution_strategies_for_status("fallback", plan)
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected.status, "selected")
        self.assertEqual(selected.provider_sequence, plan.selected_provider_sequence)
        self.assertEqual(
            selected.model_profile_sequence, plan.selected_model_profile_sequence
        )
        self.assertTrue(selected.hitl_required)
        self.assertEqual(len(guardrails), 1)
        self.assertTrue(fallbacks)
        self.assertIn("missing_api_key", selected.unavailable_reason_codes)

    def test_plan_rejects_mismatched_strategy_metadata(self) -> None:
        plan = select_dynamic_execution_strategy()
        payload = plan.model_dump(mode="json")
        payload["strategy_ids"] = ("missing",) + tuple(payload["strategy_ids"][1:])

        with self.assertRaisesRegex(ValueError, "strategy_ids must match"):
            AdaptiveExecutionStrategySelectionPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["selected_strategy_id"] = "missing"

        with self.assertRaisesRegex(ValueError, "selected_strategy_id must match"):
            AdaptiveExecutionStrategySelectionPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["selected_strategy_score"] += 1

        with self.assertRaisesRegex(ValueError, "selected_strategy_score must match"):
            AdaptiveExecutionStrategySelectionPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["applied_strategy_id"] = plan.selected_strategy_id

        with self.assertRaisesRegex(
            ValueError, "applied_strategy_id must remain unset"
        ):
            AdaptiveExecutionStrategySelectionPlan(**payload)

    def test_selector_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Choose an execution strategy for a local and cloud sketch workflow.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = select_dynamic_execution_strategy(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_selector_does_not_declare_runtime_application_terms(self) -> None:
        plan = select_dynamic_execution_strategy(route=RouteName.GENERATE)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for strategy in plan.strategies
                    for field in (
                        strategy.strategy_id,
                        strategy.strategy_kind,
                        strategy.source_execution_strategy_id,
                        strategy.source_hybrid_workflow_candidate_id,
                        strategy.source_cost_quality_candidate_id,
                        strategy.source_latency_candidate_id,
                        *strategy.source_agent_activation_candidate_ids,
                        *strategy.provider_sequence,
                        *strategy.model_profile_sequence,
                        strategy.fallback_reason_summary,
                        *strategy.advisory_actions,
                        *strategy.evidence,
                        *strategy.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "apply_strategy(",
            "switch_provider(",
            "switch_model(",
            "download_model(",
            "route_provider(",
            "execute_provider(",
            "probe_local_runtime(",
            "scan_local_models(",
            "invoke_agent(",
            "emit_hitl_request(",
            "enforce_budget(",
            "control_workflow(",
            "mutate_workflow_graph(",
            "compile_graph(",
            "execute_workflow(",
            "trigger_retry(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
