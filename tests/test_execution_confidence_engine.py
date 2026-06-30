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
    ExecutionConfidencePlan,
    evaluate_execution_confidence,
    execution_confidence_signal_by_id,
    execution_confidence_signals_for_band,
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
REQUIRED_SIGNAL_FIELDS = {
    "signal_id",
    "signal_kind",
    "status",
    "confidence_band",
    "route_name",
    "task_type",
    "execution_mode_id",
    "source_dynamic_strategy_id",
    "source_dynamic_agent_allocation_ids",
    "source_dynamic_resource_allocation_id",
    "source_self_tuning_policy_id",
    "source_cost_quality_candidate_id",
    "source_latency_candidate_id",
    "provider_sequence",
    "model_profile_sequence",
    "hybrid_policy_direction",
    "unavailable_reason_codes",
    "dynamic_strategy_score",
    "agent_allocation_score",
    "dynamic_resource_score",
    "self_tuning_score",
    "adaptive_cost_quality_score",
    "adaptive_latency_score",
    "availability_penalty",
    "guardrail_penalty",
    "execution_confidence_score",
    "hitl_required",
    "fallback_safety_summary",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "execution_confidence_engine_implemented",
    "execution_confidence_metadata_implemented",
    "provider_intelligence_metadata_used",
    "availability_awareness_metadata_used",
    "manual_assisted_auto_mode_metadata_used",
    "hybrid_transition_metadata_used",
    "task_aware_category_metadata_used",
    "execution_simulation_metadata_used",
    "fallback_safety_metadata_used",
    "confidence_application_implemented",
    "generated_output_evaluation_implemented",
    "provider_model_routing_implemented",
    "automatic_provider_switching_implemented",
    "automatic_model_switching_implemented",
    "provider_execution_implemented",
    "local_runtime_probe_implemented",
    "local_model_inventory_scan_implemented",
    "automatic_model_download_implemented",
    "workflow_control_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_execution_implemented",
    "agent_invocation_implemented",
    "resource_allocation_implemented",
    "self_tuning_application_implemented",
    "hitl_request_emitted",
    "budget_enforcement_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "advisory_only",
}


class ExecutionConfidenceEngineTests(unittest.TestCase):
    def test_plan_combines_execution_confidence_sources(self) -> None:
        plan = evaluate_execution_confidence(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "execution_confidence_engine")
        self.assertEqual(plan.serialization_version, "execution_confidence_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.source_dynamic_execution_strategy_serialization_version,
            "adaptive_execution_strategy_selection_plan.v1",
        )
        self.assertEqual(
            plan.source_dynamic_agent_allocation_serialization_version,
            "dynamic_agent_allocation_plan.v1",
        )
        self.assertEqual(
            plan.source_dynamic_resource_allocation_serialization_version,
            "dynamic_resource_allocation_plan.v1",
        )
        self.assertEqual(
            plan.source_workflow_self_tuning_serialization_version,
            "workflow_self_tuning_policy_plan.v1",
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
        self.assertEqual(plan.signal_count, 6)
        self.assertEqual(plan.review_required_signal_count, 4)
        self.assertEqual(plan.guardrail_signal_count, 2)
        self.assertEqual(plan.hitl_required_signal_count, 6)
        self.assertFalse(plan.applied_confidence_signal_ids)
        self.assertEqual(plan.overall_confidence_band, "guarded")
        self.assertIn("does not apply confidence decisions", plan.authority_boundary)
        self.assertTrue(plan.execution_confidence_engine_implemented)
        self.assertTrue(plan.execution_confidence_metadata_implemented)
        self.assertTrue(plan.provider_intelligence_metadata_used)
        self.assertTrue(plan.availability_awareness_metadata_used)
        self.assertTrue(plan.manual_assisted_auto_mode_metadata_used)
        self.assertTrue(plan.hybrid_transition_metadata_used)
        self.assertTrue(plan.task_aware_category_metadata_used)
        self.assertTrue(plan.execution_simulation_metadata_used)
        self.assertTrue(plan.fallback_safety_metadata_used)
        self.assertFalse(plan.confidence_application_implemented)
        self.assertFalse(plan.generated_output_evaluation_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.automatic_provider_switching_implemented)
        self.assertFalse(plan.automatic_model_switching_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.local_runtime_probe_implemented)
        self.assertFalse(plan.local_model_inventory_scan_implemented)
        self.assertFalse(plan.automatic_model_download_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.resource_allocation_implemented)
        self.assertFalse(plan.self_tuning_application_implemented)
        self.assertFalse(plan.hitl_request_emitted)
        self.assertFalse(plan.budget_enforcement_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_signals_score_execution_confidence_without_applying(self) -> None:
        plan = evaluate_execution_confidence(route="generate")

        for signal in plan.signals:
            dumped = signal.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_SIGNAL_FIELDS)
            self.assertEqual(
                signal.serialization_version,
                "execution_confidence_signal.v1",
            )
            self.assertEqual(signal.route_name, RouteName.GENERATE)
            self.assertEqual(
                signal.signal_id,
                f"execution_confidence::{signal.signal_kind}",
            )
            self.assertEqual(
                signal.execution_confidence_score,
                min(
                    100,
                    max(
                        0,
                        signal.dynamic_strategy_score // 12
                        + signal.agent_allocation_score // 12
                        + signal.dynamic_resource_score // 20
                        + signal.self_tuning_score // 20
                        + signal.adaptive_cost_quality_score // 16
                        + signal.adaptive_latency_score // 12
                        - signal.availability_penalty
                        - signal.guardrail_penalty,
                    ),
                ),
            )
            self.assertIn("confidence_application", signal.blocked_runtime_behaviors)
            self.assertTrue(signal.execution_confidence_engine_implemented)
            self.assertTrue(signal.provider_intelligence_metadata_used)
            self.assertTrue(signal.availability_awareness_metadata_used)
            self.assertTrue(signal.manual_assisted_auto_mode_metadata_used)
            self.assertTrue(signal.hybrid_transition_metadata_used)
            self.assertTrue(signal.task_aware_category_metadata_used)
            self.assertTrue(signal.execution_simulation_metadata_used)
            self.assertTrue(signal.fallback_safety_metadata_used)
            self.assertFalse(signal.confidence_application_implemented)
            self.assertFalse(signal.generated_output_evaluation_implemented)
            self.assertFalse(signal.provider_model_routing_implemented)
            self.assertFalse(signal.automatic_provider_switching_implemented)
            self.assertFalse(signal.automatic_model_switching_implemented)
            self.assertFalse(signal.provider_execution_implemented)
            self.assertFalse(signal.local_runtime_probe_implemented)
            self.assertFalse(signal.local_model_inventory_scan_implemented)
            self.assertFalse(signal.automatic_model_download_implemented)
            self.assertFalse(signal.workflow_control_implemented)
            self.assertFalse(signal.workflow_graph_mutation_implemented)
            self.assertFalse(signal.workflow_execution_implemented)
            self.assertFalse(signal.agent_invocation_implemented)
            self.assertFalse(signal.resource_allocation_implemented)
            self.assertFalse(signal.self_tuning_application_implemented)
            self.assertFalse(signal.hitl_request_emitted)
            self.assertFalse(signal.budget_enforcement_implemented)
            self.assertFalse(signal.retry_triggering_implemented)
            self.assertFalse(signal.generated_output_mutation_implemented)
            self.assertTrue(signal.hitl_required)
            self.assertTrue(signal.advisory_only)

        provider = execution_confidence_signal_by_id(
            "execution_confidence::provider_availability_confidence",
            plan,
        )
        guarded = execution_confidence_signals_for_band("guarded", plan)
        self.assertIsNotNone(provider)
        assert provider is not None
        self.assertEqual(provider.status, "guardrail")
        self.assertEqual(provider.confidence_band, "guarded")
        self.assertIn("missing_api_key", provider.unavailable_reason_codes)
        self.assertEqual(provider.provider_sequence, ("local", "openai"))
        self.assertEqual(len(guarded), 2)

    def test_plan_rejects_mismatched_confidence_metadata(self) -> None:
        plan = evaluate_execution_confidence()
        payload = plan.model_dump(mode="json")
        payload["signal_ids"] = ("missing",) + tuple(payload["signal_ids"][1:])

        with self.assertRaisesRegex(ValueError, "signal_ids must match"):
            ExecutionConfidencePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_confidence_score"] += 1

        with self.assertRaisesRegex(ValueError, "overall_confidence_score must match"):
            ExecutionConfidencePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["applied_confidence_signal_ids"] = (plan.signal_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "applied_confidence_signal_ids must remain empty",
        ):
            ExecutionConfidencePlan(**payload)

    def test_engine_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Evaluate execution confidence for a hybrid coding workflow.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = evaluate_execution_confidence(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_engine_does_not_declare_runtime_application_terms(self) -> None:
        plan = evaluate_execution_confidence(route=RouteName.GENERATE)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for signal in plan.signals
                    for field in (
                        signal.signal_id,
                        signal.signal_kind,
                        signal.source_dynamic_strategy_id,
                        signal.source_dynamic_resource_allocation_id,
                        signal.source_self_tuning_policy_id,
                        signal.source_cost_quality_candidate_id,
                        signal.source_latency_candidate_id,
                        *signal.source_dynamic_agent_allocation_ids,
                        *signal.provider_sequence,
                        *signal.model_profile_sequence,
                        signal.fallback_safety_summary,
                        *signal.advisory_actions,
                        *signal.evidence,
                        *signal.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "apply_confidence(",
            "evaluate_output(",
            "route_provider(",
            "switch_provider(",
            "switch_model(",
            "execute_provider(",
            "probe_local_runtime(",
            "scan_local_models(",
            "download_model(",
            "invoke_agent(",
            "allocate_resource(",
            "apply_self_tuning(",
            "emit_hitl_request(",
            "enforce_budget(",
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
