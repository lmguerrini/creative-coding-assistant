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
    RoutingLearningPlan,
    learn_routing,
    route_request,
    routing_learning_pattern_by_id,
    routing_learning_patterns_for_priority,
    routing_learning_patterns_for_status,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
REQUIRED_PATTERN_FIELDS = {
    "pattern_id",
    "pattern_kind",
    "status",
    "priority",
    "route_name",
    "task_type",
    "execution_mode_id",
    "source_routing_decision_id",
    "source_routing_task_type",
    "source_learning_signal_id",
    "source_workflow_risk_factor_id",
    "recommended_model_profile_id",
    "fallback_model_profile_id",
    "available_model_profile_ids",
    "routing_risk_band",
    "estimated_quality",
    "estimated_cost",
    "estimated_latency",
    "confidence_score",
    "unavailable_reason_count",
    "learning_priority_score",
    "routing_learning_weight",
    "routing_learning_score",
    "hitl_required",
    "routing_pattern_tags",
    "routing_summary",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "routing_learning_implemented",
    "routing_pattern_metadata_implemented",
    "task_aware_routing_metadata_used",
    "adaptive_learning_metadata_used",
    "routing_application_implemented",
    "provider_model_routing_implemented",
    "automatic_provider_switching_implemented",
    "automatic_model_switching_implemented",
    "automatic_model_download_implemented",
    "automatic_api_key_assumption_implemented",
    "provider_execution_implemented",
    "local_runtime_probe_implemented",
    "local_model_inventory_scan_implemented",
    "hitl_request_emitted",
    "workflow_control_implemented",
    "workflow_graph_mutation_implemented",
    "graph_compilation_implemented",
    "workflow_execution_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "advisory_only",
}


class RoutingLearningTests(unittest.TestCase):
    def test_plan_derives_routing_patterns_from_task_routing_metadata(self) -> None:
        plan = learn_routing(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "routing_learning")
        self.assertEqual(plan.serialization_version, "routing_learning_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.source_task_routing_serialization_version,
            "task_aware_routing_registry.v1",
        )
        self.assertEqual(
            plan.source_adaptive_learning_serialization_version,
            "adaptive_learning_plan.v1",
        )
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(
            plan.recommended_model_profile_ids,
            (
                "creative_reasoning_model_profile",
                "evaluation_review_model_profile",
                "fast_iteration_model_profile",
            ),
        )
        self.assertEqual(
            plan.fallback_model_profile_ids,
            (
                "code_assistance_model_profile",
                "creative_reasoning_model_profile",
                "evaluation_review_model_profile",
            ),
        )
        self.assertEqual(plan.pattern_count, 4)
        self.assertEqual(plan.review_required_pattern_count, 2)
        self.assertEqual(plan.guarded_pattern_count, 2)
        self.assertEqual(plan.hitl_required_pattern_count, 4)
        self.assertFalse(plan.applied_routing_pattern_ids)
        self.assertEqual(plan.overall_routing_learning_posture, "guarded")
        self.assertIn("does not apply routing", plan.authority_boundary)
        self.assertTrue(plan.routing_learning_implemented)
        self.assertTrue(plan.routing_pattern_metadata_implemented)
        self.assertTrue(plan.task_aware_routing_metadata_used)
        self.assertTrue(plan.adaptive_learning_metadata_used)
        self.assertFalse(plan.routing_application_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.automatic_provider_switching_implemented)
        self.assertFalse(plan.automatic_model_switching_implemented)
        self.assertFalse(plan.automatic_model_download_implemented)
        self.assertFalse(plan.automatic_api_key_assumption_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.local_runtime_probe_implemented)
        self.assertFalse(plan.local_model_inventory_scan_implemented)
        self.assertFalse(plan.hitl_request_emitted)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.graph_compilation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_patterns_score_routing_learning_without_route_application(self) -> None:
        plan = learn_routing(route="generate")

        quality_bonus = {"low": 20, "medium": 100, "high": 170, "maximum": 240}
        cost_bonus = {"low": 120, "medium": 60, "high": 0}
        latency_bonus = {"fast": 120, "moderate": 60, "slow": 0}
        risk_penalty = {"low": 0, "medium": 80, "high": 180}
        for pattern in plan.patterns:
            dumped = pattern.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PATTERN_FIELDS)
            self.assertEqual(
                pattern.serialization_version,
                "routing_learning_pattern.v1",
            )
            self.assertEqual(pattern.route_name, RouteName.GENERATE)
            self.assertEqual(
                pattern.pattern_id,
                f"routing_learning::{pattern.pattern_kind}",
            )
            self.assertEqual(
                pattern.routing_learning_score,
                min(
                    1000,
                    max(
                        0,
                        int(pattern.confidence_score * 500)
                        + quality_bonus[pattern.estimated_quality]
                        + cost_bonus[pattern.estimated_cost]
                        + latency_bonus[pattern.estimated_latency]
                        + pattern.learning_priority_score // 3
                        + pattern.routing_learning_weight
                        - risk_penalty[pattern.routing_risk_band]
                        - pattern.unavailable_reason_count * 60,
                    ),
                ),
            )
            self.assertIn("routing_application", pattern.blocked_runtime_behaviors)
            self.assertIn(
                pattern.recommended_model_profile_id,
                pattern.available_model_profile_ids,
            )
            self.assertIn(
                pattern.fallback_model_profile_id,
                pattern.available_model_profile_ids,
            )
            self.assertTrue(pattern.routing_pattern_tags)
            self.assertTrue(pattern.advisory_actions)
            self.assertTrue(pattern.evidence)
            self.assertTrue(pattern.hitl_required)
            self.assertTrue(pattern.routing_learning_implemented)
            self.assertTrue(pattern.task_aware_routing_metadata_used)
            self.assertTrue(pattern.adaptive_learning_metadata_used)
            self.assertFalse(pattern.routing_application_implemented)
            self.assertFalse(pattern.provider_model_routing_implemented)
            self.assertFalse(pattern.automatic_provider_switching_implemented)
            self.assertFalse(pattern.automatic_model_switching_implemented)
            self.assertFalse(pattern.automatic_model_download_implemented)
            self.assertFalse(pattern.automatic_api_key_assumption_implemented)
            self.assertFalse(pattern.provider_execution_implemented)
            self.assertFalse(pattern.local_runtime_probe_implemented)
            self.assertFalse(pattern.local_model_inventory_scan_implemented)
            self.assertFalse(pattern.workflow_control_implemented)
            self.assertFalse(pattern.workflow_graph_mutation_implemented)
            self.assertFalse(pattern.graph_compilation_implemented)
            self.assertFalse(pattern.workflow_execution_implemented)
            self.assertFalse(pattern.persistent_storage_write_implemented)
            self.assertFalse(pattern.generated_output_mutation_implemented)
            self.assertFalse(pattern.runtime_evolution_implemented)
            self.assertTrue(pattern.advisory_only)

        guarded = routing_learning_pattern_by_id(
            "routing_learning::guarded_route_learning",
            plan,
        )
        critical = routing_learning_patterns_for_priority("critical", plan)
        review = routing_learning_patterns_for_status("review_required", plan)
        self.assertIsNotNone(guarded)
        assert guarded is not None
        self.assertEqual(guarded.status, "guarded")
        self.assertEqual(guarded.priority, "guarded")
        self.assertEqual(len(critical), 2)
        self.assertEqual(len(review), 2)

    def test_plan_rejects_mismatched_routing_learning_metadata(self) -> None:
        plan = learn_routing()
        payload = plan.model_dump(mode="json")
        payload["pattern_ids"] = ("missing",) + tuple(payload["pattern_ids"][1:])

        with self.assertRaisesRegex(ValueError, "pattern_ids must match"):
            RoutingLearningPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_routing_learning_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_routing_learning_score must match",
        ):
            RoutingLearningPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["applied_routing_pattern_ids"] = (plan.pattern_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "applied_routing_pattern_ids must remain empty",
        ):
            RoutingLearningPlan(**payload)

    def test_learning_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Review routing learning signals for a creative workflow.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = learn_routing(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_learning_does_not_declare_routing_application_terms(self) -> None:
        plan = learn_routing(route=RouteName.GENERATE)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for pattern in plan.patterns
                    for field in (
                        pattern.pattern_id,
                        pattern.pattern_kind,
                        pattern.source_routing_decision_id,
                        pattern.source_routing_task_type,
                        pattern.source_learning_signal_id,
                        pattern.source_workflow_risk_factor_id,
                        pattern.recommended_model_profile_id,
                        pattern.fallback_model_profile_id,
                        pattern.routing_risk_band,
                        pattern.estimated_quality,
                        pattern.estimated_cost,
                        pattern.estimated_latency,
                        *pattern.routing_pattern_tags,
                        pattern.routing_summary,
                        *pattern.advisory_actions,
                        *pattern.evidence,
                        *pattern.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "apply_routing(",
            "route_provider(",
            "switch_provider(",
            "switch_model(",
            "download_model(",
            "assume_api_key(",
            "execute_provider(",
            "probe_local_runtime(",
            "scan_local_model_inventory(",
            "emit_hitl_request(",
            "control_workflow(",
            "mutate_workflow_graph(",
            "compile_graph(",
            "execute_workflow(",
            "trigger_retry(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
            "apply_runtime_evolution(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
