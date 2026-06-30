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
    RuntimeLearningPlan,
    learn_runtimes,
    route_request,
    runtime_learning_pattern_by_id,
    runtime_learning_patterns_for_priority,
    runtime_learning_patterns_for_status,
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
    "source_runtime_role",
    "source_learning_signal_id",
    "source_workflow_risk_factor_id",
    "runtime_id",
    "runtime_suitability",
    "runtime_confidence",
    "implementation_complexity",
    "performance_pressure",
    "preview_support",
    "learning_priority_score",
    "runtime_learning_weight",
    "runtime_learning_score",
    "hitl_required",
    "runtime_pattern_tags",
    "runtime_summary",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "runtime_learning_implemented",
    "runtime_pattern_metadata_implemented",
    "runtime_capability_metadata_used",
    "adaptive_learning_metadata_used",
    "runtime_selection_implemented",
    "execution_profile_creation_implemented",
    "local_runtime_probe_implemented",
    "runtime_installation_implemented",
    "automatic_model_download_implemented",
    "provider_model_routing_implemented",
    "automatic_provider_switching_implemented",
    "automatic_model_switching_implemented",
    "provider_execution_implemented",
    "artifact_execution_implemented",
    "preview_behavior_change_implemented",
    "hitl_request_emitted",
    "workflow_control_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_execution_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "advisory_only",
}


class RuntimeLearningTests(unittest.TestCase):
    def test_plan_derives_runtime_patterns_from_capability_metadata(self) -> None:
        plan = learn_runtimes(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "runtime_learning")
        self.assertEqual(plan.serialization_version, "runtime_learning_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.source_runtime_capability_role,
            "runtime_capability_reasoner",
        )
        self.assertEqual(
            plan.source_adaptive_learning_serialization_version,
            "adaptive_learning_plan.v1",
        )
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.likely_runtime_ids, ("p5_js", "three_js", "tone_js"))
        self.assertEqual(plan.pattern_count, 4)
        self.assertEqual(plan.review_required_pattern_count, 3)
        self.assertEqual(plan.guarded_pattern_count, 1)
        self.assertEqual(plan.hitl_required_pattern_count, 4)
        self.assertFalse(plan.applied_runtime_pattern_ids)
        self.assertEqual(plan.overall_runtime_learning_posture, "guarded")
        self.assertIn("does not select runtimes", plan.authority_boundary)
        self.assertTrue(plan.runtime_learning_implemented)
        self.assertTrue(plan.runtime_pattern_metadata_implemented)
        self.assertTrue(plan.runtime_capability_metadata_used)
        self.assertTrue(plan.adaptive_learning_metadata_used)
        self.assertFalse(plan.runtime_selection_implemented)
        self.assertFalse(plan.execution_profile_creation_implemented)
        self.assertFalse(plan.local_runtime_probe_implemented)
        self.assertFalse(plan.runtime_installation_implemented)
        self.assertFalse(plan.automatic_model_download_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.artifact_execution_implemented)
        self.assertFalse(plan.preview_behavior_change_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_patterns_score_runtime_learning_without_runtime_selection(self) -> None:
        plan = learn_runtimes(route="generate")

        fit_bonus = {"strong": 160, "moderate": 90, "weak": 20}
        pressure_penalty = {"low": 0, "medium": 40, "high": 100}
        for pattern in plan.patterns:
            dumped = pattern.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PATTERN_FIELDS)
            self.assertEqual(
                pattern.serialization_version,
                "runtime_learning_pattern.v1",
            )
            self.assertEqual(pattern.route_name, RouteName.GENERATE)
            self.assertEqual(
                pattern.pattern_id,
                f"runtime_learning::{pattern.pattern_kind}",
            )
            self.assertEqual(
                pattern.runtime_learning_score,
                min(
                    1000,
                    max(
                        0,
                        int(pattern.runtime_confidence * 520)
                        + fit_bonus[pattern.runtime_suitability]
                        + pattern.learning_priority_score // 3
                        + pattern.runtime_learning_weight
                        - pressure_penalty[pattern.implementation_complexity]
                        - pressure_penalty[pattern.performance_pressure],
                    ),
                ),
            )
            self.assertIn("runtime_selection", pattern.blocked_runtime_behaviors)
            self.assertTrue(pattern.runtime_pattern_tags)
            self.assertTrue(pattern.advisory_actions)
            self.assertTrue(pattern.evidence)
            self.assertTrue(pattern.hitl_required)
            self.assertTrue(pattern.runtime_learning_implemented)
            self.assertTrue(pattern.runtime_capability_metadata_used)
            self.assertTrue(pattern.adaptive_learning_metadata_used)
            self.assertFalse(pattern.runtime_selection_implemented)
            self.assertFalse(pattern.execution_profile_creation_implemented)
            self.assertFalse(pattern.local_runtime_probe_implemented)
            self.assertFalse(pattern.runtime_installation_implemented)
            self.assertFalse(pattern.provider_model_routing_implemented)
            self.assertFalse(pattern.provider_execution_implemented)
            self.assertFalse(pattern.artifact_execution_implemented)
            self.assertFalse(pattern.preview_behavior_change_implemented)
            self.assertFalse(pattern.workflow_control_implemented)
            self.assertFalse(pattern.workflow_graph_mutation_implemented)
            self.assertFalse(pattern.workflow_execution_implemented)
            self.assertFalse(pattern.persistent_storage_write_implemented)
            self.assertFalse(pattern.generated_output_mutation_implemented)
            self.assertFalse(pattern.runtime_evolution_implemented)
            self.assertTrue(pattern.advisory_only)

        shader = runtime_learning_pattern_by_id(
            "runtime_learning::shader_runtime_learning",
            plan,
        )
        critical = runtime_learning_patterns_for_priority("critical", plan)
        review = runtime_learning_patterns_for_status("review_required", plan)
        self.assertIsNotNone(shader)
        assert shader is not None
        self.assertEqual(shader.status, "guarded")
        self.assertEqual(shader.priority, "guarded")
        self.assertEqual(len(critical), 1)
        self.assertEqual(len(review), 3)

    def test_plan_rejects_mismatched_runtime_learning_metadata(self) -> None:
        plan = learn_runtimes()
        payload = plan.model_dump(mode="json")
        payload["pattern_ids"] = ("missing",) + tuple(payload["pattern_ids"][1:])

        with self.assertRaisesRegex(ValueError, "pattern_ids must match"):
            RuntimeLearningPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_runtime_learning_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_runtime_learning_score must match",
        ):
            RuntimeLearningPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["applied_runtime_pattern_ids"] = (plan.pattern_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "applied_runtime_pattern_ids must remain empty",
        ):
            RuntimeLearningPlan(**payload)

    def test_learning_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Review runtime learning signals for a creative workflow.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = learn_runtimes(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_learning_does_not_declare_runtime_application_terms(self) -> None:
        plan = learn_runtimes(route=RouteName.GENERATE)
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
                        pattern.source_runtime_role,
                        pattern.source_learning_signal_id,
                        pattern.source_workflow_risk_factor_id,
                        pattern.runtime_id,
                        pattern.runtime_suitability,
                        pattern.implementation_complexity,
                        pattern.performance_pressure,
                        pattern.preview_support,
                        *pattern.runtime_pattern_tags,
                        pattern.runtime_summary,
                        *pattern.advisory_actions,
                        *pattern.evidence,
                        *pattern.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "select_runtime(",
            "create_execution_profile(",
            "probe_local_runtime(",
            "install_runtime(",
            "download_model(",
            "route_provider(",
            "switch_provider(",
            "switch_model(",
            "execute_provider(",
            "execute_artifact(",
            "change_preview(",
            "emit_hitl_request(",
            "control_workflow(",
            "mutate_workflow_graph(",
            "execute_workflow(",
            "write_storage(",
            "modify_output(",
            "apply_runtime_evolution(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
