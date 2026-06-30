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
    ArtifactLearningPlan,
    artifact_learning_pattern_by_id,
    artifact_learning_patterns_for_priority,
    artifact_learning_patterns_for_status,
    learn_artifacts,
    route_request,
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
    "source_artifact_plan_role",
    "source_artifact_capability_role",
    "source_learning_signal_id",
    "source_workflow_risk_factor_id",
    "artifact_type",
    "artifact_family",
    "target_runtime_id",
    "artifact_fit",
    "creative_fit",
    "generative_fit",
    "interaction_fit",
    "audiovisual_fit",
    "export_fit",
    "interoperability_fit",
    "portability_fit",
    "capability_confidence",
    "missing_information_count",
    "hitl_question_count",
    "capability_risk_count",
    "learning_priority_score",
    "artifact_learning_weight",
    "artifact_learning_score",
    "hitl_required",
    "artifact_pattern_tags",
    "artifact_summary",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "artifact_learning_implemented",
    "artifact_pattern_metadata_implemented",
    "artifact_planning_metadata_used",
    "artifact_capability_metadata_used",
    "adaptive_learning_metadata_used",
    "artifact_selection_implemented",
    "artifact_mutation_implemented",
    "artifact_generation_implemented",
    "artifact_execution_implemented",
    "runtime_execution_implemented",
    "preview_behavior_change_implemented",
    "artifact_merge_implemented",
    "artifact_export_implemented",
    "provider_model_routing_implemented",
    "automatic_provider_switching_implemented",
    "automatic_model_switching_implemented",
    "provider_execution_implemented",
    "local_runtime_probe_implemented",
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


class ArtifactLearningTests(unittest.TestCase):
    def test_plan_derives_artifact_patterns_from_planning_metadata(self) -> None:
        plan = learn_artifacts(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "artifact_learning")
        self.assertEqual(plan.serialization_version, "artifact_learning_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.source_artifact_plan_role, "artifact_planner")
        self.assertEqual(
            plan.source_artifact_capability_role,
            "artifact_capability_matrix",
        )
        self.assertEqual(
            plan.source_adaptive_learning_serialization_version,
            "adaptive_learning_plan.v1",
        )
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.artifact_family, "p5_sketch")
        self.assertEqual(plan.strongest_target_runtime_ids, ("p5_js", "canvas", "svg"))
        self.assertEqual(plan.weakest_target_runtime_ids, ("hydra", "glsl", "gsap"))
        self.assertEqual(plan.pattern_count, 4)
        self.assertEqual(plan.review_required_pattern_count, 3)
        self.assertEqual(plan.guarded_pattern_count, 1)
        self.assertEqual(plan.hitl_required_pattern_count, 4)
        self.assertFalse(plan.applied_artifact_pattern_ids)
        self.assertEqual(plan.overall_artifact_learning_posture, "guarded")
        self.assertIn("does not select artifacts", plan.authority_boundary)
        self.assertTrue(plan.artifact_learning_implemented)
        self.assertTrue(plan.artifact_pattern_metadata_implemented)
        self.assertTrue(plan.artifact_planning_metadata_used)
        self.assertTrue(plan.artifact_capability_metadata_used)
        self.assertTrue(plan.adaptive_learning_metadata_used)
        self.assertFalse(plan.artifact_selection_implemented)
        self.assertFalse(plan.artifact_mutation_implemented)
        self.assertFalse(plan.artifact_generation_implemented)
        self.assertFalse(plan.artifact_execution_implemented)
        self.assertFalse(plan.runtime_execution_implemented)
        self.assertFalse(plan.preview_behavior_change_implemented)
        self.assertFalse(plan.artifact_merge_implemented)
        self.assertFalse(plan.artifact_export_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.local_runtime_probe_implemented)
        self.assertFalse(plan.hitl_request_emitted)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_patterns_score_artifact_learning_without_artifact_mutation(self) -> None:
        plan = learn_artifacts(route="generate")

        fit_score = {"strong": 3, "moderate": 2, "weak": 1, "unsupported": 0}
        for pattern in plan.patterns:
            dumped = pattern.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PATTERN_FIELDS)
            self.assertEqual(
                pattern.serialization_version,
                "artifact_learning_pattern.v1",
            )
            self.assertEqual(pattern.route_name, RouteName.GENERATE)
            self.assertEqual(
                pattern.pattern_id,
                f"artifact_learning::{pattern.pattern_kind}",
            )
            self.assertEqual(
                pattern.artifact_learning_score,
                min(
                    1000,
                    max(
                        0,
                        fit_score[pattern.artifact_fit] * 100
                        + fit_score[pattern.creative_fit] * 70
                        + fit_score[pattern.generative_fit] * 70
                        + fit_score[pattern.interaction_fit] * 55
                        + fit_score[pattern.audiovisual_fit] * 40
                        + fit_score[pattern.export_fit] * 35
                        + fit_score[pattern.interoperability_fit] * 35
                        + fit_score[pattern.portability_fit] * 35
                        + int(pattern.capability_confidence * 260)
                        + pattern.learning_priority_score // 4
                        + pattern.artifact_learning_weight
                        - pattern.missing_information_count * 25
                        - pattern.hitl_question_count * 15
                        - pattern.capability_risk_count * 20,
                    ),
                ),
            )
            self.assertIn("artifact_selection", pattern.blocked_runtime_behaviors)
            self.assertTrue(pattern.artifact_pattern_tags)
            self.assertTrue(pattern.advisory_actions)
            self.assertTrue(pattern.evidence)
            self.assertTrue(pattern.hitl_required)
            self.assertTrue(pattern.artifact_learning_implemented)
            self.assertTrue(pattern.artifact_planning_metadata_used)
            self.assertTrue(pattern.artifact_capability_metadata_used)
            self.assertTrue(pattern.adaptive_learning_metadata_used)
            self.assertFalse(pattern.artifact_selection_implemented)
            self.assertFalse(pattern.artifact_mutation_implemented)
            self.assertFalse(pattern.artifact_generation_implemented)
            self.assertFalse(pattern.artifact_execution_implemented)
            self.assertFalse(pattern.runtime_execution_implemented)
            self.assertFalse(pattern.preview_behavior_change_implemented)
            self.assertFalse(pattern.artifact_merge_implemented)
            self.assertFalse(pattern.artifact_export_implemented)
            self.assertFalse(pattern.provider_model_routing_implemented)
            self.assertFalse(pattern.provider_execution_implemented)
            self.assertFalse(pattern.local_runtime_probe_implemented)
            self.assertFalse(pattern.workflow_control_implemented)
            self.assertFalse(pattern.workflow_graph_mutation_implemented)
            self.assertFalse(pattern.workflow_execution_implemented)
            self.assertFalse(pattern.persistent_storage_write_implemented)
            self.assertFalse(pattern.generated_output_mutation_implemented)
            self.assertFalse(pattern.runtime_evolution_implemented)
            self.assertTrue(pattern.advisory_only)

        guarded = artifact_learning_pattern_by_id(
            "artifact_learning::artifact_guardrail_learning",
            plan,
        )
        critical = artifact_learning_patterns_for_priority("critical", plan)
        review = artifact_learning_patterns_for_status("review_required", plan)
        self.assertIsNotNone(guarded)
        assert guarded is not None
        self.assertEqual(guarded.status, "guarded")
        self.assertEqual(guarded.priority, "guarded")
        self.assertEqual(len(critical), 3)
        self.assertEqual(len(review), 3)

    def test_plan_rejects_mismatched_artifact_learning_metadata(self) -> None:
        plan = learn_artifacts()
        payload = plan.model_dump(mode="json")
        payload["pattern_ids"] = ("missing",) + tuple(payload["pattern_ids"][1:])

        with self.assertRaisesRegex(ValueError, "pattern_ids must match"):
            ArtifactLearningPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_artifact_learning_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_artifact_learning_score must match",
        ):
            ArtifactLearningPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["applied_artifact_pattern_ids"] = (plan.pattern_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "applied_artifact_pattern_ids must remain empty",
        ):
            ArtifactLearningPlan(**payload)

    def test_learning_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Review artifact learning signals for a creative workflow.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = learn_artifacts(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_learning_does_not_declare_artifact_application_terms(self) -> None:
        plan = learn_artifacts(route=RouteName.GENERATE)
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
                        pattern.source_artifact_plan_role,
                        pattern.source_artifact_capability_role,
                        pattern.source_learning_signal_id,
                        pattern.source_workflow_risk_factor_id,
                        pattern.artifact_type,
                        pattern.artifact_family,
                        pattern.target_runtime_id,
                        pattern.artifact_fit,
                        pattern.creative_fit,
                        pattern.generative_fit,
                        pattern.interaction_fit,
                        pattern.audiovisual_fit,
                        pattern.export_fit,
                        pattern.interoperability_fit,
                        pattern.portability_fit,
                        *pattern.artifact_pattern_tags,
                        pattern.artifact_summary,
                        *pattern.advisory_actions,
                        *pattern.evidence,
                        *pattern.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "select_artifact(",
            "mutate_artifact(",
            "generate_artifact(",
            "execute_artifact(",
            "execute_runtime(",
            "change_preview(",
            "merge_artifacts(",
            "export_artifact(",
            "route_provider(",
            "switch_provider(",
            "switch_model(",
            "execute_provider(",
            "probe_local_runtime(",
            "emit_hitl_request(",
            "control_workflow(",
            "mutate_workflow_graph(",
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
