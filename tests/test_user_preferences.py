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
    UserPreferencesPlan,
    build_long_term_creative_memory,
    build_user_preferences,
    route_request,
    user_preference_by_id,
    user_preferences_for_confidence,
    user_preferences_for_status,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
REQUIRED_PREFERENCE_FIELDS = {
    "preference_id",
    "preference_kind",
    "status",
    "confidence",
    "route_name",
    "task_type",
    "execution_mode_id",
    "preference_scope",
    "source_long_term_memory_record_id",
    "preference_statement",
    "explicitness_score",
    "consistency_score",
    "conflict_risk_score",
    "sensitivity_weight",
    "preference_governance_score",
    "hitl_required_before_mutation",
    "applicable_context_tags",
    "explainability_notes",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "user_preferences_implemented",
    "preference_signal_metadata_implemented",
    "long_term_memory_source_used",
    "preference_storage_write_implemented",
    "preference_record_creation_implemented",
    "preference_record_update_implemented",
    "preference_record_deletion_implemented",
    "automatic_preference_learning_implemented",
    "preference_mutation_implemented",
    "personalization_application_implemented",
    "memory_retrieval_execution_implemented",
    "memory_storage_write_implemented",
    "provider_model_routing_implemented",
    "automatic_provider_switching_implemented",
    "automatic_model_switching_implemented",
    "provider_execution_implemented",
    "agent_invocation_implemented",
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


class UserPreferencesTests(unittest.TestCase):
    def test_plan_builds_user_preference_metadata(self) -> None:
        memory = build_long_term_creative_memory(route=RouteName.GENERATE)
        plan = build_user_preferences(
            route=RouteName.GENERATE,
            long_term_memory=memory,
        )

        self.assertEqual(plan.role, "user_preferences")
        self.assertEqual(plan.serialization_version, "user_preferences_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.source_long_term_memory_serialization_version,
            "long_term_creative_memory_plan.v1",
        )
        self.assertEqual(plan.source_long_term_memory_record_ids, memory.record_ids)
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.preference_count, 5)
        self.assertEqual(plan.candidate_preference_count, 1)
        self.assertEqual(plan.review_required_preference_count, 2)
        self.assertEqual(plan.guarded_preference_count, 2)
        self.assertEqual(plan.high_confidence_preference_count, 3)
        self.assertEqual(plan.hitl_required_preference_count, 5)
        self.assertFalse(plan.learned_preference_ids)
        self.assertFalse(plan.mutated_preference_ids)
        self.assertFalse(plan.personalized_preference_ids)
        self.assertEqual(plan.overall_preference_posture, "guarded")
        self.assertIn("does not write preference storage", plan.authority_boundary)
        self.assertTrue(plan.user_preferences_implemented)
        self.assertTrue(plan.preference_signal_metadata_implemented)
        self.assertTrue(plan.long_term_memory_source_used)
        self.assertFalse(plan.preference_storage_write_implemented)
        self.assertFalse(plan.preference_record_creation_implemented)
        self.assertFalse(plan.preference_record_update_implemented)
        self.assertFalse(plan.preference_record_deletion_implemented)
        self.assertFalse(plan.automatic_preference_learning_implemented)
        self.assertFalse(plan.preference_mutation_implemented)
        self.assertFalse(plan.personalization_application_implemented)
        self.assertFalse(plan.memory_retrieval_execution_implemented)
        self.assertFalse(plan.memory_storage_write_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_preferences_score_governance_without_mutation(self) -> None:
        plan = build_user_preferences(route="generate")

        for preference in plan.preferences:
            dumped = preference.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PREFERENCE_FIELDS)
            self.assertEqual(
                preference.serialization_version,
                "user_preference_signal.v1",
            )
            self.assertEqual(preference.route_name, RouteName.GENERATE)
            self.assertEqual(
                preference.preference_id,
                f"user_preferences::{preference.preference_kind}",
            )
            self.assertEqual(
                preference.preference_governance_score,
                min(
                    1000,
                    max(
                        0,
                        preference.explicitness_score * 3
                        + preference.consistency_score * 3
                        + preference.conflict_risk_score * 4
                        + preference.sensitivity_weight,
                    ),
                ),
            )
            self.assertIn(
                "user_preferences",
                preference.applicable_context_tags,
            )
            self.assertIn(
                "preference_storage_write",
                preference.blocked_runtime_behaviors,
            )
            self.assertTrue(preference.explainability_notes)
            self.assertTrue(preference.advisory_actions)
            self.assertTrue(preference.evidence)
            self.assertTrue(preference.hitl_required_before_mutation)
            self.assertTrue(preference.user_preferences_implemented)
            self.assertTrue(preference.preference_signal_metadata_implemented)
            self.assertTrue(preference.long_term_memory_source_used)
            self.assertFalse(preference.preference_storage_write_implemented)
            self.assertFalse(preference.preference_record_creation_implemented)
            self.assertFalse(preference.preference_record_update_implemented)
            self.assertFalse(preference.preference_record_deletion_implemented)
            self.assertFalse(preference.automatic_preference_learning_implemented)
            self.assertFalse(preference.preference_mutation_implemented)
            self.assertFalse(preference.personalization_application_implemented)
            self.assertFalse(preference.memory_retrieval_execution_implemented)
            self.assertFalse(preference.memory_storage_write_implemented)
            self.assertFalse(preference.provider_model_routing_implemented)
            self.assertFalse(preference.provider_execution_implemented)
            self.assertFalse(preference.agent_invocation_implemented)
            self.assertFalse(preference.workflow_control_implemented)
            self.assertFalse(preference.workflow_graph_mutation_implemented)
            self.assertFalse(preference.workflow_execution_implemented)
            self.assertFalse(preference.persistent_storage_write_implemented)
            self.assertFalse(preference.generated_output_mutation_implemented)
            self.assertFalse(preference.runtime_evolution_implemented)
            self.assertTrue(preference.advisory_only)

        visual = user_preference_by_id(
            "user_preferences::visual_style_preference",
            plan,
        )
        self.assertIsNotNone(visual)
        assert visual is not None
        self.assertEqual(visual.status, "guarded")
        self.assertEqual(visual.confidence, "guarded")
        self.assertEqual(len(user_preferences_for_status("guarded", plan)), 2)
        self.assertEqual(len(user_preferences_for_confidence("high", plan)), 1)

    def test_plan_rejects_mismatched_preference_metadata(self) -> None:
        plan = build_user_preferences()
        payload = plan.model_dump(mode="json")
        payload["preference_ids"] = (
            "missing",
        ) + tuple(payload["preference_ids"][1:])

        with self.assertRaisesRegex(ValueError, "preference_ids must match"):
            UserPreferencesPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_preference_governance_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_preference_governance_score must match",
        ):
            UserPreferencesPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["mutated_preference_ids"] = (plan.preference_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "mutated_preference_ids must remain empty",
        ):
            UserPreferencesPlan(**payload)

    def test_user_preferences_do_not_change_routing_or_provider_factory(
        self,
    ) -> None:
        request = AssistantRequest(
            query="Review user preferences for a creative coding project.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_user_preferences(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_user_preferences_do_not_declare_runtime_application_terms(
        self,
    ) -> None:
        plan = build_user_preferences(route=RouteName.GENERATE)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for preference in plan.preferences
                    for field in (
                        preference.preference_id,
                        preference.preference_kind,
                        preference.status,
                        preference.confidence,
                        preference.preference_scope,
                        preference.source_long_term_memory_record_id,
                        preference.preference_statement,
                        *preference.applicable_context_tags,
                        *preference.explainability_notes,
                        *preference.advisory_actions,
                        *preference.evidence,
                        *preference.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "write_preference(",
            "create_preference_record(",
            "update_preference_record(",
            "delete_preference_record(",
            "learn_preference(",
            "mutate_preference(",
            "apply_personalization(",
            "retrieve_memory(",
            "write_memory(",
            "route_provider(",
            "switch_provider(",
            "switch_model(",
            "execute_provider(",
            "invoke_agent(",
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
