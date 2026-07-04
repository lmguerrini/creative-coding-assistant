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
    PersonalizationEnginePlan,
    build_creative_dna,
    build_personalization_engine,
    personalization_recommendation_by_id,
    personalization_recommendations_for_confidence,
    personalization_recommendations_for_status,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
REQUIRED_RECOMMENDATION_FIELDS = {
    "personalization_id",
    "recommendation_kind",
    "status",
    "confidence",
    "route_name",
    "task_type",
    "execution_mode_id",
    "personalization_scope",
    "source_creative_dna_id",
    "source_user_preference_id",
    "source_style_profile_id",
    "source_project_memory_signal_id",
    "personalization_summary",
    "preference_alignment_score",
    "creative_dna_alignment_score",
    "project_fit_score",
    "safety_risk_score",
    "governance_weight",
    "personalization_score",
    "hitl_required_before_application",
    "context_tags",
    "explainability_notes",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "personalization_engine_implemented",
    "personalization_metadata_implemented",
    "creative_dna_source_used",
    "user_preferences_source_used",
    "style_profile_source_used",
    "project_memory_source_used",
    "personalization_storage_write_implemented",
    "personalization_rule_creation_implemented",
    "personalization_rule_update_implemented",
    "personalization_rule_deletion_implemented",
    "automatic_personalization_learning_implemented",
    "personalization_application_implemented",
    "creative_dna_application_implemented",
    "style_profile_application_implemented",
    "preference_mutation_implemented",
    "memory_retrieval_execution_implemented",
    "memory_storage_write_implemented",
    "project_memory_storage_write_implemented",
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


class PersonalizationEngineTests(unittest.TestCase):
    def test_plan_builds_personalization_metadata(self) -> None:
        creative_dna = build_creative_dna(route=RouteName.GENERATE)
        plan = build_personalization_engine(
            route=RouteName.GENERATE,
            creative_dna=creative_dna,
        )

        self.assertEqual(plan.role, "personalization_engine")
        self.assertEqual(
            plan.serialization_version,
            "personalization_engine_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.source_creative_dna_serialization_version,
            "creative_dna_plan.v1",
        )
        self.assertEqual(
            plan.source_user_preferences_serialization_version,
            "user_preferences_plan.v1",
        )
        self.assertEqual(
            plan.source_style_profile_serialization_version,
            "style_profile_plan.v1",
        )
        self.assertEqual(
            plan.source_project_memory_serialization_version,
            "project_memory_plan.v1",
        )
        self.assertEqual(plan.source_creative_dna_ids, creative_dna.signature_ids)
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.recommendation_count, 5)
        self.assertEqual(plan.candidate_recommendation_count, 1)
        self.assertEqual(plan.review_required_recommendation_count, 2)
        self.assertEqual(plan.guarded_recommendation_count, 2)
        self.assertEqual(plan.high_confidence_recommendation_count, 3)
        self.assertEqual(plan.hitl_required_recommendation_count, 5)
        self.assertFalse(plan.persisted_personalization_ids)
        self.assertFalse(plan.learned_personalization_ids)
        self.assertFalse(plan.applied_personalization_ids)
        self.assertFalse(plan.mutated_preference_ids)
        self.assertEqual(plan.overall_personalization_posture, "guarded")
        self.assertIn(
            "does not write personalization storage",
            plan.authority_boundary,
        )
        self.assertIn(
            "apply personalization to prompts or generated output",
            plan.authority_boundary,
        )
        self.assertTrue(plan.personalization_engine_implemented)
        self.assertTrue(plan.personalization_metadata_implemented)
        self.assertTrue(plan.creative_dna_source_used)
        self.assertTrue(plan.user_preferences_source_used)
        self.assertTrue(plan.style_profile_source_used)
        self.assertTrue(plan.project_memory_source_used)
        self.assertFalse(plan.personalization_storage_write_implemented)
        self.assertFalse(plan.personalization_rule_creation_implemented)
        self.assertFalse(plan.personalization_rule_update_implemented)
        self.assertFalse(plan.personalization_rule_deletion_implemented)
        self.assertFalse(plan.automatic_personalization_learning_implemented)
        self.assertFalse(plan.personalization_application_implemented)
        self.assertFalse(plan.creative_dna_application_implemented)
        self.assertFalse(plan.style_profile_application_implemented)
        self.assertFalse(plan.preference_mutation_implemented)
        self.assertFalse(plan.memory_retrieval_execution_implemented)
        self.assertFalse(plan.memory_storage_write_implemented)
        self.assertFalse(plan.project_memory_storage_write_implemented)
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

    def test_recommendations_score_personalization_without_application(self) -> None:
        plan = build_personalization_engine(route="generate")

        for recommendation in plan.recommendations:
            dumped = recommendation.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_RECOMMENDATION_FIELDS)
            self.assertEqual(
                recommendation.serialization_version,
                "personalization_recommendation.v1",
            )
            self.assertEqual(recommendation.route_name, RouteName.GENERATE)
            self.assertEqual(
                recommendation.personalization_id,
                f"personalization_engine::{recommendation.recommendation_kind}",
            )
            self.assertEqual(
                recommendation.personalization_score,
                min(
                    1000,
                    max(
                        0,
                        recommendation.preference_alignment_score * 3
                        + recommendation.creative_dna_alignment_score * 3
                        + recommendation.project_fit_score * 2
                        + recommendation.safety_risk_score * 3
                        + recommendation.governance_weight,
                    ),
                ),
            )
            self.assertIn("personalization_engine", recommendation.context_tags)
            self.assertIn(
                "automatic_personalization_application",
                recommendation.blocked_runtime_behaviors,
            )
            self.assertTrue(recommendation.explainability_notes)
            self.assertTrue(recommendation.advisory_actions)
            self.assertTrue(recommendation.evidence)
            self.assertTrue(recommendation.hitl_required_before_application)
            self.assertTrue(recommendation.personalization_engine_implemented)
            self.assertTrue(recommendation.personalization_metadata_implemented)
            self.assertTrue(recommendation.creative_dna_source_used)
            self.assertTrue(recommendation.user_preferences_source_used)
            self.assertTrue(recommendation.style_profile_source_used)
            self.assertTrue(recommendation.project_memory_source_used)
            self.assertFalse(recommendation.personalization_storage_write_implemented)
            self.assertFalse(recommendation.personalization_rule_creation_implemented)
            self.assertFalse(recommendation.personalization_rule_update_implemented)
            self.assertFalse(recommendation.personalization_rule_deletion_implemented)
            self.assertFalse(
                recommendation.automatic_personalization_learning_implemented
            )
            self.assertFalse(recommendation.personalization_application_implemented)
            self.assertFalse(recommendation.creative_dna_application_implemented)
            self.assertFalse(recommendation.style_profile_application_implemented)
            self.assertFalse(recommendation.preference_mutation_implemented)
            self.assertFalse(recommendation.memory_retrieval_execution_implemented)
            self.assertFalse(recommendation.memory_storage_write_implemented)
            self.assertFalse(recommendation.project_memory_storage_write_implemented)
            self.assertFalse(recommendation.provider_model_routing_implemented)
            self.assertFalse(recommendation.provider_execution_implemented)
            self.assertFalse(recommendation.agent_invocation_implemented)
            self.assertFalse(recommendation.workflow_control_implemented)
            self.assertFalse(recommendation.workflow_graph_mutation_implemented)
            self.assertFalse(recommendation.workflow_execution_implemented)
            self.assertFalse(recommendation.persistent_storage_write_implemented)
            self.assertFalse(recommendation.generated_output_mutation_implemented)
            self.assertFalse(recommendation.runtime_evolution_implemented)
            self.assertTrue(recommendation.advisory_only)

        style = personalization_recommendation_by_id(
            "personalization_engine::style_personalization",
            plan,
        )
        self.assertIsNotNone(style)
        assert style is not None
        self.assertEqual(style.status, "guarded")
        self.assertEqual(style.confidence, "guarded")
        self.assertEqual(
            len(personalization_recommendations_for_status("guarded", plan)),
            2,
        )
        self.assertEqual(
            len(personalization_recommendations_for_confidence("high", plan)),
            1,
        )

    def test_plan_rejects_mismatched_personalization_metadata(self) -> None:
        plan = build_personalization_engine()
        payload = plan.model_dump(mode="json")
        payload["recommendation_ids"] = ("missing",) + tuple(
            payload["recommendation_ids"][1:]
        )

        with self.assertRaisesRegex(ValueError, "recommendation_ids must match"):
            PersonalizationEnginePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_personalization_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_personalization_score must match",
        ):
            PersonalizationEnginePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["applied_personalization_ids"] = (plan.recommendation_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "applied_personalization_ids must remain empty",
        ):
            PersonalizationEnginePlan(**payload)

    def test_personalization_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Review personalization posture for a creative coding project.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_personalization_engine(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_personalization_does_not_declare_runtime_application_terms(self) -> None:
        plan = build_personalization_engine(route=RouteName.GENERATE)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for recommendation in plan.recommendations
                    for field in (
                        recommendation.personalization_id,
                        recommendation.recommendation_kind,
                        recommendation.status,
                        recommendation.confidence,
                        recommendation.personalization_scope,
                        recommendation.source_creative_dna_id,
                        recommendation.source_user_preference_id,
                        recommendation.source_style_profile_id,
                        recommendation.source_project_memory_signal_id,
                        recommendation.personalization_summary,
                        *recommendation.context_tags,
                        *recommendation.explainability_notes,
                        *recommendation.advisory_actions,
                        *recommendation.evidence,
                        *recommendation.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "write_personalization(",
            "create_personalization(",
            "update_personalization(",
            "delete_personalization(",
            "learn_personalization(",
            "apply_personalization(",
            "apply_creative_dna(",
            "apply_style_profile(",
            "mutate_preference(",
            "retrieve_memory(",
            "write_memory(",
            "write_project_memory(",
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
