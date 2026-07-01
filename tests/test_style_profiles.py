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
    StyleProfilePlan,
    build_style_profiles,
    build_user_preferences,
    route_request,
    style_profile_by_id,
    style_profiles_for_fidelity,
    style_profiles_for_status,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
REQUIRED_PROFILE_FIELDS = {
    "profile_id",
    "profile_kind",
    "status",
    "fidelity",
    "route_name",
    "task_type",
    "execution_mode_id",
    "style_axis",
    "source_user_preference_id",
    "profile_summary",
    "preference_alignment_score",
    "evidence_strength_score",
    "conflict_risk_score",
    "governance_weight",
    "style_profile_score",
    "hitl_required_before_application",
    "style_tags",
    "explainability_notes",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "style_profiles_implemented",
    "style_profile_metadata_implemented",
    "user_preferences_source_used",
    "style_profile_storage_write_implemented",
    "style_profile_creation_implemented",
    "style_profile_update_implemented",
    "style_profile_deletion_implemented",
    "automatic_style_learning_implemented",
    "style_profile_application_implemented",
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


class StyleProfilesTests(unittest.TestCase):
    def test_plan_builds_style_profile_metadata(self) -> None:
        preferences = build_user_preferences(route=RouteName.GENERATE)
        plan = build_style_profiles(
            route=RouteName.GENERATE,
            user_preferences=preferences,
        )

        self.assertEqual(plan.role, "style_profiles")
        self.assertEqual(plan.serialization_version, "style_profile_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.source_user_preferences_serialization_version,
            "user_preferences_plan.v1",
        )
        self.assertEqual(plan.source_user_preference_ids, preferences.preference_ids)
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.profile_count, 5)
        self.assertEqual(plan.candidate_profile_count, 1)
        self.assertEqual(plan.review_required_profile_count, 2)
        self.assertEqual(plan.guarded_profile_count, 2)
        self.assertEqual(plan.high_fidelity_profile_count, 3)
        self.assertEqual(plan.hitl_required_profile_count, 5)
        self.assertFalse(plan.created_profile_ids)
        self.assertFalse(plan.updated_profile_ids)
        self.assertFalse(plan.applied_profile_ids)
        self.assertEqual(plan.overall_style_profile_posture, "guarded")
        self.assertIn("do not write style storage", plan.authority_boundary)
        self.assertTrue(plan.style_profiles_implemented)
        self.assertTrue(plan.style_profile_metadata_implemented)
        self.assertTrue(plan.user_preferences_source_used)
        self.assertFalse(plan.style_profile_storage_write_implemented)
        self.assertFalse(plan.style_profile_creation_implemented)
        self.assertFalse(plan.style_profile_update_implemented)
        self.assertFalse(plan.style_profile_deletion_implemented)
        self.assertFalse(plan.automatic_style_learning_implemented)
        self.assertFalse(plan.style_profile_application_implemented)
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

    def test_profiles_score_style_posture_without_application(self) -> None:
        plan = build_style_profiles(route="generate")

        for profile in plan.profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(profile.serialization_version, "style_profile.v1")
            self.assertEqual(profile.route_name, RouteName.GENERATE)
            self.assertEqual(
                profile.profile_id,
                f"style_profiles::{profile.profile_kind}",
            )
            self.assertEqual(
                profile.style_profile_score,
                min(
                    1000,
                    max(
                        0,
                        profile.preference_alignment_score * 3
                        + profile.evidence_strength_score * 3
                        + profile.conflict_risk_score * 4
                        + profile.governance_weight,
                    ),
                ),
            )
            self.assertIn("style_profiles", profile.style_tags)
            self.assertIn(
                "style_profile_storage_write",
                profile.blocked_runtime_behaviors,
            )
            self.assertTrue(profile.explainability_notes)
            self.assertTrue(profile.advisory_actions)
            self.assertTrue(profile.evidence)
            self.assertTrue(profile.hitl_required_before_application)
            self.assertTrue(profile.style_profiles_implemented)
            self.assertTrue(profile.style_profile_metadata_implemented)
            self.assertTrue(profile.user_preferences_source_used)
            self.assertFalse(profile.style_profile_storage_write_implemented)
            self.assertFalse(profile.style_profile_creation_implemented)
            self.assertFalse(profile.style_profile_update_implemented)
            self.assertFalse(profile.style_profile_deletion_implemented)
            self.assertFalse(profile.automatic_style_learning_implemented)
            self.assertFalse(profile.style_profile_application_implemented)
            self.assertFalse(profile.preference_mutation_implemented)
            self.assertFalse(profile.personalization_application_implemented)
            self.assertFalse(profile.memory_retrieval_execution_implemented)
            self.assertFalse(profile.memory_storage_write_implemented)
            self.assertFalse(profile.provider_model_routing_implemented)
            self.assertFalse(profile.provider_execution_implemented)
            self.assertFalse(profile.agent_invocation_implemented)
            self.assertFalse(profile.workflow_control_implemented)
            self.assertFalse(profile.workflow_graph_mutation_implemented)
            self.assertFalse(profile.workflow_execution_implemented)
            self.assertFalse(profile.persistent_storage_write_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)
            self.assertFalse(profile.runtime_evolution_implemented)
            self.assertTrue(profile.advisory_only)

        palette = style_profile_by_id("style_profiles::palette_profile", plan)
        self.assertIsNotNone(palette)
        assert palette is not None
        self.assertEqual(palette.status, "guarded")
        self.assertEqual(palette.fidelity, "guarded")
        self.assertEqual(len(style_profiles_for_status("guarded", plan)), 2)
        self.assertEqual(len(style_profiles_for_fidelity("high", plan)), 1)

    def test_plan_rejects_mismatched_style_profile_metadata(self) -> None:
        plan = build_style_profiles()
        payload = plan.model_dump(mode="json")
        payload["profile_ids"] = ("missing",) + tuple(payload["profile_ids"][1:])

        with self.assertRaisesRegex(ValueError, "profile_ids must match"):
            StyleProfilePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_style_profile_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_style_profile_score must match",
        ):
            StyleProfilePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["applied_profile_ids"] = (plan.profile_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "applied_profile_ids must remain empty",
        ):
            StyleProfilePlan(**payload)

    def test_style_profiles_do_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Review style profiles for a creative coding project.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_style_profiles(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_style_profiles_do_not_declare_runtime_application_terms(self) -> None:
        plan = build_style_profiles(route=RouteName.GENERATE)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for profile in plan.profiles
                    for field in (
                        profile.profile_id,
                        profile.profile_kind,
                        profile.status,
                        profile.fidelity,
                        profile.style_axis,
                        profile.source_user_preference_id,
                        profile.profile_summary,
                        *profile.style_tags,
                        *profile.explainability_notes,
                        *profile.advisory_actions,
                        *profile.evidence,
                        *profile.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "write_style_profile(",
            "create_style_profile(",
            "update_style_profile(",
            "delete_style_profile(",
            "learn_style(",
            "apply_style_profile(",
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
