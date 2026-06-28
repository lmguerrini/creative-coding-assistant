import unittest

from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    HitlDecisionRegistry,
    auto_mode_registry,
    hitl_decision_profile_by_id,
    hitl_decision_profiles_for_route,
    hitl_decision_registry,
    hybrid_execution_registry,
    studio_mode_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_PROFILE_IDS = (
    "hitl_visibility_decision_profile",
    "hitl_confirmation_decision_profile",
    "hitl_risk_review_decision_profile",
    "hitl_final_review_decision_profile",
)
EXPECTED_POSTURES = (
    "visible_only",
    "confirmation_advised",
    "risk_review_advised",
    "final_review_advised",
)
EXPECTED_SOURCE_REGISTRIES = (
    "studio_mode_registry",
    "auto_mode_registry",
    "hybrid_execution_registry",
    "hitl_escalation_gate_registry",
    "workflow_agent_handoff_registry",
    "agent_escalation_signal_registry",
)
EXPECTED_REVIEW_SURFACES = (
    "hitl_decision_panel",
    "studio_mode_shell",
    "auto_mode_panel",
    "hybrid_execution_panel",
    "operator_review_panel",
)
REQUIRED_PROFILE_FIELDS = {
    "hitl_decision_profile_id",
    "profile_name",
    "hitl_decision_posture",
    "source_studio_mode_profile_ids",
    "source_auto_mode_profile_ids",
    "source_execution_profile_ids",
    "route_applicability",
    "decision_inputs",
    "advisory_decision_outputs",
    "human_review_surfaces",
    "source_registries",
    "observability_surfaces",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "hitl_decision_execution_implemented",
    "human_input_request_implemented",
    "escalation_approval_implemented",
    "workflow_interruption_implemented",
    "workflow_control_implemented",
    "auto_mode_execution_implemented",
    "hybrid_execution_implemented",
    "provider_model_routing_implemented",
    "retry_triggering_implemented",
    "generated_output_mutation_implemented",
    "persistent_replay_storage_implemented",
    "serialization_version",
    "metadata_only",
}


class HybridStudioHitlDecisionTests(unittest.TestCase):
    def test_hitl_decision_registry_covers_expected_profiles(self) -> None:
        registry = hitl_decision_registry()

        self.assertEqual(registry.role, "hitl_decision_registry")
        self.assertEqual(
            registry.serialization_version,
            "hitl_decision_registry.v1",
        )
        self.assertEqual(registry.hitl_decision_profile_ids, EXPECTED_PROFILE_IDS)
        self.assertEqual(registry.hitl_decision_postures, EXPECTED_POSTURES)
        self.assertEqual(
            registry.studio_mode_profile_ids,
            studio_mode_registry().studio_mode_profile_ids,
        )
        self.assertEqual(
            registry.auto_mode_profile_ids,
            auto_mode_registry().auto_mode_profile_ids,
        )
        self.assertEqual(
            registry.execution_profile_ids,
            hybrid_execution_registry().execution_profile_ids,
        )
        self.assertEqual(registry.human_review_surfaces, EXPECTED_REVIEW_SURFACES)
        self.assertEqual(registry.profile_count, 4)
        self.assertEqual(registry.route_names, tuple(RouteName))
        self.assertEqual(registry.source_registries, EXPECTED_SOURCE_REGISTRIES)
        self.assertIn("does not request human input", registry.authority_boundary)
        self.assertIn("human_input_request", registry.blocked_runtime_behaviors)
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.hitl_decision_execution_implemented)
        self.assertFalse(registry.human_input_request_implemented)
        self.assertFalse(registry.escalation_approval_implemented)
        self.assertFalse(registry.workflow_interruption_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.auto_mode_execution_implemented)
        self.assertFalse(registry.hybrid_execution_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.persistent_replay_storage_implemented)

    def test_hitl_decision_profiles_are_passive_and_source_aligned(self) -> None:
        registry = hitl_decision_registry()
        known_routes = set(registry.route_names)
        known_studio_profiles = set(studio_mode_registry().studio_mode_profile_ids)
        known_auto_profiles = set(auto_mode_registry().auto_mode_profile_ids)
        known_execution_profiles = set(
            hybrid_execution_registry().execution_profile_ids
        )
        known_review_surfaces = set(registry.human_review_surfaces)

        for profile in registry.hitl_decision_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(profile.serialization_version, "hitl_decision_profile.v1")
            self.assertEqual(profile.source_registries, registry.source_registries)
            self.assertEqual(
                profile.observability_surfaces,
                registry.observability_surfaces,
            )
            self.assertTrue(set(profile.route_applicability).issubset(known_routes))
            self.assertTrue(
                set(profile.source_studio_mode_profile_ids).issubset(
                    known_studio_profiles
                )
            )
            self.assertTrue(
                set(profile.source_auto_mode_profile_ids).issubset(known_auto_profiles)
            )
            self.assertTrue(
                set(profile.source_execution_profile_ids).issubset(
                    known_execution_profiles
                )
            )
            self.assertTrue(
                set(profile.human_review_surfaces).issubset(known_review_surfaces)
            )
            self.assertIn("workflow_interruption", profile.blocked_runtime_behaviors)
            self.assertTrue(profile.metadata_only)
            self.assertFalse(profile.hitl_decision_execution_implemented)
            self.assertFalse(profile.human_input_request_implemented)
            self.assertFalse(profile.escalation_approval_implemented)
            self.assertFalse(profile.workflow_interruption_implemented)
            self.assertFalse(profile.workflow_control_implemented)
            self.assertFalse(profile.auto_mode_execution_implemented)
            self.assertFalse(profile.hybrid_execution_implemented)
            self.assertFalse(profile.provider_model_routing_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)
            self.assertFalse(profile.persistent_replay_storage_implemented)

    def test_hitl_decision_lookup_helpers_are_stable(self) -> None:
        profile = hitl_decision_profile_by_id("hitl_risk_review_decision_profile")
        missing_profile = hitl_decision_profile_by_id("missing_profile")
        review_profiles = hitl_decision_profiles_for_route(RouteName.REVIEW)
        preview_profiles = hitl_decision_profiles_for_route("preview")

        self.assertIsNone(missing_profile)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.hitl_decision_posture, "risk_review_advised")
        self.assertIn(
            "risk_review_recommended_metadata", profile.advisory_decision_outputs
        )
        self.assertEqual(
            tuple(item.hitl_decision_profile_id for item in preview_profiles),
            (
                "hitl_visibility_decision_profile",
                "hitl_confirmation_decision_profile",
            ),
        )
        self.assertEqual(
            tuple(item.hitl_decision_profile_id for item in review_profiles),
            EXPECTED_PROFILE_IDS,
        )

    def test_registry_rejects_mismatched_sources_or_review_refs(self) -> None:
        registry = hitl_decision_registry()
        first_profile = registry.hitl_decision_profiles[0]
        duplicate_profile = first_profile.model_copy(
            update={"profile_name": "Duplicate Profile"}
        )
        unknown_studio_profile = first_profile.model_copy(
            update={"source_studio_mode_profile_ids": ("unknown_studio_profile",)}
        )
        mismatched_source_profile = first_profile.model_copy(
            update={
                "source_registries": (
                    "studio_mode_registry",
                    "auto_mode_registry",
                    "hybrid_execution_registry",
                    "hitl_escalation_gate_registry",
                    "workflow_agent_handoff_registry",
                    "unknown_registry",
                )
            }
        )

        with self.assertRaisesRegex(
            ValueError, "hitl_decision_profile_ids must be unique"
        ):
            HitlDecisionRegistry(
                hitl_decision_profiles=(first_profile, duplicate_profile)
                + registry.hitl_decision_profiles[2:],
                hitl_decision_profile_ids=registry.hitl_decision_profile_ids,
                hitl_decision_postures=registry.hitl_decision_postures,
                studio_mode_profile_ids=registry.studio_mode_profile_ids,
                auto_mode_profile_ids=registry.auto_mode_profile_ids,
                execution_profile_ids=registry.execution_profile_ids,
                human_review_surfaces=registry.human_review_surfaces,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

        with self.assertRaisesRegex(ValueError, "known Studio Mode profiles"):
            HitlDecisionRegistry(
                hitl_decision_profiles=(unknown_studio_profile,)
                + registry.hitl_decision_profiles[1:],
                hitl_decision_profile_ids=registry.hitl_decision_profile_ids,
                hitl_decision_postures=registry.hitl_decision_postures,
                studio_mode_profile_ids=registry.studio_mode_profile_ids,
                auto_mode_profile_ids=registry.auto_mode_profile_ids,
                execution_profile_ids=registry.execution_profile_ids,
                human_review_surfaces=registry.human_review_surfaces,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

        with self.assertRaisesRegex(ValueError, "source_registries must match"):
            HitlDecisionRegistry(
                hitl_decision_profiles=(mismatched_source_profile,)
                + registry.hitl_decision_profiles[1:],
                hitl_decision_profile_ids=registry.hitl_decision_profile_ids,
                hitl_decision_postures=registry.hitl_decision_postures,
                studio_mode_profile_ids=registry.studio_mode_profile_ids,
                auto_mode_profile_ids=registry.auto_mode_profile_ids,
                execution_profile_ids=registry.execution_profile_ids,
                human_review_surfaces=registry.human_review_surfaces,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

    def test_hitl_decision_metadata_does_not_change_provider_factory(self) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
        )

        hitl_decision_registry()
        provider = build_generation_provider(settings)

        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_hitl_decision_metadata_does_not_request_human_input(self) -> None:
        registry = hitl_decision_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.source_registries,
                *registry.human_review_surfaces,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for profile in registry.hitl_decision_profiles
                    for field in (
                        profile.hitl_decision_profile_id,
                        profile.hitl_decision_posture,
                        *profile.decision_inputs,
                        *profile.advisory_decision_outputs,
                        *profile.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "route_provider",
            "provider_route",
            "execute_provider",
            "request_human_now",
            "approve_escalation_now",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
