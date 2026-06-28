import unittest

from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    QualityProfileRegistry,
    cost_profile_registry,
    execution_simulator_registry,
    hitl_decision_registry,
    model_profile_registry,
    provider_selection_registry,
    quality_escalation_registry,
    quality_profile_by_id,
    quality_profile_registry,
    quality_profiles_for_level,
    quality_profiles_for_route,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_PROFILE_IDS = (
    "planning_quality_profile",
    "creative_quality_profile",
    "refinement_quality_profile",
    "final_review_quality_profile",
)
EXPECTED_KINDS = (
    "planning_quality_review",
    "creative_quality_review",
    "refinement_quality_review",
    "final_review_quality",
)
EXPECTED_QUALITY_LEVELS = ("medium", "high", "critical", "low")
EXPECTED_SOURCE_REGISTRIES = (
    "model_profile_registry",
    "cost_profile_registry",
    "provider_selection_registry",
    "quality_escalation_registry",
    "execution_simulator_registry",
    "hitl_decision_registry",
)
EXPECTED_QUALITY_SURFACES = (
    "quality_profile_panel",
    "model_profile_panel",
    "cost_profile_panel",
    "execution_simulator_panel",
    "hitl_decision_panel",
)
REQUIRED_PROFILE_FIELDS = {
    "quality_profile_id",
    "profile_name",
    "quality_profile_kind",
    "quality_level",
    "source_model_profile_ids",
    "source_cost_profile_ids",
    "source_provider_selection_profile_ids",
    "source_quality_escalation_profile_ids",
    "source_execution_simulation_profile_ids",
    "source_hitl_decision_profile_ids",
    "route_applicability",
    "quality_dimensions",
    "quality_inputs",
    "advisory_outputs",
    "quality_surface_refs",
    "source_registries",
    "observability_surfaces",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "quality_profile_execution_implemented",
    "quality_scoring_implemented",
    "quality_evaluation_implemented",
    "quality_escalation_implemented",
    "refinement_triggering_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "model_selection_implemented",
    "cost_optimization_implemented",
    "workflow_control_implemented",
    "human_input_request_implemented",
    "retry_triggering_implemented",
    "generated_output_mutation_implemented",
    "persistent_replay_storage_implemented",
    "serialization_version",
    "metadata_only",
}


class HybridStudioQualityProfileTests(unittest.TestCase):
    def test_quality_profile_registry_covers_expected_profiles(self) -> None:
        registry = quality_profile_registry()

        self.assertEqual(registry.role, "quality_profile_registry")
        self.assertEqual(registry.serialization_version, "quality_profile_registry.v1")
        self.assertEqual(registry.quality_profile_ids, EXPECTED_PROFILE_IDS)
        self.assertEqual(registry.quality_profile_kinds, EXPECTED_KINDS)
        self.assertEqual(registry.quality_levels, EXPECTED_QUALITY_LEVELS)
        self.assertEqual(
            registry.quality_levels,
            quality_escalation_registry().quality_levels,
        )
        self.assertEqual(
            registry.model_profile_ids,
            model_profile_registry().model_profile_ids,
        )
        self.assertEqual(
            registry.cost_profile_ids,
            cost_profile_registry().cost_profile_ids,
        )
        self.assertEqual(
            registry.provider_selection_profile_ids,
            provider_selection_registry().provider_selection_profile_ids,
        )
        self.assertEqual(
            registry.quality_escalation_profile_ids,
            quality_escalation_registry().quality_profile_ids,
        )
        self.assertEqual(
            registry.execution_simulation_profile_ids,
            execution_simulator_registry().execution_simulation_profile_ids,
        )
        self.assertEqual(
            registry.hitl_decision_profile_ids,
            hitl_decision_registry().hitl_decision_profile_ids,
        )
        self.assertEqual(registry.quality_surface_refs, EXPECTED_QUALITY_SURFACES)
        self.assertEqual(registry.profile_count, 4)
        self.assertEqual(registry.route_names, tuple(RouteName))
        self.assertEqual(registry.source_registries, EXPECTED_SOURCE_REGISTRIES)
        self.assertIn("does not calculate quality", registry.authority_boundary)
        self.assertIn("quality_scoring", registry.blocked_runtime_behaviors)
        self.assertIn("refinement_triggering", registry.blocked_runtime_behaviors)
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.quality_profile_execution_implemented)
        self.assertFalse(registry.quality_scoring_implemented)
        self.assertFalse(registry.quality_evaluation_implemented)
        self.assertFalse(registry.quality_escalation_implemented)
        self.assertFalse(registry.refinement_triggering_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.provider_execution_implemented)
        self.assertFalse(registry.model_selection_implemented)
        self.assertFalse(registry.cost_optimization_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.human_input_request_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.persistent_replay_storage_implemented)

    def test_quality_profiles_are_passive_and_source_aligned(self) -> None:
        registry = quality_profile_registry()
        known_routes = set(registry.route_names)
        known_model_profiles = set(model_profile_registry().model_profile_ids)
        known_cost_profiles = set(cost_profile_registry().cost_profile_ids)
        known_provider_profiles = set(
            provider_selection_registry().provider_selection_profile_ids
        )
        known_quality_escalations = set(
            quality_escalation_registry().quality_profile_ids
        )
        known_simulations = set(
            execution_simulator_registry().execution_simulation_profile_ids
        )
        known_hitl_profiles = set(hitl_decision_registry().hitl_decision_profile_ids)
        known_quality_surfaces = set(registry.quality_surface_refs)

        for profile in registry.quality_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(profile.serialization_version, "quality_profile.v1")
            self.assertEqual(profile.source_registries, registry.source_registries)
            self.assertEqual(
                profile.observability_surfaces,
                registry.observability_surfaces,
            )
            self.assertIn(profile.quality_level, registry.quality_levels)
            self.assertTrue(set(profile.route_applicability).issubset(known_routes))
            self.assertTrue(
                set(profile.source_model_profile_ids).issubset(known_model_profiles)
            )
            self.assertTrue(
                set(profile.source_cost_profile_ids).issubset(known_cost_profiles)
            )
            self.assertTrue(
                set(profile.source_provider_selection_profile_ids).issubset(
                    known_provider_profiles
                )
            )
            self.assertTrue(
                set(profile.source_quality_escalation_profile_ids).issubset(
                    known_quality_escalations
                )
            )
            self.assertTrue(
                set(profile.source_execution_simulation_profile_ids).issubset(
                    known_simulations
                )
            )
            self.assertTrue(
                set(profile.source_hitl_decision_profile_ids).issubset(
                    known_hitl_profiles
                )
            )
            self.assertTrue(
                set(profile.quality_surface_refs).issubset(known_quality_surfaces)
            )
            self.assertIn("quality_evaluation", profile.blocked_runtime_behaviors)
            self.assertIn("quality_escalation", profile.blocked_runtime_behaviors)
            self.assertTrue(profile.metadata_only)
            self.assertFalse(profile.quality_profile_execution_implemented)
            self.assertFalse(profile.quality_scoring_implemented)
            self.assertFalse(profile.quality_evaluation_implemented)
            self.assertFalse(profile.quality_escalation_implemented)
            self.assertFalse(profile.refinement_triggering_implemented)
            self.assertFalse(profile.provider_model_routing_implemented)
            self.assertFalse(profile.provider_execution_implemented)
            self.assertFalse(profile.model_selection_implemented)
            self.assertFalse(profile.cost_optimization_implemented)
            self.assertFalse(profile.workflow_control_implemented)
            self.assertFalse(profile.human_input_request_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)
            self.assertFalse(profile.persistent_replay_storage_implemented)

    def test_quality_profile_lookup_helpers_are_stable(self) -> None:
        profile = quality_profile_by_id("refinement_quality_profile")
        missing_profile = quality_profile_by_id("missing_profile")
        review_profiles = quality_profiles_for_route(RouteName.REVIEW)
        preview_profiles = quality_profiles_for_route("preview")
        critical_profiles = quality_profiles_for_level("critical")

        self.assertIsNone(missing_profile)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.quality_profile_kind, "refinement_quality_review")
        self.assertEqual(profile.quality_level, "critical")
        self.assertIn(
            "refinement_quality_review_context",
            profile.advisory_outputs,
        )
        self.assertEqual(
            tuple(item.quality_profile_id for item in preview_profiles),
            (
                "planning_quality_profile",
                "refinement_quality_profile",
            ),
        )
        self.assertEqual(
            tuple(item.quality_profile_id for item in review_profiles),
            (
                "creative_quality_profile",
                "refinement_quality_profile",
                "final_review_quality_profile",
            ),
        )
        self.assertEqual(
            tuple(item.quality_profile_id for item in critical_profiles),
            ("refinement_quality_profile",),
        )

    def test_registry_rejects_mismatched_sources_or_quality_metadata(self) -> None:
        registry = quality_profile_registry()
        first_profile = registry.quality_profiles[0]
        duplicate_profile = first_profile.model_copy(
            update={"profile_name": "Duplicate Profile"}
        )
        unknown_cost_profile = first_profile.model_copy(
            update={"source_cost_profile_ids": ("unknown_cost_profile",)}
        )
        unknown_quality_escalation_profile = first_profile.model_copy(
            update={
                "source_quality_escalation_profile_ids": (
                    "unknown_quality_escalation_profile",
                )
            }
        )

        with self.assertRaisesRegex(ValueError, "quality_profile_ids must be unique"):
            QualityProfileRegistry(
                quality_profiles=(first_profile, duplicate_profile)
                + registry.quality_profiles[2:],
                quality_profile_ids=registry.quality_profile_ids,
                quality_profile_kinds=registry.quality_profile_kinds,
                quality_levels=registry.quality_levels,
                model_profile_ids=registry.model_profile_ids,
                cost_profile_ids=registry.cost_profile_ids,
                provider_selection_profile_ids=registry.provider_selection_profile_ids,
                quality_escalation_profile_ids=(
                    registry.quality_escalation_profile_ids
                ),
                execution_simulation_profile_ids=(
                    registry.execution_simulation_profile_ids
                ),
                hitl_decision_profile_ids=registry.hitl_decision_profile_ids,
                quality_surface_refs=registry.quality_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

        with self.assertRaisesRegex(ValueError, "source_cost_profile_ids"):
            QualityProfileRegistry(
                quality_profiles=(unknown_cost_profile,)
                + registry.quality_profiles[1:],
                quality_profile_ids=registry.quality_profile_ids,
                quality_profile_kinds=registry.quality_profile_kinds,
                quality_levels=registry.quality_levels,
                model_profile_ids=registry.model_profile_ids,
                cost_profile_ids=registry.cost_profile_ids,
                provider_selection_profile_ids=registry.provider_selection_profile_ids,
                quality_escalation_profile_ids=(
                    registry.quality_escalation_profile_ids
                ),
                execution_simulation_profile_ids=(
                    registry.execution_simulation_profile_ids
                ),
                hitl_decision_profile_ids=registry.hitl_decision_profile_ids,
                quality_surface_refs=registry.quality_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

        with self.assertRaisesRegex(
            ValueError,
            "source_quality_escalation_profile_ids",
        ):
            QualityProfileRegistry(
                quality_profiles=(unknown_quality_escalation_profile,)
                + registry.quality_profiles[1:],
                quality_profile_ids=registry.quality_profile_ids,
                quality_profile_kinds=registry.quality_profile_kinds,
                quality_levels=registry.quality_levels,
                model_profile_ids=registry.model_profile_ids,
                cost_profile_ids=registry.cost_profile_ids,
                provider_selection_profile_ids=registry.provider_selection_profile_ids,
                quality_escalation_profile_ids=(
                    registry.quality_escalation_profile_ids
                ),
                execution_simulation_profile_ids=(
                    registry.execution_simulation_profile_ids
                ),
                hitl_decision_profile_ids=registry.hitl_decision_profile_ids,
                quality_surface_refs=registry.quality_surface_refs,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                observability_surfaces=registry.observability_surfaces,
            )

    def test_quality_profiles_do_not_change_provider_factory(self) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
        )

        quality_profile_registry()
        provider = build_generation_provider(settings)

        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")
