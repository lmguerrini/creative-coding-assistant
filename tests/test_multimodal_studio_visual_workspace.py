import unittest

from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    MultimodalVisualWorkspaceRegistry,
    multimodal_interactive_canvas_registry,
    multimodal_live_preview_registry,
    multimodal_multi_preview_registry,
    multimodal_visual_workspace_profile_by_id,
    multimodal_visual_workspace_profiles_for_interactive_canvas_profile,
    multimodal_visual_workspace_profiles_for_route,
    multimodal_visual_workspace_profiles_for_surface_kind,
    multimodal_visual_workspace_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_PROFILE_IDS = (
    "shell_visual_workspace",
    "artifact_selection_visual_workspace",
    "preview_visual_workspace",
    "inspector_visual_workspace",
)
EXPECTED_PROFILE_KINDS = (
    "workspace_shell",
    "artifact_workspace",
    "preview_workspace",
    "inspector_workspace",
)
EXPECTED_SURFACE_KINDS = (
    "shell",
    "artifact_selection",
    "preview",
    "inspector",
)
EXPECTED_SOURCE_REGISTRIES = (
    "multimodal_live_preview_registry",
    "multimodal_multi_preview_registry",
    "multimodal_interactive_canvas_registry",
    "nextjs_workstation_state",
    "nextjs_workstation_dashboard",
    "nextjs_workstation_shell",
    "nextjs_workspace_persistence",
)
EXPECTED_SOURCE_REFERENCES = (
    "multimodal_studio.MULTIMODAL_LIVE_PREVIEW_REGISTRY",
    "multimodal_studio.MULTIMODAL_MULTI_PREVIEW_REGISTRY",
    "multimodal_studio.MULTIMODAL_INTERACTIVE_CANVAS_REGISTRY",
    "clients.nextjs.workstation_state.buildWorkstationState",
    "clients.nextjs.workstation_dashboard.buildWorkstationDashboardModel",
    "clients.nextjs.workstation_shell.WorkstationShell",
    "clients.nextjs.workspace_persistence.createWorkspaceSessionRecord",
    "clients.nextjs.assistant_client.AssistantWorkspaceSnapshot",
)
EXPECTED_WORKSPACE_SURFACES = (
    "visual_workspace_shell",
    "artifact_selection_surface",
    "preview_workspace_surface",
    "inspector_workspace_surface",
    "workspace_dashboard_surface",
    "visual_context_surface",
    "workspace_boundary_panel",
)
REQUIRED_PROFILE_FIELDS = {
    "profile_id",
    "profile_name",
    "workspace_profile_kind",
    "workspace_surface_kind",
    "source_live_preview_profile_ids",
    "source_multi_preview_profile_ids",
    "source_interactive_canvas_profile_ids",
    "workspace_state_fields",
    "source_reference_ids",
    "route_applicability",
    "visual_workspace_surfaces",
    "advisory_outputs",
    "source_registries",
    "observability_surfaces",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "workspace_state_mutation_implemented",
    "persistent_storage_mutation_implemented",
    "rendering_execution_implemented",
    "provider_model_routing_implemented",
    "external_provider_calls_implemented",
    "networking_implemented",
    "retry_triggering_implemented",
    "generated_output_mutation_implemented",
    "active_studio_behavior_implemented",
    "serialization_version",
    "metadata_only",
}


class MultimodalStudioVisualWorkspaceTests(unittest.TestCase):
    def test_visual_workspace_registry_covers_expected_sources(self) -> None:
        registry = multimodal_visual_workspace_registry()

        self.assertEqual(registry.role, "multimodal_visual_workspace_registry")
        self.assertEqual(
            registry.serialization_version,
            "multimodal_visual_workspace_registry.v1",
        )
        self.assertEqual(registry.profile_ids, EXPECTED_PROFILE_IDS)
        self.assertEqual(registry.workspace_profile_kinds, EXPECTED_PROFILE_KINDS)
        self.assertEqual(registry.workspace_surface_kinds, EXPECTED_SURFACE_KINDS)
        self.assertEqual(
            registry.live_preview_profile_ids,
            multimodal_live_preview_registry().profile_ids,
        )
        self.assertEqual(
            registry.multi_preview_profile_ids,
            multimodal_multi_preview_registry().profile_ids,
        )
        self.assertEqual(
            registry.interactive_canvas_profile_ids,
            multimodal_interactive_canvas_registry().profile_ids,
        )
        self.assertEqual(registry.route_names, tuple(RouteName))
        self.assertEqual(registry.profile_count, 4)
        self.assertEqual(registry.source_registries, EXPECTED_SOURCE_REGISTRIES)
        self.assertEqual(registry.source_reference_ids, EXPECTED_SOURCE_REFERENCES)
        self.assertEqual(
            registry.visual_workspace_surface_refs,
            EXPECTED_WORKSPACE_SURFACES,
        )
        self.assertIn("does not mutate workspace state", registry.authority_boundary)
        self.assertIn(
            "persistent storage behavior",
            registry.authority_boundary,
        )
        self.assertIn("workspace_state_mutation", registry.blocked_runtime_behaviors)
        self.assertIn(
            "persistent_storage_mutation",
            registry.blocked_runtime_behaviors,
        )
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.workspace_state_mutation_implemented)
        self.assertFalse(registry.persistent_storage_mutation_implemented)
        self.assertFalse(registry.rendering_execution_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.external_provider_calls_implemented)
        self.assertFalse(registry.networking_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.active_studio_behavior_implemented)

    def test_visual_workspace_profiles_are_passive_and_source_aligned(self) -> None:
        registry = multimodal_visual_workspace_registry()
        known_routes = set(registry.route_names)
        known_live_profiles = set(registry.live_preview_profile_ids)
        known_multi_profiles = set(registry.multi_preview_profile_ids)
        known_canvas_profiles = set(registry.interactive_canvas_profile_ids)
        known_surfaces = set(registry.visual_workspace_surface_refs)
        known_source_references = set(registry.source_reference_ids)
        covered_source_references: set[str] = set()

        for profile in registry.visual_workspace_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(
                profile.serialization_version,
                "multimodal_visual_workspace_profile.v1",
            )
            self.assertEqual(profile.source_registries, registry.source_registries)
            self.assertEqual(
                profile.observability_surfaces,
                registry.observability_surfaces,
            )
            self.assertTrue(set(profile.route_applicability).issubset(known_routes))
            self.assertTrue(
                set(profile.source_live_preview_profile_ids).issubset(
                    known_live_profiles
                )
            )
            self.assertTrue(
                set(profile.source_multi_preview_profile_ids).issubset(
                    known_multi_profiles
                )
            )
            self.assertTrue(
                set(profile.source_interactive_canvas_profile_ids).issubset(
                    known_canvas_profiles
                )
            )
            self.assertTrue(
                set(profile.visual_workspace_surfaces).issubset(known_surfaces)
            )
            self.assertTrue(
                set(profile.source_reference_ids).issubset(known_source_references)
            )
            covered_source_references.update(profile.source_reference_ids)
            self.assertIn(
                "workspace_state_mutation",
                profile.blocked_runtime_behaviors,
            )
            self.assertTrue(profile.metadata_only)
            self.assertFalse(profile.workspace_state_mutation_implemented)
            self.assertFalse(profile.persistent_storage_mutation_implemented)
            self.assertFalse(profile.rendering_execution_implemented)
            self.assertFalse(profile.provider_model_routing_implemented)
            self.assertFalse(profile.external_provider_calls_implemented)
            self.assertFalse(profile.networking_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)
            self.assertFalse(profile.active_studio_behavior_implemented)

        self.assertEqual(covered_source_references, known_source_references)

    def test_visual_workspace_lookup_helpers_are_stable(self) -> None:
        profile = multimodal_visual_workspace_profile_by_id(
            "shell_visual_workspace"
        )
        missing_profile = multimodal_visual_workspace_profile_by_id(
            "missing_profile"
        )
        preview_profiles = multimodal_visual_workspace_profiles_for_surface_kind(
            "preview"
        )
        route_profiles = multimodal_visual_workspace_profiles_for_route(
            RouteName.PREVIEW
        )
        input_boundary_profiles = (
            multimodal_visual_workspace_profiles_for_interactive_canvas_profile(
                "input_boundary_interactive_canvas"
            )
        )

        self.assertIsNone(missing_profile)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.workspace_profile_kind, "workspace_shell")
        self.assertIn("selection", profile.workspace_state_fields)
        self.assertIn("no_workspace_state_mutation_notice", profile.advisory_outputs)
        self.assertEqual(
            tuple(item.profile_id for item in preview_profiles),
            ("preview_visual_workspace",),
        )
        self.assertEqual(
            tuple(item.profile_id for item in route_profiles),
            EXPECTED_PROFILE_IDS,
        )
        self.assertEqual(
            tuple(item.profile_id for item in input_boundary_profiles),
            (
                "shell_visual_workspace",
                "preview_visual_workspace",
                "inspector_visual_workspace",
            ),
        )

    def test_registry_rejects_mismatched_visual_workspace_metadata(self) -> None:
        registry = multimodal_visual_workspace_registry()
        first_profile = registry.visual_workspace_profiles[0]
        duplicate_profile = first_profile.model_copy(
            update={"profile_name": "Duplicate Visual Workspace"}
        )
        unknown_canvas_profile = first_profile.model_copy(
            update={
                "source_interactive_canvas_profile_ids": (
                    "unknown_interactive_canvas",
                )
            }
        )

        duplicate_kwargs = _registry_kwargs(registry)
        duplicate_kwargs["visual_workspace_profiles"] = (
            first_profile,
            duplicate_profile,
        ) + registry.visual_workspace_profiles[2:]
        with self.assertRaisesRegex(ValueError, "profile_ids must be unique"):
            MultimodalVisualWorkspaceRegistry(**duplicate_kwargs)

        unknown_canvas_kwargs = _registry_kwargs(registry)
        unknown_canvas_kwargs["visual_workspace_profiles"] = (
            unknown_canvas_profile,
        ) + registry.visual_workspace_profiles[1:]
        with self.assertRaisesRegex(
            ValueError,
            "source_interactive_canvas_profile_ids",
        ):
            MultimodalVisualWorkspaceRegistry(**unknown_canvas_kwargs)

    def test_visual_workspace_metadata_does_not_change_provider_factory(self) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
        )

        multimodal_visual_workspace_registry()
        provider = build_generation_provider(settings)

        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")


def _registry_kwargs(
    registry: MultimodalVisualWorkspaceRegistry,
) -> dict[str, object]:
    return {
        "visual_workspace_profiles": registry.visual_workspace_profiles,
        "profile_ids": registry.profile_ids,
        "workspace_profile_kinds": registry.workspace_profile_kinds,
        "workspace_surface_kinds": registry.workspace_surface_kinds,
        "live_preview_profile_ids": registry.live_preview_profile_ids,
        "multi_preview_profile_ids": registry.multi_preview_profile_ids,
        "interactive_canvas_profile_ids": registry.interactive_canvas_profile_ids,
        "route_names": registry.route_names,
        "profile_count": registry.profile_count,
        "source_registries": registry.source_registries,
        "source_reference_ids": registry.source_reference_ids,
        "visual_workspace_surface_refs": registry.visual_workspace_surface_refs,
        "observability_surfaces": registry.observability_surfaces,
    }


if __name__ == "__main__":
    unittest.main()
