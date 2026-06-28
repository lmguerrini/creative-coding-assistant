import unittest

from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    LocalModelRegistry,
    local_model_registry,
    local_model_surface_by_id,
    local_model_surfaces_for_runtime,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_SURFACE_IDS = (
    "ollama_chat_surface",
    "lm_studio_chat_surface",
    "llama_cpp_completion_surface",
    "local_transformers_multimodal_surface",
)
EXPECTED_RUNTIME_KINDS = (
    "ollama",
    "lm_studio",
    "llama_cpp",
    "local_transformers",
)
EXPECTED_SOURCE_REGISTRIES = (
    "settings_generation_provider_config",
    "generation_provider_factory",
    "generation_provider_contract",
    "agent_routing_registry",
    "hybrid_agentic_workflow_registry",
)
EXPECTED_STUDIO_SURFACES = (
    "local_model_catalog",
    "local_model_readiness_inspector",
    "provider_selection_metadata",
    "execution_simulator_metadata",
    "local_cloud_comparison_metadata",
)
REQUIRED_SURFACE_FIELDS = {
    "surface_id",
    "surface_name",
    "runtime_kind",
    "execution_surface",
    "capability_band",
    "context_window_band",
    "latency_posture",
    "cost_posture",
    "privacy_posture",
    "route_applicability",
    "supported_payloads",
    "readiness_signals",
    "studio_surface_refs",
    "source_registries",
    "observability_surfaces",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "local_runtime_discovery_implemented",
    "local_provider_execution_implemented",
    "provider_model_routing_implemented",
    "automatic_model_selection_implemented",
    "external_provider_calls_implemented",
    "retry_triggering_implemented",
    "generated_output_mutation_implemented",
    "persistent_replay_storage_implemented",
    "serialization_version",
    "metadata_only",
}


class HybridStudioLocalModelTests(unittest.TestCase):
    def test_local_model_registry_covers_expected_surfaces(self) -> None:
        registry = local_model_registry()

        self.assertEqual(registry.role, "local_model_registry")
        self.assertEqual(
            registry.serialization_version,
            "local_model_registry.v1",
        )
        self.assertEqual(registry.surface_ids, EXPECTED_SURFACE_IDS)
        self.assertEqual(registry.runtime_kinds, EXPECTED_RUNTIME_KINDS)
        self.assertEqual(registry.profile_count, 4)
        self.assertEqual(registry.route_names, tuple(RouteName))
        self.assertEqual(registry.source_registries, EXPECTED_SOURCE_REGISTRIES)
        self.assertEqual(registry.studio_surface_refs, EXPECTED_STUDIO_SURFACES)
        self.assertIn("does not discover installed models", registry.authority_boundary)
        self.assertIn("provider_or_model_routing", registry.blocked_runtime_behaviors)
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.local_runtime_discovery_implemented)
        self.assertFalse(registry.local_provider_execution_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.automatic_model_selection_implemented)
        self.assertFalse(registry.external_provider_calls_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.persistent_replay_storage_implemented)

    def test_local_model_surfaces_are_passive_metadata(self) -> None:
        registry = local_model_registry()
        known_routes = set(registry.route_names)
        known_studio_surfaces = set(registry.studio_surface_refs)

        for surface in registry.model_surfaces:
            dumped = surface.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_SURFACE_FIELDS)
            self.assertEqual(surface.serialization_version, "local_model_surface.v1")
            self.assertEqual(surface.source_registries, registry.source_registries)
            self.assertEqual(
                surface.observability_surfaces,
                registry.observability_surfaces,
            )
            self.assertTrue(set(surface.route_applicability).issubset(known_routes))
            self.assertTrue(
                set(surface.studio_surface_refs).issubset(known_studio_surfaces)
            )
            self.assertEqual(surface.cost_posture, "local_hardware_only")
            self.assertEqual(surface.privacy_posture, "local_operator_boundary")
            self.assertIn("local_provider_execution", surface.blocked_runtime_behaviors)
            self.assertTrue(surface.metadata_only)
            self.assertFalse(surface.local_runtime_discovery_implemented)
            self.assertFalse(surface.local_provider_execution_implemented)
            self.assertFalse(surface.provider_model_routing_implemented)
            self.assertFalse(surface.automatic_model_selection_implemented)
            self.assertFalse(surface.external_provider_calls_implemented)
            self.assertFalse(surface.retry_triggering_implemented)
            self.assertFalse(surface.generated_output_mutation_implemented)
            self.assertFalse(surface.persistent_replay_storage_implemented)

    def test_local_model_lookup_helpers_are_stable(self) -> None:
        ollama_surface = local_model_surface_by_id("ollama_chat_surface")
        missing_surface = local_model_surface_by_id("missing_surface")
        llama_cpp_surfaces = local_model_surfaces_for_runtime("llama_cpp")
        missing_runtime_surfaces = local_model_surfaces_for_runtime("missing")

        self.assertIsNone(missing_surface)
        self.assertEqual(missing_runtime_surfaces, ())
        self.assertIsNotNone(ollama_surface)
        assert ollama_surface is not None
        self.assertEqual(ollama_surface.runtime_kind, "ollama")
        self.assertEqual(ollama_surface.execution_surface, "localhost_http")
        self.assertEqual(ollama_surface.route_applicability, tuple(RouteName))
        self.assertEqual(
            tuple(surface.surface_id for surface in llama_cpp_surfaces),
            ("llama_cpp_completion_surface",),
        )

    def test_registry_rejects_mismatched_sources_or_surfaces(self) -> None:
        registry = local_model_registry()
        first_surface = registry.model_surfaces[0]
        duplicate_surface = first_surface.model_copy(
            update={"surface_name": "Duplicate Surface"}
        )
        mismatched_source_surface = first_surface.model_copy(
            update={
                "source_registries": (
                    "settings_generation_provider_config",
                    "generation_provider_factory",
                    "generation_provider_contract",
                    "agent_routing_registry",
                    "unknown_registry",
                )
            }
        )

        with self.assertRaisesRegex(ValueError, "surface_ids must be unique"):
            LocalModelRegistry(
                model_surfaces=(first_surface, duplicate_surface)
                + registry.model_surfaces[2:],
                surface_ids=registry.surface_ids,
                runtime_kinds=registry.runtime_kinds,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                studio_surface_refs=registry.studio_surface_refs,
                observability_surfaces=registry.observability_surfaces,
            )

        with self.assertRaisesRegex(ValueError, "surface_ids must match"):
            LocalModelRegistry(
                model_surfaces=registry.model_surfaces,
                surface_ids=("other_surface",) + registry.surface_ids[1:],
                runtime_kinds=registry.runtime_kinds,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                studio_surface_refs=registry.studio_surface_refs,
                observability_surfaces=registry.observability_surfaces,
            )

        with self.assertRaisesRegex(ValueError, "source_registries must match"):
            LocalModelRegistry(
                model_surfaces=(mismatched_source_surface,)
                + registry.model_surfaces[1:],
                surface_ids=registry.surface_ids,
                runtime_kinds=registry.runtime_kinds,
                route_names=registry.route_names,
                profile_count=4,
                source_registries=registry.source_registries,
                studio_surface_refs=registry.studio_surface_refs,
                observability_surfaces=registry.observability_surfaces,
            )

    def test_local_model_metadata_does_not_change_provider_factory(self) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
        )

        local_model_registry()
        provider = build_generation_provider(settings)

        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_local_model_metadata_does_not_declare_active_execution(self) -> None:
        registry = local_model_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.source_registries,
                *registry.studio_surface_refs,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for surface in registry.model_surfaces
                    for field in (
                        surface.surface_id,
                        surface.runtime_kind,
                        surface.execution_surface,
                        surface.capability_band,
                        *surface.readiness_signals,
                        *surface.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "route_provider",
            "provider_route",
            "execute_provider",
            "start_local_runtime",
            "auto_select_model",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
