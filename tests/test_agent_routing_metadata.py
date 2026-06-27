import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import OpenAIGenerationProvider, build_generation_provider
from creative_coding_assistant.orchestration import (
    AgentRoutingRegistry,
    agent_contract_registry,
    agent_routing_profile_by_agent_id,
    agent_routing_profiles_for_route,
    agent_routing_registry,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_AGENT_IDS = (
    "planner_agent",
    "research_agent",
    "style_agent",
    "runtime_agent",
    "artifact_agent",
    "art_direction_agent",
    "aesthetic_critic_agent",
    "narrative_symbolic_agent",
    "creative_curator_agent",
    "critic_agent",
    "refiner_agent",
    "final_synthesizer_agent",
)

REQUIRED_PROFILE_FIELDS = {
    "agent_id",
    "role_id",
    "routing_stage",
    "routing_authority",
    "priority_band",
    "route_candidates",
    "routing_inputs",
    "routing_outputs",
    "decision_signals",
    "required_metadata_inputs",
    "produced_metadata_outputs",
    "decision_boundary",
    "source_contract_registries",
    "source_agent_contract_version",
    "blocked_runtime_behaviors",
    "provider_model_routing_implemented",
    "workflow_routing_implemented",
    "agent_execution_implemented",
    "retry_triggering_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "metadata_only",
}


class AgentRoutingMetadataTests(unittest.TestCase):
    def test_routing_registry_covers_all_agent_contracts(self) -> None:
        registry = agent_routing_registry()
        contract_registry = agent_contract_registry()

        self.assertEqual(registry.role, "agent_routing_registry")
        self.assertEqual(
            registry.serialization_version,
            "agent_routing_registry.v1",
        )
        self.assertEqual(registry.agent_ids, EXPECTED_AGENT_IDS)
        self.assertEqual(registry.agent_ids, contract_registry.agent_ids)
        self.assertEqual(registry.profile_count, 12)
        self.assertEqual(registry.route_names, tuple(RouteName))
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.workflow_routing_implemented)
        self.assertFalse(registry.agent_execution_implemented)
        self.assertIn("does not execute agents", registry.authority_boundary)
        self.assertIn(
            "provider_or_model_routing",
            registry.blocked_runtime_behaviors,
        )

    def test_routing_profiles_map_inputs_outputs_and_boundaries(self) -> None:
        registry = agent_routing_registry()

        for profile in registry.routing_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(profile.serialization_version, "agent_routing.v1")
            self.assertEqual(profile.routing_inputs, registry.routing_inputs)
            self.assertEqual(profile.routing_outputs, registry.routing_outputs)
            self.assertTrue(profile.route_candidates)
            self.assertTrue(profile.decision_signals)
            self.assertTrue(profile.required_metadata_inputs)
            self.assertTrue(profile.produced_metadata_outputs)
            self.assertIn("provider_or_model_routing", profile.blocked_runtime_behaviors)
            self.assertIn("workflow_routing_change", profile.blocked_runtime_behaviors)
            self.assertIn("must not instantiate agents", profile.decision_boundary)
            self.assertTrue(profile.metadata_only)
            self.assertFalse(profile.provider_model_routing_implemented)
            self.assertFalse(profile.workflow_routing_implemented)
            self.assertFalse(profile.agent_execution_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)

    def test_profile_lookup_and_route_filtering_are_passive(self) -> None:
        runtime_profile = agent_routing_profile_by_agent_id("runtime_agent")
        missing_profile = agent_routing_profile_by_agent_id("missing_agent")
        generate_profiles = agent_routing_profiles_for_route(RouteName.GENERATE)
        preview_profiles = agent_routing_profiles_for_route("preview")

        self.assertIsNone(missing_profile)
        self.assertIsNotNone(runtime_profile)
        assert runtime_profile is not None
        self.assertEqual(runtime_profile.priority_band, "execution_context")
        self.assertIn(RouteName.PREVIEW, runtime_profile.route_candidates)
        self.assertIn(
            "runtime_agent",
            tuple(profile.agent_id for profile in preview_profiles),
        )
        self.assertEqual(generate_profiles[0].agent_id, "planner_agent")
        self.assertIn(
            "final_synthesizer_agent",
            tuple(profile.agent_id for profile in generate_profiles),
        )

    def test_registry_rejects_mismatched_profiles(self) -> None:
        registry = agent_routing_registry()
        first_profile = registry.routing_profiles[0]
        duplicate_profile = first_profile.model_copy(update={"role_id": "duplicate"})

        with self.assertRaisesRegex(ValueError, "agent_ids must be unique"):
            AgentRoutingRegistry(
                routing_profiles=(
                    first_profile,
                    duplicate_profile,
                )
                + registry.routing_profiles[2:],
                agent_ids=registry.agent_ids,
                route_names=registry.route_names,
                profile_count=12,
                routing_inputs=registry.routing_inputs,
                routing_outputs=registry.routing_outputs,
                source_registries=registry.source_registries,
            )

        with self.assertRaisesRegex(ValueError, "agent_ids must match"):
            AgentRoutingRegistry(
                routing_profiles=registry.routing_profiles,
                agent_ids=("other_agent",) + registry.agent_ids[1:],
                route_names=registry.route_names,
                profile_count=12,
                routing_inputs=registry.routing_inputs,
                routing_outputs=registry.routing_outputs,
                source_registries=registry.source_registries,
            )

    def test_agent_routing_metadata_does_not_change_request_routing(self) -> None:
        request = AssistantRequest(
            query="Generate a p5.js particle system.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)

        agent_routing_registry()
        next_decision = route_request(request)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(next_decision.route, RouteName.GENERATE)
        self.assertEqual(next_decision.mode, AssistantMode.GENERATE)
        self.assertNotIn("agent_routing_registry", next_decision.model_dump_json())

    def test_agent_routing_metadata_does_not_change_provider_model_selection(self) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        agent_routing_registry()
        provider = build_generation_provider(settings)

        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")


if __name__ == "__main__":
    unittest.main()
