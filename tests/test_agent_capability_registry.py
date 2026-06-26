import unittest

from creative_coding_assistant.orchestration import (
    agent_capability_by_id,
    agent_capability_registry,
)

EXPECTED_CAPABILITY_IDS = (
    "v4_planner_agent",
    "v4_artifact_agent",
    "v4_runtime_agent",
    "v4_agent_router",
    "v4_agentic_studio",
    "adaptive_multi_agent_escalation",
)

REQUIRED_PROFILE_FIELDS = {
    "capability_id",
    "capability_name",
    "capability_stage",
    "authority_boundary",
    "source_contract_registries",
    "required_metadata_sources",
    "advisory_outputs",
    "readiness_signals",
    "future_agent_hooks",
    "blocked_runtime_behaviors",
    "serialization_version",
}


class AgentCapabilityRegistryTests(unittest.TestCase):
    def test_registry_exposes_metadata_only_capabilities(self) -> None:
        registry = agent_capability_registry()

        self.assertEqual(registry.role, "agent_capability_registry")
        self.assertEqual(registry.capability_ids, EXPECTED_CAPABILITY_IDS)
        self.assertEqual(registry.capability_count, 6)
        self.assertTrue(registry.metadata_only)
        self.assertEqual(
            registry.source_contract_registries,
            (
                "artifact_engine_contract_registry",
                "evaluation_engine_contract_registry",
                "workstation_engine_contract_registry",
            ),
        )
        self.assertIn("does not create agents", registry.authority_boundary)
        self.assertEqual(
            {profile.capability_id for profile in registry.capabilities},
            set(EXPECTED_CAPABILITY_IDS),
        )

        for profile in registry.capabilities:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(profile.serialization_version, "agent_capability.v1")
            self.assertTrue(profile.source_contract_registries)
            self.assertTrue(profile.required_metadata_sources)
            self.assertTrue(profile.advisory_outputs)
            self.assertTrue(profile.readiness_signals)
            self.assertTrue(profile.future_agent_hooks)
            self.assertIn(
                "provider_or_model_routing",
                profile.blocked_runtime_behaviors,
            )
            self.assertIn(
                "generated_output_modification",
                profile.blocked_runtime_behaviors,
            )
            self.assertIn("does not create agents", profile.authority_boundary)

    def test_capability_lookup_is_stable(self) -> None:
        studio = agent_capability_by_id("v4_agentic_studio")
        missing = agent_capability_by_id("missing")

        self.assertIsNone(missing)
        self.assertIsNotNone(studio)
        assert studio is not None
        self.assertEqual(studio.capability_stage, "v4_agent_readiness")
        self.assertIn(
            "workstation_engine_contract_registry",
            studio.source_contract_registries,
        )
        self.assertIn("operator_review_surface", studio.advisory_outputs)

    def test_registry_serializes_for_future_metadata_consumers(self) -> None:
        dumped = agent_capability_registry().model_dump(mode="json")

        self.assertEqual(
            dumped["serialization_version"],
            "agent_capability_registry.v1",
        )
        self.assertEqual(dumped["capability_ids"], list(EXPECTED_CAPABILITY_IDS))
        self.assertEqual(len(dumped["capabilities"]), 6)
        self.assertTrue(dumped["metadata_only"])
        self.assertEqual(
            dumped["capabilities"][0]["capability_id"],
            "v4_planner_agent",
        )

    def test_registry_does_not_declare_runtime_behaviors(self) -> None:
        registry = agent_capability_registry()

        for profile in registry.capabilities:
            combined_text = " ".join(
                (
                    profile.authority_boundary,
                    *profile.advisory_outputs,
                    *profile.future_agent_hooks,
                    *profile.blocked_runtime_behaviors,
                )
            )
            self.assertNotIn("execute_provider", combined_text)
            self.assertNotIn("autonomous_retry", combined_text)
            self.assertNotIn("runtime_auto_selection", combined_text)


if __name__ == "__main__":
    unittest.main()
