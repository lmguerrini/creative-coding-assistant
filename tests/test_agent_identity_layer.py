import unittest

from creative_coding_assistant.orchestration import (
    AgentIdentityRegistry,
    agent_identities_by_role_family,
    agent_identity_by_id,
    agent_identity_registry,
)

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

EXPECTED_ROLE_FAMILIES = (
    "planning",
    "research",
    "style",
    "runtime",
    "artifact",
    "art_direction",
    "critique",
    "narrative",
    "curation",
    "refinement",
    "synthesis",
)

REQUIRED_IDENTITY_FIELDS = {
    "agent_id",
    "agent_name",
    "role_family",
    "purpose",
    "capability_class",
    "authority_scope",
    "visibility",
    "identity_version",
    "contract_hook",
    "persistent_user_identity",
    "hidden_state",
    "blocked_runtime_behaviors",
    "serialization_version",
    "metadata_only",
}


class AgentIdentityLayerTests(unittest.TestCase):
    def test_registry_exposes_stable_agent_identities(self) -> None:
        registry = agent_identity_registry()

        self.assertEqual(registry.role, "agent_identity_registry")
        self.assertEqual(
            registry.serialization_version,
            "agent_identity_registry.v1",
        )
        self.assertEqual(registry.agent_ids, EXPECTED_AGENT_IDS)
        self.assertEqual(registry.identity_count, 12)
        self.assertEqual(registry.role_families, EXPECTED_ROLE_FAMILIES)
        self.assertEqual(registry.identity_version, "v4.1")
        self.assertTrue(registry.metadata_only)
        self.assertIn("does not create persistent user", registry.authority_boundary)
        self.assertEqual(
            {identity.agent_id for identity in registry.identities},
            set(EXPECTED_AGENT_IDS),
        )

    def test_identity_metadata_is_serializable_and_inspectable(self) -> None:
        registry = agent_identity_registry()

        for identity in registry.identities:
            dumped = identity.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_IDENTITY_FIELDS)
            self.assertEqual(identity.identity_version, "v4.1")
            self.assertEqual(identity.serialization_version, "agent_identity.v1")
            self.assertEqual(identity.authority_scope, "metadata_only_contract")
            self.assertEqual(
                identity.visibility,
                "inspectable_registry_metadata",
            )
            self.assertTrue(identity.contract_hook.endswith("_contract"))
            self.assertTrue(identity.metadata_only)
            self.assertFalse(identity.persistent_user_identity)
            self.assertFalse(identity.hidden_state)
            self.assertIn(
                "hidden_agent_state",
                identity.blocked_runtime_behaviors,
            )
            self.assertIn(
                "persistent_user_identity",
                identity.blocked_runtime_behaviors,
            )

    def test_identity_lookup_is_stable(self) -> None:
        planner = agent_identity_by_id("planner_agent")
        missing = agent_identity_by_id("missing")

        self.assertIsNone(missing)
        self.assertIsNotNone(planner)
        assert planner is not None
        self.assertEqual(planner.agent_name, "Planner Agent")
        self.assertEqual(planner.role_family, "planning")
        self.assertEqual(planner.capability_class, "planning_strategy")
        self.assertEqual(planner.contract_hook, "planner_agent_contract")

    def test_role_family_lookup_preserves_registry_order(self) -> None:
        critique_identities = agent_identities_by_role_family("critique")

        self.assertEqual(
            tuple(identity.agent_id for identity in critique_identities),
            ("aesthetic_critic_agent", "critic_agent"),
        )
        self.assertEqual(agent_identities_by_role_family("missing"), ())

    def test_registry_rejects_mismatched_or_duplicate_identity_metadata(
        self,
    ) -> None:
        registry = agent_identity_registry()
        first_identity = registry.identities[0]
        duplicate_identity = first_identity.model_copy(
            update={"agent_name": "Duplicate Planner Agent"}
        )

        with self.assertRaisesRegex(ValueError, "agent_ids must be unique"):
            AgentIdentityRegistry(
                identities=(first_identity, duplicate_identity)
                + registry.identities[2:],
                agent_ids=registry.agent_ids,
                identity_count=12,
                role_families=registry.role_families,
            )

        with self.assertRaisesRegex(ValueError, "agent_ids must match identities"):
            AgentIdentityRegistry(
                identities=registry.identities,
                agent_ids=("other_agent",) + registry.agent_ids[1:],
                identity_count=12,
                role_families=registry.role_families,
            )

    def test_identity_layer_does_not_declare_active_runtime_behavior(self) -> None:
        registry = agent_identity_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for identity in registry.identities
                    for field in (
                        identity.purpose,
                        identity.contract_hook,
                        *identity.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "execute_provider",
            "autonomous_retry",
            "runtime_auto_selection",
            "persistent_user_profile",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
