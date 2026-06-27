import unittest

from creative_coding_assistant.orchestration import (
    AgentCapabilityAlignmentRegistry,
    agent_capability_alignment_by_agent_id,
    agent_capability_alignment_registry,
    agent_role_registry,
)

REQUIRED_ALIGNMENT_FIELDS = {
    "agent_id",
    "role_id",
    "role_name",
    "capability_ids",
    "source_role_registry",
    "source_orchestration_registries",
    "alignment_boundary",
    "blocked_runtime_behaviors",
    "capabilities_activated",
    "runtime_work_routing_implemented",
    "prompt_changes_implemented",
    "serialization_version",
    "metadata_only",
}

EXPECTED_CAPABILITIES = (
    "dynamic_agent_routing",
    "blackboard_memory",
    "shared_context_view",
    "dependency_graph",
    "parallel_scheduling",
    "agent_coordination",
    "agent_debate",
    "consensus_builder",
)


class AgentCapabilityAlignmentTests(unittest.TestCase):
    def test_alignment_registry_covers_all_v4_1_roles(self) -> None:
        alignment = agent_capability_alignment_registry()
        roles = agent_role_registry()

        self.assertEqual(alignment.role, "agent_capability_alignment_registry")
        self.assertEqual(
            alignment.serialization_version,
            "agent_capability_alignment_registry.v1",
        )
        self.assertEqual(alignment.agent_ids, roles.agent_ids)
        self.assertEqual(alignment.role_ids, roles.role_ids)
        self.assertEqual(alignment.alignment_count, 12)
        self.assertEqual(alignment.capability_ids, EXPECTED_CAPABILITIES)
        self.assertIn("does not activate capabilities", alignment.authority_boundary)
        self.assertFalse(alignment.capabilities_activated)
        self.assertFalse(alignment.runtime_work_routing_implemented)
        self.assertFalse(alignment.prompt_changes_implemented)
        self.assertTrue(alignment.metadata_only)

    def test_alignment_profiles_are_export_only_metadata(self) -> None:
        registry = agent_capability_alignment_registry()

        for profile in registry.alignments:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_ALIGNMENT_FIELDS)
            self.assertEqual(
                profile.serialization_version,
                "agent_capability_alignment.v1",
            )
            self.assertIn("dynamic_agent_routing", profile.capability_ids)
            self.assertIn("blackboard_memory", profile.capability_ids)
            self.assertIn("agent_routing_registry", profile.source_orchestration_registries)
            self.assertIn("capability_activation", profile.blocked_runtime_behaviors)
            self.assertFalse(profile.capabilities_activated)
            self.assertFalse(profile.runtime_work_routing_implemented)
            self.assertFalse(profile.prompt_changes_implemented)
            self.assertTrue(profile.metadata_only)

    def test_lookup_maps_role_specific_capabilities(self) -> None:
        planner = agent_capability_alignment_by_agent_id("planner_agent")
        research = agent_capability_alignment_by_agent_id("research_agent")

        self.assertIsNone(agent_capability_alignment_by_agent_id("missing_agent"))
        self.assertIsNotNone(planner)
        self.assertIsNotNone(research)
        assert planner is not None
        assert research is not None
        self.assertIn("consensus_builder", planner.capability_ids)
        self.assertIn("agent_debate", planner.capability_ids)
        self.assertNotIn("consensus_builder", research.capability_ids)
        self.assertNotIn("agent_debate", research.capability_ids)

    def test_registry_rejects_incomplete_or_mismatched_alignment(self) -> None:
        registry = agent_capability_alignment_registry()
        incomplete = registry.alignments[0].model_copy(
            update={"capability_ids": ("dynamic_agent_routing",)}
        )

        with self.assertRaisesRegex(ValueError, "base orchestration capabilities"):
            AgentCapabilityAlignmentRegistry(
                alignments=(incomplete,) + registry.alignments[1:],
                agent_ids=registry.agent_ids,
                role_ids=registry.role_ids,
                capability_ids=registry.capability_ids,
                alignment_count=registry.alignment_count,
                source_registries=registry.source_registries,
            )

        with self.assertRaisesRegex(ValueError, "agent_ids must match"):
            AgentCapabilityAlignmentRegistry(
                alignments=registry.alignments,
                agent_ids=("other_agent",) + registry.agent_ids[1:],
                role_ids=registry.role_ids,
                capability_ids=registry.capability_ids,
                alignment_count=registry.alignment_count,
                source_registries=registry.source_registries,
            )

    def test_alignment_does_not_activate_runtime_capabilities(self) -> None:
        registry = agent_capability_alignment_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for profile in registry.alignments
                    for field in (
                        profile.agent_id,
                        profile.alignment_boundary,
                        *profile.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "activate_capability",
            "runtime_router",
            "prompt_rewrite",
            "execute_orchestration",
            "provider_route",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
