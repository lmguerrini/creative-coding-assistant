import unittest

from creative_coding_assistant.orchestration import (
    CreativeIdentityLayerPlan,
    build_creative_cognition_layer,
    build_creative_identity_layer,
    creative_identity_profile_by_id,
    creative_identity_profiles_for_agent,
    creative_identity_profiles_for_layer,
)
from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    COGNITIVE_OS_LAYER_ORDER,
)


class CreativeIdentityLayerTests(unittest.TestCase):
    def test_creative_identity_layer_builds_read_only_profiles(self) -> None:
        cognition_layer = build_creative_cognition_layer()
        layer = build_creative_identity_layer(
            creative_cognition_layer=cognition_layer,
        )

        self.assertEqual(layer.role, "creative_identity_layer")
        self.assertEqual(layer.serialization_version, "creative_identity_layer.v1")
        self.assertEqual(layer.creative_cognition_layer_role, cognition_layer.role)
        self.assertEqual(
            layer.cognitive_governance_layer_role,
            "cognitive_governance_layer",
        )
        self.assertEqual(layer.layer_order, COGNITIVE_OS_LAYER_ORDER)
        self.assertEqual(layer.capabilities, COGNITIVE_OS_CAPABILITIES)
        self.assertEqual(layer.capability_ids, cognition_layer.capability_ids)
        self.assertEqual(layer.capability_count, 6)
        self.assertEqual(layer.source_cognition_ids, cognition_layer.cognition_ids)
        self.assertEqual(layer.source_cognition_count, 6)
        self.assertEqual(
            layer.source_governance_ids,
            cognition_layer.source_governance_ids,
        )
        self.assertEqual(layer.source_governance_count, 6)
        self.assertEqual(layer.source_planning_ids, cognition_layer.source_planning_ids)
        self.assertEqual(layer.source_planning_count, 6)
        self.assertEqual(
            layer.source_reasoning_ids,
            cognition_layer.source_reasoning_ids,
        )
        self.assertEqual(layer.source_reasoning_count, 6)
        self.assertEqual(layer.source_profile_ids, cognition_layer.source_profile_ids)
        self.assertEqual(layer.source_profile_count, 6)
        self.assertEqual(layer.source_state_ids, cognition_layer.source_state_ids)
        self.assertEqual(layer.source_state_count, 6)
        self.assertEqual(len(layer.identity_profiles), 6)
        self.assertEqual(layer.identity_count, 6)
        self.assertEqual(layer.linked_agent_ids, cognition_layer.linked_agent_ids)
        self.assertEqual(
            layer.covered_roadmap_items,
            ("Creative Identity Layer",),
        )
        self.assertEqual(layer.covered_roadmap_item_count, 1)
        self.assertEqual(layer.cross_cutting_contracts, COGNITIVE_OS_CONTRACTS)
        self.assertEqual(
            layer.blocked_runtime_behaviors,
            COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        self.assertTrue(layer.creative_identity_layer_implemented)
        self.assertTrue(layer.creative_cognition_layer_integrated)
        self.assertTrue(layer.identity_profile_contract_implemented)
        self.assertTrue(layer.identity_dependency_traceability_implemented)
        self.assertTrue(layer.identity_governance_contract_implemented)
        self.assertTrue(layer.identity_explainability_contract_implemented)
        self.assertFalse(layer.identity_persistence_implemented)
        self.assertFalse(layer.identity_mutation_implemented)
        self.assertFalse(layer.personalized_runtime_behavior_implemented)
        self.assertFalse(layer.prompt_mutation_implemented)
        self.assertFalse(layer.memory_mutation_implemented)
        self.assertFalse(layer.retrieval_mutation_implemented)
        self.assertFalse(layer.storage_mutation_implemented)
        self.assertFalse(layer.provider_model_routing_implemented)
        self.assertFalse(layer.provider_execution_implemented)
        self.assertFalse(layer.generated_output_mutation_implemented)
        self.assertFalse(layer.runtime_evolution_implemented)
        self.assertFalse(layer.persisted_identity_ids)
        self.assertFalse(layer.mutated_identity_ids)
        self.assertFalse(layer.personalized_identity_ids)
        self.assertFalse(layer.emitted_hitl_request_ids)
        self.assertTrue(layer.advisory_only)

    def test_creative_identity_lookup_helpers_are_layer_and_agent_aware(
        self,
    ) -> None:
        layer = build_creative_identity_layer()

        core_profile = creative_identity_profile_by_id(
            "creative_identity::v6_6_cognitive_core",
            layer,
        )
        self.assertIsNotNone(core_profile)
        assert core_profile is not None
        self.assertEqual(core_profile.capability_name, "V6.6 Cognitive Core")
        self.assertEqual(core_profile.cognitive_layer, "cognitive_core")
        self.assertIn("planner_agent", core_profile.linked_agent_ids)
        self.assertEqual(
            core_profile.cognition_id,
            "creative_cognition::v6_6_cognitive_core",
        )
        self.assertFalse(core_profile.persistent_identity_storage_authorized)
        self.assertIn("persist identity", core_profile.governance_contracts[0])

        research_profiles = creative_identity_profiles_for_layer("research", layer)
        self.assertEqual(len(research_profiles), 1)
        self.assertEqual(
            research_profiles[0].capability_id,
            "v6_4_autonomous_research",
        )

        planner_profiles = creative_identity_profiles_for_agent(
            "planner_agent",
            layer,
        )
        self.assertEqual(
            tuple(profile.capability_id for profile in planner_profiles),
            ("v6_6_cognitive_core",),
        )
        self.assertIsNone(creative_identity_profile_by_id("missing", layer))

    def test_creative_identity_layer_rejects_persistence_and_drift(self) -> None:
        layer = build_creative_identity_layer()
        payload = layer.model_dump(mode="json")
        payload["identity_ids"] = ("missing",) + tuple(payload["identity_ids"][1:])

        with self.assertRaisesRegex(ValueError, "identity_ids must match"):
            CreativeIdentityLayerPlan(**payload)

        payload = layer.model_dump(mode="json")
        payload["persisted_identity_ids"] = ("creative_identity::v6_6_cognitive_core",)

        with self.assertRaisesRegex(
            ValueError,
            "identity persistence, mutation, personalization, and HITL ids",
        ):
            CreativeIdentityLayerPlan(**payload)

    def test_creative_identity_layer_reuses_supplied_cognition_layer(self) -> None:
        cognition_layer = build_creative_cognition_layer(route="generate")
        layer = build_creative_identity_layer(
            creative_cognition_layer=cognition_layer,
        )

        self.assertEqual(layer.route_name, cognition_layer.route_name)
        self.assertEqual(layer.task_type, cognition_layer.task_type)
        self.assertEqual(layer.source_cognition_ids, cognition_layer.cognition_ids)
        self.assertEqual(
            layer.source_governance_ids,
            cognition_layer.source_governance_ids,
        )
        self.assertEqual(
            layer.source_planning_ids,
            cognition_layer.source_planning_ids,
        )
        self.assertEqual(
            layer.source_reasoning_ids,
            cognition_layer.source_reasoning_ids,
        )
        self.assertEqual(layer.source_profile_ids, cognition_layer.source_profile_ids)
        self.assertEqual(layer.source_state_ids, cognition_layer.source_state_ids)


if __name__ == "__main__":
    unittest.main()
