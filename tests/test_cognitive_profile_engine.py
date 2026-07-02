import unittest

from creative_coding_assistant.orchestration import (
    CognitiveProfileEnginePlan,
    build_cognitive_profile_engine,
    build_cognitive_state_engine,
    cognitive_profile_by_id,
    cognitive_profiles_for_agent,
    cognitive_profiles_for_layer,
)
from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    COGNITIVE_OS_LAYER_ORDER,
)


class CognitiveProfileEngineTests(unittest.TestCase):
    def test_cognitive_profile_engine_builds_read_only_profiles(self) -> None:
        state_engine = build_cognitive_state_engine()
        engine = build_cognitive_profile_engine(state_engine=state_engine)

        self.assertEqual(engine.role, "cognitive_profile_engine")
        self.assertEqual(
            engine.serialization_version,
            "cognitive_profile_engine.v1",
        )
        self.assertEqual(engine.state_engine_role, state_engine.role)
        self.assertEqual(
            engine.optimization_layer_role,
            "cross_system_optimization_layer",
        )
        self.assertEqual(engine.learning_layer_role, "cross_system_learning_layer")
        self.assertEqual(engine.layer_order, COGNITIVE_OS_LAYER_ORDER)
        self.assertEqual(engine.capabilities, COGNITIVE_OS_CAPABILITIES)
        self.assertEqual(engine.capability_ids, state_engine.capability_ids)
        self.assertEqual(engine.capability_count, 6)
        self.assertEqual(engine.source_state_ids, state_engine.state_ids)
        self.assertEqual(engine.source_state_count, 6)
        self.assertEqual(
            engine.source_optimization_signal_ids,
            state_engine.source_optimization_signal_ids,
        )
        self.assertEqual(engine.source_optimization_signal_count, 6)
        self.assertEqual(
            engine.source_learning_signal_ids,
            state_engine.source_learning_signal_ids,
        )
        self.assertEqual(engine.source_learning_signal_count, 6)
        self.assertEqual(len(engine.cognitive_profiles), 6)
        self.assertEqual(engine.profile_count, 6)
        self.assertEqual(engine.linked_agent_ids, state_engine.linked_agent_ids)
        self.assertEqual(
            engine.covered_roadmap_items,
            ("Cognitive Profile Engine",),
        )
        self.assertEqual(engine.covered_roadmap_item_count, 1)
        self.assertEqual(engine.cross_cutting_contracts, COGNITIVE_OS_CONTRACTS)
        self.assertEqual(
            engine.blocked_runtime_behaviors,
            COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        self.assertTrue(engine.cognitive_profile_engine_implemented)
        self.assertTrue(engine.cognitive_state_engine_integrated)
        self.assertTrue(engine.profile_contract_implemented)
        self.assertTrue(engine.profile_dependency_traceability_implemented)
        self.assertTrue(engine.profile_governance_contract_implemented)
        self.assertTrue(engine.profile_explainability_contract_implemented)
        self.assertFalse(engine.profile_persistence_implemented)
        self.assertFalse(engine.profile_mutation_implemented)
        self.assertFalse(engine.personalized_runtime_behavior_implemented)
        self.assertFalse(engine.profile_driven_agent_routing_implemented)
        self.assertFalse(engine.provider_model_routing_implemented)
        self.assertFalse(engine.provider_execution_implemented)
        self.assertFalse(engine.workflow_control_implemented)
        self.assertFalse(engine.generated_output_mutation_implemented)
        self.assertFalse(engine.runtime_evolution_implemented)
        self.assertFalse(engine.persisted_profile_ids)
        self.assertFalse(engine.mutated_profile_ids)
        self.assertFalse(engine.routed_profile_ids)
        self.assertFalse(engine.emitted_hitl_request_ids)
        self.assertTrue(engine.advisory_only)

    def test_cognitive_profile_lookup_helpers_are_layer_and_agent_aware(
        self,
    ) -> None:
        engine = build_cognitive_profile_engine()

        core_profile = cognitive_profile_by_id(
            "cognitive_profile::v6_6_cognitive_core",
            engine,
        )
        self.assertIsNotNone(core_profile)
        assert core_profile is not None
        self.assertEqual(core_profile.capability_name, "V6.6 Cognitive Core")
        self.assertEqual(core_profile.cognitive_layer, "cognitive_core")
        self.assertIn("planner_agent", core_profile.linked_agent_ids)
        self.assertEqual(
            core_profile.state_id,
            "cognitive_state::v6_6_cognitive_core",
        )
        self.assertIn(
            "personalized runtime behavior",
            core_profile.governance_contracts[1],
        )

        research_profiles = cognitive_profiles_for_layer("research", engine)
        self.assertEqual(len(research_profiles), 1)
        self.assertEqual(
            research_profiles[0].capability_id,
            "v6_4_autonomous_research",
        )

        planner_profiles = cognitive_profiles_for_agent("planner_agent", engine)
        self.assertEqual(
            tuple(profile.capability_id for profile in planner_profiles),
            ("v6_6_cognitive_core",),
        )
        self.assertIsNone(cognitive_profile_by_id("missing", engine))

    def test_cognitive_profile_engine_rejects_profile_mutation_and_drift(
        self,
    ) -> None:
        engine = build_cognitive_profile_engine()
        payload = engine.model_dump(mode="json")
        payload["profile_ids"] = ("missing",) + tuple(payload["profile_ids"][1:])

        with self.assertRaisesRegex(ValueError, "profile_ids must match"):
            CognitiveProfileEnginePlan(**payload)

        payload = engine.model_dump(mode="json")
        payload["routed_profile_ids"] = (
            "cognitive_profile::v6_6_cognitive_core",
        )

        with self.assertRaisesRegex(
            ValueError,
            "profile persistence, mutation, routing, and HITL ids must be empty",
        ):
            CognitiveProfileEnginePlan(**payload)

    def test_cognitive_profile_engine_reuses_supplied_state_engine(self) -> None:
        state_engine = build_cognitive_state_engine(route="generate")
        engine = build_cognitive_profile_engine(state_engine=state_engine)

        self.assertEqual(engine.route_name, state_engine.route_name)
        self.assertEqual(engine.task_type, state_engine.task_type)
        self.assertEqual(engine.source_state_ids, state_engine.state_ids)
        self.assertEqual(
            engine.source_optimization_signal_ids,
            state_engine.source_optimization_signal_ids,
        )
        self.assertEqual(
            engine.source_learning_signal_ids,
            state_engine.source_learning_signal_ids,
        )


if __name__ == "__main__":
    unittest.main()
