import unittest

from creative_coding_assistant.orchestration import (
    CognitiveStateEnginePlan,
    build_cognitive_state_engine,
    build_cross_system_optimization_layer,
    cognitive_state_snapshot_by_id,
    cognitive_state_snapshots_for_agent,
    cognitive_state_snapshots_for_layer,
)
from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    COGNITIVE_OS_LAYER_ORDER,
)


class CognitiveStateEngineTests(unittest.TestCase):
    def test_cognitive_state_engine_builds_read_only_snapshots(self) -> None:
        optimization_layer = build_cross_system_optimization_layer()
        engine = build_cognitive_state_engine(
            optimization_layer=optimization_layer,
        )

        self.assertEqual(engine.role, "cognitive_state_engine")
        self.assertEqual(
            engine.serialization_version,
            "cognitive_state_engine.v1",
        )
        self.assertEqual(
            engine.optimization_layer_role,
            optimization_layer.role,
        )
        self.assertEqual(engine.learning_layer_role, "cross_system_learning_layer")
        self.assertEqual(engine.layer_order, COGNITIVE_OS_LAYER_ORDER)
        self.assertEqual(engine.capabilities, COGNITIVE_OS_CAPABILITIES)
        self.assertEqual(engine.capability_ids, optimization_layer.capability_ids)
        self.assertEqual(engine.capability_count, 6)
        self.assertEqual(
            engine.source_optimization_signal_ids,
            optimization_layer.optimization_signal_ids,
        )
        self.assertEqual(engine.source_optimization_signal_count, 6)
        self.assertEqual(
            engine.source_learning_signal_ids,
            optimization_layer.source_learning_signal_ids,
        )
        self.assertEqual(engine.source_learning_signal_count, 6)
        self.assertEqual(engine.source_optimization_proposal_count, 5)
        self.assertEqual(len(engine.state_snapshots), 6)
        self.assertEqual(engine.state_count, 6)
        self.assertEqual(engine.linked_agent_ids, optimization_layer.linked_agent_ids)
        self.assertEqual(
            engine.covered_roadmap_items,
            ("Cognitive State Engine",),
        )
        self.assertEqual(engine.covered_roadmap_item_count, 1)
        self.assertEqual(engine.cross_cutting_contracts, COGNITIVE_OS_CONTRACTS)
        self.assertEqual(
            engine.blocked_runtime_behaviors,
            COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        self.assertTrue(engine.cognitive_state_engine_implemented)
        self.assertTrue(engine.cross_system_optimization_layer_integrated)
        self.assertTrue(engine.state_snapshot_contract_implemented)
        self.assertTrue(engine.state_dependency_traceability_implemented)
        self.assertTrue(engine.state_governance_contract_implemented)
        self.assertTrue(engine.state_explainability_contract_implemented)
        self.assertFalse(engine.state_persistence_implemented)
        self.assertFalse(engine.state_mutation_implemented)
        self.assertFalse(engine.stateful_agent_routing_implemented)
        self.assertFalse(engine.provider_model_routing_implemented)
        self.assertFalse(engine.provider_execution_implemented)
        self.assertFalse(engine.workflow_control_implemented)
        self.assertFalse(engine.generated_output_mutation_implemented)
        self.assertFalse(engine.runtime_evolution_implemented)
        self.assertFalse(engine.persisted_state_ids)
        self.assertFalse(engine.mutated_state_ids)
        self.assertFalse(engine.routed_state_ids)
        self.assertFalse(engine.emitted_hitl_request_ids)
        self.assertTrue(engine.advisory_only)

    def test_cognitive_state_lookup_helpers_are_layer_and_agent_aware(self) -> None:
        engine = build_cognitive_state_engine()

        core_snapshot = cognitive_state_snapshot_by_id(
            "cognitive_state::v6_6_cognitive_core",
            engine,
        )
        self.assertIsNotNone(core_snapshot)
        assert core_snapshot is not None
        self.assertEqual(core_snapshot.capability_name, "V6.6 Cognitive Core")
        self.assertEqual(core_snapshot.cognitive_layer, "cognitive_core")
        self.assertIn("planner_agent", core_snapshot.linked_agent_ids)
        self.assertEqual(
            core_snapshot.source_optimization_signal_id,
            "cross_system_optimization::v6_6_cognitive_core",
        )
        self.assertIn(
            "state mutation",
            core_snapshot.governance_contracts[1],
        )

        research_snapshots = cognitive_state_snapshots_for_layer(
            "research",
            engine,
        )
        self.assertEqual(len(research_snapshots), 1)
        self.assertEqual(
            research_snapshots[0].capability_id,
            "v6_4_autonomous_research",
        )

        planner_snapshots = cognitive_state_snapshots_for_agent(
            "planner_agent",
            engine,
        )
        self.assertEqual(
            tuple(snapshot.capability_id for snapshot in planner_snapshots),
            ("v6_6_cognitive_core",),
        )
        self.assertIsNone(cognitive_state_snapshot_by_id("missing", engine))

    def test_cognitive_state_engine_rejects_state_mutation_and_drift(self) -> None:
        engine = build_cognitive_state_engine()
        payload = engine.model_dump(mode="json")
        payload["state_ids"] = ("missing",) + tuple(payload["state_ids"][1:])

        with self.assertRaisesRegex(ValueError, "state_ids must match"):
            CognitiveStateEnginePlan(**payload)

        payload = engine.model_dump(mode="json")
        payload["mutated_state_ids"] = ("cognitive_state::v6_6_cognitive_core",)

        with self.assertRaisesRegex(
            ValueError,
            "state persistence, mutation, routing, and HITL ids must be empty",
        ):
            CognitiveStateEnginePlan(**payload)

    def test_cognitive_state_engine_reuses_supplied_optimization_layer(self) -> None:
        optimization_layer = build_cross_system_optimization_layer(route="generate")
        engine = build_cognitive_state_engine(
            optimization_layer=optimization_layer,
        )

        self.assertEqual(engine.route_name, optimization_layer.route_name)
        self.assertEqual(engine.task_type, optimization_layer.task_type)
        self.assertEqual(
            engine.source_optimization_signal_ids,
            optimization_layer.optimization_signal_ids,
        )
        self.assertEqual(
            engine.source_learning_signal_ids,
            optimization_layer.source_learning_signal_ids,
        )
        self.assertEqual(
            engine.source_optimization_proposal_ids,
            optimization_layer.source_optimization_proposal_ids,
        )


if __name__ == "__main__":
    unittest.main()
