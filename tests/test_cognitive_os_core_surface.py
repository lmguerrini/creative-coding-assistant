import unittest

from creative_coding_assistant.orchestration import (
    CognitiveOSCoreSurfacePlan,
    build_cognitive_os_core_surface,
    build_core_os_consolidation,
    cognitive_os_core_surface_entries_for_agent,
    cognitive_os_core_surface_entries_for_layer,
    cognitive_os_core_surface_entries_for_status,
    cognitive_os_core_surface_entry_by_id,
)
from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    COGNITIVE_OS_LAYER_ORDER,
)


class CognitiveOSCoreSurfaceTests(unittest.TestCase):
    def test_cognitive_os_core_surface_builds_read_only_entries(self) -> None:
        consolidation = build_core_os_consolidation()
        surface = build_cognitive_os_core_surface(
            core_os_consolidation=consolidation,
        )

        self.assertEqual(surface.role, "cognitive_os_core_surface")
        self.assertEqual(
            surface.serialization_version,
            "cognitive_os_core_surface.v1",
        )
        self.assertEqual(surface.source_consolidation_role, consolidation.role)
        self.assertEqual(
            surface.source_consolidation_serialization_version,
            consolidation.serialization_version,
        )
        self.assertEqual(surface.layer_order, COGNITIVE_OS_LAYER_ORDER)
        self.assertEqual(surface.capabilities, COGNITIVE_OS_CAPABILITIES)
        self.assertEqual(surface.capability_ids, consolidation.capability_ids)
        self.assertEqual(surface.capability_count, 6)
        self.assertEqual(
            surface.source_consolidation_unit_ids,
            consolidation.consolidation_unit_ids,
        )
        self.assertEqual(surface.source_consolidation_unit_count, 6)
        self.assertEqual(
            surface.source_execution_node_ids,
            consolidation.source_execution_node_ids,
        )
        self.assertEqual(surface.source_execution_node_count, 6)
        self.assertEqual(surface.source_hitl_ids, consolidation.source_hitl_ids)
        self.assertEqual(surface.source_hitl_count, 6)
        self.assertEqual(surface.source_safety_ids, consolidation.source_safety_ids)
        self.assertEqual(surface.source_safety_count, 6)
        self.assertEqual(
            surface.source_explanation_ids,
            consolidation.source_explanation_ids,
        )
        self.assertEqual(surface.source_explanation_count, 6)
        self.assertEqual(len(surface.core_surface_entries), 6)
        self.assertEqual(surface.core_surface_count, 6)
        self.assertEqual(surface.guarded_core_surface_ids, surface.core_surface_ids)
        self.assertEqual(surface.guarded_core_surface_count, 6)
        self.assertEqual(
            surface.hitl_required_core_surface_ids,
            surface.core_surface_ids,
        )
        self.assertEqual(surface.hitl_required_core_surface_count, 6)
        self.assertEqual(surface.highest_core_surface_score, 94)
        self.assertEqual(surface.overall_core_surface_score, 92)
        self.assertEqual(surface.overall_core_surface_posture, "guarded")
        self.assertEqual(surface.linked_agent_ids, consolidation.linked_agent_ids)
        self.assertEqual(
            surface.covered_task_items,
            ("Core Surface Implementation",),
        )
        self.assertEqual(surface.covered_task_item_count, 1)
        self.assertEqual(surface.cross_cutting_contracts, COGNITIVE_OS_CONTRACTS)
        self.assertEqual(
            surface.blocked_runtime_behaviors,
            COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        self.assertTrue(surface.core_surface_implemented)
        self.assertTrue(surface.core_surface_metadata_implemented)
        self.assertTrue(surface.core_surface_lookup_helpers_implemented)
        self.assertFalse(surface.core_surface_activation_implemented)
        self.assertFalse(surface.runtime_activation_implemented)
        self.assertFalse(surface.execution_application_implemented)
        self.assertFalse(surface.routing_application_implemented)
        self.assertFalse(surface.hitl_request_emission_implemented)
        self.assertFalse(surface.hitl_decision_application_implemented)
        self.assertFalse(surface.prompt_mutation_implemented)
        self.assertFalse(surface.memory_mutation_implemented)
        self.assertFalse(surface.retrieval_mutation_implemented)
        self.assertFalse(surface.storage_mutation_implemented)
        self.assertFalse(surface.provider_model_routing_implemented)
        self.assertFalse(surface.provider_execution_implemented)
        self.assertFalse(surface.generated_output_mutation_implemented)
        self.assertFalse(surface.runtime_evolution_implemented)
        self.assertFalse(surface.activated_core_surface_ids)
        self.assertFalse(surface.persisted_core_surface_ids)
        self.assertFalse(surface.emitted_hitl_request_ids)
        self.assertFalse(surface.applied_hitl_decision_ids)
        self.assertFalse(surface.mutated_core_surface_ids)
        self.assertTrue(surface.advisory_only)

    def test_cognitive_os_core_surface_lookup_helpers_are_scope_aware(self) -> None:
        surface = build_cognitive_os_core_surface()

        core_entry = cognitive_os_core_surface_entry_by_id(
            "cognitive_os_core::v6_6_cognitive_core",
            surface,
        )
        self.assertIsNotNone(core_entry)
        assert core_entry is not None
        self.assertEqual(core_entry.capability_name, "V6.6 Cognitive Core")
        self.assertEqual(core_entry.cognitive_layer, "cognitive_core")
        self.assertEqual(core_entry.surface_kind, "cognitive_core_surface")
        self.assertIn("planner_agent", core_entry.linked_agent_ids)
        self.assertEqual(core_entry.surface_sequence_position, 6)
        self.assertEqual(core_entry.dependency_depth, 5)
        self.assertEqual(core_entry.surface_readiness_score, 89)
        self.assertEqual(
            core_entry.source_trace_ids[0],
            "core_os::v6_6_cognitive_core",
        )
        self.assertIn(
            "unified_execution::v6_6_cognitive_core",
            core_entry.source_trace_ids,
        )
        self.assertIn("hitl_required", core_entry.context_tags)
        self.assertTrue(core_entry.hitl_required_before_core_surface_activation)
        self.assertFalse(core_entry.core_surface_activation_authorized)
        self.assertFalse(core_entry.runtime_activation_authorized)

        research_entries = cognitive_os_core_surface_entries_for_layer(
            "research",
            surface,
        )
        self.assertEqual(len(research_entries), 1)
        self.assertEqual(research_entries[0].capability_id, "v6_4_autonomous_research")

        planner_entries = cognitive_os_core_surface_entries_for_agent(
            "planner_agent",
            surface,
        )
        self.assertEqual(
            tuple(entry.capability_id for entry in planner_entries),
            ("v6_6_cognitive_core",),
        )
        guarded_entries = cognitive_os_core_surface_entries_for_status(
            "guarded",
            surface,
        )
        self.assertEqual(
            tuple(entry.core_surface_id for entry in guarded_entries),
            surface.guarded_core_surface_ids,
        )
        self.assertIsNone(cognitive_os_core_surface_entry_by_id("missing", surface))

    def test_cognitive_os_core_surface_rejects_activation_and_drift(self) -> None:
        surface = build_cognitive_os_core_surface()
        payload = surface.model_dump(mode="json")
        payload["overall_core_surface_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_core_surface_score must match",
        ):
            CognitiveOSCoreSurfacePlan(**payload)

        payload = surface.model_dump(mode="json")
        payload["activated_core_surface_ids"] = (
            "cognitive_os_core::v6_6_cognitive_core",
        )

        with self.assertRaisesRegex(
            ValueError,
            "core surface activation, persistence, HITL, and mutation ids "
            "must be empty",
        ):
            CognitiveOSCoreSurfacePlan(**payload)

    def test_cognitive_os_core_surface_reuses_supplied_consolidation(self) -> None:
        consolidation = build_core_os_consolidation(route="generate")
        surface = build_cognitive_os_core_surface(
            core_os_consolidation=consolidation,
        )

        self.assertEqual(surface.route_name, consolidation.route_name)
        self.assertEqual(surface.task_type, consolidation.task_type)
        self.assertEqual(
            surface.source_consolidation_unit_ids,
            consolidation.consolidation_unit_ids,
        )
        self.assertEqual(
            surface.source_execution_node_ids,
            consolidation.source_execution_node_ids,
        )
        self.assertEqual(surface.source_hitl_ids, consolidation.source_hitl_ids)
        self.assertEqual(surface.source_safety_ids, consolidation.source_safety_ids)
        self.assertEqual(
            surface.source_explanation_ids,
            consolidation.source_explanation_ids,
        )


if __name__ == "__main__":
    unittest.main()
