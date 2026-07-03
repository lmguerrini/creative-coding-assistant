import unittest

from creative_coding_assistant.orchestration import (
    CognitiveOSSecondarySurfaceEntry,
    CognitiveOSSecondarySurfacePlan,
    build_cognitive_os_core_surface,
    build_cognitive_os_secondary_surface,
    cognitive_os_secondary_surface_entries_for_agent,
    cognitive_os_secondary_surface_entries_for_layer,
    cognitive_os_secondary_surface_entries_for_status,
    cognitive_os_secondary_surface_entry_by_id,
)
from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    COGNITIVE_OS_LAYER_ORDER,
)
from creative_coding_assistant.orchestration.cognitive_os_secondary_surface import (
    COGNITIVE_OS_FOUNDATION_SYSTEMS,
    COGNITIVE_OS_SECONDARY_REPORT_SECTIONS,
)


class CognitiveOSSecondarySurfaceTests(unittest.TestCase):
    def test_secondary_surface_builds_read_only_report_view_metadata(self) -> None:
        core_surface = build_cognitive_os_core_surface()
        surface = build_cognitive_os_secondary_surface(core_surface)

        self.assertEqual(surface.role, "cognitive_os_secondary_surface")
        self.assertEqual(
            surface.serialization_version,
            "cognitive_os_secondary_surface.v1",
        )
        self.assertEqual(surface.route_name, core_surface.route_name)
        self.assertEqual(surface.task_type, core_surface.task_type)
        self.assertEqual(surface.execution_mode_ids, core_surface.execution_mode_ids)
        self.assertEqual(surface.source_core_surface_role, core_surface.role)
        self.assertEqual(
            surface.source_core_surface_serialization_version,
            core_surface.serialization_version,
        )
        self.assertEqual(surface.layer_order, COGNITIVE_OS_LAYER_ORDER)
        self.assertEqual(surface.capabilities, COGNITIVE_OS_CAPABILITIES)
        self.assertEqual(surface.capability_ids, core_surface.capability_ids)
        self.assertEqual(surface.capability_count, 6)
        self.assertEqual(surface.foundation_systems, COGNITIVE_OS_FOUNDATION_SYSTEMS)
        self.assertEqual(surface.foundation_system_count, 7)
        self.assertEqual(surface.source_core_surface_ids, core_surface.core_surface_ids)
        self.assertEqual(surface.source_core_surface_count, 6)
        self.assertEqual(
            surface.source_core_surface_scores,
            tuple(
                entry.surface_readiness_score
                for entry in core_surface.core_surface_entries
            ),
        )
        self.assertEqual(surface.source_core_surface_score_count, 6)
        self.assertEqual(
            surface.source_consolidation_unit_ids,
            core_surface.source_consolidation_unit_ids,
        )
        self.assertEqual(surface.source_consolidation_unit_count, 6)
        self.assertEqual(
            surface.source_execution_node_ids,
            core_surface.source_execution_node_ids,
        )
        self.assertEqual(surface.source_execution_node_count, 6)
        self.assertEqual(surface.source_hitl_ids, core_surface.source_hitl_ids)
        self.assertEqual(surface.source_hitl_count, 6)
        self.assertEqual(surface.source_safety_ids, core_surface.source_safety_ids)
        self.assertEqual(surface.source_safety_count, 6)
        self.assertEqual(
            surface.source_explanation_ids,
            core_surface.source_explanation_ids,
        )
        self.assertEqual(surface.source_explanation_count, 6)
        self.assertEqual(surface.source_route_decision_count, 6)
        self.assertEqual(surface.source_plan_count, 6)
        self.assertEqual(surface.source_schedule_count, 6)
        self.assertEqual(
            surface.report_sections,
            COGNITIVE_OS_SECONDARY_REPORT_SECTIONS,
        )
        self.assertEqual(len(surface.secondary_surface_entries), 6)
        self.assertEqual(surface.secondary_surface_count, 6)
        self.assertEqual(
            surface.guarded_secondary_surface_ids,
            surface.secondary_surface_ids,
        )
        self.assertEqual(surface.guarded_secondary_surface_count, 6)
        self.assertEqual(
            surface.hitl_required_secondary_surface_ids,
            surface.secondary_surface_ids,
        )
        self.assertEqual(surface.hitl_required_secondary_surface_count, 6)
        self.assertEqual(
            surface.top_secondary_surface_id,
            "cognitive_os_secondary::v6_1_adaptive_learning",
        )
        self.assertEqual(surface.highest_secondary_surface_score, 92)
        self.assertEqual(surface.overall_secondary_surface_score, 90)
        self.assertEqual(surface.overall_secondary_surface_posture, "guarded")
        self.assertEqual(surface.linked_agent_ids, core_surface.linked_agent_ids)
        self.assertEqual(
            surface.covered_task_items,
            ("Secondary Surface Implementation",),
        )
        self.assertEqual(surface.covered_task_item_count, 1)
        self.assertEqual(surface.cross_cutting_contracts, COGNITIVE_OS_CONTRACTS)
        self.assertEqual(
            surface.blocked_runtime_behaviors,
            COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        self.assertTrue(surface.secondary_surface_implemented)
        self.assertTrue(surface.secondary_surface_metadata_implemented)
        self.assertTrue(surface.secondary_surface_lookup_helpers_implemented)
        self.assertTrue(surface.foundation_composition_metadata_implemented)
        self.assertTrue(surface.advisory_report_view_metadata_implemented)
        self.assertTrue(surface.roadmap_traceability_implemented)
        self.assertTrue(surface.dependency_traceability_implemented)
        self.assertTrue(surface.governance_contract_implemented)
        self.assertTrue(surface.explainability_contract_implemented)
        self.assertTrue(surface.safety_contract_implemented)
        self.assertTrue(surface.hitl_contract_implemented)
        self.assertTrue(surface.future_holomind_extensibility_prepared)
        self.assertFalse(surface.secondary_surface_activation_implemented)
        self.assertFalse(surface.runtime_activation_implemented)
        self.assertFalse(surface.report_artifact_generation_implemented)
        self.assertFalse(surface.storage_write_implemented)
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
        self.assertFalse(surface.generated_report_artifact_ids)
        self.assertFalse(surface.written_storage_record_ids)
        self.assertFalse(surface.activated_secondary_surface_ids)
        self.assertFalse(surface.persisted_secondary_surface_ids)
        self.assertFalse(surface.emitted_hitl_request_ids)
        self.assertFalse(surface.applied_hitl_decision_ids)
        self.assertFalse(surface.mutated_secondary_surface_ids)
        self.assertTrue(surface.advisory_only)

    def test_secondary_surface_lookup_helpers_are_scope_aware(self) -> None:
        surface = build_cognitive_os_secondary_surface()

        core_entry = cognitive_os_secondary_surface_entry_by_id(
            "cognitive_os_secondary::v6_6_cognitive_core",
            surface,
        )
        self.assertIsNotNone(core_entry)
        assert core_entry is not None
        self.assertEqual(core_entry.capability_name, "V6.6 Cognitive Core")
        self.assertEqual(core_entry.cognitive_layer, "cognitive_core")
        self.assertEqual(
            core_entry.secondary_surface_kind,
            "cognitive_core_secondary_surface",
        )
        self.assertIn("planner_agent", core_entry.linked_agent_ids)
        self.assertEqual(core_entry.surface_sequence_position, 6)
        self.assertEqual(core_entry.dependency_depth, 5)
        self.assertEqual(core_entry.source_core_surface_score, 89)
        self.assertEqual(core_entry.secondary_surface_score, 87)
        self.assertEqual(
            core_entry.source_core_surface_id,
            "cognitive_os_core::v6_6_cognitive_core",
        )
        self.assertEqual(core_entry.foundation_systems, COGNITIVE_OS_FOUNDATION_SYSTEMS)
        self.assertEqual(
            core_entry.report_sections,
            COGNITIVE_OS_SECONDARY_REPORT_SECTIONS,
        )
        self.assertIn(
            "foundation_composition",
            core_entry.context_tags,
        )
        self.assertEqual(
            core_entry.source_trace_ids[0],
            "cognitive_os_core::v6_6_cognitive_core",
        )
        self.assertIn(
            "unified_execution::v6_6_cognitive_core",
            core_entry.source_trace_ids,
        )
        self.assertTrue(core_entry.hitl_required_before_secondary_surface_activation)
        self.assertFalse(core_entry.secondary_surface_activation_authorized)
        self.assertFalse(core_entry.runtime_activation_authorized)
        self.assertFalse(core_entry.report_artifact_generation_implemented)
        self.assertFalse(core_entry.storage_write_implemented)
        self.assertTrue(core_entry.advisory_only)

        research_entries = cognitive_os_secondary_surface_entries_for_layer(
            "research",
            surface,
        )
        self.assertEqual(len(research_entries), 1)
        self.assertEqual(research_entries[0].capability_id, "v6_4_autonomous_research")

        planner_entries = cognitive_os_secondary_surface_entries_for_agent(
            "planner_agent",
            surface,
        )
        self.assertEqual(
            tuple(entry.capability_id for entry in planner_entries),
            ("v6_6_cognitive_core",),
        )
        guarded_entries = cognitive_os_secondary_surface_entries_for_status(
            "guarded",
            surface,
        )
        self.assertEqual(
            tuple(entry.secondary_surface_id for entry in guarded_entries),
            surface.guarded_secondary_surface_ids,
        )
        self.assertIsNone(
            cognitive_os_secondary_surface_entry_by_id("missing", surface),
        )

    def test_secondary_surface_rejects_artifacts_and_drift(self) -> None:
        surface = build_cognitive_os_secondary_surface()
        payload = surface.model_dump(mode="json")
        payload["overall_secondary_surface_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_secondary_surface_score must match",
        ):
            CognitiveOSSecondarySurfacePlan(**payload)

        payload = surface.model_dump(mode="json")
        payload["generated_report_artifact_ids"] = ("report_artifact",)

        with self.assertRaisesRegex(
            ValueError,
            "secondary surface artifacts, activation, persistence, HITL, "
            "and mutation ids must be empty",
        ):
            CognitiveOSSecondarySurfacePlan(**payload)

        entry_payload = surface.secondary_surface_entries[0].model_dump(mode="json")
        entry_payload["secondary_surface_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "secondary_surface_score must derive from core score",
        ):
            CognitiveOSSecondarySurfaceEntry(**entry_payload)

        entry_payload = surface.secondary_surface_entries[0].model_dump(mode="json")
        entry_payload["written_storage_record_ids"] = ("storage_record",)

        with self.assertRaisesRegex(
            ValueError,
            "secondary surface entry mutation ids must be empty",
        ):
            CognitiveOSSecondarySurfaceEntry(**entry_payload)

    def test_secondary_surface_reuses_supplied_core_surface(self) -> None:
        core_surface = build_cognitive_os_core_surface(route="generate")
        surface = build_cognitive_os_secondary_surface(core_surface)

        self.assertEqual(surface.route_name, core_surface.route_name)
        self.assertEqual(surface.task_type, core_surface.task_type)
        self.assertEqual(surface.execution_mode_ids, core_surface.execution_mode_ids)
        self.assertEqual(surface.source_core_surface_ids, core_surface.core_surface_ids)
        self.assertEqual(
            surface.source_consolidation_unit_ids,
            core_surface.source_consolidation_unit_ids,
        )
        self.assertEqual(
            surface.source_execution_node_ids,
            core_surface.source_execution_node_ids,
        )
        self.assertEqual(surface.source_hitl_ids, core_surface.source_hitl_ids)
        self.assertEqual(surface.source_safety_ids, core_surface.source_safety_ids)
        self.assertEqual(
            surface.source_explanation_ids,
            core_surface.source_explanation_ids,
        )


if __name__ == "__main__":
    unittest.main()
