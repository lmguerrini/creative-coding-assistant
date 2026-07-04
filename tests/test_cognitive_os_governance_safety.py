import unittest

from creative_coding_assistant.orchestration import (
    CognitiveOSGovernanceBoundary,
    CognitiveOSGovernanceSafetyPlan,
    build_cognitive_os_governance_safety,
    build_cognitive_os_secondary_surface,
    cognitive_os_governance_boundaries_for_agent,
    cognitive_os_governance_boundaries_for_layer,
    cognitive_os_governance_boundaries_for_priority,
    cognitive_os_governance_boundaries_for_status,
    cognitive_os_governance_boundary_by_id,
    cognitive_os_governance_score,
)
from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    COGNITIVE_OS_LAYER_ORDER,
    COGNITIVE_OS_ROADMAP_ITEMS,
)
from creative_coding_assistant.orchestration.cognitive_os_governance_safety import (
    COGNITIVE_OS_GOVERNANCE_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_GOVERNANCE_SOURCE_ROLES,
    COGNITIVE_OS_GOVERNANCE_SOURCE_SERIALIZATION_VERSIONS,
)
from creative_coding_assistant.orchestration.cognitive_os_secondary_surface import (
    COGNITIVE_OS_FOUNDATION_SYSTEMS,
    COGNITIVE_OS_SECONDARY_REPORT_SECTIONS,
)


class CognitiveOSGovernanceSafetyTests(unittest.TestCase):
    def test_governance_safety_plan_composes_os_surfaces(self) -> None:
        secondary_surface = build_cognitive_os_secondary_surface()
        plan = build_cognitive_os_governance_safety(secondary_surface)

        self.assertEqual(plan.role, "cognitive_os_governance_safety")
        self.assertEqual(
            plan.serialization_version,
            "cognitive_os_governance_safety.v1",
        )
        self.assertEqual(plan.route_name, secondary_surface.route_name)
        self.assertEqual(plan.task_type, secondary_surface.task_type)
        self.assertEqual(plan.execution_mode_ids, secondary_surface.execution_mode_ids)
        self.assertEqual(
            plan.source_surface_roles,
            COGNITIVE_OS_GOVERNANCE_SOURCE_ROLES,
        )
        self.assertEqual(
            plan.source_surface_serialization_versions,
            COGNITIVE_OS_GOVERNANCE_SOURCE_SERIALIZATION_VERSIONS,
        )
        self.assertEqual(
            plan.source_core_surface_ids,
            secondary_surface.source_core_surface_ids,
        )
        self.assertEqual(plan.source_core_surface_count, 6)
        self.assertEqual(
            plan.source_secondary_surface_ids,
            secondary_surface.secondary_surface_ids,
        )
        self.assertEqual(plan.source_secondary_surface_count, 6)
        self.assertEqual(plan.source_hitl_ids, secondary_surface.source_hitl_ids)
        self.assertEqual(plan.source_hitl_count, 6)
        self.assertEqual(plan.source_safety_ids, secondary_surface.source_safety_ids)
        self.assertEqual(plan.source_safety_count, 6)
        self.assertEqual(
            plan.source_explanation_ids,
            secondary_surface.source_explanation_ids,
        )
        self.assertEqual(plan.source_explanation_count, 6)
        self.assertEqual(plan.layer_order, COGNITIVE_OS_LAYER_ORDER)
        self.assertEqual(plan.capabilities, COGNITIVE_OS_CAPABILITIES)
        self.assertEqual(plan.capability_ids, secondary_surface.capability_ids)
        self.assertEqual(plan.capability_count, 6)
        self.assertEqual(plan.foundation_systems, COGNITIVE_OS_FOUNDATION_SYSTEMS)
        self.assertEqual(plan.foundation_system_count, 7)
        self.assertEqual(plan.governed_roadmap_items, COGNITIVE_OS_ROADMAP_ITEMS)
        self.assertEqual(plan.governed_roadmap_item_count, 24)
        self.assertEqual(len(plan.governance_boundaries), 6)
        self.assertEqual(plan.governance_boundary_count, 6)
        self.assertEqual(plan.guarded_boundary_ids, plan.governance_boundary_ids)
        self.assertEqual(plan.guarded_boundary_count, 6)
        self.assertEqual(plan.hitl_required_boundary_ids, plan.governance_boundary_ids)
        self.assertEqual(plan.hitl_required_boundary_count, 6)
        self.assertEqual(plan.highest_governance_score, 1000)
        self.assertEqual(plan.overall_governance_score, 1000)
        self.assertEqual(plan.overall_governance_posture, "guarded")
        self.assertEqual(plan.linked_agent_ids, secondary_surface.linked_agent_ids)
        self.assertEqual(plan.report_sections, COGNITIVE_OS_SECONDARY_REPORT_SECTIONS)
        self.assertEqual(plan.covered_task_items, ("Governance and Safety",))
        self.assertEqual(plan.covered_task_item_count, 1)
        self.assertEqual(plan.cross_cutting_contracts, COGNITIVE_OS_CONTRACTS)
        self.assertEqual(
            plan.blocked_runtime_behaviors,
            COGNITIVE_OS_GOVERNANCE_BLOCKED_RUNTIME_BEHAVIORS,
        )
        self.assertIn("no-automation boundaries", plan.authority_boundary)
        self.assertFalse(plan.applied_governance_boundary_ids)
        self.assertFalse(plan.enforced_safety_policy_ids)
        self.assertFalse(plan.emitted_hitl_request_ids)
        self.assertFalse(plan.requested_human_input_ids)
        self.assertFalse(plan.activated_automation_ids)
        self.assertFalse(plan.activated_core_surface_ids)
        self.assertFalse(plan.activated_secondary_surface_ids)
        self.assertFalse(plan.executed_node_ids)
        self.assertFalse(plan.traversed_edge_ids)
        self.assertFalse(plan.applied_route_decision_ids)
        self.assertFalse(plan.written_storage_record_ids)
        self.assertFalse(plan.provider_execution_ids)
        self.assertFalse(plan.mutated_prompt_ids)
        self.assertFalse(plan.mutated_workflow_ids)
        self.assertFalse(plan.mutated_memory_ids)
        self.assertFalse(plan.mutated_retrieval_ids)
        self.assertFalse(plan.mutated_output_ids)
        self.assertFalse(plan.applied_hitl_decision_ids)
        self.assertTrue(plan.cognitive_os_governance_safety_implemented)
        self.assertTrue(plan.governance_boundary_metadata_implemented)
        self.assertTrue(plan.hitl_boundary_metadata_implemented)
        self.assertTrue(plan.explainability_boundary_metadata_implemented)
        self.assertTrue(plan.no_automation_boundary_metadata_implemented)
        self.assertTrue(plan.safety_boundary_metadata_implemented)
        self.assertTrue(plan.all_capability_surfaces_traceable)
        self.assertTrue(plan.all_roadmap_items_traceable)
        self.assertTrue(plan.core_surface_foundation_used)
        self.assertTrue(plan.secondary_surface_foundation_used)
        self.assertFalse(plan.governance_policy_enforcement_implemented)
        self.assertFalse(plan.safety_policy_enforcement_implemented)
        self.assertFalse(plan.hitl_request_emitted)
        self.assertFalse(plan.human_input_request_implemented)
        self.assertFalse(plan.automation_activation_implemented)
        self.assertFalse(plan.core_surface_activation_implemented)
        self.assertFalse(plan.secondary_surface_activation_implemented)
        self.assertFalse(plan.execution_application_implemented)
        self.assertFalse(plan.routing_application_implemented)
        self.assertFalse(plan.storage_write_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.workflow_mutation_implemented)
        self.assertFalse(plan.memory_mutation_implemented)
        self.assertFalse(plan.retrieval_mutation_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.hitl_decision_application_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_governance_boundaries_are_traceable_and_advisory(self) -> None:
        plan = build_cognitive_os_governance_safety()

        for boundary in plan.governance_boundaries:
            self.assertEqual(
                boundary.source_surface_roles,
                COGNITIVE_OS_GOVERNANCE_SOURCE_ROLES,
            )
            self.assertEqual(
                boundary.source_serialization_versions,
                COGNITIVE_OS_GOVERNANCE_SOURCE_SERIALIZATION_VERSIONS,
            )
            self.assertEqual(boundary.source_item_count, 10)
            self.assertEqual(
                boundary.governed_roadmap_items,
                COGNITIVE_OS_ROADMAP_ITEMS,
            )
            self.assertEqual(boundary.governed_roadmap_item_count, 24)
            self.assertEqual(
                boundary.foundation_systems,
                COGNITIVE_OS_FOUNDATION_SYSTEMS,
            )
            self.assertEqual(
                boundary.report_sections,
                COGNITIVE_OS_SECONDARY_REPORT_SECTIONS,
            )
            self.assertEqual(boundary.hitl_requirement_count, 5)
            self.assertEqual(boundary.explainability_signal_count, 6)
            self.assertEqual(boundary.no_automation_weight, 220)
            self.assertEqual(boundary.safety_weight, 220)
            self.assertEqual(boundary.governance_score, 1000)
            self.assertEqual(
                boundary.governance_score,
                cognitive_os_governance_score(
                    source_item_count=boundary.source_item_count,
                    governed_roadmap_item_count=(boundary.governed_roadmap_item_count),
                    hitl_requirement_count=boundary.hitl_requirement_count,
                    explainability_signal_count=(boundary.explainability_signal_count),
                    no_automation_weight=boundary.no_automation_weight,
                    safety_weight=boundary.safety_weight,
                ),
            )
            self.assertEqual(boundary.status, "guarded")
            self.assertEqual(boundary.priority, "guarded")
            self.assertTrue(boundary.hitl_required_before_governance_application)
            self.assertIn("No automation", boundary.no_automation_boundary)
            self.assertIn("advisory", boundary.safety_boundary)
            self.assertIn("automation_activation", boundary.blocked_runtime_behaviors)
            self.assertFalse(boundary.applied_governance_boundary_ids)
            self.assertFalse(boundary.enforced_safety_policy_ids)
            self.assertFalse(boundary.emitted_hitl_request_ids)
            self.assertFalse(boundary.requested_human_input_ids)
            self.assertFalse(boundary.activated_automation_ids)
            self.assertFalse(boundary.activated_core_surface_ids)
            self.assertFalse(boundary.activated_secondary_surface_ids)
            self.assertFalse(boundary.executed_node_ids)
            self.assertFalse(boundary.traversed_edge_ids)
            self.assertFalse(boundary.applied_route_decision_ids)
            self.assertFalse(boundary.written_storage_record_ids)
            self.assertFalse(boundary.provider_execution_ids)
            self.assertFalse(boundary.mutated_output_ids)
            self.assertFalse(boundary.applied_hitl_decision_ids)
            self.assertFalse(boundary.governance_policy_enforcement_implemented)
            self.assertFalse(boundary.safety_policy_enforcement_implemented)
            self.assertFalse(boundary.hitl_request_emitted)
            self.assertFalse(boundary.automation_activation_implemented)
            self.assertFalse(boundary.core_surface_activation_implemented)
            self.assertFalse(boundary.secondary_surface_activation_implemented)
            self.assertFalse(boundary.execution_application_implemented)
            self.assertFalse(boundary.routing_application_implemented)
            self.assertFalse(boundary.runtime_evolution_implemented)
            self.assertTrue(boundary.advisory_only)

        core_boundary = cognitive_os_governance_boundary_by_id(
            "cognitive_os_governance::v6_6_cognitive_core",
            plan,
        )
        self.assertIsNotNone(core_boundary)
        assert core_boundary is not None
        self.assertEqual(core_boundary.capability_name, "V6.6 Cognitive Core")
        self.assertEqual(core_boundary.cognitive_layer, "cognitive_core")
        self.assertEqual(
            core_boundary.source_secondary_surface_id,
            "cognitive_os_secondary::v6_6_cognitive_core",
        )
        self.assertIn("planner_agent", core_boundary.linked_agent_ids)

        research_boundaries = cognitive_os_governance_boundaries_for_layer(
            "research",
            plan,
        )
        self.assertEqual(len(research_boundaries), 1)
        self.assertEqual(
            research_boundaries[0].capability_id,
            "v6_4_autonomous_research",
        )
        planner_boundaries = cognitive_os_governance_boundaries_for_agent(
            "planner_agent",
            plan,
        )
        self.assertEqual(
            tuple(boundary.capability_id for boundary in planner_boundaries),
            ("v6_6_cognitive_core",),
        )
        self.assertEqual(
            len(cognitive_os_governance_boundaries_for_status("guarded", plan)),
            6,
        )
        self.assertEqual(
            len(cognitive_os_governance_boundaries_for_priority("guarded", plan)),
            6,
        )
        self.assertIsNone(cognitive_os_governance_boundary_by_id("missing", plan))

    def test_governance_rejects_mismatched_or_mutating_payloads(self) -> None:
        plan = build_cognitive_os_governance_safety()
        payload = plan.model_dump(mode="json")
        payload["governance_boundary_ids"] = ("missing",) + tuple(
            payload["governance_boundary_ids"][1:]
        )

        with self.assertRaisesRegex(ValueError, "governance_boundary_ids must match"):
            CognitiveOSGovernanceSafetyPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["activated_automation_ids"] = ("automation",)

        with self.assertRaisesRegex(
            ValueError,
            "governance plan mutation ids must be empty",
        ):
            CognitiveOSGovernanceSafetyPlan(**payload)

        boundary_payload = plan.governance_boundaries[0].model_dump(mode="json")
        boundary_payload["governance_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "governance_score must combine boundary inputs",
        ):
            CognitiveOSGovernanceBoundary(**boundary_payload)

        boundary_payload = plan.governance_boundaries[0].model_dump(mode="json")
        boundary_payload["emitted_hitl_request_ids"] = ("hitl_request",)

        with self.assertRaisesRegex(
            ValueError,
            "governance boundary mutation ids must be empty",
        ):
            CognitiveOSGovernanceBoundary(**boundary_payload)

    def test_governance_reuses_supplied_secondary_surface(self) -> None:
        secondary_surface = build_cognitive_os_secondary_surface(route="generate")
        plan = build_cognitive_os_governance_safety(secondary_surface)

        self.assertEqual(plan.route_name, secondary_surface.route_name)
        self.assertEqual(plan.task_type, secondary_surface.task_type)
        self.assertEqual(plan.execution_mode_ids, secondary_surface.execution_mode_ids)
        self.assertEqual(
            plan.source_core_surface_ids,
            secondary_surface.source_core_surface_ids,
        )
        self.assertEqual(
            plan.source_secondary_surface_ids,
            secondary_surface.secondary_surface_ids,
        )
        self.assertEqual(plan.source_hitl_ids, secondary_surface.source_hitl_ids)
        self.assertEqual(plan.source_safety_ids, secondary_surface.source_safety_ids)
        self.assertEqual(
            plan.source_explanation_ids,
            secondary_surface.source_explanation_ids,
        )


if __name__ == "__main__":
    unittest.main()
