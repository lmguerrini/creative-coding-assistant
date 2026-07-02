import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    SelfEvolutionGovernanceBoundary,
    SelfEvolutionGovernancePlan,
    build_self_evolution_governance,
    build_self_evolution_secondary_surface,
    route_request,
    self_evolution_governance_boundaries_for_priority,
    self_evolution_governance_boundaries_for_status,
    self_evolution_governance_boundary_by_id,
    self_evolution_governance_boundary_by_roadmap_item,
    self_evolution_governance_score,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.self_evolution_common import (
    CROSS_CUTTING_CONTRACTS,
    UPSTREAM_CAPABILITIES,
)
from creative_coding_assistant.orchestration.self_evolution_core_surface import (
    CORE_ROADMAP_ITEMS,
)
from creative_coding_assistant.orchestration.self_evolution_governance import (
    GOVERNANCE_BLOCKED_RUNTIME_BEHAVIORS,
)
from creative_coding_assistant.orchestration.self_evolution_secondary_surface import (
    SECONDARY_REPORT_SECTIONS,
)

EXPECTED_SOURCE_SURFACE_ROLES = (
    "self_evolution_core_surface",
    "self_evolution_secondary_surface",
)
EXPECTED_SOURCE_SURFACE_VERSIONS = (
    "self_evolution_core_surface.v1",
    "self_evolution_secondary_surface.v1",
)


class SelfEvolutionGovernanceTests(unittest.TestCase):
    def test_governance_plan_composes_core_and_secondary_surfaces(self) -> None:
        plan = build_self_evolution_governance(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "self_evolution_governance_safety")
        self.assertEqual(
            plan.serialization_version,
            "self_evolution_governance_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "reasoning")
        self.assertEqual(plan.source_surface_roles, EXPECTED_SOURCE_SURFACE_ROLES)
        self.assertEqual(
            plan.source_surface_serialization_versions,
            EXPECTED_SOURCE_SURFACE_VERSIONS,
        )
        self.assertEqual(plan.source_core_surface_plan_count, 22)
        self.assertEqual(plan.source_secondary_report_count, 22)
        self.assertEqual(plan.source_proposal_count, 110)
        self.assertEqual(plan.covered_roadmap_items, CORE_ROADMAP_ITEMS)
        self.assertEqual(plan.covered_roadmap_item_count, 22)
        self.assertEqual(plan.proposal_count, 110)
        self.assertEqual(plan.hitl_required_proposal_count, 110)
        self.assertEqual(plan.governance_boundary_count, 22)
        self.assertEqual(plan.guarded_boundary_count, 22)
        self.assertEqual(plan.hitl_required_boundary_count, 22)
        self.assertEqual(plan.highest_governance_score, 1000)
        self.assertEqual(plan.overall_governance_score, 1000)
        self.assertEqual(plan.overall_governance_posture, "guarded")
        self.assertEqual(plan.upstream_capabilities, UPSTREAM_CAPABILITIES)
        self.assertEqual(plan.upstream_signal_source_count, 4)
        self.assertEqual(plan.report_sections, SECONDARY_REPORT_SECTIONS)
        self.assertEqual(plan.cross_cutting_contracts, CROSS_CUTTING_CONTRACTS)
        self.assertEqual(
            plan.blocked_runtime_behaviors,
            GOVERNANCE_BLOCKED_RUNTIME_BEHAVIORS,
        )
        self.assertFalse(plan.applied_governance_boundary_ids)
        self.assertFalse(plan.enforced_safety_policy_ids)
        self.assertFalse(plan.emitted_hitl_request_ids)
        self.assertFalse(plan.requested_human_input_ids)
        self.assertFalse(plan.activated_automation_ids)
        self.assertFalse(plan.applied_evolution_proposal_ids)
        self.assertFalse(plan.executed_rollback_ids)
        self.assertFalse(plan.generated_report_artifact_ids)
        self.assertFalse(plan.written_storage_record_ids)
        self.assertFalse(plan.mutated_prompt_ids)
        self.assertFalse(plan.mutated_workflow_ids)
        self.assertFalse(plan.mutated_routing_ids)
        self.assertFalse(plan.mutated_memory_ids)
        self.assertFalse(plan.mutated_retrieval_ids)
        self.assertFalse(plan.provider_execution_ids)
        self.assertFalse(plan.mutated_output_ids)
        self.assertIn("HITL", plan.authority_boundary)
        self.assertIn("no-automation boundaries", plan.authority_boundary)
        self.assertTrue(plan.self_evolution_governance_implemented)
        self.assertTrue(plan.governance_boundary_metadata_implemented)
        self.assertTrue(plan.hitl_boundary_metadata_implemented)
        self.assertTrue(plan.explainability_boundary_metadata_implemented)
        self.assertTrue(plan.no_automation_boundary_metadata_implemented)
        self.assertTrue(plan.safety_boundary_metadata_implemented)
        self.assertTrue(plan.all_roadmap_items_traceable)
        self.assertTrue(plan.all_proposals_traceable)
        self.assertTrue(plan.core_surface_foundation_used)
        self.assertTrue(plan.secondary_surface_foundation_used)
        self.assertFalse(plan.governance_policy_enforcement_implemented)
        self.assertFalse(plan.safety_policy_enforcement_implemented)
        self.assertFalse(plan.hitl_request_emitted)
        self.assertFalse(plan.human_input_request_implemented)
        self.assertFalse(plan.automation_activation_implemented)
        self.assertFalse(plan.proposal_application_implemented)
        self.assertFalse(plan.rollback_execution_implemented)
        self.assertFalse(plan.report_artifact_generation_implemented)
        self.assertFalse(plan.storage_write_implemented)
        self.assertFalse(plan.prompt_rewriting_implemented)
        self.assertFalse(plan.workflow_mutation_implemented)
        self.assertFalse(plan.routing_mutation_implemented)
        self.assertFalse(plan.memory_mutation_implemented)
        self.assertFalse(plan.retrieval_mutation_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertFalse(plan.hitl_decision_application_implemented)
        self.assertTrue(plan.advisory_only)

    def test_governance_boundaries_are_traceable_and_advisory_only(self) -> None:
        plan = build_self_evolution_governance(route="generate")

        for boundary in plan.governance_boundaries:
            self.assertIn(boundary.roadmap_item, CORE_ROADMAP_ITEMS)
            self.assertEqual(
                boundary.source_surface_roles,
                EXPECTED_SOURCE_SURFACE_ROLES,
            )
            self.assertEqual(
                boundary.source_serialization_versions,
                EXPECTED_SOURCE_SURFACE_VERSIONS,
            )
            self.assertEqual(boundary.source_item_count, 6)
            self.assertEqual(boundary.proposal_count, 5)
            self.assertEqual(boundary.hitl_requirement_count, 5)
            self.assertEqual(boundary.explainability_signal_count, 6)
            self.assertEqual(boundary.governance_score, 1000)
            self.assertEqual(
                boundary.governance_score,
                self_evolution_governance_score(
                    source_item_count=boundary.source_item_count,
                    hitl_requirement_count=boundary.hitl_requirement_count,
                    explainability_signal_count=(
                        boundary.explainability_signal_count
                    ),
                    ownership_boundary_check_count=(
                        boundary.ownership_boundary_check_count
                    ),
                    cross_capability_governance_check_count=(
                        boundary.cross_capability_governance_check_count
                    ),
                    no_automation_weight=boundary.no_automation_weight,
                    safety_weight=boundary.safety_weight,
                ),
            )
            self.assertEqual(boundary.status, "guarded")
            self.assertEqual(boundary.priority, "guarded")
            self.assertTrue(boundary.hitl_required_before_governance_application)
            self.assertEqual(boundary.upstream_capabilities, UPSTREAM_CAPABILITIES)
            self.assertEqual(boundary.report_sections, SECONDARY_REPORT_SECTIONS)
            self.assertIn(
                "which V6.1 through V6.4",
                boundary.explainability_requirement,
            )
            self.assertIn("not mutated", boundary.downstream_impact_boundary)
            self.assertIn("Rollback feasibility", boundary.rollback_boundary)
            self.assertIn("No automation", boundary.no_automation_boundary)
            self.assertFalse(boundary.applied_governance_boundary_ids)
            self.assertFalse(boundary.enforced_safety_policy_ids)
            self.assertFalse(boundary.emitted_hitl_request_ids)
            self.assertFalse(boundary.activated_automation_ids)
            self.assertFalse(boundary.applied_evolution_proposal_ids)
            self.assertFalse(boundary.executed_rollback_ids)
            self.assertFalse(boundary.generated_report_artifact_ids)
            self.assertFalse(boundary.written_storage_record_ids)
            self.assertFalse(boundary.provider_execution_ids)
            self.assertFalse(boundary.mutated_output_ids)
            self.assertFalse(boundary.governance_policy_enforcement_implemented)
            self.assertFalse(boundary.safety_policy_enforcement_implemented)
            self.assertFalse(boundary.hitl_request_emitted)
            self.assertFalse(boundary.automation_activation_implemented)
            self.assertFalse(boundary.proposal_application_implemented)
            self.assertFalse(boundary.runtime_evolution_implemented)
            self.assertTrue(boundary.advisory_only)

        prompt_boundary = self_evolution_governance_boundary_by_roadmap_item(
            "Prompt Evolution",
            plan,
        )
        self.assertIsNotNone(prompt_boundary)
        assert prompt_boundary is not None
        self.assertEqual(prompt_boundary.plan_role, "prompt_evolution")
        self.assertEqual(
            prompt_boundary.boundary_id,
            "self_evolution_governance::prompt_evolution",
        )
        self.assertIs(
            self_evolution_governance_boundary_by_id(
                "self_evolution_governance::prompt_evolution",
                plan,
            ),
            prompt_boundary,
        )
        self.assertEqual(
            len(self_evolution_governance_boundaries_for_status("guarded", plan)),
            22,
        )
        self.assertEqual(
            len(self_evolution_governance_boundaries_for_priority("guarded", plan)),
            22,
        )

    def test_governance_rejects_mismatched_or_mutating_payloads(self) -> None:
        plan = build_self_evolution_governance()
        payload = plan.model_dump(mode="json")
        payload["governance_boundary_ids"] = ("missing",) + tuple(
            payload["governance_boundary_ids"][1:]
        )

        with self.assertRaisesRegex(ValueError, "governance_boundary_ids must match"):
            SelfEvolutionGovernancePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["applied_evolution_proposal_ids"] = (plan.proposal_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "governance plan mutation ids must be empty",
        ):
            SelfEvolutionGovernancePlan(**payload)

        boundary_payload = plan.governance_boundaries[0].model_dump(mode="json")
        boundary_payload["emitted_hitl_request_ids"] = ("hitl_request",)

        with self.assertRaisesRegex(
            ValueError,
            "governance boundary mutation ids must be empty",
        ):
            SelfEvolutionGovernanceBoundary(**boundary_payload)

    def test_governance_does_not_change_routing_or_provider(self) -> None:
        request = AssistantRequest(
            query="Review the self evolution governance surface.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        secondary_surface = build_self_evolution_secondary_surface(
            route=RouteName.GENERATE,
        )
        plan = build_self_evolution_governance(secondary_surface)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.source_proposal_count, 110)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")


if __name__ == "__main__":
    unittest.main()
