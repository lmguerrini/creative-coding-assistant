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
    CreativeMemoryGovernancePlan,
    build_creative_memory_governance,
    creative_memory_governance_boundaries_for_status,
    creative_memory_governance_boundary_by_id,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
EXPECTED_SOURCE_ROLES = (
    "creative_memory_secondary_surface",
    "learning_governance",
    "hitl_budget_gate",
    "routing_explainability",
)
REQUIRED_BOUNDARY_FIELDS = {
    "boundary_id",
    "boundary_kind",
    "status",
    "priority",
    "route_name",
    "task_type",
    "execution_mode_id",
    "governed_area",
    "source_plan_roles",
    "source_serialization_versions",
    "source_item_ids",
    "source_item_count",
    "hitl_requirement_count",
    "explainability_signal_count",
    "no_automation_weight",
    "safety_weight",
    "governance_score",
    "hitl_required_before_governance_application",
    "governed_surface_summary",
    "review_requirement",
    "explainability_requirement",
    "no_automation_boundary",
    "safety_boundary",
    "governance_tags",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "creative_memory_governance_implemented",
    "governance_boundary_metadata_implemented",
    "hitl_boundary_metadata_implemented",
    "explainability_boundary_metadata_implemented",
    "no_automation_boundary_metadata_implemented",
    "v5_v6_governance_sources_used",
    "governance_policy_enforcement_implemented",
    "safety_policy_enforcement_implemented",
    "hitl_request_emitted",
    "human_input_request_implemented",
    "automation_activation_implemented",
    "preference_learning_execution_implemented",
    "user_model_application_implemented",
    "memory_storage_write_implemented",
    "memory_retrieval_execution_implemented",
    "memory_consolidation_execution_implemented",
    "memory_conflict_resolution_execution_implemented",
    "routing_application_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "agent_invocation_implemented",
    "workflow_control_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_execution_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "advisory_only",
}


class CreativeMemoryGovernanceTests(unittest.TestCase):
    def test_plan_builds_governance_and_safety_metadata(self) -> None:
        plan = build_creative_memory_governance(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "creative_memory_governance_safety")
        self.assertEqual(
            plan.serialization_version,
            "creative_memory_governance_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.source_plan_roles, EXPECTED_SOURCE_ROLES)
        self.assertEqual(
            plan.source_plan_serialization_versions,
            (
                "creative_memory_secondary_surface_plan.v1",
                "learning_governance_plan.v1",
                "hitl_budget_gate_plan.v1",
                "routing_explainability_plan.v1",
            ),
        )
        self.assertEqual(plan.source_item_count, 19)
        self.assertEqual(len(plan.source_item_ids), 19)
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.boundary_count, 5)
        self.assertEqual(plan.guarded_boundary_count, 5)
        self.assertEqual(plan.hitl_required_boundary_count, 5)
        self.assertFalse(plan.applied_governance_boundary_ids)
        self.assertFalse(plan.enforced_safety_policy_ids)
        self.assertFalse(plan.emitted_hitl_request_ids)
        self.assertFalse(plan.requested_human_input_ids)
        self.assertFalse(plan.activated_automation_ids)
        self.assertFalse(plan.mutated_output_ids)
        self.assertEqual(plan.overall_governance_posture, "guarded")
        self.assertIn("HITL", plan.authority_boundary)
        self.assertIn("no-automation boundaries", plan.authority_boundary)
        self.assertTrue(plan.creative_memory_governance_implemented)
        self.assertTrue(plan.governance_boundary_metadata_implemented)
        self.assertTrue(plan.hitl_boundary_metadata_implemented)
        self.assertTrue(plan.explainability_boundary_metadata_implemented)
        self.assertTrue(plan.no_automation_boundary_metadata_implemented)
        self.assertTrue(plan.v5_v6_governance_sources_used)
        self.assertFalse(plan.governance_policy_enforcement_implemented)
        self.assertFalse(plan.safety_policy_enforcement_implemented)
        self.assertFalse(plan.hitl_request_emitted)
        self.assertFalse(plan.human_input_request_implemented)
        self.assertFalse(plan.automation_activation_implemented)
        self.assertFalse(plan.preference_learning_execution_implemented)
        self.assertFalse(plan.user_model_application_implemented)
        self.assertFalse(plan.memory_storage_write_implemented)
        self.assertFalse(plan.memory_retrieval_execution_implemented)
        self.assertFalse(plan.memory_consolidation_execution_implemented)
        self.assertFalse(plan.memory_conflict_resolution_execution_implemented)
        self.assertFalse(plan.routing_application_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_boundaries_score_governance_without_enforcement(self) -> None:
        plan = build_creative_memory_governance(route="generate")
        source_items = set(plan.source_item_ids)
        source_roles = set(plan.source_plan_roles)

        for boundary in plan.boundaries:
            dumped = boundary.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_BOUNDARY_FIELDS)
            self.assertEqual(
                boundary.serialization_version,
                "creative_memory_governance_boundary.v1",
            )
            self.assertEqual(boundary.route_name, RouteName.GENERATE)
            self.assertEqual(
                boundary.boundary_id,
                f"creative_memory_governance::{boundary.boundary_kind}",
            )
            self.assertEqual(boundary.source_item_count, len(boundary.source_item_ids))
            self.assertTrue(set(boundary.source_item_ids).issubset(source_items))
            self.assertTrue(set(boundary.source_plan_roles).issubset(source_roles))
            self.assertEqual(
                boundary.governance_score,
                min(
                    1000,
                    max(
                        0,
                        boundary.source_item_count * 18
                        + boundary.hitl_requirement_count * 65
                        + boundary.explainability_signal_count * 25
                        + boundary.no_automation_weight
                        + boundary.safety_weight,
                    ),
                ),
            )
            self.assertEqual(boundary.status, "guarded")
            self.assertEqual(boundary.priority, "guarded")
            self.assertIn("governance_safety", boundary.governance_tags)
            self.assertIn("automation_activation", boundary.blocked_runtime_behaviors)
            self.assertIn("hitl_request_emission", boundary.blocked_runtime_behaviors)
            self.assertTrue(boundary.hitl_required_before_governance_application)
            self.assertTrue(boundary.creative_memory_governance_implemented)
            self.assertTrue(boundary.hitl_boundary_metadata_implemented)
            self.assertTrue(boundary.explainability_boundary_metadata_implemented)
            self.assertTrue(boundary.no_automation_boundary_metadata_implemented)
            self.assertFalse(boundary.governance_policy_enforcement_implemented)
            self.assertFalse(boundary.safety_policy_enforcement_implemented)
            self.assertFalse(boundary.hitl_request_emitted)
            self.assertFalse(boundary.human_input_request_implemented)
            self.assertFalse(boundary.automation_activation_implemented)
            self.assertFalse(boundary.preference_learning_execution_implemented)
            self.assertFalse(boundary.user_model_application_implemented)
            self.assertFalse(boundary.memory_storage_write_implemented)
            self.assertFalse(boundary.memory_retrieval_execution_implemented)
            self.assertFalse(boundary.memory_consolidation_execution_implemented)
            self.assertFalse(boundary.memory_conflict_resolution_execution_implemented)
            self.assertFalse(boundary.routing_application_implemented)
            self.assertFalse(boundary.provider_model_routing_implemented)
            self.assertFalse(boundary.provider_execution_implemented)
            self.assertFalse(boundary.agent_invocation_implemented)
            self.assertFalse(boundary.workflow_control_implemented)
            self.assertFalse(boundary.workflow_graph_mutation_implemented)
            self.assertFalse(boundary.workflow_execution_implemented)
            self.assertFalse(boundary.persistent_storage_write_implemented)
            self.assertFalse(boundary.generated_output_mutation_implemented)
            self.assertFalse(boundary.runtime_evolution_implemented)
            self.assertTrue(boundary.advisory_only)

        no_automation = creative_memory_governance_boundary_by_id(
            "creative_memory_governance::no_automation_governance",
            plan,
        )
        self.assertIsNotNone(no_automation)
        assert no_automation is not None
        self.assertEqual(no_automation.status, "guarded")
        self.assertEqual(no_automation.explainability_signal_count, 6)
        self.assertEqual(
            len(creative_memory_governance_boundaries_for_status("guarded", plan)),
            5,
        )

    def test_plan_rejects_mismatched_governance_metadata(self) -> None:
        plan = build_creative_memory_governance()
        payload = plan.model_dump(mode="json")
        payload["boundary_ids"] = ("missing",) + tuple(payload["boundary_ids"][1:])

        with self.assertRaisesRegex(ValueError, "boundary_ids must match"):
            CreativeMemoryGovernancePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_governance_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_governance_score must match",
        ):
            CreativeMemoryGovernancePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["emitted_hitl_request_ids"] = (plan.boundary_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "emitted_hitl_request_ids must remain empty",
        ):
            CreativeMemoryGovernancePlan(**payload)

    def test_governance_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Review creative memory governance.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_creative_memory_governance(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_governance_does_not_declare_enforcement_terms(self) -> None:
        plan = build_creative_memory_governance(route=RouteName.GENERATE)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for boundary in plan.boundaries
                    for field in (
                        boundary.boundary_id,
                        boundary.boundary_kind,
                        boundary.status,
                        boundary.priority,
                        boundary.governed_area,
                        *boundary.source_plan_roles,
                        *boundary.source_serialization_versions,
                        *boundary.source_item_ids,
                        boundary.governed_surface_summary,
                        boundary.review_requirement,
                        boundary.explainability_requirement,
                        boundary.no_automation_boundary,
                        boundary.safety_boundary,
                        *boundary.governance_tags,
                        *boundary.advisory_actions,
                        *boundary.evidence,
                        *boundary.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "enforce_governance(",
            "enforce_safety(",
            "emit_hitl_request(",
            "request_human_input(",
            "activate_automation(",
            "learn_preference(",
            "apply_user_model(",
            "write_memory(",
            "retrieve_memory(",
            "consolidate_memory(",
            "resolve_memory_conflict(",
            "apply_routing(",
            "route_provider(",
            "switch_provider(",
            "switch_model(",
            "execute_provider(",
            "invoke_agent(",
            "control_workflow(",
            "mutate_workflow_graph(",
            "execute_workflow(",
            "trigger_retry(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
            "apply_runtime_evolution(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
