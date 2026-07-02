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
    ResearchGovernancePlan,
    build_research_governance,
    research_governance_boundaries_for_status,
    research_governance_boundary_by_id,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
EXPECTED_SOURCE_ROLES = (
    "research_secondary_surface",
    "learning_governance",
    "hitl_budget_gate",
    "routing_explainability",
)
EXPECTED_ROADMAP_ITEMS = (
    "Research Planner",
    "Research Decomposer",
    "Paper Research",
    "Web Research",
    "Cross-source Comparison",
    "Knowledge Distillation",
    "Automatic KB Enrichment",
    "Research Reports",
    "Research Memory",
    "Source Validation Engine",
    "Source Credibility Engine",
    "Contradiction Detection",
    "Research Confidence Engine",
    "Research Gap Discovery",
    "Research Recommendation Engine",
    "Research Execution Policy",
    "Research HITL Policies",
    "Creative Research Engine",
    "Cross-domain Inspiration Discovery",
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
    "roadmap_items",
    "roadmap_item_count",
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
    "research_governance_implemented",
    "governance_boundary_metadata_implemented",
    "hitl_boundary_metadata_implemented",
    "explainability_boundary_metadata_implemented",
    "no_automation_boundary_metadata_implemented",
    "all_roadmap_items_traceable",
    "v5_v6_governance_sources_used",
    "governance_policy_enforcement_implemented",
    "safety_policy_enforcement_implemented",
    "hitl_request_emitted",
    "human_input_request_implemented",
    "automation_activation_implemented",
    "secondary_surface_activation_implemented",
    "adaptive_learning_application_implemented",
    "execution_policy_application_implemented",
    "research_execution_implemented",
    "research_plan_mutation_implemented",
    "research_task_creation_implemented",
    "paper_research_execution_implemented",
    "web_research_execution_implemented",
    "external_source_fetch_implemented",
    "web_browsing_implemented",
    "paper_download_implemented",
    "cross_source_comparison_execution_implemented",
    "knowledge_distillation_execution_implemented",
    "kb_enrichment_execution_implemented",
    "kb_storage_write_implemented",
    "research_report_generation_implemented",
    "research_memory_write_implemented",
    "source_validation_execution_implemented",
    "source_credibility_scoring_execution_implemented",
    "contradiction_detection_execution_implemented",
    "research_confidence_scoring_execution_implemented",
    "research_gap_discovery_execution_implemented",
    "research_recommendation_generation_implemented",
    "research_execution_policy_application_implemented",
    "hitl_decision_application_implemented",
    "inspiration_discovery_execution_implemented",
    "live_cross_domain_search_implemented",
    "routing_application_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "agent_invocation_implemented",
    "workflow_control_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_execution_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "advisory_only",
}


class ResearchGovernanceTests(unittest.TestCase):
    def test_plan_builds_governance_and_safety_metadata(self) -> None:
        plan = build_research_governance(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "research_governance_safety")
        self.assertEqual(
            plan.serialization_version,
            "research_governance_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.source_plan_roles, EXPECTED_SOURCE_ROLES)
        self.assertEqual(
            plan.source_plan_serialization_versions,
            (
                "research_secondary_surface_plan.v1",
                "learning_governance_plan.v1",
                "hitl_budget_gate_plan.v1",
                "routing_explainability_plan.v1",
            ),
        )
        self.assertEqual(plan.source_item_count, 19)
        self.assertEqual(len(plan.source_item_ids), 19)
        self.assertEqual(plan.covered_roadmap_items, EXPECTED_ROADMAP_ITEMS)
        self.assertEqual(plan.covered_roadmap_item_count, 19)
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.boundary_count, 5)
        self.assertEqual(plan.guarded_boundary_count, 5)
        self.assertEqual(plan.hitl_required_boundary_count, 5)
        self.assertFalse(plan.applied_governance_boundary_ids)
        self.assertFalse(plan.enforced_safety_policy_ids)
        self.assertFalse(plan.emitted_hitl_request_ids)
        self.assertFalse(plan.requested_human_input_ids)
        self.assertFalse(plan.activated_automation_ids)
        self.assertFalse(plan.activated_secondary_surface_ids)
        self.assertFalse(plan.applied_learning_ids)
        self.assertFalse(plan.applied_execution_policy_ids)
        self.assertFalse(plan.executed_research_ids)
        self.assertFalse(plan.fetched_source_ids)
        self.assertFalse(plan.written_kb_record_ids)
        self.assertFalse(plan.mutated_workflow_ids)
        self.assertFalse(plan.mutated_output_ids)
        self.assertEqual(plan.overall_governance_posture, "guarded")
        self.assertIn("HITL", plan.authority_boundary)
        self.assertIn("no-automation boundaries", plan.authority_boundary)
        self.assertTrue(plan.research_governance_implemented)
        self.assertTrue(plan.governance_boundary_metadata_implemented)
        self.assertTrue(plan.hitl_boundary_metadata_implemented)
        self.assertTrue(plan.explainability_boundary_metadata_implemented)
        self.assertTrue(plan.no_automation_boundary_metadata_implemented)
        self.assertTrue(plan.all_roadmap_items_traceable)
        self.assertTrue(plan.v5_v6_governance_sources_used)
        self.assertFalse(plan.governance_policy_enforcement_implemented)
        self.assertFalse(plan.safety_policy_enforcement_implemented)
        self.assertFalse(plan.hitl_request_emitted)
        self.assertFalse(plan.human_input_request_implemented)
        self.assertFalse(plan.automation_activation_implemented)
        self.assertFalse(plan.secondary_surface_activation_implemented)
        self.assertFalse(plan.adaptive_learning_application_implemented)
        self.assertFalse(plan.execution_policy_application_implemented)
        self.assertFalse(plan.research_execution_implemented)
        self.assertFalse(plan.research_plan_mutation_implemented)
        self.assertFalse(plan.research_task_creation_implemented)
        self.assertFalse(plan.paper_research_execution_implemented)
        self.assertFalse(plan.web_research_execution_implemented)
        self.assertFalse(plan.external_source_fetch_implemented)
        self.assertFalse(plan.web_browsing_implemented)
        self.assertFalse(plan.paper_download_implemented)
        self.assertFalse(plan.kb_enrichment_execution_implemented)
        self.assertFalse(plan.kb_storage_write_implemented)
        self.assertFalse(plan.research_report_generation_implemented)
        self.assertFalse(plan.research_memory_write_implemented)
        self.assertFalse(plan.research_recommendation_generation_implemented)
        self.assertFalse(plan.research_execution_policy_application_implemented)
        self.assertFalse(plan.hitl_decision_application_implemented)
        self.assertFalse(plan.inspiration_discovery_execution_implemented)
        self.assertFalse(plan.live_cross_domain_search_implemented)
        self.assertFalse(plan.routing_application_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_boundaries_score_governance_without_enforcement(self) -> None:
        plan = build_research_governance(route="generate")
        source_items = set(plan.source_item_ids)
        source_roles = set(plan.source_plan_roles)

        for boundary in plan.boundaries:
            dumped = boundary.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_BOUNDARY_FIELDS)
            self.assertEqual(
                boundary.serialization_version,
                "research_governance_boundary.v1",
            )
            self.assertEqual(boundary.route_name, RouteName.GENERATE)
            self.assertEqual(
                boundary.boundary_id,
                f"research_governance::{boundary.boundary_kind}",
            )
            self.assertEqual(boundary.roadmap_item_count, len(boundary.roadmap_items))
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
            self.assertTrue(boundary.research_governance_implemented)
            self.assertTrue(boundary.governance_boundary_metadata_implemented)
            self.assertTrue(boundary.hitl_boundary_metadata_implemented)
            self.assertTrue(boundary.explainability_boundary_metadata_implemented)
            self.assertTrue(boundary.no_automation_boundary_metadata_implemented)
            self.assertTrue(boundary.all_roadmap_items_traceable)
            self.assertFalse(boundary.governance_policy_enforcement_implemented)
            self.assertFalse(boundary.safety_policy_enforcement_implemented)
            self.assertFalse(boundary.hitl_request_emitted)
            self.assertFalse(boundary.human_input_request_implemented)
            self.assertFalse(boundary.automation_activation_implemented)
            self.assertFalse(boundary.secondary_surface_activation_implemented)
            self.assertFalse(boundary.adaptive_learning_application_implemented)
            self.assertFalse(boundary.execution_policy_application_implemented)
            self.assertFalse(boundary.research_execution_implemented)
            self.assertFalse(boundary.research_plan_mutation_implemented)
            self.assertFalse(boundary.research_task_creation_implemented)
            self.assertFalse(boundary.external_source_fetch_implemented)
            self.assertFalse(boundary.web_browsing_implemented)
            self.assertFalse(boundary.paper_download_implemented)
            self.assertFalse(boundary.kb_enrichment_execution_implemented)
            self.assertFalse(boundary.kb_storage_write_implemented)
            self.assertFalse(boundary.routing_application_implemented)
            self.assertFalse(boundary.provider_model_routing_implemented)
            self.assertFalse(boundary.provider_execution_implemented)
            self.assertFalse(boundary.agent_invocation_implemented)
            self.assertFalse(boundary.workflow_control_implemented)
            self.assertFalse(boundary.workflow_graph_mutation_implemented)
            self.assertFalse(boundary.workflow_execution_implemented)
            self.assertFalse(boundary.prompt_mutation_implemented)
            self.assertFalse(boundary.persistent_storage_write_implemented)
            self.assertFalse(boundary.generated_output_mutation_implemented)
            self.assertFalse(boundary.runtime_evolution_implemented)
            self.assertTrue(boundary.advisory_only)

        no_automation = research_governance_boundary_by_id(
            "research_governance::creative_inspiration_no_automation_governance",
            plan,
        )
        self.assertIsNotNone(no_automation)
        assert no_automation is not None
        self.assertEqual(no_automation.status, "guarded")
        self.assertEqual(no_automation.explainability_signal_count, 6)
        self.assertEqual(no_automation.roadmap_item_count, 3)
        self.assertEqual(
            len(research_governance_boundaries_for_status("guarded", plan)),
            5,
        )

    def test_plan_rejects_mismatched_governance_metadata(self) -> None:
        plan = build_research_governance()
        payload = plan.model_dump(mode="json")
        payload["boundary_ids"] = ("missing",) + tuple(payload["boundary_ids"][1:])

        with self.assertRaisesRegex(ValueError, "boundary_ids must match"):
            ResearchGovernancePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_governance_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_governance_score must match",
        ):
            ResearchGovernancePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["emitted_hitl_request_ids"] = (plan.boundary_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "emitted_hitl_request_ids must remain empty",
        ):
            ResearchGovernancePlan(**payload)

    def test_governance_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Review autonomous research governance.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_research_governance(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_governance_does_not_declare_enforcement_terms(self) -> None:
        plan = build_research_governance(route=RouteName.GENERATE)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *plan.covered_roadmap_items,
                *(
                    field
                    for boundary in plan.boundaries
                    for field in (
                        boundary.boundary_id,
                        boundary.boundary_kind,
                        boundary.status,
                        boundary.priority,
                        boundary.governed_area,
                        *boundary.roadmap_items,
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
            "activate_secondary_surface(",
            "apply_adaptive_learning(",
            "apply_execution_policy(",
            "execute_research(",
            "mutate_research_plan(",
            "create_research_task(",
            "execute_paper_research(",
            "execute_web_research(",
            "fetch_external_source(",
            "browse_web(",
            "download_paper(",
            "execute_cross_source_comparison(",
            "execute_knowledge_distillation(",
            "execute_kb_enrichment(",
            "write_kb_storage(",
            "generate_research_report(",
            "write_research_memory(",
            "execute_source_validation(",
            "score_source_credibility(",
            "execute_contradiction_detection(",
            "score_research_confidence(",
            "discover_research_gap(",
            "generate_research_recommendation(",
            "apply_research_execution_policy(",
            "apply_hitl_decision(",
            "execute_inspiration_discovery(",
            "perform_live_cross_domain_search(",
            "apply_routing(",
            "route_provider(",
            "execute_provider(",
            "invoke_agent(",
            "control_workflow(",
            "mutate_workflow_graph(",
            "execute_workflow(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
            "apply_runtime_evolution(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
