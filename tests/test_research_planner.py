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
    ResearchPlannerPlan,
    build_research_planner,
    research_planning_entries_for_confidence,
    research_planning_entries_for_status,
    research_planning_entry_by_id,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
EXPECTED_ROADMAP_ITEMS = ("Research Planner",)
REQUIRED_ENTRY_FIELDS = {
    "entry_id",
    "planning_kind",
    "status",
    "confidence",
    "route_name",
    "task_type",
    "execution_mode_id",
    "planning_axis",
    "source_ids",
    "source_count",
    "domain_count",
    "source_type_count",
    "planning_summary",
    "scope_clarity_score",
    "source_strategy_score",
    "governance_alignment_score",
    "mutation_risk_score",
    "governance_weight",
    "planning_score",
    "hitl_required_before_research_execution",
    "context_tags",
    "explainability_notes",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "research_planner_capability_implemented",
    "research_planner_metadata_implemented",
    "official_source_registry_used",
    "research_plan_execution_implemented",
    "research_task_decomposition_implemented",
    "paper_research_execution_implemented",
    "web_research_execution_implemented",
    "external_source_fetch_implemented",
    "paper_download_implemented",
    "web_browsing_implemented",
    "live_source_validation_implemented",
    "source_credibility_scoring_implemented",
    "contradiction_detection_implemented",
    "research_confidence_scoring_implemented",
    "research_gap_discovery_implemented",
    "research_recommendation_generation_implemented",
    "kb_enrichment_implemented",
    "kb_storage_write_implemented",
    "source_registry_mutation_implemented",
    "retrieval_configuration_mutation_implemented",
    "retrieval_execution_implemented",
    "ranking_mutation_implemented",
    "provider_provisioning_implemented",
    "api_key_inference_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "agent_invocation_implemented",
    "workflow_control_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_execution_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "advisory_only",
}


class ResearchPlannerTests(unittest.TestCase):
    def test_plan_builds_advisory_research_planning_metadata(self) -> None:
        plan = build_research_planner(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "research_planner")
        self.assertEqual(plan.serialization_version, "research_planner_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.source_registry_role, "approved_official_sources")
        self.assertEqual(
            plan.source_registry_serialization_version,
            "official_source_registry.v1",
        )
        self.assertEqual(plan.covered_roadmap_items, EXPECTED_ROADMAP_ITEMS)
        self.assertEqual(plan.covered_roadmap_item_count, 1)
        self.assertEqual(plan.source_count, 57)
        self.assertEqual(len(plan.source_ids), 57)
        self.assertEqual(plan.domain_count, 43)
        self.assertEqual(plan.source_type_count, 4)
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.entry_count, 5)
        self.assertEqual(plan.candidate_entry_count, 1)
        self.assertEqual(plan.review_required_entry_count, 3)
        self.assertEqual(plan.guarded_entry_count, 1)
        self.assertEqual(plan.high_confidence_entry_count, 3)
        self.assertEqual(plan.hitl_required_entry_count, 5)
        self.assertFalse(plan.planned_research_execution_ids)
        self.assertFalse(plan.decomposed_research_task_ids)
        self.assertFalse(plan.paper_research_execution_ids)
        self.assertFalse(plan.web_research_execution_ids)
        self.assertFalse(plan.fetched_external_source_ids)
        self.assertFalse(plan.kb_enrichment_execution_ids)
        self.assertFalse(plan.written_storage_record_ids)
        self.assertFalse(plan.mutated_retrieval_config_ids)
        self.assertEqual(plan.overall_planning_posture, "guarded")
        self.assertIn("does not execute research plans", plan.authority_boundary)
        self.assertIn("does not execute research", plan.authority_boundary)
        self.assertIn("write storage", plan.authority_boundary)
        self.assertTrue(plan.research_planner_capability_implemented)
        self.assertTrue(plan.research_planner_metadata_implemented)
        self.assertTrue(plan.roadmap_item_covered)
        self.assertTrue(plan.official_source_registry_used)
        self.assertFalse(plan.research_plan_execution_implemented)
        self.assertFalse(plan.research_task_decomposition_implemented)
        self.assertFalse(plan.paper_research_execution_implemented)
        self.assertFalse(plan.web_research_execution_implemented)
        self.assertFalse(plan.external_source_fetch_implemented)
        self.assertFalse(plan.paper_download_implemented)
        self.assertFalse(plan.web_browsing_implemented)
        self.assertFalse(plan.live_source_validation_implemented)
        self.assertFalse(plan.source_credibility_scoring_implemented)
        self.assertFalse(plan.contradiction_detection_implemented)
        self.assertFalse(plan.research_confidence_scoring_implemented)
        self.assertFalse(plan.research_gap_discovery_implemented)
        self.assertFalse(plan.research_recommendation_generation_implemented)
        self.assertFalse(plan.kb_enrichment_implemented)
        self.assertFalse(plan.kb_storage_write_implemented)
        self.assertFalse(plan.retrieval_configuration_mutation_implemented)
        self.assertFalse(plan.retrieval_execution_implemented)
        self.assertFalse(plan.ranking_mutation_implemented)
        self.assertFalse(plan.provider_provisioning_implemented)
        self.assertFalse(plan.api_key_inference_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_entries_score_planning_without_execution(self) -> None:
        plan = build_research_planner(route="generate")
        plan_source_ids = set(plan.source_ids)

        for entry in plan.entries:
            dumped = entry.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_ENTRY_FIELDS)
            self.assertEqual(
                entry.serialization_version,
                "research_planner_entry.v1",
            )
            self.assertEqual(entry.route_name, RouteName.GENERATE)
            self.assertEqual(
                entry.entry_id,
                f"research_planner::{entry.planning_kind}",
            )
            self.assertEqual(entry.source_count, len(entry.source_ids))
            self.assertTrue(set(entry.source_ids).issubset(plan_source_ids))
            self.assertEqual(
                entry.planning_score,
                min(
                    1000,
                    max(
                        0,
                        entry.scope_clarity_score * 3
                        + entry.source_strategy_score * 2
                        + entry.governance_alignment_score * 3
                        + entry.mutation_risk_score * 2
                        + entry.governance_weight,
                    ),
                ),
            )
            self.assertIn("research_planner", entry.context_tags)
            self.assertIn("research_plan_execution", entry.blocked_runtime_behaviors)
            self.assertIn("web_browsing", entry.blocked_runtime_behaviors)
            self.assertIn("kb_storage_write", entry.blocked_runtime_behaviors)
            self.assertTrue(entry.hitl_required_before_research_execution)
            self.assertTrue(entry.research_planner_capability_implemented)
            self.assertTrue(entry.research_planner_metadata_implemented)
            self.assertTrue(entry.official_source_registry_used)
            self.assertFalse(entry.research_plan_execution_implemented)
            self.assertFalse(entry.research_task_decomposition_implemented)
            self.assertFalse(entry.paper_research_execution_implemented)
            self.assertFalse(entry.web_research_execution_implemented)
            self.assertFalse(entry.external_source_fetch_implemented)
            self.assertFalse(entry.paper_download_implemented)
            self.assertFalse(entry.web_browsing_implemented)
            self.assertFalse(entry.live_source_validation_implemented)
            self.assertFalse(entry.source_credibility_scoring_implemented)
            self.assertFalse(entry.contradiction_detection_implemented)
            self.assertFalse(entry.research_confidence_scoring_implemented)
            self.assertFalse(entry.research_gap_discovery_implemented)
            self.assertFalse(entry.research_recommendation_generation_implemented)
            self.assertFalse(entry.kb_enrichment_implemented)
            self.assertFalse(entry.kb_storage_write_implemented)
            self.assertFalse(entry.retrieval_configuration_mutation_implemented)
            self.assertFalse(entry.retrieval_execution_implemented)
            self.assertFalse(entry.ranking_mutation_implemented)
            self.assertFalse(entry.provider_provisioning_implemented)
            self.assertFalse(entry.api_key_inference_implemented)
            self.assertFalse(entry.provider_model_routing_implemented)
            self.assertFalse(entry.provider_execution_implemented)
            self.assertFalse(entry.workflow_control_implemented)
            self.assertFalse(entry.generated_output_mutation_implemented)
            self.assertFalse(entry.runtime_evolution_implemented)
            self.assertTrue(entry.advisory_only)

        scope = research_planning_entry_by_id(
            "research_planner::research_scope_framing",
            plan,
        )
        self.assertIsNotNone(scope)
        assert scope is not None
        self.assertEqual(scope.status, "guarded")
        self.assertEqual(scope.confidence, "guarded")
        self.assertEqual(
            len(research_planning_entries_for_status("review_required", plan)),
            3,
        )
        self.assertEqual(
            len(research_planning_entries_for_confidence("high", plan)),
            2,
        )

    def test_plan_rejects_mismatched_research_planning_metadata(self) -> None:
        plan = build_research_planner()
        payload = plan.model_dump(mode="json")
        payload["entry_ids"] = ("missing",) + tuple(payload["entry_ids"][1:])

        with self.assertRaisesRegex(ValueError, "entry_ids must match"):
            ResearchPlannerPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_planning_score"] -= 1

        with self.assertRaisesRegex(ValueError, "overall_planning_score must match"):
            ResearchPlannerPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["planned_research_execution_ids"] = (plan.entry_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "planned_research_execution_ids must remain empty",
        ):
            ResearchPlannerPlan(**payload)

    def test_research_planner_preserves_routing_and_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Plan a governed research pass for a p5.js shader question.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_research_planner(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_research_planner_does_not_declare_runtime_mutation_terms(self) -> None:
        plan = build_research_planner(route=RouteName.GENERATE)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *plan.covered_roadmap_items,
                *(
                    field
                    for entry in plan.entries
                    for field in (
                        entry.entry_id,
                        entry.planning_kind,
                        entry.status,
                        entry.confidence,
                        entry.planning_axis,
                        *entry.source_ids,
                        entry.planning_summary,
                        *entry.context_tags,
                        *entry.explainability_notes,
                        *entry.advisory_actions,
                        *entry.evidence,
                        *entry.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "execute_research_plan(",
            "decompose_research_task(",
            "perform_paper_research(",
            "perform_web_research(",
            "fetch_external_source(",
            "download_paper(",
            "browse_web(",
            "validate_source_live(",
            "score_source_credibility(",
            "detect_contradiction(",
            "score_research_confidence(",
            "discover_research_gap(",
            "generate_research_recommendation(",
            "enrich_kb(",
            "write_kb_storage(",
            "mutate_source_registry(",
            "mutate_retrieval_config(",
            "execute_retrieval(",
            "mutate_ranking(",
            "provision_provider(",
            "infer_api_key(",
            "route_provider(",
            "execute_provider(",
            "invoke_agent(",
            "control_workflow(",
            "mutate_workflow_graph(",
            "execute_workflow(",
            "write_storage(",
            "modify_output(",
            "apply_runtime_evolution(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
