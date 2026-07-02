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
    ResearchDecomposerPlan,
    build_research_decomposer,
    build_research_planner,
    research_decomposition_entries_for_confidence,
    research_decomposition_entries_for_status,
    research_decomposition_entry_by_id,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
EXPECTED_ROADMAP_ITEMS = ("Research Decomposer",)
REQUIRED_ENTRY_FIELDS = {
    "entry_id",
    "decomposition_kind",
    "status",
    "confidence",
    "route_name",
    "task_type",
    "execution_mode_id",
    "decomposition_axis",
    "research_planner_entry_ids",
    "research_planner_entry_count",
    "source_count",
    "domain_count",
    "decomposition_summary",
    "objective_structure_score",
    "evidence_thread_score",
    "governance_alignment_score",
    "mutation_risk_score",
    "governance_weight",
    "decomposition_score",
    "hitl_required_before_decomposition_execution",
    "context_tags",
    "explainability_notes",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "research_decomposer_capability_implemented",
    "research_decomposer_metadata_implemented",
    "research_planner_metadata_used",
    "research_decomposition_execution_implemented",
    "subtask_creation_execution_implemented",
    "research_plan_execution_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_control_implemented",
    "workflow_execution_implemented",
    "paper_research_execution_implemented",
    "web_research_execution_implemented",
    "external_source_fetch_implemented",
    "live_source_validation_implemented",
    "source_credibility_scoring_implemented",
    "contradiction_detection_implemented",
    "kb_enrichment_implemented",
    "kb_storage_write_implemented",
    "retrieval_configuration_mutation_implemented",
    "retrieval_execution_implemented",
    "ranking_mutation_implemented",
    "provider_provisioning_implemented",
    "api_key_inference_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "advisory_only",
}


class ResearchDecomposerTests(unittest.TestCase):
    def test_plan_builds_advisory_research_decomposition_metadata(self) -> None:
        plan = build_research_decomposer(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "research_decomposer")
        self.assertEqual(plan.serialization_version, "research_decomposer_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.research_planner_role, "research_planner")
        self.assertEqual(
            plan.research_planner_serialization_version,
            "research_planner_plan.v1",
        )
        self.assertEqual(plan.research_planner_entry_count, 5)
        self.assertEqual(plan.covered_roadmap_items, EXPECTED_ROADMAP_ITEMS)
        self.assertEqual(plan.covered_roadmap_item_count, 1)
        self.assertEqual(plan.source_count, 57)
        self.assertEqual(plan.domain_count, 43)
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.entry_count, 5)
        self.assertEqual(plan.candidate_entry_count, 1)
        self.assertEqual(plan.review_required_entry_count, 3)
        self.assertEqual(plan.guarded_entry_count, 1)
        self.assertEqual(plan.high_confidence_entry_count, 3)
        self.assertEqual(plan.hitl_required_entry_count, 5)
        self.assertFalse(plan.planned_decomposition_execution_ids)
        self.assertFalse(plan.generated_subtask_ids)
        self.assertFalse(plan.mutated_workflow_graph_ids)
        self.assertFalse(plan.fetched_external_source_ids)
        self.assertFalse(plan.written_storage_record_ids)
        self.assertEqual(plan.overall_decomposition_posture, "guarded")
        self.assertIn(
            "does not execute research decomposition",
            plan.authority_boundary,
        )
        self.assertIn("create subtasks", plan.authority_boundary)
        self.assertIn("mutate workflows", plan.authority_boundary)
        self.assertTrue(plan.research_decomposer_capability_implemented)
        self.assertTrue(plan.research_decomposer_metadata_implemented)
        self.assertTrue(plan.roadmap_item_covered)
        self.assertTrue(plan.research_planner_metadata_used)
        self.assertFalse(plan.research_decomposition_execution_implemented)
        self.assertFalse(plan.subtask_creation_execution_implemented)
        self.assertFalse(plan.research_plan_execution_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.paper_research_execution_implemented)
        self.assertFalse(plan.web_research_execution_implemented)
        self.assertFalse(plan.external_source_fetch_implemented)
        self.assertFalse(plan.live_source_validation_implemented)
        self.assertFalse(plan.source_credibility_scoring_implemented)
        self.assertFalse(plan.contradiction_detection_implemented)
        self.assertFalse(plan.kb_enrichment_implemented)
        self.assertFalse(plan.kb_storage_write_implemented)
        self.assertFalse(plan.retrieval_configuration_mutation_implemented)
        self.assertFalse(plan.retrieval_execution_implemented)
        self.assertFalse(plan.ranking_mutation_implemented)
        self.assertFalse(plan.provider_provisioning_implemented)
        self.assertFalse(plan.api_key_inference_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_entries_score_decomposition_without_execution(self) -> None:
        plan = build_research_decomposer(route="generate")
        planner_entry_ids = set(plan.research_planner_entry_ids)

        for entry in plan.entries:
            dumped = entry.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_ENTRY_FIELDS)
            self.assertEqual(
                entry.serialization_version,
                "research_decomposer_entry.v1",
            )
            self.assertEqual(entry.route_name, RouteName.GENERATE)
            self.assertEqual(
                entry.entry_id,
                f"research_decomposer::{entry.decomposition_kind}",
            )
            self.assertEqual(
                entry.research_planner_entry_count,
                len(entry.research_planner_entry_ids),
            )
            self.assertTrue(
                set(entry.research_planner_entry_ids).issubset(planner_entry_ids)
            )
            self.assertEqual(entry.source_count, plan.source_count)
            self.assertEqual(entry.domain_count, plan.domain_count)
            self.assertEqual(
                entry.decomposition_score,
                min(
                    1000,
                    max(
                        0,
                        entry.objective_structure_score * 3
                        + entry.evidence_thread_score * 2
                        + entry.governance_alignment_score * 3
                        + entry.mutation_risk_score * 2
                        + entry.governance_weight,
                    ),
                ),
            )
            self.assertIn("research_decomposer", entry.context_tags)
            self.assertIn(
                "research_decomposition_execution",
                entry.blocked_runtime_behaviors,
            )
            self.assertIn("subtask_creation_execution", entry.blocked_runtime_behaviors)
            self.assertIn("workflow_graph_mutation", entry.blocked_runtime_behaviors)
            self.assertTrue(entry.hitl_required_before_decomposition_execution)
            self.assertTrue(entry.research_decomposer_capability_implemented)
            self.assertTrue(entry.research_decomposer_metadata_implemented)
            self.assertTrue(entry.research_planner_metadata_used)
            self.assertFalse(entry.research_decomposition_execution_implemented)
            self.assertFalse(entry.subtask_creation_execution_implemented)
            self.assertFalse(entry.workflow_graph_mutation_implemented)
            self.assertFalse(entry.workflow_control_implemented)
            self.assertFalse(entry.paper_research_execution_implemented)
            self.assertFalse(entry.web_research_execution_implemented)
            self.assertFalse(entry.external_source_fetch_implemented)
            self.assertFalse(entry.live_source_validation_implemented)
            self.assertFalse(entry.source_credibility_scoring_implemented)
            self.assertFalse(entry.contradiction_detection_implemented)
            self.assertFalse(entry.kb_enrichment_implemented)
            self.assertFalse(entry.kb_storage_write_implemented)
            self.assertFalse(entry.retrieval_configuration_mutation_implemented)
            self.assertFalse(entry.provider_provisioning_implemented)
            self.assertFalse(entry.api_key_inference_implemented)
            self.assertFalse(entry.provider_model_routing_implemented)
            self.assertFalse(entry.provider_execution_implemented)
            self.assertFalse(entry.generated_output_mutation_implemented)
            self.assertFalse(entry.runtime_evolution_implemented)
            self.assertTrue(entry.advisory_only)

        objective = research_decomposition_entry_by_id(
            "research_decomposer::research_objective_decomposition",
            plan,
        )
        self.assertIsNotNone(objective)
        assert objective is not None
        self.assertEqual(objective.status, "guarded")
        self.assertEqual(objective.confidence, "guarded")
        self.assertEqual(
            len(research_decomposition_entries_for_status("review_required", plan)),
            3,
        )
        self.assertEqual(
            len(research_decomposition_entries_for_confidence("high", plan)),
            2,
        )

    def test_plan_rejects_mismatched_decomposition_metadata(self) -> None:
        plan = build_research_decomposer()
        payload = plan.model_dump(mode="json")
        payload["entry_ids"] = ("missing",) + tuple(payload["entry_ids"][1:])

        with self.assertRaisesRegex(ValueError, "entry_ids must match"):
            ResearchDecomposerPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_decomposition_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_decomposition_score must match",
        ):
            ResearchDecomposerPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["generated_subtask_ids"] = (plan.entry_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "generated_subtask_ids must remain empty",
        ):
            ResearchDecomposerPlan(**payload)

    def test_decomposer_composes_with_research_planner_metadata(self) -> None:
        planner = build_research_planner(route=RouteName.REVIEW)
        plan = build_research_decomposer(
            route=RouteName.REVIEW,
            research_planner=planner,
        )

        self.assertEqual(plan.route_name, RouteName.REVIEW)
        self.assertEqual(plan.checked_at, planner.checked_at)
        self.assertEqual(plan.research_planner_entry_ids, planner.entry_ids)
        self.assertEqual(plan.source_count, planner.source_count)
        self.assertEqual(plan.domain_count, planner.domain_count)
        self.assertTrue(plan.research_planner_metadata_used)
        self.assertFalse(plan.research_decomposition_execution_implemented)

    def test_research_decomposer_preserves_routing_and_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Decompose a governed research pass for a p5.js shader question.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_research_decomposer(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_research_decomposer_does_not_declare_runtime_mutation_terms(
        self,
    ) -> None:
        plan = build_research_decomposer(route=RouteName.GENERATE)
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
                        entry.decomposition_kind,
                        entry.status,
                        entry.confidence,
                        entry.decomposition_axis,
                        *entry.research_planner_entry_ids,
                        entry.decomposition_summary,
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
            "execute_research_decomposition(",
            "create_subtask(",
            "execute_research_plan(",
            "mutate_workflow_graph(",
            "control_workflow(",
            "execute_workflow(",
            "perform_paper_research(",
            "perform_web_research(",
            "fetch_external_source(",
            "validate_source_live(",
            "score_source_credibility(",
            "detect_contradiction(",
            "enrich_kb(",
            "write_kb_storage(",
            "mutate_retrieval_config(",
            "execute_retrieval(",
            "mutate_ranking(",
            "provision_provider(",
            "infer_api_key(",
            "route_provider(",
            "execute_provider(",
            "write_storage(",
            "modify_output(",
            "apply_runtime_evolution(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
