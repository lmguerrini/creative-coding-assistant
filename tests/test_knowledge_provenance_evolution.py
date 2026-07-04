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
    KnowledgeProvenanceEvolutionPlan,
    build_knowledge_lifecycle_management,
    build_knowledge_provenance_evolution,
    knowledge_provenance_signal_by_id,
    knowledge_provenance_signals_for_confidence,
    knowledge_provenance_signals_for_status,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
EXPECTED_ROADMAP_ITEMS = ("Knowledge Provenance Evolution",)
REQUIRED_SIGNAL_FIELDS = {
    "signal_id",
    "signal_kind",
    "status",
    "confidence",
    "route_name",
    "task_type",
    "execution_mode_id",
    "provenance_axis",
    "knowledge_lifecycle_signal_ids",
    "knowledge_lifecycle_signal_count",
    "source_count",
    "domain_count",
    "provenance_signal_summary",
    "provenance_signal_score",
    "lineage_alignment_score",
    "governance_alignment_score",
    "mutation_risk_score",
    "governance_weight",
    "provenance_score",
    "hitl_required_before_provenance_evolution",
    "context_tags",
    "explainability_notes",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "knowledge_provenance_evolution_capability_implemented",
    "knowledge_provenance_evolution_metadata_implemented",
    "knowledge_lifecycle_metadata_used",
    "knowledge_provenance_evolution_execution_implemented",
    "provenance_graph_mutation_implemented",
    "provenance_record_write_implemented",
    "lineage_reconstruction_execution_implemented",
    "source_relinking_execution_implemented",
    "knowledge_lifecycle_management_execution_implemented",
    "lifecycle_stage_transition_implemented",
    "lifecycle_policy_mutation_implemented",
    "retention_policy_mutation_implemented",
    "archival_execution_implemented",
    "deprecation_execution_implemented",
    "deletion_execution_implemented",
    "lifecycle_record_write_implemented",
    "knowledge_consolidation_execution_implemented",
    "knowledge_merge_execution_implemented",
    "knowledge_deduplication_execution_implemented",
    "canonical_record_write_implemented",
    "consolidation_record_write_implemented",
    "kb_storage_write_implemented",
    "source_record_update_implemented",
    "retrieval_query_execution_implemented",
    "retrieval_configuration_mutation_implemented",
    "ranking_mutation_implemented",
    "embedding_request_execution_implemented",
    "embedding_refresh_execution_implemented",
    "vector_indexing_implemented",
    "vector_upsert_implemented",
    "documentation_fetch_execution_implemented",
    "provider_provisioning_implemented",
    "api_key_inference_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "agent_invocation_implemented",
    "workflow_control_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_execution_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "advisory_only",
}


class KnowledgeProvenanceEvolutionTests(unittest.TestCase):
    def test_plan_builds_knowledge_provenance_metadata(self) -> None:
        plan = build_knowledge_provenance_evolution(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "knowledge_provenance_evolution")
        self.assertEqual(plan.serialization_version, "knowledge_provenance_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.knowledge_lifecycle_role,
            "knowledge_lifecycle_management",
        )
        self.assertEqual(
            plan.knowledge_lifecycle_serialization_version,
            "knowledge_lifecycle_plan.v1",
        )
        self.assertEqual(len(plan.knowledge_lifecycle_signal_ids), 5)
        self.assertEqual(plan.knowledge_lifecycle_signal_count, 5)
        self.assertEqual(plan.covered_roadmap_items, EXPECTED_ROADMAP_ITEMS)
        self.assertEqual(plan.covered_roadmap_item_count, 1)
        self.assertEqual(plan.source_count, 57)
        self.assertEqual(plan.domain_count, 43)
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.signal_count, 5)
        self.assertEqual(plan.candidate_signal_count, 1)
        self.assertEqual(plan.review_required_signal_count, 3)
        self.assertEqual(plan.guarded_signal_count, 1)
        self.assertEqual(plan.high_confidence_signal_count, 3)
        self.assertEqual(plan.hitl_required_signal_count, 5)
        self.assertFalse(plan.planned_provenance_evolution_ids)
        self.assertFalse(plan.mutated_provenance_graph_ids)
        self.assertFalse(plan.written_provenance_record_ids)
        self.assertFalse(plan.reconstructed_lineage_ids)
        self.assertFalse(plan.relinked_source_ids)
        self.assertFalse(plan.written_kb_record_ids)
        self.assertEqual(plan.overall_provenance_posture, "guarded")
        self.assertIn("does not execute provenance evolution", plan.authority_boundary)
        self.assertIn("write KB storage", plan.authority_boundary)
        self.assertTrue(plan.knowledge_provenance_evolution_capability_implemented)
        self.assertTrue(plan.knowledge_provenance_evolution_metadata_implemented)
        self.assertTrue(plan.roadmap_item_covered)
        self.assertTrue(plan.knowledge_lifecycle_metadata_used)
        self.assertFalse(plan.knowledge_provenance_evolution_execution_implemented)
        self.assertFalse(plan.provenance_graph_mutation_implemented)
        self.assertFalse(plan.provenance_record_write_implemented)
        self.assertFalse(plan.lineage_reconstruction_execution_implemented)
        self.assertFalse(plan.source_relinking_execution_implemented)
        self.assertFalse(plan.knowledge_lifecycle_management_execution_implemented)
        self.assertFalse(plan.lifecycle_stage_transition_implemented)
        self.assertFalse(plan.lifecycle_policy_mutation_implemented)
        self.assertFalse(plan.retention_policy_mutation_implemented)
        self.assertFalse(plan.archival_execution_implemented)
        self.assertFalse(plan.deprecation_execution_implemented)
        self.assertFalse(plan.deletion_execution_implemented)
        self.assertFalse(plan.lifecycle_record_write_implemented)
        self.assertFalse(plan.knowledge_consolidation_execution_implemented)
        self.assertFalse(plan.knowledge_merge_execution_implemented)
        self.assertFalse(plan.knowledge_deduplication_execution_implemented)
        self.assertFalse(plan.canonical_record_write_implemented)
        self.assertFalse(plan.consolidation_record_write_implemented)
        self.assertFalse(plan.kb_storage_write_implemented)
        self.assertFalse(plan.source_record_update_implemented)
        self.assertFalse(plan.retrieval_query_execution_implemented)
        self.assertFalse(plan.retrieval_configuration_mutation_implemented)
        self.assertFalse(plan.ranking_mutation_implemented)
        self.assertFalse(plan.embedding_request_execution_implemented)
        self.assertFalse(plan.embedding_refresh_execution_implemented)
        self.assertFalse(plan.vector_indexing_implemented)
        self.assertFalse(plan.vector_upsert_implemented)
        self.assertFalse(plan.documentation_fetch_execution_implemented)
        self.assertFalse(plan.provider_provisioning_implemented)
        self.assertFalse(plan.api_key_inference_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_signals_score_knowledge_provenance_without_execution(self) -> None:
        plan = build_knowledge_provenance_evolution(route="generate")
        lifecycle_signal_ids = set(plan.knowledge_lifecycle_signal_ids)

        for signal in plan.signals:
            dumped = signal.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_SIGNAL_FIELDS)
            self.assertEqual(
                signal.serialization_version,
                "knowledge_provenance_entry.v1",
            )
            self.assertEqual(signal.route_name, RouteName.GENERATE)
            self.assertEqual(
                signal.signal_id,
                f"knowledge_provenance_evolution::{signal.signal_kind}",
            )
            self.assertEqual(
                signal.knowledge_lifecycle_signal_count,
                len(signal.knowledge_lifecycle_signal_ids),
            )
            self.assertTrue(
                set(signal.knowledge_lifecycle_signal_ids).issubset(
                    lifecycle_signal_ids
                )
            )
            self.assertEqual(
                signal.provenance_score,
                min(
                    1000,
                    max(
                        0,
                        signal.provenance_signal_score * 3
                        + signal.lineage_alignment_score * 3
                        + signal.governance_alignment_score * 2
                        + signal.mutation_risk_score * 2
                        + signal.governance_weight,
                    ),
                ),
            )
            self.assertIn("knowledge_provenance_evolution", signal.context_tags)
            self.assertIn(
                "knowledge_provenance_evolution_execution",
                signal.blocked_runtime_behaviors,
            )
            self.assertIn("provenance_graph_mutation", signal.blocked_runtime_behaviors)
            self.assertIn("kb_storage_write", signal.blocked_runtime_behaviors)
            self.assertTrue(signal.hitl_required_before_provenance_evolution)
            self.assertTrue(
                signal.knowledge_provenance_evolution_capability_implemented
            )
            self.assertTrue(signal.knowledge_provenance_evolution_metadata_implemented)
            self.assertTrue(signal.knowledge_lifecycle_metadata_used)
            self.assertFalse(
                signal.knowledge_provenance_evolution_execution_implemented
            )
            self.assertFalse(signal.provenance_graph_mutation_implemented)
            self.assertFalse(signal.provenance_record_write_implemented)
            self.assertFalse(signal.lineage_reconstruction_execution_implemented)
            self.assertFalse(signal.source_relinking_execution_implemented)
            self.assertFalse(
                signal.knowledge_lifecycle_management_execution_implemented
            )
            self.assertFalse(signal.lifecycle_stage_transition_implemented)
            self.assertFalse(signal.lifecycle_policy_mutation_implemented)
            self.assertFalse(signal.retention_policy_mutation_implemented)
            self.assertFalse(signal.archival_execution_implemented)
            self.assertFalse(signal.deprecation_execution_implemented)
            self.assertFalse(signal.deletion_execution_implemented)
            self.assertFalse(signal.lifecycle_record_write_implemented)
            self.assertFalse(signal.knowledge_consolidation_execution_implemented)
            self.assertFalse(signal.knowledge_merge_execution_implemented)
            self.assertFalse(signal.knowledge_deduplication_execution_implemented)
            self.assertFalse(signal.canonical_record_write_implemented)
            self.assertFalse(signal.consolidation_record_write_implemented)
            self.assertFalse(signal.kb_storage_write_implemented)
            self.assertFalse(signal.source_record_update_implemented)
            self.assertFalse(signal.retrieval_query_execution_implemented)
            self.assertFalse(signal.retrieval_configuration_mutation_implemented)
            self.assertFalse(signal.ranking_mutation_implemented)
            self.assertFalse(signal.embedding_request_execution_implemented)
            self.assertFalse(signal.embedding_refresh_execution_implemented)
            self.assertFalse(signal.vector_indexing_implemented)
            self.assertFalse(signal.vector_upsert_implemented)
            self.assertFalse(signal.documentation_fetch_execution_implemented)
            self.assertFalse(signal.provider_provisioning_implemented)
            self.assertFalse(signal.api_key_inference_implemented)
            self.assertFalse(signal.provider_model_routing_implemented)
            self.assertFalse(signal.provider_execution_implemented)
            self.assertFalse(signal.workflow_control_implemented)
            self.assertFalse(signal.generated_output_mutation_implemented)
            self.assertFalse(signal.runtime_evolution_implemented)
            self.assertTrue(signal.advisory_only)

        inventory = knowledge_provenance_signal_by_id(
            "knowledge_provenance_evolution::knowledge_provenance_inventory_review",
            plan,
        )
        self.assertIsNotNone(inventory)
        assert inventory is not None
        self.assertEqual(inventory.status, "guarded")
        self.assertEqual(inventory.confidence, "guarded")
        self.assertEqual(
            len(knowledge_provenance_signals_for_status("review_required", plan)),
            3,
        )
        self.assertEqual(
            len(knowledge_provenance_signals_for_confidence("high", plan)),
            2,
        )

    def test_plan_rejects_mismatched_knowledge_provenance_metadata(self) -> None:
        plan = build_knowledge_provenance_evolution()
        payload = plan.model_dump(mode="json")
        payload["signal_ids"] = ("missing",) + tuple(payload["signal_ids"][1:])

        with self.assertRaisesRegex(ValueError, "signal_ids must match"):
            KnowledgeProvenanceEvolutionPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_provenance_score"] -= 1

        with self.assertRaisesRegex(ValueError, "overall_provenance_score must match"):
            KnowledgeProvenanceEvolutionPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["planned_provenance_evolution_ids"] = (plan.signal_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "planned_provenance_evolution_ids must remain empty",
        ):
            KnowledgeProvenanceEvolutionPlan(**payload)

    def test_knowledge_provenance_composes_task_14_metadata(self) -> None:
        lifecycle_plan = build_knowledge_lifecycle_management(route=RouteName.REVIEW)
        plan = build_knowledge_provenance_evolution(
            route=RouteName.REVIEW,
            knowledge_lifecycle=lifecycle_plan,
        )

        self.assertEqual(plan.route_name, RouteName.REVIEW)
        self.assertEqual(plan.knowledge_lifecycle_signal_ids, lifecycle_plan.signal_ids)
        self.assertEqual(plan.source_count, lifecycle_plan.source_count)
        self.assertEqual(plan.domain_count, lifecycle_plan.domain_count)
        for signal in plan.signals:
            self.assertTrue(
                set(signal.knowledge_lifecycle_signal_ids).issubset(
                    set(lifecycle_plan.signal_ids)
                )
            )
            self.assertFalse(signal.provenance_graph_mutation_implemented)
            self.assertFalse(signal.provenance_record_write_implemented)
            self.assertFalse(signal.source_relinking_execution_implemented)

    def test_knowledge_provenance_preserves_routing_and_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Review knowledge provenance posture for p5.js examples.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_knowledge_provenance_evolution(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_knowledge_provenance_omits_runtime_mutation_calls(self) -> None:
        plan = build_knowledge_provenance_evolution(route=RouteName.GENERATE)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *plan.covered_roadmap_items,
                *(
                    field
                    for signal in plan.signals
                    for field in (
                        signal.signal_id,
                        signal.signal_kind,
                        signal.status,
                        signal.confidence,
                        signal.provenance_axis,
                        *signal.knowledge_lifecycle_signal_ids,
                        signal.provenance_signal_summary,
                        *signal.context_tags,
                        *signal.explainability_notes,
                        *signal.advisory_actions,
                        *signal.evidence,
                        *signal.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "execute_provenance_evolution(",
            "mutate_provenance_graph(",
            "write_provenance_record(",
            "reconstruct_lineage(",
            "relink_source(",
            "execute_lifecycle_management(",
            "transition_lifecycle_stage(",
            "mutate_lifecycle_policy(",
            "mutate_retention_policy(",
            "archive_knowledge(",
            "deprecate_knowledge(",
            "delete_knowledge(",
            "write_lifecycle_record(",
            "execute_knowledge_consolidation(",
            "merge_knowledge(",
            "deduplicate_knowledge(",
            "write_canonical_record(",
            "write_kb_storage(",
            "update_source_record(",
            "execute_retrieval_query(",
            "mutate_retrieval_config(",
            "mutate_ranking(",
            "request_embedding(",
            "refresh_embedding(",
            "index_vectors(",
            "upsert_vectors(",
            "fetch_documentation(",
            "provision_provider(",
            "infer_api_key(",
            "route_provider(",
            "execute_provider(",
            "invoke_agent(",
            "control_workflow(",
            "mutate_workflow_graph(",
            "execute_workflow(",
            "modify_output(",
            "apply_runtime_evolution(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
