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
    KnowledgeLifecycleManagementPlan,
    build_knowledge_consolidation,
    build_knowledge_lifecycle_management,
    knowledge_lifecycle_signal_by_id,
    knowledge_lifecycle_signals_for_confidence,
    knowledge_lifecycle_signals_for_status,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
EXPECTED_ROADMAP_ITEMS = ("Knowledge Lifecycle Management",)
REQUIRED_SIGNAL_FIELDS = {
    "signal_id",
    "signal_kind",
    "status",
    "confidence",
    "route_name",
    "task_type",
    "execution_mode_id",
    "lifecycle_axis",
    "knowledge_consolidation_signal_ids",
    "knowledge_consolidation_signal_count",
    "source_count",
    "domain_count",
    "lifecycle_signal_summary",
    "lifecycle_signal_score",
    "policy_alignment_score",
    "governance_alignment_score",
    "mutation_risk_score",
    "governance_weight",
    "lifecycle_score",
    "hitl_required_before_lifecycle_management",
    "context_tags",
    "explainability_notes",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "knowledge_lifecycle_management_capability_implemented",
    "knowledge_lifecycle_management_metadata_implemented",
    "knowledge_consolidation_metadata_used",
    "knowledge_lifecycle_management_execution_implemented",
    "lifecycle_stage_transition_implemented",
    "lifecycle_policy_mutation_implemented",
    "retention_policy_mutation_implemented",
    "archival_execution_implemented",
    "deprecation_execution_implemented",
    "deletion_execution_implemented",
    "lifecycle_record_write_implemented",
    "knowledge_consolidation_execution_implemented",
    "consolidation_candidate_generation_implemented",
    "knowledge_merge_execution_implemented",
    "knowledge_deduplication_execution_implemented",
    "canonical_record_write_implemented",
    "consolidation_record_write_implemented",
    "kb_storage_write_implemented",
    "source_record_update_implemented",
    "source_fetch_execution_implemented",
    "knowledge_drift_detection_execution_implemented",
    "drift_detection_execution_implemented",
    "conflict_resolution_execution_implemented",
    "gap_remediation_execution_implemented",
    "kb_enrichment_implemented",
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


class KnowledgeLifecycleManagementTests(unittest.TestCase):
    def test_plan_builds_knowledge_lifecycle_metadata(self) -> None:
        plan = build_knowledge_lifecycle_management(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "knowledge_lifecycle_management")
        self.assertEqual(plan.serialization_version, "knowledge_lifecycle_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.knowledge_consolidation_role, "knowledge_consolidation")
        self.assertEqual(
            plan.knowledge_consolidation_serialization_version,
            "knowledge_consolidation_plan.v1",
        )
        self.assertEqual(len(plan.knowledge_consolidation_signal_ids), 5)
        self.assertEqual(plan.knowledge_consolidation_signal_count, 5)
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
        self.assertFalse(plan.planned_lifecycle_management_ids)
        self.assertFalse(plan.transitioned_lifecycle_stage_ids)
        self.assertFalse(plan.mutated_lifecycle_policy_ids)
        self.assertFalse(plan.mutated_retention_policy_ids)
        self.assertFalse(plan.archived_knowledge_record_ids)
        self.assertFalse(plan.deprecated_knowledge_record_ids)
        self.assertFalse(plan.deleted_knowledge_record_ids)
        self.assertFalse(plan.written_lifecycle_record_ids)
        self.assertFalse(plan.written_kb_record_ids)
        self.assertEqual(plan.overall_lifecycle_posture, "guarded")
        self.assertIn("does not execute lifecycle management", plan.authority_boundary)
        self.assertIn("write KB storage", plan.authority_boundary)
        self.assertTrue(plan.knowledge_lifecycle_management_capability_implemented)
        self.assertTrue(plan.knowledge_lifecycle_management_metadata_implemented)
        self.assertTrue(plan.roadmap_item_covered)
        self.assertTrue(plan.knowledge_consolidation_metadata_used)
        self.assertFalse(plan.knowledge_lifecycle_management_execution_implemented)
        self.assertFalse(plan.lifecycle_stage_transition_implemented)
        self.assertFalse(plan.lifecycle_policy_mutation_implemented)
        self.assertFalse(plan.retention_policy_mutation_implemented)
        self.assertFalse(plan.archival_execution_implemented)
        self.assertFalse(plan.deprecation_execution_implemented)
        self.assertFalse(plan.deletion_execution_implemented)
        self.assertFalse(plan.lifecycle_record_write_implemented)
        self.assertFalse(plan.knowledge_consolidation_execution_implemented)
        self.assertFalse(plan.consolidation_candidate_generation_implemented)
        self.assertFalse(plan.knowledge_merge_execution_implemented)
        self.assertFalse(plan.knowledge_deduplication_execution_implemented)
        self.assertFalse(plan.canonical_record_write_implemented)
        self.assertFalse(plan.consolidation_record_write_implemented)
        self.assertFalse(plan.kb_storage_write_implemented)
        self.assertFalse(plan.source_record_update_implemented)
        self.assertFalse(plan.source_fetch_execution_implemented)
        self.assertFalse(plan.knowledge_drift_detection_execution_implemented)
        self.assertFalse(plan.drift_detection_execution_implemented)
        self.assertFalse(plan.conflict_resolution_execution_implemented)
        self.assertFalse(plan.gap_remediation_execution_implemented)
        self.assertFalse(plan.kb_enrichment_implemented)
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

    def test_signals_score_knowledge_lifecycle_without_execution(self) -> None:
        plan = build_knowledge_lifecycle_management(route="generate")
        consolidation_signal_ids = set(plan.knowledge_consolidation_signal_ids)

        for signal in plan.signals:
            dumped = signal.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_SIGNAL_FIELDS)
            self.assertEqual(
                signal.serialization_version,
                "knowledge_lifecycle_entry.v1",
            )
            self.assertEqual(signal.route_name, RouteName.GENERATE)
            self.assertEqual(
                signal.signal_id,
                f"knowledge_lifecycle_management::{signal.signal_kind}",
            )
            self.assertEqual(
                signal.knowledge_consolidation_signal_count,
                len(signal.knowledge_consolidation_signal_ids),
            )
            self.assertTrue(
                set(signal.knowledge_consolidation_signal_ids).issubset(
                    consolidation_signal_ids
                )
            )
            self.assertEqual(
                signal.lifecycle_score,
                min(
                    1000,
                    max(
                        0,
                        signal.lifecycle_signal_score * 3
                        + signal.policy_alignment_score * 3
                        + signal.governance_alignment_score * 2
                        + signal.mutation_risk_score * 2
                        + signal.governance_weight,
                    ),
                ),
            )
            self.assertIn("knowledge_lifecycle_management", signal.context_tags)
            self.assertIn(
                "knowledge_lifecycle_management_execution",
                signal.blocked_runtime_behaviors,
            )
            self.assertIn(
                "lifecycle_stage_transition",
                signal.blocked_runtime_behaviors,
            )
            self.assertIn("kb_storage_write", signal.blocked_runtime_behaviors)
            self.assertTrue(signal.hitl_required_before_lifecycle_management)
            self.assertTrue(
                signal.knowledge_lifecycle_management_capability_implemented
            )
            self.assertTrue(signal.knowledge_lifecycle_management_metadata_implemented)
            self.assertTrue(signal.knowledge_consolidation_metadata_used)
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
            self.assertFalse(signal.consolidation_candidate_generation_implemented)
            self.assertFalse(signal.knowledge_merge_execution_implemented)
            self.assertFalse(signal.knowledge_deduplication_execution_implemented)
            self.assertFalse(signal.canonical_record_write_implemented)
            self.assertFalse(signal.consolidation_record_write_implemented)
            self.assertFalse(signal.kb_storage_write_implemented)
            self.assertFalse(signal.source_record_update_implemented)
            self.assertFalse(signal.source_fetch_execution_implemented)
            self.assertFalse(signal.knowledge_drift_detection_execution_implemented)
            self.assertFalse(signal.drift_detection_execution_implemented)
            self.assertFalse(signal.conflict_resolution_execution_implemented)
            self.assertFalse(signal.gap_remediation_execution_implemented)
            self.assertFalse(signal.kb_enrichment_implemented)
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

        inventory = knowledge_lifecycle_signal_by_id(
            "knowledge_lifecycle_management::knowledge_lifecycle_inventory_review",
            plan,
        )
        self.assertIsNotNone(inventory)
        assert inventory is not None
        self.assertEqual(inventory.status, "guarded")
        self.assertEqual(inventory.confidence, "guarded")
        self.assertEqual(
            len(knowledge_lifecycle_signals_for_status("review_required", plan)),
            3,
        )
        self.assertEqual(
            len(knowledge_lifecycle_signals_for_confidence("high", plan)),
            2,
        )

    def test_plan_rejects_mismatched_knowledge_lifecycle_metadata(self) -> None:
        plan = build_knowledge_lifecycle_management()
        payload = plan.model_dump(mode="json")
        payload["signal_ids"] = ("missing",) + tuple(payload["signal_ids"][1:])

        with self.assertRaisesRegex(ValueError, "signal_ids must match"):
            KnowledgeLifecycleManagementPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_lifecycle_score"] -= 1

        with self.assertRaisesRegex(ValueError, "overall_lifecycle_score must match"):
            KnowledgeLifecycleManagementPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["planned_lifecycle_management_ids"] = (plan.signal_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "planned_lifecycle_management_ids must remain empty",
        ):
            KnowledgeLifecycleManagementPlan(**payload)

    def test_knowledge_lifecycle_composes_task_13_metadata(self) -> None:
        consolidation_plan = build_knowledge_consolidation(route=RouteName.REVIEW)
        plan = build_knowledge_lifecycle_management(
            route=RouteName.REVIEW,
            knowledge_consolidation=consolidation_plan,
        )

        self.assertEqual(plan.route_name, RouteName.REVIEW)
        self.assertEqual(
            plan.knowledge_consolidation_signal_ids,
            consolidation_plan.signal_ids,
        )
        self.assertEqual(plan.source_count, consolidation_plan.source_count)
        self.assertEqual(plan.domain_count, consolidation_plan.domain_count)
        for signal in plan.signals:
            self.assertTrue(
                set(signal.knowledge_consolidation_signal_ids).issubset(
                    set(consolidation_plan.signal_ids)
                )
            )
            self.assertFalse(signal.lifecycle_stage_transition_implemented)
            self.assertFalse(signal.lifecycle_policy_mutation_implemented)
            self.assertFalse(signal.lifecycle_record_write_implemented)

    def test_knowledge_lifecycle_preserves_routing_and_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Review knowledge lifecycle posture for p5.js examples.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_knowledge_lifecycle_management(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_knowledge_lifecycle_omits_runtime_mutation_calls(self) -> None:
        plan = build_knowledge_lifecycle_management(route=RouteName.GENERATE)
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
                        signal.lifecycle_axis,
                        *signal.knowledge_consolidation_signal_ids,
                        signal.lifecycle_signal_summary,
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
            "write_consolidation_record(",
            "write_kb_storage(",
            "update_source_record(",
            "fetch_source(",
            "execute_drift_detection(",
            "resolve_conflict(",
            "remediate_gap(",
            "enrich_kb(",
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
