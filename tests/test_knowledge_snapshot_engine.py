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
    KnowledgeSnapshotEnginePlan,
    build_knowledge_snapshot_engine,
    build_knowledge_versioning,
    knowledge_snapshot_signal_by_id,
    knowledge_snapshot_signals_for_confidence,
    knowledge_snapshot_signals_for_status,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
EXPECTED_ROADMAP_ITEMS = ("Knowledge Snapshot Engine",)
REQUIRED_SIGNAL_FIELDS = {
    "signal_id",
    "signal_kind",
    "status",
    "confidence",
    "route_name",
    "task_type",
    "execution_mode_id",
    "snapshot_axis",
    "knowledge_versioning_signal_ids",
    "knowledge_versioning_signal_count",
    "source_count",
    "domain_count",
    "snapshot_signal_summary",
    "snapshot_signal_score",
    "version_alignment_score",
    "governance_alignment_score",
    "mutation_risk_score",
    "governance_weight",
    "snapshot_score",
    "hitl_required_before_snapshot",
    "context_tags",
    "explainability_notes",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "knowledge_snapshot_engine_capability_implemented",
    "knowledge_snapshot_metadata_implemented",
    "knowledge_versioning_metadata_used",
    "knowledge_snapshot_engine_execution_implemented",
    "snapshot_creation_implemented",
    "snapshot_capture_execution_implemented",
    "snapshot_record_write_implemented",
    "snapshot_storage_write_implemented",
    "snapshot_index_write_implemented",
    "snapshot_manifest_write_implemented",
    "snapshot_retention_mutation_implemented",
    "knowledge_versioning_execution_implemented",
    "version_graph_mutation_implemented",
    "version_record_write_implemented",
    "version_id_assignment_implemented",
    "version_lineage_reconstruction_implemented",
    "version_history_write_implemented",
    "rollback_execution_implemented",
    "rollback_plan_application_implemented",
    "provenance_graph_mutation_implemented",
    "provenance_record_write_implemented",
    "lineage_reconstruction_execution_implemented",
    "source_relinking_execution_implemented",
    "knowledge_lifecycle_management_execution_implemented",
    "lifecycle_stage_transition_implemented",
    "lifecycle_policy_mutation_implemented",
    "retention_policy_mutation_implemented",
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


class KnowledgeSnapshotEngineTests(unittest.TestCase):
    def test_plan_builds_knowledge_snapshot_metadata(self) -> None:
        plan = build_knowledge_snapshot_engine(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "knowledge_snapshot_engine")
        self.assertEqual(plan.serialization_version, "knowledge_snapshot_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.knowledge_versioning_role, "knowledge_versioning")
        self.assertEqual(
            plan.knowledge_versioning_serialization_version,
            "knowledge_versioning_plan.v1",
        )
        self.assertEqual(len(plan.knowledge_versioning_signal_ids), 5)
        self.assertEqual(plan.knowledge_versioning_signal_count, 5)
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
        self.assertFalse(plan.planned_snapshot_ids)
        self.assertFalse(plan.created_snapshot_ids)
        self.assertFalse(plan.captured_snapshot_ids)
        self.assertFalse(plan.written_snapshot_record_ids)
        self.assertFalse(plan.written_snapshot_storage_ids)
        self.assertFalse(plan.written_snapshot_index_ids)
        self.assertFalse(plan.written_snapshot_manifest_ids)
        self.assertFalse(plan.mutated_snapshot_retention_ids)
        self.assertFalse(plan.written_kb_record_ids)
        self.assertEqual(plan.overall_snapshot_posture, "guarded")
        self.assertIn("does not execute snapshot operations", plan.authority_boundary)
        self.assertIn("write KB storage", plan.authority_boundary)
        self.assertTrue(plan.knowledge_snapshot_engine_capability_implemented)
        self.assertTrue(plan.knowledge_snapshot_metadata_implemented)
        self.assertTrue(plan.roadmap_item_covered)
        self.assertTrue(plan.knowledge_versioning_metadata_used)
        self.assertFalse(plan.knowledge_snapshot_engine_execution_implemented)
        self.assertFalse(plan.snapshot_creation_implemented)
        self.assertFalse(plan.snapshot_capture_execution_implemented)
        self.assertFalse(plan.snapshot_record_write_implemented)
        self.assertFalse(plan.snapshot_storage_write_implemented)
        self.assertFalse(plan.snapshot_index_write_implemented)
        self.assertFalse(plan.snapshot_manifest_write_implemented)
        self.assertFalse(plan.snapshot_retention_mutation_implemented)
        self.assertFalse(plan.knowledge_versioning_execution_implemented)
        self.assertFalse(plan.version_graph_mutation_implemented)
        self.assertFalse(plan.version_record_write_implemented)
        self.assertFalse(plan.rollback_execution_implemented)
        self.assertFalse(plan.provenance_graph_mutation_implemented)
        self.assertFalse(plan.kb_storage_write_implemented)
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

    def test_signals_score_knowledge_snapshot_without_execution(self) -> None:
        plan = build_knowledge_snapshot_engine(route="generate")
        versioning_signal_ids = set(plan.knowledge_versioning_signal_ids)

        for signal in plan.signals:
            dumped = signal.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_SIGNAL_FIELDS)
            self.assertEqual(
                signal.serialization_version,
                "knowledge_snapshot_entry.v1",
            )
            self.assertEqual(signal.route_name, RouteName.GENERATE)
            self.assertEqual(
                signal.signal_id,
                f"knowledge_snapshot_engine::{signal.signal_kind}",
            )
            self.assertEqual(
                signal.knowledge_versioning_signal_count,
                len(signal.knowledge_versioning_signal_ids),
            )
            self.assertTrue(
                set(signal.knowledge_versioning_signal_ids).issubset(
                    versioning_signal_ids
                )
            )
            self.assertEqual(
                signal.snapshot_score,
                min(
                    1000,
                    max(
                        0,
                        signal.snapshot_signal_score * 3
                        + signal.version_alignment_score * 3
                        + signal.governance_alignment_score * 2
                        + signal.mutation_risk_score * 2
                        + signal.governance_weight,
                    ),
                ),
            )
            self.assertIn("knowledge_snapshot_engine", signal.context_tags)
            self.assertIn(
                "knowledge_snapshot_engine_execution",
                signal.blocked_runtime_behaviors,
            )
            self.assertIn("snapshot_creation", signal.blocked_runtime_behaviors)
            self.assertIn("snapshot_storage_write", signal.blocked_runtime_behaviors)
            self.assertIn("kb_storage_write", signal.blocked_runtime_behaviors)
            self.assertTrue(signal.hitl_required_before_snapshot)
            self.assertTrue(signal.knowledge_snapshot_engine_capability_implemented)
            self.assertTrue(signal.knowledge_snapshot_metadata_implemented)
            self.assertTrue(signal.knowledge_versioning_metadata_used)
            self.assertFalse(signal.knowledge_snapshot_engine_execution_implemented)
            self.assertFalse(signal.snapshot_creation_implemented)
            self.assertFalse(signal.snapshot_capture_execution_implemented)
            self.assertFalse(signal.snapshot_record_write_implemented)
            self.assertFalse(signal.snapshot_storage_write_implemented)
            self.assertFalse(signal.snapshot_index_write_implemented)
            self.assertFalse(signal.snapshot_manifest_write_implemented)
            self.assertFalse(signal.snapshot_retention_mutation_implemented)
            self.assertFalse(signal.knowledge_versioning_execution_implemented)
            self.assertFalse(signal.version_graph_mutation_implemented)
            self.assertFalse(signal.version_record_write_implemented)
            self.assertFalse(signal.version_id_assignment_implemented)
            self.assertFalse(signal.version_lineage_reconstruction_implemented)
            self.assertFalse(signal.version_history_write_implemented)
            self.assertFalse(signal.rollback_execution_implemented)
            self.assertFalse(signal.rollback_plan_application_implemented)
            self.assertFalse(signal.provenance_graph_mutation_implemented)
            self.assertFalse(signal.provenance_record_write_implemented)
            self.assertFalse(signal.source_relinking_execution_implemented)
            self.assertFalse(
                signal.knowledge_lifecycle_management_execution_implemented
            )
            self.assertFalse(signal.lifecycle_policy_mutation_implemented)
            self.assertFalse(signal.retention_policy_mutation_implemented)
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

        inventory = knowledge_snapshot_signal_by_id(
            "knowledge_snapshot_engine::knowledge_snapshot_inventory_review",
            plan,
        )
        self.assertIsNotNone(inventory)
        assert inventory is not None
        self.assertEqual(inventory.status, "guarded")
        self.assertEqual(inventory.confidence, "guarded")
        self.assertEqual(
            len(knowledge_snapshot_signals_for_status("review_required", plan)),
            3,
        )
        self.assertEqual(
            len(knowledge_snapshot_signals_for_confidence("high", plan)),
            2,
        )

    def test_plan_rejects_mismatched_knowledge_snapshot_metadata(self) -> None:
        plan = build_knowledge_snapshot_engine()
        payload = plan.model_dump(mode="json")
        payload["signal_ids"] = ("missing",) + tuple(payload["signal_ids"][1:])

        with self.assertRaisesRegex(ValueError, "signal_ids must match"):
            KnowledgeSnapshotEnginePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_snapshot_score"] -= 1

        with self.assertRaisesRegex(ValueError, "overall_snapshot_score must match"):
            KnowledgeSnapshotEnginePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["planned_snapshot_ids"] = (plan.signal_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "planned_snapshot_ids must remain empty",
        ):
            KnowledgeSnapshotEnginePlan(**payload)

    def test_knowledge_snapshot_composes_task_16_metadata(self) -> None:
        versioning_plan = build_knowledge_versioning(route=RouteName.REVIEW)
        plan = build_knowledge_snapshot_engine(
            route=RouteName.REVIEW,
            knowledge_versioning=versioning_plan,
        )

        self.assertEqual(plan.route_name, RouteName.REVIEW)
        self.assertEqual(
            plan.knowledge_versioning_signal_ids,
            versioning_plan.signal_ids,
        )
        self.assertEqual(plan.source_count, versioning_plan.source_count)
        self.assertEqual(plan.domain_count, versioning_plan.domain_count)
        for signal in plan.signals:
            self.assertTrue(
                set(signal.knowledge_versioning_signal_ids).issubset(
                    set(versioning_plan.signal_ids)
                )
            )
            self.assertFalse(signal.snapshot_creation_implemented)
            self.assertFalse(signal.snapshot_record_write_implemented)
            self.assertFalse(signal.snapshot_storage_write_implemented)

    def test_knowledge_snapshot_preserves_routing_and_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Review knowledge snapshot posture for p5.js examples.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_knowledge_snapshot_engine(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_knowledge_snapshot_omits_runtime_mutation_calls(self) -> None:
        plan = build_knowledge_snapshot_engine(route=RouteName.GENERATE)
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
                        signal.snapshot_axis,
                        *signal.knowledge_versioning_signal_ids,
                        signal.snapshot_signal_summary,
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
            "execute_snapshot(",
            "create_snapshot(",
            "capture_snapshot(",
            "write_snapshot_record(",
            "write_snapshot_storage(",
            "write_snapshot_index(",
            "write_snapshot_manifest(",
            "mutate_snapshot_retention(",
            "execute_knowledge_versioning(",
            "mutate_version_graph(",
            "write_version_record(",
            "assign_version_id(",
            "reconstruct_version_lineage(",
            "write_version_history(",
            "execute_rollback(",
            "apply_rollback_plan(",
            "mutate_provenance_graph(",
            "write_provenance_record(",
            "reconstruct_lineage(",
            "relink_source(",
            "execute_lifecycle_management(",
            "transition_lifecycle_stage(",
            "mutate_lifecycle_policy(",
            "mutate_retention_policy(",
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
