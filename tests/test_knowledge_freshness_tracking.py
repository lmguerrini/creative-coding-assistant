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
    KnowledgeFreshnessTrackingPlan,
    build_knowledge_freshness_tracking,
    build_knowledge_rollback,
    knowledge_freshness_signal_by_id,
    knowledge_freshness_signals_for_confidence,
    knowledge_freshness_signals_for_status,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
EXPECTED_ROADMAP_ITEMS = ("Knowledge Freshness Tracking",)
REQUIRED_SIGNAL_FIELDS = {
    "signal_id",
    "signal_kind",
    "status",
    "confidence",
    "route_name",
    "task_type",
    "execution_mode_id",
    "freshness_axis",
    "knowledge_rollback_signal_ids",
    "knowledge_rollback_signal_count",
    "source_count",
    "domain_count",
    "freshness_signal_summary",
    "freshness_signal_score",
    "rollback_alignment_score",
    "governance_alignment_score",
    "mutation_risk_score",
    "governance_weight",
    "freshness_score",
    "hitl_required_before_freshness_tracking",
    "context_tags",
    "explainability_notes",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "knowledge_freshness_tracking_capability_implemented",
    "knowledge_freshness_metadata_implemented",
    "knowledge_rollback_metadata_used",
    "knowledge_freshness_tracking_execution_implemented",
    "freshness_scan_execution_implemented",
    "freshness_score_computation_implemented",
    "freshness_record_write_implemented",
    "source_timestamp_update_implemented",
    "staleness_state_mutation_implemented",
    "source_fetch_execution_implemented",
    "kb_state_restore_implemented",
    "knowledge_rollback_execution_implemented",
    "rollback_plan_application_implemented",
    "rollback_state_mutation_implemented",
    "rollback_record_write_implemented",
    "snapshot_restore_execution_implemented",
    "knowledge_snapshot_engine_execution_implemented",
    "snapshot_creation_implemented",
    "snapshot_record_write_implemented",
    "snapshot_storage_write_implemented",
    "knowledge_versioning_execution_implemented",
    "version_graph_mutation_implemented",
    "version_record_write_implemented",
    "provenance_graph_mutation_implemented",
    "provenance_record_write_implemented",
    "knowledge_lifecycle_management_execution_implemented",
    "lifecycle_policy_mutation_implemented",
    "retention_policy_mutation_implemented",
    "knowledge_consolidation_execution_implemented",
    "knowledge_merge_execution_implemented",
    "knowledge_deduplication_execution_implemented",
    "canonical_record_write_implemented",
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


class KnowledgeFreshnessTrackingTests(unittest.TestCase):
    def test_plan_builds_knowledge_freshness_metadata(self) -> None:
        plan = build_knowledge_freshness_tracking(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "knowledge_freshness_tracking")
        self.assertEqual(plan.serialization_version, "knowledge_freshness_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.knowledge_rollback_role, "knowledge_rollback")
        self.assertEqual(
            plan.knowledge_rollback_serialization_version,
            "knowledge_rollback_plan.v1",
        )
        self.assertEqual(len(plan.knowledge_rollback_signal_ids), 5)
        self.assertEqual(plan.knowledge_rollback_signal_count, 5)
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
        self.assertFalse(plan.planned_freshness_tracking_ids)
        self.assertFalse(plan.executed_freshness_scan_ids)
        self.assertFalse(plan.computed_freshness_score_ids)
        self.assertFalse(plan.written_freshness_record_ids)
        self.assertFalse(plan.updated_source_timestamp_ids)
        self.assertFalse(plan.mutated_staleness_state_ids)
        self.assertFalse(plan.fetched_source_ids)
        self.assertFalse(plan.written_kb_record_ids)
        self.assertEqual(plan.overall_freshness_posture, "guarded")
        self.assertIn("does not execute freshness tracking", plan.authority_boundary)
        self.assertIn("write KB storage", plan.authority_boundary)
        self.assertTrue(plan.knowledge_freshness_tracking_capability_implemented)
        self.assertTrue(plan.knowledge_freshness_metadata_implemented)
        self.assertTrue(plan.roadmap_item_covered)
        self.assertTrue(plan.knowledge_rollback_metadata_used)
        self.assertFalse(plan.knowledge_freshness_tracking_execution_implemented)
        self.assertFalse(plan.freshness_scan_execution_implemented)
        self.assertFalse(plan.freshness_score_computation_implemented)
        self.assertFalse(plan.freshness_record_write_implemented)
        self.assertFalse(plan.source_timestamp_update_implemented)
        self.assertFalse(plan.staleness_state_mutation_implemented)
        self.assertFalse(plan.source_fetch_execution_implemented)
        self.assertFalse(plan.kb_state_restore_implemented)
        self.assertFalse(plan.knowledge_rollback_execution_implemented)
        self.assertFalse(plan.rollback_plan_application_implemented)
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

    def test_signals_score_knowledge_freshness_without_execution(self) -> None:
        plan = build_knowledge_freshness_tracking(route="generate")
        rollback_signal_ids = set(plan.knowledge_rollback_signal_ids)

        for signal in plan.signals:
            dumped = signal.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_SIGNAL_FIELDS)
            self.assertEqual(
                signal.serialization_version,
                "knowledge_freshness_entry.v1",
            )
            self.assertEqual(signal.route_name, RouteName.GENERATE)
            self.assertEqual(
                signal.signal_id,
                f"knowledge_freshness_tracking::{signal.signal_kind}",
            )
            self.assertEqual(
                signal.knowledge_rollback_signal_count,
                len(signal.knowledge_rollback_signal_ids),
            )
            self.assertTrue(
                set(signal.knowledge_rollback_signal_ids).issubset(
                    rollback_signal_ids
                )
            )
            self.assertEqual(
                signal.freshness_score,
                min(
                    1000,
                    max(
                        0,
                        signal.freshness_signal_score * 3
                        + signal.rollback_alignment_score * 3
                        + signal.governance_alignment_score * 2
                        + signal.mutation_risk_score * 2
                        + signal.governance_weight,
                    ),
                ),
            )
            self.assertIn("knowledge_freshness_tracking", signal.context_tags)
            self.assertIn(
                "knowledge_freshness_tracking_execution",
                signal.blocked_runtime_behaviors,
            )
            self.assertIn("freshness_scan_execution", signal.blocked_runtime_behaviors)
            self.assertIn("source_fetch_execution", signal.blocked_runtime_behaviors)
            self.assertIn("kb_storage_write", signal.blocked_runtime_behaviors)
            self.assertTrue(signal.hitl_required_before_freshness_tracking)
            self.assertTrue(signal.knowledge_freshness_tracking_capability_implemented)
            self.assertTrue(signal.knowledge_freshness_metadata_implemented)
            self.assertTrue(signal.knowledge_rollback_metadata_used)
            self.assertFalse(signal.knowledge_freshness_tracking_execution_implemented)
            self.assertFalse(signal.freshness_scan_execution_implemented)
            self.assertFalse(signal.freshness_score_computation_implemented)
            self.assertFalse(signal.freshness_record_write_implemented)
            self.assertFalse(signal.source_timestamp_update_implemented)
            self.assertFalse(signal.staleness_state_mutation_implemented)
            self.assertFalse(signal.source_fetch_execution_implemented)
            self.assertFalse(signal.kb_state_restore_implemented)
            self.assertFalse(signal.knowledge_rollback_execution_implemented)
            self.assertFalse(signal.rollback_plan_application_implemented)
            self.assertFalse(signal.rollback_state_mutation_implemented)
            self.assertFalse(signal.rollback_record_write_implemented)
            self.assertFalse(signal.snapshot_restore_execution_implemented)
            self.assertFalse(signal.knowledge_snapshot_engine_execution_implemented)
            self.assertFalse(signal.snapshot_creation_implemented)
            self.assertFalse(signal.snapshot_record_write_implemented)
            self.assertFalse(signal.snapshot_storage_write_implemented)
            self.assertFalse(signal.knowledge_versioning_execution_implemented)
            self.assertFalse(signal.version_graph_mutation_implemented)
            self.assertFalse(signal.version_record_write_implemented)
            self.assertFalse(signal.provenance_graph_mutation_implemented)
            self.assertFalse(signal.provenance_record_write_implemented)
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

        inventory = knowledge_freshness_signal_by_id(
            "knowledge_freshness_tracking::knowledge_freshness_inventory_review",
            plan,
        )
        self.assertIsNotNone(inventory)
        assert inventory is not None
        self.assertEqual(inventory.status, "guarded")
        self.assertEqual(inventory.confidence, "guarded")
        self.assertEqual(
            len(knowledge_freshness_signals_for_status("review_required", plan)),
            3,
        )
        self.assertEqual(
            len(knowledge_freshness_signals_for_confidence("high", plan)),
            2,
        )

    def test_plan_rejects_mismatched_knowledge_freshness_metadata(self) -> None:
        plan = build_knowledge_freshness_tracking()
        payload = plan.model_dump(mode="json")
        payload["signal_ids"] = ("missing",) + tuple(payload["signal_ids"][1:])

        with self.assertRaisesRegex(ValueError, "signal_ids must match"):
            KnowledgeFreshnessTrackingPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_freshness_score"] -= 1

        with self.assertRaisesRegex(ValueError, "overall_freshness_score must match"):
            KnowledgeFreshnessTrackingPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["planned_freshness_tracking_ids"] = (plan.signal_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "planned_freshness_tracking_ids must remain empty",
        ):
            KnowledgeFreshnessTrackingPlan(**payload)

    def test_knowledge_freshness_composes_task_18_metadata(self) -> None:
        rollback_plan = build_knowledge_rollback(route=RouteName.REVIEW)
        plan = build_knowledge_freshness_tracking(
            route=RouteName.REVIEW,
            knowledge_rollback=rollback_plan,
        )

        self.assertEqual(plan.route_name, RouteName.REVIEW)
        self.assertEqual(plan.knowledge_rollback_signal_ids, rollback_plan.signal_ids)
        self.assertEqual(plan.source_count, rollback_plan.source_count)
        self.assertEqual(plan.domain_count, rollback_plan.domain_count)
        for signal in plan.signals:
            self.assertTrue(
                set(signal.knowledge_rollback_signal_ids).issubset(
                    set(rollback_plan.signal_ids)
                )
            )
            self.assertFalse(signal.freshness_scan_execution_implemented)
            self.assertFalse(signal.freshness_record_write_implemented)
            self.assertFalse(signal.source_fetch_execution_implemented)

    def test_knowledge_freshness_preserves_routing_and_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Review knowledge freshness posture for p5.js examples.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_knowledge_freshness_tracking(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_knowledge_freshness_omits_runtime_mutation_calls(self) -> None:
        plan = build_knowledge_freshness_tracking(route=RouteName.GENERATE)
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
                        signal.freshness_axis,
                        *signal.knowledge_rollback_signal_ids,
                        signal.freshness_signal_summary,
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
            "execute_freshness_tracking(",
            "execute_freshness_scan(",
            "compute_freshness_score(",
            "write_freshness_record(",
            "update_source_timestamp(",
            "mutate_staleness_state(",
            "fetch_source(",
            "restore_kb_state(",
            "execute_rollback(",
            "apply_rollback_plan(",
            "mutate_rollback_state(",
            "write_rollback_record(",
            "restore_snapshot(",
            "execute_snapshot(",
            "create_snapshot(",
            "write_snapshot_record(",
            "write_snapshot_storage(",
            "execute_knowledge_versioning(",
            "mutate_version_graph(",
            "write_version_record(",
            "mutate_provenance_graph(",
            "write_provenance_record(",
            "execute_lifecycle_management(",
            "mutate_lifecycle_policy(",
            "mutate_retention_policy(",
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
