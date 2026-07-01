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
    CreativeLineagePlan,
    build_creative_lineage,
    creative_lineage_record_by_id,
    creative_lineage_records_for_confidence,
    creative_lineage_records_for_status,
    multimodal_artifact_lineage_registry,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
REQUIRED_RECORD_FIELDS = {
    "creative_lineage_id",
    "lineage_kind",
    "status",
    "confidence",
    "route_name",
    "task_type",
    "execution_mode_id",
    "lineage_axis",
    "source_artifact_history_record_id",
    "source_creative_dna_signature_id",
    "source_long_term_memory_record_id",
    "source_artifact_lineage_profile_id",
    "lineage_summary",
    "continuity_score",
    "dependency_visibility_score",
    "provenance_trace_score",
    "governance_risk_score",
    "governance_weight",
    "creative_lineage_score",
    "hitl_required_before_lineage_persistence",
    "context_tags",
    "explainability_notes",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "creative_lineage_implemented",
    "creative_lineage_metadata_implemented",
    "artifact_history_source_used",
    "creative_dna_source_used",
    "long_term_memory_source_used",
    "artifact_lineage_registry_source_used",
    "creative_lineage_storage_write_implemented",
    "creative_lineage_record_creation_implemented",
    "creative_lineage_record_update_implemented",
    "creative_lineage_record_deletion_implemented",
    "lineage_inference_implemented",
    "lineage_persistence_implemented",
    "timeline_reconstruction_implemented",
    "dependency_graph_materialization_implemented",
    "provenance_recording_implemented",
    "artifact_mutation_implemented",
    "session_recording_implemented",
    "session_replay_execution_implemented",
    "memory_retrieval_execution_implemented",
    "memory_storage_write_implemented",
    "memory_consolidation_implemented",
    "creative_dna_application_implemented",
    "personalization_application_implemented",
    "provider_model_routing_implemented",
    "automatic_provider_switching_implemented",
    "automatic_model_switching_implemented",
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


class CreativeLineageTests(unittest.TestCase):
    def test_plan_builds_creative_lineage_metadata(self) -> None:
        artifact_lineage = multimodal_artifact_lineage_registry()
        plan = build_creative_lineage(
            route=RouteName.GENERATE,
            artifact_lineage=artifact_lineage,
        )

        self.assertEqual(plan.role, "creative_lineage")
        self.assertEqual(plan.serialization_version, "creative_lineage_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.source_artifact_history_serialization_version,
            "artifact_history_plan.v1",
        )
        self.assertEqual(
            plan.source_creative_dna_serialization_version,
            "creative_dna_plan.v1",
        )
        self.assertEqual(
            plan.source_long_term_memory_serialization_version,
            "long_term_creative_memory_plan.v1",
        )
        self.assertEqual(
            plan.source_artifact_lineage_serialization_version,
            "multimodal_artifact_lineage_registry.v1",
        )
        self.assertEqual(
            plan.source_artifact_lineage_profile_ids,
            artifact_lineage.profile_ids,
        )
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.record_count, 5)
        self.assertEqual(plan.candidate_record_count, 1)
        self.assertEqual(plan.review_required_record_count, 2)
        self.assertEqual(plan.guarded_record_count, 2)
        self.assertEqual(plan.high_confidence_record_count, 3)
        self.assertEqual(plan.hitl_required_record_count, 5)
        self.assertFalse(plan.persisted_creative_lineage_ids)
        self.assertFalse(plan.inferred_creative_lineage_ids)
        self.assertFalse(plan.reconstructed_lineage_ids)
        self.assertFalse(plan.materialized_dependency_graph_ids)
        self.assertFalse(plan.mutated_artifact_ids)
        self.assertEqual(plan.overall_creative_lineage_posture, "guarded")
        self.assertIn(
            "does not write creative lineage storage",
            plan.authority_boundary,
        )
        self.assertIn("infer lineage", plan.authority_boundary)
        self.assertTrue(plan.creative_lineage_implemented)
        self.assertTrue(plan.creative_lineage_metadata_implemented)
        self.assertTrue(plan.artifact_history_source_used)
        self.assertTrue(plan.creative_dna_source_used)
        self.assertTrue(plan.long_term_memory_source_used)
        self.assertTrue(plan.artifact_lineage_registry_source_used)
        self.assertFalse(plan.creative_lineage_storage_write_implemented)
        self.assertFalse(plan.creative_lineage_record_creation_implemented)
        self.assertFalse(plan.creative_lineage_record_update_implemented)
        self.assertFalse(plan.creative_lineage_record_deletion_implemented)
        self.assertFalse(plan.lineage_inference_implemented)
        self.assertFalse(plan.lineage_persistence_implemented)
        self.assertFalse(plan.timeline_reconstruction_implemented)
        self.assertFalse(plan.dependency_graph_materialization_implemented)
        self.assertFalse(plan.provenance_recording_implemented)
        self.assertFalse(plan.artifact_mutation_implemented)
        self.assertFalse(plan.session_recording_implemented)
        self.assertFalse(plan.session_replay_execution_implemented)
        self.assertFalse(plan.memory_retrieval_execution_implemented)
        self.assertFalse(plan.memory_storage_write_implemented)
        self.assertFalse(plan.memory_consolidation_implemented)
        self.assertFalse(plan.creative_dna_application_implemented)
        self.assertFalse(plan.personalization_application_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_records_score_creative_lineage_without_inference(self) -> None:
        plan = build_creative_lineage(route="generate")

        for record in plan.records:
            dumped = record.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_RECORD_FIELDS)
            self.assertEqual(
                record.serialization_version,
                "creative_lineage_record.v1",
            )
            self.assertEqual(record.route_name, RouteName.GENERATE)
            self.assertEqual(
                record.creative_lineage_id,
                f"creative_lineage::{record.lineage_kind}",
            )
            self.assertEqual(
                record.creative_lineage_score,
                min(
                    1000,
                    max(
                        0,
                        record.continuity_score * 3
                        + record.dependency_visibility_score * 2
                        + record.provenance_trace_score * 3
                        + record.governance_risk_score * 2
                        + record.governance_weight,
                    ),
                ),
            )
            self.assertIn("creative_lineage", record.context_tags)
            self.assertIn(
                "lineage_inference",
                record.blocked_runtime_behaviors,
            )
            self.assertIn(
                "dependency_graph_materialization",
                record.blocked_runtime_behaviors,
            )
            self.assertTrue(record.explainability_notes)
            self.assertTrue(record.advisory_actions)
            self.assertTrue(record.evidence)
            self.assertTrue(record.hitl_required_before_lineage_persistence)
            self.assertTrue(record.creative_lineage_implemented)
            self.assertTrue(record.creative_lineage_metadata_implemented)
            self.assertTrue(record.artifact_history_source_used)
            self.assertTrue(record.creative_dna_source_used)
            self.assertTrue(record.long_term_memory_source_used)
            self.assertTrue(record.artifact_lineage_registry_source_used)
            self.assertFalse(record.creative_lineage_storage_write_implemented)
            self.assertFalse(record.creative_lineage_record_creation_implemented)
            self.assertFalse(record.creative_lineage_record_update_implemented)
            self.assertFalse(record.creative_lineage_record_deletion_implemented)
            self.assertFalse(record.lineage_inference_implemented)
            self.assertFalse(record.lineage_persistence_implemented)
            self.assertFalse(record.timeline_reconstruction_implemented)
            self.assertFalse(record.dependency_graph_materialization_implemented)
            self.assertFalse(record.provenance_recording_implemented)
            self.assertFalse(record.artifact_mutation_implemented)
            self.assertFalse(record.session_recording_implemented)
            self.assertFalse(record.session_replay_execution_implemented)
            self.assertFalse(record.memory_retrieval_execution_implemented)
            self.assertFalse(record.memory_storage_write_implemented)
            self.assertFalse(record.memory_consolidation_implemented)
            self.assertFalse(record.creative_dna_application_implemented)
            self.assertFalse(record.personalization_application_implemented)
            self.assertFalse(record.provider_model_routing_implemented)
            self.assertFalse(record.provider_execution_implemented)
            self.assertFalse(record.agent_invocation_implemented)
            self.assertFalse(record.workflow_control_implemented)
            self.assertFalse(record.workflow_graph_mutation_implemented)
            self.assertFalse(record.workflow_execution_implemented)
            self.assertFalse(record.persistent_storage_write_implemented)
            self.assertFalse(record.generated_output_mutation_implemented)
            self.assertFalse(record.runtime_evolution_implemented)
            self.assertTrue(record.advisory_only)

        dependency = creative_lineage_record_by_id(
            "creative_lineage::artifact_dependency_lineage",
            plan,
        )
        self.assertIsNotNone(dependency)
        assert dependency is not None
        self.assertEqual(dependency.status, "guarded")
        self.assertEqual(dependency.confidence, "guarded")
        self.assertEqual(len(creative_lineage_records_for_status("guarded", plan)), 2)
        self.assertEqual(
            len(creative_lineage_records_for_confidence("high", plan)),
            1,
        )

    def test_plan_rejects_mismatched_creative_lineage_metadata(self) -> None:
        plan = build_creative_lineage()
        payload = plan.model_dump(mode="json")
        payload["record_ids"] = ("missing",) + tuple(payload["record_ids"][1:])

        with self.assertRaisesRegex(ValueError, "record_ids must match"):
            CreativeLineagePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_creative_lineage_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_creative_lineage_score must match",
        ):
            CreativeLineagePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["inferred_creative_lineage_ids"] = (plan.record_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "inferred_creative_lineage_ids must remain empty",
        ):
            CreativeLineagePlan(**payload)

    def test_creative_lineage_does_not_change_routing_or_provider_factory(
        self,
    ) -> None:
        request = AssistantRequest(
            query="Review creative lineage for a creative coding project.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_creative_lineage(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_creative_lineage_does_not_declare_runtime_mutation_terms(self) -> None:
        plan = build_creative_lineage(route=RouteName.GENERATE)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for record in plan.records
                    for field in (
                        record.creative_lineage_id,
                        record.lineage_kind,
                        record.status,
                        record.confidence,
                        record.lineage_axis,
                        record.source_artifact_history_record_id,
                        record.source_creative_dna_signature_id,
                        record.source_long_term_memory_record_id,
                        record.source_artifact_lineage_profile_id,
                        record.lineage_summary,
                        *record.context_tags,
                        *record.explainability_notes,
                        *record.advisory_actions,
                        *record.evidence,
                        *record.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "write_creative_lineage(",
            "create_creative_lineage(",
            "update_creative_lineage(",
            "delete_creative_lineage(",
            "infer_lineage(",
            "persist_lineage(",
            "reconstruct_timeline(",
            "materialize_dependency_graph(",
            "record_provenance(",
            "mutate_artifact(",
            "record_session(",
            "execute_session_replay(",
            "retrieve_memory(",
            "write_memory(",
            "consolidate_memory(",
            "apply_creative_dna(",
            "apply_personalization(",
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
