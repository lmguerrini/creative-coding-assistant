import unittest
from pathlib import Path

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    ArchitectureConsistencyPassRegistry,
    architecture_consistency_pass_registry,
    architecture_consistency_record_by_source_registry,
    architecture_consistency_records_for_layer,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_SOURCE_REGISTRY_IDS = (
    "agent_contract_audit_registry",
    "escalation_policy_audit_registry",
    "hybrid_workflow_audit_registry",
    "agent_registry_audit_registry",
    "blackboard_audit_registry",
    "shared_context_audit_registry",
    "agent_collaboration_audit_registry",
    "creative_diversity_audit_registry",
    "agent_explainability_audit_registry",
    "agent_reliability_audit_registry",
    "agent_determinism_audit_registry",
    "agent_telemetry_foundation_registry",
    "agent_cost_tracking_foundation_registry",
    "agent_performance_tracking_foundation_registry",
    "engine_contract_consistency_registry",
)
EXPECTED_ARCHITECTURE_LAYERS = (
    "agent_contract_boundary",
    "hybrid_workflow_boundary",
    "registry_boundary",
    "shared_context_boundary",
    "collaboration_boundary",
    "hardening_quality_boundary",
    "foundation_observability_boundary",
    "engine_contract_boundary",
)
EXPECTED_DOC_REFS = (
    "README.md",
    "docs/PROJECT_CONTEXT.md",
    "docs/IMPLEMENTATION_ROADMAP.md",
    "docs/ARCHITECTURE_DECISIONS.md",
    "architecture/workflow_graph.md",
    "architecture/engine_matrix.md",
)
REQUIRED_RECORD_FIELDS = {
    "source_registry_id",
    "architecture_layer",
    "architecture_stage",
    "source_role",
    "source_serialization_version",
    "source_count_field",
    "source_count",
    "architecture_doc_refs",
    "source_blocked_runtime_behaviors",
    "source_active_runtime_flags",
    "validated_architecture_surfaces",
    "passive_boundary_flags",
    "consistency_findings",
    "missing_coverage_items",
    "source_metadata_only_declared",
    "architecture_consistency_status",
    "architecture_doc_rewrite_implemented",
    "workflow_graph_mutation_implemented",
    "prompt_generation_implemented",
    "provider_model_routing_implemented",
    "runtime_selection_implemented",
    "agent_invocation_implemented",
    "artifact_execution_implemented",
    "retry_triggering_implemented",
    "storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "metadata_only",
}


class ArchitectureConsistencyPassTests(unittest.TestCase):
    def test_registry_covers_hardening_sources_and_doc_refs(self) -> None:
        registry = architecture_consistency_pass_registry()

        self.assertEqual(registry.role, "architecture_consistency_pass_registry")
        self.assertEqual(
            registry.serialization_version,
            "architecture_consistency_pass_registry.v1",
        )
        self.assertEqual(
            registry.architecture_stage,
            "v4_6_architecture_consistency_pass",
        )
        self.assertEqual(registry.source_registry_ids, EXPECTED_SOURCE_REGISTRY_IDS)
        self.assertEqual(registry.architecture_layers, EXPECTED_ARCHITECTURE_LAYERS)
        self.assertEqual(registry.architecture_doc_refs, EXPECTED_DOC_REFS)
        self.assertEqual(registry.record_count, 15)
        self.assertTrue(registry.all_sources_covered)
        self.assertTrue(registry.architecture_docs_referenced)
        self.assertTrue(registry.no_active_runtime_flags)
        self.assertTrue(registry.no_missing_coverage)
        self.assertTrue(registry.metadata_only)
        self.assertIn("workflow_graph_mutation", registry.blocked_runtime_behaviors)
        self.assertIn("provider_or_model_routing", registry.blocked_runtime_behaviors)
        self.assertIn("does not rewrite architecture", registry.authority_boundary)
        self.assertFalse(registry.architecture_doc_rewrite_implemented)
        self.assertFalse(registry.workflow_graph_mutation_implemented)
        self.assertFalse(registry.prompt_generation_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.runtime_selection_implemented)
        self.assertFalse(registry.agent_invocation_implemented)
        self.assertFalse(registry.artifact_execution_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.storage_write_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)

        for doc_ref in registry.architecture_doc_refs:
            self.assertTrue((REPO_ROOT / doc_ref).exists(), doc_ref)

    def test_records_are_passive_and_source_aligned(self) -> None:
        registry = architecture_consistency_pass_registry()

        for record in registry.records:
            dumped = record.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_RECORD_FIELDS)
            self.assertEqual(
                record.serialization_version,
                "architecture_consistency_record.v1",
            )
            self.assertEqual(record.architecture_stage, registry.architecture_stage)
            self.assertEqual(record.source_role, record.source_registry_id)
            self.assertTrue(record.source_serialization_version.endswith(".v1"))
            self.assertIn(
                record.source_count_field,
                ("audit_count", "profile_count", "family_count"),
            )
            self.assertGreaterEqual(record.source_count, 1)
            self.assertEqual(
                record.architecture_doc_refs, registry.architecture_doc_refs
            )
            self.assertEqual(
                record.validated_architecture_surfaces,
                registry.validated_architecture_surfaces,
            )
            self.assertEqual(
                record.passive_boundary_flags, registry.passive_boundary_flags
            )
            self.assertFalse(record.source_active_runtime_flags)
            self.assertFalse(record.missing_coverage_items)
            self.assertTrue(record.source_blocked_runtime_behaviors)
            self.assertTrue(record.source_metadata_only_declared)
            self.assertEqual(record.architecture_consistency_status, "pass")
            self.assertTrue(record.metadata_only)
            self.assertFalse(record.architecture_doc_rewrite_implemented)
            self.assertFalse(record.workflow_graph_mutation_implemented)
            self.assertFalse(record.prompt_generation_implemented)
            self.assertFalse(record.provider_model_routing_implemented)
            self.assertFalse(record.runtime_selection_implemented)
            self.assertFalse(record.agent_invocation_implemented)
            self.assertFalse(record.artifact_execution_implemented)
            self.assertFalse(record.retry_triggering_implemented)
            self.assertFalse(record.storage_write_implemented)
            self.assertFalse(record.generated_output_mutation_implemented)

    def test_lookup_and_layer_filters_are_stable(self) -> None:
        registry = architecture_consistency_pass_registry()
        telemetry_record = architecture_consistency_record_by_source_registry(
            "agent_telemetry_foundation_registry"
        )
        missing_record = architecture_consistency_record_by_source_registry(
            "missing_registry"
        )
        foundation_records = architecture_consistency_records_for_layer(
            "foundation_observability_boundary"
        )
        hybrid_records = architecture_consistency_records_for_layer(
            "hybrid_workflow_boundary"
        )
        missing_layer_records = architecture_consistency_records_for_layer(
            "missing_layer"
        )

        self.assertIsNone(missing_record)
        self.assertIsNotNone(telemetry_record)
        assert telemetry_record is not None
        self.assertEqual(telemetry_record.source_count, 12)
        self.assertEqual(
            telemetry_record.architecture_layer,
            "foundation_observability_boundary",
        )
        self.assertEqual(
            tuple(record.source_registry_id for record in foundation_records),
            (
                "agent_telemetry_foundation_registry",
                "agent_cost_tracking_foundation_registry",
                "agent_performance_tracking_foundation_registry",
            ),
        )
        self.assertEqual(
            tuple(record.source_registry_id for record in hybrid_records),
            (
                "escalation_policy_audit_registry",
                "hybrid_workflow_audit_registry",
                "creative_diversity_audit_registry",
            ),
        )
        self.assertEqual(missing_layer_records, ())
        self.assertIs(
            telemetry_record,
            architecture_consistency_record_by_source_registry(
                "agent_telemetry_foundation_registry",
                registry,
            ),
        )

    def test_registry_rejects_mismatched_or_active_records(self) -> None:
        registry = architecture_consistency_pass_registry()
        first_record = registry.records[0]
        duplicate_record = first_record.model_copy(
            update={"architecture_layer": "registry_boundary"}
        )
        mismatched_docs_record = first_record.model_copy(
            update={
                "architecture_doc_refs": (
                    "README.md",
                    "docs/PROJECT_CONTEXT.md",
                    "docs/IMPLEMENTATION_ROADMAP.md",
                    "docs/ARCHITECTURE_DECISIONS.md",
                    "architecture/workflow_graph.md",
                    "architecture/other.md",
                )
            }
        )
        incomplete_record = first_record.model_copy(
            update={"missing_coverage_items": ("architecture_doc_ref_missing",)}
        )
        active_record = first_record.model_copy(
            update={
                "source_active_runtime_flags": ("provider_model_routing_implemented",)
            }
        )

        with self.assertRaisesRegex(ValueError, "source_registry_ids must be unique"):
            self._registry_with_records(
                (first_record, duplicate_record) + registry.records[2:]
            )

        with self.assertRaisesRegex(ValueError, "architecture_doc_refs"):
            self._registry_with_records(
                (mismatched_docs_record,) + registry.records[1:]
            )

        with self.assertRaisesRegex(ValueError, "missing coverage"):
            self._registry_with_records((incomplete_record,) + registry.records[1:])

        with self.assertRaisesRegex(ValueError, "active runtime flags"):
            self._registry_with_records((active_record,) + registry.records[1:])

    def test_architecture_consistency_pass_does_not_change_request_routing(
        self,
    ) -> None:
        request = AssistantRequest(
            query="Generate architecture consistency metadata for a sketch.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)

        architecture_consistency_pass_registry()
        architecture_consistency_record_by_source_registry(
            "agent_telemetry_foundation_registry"
        )
        architecture_consistency_records_for_layer("hardening_quality_boundary")
        next_decision = route_request(request)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(next_decision.route, RouteName.GENERATE)
        self.assertNotIn(
            "architecture_consistency_pass_registry",
            next_decision.model_dump_json(),
        )

    def test_architecture_consistency_metadata_does_not_declare_active_terms(
        self,
    ) -> None:
        registry = architecture_consistency_pass_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *registry.passive_boundary_flags,
                *(
                    field
                    for record in registry.records
                    for field in (
                        record.source_registry_id,
                        record.architecture_layer,
                        *record.consistency_findings,
                        *record.passive_boundary_flags,
                    )
                ),
            )
        )

        for forbidden_term in (
            "rewrite_architecture_docs_now",
            "mutate_workflow_graph_now",
            "generate_provider_prompt",
            "select_model_for_architecture",
            "execute_architecture_plan",
            "write_runtime_storage",
            "mutate_generated_output",
        ):
            self.assertNotIn(forbidden_term, combined_text)

    def _registry_with_records(
        self,
        records: tuple,
    ) -> ArchitectureConsistencyPassRegistry:
        registry = architecture_consistency_pass_registry()
        return ArchitectureConsistencyPassRegistry(
            records=records,
            source_registry_ids=registry.source_registry_ids,
            record_count=registry.record_count,
            architecture_layers=registry.architecture_layers,
            architecture_doc_refs=registry.architecture_doc_refs,
            validated_architecture_surfaces=registry.validated_architecture_surfaces,
            passive_boundary_flags=registry.passive_boundary_flags,
        )


if __name__ == "__main__":
    unittest.main()
