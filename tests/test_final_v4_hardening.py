import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    ASSISTANT_WORKFLOW_NODE_ORDER,
    FinalV4HardeningRegistry,
    LangGraphErrorPathAuditRegistry,
    architecture_consistency_pass_registry,
    final_v4_hardening_record_by_domain_id,
    final_v4_hardening_records_for_source_registry,
    final_v4_hardening_registry,
    langgraph_error_path_audit_record_by_surface_id,
    langgraph_error_path_audit_records_for_node,
    langgraph_error_path_audit_registry,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_DOMAIN_IDS = (
    "contract_registry_boundary",
    "context_memory_boundary",
    "collaboration_workflow_boundary",
    "quality_reliability_determinism_boundary",
    "observability_foundation_boundary",
    "architecture_closure_boundary",
    "langgraph_error_path_boundary",
)
EXPECTED_SOURCE_REGISTRY_IDS = (
    "architecture_consistency_pass_registry",
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
    "langgraph_error_path_audit",
)
EXPECTED_LANGGRAPH_ERROR_SURFACE_IDS = (
    "runtime_node_failure_paths",
    "provider_errors",
    "stream_errors",
    "planning_sub_helper_failures",
    "prompt_rendering_failures",
    "serialization_failures",
    "workflow_state_consistency",
    "refinement_loop_failures",
    "review_failures",
    "workstation_hydration_failures",
    "preview_preparation_failures",
    "artifact_extraction_failures",
    "artifact_critique_failures",
    "registry_loading_failures",
    "passive_metadata_import_failures",
    "backend_frontend_boundary_failures",
)
REQUIRED_RECORD_FIELDS = {
    "domain_id",
    "domain_name",
    "hardening_stage",
    "source_registry_ids",
    "source_registry_count",
    "architecture_consistency_record_ids",
    "architecture_doc_refs",
    "validated_hardening_surfaces",
    "passive_boundary_flags",
    "hardening_findings",
    "source_active_runtime_flags",
    "missing_coverage_items",
    "source_metadata_only_declared",
    "architecture_consistency_confirmed",
    "final_hardening_status",
    "runtime_hardening_engine_implemented",
    "architecture_doc_mutation_implemented",
    "workflow_graph_mutation_implemented",
    "provider_model_routing_implemented",
    "runtime_selection_implemented",
    "agent_invocation_implemented",
    "workflow_control_implemented",
    "retry_triggering_implemented",
    "storage_write_implemented",
    "artifact_execution_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "metadata_only",
}


class FinalV4HardeningTests(unittest.TestCase):
    def test_registry_covers_final_hardening_domains(self) -> None:
        registry = final_v4_hardening_registry()
        architecture = architecture_consistency_pass_registry()

        self.assertEqual(registry.role, "final_v4_hardening_registry")
        self.assertEqual(
            registry.serialization_version,
            "final_v4_hardening_registry.v1",
        )
        self.assertEqual(registry.hardening_stage, "v4_6_final_v4_hardening")
        self.assertEqual(registry.domain_ids, EXPECTED_DOMAIN_IDS)
        self.assertEqual(registry.record_count, 7)
        self.assertEqual(registry.source_registry_ids, EXPECTED_SOURCE_REGISTRY_IDS)
        self.assertEqual(
            registry.source_registry_ids[1:-1],
            architecture.source_registry_ids,
        )
        self.assertEqual(
            registry.source_langgraph_error_path_audit_registry,
            "langgraph_error_path_audit",
        )
        self.assertEqual(
            registry.langgraph_error_path_surface_ids,
            EXPECTED_LANGGRAPH_ERROR_SURFACE_IDS,
        )
        self.assertEqual(registry.architecture_doc_refs, architecture.architecture_doc_refs)
        self.assertTrue(registry.all_domains_covered)
        self.assertTrue(registry.architecture_consistency_covered)
        self.assertTrue(registry.no_active_runtime_flags)
        self.assertTrue(registry.no_missing_coverage)
        self.assertTrue(registry.metadata_only)
        self.assertIn("runtime_hardening_engine", registry.blocked_runtime_behaviors)
        self.assertIn("provider_or_model_routing", registry.blocked_runtime_behaviors)
        self.assertIn("does not execute hardening", registry.authority_boundary)
        self.assertFalse(registry.runtime_hardening_engine_implemented)
        self.assertFalse(registry.architecture_doc_mutation_implemented)
        self.assertFalse(registry.workflow_graph_mutation_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.runtime_selection_implemented)
        self.assertFalse(registry.agent_invocation_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.storage_write_implemented)
        self.assertFalse(registry.artifact_execution_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)

    def test_records_are_passive_and_architecture_consistency_aligned(self) -> None:
        registry = final_v4_hardening_registry()
        architecture = architecture_consistency_pass_registry()
        known_sources = set(registry.source_registry_ids)
        architecture_sources = set(architecture.source_registry_ids)

        for record in registry.records:
            dumped = record.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_RECORD_FIELDS)
            self.assertEqual(
                record.serialization_version,
                "final_v4_hardening_record.v1",
            )
            self.assertEqual(record.hardening_stage, registry.hardening_stage)
            self.assertEqual(
                record.source_registry_count,
                len(record.source_registry_ids),
            )
            self.assertTrue(set(record.source_registry_ids).issubset(known_sources))
            self.assertTrue(
                set(record.architecture_consistency_record_ids).issubset(
                    architecture_sources
                )
            )
            self.assertEqual(record.architecture_doc_refs, registry.architecture_doc_refs)
            self.assertEqual(
                record.validated_hardening_surfaces,
                registry.validated_hardening_surfaces,
            )
            self.assertEqual(record.passive_boundary_flags, registry.passive_boundary_flags)
            self.assertFalse(record.source_active_runtime_flags)
            self.assertFalse(record.missing_coverage_items)
            self.assertTrue(record.source_metadata_only_declared)
            self.assertTrue(record.architecture_consistency_confirmed)
            self.assertEqual(record.final_hardening_status, "pass")
            self.assertTrue(record.metadata_only)
            self.assertFalse(record.runtime_hardening_engine_implemented)
            self.assertFalse(record.architecture_doc_mutation_implemented)
            self.assertFalse(record.workflow_graph_mutation_implemented)
            self.assertFalse(record.provider_model_routing_implemented)
            self.assertFalse(record.runtime_selection_implemented)
            self.assertFalse(record.agent_invocation_implemented)
            self.assertFalse(record.workflow_control_implemented)
            self.assertFalse(record.retry_triggering_implemented)
            self.assertFalse(record.storage_write_implemented)
            self.assertFalse(record.artifact_execution_implemented)
            self.assertFalse(record.generated_output_mutation_implemented)

    def test_domain_and_source_lookup_helpers_are_stable(self) -> None:
        registry = final_v4_hardening_registry()
        observability_record = final_v4_hardening_record_by_domain_id(
            "observability_foundation_boundary"
        )
        missing_record = final_v4_hardening_record_by_domain_id("missing_domain")
        telemetry_records = final_v4_hardening_records_for_source_registry(
            "agent_telemetry_foundation_registry"
        )
        architecture_records = final_v4_hardening_records_for_source_registry(
            "architecture_consistency_pass_registry"
        )
        error_path_records = final_v4_hardening_records_for_source_registry(
            "langgraph_error_path_audit"
        )
        missing_source_records = final_v4_hardening_records_for_source_registry(
            "missing_registry"
        )

        self.assertIsNone(missing_record)
        self.assertIsNotNone(observability_record)
        assert observability_record is not None
        self.assertEqual(
            observability_record.source_registry_ids,
            (
                "agent_telemetry_foundation_registry",
                "agent_cost_tracking_foundation_registry",
                "agent_performance_tracking_foundation_registry",
            ),
        )
        self.assertEqual(telemetry_records, (observability_record,))
        self.assertEqual(
            tuple(record.domain_id for record in architecture_records),
            ("architecture_closure_boundary",),
        )
        self.assertEqual(
            tuple(record.domain_id for record in error_path_records),
            ("langgraph_error_path_boundary",),
        )
        self.assertEqual(missing_source_records, ())
        self.assertIs(
            observability_record,
            final_v4_hardening_record_by_domain_id(
                "observability_foundation_boundary",
                registry,
            ),
        )

    def test_registry_rejects_mismatched_or_active_records(self) -> None:
        registry = final_v4_hardening_registry()
        first_record = registry.records[0]
        duplicate_record = first_record.model_copy(
            update={"domain_name": "Duplicate Domain"}
        )
        mismatched_sources_record = first_record.model_copy(
            update={"source_registry_ids": ("missing_registry", "agent_registry_audit_registry")}
        )
        incomplete_record = first_record.model_copy(
            update={"missing_coverage_items": ("source_registry_missing",)}
        )
        active_record = first_record.model_copy(
            update={"source_active_runtime_flags": ("workflow_graph_mutation_implemented",)}
        )

        with self.assertRaisesRegex(ValueError, "domain_ids must be unique"):
            self._registry_with_records(
                (first_record, duplicate_record) + registry.records[2:]
            )

        with self.assertRaisesRegex(ValueError, "source_registry_ids must be known"):
            self._registry_with_records(
                (mismatched_sources_record,) + registry.records[1:]
            )

        with self.assertRaisesRegex(ValueError, "missing coverage"):
            self._registry_with_records(
                (incomplete_record,) + registry.records[1:]
            )

        with self.assertRaisesRegex(ValueError, "active runtime flags"):
            self._registry_with_records((active_record,) + registry.records[1:])

    def test_final_v4_hardening_does_not_change_request_routing(self) -> None:
        request = AssistantRequest(
            query="Generate final V4 hardening metadata for a sketch.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)

        final_v4_hardening_registry()
        final_v4_hardening_record_by_domain_id("observability_foundation_boundary")
        final_v4_hardening_records_for_source_registry(
            "agent_telemetry_foundation_registry"
        )
        next_decision = route_request(request)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(next_decision.route, RouteName.GENERATE)
        self.assertNotIn(
            "final_v4_hardening_registry",
            next_decision.model_dump_json(),
        )

    def test_final_hardening_metadata_does_not_declare_active_terms(self) -> None:
        registry = final_v4_hardening_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *registry.passive_boundary_flags,
                *(
                    field
                    for record in registry.records
                    for field in (
                        record.domain_id,
                        *record.hardening_findings,
                        *record.passive_boundary_flags,
                    )
                ),
            )
        )

        for forbidden_term in (
            "run_hardening_engine_now",
            "mutate_architecture_docs_now",
            "mutate_workflow_graph_now",
            "route_provider_now",
            "select_runtime_now",
            "invoke_agent_now",
            "write_storage_now",
            "mutate_generated_output",
        ):
            self.assertNotIn(forbidden_term, combined_text)

    def test_langgraph_error_path_audit_covers_required_surfaces(self) -> None:
        audit = langgraph_error_path_audit_registry()
        runtime_failure_record = langgraph_error_path_audit_record_by_surface_id(
            "runtime_node_failure_paths"
        )
        planning_records = langgraph_error_path_audit_records_for_node("planning")
        failure_records = langgraph_error_path_audit_records_for_node("failure")

        self.assertEqual(audit.role, "langgraph_error_path_audit")
        self.assertEqual(
            audit.serialization_version,
            "langgraph_error_path_audit.v1",
        )
        self.assertEqual(audit.runtime_node_ids, ASSISTANT_WORKFLOW_NODE_ORDER)
        self.assertEqual(
            audit.source_runtime_node_ids,
            ASSISTANT_WORKFLOW_NODE_ORDER[:-1],
        )
        self.assertEqual(audit.terminal_failure_node, "failure")
        self.assertEqual(audit.surface_ids, EXPECTED_LANGGRAPH_ERROR_SURFACE_IDS)
        self.assertEqual(audit.record_count, len(EXPECTED_LANGGRAPH_ERROR_SURFACE_IDS))
        self.assertTrue(audit.all_runtime_nodes_have_failure_edges)
        self.assertTrue(audit.all_required_error_surfaces_covered)
        self.assertTrue(audit.failure_normalization_preserved)
        self.assertTrue(audit.workflow_state_consistency_preserved)
        self.assertTrue(audit.provider_model_routing_preserved)
        self.assertTrue(audit.passive_registries_runtime_blocked)
        self.assertTrue(audit.generated_output_mutation_blocked)
        self.assertFalse(audit.missing_coverage_items)
        self.assertIsNotNone(runtime_failure_record)
        assert runtime_failure_record is not None
        self.assertEqual(
            runtime_failure_record.source_runtime_node_ids,
            ASSISTANT_WORKFLOW_NODE_ORDER[:-1],
        )
        self.assertIn(
            "planning_sub_helper_failures",
            tuple(record.surface_id for record in planning_records),
        )
        self.assertEqual(failure_records, ())

        for record in audit.records:
            self.assertEqual(
                record.serialization_version,
                "langgraph_error_path_audit_record.v1",
            )
            self.assertEqual(record.hardening_stage, audit.hardening_stage)
            self.assertEqual(record.terminal_failure_node, audit.terminal_failure_node)
            self.assertTrue(
                set(record.source_runtime_node_ids).issubset(
                    set(audit.source_runtime_node_ids)
                )
            )
            self.assertEqual(record.failure_invariants, audit.failure_invariants)
            self.assertTrue(record.coverage_refs)
            self.assertFalse(record.missing_coverage_items)
            self.assertTrue(record.terminal_failure_path_confirmed)
            self.assertTrue(record.failure_normalization_confirmed)
            self.assertTrue(record.workflow_state_consistency_confirmed)
            self.assertTrue(record.provider_model_routing_preserved)
            self.assertTrue(record.passive_registry_runtime_block_confirmed)
            self.assertTrue(record.generated_output_mutation_blocked)

    def test_langgraph_error_path_audit_is_passive(self) -> None:
        request = AssistantRequest(
            query="Generate LangGraph error-path audit metadata.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        audit = langgraph_error_path_audit_registry()
        langgraph_error_path_audit_record_by_surface_id("provider_errors")
        langgraph_error_path_audit_records_for_node("generation")
        next_decision = route_request(request)

        self.assertEqual(next_decision, baseline_decision)
        self.assertTrue(audit.metadata_only)
        self.assertFalse(audit.new_langgraph_nodes_implemented)
        self.assertFalse(audit.active_multi_agent_execution_implemented)
        self.assertFalse(audit.provider_model_routing_change_implemented)
        self.assertFalse(audit.workflow_behavior_change_implemented)
        self.assertFalse(audit.passive_registry_runtime_activation_implemented)
        self.assertFalse(audit.generated_output_mutation_implemented)

        combined_text = " ".join(
            (
                audit.authority_boundary,
                *audit.surface_ids,
                *audit.failure_invariants,
                *(
                    field
                    for record in audit.records
                    for field in (
                        record.surface_id,
                        record.coverage_mode,
                        *record.coverage_refs,
                    )
                ),
            )
        )
        for forbidden_term in (
            "add_langgraph_node_now",
            "route_provider_now",
            "select_model_now",
            "activate_passive_registry_now",
            "mutate_generated_output_now",
            "invoke_agent_now",
        ):
            self.assertNotIn(forbidden_term, combined_text)

    def test_langgraph_error_path_audit_rejects_missing_or_active_records(self) -> None:
        audit = langgraph_error_path_audit_registry()
        first_record = audit.records[0]
        missing_record = first_record.model_copy(
            update={"missing_coverage_items": ("uncovered_error_path",)}
        )
        active_record = first_record.model_copy(
            update={"workflow_behavior_change_implemented": True}
        )

        with self.assertRaisesRegex(ValueError, "missing coverage"):
            self._langgraph_audit_with_records(
                (missing_record,) + audit.records[1:]
            )

        with self.assertRaisesRegex(ValueError, "must remain passive"):
            self._langgraph_audit_with_records((active_record,) + audit.records[1:])

    def _registry_with_records(
        self,
        records: tuple,
    ) -> FinalV4HardeningRegistry:
        registry = final_v4_hardening_registry()
        return FinalV4HardeningRegistry(
            records=records,
            domain_ids=registry.domain_ids,
            record_count=registry.record_count,
            source_registry_ids=registry.source_registry_ids,
            architecture_doc_refs=registry.architecture_doc_refs,
            langgraph_error_path_surface_ids=(
                registry.langgraph_error_path_surface_ids
            ),
            validated_hardening_surfaces=registry.validated_hardening_surfaces,
            passive_boundary_flags=registry.passive_boundary_flags,
        )

    def _langgraph_audit_with_records(
        self,
        records: tuple,
    ) -> LangGraphErrorPathAuditRegistry:
        audit = langgraph_error_path_audit_registry()
        return LangGraphErrorPathAuditRegistry(
            runtime_node_ids=audit.runtime_node_ids,
            source_runtime_node_ids=audit.source_runtime_node_ids,
            records=records,
            surface_ids=audit.surface_ids,
            record_count=audit.record_count,
            failure_invariants=audit.failure_invariants,
        )


if __name__ == "__main__":
    unittest.main()
