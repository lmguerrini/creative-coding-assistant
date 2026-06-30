import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    ProductionArchitectureFreeze,
    build_production_architecture_freeze,
    production_architecture_freeze_record_by_domain,
    production_architecture_freeze_records_for_status,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_SOURCE_SURFACES = (
    "production_release_final_optimization",
    "production_release_packaging",
    "production_release_candidate",
    "production_demo_assets",
    "production_deployment",
    "production_readiness_review",
    "production_creative_readiness_review",
)
REQUIRED_DOC_REFS = (
    "README.md",
    "docs/PROJECT_CONTEXT.md",
    "docs/IMPLEMENTATION_ROADMAP.md",
    "docs/ARCHITECTURE_DECISIONS.md",
)
REQUIRED_DOMAINS = (
    "runtime_topology_freeze",
    "v5_6_metadata_surface_freeze",
    "provider_model_routing_freeze",
    "deployment_release_operations_freeze",
    "generated_output_boundary_freeze",
    "runtime_evolution_gate_freeze",
)
REQUIRED_RECORD_FIELDS = {
    "record_id",
    "freeze_domain",
    "freeze_status",
    "source_surface_ids",
    "source_serialization_versions",
    "architecture_doc_refs",
    "freeze_assertions",
    "guarded_assumptions",
    "prohibited_changes",
    "evidence",
    "architecture_change_required",
    "blocked_runtime_behaviors",
    "architecture_freeze_record_implemented",
    "core_architecture_expansion_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_execution_implemented",
    "workflow_control_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "runtime_installation_implemented",
    "package_build_executed",
    "deployment_execution_implemented",
    "asset_generation_implemented",
    "retrieval_execution_implemented",
    "generated_output_mutation_implemented",
    "persistent_storage_write_implemented",
    "release_artifact_creation_implemented",
    "hitl_request_emitted",
    "merge_push_tag_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "metadata_only",
}


class ProductionArchitectureFreezeTests(unittest.TestCase):
    def test_freeze_declares_v5_6_architecture_frozen(self) -> None:
        freeze = build_production_architecture_freeze()

        self.assertEqual(freeze.role, "production_architecture_freeze")
        self.assertEqual(
            freeze.serialization_version,
            "production_architecture_freeze.v1",
        )
        self.assertEqual(freeze.version, "V5.6.0")
        self.assertEqual(freeze.target_branch, "feature/production-release")
        self.assertEqual(freeze.target_tag, "v5.6.0")
        self.assertEqual(freeze.source_surface_ids, REQUIRED_SOURCE_SURFACES)
        self.assertEqual(freeze.architecture_doc_refs, REQUIRED_DOC_REFS)
        self.assertEqual(freeze.missing_architecture_doc_refs, ())
        self.assertEqual(freeze.freeze_domains, REQUIRED_DOMAINS)
        self.assertEqual(freeze.record_count, 6)
        self.assertEqual(freeze.architecture_freeze_status, "frozen")
        self.assertGreaterEqual(freeze.guarded_assumption_count, 1)
        self.assertGreater(freeze.prohibited_change_count, 0)
        self.assertTrue(freeze.release_audit_can_proceed)
        self.assertTrue(freeze.runtime_evolution_hitl_required)
        self.assertTrue(freeze.no_architecture_expansion_required)
        self.assertIn("does not introduce new core architecture", freeze.authority_boundary)
        self.assertTrue(freeze.architecture_freeze_implemented)
        self.assertTrue(freeze.source_surfaces_frozen)
        self.assertTrue(freeze.architecture_docs_referenced)
        self.assertTrue(freeze.guarded_assumptions_documented)
        self.assertFalse(freeze.core_architecture_expansion_implemented)
        self.assertFalse(freeze.workflow_graph_mutation_implemented)
        self.assertFalse(freeze.workflow_execution_implemented)
        self.assertFalse(freeze.workflow_control_implemented)
        self.assertFalse(freeze.provider_model_routing_implemented)
        self.assertFalse(freeze.provider_execution_implemented)
        self.assertFalse(freeze.runtime_installation_implemented)
        self.assertFalse(freeze.package_build_executed)
        self.assertFalse(freeze.deployment_execution_implemented)
        self.assertFalse(freeze.asset_generation_implemented)
        self.assertFalse(freeze.retrieval_execution_implemented)
        self.assertFalse(freeze.generated_output_mutation_implemented)
        self.assertFalse(freeze.persistent_storage_write_implemented)
        self.assertFalse(freeze.release_artifact_creation_implemented)
        self.assertFalse(freeze.hitl_request_emitted)
        self.assertFalse(freeze.merge_push_tag_implemented)
        self.assertFalse(freeze.runtime_evolution_implemented)
        self.assertTrue(freeze.metadata_only)

    def test_records_freeze_boundaries_without_runtime_changes(self) -> None:
        freeze = build_production_architecture_freeze()
        routing = production_architecture_freeze_record_by_domain(
            "provider_model_routing_freeze",
            freeze,
        )
        release_ops = production_architecture_freeze_record_by_domain(
            "deployment_release_operations_freeze",
            freeze,
        )
        output = production_architecture_freeze_record_by_domain(
            "generated_output_boundary_freeze",
            freeze,
        )
        runtime_evolution = production_architecture_freeze_record_by_domain(
            "runtime_evolution_gate_freeze",
            freeze,
        )
        frozen = production_architecture_freeze_records_for_status("frozen", freeze)

        self.assertIsNotNone(routing)
        self.assertIsNotNone(release_ops)
        self.assertIsNotNone(output)
        self.assertIsNotNone(runtime_evolution)
        assert routing is not None
        assert release_ops is not None
        assert output is not None
        assert runtime_evolution is not None
        self.assertEqual(len(frozen), 6)
        self.assertIn("missing_api_key", routing.guarded_assumptions)
        self.assertIn(
            "provider_or_model_routing_mutation",
            routing.prohibited_changes,
        )
        self.assertIn("merge_push_tag_operation", release_ops.prohibited_changes)
        self.assertIn("generated_output_modification", output.prohibited_changes)
        self.assertIn(
            "runtime_evolution_requires_hitl_gate",
            runtime_evolution.freeze_assertions,
        )

        for record in freeze.records:
            dumped = record.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_RECORD_FIELDS)
            self.assertEqual(
                record.serialization_version,
                "production_architecture_freeze_record.v1",
            )
            self.assertEqual(
                record.record_id,
                f"production_architecture_freeze::{record.freeze_domain}",
            )
            self.assertEqual(
                len(record.source_surface_ids),
                len(record.source_serialization_versions),
            )
            self.assertEqual(record.architecture_doc_refs, REQUIRED_DOC_REFS)
            self.assertEqual(record.freeze_status, "frozen")
            self.assertFalse(record.architecture_change_required)
            self.assertTrue(record.architecture_freeze_record_implemented)
            self.assertFalse(record.core_architecture_expansion_implemented)
            self.assertFalse(record.workflow_graph_mutation_implemented)
            self.assertFalse(record.workflow_execution_implemented)
            self.assertFalse(record.workflow_control_implemented)
            self.assertFalse(record.provider_model_routing_implemented)
            self.assertFalse(record.provider_execution_implemented)
            self.assertFalse(record.runtime_installation_implemented)
            self.assertFalse(record.package_build_executed)
            self.assertFalse(record.deployment_execution_implemented)
            self.assertFalse(record.asset_generation_implemented)
            self.assertFalse(record.retrieval_execution_implemented)
            self.assertFalse(record.generated_output_mutation_implemented)
            self.assertFalse(record.persistent_storage_write_implemented)
            self.assertFalse(record.release_artifact_creation_implemented)
            self.assertFalse(record.hitl_request_emitted)
            self.assertFalse(record.merge_push_tag_implemented)
            self.assertFalse(record.runtime_evolution_implemented)
            self.assertTrue(record.metadata_only)

    def test_freeze_rejects_mismatched_records_or_counts(self) -> None:
        freeze = build_production_architecture_freeze()
        payload = freeze.model_dump(mode="json")
        payload["record_ids"] = ("missing",) + tuple(payload["record_ids"][1:])

        with self.assertRaisesRegex(ValueError, "record_ids must match"):
            ProductionArchitectureFreeze(**payload)

        payload = freeze.model_dump(mode="json")
        payload["guarded_assumption_count"] += 1

        with self.assertRaisesRegex(ValueError, "guarded_assumption_count must match"):
            ProductionArchitectureFreeze(**payload)

        payload = freeze.model_dump(mode="json")
        payload["architecture_freeze_status"] = "blocked"

        with self.assertRaisesRegex(ValueError, "architecture_freeze_status"):
            ProductionArchitectureFreeze(**payload)

    def test_freeze_does_not_change_request_routing(self) -> None:
        request = AssistantRequest(
            query="Create a luminous shader study for release architecture freeze.",
            mode=AssistantMode.GENERATE,
            domain=CreativeCodingDomain.THREE_JS,
        )

        before = route_request(request)
        freeze = build_production_architecture_freeze()
        after = route_request(request)

        self.assertEqual(before.route, RouteName.GENERATE)
        self.assertEqual(after.route, before.route)
        self.assertFalse(freeze.provider_model_routing_implemented)
        self.assertFalse(freeze.workflow_graph_mutation_implemented)
        self.assertFalse(freeze.runtime_evolution_implemented)


if __name__ == "__main__":
    unittest.main()
