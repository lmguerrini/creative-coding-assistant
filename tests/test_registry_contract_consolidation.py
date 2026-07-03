import unittest

from creative_coding_assistant import orchestration
from creative_coding_assistant.orchestration import (
    REGISTRY_CONTRACT_CONSOLIDATION_ROADMAP_ITEMS,
    ContractSchemaRecord,
    SourceRegistryInventory,
    audit_registry_public_exports,
    build_contract_version_migrations,
    build_registry_contract_consolidation_plan,
    build_registry_dependency_graph,
    build_schema_evolution_plan,
    check_contract_compatibility,
    diff_registry_inventories,
    explain_registry_source,
    generate_source_registry_inventory,
    normalize_contract_schemas,
    registry_family_by_id,
    registry_source_by_id,
    registry_sources_for_family,
    review_architecture_simplification,
    verify_registry_integrity,
)


class RegistryContractConsolidationTests(unittest.TestCase):
    def test_plan_covers_v7_3_roadmap_without_runtime_behavior(self) -> None:
        plan = build_registry_contract_consolidation_plan()

        self.assertEqual(plan.role, "registry_contract_consolidation")
        self.assertEqual(
            plan.serialization_version,
            "registry_contract_consolidation.v1",
        )
        self.assertEqual(
            plan.covered_roadmap_items,
            REGISTRY_CONTRACT_CONSOLIDATION_ROADMAP_ITEMS,
        )
        self.assertEqual(plan.roadmap_item_count, 24)
        self.assertEqual(plan.source_registry_count, 15)
        self.assertEqual(plan.family_count, 9)
        self.assertEqual(plan.schema_count, 15)
        self.assertEqual(len(plan.shared_builders), 5)
        self.assertEqual(len(plan.shared_passive_boundaries), 4)
        self.assertTrue(plan.coverage_report.coverage_passed)
        self.assertTrue(plan.public_export_audit.public_exports_stable)
        self.assertTrue(plan.integrity_report.integrity_passed)
        self.assertTrue(plan.compatibility_report.backward_compatibility_preserved)
        self.assertIn(
            "provider_model_routing_change",
            plan.blocked_runtime_behaviors,
        )
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.prompt_rendering_change_implemented)
        self.assertFalse(plan.jinja_template_mutation_implemented)
        self.assertFalse(plan.logging_configuration_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_inventory_splits_families_and_audits_public_exports(self) -> None:
        inventory = generate_source_registry_inventory()
        public_audit = audit_registry_public_exports(inventory)
        agent_family = registry_family_by_id("agent_contracts", inventory)
        agent_sources = registry_sources_for_family("agent_contracts", inventory)
        typed_failure = registry_source_by_id(
            "failure_taxonomy::typed_failure_taxonomy",
            inventory,
        )

        self.assertIsInstance(inventory, SourceRegistryInventory)
        self.assertEqual(inventory.source_registry_count, 15)
        self.assertEqual(inventory.family_count, 9)
        self.assertEqual(len(agent_sources), 3)
        self.assertIsNotNone(agent_family)
        assert agent_family is not None
        self.assertEqual(
            agent_family.primary_contract_kind,
            "agent_contract",
        )
        self.assertIsNotNone(typed_failure)
        assert typed_failure is not None
        self.assertEqual(
            typed_failure.dependency_registry_ids,
            ("workflow_runtime::runtime_graph_consolidation",),
        )
        self.assertIn(
            "build_typed_failure_taxonomy_registry",
            typed_failure.public_export_names,
        )
        self.assertTrue(public_audit.public_exports_stable)
        self.assertFalse(public_audit.missing_export_names)
        self.assertFalse(public_audit.duplicate_export_names)
        self.assertIn(
            "build_registry_contract_consolidation_plan",
            orchestration.__all__,
        )

    def test_public_export_audit_reports_duplicate_exports(self) -> None:
        inventory = generate_source_registry_inventory()
        first_source = inventory.source_registries[0]
        duplicated_source = first_source.model_copy(
            update={
                "public_export_names": (
                    first_source.builder_name,
                    first_source.builder_name,
                ),
            },
        )
        duplicated_inventory = SourceRegistryInventory(
            source_registries=(duplicated_source, *inventory.source_registries[1:]),
            registry_families=inventory.registry_families,
            source_registry_ids=inventory.source_registry_ids,
            family_ids=inventory.family_ids,
            source_registry_count=inventory.source_registry_count,
            family_count=inventory.family_count,
            blocked_runtime_behaviors=inventory.blocked_runtime_behaviors,
        )

        public_audit = audit_registry_public_exports(duplicated_inventory)

        self.assertFalse(public_audit.public_exports_stable)
        self.assertEqual(
            public_audit.duplicate_export_names,
            (first_source.builder_name,),
        )

    def test_schema_normalization_compatibility_and_migrations(self) -> None:
        inventory = generate_source_registry_inventory()
        schemas = normalize_contract_schemas(inventory)
        integrity = verify_registry_integrity(inventory, schemas)
        compatibility = check_contract_compatibility(schemas)
        migrations = build_contract_version_migrations(schemas)
        evolution = build_schema_evolution_plan(migrations)

        self.assertEqual(len(schemas), inventory.source_registry_count)
        self.assertTrue(
            all(isinstance(schema, ContractSchemaRecord) for schema in schemas),
        )
        self.assertTrue(
            all("role" in schema.normalized_required_fields for schema in schemas),
        )
        self.assertEqual(
            tuple(schema.normalized_schema_hash for schema in schemas),
            tuple(
                schema.normalized_schema_hash
                for schema in normalize_contract_schemas(inventory)
            ),
        )
        self.assertTrue(integrity.integrity_passed)
        self.assertFalse(integrity.duplicate_source_registry_ids)
        self.assertFalse(integrity.missing_dependency_registry_ids)
        self.assertTrue(compatibility.backward_compatibility_preserved)
        self.assertFalse(compatibility.incompatible_schema_ids)
        self.assertEqual(len(migrations), len(schemas))
        self.assertTrue(all(not migration.breaking_change for migration in migrations))
        self.assertTrue(
            all(
                not migration.migration_execution_implemented
                for migration in migrations
            ),
        )
        self.assertEqual(evolution.migration_count, len(migrations))
        self.assertFalse(evolution.automatic_migration_implemented)

    def test_compatibility_requires_all_boundary_keys(self) -> None:
        schema = normalize_contract_schemas()[0]
        incompatible_schema = schema.model_copy(
            update={
                "normalized_required_fields": (
                    "role",
                    "serialization_version",
                    "advisory_only",
                ),
            },
        )

        compatibility = check_contract_compatibility((incompatible_schema,))

        self.assertFalse(compatibility.backward_compatibility_preserved)
        self.assertEqual(
            compatibility.incompatible_schema_ids,
            (incompatible_schema.schema_id,),
        )

    def test_dependency_graph_diff_and_explainability_are_stable(self) -> None:
        inventory = generate_source_registry_inventory()
        graph = build_registry_dependency_graph(inventory)
        diff = diff_registry_inventories(inventory, inventory)
        explanation = explain_registry_source(
            "model_routing::model_routing_intelligence_registry",
        )
        plan = build_registry_contract_consolidation_plan()

        self.assertEqual(graph.node_count, inventory.source_registry_count)
        self.assertEqual(graph.edge_count, 8)
        self.assertFalse(graph.dependency_cycles_detected)
        self.assertIn(
            "hybrid_studio::model_profile_registry",
            tuple(edge.target_registry_id for edge in graph.edges),
        )
        self.assertEqual(diff.diff_status, "no_change")
        self.assertFalse(diff.behavior_change_detected)
        self.assertIsNotNone(explanation)
        assert explanation is not None
        self.assertIn("advisory-only", explanation)
        self.assertEqual(len(plan.explanations), inventory.family_count)

    def test_reviews_cover_style_jinja_logging_and_simplification(self) -> None:
        plan = build_registry_contract_consolidation_plan()
        simplification = review_architecture_simplification(plan.source_inventory)
        finding_items = tuple(finding.roadmap_item for finding in plan.review_findings)

        self.assertIn("Pydantic Review", finding_items)
        self.assertIn("Jinja2 Review", finding_items)
        self.assertIn("Style Review", finding_items)
        self.assertIn("Code Style & Comment Quality Audit", finding_items)
        self.assertIn("Logging Architecture Review", finding_items)
        self.assertIn("Registry Package Consolidation", finding_items)
        self.assertIn("Contract Simplification", finding_items)
        self.assertIn("Metadata-to-Code Ratio Review", finding_items)
        self.assertTrue(
            all(finding.status == "pass" for finding in plan.review_findings),
        )
        self.assertTrue(simplification.long_term_system_simpler)
        self.assertFalse(simplification.deferred_refactor_surfaces)
        self.assertEqual(
            simplification.covered_roadmap_items,
            ("Architecture Simplification Review",),
        )


if __name__ == "__main__":
    unittest.main()
