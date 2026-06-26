import unittest

from creative_coding_assistant.orchestration import (
    engine_contract_consistency_registry,
    engine_contract_family_consistency_by_id,
)
from creative_coding_assistant.orchestration.artifact_engine_contracts import (
    artifact_intelligence_engine_contracts,
)
from creative_coding_assistant.orchestration.evaluation_engine_contracts import (
    evaluation_engine_contracts,
)
from creative_coding_assistant.orchestration.workstation_contracts import (
    workstation_engine_contracts,
)

EXPECTED_FAMILY_IDS = (
    "artifact_intelligence",
    "creative_evaluation",
    "creative_workstation",
)

SOURCE_REGISTRIES = {
    "artifact_intelligence": (
        artifact_intelligence_engine_contracts,
        "engine_contracts",
        "engine_ids",
        "engine_category",
        "engine_id",
    ),
    "creative_evaluation": (
        evaluation_engine_contracts,
        "engine_contracts",
        "engine_ids",
        "engine_category",
        "engine_id",
    ),
    "creative_workstation": (
        workstation_engine_contracts,
        "surface_contracts",
        "surface_ids",
        "surface_category",
        "surface_id",
    ),
}

REQUIRED_SHARED_CONCEPTS = {
    "identity",
    "category",
    "version",
    "authority_boundary",
    "required_inputs",
    "optional_inputs",
    "metadata_surface",
    "signal_surface",
    "quality_or_stability_signals",
    "dependency_surface",
    "cacheability",
    "cost_metadata",
    "latency_metadata",
    "serialization_version",
    "future_agent_hooks",
    "future_execution_hooks",
}


class EngineContractConsistencyRegistryTests(unittest.TestCase):
    def test_registry_exposes_cross_family_metadata_only_contract(self) -> None:
        registry = engine_contract_consistency_registry()

        self.assertEqual(registry.role, "engine_contract_consistency_registry")
        self.assertEqual(
            registry.serialization_version,
            "engine_contract_consistency_registry.v1",
        )
        self.assertEqual(registry.family_ids, EXPECTED_FAMILY_IDS)
        self.assertEqual(registry.family_count, 3)
        self.assertTrue(registry.metadata_only)
        self.assertIn("does not change workflow graph", registry.authority_boundary)
        self.assertEqual(
            set(registry.shared_contract_concepts),
            REQUIRED_SHARED_CONCEPTS,
        )
        self.assertEqual(
            {family.family_id for family in registry.families},
            set(EXPECTED_FAMILY_IDS),
        )

        for family in registry.families:
            self.assertTrue(family.metadata_only)
            self.assertEqual(
                family.serialization_version,
                "engine_contract_family_consistency.v1",
            )
            self.assertEqual(
                family.normalized_concepts,
                registry.shared_contract_concepts,
            )
            self.assertEqual(
                family.blocked_runtime_behaviors,
                registry.blocked_runtime_behaviors,
            )

    def test_profiles_match_source_contract_registries(self) -> None:
        consistency_registry = engine_contract_consistency_registry()

        for family in consistency_registry.families:
            (
                source_registry_factory,
                contract_collection_field,
                id_collection_field,
                category_field,
                item_id_field,
            ) = SOURCE_REGISTRIES[family.family_id]
            source_registry = source_registry_factory()
            source_contracts = getattr(source_registry, contract_collection_field)
            source_ids = getattr(source_registry, id_collection_field)
            first_contract_fields = source_contracts[0].model_dump(mode="json")

            self.assertEqual(family.registry_role, source_registry.role)
            self.assertEqual(
                family.registry_serialization_version,
                source_registry.serialization_version,
            )
            self.assertEqual(
                family.contract_category,
                getattr(source_registry, category_field),
            )
            self.assertEqual(family.contract_ids, source_ids)
            self.assertEqual(family.contract_count, len(source_contracts))
            self.assertEqual(
                tuple(
                    getattr(contract, item_id_field) for contract in source_contracts
                ),
                family.contract_ids,
            )

            for contract in source_contracts:
                self.assertEqual(
                    contract.serialization_version,
                    family.contract_serialization_version,
                )

            declared_source_fields = (
                family.identity_fields
                + family.input_fields
                + family.output_fields
                + family.quality_signal_fields
                + family.relationship_fields
                + family.execution_boundary_fields
                + family.performance_fields
                + family.future_hook_fields
            )
            for source_field in declared_source_fields:
                self.assertIn(source_field, first_contract_fields)

    def test_profiles_normalize_common_engine_contract_concepts(self) -> None:
        registry = engine_contract_consistency_registry()

        for family in registry.families:
            self.assertTrue(
                any(field.endswith("_id") for field in family.identity_fields)
            )
            self.assertTrue(
                any(field.endswith("_name") for field in family.identity_fields)
            )
            self.assertTrue(
                any(field.endswith("_version") for field in family.identity_fields)
            )
            self.assertTrue(
                any(field.endswith("_category") for field in family.identity_fields)
            )
            self.assertEqual(family.input_fields, ("required_inputs", "optional_inputs"))
            self.assertIn("authority_boundary", family.execution_boundary_fields)
            self.assertIn("cacheability", family.performance_fields)
            self.assertIn("estimated_cost_metadata", family.performance_fields)
            self.assertIn("estimated_latency_metadata", family.performance_fields)
            self.assertIn("future_agent_hooks", family.future_hook_fields)
            self.assertIn("future_execution_hooks", family.future_hook_fields)

        workstation = engine_contract_family_consistency_by_id("creative_workstation")
        artifact = engine_contract_family_consistency_by_id("artifact_intelligence")
        evaluation = engine_contract_family_consistency_by_id("creative_evaluation")

        self.assertIsNotNone(workstation)
        self.assertIsNotNone(artifact)
        self.assertIsNotNone(evaluation)
        assert workstation is not None
        assert artifact is not None
        assert evaluation is not None
        self.assertEqual(
            workstation.output_fields,
            ("exposed_metadata", "exposed_signals"),
        )
        self.assertIn("future_evolution_hooks", workstation.future_hook_fields)
        self.assertEqual(
            artifact.output_fields,
            ("produced_metadata", "produced_signals"),
        )
        self.assertIn("parallelization_support", artifact.performance_fields)
        self.assertEqual(
            evaluation.output_fields,
            ("produced_metadata", "produced_signals"),
        )
        self.assertIn("evidence_contract", evaluation.quality_signal_fields)

    def test_lookup_and_serialization_are_stable(self) -> None:
        missing_family = engine_contract_family_consistency_by_id("missing")
        artifact_family = engine_contract_family_consistency_by_id(
            "artifact_intelligence"
        )
        dumped = engine_contract_consistency_registry().model_dump(mode="json")

        self.assertIsNone(missing_family)
        self.assertIsNotNone(artifact_family)
        assert artifact_family is not None
        self.assertEqual(
            artifact_family.registry_role,
            "artifact_intelligence_engine_contract_registry",
        )
        self.assertEqual(dumped["family_ids"], list(EXPECTED_FAMILY_IDS))
        self.assertEqual(len(dumped["families"]), 3)
        self.assertEqual(
            dumped["families"][0]["serialization_version"],
            "engine_contract_family_consistency.v1",
        )
        self.assertNotIn("engine_contracts", dumped["families"][0])
        self.assertNotIn("surface_contracts", dumped["families"][-1])

    def test_registry_does_not_declare_runtime_behavior(self) -> None:
        registry = engine_contract_consistency_registry()
        forbidden_runtime_terms = (
            "execute_provider",
            "autonomous_retry",
            "runtime_auto_selection",
        )
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.shared_contract_concepts,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for family in registry.families
                    for field in (
                        family.identity_fields
                        + family.input_fields
                        + family.output_fields
                        + family.quality_signal_fields
                        + family.relationship_fields
                        + family.execution_boundary_fields
                        + family.performance_fields
                        + family.future_hook_fields
                        + family.blocked_runtime_behaviors
                    )
                ),
            )
        )

        for forbidden_term in forbidden_runtime_terms:
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
