import unittest

from creative_coding_assistant.orchestration import (
    workstation_engine_contract_by_id,
    workstation_engine_contracts,
)

EXPECTED_SURFACE_IDS = (
    "workstation_state",
    "session_intelligence",
    "workflow_explorer",
    "provenance_engine",
    "creative_timeline",
    "v3_inspector_panels",
    "workstation_dashboard",
)

REQUIRED_CONTRACT_FIELDS = {
    "surface_id",
    "surface_name",
    "surface_version",
    "surface_category",
    "authority_boundary",
    "required_inputs",
    "optional_inputs",
    "exposed_metadata",
    "exposed_signals",
    "stability_signals",
    "missing_metadata_behavior",
    "downstream_consumers",
    "upstream_dependencies",
    "cacheability",
    "hydration_mode",
    "estimated_cost_metadata",
    "estimated_latency_metadata",
    "serialization_version",
    "future_agent_hooks",
    "future_execution_hooks",
    "future_evolution_hooks",
}


class WorkstationEngineContractTests(unittest.TestCase):
    def test_registry_exposes_consistent_contract_surface(self) -> None:
        registry = workstation_engine_contracts()

        self.assertEqual(registry.role, "workstation_engine_contract_registry")
        self.assertEqual(registry.surface_category, "creative_workstation")
        self.assertEqual(registry.surface_ids, EXPECTED_SURFACE_IDS)
        self.assertEqual(registry.contract_count, 7)
        self.assertEqual(
            registry.future_capability_consumers,
            (
                "v4_agentic_studio",
                "v5_adaptive_creative_execution",
                "v6_creative_evolution",
            ),
        )
        self.assertEqual(
            {contract.surface_id for contract in registry.surface_contracts},
            set(EXPECTED_SURFACE_IDS),
        )
        for contract in registry.surface_contracts:
            dumped = contract.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_CONTRACT_FIELDS)
            self.assertEqual(contract.surface_version, "v3.5")
            self.assertEqual(contract.surface_category, "creative_workstation")
            self.assertEqual(
                contract.serialization_version,
                "workstation_engine_contract.v1",
            )
            self.assertIn("does not", contract.authority_boundary)
            self.assertTrue(contract.required_inputs)
            self.assertTrue(contract.exposed_metadata)
            self.assertTrue(contract.exposed_signals)
            self.assertTrue(contract.stability_signals)
            self.assertTrue(contract.missing_metadata_behavior)
            self.assertTrue(contract.future_agent_hooks)
            self.assertTrue(contract.future_execution_hooks)
            self.assertTrue(contract.future_evolution_hooks)
            self.assertFalse(
                contract.estimated_cost_metadata.external_provider_calls
            )

    def test_contract_lookup_and_future_hooks_are_stable(self) -> None:
        timeline_contract = workstation_engine_contract_by_id("creative_timeline")
        dashboard_contract = workstation_engine_contract_by_id(
            "workstation_dashboard"
        )
        missing_contract = workstation_engine_contract_by_id("missing")

        self.assertIsNone(missing_contract)
        self.assertIsNotNone(timeline_contract)
        self.assertIsNotNone(dashboard_contract)
        assert timeline_contract is not None
        assert dashboard_contract is not None
        self.assertIn(
            "provenance_engine",
            timeline_contract.upstream_dependencies,
        )
        self.assertIn(
            "v6_creative_evolution_timeline_context",
            timeline_contract.future_evolution_hooks,
        )
        self.assertIn(
            "v5_adaptive_execution_context",
            dashboard_contract.future_execution_hooks,
        )
        self.assertIn(
            "v3_inspector_panels",
            dashboard_contract.upstream_dependencies,
        )
        self.assertIn("HITL", dashboard_contract.authority_boundary)
        self.assertIn(
            "hitl_recommendation_card",
            dashboard_contract.exposed_metadata,
        )
        self.assertIn(
            "hitl_recommendation",
            dashboard_contract.exposed_signals,
        )

    def test_registry_serializes_for_architecture_metadata(self) -> None:
        dumped = workstation_engine_contracts().model_dump(mode="json")

        self.assertEqual(
            dumped["serialization_version"],
            "workstation_engine_contract_registry.v1",
        )
        self.assertEqual(dumped["surface_ids"], list(EXPECTED_SURFACE_IDS))
        self.assertEqual(len(dumped["surface_contracts"]), 7)
        self.assertEqual(
            dumped["surface_contracts"][0]["surface_id"],
            "workstation_state",
        )
        self.assertIn(
            "future_evolution_hooks",
            dumped["surface_contracts"][-1],
        )

    def test_contracts_remain_metadata_only(self) -> None:
        registry = workstation_engine_contracts()

        for contract in registry.surface_contracts:
            combined_contract_text = " ".join(
                (
                    contract.authority_boundary,
                    contract.missing_metadata_behavior,
                    *contract.exposed_signals,
                    *contract.future_agent_hooks,
                    *contract.future_execution_hooks,
                    *contract.future_evolution_hooks,
                )
            )
            self.assertNotIn("execute_provider", combined_contract_text)
            self.assertNotIn("autonomous_retry", combined_contract_text)
            self.assertNotIn("runtime_auto_selection", combined_contract_text)


if __name__ == "__main__":
    unittest.main()
