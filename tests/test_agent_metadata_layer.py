import unittest

from creative_coding_assistant.orchestration import (
    AgentMetadataRegistry,
    agent_contract_registry,
    agent_metadata_by_agent_id,
    agent_metadata_registry,
)

EXPECTED_AGENT_IDS = (
    "planner_agent",
    "research_agent",
    "style_agent",
    "runtime_agent",
    "artifact_agent",
    "art_direction_agent",
    "aesthetic_critic_agent",
    "narrative_symbolic_agent",
    "creative_curator_agent",
    "critic_agent",
    "refiner_agent",
    "final_synthesizer_agent",
)

REQUIRED_METADATA_FIELDS = {
    "agent_id",
    "role_id",
    "cacheability",
    "parallelization_support",
    "estimated_cost_class",
    "estimated_cost_basis",
    "estimated_latency_class",
    "estimated_latency_basis",
    "observability_surfaces",
    "auditability_surfaces",
    "future_orchestration_readiness",
    "advisory_only",
    "caching_implemented",
    "parallel_execution_implemented",
    "cost_latency_routing_implemented",
    "serialization_version",
    "metadata_only",
}


class AgentMetadataLayerTests(unittest.TestCase):
    def test_metadata_registry_covers_all_agent_contracts(self) -> None:
        metadata_registry = agent_metadata_registry()
        contract_registry = agent_contract_registry()

        self.assertEqual(metadata_registry.role, "agent_metadata_registry")
        self.assertEqual(
            metadata_registry.serialization_version,
            "agent_metadata_registry.v1",
        )
        self.assertEqual(metadata_registry.agent_ids, EXPECTED_AGENT_IDS)
        self.assertEqual(metadata_registry.agent_ids, contract_registry.agent_ids)
        self.assertEqual(metadata_registry.metadata_count, 12)
        self.assertEqual(
            metadata_registry.source_contract_registry,
            "agent_contract_registry",
        )
        self.assertTrue(metadata_registry.advisory_only)
        self.assertTrue(metadata_registry.metadata_only)
        self.assertFalse(metadata_registry.caching_implemented)
        self.assertFalse(metadata_registry.parallel_execution_implemented)
        self.assertFalse(metadata_registry.cost_latency_routing_implemented)
        self.assertIn("does not implement caching", metadata_registry.authority_boundary)

    def test_metadata_fields_are_consistent_for_every_agent(self) -> None:
        registry = agent_metadata_registry()

        for item in registry.metadata:
            dumped = item.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_METADATA_FIELDS)
            self.assertEqual(item.serialization_version, "agent_metadata.v1")
            self.assertEqual(item.observability_surfaces, registry.observability_surfaces)
            self.assertEqual(item.auditability_surfaces, registry.auditability_surfaces)
            self.assertEqual(item.parallelization_support, "parallel_after_required_inputs")
            self.assertEqual(
                item.future_orchestration_readiness,
                "future_orchestration_metadata_ready",
            )
            self.assertIn(item.estimated_cost_class, ("none", "low", "medium"))
            self.assertIn(item.estimated_latency_class, ("none", "low", "medium"))
            self.assertIn("no", item.estimated_cost_basis.lower())
            self.assertIn("no", item.estimated_latency_basis.lower())
            self.assertTrue(item.advisory_only)
            self.assertTrue(item.metadata_only)
            self.assertFalse(item.caching_implemented)
            self.assertFalse(item.parallel_execution_implemented)
            self.assertFalse(item.cost_latency_routing_implemented)

    def test_metadata_lookup_is_stable(self) -> None:
        planner_metadata = agent_metadata_by_agent_id("planner_agent")
        missing_metadata = agent_metadata_by_agent_id("missing")

        self.assertIsNone(missing_metadata)
        self.assertIsNotNone(planner_metadata)
        assert planner_metadata is not None
        self.assertEqual(planner_metadata.role_id, "planner")
        self.assertEqual(
            planner_metadata.cacheability,
            "deterministic_with_upstream_metadata",
        )
        self.assertEqual(planner_metadata.estimated_cost_class, "low")
        self.assertEqual(planner_metadata.estimated_latency_class, "low")

    def test_registry_rejects_mismatched_metadata(self) -> None:
        registry = agent_metadata_registry()
        first_item = registry.metadata[0]
        duplicate_item = first_item.model_copy(update={"role_id": "duplicate"})

        with self.assertRaisesRegex(ValueError, "agent_ids must be unique"):
            AgentMetadataRegistry(
                metadata=(first_item, duplicate_item) + registry.metadata[2:],
                agent_ids=registry.agent_ids,
                metadata_count=12,
                observability_surfaces=registry.observability_surfaces,
                auditability_surfaces=registry.auditability_surfaces,
            )

        with self.assertRaisesRegex(ValueError, "agent_ids must match metadata"):
            AgentMetadataRegistry(
                metadata=registry.metadata,
                agent_ids=("other_agent",) + registry.agent_ids[1:],
                metadata_count=12,
                observability_surfaces=registry.observability_surfaces,
                auditability_surfaces=registry.auditability_surfaces,
            )

    def test_metadata_layer_does_not_declare_runtime_optimization(self) -> None:
        registry = agent_metadata_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for item in registry.metadata
                    for field in (
                        item.agent_id,
                        item.role_id,
                        item.cacheability,
                        item.parallelization_support,
                        item.future_orchestration_readiness,
                    )
                ),
            )
        )

        for forbidden_term in (
            "run_cache",
            "execute_parallel",
            "route_by_cost",
            "route_by_latency",
            "runtime_auto_selection",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
