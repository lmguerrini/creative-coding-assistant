import unittest

from creative_coding_assistant.orchestration import (
    ASSISTANT_WORKFLOW_NODE_ORDER,
    hybrid_agentic_workflow_registry,
    hybrid_agentic_workflow_stage_by_id,
)

EXPECTED_STAGE_IDS = (
    "intake_routing_context_readiness",
    "planning_reasoning_readiness",
    "generation_artifact_readiness",
    "review_refinement_readiness",
    "completion_guardrail_readiness",
)

REQUIRED_STAGE_FIELDS = {
    "stage_id",
    "stage_name",
    "authority_boundary",
    "v3_workflow_nodes",
    "future_capability_ids",
    "escalation_rule_ids",
    "source_metadata_registries",
    "advisory_outputs",
    "blocked_runtime_behaviors",
    "serialization_version",
}


class HybridAgenticWorkflowRegistryTests(unittest.TestCase):
    def test_registry_exposes_metadata_only_workflow_stages(self) -> None:
        registry = hybrid_agentic_workflow_registry()

        self.assertEqual(registry.role, "hybrid_agentic_workflow_registry")
        self.assertEqual(registry.stage_ids, EXPECTED_STAGE_IDS)
        self.assertEqual(registry.stage_count, 5)
        self.assertTrue(registry.metadata_only)
        self.assertEqual(
            registry.source_metadata_registries,
            (
                "agent_capability_registry",
                "escalation_policy_registry",
                "artifact_engine_contract_registry",
                "evaluation_engine_contract_registry",
                "workstation_engine_contract_registry",
            ),
        )
        self.assertIn("does not change workflow graph", registry.authority_boundary)
        self.assertEqual(
            {stage.stage_id for stage in registry.stages},
            set(EXPECTED_STAGE_IDS),
        )

        for stage in registry.stages:
            dumped = stage.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_STAGE_FIELDS)
            self.assertEqual(
                stage.serialization_version,
                "hybrid_workflow_stage.v1",
            )
            self.assertTrue(stage.v3_workflow_nodes)
            self.assertTrue(stage.future_capability_ids)
            self.assertTrue(stage.escalation_rule_ids)
            self.assertTrue(stage.advisory_outputs)
            self.assertIn("agent_invocation", stage.blocked_runtime_behaviors)
            self.assertIn(
                "workflow_graph_mutation",
                stage.blocked_runtime_behaviors,
            )
            self.assertIn("does not change V3 workflow", stage.authority_boundary)

    def test_stage_nodes_cover_current_v3_workflow_order(self) -> None:
        registry = hybrid_agentic_workflow_registry()
        declared_nodes = tuple(
            node for stage in registry.stages for node in stage.v3_workflow_nodes
        )

        self.assertEqual(declared_nodes, ASSISTANT_WORKFLOW_NODE_ORDER)

    def test_stage_lookup_is_stable(self) -> None:
        stage = hybrid_agentic_workflow_stage_by_id("review_refinement_readiness")
        missing = hybrid_agentic_workflow_stage_by_id("missing")

        self.assertIsNone(missing)
        self.assertIsNotNone(stage)
        assert stage is not None
        self.assertEqual(stage.v3_workflow_nodes, ("review", "refinement"))
        self.assertIn("v4_agentic_studio", stage.future_capability_ids)
        self.assertIn(
            "future_agent_escalation_readiness",
            stage.escalation_rule_ids,
        )

    def test_registry_serializes_for_future_workflow_consumers(self) -> None:
        dumped = hybrid_agentic_workflow_registry().model_dump(mode="json")

        self.assertEqual(
            dumped["serialization_version"],
            "hybrid_workflow_registry.v1",
        )
        self.assertEqual(dumped["stage_ids"], list(EXPECTED_STAGE_IDS))
        self.assertEqual(len(dumped["stages"]), 5)
        self.assertTrue(dumped["metadata_only"])
        self.assertEqual(
            dumped["stages"][0]["stage_id"],
            "intake_routing_context_readiness",
        )

    def test_registry_does_not_declare_runtime_workflow_behavior(self) -> None:
        registry = hybrid_agentic_workflow_registry()

        for stage in registry.stages:
            combined_text = " ".join(
                (
                    stage.authority_boundary,
                    *stage.advisory_outputs,
                    *stage.blocked_runtime_behaviors,
                )
            )
            self.assertNotIn("execute_provider", combined_text)
            self.assertNotIn("autonomous_retry", combined_text)
            self.assertNotIn("runtime_auto_selection", combined_text)


if __name__ == "__main__":
    unittest.main()
