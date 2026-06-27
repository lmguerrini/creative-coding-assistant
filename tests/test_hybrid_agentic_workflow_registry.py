import unittest

from creative_coding_assistant.orchestration import (
    ASSISTANT_WORKFLOW_NODE_ORDER,
    V3BackboneModeRegistry,
    hybrid_agentic_workflow_registry,
    hybrid_agentic_workflow_stage_by_id,
    v3_backbone_mode_profile_by_node_id,
    v3_backbone_mode_registry,
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
REQUIRED_BACKBONE_PROFILE_FIELDS = {
    "mode_id",
    "node_id",
    "phase",
    "active_runtime_owner",
    "preserved_surfaces",
    "source_registries",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "backbone_runtime_active",
    "workflow_order_mutation_implemented",
    "provider_model_routing_implemented",
    "agent_invocation_implemented",
    "multi_agent_escalation_executed",
    "generated_output_mutation_implemented",
    "serialization_version",
    "metadata_only",
}
EXPECTED_BACKBONE_PHASES = (
    "context_intake",
    "planning_reasoning",
    "generation_artifact",
    "review_refinement",
    "terminal_guardrail",
)
EXPECTED_BACKBONE_SOURCE_REGISTRIES = (
    "assistant_workflow_node_order",
    "workflow_step_order",
    "artifact_engine_contract_registry",
    "evaluation_engine_contract_registry",
    "workstation_engine_contract_registry",
    "hybrid_agentic_workflow_registry",
)


class V3BackboneModeRegistryTests(unittest.TestCase):
    def test_registry_declares_v3_workflow_as_active_backbone(self) -> None:
        registry = v3_backbone_mode_registry()

        self.assertEqual(registry.role, "v3_backbone_mode_registry")
        self.assertEqual(registry.mode_id, "v3_backbone_mode")
        self.assertEqual(
            registry.serialization_version,
            "v3_backbone_mode_registry.v1",
        )
        self.assertEqual(registry.node_ids, ASSISTANT_WORKFLOW_NODE_ORDER)
        self.assertEqual(
            registry.preserved_workflow_order,
            ASSISTANT_WORKFLOW_NODE_ORDER,
        )
        self.assertEqual(registry.node_count, len(ASSISTANT_WORKFLOW_NODE_ORDER))
        self.assertEqual(registry.phase_ids, EXPECTED_BACKBONE_PHASES)
        self.assertEqual(
            registry.source_registries,
            EXPECTED_BACKBONE_SOURCE_REGISTRIES,
        )
        self.assertIn("does not change workflow graph order", registry.authority_boundary)
        self.assertTrue(registry.backbone_runtime_active)
        self.assertFalse(registry.workflow_order_mutation_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.agent_invocation_implemented)
        self.assertFalse(registry.multi_agent_escalation_executed)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertTrue(registry.metadata_only)

    def test_backbone_node_profiles_are_passive_and_source_aligned(self) -> None:
        registry = v3_backbone_mode_registry()

        for profile in registry.node_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_BACKBONE_PROFILE_FIELDS)
            self.assertEqual(profile.mode_id, "v3_backbone_mode")
            self.assertEqual(profile.active_runtime_owner, "v3_workflow_graph")
            self.assertEqual(
                profile.source_registries,
                EXPECTED_BACKBONE_SOURCE_REGISTRIES,
            )
            self.assertTrue(profile.preserved_surfaces)
            self.assertIn("workflow_graph_mutation", profile.blocked_runtime_behaviors)
            self.assertIn(
                "multi_agent_escalation_execution",
                profile.blocked_runtime_behaviors,
            )
            self.assertIn("agent_invocation", profile.blocked_runtime_behaviors)
            self.assertTrue(profile.backbone_runtime_active)
            self.assertFalse(profile.workflow_order_mutation_implemented)
            self.assertFalse(profile.provider_model_routing_implemented)
            self.assertFalse(profile.agent_invocation_implemented)
            self.assertFalse(profile.multi_agent_escalation_executed)
            self.assertFalse(profile.generated_output_mutation_implemented)
            self.assertEqual(
                profile.serialization_version,
                "v3_backbone_mode_node.v1",
            )
            self.assertTrue(profile.metadata_only)

    def test_backbone_source_registries_are_complete_for_all_profiles(self) -> None:
        registry = v3_backbone_mode_registry()
        profile_sources = tuple(
            dict.fromkeys(
                source
                for profile in registry.node_profiles
                for source in profile.source_registries
            )
        )

        self.assertEqual(profile_sources, registry.source_registries)
        for source_registry in EXPECTED_BACKBONE_SOURCE_REGISTRIES:
            self.assertIn(source_registry, profile_sources)
        for profile in registry.node_profiles:
            self.assertEqual(set(profile.source_registries), set(profile_sources))

    def test_backbone_lookup_is_stable(self) -> None:
        generation = v3_backbone_mode_profile_by_node_id("generation")
        missing = v3_backbone_mode_profile_by_node_id("missing_node")

        self.assertIsNone(missing)
        self.assertIsNotNone(generation)
        assert generation is not None
        self.assertEqual(generation.phase, "generation_artifact")
        self.assertIn("generation_stream", generation.preserved_surfaces)
        self.assertFalse(generation.provider_model_routing_implemented)

    def test_backbone_registry_rejects_mismatched_metadata(self) -> None:
        registry = v3_backbone_mode_registry()
        mismatched_profile = registry.node_profiles[0].model_copy(
            update={"node_id": "other_node"}
        )
        mismatched_sources = registry.node_profiles[0].model_copy(
            update={"source_registries": ("assistant_workflow_node_order",) * 6}
        )

        with self.assertRaisesRegex(ValueError, "node_ids must match"):
            V3BackboneModeRegistry(
                node_profiles=(mismatched_profile,) + registry.node_profiles[1:],
                node_ids=registry.node_ids,
                preserved_workflow_order=registry.preserved_workflow_order,
                phase_ids=registry.phase_ids,
                source_registries=registry.source_registries,
                node_count=registry.node_count,
            )

        with self.assertRaisesRegex(ValueError, "node profile sources must match"):
            V3BackboneModeRegistry(
                node_profiles=(mismatched_sources,) + registry.node_profiles[1:],
                node_ids=registry.node_ids,
                preserved_workflow_order=registry.preserved_workflow_order,
                phase_ids=registry.phase_ids,
                source_registries=registry.source_registries,
                node_count=registry.node_count,
            )

    def test_backbone_mode_does_not_declare_active_hybrid_execution(self) -> None:
        registry = v3_backbone_mode_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for profile in registry.node_profiles
                    for field in (
                        profile.node_id,
                        profile.authority_boundary,
                        *profile.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "execute_agent",
            "route_provider",
            "run_multi_agent",
            "mutate_prompt",
            "modify_output",
        ):
            self.assertNotIn(forbidden_term, combined_text)


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
