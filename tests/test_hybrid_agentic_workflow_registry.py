import unittest

from creative_coding_assistant.orchestration import (
    ASSISTANT_WORKFLOW_NODE_ORDER,
    ConditionalMultiAgentEscalationRegistry,
    EscalationGateRegistry,
    SpecialistAgentLoopRegistry,
    V3BackboneModeRegistry,
    agent_contract_registry,
    agent_capability_registry,
    agent_escalation_signal_registry,
    conditional_multi_agent_escalation_condition_by_id,
    conditional_multi_agent_escalation_registry,
    escalation_gate_by_id,
    escalation_gate_registry,
    escalation_policy_registry,
    hybrid_agentic_workflow_registry,
    hybrid_agentic_workflow_stage_by_id,
    specialist_agent_loop_by_id,
    specialist_agent_loop_registry,
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
EXPECTED_CONDITIONAL_ESCALATION_IDS = (
    "planning_ambiguity_multi_agent_candidate",
    "artifact_risk_multi_agent_candidate",
    "runtime_fit_multi_agent_candidate",
    "evaluation_confidence_multi_agent_candidate",
    "terminal_guardrail_multi_agent_candidate",
)
EXPECTED_CONDITIONAL_ESCALATION_CATEGORIES = (
    "ambiguity",
    "risk",
    "runtime",
    "quality",
    "hitl",
)
EXPECTED_CONDITIONAL_ESCALATION_SOURCE_REGISTRIES = (
    "v3_backbone_mode_registry",
    "agent_capability_registry",
    "escalation_policy_registry",
    "agent_escalation_signal_registry",
    "hybrid_agentic_workflow_registry",
)
REQUIRED_CONDITIONAL_ESCALATION_FIELDS = {
    "condition_id",
    "condition_name",
    "category",
    "backbone_phase",
    "source_node_ids",
    "source_registries",
    "capability_ids",
    "policy_rule_ids",
    "escalation_signal_ids",
    "advisory_outputs",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "condition_evaluation_implemented",
    "multi_agent_execution_implemented",
    "provider_model_routing_implemented",
    "workflow_control_implemented",
    "retry_triggering_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "metadata_only",
}
EXPECTED_SPECIALIST_LOOP_IDS = (
    "planning_specialist_agent_loop",
    "artifact_specialist_agent_loop",
    "runtime_specialist_agent_loop",
    "evaluation_specialist_agent_loop",
    "synthesis_specialist_agent_loop",
)
EXPECTED_SPECIALIST_LOOP_CATEGORIES = (
    "planning",
    "artifact",
    "runtime",
    "evaluation",
    "synthesis",
)
EXPECTED_SPECIALIST_LOOP_SOURCE_REGISTRIES = (
    "agent_contract_registry",
    "conditional_multi_agent_escalation_registry",
    "v3_backbone_mode_registry",
    "hybrid_agentic_workflow_registry",
)
REQUIRED_SPECIALIST_LOOP_FIELDS = {
    "loop_id",
    "loop_name",
    "category",
    "specialist_agent_ids",
    "source_condition_ids",
    "source_node_ids",
    "source_registries",
    "loop_inputs",
    "advisory_outputs",
    "max_advisory_passes",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "loop_execution_implemented",
    "agent_invocation_implemented",
    "multi_agent_orchestration_implemented",
    "provider_model_routing_implemented",
    "workflow_control_implemented",
    "retry_triggering_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "metadata_only",
}
EXPECTED_ESCALATION_GATE_IDS = (
    "backbone_entry_escalation_gate",
    "evidence_completeness_escalation_gate",
    "specialist_loop_boundary_gate",
    "human_review_visibility_gate",
    "return_handoff_escalation_gate",
)
EXPECTED_ESCALATION_GATE_KINDS = (
    "backbone_entry",
    "evidence_completeness",
    "specialist_loop_boundary",
    "human_review_visibility",
    "return_handoff",
)
EXPECTED_ESCALATION_GATE_SOURCE_REGISTRIES = (
    "v3_backbone_mode_registry",
    "conditional_multi_agent_escalation_registry",
    "specialist_agent_loop_registry",
    "escalation_policy_registry",
    "hybrid_agentic_workflow_registry",
)
REQUIRED_ESCALATION_GATE_FIELDS = {
    "gate_id",
    "gate_name",
    "gate_kind",
    "source_condition_ids",
    "source_loop_ids",
    "source_registries",
    "required_passive_inputs",
    "advisory_decision_outputs",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "gate_evaluation_implemented",
    "escalation_approval_implemented",
    "agent_invocation_implemented",
    "workflow_control_implemented",
    "retry_triggering_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "metadata_only",
}


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


class ConditionalMultiAgentEscalationRegistryTests(unittest.TestCase):
    def test_registry_declares_passive_conditional_escalation_candidates(self) -> None:
        registry = conditional_multi_agent_escalation_registry()

        self.assertEqual(
            registry.role,
            "conditional_multi_agent_escalation_registry",
        )
        self.assertEqual(
            registry.serialization_version,
            "conditional_multi_agent_escalation_registry.v1",
        )
        self.assertEqual(
            registry.condition_ids,
            EXPECTED_CONDITIONAL_ESCALATION_IDS,
        )
        self.assertEqual(
            registry.categories,
            EXPECTED_CONDITIONAL_ESCALATION_CATEGORIES,
        )
        self.assertEqual(
            registry.source_registries,
            EXPECTED_CONDITIONAL_ESCALATION_SOURCE_REGISTRIES,
        )
        self.assertEqual(registry.backbone_node_ids, ASSISTANT_WORKFLOW_NODE_ORDER)
        self.assertEqual(registry.condition_count, 5)
        self.assertIn("does not evaluate conditions", registry.authority_boundary)
        self.assertFalse(registry.condition_evaluation_implemented)
        self.assertFalse(registry.multi_agent_execution_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertTrue(registry.metadata_only)

    def test_conditions_reference_existing_passive_sources(self) -> None:
        registry = conditional_multi_agent_escalation_registry()
        known_nodes = set(v3_backbone_mode_registry().node_ids)
        known_capabilities = set(agent_capability_registry().capability_ids)
        known_policy_rules = set(escalation_policy_registry().rule_ids)
        known_signals = set(agent_escalation_signal_registry().signal_ids)

        for condition in registry.conditions:
            dumped = condition.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_CONDITIONAL_ESCALATION_FIELDS)
            self.assertEqual(
                condition.source_registries,
                EXPECTED_CONDITIONAL_ESCALATION_SOURCE_REGISTRIES,
            )
            self.assertTrue(set(condition.source_node_ids).issubset(known_nodes))
            self.assertTrue(set(condition.capability_ids).issubset(known_capabilities))
            self.assertTrue(set(condition.policy_rule_ids).issubset(known_policy_rules))
            self.assertTrue(set(condition.escalation_signal_ids).issubset(known_signals))
            self.assertTrue(condition.advisory_outputs)
            self.assertIn("condition_evaluation", condition.blocked_runtime_behaviors)
            self.assertIn("multi_agent_execution", condition.blocked_runtime_behaviors)
            self.assertIn("agent_invocation", condition.blocked_runtime_behaviors)
            self.assertFalse(condition.condition_evaluation_implemented)
            self.assertFalse(condition.multi_agent_execution_implemented)
            self.assertFalse(condition.provider_model_routing_implemented)
            self.assertFalse(condition.workflow_control_implemented)
            self.assertFalse(condition.retry_triggering_implemented)
            self.assertFalse(condition.generated_output_mutation_implemented)
            self.assertEqual(
                condition.serialization_version,
                "conditional_multi_agent_escalation_condition.v1",
            )
            self.assertTrue(condition.metadata_only)

    def test_condition_source_registries_are_complete(self) -> None:
        registry = conditional_multi_agent_escalation_registry()
        condition_sources = tuple(
            dict.fromkeys(
                source
                for condition in registry.conditions
                for source in condition.source_registries
            )
        )

        self.assertEqual(condition_sources, registry.source_registries)
        for source_registry in EXPECTED_CONDITIONAL_ESCALATION_SOURCE_REGISTRIES:
            self.assertIn(source_registry, condition_sources)
        for condition in registry.conditions:
            self.assertEqual(set(condition.source_registries), set(condition_sources))

    def test_condition_lookup_is_stable(self) -> None:
        condition = conditional_multi_agent_escalation_condition_by_id(
            "runtime_fit_multi_agent_candidate"
        )
        missing = conditional_multi_agent_escalation_condition_by_id("missing")

        self.assertIsNone(missing)
        self.assertIsNotNone(condition)
        assert condition is not None
        self.assertEqual(condition.category, "runtime")
        self.assertEqual(condition.backbone_phase, "generation_artifact")
        self.assertIn("v4_runtime_agent", condition.capability_ids)
        self.assertFalse(condition.multi_agent_execution_implemented)

    def test_conditional_registry_rejects_mismatched_metadata(self) -> None:
        registry = conditional_multi_agent_escalation_registry()
        mismatched_condition = registry.conditions[0].model_copy(
            update={"condition_id": "other_condition"}
        )
        unknown_node_condition = registry.conditions[0].model_copy(
            update={"source_node_ids": ("missing_node",)}
        )

        with self.assertRaisesRegex(ValueError, "condition_ids must match"):
            ConditionalMultiAgentEscalationRegistry(
                conditions=(mismatched_condition,) + registry.conditions[1:],
                condition_ids=registry.condition_ids,
                categories=registry.categories,
                source_registries=registry.source_registries,
                backbone_node_ids=registry.backbone_node_ids,
                condition_count=registry.condition_count,
            )

        with self.assertRaisesRegex(ValueError, "source nodes must be V3 backbone"):
            ConditionalMultiAgentEscalationRegistry(
                conditions=(unknown_node_condition,) + registry.conditions[1:],
                condition_ids=registry.condition_ids,
                categories=registry.categories,
                source_registries=registry.source_registries,
                backbone_node_ids=registry.backbone_node_ids,
                condition_count=registry.condition_count,
            )

    def test_conditional_escalation_does_not_declare_active_execution(self) -> None:
        registry = conditional_multi_agent_escalation_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for condition in registry.conditions
                    for field in (
                        condition.condition_id,
                        condition.authority_boundary,
                        *condition.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "execute_agent",
            "run_agents",
            "route_provider",
            "trigger_retry",
            "modify_output",
        ):
            self.assertNotIn(forbidden_term, combined_text)


class SpecialistAgentLoopRegistryTests(unittest.TestCase):
    def test_registry_declares_passive_specialist_loops(self) -> None:
        registry = specialist_agent_loop_registry()

        self.assertEqual(registry.role, "specialist_agent_loop_registry")
        self.assertEqual(
            registry.serialization_version,
            "specialist_agent_loop_registry.v1",
        )
        self.assertEqual(registry.loop_ids, EXPECTED_SPECIALIST_LOOP_IDS)
        self.assertEqual(registry.categories, EXPECTED_SPECIALIST_LOOP_CATEGORIES)
        self.assertEqual(
            registry.source_registries,
            EXPECTED_SPECIALIST_LOOP_SOURCE_REGISTRIES,
        )
        self.assertEqual(
            registry.condition_ids,
            conditional_multi_agent_escalation_registry().condition_ids,
        )
        self.assertEqual(registry.backbone_node_ids, ASSISTANT_WORKFLOW_NODE_ORDER)
        self.assertEqual(registry.loop_count, 5)
        self.assertIn("does not execute loops", registry.authority_boundary)
        self.assertFalse(registry.loop_execution_implemented)
        self.assertFalse(registry.agent_invocation_implemented)
        self.assertFalse(registry.multi_agent_orchestration_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertTrue(registry.metadata_only)

    def test_specialist_loops_reference_known_passive_agent_contracts(self) -> None:
        registry = specialist_agent_loop_registry()
        known_agents = set(agent_contract_registry().agent_ids)
        known_conditions = set(conditional_multi_agent_escalation_registry().condition_ids)
        known_nodes = set(v3_backbone_mode_registry().node_ids)

        for loop in registry.loops:
            dumped = loop.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_SPECIALIST_LOOP_FIELDS)
            self.assertEqual(
                loop.source_registries,
                EXPECTED_SPECIALIST_LOOP_SOURCE_REGISTRIES,
            )
            self.assertTrue(set(loop.specialist_agent_ids).issubset(known_agents))
            self.assertTrue(set(loop.source_condition_ids).issubset(known_conditions))
            self.assertTrue(set(loop.source_node_ids).issubset(known_nodes))
            self.assertGreaterEqual(loop.max_advisory_passes, 1)
            self.assertLessEqual(loop.max_advisory_passes, 3)
            self.assertTrue(loop.loop_inputs)
            self.assertTrue(loop.advisory_outputs)
            self.assertIn("loop_execution", loop.blocked_runtime_behaviors)
            self.assertIn("agent_invocation", loop.blocked_runtime_behaviors)
            self.assertIn(
                "multi_agent_orchestration",
                loop.blocked_runtime_behaviors,
            )
            self.assertFalse(loop.loop_execution_implemented)
            self.assertFalse(loop.agent_invocation_implemented)
            self.assertFalse(loop.multi_agent_orchestration_implemented)
            self.assertFalse(loop.provider_model_routing_implemented)
            self.assertFalse(loop.workflow_control_implemented)
            self.assertFalse(loop.retry_triggering_implemented)
            self.assertFalse(loop.generated_output_mutation_implemented)
            self.assertEqual(loop.serialization_version, "specialist_agent_loop.v1")
            self.assertTrue(loop.metadata_only)

    def test_specialist_loop_source_registries_are_complete(self) -> None:
        registry = specialist_agent_loop_registry()
        loop_sources = tuple(
            dict.fromkeys(
                source
                for loop in registry.loops
                for source in loop.source_registries
            )
        )

        self.assertEqual(loop_sources, registry.source_registries)
        for source_registry in EXPECTED_SPECIALIST_LOOP_SOURCE_REGISTRIES:
            self.assertIn(source_registry, loop_sources)
        for loop in registry.loops:
            self.assertEqual(set(loop.source_registries), set(loop_sources))

    def test_specialist_loop_lookup_is_stable(self) -> None:
        loop = specialist_agent_loop_by_id("evaluation_specialist_agent_loop")
        missing = specialist_agent_loop_by_id("missing_loop")

        self.assertIsNone(missing)
        self.assertIsNotNone(loop)
        assert loop is not None
        self.assertEqual(loop.category, "evaluation")
        self.assertIn("critic_agent", loop.specialist_agent_ids)
        self.assertIn("refiner_agent", loop.specialist_agent_ids)
        self.assertEqual(loop.max_advisory_passes, 3)
        self.assertFalse(loop.loop_execution_implemented)

    def test_specialist_loop_registry_rejects_mismatched_metadata(self) -> None:
        registry = specialist_agent_loop_registry()
        mismatched_loop = registry.loops[0].model_copy(
            update={"loop_id": "other_loop"}
        )
        unknown_agent_loop = registry.loops[0].model_copy(
            update={"specialist_agent_ids": ("missing_agent",)}
        )

        with self.assertRaisesRegex(ValueError, "loop_ids must match"):
            SpecialistAgentLoopRegistry(
                loops=(mismatched_loop,) + registry.loops[1:],
                loop_ids=registry.loop_ids,
                categories=registry.categories,
                source_registries=registry.source_registries,
                agent_ids=registry.agent_ids,
                condition_ids=registry.condition_ids,
                backbone_node_ids=registry.backbone_node_ids,
                loop_count=registry.loop_count,
            )

        with self.assertRaisesRegex(ValueError, "loop agents must be known"):
            SpecialistAgentLoopRegistry(
                loops=(unknown_agent_loop,) + registry.loops[1:],
                loop_ids=registry.loop_ids,
                categories=registry.categories,
                source_registries=registry.source_registries,
                agent_ids=registry.agent_ids,
                condition_ids=registry.condition_ids,
                backbone_node_ids=registry.backbone_node_ids,
                loop_count=registry.loop_count,
            )

    def test_specialist_loops_do_not_declare_active_execution(self) -> None:
        registry = specialist_agent_loop_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for loop in registry.loops
                    for field in (
                        loop.loop_id,
                        loop.authority_boundary,
                        *loop.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "execute_agent",
            "run_loop",
            "route_provider",
            "trigger_retry",
            "modify_output",
        ):
            self.assertNotIn(forbidden_term, combined_text)


class EscalationGateRegistryTests(unittest.TestCase):
    def test_registry_declares_passive_escalation_gates(self) -> None:
        registry = escalation_gate_registry()

        self.assertEqual(registry.role, "escalation_gate_registry")
        self.assertEqual(
            registry.serialization_version,
            "escalation_gate_registry.v1",
        )
        self.assertEqual(registry.gate_ids, EXPECTED_ESCALATION_GATE_IDS)
        self.assertEqual(registry.gate_kinds, EXPECTED_ESCALATION_GATE_KINDS)
        self.assertEqual(
            registry.source_registries,
            EXPECTED_ESCALATION_GATE_SOURCE_REGISTRIES,
        )
        self.assertEqual(
            registry.condition_ids,
            conditional_multi_agent_escalation_registry().condition_ids,
        )
        self.assertEqual(registry.loop_ids, specialist_agent_loop_registry().loop_ids)
        self.assertEqual(registry.gate_count, 5)
        self.assertIn("does not evaluate gates", registry.authority_boundary)
        self.assertFalse(registry.gate_evaluation_implemented)
        self.assertFalse(registry.escalation_approval_implemented)
        self.assertFalse(registry.agent_invocation_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertTrue(registry.metadata_only)

    def test_gates_reference_known_conditions_and_loops(self) -> None:
        registry = escalation_gate_registry()
        known_conditions = set(conditional_multi_agent_escalation_registry().condition_ids)
        known_loops = set(specialist_agent_loop_registry().loop_ids)

        for gate in registry.gates:
            dumped = gate.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_ESCALATION_GATE_FIELDS)
            self.assertEqual(
                gate.source_registries,
                EXPECTED_ESCALATION_GATE_SOURCE_REGISTRIES,
            )
            self.assertTrue(set(gate.source_condition_ids).issubset(known_conditions))
            self.assertTrue(set(gate.source_loop_ids).issubset(known_loops))
            self.assertTrue(gate.required_passive_inputs)
            self.assertTrue(gate.advisory_decision_outputs)
            self.assertIn("gate_evaluation", gate.blocked_runtime_behaviors)
            self.assertIn("escalation_approval", gate.blocked_runtime_behaviors)
            self.assertIn("agent_invocation", gate.blocked_runtime_behaviors)
            self.assertFalse(gate.gate_evaluation_implemented)
            self.assertFalse(gate.escalation_approval_implemented)
            self.assertFalse(gate.agent_invocation_implemented)
            self.assertFalse(gate.workflow_control_implemented)
            self.assertFalse(gate.retry_triggering_implemented)
            self.assertFalse(gate.generated_output_mutation_implemented)
            self.assertEqual(gate.serialization_version, "escalation_gate.v1")
            self.assertTrue(gate.metadata_only)

    def test_gate_source_registries_are_complete(self) -> None:
        registry = escalation_gate_registry()
        gate_sources = tuple(
            dict.fromkeys(
                source
                for gate in registry.gates
                for source in gate.source_registries
            )
        )

        self.assertEqual(gate_sources, registry.source_registries)
        for source_registry in EXPECTED_ESCALATION_GATE_SOURCE_REGISTRIES:
            self.assertIn(source_registry, gate_sources)
        for gate in registry.gates:
            self.assertEqual(set(gate.source_registries), set(gate_sources))

    def test_gate_lookup_is_stable(self) -> None:
        gate = escalation_gate_by_id("specialist_loop_boundary_gate")
        missing = escalation_gate_by_id("missing_gate")

        self.assertIsNone(missing)
        self.assertIsNotNone(gate)
        assert gate is not None
        self.assertEqual(gate.gate_kind, "specialist_loop_boundary")
        self.assertEqual(gate.source_loop_ids, specialist_agent_loop_registry().loop_ids)
        self.assertFalse(gate.gate_evaluation_implemented)

    def test_gate_registry_rejects_mismatched_metadata(self) -> None:
        registry = escalation_gate_registry()
        mismatched_gate = registry.gates[0].model_copy(
            update={"gate_id": "other_gate"}
        )
        unknown_loop_gate = registry.gates[0].model_copy(
            update={"source_loop_ids": ("missing_loop",)}
        )

        with self.assertRaisesRegex(ValueError, "gate_ids must match"):
            EscalationGateRegistry(
                gates=(mismatched_gate,) + registry.gates[1:],
                gate_ids=registry.gate_ids,
                gate_kinds=registry.gate_kinds,
                source_registries=registry.source_registries,
                condition_ids=registry.condition_ids,
                loop_ids=registry.loop_ids,
                gate_count=registry.gate_count,
            )

        with self.assertRaisesRegex(ValueError, "gate loops must be known"):
            EscalationGateRegistry(
                gates=(unknown_loop_gate,) + registry.gates[1:],
                gate_ids=registry.gate_ids,
                gate_kinds=registry.gate_kinds,
                source_registries=registry.source_registries,
                condition_ids=registry.condition_ids,
                loop_ids=registry.loop_ids,
                gate_count=registry.gate_count,
            )

    def test_escalation_gates_do_not_declare_active_execution(self) -> None:
        registry = escalation_gate_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for gate in registry.gates
                    for field in (
                        gate.gate_id,
                        gate.authority_boundary,
                        *gate.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "execute_agent",
            "approve_escalation",
            "route_provider",
            "trigger_retry",
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
