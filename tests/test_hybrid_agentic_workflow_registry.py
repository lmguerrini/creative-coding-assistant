import unittest

from creative_coding_assistant.orchestration import (
    ASSISTANT_WORKFLOW_NODE_ORDER,
    AgentConfidenceFusionRegistry,
    ConditionalMultiAgentEscalationRegistry,
    CreativeExplorationBudgetRegistry,
    CreativeEscalationPolicyRegistry,
    DecisionProvenanceRegistry,
    EscalationGateRegistry,
    EscalationTraceRegistry,
    HybridAgentDebateLoopRegistry,
    HybridAgentVotingRegistry,
    ReflectionEscalationRegistry,
    SpecialistAgentLoopRegistry,
    V3BackboneModeRegistry,
    agent_contract_registry,
    agent_capability_registry,
    agent_confidence_fusion_profile_by_id,
    agent_confidence_fusion_registry,
    agent_escalation_signal_registry,
    conditional_multi_agent_escalation_condition_by_id,
    conditional_multi_agent_escalation_registry,
    creative_exploration_budget_profile_by_id,
    creative_exploration_budget_registry,
    creative_escalation_policy_by_id,
    creative_escalation_policy_registry,
    decision_provenance_profile_by_id,
    decision_provenance_registry,
    escalation_gate_by_id,
    escalation_gate_registry,
    escalation_trace_profile_by_id,
    escalation_trace_registry,
    escalation_policy_registry,
    hybrid_agent_debate_loop_by_id,
    hybrid_agent_debate_loop_registry,
    hybrid_agent_voting_profile_by_id,
    hybrid_agent_voting_registry,
    hybrid_agentic_workflow_registry,
    hybrid_agentic_workflow_stage_by_id,
    reflection_escalation_profile_by_id,
    reflection_escalation_registry,
    specialist_agent_loop_by_id,
    specialist_agent_loop_registry,
    v3_backbone_mode_profile_by_node_id,
    v3_backbone_mode_registry,
    workstation_engine_contracts,
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
EXPECTED_CREATIVE_POLICY_IDS = (
    "concept_ambiguity_creative_escalation_policy",
    "aesthetic_risk_creative_escalation_policy",
    "runtime_fit_creative_escalation_policy",
    "quality_uncertainty_creative_escalation_policy",
    "terminal_synthesis_creative_escalation_policy",
)
EXPECTED_CREATIVE_POLICY_CATEGORIES = (
    "concept",
    "aesthetic",
    "runtime",
    "quality",
    "synthesis",
)
EXPECTED_CREATIVE_POLICY_SOURCE_REGISTRIES = (
    "escalation_gate_registry",
    "specialist_agent_loop_registry",
    "conditional_multi_agent_escalation_registry",
    "artifact_engine_contract_registry",
    "evaluation_engine_contract_registry",
    "hybrid_agentic_workflow_registry",
)
REQUIRED_CREATIVE_POLICY_FIELDS = {
    "policy_id",
    "policy_name",
    "category",
    "source_gate_ids",
    "source_loop_ids",
    "source_registries",
    "creative_signal_sources",
    "advisory_policy_outputs",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "creative_policy_evaluation_implemented",
    "escalation_approval_implemented",
    "gate_evaluation_implemented",
    "agent_invocation_implemented",
    "workflow_control_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "metadata_only",
}
EXPECTED_REFLECTION_PROFILE_IDS = (
    "reflection_none_escalation_profile",
    "reflection_low_escalation_profile",
    "reflection_medium_escalation_profile",
    "reflection_high_escalation_profile",
    "reflection_critical_escalation_profile",
)
EXPECTED_REFLECTION_POSTURES = ("none", "low", "medium", "high", "critical")
EXPECTED_REFLECTION_SOURCE_REGISTRIES = (
    "reflection_loop_engine",
    "creative_escalation_policy_registry",
    "escalation_gate_registry",
    "evaluation_engine_contract_registry",
    "hybrid_agentic_workflow_registry",
)
REQUIRED_REFLECTION_PROFILE_FIELDS = {
    "profile_id",
    "profile_name",
    "posture",
    "reflection_priority",
    "source_policy_ids",
    "source_gate_ids",
    "source_registries",
    "reflection_signal_sources",
    "advisory_outputs",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "reflection_execution_implemented",
    "refinement_triggering_implemented",
    "escalation_approval_implemented",
    "agent_invocation_implemented",
    "workflow_control_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "metadata_only",
}
EXPECTED_HYBRID_DEBATE_LOOP_IDS = (
    "hybrid_debate_loop::planning_execution_fit",
    "hybrid_debate_loop::style_aesthetic_alignment",
    "hybrid_debate_loop::curation_refinement_need",
    "hybrid_debate_loop::final_synthesis_readiness",
)
EXPECTED_HYBRID_DEBATE_TOPICS = (
    "planning_execution_fit",
    "style_aesthetic_alignment",
    "curation_refinement_need",
    "final_synthesis_readiness",
)
EXPECTED_HYBRID_DEBATE_SOURCE_REGISTRIES = (
    "agent_debate_registry",
    "reflection_escalation_registry",
    "creative_escalation_policy_registry",
    "specialist_agent_loop_registry",
    "hybrid_agentic_workflow_registry",
)
REQUIRED_HYBRID_DEBATE_LOOP_FIELDS = {
    "loop_id",
    "topic_id",
    "source_debate_topic_id",
    "source_reflection_profile_ids",
    "source_policy_ids",
    "source_specialist_loop_ids",
    "source_registries",
    "advisory_outputs",
    "max_advisory_rounds",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "debate_loop_execution_implemented",
    "agent_invocation_implemented",
    "retry_triggering_implemented",
    "workflow_control_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "metadata_only",
}
EXPECTED_HYBRID_VOTING_PROFILE_IDS = (
    "hybrid_agent_voting::planning_execution_fit",
    "hybrid_agent_voting::style_aesthetic_alignment",
    "hybrid_agent_voting::curation_refinement_need",
    "hybrid_agent_voting::final_synthesis_readiness",
)
EXPECTED_HYBRID_VOTING_SOURCE_REGISTRIES = (
    "consensus_builder_registry",
    "hybrid_agent_debate_loop_registry",
    "reflection_escalation_registry",
    "creative_escalation_policy_registry",
    "hybrid_agentic_workflow_registry",
)
REQUIRED_HYBRID_VOTING_FIELDS = {
    "voting_profile_id",
    "topic_id",
    "source_debate_loop_id",
    "consensus_voting_input_id",
    "source_reflection_profile_ids",
    "source_policy_ids",
    "source_registries",
    "voting_dimensions",
    "advisory_outputs",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "voting_execution_implemented",
    "final_answer_selection_implemented",
    "agent_invocation_implemented",
    "workflow_control_implemented",
    "retry_triggering_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "metadata_only",
}
EXPECTED_AGENT_CONFIDENCE_FUSION_PROFILE_IDS = (
    "agent_confidence_fusion::planning_execution_fit",
    "agent_confidence_fusion::style_aesthetic_alignment",
    "agent_confidence_fusion::curation_refinement_need",
    "agent_confidence_fusion::final_synthesis_readiness",
)
EXPECTED_AGENT_CONFIDENCE_FUSION_SOURCE_REGISTRIES = (
    "creative_confidence_engine",
    "hybrid_agent_voting_registry",
    "hybrid_agent_debate_loop_registry",
    "reflection_escalation_registry",
    "evaluation_engine_contract_registry",
    "hybrid_agentic_workflow_registry",
)
REQUIRED_AGENT_CONFIDENCE_FUSION_FIELDS = {
    "fusion_profile_id",
    "topic_id",
    "source_voting_profile_id",
    "source_debate_loop_id",
    "source_confidence_surface_id",
    "source_reflection_profile_ids",
    "source_registries",
    "confidence_signal_inputs",
    "fusion_dimensions",
    "advisory_outputs",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "confidence_fusion_implemented",
    "confidence_score_calculation_implemented",
    "vote_weighting_implemented",
    "final_answer_selection_implemented",
    "agent_invocation_implemented",
    "workflow_control_implemented",
    "retry_triggering_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "metadata_only",
}
EXPECTED_DECISION_PROVENANCE_PROFILE_IDS = (
    "decision_provenance::planning_execution_fit",
    "decision_provenance::style_aesthetic_alignment",
    "decision_provenance::curation_refinement_need",
    "decision_provenance::final_synthesis_readiness",
)
EXPECTED_DECISION_PROVENANCE_SOURCE_REGISTRIES = (
    "agent_confidence_fusion_registry",
    "hybrid_agent_voting_registry",
    "hybrid_agent_debate_loop_registry",
    "v3_backbone_mode_registry",
    "workstation_engine_contract_registry",
    "hybrid_agentic_workflow_registry",
)
REQUIRED_DECISION_PROVENANCE_FIELDS = {
    "provenance_profile_id",
    "topic_id",
    "source_confidence_fusion_profile_id",
    "source_voting_profile_id",
    "source_debate_loop_id",
    "source_backbone_node_ids",
    "source_workstation_surface_id",
    "source_registries",
    "provenance_dimensions",
    "advisory_outputs",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "provenance_recording_implemented",
    "decision_logging_implemented",
    "trace_emission_implemented",
    "memory_write_implemented",
    "decision_selection_implemented",
    "agent_invocation_implemented",
    "workflow_control_implemented",
    "retry_triggering_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "metadata_only",
}
EXPECTED_ESCALATION_TRACE_PROFILE_IDS = (
    "escalation_trace::planning_execution_fit",
    "escalation_trace::style_aesthetic_alignment",
    "escalation_trace::curation_refinement_need",
    "escalation_trace::final_synthesis_readiness",
)
EXPECTED_ESCALATION_TRACE_SOURCE_REGISTRIES = (
    "decision_provenance_registry",
    "conditional_multi_agent_escalation_registry",
    "escalation_gate_registry",
    "agent_escalation_signal_registry",
    "reflection_escalation_registry",
    "hybrid_agentic_workflow_registry",
)
REQUIRED_ESCALATION_TRACE_FIELDS = {
    "trace_profile_id",
    "topic_id",
    "source_provenance_profile_id",
    "source_condition_ids",
    "source_gate_ids",
    "source_escalation_signal_ids",
    "source_reflection_profile_ids",
    "source_registries",
    "trace_dimensions",
    "advisory_outputs",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "trace_capture_implemented",
    "trace_emission_implemented",
    "escalation_execution_implemented",
    "gate_evaluation_implemented",
    "memory_write_implemented",
    "agent_invocation_implemented",
    "workflow_control_implemented",
    "retry_triggering_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "metadata_only",
}
EXPECTED_CREATIVE_EXPLORATION_BUDGET_PROFILE_IDS = (
    "creative_exploration_budget::planning_execution_fit",
    "creative_exploration_budget::style_aesthetic_alignment",
    "creative_exploration_budget::curation_refinement_need",
    "creative_exploration_budget::final_synthesis_readiness",
)
EXPECTED_CREATIVE_EXPLORATION_BUDGET_POSTURES = (
    "moderate",
    "broad",
    "guarded",
    "narrow",
)
EXPECTED_CREATIVE_EXPLORATION_BUDGET_SOURCE_REGISTRIES = (
    "escalation_trace_registry",
    "decision_provenance_registry",
    "creative_planning_engine",
    "creative_constraints_engine",
    "creative_tradeoff_engine",
    "hybrid_agentic_workflow_registry",
)
REQUIRED_CREATIVE_EXPLORATION_BUDGET_FIELDS = {
    "budget_profile_id",
    "topic_id",
    "source_trace_profile_id",
    "source_provenance_profile_id",
    "source_escalation_signal_ids",
    "budget_posture",
    "max_advisory_variants",
    "max_advisory_refinement_passes",
    "cost_pressure_signal",
    "source_registries",
    "budget_dimensions",
    "advisory_outputs",
    "authority_boundary",
    "blocked_runtime_behaviors",
    "budget_enforcement_implemented",
    "variant_generation_implemented",
    "refinement_triggering_implemented",
    "cost_routing_implemented",
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


class CreativeEscalationPolicyRegistryTests(unittest.TestCase):
    def test_registry_declares_passive_creative_policies(self) -> None:
        registry = creative_escalation_policy_registry()

        self.assertEqual(registry.role, "creative_escalation_policy_registry")
        self.assertEqual(
            registry.serialization_version,
            "creative_escalation_policy_registry.v1",
        )
        self.assertEqual(registry.policy_ids, EXPECTED_CREATIVE_POLICY_IDS)
        self.assertEqual(registry.categories, EXPECTED_CREATIVE_POLICY_CATEGORIES)
        self.assertEqual(
            registry.source_registries,
            EXPECTED_CREATIVE_POLICY_SOURCE_REGISTRIES,
        )
        self.assertEqual(registry.gate_ids, escalation_gate_registry().gate_ids)
        self.assertEqual(registry.loop_ids, specialist_agent_loop_registry().loop_ids)
        self.assertEqual(registry.policy_count, 5)
        self.assertIn("does not evaluate creative policy", registry.authority_boundary)
        self.assertFalse(registry.creative_policy_evaluation_implemented)
        self.assertFalse(registry.escalation_approval_implemented)
        self.assertFalse(registry.gate_evaluation_implemented)
        self.assertFalse(registry.agent_invocation_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertTrue(registry.metadata_only)

    def test_creative_policies_reference_known_gates_and_loops(self) -> None:
        registry = creative_escalation_policy_registry()
        known_gates = set(escalation_gate_registry().gate_ids)
        known_loops = set(specialist_agent_loop_registry().loop_ids)

        for policy in registry.policies:
            dumped = policy.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_CREATIVE_POLICY_FIELDS)
            self.assertEqual(
                policy.source_registries,
                EXPECTED_CREATIVE_POLICY_SOURCE_REGISTRIES,
            )
            self.assertTrue(set(policy.source_gate_ids).issubset(known_gates))
            self.assertTrue(set(policy.source_loop_ids).issubset(known_loops))
            self.assertTrue(policy.creative_signal_sources)
            self.assertTrue(policy.advisory_policy_outputs)
            self.assertIn(
                "creative_policy_evaluation",
                policy.blocked_runtime_behaviors,
            )
            self.assertIn("escalation_approval", policy.blocked_runtime_behaviors)
            self.assertIn("agent_invocation", policy.blocked_runtime_behaviors)
            self.assertFalse(policy.creative_policy_evaluation_implemented)
            self.assertFalse(policy.escalation_approval_implemented)
            self.assertFalse(policy.gate_evaluation_implemented)
            self.assertFalse(policy.agent_invocation_implemented)
            self.assertFalse(policy.workflow_control_implemented)
            self.assertFalse(policy.generated_output_mutation_implemented)
            self.assertEqual(
                policy.serialization_version,
                "creative_escalation_policy_rule.v1",
            )
            self.assertTrue(policy.metadata_only)

    def test_creative_policy_source_registries_are_complete(self) -> None:
        registry = creative_escalation_policy_registry()
        policy_sources = tuple(
            dict.fromkeys(
                source
                for policy in registry.policies
                for source in policy.source_registries
            )
        )

        self.assertEqual(policy_sources, registry.source_registries)
        for source_registry in EXPECTED_CREATIVE_POLICY_SOURCE_REGISTRIES:
            self.assertIn(source_registry, policy_sources)
        for policy in registry.policies:
            self.assertEqual(set(policy.source_registries), set(policy_sources))

    def test_creative_policy_lookup_is_stable(self) -> None:
        policy = creative_escalation_policy_by_id(
            "quality_uncertainty_creative_escalation_policy"
        )
        missing = creative_escalation_policy_by_id("missing_policy")

        self.assertIsNone(missing)
        self.assertIsNotNone(policy)
        assert policy is not None
        self.assertEqual(policy.category, "quality")
        self.assertIn("evaluation_specialist_agent_loop", policy.source_loop_ids)
        self.assertFalse(policy.creative_policy_evaluation_implemented)

    def test_creative_policy_registry_rejects_mismatched_metadata(self) -> None:
        registry = creative_escalation_policy_registry()
        mismatched_policy = registry.policies[0].model_copy(
            update={"policy_id": "other_policy"}
        )
        unknown_gate_policy = registry.policies[0].model_copy(
            update={"source_gate_ids": ("missing_gate",)}
        )

        with self.assertRaisesRegex(ValueError, "policy_ids must match"):
            CreativeEscalationPolicyRegistry(
                policies=(mismatched_policy,) + registry.policies[1:],
                policy_ids=registry.policy_ids,
                categories=registry.categories,
                source_registries=registry.source_registries,
                gate_ids=registry.gate_ids,
                loop_ids=registry.loop_ids,
                policy_count=registry.policy_count,
            )

        with self.assertRaisesRegex(ValueError, "policy gates must be known"):
            CreativeEscalationPolicyRegistry(
                policies=(unknown_gate_policy,) + registry.policies[1:],
                policy_ids=registry.policy_ids,
                categories=registry.categories,
                source_registries=registry.source_registries,
                gate_ids=registry.gate_ids,
                loop_ids=registry.loop_ids,
                policy_count=registry.policy_count,
            )

    def test_creative_policies_do_not_declare_active_execution(self) -> None:
        registry = creative_escalation_policy_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for policy in registry.policies
                    for field in (
                        policy.policy_id,
                        policy.authority_boundary,
                        *policy.blocked_runtime_behaviors,
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


class ReflectionEscalationRegistryTests(unittest.TestCase):
    def test_registry_declares_passive_reflection_postures(self) -> None:
        registry = reflection_escalation_registry()

        self.assertEqual(registry.role, "reflection_escalation_registry")
        self.assertEqual(
            registry.serialization_version,
            "reflection_escalation_registry.v1",
        )
        self.assertEqual(registry.profile_ids, EXPECTED_REFLECTION_PROFILE_IDS)
        self.assertEqual(registry.postures, EXPECTED_REFLECTION_POSTURES)
        self.assertEqual(registry.source_registries, EXPECTED_REFLECTION_SOURCE_REGISTRIES)
        self.assertEqual(
            registry.policy_ids,
            creative_escalation_policy_registry().policy_ids,
        )
        self.assertEqual(registry.gate_ids, escalation_gate_registry().gate_ids)
        self.assertEqual(registry.profile_count, 5)
        self.assertIn("does not run reflection", registry.authority_boundary)
        self.assertFalse(registry.reflection_execution_implemented)
        self.assertFalse(registry.refinement_triggering_implemented)
        self.assertFalse(registry.escalation_approval_implemented)
        self.assertFalse(registry.agent_invocation_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertTrue(registry.metadata_only)

    def test_reflection_profiles_reference_known_policies_and_gates(self) -> None:
        registry = reflection_escalation_registry()
        known_policies = set(creative_escalation_policy_registry().policy_ids)
        known_gates = set(escalation_gate_registry().gate_ids)

        for profile in registry.profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_REFLECTION_PROFILE_FIELDS)
            self.assertEqual(profile.source_registries, EXPECTED_REFLECTION_SOURCE_REGISTRIES)
            self.assertEqual(profile.posture, profile.reflection_priority)
            self.assertTrue(set(profile.source_policy_ids).issubset(known_policies))
            self.assertTrue(set(profile.source_gate_ids).issubset(known_gates))
            self.assertTrue(profile.reflection_signal_sources)
            self.assertTrue(profile.advisory_outputs)
            self.assertIn("reflection_execution", profile.blocked_runtime_behaviors)
            self.assertIn("refinement_triggering", profile.blocked_runtime_behaviors)
            self.assertFalse(profile.reflection_execution_implemented)
            self.assertFalse(profile.refinement_triggering_implemented)
            self.assertFalse(profile.escalation_approval_implemented)
            self.assertFalse(profile.agent_invocation_implemented)
            self.assertFalse(profile.workflow_control_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)
            self.assertEqual(
                profile.serialization_version,
                "reflection_escalation_profile.v1",
            )
            self.assertTrue(profile.metadata_only)

    def test_reflection_source_registries_are_complete(self) -> None:
        registry = reflection_escalation_registry()
        profile_sources = tuple(
            dict.fromkeys(
                source
                for profile in registry.profiles
                for source in profile.source_registries
            )
        )

        self.assertEqual(profile_sources, registry.source_registries)
        for source_registry in EXPECTED_REFLECTION_SOURCE_REGISTRIES:
            self.assertIn(source_registry, profile_sources)
        for profile in registry.profiles:
            self.assertEqual(set(profile.source_registries), set(profile_sources))

    def test_reflection_profile_lookup_is_stable(self) -> None:
        profile = reflection_escalation_profile_by_id(
            "reflection_critical_escalation_profile"
        )
        missing = reflection_escalation_profile_by_id("missing_profile")

        self.assertIsNone(missing)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.posture, "critical")
        self.assertIn("hitl_recommendation_required", profile.reflection_signal_sources)
        self.assertFalse(profile.refinement_triggering_implemented)

    def test_reflection_registry_rejects_mismatched_metadata(self) -> None:
        registry = reflection_escalation_registry()
        mismatched_profile = registry.profiles[0].model_copy(
            update={"profile_id": "other_profile"}
        )
        unknown_policy_profile = registry.profiles[0].model_copy(
            update={"source_policy_ids": ("missing_policy",)}
        )

        with self.assertRaisesRegex(ValueError, "profile_ids must match"):
            ReflectionEscalationRegistry(
                profiles=(mismatched_profile,) + registry.profiles[1:],
                profile_ids=registry.profile_ids,
                postures=registry.postures,
                source_registries=registry.source_registries,
                policy_ids=registry.policy_ids,
                gate_ids=registry.gate_ids,
                profile_count=registry.profile_count,
            )

        with self.assertRaisesRegex(ValueError, "reflection policies must be known"):
            ReflectionEscalationRegistry(
                profiles=(unknown_policy_profile,) + registry.profiles[1:],
                profile_ids=registry.profile_ids,
                postures=registry.postures,
                source_registries=registry.source_registries,
                policy_ids=registry.policy_ids,
                gate_ids=registry.gate_ids,
                profile_count=registry.profile_count,
            )

    def test_reflection_escalation_does_not_declare_active_execution(self) -> None:
        registry = reflection_escalation_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for profile in registry.profiles
                    for field in (
                        profile.profile_id,
                        profile.authority_boundary,
                        *profile.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "execute_agent",
            "run_reflection",
            "trigger_refinement",
            "route_provider",
            "modify_output",
        ):
            self.assertNotIn(forbidden_term, combined_text)


class HybridAgentDebateLoopRegistryTests(unittest.TestCase):
    def test_registry_declares_passive_hybrid_debate_loops(self) -> None:
        registry = hybrid_agent_debate_loop_registry()

        self.assertEqual(registry.role, "hybrid_agent_debate_loop_registry")
        self.assertEqual(
            registry.serialization_version,
            "hybrid_agent_debate_loop_registry.v1",
        )
        self.assertEqual(registry.loop_ids, EXPECTED_HYBRID_DEBATE_LOOP_IDS)
        self.assertEqual(registry.topic_ids, EXPECTED_HYBRID_DEBATE_TOPICS)
        self.assertEqual(
            registry.source_registries,
            EXPECTED_HYBRID_DEBATE_SOURCE_REGISTRIES,
        )
        self.assertEqual(
            registry.reflection_profile_ids,
            reflection_escalation_registry().profile_ids,
        )
        self.assertEqual(
            registry.policy_ids,
            creative_escalation_policy_registry().policy_ids,
        )
        self.assertEqual(
            registry.specialist_loop_ids,
            specialist_agent_loop_registry().loop_ids,
        )
        self.assertEqual(registry.loop_count, 4)
        self.assertIn("does not execute debate loops", registry.authority_boundary)
        self.assertFalse(registry.debate_loop_execution_implemented)
        self.assertFalse(registry.agent_invocation_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertTrue(registry.metadata_only)

    def test_hybrid_debate_loops_reference_known_sources(self) -> None:
        registry = hybrid_agent_debate_loop_registry()
        known_reflections = set(reflection_escalation_registry().profile_ids)
        known_policies = set(creative_escalation_policy_registry().policy_ids)
        known_loops = set(specialist_agent_loop_registry().loop_ids)

        for loop in registry.debate_loops:
            dumped = loop.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_HYBRID_DEBATE_LOOP_FIELDS)
            self.assertEqual(loop.topic_id, loop.source_debate_topic_id)
            self.assertEqual(loop.source_registries, EXPECTED_HYBRID_DEBATE_SOURCE_REGISTRIES)
            self.assertTrue(
                set(loop.source_reflection_profile_ids).issubset(known_reflections)
            )
            self.assertTrue(set(loop.source_policy_ids).issubset(known_policies))
            self.assertTrue(set(loop.source_specialist_loop_ids).issubset(known_loops))
            self.assertLessEqual(loop.max_advisory_rounds, 2)
            self.assertTrue(loop.advisory_outputs)
            self.assertIn("debate_loop_execution", loop.blocked_runtime_behaviors)
            self.assertFalse(loop.debate_loop_execution_implemented)
            self.assertFalse(loop.agent_invocation_implemented)
            self.assertFalse(loop.retry_triggering_implemented)
            self.assertFalse(loop.workflow_control_implemented)
            self.assertFalse(loop.generated_output_mutation_implemented)
            self.assertEqual(
                loop.serialization_version,
                "hybrid_agent_debate_loop_profile.v1",
            )
            self.assertTrue(loop.metadata_only)

    def test_hybrid_debate_source_registries_are_complete(self) -> None:
        registry = hybrid_agent_debate_loop_registry()
        loop_sources = tuple(
            dict.fromkeys(
                source
                for loop in registry.debate_loops
                for source in loop.source_registries
            )
        )

        self.assertEqual(loop_sources, registry.source_registries)
        for source_registry in EXPECTED_HYBRID_DEBATE_SOURCE_REGISTRIES:
            self.assertIn(source_registry, loop_sources)
        for loop in registry.debate_loops:
            self.assertEqual(set(loop.source_registries), set(loop_sources))

    def test_hybrid_debate_loop_lookup_is_stable(self) -> None:
        loop = hybrid_agent_debate_loop_by_id(
            "hybrid_debate_loop::curation_refinement_need"
        )
        missing = hybrid_agent_debate_loop_by_id("missing_loop")

        self.assertIsNone(missing)
        self.assertIsNotNone(loop)
        assert loop is not None
        self.assertEqual(loop.topic_id, "curation_refinement_need")
        self.assertIn("evaluation_specialist_agent_loop", loop.source_specialist_loop_ids)
        self.assertFalse(loop.debate_loop_execution_implemented)

    def test_hybrid_debate_registry_rejects_mismatched_metadata(self) -> None:
        registry = hybrid_agent_debate_loop_registry()
        mismatched_loop = registry.debate_loops[0].model_copy(
            update={"loop_id": "other_loop"}
        )
        unknown_reflection_loop = registry.debate_loops[0].model_copy(
            update={"source_reflection_profile_ids": ("missing_profile",)}
        )

        with self.assertRaisesRegex(ValueError, "loop_ids must match"):
            HybridAgentDebateLoopRegistry(
                debate_loops=(mismatched_loop,) + registry.debate_loops[1:],
                loop_ids=registry.loop_ids,
                topic_ids=registry.topic_ids,
                source_registries=registry.source_registries,
                reflection_profile_ids=registry.reflection_profile_ids,
                policy_ids=registry.policy_ids,
                specialist_loop_ids=registry.specialist_loop_ids,
                loop_count=registry.loop_count,
            )

        with self.assertRaisesRegex(ValueError, "debate reflections must be known"):
            HybridAgentDebateLoopRegistry(
                debate_loops=(unknown_reflection_loop,) + registry.debate_loops[1:],
                loop_ids=registry.loop_ids,
                topic_ids=registry.topic_ids,
                source_registries=registry.source_registries,
                reflection_profile_ids=registry.reflection_profile_ids,
                policy_ids=registry.policy_ids,
                specialist_loop_ids=registry.specialist_loop_ids,
                loop_count=registry.loop_count,
            )

    def test_hybrid_debate_loops_do_not_declare_active_execution(self) -> None:
        registry = hybrid_agent_debate_loop_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for loop in registry.debate_loops
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
            "execute_debate_loop",
            "trigger_retry",
            "route_provider",
            "modify_output",
        ):
            self.assertNotIn(forbidden_term, combined_text)


class HybridAgentVotingRegistryTests(unittest.TestCase):
    def test_registry_declares_passive_hybrid_voting_profiles(self) -> None:
        registry = hybrid_agent_voting_registry()

        self.assertEqual(registry.role, "hybrid_agent_voting_registry")
        self.assertEqual(
            registry.serialization_version,
            "hybrid_agent_voting_registry.v1",
        )
        self.assertEqual(registry.voting_profile_ids, EXPECTED_HYBRID_VOTING_PROFILE_IDS)
        self.assertEqual(registry.topic_ids, EXPECTED_HYBRID_DEBATE_TOPICS)
        self.assertEqual(
            registry.source_registries,
            EXPECTED_HYBRID_VOTING_SOURCE_REGISTRIES,
        )
        self.assertEqual(registry.debate_loop_ids, hybrid_agent_debate_loop_registry().loop_ids)
        self.assertEqual(
            registry.reflection_profile_ids,
            reflection_escalation_registry().profile_ids,
        )
        self.assertEqual(
            registry.policy_ids,
            creative_escalation_policy_registry().policy_ids,
        )
        self.assertEqual(registry.profile_count, 4)
        self.assertIn("does not execute voting", registry.authority_boundary)
        self.assertFalse(registry.voting_execution_implemented)
        self.assertFalse(registry.final_answer_selection_implemented)
        self.assertFalse(registry.agent_invocation_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertTrue(registry.metadata_only)

    def test_hybrid_voting_profiles_reference_known_sources(self) -> None:
        registry = hybrid_agent_voting_registry()
        known_debates = set(hybrid_agent_debate_loop_registry().loop_ids)
        known_reflections = set(reflection_escalation_registry().profile_ids)
        known_policies = set(creative_escalation_policy_registry().policy_ids)

        for profile in registry.voting_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_HYBRID_VOTING_FIELDS)
            self.assertEqual(
                profile.source_registries,
                EXPECTED_HYBRID_VOTING_SOURCE_REGISTRIES,
            )
            self.assertIn(profile.source_debate_loop_id, known_debates)
            self.assertTrue(
                set(profile.source_reflection_profile_ids).issubset(known_reflections)
            )
            self.assertTrue(set(profile.source_policy_ids).issubset(known_policies))
            self.assertTrue(profile.voting_dimensions)
            self.assertTrue(profile.advisory_outputs)
            self.assertIn("voting_execution", profile.blocked_runtime_behaviors)
            self.assertFalse(profile.voting_execution_implemented)
            self.assertFalse(profile.final_answer_selection_implemented)
            self.assertFalse(profile.agent_invocation_implemented)
            self.assertFalse(profile.workflow_control_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)
            self.assertEqual(
                profile.serialization_version,
                "hybrid_agent_voting_profile.v1",
            )
            self.assertTrue(profile.metadata_only)

    def test_hybrid_voting_source_registries_are_complete(self) -> None:
        registry = hybrid_agent_voting_registry()
        profile_sources = tuple(
            dict.fromkeys(
                source
                for profile in registry.voting_profiles
                for source in profile.source_registries
            )
        )

        self.assertEqual(profile_sources, registry.source_registries)
        for source_registry in EXPECTED_HYBRID_VOTING_SOURCE_REGISTRIES:
            self.assertIn(source_registry, profile_sources)
        for profile in registry.voting_profiles:
            self.assertEqual(set(profile.source_registries), set(profile_sources))

    def test_hybrid_voting_lookup_is_stable(self) -> None:
        profile = hybrid_agent_voting_profile_by_id(
            "hybrid_agent_voting::final_synthesis_readiness"
        )
        missing = hybrid_agent_voting_profile_by_id("missing_vote")

        self.assertIsNone(missing)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.topic_id, "final_synthesis_readiness")
        self.assertIn("synthesis_vote_placeholder", profile.advisory_outputs)
        self.assertFalse(profile.voting_execution_implemented)

    def test_hybrid_voting_registry_rejects_mismatched_metadata(self) -> None:
        registry = hybrid_agent_voting_registry()
        mismatched_profile = registry.voting_profiles[0].model_copy(
            update={"voting_profile_id": "other_vote"}
        )
        unknown_debate_profile = registry.voting_profiles[0].model_copy(
            update={"source_debate_loop_id": "missing_debate"}
        )

        with self.assertRaisesRegex(ValueError, "voting_profile_ids must match"):
            HybridAgentVotingRegistry(
                voting_profiles=(mismatched_profile,) + registry.voting_profiles[1:],
                voting_profile_ids=registry.voting_profile_ids,
                topic_ids=registry.topic_ids,
                source_registries=registry.source_registries,
                debate_loop_ids=registry.debate_loop_ids,
                reflection_profile_ids=registry.reflection_profile_ids,
                policy_ids=registry.policy_ids,
                profile_count=registry.profile_count,
            )

        with self.assertRaisesRegex(ValueError, "voting debate loops must be known"):
            HybridAgentVotingRegistry(
                voting_profiles=(unknown_debate_profile,)
                + registry.voting_profiles[1:],
                voting_profile_ids=registry.voting_profile_ids,
                topic_ids=registry.topic_ids,
                source_registries=registry.source_registries,
                debate_loop_ids=registry.debate_loop_ids,
                reflection_profile_ids=registry.reflection_profile_ids,
                policy_ids=registry.policy_ids,
                profile_count=registry.profile_count,
            )

    def test_hybrid_voting_does_not_declare_active_execution(self) -> None:
        registry = hybrid_agent_voting_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for profile in registry.voting_profiles
                    for field in (
                        profile.voting_profile_id,
                        profile.authority_boundary,
                        *profile.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "execute_vote",
            "select_final_answer",
            "execute_agent",
            "trigger_retry",
            "modify_output",
        ):
            self.assertNotIn(forbidden_term, combined_text)


class AgentConfidenceFusionRegistryTests(unittest.TestCase):
    def test_registry_declares_passive_confidence_fusion_profiles(self) -> None:
        registry = agent_confidence_fusion_registry()

        self.assertEqual(registry.role, "agent_confidence_fusion_registry")
        self.assertEqual(
            registry.serialization_version,
            "agent_confidence_fusion_registry.v1",
        )
        self.assertEqual(
            registry.fusion_profile_ids,
            EXPECTED_AGENT_CONFIDENCE_FUSION_PROFILE_IDS,
        )
        self.assertEqual(registry.topic_ids, EXPECTED_HYBRID_DEBATE_TOPICS)
        self.assertEqual(
            registry.source_registries,
            EXPECTED_AGENT_CONFIDENCE_FUSION_SOURCE_REGISTRIES,
        )
        self.assertEqual(
            registry.voting_profile_ids,
            hybrid_agent_voting_registry().voting_profile_ids,
        )
        self.assertEqual(
            registry.debate_loop_ids,
            hybrid_agent_debate_loop_registry().loop_ids,
        )
        self.assertEqual(
            registry.reflection_profile_ids,
            reflection_escalation_registry().profile_ids,
        )
        self.assertEqual(registry.confidence_surface_ids, ("creative_confidence_engine",))
        self.assertEqual(registry.profile_count, 4)
        self.assertIn("does not calculate confidence scores", registry.authority_boundary)
        self.assertFalse(registry.confidence_fusion_implemented)
        self.assertFalse(registry.confidence_score_calculation_implemented)
        self.assertFalse(registry.vote_weighting_implemented)
        self.assertFalse(registry.final_answer_selection_implemented)
        self.assertFalse(registry.agent_invocation_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertTrue(registry.metadata_only)

    def test_confidence_fusion_profiles_reference_known_sources(self) -> None:
        registry = agent_confidence_fusion_registry()
        known_votes = set(hybrid_agent_voting_registry().voting_profile_ids)
        known_debates = set(hybrid_agent_debate_loop_registry().loop_ids)
        known_reflections = set(reflection_escalation_registry().profile_ids)
        known_confidence_surfaces = set(registry.confidence_surface_ids)

        for profile in registry.fusion_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_AGENT_CONFIDENCE_FUSION_FIELDS)
            self.assertEqual(
                profile.source_registries,
                EXPECTED_AGENT_CONFIDENCE_FUSION_SOURCE_REGISTRIES,
            )
            self.assertIn(profile.source_voting_profile_id, known_votes)
            self.assertIn(profile.source_debate_loop_id, known_debates)
            self.assertIn(
                profile.source_confidence_surface_id,
                known_confidence_surfaces,
            )
            self.assertTrue(
                set(profile.source_reflection_profile_ids).issubset(known_reflections)
            )
            self.assertTrue(profile.confidence_signal_inputs)
            self.assertTrue(profile.fusion_dimensions)
            self.assertTrue(profile.advisory_outputs)
            self.assertIn(
                "confidence_fusion_execution",
                profile.blocked_runtime_behaviors,
            )
            self.assertFalse(profile.confidence_fusion_implemented)
            self.assertFalse(profile.confidence_score_calculation_implemented)
            self.assertFalse(profile.vote_weighting_implemented)
            self.assertFalse(profile.final_answer_selection_implemented)
            self.assertFalse(profile.agent_invocation_implemented)
            self.assertFalse(profile.workflow_control_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)
            self.assertEqual(
                profile.serialization_version,
                "agent_confidence_fusion_profile.v1",
            )
            self.assertTrue(profile.metadata_only)

    def test_confidence_fusion_source_registries_are_complete(self) -> None:
        registry = agent_confidence_fusion_registry()
        profile_sources = tuple(
            dict.fromkeys(
                source
                for profile in registry.fusion_profiles
                for source in profile.source_registries
            )
        )

        self.assertEqual(profile_sources, registry.source_registries)
        for source_registry in EXPECTED_AGENT_CONFIDENCE_FUSION_SOURCE_REGISTRIES:
            self.assertIn(source_registry, profile_sources)
        for profile in registry.fusion_profiles:
            self.assertEqual(set(profile.source_registries), set(profile_sources))

    def test_confidence_fusion_lookup_is_stable(self) -> None:
        profile = agent_confidence_fusion_profile_by_id(
            "agent_confidence_fusion::curation_refinement_need"
        )
        missing = agent_confidence_fusion_profile_by_id("missing_fusion")

        self.assertIsNone(missing)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.topic_id, "curation_refinement_need")
        self.assertIn(
            "curation_confidence_fusion_placeholder",
            profile.advisory_outputs,
        )
        self.assertFalse(profile.confidence_fusion_implemented)

    def test_confidence_fusion_registry_rejects_mismatched_metadata(self) -> None:
        registry = agent_confidence_fusion_registry()
        mismatched_profile = registry.fusion_profiles[0].model_copy(
            update={"fusion_profile_id": "other_fusion"}
        )
        unknown_vote_profile = registry.fusion_profiles[0].model_copy(
            update={"source_voting_profile_id": "missing_vote"}
        )

        with self.assertRaisesRegex(ValueError, "fusion_profile_ids must match"):
            AgentConfidenceFusionRegistry(
                fusion_profiles=(mismatched_profile,) + registry.fusion_profiles[1:],
                fusion_profile_ids=registry.fusion_profile_ids,
                topic_ids=registry.topic_ids,
                source_registries=registry.source_registries,
                voting_profile_ids=registry.voting_profile_ids,
                debate_loop_ids=registry.debate_loop_ids,
                reflection_profile_ids=registry.reflection_profile_ids,
                confidence_surface_ids=registry.confidence_surface_ids,
                profile_count=registry.profile_count,
            )

        with self.assertRaisesRegex(ValueError, "fusion voting profiles must be known"):
            AgentConfidenceFusionRegistry(
                fusion_profiles=(unknown_vote_profile,) + registry.fusion_profiles[1:],
                fusion_profile_ids=registry.fusion_profile_ids,
                topic_ids=registry.topic_ids,
                source_registries=registry.source_registries,
                voting_profile_ids=registry.voting_profile_ids,
                debate_loop_ids=registry.debate_loop_ids,
                reflection_profile_ids=registry.reflection_profile_ids,
                confidence_surface_ids=registry.confidence_surface_ids,
                profile_count=registry.profile_count,
            )

    def test_confidence_fusion_does_not_declare_active_execution(self) -> None:
        registry = agent_confidence_fusion_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for profile in registry.fusion_profiles
                    for field in (
                        profile.fusion_profile_id,
                        profile.authority_boundary,
                        *profile.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "run_fusion",
            "execute_fusion",
            "execute_agent",
            "trigger_retry",
            "modify_output",
        ):
            self.assertNotIn(forbidden_term, combined_text)


class DecisionProvenanceRegistryTests(unittest.TestCase):
    def test_registry_declares_passive_decision_provenance_profiles(self) -> None:
        registry = decision_provenance_registry()

        self.assertEqual(registry.role, "decision_provenance_registry")
        self.assertEqual(
            registry.serialization_version,
            "decision_provenance_registry.v1",
        )
        self.assertEqual(
            registry.provenance_profile_ids,
            EXPECTED_DECISION_PROVENANCE_PROFILE_IDS,
        )
        self.assertEqual(registry.topic_ids, EXPECTED_HYBRID_DEBATE_TOPICS)
        self.assertEqual(
            registry.source_registries,
            EXPECTED_DECISION_PROVENANCE_SOURCE_REGISTRIES,
        )
        self.assertEqual(
            registry.confidence_fusion_profile_ids,
            agent_confidence_fusion_registry().fusion_profile_ids,
        )
        self.assertEqual(
            registry.voting_profile_ids,
            hybrid_agent_voting_registry().voting_profile_ids,
        )
        self.assertEqual(
            registry.debate_loop_ids,
            hybrid_agent_debate_loop_registry().loop_ids,
        )
        self.assertEqual(registry.backbone_node_ids, v3_backbone_mode_registry().node_ids)
        self.assertEqual(
            registry.workstation_surface_ids,
            workstation_engine_contracts().surface_ids,
        )
        self.assertEqual(registry.profile_count, 4)
        self.assertIn("does not record provenance", registry.authority_boundary)
        self.assertFalse(registry.provenance_recording_implemented)
        self.assertFalse(registry.decision_logging_implemented)
        self.assertFalse(registry.trace_emission_implemented)
        self.assertFalse(registry.memory_write_implemented)
        self.assertFalse(registry.decision_selection_implemented)
        self.assertFalse(registry.agent_invocation_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertTrue(registry.metadata_only)

    def test_decision_provenance_profiles_reference_known_sources(self) -> None:
        registry = decision_provenance_registry()
        known_fusion = set(agent_confidence_fusion_registry().fusion_profile_ids)
        known_votes = set(hybrid_agent_voting_registry().voting_profile_ids)
        known_debates = set(hybrid_agent_debate_loop_registry().loop_ids)
        known_nodes = set(v3_backbone_mode_registry().node_ids)
        known_surfaces = set(workstation_engine_contracts().surface_ids)

        for profile in registry.provenance_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_DECISION_PROVENANCE_FIELDS)
            self.assertEqual(
                profile.source_registries,
                EXPECTED_DECISION_PROVENANCE_SOURCE_REGISTRIES,
            )
            self.assertIn(profile.source_confidence_fusion_profile_id, known_fusion)
            self.assertIn(profile.source_voting_profile_id, known_votes)
            self.assertIn(profile.source_debate_loop_id, known_debates)
            self.assertTrue(set(profile.source_backbone_node_ids).issubset(known_nodes))
            self.assertIn(profile.source_workstation_surface_id, known_surfaces)
            self.assertEqual(profile.source_workstation_surface_id, "provenance_engine")
            self.assertTrue(profile.provenance_dimensions)
            self.assertTrue(profile.advisory_outputs)
            self.assertIn("provenance_recording", profile.blocked_runtime_behaviors)
            self.assertIn("trace_emission", profile.blocked_runtime_behaviors)
            self.assertFalse(profile.provenance_recording_implemented)
            self.assertFalse(profile.decision_logging_implemented)
            self.assertFalse(profile.trace_emission_implemented)
            self.assertFalse(profile.memory_write_implemented)
            self.assertFalse(profile.decision_selection_implemented)
            self.assertFalse(profile.agent_invocation_implemented)
            self.assertFalse(profile.workflow_control_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)
            self.assertEqual(
                profile.serialization_version,
                "decision_provenance_profile.v1",
            )
            self.assertTrue(profile.metadata_only)

    def test_decision_provenance_source_registries_are_complete(self) -> None:
        registry = decision_provenance_registry()
        profile_sources = tuple(
            dict.fromkeys(
                source
                for profile in registry.provenance_profiles
                for source in profile.source_registries
            )
        )

        self.assertEqual(profile_sources, registry.source_registries)
        for source_registry in EXPECTED_DECISION_PROVENANCE_SOURCE_REGISTRIES:
            self.assertIn(source_registry, profile_sources)
        for profile in registry.provenance_profiles:
            self.assertEqual(set(profile.source_registries), set(profile_sources))

    def test_decision_provenance_lookup_is_stable(self) -> None:
        profile = decision_provenance_profile_by_id(
            "decision_provenance::style_aesthetic_alignment"
        )
        missing = decision_provenance_profile_by_id("missing_provenance")

        self.assertIsNone(missing)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.topic_id, "style_aesthetic_alignment")
        self.assertIn("generation", profile.source_backbone_node_ids)
        self.assertIn(
            "style_decision_lineage_placeholder",
            profile.advisory_outputs,
        )
        self.assertFalse(profile.provenance_recording_implemented)

    def test_decision_provenance_registry_rejects_mismatched_metadata(self) -> None:
        registry = decision_provenance_registry()
        mismatched_profile = registry.provenance_profiles[0].model_copy(
            update={"provenance_profile_id": "other_provenance"}
        )
        unknown_node_profile = registry.provenance_profiles[0].model_copy(
            update={"source_backbone_node_ids": ("missing_node",)}
        )

        with self.assertRaisesRegex(ValueError, "provenance_profile_ids must match"):
            DecisionProvenanceRegistry(
                provenance_profiles=(
                    mismatched_profile,
                )
                + registry.provenance_profiles[1:],
                provenance_profile_ids=registry.provenance_profile_ids,
                topic_ids=registry.topic_ids,
                source_registries=registry.source_registries,
                confidence_fusion_profile_ids=registry.confidence_fusion_profile_ids,
                voting_profile_ids=registry.voting_profile_ids,
                debate_loop_ids=registry.debate_loop_ids,
                backbone_node_ids=registry.backbone_node_ids,
                workstation_surface_ids=registry.workstation_surface_ids,
                profile_count=registry.profile_count,
            )

        with self.assertRaisesRegex(ValueError, "provenance backbone nodes"):
            DecisionProvenanceRegistry(
                provenance_profiles=(
                    unknown_node_profile,
                )
                + registry.provenance_profiles[1:],
                provenance_profile_ids=registry.provenance_profile_ids,
                topic_ids=registry.topic_ids,
                source_registries=registry.source_registries,
                confidence_fusion_profile_ids=registry.confidence_fusion_profile_ids,
                voting_profile_ids=registry.voting_profile_ids,
                debate_loop_ids=registry.debate_loop_ids,
                backbone_node_ids=registry.backbone_node_ids,
                workstation_surface_ids=registry.workstation_surface_ids,
                profile_count=registry.profile_count,
            )

    def test_decision_provenance_does_not_declare_active_execution(self) -> None:
        registry = decision_provenance_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for profile in registry.provenance_profiles
                    for field in (
                        profile.provenance_profile_id,
                        profile.authority_boundary,
                        *profile.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "record_runtime",
            "emit_runtime_trace",
            "write_runtime_memory",
            "execute_agent",
            "modify_output",
        ):
            self.assertNotIn(forbidden_term, combined_text)


class EscalationTraceRegistryTests(unittest.TestCase):
    def test_registry_declares_passive_escalation_trace_profiles(self) -> None:
        registry = escalation_trace_registry()

        self.assertEqual(registry.role, "escalation_trace_registry")
        self.assertEqual(
            registry.serialization_version,
            "escalation_trace_registry.v1",
        )
        self.assertEqual(registry.trace_profile_ids, EXPECTED_ESCALATION_TRACE_PROFILE_IDS)
        self.assertEqual(registry.topic_ids, EXPECTED_HYBRID_DEBATE_TOPICS)
        self.assertEqual(
            registry.source_registries,
            EXPECTED_ESCALATION_TRACE_SOURCE_REGISTRIES,
        )
        self.assertEqual(
            registry.provenance_profile_ids,
            decision_provenance_registry().provenance_profile_ids,
        )
        self.assertEqual(
            registry.condition_ids,
            conditional_multi_agent_escalation_registry().condition_ids,
        )
        self.assertEqual(registry.gate_ids, escalation_gate_registry().gate_ids)
        self.assertEqual(
            registry.escalation_signal_ids,
            agent_escalation_signal_registry().signal_ids,
        )
        self.assertEqual(
            registry.reflection_profile_ids,
            reflection_escalation_registry().profile_ids,
        )
        self.assertEqual(registry.profile_count, 4)
        self.assertIn("does not capture traces", registry.authority_boundary)
        self.assertFalse(registry.trace_capture_implemented)
        self.assertFalse(registry.trace_emission_implemented)
        self.assertFalse(registry.escalation_execution_implemented)
        self.assertFalse(registry.gate_evaluation_implemented)
        self.assertFalse(registry.memory_write_implemented)
        self.assertFalse(registry.agent_invocation_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertTrue(registry.metadata_only)

    def test_escalation_trace_profiles_reference_known_sources(self) -> None:
        registry = escalation_trace_registry()
        known_provenance = set(decision_provenance_registry().provenance_profile_ids)
        known_conditions = set(conditional_multi_agent_escalation_registry().condition_ids)
        known_gates = set(escalation_gate_registry().gate_ids)
        known_signals = set(agent_escalation_signal_registry().signal_ids)
        known_reflections = set(reflection_escalation_registry().profile_ids)

        for profile in registry.trace_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_ESCALATION_TRACE_FIELDS)
            self.assertEqual(
                profile.source_registries,
                EXPECTED_ESCALATION_TRACE_SOURCE_REGISTRIES,
            )
            self.assertIn(profile.source_provenance_profile_id, known_provenance)
            self.assertTrue(set(profile.source_condition_ids).issubset(known_conditions))
            self.assertTrue(set(profile.source_gate_ids).issubset(known_gates))
            self.assertTrue(
                set(profile.source_escalation_signal_ids).issubset(known_signals)
            )
            self.assertTrue(
                set(profile.source_reflection_profile_ids).issubset(known_reflections)
            )
            self.assertTrue(profile.trace_dimensions)
            self.assertTrue(profile.advisory_outputs)
            self.assertIn("trace_capture", profile.blocked_runtime_behaviors)
            self.assertIn("trace_emission", profile.blocked_runtime_behaviors)
            self.assertFalse(profile.trace_capture_implemented)
            self.assertFalse(profile.trace_emission_implemented)
            self.assertFalse(profile.escalation_execution_implemented)
            self.assertFalse(profile.gate_evaluation_implemented)
            self.assertFalse(profile.memory_write_implemented)
            self.assertFalse(profile.agent_invocation_implemented)
            self.assertFalse(profile.workflow_control_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)
            self.assertEqual(
                profile.serialization_version,
                "escalation_trace_profile.v1",
            )
            self.assertTrue(profile.metadata_only)

    def test_escalation_trace_source_registries_are_complete(self) -> None:
        registry = escalation_trace_registry()
        profile_sources = tuple(
            dict.fromkeys(
                source
                for profile in registry.trace_profiles
                for source in profile.source_registries
            )
        )

        self.assertEqual(profile_sources, registry.source_registries)
        for source_registry in EXPECTED_ESCALATION_TRACE_SOURCE_REGISTRIES:
            self.assertIn(source_registry, profile_sources)
        for profile in registry.trace_profiles:
            self.assertEqual(set(profile.source_registries), set(profile_sources))

    def test_escalation_trace_lookup_is_stable(self) -> None:
        profile = escalation_trace_profile_by_id(
            "escalation_trace::final_synthesis_readiness"
        )
        missing = escalation_trace_profile_by_id("missing_trace")

        self.assertIsNone(missing)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.topic_id, "final_synthesis_readiness")
        self.assertIn("hitl_escalation_signal", profile.source_escalation_signal_ids)
        self.assertIn("final_trace_context", profile.advisory_outputs)
        self.assertFalse(profile.trace_capture_implemented)

    def test_escalation_trace_registry_rejects_mismatched_metadata(self) -> None:
        registry = escalation_trace_registry()
        mismatched_profile = registry.trace_profiles[0].model_copy(
            update={"trace_profile_id": "other_trace"}
        )
        unknown_signal_profile = registry.trace_profiles[0].model_copy(
            update={"source_escalation_signal_ids": ("missing_signal",)}
        )

        with self.assertRaisesRegex(ValueError, "trace_profile_ids must match"):
            EscalationTraceRegistry(
                trace_profiles=(mismatched_profile,) + registry.trace_profiles[1:],
                trace_profile_ids=registry.trace_profile_ids,
                topic_ids=registry.topic_ids,
                source_registries=registry.source_registries,
                provenance_profile_ids=registry.provenance_profile_ids,
                condition_ids=registry.condition_ids,
                gate_ids=registry.gate_ids,
                escalation_signal_ids=registry.escalation_signal_ids,
                reflection_profile_ids=registry.reflection_profile_ids,
                profile_count=registry.profile_count,
            )

        with self.assertRaisesRegex(ValueError, "trace signals must be known"):
            EscalationTraceRegistry(
                trace_profiles=(unknown_signal_profile,) + registry.trace_profiles[1:],
                trace_profile_ids=registry.trace_profile_ids,
                topic_ids=registry.topic_ids,
                source_registries=registry.source_registries,
                provenance_profile_ids=registry.provenance_profile_ids,
                condition_ids=registry.condition_ids,
                gate_ids=registry.gate_ids,
                escalation_signal_ids=registry.escalation_signal_ids,
                reflection_profile_ids=registry.reflection_profile_ids,
                profile_count=registry.profile_count,
            )

    def test_escalation_trace_does_not_declare_active_execution(self) -> None:
        registry = escalation_trace_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for profile in registry.trace_profiles
                    for field in (
                        profile.trace_profile_id,
                        profile.authority_boundary,
                        *profile.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "capture_runtime_trace",
            "emit_runtime_trace",
            "execute_escalation",
            "execute_agent",
            "modify_output",
        ):
            self.assertNotIn(forbidden_term, combined_text)


class CreativeExplorationBudgetRegistryTests(unittest.TestCase):
    def test_registry_declares_passive_exploration_budget_profiles(self) -> None:
        registry = creative_exploration_budget_registry()

        self.assertEqual(registry.role, "creative_exploration_budget_registry")
        self.assertEqual(
            registry.serialization_version,
            "creative_exploration_budget_registry.v1",
        )
        self.assertEqual(
            registry.budget_profile_ids,
            EXPECTED_CREATIVE_EXPLORATION_BUDGET_PROFILE_IDS,
        )
        self.assertEqual(registry.topic_ids, EXPECTED_HYBRID_DEBATE_TOPICS)
        self.assertEqual(
            registry.budget_postures,
            EXPECTED_CREATIVE_EXPLORATION_BUDGET_POSTURES,
        )
        self.assertEqual(
            registry.source_registries,
            EXPECTED_CREATIVE_EXPLORATION_BUDGET_SOURCE_REGISTRIES,
        )
        self.assertEqual(
            registry.trace_profile_ids,
            escalation_trace_registry().trace_profile_ids,
        )
        self.assertEqual(
            registry.provenance_profile_ids,
            decision_provenance_registry().provenance_profile_ids,
        )
        self.assertEqual(
            registry.escalation_signal_ids,
            agent_escalation_signal_registry().signal_ids,
        )
        self.assertEqual(registry.profile_count, 4)
        self.assertIn("does not enforce budgets", registry.authority_boundary)
        self.assertFalse(registry.budget_enforcement_implemented)
        self.assertFalse(registry.variant_generation_implemented)
        self.assertFalse(registry.refinement_triggering_implemented)
        self.assertFalse(registry.cost_routing_implemented)
        self.assertFalse(registry.agent_invocation_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertTrue(registry.metadata_only)

    def test_exploration_budget_profiles_reference_known_sources(self) -> None:
        registry = creative_exploration_budget_registry()
        known_traces = set(escalation_trace_registry().trace_profile_ids)
        known_provenance = set(decision_provenance_registry().provenance_profile_ids)
        known_signals = set(agent_escalation_signal_registry().signal_ids)

        for profile in registry.budget_profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_CREATIVE_EXPLORATION_BUDGET_FIELDS)
            self.assertEqual(
                profile.source_registries,
                EXPECTED_CREATIVE_EXPLORATION_BUDGET_SOURCE_REGISTRIES,
            )
            self.assertIn(profile.source_trace_profile_id, known_traces)
            self.assertIn(profile.source_provenance_profile_id, known_provenance)
            self.assertTrue(
                set(profile.source_escalation_signal_ids).issubset(known_signals)
            )
            self.assertIn(profile.budget_posture, registry.budget_postures)
            self.assertGreaterEqual(profile.max_advisory_variants, 0)
            self.assertLessEqual(profile.max_advisory_variants, 3)
            self.assertGreaterEqual(profile.max_advisory_refinement_passes, 0)
            self.assertLessEqual(profile.max_advisory_refinement_passes, 3)
            self.assertTrue(profile.cost_pressure_signal)
            self.assertTrue(profile.budget_dimensions)
            self.assertTrue(profile.advisory_outputs)
            self.assertIn("budget_enforcement", profile.blocked_runtime_behaviors)
            self.assertIn("variant_generation", profile.blocked_runtime_behaviors)
            self.assertFalse(profile.budget_enforcement_implemented)
            self.assertFalse(profile.variant_generation_implemented)
            self.assertFalse(profile.refinement_triggering_implemented)
            self.assertFalse(profile.cost_routing_implemented)
            self.assertFalse(profile.agent_invocation_implemented)
            self.assertFalse(profile.workflow_control_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)
            self.assertEqual(
                profile.serialization_version,
                "creative_exploration_budget_profile.v1",
            )
            self.assertTrue(profile.metadata_only)

    def test_exploration_budget_source_registries_are_complete(self) -> None:
        registry = creative_exploration_budget_registry()
        profile_sources = tuple(
            dict.fromkeys(
                source
                for profile in registry.budget_profiles
                for source in profile.source_registries
            )
        )

        self.assertEqual(profile_sources, registry.source_registries)
        for source_registry in EXPECTED_CREATIVE_EXPLORATION_BUDGET_SOURCE_REGISTRIES:
            self.assertIn(source_registry, profile_sources)
        for profile in registry.budget_profiles:
            self.assertEqual(set(profile.source_registries), set(profile_sources))

    def test_exploration_budget_lookup_is_stable(self) -> None:
        profile = creative_exploration_budget_profile_by_id(
            "creative_exploration_budget::style_aesthetic_alignment"
        )
        missing = creative_exploration_budget_profile_by_id("missing_budget")

        self.assertIsNone(missing)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.topic_id, "style_aesthetic_alignment")
        self.assertEqual(profile.budget_posture, "broad")
        self.assertEqual(profile.max_advisory_variants, 3)
        self.assertFalse(profile.variant_generation_implemented)

    def test_exploration_budget_registry_rejects_mismatched_metadata(self) -> None:
        registry = creative_exploration_budget_registry()
        mismatched_profile = registry.budget_profiles[0].model_copy(
            update={"budget_profile_id": "other_budget"}
        )
        unknown_trace_profile = registry.budget_profiles[0].model_copy(
            update={"source_trace_profile_id": "missing_trace"}
        )

        with self.assertRaisesRegex(ValueError, "budget_profile_ids must match"):
            CreativeExplorationBudgetRegistry(
                budget_profiles=(mismatched_profile,) + registry.budget_profiles[1:],
                budget_profile_ids=registry.budget_profile_ids,
                topic_ids=registry.topic_ids,
                budget_postures=registry.budget_postures,
                source_registries=registry.source_registries,
                trace_profile_ids=registry.trace_profile_ids,
                provenance_profile_ids=registry.provenance_profile_ids,
                escalation_signal_ids=registry.escalation_signal_ids,
                profile_count=registry.profile_count,
            )

        with self.assertRaisesRegex(ValueError, "budget trace profiles must be known"):
            CreativeExplorationBudgetRegistry(
                budget_profiles=(unknown_trace_profile,) + registry.budget_profiles[1:],
                budget_profile_ids=registry.budget_profile_ids,
                topic_ids=registry.topic_ids,
                budget_postures=registry.budget_postures,
                source_registries=registry.source_registries,
                trace_profile_ids=registry.trace_profile_ids,
                provenance_profile_ids=registry.provenance_profile_ids,
                escalation_signal_ids=registry.escalation_signal_ids,
                profile_count=registry.profile_count,
            )

    def test_exploration_budget_does_not_declare_active_execution(self) -> None:
        registry = creative_exploration_budget_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for profile in registry.budget_profiles
                    for field in (
                        profile.budget_profile_id,
                        profile.authority_boundary,
                        *profile.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "enforce_runtime_budget",
            "generate_variant",
            "trigger_runtime_refinement",
            "execute_agent",
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
