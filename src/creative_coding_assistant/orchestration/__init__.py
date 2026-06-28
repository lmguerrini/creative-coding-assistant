"""Assistant orchestration and explicit routing."""

from __future__ import annotations

from importlib import import_module

_CTX = "creative_coding_assistant.orchestration.context"
_CREATIVE_TRANSLATION = (
    "creative_coding_assistant.orchestration.creative_translation"
)
_CREATIVE_PLANNING = "creative_coding_assistant.orchestration.creative_planning"
_CREATIVE_CONSTRAINTS = (
    "creative_coding_assistant.orchestration.creative_constraints"
)
_CREATIVE_CONSTRAINT_PRIORITIES = (
    "creative_coding_assistant.orchestration.creative_constraint_priorities"
)
_CREATIVE_HIERARCHY = "creative_coding_assistant.orchestration.creative_hierarchy"
_CREATIVE_INTENT = "creative_coding_assistant.orchestration.creative_intent"
_CREATIVE_STRATEGY = "creative_coding_assistant.orchestration.creative_strategy"
_CREATIVE_TECHNIQUE = "creative_coding_assistant.orchestration.creative_technique"
_CREATIVE_TRADEOFFS = "creative_coding_assistant.orchestration.creative_tradeoffs"
_CREATIVE_REASONING = "creative_coding_assistant.orchestration.creative_reasoning"
_CREATIVE_CRITIC_ENGINE = (
    "creative_coding_assistant.orchestration.creative_critic_engine"
)
_SELF_EVALUATION_ENGINE = (
    "creative_coding_assistant.orchestration.self_evaluation_engine"
)
_CREATIVE_IMPROVEMENT_PLANNER = (
    "creative_coding_assistant.orchestration.creative_improvement_planner"
)
_REFLECTION_LOOP_ENGINE = (
    "creative_coding_assistant.orchestration.reflection_loop_engine"
)
_CREATIVE_CONFIDENCE_ENGINE = (
    "creative_coding_assistant.orchestration.creative_confidence_engine"
)
_CREATIVE_SCORE_ENGINE = (
    "creative_coding_assistant.orchestration.creative_score_engine"
)
_CONSISTENCY_VALIDATION_ENGINE = (
    "creative_coding_assistant.orchestration.consistency_validation_engine"
)
_EVALUATION_REPORTS = "creative_coding_assistant.orchestration.evaluation_reports"
_EVALUATION_ENGINE_CONTRACTS = (
    "creative_coding_assistant.orchestration.evaluation_engine_contracts"
)
_WORKSTATION_ENGINE_CONTRACTS = (
    "creative_coding_assistant.orchestration.workstation_contracts"
)
_AGENT_CONTRACTS = "creative_coding_assistant.orchestration.agent_contracts"
_AGENT_CONTRACT_AUDIT = (
    "creative_coding_assistant.orchestration.agent_contract_audit"
)
_AGENT_REGISTRY_AUDIT = (
    "creative_coding_assistant.orchestration.agent_registry_audit"
)
_AGENT_IDENTITIES = "creative_coding_assistant.orchestration.agent_identities"
_AGENT_MEMORY_CONTRACTS = (
    "creative_coding_assistant.orchestration.agent_memory_contracts"
)
_AGENT_ROLES = "creative_coding_assistant.orchestration.agent_roles"
_AGENT_BOUNDARIES = "creative_coding_assistant.orchestration.agent_boundaries"
_AGENT_METADATA = "creative_coding_assistant.orchestration.agent_metadata"
_AGENT_ROUTING = "creative_coding_assistant.orchestration.agent_routing"
_BLACKBOARD_MEMORY = "creative_coding_assistant.orchestration.blackboard_memory"
_BLACKBOARD_AUDIT = "creative_coding_assistant.orchestration.blackboard_audit"
_SHARED_CONTEXT_VIEWS = (
    "creative_coding_assistant.orchestration.shared_context_views"
)
_SHARED_CONTEXT_AUDIT = (
    "creative_coding_assistant.orchestration.shared_context_audit"
)
_AGENT_DEPENDENCY_GRAPH = (
    "creative_coding_assistant.orchestration.agent_dependency_graph"
)
_AGENT_PARALLEL_SCHEDULING = (
    "creative_coding_assistant.orchestration.agent_parallel_scheduling"
)
_AGENT_COORDINATION = "creative_coding_assistant.orchestration.agent_coordination"
_AGENT_DEBATE = "creative_coding_assistant.orchestration.agent_debate"
_AGENT_CONSENSUS = "creative_coding_assistant.orchestration.agent_consensus"
_AGENT_COLLABORATION_AUDIT = (
    "creative_coding_assistant.orchestration.agent_collaboration_audit"
)
_AGENT_CAPABILITY_ALIGNMENT = (
    "creative_coding_assistant.orchestration.agent_capability_alignment"
)
_AGENT_ESCALATION_SIGNALS = (
    "creative_coding_assistant.orchestration.agent_escalation_signals"
)
_AGENT_LIFECYCLE = "creative_coding_assistant.orchestration.agent_lifecycle"
_AGENT_STATE_SYNCHRONIZATION = (
    "creative_coding_assistant.orchestration.agent_state_synchronization"
)
_WORKFLOW_AGENT_HANDOFF = (
    "creative_coding_assistant.orchestration.workflow_agent_handoff"
)
_ORCHESTRATION_CONTRACT_INTEGRATION = (
    "creative_coding_assistant.orchestration.orchestration_contract_integration"
)
_AGENT_CAPABILITY_REGISTRY = (
    "creative_coding_assistant.orchestration.agent_capabilities"
)
_ESCALATION_POLICY = (
    "creative_coding_assistant.orchestration.escalation_policy"
)
_ESCALATION_POLICY_AUDIT = (
    "creative_coding_assistant.orchestration.escalation_policy_audit"
)
_HYBRID_AGENTIC_WORKFLOW = (
    "creative_coding_assistant.orchestration.hybrid_agentic_workflow"
)
_HYBRID_WORKFLOW_AUDIT = (
    "creative_coding_assistant.orchestration.hybrid_workflow_audit"
)
_CREATIVE_DIVERSITY_AUDIT = (
    "creative_coding_assistant.orchestration.creative_diversity_audit"
)
_AGENT_EXPLAINABILITY_AUDIT = (
    "creative_coding_assistant.orchestration.agent_explainability_audit"
)
_AGENT_RELIABILITY_AUDIT = (
    "creative_coding_assistant.orchestration.agent_reliability_audit"
)
_AGENT_DETERMINISM_AUDIT = (
    "creative_coding_assistant.orchestration.agent_determinism_audit"
)
_AGENT_TELEMETRY_FOUNDATION = (
    "creative_coding_assistant.orchestration.agent_telemetry_foundation"
)
_AGENT_COST_TRACKING_FOUNDATION = (
    "creative_coding_assistant.orchestration.agent_cost_tracking_foundation"
)
_AGENT_PERFORMANCE_TRACKING_FOUNDATION = (
    "creative_coding_assistant.orchestration."
    "agent_performance_tracking_foundation"
)
_ARCHITECTURE_CONSISTENCY_PASS = (
    "creative_coding_assistant.orchestration.architecture_consistency_pass"
)
_FINAL_V4_HARDENING = (
    "creative_coding_assistant.orchestration.final_v4_hardening"
)
_HYBRID_STUDIO = "creative_coding_assistant.orchestration.hybrid_studio"
_MULTIMODAL_STUDIO = "creative_coding_assistant.orchestration.multimodal_studio"
_ENGINE_CONTRACT_CONSISTENCY = (
    "creative_coding_assistant.orchestration.engine_contract_consistency"
)
_ARTIFACT_PLANNER = "creative_coding_assistant.orchestration.artifact_planner"
_ARTIFACT_DEPENDENCY_GRAPH = (
    "creative_coding_assistant.orchestration.artifact_dependency_graph"
)
_ARTIFACT_ENGINE_CONTRACTS = (
    "creative_coding_assistant.orchestration.artifact_engine_contracts"
)
_RUNTIME_COMPATIBILITY = (
    "creative_coding_assistant.orchestration.runtime_compatibility"
)
_ARTIFACT_CAPABILITY_MATRIX = (
    "creative_coding_assistant.orchestration.artifact_capability_matrix"
)
_MULTI_ARTIFACT_STRATEGY = (
    "creative_coding_assistant.orchestration.multi_artifact_strategy"
)
_ARTIFACT_CRITIC = "creative_coding_assistant.orchestration.artifact_critic"
_ARTIFACT_REFINER = "creative_coding_assistant.orchestration.artifact_refiner"
_ARTIFACT_INTELLIGENCE_SYNTHESIS = (
    "creative_coding_assistant.orchestration.artifact_intelligence_synthesis"
)
_ARTIFACT_MERGE_PLANNER = (
    "creative_coding_assistant.orchestration.artifact_merge_planner"
)
_ARTIFACT_EXPORT_INTELLIGENCE = (
    "creative_coding_assistant.orchestration.artifact_export_intelligence"
)
_CREATIVE_QUALITY_PREDICTION = (
    "creative_coding_assistant.orchestration.creative_quality_prediction"
)
_CREATIVE_COMPOSITION = (
    "creative_coding_assistant.orchestration.creative_composition"
)
_PROCEDURAL_STRUCTURE = (
    "creative_coding_assistant.orchestration.procedural_structure"
)
_GENERATIVE_STRUCTURE = (
    "creative_coding_assistant.orchestration.generative_structure"
)
_SEMANTIC_MOTIF = "creative_coding_assistant.orchestration.semantic_motif"
_EMOTIONAL_CONSISTENCY = (
    "creative_coding_assistant.orchestration.emotional_consistency"
)
_CROSS_MODALITY = "creative_coding_assistant.orchestration.cross_modality"
_AUDIO_VISUAL_SCENE = (
    "creative_coding_assistant.orchestration.audio_visual_scene"
)
_SYMBOLIC_NARRATIVE = "creative_coding_assistant.orchestration.symbolic_narrative"
_RUNTIME_CAPABILITIES = (
    "creative_coding_assistant.orchestration.runtime_capabilities"
)
_CREATIVE_DIRECTOR = "creative_coding_assistant.orchestration.creative_director"
_AUDIO_REACTIVE = "creative_coding_assistant.orchestration.audio_reactive"
_CREATIVE_QUALITY = "creative_coding_assistant.orchestration.creative_quality"
_SACRED_CONSISTENCY = (
    "creative_coding_assistant.orchestration.sacred_consistency"
)
_SACRED_GEOMETRY = "creative_coding_assistant.orchestration.sacred_geometry"
_SHADER_PRESETS = "creative_coding_assistant.orchestration.shader_presets"
_VISUAL_STYLES = "creative_coding_assistant.orchestration.visual_styles"
_ARTIFACTS = "creative_coding_assistant.orchestration.artifacts"
_ARTIFACT_CRITIQUE = "creative_coding_assistant.orchestration.artifact_critique"
_CLARIFICATION = "creative_coding_assistant.orchestration.clarification"
_EVENTS = "creative_coding_assistant.orchestration.events"
_GEN = "creative_coding_assistant.orchestration.generation"
_MEM = "creative_coding_assistant.orchestration.memory"
_PROMPT_INPUTS = "creative_coding_assistant.orchestration.prompt_inputs"
_PROMPT_TEMPLATES = "creative_coding_assistant.orchestration.prompt_templates"
_QUALITY_CALIBRATION = (
    "creative_coding_assistant.orchestration.quality_calibration"
)
_REFINEMENT_PASSES = "creative_coding_assistant.orchestration.refinement_passes"
_REFERENCE_FUSION = "creative_coding_assistant.orchestration.reference_fusion"
_RETRIEVAL = "creative_coding_assistant.orchestration.retrieval"
_ROUTING = "creative_coding_assistant.orchestration.routing"
_SERVICE = "creative_coding_assistant.orchestration.service"
_WORKFLOW = "creative_coding_assistant.orchestration.workflow"
_WORKFLOW_GRAPH = "creative_coding_assistant.orchestration.workflow_graph"
_WORKFLOW_REVIEW = "creative_coding_assistant.orchestration.workflow_review"
_EXECUTION_GRAPH_ANALYZER = (
    "creative_coding_assistant.orchestration.execution_graph_analyzer"
)
_WORKFLOW_COST_ANALYZER = (
    "creative_coding_assistant.orchestration.workflow_cost_analyzer"
)

_EXPORT_MAP = {
    "ASSISTANT_WORKFLOW_NODE_ORDER": _WORKFLOW_GRAPH,
    "ASSISTANT_WORKFLOW_RECURSION_LIMIT": _WORKFLOW_GRAPH,
    "AssistantWorkflowGraphState": _WORKFLOW_GRAPH,
    "AssistantWorkflowRuntime": _WORKFLOW_GRAPH,
    "AssistantWorkflowState": _WORKFLOW,
    "ExecutionGraphAnalysis": _EXECUTION_GRAPH_ANALYZER,
    "ExecutionGraphEdge": _EXECUTION_GRAPH_ANALYZER,
    "ExecutionGraphNode": _EXECUTION_GRAPH_ANALYZER,
    "WorkflowCostAnalysis": _WORKFLOW_COST_ANALYZER,
    "WorkflowCostComponent": _WORKFLOW_COST_ANALYZER,
    "ArtifactDependencyEdge": _ARTIFACT_DEPENDENCY_GRAPH,
    "ArtifactDependencyGraph": _ARTIFACT_DEPENDENCY_GRAPH,
    "ArtifactDependencyNode": _ARTIFACT_DEPENDENCY_GRAPH,
    "ArtifactEngineCostMetadata": _ARTIFACT_ENGINE_CONTRACTS,
    "ArtifactEngineLatencyMetadata": _ARTIFACT_ENGINE_CONTRACTS,
    "ArtifactIntelligenceEngineContract": _ARTIFACT_ENGINE_CONTRACTS,
    "ArtifactIntelligenceEngineContractRegistry": _ARTIFACT_ENGINE_CONTRACTS,
    "EvaluationEngineContract": _EVALUATION_ENGINE_CONTRACTS,
    "EvaluationEngineContractRegistry": _EVALUATION_ENGINE_CONTRACTS,
    "EvaluationEngineCostMetadata": _EVALUATION_ENGINE_CONTRACTS,
    "EvaluationEngineEvidenceContract": _EVALUATION_ENGINE_CONTRACTS,
    "EvaluationEngineLatencyMetadata": _EVALUATION_ENGINE_CONTRACTS,
    "WorkstationEngineContract": _WORKSTATION_ENGINE_CONTRACTS,
    "WorkstationEngineContractRegistry": _WORKSTATION_ENGINE_CONTRACTS,
    "WorkstationSurfaceCostMetadata": _WORKSTATION_ENGINE_CONTRACTS,
    "WorkstationSurfaceLatencyMetadata": _WORKSTATION_ENGINE_CONTRACTS,
    "AgentContract": _AGENT_CONTRACTS,
    "AgentContractCostMetadata": _AGENT_CONTRACTS,
    "AgentContractLatencyMetadata": _AGENT_CONTRACTS,
    "AgentContractRegistry": _AGENT_CONTRACTS,
    "AgentMemoryAccessContract": _AGENT_CONTRACTS,
    "agent_contract_by_id": _AGENT_CONTRACTS,
    "agent_contract_registry": _AGENT_CONTRACTS,
    "build_agent_contract_registry": _AGENT_CONTRACTS,
    "AgentContractAuditRecord": _AGENT_CONTRACT_AUDIT,
    "AgentContractAuditRegistry": _AGENT_CONTRACT_AUDIT,
    "agent_contract_audit_by_agent_id": _AGENT_CONTRACT_AUDIT,
    "agent_contract_audit_registry": _AGENT_CONTRACT_AUDIT,
    "agent_contract_audits_for_registry_ref": _AGENT_CONTRACT_AUDIT,
    "AgentRegistryAuditEntry": _AGENT_REGISTRY_AUDIT,
    "AgentRegistryAuditRegistry": _AGENT_REGISTRY_AUDIT,
    "agent_registry_audit_by_registry_id": _AGENT_REGISTRY_AUDIT,
    "agent_registry_audit_registry": _AGENT_REGISTRY_AUDIT,
    "agent_registry_audits_for_kind": _AGENT_REGISTRY_AUDIT,
    "AgentIdentityMetadata": _AGENT_IDENTITIES,
    "AgentIdentityRegistry": _AGENT_IDENTITIES,
    "agent_identities_by_role_family": _AGENT_IDENTITIES,
    "agent_identity_by_id": _AGENT_IDENTITIES,
    "agent_identity_registry": _AGENT_IDENTITIES,
    "AgentMemoryContract": _AGENT_MEMORY_CONTRACTS,
    "AgentMemoryContractRegistry": _AGENT_MEMORY_CONTRACTS,
    "AgentMemorySurfaceContract": _AGENT_MEMORY_CONTRACTS,
    "agent_memory_contract_by_agent_id": _AGENT_MEMORY_CONTRACTS,
    "agent_memory_contract_registry": _AGENT_MEMORY_CONTRACTS,
    "AgentRoleMetadata": _AGENT_ROLES,
    "AgentRoleRegistry": _AGENT_ROLES,
    "agent_role_by_id": _AGENT_ROLES,
    "agent_role_registry": _AGENT_ROLES,
    "agent_roles_by_capability_family": _AGENT_ROLES,
    "agent_roles_by_family": _AGENT_ROLES,
    "AgentBoundaryMetadata": _AGENT_BOUNDARIES,
    "AgentBoundaryRegistry": _AGENT_BOUNDARIES,
    "agent_boundary_by_agent_id": _AGENT_BOUNDARIES,
    "agent_boundary_registry": _AGENT_BOUNDARIES,
    "AgentMetadataRegistry": _AGENT_METADATA,
    "AgentOperationalMetadata": _AGENT_METADATA,
    "agent_metadata_by_agent_id": _AGENT_METADATA,
    "agent_metadata_registry": _AGENT_METADATA,
    "AgentRoutingProfile": _AGENT_ROUTING,
    "AgentRoutingRegistry": _AGENT_ROUTING,
    "agent_routing_profile_by_agent_id": _AGENT_ROUTING,
    "agent_routing_profiles_for_route": _AGENT_ROUTING,
    "agent_routing_registry": _AGENT_ROUTING,
    "AgentConversationViewKind": _HYBRID_STUDIO,
    "AgentConversationViewProfile": _HYBRID_STUDIO,
    "AgentConversationViewRegistry": _HYBRID_STUDIO,
    "AgentWorkspaceKind": _HYBRID_STUDIO,
    "AgentWorkspaceProfile": _HYBRID_STUDIO,
    "AgentWorkspaceRegistry": _HYBRID_STUDIO,
    "BlackboardAgentPermissionContract": _BLACKBOARD_MEMORY,
    "BlackboardMemoryChannelContract": _BLACKBOARD_MEMORY,
    "BlackboardMemoryRegistry": _BLACKBOARD_MEMORY,
    "blackboard_channel_by_id": _BLACKBOARD_MEMORY,
    "blackboard_channels_for_agent": _BLACKBOARD_MEMORY,
    "blackboard_memory_registry": _BLACKBOARD_MEMORY,
    "blackboard_permissions_by_agent_id": _BLACKBOARD_MEMORY,
    "BlackboardAuditRecord": _BLACKBOARD_AUDIT,
    "BlackboardAuditRegistry": _BLACKBOARD_AUDIT,
    "blackboard_audit_by_agent_id": _BLACKBOARD_AUDIT,
    "blackboard_audit_by_channel_id": _BLACKBOARD_AUDIT,
    "blackboard_audit_registry": _BLACKBOARD_AUDIT,
    "blackboard_audits_for_source_registry": _BLACKBOARD_AUDIT,
    "SharedContextViewContract": _SHARED_CONTEXT_VIEWS,
    "SharedContextViewRegistry": _SHARED_CONTEXT_VIEWS,
    "shared_context_view_by_agent_id": _SHARED_CONTEXT_VIEWS,
    "shared_context_view_by_id": _SHARED_CONTEXT_VIEWS,
    "shared_context_view_registry": _SHARED_CONTEXT_VIEWS,
    "SharedContextAuditRecord": _SHARED_CONTEXT_AUDIT,
    "SharedContextAuditRegistry": _SHARED_CONTEXT_AUDIT,
    "shared_context_audit_by_agent_id": _SHARED_CONTEXT_AUDIT,
    "shared_context_audit_by_view_id": _SHARED_CONTEXT_AUDIT,
    "shared_context_audit_registry": _SHARED_CONTEXT_AUDIT,
    "shared_context_audits_for_source_registry": _SHARED_CONTEXT_AUDIT,
    "AgentDependencyEdge": _AGENT_DEPENDENCY_GRAPH,
    "AgentDependencyGraphRegistry": _AGENT_DEPENDENCY_GRAPH,
    "AgentDependencyNode": _AGENT_DEPENDENCY_GRAPH,
    "agent_dependency_downstream_nodes": _AGENT_DEPENDENCY_GRAPH,
    "agent_dependency_graph_registry": _AGENT_DEPENDENCY_GRAPH,
    "agent_dependency_node_by_id": _AGENT_DEPENDENCY_GRAPH,
    "agent_dependency_upstream_nodes": _AGENT_DEPENDENCY_GRAPH,
    "ParallelSchedulingGroup": _AGENT_PARALLEL_SCHEDULING,
    "ParallelSchedulingRegistry": _AGENT_PARALLEL_SCHEDULING,
    "parallel_scheduling_group_by_id": _AGENT_PARALLEL_SCHEDULING,
    "parallel_scheduling_group_for_agent": _AGENT_PARALLEL_SCHEDULING,
    "parallel_scheduling_registry": _AGENT_PARALLEL_SCHEDULING,
    "AgentCoordinationRegistry": _AGENT_COORDINATION,
    "CoordinationEventContract": _AGENT_COORDINATION,
    "CoordinationHandoffChannelContract": _AGENT_COORDINATION,
    "CoordinationResponsibilityContract": _AGENT_COORDINATION,
    "agent_coordination_registry": _AGENT_COORDINATION,
    "coordination_event_contract_by_type": _AGENT_COORDINATION,
    "coordination_handoff_channel_by_id": _AGENT_COORDINATION,
    "coordination_responsibility_by_id": _AGENT_COORDINATION,
    "AgentDebateClaimContract": _AGENT_DEBATE,
    "AgentDebateParticipant": _AGENT_DEBATE,
    "AgentDebateRegistry": _AGENT_DEBATE,
    "AgentDebateRoundContract": _AGENT_DEBATE,
    "agent_debate_registry": _AGENT_DEBATE,
    "debate_claim_by_id": _AGENT_DEBATE,
    "debate_participant_by_agent_id": _AGENT_DEBATE,
    "debate_round_by_topic": _AGENT_DEBATE,
    "ConsensusAgreementSurface": _AGENT_CONSENSUS,
    "ConsensusBuilderRegistry": _AGENT_CONSENSUS,
    "ConsensusVotingInputContract": _AGENT_CONSENSUS,
    "consensus_agreement_surface_by_topic": _AGENT_CONSENSUS,
    "consensus_builder_registry": _AGENT_CONSENSUS,
    "consensus_voting_input_by_topic": _AGENT_CONSENSUS,
    "AgentCollaborationAuditRecord": _AGENT_COLLABORATION_AUDIT,
    "AgentCollaborationAuditRegistry": _AGENT_COLLABORATION_AUDIT,
    "agent_collaboration_audit_by_registry_id": _AGENT_COLLABORATION_AUDIT,
    "agent_collaboration_audit_registry": _AGENT_COLLABORATION_AUDIT,
    "agent_collaboration_audits_for_source_registry": (
        _AGENT_COLLABORATION_AUDIT
    ),
    "agent_collaboration_audits_for_surface": _AGENT_COLLABORATION_AUDIT,
    "AgentCapabilityAlignmentProfile": _AGENT_CAPABILITY_ALIGNMENT,
    "AgentCapabilityAlignmentRegistry": _AGENT_CAPABILITY_ALIGNMENT,
    "agent_capability_alignment_by_agent_id": _AGENT_CAPABILITY_ALIGNMENT,
    "agent_capability_alignment_registry": _AGENT_CAPABILITY_ALIGNMENT,
    "AgentEscalationSignal": _AGENT_ESCALATION_SIGNALS,
    "AgentEscalationSignalRegistry": _AGENT_ESCALATION_SIGNALS,
    "agent_escalation_signal_by_id": _AGENT_ESCALATION_SIGNALS,
    "agent_escalation_signal_registry": _AGENT_ESCALATION_SIGNALS,
    "AgentLifecycleProfile": _AGENT_LIFECYCLE,
    "AgentLifecycleRegistry": _AGENT_LIFECYCLE,
    "AgentLifecycleTransition": _AGENT_LIFECYCLE,
    "agent_lifecycle_profile_by_agent_id": _AGENT_LIFECYCLE,
    "agent_lifecycle_registry": _AGENT_LIFECYCLE,
    "agent_lifecycle_transition_by_id": _AGENT_LIFECYCLE,
    "AgentStateConflictSurface": _AGENT_STATE_SYNCHRONIZATION,
    "AgentStateConsistencyConstraint": _AGENT_STATE_SYNCHRONIZATION,
    "AgentStateStaleWarning": _AGENT_STATE_SYNCHRONIZATION,
    "AgentStateSyncCheckpoint": _AGENT_STATE_SYNCHRONIZATION,
    "AgentStateSyncProfile": _AGENT_STATE_SYNCHRONIZATION,
    "AgentStateSynchronizationRegistry": _AGENT_STATE_SYNCHRONIZATION,
    "agent_state_conflict_surface_by_id": _AGENT_STATE_SYNCHRONIZATION,
    "agent_state_sync_checkpoint_by_id": _AGENT_STATE_SYNCHRONIZATION,
    "agent_state_sync_profile_by_agent_id": _AGENT_STATE_SYNCHRONIZATION,
    "agent_state_synchronization_registry": _AGENT_STATE_SYNCHRONIZATION,
    "WorkflowAgentHandoffContract": _WORKFLOW_AGENT_HANDOFF,
    "WorkflowAgentHandoffProfile": _WORKFLOW_AGENT_HANDOFF,
    "WorkflowAgentHandoffRegistry": _WORKFLOW_AGENT_HANDOFF,
    "workflow_agent_handoff_by_id": _WORKFLOW_AGENT_HANDOFF,
    "workflow_agent_handoff_profile_by_agent_id": _WORKFLOW_AGENT_HANDOFF,
    "workflow_agent_handoff_registry": _WORKFLOW_AGENT_HANDOFF,
    "workflow_agent_handoffs_for_surface": _WORKFLOW_AGENT_HANDOFF,
    "IntegratedOrchestrationRegistryContract": (
        _ORCHESTRATION_CONTRACT_INTEGRATION
    ),
    "OrchestrationContractIntegrationRegistry": (
        _ORCHESTRATION_CONTRACT_INTEGRATION
    ),
    "integrated_orchestration_registry_by_id": (
        _ORCHESTRATION_CONTRACT_INTEGRATION
    ),
    "orchestration_contract_integration_registry": (
        _ORCHESTRATION_CONTRACT_INTEGRATION
    ),
    "AgentCapabilityProfile": _AGENT_CAPABILITY_REGISTRY,
    "AgentCapabilityRegistry": _AGENT_CAPABILITY_REGISTRY,
    "agent_capability_by_id": _AGENT_CAPABILITY_REGISTRY,
    "agent_capability_registry": _AGENT_CAPABILITY_REGISTRY,
    "AdaptiveMultiAgentEscalationProfile": _HYBRID_AGENTIC_WORKFLOW,
    "AdaptiveMultiAgentEscalationRegistry": _HYBRID_AGENTIC_WORKFLOW,
    "AgentConfidenceFusionProfile": _HYBRID_AGENTIC_WORKFLOW,
    "AgentConfidenceFusionRegistry": _HYBRID_AGENTIC_WORKFLOW,
    "AmbiguityEscalationProfile": _HYBRID_AGENTIC_WORKFLOW,
    "AmbiguityEscalationRegistry": _HYBRID_AGENTIC_WORKFLOW,
    "ConfidenceThresholdRoutingProfile": _HYBRID_AGENTIC_WORKFLOW,
    "ConfidenceThresholdRoutingRegistry": _HYBRID_AGENTIC_WORKFLOW,
    "CostThresholdRoutingProfile": _HYBRID_AGENTIC_WORKFLOW,
    "CostThresholdRoutingRegistry": _HYBRID_AGENTIC_WORKFLOW,
    "ExecutionSimulationProfile": _HYBRID_STUDIO,
    "ExecutionSimulationScope": _HYBRID_STUDIO,
    "ExecutionSimulatorRegistry": _HYBRID_STUDIO,
    "LatencyThresholdRoutingProfile": _HYBRID_AGENTIC_WORKFLOW,
    "LatencyThresholdRoutingRegistry": _HYBRID_AGENTIC_WORKFLOW,
    "EscalationPolicyRule": _ESCALATION_POLICY,
    "EscalationPolicyRegistry": _ESCALATION_POLICY,
    "escalation_policy_by_id": _ESCALATION_POLICY,
    "escalation_policy_registry": _ESCALATION_POLICY,
    "EscalationPolicyAuditRecord": _ESCALATION_POLICY_AUDIT,
    "EscalationPolicyAuditRegistry": _ESCALATION_POLICY_AUDIT,
    "escalation_policy_audit_by_rule_id": _ESCALATION_POLICY_AUDIT,
    "escalation_policy_audit_registry": _ESCALATION_POLICY_AUDIT,
    "escalation_policy_audits_for_downstream_registry": (
        _ESCALATION_POLICY_AUDIT
    ),
    "ConditionalMultiAgentEscalationCondition": _HYBRID_AGENTIC_WORKFLOW,
    "ConditionalMultiAgentEscalationRegistry": _HYBRID_AGENTIC_WORKFLOW,
    "CreativeExplorationBudgetProfile": _HYBRID_AGENTIC_WORKFLOW,
    "CreativeExplorationBudgetRegistry": _HYBRID_AGENTIC_WORKFLOW,
    "CreativeEscalationPolicyRegistry": _HYBRID_AGENTIC_WORKFLOW,
    "CreativeEscalationPolicyRule": _HYBRID_AGENTIC_WORKFLOW,
    "DecisionProvenanceProfile": _HYBRID_AGENTIC_WORKFLOW,
    "DecisionProvenanceRegistry": _HYBRID_AGENTIC_WORKFLOW,
    "EscalationGateProfile": _HYBRID_AGENTIC_WORKFLOW,
    "EscalationGateRegistry": _HYBRID_AGENTIC_WORKFLOW,
    "EscalationTraceProfile": _HYBRID_AGENTIC_WORKFLOW,
    "EscalationTraceRegistry": _HYBRID_AGENTIC_WORKFLOW,
    "ExecutionReplayKind": _HYBRID_STUDIO,
    "ExecutionReplayProfile": _HYBRID_STUDIO,
    "ExecutionReplayRegistry": _HYBRID_STUDIO,
    "HitlEscalationGateProfile": _HYBRID_AGENTIC_WORKFLOW,
    "HitlEscalationGateRegistry": _HYBRID_AGENTIC_WORKFLOW,
    "HybridAgentDebateLoopProfile": _HYBRID_AGENTIC_WORKFLOW,
    "HybridAgentDebateLoopRegistry": _HYBRID_AGENTIC_WORKFLOW,
    "HybridAgentVotingProfile": _HYBRID_AGENTIC_WORKFLOW,
    "HybridAgentVotingRegistry": _HYBRID_AGENTIC_WORKFLOW,
    "HybridAgenticWorkflowRegistry": _HYBRID_AGENTIC_WORKFLOW,
    "HybridAgenticWorkflowStage": _HYBRID_AGENTIC_WORKFLOW,
    "HybridWorkflowAuditRecord": _HYBRID_WORKFLOW_AUDIT,
    "HybridWorkflowAuditRegistry": _HYBRID_WORKFLOW_AUDIT,
    "AutoModePosture": _HYBRID_STUDIO,
    "AutoModeProfile": _HYBRID_STUDIO,
    "AutoModeRegistry": _HYBRID_STUDIO,
    "HybridExecutionProfile": _HYBRID_STUDIO,
    "HybridExecutionRegistry": _HYBRID_STUDIO,
    "HybridExecutionStrategy": _HYBRID_STUDIO,
    "HybridStudioIntegrationKind": _HYBRID_STUDIO,
    "HybridStudioIntegrationProfile": _HYBRID_STUDIO,
    "HybridStudioIntegrationRegistry": _HYBRID_STUDIO,
    "ArtifactCollaborationProfile": _MULTIMODAL_STUDIO,
    "ArtifactCollaborationProfileKind": _MULTIMODAL_STUDIO,
    "ArtifactCollaborationSurfaceKind": _MULTIMODAL_STUDIO,
    "ArtifactLineageProfile": _MULTIMODAL_STUDIO,
    "ArtifactLineageProfileKind": _MULTIMODAL_STUDIO,
    "ArtifactLineageSurfaceKind": _MULTIMODAL_STUDIO,
    "ArtifactProvenanceProfile": _MULTIMODAL_STUDIO,
    "ArtifactProvenanceProfileKind": _MULTIMODAL_STUDIO,
    "ArtifactProvenanceSurfaceKind": _MULTIMODAL_STUDIO,
    "BranchingTimelineProfile": _MULTIMODAL_STUDIO,
    "BranchingTimelineProfileKind": _MULTIMODAL_STUDIO,
    "BranchingTimelineSurfaceKind": _MULTIMODAL_STUDIO,
    "CreativeEvolutionTimelineProfile": _MULTIMODAL_STUDIO,
    "CreativeEvolutionTimelineProfileKind": _MULTIMODAL_STUDIO,
    "CreativeEvolutionTimelineSurfaceKind": _MULTIMODAL_STUDIO,
    "CrossAgentWorkspaceProfile": _MULTIMODAL_STUDIO,
    "CrossAgentWorkspaceProfileKind": _MULTIMODAL_STUDIO,
    "CrossAgentWorkspaceSurfaceKind": _MULTIMODAL_STUDIO,
    "InteractiveCanvasProfile": _MULTIMODAL_STUDIO,
    "InteractiveCanvasProfileKind": _MULTIMODAL_STUDIO,
    "InteractiveCanvasSurfaceKind": _MULTIMODAL_STUDIO,
    "LivePreviewProfile": _MULTIMODAL_STUDIO,
    "LivePreviewProfileKind": _MULTIMODAL_STUDIO,
    "MultiPreviewLayoutKind": _MULTIMODAL_STUDIO,
    "MultiPreviewOutputKind": _MULTIMODAL_STUDIO,
    "MultiPreviewProfile": _MULTIMODAL_STUDIO,
    "MultiPreviewProfileKind": _MULTIMODAL_STUDIO,
    "MultimodalArtifactCollaborationRegistry": _MULTIMODAL_STUDIO,
    "MultimodalArtifactLineageRegistry": _MULTIMODAL_STUDIO,
    "MultimodalArtifactProvenanceRegistry": _MULTIMODAL_STUDIO,
    "MultimodalBranchingTimelineRegistry": _MULTIMODAL_STUDIO,
    "MultimodalCreativeEvolutionTimelineRegistry": _MULTIMODAL_STUDIO,
    "MultimodalCrossAgentWorkspaceRegistry": _MULTIMODAL_STUDIO,
    "MultimodalInteractiveCanvasRegistry": _MULTIMODAL_STUDIO,
    "MultimodalLivePreviewRegistry": _MULTIMODAL_STUDIO,
    "MultimodalMultiPreviewRegistry": _MULTIMODAL_STUDIO,
    "MultimodalRealTimeWorkflowVisualizationRegistry": _MULTIMODAL_STUDIO,
    "MultimodalRuntimeCollaborationRegistry": _MULTIMODAL_STUDIO,
    "MultimodalSharedArtifactBoardRegistry": _MULTIMODAL_STUDIO,
    "MultimodalStudioIntegrationKind": _MULTIMODAL_STUDIO,
    "MultimodalStudioIntegrationProfile": _MULTIMODAL_STUDIO,
    "MultimodalStudioIntegrationRegistry": _MULTIMODAL_STUDIO,
    "MultimodalVisualWorkspaceRegistry": _MULTIMODAL_STUDIO,
    "MultimodalWorkspaceHistoryRegistry": _MULTIMODAL_STUDIO,
    "RealTimeWorkflowVisualizationProfile": _MULTIMODAL_STUDIO,
    "RealTimeWorkflowVisualizationProfileKind": _MULTIMODAL_STUDIO,
    "RealTimeWorkflowVisualizationSurfaceKind": _MULTIMODAL_STUDIO,
    "RuntimeCollaborationProfile": _MULTIMODAL_STUDIO,
    "RuntimeCollaborationProfileKind": _MULTIMODAL_STUDIO,
    "RuntimeCollaborationSurfaceKind": _MULTIMODAL_STUDIO,
    "SharedArtifactBoardProfile": _MULTIMODAL_STUDIO,
    "SharedArtifactBoardProfileKind": _MULTIMODAL_STUDIO,
    "SharedArtifactBoardSurfaceKind": _MULTIMODAL_STUDIO,
    "VisualWorkspaceProfile": _MULTIMODAL_STUDIO,
    "VisualWorkspaceProfileKind": _MULTIMODAL_STUDIO,
    "VisualWorkspaceSurfaceKind": _MULTIMODAL_STUDIO,
    "WorkspaceHistoryProfile": _MULTIMODAL_STUDIO,
    "WorkspaceHistoryProfileKind": _MULTIMODAL_STUDIO,
    "WorkspaceHistorySurfaceKind": _MULTIMODAL_STUDIO,
    "HitlDecisionPosture": _HYBRID_STUDIO,
    "HitlDecisionProfile": _HYBRID_STUDIO,
    "HitlDecisionRegistry": _HYBRID_STUDIO,
    "CloudModelCapabilityBand": _HYBRID_STUDIO,
    "CloudModelConfigurationSource": _HYBRID_STUDIO,
    "CloudModelLatencyPosture": _HYBRID_STUDIO,
    "CloudModelProviderKind": _HYBRID_STUDIO,
    "CloudModelRegistry": _HYBRID_STUDIO,
    "CloudModelSurface": _HYBRID_STUDIO,
    "CostProfile": _HYBRID_STUDIO,
    "CostProfileBand": _HYBRID_STUDIO,
    "CostProfileKind": _HYBRID_STUDIO,
    "CostProfileRegistry": _HYBRID_STUDIO,
    "LocalModelCapabilityBand": _HYBRID_STUDIO,
    "LocalModelContextWindowBand": _HYBRID_STUDIO,
    "LocalModelExecutionSurface": _HYBRID_STUDIO,
    "LocalModelLatencyPosture": _HYBRID_STUDIO,
    "LocalModelRegistry": _HYBRID_STUDIO,
    "LocalModelRuntimeKind": _HYBRID_STUDIO,
    "LocalModelSurface": _HYBRID_STUDIO,
    "LocalCloudComparisonKind": _HYBRID_STUDIO,
    "LocalCloudComparisonProfile": _HYBRID_STUDIO,
    "LocalCloudComparisonRegistry": _HYBRID_STUDIO,
    "ModelProfile": _HYBRID_STUDIO,
    "ModelProfileKind": _HYBRID_STUDIO,
    "ModelProfileRegistry": _HYBRID_STUDIO,
    "ProviderSelectionPosture": _HYBRID_STUDIO,
    "ProviderSelectionProfile": _HYBRID_STUDIO,
    "ProviderSelectionRegistry": _HYBRID_STUDIO,
    "QualityProfile": _HYBRID_STUDIO,
    "QualityProfileKind": _HYBRID_STUDIO,
    "QualityProfileLevel": _HYBRID_STUDIO,
    "QualityProfileRegistry": _HYBRID_STUDIO,
    "SessionReplayKind": _HYBRID_STUDIO,
    "SessionReplayProfile": _HYBRID_STUDIO,
    "SessionReplayRegistry": _HYBRID_STUDIO,
    "StudioModePosture": _HYBRID_STUDIO,
    "StudioModeProfile": _HYBRID_STUDIO,
    "StudioModeRegistry": _HYBRID_STUDIO,
    "WorkspaceSnapshotKind": _HYBRID_STUDIO,
    "WorkspaceSnapshotProfile": _HYBRID_STUDIO,
    "WorkspaceSnapshotRegistry": _HYBRID_STUDIO,
    "QualityEscalationProfile": _HYBRID_AGENTIC_WORKFLOW,
    "QualityEscalationRegistry": _HYBRID_AGENTIC_WORKFLOW,
    "ReflectionEscalationProfile": _HYBRID_AGENTIC_WORKFLOW,
    "ReflectionEscalationRegistry": _HYBRID_AGENTIC_WORKFLOW,
    "ResultNormalizationProfile": _HYBRID_AGENTIC_WORKFLOW,
    "ResultNormalizationRegistry": _HYBRID_AGENTIC_WORKFLOW,
    "RiskEscalationProfile": _HYBRID_AGENTIC_WORKFLOW,
    "RiskEscalationRegistry": _HYBRID_AGENTIC_WORKFLOW,
    "ReturnToWorkflowHandoffProfile": _HYBRID_AGENTIC_WORKFLOW,
    "ReturnToWorkflowHandoffRegistry": _HYBRID_AGENTIC_WORKFLOW,
    "SpecialistAgentLoopProfile": _HYBRID_AGENTIC_WORKFLOW,
    "SpecialistAgentLoopRegistry": _HYBRID_AGENTIC_WORKFLOW,
    "V3BackboneModeNodeProfile": _HYBRID_AGENTIC_WORKFLOW,
    "V3BackboneModeRegistry": _HYBRID_AGENTIC_WORKFLOW,
    "adaptive_multi_agent_escalation_profile_by_id": _HYBRID_AGENTIC_WORKFLOW,
    "adaptive_multi_agent_escalation_registry": _HYBRID_AGENTIC_WORKFLOW,
    "agent_confidence_fusion_profile_by_id": _HYBRID_AGENTIC_WORKFLOW,
    "agent_confidence_fusion_registry": _HYBRID_AGENTIC_WORKFLOW,
    "agent_conversation_view_profile_by_id": _HYBRID_STUDIO,
    "agent_conversation_view_profiles_for_agent_id": _HYBRID_STUDIO,
    "agent_conversation_view_profiles_for_route": _HYBRID_STUDIO,
    "agent_conversation_view_profiles_for_workspace": _HYBRID_STUDIO,
    "agent_conversation_view_registry": _HYBRID_STUDIO,
    "agent_workspace_profile_by_id": _HYBRID_STUDIO,
    "agent_workspace_profiles_for_agent_id": _HYBRID_STUDIO,
    "agent_workspace_profiles_for_route": _HYBRID_STUDIO,
    "agent_workspace_registry": _HYBRID_STUDIO,
    "ambiguity_escalation_profile_by_id": _HYBRID_AGENTIC_WORKFLOW,
    "ambiguity_escalation_registry": _HYBRID_AGENTIC_WORKFLOW,
    "auto_mode_profile_by_id": _HYBRID_STUDIO,
    "auto_mode_profiles_for_route": _HYBRID_STUDIO,
    "auto_mode_registry": _HYBRID_STUDIO,
    "confidence_threshold_routing_profile_by_id": _HYBRID_AGENTIC_WORKFLOW,
    "confidence_threshold_routing_registry": _HYBRID_AGENTIC_WORKFLOW,
    "cost_profile_by_id": _HYBRID_STUDIO,
    "cost_profile_registry": _HYBRID_STUDIO,
    "cost_profiles_for_band": _HYBRID_STUDIO,
    "cost_profiles_for_route": _HYBRID_STUDIO,
    "cost_threshold_routing_profile_by_id": _HYBRID_AGENTIC_WORKFLOW,
    "cost_threshold_routing_registry": _HYBRID_AGENTIC_WORKFLOW,
    "latency_threshold_routing_profile_by_id": _HYBRID_AGENTIC_WORKFLOW,
    "latency_threshold_routing_registry": _HYBRID_AGENTIC_WORKFLOW,
    "quality_escalation_profile_by_id": _HYBRID_AGENTIC_WORKFLOW,
    "quality_escalation_registry": _HYBRID_AGENTIC_WORKFLOW,
    "quality_profile_by_id": _HYBRID_STUDIO,
    "quality_profile_registry": _HYBRID_STUDIO,
    "quality_profiles_for_level": _HYBRID_STUDIO,
    "quality_profiles_for_route": _HYBRID_STUDIO,
    "provider_selection_profile_by_id": _HYBRID_STUDIO,
    "provider_selection_profiles_for_route": _HYBRID_STUDIO,
    "provider_selection_registry": _HYBRID_STUDIO,
    "conditional_multi_agent_escalation_condition_by_id": (
        _HYBRID_AGENTIC_WORKFLOW
    ),
    "conditional_multi_agent_escalation_registry": _HYBRID_AGENTIC_WORKFLOW,
    "cloud_model_registry": _HYBRID_STUDIO,
    "cloud_model_surface_by_id": _HYBRID_STUDIO,
    "cloud_model_surfaces_for_provider": _HYBRID_STUDIO,
    "creative_exploration_budget_profile_by_id": _HYBRID_AGENTIC_WORKFLOW,
    "creative_exploration_budget_registry": _HYBRID_AGENTIC_WORKFLOW,
    "creative_escalation_policy_by_id": _HYBRID_AGENTIC_WORKFLOW,
    "creative_escalation_policy_registry": _HYBRID_AGENTIC_WORKFLOW,
    "decision_provenance_profile_by_id": _HYBRID_AGENTIC_WORKFLOW,
    "decision_provenance_registry": _HYBRID_AGENTIC_WORKFLOW,
    "escalation_gate_by_id": _HYBRID_AGENTIC_WORKFLOW,
    "escalation_gate_registry": _HYBRID_AGENTIC_WORKFLOW,
    "escalation_trace_profile_by_id": _HYBRID_AGENTIC_WORKFLOW,
    "escalation_trace_registry": _HYBRID_AGENTIC_WORKFLOW,
    "execution_simulation_profile_by_id": _HYBRID_STUDIO,
    "execution_simulation_profiles_for_route": _HYBRID_STUDIO,
    "execution_simulator_registry": _HYBRID_STUDIO,
    "execution_replay_profile_by_id": _HYBRID_STUDIO,
    "execution_replay_profiles_for_execution_simulation": _HYBRID_STUDIO,
    "execution_replay_profiles_for_route": _HYBRID_STUDIO,
    "execution_replay_profiles_for_session_replay": _HYBRID_STUDIO,
    "execution_replay_registry": _HYBRID_STUDIO,
    "hitl_escalation_gate_profile_by_id": _HYBRID_AGENTIC_WORKFLOW,
    "hitl_escalation_gate_registry": _HYBRID_AGENTIC_WORKFLOW,
    "hitl_decision_profile_by_id": _HYBRID_STUDIO,
    "hitl_decision_profiles_for_route": _HYBRID_STUDIO,
    "hitl_decision_registry": _HYBRID_STUDIO,
    "hybrid_agent_debate_loop_by_id": _HYBRID_AGENTIC_WORKFLOW,
    "hybrid_agent_debate_loop_registry": _HYBRID_AGENTIC_WORKFLOW,
    "hybrid_agent_voting_profile_by_id": _HYBRID_AGENTIC_WORKFLOW,
    "hybrid_agent_voting_registry": _HYBRID_AGENTIC_WORKFLOW,
    "hybrid_agentic_workflow_registry": _HYBRID_AGENTIC_WORKFLOW,
    "hybrid_agentic_workflow_stage_by_id": _HYBRID_AGENTIC_WORKFLOW,
    "hybrid_workflow_audit_by_stage_id": _HYBRID_WORKFLOW_AUDIT,
    "hybrid_workflow_audit_registry": _HYBRID_WORKFLOW_AUDIT,
    "hybrid_workflow_audits_for_source_registry": _HYBRID_WORKFLOW_AUDIT,
    "CreativeDiversityAuditRecord": _CREATIVE_DIVERSITY_AUDIT,
    "CreativeDiversityAuditRegistry": _CREATIVE_DIVERSITY_AUDIT,
    "creative_diversity_audit_by_profile_id": _CREATIVE_DIVERSITY_AUDIT,
    "creative_diversity_audit_registry": _CREATIVE_DIVERSITY_AUDIT,
    "creative_diversity_audits_for_posture": _CREATIVE_DIVERSITY_AUDIT,
    "creative_diversity_audits_for_source_registry": (
        _CREATIVE_DIVERSITY_AUDIT
    ),
    "AgentExplainabilityAuditRecord": _AGENT_EXPLAINABILITY_AUDIT,
    "AgentExplainabilityAuditRegistry": _AGENT_EXPLAINABILITY_AUDIT,
    "agent_explainability_audit_by_agent_id": _AGENT_EXPLAINABILITY_AUDIT,
    "agent_explainability_audit_registry": _AGENT_EXPLAINABILITY_AUDIT,
    "agent_explainability_audits_for_memory_source": (
        _AGENT_EXPLAINABILITY_AUDIT
    ),
    "agent_explainability_audits_for_source_registry": (
        _AGENT_EXPLAINABILITY_AUDIT
    ),
    "AgentReliabilityAuditRecord": _AGENT_RELIABILITY_AUDIT,
    "AgentReliabilityAuditRegistry": _AGENT_RELIABILITY_AUDIT,
    "agent_reliability_audit_by_agent_id": _AGENT_RELIABILITY_AUDIT,
    "agent_reliability_audit_registry": _AGENT_RELIABILITY_AUDIT,
    "agent_reliability_audits_for_consistency_family": (
        _AGENT_RELIABILITY_AUDIT
    ),
    "agent_reliability_audits_for_escalation_category": (
        _AGENT_RELIABILITY_AUDIT
    ),
    "AgentDeterminismAuditRecord": _AGENT_DETERMINISM_AUDIT,
    "AgentDeterminismAuditRegistry": _AGENT_DETERMINISM_AUDIT,
    "agent_determinism_audit_by_agent_id": _AGENT_DETERMINISM_AUDIT,
    "agent_determinism_audit_registry": _AGENT_DETERMINISM_AUDIT,
    "agent_determinism_audits_for_cacheability": (
        _AGENT_DETERMINISM_AUDIT
    ),
    "agent_determinism_audits_for_routing_priority_band": (
        _AGENT_DETERMINISM_AUDIT
    ),
    "agent_determinism_audits_for_scheduling_group": (
        _AGENT_DETERMINISM_AUDIT
    ),
    "AgentTelemetryFoundationProfile": _AGENT_TELEMETRY_FOUNDATION,
    "AgentTelemetryFoundationRegistry": _AGENT_TELEMETRY_FOUNDATION,
    "agent_telemetry_foundation_registry": _AGENT_TELEMETRY_FOUNDATION,
    "agent_telemetry_profile_by_agent_id": _AGENT_TELEMETRY_FOUNDATION,
    "agent_telemetry_profiles_for_dimension": _AGENT_TELEMETRY_FOUNDATION,
    "agent_telemetry_profiles_for_event_type": _AGENT_TELEMETRY_FOUNDATION,
    "AgentCostTrackingFoundationProfile": _AGENT_COST_TRACKING_FOUNDATION,
    "AgentCostTrackingFoundationRegistry": _AGENT_COST_TRACKING_FOUNDATION,
    "agent_cost_tracking_foundation_registry": (
        _AGENT_COST_TRACKING_FOUNDATION
    ),
    "agent_cost_tracking_profile_by_agent_id": _AGENT_COST_TRACKING_FOUNDATION,
    "agent_cost_tracking_profiles_for_cost_class": (
        _AGENT_COST_TRACKING_FOUNDATION
    ),
    "agent_cost_tracking_profiles_for_cost_profile": (
        _AGENT_COST_TRACKING_FOUNDATION
    ),
    "AgentPerformanceTrackingFoundationProfile": (
        _AGENT_PERFORMANCE_TRACKING_FOUNDATION
    ),
    "AgentPerformanceTrackingFoundationRegistry": (
        _AGENT_PERFORMANCE_TRACKING_FOUNDATION
    ),
    "agent_performance_tracking_foundation_registry": (
        _AGENT_PERFORMANCE_TRACKING_FOUNDATION
    ),
    "agent_performance_tracking_profile_by_agent_id": (
        _AGENT_PERFORMANCE_TRACKING_FOUNDATION
    ),
    "agent_performance_tracking_profiles_for_latency_class": (
        _AGENT_PERFORMANCE_TRACKING_FOUNDATION
    ),
    "agent_performance_tracking_profiles_for_latency_threshold": (
        _AGENT_PERFORMANCE_TRACKING_FOUNDATION
    ),
    "ArchitectureConsistencyPassRegistry": _ARCHITECTURE_CONSISTENCY_PASS,
    "ArchitectureConsistencyRecord": _ARCHITECTURE_CONSISTENCY_PASS,
    "architecture_consistency_pass_registry": _ARCHITECTURE_CONSISTENCY_PASS,
    "architecture_consistency_record_by_source_registry": (
        _ARCHITECTURE_CONSISTENCY_PASS
    ),
    "architecture_consistency_records_for_layer": (
        _ARCHITECTURE_CONSISTENCY_PASS
    ),
    "FinalV4HardeningRecord": _FINAL_V4_HARDENING,
    "FinalV4HardeningRegistry": _FINAL_V4_HARDENING,
    "LangGraphErrorPathAuditRecord": _FINAL_V4_HARDENING,
    "LangGraphErrorPathAuditRegistry": _FINAL_V4_HARDENING,
    "final_v4_hardening_registry": _FINAL_V4_HARDENING,
    "final_v4_hardening_record_by_domain_id": _FINAL_V4_HARDENING,
    "final_v4_hardening_records_for_source_registry": _FINAL_V4_HARDENING,
    "langgraph_error_path_audit_registry": _FINAL_V4_HARDENING,
    "langgraph_error_path_audit_record_by_surface_id": _FINAL_V4_HARDENING,
    "langgraph_error_path_audit_records_for_node": _FINAL_V4_HARDENING,
    "hybrid_execution_profile_by_id": _HYBRID_STUDIO,
    "hybrid_execution_profiles_for_route": _HYBRID_STUDIO,
    "hybrid_execution_registry": _HYBRID_STUDIO,
    "hybrid_studio_integration_profile_by_id": _HYBRID_STUDIO,
    "hybrid_studio_integration_profiles_for_route": _HYBRID_STUDIO,
    "hybrid_studio_integration_profiles_for_source_registry": _HYBRID_STUDIO,
    "hybrid_studio_integration_registry": _HYBRID_STUDIO,
    "local_cloud_comparison_profile_by_id": _HYBRID_STUDIO,
    "local_cloud_comparison_profiles_for_route": _HYBRID_STUDIO,
    "local_cloud_comparison_registry": _HYBRID_STUDIO,
    "local_model_registry": _HYBRID_STUDIO,
    "local_model_surface_by_id": _HYBRID_STUDIO,
    "local_model_surfaces_for_runtime": _HYBRID_STUDIO,
    "model_profile_by_id": _HYBRID_STUDIO,
    "model_profile_registry": _HYBRID_STUDIO,
    "model_profiles_for_route": _HYBRID_STUDIO,
    "multimodal_artifact_collaboration_profile_by_id": _MULTIMODAL_STUDIO,
    "multimodal_artifact_collaboration_profiles_for_route": _MULTIMODAL_STUDIO,
    "multimodal_artifact_collaboration_profiles_for_surface_kind": (
        _MULTIMODAL_STUDIO
    ),
    "multimodal_artifact_collaboration_profiles_for_workspace_profile": (
        _MULTIMODAL_STUDIO
    ),
    "multimodal_artifact_collaboration_registry": _MULTIMODAL_STUDIO,
    "multimodal_artifact_lineage_profile_by_id": _MULTIMODAL_STUDIO,
    "multimodal_artifact_lineage_profiles_for_artifact_provenance_profile": (
        _MULTIMODAL_STUDIO
    ),
    "multimodal_artifact_lineage_profiles_for_route": _MULTIMODAL_STUDIO,
    "multimodal_artifact_lineage_profiles_for_surface_kind": (
        _MULTIMODAL_STUDIO
    ),
    "multimodal_artifact_lineage_registry": _MULTIMODAL_STUDIO,
    "multimodal_artifact_provenance_profile_by_id": _MULTIMODAL_STUDIO,
    "multimodal_artifact_provenance_profiles_for_artifact_collaboration_profile": (
        _MULTIMODAL_STUDIO
    ),
    "multimodal_artifact_provenance_profiles_for_route": _MULTIMODAL_STUDIO,
    "multimodal_artifact_provenance_profiles_for_surface_kind": (
        _MULTIMODAL_STUDIO
    ),
    "multimodal_artifact_provenance_registry": _MULTIMODAL_STUDIO,
    "multimodal_branching_timeline_profile_by_id": _MULTIMODAL_STUDIO,
    "multimodal_branching_timeline_profiles_for_route": _MULTIMODAL_STUDIO,
    "multimodal_branching_timeline_profiles_for_surface_kind": (
        _MULTIMODAL_STUDIO
    ),
    "multimodal_branching_timeline_profiles_for_workspace_history_profile": (
        _MULTIMODAL_STUDIO
    ),
    "multimodal_branching_timeline_registry": _MULTIMODAL_STUDIO,
    "multimodal_creative_evolution_timeline_profile_by_id": (
        _MULTIMODAL_STUDIO
    ),
    "multimodal_creative_evolution_timeline_profiles_for_branching_timeline_profile": (
        _MULTIMODAL_STUDIO
    ),
    "multimodal_creative_evolution_timeline_profiles_for_route": (
        _MULTIMODAL_STUDIO
    ),
    "multimodal_creative_evolution_timeline_profiles_for_surface_kind": (
        _MULTIMODAL_STUDIO
    ),
    "multimodal_creative_evolution_timeline_registry": _MULTIMODAL_STUDIO,
    "multimodal_cross_agent_workspace_profile_by_id": _MULTIMODAL_STUDIO,
    "multimodal_cross_agent_workspace_profiles_for_agent_workspace_profile": (
        _MULTIMODAL_STUDIO
    ),
    "multimodal_cross_agent_workspace_profiles_for_route": _MULTIMODAL_STUDIO,
    "multimodal_cross_agent_workspace_profiles_for_surface_kind": (
        _MULTIMODAL_STUDIO
    ),
    "multimodal_cross_agent_workspace_registry": _MULTIMODAL_STUDIO,
    "multimodal_interactive_canvas_profile_by_id": _MULTIMODAL_STUDIO,
    "multimodal_interactive_canvas_profiles_for_live_preview_profile": (
        _MULTIMODAL_STUDIO
    ),
    "multimodal_interactive_canvas_profiles_for_multi_preview_profile": (
        _MULTIMODAL_STUDIO
    ),
    "multimodal_interactive_canvas_profiles_for_route": _MULTIMODAL_STUDIO,
    "multimodal_interactive_canvas_profiles_for_surface_kind": (
        _MULTIMODAL_STUDIO
    ),
    "multimodal_interactive_canvas_registry": _MULTIMODAL_STUDIO,
    "multimodal_live_preview_profile_by_id": _MULTIMODAL_STUDIO,
    "multimodal_live_preview_profiles_for_route": _MULTIMODAL_STUDIO,
    "multimodal_live_preview_profiles_for_source_reference": (
        _MULTIMODAL_STUDIO
    ),
    "multimodal_live_preview_profiles_for_target": _MULTIMODAL_STUDIO,
    "multimodal_live_preview_registry": _MULTIMODAL_STUDIO,
    "multimodal_multi_preview_profile_by_id": _MULTIMODAL_STUDIO,
    "multimodal_multi_preview_profiles_for_layout": _MULTIMODAL_STUDIO,
    "multimodal_multi_preview_profiles_for_live_preview_profile": (
        _MULTIMODAL_STUDIO
    ),
    "multimodal_multi_preview_profiles_for_route": _MULTIMODAL_STUDIO,
    "multimodal_multi_preview_registry": _MULTIMODAL_STUDIO,
    "multimodal_runtime_collaboration_profile_by_id": _MULTIMODAL_STUDIO,
    "multimodal_runtime_collaboration_profiles_for_route": _MULTIMODAL_STUDIO,
    "multimodal_runtime_collaboration_profiles_for_surface_kind": (
        _MULTIMODAL_STUDIO
    ),
    "multimodal_runtime_collaboration_profiles_for_visual_workspace_profile": (
        _MULTIMODAL_STUDIO
    ),
    "multimodal_runtime_collaboration_registry": _MULTIMODAL_STUDIO,
    "multimodal_real_time_workflow_visualization_profile_by_id": (
        _MULTIMODAL_STUDIO
    ),
    "multimodal_real_time_workflow_visualization_profiles_for_creative_evolution_timeline_profile": (
        _MULTIMODAL_STUDIO
    ),
    "multimodal_real_time_workflow_visualization_profiles_for_route": (
        _MULTIMODAL_STUDIO
    ),
    "multimodal_real_time_workflow_visualization_profiles_for_surface_kind": (
        _MULTIMODAL_STUDIO
    ),
    "multimodal_real_time_workflow_visualization_registry": _MULTIMODAL_STUDIO,
    "multimodal_shared_artifact_board_profile_by_id": _MULTIMODAL_STUDIO,
    "multimodal_shared_artifact_board_profiles_for_cross_agent_workspace_profile": (
        _MULTIMODAL_STUDIO
    ),
    "multimodal_shared_artifact_board_profiles_for_route": _MULTIMODAL_STUDIO,
    "multimodal_shared_artifact_board_profiles_for_surface_kind": (
        _MULTIMODAL_STUDIO
    ),
    "multimodal_shared_artifact_board_registry": _MULTIMODAL_STUDIO,
    "multimodal_studio_integration_profile_by_id": _MULTIMODAL_STUDIO,
    "multimodal_studio_integration_profiles_for_route": _MULTIMODAL_STUDIO,
    "multimodal_studio_integration_profiles_for_source_registry": (
        _MULTIMODAL_STUDIO
    ),
    "multimodal_studio_integration_registry": _MULTIMODAL_STUDIO,
    "multimodal_visual_workspace_profile_by_id": _MULTIMODAL_STUDIO,
    "multimodal_visual_workspace_profiles_for_interactive_canvas_profile": (
        _MULTIMODAL_STUDIO
    ),
    "multimodal_visual_workspace_profiles_for_route": _MULTIMODAL_STUDIO,
    "multimodal_visual_workspace_profiles_for_surface_kind": _MULTIMODAL_STUDIO,
    "multimodal_visual_workspace_registry": _MULTIMODAL_STUDIO,
    "multimodal_workspace_history_profile_by_id": _MULTIMODAL_STUDIO,
    "multimodal_workspace_history_profiles_for_route": _MULTIMODAL_STUDIO,
    "multimodal_workspace_history_profiles_for_surface_kind": _MULTIMODAL_STUDIO,
    "multimodal_workspace_history_profiles_for_workspace_snapshot_profile": (
        _MULTIMODAL_STUDIO
    ),
    "multimodal_workspace_history_registry": _MULTIMODAL_STUDIO,
    "reflection_escalation_profile_by_id": _HYBRID_AGENTIC_WORKFLOW,
    "reflection_escalation_registry": _HYBRID_AGENTIC_WORKFLOW,
    "result_normalization_profile_by_id": _HYBRID_AGENTIC_WORKFLOW,
    "result_normalization_registry": _HYBRID_AGENTIC_WORKFLOW,
    "risk_escalation_profile_by_id": _HYBRID_AGENTIC_WORKFLOW,
    "risk_escalation_registry": _HYBRID_AGENTIC_WORKFLOW,
    "return_to_workflow_handoff_profile_by_id": _HYBRID_AGENTIC_WORKFLOW,
    "return_to_workflow_handoff_registry": _HYBRID_AGENTIC_WORKFLOW,
    "session_replay_profile_by_id": _HYBRID_STUDIO,
    "session_replay_profiles_for_conversation_view": _HYBRID_STUDIO,
    "session_replay_profiles_for_route": _HYBRID_STUDIO,
    "session_replay_profiles_for_workspace_snapshot": _HYBRID_STUDIO,
    "session_replay_registry": _HYBRID_STUDIO,
    "specialist_agent_loop_by_id": _HYBRID_AGENTIC_WORKFLOW,
    "specialist_agent_loop_registry": _HYBRID_AGENTIC_WORKFLOW,
    "studio_mode_profile_by_id": _HYBRID_STUDIO,
    "studio_mode_profiles_for_route": _HYBRID_STUDIO,
    "studio_mode_registry": _HYBRID_STUDIO,
    "workspace_snapshot_profile_by_id": _HYBRID_STUDIO,
    "workspace_snapshot_profiles_for_conversation_view": _HYBRID_STUDIO,
    "workspace_snapshot_profiles_for_route": _HYBRID_STUDIO,
    "workspace_snapshot_profiles_for_workspace": _HYBRID_STUDIO,
    "workspace_snapshot_registry": _HYBRID_STUDIO,
    "v3_backbone_mode_profile_by_node_id": _HYBRID_AGENTIC_WORKFLOW,
    "v3_backbone_mode_registry": _HYBRID_AGENTIC_WORKFLOW,
    "EngineContractConsistencyRegistry": _ENGINE_CONTRACT_CONSISTENCY,
    "EngineContractFamilyConsistencyProfile": _ENGINE_CONTRACT_CONSISTENCY,
    "engine_contract_consistency_registry": _ENGINE_CONTRACT_CONSISTENCY,
    "engine_contract_family_consistency_by_id": (
        _ENGINE_CONTRACT_CONSISTENCY
    ),
    "ArtifactCapabilityConfidence": _ARTIFACT_CAPABILITY_MATRIX,
    "ArtifactCapabilityFit": _ARTIFACT_CAPABILITY_MATRIX,
    "ArtifactCapabilityMatrix": _ARTIFACT_CAPABILITY_MATRIX,
    "ArtifactCapabilityProfile": _ARTIFACT_CAPABILITY_MATRIX,
    "ArtifactStrategyAction": _MULTI_ARTIFACT_STRATEGY,
    "ArtifactStrategyArtifact": _MULTI_ARTIFACT_STRATEGY,
    "ArtifactStrategyCombinationMode": _MULTI_ARTIFACT_STRATEGY,
    "ArtifactStrategyGroup": _MULTI_ARTIFACT_STRATEGY,
    "ArtifactStrategyPriority": _MULTI_ARTIFACT_STRATEGY,
    "ArtifactStrategyPriorityEntry": _MULTI_ARTIFACT_STRATEGY,
    "ArtifactStrategyRole": _MULTI_ARTIFACT_STRATEGY,
    "ArtifactStrategySequenceStep": _MULTI_ARTIFACT_STRATEGY,
    "MultiArtifactStrategy": _MULTI_ARTIFACT_STRATEGY,
    "ArtifactCriticProfile": _ARTIFACT_CRITIC,
    "ArtifactCriticRiskAssessment": _ARTIFACT_CRITIC,
    "ArtifactRefinementFocus": _ARTIFACT_REFINER,
    "ArtifactRefinerProfile": _ARTIFACT_REFINER,
    "DependencyNodeStatus": _ARTIFACT_DEPENDENCY_GRAPH,
    "DependencyNodeType": _ARTIFACT_DEPENDENCY_GRAPH,
    "DependencyRelationship": _ARTIFACT_DEPENDENCY_GRAPH,
    "DependencyStrength": _ARTIFACT_DEPENDENCY_GRAPH,
    "ArtifactCritiqueDimension": _ARTIFACTS,
    "ArtifactCritiqueSummary": _ARTIFACT_CRITIQUE,
    "ArtifactFamily": _ARTIFACT_PLANNER,
    "ArtifactPlan": _ARTIFACT_PLANNER,
    "ArtifactType": _ARTIFACT_PLANNER,
    "AudioReactiveGuidance": _AUDIO_REACTIVE,
    "AudioReactiveIntensity": _AUDIO_REACTIVE,
    "AudioReactiveMapping": _AUDIO_REACTIVE,
    "AudioReactiveSource": _AUDIO_REACTIVE,
    "AudioReactiveVisualTarget": _AUDIO_REACTIVE,
    "AssistantService": _SERVICE,
    "AssembledContextRequest": _CTX,
    "AssembledContextResponse": _CTX,
    "AssembledContextSummary": _CTX,
    "ChromaMemoryAdapter": _MEM,
    "ContextAssembler": _CTX,
    "ConversationSummaryContext": _MEM,
    "CalibratedQualityEvaluation": _ARTIFACTS,
    "CalibratedQualitySignal": _ARTIFACTS,
    "ClarificationQuestion": _CLARIFICATION,
    "ClarificationReason": _CLARIFICATION,
    "ClarificationRequest": _CLARIFICATION,
    "CreativeOutputModality": _CREATIVE_TRANSLATION,
    "CreativeExecutionPlan": _CREATIVE_PLANNING,
    "CreativeHierarchyPlan": _CREATIVE_HIERARCHY,
    "CreativeHierarchyPriority": _CREATIVE_HIERARCHY,
    "CreativeIntentDecomposition": _CREATIVE_INTENT,
    "CreativeIntentDimension": _CREATIVE_INTENT,
    "CreativeConstraint": _CREATIVE_CONSTRAINTS,
    "CreativeConstraintPrioritization": _CREATIVE_CONSTRAINT_PRIORITIES,
    "CreativeConstraintPriority": _CREATIVE_CONSTRAINT_PRIORITIES,
    "CreativeConstraintPriorityConflict": _CREATIVE_CONSTRAINT_PRIORITIES,
    "CreativeConstraintSolution": _CREATIVE_CONSTRAINTS,
    "CreativeConstraintTradeoff": _CREATIVE_CONSTRAINTS,
    "CreativeStrategyAlternative": _CREATIVE_STRATEGY,
    "CreativeStrategyProfile": _CREATIVE_STRATEGY,
    "CreativeTechniqueAlternative": _CREATIVE_TECHNIQUE,
    "CreativeTechniqueProfile": _CREATIVE_TECHNIQUE,
    "CreativeTradeoff": _CREATIVE_TRADEOFFS,
    "CreativeTradeoffProfile": _CREATIVE_TRADEOFFS,
    "CreativeQualityPrediction": _CREATIVE_QUALITY_PREDICTION,
    "CreativeQualitySignal": _CREATIVE_QUALITY_PREDICTION,
    "CompositionPattern": _CREATIVE_COMPOSITION,
    "CreativeCompositionPlan": _CREATIVE_COMPOSITION,
    "ProceduralComplexityLevel": _PROCEDURAL_STRUCTURE,
    "ProceduralFamily": _PROCEDURAL_STRUCTURE,
    "ProceduralStructureChoice": _PROCEDURAL_STRUCTURE,
    "ProceduralStructurePlan": _PROCEDURAL_STRUCTURE,
    "GenerativeArchitecture": _GENERATIVE_STRUCTURE,
    "GenerativeEvolutionPhase": _GENERATIVE_STRUCTURE,
    "GenerativeEvolutionRule": _GENERATIVE_STRUCTURE,
    "GenerativeEvolutionTrigger": _GENERATIVE_STRUCTURE,
    "GenerativeFallbackBlueprint": _GENERATIVE_STRUCTURE,
    "GenerativeHookType": _GENERATIVE_STRUCTURE,
    "GenerativeModule": _GENERATIVE_STRUCTURE,
    "GenerativeModuleKind": _GENERATIVE_STRUCTURE,
    "GenerativeModuleRelationship": _GENERATIVE_STRUCTURE,
    "GenerativeParameter": _GENERATIVE_STRUCTURE,
    "GenerativeParameterRole": _GENERATIVE_STRUCTURE,
    "GenerativeParameterValueType": _GENERATIVE_STRUCTURE,
    "GenerativeRelationshipType": _GENERATIVE_STRUCTURE,
    "GenerativeStructureBlueprint": _GENERATIVE_STRUCTURE,
    "GenerativeStructureHook": _GENERATIVE_STRUCTURE,
    "SemanticMotif": _SEMANTIC_MOTIF,
    "SemanticMotifCompositionMapping": _SEMANTIC_MOTIF,
    "SemanticMotifFallbackPlan": _SEMANTIC_MOTIF,
    "SemanticMotifHierarchyLevel": _SEMANTIC_MOTIF,
    "SemanticMotifId": _SEMANTIC_MOTIF,
    "SemanticMotifNarrativeMapping": _SEMANTIC_MOTIF,
    "SemanticMotifParameterMapping": _SEMANTIC_MOTIF,
    "SemanticMotifRole": _SEMANTIC_MOTIF,
    "SemanticMotifStructureMapping": _SEMANTIC_MOTIF,
    "SemanticMotifSystem": _SEMANTIC_MOTIF,
    "EmotionalCompositionMapping": _EMOTIONAL_CONSISTENCY,
    "EmotionalConsistencyProfile": _EMOTIONAL_CONSISTENCY,
    "EmotionalFallbackStrategy": _EMOTIONAL_CONSISTENCY,
    "EmotionalIntensity": _EMOTIONAL_CONSISTENCY,
    "EmotionalMotifMapping": _EMOTIONAL_CONSISTENCY,
    "EmotionalNarrativeMapping": _EMOTIONAL_CONSISTENCY,
    "EmotionalParameterMapping": _EMOTIONAL_CONSISTENCY,
    "EmotionalPhaseMapping": _EMOTIONAL_CONSISTENCY,
    "EmotionalStructureMapping": _EMOTIONAL_CONSISTENCY,
    "EmotionalTone": _EMOTIONAL_CONSISTENCY,
    "CrossModalityChannel": _CROSS_MODALITY,
    "CrossModalityCompositionProfile": _CROSS_MODALITY,
    "CrossModalityFallbackStrategy": _CROSS_MODALITY,
    "CrossModalityMapping": _CROSS_MODALITY,
    "CrossModalityPattern": _CROSS_MODALITY,
    "CrossModalityRole": _CROSS_MODALITY,
    "CrossModalityTemporalCue": _CROSS_MODALITY,
    "AudioVisualCueType": _AUDIO_VISUAL_SCENE,
    "AudioVisualFallbackSceneStrategy": _AUDIO_VISUAL_SCENE,
    "AudioVisualSceneCue": _AUDIO_VISUAL_SCENE,
    "AudioVisualScenePattern": _AUDIO_VISUAL_SCENE,
    "AudioVisualScenePhase": _AUDIO_VISUAL_SCENE,
    "AudioVisualSceneProfile": _AUDIO_VISUAL_SCENE,
    "AudioVisualSceneTransition": _AUDIO_VISUAL_SCENE,
    "NarrativeArchetype": _SYMBOLIC_NARRATIVE,
    "NarrativePhaseName": _SYMBOLIC_NARRATIVE,
    "SymbolicNarrativePhase": _SYMBOLIC_NARRATIVE,
    "SymbolicNarrativePlan": _SYMBOLIC_NARRATIVE,
    "CreativeReasoningEvidence": _CREATIVE_REASONING,
    "CreativeReasoningResult": _CREATIVE_REASONING,
    "CreativeReasoningStep": _CREATIVE_REASONING,
    "CreativeRejectedAlternative": _CREATIVE_REASONING,
    "CreativeCriticProfile": _CREATIVE_CRITIC_ENGINE,
    "CreativeCriticRiskAssessment": _CREATIVE_CRITIC_ENGINE,
    "CreativeConfidenceComponent": _CREATIVE_CONFIDENCE_ENGINE,
    "CreativeConfidenceProfile": _CREATIVE_CONFIDENCE_ENGINE,
    "ReflectionLoopProfile": _REFLECTION_LOOP_ENGINE,
    "RuntimeCapabilityCandidate": _RUNTIME_CAPABILITIES,
    "RuntimeCompatibilityAssessment": _RUNTIME_COMPATIBILITY,
    "RuntimeCompatibilityConfidence": _RUNTIME_COMPATIBILITY,
    "RuntimeCompatibilityLevel": _RUNTIME_COMPATIBILITY,
    "RuntimeCompatibilityProfile": _RUNTIME_COMPATIBILITY,
    "RuntimeInteroperability": _RUNTIME_COMPATIBILITY,
    "RuntimePortability": _RUNTIME_COMPATIBILITY,
    "RuntimeCapabilityProfile": _RUNTIME_CAPABILITIES,
    "CreativeAssistantDirectorBrief": _CREATIVE_DIRECTOR,
    "CreativeQualityEvaluation": _ARTIFACTS,
    "CreativeQualityObservation": _ARTIFACTS,
    "CreativeTranslation": _CREATIVE_TRANSLATION,
    "DEFAULT_REFINEMENT_PASS_LIMIT": _REFINEMENT_PASSES,
    "MAX_REFINEMENT_PASS_LIMIT": _REFINEMENT_PASSES,
    "QUALITY_IMPROVEMENT_THRESHOLD": _REFINEMENT_PASSES,
    "RefinementPassDecision": _REFINEMENT_PASSES,
    "RefinementPassRecord": _ARTIFACTS,
    "RefinementPassStopReason": _ARTIFACTS,
    "ReferenceFusionGuidance": _REFERENCE_FUSION,
    "SacredConsistencyEvaluation": _ARTIFACTS,
    "SacredConsistencyObservation": _ARTIFACTS,
    "SacredGeometryGuidance": _SACRED_GEOMETRY,
    "ShaderPresetGuidance": _SHADER_PRESETS,
    "ShaderPresetId": _SHADER_PRESETS,
    "VisualStyleGuidance": _VISUAL_STYLES,
    "VisualStyleId": _VISUAL_STYLES,
    "DEFAULT_RECENT_TURN_LIMIT": _MEM,
    "DEFAULT_RETRIEVAL_LIMIT": _RETRIEVAL,
    "DomainSelectionShape": _ROUTING,
    "KnowledgeBaseRetrievalAdapter": _RETRIEVAL,
    "LlmGenerationAdapter": _GEN,
    "MemoryContextRequest": _MEM,
    "MemoryContextResponse": _MEM,
    "MemoryContextSource": _MEM,
    "MemoryGateway": _MEM,
    "OrchestrationContextAssembler": _CTX,
    "ProjectMemoryContext": _MEM,
    "PromptConversationTurnInput": _PROMPT_INPUTS,
    "PromptArtifactRefinementInput": _PROMPT_INPUTS,
    "PromptImageReferenceInput": _PROMPT_INPUTS,
    "PromptInputBuilder": _PROMPT_INPUTS,
    "PromptInputRequest": _PROMPT_INPUTS,
    "PromptInputResponse": _PROMPT_INPUTS,
    "PromptKnowledgeChunkInput": _PROMPT_INPUTS,
    "PromptMemoryInput": _PROMPT_INPUTS,
    "PromptProjectMemoryInput": _PROMPT_INPUTS,
    "PromptRetrievalInput": _PROMPT_INPUTS,
    "PromptRenderer": _PROMPT_TEMPLATES,
    "PromptRunningSummaryInput": _PROMPT_INPUTS,
    "PromptUserInput": _PROMPT_INPUTS,
    "ProviderGenerationGateway": _GEN,
    "ProviderGenerationRequest": _GEN,
    "RecentConversationTurn": _MEM,
    "RenderedPromptRequest": _PROMPT_TEMPLATES,
    "RenderedPromptResponse": _PROMPT_TEMPLATES,
    "RenderedPromptRole": _PROMPT_TEMPLATES,
    "RenderedPromptSection": _PROMPT_TEMPLATES,
    "RenderedPromptSectionName": _PROMPT_TEMPLATES,
    "RetrievalContextFilter": _RETRIEVAL,
    "RetrievalContextRequest": _RETRIEVAL,
    "RetrievalContextResponse": _RETRIEVAL,
    "RetrievalContextSource": _RETRIEVAL,
    "RetrievalGateway": _RETRIEVAL,
    "RetrievedKnowledgeChunk": _RETRIEVAL,
    "RouteCapability": _ROUTING,
    "RouteDecision": _ROUTING,
    "RouteName": _ROUTING,
    "StreamEventBuilder": _EVENTS,
    "StructuredPromptInputBuilder": _PROMPT_INPUTS,
    "WORKFLOW_STEP_ORDER": _WORKFLOW,
    "MAX_WORKFLOW_REFINEMENT_COUNT": _WORKFLOW_REVIEW,
    "WorkflowEventMetadata": _WORKFLOW,
    "WorkflowFailureInfo": _WORKFLOW,
    "WorkflowReviewOutcome": _WORKFLOW_REVIEW,
    "WorkflowReviewResult": _WORKFLOW_REVIEW,
    "WorkflowStatus": _WORKFLOW,
    "WorkflowStep": _WORKFLOW,
    "WorkflowArtifact": _ARTIFACTS,
    "WorkflowArtifactCritique": _ARTIFACTS,
    "JinjaPromptRenderer": _PROMPT_TEMPLATES,
    "begin_assistant_workflow": _WORKFLOW,
    "analyze_assistant_execution_graph": _EXECUTION_GRAPH_ANALYZER,
    "analyze_workflow_cost": _WORKFLOW_COST_ANALYZER,
    "build_assistant_workflow_graph": _WORKFLOW_GRAPH,
    "calibrate_artifact_quality": _QUALITY_CALIBRATION,
    "artifact_quality_score": _REFINEMENT_PASSES,
    "attach_refinement_history": _REFINEMENT_PASSES,
    "build_initial_workflow_graph_state": _WORKFLOW_GRAPH,
    "build_refinement_objective": _REFINEMENT_PASSES,
    "build_assembled_context_request": _CTX,
    "build_memory_context_request": _MEM,
    "build_prompt_input_request": _PROMPT_INPUTS,
    "build_provider_generation_request": _GEN,
    "build_rendered_prompt_request": _PROMPT_TEMPLATES,
    "build_retrieval_context_request": _RETRIEVAL,
    "complete_workflow_step": _WORKFLOW,
    "execution_graph_edges_from": _EXECUTION_GRAPH_ANALYZER,
    "execution_graph_edges_to": _EXECUTION_GRAPH_ANALYZER,
    "execution_graph_node_by_id": _EXECUTION_GRAPH_ANALYZER,
    "workflow_cost_component_by_id": _WORKFLOW_COST_ANALYZER,
    "workflow_cost_components_for_kind": _WORKFLOW_COST_ANALYZER,
    "complete_latest_refinement_pass": _REFINEMENT_PASSES,
    "critique_workflow_artifacts": _ARTIFACT_CRITIQUE,
    "creative_translation_prompt_lines": _CREATIVE_TRANSLATION,
    "audio_reactive_prompt_lines": _AUDIO_REACTIVE,
    "derive_audio_reactive_guidance": _AUDIO_REACTIVE,
    "derive_creative_translation": _CREATIVE_TRANSLATION,
    "derive_creative_hierarchy_plan": _CREATIVE_HIERARCHY,
    "creative_hierarchy_plan_prompt_lines": _CREATIVE_HIERARCHY,
    "derive_creative_intent_decomposition": _CREATIVE_INTENT,
    "creative_intent_decomposition_prompt_lines": _CREATIVE_INTENT,
    "derive_creative_execution_plan": _CREATIVE_PLANNING,
    "creative_execution_plan_prompt_lines": _CREATIVE_PLANNING,
    "derive_creative_constraint_solution": _CREATIVE_CONSTRAINTS,
    "creative_constraint_solution_prompt_lines": _CREATIVE_CONSTRAINTS,
    "derive_creative_constraint_priorities": _CREATIVE_CONSTRAINT_PRIORITIES,
    "creative_constraint_priorities_prompt_lines": _CREATIVE_CONSTRAINT_PRIORITIES,
    "derive_creative_strategy_profile": _CREATIVE_STRATEGY,
    "creative_strategy_prompt_lines": _CREATIVE_STRATEGY,
    "derive_creative_technique_profile": _CREATIVE_TECHNIQUE,
    "creative_technique_prompt_lines": _CREATIVE_TECHNIQUE,
    "derive_creative_tradeoff_profile": _CREATIVE_TRADEOFFS,
    "creative_tradeoff_prompt_lines": _CREATIVE_TRADEOFFS,
    "derive_creative_quality_prediction": _CREATIVE_QUALITY_PREDICTION,
    "creative_quality_prediction_prompt_lines": _CREATIVE_QUALITY_PREDICTION,
    "derive_creative_composition_plan": _CREATIVE_COMPOSITION,
    "creative_composition_prompt_lines": _CREATIVE_COMPOSITION,
    "derive_procedural_structure_plan": _PROCEDURAL_STRUCTURE,
    "procedural_structure_prompt_lines": _PROCEDURAL_STRUCTURE,
    "derive_generative_structure_blueprint": _GENERATIVE_STRUCTURE,
    "generative_structure_prompt_lines": _GENERATIVE_STRUCTURE,
    "derive_semantic_motif_system": _SEMANTIC_MOTIF,
    "semantic_motif_prompt_lines": _SEMANTIC_MOTIF,
    "derive_emotional_consistency_profile": _EMOTIONAL_CONSISTENCY,
    "emotional_consistency_prompt_lines": _EMOTIONAL_CONSISTENCY,
    "derive_cross_modality_composition_profile": _CROSS_MODALITY,
    "cross_modality_prompt_lines": _CROSS_MODALITY,
    "derive_audio_visual_scene_profile": _AUDIO_VISUAL_SCENE,
    "audio_visual_scene_prompt_lines": _AUDIO_VISUAL_SCENE,
    "artifact_dependency_graph_prompt_lines": _ARTIFACT_DEPENDENCY_GRAPH,
    "derive_artifact_dependency_graph": _ARTIFACT_DEPENDENCY_GRAPH,
    "artifact_intelligence_engine_contract_by_id": _ARTIFACT_ENGINE_CONTRACTS,
    "artifact_intelligence_engine_contracts": _ARTIFACT_ENGINE_CONTRACTS,
    "artifact_capability_matrix_prompt_lines": _ARTIFACT_CAPABILITY_MATRIX,
    "derive_artifact_capability_matrix": _ARTIFACT_CAPABILITY_MATRIX,
    "derive_multi_artifact_strategy": _MULTI_ARTIFACT_STRATEGY,
    "multi_artifact_strategy_prompt_lines": _MULTI_ARTIFACT_STRATEGY,
    "artifact_critic_prompt_lines": _ARTIFACT_CRITIC,
    "derive_artifact_critic_profile": _ARTIFACT_CRITIC,
    "artifact_refiner_prompt_lines": _ARTIFACT_REFINER,
    "derive_artifact_refiner_profile": _ARTIFACT_REFINER,
    "artifact_intelligence_synthesis_prompt_lines": (
        _ARTIFACT_INTELLIGENCE_SYNTHESIS
    ),
    "derive_artifact_intelligence_synthesis_profile": (
        _ARTIFACT_INTELLIGENCE_SYNTHESIS
    ),
    "artifact_merge_planner_prompt_lines": _ARTIFACT_MERGE_PLANNER,
    "derive_artifact_merge_planner_profile": _ARTIFACT_MERGE_PLANNER,
    "artifact_export_intelligence_prompt_lines": (
        _ARTIFACT_EXPORT_INTELLIGENCE
    ),
    "derive_artifact_export_intelligence_profile": (
        _ARTIFACT_EXPORT_INTELLIGENCE
    ),
    "artifact_plan_prompt_lines": _ARTIFACT_PLANNER,
    "derive_artifact_plan": _ARTIFACT_PLANNER,
    "derive_symbolic_narrative_plan": _SYMBOLIC_NARRATIVE,
    "symbolic_narrative_prompt_lines": _SYMBOLIC_NARRATIVE,
    "derive_creative_reasoning_result": _CREATIVE_REASONING,
    "creative_reasoning_prompt_lines": _CREATIVE_REASONING,
    "derive_creative_critic_profile": _CREATIVE_CRITIC_ENGINE,
    "creative_critic_prompt_lines": _CREATIVE_CRITIC_ENGINE,
    "derive_self_evaluation_profile": _SELF_EVALUATION_ENGINE,
    "self_evaluation_prompt_lines": _SELF_EVALUATION_ENGINE,
    "derive_creative_improvement_planner_profile": (
        _CREATIVE_IMPROVEMENT_PLANNER
    ),
    "creative_improvement_planner_prompt_lines": (
        _CREATIVE_IMPROVEMENT_PLANNER
    ),
    "derive_reflection_loop_profile": _REFLECTION_LOOP_ENGINE,
    "reflection_loop_prompt_lines": _REFLECTION_LOOP_ENGINE,
    "derive_creative_confidence_profile": _CREATIVE_CONFIDENCE_ENGINE,
    "creative_confidence_prompt_lines": _CREATIVE_CONFIDENCE_ENGINE,
    "derive_creative_score_profile": _CREATIVE_SCORE_ENGINE,
    "creative_score_prompt_lines": _CREATIVE_SCORE_ENGINE,
    "derive_consistency_validation_profile": _CONSISTENCY_VALIDATION_ENGINE,
    "consistency_validation_prompt_lines": _CONSISTENCY_VALIDATION_ENGINE,
    "derive_evaluation_report_profile": _EVALUATION_REPORTS,
    "evaluation_report_prompt_lines": _EVALUATION_REPORTS,
    "evaluation_engine_contract_by_id": _EVALUATION_ENGINE_CONTRACTS,
    "workstation_engine_contract_by_id": _WORKSTATION_ENGINE_CONTRACTS,
    "workstation_engine_contracts": _WORKSTATION_ENGINE_CONTRACTS,
    "derive_runtime_capability_profile": _RUNTIME_CAPABILITIES,
    "runtime_capability_prompt_lines": _RUNTIME_CAPABILITIES,
    "derive_runtime_compatibility_profile": _RUNTIME_COMPATIBILITY,
    "runtime_compatibility_prompt_lines": _RUNTIME_COMPATIBILITY,
    "derive_creative_assistant_director_brief": _CREATIVE_DIRECTOR,
    "creative_assistant_director_prompt_lines": _CREATIVE_DIRECTOR,
    "derive_hitl_clarification": _CLARIFICATION,
    "evaluate_artifact_sacred_consistency": _SACRED_CONSISTENCY,
    "evaluate_artifact_creative_quality": _CREATIVE_QUALITY,
    "derive_sacred_geometry_guidance": _SACRED_GEOMETRY,
    "detect_sacred_geometry_concepts": _SACRED_GEOMETRY,
    "detect_shader_presets": _SHADER_PRESETS,
    "derive_shader_preset_guidance": _SHADER_PRESETS,
    "derive_visual_style_guidance": _VISUAL_STYLES,
    "derive_reference_fusion_guidance": _REFERENCE_FUSION,
    "detect_visual_styles": _VISUAL_STYLES,
    "extract_workflow_artifacts": _ARTIFACTS,
    "fail_workflow": _WORKFLOW,
    "finish_workflow": _WORKFLOW,
    "next_workflow_step": _WORKFLOW,
    "prepare_workflow_preview_results": _ARTIFACTS,
    "plan_next_refinement_pass": _REFINEMENT_PASSES,
    "refinement_opportunities": _REFINEMENT_PASSES,
    "reference_fusion_prompt_lines": _REFERENCE_FUSION,
    "restart_workflow_step": _WORKFLOW,
    "review_assistant_answer": _WORKFLOW_REVIEW,
    "route_request": _ROUTING,
    "sacred_geometry_prompt_lines": _SACRED_GEOMETRY,
    "shader_preset_prompt_lines": _SHADER_PRESETS,
    "visual_style_prompt_lines": _VISUAL_STYLES,
    "skip_workflow_step": _WORKFLOW,
    "start_refinement_pass_record": _REFINEMENT_PASSES,
    "stream_assistant_workflow_events": _WORKFLOW_GRAPH,
    "start_workflow_step": _WORKFLOW,
}

__all__ = list(_EXPORT_MAP)


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
