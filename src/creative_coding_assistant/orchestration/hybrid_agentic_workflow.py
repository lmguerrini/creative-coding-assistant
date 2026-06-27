"""Passive hybrid agentic workflow metadata for V4.3 preparation."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

BackboneModePhase = Literal[
    "context_intake",
    "planning_reasoning",
    "generation_artifact",
    "review_refinement",
    "terminal_guardrail",
]
ConditionalEscalationCategory = Literal[
    "ambiguity",
    "risk",
    "runtime",
    "quality",
    "hitl",
]
SpecialistLoopCategory = Literal[
    "planning",
    "artifact",
    "runtime",
    "evaluation",
    "synthesis",
]
EscalationGateKind = Literal[
    "backbone_entry",
    "evidence_completeness",
    "specialist_loop_boundary",
    "human_review_visibility",
    "return_handoff",
]
CreativeEscalationPolicyCategory = Literal[
    "concept",
    "aesthetic",
    "runtime",
    "quality",
    "synthesis",
]
ReflectionEscalationPosture = Literal[
    "none",
    "low",
    "medium",
    "high",
    "critical",
]
HybridDebateLoopTopic = Literal[
    "planning_execution_fit",
    "style_aesthetic_alignment",
    "curation_refinement_need",
    "final_synthesis_readiness",
]
HybridVotingTopic = HybridDebateLoopTopic
AgentConfidenceFusionTopic = HybridVotingTopic
DecisionProvenanceTopic = AgentConfidenceFusionTopic
EscalationTraceTopic = DecisionProvenanceTopic
CreativeExplorationBudgetTopic = EscalationTraceTopic
CreativeExplorationBudgetPosture = Literal["narrow", "moderate", "broad", "guarded"]
ResultNormalizationTopic = CreativeExplorationBudgetTopic
ReturnToWorkflowHandoffTopic = ResultNormalizationTopic
ReturnToWorkflowSurface = Literal["planning", "artifact", "evaluation", "finalization"]
HitlEscalationPosture = Literal["optional", "recommended", "required"]
HitlEscalationGateTopic = ReturnToWorkflowHandoffTopic
ConfidenceThresholdRoutingTopic = HitlEscalationGateTopic
ConfidenceThresholdBand = Literal["critical", "low", "medium", "high"]
CostThresholdRoutingTopic = ConfidenceThresholdRoutingTopic
CostThresholdBand = Literal["guarded", "high", "low", "medium"]
LatencyThresholdRoutingTopic = CostThresholdRoutingTopic
LatencyThresholdBand = Literal["guarded", "high", "low", "medium"]
AmbiguityEscalationTopic = LatencyThresholdRoutingTopic
AmbiguityEscalationLevel = Literal["critical", "high", "low", "medium"]
RiskEscalationTopic = AmbiguityEscalationTopic
RiskEscalationLevel = Literal["critical", "high", "low", "medium"]
QualityEscalationTopic = RiskEscalationTopic
QualityEscalationLevel = Literal["critical", "high", "low", "medium"]
AdaptiveMultiAgentEscalationTopic = QualityEscalationTopic
AdaptiveMultiAgentEscalationPosture = Literal[
    "context_packet",
    "specialist_pairing_candidate",
    "multi_loop_review_candidate",
    "terminal_guardrail_candidate",
]

V3_BACKBONE_MODE_ID = "v3_backbone_mode"
V3_BACKBONE_MODE_NODE_SERIALIZATION_VERSION = "v3_backbone_mode_node.v1"
V3_BACKBONE_MODE_REGISTRY_SERIALIZATION_VERSION = "v3_backbone_mode_registry.v1"
CONDITIONAL_ESCALATION_CONDITION_SERIALIZATION_VERSION = (
    "conditional_multi_agent_escalation_condition.v1"
)
CONDITIONAL_ESCALATION_REGISTRY_SERIALIZATION_VERSION = (
    "conditional_multi_agent_escalation_registry.v1"
)
SPECIALIST_AGENT_LOOP_SERIALIZATION_VERSION = "specialist_agent_loop.v1"
SPECIALIST_AGENT_LOOP_REGISTRY_SERIALIZATION_VERSION = (
    "specialist_agent_loop_registry.v1"
)
ESCALATION_GATE_SERIALIZATION_VERSION = "escalation_gate.v1"
ESCALATION_GATE_REGISTRY_SERIALIZATION_VERSION = "escalation_gate_registry.v1"
CREATIVE_ESCALATION_POLICY_RULE_SERIALIZATION_VERSION = (
    "creative_escalation_policy_rule.v1"
)
CREATIVE_ESCALATION_POLICY_REGISTRY_SERIALIZATION_VERSION = (
    "creative_escalation_policy_registry.v1"
)
REFLECTION_ESCALATION_PROFILE_SERIALIZATION_VERSION = (
    "reflection_escalation_profile.v1"
)
REFLECTION_ESCALATION_REGISTRY_SERIALIZATION_VERSION = (
    "reflection_escalation_registry.v1"
)
HYBRID_DEBATE_LOOP_PROFILE_SERIALIZATION_VERSION = (
    "hybrid_agent_debate_loop_profile.v1"
)
HYBRID_DEBATE_LOOP_REGISTRY_SERIALIZATION_VERSION = (
    "hybrid_agent_debate_loop_registry.v1"
)
HYBRID_AGENT_VOTING_PROFILE_SERIALIZATION_VERSION = (
    "hybrid_agent_voting_profile.v1"
)
HYBRID_AGENT_VOTING_REGISTRY_SERIALIZATION_VERSION = (
    "hybrid_agent_voting_registry.v1"
)
AGENT_CONFIDENCE_FUSION_PROFILE_SERIALIZATION_VERSION = (
    "agent_confidence_fusion_profile.v1"
)
AGENT_CONFIDENCE_FUSION_REGISTRY_SERIALIZATION_VERSION = (
    "agent_confidence_fusion_registry.v1"
)
DECISION_PROVENANCE_PROFILE_SERIALIZATION_VERSION = (
    "decision_provenance_profile.v1"
)
DECISION_PROVENANCE_REGISTRY_SERIALIZATION_VERSION = (
    "decision_provenance_registry.v1"
)
ESCALATION_TRACE_PROFILE_SERIALIZATION_VERSION = (
    "escalation_trace_profile.v1"
)
ESCALATION_TRACE_REGISTRY_SERIALIZATION_VERSION = (
    "escalation_trace_registry.v1"
)
CREATIVE_EXPLORATION_BUDGET_PROFILE_SERIALIZATION_VERSION = (
    "creative_exploration_budget_profile.v1"
)
CREATIVE_EXPLORATION_BUDGET_REGISTRY_SERIALIZATION_VERSION = (
    "creative_exploration_budget_registry.v1"
)
RESULT_NORMALIZATION_PROFILE_SERIALIZATION_VERSION = (
    "result_normalization_profile.v1"
)
RESULT_NORMALIZATION_REGISTRY_SERIALIZATION_VERSION = (
    "result_normalization_registry.v1"
)
RETURN_TO_WORKFLOW_HANDOFF_PROFILE_SERIALIZATION_VERSION = (
    "return_to_workflow_handoff_profile.v1"
)
RETURN_TO_WORKFLOW_HANDOFF_REGISTRY_SERIALIZATION_VERSION = (
    "return_to_workflow_handoff_registry.v1"
)
HITL_ESCALATION_GATE_PROFILE_SERIALIZATION_VERSION = (
    "hitl_escalation_gate_profile.v1"
)
HITL_ESCALATION_GATE_REGISTRY_SERIALIZATION_VERSION = (
    "hitl_escalation_gate_registry.v1"
)
CONFIDENCE_THRESHOLD_ROUTING_PROFILE_SERIALIZATION_VERSION = (
    "confidence_threshold_routing_profile.v1"
)
CONFIDENCE_THRESHOLD_ROUTING_REGISTRY_SERIALIZATION_VERSION = (
    "confidence_threshold_routing_registry.v1"
)
COST_THRESHOLD_ROUTING_PROFILE_SERIALIZATION_VERSION = (
    "cost_threshold_routing_profile.v1"
)
COST_THRESHOLD_ROUTING_REGISTRY_SERIALIZATION_VERSION = (
    "cost_threshold_routing_registry.v1"
)
LATENCY_THRESHOLD_ROUTING_PROFILE_SERIALIZATION_VERSION = (
    "latency_threshold_routing_profile.v1"
)
LATENCY_THRESHOLD_ROUTING_REGISTRY_SERIALIZATION_VERSION = (
    "latency_threshold_routing_registry.v1"
)
AMBIGUITY_ESCALATION_PROFILE_SERIALIZATION_VERSION = (
    "ambiguity_escalation_profile.v1"
)
AMBIGUITY_ESCALATION_REGISTRY_SERIALIZATION_VERSION = (
    "ambiguity_escalation_registry.v1"
)
RISK_ESCALATION_PROFILE_SERIALIZATION_VERSION = "risk_escalation_profile.v1"
RISK_ESCALATION_REGISTRY_SERIALIZATION_VERSION = "risk_escalation_registry.v1"
QUALITY_ESCALATION_PROFILE_SERIALIZATION_VERSION = "quality_escalation_profile.v1"
QUALITY_ESCALATION_REGISTRY_SERIALIZATION_VERSION = "quality_escalation_registry.v1"
ADAPTIVE_MULTI_AGENT_ESCALATION_PROFILE_SERIALIZATION_VERSION = (
    "adaptive_multi_agent_escalation_profile.v1"
)
ADAPTIVE_MULTI_AGENT_ESCALATION_REGISTRY_SERIALIZATION_VERSION = (
    "adaptive_multi_agent_escalation_registry.v1"
)
HYBRID_WORKFLOW_STAGE_SERIALIZATION_VERSION = "hybrid_workflow_stage.v1"
HYBRID_WORKFLOW_REGISTRY_SERIALIZATION_VERSION = "hybrid_workflow_registry.v1"
V3_BACKBONE_MODE_AUTHORITY_BOUNDARY = (
    "V3 Backbone Mode metadata declares the current deterministic V3 workflow "
    "graph as the active backbone for V4.3 hybrid workflow readiness only; it "
    "does not change workflow graph order, perform multi-agent escalation, "
    "invoke agents, route providers or models, select runtimes, trigger "
    "retries, mutate prompts, write memory, execute artifacts, or modify "
    "generated output."
)
CONDITIONAL_ESCALATION_AUTHORITY_BOUNDARY = (
    "Conditional multi-agent escalation metadata describes advisory candidate "
    "conditions that could prepare future escalation context from the V3 "
    "backbone; it does not evaluate conditions, invoke agents, route providers "
    "or models, select runtimes, control workflow transitions, trigger retries, "
    "execute voting, write memory, or modify generated output."
)
SPECIALIST_AGENT_LOOP_AUTHORITY_BOUNDARY = (
    "Specialist agent loop metadata describes bounded future loop candidates "
    "for known passive agent contracts only; it does not execute loops, invoke "
    "agents, coordinate multi-agent work, route providers or models, select "
    "runtimes, control workflow transitions, trigger retries, write memory, or "
    "modify generated output."
)
ESCALATION_GATE_AUTHORITY_BOUNDARY = (
    "Escalation gate metadata describes passive advisory gates across the V3 "
    "backbone, conditional escalation candidates, and specialist loop profiles "
    "only; it does not evaluate gates, approve escalation, invoke agents, "
    "control workflow transitions, route providers or models, trigger retries, "
    "execute artifacts, write memory, or modify generated output."
)
CREATIVE_ESCALATION_POLICY_AUTHORITY_BOUNDARY = (
    "Creative escalation policy metadata describes passive creative-domain "
    "escalation rules tied to advisory gates and specialist loop candidates "
    "only; it does not evaluate creative policy, approve escalation, invoke "
    "agents, route providers or models, control workflow transitions, trigger "
    "retries, execute artifacts, write memory, or modify generated output."
)
REFLECTION_ESCALATION_AUTHORITY_BOUNDARY = (
    "Reflection escalation metadata describes passive escalation posture for "
    "existing Reflection Loop Engine signals only; it does not run reflection, "
    "trigger refinement, approve escalation, invoke agents, route providers or "
    "models, control workflow transitions, write memory, or modify generated "
    "output."
)
HYBRID_DEBATE_LOOP_AUTHORITY_BOUNDARY = (
    "Hybrid agent debate loop metadata maps existing passive debate topics to "
    "V4.3 escalation policy and specialist loop context only; it does not "
    "execute debate loops, invoke agents, trigger retries, route providers or "
    "models, control workflow transitions, write memory, or modify generated "
    "output."
)
HYBRID_AGENT_VOTING_AUTHORITY_BOUNDARY = (
    "Hybrid agent voting metadata maps passive consensus voting placeholders "
    "to V4.3 debate loops and escalation context only; it does not execute "
    "voting, select final answers, invoke agents, route providers or models, "
    "control workflow transitions, trigger retries, or modify generated output."
)
AGENT_CONFIDENCE_FUSION_AUTHORITY_BOUNDARY = (
    "Agent confidence fusion metadata maps passive voting profiles to the "
    "existing Creative Confidence Engine surface for future synthesis context "
    "only; it does not calculate confidence scores, fuse confidence, weight "
    "votes, select final answers, invoke agents, route providers or models, "
    "control workflow transitions, trigger retries, or modify generated output."
)
DECISION_PROVENANCE_AUTHORITY_BOUNDARY = (
    "Decision provenance metadata describes passive lineage for future hybrid "
    "agentic decisions across V3 workflow nodes and advisory metadata only; it "
    "does not record provenance, emit traces, write memory, select decisions, "
    "invoke agents, route providers or models, control workflow transitions, "
    "trigger retries, or modify generated output."
)
ESCALATION_TRACE_AUTHORITY_BOUNDARY = (
    "Escalation trace metadata describes passive trace context for future "
    "hybrid escalation visibility across known conditions, gates, signals, "
    "and provenance metadata only; it does not capture traces, emit traces, "
    "execute escalation, evaluate gates, write memory, invoke agents, control "
    "workflow transitions, trigger retries, or modify generated output."
)
CREATIVE_EXPLORATION_BUDGET_AUTHORITY_BOUNDARY = (
    "Creative exploration budget metadata describes passive future budget "
    "posture for advisory exploration only; it does not enforce budgets, "
    "generate variants, trigger refinement, route by cost, invoke agents, "
    "control workflow transitions, trigger retries, or modify generated "
    "output."
)
RESULT_NORMALIZATION_AUTHORITY_BOUNDARY = (
    "Result normalization metadata describes passive advisory result packet "
    "surfaces for future hybrid synthesis only; it does not transform results, "
    "rewrite outputs, enforce schemas, mutate artifacts, invoke agents, route "
    "providers or models, control workflow transitions, trigger retries, or "
    "modify generated output."
)
RETURN_TO_WORKFLOW_HANDOFF_AUTHORITY_BOUNDARY = (
    "Return-to-workflow handoff metadata describes passive future handoff "
    "context from normalized advisory results back to existing V3 workflow "
    "surfaces only; it does not perform runtime handoffs, change workflow "
    "graph order, alter prompts, execute agents, control workflow transitions, "
    "trigger retries, or modify generated output."
)
HITL_ESCALATION_GATE_AUTHORITY_BOUNDARY = (
    "HITL escalation gate metadata describes passive human-review visibility "
    "posture for future hybrid escalation only; it does not trigger human "
    "review, request human input, evaluate gates, approve escalation, invoke "
    "agents, control workflow transitions, trigger retries, or modify "
    "generated output."
)
CONFIDENCE_THRESHOLD_ROUTING_AUTHORITY_BOUNDARY = (
    "Confidence threshold routing metadata describes passive advisory "
    "confidence bands for future hybrid routing visibility only; it does not "
    "route by confidence, change providers or models, invoke agents, evaluate "
    "thresholds, control workflow transitions, trigger retries, or modify "
    "generated output."
)
COST_THRESHOLD_ROUTING_AUTHORITY_BOUNDARY = (
    "Cost threshold routing metadata describes passive advisory cost bands "
    "for future hybrid routing visibility only; it does not route by cost, "
    "enforce budgets, change providers or models, invoke agents, evaluate "
    "thresholds, control workflow transitions, trigger retries, or modify "
    "generated output."
)
LATENCY_THRESHOLD_ROUTING_AUTHORITY_BOUNDARY = (
    "Latency threshold routing metadata describes passive advisory latency "
    "bands for future hybrid routing visibility only; it does not route by "
    "latency, select runtimes, change providers or models, invoke agents, "
    "evaluate thresholds, control workflow transitions, trigger retries, or "
    "modify generated output."
)
AMBIGUITY_ESCALATION_AUTHORITY_BOUNDARY = (
    "Ambiguity escalation metadata describes passive advisory ambiguity "
    "posture for future hybrid escalation visibility only; it does not "
    "evaluate ambiguity, trigger clarification, execute escalation, invoke "
    "agents, route providers or models, control workflow transitions, trigger "
    "retries, or modify generated output."
)
RISK_ESCALATION_AUTHORITY_BOUNDARY = (
    "Risk escalation metadata describes passive advisory risk posture for "
    "future hybrid escalation visibility only; it does not evaluate risk, "
    "execute escalation, apply mitigation, invoke agents, route providers or "
    "models, control workflow transitions, trigger retries, or modify "
    "generated output."
)
QUALITY_ESCALATION_AUTHORITY_BOUNDARY = (
    "Quality escalation metadata describes passive advisory quality posture "
    "for future hybrid escalation visibility only; it does not evaluate "
    "quality, execute escalation, trigger refinement, invoke agents, route "
    "providers or models, control workflow transitions, trigger retries, or "
    "modify generated output."
)
ADAPTIVE_MULTI_AGENT_ESCALATION_AUTHORITY_BOUNDARY = (
    "Adaptive multi-agent escalation metadata describes passive advisory "
    "escalation posture across conditional candidates, specialist loops, "
    "HITL gates, and quality, risk, and ambiguity metadata only; it does not "
    "evaluate adaptation, execute escalation, orchestrate agents, invoke "
    "agents, select runtimes, route providers or models, control workflow "
    "transitions, trigger retries, or modify generated output."
)
HYBRID_WORKFLOW_REGISTRY_AUTHORITY_BOUNDARY = (
    "Hybrid agentic workflow metadata maps current V3 workflow nodes to future "
    "agent capability and escalation policy readiness only; it does not change "
    "workflow graph order, create agents, route providers or models, select "
    "runtimes, trigger retries, execute artifacts, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "workflow_graph_mutation",
    "provider_or_model_routing",
    "runtime_selection",
    "retry_or_refinement_triggering",
    "agent_invocation",
    "artifact_execution",
    "generated_output_modification",
)
_V3_BACKBONE_MODE_BLOCKED_RUNTIME_BEHAVIORS = (
    "workflow_graph_mutation",
    "provider_or_model_routing",
    "runtime_selection",
    "retry_or_refinement_triggering",
    "multi_agent_escalation_execution",
    "agent_invocation",
    "prompt_mutation",
    "memory_write",
    "generated_output_modification",
)
_V3_BACKBONE_MODE_SOURCE_REGISTRIES = (
    "assistant_workflow_node_order",
    "workflow_step_order",
    "artifact_engine_contract_registry",
    "evaluation_engine_contract_registry",
    "workstation_engine_contract_registry",
    "hybrid_agentic_workflow_registry",
)
_CONDITIONAL_ESCALATION_BLOCKED_RUNTIME_BEHAVIORS = (
    "condition_evaluation",
    "multi_agent_execution",
    "provider_or_model_routing",
    "runtime_selection",
    "workflow_control",
    "retry_or_refinement_triggering",
    "agent_invocation",
    "voting_execution",
    "memory_write",
    "generated_output_modification",
)
_CONDITIONAL_ESCALATION_SOURCE_REGISTRIES = (
    "v3_backbone_mode_registry",
    "agent_capability_registry",
    "escalation_policy_registry",
    "agent_escalation_signal_registry",
    "hybrid_agentic_workflow_registry",
)
_SPECIALIST_AGENT_LOOP_BLOCKED_RUNTIME_BEHAVIORS = (
    "loop_execution",
    "agent_invocation",
    "multi_agent_orchestration",
    "provider_or_model_routing",
    "runtime_selection",
    "workflow_control",
    "retry_or_refinement_triggering",
    "memory_write",
    "generated_output_modification",
)
_SPECIALIST_AGENT_LOOP_SOURCE_REGISTRIES = (
    "agent_contract_registry",
    "conditional_multi_agent_escalation_registry",
    "v3_backbone_mode_registry",
    "hybrid_agentic_workflow_registry",
)
_ESCALATION_GATE_BLOCKED_RUNTIME_BEHAVIORS = (
    "gate_evaluation",
    "escalation_approval",
    "agent_invocation",
    "multi_agent_orchestration",
    "provider_or_model_routing",
    "runtime_selection",
    "workflow_control",
    "retry_or_refinement_triggering",
    "artifact_execution",
    "memory_write",
    "generated_output_modification",
)
_ESCALATION_GATE_SOURCE_REGISTRIES = (
    "v3_backbone_mode_registry",
    "conditional_multi_agent_escalation_registry",
    "specialist_agent_loop_registry",
    "escalation_policy_registry",
    "hybrid_agentic_workflow_registry",
)
_CREATIVE_ESCALATION_POLICY_BLOCKED_RUNTIME_BEHAVIORS = (
    "creative_policy_evaluation",
    "escalation_approval",
    "gate_evaluation",
    "agent_invocation",
    "provider_or_model_routing",
    "runtime_selection",
    "workflow_control",
    "retry_or_refinement_triggering",
    "artifact_execution",
    "memory_write",
    "generated_output_modification",
)
_CREATIVE_ESCALATION_POLICY_SOURCE_REGISTRIES = (
    "escalation_gate_registry",
    "specialist_agent_loop_registry",
    "conditional_multi_agent_escalation_registry",
    "artifact_engine_contract_registry",
    "evaluation_engine_contract_registry",
    "hybrid_agentic_workflow_registry",
)
_REFLECTION_ESCALATION_BLOCKED_RUNTIME_BEHAVIORS = (
    "reflection_execution",
    "refinement_triggering",
    "creative_policy_evaluation",
    "escalation_approval",
    "agent_invocation",
    "provider_or_model_routing",
    "workflow_control",
    "memory_write",
    "generated_output_modification",
)
_REFLECTION_ESCALATION_SOURCE_REGISTRIES = (
    "reflection_loop_engine",
    "creative_escalation_policy_registry",
    "escalation_gate_registry",
    "evaluation_engine_contract_registry",
    "hybrid_agentic_workflow_registry",
)
_HYBRID_DEBATE_LOOP_BLOCKED_RUNTIME_BEHAVIORS = (
    "debate_loop_execution",
    "agent_invocation",
    "retry_triggering",
    "provider_or_model_routing",
    "workflow_control",
    "memory_write",
    "generated_output_modification",
)
_HYBRID_DEBATE_LOOP_SOURCE_REGISTRIES = (
    "agent_debate_registry",
    "reflection_escalation_registry",
    "creative_escalation_policy_registry",
    "specialist_agent_loop_registry",
    "hybrid_agentic_workflow_registry",
)
_HYBRID_AGENT_VOTING_BLOCKED_RUNTIME_BEHAVIORS = (
    "voting_execution",
    "final_answer_selection",
    "agent_invocation",
    "provider_or_model_routing",
    "workflow_control",
    "retry_triggering",
    "generated_output_modification",
)
_HYBRID_AGENT_VOTING_SOURCE_REGISTRIES = (
    "consensus_builder_registry",
    "hybrid_agent_debate_loop_registry",
    "reflection_escalation_registry",
    "creative_escalation_policy_registry",
    "hybrid_agentic_workflow_registry",
)
_AGENT_CONFIDENCE_FUSION_BLOCKED_RUNTIME_BEHAVIORS = (
    "confidence_score_calculation",
    "confidence_fusion_execution",
    "vote_weighting_execution",
    "final_answer_selection",
    "agent_invocation",
    "provider_or_model_routing",
    "workflow_control",
    "retry_triggering",
    "generated_output_modification",
)
_AGENT_CONFIDENCE_FUSION_SOURCE_REGISTRIES = (
    "creative_confidence_engine",
    "hybrid_agent_voting_registry",
    "hybrid_agent_debate_loop_registry",
    "reflection_escalation_registry",
    "evaluation_engine_contract_registry",
    "hybrid_agentic_workflow_registry",
)
_DECISION_PROVENANCE_BLOCKED_RUNTIME_BEHAVIORS = (
    "provenance_recording",
    "decision_logging",
    "trace_emission",
    "memory_write",
    "decision_selection",
    "agent_invocation",
    "provider_or_model_routing",
    "workflow_control",
    "retry_triggering",
    "generated_output_modification",
)
_DECISION_PROVENANCE_SOURCE_REGISTRIES = (
    "agent_confidence_fusion_registry",
    "hybrid_agent_voting_registry",
    "hybrid_agent_debate_loop_registry",
    "v3_backbone_mode_registry",
    "workstation_engine_contract_registry",
    "hybrid_agentic_workflow_registry",
)
_ESCALATION_TRACE_BLOCKED_RUNTIME_BEHAVIORS = (
    "trace_capture",
    "trace_emission",
    "escalation_execution",
    "gate_evaluation",
    "memory_write",
    "agent_invocation",
    "provider_or_model_routing",
    "workflow_control",
    "retry_triggering",
    "generated_output_modification",
)
_ESCALATION_TRACE_SOURCE_REGISTRIES = (
    "decision_provenance_registry",
    "conditional_multi_agent_escalation_registry",
    "escalation_gate_registry",
    "agent_escalation_signal_registry",
    "reflection_escalation_registry",
    "hybrid_agentic_workflow_registry",
)
_CREATIVE_EXPLORATION_BUDGET_BLOCKED_RUNTIME_BEHAVIORS = (
    "budget_enforcement",
    "variant_generation",
    "refinement_triggering",
    "cost_routing",
    "agent_invocation",
    "provider_or_model_routing",
    "workflow_control",
    "retry_triggering",
    "generated_output_modification",
)
_CREATIVE_EXPLORATION_BUDGET_SOURCE_REGISTRIES = (
    "escalation_trace_registry",
    "decision_provenance_registry",
    "creative_planning_engine",
    "creative_constraints_engine",
    "creative_tradeoff_engine",
    "hybrid_agentic_workflow_registry",
)
_RESULT_NORMALIZATION_BLOCKED_RUNTIME_BEHAVIORS = (
    "result_transformation",
    "output_rewriting",
    "schema_enforcement",
    "artifact_mutation",
    "agent_invocation",
    "provider_or_model_routing",
    "workflow_control",
    "retry_triggering",
    "generated_output_modification",
)
_RESULT_NORMALIZATION_SOURCE_REGISTRIES = (
    "creative_exploration_budget_registry",
    "agent_confidence_fusion_registry",
    "decision_provenance_registry",
    "escalation_trace_registry",
    "artifact_engine_contract_registry",
    "evaluation_engine_contract_registry",
)
_RETURN_TO_WORKFLOW_HANDOFF_BLOCKED_RUNTIME_BEHAVIORS = (
    "runtime_handoff_execution",
    "workflow_graph_change",
    "prompt_alteration",
    "agent_execution",
    "provider_or_model_routing",
    "workflow_control",
    "retry_triggering",
    "generated_output_modification",
)
_RETURN_TO_WORKFLOW_HANDOFF_SOURCE_REGISTRIES = (
    "result_normalization_registry",
    "escalation_gate_registry",
    "workflow_agent_handoff_registry",
    "v3_backbone_mode_registry",
    "workstation_engine_contract_registry",
    "hybrid_agentic_workflow_registry",
)
_HITL_ESCALATION_GATE_BLOCKED_RUNTIME_BEHAVIORS = (
    "hitl_triggering",
    "human_review_request",
    "gate_evaluation",
    "escalation_approval",
    "agent_invocation",
    "provider_or_model_routing",
    "workflow_control",
    "retry_triggering",
    "generated_output_modification",
)
_HITL_ESCALATION_GATE_SOURCE_REGISTRIES = (
    "return_to_workflow_handoff_registry",
    "escalation_gate_registry",
    "agent_escalation_signal_registry",
    "creative_confidence_engine",
    "reflection_escalation_registry",
    "hybrid_agentic_workflow_registry",
)
_CONFIDENCE_THRESHOLD_ROUTING_BLOCKED_RUNTIME_BEHAVIORS = (
    "confidence_threshold_evaluation",
    "confidence_based_routing",
    "provider_or_model_routing",
    "agent_invocation",
    "workflow_control",
    "retry_triggering",
    "generated_output_modification",
)
_CONFIDENCE_THRESHOLD_ROUTING_SOURCE_REGISTRIES = (
    "hitl_escalation_gate_registry",
    "agent_confidence_fusion_registry",
    "creative_confidence_engine",
    "agent_escalation_signal_registry",
    "escalation_policy_registry",
    "hybrid_agentic_workflow_registry",
)
_COST_THRESHOLD_ROUTING_BLOCKED_RUNTIME_BEHAVIORS = (
    "cost_threshold_evaluation",
    "cost_based_routing",
    "budget_enforcement",
    "provider_or_model_routing",
    "agent_invocation",
    "workflow_control",
    "retry_triggering",
    "generated_output_modification",
)
_COST_THRESHOLD_ROUTING_SOURCE_REGISTRIES = (
    "creative_exploration_budget_registry",
    "confidence_threshold_routing_registry",
    "creative_planning_engine",
    "creative_constraints_engine",
    "creative_tradeoff_engine",
    "agent_escalation_signal_registry",
    "hybrid_agentic_workflow_registry",
)
_LATENCY_THRESHOLD_ROUTING_BLOCKED_RUNTIME_BEHAVIORS = (
    "latency_threshold_evaluation",
    "latency_based_routing",
    "runtime_selection",
    "provider_or_model_routing",
    "agent_invocation",
    "workflow_control",
    "retry_triggering",
    "generated_output_modification",
)
_LATENCY_THRESHOLD_ROUTING_SOURCE_REGISTRIES = (
    "cost_threshold_routing_registry",
    "agent_metadata_registry",
    "artifact_engine_contract_registry",
    "evaluation_engine_contract_registry",
    "workstation_engine_contract_registry",
    "agent_escalation_signal_registry",
    "hybrid_agentic_workflow_registry",
)
_AMBIGUITY_ESCALATION_BLOCKED_RUNTIME_BEHAVIORS = (
    "ambiguity_evaluation",
    "escalation_execution",
    "clarification_request_triggering",
    "agent_invocation",
    "provider_or_model_routing",
    "workflow_control",
    "retry_triggering",
    "generated_output_modification",
)
_AMBIGUITY_ESCALATION_SOURCE_REGISTRIES = (
    "latency_threshold_routing_registry",
    "conditional_multi_agent_escalation_registry",
    "escalation_policy_registry",
    "agent_escalation_signal_registry",
    "clarification_engine",
    "creative_confidence_engine",
    "hybrid_agentic_workflow_registry",
)
_RISK_ESCALATION_BLOCKED_RUNTIME_BEHAVIORS = (
    "risk_evaluation",
    "escalation_execution",
    "mitigation_execution",
    "agent_invocation",
    "provider_or_model_routing",
    "workflow_control",
    "retry_triggering",
    "generated_output_modification",
)
_RISK_ESCALATION_SOURCE_REGISTRIES = (
    "ambiguity_escalation_registry",
    "conditional_multi_agent_escalation_registry",
    "escalation_policy_registry",
    "creative_escalation_policy_registry",
    "agent_escalation_signal_registry",
    "artifact_engine_contract_registry",
    "hybrid_agentic_workflow_registry",
)
_QUALITY_ESCALATION_BLOCKED_RUNTIME_BEHAVIORS = (
    "quality_evaluation",
    "escalation_execution",
    "refinement_triggering",
    "agent_invocation",
    "provider_or_model_routing",
    "workflow_control",
    "retry_triggering",
    "generated_output_modification",
)
_QUALITY_ESCALATION_SOURCE_REGISTRIES = (
    "risk_escalation_registry",
    "creative_escalation_policy_registry",
    "agent_confidence_fusion_registry",
    "creative_confidence_engine",
    "creative_quality_prediction_engine",
    "agent_escalation_signal_registry",
    "hybrid_agentic_workflow_registry",
)
_ADAPTIVE_MULTI_AGENT_ESCALATION_BLOCKED_RUNTIME_BEHAVIORS = (
    "adaptation_evaluation",
    "escalation_execution",
    "multi_agent_orchestration",
    "agent_invocation",
    "runtime_selection",
    "provider_or_model_routing",
    "workflow_control",
    "retry_triggering",
    "generated_output_modification",
)
_ADAPTIVE_MULTI_AGENT_ESCALATION_SOURCE_REGISTRIES = (
    "quality_escalation_registry",
    "risk_escalation_registry",
    "ambiguity_escalation_registry",
    "conditional_multi_agent_escalation_registry",
    "specialist_agent_loop_registry",
    "hitl_escalation_gate_registry",
    "agent_capability_registry",
    "agent_escalation_signal_registry",
    "hybrid_agentic_workflow_registry",
)
_V3_BACKBONE_MODE_PHASE_IDS: tuple[BackboneModePhase, ...] = (
    "context_intake",
    "planning_reasoning",
    "generation_artifact",
    "review_refinement",
    "terminal_guardrail",
)
_CONDITIONAL_ESCALATION_CATEGORIES: tuple[ConditionalEscalationCategory, ...] = (
    "ambiguity",
    "risk",
    "runtime",
    "quality",
    "hitl",
)
_SPECIALIST_AGENT_LOOP_CATEGORIES: tuple[SpecialistLoopCategory, ...] = (
    "planning",
    "artifact",
    "runtime",
    "evaluation",
    "synthesis",
)
_ESCALATION_GATE_KINDS: tuple[EscalationGateKind, ...] = (
    "backbone_entry",
    "evidence_completeness",
    "specialist_loop_boundary",
    "human_review_visibility",
    "return_handoff",
)
_CREATIVE_ESCALATION_POLICY_CATEGORIES: tuple[
    CreativeEscalationPolicyCategory, ...
] = (
    "concept",
    "aesthetic",
    "runtime",
    "quality",
    "synthesis",
)
_REFLECTION_ESCALATION_POSTURES: tuple[ReflectionEscalationPosture, ...] = (
    "none",
    "low",
    "medium",
    "high",
    "critical",
)
_HYBRID_DEBATE_LOOP_TOPICS: tuple[HybridDebateLoopTopic, ...] = (
    "planning_execution_fit",
    "style_aesthetic_alignment",
    "curation_refinement_need",
    "final_synthesis_readiness",
)
_HYBRID_AGENT_VOTING_TOPICS: tuple[HybridVotingTopic, ...] = (
    "planning_execution_fit",
    "style_aesthetic_alignment",
    "curation_refinement_need",
    "final_synthesis_readiness",
)
_AGENT_CONFIDENCE_FUSION_TOPICS: tuple[AgentConfidenceFusionTopic, ...] = (
    "planning_execution_fit",
    "style_aesthetic_alignment",
    "curation_refinement_need",
    "final_synthesis_readiness",
)
_DECISION_PROVENANCE_TOPICS: tuple[DecisionProvenanceTopic, ...] = (
    "planning_execution_fit",
    "style_aesthetic_alignment",
    "curation_refinement_need",
    "final_synthesis_readiness",
)
_ESCALATION_TRACE_TOPICS: tuple[EscalationTraceTopic, ...] = (
    "planning_execution_fit",
    "style_aesthetic_alignment",
    "curation_refinement_need",
    "final_synthesis_readiness",
)
_CREATIVE_EXPLORATION_BUDGET_TOPICS: tuple[CreativeExplorationBudgetTopic, ...] = (
    "planning_execution_fit",
    "style_aesthetic_alignment",
    "curation_refinement_need",
    "final_synthesis_readiness",
)
_CREATIVE_EXPLORATION_BUDGET_POSTURES: tuple[
    CreativeExplorationBudgetPosture, ...
] = (
    "moderate",
    "broad",
    "guarded",
    "narrow",
)
_RESULT_NORMALIZATION_TOPICS: tuple[ResultNormalizationTopic, ...] = (
    "planning_execution_fit",
    "style_aesthetic_alignment",
    "curation_refinement_need",
    "final_synthesis_readiness",
)
_RETURN_TO_WORKFLOW_HANDOFF_TOPICS: tuple[ReturnToWorkflowHandoffTopic, ...] = (
    "planning_execution_fit",
    "style_aesthetic_alignment",
    "curation_refinement_need",
    "final_synthesis_readiness",
)
_WORKFLOW_AGENT_HANDOFF_IDS = (
    "planning_surface_agent_handoff",
    "artifact_surface_agent_handoff",
    "evaluation_surface_agent_handoff",
    "provenance_surface_agent_handoff",
    "finalization_surface_agent_handoff",
)
_HITL_ESCALATION_GATE_TOPICS: tuple[HitlEscalationGateTopic, ...] = (
    "planning_execution_fit",
    "style_aesthetic_alignment",
    "curation_refinement_need",
    "final_synthesis_readiness",
)
_HITL_ESCALATION_POSTURES: tuple[HitlEscalationPosture, ...] = (
    "recommended",
    "optional",
    "recommended",
    "required",
)
_CONFIDENCE_THRESHOLD_ROUTING_TOPICS: tuple[
    ConfidenceThresholdRoutingTopic, ...
] = (
    "planning_execution_fit",
    "style_aesthetic_alignment",
    "curation_refinement_need",
    "final_synthesis_readiness",
)
_CONFIDENCE_THRESHOLD_BANDS: tuple[ConfidenceThresholdBand, ...] = (
    "medium",
    "high",
    "low",
    "critical",
)
_COST_THRESHOLD_ROUTING_TOPICS: tuple[CostThresholdRoutingTopic, ...] = (
    "planning_execution_fit",
    "style_aesthetic_alignment",
    "curation_refinement_need",
    "final_synthesis_readiness",
)
_COST_THRESHOLD_BANDS: tuple[CostThresholdBand, ...] = (
    "medium",
    "high",
    "guarded",
    "low",
)
_LATENCY_THRESHOLD_ROUTING_TOPICS: tuple[LatencyThresholdRoutingTopic, ...] = (
    "planning_execution_fit",
    "style_aesthetic_alignment",
    "curation_refinement_need",
    "final_synthesis_readiness",
)
_LATENCY_THRESHOLD_BANDS: tuple[LatencyThresholdBand, ...] = (
    "medium",
    "high",
    "guarded",
    "low",
)
_LATENCY_METADATA_SOURCE_SURFACES = (
    "agent_estimated_latency_metadata",
    "artifact_engine_estimated_latency_metadata",
    "evaluation_engine_estimated_latency_metadata",
    "workstation_surface_estimated_latency_metadata",
)
_AMBIGUITY_ESCALATION_TOPICS: tuple[AmbiguityEscalationTopic, ...] = (
    "planning_execution_fit",
    "style_aesthetic_alignment",
    "curation_refinement_need",
    "final_synthesis_readiness",
)
_AMBIGUITY_ESCALATION_LEVELS: tuple[AmbiguityEscalationLevel, ...] = (
    "high",
    "medium",
    "critical",
    "low",
)
_AMBIGUITY_EVIDENCE_SURFACES = (
    "missing_information",
    "planning_gap_summary",
    "disagreement_points",
    "hitl_questions",
)
_RISK_ESCALATION_TOPICS: tuple[RiskEscalationTopic, ...] = (
    "planning_execution_fit",
    "style_aesthetic_alignment",
    "curation_refinement_need",
    "final_synthesis_readiness",
)
_RISK_ESCALATION_LEVELS: tuple[RiskEscalationLevel, ...] = (
    "medium",
    "high",
    "critical",
    "low",
)
_RISK_EVIDENCE_SURFACES = (
    "implementation_risks",
    "risk_assessment",
    "unsupported_assumptions",
    "capability_risks",
    "escalation_candidates",
)
_QUALITY_ESCALATION_TOPICS: tuple[QualityEscalationTopic, ...] = (
    "planning_execution_fit",
    "style_aesthetic_alignment",
    "curation_refinement_need",
    "final_synthesis_readiness",
)
_QUALITY_ESCALATION_LEVELS: tuple[QualityEscalationLevel, ...] = (
    "medium",
    "high",
    "critical",
    "low",
)
_QUALITY_EVIDENCE_SURFACES = (
    "quality_signal_metadata",
    "quality_review_signals",
    "weakest_quality_signals",
    "quality_risks",
    "confidence_uncertainties",
)
_ADAPTIVE_MULTI_AGENT_ESCALATION_TOPICS: tuple[
    AdaptiveMultiAgentEscalationTopic, ...
] = (
    "planning_execution_fit",
    "style_aesthetic_alignment",
    "curation_refinement_need",
    "final_synthesis_readiness",
)
_ADAPTIVE_MULTI_AGENT_ESCALATION_POSTURES: tuple[
    AdaptiveMultiAgentEscalationPosture, ...
] = (
    "context_packet",
    "specialist_pairing_candidate",
    "multi_loop_review_candidate",
    "terminal_guardrail_candidate",
)
_ADAPTIVE_MULTI_AGENT_ESCALATION_EVIDENCE_SURFACES = (
    "escalation_candidates",
    "agent_escalation_candidates",
    "human_review_posture",
    "ambiguity_context",
    "risk_context",
    "quality_uncertainty_summary",
)
_ADAPTIVE_MULTI_AGENT_ESCALATION_CAPABILITY_IDS = (
    "adaptive_multi_agent_escalation",
)
_KNOWN_SPECIALIST_AGENT_IDS = (
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
_KNOWN_CONDITIONAL_ESCALATION_CAPABILITY_IDS = (
    "v4_planner_agent",
    "v4_artifact_agent",
    "v4_runtime_agent",
    "v4_agent_router",
    "v4_agentic_studio",
    "adaptive_multi_agent_escalation",
)
_KNOWN_CONDITIONAL_ESCALATION_POLICY_RULE_IDS = (
    "missing_information_review",
    "artifact_risk_review",
    "runtime_incompatibility_review",
    "evaluation_confidence_review",
    "future_agent_escalation_readiness",
)
_KNOWN_CONDITIONAL_ESCALATION_SIGNAL_IDS = (
    "confidence_escalation_signal",
    "risk_escalation_signal",
    "ambiguity_escalation_signal",
    "cost_escalation_signal",
    "latency_escalation_signal",
    "quality_escalation_signal",
    "hitl_escalation_signal",
)


class V3BackboneModeNodeProfile(BaseModel):
    """Metadata-only profile for one preserved V3 workflow backbone node."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    mode_id: Literal["v3_backbone_mode"] = V3_BACKBONE_MODE_ID
    node_id: str = Field(min_length=1, max_length=80)
    phase: BackboneModePhase
    active_runtime_owner: Literal["v3_workflow_graph"] = "v3_workflow_graph"
    preserved_surfaces: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    authority_boundary: str = Field(min_length=1, max_length=900)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_V3_BACKBONE_MODE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    backbone_runtime_active: Literal[True] = True
    workflow_order_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    multi_agent_escalation_executed: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["v3_backbone_mode_node.v1"] = (
        V3_BACKBONE_MODE_NODE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class V3BackboneModeRegistry(BaseModel):
    """Stable passive registry declaring the preserved V3 workflow backbone."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["v3_backbone_mode_registry"] = "v3_backbone_mode_registry"
    mode_id: Literal["v3_backbone_mode"] = V3_BACKBONE_MODE_ID
    serialization_version: Literal["v3_backbone_mode_registry.v1"] = (
        V3_BACKBONE_MODE_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=V3_BACKBONE_MODE_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    node_profiles: tuple[V3BackboneModeNodeProfile, ...] = Field(
        min_length=18,
        max_length=18,
    )
    node_ids: tuple[str, ...] = Field(min_length=18, max_length=18)
    preserved_workflow_order: tuple[str, ...] = Field(min_length=18, max_length=18)
    phase_ids: tuple[BackboneModePhase, ...] = Field(min_length=5, max_length=5)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    node_count: int = Field(ge=18, le=18)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_V3_BACKBONE_MODE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    backbone_runtime_active: Literal[True] = True
    workflow_order_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    multi_agent_escalation_executed: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_backbone_profiles(self) -> Self:
        derived_node_ids = tuple(profile.node_id for profile in self.node_profiles)
        derived_phase_ids = tuple(
            dict.fromkeys(profile.phase for profile in self.node_profiles)
        )
        if self.node_ids != derived_node_ids:
            raise ValueError("node_ids must match node_profiles")
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("node_ids must be unique")
        if self.preserved_workflow_order != self.node_ids:
            raise ValueError("preserved_workflow_order must match node_ids")
        if self.phase_ids != derived_phase_ids:
            raise ValueError("phase_ids must match node profile phases")
        if self.node_count != len(self.node_profiles):
            raise ValueError("node_count must match node_profiles")

        source_registries = set(self.source_registries)
        profile_source_registries = {
            source_registry
            for profile in self.node_profiles
            for source_registry in profile.source_registries
        }
        if source_registries != profile_source_registries:
            raise ValueError("source_registries must match node profile sources")
        for profile in self.node_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("node profile sources must match registry sources")
            if profile.workflow_order_mutation_implemented:
                raise ValueError("V3 backbone profiles must not mutate workflow order")
            if profile.multi_agent_escalation_executed:
                raise ValueError("V3 backbone profiles must not execute escalation")
        return self


def v3_backbone_mode_registry() -> V3BackboneModeRegistry:
    """Return passive V3 backbone mode metadata without changing workflow behavior."""

    return V3_BACKBONE_MODE_REGISTRY


def v3_backbone_mode_profile_by_node_id(
    node_id: str,
    registry: V3BackboneModeRegistry | None = None,
) -> V3BackboneModeNodeProfile | None:
    """Return one V3 backbone node profile without executing workflow changes."""

    source_registry = registry or V3_BACKBONE_MODE_REGISTRY
    for profile in source_registry.node_profiles:
        if profile.node_id == node_id:
            return profile
    return None


class ConditionalMultiAgentEscalationCondition(BaseModel):
    """Passive condition metadata for future multi-agent escalation candidates."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    condition_id: str = Field(min_length=1, max_length=120)
    condition_name: str = Field(min_length=1, max_length=160)
    category: ConditionalEscalationCategory
    backbone_phase: BackboneModePhase
    source_node_ids: tuple[str, ...] = Field(min_length=1, max_length=6)
    source_registries: tuple[str, ...] = Field(min_length=5, max_length=5)
    capability_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    policy_rule_ids: tuple[str, ...] = Field(min_length=1, max_length=3)
    escalation_signal_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(min_length=1, max_length=900)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_CONDITIONAL_ESCALATION_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    condition_evaluation_implemented: Literal[False] = False
    multi_agent_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal[
        "conditional_multi_agent_escalation_condition.v1"
    ] = CONDITIONAL_ESCALATION_CONDITION_SERIALIZATION_VERSION
    metadata_only: Literal[True] = True


class ConditionalMultiAgentEscalationRegistry(BaseModel):
    """Stable passive registry for conditional multi-agent escalation metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["conditional_multi_agent_escalation_registry"] = (
        "conditional_multi_agent_escalation_registry"
    )
    serialization_version: Literal[
        "conditional_multi_agent_escalation_registry.v1"
    ] = CONDITIONAL_ESCALATION_REGISTRY_SERIALIZATION_VERSION
    authority_boundary: str = Field(
        default=CONDITIONAL_ESCALATION_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    conditions: tuple[ConditionalMultiAgentEscalationCondition, ...] = Field(
        min_length=5,
        max_length=5,
    )
    condition_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    categories: tuple[ConditionalEscalationCategory, ...] = Field(
        min_length=5,
        max_length=5,
    )
    source_registries: tuple[str, ...] = Field(min_length=5, max_length=5)
    backbone_node_ids: tuple[str, ...] = Field(min_length=18, max_length=18)
    condition_count: int = Field(ge=5, le=5)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_CONDITIONAL_ESCALATION_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    condition_evaluation_implemented: Literal[False] = False
    multi_agent_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_conditional_escalation_metadata(self) -> Self:
        derived_condition_ids = tuple(
            condition.condition_id for condition in self.conditions
        )
        derived_categories = tuple(condition.category for condition in self.conditions)
        if self.condition_ids != derived_condition_ids:
            raise ValueError("condition_ids must match conditions")
        if len(set(self.condition_ids)) != len(self.condition_ids):
            raise ValueError("condition_ids must be unique")
        if self.categories != derived_categories:
            raise ValueError("categories must match conditions")
        if self.condition_count != len(self.conditions):
            raise ValueError("condition_count must match conditions")

        source_registries = set(self.source_registries)
        condition_source_registries = {
            source_registry
            for condition in self.conditions
            for source_registry in condition.source_registries
        }
        if source_registries != condition_source_registries:
            raise ValueError("source_registries must match condition sources")

        known_nodes = set(self.backbone_node_ids)
        known_capabilities = set(_KNOWN_CONDITIONAL_ESCALATION_CAPABILITY_IDS)
        known_policies = set(_KNOWN_CONDITIONAL_ESCALATION_POLICY_RULE_IDS)
        known_signals = set(_KNOWN_CONDITIONAL_ESCALATION_SIGNAL_IDS)
        for condition in self.conditions:
            if condition.source_registries != self.source_registries:
                raise ValueError("condition sources must match registry sources")
            if not set(condition.source_node_ids).issubset(known_nodes):
                raise ValueError("condition source nodes must be V3 backbone nodes")
            if not set(condition.capability_ids).issubset(known_capabilities):
                raise ValueError("condition capabilities must be known metadata")
            if not set(condition.policy_rule_ids).issubset(known_policies):
                raise ValueError("condition policy rules must be known metadata")
            if not set(condition.escalation_signal_ids).issubset(known_signals):
                raise ValueError("condition signals must be known metadata")
            if condition.condition_evaluation_implemented:
                raise ValueError("conditions must remain unevaluated metadata")
            if condition.multi_agent_execution_implemented:
                raise ValueError("conditions must not execute multi-agent workflow")
        return self


def conditional_multi_agent_escalation_registry() -> (
    ConditionalMultiAgentEscalationRegistry
):
    """Return passive conditional escalation metadata without invoking agents."""

    return CONDITIONAL_MULTI_AGENT_ESCALATION_REGISTRY


def conditional_multi_agent_escalation_condition_by_id(
    condition_id: str,
    registry: ConditionalMultiAgentEscalationRegistry | None = None,
) -> ConditionalMultiAgentEscalationCondition | None:
    """Return one escalation condition without evaluating it."""

    source_registry = registry or CONDITIONAL_MULTI_AGENT_ESCALATION_REGISTRY
    for condition in source_registry.conditions:
        if condition.condition_id == condition_id:
            return condition
    return None


class SpecialistAgentLoopProfile(BaseModel):
    """Passive profile for one future specialist-agent loop candidate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    loop_id: str = Field(min_length=1, max_length=120)
    loop_name: str = Field(min_length=1, max_length=160)
    category: SpecialistLoopCategory
    specialist_agent_ids: tuple[str, ...] = Field(min_length=1, max_length=6)
    source_condition_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_node_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_registries: tuple[str, ...] = Field(min_length=4, max_length=4)
    loop_inputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    max_advisory_passes: int = Field(ge=1, le=3)
    authority_boundary: str = Field(min_length=1, max_length=900)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_SPECIALIST_AGENT_LOOP_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    loop_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    multi_agent_orchestration_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["specialist_agent_loop.v1"] = (
        SPECIALIST_AGENT_LOOP_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class SpecialistAgentLoopRegistry(BaseModel):
    """Stable passive registry for specialist-agent loop metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["specialist_agent_loop_registry"] = (
        "specialist_agent_loop_registry"
    )
    serialization_version: Literal["specialist_agent_loop_registry.v1"] = (
        SPECIALIST_AGENT_LOOP_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=SPECIALIST_AGENT_LOOP_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    loops: tuple[SpecialistAgentLoopProfile, ...] = Field(
        min_length=5,
        max_length=5,
    )
    loop_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    categories: tuple[SpecialistLoopCategory, ...] = Field(
        min_length=5,
        max_length=5,
    )
    source_registries: tuple[str, ...] = Field(min_length=4, max_length=4)
    agent_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    condition_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    backbone_node_ids: tuple[str, ...] = Field(min_length=18, max_length=18)
    loop_count: int = Field(ge=5, le=5)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_SPECIALIST_AGENT_LOOP_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    loop_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    multi_agent_orchestration_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_specialist_loop_metadata(self) -> Self:
        derived_loop_ids = tuple(loop.loop_id for loop in self.loops)
        derived_categories = tuple(loop.category for loop in self.loops)
        if self.loop_ids != derived_loop_ids:
            raise ValueError("loop_ids must match loops")
        if len(set(self.loop_ids)) != len(self.loop_ids):
            raise ValueError("loop_ids must be unique")
        if self.categories != derived_categories:
            raise ValueError("categories must match loops")
        if self.loop_count != len(self.loops):
            raise ValueError("loop_count must match loops")

        source_registries = set(self.source_registries)
        loop_source_registries = {
            source_registry
            for loop in self.loops
            for source_registry in loop.source_registries
        }
        if source_registries != loop_source_registries:
            raise ValueError("source_registries must match loop sources")

        known_agents = set(self.agent_ids)
        known_conditions = set(self.condition_ids)
        known_nodes = set(self.backbone_node_ids)
        for loop in self.loops:
            if loop.source_registries != self.source_registries:
                raise ValueError("loop sources must match registry sources")
            if not set(loop.specialist_agent_ids).issubset(known_agents):
                raise ValueError("loop agents must be known passive agents")
            if not set(loop.source_condition_ids).issubset(known_conditions):
                raise ValueError("loop conditions must be known metadata")
            if not set(loop.source_node_ids).issubset(known_nodes):
                raise ValueError("loop source nodes must be V3 backbone nodes")
            if loop.loop_execution_implemented:
                raise ValueError("specialist loops must not execute")
            if loop.agent_invocation_implemented:
                raise ValueError("specialist loops must not invoke agents")
        return self


def specialist_agent_loop_registry() -> SpecialistAgentLoopRegistry:
    """Return passive specialist loop metadata without executing loops."""

    return SPECIALIST_AGENT_LOOP_REGISTRY


def specialist_agent_loop_by_id(
    loop_id: str,
    registry: SpecialistAgentLoopRegistry | None = None,
) -> SpecialistAgentLoopProfile | None:
    """Return one specialist loop profile without invoking agents."""

    source_registry = registry or SPECIALIST_AGENT_LOOP_REGISTRY
    for loop in source_registry.loops:
        if loop.loop_id == loop_id:
            return loop
    return None


class EscalationGateProfile(BaseModel):
    """Passive advisory gate metadata for future escalation readiness."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    gate_id: str = Field(min_length=1, max_length=120)
    gate_name: str = Field(min_length=1, max_length=160)
    gate_kind: EscalationGateKind
    source_condition_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_loop_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    source_registries: tuple[str, ...] = Field(min_length=5, max_length=5)
    required_passive_inputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_decision_outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(min_length=1, max_length=900)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_ESCALATION_GATE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    gate_evaluation_implemented: Literal[False] = False
    escalation_approval_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["escalation_gate.v1"] = (
        ESCALATION_GATE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class EscalationGateRegistry(BaseModel):
    """Stable passive registry for V4.3 escalation gate metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["escalation_gate_registry"] = "escalation_gate_registry"
    serialization_version: Literal["escalation_gate_registry.v1"] = (
        ESCALATION_GATE_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=ESCALATION_GATE_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    gates: tuple[EscalationGateProfile, ...] = Field(min_length=5, max_length=5)
    gate_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    gate_kinds: tuple[EscalationGateKind, ...] = Field(min_length=5, max_length=5)
    source_registries: tuple[str, ...] = Field(min_length=5, max_length=5)
    condition_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    loop_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    gate_count: int = Field(ge=5, le=5)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_ESCALATION_GATE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    gate_evaluation_implemented: Literal[False] = False
    escalation_approval_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_escalation_gate_metadata(self) -> Self:
        derived_gate_ids = tuple(gate.gate_id for gate in self.gates)
        derived_gate_kinds = tuple(gate.gate_kind for gate in self.gates)
        if self.gate_ids != derived_gate_ids:
            raise ValueError("gate_ids must match gates")
        if len(set(self.gate_ids)) != len(self.gate_ids):
            raise ValueError("gate_ids must be unique")
        if self.gate_kinds != derived_gate_kinds:
            raise ValueError("gate_kinds must match gates")
        if self.gate_count != len(self.gates):
            raise ValueError("gate_count must match gates")

        source_registries = set(self.source_registries)
        gate_source_registries = {
            source_registry
            for gate in self.gates
            for source_registry in gate.source_registries
        }
        if source_registries != gate_source_registries:
            raise ValueError("source_registries must match gate sources")

        known_conditions = set(self.condition_ids)
        known_loops = set(self.loop_ids)
        for gate in self.gates:
            if gate.source_registries != self.source_registries:
                raise ValueError("gate sources must match registry sources")
            if not set(gate.source_condition_ids).issubset(known_conditions):
                raise ValueError("gate conditions must be known metadata")
            if not set(gate.source_loop_ids).issubset(known_loops):
                raise ValueError("gate loops must be known metadata")
            if gate.gate_evaluation_implemented:
                raise ValueError("escalation gates must not evaluate")
            if gate.escalation_approval_implemented:
                raise ValueError("escalation gates must not approve escalation")
        return self


def escalation_gate_registry() -> EscalationGateRegistry:
    """Return passive escalation gate metadata without evaluating gates."""

    return ESCALATION_GATE_REGISTRY


def escalation_gate_by_id(
    gate_id: str,
    registry: EscalationGateRegistry | None = None,
) -> EscalationGateProfile | None:
    """Return one escalation gate profile without evaluating it."""

    source_registry = registry or ESCALATION_GATE_REGISTRY
    for gate in source_registry.gates:
        if gate.gate_id == gate_id:
            return gate
    return None


class CreativeEscalationPolicyRule(BaseModel):
    """Passive creative-domain escalation policy rule metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    policy_id: str = Field(min_length=1, max_length=120)
    policy_name: str = Field(min_length=1, max_length=160)
    category: CreativeEscalationPolicyCategory
    source_gate_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_loop_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    creative_signal_sources: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_policy_outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(min_length=1, max_length=900)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_CREATIVE_ESCALATION_POLICY_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    creative_policy_evaluation_implemented: Literal[False] = False
    escalation_approval_implemented: Literal[False] = False
    gate_evaluation_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["creative_escalation_policy_rule.v1"] = (
        CREATIVE_ESCALATION_POLICY_RULE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class CreativeEscalationPolicyRegistry(BaseModel):
    """Stable passive registry for creative escalation policy metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["creative_escalation_policy_registry"] = (
        "creative_escalation_policy_registry"
    )
    serialization_version: Literal["creative_escalation_policy_registry.v1"] = (
        CREATIVE_ESCALATION_POLICY_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CREATIVE_ESCALATION_POLICY_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    policies: tuple[CreativeEscalationPolicyRule, ...] = Field(
        min_length=5,
        max_length=5,
    )
    policy_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    categories: tuple[CreativeEscalationPolicyCategory, ...] = Field(
        min_length=5,
        max_length=5,
    )
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    gate_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    loop_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    policy_count: int = Field(ge=5, le=5)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_CREATIVE_ESCALATION_POLICY_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    creative_policy_evaluation_implemented: Literal[False] = False
    escalation_approval_implemented: Literal[False] = False
    gate_evaluation_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_creative_policy_metadata(self) -> Self:
        derived_policy_ids = tuple(policy.policy_id for policy in self.policies)
        derived_categories = tuple(policy.category for policy in self.policies)
        if self.policy_ids != derived_policy_ids:
            raise ValueError("policy_ids must match policies")
        if len(set(self.policy_ids)) != len(self.policy_ids):
            raise ValueError("policy_ids must be unique")
        if self.categories != derived_categories:
            raise ValueError("categories must match policies")
        if self.policy_count != len(self.policies):
            raise ValueError("policy_count must match policies")

        source_registries = set(self.source_registries)
        policy_source_registries = {
            source_registry
            for policy in self.policies
            for source_registry in policy.source_registries
        }
        if source_registries != policy_source_registries:
            raise ValueError("source_registries must match policy sources")

        known_gates = set(self.gate_ids)
        known_loops = set(self.loop_ids)
        for policy in self.policies:
            if policy.source_registries != self.source_registries:
                raise ValueError("policy sources must match registry sources")
            if not set(policy.source_gate_ids).issubset(known_gates):
                raise ValueError("policy gates must be known metadata")
            if not set(policy.source_loop_ids).issubset(known_loops):
                raise ValueError("policy loops must be known metadata")
            if policy.creative_policy_evaluation_implemented:
                raise ValueError("creative policies must not evaluate policy")
            if policy.escalation_approval_implemented:
                raise ValueError("creative policies must not approve escalation")
        return self


def creative_escalation_policy_registry() -> CreativeEscalationPolicyRegistry:
    """Return passive creative escalation policy metadata."""

    return CREATIVE_ESCALATION_POLICY_REGISTRY


def creative_escalation_policy_by_id(
    policy_id: str,
    registry: CreativeEscalationPolicyRegistry | None = None,
) -> CreativeEscalationPolicyRule | None:
    """Return one creative policy rule without evaluating it."""

    source_registry = registry or CREATIVE_ESCALATION_POLICY_REGISTRY
    for policy in source_registry.policies:
        if policy.policy_id == policy_id:
            return policy
    return None


class ReflectionEscalationProfile(BaseModel):
    """Passive reflection escalation posture metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    profile_id: str = Field(min_length=1, max_length=120)
    profile_name: str = Field(min_length=1, max_length=160)
    posture: ReflectionEscalationPosture
    reflection_priority: ReflectionEscalationPosture
    source_policy_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_gate_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_registries: tuple[str, ...] = Field(min_length=5, max_length=5)
    reflection_signal_sources: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(min_length=1, max_length=900)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_REFLECTION_ESCALATION_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    reflection_execution_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    escalation_approval_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["reflection_escalation_profile.v1"] = (
        REFLECTION_ESCALATION_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class ReflectionEscalationRegistry(BaseModel):
    """Stable passive registry for reflection escalation metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["reflection_escalation_registry"] = (
        "reflection_escalation_registry"
    )
    serialization_version: Literal["reflection_escalation_registry.v1"] = (
        REFLECTION_ESCALATION_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=REFLECTION_ESCALATION_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    profiles: tuple[ReflectionEscalationProfile, ...] = Field(
        min_length=5,
        max_length=5,
    )
    profile_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    postures: tuple[ReflectionEscalationPosture, ...] = Field(
        min_length=5,
        max_length=5,
    )
    source_registries: tuple[str, ...] = Field(min_length=5, max_length=5)
    policy_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    gate_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    profile_count: int = Field(ge=5, le=5)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_REFLECTION_ESCALATION_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    reflection_execution_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    escalation_approval_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_reflection_escalation_metadata(self) -> Self:
        derived_profile_ids = tuple(profile.profile_id for profile in self.profiles)
        derived_postures = tuple(profile.posture for profile in self.profiles)
        if self.profile_ids != derived_profile_ids:
            raise ValueError("profile_ids must match profiles")
        if self.postures != derived_postures:
            raise ValueError("postures must match profiles")
        if self.profile_count != len(self.profiles):
            raise ValueError("profile_count must match profiles")
        if len(set(self.profile_ids)) != len(self.profile_ids):
            raise ValueError("profile_ids must be unique")

        source_registries = set(self.source_registries)
        profile_sources = {
            source_registry
            for profile in self.profiles
            for source_registry in profile.source_registries
        }
        if source_registries != profile_sources:
            raise ValueError("source_registries must match profile sources")

        known_policies = set(self.policy_ids)
        known_gates = set(self.gate_ids)
        for profile in self.profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("profile sources must match registry sources")
            if not set(profile.source_policy_ids).issubset(known_policies):
                raise ValueError("reflection policies must be known metadata")
            if not set(profile.source_gate_ids).issubset(known_gates):
                raise ValueError("reflection gates must be known metadata")
            if profile.reflection_execution_implemented:
                raise ValueError("reflection escalation must not execute reflection")
            if profile.refinement_triggering_implemented:
                raise ValueError("reflection escalation must not trigger refinement")
        return self


def reflection_escalation_registry() -> ReflectionEscalationRegistry:
    """Return passive reflection escalation metadata."""

    return REFLECTION_ESCALATION_REGISTRY


def reflection_escalation_profile_by_id(
    profile_id: str,
    registry: ReflectionEscalationRegistry | None = None,
) -> ReflectionEscalationProfile | None:
    """Return one reflection escalation profile without executing reflection."""

    source_registry = registry or REFLECTION_ESCALATION_REGISTRY
    for profile in source_registry.profiles:
        if profile.profile_id == profile_id:
            return profile
    return None


class HybridAgentDebateLoopProfile(BaseModel):
    """Passive V4.3 debate loop readiness profile."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    loop_id: str = Field(min_length=1, max_length=140)
    topic_id: HybridDebateLoopTopic
    source_debate_topic_id: HybridDebateLoopTopic
    source_reflection_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_policy_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_specialist_loop_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_registries: tuple[str, ...] = Field(min_length=5, max_length=5)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    max_advisory_rounds: int = Field(ge=1, le=2)
    authority_boundary: str = Field(min_length=1, max_length=900)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_HYBRID_DEBATE_LOOP_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    debate_loop_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["hybrid_agent_debate_loop_profile.v1"] = (
        HYBRID_DEBATE_LOOP_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class HybridAgentDebateLoopRegistry(BaseModel):
    """Stable passive registry for V4.3 hybrid debate loop metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["hybrid_agent_debate_loop_registry"] = (
        "hybrid_agent_debate_loop_registry"
    )
    serialization_version: Literal["hybrid_agent_debate_loop_registry.v1"] = (
        HYBRID_DEBATE_LOOP_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=HYBRID_DEBATE_LOOP_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    debate_loops: tuple[HybridAgentDebateLoopProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    loop_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    topic_ids: tuple[HybridDebateLoopTopic, ...] = Field(min_length=4, max_length=4)
    source_registries: tuple[str, ...] = Field(min_length=5, max_length=5)
    reflection_profile_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    policy_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    specialist_loop_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    loop_count: int = Field(ge=4, le=4)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_HYBRID_DEBATE_LOOP_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    debate_loop_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_hybrid_debate_loop_metadata(self) -> Self:
        derived_loop_ids = tuple(loop.loop_id for loop in self.debate_loops)
        derived_topic_ids = tuple(loop.topic_id for loop in self.debate_loops)
        if self.loop_ids != derived_loop_ids:
            raise ValueError("loop_ids must match debate_loops")
        if self.topic_ids != derived_topic_ids:
            raise ValueError("topic_ids must match debate_loops")
        if self.topic_ids != _HYBRID_DEBATE_LOOP_TOPICS:
            raise ValueError("topic_ids must preserve debate topic order")
        if self.loop_count != len(self.debate_loops):
            raise ValueError("loop_count must match debate_loops")

        source_registries = set(self.source_registries)
        loop_sources = {
            source_registry
            for loop in self.debate_loops
            for source_registry in loop.source_registries
        }
        if source_registries != loop_sources:
            raise ValueError("source_registries must match debate loop sources")

        known_reflections = set(self.reflection_profile_ids)
        known_policies = set(self.policy_ids)
        known_loops = set(self.specialist_loop_ids)
        for loop in self.debate_loops:
            if loop.source_registries != self.source_registries:
                raise ValueError("debate loop sources must match registry sources")
            if not set(loop.source_reflection_profile_ids).issubset(known_reflections):
                raise ValueError("debate reflections must be known metadata")
            if not set(loop.source_policy_ids).issubset(known_policies):
                raise ValueError("debate policies must be known metadata")
            if not set(loop.source_specialist_loop_ids).issubset(known_loops):
                raise ValueError("debate specialist loops must be known metadata")
            if loop.debate_loop_execution_implemented:
                raise ValueError("hybrid debate loops must not execute")
        return self


def hybrid_agent_debate_loop_registry() -> HybridAgentDebateLoopRegistry:
    """Return passive V4.3 hybrid debate loop metadata."""

    return HYBRID_AGENT_DEBATE_LOOP_REGISTRY


def hybrid_agent_debate_loop_by_id(
    loop_id: str,
    registry: HybridAgentDebateLoopRegistry | None = None,
) -> HybridAgentDebateLoopProfile | None:
    """Return one hybrid debate loop profile without executing debate."""

    source_registry = registry or HYBRID_AGENT_DEBATE_LOOP_REGISTRY
    for loop in source_registry.debate_loops:
        if loop.loop_id == loop_id:
            return loop
    return None


class HybridAgentVotingProfile(BaseModel):
    """Passive V4.3 voting profile metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    voting_profile_id: str = Field(min_length=1, max_length=140)
    topic_id: HybridVotingTopic
    source_debate_loop_id: str = Field(min_length=1, max_length=160)
    consensus_voting_input_id: str = Field(min_length=1, max_length=160)
    source_reflection_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_policy_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_registries: tuple[str, ...] = Field(min_length=5, max_length=5)
    voting_dimensions: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(min_length=1, max_length=900)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_HYBRID_AGENT_VOTING_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    voting_execution_implemented: Literal[False] = False
    final_answer_selection_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["hybrid_agent_voting_profile.v1"] = (
        HYBRID_AGENT_VOTING_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class HybridAgentVotingRegistry(BaseModel):
    """Stable passive registry for V4.3 hybrid agent voting metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["hybrid_agent_voting_registry"] = "hybrid_agent_voting_registry"
    serialization_version: Literal["hybrid_agent_voting_registry.v1"] = (
        HYBRID_AGENT_VOTING_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=HYBRID_AGENT_VOTING_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    voting_profiles: tuple[HybridAgentVotingProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    voting_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    topic_ids: tuple[HybridVotingTopic, ...] = Field(min_length=4, max_length=4)
    source_registries: tuple[str, ...] = Field(min_length=5, max_length=5)
    debate_loop_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    reflection_profile_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    policy_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    profile_count: int = Field(ge=4, le=4)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_HYBRID_AGENT_VOTING_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    voting_execution_implemented: Literal[False] = False
    final_answer_selection_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_hybrid_voting_metadata(self) -> Self:
        derived_profile_ids = tuple(
            profile.voting_profile_id for profile in self.voting_profiles
        )
        derived_topic_ids = tuple(profile.topic_id for profile in self.voting_profiles)
        if self.voting_profile_ids != derived_profile_ids:
            raise ValueError("voting_profile_ids must match voting_profiles")
        if self.topic_ids != derived_topic_ids:
            raise ValueError("topic_ids must match voting_profiles")
        if self.topic_ids != _HYBRID_AGENT_VOTING_TOPICS:
            raise ValueError("topic_ids must preserve voting topic order")
        if self.profile_count != len(self.voting_profiles):
            raise ValueError("profile_count must match voting_profiles")

        profile_sources = {
            source_registry
            for profile in self.voting_profiles
            for source_registry in profile.source_registries
        }
        if set(self.source_registries) != profile_sources:
            raise ValueError("source_registries must match voting profile sources")

        known_debates = set(self.debate_loop_ids)
        known_reflections = set(self.reflection_profile_ids)
        known_policies = set(self.policy_ids)
        for profile in self.voting_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("voting sources must match registry sources")
            if profile.source_debate_loop_id not in known_debates:
                raise ValueError("voting debate loops must be known metadata")
            if not set(profile.source_reflection_profile_ids).issubset(known_reflections):
                raise ValueError("voting reflections must be known metadata")
            if not set(profile.source_policy_ids).issubset(known_policies):
                raise ValueError("voting policies must be known metadata")
            if profile.voting_execution_implemented:
                raise ValueError("hybrid voting must not execute voting")
        return self


def hybrid_agent_voting_registry() -> HybridAgentVotingRegistry:
    """Return passive V4.3 hybrid agent voting metadata."""

    return HYBRID_AGENT_VOTING_REGISTRY


def hybrid_agent_voting_profile_by_id(
    voting_profile_id: str,
    registry: HybridAgentVotingRegistry | None = None,
) -> HybridAgentVotingProfile | None:
    """Return one voting profile without executing voting."""

    source_registry = registry or HYBRID_AGENT_VOTING_REGISTRY
    for profile in source_registry.voting_profiles:
        if profile.voting_profile_id == voting_profile_id:
            return profile
    return None


class AgentConfidenceFusionProfile(BaseModel):
    """Passive V4.3 confidence fusion profile metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    fusion_profile_id: str = Field(min_length=1, max_length=140)
    topic_id: AgentConfidenceFusionTopic
    source_voting_profile_id: str = Field(min_length=1, max_length=160)
    source_debate_loop_id: str = Field(min_length=1, max_length=160)
    source_confidence_surface_id: str = Field(min_length=1, max_length=120)
    source_reflection_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    confidence_signal_inputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    fusion_dimensions: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(min_length=1, max_length=900)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_AGENT_CONFIDENCE_FUSION_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    confidence_fusion_implemented: Literal[False] = False
    confidence_score_calculation_implemented: Literal[False] = False
    vote_weighting_implemented: Literal[False] = False
    final_answer_selection_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["agent_confidence_fusion_profile.v1"] = (
        AGENT_CONFIDENCE_FUSION_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AgentConfidenceFusionRegistry(BaseModel):
    """Stable passive registry for V4.3 agent confidence fusion metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["agent_confidence_fusion_registry"] = (
        "agent_confidence_fusion_registry"
    )
    serialization_version: Literal["agent_confidence_fusion_registry.v1"] = (
        AGENT_CONFIDENCE_FUSION_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=AGENT_CONFIDENCE_FUSION_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    fusion_profiles: tuple[AgentConfidenceFusionProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    fusion_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    topic_ids: tuple[AgentConfidenceFusionTopic, ...] = Field(
        min_length=4,
        max_length=4,
    )
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    voting_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    debate_loop_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    reflection_profile_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    confidence_surface_ids: tuple[str, ...] = Field(min_length=1, max_length=1)
    profile_count: int = Field(ge=4, le=4)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_AGENT_CONFIDENCE_FUSION_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    confidence_fusion_implemented: Literal[False] = False
    confidence_score_calculation_implemented: Literal[False] = False
    vote_weighting_implemented: Literal[False] = False
    final_answer_selection_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_confidence_fusion_metadata(self) -> Self:
        derived_profile_ids = tuple(
            profile.fusion_profile_id for profile in self.fusion_profiles
        )
        derived_topic_ids = tuple(profile.topic_id for profile in self.fusion_profiles)
        if self.fusion_profile_ids != derived_profile_ids:
            raise ValueError("fusion_profile_ids must match fusion_profiles")
        if self.topic_ids != derived_topic_ids:
            raise ValueError("topic_ids must match fusion_profiles")
        if self.topic_ids != _AGENT_CONFIDENCE_FUSION_TOPICS:
            raise ValueError("topic_ids must preserve confidence fusion topic order")
        if self.profile_count != len(self.fusion_profiles):
            raise ValueError("profile_count must match fusion_profiles")

        profile_sources = {
            source_registry
            for profile in self.fusion_profiles
            for source_registry in profile.source_registries
        }
        if set(self.source_registries) != profile_sources:
            raise ValueError("source_registries must match fusion profile sources")

        known_votes = set(self.voting_profile_ids)
        known_debates = set(self.debate_loop_ids)
        known_reflections = set(self.reflection_profile_ids)
        known_confidence_surfaces = set(self.confidence_surface_ids)
        for profile in self.fusion_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("fusion sources must match registry sources")
            if profile.source_voting_profile_id not in known_votes:
                raise ValueError("fusion voting profiles must be known metadata")
            if profile.source_debate_loop_id not in known_debates:
                raise ValueError("fusion debate loops must be known metadata")
            if profile.source_confidence_surface_id not in known_confidence_surfaces:
                raise ValueError("fusion confidence surfaces must be known metadata")
            if not set(profile.source_reflection_profile_ids).issubset(
                known_reflections
            ):
                raise ValueError("fusion reflections must be known metadata")
            if profile.confidence_fusion_implemented:
                raise ValueError("agent confidence fusion must not execute fusion")
        return self


def agent_confidence_fusion_registry() -> AgentConfidenceFusionRegistry:
    """Return passive V4.3 agent confidence fusion metadata."""

    return AGENT_CONFIDENCE_FUSION_REGISTRY


def agent_confidence_fusion_profile_by_id(
    fusion_profile_id: str,
    registry: AgentConfidenceFusionRegistry | None = None,
) -> AgentConfidenceFusionProfile | None:
    """Return one confidence fusion profile without executing fusion."""

    source_registry = registry or AGENT_CONFIDENCE_FUSION_REGISTRY
    for profile in source_registry.fusion_profiles:
        if profile.fusion_profile_id == fusion_profile_id:
            return profile
    return None


class DecisionProvenanceProfile(BaseModel):
    """Passive V4.3 decision provenance profile metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    provenance_profile_id: str = Field(min_length=1, max_length=140)
    topic_id: DecisionProvenanceTopic
    source_confidence_fusion_profile_id: str = Field(min_length=1, max_length=160)
    source_voting_profile_id: str = Field(min_length=1, max_length=160)
    source_debate_loop_id: str = Field(min_length=1, max_length=160)
    source_backbone_node_ids: tuple[str, ...] = Field(min_length=1, max_length=6)
    source_workstation_surface_id: str = Field(min_length=1, max_length=120)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    provenance_dimensions: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(min_length=1, max_length=900)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_DECISION_PROVENANCE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    provenance_recording_implemented: Literal[False] = False
    decision_logging_implemented: Literal[False] = False
    trace_emission_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
    decision_selection_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["decision_provenance_profile.v1"] = (
        DECISION_PROVENANCE_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class DecisionProvenanceRegistry(BaseModel):
    """Stable passive registry for V4.3 decision provenance metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["decision_provenance_registry"] = "decision_provenance_registry"
    serialization_version: Literal["decision_provenance_registry.v1"] = (
        DECISION_PROVENANCE_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=DECISION_PROVENANCE_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    provenance_profiles: tuple[DecisionProvenanceProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    provenance_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    topic_ids: tuple[DecisionProvenanceTopic, ...] = Field(
        min_length=4,
        max_length=4,
    )
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    confidence_fusion_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    voting_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    debate_loop_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    backbone_node_ids: tuple[str, ...] = Field(min_length=18, max_length=18)
    workstation_surface_ids: tuple[str, ...] = Field(min_length=7, max_length=7)
    profile_count: int = Field(ge=4, le=4)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_DECISION_PROVENANCE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    provenance_recording_implemented: Literal[False] = False
    decision_logging_implemented: Literal[False] = False
    trace_emission_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
    decision_selection_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_decision_provenance_metadata(self) -> Self:
        derived_profile_ids = tuple(
            profile.provenance_profile_id for profile in self.provenance_profiles
        )
        derived_topic_ids = tuple(
            profile.topic_id for profile in self.provenance_profiles
        )
        if self.provenance_profile_ids != derived_profile_ids:
            raise ValueError("provenance_profile_ids must match profiles")
        if self.topic_ids != derived_topic_ids:
            raise ValueError("topic_ids must match provenance profiles")
        if self.topic_ids != _DECISION_PROVENANCE_TOPICS:
            raise ValueError("topic_ids must preserve provenance topic order")
        if self.profile_count != len(self.provenance_profiles):
            raise ValueError("profile_count must match provenance profiles")

        profile_sources = {
            source_registry
            for profile in self.provenance_profiles
            for source_registry in profile.source_registries
        }
        if set(self.source_registries) != profile_sources:
            raise ValueError("source_registries must match provenance profile sources")

        known_fusion = set(self.confidence_fusion_profile_ids)
        known_votes = set(self.voting_profile_ids)
        known_debates = set(self.debate_loop_ids)
        known_nodes = set(self.backbone_node_ids)
        known_surfaces = set(self.workstation_surface_ids)
        for profile in self.provenance_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("provenance sources must match registry sources")
            if profile.source_confidence_fusion_profile_id not in known_fusion:
                raise ValueError("provenance fusion profiles must be known metadata")
            if profile.source_voting_profile_id not in known_votes:
                raise ValueError("provenance voting profiles must be known metadata")
            if profile.source_debate_loop_id not in known_debates:
                raise ValueError("provenance debate loops must be known metadata")
            if not set(profile.source_backbone_node_ids).issubset(known_nodes):
                raise ValueError("provenance backbone nodes must be known metadata")
            if profile.source_workstation_surface_id not in known_surfaces:
                raise ValueError("provenance workstation surfaces must be known metadata")
            if profile.provenance_recording_implemented:
                raise ValueError("decision provenance must not record provenance")
        return self


def decision_provenance_registry() -> DecisionProvenanceRegistry:
    """Return passive V4.3 decision provenance metadata."""

    return DECISION_PROVENANCE_REGISTRY


def decision_provenance_profile_by_id(
    provenance_profile_id: str,
    registry: DecisionProvenanceRegistry | None = None,
) -> DecisionProvenanceProfile | None:
    """Return one decision provenance profile without recording provenance."""

    source_registry = registry or DECISION_PROVENANCE_REGISTRY
    for profile in source_registry.provenance_profiles:
        if profile.provenance_profile_id == provenance_profile_id:
            return profile
    return None


class EscalationTraceProfile(BaseModel):
    """Passive V4.3 escalation trace profile metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    trace_profile_id: str = Field(min_length=1, max_length=140)
    topic_id: EscalationTraceTopic
    source_provenance_profile_id: str = Field(min_length=1, max_length=160)
    source_condition_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_gate_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_escalation_signal_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_reflection_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    trace_dimensions: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(min_length=1, max_length=900)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_ESCALATION_TRACE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    trace_capture_implemented: Literal[False] = False
    trace_emission_implemented: Literal[False] = False
    escalation_execution_implemented: Literal[False] = False
    gate_evaluation_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["escalation_trace_profile.v1"] = (
        ESCALATION_TRACE_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class EscalationTraceRegistry(BaseModel):
    """Stable passive registry for V4.3 escalation trace metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["escalation_trace_registry"] = "escalation_trace_registry"
    serialization_version: Literal["escalation_trace_registry.v1"] = (
        ESCALATION_TRACE_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=ESCALATION_TRACE_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    trace_profiles: tuple[EscalationTraceProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    trace_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    topic_ids: tuple[EscalationTraceTopic, ...] = Field(min_length=4, max_length=4)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    provenance_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    condition_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    gate_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    escalation_signal_ids: tuple[str, ...] = Field(min_length=7, max_length=7)
    reflection_profile_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    profile_count: int = Field(ge=4, le=4)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_ESCALATION_TRACE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    trace_capture_implemented: Literal[False] = False
    trace_emission_implemented: Literal[False] = False
    escalation_execution_implemented: Literal[False] = False
    gate_evaluation_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_escalation_trace_metadata(self) -> Self:
        derived_profile_ids = tuple(
            profile.trace_profile_id for profile in self.trace_profiles
        )
        derived_topic_ids = tuple(profile.topic_id for profile in self.trace_profiles)
        if self.trace_profile_ids != derived_profile_ids:
            raise ValueError("trace_profile_ids must match trace_profiles")
        if self.topic_ids != derived_topic_ids:
            raise ValueError("topic_ids must match trace profiles")
        if self.topic_ids != _ESCALATION_TRACE_TOPICS:
            raise ValueError("topic_ids must preserve escalation trace topic order")
        if self.profile_count != len(self.trace_profiles):
            raise ValueError("profile_count must match trace profiles")

        profile_sources = {
            source_registry
            for profile in self.trace_profiles
            for source_registry in profile.source_registries
        }
        if set(self.source_registries) != profile_sources:
            raise ValueError("source_registries must match trace profile sources")

        known_provenance = set(self.provenance_profile_ids)
        known_conditions = set(self.condition_ids)
        known_gates = set(self.gate_ids)
        known_signals = set(self.escalation_signal_ids)
        known_reflections = set(self.reflection_profile_ids)
        for profile in self.trace_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("trace sources must match registry sources")
            if profile.source_provenance_profile_id not in known_provenance:
                raise ValueError("trace provenance profiles must be known metadata")
            if not set(profile.source_condition_ids).issubset(known_conditions):
                raise ValueError("trace conditions must be known metadata")
            if not set(profile.source_gate_ids).issubset(known_gates):
                raise ValueError("trace gates must be known metadata")
            if not set(profile.source_escalation_signal_ids).issubset(known_signals):
                raise ValueError("trace signals must be known metadata")
            if not set(profile.source_reflection_profile_ids).issubset(
                known_reflections
            ):
                raise ValueError("trace reflections must be known metadata")
            if profile.trace_capture_implemented:
                raise ValueError("escalation trace must not capture traces")
        return self


def escalation_trace_registry() -> EscalationTraceRegistry:
    """Return passive V4.3 escalation trace metadata."""

    return ESCALATION_TRACE_REGISTRY


def escalation_trace_profile_by_id(
    trace_profile_id: str,
    registry: EscalationTraceRegistry | None = None,
) -> EscalationTraceProfile | None:
    """Return one escalation trace profile without capturing traces."""

    source_registry = registry or ESCALATION_TRACE_REGISTRY
    for profile in source_registry.trace_profiles:
        if profile.trace_profile_id == trace_profile_id:
            return profile
    return None


class CreativeExplorationBudgetProfile(BaseModel):
    """Passive V4.3 creative exploration budget profile metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    budget_profile_id: str = Field(min_length=1, max_length=150)
    topic_id: CreativeExplorationBudgetTopic
    source_trace_profile_id: str = Field(min_length=1, max_length=160)
    source_provenance_profile_id: str = Field(min_length=1, max_length=160)
    source_escalation_signal_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    budget_posture: CreativeExplorationBudgetPosture
    max_advisory_variants: int = Field(ge=0, le=3)
    max_advisory_refinement_passes: int = Field(ge=0, le=3)
    cost_pressure_signal: str = Field(min_length=1, max_length=120)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    budget_dimensions: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(min_length=1, max_length=900)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_CREATIVE_EXPLORATION_BUDGET_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    budget_enforcement_implemented: Literal[False] = False
    variant_generation_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    cost_routing_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["creative_exploration_budget_profile.v1"] = (
        CREATIVE_EXPLORATION_BUDGET_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class CreativeExplorationBudgetRegistry(BaseModel):
    """Stable passive registry for V4.3 creative exploration budget metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["creative_exploration_budget_registry"] = (
        "creative_exploration_budget_registry"
    )
    serialization_version: Literal["creative_exploration_budget_registry.v1"] = (
        CREATIVE_EXPLORATION_BUDGET_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CREATIVE_EXPLORATION_BUDGET_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    budget_profiles: tuple[CreativeExplorationBudgetProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    budget_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    topic_ids: tuple[CreativeExplorationBudgetTopic, ...] = Field(
        min_length=4,
        max_length=4,
    )
    budget_postures: tuple[CreativeExplorationBudgetPosture, ...] = Field(
        min_length=4,
        max_length=4,
    )
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    trace_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    provenance_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    escalation_signal_ids: tuple[str, ...] = Field(min_length=7, max_length=7)
    profile_count: int = Field(ge=4, le=4)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_CREATIVE_EXPLORATION_BUDGET_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    budget_enforcement_implemented: Literal[False] = False
    variant_generation_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    cost_routing_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_creative_budget_metadata(self) -> Self:
        derived_profile_ids = tuple(
            profile.budget_profile_id for profile in self.budget_profiles
        )
        derived_topic_ids = tuple(profile.topic_id for profile in self.budget_profiles)
        derived_postures = tuple(
            profile.budget_posture for profile in self.budget_profiles
        )
        if self.budget_profile_ids != derived_profile_ids:
            raise ValueError("budget_profile_ids must match budget_profiles")
        if self.topic_ids != derived_topic_ids:
            raise ValueError("topic_ids must match budget profiles")
        if self.topic_ids != _CREATIVE_EXPLORATION_BUDGET_TOPICS:
            raise ValueError("topic_ids must preserve exploration budget topic order")
        if self.budget_postures != derived_postures:
            raise ValueError("budget_postures must match budget profiles")
        if self.budget_postures != _CREATIVE_EXPLORATION_BUDGET_POSTURES:
            raise ValueError("budget_postures must preserve budget posture order")
        if self.profile_count != len(self.budget_profiles):
            raise ValueError("profile_count must match budget profiles")

        profile_sources = {
            source_registry
            for profile in self.budget_profiles
            for source_registry in profile.source_registries
        }
        if set(self.source_registries) != profile_sources:
            raise ValueError("source_registries must match budget profile sources")

        known_traces = set(self.trace_profile_ids)
        known_provenance = set(self.provenance_profile_ids)
        known_signals = set(self.escalation_signal_ids)
        for profile in self.budget_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("budget sources must match registry sources")
            if profile.source_trace_profile_id not in known_traces:
                raise ValueError("budget trace profiles must be known metadata")
            if profile.source_provenance_profile_id not in known_provenance:
                raise ValueError("budget provenance profiles must be known metadata")
            if not set(profile.source_escalation_signal_ids).issubset(known_signals):
                raise ValueError("budget escalation signals must be known metadata")
            if profile.budget_enforcement_implemented:
                raise ValueError("creative exploration budget must not enforce budgets")
        return self


def creative_exploration_budget_registry() -> CreativeExplorationBudgetRegistry:
    """Return passive V4.3 creative exploration budget metadata."""

    return CREATIVE_EXPLORATION_BUDGET_REGISTRY


def creative_exploration_budget_profile_by_id(
    budget_profile_id: str,
    registry: CreativeExplorationBudgetRegistry | None = None,
) -> CreativeExplorationBudgetProfile | None:
    """Return one budget profile without enforcing exploration limits."""

    source_registry = registry or CREATIVE_EXPLORATION_BUDGET_REGISTRY
    for profile in source_registry.budget_profiles:
        if profile.budget_profile_id == budget_profile_id:
            return profile
    return None


class ResultNormalizationProfile(BaseModel):
    """Passive V4.3 result normalization profile metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    normalization_profile_id: str = Field(min_length=1, max_length=150)
    topic_id: ResultNormalizationTopic
    source_budget_profile_id: str = Field(min_length=1, max_length=170)
    source_confidence_fusion_profile_id: str = Field(min_length=1, max_length=170)
    source_provenance_profile_id: str = Field(min_length=1, max_length=170)
    source_trace_profile_id: str = Field(min_length=1, max_length=170)
    normalized_result_surfaces: tuple[str, ...] = Field(min_length=1, max_length=6)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    normalization_dimensions: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(min_length=1, max_length=900)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_RESULT_NORMALIZATION_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    result_normalization_implemented: Literal[False] = False
    output_rewriting_implemented: Literal[False] = False
    schema_enforcement_implemented: Literal[False] = False
    artifact_mutation_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["result_normalization_profile.v1"] = (
        RESULT_NORMALIZATION_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class ResultNormalizationRegistry(BaseModel):
    """Stable passive registry for V4.3 result normalization metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["result_normalization_registry"] = "result_normalization_registry"
    serialization_version: Literal["result_normalization_registry.v1"] = (
        RESULT_NORMALIZATION_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=RESULT_NORMALIZATION_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    normalization_profiles: tuple[ResultNormalizationProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    normalization_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    topic_ids: tuple[ResultNormalizationTopic, ...] = Field(
        min_length=4,
        max_length=4,
    )
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    budget_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    confidence_fusion_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    provenance_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    trace_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    profile_count: int = Field(ge=4, le=4)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_RESULT_NORMALIZATION_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    result_normalization_implemented: Literal[False] = False
    output_rewriting_implemented: Literal[False] = False
    schema_enforcement_implemented: Literal[False] = False
    artifact_mutation_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_result_normalization_metadata(self) -> Self:
        derived_profile_ids = tuple(
            profile.normalization_profile_id
            for profile in self.normalization_profiles
        )
        derived_topic_ids = tuple(
            profile.topic_id for profile in self.normalization_profiles
        )
        if self.normalization_profile_ids != derived_profile_ids:
            raise ValueError("normalization_profile_ids must match profiles")
        if self.topic_ids != derived_topic_ids:
            raise ValueError("topic_ids must match normalization profiles")
        if self.topic_ids != _RESULT_NORMALIZATION_TOPICS:
            raise ValueError("topic_ids must preserve result normalization order")
        if self.profile_count != len(self.normalization_profiles):
            raise ValueError("profile_count must match normalization profiles")

        profile_sources = {
            source_registry
            for profile in self.normalization_profiles
            for source_registry in profile.source_registries
        }
        if set(self.source_registries) != profile_sources:
            raise ValueError("source_registries must match normalization sources")

        known_budgets = set(self.budget_profile_ids)
        known_fusion = set(self.confidence_fusion_profile_ids)
        known_provenance = set(self.provenance_profile_ids)
        known_traces = set(self.trace_profile_ids)
        for profile in self.normalization_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("normalization sources must match registry sources")
            if profile.source_budget_profile_id not in known_budgets:
                raise ValueError("normalization budgets must be known metadata")
            if profile.source_confidence_fusion_profile_id not in known_fusion:
                raise ValueError("normalization fusion profiles must be known metadata")
            if profile.source_provenance_profile_id not in known_provenance:
                raise ValueError("normalization provenance must be known metadata")
            if profile.source_trace_profile_id not in known_traces:
                raise ValueError("normalization traces must be known metadata")
            if profile.result_normalization_implemented:
                raise ValueError("result normalization must not transform results")
        return self


def result_normalization_registry() -> ResultNormalizationRegistry:
    """Return passive V4.3 result normalization metadata."""

    return RESULT_NORMALIZATION_REGISTRY


def result_normalization_profile_by_id(
    normalization_profile_id: str,
    registry: ResultNormalizationRegistry | None = None,
) -> ResultNormalizationProfile | None:
    """Return one normalization profile without transforming results."""

    source_registry = registry or RESULT_NORMALIZATION_REGISTRY
    for profile in source_registry.normalization_profiles:
        if profile.normalization_profile_id == normalization_profile_id:
            return profile
    return None


class ReturnToWorkflowHandoffProfile(BaseModel):
    """Passive V4.3 return-to-workflow handoff profile metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    return_handoff_profile_id: str = Field(min_length=1, max_length=160)
    topic_id: ReturnToWorkflowHandoffTopic
    source_normalization_profile_id: str = Field(min_length=1, max_length=170)
    source_gate_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_workflow_handoff_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    target_backbone_node_ids: tuple[str, ...] = Field(min_length=1, max_length=6)
    target_workflow_surface: ReturnToWorkflowSurface
    handoff_payload_surfaces: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    handoff_dimensions: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(min_length=1, max_length=900)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_RETURN_TO_WORKFLOW_HANDOFF_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    return_handoff_implemented: Literal[False] = False
    runtime_handoff_implemented: Literal[False] = False
    workflow_graph_change_implemented: Literal[False] = False
    prompt_alteration_implemented: Literal[False] = False
    agent_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["return_to_workflow_handoff_profile.v1"] = (
        RETURN_TO_WORKFLOW_HANDOFF_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class ReturnToWorkflowHandoffRegistry(BaseModel):
    """Stable passive registry for V4.3 return-to-workflow handoff metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["return_to_workflow_handoff_registry"] = (
        "return_to_workflow_handoff_registry"
    )
    serialization_version: Literal["return_to_workflow_handoff_registry.v1"] = (
        RETURN_TO_WORKFLOW_HANDOFF_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=RETURN_TO_WORKFLOW_HANDOFF_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    handoff_profiles: tuple[ReturnToWorkflowHandoffProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    return_handoff_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    topic_ids: tuple[ReturnToWorkflowHandoffTopic, ...] = Field(
        min_length=4,
        max_length=4,
    )
    target_workflow_surfaces: tuple[ReturnToWorkflowSurface, ...] = Field(
        min_length=4,
        max_length=4,
    )
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    normalization_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    gate_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    workflow_handoff_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    backbone_node_ids: tuple[str, ...] = Field(min_length=18, max_length=18)
    profile_count: int = Field(ge=4, le=4)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_RETURN_TO_WORKFLOW_HANDOFF_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    return_handoff_implemented: Literal[False] = False
    runtime_handoff_implemented: Literal[False] = False
    workflow_graph_change_implemented: Literal[False] = False
    prompt_alteration_implemented: Literal[False] = False
    agent_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_return_handoff_metadata(self) -> Self:
        derived_profile_ids = tuple(
            profile.return_handoff_profile_id for profile in self.handoff_profiles
        )
        derived_topic_ids = tuple(profile.topic_id for profile in self.handoff_profiles)
        derived_surfaces = tuple(
            profile.target_workflow_surface for profile in self.handoff_profiles
        )
        if self.return_handoff_profile_ids != derived_profile_ids:
            raise ValueError("return_handoff_profile_ids must match profiles")
        if self.topic_ids != derived_topic_ids:
            raise ValueError("topic_ids must match return handoff profiles")
        if self.topic_ids != _RETURN_TO_WORKFLOW_HANDOFF_TOPICS:
            raise ValueError("topic_ids must preserve return handoff topic order")
        if self.target_workflow_surfaces != derived_surfaces:
            raise ValueError("target_workflow_surfaces must match profiles")
        if self.profile_count != len(self.handoff_profiles):
            raise ValueError("profile_count must match return handoff profiles")

        profile_sources = {
            source_registry
            for profile in self.handoff_profiles
            for source_registry in profile.source_registries
        }
        if set(self.source_registries) != profile_sources:
            raise ValueError("source_registries must match return handoff sources")

        known_normalization = set(self.normalization_profile_ids)
        known_gates = set(self.gate_ids)
        known_handoffs = set(self.workflow_handoff_ids)
        known_nodes = set(self.backbone_node_ids)
        for profile in self.handoff_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("return handoff sources must match registry sources")
            if profile.source_normalization_profile_id not in known_normalization:
                raise ValueError("return handoff normalization must be known metadata")
            if not set(profile.source_gate_ids).issubset(known_gates):
                raise ValueError("return handoff gates must be known metadata")
            if not set(profile.source_workflow_handoff_ids).issubset(known_handoffs):
                raise ValueError("return workflow handoffs must be known metadata")
            if not set(profile.target_backbone_node_ids).issubset(known_nodes):
                raise ValueError("return handoff backbone nodes must be known metadata")
            if profile.return_handoff_implemented:
                raise ValueError("return-to-workflow handoff must not execute")
        return self


def return_to_workflow_handoff_registry() -> ReturnToWorkflowHandoffRegistry:
    """Return passive V4.3 return-to-workflow handoff metadata."""

    return RETURN_TO_WORKFLOW_HANDOFF_REGISTRY


def return_to_workflow_handoff_profile_by_id(
    return_handoff_profile_id: str,
    registry: ReturnToWorkflowHandoffRegistry | None = None,
) -> ReturnToWorkflowHandoffProfile | None:
    """Return one return handoff profile without executing a runtime handoff."""

    source_registry = registry or RETURN_TO_WORKFLOW_HANDOFF_REGISTRY
    for profile in source_registry.handoff_profiles:
        if profile.return_handoff_profile_id == return_handoff_profile_id:
            return profile
    return None


class HitlEscalationGateProfile(BaseModel):
    """Passive V4.3 HITL escalation gate profile metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    hitl_gate_profile_id: str = Field(min_length=1, max_length=160)
    topic_id: HitlEscalationGateTopic
    source_return_handoff_profile_id: str = Field(min_length=1, max_length=180)
    source_gate_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_escalation_signal_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_reflection_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    hitl_posture: HitlEscalationPosture
    human_review_inputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    gate_dimensions: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(min_length=1, max_length=900)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_HITL_ESCALATION_GATE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    hitl_triggering_implemented: Literal[False] = False
    human_review_request_implemented: Literal[False] = False
    gate_evaluation_implemented: Literal[False] = False
    escalation_approval_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["hitl_escalation_gate_profile.v1"] = (
        HITL_ESCALATION_GATE_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class HitlEscalationGateRegistry(BaseModel):
    """Stable passive registry for V4.3 HITL escalation gate metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["hitl_escalation_gate_registry"] = "hitl_escalation_gate_registry"
    serialization_version: Literal["hitl_escalation_gate_registry.v1"] = (
        HITL_ESCALATION_GATE_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=HITL_ESCALATION_GATE_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    hitl_gate_profiles: tuple[HitlEscalationGateProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    hitl_gate_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    topic_ids: tuple[HitlEscalationGateTopic, ...] = Field(
        min_length=4,
        max_length=4,
    )
    hitl_postures: tuple[HitlEscalationPosture, ...] = Field(
        min_length=4,
        max_length=4,
    )
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    return_handoff_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    gate_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    escalation_signal_ids: tuple[str, ...] = Field(min_length=7, max_length=7)
    reflection_profile_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    profile_count: int = Field(ge=4, le=4)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_HITL_ESCALATION_GATE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    hitl_triggering_implemented: Literal[False] = False
    human_review_request_implemented: Literal[False] = False
    gate_evaluation_implemented: Literal[False] = False
    escalation_approval_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_hitl_gate_metadata(self) -> Self:
        derived_profile_ids = tuple(
            profile.hitl_gate_profile_id for profile in self.hitl_gate_profiles
        )
        derived_topic_ids = tuple(
            profile.topic_id for profile in self.hitl_gate_profiles
        )
        derived_postures = tuple(
            profile.hitl_posture for profile in self.hitl_gate_profiles
        )
        if self.hitl_gate_profile_ids != derived_profile_ids:
            raise ValueError("hitl_gate_profile_ids must match profiles")
        if self.topic_ids != derived_topic_ids:
            raise ValueError("topic_ids must match HITL gate profiles")
        if self.topic_ids != _HITL_ESCALATION_GATE_TOPICS:
            raise ValueError("topic_ids must preserve HITL gate topic order")
        if self.hitl_postures != derived_postures:
            raise ValueError("hitl_postures must match profiles")
        if self.hitl_postures != _HITL_ESCALATION_POSTURES:
            raise ValueError("hitl_postures must preserve HITL posture order")
        if self.profile_count != len(self.hitl_gate_profiles):
            raise ValueError("profile_count must match HITL gate profiles")

        profile_sources = {
            source_registry
            for profile in self.hitl_gate_profiles
            for source_registry in profile.source_registries
        }
        if set(self.source_registries) != profile_sources:
            raise ValueError("source_registries must match HITL gate sources")

        known_returns = set(self.return_handoff_profile_ids)
        known_gates = set(self.gate_ids)
        known_signals = set(self.escalation_signal_ids)
        known_reflections = set(self.reflection_profile_ids)
        for profile in self.hitl_gate_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("HITL gate sources must match registry sources")
            if profile.source_return_handoff_profile_id not in known_returns:
                raise ValueError("HITL return handoffs must be known metadata")
            if not set(profile.source_gate_ids).issubset(known_gates):
                raise ValueError("HITL gates must be known metadata")
            if not set(profile.source_escalation_signal_ids).issubset(known_signals):
                raise ValueError("HITL signals must be known metadata")
            if "hitl_escalation_signal" not in profile.source_escalation_signal_ids:
                raise ValueError("HITL profiles must reference hitl_escalation_signal")
            if not set(profile.source_reflection_profile_ids).issubset(
                known_reflections
            ):
                raise ValueError("HITL reflections must be known metadata")
            if profile.hitl_triggering_implemented:
                raise ValueError("HITL escalation gate must not trigger review")
        return self


def hitl_escalation_gate_registry() -> HitlEscalationGateRegistry:
    """Return passive V4.3 HITL escalation gate metadata."""

    return HITL_ESCALATION_GATE_REGISTRY


def hitl_escalation_gate_profile_by_id(
    hitl_gate_profile_id: str,
    registry: HitlEscalationGateRegistry | None = None,
) -> HitlEscalationGateProfile | None:
    """Return one HITL gate profile without triggering human review."""

    source_registry = registry or HITL_ESCALATION_GATE_REGISTRY
    for profile in source_registry.hitl_gate_profiles:
        if profile.hitl_gate_profile_id == hitl_gate_profile_id:
            return profile
    return None


class ConfidenceThresholdRoutingProfile(BaseModel):
    """Passive V4.3 confidence threshold routing profile metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    threshold_profile_id: str = Field(min_length=1, max_length=170)
    topic_id: ConfidenceThresholdRoutingTopic
    source_hitl_gate_profile_id: str = Field(min_length=1, max_length=180)
    source_confidence_fusion_profile_id: str = Field(min_length=1, max_length=180)
    source_escalation_signal_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    confidence_band: ConfidenceThresholdBand
    advisory_threshold_range: tuple[float, float]
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    routing_dimensions: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(min_length=1, max_length=900)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_CONFIDENCE_THRESHOLD_ROUTING_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    threshold_evaluation_implemented: Literal[False] = False
    confidence_based_routing_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["confidence_threshold_routing_profile.v1"] = (
        CONFIDENCE_THRESHOLD_ROUTING_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class ConfidenceThresholdRoutingRegistry(BaseModel):
    """Stable passive registry for V4.3 confidence threshold routing metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["confidence_threshold_routing_registry"] = (
        "confidence_threshold_routing_registry"
    )
    serialization_version: Literal["confidence_threshold_routing_registry.v1"] = (
        CONFIDENCE_THRESHOLD_ROUTING_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CONFIDENCE_THRESHOLD_ROUTING_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    threshold_profiles: tuple[ConfidenceThresholdRoutingProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    threshold_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    topic_ids: tuple[ConfidenceThresholdRoutingTopic, ...] = Field(
        min_length=4,
        max_length=4,
    )
    confidence_bands: tuple[ConfidenceThresholdBand, ...] = Field(
        min_length=4,
        max_length=4,
    )
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    hitl_gate_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    confidence_fusion_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    escalation_signal_ids: tuple[str, ...] = Field(min_length=7, max_length=7)
    profile_count: int = Field(ge=4, le=4)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_CONFIDENCE_THRESHOLD_ROUTING_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    threshold_evaluation_implemented: Literal[False] = False
    confidence_based_routing_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_confidence_threshold_metadata(self) -> Self:
        derived_profile_ids = tuple(
            profile.threshold_profile_id for profile in self.threshold_profiles
        )
        derived_topic_ids = tuple(
            profile.topic_id for profile in self.threshold_profiles
        )
        derived_bands = tuple(
            profile.confidence_band for profile in self.threshold_profiles
        )
        if self.threshold_profile_ids != derived_profile_ids:
            raise ValueError("threshold_profile_ids must match profiles")
        if self.topic_ids != derived_topic_ids:
            raise ValueError("topic_ids must match confidence threshold profiles")
        if self.topic_ids != _CONFIDENCE_THRESHOLD_ROUTING_TOPICS:
            raise ValueError("topic_ids must preserve confidence routing topic order")
        if self.confidence_bands != derived_bands:
            raise ValueError("confidence_bands must match profiles")
        if self.confidence_bands != _CONFIDENCE_THRESHOLD_BANDS:
            raise ValueError("confidence_bands must preserve threshold band order")
        if self.profile_count != len(self.threshold_profiles):
            raise ValueError("profile_count must match confidence threshold profiles")

        profile_sources = {
            source_registry
            for profile in self.threshold_profiles
            for source_registry in profile.source_registries
        }
        if set(self.source_registries) != profile_sources:
            raise ValueError("source_registries must match confidence threshold sources")

        known_hitl = set(self.hitl_gate_profile_ids)
        known_fusion = set(self.confidence_fusion_profile_ids)
        known_signals = set(self.escalation_signal_ids)
        for profile in self.threshold_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("confidence threshold sources must match registry")
            if profile.source_hitl_gate_profile_id not in known_hitl:
                raise ValueError("confidence threshold HITL gates must be known")
            if profile.source_confidence_fusion_profile_id not in known_fusion:
                raise ValueError("confidence threshold fusion profiles must be known")
            if not set(profile.source_escalation_signal_ids).issubset(known_signals):
                raise ValueError("confidence threshold signals must be known")
            low, high = profile.advisory_threshold_range
            if not 0 <= low <= high <= 1:
                raise ValueError("confidence threshold range must be within 0..1")
            if profile.threshold_evaluation_implemented:
                raise ValueError("confidence threshold routing must not evaluate")
        return self


def confidence_threshold_routing_registry() -> ConfidenceThresholdRoutingRegistry:
    """Return passive V4.3 confidence threshold routing metadata."""

    return CONFIDENCE_THRESHOLD_ROUTING_REGISTRY


def confidence_threshold_routing_profile_by_id(
    threshold_profile_id: str,
    registry: ConfidenceThresholdRoutingRegistry | None = None,
) -> ConfidenceThresholdRoutingProfile | None:
    """Return one confidence threshold profile without routing."""

    source_registry = registry or CONFIDENCE_THRESHOLD_ROUTING_REGISTRY
    for profile in source_registry.threshold_profiles:
        if profile.threshold_profile_id == threshold_profile_id:
            return profile
    return None


class CostThresholdRoutingProfile(BaseModel):
    """Passive V4.3 cost threshold routing profile metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    cost_threshold_profile_id: str = Field(min_length=1, max_length=170)
    topic_id: CostThresholdRoutingTopic
    source_budget_profile_id: str = Field(min_length=1, max_length=180)
    source_confidence_threshold_profile_id: str = Field(
        min_length=1,
        max_length=180,
    )
    source_escalation_signal_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    cost_band: CostThresholdBand
    advisory_cost_range: tuple[int, int]
    cost_pressure_signal: str = Field(min_length=1, max_length=120)
    source_registries: tuple[str, ...] = Field(min_length=7, max_length=7)
    routing_dimensions: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(min_length=1, max_length=900)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_COST_THRESHOLD_ROUTING_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    threshold_evaluation_implemented: Literal[False] = False
    cost_based_routing_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["cost_threshold_routing_profile.v1"] = (
        COST_THRESHOLD_ROUTING_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class CostThresholdRoutingRegistry(BaseModel):
    """Stable passive registry for V4.3 cost threshold routing metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["cost_threshold_routing_registry"] = (
        "cost_threshold_routing_registry"
    )
    serialization_version: Literal["cost_threshold_routing_registry.v1"] = (
        COST_THRESHOLD_ROUTING_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=COST_THRESHOLD_ROUTING_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    cost_threshold_profiles: tuple[CostThresholdRoutingProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    cost_threshold_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    topic_ids: tuple[CostThresholdRoutingTopic, ...] = Field(
        min_length=4,
        max_length=4,
    )
    cost_bands: tuple[CostThresholdBand, ...] = Field(min_length=4, max_length=4)
    source_registries: tuple[str, ...] = Field(min_length=7, max_length=7)
    budget_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    confidence_threshold_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    escalation_signal_ids: tuple[str, ...] = Field(min_length=7, max_length=7)
    profile_count: int = Field(ge=4, le=4)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_COST_THRESHOLD_ROUTING_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    threshold_evaluation_implemented: Literal[False] = False
    cost_based_routing_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_cost_threshold_metadata(self) -> Self:
        derived_profile_ids = tuple(
            profile.cost_threshold_profile_id
            for profile in self.cost_threshold_profiles
        )
        derived_topic_ids = tuple(
            profile.topic_id for profile in self.cost_threshold_profiles
        )
        derived_bands = tuple(
            profile.cost_band for profile in self.cost_threshold_profiles
        )
        if self.cost_threshold_profile_ids != derived_profile_ids:
            raise ValueError("cost_threshold_profile_ids must match profiles")
        if self.topic_ids != derived_topic_ids:
            raise ValueError("topic_ids must match cost threshold profiles")
        if self.topic_ids != _COST_THRESHOLD_ROUTING_TOPICS:
            raise ValueError("topic_ids must preserve cost routing topic order")
        if self.cost_bands != derived_bands:
            raise ValueError("cost_bands must match profiles")
        if self.cost_bands != _COST_THRESHOLD_BANDS:
            raise ValueError("cost_bands must preserve cost band order")
        if self.profile_count != len(self.cost_threshold_profiles):
            raise ValueError("profile_count must match cost threshold profiles")

        profile_sources = {
            source_registry
            for profile in self.cost_threshold_profiles
            for source_registry in profile.source_registries
        }
        if set(self.source_registries) != profile_sources:
            raise ValueError("source_registries must match cost threshold sources")

        known_budgets = set(self.budget_profile_ids)
        known_thresholds = set(self.confidence_threshold_profile_ids)
        known_signals = set(self.escalation_signal_ids)
        for profile in self.cost_threshold_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("cost threshold sources must match registry")
            if profile.source_budget_profile_id not in known_budgets:
                raise ValueError("cost threshold budgets must be known metadata")
            if profile.source_confidence_threshold_profile_id not in known_thresholds:
                raise ValueError("cost threshold confidence profiles must be known")
            if not set(profile.source_escalation_signal_ids).issubset(known_signals):
                raise ValueError("cost threshold signals must be known")
            if "cost_escalation_signal" not in profile.source_escalation_signal_ids:
                raise ValueError("cost profiles must reference cost_escalation_signal")
            low, high = profile.advisory_cost_range
            if not 0 <= low <= high:
                raise ValueError("cost threshold range must be non-negative")
            if profile.threshold_evaluation_implemented:
                raise ValueError("cost threshold routing must not evaluate")
            if profile.cost_based_routing_implemented:
                raise ValueError("cost threshold routing must not route")
            if profile.budget_enforcement_implemented:
                raise ValueError("cost threshold routing must not enforce budgets")
        return self


def cost_threshold_routing_registry() -> CostThresholdRoutingRegistry:
    """Return passive V4.3 cost threshold routing metadata."""

    return COST_THRESHOLD_ROUTING_REGISTRY


def cost_threshold_routing_profile_by_id(
    cost_threshold_profile_id: str,
    registry: CostThresholdRoutingRegistry | None = None,
) -> CostThresholdRoutingProfile | None:
    """Return one cost threshold profile without routing by cost."""

    source_registry = registry or COST_THRESHOLD_ROUTING_REGISTRY
    for profile in source_registry.cost_threshold_profiles:
        if profile.cost_threshold_profile_id == cost_threshold_profile_id:
            return profile
    return None


class LatencyThresholdRoutingProfile(BaseModel):
    """Passive V4.3 latency threshold routing profile metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    latency_threshold_profile_id: str = Field(min_length=1, max_length=170)
    topic_id: LatencyThresholdRoutingTopic
    source_cost_threshold_profile_id: str = Field(min_length=1, max_length=180)
    source_escalation_signal_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    latency_band: LatencyThresholdBand
    advisory_latency_range_ms: tuple[int, int]
    latency_metadata_sources: tuple[str, ...] = Field(min_length=4, max_length=4)
    source_registries: tuple[str, ...] = Field(min_length=7, max_length=7)
    routing_dimensions: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(min_length=1, max_length=900)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_LATENCY_THRESHOLD_ROUTING_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    threshold_evaluation_implemented: Literal[False] = False
    latency_based_routing_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["latency_threshold_routing_profile.v1"] = (
        LATENCY_THRESHOLD_ROUTING_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class LatencyThresholdRoutingRegistry(BaseModel):
    """Stable passive registry for V4.3 latency threshold routing metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["latency_threshold_routing_registry"] = (
        "latency_threshold_routing_registry"
    )
    serialization_version: Literal["latency_threshold_routing_registry.v1"] = (
        LATENCY_THRESHOLD_ROUTING_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=LATENCY_THRESHOLD_ROUTING_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    latency_threshold_profiles: tuple[LatencyThresholdRoutingProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    latency_threshold_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    topic_ids: tuple[LatencyThresholdRoutingTopic, ...] = Field(
        min_length=4,
        max_length=4,
    )
    latency_bands: tuple[LatencyThresholdBand, ...] = Field(
        min_length=4,
        max_length=4,
    )
    source_registries: tuple[str, ...] = Field(min_length=7, max_length=7)
    cost_threshold_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    escalation_signal_ids: tuple[str, ...] = Field(min_length=7, max_length=7)
    latency_metadata_sources: tuple[str, ...] = Field(min_length=4, max_length=4)
    profile_count: int = Field(ge=4, le=4)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_LATENCY_THRESHOLD_ROUTING_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    threshold_evaluation_implemented: Literal[False] = False
    latency_based_routing_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_latency_threshold_metadata(self) -> Self:
        derived_profile_ids = tuple(
            profile.latency_threshold_profile_id
            for profile in self.latency_threshold_profiles
        )
        derived_topic_ids = tuple(
            profile.topic_id for profile in self.latency_threshold_profiles
        )
        derived_bands = tuple(
            profile.latency_band for profile in self.latency_threshold_profiles
        )
        if self.latency_threshold_profile_ids != derived_profile_ids:
            raise ValueError("latency_threshold_profile_ids must match profiles")
        if self.topic_ids != derived_topic_ids:
            raise ValueError("topic_ids must match latency threshold profiles")
        if self.topic_ids != _LATENCY_THRESHOLD_ROUTING_TOPICS:
            raise ValueError("topic_ids must preserve latency routing topic order")
        if self.latency_bands != derived_bands:
            raise ValueError("latency_bands must match profiles")
        if self.latency_bands != _LATENCY_THRESHOLD_BANDS:
            raise ValueError("latency_bands must preserve latency band order")
        if self.profile_count != len(self.latency_threshold_profiles):
            raise ValueError("profile_count must match latency threshold profiles")

        profile_sources = {
            source_registry
            for profile in self.latency_threshold_profiles
            for source_registry in profile.source_registries
        }
        if set(self.source_registries) != profile_sources:
            raise ValueError("source_registries must match latency threshold sources")

        known_cost_thresholds = set(self.cost_threshold_profile_ids)
        known_signals = set(self.escalation_signal_ids)
        for profile in self.latency_threshold_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("latency threshold sources must match registry")
            if profile.source_cost_threshold_profile_id not in known_cost_thresholds:
                raise ValueError("latency threshold cost profiles must be known")
            if not set(profile.source_escalation_signal_ids).issubset(known_signals):
                raise ValueError("latency threshold signals must be known")
            if "latency_escalation_signal" not in profile.source_escalation_signal_ids:
                raise ValueError("latency profiles must reference latency signal")
            if profile.latency_metadata_sources != self.latency_metadata_sources:
                raise ValueError("latency metadata sources must match registry")
            latency_low, latency_high = profile.advisory_latency_range_ms
            if not 0 <= latency_low <= latency_high:
                raise ValueError("latency threshold range must be non-negative")
            if profile.threshold_evaluation_implemented:
                raise ValueError("latency threshold routing must not evaluate")
            if profile.latency_based_routing_implemented:
                raise ValueError("latency threshold routing must not route")
            if profile.runtime_selection_implemented:
                raise ValueError("latency threshold routing must not select runtime")
        return self


def latency_threshold_routing_registry() -> LatencyThresholdRoutingRegistry:
    """Return passive V4.3 latency threshold routing metadata."""

    return LATENCY_THRESHOLD_ROUTING_REGISTRY


def latency_threshold_routing_profile_by_id(
    latency_threshold_profile_id: str,
    registry: LatencyThresholdRoutingRegistry | None = None,
) -> LatencyThresholdRoutingProfile | None:
    """Return one latency threshold profile without routing by latency."""

    source_registry = registry or LATENCY_THRESHOLD_ROUTING_REGISTRY
    for profile in source_registry.latency_threshold_profiles:
        if profile.latency_threshold_profile_id == latency_threshold_profile_id:
            return profile
    return None


class AmbiguityEscalationProfile(BaseModel):
    """Passive V4.3 ambiguity escalation profile metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    ambiguity_profile_id: str = Field(min_length=1, max_length=170)
    topic_id: AmbiguityEscalationTopic
    source_latency_threshold_profile_id: str = Field(min_length=1, max_length=180)
    source_condition_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_policy_rule_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_escalation_signal_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    ambiguity_level: AmbiguityEscalationLevel
    ambiguity_evidence_surfaces: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_registries: tuple[str, ...] = Field(min_length=7, max_length=7)
    escalation_dimensions: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(min_length=1, max_length=900)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_AMBIGUITY_ESCALATION_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    ambiguity_evaluation_implemented: Literal[False] = False
    escalation_execution_implemented: Literal[False] = False
    clarification_request_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["ambiguity_escalation_profile.v1"] = (
        AMBIGUITY_ESCALATION_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AmbiguityEscalationRegistry(BaseModel):
    """Stable passive registry for V4.3 ambiguity escalation metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["ambiguity_escalation_registry"] = "ambiguity_escalation_registry"
    serialization_version: Literal["ambiguity_escalation_registry.v1"] = (
        AMBIGUITY_ESCALATION_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=AMBIGUITY_ESCALATION_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    ambiguity_profiles: tuple[AmbiguityEscalationProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    ambiguity_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    topic_ids: tuple[AmbiguityEscalationTopic, ...] = Field(
        min_length=4,
        max_length=4,
    )
    ambiguity_levels: tuple[AmbiguityEscalationLevel, ...] = Field(
        min_length=4,
        max_length=4,
    )
    source_registries: tuple[str, ...] = Field(min_length=7, max_length=7)
    latency_threshold_profile_ids: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    condition_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    policy_rule_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    escalation_signal_ids: tuple[str, ...] = Field(min_length=7, max_length=7)
    ambiguity_evidence_surfaces: tuple[str, ...] = Field(min_length=4, max_length=4)
    profile_count: int = Field(ge=4, le=4)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_AMBIGUITY_ESCALATION_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    ambiguity_evaluation_implemented: Literal[False] = False
    escalation_execution_implemented: Literal[False] = False
    clarification_request_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_ambiguity_escalation_metadata(self) -> Self:
        derived_profile_ids = tuple(
            profile.ambiguity_profile_id for profile in self.ambiguity_profiles
        )
        derived_topic_ids = tuple(
            profile.topic_id for profile in self.ambiguity_profiles
        )
        derived_levels = tuple(
            profile.ambiguity_level for profile in self.ambiguity_profiles
        )
        if self.ambiguity_profile_ids != derived_profile_ids:
            raise ValueError("ambiguity_profile_ids must match profiles")
        if self.topic_ids != derived_topic_ids:
            raise ValueError("topic_ids must match ambiguity profiles")
        if self.topic_ids != _AMBIGUITY_ESCALATION_TOPICS:
            raise ValueError("topic_ids must preserve ambiguity topic order")
        if self.ambiguity_levels != derived_levels:
            raise ValueError("ambiguity_levels must match profiles")
        if self.ambiguity_levels != _AMBIGUITY_ESCALATION_LEVELS:
            raise ValueError("ambiguity_levels must preserve ambiguity order")
        if self.profile_count != len(self.ambiguity_profiles):
            raise ValueError("profile_count must match ambiguity profiles")

        profile_sources = {
            source_registry
            for profile in self.ambiguity_profiles
            for source_registry in profile.source_registries
        }
        if set(self.source_registries) != profile_sources:
            raise ValueError("source_registries must match ambiguity sources")

        known_latency = set(self.latency_threshold_profile_ids)
        known_conditions = set(self.condition_ids)
        known_policies = set(self.policy_rule_ids)
        known_signals = set(self.escalation_signal_ids)
        known_evidence = set(self.ambiguity_evidence_surfaces)
        for profile in self.ambiguity_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("ambiguity sources must match registry")
            if profile.source_latency_threshold_profile_id not in known_latency:
                raise ValueError("ambiguity latency profiles must be known")
            if not set(profile.source_condition_ids).issubset(known_conditions):
                raise ValueError("ambiguity conditions must be known")
            if not set(profile.source_policy_rule_ids).issubset(known_policies):
                raise ValueError("ambiguity policies must be known")
            if not set(profile.source_escalation_signal_ids).issubset(known_signals):
                raise ValueError("ambiguity signals must be known")
            if "ambiguity_escalation_signal" not in profile.source_escalation_signal_ids:
                raise ValueError("ambiguity profiles must reference ambiguity signal")
            if not set(profile.ambiguity_evidence_surfaces).issubset(known_evidence):
                raise ValueError("ambiguity evidence surfaces must be known")
            if profile.ambiguity_evaluation_implemented:
                raise ValueError("ambiguity escalation must not evaluate ambiguity")
            if profile.escalation_execution_implemented:
                raise ValueError("ambiguity escalation must not execute escalation")
            if profile.clarification_request_implemented:
                raise ValueError("ambiguity escalation must not request clarification")
        return self


def ambiguity_escalation_registry() -> AmbiguityEscalationRegistry:
    """Return passive V4.3 ambiguity escalation metadata."""

    return AMBIGUITY_ESCALATION_REGISTRY


def ambiguity_escalation_profile_by_id(
    ambiguity_profile_id: str,
    registry: AmbiguityEscalationRegistry | None = None,
) -> AmbiguityEscalationProfile | None:
    """Return one ambiguity profile without triggering escalation."""

    source_registry = registry or AMBIGUITY_ESCALATION_REGISTRY
    for profile in source_registry.ambiguity_profiles:
        if profile.ambiguity_profile_id == ambiguity_profile_id:
            return profile
    return None


class RiskEscalationProfile(BaseModel):
    """Passive V4.3 risk escalation profile metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    risk_profile_id: str = Field(min_length=1, max_length=170)
    topic_id: RiskEscalationTopic
    source_ambiguity_profile_id: str = Field(min_length=1, max_length=180)
    source_condition_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_policy_rule_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_creative_policy_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_escalation_signal_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    risk_level: RiskEscalationLevel
    risk_evidence_surfaces: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_registries: tuple[str, ...] = Field(min_length=7, max_length=7)
    escalation_dimensions: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(min_length=1, max_length=900)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_RISK_ESCALATION_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    risk_evaluation_implemented: Literal[False] = False
    escalation_execution_implemented: Literal[False] = False
    mitigation_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["risk_escalation_profile.v1"] = (
        RISK_ESCALATION_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class RiskEscalationRegistry(BaseModel):
    """Stable passive registry for V4.3 risk escalation metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["risk_escalation_registry"] = "risk_escalation_registry"
    serialization_version: Literal["risk_escalation_registry.v1"] = (
        RISK_ESCALATION_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=RISK_ESCALATION_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    risk_profiles: tuple[RiskEscalationProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    risk_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    topic_ids: tuple[RiskEscalationTopic, ...] = Field(min_length=4, max_length=4)
    risk_levels: tuple[RiskEscalationLevel, ...] = Field(min_length=4, max_length=4)
    source_registries: tuple[str, ...] = Field(min_length=7, max_length=7)
    ambiguity_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    condition_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    policy_rule_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    creative_policy_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    escalation_signal_ids: tuple[str, ...] = Field(min_length=7, max_length=7)
    risk_evidence_surfaces: tuple[str, ...] = Field(min_length=5, max_length=5)
    profile_count: int = Field(ge=4, le=4)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_RISK_ESCALATION_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    risk_evaluation_implemented: Literal[False] = False
    escalation_execution_implemented: Literal[False] = False
    mitigation_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_risk_escalation_metadata(self) -> Self:
        derived_profile_ids = tuple(
            profile.risk_profile_id for profile in self.risk_profiles
        )
        derived_topic_ids = tuple(profile.topic_id for profile in self.risk_profiles)
        derived_levels = tuple(profile.risk_level for profile in self.risk_profiles)
        if self.risk_profile_ids != derived_profile_ids:
            raise ValueError("risk_profile_ids must match profiles")
        if self.topic_ids != derived_topic_ids:
            raise ValueError("topic_ids must match risk profiles")
        if self.topic_ids != _RISK_ESCALATION_TOPICS:
            raise ValueError("topic_ids must preserve risk topic order")
        if self.risk_levels != derived_levels:
            raise ValueError("risk_levels must match profiles")
        if self.risk_levels != _RISK_ESCALATION_LEVELS:
            raise ValueError("risk_levels must preserve risk order")
        if self.profile_count != len(self.risk_profiles):
            raise ValueError("profile_count must match risk profiles")

        profile_sources = {
            source_registry
            for profile in self.risk_profiles
            for source_registry in profile.source_registries
        }
        if set(self.source_registries) != profile_sources:
            raise ValueError("source_registries must match risk sources")

        known_ambiguity = set(self.ambiguity_profile_ids)
        known_conditions = set(self.condition_ids)
        known_policies = set(self.policy_rule_ids)
        known_creative = set(self.creative_policy_ids)
        known_signals = set(self.escalation_signal_ids)
        known_evidence = set(self.risk_evidence_surfaces)
        for profile in self.risk_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("risk sources must match registry")
            if profile.source_ambiguity_profile_id not in known_ambiguity:
                raise ValueError("risk ambiguity profiles must be known")
            if not set(profile.source_condition_ids).issubset(known_conditions):
                raise ValueError("risk conditions must be known")
            if not set(profile.source_policy_rule_ids).issubset(known_policies):
                raise ValueError("risk policies must be known")
            if not set(profile.source_creative_policy_ids).issubset(known_creative):
                raise ValueError("risk creative policies must be known")
            if not set(profile.source_escalation_signal_ids).issubset(known_signals):
                raise ValueError("risk signals must be known")
            if "risk_escalation_signal" not in profile.source_escalation_signal_ids:
                raise ValueError("risk profiles must reference risk signal")
            if not set(profile.risk_evidence_surfaces).issubset(known_evidence):
                raise ValueError("risk evidence surfaces must be known")
            if profile.risk_evaluation_implemented:
                raise ValueError("risk escalation must not evaluate risk")
            if profile.escalation_execution_implemented:
                raise ValueError("risk escalation must not execute escalation")
            if profile.mitigation_execution_implemented:
                raise ValueError("risk escalation must not apply mitigation")
        return self


def risk_escalation_registry() -> RiskEscalationRegistry:
    """Return passive V4.3 risk escalation metadata."""

    return RISK_ESCALATION_REGISTRY


def risk_escalation_profile_by_id(
    risk_profile_id: str,
    registry: RiskEscalationRegistry | None = None,
) -> RiskEscalationProfile | None:
    """Return one risk profile without triggering escalation."""

    source_registry = registry or RISK_ESCALATION_REGISTRY
    for profile in source_registry.risk_profiles:
        if profile.risk_profile_id == risk_profile_id:
            return profile
    return None


class QualityEscalationProfile(BaseModel):
    """Passive V4.3 quality escalation profile metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    quality_profile_id: str = Field(min_length=1, max_length=170)
    topic_id: QualityEscalationTopic
    source_risk_profile_id: str = Field(min_length=1, max_length=180)
    source_confidence_fusion_profile_id: str = Field(min_length=1, max_length=180)
    source_creative_policy_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_escalation_signal_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    quality_level: QualityEscalationLevel
    quality_evidence_surfaces: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_registries: tuple[str, ...] = Field(min_length=7, max_length=7)
    escalation_dimensions: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(min_length=1, max_length=900)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_QUALITY_ESCALATION_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    quality_evaluation_implemented: Literal[False] = False
    escalation_execution_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["quality_escalation_profile.v1"] = (
        QUALITY_ESCALATION_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class QualityEscalationRegistry(BaseModel):
    """Stable passive registry for V4.3 quality escalation metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["quality_escalation_registry"] = "quality_escalation_registry"
    serialization_version: Literal["quality_escalation_registry.v1"] = (
        QUALITY_ESCALATION_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=QUALITY_ESCALATION_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    quality_profiles: tuple[QualityEscalationProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    quality_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    topic_ids: tuple[QualityEscalationTopic, ...] = Field(min_length=4, max_length=4)
    quality_levels: tuple[QualityEscalationLevel, ...] = Field(
        min_length=4,
        max_length=4,
    )
    source_registries: tuple[str, ...] = Field(min_length=7, max_length=7)
    risk_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    confidence_fusion_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    creative_policy_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    escalation_signal_ids: tuple[str, ...] = Field(min_length=7, max_length=7)
    quality_evidence_surfaces: tuple[str, ...] = Field(min_length=5, max_length=5)
    profile_count: int = Field(ge=4, le=4)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_QUALITY_ESCALATION_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    quality_evaluation_implemented: Literal[False] = False
    escalation_execution_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_quality_escalation_metadata(self) -> Self:
        derived_profile_ids = tuple(
            profile.quality_profile_id for profile in self.quality_profiles
        )
        derived_topic_ids = tuple(profile.topic_id for profile in self.quality_profiles)
        derived_levels = tuple(profile.quality_level for profile in self.quality_profiles)
        if self.quality_profile_ids != derived_profile_ids:
            raise ValueError("quality_profile_ids must match profiles")
        if self.topic_ids != derived_topic_ids:
            raise ValueError("topic_ids must match quality profiles")
        if self.topic_ids != _QUALITY_ESCALATION_TOPICS:
            raise ValueError("topic_ids must preserve quality topic order")
        if self.quality_levels != derived_levels:
            raise ValueError("quality_levels must match profiles")
        if self.quality_levels != _QUALITY_ESCALATION_LEVELS:
            raise ValueError("quality_levels must preserve quality order")
        if self.profile_count != len(self.quality_profiles):
            raise ValueError("profile_count must match quality profiles")

        profile_sources = {
            source_registry
            for profile in self.quality_profiles
            for source_registry in profile.source_registries
        }
        if set(self.source_registries) != profile_sources:
            raise ValueError("source_registries must match quality sources")

        known_risk = set(self.risk_profile_ids)
        known_fusion = set(self.confidence_fusion_profile_ids)
        known_creative = set(self.creative_policy_ids)
        known_signals = set(self.escalation_signal_ids)
        known_evidence = set(self.quality_evidence_surfaces)
        for profile in self.quality_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("quality sources must match registry")
            if profile.source_risk_profile_id not in known_risk:
                raise ValueError("quality risk profiles must be known")
            if profile.source_confidence_fusion_profile_id not in known_fusion:
                raise ValueError("quality fusion profiles must be known")
            if not set(profile.source_creative_policy_ids).issubset(known_creative):
                raise ValueError("quality creative policies must be known")
            if not set(profile.source_escalation_signal_ids).issubset(known_signals):
                raise ValueError("quality signals must be known")
            if "quality_escalation_signal" not in profile.source_escalation_signal_ids:
                raise ValueError("quality profiles must reference quality signal")
            if not set(profile.quality_evidence_surfaces).issubset(known_evidence):
                raise ValueError("quality evidence surfaces must be known")
            if profile.quality_evaluation_implemented:
                raise ValueError("quality escalation must not evaluate quality")
            if profile.escalation_execution_implemented:
                raise ValueError("quality escalation must not execute escalation")
            if profile.refinement_triggering_implemented:
                raise ValueError("quality escalation must not trigger refinement")
        return self


def quality_escalation_registry() -> QualityEscalationRegistry:
    """Return passive V4.3 quality escalation metadata."""

    return QUALITY_ESCALATION_REGISTRY


def quality_escalation_profile_by_id(
    quality_profile_id: str,
    registry: QualityEscalationRegistry | None = None,
) -> QualityEscalationProfile | None:
    """Return one quality profile without triggering escalation."""

    source_registry = registry or QUALITY_ESCALATION_REGISTRY
    for profile in source_registry.quality_profiles:
        if profile.quality_profile_id == quality_profile_id:
            return profile
    return None


class AdaptiveMultiAgentEscalationProfile(BaseModel):
    """Passive V4.3 adaptive multi-agent escalation profile metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    adaptive_profile_id: str = Field(min_length=1, max_length=190)
    topic_id: AdaptiveMultiAgentEscalationTopic
    source_capability_id: Literal["adaptive_multi_agent_escalation"] = (
        "adaptive_multi_agent_escalation"
    )
    source_condition_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_specialist_loop_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_quality_profile_id: str = Field(min_length=1, max_length=190)
    source_risk_profile_id: str = Field(min_length=1, max_length=190)
    source_ambiguity_profile_id: str = Field(min_length=1, max_length=190)
    source_hitl_gate_profile_id: str = Field(min_length=1, max_length=190)
    source_escalation_signal_ids: tuple[str, ...] = Field(min_length=1, max_length=7)
    adaptive_posture: AdaptiveMultiAgentEscalationPosture
    adaptive_evidence_surfaces: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_registries: tuple[str, ...] = Field(min_length=9, max_length=9)
    escalation_dimensions: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(min_length=1, max_length=1000)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_ADAPTIVE_MULTI_AGENT_ESCALATION_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    adaptation_evaluation_implemented: Literal[False] = False
    escalation_execution_implemented: Literal[False] = False
    multi_agent_orchestration_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["adaptive_multi_agent_escalation_profile.v1"] = (
        ADAPTIVE_MULTI_AGENT_ESCALATION_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AdaptiveMultiAgentEscalationRegistry(BaseModel):
    """Stable passive registry for V4.3 adaptive escalation metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["adaptive_multi_agent_escalation_registry"] = (
        "adaptive_multi_agent_escalation_registry"
    )
    serialization_version: Literal[
        "adaptive_multi_agent_escalation_registry.v1"
    ] = ADAPTIVE_MULTI_AGENT_ESCALATION_REGISTRY_SERIALIZATION_VERSION
    authority_boundary: str = Field(
        default=ADAPTIVE_MULTI_AGENT_ESCALATION_AUTHORITY_BOUNDARY,
        max_length=1100,
    )
    adaptive_profiles: tuple[AdaptiveMultiAgentEscalationProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    adaptive_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    topic_ids: tuple[AdaptiveMultiAgentEscalationTopic, ...] = Field(
        min_length=4,
        max_length=4,
    )
    adaptive_postures: tuple[AdaptiveMultiAgentEscalationPosture, ...] = Field(
        min_length=4,
        max_length=4,
    )
    source_registries: tuple[str, ...] = Field(min_length=9, max_length=9)
    capability_ids: tuple[str, ...] = Field(min_length=1, max_length=1)
    condition_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    specialist_loop_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    quality_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    risk_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    ambiguity_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    hitl_gate_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    escalation_signal_ids: tuple[str, ...] = Field(min_length=7, max_length=7)
    adaptive_evidence_surfaces: tuple[str, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=4, le=4)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_ADAPTIVE_MULTI_AGENT_ESCALATION_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    adaptation_evaluation_implemented: Literal[False] = False
    escalation_execution_implemented: Literal[False] = False
    multi_agent_orchestration_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_adaptive_multi_agent_escalation_metadata(self) -> Self:
        derived_profile_ids = tuple(
            profile.adaptive_profile_id for profile in self.adaptive_profiles
        )
        derived_topic_ids = tuple(
            profile.topic_id for profile in self.adaptive_profiles
        )
        derived_postures = tuple(
            profile.adaptive_posture for profile in self.adaptive_profiles
        )
        if self.adaptive_profile_ids != derived_profile_ids:
            raise ValueError("adaptive_profile_ids must match profiles")
        if self.topic_ids != derived_topic_ids:
            raise ValueError("topic_ids must match adaptive profiles")
        if self.topic_ids != _ADAPTIVE_MULTI_AGENT_ESCALATION_TOPICS:
            raise ValueError("topic_ids must preserve adaptive topic order")
        if self.adaptive_postures != derived_postures:
            raise ValueError("adaptive_postures must match profiles")
        if (
            self.adaptive_postures
            != _ADAPTIVE_MULTI_AGENT_ESCALATION_POSTURES
        ):
            raise ValueError("adaptive_postures must preserve adaptive order")
        if self.profile_count != len(self.adaptive_profiles):
            raise ValueError("profile_count must match adaptive profiles")

        profile_sources = {
            source_registry
            for profile in self.adaptive_profiles
            for source_registry in profile.source_registries
        }
        if set(self.source_registries) != profile_sources:
            raise ValueError("source_registries must match adaptive sources")

        known_capabilities = set(self.capability_ids)
        known_conditions = set(self.condition_ids)
        known_loops = set(self.specialist_loop_ids)
        known_quality = set(self.quality_profile_ids)
        known_risk = set(self.risk_profile_ids)
        known_ambiguity = set(self.ambiguity_profile_ids)
        known_hitl = set(self.hitl_gate_profile_ids)
        known_signals = set(self.escalation_signal_ids)
        known_evidence = set(self.adaptive_evidence_surfaces)
        for profile in self.adaptive_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("adaptive sources must match registry")
            if profile.source_capability_id not in known_capabilities:
                raise ValueError("adaptive capability must be known")
            if not set(profile.source_condition_ids).issubset(known_conditions):
                raise ValueError("adaptive conditions must be known")
            if not set(profile.source_specialist_loop_ids).issubset(known_loops):
                raise ValueError("adaptive specialist loops must be known")
            if profile.source_quality_profile_id not in known_quality:
                raise ValueError("adaptive quality profiles must be known")
            if profile.source_risk_profile_id not in known_risk:
                raise ValueError("adaptive risk profiles must be known")
            if profile.source_ambiguity_profile_id not in known_ambiguity:
                raise ValueError("adaptive ambiguity profiles must be known")
            if profile.source_hitl_gate_profile_id not in known_hitl:
                raise ValueError("adaptive HITL profiles must be known")
            if not set(profile.source_escalation_signal_ids).issubset(known_signals):
                raise ValueError("adaptive signals must be known")
            if "quality_escalation_signal" not in profile.source_escalation_signal_ids:
                raise ValueError("adaptive profiles must reference quality signal")
            if not set(profile.adaptive_evidence_surfaces).issubset(known_evidence):
                raise ValueError("adaptive evidence surfaces must be known")
            if profile.adaptation_evaluation_implemented:
                raise ValueError("adaptive escalation must not evaluate adaptation")
            if profile.escalation_execution_implemented:
                raise ValueError("adaptive escalation must not execute escalation")
            if profile.multi_agent_orchestration_implemented:
                raise ValueError("adaptive escalation must not orchestrate agents")
        return self


def adaptive_multi_agent_escalation_registry() -> (
    AdaptiveMultiAgentEscalationRegistry
):
    """Return passive V4.3 adaptive multi-agent escalation metadata."""

    return ADAPTIVE_MULTI_AGENT_ESCALATION_REGISTRY


def adaptive_multi_agent_escalation_profile_by_id(
    adaptive_profile_id: str,
    registry: AdaptiveMultiAgentEscalationRegistry | None = None,
) -> AdaptiveMultiAgentEscalationProfile | None:
    """Return one adaptive escalation profile without invoking agents."""

    source_registry = registry or ADAPTIVE_MULTI_AGENT_ESCALATION_REGISTRY
    for profile in source_registry.adaptive_profiles:
        if profile.adaptive_profile_id == adaptive_profile_id:
            return profile
    return None


class HybridAgenticWorkflowStage(BaseModel):
    """Metadata-only future hybrid workflow readiness stage."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    stage_id: str = Field(min_length=1, max_length=80)
    stage_name: str = Field(min_length=1, max_length=140)
    authority_boundary: str = Field(min_length=1, max_length=900)
    v3_workflow_nodes: tuple[str, ...] = Field(min_length=1, max_length=8)
    future_capability_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    escalation_rule_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_metadata_registries: tuple[str, ...] = Field(min_length=1, max_length=6)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=1, max_length=12)
    serialization_version: Literal["hybrid_workflow_stage.v1"] = (
        HYBRID_WORKFLOW_STAGE_SERIALIZATION_VERSION
    )


class HybridAgenticWorkflowRegistry(BaseModel):
    """Stable metadata registry for future hybrid agentic workflow readiness."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["hybrid_agentic_workflow_registry"] = (
        "hybrid_agentic_workflow_registry"
    )
    serialization_version: Literal["hybrid_workflow_registry.v1"] = (
        HYBRID_WORKFLOW_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=HYBRID_WORKFLOW_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    stages: tuple[HybridAgenticWorkflowStage, ...] = Field(
        min_length=5,
        max_length=5,
    )
    stage_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    stage_count: int = Field(ge=5, le=5)
    source_metadata_registries: tuple[str, ...] = Field(min_length=5, max_length=5)
    metadata_only: Literal[True] = True


def hybrid_agentic_workflow_registry() -> HybridAgenticWorkflowRegistry:
    """Return the static future hybrid workflow readiness registry."""

    return HYBRID_AGENTIC_WORKFLOW_REGISTRY


def hybrid_agentic_workflow_stage_by_id(
    stage_id: str,
) -> HybridAgenticWorkflowStage | None:
    """Return one hybrid workflow readiness stage without changing behavior."""

    for stage in HYBRID_AGENTIC_WORKFLOW_STAGES:
        if stage.stage_id == stage_id:
            return stage
    return None


def _backbone_profile(
    *,
    node_id: str,
    phase: BackboneModePhase,
    preserved_surfaces: tuple[str, ...],
) -> V3BackboneModeNodeProfile:
    return V3BackboneModeNodeProfile(
        node_id=node_id,
        phase=phase,
        preserved_surfaces=preserved_surfaces,
        source_registries=_V3_BACKBONE_MODE_SOURCE_REGISTRIES,
        authority_boundary=(
            "This node remains part of the deterministic V3 workflow backbone; "
            "its metadata does not change workflow order, perform multi-agent "
            "escalation, invoke agents, route providers or models, mutate "
            "prompts, write memory, or modify generated output."
        ),
    )


def _conditional_escalation_condition(
    *,
    condition_id: str,
    condition_name: str,
    category: ConditionalEscalationCategory,
    backbone_phase: BackboneModePhase,
    source_node_ids: tuple[str, ...],
    capability_ids: tuple[str, ...],
    policy_rule_ids: tuple[str, ...],
    escalation_signal_ids: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> ConditionalMultiAgentEscalationCondition:
    return ConditionalMultiAgentEscalationCondition(
        condition_id=condition_id,
        condition_name=condition_name,
        category=category,
        backbone_phase=backbone_phase,
        source_node_ids=source_node_ids,
        source_registries=_CONDITIONAL_ESCALATION_SOURCE_REGISTRIES,
        capability_ids=capability_ids,
        policy_rule_ids=policy_rule_ids,
        escalation_signal_ids=escalation_signal_ids,
        advisory_outputs=advisory_outputs,
        authority_boundary=(
            "This condition is advisory escalation metadata only; it does not "
            "evaluate conditions, invoke agents, route providers or models, "
            "control workflow transitions, trigger retries, execute voting, "
            "write memory, or modify generated output."
        ),
    )


def _specialist_agent_loop(
    *,
    loop_id: str,
    loop_name: str,
    category: SpecialistLoopCategory,
    specialist_agent_ids: tuple[str, ...],
    source_condition_ids: tuple[str, ...],
    source_node_ids: tuple[str, ...],
    loop_inputs: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
    max_advisory_passes: int,
) -> SpecialistAgentLoopProfile:
    return SpecialistAgentLoopProfile(
        loop_id=loop_id,
        loop_name=loop_name,
        category=category,
        specialist_agent_ids=specialist_agent_ids,
        source_condition_ids=source_condition_ids,
        source_node_ids=source_node_ids,
        source_registries=_SPECIALIST_AGENT_LOOP_SOURCE_REGISTRIES,
        loop_inputs=loop_inputs,
        advisory_outputs=advisory_outputs,
        max_advisory_passes=max_advisory_passes,
        authority_boundary=(
            "This specialist loop is advisory metadata only; it does not "
            "execute loops, invoke agents, coordinate multi-agent work, route "
            "providers or models, control workflow transitions, trigger "
            "retries, write memory, or modify generated output."
        ),
    )


def _escalation_gate(
    *,
    gate_id: str,
    gate_name: str,
    gate_kind: EscalationGateKind,
    source_condition_ids: tuple[str, ...],
    source_loop_ids: tuple[str, ...],
    required_passive_inputs: tuple[str, ...],
    advisory_decision_outputs: tuple[str, ...],
) -> EscalationGateProfile:
    return EscalationGateProfile(
        gate_id=gate_id,
        gate_name=gate_name,
        gate_kind=gate_kind,
        source_condition_ids=source_condition_ids,
        source_loop_ids=source_loop_ids,
        source_registries=_ESCALATION_GATE_SOURCE_REGISTRIES,
        required_passive_inputs=required_passive_inputs,
        advisory_decision_outputs=advisory_decision_outputs,
        authority_boundary=(
            "This gate is advisory metadata only; it does not evaluate gates, "
            "approve escalation, invoke agents, route providers or models, "
            "control workflow transitions, trigger retries, execute artifacts, "
            "write memory, or modify generated output."
        ),
    )


def _creative_escalation_policy(
    *,
    policy_id: str,
    policy_name: str,
    category: CreativeEscalationPolicyCategory,
    source_gate_ids: tuple[str, ...],
    source_loop_ids: tuple[str, ...],
    creative_signal_sources: tuple[str, ...],
    advisory_policy_outputs: tuple[str, ...],
) -> CreativeEscalationPolicyRule:
    return CreativeEscalationPolicyRule(
        policy_id=policy_id,
        policy_name=policy_name,
        category=category,
        source_gate_ids=source_gate_ids,
        source_loop_ids=source_loop_ids,
        source_registries=_CREATIVE_ESCALATION_POLICY_SOURCE_REGISTRIES,
        creative_signal_sources=creative_signal_sources,
        advisory_policy_outputs=advisory_policy_outputs,
        authority_boundary=(
            "This creative escalation policy is advisory metadata only; it "
            "does not evaluate creative policy, approve escalation, evaluate "
            "gates, invoke agents, control workflow transitions, trigger "
            "retries, execute artifacts, write memory, or modify generated "
            "output."
        ),
    )


def _reflection_escalation_profile(
    *,
    profile_id: str,
    profile_name: str,
    posture: ReflectionEscalationPosture,
    source_policy_ids: tuple[str, ...],
    source_gate_ids: tuple[str, ...],
    reflection_signal_sources: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> ReflectionEscalationProfile:
    return ReflectionEscalationProfile(
        profile_id=profile_id,
        profile_name=profile_name,
        posture=posture,
        reflection_priority=posture,
        source_policy_ids=source_policy_ids,
        source_gate_ids=source_gate_ids,
        source_registries=_REFLECTION_ESCALATION_SOURCE_REGISTRIES,
        reflection_signal_sources=reflection_signal_sources,
        advisory_outputs=advisory_outputs,
        authority_boundary=(
            "This reflection escalation profile is advisory metadata only; it "
            "does not run reflection, trigger refinement, approve escalation, "
            "invoke agents, control workflow transitions, write memory, or "
            "modify generated output."
        ),
    )


def _hybrid_debate_loop(
    *,
    loop_id: str,
    topic_id: HybridDebateLoopTopic,
    source_reflection_profile_ids: tuple[str, ...],
    source_policy_ids: tuple[str, ...],
    source_specialist_loop_ids: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> HybridAgentDebateLoopProfile:
    return HybridAgentDebateLoopProfile(
        loop_id=loop_id,
        topic_id=topic_id,
        source_debate_topic_id=topic_id,
        source_reflection_profile_ids=source_reflection_profile_ids,
        source_policy_ids=source_policy_ids,
        source_specialist_loop_ids=source_specialist_loop_ids,
        source_registries=_HYBRID_DEBATE_LOOP_SOURCE_REGISTRIES,
        advisory_outputs=advisory_outputs,
        max_advisory_rounds=2,
        authority_boundary=(
            "This hybrid debate loop profile is advisory metadata only; it "
            "does not execute debate loops, invoke agents, trigger retries, "
            "route providers or models, control workflow transitions, write "
            "memory, or modify generated output."
        ),
    )


def _hybrid_agent_voting_profile(
    *,
    voting_profile_id: str,
    topic_id: HybridVotingTopic,
    source_debate_loop_id: str,
    source_reflection_profile_ids: tuple[str, ...],
    source_policy_ids: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> HybridAgentVotingProfile:
    return HybridAgentVotingProfile(
        voting_profile_id=voting_profile_id,
        topic_id=topic_id,
        source_debate_loop_id=source_debate_loop_id,
        consensus_voting_input_id=f"consensus_voting_input::{topic_id}",
        source_reflection_profile_ids=source_reflection_profile_ids,
        source_policy_ids=source_policy_ids,
        source_registries=_HYBRID_AGENT_VOTING_SOURCE_REGISTRIES,
        voting_dimensions=("agreement", "confidence", "risk", "evidence_coverage"),
        advisory_outputs=advisory_outputs,
        authority_boundary=(
            "This hybrid voting profile is advisory metadata only; it does "
            "not execute voting, select final answers, invoke agents, control "
            "workflow transitions, trigger retries, or modify generated output."
        ),
    )


def _agent_confidence_fusion_profile(
    *,
    fusion_profile_id: str,
    topic_id: AgentConfidenceFusionTopic,
    source_voting_profile_id: str,
    source_debate_loop_id: str,
    source_reflection_profile_ids: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> AgentConfidenceFusionProfile:
    return AgentConfidenceFusionProfile(
        fusion_profile_id=fusion_profile_id,
        topic_id=topic_id,
        source_voting_profile_id=source_voting_profile_id,
        source_debate_loop_id=source_debate_loop_id,
        source_confidence_surface_id="creative_confidence_engine",
        source_reflection_profile_ids=source_reflection_profile_ids,
        source_registries=_AGENT_CONFIDENCE_FUSION_SOURCE_REGISTRIES,
        confidence_signal_inputs=(
            "creative_confidence_profile",
            "consensus_vote_placeholder",
            "reflection_escalation_posture",
            "evidence_coverage_signal",
        ),
        fusion_dimensions=(
            "confidence_level",
            "agent_vote_alignment",
            "risk_uncertainty",
            "evidence_coverage",
        ),
        advisory_outputs=advisory_outputs,
        authority_boundary=(
            "This confidence fusion profile is advisory metadata only; it "
            "does not calculate confidence scores, fuse confidence, weight "
            "votes, select final answers, invoke agents, control workflow "
            "transitions, trigger retries, or modify generated output."
        ),
    )


def _decision_provenance_profile(
    *,
    provenance_profile_id: str,
    topic_id: DecisionProvenanceTopic,
    source_confidence_fusion_profile_id: str,
    source_voting_profile_id: str,
    source_debate_loop_id: str,
    source_backbone_node_ids: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> DecisionProvenanceProfile:
    return DecisionProvenanceProfile(
        provenance_profile_id=provenance_profile_id,
        topic_id=topic_id,
        source_confidence_fusion_profile_id=source_confidence_fusion_profile_id,
        source_voting_profile_id=source_voting_profile_id,
        source_debate_loop_id=source_debate_loop_id,
        source_backbone_node_ids=source_backbone_node_ids,
        source_workstation_surface_id="provenance_engine",
        source_registries=_DECISION_PROVENANCE_SOURCE_REGISTRIES,
        provenance_dimensions=(
            "decision_context",
            "confidence_lineage",
            "agent_advisory_lineage",
            "v3_backbone_lineage",
        ),
        advisory_outputs=advisory_outputs,
        authority_boundary=(
            "This decision provenance profile is advisory metadata only; it "
            "does not record provenance, emit traces, write memory, select "
            "decisions, invoke agents, control workflow transitions, trigger "
            "retries, or modify generated output."
        ),
    )


def _escalation_trace_profile(
    *,
    trace_profile_id: str,
    topic_id: EscalationTraceTopic,
    source_provenance_profile_id: str,
    source_condition_ids: tuple[str, ...],
    source_gate_ids: tuple[str, ...],
    source_escalation_signal_ids: tuple[str, ...],
    source_reflection_profile_ids: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> EscalationTraceProfile:
    return EscalationTraceProfile(
        trace_profile_id=trace_profile_id,
        topic_id=topic_id,
        source_provenance_profile_id=source_provenance_profile_id,
        source_condition_ids=source_condition_ids,
        source_gate_ids=source_gate_ids,
        source_escalation_signal_ids=source_escalation_signal_ids,
        source_reflection_profile_ids=source_reflection_profile_ids,
        source_registries=_ESCALATION_TRACE_SOURCE_REGISTRIES,
        trace_dimensions=(
            "escalation_candidate_lineage",
            "gate_visibility",
            "signal_lineage",
            "reflection_posture_lineage",
        ),
        advisory_outputs=advisory_outputs,
        authority_boundary=(
            "This escalation trace profile is advisory metadata only; it does "
            "not capture traces, emit traces, execute escalation, evaluate "
            "gates, write memory, invoke agents, control workflow transitions, "
            "trigger retries, or modify generated output."
        ),
    )


def _creative_exploration_budget_profile(
    *,
    budget_profile_id: str,
    topic_id: CreativeExplorationBudgetTopic,
    source_trace_profile_id: str,
    source_provenance_profile_id: str,
    source_escalation_signal_ids: tuple[str, ...],
    budget_posture: CreativeExplorationBudgetPosture,
    max_advisory_variants: int,
    max_advisory_refinement_passes: int,
    cost_pressure_signal: str,
    advisory_outputs: tuple[str, ...],
) -> CreativeExplorationBudgetProfile:
    return CreativeExplorationBudgetProfile(
        budget_profile_id=budget_profile_id,
        topic_id=topic_id,
        source_trace_profile_id=source_trace_profile_id,
        source_provenance_profile_id=source_provenance_profile_id,
        source_escalation_signal_ids=source_escalation_signal_ids,
        budget_posture=budget_posture,
        max_advisory_variants=max_advisory_variants,
        max_advisory_refinement_passes=max_advisory_refinement_passes,
        cost_pressure_signal=cost_pressure_signal,
        source_registries=_CREATIVE_EXPLORATION_BUDGET_SOURCE_REGISTRIES,
        budget_dimensions=(
            "variant_breadth",
            "refinement_depth",
            "cost_pressure",
            "escalation_visibility",
        ),
        advisory_outputs=advisory_outputs,
        authority_boundary=(
            "This exploration budget profile is advisory metadata only; it "
            "does not enforce budgets, generate variants, trigger refinement, "
            "route by cost, invoke agents, control workflow transitions, "
            "trigger retries, or modify generated output."
        ),
    )


def _result_normalization_profile(
    *,
    normalization_profile_id: str,
    topic_id: ResultNormalizationTopic,
    source_budget_profile_id: str,
    source_confidence_fusion_profile_id: str,
    source_provenance_profile_id: str,
    source_trace_profile_id: str,
    normalized_result_surfaces: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> ResultNormalizationProfile:
    return ResultNormalizationProfile(
        normalization_profile_id=normalization_profile_id,
        topic_id=topic_id,
        source_budget_profile_id=source_budget_profile_id,
        source_confidence_fusion_profile_id=source_confidence_fusion_profile_id,
        source_provenance_profile_id=source_provenance_profile_id,
        source_trace_profile_id=source_trace_profile_id,
        normalized_result_surfaces=normalized_result_surfaces,
        source_registries=_RESULT_NORMALIZATION_SOURCE_REGISTRIES,
        normalization_dimensions=(
            "advisory_packet_shape",
            "confidence_context",
            "provenance_context",
            "trace_context",
        ),
        advisory_outputs=advisory_outputs,
        authority_boundary=(
            "This result normalization profile is advisory metadata only; it "
            "does not transform results, rewrite outputs, enforce schemas, "
            "mutate artifacts, invoke agents, control workflow transitions, "
            "trigger retries, or modify generated output."
        ),
    )


def _return_to_workflow_handoff_profile(
    *,
    return_handoff_profile_id: str,
    topic_id: ReturnToWorkflowHandoffTopic,
    source_normalization_profile_id: str,
    source_workflow_handoff_ids: tuple[str, ...],
    target_backbone_node_ids: tuple[str, ...],
    target_workflow_surface: ReturnToWorkflowSurface,
    handoff_payload_surfaces: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> ReturnToWorkflowHandoffProfile:
    return ReturnToWorkflowHandoffProfile(
        return_handoff_profile_id=return_handoff_profile_id,
        topic_id=topic_id,
        source_normalization_profile_id=source_normalization_profile_id,
        source_gate_ids=("return_handoff_escalation_gate",),
        source_workflow_handoff_ids=source_workflow_handoff_ids,
        target_backbone_node_ids=target_backbone_node_ids,
        target_workflow_surface=target_workflow_surface,
        handoff_payload_surfaces=handoff_payload_surfaces,
        source_registries=_RETURN_TO_WORKFLOW_HANDOFF_SOURCE_REGISTRIES,
        handoff_dimensions=(
            "normalized_packet_reference",
            "return_gate_visibility",
            "workflow_surface_alignment",
            "v3_backbone_node_alignment",
        ),
        advisory_outputs=advisory_outputs,
        authority_boundary=(
            "This return-to-workflow handoff profile is advisory metadata "
            "only; it does not perform runtime handoffs, change workflow graph "
            "order, alter prompts, execute agents, control workflow transitions, "
            "trigger retries, or modify generated output."
        ),
    )


def _hitl_escalation_gate_profile(
    *,
    hitl_gate_profile_id: str,
    topic_id: HitlEscalationGateTopic,
    source_return_handoff_profile_id: str,
    source_escalation_signal_ids: tuple[str, ...],
    source_reflection_profile_ids: tuple[str, ...],
    hitl_posture: HitlEscalationPosture,
    advisory_outputs: tuple[str, ...],
) -> HitlEscalationGateProfile:
    return HitlEscalationGateProfile(
        hitl_gate_profile_id=hitl_gate_profile_id,
        topic_id=topic_id,
        source_return_handoff_profile_id=source_return_handoff_profile_id,
        source_gate_ids=("human_review_visibility_gate",),
        source_escalation_signal_ids=source_escalation_signal_ids,
        source_reflection_profile_ids=source_reflection_profile_ids,
        hitl_posture=hitl_posture,
        human_review_inputs=(
            "normalized_result_context",
            "return_handoff_context",
            "hitl_escalation_signal",
            "reflection_posture_context",
        ),
        source_registries=_HITL_ESCALATION_GATE_SOURCE_REGISTRIES,
        gate_dimensions=(
            "human_review_visibility",
            "hitl_signal_presence",
            "confidence_uncertainty",
            "return_handoff_context",
        ),
        advisory_outputs=advisory_outputs,
        authority_boundary=(
            "This HITL escalation gate profile is advisory metadata only; it "
            "does not trigger human review, request human input, evaluate "
            "gates, approve escalation, invoke agents, control workflow "
            "transitions, trigger retries, or modify generated output."
        ),
    )


def _confidence_threshold_routing_profile(
    *,
    threshold_profile_id: str,
    topic_id: ConfidenceThresholdRoutingTopic,
    source_hitl_gate_profile_id: str,
    source_confidence_fusion_profile_id: str,
    source_escalation_signal_ids: tuple[str, ...],
    confidence_band: ConfidenceThresholdBand,
    advisory_threshold_range: tuple[float, float],
    advisory_outputs: tuple[str, ...],
) -> ConfidenceThresholdRoutingProfile:
    return ConfidenceThresholdRoutingProfile(
        threshold_profile_id=threshold_profile_id,
        topic_id=topic_id,
        source_hitl_gate_profile_id=source_hitl_gate_profile_id,
        source_confidence_fusion_profile_id=source_confidence_fusion_profile_id,
        source_escalation_signal_ids=source_escalation_signal_ids,
        confidence_band=confidence_band,
        advisory_threshold_range=advisory_threshold_range,
        source_registries=_CONFIDENCE_THRESHOLD_ROUTING_SOURCE_REGISTRIES,
        routing_dimensions=(
            "confidence_band",
            "uncertainty_visibility",
            "hitl_gate_posture",
            "escalation_signal_context",
        ),
        advisory_outputs=advisory_outputs,
        authority_boundary=(
            "This confidence threshold routing profile is advisory metadata "
            "only; it does not route by confidence, change providers or models, "
            "invoke agents, evaluate thresholds, control workflow transitions, "
            "trigger retries, or modify generated output."
        ),
    )


def _cost_threshold_routing_profile(
    *,
    cost_threshold_profile_id: str,
    topic_id: CostThresholdRoutingTopic,
    source_budget_profile_id: str,
    source_confidence_threshold_profile_id: str,
    source_escalation_signal_ids: tuple[str, ...],
    cost_band: CostThresholdBand,
    advisory_cost_range: tuple[int, int],
    cost_pressure_signal: str,
    advisory_outputs: tuple[str, ...],
) -> CostThresholdRoutingProfile:
    return CostThresholdRoutingProfile(
        cost_threshold_profile_id=cost_threshold_profile_id,
        topic_id=topic_id,
        source_budget_profile_id=source_budget_profile_id,
        source_confidence_threshold_profile_id=source_confidence_threshold_profile_id,
        source_escalation_signal_ids=source_escalation_signal_ids,
        cost_band=cost_band,
        advisory_cost_range=advisory_cost_range,
        cost_pressure_signal=cost_pressure_signal,
        source_registries=_COST_THRESHOLD_ROUTING_SOURCE_REGISTRIES,
        routing_dimensions=(
            "cost_band",
            "budget_posture_context",
            "confidence_threshold_context",
            "cost_escalation_signal_context",
        ),
        advisory_outputs=advisory_outputs,
        authority_boundary=(
            "This cost threshold routing profile is advisory metadata only; "
            "it does not route by cost, enforce budgets, change providers or "
            "models, invoke agents, evaluate thresholds, control workflow "
            "transitions, trigger retries, or modify generated output."
        ),
    )


def _latency_threshold_routing_profile(
    *,
    latency_threshold_profile_id: str,
    topic_id: LatencyThresholdRoutingTopic,
    source_cost_threshold_profile_id: str,
    source_escalation_signal_ids: tuple[str, ...],
    latency_band: LatencyThresholdBand,
    advisory_latency_range_ms: tuple[int, int],
    advisory_outputs: tuple[str, ...],
) -> LatencyThresholdRoutingProfile:
    return LatencyThresholdRoutingProfile(
        latency_threshold_profile_id=latency_threshold_profile_id,
        topic_id=topic_id,
        source_cost_threshold_profile_id=source_cost_threshold_profile_id,
        source_escalation_signal_ids=source_escalation_signal_ids,
        latency_band=latency_band,
        advisory_latency_range_ms=advisory_latency_range_ms,
        latency_metadata_sources=_LATENCY_METADATA_SOURCE_SURFACES,
        source_registries=_LATENCY_THRESHOLD_ROUTING_SOURCE_REGISTRIES,
        routing_dimensions=(
            "latency_band",
            "cost_threshold_context",
            "latency_metadata_source_context",
            "latency_escalation_signal_context",
        ),
        advisory_outputs=advisory_outputs,
        authority_boundary=(
            "This latency threshold routing profile is advisory metadata only; "
            "it does not route by latency, select runtimes, change providers "
            "or models, invoke agents, evaluate thresholds, control workflow "
            "transitions, trigger retries, or modify generated output."
        ),
    )


def _ambiguity_escalation_profile(
    *,
    ambiguity_profile_id: str,
    topic_id: AmbiguityEscalationTopic,
    source_latency_threshold_profile_id: str,
    source_condition_ids: tuple[str, ...],
    source_policy_rule_ids: tuple[str, ...],
    source_escalation_signal_ids: tuple[str, ...],
    ambiguity_level: AmbiguityEscalationLevel,
    ambiguity_evidence_surfaces: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> AmbiguityEscalationProfile:
    return AmbiguityEscalationProfile(
        ambiguity_profile_id=ambiguity_profile_id,
        topic_id=topic_id,
        source_latency_threshold_profile_id=source_latency_threshold_profile_id,
        source_condition_ids=source_condition_ids,
        source_policy_rule_ids=source_policy_rule_ids,
        source_escalation_signal_ids=source_escalation_signal_ids,
        ambiguity_level=ambiguity_level,
        ambiguity_evidence_surfaces=ambiguity_evidence_surfaces,
        source_registries=_AMBIGUITY_ESCALATION_SOURCE_REGISTRIES,
        escalation_dimensions=(
            "ambiguity_level",
            "missing_information_visibility",
            "conditional_escalation_context",
            "clarification_visibility",
        ),
        advisory_outputs=advisory_outputs,
        authority_boundary=(
            "This ambiguity escalation profile is advisory metadata only; it "
            "does not evaluate ambiguity, trigger clarification, execute "
            "escalation, invoke agents, route providers or models, control "
            "workflow transitions, trigger retries, or modify generated output."
        ),
    )


def _risk_escalation_profile(
    *,
    risk_profile_id: str,
    topic_id: RiskEscalationTopic,
    source_ambiguity_profile_id: str,
    source_condition_ids: tuple[str, ...],
    source_policy_rule_ids: tuple[str, ...],
    source_creative_policy_ids: tuple[str, ...],
    source_escalation_signal_ids: tuple[str, ...],
    risk_level: RiskEscalationLevel,
    risk_evidence_surfaces: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> RiskEscalationProfile:
    return RiskEscalationProfile(
        risk_profile_id=risk_profile_id,
        topic_id=topic_id,
        source_ambiguity_profile_id=source_ambiguity_profile_id,
        source_condition_ids=source_condition_ids,
        source_policy_rule_ids=source_policy_rule_ids,
        source_creative_policy_ids=source_creative_policy_ids,
        source_escalation_signal_ids=source_escalation_signal_ids,
        risk_level=risk_level,
        risk_evidence_surfaces=risk_evidence_surfaces,
        source_registries=_RISK_ESCALATION_SOURCE_REGISTRIES,
        escalation_dimensions=(
            "risk_level",
            "artifact_risk_visibility",
            "creative_policy_context",
            "risk_signal_context",
        ),
        advisory_outputs=advisory_outputs,
        authority_boundary=(
            "This risk escalation profile is advisory metadata only; it does "
            "not evaluate risk, execute escalation, apply mitigation, invoke "
            "agents, route providers or models, control workflow transitions, "
            "trigger retries, or modify generated output."
        ),
    )


def _quality_escalation_profile(
    *,
    quality_profile_id: str,
    topic_id: QualityEscalationTopic,
    source_risk_profile_id: str,
    source_confidence_fusion_profile_id: str,
    source_creative_policy_ids: tuple[str, ...],
    source_escalation_signal_ids: tuple[str, ...],
    quality_level: QualityEscalationLevel,
    quality_evidence_surfaces: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> QualityEscalationProfile:
    return QualityEscalationProfile(
        quality_profile_id=quality_profile_id,
        topic_id=topic_id,
        source_risk_profile_id=source_risk_profile_id,
        source_confidence_fusion_profile_id=source_confidence_fusion_profile_id,
        source_creative_policy_ids=source_creative_policy_ids,
        source_escalation_signal_ids=source_escalation_signal_ids,
        quality_level=quality_level,
        quality_evidence_surfaces=quality_evidence_surfaces,
        source_registries=_QUALITY_ESCALATION_SOURCE_REGISTRIES,
        escalation_dimensions=(
            "quality_level",
            "risk_context",
            "confidence_fusion_context",
            "quality_signal_context",
        ),
        advisory_outputs=advisory_outputs,
        authority_boundary=(
            "This quality escalation profile is advisory metadata only; it "
            "does not evaluate quality, execute escalation, trigger "
            "refinement, invoke agents, route providers or models, control "
            "workflow transitions, trigger retries, or modify generated output."
        ),
    )


def _adaptive_multi_agent_escalation_profile(
    *,
    adaptive_profile_id: str,
    topic_id: AdaptiveMultiAgentEscalationTopic,
    source_condition_ids: tuple[str, ...],
    source_specialist_loop_ids: tuple[str, ...],
    source_quality_profile_id: str,
    source_risk_profile_id: str,
    source_ambiguity_profile_id: str,
    source_hitl_gate_profile_id: str,
    source_escalation_signal_ids: tuple[str, ...],
    adaptive_posture: AdaptiveMultiAgentEscalationPosture,
    adaptive_evidence_surfaces: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> AdaptiveMultiAgentEscalationProfile:
    return AdaptiveMultiAgentEscalationProfile(
        adaptive_profile_id=adaptive_profile_id,
        topic_id=topic_id,
        source_condition_ids=source_condition_ids,
        source_specialist_loop_ids=source_specialist_loop_ids,
        source_quality_profile_id=source_quality_profile_id,
        source_risk_profile_id=source_risk_profile_id,
        source_ambiguity_profile_id=source_ambiguity_profile_id,
        source_hitl_gate_profile_id=source_hitl_gate_profile_id,
        source_escalation_signal_ids=source_escalation_signal_ids,
        adaptive_posture=adaptive_posture,
        adaptive_evidence_surfaces=adaptive_evidence_surfaces,
        source_registries=_ADAPTIVE_MULTI_AGENT_ESCALATION_SOURCE_REGISTRIES,
        escalation_dimensions=(
            "adaptive_escalation_posture",
            "conditional_candidate_context",
            "specialist_loop_context",
            "quality_risk_ambiguity_context",
            "human_review_visibility",
        ),
        advisory_outputs=advisory_outputs,
        authority_boundary=(
            "This adaptive multi-agent escalation profile is advisory "
            "metadata only; it does not evaluate adaptation, execute "
            "escalation, orchestrate agents, invoke agents, select runtimes, "
            "route providers or models, control workflow transitions, "
            "trigger retries, or modify generated output."
        ),
    )


def _stage(
    *,
    stage_id: str,
    stage_name: str,
    v3_workflow_nodes: tuple[str, ...],
    future_capability_ids: tuple[str, ...],
    escalation_rule_ids: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> HybridAgenticWorkflowStage:
    return HybridAgenticWorkflowStage(
        stage_id=stage_id,
        stage_name=stage_name,
        authority_boundary=(
            "This stage is future hybrid workflow readiness metadata only; it "
            "does not change V3 workflow graph order, create agents, route "
            "providers or models, select runtimes, trigger retries, execute "
            "artifacts, or modify generated output."
        ),
        v3_workflow_nodes=v3_workflow_nodes,
        future_capability_ids=future_capability_ids,
        escalation_rule_ids=escalation_rule_ids,
        source_metadata_registries=(
            "agent_capability_registry",
            "escalation_policy_registry",
            "artifact_engine_contract_registry",
            "evaluation_engine_contract_registry",
            "workstation_engine_contract_registry",
        ),
        advisory_outputs=advisory_outputs,
        blocked_runtime_behaviors=_BLOCKED_RUNTIME_BEHAVIORS,
    )


V3_BACKBONE_MODE_NODE_PROFILES = (
    _backbone_profile(
        node_id="intake",
        phase="context_intake",
        preserved_surfaces=("request_received_event", "workflow_state"),
    ),
    _backbone_profile(
        node_id="routing",
        phase="context_intake",
        preserved_surfaces=("route_decision", "workflow_transition"),
    ),
    _backbone_profile(
        node_id="memory",
        phase="context_intake",
        preserved_surfaces=("memory_context", "project_memory_context"),
    ),
    _backbone_profile(
        node_id="retrieval",
        phase="context_intake",
        preserved_surfaces=("retrieval_context", "knowledge_chunks"),
    ),
    _backbone_profile(
        node_id="context_assembly",
        phase="context_intake",
        preserved_surfaces=("assembled_context", "context_summary"),
    ),
    _backbone_profile(
        node_id="prompt_input",
        phase="planning_reasoning",
        preserved_surfaces=("prompt_input", "operator_request_shape"),
    ),
    _backbone_profile(
        node_id="planning",
        phase="planning_reasoning",
        preserved_surfaces=("creative_plan", "planning_metadata"),
    ),
    _backbone_profile(
        node_id="director",
        phase="planning_reasoning",
        preserved_surfaces=("director_brief", "creative_direction"),
    ),
    _backbone_profile(
        node_id="reasoning",
        phase="planning_reasoning",
        preserved_surfaces=("creative_reasoning", "reasoning_evidence"),
    ),
    _backbone_profile(
        node_id="prompt_rendering",
        phase="planning_reasoning",
        preserved_surfaces=("rendered_prompt", "prompt_sections"),
    ),
    _backbone_profile(
        node_id="generation",
        phase="generation_artifact",
        preserved_surfaces=("generation_stream", "provider_generation_request"),
    ),
    _backbone_profile(
        node_id="artifact_extraction",
        phase="generation_artifact",
        preserved_surfaces=("workflow_artifacts", "artifact_metadata"),
    ),
    _backbone_profile(
        node_id="preview_preparation",
        phase="generation_artifact",
        preserved_surfaces=("preview_results", "preview_runtime_metadata"),
    ),
    _backbone_profile(
        node_id="artifact_critique",
        phase="generation_artifact",
        preserved_surfaces=("artifact_critique", "quality_observations"),
    ),
    _backbone_profile(
        node_id="review",
        phase="review_refinement",
        preserved_surfaces=("workflow_review", "hitl_recommendation"),
    ),
    _backbone_profile(
        node_id="refinement",
        phase="review_refinement",
        preserved_surfaces=("refinement_history", "refinement_decision"),
    ),
    _backbone_profile(
        node_id="finalization",
        phase="terminal_guardrail",
        preserved_surfaces=("final_workflow_event", "final_answer"),
    ),
    _backbone_profile(
        node_id="failure",
        phase="terminal_guardrail",
        preserved_surfaces=("failure_info", "failure_answer"),
    ),
)
V3_BACKBONE_MODE_REGISTRY = V3BackboneModeRegistry(
    node_profiles=V3_BACKBONE_MODE_NODE_PROFILES,
    node_ids=tuple(profile.node_id for profile in V3_BACKBONE_MODE_NODE_PROFILES),
    preserved_workflow_order=tuple(
        profile.node_id for profile in V3_BACKBONE_MODE_NODE_PROFILES
    ),
    phase_ids=_V3_BACKBONE_MODE_PHASE_IDS,
    source_registries=_V3_BACKBONE_MODE_SOURCE_REGISTRIES,
    node_count=len(V3_BACKBONE_MODE_NODE_PROFILES),
)
CONDITIONAL_MULTI_AGENT_ESCALATION_CONDITIONS = (
    _conditional_escalation_condition(
        condition_id="planning_ambiguity_multi_agent_candidate",
        condition_name="Planning Ambiguity Multi-Agent Candidate",
        category="ambiguity",
        backbone_phase="planning_reasoning",
        source_node_ids=("prompt_input", "planning", "reasoning"),
        capability_ids=(
            "v4_planner_agent",
            "adaptive_multi_agent_escalation",
        ),
        policy_rule_ids=(
            "missing_information_review",
            "future_agent_escalation_readiness",
        ),
        escalation_signal_ids=(
            "ambiguity_escalation_signal",
            "hitl_escalation_signal",
        ),
        advisory_outputs=(
            "planning_escalation_context_packet",
            "unresolved_question_summary",
            "candidate_planner_handoff",
        ),
    ),
    _conditional_escalation_condition(
        condition_id="artifact_risk_multi_agent_candidate",
        condition_name="Artifact Risk Multi-Agent Candidate",
        category="risk",
        backbone_phase="generation_artifact",
        source_node_ids=(
            "generation",
            "artifact_extraction",
            "artifact_critique",
        ),
        capability_ids=(
            "v4_artifact_agent",
            "adaptive_multi_agent_escalation",
        ),
        policy_rule_ids=(
            "artifact_risk_review",
            "future_agent_escalation_readiness",
        ),
        escalation_signal_ids=(
            "risk_escalation_signal",
            "quality_escalation_signal",
        ),
        advisory_outputs=(
            "artifact_risk_context_packet",
            "implementation_risk_summary",
            "candidate_artifact_handoff",
        ),
    ),
    _conditional_escalation_condition(
        condition_id="runtime_fit_multi_agent_candidate",
        condition_name="Runtime Fit Multi-Agent Candidate",
        category="runtime",
        backbone_phase="generation_artifact",
        source_node_ids=(
            "generation",
            "artifact_extraction",
            "preview_preparation",
        ),
        capability_ids=(
            "v4_runtime_agent",
            "v4_agent_router",
            "adaptive_multi_agent_escalation",
        ),
        policy_rule_ids=(
            "runtime_incompatibility_review",
            "future_agent_escalation_readiness",
        ),
        escalation_signal_ids=(
            "risk_escalation_signal",
            "latency_escalation_signal",
        ),
        advisory_outputs=(
            "runtime_fit_context_packet",
            "compatibility_gap_summary",
            "candidate_runtime_handoff",
        ),
    ),
    _conditional_escalation_condition(
        condition_id="evaluation_confidence_multi_agent_candidate",
        condition_name="Evaluation Confidence Multi-Agent Candidate",
        category="quality",
        backbone_phase="review_refinement",
        source_node_ids=("review", "refinement"),
        capability_ids=(
            "v4_agentic_studio",
            "adaptive_multi_agent_escalation",
        ),
        policy_rule_ids=(
            "evaluation_confidence_review",
            "future_agent_escalation_readiness",
        ),
        escalation_signal_ids=(
            "confidence_escalation_signal",
            "quality_escalation_signal",
            "hitl_escalation_signal",
        ),
        advisory_outputs=(
            "evaluation_escalation_context_packet",
            "quality_uncertainty_summary",
            "candidate_studio_handoff",
        ),
    ),
    _conditional_escalation_condition(
        condition_id="terminal_guardrail_multi_agent_candidate",
        condition_name="Terminal Guardrail Multi-Agent Candidate",
        category="hitl",
        backbone_phase="terminal_guardrail",
        source_node_ids=("finalization", "failure"),
        capability_ids=(
            "v4_agent_router",
            "adaptive_multi_agent_escalation",
        ),
        policy_rule_ids=("future_agent_escalation_readiness",),
        escalation_signal_ids=(
            "risk_escalation_signal",
            "hitl_escalation_signal",
        ),
        advisory_outputs=(
            "completion_guardrail_context_packet",
            "failure_review_posture",
            "candidate_router_handoff",
        ),
    ),
)
CONDITIONAL_MULTI_AGENT_ESCALATION_REGISTRY = (
    ConditionalMultiAgentEscalationRegistry(
        conditions=CONDITIONAL_MULTI_AGENT_ESCALATION_CONDITIONS,
        condition_ids=tuple(
            condition.condition_id
            for condition in CONDITIONAL_MULTI_AGENT_ESCALATION_CONDITIONS
        ),
        categories=tuple(
            condition.category
            for condition in CONDITIONAL_MULTI_AGENT_ESCALATION_CONDITIONS
        ),
        source_registries=_CONDITIONAL_ESCALATION_SOURCE_REGISTRIES,
        backbone_node_ids=V3_BACKBONE_MODE_REGISTRY.node_ids,
        condition_count=len(CONDITIONAL_MULTI_AGENT_ESCALATION_CONDITIONS),
    )
)
SPECIALIST_AGENT_LOOPS = (
    _specialist_agent_loop(
        loop_id="planning_specialist_agent_loop",
        loop_name="Planning Specialist Agent Loop",
        category="planning",
        specialist_agent_ids=("planner_agent", "research_agent"),
        source_condition_ids=("planning_ambiguity_multi_agent_candidate",),
        source_node_ids=("prompt_input", "planning", "reasoning"),
        loop_inputs=(
            "planning_escalation_context_packet",
            "unresolved_question_summary",
        ),
        advisory_outputs=(
            "planning_loop_notes",
            "research_gap_summary",
            "planner_handoff_recommendation",
        ),
        max_advisory_passes=2,
    ),
    _specialist_agent_loop(
        loop_id="artifact_specialist_agent_loop",
        loop_name="Artifact Specialist Agent Loop",
        category="artifact",
        specialist_agent_ids=(
            "artifact_agent",
            "art_direction_agent",
            "style_agent",
        ),
        source_condition_ids=("artifact_risk_multi_agent_candidate",),
        source_node_ids=(
            "generation",
            "artifact_extraction",
            "artifact_critique",
        ),
        loop_inputs=(
            "artifact_risk_context_packet",
            "implementation_risk_summary",
        ),
        advisory_outputs=(
            "artifact_loop_notes",
            "art_direction_review_summary",
            "style_consistency_handoff",
        ),
        max_advisory_passes=2,
    ),
    _specialist_agent_loop(
        loop_id="runtime_specialist_agent_loop",
        loop_name="Runtime Specialist Agent Loop",
        category="runtime",
        specialist_agent_ids=("runtime_agent", "artifact_agent"),
        source_condition_ids=("runtime_fit_multi_agent_candidate",),
        source_node_ids=(
            "generation",
            "artifact_extraction",
            "preview_preparation",
        ),
        loop_inputs=(
            "runtime_fit_context_packet",
            "compatibility_gap_summary",
        ),
        advisory_outputs=(
            "runtime_loop_notes",
            "compatibility_review_summary",
            "runtime_handoff_recommendation",
        ),
        max_advisory_passes=2,
    ),
    _specialist_agent_loop(
        loop_id="evaluation_specialist_agent_loop",
        loop_name="Evaluation Specialist Agent Loop",
        category="evaluation",
        specialist_agent_ids=(
            "critic_agent",
            "aesthetic_critic_agent",
            "creative_curator_agent",
            "refiner_agent",
        ),
        source_condition_ids=("evaluation_confidence_multi_agent_candidate",),
        source_node_ids=("review", "refinement"),
        loop_inputs=(
            "evaluation_escalation_context_packet",
            "quality_uncertainty_summary",
        ),
        advisory_outputs=(
            "evaluation_loop_notes",
            "critic_disagreement_summary",
            "refinement_handoff_recommendation",
        ),
        max_advisory_passes=3,
    ),
    _specialist_agent_loop(
        loop_id="synthesis_specialist_agent_loop",
        loop_name="Synthesis Specialist Agent Loop",
        category="synthesis",
        specialist_agent_ids=(
            "final_synthesizer_agent",
            "narrative_symbolic_agent",
            "creative_curator_agent",
        ),
        source_condition_ids=("terminal_guardrail_multi_agent_candidate",),
        source_node_ids=("finalization", "failure"),
        loop_inputs=(
            "completion_guardrail_context_packet",
            "failure_review_posture",
        ),
        advisory_outputs=(
            "synthesis_loop_notes",
            "terminal_handoff_summary",
            "final_synthesis_recommendation",
        ),
        max_advisory_passes=1,
    ),
)
SPECIALIST_AGENT_LOOP_REGISTRY = SpecialistAgentLoopRegistry(
    loops=SPECIALIST_AGENT_LOOPS,
    loop_ids=tuple(loop.loop_id for loop in SPECIALIST_AGENT_LOOPS),
    categories=tuple(loop.category for loop in SPECIALIST_AGENT_LOOPS),
    source_registries=_SPECIALIST_AGENT_LOOP_SOURCE_REGISTRIES,
    agent_ids=_KNOWN_SPECIALIST_AGENT_IDS,
    condition_ids=CONDITIONAL_MULTI_AGENT_ESCALATION_REGISTRY.condition_ids,
    backbone_node_ids=V3_BACKBONE_MODE_REGISTRY.node_ids,
    loop_count=len(SPECIALIST_AGENT_LOOPS),
)
ESCALATION_GATES = (
    _escalation_gate(
        gate_id="backbone_entry_escalation_gate",
        gate_name="Backbone Entry Escalation Gate",
        gate_kind="backbone_entry",
        source_condition_ids=(
            "planning_ambiguity_multi_agent_candidate",
            "artifact_risk_multi_agent_candidate",
            "runtime_fit_multi_agent_candidate",
        ),
        source_loop_ids=(),
        required_passive_inputs=(
            "v3_backbone_mode_registry",
            "conditional_escalation_conditions",
        ),
        advisory_decision_outputs=(
            "backbone_entry_gate_notes",
            "candidate_condition_summary",
        ),
    ),
    _escalation_gate(
        gate_id="evidence_completeness_escalation_gate",
        gate_name="Evidence Completeness Escalation Gate",
        gate_kind="evidence_completeness",
        source_condition_ids=CONDITIONAL_MULTI_AGENT_ESCALATION_REGISTRY.condition_ids,
        source_loop_ids=(),
        required_passive_inputs=(
            "condition_source_registries",
            "policy_rule_references",
            "escalation_signal_references",
        ),
        advisory_decision_outputs=(
            "evidence_completeness_notes",
            "missing_metadata_summary",
        ),
    ),
    _escalation_gate(
        gate_id="specialist_loop_boundary_gate",
        gate_name="Specialist Loop Boundary Gate",
        gate_kind="specialist_loop_boundary",
        source_condition_ids=CONDITIONAL_MULTI_AGENT_ESCALATION_REGISTRY.condition_ids,
        source_loop_ids=SPECIALIST_AGENT_LOOP_REGISTRY.loop_ids,
        required_passive_inputs=(
            "specialist_agent_loop_registry",
            "agent_contract_registry",
            "loop_pass_limits",
        ),
        advisory_decision_outputs=(
            "loop_boundary_notes",
            "specialist_loop_candidate_summary",
        ),
    ),
    _escalation_gate(
        gate_id="human_review_visibility_gate",
        gate_name="Human Review Visibility Gate",
        gate_kind="human_review_visibility",
        source_condition_ids=(
            "planning_ambiguity_multi_agent_candidate",
            "evaluation_confidence_multi_agent_candidate",
            "terminal_guardrail_multi_agent_candidate",
        ),
        source_loop_ids=(
            "planning_specialist_agent_loop",
            "evaluation_specialist_agent_loop",
            "synthesis_specialist_agent_loop",
        ),
        required_passive_inputs=(
            "hitl_escalation_signal",
            "human_review_posture",
            "operator_review_surface",
        ),
        advisory_decision_outputs=(
            "human_review_visibility_notes",
            "hitl_surface_summary",
        ),
    ),
    _escalation_gate(
        gate_id="return_handoff_escalation_gate",
        gate_name="Return Handoff Escalation Gate",
        gate_kind="return_handoff",
        source_condition_ids=CONDITIONAL_MULTI_AGENT_ESCALATION_REGISTRY.condition_ids,
        source_loop_ids=SPECIALIST_AGENT_LOOP_REGISTRY.loop_ids,
        required_passive_inputs=(
            "v3_backbone_mode_registry",
            "specialist_loop_advisory_outputs",
            "final_handoff_summary",
        ),
        advisory_decision_outputs=(
            "return_handoff_gate_notes",
            "backbone_rejoin_summary",
        ),
    ),
)
ESCALATION_GATE_REGISTRY = EscalationGateRegistry(
    gates=ESCALATION_GATES,
    gate_ids=tuple(gate.gate_id for gate in ESCALATION_GATES),
    gate_kinds=tuple(gate.gate_kind for gate in ESCALATION_GATES),
    source_registries=_ESCALATION_GATE_SOURCE_REGISTRIES,
    condition_ids=CONDITIONAL_MULTI_AGENT_ESCALATION_REGISTRY.condition_ids,
    loop_ids=SPECIALIST_AGENT_LOOP_REGISTRY.loop_ids,
    gate_count=len(ESCALATION_GATES),
)
CREATIVE_ESCALATION_POLICIES = (
    _creative_escalation_policy(
        policy_id="concept_ambiguity_creative_escalation_policy",
        policy_name="Concept Ambiguity Creative Escalation Policy",
        category="concept",
        source_gate_ids=(
            "backbone_entry_escalation_gate",
            "evidence_completeness_escalation_gate",
        ),
        source_loop_ids=("planning_specialist_agent_loop",),
        creative_signal_sources=(
            "planning_gap_summary",
            "missing_information",
            "creative_intent",
        ),
        advisory_policy_outputs=(
            "concept_escalation_policy_notes",
            "planning_clarity_review_summary",
        ),
    ),
    _creative_escalation_policy(
        policy_id="aesthetic_risk_creative_escalation_policy",
        policy_name="Aesthetic Risk Creative Escalation Policy",
        category="aesthetic",
        source_gate_ids=(
            "specialist_loop_boundary_gate",
            "human_review_visibility_gate",
        ),
        source_loop_ids=("artifact_specialist_agent_loop",),
        creative_signal_sources=(
            "artifact_risk_summary",
            "style_consistency_handoff",
            "aesthetic_critic_signals",
        ),
        advisory_policy_outputs=(
            "aesthetic_escalation_policy_notes",
            "artifact_style_review_summary",
        ),
    ),
    _creative_escalation_policy(
        policy_id="runtime_fit_creative_escalation_policy",
        policy_name="Runtime Fit Creative Escalation Policy",
        category="runtime",
        source_gate_ids=(
            "evidence_completeness_escalation_gate",
            "specialist_loop_boundary_gate",
        ),
        source_loop_ids=("runtime_specialist_agent_loop",),
        creative_signal_sources=(
            "runtime_fit_context_packet",
            "compatibility_gap_summary",
            "preview_runtime_metadata",
        ),
        advisory_policy_outputs=(
            "runtime_escalation_policy_notes",
            "compatibility_policy_summary",
        ),
    ),
    _creative_escalation_policy(
        policy_id="quality_uncertainty_creative_escalation_policy",
        policy_name="Quality Uncertainty Creative Escalation Policy",
        category="quality",
        source_gate_ids=(
            "human_review_visibility_gate",
            "return_handoff_escalation_gate",
        ),
        source_loop_ids=("evaluation_specialist_agent_loop",),
        creative_signal_sources=(
            "quality_uncertainty_summary",
            "creative_confidence",
            "evaluation_reports",
        ),
        advisory_policy_outputs=(
            "quality_escalation_policy_notes",
            "evaluation_confidence_review_summary",
        ),
    ),
    _creative_escalation_policy(
        policy_id="terminal_synthesis_creative_escalation_policy",
        policy_name="Terminal Synthesis Creative Escalation Policy",
        category="synthesis",
        source_gate_ids=(
            "human_review_visibility_gate",
            "return_handoff_escalation_gate",
        ),
        source_loop_ids=("synthesis_specialist_agent_loop",),
        creative_signal_sources=(
            "completion_guardrail_context_packet",
            "failure_review_posture",
            "final_handoff_summary",
        ),
        advisory_policy_outputs=(
            "synthesis_escalation_policy_notes",
            "terminal_creative_review_summary",
        ),
    ),
)
CREATIVE_ESCALATION_POLICY_REGISTRY = CreativeEscalationPolicyRegistry(
    policies=CREATIVE_ESCALATION_POLICIES,
    policy_ids=tuple(policy.policy_id for policy in CREATIVE_ESCALATION_POLICIES),
    categories=tuple(policy.category for policy in CREATIVE_ESCALATION_POLICIES),
    source_registries=_CREATIVE_ESCALATION_POLICY_SOURCE_REGISTRIES,
    gate_ids=ESCALATION_GATE_REGISTRY.gate_ids,
    loop_ids=SPECIALIST_AGENT_LOOP_REGISTRY.loop_ids,
    policy_count=len(CREATIVE_ESCALATION_POLICIES),
)
REFLECTION_ESCALATION_PROFILES = (
    _reflection_escalation_profile(
        profile_id="reflection_none_escalation_profile",
        profile_name="Reflection None Escalation Profile",
        posture="none",
        source_policy_ids=("quality_uncertainty_creative_escalation_policy",),
        source_gate_ids=("return_handoff_escalation_gate",),
        reflection_signal_sources=(
            "reflection_priority_none",
            "expected_quality_gain_none",
        ),
        advisory_outputs=("reflection_no_escalation_notes",),
    ),
    _reflection_escalation_profile(
        profile_id="reflection_low_escalation_profile",
        profile_name="Reflection Low Escalation Profile",
        posture="low",
        source_policy_ids=(
            "concept_ambiguity_creative_escalation_policy",
            "quality_uncertainty_creative_escalation_policy",
        ),
        source_gate_ids=(
            "evidence_completeness_escalation_gate",
            "return_handoff_escalation_gate",
        ),
        reflection_signal_sources=(
            "reflection_priority_low",
            "expected_quality_gain_low",
        ),
        advisory_outputs=("reflection_low_escalation_notes",),
    ),
    _reflection_escalation_profile(
        profile_id="reflection_medium_escalation_profile",
        profile_name="Reflection Medium Escalation Profile",
        posture="medium",
        source_policy_ids=(
            "concept_ambiguity_creative_escalation_policy",
            "runtime_fit_creative_escalation_policy",
        ),
        source_gate_ids=(
            "backbone_entry_escalation_gate",
            "evidence_completeness_escalation_gate",
        ),
        reflection_signal_sources=(
            "reflection_priority_medium",
            "expected_risk_reduction_medium",
        ),
        advisory_outputs=("reflection_medium_escalation_notes",),
    ),
    _reflection_escalation_profile(
        profile_id="reflection_high_escalation_profile",
        profile_name="Reflection High Escalation Profile",
        posture="high",
        source_policy_ids=(
            "runtime_fit_creative_escalation_policy",
            "quality_uncertainty_creative_escalation_policy",
        ),
        source_gate_ids=(
            "specialist_loop_boundary_gate",
            "human_review_visibility_gate",
        ),
        reflection_signal_sources=(
            "reflection_priority_high",
            "expected_quality_gain_high",
            "expected_risk_reduction_high",
        ),
        advisory_outputs=("reflection_high_escalation_notes",),
    ),
    _reflection_escalation_profile(
        profile_id="reflection_critical_escalation_profile",
        profile_name="Reflection Critical Escalation Profile",
        posture="critical",
        source_policy_ids=(
            "quality_uncertainty_creative_escalation_policy",
            "terminal_synthesis_creative_escalation_policy",
        ),
        source_gate_ids=(
            "human_review_visibility_gate",
            "return_handoff_escalation_gate",
        ),
        reflection_signal_sources=(
            "reflection_priority_critical",
            "hitl_recommendation_required",
            "unresolved_questions",
        ),
        advisory_outputs=("reflection_critical_escalation_notes",),
    ),
)
REFLECTION_ESCALATION_REGISTRY = ReflectionEscalationRegistry(
    profiles=REFLECTION_ESCALATION_PROFILES,
    profile_ids=tuple(profile.profile_id for profile in REFLECTION_ESCALATION_PROFILES),
    postures=tuple(profile.posture for profile in REFLECTION_ESCALATION_PROFILES),
    source_registries=_REFLECTION_ESCALATION_SOURCE_REGISTRIES,
    policy_ids=CREATIVE_ESCALATION_POLICY_REGISTRY.policy_ids,
    gate_ids=ESCALATION_GATE_REGISTRY.gate_ids,
    profile_count=len(REFLECTION_ESCALATION_PROFILES),
)
HYBRID_AGENT_DEBATE_LOOPS = (
    _hybrid_debate_loop(
        loop_id="hybrid_debate_loop::planning_execution_fit",
        topic_id="planning_execution_fit",
        source_reflection_profile_ids=(
            "reflection_medium_escalation_profile",
            "reflection_high_escalation_profile",
        ),
        source_policy_ids=("concept_ambiguity_creative_escalation_policy",),
        source_specialist_loop_ids=("planning_specialist_agent_loop",),
        advisory_outputs=(
            "planning_debate_readiness_notes",
            "planner_counterclaim_context",
        ),
    ),
    _hybrid_debate_loop(
        loop_id="hybrid_debate_loop::style_aesthetic_alignment",
        topic_id="style_aesthetic_alignment",
        source_reflection_profile_ids=("reflection_high_escalation_profile",),
        source_policy_ids=("aesthetic_risk_creative_escalation_policy",),
        source_specialist_loop_ids=("artifact_specialist_agent_loop",),
        advisory_outputs=(
            "style_debate_readiness_notes",
            "aesthetic_counterclaim_context",
        ),
    ),
    _hybrid_debate_loop(
        loop_id="hybrid_debate_loop::curation_refinement_need",
        topic_id="curation_refinement_need",
        source_reflection_profile_ids=(
            "reflection_high_escalation_profile",
            "reflection_critical_escalation_profile",
        ),
        source_policy_ids=("quality_uncertainty_creative_escalation_policy",),
        source_specialist_loop_ids=("evaluation_specialist_agent_loop",),
        advisory_outputs=(
            "curation_debate_readiness_notes",
            "refinement_counterclaim_context",
        ),
    ),
    _hybrid_debate_loop(
        loop_id="hybrid_debate_loop::final_synthesis_readiness",
        topic_id="final_synthesis_readiness",
        source_reflection_profile_ids=("reflection_critical_escalation_profile",),
        source_policy_ids=("terminal_synthesis_creative_escalation_policy",),
        source_specialist_loop_ids=("synthesis_specialist_agent_loop",),
        advisory_outputs=(
            "synthesis_debate_readiness_notes",
            "final_counterclaim_context",
        ),
    ),
)
HYBRID_AGENT_DEBATE_LOOP_REGISTRY = HybridAgentDebateLoopRegistry(
    debate_loops=HYBRID_AGENT_DEBATE_LOOPS,
    loop_ids=tuple(loop.loop_id for loop in HYBRID_AGENT_DEBATE_LOOPS),
    topic_ids=tuple(loop.topic_id for loop in HYBRID_AGENT_DEBATE_LOOPS),
    source_registries=_HYBRID_DEBATE_LOOP_SOURCE_REGISTRIES,
    reflection_profile_ids=REFLECTION_ESCALATION_REGISTRY.profile_ids,
    policy_ids=CREATIVE_ESCALATION_POLICY_REGISTRY.policy_ids,
    specialist_loop_ids=SPECIALIST_AGENT_LOOP_REGISTRY.loop_ids,
    loop_count=len(HYBRID_AGENT_DEBATE_LOOPS),
)
HYBRID_AGENT_VOTING_PROFILES = (
    _hybrid_agent_voting_profile(
        voting_profile_id="hybrid_agent_voting::planning_execution_fit",
        topic_id="planning_execution_fit",
        source_debate_loop_id="hybrid_debate_loop::planning_execution_fit",
        source_reflection_profile_ids=(
            "reflection_medium_escalation_profile",
            "reflection_high_escalation_profile",
        ),
        source_policy_ids=("concept_ambiguity_creative_escalation_policy",),
        advisory_outputs=(
            "planning_vote_placeholder",
            "planning_consensus_context",
        ),
    ),
    _hybrid_agent_voting_profile(
        voting_profile_id="hybrid_agent_voting::style_aesthetic_alignment",
        topic_id="style_aesthetic_alignment",
        source_debate_loop_id="hybrid_debate_loop::style_aesthetic_alignment",
        source_reflection_profile_ids=("reflection_high_escalation_profile",),
        source_policy_ids=("aesthetic_risk_creative_escalation_policy",),
        advisory_outputs=(
            "style_vote_placeholder",
            "aesthetic_consensus_context",
        ),
    ),
    _hybrid_agent_voting_profile(
        voting_profile_id="hybrid_agent_voting::curation_refinement_need",
        topic_id="curation_refinement_need",
        source_debate_loop_id="hybrid_debate_loop::curation_refinement_need",
        source_reflection_profile_ids=(
            "reflection_high_escalation_profile",
            "reflection_critical_escalation_profile",
        ),
        source_policy_ids=("quality_uncertainty_creative_escalation_policy",),
        advisory_outputs=(
            "curation_vote_placeholder",
            "refinement_consensus_context",
        ),
    ),
    _hybrid_agent_voting_profile(
        voting_profile_id="hybrid_agent_voting::final_synthesis_readiness",
        topic_id="final_synthesis_readiness",
        source_debate_loop_id="hybrid_debate_loop::final_synthesis_readiness",
        source_reflection_profile_ids=("reflection_critical_escalation_profile",),
        source_policy_ids=("terminal_synthesis_creative_escalation_policy",),
        advisory_outputs=(
            "synthesis_vote_placeholder",
            "final_consensus_context",
        ),
    ),
)
HYBRID_AGENT_VOTING_REGISTRY = HybridAgentVotingRegistry(
    voting_profiles=HYBRID_AGENT_VOTING_PROFILES,
    voting_profile_ids=tuple(
        profile.voting_profile_id for profile in HYBRID_AGENT_VOTING_PROFILES
    ),
    topic_ids=tuple(profile.topic_id for profile in HYBRID_AGENT_VOTING_PROFILES),
    source_registries=_HYBRID_AGENT_VOTING_SOURCE_REGISTRIES,
    debate_loop_ids=HYBRID_AGENT_DEBATE_LOOP_REGISTRY.loop_ids,
    reflection_profile_ids=REFLECTION_ESCALATION_REGISTRY.profile_ids,
    policy_ids=CREATIVE_ESCALATION_POLICY_REGISTRY.policy_ids,
    profile_count=len(HYBRID_AGENT_VOTING_PROFILES),
)
AGENT_CONFIDENCE_FUSION_PROFILES = (
    _agent_confidence_fusion_profile(
        fusion_profile_id="agent_confidence_fusion::planning_execution_fit",
        topic_id="planning_execution_fit",
        source_voting_profile_id="hybrid_agent_voting::planning_execution_fit",
        source_debate_loop_id="hybrid_debate_loop::planning_execution_fit",
        source_reflection_profile_ids=(
            "reflection_medium_escalation_profile",
            "reflection_high_escalation_profile",
        ),
        advisory_outputs=(
            "planning_confidence_fusion_placeholder",
            "planning_confidence_context",
        ),
    ),
    _agent_confidence_fusion_profile(
        fusion_profile_id="agent_confidence_fusion::style_aesthetic_alignment",
        topic_id="style_aesthetic_alignment",
        source_voting_profile_id="hybrid_agent_voting::style_aesthetic_alignment",
        source_debate_loop_id="hybrid_debate_loop::style_aesthetic_alignment",
        source_reflection_profile_ids=("reflection_high_escalation_profile",),
        advisory_outputs=(
            "style_confidence_fusion_placeholder",
            "aesthetic_confidence_context",
        ),
    ),
    _agent_confidence_fusion_profile(
        fusion_profile_id="agent_confidence_fusion::curation_refinement_need",
        topic_id="curation_refinement_need",
        source_voting_profile_id="hybrid_agent_voting::curation_refinement_need",
        source_debate_loop_id="hybrid_debate_loop::curation_refinement_need",
        source_reflection_profile_ids=(
            "reflection_high_escalation_profile",
            "reflection_critical_escalation_profile",
        ),
        advisory_outputs=(
            "curation_confidence_fusion_placeholder",
            "refinement_confidence_context",
        ),
    ),
    _agent_confidence_fusion_profile(
        fusion_profile_id="agent_confidence_fusion::final_synthesis_readiness",
        topic_id="final_synthesis_readiness",
        source_voting_profile_id="hybrid_agent_voting::final_synthesis_readiness",
        source_debate_loop_id="hybrid_debate_loop::final_synthesis_readiness",
        source_reflection_profile_ids=("reflection_critical_escalation_profile",),
        advisory_outputs=(
            "synthesis_confidence_fusion_placeholder",
            "final_confidence_context",
        ),
    ),
)
AGENT_CONFIDENCE_FUSION_REGISTRY = AgentConfidenceFusionRegistry(
    fusion_profiles=AGENT_CONFIDENCE_FUSION_PROFILES,
    fusion_profile_ids=tuple(
        profile.fusion_profile_id for profile in AGENT_CONFIDENCE_FUSION_PROFILES
    ),
    topic_ids=tuple(profile.topic_id for profile in AGENT_CONFIDENCE_FUSION_PROFILES),
    source_registries=_AGENT_CONFIDENCE_FUSION_SOURCE_REGISTRIES,
    voting_profile_ids=HYBRID_AGENT_VOTING_REGISTRY.voting_profile_ids,
    debate_loop_ids=HYBRID_AGENT_DEBATE_LOOP_REGISTRY.loop_ids,
    reflection_profile_ids=REFLECTION_ESCALATION_REGISTRY.profile_ids,
    confidence_surface_ids=("creative_confidence_engine",),
    profile_count=len(AGENT_CONFIDENCE_FUSION_PROFILES),
)
DECISION_PROVENANCE_PROFILES = (
    _decision_provenance_profile(
        provenance_profile_id="decision_provenance::planning_execution_fit",
        topic_id="planning_execution_fit",
        source_confidence_fusion_profile_id=(
            "agent_confidence_fusion::planning_execution_fit"
        ),
        source_voting_profile_id="hybrid_agent_voting::planning_execution_fit",
        source_debate_loop_id="hybrid_debate_loop::planning_execution_fit",
        source_backbone_node_ids=("planning", "director", "reasoning"),
        advisory_outputs=(
            "planning_decision_lineage_placeholder",
            "planning_provenance_context",
        ),
    ),
    _decision_provenance_profile(
        provenance_profile_id="decision_provenance::style_aesthetic_alignment",
        topic_id="style_aesthetic_alignment",
        source_confidence_fusion_profile_id=(
            "agent_confidence_fusion::style_aesthetic_alignment"
        ),
        source_voting_profile_id="hybrid_agent_voting::style_aesthetic_alignment",
        source_debate_loop_id="hybrid_debate_loop::style_aesthetic_alignment",
        source_backbone_node_ids=(
            "prompt_rendering",
            "generation",
            "artifact_extraction",
        ),
        advisory_outputs=(
            "style_decision_lineage_placeholder",
            "aesthetic_provenance_context",
        ),
    ),
    _decision_provenance_profile(
        provenance_profile_id="decision_provenance::curation_refinement_need",
        topic_id="curation_refinement_need",
        source_confidence_fusion_profile_id=(
            "agent_confidence_fusion::curation_refinement_need"
        ),
        source_voting_profile_id="hybrid_agent_voting::curation_refinement_need",
        source_debate_loop_id="hybrid_debate_loop::curation_refinement_need",
        source_backbone_node_ids=("artifact_critique", "review", "refinement"),
        advisory_outputs=(
            "curation_decision_lineage_placeholder",
            "refinement_provenance_context",
        ),
    ),
    _decision_provenance_profile(
        provenance_profile_id="decision_provenance::final_synthesis_readiness",
        topic_id="final_synthesis_readiness",
        source_confidence_fusion_profile_id=(
            "agent_confidence_fusion::final_synthesis_readiness"
        ),
        source_voting_profile_id="hybrid_agent_voting::final_synthesis_readiness",
        source_debate_loop_id="hybrid_debate_loop::final_synthesis_readiness",
        source_backbone_node_ids=("review", "refinement", "finalization"),
        advisory_outputs=(
            "synthesis_decision_lineage_placeholder",
            "final_provenance_context",
        ),
    ),
)
DECISION_PROVENANCE_REGISTRY = DecisionProvenanceRegistry(
    provenance_profiles=DECISION_PROVENANCE_PROFILES,
    provenance_profile_ids=tuple(
        profile.provenance_profile_id for profile in DECISION_PROVENANCE_PROFILES
    ),
    topic_ids=tuple(profile.topic_id for profile in DECISION_PROVENANCE_PROFILES),
    source_registries=_DECISION_PROVENANCE_SOURCE_REGISTRIES,
    confidence_fusion_profile_ids=AGENT_CONFIDENCE_FUSION_REGISTRY.fusion_profile_ids,
    voting_profile_ids=HYBRID_AGENT_VOTING_REGISTRY.voting_profile_ids,
    debate_loop_ids=HYBRID_AGENT_DEBATE_LOOP_REGISTRY.loop_ids,
    backbone_node_ids=V3_BACKBONE_MODE_REGISTRY.node_ids,
    workstation_surface_ids=(
        "workstation_state",
        "session_intelligence",
        "workflow_explorer",
        "provenance_engine",
        "creative_timeline",
        "v3_inspector_panels",
        "workstation_dashboard",
    ),
    profile_count=len(DECISION_PROVENANCE_PROFILES),
)
ESCALATION_TRACE_PROFILES = (
    _escalation_trace_profile(
        trace_profile_id="escalation_trace::planning_execution_fit",
        topic_id="planning_execution_fit",
        source_provenance_profile_id="decision_provenance::planning_execution_fit",
        source_condition_ids=("planning_ambiguity_multi_agent_candidate",),
        source_gate_ids=(
            "backbone_entry_escalation_gate",
            "evidence_completeness_escalation_gate",
        ),
        source_escalation_signal_ids=(
            "ambiguity_escalation_signal",
            "hitl_escalation_signal",
        ),
        source_reflection_profile_ids=(
            "reflection_medium_escalation_profile",
            "reflection_high_escalation_profile",
        ),
        advisory_outputs=(
            "planning_escalation_trace_placeholder",
            "planning_trace_context",
        ),
    ),
    _escalation_trace_profile(
        trace_profile_id="escalation_trace::style_aesthetic_alignment",
        topic_id="style_aesthetic_alignment",
        source_provenance_profile_id=(
            "decision_provenance::style_aesthetic_alignment"
        ),
        source_condition_ids=("artifact_risk_multi_agent_candidate",),
        source_gate_ids=("specialist_loop_boundary_gate",),
        source_escalation_signal_ids=(
            "risk_escalation_signal",
            "quality_escalation_signal",
        ),
        source_reflection_profile_ids=("reflection_high_escalation_profile",),
        advisory_outputs=(
            "style_escalation_trace_placeholder",
            "aesthetic_trace_context",
        ),
    ),
    _escalation_trace_profile(
        trace_profile_id="escalation_trace::curation_refinement_need",
        topic_id="curation_refinement_need",
        source_provenance_profile_id="decision_provenance::curation_refinement_need",
        source_condition_ids=("evaluation_confidence_multi_agent_candidate",),
        source_gate_ids=(
            "evidence_completeness_escalation_gate",
            "return_handoff_escalation_gate",
        ),
        source_escalation_signal_ids=(
            "confidence_escalation_signal",
            "quality_escalation_signal",
        ),
        source_reflection_profile_ids=(
            "reflection_high_escalation_profile",
            "reflection_critical_escalation_profile",
        ),
        advisory_outputs=(
            "curation_escalation_trace_placeholder",
            "refinement_trace_context",
        ),
    ),
    _escalation_trace_profile(
        trace_profile_id="escalation_trace::final_synthesis_readiness",
        topic_id="final_synthesis_readiness",
        source_provenance_profile_id="decision_provenance::final_synthesis_readiness",
        source_condition_ids=("terminal_guardrail_multi_agent_candidate",),
        source_gate_ids=(
            "human_review_visibility_gate",
            "return_handoff_escalation_gate",
        ),
        source_escalation_signal_ids=(
            "hitl_escalation_signal",
            "quality_escalation_signal",
        ),
        source_reflection_profile_ids=("reflection_critical_escalation_profile",),
        advisory_outputs=(
            "synthesis_escalation_trace_placeholder",
            "final_trace_context",
        ),
    ),
)
ESCALATION_TRACE_REGISTRY = EscalationTraceRegistry(
    trace_profiles=ESCALATION_TRACE_PROFILES,
    trace_profile_ids=tuple(
        profile.trace_profile_id for profile in ESCALATION_TRACE_PROFILES
    ),
    topic_ids=tuple(profile.topic_id for profile in ESCALATION_TRACE_PROFILES),
    source_registries=_ESCALATION_TRACE_SOURCE_REGISTRIES,
    provenance_profile_ids=DECISION_PROVENANCE_REGISTRY.provenance_profile_ids,
    condition_ids=CONDITIONAL_MULTI_AGENT_ESCALATION_REGISTRY.condition_ids,
    gate_ids=ESCALATION_GATE_REGISTRY.gate_ids,
    escalation_signal_ids=_KNOWN_CONDITIONAL_ESCALATION_SIGNAL_IDS,
    reflection_profile_ids=REFLECTION_ESCALATION_REGISTRY.profile_ids,
    profile_count=len(ESCALATION_TRACE_PROFILES),
)
CREATIVE_EXPLORATION_BUDGET_PROFILES = (
    _creative_exploration_budget_profile(
        budget_profile_id="creative_exploration_budget::planning_execution_fit",
        topic_id="planning_execution_fit",
        source_trace_profile_id="escalation_trace::planning_execution_fit",
        source_provenance_profile_id="decision_provenance::planning_execution_fit",
        source_escalation_signal_ids=(
            "ambiguity_escalation_signal",
            "hitl_escalation_signal",
        ),
        budget_posture="moderate",
        max_advisory_variants=2,
        max_advisory_refinement_passes=1,
        cost_pressure_signal="planning_token_budget_context",
        advisory_outputs=(
            "planning_budget_posture_placeholder",
            "planning_exploration_context",
        ),
    ),
    _creative_exploration_budget_profile(
        budget_profile_id="creative_exploration_budget::style_aesthetic_alignment",
        topic_id="style_aesthetic_alignment",
        source_trace_profile_id="escalation_trace::style_aesthetic_alignment",
        source_provenance_profile_id=(
            "decision_provenance::style_aesthetic_alignment"
        ),
        source_escalation_signal_ids=(
            "risk_escalation_signal",
            "quality_escalation_signal",
        ),
        budget_posture="broad",
        max_advisory_variants=3,
        max_advisory_refinement_passes=1,
        cost_pressure_signal="style_variant_budget_context",
        advisory_outputs=(
            "style_budget_posture_placeholder",
            "aesthetic_exploration_context",
        ),
    ),
    _creative_exploration_budget_profile(
        budget_profile_id="creative_exploration_budget::curation_refinement_need",
        topic_id="curation_refinement_need",
        source_trace_profile_id="escalation_trace::curation_refinement_need",
        source_provenance_profile_id="decision_provenance::curation_refinement_need",
        source_escalation_signal_ids=(
            "confidence_escalation_signal",
            "quality_escalation_signal",
        ),
        budget_posture="guarded",
        max_advisory_variants=1,
        max_advisory_refinement_passes=2,
        cost_pressure_signal="refinement_budget_context",
        advisory_outputs=(
            "curation_budget_posture_placeholder",
            "refinement_exploration_context",
        ),
    ),
    _creative_exploration_budget_profile(
        budget_profile_id="creative_exploration_budget::final_synthesis_readiness",
        topic_id="final_synthesis_readiness",
        source_trace_profile_id="escalation_trace::final_synthesis_readiness",
        source_provenance_profile_id="decision_provenance::final_synthesis_readiness",
        source_escalation_signal_ids=(
            "hitl_escalation_signal",
            "quality_escalation_signal",
        ),
        budget_posture="narrow",
        max_advisory_variants=0,
        max_advisory_refinement_passes=1,
        cost_pressure_signal="final_synthesis_budget_context",
        advisory_outputs=(
            "synthesis_budget_posture_placeholder",
            "final_exploration_context",
        ),
    ),
)
CREATIVE_EXPLORATION_BUDGET_REGISTRY = CreativeExplorationBudgetRegistry(
    budget_profiles=CREATIVE_EXPLORATION_BUDGET_PROFILES,
    budget_profile_ids=tuple(
        profile.budget_profile_id for profile in CREATIVE_EXPLORATION_BUDGET_PROFILES
    ),
    topic_ids=tuple(
        profile.topic_id for profile in CREATIVE_EXPLORATION_BUDGET_PROFILES
    ),
    budget_postures=tuple(
        profile.budget_posture for profile in CREATIVE_EXPLORATION_BUDGET_PROFILES
    ),
    source_registries=_CREATIVE_EXPLORATION_BUDGET_SOURCE_REGISTRIES,
    trace_profile_ids=ESCALATION_TRACE_REGISTRY.trace_profile_ids,
    provenance_profile_ids=DECISION_PROVENANCE_REGISTRY.provenance_profile_ids,
    escalation_signal_ids=_KNOWN_CONDITIONAL_ESCALATION_SIGNAL_IDS,
    profile_count=len(CREATIVE_EXPLORATION_BUDGET_PROFILES),
)
RESULT_NORMALIZATION_PROFILES = (
    _result_normalization_profile(
        normalization_profile_id="result_normalization::planning_execution_fit",
        topic_id="planning_execution_fit",
        source_budget_profile_id=(
            "creative_exploration_budget::planning_execution_fit"
        ),
        source_confidence_fusion_profile_id=(
            "agent_confidence_fusion::planning_execution_fit"
        ),
        source_provenance_profile_id="decision_provenance::planning_execution_fit",
        source_trace_profile_id="escalation_trace::planning_execution_fit",
        normalized_result_surfaces=(
            "planning_advisory_packet",
            "decision_context_summary",
        ),
        advisory_outputs=(
            "planning_result_normalization_placeholder",
            "planning_normalized_context",
        ),
    ),
    _result_normalization_profile(
        normalization_profile_id="result_normalization::style_aesthetic_alignment",
        topic_id="style_aesthetic_alignment",
        source_budget_profile_id=(
            "creative_exploration_budget::style_aesthetic_alignment"
        ),
        source_confidence_fusion_profile_id=(
            "agent_confidence_fusion::style_aesthetic_alignment"
        ),
        source_provenance_profile_id=(
            "decision_provenance::style_aesthetic_alignment"
        ),
        source_trace_profile_id="escalation_trace::style_aesthetic_alignment",
        normalized_result_surfaces=(
            "aesthetic_advisory_packet",
            "style_consensus_summary",
        ),
        advisory_outputs=(
            "style_result_normalization_placeholder",
            "aesthetic_normalized_context",
        ),
    ),
    _result_normalization_profile(
        normalization_profile_id="result_normalization::curation_refinement_need",
        topic_id="curation_refinement_need",
        source_budget_profile_id=(
            "creative_exploration_budget::curation_refinement_need"
        ),
        source_confidence_fusion_profile_id=(
            "agent_confidence_fusion::curation_refinement_need"
        ),
        source_provenance_profile_id="decision_provenance::curation_refinement_need",
        source_trace_profile_id="escalation_trace::curation_refinement_need",
        normalized_result_surfaces=(
            "refinement_advisory_packet",
            "quality_context_summary",
        ),
        advisory_outputs=(
            "curation_result_normalization_placeholder",
            "refinement_normalized_context",
        ),
    ),
    _result_normalization_profile(
        normalization_profile_id="result_normalization::final_synthesis_readiness",
        topic_id="final_synthesis_readiness",
        source_budget_profile_id=(
            "creative_exploration_budget::final_synthesis_readiness"
        ),
        source_confidence_fusion_profile_id=(
            "agent_confidence_fusion::final_synthesis_readiness"
        ),
        source_provenance_profile_id="decision_provenance::final_synthesis_readiness",
        source_trace_profile_id="escalation_trace::final_synthesis_readiness",
        normalized_result_surfaces=(
            "final_synthesis_packet",
            "handoff_context_summary",
        ),
        advisory_outputs=(
            "synthesis_result_normalization_placeholder",
            "final_normalized_context",
        ),
    ),
)
RESULT_NORMALIZATION_REGISTRY = ResultNormalizationRegistry(
    normalization_profiles=RESULT_NORMALIZATION_PROFILES,
    normalization_profile_ids=tuple(
        profile.normalization_profile_id for profile in RESULT_NORMALIZATION_PROFILES
    ),
    topic_ids=tuple(profile.topic_id for profile in RESULT_NORMALIZATION_PROFILES),
    source_registries=_RESULT_NORMALIZATION_SOURCE_REGISTRIES,
    budget_profile_ids=CREATIVE_EXPLORATION_BUDGET_REGISTRY.budget_profile_ids,
    confidence_fusion_profile_ids=AGENT_CONFIDENCE_FUSION_REGISTRY.fusion_profile_ids,
    provenance_profile_ids=DECISION_PROVENANCE_REGISTRY.provenance_profile_ids,
    trace_profile_ids=ESCALATION_TRACE_REGISTRY.trace_profile_ids,
    profile_count=len(RESULT_NORMALIZATION_PROFILES),
)
RETURN_TO_WORKFLOW_HANDOFF_PROFILES = (
    _return_to_workflow_handoff_profile(
        return_handoff_profile_id="return_to_workflow_handoff::planning_execution_fit",
        topic_id="planning_execution_fit",
        source_normalization_profile_id="result_normalization::planning_execution_fit",
        source_workflow_handoff_ids=("planning_surface_agent_handoff",),
        target_backbone_node_ids=("planning", "director", "reasoning"),
        target_workflow_surface="planning",
        handoff_payload_surfaces=(
            "planning_advisory_packet",
            "decision_context_summary",
        ),
        advisory_outputs=(
            "planning_return_handoff_placeholder",
            "planning_workflow_handoff_context",
        ),
    ),
    _return_to_workflow_handoff_profile(
        return_handoff_profile_id=(
            "return_to_workflow_handoff::style_aesthetic_alignment"
        ),
        topic_id="style_aesthetic_alignment",
        source_normalization_profile_id=(
            "result_normalization::style_aesthetic_alignment"
        ),
        source_workflow_handoff_ids=("artifact_surface_agent_handoff",),
        target_backbone_node_ids=(
            "generation",
            "artifact_extraction",
            "preview_preparation",
        ),
        target_workflow_surface="artifact",
        handoff_payload_surfaces=(
            "aesthetic_advisory_packet",
            "style_consensus_summary",
        ),
        advisory_outputs=(
            "style_return_handoff_placeholder",
            "artifact_workflow_handoff_context",
        ),
    ),
    _return_to_workflow_handoff_profile(
        return_handoff_profile_id=(
            "return_to_workflow_handoff::curation_refinement_need"
        ),
        topic_id="curation_refinement_need",
        source_normalization_profile_id=(
            "result_normalization::curation_refinement_need"
        ),
        source_workflow_handoff_ids=(
            "artifact_surface_agent_handoff",
            "evaluation_surface_agent_handoff",
        ),
        target_backbone_node_ids=("artifact_critique", "review", "refinement"),
        target_workflow_surface="evaluation",
        handoff_payload_surfaces=(
            "refinement_advisory_packet",
            "quality_context_summary",
        ),
        advisory_outputs=(
            "curation_return_handoff_placeholder",
            "evaluation_workflow_handoff_context",
        ),
    ),
    _return_to_workflow_handoff_profile(
        return_handoff_profile_id=(
            "return_to_workflow_handoff::final_synthesis_readiness"
        ),
        topic_id="final_synthesis_readiness",
        source_normalization_profile_id=(
            "result_normalization::final_synthesis_readiness"
        ),
        source_workflow_handoff_ids=(
            "provenance_surface_agent_handoff",
            "finalization_surface_agent_handoff",
        ),
        target_backbone_node_ids=("review", "finalization"),
        target_workflow_surface="finalization",
        handoff_payload_surfaces=(
            "final_synthesis_packet",
            "handoff_context_summary",
        ),
        advisory_outputs=(
            "synthesis_return_handoff_placeholder",
            "finalization_workflow_handoff_context",
        ),
    ),
)
RETURN_TO_WORKFLOW_HANDOFF_REGISTRY = ReturnToWorkflowHandoffRegistry(
    handoff_profiles=RETURN_TO_WORKFLOW_HANDOFF_PROFILES,
    return_handoff_profile_ids=tuple(
        profile.return_handoff_profile_id
        for profile in RETURN_TO_WORKFLOW_HANDOFF_PROFILES
    ),
    topic_ids=tuple(profile.topic_id for profile in RETURN_TO_WORKFLOW_HANDOFF_PROFILES),
    target_workflow_surfaces=tuple(
        profile.target_workflow_surface
        for profile in RETURN_TO_WORKFLOW_HANDOFF_PROFILES
    ),
    source_registries=_RETURN_TO_WORKFLOW_HANDOFF_SOURCE_REGISTRIES,
    normalization_profile_ids=RESULT_NORMALIZATION_REGISTRY.normalization_profile_ids,
    gate_ids=ESCALATION_GATE_REGISTRY.gate_ids,
    workflow_handoff_ids=_WORKFLOW_AGENT_HANDOFF_IDS,
    backbone_node_ids=V3_BACKBONE_MODE_REGISTRY.node_ids,
    profile_count=len(RETURN_TO_WORKFLOW_HANDOFF_PROFILES),
)
HITL_ESCALATION_GATE_PROFILES = (
    _hitl_escalation_gate_profile(
        hitl_gate_profile_id="hitl_escalation_gate::planning_execution_fit",
        topic_id="planning_execution_fit",
        source_return_handoff_profile_id=(
            "return_to_workflow_handoff::planning_execution_fit"
        ),
        source_escalation_signal_ids=(
            "ambiguity_escalation_signal",
            "hitl_escalation_signal",
        ),
        source_reflection_profile_ids=(
            "reflection_medium_escalation_profile",
            "reflection_high_escalation_profile",
        ),
        hitl_posture="recommended",
        advisory_outputs=(
            "planning_hitl_gate_placeholder",
            "planning_human_review_context",
        ),
    ),
    _hitl_escalation_gate_profile(
        hitl_gate_profile_id="hitl_escalation_gate::style_aesthetic_alignment",
        topic_id="style_aesthetic_alignment",
        source_return_handoff_profile_id=(
            "return_to_workflow_handoff::style_aesthetic_alignment"
        ),
        source_escalation_signal_ids=(
            "risk_escalation_signal",
            "hitl_escalation_signal",
        ),
        source_reflection_profile_ids=("reflection_high_escalation_profile",),
        hitl_posture="optional",
        advisory_outputs=(
            "style_hitl_gate_placeholder",
            "aesthetic_human_review_context",
        ),
    ),
    _hitl_escalation_gate_profile(
        hitl_gate_profile_id="hitl_escalation_gate::curation_refinement_need",
        topic_id="curation_refinement_need",
        source_return_handoff_profile_id=(
            "return_to_workflow_handoff::curation_refinement_need"
        ),
        source_escalation_signal_ids=(
            "confidence_escalation_signal",
            "quality_escalation_signal",
            "hitl_escalation_signal",
        ),
        source_reflection_profile_ids=(
            "reflection_high_escalation_profile",
            "reflection_critical_escalation_profile",
        ),
        hitl_posture="recommended",
        advisory_outputs=(
            "curation_hitl_gate_placeholder",
            "refinement_human_review_context",
        ),
    ),
    _hitl_escalation_gate_profile(
        hitl_gate_profile_id="hitl_escalation_gate::final_synthesis_readiness",
        topic_id="final_synthesis_readiness",
        source_return_handoff_profile_id=(
            "return_to_workflow_handoff::final_synthesis_readiness"
        ),
        source_escalation_signal_ids=(
            "quality_escalation_signal",
            "hitl_escalation_signal",
        ),
        source_reflection_profile_ids=("reflection_critical_escalation_profile",),
        hitl_posture="required",
        advisory_outputs=(
            "synthesis_hitl_gate_placeholder",
            "final_human_review_context",
        ),
    ),
)
HITL_ESCALATION_GATE_REGISTRY = HitlEscalationGateRegistry(
    hitl_gate_profiles=HITL_ESCALATION_GATE_PROFILES,
    hitl_gate_profile_ids=tuple(
        profile.hitl_gate_profile_id for profile in HITL_ESCALATION_GATE_PROFILES
    ),
    topic_ids=tuple(profile.topic_id for profile in HITL_ESCALATION_GATE_PROFILES),
    hitl_postures=tuple(
        profile.hitl_posture for profile in HITL_ESCALATION_GATE_PROFILES
    ),
    source_registries=_HITL_ESCALATION_GATE_SOURCE_REGISTRIES,
    return_handoff_profile_ids=(
        RETURN_TO_WORKFLOW_HANDOFF_REGISTRY.return_handoff_profile_ids
    ),
    gate_ids=ESCALATION_GATE_REGISTRY.gate_ids,
    escalation_signal_ids=_KNOWN_CONDITIONAL_ESCALATION_SIGNAL_IDS,
    reflection_profile_ids=REFLECTION_ESCALATION_REGISTRY.profile_ids,
    profile_count=len(HITL_ESCALATION_GATE_PROFILES),
)
CONFIDENCE_THRESHOLD_ROUTING_PROFILES = (
    _confidence_threshold_routing_profile(
        threshold_profile_id="confidence_threshold_routing::planning_execution_fit",
        topic_id="planning_execution_fit",
        source_hitl_gate_profile_id="hitl_escalation_gate::planning_execution_fit",
        source_confidence_fusion_profile_id=(
            "agent_confidence_fusion::planning_execution_fit"
        ),
        source_escalation_signal_ids=(
            "confidence_escalation_signal",
            "ambiguity_escalation_signal",
        ),
        confidence_band="medium",
        advisory_threshold_range=(0.55, 0.72),
        advisory_outputs=(
            "planning_confidence_threshold_placeholder",
            "planning_confidence_routing_context",
        ),
    ),
    _confidence_threshold_routing_profile(
        threshold_profile_id="confidence_threshold_routing::style_aesthetic_alignment",
        topic_id="style_aesthetic_alignment",
        source_hitl_gate_profile_id="hitl_escalation_gate::style_aesthetic_alignment",
        source_confidence_fusion_profile_id=(
            "agent_confidence_fusion::style_aesthetic_alignment"
        ),
        source_escalation_signal_ids=(
            "confidence_escalation_signal",
            "quality_escalation_signal",
        ),
        confidence_band="high",
        advisory_threshold_range=(0.72, 0.86),
        advisory_outputs=(
            "style_confidence_threshold_placeholder",
            "aesthetic_confidence_routing_context",
        ),
    ),
    _confidence_threshold_routing_profile(
        threshold_profile_id="confidence_threshold_routing::curation_refinement_need",
        topic_id="curation_refinement_need",
        source_hitl_gate_profile_id="hitl_escalation_gate::curation_refinement_need",
        source_confidence_fusion_profile_id=(
            "agent_confidence_fusion::curation_refinement_need"
        ),
        source_escalation_signal_ids=(
            "confidence_escalation_signal",
            "quality_escalation_signal",
            "hitl_escalation_signal",
        ),
        confidence_band="low",
        advisory_threshold_range=(0.35, 0.55),
        advisory_outputs=(
            "curation_confidence_threshold_placeholder",
            "refinement_confidence_routing_context",
        ),
    ),
    _confidence_threshold_routing_profile(
        threshold_profile_id="confidence_threshold_routing::final_synthesis_readiness",
        topic_id="final_synthesis_readiness",
        source_hitl_gate_profile_id="hitl_escalation_gate::final_synthesis_readiness",
        source_confidence_fusion_profile_id=(
            "agent_confidence_fusion::final_synthesis_readiness"
        ),
        source_escalation_signal_ids=(
            "confidence_escalation_signal",
            "hitl_escalation_signal",
        ),
        confidence_band="critical",
        advisory_threshold_range=(0.0, 0.35),
        advisory_outputs=(
            "synthesis_confidence_threshold_placeholder",
            "final_confidence_routing_context",
        ),
    ),
)
CONFIDENCE_THRESHOLD_ROUTING_REGISTRY = ConfidenceThresholdRoutingRegistry(
    threshold_profiles=CONFIDENCE_THRESHOLD_ROUTING_PROFILES,
    threshold_profile_ids=tuple(
        profile.threshold_profile_id
        for profile in CONFIDENCE_THRESHOLD_ROUTING_PROFILES
    ),
    topic_ids=tuple(
        profile.topic_id for profile in CONFIDENCE_THRESHOLD_ROUTING_PROFILES
    ),
    confidence_bands=tuple(
        profile.confidence_band for profile in CONFIDENCE_THRESHOLD_ROUTING_PROFILES
    ),
    source_registries=_CONFIDENCE_THRESHOLD_ROUTING_SOURCE_REGISTRIES,
    hitl_gate_profile_ids=HITL_ESCALATION_GATE_REGISTRY.hitl_gate_profile_ids,
    confidence_fusion_profile_ids=AGENT_CONFIDENCE_FUSION_REGISTRY.fusion_profile_ids,
    escalation_signal_ids=_KNOWN_CONDITIONAL_ESCALATION_SIGNAL_IDS,
    profile_count=len(CONFIDENCE_THRESHOLD_ROUTING_PROFILES),
)
COST_THRESHOLD_ROUTING_PROFILES = (
    _cost_threshold_routing_profile(
        cost_threshold_profile_id="cost_threshold_routing::planning_execution_fit",
        topic_id="planning_execution_fit",
        source_budget_profile_id=(
            "creative_exploration_budget::planning_execution_fit"
        ),
        source_confidence_threshold_profile_id=(
            "confidence_threshold_routing::planning_execution_fit"
        ),
        source_escalation_signal_ids=(
            "cost_escalation_signal",
            "ambiguity_escalation_signal",
        ),
        cost_band="medium",
        advisory_cost_range=(2, 4),
        cost_pressure_signal="planning_token_budget_context",
        advisory_outputs=(
            "planning_cost_threshold_placeholder",
            "planning_cost_routing_context",
        ),
    ),
    _cost_threshold_routing_profile(
        cost_threshold_profile_id="cost_threshold_routing::style_aesthetic_alignment",
        topic_id="style_aesthetic_alignment",
        source_budget_profile_id=(
            "creative_exploration_budget::style_aesthetic_alignment"
        ),
        source_confidence_threshold_profile_id=(
            "confidence_threshold_routing::style_aesthetic_alignment"
        ),
        source_escalation_signal_ids=(
            "cost_escalation_signal",
            "quality_escalation_signal",
        ),
        cost_band="high",
        advisory_cost_range=(4, 7),
        cost_pressure_signal="style_variant_budget_context",
        advisory_outputs=(
            "style_cost_threshold_placeholder",
            "aesthetic_cost_routing_context",
        ),
    ),
    _cost_threshold_routing_profile(
        cost_threshold_profile_id="cost_threshold_routing::curation_refinement_need",
        topic_id="curation_refinement_need",
        source_budget_profile_id=(
            "creative_exploration_budget::curation_refinement_need"
        ),
        source_confidence_threshold_profile_id=(
            "confidence_threshold_routing::curation_refinement_need"
        ),
        source_escalation_signal_ids=(
            "cost_escalation_signal",
            "confidence_escalation_signal",
            "quality_escalation_signal",
        ),
        cost_band="guarded",
        advisory_cost_range=(3, 5),
        cost_pressure_signal="refinement_budget_context",
        advisory_outputs=(
            "curation_cost_threshold_placeholder",
            "refinement_cost_routing_context",
        ),
    ),
    _cost_threshold_routing_profile(
        cost_threshold_profile_id="cost_threshold_routing::final_synthesis_readiness",
        topic_id="final_synthesis_readiness",
        source_budget_profile_id=(
            "creative_exploration_budget::final_synthesis_readiness"
        ),
        source_confidence_threshold_profile_id=(
            "confidence_threshold_routing::final_synthesis_readiness"
        ),
        source_escalation_signal_ids=(
            "cost_escalation_signal",
            "hitl_escalation_signal",
        ),
        cost_band="low",
        advisory_cost_range=(0, 2),
        cost_pressure_signal="final_synthesis_budget_context",
        advisory_outputs=(
            "synthesis_cost_threshold_placeholder",
            "final_cost_routing_context",
        ),
    ),
)
COST_THRESHOLD_ROUTING_REGISTRY = CostThresholdRoutingRegistry(
    cost_threshold_profiles=COST_THRESHOLD_ROUTING_PROFILES,
    cost_threshold_profile_ids=tuple(
        profile.cost_threshold_profile_id for profile in COST_THRESHOLD_ROUTING_PROFILES
    ),
    topic_ids=tuple(profile.topic_id for profile in COST_THRESHOLD_ROUTING_PROFILES),
    cost_bands=tuple(profile.cost_band for profile in COST_THRESHOLD_ROUTING_PROFILES),
    source_registries=_COST_THRESHOLD_ROUTING_SOURCE_REGISTRIES,
    budget_profile_ids=CREATIVE_EXPLORATION_BUDGET_REGISTRY.budget_profile_ids,
    confidence_threshold_profile_ids=(
        CONFIDENCE_THRESHOLD_ROUTING_REGISTRY.threshold_profile_ids
    ),
    escalation_signal_ids=_KNOWN_CONDITIONAL_ESCALATION_SIGNAL_IDS,
    profile_count=len(COST_THRESHOLD_ROUTING_PROFILES),
)
LATENCY_THRESHOLD_ROUTING_PROFILES = (
    _latency_threshold_routing_profile(
        latency_threshold_profile_id=(
            "latency_threshold_routing::planning_execution_fit"
        ),
        topic_id="planning_execution_fit",
        source_cost_threshold_profile_id=(
            "cost_threshold_routing::planning_execution_fit"
        ),
        source_escalation_signal_ids=(
            "latency_escalation_signal",
            "ambiguity_escalation_signal",
        ),
        latency_band="medium",
        advisory_latency_range_ms=(500, 1500),
        advisory_outputs=(
            "planning_latency_threshold_placeholder",
            "planning_latency_routing_context",
        ),
    ),
    _latency_threshold_routing_profile(
        latency_threshold_profile_id=(
            "latency_threshold_routing::style_aesthetic_alignment"
        ),
        topic_id="style_aesthetic_alignment",
        source_cost_threshold_profile_id=(
            "cost_threshold_routing::style_aesthetic_alignment"
        ),
        source_escalation_signal_ids=(
            "latency_escalation_signal",
            "quality_escalation_signal",
        ),
        latency_band="high",
        advisory_latency_range_ms=(1500, 3000),
        advisory_outputs=(
            "style_latency_threshold_placeholder",
            "aesthetic_latency_routing_context",
        ),
    ),
    _latency_threshold_routing_profile(
        latency_threshold_profile_id=(
            "latency_threshold_routing::curation_refinement_need"
        ),
        topic_id="curation_refinement_need",
        source_cost_threshold_profile_id=(
            "cost_threshold_routing::curation_refinement_need"
        ),
        source_escalation_signal_ids=(
            "latency_escalation_signal",
            "confidence_escalation_signal",
            "quality_escalation_signal",
        ),
        latency_band="guarded",
        advisory_latency_range_ms=(1000, 2500),
        advisory_outputs=(
            "curation_latency_threshold_placeholder",
            "refinement_latency_routing_context",
        ),
    ),
    _latency_threshold_routing_profile(
        latency_threshold_profile_id=(
            "latency_threshold_routing::final_synthesis_readiness"
        ),
        topic_id="final_synthesis_readiness",
        source_cost_threshold_profile_id=(
            "cost_threshold_routing::final_synthesis_readiness"
        ),
        source_escalation_signal_ids=(
            "latency_escalation_signal",
            "hitl_escalation_signal",
        ),
        latency_band="low",
        advisory_latency_range_ms=(0, 500),
        advisory_outputs=(
            "synthesis_latency_threshold_placeholder",
            "final_latency_routing_context",
        ),
    ),
)
LATENCY_THRESHOLD_ROUTING_REGISTRY = LatencyThresholdRoutingRegistry(
    latency_threshold_profiles=LATENCY_THRESHOLD_ROUTING_PROFILES,
    latency_threshold_profile_ids=tuple(
        profile.latency_threshold_profile_id
        for profile in LATENCY_THRESHOLD_ROUTING_PROFILES
    ),
    topic_ids=tuple(
        profile.topic_id for profile in LATENCY_THRESHOLD_ROUTING_PROFILES
    ),
    latency_bands=tuple(
        profile.latency_band for profile in LATENCY_THRESHOLD_ROUTING_PROFILES
    ),
    source_registries=_LATENCY_THRESHOLD_ROUTING_SOURCE_REGISTRIES,
    cost_threshold_profile_ids=COST_THRESHOLD_ROUTING_REGISTRY.cost_threshold_profile_ids,
    escalation_signal_ids=_KNOWN_CONDITIONAL_ESCALATION_SIGNAL_IDS,
    latency_metadata_sources=_LATENCY_METADATA_SOURCE_SURFACES,
    profile_count=len(LATENCY_THRESHOLD_ROUTING_PROFILES),
)
AMBIGUITY_ESCALATION_PROFILES = (
    _ambiguity_escalation_profile(
        ambiguity_profile_id="ambiguity_escalation::planning_execution_fit",
        topic_id="planning_execution_fit",
        source_latency_threshold_profile_id=(
            "latency_threshold_routing::planning_execution_fit"
        ),
        source_condition_ids=("planning_ambiguity_multi_agent_candidate",),
        source_policy_rule_ids=("missing_information_review",),
        source_escalation_signal_ids=(
            "ambiguity_escalation_signal",
            "hitl_escalation_signal",
        ),
        ambiguity_level="high",
        ambiguity_evidence_surfaces=(
            "missing_information",
            "planning_gap_summary",
        ),
        advisory_outputs=(
            "planning_ambiguity_escalation_placeholder",
            "planning_ambiguity_context",
        ),
    ),
    _ambiguity_escalation_profile(
        ambiguity_profile_id="ambiguity_escalation::style_aesthetic_alignment",
        topic_id="style_aesthetic_alignment",
        source_latency_threshold_profile_id=(
            "latency_threshold_routing::style_aesthetic_alignment"
        ),
        source_condition_ids=(
            "planning_ambiguity_multi_agent_candidate",
            "artifact_risk_multi_agent_candidate",
        ),
        source_policy_rule_ids=(
            "missing_information_review",
            "artifact_risk_review",
        ),
        source_escalation_signal_ids=(
            "ambiguity_escalation_signal",
            "quality_escalation_signal",
        ),
        ambiguity_level="medium",
        ambiguity_evidence_surfaces=(
            "missing_information",
            "disagreement_points",
        ),
        advisory_outputs=(
            "style_ambiguity_escalation_placeholder",
            "aesthetic_ambiguity_context",
        ),
    ),
    _ambiguity_escalation_profile(
        ambiguity_profile_id="ambiguity_escalation::curation_refinement_need",
        topic_id="curation_refinement_need",
        source_latency_threshold_profile_id=(
            "latency_threshold_routing::curation_refinement_need"
        ),
        source_condition_ids=(
            "planning_ambiguity_multi_agent_candidate",
            "evaluation_confidence_multi_agent_candidate",
        ),
        source_policy_rule_ids=(
            "missing_information_review",
            "evaluation_confidence_review",
        ),
        source_escalation_signal_ids=(
            "ambiguity_escalation_signal",
            "confidence_escalation_signal",
            "quality_escalation_signal",
        ),
        ambiguity_level="critical",
        ambiguity_evidence_surfaces=(
            "missing_information",
            "disagreement_points",
            "hitl_questions",
        ),
        advisory_outputs=(
            "curation_ambiguity_escalation_placeholder",
            "refinement_ambiguity_context",
        ),
    ),
    _ambiguity_escalation_profile(
        ambiguity_profile_id="ambiguity_escalation::final_synthesis_readiness",
        topic_id="final_synthesis_readiness",
        source_latency_threshold_profile_id=(
            "latency_threshold_routing::final_synthesis_readiness"
        ),
        source_condition_ids=(
            "terminal_guardrail_multi_agent_candidate",
        ),
        source_policy_rule_ids=(
            "missing_information_review",
            "future_agent_escalation_readiness",
        ),
        source_escalation_signal_ids=(
            "ambiguity_escalation_signal",
            "hitl_escalation_signal",
        ),
        ambiguity_level="low",
        ambiguity_evidence_surfaces=(
            "missing_information",
            "hitl_questions",
        ),
        advisory_outputs=(
            "synthesis_ambiguity_escalation_placeholder",
            "final_ambiguity_context",
        ),
    ),
)
AMBIGUITY_ESCALATION_REGISTRY = AmbiguityEscalationRegistry(
    ambiguity_profiles=AMBIGUITY_ESCALATION_PROFILES,
    ambiguity_profile_ids=tuple(
        profile.ambiguity_profile_id for profile in AMBIGUITY_ESCALATION_PROFILES
    ),
    topic_ids=tuple(profile.topic_id for profile in AMBIGUITY_ESCALATION_PROFILES),
    ambiguity_levels=tuple(
        profile.ambiguity_level for profile in AMBIGUITY_ESCALATION_PROFILES
    ),
    source_registries=_AMBIGUITY_ESCALATION_SOURCE_REGISTRIES,
    latency_threshold_profile_ids=(
        LATENCY_THRESHOLD_ROUTING_REGISTRY.latency_threshold_profile_ids
    ),
    condition_ids=CONDITIONAL_MULTI_AGENT_ESCALATION_REGISTRY.condition_ids,
    policy_rule_ids=_KNOWN_CONDITIONAL_ESCALATION_POLICY_RULE_IDS,
    escalation_signal_ids=_KNOWN_CONDITIONAL_ESCALATION_SIGNAL_IDS,
    ambiguity_evidence_surfaces=_AMBIGUITY_EVIDENCE_SURFACES,
    profile_count=len(AMBIGUITY_ESCALATION_PROFILES),
)
RISK_ESCALATION_PROFILES = (
    _risk_escalation_profile(
        risk_profile_id="risk_escalation::planning_execution_fit",
        topic_id="planning_execution_fit",
        source_ambiguity_profile_id="ambiguity_escalation::planning_execution_fit",
        source_condition_ids=(
            "planning_ambiguity_multi_agent_candidate",
            "artifact_risk_multi_agent_candidate",
        ),
        source_policy_rule_ids=(
            "missing_information_review",
            "artifact_risk_review",
        ),
        source_creative_policy_ids=(
            "concept_ambiguity_creative_escalation_policy",
            "aesthetic_risk_creative_escalation_policy",
        ),
        source_escalation_signal_ids=(
            "risk_escalation_signal",
            "ambiguity_escalation_signal",
        ),
        risk_level="medium",
        risk_evidence_surfaces=(
            "implementation_risks",
            "risk_assessment",
        ),
        advisory_outputs=(
            "planning_risk_escalation_placeholder",
            "planning_risk_context",
        ),
    ),
    _risk_escalation_profile(
        risk_profile_id="risk_escalation::style_aesthetic_alignment",
        topic_id="style_aesthetic_alignment",
        source_ambiguity_profile_id=(
            "ambiguity_escalation::style_aesthetic_alignment"
        ),
        source_condition_ids=("artifact_risk_multi_agent_candidate",),
        source_policy_rule_ids=("artifact_risk_review",),
        source_creative_policy_ids=("aesthetic_risk_creative_escalation_policy",),
        source_escalation_signal_ids=(
            "risk_escalation_signal",
            "quality_escalation_signal",
        ),
        risk_level="high",
        risk_evidence_surfaces=(
            "risk_assessment",
            "unsupported_assumptions",
            "capability_risks",
        ),
        advisory_outputs=(
            "style_risk_escalation_placeholder",
            "aesthetic_risk_context",
        ),
    ),
    _risk_escalation_profile(
        risk_profile_id="risk_escalation::curation_refinement_need",
        topic_id="curation_refinement_need",
        source_ambiguity_profile_id="ambiguity_escalation::curation_refinement_need",
        source_condition_ids=(
            "artifact_risk_multi_agent_candidate",
            "evaluation_confidence_multi_agent_candidate",
        ),
        source_policy_rule_ids=(
            "artifact_risk_review",
            "evaluation_confidence_review",
        ),
        source_creative_policy_ids=(
            "aesthetic_risk_creative_escalation_policy",
            "quality_uncertainty_creative_escalation_policy",
        ),
        source_escalation_signal_ids=(
            "risk_escalation_signal",
            "confidence_escalation_signal",
            "quality_escalation_signal",
        ),
        risk_level="critical",
        risk_evidence_surfaces=(
            "risk_assessment",
            "unsupported_assumptions",
            "escalation_candidates",
        ),
        advisory_outputs=(
            "curation_risk_escalation_placeholder",
            "refinement_risk_context",
        ),
    ),
    _risk_escalation_profile(
        risk_profile_id="risk_escalation::final_synthesis_readiness",
        topic_id="final_synthesis_readiness",
        source_ambiguity_profile_id="ambiguity_escalation::final_synthesis_readiness",
        source_condition_ids=("terminal_guardrail_multi_agent_candidate",),
        source_policy_rule_ids=(
            "artifact_risk_review",
            "future_agent_escalation_readiness",
        ),
        source_creative_policy_ids=("terminal_synthesis_creative_escalation_policy",),
        source_escalation_signal_ids=(
            "risk_escalation_signal",
            "hitl_escalation_signal",
        ),
        risk_level="low",
        risk_evidence_surfaces=(
            "risk_assessment",
            "escalation_candidates",
        ),
        advisory_outputs=(
            "synthesis_risk_escalation_placeholder",
            "final_risk_context",
        ),
    ),
)
RISK_ESCALATION_REGISTRY = RiskEscalationRegistry(
    risk_profiles=RISK_ESCALATION_PROFILES,
    risk_profile_ids=tuple(
        profile.risk_profile_id for profile in RISK_ESCALATION_PROFILES
    ),
    topic_ids=tuple(profile.topic_id for profile in RISK_ESCALATION_PROFILES),
    risk_levels=tuple(profile.risk_level for profile in RISK_ESCALATION_PROFILES),
    source_registries=_RISK_ESCALATION_SOURCE_REGISTRIES,
    ambiguity_profile_ids=AMBIGUITY_ESCALATION_REGISTRY.ambiguity_profile_ids,
    condition_ids=CONDITIONAL_MULTI_AGENT_ESCALATION_REGISTRY.condition_ids,
    policy_rule_ids=_KNOWN_CONDITIONAL_ESCALATION_POLICY_RULE_IDS,
    creative_policy_ids=CREATIVE_ESCALATION_POLICY_REGISTRY.policy_ids,
    escalation_signal_ids=_KNOWN_CONDITIONAL_ESCALATION_SIGNAL_IDS,
    risk_evidence_surfaces=_RISK_EVIDENCE_SURFACES,
    profile_count=len(RISK_ESCALATION_PROFILES),
)
QUALITY_ESCALATION_PROFILES = (
    _quality_escalation_profile(
        quality_profile_id="quality_escalation::planning_execution_fit",
        topic_id="planning_execution_fit",
        source_risk_profile_id="risk_escalation::planning_execution_fit",
        source_confidence_fusion_profile_id=(
            "agent_confidence_fusion::planning_execution_fit"
        ),
        source_creative_policy_ids=(
            "concept_ambiguity_creative_escalation_policy",
            "quality_uncertainty_creative_escalation_policy",
        ),
        source_escalation_signal_ids=(
            "quality_escalation_signal",
            "ambiguity_escalation_signal",
        ),
        quality_level="medium",
        quality_evidence_surfaces=(
            "quality_signal_metadata",
            "confidence_uncertainties",
        ),
        advisory_outputs=(
            "planning_quality_escalation_placeholder",
            "planning_quality_context",
        ),
    ),
    _quality_escalation_profile(
        quality_profile_id="quality_escalation::style_aesthetic_alignment",
        topic_id="style_aesthetic_alignment",
        source_risk_profile_id="risk_escalation::style_aesthetic_alignment",
        source_confidence_fusion_profile_id=(
            "agent_confidence_fusion::style_aesthetic_alignment"
        ),
        source_creative_policy_ids=(
            "aesthetic_risk_creative_escalation_policy",
            "quality_uncertainty_creative_escalation_policy",
        ),
        source_escalation_signal_ids=(
            "quality_escalation_signal",
            "risk_escalation_signal",
        ),
        quality_level="high",
        quality_evidence_surfaces=(
            "quality_review_signals",
            "weakest_quality_signals",
            "quality_risks",
        ),
        advisory_outputs=(
            "style_quality_escalation_placeholder",
            "aesthetic_quality_context",
        ),
    ),
    _quality_escalation_profile(
        quality_profile_id="quality_escalation::curation_refinement_need",
        topic_id="curation_refinement_need",
        source_risk_profile_id="risk_escalation::curation_refinement_need",
        source_confidence_fusion_profile_id=(
            "agent_confidence_fusion::curation_refinement_need"
        ),
        source_creative_policy_ids=(
            "quality_uncertainty_creative_escalation_policy",
            "terminal_synthesis_creative_escalation_policy",
        ),
        source_escalation_signal_ids=(
            "quality_escalation_signal",
            "confidence_escalation_signal",
            "risk_escalation_signal",
        ),
        quality_level="critical",
        quality_evidence_surfaces=(
            "quality_review_signals",
            "weakest_quality_signals",
            "quality_risks",
            "confidence_uncertainties",
        ),
        advisory_outputs=(
            "curation_quality_escalation_placeholder",
            "refinement_quality_context",
        ),
    ),
    _quality_escalation_profile(
        quality_profile_id="quality_escalation::final_synthesis_readiness",
        topic_id="final_synthesis_readiness",
        source_risk_profile_id="risk_escalation::final_synthesis_readiness",
        source_confidence_fusion_profile_id=(
            "agent_confidence_fusion::final_synthesis_readiness"
        ),
        source_creative_policy_ids=("terminal_synthesis_creative_escalation_policy",),
        source_escalation_signal_ids=(
            "quality_escalation_signal",
            "hitl_escalation_signal",
        ),
        quality_level="low",
        quality_evidence_surfaces=(
            "quality_signal_metadata",
            "quality_review_signals",
        ),
        advisory_outputs=(
            "synthesis_quality_escalation_placeholder",
            "final_quality_context",
        ),
    ),
)
QUALITY_ESCALATION_REGISTRY = QualityEscalationRegistry(
    quality_profiles=QUALITY_ESCALATION_PROFILES,
    quality_profile_ids=tuple(
        profile.quality_profile_id for profile in QUALITY_ESCALATION_PROFILES
    ),
    topic_ids=tuple(profile.topic_id for profile in QUALITY_ESCALATION_PROFILES),
    quality_levels=tuple(
        profile.quality_level for profile in QUALITY_ESCALATION_PROFILES
    ),
    source_registries=_QUALITY_ESCALATION_SOURCE_REGISTRIES,
    risk_profile_ids=RISK_ESCALATION_REGISTRY.risk_profile_ids,
    confidence_fusion_profile_ids=AGENT_CONFIDENCE_FUSION_REGISTRY.fusion_profile_ids,
    creative_policy_ids=CREATIVE_ESCALATION_POLICY_REGISTRY.policy_ids,
    escalation_signal_ids=_KNOWN_CONDITIONAL_ESCALATION_SIGNAL_IDS,
    quality_evidence_surfaces=_QUALITY_EVIDENCE_SURFACES,
    profile_count=len(QUALITY_ESCALATION_PROFILES),
)
ADAPTIVE_MULTI_AGENT_ESCALATION_PROFILES = (
    _adaptive_multi_agent_escalation_profile(
        adaptive_profile_id="adaptive_multi_agent_escalation::planning_execution_fit",
        topic_id="planning_execution_fit",
        source_condition_ids=("planning_ambiguity_multi_agent_candidate",),
        source_specialist_loop_ids=("planning_specialist_agent_loop",),
        source_quality_profile_id="quality_escalation::planning_execution_fit",
        source_risk_profile_id="risk_escalation::planning_execution_fit",
        source_ambiguity_profile_id="ambiguity_escalation::planning_execution_fit",
        source_hitl_gate_profile_id="hitl_escalation_gate::planning_execution_fit",
        source_escalation_signal_ids=(
            "quality_escalation_signal",
            "ambiguity_escalation_signal",
            "hitl_escalation_signal",
        ),
        adaptive_posture="context_packet",
        adaptive_evidence_surfaces=(
            "escalation_candidates",
            "ambiguity_context",
            "human_review_posture",
        ),
        advisory_outputs=(
            "planning_adaptive_escalation_placeholder",
            "planning_adaptive_context_packet",
        ),
    ),
    _adaptive_multi_agent_escalation_profile(
        adaptive_profile_id=(
            "adaptive_multi_agent_escalation::style_aesthetic_alignment"
        ),
        topic_id="style_aesthetic_alignment",
        source_condition_ids=("artifact_risk_multi_agent_candidate",),
        source_specialist_loop_ids=("artifact_specialist_agent_loop",),
        source_quality_profile_id="quality_escalation::style_aesthetic_alignment",
        source_risk_profile_id="risk_escalation::style_aesthetic_alignment",
        source_ambiguity_profile_id=(
            "ambiguity_escalation::style_aesthetic_alignment"
        ),
        source_hitl_gate_profile_id=(
            "hitl_escalation_gate::style_aesthetic_alignment"
        ),
        source_escalation_signal_ids=(
            "quality_escalation_signal",
            "risk_escalation_signal",
        ),
        adaptive_posture="specialist_pairing_candidate",
        adaptive_evidence_surfaces=(
            "agent_escalation_candidates",
            "risk_context",
            "quality_uncertainty_summary",
        ),
        advisory_outputs=(
            "style_adaptive_escalation_placeholder",
            "artifact_specialist_pairing_context",
        ),
    ),
    _adaptive_multi_agent_escalation_profile(
        adaptive_profile_id="adaptive_multi_agent_escalation::curation_refinement_need",
        topic_id="curation_refinement_need",
        source_condition_ids=(
            "evaluation_confidence_multi_agent_candidate",
            "artifact_risk_multi_agent_candidate",
        ),
        source_specialist_loop_ids=(
            "evaluation_specialist_agent_loop",
            "artifact_specialist_agent_loop",
        ),
        source_quality_profile_id="quality_escalation::curation_refinement_need",
        source_risk_profile_id="risk_escalation::curation_refinement_need",
        source_ambiguity_profile_id="ambiguity_escalation::curation_refinement_need",
        source_hitl_gate_profile_id=(
            "hitl_escalation_gate::curation_refinement_need"
        ),
        source_escalation_signal_ids=(
            "quality_escalation_signal",
            "confidence_escalation_signal",
            "risk_escalation_signal",
            "hitl_escalation_signal",
        ),
        adaptive_posture="multi_loop_review_candidate",
        adaptive_evidence_surfaces=(
            "escalation_candidates",
            "agent_escalation_candidates",
            "quality_uncertainty_summary",
            "human_review_posture",
        ),
        advisory_outputs=(
            "curation_adaptive_escalation_placeholder",
            "multi_loop_review_context",
        ),
    ),
    _adaptive_multi_agent_escalation_profile(
        adaptive_profile_id=(
            "adaptive_multi_agent_escalation::final_synthesis_readiness"
        ),
        topic_id="final_synthesis_readiness",
        source_condition_ids=("terminal_guardrail_multi_agent_candidate",),
        source_specialist_loop_ids=("synthesis_specialist_agent_loop",),
        source_quality_profile_id="quality_escalation::final_synthesis_readiness",
        source_risk_profile_id="risk_escalation::final_synthesis_readiness",
        source_ambiguity_profile_id="ambiguity_escalation::final_synthesis_readiness",
        source_hitl_gate_profile_id=(
            "hitl_escalation_gate::final_synthesis_readiness"
        ),
        source_escalation_signal_ids=(
            "quality_escalation_signal",
            "hitl_escalation_signal",
            "risk_escalation_signal",
        ),
        adaptive_posture="terminal_guardrail_candidate",
        adaptive_evidence_surfaces=(
            "human_review_posture",
            "risk_context",
            "agent_escalation_candidates",
        ),
        advisory_outputs=(
            "synthesis_adaptive_escalation_placeholder",
            "terminal_guardrail_context",
        ),
    ),
)
ADAPTIVE_MULTI_AGENT_ESCALATION_REGISTRY = AdaptiveMultiAgentEscalationRegistry(
    adaptive_profiles=ADAPTIVE_MULTI_AGENT_ESCALATION_PROFILES,
    adaptive_profile_ids=tuple(
        profile.adaptive_profile_id
        for profile in ADAPTIVE_MULTI_AGENT_ESCALATION_PROFILES
    ),
    topic_ids=tuple(
        profile.topic_id for profile in ADAPTIVE_MULTI_AGENT_ESCALATION_PROFILES
    ),
    adaptive_postures=tuple(
        profile.adaptive_posture
        for profile in ADAPTIVE_MULTI_AGENT_ESCALATION_PROFILES
    ),
    source_registries=_ADAPTIVE_MULTI_AGENT_ESCALATION_SOURCE_REGISTRIES,
    capability_ids=_ADAPTIVE_MULTI_AGENT_ESCALATION_CAPABILITY_IDS,
    condition_ids=CONDITIONAL_MULTI_AGENT_ESCALATION_REGISTRY.condition_ids,
    specialist_loop_ids=SPECIALIST_AGENT_LOOP_REGISTRY.loop_ids,
    quality_profile_ids=QUALITY_ESCALATION_REGISTRY.quality_profile_ids,
    risk_profile_ids=RISK_ESCALATION_REGISTRY.risk_profile_ids,
    ambiguity_profile_ids=AMBIGUITY_ESCALATION_REGISTRY.ambiguity_profile_ids,
    hitl_gate_profile_ids=HITL_ESCALATION_GATE_REGISTRY.hitl_gate_profile_ids,
    escalation_signal_ids=_KNOWN_CONDITIONAL_ESCALATION_SIGNAL_IDS,
    adaptive_evidence_surfaces=_ADAPTIVE_MULTI_AGENT_ESCALATION_EVIDENCE_SURFACES,
    profile_count=len(ADAPTIVE_MULTI_AGENT_ESCALATION_PROFILES),
)

HYBRID_AGENTIC_WORKFLOW_STAGES = (
    _stage(
        stage_id="intake_routing_context_readiness",
        stage_name="Intake Routing Context Readiness",
        v3_workflow_nodes=(
            "intake",
            "routing",
            "memory",
            "retrieval",
            "context_assembly",
        ),
        future_capability_ids=("v4_agent_router",),
        escalation_rule_ids=("missing_information_review",),
        advisory_outputs=(
            "routing_context_packet",
            "retrieval_gap_summary",
            "context_handoff_notes",
        ),
    ),
    _stage(
        stage_id="planning_reasoning_readiness",
        stage_name="Planning Reasoning Readiness",
        v3_workflow_nodes=(
            "prompt_input",
            "planning",
            "director",
            "reasoning",
            "prompt_rendering",
        ),
        future_capability_ids=("v4_planner_agent", "v4_agentic_studio"),
        escalation_rule_ids=(
            "missing_information_review",
            "evaluation_confidence_review",
        ),
        advisory_outputs=(
            "planning_context_packet",
            "reasoning_review_notes",
            "prompt_handoff_summary",
        ),
    ),
    _stage(
        stage_id="generation_artifact_readiness",
        stage_name="Generation Artifact Readiness",
        v3_workflow_nodes=(
            "generation",
            "artifact_extraction",
            "preview_preparation",
            "artifact_critique",
        ),
        future_capability_ids=("v4_artifact_agent", "v4_runtime_agent"),
        escalation_rule_ids=(
            "artifact_risk_review",
            "runtime_incompatibility_review",
        ),
        advisory_outputs=(
            "artifact_context_packet",
            "runtime_fit_notes",
            "preview_readiness_summary",
        ),
    ),
    _stage(
        stage_id="review_refinement_readiness",
        stage_name="Review Refinement Readiness",
        v3_workflow_nodes=("review", "refinement"),
        future_capability_ids=(
            "v4_agentic_studio",
            "adaptive_multi_agent_escalation",
        ),
        escalation_rule_ids=(
            "evaluation_confidence_review",
            "future_agent_escalation_readiness",
        ),
        advisory_outputs=(
            "review_context_packet",
            "refinement_candidate_summary",
            "human_review_posture",
        ),
    ),
    _stage(
        stage_id="completion_guardrail_readiness",
        stage_name="Completion Guardrail Readiness",
        v3_workflow_nodes=("finalization", "failure"),
        future_capability_ids=(
            "v4_agent_router",
            "adaptive_multi_agent_escalation",
        ),
        escalation_rule_ids=("future_agent_escalation_readiness",),
        advisory_outputs=(
            "completion_context_packet",
            "failure_guardrail_notes",
            "final_handoff_summary",
        ),
    ),
)

HYBRID_AGENTIC_WORKFLOW_REGISTRY = HybridAgenticWorkflowRegistry(
    stages=HYBRID_AGENTIC_WORKFLOW_STAGES,
    stage_ids=tuple(stage.stage_id for stage in HYBRID_AGENTIC_WORKFLOW_STAGES),
    stage_count=len(HYBRID_AGENTIC_WORKFLOW_STAGES),
    source_metadata_registries=(
        "agent_capability_registry",
        "escalation_policy_registry",
        "artifact_engine_contract_registry",
        "evaluation_engine_contract_registry",
        "workstation_engine_contract_registry",
    ),
)
