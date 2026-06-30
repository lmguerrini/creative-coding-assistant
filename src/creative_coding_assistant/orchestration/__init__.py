"""Assistant orchestration and explicit routing."""

from __future__ import annotations

from importlib import import_module

_CTX = "creative_coding_assistant.orchestration.context"
_CACHE_LAYER = "creative_coding_assistant.orchestration.cache_layer"
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
_CONTEXT_BUDGET_PLANNER = (
    "creative_coding_assistant.orchestration.context_budget_planner"
)
_CONTEXT_ROUTER = "creative_coding_assistant.orchestration.context_router"
_CONTEXT_REUSE = "creative_coding_assistant.orchestration.context_reuse"
_EXPLORATION_BUDGET_PLANNER = (
    "creative_coding_assistant.orchestration.exploration_budget_planner"
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
_CREATIVE_COMPLEXITY_ANALYZER = (
    "creative_coding_assistant.orchestration.creative_complexity_analyzer"
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
_MEMORY_SUMMARIZATION = (
    "creative_coding_assistant.orchestration.memory_summarization"
)
_PROMPT_INPUTS = "creative_coding_assistant.orchestration.prompt_inputs"
_PROMPT_COMPRESSION = "creative_coding_assistant.orchestration.prompt_compression"
_PROMPT_TEMPLATES = "creative_coding_assistant.orchestration.prompt_templates"
_QUALITY_CALIBRATION = (
    "creative_coding_assistant.orchestration.quality_calibration"
)
_REFINEMENT_PASSES = "creative_coding_assistant.orchestration.refinement_passes"
_REFERENCE_FUSION = "creative_coding_assistant.orchestration.reference_fusion"
_RETRIEVAL = "creative_coding_assistant.orchestration.retrieval"
_RETRIEVAL_COMPRESSION = (
    "creative_coding_assistant.orchestration.retrieval_compression"
)
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
_WORKFLOW_COMPLEXITY_ANALYZER = (
    "creative_coding_assistant.orchestration.workflow_complexity_analyzer"
)
_WORKFLOW_PRUNING = "creative_coding_assistant.orchestration.workflow_pruning"
_EXECUTION_COST_FORECASTING = (
    "creative_coding_assistant.orchestration.execution_cost_forecasting"
)
_EXECUTION_PATH_OPTIMIZATION = (
    "creative_coding_assistant.orchestration.execution_path_optimization"
)
_EXECUTION_STRATEGY_SELECTION = (
    "creative_coding_assistant.orchestration.execution_strategy_selection"
)
_PARALLEL_SCHEDULER = (
    "creative_coding_assistant.orchestration.parallel_scheduler"
)
_LATENCY_OPTIMIZER = (
    "creative_coding_assistant.orchestration.latency_optimizer"
)
_ASYNC_EXECUTION = (
    "creative_coding_assistant.orchestration.async_execution"
)
_STREAMING_OPTIMIZER = (
    "creative_coding_assistant.orchestration.streaming_optimizer"
)
_RETRY_POLICIES = (
    "creative_coding_assistant.orchestration.retry_policies"
)
_LOAD_BALANCER = (
    "creative_coding_assistant.orchestration.load_balancer"
)
_EXECUTION_PROFILING = (
    "creative_coding_assistant.orchestration.execution_profiling"
)
_WORKFLOW_REPLAY_ENGINE = (
    "creative_coding_assistant.orchestration.workflow_replay_engine"
)
_EXECUTION_REPLAY_ENGINE = (
    "creative_coding_assistant.orchestration.execution_replay_engine"
)
_BOTTLENECK_DETECTION = (
    "creative_coding_assistant.orchestration.bottleneck_detection"
)
_THROUGHPUT_OPTIMIZER = (
    "creative_coding_assistant.orchestration.throughput_optimizer"
)
_PERFORMANCE_PREDICTION = (
    "creative_coding_assistant.orchestration.performance_prediction"
)
_PERFORMANCE_BENCHMARKING = (
    "creative_coding_assistant.orchestration.performance_benchmarking"
)
_REASONING_BUDGET_OPTIMIZER = (
    "creative_coding_assistant.orchestration.reasoning_budget_optimizer"
)
_PERFORMANCE_REGRESSION_DETECTION = (
    "creative_coding_assistant.orchestration.performance_regression_detection"
)
_RESOURCE_UTILIZATION_OPTIMIZER = (
    "creative_coding_assistant.orchestration.resource_utilization_optimizer"
)
_PERFORMANCE_ARCHITECTURE_CONSISTENCY = (
    "creative_coding_assistant.orchestration.performance_architecture_consistency"
)
_PERFORMANCE_FAILURE_PATH_AUDIT = (
    "creative_coding_assistant.orchestration.performance_failure_path_audit"
)
_TOKEN_DASHBOARD = "creative_coding_assistant.orchestration.token_dashboard"
_COST_DASHBOARD = "creative_coding_assistant.orchestration.cost_dashboard"
_QUALITY_DASHBOARD = "creative_coding_assistant.orchestration.quality_dashboard"
_PERFORMANCE_DASHBOARD = (
    "creative_coding_assistant.orchestration.performance_dashboard"
)
_PRODUCTION_TELEMETRY = (
    "creative_coding_assistant.orchestration.production_telemetry"
)
_WORKFLOW_DIAGNOSTICS = (
    "creative_coding_assistant.orchestration.workflow_diagnostics"
)
_AGENT_DIAGNOSTICS = (
    "creative_coding_assistant.orchestration.agent_diagnostics"
)
_ROUTING_DIAGNOSTICS = (
    "creative_coding_assistant.orchestration.routing_diagnostics"
)
_ESCALATION_DIAGNOSTICS = (
    "creative_coding_assistant.orchestration.escalation_diagnostics"
)
_FAILURE_ANALYSIS = "creative_coding_assistant.orchestration.failure_analysis"
_ERROR_INTELLIGENCE = "creative_coding_assistant.orchestration.error_intelligence"
_WORKFLOW_HEALTH_MONITORING = (
    "creative_coding_assistant.orchestration.workflow_health_monitoring"
)
_SYSTEM_HEALTH_MONITORING = (
    "creative_coding_assistant.orchestration.system_health_monitoring"
)
_CREATIVE_ANALYTICS = "creative_coding_assistant.orchestration.creative_analytics"
_CONFIDENCE_ANALYTICS = (
    "creative_coding_assistant.orchestration.confidence_analytics"
)
_CREATIVE_DIVERSITY_ANALYTICS = (
    "creative_coding_assistant.orchestration.creative_diversity_analytics"
)
_RUNTIME_TIMELINE = "creative_coding_assistant.orchestration.runtime_timeline"
_WORKFLOW_EXPLAINABILITY_DASHBOARD = (
    "creative_coding_assistant.orchestration.workflow_explainability_dashboard"
)
_PRODUCTION_OBSERVABILITY_ARCHITECTURE_CONSISTENCY = (
    "creative_coding_assistant.orchestration."
    "production_observability_architecture_consistency"
)
_PRODUCTION_OBSERVABILITY_FAILURE_PATH_AUDIT = (
    "creative_coding_assistant.orchestration."
    "production_observability_failure_path_audit"
)
_EXECUTION_OPTIMIZATION_FAILURE_AUDIT = (
    "creative_coding_assistant.orchestration.execution_optimization_failure_audit"
)
_MODEL_ROUTER = "creative_coding_assistant.orchestration.model_router"
_LOCAL_CLOUD_ROUTING = (
    "creative_coding_assistant.orchestration.local_cloud_routing"
)
_HYBRID_ROUTING = "creative_coding_assistant.orchestration.hybrid_routing"
_QUALITY_COST_OPTIMIZER = (
    "creative_coding_assistant.orchestration.quality_cost_optimizer"
)
_COST_ESTIMATOR = "creative_coding_assistant.orchestration.cost_estimator"
_BUDGET_POLICIES = "creative_coding_assistant.orchestration.budget_policies"
_HITL_BUDGET_GATE = "creative_coding_assistant.orchestration.hitl_budget_gate"
_RUNTIME_RECOMMENDATION_ENGINE = (
    "creative_coding_assistant.orchestration.runtime_recommendation_engine"
)
_EXECUTION_POLICY_ENGINE = (
    "creative_coding_assistant.orchestration.execution_policy_engine"
)
_MODEL_RECOMMENDATION_ENGINE = (
    "creative_coding_assistant.orchestration.model_recommendation_engine"
)
_MODEL_CAPABILITY_MATRIX = (
    "creative_coding_assistant.orchestration.model_capability_matrix"
)
_PROVIDER_CAPABILITY_MATRIX = (
    "creative_coding_assistant.orchestration.provider_capability_matrix"
)
_QUALITY_PREDICTION_ENGINE = (
    "creative_coding_assistant.orchestration.quality_prediction_engine"
)
_COST_PREDICTION_ENGINE = (
    "creative_coding_assistant.orchestration.cost_prediction_engine"
)
_CREATIVE_DIVERSITY_PREDICTOR = (
    "creative_coding_assistant.orchestration.creative_diversity_predictor"
)
_CREATIVE_CONSISTENCY_PREDICTOR = (
    "creative_coding_assistant.orchestration.creative_consistency_predictor"
)
_ROUTING_EXPLAINABILITY = (
    "creative_coding_assistant.orchestration.routing_explainability"
)
_MODEL_ROUTING_ARCHITECTURE_CONSISTENCY = (
    "creative_coding_assistant.orchestration."
    "model_routing_architecture_consistency"
)
_MODEL_ROUTING_FAILURE_PATH_AUDIT = (
    "creative_coding_assistant.orchestration.model_routing_failure_path_audit"
)
_ROUTING_INTELLIGENCE = (
    "creative_coding_assistant.orchestration.routing_intelligence"
)
_ADAPTIVE_HYBRID_WORKFLOW_OPTIMIZER = (
    "creative_coding_assistant.orchestration."
    "adaptive_hybrid_workflow_optimizer"
)
_ADAPTIVE_ESCALATION_OPTIMIZER = (
    "creative_coding_assistant.orchestration.adaptive_escalation_optimizer"
)
_AGENT_ACTIVATION_OPTIMIZER = (
    "creative_coding_assistant.orchestration.agent_activation_optimizer"
)
_ADAPTIVE_COST_QUALITY_OPTIMIZER = (
    "creative_coding_assistant.orchestration.adaptive_cost_quality_optimizer"
)
_ADAPTIVE_LATENCY_OPTIMIZER = (
    "creative_coding_assistant.orchestration.adaptive_latency_optimizer"
)
_ADAPTIVE_EXECUTION_STRATEGY_SELECTION = (
    "creative_coding_assistant.orchestration."
    "adaptive_execution_strategy_selection"
)
_DYNAMIC_AGENT_ALLOCATION = (
    "creative_coding_assistant.orchestration.dynamic_agent_allocation"
)
_AGENT_DIVERSITY_OPTIMIZER = (
    "creative_coding_assistant.orchestration.agent_diversity_optimizer"
)
_DYNAMIC_RESOURCE_ALLOCATION = (
    "creative_coding_assistant.orchestration.dynamic_resource_allocation"
)
_WORKFLOW_SELF_TUNING_POLICIES = (
    "creative_coding_assistant.orchestration.workflow_self_tuning_policies"
)
_EXECUTION_CONFIDENCE_ENGINE = (
    "creative_coding_assistant.orchestration.execution_confidence_engine"
)
_WORKFLOW_RISK_ENGINE = (
    "creative_coding_assistant.orchestration.workflow_risk_engine"
)
_CREATIVE_EXPLORATION_OPTIMIZER = (
    "creative_coding_assistant.orchestration.creative_exploration_optimizer"
)
_EMERGENCE_OPTIMIZER = (
    "creative_coding_assistant.orchestration.emergence_optimizer"
)

_EXPORT_MAP = {
    "ASSISTANT_WORKFLOW_NODE_ORDER": _WORKFLOW_GRAPH,
    "ASSISTANT_WORKFLOW_RECURSION_LIMIT": _WORKFLOW_GRAPH,
    "AssistantWorkflowGraphState": _WORKFLOW_GRAPH,
    "AssistantWorkflowRuntime": _WORKFLOW_GRAPH,
    "AssistantWorkflowState": _WORKFLOW,
    "ExecutionCacheEntry": _CACHE_LAYER,
    "ExecutionCacheLookup": _CACHE_LAYER,
    "ExecutionGraphAnalysis": _EXECUTION_GRAPH_ANALYZER,
    "ExecutionGraphEdge": _EXECUTION_GRAPH_ANALYZER,
    "ExecutionGraphNode": _EXECUTION_GRAPH_ANALYZER,
    "CreativeComplexityAnalysis": _CREATIVE_COMPLEXITY_ANALYZER,
    "CreativeComplexityFactor": _CREATIVE_COMPLEXITY_ANALYZER,
    "ContextBudgetAllocation": _CONTEXT_BUDGET_PLANNER,
    "ContextBudgetPlan": _CONTEXT_BUDGET_PLANNER,
    "ContextRouteDecision": _CONTEXT_ROUTER,
    "ContextRoutingPlan": _CONTEXT_ROUTER,
    "ContextReuseCandidate": _CONTEXT_REUSE,
    "ContextReusePlan": _CONTEXT_REUSE,
    "ExplorationBudgetAllocation": _EXPLORATION_BUDGET_PLANNER,
    "ExplorationBudgetPlan": _EXPLORATION_BUDGET_PLANNER,
    "WorkflowCostAnalysis": _WORKFLOW_COST_ANALYZER,
    "WorkflowCostComponent": _WORKFLOW_COST_ANALYZER,
    "WorkflowComplexityAnalysis": _WORKFLOW_COMPLEXITY_ANALYZER,
    "WorkflowComplexityFactor": _WORKFLOW_COMPLEXITY_ANALYZER,
    "WorkflowPruningCandidate": _WORKFLOW_PRUNING,
    "WorkflowPruningPlan": _WORKFLOW_PRUNING,
    "ExecutionCostForecast": _EXECUTION_COST_FORECASTING,
    "ExecutionCostForecastScenario": _EXECUTION_COST_FORECASTING,
    "ExecutionPathCandidate": _EXECUTION_PATH_OPTIMIZATION,
    "ExecutionPathOptimizationPlan": _EXECUTION_PATH_OPTIMIZATION,
    "ExecutionStrategyCandidate": _EXECUTION_STRATEGY_SELECTION,
    "ExecutionStrategySelection": _EXECUTION_STRATEGY_SELECTION,
    "ParallelScheduleCandidate": _PARALLEL_SCHEDULER,
    "ParallelSchedulerPlan": _PARALLEL_SCHEDULER,
    "LatencyOptimizationCandidate": _LATENCY_OPTIMIZER,
    "LatencyOptimizationPlan": _LATENCY_OPTIMIZER,
    "AsyncExecutionCandidate": _ASYNC_EXECUTION,
    "AsyncExecutionPlan": _ASYNC_EXECUTION,
    "StreamingOptimizationCandidate": _STREAMING_OPTIMIZER,
    "StreamingOptimizationPlan": _STREAMING_OPTIMIZER,
    "RetryPolicyCandidate": _RETRY_POLICIES,
    "RetryPolicyPlan": _RETRY_POLICIES,
    "LoadBalanceCandidate": _LOAD_BALANCER,
    "LoadBalancerPlan": _LOAD_BALANCER,
    "ExecutionProfileCandidate": _EXECUTION_PROFILING,
    "ExecutionProfilingPlan": _EXECUTION_PROFILING,
    "WorkflowReplayCandidate": _WORKFLOW_REPLAY_ENGINE,
    "WorkflowReplayPlan": _WORKFLOW_REPLAY_ENGINE,
    "ExecutionReplayCandidate": _EXECUTION_REPLAY_ENGINE,
    "ExecutionReplayPlan": _EXECUTION_REPLAY_ENGINE,
    "BottleneckCandidate": _BOTTLENECK_DETECTION,
    "BottleneckDetectionPlan": _BOTTLENECK_DETECTION,
    "ThroughputOptimizationCandidate": _THROUGHPUT_OPTIMIZER,
    "ThroughputOptimizationPlan": _THROUGHPUT_OPTIMIZER,
    "PerformancePrediction": _PERFORMANCE_PREDICTION,
    "PerformancePredictionPlan": _PERFORMANCE_PREDICTION,
    "PerformanceBenchmarkScenario": _PERFORMANCE_BENCHMARKING,
    "PerformanceBenchmarkingPlan": _PERFORMANCE_BENCHMARKING,
    "ReasoningBudgetOptimizationPlan": _REASONING_BUDGET_OPTIMIZER,
    "ReasoningBudgetRecommendation": _REASONING_BUDGET_OPTIMIZER,
    "PerformanceRegressionDetectionPlan": _PERFORMANCE_REGRESSION_DETECTION,
    "PerformanceRegressionSignal": _PERFORMANCE_REGRESSION_DETECTION,
    "ResourceUtilizationOptimizationPlan": _RESOURCE_UTILIZATION_OPTIMIZER,
    "ResourceUtilizationRecommendation": _RESOURCE_UTILIZATION_OPTIMIZER,
    "PerformanceArchitectureConsistencyRecord": (
        _PERFORMANCE_ARCHITECTURE_CONSISTENCY
    ),
    "PerformanceArchitectureConsistencyRegistry": (
        _PERFORMANCE_ARCHITECTURE_CONSISTENCY
    ),
    "PerformanceFailurePathAuditRecord": _PERFORMANCE_FAILURE_PATH_AUDIT,
    "PerformanceFailurePathAuditRegistry": _PERFORMANCE_FAILURE_PATH_AUDIT,
    "TokenDashboard": _TOKEN_DASHBOARD,
    "TokenDashboardPanel": _TOKEN_DASHBOARD,
    "CostDashboard": _COST_DASHBOARD,
    "CostDashboardPanel": _COST_DASHBOARD,
    "QualityDashboard": _QUALITY_DASHBOARD,
    "QualityDashboardPanel": _QUALITY_DASHBOARD,
    "PerformanceDashboard": _PERFORMANCE_DASHBOARD,
    "PerformanceDashboardPanel": _PERFORMANCE_DASHBOARD,
    "ProductionTelemetryChannel": _PRODUCTION_TELEMETRY,
    "ProductionTelemetrySurface": _PRODUCTION_TELEMETRY,
    "WorkflowDiagnosticPanel": _WORKFLOW_DIAGNOSTICS,
    "WorkflowDiagnostics": _WORKFLOW_DIAGNOSTICS,
    "AgentDiagnosticPanel": _AGENT_DIAGNOSTICS,
    "AgentDiagnostics": _AGENT_DIAGNOSTICS,
    "RoutingDiagnosticPanel": _ROUTING_DIAGNOSTICS,
    "RoutingDiagnostics": _ROUTING_DIAGNOSTICS,
    "EscalationDiagnosticPanel": _ESCALATION_DIAGNOSTICS,
    "EscalationDiagnostics": _ESCALATION_DIAGNOSTICS,
    "FailureAnalysisPanel": _FAILURE_ANALYSIS,
    "FailureAnalysis": _FAILURE_ANALYSIS,
    "ErrorIntelligencePanel": _ERROR_INTELLIGENCE,
    "ErrorIntelligence": _ERROR_INTELLIGENCE,
    "WorkflowHealthPanel": _WORKFLOW_HEALTH_MONITORING,
    "WorkflowHealthMonitoring": _WORKFLOW_HEALTH_MONITORING,
    "SystemHealthPanel": _SYSTEM_HEALTH_MONITORING,
    "SystemHealthMonitoring": _SYSTEM_HEALTH_MONITORING,
    "CreativeAnalyticsPanel": _CREATIVE_ANALYTICS,
    "CreativeAnalytics": _CREATIVE_ANALYTICS,
    "ConfidenceAnalyticsPanel": _CONFIDENCE_ANALYTICS,
    "ConfidenceAnalytics": _CONFIDENCE_ANALYTICS,
    "CreativeDiversityAnalyticsPanel": _CREATIVE_DIVERSITY_ANALYTICS,
    "CreativeDiversityAnalytics": _CREATIVE_DIVERSITY_ANALYTICS,
    "RuntimeTimelinePanel": _RUNTIME_TIMELINE,
    "RuntimeTimeline": _RUNTIME_TIMELINE,
    "WorkflowExplainabilityPanel": _WORKFLOW_EXPLAINABILITY_DASHBOARD,
    "WorkflowExplainabilityDashboard": _WORKFLOW_EXPLAINABILITY_DASHBOARD,
    "ProductionObservabilityArchitectureRecord": (
        _PRODUCTION_OBSERVABILITY_ARCHITECTURE_CONSISTENCY
    ),
    "ProductionObservabilityArchitectureRegistry": (
        _PRODUCTION_OBSERVABILITY_ARCHITECTURE_CONSISTENCY
    ),
    "ProductionObservabilityFailurePathAuditRecord": (
        _PRODUCTION_OBSERVABILITY_FAILURE_PATH_AUDIT
    ),
    "ProductionObservabilityFailurePathAuditRegistry": (
        _PRODUCTION_OBSERVABILITY_FAILURE_PATH_AUDIT
    ),
    "ExecutionOptimizationFailureAuditRecord": (
        _EXECUTION_OPTIMIZATION_FAILURE_AUDIT
    ),
    "ExecutionOptimizationFailureAuditRegistry": (
        _EXECUTION_OPTIMIZATION_FAILURE_AUDIT
    ),
    "ModelRouteCandidate": _MODEL_ROUTER,
    "ModelRoutingPlan": _MODEL_ROUTER,
    "LocalCloudRouteDecision": _LOCAL_CLOUD_ROUTING,
    "LocalCloudRoutingPlan": _LOCAL_CLOUD_ROUTING,
    "HybridRouteDecision": _HYBRID_ROUTING,
    "HybridRoutingPlan": _HYBRID_ROUTING,
    "QualityCostOptimizationCandidate": _QUALITY_COST_OPTIMIZER,
    "QualityCostOptimizationPlan": _QUALITY_COST_OPTIMIZER,
    "CostEstimateScenario": _COST_ESTIMATOR,
    "CostEstimationPlan": _COST_ESTIMATOR,
    "BudgetPolicyDecision": _BUDGET_POLICIES,
    "BudgetPolicyPlan": _BUDGET_POLICIES,
    "HitlBudgetGateDecision": _HITL_BUDGET_GATE,
    "HitlBudgetGatePlan": _HITL_BUDGET_GATE,
    "RuntimeRecommendationDecision": _RUNTIME_RECOMMENDATION_ENGINE,
    "RuntimeRecommendationPlan": _RUNTIME_RECOMMENDATION_ENGINE,
    "ExecutionPolicyDecision": _EXECUTION_POLICY_ENGINE,
    "ExecutionPolicyPlan": _EXECUTION_POLICY_ENGINE,
    "ModelRecommendationDecision": _MODEL_RECOMMENDATION_ENGINE,
    "ModelRecommendationPlan": _MODEL_RECOMMENDATION_ENGINE,
    "ModelCapabilityMatrix": _MODEL_CAPABILITY_MATRIX,
    "ModelCapabilityMatrixRow": _MODEL_CAPABILITY_MATRIX,
    "ProviderCapabilityMatrix": _PROVIDER_CAPABILITY_MATRIX,
    "ProviderCapabilityMatrixRow": _PROVIDER_CAPABILITY_MATRIX,
    "QualityPredictionDecision": _QUALITY_PREDICTION_ENGINE,
    "QualityPredictionPlan": _QUALITY_PREDICTION_ENGINE,
    "CostPredictionDecision": _COST_PREDICTION_ENGINE,
    "CostPredictionPlan": _COST_PREDICTION_ENGINE,
    "CreativeDiversityPrediction": _CREATIVE_DIVERSITY_PREDICTOR,
    "CreativeDiversityPredictionPlan": _CREATIVE_DIVERSITY_PREDICTOR,
    "CreativeConsistencyPrediction": _CREATIVE_CONSISTENCY_PREDICTOR,
    "CreativeConsistencyPredictionPlan": _CREATIVE_CONSISTENCY_PREDICTOR,
    "RoutingExplainabilityPlan": _ROUTING_EXPLAINABILITY,
    "RoutingExplanationRecord": _ROUTING_EXPLAINABILITY,
    "ModelRoutingArchitectureConsistencyRecord": (
        _MODEL_ROUTING_ARCHITECTURE_CONSISTENCY
    ),
    "ModelRoutingArchitectureConsistencyRegistry": (
        _MODEL_ROUTING_ARCHITECTURE_CONSISTENCY
    ),
    "ModelRoutingFailurePathAuditRecord": _MODEL_ROUTING_FAILURE_PATH_AUDIT,
    "ModelRoutingFailurePathAuditRegistry": _MODEL_ROUTING_FAILURE_PATH_AUDIT,
    "AdvisoryHybridRoutingPolicy": _ROUTING_INTELLIGENCE,
    "AdvisoryHybridRoutingPolicyRegistry": _ROUTING_INTELLIGENCE,
    "ApiKeyDetectionMetadata": _ROUTING_INTELLIGENCE,
    "CredentialBoundary": _ROUTING_INTELLIGENCE,
    "LocalModelAvailabilityMetadata": _ROUTING_INTELLIGENCE,
    "LocalModelInventoryMetadata": _ROUTING_INTELLIGENCE,
    "LocalRuntimeDetectionMetadata": _ROUTING_INTELLIGENCE,
    "ModelRoutingIntelligenceRegistry": _ROUTING_INTELLIGENCE,
    "ProviderAvailabilityMetadata": _ROUTING_INTELLIGENCE,
    "ProviderAvailabilityRegistry": _ROUTING_INTELLIGENCE,
    "RoutingExecutionModeProfile": _ROUTING_INTELLIGENCE,
    "RoutingExecutionModeRegistry": _ROUTING_INTELLIGENCE,
    "RoutingProviderProfile": _ROUTING_INTELLIGENCE,
    "RoutingProviderProfileRegistry": _ROUTING_INTELLIGENCE,
    "RoutingSafetyContract": _ROUTING_INTELLIGENCE,
    "RoutingSafetyContractRegistry": _ROUTING_INTELLIGENCE,
    "RoutingUnavailableReason": _ROUTING_INTELLIGENCE,
    "TaskAwareRoutingDecision": _ROUTING_INTELLIGENCE,
    "TaskAwareRoutingRegistry": _ROUTING_INTELLIGENCE,
    "HybridWorkflowFallback": _ADAPTIVE_HYBRID_WORKFLOW_OPTIMIZER,
    "HybridWorkflowOptimizationCandidate": (
        _ADAPTIVE_HYBRID_WORKFLOW_OPTIMIZER
    ),
    "HybridWorkflowOptimizationPlan": _ADAPTIVE_HYBRID_WORKFLOW_OPTIMIZER,
    "HybridWorkflowSimulationEstimate": (
        _ADAPTIVE_HYBRID_WORKFLOW_OPTIMIZER
    ),
    "EscalationOptimizationDecision": _ADAPTIVE_ESCALATION_OPTIMIZER,
    "EscalationOptimizationPlan": _ADAPTIVE_ESCALATION_OPTIMIZER,
    "AgentActivationOptimizationCandidate": _AGENT_ACTIVATION_OPTIMIZER,
    "AgentActivationOptimizationPlan": _AGENT_ACTIVATION_OPTIMIZER,
    "AdaptiveCostQualityCandidate": _ADAPTIVE_COST_QUALITY_OPTIMIZER,
    "AdaptiveCostQualityPlan": _ADAPTIVE_COST_QUALITY_OPTIMIZER,
    "AdaptiveLatencyCandidate": _ADAPTIVE_LATENCY_OPTIMIZER,
    "AdaptiveLatencyPlan": _ADAPTIVE_LATENCY_OPTIMIZER,
    "AdaptiveExecutionStrategyCandidate": (
        _ADAPTIVE_EXECUTION_STRATEGY_SELECTION
    ),
    "AdaptiveExecutionStrategySelectionPlan": (
        _ADAPTIVE_EXECUTION_STRATEGY_SELECTION
    ),
    "DynamicAgentAllocationCandidate": _DYNAMIC_AGENT_ALLOCATION,
    "DynamicAgentAllocationPlan": _DYNAMIC_AGENT_ALLOCATION,
    "AgentDiversityOptimizationCandidate": _AGENT_DIVERSITY_OPTIMIZER,
    "AgentDiversityOptimizationPlan": _AGENT_DIVERSITY_OPTIMIZER,
    "DynamicResourceAllocationCandidate": _DYNAMIC_RESOURCE_ALLOCATION,
    "DynamicResourceAllocationPlan": _DYNAMIC_RESOURCE_ALLOCATION,
    "WorkflowSelfTuningPolicy": _WORKFLOW_SELF_TUNING_POLICIES,
    "WorkflowSelfTuningPolicyPlan": _WORKFLOW_SELF_TUNING_POLICIES,
    "ExecutionConfidenceSignal": _EXECUTION_CONFIDENCE_ENGINE,
    "ExecutionConfidencePlan": _EXECUTION_CONFIDENCE_ENGINE,
    "WorkflowRiskFactor": _WORKFLOW_RISK_ENGINE,
    "WorkflowRiskPlan": _WORKFLOW_RISK_ENGINE,
    "CreativeExplorationOptimizationCandidate": _CREATIVE_EXPLORATION_OPTIMIZER,
    "CreativeExplorationOptimizationPlan": _CREATIVE_EXPLORATION_OPTIMIZER,
    "EmergenceOptimizationCandidate": _EMERGENCE_OPTIMIZER,
    "EmergenceOptimizationPlan": _EMERGENCE_OPTIMIZER,
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
    (
        "multimodal_real_time_workflow_visualization_profiles_for_"
        "creative_evolution_timeline_profile"
    ): (
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
    "MemorySummarizationResult": _MEMORY_SUMMARIZATION,
    "MemorySummarySegment": _MEMORY_SUMMARIZATION,
    "OrchestrationContextAssembler": _CTX,
    "ProjectMemoryContext": _MEM,
    "PromptConversationTurnInput": _PROMPT_INPUTS,
    "PromptArtifactRefinementInput": _PROMPT_INPUTS,
    "PromptCompressionInputSection": _PROMPT_COMPRESSION,
    "PromptCompressionResult": _PROMPT_COMPRESSION,
    "PromptCompressionSection": _PROMPT_COMPRESSION,
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
    "RetrievalCompressionChunk": _RETRIEVAL_COMPRESSION,
    "RetrievalCompressionResult": _RETRIEVAL_COMPRESSION,
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
    "InMemoryExecutionCache": _CACHE_LAYER,
    "JinjaPromptRenderer": _PROMPT_TEMPLATES,
    "begin_assistant_workflow": _WORKFLOW,
    "analyze_assistant_execution_graph": _EXECUTION_GRAPH_ANALYZER,
    "analyze_creative_complexity": _CREATIVE_COMPLEXITY_ANALYZER,
    "analyze_workflow_cost": _WORKFLOW_COST_ANALYZER,
    "analyze_workflow_complexity": _WORKFLOW_COMPLEXITY_ANALYZER,
    "build_assistant_workflow_graph": _WORKFLOW_GRAPH,
    "calibrate_artifact_quality": _QUALITY_CALIBRATION,
    "artifact_quality_score": _REFINEMENT_PASSES,
    "attach_refinement_history": _REFINEMENT_PASSES,
    "build_initial_workflow_graph_state": _WORKFLOW_GRAPH,
    "build_refinement_objective": _REFINEMENT_PASSES,
    "build_assembled_context_request": _CTX,
    "build_execution_cache_key": _CACHE_LAYER,
    "build_memory_context_request": _MEM,
    "build_prompt_input_request": _PROMPT_INPUTS,
    "build_provider_generation_request": _GEN,
    "build_rendered_prompt_request": _PROMPT_TEMPLATES,
    "build_retrieval_context_request": _RETRIEVAL,
    "compress_prompt_sections": _PROMPT_COMPRESSION,
    "compress_prompt_text": _PROMPT_COMPRESSION,
    "compress_rendered_prompt": _PROMPT_COMPRESSION,
    "compress_retrieval_chunks": _RETRIEVAL_COMPRESSION,
    "compress_retrieval_context": _RETRIEVAL_COMPRESSION,
    "complete_workflow_step": _WORKFLOW,
    "creative_complexity_factor_by_id": _CREATIVE_COMPLEXITY_ANALYZER,
    "creative_complexity_factors_for_kind": _CREATIVE_COMPLEXITY_ANALYZER,
    "context_budget_allocation_by_id": _CONTEXT_BUDGET_PLANNER,
    "context_budget_allocations_for_kind": _CONTEXT_BUDGET_PLANNER,
    "context_route_decision_by_id": _CONTEXT_ROUTER,
    "context_route_decisions_for_lane": _CONTEXT_ROUTER,
    "context_reuse_candidate_by_id": _CONTEXT_REUSE,
    "context_reuse_candidates_for_status": _CONTEXT_REUSE,
    "exploration_budget_allocation_by_id": _EXPLORATION_BUDGET_PLANNER,
    "exploration_budget_allocations_for_topic": _EXPLORATION_BUDGET_PLANNER,
    "execution_graph_edges_from": _EXECUTION_GRAPH_ANALYZER,
    "execution_graph_edges_to": _EXECUTION_GRAPH_ANALYZER,
    "execution_graph_node_by_id": _EXECUTION_GRAPH_ANALYZER,
    "execution_cache_entry_is_fresh": _CACHE_LAYER,
    "workflow_cost_component_by_id": _WORKFLOW_COST_ANALYZER,
    "workflow_cost_components_for_kind": _WORKFLOW_COST_ANALYZER,
    "workflow_complexity_factor_by_id": _WORKFLOW_COMPLEXITY_ANALYZER,
    "workflow_complexity_factors_for_kind": _WORKFLOW_COMPLEXITY_ANALYZER,
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
    "evaluate_hitl_budget_gate": _HITL_BUDGET_GATE,
    "recommend_runtime_execution": _RUNTIME_RECOMMENDATION_ENGINE,
    "evaluate_execution_policies": _EXECUTION_POLICY_ENGINE,
    "recommend_model_profile": _MODEL_RECOMMENDATION_ENGINE,
    "build_model_capability_matrix": _MODEL_CAPABILITY_MATRIX,
    "build_provider_capability_matrix": _PROVIDER_CAPABILITY_MATRIX,
    "predict_quality_for_route": _QUALITY_PREDICTION_ENGINE,
    "predict_cost_for_route": _COST_PREDICTION_ENGINE,
    "predict_creative_diversity": _CREATIVE_DIVERSITY_PREDICTOR,
    "predict_creative_consistency": _CREATIVE_CONSISTENCY_PREDICTOR,
    "predict_performance": _PERFORMANCE_PREDICTION,
    "detect_performance_regressions": _PERFORMANCE_REGRESSION_DETECTION,
    "optimize_resource_utilization": _RESOURCE_UTILIZATION_OPTIMIZER,
    "explain_routing_decision": _ROUTING_EXPLAINABILITY,
    "model_routing_architecture_consistency_registry": (
        _MODEL_ROUTING_ARCHITECTURE_CONSISTENCY
    ),
    "derive_sacred_geometry_guidance": _SACRED_GEOMETRY,
    "detect_sacred_geometry_concepts": _SACRED_GEOMETRY,
    "detect_shader_presets": _SHADER_PRESETS,
    "derive_shader_preset_guidance": _SHADER_PRESETS,
    "derive_visual_style_guidance": _VISUAL_STYLES,
    "evaluate_budget_policies": _BUDGET_POLICIES,
    "derive_reference_fusion_guidance": _REFERENCE_FUSION,
    "detect_visual_styles": _VISUAL_STYLES,
    "extract_workflow_artifacts": _ARTIFACTS,
    "fail_workflow": _WORKFLOW,
    "execution_cost_forecast_scenario_by_id": _EXECUTION_COST_FORECASTING,
    "execution_cost_forecast_scenarios_for_kind": _EXECUTION_COST_FORECASTING,
    "execution_path_candidate_by_id": _EXECUTION_PATH_OPTIMIZATION,
    "execution_path_candidates_for_status": _EXECUTION_PATH_OPTIMIZATION,
    "parallel_schedule_candidate_by_id": _PARALLEL_SCHEDULER,
    "parallel_schedule_candidates_for_status": _PARALLEL_SCHEDULER,
    "latency_optimization_candidate_by_id": _LATENCY_OPTIMIZER,
    "latency_optimization_candidates_for_status": _LATENCY_OPTIMIZER,
    "async_execution_candidate_by_id": _ASYNC_EXECUTION,
    "async_execution_candidates_for_status": _ASYNC_EXECUTION,
    "streaming_optimization_candidate_by_id": _STREAMING_OPTIMIZER,
    "streaming_optimization_candidates_for_status": _STREAMING_OPTIMIZER,
    "retry_policy_candidate_by_id": _RETRY_POLICIES,
    "retry_policy_candidates_for_status": _RETRY_POLICIES,
    "load_balance_candidate_by_id": _LOAD_BALANCER,
    "load_balance_candidates_for_status": _LOAD_BALANCER,
    "execution_profile_candidate_by_id": _EXECUTION_PROFILING,
    "execution_profile_candidates_for_status": _EXECUTION_PROFILING,
    "workflow_replay_candidate_by_id": _WORKFLOW_REPLAY_ENGINE,
    "workflow_replay_candidates_for_status": _WORKFLOW_REPLAY_ENGINE,
    "execution_replay_candidate_by_id": _EXECUTION_REPLAY_ENGINE,
    "execution_replay_candidates_for_status": _EXECUTION_REPLAY_ENGINE,
    "bottleneck_candidate_by_id": _BOTTLENECK_DETECTION,
    "bottleneck_candidates_for_status": _BOTTLENECK_DETECTION,
    "throughput_optimization_candidate_by_id": _THROUGHPUT_OPTIMIZER,
    "throughput_optimization_candidates_for_status": _THROUGHPUT_OPTIMIZER,
    "performance_prediction_by_id": _PERFORMANCE_PREDICTION,
    "performance_predictions_for_band": _PERFORMANCE_PREDICTION,
    "performance_benchmark_scenario_by_id": _PERFORMANCE_BENCHMARKING,
    "performance_benchmark_scenarios_for_status": _PERFORMANCE_BENCHMARKING,
    "reasoning_budget_recommendation_by_id": _REASONING_BUDGET_OPTIMIZER,
    "reasoning_budget_recommendations_for_status": _REASONING_BUDGET_OPTIMIZER,
    "performance_regression_signal_by_id": _PERFORMANCE_REGRESSION_DETECTION,
    "performance_regression_signals_for_status": (
        _PERFORMANCE_REGRESSION_DETECTION
    ),
    "resource_utilization_recommendation_by_id": (
        _RESOURCE_UTILIZATION_OPTIMIZER
    ),
    "resource_utilization_recommendations_for_status": (
        _RESOURCE_UTILIZATION_OPTIMIZER
    ),
    "production_observability_architecture_registry": (
        _PRODUCTION_OBSERVABILITY_ARCHITECTURE_CONSISTENCY
    ),
    "production_observability_architecture_by_surface": (
        _PRODUCTION_OBSERVABILITY_ARCHITECTURE_CONSISTENCY
    ),
    "production_observability_architecture_records_for_layer": (
        _PRODUCTION_OBSERVABILITY_ARCHITECTURE_CONSISTENCY
    ),
    "production_observability_failure_path_audit_registry": (
        _PRODUCTION_OBSERVABILITY_FAILURE_PATH_AUDIT
    ),
    "production_observability_failure_path_audit_by_id": (
        _PRODUCTION_OBSERVABILITY_FAILURE_PATH_AUDIT
    ),
    "production_observability_failure_path_audits_for_check": (
        _PRODUCTION_OBSERVABILITY_FAILURE_PATH_AUDIT
    ),
    "production_observability_failure_path_audits_for_surface": (
        _PRODUCTION_OBSERVABILITY_FAILURE_PATH_AUDIT
    ),
    "performance_architecture_consistency_registry": (
        _PERFORMANCE_ARCHITECTURE_CONSISTENCY
    ),
    "performance_architecture_consistency_by_surface": (
        _PERFORMANCE_ARCHITECTURE_CONSISTENCY
    ),
    "performance_architecture_consistency_records_for_layer": (
        _PERFORMANCE_ARCHITECTURE_CONSISTENCY
    ),
    "performance_failure_path_audit_registry": _PERFORMANCE_FAILURE_PATH_AUDIT,
    "performance_failure_path_audit_by_id": _PERFORMANCE_FAILURE_PATH_AUDIT,
    "performance_failure_path_audits_for_check": (
        _PERFORMANCE_FAILURE_PATH_AUDIT
    ),
    "performance_failure_path_audits_for_surface": (
        _PERFORMANCE_FAILURE_PATH_AUDIT
    ),
    "execution_strategy_by_id": _EXECUTION_STRATEGY_SELECTION,
    "execution_strategies_for_status": _EXECUTION_STRATEGY_SELECTION,
    "execution_optimization_failure_audit_by_id": (
        _EXECUTION_OPTIMIZATION_FAILURE_AUDIT
    ),
    "execution_optimization_failure_audit_registry": (
        _EXECUTION_OPTIMIZATION_FAILURE_AUDIT
    ),
    "execution_optimization_failure_audits_for_check": (
        _EXECUTION_OPTIMIZATION_FAILURE_AUDIT
    ),
    "model_route_candidate_by_id": _MODEL_ROUTER,
    "model_route_candidates_for_status": _MODEL_ROUTER,
    "local_cloud_route_decision_by_id": _LOCAL_CLOUD_ROUTING,
    "local_cloud_route_decisions_for_lane": _LOCAL_CLOUD_ROUTING,
    "hybrid_route_decision_by_id": _HYBRID_ROUTING,
    "hybrid_route_decisions_for_mode": _HYBRID_ROUTING,
    "quality_cost_candidate_by_id": _QUALITY_COST_OPTIMIZER,
    "quality_cost_candidates_for_posture": _QUALITY_COST_OPTIMIZER,
    "cost_estimate_scenario_by_id": _COST_ESTIMATOR,
    "cost_estimate_scenarios_for_confidence": _COST_ESTIMATOR,
    "budget_policy_by_id": _BUDGET_POLICIES,
    "budget_policies_for_posture": _BUDGET_POLICIES,
    "hitl_budget_gate_by_id": _HITL_BUDGET_GATE,
    "hitl_budget_gates_for_status": _HITL_BUDGET_GATE,
    "runtime_recommendation_by_id": _RUNTIME_RECOMMENDATION_ENGINE,
    "runtime_recommendations_for_posture": _RUNTIME_RECOMMENDATION_ENGINE,
    "execution_policy_by_id": _EXECUTION_POLICY_ENGINE,
    "execution_policies_for_posture": _EXECUTION_POLICY_ENGINE,
    "model_recommendation_by_id": _MODEL_RECOMMENDATION_ENGINE,
    "model_recommendations_for_policy_posture": _MODEL_RECOMMENDATION_ENGINE,
    "model_capability_row_by_profile_id": _MODEL_CAPABILITY_MATRIX,
    "model_capability_rows_for_route": _MODEL_CAPABILITY_MATRIX,
    "provider_capability_row_by_profile_id": _PROVIDER_CAPABILITY_MATRIX,
    "provider_capability_rows_for_provider": _PROVIDER_CAPABILITY_MATRIX,
    "provider_capability_rows_for_route": _PROVIDER_CAPABILITY_MATRIX,
    "quality_prediction_by_id": _QUALITY_PREDICTION_ENGINE,
    "quality_predictions_for_level": _QUALITY_PREDICTION_ENGINE,
    "cost_prediction_by_id": _COST_PREDICTION_ENGINE,
    "cost_predictions_for_band": _COST_PREDICTION_ENGINE,
    "creative_diversity_prediction_by_id": _CREATIVE_DIVERSITY_PREDICTOR,
    "creative_diversity_predictions_for_band": _CREATIVE_DIVERSITY_PREDICTOR,
    "creative_consistency_prediction_by_id": _CREATIVE_CONSISTENCY_PREDICTOR,
    "creative_consistency_predictions_for_band": _CREATIVE_CONSISTENCY_PREDICTOR,
    "routing_explanation_by_id": _ROUTING_EXPLAINABILITY,
    "routing_explanations_for_source": _ROUTING_EXPLAINABILITY,
    "model_routing_architecture_consistency_by_surface": (
        _MODEL_ROUTING_ARCHITECTURE_CONSISTENCY
    ),
    "model_routing_architecture_consistency_records_for_layer": (
        _MODEL_ROUTING_ARCHITECTURE_CONSISTENCY
    ),
    "model_routing_failure_path_audit_by_id": (
        _MODEL_ROUTING_FAILURE_PATH_AUDIT
    ),
    "model_routing_failure_path_audit_registry": (
        _MODEL_ROUTING_FAILURE_PATH_AUDIT
    ),
    "model_routing_failure_path_audits_for_check": (
        _MODEL_ROUTING_FAILURE_PATH_AUDIT
    ),
    "model_routing_failure_path_audits_for_surface": (
        _MODEL_ROUTING_FAILURE_PATH_AUDIT
    ),
    "advisory_hybrid_routing_policy_by_direction": _ROUTING_INTELLIGENCE,
    "advisory_hybrid_routing_policy_registry": _ROUTING_INTELLIGENCE,
    "credential_boundary_by_provider_id": _ROUTING_INTELLIGENCE,
    "model_routing_intelligence_registry": _ROUTING_INTELLIGENCE,
    "optimize_latency": _LATENCY_OPTIMIZER,
    "optimize_streaming": _STREAMING_OPTIMIZER,
    "provider_availability_by_provider_id": _ROUTING_INTELLIGENCE,
    "provider_availability_registry": _ROUTING_INTELLIGENCE,
    "routing_execution_mode_by_id": _ROUTING_INTELLIGENCE,
    "routing_execution_mode_registry": _ROUTING_INTELLIGENCE,
    "routing_provider_profile_by_id": _ROUTING_INTELLIGENCE,
    "routing_provider_profile_registry": _ROUTING_INTELLIGENCE,
    "routing_safety_contract_by_boundary": _ROUTING_INTELLIGENCE,
    "routing_safety_contract_registry": _ROUTING_INTELLIGENCE,
    "routing_unavailable_reason_by_code": _ROUTING_INTELLIGENCE,
    "task_aware_routing_registry": _ROUTING_INTELLIGENCE,
    "task_routing_decision_by_task_type": _ROUTING_INTELLIGENCE,
    "task_routing_decisions_requiring_hitl": _ROUTING_INTELLIGENCE,
    "finish_workflow": _WORKFLOW,
    "forecast_execution_cost": _EXECUTION_COST_FORECASTING,
    "build_token_dashboard": _TOKEN_DASHBOARD,
    "build_cost_dashboard": _COST_DASHBOARD,
    "build_quality_dashboard": _QUALITY_DASHBOARD,
    "build_performance_dashboard": _PERFORMANCE_DASHBOARD,
    "build_production_telemetry": _PRODUCTION_TELEMETRY,
    "build_workflow_diagnostics": _WORKFLOW_DIAGNOSTICS,
    "build_agent_diagnostics": _AGENT_DIAGNOSTICS,
    "build_routing_diagnostics": _ROUTING_DIAGNOSTICS,
    "build_escalation_diagnostics": _ESCALATION_DIAGNOSTICS,
    "build_failure_analysis": _FAILURE_ANALYSIS,
    "build_error_intelligence": _ERROR_INTELLIGENCE,
    "build_workflow_health_monitoring": _WORKFLOW_HEALTH_MONITORING,
    "build_system_health_monitoring": _SYSTEM_HEALTH_MONITORING,
    "build_creative_analytics": _CREATIVE_ANALYTICS,
    "build_confidence_analytics": _CONFIDENCE_ANALYTICS,
    "build_creative_diversity_analytics": _CREATIVE_DIVERSITY_ANALYTICS,
    "build_runtime_timeline": _RUNTIME_TIMELINE,
    "build_workflow_explainability_dashboard": (
        _WORKFLOW_EXPLAINABILITY_DASHBOARD
    ),
    "memory_summary_segment_by_id": _MEMORY_SUMMARIZATION,
    "memory_summary_segments_for_kind": _MEMORY_SUMMARIZATION,
    "next_workflow_step": _WORKFLOW,
    "optimize_quality_cost": _QUALITY_COST_OPTIMIZER,
    "estimate_routing_cost": _COST_ESTIMATOR,
    "optimize_reasoning_budget": _REASONING_BUDGET_OPTIMIZER,
    "optimize_throughput": _THROUGHPUT_OPTIMIZER,
    "prepare_workflow_preview_results": _ARTIFACTS,
    "plan_context_budget": _CONTEXT_BUDGET_PLANNER,
    "plan_context_reuse": _CONTEXT_REUSE,
    "plan_async_execution": _ASYNC_EXECUTION,
    "detect_bottlenecks": _BOTTLENECK_DETECTION,
    "plan_exploration_budget": _EXPLORATION_BUDGET_PLANNER,
    "plan_execution_profiling": _EXECUTION_PROFILING,
    "plan_execution_path_optimization": _EXECUTION_PATH_OPTIMIZATION,
    "plan_execution_replay": _EXECUTION_REPLAY_ENGINE,
    "plan_load_balancer": _LOAD_BALANCER,
    "plan_performance_benchmarking": _PERFORMANCE_BENCHMARKING,
    "plan_parallel_scheduler": _PARALLEL_SCHEDULER,
    "plan_workflow_replay": _WORKFLOW_REPLAY_ENGINE,
    "plan_retry_policies": _RETRY_POLICIES,
    "plan_workflow_pruning": _WORKFLOW_PRUNING,
    "plan_next_refinement_pass": _REFINEMENT_PASSES,
    "prompt_compression_section_by_id": _PROMPT_COMPRESSION,
    "prompt_compression_sections_for_status": _PROMPT_COMPRESSION,
    "refinement_opportunities": _REFINEMENT_PASSES,
    "reference_fusion_prompt_lines": _REFERENCE_FUSION,
    "restart_workflow_step": _WORKFLOW,
    "retrieval_compression_chunk_by_id": _RETRIEVAL_COMPRESSION,
    "retrieval_compression_chunks_for_status": _RETRIEVAL_COMPRESSION,
    "review_assistant_answer": _WORKFLOW_REVIEW,
    "route_context_sources": _CONTEXT_ROUTER,
    "route_local_vs_cloud": _LOCAL_CLOUD_ROUTING,
    "route_hybrid_model_request": _HYBRID_ROUTING,
    "route_model_request": _MODEL_ROUTER,
    "route_request": _ROUTING,
    "optimize_hybrid_workflow": _ADAPTIVE_HYBRID_WORKFLOW_OPTIMIZER,
    "hybrid_workflow_candidate_by_id": _ADAPTIVE_HYBRID_WORKFLOW_OPTIMIZER,
    "hybrid_workflow_candidates_requiring_hitl": (
        _ADAPTIVE_HYBRID_WORKFLOW_OPTIMIZER
    ),
    "optimize_escalation_policy": _ADAPTIVE_ESCALATION_OPTIMIZER,
    "escalation_optimization_decision_by_id": (
        _ADAPTIVE_ESCALATION_OPTIMIZER
    ),
    "escalation_optimization_decisions_for_posture": (
        _ADAPTIVE_ESCALATION_OPTIMIZER
    ),
    "optimize_agent_activation": _AGENT_ACTIVATION_OPTIMIZER,
    "agent_activation_candidate_by_agent_id": _AGENT_ACTIVATION_OPTIMIZER,
    "agent_activation_candidates_for_status": _AGENT_ACTIVATION_OPTIMIZER,
    "optimize_adaptive_cost_quality": _ADAPTIVE_COST_QUALITY_OPTIMIZER,
    "adaptive_cost_quality_candidate_by_id": _ADAPTIVE_COST_QUALITY_OPTIMIZER,
    "adaptive_cost_quality_candidates_for_posture": (
        _ADAPTIVE_COST_QUALITY_OPTIMIZER
    ),
    "optimize_adaptive_latency": _ADAPTIVE_LATENCY_OPTIMIZER,
    "adaptive_latency_candidate_by_id": _ADAPTIVE_LATENCY_OPTIMIZER,
    "adaptive_latency_candidates_for_posture": _ADAPTIVE_LATENCY_OPTIMIZER,
    "select_dynamic_execution_strategy": _ADAPTIVE_EXECUTION_STRATEGY_SELECTION,
    "adaptive_execution_strategy_by_id": (
        _ADAPTIVE_EXECUTION_STRATEGY_SELECTION
    ),
    "adaptive_execution_strategies_for_status": (
        _ADAPTIVE_EXECUTION_STRATEGY_SELECTION
    ),
    "allocate_dynamic_agents": _DYNAMIC_AGENT_ALLOCATION,
    "dynamic_agent_allocation_by_agent_id": _DYNAMIC_AGENT_ALLOCATION,
    "dynamic_agent_allocations_for_lane": _DYNAMIC_AGENT_ALLOCATION,
    "optimize_agent_diversity": _AGENT_DIVERSITY_OPTIMIZER,
    "agent_diversity_candidate_by_agent_id": _AGENT_DIVERSITY_OPTIMIZER,
    "agent_diversity_candidates_for_status": _AGENT_DIVERSITY_OPTIMIZER,
    "allocate_dynamic_resources": _DYNAMIC_RESOURCE_ALLOCATION,
    "dynamic_resource_allocation_by_id": _DYNAMIC_RESOURCE_ALLOCATION,
    "dynamic_resource_allocations_for_status": _DYNAMIC_RESOURCE_ALLOCATION,
    "plan_workflow_self_tuning_policies": _WORKFLOW_SELF_TUNING_POLICIES,
    "workflow_self_tuning_policy_by_id": _WORKFLOW_SELF_TUNING_POLICIES,
    "workflow_self_tuning_policies_for_status": _WORKFLOW_SELF_TUNING_POLICIES,
    "evaluate_execution_confidence": _EXECUTION_CONFIDENCE_ENGINE,
    "execution_confidence_signal_by_id": _EXECUTION_CONFIDENCE_ENGINE,
    "execution_confidence_signals_for_band": _EXECUTION_CONFIDENCE_ENGINE,
    "evaluate_workflow_risk": _WORKFLOW_RISK_ENGINE,
    "workflow_risk_factor_by_id": _WORKFLOW_RISK_ENGINE,
    "workflow_risk_factors_for_severity": _WORKFLOW_RISK_ENGINE,
    "optimize_creative_exploration": _CREATIVE_EXPLORATION_OPTIMIZER,
    "creative_exploration_candidate_by_id": _CREATIVE_EXPLORATION_OPTIMIZER,
    "creative_exploration_candidates_for_status": (
        _CREATIVE_EXPLORATION_OPTIMIZER
    ),
    "optimize_emergence": _EMERGENCE_OPTIMIZER,
    "emergence_candidate_by_id": _EMERGENCE_OPTIMIZER,
    "emergence_candidates_for_status": _EMERGENCE_OPTIMIZER,
    "select_execution_strategy": _EXECUTION_STRATEGY_SELECTION,
    "summarize_memory_context": _MEMORY_SUMMARIZATION,
    "sacred_geometry_prompt_lines": _SACRED_GEOMETRY,
    "shader_preset_prompt_lines": _SHADER_PRESETS,
    "visual_style_prompt_lines": _VISUAL_STYLES,
    "workflow_pruning_candidate_by_id": _WORKFLOW_PRUNING,
    "workflow_pruning_candidates_for_status": _WORKFLOW_PRUNING,
    "skip_workflow_step": _WORKFLOW,
    "start_refinement_pass_record": _REFINEMENT_PASSES,
    "stream_assistant_workflow_events": _WORKFLOW_GRAPH,
    "start_workflow_step": _WORKFLOW,
    "cost_dashboard_panel_by_id": _COST_DASHBOARD,
    "cost_dashboard_panels_for_pressure": _COST_DASHBOARD,
    "quality_dashboard_panel_by_id": _QUALITY_DASHBOARD,
    "quality_dashboard_panels_for_pressure": _QUALITY_DASHBOARD,
    "performance_dashboard_panel_by_id": _PERFORMANCE_DASHBOARD,
    "performance_dashboard_panels_for_pressure": _PERFORMANCE_DASHBOARD,
    "production_telemetry_channel_by_id": _PRODUCTION_TELEMETRY,
    "production_telemetry_channels_for_status": _PRODUCTION_TELEMETRY,
    "token_dashboard_panel_by_id": _TOKEN_DASHBOARD,
    "token_dashboard_panels_for_pressure": _TOKEN_DASHBOARD,
    "agent_diagnostic_panel_by_id": _AGENT_DIAGNOSTICS,
    "agent_diagnostic_panels_for_status": _AGENT_DIAGNOSTICS,
    "routing_diagnostic_panel_by_id": _ROUTING_DIAGNOSTICS,
    "routing_diagnostic_panels_for_status": _ROUTING_DIAGNOSTICS,
    "escalation_diagnostic_panel_by_id": _ESCALATION_DIAGNOSTICS,
    "escalation_diagnostic_panels_for_status": _ESCALATION_DIAGNOSTICS,
    "failure_analysis_panel_by_id": _FAILURE_ANALYSIS,
    "failure_analysis_panels_for_status": _FAILURE_ANALYSIS,
    "error_intelligence_panel_by_id": _ERROR_INTELLIGENCE,
    "error_intelligence_panels_for_status": _ERROR_INTELLIGENCE,
    "workflow_health_panel_by_id": _WORKFLOW_HEALTH_MONITORING,
    "workflow_health_panels_for_status": _WORKFLOW_HEALTH_MONITORING,
    "system_health_panel_by_id": _SYSTEM_HEALTH_MONITORING,
    "system_health_panels_for_status": _SYSTEM_HEALTH_MONITORING,
    "creative_analytics_panel_by_id": _CREATIVE_ANALYTICS,
    "creative_analytics_panels_for_status": _CREATIVE_ANALYTICS,
    "confidence_analytics_panel_by_id": _CONFIDENCE_ANALYTICS,
    "confidence_analytics_panels_for_status": _CONFIDENCE_ANALYTICS,
    "creative_diversity_analytics_panel_by_id": _CREATIVE_DIVERSITY_ANALYTICS,
    "creative_diversity_analytics_panels_for_status": (
        _CREATIVE_DIVERSITY_ANALYTICS
    ),
    "runtime_timeline_panel_by_id": _RUNTIME_TIMELINE,
    "runtime_timeline_panels_for_status": _RUNTIME_TIMELINE,
    "workflow_explainability_panel_by_id": _WORKFLOW_EXPLAINABILITY_DASHBOARD,
    "workflow_explainability_panels_for_status": (
        _WORKFLOW_EXPLAINABILITY_DASHBOARD
    ),
    "workflow_diagnostic_panel_by_id": _WORKFLOW_DIAGNOSTICS,
    "workflow_diagnostic_panels_for_status": _WORKFLOW_DIAGNOSTICS,
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
