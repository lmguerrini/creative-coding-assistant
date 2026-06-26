import type { WorkstationError } from "./workstation-errors";

export type InspectorTabName =
  | "Overview"
  | "Preview"
  | "Runtime"
  | "Code"
  | "Workflow"
  | "Telemetry"
  | "Artifacts"
  | "Retrieval";

export type InspectorTabState = {
  label: InspectorTabName;
  active: boolean;
  summary: string;
  badge?: string;
};

export type WorkflowNodeId =
  | "intake"
  | "routing"
  | "memory"
  | "retrieval"
  | "context_assembly"
  | "prompt_input"
  | "planning"
  | "director"
  | "reasoning"
  | "prompt_rendering"
  | "generation"
  | "artifact_extraction"
  | "preview_preparation"
  | "artifact_critique"
  | "review"
  | "refinement"
  | "finalization"
  | "failure";

export const workflowNodeOrder = [
  "intake",
  "routing",
  "memory",
  "retrieval",
  "context_assembly",
  "prompt_input",
  "planning",
  "director",
  "reasoning",
  "prompt_rendering",
  "generation",
  "artifact_extraction",
  "preview_preparation",
  "artifact_critique",
  "review",
  "refinement",
  "finalization",
  "failure"
] as const satisfies readonly WorkflowNodeId[];

export type WorkflowStepState = {
  nodeId: WorkflowNodeId;
  displayLabel: string;
  state: "complete" | "active" | "queued" | "skipped" | "branch";
  detail: string;
};

export type ClarificationQuestionSummary = {
  id: string;
  prompt: string;
  kind: "single_choice" | "short_answer";
  suggestedOptions: string[];
  defaultRecommendation: string | null;
};

export type ClarificationSummary = {
  reason: string;
  confidence: number;
  summary: string;
  originalQuery: string;
  questions: ClarificationQuestionSummary[];
  suggestedOptions: string[];
  defaultRecommendation: string | null;
  signalSummary: string[];
};

export type CreativeIntentDimensionName =
  | "narrative"
  | "symbolic"
  | "emotional"
  | "geometric"
  | "motion"
  | "rhythm"
  | "light_color"
  | "audio"
  | "interaction"
  | "climax_transformation";

export type CreativeIntentExplicitness =
  | "explicit"
  | "inferred"
  | "absent"
  | "ambiguous";

export type CreativeAbstractionLevel =
  | "literal"
  | "stylized"
  | "symbolic"
  | "abstract"
  | "mixed"
  | "unspecified";

export type CreativeIntentDimensionSummary = {
  name: CreativeIntentDimensionName;
  explicitness: CreativeIntentExplicitness;
  summary: string;
  signals: string[];
  guidance: string[];
};

export type CreativeIntentDecompositionSummary = {
  role: "creative_intent_decomposer";
  normalizedIntent: string;
  primaryExpression: string;
  narrativeIntent: CreativeIntentDimensionSummary;
  symbolicIntent: CreativeIntentDimensionSummary;
  emotionalIntent: CreativeIntentDimensionSummary;
  geometricIntent: CreativeIntentDimensionSummary;
  motionIntent: CreativeIntentDimensionSummary;
  rhythmIntent: CreativeIntentDimensionSummary;
  lightColorIntent: CreativeIntentDimensionSummary;
  audioIntent: CreativeIntentDimensionSummary;
  interactionIntent: CreativeIntentDimensionSummary;
  climaxTransformationIntent: CreativeIntentDimensionSummary;
  abstractionLevel: CreativeAbstractionLevel;
  experientialGoal: string;
  unresolvedIntentGaps: string[];
  hitlQuestions: string[];
  atomicDimensions: CreativeIntentDimensionSummary[];
  promptGuidance: string[];
  authorityBoundary: string;
  evidence: string[];
};

export type CreativeHierarchyDimension =
  | "symbolism"
  | "narrative"
  | "emotion"
  | "geometry"
  | "motion"
  | "rhythm"
  | "light_color"
  | "audio"
  | "interaction"
  | "visual_impact"
  | "performance"
  | "simplicity"
  | "complexity"
  | "runtime_safety"
  | "experiential_depth";

export type CreativeHierarchyTier = "primary" | "secondary" | "flexible";

export type CreativeHierarchySource =
  | "explicit"
  | "implied"
  | "coherence"
  | "constraint";

export type CreativeHierarchyPrioritySummary = {
  dimension: CreativeHierarchyDimension;
  tier: CreativeHierarchyTier;
  rank: number;
  priorityScore: number;
  source: CreativeHierarchySource;
  rationale: string;
  evidence: string[];
  sacrificeGuidance: string;
};

export type CreativeHierarchyPlanSummary = {
  role: "creative_hierarchy_planner";
  primaryCreativePriorities: CreativeHierarchyPrioritySummary[];
  secondaryCreativePriorities: CreativeHierarchyPrioritySummary[];
  nonNegotiableDimensions: CreativeHierarchyDimension[];
  flexibleDimensions: CreativeHierarchyDimension[];
  priorityRationale: string[];
  priorityConflicts: string[];
  hierarchyConfidence: number;
  hitlQuestions: string[];
  promptGuidance: string[];
  authorityBoundary: string;
  evidence: string[];
};

export type CreativeExecutionPlanSummary = {
  outputModality: "visual" | "audio" | "audiovisual";
  generationStrategy: string;
  recommendedRuntime: string | null;
  recommendedRendererId: string | null;
  recommendedPreviewTarget: string | null;
  recommendedShaderStyle: string | null;
  candidateCount: number;
  refinementBudget: number;
  expectedComplexity: "low" | "medium" | "high";
  estimatedTokenCost: number;
  exportReadiness: "ready" | "partial" | "blocked";
  runtimeAvailable: boolean;
  runtimeSupportSummary: string;
  planSteps: string[];
  constraints: string[];
  evidence: string[];
};

export type CreativeStrategyId =
  | "recursive_emergence"
  | "fractal_growth"
  | "particle_cosmology"
  | "cellular_evolution"
  | "sacred_geometry"
  | "field_dynamics"
  | "minimal_generative_systems";

export type CreativeStrategyAlternativeSummary = {
  strategy: CreativeStrategyId;
  confidence: number;
  rationale: string;
};

export type CreativeStrategySummary = {
  role: "creative_strategy_engine";
  primaryStrategy: CreativeStrategyId;
  confidence: number;
  rationale: string;
  creativeGoals: string[];
  symbolicAlignment: string[];
  alternativeStrategies: CreativeStrategyAlternativeSummary[];
  strategyDirectives: string[];
  implementationBoundary: string;
  evidence: string[];
};

export type CreativeTechniqueId =
  | "fractal_recursion"
  | "particle_systems"
  | "reaction_diffusion"
  | "boids"
  | "cellular_automata"
  | "voronoi"
  | "noise_fields"
  | "recursive_geometry"
  | "sdf"
  | "signed_distance_composition"
  | "feedback_systems"
  | "audio_reactive_mappings";

export type CreativeTechniqueCompatibility = "strong" | "moderate" | "weak";

export type CreativeTechniquePressure = "low" | "medium" | "high";

export type CreativeTechniqueAlternativeSummary = {
  technique: CreativeTechniqueId;
  confidence: number;
  rationale: string;
};

export type CreativeTechniqueSummary = {
  role: "creative_technique_selector";
  primaryTechnique: CreativeTechniqueId;
  confidence: number;
  rationale: string;
  strategyAlignment: CreativeStrategyId | null;
  compatibility: CreativeTechniqueCompatibility;
  complexityPressure: CreativeTechniquePressure;
  performancePressure: CreativeTechniquePressure;
  artisticSuitability: string[];
  implementationNotes: string[];
  alternativeTechniques: CreativeTechniqueAlternativeSummary[];
  techniqueConstraints: string[];
  selectionBoundary: string;
  evidence: string[];
};

export type RuntimeCapabilityId =
  | "p5_js"
  | "three_js"
  | "react_three_fiber"
  | "glsl"
  | "hydra"
  | "tone_js"
  | "gsap"
  | "svg"
  | "canvas";

export type RuntimeCapabilityFit = "strong" | "moderate" | "weak";

export type RuntimeCapabilityComplexity = "low" | "medium" | "high";

export type RuntimePreviewSupport =
  | "backend_preview_supported"
  | "workstation_preview_bounded"
  | "code_only";

export type RuntimeCapabilityCandidateSummary = {
  runtime: RuntimeCapabilityId;
  label: string;
  suitability: RuntimeCapabilityFit;
  confidence: number;
  strategyAlignment: RuntimeCapabilityFit;
  techniqueCompatibility: RuntimeCapabilityFit;
  outputGoalFit: RuntimeCapabilityFit;
  implementationComplexity: RuntimeCapabilityComplexity;
  performancePressure: CreativeConstraintPressure;
  previewSupport: RuntimePreviewSupport;
  strengths: string[];
  limitations: string[];
  risks: string[];
  promptGuidance: string[];
  evidence: string[];
};

export type RuntimeCapabilityReasonerSummary = {
  role: "runtime_capability_reasoner";
  outputGoal: string;
  likelyCandidates: RuntimeCapabilityId[];
  candidateRuntimes: RuntimeCapabilityCandidateSummary[];
  strategyContext: string | null;
  techniqueContext: string | null;
  constraintContext: string | null;
  hitlAdvisable: boolean;
  hitlReason: string | null;
  promptGuidance: string[];
  authorityBoundary: string;
  evidence: string[];
};

export type RuntimeCompatibilityLevel =
  | "compatible"
  | "partially_compatible"
  | "unsupported";

export type RuntimePortability = "high" | "medium" | "low";

export type RuntimeInteroperability = "high" | "medium" | "low";

export type RuntimeCompatibilityConfidenceSummary = {
  runtime: RuntimeCapabilityId;
  label: string;
  confidence: number;
};

export type RuntimeCompatibilityAssessmentSummary = {
  runtime: RuntimeCapabilityId;
  label: string;
  compatibility: RuntimeCompatibilityLevel;
  confidence: number;
  compatibilityReasons: string[];
  runtimeRequirements: string[];
  runtimeLimitations: string[];
  dependencyCompatibility: string[];
  expectedImplementationComplexity: RuntimeCapabilityComplexity;
  portability: RuntimePortability;
  interoperability: RuntimeInteroperability;
  implementationRisks: string[];
  promptGuidance: string[];
  evidence: string[];
};

export type RuntimeCompatibilityProfileSummary = {
  role: "runtime_compatibility_engine";
  compatibleRuntimes: RuntimeCapabilityId[];
  unsupportedRuntimes: RuntimeCapabilityId[];
  preferredRuntimes: RuntimeCapabilityId[];
  runtimeConfidence: RuntimeCompatibilityConfidenceSummary[];
  compatibilityAssessments: RuntimeCompatibilityAssessmentSummary[];
  runtimeRequirements: string[];
  runtimeLimitations: string[];
  dependencyCompatibility: string[];
  expectedImplementationComplexity: RuntimeCapabilityComplexity;
  portability: RuntimePortability;
  interoperability: RuntimeInteroperability;
  missingRuntimeInformation: string[];
  implementationRisks: string[];
  hitlQuestions: string[];
  promptGuidance: string[];
  authorityBoundary: string;
  evidence: string[];
};

export type ArtifactCapabilityFit =
  | "strong"
  | "moderate"
  | "weak"
  | "unsupported";

export type ArtifactCapabilityConfidenceSummary = {
  target: RuntimeCapabilityId;
  label: string;
  confidence: number;
};

export type ArtifactCapabilityProfileSummary = {
  target: RuntimeCapabilityId;
  label: string;
  capabilityConfidence: number;
  capabilityReasons: string[];
  strengths: string[];
  weaknesses: string[];
  unsupportedCapabilities: string[];
  riskyCapabilities: string[];
  artifactFit: ArtifactCapabilityFit;
  creativeFit: ArtifactCapabilityFit;
  generativeFit: ArtifactCapabilityFit;
  interactionFit: ArtifactCapabilityFit;
  audiovisualFit: ArtifactCapabilityFit;
  exportFit: ArtifactCapabilityFit;
  interoperabilityFit: ArtifactCapabilityFit;
  portabilityFit: ArtifactCapabilityFit;
  capabilityRisks: string[];
  promptGuidance: string[];
  evidence: string[];
};

export type ArtifactCapabilityMatrixSummary = {
  role: "artifact_capability_matrix";
  capabilityProfiles: ArtifactCapabilityProfileSummary[];
  strongestTargets: RuntimeCapabilityId[];
  weakestTargets: RuntimeCapabilityId[];
  targetStrengths: string[];
  targetWeaknesses: string[];
  unsupportedOrRiskyCapabilities: string[];
  capabilityConfidence: ArtifactCapabilityConfidenceSummary[];
  artifactFit: ArtifactCapabilityFit;
  creativeFit: ArtifactCapabilityFit;
  generativeFit: ArtifactCapabilityFit;
  interactionFit: ArtifactCapabilityFit;
  audiovisualFit: ArtifactCapabilityFit;
  exportFit: ArtifactCapabilityFit;
  interoperabilityFit: ArtifactCapabilityFit;
  portabilityFit: ArtifactCapabilityFit;
  missingCapabilityInformation: string[];
  capabilityRisks: string[];
  hitlQuestions: string[];
  promptGuidance: string[];
  authorityBoundary: string;
  evidence: string[];
};

export type MultiArtifactStrategyRole = "primary" | "supporting" | "optional";

export type MultiArtifactStrategyPriority = "critical" | "high" | "medium" | "low";

export type MultiArtifactStrategyAction =
  | "produce"
  | "separate"
  | "document"
  | "handoff";

export type MultiArtifactStrategyCombinationMode =
  | "primary_with_supporting_sections"
  | "separated_parallel_sections"
  | "defer_combination";

export type MultiArtifactStrategyArtifactSummary = {
  artifactId: string;
  title: string;
  role: MultiArtifactStrategyRole;
  artifactType: ArtifactType;
  artifactFamily: ArtifactFamily;
  priority: MultiArtifactStrategyPriority;
  purpose: string;
  runtimeTargets: RuntimeCapabilityId[];
  capabilityTargets: RuntimeCapabilityId[];
  dependsOn: string[];
  handoffPoints: string[];
  evidence: string[];
};

export type MultiArtifactStrategySequenceStepSummary = {
  stepId: string;
  order: number;
  artifactId: string;
  action: MultiArtifactStrategyAction;
  rationale: string;
  dependsOn: string[];
  promptGuidance: string[];
};

export type MultiArtifactStrategyPriorityEntrySummary = {
  artifactId: string;
  priority: MultiArtifactStrategyPriority;
  rationale: string;
};

export type MultiArtifactStrategyGroupSummary = {
  groupId: string;
  label: string;
  artifactIds: string[];
  groupingRationale: string;
  separationRationale: string;
};

export type MultiArtifactStrategySummary = {
  role: "multi_artifact_strategy";
  artifactStrategySummary: string;
  primaryArtifact: MultiArtifactStrategyArtifactSummary;
  supportingArtifacts: MultiArtifactStrategyArtifactSummary[];
  artifactSequence: MultiArtifactStrategySequenceStepSummary[];
  artifactPriority: MultiArtifactStrategyPriorityEntrySummary[];
  artifactGrouping: MultiArtifactStrategyGroupSummary[];
  artifactSeparationStrategy: string[];
  artifactCombinationStrategy: string[];
  artifactDependencyOrder: string[];
  artifactHandoffPoints: string[];
  runtimeAwareArtifactStrategy: string[];
  capabilityAwareArtifactStrategy: string[];
  combinationMode: MultiArtifactStrategyCombinationMode;
  riskAreas: string[];
  missingInformation: string[];
  hitlQuestions: string[];
  promptGuidance: string[];
  authorityBoundary: string;
  evidence: string[];
};

export type ArtifactCriticRiskAssessment = "low" | "medium" | "high" | "blocked";

export type ArtifactCriticSummary = {
  role: "artifact_critic";
  critiqueConfidence: number;
  critiqueSummary: string;
  strengths: string[];
  weaknesses: string[];
  capabilityGaps: string[];
  dependencyConcerns: string[];
  runtimeConcerns: string[];
  scalabilityConcerns: string[];
  maintainabilityConcerns: string[];
  complexityConcerns: string[];
  riskAssessment: ArtifactCriticRiskAssessment;
  unsupportedAssumptions: string[];
  missingInformation: string[];
  openQuestions: string[];
  hitlQuestions: string[];
  improvementOpportunities: string[];
  promptGuidance: string[];
  authorityBoundary: string;
  evidence: string[];
};

export type CreativeCriticRiskAssessment =
  | "low"
  | "medium"
  | "high"
  | "blocked";

export type CreativeCriticSummary = {
  role: "creative_critic_engine";
  criticConfidence: number;
  critiqueSummary: string;
  creativeStrengths: string[];
  creativeWeaknesses: string[];
  conceptQuality: number;
  executionQuality: number;
  artifactQuality: number;
  coherenceQuality: number;
  runtimeFitQuality: number;
  originalityQuality: number;
  clarityQuality: number;
  feasibilityQuality: number;
  riskAssessment: CreativeCriticRiskAssessment;
  missingInformation: string[];
  unsupportedAssumptions: string[];
  improvementOpportunities: string[];
  hitlQuestions: string[];
  promptGuidance: string[];
  authorityBoundary: string;
  evidence: string[];
};

export type SelfEvaluationCompleteness =
  | "complete"
  | "mostly_complete"
  | "partial"
  | "blocked";

export type SelfEvaluationRisk = "low" | "medium" | "high";

export type SelfEvaluationAmbiguity = "low" | "medium" | "high";

export type SelfEvaluationSummary = {
  role: "self_evaluation_engine";
  selfEvaluationConfidence: number;
  evaluationSummary: string;
  requestAlignment: number;
  intentAlignment: number;
  constraintAlignment: number;
  artifactAlignment: number;
  runtimeAlignment: number;
  creativeCoherence: number;
  technicalCoherence: number;
  completenessAssessment: SelfEvaluationCompleteness;
  ambiguityAssessment: SelfEvaluationAmbiguity;
  hallucinationRisk: SelfEvaluationRisk;
  overreachRisk: SelfEvaluationRisk;
  underdeliveryRisk: SelfEvaluationRisk;
  missingInformation: string[];
  unsupportedAssumptions: string[];
  qualityGaps: string[];
  improvementOpportunities: string[];
  hitlQuestions: string[];
  promptGuidance: string[];
  authorityBoundary: string;
  evidence: string[];
};

export type ImprovementPriorityLevel =
  | "critical"
  | "high"
  | "medium"
  | "low";

export type ImprovementImpact = "high" | "medium" | "low";

export type ImprovementRisk = "low" | "medium" | "high";

export type ImprovementSource =
  | "creative_critic"
  | "self_evaluation"
  | "artifact_context"
  | "workflow_context";

export type CreativeImprovementPrioritySummary = {
  priorityId: string;
  title: string;
  priority: ImprovementPriorityLevel;
  impact: ImprovementImpact;
  risk: ImprovementRisk;
  source: ImprovementSource;
  rationale: string;
  evidence: string[];
};

export type CreativeImprovementPlannerSummary = {
  role: "creative_improvement_planner";
  serializationVersion: "v1";
  confidence: number;
  improvementSummary: string;
  improvementPriorities: CreativeImprovementPrioritySummary[];
  highestImpactOpportunities: string[];
  lowRiskImprovements: string[];
  experimentalImprovements: string[];
  tradeOffRecommendations: string[];
  improvementRationale: string[];
  evidence: string[];
  futureRefinementCandidates: string[];
  hitlQuestions: string[];
  promptGuidance: string[];
  authorityBoundary: string;
};

export type ReflectionLoopPriority =
  | "critical"
  | "high"
  | "medium"
  | "low"
  | "none";

export type ReflectionLoopDepth = "none" | "light" | "moderate" | "deep";

export type ReflectionLoopEstimate = "none" | "low" | "medium" | "high";

export type ReflectionLoopHitlRecommendation =
  | "not_needed"
  | "optional"
  | "recommended"
  | "required";

export type ReflectionLoopSummary = {
  role: "reflection_loop_engine";
  serializationVersion: "v1";
  reflectionConfidence: number;
  reflectionSummary: string;
  reflectionRequired: boolean;
  reflectionPriority: ReflectionLoopPriority;
  reflectionRationale: string[];
  reflectionDepth: ReflectionLoopDepth;
  expectedQualityGain: ReflectionLoopEstimate;
  expectedRiskReduction: ReflectionLoopEstimate;
  expectedCost: ReflectionLoopEstimate;
  expectedLatency: ReflectionLoopEstimate;
  confidenceAfterReflection: number;
  unresolvedQuestions: string[];
  refinementCandidates: string[];
  stopConditions: string[];
  hitlRecommendation: ReflectionLoopHitlRecommendation;
  promptGuidance: string[];
  evidence: string[];
  authorityBoundary: string;
};

export type CreativeConfidenceLevel =
  | "very_high"
  | "high"
  | "medium"
  | "low"
  | "critical";

export type CreativeConfidenceComponentSource =
  | "creative_critic"
  | "self_evaluation"
  | "creative_improvement_planner"
  | "reflection_loop"
  | "planning_metadata";

export type ExpectedOutputReliability =
  | "very_high"
  | "high"
  | "medium"
  | "low"
  | "blocked";

export type ExpectedExecutionReadiness =
  | "ready"
  | "needs_caveats"
  | "needs_hitl"
  | "blocked";

export type ExpectedHumanReviewNeed =
  | "not_needed"
  | "optional"
  | "recommended"
  | "required";

export type EscalationRecommendation =
  | "none"
  | "monitor"
  | "hitl_review"
  | "future_escalation";

export type ConfidenceTrend =
  | "improving"
  | "stable"
  | "declining"
  | "conflicting"
  | "unknown";

export type CreativeConfidenceComponentSummary = {
  source: CreativeConfidenceComponentSource;
  score: number;
  weight: number;
  rationale: string;
  evidence: string[];
};

export type CreativeConfidenceSummary = {
  role: "creative_confidence_engine";
  serializationVersion: "v1";
  confidenceScore: number;
  confidenceLevel: CreativeConfidenceLevel;
  confidenceSummary: string;
  confidenceRationale: string[];
  confidenceComponents: CreativeConfidenceComponentSummary[];
  confidenceLimitations: string[];
  confidenceUncertainties: string[];
  confidenceStrengths: string[];
  confidenceWeaknesses: string[];
  expectedOutputReliability: ExpectedOutputReliability;
  expectedExecutionReadiness: ExpectedExecutionReadiness;
  expectedHumanReviewNeed: ExpectedHumanReviewNeed;
  escalationRecommendation: EscalationRecommendation;
  confidenceTrend: ConfidenceTrend;
  confidenceEvidence: string[];
  hitlRecommendation: ExpectedHumanReviewNeed;
  promptGuidance: string[];
  authorityBoundary: string;
};

export type CreativeScoreDimension =
  | "creativity"
  | "technical"
  | "coherence"
  | "feasibility"
  | "artifact"
  | "runtime";

export type CreativeScoreBand =
  | "excellent"
  | "strong"
  | "solid"
  | "weak"
  | "critical";

export type CreativeScoreSignalSource =
  | "creative_critic"
  | "self_evaluation"
  | "creative_improvement_planner"
  | "reflection_loop"
  | "creative_confidence"
  | "planning_metadata";

export type CreativeScoreBreakdownSummary = {
  dimension: CreativeScoreDimension;
  score: number;
  weight: number;
  rationale: string;
  evidence: string[];
};

export type CreativeScoreSummary = {
  role: "creative_score_engine";
  serializationVersion: "v1";
  overallCreativeScore: number;
  scoreBand: CreativeScoreBand;
  scoreSummary: string;
  scoreBreakdown: CreativeScoreBreakdownSummary[];
  creativityScore: number;
  technicalScore: number;
  coherenceScore: number;
  feasibilityScore: number;
  artifactScore: number;
  runtimeScore: number;
  confidenceWeight: number;
  uncertaintyPenalty: number;
  riskPenalty: number;
  strengths: string[];
  weaknesses: string[];
  scoreRationale: string[];
  scoreEvidence: string[];
  hitlRecommendation: ExpectedHumanReviewNeed;
  promptGuidance: string[];
  authorityBoundary: string;
};

export type ArtifactRefinerSummary = {
  role: "artifact_refiner";
  refinementConfidence: number;
  refinementSummary: string;
  recommendedImprovements: string[];
  priorityImprovements: string[];
  capabilityImprovements: string[];
  dependencyImprovements: string[];
  runtimeImprovements: string[];
  scalabilityImprovements: string[];
  maintainabilityImprovements: string[];
  complexityReductions: string[];
  riskReductions: string[];
  refinementCandidates: string[];
  implementationSuggestions: string[];
  alternativeRefinementPaths: string[];
  hitlQuestions: string[];
  promptGuidance: string[];
  authorityBoundary: string;
  evidence: string[];
};

export type ArtifactImplementationReadiness =
  | "ready"
  | "needs_caveats"
  | "needs_hitl"
  | "blocked";

export type ArtifactImplementationComplexity = "low" | "medium" | "high";

export type ArtifactImplementationRisk = "low" | "medium" | "high" | "blocked";

export type ArtifactImplementationPriority =
  | "critical"
  | "high"
  | "medium"
  | "low";

export type ArtifactIntelligenceSynthesisSummary = {
  role: "artifact_intelligence_synthesis";
  synthesisConfidence: number;
  synthesisSummary: string;
  recommendedArtifactPath: string;
  recommendedStrategySummary: string;
  recommendedRuntimeDirection: string;
  majorStrengths: string[];
  majorWeaknesses: string[];
  majorRisks: string[];
  dependencyOverview: string;
  capabilityOverview: string;
  refinementOverview: string;
  critiqueOverview: string;
  implementationReadiness: ArtifactImplementationReadiness;
  implementationComplexity: ArtifactImplementationComplexity;
  implementationRisk: ArtifactImplementationRisk;
  implementationPriority: ArtifactImplementationPriority;
  hitlQuestions: string[];
  promptGuidance: string[];
  authorityBoundary: string;
  evidence: string[];
};

export type ArtifactMergeStrategy =
  | "single_artifact_no_merge"
  | "primary_with_supporting_sections"
  | "separated_advisory_sections"
  | "defer_merge_preserve_separation";

export type ArtifactMergePlannerSummary = {
  role: "artifact_merge_planner";
  mergeConfidence: number;
  mergeSummary: string;
  mergeStrategy: ArtifactMergeStrategy;
  compositionStrategy: string;
  artifactBoundaries: string[];
  artifactJoinPoints: string[];
  artifactSeparationPoints: string[];
  integrationOrder: string[];
  compositionRisks: string[];
  dependencyMergeRisks: string[];
  runtimeMergeRisks: string[];
  capabilityMergeRisks: string[];
  recommendedMergePath: string;
  alternativeMergePaths: string[];
  rejectedMergePaths: string[];
  hitlQuestions: string[];
  promptGuidance: string[];
  authorityBoundary: string;
  evidence: string[];
};

export type ArtifactExportReadiness =
  | "ready_with_caveats"
  | "needs_handoff_metadata"
  | "blocked_by_missing_metadata"
  | "defer_export";

export type ArtifactExportIntelligenceSummary = {
  role: "artifact_export_intelligence";
  exportConfidence: number;
  exportSummary: string;
  exportTargets: string[];
  preferredExportTarget: string;
  exportFormatRecommendations: string[];
  exportReadiness: ArtifactExportReadiness;
  exportRequirements: string[];
  exportConstraints: string[];
  exportRisks: string[];
  runtimeExportNotes: string[];
  artifactPackageNotes: string[];
  portabilityNotes: string[];
  interoperabilityNotes: string[];
  documentationRequirements: string[];
  downstreamToolHandoffs: string[];
  rejectedExportPaths: string[];
  hitlQuestions: string[];
  promptGuidance: string[];
  authorityBoundary: string;
  evidence: string[];
};

export type ArtifactEngineCategory = "artifact_intelligence";

export type ArtifactEngineCacheability =
  | "deterministic_per_request"
  | "deterministic_with_upstream_metadata";

export type ArtifactEngineParallelizationSupport =
  | "requires_ordered_upstream_metadata"
  | "parallel_after_required_inputs";

export type ArtifactEngineCostMetadataSummary = {
  relativeCost: "low" | "medium";
  externalProviderCalls: boolean;
  costBasis: string;
  cacheSensitivity: string;
};

export type ArtifactEngineLatencyMetadataSummary = {
  relativeLatency: "low" | "medium";
  latencyBasis: string;
  blockingInputs: string[];
};

export type ArtifactIntelligenceEngineContractSummary = {
  engineId: string;
  engineName: string;
  engineVersion: string;
  engineCategory: ArtifactEngineCategory;
  authorityBoundary: string;
  requiredInputs: string[];
  optionalInputs: string[];
  producedMetadata: string[];
  producedSignals: string[];
  confidenceSignals: string[];
  ambiguitySignals: string[];
  riskSignals: string[];
  escalationCandidates: string[];
  downstreamDependencies: string[];
  upstreamDependencies: string[];
  cacheability: ArtifactEngineCacheability;
  parallelizationSupport: ArtifactEngineParallelizationSupport;
  estimatedCostMetadata: ArtifactEngineCostMetadataSummary;
  estimatedLatencyMetadata: ArtifactEngineLatencyMetadataSummary;
  serializationVersion: "artifact_engine_contract.v1";
  futureAgentHooks: string[];
  futureExecutionHooks: string[];
};

export type ArtifactIntelligenceEngineContractRegistrySummary = {
  role: "artifact_intelligence_engine_contract_registry";
  engineCategory: ArtifactEngineCategory;
  serializationVersion: "artifact_engine_contract_registry.v1";
  authorityBoundary: string;
  engineContracts: ArtifactIntelligenceEngineContractSummary[];
  engineIds: string[];
  contractCount: number;
  futureAgentConsumers: string[];
};

export type CreativeTradeoffAxis =
  | "creative_expressiveness"
  | "concept_fidelity"
  | "implementation_complexity"
  | "performance"
  | "runtime_support"
  | "previewability"
  | "cost_sensitivity"
  | "safety"
  | "maintainability"
  | "hitl";

export type CreativeTradeoffSeverity =
  | "info"
  | "watch"
  | "risk"
  | "blocking";

export type CreativeTradeoffPressure = "low" | "medium" | "high";

export type CreativeTradeoffSummary = {
  sourceAxis: CreativeTradeoffAxis;
  targetAxis: CreativeTradeoffAxis;
  severity: CreativeTradeoffSeverity;
  summary: string;
  creativeBenefit: string;
  technicalCost: string;
  runtimeImplication: string;
  mitigation: string;
  directorDiscussionPoint: string;
  hitlRecommended: boolean;
  evidence: string[];
};

export type CreativeTradeoffExplorerSummary = {
  role: "creative_tradeoff_explorer";
  outputGoal: string;
  primaryTradeoffs: CreativeTradeoffSummary[];
  creativeBenefits: string[];
  technicalCosts: string[];
  runtimeRisks: string[];
  performanceConcerns: string[];
  complexityRisks: string[];
  fidelityRisks: string[];
  costSensitivity: CreativeTradeoffPressure;
  safetyConcerns: string[];
  maintainabilityConcerns: string[];
  hitlAdvisable: boolean;
  hitlReason: string | null;
  directorDiscussionPoints: string[];
  promptGuidance: string[];
  authorityBoundary: string;
  evidence: string[];
};

export type CreativeQualityPredictionLevel =
  | "strong"
  | "promising"
  | "ambiguous"
  | "risky"
  | "blocked";

export type CreativeQualityDimension =
  | "intent_clarity"
  | "symbolic_coherence"
  | "narrative_coherence"
  | "emotional_coherence"
  | "geometric_formal_clarity"
  | "technique_suitability"
  | "runtime_suitability"
  | "tradeoff_balance"
  | "constraint_alignment"
  | "implementation_feasibility"
  | "previewability"
  | "performance_risk"
  | "originality_potential"
  | "aesthetic_coherence_potential";

export type CreativeQualitySignalSummary = {
  dimension: CreativeQualityDimension;
  score: number;
  summary: string;
  evidence: string[];
};

export type CreativeQualityPredictionSummary = {
  role: "creative_quality_predictor";
  predictedQualityLevel: CreativeQualityPredictionLevel;
  confidence: number;
  readinessScore: number;
  strongestQualitySignals: CreativeQualitySignalSummary[];
  weakestQualitySignals: CreativeQualitySignalSummary[];
  qualityRisks: string[];
  missingInformation: string[];
  likelyFailureModes: string[];
  suggestedImprovements: string[];
  hitlQuestions: string[];
  promptGuidance: string[];
  authorityBoundary: string;
  evidence: string[];
};

export type SymbolicNarrativeArchetype =
  | "descent_and_return"
  | "death_and_rebirth"
  | "emergence_from_chaos"
  | "initiation"
  | "ascent"
  | "dissolution_and_reintegration"
  | "expansion_from_seed_to_cosmos"
  | "fragmentation_and_recomposition"
  | "threshold_crossing"
  | "spiral_transformation"
  | "mirror_reflection_journey"
  | "dark_to_light_transformation"
  | "symbolic_vignette";

export type SymbolicNarrativePhaseName =
  | "opening"
  | "development"
  | "threshold"
  | "climax"
  | "resolution";

export type SymbolicNarrativePhaseSummary = {
  phase: SymbolicNarrativePhaseName;
  title: string;
  symbolicFunction: string;
  emotionalState: string;
  visualState: string;
  motionState: string;
  audioState: string | null;
  guidance: string[];
  evidence: string[];
};

export type SymbolicNarrativePlanSummary = {
  role: "symbolic_narrative_planner";
  narrativeArchetype: SymbolicNarrativeArchetype;
  symbolicArc: string;
  openingPhase: SymbolicNarrativePhaseSummary;
  developmentPhase: SymbolicNarrativePhaseSummary;
  thresholdPhase: SymbolicNarrativePhaseSummary;
  climaxPhase: SymbolicNarrativePhaseSummary;
  resolutionPhase: SymbolicNarrativePhaseSummary;
  symbolicTransitions: string[];
  emotionalProgression: string[];
  visualProgression: string[];
  motionProgression: string[];
  audioProgression: string[];
  experientialGoal: string;
  unresolvedNarrativeGaps: string[];
  hitlQuestions: string[];
  promptGuidance: string[];
  authorityBoundary: string;
  evidence: string[];
};

export type CreativeCompositionPattern =
  | "central_emergence"
  | "radial_expansion"
  | "spiral_composition"
  | "layered_depth"
  | "field_composition"
  | "threshold_composition"
  | "descent_ascent_composition"
  | "fragmented_recomposition"
  | "mirrored_composition"
  | "orbiting_focal_structure"
  | "distributed_constellation"
  | "minimal_void_and_form_composition";

export type CreativeCompositionPlanSummary = {
  role: "creative_composition_planner";
  compositionPattern: CreativeCompositionPattern;
  primaryFocalPoint: string;
  secondaryFocalElements: string[];
  spatialOrganization: string;
  foregroundBackgroundRelationship: string;
  visualHierarchy: string[];
  densityPlan: string;
  rhythmPlan: string;
  balancePlan: string;
  symmetryAsymmetryGuidance: string;
  depthLayeringGuidance: string;
  transitionGuidance: string[];
  cameraViewpointGuidance: string | null;
  audiovisualCompositionNotes: string[];
  compositionRisks: string[];
  unresolvedCompositionGaps: string[];
  hitlQuestions: string[];
  promptGuidance: string[];
  authorityBoundary: string;
  evidence: string[];
};

export type ProceduralFamily =
  | "fractals"
  | "recursive_geometry"
  | "l_systems"
  | "particle_systems"
  | "boids"
  | "cellular_automata"
  | "reaction_diffusion"
  | "voronoi_systems"
  | "noise_fields"
  | "flow_fields"
  | "signed_distance_fields"
  | "polar_radial_systems"
  | "grid_systems"
  | "graph_network_systems"
  | "swarm_systems"
  | "wave_systems"
  | "harmonic_oscillators"
  | "modular_tiling"
  | "sacred_geometry_pattern_systems";

export type ProceduralComplexityLevel = "low" | "medium" | "high";

export type ProceduralStructureChoiceSummary = {
  family: ProceduralFamily;
  label: string;
  rationale: string;
  evidence: string[];
};

export type ProceduralStructurePlanSummary = {
  role: "procedural_structure_planner";
  recommendedFamilies: ProceduralFamily[];
  primaryStructure: ProceduralStructureChoiceSummary;
  secondaryStructures: ProceduralStructureChoiceSummary[];
  combinationStrategy: string;
  spatialStructurePlan: string;
  temporalStructurePlan: string;
  interactionStructurePlan: string | null;
  audiovisualStructurePlan: string | null;
  complexityLevel: ProceduralComplexityLevel;
  runtimeSuitabilityNotes: string[];
  performanceRisks: string[];
  implementationRisks: string[];
  fallbackStructureOptions: ProceduralStructureChoiceSummary[];
  unresolvedProceduralGaps: string[];
  hitlQuestions: string[];
  promptGuidance: string[];
  authorityBoundary: string;
  evidence: string[];
};

export type GenerativeArchitecture =
  | "recursive_modular_blueprint"
  | "agent_field_blueprint"
  | "grid_state_blueprint"
  | "radial_pattern_blueprint"
  | "network_relation_blueprint"
  | "wave_modulation_blueprint"
  | "minimal_parameter_blueprint";

export type GenerativeModuleKind =
  | "seed_system"
  | "recursive_module"
  | "particle_emitter"
  | "force_field"
  | "attractor_field"
  | "noise_modulation_layer"
  | "symmetry_transform"
  | "tiling_layer"
  | "graph_network_layer"
  | "cellular_grid_layer"
  | "wave_oscillator"
  | "geometry_reassembly_layer"
  | "color_modulation_layer"
  | "audio_reactive_modulation_layer"
  | "camera_motion_path_hook";

export type GenerativeRelationshipType =
  | "feeds"
  | "modulates"
  | "constrains"
  | "emits"
  | "attracts"
  | "mirrors"
  | "samples"
  | "reassembles"
  | "times"
  | "fallback_for";

export type GenerativeParameterValueType =
  | "integer"
  | "float"
  | "boolean"
  | "vector"
  | "color"
  | "enum";

export type GenerativeParameterRole = "control" | "derived" | "constraint";

export type GenerativeEvolutionPhase =
  | "seed"
  | "growth"
  | "fragmentation"
  | "threshold"
  | "reassembly"
  | "stabilization"
  | "loop";

export type GenerativeEvolutionTrigger =
  | "time"
  | "interaction"
  | "audio"
  | "parameter"
  | "narrative_phase";

export type GenerativeHookType = "interaction" | "audiovisual";

export type GenerativeModuleSummary = {
  moduleId: string;
  kind: GenerativeModuleKind;
  label: string;
  sourceFamily: ProceduralFamily | null;
  purpose: string;
  inputs: string[];
  outputs: string[];
  parameters: string[];
  evolutionRole: string;
  implementationNotes: string[];
  safeguards: string[];
  evidence: string[];
};

export type GenerativeModuleRelationshipSummary = {
  sourceModuleId: string;
  targetModuleId: string;
  relationshipType: GenerativeRelationshipType;
  description: string;
  parameters: string[];
  evidence: string[];
};

export type GenerativeParameterSummary = {
  name: string;
  label: string;
  valueType: GenerativeParameterValueType;
  role: GenerativeParameterRole;
  defaultValue: string;
  bounds: string | null;
  controlledBy: string | null;
  targetModules: string[];
  rationale: string;
};

export type GenerativeEvolutionRuleSummary = {
  phase: GenerativeEvolutionPhase;
  trigger: GenerativeEvolutionTrigger;
  rule: string;
  affectedModules: string[];
  parameterChanges: string[];
  safeguards: string[];
};

export type GenerativeStructureHookSummary = {
  hookId: string;
  hookType: GenerativeHookType;
  signal: string;
  targetModules: string[];
  parameterMapping: string[];
  fallbackBehavior: string;
};

export type GenerativeFallbackBlueprintSummary = {
  name: string;
  architecture: GenerativeArchitecture;
  moduleKinds: GenerativeModuleKind[];
  parameterReductions: string[];
  reason: string;
  promptGuidance: string[];
};

export type GenerativeStructureBlueprintSummary = {
  role: "generative_structure_engine";
  blueprintName: string;
  generativeArchitecture: GenerativeArchitecture;
  proceduralModules: GenerativeModuleSummary[];
  moduleRelationships: GenerativeModuleRelationshipSummary[];
  parameterSchema: GenerativeParameterSummary[];
  controlParameters: string[];
  evolutionRules: GenerativeEvolutionRuleSummary[];
  spatialEvolution: string;
  temporalEvolution: string;
  interactionHooks: GenerativeStructureHookSummary[];
  audiovisualHooks: GenerativeStructureHookSummary[];
  runtimeImplementationGuidance: string[];
  performanceSafeguards: string[];
  fallbackBlueprint: GenerativeFallbackBlueprintSummary;
  unresolvedImplementationGaps: string[];
  hitlQuestions: string[];
  promptGuidance: string[];
  authorityBoundary: string;
  evidence: string[];
};

export type SemanticMotifId =
  | "seed"
  | "spiral"
  | "threshold"
  | "mirror"
  | "void"
  | "center"
  | "circumference"
  | "axis"
  | "descent"
  | "ascent"
  | "fragmentation"
  | "reintegration"
  | "wave"
  | "lattice"
  | "network"
  | "pearl"
  | "flame"
  | "root"
  | "tree"
  | "vessel"
  | "mandala"
  | "grid"
  | "swarm"
  | "orbit"
  | "pulse"
  | "breath"
  | "gate"
  | "eye"
  | "river"
  | "constellation";

export type SemanticMotifRole =
  | "anchor"
  | "threshold"
  | "transformation"
  | "connector"
  | "counterpoint"
  | "rhythm"
  | "spatial_order"
  | "material_signal"
  | "fallback";

export type SemanticMotifHierarchyLevel =
  | "primary"
  | "secondary"
  | "supporting"
  | "fallback";

export type SemanticMotifSummary = {
  motifId: SemanticMotifId;
  label: string;
  role: SemanticMotifRole;
  hierarchyLevel: SemanticMotifHierarchyLevel;
  rationale: string;
  recurrenceGuidance: string[];
  transformationGuidance: string[];
  evidence: string[];
};

export type SemanticMotifStructureMappingSummary = {
  motifId: SemanticMotifId;
  proceduralFamilies: ProceduralFamily[];
  generativeModuleIds: string[];
  generativeModuleKinds: GenerativeModuleKind[];
  structuralBehavior: string;
  evidence: string[];
};

export type SemanticMotifCompositionMappingSummary = {
  motifId: SemanticMotifId;
  compositionRole: string;
  spatialAnchor: string;
  rhythmOrDensityGuidance: string;
  evidence: string[];
};

export type SemanticMotifNarrativeMappingSummary = {
  motifId: SemanticMotifId;
  narrativeFunction: string;
  phaseAlignment: string[];
  evidence: string[];
};

export type SemanticMotifParameterMappingSummary = {
  motifId: SemanticMotifId;
  parameterNames: string[];
  parameterGuidance: string;
  evidence: string[];
};

export type SemanticMotifFallbackPlanSummary = {
  fallbackPrimaryMotif: SemanticMotifId;
  fallbackSecondaryMotifs: SemanticMotifId[];
  simplificationStrategy: string;
  preservedMeaning: string;
  promptGuidance: string[];
};

export type SemanticMotifSystemSummary = {
  role: "semantic_motif_engine";
  motifSystemName: string;
  primaryMotifs: SemanticMotifSummary[];
  secondaryMotifs: SemanticMotifSummary[];
  motifHierarchy: string[];
  motifRecurrencePlan: string[];
  motifTransformationPlan: string[];
  motifToStructureMapping: SemanticMotifStructureMappingSummary[];
  motifToCompositionMapping: SemanticMotifCompositionMappingSummary[];
  motifToNarrativeMapping: SemanticMotifNarrativeMappingSummary[];
  motifToParameterMapping: SemanticMotifParameterMappingSummary[];
  coherenceRisks: string[];
  overuseRisks: string[];
  underuseRisks: string[];
  unsupportedSymbolicClaims: string[];
  motifFallbackPlan: SemanticMotifFallbackPlanSummary;
  unresolvedMotifGaps: string[];
  hitlQuestions: string[];
  promptGuidance: string[];
  authorityBoundary: string;
  evidence: string[];
};

export type EmotionalTone =
  | "awe"
  | "wonder"
  | "mystery"
  | "serenity"
  | "tension"
  | "rupture"
  | "grief"
  | "dissolution"
  | "suspension"
  | "emergence"
  | "ecstasy"
  | "clarity"
  | "intimacy"
  | "vastness"
  | "ritual solemnity"
  | "playful curiosity"
  | "dread"
  | "release"
  | "transformation"
  | "integration";

export type EmotionalIntensity = "low" | "medium" | "high" | "variable";

export type EmotionalPhaseMappingSummary = {
  phase: SymbolicNarrativePhaseName;
  tone: EmotionalTone;
  intensity: EmotionalIntensity;
  guidance: string;
  evidence: string[];
};

export type EmotionalNarrativeMappingSummary = {
  tone: EmotionalTone;
  narrativePhase: SymbolicNarrativePhaseName;
  narrativeFunction: string;
  evidence: string[];
};

export type EmotionalMotifMappingSummary = {
  tone: EmotionalTone;
  motifId: SemanticMotifId | null;
  emotionalFunction: string;
  evidence: string[];
};

export type EmotionalCompositionMappingSummary = {
  tone: EmotionalTone;
  compositionPattern: CreativeCompositionPattern | null;
  compositionGuidance: string;
  spatialOrDensityGuidance: string;
  evidence: string[];
};

export type EmotionalStructureMappingSummary = {
  tone: EmotionalTone;
  proceduralFamilies: ProceduralFamily[];
  generativeModuleKinds: GenerativeModuleKind[];
  structuralGuidance: string;
  evidence: string[];
};

export type EmotionalParameterMappingSummary = {
  tone: EmotionalTone;
  parameterNames: string[];
  parameterGuidance: string;
  evidence: string[];
};

export type EmotionalFallbackStrategySummary = {
  fallbackPrimaryTone: EmotionalTone;
  fallbackSecondaryTones: EmotionalTone[];
  simplificationStrategy: string;
  preservedFeeling: string;
  promptGuidance: string[];
};

export type EmotionalConsistencyProfileSummary = {
  role: "emotional_consistency_engine";
  primaryEmotionalTone: EmotionalTone;
  secondaryEmotionalTones: EmotionalTone[];
  emotionalArc: string[];
  emotionalPhaseMapping: EmotionalPhaseMappingSummary[];
  emotionalToNarrativeMapping: EmotionalNarrativeMappingSummary[];
  emotionalToMotifMapping: EmotionalMotifMappingSummary[];
  emotionalToCompositionMapping: EmotionalCompositionMappingSummary[];
  emotionalToStructureMapping: EmotionalStructureMappingSummary[];
  emotionalToParameterMapping: EmotionalParameterMappingSummary[];
  colorLightGuidance: string[];
  motionRhythmGuidance: string[];
  audiovisualGuidance: string[];
  emotionalCoherenceScore: number;
  emotionalTensions: string[];
  mismatchRisks: string[];
  flatteningRisks: string[];
  overIntensityRisks: string[];
  underIntensityRisks: string[];
  fallbackEmotionalStrategy: EmotionalFallbackStrategySummary;
  unresolvedEmotionalGaps: string[];
  hitlQuestions: string[];
  promptGuidance: string[];
  authorityBoundary: string;
  evidence: string[];
};

export type CrossModalityChannel =
  | "visual_structure"
  | "motion"
  | "audio"
  | "rhythm"
  | "camera"
  | "structure"
  | "motif"
  | "emotion"
  | "interaction";

export type CrossModalityPattern =
  | "visual_led_composition"
  | "audio_reactive_composition"
  | "motion_led_transformation"
  | "rhythm_led_scene_evolution"
  | "camera_led_immersion"
  | "motif_led_symbolic_recurrence"
  | "structure_led_procedural_evolution"
  | "emotion_led_modulation"
  | "balanced_audiovisual_composition"
  | "minimal_visual_strong_sonic_cueing"
  | "dense_visual_restrained_audio"
  | "ritual_pulse_geometry_synchronization"
  | "fragmentation_reassembly_visual_motion_layers";

export type CrossModalityRoleSummary = {
  modality: CrossModalityChannel;
  role: string;
  priority: "primary" | "secondary" | "supporting" | "fallback";
  evidence: string[];
};

export type CrossModalityMappingSummary = {
  sourceModality: CrossModalityChannel;
  targetModality: CrossModalityChannel;
  mapping: string;
  cues: string[];
  motifId: SemanticMotifId | null;
  emotionalTone: EmotionalTone | null;
  evidence: string[];
};

export type CrossModalityTemporalCueSummary = {
  phase: SymbolicNarrativePhaseName;
  cue: string;
  modalities: CrossModalityChannel[];
  timingGuidance: string;
  evidence: string[];
};

export type CrossModalityFallbackStrategySummary = {
  fallbackPattern: CrossModalityPattern;
  preservedModalities: CrossModalityChannel[];
  reducedModalities: CrossModalityChannel[];
  simplificationStrategy: string;
  promptGuidance: string[];
};

export type CrossModalityCompositionProfileSummary = {
  role: "cross_modality_composer";
  modalityPattern: CrossModalityPattern;
  primaryModality: CrossModalityChannel;
  supportingModalities: CrossModalityChannel[];
  modalityHierarchy: CrossModalityRoleSummary[];
  visualRole: string;
  motionRole: string;
  audioRole: string | null;
  rhythmRole: string;
  cameraViewpointRole: string | null;
  structureRole: string;
  motifRole: string;
  emotionRole: string;
  modalitySynchronizationPlan: string[];
  visualToAudioMapping: CrossModalityMappingSummary[];
  audioToMotionMapping: CrossModalityMappingSummary[];
  motionToStructureMapping: CrossModalityMappingSummary[];
  motifToModalityMapping: CrossModalityMappingSummary[];
  emotionalToModalityMapping: CrossModalityMappingSummary[];
  temporalCuePlan: CrossModalityTemporalCueSummary[];
  contrastBalancePlan: string[];
  modalityConflicts: string[];
  overloadRisks: string[];
  underuseRisks: string[];
  fallbackMultimodalStrategy: CrossModalityFallbackStrategySummary;
  unresolvedModalityGaps: string[];
  hitlQuestions: string[];
  promptGuidance: string[];
  authorityBoundary: string;
  evidence: string[];
};

export type AudioVisualScenePattern =
  | "seed_to_expansion"
  | "descent_to_return"
  | "fragmentation_to_reintegration"
  | "threshold_crossing"
  | "spiral_ascent"
  | "chaos_to_order"
  | "void_to_emergence"
  | "contraction_to_release"
  | "ritual_opening_to_climax"
  | "wave_build_and_collapse"
  | "constellation_activation"
  | "mirror_inversion"
  | "pulse_escalation"
  | "calm_expansion_after_rupture";

export type AudioVisualCueType =
  | "visual"
  | "motion"
  | "audio"
  | "rhythm"
  | "camera"
  | "motif"
  | "emotion"
  | "procedural"
  | "synchronization";

export type AudioVisualScenePhaseSummary = {
  phase: SymbolicNarrativePhaseName;
  title: string;
  sceneFunction: string;
  visualState: string;
  motionState: string;
  audioState: string | null;
  rhythmState: string;
  cameraState: string | null;
  motifState: string;
  emotionalState: string;
  proceduralState: string;
  cueIds: string[];
  transitionOut: string;
  evidence: string[];
};

export type AudioVisualSceneCueSummary = {
  cueId: string;
  phase: SymbolicNarrativePhaseName;
  cueType: AudioVisualCueType;
  description: string;
  timing: string;
  modalities: CrossModalityChannel[];
  evidence: string[];
};

export type AudioVisualSceneTransitionSummary = {
  fromPhase: SymbolicNarrativePhaseName;
  toPhase: SymbolicNarrativePhaseName;
  transition: string;
  visualMotionGuidance: string;
  audioRhythmGuidance: string | null;
  continuityGuidance: string;
  evidence: string[];
};

export type AudioVisualFallbackSceneStrategySummary = {
  fallbackPattern: AudioVisualScenePattern;
  preservedPhases: SymbolicNarrativePhaseName[];
  reducedElements: string[];
  simplificationStrategy: string;
  promptGuidance: string[];
};

export type AudioVisualSceneProfileSummary = {
  role: "audio_visual_scene_system";
  scenePattern: AudioVisualScenePattern;
  sceneArc: string;
  scenePhases: AudioVisualScenePhaseSummary[];
  openingScene: AudioVisualScenePhaseSummary;
  developmentScene: AudioVisualScenePhaseSummary;
  thresholdScene: AudioVisualScenePhaseSummary;
  climaxScene: AudioVisualScenePhaseSummary;
  resolutionScene: AudioVisualScenePhaseSummary;
  cuePlan: AudioVisualSceneCueSummary[];
  transitionPlan: AudioVisualSceneTransitionSummary[];
  climaxStrategy: string;
  resolutionStrategy: string;
  visualTimingPlan: string[];
  motionTimingPlan: string[];
  audioTimingPlan: string[];
  rhythmTimingPlan: string[];
  cameraTimingPlan: string[];
  motifTimingPlan: string[];
  emotionalTimingPlan: string[];
  proceduralTimingPlan: string[];
  synchronizationCheckpoints: string[];
  sceneContrastPlan: string[];
  sceneContinuityPlan: string[];
  sceneRisks: string[];
  pacingRisks: string[];
  overloadRisks: string[];
  fallbackSceneStrategy: AudioVisualFallbackSceneStrategySummary;
  unresolvedSceneGaps: string[];
  hitlQuestions: string[];
  promptGuidance: string[];
  authorityBoundary: string;
  evidence: string[];
};

export type ArtifactType =
  | "runnable_code"
  | "design_spec"
  | "explanation"
  | "debug_patch"
  | "review_report"
  | "refinement_patch"
  | "preview_request";

export type ArtifactFamily =
  | "p5_sketch"
  | "three_scene"
  | "react_three_fiber_scene"
  | "glsl_shader"
  | "hydra_patch"
  | "tone_sketch"
  | "canvas_sketch"
  | "audiovisual_scene"
  | "generative_artifact"
  | "multimodal_reference_artifact"
  | "creative_coding_response";

export type ArtifactPlanSummary = {
  role: "artifact_planner";
  primaryArtifactIntent: string;
  artifactType: ArtifactType;
  artifactFamily: ArtifactFamily;
  requiredComponents: string[];
  runtimeRequirements: string[];
  creativeDependencies: string[];
  generativeDependencies: string[];
  expectedOutputStructure: string[];
  implementationRisks: string[];
  missingInformation: string[];
  hitlQuestions: string[];
  promptGuidance: string[];
  authorityBoundary: string;
  evidence: string[];
};

export type DependencyNodeType =
  | "planned_artifact"
  | "required_component"
  | "runtime_requirement"
  | "creative_metadata"
  | "generative_metadata"
  | "output_structure"
  | "prompt_guidance"
  | "downstream_consumer";

export type DependencyNodeStatus = "available" | "inferred" | "missing";

export type DependencyRelationship =
  | "requires"
  | "informs"
  | "blocks"
  | "soft_informs"
  | "feeds_prompt"
  | "consumed_by";

export type DependencyStrength =
  | "required"
  | "optional"
  | "blocking"
  | "soft";

export type ArtifactDependencyNodeSummary = {
  nodeId: string;
  label: string;
  nodeType: DependencyNodeType;
  status: DependencyNodeStatus;
  summary: string;
  evidence: string[];
};

export type ArtifactDependencyEdgeSummary = {
  sourceNodeId: string;
  targetNodeId: string;
  relationship: DependencyRelationship;
  strength: DependencyStrength;
  rationale: string;
};

export type ArtifactDependencyGraphSummary = {
  role: "artifact_dependency_graph";
  primaryArtifactNodeId: string;
  artifactNodes: ArtifactDependencyNodeSummary[];
  dependencyEdges: ArtifactDependencyEdgeSummary[];
  requiredUpstreamMetadata: string[];
  optionalUpstreamMetadata: string[];
  blockingDependencies: string[];
  softDependencies: string[];
  runtimeFacingDependencies: string[];
  promptFacingDependencies: string[];
  downstreamConsumers: string[];
  missingDependencyRisks: string[];
  dependencyConflicts: string[];
  hitlQuestions: string[];
  promptGuidance: string[];
  authorityBoundary: string;
  evidence: string[];
};

export type CreativeConstraintAxis =
  | "intent"
  | "modality"
  | "runtime"
  | "safety"
  | "performance"
  | "complexity"
  | "cost"
  | "hitl"
  | "output_goal";

export type CreativeConstraintSeverity = "info" | "watch" | "risk" | "blocking";

export type CreativeConstraintPressure = "low" | "medium" | "high";

export type CreativeConstraintSummary = {
  axis: CreativeConstraintAxis;
  severity: CreativeConstraintSeverity;
  summary: string;
  recommendation: string;
  evidence: string[];
};

export type CreativeConstraintTradeoffSummary = {
  sourceAxis: CreativeConstraintAxis;
  targetAxis: CreativeConstraintAxis;
  severity: CreativeConstraintSeverity;
  summary: string;
  recommendation: string;
};

export type CreativeConstraintSolverSummary = {
  role: "creative_constraint_solver";
  intentSummary: string;
  outputGoal: string;
  modality: string | null;
  runtimeFit: "supported" | "code_only" | "undetermined";
  recommendedRuntime: string | null;
  complexityPressure: CreativeConstraintPressure;
  safetyPressure: CreativeConstraintPressure;
  performancePressure: CreativeConstraintPressure;
  costPressure: CreativeConstraintPressure;
  hitlAdvisable: boolean;
  hitlReason: string | null;
  activeConstraints: CreativeConstraintSummary[];
  tradeoffs: CreativeConstraintTradeoffSummary[];
  conflicts: string[];
  promptGuidance: string[];
  authorityBoundary: string;
  evidence: string[];
};

export type CreativeConstraintPriorityCategory =
  | "symbolic_fidelity"
  | "narrative_fidelity"
  | "emotional_fidelity"
  | "geometric_fidelity"
  | "visual_quality"
  | "motion_quality"
  | "audio_quality"
  | "runtime_safety"
  | "previewability"
  | "performance"
  | "implementation_simplicity"
  | "cost_sensitivity"
  | "interaction_complexity"
  | "maintainability";

export type CreativeConstraintPriorityLevel =
  | "non_negotiable"
  | "high_priority"
  | "flexible"
  | "relaxable"
  | "sacrificial";

export type CreativeConstraintPrioritySource =
  | "explicit"
  | "hierarchy"
  | "solver"
  | "runtime"
  | "tradeoff"
  | "coherence";

export type CreativeConstraintPrioritySummary = {
  category: CreativeConstraintPriorityCategory;
  priorityLevel: CreativeConstraintPriorityLevel;
  rank: number;
  priorityScore: number;
  source: CreativeConstraintPrioritySource;
  rationale: string;
  negotiationGuidance: string;
  evidence: string[];
};

export type CreativeConstraintPriorityConflictSummary = {
  protectedCategory: CreativeConstraintPriorityCategory;
  competingCategory: CreativeConstraintPriorityCategory;
  severity: CreativeConstraintSeverity;
  summary: string;
  negotiationNote: string;
  hitlRecommended: boolean;
};

export type CreativeConstraintPrioritizationSummary = {
  role: "creative_constraint_prioritizer";
  nonNegotiableConstraints: CreativeConstraintPrioritySummary[];
  highPriorityConstraints: CreativeConstraintPrioritySummary[];
  flexibleConstraints: CreativeConstraintPrioritySummary[];
  relaxableConstraints: CreativeConstraintPrioritySummary[];
  sacrificialConstraints: CreativeConstraintPrioritySummary[];
  priorityRationale: string[];
  negotiationNotes: string[];
  conflictRelationships: CreativeConstraintPriorityConflictSummary[];
  hitlQuestions: string[];
  promptGuidance: string[];
  authorityBoundary: string;
  evidence: string[];
};

export type CreativeAssistantDirectorSummary = {
  role: "creative_assistant_director";
  creativeBrief: string;
  ambiguityLevel: "low" | "medium" | "high";
  ambiguitySignals: string[];
  retrievalPosture: "not_requested" | "useful" | "available";
  modalityDirection: string | null;
  runtimeDirection: string | null;
  planningFocus: string[];
  critiqueFocus: string[];
  refinementFocus: string[];
  nextActions: string[];
  hitlRequired: boolean;
  hitlReason: string | null;
  authorityBoundary: string;
  evidence: string[];
};

export type CreativeReasoningStage =
  | "strategy"
  | "technique"
  | "runtime"
  | "tradeoff"
  | "recommendation";

export type CreativeReasoningEvidenceSource =
  | "request"
  | "translation"
  | "creative_intent"
  | "creative_hierarchy"
  | "planning"
  | "director"
  | "constraint_solver"
  | "constraint_prioritizer"
  | "creative_strategy"
  | "creative_technique"
  | "runtime_capability"
  | "tradeoff_explorer"
  | "quality_predictor"
  | "symbolic_narrative"
  | "creative_composition"
  | "procedural_structure"
  | "generative_structure"
  | "semantic_motif"
  | "emotional_consistency"
  | "cross_modality"
  | "audio_visual_scene"
  | "artifact_plan"
  | "artifact_dependency_graph"
  | "runtime_compatibility"
  | "artifact_capability_matrix"
  | "multi_artifact_strategy"
  | "artifact_critic"
  | "artifact_refiner"
  | "artifact_intelligence_synthesis"
  | "artifact_merge_planner"
  | "artifact_export_intelligence"
  | "creative_critic"
  | "self_evaluation"
  | "creative_improvement_planner"
  | "reflection_loop"
  | "creative_confidence"
  | "creative_score"
  | "future_knowledge";

export type CreativeReasoningStepSummary = {
  stage: CreativeReasoningStage;
  claim: string;
  because: string;
  implications: string[];
};

export type CreativeReasoningEvidenceSummary = {
  source: CreativeReasoningEvidenceSource;
  signal: string;
  interpretation: string;
};

export type CreativeRejectedAlternativeSummary = {
  alternative: string;
  reason: string;
  evidence: string[];
};

export type CreativeReasoningSummary = {
  role: "creative_reasoning_engine";
  recommendedCreativeDirection: string;
  reasoningPath: CreativeReasoningStepSummary[];
  evidenceChain: CreativeReasoningEvidenceSummary[];
  strongestSupportingSignals: string[];
  rejectedAlternatives: CreativeRejectedAlternativeSummary[];
  unresolvedDecisions: string[];
  implementationGuidance: string[];
  promptGuidance: string[];
  hitlQuestions: string[];
  futureKnowledgeContext: Record<string, unknown>;
  authorityBoundary: string;
};

export type AssistantMessage = {
  role: "user" | "assistant";
  time: string;
  content: string;
};

export type ImageAttachmentSummary = {
  id: string;
  kind: "image";
  name: string;
  mimeType: string;
  sizeBytes: number;
  dataUrl: string;
  createdAt: string;
};

export type MultimodalSummary = {
  state: "empty" | "ready" | "error";
  status: string;
  detail: string;
  imageAttachments: ImageAttachmentSummary[];
  error?: WorkstationError | null;
};

export type ArtifactAction =
  | "Open"
  | "Preview"
  | "Copy"
  | "Download"
  | "Export";

export type RefinementPassStopReason =
  | "continue_available"
  | "quality_improved"
  | "no_useful_opportunities"
  | "runtime_preview_safety_failed"
  | "max_passes_reached";

export type RefinementPassRecord = {
  passNumber: number;
  sourceArtifactId: string;
  sourceArtifactTitle?: string | null;
  resultArtifactId?: string | null;
  resultArtifactTitle?: string | null;
  refinementObjective: string;
  qualityBefore?: number | null;
  qualityAfter?: number | null;
  stopReason: RefinementPassStopReason;
  summary: string;
};

export type ArtifactSummary = {
  id: string;
  title: string;
  type: "code" | "preview" | "export";
  language: string;
  status: string;
  summary: string;
  content?: string;
  creativeTranslation?: CreativeTranslationSummary | null;
  creativePlan?: CreativeExecutionPlanSummary | null;
  domain?: string | null;
  isDefault?: boolean;
  previewEligible?: boolean;
  previewTarget?: PreviewTargetId | "";
  rendererId?: string | null;
  runtime?: string | null;
  sourceOrder?: number;
  qualityScore?: number | null;
  qualityRank?: number | null;
  isRecommended?: boolean;
  refinementReason?: string | null;
  refinedAt?: string | null;
  refinedFromArtifactId?: string | null;
  refinedFromTitle?: string | null;
  refinementInstruction?: string | null;
  refinementPasses?: RefinementPassRecord[];
  critique?: ArtifactCritique;
  actions: ArtifactAction[];
};

export type CreativeTranslationSummary = {
  outputModality: "visual" | "audio" | "audiovisual" | null;
  creativeIntent: string;
  symbolicReferences: string[];
  geometricReferences: string[];
  musicalReferences: string[];
  moodAtmosphere: string[];
  movementLanguage: string[];
  colorMaterialDirection: string[];
  runtimeRecommendations: string[];
  structureDirection: string[];
  generationConstraints: string[];
  refinementTargets: string[];
  sacredGeometry?: SacredGeometrySummary | null;
  shaderPresets?: ShaderPresetSummary | null;
  visualStyle?: VisualStyleSummary | null;
  audioReactive?: AudioReactiveGuidanceSummary | null;
  referenceFusion?: ReferenceFusionSummary | null;
};

export type ReferenceFusionSummary = {
  sourceCount: number;
  sourceNames: string[];
  paletteDirection: string[];
  composition: string[];
  lightingContrast: string[];
  textureMaterialCues: string[];
  geometricStructure: string[];
  moodAtmosphere: string[];
  motionImplications: string[];
  runtimeStyleImplications: string[];
  safetyConstraints: string[];
  summary: string;
};

export type AudioReactiveSource =
  | "amplitude"
  | "bass"
  | "mids"
  | "highs"
  | "rhythm"
  | "envelope"
  | "drone_intensity";

export type AudioReactiveVisualTarget =
  | "scale"
  | "glow"
  | "brightness"
  | "pulse"
  | "expansion"
  | "camera_movement"
  | "color_shift"
  | "texture_modulation"
  | "sparkle"
  | "particles"
  | "detail"
  | "rotation"
  | "pattern_phase"
  | "scene_transitions"
  | "opacity"
  | "bloom"
  | "geometry_emergence"
  | "fog"
  | "aura"
  | "field_density";

export type AudioReactiveMappingSummary = {
  source: AudioReactiveSource;
  targets: AudioReactiveVisualTarget[];
  intensity: "subtle" | "balanced" | "strong";
  behavior: string;
  evidence: string[];
};

export type AudioReactiveGuidanceSummary = {
  mappings: AudioReactiveMappingSummary[];
  audioRuntime: string | null;
  visualRuntime: string | null;
  activation: "explicit_user_gesture";
  summary: string;
};

export type SacredGeometrySummary = {
  concepts: string[];
  geometricStructure: string[];
  symmetryType: string[];
  movementBehavior: string[];
  visualComposition: string[];
  colorMaterialDirection: string[];
  runtimeRecommendations: string[];
  audioImplications: string[];
  generationConstraints: string[];
};

export type ShaderPresetName =
  | "glow"
  | "aura"
  | "plasma"
  | "bloom-like emission"
  | "refraction"
  | "glass / crystal"
  | "volumetric atmosphere"
  | "fractal field"
  | "kaleidoscopic symmetry"
  | "sacred light / ritual ambience";

export type ShaderPresetSummary = {
  presets: ShaderPresetName[];
  colorBehavior: string[];
  lightMaterialBehavior: string[];
  motionBehavior: string[];
  shaderStructure: string[];
  runtimeSuitability: string[];
  performanceConstraints: string[];
};

export type VisualStyleName =
  | "minimal"
  | "cyberpunk"
  | "organic"
  | "ritual"
  | "sacred geometry"
  | "generative modernism"
  | "retro computational"
  | "ethereal"
  | "psychedelic"
  | "architectural"
  | "monochrome"
  | "maximalist";

export type VisualStyleSummary = {
  styles: VisualStyleName[];
  paletteBehavior: string[];
  contrastBehavior: string[];
  compositionTendencies: string[];
  motionTendencies: string[];
  textureTendencies: string[];
  spatialOrganization: string[];
  runtimeSuitability: string[];
};

export type ArtifactCritiqueDimension = {
  score: number;
  rationale: string;
};

export type CreativeQualityLevel = "strong" | "developing" | "weak";

export type CreativeQualityObservation = {
  score: number;
  level: CreativeQualityLevel;
  observation: string;
  evidence: string[];
};

export type CreativeQualityEvaluation = {
  overallScore: number;
  composition: CreativeQualityObservation;
  originality: CreativeQualityObservation;
  coherence: CreativeQualityObservation;
  aestheticConsistency: CreativeQualityObservation;
  expressiveness: CreativeQualityObservation;
  strengths: string[];
  refinementOpportunities: string[];
  summary: string;
};

export type SacredConsistencyLevel = "aligned" | "partial" | "unsupported";

export type SacredConsistencyObservation = {
  score: number;
  level: SacredConsistencyLevel;
  observation: string;
  evidence: string[];
};

export type SacredConsistencyEvaluation = {
  overallScore: number;
  alignment: SacredConsistencyObservation;
  motifConsistency: SacredConsistencyObservation;
  modalityCoherence: SacredConsistencyObservation;
  claimSafety: SacredConsistencyObservation;
  strengths: string[];
  refinementOpportunities: string[];
  summary: string;
};

export type CalibratedQualitySignalKey =
  | "legacy_critique"
  | "creative_quality"
  | "sacred_consistency"
  | "runtime_preview"
  | "refinement_pressure"
  | "grounding";

export type CalibratedQualitySignal = {
  key: CalibratedQualitySignalKey;
  label: string;
  score: number;
  weight: number;
  rationale: string;
};

export type CalibratedQualityDecisionBand =
  | "strong_candidate"
  | "usable_candidate"
  | "needs_refinement"
  | "high_risk";

export type CalibratedQualityEvaluation = {
  score: number;
  legacyScore: number;
  decisionBand: CalibratedQualityDecisionBand;
  confidence: "high" | "medium" | "low";
  signals: CalibratedQualitySignal[];
  adjustments: string[];
  rationale: string;
  summary: string;
};

export type ArtifactCritique = {
  artifactId: string;
  artifactTitle: string;
  sourceOrder: number;
  overallScore: number;
  rank: number;
  passed: boolean;
  recommended: boolean;
  promptAlignment: ArtifactCritiqueDimension;
  creativeQuality: ArtifactCritiqueDimension;
  runtimeSuitability: ArtifactCritiqueDimension;
  codeQuality: ArtifactCritiqueDimension;
  previewReadiness: ArtifactCritiqueDimension;
  domainAppropriateness: ArtifactCritiqueDimension;
  creativeEvaluation?: CreativeQualityEvaluation | null;
  sacredConsistency?: SacredConsistencyEvaluation | null;
  calibratedQuality?: CalibratedQualityEvaluation | null;
  legacyRank?: number | null;
  reasons: string[];
  rationale: string;
  refinementGuidance: string | null;
};

export type PreviewTargetId =
  | "browser_sandbox"
  | "image_asset"
  | "audio_asset"
  | "video_asset"
  | "text_panel"
  | "json_panel";

export type PreviewSummary = {
  available: boolean;
  active: boolean;
  collapsed: boolean;
  state: "generating" | "ready" | "unavailable" | "error";
  title: string;
  targetId: PreviewTargetId | "";
  target: string;
  status: string;
  artifactName: string;
  sourceArtifactId: string;
  sourceArtifactName: string;
  outputArtifactName: string;
  summary: string;
  renderer: string;
  trigger: string;
  version: string;
  error?: WorkstationError | null;
};

export type CodeSummary = {
  title: string;
  language: string;
  status: string;
  excerpt: string[];
};

export type RetrievalState =
  | "available"
  | "pending"
  | "empty"
  | "unavailable"
  | "error";

export type RetrievalQuality = "high" | "medium" | "low" | "unknown";

export type RetrievalFreshness = "fresh" | "stale" | "unknown";

export type RetrievalSourceHealthStatus =
  | "healthy"
  | "warning"
  | "stale"
  | "failed"
  | "unknown";

export type RetrievalSourceAvailability =
  | "available"
  | "degraded"
  | "unavailable"
  | "unknown";

export type RetrievalSourceSyncOutcome =
  | "succeeded"
  | "failed"
  | "pending"
  | "unknown";

export type RetrievalSourceHealthMetadata = {
  status?: RetrievalSourceHealthStatus | "sync_failed" | null;
  freshnessStatus?: RetrievalFreshness | null;
  availability?: RetrievalSourceAvailability | null;
  domainOwner?: string | null;
  indexedChunkCount?: number | null;
  lastSuccessfulSyncAt?: string | null;
  lastAttemptedSyncAt?: string | null;
  syncOutcome?: RetrievalSourceSyncOutcome | null;
  refreshRecommended?: boolean | null;
  checkedAt?: string | null;
  warnings?: string[] | null;
};

export type RetrievalChunkSummary = {
  id: string;
  chunkIndex: number;
  score: number | null;
  snippet: string;
  relevanceLabel: string;
  rank?: number | null;
  originalScore?: number | null;
  scoreAdjustment?: number | null;
  domainMatch?: boolean | null;
  selectionReason?: string | null;
  usedInContext?: boolean | null;
};

export type RetrievalSourceSummary = {
  sourceId: string;
  title: string;
  detail: string;
  domain: string;
  domainLabel: string;
  publisher: string;
  sourceType: string;
  sourceTypeLabel: string;
  href: string;
  host: string;
  score: number | null;
  quality: RetrievalQuality;
  qualityLabel: string;
  freshness: RetrievalFreshness;
  freshnessLabel: string;
  updatedAt: string | null;
  whyUsed: string;
  chunks: RetrievalChunkSummary[];
  bestRank?: number | null;
  selectedForContext?: boolean | null;
  health?: RetrievalSourceHealthMetadata | null;
};

export type RetrievalSummary = {
  state: RetrievalState;
  status: string;
  headline: string;
  detail: string;
  source: string;
  query: string | null;
  requestedDomains: string[];
  warning: string | null;
  sources: RetrievalSourceSummary[];
  error?: WorkstationError | null;
};

export type DebugEventSummary = {
  code: string;
  label: string;
  detail: string;
};

export type AssistantWorkspaceSession = {
  userId: string;
  sessionId: string;
  projectId: string;
  title: string;
  updatedAt?: string;
};

export type AssistantWorkspaceSnapshot = {
  session: AssistantWorkspaceSession;
  workspace: {
    name: string;
    focus: string;
  };
  inspectorTabs: InspectorTabState[];
  messages: AssistantMessage[];
  workflow: {
    status: string;
    currentNode: WorkflowNodeId;
    currentStep: string;
    steps: WorkflowStepState[];
  };
  artifacts: ArtifactSummary[];
  clarification?: ClarificationSummary | null;
  creativePlan?: CreativeExecutionPlanSummary | null;
  multimodal: MultimodalSummary;
  preview: PreviewSummary;
  code: CodeSummary;
  retrieval: RetrievalSummary;
  debug: {
    traceId: string;
    status: string;
    events: DebugEventSummary[];
  };
};

export type AssistantFrontendClient = {
  getWorkspaceSnapshot: () => Promise<AssistantWorkspaceSnapshot>;
};

export function createAssistantClient(): AssistantFrontendClient {
  return {
    async getWorkspaceSnapshot() {
      return getInitialWorkspaceSnapshot();
    }
  };
}

export function getInitialWorkspaceSnapshot(): AssistantWorkspaceSnapshot {
  return {
    session: {
      userId: "local-user",
      sessionId: "local-nextjs-session",
      projectId: "local-nextjs-workspace",
      title: "Creative workspace"
    },
    workspace: {
      name: "Creative workspace",
      focus: "Start a creative coding session"
    },
    inspectorTabs: [
      {
        label: "Overview",
        active: true,
        summary: "Session plan, readiness, and compact workflow state",
        badge: "Ready"
      },
      {
        label: "Preview",
        active: false,
        summary: "Canvas appears after a runnable artifact is generated"
      },
      {
        label: "Runtime",
        active: false,
        summary: "Live runtime diagnostics appear after the preview renderer starts"
      },
      {
        label: "Code",
        active: false,
        summary: "Generated source appears after the first creative pass"
      },
      {
        label: "Workflow",
        active: false,
        summary: "LangGraph orchestration view for active runs"
      },
      {
        label: "Telemetry",
        active: false,
        summary: "Runtime and provider signals for active runs"
      },
      {
        label: "Artifacts",
        active: false,
        summary: "Generated files and exports"
      },
      {
        label: "Retrieval",
        active: false,
        summary: "Reference grounding for creative requests"
      }
    ],
    messages: [],
    workflow: {
      status: "Idle",
      currentNode: "intake",
      currentStep: "Ready to start",
      steps: [
        {
          nodeId: "intake",
          displayLabel: "Intake",
          state: "queued",
          detail: "Capture the creative brief when you send the first prompt."
        },
        {
          nodeId: "routing",
          displayLabel: "Routing",
          state: "queued",
          detail: "Choose the generation path for the request."
        },
        {
          nodeId: "memory",
          displayLabel: "Memory",
          state: "queued",
          detail: "Apply relevant session memory when available."
        },
        {
          nodeId: "retrieval",
          displayLabel: "Retrieval",
          state: "queued",
          detail: "Gather grounded references when the request needs them."
        },
        {
          nodeId: "context_assembly",
          displayLabel: "Context assembly",
          state: "queued",
          detail: "Prepare memory and retrieval context for generation."
        },
        {
          nodeId: "prompt_input",
          displayLabel: "Prompt input",
          state: "queued",
          detail: "Structure prompt inputs for the provider request."
        },
        {
          nodeId: "planning",
          displayLabel: "Planning",
          state: "queued",
          detail: "Prepare a deterministic execution plan before generation."
        },
        {
          nodeId: "director",
          displayLabel: "Director",
          state: "queued",
          detail: "Prepare bounded creative decision support for the run."
        },
        {
          nodeId: "reasoning",
          displayLabel: "Reasoning",
          state: "queued",
          detail: "Synthesize creative intelligence into a decision brief."
        },
        {
          nodeId: "prompt_rendering",
          displayLabel: "Prompt rendering",
          state: "queued",
          detail: "Render the provider prompt."
        },
        {
          nodeId: "generation",
          displayLabel: "Generation",
          state: "queued",
          detail: "Generate the creative code or response."
        },
        {
          nodeId: "artifact_extraction",
          displayLabel: "Artifact extraction",
          state: "queued",
          detail: "Normalize generated output into workspace artifacts."
        },
        {
          nodeId: "preview_preparation",
          displayLabel: "Preview preparation",
          state: "queued",
          detail: "Prepare preview routing for runnable artifacts."
        },
        {
          nodeId: "artifact_critique",
          displayLabel: "Artifact critique",
          state: "queued",
          detail: "Score and rank generated artifacts before answer review."
        },
        {
          nodeId: "review",
          displayLabel: "Review",
          state: "queued",
          detail: "Check the generated result before finalization."
        },
        {
          nodeId: "refinement",
          displayLabel: "Refinement",
          state: "queued",
          detail: "Run one refinement loop when review asks for changes."
        },
        {
          nodeId: "finalization",
          displayLabel: "Finalization",
          state: "queued",
          detail: "Emit the final response and updated workspace state."
        },
        {
          nodeId: "failure",
          displayLabel: "Failure",
          state: "branch",
          detail: "Terminal branch used only when a workflow node fails."
        }
      ]
    },
    artifacts: [],
    multimodal: {
      state: "empty",
      status: "No image references",
      detail: "Attach image references when a visual brief needs palette, mood, or composition guidance.",
      imageAttachments: [],
      error: null
    },
    preview: {
      available: false,
      active: false,
      collapsed: true,
      state: "unavailable",
      title: "Preview ready when output exists",
      targetId: "",
      target: "",
      status: "Waiting for artifact",
      artifactName: "No preview yet",
      sourceArtifactId: "",
      sourceArtifactName: "",
      outputArtifactName: "",
      summary:
        "Generate a runnable sketch and the preview shelf will dock below the session.",
      renderer: "",
      trigger: "",
      version: "v1"
    },
    code: {
      title: "No artifact yet",
      language: "Creative code",
      status: "Awaiting first artifact",
      excerpt: ["// Generated code appears here after your first creative request."]
    },
    retrieval: {
      state: "empty",
      status: "Ready",
      headline: "No references loaded yet",
      detail:
        "Reference grounding appears here when a request benefits from documentation or source context.",
      source: "",
      query: null,
      requestedDomains: [],
      warning: null,
      sources: []
    },
    debug: {
      traceId: "trace.local.first-run",
      status: "Ready",
      events: []
    }
  };
}

export function getLocalWorkspaceSnapshot(): AssistantWorkspaceSnapshot {
  return {
    session: {
      userId: "local-user",
      sessionId: "local-nextjs-session",
      projectId: "local-nextjs-workspace",
      title: "Session workspace"
    },
    workspace: {
      name: "Session workspace",
      focus: "p5 aurora field"
    },
    inspectorTabs: [
      {
        label: "Overview",
        active: true,
        summary: "Live creative session summary",
        badge: "Live"
      },
      {
        label: "Preview",
        active: false,
        summary: "Canvas runtime and renderer context",
        badge: "Run"
      },
      {
        label: "Runtime",
        active: false,
        summary: "Focused runtime diagnostics and lifecycle history",
        badge: "Diag"
      },
      {
        label: "Code",
        active: false,
        summary: "Generated sketch source",
        badge: "JS"
      },
      {
        label: "Workflow",
        active: false,
        summary: "LangGraph-style orchestration",
        badge: "Running"
      },
      {
        label: "Telemetry",
        active: false,
        summary: "Operator observability console",
        badge: "Ops"
      },
      {
        label: "Artifacts",
        active: false,
        summary: "Generated outputs",
        badge: "3"
      },
      {
        label: "Retrieval",
        active: false,
        summary: "Creative references",
        badge: "2"
      }
    ],
    messages: [
      {
        role: "user",
        time: "09:24",
        content:
          "Build a luminous particle field that reacts to low-frequency audio and keeps motion legible on a projection wall."
      },
      {
        role: "assistant",
        time: "09:25",
        content:
          "Drafting a p5.js sketch with stable motion, palette controls, and an artifact that can be opened or previewed on demand."
      }
    ],
    workflow: {
      status: "Running",
      currentNode: "generation",
      currentStep: "Generation",
      steps: [
        {
          nodeId: "intake",
          displayLabel: "Intake",
          state: "complete",
          detail: "Request received and normalized."
        },
        {
          nodeId: "routing",
          displayLabel: "Routing",
          state: "complete",
          detail: "Generate route selected."
        },
        {
          nodeId: "memory",
          displayLabel: "Memory",
          state: "skipped",
          detail: "No local session memories applied."
        },
        {
          nodeId: "retrieval",
          displayLabel: "Retrieval",
          state: "complete",
          detail: "Creative references resolved."
        },
        {
          nodeId: "context_assembly",
          displayLabel: "Context assembly",
          state: "complete",
          detail: "Memory and retrieval context prepared."
        },
        {
          nodeId: "prompt_input",
          displayLabel: "Prompt input",
          state: "complete",
          detail: "Prompt inputs structured for rendering."
        },
        {
          nodeId: "planning",
          displayLabel: "Planning",
          state: "complete",
          detail: "Execution strategy and runtime plan prepared."
        },
        {
          nodeId: "director",
          displayLabel: "Director",
          state: "complete",
          detail: "Creative Assistant Director guidance prepared."
        },
        {
          nodeId: "reasoning",
          displayLabel: "Reasoning",
          state: "complete",
          detail: "Creative reasoning synthesis prepared."
        },
        {
          nodeId: "prompt_rendering",
          displayLabel: "Prompt rendering",
          state: "complete",
          detail: "Provider prompt assembled."
        },
        {
          nodeId: "generation",
          displayLabel: "Generation",
          state: "active",
          detail: "Generated sketch artifact is being drafted."
        },
        {
          nodeId: "artifact_extraction",
          displayLabel: "Artifact extraction",
          state: "queued",
          detail: "Generated code will be normalized into workflow artifacts."
        },
        {
          nodeId: "preview_preparation",
          displayLabel: "Preview preparation",
          state: "queued",
          detail: "Preview runtime metadata will be prepared for runnable artifacts."
        },
        {
          nodeId: "artifact_critique",
          displayLabel: "Artifact critique",
          state: "queued",
          detail: "Generated artifacts will be scored and ranked."
        },
        {
          nodeId: "review",
          displayLabel: "Review",
          state: "queued",
          detail: "Internal quality gate before finalization."
        },
        {
          nodeId: "refinement",
          displayLabel: "Refinement",
          state: "queued",
          detail: "Retry loop back to generation if review needs changes."
        },
        {
          nodeId: "finalization",
          displayLabel: "Finalization",
          state: "queued",
          detail: "Final response emitted when workflow completes."
        },
        {
          nodeId: "failure",
          displayLabel: "Failure",
          state: "branch",
          detail: "Terminal branch used only when a graph node fails."
        }
      ]
    },
    artifacts: [
      {
        id: "source-sketch",
        title: "aurora-field.p5.js",
        type: "code",
        language: "p5.js",
        status: "Ready",
        summary: "Primary generated p5 sketch artifact with a browser preview target.",
        actions: ["Open", "Preview", "Copy", "Download"]
      },
      {
        id: "preview-manifest",
        title: "preview-request.json",
        type: "preview",
        language: "JSON",
        status: "Queued",
        summary: "Renderer identity, browser preview target, and artifact v1 linkage.",
        actions: ["Open", "Preview", "Download"]
      },
      {
        id: "session-notes",
        title: "projection-notes.md",
        type: "export",
        language: "Markdown",
        status: "Ready",
        summary: "Projection scale, motion density, and palette constraints.",
        actions: ["Open", "Export"]
      }
    ],
    multimodal: {
      state: "empty",
      status: "No image references",
      detail:
        "Attach image references to ground the next creative coding request visually.",
      imageAttachments: [],
      error: null
    },
    preview: {
      available: true,
      active: false,
      collapsed: true,
      state: "ready",
      title: "Preview available",
      targetId: "browser_sandbox",
      target: "Browser preview / p5.js",
      status: "Ready",
      artifactName: "aurora-field.p5.js",
      sourceArtifactId: "source-sketch",
      sourceArtifactName: "aurora-field.p5.js",
      outputArtifactName: "",
      summary:
        "Runtime context is ready for the generated p5 sketch. Open the preview shelf to render it in the browser preview.",
      renderer: "surface.p5",
      trigger: "Workflow Generation",
      version: "v1"
    },
    code: {
      title: "aurora-field.p5.js",
      language: "p5.js",
      status: "Ready artifact",
      excerpt: [
        "let phase = 0;",
        "function setup() {",
        "  createCanvas(windowWidth, 320);",
        "  noStroke();",
        "}",
        "function draw() {",
        "  phase += 0.012;",
        "  background(5, 8, 11);",
        "  for (let i = 0; i < 18; i += 1) {",
        "    const x = map(i, 0, 17, 36, width - 36);",
        "    const y = height * 0.5 + sin(phase + i * 0.52) * 74;",
        "    fill(76 + i * 4, 215, 200, 160);",
        "    circle(x, y, 18 + sin(phase * 1.7 + i) * 8);",
        "  }",
        "}"
      ]
    },
    retrieval: {
      state: "available",
      status: "Grounded",
      headline: "3 chunks from 2 official sources",
      detail:
        "Official knowledge base context grounded the WebGPU particle-field draft before code generation and preview routing.",
      source: "official_kb",
      query:
        "Stable WebGPU particle field for a projection wall with low-frequency audio response",
      requestedDomains: ["webgpu_wgsl", "glsl"],
      warning:
        "1 source is older than the preferred refresh window for shader guidance.",
      sources: [
        {
          sourceId: "webgpu_mdn_api",
          title: "WebGPU API",
          detail:
            "Stable compute and render pass separation guidance for the browser preview renderer.",
          domain: "webgpu_wgsl",
          domainLabel: "WebGPU / WGSL",
          publisher: "MDN",
          sourceType: "api_reference",
          sourceTypeLabel: "API reference",
          href: "https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API",
          host: "developer.mozilla.org",
          score: 0.91,
          quality: "high",
          qualityLabel: "High relevance",
          freshness: "fresh",
          freshnessLabel: "Current",
          updatedAt: "2026-05-20T08:30:00Z",
          whyUsed:
            "Matched the request for stable compute and render pass separation in a browser preview.",
          chunks: [
            {
              id: "webgpu_mdn_api::chunk-0001",
              chunkIndex: 0,
              score: 0.91,
              relevanceLabel: "Best match",
              snippet:
                "The WebGPU API separates device setup, command encoding, and queue submission so compute and render work can stay isolated."
            },
            {
              id: "webgpu_mdn_api::chunk-0004",
              chunkIndex: 3,
              score: 0.84,
              relevanceLabel: "Supporting match",
              snippet:
                "GPUCanvasContext configuration should be applied once per presentation surface so preview updates stay predictable during iteration."
            }
          ],
          health: {
            status: "healthy",
            freshnessStatus: "fresh",
            availability: "available",
            domainOwner: "Web platform / MDN",
            indexedChunkCount: 184,
            lastSuccessfulSyncAt: "2026-05-20T08:30:00Z",
            lastAttemptedSyncAt: "2026-05-20T08:30:00Z",
            syncOutcome: "succeeded",
            refreshRecommended: false,
            checkedAt: "2026-06-09T08:30:00Z",
            warnings: []
          }
        },
        {
          sourceId: "glsl_language_spec_460",
          title: "OpenGL Shading Language 4.60 Specification",
          detail:
            "Lower-level shader typing and buffer-layout grounding for deterministic output.",
          domain: "glsl",
          domainLabel: "GLSL",
          publisher: "Khronos Group",
          sourceType: "specification",
          sourceTypeLabel: "Specification",
          href:
            "https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.60.html",
          host: "registry.khronos.org",
          score: 0.73,
          quality: "medium",
          qualityLabel: "Relevant grounding",
          freshness: "stale",
          freshnessLabel: "Review soon",
          updatedAt: "2025-10-12T09:00:00Z",
          whyUsed:
            "Provided lower-level shader language grounding for deterministic buffer layout and fragment output.",
          chunks: [
            {
              id: "glsl_language_spec_460::chunk-0003",
              chunkIndex: 2,
              score: 0.73,
              relevanceLabel: "Supporting match",
              snippet:
                "Explicit shader types and layout-compatible data flow keep buffer-backed particle pipelines deterministic across stages."
            }
          ],
          health: {
            status: "stale",
            freshnessStatus: "stale",
            availability: "available",
            domainOwner: "Graphics standards / Khronos Group",
            indexedChunkCount: 96,
            lastSuccessfulSyncAt: "2025-10-12T09:00:00Z",
            lastAttemptedSyncAt: "2025-10-12T09:00:00Z",
            syncOutcome: "succeeded",
            refreshRecommended: true,
            checkedAt: "2026-06-09T08:30:00Z",
            warnings: ["Source exceeds the preferred shader guidance refresh window."]
          }
        }
      ]
    },
    debug: {
      traceId: "trace.local.nextjs-foundation",
      status: "Contextual",
      events: [
        {
          code: "route_selected",
          label: "Route",
          detail: "generate route with tool and preview artifact capability"
        },
        {
          code: "artifact_linked",
          label: "Artifact",
          detail: "source-sketch is linked to the current workspace session"
        },
        {
          code: "preview_queued",
          label: "Preview",
          detail: "browser preview target resolved from p5.js artifact metadata"
        }
      ]
    }
  };
}
