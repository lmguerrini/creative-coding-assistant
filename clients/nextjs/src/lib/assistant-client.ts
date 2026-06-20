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
