"use client";

import {
  startTransition,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type FormEvent,
  type KeyboardEvent,
  type MouseEvent
} from "react";
import {
  Activity,
  Boxes,
  Braces,
  Command,
  Database,
  Gauge,
  LayoutDashboard,
  LayoutGrid,
  Paintbrush,
  PanelRight,
  Play,
  Settings,
  Sparkles,
  TerminalSquare,
  X
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type {
  ArtifactAction,
  ArtifactSummary,
  AssistantWorkspaceSnapshot,
  ClarificationSummary,
  ImageAttachmentSummary,
  InspectorTabName,
  WorkflowStepState
} from "@/lib/assistant-client";
import { getInitialWorkspaceSnapshot } from "@/lib/assistant-client";
import {
  readClarificationSummary,
  readCreativeExecutionPlanSummary,
  readStreamEventError,
  readPreviewArtifactUpdate,
  readWorkflowMetadata,
  streamAssistantEvents as streamBackendAssistantEvents,
  workflowNodeFromAssistantStreamEvent,
  type AssistantArtifactRefinementRequest,
  type AssistantStreamEvent,
  type AssistantStreamRequest
} from "@/lib/assistant-stream";
import {
  resolveAssistantRequestMode,
  type AssistantRequestMode
} from "@/lib/assistant-intent";
import {
  createWorkspacePersistenceClient,
  createWorkspaceSessionRecord,
  deletePersistedWorkspaceSession,
  defaultWorkspacePreferences,
  defaultWorkspaceLayoutState,
  fingerprintWorkspaceSessionRecord,
  listLocalWorkspaceSessions,
  normalizeWorkspacePreferences,
  normalizeWorkspaceLayoutState,
  removeLocalWorkspaceSession,
  snapshotFromWorkspaceSessionRecord,
  withWorkspaceIdentity,
  workspaceLayoutBounds,
  type WorkspaceLayoutState,
  type WorkspacePreferences,
  type WorkspacePersistenceClient,
  type WorkspacePersistenceLoadResult,
  type WorkspaceSessionSummary
} from "@/lib/workspace-persistence";
import {
  buildArtifactDocument,
  copyArtifactDocument,
  downloadArtifactDocument,
  formatArtifactActionLabel,
  highlightArtifactDocument,
  type ArtifactDocument,
  type HighlightedLine
} from "@/lib/artifact-inspector";
import { renameWorkspaceArtifact } from "@/lib/artifact-naming";
import {
  removeWorkspaceArtifact,
  restoreWorkspaceArtifact,
  type RemovedArtifact
} from "@/lib/artifact-lifecycle";
import { buildProjectBundle } from "@/lib/project-bundle";
import {
  buildWorkflowRuntimeModel,
  deriveWorkflowRuntimeActivity,
  type WorkflowRuntimeModel,
  type WorkflowRuntimeActivity,
  type WorkflowRuntimeTraceEvent,
  type WorkflowRuntimeVisualState
} from "@/lib/workflow-runtime";
import {
  buildWorkflowExecutionModel,
  type WorkflowExecutionMode,
  type WorkflowExecutionModel
} from "@/lib/workflow-execution";
import {
  formatWorkflowGraphRoute,
  selectWorkflowGraphSteps
} from "@/lib/workflow-graph";
import {
  buildConversationContextModel,
  type ConversationContextModel
} from "@/lib/conversation-context";
import {
  buildWorkflowExplorerModel,
  type WorkflowExplorerModel
} from "@/lib/workflow-explorer";
import {
  buildRetrievalRuntimeModel,
  type RetrievalRuntimeModel
} from "@/lib/retrieval-runtime";
import {
  buildPreviewControllerModel,
  createPreviewSessionOverride,
  type PreviewControllerModel,
  type PreviewRuntimeSessionOverride
} from "@/lib/preview-controller";
import {
  buildPreviewRendererRoute,
  matchCreativePreviewRenderer,
  type PreviewRendererRoute
} from "@/lib/preview-renderers";
import {
  buildPreviewRuntimeSummary,
  isArtifactPreviewable
} from "@/lib/preview-runtime";
import {
  appendRefinementPassRecord,
  enrichArtifactRefinementRequest
} from "@/lib/refinement-passes";
import { resolveArtifactFollowUp } from "@/lib/artifact-follow-up";
import {
  hydrateWorkspaceFromArtifactExtractedEvent,
  hydrateWorkspaceFromFinalEvent,
  type LiveArtifactHydrationResult
} from "@/lib/live-artifact-hydration";
import {
  buildProviderTelemetryModel,
  type ProviderTelemetryLifecycleStep,
  type ProviderTelemetryModel
} from "@/lib/provider-telemetry";
import {
  buildProvenanceEngineModel,
  type ProvenanceEngineModel
} from "@/lib/provenance-engine";
import {
  buildCreativeCostRunRecord,
  type CreativeCostRunRecord
} from "@/lib/creative-cost-intelligence";
import {
  deleteSessionUsage,
  readSessionUsageSummaries,
  recordSessionUsageRun,
  renameSessionUsage,
  type SessionUsageSummary
} from "@/lib/session-usage-ledger";
import {
  buildTelemetryDashboardModel,
  type TelemetryDashboardModel
} from "@/lib/telemetry-dashboard";
import {
  buildRuntimeConsoleModel,
  type RuntimeConsoleLiveSnapshot,
  type RuntimeConsoleModel
} from "@/lib/runtime-console";
import {
  buildPreviewRuntimeSource,
  getExecutablePreviewRuntimeKind,
  type PreviewExecutableRuntimeKind,
  type PreviewRuntimeFrameSample,
  type PreviewRuntimeSource,
  type PreviewRuntimeStatus
} from "@/lib/preview-runtime-adapters";
import {
  buildMultimodalSummary,
  createImageAttachmentFromFile,
  normalizeImageAttachments,
  toAssistantRequestImageAttachments
} from "@/lib/multimodal-attachments";
import {
  buildConversationEntries,
  toPersistedConversation,
  type ConversationEntry,
  type ConversationEntryPhase
} from "@/lib/streaming-conversation";
import {
  buildHitlApprovalStreamEvent,
  createHitlApprovalRequest,
  getHitlApprovalStateLabel,
  isHitlApprovalBlockingState,
  isHitlApprovalTerminalState,
  summarizeHitlApprovalRequests,
  updateHitlApprovalRequest,
  type HitlActionId,
  type HitlActionState,
  type HitlApprovalRequest
} from "@/lib/hitl-runtime";
import {
  demoModeScenarios,
  getDefaultDemoModeScenario,
  type DemoModeScenario
} from "@/lib/demo-mode";
import { homepagePromptLibrary } from "@/lib/curated-prompt-library";
import {
  buildCreativeTimelineModel,
  type CreativeTimelineModel
} from "@/lib/creative-timeline";
import {
  buildV3InspectorPanelsModel,
  type V3InspectorPanelsModel
} from "@/lib/v3-inspector-panels";
import { buildWorkstationDashboardModel } from "@/lib/workstation-dashboard";
import {
  buildProductIntelligenceModel,
  getProductIntelligenceSection,
  productIntelligenceCategories,
  type ProductIntelligenceCategory,
  type ProductIntelligenceModel
} from "@/lib/product-intelligence";
import {
  fetchDomainExperienceCatalog,
  loadingDomainExperienceCatalog,
  type DomainExperienceCatalog
} from "@/lib/domain-experience";
import {
  buildSessionIntelligenceModel,
  readSessionIntelligenceMetadata,
  type SessionIntelligenceMetadataInput
} from "@/lib/session-intelligence";
import {
  createWorkstationError,
  type WorkstationError
} from "@/lib/workstation-errors";
import { buildWorkstationState } from "@/lib/workstation-state";
import { buildZipArchive, downloadZipArchive } from "@/lib/zip-archive";
import {
  buildGenerationControls,
  createFeedbackSignal,
  selectPersonalizationContext,
  type EvaluationHistoryRecord,
  type FeedbackSentiment
} from "@/lib/product-controls";
import {
  buildEvaluationBenchmarkRun,
  buildGoldenEvaluationDataset,
  CURRENT_PRODUCT_RETRIEVAL_CASE_IDS,
  currentProductRetrievalScoreFromEvidence,
  emptyRagasEvidence,
  normalizeEvaluationBenchmarkMode,
  type EvaluationCategory,
  type EvaluationExecutionProgress,
  type EvaluationProgressCallback,
  type EvaluationRunRequest,
  type RagasExecutionEvidence
} from "@/lib/evaluation-benchmark";
import { ArtifactRefinementPanel } from "./artifact-refinement-panel";
import {
  ApplicationConfirmDialog,
  ApplicationFloatingPanel,
  type ApplicationConfirmationRequest
} from "./application-floating-surfaces";
import { CreativeTimelineSurface } from "./creative-timeline-surface";
import { ConversationContextInspector } from "./conversation-context-inspector";
import { CreativeCostIntelligenceDashboard } from "./creative-cost-intelligence-dashboard";
import { DemoModePanel } from "./demo-mode-panel";
import { EvaluationSessionDashboard } from "./evaluation-session-dashboard";
import { LangSmithTraceDeepDive } from "./langsmith-trace-deep-dive";
import { ProviderObservabilityDeepDive } from "./provider-observability-deep-dive";
import { RetrievalInspector } from "./retrieval-inspector";
import { RuntimeConsoleInspector } from "./runtime-console-inspector";
import { RightInspector } from "./right-inspector";
import { SessionSidebar } from "./session-sidebar";
import { SubsystemErrorCallout } from "./subsystem-error-callout";
import { PreviewWorkspace } from "./preview-workspace";
import { V3InspectorPanelsSurface } from "./v3-inspector-panels-surface";
import {
  ProductIntelligenceDashboard,
  ProductIntelligenceInspector
} from "./product-intelligence-dashboard";
import { WorkflowExplorerSurface } from "./workflow-explorer-surface";
import { WorkflowTimelineExplorer } from "./workflow-timeline-explorer";
import {
  WorkspaceComposer,
  WorkspaceConversation
} from "./workspace-conversation";
import {
  WorkspaceAttachmentControl,
  WorkspaceGenerationControls,
  WorkspaceImageReferences
} from "./workspace-request-controls";
import { WorkflowExecutionInspector } from "./workflow-execution-inspector";
import {
  DashboardActionCard,
  DashboardCardGrid,
  DashboardDisclosure,
  DashboardInfoCard,
  DashboardPageHero,
  DashboardProcessRail,
  DashboardSection
} from "./dashboard-page-primitives";

type WorkstationShellProps = {
  snapshot: AssistantWorkspaceSnapshot;
  streamAssistantEvents?: AssistantStreamClient;
  persistenceClient?: WorkspacePersistenceClient;
};

type AssistantStreamClient = (
  request: AssistantStreamRequest,
  binding: AssistantStreamInvocationBinding
) => AsyncIterable<AssistantStreamEvent>;

type AssistantStreamInvocationBinding = Readonly<{
  epoch: number;
  requestId: string;
  sessionId: string;
  signal: AbortSignal;
}>;

type PreviewRuntimeTelemetryBase = {
  kind: PreviewExecutableRuntimeKind;
  route: PreviewRendererRoute;
  runtimeId: string;
  source: PreviewRuntimeSource;
};

type PreviewRuntimeStatusTelemetryEvent = PreviewRuntimeTelemetryBase & {
  status: PreviewRuntimeStatus;
};

type PreviewRuntimeFrameTelemetryEvent = PreviewRuntimeTelemetryBase & {
  sample: PreviewRuntimeFrameSample;
};

const inspectorTabIcons = {
  Overview: Sparkles,
  Architecture: Boxes,
  Workflow: Activity,
  Agents: Sparkles,
  Runtime: Command,
  Preview: Play,
  Code: Braces,
  Domains: Boxes,
  "Knowledge Base": Database,
  Memory: Database,
  Sessions: LayoutGrid,
  Providers: Command,
  Telemetry: Gauge,
  Metrics: Gauge,
  Validation: Activity,
  "Product Bugs": Activity,
  LangSmith: TerminalSquare,
  Settings,
  Artifacts: Boxes,
  Retrieval: TerminalSquare
} satisfies Record<ProductIntelligenceCategory, LucideIcon>;

type WorkflowState = WorkflowStepState["state"];
type WorkspaceWorkflow = AssistantWorkspaceSnapshot["workflow"];
type WorkspacePersistenceState =
  | "loading"
  | "ready"
  | "restored"
  | "saving"
  | "saved"
  | "local"
  | "unavailable";
type ArtifactTransferAction = Extract<ArtifactAction, "Download" | "Export">;
type ArtifactActionFeedback = {
  action: "Copy" | ArtifactTransferAction;
  artifactId: string;
  state: "success" | "error";
};
type ApprovalActionExecutor = () => Promise<void> | void;
type ResizeTarget = "inspector" | "preview";
type UtilityPanelName = "theme" | "settings";
type FocusRestoreState = {
  inspectorCollapsed: boolean;
  previewOpen: boolean;
  sidebarCollapsed: boolean;
};
type WorkspaceLayoutStyle = CSSProperties & {
  "--inspector-width": string;
  "--preview-height": string;
};
type ThemePresetOption = {
  value: WorkspacePreferences["theme"];
  label: string;
  description: string;
  accent: string;
  surface: string;
};
type PendingArtifactRefinement = AssistantArtifactRefinementRequest & {
  requestedAt: string;
};
type ActiveAssistantRequest = Readonly<{
  abortController: AbortController;
  assistantMessageId: string;
  epoch: number;
  forcePreviewOpen: boolean;
  pendingRefinement: PendingArtifactRefinement | null;
  projectId: string;
  prompt: string;
  requestId: string;
  requestMode: AssistantRequestMode;
  sessionId: string;
}>;
type AssistantStreamRuntimeState = {
  hasPreviewRuntimeEvent: boolean;
};

const localWorkflowIntervalMs = 850;
const artifactFeedbackDurationMs = 1400;
const evaluationRunEndpoint =
  process.env.NEXT_PUBLIC_EVALUATION_RUN_URL ??
  "http://localhost:8000/api/evaluation/run";
const evaluationPollIntervalMs = 400;
const evaluationReconnectBaseDelayMs = 800;
const evaluationReconnectMaxDelayMs = 10_000;
const evaluationMaxConsecutiveRefreshFailures = 3;
const defaultWorkspacePersistenceClient = createWorkspacePersistenceClient();
const userModeInspectorTabs = new Set<ProductIntelligenceCategory>([
  "Preview",
  "Code",
  "Artifacts",
  "Domains",
  "Settings"
]);
const userModeDefaultInspectorTab: InspectorTabName = "Preview";
const persistenceStateLabels = {
  loading: "Restoring session",
  ready: "Local session ready",
  restored: "Session restored",
  saving: "Saving session",
  saved: "Session saved",
  local: "Stored locally",
  unavailable: "Stored for this tab"
} satisfies Record<WorkspacePersistenceState, string>;
const emptyWorkspaceArtifact: ArtifactSummary = {
  id: "empty-workspace-artifact",
  title: "No artifact yet",
  type: "code",
  language: "Creative code",
  status: "Ready for first prompt",
  summary: "Generated source appears here after the first creative request.",
  content: "// Generated code appears here after your first creative request.",
  actions: []
};
const themePresetOptions = [
  {
    value: "aqua",
    label: "Aqua",
    description: "Teal-aqua studio accents with a cool, luminous contrast.",
    accent: "#4cd7c8",
    surface: "linear-gradient(135deg, rgba(76, 215, 200, 0.24), rgba(124, 167, 255, 0.16))"
  },
  {
    value: "codex",
    label: "Deep Blue",
    description: "Neutral graphite with cool blue, high-contrast accents.",
    accent: "#339cff",
    surface: "linear-gradient(135deg, rgba(51, 156, 255, 0.18), rgba(24, 24, 24, 0.94))"
  },
  {
    value: "codex_white",
    label: "Dark",
    description: "Neutral graphite with crisp white accents and panel outlines.",
    accent: "#f4f7fb",
    surface: "linear-gradient(135deg, rgba(244, 247, 251, 0.16), rgba(24, 24, 24, 0.94))"
  },
  {
    value: "light",
    label: "Light",
    description: "Calm daylight workspace with neutral surfaces.",
    accent: "#339cff",
    surface: "linear-gradient(135deg, rgba(255, 255, 255, 0.98), rgba(51, 156, 255, 0.1))"
  },
  {
    value: "matrix",
    label: "Matrix",
    description: "Obsidian console with restrained lime signal highlights.",
    accent: "#9fe870",
    surface:
      "linear-gradient(135deg, rgba(159, 232, 112, 0.16), rgba(38, 58, 30, 0.16), rgba(7, 11, 6, 0.26))"
  },
  {
    value: "terminal",
    label: "Terminal",
    description: "Ink-black console surfaces with warm phosphor-green signals.",
    accent: "#86efac",
    surface: "linear-gradient(135deg, rgba(134, 239, 172, 0.18), rgba(3, 12, 8, 0.98))"
  },
  {
    value: "horizon",
    label: "Horizon",
    description: "Midnight indigo with a restrained coral horizon accent.",
    accent: "#fb7185",
    surface: "linear-gradient(135deg, rgba(251, 113, 133, 0.2), rgba(30, 27, 75, 0.94))"
  },
  {
    value: "zen",
    label: "Zen",
    description: "Low-noise slate surfaces with a calm sage highlight.",
    accent: "#a7c4a0",
    surface: "linear-gradient(135deg, rgba(167, 196, 160, 0.2), rgba(25, 31, 29, 0.96))"
  },
  {
    value: "blueprint",
    label: "Blueprint",
    description: "Technical navy surfaces with crisp cyan drafting lines.",
    accent: "#67e8f9",
    surface: "linear-gradient(135deg, rgba(103, 232, 249, 0.2), rgba(8, 47, 73, 0.96))"
  }
] satisfies readonly ThemePresetOption[];

export function WorkstationShell({
  snapshot: initialSnapshot,
  streamAssistantEvents = streamBackendAssistantEvents,
  persistenceClient = defaultWorkspacePersistenceClient
}: WorkstationShellProps) {
  const [activePersistenceClient, setActivePersistenceClient] = useState(
    () => persistenceClient
  );
  const workspaceIdentity =
    activePersistenceClient.identity ?? initialSnapshot.session;
  const [snapshot, setSnapshot] = useState(() =>
    withWorkspaceIdentity(initialSnapshot, workspaceIdentity)
  );
  const snapshotRef = useRef(snapshot);
  const entryIdCounterRef = useRef(0);
  const approvalIdCounterRef = useRef(0);
  const assistantRequestCounterRef = useRef(0);
  const assistantStreamEpochRef = useRef(0);
  const activeAssistantRequestRef = useRef<ActiveAssistantRequest | null>(null);
  const workflowTraceSessionIdRef = useRef<string | null>(null);
  const localRuntimeSequenceRef = useRef(1000);
  const activeRequestModeRef = useRef<AssistantRequestMode>("generate");
  const streamingAssistantIdRef = useRef<string | null>(null);
  const previewRuntimeTelemetryKeysRef = useRef<Set<string>>(new Set());
  const previewRuntimeErrorScopesRef = useRef<Set<string>>(new Set());
  const [conversationEntries, setConversationEntries] = useState(() =>
    buildConversationEntries(
      initialSnapshot.messages,
      createConversationEntryId,
      initialSnapshot.workflow
    )
  );
  const [composerValue, setComposerValue] = useState("");
  const [isAttachmentMenuOpen, setIsAttachmentMenuOpen] = useState(false);
  const [isDemoModeOpen, setIsDemoModeOpen] = useState(false);
  const [activeDemoScenarioId, setActiveDemoScenarioId] = useState(
    () => getDefaultDemoModeScenario().id
  );
  const [imageAttachments, setImageAttachments] = useState(() =>
    normalizeImageAttachments(initialSnapshot.multimodal.imageAttachments)
  );
  const [imageUploadError, setImageUploadError] = useState<WorkstationError | null>(
    initialSnapshot.multimodal.error ?? null
  );
  const [pendingImageUploadCount, setPendingImageUploadCount] = useState(0);
  const [domainExperience, setDomainExperience] = useState<DomainExperienceCatalog>(
    loadingDomainExperienceCatalog
  );
  const [activeTab, setActiveTab] = useState<ProductIntelligenceCategory>(
    getInitialActiveTab(initialSnapshot)
  );
  const [inspectorTabs, setInspectorTabs] = useState<
    ProductIntelligenceCategory[]
  >(() =>
    Array.from(
      new Set<ProductIntelligenceCategory>([
        "Overview",
        ...initialSnapshot.inspectorTabs.map((tab) => tab.label)
      ])
    )
  );
  const [isInspectorAddMenuOpen, setIsInspectorAddMenuOpen] = useState(false);
  const [isDashboardOpen, setIsDashboardOpen] = useState(false);
  const [dashboardCategory, setDashboardCategory] =
    useState<ProductIntelligenceCategory>("Overview");
  const [activeArtifactId, setActiveArtifactId] = useState(
    initialSnapshot.artifacts[0]?.id ?? ""
  );
  const [previewArtifactId, setPreviewArtifactId] = useState(
    getInitialPreviewArtifactId(initialSnapshot)
  );
  const [isPreviewOpen, setIsPreviewOpen] = useState(
    initialSnapshot.preview.active
  );
  const [isPreviewFullscreen, setIsPreviewFullscreen] = useState(false);
  const [previewSessionOverride, setPreviewSessionOverride] =
    useState<PreviewRuntimeSessionOverride | null>(null);
  const [workflowProgressIndex, setWorkflowProgressIndex] = useState(
    getInitialWorkflowIndex(initialSnapshot.workflow.steps)
  );
  const [workflowRunId, setWorkflowRunId] = useState(0);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamError, setStreamError] = useState<WorkstationError | null>(null);
  const [streamEvents, setStreamEvents] = useState(initialSnapshot.debug.events);
  const [clarification, setClarification] = useState<ClarificationSummary | null>(
    initialSnapshot.clarification ?? null
  );
  const [sessionIntelligenceMetadata, setSessionIntelligenceMetadata] =
    useState<SessionIntelligenceMetadataInput | null>(null);
  const [workflowTraceEvents, setWorkflowTraceEvents] = useState<
    WorkflowRuntimeTraceEvent[]
  >([]);
  const [creativeCostRunHistory, setCreativeCostRunHistory] = useState<
    CreativeCostRunRecord[]
  >([]);
  const [sessionUsageSummaries, setSessionUsageSummaries] = useState<
    SessionUsageSummary[]
  >([]);
  const [previewRuntimeLive, setPreviewRuntimeLive] =
    useState<RuntimeConsoleLiveSnapshot | null>(null);
  const [persistenceState, setPersistenceState] =
    useState<WorkspacePersistenceState>("loading");
  const [, setPersistenceError] = useState<WorkstationError | null>(null);
  const [layoutState, setLayoutState] = useState<WorkspaceLayoutState>(
    defaultWorkspaceLayoutState
  );
  const [workspacePreferences, setWorkspacePreferences] =
    useState<WorkspacePreferences>(defaultWorkspacePreferences);
  const workflowMode = workspacePreferences.workflowMode;
  const [workspaceSessions, setWorkspaceSessions] = useState<
    WorkspaceSessionSummary[]
  >([]);
  const feedbackIdCounterRef = useRef(0);
  const [isFocusMode, setIsFocusMode] = useState(false);
  const [activeResizeTarget, setActiveResizeTarget] =
    useState<ResizeTarget | null>(null);
  const [openUtilityPanel, setOpenUtilityPanel] = useState<UtilityPanelName | null>(
    null
  );
  const [applicationConfirmation, setApplicationConfirmation] =
    useState<ApplicationConfirmationRequest | null>(null);
  const [copyFeedback, setCopyFeedback] = useState<ArtifactActionFeedback | null>(
    null
  );
  const [lastRemovedArtifact, setLastRemovedArtifact] =
    useState<RemovedArtifact | null>(null);
  const [lastRestoredArtifact, setLastRestoredArtifact] =
    useState<RemovedArtifact | null>(null);
  const [transferFeedback, setTransferFeedback] =
    useState<ArtifactActionFeedback | null>(null);
  const [artifactTransferError, setArtifactTransferError] =
    useState<WorkstationError | null>(null);
  const [approvalRequests, setApprovalRequests] = useState<HitlApprovalRequest[]>(
    []
  );
  const [dismissedApprovalRequestId, setDismissedApprovalRequestId] = useState<
    string | null
  >(null);
  const previousPreviewRuntimeSessionKeyRef = useRef<string | null>(null);
  const composerTextareaRef = useRef<HTMLTextAreaElement | null>(null);
  const approvalCardRef = useRef<HTMLElement | null>(null);
  const approvalFocusOriginRef = useRef<HTMLElement | null>(null);
  const isShellMountedRef = useRef(true);
  const evaluationAbortControllerRef = useRef<AbortController | null>(null);
  const hasLoadedPersistenceRef = useRef(false);
  const persistenceSessionEpochRef = useRef(0);
  const lastPersistedFingerprintRef = useRef<string | null>(null);
  const skipNextPersistenceSaveRef = useRef(false);
  const focusRestoreRef = useRef<FocusRestoreState | null>(null);
  const copyFeedbackTimerRef = useRef<number | null>(null);
  const transferFeedbackTimerRef = useRef<number | null>(null);
  const dragCleanupRef = useRef<(() => void) | null>(null);
  const utilityTrayRef = useRef<HTMLDivElement>(null);
  const themeTriggerRef = useRef<HTMLButtonElement>(null);
  const settingsTriggerRef = useRef<HTMLButtonElement>(null);
  const approvalExecutorsRef = useRef<Record<string, ApprovalActionExecutor>>({});
  const imageAttachmentCounterRef = useRef(imageAttachments.length);
  const imageAttachmentsRef = useRef(imageAttachments);
  const imageUploadEpochRef = useRef(0);
  const imageUploadQueueRef = useRef<Promise<void>>(Promise.resolve());

  function clearFeedbackTimers() {
    clearTimer(copyFeedbackTimerRef.current);
    clearTimer(transferFeedbackTimerRef.current);
  }

  function createConversationEntryId() {
    entryIdCounterRef.current += 1;
    return `conversation-entry-${entryIdCounterRef.current}`;
  }

  function createApprovalRequestId() {
    approvalIdCounterRef.current += 1;
    return `approval-request-${approvalIdCounterRef.current}`;
  }

  useEffect(() => {
    isShellMountedRef.current = true;

    return () => {
      isShellMountedRef.current = false;
      invalidateActiveAssistantRequest({ updateUi: false });
      evaluationAbortControllerRef.current?.abort();
      evaluationAbortControllerRef.current = null;
      clearFeedbackTimers();
      clearDragState();
    };
  }, []);

  useEffect(() => {
    imageUploadEpochRef.current += 1;
    imageUploadQueueRef.current = Promise.resolve();
    setPendingImageUploadCount(0);
  }, [workspaceIdentity.sessionId]);

  useEffect(() => {
    document.documentElement.dataset.ccaTheme = workspacePreferences.theme;
    document.documentElement.dataset.ccaHeadingFontSize = workspacePreferences.headingFontSize;
    document.documentElement.dataset.ccaUiFontSize = workspacePreferences.uiFontSize;
    document.documentElement.dataset.ccaLabelFontSize = workspacePreferences.labelFontSize;
    document.documentElement.dataset.ccaCodeFontSize = workspacePreferences.codeFontSize;

    return () => {
      delete document.documentElement.dataset.ccaTheme;
      delete document.documentElement.dataset.ccaHeadingFontSize;
      delete document.documentElement.dataset.ccaUiFontSize;
      delete document.documentElement.dataset.ccaLabelFontSize;
      delete document.documentElement.dataset.ccaCodeFontSize;
    };
  }, [
    workspacePreferences.codeFontSize,
    workspacePreferences.headingFontSize,
    workspacePreferences.labelFontSize,
    workspacePreferences.theme,
    workspacePreferences.uiFontSize
  ]);

  useEffect(() => {
    setWorkspaceSessions(listLocalWorkspaceSessions(workspaceIdentity.userId));
    setSessionUsageSummaries(readSessionUsageSummaries(workspaceIdentity.userId));
  }, [workspaceIdentity.userId]);

  useEffect(() => {
    if (
      !workspacePreferences.showDebugPanels &&
      !userModeInspectorTabs.has(activeTab)
    ) {
      setActiveTab(userModeDefaultInspectorTab);
    }
  }, [activeTab, workspacePreferences.showDebugPanels]);

  useEffect(() => {
    setInspectorTabs((currentTabs) =>
      currentTabs.includes(activeTab) ? currentTabs : [...currentTabs, activeTab]
    );
  }, [activeTab]);

  useEffect(() => {
    if (workspacePreferences.showDebugPanels) {
      updateLayout({ inspectorCollapsed: false });
      return;
    }

    setActiveTab((currentTab) =>
      userModeInspectorTabs.has(currentTab) ? currentTab : userModeDefaultInspectorTab
    );
    updateLayout({ inspectorCollapsed: true });
  }, [workspacePreferences.showDebugPanels]);

  useEffect(() => {
    const activeApproval = summarizeHitlApprovalRequests(approvalRequests).activeRequest;
    if (!activeApproval || activeApproval.state !== "pending_approval") {
      return;
    }

    const frameId = window.requestAnimationFrame(() => {
      approvalCardRef.current?.focus();
    });

    return () => window.cancelAnimationFrame(frameId);
  }, [approvalRequests]);

  useEffect(() => {
    if (workflowRunId === 0) {
      return undefined;
    }

    const finalizationIndex = getWorkflowNodeIndex(
      snapshot.workflow.steps,
      "finalization"
    );
    const timer = window.setInterval(() => {
      setWorkflowProgressIndex((currentIndex) => {
        if (currentIndex >= finalizationIndex) {
          window.clearInterval(timer);
          return currentIndex;
        }

        return currentIndex + 1;
      });
    }, localWorkflowIntervalMs);

    return () => window.clearInterval(timer);
  }, [snapshot.workflow.steps, workflowRunId]);

  useEffect(() => {
    if (!openUtilityPanel) {
      return undefined;
    }

    const handlePointerDown = (event: PointerEvent) => {
      if (
        utilityTrayRef.current &&
        event.target instanceof Node &&
        !utilityTrayRef.current.contains(event.target)
      ) {
        setOpenUtilityPanel(null);
      }
    };

    document.addEventListener("pointerdown", handlePointerDown);

    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
    };
  }, [openUtilityPanel]);

  useEffect(() => {
    let isMounted = true;

    async function restoreWorkspaceSession() {
      try {
        const restoredSession = await withPersistenceTimeout<WorkspacePersistenceLoadResult>(
          activePersistenceClient.load(),
          {
            error: buildPersistenceTimeoutError("load"),
            record: null,
            source: "none"
          },
          1500
        );
        if (!isMounted) {
          return;
        }

        setPersistenceError(restoredSession.error);

        if (restoredSession.record) {
          const restoredSnapshot = snapshotFromWorkspaceSessionRecord(
            initialSnapshot,
            restoredSession.record
          );

          if (
            shouldIgnoreRestoredWorkspaceSession(initialSnapshot, restoredSnapshot)
          ) {
            setPersistenceError(null);
            setPersistenceState("ready");
            return;
          }

          const restoredImageAttachments = normalizeImageAttachments(
            restoredSnapshot.multimodal.imageAttachments
          );
          invalidateActiveAssistantRequest();
          replaceSnapshot(restoredSnapshot);
          resetAssistantStreamProjection(restoredSnapshot.debug.events);
          setConversationEntries(
            buildConversationEntries(
              restoredSnapshot.messages,
              createConversationEntryId,
              restoredSnapshot.workflow
            )
          );
          imageUploadEpochRef.current += 1;
          imageUploadQueueRef.current = Promise.resolve();
          setPendingImageUploadCount(0);
          imageAttachmentsRef.current = restoredImageAttachments;
          setImageAttachments(restoredImageAttachments);
          setImageUploadError(restoredSnapshot.multimodal.error ?? null);
          setClarification(restoredSnapshot.clarification ?? null);
          setSessionIntelligenceMetadata(null);
          setLastRemovedArtifact(null);
          setLastRestoredArtifact(null);
          imageAttachmentCounterRef.current = restoredImageAttachments.length;
          const restoredActiveArtifactId = resolveRestoredArtifactId(
            restoredSession.record.activeArtifactId,
            restoredSnapshot.artifacts,
            restoredSnapshot.artifacts[0]?.id ?? ""
          );
          const restoredPreviewArtifactId = resolveRestoredArtifactId(
            restoredSession.record.previewArtifactId,
            restoredSnapshot.artifacts,
            getInitialPreviewArtifactId(restoredSnapshot)
          );
          const restoredArtifactSelectionWasNormalized =
            restoredActiveArtifactId !== restoredSession.record.activeArtifactId ||
            restoredPreviewArtifactId !== restoredSession.record.previewArtifactId;
          setActiveTab(restoredSession.record.activeInspectorTab);
          setActiveArtifactId(restoredActiveArtifactId);
          setPreviewArtifactId(restoredPreviewArtifactId);
          setIsPreviewOpen(restoredSession.record.previewOpen);
          setLayoutState(normalizeWorkspaceLayoutState(restoredSession.record.layout));
          setWorkspacePreferences(
            normalizeWorkspacePreferences(restoredSession.record.preferences)
          );
          setIsFocusMode(false);
          setIsPreviewFullscreen(false);
          setPreviewSessionOverride(null);
          setOpenUtilityPanel(null);
          focusRestoreRef.current = null;
          setWorkflowProgressIndex(
            getWorkflowNodeIndex(
              restoredSnapshot.workflow.steps,
              restoredSnapshot.workflow.currentNode
            )
          );
          lastPersistedFingerprintRef.current =
            fingerprintWorkspaceSessionRecord(restoredSession.record);
          skipNextPersistenceSaveRef.current = !restoredArtifactSelectionWasNormalized;
          setPersistenceState(
            restoredSession.source === "local" ? "local" : "restored"
          );
          setWorkspaceSessions(listLocalWorkspaceSessions(workspaceIdentity.userId));
          return;
        }

        invalidateActiveAssistantRequest();
        replaceSnapshot(withWorkspaceIdentity(initialSnapshot, workspaceIdentity));
        setPersistenceState(restoredSession.error ? "unavailable" : "ready");
      } catch {
        if (isMounted) {
          setPersistenceError(buildPersistenceTimeoutError("load"));
          setPersistenceState("unavailable");
        }
      } finally {
        if (isMounted) {
          hasLoadedPersistenceRef.current = true;
        }
      }
    }

    restoreWorkspaceSession();

    return () => {
      isMounted = false;
    };
  }, [activePersistenceClient, initialSnapshot, workspaceIdentity]);

  const activeArtifact =
    snapshot.artifacts.find((artifact) => artifact.id === activeArtifactId) ??
    snapshot.artifacts[0] ??
    emptyWorkspaceArtifact;
  const activeDemoScenario = useMemo(
    () =>
      demoModeScenarios.find((scenario) => scenario.id === activeDemoScenarioId) ??
      getDefaultDemoModeScenario(),
    [activeDemoScenarioId]
  );
  const persistedMessages = useMemo(
    () => toPersistedConversation(conversationEntries),
    [conversationEntries]
  );
  const liveAssistantEntry =
    streamingAssistantIdRef.current == null
      ? null
      : conversationEntries.find(
          (entry) => entry.id === streamingAssistantIdRef.current
        ) ?? null;
  const hasActiveWorkflowRun =
    isStreaming || workflowRunId > 0 || workflowTraceEvents.length > 0;
  const workflow = useMemo(
    () =>
      buildInteractiveWorkflow(snapshot.workflow, workflowProgressIndex, {
        hasActiveRun: hasActiveWorkflowRun
      }),
    [hasActiveWorkflowRun, snapshot.workflow, workflowProgressIndex]
  );
  const persistedInspectorTab = toSnapshotInspectorTab(activeTab);
  const interactiveSnapshot: AssistantWorkspaceSnapshot = useMemo(
    () => ({
      ...snapshot,
      clarification,
      code: buildCodeSummaryForArtifact(snapshot.code, activeArtifact),
      inspectorTabs: snapshot.inspectorTabs.map((tab) => ({
        ...tab,
        active: tab.label === persistedInspectorTab,
        badge:
          tab.label === "Artifacts" ? String(snapshot.artifacts.length) : tab.badge
      })),
      messages: persistedMessages,
      multimodal: buildMultimodalSummary({
        baseMultimodal: snapshot.multimodal,
        imageAttachments,
        uploadError: imageUploadError
      }),
      preview: buildPreviewRuntimeSummary({
        artifacts: snapshot.artifacts,
        basePreview: snapshot.preview,
        isOpen: isPreviewOpen,
        previewArtifactId: previewArtifactId || activeArtifact.id,
        sessionOverride: previewSessionOverride,
        streamError,
        traceEvents: workflowTraceEvents,
        workflow
      }),
      workflow,
      debug: {
        ...snapshot.debug,
        status: isStreaming
          ? "Streaming live response"
          : streamError
            ? "Local draft available"
            : snapshot.debug.status,
        events: streamEvents
      }
    }),
    [
      activeArtifact,
      clarification,
      imageAttachments,
      imageUploadError,
      isPreviewOpen,
      previewSessionOverride,
      isStreaming,
      persistedMessages,
      previewArtifactId,
      persistedInspectorTab,
      snapshot,
      streamError,
      streamEvents,
      workflowTraceEvents,
      workflow
    ]
  );
  const previewRendererRoute = useMemo(
    () =>
      buildPreviewRendererRoute({
        artifacts: interactiveSnapshot.artifacts,
        preview: interactiveSnapshot.preview,
        previewArtifactId
      }),
    [interactiveSnapshot.artifacts, interactiveSnapshot.preview, previewArtifactId]
  );
  const previewRuntimeCode = useMemo(() => {
    const previewRuntimeArtifact =
      interactiveSnapshot.artifacts.find(
        (artifact) => artifact.id === previewRendererRoute.sourceArtifactId
      ) ??
      interactiveSnapshot.artifacts.find(
        (artifact) => artifact.id === previewRendererRoute.selectedArtifactId
      ) ??
      activeArtifact;

    return buildCodeSummaryForArtifact(snapshot.code, previewRuntimeArtifact);
  }, [
    activeArtifact,
    interactiveSnapshot.artifacts,
    previewRendererRoute.selectedArtifactId,
    previewRendererRoute.sourceArtifactId,
    snapshot.code
  ]);
  const previewRuntimeSource = useMemo(
    () =>
      buildPreviewRuntimeSource({
        code: previewRuntimeCode,
        route: previewRendererRoute
      }),
    [previewRuntimeCode, previewRendererRoute]
  );
  const previewRuntimeSessionKey =
    previewSessionOverride?.requestedAt ??
    `${previewArtifactId}:${interactiveSnapshot.preview.version}`;
  const previewController = useMemo(
    () =>
      buildPreviewControllerModel({
        isFullscreen: isPreviewFullscreen,
        preview: interactiveSnapshot.preview,
        route: previewRendererRoute,
        sessionOverride: previewSessionOverride
      }),
    [
      interactiveSnapshot.preview,
      isPreviewFullscreen,
      previewRendererRoute,
      previewSessionOverride
    ]
  );
  useEffect(() => {
    if (previousPreviewRuntimeSessionKeyRef.current === null) {
      previousPreviewRuntimeSessionKeyRef.current = previewRuntimeSessionKey;
      return;
    }

    if (previousPreviewRuntimeSessionKeyRef.current === previewRuntimeSessionKey) {
      return;
    }

    previousPreviewRuntimeSessionKeyRef.current = previewRuntimeSessionKey;
    setPreviewRuntimeLive(null);
  }, [previewRuntimeSessionKey]);
  const persistenceRecord = useMemo(
    () =>
      createWorkspaceSessionRecord({
        activeArtifactId,
        activeInspectorTab: persistedInspectorTab,
        layout: layoutState,
        preferences: workspacePreferences,
        previewArtifactId,
        previewOpen: isPreviewOpen,
        snapshot: interactiveSnapshot
      }),
    [
      activeArtifactId,
      interactiveSnapshot,
      layoutState,
      workspacePreferences,
      isPreviewOpen,
      previewArtifactId,
      persistedInspectorTab
    ]
  );
  const activeArtifactDocument = useMemo(
    () => buildArtifactDocument(interactiveSnapshot, activeArtifact),
    [activeArtifact, interactiveSnapshot]
  );
  const activeArtifactHighlights = useMemo(
    () => highlightArtifactDocument(activeArtifactDocument),
    [activeArtifactDocument]
  );
  const workflowRuntime = useMemo(
    () => buildWorkflowRuntimeModel(interactiveSnapshot.workflow, workflowTraceEvents),
    [interactiveSnapshot.workflow, workflowTraceEvents]
  );
  const workflowExecution = useMemo(
    () => buildWorkflowExecutionModel(workflowTraceEvents),
    [workflowTraceEvents]
  );
  const conversationContext = useMemo(
    () =>
      buildConversationContextModel({
        traceEvents: workflowTraceEvents,
        visibleEntryCount: conversationEntries.length
      }),
    [conversationEntries.length, workflowTraceEvents]
  );
  const providerTelemetry = useMemo(
    () => buildProviderTelemetryModel(workflowTraceEvents),
    [workflowTraceEvents]
  );
  useEffect(() => {
    let current = true;

    void fetchDomainExperienceCatalog().then((catalog) => {
      if (current) {
        setDomainExperience(catalog);
      }
    });

    return () => {
      current = false;
    };
  }, []);
  useEffect(() => {
    const completedRun = buildCreativeCostRunRecord({
      providerTelemetry,
      traceEvents: workflowTraceEvents
    });
    const traceSessionId = workflowTraceSessionIdRef.current;
    if (
      !completedRun ||
      !traceSessionId ||
      traceSessionId !== workspaceIdentity.sessionId
    ) {
      return;
    }

    setCreativeCostRunHistory((currentHistory) =>
      currentHistory.some((run) => run.id === completedRun.id)
        ? currentHistory
        : [...currentHistory, completedRun]
    );
    setSessionUsageSummaries(
      recordSessionUsageRun({
        run: completedRun,
        sessionId: traceSessionId,
        title: snapshot.session.title || snapshot.workspace.name,
        userId: workspaceIdentity.userId
      })
    );
  }, [providerTelemetry, snapshot.session.title, snapshot.workspace.name, workflowTraceEvents, workspaceIdentity.sessionId, workspaceIdentity.userId]);
  const retrievalRuntime = useMemo(
    () => buildRetrievalRuntimeModel(interactiveSnapshot.retrieval, workflowTraceEvents),
    [interactiveSnapshot.retrieval, workflowTraceEvents]
  );
  const telemetryDashboard = useMemo(
    () =>
      buildTelemetryDashboardModel({
        activeArtifact,
        creativeCostHistory: creativeCostRunHistory,
        draftPrompt: composerValue,
        providerTelemetry,
        retrievalRuntime,
        snapshot: interactiveSnapshot,
        traceEvents: workflowTraceEvents,
        workflowRuntime
      }),
    [
      activeArtifact,
      composerValue,
      creativeCostRunHistory,
      interactiveSnapshot,
      providerTelemetry,
      retrievalRuntime,
      workflowRuntime,
      workflowTraceEvents
    ]
  );
  const workstationState = useMemo(
    () =>
      buildWorkstationState({
        activeArtifactId,
        activeInspectorTab: persistedInspectorTab,
        activeWorkflowNodeId: workflowRuntime.summary.currentNode,
        inspectorCollapsed: layoutState.inspectorCollapsed,
        isStreaming,
        previewArtifactId,
        previewFullscreen: isPreviewFullscreen,
        previewOpen: isPreviewOpen,
        selectedEvaluation: telemetryDashboard.evaluation,
        snapshot: interactiveSnapshot,
        streamError,
        traceEvents: workflowTraceEvents
      }),
    [
      activeArtifactId,
      interactiveSnapshot,
      isPreviewFullscreen,
      isPreviewOpen,
      isStreaming,
      layoutState.inspectorCollapsed,
      previewArtifactId,
      streamError,
      telemetryDashboard.evaluation,
      workflowRuntime.summary.currentNode,
      workflowTraceEvents,
      persistedInspectorTab
    ]
  );
  const sessionIntelligence = useMemo(
    () =>
      buildSessionIntelligenceModel({
        snapshot: interactiveSnapshot,
        streamedMetadata: sessionIntelligenceMetadata,
        workstationState
      }),
    [interactiveSnapshot, sessionIntelligenceMetadata, workstationState]
  );
  const workflowExplorer = useMemo(
    () =>
      buildWorkflowExplorerModel({
        runtime: workflowRuntime,
        snapshot: interactiveSnapshot,
        traceEvents: workflowTraceEvents,
        workstationState
      }),
    [interactiveSnapshot, workflowRuntime, workflowTraceEvents, workstationState]
  );
  const provenance = useMemo(
    () =>
      buildProvenanceEngineModel({
        snapshot: interactiveSnapshot,
        traceEvents: workflowTraceEvents,
        workstationState
      }),
    [interactiveSnapshot, workflowTraceEvents, workstationState]
  );
  const creativeTimeline = useMemo(
    () =>
      buildCreativeTimelineModel({
        explorer: workflowExplorer,
        provenance,
        runtime: workflowRuntime,
        workstationState
      }),
    [provenance, workflowExplorer, workflowRuntime, workstationState]
  );
  const v3InspectorPanels = useMemo(
    () =>
      buildV3InspectorPanelsModel({
        provenance,
        traceEvents: workflowTraceEvents,
        workstationState
      }),
    [provenance, workflowTraceEvents, workstationState]
  );
  const workstationDashboard = useMemo(
    () =>
      buildWorkstationDashboardModel({
        runtime: workflowRuntime,
        snapshot: interactiveSnapshot,
        v3InspectorPanels,
        workstationState
      }),
    [interactiveSnapshot, v3InspectorPanels, workflowRuntime, workstationState]
  );
  const runtimeConsole = useMemo(
    () =>
      buildRuntimeConsoleModel({
        liveRuntime: previewRuntimeLive,
        preview: interactiveSnapshot.preview,
        route: previewRendererRoute,
        runtimeSource: previewRuntimeSource,
        traceEvents: workflowTraceEvents
      }),
    [
      interactiveSnapshot.preview,
      previewRendererRoute,
      previewRuntimeLive,
      previewRuntimeSource,
      workflowTraceEvents
    ]
  );
  const productIntelligence = useMemo(
    () =>
      buildProductIntelligenceModel({
        activeArtifactId,
        conversationContext,
        domainExperience,
        providerTelemetry,
        retrievalRuntime,
        runtimeConsole,
        sessionIntelligence,
        snapshot: interactiveSnapshot,
        telemetryDashboard,
        v3InspectorPanels,
        workflowExecution,
        workflowRuntime,
        workstationDashboard
      }),
    [
      activeArtifactId,
      conversationContext,
      domainExperience,
      interactiveSnapshot,
      providerTelemetry,
      retrievalRuntime,
      runtimeConsole,
      sessionIntelligence,
      telemetryDashboard,
      v3InspectorPanels,
      workflowExecution,
      workflowRuntime,
      workstationDashboard
    ]
  );
  const availableInspectorCategories = useMemo(
    () =>
      productIntelligenceCategories.filter(
        (category) =>
          workspacePreferences.showDebugPanels || userModeInspectorTabs.has(category)
      ),
    [workspacePreferences.showDebugPanels]
  );
  const visibleInspectorTabs = useMemo(
    () =>
      inspectorTabs.filter((tab) => availableInspectorCategories.includes(tab)),
    [availableInspectorCategories, inspectorTabs]
  );
  useEffect(() => {
    if (!visibleInspectorTabs.includes(activeTab)) {
      return;
    }

    document
      .getElementById(`${getInspectorPanelId(activeTab)}-tab`)
      ?.scrollIntoView?.({ block: "nearest", inline: "nearest" });
  }, [activeTab, layoutState.inspectorWidth, visibleInspectorTabs]);
  const activeTabSummary = getProductIntelligenceSection(
    productIntelligence,
    activeTab
  ).detail;
  const isImageUploadPending = pendingImageUploadCount > 0;
  const isComposerReady =
    hasLoadedPersistenceRef.current &&
    Boolean(composerValue.trim()) &&
    !isStreaming &&
    !isImageUploadPending;
  const approvalSummary = useMemo(
    () => summarizeHitlApprovalRequests(approvalRequests),
    [approvalRequests]
  );
  const blockingApprovalRequest = approvalSummary.activeRequest;
  const latestApprovalRequest = approvalSummary.latestRequest;
  const visibleApprovalRequest =
    blockingApprovalRequest ??
    (latestApprovalRequest &&
    latestApprovalRequest.id !== dismissedApprovalRequestId
      ? latestApprovalRequest
      : null);
  const workflowIssues = useMemo(
    () =>
      [
        workflowRuntime.error,
        ...approvalRequests.map((request) => buildHitlApprovalError(request))
      ].filter((error): error is WorkstationError => error !== null),
    [approvalRequests, workflowRuntime.error]
  );
  const streamState = blockingApprovalRequest
    ? blockingApprovalRequest.state === "pending_approval"
      ? "approval"
      : "executing"
    : isStreaming
      ? "streaming"
      : streamError
        ? "fallback"
        : "idle";
  const activeWorkflowActivity =
    streamState === "streaming" || streamState === "executing"
      ? workflowRuntime.summary.activity
      : null;
  const presentedWorkflowActivity = activeWorkflowActivity
    ? {
        ...activeWorkflowActivity,
        label:
          activeRequestModeRef.current === "explain"
            ? "Answering"
            : activeWorkflowActivity.label,
        detail: conversationActivityForRequestMode(
          activeWorkflowActivity,
          activeRequestModeRef.current
        )
      }
    : null;
  const persistenceStatusLabel =
    persistenceStateLabels[persistenceState] ?? "Local session ready";
  const hasWorkspaceArtifacts = snapshot.artifacts.length > 0;
  const visibleWorkspaceSessions = useMemo(() => {
    const current = {
      artifactCount: snapshot.artifacts.length,
      projectId: workspaceIdentity.projectId,
      sessionId: workspaceIdentity.sessionId,
      title: snapshot.session.title || "Creative session",
      updatedAt: snapshot.session.updatedAt ?? null
    };
    return workspaceSessions.some(
      (summary) => summary.sessionId === current.sessionId
    )
      ? workspaceSessions
      : [current, ...workspaceSessions];
  }, [snapshot, workspaceIdentity, workspaceSessions]);
  const currentSessionUsage = sessionUsageSummaries.find(
    (usage) => usage.sessionId === workspaceIdentity.sessionId
  ) ?? null;
  const hasPreviewOutcomeToExplain =
    workflowRuntime.summary.productOutcome.product_outcome === "PARTIAL" ||
    workflowRuntime.summary.productOutcome.product_outcome === "FAILURE" ||
    streamError !== null;
  const shouldRenderPreviewShelf =
    !isFocusMode &&
    !isDemoModeOpen &&
    (interactiveSnapshot.preview.available ||
      (interactiveSnapshot.preview.state === "unavailable" &&
        (hasPreviewOutcomeToExplain ||
          (!workspacePreferences.showDebugPanels &&
            (hasWorkspaceArtifacts || isDemoModeOpen)))));
  const activeArtifactDisplayLabel = workspacePreferences.showDebugPanels
    ? activeArtifact.title
    : formatUserArtifactLabel(activeArtifact);
  const isInspectorCollapsed = layoutState.inspectorCollapsed;
  const semanticSessionStatus = formatSemanticProductOutcomeStatus(
    workflowRuntime.summary.productOutcome
  );
  const sessionStatusLabel = blockingApprovalRequest
    ? getHitlApprovalStateLabel(blockingApprovalRequest.state)
    : presentedWorkflowActivity?.label ??
      semanticSessionStatus?.label ??
      workstationState.status.label;
  const sessionStatusDetail = blockingApprovalRequest
    ? blockingApprovalRequest.title
    : presentedWorkflowActivity?.detail ??
      semanticSessionStatus?.detail ??
      workstationState.status.detail;
  const userSessionStatus = formatUserModeSessionStatus({
    activity: presentedWorkflowActivity,
    hasFailedPreviewRuntime: runtimeConsole.health.signal === "failed",
    hasWorkspaceArtifacts,
    isDemoModeOpen,
    productOutcome: workflowRuntime.summary.productOutcome,
    streamError,
    streamState
  });
  const visibleSessionStatusLabel = workspacePreferences.showDebugPanels
    ? sessionStatusLabel
    : userSessionStatus.label;
  const visibleSessionStatusDetail = workspacePreferences.showDebugPanels
    ? sessionStatusDetail
    : userSessionStatus.detail;
  const isPristineSession =
    persistenceState !== "loading" &&
    streamState === "idle" &&
    !streamError &&
    !isDemoModeOpen &&
    !hasWorkspaceArtifacts &&
    conversationEntries.length === 0;
  const displayedSessionStatusLabel = isPristineSession
    ? "Ready"
    : visibleSessionStatusLabel;
  const displayedSessionStatusDetail = isPristineSession
    ? "Start a prompt"
    : visibleSessionStatusDetail;
  const workspaceLayoutStyle = useMemo(
    () =>
      ({
        "--inspector-width": `${layoutState.inspectorWidth}px`,
        "--preview-height": `${layoutState.previewHeight}px`
      }) as WorkspaceLayoutStyle,
    [layoutState.inspectorWidth, layoutState.previewHeight]
  );

  useEffect(() => {
    if (
      isPreviewFullscreen &&
      (!isPreviewOpen || !interactiveSnapshot.preview.available || isFocusMode)
    ) {
      setIsPreviewFullscreen(false);
    }
  }, [
    interactiveSnapshot.preview.available,
    isFocusMode,
    isPreviewFullscreen,
    isPreviewOpen
  ]);

  useEffect(() => {
    if (!hasLoadedPersistenceRef.current || persistenceState === "saving") {
      return undefined;
    }

    const fingerprint = fingerprintWorkspaceSessionRecord(persistenceRecord);
    if (skipNextPersistenceSaveRef.current) {
      skipNextPersistenceSaveRef.current = false;
      lastPersistedFingerprintRef.current = fingerprint;
      return undefined;
    }

    if (lastPersistedFingerprintRef.current === fingerprint) {
      return undefined;
    }

    lastPersistedFingerprintRef.current = fingerprint;
    setPersistenceState("saving");
    const persistenceSessionEpoch = persistenceSessionEpochRef.current;
    const persistenceSessionId = persistenceRecord.sessionId;
    const canFinalizePersistenceSave = () =>
      isShellMountedRef.current &&
      persistenceSessionEpochRef.current === persistenceSessionEpoch &&
      snapshotRef.current.session.sessionId === persistenceSessionId;
    withPersistenceTimeout(
      activePersistenceClient.save(persistenceRecord),
      {
        error: buildPersistenceTimeoutError("save"),
        target: "local"
      },
      1500
    )
      .then((result) => {
        if (!canFinalizePersistenceSave()) {
          return;
        }

        setPersistenceError(result.error);
        setPersistenceState(result.target === "remote" ? "saved" : "local");
        setWorkspaceSessions(listLocalWorkspaceSessions(workspaceIdentity.userId));
      })
      .catch(() => {
        if (!canFinalizePersistenceSave()) {
          return;
        }

        setPersistenceError(buildPersistenceTimeoutError("save"));
        setPersistenceState("unavailable");
      });
    return undefined;
  }, [
    activePersistenceClient,
    persistenceRecord,
    persistenceState,
    workspaceIdentity.userId
  ]);

  function clearDragState() {
    dragCleanupRef.current?.();
    dragCleanupRef.current = null;
    setActiveResizeTarget(null);
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
  }

  function beginLayoutDrag({
    cursor,
    event,
    onMove,
    target
  }: {
    cursor: "col-resize" | "row-resize";
    event: MouseEvent<HTMLElement>;
    onMove: (moveEvent: globalThis.MouseEvent) => void;
    target: ResizeTarget;
  }) {
    event.preventDefault();
    clearDragState();
    setActiveResizeTarget(target);
    document.body.style.cursor = cursor;
    document.body.style.userSelect = "none";

    const handleMouseMove = (moveEvent: globalThis.MouseEvent) => {
      onMove(moveEvent);
    };
    const handleMouseUp = () => {
      clearDragState();
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp, { once: true });
    dragCleanupRef.current = () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }

  function updateLayout(partialState: Partial<WorkspaceLayoutState>) {
    setLayoutState((currentState) => ({
      ...currentState,
      ...partialState
    }));
  }

  function updateWorkspacePreferences(
    partialState: Partial<WorkspacePreferences>
  ) {
    setWorkspacePreferences((currentState) =>
      normalizeWorkspacePreferences({
        ...currentState,
        ...partialState
      })
    );
  }

  function commitImageAttachments(nextAttachments: ImageAttachmentSummary[]) {
    imageAttachmentsRef.current = nextAttachments;
    setImageAttachments(nextAttachments);
  }

  function resetImageUploadBoundary(nextAttachments: ImageAttachmentSummary[]) {
    imageUploadEpochRef.current += 1;
    imageUploadQueueRef.current = Promise.resolve();
    setPendingImageUploadCount(0);
    commitImageAttachments(nextAttachments);
  }

  function createActiveAssistantRequest({
    assistantMessageId,
    forcePreviewOpen,
    pendingRefinement,
    prompt,
    requestMode
  }: {
    assistantMessageId: string;
    forcePreviewOpen: boolean;
    pendingRefinement: PendingArtifactRefinement | null;
    prompt: string;
    requestMode: AssistantRequestMode;
  }): ActiveAssistantRequest {
    assistantRequestCounterRef.current += 1;
    assistantStreamEpochRef.current += 1;

    return Object.freeze({
      abortController: new AbortController(),
      assistantMessageId,
      epoch: assistantStreamEpochRef.current,
      forcePreviewOpen,
      pendingRefinement,
      projectId: workspaceIdentity.projectId,
      prompt,
      requestId: `${workspaceIdentity.sessionId}:request-${assistantRequestCounterRef.current}`,
      requestMode,
      sessionId: workspaceIdentity.sessionId
    });
  }

  function isAssistantRequestEpochCurrent(request: ActiveAssistantRequest) {
    return (
      isShellMountedRef.current &&
      !request.abortController.signal.aborted &&
      assistantStreamEpochRef.current === request.epoch &&
      snapshotRef.current.session.sessionId === request.sessionId
    );
  }

  function isActiveAssistantRequest(request: ActiveAssistantRequest) {
    const activeRequest = activeAssistantRequestRef.current;

    return (
      isAssistantRequestEpochCurrent(request) &&
      activeRequest?.epoch === request.epoch &&
      activeRequest.requestId === request.requestId &&
      activeRequest.sessionId === request.sessionId
    );
  }

  function invalidateActiveAssistantRequest({
    updateUi = true
  }: { updateUi?: boolean } = {}) {
    const activeRequest = activeAssistantRequestRef.current;

    assistantStreamEpochRef.current += 1;
    activeAssistantRequestRef.current = null;
    activeRequest?.abortController.abort();
    if (
      !activeRequest ||
      streamingAssistantIdRef.current === activeRequest.assistantMessageId
    ) {
      streamingAssistantIdRef.current = null;
    }
    activeRequestModeRef.current = "generate";

    if (updateUi && isShellMountedRef.current) {
      setIsStreaming(false);
    }
  }

  function resetAssistantStreamProjection(
    nextStreamEvents: AssistantWorkspaceSnapshot["debug"]["events"] = []
  ) {
    workflowTraceSessionIdRef.current = null;
    streamingAssistantIdRef.current = null;
    activeRequestModeRef.current = "generate";
    setIsStreaming(false);
    setStreamError(null);
    setStreamEvents(nextStreamEvents);
    setClarification(null);
    setSessionIntelligenceMetadata(null);
    setWorkflowTraceEvents([]);
    setCreativeCostRunHistory([]);
    setWorkflowRunId(0);
  }

  function sessionPersistenceClient(sessionId: string) {
    return createWorkspacePersistenceClient({
      projectId: workspaceIdentity.projectId,
      sessionId,
      useProfileIdentity: false,
      userId: workspaceIdentity.userId
    });
  }

  function resetSessionState(nextSnapshot: AssistantWorkspaceSnapshot) {
    invalidateActiveAssistantRequest();
    const normalizedAttachments = normalizeImageAttachments(
      nextSnapshot.multimodal.imageAttachments
    );
    replaceSnapshot(nextSnapshot);
    setConversationEntries(
      buildConversationEntries(
        nextSnapshot.messages,
        createConversationEntryId,
        nextSnapshot.workflow
      )
    );
    resetImageUploadBoundary(normalizedAttachments);
    setImageUploadError(nextSnapshot.multimodal.error ?? null);
    imageAttachmentCounterRef.current = normalizedAttachments.length;
    setActiveArtifactId(nextSnapshot.artifacts[0]?.id ?? "");
    setPreviewArtifactId(getInitialPreviewArtifactId(nextSnapshot));
    setActiveTab(getInitialActiveTab(nextSnapshot));
    setIsPreviewOpen(false);
    setComposerValue("");
    resetAssistantStreamProjection(nextSnapshot.debug.events);
    setPreviewSessionOverride(null);
    setPreviewRuntimeLive(null);
  }

  function activateSession(sessionId: string) {
    if (sessionId === workspaceIdentity.sessionId) {
      return;
    }
    persistenceSessionEpochRef.current += 1;
    invalidateActiveAssistantRequest();
    resetAssistantStreamProjection();
    resetImageUploadBoundary([]);
    setImageUploadError(null);
    setIsAttachmentMenuOpen(false);
    hasLoadedPersistenceRef.current = false;
    lastPersistedFingerprintRef.current = null;
    skipNextPersistenceSaveRef.current = true;
    setPersistenceState("loading");
    setActivePersistenceClient(sessionPersistenceClient(sessionId));
  }

  async function handleCreateSession() {
    const sessionId = `browser-session-${Date.now().toString(36)}-${Math.random()
      .toString(36)
      .slice(2, 8)}`;
    const nextClient = sessionPersistenceClient(sessionId);
    const nextSnapshot = withWorkspaceIdentity(getInitialWorkspaceSnapshot(), {
      ...workspaceIdentity,
      sessionId
    });
    const titledSnapshot = {
      ...nextSnapshot,
      session: {
        ...nextSnapshot.session,
        title: "New creative session"
      }
    };
    const nextLayout = normalizeWorkspaceLayoutState({
      ...layoutState,
      inspectorCollapsed: defaultWorkspaceLayoutState.inspectorCollapsed
    });
    const nextPreferences = normalizeWorkspacePreferences({
      ...workspacePreferences,
      showDebugPanels: defaultWorkspacePreferences.showDebugPanels
    });
    persistenceSessionEpochRef.current += 1;
    hasLoadedPersistenceRef.current = false;
    lastPersistedFingerprintRef.current = null;
    skipNextPersistenceSaveRef.current = true;
    setPersistenceState("loading");
    resetSessionState(titledSnapshot);
    setActiveTab(userModeDefaultInspectorTab);
    setLayoutState(nextLayout);
    setWorkspacePreferences(nextPreferences);
    setActivePersistenceClient(nextClient);
    await nextClient.save(
      createWorkspaceSessionRecord({
        activeArtifactId: "",
        activeInspectorTab: userModeDefaultInspectorTab,
        layout: nextLayout,
        preferences: nextPreferences,
        previewArtifactId: "",
        previewOpen: false,
        snapshot: titledSnapshot
      })
    );
    setWorkspaceSessions(listLocalWorkspaceSessions(workspaceIdentity.userId));
  }

  async function handleSessionRename(sessionId: string, title: string) {
    const nextTitle = title.trim();
    if (!nextTitle) {
      return;
    }

    if (sessionId === workspaceIdentity.sessionId) {
      replaceSnapshot({
        ...snapshot,
        session: { ...snapshot.session, title: nextTitle }
      });
      setSessionUsageSummaries(
        renameSessionUsage(
          workspaceIdentity.userId,
          sessionId,
          nextTitle
        )
      );
      setWorkspaceSessions(listLocalWorkspaceSessions(workspaceIdentity.userId));
      return;
    }

    const client = sessionPersistenceClient(sessionId);
    const loaded = await client.load();
    if (!loaded.record) {
      return;
    }
    await client.save({
      ...loaded.record,
      title: nextTitle,
      snapshot: {
        ...loaded.record.snapshot,
        session: { ...loaded.record.snapshot.session, title: nextTitle }
      }
    });
    setSessionUsageSummaries(
      renameSessionUsage(workspaceIdentity.userId, sessionId, nextTitle)
    );
    setWorkspaceSessions(listLocalWorkspaceSessions(workspaceIdentity.userId));
  }

  async function handleSessionDelete(sessionId: string) {
    const summary = workspaceSessions.find((item) => item.sessionId === sessionId);
    if (!summary) {
      return;
    }

    requestApplicationConfirmation({
      cancelLabel: "Keep session",
      confirmLabel: "Delete session",
      detail:
        "This permanently removes the browser-local session and its saved artifacts. This action cannot be undone.",
      eyebrow: "Session deletion",
      id: `delete-session-${summary.sessionId}`,
      onConfirm: async () => {
        if (sessionId === workspaceIdentity.sessionId) {
          invalidateActiveAssistantRequest();
          resetAssistantStreamProjection();
        }
        await deletePersistedWorkspaceSession({
          identity: {
            projectId: summary.projectId,
            sessionId: summary.sessionId,
            userId: workspaceIdentity.userId
          }
        });
        setSessionUsageSummaries(
          deleteSessionUsage(workspaceIdentity.userId, sessionId)
        );
        const remaining = listLocalWorkspaceSessions(workspaceIdentity.userId);
        setWorkspaceSessions(remaining);
        if (sessionId === workspaceIdentity.sessionId) {
          if (remaining[0]) {
            activateSession(remaining[0].sessionId);
          } else {
            await handleCreateSession();
          }
        }
      },
      title: `Delete ${summary.title}?`,
      tone: "danger"
    });
  }

  function requestApplicationConfirmation(
    request: Omit<ApplicationConfirmationRequest, "returnFocus">
  ) {
    const returnFocus =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    setApplicationConfirmation({ ...request, returnFocus });
    setOpenUtilityPanel(null);
  }

  function closeUtilityPanel(
    panelName: UtilityPanelName,
    options: { restoreFocus?: boolean } = {}
  ) {
    setOpenUtilityPanel(null);
    if (options.restoreFocus === false) {
      return;
    }
    window.requestAnimationFrame(() => {
      (panelName === "theme"
        ? themeTriggerRef.current
        : settingsTriggerRef.current
      )?.focus({ preventScroll: true });
    });
  }

  function toggleUtilityPanel(panelName: UtilityPanelName) {
    setIsAttachmentMenuOpen(false);
    setOpenUtilityPanel((currentPanel) =>
      currentPanel === panelName ? null : panelName
    );
  }

  function focusInspectorTab(nextTab: ProductIntelligenceCategory) {
    window.requestAnimationFrame(() => {
      const tab = document.getElementById(`${getInspectorPanelId(nextTab)}-tab`);
      tab?.scrollIntoView?.({ block: "nearest", inline: "nearest" });
      tab?.focus({ preventScroll: true });
      tab?.scrollIntoView?.({ block: "nearest", inline: "nearest" });
    });
  }

  function openInspectorTabFromUtility(nextTab: ProductIntelligenceCategory) {
    revealInspectorTab(nextTab);
    focusInspectorTab(nextTab);
  }

  function toggleFocusModeFromUtility() {
    setOpenUtilityPanel(null);
    handleFocusModeToggle();
    window.requestAnimationFrame(() => {
      composerTextareaRef.current?.focus({ preventScroll: true });
    });
  }

  function togglePreviewFromUtility() {
    setOpenUtilityPanel(null);
    handlePreviewShelfFromControl();
    window.requestAnimationFrame(() => {
      document
        .querySelector<HTMLElement>('.previewShelf > summary, [aria-label="Settings"]')
        ?.focus({ preventScroll: true });
    });
  }

  function revealInspectorTab(nextTab: ProductIntelligenceCategory) {
    if (isFocusMode) {
      handleFocusModeToggle();
    }

    if (layoutState.inspectorCollapsed) {
      handleInspectorCollapsedChange(false, { preserveFocusMode: true });
    }

    setActiveTab(nextTab);
    setInspectorTabs((currentTabs) =>
      currentTabs.includes(nextTab) ? currentTabs : [...currentTabs, nextTab]
    );
    setIsInspectorAddMenuOpen(false);
    setOpenUtilityPanel(null);
  }

  function openDashboard(category: ProductIntelligenceCategory = activeTab) {
    setDashboardCategory(category);
    setIsDashboardOpen(true);
    setIsInspectorAddMenuOpen(false);
    setOpenUtilityPanel(null);
  }

  function resolvePreviewSourceArtifactId() {
    return (
      interactiveSnapshot.preview.sourceArtifactId ||
      previewArtifactId ||
      activeArtifact.id ||
      getInitialPreviewArtifactId(snapshot)
    );
  }

  function resolvePreviewResetArtifactId() {
    return (
      getInitialPreviewArtifactId(snapshot) ||
      interactiveSnapshot.preview.sourceArtifactId ||
      activeArtifact.id
    );
  }

  function setPreviewContextArtifactId(nextArtifactId: string) {
    setPreviewArtifactId(nextArtifactId);
    setPreviewSessionOverride(null);
    setPreviewRuntimeLive(null);
  }

  function handlePreviewShelfFromControl() {
    if (!interactiveSnapshot.preview.available) {
      return;
    }

    handlePreviewOpenChange(!isPreviewOpen);
    setOpenUtilityPanel(null);
  }

  function appendLocalRuntimeEvent(event: AssistantStreamEvent) {
    const receivedAt = new Date().toISOString();
    const receivedAtMs = Date.now();
    const code = readPayloadText(event, "code") ?? event.event_type;

    workflowTraceSessionIdRef.current = snapshotRef.current.session.sessionId;
    setWorkflowTraceEvents((currentEvents) => [
      ...currentEvents,
      {
        event,
        receivedAt,
        receivedAtMs
      }
    ]);
    setStreamEvents((currentEvents) => [
      ...currentEvents,
      {
        code: `${event.sequence}:${event.event_type}`,
        label: formatRuntimeCode(code),
        detail:
          readPayloadText(event, "message") ??
          readPayloadText(event, "answer") ??
          readPayloadText(event, "text") ??
          "Local operator event received."
      }
    ]);

    const workflowNode = workflowNodeFromAssistantStreamEvent(event);
    if (workflowNode) {
      setWorkflowProgressIndex(
        getWorkflowNodeIndex(interactiveSnapshot.workflow.steps, workflowNode)
      );
    }
  }

  function appendPreviewRuntimeStatusEvent({
    kind,
    route,
    runtimeId,
    source,
    status
  }: PreviewRuntimeStatusTelemetryEvent) {
    const errorScopeKey = [source.fingerprint, kind].join(":");
    const recoveredFromError =
      status.state === "running" &&
      previewRuntimeErrorScopesRef.current.has(errorScopeKey);
    const code =
      status.state === "error"
        ? "preview_runtime_error"
        : recoveredFromError
          ? "preview_runtime_recovered"
          : `preview_runtime_${status.state}`;
    const telemetryScopeId =
      status.state === "idle" || status.state === "starting"
        ? "session"
        : runtimeId;
    const key = [
      "status",
      telemetryScopeId,
      source.fingerprint,
      kind,
      code,
      status.state,
      status.label,
      status.detail
    ].join(":");

    if (previewRuntimeTelemetryKeysRef.current.has(key)) {
      return;
    }

    previewRuntimeTelemetryKeysRef.current.add(key);
    if (status.state === "error") {
      previewRuntimeErrorScopesRef.current.add(errorScopeKey);
    } else if (recoveredFromError) {
      previewRuntimeErrorScopesRef.current.delete(errorScopeKey);
    }
    appendLocalRuntimeEvent({
      event_type: "status",
      sequence: localRuntimeSequenceRef.current++,
      payload: {
        code,
        message: `${status.label}: ${status.detail}`,
        category: "preview_runtime",
        subsystem: `${kind}_sandbox_runtime`,
        preview_runtime: {
          artifact: source.title,
          fingerprint: source.fingerprint,
          kind,
          renderer_id: route.rendererId,
          renderer_label: route.rendererLabel,
          runtime_id: runtimeId,
          state: status.state,
          diagnostics: status.diagnostics ?? [],
          error: status.error?.userMessage ?? null,
          recovered_from_error: recoveredFromError
        }
      }
    });

    if (status.state === "error") {
      const fallbackArtifact = resolveRefinedPreviewFallbackArtifact({
        artifacts: snapshotRef.current.artifacts,
        failedArtifactIds: [
          route.sourceArtifactId,
          route.selectedArtifactId,
          previewArtifactId
        ],
        failedArtifactTitle: source.title
      });

      if (fallbackArtifact) {
        const fallbackOverride = createPreviewSessionOverride(
          fallbackArtifact.id,
          "reloading"
        );

        setPreviewArtifactId(fallbackArtifact.id);
        setPreviewSessionOverride(fallbackOverride);
        setPreviewRuntimeLive(null);
        appendLocalRuntimeEvent({
          event_type: "status",
          sequence: localRuntimeSequenceRef.current++,
          payload: {
            code: "preview_refinement_fallback_requested",
            message: `The refined preview failed to run. Restoring ${fallbackArtifact.title} while keeping ${source.title} as a saved artifact.`,
            category: "preview_runtime",
            subsystem: `${kind}_sandbox_runtime`,
            preview_runtime: {
              artifact: fallbackArtifact.title,
              failed_artifact: source.title,
              kind,
              renderer_id: route.rendererId,
              runtime_id: runtimeId,
              state: "reloading"
            }
          }
        });
      }
    }
  }

  function appendPreviewRuntimeFrameEvent({
    kind,
    route,
    runtimeId,
    sample,
    source
  }: PreviewRuntimeFrameTelemetryEvent) {
    const key = ["frame", runtimeId, source.fingerprint, kind, "first"].join(":");

    if (previewRuntimeTelemetryKeysRef.current.has(key)) {
      return;
    }

    previewRuntimeTelemetryKeysRef.current.add(key);
    appendLocalRuntimeEvent({
      event_type: "status",
      sequence: localRuntimeSequenceRef.current++,
      payload: {
        code: "preview_runtime_frame",
        message: `First preview frame rendered for ${source.title}.`,
        category: "preview_runtime",
        subsystem: `${kind}_sandbox_runtime`,
        preview_runtime: {
          artifact: source.title,
          fingerprint: source.fingerprint,
          kind,
          renderer_id: route.rendererId,
          renderer_label: route.rendererLabel,
          runtime_id: runtimeId,
          rendered_at_ms: sample.renderedAtMs,
          state: "running"
        }
      }
    });
  }

  function handlePreviewRuntimeDiagnostics(
    nextRuntimeLive: Omit<RuntimeConsoleLiveSnapshot, "updatedAt">
  ) {
    setPreviewRuntimeLive({
      ...nextRuntimeLive,
      updatedAt: new Date().toISOString()
    });
    if (nextRuntimeLive.status.state === "running") {
      setPreviewSessionOverride((currentOverride) =>
        currentOverride?.mode === "reloading" ||
        currentOverride?.mode === "restarting"
          ? { ...currentOverride, mode: "settled" }
          : currentOverride
      );
    }
  }

  function setApprovalRequestState(
    request: HitlApprovalRequest,
    nextState: HitlActionState,
    options: { failureReason?: string | null } = {}
  ) {
    const nextRequest = updateHitlApprovalRequest(
      request,
      nextState,
      new Date().toISOString(),
      options.failureReason ?? null
    );
    setApprovalRequests((currentRequests) =>
      currentRequests.map((currentRequest) =>
        currentRequest.id === request.id ? nextRequest : currentRequest
      )
    );
    appendLocalRuntimeEvent(
      buildHitlApprovalStreamEvent({
        request: nextRequest,
        sequence: localRuntimeSequenceRef.current++,
        state: nextState,
        workflow: interactiveSnapshot.workflow
      })
    );

    return nextRequest;
  }

  function requestOperatorApproval({
    actionId,
    artifactTitle,
    execute
  }: {
    actionId: HitlActionId;
    artifactTitle?: string | null;
    execute: ApprovalActionExecutor;
  }) {
    if (blockingApprovalRequest) {
      approvalCardRef.current?.focus();
      return;
    }

    approvalFocusOriginRef.current =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    const request = createHitlApprovalRequest({
      actionId,
      artifactTitle,
      id: createApprovalRequestId(),
      nodeId: interactiveSnapshot.workflow.currentNode,
      workspaceName: interactiveSnapshot.workspace.name
    });
    approvalExecutorsRef.current[request.id] = execute;
    setDismissedApprovalRequestId(null);
    setApprovalRequests((currentRequests) => [...currentRequests, request]);
    appendLocalRuntimeEvent(
      buildHitlApprovalStreamEvent({
        request,
        sequence: localRuntimeSequenceRef.current++,
        state: "pending_approval",
        workflow: interactiveSnapshot.workflow
      })
    );
    setOpenUtilityPanel(null);
    focusApprovalCard();
  }

  function focusApprovalCard() {
    window.requestAnimationFrame(() => {
      approvalCardRef.current?.scrollIntoView?.({
        behavior: "auto",
        block: "nearest",
        inline: "nearest"
      });
      approvalCardRef.current?.focus({ preventScroll: true });
    });
  }

  async function handleApprovalApprove(request: HitlApprovalRequest) {
    const execute = approvalExecutorsRef.current[request.id];
    const approvedRequest = setApprovalRequestState(request, "approved");
    const executingRequest = setApprovalRequestState(approvedRequest, "executing");

    if (!execute) {
      setApprovalRequestState(executingRequest, "failed", {
        failureReason: "No approval executor was available for this action."
      });
      focusApprovalCard();
      return;
    }

    try {
      await Promise.resolve(execute());
      setApprovalRequestState(executingRequest, "completed");
    } catch (error) {
      setApprovalRequestState(executingRequest, "failed", {
        failureReason:
          error instanceof Error
            ? error.message
            : "The operator action could not be completed."
      });
    } finally {
      delete approvalExecutorsRef.current[request.id];
      focusApprovalCard();
    }
  }

  function handleApprovalReject(request: HitlApprovalRequest) {
    delete approvalExecutorsRef.current[request.id];
    setApprovalRequestState(request, "rejected");
    focusApprovalCard();
  }

  function handleApprovalDismiss(request: HitlApprovalRequest) {
    if (!isHitlApprovalTerminalState(request.state)) {
      return;
    }

    setDismissedApprovalRequestId(request.id);
    window.requestAnimationFrame(() => {
      const origin = approvalFocusOriginRef.current;
      const fallback = settingsTriggerRef.current ?? composerTextareaRef.current;
      (origin?.isConnected ? origin : fallback)?.focus({ preventScroll: true });
      approvalFocusOriginRef.current = null;
    });
  }

  function clearWorkspaceSession() {
    invalidateActiveAssistantRequest();
    const clearedSnapshot = withWorkspaceIdentity(
      getInitialWorkspaceSnapshot(),
      workspaceIdentity
    );

    clearFeedbackTimers();
    setCopyFeedback(null);
    setTransferFeedback(null);
    setArtifactTransferError(null);
    previousPreviewRuntimeSessionKeyRef.current = null;
    previewRuntimeTelemetryKeysRef.current.clear();
    previewRuntimeErrorScopesRef.current.clear();
    replaceSnapshot(clearedSnapshot);
    setConversationEntries(
      buildConversationEntries(
        clearedSnapshot.messages,
        createConversationEntryId,
        clearedSnapshot.workflow
      )
    );
    resetImageUploadBoundary(
      normalizeImageAttachments(clearedSnapshot.multimodal.imageAttachments)
    );
    setImageUploadError(clearedSnapshot.multimodal.error ?? null);
    imageAttachmentCounterRef.current = normalizeImageAttachments(
      clearedSnapshot.multimodal.imageAttachments
    ).length;
    setComposerValue("");
    setActiveTab(getInitialActiveTab(clearedSnapshot));
    setActiveArtifactId(clearedSnapshot.artifacts[0]?.id ?? "");
    setPreviewArtifactId(getInitialPreviewArtifactId(clearedSnapshot));
    setIsPreviewOpen(clearedSnapshot.preview.active);
    setIsPreviewFullscreen(false);
    setPreviewSessionOverride(null);
    setWorkflowProgressIndex(getInitialWorkflowIndex(clearedSnapshot.workflow.steps));
    setWorkflowRunId(0);
    resetAssistantStreamProjection(clearedSnapshot.debug.events);
    setClarification(clearedSnapshot.clarification ?? null);
    setLastRemovedArtifact(null);
    setLastRestoredArtifact(null);
    setPreviewRuntimeLive(null);
    updateLayout({
      inspectorCollapsed: defaultWorkspacePreferences.showDebugPanels
        ? false
        : defaultWorkspaceLayoutState.inspectorCollapsed,
      previewHeight: defaultWorkspaceLayoutState.previewHeight
    });
    updateWorkspacePreferences({
      autoOpenPreview: defaultWorkspacePreferences.autoOpenPreview,
      showDebugPanels: defaultWorkspacePreferences.showDebugPanels
    });
    setOpenUtilityPanel(null);
    setIsAttachmentMenuOpen(false);
    setIsDemoModeOpen(false);
    setActiveDemoScenarioId(getDefaultDemoModeScenario().id);
  }

  function handleInspectorCollapsedChange(
    nextCollapsed: boolean,
    options: { preserveFocusMode?: boolean } = {}
  ) {
    updateLayout({ inspectorCollapsed: nextCollapsed });
    if (!options.preserveFocusMode && isFocusMode) {
      focusRestoreRef.current = null;
      setIsFocusMode(false);
    }
  }

  function handlePreviewOpenChange(
    nextOpen: boolean,
    options: { preserveFocusMode?: boolean } = {}
  ) {
    setIsPreviewOpen(nextOpen);
    if (!nextOpen) {
      setIsPreviewFullscreen(false);
    }
    if (!options.preserveFocusMode && isFocusMode) {
      focusRestoreRef.current = null;
      setIsFocusMode(false);
    }
  }

  function handlePreviewFullscreenChange(nextFullscreen: boolean) {
    if (!interactiveSnapshot.preview.available) {
      return;
    }

    if (nextFullscreen && !isPreviewOpen) {
      handlePreviewOpenChange(true, { preserveFocusMode: true });
    }

    setIsPreviewFullscreen(nextFullscreen);
  }

  function handleFocusModeToggle() {
    if (isFocusMode) {
      const focusRestore = focusRestoreRef.current;

      handleInspectorCollapsedChange(
        focusRestore?.inspectorCollapsed ?? false,
        { preserveFocusMode: true }
      );
      updateLayout({
        sidebarCollapsed: focusRestore?.sidebarCollapsed ?? false
      });
      if (interactiveSnapshot.preview.available) {
        handlePreviewOpenChange(focusRestore?.previewOpen ?? false, {
          preserveFocusMode: true
        });
      }
      focusRestoreRef.current = null;
      setIsFocusMode(false);
      return;
    }

    focusRestoreRef.current = {
      inspectorCollapsed: isInspectorCollapsed,
      previewOpen: isPreviewOpen,
      sidebarCollapsed: layoutState.sidebarCollapsed
    };
    setIsPreviewFullscreen(false);
    handleInspectorCollapsedChange(true, { preserveFocusMode: true });
    updateLayout({ sidebarCollapsed: true });
    if (interactiveSnapshot.preview.available) {
      handlePreviewOpenChange(false, { preserveFocusMode: true });
    }
    setIsFocusMode(true);
  }

  function handlePreviewSessionRestart() {
    requestOperatorApproval({
      actionId: "preview_runtime_restart",
      artifactTitle: interactiveSnapshot.preview.artifactName,
      execute: () => {
        const nextArtifactId = resolvePreviewSourceArtifactId();

        setPreviewArtifactId(nextArtifactId);
        setPreviewSessionOverride(
          createPreviewSessionOverride(nextArtifactId, "restarting")
        );
        setPreviewRuntimeLive(null);
        handlePreviewOpenChange(true, { preserveFocusMode: true });
      }
    });
  }

  function handlePreviewStateClear() {
    requestOperatorApproval({
      actionId: "preview_runtime_clear",
      artifactTitle: interactiveSnapshot.preview.artifactName,
      execute: () => {
        const nextArtifactId = resolvePreviewSourceArtifactId();

        setPreviewArtifactId(nextArtifactId);
        setPreviewSessionOverride(
          createPreviewSessionOverride(nextArtifactId, "cleared")
        );
        setPreviewRuntimeLive(null);
        handlePreviewOpenChange(true, { preserveFocusMode: true });
      }
    });
  }

  function handlePreviewStateReload() {
    if (!interactiveSnapshot.preview.available) {
      return;
    }

    if (previewSessionOverride?.mode === "cleared") {
      appendPreviewRuntimeReloadEvent(resolvePreviewSourceArtifactId());
      setPreviewSessionOverride(null);
      handlePreviewOpenChange(true, { preserveFocusMode: true });
      return;
    }

    const currentPreviewArtifact =
      snapshot.artifacts.find((artifact) => artifact.id === previewArtifactId) ??
      activeArtifact;
    const nextArtifactId = isArtifactPreviewable(currentPreviewArtifact)
      ? resolvePreviewSourceArtifactId()
      : resolvePreviewResetArtifactId();
    const nextSessionOverride = createPreviewSessionOverride(
      nextArtifactId,
      "reloading"
    );

    setPreviewArtifactId(nextArtifactId);
    setPreviewSessionOverride(nextSessionOverride);
    setPreviewRuntimeLive(null);
    appendPreviewRuntimeReloadEvent(nextArtifactId, nextSessionOverride.requestedAt);
    handlePreviewOpenChange(true, { preserveFocusMode: true });
  }

  function appendPreviewRuntimeReloadEvent(
    artifactId: string,
    requestedAt = new Date().toISOString()
  ) {
    const runtimeKind = getExecutablePreviewRuntimeKind(previewRendererRoute);
    const artifactTitle =
      snapshot.artifacts.find((artifact) => artifact.id === artifactId)?.title ??
      previewRuntimeSource.title;

    appendLocalRuntimeEvent({
      event_type: "status",
      sequence: localRuntimeSequenceRef.current++,
      payload: {
        code: "preview_runtime_reload_requested",
        message: `Reload requested for ${artifactTitle}.`,
        category: "preview_runtime",
        subsystem: runtimeKind ? `${runtimeKind}_sandbox_runtime` : "preview_runtime",
        preview_runtime: {
          artifact: artifactTitle,
          fingerprint: previewRuntimeSource.fingerprint,
          kind: runtimeKind ?? previewRendererRoute.surfaceKind,
          renderer_id: previewRendererRoute.rendererId,
          renderer_label: previewRendererRoute.rendererLabel,
          requested_at: requestedAt,
          state: "reloading"
        }
      }
    });
  }

  function handleInspectorResizeStart(event: MouseEvent<HTMLElement>) {
    if (isInspectorCollapsed || isFocusMode) {
      return;
    }

    const startX = event.clientX;
    const startWidth = layoutState.inspectorWidth;

    beginLayoutDrag({
      cursor: "col-resize",
      event,
      onMove: (moveEvent) => {
        const deltaX = startX - moveEvent.clientX;
        updateLayout({
          inspectorWidth: clampNumber(
            startWidth + deltaX,
            workspaceLayoutBounds.minInspectorWidth,
            workspaceLayoutBounds.maxInspectorWidth
          )
        });
      },
      target: "inspector"
    });
  }

  function handleInspectorResizeKeyDown(event: KeyboardEvent<HTMLElement>) {
    if (isInspectorCollapsed || isFocusMode) {
      return;
    }

    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault();
      const delta = event.key === "ArrowLeft" ? 16 : -16;
      updateLayout({
        inspectorWidth: clampNumber(
          layoutState.inspectorWidth + delta,
          workspaceLayoutBounds.minInspectorWidth,
          workspaceLayoutBounds.maxInspectorWidth
        )
      });
    }
  }

  function handlePreviewResizeStart(event: MouseEvent<HTMLElement>) {
    if (!isPreviewOpen || isFocusMode) {
      return;
    }

    const startY = event.clientY;
    const startHeight = layoutState.previewHeight;

    beginLayoutDrag({
      cursor: "row-resize",
      event,
      onMove: (moveEvent) => {
        const deltaY = moveEvent.clientY - startY;
        updateLayout({
          previewHeight: clampNumber(
            startHeight + deltaY,
            workspaceLayoutBounds.minPreviewHeight,
            workspaceLayoutBounds.maxPreviewHeight
          )
        });
      },
      target: "preview"
    });
  }

  function handlePreviewResizeKeyDown(event: KeyboardEvent<HTMLElement>) {
    if (!isPreviewOpen || isFocusMode) {
      return;
    }

    if (event.key === "ArrowUp" || event.key === "ArrowDown") {
      event.preventDefault();
      const delta = event.key === "ArrowDown" ? 16 : -16;
      updateLayout({
        previewHeight: clampNumber(
          layoutState.previewHeight + delta,
          workspaceLayoutBounds.minPreviewHeight,
          workspaceLayoutBounds.maxPreviewHeight
        )
      });
    }
  }

  function handleImageFilesSelected(files: File[]) {
    setIsAttachmentMenuOpen(false);

    if (files.length === 0) {
      return Promise.resolve();
    }

    const uploadEpoch = imageUploadEpochRef.current;
    setImageUploadError(null);
    setPendingImageUploadCount((currentCount) => currentCount + 1);

    const uploadTask = imageUploadQueueRef.current
      .catch(() => undefined)
      .then(async () => {
        if (uploadEpoch !== imageUploadEpochRef.current) {
          return;
        }

        const nextAttachments: ImageAttachmentSummary[] = [];
        let nextError: WorkstationError | null = null;

        for (const file of files) {
          const result = await createImageAttachmentFromFile({
            createdAt: new Date().toISOString(),
            existingCount:
              imageAttachmentsRef.current.length + nextAttachments.length,
            file,
            id: createImageAttachmentId(file.name)
          });

          if (uploadEpoch !== imageUploadEpochRef.current) {
            return;
          }

          if (result.ok) {
            nextAttachments.push(result.attachment);
          } else {
            nextError = result.error;
            break;
          }
        }

        if (!nextError && nextAttachments.length > 0) {
          const updatedAttachments = [
            ...imageAttachmentsRef.current,
            ...nextAttachments
          ];
          commitImageAttachments(updatedAttachments);
          appendImageReferenceRuntimeEvent({
            attachments: updatedAttachments,
            code: "image_reference_attached",
            message: `${nextAttachments.length} ${pluralize(
              nextAttachments.length,
              "image reference",
              "image references"
            )} attached to the next request.`
          });
        }

        setImageUploadError(nextError);
      })
      .finally(() => {
        if (uploadEpoch === imageUploadEpochRef.current) {
          setPendingImageUploadCount((currentCount) =>
            Math.max(0, currentCount - 1)
          );
        }
      });

    imageUploadQueueRef.current = uploadTask;
    return uploadTask;
  }

  function handleImageAttachmentRemove(attachmentId: string) {
    const nextAttachments = imageAttachmentsRef.current.filter(
      (attachment) => attachment.id !== attachmentId
    );
    commitImageAttachments(nextAttachments);
    setImageUploadError(null);
    appendImageReferenceRuntimeEvent({
      attachments: nextAttachments,
      code: "image_reference_removed",
      message: "Image reference removed before submission."
    });
  }

  function handleImageUploadErrorDismiss() {
    setImageUploadError(null);
  }

  function createImageAttachmentId(fileName: string) {
    imageAttachmentCounterRef.current += 1;
    const safeName = fileName
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "")
      .slice(0, 32);

    return `image-reference-${imageAttachmentCounterRef.current}${
      safeName ? `-${safeName}` : ""
    }`;
  }

  function appendImageReferenceRuntimeEvent({
    attachments,
    code,
    message
  }: {
    attachments: ImageAttachmentSummary[];
    code:
      | "image_reference_attached"
      | "image_reference_removed"
      | "image_reference_submitted";
    message: string;
  }) {
    appendLocalRuntimeEvent({
      event_type: "status",
      sequence: localRuntimeSequenceRef.current++,
      payload: {
        code,
        message,
        category: "multimodal",
        subsystem: "image_upload",
        multimodal: {
          image_count: attachments.length,
          images: attachments.map((attachment) => ({
            id: attachment.id,
            name: attachment.name,
            mime_type: attachment.mimeType,
            size_bytes: attachment.sizeBytes
          }))
        }
      }
    });
  }

  function handleEmptyStatePromptSelect(prompt: string) {
    setComposerValue(prompt);

    window.requestAnimationFrame(() => {
      composerTextareaRef.current?.focus();
    });
  }

  function handleDemoModeToggle() {
    const nextOpen = !isDemoModeOpen;

    setIsDemoModeOpen(nextOpen);
    if (nextOpen) {
      setIsPreviewOpen(false);
      setIsPreviewFullscreen(false);
    }
    setOpenUtilityPanel(null);
    setIsAttachmentMenuOpen(false);
  }

  function handleDemoScenarioSelect(scenario: DemoModeScenario) {
    setActiveDemoScenarioId(scenario.id);
  }

  function handleDemoScenarioLoad(scenario: DemoModeScenario) {
    if (
      activeAssistantRequestRef.current ||
      isStreaming ||
      isImageUploadPending
    ) {
      return;
    }

    if (scenario.requiresImageAttachment && imageAttachments.length === 0) {
      return;
    }

    setActiveDemoScenarioId(scenario.id);
    setIsDemoModeOpen(false);
    setOpenUtilityPanel(null);
    setIsAttachmentMenuOpen(false);
    void submitAssistantRequest({
      forcePreviewOpen: true,
      prompt: scenario.prompt,
      workflowModeOverride: scenario.workflowMode
    });
  }

  function handleInspectorTabClose(tab: ProductIntelligenceCategory) {
    if (tab === "Overview") {
      return;
    }

    const closingIndex = visibleInspectorTabs.indexOf(tab);
    const remainingVisibleTabs = visibleInspectorTabs.filter(
      (item) => item !== tab
    );
    setInspectorTabs((currentTabs) => currentTabs.filter((item) => item !== tab));
    if (activeTab === tab) {
      setActiveTab(
        remainingVisibleTabs[
          Math.min(Math.max(closingIndex, 0), remainingVisibleTabs.length - 1)
        ] ??
          (workspacePreferences.showDebugPanels
            ? "Overview"
            : userModeDefaultInspectorTab)
      );
    }
    setIsInspectorAddMenuOpen(false);
  }

  async function handleComposerSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const prompt = composerValue.trim();

    if (!hasLoadedPersistenceRef.current || !prompt || isImageUploadPending) {
      return;
    }

    setComposerValue("");
    setIsAttachmentMenuOpen(false);
    const clarificationOption = resolveClarificationNumericAnswer(
      clarification,
      prompt
    );
    if (clarificationOption) {
      await submitClarificationAnswer(clarificationOption);
      return;
    }

    const artifactFollowUp = resolveArtifactFollowUp({
      activeArtifact,
      artifacts: interactiveSnapshot.artifacts,
      prompt
    });
    if (artifactFollowUp.kind === "refinement") {
      await handleArtifactRefine(artifactFollowUp.artifact, prompt, {
        displayMessage: prompt
      });
      return;
    }

    await submitAssistantRequest({ prompt });
  }

  async function handleArtifactRefine(
    artifact: ArtifactSummary,
    instruction: string,
    options: { displayMessage?: string } = {}
  ) {
    const prompt = instruction.trim();

    if (!prompt) {
      return;
    }

    const artifactDocument = buildArtifactDocument(interactiveSnapshot, artifact);
    const artifactRefinement = buildArtifactRefinementRequest({
      artifact,
      document: artifactDocument,
      instruction: prompt
    });

    setActiveArtifactId(artifact.id);
    if (isArtifactPreviewable(artifact)) {
      setPreviewContextArtifactId(artifact.id);
    }

    await submitAssistantRequest({
      artifactRefinement,
      prompt,
      userMessageContent: options.displayMessage
    });
  }

  async function handleClarificationOptionSelect(option: string) {
    await submitClarificationAnswer(option);
  }

  async function submitClarificationAnswer(option: string) {
    if (!clarification || isStreaming) {
      return;
    }

    await submitAssistantRequest({
      clarificationResponse: option,
      prompt: buildClarificationContinuationPrompt(clarification, option)
    });
  }

  async function submitAssistantRequest({
    artifactRefinement,
    clarificationResponse,
    forcePreviewOpen = false,
    prompt,
    userMessageContent,
    workflowModeOverride
  }: {
    artifactRefinement?: AssistantArtifactRefinementRequest;
    clarificationResponse?: string;
    forcePreviewOpen?: boolean;
    prompt: string;
    userMessageContent?: string;
    workflowModeOverride?: WorkflowExecutionMode;
  }) {
    if (
      activeAssistantRequestRef.current ||
      !hasLoadedPersistenceRef.current ||
      isStreaming ||
      isImageUploadPending
    ) {
      return;
    }

    const timestamp = formatMessageTime();
    const userMessageId = createConversationEntryId();
    const assistantMessageId = createConversationEntryId();
    const requestMode = resolveAssistantRequestMode({
      hasArtifactRefinement: Boolean(artifactRefinement),
      hasClarificationResponse: Boolean(clarificationResponse),
      prompt
    });
    const pendingRefinement = artifactRefinement
      ? {
          ...artifactRefinement,
          requestedAt: new Date().toISOString()
        }
      : null;
    const visibleUserMessage =
      userMessageContent ??
      (clarificationResponse
        ? `Clarification: ${clarificationResponse}`
        : artifactRefinement
          ? `Refine ${artifactRefinement.title}: ${prompt}`
          : prompt);
    const initialWorkflowActivity = deriveWorkflowRuntimeActivity({
      currentNode:
        artifactRefinement
          ? "refinement"
          : requestMode === "explain"
            ? "routing"
            : "planning",
      productOutcome: null,
      workflowStatus: "running"
    });
    const activeRequest = createActiveAssistantRequest({
      assistantMessageId,
      forcePreviewOpen,
      pendingRefinement,
      prompt,
      requestMode
    });
    const streamRuntime: AssistantStreamRuntimeState = {
      hasPreviewRuntimeEvent: false
    };

    activeAssistantRequestRef.current = activeRequest;
    activeRequestModeRef.current = requestMode;
    streamingAssistantIdRef.current = assistantMessageId;
    setConversationEntries((currentMessages) => [
      ...currentMessages,
      {
        content: visibleUserMessage,
        activity: null,
        id: userMessageId,
        pending: false,
        phase: "complete",
        role: "user",
        time: timestamp
      },
      {
        content: "",
        activity: conversationActivityForRequestMode(
          initialWorkflowActivity,
          requestMode
        ),
        id: assistantMessageId,
        pending: true,
        phase: conversationPhaseForRequestActivity(
          initialWorkflowActivity,
          requestMode
        ),
        requestMode,
        role: "assistant",
        time: timestamp
      }
    ]);
    setComposerValue("");
    setWorkflowProgressIndex(0);
    setWorkflowRunId(0);
    setStreamError(null);
    setStreamEvents([]);
    setClarification(null);
    setSessionIntelligenceMetadata(null);
    setWorkflowTraceEvents([]);
    workflowTraceSessionIdRef.current = activeRequest.sessionId;
    previewRuntimeTelemetryKeysRef.current.clear();
    previewRuntimeErrorScopesRef.current.clear();
    setIsStreaming(true);
    setActiveTab("Overview");

    let streamedAnswer = "";
    let receivedTerminalStreamError = false;
    let assistantFinalized = false;
    let latestWorkflowActivity = initialWorkflowActivity;
    const requestAttachments = toAssistantRequestImageAttachments(
      imageAttachmentsRef.current
    );
    if (requestAttachments.length > 0) {
      commitImageAttachments([]);
      setImageUploadError(null);
      appendImageReferenceRuntimeEvent({
        attachments: [],
        code: "image_reference_submitted",
        message: `${requestAttachments.length} ${pluralize(
          requestAttachments.length,
          "image reference was",
          "image references were"
        )} sent for this request and removed from the composer.`
      });
    }
    const personalizationContext = selectPersonalizationContext({
      enabled: workspacePreferences.personalizationEnabled,
      prompt,
      signals: workspacePreferences.feedbackSignals
    });

    try {
      const streamRequest: AssistantStreamRequest = {
        conversationId: activeRequest.sessionId,
        generationControls: {
          profile: buildGenerationControls(workspacePreferences.creativity).profile
        },
        mode: requestMode,
        personalizationContext: {
          categories: personalizationContext.categories,
          enabled: personalizationContext.enabled,
          signalCount: personalizationContext.signalCount
        },
        projectId: activeRequest.projectId,
        query: prompt,
        workflowMode: workflowModeOverride ?? workflowMode
      };

      if (clarificationResponse) {
        streamRequest.clarificationResponse = clarificationResponse;
      }

      if (artifactRefinement) {
        streamRequest.artifactRefinement = artifactRefinement;
        if (artifactRefinement.domain) {
          streamRequest.domain = artifactRefinement.domain;
          streamRequest.domains = [artifactRefinement.domain];
        }
      }

      if (requestAttachments.length > 0) {
        streamRequest.attachments = requestAttachments;
      }

      const streamBinding = Object.freeze({
        epoch: activeRequest.epoch,
        requestId: activeRequest.requestId,
        sessionId: activeRequest.sessionId,
        signal: activeRequest.abortController.signal
      });

      for await (const streamEvent of streamAssistantEvents(
        streamRequest,
        streamBinding
      )) {
        if (!isActiveAssistantRequest(activeRequest)) {
          return;
        }
        latestWorkflowActivity = deriveWorkflowRuntimeActivityForStreamEvent(
          streamEvent
        );
        applyStreamEventToWorkspace(streamEvent, activeRequest, streamRuntime);

        if (
          streamEvent.event_type === "token_delta" &&
          !receivedTerminalStreamError
        ) {
          const delta = readPayloadText(streamEvent, "text");
          if (delta) {
            streamedAnswer += delta;
            startTransition(() => {
              updateStreamingAssistantMessage(activeRequest, {
                activity: conversationActivityForRequestMode(
                  latestWorkflowActivity,
                  requestMode
                ),
                content: streamingConversationSummaryForMode(requestMode),
                phase: conversationPhaseForRequestActivity(
                  latestWorkflowActivity,
                  requestMode
                )
              });
            });
          }
        }

        if (streamEvent.event_type === "final" && !receivedTerminalStreamError) {
          const answer = readPayloadText(streamEvent, "answer");
          streamedAnswer = answer ?? streamedAnswer;
          const reconciledProductOutcome =
            snapshotRef.current.workflow.productOutcome;
          const reconciledActivity =
            reconciledProductOutcome?.product_outcome === "SUCCESS"
              ? deriveWorkflowRuntimeActivity({
                  currentNode: "finalization",
                  productOutcome: reconciledProductOutcome,
                  workflowStatus: "completed"
                })
              : latestWorkflowActivity;
          const conversationOutcome = formatConversationOutcome(
            reconciledActivity
          );
          finalizeStreamingAssistantMessage(activeRequest, {
            activity: conversationOutcome.activity,
            content: buildAssistantConversationSummary(streamedAnswer, requestMode),
            phase: conversationOutcome.phase
          });
          assistantFinalized = true;
        }

        if (streamEvent.event_type === "error") {
          receivedTerminalStreamError = true;
          const failedActivity = deriveWorkflowRuntimeActivityForStreamEvent(
            streamEvent
          );
          const error =
            readStreamEventError(streamEvent) ??
            createWorkstationError({
              type: "assistant_stream_failed",
              category: "stream",
              subsystem: "assistant_stream",
              userMessage: "The live response stopped before completion.",
              recoverable: true,
              suggestedAction: "Retry the request from the composer.",
              retryLabel: "Send prompt again",
              resetLabel: "Clear workspace session"
            });
          setStreamError(error);
          finalizeStreamingAssistantMessage(activeRequest, {
            activity: failedActivity.detail,
            content: streamedAnswer
              ? `${buildAssistantConversationSummary(
                  streamedAnswer,
                  requestMode
                )}\n\nLive response error: ${error.userMessage}`
              : `Live response error: ${error.userMessage}`,
            phase: terminalConversationPhaseForWorkflowActivity(failedActivity)
          });
          assistantFinalized = true;
        }
      }

      if (!isActiveAssistantRequest(activeRequest)) {
        return;
      }

      if (!receivedTerminalStreamError && !assistantFinalized && streamedAnswer) {
        const completedActivity = latestWorkflowActivity.terminal
          ? latestWorkflowActivity
          : deriveWorkflowRuntimeActivity({
              currentNode: "finalization",
              productOutcome: null,
              workflowStatus: "completed"
            });
        finalizeStreamingAssistantMessage(activeRequest, {
          activity: completedActivity.detail,
          content: buildAssistantConversationSummary(streamedAnswer, requestMode),
          phase: terminalConversationPhaseForWorkflowActivity(completedActivity)
        });
      } else if (!receivedTerminalStreamError && !assistantFinalized) {
        const error = createWorkstationError({
          type: "stream_ended_before_completion",
          category: "stream",
          subsystem: "assistant_stream",
          userMessage: "The live response ended before completion.",
          recoverable: true,
          suggestedAction: "Retry the request from the composer.",
          retryLabel: "Send prompt again",
          resetLabel: "Clear workspace session"
        });
        setStreamError(error);
        const failedActivity = deriveWorkflowRuntimeActivity({
          currentNode: "failure",
          productOutcome: null,
          workflowStatus: "failed"
        });
        finalizeStreamingAssistantMessage(activeRequest, {
          activity: failedActivity.detail,
          content: error.userMessage,
          phase: terminalConversationPhaseForWorkflowActivity(failedActivity)
        });
      }
    } catch (error) {
      if (!isActiveAssistantRequest(activeRequest)) {
        return;
      }
      const streamFailure =
        error instanceof Error && "detail" in error && error.detail
          ? (error.detail as WorkstationError)
          : createWorkstationError({
              type: "assistant_stream_unavailable",
              category: "stream",
              subsystem: "assistant_stream",
              userMessage: "The live response is unavailable.",
              debugMessage: error instanceof Error ? error.message : null,
              recoverable: true,
              suggestedAction:
                "Retry the request when the connection recovers, or continue with the local draft.",
              retryLabel: "Send prompt again",
              resetLabel: "Clear workspace session"
            });
      const fallbackMessage = `Live response unavailable; showing a local draft. ${buildLocalDraftReply(
        prompt,
        artifactRefinement?.title ?? activeArtifact.title
      )}`;
      setStreamError(streamFailure);
      finalizeStreamingAssistantMessage(activeRequest, {
        activity: "Switching to a local draft.",
        content: fallbackMessage,
        phase: "fallback"
      });
      setWorkflowProgressIndex(0);
      setWorkflowRunId((currentRunId) => currentRunId + 1);
    } finally {
      if (isActiveAssistantRequest(activeRequest)) {
        activeAssistantRequestRef.current = null;
        if (streamingAssistantIdRef.current === activeRequest.assistantMessageId) {
          streamingAssistantIdRef.current = null;
        }
        activeRequestModeRef.current = "generate";
        setIsStreaming(false);
      }
    }
  }

  function applyStreamEventToWorkspace(
    streamEvent: AssistantStreamEvent,
    activeRequest: ActiveAssistantRequest,
    streamRuntime: AssistantStreamRuntimeState
  ) {
    if (!isActiveAssistantRequest(activeRequest)) {
      return;
    }
    const receivedAt = new Date().toISOString();
    const receivedAtMs = Date.now();
    const workflowActivity = deriveWorkflowRuntimeActivityForStreamEvent(
      streamEvent
    );

    workflowTraceSessionIdRef.current = activeRequest.sessionId;
    setWorkflowTraceEvents((currentEvents) => [
      ...currentEvents,
      {
        event: streamEvent,
        receivedAt,
        receivedAtMs
      }
    ]);
    setStreamEvents((currentEvents) => [
      ...currentEvents,
      {
        code: `${streamEvent.sequence}:${streamEvent.event_type}`,
        label: streamEvent.event_type,
        detail:
          readPayloadText(streamEvent, "message") ??
          readPayloadText(streamEvent, "answer") ??
          readPayloadText(streamEvent, "text") ??
          "Live response event received."
      }
    ]);

    const sessionIntelligenceUpdate = readSessionIntelligenceMetadata(
      streamEvent.payload
    );
    if (sessionIntelligenceUpdate) {
      setSessionIntelligenceMetadata(sessionIntelligenceUpdate);
    }

    const clarificationUpdate = readClarificationSummary(
      streamEvent.payload.clarification
    );
    if (clarificationUpdate) {
      setClarification(clarificationUpdate);
      setActiveTab("Overview");
    }

    const workflowMetadata =
      typeof streamEvent.payload.workflow === "object" &&
      streamEvent.payload.workflow !== null
        ? (streamEvent.payload.workflow as Record<string, unknown>)
        : null;
    const creativePlanUpdate = readCreativeExecutionPlanSummary(
      streamEvent.payload.creative_plan ??
        streamEvent.payload.creativePlan ??
        workflowMetadata?.creative_plan ??
        workflowMetadata?.creativePlan
    );
    if (creativePlanUpdate) {
      replaceSnapshot({
        ...snapshotRef.current,
        creativePlan: creativePlanUpdate
      });
    }

    let currentSnapshot = snapshotRef.current;

    if (readPayloadText(streamEvent, "status") === "provider_fallback_selected") {
      currentSnapshot = {
        ...currentSnapshot,
        workflow: {
          ...currentSnapshot.workflow,
          productOutcome: createProviderFallbackInProgressOutcome()
        }
      };
      replaceSnapshot(currentSnapshot);
    }

    const workflowNode = workflowNodeFromAssistantStreamEvent(streamEvent);
    if (workflowNode) {
      setWorkflowProgressIndex(
        getWorkflowNodeIndex(currentSnapshot.workflow.steps, workflowNode)
      );
    }

    if (
      streamEvent.event_type !== "token_delta" &&
      streamEvent.event_type !== "final" &&
      streamEvent.event_type !== "error"
    ) {
      updateStreamingAssistantMessage(activeRequest, {
        activity: conversationActivityForRequestMode(
          workflowActivity,
          activeRequest.requestMode
        ),
        phase: conversationPhaseForRequestActivity(
          workflowActivity,
          activeRequest.requestMode
        )
      });
    }

    if (
      streamEvent.event_type === "artifact_extracted" &&
      activeRequest.requestMode !== "explain"
    ) {
      const hydration = annotateRefinedHydration(
        hydrateWorkspaceFromArtifactExtractedEvent(currentSnapshot, streamEvent, {
          prompt: activeRequest.prompt
        }),
        activeRequest.pendingRefinement,
        currentSnapshot
      );

      if (hydration.artifact) {
        replaceSnapshot(
          creativePlanUpdate
            ? { ...hydration.snapshot, creativePlan: creativePlanUpdate }
            : hydration.snapshot
        );
        setActiveArtifactId(hydration.activeArtifactId);
        setPreviewArtifactId(hydration.previewArtifactId);
        setPreviewSessionOverride(null);
        if (
          hydration.previewAvailable &&
          activeRequest.forcePreviewOpen
        ) {
          handlePreviewOpenChange(true);
          setActiveTab("Preview");
        }
      }
    }

    if (
      streamEvent.event_type === "preview_artifact" &&
      activeRequest.requestMode !== "explain"
    ) {
      const previewUpdate = readPreviewArtifactUpdate(streamEvent);
      const nextPreviewArtifactId =
        previewUpdate?.previewArtifactId ?? previewUpdate?.artifactId ?? null;
      const emittedPreviewArtifact = nextPreviewArtifactId
        ? currentSnapshot.artifacts.find((artifact) => artifact.id === nextPreviewArtifactId) ??
          null
        : null;
      const previewArtifact = nextPreviewArtifactId
        ? currentSnapshot.artifacts.find(
            (artifact) => artifact.refinedFromArtifactId === nextPreviewArtifactId
          ) ?? emittedPreviewArtifact
        : null;
      const previewEventIsCodeOnly =
        previewUpdate?.artifactDomain === "react_three_fiber" ||
        previewUpdate?.artifactPreviewEligible === false;
      const previewCanOpen =
        previewUpdate?.status === "succeeded" &&
        !previewEventIsCodeOnly &&
        (!previewArtifact || isArtifactPreviewable(previewArtifact));

      if (!previewCanOpen) {
        streamRuntime.hasPreviewRuntimeEvent = false;
        const fallbackArtifact = previewArtifact
          ? resolveRefinedPreviewFallbackArtifact({
              artifacts: currentSnapshot.artifacts,
              failedArtifactIds: [previewArtifact.id, nextPreviewArtifactId],
              failedArtifactTitle: previewArtifact.title
            })
          : null;

        if (fallbackArtifact) {
          setPreviewArtifactId(fallbackArtifact.id);
          setPreviewSessionOverride(
            createPreviewSessionOverride(fallbackArtifact.id, "reloading")
          );
          setPreviewRuntimeLive(null);
          return;
        }
        if (previewArtifact && !isArtifactPreviewable(previewArtifact)) {
          setPreviewArtifactId("");
        }
        handlePreviewOpenChange(false);
        return;
      }

      streamRuntime.hasPreviewRuntimeEvent = true;

      if (previewUpdate) {
        setPreviewSessionOverride((currentOverride) => {
          if (!currentOverride) {
            return null;
          }

          return currentOverride.artifactId === previewUpdate.artifactId
            ? null
            : currentOverride;
        });
      }

      if (
        previewArtifact &&
        currentSnapshot.artifacts.some((artifact) => artifact.id === previewArtifact.id)
      ) {
        setPreviewArtifactId(previewArtifact.id);
      }
      if (
        workspacePreferences.autoOpenPreview ||
        activeRequest.forcePreviewOpen
      ) {
        handlePreviewOpenChange(true);
        setActiveTab("Preview");
      }
    }

    if (streamEvent.event_type === "final") {
      const hydration = annotateRefinedHydration(
        hydrateWorkspaceFromFinalEvent(currentSnapshot, streamEvent, {
          prompt: activeRequest.prompt,
          skipArtifacts: activeRequest.requestMode === "explain",
          skipPlainTextArtifact:
            activeRequest.requestMode === "explain" ||
            streamRuntime.hasPreviewRuntimeEvent
        }),
        activeRequest.pendingRefinement,
        currentSnapshot
      );

      if (!hydration.artifact) {
        replaceSnapshot(
          creativePlanUpdate
            ? { ...hydration.snapshot, creativePlan: creativePlanUpdate }
            : hydration.snapshot
        );
        if (
          hydration.previewAvailable &&
          activeRequest.forcePreviewOpen
        ) {
          handlePreviewOpenChange(true);
          setActiveTab("Preview");
        }
        return;
      }

      replaceSnapshot(
        creativePlanUpdate
          ? { ...hydration.snapshot, creativePlan: creativePlanUpdate }
          : hydration.snapshot
      );
      setActiveArtifactId(hydration.activeArtifactId);
      setPreviewArtifactId(hydration.previewArtifactId);
      setPreviewSessionOverride(null);
      handlePreviewOpenChange(
        hydration.previewAvailable &&
          (workspacePreferences.autoOpenPreview ||
            activeRequest.forcePreviewOpen)
      );
      if (
        hydration.previewAvailable &&
        (workspacePreferences.autoOpenPreview ||
          activeRequest.forcePreviewOpen)
      ) {
        setActiveTab("Preview");
      }
    }
  }

  function updateStreamingAssistantMessage(
    activeRequest: ActiveAssistantRequest,
    nextState: Partial<
      Pick<ConversationEntry, "activity" | "content" | "pending" | "phase">
    >
  ) {
    if (
      !isActiveAssistantRequest(activeRequest) ||
      streamingAssistantIdRef.current !== activeRequest.assistantMessageId
    ) {
      return;
    }

    setConversationEntries((currentMessages) => {
      if (!isAssistantRequestEpochCurrent(activeRequest)) {
        return currentMessages;
      }
      const nextMessages = [...currentMessages];
      const assistantIndex = nextMessages.findIndex(
        (message) => message.id === activeRequest.assistantMessageId
      );

      if (assistantIndex < 0) {
        return currentMessages;
      }

      nextMessages[assistantIndex] = {
        ...nextMessages[assistantIndex],
        ...nextState
      };
      return nextMessages;
    });
  }

  function replaceSnapshot(nextSnapshot: AssistantWorkspaceSnapshot) {
    snapshotRef.current = nextSnapshot;
    setSnapshot(nextSnapshot);
  }

  function finalizeStreamingAssistantMessage(
    activeRequest: ActiveAssistantRequest,
    {
      activity,
      content,
      phase
    }: {
      activity: string;
      content: string;
      phase: Extract<
        ConversationEntryPhase,
        "complete" | "completed" | "partial" | "failed" | "error" | "fallback"
      >;
    }
  ) {
    if (!isActiveAssistantRequest(activeRequest)) {
      return;
    }
    updateStreamingAssistantMessage(activeRequest, {
      activity,
      content,
      pending: false,
      phase
    });
    if (streamingAssistantIdRef.current === activeRequest.assistantMessageId) {
      streamingAssistantIdRef.current = null;
    }
  }

  async function handleArtifactCopy(artifact: ArtifactSummary) {
    setActiveArtifactId(artifact.id);
    if (isArtifactPreviewable(artifact)) {
      setPreviewContextArtifactId(artifact.id);
    }
    const wasCopied = await copyArtifactDocument(
      buildArtifactDocument(interactiveSnapshot, artifact)
    );
    setFeedbackState(
      artifact.id,
      "Copy",
      wasCopied ? "success" : "error",
      copyFeedbackTimerRef,
      setCopyFeedback
    );
  }

  function handleArtifactTransfer(
    action: ArtifactTransferAction,
    artifact: ArtifactSummary
  ) {
    const exportsProjectBundle = isProjectBundleExportAction(action, artifact);
    requestOperatorApproval({
      actionId: getArtifactTransferApprovalActionId(action, artifact),
      artifactTitle: exportsProjectBundle
        ? interactiveSnapshot.workspace.name
        : artifact.title,
      execute: () => {
        setActiveArtifactId(artifact.id);
        if (isArtifactPreviewable(artifact)) {
          setPreviewContextArtifactId(artifact.id);
        }
        setArtifactTransferError(null);
        const wasTransferred = exportsProjectBundle
          ? (() => {
              const bundle = buildProjectBundle({
                approvalSummary,
                domainContracts: domainExperience.domains,
                persistenceRecord,
                previewController,
                previewRoute: previewRendererRoute,
                previewRuntimeSource,
                retrievalRuntime,
                snapshot: interactiveSnapshot,
                workflowRuntime
              });

              return downloadZipArchive(
                bundle.fileName,
                buildZipArchive(
                  bundle.files.map((file) => ({
                    bytes: file.bytes,
                    path: file.path
                  }))
                )
              );
            })()
          : downloadArtifactDocument(
              buildArtifactDocument(interactiveSnapshot, artifact)
            );
        setFeedbackState(
          artifact.id,
          action,
          wasTransferred ? "success" : "error",
          transferFeedbackTimerRef,
          setTransferFeedback
        );
        if (!wasTransferred) {
          const transferError = createArtifactTransferError(action, artifact);
          setArtifactTransferError(transferError);
          throw new Error(transferError.userMessage);
        }
      }
    });
  }

  function handleArtifactAction(action: ArtifactAction, artifact: ArtifactSummary) {
    setActiveArtifactId(artifact.id);
    if (isArtifactPreviewable(artifact)) {
      setPreviewContextArtifactId(artifact.id);
    }

    if (action === "Open") {
      setActiveTab("Code");
      return;
    }

    if (action === "Preview") {
      setPreviewContextArtifactId(artifact.id);
      handlePreviewOpenChange(true);
      setActiveTab("Preview");
      return;
    }

    if (action === "Copy") {
      void handleArtifactCopy(artifact);
      return;
    }

    if (action === "Download" || action === "Export") {
      handleArtifactTransfer(action, artifact);
      return;
    }

    setActiveTab("Artifacts");
  }

  function handleArtifactRename(artifact: ArtifactSummary, requestedTitle: string) {
    const renamed = renameWorkspaceArtifact({
      artifactId: artifact.id,
      requestedTitle,
      snapshot
    });
    if (!renamed) {
      return null;
    }

    replaceSnapshot(renamed.snapshot);
    setActiveArtifactId(artifact.id);
    if (isArtifactPreviewable(artifact)) {
      setPreviewArtifactId(artifact.id);
    }
    return renamed.title;
  }

  function handleArtifactDelete(artifact: ArtifactSummary) {
    requestApplicationConfirmation({
      cancelLabel: "Keep artifact",
      confirmLabel: "Delete artifact",
      detail:
        "This removes the artifact from the current session. You can still undo the removal while this session remains open.",
      eyebrow: "Artifact deletion",
      id: `delete-artifact-${artifact.id}`,
      onConfirm: () => {
        const result = removeWorkspaceArtifact({
          activeArtifactId,
          artifactId: artifact.id,
          previewArtifactId,
          snapshot
        });
        if (!result) {
          return;
        }
        replaceSnapshot(result.snapshot);
        setActiveArtifactId(result.activeArtifactId);
        setPreviewArtifactId(result.previewArtifactId);
        setLastRemovedArtifact(result.removed);
        setLastRestoredArtifact(null);
      },
      title: `Delete ${artifact.title}?`,
      tone: "danger"
    });
  }

  function handleArtifactUndo() {
    if (!lastRemovedArtifact) {
      return;
    }
    const restored = restoreWorkspaceArtifact({
      removed: lastRemovedArtifact,
      snapshot
    });
    replaceSnapshot(restored);
    setActiveArtifactId(lastRemovedArtifact.artifact.id);
    if (isArtifactPreviewable(lastRemovedArtifact.artifact)) {
      setPreviewArtifactId(lastRemovedArtifact.artifact.id);
    }
    setLastRestoredArtifact(lastRemovedArtifact);
    setLastRemovedArtifact(null);
  }

  function handleArtifactRedo() {
    if (!lastRestoredArtifact) {
      return;
    }
    const result = removeWorkspaceArtifact({
      activeArtifactId,
      artifactId: lastRestoredArtifact.artifact.id,
      previewArtifactId,
      snapshot
    });
    if (!result) {
      return;
    }
    replaceSnapshot(result.snapshot);
    setActiveArtifactId(result.activeArtifactId);
    setPreviewArtifactId(result.previewArtifactId);
    setLastRemovedArtifact(result.removed);
    setLastRestoredArtifact(null);
  }

  function handleOutputFeedback(
    sentiment: FeedbackSentiment,
    comment: string | null
  ) {
    feedbackIdCounterRef.current += 1;
    const signal = createFeedbackSignal({
      artifact: activeArtifact.id === emptyWorkspaceArtifact.id ? null : activeArtifact,
      comment,
      creativity: workspacePreferences.creativity,
      effectiveTemperature: providerTelemetry.configuration.temperature,
      id: `feedback-${Date.now()}-${feedbackIdCounterRef.current}`,
      parameterApplication:
        providerTelemetry.configuration.parameterSource === "provider_reported"
          ? "provider_reported"
          : "requested_not_confirmed",
      productOutcome:
        snapshot.workflow.productOutcome?.product_outcome ?? null,
      promptExcerpt: [...conversationEntries]
        .reverse()
        .find((entry) => entry.role === "user")?.content ?? null,
      providerModel: providerTelemetry.provider.model,
      providerName: providerTelemetry.provider.name,
      requestedTemperature: buildGenerationControls(
        workspacePreferences.creativity
      ).requestedTemperature,
      sentiment,
      sessionId: workspaceIdentity.sessionId,
      workflowMode
    });
    updateWorkspacePreferences({
      feedbackSignals: [...workspacePreferences.feedbackSignals, signal].slice(-120)
    });
  }

  async function handleRunEvaluation(
    request: EvaluationRunRequest,
    onProgress: EvaluationProgressCallback
  ) {
    const evaluationController = new AbortController();
    evaluationAbortControllerRef.current?.abort();
    evaluationAbortControllerRef.current = evaluationController;
    const latestVisibleProgress: { current: EvaluationExecutionProgress | null } = {
      current: null
    };
    const releaseEvaluationController = () => {
      if (evaluationAbortControllerRef.current === evaluationController) {
        evaluationAbortControllerRef.current = null;
      }
    };
    const publishProgress = (progress: EvaluationExecutionProgress) => {
      if (!evaluationController.signal.aborted && isShellMountedRef.current) {
        latestVisibleProgress.current = normalizeVisibleEvaluationProgress(progress, request.scope);
        onProgress(latestVisibleProgress.current);
      }
    };
    let payload: Record<string, unknown> = {};
    let ragas = emptyRagasEvidence();
    const dataset = buildGoldenEvaluationDataset();
    const canonicalRetrievalCaseIds = new Set<string>(CURRENT_PRODUCT_RETRIEVAL_CASE_IDS);
    const remoteCaseIds = request.scope === "cases"
      ? [...new Set(request.caseIds.flatMap((caseId) => {
          const canonicalCaseId = caseId.startsWith("retrieval/")
            ? caseId.slice("retrieval/".length)
            : caseId;
          return canonicalRetrievalCaseIds.has(canonicalCaseId)
            ? [canonicalCaseId]
            : [];
        }))]
      : [];
    const selectedHasRag = request.scope === "full" || request.scope === "rag" || remoteCaseIds.length > 0;
    const providerCallsUsed = selectedHasRag && request.allowProviderCalls;
    const executionRequest: EvaluationRunRequest = {
      ...request,
      allowProviderCalls: providerCallsUsed
    };
    const selectedContracts = request.scope === "cases"
      ? request.caseIds.length
      : request.scope === "full"
        ? CURRENT_PRODUCT_RETRIEVAL_CASE_IDS.length
        : request.scope === "rag"
          ? CURRENT_PRODUCT_RETRIEVAL_CASE_IDS.length
          : dataset.cases.filter((item) => item.categories.includes(request.scope as EvaluationCategory)).length;
    let terminalStatus = "failed";
    let localOnly = false;

    publishProgress({
      runId: null,
      status: "preflight",
      phase: "preflight",
      lane: request.scope === "full" ? "Full evaluation" : request.scope.replace(/_/g, " "),
      currentCaseId: null,
      currentCaseLabel: selectedHasRag ? "Preparing current-product benchmark" : "Preparing local workspace-lane snapshot",
      completedCases: 0,
      totalCases: selectedContracts,
      remainingCases: selectedContracts,
      percent: 0,
      executionState: providerCallsUsed ? "provider_authorized" : selectedHasRag ? "local_preflight" : "local_workspace",
      detail: providerCallsUsed
        ? "The selected current-product contract is being submitted for retrieval, generation, and evaluation."
        : selectedHasRag
          ? "The canonical retrieval selection is limited to local preflight and cannot publish a new Retrieval Quality score."
          : "No canonical retrieval cases are selected; only current local workspace-lane evidence will be inspected."
    });

    try {
      if (!selectedHasRag) {
        localOnly = true;
        terminalStatus = "completed";
        const evaluatedAt = new Date().toISOString();
        const runId = `evaluation-local-${Date.now()}`;
        payload = { runId, status: terminalStatus };
        ragas = localWorkspaceLaneEvidence(runId, evaluatedAt);
      } else {
      const response = await fetch(evaluationRunEndpoint, {
        body: JSON.stringify({
          benchmarkMode: "current_product",
          scope: request.scope,
          caseIds: request.scope === "cases" ? remoteCaseIds : [],
          allowProviderCalls: providerCallsUsed,
          approvedDataset: request.approvedRagasDataset,
          dryRun: !providerCallsUsed
        }),
        headers: { "Content-Type": "application/json" },
        method: "POST",
        signal: evaluationController.signal
      });
      const responsePayload: unknown = await response.json();
      payload = isUnknownRecord(responsePayload) ? responsePayload : {};
      if (!response.ok) {
        terminalStatus = payload.error === "blocked_by_execution_environment" ? "blocked" : "failed";
        publishProgress({
          runId: typeof payload.runId === "string" ? payload.runId : null,
          status: terminalStatus,
          phase: "terminal",
          lane: request.scope === "full" ? "Full evaluation" : request.scope.replace(/_/g, " "),
          currentCaseId: null,
          currentCaseLabel: "Evaluation stopped",
          completedCases: 0,
          totalCases: selectedContracts,
          remainingCases: selectedContracts,
          percent: null,
          executionState: request.allowProviderCalls ? "provider_blocked" : "local_blocked",
          detail: typeof payload.message === "string" ? payload.message : "The evaluation service rejected the run."
        });
      } else {
        let snapshot = parseEvaluationApiSnapshot(payload, {
          request: executionRequest,
          selectedContracts
        });
        publishProgress(snapshot.progress);
        let consecutiveRefreshFailures = 0;
        let nextPollDelayMs = 0;

        while (!isTerminalEvaluationStatus(snapshot.status)) {
          if (!snapshot.runId) {
            throw new Error("Evaluation service did not publish a run identifier.");
          }
          if (nextPollDelayMs > 0) {
            await waitForEvaluationPoll(nextPollDelayMs, evaluationController.signal);
          }
          if (evaluationController.signal.aborted || !isShellMountedRef.current) {
            releaseEvaluationController();
            return;
          }
          try {
            const pollResponse = await fetchEvaluationPoll(
              snapshot.runId,
              evaluationController.signal
            );
            if (pollResponse.status === 404 || pollResponse.status === 410) {
              throw new EvaluationPollingTerminalError(
                `Evaluation run ${snapshot.runId} is no longer available (HTTP ${pollResponse.status}).`
              );
            }
            if (!pollResponse.ok) {
              throw new Error(
                `Evaluation status endpoint returned HTTP ${pollResponse.status}.`
              );
            }
            const pollPayload: unknown = await pollResponse.json();
            if (!isUnknownRecord(pollPayload)) {
              throw new Error("Evaluation status endpoint returned an invalid snapshot.");
            }
            payload = { ...payload, ...pollPayload };
            snapshot = parseEvaluationApiSnapshot(pollPayload, {
              request: executionRequest,
              runId: snapshot.runId,
              selectedContracts
            });
            consecutiveRefreshFailures = 0;
            nextPollDelayMs = evaluationPollIntervalMs;
            publishProgress(snapshot.progress);
          } catch (error) {
            if (evaluationController.signal.aborted || !isShellMountedRef.current) {
              releaseEvaluationController();
              return;
            }
            if (error instanceof EvaluationPollingTerminalError) {
              throw error;
            }
            consecutiveRefreshFailures += 1;
            if (consecutiveRefreshFailures >= evaluationMaxConsecutiveRefreshFailures) {
              throw new EvaluationPollingTerminalError(
                `Evaluation status remained unavailable after ${evaluationMaxConsecutiveRefreshFailures} consecutive refresh attempts.`
              );
            }
            nextPollDelayMs = evaluationReconnectDelayMs(consecutiveRefreshFailures);
            publishProgress({
              ...snapshot.progress,
              status: "running",
              phase: "reconnecting",
              currentCaseLabel: "Reconnecting to the evaluation service",
              executionState: "reconnecting",
              detail: `The status refresh was interrupted. Retrying automatically ${formatEvaluationReconnectDelay(nextPollDelayMs)}; the server run remains active.`
            });
          }
        }

        terminalStatus = normalizeEvaluationStatus(snapshot.status);
        payload = {
          ...(snapshot.result ?? {}),
          runId: snapshot.runId,
          status: terminalStatus,
          progress: snapshot.progress
        };
      }

      if (terminalStatus === "completed") {
        ragas = selectedHasRag
          ? parseRagasExecutionEvidence(payload, executionRequest, terminalStatus)
          : unscoredCurrentProductEvidence(payload, executionRequest, terminalStatus);
      } else if (terminalStatus === "prepared") {
        ragas = selectedHasRag
          ? parseRagasExecutionEvidence(payload, executionRequest, terminalStatus)
          : unscoredCurrentProductEvidence(payload, executionRequest, terminalStatus);
      } else if (terminalStatus === "blocked") {
        ragas = blockedRagasEvidence(executionRequest, payload);
      } else {
        ragas = failedRagasEvidence(executionRequest, payload);
      }
      }
    } catch (error) {
      if (evaluationController.signal.aborted || !isShellMountedRef.current) {
        releaseEvaluationController();
        return;
      }
      const pollingFailure = error instanceof EvaluationPollingTerminalError;
      terminalStatus = pollingFailure ? "failed" : "blocked";
      const message = error instanceof Error ? error.message : "the local evaluation service is unavailable";
      payload = {
        ...payload,
        message: pollingFailure
          ? `${message} Use Run Evaluation to retry.`
          : `BLOCKED_BY_EXECUTION_ENVIRONMENT: ${message}`,
        status: terminalStatus
      };
      ragas = pollingFailure
        ? failedRagasEvidence(executionRequest, payload)
        : blockedRagasEvidence(executionRequest, payload);
      publishProgress({
        runId: typeof payload.runId === "string" ? payload.runId : null,
        status: terminalStatus,
        phase: "terminal",
        lane: latestVisibleProgress.current?.lane ?? (request.scope === "full" ? "Full evaluation" : request.scope.replace(/_/g, " ")),
        currentCaseId: latestVisibleProgress.current?.currentCaseId ?? null,
        currentCaseLabel: "Evaluation stopped",
        completedCases: latestVisibleProgress.current?.completedCases ?? 0,
        totalCases: latestVisibleProgress.current?.totalCases ?? selectedContracts,
        remainingCases: latestVisibleProgress.current?.remainingCases ?? selectedContracts,
        percent: latestVisibleProgress.current?.percent ?? null,
        executionState: pollingFailure
          ? "polling_failed"
          : providerCallsUsed ? "provider_blocked" : "local_blocked",
        detail: String(payload.message)
      });
    }

    if (evaluationController.signal.aborted || !isShellMountedRef.current) {
      releaseEvaluationController();
      return;
    }

    const previousRun = [...workspacePreferences.evaluationHistory]
      .reverse()
      .find((entry) =>
        entry.benchmark?.scoreOrigin === "current_product" &&
        entry.benchmark.ragas.state === "completed"
      )?.benchmark ?? null;
    const benchmark = buildEvaluationBenchmarkRun({
      model: productIntelligence,
      previousRun,
      ragas,
      request: executionRequest
    });
    if (localOnly) {
      const completedCases = benchmark.executedCases;
      publishProgress({
        runId: benchmark.runId,
        status: "completed",
        phase: "local_snapshot_completed",
        lane: request.scope === "cases" ? "Selected local-only cases" : request.scope.replace(/_/g, " "),
        currentCaseId: null,
        currentCaseLabel: "Local workspace-lane snapshot complete",
        completedCases,
        totalCases: selectedContracts,
        remainingCases: Math.max(0, selectedContracts - completedCases),
        percent: 100,
        executionState: "local_workspace",
        detail: `${completedCases}/${selectedContracts} selected contracts had current observable local evidence. No retrieval, generation, or evaluator provider calls were made; no Retrieval Quality score was published.`
      });
    }
    const evaluationRecord: EvaluationHistoryRecord = {
      id: benchmark.id,
      runId: typeof payload.runId === "string" ? payload.runId : benchmark.id,
      datasetId: ragas.datasetId,
      metrics: ragas.metrics,
      status: benchmark.statusLabel,
      detail: ragas.detail,
      evaluatedAt: benchmark.completedAt,
      resultRows: ragas.resultRows,
      metricFailures: ragas.metricFailures,
      dryRun: !providerCallsUsed,
      providerCallsAllowed: providerCallsUsed,
      benchmark
    };
    setWorkspacePreferences((current) => normalizeWorkspacePreferences({
      ...current,
      evaluationHistory: [...current.evaluationHistory, evaluationRecord].slice(-24)
    }));
    appendLocalRuntimeEvent({
      event_type: "eval_update",
      sequence: localRuntimeSequenceRef.current++,
      payload: {
        code: ragas.state === "blocked"
          ? "evaluation_blocked"
          : ragas.state === "failed" ? "evaluation_failed" : "evaluation_run_completed",
        message: ragas.state === "blocked"
          ? "Provider evidence was blocked by the execution environment; local evidence was retained."
          : ragas.state === "failed"
            ? "Evaluator evidence was unavailable after an unexpected error; local evidence was retained."
            : "Evaluation run completed.",
        evaluation: payload,
        status: ragas.state === "blocked"
          ? "BLOCKED_BY_EXECUTION_ENVIRONMENT"
          : ragas.state === "failed" ? "MISSING_EVIDENCE" : "completed"
      }
    });
    releaseEvaluationController();
  }

  function handleArtifactSelect(artifact: ArtifactSummary) {
    setActiveArtifactId(artifact.id);
    setPreviewContextArtifactId(artifact.id);
    setActiveTab("Artifacts");
  }

  return (
    <main
      className="workstation"
      data-active-tab={activeTab.toLowerCase()}
      data-dashboard={isDashboardOpen ? "open" : "closed"}
      data-density={layoutState.density}
      data-focus-mode={isFocusMode ? "true" : "false"}
      data-inspector-state={isInspectorCollapsed ? "collapsed" : "open"}
      data-preview={isPreviewOpen ? "open" : "closed"}
      data-readiness={workstationState.readiness.state}
      data-resizing={activeResizeTarget ?? "idle"}
      data-sidebar-state={layoutState.sidebarCollapsed ? "collapsed" : "open"}
      data-stream-state={streamState}
      data-theme={workspacePreferences.theme}
      style={workspaceLayoutStyle}
    >
      <header className="topbar">
        <div className="brand">
          <div className="brandMark" aria-hidden="true">
            <Sparkles size={20} strokeWidth={1.8} />
          </div>
          <div className="brandText">
            <strong>Creative Coding Assistant</strong>
            <span>{snapshot.workspace.name}</span>
          </div>
        </div>

        <div
          className="sessionStatus"
          aria-label="Current session"
          data-activity={
            activeRequestModeRef.current === "explain" && activeWorkflowActivity
              ? "answering"
              : activeWorkflowActivity?.state
          }
          data-state={streamState}
        >
          <span>{displayedSessionStatusLabel}</span>
          {!isPristineSession ? <strong>{displayedSessionStatusDetail}</strong> : null}
          <small>
            {formatSessionTelemetryLabel(providerTelemetry, currentSessionUsage)}
          </small>
        </div>

        <div
          ref={utilityTrayRef}
        className="topbarActions"
        aria-label="Workspace actions"
      >
          <button
            aria-controls="demo-mode-panel"
            aria-expanded={isDemoModeOpen}
            aria-label="Demo Mode"
            className="toolbarToggle"
            onClick={handleDemoModeToggle}
            title={isDemoModeOpen ? "Close Demo Mode" : "Open Demo Mode"}
            type="button"
          >
            <Play size={16} />
            <span>Demo Mode</span>
          </button>
          <button
            aria-label="Open Product Intelligence Dashboard"
            aria-pressed={isDashboardOpen}
            className="iconButton"
            onClick={() => openDashboard()}
            title="Open Product Intelligence Dashboard"
            type="button"
          >
            <LayoutDashboard size={18} />
          </button>
          <div className="utilityControl">
            <button
              aria-controls="theme-presets-panel"
              aria-expanded={openUtilityPanel === "theme"}
              aria-haspopup="dialog"
              aria-label="Theme"
              className="iconButton"
              onClick={() => toggleUtilityPanel("theme")}
              ref={themeTriggerRef}
              title="Open theme presets"
              type="button"
            >
              <Paintbrush size={17} />
            </button>
            {openUtilityPanel === "theme" ? (
              <ThemePresetsPanel
                activeTheme={workspacePreferences.theme}
                onRequestClose={() => closeUtilityPanel("theme")}
                onSelectTheme={(theme) => {
                  updateWorkspacePreferences({ theme });
                  closeUtilityPanel("theme");
                }}
              />
            ) : null}
          </div>
          <div className="utilityControl">
            <button
              aria-controls="workspace-settings-panel"
              aria-expanded={openUtilityPanel === "settings"}
              aria-haspopup="dialog"
              aria-label="Settings"
              className="iconButton"
              onClick={() => toggleUtilityPanel("settings")}
              ref={settingsTriggerRef}
              title="Open workspace settings"
              type="button"
            >
              <Settings size={18} />
            </button>
            {openUtilityPanel === "settings" ? (
              <WorkspaceSettingsPanel
                activeTab={activeTab}
                hasBlockingApproval={Boolean(blockingApprovalRequest)}
                isFocusMode={isFocusMode}
                isPreviewAvailable={interactiveSnapshot.preview.available}
                isPreviewOpen={isPreviewOpen}
                layoutState={layoutState}
                onDensityChange={(density) => updateLayout({ density })}
                onFocusModeToggle={toggleFocusModeFromUtility}
                onOpenDashboardSettings={() => openDashboard("Settings")}
                onOpenTab={openInspectorTabFromUtility}
                onPreferencesChange={updateWorkspacePreferences}
                onPreviewToggle={togglePreviewFromUtility}
                onRequestClose={() => closeUtilityPanel("settings")}
                onWorkspaceClear={() =>
                  requestOperatorApproval({
                    actionId: "workspace_clear",
                    execute: clearWorkspaceSession
                  })
                }
                preferences={workspacePreferences}
                showDebugPanels={workspacePreferences.showDebugPanels}
              />
            ) : null}
          </div>
        </div>
      </header>

      {isDashboardOpen ? (
        <ProductIntelligenceDashboard
          activeCategory={dashboardCategory}
          model={productIntelligence}
          onCategoryChange={setDashboardCategory}
          onClose={() => setIsDashboardOpen(false)}
          onRunEvaluation={handleRunEvaluation}
          evaluationHistory={workspacePreferences.evaluationHistory}
          feedback={
            hasWorkspaceArtifacts && !isStreaming
              ? {
                  artifactTitle: activeArtifact.title,
                  onSubmit: handleOutputFeedback
                }
              : undefined
          }
          sessions={{
            activeSessionId: workspaceIdentity.sessionId,
            onCreate: () => void handleCreateSession(),
            onDelete: (sessionId) => void handleSessionDelete(sessionId),
            onRename: (sessionId, title) => void handleSessionRename(sessionId, title),
            onSelect: activateSession,
            sessions: visibleWorkspaceSessions,
            usage: sessionUsageSummaries
          }}
          settings={{
            isFocusMode,
            isPreviewOpen,
            layoutState,
            onDensityChange: (density) => updateLayout({ density }),
            onFocusModeToggle: handleFocusModeToggle,
            onInspectorToggle: () =>
              handleInspectorCollapsedChange(!layoutState.inspectorCollapsed),
            onPreferencesChange: updateWorkspacePreferences,
            onPreviewToggle: handlePreviewShelfFromControl,
            onSidebarToggle: () =>
              updateLayout({ sidebarCollapsed: !layoutState.sidebarCollapsed }),
            onWorkflowModeChange: (workflowMode) =>
              updateWorkspacePreferences({ workflowMode }),
            preferences: workspacePreferences,
            workflowMode
          }}
        />
      ) : (
      <section className="workspaceLayout" aria-label="Creative workspace">
        <SessionSidebar
          activeSessionId={workspaceIdentity.sessionId}
          collapsed={layoutState.sidebarCollapsed}
          onCreate={() => void handleCreateSession()}
          onDelete={(sessionId) => void handleSessionDelete(sessionId)}
          onRename={handleSessionRename}
          onSelect={activateSession}
          onToggle={() =>
            updateLayout({ sidebarCollapsed: !layoutState.sidebarCollapsed })
          }
          sessions={visibleWorkspaceSessions}
        />
        <div className="mainColumn">
          <section
            aria-label="Creative session"
            className="sessionPanel"
            data-checkpoint={visibleApprovalRequest ? "true" : undefined}
            data-demo={isDemoModeOpen ? "true" : undefined}
            data-homepage={isPristineSession ? "true" : undefined}
          >
            <div className="sessionIntro" hidden={isPristineSession}>
              <header className="sessionHeader" hidden={isDemoModeOpen}>
                <div>
                  <span className="eyebrow">Creative session</span>
                  <h1>{snapshot.workspace.focus}</h1>
                  <p>
                    Generate, refine, preview, and save browser-native creative
                    coding artifacts in one workspace.
                  </p>
                </div>
                <div
                  className="sessionMetric"
                  aria-label="Active artifact"
                  data-active={hasWorkspaceArtifacts ? "true" : "false"}
                >
                  <span>{hasWorkspaceArtifacts ? "Active artifact" : "Workspace"}</span>
                  <strong>
                    {hasWorkspaceArtifacts
                      ? activeArtifactDisplayLabel
                      : "Ready for first prompt"}
                  </strong>
                  <small aria-live="polite">{persistenceStatusLabel}</small>
                </div>
              </header>
              {isDemoModeOpen ? (
                <DemoModePanel
                  activeScenario={activeDemoScenario}
                  hasImageAttachment={imageAttachments.length > 0}
                  scenarios={demoModeScenarios}
                  onLoadScenario={handleDemoScenarioLoad}
                  onSelectScenario={handleDemoScenarioSelect}
                  showDebugPanels={workspacePreferences.showDebugPanels}
                />
              ) : null}
            </div>

            <WorkspaceConversation
              leadingSlot={visibleApprovalRequest ? (
                <article
                aria-label="Operator checkpoint"
                aria-live="polite"
                className="operatorCheckpoint"
                data-kind={visibleApprovalRequest.kind}
                data-state={visibleApprovalRequest.state}
                ref={approvalCardRef}
                tabIndex={-1}
              >
                <header className="operatorCheckpointHeader">
                  <div>
                    <span className="eyebrow">Operator checkpoint</span>
                    <strong>{visibleApprovalRequest.title}</strong>
                    <p>{visibleApprovalRequest.summary}</p>
                  </div>
                  <div className="operatorCheckpointStatus">
                    <small>{getHitlApprovalStateLabel(visibleApprovalRequest.state)}</small>
                    {isHitlApprovalTerminalState(visibleApprovalRequest.state) ? (
                      <button
                        aria-label="Dismiss operator checkpoint"
                        className="iconButton"
                        onClick={() => handleApprovalDismiss(visibleApprovalRequest)}
                        type="button"
                      >
                        <X size={15} />
                      </button>
                    ) : null}
                  </div>
                </header>
                <p>{visibleApprovalRequest.detail}</p>
                <dl className="operatorCheckpointMeta">
                  <div>
                    <dt>Target</dt>
                    <dd>{visibleApprovalRequest.targetLabel}</dd>
                  </div>
                  <div>
                    <dt>Workflow</dt>
                    <dd>{interactiveSnapshot.workflow.currentStep}</dd>
                  </div>
                  <div>
                    <dt>Requested</dt>
                    <dd>{formatTraceTime(visibleApprovalRequest.requestedAt)}</dd>
                  </div>
                </dl>
                {visibleApprovalRequest.failureReason ? (
                  <p className="operatorCheckpointFailure" role="status">
                    {visibleApprovalRequest.failureReason}
                  </p>
                ) : null}
                {visibleApprovalRequest.state === "pending_approval" ? (
                  <div className="operatorCheckpointActions">
                    <button
                      onClick={() => handleApprovalReject(visibleApprovalRequest)}
                      type="button"
                    >
                      {visibleApprovalRequest.cancelLabel}
                    </button>
                    <button
                      className="operatorCheckpointApprove"
                      onClick={() => void handleApprovalApprove(visibleApprovalRequest)}
                      type="button"
                    >
                      {visibleApprovalRequest.confirmLabel}
                    </button>
                  </div>
                ) : (
                  <div className="operatorCheckpointFooter">
                    <span>{getHitlApprovalStateLabel(visibleApprovalRequest.state)}</span>
                    <small>
                      {visibleApprovalRequest.resolvedAt
                        ? formatTraceTime(visibleApprovalRequest.resolvedAt)
                        : "Awaiting operator decision"}
                    </small>
                  </div>
                )}
                </article>
              ) : null}
              emptyState={
                <EmptyWorkspaceState onSelectPrompt={handleEmptyStatePromptSelect} />
              }
              entries={conversationEntries}
              getDisplayContent={(message) =>
                getConversationDisplayContent(
                  message,
                  workspacePreferences.showDebugPanels
                )
              }
              initialEntryCount={snapshot.messages.length}
              isPristine={isPristineSession}
              isStreaming={isStreaming}
              sessionKey={workspaceIdentity.sessionId}
            />
            {streamError ? (
              <SubsystemErrorCallout
                className="chatErrorCallout"
                error={streamError}
                title="Live stream interrupted"
              />
            ) : null}
            {lastRemovedArtifact ? (
              <div className="artifactUndoNotice" role="status">
                <span>{lastRemovedArtifact.artifact.title} was deleted.</span>
                <button onClick={handleArtifactUndo} type="button">
                  Undo
                </button>
              </div>
            ) : null}
            {lastRestoredArtifact ? (
              <div className="artifactUndoNotice" role="status">
                <span>{lastRestoredArtifact.artifact.title} was restored.</span>
                <button onClick={handleArtifactRedo} type="button">
                  Redo delete
                </button>
              </div>
            ) : null}

            {interactiveSnapshot.multimodal.imageAttachments.length > 0 ||
            interactiveSnapshot.multimodal.error ? (
              <WorkspaceImageReferences
                multimodal={interactiveSnapshot.multimodal}
                onDismissError={handleImageUploadErrorDismiss}
                onRemove={handleImageAttachmentRemove}
              />
            ) : null}

            <WorkspaceComposer
              attachmentSlot={
                <WorkspaceAttachmentControl
                  disabled={isStreaming}
                  isOpen={isAttachmentMenuOpen}
                  isProcessing={isImageUploadPending}
                  onFilesSelected={handleImageFilesSelected}
                  onOpenChange={setIsAttachmentMenuOpen}
                />
              }
              controlsSlot={
                <WorkspaceGenerationControls
                  creativity={workspacePreferences.creativity}
                  disabled={isStreaming}
                  onCreativityChange={(creativity) =>
                    updateWorkspacePreferences({ creativity })
                  }
                  onWorkflowChange={(workflowMode) =>
                    updateWorkspacePreferences({ workflowMode })
                  }
                  workflowMode={workflowMode}
                />
              }
              hasImages={
                interactiveSnapshot.multimodal.imageAttachments.length > 0
              }
              isReady={isComposerReady}
              isPreparingAttachments={isImageUploadPending}
              isStreaming={isStreaming}
              mode={workspacePreferences.showDebugPanels ? "developer" : "user"}
              onChange={setComposerValue}
              onSubmit={handleComposerSubmit}
              ref={composerTextareaRef}
              value={composerValue}
            />
          </section>

          {shouldRenderPreviewShelf ? (
            <PreviewWorkspace
              controller={previewController}
              height={layoutState.previewHeight}
              onClear={handlePreviewStateClear}
              onFullscreenToggle={handlePreviewFullscreenChange}
              onOpenArtifacts={() => revealInspectorTab("Artifacts")}
              onOpenCode={() => revealInspectorTab("Code")}
              onReload={handlePreviewStateReload}
              onRuntimeDiagnostics={handlePreviewRuntimeDiagnostics}
              onResizeKeyDown={handlePreviewResizeKeyDown}
              onResizeStart={handlePreviewResizeStart}
              onRestart={handlePreviewSessionRestart}
              onRuntimeFrame={appendPreviewRuntimeFrameEvent}
              onRuntimeStatus={appendPreviewRuntimeStatusEvent}
              onToggle={handlePreviewOpenChange}
              route={previewRendererRoute}
              runtimeSessionKey={previewRuntimeSessionKey}
              runtimeSource={previewRuntimeSource}
              resizing={activeResizeTarget === "preview"}
              showDebugPanels={workspacePreferences.showDebugPanels}
              snapshot={interactiveSnapshot}
              userArtifactLabel={formatUserPreviewArtifactLabel(interactiveSnapshot)}
            />
          ) : null}
        </div>

        {!isFocusMode ? (
          <>
            <div
              aria-label="Resize inspector"
              aria-orientation="vertical"
              aria-valuemax={workspaceLayoutBounds.maxInspectorWidth}
              aria-valuemin={workspaceLayoutBounds.minInspectorWidth}
              aria-valuenow={layoutState.inspectorWidth}
              className="layoutResizeHandle inspectorResizeHandle"
              data-active={activeResizeTarget === "inspector"}
              onKeyDown={handleInspectorResizeKeyDown}
              onMouseDown={handleInspectorResizeStart}
              role="separator"
              tabIndex={isInspectorCollapsed ? -1 : 0}
            >
              <span aria-hidden="true" />
            </div>

            <RightInspector
              activeTab={activeTab}
              addMenuOpen={isInspectorAddMenuOpen}
              availableTabs={availableInspectorCategories
                .filter((category) => !inspectorTabs.includes(category))
                .map((category) => ({
                  closeable: category !== "Overview",
                  icon: inspectorTabIcons[category],
                  id: category,
                  label: formatInspectorTabDisplayLabel(
                    category,
                    workspacePreferences.showDebugPanels
                  ),
                  panelId: getInspectorPanelId(category)
                }))}
              collapsed={isInspectorCollapsed}
              detail={activeTabSummary}
              onAddMenuOpenChange={setIsInspectorAddMenuOpen}
              onAddTab={revealInspectorTab}
              onCloseTab={handleInspectorTabClose}
              onOpenDashboard={openDashboard}
              onSelectTab={setActiveTab}
              onToggle={() => handleInspectorCollapsedChange(!isInspectorCollapsed)}
              tabs={visibleInspectorTabs.map((tab) => ({
                closeable: tab !== "Overview" && visibleInspectorTabs.length > 1,
                icon: inspectorTabIcons[tab],
                id: tab,
                label: formatInspectorTabDisplayLabel(
                  tab,
                  workspacePreferences.showDebugPanels
                ),
                panelId: getInspectorPanelId(tab)
              }))}
            >
              <InspectorPanel
                    activeArtifact={activeArtifact}
                    activeArtifactDocument={activeArtifactDocument}
                    activeArtifactHighlights={activeArtifactHighlights}
                    activeArtifactId={activeArtifactId}
                    activeTab={activeTab}
                    artifactTransferError={artifactTransferError}
                    copyFeedback={copyFeedback}
                    isStreaming={isStreaming}
                    onArtifactCopy={handleArtifactCopy}
                    onArtifactAction={handleArtifactAction}
                    onArtifactRefine={handleArtifactRefine}
                    onArtifactDelete={handleArtifactDelete}
                    onArtifactRename={handleArtifactRename}
                    onArtifactSelect={handleArtifactSelect}
                    onArtifactTransfer={handleArtifactTransfer}
                    onClarificationOptionSelect={handleClarificationOptionSelect}
                    productIntelligence={productIntelligence}
                    previewController={previewController}
                    runtimeConsole={runtimeConsole}
                    previewRoute={previewRendererRoute}
                    retrievalRuntime={retrievalRuntime}
                    showDebugPanels={workspacePreferences.showDebugPanels}
                    snapshot={interactiveSnapshot}
                    streamError={streamError}
                    sessionUsage={currentSessionUsage}
                    telemetryDashboard={telemetryDashboard}
                    providerTelemetry={providerTelemetry}
                    transferFeedback={transferFeedback}
                    workflowExecution={workflowExecution}
                    workflowMode={workflowMode}
                    workflowRuntime={workflowRuntime}
                    workflowIssues={workflowIssues}
                  />
            </RightInspector>
          </>
        ) : null}
      </section>
      )}
      {applicationConfirmation ? (
        <ApplicationConfirmDialog
          onClose={() => setApplicationConfirmation(null)}
          request={applicationConfirmation}
        />
      ) : null}
    </main>
  );
}

function EmptyWorkspaceState({
  onSelectPrompt
}: {
  onSelectPrompt: (prompt: string) => void;
}) {
  const valueHighlights = [
    {
      icon: Braces,
      title: "Build browser-native visuals",
      detail: "Generate bounded p5.js, Three.js, GLSL, or Tone.js artifacts."
    },
    {
      icon: Database,
      title: "Ground answers in official sources",
      detail: "Use retrieval context when source-backed guidance matters."
    },
    {
      icon: Play,
      title: "Preview, refine, and save artifacts",
      detail: "Move from prompt to Code, Preview, and Saved outputs."
    },
    {
      icon: Activity,
      title: "Support creative-coding workflows",
      detail: "Plan, generate, inspect, preview, export, and recover clearly."
    }
  ];
  const journey = [
    { title: "Write the brief", detail: "Describe the idea, medium, and constraints." },
    { title: "Choose the route", detail: "Auto selects a bounded execution path." },
    { title: "Generate", detail: "Follow the streamed run and its published state." },
    { title: "Inspect", detail: "Review Code, Preview, and Runtime independently." },
    { title: "Keep the result", detail: "Save, refine, copy, or export the useful output." }
  ];
  return (
    <article
      aria-label="Empty creative workspace"
      className="emptyWorkspace"
      role="group"
    >
      <DashboardPageHero
        badgeLabel="Creative session capabilities"
        badges={["Browser-native", "Source-grounded", "Session-scoped"]}
        className="emptyWorkspaceHero"
        detail="Start with an idea, medium, constraint, or reference. Move from one clear brief to inspectable code, visible runtime evidence, and saved output."
        eyebrow="New creative session"
        headingLevel="h1"
        icon={Sparkles}
        title="Describe the creative system you want to build."
      />

      <DashboardSection
        className="emptyWorkspaceStarters"
        detail="Select a bounded browser-ready brief, review it in the composer, then send when it matches your intent."
        eyebrow="Quick start"
        icon={Sparkles}
        label="Prompt suggestions"
        title="Choose a proven creative starting point"
      >
        <DashboardCardGrid className="emptyWorkspaceSuggestionGrid" label="Prompt suggestion cards" layout="quad" role="group">
          {homepagePromptLibrary.map((prompt) => (
            <DashboardActionCard
              badge={prompt.runtime}
              detail={prompt.description}
              key={prompt.id}
              onClick={() => onSelectPrompt(prompt.prompt)}
              title={prompt.title}
            />
          ))}
        </DashboardCardGrid>
        <DashboardDisclosure className="emptyWorkspaceJourneyDisclosure" summary="How it works">
          <DashboardProcessRail className="emptyWorkspaceJourney" label="Creative workflow" steps={journey} />
        </DashboardDisclosure>
      </DashboardSection>

      <DashboardSection
        className="emptyWorkspaceCapabilities"
        detail="Four product boundaries keep the first prompt focused and the result reviewable."
        eyebrow="Product capabilities"
        icon={LayoutGrid}
        label="Product capabilities"
        title="From creative brief to retained result"
      >
        <DashboardCardGrid className="emptyWorkspaceCapabilityGrid" label="Product capability cards" layout="compact" role="list">
          {valueHighlights.map((item) => (
            <DashboardInfoCard detail={item.detail} icon={item.icon} key={item.title} role="listitem" title={item.title} />
          ))}
        </DashboardCardGrid>
      </DashboardSection>
    </article>
  );
}

function formatUserPreviewArtifactLabel(snapshot: AssistantWorkspaceSnapshot) {
  const previewArtifact = snapshot.artifacts.find(
    (artifact) =>
      artifact.id === snapshot.preview.sourceArtifactId ||
      artifact.title === snapshot.preview.artifactName ||
      artifact.title === snapshot.preview.sourceArtifactName ||
      artifact.title === snapshot.preview.outputArtifactName
  );

  if (previewArtifact) {
    return formatUserArtifactLabel(previewArtifact);
  }

  return formatUserArtifactLabel({
    ...emptyWorkspaceArtifact,
    language: snapshot.code.language,
    runtime: snapshot.preview.renderer,
    title: [
      snapshot.preview.artifactName,
      snapshot.preview.sourceArtifactName,
      snapshot.preview.outputArtifactName,
      snapshot.preview.title,
      snapshot.preview.renderer,
      snapshot.preview.target
    ].join(" ")
  });
}

type InspectorPanelProps = {
  activeArtifact: ArtifactSummary;
  activeArtifactDocument: ArtifactDocument;
  activeArtifactHighlights: HighlightedLine[];
  activeArtifactId: string;
  activeTab: ProductIntelligenceCategory;
  artifactTransferError: WorkstationError | null;
  copyFeedback: ArtifactActionFeedback | null;
  isStreaming: boolean;
  onArtifactCopy: (artifact: ArtifactSummary) => Promise<void>;
  onArtifactAction: (action: ArtifactAction, artifact: ArtifactSummary) => void;
  onArtifactDelete: (artifact: ArtifactSummary) => void;
  onArtifactRefine: (artifact: ArtifactSummary, instruction: string) => Promise<void>;
  onArtifactRename: (artifact: ArtifactSummary, requestedTitle: string) => string | null;
  onArtifactSelect: (artifact: ArtifactSummary) => void;
  onArtifactTransfer: (
    action: ArtifactTransferAction,
    artifact: ArtifactSummary
  ) => void;
  onClarificationOptionSelect: (option: string) => Promise<void>;
  productIntelligence: ProductIntelligenceModel;
  previewController: PreviewControllerModel;
  runtimeConsole: RuntimeConsoleModel;
  previewRoute: PreviewRendererRoute;
  retrievalRuntime: RetrievalRuntimeModel;
  showDebugPanels: boolean;
  snapshot: AssistantWorkspaceSnapshot;
  streamError: WorkstationError | null;
  sessionUsage: SessionUsageSummary | null;
  telemetryDashboard: TelemetryDashboardModel;
  providerTelemetry: ProviderTelemetryModel;
  transferFeedback: ArtifactActionFeedback | null;
  workflowExecution: WorkflowExecutionModel;
  workflowMode: WorkflowExecutionMode;
  workflowRuntime: WorkflowRuntimeModel;
  workflowIssues: WorkstationError[];
};

function InspectorPanel({
  activeArtifact,
  activeArtifactDocument,
  activeArtifactHighlights,
  activeArtifactId,
  activeTab,
  artifactTransferError,
  copyFeedback,
  isStreaming,
  onArtifactCopy,
  onArtifactAction,
  onArtifactDelete,
  onArtifactRefine,
  onArtifactRename,
  onArtifactSelect,
  onArtifactTransfer,
  onClarificationOptionSelect,
  productIntelligence,
  previewController,
  runtimeConsole,
  previewRoute,
  retrievalRuntime,
  showDebugPanels,
  snapshot,
  streamError,
  sessionUsage,
  telemetryDashboard,
  providerTelemetry,
  transferFeedback,
  workflowExecution,
  workflowMode,
  workflowRuntime,
  workflowIssues
}: InspectorPanelProps) {
  if (!isEstablishedInspectorTab(activeTab)) {
    return <ProductIntelligenceInspector category={activeTab} model={productIntelligence} />;
  }

  if (activeTab === "Code") {
    return (
      <CodeInspector
        artifact={activeArtifact}
        copyFeedback={copyFeedback}
        document={activeArtifactDocument}
        highlightedLines={activeArtifactHighlights}
        onArtifactCopy={onArtifactCopy}
        onArtifactTransfer={onArtifactTransfer}
        showDebugPanels={showDebugPanels}
        transferFeedback={transferFeedback}
      />
    );
  }

  if (activeTab === "Preview") {
    return (
      <PreviewInspector
        controller={previewController}
        preview={snapshot.preview}
        productOutcome={workflowRuntime.summary.productOutcome}
        route={previewRoute}
        showDebugPanels={showDebugPanels}
      />
    );
  }

  if (activeTab === "Runtime") {
    return (
      <RuntimeConsoleInspector
        console={runtimeConsole}
        productOutcome={workflowRuntime.summary.productOutcome}
      />
    );
  }

  if (activeTab === "Workflow") {
    return (
      <WorkflowInspector
        execution={workflowExecution}
        runtime={workflowRuntime}
        issues={workflowIssues}
        workflowMode={workflowMode}
      />
    );
  }

  if (activeTab === "Telemetry") {
    return (
      <TelemetryInspector
        dashboard={telemetryDashboard}
        providerTelemetry={providerTelemetry}
        sessionUsage={sessionUsage}
      />
    );
  }

  if (activeTab === "Artifacts") {
    return (
      <ArtifactsInspector
        activeArtifact={activeArtifact}
        activeArtifactDocument={activeArtifactDocument}
        activeArtifactId={activeArtifactId}
        artifacts={snapshot.artifacts}
        artifactTransferError={artifactTransferError}
        copyFeedback={copyFeedback}
        isStreaming={isStreaming}
        onArtifactAction={onArtifactAction}
        onArtifactDelete={onArtifactDelete}
        onArtifactRefine={onArtifactRefine}
        onArtifactRename={onArtifactRename}
        onArtifactSelect={onArtifactSelect}
        productOutcome={workflowRuntime.summary.productOutcome}
        refinementError={streamError}
        showDebugPanels={showDebugPanels}
        transferFeedback={transferFeedback}
      />
    );
  }

  if (activeTab === "Retrieval") {
    return (
      <RetrievalInspector
        inventory={productIntelligence.domainExperience.knowledgeBase}
        runtime={retrievalRuntime}
        showDebugPanels={showDebugPanels}
      />
    );
  }

  return (
    <OverviewInspector
      activeArtifact={activeArtifact}
      runtime={workflowRuntime}
      workflowExecution={workflowExecution}
      workflowMode={workflowMode}
      isStreaming={isStreaming}
      onClarificationOptionSelect={onClarificationOptionSelect}
      showDebugPanels={showDebugPanels}
      snapshot={snapshot}
    />
  );
}

function OverviewInspector({
  activeArtifact,
  isStreaming,
  onClarificationOptionSelect,
  runtime,
  workflowExecution,
  workflowMode,
  showDebugPanels,
  snapshot
}: {
  activeArtifact: ArtifactSummary;
  isStreaming: boolean;
  onClarificationOptionSelect: (option: string) => Promise<void>;
  runtime: WorkflowRuntimeModel;
  workflowExecution: WorkflowExecutionModel;
  workflowMode: WorkflowExecutionMode;
  showDebugPanels: boolean;
  snapshot: AssistantWorkspaceSnapshot;
}) {
  const visibleWorkflowSteps = selectWorkflowGraphSteps({
    execution: workflowExecution,
    requestedMode: workflowMode,
    steps: runtime.steps
  });
  const workflowProgress = getWorkflowRuntimeProgress(visibleWorkflowSteps);
  const workflowRouteLabel = formatWorkflowGraphRoute({
    execution: workflowExecution,
    requestedMode: workflowMode
  });

  return (
    <section
      aria-label="Overview inspector"
      aria-labelledby="overview-inspector-panel-tab"
      className="inspectorPanel overviewPanel"
      id="overview-inspector-panel"
      role="tabpanel"
    >
      <div className="overviewGrid" aria-label="Compact session summaries">
        <div
          aria-label="Workflow summary"
          className="overviewTile overviewWorkflowTile"
          data-state={runtime.summary.status}
          role="group"
        >
          <header>
            <div>
              <span>Workflow</span>
              <strong>{workflowRouteLabel}</strong>
              <p>{runtime.summary.activity.label}</p>
              <small>{runtime.summary.activity.detail}</small>
            </div>
            <span
              className="liveDot"
              aria-hidden="true"
              data-state={runtime.summary.activity.state}
            />
          </header>
          <div className="workflowSummaryMeta">
            <span>{formatRuntimeDuration(runtime.summary.totalRuntimeMs)}</span>
            <span>{formatRetryCount(runtime.summary.retryCount)}</span>
            <span>
              {showDebugPanels
                ? `${runtime.summary.traceEventCount} trace events`
                : "Traces hidden"}
            </span>
          </div>
          <WorkflowProgress
            label="Overview workflow progress"
            progress={workflowProgress}
          />
          <div className="miniWorkflow" aria-label="Full live workflow">
            {visibleWorkflowSteps.map((step) => (
              <div
                aria-current={
                  step.state === "active" || step.state === "failed"
                    ? "step"
                    : undefined
                }
                className="miniStep"
                data-state={step.state}
                key={step.nodeId}
              >
                <span aria-hidden="true" />
                <div>
                  <strong>{step.displayLabel}</strong>
                  <small>{formatWorkflowMiniMeta(step)}</small>
                </div>
              </div>
            ))}
          </div>
        </div>
        <div className="overviewTile" role="group" aria-label="Artifacts summary">
          <span>Artifacts</span>
          <strong>{snapshot.artifacts.length}</strong>
          <p>{activeArtifact.title}</p>
        </div>
        <div
          aria-label="Product outcome summary"
          className="overviewTile"
          data-state={runtime.summary.status}
          role="group"
        >
          <span>Product outcome</span>
          <strong>{runtime.summary.productOutcome.product_outcome}</strong>
          <p>{runtime.summary.productOutcome.summary}</p>
          <small>{runtime.summary.productOutcome.recovery_action || "No recovery action"}</small>
        </div>
        {snapshot.clarification ? (
          <ClarificationOverviewTile
            clarification={snapshot.clarification}
            disabled={isStreaming}
            onOptionSelect={onClarificationOptionSelect}
          />
        ) : null}
        <div
          aria-label="Image references summary"
          className="overviewTile"
          data-state={snapshot.multimodal.state}
          role="group"
        >
          <span>Image references</span>
          <strong>{snapshot.multimodal.imageAttachments.length}</strong>
          <p>{snapshot.multimodal.status}</p>
        </div>
      </div>
    </section>
  );
}

function ClarificationOverviewTile({
  clarification,
  disabled,
  onOptionSelect
}: {
  clarification: ClarificationSummary;
  disabled: boolean;
  onOptionSelect: (option: string) => Promise<void>;
}) {
  return (
    <div
      aria-label="Clarification summary"
      className="overviewTile overviewClarificationTile"
      data-state="pending"
      role="group"
    >
      <span>Clarification</span>
      <strong>{formatClarificationReason(clarification.reason)}</strong>
      <p>{clarification.summary}</p>
      <small>
        {`${clarification.questions.length} question${
          clarification.questions.length === 1 ? "" : "s"
        } / ${formatConfidenceLabel(clarification.confidence)} confidence`}
      </small>
      <div className="clarificationQuestionStack">
        {clarification.questions.map((question) => (
          <article className="clarificationQuestion" key={question.id}>
            <p>{question.prompt}</p>
            {question.suggestedOptions.length > 0 ? (
              <div className="clarificationOptionGrid">
                {question.suggestedOptions.map((option, index) => {
                  const isRecommended = option === question.defaultRecommendation;
                  const optionNumber = index + 1;

                  return (
                    <button
                      aria-label={`Option ${optionNumber}: ${option}${
                        isRecommended ? " (Recommended)" : ""
                      }`}
                      data-recommended={isRecommended}
                      disabled={disabled}
                      key={option}
                      onClick={() => void onOptionSelect(option)}
                      type="button"
                    >
                      <span aria-hidden="true" className="clarificationOptionNumber">
                        {optionNumber}
                      </span>
                      <span className="clarificationOptionLabel">{option}</span>
                      {isRecommended ? (
                        <span className="clarificationOptionRecommendation">
                          Recommended
                        </span>
                      ) : null}
                    </button>
                  );
                })}
              </div>
            ) : null}
            {question.defaultRecommendation ? (
              <small>{`Recommended: ${question.defaultRecommendation}`}</small>
            ) : null}
          </article>
        ))}
      </div>
    </div>
  );
}

function PreviewInspector({
  controller,
  preview,
  productOutcome,
  route,
  showDebugPanels
}: {
  controller: PreviewControllerModel;
  preview: AssistantWorkspaceSnapshot["preview"];
  productOutcome: WorkflowRuntimeModel["summary"]["productOutcome"];
  route: PreviewRendererRoute;
  showDebugPanels: boolean;
}) {
  if (!showDebugPanels) {
    return (
      <section
        aria-label="Preview inspector"
        aria-labelledby="preview-inspector-panel-tab"
        className="inspectorPanel previewInspectorPanel previewInspectorPanel--user"
        data-state={preview.state}
        id="preview-inspector-panel"
        role="tabpanel"
      >
        <article
          aria-label="Preview canvas status"
          className="previewInspectorHero previewInspectorHero--user"
          data-state={preview.state}
          role="group"
        >
          <div>
            <span>Preview</span>
            <strong>{formatPreviewStateLabel(preview.state, preview.active)}</strong>
            <p>
              {productOutcome.summary}
            </p>
          </div>
          <span>{productOutcome.product_outcome}</span>
        </article>
        <article className="previewInspectorCard previewInspectorCard--user" role="group">
          <header>
            <span>Visual target</span>
            <strong>{route.surfaceTitle}</strong>
          </header>
          <p>
            {productOutcome.recovery_action || "Use the preview shelf for the visual canvas."}
          </p>
        </article>
      </section>
    );
  }

  return (
    <section
      aria-label="Preview inspector"
      aria-labelledby="preview-inspector-panel-tab"
      className="inspectorPanel previewInspectorPanel"
      data-state={preview.state}
      id="preview-inspector-panel"
      role="tabpanel"
    >
      <div className="previewInspectorGrid">
        <article
          aria-label="Preview runtime metadata"
          className="previewInspectorCard"
          role="group"
        >
          <header>
            <span>Runtime context</span>
            <strong>{route.rendererLabel}</strong>
          </header>
          <dl>
            <div>
              <dt>Artifact</dt>
              <dd>{route.selectedArtifactName}</dd>
            </div>
            <div>
              <dt>Source</dt>
              <dd>{route.sourceArtifactName || preview.sourceArtifactName}</dd>
            </div>
            {preview.outputArtifactName ? (
              <div>
                <dt>Runtime output</dt>
                <dd>{preview.outputArtifactName}</dd>
              </div>
            ) : null}
            <div>
              <dt>Target</dt>
              <dd>{route.targetLabel}</dd>
            </div>
            <div>
              <dt>Support</dt>
              <dd>{route.supportLabel}</dd>
            </div>
            <div>
              <dt>Product outcome</dt>
              <dd>{productOutcome.product_outcome}</dd>
            </div>
            <div>
              <dt>Delivery</dt>
              <dd>{productOutcome.deliverable_status}</dd>
            </div>
            <div>
              <dt>Opened from</dt>
              <dd>{preview.trigger}</dd>
            </div>
          </dl>
        </article>

        <article
          aria-label="Preview controls metadata"
          className="previewInspectorCard"
          role="group"
        >
          <header>
            <span>Controls</span>
            <strong>{controller.isSessionOverridden ? "Session override" : "Live session"}</strong>
          </header>
          <div className="previewInspectorControlGrid" role="list">
            {controller.indicators.map((indicator) => (
              <div
                data-tone={indicator.tone}
                key={indicator.id}
                role="listitem"
              >
                <span>{indicator.label}</span>
                <strong>{indicator.value}</strong>
              </div>
            ))}
          </div>
        </article>

        <article
          aria-label="Preview renderer notes"
          className="previewInspectorCard"
          role="group"
        >
          <header>
            <span>Renderer notes</span>
            <strong>{route.surfaceTitle}</strong>
          </header>
          <p>{route.surfaceSummary}</p>
          <div className="previewInspectorNotes">
            {route.notes.map((note) => (
              <span key={note}>{note}</span>
            ))}
          </div>
        </article>
      </div>

      {preview.error ? (
        <SubsystemErrorCallout
          className="previewInspectorError"
          error={preview.error}
          title="Preview runtime failed"
        />
      ) : null}
    </section>
  );
}

type CodeInspectorProps = {
  artifact: ArtifactSummary;
  copyFeedback: ArtifactActionFeedback | null;
  document: ArtifactDocument;
  highlightedLines: HighlightedLine[];
  onArtifactCopy: (artifact: ArtifactSummary) => Promise<void>;
  onArtifactTransfer: (
    action: ArtifactTransferAction,
    artifact: ArtifactSummary
  ) => void;
  showDebugPanels: boolean;
  transferFeedback: ArtifactActionFeedback | null;
};

function CodeInspector({
  artifact,
  copyFeedback,
  document,
  highlightedLines,
  onArtifactCopy,
  onArtifactTransfer,
  showDebugPanels,
  transferFeedback
}: CodeInspectorProps) {
  const transferActions = artifact.actions.filter(
    (action): action is ArtifactTransferAction =>
      action === "Download" || action === "Export"
  );
  const actionMessage = getArtifactActionMessage(
    artifact,
    copyFeedback,
    transferFeedback
  );
  const displayDocumentName = showDebugPanels
    ? document.fileName
    : formatUserArtifactLabel(artifact);
  const actionMessageText =
    actionMessage && !showDebugPanels
      ? toUserArtifactActionMessage(actionMessage)
      : actionMessage;

  return (
    <section
      aria-label="Code inspector"
      aria-labelledby="code-inspector-panel-tab"
      className="inspectorPanel codePanel"
      data-opened-artifact={displayDocumentName}
      id="code-inspector-panel"
      role="tabpanel"
    >
      <header className="codePanelHeader">
        <div>
          <span>{showDebugPanels ? "Active document" : "Generated code"}</span>
          <strong>{displayDocumentName}</strong>
          <p>{document.summary}</p>
        </div>
        <div className="codePanelActions">
          <button
            aria-label={`Copy ${displayDocumentName}`}
            onClick={() => void onArtifactCopy(artifact)}
            type="button"
          >
            {showDebugPanels
              ? getArtifactActionButtonLabel(
                  "Copy",
                  artifact,
                  copyFeedback,
                  transferFeedback
                )
              : getUserArtifactActionButtonLabel(
                  "Copy",
                  artifact,
                  copyFeedback,
                  transferFeedback
                )}
          </button>
          {transferActions.map((transferAction) => (
            <button
              aria-label={`${
                showDebugPanels
                  ? formatArtifactActionLabel(transferAction, artifact)
                  : formatUserArtifactActionLabel(transferAction)
              } ${displayDocumentName}`}
              key={transferAction}
              onClick={() => onArtifactTransfer(transferAction, artifact)}
              type="button"
            >
              {showDebugPanels
                ? getArtifactActionButtonLabel(
                    transferAction,
                    artifact,
                    copyFeedback,
                    transferFeedback
                  )
                : getUserArtifactActionButtonLabel(
                    transferAction,
                    artifact,
                    copyFeedback,
                    transferFeedback
                  )}
            </button>
          ))}
        </div>
      </header>
      <div className="codePanelMeta" aria-label="Artifact metadata" role="list">
        <span role="listitem">{document.languageLabel}</span>
        <span role="listitem">{document.typeLabel}</span>
        <span role="listitem">{document.status}</span>
        <span role="listitem">{document.lineCount} lines</span>
      </div>
      {actionMessageText ? (
        <p className="artifactActionFeedback" aria-live="polite">
          {actionMessageText}
        </p>
      ) : null}
      <div
        aria-label={`${displayDocumentName} content`}
        className="codeViewer"
        role="region"
      >
        <pre>
          <code>
            {highlightedLines.map((line) => (
              <span className="codeLine" key={`${document.artifactId}-${line.lineNumber}`}>
                <span className="codeLineNumber" aria-hidden="true">
                  {String(line.lineNumber).padStart(2, "0")}
                </span>
                <span className="codeLineContent">
                  {line.tokens.map((token, index) => (
                    <span
                      className={`codeToken codeToken--${token.kind}`}
                      key={`${line.lineNumber}-${index}-${token.kind}`}
                    >
                      {token.text || "\u00a0"}
                    </span>
                  ))}
                </span>
              </span>
            ))}
          </code>
        </pre>
      </div>
    </section>
  );
}

function WorkflowInspector({
  execution,
  issues,
  runtime,
  workflowMode
}: {
  execution: WorkflowExecutionModel;
  issues: WorkstationError[];
  runtime: WorkflowRuntimeModel;
  workflowMode: WorkflowExecutionMode;
}) {
  const visibleWorkflowSteps = selectWorkflowGraphSteps({
    execution,
    requestedMode: workflowMode,
    steps: runtime.steps
  });
  const workflowProgress = getWorkflowRuntimeProgress(visibleWorkflowSteps);
  const workflowRouteLabel = formatWorkflowGraphRoute({
    execution,
    requestedMode: workflowMode
  });
  return (
    <section
      aria-label="Workflow inspector"
      aria-labelledby="workflow-inspector-panel-tab"
      className="inspectorPanel workflowPanel"
      id="workflow-inspector-panel"
      role="tabpanel"
    >
      {issues.length > 0 ? (
        <div className="inspectorCockpitIssueStack" aria-label="Workflow runtime issues">
          {issues.map((issue) => (
            <SubsystemErrorCallout
              className="workflowIssueCallout"
              error={issue}
              key={issue.id}
              title="Runtime issue"
            />
          ))}
        </div>
      ) : null}
      <WorkflowExecutionInspector execution={execution} />
      <div className="inspectorCockpitProgress">
        <WorkflowProgress
          label="Workflow inspector progress"
          progress={workflowProgress}
        />
      </div>
      <header className="workflowGraphHeader">
        <div>
          <span>Live route</span>
          <strong>{workflowRouteLabel}</strong>
        </div>
        <small>{`${visibleWorkflowSteps.length} route node${
          visibleWorkflowSteps.length === 1 ? "" : "s"
        }`}</small>
      </header>
      <div
        aria-label="LangGraph workflow visualization"
        className="workflowGraph"
        role="group"
      >
        {visibleWorkflowSteps.map((step, index) => (
          <article
            aria-current={
              step.state === "active" || step.state === "failed"
                ? "step"
                : undefined
            }
            className="workflowNode"
            data-state={step.state}
            key={step.nodeId}
          >
            <span className="nodeIndex">{String(index + 1).padStart(2, "0")}</span>
            <div>
              <strong>{step.displayLabel}</strong>
              <small>{formatWorkflowRuntimeState(step.state)}</small>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function TelemetryInspector({
  dashboard,
  providerTelemetry,
  sessionUsage
}: {
  dashboard: TelemetryDashboardModel;
  providerTelemetry: ProviderTelemetryModel;
  sessionUsage: SessionUsageSummary | null;
}) {
  return (
    <section
      aria-label="Telemetry inspector"
      aria-labelledby="telemetry-inspector-panel-tab"
      className="inspectorPanel telemetryDashboardPanel"
      data-state={dashboard.status}
      id="telemetry-inspector-panel"
      role="tabpanel"
    >
      <header className="telemetryDashboardHero">
        <div>
          <span>Advanced telemetry</span>
          <strong>{dashboard.summary.operatorStatus}</strong>
          <p>{dashboard.summary.signalLabel}</p>
        </div>
        <div className="telemetryDashboardHeroMeta">
          <span>{dashboard.summary.coverageLabel}</span>
          <small>{dashboard.summary.runtimeLabel}</small>
        </div>
      </header>

      <div aria-label="Session usage" className="telemetrySessionUsage" role="group">
        <article>
          <span>Latest request</span>
          <strong>{formatLatestUsageLabel(providerTelemetry, sessionUsage)}</strong>
          <p>Provider-published tokens and estimated cost for the most recent completed request.</p>
        </article>
        <article>
          <span>Current session total</span>
          <strong>{formatUsageTotalLabel(sessionUsage)}</strong>
          <p>{sessionUsage ? `${sessionUsage.runCount} retained request${sessionUsage.runCount === 1 ? "" : "s"}` : "No completed request retained yet."}</p>
        </article>
      </div>

      <div className="telemetrySignalGrid" aria-label="Telemetry signal summary">
        {dashboard.signals.map((signal) => (
          <article
            aria-label={`${signal.label} signal`}
            className="telemetrySignalCard"
            data-tone={signal.tone}
            key={signal.id}
            role="group"
          >
            <span>{signal.label}</span>
            <strong>{signal.value}</strong>
            <p>{signal.detail}</p>
          </article>
        ))}
      </div>


    </section>
  );
}

type ArtifactsInspectorProps = {
  activeArtifact: ArtifactSummary;
  activeArtifactDocument: ArtifactDocument;
  activeArtifactId: string;
  artifacts: ArtifactSummary[];
  artifactTransferError: WorkstationError | null;
  copyFeedback: ArtifactActionFeedback | null;
  isStreaming: boolean;
  onArtifactAction: (action: ArtifactAction, artifact: ArtifactSummary) => void;
  onArtifactDelete: (artifact: ArtifactSummary) => void;
  onArtifactRefine: (artifact: ArtifactSummary, instruction: string) => Promise<void>;
  onArtifactRename: (artifact: ArtifactSummary, requestedTitle: string) => string | null;
  onArtifactSelect: (artifact: ArtifactSummary) => void;
  productOutcome: WorkflowRuntimeModel["summary"]["productOutcome"];
  refinementError: WorkstationError | null;
  showDebugPanels: boolean;
  transferFeedback: ArtifactActionFeedback | null;
};

function ArtifactsInspector({
  activeArtifact,
  activeArtifactDocument,
  activeArtifactId,
  artifacts,
  artifactTransferError,
  copyFeedback,
  isStreaming,
  onArtifactAction,
  onArtifactDelete,
  onArtifactRefine,
  onArtifactRename,
  onArtifactSelect,
  productOutcome,
  refinementError,
  showDebugPanels,
  transferFeedback
}: ArtifactsInspectorProps) {
  const actionMessage = getArtifactActionMessage(
    activeArtifact,
    copyFeedback,
    transferFeedback
  );
  const artifactDeliveryStatus =
    productOutcome.deliverable_status === "UNKNOWN"
      ? activeArtifact.status
      : productOutcome.deliverable_status;

  if (!showDebugPanels) {
    return (
      <UserArtifactsInspector
        activeArtifact={activeArtifact}
        activeArtifactId={activeArtifactId}
        artifacts={artifacts}
        copyFeedback={copyFeedback}
        isStreaming={isStreaming}
        onArtifactAction={onArtifactAction}
        onArtifactDelete={onArtifactDelete}
        onArtifactRefine={onArtifactRefine}
        onArtifactRename={onArtifactRename}
        onArtifactSelect={onArtifactSelect}
        refinementError={refinementError}
        transferFeedback={transferFeedback}
      />
    );
  }

  return (
    <section
      aria-label="Artifacts inspector"
      aria-labelledby="artifacts-inspector-panel-tab"
      className="inspectorPanel artifactPanel"
      id="artifacts-inspector-panel"
      role="tabpanel"
    >
      {artifactTransferError ? (
        <SubsystemErrorCallout
          className="artifactErrorCallout"
          error={artifactTransferError}
          title="Artifact transfer failed"
        />
      ) : null}
      <article
        aria-label="Active artifact details"
        className="artifactDetailCard"
        role="group"
      >
        <header className="artifactDetailHeader">
          <div>
            <span>Selected artifact</span>
            <strong>{activeArtifactDocument.fileName}</strong>
            <p>{sanitizeArtifactDisplaySummary(activeArtifact.summary)}</p>
          </div>
          <div className="artifactBadges">
            <span className="artifactSelected">Selected</span>
            {activeArtifact.isRecommended ? (
              <span className="artifactSelected">Recommended</span>
            ) : null}
            {activeArtifact.refinedFromTitle ? (
              <span className="artifactSelected">Refined</span>
            ) : null}
            <span className="artifactType">
              {getArtifactRuntimeSupportLabel(activeArtifact)}
            </span>
            {activeArtifact.domain ? (
              <span className="artifactType">
                {formatArtifactDomainLabel(activeArtifact.domain)}
              </span>
            ) : null}
            <span className="artifactType">{activeArtifactDocument.typeLabel}</span>
          </div>
        </header>
        <dl className="artifactDetailMeta">
          <div>
            <dt>Language</dt>
            <dd>{activeArtifactDocument.languageLabel}</dd>
          </div>
          <div>
            <dt>Status</dt>
            <dd>{artifactDeliveryStatus}</dd>
          </div>
          <div>
            <dt>Lines</dt>
            <dd>{activeArtifactDocument.lineCount}</dd>
          </div>
          <div>
            <dt>Actions</dt>
            <dd>{activeArtifact.actions.length}</dd>
          </div>
          <div>
            <dt>Domain</dt>
            <dd>{formatArtifactDomainLabel(activeArtifact.domain)}</dd>
          </div>
          <div>
            <dt>Runtime</dt>
            <dd>{formatArtifactRuntimeDetail(activeArtifact)}</dd>
          </div>
          {productOutcome.artifact_runnability !== "UNKNOWN" ? (
            <div>
              <dt>Runnability</dt>
              <dd>{productOutcome.artifact_runnability}</dd>
            </div>
          ) : null}
          {activeArtifact.qualityScore !== undefined &&
          activeArtifact.qualityScore !== null ? (
            <div>
              <dt>Quality</dt>
              <dd>{formatQualityScore(activeArtifact.qualityScore)}</dd>
            </div>
          ) : null}
          {activeArtifact.qualityRank ? (
            <div>
              <dt>Rank</dt>
              <dd>{`#${activeArtifact.qualityRank}`}</dd>
            </div>
          ) : null}
        </dl>
        {activeArtifact.type === "code" && activeArtifact.actions.length > 0 ? (
          <ArtifactRefinementPanel
            artifact={activeArtifact}
            disabled={isStreaming}
            error={refinementError}
            key={activeArtifact.id}
            onArtifactRefine={onArtifactRefine}
          />
        ) : null}
        <ArtifactActionRow
          artifact={activeArtifact}
          copyFeedback={copyFeedback}
          onArtifactAction={onArtifactAction}
          transferFeedback={transferFeedback}
        />
        <ArtifactRenameControl
          artifact={activeArtifact}
          onArtifactRename={onArtifactRename}
        />
        <button
          className="artifactDeleteButton"
          onClick={() => onArtifactDelete(activeArtifact)}
          type="button"
        >
          Delete artifact
        </button>
        {actionMessage ? (
          <p className="artifactActionFeedback" aria-live="polite">
            {actionMessage}
          </p>
        ) : null}
      </article>
      <div className="artifactList">
        {artifacts.map((artifact) => (
          <ArtifactCard
            artifact={artifact}
            isActive={artifact.id === activeArtifactId}
            key={artifact.id}
            copyFeedback={copyFeedback}
            onArtifactAction={onArtifactAction}
            transferFeedback={transferFeedback}
          />
        ))}
      </div>
    </section>
  );
}

function UserArtifactsInspector({
  activeArtifact,
  activeArtifactId,
  artifacts,
  copyFeedback,
  isStreaming,
  onArtifactAction,
  onArtifactDelete,
  onArtifactRefine,
  onArtifactRename,
  onArtifactSelect,
  refinementError,
  transferFeedback
}: {
  activeArtifact: ArtifactSummary;
  activeArtifactId: string;
  artifacts: ArtifactSummary[];
  copyFeedback: ArtifactActionFeedback | null;
  isStreaming: boolean;
  onArtifactAction: (action: ArtifactAction, artifact: ArtifactSummary) => void;
  onArtifactDelete: (artifact: ArtifactSummary) => void;
  onArtifactRefine: (artifact: ArtifactSummary, instruction: string) => Promise<void>;
  onArtifactRename: (artifact: ArtifactSummary, requestedTitle: string) => string | null;
  onArtifactSelect: (artifact: ArtifactSummary) => void;
  refinementError: WorkstationError | null;
  transferFeedback: ArtifactActionFeedback | null;
}) {
  const actionMessage = getArtifactActionMessage(
    activeArtifact,
    copyFeedback,
    transferFeedback
  );
  const userArtifactLabels = buildUserArtifactDisplayLabels(artifacts);
  const activeArtifactLabel =
    userArtifactLabels.get(activeArtifact.id) ?? formatUserArtifactLabel(activeArtifact);

  return (
    <section
      aria-label="Saved outputs inspector"
      aria-labelledby="artifacts-inspector-panel-tab"
      className="inspectorPanel artifactPanel artifactPanel--user"
      id="artifacts-inspector-panel"
      role="tabpanel"
    >
      <article className="savedOutputsHero" role="group" aria-label="Saved outputs">
        <div>
          <span>Saved outputs</span>
          <strong>{activeArtifactLabel}</strong>
          <p>{getUserArtifactSummary(activeArtifact)}</p>
        </div>
        <span>{`${artifacts.length} saved`}</span>
      </article>
      {artifacts.length > 1 ? (
        <div className="savedOutputList" role="list" aria-label="Saved output list">
          {artifacts.map((artifact) => {
            const isActive = artifact.id === activeArtifactId;
            return (
              <div key={artifact.id} role="listitem">
                <button
                  aria-pressed={isActive}
                  className="savedOutputCard"
                  data-active={isActive ? "true" : "false"}
                  onClick={() => onArtifactSelect(artifact)}
                  type="button"
                >
                  <span>{formatUserArtifactRuntimeLabel(artifact)}</span>
                  <strong>
                    {userArtifactLabels.get(artifact.id) ??
                      formatUserArtifactLabel(artifact)}
                  </strong>
                  <small>{getUserArtifactSummary(artifact)}</small>
                </button>
              </div>
            );
          })}
        </div>
      ) : null}
      {activeArtifact.type === "code" && activeArtifact.actions.length > 0 ? (
        <ArtifactRefinementPanel
          artifact={activeArtifact}
          disabled={isStreaming}
          error={refinementError}
          key={activeArtifact.id}
          onArtifactRefine={onArtifactRefine}
        />
      ) : null}
      <UserArtifactActionRow
        artifact={activeArtifact}
        copyFeedback={copyFeedback}
        onArtifactAction={onArtifactAction}
        transferFeedback={transferFeedback}
      />
      <ArtifactRenameControl
        artifact={activeArtifact}
        onArtifactRename={onArtifactRename}
      />
      <button
        className="artifactDeleteButton"
        onClick={() => onArtifactDelete(activeArtifact)}
        type="button"
      >
        Delete saved output
      </button>
      {actionMessage ? (
        <p className="artifactActionFeedback" aria-live="polite">
          {toUserArtifactActionMessage(actionMessage)}
        </p>
      ) : null}
    </section>
  );
}

function UserArtifactActionRow({
  artifact,
  copyFeedback,
  onArtifactAction,
  transferFeedback
}: ArtifactActionRowProps) {
  return (
    <div className="artifactActions artifactActions--user">
      {artifact.actions.map((action) => (
        <button
          aria-label={`${formatUserArtifactActionLabel(action)} ${formatUserArtifactLabel(
            artifact
          )}`}
          data-action={action.toLowerCase()}
          key={action}
          onClick={() => onArtifactAction(action, artifact)}
          type="button"
        >
          {getUserArtifactActionButtonLabel(
            action,
            artifact,
            copyFeedback,
            transferFeedback
          )}
        </button>
      ))}
    </div>
  );
}

type WorkspaceQuickActionsProps = {
  activeTab: ProductIntelligenceCategory;
  hasBlockingApproval: boolean;
  isFocusMode: boolean;
  isPreviewAvailable: boolean;
  isPreviewOpen: boolean;
  onFocusModeToggle: () => void;
  onOpenTab: (tab: ProductIntelligenceCategory) => void;
  onPreviewToggle: () => void;
  onWorkspaceClear: () => void;
  showDebugPanels: boolean;
};

function WorkspaceQuickActions({
  activeTab,
  hasBlockingApproval,
  isFocusMode,
  isPreviewAvailable,
  isPreviewOpen,
  onFocusModeToggle,
  onOpenTab,
  onPreviewToggle,
  onWorkspaceClear,
  showDebugPanels
}: WorkspaceQuickActionsProps) {
  return (
    <div aria-label="Quick actions" className="commandMenuGrid" role="group">
        {showDebugPanels ? (
          <button
            aria-label="Overview Return to the compact session summary"
            data-active={activeTab === "Overview"}
            onClick={() => onOpenTab("Overview")}
            type="button"
          >
            <strong>Overview</strong>
            <span>Return to the compact session summary.</span>
          </button>
        ) : null}
        <button
          aria-label="Preview Inspect the current visual output and preview readiness"
          data-active={activeTab === "Preview"}
          onClick={() => onOpenTab("Preview")}
          type="button"
        >
          <strong>Preview</strong>
          <span>Inspect the current visual output and preview readiness.</span>
        </button>
        {showDebugPanels ? (
          <button
            aria-label="Runtime console Inspect live runtime status, FPS, reloads, and renderer errors"
            data-active={activeTab === "Runtime"}
            onClick={() => onOpenTab("Runtime")}
            type="button"
          >
            <strong>Runtime console</strong>
            <span>Inspect live runtime status, FPS, reloads, and renderer errors.</span>
          </button>
        ) : null}
        <button
          aria-label={`${showDebugPanels ? "Artifacts" : "Saved"} Inspect generated and saved results`}
          data-active={activeTab === "Artifacts"}
          onClick={() => onOpenTab("Artifacts")}
          type="button"
        >
          <strong>{showDebugPanels ? "Artifacts" : "Saved"}</strong>
          <span>Inspect generated and saved results.</span>
        </button>
        <button
          aria-label="Code Open generated code"
          data-active={activeTab === "Code"}
          onClick={() => onOpenTab("Code")}
          type="button"
        >
          <strong>Code</strong>
          <span>Open generated code.</span>
        </button>
        {showDebugPanels ? (
          <>
            <button
              aria-label="Workflow inspector Review the live orchestration runtime"
              data-active={activeTab === "Workflow"}
              onClick={() => onOpenTab("Workflow")}
              type="button"
            >
              <strong>Workflow inspector</strong>
              <span>Review the live orchestration runtime.</span>
            </button>
            <button
              aria-label="Telemetry dashboard Inspect runtime, provider, retrieval, and observability signals"
              data-active={activeTab === "Telemetry"}
              onClick={() => onOpenTab("Telemetry")}
              type="button"
            >
              <strong>Telemetry dashboard</strong>
              <span>Inspect runtime, provider, retrieval, and observability signals.</span>
            </button>
          </>
        ) : null}
        <button
          aria-label="Toggle preview shelf"
          disabled={!isPreviewAvailable}
          onClick={onPreviewToggle}
          type="button"
        >
          <strong>{isPreviewOpen ? "Close preview shelf" : "Open preview shelf"}</strong>
          <span>
            {isPreviewAvailable
              ? "Keep the lower preview shelf available on demand."
              : "No preview target is available yet."}
          </span>
        </button>
        <button
          aria-label="Toggle Fullscreen Creative Session from quick actions"
          onClick={onFocusModeToggle}
          type="button"
        >
          <strong>
            {isFocusMode
              ? "Exit Fullscreen Creative Session"
              : "Enter Fullscreen Creative Session"}
          </strong>
          <span>Collapse or restore Sessions, Inspector, and the preview shelf.</span>
        </button>
        <button
          aria-label="Clear workspace session"
          disabled={hasBlockingApproval}
          onClick={onWorkspaceClear}
          type="button"
        >
          <strong>Clear workspace session</strong>
          <span>
            {hasBlockingApproval
              ? "Finish the active operator checkpoint before resetting the session."
              : "Reset the local creative session to the starter snapshot."}
          </span>
        </button>
    </div>
  );
}

function ThemePresetsPanel({
  activeTheme,
  onRequestClose,
  onSelectTheme
}: {
  activeTheme: WorkspacePreferences["theme"];
  onRequestClose: () => void;
  onSelectTheme: (theme: WorkspacePreferences["theme"]) => void;
}) {
  return (
    <ApplicationFloatingPanel
      description="Switch the workspace accent and shell tone without changing the layout."
      id="theme-presets-panel"
      label="Theme presets"
      onRequestClose={onRequestClose}
      title="Theme presets"
    >
      <ThemePresetPicker activeTheme={activeTheme} onSelectTheme={onSelectTheme} />
    </ApplicationFloatingPanel>
  );
}

type WorkspaceSettingsPanelProps = WorkspaceQuickActionsProps & {
  layoutState: WorkspaceLayoutState;
  onDensityChange: (density: WorkspaceLayoutState["density"]) => void;
  onOpenDashboardSettings: () => void;
  onPreferencesChange: (preferences: Partial<WorkspacePreferences>) => void;
  onRequestClose: () => void;
  preferences: WorkspacePreferences;
};

function WorkspaceSettingsPanel({
  activeTab,
  hasBlockingApproval,
  isFocusMode,
  isPreviewAvailable,
  isPreviewOpen,
  layoutState,
  onDensityChange,
  onFocusModeToggle,
  onOpenDashboardSettings,
  onOpenTab,
  onPreferencesChange,
  onPreviewToggle,
  onRequestClose,
  onWorkspaceClear,
  preferences,
  showDebugPanels
}: WorkspaceSettingsPanelProps) {
  return (
    <ApplicationFloatingPanel
      className="applicationFloatingPanel--settings"
      description="Compact controls for the active creative session."
      id="workspace-settings-panel"
      label="Workspace settings"
      onRequestClose={onRequestClose}
      title="Workspace settings"
    >
      <button
        className="applicationFloatingPanelReference"
        onClick={onOpenDashboardSettings}
        type="button"
      >
        <LayoutDashboard aria-hidden="true" size={17} />
        <span>
          <strong>Open Dashboard Settings</strong>
          <small>Complete appearance, typography, privacy, and prompt defaults.</small>
        </span>
      </button>
      <div className="settingsSection">
        <div className="settingsSectionHeader">
          <strong>Active session</strong>
          <p>Adjust the compact cockpit without duplicating the complete Dashboard reference.</p>
        </div>
        <div className="settingsChoiceRow" role="group" aria-label="Workspace density options">
          <button
            aria-pressed={layoutState.density === "cozy"}
            data-active={layoutState.density === "cozy"}
            onClick={() => onDensityChange("cozy")}
            type="button"
          >
            Cozy
          </button>
          <button
            aria-pressed={layoutState.density === "compact"}
            data-active={layoutState.density === "compact"}
            onClick={() => onDensityChange("compact")}
            type="button"
          >
            Compact
          </button>
        </div>
        <div className="settingsToggle">
          <div>
            <strong>Preview behavior</strong>
            <p>Open the preview shelf automatically when a ready preview artifact arrives.</p>
          </div>
          <button
            aria-label="Preview auto-open"
            aria-pressed={preferences.autoOpenPreview}
            data-active={preferences.autoOpenPreview}
            onClick={() =>
              onPreferencesChange({
                autoOpenPreview: !preferences.autoOpenPreview
              })
            }
            type="button"
          >
            {preferences.autoOpenPreview ? "Auto" : "Manual"}
          </button>
        </div>
      </div>
      <div className="settingsSection">
        <div className="settingsToggle">
          <div>
            <strong>Display mode</strong>
            <p>User Mode keeps the workspace quiet; Developer Mode exposes traces.</p>
          </div>
          <button
            aria-label="Display mode"
            aria-pressed={preferences.showDebugPanels}
            data-active={preferences.showDebugPanels}
            onClick={() =>
              onPreferencesChange({
                showDebugPanels: !preferences.showDebugPanels
              })
            }
            type="button"
          >
            {preferences.showDebugPanels ? "Developer" : "User"}
          </button>
        </div>
      </div>
      <div className="settingsSection">
        <div className="settingsToggle">
          <div>
            <strong>Personalization</strong>
            <p>
              {preferences.personalizationEnabled
                ? `${preferences.feedbackSignals.length} explicit local preference signal${preferences.feedbackSignals.length === 1 ? "" : "s"} can be considered when relevant.`
                : "Stored preference signals are not used for new requests."}
            </p>
          </div>
          <button
            aria-label="Personalization"
            aria-pressed={preferences.personalizationEnabled}
            data-active={preferences.personalizationEnabled}
            onClick={() =>
              onPreferencesChange({
                personalizationEnabled: !preferences.personalizationEnabled
              })
            }
            type="button"
          >
            {preferences.personalizationEnabled ? "On" : "Off"}
          </button>
        </div>
      </div>
      <div className="settingsSection">
        <div className="settingsSectionHeader">
          <strong>Quick actions</strong>
          <p>Move directly to the active workspace surface.</p>
        </div>
        <WorkspaceQuickActions
          activeTab={activeTab}
          hasBlockingApproval={hasBlockingApproval}
          isFocusMode={isFocusMode}
          isPreviewAvailable={isPreviewAvailable}
          isPreviewOpen={isPreviewOpen}
          onFocusModeToggle={onFocusModeToggle}
          onOpenTab={onOpenTab}
          onPreviewToggle={onPreviewToggle}
          onWorkspaceClear={onWorkspaceClear}
          showDebugPanels={showDebugPanels}
        />
      </div>
    </ApplicationFloatingPanel>
  );
}

function ThemePresetPicker({
  activeTheme,
  compact = false,
  onSelectTheme
}: {
  activeTheme: WorkspacePreferences["theme"];
  compact?: boolean;
  onSelectTheme: (theme: WorkspacePreferences["theme"]) => void;
}) {
  return (
    <div
      className="themePresetList"
      data-compact={compact ? "true" : "false"}
      role="list"
    >
      {themePresetOptions.map((option) => (
        <div
          className="themePresetListItem"
          key={option.value}
          role="listitem"
        >
          <button
            aria-label={`Use ${option.label} theme`}
            aria-pressed={activeTheme === option.value}
            className="themePresetButton"
            data-active={activeTheme === option.value}
            data-theme={option.value}
            onClick={() => onSelectTheme(option.value)}
            type="button"
          >
            <span
              aria-hidden="true"
              className="themePresetSwatch"
              style={
                {
                  "--theme-accent": option.accent,
                  "--theme-surface": option.surface
                } as CSSProperties
              }
            />
            <div>
              <strong>{option.label}</strong>
              <p>{option.description}</p>
            </div>
            <small>{activeTheme === option.value ? "Active" : "Preset"}</small>
          </button>
        </div>
      ))}
    </div>
  );
}

type ArtifactCardProps = {
  artifact: ArtifactSummary;
  copyFeedback: ArtifactActionFeedback | null;
  isActive: boolean;
  onArtifactAction: (action: ArtifactAction, artifact: ArtifactSummary) => void;
  transferFeedback: ArtifactActionFeedback | null;
};

function ArtifactCard({
  artifact,
  copyFeedback,
  isActive,
  onArtifactAction,
  transferFeedback
}: ArtifactCardProps) {
  return (
    <article
      aria-current={isActive ? "true" : undefined}
      aria-label={`${artifact.title} artifact`}
      className="artifactItem"
      data-active={isActive}
    >
      <div className="artifactItemHeader">
        <div>
          <strong>{artifact.title}</strong>
          <span>
            {artifact.language} / {getArtifactTypeLabel(artifact.type)} / {artifact.status}
          </span>
        </div>
        <div className="artifactBadges">
          {artifact.isRecommended ? (
            <span className="artifactSelected">Recommended</span>
          ) : null}
          {artifact.refinedFromTitle ? (
            <span className="artifactSelected">Refined</span>
          ) : null}
          {isActive ? <span className="artifactSelected">Selected</span> : null}
          <span className="artifactType">{getArtifactRuntimeSupportLabel(artifact)}</span>
          {artifact.domain ? (
            <span className="artifactType">{formatArtifactDomainLabel(artifact.domain)}</span>
          ) : null}
          <span className="artifactType">{getArtifactTypeLabel(artifact.type)}</span>
        </div>
      </div>
      <p>{sanitizeArtifactDisplaySummary(artifact.summary)}</p>
      {artifact.critique ? (
        <p className="artifactQualityLine">
          {`Rank #${artifact.critique.rank} / Quality ${formatQualityScore(
            artifact.critique.overallScore
          )}: ${artifact.critique.rationale}`}
        </p>
      ) : null}
      <ArtifactActionRow
        artifact={artifact}
        copyFeedback={copyFeedback}
        onArtifactAction={onArtifactAction}
        transferFeedback={transferFeedback}
      />
    </article>
  );
}

type ArtifactActionRowProps = {
  artifact: ArtifactSummary;
  copyFeedback: ArtifactActionFeedback | null;
  onArtifactAction: (action: ArtifactAction, artifact: ArtifactSummary) => void;
  transferFeedback: ArtifactActionFeedback | null;
};

function ArtifactActionRow({
  artifact,
  copyFeedback,
  onArtifactAction,
  transferFeedback
}: ArtifactActionRowProps) {
  return (
    <div className="artifactActions">
      {artifact.actions.map((action) => (
        <button
          aria-label={`${formatArtifactActionLabel(action, artifact)} ${artifact.title}`}
          data-action={action.toLowerCase()}
          key={action}
          onClick={() => onArtifactAction(action, artifact)}
          type="button"
        >
          {getArtifactActionButtonLabel(action, artifact, copyFeedback, transferFeedback)}
        </button>
      ))}
    </div>
  );
}

function ArtifactRenameControl({
  artifact,
  onArtifactRename
}: {
  artifact: ArtifactSummary;
  onArtifactRename: (artifact: ArtifactSummary, requestedTitle: string) => string | null;
}) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(artifact.title);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setValue(artifact.title);
    setError(null);
    setEditing(false);
  }, [artifact.id, artifact.title]);

  if (!editing) {
    return (
      <div className="artifactRenameControl">
        <button
          aria-label={`Rename ${artifact.title}`}
          onClick={() => setEditing(true)}
          type="button"
        >
          Rename file
        </button>
      </div>
    );
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const title = onArtifactRename(artifact, value);
    if (!title) {
      setError("Enter a file name using letters, numbers, spaces, or dashes.");
      return;
    }
    setValue(title);
    setEditing(false);
  }

  return (
    <form className="artifactRenameControl" onSubmit={handleSubmit}>
      <label>
        File name
        <input
          aria-label={`New file name for ${artifact.title}`}
          autoFocus
          onChange={(event) => setValue(event.currentTarget.value)}
          value={value}
        />
      </label>
      <button type="submit">Save name</button>
      <button onClick={() => setEditing(false)} type="button">
        Cancel
      </button>
      {error ? <small role="alert">{error}</small> : null}
    </form>
  );
}

function clearTimer(timerId: number | null) {
  if (timerId !== null) {
    window.clearTimeout(timerId);
  }
}

function clampNumber(value: number, minimum: number, maximum: number) {
  return Math.min(Math.max(Math.round(value), minimum), maximum);
}

function setFeedbackState(
  artifactId: string,
  action: ArtifactActionFeedback["action"],
  state: ArtifactActionFeedback["state"],
  timerRef: { current: number | null },
  setFeedback: (feedback: ArtifactActionFeedback | null) => void
) {
  clearTimer(timerRef.current);
  setFeedback({ action, artifactId, state });
  timerRef.current = window.setTimeout(() => {
    setFeedback(null);
    timerRef.current = null;
  }, artifactFeedbackDurationMs);
}

function formatUserArtifactLabel(artifact: ArtifactSummary) {
  const searchable = [
    artifact.title,
    artifact.language,
    artifact.runtime ?? "",
    artifact.rendererId ?? "",
    artifact.domain ?? ""
  ]
    .join(" ")
    .toLowerCase();

  if (
    artifact.domain === "react_three_fiber" ||
    artifact.title.toLowerCase().endsWith(".r3f.tsx") ||
    searchable.includes("react three fiber")
  ) {
    return "React Three Fiber";
  }

  if (searchable.includes("three")) {
    return "Three.js Scene";
  }

  if (searchable.includes("glsl") || searchable.includes("shader")) {
    return "GLSL Shader";
  }

  if (searchable.includes("hydra")) {
    return "Hydra Pattern";
  }

  if (searchable.includes("p5")) {
    return "P5 Sketch";
  }

  return "Generated Code";
}

function buildUserArtifactDisplayLabels(artifacts: ArtifactSummary[]) {
  const baseLabels = artifacts.map((artifact) => ({
    artifact,
    label: formatUserArtifactLabel(artifact)
  }));
  const totals = baseLabels.reduce((counts, item) => {
    counts.set(item.label, (counts.get(item.label) ?? 0) + 1);
    return counts;
  }, new Map<string, number>());
  const seen = new Map<string, number>();
  const labels = new Map<string, string>();

  for (const item of baseLabels) {
    const total = totals.get(item.label) ?? 0;
    if (total <= 1) {
      labels.set(item.artifact.id, item.label);
      continue;
    }

    const nextIndex = (seen.get(item.label) ?? 0) + 1;
    seen.set(item.label, nextIndex);
    labels.set(item.artifact.id, `${item.label} ${nextIndex}`);
  }

  return labels;
}

function formatUserArtifactRuntimeLabel(artifact: ArtifactSummary) {
  const label = formatUserArtifactLabel(artifact);

  if (label === "Generated Code") {
    return artifact.type === "export" ? "Export" : "Code";
  }

  return label
    .replace(" Scene", "")
    .replace(" Sketch", "")
    .replace(" Pattern", "");
}

function getUserArtifactSummary(artifact: ArtifactSummary) {
  if (artifact.summary.trim()) {
    return sanitizeArtifactDisplaySummary(artifact.summary);
  }

  if (isArtifactPreviewable(artifact)) {
    return "Visual output ready for preview.";
  }

  return "Generated output ready to inspect.";
}

function sanitizeArtifactDisplaySummary(summary: string) {
  const normalized = summary.trim();
  if (
    /(?:\b(?:error|exception)\b|\n\s*at\s+\S|\b(?:stack|trace)\b)/i.test(
      normalized
    ) ||
    normalized.length > 360
  ) {
    return "Preview did not start. Open Code to review the generated artifact or choose another saved output.";
  }
  return normalized;
}

function formatUserArtifactActionLabel(action: ArtifactAction) {
  if (action === "Open") {
    return "Open";
  }

  if (action === "Copy") {
    return "Copy";
  }

  if (action === "Download") {
    return "Download";
  }

  if (action === "Export") {
    return "Export project";
  }

  return action;
}

function getUserArtifactActionButtonLabel(
  action: ArtifactAction,
  artifact: ArtifactSummary,
  copyFeedback: ArtifactActionFeedback | null,
  transferFeedback: ArtifactActionFeedback | null
) {
  if (action === "Copy" && copyFeedback?.artifactId === artifact.id) {
    return copyFeedback.state === "success" ? "Copied" : "Copy unavailable";
  }

  if (
    (action === "Download" || action === "Export") &&
    transferFeedback?.artifactId === artifact.id &&
    transferFeedback.action === action
  ) {
    return transferFeedback.state === "success" ? "Saved" : "Save unavailable";
  }

  return formatUserArtifactActionLabel(action);
}

function toUserArtifactActionMessage(message: string) {
  return message
    .replace(/copied to clipboard\./i, "copied.")
    .replace(/Clipboard is unavailable for .+\./i, "Copy is unavailable.")
    .replace(/Download is unavailable for .+\./i, "Save is unavailable.")
    .replace(/Export is unavailable for .+\./i, "Save is unavailable.");
}

function getArtifactTypeLabel(type: ArtifactSummary["type"]) {
  switch (type) {
    case "code":
      return "Source code";
    case "preview":
      return "Preview manifest";
    case "export":
      return "Markdown export";
    default:
      return type;
  }
}

function formatQualityScore(score: number) {
  return `${Math.round(score * 100)}%`;
}

function getArtifactRuntimeSupportLabel(artifact: ArtifactSummary) {
  return isArtifactPreviewable(artifact) ? "Previewable" : "Code-only";
}

function formatArtifactRuntimeDetail(artifact: ArtifactSummary) {
  if (!isArtifactPreviewable(artifact)) {
    return "Code-only";
  }
  const matchedRenderer = matchCreativePreviewRenderer(artifact);
  const runtime = artifact.runtime
    ? formatArtifactRuntimeLabel(artifact.runtime)
    : matchedRenderer?.displayName ?? "Preview";
  const rendererId = artifact.rendererId ?? matchedRenderer?.id ?? null;
  return rendererId ? `${runtime} / ${rendererId}` : runtime;
}

function formatArtifactRuntimeLabel(runtime: string) {
  switch (runtime) {
    case "p5":
      return "p5.js";
    case "three":
      return "Three.js";
    case "glsl":
      return "GLSL";
    case "hydra":
      return "Hydra";
    case "gsap":
      return "GSAP";
    case "svg":
      return "SVG";
    case "canvas":
      return "Canvas";
    case "tone":
      return "Tone.js";
    default:
      return sentenceCase(runtime.replace(/[_-]+/g, " "));
  }
}

function formatArtifactDomainLabel(domain: string | null | undefined) {
  if (!domain) {
    return "No selected domain";
  }
  switch (domain) {
    case "p5_js":
      return "p5.js";
    case "three_js":
      return "Three.js";
    case "react_three_fiber":
      return "React Three Fiber";
    case "glsl":
      return "GLSL";
    case "hydra":
      return "Hydra";
    case "gsap":
      return "GSAP";
    case "svg":
      return "SVG";
    case "canvas":
    case "canvas_2d":
      return "Canvas";
    case "tone_js":
      return "Tone.js";
    default:
      return sentenceCase(domain.replace(/_/g, " "));
  }
}

function sentenceCase(value: string) {
  const normalized = value.trim();
  return normalized ? normalized[0].toUpperCase() + normalized.slice(1) : normalized;
}

function formatPreviewStateLabel(
  state: AssistantWorkspaceSnapshot["preview"]["state"],
  isOpen: boolean
) {
  switch (state) {
    case "generating":
      return "Generating";
    case "ready":
      return isOpen ? "Open" : "Ready";
    case "error":
      return "Preview failed";
    case "unavailable":
      return "Unavailable";
    default:
      return isOpen ? "Open" : "Ready";
  }
}

function getArtifactActionButtonLabel(
  action: ArtifactAction,
  artifact: ArtifactSummary,
  copyFeedback: ArtifactActionFeedback | null,
  transferFeedback: ArtifactActionFeedback | null
) {
  const isBundleExport = isProjectBundleExportArtifact(artifact);

  if (action === "Copy" && copyFeedback?.artifactId === artifact.id) {
    return copyFeedback.state === "success" ? "Copied" : "Copy Unavailable";
  }

  if (
    (action === "Download" || action === "Export") &&
    transferFeedback?.artifactId === artifact.id &&
    transferFeedback.action === action
  ) {
    if (transferFeedback.state === "success") {
      if (action === "Export" && isBundleExport) {
        return "Bundle Exported";
      }

      return action === "Export" ? "Exported" : "Downloaded";
    }

    if (action === "Export" && isBundleExport) {
      return "Bundle Unavailable";
    }

    return action === "Export" ? "Export Unavailable" : "Download Unavailable";
  }

  return formatArtifactActionLabel(action, artifact);
}

function getArtifactActionMessage(
  artifact: ArtifactSummary,
  copyFeedback: ArtifactActionFeedback | null,
  transferFeedback: ArtifactActionFeedback | null
) {
  if (copyFeedback?.artifactId === artifact.id) {
    return copyFeedback.state === "success"
      ? `${artifact.title} copied to clipboard.`
      : `Clipboard is unavailable for ${artifact.title}.`;
  }

  if (transferFeedback?.artifactId === artifact.id) {
    const transferAction = transferFeedback.action as ArtifactTransferAction;
    const isBundleExport = isProjectBundleExportAction(transferAction, artifact);
    const transferVerb = transferAction === "Export" ? "exported" : "downloaded";
    const transferTarget = isBundleExport ? "Project bundle" : artifact.title;

    return transferFeedback.state === "success"
      ? `${transferTarget} ${transferVerb}.`
      : isBundleExport
        ? "Bundle export is unavailable for the current workspace."
        : `${transferAction === "Export" ? "Export" : "Download"} is unavailable for ${artifact.title}.`;
  }

  return null;
}

function buildCodeSummaryForArtifact(
  baseCode: AssistantWorkspaceSnapshot["code"],
  artifact: ArtifactSummary
): AssistantWorkspaceSnapshot["code"] {
  if (artifact.type !== "code") {
    return baseCode;
  }

  return {
    title: artifact.title,
    language: artifact.language,
    status: artifact.status,
    excerpt: splitArtifactContentLines(artifact.content ?? baseCode.excerpt.join("\n"))
  };
}

function buildArtifactRefinementRequest({
  artifact,
  document,
  instruction
}: {
  artifact: ArtifactSummary;
  document: ArtifactDocument;
  instruction: string;
}): AssistantArtifactRefinementRequest {
  return enrichArtifactRefinementRequest({
    artifactId: artifact.id,
    title: artifact.title,
    language: document.languageLabel,
    content: document.content,
    instruction,
    domain: artifact.domain ?? null,
    runtime: artifact.runtime ?? null,
    rendererId: artifact.rendererId ?? null,
    previewEligible: artifact.previewEligible ?? isArtifactPreviewable(artifact),
    qualityScore: artifact.qualityScore ?? artifact.critique?.overallScore ?? null,
    qualityRank: artifact.qualityRank ?? artifact.critique?.rank ?? null,
    critiqueRationale: artifact.critique?.rationale ?? null,
    refinementGuidance:
      artifact.critique?.refinementGuidance ?? artifact.refinementReason ?? null,
    creativeTranslation: artifact.creativeTranslation ?? null,
    creativePlan: artifact.creativePlan ?? null,
    critique: artifact.critique ?? null
  }, artifact);
}

function annotateRefinedHydration(
  hydration: LiveArtifactHydrationResult,
  refinement: PendingArtifactRefinement | null,
  sourceSnapshot: AssistantWorkspaceSnapshot
): LiveArtifactHydrationResult {
  if (!refinement || !hydration.artifact) {
    return hydration;
  }

  const sourceArtifact =
    sourceSnapshot.artifacts.find(
      (artifact) => artifact.id === refinement.artifactId
    ) ?? null;
  const idCollidesWithSource = hydration.artifact.id === refinement.artifactId;
  const refinedArtifactId = idCollidesWithSource
    ? createRefinedArtifactId(refinement.artifactId, sourceSnapshot.artifacts)
    : hydration.artifact.id;
  const refinedArtifactTitle = createRefinedArtifactTitle(
    refinement.title,
    sourceSnapshot.artifacts
  );
  const refinementPasses = appendRefinementPassRecord({
    refinement,
    resultArtifact: {
      ...hydration.artifact,
      id: refinedArtifactId,
      title: refinedArtifactTitle
    },
    sourceArtifact
  });
  const refinedArtifact: ArtifactSummary = {
    ...hydration.artifact,
    id: refinedArtifactId,
    title: refinedArtifactTitle,
    status: "Refined",
    refinedAt: refinement.requestedAt,
    refinedFromArtifactId: refinement.artifactId,
    refinedFromTitle: refinement.title,
    refinementInstruction: refinement.instruction,
    refinementPasses,
    refinementReason:
      hydration.artifact.refinementReason ??
      refinement.refinementGuidance ??
      refinement.instruction,
    summary: buildRefinedArtifactSummary(hydration.artifact, refinement)
  };
  const nextArtifacts = [
    refinedArtifact,
    ...(idCollidesWithSource && sourceArtifact ? [sourceArtifact] : []),
    ...hydration.snapshot.artifacts.filter(
      (artifact) =>
        artifact.id !== hydration.artifact?.id &&
        artifact.id !== refinedArtifact.id &&
        (!idCollidesWithSource || artifact.id !== refinement.artifactId)
    )
  ];
  const refinedPreviewWasPromoted =
    hydration.snapshot.preview.sourceArtifactId === hydration.artifact.id ||
    hydration.previewArtifactId === hydration.artifact.id;
  const sourcePreviewFallback =
    refinedPreviewWasPromoted &&
    !isArtifactPreviewable(refinedArtifact) &&
    sourceArtifact &&
    isArtifactPreviewable(sourceArtifact)
      ? sourceArtifact
      : null;
  const preview =
    sourcePreviewFallback
      ? {
          ...sourceSnapshot.preview,
          available: true,
          artifactName: sourcePreviewFallback.title,
          sourceArtifactId: sourcePreviewFallback.id,
          sourceArtifactName: sourcePreviewFallback.title,
          summary: `The refined output is saved for review, but its source is not runnable. Preview remains on ${sourcePreviewFallback.title}.`
        }
      : refinedPreviewWasPromoted
      ? {
          ...hydration.snapshot.preview,
          artifactName: refinedArtifact.title,
          outputArtifactName: refinedArtifact.title,
          sourceArtifactId: refinedArtifact.id,
          sourceArtifactName: refinedArtifact.title,
          summary:
            "Preview routing is ready for the refined artifact version. Open the shelf to mount the live preview surface."
        }
      : hydration.snapshot.preview;

  return {
    activeArtifactId: refinedArtifact.id,
    artifact: refinedArtifact,
    previewArtifactId:
      sourcePreviewFallback
        ? sourcePreviewFallback.id
        : refinedPreviewWasPromoted
        ? refinedArtifact.id
        : hydration.previewArtifactId,
    previewAvailable: sourcePreviewFallback
      ? true
      : hydration.previewAvailable,
    snapshot: {
      ...hydration.snapshot,
      artifacts: nextArtifacts,
      code: buildCodeSummaryForArtifact(hydration.snapshot.code, refinedArtifact),
      preview
    }
  };
}

function resolveRefinedPreviewFallbackArtifact({
  artifacts,
  failedArtifactIds,
  failedArtifactTitle
}: {
  artifacts: ArtifactSummary[];
  failedArtifactIds: Array<string | null | undefined>;
  failedArtifactTitle: string;
}) {
  const candidateArtifacts = [
    ...failedArtifactIds
      .filter((artifactId): artifactId is string => Boolean(artifactId))
      .map((artifactId) =>
        artifacts.find((artifact) => artifact.id === artifactId)
      )
      .filter((artifact): artifact is ArtifactSummary => Boolean(artifact)),
    ...artifacts.filter((artifact) => artifact.title === failedArtifactTitle)
  ];
  const failedArtifact =
    candidateArtifacts.find(
      (artifact) =>
        artifact.title === failedArtifactTitle &&
        Boolean(artifact.refinedFromArtifactId)
    ) ??
    candidateArtifacts.find((artifact) =>
      Boolean(artifact.refinedFromArtifactId)
    ) ??
    null;
  const sourceArtifact = failedArtifact?.refinedFromArtifactId
    ? artifacts.find(
        (artifact) => artifact.id === failedArtifact.refinedFromArtifactId
      ) ?? null
    : null;

  return sourceArtifact && isArtifactPreviewable(sourceArtifact)
    ? sourceArtifact
    : null;
}

function createRefinedArtifactId(
  sourceArtifactId: string,
  artifacts: ArtifactSummary[]
) {
  const existingIds = new Set(artifacts.map((artifact) => artifact.id));
  let suffix = 1;
  let candidate = `${sourceArtifactId}-refined`;

  while (existingIds.has(candidate)) {
    suffix += 1;
    candidate = `${sourceArtifactId}-refined-${suffix}`;
  }

  return candidate;
}

function createRefinedArtifactTitle(
  title: string,
  artifacts: ArtifactSummary[]
) {
  const extensionIndex = title.lastIndexOf(".");

  if (extensionIndex <= 0) {
    const stem = title.replace(/-refined(?:-\d+)?$/i, "");
    return createAvailableRefinedTitle({
      artifacts,
      extension: "",
      separator: "-",
      stem
    });
  }

  const stem = title
    .slice(0, extensionIndex)
    .replace(/\.refined(?:-\d+)?$/i, "");
  return createAvailableRefinedTitle({
    artifacts,
    extension: title.slice(extensionIndex),
    separator: ".",
    stem
  });
}

function createAvailableRefinedTitle({
  artifacts,
  extension,
  separator,
  stem
}: {
  artifacts: ArtifactSummary[];
  extension: string;
  separator: "." | "-";
  stem: string;
}) {
  const existingTitles = new Set(artifacts.map((artifact) => artifact.title));
  let version = 1;
  let candidate = `${stem}${separator}refined${extension}`;

  while (existingTitles.has(candidate)) {
    version += 1;
    candidate = `${stem}${separator}refined-${version}${extension}`;
  }

  return candidate;
}

function buildRefinedArtifactSummary(
  artifact: ArtifactSummary,
  refinement: PendingArtifactRefinement
) {
  const sourceSummary = artifact.summary.trim();
  const refinementSummary = `Refined from ${refinement.title}: ${refinement.instruction}`;

  return sourceSummary ? `${sourceSummary} ${refinementSummary}` : refinementSummary;
}

function splitArtifactContentLines(content: string) {
  const lines = content.replace(/\r\n/g, "\n").split("\n");
  return lines.length > 0 ? lines : [""];
}

function createArtifactTransferError(
  action: ArtifactTransferAction,
  artifact: ArtifactSummary
) {
  if (isProjectBundleExportAction(action, artifact)) {
    return createWorkstationError({
      type: "project_bundle_export_failed",
      category: "artifact_export",
      subsystem: "artifact_transfer",
      userMessage: "The workspace could not export the current project bundle.",
      recoverable: true,
      suggestedAction:
        "Retry the bundle export from the Artifacts tab or continue working in the current session.",
      retryLabel: "Retry export"
    });
  }

  const actionLabel = action === "Export" ? "export" : "download";

  return createWorkstationError({
    type: action === "Export" ? "artifact_export_failed" : "artifact_download_failed",
    category: "artifact_export",
    subsystem: "artifact_transfer",
    userMessage: `The workspace could not ${actionLabel} ${artifact.title}.`,
    recoverable: true,
    suggestedAction:
      "Retry the transfer from the Artifacts tab or continue working in the current session.",
    retryLabel: action === "Export" ? "Retry export" : "Retry download"
  });
}

function getArtifactTransferApprovalActionId(
  action: ArtifactTransferAction,
  artifact: ArtifactSummary
): HitlActionId {
  if (isProjectBundleExportAction(action, artifact)) {
    return "project_bundle_export";
  }

  return action === "Download" ? "artifact_download" : "artifact_export";
}

function isProjectBundleExportArtifact(artifact: ArtifactSummary) {
  return artifact.type === "export" && artifact.actions.includes("Export");
}

function isProjectBundleExportAction(
  action: ArtifactTransferAction,
  artifact: ArtifactSummary
) {
  return action === "Export" && isProjectBundleExportArtifact(artifact);
}

function buildHitlApprovalError(request: HitlApprovalRequest) {
  if (request.state !== "failed") {
    return null;
  }

  return createWorkstationError({
    type: `${request.actionId}_failed`,
    category: "hitl_approval",
    subsystem: "operator_checkpoint",
    userMessage:
      request.failureReason ?? `${request.title} could not be completed.`,
    debugMessage: request.detail,
    recoverable: true,
    suggestedAction:
      "Retry the operator action after reviewing the checkpoint details.",
    retryLabel: request.confirmLabel
  });
}

function buildPersistenceTimeoutError(operation: "load" | "save") {
  return createWorkstationError({
    type: operation === "load" ? "session_restore_timed_out" : "session_save_timed_out",
    category: "persistence",
    subsystem: "workspace_session_store",
    userMessage:
      operation === "load"
        ? "No saved session was restored, so the workspace is ready for a fresh start."
        : "Changes are stored locally for now.",
    recoverable: true,
    suggestedAction:
      operation === "load"
        ? "Continue from the current workspace or clear the session if you want to reset it."
        : "Keep editing; the workspace can save again when the connection is available.",
    retryLabel: operation === "save" ? "Retry save" : null,
    resetLabel: operation === "load" ? "Clear workspace session" : null
  });
}

function createProviderFallbackInProgressOutcome() {
  return {
    orchestration_status: "FALLBACK",
    provider_status: "FALLBACK",
    generation_status: "PENDING",
    deliverable_status: "UNKNOWN",
    artifact_extraction_status: "UNKNOWN",
    artifact_runnability: "UNKNOWN",
    preview_status: "UNKNOWN",
    runtime_health: "UNKNOWN",
    product_outcome: "IN_PROGRESS" as const,
    summary: "A local fallback is being prepared after the provider became unavailable.",
    recovery_action: ""
  };
}

type WorkflowProgressSummary = {
  percent: number;
  reached: number;
  total: number;
};

type WorkflowProgressStep = {
  state: WorkflowRuntimeVisualState | WorkflowState;
};

function WorkflowProgress({
  label,
  progress
}: {
  label: string;
  progress: WorkflowProgressSummary;
}) {
  return (
    <div
      aria-label={label}
      aria-valuemax={progress.total}
      aria-valuemin={0}
      aria-valuenow={progress.reached}
      aria-valuetext={`${progress.reached} of ${progress.total} workflow nodes reached`}
      className="workflowProgress"
      role="progressbar"
    >
      <span style={{ width: `${progress.percent}%` }} />
    </div>
  );
}

function getInitialActiveTab(snapshot: AssistantWorkspaceSnapshot): ProductIntelligenceCategory {
  return snapshot.inspectorTabs.find((tab) => tab.active)?.label ?? "Overview";
}

function toSnapshotInspectorTab(
  tab: ProductIntelligenceCategory
): InspectorTabName {
  return isSnapshotInspectorTab(tab) ? tab : "Overview";
}

function isSnapshotInspectorTab(
  tab: ProductIntelligenceCategory
): tab is InspectorTabName {
  return [
    "Overview",
    "Preview",
    "Runtime",
    "Code",
    "Workflow",
    "Telemetry",
    "Artifacts",
    "Retrieval"
  ].includes(tab);
}

function isEstablishedInspectorTab(
  tab: ProductIntelligenceCategory
): tab is InspectorTabName {
  return isSnapshotInspectorTab(tab);
}

function getInspectorPanelId(tab: ProductIntelligenceCategory) {
  return `${tab.toLowerCase().replace(/\s+/g, "-")}-inspector-panel`;
}

function formatInspectorTabDisplayLabel(
  tab: ProductIntelligenceCategory,
  showDebugPanels: boolean
) {
  if (!showDebugPanels && tab === "Artifacts") {
    return "Saved";
  }

  return tab;
}

function getInitialPreviewArtifactId(snapshot: AssistantWorkspaceSnapshot): string {
  return (
    snapshot.artifacts.find(
      (artifact) => artifact.title === snapshot.preview.artifactName
    )?.id ??
    snapshot.artifacts[0]?.id ??
    ""
  );
}

function resolveRestoredArtifactId(
  requestedArtifactId: string,
  artifacts: ArtifactSummary[],
  fallbackArtifactId: string
) {
  return artifacts.some((artifact) => artifact.id === requestedArtifactId)
    ? requestedArtifactId
    : fallbackArtifactId;
}

function getInitialWorkflowIndex(steps: WorkflowStepState[]) {
  const activeIndex = steps.findIndex((step) => step.state === "active");

  return activeIndex >= 0 ? activeIndex : 0;
}

function shouldIgnoreRestoredWorkspaceSession(
  initialSnapshot: AssistantWorkspaceSnapshot,
  restoredSnapshot: AssistantWorkspaceSnapshot
) {
  return (
    isFirstRunSnapshot(initialSnapshot) &&
    isSeededDemoWorkspaceSnapshot(restoredSnapshot)
  );
}

function isFirstRunSnapshot(snapshot: AssistantWorkspaceSnapshot) {
  return (
    snapshot.messages.length === 0 &&
    snapshot.artifacts.length === 0 &&
    !snapshot.preview.available
  );
}

function isSeededDemoWorkspaceSnapshot(snapshot: AssistantWorkspaceSnapshot) {
  return (
    snapshot.workspace.focus === "p5 aurora field" ||
    snapshot.artifacts.some((artifact) => artifact.title === "aurora-field.p5.js") ||
    snapshot.messages.some((message) =>
      message.content.includes("luminous particle field")
    )
  );
}

function getWorkflowNodeIndex(
  steps: WorkflowStepState[],
  nodeId: WorkflowStepState["nodeId"]
) {
  const nodeIndex = steps.findIndex((step) => step.nodeId === nodeId);

  return nodeIndex >= 0 ? nodeIndex : Math.max(steps.length - 1, 0);
}

function buildInteractiveWorkflow(
  workflow: WorkspaceWorkflow,
  progressIndex: number,
  options: { hasActiveRun?: boolean } = {}
): WorkspaceWorkflow {
  if (!options.hasActiveRun && isWorkflowIdleStatus(workflow.status)) {
    return {
      ...workflow,
      status: "Idle",
      steps: workflow.steps.map((step) =>
        step.nodeId === "failure"
          ? { ...step, state: "branch" as WorkflowState }
          : { ...step, state: "queued" as WorkflowState }
      )
    };
  }

  const finalizationIndex = getWorkflowNodeIndex(workflow.steps, "finalization");
  const boundedProgressIndex = Math.min(
    Math.max(progressIndex, 0),
    finalizationIndex
  );
  const steps = workflow.steps.map((step, index) => {
    if (step.nodeId === "failure") {
      return { ...step, state: "branch" as WorkflowState };
    }

    if (index < boundedProgressIndex) {
      return {
        ...step,
        state: step.state === "skipped" ? "skipped" : ("complete" as WorkflowState)
      };
    }

    if (index === boundedProgressIndex) {
      return { ...step, state: "active" as WorkflowState };
    }

    return { ...step, state: "queued" as WorkflowState };
  });
  const currentStep =
    steps.find((step) => step.state === "active") ?? steps[boundedProgressIndex];

  return {
    ...workflow,
    currentNode: currentStep.nodeId,
    currentStep: currentStep.displayLabel,
    status: boundedProgressIndex >= finalizationIndex ? "Success" : "Running",
    steps
  };
}

function isWorkflowIdleStatus(status: string) {
  return ["idle", "ready"].includes(status.toLowerCase());
}

function getWorkflowRuntimeProgress(
  steps: WorkflowProgressStep[]
): WorkflowProgressSummary {
  const visibleSteps = steps.filter((step) => step.state !== "branch");
  const reached = visibleSteps.filter((step) =>
    ["active", "complete", "skipped", "failed"].includes(step.state)
  ).length;
  const total = Math.max(visibleSteps.length, 1);

  return {
    percent: Math.round((reached / total) * 100),
    reached,
    total
  };
}

function formatWorkflowRuntimeState(state: WorkflowRuntimeVisualState) {
  switch (state) {
    case "queued":
      return "pending";
    case "failed":
      return "error";
    default:
      return state;
  }
}

function formatWorkflowStatusCopy(status: string) {
  switch (status) {
    case "idle":
    case "ready":
      return "Ready";
    case "complete":
    case "completed":
      return "Success";
    case "completed_with_preview_error":
      return "Partial";
    case "partial":
      return "Partial";
    case "failed":
      return "Failure";
    default:
      return "Running";
  }
}

function formatRetryCount(retryCount: number) {
  return retryCount === 1 ? "1 retry loop" : `${retryCount} retry loops`;
}

function formatClarificationReason(reason: string) {
  return reason
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function formatConfidenceLabel(confidence: number) {
  return `${Math.round(confidence * 100)}%`;
}

function buildClarificationContinuationPrompt(
  clarification: ClarificationSummary,
  answer: string
) {
  return `${clarification.originalQuery}\n\nClarification answer: ${answer}`;
}

function resolveClarificationNumericAnswer(
  clarification: ClarificationSummary | null,
  answer: string
) {
  const normalized = answer.trim();
  if (!clarification || !/^[1-9]$/.test(normalized)) {
    return null;
  }

  const options = clarification.questions.flatMap(
    (question) => question.suggestedOptions
  );
  const optionIndex = Number.parseInt(normalized, 10) - 1;

  return options[optionIndex] ?? null;
}

function formatSessionTelemetryLabel(
  telemetry: ProviderTelemetryModel,
  usage: SessionUsageSummary | null
) {
  if (usage) {
    return `Total ${formatUsageTotalLabel(usage)}`;
  }
  const hasTokenUsage = telemetry.tokenUsage.totalTokens != null;
  const hasCostEstimate = telemetry.cost.totalCost != null;

  if (!hasTokenUsage && !hasCostEstimate && telemetry.status === "idle") {
    return "Tokens and estimated cost pending";
  }

  return `${telemetry.summary.tokenLabel} · ${telemetry.summary.costLabel}`;
}

function formatLatestUsageLabel(
  telemetry: ProviderTelemetryModel,
  usage: SessionUsageSummary | null
) {
  const tokens = telemetry.tokenUsage.totalTokens ?? usage?.latestTokens ?? null;
  const cost = telemetry.cost.totalCost ?? usage?.latestCost ?? null;
  return `${formatUsageTokens(tokens)} · ${formatUsageCost(cost)}`;
}

function formatUsageTotalLabel(usage: SessionUsageSummary | null) {
  return `${formatUsageTokens(usage?.totalTokens ?? null)} · ${formatUsageCost(
    usage?.totalCost ?? null
  )}`;
}

function formatUsageTokens(value: number | null) {
  return value == null ? "Tokens not reported" : `${formatCompactNumber(value)} tokens`;
}

function formatUsageCost(value: number | null) {
  return value == null ? "Cost not reported" : `$${value.toFixed(4)}`;
}

function formatWorkflowModeLabel(mode: WorkflowExecutionMode) {
  return mode === "single_agent"
    ? "Single Agent workflow"
    : mode === "multi_agent"
      ? "Multi Agent workflow"
      : "Auto workflow";
}

function formatUserModeSessionStatus({
  activity,
  hasFailedPreviewRuntime,
  hasWorkspaceArtifacts,
  isDemoModeOpen,
  productOutcome,
  streamError,
  streamState
}: {
  activity: WorkflowRuntimeActivity | null;
  hasFailedPreviewRuntime: boolean;
  hasWorkspaceArtifacts: boolean;
  isDemoModeOpen: boolean;
  productOutcome: WorkflowRuntimeModel["summary"]["productOutcome"];
  streamError: WorkstationError | null;
  streamState: string;
}) {
  switch (streamState) {
    case "approval":
      return {
        label: "Needs review",
        detail: "Action required"
      };
    case "executing":
    case "streaming":
      return {
        label: activity?.label ?? "Planning",
        detail: activity?.detail ?? "Planning the requested work."
      };
    case "fallback":
      return {
        label: "Needs attention",
        detail: streamError ? "Live response unavailable" : "Fallback available"
      };
    default:
      {
        const semanticStatus = formatSemanticProductOutcomeStatus(productOutcome);
        if (semanticStatus) {
          return semanticStatus;
        }
      }
      if (hasFailedPreviewRuntime) {
        return {
          label: "Needs attention",
          detail: "Preview needs attention"
        };
      }

      if (hasWorkspaceArtifacts) {
        return {
          label: "Success",
          detail: "Output ready"
        };
      }

      if (isDemoModeOpen) {
        return {
          label: "Demo ready",
          detail: "Choose a scenario"
        };
      }

      return {
        label: "Ready",
        detail: "Start a prompt"
      };
  }
}

function formatSemanticProductOutcomeStatus(
  productOutcome: WorkflowRuntimeModel["summary"]["productOutcome"]
) {
  switch (productOutcome.product_outcome) {
    case "SUCCESS":
      return {
        label: "Success",
        detail: productOutcome.summary
      };
    case "PARTIAL":
      return {
        label: "Partial",
        detail: productOutcome.summary
      };
    case "FAILURE":
      return {
        label: "Failure",
        detail: productOutcome.summary
      };
    default:
      return null;
  }
}

function deriveWorkflowRuntimeActivityForStreamEvent(
  streamEvent: AssistantStreamEvent
): WorkflowRuntimeActivity {
  const workflow = readWorkflowMetadata(streamEvent);
  const workflowStatus =
    workflow?.status ??
    (streamEvent.event_type === "error"
      ? "failed"
      : streamEvent.event_type === "final"
        ? "completed"
        : "running");

  return deriveWorkflowRuntimeActivity({
    currentNode:
      workflow?.current_step ??
      workflow?.step ??
      workflowNodeFromAssistantStreamEvent(streamEvent) ??
      null,
    productOutcome: workflow?.product_outcome,
    workflowStatus
  });
}

function formatConversationOutcome(activity: WorkflowRuntimeActivity) {
  return {
    activity: activity.detail,
    phase: terminalConversationPhaseForWorkflowActivity(activity)
  };
}

function conversationPhaseForWorkflowActivity(
  activity: WorkflowRuntimeActivity
): Extract<
  ConversationEntryPhase,
  | "planning"
  | "retrieving"
  | "generating"
  | "reviewing"
  | "refining"
  | "finalizing"
  | "completed"
  | "partial"
  | "failed"
> {
  return activity.state;
}

function conversationPhaseForRequestActivity(
  activity: WorkflowRuntimeActivity,
  mode: AssistantRequestMode
): ConversationEntryPhase {
  if (
    mode === "explain" &&
    !activity.terminal &&
    activity.state !== "retrieving"
  ) {
    return "answering";
  }
  return conversationPhaseForWorkflowActivity(activity);
}

function conversationActivityForRequestMode(
  activity: WorkflowRuntimeActivity,
  mode: AssistantRequestMode
) {
  if (mode !== "explain" || activity.terminal) {
    return activity.detail;
  }
  switch (activity.state) {
    case "retrieving":
      return "Retrieving context for the answer.";
    case "finalizing":
      return "Finalizing the answer.";
    case "reviewing":
      return "Checking the answer.";
    default:
      return "Writing the answer.";
  }
}

function terminalConversationPhaseForWorkflowActivity(
  activity: WorkflowRuntimeActivity
): Extract<
  ConversationEntryPhase,
  "completed" | "partial" | "failed"
> {
  const phase = conversationPhaseForWorkflowActivity(activity);
  return phase === "partial" || phase === "failed" ? phase : "completed";
}

function formatTokenUsageTotal(telemetry: ProviderTelemetryModel) {
  return telemetry.tokenUsage.totalTokens == null
    ? "Pending"
    : formatCompactNumber(telemetry.tokenUsage.totalTokens);
}

function formatTokenUsageDetail(telemetry: ProviderTelemetryModel) {
  if (telemetry.tokenUsage.source !== "provider") {
    return telemetry.stream.streamedCharacterCount > 0
      ? `${telemetry.stream.streamedCharacterCount} streamed chars; provider usage pending`
      : "Provider usage pending";
  }

  const parts = [];
  if (telemetry.tokenUsage.inputTokens != null) {
    parts.push(`${formatCompactNumber(telemetry.tokenUsage.inputTokens)} input`);
  }
  if (telemetry.tokenUsage.outputTokens != null) {
    parts.push(`${formatCompactNumber(telemetry.tokenUsage.outputTokens)} output`);
  }
  if (telemetry.tokenUsage.reasoningTokens != null) {
    parts.push(`${formatCompactNumber(telemetry.tokenUsage.reasoningTokens)} reasoning`);
  }

  return parts.join(" / ") || "Provider usage captured";
}

function formatTelemetryCostSource(telemetry: ProviderTelemetryModel) {
  switch (telemetry.cost.source) {
    case "provider_reported":
      return "Provider reported";
    case "pricing_metadata":
      return "Estimated from pricing metadata";
    default:
      return "Awaiting usage and pricing metadata";
  }
}

function formatProviderRuntimeLabel(telemetry: ProviderTelemetryModel) {
  return `${telemetry.summary.providerLabel} / ${telemetry.summary.modelLabel}`;
}

function formatTelemetryStatus(status: ProviderTelemetryModel["status"]) {
  switch (status) {
    case "complete":
      return "Success";
    case "error":
      return "Failure";
    case "streaming":
      return "Generating";
    default:
      return "Idle";
  }
}

function formatDashboardStatusLabel(status: TelemetryDashboardModel["status"]) {
  switch (status) {
    case "complete":
      return "Success";
    case "degraded":
      return "Partial";
    case "error":
      return "Failure";
    case "running":
      return "Generating";
    default:
      return "Idle";
  }
}

function formatTelemetryLifecycleStep(step: ProviderTelemetryLifecycleStep) {
  if (!step.at) {
    return step.state === "active" ? "Active" : "Pending";
  }

  const offset =
    step.offsetMs == null ? "" : ` / +${formatRuntimeDuration(step.offsetMs)}`;
  return `${formatTraceTime(step.at)}${offset}`;
}

function formatCompactNumber(value: number) {
  return new Intl.NumberFormat("en", {
    maximumFractionDigits: 0
  }).format(value);
}

function formatRuntimeDuration(durationMs: number | null) {
  if (durationMs == null) {
    return "No timing yet";
  }

  if (durationMs < 1000) {
    return `${durationMs}ms`;
  }

  return `${(durationMs / 1000).toFixed(durationMs >= 10000 ? 0 : 1)}s`;
}

function formatAttemptMeta(attemptCount: number) {
  return attemptCount === 1 ? "1 attempt" : `${attemptCount} attempts`;
}

function pluralize(count: number, singular: string, plural: string) {
  return count === 1 ? singular : plural;
}

function formatTraceTime(timestamp: string) {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) {
    return "Now";
  }

  return new Intl.DateTimeFormat("en-US", {
    hour: "2-digit",
    hour12: false,
    minute: "2-digit",
    second: "2-digit"
  }).format(date);
}

function formatNullableTraceTime(timestamp: string | null) {
  return timestamp ? formatTraceTime(timestamp) : "Not captured";
}

function formatWorkflowMiniMeta(step: WorkflowRuntimeModel["steps"][number]) {
  const meta = [];

  if (step.durationMs != null) {
    meta.push(formatRuntimeDuration(step.durationMs));
  }

  if (step.attemptCount > 1) {
    meta.push(`attempt ${step.attemptCount}`);
  }

  if (step.state === "failed") {
    meta.push("error");
  }

  return meta.join(" / ") || step.lastEventLabel || step.detail;
}

function formatRuntimeCode(value: string) {
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function readPayloadText(
  event: AssistantStreamEvent,
  key: string
): string | undefined {
  const value = event.payload[key];
  return typeof value === "string" ? value : undefined;
}

type EvaluationApiSnapshot = {
  runId: string | null;
  status: string;
  progress: EvaluationExecutionProgress;
  result: Record<string, unknown> | null;
};

class EvaluationPollingTerminalError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "EvaluationPollingTerminalError";
  }
}

function parseEvaluationApiSnapshot(
  record: Record<string, unknown>,
  fallback: {
    request: EvaluationRunRequest;
    runId?: string | null;
    selectedContracts: number;
  }
): EvaluationApiSnapshot {
  const nestedResult = isUnknownRecord(record.result) ? record.result : null;
  const progressRecord = isUnknownRecord(record.progress) ? record.progress : {};
  const runId = typeof record.runId === "string"
    ? record.runId
    : typeof nestedResult?.runId === "string"
      ? nestedResult.runId
      : fallback.runId ?? null;
  const status = normalizeEvaluationStatus(
    typeof record.status === "string"
      ? record.status
      : typeof nestedResult?.status === "string"
        ? nestedResult.status
        : "running"
  );
  const result = nestedResult ?? (isTerminalEvaluationStatus(status) ? record : null);
  const totalCases = finiteNumber(progressRecord.totalCases) ?? fallback.selectedContracts;
  const completedCases = Math.min(
    totalCases,
    Math.max(0, finiteNumber(progressRecord.completedCases) ?? 0)
  );
  const reportedRemaining = finiteNumber(progressRecord.remainingCases);
  const percent = finiteNumber(progressRecord.percent);
  const defaultLane = fallback.request.scope === "full"
    ? "Full evaluation"
    : fallback.request.scope.replace(/_/g, " ");

  return {
    runId,
    status,
    result,
    progress: {
      runId,
      status,
      phase: typeof progressRecord.phase === "string"
        ? progressRecord.phase
        : isTerminalEvaluationStatus(status) ? "terminal" : "running",
      lane: typeof progressRecord.lane === "string" && progressRecord.lane.trim()
        ? progressRecord.lane
        : defaultLane,
      currentCaseId: typeof progressRecord.currentCaseId === "string"
        ? progressRecord.currentCaseId
        : null,
      currentCaseLabel: typeof progressRecord.currentCaseLabel === "string" && progressRecord.currentCaseLabel.trim()
        ? progressRecord.currentCaseLabel
        : isTerminalEvaluationStatus(status) ? "Evaluation complete" : "Waiting for the next case",
      completedCases,
      totalCases,
      remainingCases: Math.max(0, reportedRemaining ?? totalCases - completedCases),
      percent: percent == null ? null : Math.min(100, Math.max(0, percent)),
      executionState: typeof progressRecord.executionState === "string"
        ? progressRecord.executionState
        : fallback.request.allowProviderCalls ? "provider_authorized" : "local_preflight",
      detail: typeof progressRecord.detail === "string" && progressRecord.detail.trim()
        ? progressRecord.detail
        : isTerminalEvaluationStatus(status)
          ? `Evaluation ended with status ${status}.`
          : "Evaluation progress is being refreshed automatically."
    }
  };
}

function normalizeEvaluationStatus(value: string) {
  const normalized = value.trim().toLowerCase().replace(/[\s-]+/g, "_");
  if (["evaluation_completed", "complete", "completed", "succeeded", "success"].includes(normalized)) return "completed";
  if (["evaluation_prepared", "prepared", "dry_run_complete"].includes(normalized)) return "prepared";
  if (["blocked", "blocked_by_execution_environment"].includes(normalized)) return "blocked";
  if (["failed", "failure", "error", "evaluation_failed", "cancelled", "canceled"].includes(normalized)) return "failed";
  if (["queued", "pending", "preflight"].includes(normalized)) return normalized;
  return "running";
}

function isTerminalEvaluationStatus(value: string) {
  return ["completed", "prepared", "blocked", "failed"].includes(
    normalizeEvaluationStatus(value)
  );
}

function evaluationStatusUrl(runId: string) {
  const separator = evaluationRunEndpoint.includes("?") ? "&" : "?";
  return `${evaluationRunEndpoint}${separator}runId=${encodeURIComponent(runId)}`;
}

function waitForEvaluationPoll(delayMs: number, signal: AbortSignal) {
  return new Promise<void>((resolve) => {
    if (signal.aborted) {
      resolve();
      return;
    }
    const finish = () => {
      window.clearTimeout(timeoutId);
      signal.removeEventListener("abort", finish);
      resolve();
    };
    const timeoutId = window.setTimeout(finish, delayMs);
    signal.addEventListener("abort", finish, { once: true });
  });
}

async function fetchEvaluationPoll(runId: string, signal: AbortSignal) {
  return fetch(evaluationStatusUrl(runId), {
    cache: "no-store",
    headers: { Accept: "application/json" },
    method: "GET",
    signal
  });
}

function evaluationReconnectDelayMs(consecutiveFailures: number) {
  const exponent = Math.min(Math.max(0, consecutiveFailures - 1), 8);
  return Math.min(
    evaluationReconnectMaxDelayMs,
    evaluationReconnectBaseDelayMs * 2 ** exponent
  );
}

function formatEvaluationReconnectDelay(delayMs: number) {
  return delayMs < 1_000
    ? "in under a second"
    : `in ${Math.ceil(delayMs / 1_000)} seconds`;
}

function normalizeVisibleEvaluationProgress(
  progress: EvaluationExecutionProgress,
  scope: EvaluationRunRequest["scope"]
): EvaluationExecutionProgress {
  if (scope !== "full") {
    return progress;
  }
  const totalCases = CURRENT_PRODUCT_RETRIEVAL_CASE_IDS.length;
  const completedCases = Math.min(totalCases, Math.max(0, progress.completedCases));
  const percent = progress.percent == null
    ? null
    : Math.min(100, Math.max(0, progress.percent));
  return {
    ...progress,
    lane: "Full evaluation",
    completedCases,
    totalCases,
    remainingCases: Math.max(0, totalCases - completedCases),
    percent,
    detail: `${progress.detail} Full evaluation covers seven canonical RAG cases plus current local workspace snapshots. Numeric case progress tracks only the seven canonical RAG cases; local snapshots are not counted as generated or evaluated contracts.`
  };
}

function finiteNumber(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function parseRagasExecutionEvidence(
  record: Record<string, unknown>,
  _request: EvaluationRunRequest,
  terminalStatus = "completed"
): RagasExecutionEvidence {
  const canonicalCurrentProductPayload =
    record.benchmarkMode === "current_product" && record.scoreOrigin === "current_product";
  const metricScores = canonicalCurrentProductPayload
    ? strictNumberRecord(record.metricScores)
    : numberRecord(record.metricScores);
  const rawCaseResults = Array.isArray(record.caseResults) ? record.caseResults : [];
  const parsedCaseRows = rawCaseResults.flatMap((value) => {
        if (!isUnknownRecord(value)) return [];
        const sampleId = typeof value.sampleId === "string"
          ? value.sampleId
          : typeof value.caseId === "string" ? value.caseId : null;
        if (!sampleId) return [];
        const retrievedContexts = Array.isArray(value.retrievedContexts)
          ? value.retrievedContexts.filter(isUnknownRecord)
          : [];
        const metricErrors = canonicalCurrentProductPayload
          ? strictMetricErrorRecord(value.metricErrors)
          : stringRecord(value.metricErrors);
        const directSourceIds = canonicalCurrentProductPayload
          ? strictStringArray(value.sourceIds)
          : stringArray(value.sourceIds);
        const directDomains = canonicalCurrentProductPayload
          ? strictStringArray(value.domains)
          : stringArray(value.domains);
        return [{
          sampleId,
          metrics: canonicalCurrentProductPayload
            ? strictNullableNumberRecord(value.metrics)
            : nullableNumberRecord(value.metrics),
          metricErrors,
          sourceIds: canonicalCurrentProductPayload
            ? directSourceIds
            : directSourceIds.length
              ? directSourceIds
              : retrievedContexts.flatMap((context) => {
                  const sourceId = typeof context.sourceId === "string"
                    ? context.sourceId
                    : typeof context.source_id === "string" ? context.source_id : null;
                  return sourceId ? [sourceId] : [];
                }),
          domains: canonicalCurrentProductPayload
            ? directDomains
            : directDomains.length
              ? directDomains
              : stringArray(value.expectedDomains),
          promptFingerprint: typeof value.promptFingerprint === "string" ? value.promptFingerprint : null,
          generationFingerprint: typeof value.generationFingerprint === "string" ? value.generationFingerprint : null
        }];
      });
  const caseRows = canonicalCurrentProductPayload && (
    !Array.isArray(record.caseResults) || parsedCaseRows.length !== rawCaseResults.length
  ) ? [] : parsedCaseRows;
  const state = terminalStatus === "prepared"
    ? "prepared"
    : terminalStatus === "blocked"
      ? "blocked"
      : terminalStatus === "failed" ? "failed" : "completed";
  const benchmarkMode = normalizeEvaluationBenchmarkMode(record.benchmarkMode);
  const historicalFixtureMetricIds = ["context_precision", "faithfulness", "answer_relevancy", "context_relevancy"];
  const historicalMetricSetComplete = historicalFixtureMetricIds.every((metricId) => metricScores[metricId] != null);
  const declaredScoreOrigin = record.scoreOrigin === "current_product"
    ? "current_product"
    : record.scoreOrigin === "historical_fixture" ? "historical_fixture" : "unscored";
  const provider = typeof record.provider === "string" ? record.provider : null;
  const model = typeof record.model === "string" ? record.model : null;
  const evaluatedAt = typeof record.evaluatedAt === "string" ? record.evaluatedAt : null;
  const evidence: RagasExecutionEvidence = {
    schemaVersion: typeof record.schemaVersion === "string" ? record.schemaVersion : null,
    scope: typeof record.scope === "string" ? record.scope : null,
    state,
    runId: typeof record.runId === "string" ? record.runId : null,
    evaluatedAt,
    datasetId: typeof record.datasetId === "string" ? record.datasetId : "current_product",
    datasetVersion: typeof record.datasetVersion === "string"
      ? record.datasetVersion
      : canonicalCurrentProductPayload
        ? ""
        : typeof record.benchmarkVersion === "string" ? record.benchmarkVersion : "current-product.v1",
    privacyClass: typeof record.privacyClass === "string" ? record.privacyClass : "current_product_local",
    metrics: canonicalCurrentProductPayload
      ? strictStringArray(record.metrics)
      : Array.isArray(record.metrics)
        ? record.metrics.filter((value): value is string => typeof value === "string")
        : Object.keys(metricScores),
    metricScores,
    retrievalScore: typeof record.retrievalScore === "number" ? record.retrievalScore : null,
    resultRows: numberValue(record.resultRows),
    totalSamples: numberValue(record.totalSamples),
    eligibleSamples: numberValue(record.eligibleSamples),
    skippedSamples: numberValue(record.skippedSamples),
    metricFailures: numberValue(record.metricFailures),
    provider,
    model,
    embeddingModel: typeof record.embeddingModel === "string" ? record.embeddingModel : null,
    ragasVersion: typeof record.ragasVersion === "string" ? record.ragasVersion : null,
    metricContract: typeof record.metricContract === "string" ? record.metricContract : null,
    durationMs: typeof record.durationMs === "number" ? record.durationMs : null,
    detail: typeof record.detail === "string"
      ? record.detail
      : canonicalCurrentProductPayload ? "" : "Current-product evaluation evidence recorded.",
    caseRows,
    benchmarkMode,
    scoreOrigin: declaredScoreOrigin,
    benchmarkVersion: typeof record.benchmarkVersion === "string" ? record.benchmarkVersion : null,
    selectedCaseIds: canonicalCurrentProductPayload
      ? strictStringArray(record.selectedCaseIds)
      : stringArray(record.selectedCaseIds),
    datasetFingerprint: typeof record.datasetFingerprint === "string" ? record.datasetFingerprint : null,
    retrievalFingerprint: typeof record.retrievalFingerprint === "string" ? record.retrievalFingerprint : null,
    promptFingerprint: typeof record.promptFingerprint === "string" ? record.promptFingerprint : null,
    generationFingerprint: typeof record.generationFingerprint === "string" ? record.generationFingerprint : null,
    outputFingerprint: typeof record.outputFingerprint === "string" ? record.outputFingerprint : null,
    selectionFingerprint: typeof record.selectionFingerprint === "string" ? record.selectionFingerprint : null,
    kbFingerprint: typeof record.kbFingerprint === "string" ? record.kbFingerprint : null,
    generationModel: typeof record.generationModel === "string"
      ? record.generationModel
      : canonicalCurrentProductPayload ? null : model,
    evaluator: typeof record.evaluator === "string"
      ? record.evaluator
      : canonicalCurrentProductPayload
        ? null
        : [provider, model].filter(Boolean).join(" / ") || null,
    evaluatorModel: typeof record.evaluatorModel === "string" ? record.evaluatorModel : null,
    timestamp: typeof record.timestamp === "string"
      ? record.timestamp
      : canonicalCurrentProductPayload ? null : evaluatedAt
  };
  if (
    declaredScoreOrigin === "current_product" &&
    currentProductRetrievalScoreFromEvidence(evidence) == null
  ) {
    return { ...evidence, scoreOrigin: "unscored" };
  }
  if (
    declaredScoreOrigin === "historical_fixture" &&
    !(benchmarkMode === "historical_fixture" && historicalMetricSetComplete)
  ) {
    return { ...evidence, scoreOrigin: "unscored" };
  }
  return evidence;
}

function unscoredCurrentProductEvidence(
  record: Record<string, unknown>,
  request: EvaluationRunRequest,
  terminalStatus: string
) {
  return {
    ...parseRagasExecutionEvidence(record, request, terminalStatus),
    scoreOrigin: "unscored" as const
  };
}

function localWorkspaceLaneEvidence(
  runId: string,
  evaluatedAt: string
): RagasExecutionEvidence {
  return {
    ...emptyRagasEvidence(),
    state: "completed",
    runId,
    evaluatedAt,
    datasetId: "not_selected",
    datasetVersion: "local-workspace-snapshot.v1",
    privacyClass: "local_workspace_only",
    metrics: [],
    detail: "Selected local-only workspace lanes were inspected from current evidence. No retrieval, generation, or evaluator provider calls were made, and no Retrieval Quality score was published.",
    benchmarkMode: "not_selected",
    scoreOrigin: "unscored",
    timestamp: evaluatedAt
  };
}

function blockedRagasEvidence(
  request: EvaluationRunRequest,
  record: Record<string, unknown>
): RagasExecutionEvidence {
  return {
    ...parseRagasExecutionEvidence(record, request, "blocked"),
    state: "blocked",
    scoreOrigin: "unscored",
    detail: typeof record.message === "string"
      ? record.message
      : "BLOCKED_BY_EXECUTION_ENVIRONMENT: provider evaluation was unavailable."
  };
}

function failedRagasEvidence(
  request: EvaluationRunRequest,
  record: Record<string, unknown>
): RagasExecutionEvidence {
  return {
    ...parseRagasExecutionEvidence(record, request, "failed"),
    state: "failed",
    scoreOrigin: "unscored",
    detail: typeof record.message === "string"
      ? record.message
      : "MISSING_EVIDENCE: the evaluator returned no defensible result."
  };
}

function isUnknownRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function numberValue(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function numberRecord(value: unknown) {
  if (!isUnknownRecord(value)) return {};
  return Object.fromEntries(Object.entries(value).filter((entry): entry is [string, number] => typeof entry[1] === "number"));
}

function strictNumberRecord(value: unknown) {
  if (!isUnknownRecord(value)) return {};
  return Object.fromEntries(Object.entries(value).map(([key, item]) => [
    key,
    typeof item === "number" ? item : Number.NaN
  ]));
}

function nullableNumberRecord(value: unknown) {
  if (!isUnknownRecord(value)) return {};
  return Object.fromEntries(Object.entries(value).filter((entry): entry is [string, number | null] => entry[1] === null || typeof entry[1] === "number"));
}

function strictNullableNumberRecord(value: unknown) {
  if (!isUnknownRecord(value)) return {};
  return Object.fromEntries(Object.entries(value).map(([key, item]) => [
    key,
    item === null || typeof item === "number" ? item : null
  ]));
}

function stringRecord(value: unknown) {
  if (!isUnknownRecord(value)) return {};
  return Object.fromEntries(Object.entries(value).filter((entry): entry is [string, string] => typeof entry[1] === "string"));
}

function strictMetricErrorRecord(value: unknown) {
  if (!isUnknownRecord(value)) {
    return { __invalid_metric_errors__: "metricErrors must be an object" };
  }
  return Object.fromEntries(Object.entries(value).map(([key, item]) => [
    key,
    typeof item === "string" ? item : "metricErrors contained a non-string value"
  ]));
}

function stringArray(value: unknown) {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

function strictStringArray(value: unknown) {
  return Array.isArray(value) && value.every((item): item is string => typeof item === "string")
    ? value
    : [];
}

function formatMessageTime() {
  return new Intl.DateTimeFormat("en-US", {
    hour: "2-digit",
    hour12: false,
    minute: "2-digit"
  }).format(new Date());
}

function streamingConversationSummaryForMode(mode: AssistantRequestMode) {
  return mode === "explain"
    ? "Answering your question. The complete response will appear here when it is ready."
    : "Generating the requested artifact. Code and long-form output will appear in the Code panel, artifacts, and preview surfaces when the run completes.";
}

const generatedCodePattern =
  /```|<!doctype|<html|<script|function\s+(setup|draw)\s*\(|import\s+\*\s+as\s+THREE|gl_FragColor|void\s+main\s*\(/i;

function getConversationDisplayContent(
  message: ConversationEntry,
  showDebugPanels: boolean
) {
  if (
    message.role !== "assistant" ||
    showDebugPanels ||
    message.requestMode === "explain"
  ) {
    return message.content;
  }

  return buildUserModeAssistantSummary(message.content);
}

function buildUserModeAssistantSummary(content: string) {
  const trimmedContent = content.trim();

  if (!trimmedContent) {
    return "";
  }

  if (/^live response (error|unavailable)/i.test(trimmedContent)) {
    return "The live response could not complete. Retry from the composer, switch to prepared demo evidence, or inspect details in Developer Mode.";
  }

  const containsGeneratedCode = generatedCodePattern.test(trimmedContent);
  const strippedContent = stripGeneratedCodeFromConversation(trimmedContent)
    .replace(/<[^>]+>/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  if (containsGeneratedCode) {
    const summary = strippedContent
      ? truncateConversationSummary(strippedContent, 220)
      : "Generated code is ready.";

    return `${summary}\n\nCode and long-form output are in Code, Artifacts, and Preview.`;
  }

  return strippedContent;
}

function buildAssistantConversationSummary(
  answer: string,
  mode: AssistantRequestMode = "generate"
) {
  const trimmedAnswer = answer.trim();

  if (!trimmedAnswer) {
    return mode === "explain"
      ? "The response completed without any answer text. Please retry the question."
      : "Response completed. Check Code, Preview, and Retrieval for output details.";
  }

  if (mode === "explain") {
    return trimmedAnswer;
  }

  const lineCount = trimmedAnswer.split(/\r?\n/).length;
  const shouldSummarize =
    generatedCodePattern.test(trimmedAnswer) ||
    trimmedAnswer.length > 900 ||
    lineCount > 14;

  if (!shouldSummarize) {
    return trimmedAnswer;
  }

  const textOnly = stripGeneratedCodeFromConversation(trimmedAnswer);
  const summary =
    textOnly
      .split(/\r?\n+/)
      .map((line) => line.replace(/\s+/g, " ").trim())
      .find(
        (line) =>
          line.length > 0 &&
          !/^file(name)?:/i.test(line) &&
          !/^```/.test(line)
      ) ?? "Your requested creative-coding output is ready.";

  return `${truncateConversationSummary(
    summary,
    240
  )}\n\nCode and long-form output are available in the Code panel, artifacts, and preview surfaces. Next: inspect Preview, Code, and Retrieval evidence.`;
}

function stripGeneratedCodeFromConversation(value: string) {
  return value
    .replace(/```[\s\S]*?(?:```|$)/g, " ")
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ");
}

function truncateConversationSummary(value: string, maxLength: number) {
  if (value.length <= maxLength) {
    return value;
  }

  return `${value.slice(0, Math.max(0, maxLength - 1)).trim()}...`;
}

function buildLocalDraftReply(prompt: string, artifactTitle: string) {
  const trimmedPrompt = prompt.trim();
  const promptSummary =
    trimmedPrompt.length > 88
      ? `${trimmedPrompt.slice(0, 85).trim()}...`
      : trimmedPrompt;

  return `Local draft started for "${promptSummary}". I kept ${artifactTitle} active and advanced the workflow in this workspace.`;
}

function withPersistenceTimeout<T>(
  promise: Promise<T>,
  fallback: T,
  timeoutMs: number
): Promise<T> {
  return new Promise((resolve) => {
    const timeoutId = window.setTimeout(() => {
      resolve(fallback);
    }, timeoutMs);

    promise
      .then(resolve, () => resolve(fallback))
      .finally(() => window.clearTimeout(timeoutId));
  });
}

export const workstationIcons = {
  code: Braces,
  context: PanelRight,
  preview: Play
};
