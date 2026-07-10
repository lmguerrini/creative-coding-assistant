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
  type MouseEvent,
  type SyntheticEvent
} from "react";
import {
  Activity,
  Boxes,
  Braces,
  ChevronDown,
  Command,
  Database,
  Gauge,
  LayoutGrid,
  Maximize2,
  Minimize2,
  Moon,
  PanelRight,
  Play,
  Plus,
  RefreshCw,
  RotateCcw,
  SendHorizontal,
  Settings,
  Sparkles,
  TerminalSquare,
  Undo2,
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
  streamAssistantEvents as streamBackendAssistantEvents,
  workflowNodeFromAssistantStreamEvent,
  type AssistantArtifactRefinementRequest,
  type AssistantStreamEvent,
  type AssistantStreamRequest
} from "@/lib/assistant-stream";
import {
  createWorkspacePersistenceClient,
  createWorkspaceSessionRecord,
  defaultWorkspacePreferences,
  defaultWorkspaceLayoutState,
  fingerprintWorkspaceSessionRecord,
  normalizeWorkspacePreferences,
  normalizeWorkspaceLayoutState,
  snapshotFromWorkspaceSessionRecord,
  workspaceLayoutBounds,
  type WorkspaceLayoutState,
  type WorkspacePreferences,
  type WorkspacePersistenceClient,
  type WorkspacePersistenceLoadResult
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
import { buildMultiPreviewComparisonModel } from "@/lib/multi-preview-comparison";
import { buildProjectBundle } from "@/lib/project-bundle";
import {
  buildWorkflowRuntimeModel,
  type WorkflowRuntimeModel,
  type WorkflowRuntimeTraceEvent,
  type WorkflowRuntimeVisualState
} from "@/lib/workflow-runtime";
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
  formatImageAttachmentSize,
  normalizeImageAttachments,
  supportedImageUploadAccept,
  toAssistantRequestImageAttachments
} from "@/lib/multimodal-attachments";
import {
  buildConversationEntries,
  getComposerStatusLabel,
  getConversationPhaseBadge,
  getConversationPhasePlaceholder,
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
  demoModeRecommendedLiveSequence,
  demoModeScenarios,
  getDefaultDemoModeScenario,
  type DemoModeScenario
} from "@/lib/demo-mode";
import {
  buildCreativeTimelineModel,
  type CreativeTimelineModel
} from "@/lib/creative-timeline";
import {
  buildV3InspectorPanelsModel,
  type V3InspectorPanelsModel
} from "@/lib/v3-inspector-panels";
import {
  buildWorkstationDashboardModel,
  type WorkstationDashboardModel
} from "@/lib/workstation-dashboard";
import {
  buildSessionIntelligenceModel,
  readSessionIntelligenceMetadata,
  type SessionIntelligenceMetadataInput,
  type SessionIntelligenceModel
} from "@/lib/session-intelligence";
import {
  createWorkstationError,
  type WorkstationError
} from "@/lib/workstation-errors";
import { buildWorkstationState } from "@/lib/workstation-state";
import { buildZipArchive, downloadZipArchive } from "@/lib/zip-archive";
import { PreviewRendererSurface } from "./preview-renderer-surface";
import { AudioReactiveMappingSummaryCard } from "./audio-reactive-mapping-summary";
import { ArtifactRefinementPanel } from "./artifact-refinement-panel";
import { CalibratedQualitySummary } from "./calibrated-quality-summary";
import { CreativeTimelineSurface } from "./creative-timeline-surface";
import { CreativeCostIntelligenceDashboard } from "./creative-cost-intelligence-dashboard";
import { CreativeQualityCriticSummary } from "./creative-quality-critic-summary";
import { CreativeTranslationSummaryCard } from "./creative-translation-summary";
import { EvaluationSessionDashboard } from "./evaluation-session-dashboard";
import { LangSmithTraceDeepDive } from "./langsmith-trace-deep-dive";
import { MultiPreviewComparisonWorkspace } from "./multi-preview-comparison-workspace";
import { ProviderObservabilityDeepDive } from "./provider-observability-deep-dive";
import {
  KnowledgeBaseStatusSurface,
  RetrievalInspector,
  RetrievalRunStatusSurface
} from "./retrieval-inspector";
import { RuntimeConsoleInspector } from "./runtime-console-inspector";
import { SacredConsistencySummary } from "./sacred-consistency-summary";
import { SubsystemErrorCallout } from "./subsystem-error-callout";
import { V3InspectorPanelsSurface } from "./v3-inspector-panels-surface";
import { WorkstationDashboardSurface } from "./workstation-dashboard-surface";
import { WorkflowExplorerSurface } from "./workflow-explorer-surface";
import { WorkflowTimelineExplorer } from "./workflow-timeline-explorer";

type WorkstationShellProps = {
  snapshot: AssistantWorkspaceSnapshot;
  streamAssistantEvents?: AssistantStreamClient;
  persistenceClient?: WorkspacePersistenceClient;
};

type AssistantStreamClient = (
  request: AssistantStreamRequest
) => AsyncIterable<AssistantStreamEvent>;

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
  Preview: Play,
  Runtime: Command,
  Code: Braces,
  Workflow: Activity,
  Telemetry: Gauge,
  Artifacts: Boxes,
  Retrieval: TerminalSquare
} satisfies Record<InspectorTabName, LucideIcon>;

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
  artifactId: string;
  state: "success" | "error";
};
type ApprovalActionExecutor = () => Promise<void> | void;
type ResizeTarget = "inspector" | "preview";
type UtilityPanelName = "commands" | "theme" | "kb" | "settings";
type FocusRestoreState = {
  inspectorCollapsed: boolean;
  previewOpen: boolean;
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

const localWorkflowIntervalMs = 850;
const artifactFeedbackDurationMs = 1400;
const defaultWorkspacePersistenceClient = createWorkspacePersistenceClient();
const userModeInspectorTabs = new Set<InspectorTabName>([
  "Preview",
  "Code",
  "Artifacts"
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
    description: "Current studio default with cool aqua accents.",
    accent: "#4cd7c8",
    surface: "linear-gradient(135deg, rgba(76, 215, 200, 0.24), rgba(124, 167, 255, 0.16))"
  },
  {
    value: "codex",
    label: "Codex",
    description: "Neutral graphite with restrained, high-contrast accents.",
    accent: "#b9c4cf",
    surface: "linear-gradient(135deg, rgba(185, 196, 207, 0.18), rgba(132, 146, 160, 0.14))"
  },
  {
    value: "light",
    label: "Light",
    description: "Calm daylight workspace with neutral surfaces.",
    accent: "#4f667a",
    surface: "linear-gradient(135deg, rgba(255, 255, 255, 0.92), rgba(223, 231, 239, 0.86))"
  },
  {
    value: "matrix",
    label: "Matrix",
    description: "Obsidian console with restrained lime signal highlights.",
    accent: "#9fe870",
    surface:
      "linear-gradient(135deg, rgba(159, 232, 112, 0.16), rgba(38, 58, 30, 0.16), rgba(7, 11, 6, 0.26))"
  }
] satisfies readonly ThemePresetOption[];

export function WorkstationShell({
  snapshot: initialSnapshot,
  streamAssistantEvents = streamBackendAssistantEvents,
  persistenceClient = defaultWorkspacePersistenceClient
}: WorkstationShellProps) {
  const [snapshot, setSnapshot] = useState(initialSnapshot);
  const entryIdCounterRef = useRef(0);
  const approvalIdCounterRef = useRef(0);
  const localRuntimeSequenceRef = useRef(1000);
  const streamingAssistantIdRef = useRef<string | null>(null);
  const hasPreviewRuntimeEventRef = useRef(false);
  const pendingArtifactRefinementRef = useRef<PendingArtifactRefinement | null>(
    null
  );
  const previewRuntimeTelemetryKeysRef = useRef<Set<string>>(new Set());
  const previewRuntimeErrorScopesRef = useRef<Set<string>>(new Set());
  const [conversationEntries, setConversationEntries] = useState(() =>
    buildConversationEntries(initialSnapshot.messages, createConversationEntryId)
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
  const [activeTab, setActiveTab] = useState<InspectorTabName>(
    getInitialActiveTab(initialSnapshot)
  );
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
  const [isFocusMode, setIsFocusMode] = useState(false);
  const [activeResizeTarget, setActiveResizeTarget] =
    useState<ResizeTarget | null>(null);
  const [openUtilityPanel, setOpenUtilityPanel] = useState<UtilityPanelName | null>(
    null
  );
  const [copyFeedback, setCopyFeedback] = useState<ArtifactActionFeedback | null>(
    null
  );
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
  const chatLogRef = useRef<HTMLDivElement>(null);
  const composerTextareaRef = useRef<HTMLTextAreaElement | null>(null);
  const attachmentMenuRef = useRef<HTMLDivElement | null>(null);
  const approvalCardRef = useRef<HTMLElement | null>(null);
  const shouldAutoScrollRef = useRef(true);
  const isShellMountedRef = useRef(true);
  const hasLoadedPersistenceRef = useRef(false);
  const lastPersistedFingerprintRef = useRef<string | null>(null);
  const skipNextPersistenceSaveRef = useRef(false);
  const focusRestoreRef = useRef<FocusRestoreState | null>(null);
  const copyFeedbackTimerRef = useRef<number | null>(null);
  const transferFeedbackTimerRef = useRef<number | null>(null);
  const dragCleanupRef = useRef<(() => void) | null>(null);
  const utilityTrayRef = useRef<HTMLDivElement>(null);
  const approvalExecutorsRef = useRef<Record<string, ApprovalActionExecutor>>({});
  const imageAttachmentCounterRef = useRef(imageAttachments.length);

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
      clearFeedbackTimers();
      clearDragState();
    };
  }, []);

  useEffect(() => {
    document.documentElement.dataset.ccaTheme = workspacePreferences.theme;

    return () => {
      delete document.documentElement.dataset.ccaTheme;
    };
  }, [workspacePreferences.theme]);

  useEffect(() => {
    if (
      !workspacePreferences.showDebugPanels &&
      !userModeInspectorTabs.has(activeTab)
    ) {
      setActiveTab(userModeDefaultInspectorTab);
    }
  }, [activeTab, workspacePreferences.showDebugPanels]);

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
    const textarea = composerTextareaRef.current;
    if (!textarea) {
      return;
    }

    const maxHeight = 168;
    textarea.style.height = "auto";
    const nextHeight = Math.min(textarea.scrollHeight || 38, maxHeight);
    textarea.style.height = `${nextHeight}px`;
    textarea.style.overflowY = textarea.scrollHeight > maxHeight ? "auto" : "hidden";
  }, [composerValue]);

  useEffect(() => {
    const chatLog = chatLogRef.current;
    if (!chatLog) {
      return undefined;
    }

    const syncAutoScrollPreference = () => {
      const distanceFromBottom =
        chatLog.scrollHeight - chatLog.scrollTop - chatLog.clientHeight;
      shouldAutoScrollRef.current = distanceFromBottom <= 88;
    };

    syncAutoScrollPreference();
    chatLog.addEventListener("scroll", syncAutoScrollPreference, {
      passive: true
    });

    return () => {
      chatLog.removeEventListener("scroll", syncAutoScrollPreference);
    };
  }, []);

  useEffect(() => {
    const chatLog = chatLogRef.current;
    if (!chatLog) {
      return;
    }

    if (conversationEntries.length === 0 && !isStreaming && !isDemoModeOpen) {
      chatLog.scrollTop = 0;
      return;
    }

    if (!shouldAutoScrollRef.current) {
      return;
    }

    chatLog.scrollTop = chatLog.scrollHeight;
  }, [conversationEntries, isDemoModeOpen, isStreaming]);

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
    const handleKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpenUtilityPanel(null);
      }
    };

    document.addEventListener("pointerdown", handlePointerDown);
    window.addEventListener("keydown", handleKeyDown);

    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [openUtilityPanel]);

  useEffect(() => {
    if (!isAttachmentMenuOpen) {
      return undefined;
    }

    const handlePointerDown = (event: PointerEvent) => {
      if (
        attachmentMenuRef.current &&
        event.target instanceof Node &&
        !attachmentMenuRef.current.contains(event.target)
      ) {
        setIsAttachmentMenuOpen(false);
      }
    };
    const handleKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsAttachmentMenuOpen(false);
      }
    };

    document.addEventListener("pointerdown", handlePointerDown);
    window.addEventListener("keydown", handleKeyDown);

    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [isAttachmentMenuOpen]);

  useEffect(() => {
    if (!isPreviewFullscreen) {
      return undefined;
    }

    const previousOverflow = document.body.style.overflow;
    const handleKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsPreviewFullscreen(false);
      }
    };

    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", handleKeyDown);

    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [isPreviewFullscreen]);

  useEffect(() => {
    let isMounted = true;

    async function restoreWorkspaceSession() {
      try {
        const restoredSession = await withPersistenceTimeout<WorkspacePersistenceLoadResult>(
          persistenceClient.load(),
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
          setSnapshot(restoredSnapshot);
          setConversationEntries(
            buildConversationEntries(restoredSnapshot.messages, createConversationEntryId)
          );
          setImageAttachments(restoredImageAttachments);
          setImageUploadError(restoredSnapshot.multimodal.error ?? null);
          setClarification(restoredSnapshot.clarification ?? null);
          setSessionIntelligenceMetadata(null);
          imageAttachmentCounterRef.current = restoredImageAttachments.length;
          streamingAssistantIdRef.current = null;
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
          setStreamEvents(restoredSnapshot.debug.events);
          lastPersistedFingerprintRef.current =
            fingerprintWorkspaceSessionRecord(restoredSession.record);
          skipNextPersistenceSaveRef.current = !restoredArtifactSelectionWasNormalized;
          setPersistenceState(
            restoredSession.source === "local" ? "local" : "restored"
          );
          return;
        }

        setPersistenceState(restoredSession.error ? "unavailable" : "ready");
      } catch {
        if (isMounted) {
          setPersistenceError(buildPersistenceTimeoutError("load"));
          setPersistenceState("unavailable");
        }
      } finally {
        hasLoadedPersistenceRef.current = true;
      }
    }

    restoreWorkspaceSession();

    return () => {
      isMounted = false;
    };
  }, [initialSnapshot, persistenceClient]);

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
  const interactiveSnapshot: AssistantWorkspaceSnapshot = useMemo(
    () => ({
      ...snapshot,
      clarification,
      code: buildCodeSummaryForArtifact(snapshot.code, activeArtifact),
      inspectorTabs: snapshot.inspectorTabs.map((tab) => ({
        ...tab,
        active: tab.label === activeTab,
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
      activeTab,
      clarification,
      imageAttachments,
      imageUploadError,
      isPreviewOpen,
      previewSessionOverride,
      isStreaming,
      persistedMessages,
      previewArtifactId,
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
        activeInspectorTab: activeTab,
        layout: layoutState,
        preferences: workspacePreferences,
        previewArtifactId,
        previewOpen: isPreviewOpen,
        snapshot: interactiveSnapshot
      }),
    [
      activeArtifactId,
      activeTab,
      interactiveSnapshot,
      layoutState,
      workspacePreferences,
      isPreviewOpen,
      previewArtifactId
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
  const providerTelemetry = useMemo(
    () => buildProviderTelemetryModel(workflowTraceEvents),
    [workflowTraceEvents]
  );
  useEffect(() => {
    const completedRun = buildCreativeCostRunRecord({
      providerTelemetry,
      traceEvents: workflowTraceEvents
    });
    if (!completedRun) {
      return;
    }

    setCreativeCostRunHistory((currentHistory) =>
      currentHistory.some((run) => run.id === completedRun.id)
        ? currentHistory
        : [...currentHistory, completedRun]
    );
  }, [providerTelemetry, workflowTraceEvents]);
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
        activeInspectorTab: activeTab,
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
      activeTab,
      interactiveSnapshot,
      isPreviewFullscreen,
      isPreviewOpen,
      isStreaming,
      layoutState.inspectorCollapsed,
      previewArtifactId,
      streamError,
      telemetryDashboard.evaluation,
      workflowRuntime.summary.currentNode,
      workflowTraceEvents
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
  const inspectorTabs = useMemo(
    () =>
      interactiveSnapshot.inspectorTabs.map((tab) =>
        tab.label === "Runtime"
          ? {
              ...tab,
              badge: runtimeConsole.badge ?? tab.badge
            }
          : tab
      ),
    [interactiveSnapshot.inspectorTabs, runtimeConsole.badge]
  );
  const visibleInspectorTabs = useMemo(
    () =>
      workspacePreferences.showDebugPanels
        ? inspectorTabs
        : inspectorTabs.filter((tab) => userModeInspectorTabs.has(tab.label)),
    [inspectorTabs, workspacePreferences.showDebugPanels]
  );
  const activeTabSummary =
    activeTab === "Runtime"
      ? runtimeConsole.summary
      : workstationState.panels.activeTabSummary;
  const isComposerReady = Boolean(composerValue.trim()) && !isStreaming;
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
  const composerStateLabel = getComposerStatusLabel({
    isReady: isComposerReady,
    isStreaming,
    phase: liveAssistantEntry?.phase ?? null,
    streamError
  });
  const persistenceStatusLabel =
    persistenceStateLabels[persistenceState] ?? "Local session ready";
  const hasWorkspaceArtifacts = snapshot.artifacts.length > 0;
  const shouldRenderPreviewShelf =
    !isFocusMode &&
    (interactiveSnapshot.preview.available ||
      (!workspacePreferences.showDebugPanels &&
        (hasWorkspaceArtifacts || isDemoModeOpen)));
  const activeArtifactDisplayLabel = workspacePreferences.showDebugPanels
    ? activeArtifact.title
    : formatUserArtifactLabel(activeArtifact);
  const isInspectorCollapsed = layoutState.inspectorCollapsed;
  const sessionStatusLabel = blockingApprovalRequest
    ? getHitlApprovalStateLabel(blockingApprovalRequest.state)
    : workstationState.status.label;
  const sessionStatusDetail = blockingApprovalRequest
    ? blockingApprovalRequest.title
    : workstationState.status.detail;
  const userSessionStatus = formatUserModeSessionStatus({
    hasFailedPreviewRuntime: runtimeConsole.health.signal === "failed",
    hasWorkspaceArtifacts,
    isDemoModeOpen,
    streamError,
    streamState
  });
  const visibleSessionStatusLabel = workspacePreferences.showDebugPanels
    ? sessionStatusLabel
    : userSessionStatus.label;
  const visibleSessionStatusDetail = workspacePreferences.showDebugPanels
    ? sessionStatusDetail
    : userSessionStatus.detail;
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
    withPersistenceTimeout(
      persistenceClient.save(persistenceRecord),
      {
        error: buildPersistenceTimeoutError("save"),
        target: "local"
      },
      1500
    )
      .then((result) => {
        if (!isShellMountedRef.current) {
          return;
        }

        setPersistenceError(result.error);
        setPersistenceState(result.target === "remote" ? "saved" : "local");
      })
      .catch(() => {
        if (isShellMountedRef.current) {
          setPersistenceError(buildPersistenceTimeoutError("save"));
          setPersistenceState("unavailable");
        }
      });
    return undefined;
  }, [persistenceClient, persistenceRecord, persistenceState]);

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

  function handleDisplayModeToggle() {
    updateWorkspacePreferences({
      showDebugPanels: !workspacePreferences.showDebugPanels
    });
  }

  function handleDensityToggle() {
    updateLayout({
      density: layoutState.density === "compact" ? "cozy" : "compact"
    });
  }

  function toggleUtilityPanel(panelName: UtilityPanelName) {
    setIsAttachmentMenuOpen(false);
    setOpenUtilityPanel((currentPanel) =>
      currentPanel === panelName ? null : panelName
    );
  }

  function revealInspectorTab(nextTab: InspectorTabName) {
    if (isFocusMode) {
      handleFocusModeToggle();
    }

    if (layoutState.inspectorCollapsed) {
      handleInspectorCollapsedChange(false, { preserveFocusMode: true });
    }

    setActiveTab(nextTab);
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
  }

  async function handleApprovalApprove(request: HitlApprovalRequest) {
    const execute = approvalExecutorsRef.current[request.id];
    const approvedRequest = setApprovalRequestState(request, "approved");
    const executingRequest = setApprovalRequestState(approvedRequest, "executing");

    if (!execute) {
      setApprovalRequestState(executingRequest, "failed", {
        failureReason: "No approval executor was available for this action."
      });
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
    }
  }

  function handleApprovalReject(request: HitlApprovalRequest) {
    delete approvalExecutorsRef.current[request.id];
    setApprovalRequestState(request, "rejected");
  }

  function handleApprovalDismiss(request: HitlApprovalRequest) {
    if (!isHitlApprovalTerminalState(request.state)) {
      return;
    }

    setDismissedApprovalRequestId(request.id);
  }

  function clearWorkspaceSession() {
    const clearedSnapshot = getInitialWorkspaceSnapshot();

    clearFeedbackTimers();
    setCopyFeedback(null);
    setTransferFeedback(null);
    setArtifactTransferError(null);
    streamingAssistantIdRef.current = null;
    hasPreviewRuntimeEventRef.current = false;
    pendingArtifactRefinementRef.current = null;
    previousPreviewRuntimeSessionKeyRef.current = null;
    previewRuntimeTelemetryKeysRef.current.clear();
    previewRuntimeErrorScopesRef.current.clear();
    setSnapshot(clearedSnapshot);
    setConversationEntries(
      buildConversationEntries(clearedSnapshot.messages, createConversationEntryId)
    );
    setImageAttachments(
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
    setIsStreaming(false);
    setStreamError(null);
    setStreamEvents(clearedSnapshot.debug.events);
    setClarification(clearedSnapshot.clarification ?? null);
    setSessionIntelligenceMetadata(null);
    setWorkflowTraceEvents([]);
    setCreativeCostRunHistory([]);
    setPreviewRuntimeLive(null);
    updateLayout({
      inspectorCollapsed: defaultWorkspaceLayoutState.inspectorCollapsed,
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
      previewOpen: isPreviewOpen
    };
    setIsPreviewFullscreen(false);
    handleInspectorCollapsedChange(true, { preserveFocusMode: true });
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

  function handlePreviewSessionReset() {
    requestOperatorApproval({
      actionId: "preview_runtime_reset",
      artifactTitle:
        snapshot.artifacts.find((artifact) => artifact.id === resolvePreviewResetArtifactId())
          ?.title ?? interactiveSnapshot.preview.artifactName,
      execute: () => {
        const nextArtifactId = resolvePreviewResetArtifactId();

        setPreviewArtifactId(nextArtifactId);
        setPreviewSessionOverride(null);
        setIsPreviewFullscreen(false);
        setPreviewRuntimeLive(null);
        handlePreviewOpenChange(true, { preserveFocusMode: true });
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

  async function handleImageUploadChange(event: FormEvent<HTMLInputElement>) {
    const input = event.currentTarget;
    const files = Array.from(input.files ?? []);
    input.value = "";
    setIsAttachmentMenuOpen(false);

    if (files.length === 0) {
      return;
    }

    const nextAttachments: ImageAttachmentSummary[] = [];
    let nextError: WorkstationError | null = null;

    for (const file of files) {
      const result = await createImageAttachmentFromFile({
        createdAt: new Date().toISOString(),
        existingCount: imageAttachments.length + nextAttachments.length,
        file,
        id: createImageAttachmentId(file.name)
      });

      if (result.ok) {
        nextAttachments.push(result.attachment);
      } else {
        nextError = result.error;
        break;
      }
    }

    if (nextAttachments.length > 0) {
      const updatedAttachments = [...imageAttachments, ...nextAttachments];
      setImageAttachments(updatedAttachments);
      appendImageReferenceRuntimeEvent({
        attachments: updatedAttachments,
        code: "image_reference_attached",
        message: `${nextAttachments.length} ${pluralize(
          nextAttachments.length,
          "image reference",
          "image references"
        )} attached to the session.`
      });
    }

    setImageUploadError(nextError);
  }

  function handleImageAttachmentRemove(attachmentId: string) {
    const nextAttachments = imageAttachments.filter(
      (attachment) => attachment.id !== attachmentId
    );
    setImageAttachments(nextAttachments);
    setImageUploadError(null);
    appendImageReferenceRuntimeEvent({
      attachments: nextAttachments,
      code: "image_reference_removed",
      message: "Image reference removed from the session."
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
    code: "image_reference_attached" | "image_reference_removed";
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
      setActiveTab(userModeDefaultInspectorTab);
      updateLayout({ inspectorCollapsed: true });
      updateWorkspacePreferences({ showDebugPanels: false });
    }
    setOpenUtilityPanel(null);
    setIsAttachmentMenuOpen(false);
  }

  function handleDemoScenarioLoad(scenario: DemoModeScenario) {
    setActiveDemoScenarioId(scenario.id);
    setIsDemoModeOpen(true);
    setActiveTab(userModeDefaultInspectorTab);
    updateLayout({ inspectorCollapsed: true });
    setComposerValue(scenario.prompt);
    updateWorkspacePreferences({ showDebugPanels: false });
    setOpenUtilityPanel(null);
    setIsAttachmentMenuOpen(false);

    window.requestAnimationFrame(() => {
      composerTextareaRef.current?.focus({ preventScroll: true });
    });
  }

  async function handleComposerSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const prompt = composerValue.trim();

    if (!prompt) {
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

    await submitAssistantRequest({ prompt });
  }

  async function handleArtifactRefine(
    artifact: ArtifactSummary,
    instruction: string
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
      prompt
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
    prompt
  }: {
    artifactRefinement?: AssistantArtifactRefinementRequest;
    clarificationResponse?: string;
    prompt: string;
  }) {
    const timestamp = formatMessageTime();
    const userMessageId = createConversationEntryId();
    const assistantMessageId = createConversationEntryId();
    const pendingRefinement = artifactRefinement
      ? {
          ...artifactRefinement,
          requestedAt: new Date().toISOString()
        }
      : null;
    const userMessageContent = clarificationResponse
      ? `Clarification: ${clarificationResponse}`
      : artifactRefinement
      ? `Refine ${artifactRefinement.title}: ${prompt}`
      : prompt;

    pendingArtifactRefinementRef.current = pendingRefinement;
    streamingAssistantIdRef.current = assistantMessageId;
    setConversationEntries((currentMessages) => [
      ...currentMessages,
      {
        content: userMessageContent,
        activity: null,
        id: userMessageId,
        pending: false,
        phase: "complete",
        role: "user",
        time: timestamp
      },
      {
        content: "",
        activity: artifactRefinement
          ? "Opening refinement pass."
          : "Opening live response.",
        id: assistantMessageId,
        pending: true,
        phase: "connecting",
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
    hasPreviewRuntimeEventRef.current = false;
    previewRuntimeTelemetryKeysRef.current.clear();
    previewRuntimeErrorScopesRef.current.clear();
    setIsStreaming(true);
    setActiveTab("Overview");

    let streamedAnswer = "";
    let receivedTerminalStreamError = false;
    const requestAttachments = toAssistantRequestImageAttachments(imageAttachments);

    try {
      const streamRequest: AssistantStreamRequest = {
        conversationId: "local-nextjs-session",
        mode: "generate",
        projectId: "local-nextjs-workspace",
        query: prompt
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

      for await (const streamEvent of streamAssistantEvents(streamRequest)) {
        applyStreamEventToWorkspace(streamEvent);

        if (
          streamEvent.event_type === "token_delta" &&
          !receivedTerminalStreamError
        ) {
          const delta = readPayloadText(streamEvent, "text");
          if (delta) {
            streamedAnswer += delta;
            startTransition(() => {
              updateStreamingAssistantMessage({
                activity: artifactRefinement
                  ? "Refining selected artifact."
                  : "Generating response.",
                content: streamingConversationSummary,
                phase: "streaming"
              });
            });
          }
        }

        if (streamEvent.event_type === "final" && !receivedTerminalStreamError) {
          const answer = readPayloadText(streamEvent, "answer");
          streamedAnswer = answer ?? streamedAnswer;
          finalizeStreamingAssistantMessage({
            activity: artifactRefinement
              ? "Refinement completed."
              : "Response completed.",
            content: buildAssistantConversationSummary(streamedAnswer),
            phase: "complete"
          });
        }

        if (streamEvent.event_type === "error") {
          receivedTerminalStreamError = true;
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
          finalizeStreamingAssistantMessage({
            activity: error.userMessage,
            content: streamedAnswer
              ? `${buildAssistantConversationSummary(
                  streamedAnswer
                )}\n\nLive response error: ${error.userMessage}`
              : `Live response error: ${error.userMessage}`,
            phase: "error"
          });
        }
      }

      if (!receivedTerminalStreamError && streamingAssistantIdRef.current && streamedAnswer) {
        finalizeStreamingAssistantMessage({
          activity: artifactRefinement
            ? "Refinement completed."
            : "Response completed.",
          content: buildAssistantConversationSummary(streamedAnswer),
          phase: "complete"
        });
      } else if (!receivedTerminalStreamError && streamingAssistantIdRef.current) {
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
        finalizeStreamingAssistantMessage({
          activity: error.userMessage,
          content: error.userMessage,
          phase: "error"
        });
      }
    } catch (error) {
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
      finalizeStreamingAssistantMessage({
        activity: "Switching to a local draft.",
        content: fallbackMessage,
        phase: "fallback"
      });
      setWorkflowProgressIndex(0);
      setWorkflowRunId((currentRunId) => currentRunId + 1);
    } finally {
      if (pendingArtifactRefinementRef.current?.requestedAt === pendingRefinement?.requestedAt) {
        pendingArtifactRefinementRef.current = null;
      }
      setIsStreaming(false);
    }
  }

  function applyStreamEventToWorkspace(streamEvent: AssistantStreamEvent) {
    const receivedAt = new Date().toISOString();
    const receivedAtMs = Date.now();

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
      setSnapshot((currentSnapshot) => ({
        ...currentSnapshot,
        creativePlan: creativePlanUpdate
      }));
    }

    const workflowNode = workflowNodeFromAssistantStreamEvent(streamEvent);
    if (workflowNode) {
      setWorkflowProgressIndex(
        getWorkflowNodeIndex(snapshot.workflow.steps, workflowNode)
      );
    }

    const eventDetail =
      readPayloadText(streamEvent, "message") ??
      readPayloadText(streamEvent, "answer") ??
      null;

    if (
      streamEvent.event_type !== "token_delta" &&
      streamEvent.event_type !== "final" &&
      streamEvent.event_type !== "error"
    ) {
      updateStreamingAssistantMessage({
        activity: eventDetail ?? "Thinking through the request.",
        phase:
          streamEvent.event_type === "status" && streamEvent.sequence === 0
            ? "connecting"
            : "thinking"
      });
    }

    if (streamEvent.event_type === "artifact_extracted") {
      const hydration = annotateRefinedHydration(
        hydrateWorkspaceFromArtifactExtractedEvent(snapshot, streamEvent),
        pendingArtifactRefinementRef.current,
        snapshot
      );

      if (hydration.artifact) {
        setSnapshot(
          creativePlanUpdate
            ? { ...hydration.snapshot, creativePlan: creativePlanUpdate }
            : hydration.snapshot
        );
        setActiveArtifactId(hydration.activeArtifactId);
        setPreviewArtifactId(hydration.previewArtifactId);
        setPreviewSessionOverride(null);
      }
    }

    if (streamEvent.event_type === "preview_artifact") {
      const previewUpdate = readPreviewArtifactUpdate(streamEvent);
      const nextPreviewArtifactId =
        previewUpdate?.previewArtifactId ?? previewUpdate?.artifactId ?? null;
      const previewArtifact = nextPreviewArtifactId
        ? snapshot.artifacts.find((artifact) => artifact.id === nextPreviewArtifactId) ??
          null
        : null;
      const previewEventIsCodeOnly =
        previewUpdate?.artifactDomain === "react_three_fiber" ||
        previewUpdate?.artifactPreviewEligible === false;
      const previewCanOpen =
        previewUpdate?.status === "succeeded" &&
        !previewEventIsCodeOnly &&
        (!previewArtifact || isArtifactPreviewable(previewArtifact));

      if (!previewCanOpen) {
        hasPreviewRuntimeEventRef.current = false;
        if (previewArtifact && !isArtifactPreviewable(previewArtifact)) {
          setPreviewArtifactId("");
        }
        handlePreviewOpenChange(false);
        return;
      }

      hasPreviewRuntimeEventRef.current = true;

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
        nextPreviewArtifactId &&
        snapshot.artifacts.some((artifact) => artifact.id === nextPreviewArtifactId)
      ) {
        setPreviewArtifactId(nextPreviewArtifactId);
      }
      if (workspacePreferences.autoOpenPreview) {
        handlePreviewOpenChange(true);
        setActiveTab("Preview");
      }
    }

    if (streamEvent.event_type === "final") {
      const hydration = annotateRefinedHydration(
        hydrateWorkspaceFromFinalEvent(snapshot, streamEvent, {
          skipPlainTextArtifact: hasPreviewRuntimeEventRef.current
        }),
        pendingArtifactRefinementRef.current,
        snapshot
      );

      if (!hydration.artifact) {
        return;
      }

      setSnapshot(
        creativePlanUpdate
          ? { ...hydration.snapshot, creativePlan: creativePlanUpdate }
          : hydration.snapshot
      );
      setActiveArtifactId(hydration.activeArtifactId);
      setPreviewArtifactId(hydration.previewArtifactId);
      setPreviewSessionOverride(null);
      handlePreviewOpenChange(
        hydration.previewAvailable && workspacePreferences.autoOpenPreview
      );
      if (hydration.previewAvailable && workspacePreferences.autoOpenPreview) {
        setActiveTab("Preview");
      }
    }
  }

  function updateStreamingAssistantMessage(
    nextState: Partial<
      Pick<ConversationEntry, "activity" | "content" | "pending" | "phase">
    >
  ) {
    const streamingAssistantId = streamingAssistantIdRef.current;
    if (!streamingAssistantId) {
      return;
    }

    setConversationEntries((currentMessages) => {
      const nextMessages = [...currentMessages];
      const assistantIndex = nextMessages.findIndex(
        (message) => message.id === streamingAssistantId
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

  function finalizeStreamingAssistantMessage({
    activity,
    content,
    phase
  }: {
    activity: string;
    content: string;
    phase: Extract<ConversationEntryPhase, "complete" | "error" | "fallback">;
  }) {
    updateStreamingAssistantMessage({
      activity,
      content,
      pending: false,
      phase
    });
    streamingAssistantIdRef.current = null;
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
      wasCopied ? "success" : "error",
      copyFeedbackTimerRef,
      setCopyFeedback
    );
  }

  function handleArtifactTransfer(artifact: ArtifactSummary) {
    requestOperatorApproval({
      actionId: getArtifactTransferApprovalActionId(artifact),
      artifactTitle: isProjectBundleExportArtifact(artifact)
        ? interactiveSnapshot.workspace.name
        : artifact.title,
      execute: () => {
        setActiveArtifactId(artifact.id);
        if (isArtifactPreviewable(artifact)) {
          setPreviewContextArtifactId(artifact.id);
        }
        setArtifactTransferError(null);
        const wasTransferred = isProjectBundleExportArtifact(artifact)
          ? (() => {
              const bundle = buildProjectBundle({
                approvalSummary,
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
          wasTransferred ? "success" : "error",
          transferFeedbackTimerRef,
          setTransferFeedback
        );
        if (!wasTransferred) {
          const transferError = createArtifactTransferError(artifact);
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
      handleArtifactTransfer(artifact);
      return;
    }

    setActiveTab("Artifacts");
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
      data-density={layoutState.density}
      data-focus-mode={isFocusMode ? "true" : "false"}
      data-inspector-state={isInspectorCollapsed ? "collapsed" : "open"}
      data-preview={isPreviewOpen ? "open" : "closed"}
      data-readiness={workstationState.readiness.state}
      data-resizing={activeResizeTarget ?? "idle"}
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
          data-state={streamState}
        >
          <span>{visibleSessionStatusLabel}</span>
          <strong>{visibleSessionStatusDetail}</strong>
          {workspacePreferences.showDebugPanels ? (
            <small>{formatSessionTelemetryLabel(providerTelemetry)}</small>
          ) : null}
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
            aria-pressed={isDemoModeOpen}
            className="toolbarToggle"
            onClick={handleDemoModeToggle}
            title={isDemoModeOpen ? "Close Demo Mode" : "Open Demo Mode"}
            type="button"
          >
            <Play size={16} />
            <span>Demo Mode</span>
          </button>
          <button
            aria-label="Display mode"
            aria-pressed={workspacePreferences.showDebugPanels}
            className="toolbarToggle"
            onClick={handleDisplayModeToggle}
            title={
              workspacePreferences.showDebugPanels
                ? "Switch to User Mode"
                : "Switch to Developer Mode"
            }
            type="button"
          >
            <Braces size={16} />
            <span>{workspacePreferences.showDebugPanels ? "Developer" : "User"}</span>
          </button>
          <div className="utilityControl">
            <button
              aria-controls="kb-status-panel"
              aria-expanded={openUtilityPanel === "kb"}
              aria-haspopup="dialog"
              aria-label="Knowledge Base status"
              className="toolbarToggle"
              onClick={() => toggleUtilityPanel("kb")}
              title="Check Knowledge Base and retrieval status"
              type="button"
            >
              <Database size={16} />
              <span>KB</span>
            </button>
            {openUtilityPanel === "kb" ? (
              <KnowledgeBaseStatusPanel runtime={retrievalRuntime} />
            ) : null}
          </div>
          <button
            aria-label="Focus mode"
            aria-pressed={isFocusMode}
            className="toolbarToggle"
            onClick={handleFocusModeToggle}
            title={isFocusMode ? "Exit focus mode" : "Enter focus mode"}
            type="button"
          >
            <span>Focus</span>
          </button>
          <button
            aria-label="Workspace density"
            aria-pressed={layoutState.density === "compact"}
            className="toolbarToggle"
            onClick={handleDensityToggle}
            title="Toggle workspace density"
            type="button"
          >
            <LayoutGrid size={16} />
            <span>{layoutState.density === "compact" ? "Compact" : "Cozy"}</span>
          </button>
          <div className="utilityControl">
            <button
              aria-controls="command-menu-panel"
              aria-expanded={openUtilityPanel === "commands"}
              aria-haspopup="dialog"
              aria-label="Command menu"
              className="iconButton"
              onClick={() => toggleUtilityPanel("commands")}
              title="Open quick actions"
              type="button"
            >
              <Command size={18} />
            </button>
            {openUtilityPanel === "commands" ? (
              <CommandMenuPanel
                activeTab={activeTab}
                hasBlockingApproval={Boolean(blockingApprovalRequest)}
                isFocusMode={isFocusMode}
                isPreviewAvailable={interactiveSnapshot.preview.available}
                isPreviewOpen={isPreviewOpen}
                onFocusModeToggle={() => {
                  handleFocusModeToggle();
                  setOpenUtilityPanel(null);
                }}
                onOpenTab={revealInspectorTab}
                onPreviewToggle={handlePreviewShelfFromControl}
                showDebugPanels={workspacePreferences.showDebugPanels}
                onWorkspaceClear={() =>
                  requestOperatorApproval({
                    actionId: "workspace_clear",
                    execute: clearWorkspaceSession
                  })
                }
              />
            ) : null}
          </div>
          <div className="utilityControl">
            <button
              aria-controls="theme-presets-panel"
              aria-expanded={openUtilityPanel === "theme"}
              aria-haspopup="dialog"
              aria-label="Theme"
              className="iconButton"
              onClick={() => toggleUtilityPanel("theme")}
              title="Open theme presets"
              type="button"
            >
              <Moon size={17} />
            </button>
            {openUtilityPanel === "theme" ? (
              <ThemePresetsPanel
                activeTheme={workspacePreferences.theme}
                onSelectTheme={(theme) => {
                  updateWorkspacePreferences({ theme });
                  setOpenUtilityPanel(null);
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
              title="Open workspace settings"
              type="button"
            >
              <Settings size={18} />
            </button>
            {openUtilityPanel === "settings" ? (
              <WorkspaceSettingsPanel
                layoutState={layoutState}
                preferences={workspacePreferences}
                onDensityChange={(density) => updateLayout({ density })}
                onPreferencesChange={updateWorkspacePreferences}
              />
            ) : null}
          </div>
        </div>
      </header>

      <section className="workspaceLayout" aria-label="Creative workspace">
        <div className="mainColumn">
          <section className="sessionPanel" aria-label="Creative session">
            <div className="sessionIntro">
              <header className="sessionHeader">
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
              {visibleApprovalRequest ? (
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
              {isDemoModeOpen ? (
                <DemoModePanel
                  activeScenario={activeDemoScenario}
                  isPromptLoaded={composerValue === activeDemoScenario.prompt}
                  scenarios={demoModeScenarios}
                  onLoadScenario={handleDemoScenarioLoad}
                  showDebugPanels={workspacePreferences.showDebugPanels}
                />
              ) : null}
            </div>

            <div
              aria-label="Conversation"
              aria-busy={isStreaming}
              aria-live="polite"
              className="chatLog"
              ref={chatLogRef}
              role="log"
            >
              {conversationEntries.length === 0 && !isStreaming && !isDemoModeOpen ? (
                <EmptyWorkspaceState onSelectPrompt={handleEmptyStatePromptSelect} />
              ) : null}
              {conversationEntries.map((message, index) => {
                const displayContent = getConversationDisplayContent(
                  message,
                  workspacePreferences.showDebugPanels
                );

                return (
                  <article
                    className="message"
                    data-fresh={index >= snapshot.messages.length ? "true" : undefined}
                    data-role={message.role}
                    data-stream-phase={message.phase}
                    key={message.id}
                  >
                    <div className="messageMeta">
                      <span>{message.role}</span>
                      <div className="messageMetaDetail">
                        {message.role === "assistant" ? (
                          <small data-phase={message.phase}>
                            {getConversationPhaseBadge(message.phase)}
                          </small>
                        ) : null}
                        <span>{message.time}</span>
                      </div>
                    </div>
                    <p>
                      {displayContent || getConversationPhasePlaceholder(message.phase)}
                      {message.phase === "streaming" ? (
                        <span className="streamingCaret" aria-hidden="true" />
                      ) : null}
                    </p>
                    {message.activity && message.phase !== "complete" ? (
                      <div className="messageActivity">
                        <span aria-hidden="true" />
                        <small>{message.activity}</small>
                      </div>
                    ) : null}
                  </article>
                );
              })}
            </div>
            {streamError ? (
              <SubsystemErrorCallout
                className="chatErrorCallout"
                error={streamError}
                title="Live stream interrupted"
              />
            ) : null}

            {interactiveSnapshot.multimodal.imageAttachments.length > 0 ||
            interactiveSnapshot.multimodal.error ? (
              <ImageReferenceShelf
                multimodal={interactiveSnapshot.multimodal}
                onDismissError={handleImageUploadErrorDismiss}
                onRemove={handleImageAttachmentRemove}
              />
            ) : null}

            <form
              className="composer"
              data-has-images={
                interactiveSnapshot.multimodal.imageAttachments.length > 0
                  ? "true"
                  : "false"
              }
              data-mode={
                workspacePreferences.showDebugPanels ? "developer" : "user"
              }
              data-ready={isComposerReady}
              onSubmit={handleComposerSubmit}
            >
              <div className="composerInputFrame">
                <div className="composerAttach" ref={attachmentMenuRef}>
                  <button
                    aria-controls="composer-attachment-menu"
                    aria-expanded={isAttachmentMenuOpen}
                    aria-label="Add attachment"
                    className="composerAttachButton"
                    disabled={isStreaming}
                    onClick={() =>
                      setIsAttachmentMenuOpen((currentValue) => !currentValue)
                    }
                    title="Add attachment"
                    type="button"
                  >
                    <Plus size={18} aria-hidden="true" />
                  </button>
                  {isAttachmentMenuOpen ? (
                    <div
                      aria-label="Attachment options"
                      className="attachmentMenu"
                      id="composer-attachment-menu"
                      role="menu"
                    >
                      <label className="attachmentMenuItem">
                        <input
                          accept={supportedImageUploadAccept}
                          aria-label="Upload image attachment"
                          disabled={isStreaming}
                          multiple
                          onChange={(event) => void handleImageUploadChange(event)}
                          type="file"
                        />
                        <span>Upload image</span>
                      </label>
                    </div>
                  ) : null}
                </div>
                <textarea
                  aria-label="Assistant prompt"
                  onChange={(event) => setComposerValue(event.currentTarget.value)}
                  placeholder="Ask for a denser particle field, a softer palette, or a preview pass."
                  ref={composerTextareaRef}
                  value={composerValue}
                />
              </div>
              {workspacePreferences.showDebugPanels ? (
                <span className="composerState" aria-live="polite">
                  {composerStateLabel}
                </span>
              ) : null}
              <button
                aria-label="Send prompt"
                className="sendButton"
                data-ready={isComposerReady}
                disabled={!isComposerReady}
                title={isComposerReady ? "Send prompt" : "Type a prompt to send"}
                type="submit"
              >
                <SendHorizontal size={18} />
              </button>
            </form>
          </section>

          {shouldRenderPreviewShelf ? (
            <PreviewShelf
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
              onReset={handlePreviewSessionReset}
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

            <aside
              aria-label="Right inspector"
              className="inspector"
              data-state={isInspectorCollapsed ? "collapsed" : "open"}
            >
              {isInspectorCollapsed ? (
                <div className="inspectorRail">
                  <button
                    aria-label="Expand inspector"
                    className="iconButton"
                    onClick={() => handleInspectorCollapsedChange(false)}
                    type="button"
                  >
                    <PanelRight size={18} />
                  </button>
                  <strong>Inspector</strong>
                  <span>
                    {formatInspectorTabDisplayLabel(
                      activeTab,
                      workspacePreferences.showDebugPanels
                    )}
                  </span>
                </div>
              ) : (
                <>
                  <header className="inspectorHeader">
                    <div>
                      <span className="eyebrow">Inspector</span>
                      <h2>
                        {formatInspectorTabDisplayLabel(
                          activeTab,
                          workspacePreferences.showDebugPanels
                        )}
                      </h2>
                      <p>{activeTabSummary}</p>
                    </div>
                    <button
                      className="iconButton"
                      type="button"
                      aria-label="Collapse inspector"
                      onClick={() => handleInspectorCollapsedChange(true)}
                    >
                      <PanelRight size={18} />
                    </button>
                  </header>

                  <div className="inspectorTabs" role="tablist" aria-label="Inspector tabs">
                    {visibleInspectorTabs.map((tab) => {
                      const Icon = inspectorTabIcons[tab.label];
                      const displayLabel = formatInspectorTabDisplayLabel(
                        tab.label,
                        workspacePreferences.showDebugPanels
                      );

                      return (
                        <button
                          aria-controls={`${tab.label.toLowerCase()}-inspector-panel`}
                          aria-label={displayLabel}
                          aria-selected={tab.active}
                          data-active={tab.active}
                          id={`${tab.label.toLowerCase()}-inspector-tab`}
                          key={tab.label}
                          onClick={() => setActiveTab(tab.label)}
                          role="tab"
                          tabIndex={tab.active ? 0 : -1}
                          title={tab.summary}
                          type="button"
                        >
                          <Icon size={15} aria-hidden="true" />
                          <span>{displayLabel}</span>
                          {workspacePreferences.showDebugPanels && tab.badge ? (
                            <small>{` ${tab.badge}`}</small>
                          ) : null}
                        </button>
                      );
                    })}
                  </div>

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
                    onArtifactSelect={handleArtifactSelect}
                    onArtifactTransfer={handleArtifactTransfer}
                    onClarificationOptionSelect={handleClarificationOptionSelect}
                    providerTelemetry={providerTelemetry}
                    workstationDashboard={workstationDashboard}
                    previewController={previewController}
                    runtimeConsole={runtimeConsole}
                    provenance={provenance}
                    previewRoute={previewRendererRoute}
                    previewRuntimeSource={previewRuntimeSource}
                    retrievalRuntime={retrievalRuntime}
                    sessionIntelligence={sessionIntelligence}
                    showDebugPanels={workspacePreferences.showDebugPanels}
                    snapshot={interactiveSnapshot}
                    telemetryDashboard={telemetryDashboard}
                    transferFeedback={transferFeedback}
                    workflowExplorer={workflowExplorer}
                    creativeTimeline={creativeTimeline}
                    v3InspectorPanels={v3InspectorPanels}
                    workflowRuntime={workflowRuntime}
                    workflowIssues={workflowIssues}
                  />
                </>
              )}
            </aside>
          </>
        ) : null}
      </section>
    </main>
  );
}

type PreviewShelfProps = WorkstationShellProps & {
  controller: PreviewControllerModel;
  height: number;
  onClear: () => void;
  onFullscreenToggle: (isFullscreen: boolean) => void;
  onOpenArtifacts: () => void;
  onOpenCode: () => void;
  onReload: () => void;
  onRuntimeDiagnostics: (event: Omit<RuntimeConsoleLiveSnapshot, "updatedAt">) => void;
  onResizeKeyDown: (event: KeyboardEvent<HTMLElement>) => void;
  onResizeStart: (event: MouseEvent<HTMLElement>) => void;
  onReset: () => void;
  onRestart: () => void;
  onRuntimeFrame: (event: PreviewRuntimeFrameTelemetryEvent) => void;
  onRuntimeStatus: (event: PreviewRuntimeStatusTelemetryEvent) => void;
  onToggle: (isOpen: boolean) => void;
  route: PreviewRendererRoute;
  runtimeSessionKey: string;
  runtimeSource: ReturnType<typeof buildPreviewRuntimeSource>;
  resizing: boolean;
  showDebugPanels: boolean;
};

function ImageReferenceShelf({
  multimodal,
  onDismissError,
  onRemove
}: {
  multimodal: AssistantWorkspaceSnapshot["multimodal"];
  onDismissError: () => void;
  onRemove: (attachmentId: string) => void;
}) {
  return (
    <section
      aria-label="Image references"
      className="imageReferenceShelf"
      data-state={multimodal.state}
    >
      <header className="imageReferenceHeader">
        <div>
          <span className="eyebrow">Visual context</span>
          <strong>{multimodal.status}</strong>
          <p>{multimodal.detail}</p>
        </div>
        {multimodal.error ? (
          <button
            aria-label="Dismiss image upload issue"
            className="imageReferenceDismiss"
            onClick={onDismissError}
            type="button"
          >
            <X size={14} />
          </button>
        ) : null}
      </header>
      {multimodal.error ? (
        <SubsystemErrorCallout
          className="imageUploadErrorCallout"
          error={multimodal.error}
          title="Image upload issue"
        />
      ) : null}
      {multimodal.imageAttachments.length > 0 ? (
        <div className="imageReferenceList" role="list">
          {multimodal.imageAttachments.map((attachment) => (
            <article
              aria-label={`${attachment.name} image reference`}
              className="imageReferenceCard"
              key={attachment.id}
              role="listitem"
            >
              <div
                aria-hidden="true"
                className="imageReferenceThumb"
                style={{ backgroundImage: `url(${attachment.dataUrl})` }}
              />
              <div>
                <strong>{attachment.name}</strong>
                <span>
                  {attachment.mimeType.replace("image/", "").toUpperCase()} /{" "}
                  {formatImageAttachmentSize(attachment.sizeBytes)}
                </span>
              </div>
              <button
                aria-label={`Remove image reference ${attachment.name}`}
                onClick={() => onRemove(attachment.id)}
                type="button"
              >
                <X size={13} />
              </button>
            </article>
          ))}
        </div>
      ) : null}
    </section>
  );
}

function EmptyWorkspaceState({
  onSelectPrompt
}: {
  onSelectPrompt: (prompt: string) => void;
}) {
  const valueHighlights = [
    {
      title: "Build browser-native visuals",
      detail: "Create p5.js, Three.js, GLSL, and Hydra-ready artifacts."
    },
    {
      title: "Ground answers in official sources",
      detail: "Use retrieval context when source-backed guidance matters."
    },
    {
      title: "Preview, refine, and save artifacts",
      detail: "Move from prompt to Code, Preview, and Saved outputs."
    },
    {
      title: "Support creative-coding workflows",
      detail: "Plan sketches, shaders, visual systems, and installations."
    }
  ];
  const promptSuggestions = [
    "Create a single .p5.js JavaScript sketch for a flow-field particle system with setup(), draw(), soft trails, and interaction controls.",
    "Design a Three.js kinetic sculpture with camera motion and audio-reactive lighting.",
    "Generate a GLSL fragment shader with liquid glass refraction and restrained color.",
    "Build a Hydra feedback pattern with slow color modulation and clear fallback notes."
  ];
  const domainExamples = [
    "p5.js sketches",
    "Three.js scenes",
    "GLSL shaders",
    "Hydra feedback"
  ];
  const workflowExamples = [
    "Describe a visual system",
    "Generate browser-safe code",
    "Preview and refine",
    "Save or export artifacts"
  ];

  return (
    <article
      aria-label="Empty creative workspace"
      className="emptyWorkspace"
      role="group"
    >
      <header className="emptyWorkspaceHero">
        <span className="eyebrow">New creative session</span>
        <strong>Describe the visual system you want to build.</strong>
        <p>
          Start with an idea, medium, constraint, or reference. Creative Coding
          Assistant turns it into grounded guidance, generated code, previewable
          artifacts, and saved outputs.
        </p>
      </header>

      <section className="emptyWorkspaceValue" aria-label="Product capabilities">
        {valueHighlights.map((item) => (
          <article key={item.title}>
            <strong>{item.title}</strong>
            <span>{item.detail}</span>
          </article>
        ))}
      </section>

      <div className="emptyWorkspaceSuggestions" aria-label="Prompt suggestions">
        {promptSuggestions.map((prompt) => (
          <button
            key={prompt}
            onClick={() => onSelectPrompt(prompt)}
            type="button"
          >
            {prompt}
          </button>
        ))}
      </div>

      <div className="emptyWorkspaceGrid">
        <section aria-label="Domain examples">
          <span>Domains</span>
          <div>
            {domainExamples.map((domain) => (
              <small key={domain}>{domain}</small>
            ))}
          </div>
        </section>
        <section aria-label="Ways to work">
          <span>Ways to work</span>
          <div>
            {workflowExamples.map((workflow) => (
              <small key={workflow}>{workflow}</small>
            ))}
          </div>
        </section>
      </div>

      <details className="emptyWorkspaceLearnMore">
        <summary>How it works</summary>
        <p>
          Prompt the assistant, inspect concise chat guidance, open generated
          code in Code, review visual output in Preview, and keep useful results
          in Saved. Developer Mode exposes retrieval, workflow, and telemetry
          details when you need the engineering trace.
        </p>
      </details>
    </article>
  );
}

function DemoModePanel({
  activeScenario,
  isPromptLoaded,
  onLoadScenario,
  scenarios,
  showDebugPanels
}: {
  activeScenario: DemoModeScenario;
  isPromptLoaded: boolean;
  onLoadScenario: (scenario: DemoModeScenario) => void;
  scenarios: readonly DemoModeScenario[];
  showDebugPanels: boolean;
}) {
  const scenarioFacts = showDebugPanels
    ? ([
        ["Capability", activeScenario.recommendedForDemo],
        ["Technology", activeScenario.runtime],
        ["Generation", activeScenario.estimatedGenerationTime],
        ["Presenter time", activeScenario.presentationTime],
        ["Tokens", activeScenario.estimatedTokenUsage],
        ["Workflow", activeScenario.workflowType],
        ["Provider", activeScenario.providerRequirement],
        ["Retrieval", activeScenario.retrievalRequirement],
        ["Preview", activeScenario.previewAvailability],
        ["Fallback", activeScenario.fallbackAvailability],
        ["Expected output", activeScenario.expectedOutput],
        ["Complexity", activeScenario.complexity]
      ] as const)
    : ([
        ["Capability", activeScenario.recommendedForDemo],
        ["Runtime", activeScenario.runtime],
        ["Estimated time", activeScenario.estimatedGenerationTime],
        ["Expected output", activeScenario.expectedOutput]
      ] as const);

  return (
    <section
      aria-label="Demo Mode"
      className="demoModePanel"
      data-debug={showDebugPanels ? "true" : "false"}
      id="demo-mode-panel"
    >
      <header className="demoModeHeader">
        <div>
          <span className="eyebrow">Demo Mode</span>
          <strong>Capstone scenarios</strong>
          <p>
            Curated creative-coding flows with ready prompts and safe fallback
            paths.
          </p>
        </div>
        <span className="demoModeCount">{scenarios.length} flows</span>
      </header>

      {showDebugPanels ? (
        <div className="demoLiveSequence" aria-label="Featured demo paths">
          {demoModeRecommendedLiveSequence.map((item) => (
            <button
              key={`${item.role}-${item.scenarioId}`}
              onClick={() => {
                const scenario = scenarios.find(
                  (candidate) => candidate.id === item.scenarioId
                );
                if (scenario) {
                  onLoadScenario(scenario);
                }
              }}
              type="button"
            >
              <span>{item.role}</span>
              <strong>{item.title}</strong>
              <small>{item.rationale}</small>
            </button>
          ))}
        </div>
      ) : null}

      <div className="demoModeBody">
        <div
          aria-label="Demo Mode scenarios"
          className="demoScenarioList"
          role="list"
        >
          {scenarios.map((scenario) => {
            const isActive = scenario.id === activeScenario.id;

            return (
              <button
                aria-pressed={isActive}
                className="demoScenarioButton"
                data-active={isActive ? "true" : "false"}
                key={scenario.id}
                onClick={() => onLoadScenario(scenario)}
                type="button"
              >
                <span>{getDemoScenarioPublicCategory(scenario)}</span>
                <strong>{scenario.title}</strong>
                <small>
                  {showDebugPanels
                    ? `${scenario.category} / ${scenario.recommendedForDemo}`
                    : scenario.recommendedForDemo}
                </small>
                {isActive ? <ChevronDown size={14} aria-hidden="true" /> : null}
              </button>
            );
          })}
        </div>

        <article
          aria-label="Selected demo scenario"
          className="demoScenarioDetail"
        >
          <header>
            <div>
              <span>{getDemoScenarioPublicCategory(activeScenario)}</span>
              <strong>{activeScenario.title}</strong>
            </div>
            <button
              className="demoModeLoadButton"
              onClick={() => onLoadScenario(activeScenario)}
              type="button"
            >
              <Play size={15} aria-hidden="true" />
              <span>{isPromptLoaded ? "Prompt loaded" : "Load prompt"}</span>
            </button>
          </header>

          <p className="demoScenarioDescription">{activeScenario.description}</p>

          <dl className="demoScenarioQuickFacts">
            {scenarioFacts.map(([label, value]) => (
              <div key={label}>
                <dt>{label}</dt>
                <dd>{value}</dd>
              </div>
            ))}
          </dl>

          <p
            className={
              showDebugPanels
                ? "demoPromptPreview"
                : "demoPromptPreview demoPromptPreview--user"
            }
          >
            {showDebugPanels
              ? activeScenario.prompt
              : formatDemoPromptPreview(activeScenario.prompt)}
          </p>

          {showDebugPanels ? (
            <dl className="demoScenarioMeta">
              <div>
                <dt>Expected behavior</dt>
                <dd>{activeScenario.expectedBehavior}</dd>
              </div>
              <div>
                <dt>Fallback</dt>
                <dd>{activeScenario.fallback}</dd>
              </div>
              <div>
                <dt>Output guidance</dt>
                <dd>{activeScenario.outputGuidance}</dd>
              </div>
            </dl>
          ) : (
            <div className="demoUserEssentials">
              <p>
                <strong>Validates:</strong> {activeScenario.recommendedForDemo}
              </p>
            </div>
          )}

          {showDebugPanels ? (
            <>
              <dl className="demoScenarioMeta demoScenarioMeta--developer">
                <div>
                  <dt>Source boundary</dt>
                  <dd>{activeScenario.sourceBoundary}</dd>
                </div>
                <div>
                  <dt>Validation</dt>
                  <dd>{activeScenario.validationPath}</dd>
                </div>
              </dl>

              <div className="demoScenarioEvidence">
                <span>Evidence</span>
                <div>
                  {activeScenario.evidence.map((item) => (
                    <code key={item}>{item}</code>
                  ))}
                </div>
              </div>
            </>
          ) : null}
        </article>
      </div>
    </section>
  );
}

function getDemoScenarioPublicCategory(scenario: DemoModeScenario) {
  const searchable = [
    scenario.id,
    scenario.runtime,
    scenario.category,
    scenario.workflowType,
    scenario.title
  ]
    .join(" ")
    .toLowerCase();

  if (searchable.includes("concept") || searchable.includes("translation")) {
    return "Concept Translation";
  }

  if (searchable.includes("installation")) {
    return "Installation Planning";
  }

  if (searchable.includes("geometry")) {
    return "Visual Planning";
  }

  if (searchable.includes("three")) {
    return "Three.js";
  }

  if (searchable.includes("p5")) {
    return "p5.js";
  }

  if (searchable.includes("glsl") || searchable.includes("shader")) {
    return "GLSL";
  }

  if (searchable.includes("hydra")) {
    return "Hydra";
  }

  if (searchable.includes("retrieval") || searchable.includes("rag")) {
    return "Retrieval";
  }

  return "Visual Planning";
}

function formatDemoPromptPreview(prompt: string) {
  const normalizedPrompt = prompt.replace(/\s+/g, " ").trim();

  if (normalizedPrompt.length <= 170) {
    return normalizedPrompt;
  }

  return `${normalizedPrompt.slice(0, 167).trimEnd()}...`;
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

function PreviewShelf({
  controller,
  height,
  onClear,
  onFullscreenToggle,
  onOpenArtifacts,
  onOpenCode,
  onReload,
  onRuntimeDiagnostics,
  onResizeKeyDown,
  onResizeStart,
  onReset,
  onRestart,
  onRuntimeFrame,
  onRuntimeStatus,
  onToggle,
  route,
  runtimeSessionKey,
  runtimeSource,
  resizing,
  showDebugPanels,
  snapshot
}: PreviewShelfProps) {
  const canOpenUserPreview = showDebugPanels || snapshot.preview.state === "ready";
  const isPreviewPanelOpen = snapshot.preview.active && canOpenUserPreview;

  function handleSummaryClick(event: MouseEvent<HTMLElement>) {
    event.preventDefault();
    if (!canOpenUserPreview) {
      onToggle(false);
      return;
    }
    onToggle(!isPreviewPanelOpen);
  }

  function handleToggle(event: SyntheticEvent<HTMLDetailsElement>) {
    if (!canOpenUserPreview && event.currentTarget.open) {
      onToggle(false);
      return;
    }
    onToggle(event.currentTarget.open);
  }

  const layoutSize = resolvePreviewShelfLayoutSize(snapshot.preview);
  const panelHeight = resolvePreviewShelfPanelHeight(height, snapshot.preview);
  const canResizePreview =
    isPreviewPanelOpen && layoutSize === "visual" && !controller.isFullscreen;
  const panelStyle = controller.isFullscreen ? undefined : { height: panelHeight };

  if (!showDebugPanels && snapshot.preview.state !== "ready") {
    return (
      <section className="previewZone" aria-label="Preview workspace">
        <section
          aria-label="Preview fallback"
          className="previewShelf previewShelf--userFallback"
          data-runtime-state={snapshot.preview.state}
          data-user-mode="true"
        >
          <div className="previewUserFallbackCard">
            <div>
              <span>Preview</span>
              <strong>Preview unavailable</strong>
              <p>
                Choose a previewable artifact, or inspect Code and Saved while
                a runnable visual is prepared.
              </p>
            </div>
            <div className="previewUserFallbackActions">
              <button onClick={onOpenCode} type="button">
                Open Code
              </button>
              <button onClick={onOpenArtifacts} type="button">
                Open Saved
              </button>
            </div>
          </div>
        </section>
      </section>
    );
  }

  return (
    <section className="previewZone" aria-label="Preview workspace">
      <details
        data-fullscreen={controller.isFullscreen ? "true" : "false"}
        data-layout-size={layoutSize}
        className="previewShelf"
        data-state={isPreviewPanelOpen ? "open" : "closed"}
        data-runtime-state={snapshot.preview.state}
        data-user-mode={showDebugPanels ? "false" : "true"}
        onToggle={handleToggle}
        open={isPreviewPanelOpen}
      >
        <summary
          aria-expanded={isPreviewPanelOpen}
          onClick={handleSummaryClick}
        >
          <span className="previewSummaryIcon" aria-hidden="true">
            <Play size={16} />
          </span>
          <div>
            <strong>{snapshot.preview.title}</strong>
            <span>
              {showDebugPanels
                ? snapshot.preview.artifactName
                : formatUserPreviewArtifactLabel(snapshot)}
            </span>
          </div>
          <div className="previewSummaryMeta">
            <small data-state={snapshot.preview.state}>{snapshot.preview.status}</small>
            <span className="previewSummaryChevron" aria-hidden="true">
              <ChevronDown size={15} />
            </span>
          </div>
        </summary>
        <div className="previewPanel" style={panelStyle}>
          <div className="previewToolbar">
            <div className="previewToolbarFocus" aria-label="Focused preview context">
              <span>{route.surfaceEyebrow}</span>
              <strong>{route.surfaceTitle}</strong>
              <small>
                {showDebugPanels
                  ? `${snapshot.preview.status} / ${route.rendererLabel}`
                  : snapshot.preview.status}
              </small>
            </div>
            <div className="previewToolbarActions" aria-label="Preview controls">
              {showDebugPanels ? (
                <button
                  aria-label="Collapse preview"
                  className="previewControlButton"
                  onClick={() => onToggle(false)}
                  title="Collapse preview"
                  type="button"
                >
                  <ChevronDown size={15} />
                </button>
              ) : null}
              <button
                aria-label={
                  controller.isFullscreen
                    ? "Exit preview fullscreen"
                    : "Enter preview fullscreen"
                }
                aria-pressed={controller.isFullscreen}
                className="previewControlButton"
                disabled={!controller.canFullscreen}
                onClick={() => onFullscreenToggle(!controller.isFullscreen)}
                title={
                  controller.isFullscreen
                    ? "Exit preview fullscreen"
                    : "Enter preview fullscreen"
                }
                type="button"
              >
                {controller.isFullscreen ? (
                  <Minimize2 size={15} />
                ) : (
                  <Maximize2 size={15} />
                )}
              </button>
              <button
                aria-label="Restart preview session"
                className="previewControlButton"
                disabled={!controller.canRestart}
                onClick={onRestart}
                title="Restart preview session"
                type="button"
              >
                <RotateCcw size={15} />
              </button>
              {showDebugPanels ? (
                <>
                  <button
                    aria-label="Clear preview state"
                    className="previewControlButton"
                    disabled={!controller.canClear}
                    onClick={onClear}
                    title="Clear preview state"
                    type="button"
                  >
                    <X size={15} />
                  </button>
                  <button
                    aria-label="Reload preview state"
                    className="previewControlButton"
                    disabled={!controller.canReload}
                    onClick={onReload}
                    title="Reload preview state"
                    type="button"
                  >
                    <RefreshCw size={15} />
                  </button>
                  <button
                    aria-label="Reset preview session"
                    className="previewControlButton"
                    disabled={!controller.canReset}
                    onClick={onReset}
                    title="Reset preview session"
                    type="button"
                  >
                    <Undo2 size={15} />
                  </button>
                </>
              ) : null}
            </div>
          </div>
          <div className="previewBody">
            {snapshot.preview.error ? (
              <SubsystemErrorCallout
                className="previewErrorCallout"
                error={snapshot.preview.error}
                title="Preview runtime failed"
              />
            ) : null}
            <PreviewRendererSurface
              chrome="immersive"
              onReload={onReload}
              onRuntimeDiagnostics={onRuntimeDiagnostics}
              onRuntimeFrame={onRuntimeFrame}
              onRuntimeStatus={onRuntimeStatus}
              preview={snapshot.preview}
              route={route}
              runtimeSessionKey={runtimeSessionKey}
              runtimeSource={runtimeSource}
              showDiagnostics={showDebugPanels}
            />
          </div>
        </div>
        <div
          aria-label="Resize preview shelf"
          aria-disabled={!canResizePreview}
          aria-orientation="horizontal"
          aria-valuemax={
            layoutSize === "visual"
              ? workspaceLayoutBounds.maxPreviewHeight
              : workspaceLayoutBounds.compactPreviewHeight
          }
          aria-valuemin={workspaceLayoutBounds.minPreviewHeight}
          aria-valuenow={panelHeight}
          className="layoutResizeHandle previewResizeHandle"
          data-active={resizing}
          onKeyDown={canResizePreview ? onResizeKeyDown : undefined}
          onMouseDown={canResizePreview ? onResizeStart : undefined}
          role="separator"
          tabIndex={canResizePreview ? 0 : -1}
        >
          <span aria-hidden="true" />
        </div>
      </details>
    </section>
  );
}

function resolvePreviewShelfLayoutSize(
  preview: AssistantWorkspaceSnapshot["preview"]
) {
  return preview.active && preview.state === "ready" ? "visual" : "compact";
}

function resolvePreviewShelfPanelHeight(
  height: number,
  preview: AssistantWorkspaceSnapshot["preview"]
) {
  if (resolvePreviewShelfLayoutSize(preview) === "visual") {
    return height;
  }

  return Math.min(height, workspaceLayoutBounds.compactPreviewHeight);
}

type InspectorPanelProps = {
  activeArtifact: ArtifactSummary;
  activeArtifactDocument: ArtifactDocument;
  activeArtifactHighlights: HighlightedLine[];
  activeArtifactId: string;
  activeTab: InspectorTabName;
  artifactTransferError: WorkstationError | null;
  copyFeedback: ArtifactActionFeedback | null;
  isStreaming: boolean;
  onArtifactCopy: (artifact: ArtifactSummary) => Promise<void>;
  onArtifactAction: (action: ArtifactAction, artifact: ArtifactSummary) => void;
  onArtifactRefine: (artifact: ArtifactSummary, instruction: string) => Promise<void>;
  onArtifactSelect: (artifact: ArtifactSummary) => void;
  onArtifactTransfer: (artifact: ArtifactSummary) => void;
  onClarificationOptionSelect: (option: string) => Promise<void>;
  providerTelemetry: ProviderTelemetryModel;
  workstationDashboard: WorkstationDashboardModel;
  previewController: PreviewControllerModel;
  provenance: ProvenanceEngineModel;
  runtimeConsole: RuntimeConsoleModel;
  previewRoute: PreviewRendererRoute;
  previewRuntimeSource: PreviewRuntimeSource;
  retrievalRuntime: RetrievalRuntimeModel;
  sessionIntelligence: SessionIntelligenceModel;
  showDebugPanels: boolean;
  snapshot: AssistantWorkspaceSnapshot;
  telemetryDashboard: TelemetryDashboardModel;
  transferFeedback: ArtifactActionFeedback | null;
  workflowExplorer: WorkflowExplorerModel;
  creativeTimeline: CreativeTimelineModel;
  v3InspectorPanels: V3InspectorPanelsModel;
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
  onArtifactRefine,
  onArtifactSelect,
  onArtifactTransfer,
  onClarificationOptionSelect,
  providerTelemetry,
  workstationDashboard,
  previewController,
  provenance,
  runtimeConsole,
  previewRoute,
  previewRuntimeSource,
  retrievalRuntime,
  sessionIntelligence,
  showDebugPanels,
  snapshot,
  telemetryDashboard,
  transferFeedback,
  workflowExplorer,
  creativeTimeline,
  v3InspectorPanels,
  workflowRuntime,
  workflowIssues
}: InspectorPanelProps) {
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
        dashboard={telemetryDashboard}
        preview={snapshot.preview}
        route={previewRoute}
        runtimeSource={previewRuntimeSource}
        showDebugPanels={showDebugPanels}
      />
    );
  }

  if (activeTab === "Runtime") {
    return <RuntimeConsoleInspector console={runtimeConsole} />;
  }

  if (activeTab === "Workflow") {
    return (
      <WorkflowInspector
        creativeTimeline={creativeTimeline}
        explorer={workflowExplorer}
        runtime={workflowRuntime}
        provenance={provenance}
        v3InspectorPanels={v3InspectorPanels}
        telemetry={providerTelemetry}
        showDebugPanels={showDebugPanels}
        issues={workflowIssues}
      />
    );
  }

  if (activeTab === "Telemetry") {
    return (
      <TelemetryInspector
        dashboard={telemetryDashboard}
        showDebugPanels={showDebugPanels}
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
        code={snapshot.code}
        copyFeedback={copyFeedback}
        isStreaming={isStreaming}
        onArtifactAction={onArtifactAction}
        onArtifactRefine={onArtifactRefine}
        onArtifactSelect={onArtifactSelect}
        preview={snapshot.preview}
        showDebugPanels={showDebugPanels}
        transferFeedback={transferFeedback}
      />
    );
  }

  if (activeTab === "Retrieval") {
    return (
      <RetrievalInspector
        runtime={retrievalRuntime}
        showDebugPanels={showDebugPanels}
      />
    );
  }

  return (
    <OverviewInspector
      activeArtifact={activeArtifact}
      retrieval={retrievalRuntime}
      runtime={workflowRuntime}
      sessionIntelligence={sessionIntelligence}
      workstationDashboard={workstationDashboard}
      telemetry={providerTelemetry}
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
  retrieval,
  runtime,
  sessionIntelligence,
  workstationDashboard,
  telemetry,
  showDebugPanels,
  snapshot
}: {
  activeArtifact: ArtifactSummary;
  isStreaming: boolean;
  onClarificationOptionSelect: (option: string) => Promise<void>;
  retrieval: RetrievalRuntimeModel;
  runtime: WorkflowRuntimeModel;
  sessionIntelligence: SessionIntelligenceModel;
  workstationDashboard: WorkstationDashboardModel;
  telemetry: ProviderTelemetryModel;
  showDebugPanels: boolean;
  snapshot: AssistantWorkspaceSnapshot;
}) {
  const workflowProgress = getWorkflowRuntimeProgress(runtime.steps);
  const latestTransitions = runtime.transitions.slice(-3);

  return (
    <section
      aria-label="Overview inspector"
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
              <strong>{runtime.summary.currentStep}</strong>
              <p>{formatWorkflowStatusCopy(runtime.summary.status)}</p>
            </div>
            <span
              className="liveDot"
              aria-hidden="true"
              data-state={runtime.summary.status}
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
          <div className="miniWorkflow" aria-label="Minimal live workflow state">
            {runtime.steps.map((step) => (
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
          <div className="workflowMiniTransitions" aria-label="Workflow transitions">
            {latestTransitions.length > 0 ? (
              latestTransitions.map((transition) => (
                <div
                  className="workflowTransitionPill"
                  data-kind={transition.kind}
                  key={`${transition.sequence}-${transition.label}`}
                >
                  {transition.label}
                </div>
              ))
            ) : (
              <p>Waiting for runtime transitions.</p>
            )}
          </div>
        </div>
        <SessionIntelligenceOverviewTile intelligence={sessionIntelligence} />
        <div className="overviewTile" role="group" aria-label="Artifacts summary">
          <span>Artifacts</span>
          <strong>{snapshot.artifacts.length}</strong>
          <p>{activeArtifact.title}</p>
        </div>
        {snapshot.creativePlan ? (
          <CreativePlanOverviewTile plan={snapshot.creativePlan} />
        ) : null}
        {snapshot.clarification ? (
          <ClarificationOverviewTile
            clarification={snapshot.clarification}
            disabled={isStreaming}
            onOptionSelect={onClarificationOptionSelect}
          />
        ) : null}
        <div
          aria-label="Preview summary"
          className="overviewTile"
          data-state={snapshot.preview.state}
          role="group"
        >
          <span>Preview</span>
          <strong>{formatPreviewStateLabel(snapshot.preview.state, snapshot.preview.active)}</strong>
          <p>{snapshot.preview.available ? snapshot.preview.artifactName : "No target"}</p>
        </div>
        <div
          className="overviewTile overviewTelemetryTile"
          data-state={telemetry.status}
          role="group"
          aria-label="Telemetry summary"
        >
          <span>Telemetry</span>
          <strong>{telemetry.summary.costLabel}</strong>
          <p>{`${telemetry.summary.tokenLabel} / ${telemetry.summary.latencyLabel}`}</p>
          <small>{`${telemetry.summary.providerLabel} / ${telemetry.summary.modelLabel}`}</small>
        </div>
        <div
          aria-label="Image references summary"
          className="overviewTile"
          data-state={snapshot.multimodal.state}
          role="group"
        >
          <span>Image references</span>
          <strong>{snapshot.multimodal.imageAttachments.length}</strong>
          <p>{snapshot.multimodal.status}</p>
          <small>{snapshot.multimodal.detail}</small>
        </div>
        <div
          aria-label="Retrieval summary"
          className="overviewTile"
          data-state={retrieval.summary.state}
          role="group"
        >
          <span>Retrieval</span>
          <strong>{retrieval.summary.status}</strong>
          <p>
            {retrieval.summary.sourceCount > 0
              ? `${retrieval.summary.sourceCount} sources / ${retrieval.summary.chunkCount} chunks`
              : retrieval.summary.headline}
          </p>
          <small>{retrieval.summary.freshnessLabel}</small>
        </div>
      </div>
      <WorkstationDashboardSurface dashboard={workstationDashboard} />
    </section>
  );
}

function SessionIntelligenceOverviewTile({
  intelligence
}: {
  intelligence: SessionIntelligenceModel;
}) {
  const metadata = intelligence.metadata;
  const dataState =
    metadata.completion_status === "needs_attention"
      ? "error"
      : metadata.completion_status;

  return (
    <div
      aria-label="Session intelligence summary"
      className="overviewTile overviewSessionIntelligenceTile"
      data-state={dataState}
      role="group"
    >
      <span>Session</span>
      <strong>{intelligence.statusLabel}</strong>
      <p>{metadata.session_summary}</p>
      <small>{metadata.active_request_summary}</small>
      <div
        aria-label="Available metadata groups"
        className="sessionIntelligencePills"
      >
        {metadata.available_metadata_groups.length > 0 ? (
          metadata.available_metadata_groups.map((group) => (
            <span key={group}>{group}</span>
          ))
        ) : (
          <span>No metadata groups</span>
        )}
      </div>
      <div className="sessionIntelligenceMeta">
        <span>{`${intelligence.availableMetadataCount} metadata group${
          intelligence.availableMetadataCount === 1 ? "" : "s"
        }`}</span>
        <span>{`${intelligence.warningCount} warning${
          intelligence.warningCount === 1 ? "" : "s"
        }`}</span>
        <span>{intelligence.source === "stream" ? "Stream" : "Derived"}</span>
      </div>
      <div
        aria-label="Session warnings"
        className="sessionIntelligenceList"
      >
        {metadata.session_warnings.length > 0 ? (
          metadata.session_warnings.map((warning) => <p key={warning}>{warning}</p>)
        ) : (
          <p>No session warnings.</p>
        )}
      </div>
      <div
        aria-label="Recommended next user actions"
        className="sessionIntelligenceList"
      >
        {metadata.recommended_next_user_actions.length > 0 ? (
          metadata.recommended_next_user_actions.map((action) => (
            <p key={action}>{action}</p>
          ))
        ) : (
          <p>No recommended next action.</p>
        )}
      </div>
    </div>
  );
}

function CreativePlanOverviewTile({
  plan
}: {
  plan: NonNullable<AssistantWorkspaceSnapshot["creativePlan"]>;
}) {
  return (
    <div
      aria-label="Planning summary"
      className="overviewTile overviewPlanningTile"
      data-state={plan.exportReadiness}
      role="group"
    >
      <span>Planning</span>
      <strong>{formatPlanningHeadline(plan)}</strong>
      <p>{plan.generationStrategy}</p>
      <small>
        {`${plan.candidateCount} candidate${
          plan.candidateCount === 1 ? "" : "s"
        } / ${plan.refinementBudget} refinement pass${
          plan.refinementBudget === 1 ? "" : "es"
        } / ${plan.estimatedTokenCost} est. tokens`}
      </small>
    </div>
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
                {question.suggestedOptions.map((option) => (
                  <button
                    disabled={disabled}
                    key={option}
                    onClick={() => void onOptionSelect(option)}
                    type="button"
                  >
                    {option}
                  </button>
                ))}
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
  dashboard,
  preview,
  route,
  runtimeSource,
  showDebugPanels
}: {
  controller: PreviewControllerModel;
  dashboard: TelemetryDashboardModel;
  preview: AssistantWorkspaceSnapshot["preview"];
  route: PreviewRendererRoute;
  runtimeSource: PreviewRuntimeSource;
  showDebugPanels: boolean;
}) {
  if (!showDebugPanels) {
    return (
      <section
        aria-label="Preview inspector"
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
              {preview.state === "ready"
                ? "Visual output is ready in the preview shelf."
                : "Preview will open when a runnable visual artifact is ready."}
            </p>
          </div>
          <span>{preview.state === "ready" ? "Ready" : "Fallback"}</span>
        </article>
        <article className="previewInspectorCard previewInspectorCard--user" role="group">
          <header>
            <span>Visual target</span>
            <strong>{route.surfaceTitle}</strong>
          </header>
          <p>
            {preview.state === "ready"
              ? "Use the preview shelf for the visual canvas. Developer diagnostics stay hidden in User Mode."
              : "Continue with Code or Saved outputs if preview is not available yet."}
          </p>
        </article>
      </section>
    );
  }

  return (
    <section
      aria-label="Preview inspector"
      className="inspectorPanel previewInspectorPanel"
      data-state={preview.state}
      id="preview-inspector-panel"
      role="tabpanel"
    >
      <article
        aria-label="Preview canvas status"
        className="previewInspectorHero"
        data-state={preview.state}
        role="group"
      >
        <div>
          <span>Canvas runtime</span>
          <strong>{formatPreviewStateLabel(preview.state, preview.active)}</strong>
          <p>{preview.summary}</p>
        </div>
        <span>{controller.sessionLabel}</span>
      </article>

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
              <dt>Opened from</dt>
              <dd>{preview.trigger}</dd>
            </div>
          </dl>
        </article>

        <article
          aria-label="Preview source metadata"
          className="previewInspectorCard"
          role="group"
        >
          <header>
            <span>Executable source</span>
            <strong>{runtimeSource.title}</strong>
          </header>
          <dl>
            <div>
              <dt>Fingerprint</dt>
              <dd>{runtimeSource.fingerprint}</dd>
            </div>
            <div>
              <dt>Lines</dt>
              <dd>{runtimeSource.lineCount}</dd>
            </div>
            <div>
              <dt>Renderer</dt>
              <dd>{route.rendererId ?? "Pending renderer"}</dd>
            </div>
            <div>
              <dt>Health</dt>
              <dd>{dashboard.preview.healthLabel}</dd>
            </div>
          </dl>
          <p>{dashboard.preview.detail}</p>
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
  onArtifactTransfer: (artifact: ArtifactSummary) => void;
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
  const transferAction = getArtifactTransferAction(artifact.actions);
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
          {transferAction ? (
            <button
              aria-label={`${
                showDebugPanels
                  ? formatArtifactActionLabel(transferAction, artifact)
                  : formatUserArtifactActionLabel(transferAction)
              } ${displayDocumentName}`}
              onClick={() => onArtifactTransfer(artifact)}
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
          ) : null}
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

function ProvenanceSummaryCard({
  provenance
}: {
  provenance: ProvenanceEngineModel;
}) {
  return (
    <article
      aria-label="Provenance summary"
      className="workflowProvenanceCard"
      role="group"
    >
      <header>
        <div>
          <span>Provenance</span>
          <strong>Source trace</strong>
          <p>{provenance.provenance_summary}</p>
        </div>
      </header>
      <div
        aria-label="Provenance source counts"
        className="workflowProvenanceMetrics"
        role="group"
      >
        <span>{`${provenance.evidence_sources.length} evidence`}</span>
        <span>{`${provenance.dependency_sources.length} dependencies`}</span>
        <span>{`${provenance.artifact_sources.length} artifacts`}</span>
        <span>{`${provenance.evaluation_sources.length} evaluation/final`}</span>
        <span>{`${provenance.unsupported_or_missing_sources.length} missing`}</span>
      </div>
      <div className="workflowProvenanceLists">
        <ProvenanceSourceList
          label="Evidence sources"
          sources={provenance.evidence_sources}
        />
        <ProvenanceSourceList
          label="Artifact sources"
          sources={provenance.artifact_sources}
        />
        <ProvenanceSourceList
          label="Unsupported or missing sources"
          sources={provenance.unsupported_or_missing_sources}
        />
      </div>
    </article>
  );
}

function ProvenanceSourceList({
  label,
  sources
}: {
  label: string;
  sources: ProvenanceEngineModel["evidence_sources"];
}) {
  const visibleSources = sources.slice(0, 3);

  return (
    <section aria-label={label} className="workflowProvenanceList">
      <span>{label}</span>
      {visibleSources.length > 0 ? (
        visibleSources.map((source) => (
          <p key={source.id}>
            <strong>{source.label}</strong>
            {` / ${source.summary}`}
          </p>
        ))
      ) : (
        <p>No sources captured.</p>
      )}
    </section>
  );
}

function WorkflowInspector({
  creativeTimeline,
  explorer,
  issues,
  provenance,
  runtime,
  v3InspectorPanels,
  telemetry,
  showDebugPanels
}: {
  creativeTimeline: CreativeTimelineModel;
  explorer: WorkflowExplorerModel;
  issues: WorkstationError[];
  provenance: ProvenanceEngineModel;
  runtime: WorkflowRuntimeModel;
  v3InspectorPanels: V3InspectorPanelsModel;
  telemetry: ProviderTelemetryModel;
  showDebugPanels: boolean;
}) {
  const workflowProgress = getWorkflowRuntimeProgress(runtime.steps);
  const recentEvents = runtime.events.slice(-6).reverse();

  return (
    <section
      aria-label="Workflow inspector"
      className="inspectorPanel workflowPanel"
      id="workflow-inspector-panel"
      role="tabpanel"
    >
      {issues.length > 0 ? (
        <div className="workflowIssueStack" aria-label="Workflow runtime issues">
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
      <WorkflowProgress
        label="Workflow inspector progress"
        progress={workflowProgress}
      />
      <div className="workflowSummaryGrid" aria-label="Workflow execution summary">
        <article className="workflowSummaryCard" role="group" aria-label="Workflow status">
          <span>Status</span>
          <strong>{formatWorkflowStatusCopy(runtime.summary.status)}</strong>
          <p>{runtime.summary.currentStep}</p>
        </article>
        <article className="workflowSummaryCard" role="group" aria-label="Workflow runtime">
          <span>Runtime</span>
          <strong>{formatRuntimeDuration(runtime.summary.totalRuntimeMs)}</strong>
          <p>
            {runtime.summary.activeRuntimeMs != null
              ? `Active ${formatRuntimeDuration(runtime.summary.activeRuntimeMs)}`
              : "Awaiting next transition"}
          </p>
        </article>
        <article className="workflowSummaryCard" role="group" aria-label="Workflow retries">
          <span>Retries</span>
          <strong>{runtime.summary.retryCount}</strong>
          <p>{formatRetryCount(runtime.summary.retryCount)}</p>
        </article>
        <article
          className="workflowSummaryCard"
          role="group"
          aria-label="Workflow transitions"
        >
          <span>Transitions</span>
          <strong>{runtime.summary.transitionCount}</strong>
          <p>{runtime.summary.traceEventCount} streamed events</p>
        </article>
        <article
          className="workflowSummaryCard"
          role="group"
          aria-label="Workflow token usage"
        >
          <span>Tokens</span>
          <strong>{formatTokenUsageTotal(telemetry)}</strong>
          <p>{formatTokenUsageDetail(telemetry)}</p>
        </article>
        <article
          className="workflowSummaryCard"
          role="group"
          aria-label="Workflow cost estimate"
        >
          <span>Cost</span>
          <strong>{telemetry.summary.costLabel}</strong>
          <p>{formatTelemetryCostSource(telemetry)}</p>
        </article>
      </div>
      <WorkflowExplorerSurface model={explorer} />
      <ProvenanceSummaryCard provenance={provenance} />
      <CreativeTimelineSurface timeline={creativeTimeline} />
      <V3InspectorPanelsSurface model={v3InspectorPanels} />
      <TelemetryLifecycleCard telemetry={telemetry} />
      <WorkflowTimelineExplorer timeline={runtime.timeline} />
      <div
        aria-label="LangGraph workflow visualization"
        className="workflowGraph"
        role="group"
      >
        {runtime.steps.map((step, index) => (
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
              <p>
                <code>{step.nodeId}</code>
                <span>{formatWorkflowRuntimeState(step.state)}</span>
              </p>
              <small>{step.lastEventDetail ?? step.detail}</small>
            </div>
            <div className="workflowNodeMeta">
              <span>{formatRuntimeDuration(step.durationMs)}</span>
              <span>{formatAttemptMeta(step.attemptCount)}</span>
            </div>
          </article>
        ))}
      </div>
      {showDebugPanels ? (
        <div className="workflowTraceLayout">
          <article
            className="workflowTraceCard"
            role="group"
            aria-label="Workflow transition trace"
          >
            <header>
              <strong>Transitions</strong>
              <span>{runtime.summary.transitionCount}</span>
            </header>
            <div className="workflowTraceList">
              {runtime.transitions.length > 0 ? (
                runtime.transitions.map((transition) => (
                  <article
                    className="workflowTraceItem"
                    data-kind={transition.kind}
                    key={`${transition.sequence}-${transition.label}`}
                  >
                    <strong>{transition.label}</strong>
                    <p>
                      <span>{transition.sequence}</span>
                      <span>{formatTraceTime(transition.at)}</span>
                      {transition.reason ? (
                        <span>{formatRuntimeCode(transition.reason)}</span>
                      ) : null}
                    </p>
                  </article>
                ))
              ) : (
                <p className="workflowTraceEmpty">No runtime transitions recorded yet.</p>
              )}
            </div>
          </article>
          <article
            className="workflowTraceCard"
            role="group"
            aria-label="Workflow event trace"
          >
            <header>
              <strong>Event trace</strong>
              <span>{runtime.summary.traceEventCount}</span>
            </header>
            <div className="workflowTraceList">
              {recentEvents.length > 0 ? (
                recentEvents.map((event) => (
                  <article
                    className="workflowTraceItem"
                    data-kind={event.phase ?? "running"}
                    key={`${event.sequence}-${event.label}`}
                  >
                    <strong>{event.label}</strong>
                    <p>
                      <span>{event.nodeId ?? "runtime"}</span>
                      <span>{formatTraceTime(event.at)}</span>
                    </p>
                    <small>{event.detail}</small>
                  </article>
                ))
              ) : (
                <p className="workflowTraceEmpty">No streamed workflow events yet.</p>
              )}
            </div>
          </article>
        </div>
      ) : (
        <article
          className="workflowTraceCard workflowTraceCard--muted"
          role="group"
          aria-label="Workflow traces hidden"
        >
          <header>
            <strong>Advanced traces</strong>
            <span>Off</span>
          </header>
          <p className="workflowTraceEmpty">
            Workflow trace panels are hidden in Settings.
          </p>
        </article>
      )}
    </section>
  );
}

function TelemetryLifecycleCard({
  telemetry
}: {
  telemetry: ProviderTelemetryModel;
}) {
  return (
    <article
      className="telemetryLifecycleCard"
      role="group"
      aria-label="Generation telemetry lifecycle"
    >
      <header>
        <div>
          <span>Generation telemetry</span>
          <strong>{formatProviderRuntimeLabel(telemetry)}</strong>
          <p>{telemetry.summary.lifecycleLabel}</p>
        </div>
        <span className="telemetryStateBadge" data-state={telemetry.status}>
          {formatTelemetryStatus(telemetry.status)}
        </span>
      </header>
      <div className="telemetryMetricRow" aria-label="Telemetry timing summary">
        <div>
          <span>First token</span>
          <strong>{formatRuntimeDuration(telemetry.timing.timeToFirstTokenMs)}</strong>
        </div>
        <div>
          <span>Generation</span>
          <strong>{formatRuntimeDuration(telemetry.timing.generationDurationMs)}</strong>
        </div>
        <div>
          <span>Stream</span>
          <strong>{telemetry.summary.streamLabel}</strong>
        </div>
      </div>
      <div className="telemetryLifecycleSteps" aria-label="Generation lifecycle stages">
        {telemetry.lifecycle.map((step) => (
          <div
            className="telemetryLifecycleStep"
            data-state={step.state}
            key={step.id}
          >
            <span aria-hidden="true" />
            <div>
              <strong>{step.label}</strong>
              <small>{formatTelemetryLifecycleStep(step)}</small>
            </div>
          </div>
        ))}
      </div>
    </article>
  );
}

function TelemetryInspector({
  dashboard,
  showDebugPanels
}: {
  dashboard: TelemetryDashboardModel;
  showDebugPanels: boolean;
}) {
  const populatedEventTypes = Object.entries(dashboard.stream.eventTypeCounts).filter(
    ([, count]) => count > 0
  );

  return (
    <section
      aria-label="Telemetry dashboard"
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

      <div className="telemetryDashboardGrid">
        <article
          aria-label="Stream lifecycle"
          className="telemetryDashboardCard"
          data-state={dashboard.stream.state}
          role="group"
        >
          <header>
            <span>Stream lifecycle</span>
            <strong>{formatDashboardStatusLabel(dashboard.stream.state)}</strong>
          </header>
          <dl>
            <div>
              <dt>Events</dt>
              <dd>{dashboard.stream.eventCount}</dd>
            </div>
            <div>
              <dt>Errors</dt>
              <dd>{dashboard.stream.errorCount}</dd>
            </div>
            <div>
              <dt>Preview</dt>
              <dd>{dashboard.stream.previewEventCount}</dd>
            </div>
            <div>
              <dt>Eval</dt>
              <dd>{dashboard.stream.evalEventCount}</dd>
            </div>
          </dl>
          <p>{dashboard.stream.latestEventLabel}</p>
          <small>{formatNullableTraceTime(dashboard.stream.latestEventAt)}</small>
        </article>

        <ProviderObservabilityDeepDive telemetry={dashboard.provider} />

        <CreativeCostIntelligenceDashboard
          intelligence={dashboard.creativeCost}
        />

        <article
          aria-label="Runtime lifecycle"
          className="telemetryDashboardCard"
          data-state={dashboard.runtime.workflowStatus}
          role="group"
        >
          <header>
            <span>Runtime</span>
            <strong>{dashboard.runtime.currentStep}</strong>
          </header>
          <dl>
            <div>
              <dt>Nodes</dt>
              <dd>{`${dashboard.runtime.reachedNodes}/${dashboard.runtime.totalNodes}`}</dd>
            </div>
            <div>
              <dt>Transitions</dt>
              <dd>{dashboard.runtime.transitionCount}</dd>
            </div>
            <div>
              <dt>Retries</dt>
              <dd>{dashboard.runtime.retryCount}</dd>
            </div>
            <div>
              <dt>Runtime</dt>
              <dd>{formatRuntimeDuration(dashboard.runtime.totalRuntimeMs)}</dd>
            </div>
          </dl>
          <p>{formatWorkflowStatusCopy(dashboard.runtime.workflowStatus)}</p>
          <small>
            {dashboard.runtime.activeRuntimeMs != null
              ? `Active ${formatRuntimeDuration(dashboard.runtime.activeRuntimeMs)}`
              : "No active node timing"}
          </small>
        </article>

        <article
          aria-label="Renderer and preview health"
          className="telemetryDashboardCard"
          data-state={dashboard.preview.state}
          role="group"
        >
          <header>
            <span>Preview runtime</span>
            <strong>{dashboard.preview.healthLabel}</strong>
          </header>
          <dl>
            <div>
              <dt>Renderer</dt>
              <dd>{dashboard.preview.renderer}</dd>
            </div>
            <div>
              <dt>Target</dt>
              <dd>{dashboard.preview.target}</dd>
            </div>
          </dl>
          <p>{dashboard.preview.detail}</p>
          <small>
            {dashboard.preview.error ??
              `Latest preview event ${formatNullableTraceTime(dashboard.preview.latestPreviewEventAt)}`}
          </small>
        </article>

        <article
          aria-label="Retrieval activity"
          className="telemetryDashboardCard"
          data-state={dashboard.retrieval.state}
          role="group"
        >
          <header>
            <span>Retrieval</span>
            <strong>{dashboard.retrieval.status}</strong>
          </header>
          <dl>
            <div>
              <dt>Sources</dt>
              <dd>{dashboard.retrieval.sourceCount}</dd>
            </div>
            <div>
              <dt>Chunks</dt>
              <dd>{dashboard.retrieval.chunkCount}</dd>
            </div>
            <div>
              <dt>Quality</dt>
              <dd>{dashboard.retrieval.qualityLabel}</dd>
            </div>
            <div>
              <dt>Freshness</dt>
              <dd>{dashboard.retrieval.freshnessLabel}</dd>
            </div>
          </dl>
          <p>{dashboard.retrieval.query ?? "No retrieval query captured yet."}</p>
          <small>
            {dashboard.retrieval.error ??
              dashboard.retrieval.warning ??
              dashboard.retrieval.providerLabel}
          </small>
        </article>

        <LangSmithTraceDeepDive trace={dashboard.langsmithTrace} />

        <EvaluationSessionDashboard evaluation={dashboard.evaluation} />

        <article
          aria-label="Artifact runtime linkage"
          className="telemetryDashboardCard telemetryDashboardCard--wide"
          data-state={dashboard.preview.state}
          role="group"
        >
          <header>
            <span>Artifact linkage</span>
            <strong>{dashboard.artifactLink.linkLabel}</strong>
          </header>
          <dl>
            <div>
              <dt>Active</dt>
              <dd>{dashboard.artifactLink.activeArtifactTitle}</dd>
            </div>
            <div>
              <dt>Preview source</dt>
              <dd>{dashboard.artifactLink.previewArtifactId ?? "None"}</dd>
            </div>
            <div>
              <dt>Renderer</dt>
              <dd>{dashboard.artifactLink.renderer}</dd>
            </div>
            <div>
              <dt>Status</dt>
              <dd>{dashboard.artifactLink.status}</dd>
            </div>
          </dl>
          <p>{dashboard.artifactLink.target}</p>
        </article>
      </div>

      {showDebugPanels ? (
        <article
          aria-label="Telemetry event type counts"
          className="telemetryDashboardCard telemetryDashboardCard--wide"
          role="group"
        >
          <header>
            <span>Event distribution</span>
            <strong>{dashboard.stream.eventCount} events</strong>
          </header>
          <div className="telemetryEventPills">
            {populatedEventTypes.length > 0 ? (
              populatedEventTypes.map(([eventType, count]) => (
                <span key={eventType}>
                  {formatRuntimeCode(eventType)}
                  <strong>{count}</strong>
                </span>
              ))
            ) : (
              <p>No stream events captured yet.</p>
            )}
          </div>
        </article>
      ) : (
        <article
          aria-label="Telemetry debug panels hidden"
          className="telemetryDashboardCard telemetryDashboardCard--wide"
          role="group"
        >
          <header>
            <span>Event distribution</span>
            <strong>Hidden</strong>
          </header>
          <p>Raw event counts are hidden in Settings.</p>
        </article>
      )}
    </section>
  );
}

type ArtifactsInspectorProps = {
  activeArtifact: ArtifactSummary;
  activeArtifactDocument: ArtifactDocument;
  activeArtifactId: string;
  artifacts: ArtifactSummary[];
  artifactTransferError: WorkstationError | null;
  code: AssistantWorkspaceSnapshot["code"];
  copyFeedback: ArtifactActionFeedback | null;
  isStreaming: boolean;
  onArtifactAction: (action: ArtifactAction, artifact: ArtifactSummary) => void;
  onArtifactRefine: (artifact: ArtifactSummary, instruction: string) => Promise<void>;
  onArtifactSelect: (artifact: ArtifactSummary) => void;
  preview: AssistantWorkspaceSnapshot["preview"];
  showDebugPanels: boolean;
  transferFeedback: ArtifactActionFeedback | null;
};

function ArtifactsInspector({
  activeArtifact,
  activeArtifactDocument,
  activeArtifactId,
  artifacts,
  artifactTransferError,
  code,
  copyFeedback,
  isStreaming,
  onArtifactAction,
  onArtifactRefine,
  onArtifactSelect,
  preview,
  showDebugPanels,
  transferFeedback
}: ArtifactsInspectorProps) {
  const actionMessage = getArtifactActionMessage(
    activeArtifact,
    copyFeedback,
    transferFeedback
  );
  const comparison = buildMultiPreviewComparisonModel({
    activeArtifactId,
    artifacts,
    code,
    preview
  });

  if (!showDebugPanels) {
    return (
      <UserArtifactsInspector
        activeArtifact={activeArtifact}
        activeArtifactId={activeArtifactId}
        artifacts={artifacts}
        copyFeedback={copyFeedback}
        onArtifactAction={onArtifactAction}
        onArtifactSelect={onArtifactSelect}
        transferFeedback={transferFeedback}
      />
    );
  }

  return (
    <section
      aria-label="Artifacts inspector"
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
      <MultiPreviewComparisonWorkspace
        comparison={comparison}
        onArtifactAction={onArtifactAction}
        onArtifactSelect={onArtifactSelect}
      />
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
            <dd>{activeArtifact.status}</dd>
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
        {activeArtifact.critique ? (
          <ArtifactCritiqueSummaryCard artifact={activeArtifact} />
        ) : null}
        <ArtifactPlanSummaryCard artifact={activeArtifact} />
        <CreativeTranslationSummaryCard
          translation={activeArtifact.creativeTranslation}
        />
        <AudioReactiveMappingSummaryCard
          translation={activeArtifact.creativeTranslation}
        />
        {activeArtifact.type === "code" && activeArtifact.actions.length > 0 ? (
          <ArtifactRefinementPanel
            artifact={activeArtifact}
            disabled={isStreaming}
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
  onArtifactAction,
  onArtifactSelect,
  transferFeedback
}: {
  activeArtifact: ArtifactSummary;
  activeArtifactId: string;
  artifacts: ArtifactSummary[];
  copyFeedback: ArtifactActionFeedback | null;
  onArtifactAction: (action: ArtifactAction, artifact: ArtifactSummary) => void;
  onArtifactSelect: (artifact: ArtifactSummary) => void;
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
      <UserArtifactActionRow
        artifact={activeArtifact}
        copyFeedback={copyFeedback}
        onArtifactAction={onArtifactAction}
        transferFeedback={transferFeedback}
      />
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

type CommandMenuPanelProps = {
  activeTab: InspectorTabName;
  hasBlockingApproval: boolean;
  isFocusMode: boolean;
  isPreviewAvailable: boolean;
  isPreviewOpen: boolean;
  onFocusModeToggle: () => void;
  onOpenTab: (tab: InspectorTabName) => void;
  onPreviewToggle: () => void;
  onWorkspaceClear: () => void;
  showDebugPanels: boolean;
};

function KnowledgeBaseStatusPanel({
  runtime
}: {
  runtime: RetrievalRuntimeModel;
}) {
  return (
    <section
      aria-label="Knowledge Base"
      className="utilityPanel utilityPanel--kb"
      id="kb-status-panel"
      role="dialog"
    >
      <header className="utilityPanelHeader">
        <strong>Knowledge Base</strong>
        <p>Current retrieval state and reported local KB coverage.</p>
      </header>
      <RetrievalRunStatusSurface runtime={runtime} />
      <KnowledgeBaseStatusSurface runtime={runtime} />
    </section>
  );
}

function CommandMenuPanel({
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
}: CommandMenuPanelProps) {
  return (
    <section
      aria-label="Quick actions"
      className="utilityPanel utilityPanel--menu"
      id="command-menu-panel"
      role="dialog"
    >
      <header className="utilityPanelHeader">
        <strong>Quick actions</strong>
        <p>Jump to the next workspace surface without changing the current flow.</p>
      </header>
      <div className="commandMenuGrid">
        {showDebugPanels ? (
          <button
            data-active={activeTab === "Overview"}
            onClick={() => onOpenTab("Overview")}
            type="button"
          >
            <strong>Overview</strong>
            <span>Return to the compact session summary.</span>
          </button>
        ) : null}
        <button
          data-active={activeTab === "Preview"}
          onClick={() => onOpenTab("Preview")}
          type="button"
        >
          <strong>Preview</strong>
          <span>Inspect the current visual output and preview readiness.</span>
        </button>
        {showDebugPanels ? (
          <button
            data-active={activeTab === "Runtime"}
            onClick={() => onOpenTab("Runtime")}
            type="button"
          >
            <strong>Runtime console</strong>
            <span>Inspect live runtime status, FPS, reloads, and renderer errors.</span>
          </button>
        ) : null}
        <button
          data-active={activeTab === "Artifacts"}
          onClick={() => onOpenTab("Artifacts")}
          type="button"
        >
          <strong>{showDebugPanels ? "Artifacts" : "Saved"}</strong>
          <span>Inspect generated and saved results.</span>
        </button>
        <button
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
              data-active={activeTab === "Workflow"}
              onClick={() => onOpenTab("Workflow")}
              type="button"
            >
              <strong>Workflow inspector</strong>
              <span>Review the live orchestration runtime.</span>
            </button>
            <button
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
          aria-label="Toggle focus mode from quick actions"
          onClick={onFocusModeToggle}
          type="button"
        >
          <strong>{isFocusMode ? "Exit focus mode" : "Enter focus mode"}</strong>
          <span>Hide or restore the surrounding workspace chrome.</span>
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
    </section>
  );
}

function ThemePresetsPanel({
  activeTheme,
  onSelectTheme
}: {
  activeTheme: WorkspacePreferences["theme"];
  onSelectTheme: (theme: WorkspacePreferences["theme"]) => void;
}) {
  return (
    <section
      aria-label="Theme presets"
      className="utilityPanel utilityPanel--theme"
      id="theme-presets-panel"
      role="dialog"
    >
      <header className="utilityPanelHeader">
        <strong>Theme presets</strong>
        <p>Switch the workspace accent and shell tone without changing the layout.</p>
      </header>
      <ThemePresetPicker activeTheme={activeTheme} onSelectTheme={onSelectTheme} />
    </section>
  );
}

type WorkspaceSettingsPanelProps = {
  layoutState: WorkspaceLayoutState;
  onDensityChange: (density: WorkspaceLayoutState["density"]) => void;
  onPreferencesChange: (preferences: Partial<WorkspacePreferences>) => void;
  preferences: WorkspacePreferences;
};

function WorkspaceSettingsPanel({
  layoutState,
  onDensityChange,
  onPreferencesChange,
  preferences
}: WorkspaceSettingsPanelProps) {
  return (
    <section
      aria-label="Workspace settings"
      className="utilityPanel utilityPanel--settings"
      id="workspace-settings-panel"
      role="dialog"
    >
      <header className="utilityPanelHeader">
        <strong>Workspace settings</strong>
        <p>Lightweight preferences are restored with the current local workspace session.</p>
      </header>
      <div className="settingsSection">
        <div className="settingsSectionHeader">
          <strong>Theme</strong>
          <p>Pick the overall shell character for this session.</p>
        </div>
        <ThemePresetPicker
          activeTheme={preferences.theme}
          compact
          onSelectTheme={(theme) => onPreferencesChange({ theme })}
        />
      </div>
      <div className="settingsSection">
        <div className="settingsSectionHeader">
          <strong>Workspace</strong>
          <p>Density is persisted alongside the current layout widths and shelf sizing.</p>
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
      </div>
      <div className="settingsSection">
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
    </section>
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

function ArtifactCritiqueSummaryCard({
  artifact
}: {
  artifact: ArtifactSummary;
}) {
  const critique = artifact.critique;
  if (!critique) {
    return null;
  }

  const dimensions = [
    ["Prompt", critique.promptAlignment.score],
    ["Creative", critique.creativeQuality.score],
    ["Runtime", critique.runtimeSuitability.score],
    ["Code", critique.codeQuality.score],
    ["Preview", critique.previewReadiness.score],
    ["Domain", critique.domainAppropriateness.score]
  ] as const;

  return (
    <section
      aria-label="Artifact quality summary"
      className="artifactCritiqueCard"
    >
      <header>
        <div>
          <span>Artifact critique</span>
          <strong>
            {artifact.isRecommended
              ? "Recommended candidate"
              : critique.passed
                ? "Quality gate passed"
                : "Refinement advised"}
          </strong>
        </div>
        <span className="artifactQualityBadge">
          {formatQualityScore(critique.overallScore)}
        </span>
      </header>
      <p>{critique.rationale}</p>
      {critique.refinementGuidance ? (
        <p className="artifactRefinementReason">{critique.refinementGuidance}</p>
      ) : null}
      <div className="artifactCritiqueDimensions" aria-label="Critique dimensions">
        {dimensions.map(([label, score]) => (
          <span key={label}>{`${label} ${formatQualityScore(score)}`}</span>
        ))}
      </div>
      <CalibratedQualitySummary
        evaluation={critique.calibratedQuality}
      />
      <CreativeQualityCriticSummary
        evaluation={critique.creativeEvaluation}
      />
      <SacredConsistencySummary
        evaluation={critique.sacredConsistency}
      />
    </section>
  );
}

function ArtifactPlanSummaryCard({
  artifact
}: {
  artifact: ArtifactSummary;
}) {
  const plan = artifact.creativePlan;
  if (!plan) {
    return null;
  }

  return (
    <section
      aria-label="Artifact planning summary"
      className="artifactPlanCard"
    >
      <header>
        <div>
          <span>Execution plan</span>
          <strong>{formatPlanningHeadline(plan)}</strong>
        </div>
        <span className="artifactQualityBadge">
          {formatExportReadiness(plan.exportReadiness)}
        </span>
      </header>
      <p>{plan.generationStrategy}</p>
      <dl className="artifactPlanMeta">
        <div>
          <dt>Runtime</dt>
          <dd>{plan.recommendedRuntime ?? "Code-only"}</dd>
        </div>
        <div>
          <dt>Candidates</dt>
          <dd>{plan.candidateCount}</dd>
        </div>
        <div>
          <dt>Refinement</dt>
          <dd>{plan.refinementBudget}</dd>
        </div>
        <div>
          <dt>Complexity</dt>
          <dd>{formatPlanningComplexity(plan.expectedComplexity)}</dd>
        </div>
      </dl>
      {plan.planSteps.length > 0 ? (
        <ul className="artifactPlanSteps">
          {plan.planSteps.slice(0, 4).map((step) => (
            <li key={step}>{step}</li>
          ))}
        </ul>
      ) : null}
    </section>
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
  state: ArtifactActionFeedback["state"],
  timerRef: { current: number | null },
  setFeedback: (feedback: ArtifactActionFeedback | null) => void
) {
  clearTimer(timerRef.current);
  setFeedback({ artifactId, state });
  timerRef.current = window.setTimeout(() => {
    setFeedback(null);
    timerRef.current = null;
  }, artifactFeedbackDurationMs);
}

function getArtifactTransferAction(
  actions: ArtifactAction[]
): ArtifactTransferAction | null {
  if (actions.includes("Download")) {
    return "Download";
  }

  if (actions.includes("Export")) {
    return "Export";
  }

  return null;
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

  if (action === "Download" || action === "Export") {
    return "Save";
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
    transferFeedback?.artifactId === artifact.id
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

function formatPlanningHeadline(
  plan: NonNullable<AssistantWorkspaceSnapshot["creativePlan"]>
) {
  const runtime = plan.recommendedRuntime ?? "code-only";
  return `${formatOutputModality(plan.outputModality)} / ${runtime}`;
}

function formatOutputModality(value: string) {
  if (value === "audiovisual") {
    return "Audiovisual";
  }
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function formatPlanningComplexity(value: string) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function formatExportReadiness(value: string) {
  if (value === "ready") {
    return "Export ready";
  }
  if (value === "blocked") {
    return "Blocked";
  }
  return "Partially ready";
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
    transferFeedback?.artifactId === artifact.id
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
    const transferAction = getArtifactTransferAction(artifact.actions);
    const isBundleExport = isProjectBundleExportArtifact(artifact);
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
  const refinedArtifactTitle = idCollidesWithSource
    ? createRefinedArtifactTitle(hydration.artifact.title)
    : hydration.artifact.title;
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
  const preview =
    hydration.snapshot.preview.sourceArtifactId === hydration.artifact.id
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
      hydration.previewArtifactId === hydration.artifact.id
        ? refinedArtifact.id
        : hydration.previewArtifactId,
    previewAvailable: hydration.previewAvailable,
    snapshot: {
      ...hydration.snapshot,
      artifacts: nextArtifacts,
      code: buildCodeSummaryForArtifact(hydration.snapshot.code, refinedArtifact),
      preview
    }
  };
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

function createRefinedArtifactTitle(title: string) {
  const extensionIndex = title.lastIndexOf(".");

  if (extensionIndex <= 0) {
    return `${title}-refined`;
  }

  return `${title.slice(0, extensionIndex)}.refined${title.slice(extensionIndex)}`;
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

function createArtifactTransferError(artifact: ArtifactSummary) {
  if (isProjectBundleExportArtifact(artifact)) {
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

  const transferAction = getArtifactTransferAction(artifact.actions) ?? "Export";
  const actionLabel = transferAction === "Export" ? "export" : "download";

  return createWorkstationError({
    type: transferAction === "Export" ? "artifact_export_failed" : "artifact_download_failed",
    category: "artifact_export",
    subsystem: "artifact_transfer",
    userMessage: `The workspace could not ${actionLabel} ${artifact.title}.`,
    recoverable: true,
    suggestedAction:
      "Retry the transfer from the Artifacts tab or continue working in the current session.",
    retryLabel: transferAction === "Export" ? "Retry export" : "Retry download"
  });
}

function getArtifactTransferApprovalActionId(
  artifact: ArtifactSummary
): HitlActionId {
  if (isProjectBundleExportArtifact(artifact)) {
    return "project_bundle_export";
  }

  return artifact.actions.includes("Download")
    ? "artifact_download"
    : "artifact_export";
}

function isProjectBundleExportArtifact(artifact: ArtifactSummary) {
  return artifact.type === "export" && artifact.actions.includes("Export");
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

function getInitialActiveTab(snapshot: AssistantWorkspaceSnapshot): InspectorTabName {
  return snapshot.inspectorTabs.find((tab) => tab.active)?.label ?? "Overview";
}

function formatInspectorTabDisplayLabel(
  tab: InspectorTabName,
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
    status: boundedProgressIndex >= finalizationIndex ? "Complete" : "Running",
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
      return "Completed";
    case "completed_with_preview_error":
      return "Completed with preview error";
    case "failed":
      return "Failed";
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

function formatSessionTelemetryLabel(telemetry: ProviderTelemetryModel) {
  if (telemetry.status === "idle") {
    return "Telemetry pending";
  }

  return `${telemetry.summary.providerLabel} / ${telemetry.summary.tokenLabel}`;
}

function formatUserModeSessionStatus({
  hasFailedPreviewRuntime,
  hasWorkspaceArtifacts,
  isDemoModeOpen,
  streamError,
  streamState
}: {
  hasFailedPreviewRuntime: boolean;
  hasWorkspaceArtifacts: boolean;
  isDemoModeOpen: boolean;
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
        label: "Working",
        detail: "Generating response"
      };
    case "fallback":
      return {
        label: "Needs attention",
        detail: streamError ? "Live response unavailable" : "Fallback available"
      };
    default:
      if (hasFailedPreviewRuntime) {
        return {
          label: "Needs attention",
          detail: "Preview needs attention"
        };
      }

      if (hasWorkspaceArtifacts) {
        return {
          label: "Complete",
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
      return "Complete";
    case "error":
      return "Error";
    case "streaming":
      return "Streaming";
    default:
      return "Idle";
  }
}

function formatDashboardStatusLabel(status: TelemetryDashboardModel["status"]) {
  switch (status) {
    case "complete":
      return "Complete";
    case "degraded":
      return "Degraded";
    case "error":
      return "Error";
    case "running":
      return "Running";
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

function formatMessageTime() {
  return new Intl.DateTimeFormat("en-US", {
    hour: "2-digit",
    hour12: false,
    minute: "2-digit"
  }).format(new Date());
}

const streamingConversationSummary =
  "Generating response. Code and long-form output will appear in the Code panel, artifacts, and preview surfaces when the run completes.";

const generatedCodePattern =
  /```|<!doctype|<html|<script|function\s+(setup|draw)\s*\(|import\s+\*\s+as\s+THREE|gl_FragColor|void\s+main\s*\(/i;

function getConversationDisplayContent(
  message: ConversationEntry,
  showDebugPanels: boolean
) {
  if (message.role !== "assistant" || showDebugPanels) {
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
  const strippedContent = trimmedContent
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<!doctype[\s\S]*?<body[^>]*>/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  if (containsGeneratedCode) {
    const summary = strippedContent
      ? truncateConversationSummary(strippedContent, 220)
      : "Generated code is ready.";

    return `${summary}\n\nCode and long-form output are in Code, Artifacts, and Preview.`;
  }

  if (strippedContent.length > 520) {
    return `${truncateConversationSummary(
      strippedContent,
      360
    )}\n\nOpen Developer Mode for the full response details.`;
  }

  return strippedContent;
}

function buildAssistantConversationSummary(answer: string) {
  const trimmedAnswer = answer.trim();

  if (!trimmedAnswer) {
    return "Response completed. Check Code, Preview, and Retrieval for output details.";
  }

  const lineCount = trimmedAnswer.split(/\r?\n/).length;
  const shouldSummarize =
    generatedCodePattern.test(trimmedAnswer) ||
    trimmedAnswer.length > 900 ||
    lineCount > 14;

  if (!shouldSummarize) {
    return trimmedAnswer;
  }

  const textOnly = trimmedAnswer
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ");
  const summary =
    textOnly
      .split(/\r?\n+/)
      .map((line) => line.replace(/\s+/g, " ").trim())
      .find((line) => line.length > 0 && !/^file(name)?:/i.test(line)) ??
    "The assistant generated a reviewable creative-coding output.";

  return `${truncateConversationSummary(
    summary,
    240
  )}\n\nCode and long-form output are available in the Code panel, artifacts, and preview surfaces. Next: inspect Preview, Code, and Retrieval evidence.`;
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
