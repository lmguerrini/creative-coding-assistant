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
  Gauge,
  ImagePlus,
  LayoutGrid,
  Maximize2,
  Minimize2,
  Moon,
  PanelRight,
  Play,
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
  ImageAttachmentSummary,
  InspectorTabName,
  WorkflowStepState
} from "@/lib/assistant-client";
import {
  readStreamEventError,
  readPreviewArtifactUpdate,
  streamAssistantEvents as streamBackendAssistantEvents,
  workflowNodeFromAssistantStreamEvent,
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
import { buildProjectBundle } from "@/lib/project-bundle";
import {
  buildWorkflowRuntimeModel,
  type WorkflowRuntimeModel,
  type WorkflowRuntimeTraceEvent,
  type WorkflowRuntimeVisualState
} from "@/lib/workflow-runtime";
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
  type PreviewRendererRoute
} from "@/lib/preview-renderers";
import {
  buildPreviewRuntimeSummary,
  isArtifactPreviewable
} from "@/lib/preview-runtime";
import {
  hydrateWorkspaceFromArtifactExtractedEvent,
  hydrateWorkspaceFromFinalEvent
} from "@/lib/live-artifact-hydration";
import {
  buildProviderTelemetryModel,
  type ProviderTelemetryLifecycleStep,
  type ProviderTelemetryModel
} from "@/lib/provider-telemetry";
import {
  buildTelemetryDashboardModel,
  type TelemetryDashboardModel
} from "@/lib/telemetry-dashboard";
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
  createWorkstationError,
  type WorkstationError
} from "@/lib/workstation-errors";
import { buildZipArchive, downloadZipArchive } from "@/lib/zip-archive";
import { PreviewRendererSurface } from "./preview-renderer-surface";
import { SubsystemErrorCallout } from "./subsystem-error-callout";

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
type UtilityPanelName = "commands" | "theme" | "settings";
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

const mockWorkflowIntervalMs = 850;
const artifactFeedbackDurationMs = 1400;
const defaultWorkspacePersistenceClient = createWorkspacePersistenceClient();
const persistenceStateLabels = {
  loading: "Restoring session",
  ready: "Local session ready",
  restored: "Session restored",
  saving: "Saving session",
  saved: "Session saved",
  local: "Saved locally",
  unavailable: "Persistence offline"
} satisfies Record<WorkspacePersistenceState, string>;
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
    description: "Focused command-console shell with warm phosphor highlights.",
    accent: "#f1c56f",
    surface:
      "linear-gradient(135deg, rgba(241, 197, 111, 0.18), rgba(86, 74, 42, 0.14), rgba(9, 10, 11, 0.28))"
  },
  {
    value: "horizon",
    label: "Horizon",
    description: "Warm dusk palette tuned for cinematic creative sessions.",
    accent: "#f4a36e",
    surface:
      "linear-gradient(135deg, rgba(244, 163, 110, 0.18), rgba(124, 84, 132, 0.14), rgba(19, 18, 28, 0.28))"
  },
  {
    value: "zen",
    label: "Zen",
    description: "Calm, low-noise workspace with soft sage-blue guidance.",
    accent: "#8fd4c7",
    surface:
      "linear-gradient(135deg, rgba(143, 212, 199, 0.16), rgba(112, 142, 154, 0.14), rgba(15, 20, 21, 0.28))"
  },
  {
    value: "blueprint",
    label: "Blueprint",
    description: "Technical drafting atmosphere with crisp cyan planning cues.",
    accent: "#7fd4ff",
    surface:
      "linear-gradient(135deg, rgba(127, 212, 255, 0.18), rgba(53, 95, 152, 0.14), rgba(8, 16, 31, 0.28))"
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
  const previewRuntimeTelemetryKeysRef = useRef<Set<string>>(new Set());
  const previewRuntimeErrorScopesRef = useRef<Set<string>>(new Set());
  const [conversationEntries, setConversationEntries] = useState(() =>
    buildConversationEntries(initialSnapshot.messages, createConversationEntryId)
  );
  const [composerValue, setComposerValue] = useState("");
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
  const [workflowTraceEvents, setWorkflowTraceEvents] = useState<
    WorkflowRuntimeTraceEvent[]
  >([]);
  const [persistenceState, setPersistenceState] =
    useState<WorkspacePersistenceState>("loading");
  const [persistenceError, setPersistenceError] = useState<WorkstationError | null>(
    null
  );
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
  const chatLogRef = useRef<HTMLDivElement>(null);
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
    if (!chatLog || !shouldAutoScrollRef.current) {
      return;
    }

    chatLog.scrollTop = chatLog.scrollHeight;
  }, [conversationEntries]);

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
    }, mockWorkflowIntervalMs);

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
          const restoredImageAttachments = normalizeImageAttachments(
            restoredSnapshot.multimodal.imageAttachments
          );
          setSnapshot(restoredSnapshot);
          setConversationEntries(
            buildConversationEntries(restoredSnapshot.messages, createConversationEntryId)
          );
          setImageAttachments(restoredImageAttachments);
          setImageUploadError(restoredSnapshot.multimodal.error ?? null);
          imageAttachmentCounterRef.current = restoredImageAttachments.length;
          streamingAssistantIdRef.current = null;
          setActiveTab(restoredSession.record.activeInspectorTab);
          setActiveArtifactId(
            restoredSession.record.activeArtifactId ||
              restoredSnapshot.artifacts[0]?.id ||
              ""
          );
          setPreviewArtifactId(
            restoredSession.record.previewArtifactId ||
              getInitialPreviewArtifactId(restoredSnapshot)
          );
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
          skipNextPersistenceSaveRef.current = true;
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
    snapshot.artifacts[0];
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
  const workflow = useMemo(
    () => buildInteractiveWorkflow(snapshot.workflow, workflowProgressIndex),
    [snapshot.workflow, workflowProgressIndex]
  );
  const interactiveSnapshot: AssistantWorkspaceSnapshot = useMemo(
    () => ({
      ...snapshot,
      code:
        activeArtifact.type === "code"
          ? { ...snapshot.code, title: activeArtifact.title }
          : snapshot.code,
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
          ? "Streaming backend"
          : streamError
            ? "Backend fallback"
            : snapshot.debug.status,
        events: streamEvents
      }
    }),
    [
      activeArtifact.id,
      activeArtifact.title,
      activeArtifact.type,
      activeTab,
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
  const previewRuntimeSource = useMemo(
    () =>
      buildPreviewRuntimeSource({
        code: interactiveSnapshot.code,
        route: previewRendererRoute
      }),
    [interactiveSnapshot.code, previewRendererRoute]
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
  const activeTabSummary =
    interactiveSnapshot.inspectorTabs.find((tab) => tab.label === activeTab)
      ?.summary ?? "";
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
  const retrievalRuntime = useMemo(
    () => buildRetrievalRuntimeModel(interactiveSnapshot.retrieval, workflowTraceEvents),
    [interactiveSnapshot.retrieval, workflowTraceEvents]
  );
  const telemetryDashboard = useMemo(
    () =>
      buildTelemetryDashboardModel({
        activeArtifact,
        providerTelemetry,
        retrievalRuntime,
        snapshot: interactiveSnapshot,
        traceEvents: workflowTraceEvents,
        workflowRuntime
      }),
    [
      activeArtifact,
      interactiveSnapshot,
      providerTelemetry,
      retrievalRuntime,
      workflowRuntime,
      workflowTraceEvents
    ]
  );
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
        persistenceError,
        ...approvalRequests.map((request) => buildHitlApprovalError(request))
      ].filter((error): error is WorkstationError => error !== null),
    [approvalRequests, persistenceError, workflowRuntime.error]
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
  const isInspectorCollapsed = layoutState.inspectorCollapsed;
  const sessionStatusLabel = blockingApprovalRequest
    ? getHitlApprovalStateLabel(blockingApprovalRequest.state)
    : isStreaming
      ? "Streaming"
      : streamError
        ? "Fallback"
        : interactiveSnapshot.workflow.status;
  const sessionStatusDetail = blockingApprovalRequest
    ? blockingApprovalRequest.title
    : interactiveSnapshot.workflow.currentStep;
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

  function handleDensityToggle() {
    updateLayout({
      density: layoutState.density === "compact" ? "cozy" : "compact"
    });
  }

  function toggleUtilityPanel(panelName: UtilityPanelName) {
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
        message: `First sandbox frame rendered for ${source.title}.`,
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
    clearFeedbackTimers();
    setCopyFeedback(null);
    setTransferFeedback(null);
    setArtifactTransferError(null);
    streamingAssistantIdRef.current = null;
    previewRuntimeTelemetryKeysRef.current.clear();
    previewRuntimeErrorScopesRef.current.clear();
    setSnapshot(initialSnapshot);
    setConversationEntries(
      buildConversationEntries(initialSnapshot.messages, createConversationEntryId)
    );
    setImageAttachments(
      normalizeImageAttachments(initialSnapshot.multimodal.imageAttachments)
    );
    setImageUploadError(initialSnapshot.multimodal.error ?? null);
    imageAttachmentCounterRef.current = normalizeImageAttachments(
      initialSnapshot.multimodal.imageAttachments
    ).length;
    setComposerValue("");
    setActiveTab(getInitialActiveTab(initialSnapshot));
    setActiveArtifactId(initialSnapshot.artifacts[0]?.id ?? "");
    setPreviewArtifactId(getInitialPreviewArtifactId(initialSnapshot));
    setIsPreviewOpen(initialSnapshot.preview.active);
    setIsPreviewFullscreen(false);
    setPreviewSessionOverride(null);
    setWorkflowProgressIndex(getInitialWorkflowIndex(initialSnapshot.workflow.steps));
    setWorkflowRunId(0);
    setIsStreaming(false);
    setStreamError(null);
    setStreamEvents(initialSnapshot.debug.events);
    setWorkflowTraceEvents([]);
    setOpenUtilityPanel(null);
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

  async function handleComposerSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const prompt = composerValue.trim();

    if (!prompt) {
      return;
    }

    const timestamp = formatMessageTime();
    const userMessageId = createConversationEntryId();
    const assistantMessageId = createConversationEntryId();

    streamingAssistantIdRef.current = assistantMessageId;
    setConversationEntries((currentMessages) => [
      ...currentMessages,
      {
        content: prompt,
        activity: null,
        id: userMessageId,
        pending: false,
        phase: "complete",
        role: "user",
        time: timestamp
      },
      {
        content: "",
        activity: "Opening live response.",
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
    setWorkflowTraceEvents([]);
    hasPreviewRuntimeEventRef.current = false;
    previewRuntimeTelemetryKeysRef.current.clear();
    previewRuntimeErrorScopesRef.current.clear();
    setIsStreaming(true);
    setActiveTab("Overview");

    let streamedAnswer = "";
    const requestAttachments = toAssistantRequestImageAttachments(imageAttachments);

    try {
      const streamRequest: AssistantStreamRequest = {
        conversationId: "local-nextjs-session",
        domain: "webgpu_wgsl",
        mode: "generate",
        projectId: "local-nextjs-workspace",
        query: prompt
      };

      if (requestAttachments.length > 0) {
        streamRequest.attachments = requestAttachments;
      }

      for await (const streamEvent of streamAssistantEvents(streamRequest)) {
        applyStreamEventToWorkspace(streamEvent);

        if (streamEvent.event_type === "token_delta") {
          const delta = readPayloadText(streamEvent, "text");
          if (delta) {
            streamedAnswer += delta;
            startTransition(() => {
              updateStreamingAssistantMessage({
                activity: "Generating response.",
                content: streamedAnswer,
                phase: "streaming"
              });
            });
          }
        }

        if (streamEvent.event_type === "final") {
          const answer = readPayloadText(streamEvent, "answer");
          streamedAnswer = answer ?? streamedAnswer;
          finalizeStreamingAssistantMessage({
            activity: "Response completed.",
            content: streamedAnswer,
            phase: "complete"
          });
        }

        if (streamEvent.event_type === "error") {
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
              ? `${streamedAnswer}\n\nBackend stream error: ${error.userMessage}`
              : `Backend stream error: ${error.userMessage}`,
            phase: "error"
          });
        }
      }

      if (streamingAssistantIdRef.current && streamedAnswer) {
        finalizeStreamingAssistantMessage({
          activity: "Response completed.",
          content: streamedAnswer,
          phase: "complete"
        });
      } else if (streamingAssistantIdRef.current) {
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
              userMessage: "The backend stream is unavailable.",
              debugMessage: error instanceof Error ? error.message : null,
              recoverable: true,
              suggestedAction:
                "Retry the request after the backend recovers, or use the local fallback response.",
              retryLabel: "Send prompt again",
              resetLabel: "Clear workspace session"
            });
      const fallbackMessage = `Backend stream unavailable; showing local fallback. ${buildMockAssistantReply(
        prompt,
        activeArtifact.title
      )}`;
      setStreamError(streamFailure);
      finalizeStreamingAssistantMessage({
        activity: "Switching to the local fallback response.",
        content: fallbackMessage,
        phase: "fallback"
      });
      setWorkflowProgressIndex(0);
      setWorkflowRunId((currentRunId) => currentRunId + 1);
    } finally {
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
          "Backend stream event received."
      }
    ]);

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
      const hydration = hydrateWorkspaceFromArtifactExtractedEvent(
        snapshot,
        streamEvent
      );

      if (hydration.artifact) {
        setSnapshot(hydration.snapshot);
        setActiveArtifactId(hydration.activeArtifactId);
        setPreviewArtifactId(hydration.previewArtifactId);
        setPreviewSessionOverride(null);
      }
    }

    if (streamEvent.event_type === "preview_artifact") {
      hasPreviewRuntimeEventRef.current = true;
      const previewUpdate = readPreviewArtifactUpdate(streamEvent);
      const nextPreviewArtifactId =
        previewUpdate?.previewArtifactId ?? previewUpdate?.artifactId ?? null;

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
      const hydration = hydrateWorkspaceFromFinalEvent(snapshot, streamEvent, {
        skipPlainTextArtifact: hasPreviewRuntimeEventRef.current
      });

      if (!hydration.artifact) {
        return;
      }

      setSnapshot(hydration.snapshot);
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
    setPreviewContextArtifactId(artifact.id);
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
        setPreviewContextArtifactId(artifact.id);
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
    setPreviewContextArtifactId(artifact.id);

    if (action === "Open") {
      setActiveTab("Code");
      return;
    }

    if (action === "Preview") {
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

  return (
    <main
      className="workstation"
      data-active-tab={activeTab.toLowerCase()}
      data-density={layoutState.density}
      data-focus-mode={isFocusMode ? "true" : "false"}
      data-inspector-state={isInspectorCollapsed ? "collapsed" : "open"}
      data-preview={isPreviewOpen ? "open" : "closed"}
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
          <span>{sessionStatusLabel}</span>
          <strong>{sessionStatusDetail}</strong>
          <small>{formatSessionTelemetryLabel(providerTelemetry)}</small>
        </div>

        <div
          ref={utilityTrayRef}
          className="topbarActions"
          aria-label="Workspace actions"
        >
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
                    Generate, refine, and open artifacts without leaving the current
                    workspace flow.
                  </p>
                </div>
                <div
                  className="sessionMetric"
                  aria-label="Active artifact"
                  data-active="true"
                >
                  <span>Active artifact</span>
                  <strong>{activeArtifact.title}</strong>
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
              {persistenceError ? (
                <SubsystemErrorCallout
                  className="sessionErrorCallout"
                  error={persistenceError}
                  title="Session persistence issue"
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
              {conversationEntries.map((message, index) => (
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
                    {message.content || getConversationPhasePlaceholder(message.phase)}
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
              ))}
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
              data-ready={isComposerReady}
              onSubmit={handleComposerSubmit}
            >
              <div className="composerInputFrame">
                <label className="composerUploadButton">
                  <input
                    accept={supportedImageUploadAccept}
                    aria-label="Upload image reference"
                    disabled={isStreaming}
                    multiple
                    onChange={(event) => void handleImageUploadChange(event)}
                    type="file"
                  />
                  <ImagePlus size={16} aria-hidden="true" />
                  <span>Image</span>
                </label>
                <textarea
                  aria-label="Assistant prompt"
                  onChange={(event) => setComposerValue(event.currentTarget.value)}
                  placeholder="Ask for a denser particle field, a softer palette, or a preview pass."
                  value={composerValue}
                />
              </div>
              <span className="composerState" aria-live="polite">
                {composerStateLabel}
              </span>
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

          {interactiveSnapshot.preview.available && !isFocusMode ? (
            <PreviewShelf
              controller={previewController}
              height={layoutState.previewHeight}
              onClear={handlePreviewStateClear}
              onFullscreenToggle={handlePreviewFullscreenChange}
              onReload={handlePreviewStateReload}
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
                  <span>{activeTab}</span>
                </div>
              ) : (
                <>
                  <header className="inspectorHeader">
                    <div>
                      <span className="eyebrow">Inspector</span>
                      <h2>{activeTab}</h2>
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
                    {interactiveSnapshot.inspectorTabs.map((tab) => {
                      const Icon = inspectorTabIcons[tab.label];

                      return (
                        <button
                          aria-controls={`${tab.label.toLowerCase()}-inspector-panel`}
                          aria-label={tab.label}
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
                          <span>{tab.label}</span>
                          {tab.badge ? <small>{tab.badge}</small> : null}
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
                    onArtifactCopy={handleArtifactCopy}
                    onArtifactAction={handleArtifactAction}
                    onArtifactTransfer={handleArtifactTransfer}
                    providerTelemetry={providerTelemetry}
                    previewController={previewController}
                    previewRoute={previewRendererRoute}
                    previewRuntimeSource={previewRuntimeSource}
                    retrievalRuntime={retrievalRuntime}
                    showDebugPanels={workspacePreferences.showDebugPanels}
                    snapshot={interactiveSnapshot}
                    telemetryDashboard={telemetryDashboard}
                    transferFeedback={transferFeedback}
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
  onReload: () => void;
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

function PreviewShelf({
  controller,
  height,
  onClear,
  onFullscreenToggle,
  onReload,
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
  snapshot
}: PreviewShelfProps) {
  function handleSummaryClick(event: MouseEvent<HTMLElement>) {
    event.preventDefault();
    onToggle(!snapshot.preview.active);
  }

  function handleToggle(event: SyntheticEvent<HTMLDetailsElement>) {
    onToggle(event.currentTarget.open);
  }

  const layoutSize = resolvePreviewShelfLayoutSize(snapshot.preview);
  const panelHeight = resolvePreviewShelfPanelHeight(height, snapshot.preview);
  const canResizePreview =
    snapshot.preview.active && layoutSize === "visual" && !controller.isFullscreen;
  const panelStyle = controller.isFullscreen ? undefined : { height: panelHeight };

  return (
    <section className="previewZone" aria-label="Preview workspace">
      <details
        data-fullscreen={controller.isFullscreen ? "true" : "false"}
        data-layout-size={layoutSize}
        className="previewShelf"
        data-state={snapshot.preview.active ? "open" : "closed"}
        data-runtime-state={snapshot.preview.state}
        onToggle={handleToggle}
        open={snapshot.preview.active}
      >
        <summary
          aria-expanded={snapshot.preview.active}
          onClick={handleSummaryClick}
        >
          <span className="previewSummaryIcon" aria-hidden="true">
            <Play size={16} />
          </span>
          <div>
            <strong>{snapshot.preview.title}</strong>
            <span>{snapshot.preview.artifactName}</span>
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
              <small>{`${snapshot.preview.status} / ${route.rendererLabel}`}</small>
            </div>
            <div className="previewToolbarActions" aria-label="Preview controls">
              <button
                aria-label="Collapse preview"
                className="previewControlButton"
                onClick={() => onToggle(false)}
                title="Collapse preview"
                type="button"
              >
                <ChevronDown size={15} />
              </button>
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
              onRuntimeFrame={onRuntimeFrame}
              onRuntimeStatus={onRuntimeStatus}
              preview={snapshot.preview}
              route={route}
              runtimeSessionKey={runtimeSessionKey}
              runtimeSource={runtimeSource}
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
  onArtifactCopy: (artifact: ArtifactSummary) => Promise<void>;
  onArtifactAction: (action: ArtifactAction, artifact: ArtifactSummary) => void;
  onArtifactTransfer: (artifact: ArtifactSummary) => void;
  providerTelemetry: ProviderTelemetryModel;
  previewController: PreviewControllerModel;
  previewRoute: PreviewRendererRoute;
  previewRuntimeSource: PreviewRuntimeSource;
  retrievalRuntime: RetrievalRuntimeModel;
  showDebugPanels: boolean;
  snapshot: AssistantWorkspaceSnapshot;
  telemetryDashboard: TelemetryDashboardModel;
  transferFeedback: ArtifactActionFeedback | null;
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
  onArtifactCopy,
  onArtifactAction,
  onArtifactTransfer,
  providerTelemetry,
  previewController,
  previewRoute,
  previewRuntimeSource,
  retrievalRuntime,
  showDebugPanels,
  snapshot,
  telemetryDashboard,
  transferFeedback,
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
      />
    );
  }

  if (activeTab === "Workflow") {
    return (
      <WorkflowInspector
        runtime={workflowRuntime}
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
        copyFeedback={copyFeedback}
        onArtifactAction={onArtifactAction}
        transferFeedback={transferFeedback}
      />
    );
  }

  if (activeTab === "Retrieval") {
    return <RetrievalInspector runtime={retrievalRuntime} />;
  }

  return (
    <OverviewInspector
      activeArtifact={activeArtifact}
      retrieval={retrievalRuntime}
      runtime={workflowRuntime}
      telemetry={providerTelemetry}
      showDebugPanels={showDebugPanels}
      snapshot={snapshot}
    />
  );
}

function OverviewInspector({
  activeArtifact,
  retrieval,
  runtime,
  telemetry,
  showDebugPanels,
  snapshot
}: {
  activeArtifact: ArtifactSummary;
  retrieval: RetrievalRuntimeModel;
  runtime: WorkflowRuntimeModel;
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
        <div className="overviewTile" role="group" aria-label="Artifacts summary">
          <span>Artifacts</span>
          <strong>{snapshot.artifacts.length}</strong>
          <p>{activeArtifact.title}</p>
        </div>
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
    </section>
  );
}

function PreviewInspector({
  controller,
  dashboard,
  preview,
  route,
  runtimeSource
}: {
  controller: PreviewControllerModel;
  dashboard: TelemetryDashboardModel;
  preview: AssistantWorkspaceSnapshot["preview"];
  route: PreviewRendererRoute;
  runtimeSource: PreviewRuntimeSource;
}) {
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
  transferFeedback: ArtifactActionFeedback | null;
};

function CodeInspector({
  artifact,
  copyFeedback,
  document,
  highlightedLines,
  onArtifactCopy,
  onArtifactTransfer,
  transferFeedback
}: CodeInspectorProps) {
  const transferAction = getArtifactTransferAction(artifact.actions);
  const actionMessage = getArtifactActionMessage(
    artifact,
    copyFeedback,
    transferFeedback
  );

  return (
    <section
      aria-label="Code inspector"
      className="inspectorPanel codePanel"
      data-opened-artifact={document.fileName}
      id="code-inspector-panel"
      role="tabpanel"
    >
      <header className="codePanelHeader">
        <div>
          <span>Active document</span>
          <strong>{document.fileName}</strong>
          <p>{document.summary}</p>
        </div>
        <div className="codePanelActions">
          <button
            aria-label={`Copy ${document.fileName}`}
            onClick={() => void onArtifactCopy(artifact)}
            type="button"
          >
            {getArtifactActionButtonLabel("Copy", artifact, copyFeedback, transferFeedback)}
          </button>
          {transferAction ? (
            <button
              aria-label={`${formatArtifactActionLabel(transferAction, artifact)} ${document.fileName}`}
              onClick={() => onArtifactTransfer(artifact)}
              type="button"
            >
              {getArtifactActionButtonLabel(
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
      {actionMessage ? (
        <p className="artifactActionFeedback" aria-live="polite">
          {actionMessage}
        </p>
      ) : null}
      <div
        aria-label={`${document.fileName} content`}
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
  issues,
  runtime,
  telemetry,
  showDebugPanels
}: {
  issues: WorkstationError[];
  runtime: WorkflowRuntimeModel;
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
      <TelemetryLifecycleCard telemetry={telemetry} />
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

        <article
          aria-label="Provider token and cost telemetry"
          className="telemetryDashboardCard"
          data-state={dashboard.provider.status}
          role="group"
        >
          <header>
            <span>Provider</span>
            <strong>{formatProviderRuntimeLabel(dashboard.provider)}</strong>
          </header>
          <dl>
            <div>
              <dt>Tokens</dt>
              <dd>{formatTokenUsageTotal(dashboard.provider)}</dd>
            </div>
            <div>
              <dt>Cost</dt>
              <dd>{dashboard.provider.summary.costLabel}</dd>
            </div>
            <div>
              <dt>First token</dt>
              <dd>{formatRuntimeDuration(dashboard.provider.timing.timeToFirstTokenMs)}</dd>
            </div>
            <div>
              <dt>Stream</dt>
              <dd>{dashboard.provider.stream.eventCount}</dd>
            </div>
          </dl>
          <p>{formatTokenUsageDetail(dashboard.provider)}</p>
          <small>{formatTelemetryCostSource(dashboard.provider)}</small>
        </article>

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

        <article
          aria-label="LangSmith observability"
          className="telemetryDashboardCard"
          data-state={dashboard.observability.state}
          role="group"
        >
          <header>
            <span>LangSmith</span>
            <strong>{formatObservabilityDashboardState(dashboard.observability.state)}</strong>
          </header>
          <dl>
            <div>
              <dt>Trace</dt>
              <dd>{dashboard.observability.traceId ?? "Not linked"}</dd>
            </div>
            <div>
              <dt>Project</dt>
              <dd>{dashboard.observability.projectName ?? "Local only"}</dd>
            </div>
          </dl>
          <p>{dashboard.observability.traceKind ?? "Optional tracing unavailable."}</p>
          <small>
            {dashboard.observability.reason ??
              dashboard.observability.status ??
              "No LangSmith metadata in stream"}
          </small>
        </article>

        <article
          aria-label="RAGAs evaluation lineage"
          className="telemetryDashboardCard"
          data-state={dashboard.evaluation.state}
          role="group"
        >
          <header>
            <span>Evaluation</span>
            <strong>{dashboard.evaluation.statusLabel}</strong>
          </header>
          <dl>
            <div>
              <dt>Run</dt>
              <dd>{dashboard.evaluation.runId ?? "None"}</dd>
            </div>
            <div>
              <dt>Dataset</dt>
              <dd>{dashboard.evaluation.datasetId ?? "None"}</dd>
            </div>
            <div>
              <dt>Rows</dt>
              <dd>{dashboard.evaluation.resultRows ?? "N/A"}</dd>
            </div>
            <div>
              <dt>Failures</dt>
              <dd>{dashboard.evaluation.metricFailures ?? "N/A"}</dd>
            </div>
          </dl>
          <p>{dashboard.evaluation.detail}</p>
          <small>
            {dashboard.evaluation.metrics.length > 0
              ? dashboard.evaluation.metrics.join(" / ")
              : "RAGAs metadata pending"}
          </small>
        </article>

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
  copyFeedback: ArtifactActionFeedback | null;
  onArtifactAction: (action: ArtifactAction, artifact: ArtifactSummary) => void;
  transferFeedback: ArtifactActionFeedback | null;
};

function ArtifactsInspector({
  activeArtifact,
  activeArtifactDocument,
  activeArtifactId,
  artifacts,
  artifactTransferError,
  copyFeedback,
  onArtifactAction,
  transferFeedback
}: ArtifactsInspectorProps) {
  const actionMessage = getArtifactActionMessage(
    activeArtifact,
    copyFeedback,
    transferFeedback
  );

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
      <article
        aria-label="Active artifact details"
        className="artifactDetailCard"
        role="group"
      >
        <header className="artifactDetailHeader">
          <div>
            <span>Selected artifact</span>
            <strong>{activeArtifactDocument.fileName}</strong>
            <p>{activeArtifact.summary}</p>
          </div>
          <div className="artifactBadges">
            <span className="artifactSelected">Selected</span>
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
        </dl>
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

function RetrievalInspector({ runtime }: { runtime: RetrievalRuntimeModel }) {
  const hasSources = runtime.sources.length > 0;

  return (
    <section
      aria-label="Retrieval inspector"
      className="inspectorPanel retrievalPanel"
      id="retrieval-inspector-panel"
      role="tabpanel"
    >
      <article
        aria-label="Retrieval status"
        className="retrievalSummaryCard"
        data-state={runtime.summary.state}
        role="group"
      >
        <header className="retrievalSummaryHeader">
          <div>
            <span>Retrieval status</span>
            <strong>{runtime.summary.status}</strong>
            <p>{runtime.summary.headline}</p>
          </div>
          <span className="retrievalStateBadge" data-state={runtime.summary.state}>
            {runtime.summary.providerLabel}
          </span>
        </header>
        <div className="retrievalSummaryMeta" aria-label="Retrieval metadata" role="list">
          <span role="listitem">{runtime.summary.sourceCount} sources</span>
          <span role="listitem">{runtime.summary.chunkCount} chunks</span>
          <span role="listitem">{runtime.summary.coverageLabel}</span>
          <span role="listitem">{runtime.summary.qualityLabel}</span>
          <span role="listitem">{runtime.summary.freshnessLabel}</span>
        </div>
        {runtime.request.query ? (
          <p className="retrievalQuery">
            Query
            <code>{runtime.request.query}</code>
          </p>
        ) : null}
        {runtime.request.filterLabels.length > 0 ? (
          <div className="retrievalFilterRow" aria-label="Retrieval filters">
            {runtime.request.filterLabels.map((label) => (
              <span className="retrievalFilterPill" key={label}>
                {label}
              </span>
            ))}
          </div>
        ) : null}
        <p className="retrievalSummaryDetail">{runtime.summary.detail}</p>
        {runtime.summary.error ? (
          <SubsystemErrorCallout
            className="retrievalErrorCallout"
            error={runtime.summary.error}
            title="Retrieval failed"
          />
        ) : null}
        {runtime.summary.warning ? (
          <p className="retrievalWarning" role="status">
            {runtime.summary.warning}
          </p>
        ) : null}
      </article>
      {hasSources ? (
        <div className="retrievalList">
          {runtime.sources.map((source) => (
            <article className="retrievalItem" key={source.sourceId}>
              <header className="retrievalItemHeader">
                <div>
                  <div className="retrievalItemMeta">
                    <span className="retrievalDomainBadge">{source.domainLabel}</span>
                    <span className="retrievalSourceType">{source.sourceTypeLabel}</span>
                  </div>
                  <strong>{source.title}</strong>
                  <p>
                    {source.publisher}
                    {source.host ? ` • ${source.host}` : ""}
                  </p>
                </div>
                <div className="retrievalItemSignals">
                  <span className="retrievalScoreBadge" data-quality={source.quality}>
                    {source.qualityLabel}
                  </span>
                  <span
                    className="retrievalFreshnessBadge"
                    data-freshness={source.freshness}
                  >
                    {source.freshnessLabel}
                  </span>
                </div>
              </header>
              <p className="retrievalWhyUsed">{source.whyUsed}</p>
              <div className="retrievalChunkList" aria-label={`${source.title} chunks`}>
                {source.chunks.map((chunk) => (
                  <article className="retrievalChunk" key={chunk.id}>
                    <header>
                      <strong>{`Chunk ${chunk.chunkIndex + 1}`}</strong>
                      <span>{chunk.relevanceLabel}</span>
                    </header>
                    <p>{chunk.snippet}</p>
                  </article>
                ))}
              </div>
              {source.href ? (
                <a
                  className="retrievalSourceLink"
                  href={source.href}
                  rel="noreferrer"
                  target="_blank"
                >
                  Open source reference
                </a>
              ) : null}
            </article>
          ))}
        </div>
      ) : (
        <article
          aria-label="Retrieval empty state"
          className="retrievalEmptyCard"
          data-state={runtime.summary.state}
          role="group"
        >
          <strong>{runtime.summary.headline}</strong>
          <p>{runtime.summary.detail}</p>
        </article>
      )}
    </section>
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
};

function CommandMenuPanel({
  activeTab,
  hasBlockingApproval,
  isFocusMode,
  isPreviewAvailable,
  isPreviewOpen,
  onFocusModeToggle,
  onOpenTab,
  onPreviewToggle,
  onWorkspaceClear
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
        <p>Jump to the next workspace surface without changing the current mock flow.</p>
      </header>
      <div className="commandMenuGrid">
        <button
          data-active={activeTab === "Overview"}
          onClick={() => onOpenTab("Overview")}
          type="button"
        >
          <strong>Overview inspector</strong>
          <span>Return to the compact session summary.</span>
        </button>
        <button
          data-active={activeTab === "Preview"}
          onClick={() => onOpenTab("Preview")}
          type="button"
        >
          <strong>Preview inspector</strong>
          <span>Review runtime, renderer, source, and preview health metadata.</span>
        </button>
        <button
          data-active={activeTab === "Code"}
          onClick={() => onOpenTab("Code")}
          type="button"
        >
          <strong>Code inspector</strong>
          <span>Open the active document in the full-height inspector.</span>
        </button>
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
            <strong>Advanced traces</strong>
            <p>Show or hide workflow transition and event trace panels in the inspector.</p>
          </div>
          <button
            aria-label="Advanced traces"
            aria-pressed={preferences.showDebugPanels}
            data-active={preferences.showDebugPanels}
            onClick={() =>
              onPreferencesChange({
                showDebugPanels: !preferences.showDebugPanels
              })
            }
            type="button"
          >
            {preferences.showDebugPanels ? "Visible" : "Hidden"}
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
          {isActive ? <span className="artifactSelected">Selected</span> : null}
          <span className="artifactType">{getArtifactTypeLabel(artifact.type)}</span>
        </div>
      </div>
      <p>{artifact.summary}</p>
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
        ? "Session restore timed out, so the workspace stayed on the current snapshot."
        : "Session save timed out, so the workspace is staying local for now.",
    recoverable: true,
    suggestedAction:
      operation === "load"
        ? "Continue from the current snapshot or reset the workspace session."
        : "Keep editing locally and retry the save after the backend recovers.",
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

function getInitialPreviewArtifactId(snapshot: AssistantWorkspaceSnapshot): string {
  return (
    snapshot.artifacts.find(
      (artifact) => artifact.title === snapshot.preview.artifactName
    )?.id ??
    snapshot.artifacts[0]?.id ??
    ""
  );
}

function getInitialWorkflowIndex(steps: WorkflowStepState[]) {
  const activeIndex = steps.findIndex((step) => step.state === "active");

  return activeIndex >= 0 ? activeIndex : 0;
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
  progressIndex: number
): WorkspaceWorkflow {
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
    case "completed":
      return "Completed";
    case "failed":
      return "Failed";
    default:
      return "Running";
  }
}

function formatRetryCount(retryCount: number) {
  return retryCount === 1 ? "1 retry loop" : `${retryCount} retry loops`;
}

function formatSessionTelemetryLabel(telemetry: ProviderTelemetryModel) {
  if (telemetry.status === "idle") {
    return "Telemetry pending";
  }

  return `${telemetry.summary.providerLabel} / ${telemetry.summary.tokenLabel}`;
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

function formatObservabilityDashboardState(
  state: TelemetryDashboardModel["observability"]["state"]
) {
  switch (state) {
    case "linked":
      return "Linked";
    case "requested":
      return "Requested";
    case "disabled":
      return "Disabled";
    default:
      return "Unavailable";
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

function buildMockAssistantReply(prompt: string, artifactTitle: string) {
  const trimmedPrompt = prompt.trim();
  const promptSummary =
    trimmedPrompt.length > 88
      ? `${trimmedPrompt.slice(0, 85).trim()}...`
      : trimmedPrompt;

  return `Mock orchestration pass started for "${promptSummary}". I kept ${artifactTitle} active and advanced the workflow locally without contacting the backend.`;
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
