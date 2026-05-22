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
  Command,
  LayoutGrid,
  Moon,
  PanelRight,
  Play,
  SendHorizontal,
  Settings,
  Sparkles,
  TerminalSquare
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type {
  ArtifactAction,
  ArtifactSummary,
  AssistantWorkspaceSnapshot,
  InspectorTabName,
  WorkflowStepState
} from "@/lib/assistant-client";
import {
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
  type WorkspacePersistenceClient
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
import {
  buildWorkflowRuntimeModel,
  type WorkflowRuntimeModel,
  type WorkflowRuntimeTraceEvent,
  type WorkflowRuntimeVisualState
} from "@/lib/workflow-runtime";
import {
  buildConversationEntries,
  getComposerStatusLabel,
  getConversationPhaseBadge,
  getConversationPhasePlaceholder,
  toPersistedConversation,
  type ConversationEntry,
  type ConversationEntryPhase
} from "@/lib/streaming-conversation";

type WorkstationShellProps = {
  snapshot: AssistantWorkspaceSnapshot;
  streamAssistantEvents?: AssistantStreamClient;
  persistenceClient?: WorkspacePersistenceClient;
};

type AssistantStreamClient = (
  request: AssistantStreamRequest
) => AsyncIterable<AssistantStreamEvent>;

const inspectorTabIcons = {
  Overview: Sparkles,
  Code: Braces,
  Workflow: Activity,
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
    description: "Blackened shell with lime signal highlights.",
    accent: "#9fe870",
    surface: "linear-gradient(135deg, rgba(159, 232, 112, 0.2), rgba(44, 75, 36, 0.16))"
  }
] satisfies readonly ThemePresetOption[];

export function WorkstationShell({
  snapshot: initialSnapshot,
  streamAssistantEvents = streamBackendAssistantEvents,
  persistenceClient = defaultWorkspacePersistenceClient
}: WorkstationShellProps) {
  const [snapshot, setSnapshot] = useState(initialSnapshot);
  const entryIdCounterRef = useRef(0);
  const streamingAssistantIdRef = useRef<string | null>(null);
  const [conversationEntries, setConversationEntries] = useState(() =>
    buildConversationEntries(initialSnapshot.messages, createConversationEntryId)
  );
  const [composerValue, setComposerValue] = useState("");
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
  const [workflowProgressIndex, setWorkflowProgressIndex] = useState(
    getInitialWorkflowIndex(initialSnapshot.workflow.steps)
  );
  const [workflowRunId, setWorkflowRunId] = useState(0);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamError, setStreamError] = useState<string | null>(null);
  const [streamEvents, setStreamEvents] = useState(initialSnapshot.debug.events);
  const [workflowTraceEvents, setWorkflowTraceEvents] = useState<
    WorkflowRuntimeTraceEvent[]
  >([]);
  const [persistenceState, setPersistenceState] =
    useState<WorkspacePersistenceState>("loading");
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
  const chatLogRef = useRef<HTMLDivElement>(null);
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

  function clearFeedbackTimers() {
    clearTimer(copyFeedbackTimerRef.current);
    clearTimer(transferFeedbackTimerRef.current);
  }

  function createConversationEntryId() {
    entryIdCounterRef.current += 1;
    return `conversation-entry-${entryIdCounterRef.current}`;
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
    let isMounted = true;

    async function restoreWorkspaceSession() {
      try {
        const restoredSession = await withPersistenceTimeout(
          persistenceClient.load(),
          null,
          1500
        );
        if (!isMounted) {
          return;
        }

        if (restoredSession) {
          const restoredSnapshot = snapshotFromWorkspaceSessionRecord(
            initialSnapshot,
            restoredSession
          );
          setSnapshot(restoredSnapshot);
          setConversationEntries(
            buildConversationEntries(restoredSnapshot.messages, createConversationEntryId)
          );
          streamingAssistantIdRef.current = null;
          setActiveTab(restoredSession.activeInspectorTab);
          setActiveArtifactId(
            restoredSession.activeArtifactId ||
              restoredSnapshot.artifacts[0]?.id ||
              ""
          );
          setPreviewArtifactId(
            restoredSession.previewArtifactId ||
              getInitialPreviewArtifactId(restoredSnapshot)
          );
          setIsPreviewOpen(restoredSession.previewOpen);
          setLayoutState(normalizeWorkspaceLayoutState(restoredSession.layout));
          setWorkspacePreferences(
            normalizeWorkspacePreferences(restoredSession.preferences)
          );
          setIsFocusMode(false);
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
            fingerprintWorkspaceSessionRecord(restoredSession);
          skipNextPersistenceSaveRef.current = true;
          setPersistenceState("restored");
          return;
        }

        setPersistenceState("ready");
      } catch {
        if (isMounted) {
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
  const previewArtifact =
    snapshot.artifacts.find((artifact) => artifact.id === previewArtifactId) ??
    activeArtifact;
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
      preview: {
        ...snapshot.preview,
        active: isPreviewOpen,
        artifactName: previewArtifact.title,
        collapsed: !isPreviewOpen,
        status: isPreviewOpen ? "Preview open" : "Ready when opened",
        trigger: `Preview ${previewArtifact.title}`
      },
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
      activeArtifact.title,
      activeArtifact.type,
      activeTab,
      isPreviewOpen,
      isStreaming,
      persistedMessages,
      previewArtifact.title,
      snapshot,
      streamError,
      streamEvents,
      workflow
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
  const isComposerReady = Boolean(composerValue.trim()) && !isStreaming;
  const streamState = isStreaming ? "streaming" : streamError ? "fallback" : "idle";
  const composerStateLabel = getComposerStatusLabel({
    isReady: isComposerReady,
    isStreaming,
    phase: liveAssistantEntry?.phase ?? null,
    streamError
  });
  const persistenceStatusLabel =
    persistenceStateLabels[persistenceState] ?? "Local session ready";
  const isInspectorCollapsed = layoutState.inspectorCollapsed;
  const workspaceLayoutStyle = useMemo(
    () =>
      ({
        "--inspector-width": `${layoutState.inspectorWidth}px`,
        "--preview-height": `${layoutState.previewHeight}px`
      }) as WorkspaceLayoutStyle,
    [layoutState.inspectorWidth, layoutState.previewHeight]
  );

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
      { target: "local" },
      1500
    )
      .then((result) => {
        if (!isShellMountedRef.current) {
          return;
        }

        setPersistenceState(result.target === "remote" ? "saved" : "local");
      })
      .catch(() => {
        if (isShellMountedRef.current) {
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

  function handlePreviewShelfFromControl() {
    if (!interactiveSnapshot.preview.available) {
      return;
    }

    handlePreviewOpenChange(!isPreviewOpen);
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
    if (!options.preserveFocusMode && isFocusMode) {
      focusRestoreRef.current = null;
      setIsFocusMode(false);
    }
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
    handleInspectorCollapsedChange(true, { preserveFocusMode: true });
    if (interactiveSnapshot.preview.available) {
      handlePreviewOpenChange(false, { preserveFocusMode: true });
    }
    setIsFocusMode(true);
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
    setIsStreaming(true);
    setActiveTab("Overview");

    let streamedAnswer = "";

    try {
      for await (const streamEvent of streamAssistantEvents({
        conversationId: "local-nextjs-session",
        domain: "webgpu_wgsl",
        mode: "generate",
        projectId: "local-nextjs-workspace",
        query: prompt
      })) {
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
          const message =
            readPayloadText(streamEvent, "message") ?? "Backend stream failed.";
          setStreamError(message);
          finalizeStreamingAssistantMessage({
            activity: message,
            content: streamedAnswer
              ? `${streamedAnswer}\n\nBackend stream error: ${message}`
              : `Backend stream error: ${message}`,
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
        setStreamError("Live response ended before completion.");
        finalizeStreamingAssistantMessage({
          activity: "Live response ended before completion.",
          content: "Live response ended before completion.",
          phase: "error"
        });
      }
    } catch {
      const fallbackMessage = `Backend stream unavailable; showing local fallback. ${buildMockAssistantReply(
        prompt,
        activeArtifact.title
      )}`;
      setStreamError("Backend stream unavailable. Showing local fallback.");
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

    if (streamEvent.event_type === "preview_artifact") {
      const artifactId = readPayloadText(streamEvent, "artifact_id");
      if (
        artifactId &&
        snapshot.artifacts.some((artifact) => artifact.id === artifactId)
      ) {
        setPreviewArtifactId(artifactId);
      }
      if (workspacePreferences.autoOpenPreview) {
        handlePreviewOpenChange(true);
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
    setActiveArtifactId(artifact.id);
    const wasTransferred = downloadArtifactDocument(
      buildArtifactDocument(interactiveSnapshot, artifact)
    );
    setFeedbackState(
      artifact.id,
      wasTransferred ? "success" : "error",
      transferFeedbackTimerRef,
      setTransferFeedback
    );
  }

  function handleArtifactAction(action: ArtifactAction, artifact: ArtifactSummary) {
    setActiveArtifactId(artifact.id);

    if (action === "Open") {
      setActiveTab("Code");
      return;
    }

    if (action === "Preview") {
      setPreviewArtifactId(artifact.id);
      handlePreviewOpenChange(true);
      setActiveTab("Overview");
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
          <span>
            {isStreaming
              ? "Streaming"
              : streamError
                ? "Fallback"
                : interactiveSnapshot.workflow.status}
          </span>
          <strong>{interactiveSnapshot.workflow.currentStep}</strong>
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
                isFocusMode={isFocusMode}
                isPreviewAvailable={interactiveSnapshot.preview.available}
                isPreviewOpen={isPreviewOpen}
                onFocusModeToggle={() => {
                  handleFocusModeToggle();
                  setOpenUtilityPanel(null);
                }}
                onOpenTab={revealInspectorTab}
                onPreviewToggle={handlePreviewShelfFromControl}
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

            <form
              className="composer"
              data-ready={isComposerReady}
              onSubmit={handleComposerSubmit}
            >
              <textarea
                aria-label="Assistant prompt"
                onChange={(event) => setComposerValue(event.currentTarget.value)}
                placeholder="Ask for a denser particle field, a softer palette, or a preview pass."
                value={composerValue}
              />
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
              height={layoutState.previewHeight}
              onResizeKeyDown={handlePreviewResizeKeyDown}
              onResizeStart={handlePreviewResizeStart}
              onToggle={handlePreviewOpenChange}
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
                    copyFeedback={copyFeedback}
                    onArtifactCopy={handleArtifactCopy}
                    onArtifactAction={handleArtifactAction}
                    onArtifactTransfer={handleArtifactTransfer}
                    showDebugPanels={workspacePreferences.showDebugPanels}
                    snapshot={interactiveSnapshot}
                    transferFeedback={transferFeedback}
                    workflowRuntime={workflowRuntime}
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
  height: number;
  onResizeKeyDown: (event: KeyboardEvent<HTMLElement>) => void;
  onResizeStart: (event: MouseEvent<HTMLElement>) => void;
  onToggle: (isOpen: boolean) => void;
  resizing: boolean;
};

function PreviewShelf({
  height,
  onResizeKeyDown,
  onResizeStart,
  onToggle,
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

  return (
    <section className="previewZone" aria-label="Preview workspace">
      <details
        className="previewShelf"
        data-state={snapshot.preview.active ? "open" : "closed"}
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
          <small>{snapshot.preview.status}</small>
        </summary>
        <div className="previewBody" style={{ height }}>
          <div className="previewFrame" aria-label="Preview placeholder">
            <div>
              <strong>{snapshot.preview.version}</strong>
              <span>{snapshot.preview.renderer}</span>
            </div>
          </div>
          <div className="previewCopy">
            <p>{snapshot.preview.summary}</p>
            <dl>
              <div>
                <dt>Target</dt>
                <dd>{snapshot.preview.target}</dd>
              </div>
              <div>
                <dt>Opened from</dt>
                <dd>{snapshot.preview.trigger}</dd>
              </div>
            </dl>
          </div>
        </div>
        <div
          aria-label="Resize preview shelf"
          aria-orientation="horizontal"
          aria-valuemax={workspaceLayoutBounds.maxPreviewHeight}
          aria-valuemin={workspaceLayoutBounds.minPreviewHeight}
          aria-valuenow={height}
          className="layoutResizeHandle previewResizeHandle"
          data-active={resizing}
          onKeyDown={onResizeKeyDown}
          onMouseDown={onResizeStart}
          role="separator"
          tabIndex={snapshot.preview.active ? 0 : -1}
        >
          <span aria-hidden="true" />
        </div>
      </details>
    </section>
  );
}

type InspectorPanelProps = {
  activeArtifact: ArtifactSummary;
  activeArtifactDocument: ArtifactDocument;
  activeArtifactHighlights: HighlightedLine[];
  activeArtifactId: string;
  activeTab: InspectorTabName;
  copyFeedback: ArtifactActionFeedback | null;
  onArtifactCopy: (artifact: ArtifactSummary) => Promise<void>;
  onArtifactAction: (action: ArtifactAction, artifact: ArtifactSummary) => void;
  onArtifactTransfer: (artifact: ArtifactSummary) => void;
  showDebugPanels: boolean;
  snapshot: AssistantWorkspaceSnapshot;
  transferFeedback: ArtifactActionFeedback | null;
  workflowRuntime: WorkflowRuntimeModel;
};

function InspectorPanel({
  activeArtifact,
  activeArtifactDocument,
  activeArtifactHighlights,
  activeArtifactId,
  activeTab,
  copyFeedback,
  onArtifactCopy,
  onArtifactAction,
  onArtifactTransfer,
  showDebugPanels,
  snapshot,
  transferFeedback,
  workflowRuntime
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

  if (activeTab === "Workflow") {
    return (
      <WorkflowInspector
        runtime={workflowRuntime}
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
        copyFeedback={copyFeedback}
        onArtifactAction={onArtifactAction}
        transferFeedback={transferFeedback}
      />
    );
  }

  if (activeTab === "Retrieval") {
    return <RetrievalInspector snapshot={snapshot} />;
  }

  return (
    <OverviewInspector
      activeArtifact={activeArtifact}
      runtime={workflowRuntime}
      showDebugPanels={showDebugPanels}
      snapshot={snapshot}
    />
  );
}

function OverviewInspector({
  activeArtifact,
  runtime,
  showDebugPanels,
  snapshot
}: WorkstationShellProps & {
  activeArtifact: ArtifactSummary;
  runtime: WorkflowRuntimeModel;
  showDebugPanels: boolean;
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
        <div className="overviewTile" role="group" aria-label="Preview summary">
          <span>Preview</span>
          <strong>{snapshot.preview.active ? "Open" : "Ready"}</strong>
          <p>{snapshot.preview.available ? snapshot.preview.artifactName : "No target"}</p>
        </div>
        <div className="overviewTile" role="group" aria-label="Retrieval summary">
          <span>Retrieval</span>
          <strong>{snapshot.retrieval.status}</strong>
          <p>{snapshot.retrieval.sources.length} references</p>
        </div>
      </div>
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
              aria-label={`${formatArtifactActionLabel(transferAction)} ${document.fileName}`}
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
  runtime,
  showDebugPanels
}: {
  runtime: WorkflowRuntimeModel;
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
      </div>
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

type ArtifactsInspectorProps = {
  activeArtifact: ArtifactSummary;
  activeArtifactDocument: ArtifactDocument;
  activeArtifactId: string;
  artifacts: ArtifactSummary[];
  copyFeedback: ArtifactActionFeedback | null;
  onArtifactAction: (action: ArtifactAction, artifact: ArtifactSummary) => void;
  transferFeedback: ArtifactActionFeedback | null;
};

function ArtifactsInspector({
  activeArtifact,
  activeArtifactDocument,
  activeArtifactId,
  artifacts,
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

function RetrievalInspector({ snapshot }: WorkstationShellProps) {
  return (
    <section
      aria-label="Retrieval inspector"
      className="inspectorPanel retrievalPanel"
      id="retrieval-inspector-panel"
      role="tabpanel"
    >
      <div className="retrievalList">
        {snapshot.retrieval.sources.map((source) => (
          <article className="retrievalItem" key={source.title}>
            <strong>{source.title}</strong>
            <p>{source.detail}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

type CommandMenuPanelProps = {
  activeTab: InspectorTabName;
  isFocusMode: boolean;
  isPreviewAvailable: boolean;
  isPreviewOpen: boolean;
  onFocusModeToggle: () => void;
  onOpenTab: (tab: InspectorTabName) => void;
  onPreviewToggle: () => void;
};

function CommandMenuPanel({
  activeTab,
  isFocusMode,
  isPreviewAvailable,
  isPreviewOpen,
  onFocusModeToggle,
  onOpenTab,
  onPreviewToggle
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
          aria-label={`${formatArtifactActionLabel(action)} ${artifact.title}`}
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

function getArtifactActionButtonLabel(
  action: ArtifactAction,
  artifact: ArtifactSummary,
  copyFeedback: ArtifactActionFeedback | null,
  transferFeedback: ArtifactActionFeedback | null
) {
  if (action === "Copy" && copyFeedback?.artifactId === artifact.id) {
    return copyFeedback.state === "success" ? "Copied" : "Copy Unavailable";
  }

  if (
    (action === "Download" || action === "Export") &&
    transferFeedback?.artifactId === artifact.id
  ) {
    if (transferFeedback.state === "success") {
      return action === "Export" ? "Exported" : "Downloaded";
    }

    return action === "Export" ? "Export Unavailable" : "Download Unavailable";
  }

  return formatArtifactActionLabel(action);
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
    const transferVerb = transferAction === "Export" ? "exported" : "downloaded";

    return transferFeedback.state === "success"
      ? `${artifact.title} ${transferVerb}.`
      : `${transferAction === "Export" ? "Export" : "Download"} is unavailable for ${artifact.title}.`;
  }

  return null;
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
