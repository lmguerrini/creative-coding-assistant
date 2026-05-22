"use client";

import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
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
  AssistantMessage,
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
  fingerprintWorkspaceSessionRecord,
  snapshotFromWorkspaceSessionRecord,
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

export function WorkstationShell({
  snapshot: initialSnapshot,
  streamAssistantEvents = streamBackendAssistantEvents,
  persistenceClient = defaultWorkspacePersistenceClient
}: WorkstationShellProps) {
  const [snapshot, setSnapshot] = useState(initialSnapshot);
  const [messages, setMessages] = useState(initialSnapshot.messages);
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
  const [copyFeedback, setCopyFeedback] = useState<ArtifactActionFeedback | null>(
    null
  );
  const [transferFeedback, setTransferFeedback] =
    useState<ArtifactActionFeedback | null>(null);
  const chatLogRef = useRef<HTMLDivElement>(null);
  const isShellMountedRef = useRef(true);
  const hasLoadedPersistenceRef = useRef(false);
  const lastPersistedFingerprintRef = useRef<string | null>(null);
  const skipNextPersistenceSaveRef = useRef(false);
  const copyFeedbackTimerRef = useRef<number | null>(null);
  const transferFeedbackTimerRef = useRef<number | null>(null);

  function clearFeedbackTimers() {
    clearTimer(copyFeedbackTimerRef.current);
    clearTimer(transferFeedbackTimerRef.current);
  }

  useEffect(() => {
    return () => {
      isShellMountedRef.current = false;
      clearFeedbackTimers();
    };
  }, []);

  useEffect(() => {
    const chatLog = chatLogRef.current;

    if (chatLog) {
      chatLog.scrollTop = chatLog.scrollHeight;
    }
  }, [messages.length]);

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
          setMessages(restoredSnapshot.messages);
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
      messages,
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
      messages,
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
        previewArtifactId,
        previewOpen: isPreviewOpen,
        snapshot: interactiveSnapshot
      }),
    [
      activeArtifactId,
      activeTab,
      interactiveSnapshot,
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
  const composerStateLabel = isStreaming
    ? "Streaming backend events"
    : streamError
      ? "Backend fallback active"
      : isComposerReady
        ? "Ready to stream"
        : "Type to stream backend";
  const persistenceStatusLabel =
    persistenceStateLabels[persistenceState] ?? "Local session ready";

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

  async function handleComposerSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const prompt = composerValue.trim();

    if (!prompt) {
      return;
    }

    const timestamp = formatMessageTime();
    const assistantPlaceholder: AssistantMessage = {
      role: "assistant",
      time: timestamp,
      content: "Connecting to backend stream..."
    };
    const newMessages: AssistantMessage[] = [
      {
        role: "user",
        time: timestamp,
        content: prompt
      },
      assistantPlaceholder
    ];

    setMessages((currentMessages) => [...currentMessages, ...newMessages]);
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
            updateLatestAssistantMessage(streamedAnswer);
          }
        }

        if (streamEvent.event_type === "final") {
          const answer = readPayloadText(streamEvent, "answer");
          if (answer) {
            streamedAnswer = answer;
            updateLatestAssistantMessage(answer);
          }
        }

        if (streamEvent.event_type === "error") {
          const message =
            readPayloadText(streamEvent, "message") ?? "Backend stream failed.";
          setStreamError(message);
          updateLatestAssistantMessage(`Backend stream error: ${message}`);
        }
      }
    } catch {
      const fallbackMessage = `Backend stream unavailable; showing local fallback. ${buildMockAssistantReply(
        prompt,
        activeArtifact.title
      )}`;
      setStreamError("Backend stream unavailable. Showing local fallback.");
      updateLatestAssistantMessage(fallbackMessage);
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

    if (streamEvent.event_type === "preview_artifact") {
      const artifactId = readPayloadText(streamEvent, "artifact_id");
      if (
        artifactId &&
        snapshot.artifacts.some((artifact) => artifact.id === artifactId)
      ) {
        setPreviewArtifactId(artifactId);
      }
      setIsPreviewOpen(true);
    }
  }

  function updateLatestAssistantMessage(content: string) {
    setMessages((currentMessages) => {
      const nextMessages = [...currentMessages];
      const assistantIndex = findLatestAssistantMessageIndex(nextMessages);

      if (assistantIndex < 0) {
        return currentMessages;
      }

      nextMessages[assistantIndex] = {
        ...nextMessages[assistantIndex],
        content
      };
      return nextMessages;
    });
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
      setIsPreviewOpen(true);
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
      data-preview={isPreviewOpen ? "open" : "closed"}
      data-stream-state={streamState}
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

        <div className="topbarActions" aria-label="Workspace actions">
          <button className="iconButton" type="button" aria-label="Command menu">
            <Command size={18} />
          </button>
          <button className="iconButton" type="button" aria-label="Dashboard">
            <LayoutGrid size={18} />
          </button>
          <button className="iconButton" type="button" aria-label="Theme">
            <Moon size={17} />
          </button>
          <button className="iconButton" type="button" aria-label="Settings">
            <Settings size={18} />
          </button>
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
              aria-live="polite"
              className="chatLog"
              ref={chatLogRef}
              role="log"
            >
              {interactiveSnapshot.messages.map((message, index) => (
                <article
                  className="message"
                  data-fresh={index >= snapshot.messages.length ? "true" : undefined}
                  data-role={message.role}
                  key={`${message.role}-${message.time}-${message.content}`}
                >
                  <div className="messageMeta">
                    <span>{message.role}</span>
                    <span>{message.time}</span>
                  </div>
                  <p>{message.content}</p>
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

          {interactiveSnapshot.preview.available ? (
            <PreviewShelf
              onToggle={setIsPreviewOpen}
              snapshot={interactiveSnapshot}
            />
          ) : null}
        </div>

        <aside className="inspector" aria-label="Right inspector">
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
            snapshot={interactiveSnapshot}
            transferFeedback={transferFeedback}
            workflowRuntime={workflowRuntime}
          />
        </aside>
      </section>
    </main>
  );
}

type PreviewShelfProps = WorkstationShellProps & {
  onToggle: (isOpen: boolean) => void;
};

function PreviewShelf({ onToggle, snapshot }: PreviewShelfProps) {
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
        <div className="previewBody">
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
    return <WorkflowInspector runtime={workflowRuntime} />;
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
      snapshot={snapshot}
    />
  );
}

function OverviewInspector({
  activeArtifact,
  runtime,
  snapshot
}: WorkstationShellProps & {
  activeArtifact: ArtifactSummary;
  runtime: WorkflowRuntimeModel;
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
            <span>{runtime.summary.traceEventCount} trace events</span>
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

function WorkflowInspector({ runtime }: { runtime: WorkflowRuntimeModel }) {
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

function findLatestAssistantMessageIndex(messages: AssistantMessage[]) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index].role === "assistant") {
      return index;
    }
  }

  return -1;
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
