"use client";

import {
  useEffect,
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

type WorkstationShellProps = {
  snapshot: AssistantWorkspaceSnapshot;
};

const inspectorTabIcons = {
  Overview: Sparkles,
  Code: Braces,
  Workflow: Activity,
  Artifacts: Boxes,
  Retrieval: TerminalSquare
} satisfies Record<InspectorTabName, LucideIcon>;

type WorkflowState = WorkflowStepState["state"];
type WorkspaceWorkflow = AssistantWorkspaceSnapshot["workflow"];

const mockWorkflowIntervalMs = 850;

export function WorkstationShell({ snapshot }: WorkstationShellProps) {
  const [messages, setMessages] = useState(snapshot.messages);
  const [composerValue, setComposerValue] = useState("");
  const [activeTab, setActiveTab] = useState<InspectorTabName>(
    getInitialActiveTab(snapshot)
  );
  const [activeArtifactId, setActiveArtifactId] = useState(
    snapshot.artifacts[0]?.id ?? ""
  );
  const [previewArtifactId, setPreviewArtifactId] = useState(
    getInitialPreviewArtifactId(snapshot)
  );
  const [isPreviewOpen, setIsPreviewOpen] = useState(snapshot.preview.active);
  const [workflowProgressIndex, setWorkflowProgressIndex] = useState(
    getInitialWorkflowIndex(snapshot.workflow.steps)
  );
  const [workflowRunId, setWorkflowRunId] = useState(0);
  const chatLogRef = useRef<HTMLDivElement>(null);

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

  const activeArtifact =
    snapshot.artifacts.find((artifact) => artifact.id === activeArtifactId) ??
    snapshot.artifacts[0];
  const previewArtifact =
    snapshot.artifacts.find((artifact) => artifact.id === previewArtifactId) ??
    activeArtifact;
  const workflow = buildInteractiveWorkflow(
    snapshot.workflow,
    workflowProgressIndex
  );
  const interactiveSnapshot: AssistantWorkspaceSnapshot = {
    ...snapshot,
    code:
      activeArtifact.type === "code"
        ? { ...snapshot.code, title: activeArtifact.title }
        : snapshot.code,
    inspectorTabs: snapshot.inspectorTabs.map((tab) => ({
      ...tab,
      active: tab.label === activeTab,
      badge: tab.label === "Artifacts" ? String(snapshot.artifacts.length) : tab.badge
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
    workflow
  };
  const activeTabSummary =
    interactiveSnapshot.inspectorTabs.find((tab) => tab.label === activeTab)
      ?.summary ?? "";

  function handleComposerSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const prompt = composerValue.trim();

    if (!prompt) {
      return;
    }

    const timestamp = formatMessageTime();
    const newMessages: AssistantMessage[] = [
      {
        role: "user",
        time: timestamp,
        content: prompt
      },
      {
        role: "assistant",
        time: timestamp,
        content: buildMockAssistantReply(prompt, activeArtifact.title)
      }
    ];

    setMessages((currentMessages) => [...currentMessages, ...newMessages]);
    setComposerValue("");
    setWorkflowProgressIndex(0);
    setWorkflowRunId((currentRunId) => currentRunId + 1);
    setActiveTab("Overview");
  }

  function handleArtifactAction(action: ArtifactAction, artifact: ArtifactSummary) {
    setActiveArtifactId(artifact.id);

    if (action === "Open") {
      setActiveTab(artifact.type === "code" ? "Code" : "Artifacts");
      return;
    }

    if (action === "Preview") {
      setPreviewArtifactId(artifact.id);
      setIsPreviewOpen(true);
      setActiveTab("Overview");
      return;
    }

    setActiveTab("Artifacts");
  }

  return (
    <main className="workstation">
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

        <div className="sessionStatus" aria-label="Current session">
          <span>{interactiveSnapshot.workflow.status}</span>
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
              <div className="sessionMetric" aria-label="Active artifact">
                <span>Active artifact</span>
                <strong>{activeArtifact.title}</strong>
              </div>
            </header>

            <div
              aria-label="Conversation"
              aria-live="polite"
              className="chatLog"
              ref={chatLogRef}
              role="log"
            >
              {interactiveSnapshot.messages.map((message) => (
                <article
                  className="message"
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

            <form className="composer" onSubmit={handleComposerSubmit}>
              <textarea
                aria-label="Assistant prompt"
                onChange={(event) => setComposerValue(event.currentTarget.value)}
                placeholder="Ask for a denser particle field, a softer palette, or a preview pass."
                value={composerValue}
              />
              <button
                aria-label="Send prompt"
                className="sendButton"
                disabled={!composerValue.trim()}
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
            activeArtifactId={activeArtifactId}
            activeTab={activeTab}
            onArtifactAction={handleArtifactAction}
            snapshot={interactiveSnapshot}
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
  activeArtifactId: string;
  activeTab: InspectorTabName;
  onArtifactAction: (action: ArtifactAction, artifact: ArtifactSummary) => void;
  snapshot: AssistantWorkspaceSnapshot;
};

function InspectorPanel({
  activeArtifact,
  activeArtifactId,
  activeTab,
  onArtifactAction,
  snapshot
}: InspectorPanelProps) {
  if (activeTab === "Code") {
    return <CodeInspector snapshot={snapshot} />;
  }

  if (activeTab === "Workflow") {
    return <WorkflowInspector snapshot={snapshot} />;
  }

  if (activeTab === "Artifacts") {
    return (
      <ArtifactsInspector
        activeArtifactId={activeArtifactId}
        artifacts={snapshot.artifacts}
        onArtifactAction={onArtifactAction}
      />
    );
  }

  if (activeTab === "Retrieval") {
    return <RetrievalInspector snapshot={snapshot} />;
  }

  return <OverviewInspector activeArtifact={activeArtifact} snapshot={snapshot} />;
}

function OverviewInspector({
  activeArtifact,
  snapshot
}: WorkstationShellProps & { activeArtifact: ArtifactSummary }) {
  const activeStep = snapshot.workflow.steps.find((step) => step.state === "active");

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
              <strong>{snapshot.workflow.currentStep}</strong>
              <p>{activeStep?.displayLabel ?? "Waiting for next node"}</p>
            </div>
            <span className="liveDot" aria-hidden="true" />
          </header>
          <div className="miniWorkflow" aria-label="Minimal live workflow state">
            {snapshot.workflow.steps.map((step) => (
              <div
                aria-current={step.state === "active" ? "step" : undefined}
                className="miniStep"
                data-state={step.state}
                key={step.nodeId}
              >
                <span aria-hidden="true" />
                <strong>{step.displayLabel}</strong>
              </div>
            ))}
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

function CodeInspector({ snapshot }: WorkstationShellProps) {
  return (
    <section
      aria-label="Code inspector"
      className="inspectorPanel codePanel"
      id="code-inspector-panel"
      role="tabpanel"
    >
      <header className="codePanelHeader">
        <div>
          <strong>{snapshot.code.title}</strong>
          <span>
            {snapshot.code.language} / {snapshot.code.status}
          </span>
        </div>
        <span className="artifactType">code</span>
      </header>
      <pre>
        <code>{snapshot.code.excerpt.join("\n")}</code>
      </pre>
    </section>
  );
}

function WorkflowInspector({ snapshot }: WorkstationShellProps) {
  return (
    <section
      aria-label="Workflow inspector"
      className="inspectorPanel workflowPanel"
      id="workflow-inspector-panel"
      role="tabpanel"
    >
      <div
        aria-label="LangGraph workflow visualization"
        className="workflowGraph"
        role="group"
      >
        {snapshot.workflow.steps.map((step, index) => (
          <article
            aria-current={step.state === "active" ? "step" : undefined}
            className="workflowNode"
            data-state={step.state}
            key={step.nodeId}
          >
            <span className="nodeIndex">{String(index + 1).padStart(2, "0")}</span>
            <div>
              <strong>{step.displayLabel}</strong>
              <p>
                <code>{step.nodeId}</code>
                <span>{formatWorkflowState(step.state)}</span>
              </p>
              <small>{step.detail}</small>
            </div>
          </article>
        ))}
      </div>
      <div className="loopHint">
        <span aria-hidden="true" />
        <p>{"Real retry edge: refinement -> generation, bounded by review state."}</p>
      </div>
    </section>
  );
}

type ArtifactsInspectorProps = {
  activeArtifactId: string;
  artifacts: ArtifactSummary[];
  onArtifactAction: (action: ArtifactAction, artifact: ArtifactSummary) => void;
};

function ArtifactsInspector({
  activeArtifactId,
  artifacts,
  onArtifactAction
}: ArtifactsInspectorProps) {
  return (
    <section
      aria-label="Artifacts inspector"
      className="inspectorPanel artifactPanel"
      id="artifacts-inspector-panel"
      role="tabpanel"
    >
      <div className="artifactList">
        {artifacts.map((artifact) => (
          <ArtifactCard
            artifact={artifact}
            isActive={artifact.id === activeArtifactId}
            key={artifact.id}
            onArtifactAction={onArtifactAction}
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
  isActive: boolean;
  onArtifactAction: (action: ArtifactAction, artifact: ArtifactSummary) => void;
};

function ArtifactCard({
  artifact,
  isActive,
  onArtifactAction
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
            {artifact.language} / {artifact.status}
          </span>
        </div>
        <span className="artifactType">{artifact.type}</span>
      </div>
      <p>{artifact.summary}</p>
      <div className="artifactActions">
        {artifact.actions.map((action) => (
          <button
            key={action}
            onClick={() => onArtifactAction(action, artifact)}
            type="button"
            aria-label={`${action} ${artifact.title}`}
          >
            {action}
          </button>
        ))}
      </div>
    </article>
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

function formatWorkflowState(state: WorkflowState) {
  return state === "queued" ? "pending" : state;
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

export const workstationIcons = {
  code: Braces,
  context: PanelRight,
  preview: Play
};
