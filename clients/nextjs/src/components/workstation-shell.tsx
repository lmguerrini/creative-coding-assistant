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
  ArtifactSummary,
  AssistantWorkspaceSnapshot,
  InspectorTabName
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

export function WorkstationShell({ snapshot }: WorkstationShellProps) {
  const activeArtifact = snapshot.artifacts[0];
  const activeTab =
    snapshot.inspectorTabs.find((tab) => tab.active)?.label ?? "Overview";
  const activeTabSummary =
    snapshot.inspectorTabs.find((tab) => tab.label === activeTab)?.summary ?? "";

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
          <span>{snapshot.workflow.status}</span>
          <strong>{snapshot.workflow.currentStep}</strong>
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

            <div className="chatLog">
              {snapshot.messages.map((message) => (
                <article
                  className="message"
                  data-role={message.role}
                  key={`${message.role}-${message.time}`}
                >
                  <div className="messageMeta">
                    <span>{message.role}</span>
                    <span>{message.time}</span>
                  </div>
                  <p>{message.content}</p>
                </article>
              ))}
            </div>

            <form className="composer">
              <textarea
                aria-label="Assistant prompt"
                placeholder="Ask for a denser particle field, a softer palette, or a preview pass."
              />
              <button className="sendButton" type="button" aria-label="Send prompt">
                <SendHorizontal size={18} />
              </button>
            </form>
          </section>

          {snapshot.preview.available ? (
            <PreviewShelf snapshot={snapshot} />
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
            {snapshot.inspectorTabs.map((tab) => {
              const Icon = inspectorTabIcons[tab.label];

              return (
                <button
                  aria-controls={`${tab.label.toLowerCase()}-inspector-panel`}
                  aria-label={tab.label}
                  aria-selected={tab.active}
                  id={`${tab.label.toLowerCase()}-inspector-tab`}
                  key={tab.label}
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

          <InspectorPanel activeTab={activeTab} snapshot={snapshot} />
        </aside>
      </section>
    </main>
  );
}

function PreviewShelf({ snapshot }: WorkstationShellProps) {
  return (
    <section className="previewZone" aria-label="Preview workspace">
      <details className="previewShelf" open={!snapshot.preview.collapsed}>
        <summary>
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
  activeTab: InspectorTabName;
  snapshot: AssistantWorkspaceSnapshot;
};

function InspectorPanel({ activeTab, snapshot }: InspectorPanelProps) {
  if (activeTab === "Code") {
    return <CodeInspector snapshot={snapshot} />;
  }

  if (activeTab === "Workflow") {
    return <WorkflowInspector snapshot={snapshot} />;
  }

  if (activeTab === "Artifacts") {
    return <ArtifactsInspector artifacts={snapshot.artifacts} />;
  }

  if (activeTab === "Retrieval") {
    return <RetrievalInspector snapshot={snapshot} />;
  }

  return <OverviewInspector snapshot={snapshot} />;
}

function OverviewInspector({ snapshot }: WorkstationShellProps) {
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
              <div className="miniStep" data-state={step.state} key={step.nodeId}>
                <span aria-hidden="true" />
                <strong>{step.displayLabel}</strong>
              </div>
            ))}
          </div>
        </div>
        <div className="overviewTile" role="group" aria-label="Artifacts summary">
          <span>Artifacts</span>
          <strong>{snapshot.artifacts.length}</strong>
          <p>{snapshot.artifacts[0].title}</p>
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
          <article className="workflowNode" data-state={step.state} key={step.nodeId}>
            <span className="nodeIndex">{String(index + 1).padStart(2, "0")}</span>
            <div>
              <strong>{step.displayLabel}</strong>
              <p>
                <code>{step.nodeId}</code>
                <span>{step.state}</span>
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

function ArtifactsInspector({ artifacts }: { artifacts: ArtifactSummary[] }) {
  return (
    <section
      aria-label="Artifacts inspector"
      className="inspectorPanel artifactPanel"
      id="artifacts-inspector-panel"
      role="tabpanel"
    >
      <div className="artifactList">
        {artifacts.map((artifact) => (
          <ArtifactCard artifact={artifact} key={artifact.id} />
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

function ArtifactCard({ artifact }: { artifact: ArtifactSummary }) {
  return (
    <article className="artifactItem">
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

export const workstationIcons = {
  code: Braces,
  context: PanelRight,
  preview: Play
};
