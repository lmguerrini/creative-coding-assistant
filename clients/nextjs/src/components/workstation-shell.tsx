import {
  Activity,
  Boxes,
  Braces,
  Command,
  FlaskConical,
  LayoutGrid,
  PanelRight,
  Play,
  RefreshCw,
  SendHorizontal,
  Sparkles,
  TerminalSquare
} from "lucide-react";
import type {
  AssistantWorkspaceSnapshot,
  WorkflowStepState
} from "@/lib/assistant-client";

type WorkstationShellProps = {
  snapshot: AssistantWorkspaceSnapshot;
};

export function WorkstationShell({ snapshot }: WorkstationShellProps) {
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

        <nav className="modeSwitch" aria-label="Assistant mode">
          {snapshot.modes.map((mode) => (
            <button
              key={mode.label}
              type="button"
              aria-pressed={mode.active}
              title={mode.label}
            >
              {mode.label}
            </button>
          ))}
        </nav>

        <div className="topbarActions" aria-label="Workspace actions">
          <button className="iconButton" type="button" aria-label="Command menu">
            <Command size={18} />
          </button>
          <button className="iconButton" type="button" aria-label="Refresh state">
            <RefreshCw size={17} />
          </button>
          <button className="iconButton" type="button" aria-label="Panel layout">
            <PanelRight size={18} />
          </button>
        </div>
      </header>

      <section className="shellGrid" aria-label="Creative workflow workspace">
        <div className="stack">
          <Panel
            title="Chat Workflow"
            subtitle={snapshot.workflow.currentStep}
            status={snapshot.workflow.status}
            icon={<Activity size={17} />}
          >
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
                placeholder="Generate a reactive WebGPU sketch with audio-linked motion."
              />
              <button className="sendButton" type="button" aria-label="Send prompt">
                <SendHorizontal size={18} />
              </button>
            </form>

            <WorkflowTimeline steps={snapshot.workflow.steps} />
          </Panel>
        </div>

        <PreviewPanel snapshot={snapshot} />

        <div className="stack">
          <Panel
            title="Artifacts"
            subtitle={`${snapshot.artifacts.length} workspace outputs`}
            status="Indexed"
            icon={<Boxes size={17} />}
          >
            <div className="artifactList">
              {snapshot.artifacts.map((artifact) => (
                <article className="artifactItem" key={artifact.id}>
                  <div className="artifactItemHeader">
                    <strong>{artifact.title}</strong>
                    <span className="artifactType">{artifact.type}</span>
                  </div>
                  <p>{artifact.summary}</p>
                </article>
              ))}
            </div>
          </Panel>

          <Panel
            title="Workflow Debug"
            subtitle={snapshot.debug.traceId}
            status={snapshot.debug.status}
            icon={<TerminalSquare size={17} />}
          >
            <div className="debugList">
              {snapshot.debug.events.map((event) => (
                <article className="debugRow" key={event.code}>
                  <div className="debugRowHeader">
                    <strong>{event.label}</strong>
                    <span className="debugCode">{event.code}</span>
                  </div>
                  <p>{event.detail}</p>
                </article>
              ))}
            </div>
          </Panel>
        </div>
      </section>
    </main>
  );
}

type PanelProps = {
  title: string;
  subtitle: string;
  status: string;
  icon: React.ReactNode;
  children: React.ReactNode;
};

function Panel({ title, subtitle, status, icon, children }: PanelProps) {
  return (
    <section className="panel" aria-label={title}>
      <header className="panelHeader">
        <div className="panelTitle">
          <span className="panelTitleIcon" aria-hidden="true">
            {icon}
          </span>
          <div>
            <h2>{title}</h2>
            <p>{subtitle}</p>
          </div>
        </div>
        <span className="panelStatus">{status}</span>
      </header>
      <div className="panelBody">{children}</div>
    </section>
  );
}

function PreviewPanel({ snapshot }: WorkstationShellProps) {
  return (
    <section className="previewPanel" aria-label="Preview">
      <header className="previewHeader">
        <div className="panelTitle">
          <span className="panelTitleIcon" aria-hidden="true">
            <Play size={17} />
          </span>
          <div className="previewTitle">
            <h2>{snapshot.preview.title}</h2>
            <p>{snapshot.preview.target}</p>
          </div>
        </div>
        <span className="panelStatus">{snapshot.preview.status}</span>
      </header>

      <div className="previewStage">
        <div className="previewCanvas" aria-label="Preview stage placeholder">
          <div className="previewCanvasContent">
            <strong>{snapshot.preview.artifactName}</strong>
            <span>{snapshot.preview.summary}</span>
          </div>
        </div>
      </div>

      <footer className="previewFooter">
        <div className="previewStats">
          <strong>{snapshot.preview.renderer}</strong>
          <p>Renderer</p>
        </div>
        <div className="previewStats">
          <strong>{snapshot.preview.latency}</strong>
          <p>Latency</p>
        </div>
        <div className="previewStats">
          <strong>{snapshot.preview.version}</strong>
          <p>Artifact</p>
        </div>
      </footer>
    </section>
  );
}

function WorkflowTimeline({ steps }: { steps: WorkflowStepState[] }) {
  return (
    <div className="timeline" aria-label="Workflow timeline">
      {steps.map((step) => (
        <div className="timelineStep" data-state={step.state} key={step.name}>
          <span className="stepDot" aria-hidden="true" />
          <span className="stepName">{step.name}</span>
          <span className="stepState">{step.state}</span>
        </div>
      ))}
    </div>
  );
}

export const workstationIcons = {
  code: Braces,
  lab: FlaskConical,
  layout: LayoutGrid
};
