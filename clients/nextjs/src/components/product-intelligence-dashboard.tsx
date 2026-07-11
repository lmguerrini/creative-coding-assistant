"use client";

import { useEffect, useState } from "react";
import { CircleHelp, X } from "lucide-react";
import {
  DomainExperienceSurface,
  KnowledgeBaseInventorySurface
} from "./domain-experience-surface";
import {
  getProductIntelligenceSection,
  type ProductIntelligenceCategory,
  type ProductIntelligenceModel,
  type ProductIntelligenceSection
} from "@/lib/product-intelligence";
import type {
  WorkspaceLayoutState,
  WorkspacePreferences,
  WorkspaceSessionSummary
} from "@/lib/workspace-persistence";
import type { SessionUsageSummary } from "@/lib/session-usage-ledger";
import { formatUiStatusLabel } from "@/lib/ui-copy";
import type {
  EvaluationHistoryRecord,
  FeedbackSentiment
} from "@/lib/product-controls";
import { OutputFeedbackPanel } from "./output-feedback-panel";

type DashboardFeedback = {
  artifactTitle: string;
  onSubmit: (sentiment: FeedbackSentiment, comment: string | null) => void;
};

type DashboardSettingsControls = {
  isFocusMode: boolean;
  isPreviewOpen: boolean;
  layoutState: WorkspaceLayoutState;
  onDensityChange: (density: WorkspaceLayoutState["density"]) => void;
  onFocusModeToggle: () => void;
  onInspectorToggle: () => void;
  onPreferencesChange: (preferences: Partial<WorkspacePreferences>) => void;
  onPreviewToggle: () => void;
  onSidebarToggle: () => void;
  preferences: WorkspacePreferences;
};

type DashboardSessionControls = {
  activeSessionId: string;
  onCreate: () => void;
  onDelete: (sessionId: string) => void;
  onRename: (sessionId: string, title: string) => void;
  onSelect: (sessionId: string) => void;
  sessions: WorkspaceSessionSummary[];
  usage: SessionUsageSummary[];
};

type ProductIntelligenceDashboardProps = {
  activeCategory: ProductIntelligenceCategory;
  model: ProductIntelligenceModel;
  onCategoryChange: (category: ProductIntelligenceCategory) => void;
  onClose: () => void;
  onRunEvaluation?: () => Promise<void>;
  evaluationHistory?: EvaluationHistoryRecord[];
  feedback?: DashboardFeedback;
  settings?: DashboardSettingsControls;
  sessions?: DashboardSessionControls;
};

type DashboardGroupId =
  | "overview"
  | "architecture"
  | "workflow"
  | "workspace"
  | "runtime"
  | "preview"
  | "artifacts"
  | "domains"
  | "knowledge"
  | "ai_agents"
  | "memory"
  | "sessions"
  | "telemetry"
  | "evaluation"
  | "manual"
  | "settings";

type DashboardGroup = {
  id: DashboardGroupId;
  label: string;
  detail: string;
  categories: ProductIntelligenceCategory[];
  secondary?: boolean;
};

const dashboardGroups: DashboardGroup[] = [
  {
    id: "overview",
    label: "Overview",
    detail: "Current workspace outcome and selected artifact.",
    categories: ["Overview"]
  },
  {
    id: "architecture",
    label: "Architecture",
    detail: "How single- and multi-agent routes are assembled.",
    categories: ["Architecture"]
  },
  {
    id: "workflow",
    label: "Workflow",
    detail: "Live progress, transitions, and recovery state.",
    categories: ["Workflow"]
  },
  {
    id: "workspace",
    label: "Workspace",
    detail: "Generated source and the active creative document.",
    categories: ["Code"]
  },
  {
    id: "runtime",
    label: "Runtime",
    detail: "Renderer health, diagnostics, and recovery state.",
    categories: ["Runtime"]
  },
  {
    id: "preview",
    label: "Preview",
    detail: "Browser output, renderer route, and ready state.",
    categories: ["Preview"]
  },
  {
    id: "artifacts",
    label: "Artifacts",
    detail: "Every generated deliverable, source excerpt, and session.",
    categories: ["Artifacts"]
  },
  {
    id: "domains",
    label: "Domains",
    detail: "What can run live, export source, or hand off externally.",
    categories: ["Domains"]
  },
  {
    id: "knowledge",
    label: "Knowledge Base",
    detail: "Official sources, index coverage, and freshness controls.",
    categories: ["Knowledge Base", "Retrieval"]
  },
  {
    id: "ai_agents",
    label: "AI & agents",
    detail: "Provider route, model context, and agent responsibilities.",
    categories: ["Agents", "Providers"]
  },
  {
    id: "memory",
    label: "Memory",
    detail: "Published context counts and privacy-safe session history.",
    categories: ["Memory"]
  },
  {
    id: "sessions",
    label: "Sessions",
    detail: "Manage saved creative sessions and their artifacts.",
    categories: ["Sessions"]
  },
  {
    id: "telemetry",
    label: "Telemetry",
    detail: "Usage, token accounting, cost estimates, and runtime signals.",
    categories: ["Telemetry", "Metrics"]
  },
  {
    id: "evaluation",
    label: "Evaluation",
    detail: "Validation, product signals, and observable traces.",
    categories: ["Validation", "Product Bugs", "LangSmith"]
  },
  {
    id: "manual",
    label: "Manual guide",
    detail: "A concise guide to inspecting a creative run.",
    categories: [],
    secondary: true
  },
  {
    id: "settings",
    label: "Settings",
    detail: "Workspace display, density, focus, and provider configuration.",
    categories: ["Settings"],
    secondary: true
  }
];

export function ProductIntelligenceDashboard({
  activeCategory,
  model,
  onCategoryChange,
  onClose,
  onRunEvaluation,
  evaluationHistory = [],
  feedback,
  settings,
  sessions
}: ProductIntelligenceDashboardProps) {
  const [activeGroupId, setActiveGroupId] = useState<DashboardGroupId>(() =>
    getDashboardGroup(activeCategory).id
  );
  const activeGroup =
    dashboardGroups.find((group) => group.id === activeGroupId) ?? dashboardGroups[0];
  const primarySection = activeGroup.categories[0]
    ? getProductIntelligenceSection(model, activeGroup.categories[0])
    : null;
  const [evaluationRunning, setEvaluationRunning] = useState(false);

  async function runEvaluation() {
    if (!onRunEvaluation || evaluationRunning) {
      return;
    }
    setEvaluationRunning(true);
    try {
      await onRunEvaluation();
    } finally {
      setEvaluationRunning(false);
    }
  }

  useEffect(() => {
    if (activeGroupId !== "manual") {
      setActiveGroupId(getDashboardGroup(activeCategory).id);
    }
  }, [activeCategory, activeGroupId]);

  function selectGroup(group: DashboardGroup) {
    setActiveGroupId(group.id);
    if (group.categories[0]) {
      onCategoryChange(group.categories[0]);
    }
  }

  return (
    <section aria-label="Advanced Dashboard" className="productDashboard">
      <nav aria-label="Dashboard categories" className="productDashboardNav">
        <header>
          <span>Advanced Dashboard</span>
          <strong>Workspace intelligence</strong>
          <p>Detailed state, grouped by the decisions you need to make.</p>
        </header>
        <div role="list">
          {dashboardGroups.map((group) => {
            const item = group.categories[0]
              ? getProductIntelligenceSection(model, group.categories[0])
              : null;
            return (
              <button
                aria-current={group.id === activeGroup.id ? "page" : undefined}
                className={group.secondary ? "productDashboardSecondary" : undefined}
                data-tone={item?.tone ?? "empty"}
                key={group.id}
                onClick={() => selectGroup(group)}
                type="button"
              >
                <span>{group.label}</span>
                <small>{group.detail}</small>
              </button>
            );
          })}
        </div>
      </nav>
      <div className="productDashboardContent">
        <header className="productDashboardContentHeader">
          <div>
            <span>Advanced Dashboard</span>
            <h1>{activeGroup.label}</h1>
            <p>{activeGroup.detail}</p>
          </div>
          {primarySection ? <ProductIntelligenceHelp section={primarySection} /> : null}
          <div className="productDashboardStatus" data-tone={primarySection?.tone ?? "empty"}>
            {formatUiStatusLabel(primarySection?.summary ?? "Guide")}
          </div>
          <button
            aria-label="Close dashboard"
            onClick={onClose}
            title="Close dashboard"
            type="button"
          >
            <X aria-hidden="true" size={18} />
          </button>
        </header>
        <DashboardGroupView
          evaluationRunning={evaluationRunning}
          group={activeGroup}
          model={model}
          onRunEvaluation={onRunEvaluation ? runEvaluation : undefined}
          evaluationHistory={evaluationHistory}
          feedback={feedback}
          settings={settings}
          sessions={sessions}
        />
      </div>
    </section>
  );
}

function DashboardGroupView({
  evaluationRunning,
  group,
  model,
  onRunEvaluation,
  evaluationHistory,
  feedback,
  settings,
  sessions
}: {
  evaluationRunning: boolean;
  group: DashboardGroup;
  model: ProductIntelligenceModel;
  onRunEvaluation?: () => Promise<void>;
  evaluationHistory: EvaluationHistoryRecord[];
  feedback?: DashboardFeedback;
  settings?: DashboardSettingsControls;
  sessions?: DashboardSessionControls;
}) {
  if (group.id === "manual") {
    return (
      <section aria-label="Workspace manual" className="productDashboardManual">
        <article>
          <span>Start a run</span>
          <strong>Describe a visual system, then choose a workflow route.</strong>
          <p>Use the workspace for the conversation; open Preview, Code, or Saved only when they add context.</p>
        </article>
        <article>
          <span>Read the result</span>
          <strong>Check the artifact, visible output, and runtime health separately.</strong>
          <p>Advanced Dashboard keeps diagnostics, source evidence, and workflow detail together without crowding the creative session.</p>
        </article>
        <article>
          <span>Keep boundaries honest</span>
          <strong>Live preview, code/export, and external-tool handoff are distinct outcomes.</strong>
          <p>Use the Domain and Knowledge sections to confirm what is available in this browser workspace.</p>
        </article>
      </section>
    );
  }

  if (group.id === "settings" && settings) {
    return <DashboardSettings controls={settings} />;
  }

  return (
    <div className="productDashboardGroup" aria-label={`${group.label} details`}>
      {group.categories.map((category) => {
        const section = getProductIntelligenceSection(model, category);
        return (
          <section className="productDashboardGroupSection" key={category}>
            <header>
              <div>
                <span>{category}</span>
                <strong>{formatUiStatusLabel(section.summary)}</strong>
                <p>{section.detail}</p>
              </div>
              <ProductIntelligenceHelp section={section} />
            </header>
            {category === "Metrics" && onRunEvaluation ? (
              <>
                <button
                  className="evaluationRunButton"
                  disabled={evaluationRunning}
                  onClick={() => void onRunEvaluation()}
                  type="button"
                >
                  {evaluationRunning ? "Preparing evaluation…" : "Run Evaluation"}
                </button>
                <EvaluationHistory history={evaluationHistory} />
              </>
            ) : null}
            <ProductIntelligenceSectionView detailed model={model} section={section} />
          </section>
        );
      })}
      {group.id === "architecture" ? <ArchitectureRouteGuide /> : null}
      {group.id === "knowledge" ? <KnowledgePrinciples /> : null}
      {group.id === "artifacts" ? <ArtifactRegistry model={model} /> : null}
      {group.id === "sessions" && sessions ? <SessionRegistry controls={sessions} /> : null}
      {group.id === "telemetry" && sessions ? <UserUsageOverview usage={sessions.usage} /> : null}
      {group.id === "ai_agents" && feedback ? (
        <section className="productDashboardGroupSection productDashboardFeedback">
          <header>
            <span>Feedback</span>
            <strong>Shape future creative requests</strong>
            <p>Explicit feedback stays in this local workspace profile.</p>
          </header>
          <OutputFeedbackPanel
            artifactTitle={feedback.artifactTitle}
            onSubmit={feedback.onSubmit}
          />
        </section>
      ) : null}
    </div>
  );
}

function ArchitectureRouteGuide() {
  const routes = [
    {
      title: "Single agent",
      detail: "One focused path keeps a contained request direct and fast.",
      nodes: ["Understand", "Plan", "Create", "Review"]
    },
    {
      title: "Multi-agent",
      detail: "Specialist roles add retrieval and critique when the brief needs it.",
      nodes: ["Understand", "Research", "Create", "Critique", "Review"]
    },
    {
      title: "Auto",
      detail: "The route is selected from the request’s scope and evidence needs.",
      nodes: ["Assess", "Choose route", "Run safely", "Review"]
    }
  ];

  return (
    <section aria-label="Workflow route guide" className="dashboardFeature architectureRouteGuide">
      <header>
        <div>
          <span>Route guide</span>
          <strong>Three clear ways work can move through the studio</strong>
          <p>The live Architecture card above reports the route used for the current run.</p>
        </div>
      </header>
      <div className="architectureRouteGrid">
        {routes.map((route) => (
          <article key={route.title}>
            <strong>{route.title}</strong>
            <p>{route.detail}</p>
            <ol aria-label={`${route.title} workflow`}>
              {route.nodes.map((node, index) => (
                <li key={node}>
                  <span>{index + 1}</span>
                  {node}
                </li>
              ))}
            </ol>
          </article>
        ))}
      </div>
    </section>
  );
}

function KnowledgePrinciples() {
  const principles = [
    ["Composition", "Balance focus, hierarchy, negative space, and framing."],
    ["Light & colour", "Use contrast, palette rhythm, and legible visual depth."],
    ["Motion & sound", "Shape timing, repetition, response, and emotional cadence."],
    ["Interaction", "Make controls discoverable and the response easy to read."],
    ["Runtime care", "Prefer browser-safe choices and state fallback boundaries clearly."]
  ];

  return (
    <section aria-label="Creative design principles" className="dashboardFeature knowledgePrinciples">
      <header>
        <div>
          <span>Creative principles</span>
          <strong>Visible creative criteria, separate from the source inventory</strong>
          <p>These are the product’s readable design lenses—not hidden provider reasoning.</p>
        </div>
      </header>
      <ul>
        {principles.map(([label, detail]) => (
          <li key={label}>
            <strong>{label}</strong>
            <span>{detail}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}

function ArtifactRegistry({ model }: { model: ProductIntelligenceModel }) {
  if (model.artifactRegistry.length === 0) {
    return (
      <section aria-label="Artifact registry" className="dashboardFeature dashboardEmptyState">
        <strong>No artifacts in this session yet</strong>
        <p>Generated source and export handoffs will appear here with their session and code excerpt.</p>
      </section>
    );
  }

  return (
    <section aria-label="Artifact registry" className="dashboardFeature artifactRegistry">
      <header>
        <div>
          <span>Artifact registry</span>
          <strong>{model.artifactRegistry.length} saved deliverable{model.artifactRegistry.length === 1 ? "" : "s"}</strong>
          <p>Every entry names its source session and keeps a readable, bounded source excerpt.</p>
        </div>
      </header>
      <div role="list">
        {model.artifactRegistry.map((artifact) => (
          <article key={artifact.id} role="listitem">
            <header>
              <div>
                <span>{artifact.type}</span>
                <strong>{artifact.title}</strong>
                <p>{artifact.summary}</p>
              </div>
              <span data-status={artifact.status}>{artifact.status}</span>
            </header>
            <dl>
              <div><dt>Session</dt><dd>{model.session.title}</dd></div>
              <div><dt>Language</dt><dd>{artifact.language}</dd></div>
              <div><dt>Preview</dt><dd>{artifact.previewEligible ? "Available" : "Not available"}</dd></div>
            </dl>
            <details>
              <summary>View source excerpt</summary>
              <pre><code>{artifactSnippet(artifact.content)}</code></pre>
            </details>
          </article>
        ))}
      </div>
    </section>
  );
}

function artifactSnippet(content: string | undefined) {
  if (!content?.trim()) {
    return "Source was not retained for this artifact.";
  }
  const lines = content.trim().split("\n");
  return `${lines.slice(0, 18).join("\n")}${lines.length > 18 ? "\n…" : ""}`;
}

function SessionRegistry({ controls }: { controls: DashboardSessionControls }) {
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [title, setTitle] = useState("");

  function beginRename(session: WorkspaceSessionSummary) {
    setEditingSessionId(session.sessionId);
    setTitle(session.title);
  }

  function saveRename(sessionId: string) {
    const nextTitle = title.trim();
    if (nextTitle) {
      controls.onRename(sessionId, nextTitle);
    }
    setEditingSessionId(null);
    setTitle("");
  }

  return (
    <section aria-label="Session registry" className="dashboardFeature sessionRegistry">
      <header>
        <div>
          <span>Session registry</span>
          <strong>{controls.sessions.length} local creative session{controls.sessions.length === 1 ? "" : "s"}</strong>
          <p>Sessions are stored for this browser profile. Rename, open, or remove them here.</p>
        </div>
        <button onClick={controls.onCreate} type="button">New session</button>
      </header>
      <div className="sessionRegistryTable" role="table" aria-label="Saved sessions">
        <div role="row">
          <span role="columnheader">Session</span>
          <span role="columnheader">Artifacts</span>
          <span role="columnheader">Tokens</span>
          <span role="columnheader">Last activity</span>
          <span role="columnheader">Actions</span>
        </div>
        {controls.sessions.map((session) => {
          const isEditing = editingSessionId === session.sessionId;
          const isActive = session.sessionId === controls.activeSessionId;
          const usage = controls.usage.find((entry) => entry.sessionId === session.sessionId);
          return (
            <div data-active={isActive ? "true" : "false"} key={session.sessionId} role="row">
              <div role="cell">
                {isEditing ? (
                  <form onSubmit={(event) => { event.preventDefault(); saveRename(session.sessionId); }}>
                    <input aria-label={`Session name for ${session.title}`} autoFocus onChange={(event) => setTitle(event.currentTarget.value)} value={title} />
                    <button type="submit">Save</button>
                  </form>
                ) : (
                  <button onClick={() => controls.onSelect(session.sessionId)} type="button">
                    {session.title}{isActive ? " · Current" : ""}
                  </button>
                )}
              </div>
              <span role="cell">{session.artifactCount}</span>
              <span role="cell">
                {usage?.totalTokens != null
                  ? `${formatCompactUsage(usage.totalTokens)} · ${usage.knownTokenRunCount}/${usage.runCount} runs`
                  : "Not reported"}
              </span>
              <time dateTime={session.updatedAt ?? undefined} role="cell">{formatSessionTimestamp(session.updatedAt)}</time>
              <div role="cell">
                <button onClick={() => beginRename(session)} type="button">Rename</button>
                <button onClick={() => controls.onDelete(session.sessionId)} type="button">Delete</button>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function UserUsageOverview({ usage }: { usage: SessionUsageSummary[] }) {
  const knownTokenSessions = usage.filter((entry) => entry.totalTokens != null);
  const knownCostSessions = usage.filter((entry) => entry.totalCost != null);
  const totalTokens = knownTokenSessions.reduce(
    (total, entry) => total + (entry.totalTokens ?? 0),
    0
  );
  const totalCost = knownCostSessions.reduce(
    (total, entry) => total + (entry.totalCost ?? 0),
    0
  );
  const runCount = usage.reduce((total, entry) => total + entry.runCount, 0);

  return (
    <section aria-label="Browser profile usage" className="dashboardFeature userUsageOverview">
      <header>
        <div>
          <span>Browser profile totals</span>
          <strong>Usage retained across local sessions</strong>
          <p>Only provider-published token and cost data is counted. Missing provider fields remain visibly unreported.</p>
        </div>
      </header>
      <dl>
        <div><dt>Sessions</dt><dd>{usage.length}</dd></div>
        <div><dt>Completed runs</dt><dd>{runCount}</dd></div>
        <div><dt>Known tokens</dt><dd>{knownTokenSessions.length ? formatCompactUsage(totalTokens) : "Not reported"}</dd></div>
        <div><dt>Known cost</dt><dd>{knownCostSessions.length ? `$${totalCost.toFixed(4)}` : "Not reported"}</dd></div>
      </dl>
    </section>
  );
}

function formatSessionTimestamp(value: string | null) {
  if (!value) return "Not recorded";
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function formatCompactUsage(value: number) {
  return new Intl.NumberFormat(undefined, {
    notation: "compact",
    maximumFractionDigits: 1
  }).format(value);
}

function DashboardSettings({ controls }: { controls: DashboardSettingsControls }) {
  const { layoutState, preferences } = controls;
  const themes: WorkspacePreferences["theme"][] = ["aqua", "codex", "light", "matrix", "terminal", "horizon", "zen", "blueprint"];
  const scales: WorkspacePreferences["uiFontSize"][] = ["small", "medium", "large"];

  return (
    <section aria-label="Dashboard settings" className="dashboardSettings">
      <article>
        <header><span>Appearance</span><strong>Theme and colour</strong><p>The selected theme controls the accessible text, surface, and accent colours for this saved session.</p></header>
        <div className="dashboardThemeChoices" role="group" aria-label="Theme options">
          {themes.map((theme) => (
            <button aria-pressed={preferences.theme === theme} data-theme={theme} key={theme} onClick={() => controls.onPreferencesChange({ theme })} type="button">{theme}</button>
          ))}
        </div>
      </article>
      <article>
        <header><span>Typography</span><strong>Comfortable reading</strong><p>Adjust the application and code scales independently. These settings are restored with this session.</p></header>
        <div className="dashboardSettingRows">
          <DashboardScaleControl label="Interface text" onChange={(uiFontSize) => controls.onPreferencesChange({ uiFontSize })} scales={scales} value={preferences.uiFontSize} />
          <DashboardScaleControl label="Code text" onChange={(codeFontSize) => controls.onPreferencesChange({ codeFontSize })} scales={scales} value={preferences.codeFontSize} />
        </div>
      </article>
      <article>
        <header><span>Workspace</span><strong>Layout and focus</strong><p>Bring back the compact, focus, preview, and developer controls that are intentionally removed from the top bar.</p></header>
        <div className="dashboardSettingRows">
          <DashboardToggle label="Density" onClick={() => controls.onDensityChange(layoutState.density === "cozy" ? "compact" : "cozy")} value={layoutState.density === "cozy" ? "Cozy" : "Compact"} />
          <DashboardToggle label="Focus" onClick={controls.onFocusModeToggle} value={controls.isFocusMode ? "On" : "Off"} />
          <DashboardToggle label="Sessions rail" onClick={controls.onSidebarToggle} value={layoutState.sidebarCollapsed ? "Collapsed" : "Open"} />
          <DashboardToggle label="Inspector" onClick={controls.onInspectorToggle} value={layoutState.inspectorCollapsed ? "Collapsed" : "Open"} />
          <DashboardToggle label="Preview shelf" onClick={controls.onPreviewToggle} value={controls.isPreviewOpen ? "Open" : "Closed"} />
          <DashboardToggle label="Display mode" onClick={() => controls.onPreferencesChange({ showDebugPanels: !preferences.showDebugPanels })} value={preferences.showDebugPanels ? "Developer" : "User"} />
          <DashboardToggle label="Preview behavior" onClick={() => controls.onPreferencesChange({ autoOpenPreview: !preferences.autoOpenPreview })} value={preferences.autoOpenPreview ? "Automatic" : "Manual"} />
        </div>
      </article>
    </section>
  );
}

function DashboardScaleControl({ label, onChange, scales, value }: { label: string; onChange: (value: WorkspacePreferences["uiFontSize"]) => void; scales: WorkspacePreferences["uiFontSize"][]; value: WorkspacePreferences["uiFontSize"]; }) {
  return <div className="dashboardScaleControl"><strong>{label}</strong><div>{scales.map((scale) => <button aria-pressed={value === scale} key={scale} onClick={() => onChange(scale)} type="button">{scale}</button>)}</div></div>;
}

function DashboardToggle({ label, onClick, value }: { label: string; onClick: () => void; value: string }) {
  return <button className="dashboardToggle" onClick={onClick} type="button"><span>{label}</span><strong>{value}</strong></button>;
}

function EvaluationHistory({ history }: { history: EvaluationHistoryRecord[] }) {
  if (history.length === 0) {
    return (
      <p className="evaluationHistoryEmpty">
        No explicit evaluation attempt has been stored for this session.
      </p>
    );
  }
  return (
    <section aria-label="Evaluation history" className="evaluationHistory">
      <header>
        <strong>Evaluation history</strong>
        <span>{history.length} retained</span>
      </header>
      <ul>
        {[...history].reverse().map((entry) => (
          <li key={entry.id}>
            <div>
              <strong>{entry.status.replace(/_/g, " ")}</strong>
              <span>{entry.datasetId ?? "Dataset not recorded"}</span>
              <small>{entry.metrics.length > 0 ? entry.metrics.join(", ") : entry.detail}</small>
            </div>
            <time dateTime={entry.evaluatedAt}>
              {formatHistoryTimestamp(entry.evaluatedAt)}
            </time>
          </li>
        ))}
      </ul>
    </section>
  );
}

function formatHistoryTimestamp(value: string) {
  const timestamp = new Date(value);
  return Number.isNaN(timestamp.getTime())
    ? value
    : timestamp.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function getDashboardGroup(category: ProductIntelligenceCategory) {
  return (
    dashboardGroups.find((group) => group.categories.includes(category)) ??
    dashboardGroups[0]
  );
}

export function ProductIntelligenceInspector({
  category,
  model
}: {
  category: ProductIntelligenceCategory;
  model: ProductIntelligenceModel;
}) {
  return (
    <section
      aria-label={`${category} inspector`}
      className="inspectorPanel productIntelligenceInspector"
      id={`${category.toLowerCase().replace(/\s+/g, "-")}-inspector-panel`}
      role="tabpanel"
    >
      <ProductIntelligenceHelp
        section={getProductIntelligenceSection(model, category)}
      />
      <ProductIntelligenceSectionView
        model={model}
        section={getProductIntelligenceSection(model, category)}
      />
    </section>
  );
}

export function ProductIntelligenceHelp({
  section
}: {
  section: ProductIntelligenceSection;
}) {
  return (
    <details className="productIntelligenceHelp">
      <summary aria-label={`Help with ${section.category}`}>
        <CircleHelp aria-hidden="true" size={15} />
      </summary>
      <div role="note">
        <strong>{section.category}</strong>
        <p>{section.detail}</p>
        <p>
          Review the metric cards for the current values, then use the
          Dashboard categories or Inspector tabs to change context.
        </p>
      </div>
    </details>
  );
}

function ProductIntelligenceSectionView({
  detailed = false,
  model,
  section
}: {
  detailed?: boolean;
  model: ProductIntelligenceModel;
  section: ProductIntelligenceSection;
}) {
  if (section.category === "Domains") {
    return (
      <DomainExperienceSurface
        activeDomainId={model.activeDomainId}
        catalog={model.domainExperience}
        detailed={detailed}
        includeKnowledgeBase={false}
      />
    );
  }

  if (section.category === "Knowledge Base") {
    return (
      <KnowledgeBaseInventorySurface
        detailed={detailed}
        inventory={model.domainExperience.knowledgeBase}
      />
    );
  }

  const notes = detailed ? section.notes : section.notes.slice(0, 2);
  const metrics = detailed ? section.metrics : section.metrics.slice(0, 3);

  return (
    <div className="productIntelligenceSection" data-tone={section.tone}>
      <article className="productIntelligenceHero">
        <span>{section.category}</span>
        <strong>{formatUiStatusLabel(section.summary)}</strong>
        <p>{section.detail}</p>
      </article>
      <dl className="productIntelligenceMetrics">
        {metrics.map((metric) => (
          <div key={metric.label}>
            <dt>{metric.label}</dt>
            <dd title={metric.value}>{metric.value}</dd>
          </div>
        ))}
      </dl>
      <div className="productIntelligenceNotes" aria-label={`${section.category} details`}>
        {notes.map((note) => (
          <p key={note}>{note}</p>
        ))}
      </div>
    </div>
  );
}
