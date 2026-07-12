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

const dashboardGroupsWithSignalBoard = new Set<DashboardGroupId>([
  "overview",
  "architecture",
  "workflow",
  "workspace",
  "runtime",
  "preview",
  "memory"
]);

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
          <header>
            <div>
              <span>Start a run</span>
              <strong>Describe a visual system, then choose a workflow route.</strong>
              <p>Use the workspace for the conversation; open Preview, Code, or Saved only when they add context.</p>
            </div>
            <DashboardPanelHelp detail="A first prompt establishes the creative brief. Workflow selection changes orchestration, not the proven runtime boundaries of the workspace." label="Start a run" />
          </header>
        </article>
        <article>
          <header>
            <div>
              <span>Read the result</span>
              <strong>Check the artifact, visible output, and runtime health separately.</strong>
              <p>Advanced Dashboard keeps diagnostics, source evidence, and workflow detail together without crowding the creative session.</p>
            </div>
            <DashboardPanelHelp detail="An artifact, its browser preview, and its runtime health are related but distinct signals. Review each before deciding whether to refine." label="Read the result" />
          </header>
        </article>
        <article>
          <header>
            <div>
              <span>Read knowledge in order</span>
              <strong>Technical Knowledge → Creative Knowledge Base → Retrieval.</strong>
              <p>Official sources establish the local index; published artifact guidance records creative direction; Retrieval reports the current run’s selected evidence and boundaries.</p>
            </div>
            <DashboardPanelHelp detail="Technical Knowledge is the official-source inventory. Creative Knowledge Base is explicit, published creative guidance attached to an artifact. Retrieval is the current-run evidence path. None of these surfaces expose private provider reasoning." label="Knowledge flow" />
          </header>
        </article>
      </section>
    );
  }

  if (group.id === "settings" && settings) {
    return <DashboardSettings controls={settings} />;
  }

  if (group.id === "knowledge") {
    return <KnowledgeDashboardView model={model} />;
  }

  return (
    <div className="productDashboardGroup" aria-label={`${group.label} details`}>
      {dashboardGroupsWithSignalBoard.has(group.id) ? (
        <DashboardSignalBoard group={group} model={model} />
      ) : null}
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
      {group.id === "architecture" ? <ArchitectureRouteGuide model={model} /> : null}
      {group.id === "workflow" ? <WorkflowLiveMap model={model} /> : null}
      {group.id === "ai_agents" ? <AiAgentSystemMap model={model} /> : null}
      {group.id === "artifacts" ? <ArtifactRegistry model={model} /> : null}
      {group.id === "sessions" && sessions ? <SessionRegistry controls={sessions} /> : null}
      {group.id === "telemetry" ? <TelemetryObservatory model={model} /> : null}
      {group.id === "telemetry" && sessions ? <UserUsageOverview usage={sessions.usage} /> : null}
      {group.id === "ai_agents" && feedback ? (
        <section className="productDashboardGroupSection productDashboardFeedback">
          <header>
            <div>
              <span>Feedback</span>
              <strong>Shape future creative requests</strong>
              <p>Explicit feedback stays in this local workspace profile.</p>
            </div>
            <DashboardPanelHelp detail="Feedback records an explicit local response to the current artifact. It is not silently sent to an external provider." label="Output feedback" />
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

function DashboardSignalBoard({
  group,
  model
}: {
  group: DashboardGroup;
  model: ProductIntelligenceModel;
}) {
  const sections = group.categories.map((category) => getProductIntelligenceSection(model, category));
  const primary = sections[0]!;
  const metrics = sections.flatMap((section) => section.metrics.map((metric) => ({
    ...metric,
    category: section.category
  })));
  const evidence = sections.flatMap((section) => section.notes.map((note) => ({
    category: section.category,
    note
  }))).slice(0, 3);
  const signalMetrics = [
    ...metrics,
    { category: "Live board", label: "Published metrics", value: String(metrics.length) },
    { category: "Live board", label: "Evidence cues", value: String(evidence.length) }
  ].slice(0, 6);

  return (
    <section
      aria-label={`${group.label} live signal board`}
      className="dashboardFeature dashboardSignalBoard"
      data-tone={primary.tone}
    >
      <header>
        <div>
          <span>Live signal board</span>
          <strong>{formatUiStatusLabel(primary.summary)}</strong>
          <p>Current workspace evidence for {group.label.toLowerCase()}.</p>
        </div>
        <DashboardPanelHelp
          detail={`This board visualizes published ${group.label.toLowerCase()} state from the shared workspace model. It never infers private provider reasoning or unpublished runtime data.`}
          label={`${group.label} live signal board`}
        />
      </header>
      <div className="dashboardSignalBoardGrid">
        <article className="dashboardSignalLead" data-tone={primary.tone}>
          <span>Current state</span>
          <strong>{formatUiStatusLabel(primary.summary)}</strong>
          <p>{primary.detail}</p>
          <small>{primary.category} · {formatUiStatusLabel(primary.tone)}</small>
        </article>
        <dl aria-label={`${group.label} signal metrics`} className="dashboardSignalMetrics">
          {signalMetrics.map((metric) => (
            <div key={`${metric.category}-${metric.label}`}>
              <dt>{metric.label}</dt>
              <dd title={metric.value}>{metric.value}</dd>
              <small>{metric.category}</small>
            </div>
          ))}
        </dl>
      </div>
      {evidence.length > 0 ? (
        <ul aria-label={`${group.label} evidence cues`} className="dashboardEvidenceCues">
          {evidence.map((item) => (
            <li key={`${item.category}-${item.note}`}>
              <span>{item.category}</span>
              <p>{item.note}</p>
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}

function ArchitectureRouteGuide({ model }: { model: ProductIntelligenceModel }) {
  const execution = model.details?.workflowExecution;
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
          <p>{execution?.state === "available"
            ? `This run published the ${formatRoute(execution.resolvedMode)} route.`
            : "The live Architecture card above reports the route used for the current run."}</p>
        </div>
        <DashboardPanelHelp
          detail="These route diagrams explain the product’s published execution shapes. The active route is reported from the current run, not guessed from the request."
          label="Workflow route guide"
        />
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
      <div className="architectureExecutionFacts" aria-label="Published execution decision">
        <div>
          <span>Requested route</span>
          <strong>{formatRoute(execution?.requestedMode)}</strong>
        </div>
        <div>
          <span>Resolved route</span>
          <strong>{execution?.resolvedMode ? formatRoute(execution.resolvedMode) : "Not published"}</strong>
        </div>
        <div>
          <span>Research</span>
          <strong>{execution?.researcherRequired == null ? "Not published" : execution.researcherRequired ? "Required" : "Skipped"}</strong>
        </div>
        <div>
          <span>Refinement limit</span>
          <strong>{execution?.maxRefinementLoops == null ? "Not published" : execution.maxRefinementLoops}</strong>
        </div>
      </div>
      {execution?.agentRoles.length ? (
        <div className="architectureRoleRail" aria-label="Published agent responsibilities">
          <span>Published responsibilities</span>
          {execution.agentRoles.map((role) => <strong key={role}>{role}</strong>)}
        </div>
      ) : null}
    </section>
  );
}

function KnowledgeDashboardView({ model }: { model: ProductIntelligenceModel }) {
  const knowledge = getProductIntelligenceSection(model, "Knowledge Base");
  const retrieval = getProductIntelligenceSection(model, "Retrieval");

  return (
    <div className="productDashboardGroup knowledgeDashboardView" aria-label="Knowledge details">
      <section className="productDashboardGroupSection">
        <header>
          <div>
            <span>Technical knowledge</span>
            <strong>Official source inventory and local index</strong>
            <p>Registered documentation, freshness checks, update controls, and local search coverage.</p>
          </div>
          <ProductIntelligenceHelp section={knowledge} />
        </header>
        <ProductIntelligenceSectionView detailed model={model} section={knowledge} />
      </section>
      <CreativeKnowledgePanel model={model} />
      <section className="productDashboardGroupSection">
        <header>
          <div>
            <span>Retrieval</span>
            <strong>{formatUiStatusLabel(retrieval.summary)}</strong>
            <p>Published request, source, chunk, quality, and freshness signals for this run.</p>
          </div>
          <ProductIntelligenceHelp section={retrieval} />
        </header>
        <ProductIntelligenceSectionView detailed model={model} section={retrieval} />
      </section>
    </div>
  );
}

function CreativeKnowledgePanel({ model }: { model: ProductIntelligenceModel }) {
  const translation = model.artifactRegistry[0]?.creativeTranslation ?? null;
  const groups = translation
    ? [
        { label: "Intent", values: [translation.creativeIntent] },
        { label: "Style", values: translation.visualStyle?.styles ?? [] },
        { label: "Atmosphere", values: [...translation.moodAtmosphere, ...translation.colorMaterialDirection] },
        { label: "Structure", values: [...translation.structureDirection, ...translation.geometricReferences] },
        { label: "Motion", values: translation.movementLanguage },
        { label: "Runtime", values: translation.runtimeRecommendations }
      ].filter((group) => group.values.length > 0)
    : [];

  return (
    <section aria-label="Creative Knowledge Base" className="dashboardFeature knowledgePrinciples">
      <header>
        <div>
          <span>Creative Knowledge Base</span>
          <strong>{translation ? "Published direction for the selected artifact" : "Creative direction is recorded with each generated artifact"}</strong>
          <p>{translation
            ? "These are the structured, user-facing creative directions retained with the artifact."
            : "Generate an artifact to inspect its published intent, aesthetic direction, structure, motion, and runtime recommendations."}</p>
        </div>
        <DashboardPanelHelp
          detail="Creative knowledge is artifact metadata that is explicitly published for inspection. It is separate from the official technical-source inventory and does not expose private model reasoning."
          label="Creative Knowledge Base"
        />
      </header>
      {groups.length ? (
        <ul>
          {groups.map((group) => (
            <li key={group.label}>
              <strong>{group.label}</strong>
              <span>{group.values.slice(0, 4).join(" · ")}</span>
            </li>
          ))}
        </ul>
      ) : (
        <div className="knowledgeEmptyState">
          <span>Awaiting published artifact metadata</span>
          <p>Legacy or not-yet-generated artifacts do not provide structured creative guidance.</p>
        </div>
      )}
    </section>
  );
}

function WorkflowLiveMap({ model }: { model: ProductIntelligenceModel }) {
  const runtime = model.details?.workflowRuntime;
  const execution = model.details?.workflowExecution;
  if (!runtime) {
    return null;
  }

  const visibleSteps = runtime.steps.filter((step) => step.state !== "queued");
  const steps = (visibleSteps.length ? visibleSteps : runtime.steps).slice(0, 12);
  const transitions = runtime.transitions.slice(-6);

  return (
    <section aria-label="Live workflow map" className="dashboardFeature workflowLiveMap" data-state={runtime.summary.activity.state}>
      <header>
        <div>
          <span>Execution path</span>
          <strong>{formatUiStatusLabel(runtime.summary.activity.label)}</strong>
          <p>{runtime.summary.activity.detail}</p>
        </div>
        <DashboardPanelHelp
          detail="This map follows workflow events published by the current run. Nodes and timings appear only after the runtime emitted them; queued nodes are not shown as completed work."
          label="Live workflow map"
        />
      </header>
      <div className="workflowLiveSummary">
        <div><span>Route</span><strong>{formatRoute(execution?.resolvedMode ?? execution?.requestedMode)}</strong></div>
        <div><span>Reached</span><strong>{runtime.summary.reached}/{runtime.summary.total}</strong></div>
        <div><span>Runtime</span><strong>{formatDashboardDuration(runtime.summary.totalRuntimeMs)}</strong></div>
        <div><span>Retries</span><strong>{runtime.summary.retryCount}</strong></div>
      </div>
      <ol className="workflowLivePath" aria-label="Published workflow nodes">
        {steps.map((step) => (
          <li data-state={step.state} key={step.nodeId}>
            <span aria-hidden="true" />
            <div>
              <strong>{step.displayLabel}</strong>
              <p>{step.lastEventDetail ?? step.detail}</p>
            </div>
            <small>{step.durationMs == null ? formatUiStatusLabel(step.state) : formatDashboardDuration(step.durationMs)}</small>
          </li>
        ))}
      </ol>
      {transitions.length ? (
        <ul aria-label="Recent workflow transitions" className="workflowTransitionRail">
          {transitions.map((transition) => (
            <li key={`${transition.sequence}-${transition.fromNodeId}-${transition.toNodeId}`} data-kind={transition.kind}>
              <span>{transition.label}</span>
              <strong>{transition.fromNodeId} → {transition.toNodeId}</strong>
              {transition.reason ? <small>{transition.reason}</small> : null}
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}

function AiAgentSystemMap({ model }: { model: ProductIntelligenceModel }) {
  const details = model.details;
  if (!details) {
    return null;
  }
  const { conversationContext, providerTelemetry, workflowExecution } = details;
  const roles = workflowExecution.agentRoles;

  return (
    <section aria-label="AI and agent system map" className="dashboardFeature aiAgentSystemMap">
      <header>
        <div>
          <span>AI system map</span>
          <strong>{providerTelemetry.summary.providerLabel} · {providerTelemetry.summary.modelLabel}</strong>
          <p>Provider, execution route, and published context counts remain separate so each boundary is easy to inspect.</p>
        </div>
        <DashboardPanelHelp
          detail="This panel reports the provider identity and telemetry emitted for the run, the selected execution route, and context counts. It deliberately does not display prompts, private memory, or model chain-of-thought."
          label="AI system map"
        />
      </header>
      <div className="aiAgentRoute" aria-label="Execution and provider route">
        <article><span>Provider</span><strong>{providerTelemetry.summary.providerLabel}</strong><p>{providerTelemetry.summary.generationModeLabel}</p></article>
        <article><span>Model</span><strong>{providerTelemetry.summary.modelLabel}</strong><p>{providerTelemetry.summary.streamingStatusLabel}</p></article>
        <article><span>Route</span><strong>{formatRoute(workflowExecution.resolvedMode ?? workflowExecution.requestedMode)}</strong><p>{workflowExecution.rationale}</p></article>
      </div>
      <div className="aiAgentSplit">
        <section>
          <header><span>Responsibilities</span><strong>{roles.length ? `${roles.length} published role${roles.length === 1 ? "" : "s"}` : "No role list published"}</strong></header>
          {roles.length ? <div className="aiAgentRoleChips">{roles.map((role) => <span key={role}>{role}</span>)}</div> : <p>Run a request to inspect the responsibilities selected for that route.</p>}
        </section>
        <section>
          <header><span>Context boundary</span><strong>{conversationContext.source === "stream" ? "Published context counts" : "No context counts published"}</strong></header>
          <dl>
            {conversationContext.diagnostics.slice(0, 4).map((diagnostic) => <div key={diagnostic.id}><dt>{diagnostic.label}</dt><dd>{diagnostic.value}</dd></div>)}
          </dl>
        </section>
      </div>
    </section>
  );
}

function TelemetryObservatory({ model }: { model: ProductIntelligenceModel }) {
  const telemetry = model.details?.telemetryDashboard;
  if (!telemetry) {
    return null;
  }
  const signals = telemetry.signals.slice(0, 6);

  return (
    <section aria-label="Telemetry observatory" className="dashboardFeature telemetryObservatory" data-state={telemetry.status}>
      <header>
        <div>
          <span>Run observatory</span>
          <strong>{telemetry.summary.operatorStatus}</strong>
          <p>{telemetry.summary.signalLabel}</p>
        </div>
        <DashboardPanelHelp
          detail="The observatory groups published stream, workflow, preview, retrieval, evaluation, and observability signals. Unreported provider usage and cost values stay visibly unavailable."
          label="Telemetry observatory"
        />
      </header>
      <div className="telemetrySignalGrid">
        {signals.map((signal) => (
          <article data-tone={signal.tone} key={signal.id}>
            <span>{signal.label}</span>
            <strong>{signal.value}</strong>
            <p>{signal.detail}</p>
          </article>
        ))}
      </div>
      <div className="telemetryRunFacts" aria-label="Run measurement facts">
        <div><span>Events</span><strong>{telemetry.stream.eventCount}</strong></div>
        <div><span>Errors</span><strong>{telemetry.stream.errorCount}</strong></div>
        <div><span>Tokens</span><strong>{telemetry.provider.summary.tokenLabel}</strong></div>
        <div><span>Est. cost</span><strong>{telemetry.provider.summary.costLabel}</strong></div>
        <div><span>Runtime</span><strong>{telemetry.summary.runtimeLabel}</strong></div>
      </div>
    </section>
  );
}

function formatRoute(route: "auto" | "single_agent" | "multi_agent" | null | undefined) {
  if (route === "single_agent") return "Single agent";
  if (route === "multi_agent") return "Multi-agent";
  return "Auto";
}

function formatDashboardDuration(value: number | null) {
  if (value == null) return "Timing pending";
  if (value < 1_000) return `${value} ms`;
  return `${(value / 1_000).toFixed(value >= 10_000 ? 0 : 1)} s`;
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
        <DashboardPanelHelp
          detail="This registry lists artifacts retained by the active workspace snapshot. Source excerpts are bounded for readability and do not claim that an external handoff runs in this browser."
          label="Artifact registry"
        />
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
        <div className="dashboardFeatureActions">
          <DashboardPanelHelp
            detail="Sessions are isolated to this browser profile. Tokens and cost are shown only when the provider published them for the retained runs."
            label="Session registry"
          />
          <button onClick={controls.onCreate} type="button">New session</button>
        </div>
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
        <DashboardPanelHelp
          detail="These totals aggregate only local-session records with provider-published usage fields. Unknown token or cost values are deliberately kept out of the totals."
          label="Browser profile usage"
        />
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
        <header>
          <div><span>Appearance</span><strong>Theme and colour</strong><p>The selected theme controls the accessible text, surface, and accent colours for this saved session.</p></div>
          <DashboardPanelHelp detail="Theme changes the saved visual palette for this browser workspace. It does not change the generated artifact or workflow configuration." label="Appearance settings" />
        </header>
        <div className="dashboardThemeChoices" role="group" aria-label="Theme options">
          {themes.map((theme) => (
            <button aria-pressed={preferences.theme === theme} data-theme={theme} key={theme} onClick={() => controls.onPreferencesChange({ theme })} type="button">{theme}</button>
          ))}
        </div>
      </article>
      <article>
        <header>
          <div><span>Typography</span><strong>Comfortable reading</strong><p>Adjust the application and code scales independently. These settings are restored with this session.</p></div>
          <DashboardPanelHelp detail="Interface and code sizes are saved per local workspace session, so the reading preference returns when that session is reopened." label="Typography settings" />
        </header>
        <div className="dashboardSettingRows">
          <DashboardScaleControl label="Interface text" onChange={(uiFontSize) => controls.onPreferencesChange({ uiFontSize })} scales={scales} value={preferences.uiFontSize} />
          <DashboardScaleControl label="Code text" onChange={(codeFontSize) => controls.onPreferencesChange({ codeFontSize })} scales={scales} value={preferences.codeFontSize} />
        </div>
      </article>
      <article>
        <header>
          <div><span>Workspace</span><strong>Layout and focus</strong><p>Bring back the compact, focus, preview, and developer controls that are intentionally removed from the top bar.</p></div>
          <DashboardPanelHelp detail="These controls restore the workspace, preview, inspector, density, and display choices that are saved for this local session." label="Workspace settings" />
        </header>
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
    <DashboardPanelHelp
      detail={section.detail}
      guidance="Review the metric cards for the current values, then use the Dashboard categories or Inspector tabs to change context."
      label={section.category}
    />
  );
}

function DashboardPanelHelp({
  detail,
  guidance,
  label
}: {
  detail: string;
  guidance?: string;
  label: string;
}) {
  return (
    <details className="productIntelligenceHelp">
      <summary aria-label={`Help with ${label}`}>
        <CircleHelp aria-hidden="true" size={15} />
      </summary>
      <div role="note">
        <strong>{label}</strong>
        <p>{detail}</p>
        {guidance ? <p>{guidance}</p> : null}
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
