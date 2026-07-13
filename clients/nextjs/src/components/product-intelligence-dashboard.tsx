"use client";

import { useEffect, useState } from "react";
import {
  Activity,
  CircleHelp,
  Eye,
  FileOutput,
  Radio,
  ShieldCheck,
  X
} from "lucide-react";
import {
  DomainExperienceSurface,
  KnowledgeBaseInventorySurface
} from "./domain-experience-surface";
import { RetrievalInspector } from "./retrieval-inspector";
import { RuntimeConsoleInspector } from "./runtime-console-inspector";
import { ConversationContextInspector } from "./conversation-context-inspector";
import { ProviderObservabilityDeepDive } from "./provider-observability-deep-dive";
import { EvaluationSessionDashboard } from "./evaluation-session-dashboard";
import { WorkstationDashboardSurface } from "./workstation-dashboard-surface";
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
import type { WorkflowExecutionMode } from "@/lib/workflow-execution";
import { formatWorkflowGraphRoute } from "@/lib/workflow-graph";
import type { SessionUsageSummary } from "@/lib/session-usage-ledger";
import { formatUiStatusLabel } from "@/lib/ui-copy";
import { morphogenesisPromptLibrary } from "@/lib/curated-prompt-library";
import type {
  EvaluationHistoryRecord,
  FeedbackSentiment
} from "@/lib/product-controls";
import { OutputFeedbackPanel } from "./output-feedback-panel";
import { UserGuide } from "./user-guide";
import { CapstoneEvaluationWorkspace } from "./capstone-evaluation-workspace";
import type { EvaluationRunRequest } from "@/lib/evaluation-benchmark";
import { buildMultiPreviewComparisonModel } from "@/lib/multi-preview-comparison";
import { PreviewRendererSurface } from "./preview-renderer-surface";

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
  onWorkflowModeChange: (mode: WorkflowExecutionMode) => void;
  preferences: WorkspacePreferences;
  workflowMode: WorkflowExecutionMode;
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
  onRunEvaluation?: (request: EvaluationRunRequest) => Promise<void>;
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
    detail: "How route selection maps to the published run topology.",
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
    detail: "Benchmark runs, defensible RAGAS evidence, and product validation.",
    categories: ["Validation", "Product Bugs", "LangSmith"]
  },
  {
    id: "manual",
    label: "User Guide",
    detail: "Canonical documentation for the complete product workflow.",
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

const dashboardGroupsWithDedicatedSurface = new Set<DashboardGroupId>([
  ...dashboardGroupsWithSignalBoard,
  "ai_agents",
  "artifacts",
  "telemetry"
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

  async function runEvaluation(request: EvaluationRunRequest) {
    if (!onRunEvaluation || evaluationRunning) {
      return;
    }
    setEvaluationRunning(true);
    try {
      await onRunEvaluation(request);
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
  onRunEvaluation?: (request: EvaluationRunRequest) => Promise<void>;
  evaluationHistory: EvaluationHistoryRecord[];
  feedback?: DashboardFeedback;
  settings?: DashboardSettingsControls;
  sessions?: DashboardSessionControls;
}) {
  if (group.id === "manual") {
    return <UserGuide />;
  }

  if (group.id === "settings" && settings) {
    return <DashboardSettings controls={settings} />;
  }

  if (group.id === "knowledge") {
    return <KnowledgeDashboardView model={model} />;
  }

  if (group.id === "evaluation" && onRunEvaluation) {
    return (
      <div className="productDashboardGroup" aria-label="Evaluation details">
        <EvaluationDashboardSurface
          evaluationHistory={evaluationHistory}
          evaluationRunning={evaluationRunning}
          model={model}
          onRunEvaluation={onRunEvaluation}
        />
        <details className="dashboardFeature evaluationEvidenceDisclosure">
          <summary>Supporting validation, Product Bug, and LangSmith signals</summary>
          <p>These current-workspace signals remain separate from benchmark results.</p>
          {group.categories.map((category) => {
            const section = getProductIntelligenceSection(model, category);
            return (
              <section className="productDashboardGroupSection" key={category}>
                <header>
                  <div><span>{category}</span><strong>{formatUiStatusLabel(section.summary)}</strong><p>{section.detail}</p></div>
                  <ProductIntelligenceHelp section={section} />
                </header>
                <ProductIntelligenceSectionView detailed model={model} section={section} />
              </section>
            );
          })}
        </details>
      </div>
    );
  }

  return (
    <div className="productDashboardGroup" aria-label={`${group.label} details`}>
      {dashboardGroupsWithSignalBoard.has(group.id) ? (
        <DashboardSignalBoard group={group} model={model} />
      ) : null}
      {!dashboardGroupsWithDedicatedSurface.has(group.id) &&
      !(group.id === "sessions" && sessions) ? group.categories.map((category) => {
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
            <ProductIntelligenceSectionView detailed model={model} section={section} />
          </section>
        );
      }) : null}
      {group.id === "architecture" ? <ArchitectureRouteGuide model={model} /> : null}
      {group.id === "workflow" ? <WorkflowLiveMap model={model} /> : null}
      {group.id === "workspace" ? <ActiveDocumentBoard model={model} /> : null}
      {group.id === "runtime" ? <RuntimeDashboardSurface model={model} /> : null}
      {group.id === "preview" ? <PreviewReadinessBoard model={model} /> : null}
      {group.id === "ai_agents" ? <AiAgentSystemMap model={model} /> : null}
      {group.id === "ai_agents" ? <ProviderDashboardSurface model={model} /> : null}
      {group.id === "memory" ? <MemoryDashboardSurface model={model} /> : null}
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
  const runtime = model.details?.workflowRuntime;
  const executionPublished = execution?.state === "available";
  const routes = [
    {
      title: "Single agent",
      detail: "A focused orchestration contract for contained requests.",
      selected: executionPublished && execution.resolvedMode === "single_agent",
      selectionLabel: "Resolved route"
    },
    {
      title: "Multi-agent",
      detail: "A specialist coordination contract used when the published plan requires additional roles.",
      selected: executionPublished && execution.resolvedMode === "multi_agent",
      selectionLabel: "Resolved route"
    },
    {
      title: "Auto",
      detail: "A routing policy that resolves to a published execution route; it is not a third hidden node sequence.",
      selected: executionPublished && execution.requestedMode === "auto",
      selectionLabel: "Requested policy"
    }
  ];

  return (
    <section aria-label="Workflow route guide" className="dashboardFeature architectureRouteGuide">
      <header>
        <div>
          <span>Route guide</span>
          <strong>Selection policy and executed topology stay separate</strong>
          <p>{executionPublished
            ? `This run resolved to ${formatRoute(execution.resolvedMode)} from ${formatRoute(execution.requestedMode)}.`
            : "Run a request to inspect its published route, roles, and workflow topology."}</p>
        </div>
        <DashboardPanelHelp
          detail="The three cards explain route-selection contracts only. The topology below comes from published workflow steps for the current run, so illustrative labels are never presented as executed nodes."
          label="Workflow route guide"
        />
      </header>
      <div className="architectureRouteGrid">
        {routes.map((route) => (
          <article data-selected={route.selected ? "true" : "false"} key={route.title}>
            <span>{route.selected ? route.selectionLabel : "Route contract"}</span>
            <strong>{route.title}</strong>
            <p>{route.detail}</p>
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
      <div className="architectureTopology" aria-label="Published workflow topology">
        <header>
          <span>Current run topology</span>
          <strong>{runtime?.steps.length
            ? `${runtime.steps.length} published workflow node${runtime.steps.length === 1 ? "" : "s"}`
            : "No workflow topology published"}</strong>
        </header>
        {runtime?.steps.length ? (
          <ol>
            {runtime.steps.map((step, index) => (
              <li data-state={step.state} key={step.nodeId}>
                <span>{index + 1}</span>
                <div>
                  <strong>{step.displayLabel}</strong>
                  <small>{step.lastEventDetail ?? step.detail}</small>
                </div>
                <em>{formatUiStatusLabel(step.state)}</em>
              </li>
            ))}
          </ol>
        ) : (
          <p>The Dashboard will draw the real nodes after the runtime publishes a plan. No illustrative sequence is substituted.</p>
        )}
      </div>
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
            <strong>{formatUiStatusLabel(knowledge.summary)}</strong>
            <p>{knowledge.detail}</p>
          </div>
          <ProductIntelligenceHelp section={knowledge} />
        </header>
        <KnowledgeBaseInventorySurface
          detailed
          headerMode="embedded"
          inventory={model.domainExperience.knowledgeBase}
        />
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
        {model.details ? (
          <RetrievalInspector
            inventory={model.domainExperience.knowledgeBase}
            runtime={model.details.retrievalRuntime}
          />
        ) : (
          <ProductIntelligenceSectionView detailed model={model} section={retrieval} />
        )}
      </section>
    </div>
  );
}

function CreativeKnowledgePanel({ model }: { model: ProductIntelligenceModel }) {
  const translation = model.artifactRegistry[0]?.creativeTranslation ?? null;
  const [selectedKind, setSelectedKind] = useState<string>("all");
  const inventory = model.domainExperience.creativeKnowledge;
  const recordKinds = [...new Set(inventory.records.map((record) => record.kind))];
  const visibleRecords = inventory.records.filter(
    (record) => selectedKind === "all" || record.kind === selectedKind
  );
  const artifactGroups = translation
    ? [
        { label: "Intent", values: [translation.creativeIntent] },
        { label: "Style", values: translation.visualStyle?.styles ?? [] },
        { label: "Atmosphere", values: [...translation.moodAtmosphere, ...translation.colorMaterialDirection] },
        { label: "Structure", values: [...translation.structureDirection, ...translation.geometricReferences] },
        { label: "Motion", values: translation.movementLanguage },
        { label: "Runtime", values: translation.runtimeRecommendations }
      ].filter((group) => group.values.length > 0)
    : [];
  const inventoryMetrics = [
    { label: "Records", value: inventory.recordCount },
    { label: "Techniques", value: inventory.records.filter((record) => record.kind === "technique").length },
    { label: "Workflows", value: inventory.records.filter((record) => record.kind === "workflow").length },
    { label: "Best practices", value: inventory.records.filter((record) => record.kind === "best_practice").length }
  ];

  return (
    <section aria-label="Creative Knowledge Base" className="dashboardFeature creativeKnowledgeBase">
      <header>
        <div>
          <span>Creative Knowledge Base</span>
          <strong>{inventory.status === "available" ? "Explicit techniques, workflows, patterns, and practical safeguards" : "Curated creative studies are available while the local inventory loads"}</strong>
          <p>Inspectable guidance only: source-backed patterns and browser boundaries, never hidden provider reasoning.</p>
        </div>
        <DashboardPanelHelp
          detail="Creative Knowledge Base exposes deterministic records from the existing creative-distillation architecture, plus the product's curated studies. It is separate from Technical Knowledge and contains no private prompt, memory, or model chain-of-thought."
          label="Creative Knowledge Base"
        />
      </header>
      <div aria-label="Creative Knowledge Base summary" className="creativeKnowledgeMetrics">
        {inventoryMetrics.map((metric) => (
          <div key={metric.label}>
            <span>{metric.label}</span>
            <strong>{metric.value}</strong>
          </div>
        ))}
        <div>
          <span>Boundary</span>
          <strong>{inventory.status === "available" ? "Inspectable" : "Loading"}</strong>
        </div>
      </div>
      {inventory.records.length ? (
        <>
          <div aria-label="Creative Knowledge filters" className="creativeKnowledgeFilters" role="group">
            <button aria-pressed={selectedKind === "all"} onClick={() => setSelectedKind("all")} type="button">All</button>
            {recordKinds.map((kind) => (
              <button aria-pressed={selectedKind === kind} key={kind} onClick={() => setSelectedKind(kind)} type="button">
                {formatCreativeKnowledgeKind(kind)}
              </button>
            ))}
          </div>
          <div aria-label="Creative knowledge records" className="creativeKnowledgeRecordGrid" role="list">
            {visibleRecords.map((record) => (
              <article data-kind={record.kind} key={record.id} role="listitem">
                <header>
                  <span>{formatCreativeKnowledgeKind(record.kind)}</span>
                  <strong>{record.title}</strong>
                </header>
                <p>{record.summary}</p>
                <div className="creativeKnowledgeDomainChips" aria-label={`${record.title} domains`}>
                  {record.domains.map((domain) => <span key={domain}>{formatCreativeDomain(model, domain)}</span>)}
                </div>
                <dl>
                  <div><dt>Confidence</dt><dd>{Math.round(record.confidence.score * 100)}% · {record.confidence.band}</dd></div>
                  <div><dt>Provenance</dt><dd>{record.provenanceCount} record{record.provenanceCount === 1 ? "" : "s"}</dd></div>
                </dl>
                {record.techniqueTags.length || record.patternTags.length ? (
                  <div className="creativeKnowledgeTagRow">
                    {[...record.techniqueTags, ...record.patternTags].slice(0, 5).map((tag) => <span key={tag}>{formatCreativeTag(tag)}</span>)}
                  </div>
                ) : null}
                {record.workflowSteps.length ? (
                  <ol aria-label={`${record.title} workflow`}>
                    {record.workflowSteps.slice(0, 3).map((step, index) => <li key={step}><span>{index + 1}</span>{step}</li>)}
                  </ol>
                ) : null}
              </article>
            ))}
          </div>
        </>
      ) : null}
      <section aria-label="Curated creative studies" className="creativeStudyShelf">
        <header>
          <div>
            <span>Curated creative studies</span>
            <strong>Browser-aware pattern starters</strong>
          </div>
          <DashboardPanelHelp
            detail="These are existing curated prompt-library studies. They are deterministic starter briefs and honest runtime boundaries, not a claim of a new generative or semantic system."
            label="Curated creative studies"
          />
        </header>
        <div role="list">
          {morphogenesisPromptLibrary.map((study) => (
            <article key={study.id} role="listitem">
              <div>
                <strong>{study.title}</strong>
                <span>{study.concept}</span>
              </div>
              <p>{study.description}</p>
              <dl>
                <div><dt>Runtime</dt><dd>{study.runtime}</dd></div>
                <div><dt>Boundary</dt><dd>{study.previewBoundary}</dd></div>
              </dl>
            </article>
          ))}
        </div>
      </section>
      {artifactGroups.length ? (
        <section aria-label="Selected artifact creative direction" className="artifactCreativeDirection">
          <header>
            <div><span>Selected artifact direction</span><strong>Published with the current artifact</strong></div>
            <DashboardPanelHelp detail="These fields are structured creative metadata retained with the selected artifact. They are useful for follow-up refinement but do not replace the persistent Creative Knowledge Base." label="Selected artifact direction" />
          </header>
          <ul>
            {artifactGroups.map((group) => <li key={group.label}><strong>{group.label}</strong><span>{group.values.slice(0, 4).join(" · ")}</span></li>)}
          </ul>
        </section>
      ) : null}
      <footer className="creativeKnowledgeBoundary">{inventory.authorityBoundary}</footer>
    </section>
  );
}

function formatCreativeKnowledgeKind(kind: string) {
  return kind.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatCreativeTag(tag: string) {
  return tag.replace(/_/g, " ");
}

function formatCreativeDomain(model: ProductIntelligenceModel, domain: string) {
  return model.domainExperience.domains.find((record) => record.id === domain)?.displayName ?? formatCreativeTag(domain);
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
        <div>
          <span>Route</span>
          <strong>
            {execution
              ? formatWorkflowGraphRoute({
                  execution,
                  requestedMode: execution.requestedMode
                })
              : "Not published"}
          </strong>
        </div>
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
  const primarySignals = telemetry.signals.filter((signal) =>
    ["workflow", "preview", "retrieval"].includes(signal.id)
  );
  const evidenceSignals = telemetry.signals.filter((signal) =>
    !["workflow", "preview", "retrieval"].includes(signal.id)
  );
  const outcome = telemetry.runtime.productOutcome;
  const lifecycle = [
    {
      detail: telemetry.stream.eventCount
        ? `${telemetry.stream.eventCount} published event${telemetry.stream.eventCount === 1 ? "" : "s"}; latest: ${telemetry.stream.latestEventLabel}.`
        : "No stream event has been published for this run.",
      label: "Stream",
      state: telemetry.stream.state,
      value: telemetry.stream.eventCount ? formatUiStatusLabel(telemetry.stream.state) : "Awaiting run"
    },
    {
      detail: `${telemetry.runtime.reachedNodes}/${telemetry.runtime.totalNodes} workflow nodes reached; ${telemetry.runtime.retryCount} retries reported.`,
      label: "Workflow",
      state: telemetry.runtime.activity.state,
      value: telemetry.runtime.activity.label
    },
    {
      detail: telemetry.preview.detail,
      label: "Preview",
      state: telemetry.preview.error ? "error" : telemetry.preview.state,
      value: telemetry.preview.healthLabel
    },
    {
      detail: telemetry.evaluation.state === "available"
        ? `${telemetry.evaluation.statusLabel}; ${telemetry.observability.state} observability lineage.`
        : `Evaluation evidence is unavailable; observability is ${telemetry.observability.state}.`,
      label: "Evidence",
      state: telemetry.evaluation.state,
      value: telemetry.evaluation.statusLabel
    }
  ];

  return (
    <section aria-label="Telemetry observatory" className="dashboardFeature telemetryObservatory" data-state={telemetry.status}>
      <header>
        <div>
          <span><Activity aria-hidden="true" size={14} /> Run observatory</span>
          <strong>One run, four evidence checkpoints</strong>
          <p>{telemetry.summary.coverageLabel}. Unpublished values remain unavailable.</p>
        </div>
        <DashboardPanelHelp
          detail="The observatory groups published stream, workflow, preview, retrieval, evaluation, and observability signals. Unreported provider usage and cost values stay visibly unavailable."
          label="Telemetry observatory"
        />
      </header>
      <div className="telemetryReviewerHero">
        <article className="telemetryOutcomeCard" data-state={telemetry.status}>
          <span>Current product outcome</span>
          <strong>{formatUiStatusLabel(outcome.product_outcome.toLowerCase().replace(/_/g, " "))}</strong>
          <p>{outcome.summary}</p>
          {outcome.recovery_action ? <small>{outcome.recovery_action}</small> : null}
        </article>
        <dl className="telemetryRunFacts" aria-label="Run measurement facts">
          <div><dt>Operator state</dt><dd>{telemetry.summary.operatorStatus}</dd></div>
          <div><dt>Runtime</dt><dd>{telemetry.summary.runtimeLabel}</dd></div>
          <div><dt>Events / errors</dt><dd>{telemetry.stream.eventCount} / {telemetry.stream.errorCount}</dd></div>
          <div><dt>Tokens</dt><dd>{telemetry.provider.summary.tokenLabel}</dd></div>
          <div><dt>Estimated cost</dt><dd>{telemetry.provider.summary.costLabel}</dd></div>
        </dl>
      </div>
      <ol aria-label="Run evidence checkpoints" className="telemetryLifecycleRail">
        {lifecycle.map((step, index) => (
          <li data-state={step.state} key={step.label}>
            <span>{index + 1}</span>
            <div><strong>{step.label}</strong><small>{step.detail}</small></div>
            <em>{step.value}</em>
          </li>
        ))}
      </ol>
      <div aria-label="Primary telemetry signals" className="telemetrySignalGrid">
        {primarySignals.map((signal) => (
          <article data-tone={signal.tone} key={signal.id}>
            <span>{signal.label}</span>
            <strong>{signal.value}</strong>
            <p>{signal.detail}</p>
          </article>
        ))}
      </div>
      {evidenceSignals.length ? (
        <details className="telemetryEvidenceDisclosure">
          <summary><Radio aria-hidden="true" size={14} /> Provider, observability, and evaluation evidence</summary>
          <div className="telemetrySignalGrid">
            {evidenceSignals.map((signal) => (
              <article data-tone={signal.tone} key={signal.id}>
                <span>{signal.label}</span>
                <strong>{signal.value}</strong>
                <p>{signal.detail}</p>
              </article>
            ))}
          </div>
        </details>
      ) : null}
      <footer className="telemetryTruthBoundary">
        <ShieldCheck aria-hidden="true" size={15} />
        <span><strong>Published evidence only.</strong> Telemetry describes this workspace run; it does not expose provider reasoning or turn missing usage into an estimate.</span>
      </footer>
    </section>
  );
}

function ActiveDocumentBoard({ model }: { model: ProductIntelligenceModel }) {
  const artifact = getDashboardActiveArtifact(model);
  if (!artifact) {
    return (
      <section aria-label="Active document" className="dashboardFeature dashboardEmptyState">
        <header>
          <div><span>Active document</span><strong>No generated source yet</strong><p>Generate an artifact to inspect its source, delivery boundary, and preview eligibility here.</p></div>
          <DashboardPanelHelp detail="This Dashboard view reports the current saved artifact. Source, preview capability, and delivery boundary remain separate product signals." label="Active document" />
        </header>
      </section>
    );
  }

  return (
    <section aria-label="Active document" className="dashboardFeature activeDocumentBoard">
      <header>
        <div>
          <span>Active document</span>
          <strong>{artifact.title}</strong>
          <p>{artifact.summary}</p>
        </div>
        <DashboardPanelHelp detail="The active document is the retained source artifact selected by the workspace. Preview status is reported separately because a saved artifact is not necessarily runnable in this browser." label="Active document" />
      </header>
      <dl>
        <div><dt>Language</dt><dd>{artifact.language}</dd></div>
        <div><dt>Artifact status</dt><dd>{artifact.status}</dd></div>
        <div><dt>Preview</dt><dd>{artifact.previewEligible ? "Eligible" : "Not eligible"}</dd></div>
        <div><dt>Domain</dt><dd>{artifact.domain ? formatCreativeDomain(model, artifact.domain) : "Not published"}</dd></div>
      </dl>
      <pre aria-label="Active document source excerpt"><code>{artifactSnippet(artifact.content)}</code></pre>
    </section>
  );
}

function getDashboardActiveArtifact(model: ProductIntelligenceModel) {
  return model.artifactRegistry.find((artifact) => artifact.id === model.activeArtifactId) ??
    model.artifactRegistry[0] ??
    null;
}

function PreviewReadinessBoard({ model }: { model: ProductIntelligenceModel }) {
  const preview = model.details?.telemetryDashboard.preview;
  if (!preview) return null;
  const stages = [
    { label: "Artifact", value: preview.artifactName ? "Ready" : "Pending", complete: Boolean(preview.artifactName) },
    { label: "Route", value: preview.renderer || "Pending", complete: Boolean(preview.renderer) },
    { label: "Mount", value: preview.available ? "Available" : preview.active ? "Starting" : "Waiting", complete: preview.available },
    { label: "Health", value: preview.healthLabel, complete: !preview.error && preview.available }
  ];
  return (
    <section aria-label="Preview readiness" className="dashboardFeature previewReadinessBoard" data-state={preview.state}>
      <header>
        <div><span>Preview readiness</span><strong>{preview.healthLabel}</strong><p>{preview.detail}</p></div>
        <DashboardPanelHelp detail="Preview readiness separates the selected artifact, renderer route, browser mount state, and runtime health. A saved artifact or renderer label alone does not claim that a visible preview is running." label="Preview readiness" />
      </header>
      <div className="previewReadinessMetrics">
        <div><span>State</span><strong>{formatUiStatusLabel(preview.state)}</strong></div>
        <div><span>Renderer</span><strong>{preview.renderer || "Pending"}</strong></div>
        <div><span>Target</span><strong>{preview.target}</strong></div>
        <div><span>Artifact</span><strong>{preview.artifactName || "No artifact"}</strong></div>
      </div>
      <ol aria-label="Preview lifecycle" className="previewLifecycle">
        {stages.map((stage, index) => <li data-complete={stage.complete ? "true" : "false"} key={stage.label}><span>{index + 1}</span><div><strong>{stage.label}</strong><small>{stage.value}</small></div></li>)}
      </ol>
      {preview.error ? <p className="previewReadinessError">{preview.error}</p> : null}
    </section>
  );
}

function RuntimeDashboardSurface({ model }: { model: ProductIntelligenceModel }) {
  const details = model.details;
  if (!details) return null;
  return (
    <section aria-label="Runtime health console" className="dashboardFeature dashboardComponentSurface">
      <header>
        <div><span>Runtime health</span><strong>Renderer, diagnostics, and recovery evidence</strong><p>Current runtime state is separate from the generated source and preview availability.</p></div>
        <DashboardPanelHelp detail="The runtime console presents published renderer health, metric signals, diagnostics, context, and events. It never treats a source artifact as proof that a runtime executed successfully." label="Runtime health console" />
      </header>
      <RuntimeConsoleInspector console={details.runtimeConsole} presentation="dashboard" productOutcome={details.workflowRuntime.summary.productOutcome} />
    </section>
  );
}

function ProviderDashboardSurface({ model }: { model: ProductIntelligenceModel }) {
  const telemetry = model.details?.providerTelemetry;
  if (!telemetry) return null;
  return (
    <section aria-label="Provider observability" className="dashboardFeature dashboardComponentSurface">
      <header>
        <div><span>Provider observability</span><strong>Usage, latency, configuration, and recovery path</strong><p>Only provider-published identifiers, usage fields, and configuration provenance are shown.</p></div>
        <DashboardPanelHelp detail="Provider observability distinguishes known values from unavailable values. It does not reveal prompts, private memory, or internal model reasoning." label="Provider observability" />
      </header>
      <ProviderObservabilityDeepDive telemetry={telemetry} />
    </section>
  );
}

function MemoryDashboardSurface({ model }: { model: ProductIntelligenceModel }) {
  const context = model.details?.conversationContext;
  if (!context) return null;
  return (
    <section aria-label="Memory and context" className="dashboardFeature dashboardComponentSurface">
      <header>
        <div><span>Context boundary</span><strong>What the current run explicitly published</strong><p>Counts make the session context inspectable without exposing conversation or private memory content.</p></div>
        <DashboardPanelHelp detail="This panel is privacy-safe: it exposes only counts and context-boundary facts published by the runtime. No private session text, embeddings, or provider prompt is shown." label="Memory and context" />
      </header>
      <ConversationContextInspector context={context} />
    </section>
  );
}

function EvaluationDashboardSurface({
  evaluationHistory,
  evaluationRunning,
  model,
  onRunEvaluation
}: {
  evaluationHistory: EvaluationHistoryRecord[];
  evaluationRunning: boolean;
  model: ProductIntelligenceModel;
  onRunEvaluation: (request: EvaluationRunRequest) => Promise<void>;
}) {
  const telemetry = model.details?.telemetryDashboard;
  const workstation = model.details?.workstationDashboard;
  if (!telemetry || !workstation) return null;
  return (
    <>
      <CapstoneEvaluationWorkspace
        history={evaluationHistory}
        model={model}
        onRun={onRunEvaluation}
        running={evaluationRunning}
      />
      <details className="dashboardFeature evaluationEvidenceDisclosure">
        <summary>Current trace and workstation evidence</summary>
        <p>Session evaluation lineage and workstation health are supporting evidence, not a replacement for benchmark results.</p>
        <div className="evaluationDashboardSplit">
          <EvaluationSessionDashboard evaluation={telemetry.evaluation} />
          <WorkstationDashboardSurface dashboard={workstation} />
        </div>
      </details>
    </>
  );
}

function formatRoute(route: "auto" | "single_agent" | "multi_agent" | null | undefined) {
  if (route === "single_agent") return "Single agent";
  if (route === "multi_agent") return "Multi-agent";
  if (route === "auto") return "Auto";
  return "Not published";
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

  const activeArtifact = getDashboardActiveArtifact(model);
  const comparison = model.details
    ? buildMultiPreviewComparisonModel({
        activeArtifactId: activeArtifact?.id ?? model.artifactRegistry[0]!.id,
        artifacts: model.artifactRegistry,
        code: model.details.snapshot.code,
        preview: model.details.snapshot.preview
      })
    : null;
  const previewCandidates = new Map(
    comparison?.candidates.map((candidate) => [candidate.artifact.id, candidate]) ?? []
  );

  return (
    <section aria-label="Artifact registry" className="dashboardFeature artifactRegistry">
      <header>
        <div>
          <span>Artifact registry</span>
          <strong>{model.artifactRegistry.length} saved deliverable{model.artifactRegistry.length === 1 ? "" : "s"}</strong>
          <p>Each deliverable exposes its visual output when a supported runtime can render it; code-only work states its export boundary instead.</p>
        </div>
        <DashboardPanelHelp
          detail="This registry lists artifacts retained by the active workspace snapshot. A live visual is mounted only through a supported preview contract. Otherwise the card explains the code or external-tool handoff without simulating output."
          label="Artifact registry"
        />
      </header>
      <div role="list">
        {model.artifactRegistry.map((artifact) => {
          const candidate = previewCandidates.get(artifact.id) ?? null;
          const previewLabel = candidate?.canRender
            ? "Live visual"
            : artifact.previewEligible
              ? "Preview evidence unavailable"
              : "Code / export only";
          return (
          <article aria-current={artifact.id === activeArtifact?.id ? "true" : undefined} key={artifact.id} role="listitem">
            <header>
              <div>
                <span>{artifact.type}{artifact.id === activeArtifact?.id ? " · selected" : ""}</span>
                <strong>{artifact.title}</strong>
                <p>{artifact.summary}</p>
              </div>
              <span data-status={artifact.status}>{artifact.status}</span>
            </header>
            <ArtifactVisualPreview artifact={artifact} candidate={candidate} />
            <dl>
              <div><dt>Session</dt><dd>{model.session.title}</dd></div>
              <div><dt>Language</dt><dd>{artifact.language}</dd></div>
              <div><dt>Delivery</dt><dd>{previewLabel}</dd></div>
            </dl>
            <details>
              <summary>View source excerpt</summary>
              <pre><code>{artifactSnippet(artifact.content)}</code></pre>
            </details>
          </article>
          );
        })}
      </div>
    </section>
  );
}

type DashboardPreviewCandidate = ReturnType<
  typeof buildMultiPreviewComparisonModel
>["candidates"][number];

function ArtifactVisualPreview({
  artifact,
  candidate
}: {
  artifact: ProductIntelligenceModel["artifactRegistry"][number];
  candidate: DashboardPreviewCandidate | null;
}) {
  if (candidate?.canRender) {
    return (
      <figure aria-label={`${artifact.title} visual preview`} className="artifactVisualPreview" data-kind="live">
        <div>
          <PreviewRendererSurface
            chrome="comparison"
            preview={candidate.preview}
            route={candidate.route}
            runtimeSessionKey={`dashboard:${candidate.runtimeSessionKey}`}
            runtimeSource={candidate.runtimeSource}
            showDiagnostics={false}
          />
        </div>
        <figcaption>
          <span><Eye aria-hidden="true" size={13} /> Live preview</span>
          <strong>{candidate.route.rendererLabel}</strong>
          <small>{candidate.route.targetLabel}</small>
        </figcaption>
      </figure>
    );
  }

  const codeOnly = artifact.type === "export" || !artifact.previewEligible;
  const boundaryLabel = codeOnly ? "Code / export boundary" : "Preview evidence unavailable";
  const boundaryTitle = candidate?.route.surfaceTitle ??
    (codeOnly ? "Use the retained source in its target runtime" : "No renderable preview was published");
  const boundaryDetail = candidate?.route.supportReason ??
    (codeOnly
      ? `${artifact.title} is retained as ${artifact.language} source. The Dashboard does not simulate an unsupported runtime.`
      : "This artifact is marked preview-eligible, but the current Dashboard model did not publish the route and source required to render it truthfully.");

  return (
    <figure aria-label={`${artifact.title} export boundary`} className="artifactVisualPreview" data-kind="boundary">
      <div className="artifactPreviewBoundary">
        <FileOutput aria-hidden="true" size={24} />
        <span>{boundaryLabel}</span>
        <strong>{boundaryTitle}</strong>
        <p>{boundaryDetail}</p>
      </div>
      <figcaption>
        <span><ShieldCheck aria-hidden="true" size={13} /> Truthful handoff</span>
        <strong>{candidate?.route.supportLabel ?? artifact.language}</strong>
        <small>{candidate?.route.targetLabel ?? "Source artifact"}</small>
      </figcaption>
    </figure>
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
  const themeGroups = [
    {
      id: "colour",
      label: "Colour themes",
      themes: [
        { value: "aqua", label: "Aqua" },
        { value: "codex", label: "Deep Blue" },
        { value: "matrix", label: "Matrix" },
        { value: "terminal", label: "Terminal" },
        { value: "horizon", label: "Horizon" },
        { value: "zen", label: "Zen" },
        { value: "blueprint", label: "Blueprint" }
      ]
    },
    {
      id: "monochrome",
      label: "Black & white",
      themes: [
        { value: "codex_white", label: "Dark" },
        { value: "light", label: "Light" }
      ]
    }
  ] satisfies readonly {
    id: string;
    label: string;
    themes: readonly { label: string; value: WorkspacePreferences["theme"] }[];
  }[];
  const scales: WorkspacePreferences["uiFontSize"][] = ["small", "medium", "large"];

  return (
    <section aria-label="Dashboard settings" className="dashboardSettings">
      <article>
        <header>
          <div><span>Appearance</span><strong>Theme and colour</strong><p>The selected theme controls the accessible text, surface, and accent colours for this saved session.</p></div>
          <DashboardPanelHelp detail="Theme changes the saved visual palette for this browser workspace. It does not change the generated artifact or workflow configuration." label="Appearance settings" />
        </header>
        <div className="dashboardThemeChoices" role="group" aria-label="Theme options">
          {themeGroups.map((group) => (
            <div className="dashboardThemeChoiceGroup" data-group={group.id} key={group.id}>
              <span>{group.label}</span>
              <div role="group" aria-label={group.label}>
                {group.themes.map((theme) => (
                  <button aria-pressed={preferences.theme === theme.value} data-theme={theme.value} key={theme.value} onClick={() => controls.onPreferencesChange({ theme: theme.value })} type="button">{theme.label}</button>
                ))}
              </div>
            </div>
          ))}
        </div>
      </article>
      <article>
        <header>
          <div><span>Typography</span><strong>Comfortable reading</strong><p>Set headings, body copy, labels and controls, and code independently. These settings are restored with this session.</p></div>
          <DashboardPanelHelp detail="Heading, body, label, control, and code scales are saved per local workspace session, so the reading preference returns when that session is reopened." label="Typography settings" />
        </header>
        <div className="dashboardSettingRows">
          <DashboardScaleControl label="Headings" onChange={(headingFontSize) => controls.onPreferencesChange({ headingFontSize })} preview="Heading" previewKind="heading" scales={scales} value={preferences.headingFontSize} />
          <DashboardScaleControl label="Body text" onChange={(uiFontSize) => controls.onPreferencesChange({ uiFontSize })} preview="Body text" previewKind="body" scales={scales} value={preferences.uiFontSize} />
          <DashboardScaleControl label="Labels and controls" onChange={(labelFontSize) => controls.onPreferencesChange({ labelFontSize })} preview="Label" previewKind="label" scales={scales} value={preferences.labelFontSize} />
          <DashboardScaleControl label="Code text" onChange={(codeFontSize) => controls.onPreferencesChange({ codeFontSize })} preview="Code" previewKind="code" scales={scales} value={preferences.codeFontSize} />
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
      <article>
        <header>
          <div><span>Generation</span><strong>Prompt defaults</strong><p>Set the workflow route and creative profile used for new prompts. Provider routing remains securely configured for this workspace.</p></div>
          <DashboardPanelHelp detail="These settings mirror the composer controls. Workflow and creativity apply to the next prompt, while the selected provider is shown for clarity and remains server-configured." label="Generation settings" />
        </header>
        <div className="dashboardGenerationRows" role="group" aria-label="Generation defaults">
          <label className="dashboardGenerationControl">
            <span>Workflow</span>
            <select
              aria-label="Default workflow"
              onChange={(event) => controls.onWorkflowModeChange(event.currentTarget.value as WorkflowExecutionMode)}
              value={controls.workflowMode}
            >
              <option value="auto">Auto</option>
              <option value="single_agent">Single Agent</option>
              <option value="multi_agent">Multi Agent</option>
            </select>
            <small>Choose the bounded route for each new request.</small>
          </label>
          <div className="dashboardGenerationControl" role="group" aria-label="AI Providers">
            <span>AI Providers</span>
            <details className="dashboardProviderDisclosure">
              <summary aria-label="Selected AI provider: OpenAI">OpenAI</summary>
              <div className="dashboardProviderAvailability">
                <span>Selected provider</span>
                <strong>OpenAI</strong>
                <small>Configured server-side.</small>
              </div>
            </details>
            <small>Review the configured provider for new prompts.</small>
          </div>
          <label className="dashboardGenerationControl">
            <span>Creativity</span>
            <select
              aria-label="Default creativity"
              onChange={(event) =>
                controls.onPreferencesChange({
                  creativity: event.currentTarget.value as WorkspacePreferences["creativity"]
                })
              }
              value={preferences.creativity}
            >
              <option value="controlled">Controlled</option>
              <option value="balanced">Balanced</option>
              <option value="exploratory">Exploratory</option>
            </select>
            <small>Balance exploration with a reliable implementation path.</small>
          </label>
        </div>
      </article>
    </section>
  );
}

function DashboardScaleControl({ label, onChange, preview, previewKind, scales, value }: { label: string; onChange: (value: WorkspacePreferences["uiFontSize"]) => void; preview: string; previewKind: "heading" | "body" | "label" | "code"; scales: WorkspacePreferences["uiFontSize"][]; value: WorkspacePreferences["uiFontSize"]; }) {
  return <div className="dashboardScaleControl"><strong>{label}</strong><div>{scales.map((scale) => <button aria-pressed={value === scale} key={scale} onClick={() => onChange(scale)} type="button">{scale}</button>)}</div><span className="dashboardTypographyPreview" data-kind={previewKind}>{preview}</span></div>;
}

function DashboardToggle({ label, onClick, value }: { label: string; onClick: () => void; value: string }) {
  return <button className="dashboardToggle" onClick={onClick} type="button"><span>{label}</span><strong>{value}</strong></button>;
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
  const [isOpen, setIsOpen] = useState(false);

  return (
    <details
      className="productIntelligenceHelp"
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
          setIsOpen(false);
        }
      }}
      onFocus={() => setIsOpen(true)}
      onMouseEnter={() => setIsOpen(true)}
      onMouseLeave={() => setIsOpen(false)}
      open={isOpen}
    >
      <summary
        aria-label={`Help with ${label}`}
        onClick={(event) => {
          event.preventDefault();
          setIsOpen((open) => !open);
        }}
      >
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
