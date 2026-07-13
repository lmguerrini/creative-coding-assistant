"use client";

import { useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BookOpen,
  Bot,
  Boxes,
  Brain,
  CircleHelp,
  Database,
  Eye,
  FileCode2,
  FileOutput,
  Gauge,
  GitBranch,
  History,
  LayoutDashboard,
  MonitorPlay,
  Network,
  Palette,
  PanelLeft,
  Radio,
  Settings2,
  ShieldCheck,
  Shapes,
  SlidersHorizontal,
  Type,
  X,
  type LucideIcon
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
import {
  DashboardPage,
  DashboardCallout,
  DashboardCardGrid,
  DashboardDisclosure,
  DashboardInfoCard,
  DashboardMetricGrid,
  DashboardSection,
  DashboardTableFrame
} from "./dashboard-page-primitives";

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
  icon: LucideIcon;
  navSection: "workspace" | "knowledge" | "review";
};

const dashboardNavigationSections = [
  { id: "workspace", label: "Workspace flow" },
  { id: "knowledge", label: "Knowledge & systems" },
  { id: "review", label: "Review & configure" }
] as const;

const dashboardGroups: DashboardGroup[] = [
  {
    id: "overview",
    label: "Overview",
    detail: "Current workspace outcome and selected artifact.",
    categories: ["Overview"],
    icon: LayoutDashboard,
    navSection: "workspace"
  },
  {
    id: "architecture",
    label: "Architecture",
    detail: "How route selection maps to the published run topology.",
    categories: ["Architecture"],
    icon: Network,
    navSection: "workspace"
  },
  {
    id: "workflow",
    label: "Workflow",
    detail: "Live progress, transitions, and recovery state.",
    categories: ["Workflow"],
    icon: GitBranch,
    navSection: "workspace"
  },
  {
    id: "workspace",
    label: "Workspace",
    detail: "Generated source and the active creative document.",
    categories: ["Code"],
    icon: FileCode2,
    navSection: "workspace"
  },
  {
    id: "runtime",
    label: "Runtime",
    detail: "Renderer health, diagnostics, and recovery state.",
    categories: ["Runtime"],
    icon: Activity,
    navSection: "workspace"
  },
  {
    id: "preview",
    label: "Preview",
    detail: "Browser output, renderer route, and ready state.",
    categories: ["Preview"],
    icon: MonitorPlay,
    navSection: "workspace"
  },
  {
    id: "artifacts",
    label: "Artifacts",
    detail: "Every generated deliverable, source excerpt, and session.",
    categories: ["Artifacts"],
    icon: Boxes,
    navSection: "workspace"
  },
  {
    id: "domains",
    label: "Domains",
    detail: "What can run live, export source, or hand off externally.",
    categories: ["Domains"],
    icon: Shapes,
    navSection: "knowledge"
  },
  {
    id: "knowledge",
    label: "Knowledge Base",
    detail: "Official sources, index coverage, and freshness controls.",
    categories: ["Knowledge Base", "Retrieval"],
    icon: Database,
    navSection: "knowledge"
  },
  {
    id: "ai_agents",
    label: "AI & agents",
    detail: "Provider route, model context, and agent responsibilities.",
    categories: ["Agents", "Providers"],
    icon: Bot,
    navSection: "knowledge"
  },
  {
    id: "memory",
    label: "Memory",
    detail: "Published context counts and privacy-safe session history.",
    categories: ["Memory"],
    icon: Brain,
    navSection: "knowledge"
  },
  {
    id: "sessions",
    label: "Sessions",
    detail: "Manage saved creative sessions and their artifacts.",
    categories: ["Sessions"],
    icon: History,
    navSection: "review"
  },
  {
    id: "telemetry",
    label: "Telemetry",
    detail: "Usage, token accounting, cost estimates, and runtime signals.",
    categories: ["Telemetry", "Metrics"],
    icon: Radio,
    navSection: "review"
  },
  {
    id: "evaluation",
    label: "Evaluation",
    detail: "Benchmark runs, defensible RAGAS evidence, and product validation.",
    categories: ["Validation", "Product Bugs", "LangSmith"],
    icon: Gauge,
    navSection: "review"
  },
  {
    id: "manual",
    label: "User Guide",
    detail: "Canonical documentation for the complete product workflow.",
    categories: [],
    icon: BookOpen,
    navSection: "review"
  },
  {
    id: "settings",
    label: "Settings",
    detail: "Workspace display, density, focus, and provider configuration.",
    categories: ["Settings"],
    icon: Settings2,
    navSection: "review"
  }
];

const dashboardPageCopy: Record<DashboardGroupId, {
  detail: string;
  eyebrow: string;
  title: string;
}> = {
  overview: {
    eyebrow: "Workspace pulse",
    title: "Read the current workspace in one glance",
    detail: "Outcome, selected artifact, and the strongest published signals for the active creative session."
  },
  architecture: {
    eyebrow: "Execution architecture",
    title: "Trace route decisions to the real run topology",
    detail: "Separate requested policy, resolved route, agent responsibilities, and published workflow nodes."
  },
  workflow: {
    eyebrow: "Live orchestration",
    title: "Follow every published execution step",
    detail: "See progress, transitions, timing, retries, and recovery without inferring unpublished runtime state."
  },
  workspace: {
    eyebrow: "Creative document",
    title: "Inspect the active source and its delivery boundary",
    detail: "Keep the retained artifact, its metadata, preview eligibility, and source excerpt in one focused view."
  },
  runtime: {
    eyebrow: "Runtime health",
    title: "Separate renderer evidence from generated source",
    detail: "Inspect lifecycle, diagnostics, events, and recovery signals published by the current runtime."
  },
  preview: {
    eyebrow: "Visible output",
    title: "Know exactly what can render now",
    detail: "Review artifact selection, renderer route, browser mount state, and health as separate readiness gates."
  },
  artifacts: {
    eyebrow: "Saved deliverables",
    title: "Review every retained artifact and handoff",
    detail: "Compare visual output, source boundaries, status, session provenance, and available delivery actions."
  },
  domains: {
    eyebrow: "Capability map",
    title: "Match creative domains to truthful delivery routes",
    detail: "Distinguish live browser runtimes from code, export, and external-tool handoffs."
  },
  knowledge: {
    eyebrow: "Evidence foundation",
    title: "Audit sources, retrieval, and creative knowledge",
    detail: "Keep registered technical sources, inspectable creative guidance, and current-run retrieval evidence distinct."
  },
  ai_agents: {
    eyebrow: "AI system",
    title: "Inspect provider and agent responsibilities",
    detail: "Review provider identity, model context, execution route, published roles, and observability boundaries."
  },
  memory: {
    eyebrow: "Context boundary",
    title: "Understand what the current run remembers",
    detail: "Use privacy-safe counts and published context facts without exposing conversation text or private reasoning."
  },
  sessions: {
    eyebrow: "Local workspaces",
    title: "Manage saved creative sessions with confidence",
    detail: "Open, rename, create, or remove browser-profile sessions and review their retained usage evidence."
  },
  telemetry: {
    eyebrow: "Run telemetry",
    title: "One run, four evidence checkpoints",
    detail: "Connect stream, workflow, preview, and evaluation signals while unknown values remain visibly unavailable."
  },
  evaluation: {
    eyebrow: "AI engineering lab",
    title: "Measure each quality claim with defensible evidence",
    detail: "Keep RAG, creative artifact, workflow, and product reliability criteria separate and reviewer-ready."
  },
  manual: {
    eyebrow: "Canonical product documentation",
    title: "From idea to inspected, previewed, and saved output",
    detail: "Follow the five-step path first, then open focused reference cards only when deeper product detail is needed."
  },
  settings: {
    eyebrow: "Workspace preferences",
    title: "Tune the workspace without changing your work",
    detail: "Choose the visual language, reading scale, workspace layout, and prompt defaults for this saved session."
  }
};

const dashboardGroupsWithDedicatedSurface = new Set<DashboardGroupId>([
  "architecture",
  "overview",
  "workflow",
  "workspace",
  "runtime",
  "preview",
  "domains",
  "ai_agents",
  "memory",
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
  const activeStatusLabel = activeGroup.id === "sessions" && sessions
    ? `${sessions.sessions.length} session${sessions.sessions.length === 1 ? "" : "s"}`
    : getDashboardGroupStatus(activeGroup, model, primarySection);
  const activeTone = activeGroup.id === "sessions" && sessions
    ? "ready"
    : getDashboardGroupTone(activeGroup, model, primarySection);
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
        <header className="dashboardNavHeader">
          <div aria-hidden="true" className="dashboardNavHeaderIcon">
            <LayoutDashboard size={19} />
          </div>
          <div>
            <span>Advanced Dashboard</span>
            <strong>Workspace intelligence</strong>
            <p>Create, inspect, and review the current workspace.</p>
          </div>
        </header>
        <div className="dashboardNavSections">
          {dashboardNavigationSections.map((navSection) => (
            <section aria-labelledby={`dashboard-nav-${navSection.id}`} className="dashboardNavSection" key={navSection.id}>
              <h2 id={`dashboard-nav-${navSection.id}`}>{navSection.label}</h2>
              <ul>
                {dashboardGroups.filter((group) => group.navSection === navSection.id).map((group) => {
                  const item = group.categories[0]
                    ? getProductIntelligenceSection(model, group.categories[0])
                    : null;
                  const GroupIcon = group.icon;
                  return (
                    <li key={group.id}>
                      <button
                        aria-current={group.id === activeGroup.id ? "page" : undefined}
                        aria-label={group.label}
                        data-tone={group.id === "sessions" && sessions
                          ? "ready"
                          : getDashboardGroupTone(group, model, item)}
                        onClick={() => selectGroup(group)}
                        type="button"
                      >
                        <span aria-hidden="true" className="dashboardNavItemIcon"><GroupIcon size={15} /></span>
                        <span className="dashboardNavItemLabel">{group.label}</span>
                        <span aria-hidden="true" className="dashboardNavItemStatus" />
                      </button>
                    </li>
                  );
                })}
              </ul>
            </section>
          ))}
        </div>
      </nav>
      <div className="productDashboardContent">
        <header className="productDashboardContentHeader">
          <div>
            <span>Advanced Dashboard</span>
            <h1>{activeGroup.label}</h1>
          </div>
          {primarySection ? <ProductIntelligenceHelp section={primarySection} /> : null}
          <div className="productDashboardStatus" data-tone={activeTone}>
            {activeStatusLabel}
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
  return (
    <DashboardPage
      hero={getDashboardPageHero(group, model, sessions)}
      label={`${group.label} dashboard page`}
    >
      <DashboardGroupBody
        evaluationHistory={evaluationHistory}
        evaluationRunning={evaluationRunning}
        feedback={feedback}
        group={group}
        model={model}
        onRunEvaluation={onRunEvaluation}
        sessions={sessions}
        settings={settings}
      />
    </DashboardPage>
  );
}

function DashboardGroupBody({
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
          categories={group.categories}
          evaluationHistory={evaluationHistory}
          evaluationRunning={evaluationRunning}
          model={model}
          onRunEvaluation={onRunEvaluation}
        />
      </div>
    );
  }

  return (
    <div className="productDashboardGroup" aria-label={`${group.label} details`}>
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
      {group.id === "overview" ? <OverviewDashboardSurface model={model} /> : null}
      {group.id === "workflow" ? <WorkflowLiveMap model={model} /> : null}
      {group.id === "workspace" ? <ActiveDocumentBoard model={model} /> : null}
      {group.id === "runtime" ? <RuntimeDashboardSurface model={model} /> : null}
      {group.id === "preview" ? <PreviewReadinessBoard model={model} /> : null}
      {group.id === "domains" ? <DomainsDashboardSurface model={model} /> : null}
      {group.id === "ai_agents" ? <AiAgentSystemMap model={model} /> : null}
      {group.id === "ai_agents" ? <ProviderDashboardSurface model={model} /> : null}
      {group.id === "memory" ? <MemoryDashboardSurface model={model} /> : null}
      {group.id === "artifacts" ? <ArtifactRegistry model={model} /> : null}
      {group.id === "sessions" && sessions ? <SessionRegistry controls={sessions} /> : null}
      {group.id === "telemetry" ? <TelemetryObservatory model={model} /> : null}
      {group.id === "telemetry" && sessions ? <UserUsageOverview usage={sessions.usage} /> : null}
      {group.id === "ai_agents" && feedback ? (
        <DashboardSection
          action={<DashboardPanelHelp detail="Feedback records an explicit local response to the current artifact. It is not silently sent to an external provider." label="Output feedback" />}
          className="productDashboardFeedback"
          detail="Mark the selected artifact helpful or in need of work; an optional note remains in this local workspace profile."
          eyebrow="Explicit feedback"
          icon={History}
          label="Feedback workspace"
          title="Shape future creative requests"
        >
          <OutputFeedbackPanel
            artifactTitle={feedback.artifactTitle}
            onSubmit={feedback.onSubmit}
          />
        </DashboardSection>
      ) : null}
    </div>
  );
}

function getDashboardPageHero(
  group: DashboardGroup,
  model: ProductIntelligenceModel,
  sessions?: DashboardSessionControls
) {
  const copy = dashboardPageCopy[group.id];
  const primary = group.categories[0]
    ? getProductIntelligenceSection(model, group.categories[0])
    : null;
  const status = group.id === "sessions" && sessions
    ? `${sessions.sessions.length} session${sessions.sessions.length === 1 ? "" : "s"}`
    : getDashboardGroupStatus(group, model, primary);
  const scope = group.id === "manual"
    ? "2-minute start"
    : group.id === "settings"
      ? "Immediately applied"
      : group.id === "telemetry"
        ? "Published values only"
        : `${Math.max(group.categories.length, 1)} evidence ${group.categories.length === 1 ? "stream" : "streams"}`;
  const boundary = group.id === "manual"
    ? "Current product only"
    : group.id === "settings"
      ? "Artifact-safe"
      : "Current workspace";

  return {
    badgeLabel: `${group.label} reading summary`,
    badges: [status, scope, boundary],
    className: `dashboardPageHero--${group.id}`,
    detail: copy.detail,
    eyebrow: copy.eyebrow,
    icon: group.icon,
    title: copy.title,
    tone: group.id === "sessions" && sessions ? "ready" : getDashboardGroupTone(group, model, primary)
  };
}

function getDashboardGroupStatus(
  group: DashboardGroup,
  model: ProductIntelligenceModel,
  section: ProductIntelligenceSection | null
) {
  const details = model.details;
  if (group.id === "manual") return "Complete reference";
  if (group.id === "settings") return "Session scoped";
  if (group.id === "overview" && details?.workflowRuntime) {
    return formatUiStatusLabel(details.workflowRuntime.summary.productOutcome.product_outcome.toLowerCase());
  }
  if (group.id === "architecture") {
    return details?.workflowExecution?.state === "available" ? "Published" : "Evidence pending";
  }
  if (group.id === "workflow" && details?.workflowRuntime) {
    return formatUiStatusLabel(details.workflowRuntime.summary.activity.label);
  }
  if (group.id === "workspace") {
    return getDashboardActiveArtifact(model) ? "Source ready" : "No source";
  }
  if (group.id === "runtime" && details?.runtimeConsole) return details.runtimeConsole.health.label;
  if (group.id === "preview" && details?.telemetryDashboard?.preview) return details.telemetryDashboard.preview.healthLabel;
  if (group.id === "artifacts") {
    return `${model.artifactRegistry.length} artifact${model.artifactRegistry.length === 1 ? "" : "s"}`;
  }
  if (group.id === "domains") {
    return model.domainExperience.state === "available"
      ? `${model.domainExperience.domains.length} contracts`
      : "Registry unavailable";
  }
  if (group.id === "ai_agents" && details?.workflowExecution) {
    return details.workflowExecution.resolvedMode
      ? `${formatRoute(details.workflowExecution.resolvedMode)} route`
      : "Route pending";
  }
  if (group.id === "memory" && details?.conversationContext) {
    return details.conversationContext.source === "stream" ? "Counts published" : "Evidence pending";
  }
  if (group.id === "telemetry" && details?.telemetryDashboard) return details.telemetryDashboard.summary.operatorStatus;
  return formatDashboardStatus(section, "Current workspace");
}

function getDashboardGroupTone(
  group: DashboardGroup,
  model: ProductIntelligenceModel,
  section: ProductIntelligenceSection | null
) {
  const details = model.details;
  if (group.id === "manual" || group.id === "settings") return "ready";
  if (group.id === "runtime" && details?.runtimeConsole) {
    return details.runtimeConsole.health.signal === "healthy" ? "ready" : "attention";
  }
  if (group.id === "preview" && details?.telemetryDashboard?.preview) {
    const preview = details.telemetryDashboard.preview;
    const ready = preview.state === "ready" &&
      preview.active &&
      preview.available &&
      !preview.error &&
      !/(pending|unavailable|failed)/i.test(preview.healthLabel);
    return ready ? "ready" : "attention";
  }
  if (group.id === "domains") {
    return model.domainExperience.state === "available" ? "ready" : "attention";
  }
  if (group.id === "ai_agents" && details?.workflowExecution) {
    return details.workflowExecution.resolvedMode ? "ready" : "attention";
  }
  if (group.id === "memory" && details?.conversationContext) {
    return details.conversationContext.source === "stream" ? "ready" : "attention";
  }
  return section?.tone ?? "empty";
}

function formatDashboardStatus(
  section: ProductIntelligenceSection | null,
  fallback: string
) {
  if (!section) return fallback;
  const summary = formatUiStatusLabel(section.summary);
  if (summary.length <= 28) return summary;
  return {
    active: "Active",
    attention: "Needs attention",
    empty: "Evidence pending",
    ready: "Ready"
  }[section.tone];
}

function OverviewDashboardSurface({ model }: { model: ProductIntelligenceModel }) {
  const section = getProductIntelligenceSection(model, "Overview");
  const details = model.details;
  const outcome = details?.workflowRuntime.summary.productOutcome;
  const artifact = getDashboardActiveArtifact(model);
  const preview = details?.telemetryDashboard.preview;
  const runtime = details?.runtimeConsole.health;
  const retrieval = details?.retrievalRuntime.summary;
  const resolvedRoute = details?.workflowExecution.resolvedMode ?? null;
  const requestedRoute = details?.workflowExecution.requestedMode ?? null;
  const sectionMetric = (label: string) =>
    section.metrics.find((metric) => metric.label === label)?.value ?? "Not published";
  const outcomeLabel = outcome
    ? formatUiStatusLabel(outcome.product_outcome.toLowerCase().replace(/_/g, " "))
    : formatUiStatusLabel(section.summary);
  const outcomeDetail = outcome?.summary ?? section.detail;
  const recovery = outcome?.recovery_action ?? section.notes[0] ?? null;
  const previewLabel = preview?.healthLabel ?? sectionMetric("Preview");
  const previewDetail = preview?.detail ?? "Preview state has not been published for this workspace.";
  const visiblePreviewConfirmed = Boolean(
    preview?.available &&
    !preview.error &&
    preview.healthLabel.toLowerCase() !== "unavailable"
  );
  const runtimeLabel = runtime?.label ?? "Not published";
  const runtimeDetail = runtime?.explanation ?? "Runtime health appears after a renderer publishes evidence.";
  const artifactTitle = artifact?.title ?? sectionMetric("Artifact");
  const artifactDetail = artifact
    ? `${artifact.language} · ${artifact.status} · ${artifact.previewEligible ? "Preview eligible" : "Code or export only"}`
    : "Generate an artifact to establish a retained source and delivery boundary.";
  const needsAttention = section.tone === "attention" ||
    (preview != null && !visiblePreviewConfirmed) ||
    runtime?.signal === "degraded" ||
    runtime?.signal === "failed";

  return (
    <DashboardSection
      action={(
        <DashboardPanelHelp
          detail="This snapshot uses the published workflow outcome, selected artifact, Preview, Runtime, Retrieval, and route models. Conflicting or absent values stay explicit instead of being normalized into a generic success state."
          label="Overview decision snapshot"
        />
      )}
      className="overviewDecisionSurface"
      detail="The few signals a reviewer needs before opening a deeper Dashboard page."
      eyebrow="Decision snapshot"
      icon={LayoutDashboard}
      label="Overview decision snapshot"
      title="Current result and readiness"
      tone={section.tone}
    >
      <div className="overviewDecisionGrid">
        <article className="dashboardInnerCard overviewOutcomeCard" data-tone={section.tone}>
          <span>Current product outcome</span>
          <strong>{outcomeLabel}</strong>
          <p>{outcomeDetail}</p>
          {recovery ? <small>{recovery}</small> : null}
        </article>
        <DashboardMetricGrid
          label="Overview facts"
          metrics={[
            { label: "Execution route", value: formatRoute(resolvedRoute ?? requestedRoute), detail: resolvedRoute ? "Published resolved route" : requestedRoute ? "Requested policy; resolution pending" : "No route published" },
            { label: "Artifacts", value: model.artifactRegistry.length, detail: model.session.title },
            { label: "Preview", value: previewLabel, detail: visiblePreviewConfirmed ? "Visible output confirmed" : "No visible output confirmed" },
            { label: "Runtime", value: runtimeLabel, detail: runtime?.signal ?? "Evidence pending" },
            { label: "Retrieval", value: retrieval ? retrieval.sourceCount === 0 ? "Not used" : retrieval.status : "Not published", detail: retrieval ? `${retrieval.sourceCount} sources · ${retrieval.chunkCount} chunks` : "No run evidence" }
          ]}
        />
      </div>
      <DashboardCardGrid label="Overview reviewer gates" layout="equal" role="list">
        <DashboardInfoCard detail={artifactDetail} icon={FileCode2} role="listitem" title={artifactTitle} />
        <DashboardInfoCard detail={previewDetail} icon={MonitorPlay} role="listitem" title={`Preview · ${previewLabel}`} tone={preview?.available ? "success" : "warning"} />
        <DashboardInfoCard detail={runtimeDetail} icon={Activity} role="listitem" title={`Runtime · ${runtimeLabel}`} tone={runtime?.signal === "healthy" ? "success" : "warning"} />
      </DashboardCardGrid>
      <DashboardCallout
        detail={needsAttention
          ? "The source artifact can still be usable, but visible Preview and Runtime health must be confirmed independently before calling the result ready."
          : "Artifact, Preview, and Runtime evidence are aligned for this workspace snapshot."}
        icon={needsAttention ? AlertTriangle : ShieldCheck}
        title={needsAttention ? "Readiness still needs one explicit check" : "Published readiness signals agree"}
        tone={needsAttention ? "warning" : "success"}
      />
    </DashboardSection>
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
      icon: Bot,
      selected: executionPublished && execution.resolvedMode === "single_agent",
      selectionLabel: "Resolved route"
    },
    {
      title: "Multi-agent",
      detail: "A specialist coordination contract used when the published plan requires additional roles.",
      icon: Network,
      selected: executionPublished && execution.resolvedMode === "multi_agent",
      selectionLabel: "Resolved route"
    },
    {
      title: "Auto",
      detail: "A routing policy that resolves to a published execution route; it is not a third hidden node sequence.",
      icon: SlidersHorizontal,
      selected: executionPublished && execution.requestedMode === "auto",
      selectionLabel: "Requested policy"
    }
  ];

  return (
    <DashboardSection
      action={(
        <DashboardPanelHelp
          detail="Route cards describe selection contracts. Decision facts come from the published execution model, while the expandable topology comes from workflow runtime steps; one never substitutes for the other."
          label="Workflow route guide"
        />
      )}
      className="architectureRouteGuide"
      detail={executionPublished
        ? `This run resolved to ${formatRoute(execution.resolvedMode)} from ${formatRoute(execution.requestedMode)}.`
        : "The requested policy is visible, but a resolved execution route has not been published for this run."}
      eyebrow="Route architecture"
      icon={Network}
      label="Workflow route guide"
      title="Selection policy and executed topology stay separate"
    >
      <DashboardCardGrid className="architectureRouteGrid" layout="equal" role="list">
        {routes.map((route) => (
          <DashboardInfoCard
            detail={route.detail}
            eyebrow={route.selected ? route.selectionLabel : "Route contract"}
            icon={route.icon}
            key={route.title}
            role="listitem"
            selected={route.selected}
            title={route.title}
            tone={route.selected ? "success" : undefined}
          />
        ))}
      </DashboardCardGrid>
      <DashboardMetricGrid
        className="architectureExecutionFacts"
        label="Published execution decision"
        metrics={[
          { label: "Requested policy", value: formatRoute(execution?.requestedMode), detail: "Input routing policy" },
          { label: "Resolved route", value: execution?.resolvedMode ? formatRoute(execution.resolvedMode) : "Not published", detail: "Executed orchestration path" },
          { label: "Research", value: execution?.researcherRequired == null ? "Not published" : execution.researcherRequired ? "Required" : "Skipped", detail: execution?.researcherReason ?? "No decision evidence" },
          { label: "Refinement limit", value: execution?.maxRefinementLoops == null ? "Not published" : execution.maxRefinementLoops, detail: "Published maximum loops" }
        ]}
      />
      {execution?.agentRoles.length ? (
        <div className="architectureRoleRail" aria-label="Published agent responsibilities">
          <span>Published responsibilities</span>
          {execution.agentRoles.map((role) => <strong key={role}>{role}</strong>)}
        </div>
      ) : null}
      <DashboardDisclosure
        className="architectureTopologyDisclosure"
        summary={runtime?.steps.length
          ? `Current run topology · ${runtime.steps.length} published workflow node${runtime.steps.length === 1 ? "" : "s"}`
          : "Current run topology · no workflow nodes published"}
      >
        <div className="architectureTopology" aria-label="Published workflow topology">
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
      </DashboardDisclosure>
      <DashboardCallout
        detail={executionPublished
          ? "The selected route is backed by published execution evidence; expand the topology only when node-level review is needed."
          : "Runtime nodes may exist even when the execution decision is unavailable. They describe observed workflow shape, not a resolved route claim."}
        icon={ShieldCheck}
        title={executionPublished ? "Architecture evidence is linked" : "Policy and topology remain separate"}
      />
    </DashboardSection>
  );
}

function KnowledgeDashboardView({ model }: { model: ProductIntelligenceModel }) {
  const knowledge = getProductIntelligenceSection(model, "Knowledge Base");
  const retrieval = getProductIntelligenceSection(model, "Retrieval");
  const inventory = model.domainExperience.knowledgeBase;
  const retrievalRuntime = model.details?.retrievalRuntime;

  return (
    <div className="productDashboardGroup knowledgeDashboardView" aria-label="Knowledge details">
      <DashboardSection
        action={<ProductIntelligenceHelp section={knowledge} />}
        className="technicalKnowledgeSurface"
        detail={knowledge.detail}
        eyebrow="Technical knowledge"
        icon={Database}
        label="Technical Knowledge Base"
        title={inventory.status === "available"
          ? `${inventory.indexedSourceCount} of ${inventory.registeredSourceCount} official sources indexed`
          : formatUiStatusLabel(knowledge.summary)}
        tone={knowledge.tone}
      >
        <KnowledgeBaseInventorySurface
          detailed
          headerMode="embedded"
          inventory={inventory}
          progressive
        />
        <DashboardCallout
          detail="Persistent registry coverage and the references used by the current request are separate evidence streams. An indexed source is not proof that this run retrieved it."
          icon={ShieldCheck}
          title="Index inventory is not run retrieval"
        />
      </DashboardSection>
      <CreativeKnowledgePanel model={model} />
      <DashboardSection
        action={<ProductIntelligenceHelp section={retrieval} />}
        className="knowledgeRetrievalSurface"
        detail="Published request, source, chunk, quality, and freshness signals for this run."
        eyebrow="Current-run retrieval"
        icon={Gauge}
        label="Current-run retrieval"
        title={retrievalRuntime
          ? retrievalRuntime.summary.sourceCount
            ? `${retrievalRuntime.summary.sourceCount} retrieved source${retrievalRuntime.summary.sourceCount === 1 ? "" : "s"} for this run`
            : "No retrieval used for this run"
          : formatUiStatusLabel(retrieval.summary)}
        tone={retrieval.tone}
      >
        <DashboardMetricGrid
          label="Current-run retrieval facts"
          metrics={[
            { label: "Sources", value: retrievalRuntime?.summary.sourceCount ?? "Not published", detail: "References returned" },
            { label: "Chunks", value: retrievalRuntime?.summary.chunkCount ?? "Not published", detail: "Passages returned" },
            { label: "Context used", value: retrievalRuntime?.summary.usedChunkLabel ?? "Not published", detail: "Generation grounding" },
            { label: "Quality", value: retrievalRuntime?.summary.qualityLabel ?? "Not published", detail: "Published score band" },
            { label: "Freshness", value: retrievalRuntime?.summary.freshnessLabel ?? "Not published", detail: "Run evidence only" }
          ]}
        />
        <DashboardCallout
          detail={retrievalRuntime?.summary.detail ?? retrieval.detail}
          icon={retrievalRuntime?.summary.sourceCount ? ShieldCheck : AlertTriangle}
          title={retrievalRuntime?.summary.headline ?? formatUiStatusLabel(retrieval.summary)}
          tone={retrievalRuntime?.summary.sourceCount ? "success" : "warning"}
        />
        {retrievalRuntime ? (
          <DashboardDisclosure summary="Open current-run retrieval evidence">
            <RetrievalInspector inventory={inventory} runtime={retrievalRuntime} />
          </DashboardDisclosure>
        ) : null}
      </DashboardSection>
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
  const featuredRecords = selectedKind === "all" ? visibleRecords.slice(0, 3) : visibleRecords;
  const remainingRecords = selectedKind === "all" ? visibleRecords.slice(3) : [];

  return (
    <DashboardSection
      action={(
        <DashboardPanelHelp
          detail="Creative Knowledge Base exposes deterministic records from the existing creative-distillation architecture, plus the product's curated studies. It is separate from Technical Knowledge and contains no private prompt, memory, or model chain-of-thought."
          label="Creative Knowledge Base"
        />
      )}
      className="creativeKnowledgeBase"
      detail="Source-backed techniques, workflows, patterns, and browser boundaries—never hidden provider reasoning."
      eyebrow="Creative Knowledge Base"
      icon={Brain}
      label="Creative Knowledge Base"
      title={inventory.status === "available" ? `${inventory.recordCount} inspectable creative guidance records` : "Curated creative studies remain available"}
      tone={inventory.status === "available" ? "ready" : "attention"}
    >
      <DashboardMetricGrid
        className="creativeKnowledgeMetrics"
        label="Creative Knowledge Base summary"
        metrics={[
          ...inventoryMetrics.map((metric) => ({ ...metric, detail: "Published record count" })),
          { label: "Boundary", value: inventory.status === "available" ? "Inspectable" : "Loading", detail: "No private reasoning" }
        ]}
      />
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
          <DashboardCardGrid className="creativeKnowledgeRecordGrid" label="Featured creative knowledge records" role="list">
            {featuredRecords.map((record) => <CreativeKnowledgeCard key={record.id} model={model} record={record} />)}
          </DashboardCardGrid>
          {remainingRecords.length ? (
            <DashboardDisclosure summary={`${remainingRecords.length} more creative knowledge records`}>
              <DashboardCardGrid className="creativeKnowledgeRecordGrid creativeKnowledgeRecordGrid--more" label="Additional creative knowledge records" role="list">
                {remainingRecords.map((record) => <CreativeKnowledgeCard key={record.id} model={model} record={record} />)}
              </DashboardCardGrid>
            </DashboardDisclosure>
          ) : null}
        </>
      ) : null}
      <DashboardDisclosure summary={`Curated creative studies · ${morphogenesisPromptLibrary.length} browser-aware starters`}>
        <section aria-label="Curated creative studies" className="creativeStudyShelf">
          <header>
            <div>
              <span>Curated creative studies</span>
              <strong>Deterministic starter briefs with honest runtime boundaries</strong>
            </div>
            <DashboardPanelHelp
              detail="These are existing curated prompt-library studies. They are deterministic starter briefs and honest runtime boundaries, not a claim of a new generative or semantic system."
              label="Curated creative studies"
            />
          </header>
          <DashboardCardGrid className="creativeStudyGrid" label="Curated study cards" role="list">
            {morphogenesisPromptLibrary.map((study) => (
              <article className="dashboardInnerCard" key={study.id} role="listitem">
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
          </DashboardCardGrid>
        </section>
      </DashboardDisclosure>
      {artifactGroups.length ? (
        <DashboardDisclosure summary="Selected artifact creative direction">
          <section aria-label="Selected artifact creative direction" className="artifactCreativeDirection">
            <header>
              <div><span>Selected artifact direction</span><strong>Published with the current artifact</strong></div>
              <DashboardPanelHelp detail="These fields are structured creative metadata retained with the selected artifact. They are useful for follow-up refinement but do not replace the persistent Creative Knowledge Base." label="Selected artifact direction" />
            </header>
            <ul>
              {artifactGroups.map((group) => <li key={group.label}><strong>{group.label}</strong><span>{group.values.slice(0, 4).join(" · ")}</span></li>)}
            </ul>
          </section>
        </DashboardDisclosure>
      ) : null}
      <DashboardCallout
        detail="Records are typed, local, and source-backed. This view does not fetch or write the index, change provider or workflow routing, or expose private reasoning."
        icon={ShieldCheck}
        title="Inspectable knowledge only"
      />
    </DashboardSection>
  );
}

function CreativeKnowledgeCard({
  model,
  record
}: {
  model: ProductIntelligenceModel;
  record: ProductIntelligenceModel["domainExperience"]["creativeKnowledge"]["records"][number];
}) {
  return (
    <article className="dashboardInnerCard" data-kind={record.kind} role="listitem">
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
  const steps = visibleSteps.length ? visibleSteps : runtime.steps;
  const progressSteps = steps.filter((step) => step.state !== "branch");
  const focusedSteps = progressSteps.length <= 8
    ? progressSteps
    : [...progressSteps.slice(0, 3), ...progressSteps.slice(-5)];
  const hiddenStepCount = steps.length - focusedSteps.length;
  const transitions = runtime.transitions.slice(-6);
  const needsAttention = ["partial", "failed"].includes(runtime.summary.activity.state);

  const renderSteps = (items: typeof steps) => items.map((step) => (
    <li data-state={step.state} key={step.nodeId}>
      <span aria-hidden="true" />
      <div>
        <strong>{step.displayLabel}</strong>
        <p>{step.lastEventDetail ?? step.detail}</p>
      </div>
      <small>{step.durationMs == null ? formatUiStatusLabel(step.state) : formatDashboardDuration(step.durationMs)}</small>
    </li>
  ));

  return (
    <DashboardSection
      action={(
        <DashboardPanelHelp
          detail="This path uses workflow events published by the current run. The default view keeps the opening context and latest nodes visible; the complete ordered path remains expandable without presenting queued work as completed."
          label="Live workflow map"
        />
      )}
      className="workflowLiveMap"
      detail={runtime.summary.activity.detail}
      eyebrow="Execution path"
      icon={GitBranch}
      label="Live workflow map"
      state={runtime.summary.activity.state}
      title={formatUiStatusLabel(runtime.summary.activity.label)}
    >
      <DashboardMetricGrid
        className="workflowLiveSummary"
        label="Workflow run facts"
        metrics={[
          {
            label: "Route",
            value: execution
              ? formatWorkflowGraphRoute({ execution, requestedMode: execution.requestedMode })
              : "Not published",
            detail: execution?.resolvedMode ? "Resolved execution route" : "Resolution pending"
          },
          { label: "Reached", value: `${runtime.summary.reached}/${runtime.summary.total}`, detail: "Published nodes" },
          { label: "Runtime", value: formatDashboardDuration(runtime.summary.totalRuntimeMs), detail: "Published total" },
          { label: "Retries", value: runtime.summary.retryCount, detail: "Reported attempts" }
        ]}
      />
      <ol className="workflowLivePath" aria-label="Focused workflow path">
        {renderSteps(focusedSteps)}
      </ol>
      {hiddenStepCount > 0 ? (
        <DashboardDisclosure summary={`Complete workflow path · ${steps.length} published nodes`}>
          <ol className="workflowLivePath workflowLivePath--all" aria-label="All published workflow nodes">
            {renderSteps(steps)}
          </ol>
        </DashboardDisclosure>
      ) : null}
      {transitions.length ? (
        <DashboardDisclosure summary={`Recent transitions · ${transitions.length}`}>
          <ul aria-label="Recent workflow transitions" className="workflowTransitionRail">
            {transitions.map((transition) => (
              <li key={`${transition.sequence}-${transition.fromNodeId}-${transition.toNodeId}`} data-kind={transition.kind}>
                <span>{transition.label}</span>
                <strong>{transition.fromNodeId} → {transition.toNodeId}</strong>
                {transition.reason ? <small>{transition.reason}</small> : null}
              </li>
            ))}
          </ul>
        </DashboardDisclosure>
      ) : null}
      <DashboardCallout
        detail={needsAttention
          ? "The workflow reached a terminal product state, but downstream Preview or Runtime readiness still needs review. Workflow completion alone is not visual readiness."
          : "The published path, route, and retry evidence are aligned for this run."}
        icon={needsAttention ? AlertTriangle : ShieldCheck}
        title={needsAttention ? "Execution and product readiness differ" : "Workflow evidence is aligned"}
        tone={needsAttention ? "warning" : "success"}
      />
    </DashboardSection>
  );
}

function AiAgentSystemMap({ model }: { model: ProductIntelligenceModel }) {
  const details = model.details;
  if (!details) {
    return null;
  }
  const { conversationContext, providerTelemetry, workflowExecution } = details;
  const roles = workflowExecution.agentRoles;
  const resolvedRoute = workflowExecution.resolvedMode ?? null;

  return (
    <DashboardSection
      action={(
        <DashboardPanelHelp
          detail="This panel reports the provider identity and telemetry emitted for the run, the selected execution route, and context counts. It deliberately does not display prompts, private memory, or model chain-of-thought."
          label="AI system map"
        />
      )}
      className="aiAgentSystemMap"
      detail="Provider identity, route policy, agent responsibilities, and privacy-safe context counts remain separate evidence."
      eyebrow="AI system map"
      icon={Bot}
      label="AI and agent system map"
      title={resolvedRoute ? `${formatRoute(resolvedRoute)} route published` : "Agent route evidence is pending"}
      tone={resolvedRoute ? "ready" : "attention"}
    >
      <DashboardMetricGrid
        label="Execution and provider route"
        metrics={[
          { label: "Provider", value: providerTelemetry.summary.providerLabel, detail: providerTelemetry.summary.generationModeLabel },
          { label: "Model", value: providerTelemetry.summary.modelLabel, detail: providerTelemetry.summary.streamingStatusLabel },
          { label: "Route", value: formatRoute(resolvedRoute ?? workflowExecution.requestedMode), detail: resolvedRoute ? "Resolved execution route" : "Requested policy only" }
        ]}
      />
      <DashboardCardGrid className="aiAgentEvidenceGrid" label="Agent responsibility and context evidence" role="list">
        <DashboardInfoCard
          detail={roles.length ? "Responsibilities published for the resolved route." : "Run a request to publish the responsibilities selected for a resolved route."}
          eyebrow="Responsibilities"
          icon={Bot}
          role="listitem"
          title={roles.length ? `${roles.length} published role${roles.length === 1 ? "" : "s"}` : "No role list published"}
        >
          {roles.length ? <div className="aiAgentRoleChips">{roles.map((role) => <span key={role}>{role}</span>)}</div> : null}
        </DashboardInfoCard>
        <DashboardInfoCard
          detail="Only counts and injection states are exposed; conversation text and private memory remain outside this view."
          eyebrow="Context boundary"
          icon={ShieldCheck}
          role="listitem"
          title={conversationContext.source === "stream" ? "Published context counts" : "No context counts published"}
        >
          <DashboardMetricGrid
            className="aiAgentContextMetrics"
            label="Published context facts"
            metrics={conversationContext.diagnostics.slice(0, 4).map((diagnostic) => ({
              label: diagnostic.label,
              value: diagnostic.value
            }))}
          />
        </DashboardInfoCard>
      </DashboardCardGrid>
      <DashboardCallout
        detail="Requested route policy, resolved agent roles, provider telemetry, and model-visible context are separate claims. Missing provider or context fields remain visibly unpublished."
        icon={ShieldCheck}
        title="No private reasoning is exposed"
      />
    </DashboardSection>
  );
}

function TelemetryObservatory({ model }: { model: ProductIntelligenceModel }) {
  const telemetry = model.details?.telemetryDashboard;
  if (!telemetry) {
    return null;
  }
  const primarySignals = telemetry.signals
    .filter((signal) => ["workflow", "preview", "retrieval"].includes(signal.id))
    .map((signal) => signal.id === "retrieval" && telemetry.retrieval?.sourceCount === 0
      ? {
          ...signal,
          detail: "No retrieval sources or chunks were published for this run.",
          tone: "info" as const,
          value: "Not used this run"
        }
      : signal);
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
    <DashboardSection
      action={(
        <DashboardPanelHelp
          detail="The observatory groups published stream, workflow, preview, retrieval, evaluation, and observability signals. Unreported provider usage and cost values stay visibly unavailable."
          label="Telemetry observatory"
        />
      )}
      className="telemetryObservatory"
      detail="Current outcome, run facts, lifecycle, and supporting evidence for this workspace run."
      eyebrow="Run observatory"
      icon={Radio}
      label="Telemetry observatory"
      state={telemetry.status}
      title="Outcome and measurement facts"
    >
        <div className="telemetryReviewerHero">
          <article className="dashboardInnerCard telemetryOutcomeCard" data-state={telemetry.status}>
            <span>Current product outcome</span>
            <strong>{formatUiStatusLabel(outcome.product_outcome.toLowerCase().replace(/_/g, " "))}</strong>
            <p>{outcome.summary}</p>
            {outcome.recovery_action ? <small>{outcome.recovery_action}</small> : null}
          </article>
          <DashboardMetricGrid
            className="telemetryRunMetrics"
            label="Run measurement facts"
            metrics={[
              { label: "Operator state", value: telemetry.summary.operatorStatus },
              { label: "Runtime", value: telemetry.summary.runtimeLabel },
              { label: "Events / errors", value: `${telemetry.stream.eventCount} / ${telemetry.stream.errorCount}` },
              { label: "Tokens", value: telemetry.provider.summary.tokenLabel },
              { label: "Estimated cost", value: telemetry.provider.summary.costLabel }
            ]}
          />
        </div>
        <ol aria-label="Run evidence checkpoints" className="telemetryLifecycleRail">
          {lifecycle.map((step, index) => (
            <li className="dashboardInnerCard" data-state={step.state} key={step.label}>
              <span>{index + 1}</span>
              <div><strong>{step.label}</strong><small>{step.detail}</small></div>
              <em>{step.value}</em>
            </li>
          ))}
        </ol>
        <DashboardMetricGrid
          className="telemetrySignalMetrics"
          label="Primary telemetry signals"
          metrics={primarySignals.map((signal) => ({
            detail: signal.detail,
            label: signal.label,
            tone: signal.tone,
            value: signal.value
          }))}
        />
        {evidenceSignals.length ? (
          <DashboardDisclosure
            className="telemetryEvidenceDetails"
            summary="Provider, observability, and evaluation evidence"
          >
            <DashboardMetricGrid
              className="telemetrySignalMetrics"
              label="Supporting telemetry signals"
              metrics={evidenceSignals.map((signal) => ({
                detail: signal.detail,
                label: signal.label,
                tone: signal.tone,
                value: signal.value
              }))}
            />
          </DashboardDisclosure>
        ) : null}
        <DashboardCallout
          as="footer"
          className="telemetryEvidenceBoundary"
          detail="Telemetry describes this workspace run; it does not expose provider reasoning or turn missing usage into an estimate."
          icon={ShieldCheck}
          title="Published evidence only"
        />
    </DashboardSection>
  );
}

function ActiveDocumentBoard({ model }: { model: ProductIntelligenceModel }) {
  const artifact = getDashboardActiveArtifact(model);
  if (!artifact) {
    return (
      <DashboardSection
        action={<DashboardPanelHelp detail="This view reports the current saved artifact. Source, Preview capability, and delivery boundary remain separate product signals." label="Active document" />}
        className="activeDocumentBoard dashboardEmptyState"
        detail="Generate an artifact to inspect its source, delivery boundary, and Preview eligibility here."
        eyebrow="Active document"
        icon={FileCode2}
        label="Active document"
        title="No generated source yet"
      >
        <DashboardCallout
          detail="Workspace will populate this page only from a retained artifact; it never fabricates source or Preview evidence."
          icon={ShieldCheck}
          title="Waiting for published source"
        />
      </DashboardSection>
    );
  }

  const lineCount = artifact.content?.trim()
    ? artifact.content.trim().split("\n").length
    : 0;

  return (
    <DashboardSection
      action={<DashboardPanelHelp detail="The active document is the retained source artifact selected by the workspace. Preview status is reported separately because saved source is not proof of a runnable browser output." label="Active document" />}
      className="activeDocumentBoard"
      detail={artifact.summary}
      eyebrow="Active document"
      icon={FileCode2}
      label="Active document"
      title={artifact.title}
    >
      <DashboardMetricGrid
        className="activeDocumentFacts"
        label="Active document facts"
        metrics={[
          { label: "Language", value: artifact.language, detail: `${lineCount} source line${lineCount === 1 ? "" : "s"}` },
          { label: "Artifact status", value: artifact.status, detail: "Retained workspace state" },
          { label: "Preview", value: artifact.previewEligible ? "Eligible" : "Not eligible", detail: "Separate renderer contract" },
          { label: "Domain", value: artifact.domain ? formatCreativeDomain(model, artifact.domain) : "Not published", detail: "Creative runtime classification" }
        ]}
      />
      <DashboardDisclosure summary={`Source excerpt · ${lineCount} line${lineCount === 1 ? "" : "s"}`}>
        <pre aria-label="Active document source excerpt"><code>{artifactSnippet(artifact.content)}</code></pre>
      </DashboardDisclosure>
      <DashboardCallout
        detail={artifact.previewEligible
          ? "The artifact advertises a supported Preview route, but visible output and Runtime health still require their own published evidence."
          : "The source remains usable in Code or export workflows even though this artifact does not currently advertise an internal Preview route."}
        icon={ShieldCheck}
        title="Saved source and visible output are separate"
      />
    </DashboardSection>
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
  const rendererPublished = Boolean(
    preview.renderer && !/(pending|unavailable)/i.test(preview.renderer)
  );
  const mountReady = preview.available && preview.active && preview.state === "ready";
  const healthReady = mountReady &&
    rendererPublished &&
    !preview.error &&
    !/(pending|unavailable|failed)/i.test(preview.healthLabel);
  const stages = [
    { label: "Artifact", value: preview.artifactName ? "Ready" : "Pending", complete: Boolean(preview.artifactName) },
    { label: "Route", value: preview.renderer || "Pending", complete: rendererPublished },
    { label: "Mount", value: mountReady ? "Ready" : preview.state === "generating" ? "Starting" : preview.state === "unavailable" || preview.state === "error" ? "Unavailable" : "Waiting", complete: mountReady },
    { label: "Health", value: preview.healthLabel, complete: healthReady }
  ];
  const ready = stages.every((stage) => stage.complete);
  return (
    <DashboardSection
      action={<DashboardPanelHelp detail="Preview readiness separates selected artifact, renderer route, browser mount, and Runtime health. Saved source or a renderer label alone never claims that visible output is running." label="Preview readiness" />}
      className="previewReadinessBoard"
      detail={preview.detail}
      eyebrow="Preview readiness"
      icon={MonitorPlay}
      label="Preview readiness"
      state={preview.state}
      title={preview.healthLabel}
    >
      <DashboardMetricGrid
        className="previewReadinessMetrics"
        label="Preview readiness facts"
        metrics={[
          { label: "State", value: formatUiStatusLabel(preview.state), detail: mountReady ? "Visible mount confirmed" : "Visible mount unconfirmed" },
          { label: "Renderer", value: preview.renderer || "Pending", detail: "Selected runtime route" },
          { label: "Target", value: preview.target, detail: "Delivery surface" },
          { label: "Artifact", value: preview.artifactName || "No artifact", detail: "Selected source" }
        ]}
      />
      <ol aria-label="Preview lifecycle" className="previewLifecycle">
        {stages.map((stage, index) => <li data-complete={stage.complete ? "true" : "false"} key={stage.label}><span>{index + 1}</span><div><strong>{stage.label}</strong><small>{stage.value}</small></div></li>)}
      </ol>
      <DashboardCallout
        detail={preview.error ?? (ready
          ? "Artifact, renderer, browser mount, and Runtime health are all published as ready."
          : "At least one readiness gate is still pending. Use the lifecycle above to identify the exact missing contract before reviewing visible output.")}
        icon={ready ? ShieldCheck : AlertTriangle}
        title={ready ? "Visible Preview is confirmed" : "Preview is not yet reviewer-ready"}
        tone={ready ? "success" : "warning"}
      />
    </DashboardSection>
  );
}

function RuntimeDashboardSurface({ model }: { model: ProductIntelligenceModel }) {
  const details = model.details;
  if (!details) return null;
  return (
    <DashboardSection
      action={<DashboardPanelHelp detail="The runtime console presents published renderer health, metric signals, diagnostics, context, and events. It never treats saved source as proof that a runtime executed successfully." label="Runtime health console" />}
      className="dashboardComponentSurface runtimeDashboardSurface"
      detail="Current Runtime evidence stays separate from generated source and Preview availability."
      eyebrow="Runtime console"
      icon={Activity}
      label="Runtime health console"
      state={details.runtimeConsole.health.signal}
      title="Renderer, diagnostics, and recovery evidence"
    >
      <RuntimeConsoleInspector console={details.runtimeConsole} presentation="dashboard" productOutcome={details.workflowRuntime.summary.productOutcome} />
    </DashboardSection>
  );
}

function ProviderDashboardSurface({ model }: { model: ProductIntelligenceModel }) {
  const telemetry = model.details?.providerTelemetry;
  if (!telemetry) return null;
  const evidencePending = telemetry.status === "idle" || /pending|unavailable/i.test(telemetry.summary.providerLabel);
  return (
    <DashboardSection
      action={<DashboardPanelHelp detail="Provider observability distinguishes known values from unavailable values. It does not reveal prompts, private memory, or internal model reasoning." label="Provider observability" />}
      className="dashboardComponentSurface providerDashboardSurface"
      detail="Only provider-published identity, usage, timing, configuration provenance, issues, retries, and fallback evidence are shown."
      eyebrow="Provider observability"
      icon={Radio}
      label="Provider observability"
      state={telemetry.status}
      title={evidencePending ? "Provider execution evidence is pending" : `${telemetry.summary.providerLabel} · ${telemetry.summary.modelLabel}`}
      tone={evidencePending ? "attention" : "ready"}
    >
      <DashboardMetricGrid
        label="Provider execution facts"
        metrics={[
          { label: "Generation", value: telemetry.summary.generationModeLabel, detail: telemetry.summary.streamingStatusLabel },
          { label: "Duration", value: telemetry.summary.requestDurationLabel, detail: "Provider-published timing" },
          { label: "Tokens", value: telemetry.summary.tokenLabel, detail: "No estimated usage" },
          { label: "Estimated cost", value: telemetry.summary.costLabel, detail: "Published metadata only" },
          { label: "Issues", value: telemetry.summary.issueLabel, detail: "Errors and warnings" }
        ]}
      />
      <DashboardCallout
        detail={evidencePending
          ? "No provider run metadata is attached to the current workspace snapshot. Unknown identity, usage, timing, and configuration values remain unavailable."
          : "Provider identity, usage, timing, and recovery evidence were published for this run; expand the deep dive for field-level review."}
        icon={evidencePending ? AlertTriangle : ShieldCheck}
        title={evidencePending ? "Unknown provider values stay unknown" : "Provider evidence is inspectable"}
        tone={evidencePending ? "warning" : "success"}
      />
      <DashboardDisclosure className="providerObservabilityDisclosure" summary="Open provider execution evidence">
        <ProviderObservabilityDeepDive telemetry={telemetry} />
      </DashboardDisclosure>
    </DashboardSection>
  );
}

function MemoryDashboardSurface({ model }: { model: ProductIntelligenceModel }) {
  const context = model.details?.conversationContext;
  if (!context) return null;
  const contextPublished = context.source === "stream";
  return (
    <DashboardSection
      action={<DashboardPanelHelp detail="This panel is privacy-safe: it exposes only counts and context-boundary facts published by the runtime. No private session text, embeddings, or provider prompt is shown." label="Memory and context" />}
      className="dashboardComponentSurface memoryDashboardSurface"
      detail="Counts make context assembly inspectable without exposing conversation content, embeddings, provider prompts, or private reasoning."
      eyebrow="Context boundary"
      icon={Brain}
      label="Memory and context"
      state={context.source}
      title={contextPublished ? "Published context counts for this run" : "No context-assembly evidence published"}
      tone={contextPublished ? "ready" : "attention"}
    >
      <DashboardMetricGrid
        label="Published memory and context facts"
        metrics={context.diagnostics.slice(0, 4).map((diagnostic) => ({
          label: diagnostic.label,
          value: diagnostic.value,
          detail: diagnostic.detail
        }))}
      />
      <DashboardCallout
        detail="Visible conversation history is a workspace fact, not proof that those turns, project memory, a running summary, or retrieval context were injected into the model."
        icon={ShieldCheck}
        title="Visible history is not model-visible context"
      />
      <DashboardDisclosure summary={`Open all ${context.diagnostics.length} privacy-safe context diagnostics`}>
        <ConversationContextInspector context={context} />
      </DashboardDisclosure>
    </DashboardSection>
  );
}

function EvaluationDashboardSurface({
  categories,
  evaluationHistory,
  evaluationRunning,
  model,
  onRunEvaluation
}: {
  categories: ProductIntelligenceCategory[];
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
      <DashboardDisclosure
        className="evaluationSupportingEvidence"
        summary="Supporting validation, trace, Product Bug, and LangSmith signals"
      >
        <DashboardCallout
          detail="Session lineage, workstation health, validation, Product Bug, and LangSmith signals remain separate from benchmark results."
          icon={ShieldCheck}
          title="Supporting evidence does not replace measured results"
        />
        <div className="evaluationDashboardSplit">
          <EvaluationSessionDashboard evaluation={telemetry.evaluation} />
          <WorkstationDashboardSurface dashboard={workstation} />
        </div>
        {categories.map((category) => {
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
      </DashboardDisclosure>
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

function DomainsDashboardSurface({ model }: { model: ProductIntelligenceModel }) {
  const catalog = model.domainExperience;
  const liveCount = catalog.domains.filter((domain) => domain.livePreview).length;
  const exportCount = catalog.domains.filter((domain) => domain.deliveryKind === "code_export").length;
  const handoffCount = catalog.domains.filter((domain) => domain.deliveryKind === "external_handoff").length;
  const available = catalog.state === "available";

  return (
    <DashboardSection
      action={(
        <DashboardPanelHelp
          detail="Domain contracts publish three mutually exclusive delivery routes: a validated browser runtime, retained source for another runtime, or a documented external-tool handoff."
          label="Domain capability map"
        />
      )}
      className="domainsDashboardSurface"
      detail="Start with the four validated browser runtimes; expand the larger export registries only when a delivery decision requires them."
      eyebrow="Delivery capability map"
      icon={Shapes}
      label="Domain capability map"
      title={available ? `${liveCount} live runtimes across ${catalog.domains.length} domain contracts` : "Domain registry unavailable"}
      tone={available ? "ready" : "attention"}
    >
      <DashboardMetricGrid
        label="Domain delivery totals"
        metrics={[
          { label: "Registered", value: catalog.domains.length, detail: "Published domain contracts" },
          { label: "Live in browser", value: liveCount, detail: "Validated internal runtimes" },
          { label: "Source export", value: exportCount, detail: "Code without runtime claim" },
          { label: "External handoff", value: handoffCount, detail: "Named tool packages" }
        ]}
      />
      <DomainExperienceSurface
        activeDomainId={model.activeDomainId}
        catalog={catalog}
        collapseSecondary
        detailed
        embedded
        includeKnowledgeBase={false}
      />
      <DashboardCallout
        detail="Only cards in Live in this browser claim executable output inside the workspace. Source exports and external handoffs remain usable deliverables without implying an internal runtime."
        icon={ShieldCheck}
        title="Delivery labels are product boundaries"
        tone="success"
      />
    </DashboardSection>
  );
}

function ArtifactRegistry({ model }: { model: ProductIntelligenceModel }) {
  if (model.artifactRegistry.length === 0) {
    return (
      <DashboardSection
        action={<DashboardPanelHelp detail="This registry is populated only by artifacts retained in the active workspace snapshot." label="Artifact registry" />}
        className="artifactRegistry dashboardEmptyState"
        detail="Generate or retain an artifact to review its source, delivery boundary, and Preview evidence here."
        eyebrow="Artifact registry"
        icon={FileOutput}
        label="Artifact registry"
        title="No saved deliverables yet"
      >
        <DashboardCallout
          detail="The Dashboard does not invent a deliverable or simulate a visual result before the workspace publishes one."
          icon={ShieldCheck}
          title="Waiting for a retained artifact"
        />
      </DashboardSection>
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
    <DashboardSection
      action={
        <DashboardPanelHelp
          detail="This registry lists artifacts retained by the active workspace snapshot. A live visual is mounted only through a supported preview contract. Otherwise the card explains the code or external-tool handoff without simulating output."
          label="Artifact registry"
        />
      }
      className="artifactRegistry"
      detail="Each deliverable exposes a real visual only when a supported runtime can render it; every other artifact states its handoff boundary instead."
      eyebrow="Artifact registry"
      icon={FileOutput}
      label="Artifact registry"
      title={`${model.artifactRegistry.length} saved deliverable${model.artifactRegistry.length === 1 ? "" : "s"}`}
    >
      <DashboardCardGrid className="artifactRegistryList" label="Saved deliverables" role="list">
        {model.artifactRegistry.map((artifact) => {
          const candidate = previewCandidates.get(artifact.id) ?? null;
          const previewLabel = candidate?.canRender
            ? "Live visual"
            : artifact.previewEligible
              ? "Preview evidence unavailable"
              : "Code / export only";
          return (
          <article
            aria-current={artifact.id === activeArtifact?.id ? "true" : undefined}
            className="dashboardInnerCard artifactRegistryCard"
            key={artifact.id}
            role="listitem"
          >
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
            <DashboardDisclosure className="artifactSourceDisclosure" summary="View source excerpt">
              <pre><code>{artifactSnippet(artifact.content)}</code></pre>
            </DashboardDisclosure>
          </article>
          );
        })}
      </DashboardCardGrid>
      <DashboardCallout
        detail="A retained source file, a supported Preview route, and a visible browser runtime are separate facts. Each card reports only the delivery evidence available for that artifact."
        icon={ShieldCheck}
        title="Artifact truth stays attached to the deliverable"
      />
    </DashboardSection>
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
  const totalArtifacts = controls.sessions.reduce((total, session) => total + session.artifactCount, 0);
  const registeredSessionIds = new Set(controls.sessions.map((session) => session.sessionId));
  const registeredUsage = controls.usage.filter((usage) => registeredSessionIds.has(usage.sessionId));
  const totalRuns = registeredUsage.reduce((total, usage) => total + usage.runCount, 0);
  const knownTokenSessions = registeredUsage.filter((usage) => usage.totalTokens != null);
  const totalKnownTokens = knownTokenSessions.reduce((total, usage) => total + (usage.totalTokens ?? 0), 0);

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
    <DashboardSection
      action={(
        <div className="dashboardFeatureActions">
          <DashboardPanelHelp
            detail="Sessions are isolated to this browser profile. Tokens and cost are shown only when the provider published them for the retained runs."
            label="Session registry"
          />
          <button onClick={controls.onCreate} type="button">New session</button>
        </div>
      )}
      className="sessionRegistry"
      detail="Open, rename, create, or remove browser-profile sessions while retained usage remains explicitly reported or unknown."
      eyebrow="Session registry"
      icon={History}
      label="Session registry"
      title={`${controls.sessions.length} local creative session${controls.sessions.length === 1 ? "" : "s"}`}
    >
      <DashboardMetricGrid
        label="Local session totals"
        metrics={[
          { label: "Sessions", value: controls.sessions.length, detail: "This browser profile" },
          { label: "Artifacts", value: totalArtifacts, detail: "Retained deliverables" },
          { label: "Runs", value: totalRuns, detail: "Retained usage records" },
          { label: "Known tokens", value: knownTokenSessions.length ? formatCompactUsage(totalKnownTokens) : "Not reported", detail: `${knownTokenSessions.length}/${registeredUsage.length} sessions reported` }
        ]}
      />
      <DashboardTableFrame className="sessionRegistryTableFrame">
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
      </DashboardTableFrame>
      <DashboardCallout
        detail="Session records are isolated to this browser profile. Unknown tokens or cost never enter totals, and a damaged local record is skipped instead of replacing the active workspace."
        icon={ShieldCheck}
        title="Local session boundaries stay explicit"
      />
    </DashboardSection>
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
    <DashboardSection
      action={(
        <DashboardPanelHelp
          detail="These totals aggregate only local-session records with provider-published usage fields. Unknown token or cost values are deliberately kept out of the totals."
          label="Browser profile usage"
        />
      )}
      className="browserUsageOverview"
      detail="Only provider-published token and cost data is counted. Missing provider fields remain visibly unreported."
      eyebrow="Browser profile totals"
      icon={Database}
      label="Browser profile usage"
      title="Usage retained across local sessions"
    >
      <DashboardMetricGrid
        className="userUsageMetrics"
        label="Browser profile usage totals"
        metrics={[
          { label: "Sessions", value: usage.length },
          { label: "Completed runs", value: runCount },
          { label: "Known tokens", value: knownTokenSessions.length ? formatCompactUsage(totalTokens) : "Not reported" },
          { label: "Known cost", value: knownCostSessions.length ? `$${totalCost.toFixed(4)}` : "Not reported" }
        ]}
      />
    </DashboardSection>
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
    <section aria-label="Dashboard settings" className="productDashboardGroup dashboardSettings">
      <DashboardSection
        action={<DashboardPanelHelp detail="Theme changes the saved visual palette for this browser workspace. It does not change the generated artifact or workflow configuration." label="Appearance settings" />}
        as="article"
        detail="The selected theme controls the accessible text, surface, and accent colours for this saved session."
        eyebrow="Appearance"
        icon={Palette}
        title="Theme and colour"
      >
        <DashboardCardGrid className="dashboardThemeChoices" label="Theme options" role="group">
          {themeGroups.map((group) => (
            <div className="dashboardInnerCard dashboardThemeChoiceGroup" data-group={group.id} key={group.id}>
              <span>{group.label}</span>
              <div role="group" aria-label={group.label}>
                {group.themes.map((theme) => (
                  <button aria-pressed={preferences.theme === theme.value} data-theme={theme.value} key={theme.value} onClick={() => controls.onPreferencesChange({ theme: theme.value })} type="button">{theme.label}</button>
                ))}
              </div>
            </div>
          ))}
        </DashboardCardGrid>
      </DashboardSection>
      <DashboardSection
        action={<DashboardPanelHelp detail="Heading, body, label, control, and code scales are saved per local workspace session, so the reading preference returns when that session is reopened." label="Typography settings" />}
        as="article"
        detail="Set headings, body copy, labels and controls, and code independently. These settings are restored with this session."
        eyebrow="Typography"
        icon={Type}
        title="Comfortable reading"
      >
        <DashboardCardGrid className="dashboardSettingRows" label="Typography scales" role="group">
          <DashboardScaleControl label="Headings" onChange={(headingFontSize) => controls.onPreferencesChange({ headingFontSize })} preview="Heading" previewKind="heading" scales={scales} value={preferences.headingFontSize} />
          <DashboardScaleControl label="Body text" onChange={(uiFontSize) => controls.onPreferencesChange({ uiFontSize })} preview="Body text" previewKind="body" scales={scales} value={preferences.uiFontSize} />
          <DashboardScaleControl label="Labels and controls" onChange={(labelFontSize) => controls.onPreferencesChange({ labelFontSize })} preview="Label" previewKind="label" scales={scales} value={preferences.labelFontSize} />
          <DashboardScaleControl label="Code text" onChange={(codeFontSize) => controls.onPreferencesChange({ codeFontSize })} preview="Code" previewKind="code" scales={scales} value={preferences.codeFontSize} />
        </DashboardCardGrid>
      </DashboardSection>
      <DashboardSection
        action={<DashboardPanelHelp detail="These controls restore the workspace, preview, inspector, density, and display choices that are saved for this local session." label="Workspace settings" />}
        as="article"
        detail="Bring back the compact, focus, preview, and developer controls that are intentionally removed from the top bar."
        eyebrow="Workspace"
        icon={PanelLeft}
        title="Layout and focus"
      >
        <DashboardCardGrid className="dashboardSettingRows" label="Workspace layout controls" role="group">
          <DashboardToggle label="Density" onClick={() => controls.onDensityChange(layoutState.density === "cozy" ? "compact" : "cozy")} pressed={layoutState.density === "compact"} value={layoutState.density === "cozy" ? "Cozy" : "Compact"} />
          <DashboardToggle label="Focus" onClick={controls.onFocusModeToggle} pressed={controls.isFocusMode} value={controls.isFocusMode ? "On" : "Off"} />
          <DashboardToggle label="Sessions rail" onClick={controls.onSidebarToggle} pressed={!layoutState.sidebarCollapsed} value={layoutState.sidebarCollapsed ? "Collapsed" : "Open"} />
          <DashboardToggle label="Inspector" onClick={controls.onInspectorToggle} pressed={!layoutState.inspectorCollapsed} value={layoutState.inspectorCollapsed ? "Collapsed" : "Open"} />
          <DashboardToggle label="Preview shelf" onClick={controls.onPreviewToggle} pressed={controls.isPreviewOpen} value={controls.isPreviewOpen ? "Open" : "Closed"} />
          <DashboardToggle label="Display mode" onClick={() => controls.onPreferencesChange({ showDebugPanels: !preferences.showDebugPanels })} pressed={preferences.showDebugPanels} value={preferences.showDebugPanels ? "Developer" : "User"} />
          <DashboardToggle label="Preview behavior" onClick={() => controls.onPreferencesChange({ autoOpenPreview: !preferences.autoOpenPreview })} pressed={preferences.autoOpenPreview} value={preferences.autoOpenPreview ? "Automatic" : "Manual"} />
        </DashboardCardGrid>
      </DashboardSection>
      <DashboardSection
        action={<DashboardPanelHelp detail="These settings mirror the composer controls. Workflow and creativity apply to the next prompt, while the selected provider is shown for clarity and remains server-configured." label="Generation settings" />}
        as="article"
        detail="Set the workflow route and creative profile used for new prompts. Provider routing remains securely configured for this workspace."
        eyebrow="Generation"
        icon={SlidersHorizontal}
        title="Prompt defaults"
      >
        <DashboardCardGrid className="dashboardGenerationRows" label="Generation defaults" layout="equal" role="group">
          <label className="dashboardGenerationControl dashboardInnerCard">
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
          <DashboardInfoCard
            className="dashboardGenerationControl dashboardProviderCard"
            detail="Configured server-side and shown here for clarity. Provider credentials and routing cannot be changed from this workspace."
            eyebrow="AI provider"
            icon={Bot}
            label="AI provider"
            role="group"
            title="OpenAI"
          />
          <label className="dashboardGenerationControl dashboardInnerCard">
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
        </DashboardCardGrid>
      </DashboardSection>
    </section>
  );
}

function DashboardScaleControl({ label, onChange, preview, previewKind, scales, value }: { label: string; onChange: (value: WorkspacePreferences["uiFontSize"]) => void; preview: string; previewKind: "heading" | "body" | "label" | "code"; scales: WorkspacePreferences["uiFontSize"][]; value: WorkspacePreferences["uiFontSize"]; }) {
  return <div aria-label={`${label} scale`} className="dashboardInnerCard dashboardScaleControl" role="group"><strong>{label}</strong><div>{scales.map((scale) => <button aria-pressed={value === scale} key={scale} onClick={() => onChange(scale)} type="button">{scale}</button>)}</div><span className="dashboardTypographyPreview" data-kind={previewKind}>{preview}</span></div>;
}

function DashboardToggle({ label, onClick, pressed, value }: { label: string; onClick: () => void; pressed: boolean; value: string }) {
  return <button aria-pressed={pressed} className="dashboardInnerCard dashboardToggle" onClick={onClick} type="button"><span>{label}</span><strong>{value}</strong></button>;
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
