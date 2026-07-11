"use client";

import { useEffect, useState } from "react";
import { ArrowLeft, CircleHelp } from "lucide-react";
import { DomainExperienceSurface } from "./domain-experience-surface";
import {
  getProductIntelligenceSection,
  type ProductIntelligenceCategory,
  type ProductIntelligenceModel,
  type ProductIntelligenceSection
} from "@/lib/product-intelligence";
import type {
  EvaluationHistoryRecord,
  FeedbackSentiment
} from "@/lib/product-controls";
import { OutputFeedbackPanel } from "./output-feedback-panel";

type DashboardFeedback = {
  artifactTitle: string;
  onSubmit: (sentiment: FeedbackSentiment, comment: string | null) => void;
};

type ProductIntelligenceDashboardProps = {
  activeCategory: ProductIntelligenceCategory;
  model: ProductIntelligenceModel;
  onCategoryChange: (category: ProductIntelligenceCategory) => void;
  onClose: () => void;
  onRunEvaluation?: () => Promise<void>;
  evaluationHistory?: EvaluationHistoryRecord[];
  feedback?: DashboardFeedback;
};

type DashboardGroupId =
  | "overview"
  | "architecture"
  | "workspace"
  | "knowledge"
  | "ai_memory"
  | "telemetry"
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
    label: "Architecture & Workflow",
    detail: "Routing, workflow progress, and agent responsibilities.",
    categories: ["Architecture", "Workflow", "Agents"]
  },
  {
    id: "workspace",
    label: "Workspace & Runtime",
    detail: "Preview, code, artifacts, and runtime health.",
    categories: ["Runtime", "Preview", "Code", "Artifacts"]
  },
  {
    id: "knowledge",
    label: "Knowledge & Domains",
    detail: "Domain contracts, source inventory, and current retrieval.",
    categories: ["Domains", "Knowledge Base", "Retrieval"]
  },
  {
    id: "ai_memory",
    label: "AI, Agents & Memory",
    detail: "Session context, provider route, and privacy-safe memory detail.",
    categories: ["Memory", "Sessions", "Providers"]
  },
  {
    id: "telemetry",
    label: "Telemetry & Evaluation",
    detail: "Usage, runtime signals, validation, product bugs, and traces.",
    categories: ["Telemetry", "Metrics", "Validation", "Product Bugs", "LangSmith"]
  },
  {
    id: "manual",
    label: "Manual",
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
  feedback
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
            {primarySection?.summary ?? "Guide"}
          </div>
          <button
            aria-label="Return to workspace"
            onClick={onClose}
            title="Return to workspace"
            type="button"
          >
            <ArrowLeft aria-hidden="true" size={16} />
          </button>
        </header>
        <DashboardGroupView
          evaluationRunning={evaluationRunning}
          group={activeGroup}
          model={model}
          onRunEvaluation={onRunEvaluation ? runEvaluation : undefined}
          evaluationHistory={evaluationHistory}
          feedback={feedback}
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
  feedback
}: {
  evaluationRunning: boolean;
  group: DashboardGroup;
  model: ProductIntelligenceModel;
  onRunEvaluation?: () => Promise<void>;
  evaluationHistory: EvaluationHistoryRecord[];
  feedback?: DashboardFeedback;
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

  return (
    <div className="productDashboardGroup" aria-label={`${group.label} details`}>
      {group.categories.map((category) => {
        const section = getProductIntelligenceSection(model, category);
        return (
          <section className="productDashboardGroupSection" key={category}>
            <header>
              <span>{category}</span>
              <strong>{section.summary}</strong>
              <p>{section.detail}</p>
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
      {group.id === "ai_memory" && feedback ? (
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
      />
    );
  }

  const notes = detailed ? section.notes : section.notes.slice(0, 2);
  const metrics = detailed ? section.metrics : section.metrics.slice(0, 3);

  return (
    <div className="productIntelligenceSection" data-tone={section.tone}>
      <article className="productIntelligenceHero">
        <span>{section.category}</span>
        <strong>{section.summary}</strong>
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
