import type { AssistantWorkspaceSnapshot } from "./assistant-client";
import type {
  V3InspectorPanelItem,
  V3InspectorPanelsModel
} from "./v3-inspector-panels";
import type { WorkflowRuntimeModel } from "./workflow-runtime";
import type { WorkstationState } from "./workstation-state";

export type WorkstationDashboardCardTone =
  | "good"
  | "watch"
  | "missing"
  | "error";

export type WorkstationDashboardCard = {
  id:
    | "creative_quality"
    | "confidence"
    | "consistency"
    | "artifact_readiness"
    | "runtime_fit"
    | "evaluation_report"
    | "workflow_health"
    | "hitl_recommendation";
  label: string;
  tone: WorkstationDashboardCardTone;
  value: string;
  summary: string;
  detail: string;
  source: string;
};

export type WorkstationDashboardModel = {
  state: "available" | "empty";
  cards: WorkstationDashboardCard[];
  summary: {
    goodCount: number;
    watchCount: number;
    missingCount: number;
    errorCount: number;
  };
};

export type BuildWorkstationDashboardInput = {
  runtime: WorkflowRuntimeModel;
  snapshot: AssistantWorkspaceSnapshot;
  v3InspectorPanels: V3InspectorPanelsModel;
  workstationState: WorkstationState;
};

export function buildWorkstationDashboardModel({
  runtime,
  snapshot,
  v3InspectorPanels,
  workstationState
}: BuildWorkstationDashboardInput): WorkstationDashboardModel {
  const activeArtifact = workstationState.selection.activeArtifact;
  const cards: WorkstationDashboardCard[] = [
    metadataCard({
      id: "creative_quality",
      item:
        findV3Item(v3InspectorPanels, "creative_score") ??
        findV3Item(v3InspectorPanels, "creative_critic") ??
        findV3Item(v3InspectorPanels, "self_evaluation"),
      label: "Creative Quality",
      missingSummary: "Creative quality score metadata has not been captured yet."
    }),
    metadataCard({
      id: "confidence",
      item: findV3Item(v3InspectorPanels, "creative_confidence"),
      label: "Confidence",
      missingSummary: "Creative confidence metadata has not been captured yet."
    }),
    metadataCard({
      id: "consistency",
      item: findV3Item(v3InspectorPanels, "consistency_validation"),
      label: "Consistency",
      missingSummary: "Consistency validation metadata has not been captured yet."
    }),
    artifactReadinessCard(snapshot, activeArtifact),
    runtimeFitCard(snapshot, activeArtifact, v3InspectorPanels),
    metadataCard({
      id: "evaluation_report",
      item:
        findV3Item(v3InspectorPanels, "evaluation_report") ??
        findV3Item(v3InspectorPanels, "evaluation_trace"),
      label: "Evaluation Report",
      missingSummary: "Evaluation report metadata has not been captured yet."
    }),
    workflowHealthCard(runtime, workstationState),
    hitlRecommendationCard(v3InspectorPanels)
  ];
  const goodCount = cards.filter((card) => card.tone === "good").length;
  const watchCount = cards.filter((card) => card.tone === "watch").length;
  const missingCount = cards.filter((card) => card.tone === "missing").length;
  const errorCount = cards.filter((card) => card.tone === "error").length;

  return {
    state: cards.some((card) => card.tone !== "missing") ? "available" : "empty",
    cards,
    summary: {
      goodCount,
      watchCount,
      missingCount,
      errorCount
    }
  };
}

function metadataCard({
  id,
  item,
  label,
  missingSummary
}: {
  id: WorkstationDashboardCard["id"];
  item: V3InspectorPanelItem | null;
  label: string;
  missingSummary: string;
}): WorkstationDashboardCard {
  if (!item || item.status === "missing") {
    return {
      id,
      label,
      tone: "missing",
      value: "Missing",
      summary: missingSummary,
      detail: "Awaiting compatible V3 metadata.",
      source: "V3 metadata"
    };
  }

  return {
    id,
    label,
    tone: item.status === "available" ? "good" : "watch",
    value: item.status === "available" ? "Available" : "Partial",
    summary: item.summary,
    detail: item.details[0] ?? dashboardItemSource(item),
    source: dashboardItemSource(item)
  };
}

function artifactReadinessCard(
  snapshot: AssistantWorkspaceSnapshot,
  activeArtifact: AssistantWorkspaceSnapshot["artifacts"][number] | null
): WorkstationDashboardCard {
  const artifact = activeArtifact ?? snapshot.artifacts[0] ?? null;
  if (!artifact) {
    return {
      id: "artifact_readiness",
      label: "Artifact Readiness",
      tone: "missing",
      value: "Missing",
      summary: "No generated artifact is available for readiness review.",
      detail: "Generate an artifact to populate readiness metadata.",
      source: "Workspace artifact"
    };
  }

  const previewReady = artifact.previewEligible === true || snapshot.preview.available;
  const tone = previewReady || artifact.status.toLowerCase().includes("ready")
    ? "good"
    : "watch";

  return {
    id: "artifact_readiness",
    label: "Artifact Readiness",
    tone,
    value: artifact.status,
    summary: artifact.summary,
    detail: previewReady ? "Preview-capable artifact." : "Preview readiness is not confirmed.",
    source: artifact.title
  };
}

function runtimeFitCard(
  snapshot: AssistantWorkspaceSnapshot,
  activeArtifact: AssistantWorkspaceSnapshot["artifacts"][number] | null,
  v3InspectorPanels: V3InspectorPanelsModel
): WorkstationDashboardCard {
  const capability =
    findV3Item(v3InspectorPanels, "artifact_capability_matrix") ??
    findV3Item(v3InspectorPanels, "artifact_plan");
  if (capability && capability.status !== "missing") {
    return metadataCard({
      id: "runtime_fit",
      item: capability,
      label: "Runtime Fit",
      missingSummary: "Runtime fit metadata has not been captured yet."
    });
  }

  const artifact = activeArtifact ?? snapshot.artifacts[0] ?? null;
  if (snapshot.preview.available || artifact?.runtime) {
    return {
      id: "runtime_fit",
      label: "Runtime Fit",
      tone: "good",
      value: artifact?.runtime ?? snapshot.preview.target,
      summary: snapshot.preview.summary,
      detail: snapshot.preview.renderer || "Runtime inferred from workspace preview.",
      source: "Preview metadata"
    };
  }

  return {
    id: "runtime_fit",
    label: "Runtime Fit",
    tone: "missing",
    value: "Missing",
    summary: "Runtime fit metadata has not been captured yet.",
    detail: "Awaiting artifact capability or preview metadata.",
    source: "Runtime metadata"
  };
}

function workflowHealthCard(
  runtime: WorkflowRuntimeModel,
  workstationState: WorkstationState
): WorkstationDashboardCard {
  const hasError =
    runtime.summary.status === "failed" ||
    workstationState.currentRun.errorSeen;
  const tone: WorkstationDashboardCardTone = hasError
    ? "error"
    : runtime.summary.status === "partial" || runtime.summary.retryCount > 0
      ? "watch"
      : "good";

  return {
    id: "workflow_health",
    label: "Workflow Health",
    tone,
    value: formatWorkflowStatus(runtime.summary.status),
    summary: runtime.summary.currentStep,
    detail: `${runtime.summary.transitionCount} transitions / ${runtime.summary.retryCount} retries`,
    source: "Workflow runtime"
  };
}

function hitlRecommendationCard(
  v3InspectorPanels: V3InspectorPanelsModel
): WorkstationDashboardCard {
  const hitlSignal = v3InspectorPanels.panels
    .flatMap((panel) => panel.items)
    .flatMap((item) => [item.summary, ...item.details])
    .find((text) => /\b(hitl|human review|review required)\b/i.test(text));

  if (hitlSignal) {
    return {
      id: "hitl_recommendation",
      label: "HITL Recommendation",
      tone: "watch",
      value: "Review",
      summary: hitlSignal,
      detail: "Review-sensitive metadata surfaced by V3 evaluation.",
      source: "V3 metadata"
    };
  }

  if (v3InspectorPanels.state === "available") {
    return {
      id: "hitl_recommendation",
      label: "HITL Recommendation",
      tone: "good",
      value: "No flag",
      summary: "No HITL recommendation is surfaced in available V3 metadata.",
      detail: "This does not override future human review metadata.",
      source: "V3 metadata"
    };
  }

  return {
    id: "hitl_recommendation",
    label: "HITL Recommendation",
    tone: "missing",
    value: "Unknown",
    summary: "HITL recommendation metadata has not been captured yet.",
    detail: "Awaiting evaluation or confidence metadata.",
    source: "V3 metadata"
  };
}

function findV3Item(
  model: V3InspectorPanelsModel,
  id: string
): V3InspectorPanelItem | null {
  return (
    model.panels
      .flatMap((panel) => panel.items)
      .find((item) => item.id === id) ?? null
  );
}

function dashboardItemSource(item: V3InspectorPanelItem) {
  if (item.eventSequence != null) {
    return `Event ${item.eventSequence}`;
  }
  return item.source === "provenance" ? "Provenance" : "V3 metadata";
}

function formatWorkflowStatus(status: string) {
  switch (status) {
    case "completed":
      return "Success";
    case "completed_with_preview_error":
      return "Partial";
    case "partial":
      return "Partial";
    case "failed":
      return "Failure";
    case "running":
      return "Running";
    default:
      return "Idle";
  }
}
