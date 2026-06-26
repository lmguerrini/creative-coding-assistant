import type { ProvenanceEngineModel } from "./provenance-engine";
import type {
  WorkflowExplorerModel,
  WorkflowExplorerStage,
  WorkflowExplorerStageId
} from "./workflow-explorer";
import type { WorkflowRuntimeModel } from "./workflow-runtime";
import type { WorkstationState } from "./workstation-state";

export type CreativeTimelineEventStatus =
  | "complete"
  | "active"
  | "missing"
  | "warning"
  | "error";

export type CreativeTimelineEvent = {
  id: string;
  label: string;
  status: CreativeTimelineEventStatus;
  summary: string;
  metadataAvailability: string;
  eventCount: number;
  sourceCount: number;
  warning: string | null;
};

export type CreativeTimelineModel = {
  state: "available" | "empty";
  events: CreativeTimelineEvent[];
  summary: {
    completeCount: number;
    activeCount: number;
    warningCount: number;
    missingCount: number;
    errorCount: number;
  };
};

export type BuildCreativeTimelineInput = {
  explorer: WorkflowExplorerModel;
  provenance: ProvenanceEngineModel;
  runtime: WorkflowRuntimeModel;
  workstationState: WorkstationState;
};

type TimelineStageDefinition = {
  id: string;
  label: string;
  explorerStageId?: WorkflowExplorerStageId;
};

const timelineStages: TimelineStageDefinition[] = [
  {
    id: "request_intake",
    label: "Request intake"
  },
  {
    id: "planning",
    label: "Planning",
    explorerStageId: "planning"
  },
  {
    id: "retrieval",
    label: "Retrieval",
    explorerStageId: "retrieval"
  },
  {
    id: "creative_intelligence",
    label: "Creative intelligence",
    explorerStageId: "creative_intelligence"
  },
  {
    id: "generative_design",
    label: "Generative design",
    explorerStageId: "generative_design"
  },
  {
    id: "artifact_intelligence",
    label: "Artifact intelligence",
    explorerStageId: "artifact_intelligence"
  },
  {
    id: "creative_evaluation",
    label: "Creative evaluation",
    explorerStageId: "creative_evaluation"
  },
  {
    id: "final_synthesis",
    label: "Final synthesis",
    explorerStageId: "final_response"
  }
];

export function buildCreativeTimelineModel({
  explorer,
  provenance,
  runtime,
  workstationState
}: BuildCreativeTimelineInput): CreativeTimelineModel {
  const events = timelineStages.map((definition) =>
    buildTimelineEvent({
      definition,
      explorer,
      provenance,
      runtime,
      workstationState
    })
  );

  return {
    state: events.some((event) => event.status !== "missing")
      ? "available"
      : "empty",
    events,
    summary: {
      activeCount: events.filter((event) => event.status === "active").length,
      completeCount: events.filter((event) => event.status === "complete").length,
      errorCount: events.filter((event) => event.status === "error").length,
      missingCount: events.filter((event) => event.status === "missing").length,
      warningCount: events.filter((event) => event.status === "warning").length
    }
  };
}

function buildTimelineEvent({
  definition,
  explorer,
  provenance,
  runtime,
  workstationState
}: {
  definition: TimelineStageDefinition;
  explorer: WorkflowExplorerModel;
  provenance: ProvenanceEngineModel;
  runtime: WorkflowRuntimeModel;
  workstationState: WorkstationState;
}): CreativeTimelineEvent {
  if (definition.id === "request_intake") {
    return requestIntakeEvent(workstationState);
  }

  const stage = definition.explorerStageId
    ? explorer.stages.find((candidate) => candidate.id === definition.explorerStageId)
    : null;

  if (!stage) {
    return missingEvent(definition);
  }

  const sourceCount = provenanceSourceCount(definition.id, provenance);
  const metadataAvailability = `${availableGroupCount(stage)}/${stage.metadataGroups.length} metadata groups`;
  const warning = stageWarning(stage, sourceCount);

  return {
    id: definition.id,
    label: definition.label,
    status: timelineStatus(stage),
    summary:
      stage.latestEventLabel ??
      stage.metadataGroups.find((group) => group.status === "available")?.summary ??
      stage.summary ??
      runtime.summary.currentStep,
    metadataAvailability,
    eventCount: stage.eventCount,
    sourceCount,
    warning
  };
}

function requestIntakeEvent(
  workstationState: WorkstationState
): CreativeTimelineEvent {
  const isActive = workstationState.currentRun.state === "streaming";
  const hasSession = workstationState.metadata.session.status === "available";

  return {
    id: "request_intake",
    label: "Request intake",
    status: isActive ? "active" : hasSession ? "complete" : "missing",
    summary: workstationState.status.detail,
    metadataAvailability: hasSession ? "Session metadata available" : "Session missing",
    eventCount: workstationState.currentRun.traceEventCount,
    sourceCount: hasSession ? 1 : 0,
    warning: hasSession ? null : "Session identifiers are missing."
  };
}

function missingEvent(definition: TimelineStageDefinition): CreativeTimelineEvent {
  return {
    id: definition.id,
    label: definition.label,
    status: "missing",
    summary: "No timeline metadata is available for this stage yet.",
    metadataAvailability: "0 metadata groups",
    eventCount: 0,
    sourceCount: 0,
    warning: "Metadata is missing for this stage."
  };
}

function timelineStatus(stage: WorkflowExplorerStage): CreativeTimelineEventStatus {
  switch (stage.status) {
    case "available":
      return "complete";
    case "running":
      return "active";
    case "partial":
      return "warning";
    case "error":
      return "error";
    default:
      return "missing";
  }
}

function stageWarning(stage: WorkflowExplorerStage, sourceCount: number): string | null {
  if (stage.status === "missing") {
    return "No metadata has been captured for this stage.";
  }

  if (stage.status === "partial") {
    return "Some metadata groups are still missing for this stage.";
  }

  if (sourceCount === 0) {
    return "No provenance sources are linked to this stage yet.";
  }

  return null;
}

function availableGroupCount(stage: WorkflowExplorerStage) {
  return stage.metadataGroups.filter((group) => group.status === "available").length;
}

function provenanceSourceCount(
  timelineEventId: string,
  provenance: ProvenanceEngineModel
) {
  switch (timelineEventId) {
    case "retrieval":
      return provenance.evidence_sources.filter(
        (source) => source.kind === "retrieval"
      ).length;
    case "artifact_intelligence":
      return provenance.artifact_sources.length + provenance.dependency_sources.length;
    case "creative_evaluation":
      return provenance.evaluation_sources.length;
    case "final_synthesis":
      return provenance.evaluation_sources.filter(
        (source) => source.kind === "final_payload"
      ).length;
    default:
      return provenance.evidence_sources.filter(
        (source) => source.kind === "reasoning"
      ).length;
  }
}
