import type { AssistantWorkspaceSnapshot, WorkflowNodeId } from "./assistant-client";
import type {
  WorkflowRuntimeModel,
  WorkflowRuntimeTraceEvent
} from "./workflow-runtime";
import { formatCode, truncate } from "./text-utils";
import type { WorkstationState } from "./workstation-state";

export type WorkflowExplorerStageId =
  | "planning"
  | "retrieval"
  | "creative_intelligence"
  | "generative_design"
  | "artifact_intelligence"
  | "creative_evaluation"
  | "final_response";

export type WorkflowExplorerAvailability =
  | "available"
  | "partial"
  | "missing"
  | "running"
  | "error";

export type WorkflowExplorerMetadataGroup = {
  id: string;
  label: string;
  status: WorkflowExplorerAvailability;
  summary: string;
  keys: string[];
};

export type WorkflowExplorerStage = {
  id: WorkflowExplorerStageId;
  label: string;
  status: WorkflowExplorerAvailability;
  summary: string;
  nodeIds: WorkflowNodeId[];
  eventCount: number;
  latestEventLabel: string | null;
  metadataGroups: WorkflowExplorerMetadataGroup[];
};

export type WorkflowExplorerModel = {
  state: "available" | "empty";
  stages: WorkflowExplorerStage[];
  summary: {
    availableStageCount: number;
    partialStageCount: number;
    missingStageCount: number;
    errorStageCount: number;
    metadataGroupCount: number;
    availableMetadataGroupCount: number;
  };
};

export type BuildWorkflowExplorerInput = {
  runtime: WorkflowRuntimeModel;
  snapshot: AssistantWorkspaceSnapshot;
  traceEvents: readonly WorkflowRuntimeTraceEvent[];
  workstationState: WorkstationState;
};

type StageDefinition = {
  id: WorkflowExplorerStageId;
  label: string;
  nodeIds: WorkflowNodeId[];
  groups: MetadataGroupDefinition[];
};

type MetadataGroupDefinition = {
  id: string;
  label: string;
  keys: string[];
  snapshotSummary?: (snapshot: AssistantWorkspaceSnapshot) => string | null;
};

const stageDefinitions: StageDefinition[] = [
  {
    id: "planning",
    label: "Planning",
    nodeIds: ["prompt_input", "planning"],
    groups: [
      {
        id: "creative_plan",
        label: "Creative plan",
        keys: ["creative_plan", "creativePlan"],
        snapshotSummary: (snapshot) =>
          snapshot.creativePlan?.generationStrategy ?? null
      },
      {
        id: "runtime_plan",
        label: "Runtime plan",
        keys: [
          "runtime_capabilities",
          "runtimeCapabilities",
          "runtime_compatibility",
          "runtimeCompatibility"
        ]
      },
      {
        id: "constraints",
        label: "Constraints",
        keys: [
          "creative_constraints",
          "creativeConstraints",
          "creative_constraint_priorities",
          "creativeConstraintPriorities"
        ]
      }
    ]
  },
  {
    id: "retrieval",
    label: "Retrieval",
    nodeIds: ["retrieval", "context_assembly"],
    groups: [
      {
        id: "retrieval_context",
        label: "Retrieval context",
        keys: ["retrieval", "retrieval_context", "retrievalContext"],
        snapshotSummary: (snapshot) =>
          snapshot.retrieval.sources.length > 0
            ? `${snapshot.retrieval.sources.length} sources available.`
            : null
      },
      {
        id: "assembled_context",
        label: "Assembled context",
        keys: ["context", "assembled_context", "assembledContext"]
      }
    ]
  },
  {
    id: "creative_intelligence",
    label: "Creative intelligence",
    nodeIds: ["planning", "director", "reasoning"],
    groups: [
      {
        id: "intent",
        label: "Intent and hierarchy",
        keys: [
          "creative_intent",
          "creativeIntent",
          "creative_hierarchy",
          "creativeHierarchy"
        ]
      },
      {
        id: "strategy",
        label: "Strategy and technique",
        keys: [
          "creative_strategy",
          "creativeStrategy",
          "creative_techniques",
          "creativeTechniques"
        ]
      },
      {
        id: "director_reasoning",
        label: "Director and reasoning",
        keys: [
          "creative_director",
          "creativeDirector",
          "creative_reasoning",
          "creativeReasoning"
        ]
      }
    ]
  },
  {
    id: "generative_design",
    label: "Generative design",
    nodeIds: ["planning", "reasoning"],
    groups: [
      {
        id: "structure",
        label: "Structure engines",
        keys: [
          "procedural_structure",
          "proceduralStructure",
          "generative_structure",
          "generativeStructure"
        ]
      },
      {
        id: "motif",
        label: "Motif and composition",
        keys: [
          "semantic_motif",
          "semanticMotif",
          "symbolic_narrative",
          "symbolicNarrative",
          "creative_composition",
          "creativeComposition"
        ]
      },
      {
        id: "cross_modality",
        label: "Cross-modality",
        keys: [
          "emotional_consistency",
          "emotionalConsistency",
          "cross_modality",
          "crossModality",
          "audio_visual_scene",
          "audioVisualScene"
        ]
      }
    ]
  },
  {
    id: "artifact_intelligence",
    label: "Artifact intelligence",
    nodeIds: [
      "artifact_extraction",
      "preview_preparation",
      "artifact_critique",
      "review",
      "refinement"
    ],
    groups: [
      {
        id: "artifact_plan",
        label: "Artifact plan",
        keys: [
          "artifact_plan",
          "artifactPlan",
          "artifact_dependency_graph",
          "artifactDependencyGraph",
          "multi_artifact_strategy",
          "multiArtifactStrategy"
        ],
        snapshotSummary: (snapshot) =>
          snapshot.artifacts.length > 0
            ? `${snapshot.artifacts.length} workspace artifacts.`
            : null
      },
      {
        id: "capability_matrix",
        label: "Capability matrix",
        keys: [
          "artifact_capability_matrix",
          "artifactCapabilityMatrix",
          "runtime_compatibility",
          "runtimeCompatibility"
        ]
      },
      {
        id: "critique_refinement",
        label: "Critique and refinement",
        keys: [
          "artifact_critic",
          "artifactCritic",
          "artifact_refiner",
          "artifactRefiner",
          "artifact_intelligence_synthesis",
          "artifactIntelligenceSynthesis",
          "artifact_merge_planner",
          "artifactMergePlanner",
          "artifact_export_intelligence",
          "artifactExportIntelligence"
        ]
      }
    ]
  },
  {
    id: "creative_evaluation",
    label: "Creative evaluation",
    nodeIds: ["artifact_critique", "review", "finalization"],
    groups: [
      {
        id: "self_evaluation",
        label: "Self evaluation",
        keys: ["self_evaluation", "selfEvaluation"]
      },
      {
        id: "evaluation_report",
        label: "Evaluation report",
        keys: [
          "evaluation_report",
          "evaluationReport",
          "evaluation",
          "ragas"
        ]
      },
      {
        id: "quality_signals",
        label: "Quality signals",
        keys: [
          "creative_quality_prediction",
          "creativeQualityPrediction",
          "creative_score",
          "creativeScore",
          "creative_confidence",
          "creativeConfidence",
          "consistency_validation",
          "consistencyValidation"
        ]
      }
    ]
  },
  {
    id: "final_response",
    label: "Final response",
    nodeIds: ["finalization"],
    groups: [
      {
        id: "answer",
        label: "Answer",
        keys: ["answer"],
        snapshotSummary: (snapshot) => latestAssistantMessage(snapshot)
      },
      {
        id: "session_intelligence",
        label: "Session intelligence",
        keys: ["session_intelligence", "sessionIntelligence"]
      },
      {
        id: "final_payload",
        label: "Final payload",
        keys: ["workflow", "artifacts", "generated_artifacts", "outputs"]
      }
    ]
  }
];

export function buildWorkflowExplorerModel({
  runtime,
  snapshot,
  traceEvents,
  workstationState
}: BuildWorkflowExplorerInput): WorkflowExplorerModel {
  const stages = stageDefinitions.map((definition) =>
    buildExplorerStage({
      definition,
      runtime,
      snapshot,
      traceEvents,
      workstationState
    })
  );
  const metadataGroups = stages.flatMap((stage) => stage.metadataGroups);
  const availableStageCount = stages.filter(
    (stage) => stage.status === "available" || stage.status === "running"
  ).length;
  const partialStageCount = stages.filter(
    (stage) => stage.status === "partial"
  ).length;
  const missingStageCount = stages.filter(
    (stage) => stage.status === "missing"
  ).length;
  const errorStageCount = stages.filter((stage) => stage.status === "error").length;
  const availableMetadataGroupCount = metadataGroups.filter(
    (group) => group.status === "available"
  ).length;

  return {
    state: availableMetadataGroupCount > 0 ? "available" : "empty",
    stages,
    summary: {
      availableStageCount,
      partialStageCount,
      missingStageCount,
      errorStageCount,
      metadataGroupCount: metadataGroups.length,
      availableMetadataGroupCount
    }
  };
}

function buildExplorerStage({
  definition,
  runtime,
  snapshot,
  traceEvents,
  workstationState
}: {
  definition: StageDefinition;
  runtime: WorkflowRuntimeModel;
  snapshot: AssistantWorkspaceSnapshot;
  traceEvents: readonly WorkflowRuntimeTraceEvent[];
  workstationState: WorkstationState;
}): WorkflowExplorerStage {
  const relevantEvents = runtime.events.filter(
    (event) => event.nodeId && definition.nodeIds.includes(event.nodeId)
  );
  const metadataGroups = definition.groups.map((group) =>
    buildMetadataGroup(group, snapshot, traceEvents)
  );
  const availableGroupCount = metadataGroups.filter(
    (group) => group.status === "available"
  ).length;
  const hasError = relevantEvents.some((event) => event.eventType === "error");
  const isActive = definition.nodeIds.includes(
    workstationState.selection.activeWorkflowNodeId
  );
  const status = stageStatus({
    availableGroupCount,
    hasError,
    isActive,
    runtimeStatus: runtime.summary.status,
    totalGroupCount: metadataGroups.length
  });

  return {
    id: definition.id,
    label: definition.label,
    status,
    summary: stageSummary(availableGroupCount, metadataGroups.length, status),
    nodeIds: definition.nodeIds,
    eventCount: relevantEvents.length,
    latestEventLabel: relevantEvents[relevantEvents.length - 1]?.label ?? null,
    metadataGroups
  };
}

function buildMetadataGroup(
  definition: MetadataGroupDefinition,
  snapshot: AssistantWorkspaceSnapshot,
  traceEvents: readonly WorkflowRuntimeTraceEvent[]
): WorkflowExplorerMetadataGroup {
  const snapshotSummary = definition.snapshotSummary?.(snapshot) ?? null;
  const streamSummary = findStreamMetadataSummary(definition.keys, traceEvents);
  const summary = snapshotSummary ?? streamSummary;

  return {
    id: definition.id,
    label: definition.label,
    status: summary ? "available" : "missing",
    summary: summary ?? "Metadata has not been captured for this group yet.",
    keys: definition.keys
  };
}

function findStreamMetadataSummary(
  keys: string[],
  traceEvents: readonly WorkflowRuntimeTraceEvent[]
): string | null {
  for (let index = traceEvents.length - 1; index >= 0; index -= 1) {
    const payload = traceEvents[index].event.payload;
    const workflow = readRecord(payload.workflow);

    for (const key of keys) {
      const value = payload[key] ?? workflow?.[key];
      const summary = summarizeMetadataValue(value);
      if (summary) {
        return summary;
      }
    }
  }

  return null;
}

function summarizeMetadataValue(value: unknown): string | null {
  if (typeof value === "string" && value.trim()) {
    return truncate(value.trim());
  }

  if (Array.isArray(value)) {
    return `${value.length} item${value.length === 1 ? "" : "s"} captured.`;
  }

  const record = readRecord(value);
  if (!record) {
    return null;
  }

  const preferred =
    readString(record.summary) ??
    readString(record.rationale) ??
    readString(record.generation_strategy) ??
    readString(record.generationStrategy) ??
    readString(record.evaluation_summary) ??
    readString(record.evaluationSummary);
  if (preferred) {
    return truncate(preferred);
  }

  const role = readString(record.role);
  if (role) {
    return `${formatCode(role)} metadata captured.`;
  }

  const keyCount = Object.keys(record).length;
  return keyCount > 0
    ? `${keyCount} metadata field${keyCount === 1 ? "" : "s"} captured.`
    : null;
}

function stageStatus({
  availableGroupCount,
  hasError,
  isActive,
  runtimeStatus,
  totalGroupCount
}: {
  availableGroupCount: number;
  hasError: boolean;
  isActive: boolean;
  runtimeStatus: string;
  totalGroupCount: number;
}): WorkflowExplorerAvailability {
  if (hasError) {
    return "error";
  }

  if (isActive && runtimeStatus !== "completed" && runtimeStatus !== "failed") {
    return "running";
  }

  if (availableGroupCount === 0) {
    return "missing";
  }

  return availableGroupCount === totalGroupCount ? "available" : "partial";
}

function stageSummary(
  availableGroupCount: number,
  totalGroupCount: number,
  status: WorkflowExplorerAvailability
) {
  if (status === "running") {
    return `${availableGroupCount}/${totalGroupCount} metadata groups available while this stage is active.`;
  }

  if (status === "missing") {
    return "No metadata groups are available for this stage yet.";
  }

  return `${availableGroupCount}/${totalGroupCount} metadata groups available.`;
}

function latestAssistantMessage(snapshot: AssistantWorkspaceSnapshot): string | null {
  const message = [...snapshot.messages]
    .reverse()
    .find((entry) => entry.role === "assistant");

  return message ? truncate(message.content) : null;
}

function readRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function readString(value: unknown): string | null {
  return typeof value === "string" && value.trim().length > 0
    ? value.trim()
    : null;
}
