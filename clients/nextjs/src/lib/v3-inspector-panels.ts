import {
  readWorkflowMetadata,
  type AssistantStreamWorkflowMetadata
} from "./assistant-stream";
import type { ProvenanceEngineModel, ProvenanceSource } from "./provenance-engine";
import { formatCode, truncate } from "./text-utils";
import type { WorkflowRuntimeTraceEvent } from "./workflow-runtime";
import type { WorkstationState } from "./workstation-state";

export type V3InspectorPanelStatus = "available" | "partial" | "missing";

export type V3InspectorPanelItem = {
  id: string;
  label: string;
  status: V3InspectorPanelStatus;
  summary: string;
  details: string[];
  eventSequence: number | null;
  source: "hydrated" | "partial" | "missing" | "provenance";
};

export type V3InspectorPanel = {
  id: string;
  label: string;
  status: V3InspectorPanelStatus;
  summary: string;
  availableItemCount: number;
  partialItemCount: number;
  missingItemCount: number;
  items: V3InspectorPanelItem[];
};

export type V3InspectorPanelsModel = {
  state: "available" | "empty";
  panels: V3InspectorPanel[];
  summary: {
    panelCount: number;
    availablePanelCount: number;
    partialPanelCount: number;
    missingPanelCount: number;
    availableItemCount: number;
    partialItemCount: number;
    missingItemCount: number;
    currentRunState: WorkstationState["currentRun"]["state"];
  };
};

export type BuildV3InspectorPanelsInput = {
  provenance: ProvenanceEngineModel;
  traceEvents: readonly WorkflowRuntimeTraceEvent[];
  workstationState: WorkstationState;
};

type WorkflowMetadataKey = keyof AssistantStreamWorkflowMetadata;

type MetadataDefinition = {
  id: string;
  label: string;
  metadataKey: WorkflowMetadataKey;
  availableKey?: WorkflowMetadataKey;
  keys: string[];
  summaryKeys: string[];
  detailKeys: string[];
};

type PanelDefinition = {
  id: string;
  label: string;
  items: MetadataDefinition[];
};

const metadataAvailabilityFlags: Partial<Record<string, WorkflowMetadataKey>> = {
  artifact_capability_matrix: "artifact_capability_matrix_available",
  artifact_critic: "artifact_critic_available",
  artifact_dependency_graph: "artifact_dependency_graph_available",
  artifact_export_intelligence: "artifact_export_intelligence_available",
  artifact_intelligence_synthesis: "artifact_intelligence_synthesis_available",
  artifact_merge_planner: "artifact_merge_planner_available",
  artifact_plan: "artifact_planner_available",
  artifact_refiner: "artifact_refiner_available",
  audio_visual_scene: "audio_visual_scene_available",
  consistency_validation: "consistency_validation_available",
  creative_composition: "creative_composition_available",
  creative_confidence: "creative_confidence_available",
  creative_critic: "creative_critic_available",
  creative_director: "director_available",
  creative_hierarchy: "hierarchy_planner_available",
  creative_improvement_planner: "creative_improvement_planner_available",
  creative_intent: "intent_decomposer_available",
  creative_quality_prediction: "quality_predictor_available",
  creative_reasoning: "creative_reasoning_available",
  creative_score: "creative_score_available",
  creative_strategy: "strategy_available",
  creative_techniques: "technique_selector_available",
  creative_tradeoffs: "tradeoff_explorer_available",
  cross_modality: "cross_modality_available",
  emotional_consistency: "emotional_consistency_available",
  evaluation_report: "evaluation_report_available",
  generative_structure: "generative_structure_available",
  multi_artifact_strategy: "multi_artifact_strategy_available",
  procedural_structure: "procedural_structure_available",
  reflection_loop: "reflection_loop_available",
  self_evaluation: "self_evaluation_available",
  semantic_motif: "semantic_motif_available",
  symbolic_narrative: "symbolic_narrative_available"
};

const defaultSummaryKeys = [
  "summary",
  "rationale",
  "normalizedIntent",
  "primaryExpression",
  "recommendedCreativeDirection",
  "generationStrategy",
  "confidenceSummary",
  "scoreSummary",
  "executiveSummary",
  "qualitySummary",
  "evaluationSummary",
  "critiqueSummary",
  "reflectionSummary",
  "improvementSummary",
  "synthesisSummary",
  "artifactStrategySummary",
  "primaryArtifactIntent",
  "spatialStructurePlan",
  "temporalStructurePlan",
  "blueprintName",
  "motifSystemName",
  "symbolicArc",
  "outputGoal"
];

const panelDefinitions: PanelDefinition[] = [
  {
    id: "creative_intelligence",
    label: "Creative Intelligence",
    items: [
      metadataDefinition("creative_intent", "Intent", "creative_intent", [
        "normalizedIntent",
        "primaryExpression",
        "experientialGoal"
      ]),
      metadataDefinition(
        "creative_hierarchy",
        "Hierarchy",
        "creative_hierarchy",
        ["priorityRationale", "hierarchyConfidence", "primaryCreativePriorities"]
      ),
      metadataDefinition("creative_strategy", "Strategy", "creative_strategy", [
        "rationale",
        "primaryStrategy",
        "creativeGoals"
      ]),
      metadataDefinition("creative_techniques", "Techniques", "creative_techniques", [
        "rationale",
        "primaryTechnique",
        "artisticSuitability"
      ]),
      metadataDefinition("creative_director", "Director", "creative_director", [
        "directorSummary",
        "recommendation",
        "rationale",
        "promptGuidance"
      ]),
      metadataDefinition("creative_reasoning", "Reasoning", "creative_reasoning", [
        "recommendedCreativeDirection",
        "strongestSupportingSignals",
        "implementationGuidance"
      ]),
      metadataDefinition("creative_tradeoffs", "Tradeoffs", "creative_tradeoffs", [
        "outputGoal",
        "primaryTradeoffs",
        "creativeBenefits"
      ]),
      metadataDefinition(
        "creative_quality_prediction",
        "Quality Prediction",
        "creative_quality_prediction",
        ["predictedQualityLevel", "qualityRisks", "suggestedImprovements"]
      )
    ]
  },
  {
    id: "generative_design",
    label: "Generative Design",
    items: [
      metadataDefinition(
        "procedural_structure",
        "Procedural Structure",
        "procedural_structure",
        ["spatialStructurePlan", "temporalStructurePlan", "primaryStructure"]
      ),
      metadataDefinition(
        "generative_structure",
        "Generative Structure",
        "generative_structure",
        ["blueprintName", "spatialEvolution", "temporalEvolution"]
      ),
      metadataDefinition("semantic_motif", "Semantic Motif", "semantic_motif", [
        "motifSystemName",
        "motifHierarchy",
        "motifRecurrencePlan"
      ]),
      metadataDefinition(
        "symbolic_narrative",
        "Symbolic Narrative",
        "symbolic_narrative",
        ["symbolicArc", "experientialGoal", "narrativeArchetype"]
      ),
      metadataDefinition(
        "creative_composition",
        "Composition",
        "creative_composition",
        ["spatialOrganization", "densityPlan", "rhythmPlan"]
      ),
      metadataDefinition(
        "emotional_consistency",
        "Emotional Consistency",
        "emotional_consistency",
        ["emotionalArc", "primaryEmotionalTone", "emotionalCoherenceScore"]
      ),
      metadataDefinition("cross_modality", "Cross-Modality", "cross_modality", [
        "compositionPattern",
        "primaryModality",
        "fallbackStrategy"
      ]),
      metadataDefinition("audio_visual_scene", "Audio-Visual Scene", "audio_visual_scene", [
        "sceneSummary",
        "scenePhases",
        "timingGuidance"
      ])
    ]
  },
  {
    id: "artifact_intelligence",
    label: "Artifact Intelligence",
    items: [
      metadataDefinition("artifact_plan", "Artifact Plan", "artifact_plan", [
        "primaryArtifactIntent",
        "artifactType",
        "expectedOutputStructure"
      ]),
      metadataDefinition(
        "artifact_dependency_graph",
        "Dependency Graph",
        "artifact_dependency_graph",
        ["primaryArtifactNodeId", "blockingDependencies", "runtimeFacingDependencies"]
      ),
      metadataDefinition(
        "artifact_capability_matrix",
        "Capability Matrix",
        "artifact_capability_matrix",
        ["strongestTargets", "artifactFit", "capabilityRisks"]
      ),
      metadataDefinition(
        "multi_artifact_strategy",
        "Multi-Artifact Strategy",
        "multi_artifact_strategy",
        ["artifactStrategySummary", "combinationMode", "riskAreas"]
      ),
      metadataDefinition("artifact_critic", "Artifact Critic", "artifact_critic", [
        "critiqueSummary",
        "riskAssessment",
        "strengths"
      ]),
      metadataDefinition("artifact_refiner", "Artifact Refiner", "artifact_refiner", [
        "refinementSummary",
        "recommendedImprovements",
        "priorityImprovements"
      ]),
      metadataDefinition(
        "artifact_intelligence_synthesis",
        "Intelligence Synthesis",
        "artifact_intelligence_synthesis",
        ["synthesisSummary", "recommendedStrategySummary", "implementationReadiness"]
      ),
      metadataDefinition(
        "artifact_merge_planner",
        "Merge Planner",
        "artifact_merge_planner",
        ["mergeSummary", "mergeStrategy", "recommendedMergePath"]
      ),
      metadataDefinition(
        "artifact_export_intelligence",
        "Export Intelligence",
        "artifact_export_intelligence",
        ["exportSummary", "exportReadiness", "preferredExportTarget"]
      )
    ]
  },
  {
    id: "creative_evaluation",
    label: "Creative Evaluation",
    items: [
      metadataDefinition("creative_critic", "Creative Critic", "creative_critic", [
        "critiqueSummary",
        "riskAssessment",
        "creativeStrengths"
      ]),
      metadataDefinition("self_evaluation", "Self Evaluation", "self_evaluation", [
        "evaluationSummary",
        "completenessAssessment",
        "qualityGaps"
      ]),
      metadataDefinition(
        "consistency_validation",
        "Consistency Validation",
        "consistency_validation",
        ["consistencySummary", "consistencyStatus", "detectedConflicts"]
      ),
      metadataDefinition("evaluation_report", "Evaluation Report", "evaluation_report", [
        "executiveSummary",
        "qualitySummary",
        "recommendations"
      ])
    ]
  },
  {
    id: "confidence",
    label: "Confidence",
    items: [
      metadataDefinition("creative_confidence", "Creative Confidence", "creative_confidence", [
        "confidenceSummary",
        "confidenceLevel",
        "confidenceScore"
      ]),
      metadataDefinition("creative_score", "Creative Score", "creative_score", [
        "scoreSummary",
        "overallCreativeScore",
        "scoreBand"
      ])
    ]
  },
  {
    id: "reflection",
    label: "Reflection",
    items: [
      metadataDefinition("reflection_loop", "Reflection Loop", "reflection_loop", [
        "reflectionSummary",
        "reflectionPriority",
        "expectedQualityGain"
      ])
    ]
  },
  {
    id: "improvement_plan",
    label: "Improvement Plan",
    items: [
      metadataDefinition(
        "creative_improvement_planner",
        "Improvement Planner",
        "creative_improvement_planner",
        ["improvementSummary", "improvementPriorities", "highestImpactOpportunities"]
      )
    ]
  },
  {
    id: "evaluation_trace",
    label: "Evaluation Trace",
    items: [
      metadataDefinition("evaluation_trace", "Evaluation Trace", "evaluation_report", [
        "executiveSummary",
        "evaluationTrace",
        "evaluationProvenance"
      ])
    ]
  }
];

export function buildV3InspectorPanelsModel({
  provenance,
  traceEvents,
  workstationState
}: BuildV3InspectorPanelsInput): V3InspectorPanelsModel {
  const panels = [
    ...panelDefinitions.map((definition) =>
      buildMetadataPanel(definition, traceEvents)
    ),
    buildProvenancePanel(provenance)
  ];
  const items = panels.flatMap((panel) => panel.items);
  const availablePanelCount = panels.filter(
    (panel) => panel.status === "available"
  ).length;
  const partialPanelCount = panels.filter(
    (panel) => panel.status === "partial"
  ).length;
  const missingPanelCount = panels.filter(
    (panel) => panel.status === "missing"
  ).length;
  const availableItemCount = items.filter(
    (item) => item.status === "available"
  ).length;
  const partialItemCount = items.filter((item) => item.status === "partial").length;
  const missingItemCount = items.filter((item) => item.status === "missing").length;

  return {
    state:
      availableItemCount > 0 || partialItemCount > 0 ? "available" : "empty",
    panels,
    summary: {
      panelCount: panels.length,
      availablePanelCount,
      partialPanelCount,
      missingPanelCount,
      availableItemCount,
      partialItemCount,
      missingItemCount,
      currentRunState: workstationState.currentRun.state
    }
  };
}

function metadataDefinition(
  id: string,
  label: string,
  metadataKey: WorkflowMetadataKey,
  summaryKeys: string[]
): MetadataDefinition {
  return {
    id,
    label,
    metadataKey,
    availableKey: metadataAvailabilityFlags[String(metadataKey)],
    keys: [String(metadataKey), toCamelCase(String(metadataKey))],
    summaryKeys: [...summaryKeys, ...defaultSummaryKeys],
    detailKeys: [
      ...summaryKeys,
      "strengths",
      "weaknesses",
      "risks",
      "missingInformation",
      "hitlQuestions",
      "promptGuidance",
      "evidence",
      "authorityBoundary"
    ]
  };
}

function buildMetadataPanel(
  definition: PanelDefinition,
  traceEvents: readonly WorkflowRuntimeTraceEvent[]
): V3InspectorPanel {
  const items = definition.items.map((item) => metadataItem(item, traceEvents));

  return buildPanel({
    id: definition.id,
    items,
    label: definition.label,
    missingSummary: `No ${definition.label.toLowerCase()} metadata has been captured yet.`
  });
}

function metadataItem(
  definition: MetadataDefinition,
  traceEvents: readonly WorkflowRuntimeTraceEvent[]
): V3InspectorPanelItem {
  for (let index = traceEvents.length - 1; index >= 0; index -= 1) {
    const traceEvent = traceEvents[index];
    const hydrated = readWorkflowMetadata(traceEvent.event);
    const hydratedValue = hydrated?.[definition.metadataKey] ?? null;
    if (hasMetadataValue(hydratedValue)) {
      return metadataItemFromValue({
        definition,
        eventSequence: traceEvent.event.sequence,
        source: "hydrated",
        status: "available",
        value: hydratedValue
      });
    }

    const rawValue = findRawMetadataValue(traceEvent, definition.keys);
    if (hasMetadataValue(rawValue)) {
      return metadataItemFromValue({
        definition,
        eventSequence: traceEvent.event.sequence,
        source: "partial",
        status: "partial",
        value: rawValue
      });
    }

    if (
      definition.availableKey &&
      hydrated?.[definition.availableKey] === true
    ) {
      return {
        id: definition.id,
        label: definition.label,
        status: "partial",
        summary: "Metadata was signaled but could not be hydrated.",
        details: [],
        eventSequence: traceEvent.event.sequence,
        source: "partial"
      };
    }
  }

  return {
    id: definition.id,
    label: definition.label,
    status: "missing",
    summary: `${definition.label} metadata has not been captured yet.`,
    details: [],
    eventSequence: null,
    source: "missing"
  };
}

function metadataItemFromValue({
  definition,
  eventSequence,
  source,
  status,
  value
}: {
  definition: MetadataDefinition;
  eventSequence: number;
  source: V3InspectorPanelItem["source"];
  status: V3InspectorPanelStatus;
  value: unknown;
}): V3InspectorPanelItem {
  return {
    id: definition.id,
    label: definition.label,
    status,
    summary: summarizeMetadataValue(value, definition.summaryKeys),
    details: summarizeMetadataDetails(value, definition.detailKeys),
    eventSequence,
    source
  };
}

function buildProvenancePanel(provenance: ProvenanceEngineModel): V3InspectorPanel {
  const items: V3InspectorPanelItem[] = [
    provenanceItem("evidence", "Evidence", "evidence", provenance.evidence_sources),
    provenanceItem(
      "dependency",
      "Dependencies",
      "dependency",
      provenance.dependency_sources
    ),
    provenanceItem("artifact", "Artifacts", "artifact", provenance.artifact_sources),
    provenanceItem(
      "evaluation",
      "Evaluation",
      "evaluation",
      provenance.evaluation_sources
    ),
    missingProvenanceItem(provenance.unsupported_or_missing_sources)
  ];

  return buildPanel({
    id: "provenance",
    items,
    label: "Provenance",
    missingSummary: "No provenance sources have been captured yet.",
    summaryOverride: provenance.provenance_summary
  });
}

function provenanceItem(
  id: string,
  label: string,
  sourceLabel: string,
  sources: ProvenanceSource[]
): V3InspectorPanelItem {
  return {
    id,
    label,
    status: sources.length > 0 ? "available" : "missing",
    summary:
      sources.length > 0
        ? `${sources.length} ${sourceLabel} source${
            sources.length === 1 ? "" : "s"
          }`
        : `${label} sources are missing.`,
    details: sources.slice(0, 3).map((source) => `${source.label}: ${source.summary}`),
    eventSequence: sources.find((source) => source.eventSequence !== null)
      ?.eventSequence ?? null,
    source: "provenance"
  };
}

function missingProvenanceItem(
  sources: ProvenanceSource[]
): V3InspectorPanelItem {
  return {
    id: "missing",
    label: "Missing Groups",
    status: sources.length > 0 ? "missing" : "available",
    summary:
      sources.length > 0
        ? `${sources.length} provenance group${sources.length === 1 ? "" : "s"} missing`
        : "No provenance groups are missing.",
    details: sources.slice(0, 3).map((source) => `${source.label}: ${source.summary}`),
    eventSequence: null,
    source: "provenance"
  };
}

function buildPanel({
  id,
  items,
  label,
  missingSummary,
  summaryOverride
}: {
  id: string;
  items: V3InspectorPanelItem[];
  label: string;
  missingSummary: string;
  summaryOverride?: string;
}): V3InspectorPanel {
  const availableItemCount = items.filter(
    (item) => item.status === "available"
  ).length;
  const partialItemCount = items.filter((item) => item.status === "partial").length;
  const missingItemCount = items.filter((item) => item.status === "missing").length;
  const status =
    availableItemCount === items.length
      ? "available"
      : availableItemCount > 0 || partialItemCount > 0
        ? "partial"
        : "missing";
  const leadingItem = items.find((item) => item.status !== "missing");

  return {
    id,
    label,
    status,
    summary:
      summaryOverride ??
      (leadingItem
        ? `${leadingItem.label}: ${leadingItem.summary}`
        : missingSummary),
    availableItemCount,
    partialItemCount,
    missingItemCount,
    items
  };
}

function findRawMetadataValue(
  traceEvent: WorkflowRuntimeTraceEvent,
  keys: string[]
): unknown {
  const payload = traceEvent.event.payload;
  const workflow = readRecord(payload.workflow);

  for (const key of keys) {
    if (hasMetadataValue(payload[key])) {
      return payload[key];
    }
    if (workflow && hasMetadataValue(workflow[key])) {
      return workflow[key];
    }
  }

  return null;
}

function summarizeMetadataValue(value: unknown, preferredKeys: string[]) {
  const record = readRecord(value);
  if (record) {
    for (const key of preferredKeys) {
      const summary = summarizeValue(record[key] ?? record[toSnakeCase(key)]);
      if (summary) {
        return truncate(summary, 150);
      }
    }

    const role = readString(record.role);
    if (role) {
      return `${formatCode(role, { splitCamelCase: true })} metadata captured.`;
    }

    const keyCount = Object.keys(record).length;
    if (keyCount > 0) {
      return `${keyCount} metadata field${keyCount === 1 ? "" : "s"} captured.`;
    }
  }

  return summarizeValue(value) ?? "Metadata captured.";
}

function summarizeMetadataDetails(value: unknown, keys: string[]) {
  const record = readRecord(value);
  if (!record) {
    return [];
  }

  const details: string[] = [];
  for (const key of keys) {
    const detail = summarizeDetail(key, record[key] ?? record[toSnakeCase(key)]);
    if (detail && !details.includes(detail)) {
      details.push(truncate(detail, 120));
    }
    if (details.length >= 3) {
      break;
    }
  }

  return details;
}

function summarizeDetail(key: string, value: unknown): string | null {
  const summary = summarizeValue(value);
  return summary ? `${formatCode(key, { splitCamelCase: true })}: ${summary}` : null;
}

function summarizeValue(value: unknown): string | null {
  if (typeof value === "string" && value.trim()) {
    return value.trim();
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return Number.isInteger(value) ? String(value) : value.toFixed(2);
  }
  if (typeof value === "boolean") {
    return value ? "Yes" : "No";
  }
  if (Array.isArray(value)) {
    return summarizeArray(value);
  }

  const record = readRecord(value);
  if (!record) {
    return null;
  }

  for (const key of [
    "summary",
    "rationale",
    "contribution",
    "claim",
    "note",
    "label",
    "title",
    "name",
    "primaryArtifactIntent",
    "recommendedCreativeDirection"
  ]) {
    const summary = summarizeValue(record[key]);
    if (summary) {
      return summary;
    }
  }

  return null;
}

function summarizeArray(value: unknown[]) {
  if (value.length === 0) {
    return null;
  }

  const stringValues = value.filter(
    (item): item is string => typeof item === "string" && item.trim().length > 0
  );
  if (stringValues.length > 0) {
    return stringValues.slice(0, 2).join(" / ");
  }

  const nestedSummaries = value.flatMap((item) => {
    const summary = summarizeValue(item);
    return summary ? [summary] : [];
  });
  if (nestedSummaries.length > 0) {
    return nestedSummaries.slice(0, 2).join(" / ");
  }

  return `${value.length} item${value.length === 1 ? "" : "s"} captured.`;
}

function hasMetadataValue(value: unknown) {
  if (value === null || value === undefined) {
    return false;
  }
  if (typeof value === "string") {
    return value.trim().length > 0;
  }
  if (Array.isArray(value)) {
    return value.length > 0;
  }
  if (typeof value === "object") {
    return Object.keys(value).length > 0;
  }
  return true;
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

function toCamelCase(value: string) {
  return value.replace(/_([a-z])/g, (_, letter: string) => letter.toUpperCase());
}

function toSnakeCase(value: string) {
  return value.replace(/[A-Z]/g, (letter) => `_${letter.toLowerCase()}`);
}
