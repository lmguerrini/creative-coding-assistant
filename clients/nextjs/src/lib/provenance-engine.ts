import type {
  ArtifactSummary,
  AssistantWorkspaceSnapshot,
  RetrievalSourceSummary
} from "./assistant-client";
import type { WorkflowRuntimeTraceEvent } from "./workflow-runtime";
import type { WorkstationState } from "./workstation-state";

export type ProvenanceSourceStatus =
  | "available"
  | "partial"
  | "missing"
  | "unsupported";

export type ProvenanceSourceKind =
  | "retrieval"
  | "reasoning"
  | "dependency"
  | "artifact"
  | "evaluation"
  | "final_payload"
  | "missing";

export type ProvenanceSource = {
  id: string;
  kind: ProvenanceSourceKind;
  label: string;
  status: ProvenanceSourceStatus;
  summary: string;
  sourceKeys: string[];
  eventSequence: number | null;
};

export type ProvenanceEngineModel = {
  provenance_summary: string;
  evidence_sources: ProvenanceSource[];
  dependency_sources: ProvenanceSource[];
  artifact_sources: ProvenanceSource[];
  evaluation_sources: ProvenanceSource[];
  unsupported_or_missing_sources: ProvenanceSource[];
  totals: {
    availableSourceCount: number;
    missingSourceCount: number;
    partialSourceCount: number;
    unsupportedSourceCount: number;
  };
};

export type BuildProvenanceEngineInput = {
  snapshot: AssistantWorkspaceSnapshot;
  traceEvents: readonly WorkflowRuntimeTraceEvent[];
  workstationState: WorkstationState;
};

type PayloadSourceDefinition = {
  id: string;
  kind: ProvenanceSourceKind;
  label: string;
  keys: string[];
};

const evidenceDefinitions: PayloadSourceDefinition[] = [
  {
    id: "reasoning_evidence",
    kind: "reasoning",
    label: "Reasoning evidence",
    keys: [
      "evidence",
      "evidence_chain",
      "evidenceChain",
      "strongest_supporting_signals",
      "strongestSupportingSignals"
    ]
  },
  {
    id: "planning_evidence",
    kind: "reasoning",
    label: "Planning evidence",
    keys: ["creative_plan", "creativePlan", "creative_reasoning", "creativeReasoning"]
  }
];

const dependencyDefinitions: PayloadSourceDefinition[] = [
  {
    id: "artifact_dependency_graph",
    kind: "dependency",
    label: "Artifact dependency graph",
    keys: [
      "artifact_dependency_graph",
      "artifactDependencyGraph",
      "dependency_graph",
      "dependencyGraph"
    ]
  },
  {
    id: "evaluation_dependencies",
    kind: "dependency",
    label: "Evaluation dependencies",
    keys: ["evaluation_dependencies", "evaluationDependencies", "dependencies"]
  }
];

const evaluationDefinitions: PayloadSourceDefinition[] = [
  {
    id: "self_evaluation",
    kind: "evaluation",
    label: "Self evaluation",
    keys: ["self_evaluation", "selfEvaluation"]
  },
  {
    id: "evaluation_report",
    kind: "evaluation",
    label: "Evaluation report",
    keys: ["evaluation_report", "evaluationReport"]
  },
  {
    id: "live_evaluation",
    kind: "evaluation",
    label: "Live evaluation",
    keys: ["evaluation", "ragas", "result"]
  }
];

const finalPayloadDefinitions: PayloadSourceDefinition[] = [
  {
    id: "final_payload",
    kind: "final_payload",
    label: "Final payload",
    keys: ["answer", "workflow", "artifacts", "generated_artifacts", "outputs"]
  }
];

export function buildProvenanceEngineModel({
  snapshot,
  traceEvents,
  workstationState
}: BuildProvenanceEngineInput): ProvenanceEngineModel {
  const evidenceSources = [
    ...snapshot.retrieval.sources.map((source) => retrievalEvidenceSource(source)),
    ...payloadSources(evidenceDefinitions, traceEvents)
  ];
  const dependencySources = payloadSources(dependencyDefinitions, traceEvents);
  const artifactSources = [
    ...snapshot.artifacts.map((artifact) => artifactProvenanceSource(artifact)),
    ...payloadArtifactSources(traceEvents)
  ];
  const evaluationSources = payloadSources(evaluationDefinitions, traceEvents, {
    eventType: "eval_update"
  });
  const finalPayloadSources = payloadSources(finalPayloadDefinitions, traceEvents, {
    eventType: "final"
  });
  const unsupportedOrMissingSources = missingSources({
    dependencySources,
    evaluationSources,
    evidenceSources,
    finalPayloadSources,
    snapshot,
    workstationState
  });
  const availableSources = [
    ...evidenceSources,
    ...dependencySources,
    ...artifactSources,
    ...evaluationSources,
    ...finalPayloadSources
  ].filter((source) => source.status === "available");
  const partialSources = [
    ...evidenceSources,
    ...dependencySources,
    ...artifactSources,
    ...evaluationSources,
    ...finalPayloadSources
  ].filter((source) => source.status === "partial");

  return {
    provenance_summary: provenanceSummary({
      artifactSources,
      availableSources,
      dependencySources,
      evaluationSources,
      evidenceSources,
      finalPayloadSources,
      unsupportedOrMissingSources
    }),
    evidence_sources: evidenceSources,
    dependency_sources: dependencySources,
    artifact_sources: artifactSources,
    evaluation_sources: [...evaluationSources, ...finalPayloadSources],
    unsupported_or_missing_sources: unsupportedOrMissingSources,
    totals: {
      availableSourceCount: availableSources.length,
      missingSourceCount: unsupportedOrMissingSources.filter(
        (source) => source.status === "missing"
      ).length,
      partialSourceCount: partialSources.length,
      unsupportedSourceCount: unsupportedOrMissingSources.filter(
        (source) => source.status === "unsupported"
      ).length
    }
  };
}

function retrievalEvidenceSource(source: RetrievalSourceSummary): ProvenanceSource {
  return {
    id: `retrieval:${source.sourceId}`,
    kind: "retrieval",
    label: source.title,
    status: source.chunks.length > 0 ? "available" : "partial",
    summary:
      source.chunks.length > 0
        ? `${source.domainLabel} source with ${source.chunks.length} retrieved chunks.`
        : `${source.domainLabel} source was listed without retrieved chunks.`,
    sourceKeys: ["retrieval.sources", source.sourceId],
    eventSequence: null
  };
}

function artifactProvenanceSource(artifact: ArtifactSummary): ProvenanceSource {
  const summaryParts = [
    artifact.language,
    artifact.runtime,
    artifact.rendererId,
    artifact.qualityScore != null
      ? `${Math.round(artifact.qualityScore * 100)}% quality`
      : null
  ].filter((part): part is string => Boolean(part));

  return {
    id: `artifact:${artifact.id}`,
    kind: "artifact",
    label: artifact.title,
    status: "available",
    summary: summaryParts.length > 0 ? summaryParts.join(" / ") : artifact.summary,
    sourceKeys: ["artifacts", artifact.id],
    eventSequence: null
  };
}

function payloadSources(
  definitions: PayloadSourceDefinition[],
  traceEvents: readonly WorkflowRuntimeTraceEvent[],
  options: { eventType?: string } = {}
): ProvenanceSource[] {
  const sources: ProvenanceSource[] = [];

  for (const definition of definitions) {
    const source = payloadSource(definition, traceEvents, options);
    if (source) {
      sources.push(source);
    }
  }

  return sources;
}

function payloadSource(
  definition: PayloadSourceDefinition,
  traceEvents: readonly WorkflowRuntimeTraceEvent[],
  options: { eventType?: string }
): ProvenanceSource | null {
  for (let index = traceEvents.length - 1; index >= 0; index -= 1) {
    const traceEvent = traceEvents[index];
    if (options.eventType && traceEvent.event.event_type !== options.eventType) {
      continue;
    }

    const payload = traceEvent.event.payload;
    const workflow = readRecord(payload.workflow);
    for (const key of definition.keys) {
      const value = payload[key] ?? workflow?.[key];
      const summary = summarizePayloadValue(value);
      if (summary) {
        return {
          id: `${definition.id}:${traceEvent.event.sequence}`,
          kind: definition.kind,
          label: definition.label,
          status: "available",
          summary,
          sourceKeys: [key],
          eventSequence: traceEvent.event.sequence
        };
      }
    }
  }

  return null;
}

function payloadArtifactSources(
  traceEvents: readonly WorkflowRuntimeTraceEvent[]
): ProvenanceSource[] {
  const sources: ProvenanceSource[] = [];

  for (const traceEvent of traceEvents) {
    const payload = traceEvent.event.payload;
    const artifactRecords = [
      ...readRecordList(payload.artifacts),
      ...readRecordList(payload.generated_artifacts),
      ...readRecordList(payload.generatedArtifacts),
      ...readRecordList(payload.outputs),
      ...readRecordList(payload.artifact ? [payload.artifact] : [])
    ];

    artifactRecords.forEach((artifact, index) => {
      const id =
        readString(artifact.id) ??
        readString(artifact.title) ??
        `event-${traceEvent.event.sequence}-${index}`;
      sources.push({
        id: `payload-artifact:${id}`,
        kind: "artifact",
        label: readString(artifact.title) ?? readString(artifact.name) ?? id,
        status: "available",
        summary:
          readString(artifact.summary) ??
          readString(artifact.language) ??
          "Artifact metadata captured from stream payload.",
        sourceKeys: ["artifacts", "generated_artifacts", "outputs"],
        eventSequence: traceEvent.event.sequence
      });
    });
  }

  return dedupeSources(sources);
}

function missingSources({
  dependencySources,
  evaluationSources,
  evidenceSources,
  finalPayloadSources,
  snapshot,
  workstationState
}: {
  dependencySources: ProvenanceSource[];
  evaluationSources: ProvenanceSource[];
  evidenceSources: ProvenanceSource[];
  finalPayloadSources: ProvenanceSource[];
  snapshot: AssistantWorkspaceSnapshot;
  workstationState: WorkstationState;
}): ProvenanceSource[] {
  const missing: ProvenanceSource[] = [];

  if (evidenceSources.length === 0) {
    missing.push(missingSource("retrieval_evidence", "Retrieval evidence"));
  }
  if (dependencySources.length === 0) {
    missing.push(missingSource("dependency_sources", "Dependency sources"));
  }
  if (evaluationSources.length === 0) {
    missing.push(missingSource("evaluation_sources", "Evaluation sources"));
  }
  if (finalPayloadSources.length === 0 && workstationState.currentRun.state !== "idle") {
    missing.push(missingSource("final_payload", "Final payload"));
  }
  if (snapshot.artifacts.length === 0) {
    missing.push(missingSource("artifact_sources", "Artifact sources"));
  }

  return missing;
}

function missingSource(id: string, label: string): ProvenanceSource {
  return {
    id: `missing:${id}`,
    kind: "missing",
    label,
    status: "missing",
    summary: `${label} have not been captured in existing metadata.`,
    sourceKeys: [],
    eventSequence: null
  };
}

function provenanceSummary({
  artifactSources,
  availableSources,
  dependencySources,
  evaluationSources,
  evidenceSources,
  finalPayloadSources,
  unsupportedOrMissingSources
}: {
  artifactSources: ProvenanceSource[];
  availableSources: ProvenanceSource[];
  dependencySources: ProvenanceSource[];
  evaluationSources: ProvenanceSource[];
  evidenceSources: ProvenanceSource[];
  finalPayloadSources: ProvenanceSource[];
  unsupportedOrMissingSources: ProvenanceSource[];
}) {
  return `${availableSources.length} provenance sources available across ${evidenceSources.length} evidence, ${dependencySources.length} dependency, ${artifactSources.length} artifact, ${evaluationSources.length + finalPayloadSources.length} evaluation/final, and ${unsupportedOrMissingSources.length} missing groups.`;
}

function summarizePayloadValue(value: unknown): string | null {
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
    readString(record.evaluation_summary) ??
    readString(record.evaluationSummary) ??
    readString(record.dependency_summary) ??
    readString(record.dependencySummary);
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

function dedupeSources(sources: ProvenanceSource[]): ProvenanceSource[] {
  const seen = new Set<string>();
  return sources.filter((source) => {
    if (seen.has(source.id)) {
      return false;
    }
    seen.add(source.id);
    return true;
  });
}

function readRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function readRecordList(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value)
    ? value.filter(
        (item): item is Record<string, unknown> => readRecord(item) !== null
      )
    : [];
}

function readString(value: unknown): string | null {
  return typeof value === "string" && value.trim().length > 0
    ? value.trim()
    : null;
}

function truncate(value: string) {
  return value.length > 140 ? `${value.slice(0, 137)}...` : value;
}

function formatCode(value: string) {
  return value.replace(/[_-]+/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}
