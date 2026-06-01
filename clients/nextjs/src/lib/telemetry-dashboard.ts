import type {
  ArtifactSummary,
  AssistantWorkspaceSnapshot,
  PreviewSummary
} from "./assistant-client";
import {
  readEventTimestamp,
  readPreviewArtifactUpdate,
  type AssistantStreamEvent,
  type AssistantStreamEventType
} from "./assistant-stream";
import type { ProviderTelemetryModel } from "./provider-telemetry";
import type { RetrievalRuntimeModel } from "./retrieval-runtime";
import type {
  WorkflowRuntimeModel,
  WorkflowRuntimeTraceEvent
} from "./workflow-runtime";

export type TelemetryDashboardStatus =
  | "idle"
  | "running"
  | "complete"
  | "degraded"
  | "error";

export type TelemetrySignalTone =
  | "neutral"
  | "info"
  | "good"
  | "warning"
  | "danger";

export type TelemetrySignalSummary = {
  id:
    | "workflow"
    | "provider"
    | "preview"
    | "retrieval"
    | "observability"
    | "evaluation";
  label: string;
  value: string;
  detail: string;
  tone: TelemetrySignalTone;
};

export type TelemetryStreamLifecycle = {
  state: TelemetryDashboardStatus;
  eventCount: number;
  errorCount: number;
  previewEventCount: number;
  evalEventCount: number;
  eventTypeCounts: Record<AssistantStreamEventType, number>;
  startedAt: string | null;
  completedAt: string | null;
  latestEventAt: string | null;
  latestEventLabel: string;
};

export type TelemetryRuntimeLifecycle = {
  workflowStatus: string;
  currentStep: string;
  reachedNodes: number;
  totalNodes: number;
  retryCount: number;
  transitionCount: number;
  totalRuntimeMs: number | null;
  activeRuntimeMs: number | null;
};

export type TelemetryPreviewHealth = {
  state: PreviewSummary["state"];
  active: boolean;
  available: boolean;
  renderer: string;
  target: string;
  artifactName: string;
  healthLabel: string;
  detail: string;
  error: string | null;
  latestPreviewEventAt: string | null;
};

export type TelemetryRetrievalActivity = {
  state: RetrievalRuntimeModel["summary"]["state"];
  status: string;
  providerLabel: string;
  sourceCount: number;
  chunkCount: number;
  query: string | null;
  qualityLabel: string;
  freshnessLabel: string;
  warning: string | null;
  error: string | null;
};

export type TelemetryObservabilitySummary = {
  state: "linked" | "requested" | "disabled" | "unavailable";
  providerLabel: string;
  traceId: string | null;
  traceKind: string | null;
  projectName: string | null;
  status: string | null;
  reason: string | null;
  enabled: boolean;
  requested: boolean;
  latestAt: string | null;
  tags: string[];
};

export type TelemetryEvaluationLineage = {
  state: "available" | "pending" | "unavailable";
  runId: string | null;
  datasetId: string | null;
  metrics: string[];
  resultRows: number | null;
  metricFailures: number | null;
  dryRun: boolean | null;
  providerCallsAllowed: boolean | null;
  statusLabel: string;
  detail: string;
  latestAt: string | null;
};

export type TelemetryArtifactRuntimeLink = {
  activeArtifactId: string;
  activeArtifactTitle: string;
  previewArtifactId: string | null;
  renderer: string;
  target: string;
  status: string;
  linkLabel: string;
};

export type TelemetryDashboardModel = {
  status: TelemetryDashboardStatus;
  summary: {
    operatorStatus: string;
    signalLabel: string;
    coverageLabel: string;
    runtimeLabel: string;
  };
  signals: TelemetrySignalSummary[];
  stream: TelemetryStreamLifecycle;
  runtime: TelemetryRuntimeLifecycle;
  provider: ProviderTelemetryModel;
  preview: TelemetryPreviewHealth;
  retrieval: TelemetryRetrievalActivity;
  observability: TelemetryObservabilitySummary;
  evaluation: TelemetryEvaluationLineage;
  artifactLink: TelemetryArtifactRuntimeLink;
};

export type BuildTelemetryDashboardModelInput = {
  activeArtifact: ArtifactSummary;
  providerTelemetry: ProviderTelemetryModel;
  retrievalRuntime: RetrievalRuntimeModel;
  snapshot: AssistantWorkspaceSnapshot;
  traceEvents: WorkflowRuntimeTraceEvent[];
  workflowRuntime: WorkflowRuntimeModel;
};

type ObservabilityRecord = {
  provider: string | null;
  traceId: string | null;
  traceKind: string | null;
  projectName: string | null;
  status: string | null;
  reason: string | null;
  enabled: boolean;
  requested: boolean;
  tags: string[];
};

type EvaluationRecord = {
  runId: string | null;
  datasetId: string | null;
  metrics: string[];
  resultRows: number | null;
  metricFailures: number | null;
  dryRun: boolean | null;
  providerCallsAllowed: boolean | null;
  status: string | null;
  detail: string | null;
};

type PreviewRuntimeRecord = {
  artifact: string | null;
  error: string | null;
  fingerprint: string | null;
  kind: string | null;
  rendererLabel: string | null;
  state: string | null;
};

const allStreamEventTypes: AssistantStreamEventType[] = [
  "status",
  "memory",
  "retrieval",
  "context",
  "prompt_input",
  "prompt_rendered",
  "generation_input",
  "tool_start",
  "tool_result",
  "token_delta",
  "artifact_extracted",
  "preview_artifact",
  "eval_update",
  "final",
  "error"
];

export function buildTelemetryDashboardModel({
  activeArtifact,
  providerTelemetry,
  retrievalRuntime,
  snapshot,
  traceEvents,
  workflowRuntime
}: BuildTelemetryDashboardModelInput): TelemetryDashboardModel {
  const stream = buildStreamLifecycle(traceEvents);
  const runtime = buildRuntimeLifecycle(workflowRuntime);
  const preview = buildPreviewHealth(snapshot.preview, traceEvents);
  const retrieval = buildRetrievalActivity(retrievalRuntime);
  const observability = buildObservabilitySummary(traceEvents);
  const evaluation = buildEvaluationLineage(traceEvents, observability);
  const artifactLink = buildArtifactRuntimeLink({
    activeArtifact,
    preview: snapshot.preview
  });
  const status = deriveDashboardStatus({
    observability,
    preview,
    providerTelemetry,
    retrieval,
    stream,
    workflowRuntime
  });
  const signals = buildSignals({
    evaluation,
    observability,
    preview,
    providerTelemetry,
    retrieval,
    runtime
  });

  return {
    status,
    summary: {
      operatorStatus: formatDashboardStatus(status),
      signalLabel: summarizeSignals(signals),
      coverageLabel: summarizeCoverage({
        evaluation,
        observability,
        preview,
        providerTelemetry,
        retrieval,
        stream
      }),
      runtimeLabel:
        runtime.totalRuntimeMs != null
          ? `${formatDuration(runtime.totalRuntimeMs)} runtime`
          : "Runtime timing pending"
    },
    signals,
    stream,
    runtime,
    provider: providerTelemetry,
    preview,
    retrieval,
    observability,
    evaluation,
    artifactLink
  };
}

function buildStreamLifecycle(
  traceEvents: WorkflowRuntimeTraceEvent[]
): TelemetryStreamLifecycle {
  const eventTypeCounts = Object.fromEntries(
    allStreamEventTypes.map((eventType) => [eventType, 0])
  ) as Record<AssistantStreamEventType, number>;
  let errorCount = 0;
  let previewEventCount = 0;
  let evalEventCount = 0;

  for (const traceEvent of traceEvents) {
    eventTypeCounts[traceEvent.event.event_type] += 1;
    if (traceEvent.event.event_type === "error") {
      errorCount += 1;
    }
    if (traceEvent.event.event_type === "preview_artifact") {
      previewEventCount += 1;
    }
    if (traceEvent.event.event_type === "eval_update") {
      evalEventCount += 1;
    }
  }

  const firstEvent = traceEvents[0] ?? null;
  const latestEvent = traceEvents[traceEvents.length - 1] ?? null;
  const terminalEvent =
    [...traceEvents].reverse().find((traceEvent) =>
      ["final", "error"].includes(traceEvent.event.event_type)
    ) ?? null;

  return {
    state:
      errorCount > 0
        ? "error"
        : terminalEvent?.event.event_type === "final"
          ? "complete"
          : traceEvents.length > 0
            ? "running"
            : "idle",
    eventCount: traceEvents.length,
    errorCount,
    previewEventCount,
    evalEventCount,
    eventTypeCounts,
    startedAt: firstEvent?.receivedAt ?? null,
    completedAt: terminalEvent?.receivedAt ?? null,
    latestEventAt: latestEvent?.receivedAt ?? null,
    latestEventLabel: latestEvent
      ? formatEventLabel(latestEvent.event)
      : "No stream events"
  };
}

function buildRuntimeLifecycle(
  workflowRuntime: WorkflowRuntimeModel
): TelemetryRuntimeLifecycle {
  return {
    workflowStatus: workflowRuntime.summary.status,
    currentStep: workflowRuntime.summary.currentStep,
    reachedNodes: workflowRuntime.summary.reached,
    totalNodes: workflowRuntime.summary.total,
    retryCount: workflowRuntime.summary.retryCount,
    transitionCount: workflowRuntime.summary.transitionCount,
    totalRuntimeMs: workflowRuntime.summary.totalRuntimeMs,
    activeRuntimeMs: workflowRuntime.summary.activeRuntimeMs
  };
}

function buildPreviewHealth(
  preview: PreviewSummary,
  traceEvents: WorkflowRuntimeTraceEvent[]
): TelemetryPreviewHealth {
  const latestPreview = findLatestPreviewArtifact(traceEvents);
  const latestRuntime = findLatestPreviewRuntime(traceEvents);
  const error =
    preview.error?.userMessage ??
    latestRuntime?.record.error ??
    latestPreview?.errorMessage ??
    null;
  const runtimeMessage =
    latestRuntime?.record.state === "running"
      ? `${latestRuntime.record.rendererLabel ?? latestRuntime.record.kind ?? "Preview"} sandbox runtime is executing ${latestRuntime.record.artifact ?? preview.artifactName}.`
      : null;

  return {
    state: preview.state,
    active: preview.active,
    available: preview.available,
    renderer:
      preview.renderer ||
      latestRuntime?.record.rendererLabel ||
      "Renderer pending",
    target: preview.target || "Target pending",
    artifactName:
      preview.artifactName ||
      latestRuntime?.record.artifact ||
      "No artifact",
    healthLabel: formatPreviewHealthLabel(preview),
    detail:
      runtimeMessage ??
      preview.summary ??
      "Preview runtime metadata has not arrived yet.",
    error,
    latestPreviewEventAt:
      latestRuntime?.at ??
      latestPreview?.emittedAt ??
      latestPreview?.completedAt ??
      null
  };
}

function buildRetrievalActivity(
  retrievalRuntime: RetrievalRuntimeModel
): TelemetryRetrievalActivity {
  return {
    state: retrievalRuntime.summary.state,
    status: retrievalRuntime.summary.status,
    providerLabel: retrievalRuntime.summary.providerLabel,
    sourceCount: retrievalRuntime.summary.sourceCount,
    chunkCount: retrievalRuntime.summary.chunkCount,
    query: retrievalRuntime.request.query,
    qualityLabel: retrievalRuntime.summary.qualityLabel,
    freshnessLabel: retrievalRuntime.summary.freshnessLabel,
    warning: retrievalRuntime.summary.warning,
    error: retrievalRuntime.summary.error?.userMessage ?? null
  };
}

function buildObservabilitySummary(
  traceEvents: WorkflowRuntimeTraceEvent[]
): TelemetryObservabilitySummary {
  const latest = findLatestObservabilityRecord(traceEvents);
  if (!latest) {
    return {
      state: "unavailable",
      providerLabel: "LangSmith optional",
      traceId: null,
      traceKind: null,
      projectName: null,
      status: null,
      reason: null,
      enabled: false,
      requested: false,
      latestAt: null,
      tags: []
    };
  }

  return {
    state: latest.record.enabled
      ? "linked"
      : latest.record.requested
        ? "requested"
        : "disabled",
    providerLabel: latest.record.provider ?? "langsmith",
    traceId: latest.record.traceId,
    traceKind: latest.record.traceKind,
    projectName: latest.record.projectName,
    status: latest.record.status,
    reason: latest.record.reason,
    enabled: latest.record.enabled,
    requested: latest.record.requested,
    latestAt: latest.at,
    tags: latest.record.tags
  };
}

function buildEvaluationLineage(
  traceEvents: WorkflowRuntimeTraceEvent[],
  observability: TelemetryObservabilitySummary
): TelemetryEvaluationLineage {
  const latest = findLatestEvaluationRecord(traceEvents);
  if (latest) {
    return {
      state: "available",
      runId: latest.record.runId,
      datasetId: latest.record.datasetId,
      metrics: latest.record.metrics,
      resultRows: latest.record.resultRows,
      metricFailures: latest.record.metricFailures,
      dryRun: latest.record.dryRun,
      providerCallsAllowed: latest.record.providerCallsAllowed,
      statusLabel: latest.record.status ?? "Evaluation lineage captured",
      detail:
        latest.record.detail ??
        (latest.record.datasetId
          ? `Dataset ${latest.record.datasetId}`
          : "Evaluation event captured in the stream."),
      latestAt: latest.at
    };
  }

  if (observability.traceKind === "ragas_evaluation") {
    return {
      state: "available",
      runId: null,
      datasetId: null,
      metrics: [],
      resultRows: null,
      metricFailures: null,
      dryRun: null,
      providerCallsAllowed: null,
      statusLabel: "RAGAs trace linked",
      detail: "Evaluation lineage is available through the LangSmith trace metadata.",
      latestAt: observability.latestAt
    };
  }

  return {
    state: traceEvents.length > 0 ? "pending" : "unavailable",
    runId: null,
    datasetId: null,
    metrics: [],
    resultRows: null,
    metricFailures: null,
    dryRun: null,
    providerCallsAllowed: null,
    statusLabel:
      traceEvents.length > 0 ? "No eval event in stream" : "No evaluation run",
    detail:
      "RAGAs lineage appears here when eval_update events or linked eval traces are available.",
    latestAt: null
  };
}

function buildArtifactRuntimeLink({
  activeArtifact,
  preview
}: {
  activeArtifact: ArtifactSummary;
  preview: PreviewSummary;
}): TelemetryArtifactRuntimeLink {
  const previewArtifactId = preview.sourceArtifactId || null;
  const isLinked =
    previewArtifactId === activeArtifact.id ||
    preview.artifactName === activeArtifact.title ||
    preview.sourceArtifactName === activeArtifact.title;

  return {
    activeArtifactId: activeArtifact.id,
    activeArtifactTitle: activeArtifact.title,
    previewArtifactId,
    renderer: preview.renderer || "Renderer pending",
    target: preview.target || "Target pending",
    status: preview.status,
    linkLabel: isLinked
      ? "Active artifact linked to preview runtime"
      : "Preview is using a different artifact context"
  };
}

function deriveDashboardStatus({
  observability,
  preview,
  providerTelemetry,
  retrieval,
  stream,
  workflowRuntime
}: {
  observability: TelemetryObservabilitySummary;
  preview: TelemetryPreviewHealth;
  providerTelemetry: ProviderTelemetryModel;
  retrieval: TelemetryRetrievalActivity;
  stream: TelemetryStreamLifecycle;
  workflowRuntime: WorkflowRuntimeModel;
}): TelemetryDashboardStatus {
  if (
    stream.state === "error" ||
    workflowRuntime.error ||
    providerTelemetry.status === "error" ||
    preview.state === "error" ||
    retrieval.state === "error"
  ) {
    return "error";
  }

  if (
    preview.state === "unavailable" ||
    retrieval.state === "unavailable" ||
    observability.state === "requested"
  ) {
    return "degraded";
  }

  if (stream.state === "running" || providerTelemetry.status === "streaming") {
    return "running";
  }

  if (stream.state === "complete" || providerTelemetry.status === "complete") {
    return "complete";
  }

  return "idle";
}

function buildSignals({
  evaluation,
  observability,
  preview,
  providerTelemetry,
  retrieval,
  runtime
}: {
  evaluation: TelemetryEvaluationLineage;
  observability: TelemetryObservabilitySummary;
  preview: TelemetryPreviewHealth;
  providerTelemetry: ProviderTelemetryModel;
  retrieval: TelemetryRetrievalActivity;
  runtime: TelemetryRuntimeLifecycle;
}): TelemetrySignalSummary[] {
  return [
    {
      id: "workflow",
      label: "Workflow",
      value: runtime.currentStep,
      detail: `${runtime.reachedNodes}/${runtime.totalNodes} nodes reached; ${runtime.transitionCount} transitions`,
      tone:
        runtime.workflowStatus === "failed"
          ? "danger"
          : runtime.workflowStatus === "completed"
            ? "good"
            : "info"
    },
    {
      id: "provider",
      label: "Provider",
      value: providerTelemetry.summary.tokenLabel,
      detail: `${providerTelemetry.summary.providerLabel} / ${providerTelemetry.summary.costLabel}`,
      tone: providerTone(providerTelemetry.status)
    },
    {
      id: "preview",
      label: "Preview",
      value: formatPreviewState(preview.state),
      detail: `${preview.renderer} / ${preview.artifactName}`,
      tone: previewTone(preview.state)
    },
    {
      id: "retrieval",
      label: "Retrieval",
      value: retrieval.status,
      detail: `${retrieval.sourceCount} sources / ${retrieval.chunkCount} chunks`,
      tone: retrievalTone(retrieval.state)
    },
    {
      id: "observability",
      label: "LangSmith",
      value: observability.traceId
        ? compactIdentifier(observability.traceId)
        : formatObservabilityState(observability.state),
      detail: observability.projectName ?? observability.reason ?? "Optional tracing",
      tone: observabilityTone(observability.state)
    },
    {
      id: "evaluation",
      label: "Evaluation",
      value: evaluation.runId
        ? compactIdentifier(evaluation.runId)
        : evaluation.statusLabel,
      detail:
        evaluation.metrics.length > 0
          ? evaluation.metrics.join(", ")
          : evaluation.detail,
      tone: evaluation.state === "available" ? "good" : "neutral"
    }
  ];
}

function findLatestPreviewArtifact(traceEvents: WorkflowRuntimeTraceEvent[]) {
  for (let index = traceEvents.length - 1; index >= 0; index -= 1) {
    const update = readPreviewArtifactUpdate(traceEvents[index].event);
    if (update) {
      return update;
    }
  }
  return null;
}

function findLatestPreviewRuntime(traceEvents: WorkflowRuntimeTraceEvent[]) {
  for (let index = traceEvents.length - 1; index >= 0; index -= 1) {
    const event = traceEvents[index];
    const record = parsePreviewRuntimeRecord(event.event.payload.preview_runtime);
    if (record) {
      return {
        at: readEventTimestamp(event.event) ?? event.receivedAt,
        record
      };
    }
  }
  return null;
}

function findLatestObservabilityRecord(traceEvents: WorkflowRuntimeTraceEvent[]) {
  for (let index = traceEvents.length - 1; index >= 0; index -= 1) {
    const event = traceEvents[index];
    const record = parseObservabilityRecord(event.event.payload.observability);
    if (record) {
      return {
        at: readEventTimestamp(event.event) ?? event.receivedAt,
        record
      };
    }
  }
  return null;
}

function findLatestEvaluationRecord(traceEvents: WorkflowRuntimeTraceEvent[]) {
  for (let index = traceEvents.length - 1; index >= 0; index -= 1) {
    const event = traceEvents[index];
    const record = parseEvaluationRecord(event.event);
    if (record) {
      return {
        at: readEventTimestamp(event.event) ?? event.receivedAt,
        record
      };
    }
  }
  return null;
}

function parsePreviewRuntimeRecord(value: unknown): PreviewRuntimeRecord | null {
  if (!isRecord(value)) {
    return null;
  }

  return {
    artifact: readString(value.artifact),
    error: readString(value.error),
    fingerprint: readString(value.fingerprint),
    kind: readString(value.kind),
    rendererLabel: readString(value.renderer_label) ?? readString(value.rendererLabel),
    state: readString(value.state)
  };
}

function parseObservabilityRecord(value: unknown): ObservabilityRecord | null {
  if (!isRecord(value)) {
    return null;
  }

  return {
    provider: readString(value.provider),
    traceId: readString(value.trace_id) ?? readString(value.traceId),
    traceKind: readString(value.trace_kind) ?? readString(value.traceKind),
    projectName: readString(value.project_name) ?? readString(value.projectName),
    status: readString(value.status),
    reason: readString(value.reason),
    enabled: readBoolean(value.enabled),
    requested: readBoolean(value.requested),
    tags: readStringArray(value.tags)
  };
}

function parseEvaluationRecord(event: AssistantStreamEvent): EvaluationRecord | null {
  if (event.event_type !== "eval_update") {
    return null;
  }

  const payload = event.payload;
  const record =
    readRecord(payload.evaluation) ??
    readRecord(payload.ragas) ??
    readRecord(payload.result) ??
    payload;
  const dataset = readRecord(record.dataset);
  const observability = readRecord(record.observability) ?? readRecord(record.langsmith);
  const observabilityMetadata = readRecord(observability?.metadata);

  return {
    runId:
      readString(record.run_id) ??
      readString(record.runId) ??
      readString(observabilityMetadata?.run_id),
    datasetId:
      readString(record.dataset_id) ??
      readString(record.datasetId) ??
      readString(dataset?.dataset_id) ??
      readString(dataset?.datasetId),
    metrics:
      readStringArray(record.metrics).length > 0
        ? readStringArray(record.metrics)
        : readStringArray(dataset?.metrics),
    resultRows: readNumber(record.result_rows) ?? readNumber(record.resultRows),
    metricFailures:
      readNumber(record.metric_failures) ?? readNumber(record.metricFailures),
    dryRun: readOptionalBoolean(record.dry_run) ?? readOptionalBoolean(record.dryRun),
    providerCallsAllowed:
      readOptionalBoolean(record.provider_calls_allowed) ??
      readOptionalBoolean(record.providerCallsAllowed),
    status: readString(record.status) ?? readString(payload.code),
    detail: readString(record.detail) ?? readString(payload.message)
  };
}

function summarizeSignals(signals: TelemetrySignalSummary[]) {
  const danger = signals.filter((signal) => signal.tone === "danger").length;
  const warning = signals.filter((signal) => signal.tone === "warning").length;
  if (danger > 0) {
    return `${danger} critical signal${danger === 1 ? "" : "s"}`;
  }
  if (warning > 0) {
    return `${warning} degraded signal${warning === 1 ? "" : "s"}`;
  }
  return "All available signals nominal";
}

function summarizeCoverage({
  evaluation,
  observability,
  preview,
  providerTelemetry,
  retrieval,
  stream
}: {
  evaluation: TelemetryEvaluationLineage;
  observability: TelemetryObservabilitySummary;
  preview: TelemetryPreviewHealth;
  providerTelemetry: ProviderTelemetryModel;
  retrieval: TelemetryRetrievalActivity;
  stream: TelemetryStreamLifecycle;
}) {
  const available = [
    stream.eventCount > 0,
    providerTelemetry.tokenUsage.source === "provider",
    preview.available,
    retrieval.sourceCount > 0,
    observability.state !== "unavailable",
    evaluation.state === "available"
  ].filter(Boolean).length;

  return `${available}/6 telemetry domains populated`;
}

function formatDashboardStatus(status: TelemetryDashboardStatus) {
  switch (status) {
    case "complete":
      return "Run complete";
    case "degraded":
      return "Degraded telemetry";
    case "error":
      return "Runtime issue";
    case "running":
      return "Live run active";
    default:
      return "Telemetry idle";
  }
}

function formatPreviewHealthLabel(preview: PreviewSummary) {
  if (!preview.available) {
    return "Unavailable";
  }
  if (preview.error) {
    return "Failed";
  }
  if (preview.active && preview.state === "ready") {
    return "Running";
  }
  return formatPreviewState(preview.state);
}

function formatPreviewState(state: PreviewSummary["state"]) {
  switch (state) {
    case "ready":
      return "Ready";
    case "generating":
      return "Generating";
    case "error":
      return "Failed";
    case "unavailable":
      return "Unavailable";
    default:
      return "Unknown";
  }
}

function formatObservabilityState(
  state: TelemetryObservabilitySummary["state"]
) {
  switch (state) {
    case "linked":
      return "Linked";
    case "requested":
      return "Requested";
    case "disabled":
      return "Disabled";
    default:
      return "Unavailable";
  }
}

function providerTone(
  status: ProviderTelemetryModel["status"]
): TelemetrySignalTone {
  switch (status) {
    case "complete":
      return "good";
    case "error":
      return "danger";
    case "streaming":
      return "info";
    default:
      return "neutral";
  }
}

function previewTone(state: PreviewSummary["state"]): TelemetrySignalTone {
  switch (state) {
    case "ready":
      return "good";
    case "generating":
      return "info";
    case "error":
      return "danger";
    default:
      return "warning";
  }
}

function retrievalTone(
  state: RetrievalRuntimeModel["summary"]["state"]
): TelemetrySignalTone {
  switch (state) {
    case "available":
      return "good";
    case "pending":
      return "info";
    case "error":
      return "danger";
    case "unavailable":
      return "warning";
    default:
      return "neutral";
  }
}

function observabilityTone(
  state: TelemetryObservabilitySummary["state"]
): TelemetrySignalTone {
  switch (state) {
    case "linked":
      return "good";
    case "requested":
      return "warning";
    default:
      return "neutral";
  }
}

function formatEventLabel(event: AssistantStreamEvent) {
  const code = readString(event.payload.code);
  return code ? formatLabel(code) : formatLabel(event.event_type);
}

function formatLabel(value: string) {
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function compactIdentifier(value: string) {
  return value.length > 12 ? `${value.slice(0, 8)}...${value.slice(-4)}` : value;
}

function formatDuration(durationMs: number) {
  if (durationMs < 1000) {
    return `${durationMs}ms`;
  }
  return `${(durationMs / 1000).toFixed(durationMs >= 10000 ? 0 : 1)}s`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readRecord(value: unknown): Record<string, unknown> | null {
  return isRecord(value) ? value : null;
}

function readString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function readBoolean(value: unknown): boolean {
  return value === true;
}

function readOptionalBoolean(value: unknown): boolean | null {
  return typeof value === "boolean" ? value : null;
}

function readNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function readStringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}
