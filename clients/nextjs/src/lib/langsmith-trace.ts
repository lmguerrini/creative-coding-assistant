import { readEventTimestamp } from "./assistant-stream";
import type { WorkflowRuntimeTraceEvent } from "./workflow-runtime";

export type LangSmithTraceState =
  | "unavailable"
  | "local"
  | "linked";

export type LangSmithTraceStatus =
  | "idle"
  | "running"
  | "complete"
  | "error"
  | "disabled";

export type LangSmithTraceMetadataEntry = {
  key: string;
  label: string;
  value: string;
};

export type LangSmithTraceMetadataGroup = {
  id: "provider" | "retrieval" | "evaluation" | "execution";
  label: string;
  entries: LangSmithTraceMetadataEntry[];
};

export type LangSmithTraceSpan = {
  id: string;
  parentId: string | null;
  runId: string | null;
  name: string;
  runType: string;
  stage: string | null;
  status: LangSmithTraceStatus;
  startedAt: string | null;
  endedAt: string | null;
  durationMs: number | null;
  depth: number;
  transitionFrom: string | null;
  transitionReason: string | null;
};

export type LangSmithTraceModel = {
  state: LangSmithTraceState;
  status: LangSmithTraceStatus;
  availabilityLabel: string;
  statusLabel: string;
  traceId: string | null;
  runId: string | null;
  parentRunId: string | null;
  runName: string | null;
  traceKind: string | null;
  projectName: string | null;
  providerLabel: string;
  startedAt: string | null;
  endedAt: string | null;
  durationMs: number | null;
  tags: string[];
  spans: LangSmithTraceSpan[];
  metadataGroups: LangSmithTraceMetadataGroup[];
  summary: {
    spanCount: number;
    nestedSpanCount: number;
    transitionCount: number;
    metadataCount: number;
  };
};

type TraceIdentity = {
  enabled: boolean;
  requested: boolean;
  provider: string | null;
  traceId: string | null;
  runId: string | null;
  parentRunId: string | null;
  runName: string | null;
  traceKind: string | null;
  projectName: string | null;
  status: string | null;
  reason: string | null;
  createdAt: string | null;
  endedAt: string | null;
  durationMs: number | null;
  tags: string[];
  metadata: Record<string, unknown> | null;
};

type SpanCandidate = LangSmithTraceSpan & {
  sequence: number;
  sortAtMs: number | null;
};

const emptyMetadataGroups: LangSmithTraceMetadataGroup[] = [
  { id: "provider", label: "Provider metadata", entries: [] },
  { id: "retrieval", label: "Retrieval metadata", entries: [] },
  { id: "evaluation", label: "Evaluation metadata", entries: [] },
  { id: "execution", label: "Execution metadata", entries: [] }
];

const ignoredMetadataKeys = new Set([
  "answer",
  "code",
  "emitted_at",
  "message",
  "observability",
  "text",
  "workflow"
]);

export function buildLangSmithTraceModel(
  traceEvents: WorkflowRuntimeTraceEvent[]
): LangSmithTraceModel {
  const observations = traceEvents
    .map((traceEvent) => ({
      traceEvent,
      record: readRecord(traceEvent.event.payload.observability)
    }))
    .filter(
      (
        observation
      ): observation is {
        traceEvent: WorkflowRuntimeTraceEvent;
        record: Record<string, unknown>;
      } => observation.record !== null
    );

  if (observations.length === 0) {
    return emptyTraceModel();
  }

  const identities = observations.map(({ record }) => parseTraceIdentity(record));
  const latestIdentity = mergeTraceIdentities(identities);
  const terminalEvent = [...traceEvents]
    .reverse()
    .find(({ event }) => event.event_type === "final" || event.event_type === "error");
  const terminalEventType =
    terminalEvent?.event.event_type === "final" ||
    terminalEvent?.event.event_type === "error"
      ? terminalEvent.event.event_type
      : null;
  const firstEvent = traceEvents[0] ?? observations[0].traceEvent;
  const startedAt =
    latestIdentity.createdAt ??
    readEventTimestamp(firstEvent.event) ??
    firstEvent.receivedAt;
  const endedAt =
    latestIdentity.endedAt ??
    (terminalEvent
      ? readEventTimestamp(terminalEvent.event) ?? terminalEvent.receivedAt
      : null);
  const durationMs =
    latestIdentity.durationMs ?? differenceMs(startedAt, endedAt);
  const status = deriveTraceStatus({
    enabled: latestIdentity.enabled,
    requested: latestIdentity.requested,
    rawStatus: latestIdentity.status,
    terminalEventType
  });
  const spans = buildTraceSpans(observations, latestIdentity.traceId);
  const metadataGroups = buildMetadataGroups(traceEvents, latestIdentity);

  return {
    state: latestIdentity.enabled
      ? "linked"
      : latestIdentity.requested || latestIdentity.traceId
        ? "local"
        : "unavailable",
    status,
    availabilityLabel: latestIdentity.enabled
      ? "LangSmith linked"
      : latestIdentity.requested || latestIdentity.traceId
        ? "Local trace metadata"
        : "Trace unavailable",
    statusLabel: formatTraceStatus(status, latestIdentity.reason),
    traceId: latestIdentity.traceId,
    runId: latestIdentity.runId,
    parentRunId: latestIdentity.parentRunId,
    runName: latestIdentity.runName,
    traceKind: latestIdentity.traceKind,
    projectName: latestIdentity.projectName,
    providerLabel: latestIdentity.provider ?? "langsmith",
    startedAt,
    endedAt,
    durationMs,
    tags: latestIdentity.tags,
    spans,
    metadataGroups,
    summary: {
      spanCount: spans.length,
      nestedSpanCount: spans.filter((span) => span.depth > 0).length,
      transitionCount: spans.filter((span) => span.transitionFrom !== null).length,
      metadataCount: metadataGroups.reduce(
        (total, group) => total + group.entries.length,
        0
      )
    }
  };
}

function emptyTraceModel(): LangSmithTraceModel {
  return {
    state: "unavailable",
    status: "idle",
    availabilityLabel: "Trace unavailable",
    statusLabel: "No LangSmith trace",
    traceId: null,
    runId: null,
    parentRunId: null,
    runName: null,
    traceKind: null,
    projectName: null,
    providerLabel: "langsmith",
    startedAt: null,
    endedAt: null,
    durationMs: null,
    tags: [],
    spans: [],
    metadataGroups: emptyMetadataGroups.map((group) => ({
      ...group,
      entries: []
    })),
    summary: {
      spanCount: 0,
      nestedSpanCount: 0,
      transitionCount: 0,
      metadataCount: 0
    }
  };
}

function parseTraceIdentity(record: Record<string, unknown>): TraceIdentity {
  const metadata = readRecord(record.metadata);
  return {
    enabled: readBoolean(record.enabled),
    requested: readBoolean(record.requested),
    provider: readString(record.provider),
    traceId: readString(record.trace_id) ?? readString(record.traceId),
    runId:
      readString(record.run_id) ??
      readString(record.runId) ??
      readString(metadata?.run_id) ??
      readString(metadata?.runId),
    parentRunId:
      readString(record.parent_run_id) ??
      readString(record.parentRunId) ??
      readString(metadata?.parent_run_id) ??
      readString(metadata?.parentRunId),
    runName: readString(record.run_name) ?? readString(record.runName),
    traceKind: readString(record.trace_kind) ?? readString(record.traceKind),
    projectName:
      readString(record.project_name) ?? readString(record.projectName),
    status: readString(record.status),
    reason: readString(record.reason),
    createdAt:
      readString(record.created_at) ??
      readString(record.createdAt) ??
      readString(record.start_time) ??
      readString(record.startTime),
    endedAt:
      readString(record.ended_at) ??
      readString(record.endedAt) ??
      readString(record.end_time) ??
      readString(record.endTime),
    durationMs:
      readNumber(record.duration_ms) ??
      readNumber(record.durationMs) ??
      readNumber(record.latency_ms),
    tags: readStringArray(record.tags),
    metadata
  };
}

function mergeTraceIdentities(identities: TraceIdentity[]): TraceIdentity {
  return identities.reduce<TraceIdentity>(
    (current, identity) => ({
      enabled: current.enabled || identity.enabled,
      requested: current.requested || identity.requested,
      provider: identity.provider ?? current.provider,
      traceId: identity.traceId ?? current.traceId,
      runId: identity.runId ?? current.runId,
      parentRunId: identity.parentRunId ?? current.parentRunId,
      runName: identity.runName ?? current.runName,
      traceKind: identity.traceKind ?? current.traceKind,
      projectName: identity.projectName ?? current.projectName,
      status: identity.status ?? current.status,
      reason:
        identity.status !== null
          ? identity.reason
          : identity.reason ?? current.reason,
      createdAt: current.createdAt ?? identity.createdAt,
      endedAt: identity.endedAt ?? current.endedAt,
      durationMs: identity.durationMs ?? current.durationMs,
      tags: uniqueStrings([...current.tags, ...identity.tags]),
      metadata: {
        ...(current.metadata ?? {}),
        ...(identity.metadata ?? {})
      }
    }),
    {
      enabled: false,
      requested: false,
      provider: null,
      traceId: null,
      runId: null,
      parentRunId: null,
      runName: null,
      traceKind: null,
      projectName: null,
      status: null,
      reason: null,
      createdAt: null,
      endedAt: null,
      durationMs: null,
      tags: [],
      metadata: null
    }
  );
}

function buildTraceSpans(
  observations: {
    traceEvent: WorkflowRuntimeTraceEvent;
    record: Record<string, unknown>;
  }[],
  traceId: string | null
): LangSmithTraceSpan[] {
  const candidates: SpanCandidate[] = [];
  let previousStage: string | null = null;

  observations.forEach(({ traceEvent, record }, observationIndex) => {
    const explicitCollections = [record.spans, record.runs, record.children];
    for (const collection of explicitCollections) {
      collectExplicitSpans({
        candidates,
        depth: 0,
        fallbackTraceId: traceId,
        sequence: traceEvent.event.sequence,
        value: collection
      });
    }

    const lineage = readRecord(record.lineage);
    const stage =
      readString(lineage?.stage) ??
      readString(lineage?.node) ??
      readString(lineage?.step) ??
      readWorkflowStage(traceEvent);
    if (!stage) {
      return;
    }

    const at =
      readEventTimestamp(traceEvent.event) ??
      traceEvent.receivedAt;
    const runId =
      readString(lineage?.run_id) ??
      readString(lineage?.runId);
    const code =
      readString(traceEvent.event.payload.code) ??
      traceEvent.event.event_type;
    const reason =
      readString(lineage?.transition_reason) ??
      readString(lineage?.transitionReason) ??
      readString(traceEvent.event.payload.message) ??
      formatLabel(code);

    candidates.push({
      id:
        runId ??
        `${traceId ?? "trace"}:${stage}:${traceEvent.event.sequence}:${observationIndex}`,
      parentId:
        readString(record.parent_run_id) ??
        readString(record.parentRunId) ??
        readString(record.run_id) ??
        readString(record.runId) ??
        null,
      runId,
      name:
        readString(lineage?.name) ??
        readString(record.run_name) ??
        formatLabel(stage),
      runType:
        readString(lineage?.run_type) ??
        readString(lineage?.runType) ??
        "workflow",
      stage,
      status: statusFromEvent(traceEvent.event.event_type),
      startedAt: at,
      endedAt:
        traceEvent.event.event_type === "node_started" ? null : at,
      durationMs:
        readNumber(lineage?.duration_ms) ??
        readNumber(lineage?.durationMs),
      depth: 0,
      transitionFrom:
        previousStage && previousStage !== stage ? previousStage : null,
      transitionReason:
        previousStage && previousStage !== stage ? reason : null,
      sequence: traceEvent.event.sequence,
      sortAtMs: parseTimestamp(at)
    });
    previousStage = stage;
  });

  const deduped = new Map<string, SpanCandidate>();
  for (const candidate of candidates) {
    const existing = deduped.get(candidate.id);
    if (!existing || candidate.sequence >= existing.sequence) {
      deduped.set(candidate.id, existing ? mergeSpan(existing, candidate) : candidate);
    }
  }

  const spans = [...deduped.values()].sort(compareSpans);
  const depths = resolveSpanDepths(spans);
  return spans.map(({ sequence: _sequence, sortAtMs: _sortAtMs, ...span }) => ({
    ...span,
    depth: Math.max(span.depth, depths.get(span.id) ?? 0)
  }));
}

function collectExplicitSpans({
  candidates,
  depth,
  fallbackTraceId,
  parentId = null,
  sequence,
  value
}: {
  candidates: SpanCandidate[];
  depth: number;
  fallbackTraceId: string | null;
  parentId?: string | null;
  sequence: number;
  value: unknown;
}) {
  if (!Array.isArray(value)) {
    return;
  }

  value.forEach((item, index) => {
    const record = readRecord(item);
    if (!record) {
      return;
    }
    const metadata = readRecord(record.metadata);
    const lineage = readRecord(record.lineage);
    const runId =
      readString(record.run_id) ??
      readString(record.runId) ??
      readString(record.id);
    const id =
      runId ??
      `${fallbackTraceId ?? "trace"}:span:${sequence}:${depth}:${index}`;
    const startedAt =
      readString(record.start_time) ??
      readString(record.startTime) ??
      readString(record.started_at) ??
      readString(record.startedAt);
    const endedAt =
      readString(record.end_time) ??
      readString(record.endTime) ??
      readString(record.ended_at) ??
      readString(record.endedAt);
    const rawStatus =
      readString(record.status) ??
      readString(record.state);
    const stage =
      readString(lineage?.stage) ??
      readString(metadata?.stage) ??
      readString(record.stage);

    candidates.push({
      id,
      parentId:
        readString(record.parent_run_id) ??
        readString(record.parentRunId) ??
        parentId,
      runId,
      name:
        readString(record.name) ??
        readString(record.run_name) ??
        readString(record.runName) ??
        (stage ? formatLabel(stage) : `Span ${index + 1}`),
      runType:
        readString(record.run_type) ??
        readString(record.runType) ??
        readString(record.type) ??
        "span",
      stage,
      status: normalizeTraceStatus(rawStatus, endedAt),
      startedAt,
      endedAt,
      durationMs:
        readNumber(record.duration_ms) ??
        readNumber(record.durationMs) ??
        differenceMs(startedAt, endedAt),
      depth,
      transitionFrom:
        readString(lineage?.from) ??
        readString(lineage?.previous_stage) ??
        null,
      transitionReason:
        readString(lineage?.transition_reason) ??
        readString(lineage?.reason) ??
        null,
      sequence,
      sortAtMs: parseTimestamp(startedAt)
    });

    collectExplicitSpans({
      candidates,
      depth: depth + 1,
      fallbackTraceId,
      parentId: id,
      sequence,
      value: record.children ?? record.spans ?? record.runs
    });
  });
}

function mergeSpan(
  current: SpanCandidate,
  next: SpanCandidate
): SpanCandidate {
  return {
    ...current,
    ...next,
    parentId: next.parentId ?? current.parentId,
    runId: next.runId ?? current.runId,
    stage: next.stage ?? current.stage,
    startedAt: current.startedAt ?? next.startedAt,
    endedAt: next.endedAt ?? current.endedAt,
    durationMs: next.durationMs ?? current.durationMs,
    transitionFrom: next.transitionFrom ?? current.transitionFrom,
    transitionReason: next.transitionReason ?? current.transitionReason,
    sortAtMs: current.sortAtMs ?? next.sortAtMs
  };
}

function resolveSpanDepths(spans: SpanCandidate[]) {
  const spansById = new Map(spans.map((span) => [span.id, span]));
  const depths = new Map<string, number>();

  const resolveDepth = (span: SpanCandidate, visited: Set<string>): number => {
    if (depths.has(span.id)) {
      return depths.get(span.id) ?? 0;
    }
    if (!span.parentId || visited.has(span.id)) {
      depths.set(span.id, span.depth);
      return span.depth;
    }
    const parent = spansById.get(span.parentId);
    if (!parent) {
      depths.set(span.id, span.depth);
      return span.depth;
    }
    const nextVisited = new Set(visited);
    nextVisited.add(span.id);
    const depth = Math.max(span.depth, resolveDepth(parent, nextVisited) + 1);
    depths.set(span.id, depth);
    return depth;
  };

  spans.forEach((span) => resolveDepth(span, new Set()));
  return depths;
}

function buildMetadataGroups(
  traceEvents: WorkflowRuntimeTraceEvent[],
  identity: TraceIdentity
): LangSmithTraceMetadataGroup[] {
  const provider = new Map<string, LangSmithTraceMetadataEntry>();
  const retrieval = new Map<string, LangSmithTraceMetadataEntry>();
  const evaluation = new Map<string, LangSmithTraceMetadataEntry>();
  const execution = new Map<string, LangSmithTraceMetadataEntry>();

  addRecordEntries(execution, identity.metadata);
  addMetadataEntry(execution, "trace_id", identity.traceId);
  addMetadataEntry(execution, "run_id", identity.runId);
  addMetadataEntry(execution, "parent_run_id", identity.parentRunId);
  addMetadataEntry(execution, "run_name", identity.runName);
  addMetadataEntry(execution, "trace_kind", identity.traceKind);
  addMetadataEntry(execution, "project_name", identity.projectName);
  addMetadataEntry(execution, "observability_status", identity.status);
  addMetadataEntry(execution, "observability_reason", identity.reason);
  addMetadataEntry(execution, "requested", identity.requested);
  addMetadataEntry(execution, "enabled", identity.enabled);

  for (const { event } of traceEvents) {
    const telemetry = readRecord(event.payload.telemetry);
    addRecordEntries(
      provider,
      readRecord(telemetry?.provider) ?? readRecord(event.payload.provider)
    );
    addRecordEntries(provider, readRecord(telemetry?.execution));

    if (event.event_type === "retrieval") {
      addRecordEntries(
        retrieval,
        readRecord(event.payload.retrieval) ?? event.payload
      );
      const observability = readRecord(event.payload.observability);
      addRecordEntries(retrieval, readRecord(observability?.lineage));
    }

    if (event.event_type === "eval_update") {
      addRecordEntries(
        evaluation,
        readRecord(event.payload.evaluation) ??
          readRecord(event.payload.ragas) ??
          event.payload
      );
    }
  }

  return [
    metadataGroup("provider", "Provider metadata", provider),
    metadataGroup("retrieval", "Retrieval metadata", retrieval),
    metadataGroup("evaluation", "Evaluation metadata", evaluation),
    metadataGroup("execution", "Execution metadata", execution)
  ];
}

function addRecordEntries(
  target: Map<string, LangSmithTraceMetadataEntry>,
  record: Record<string, unknown> | null,
  prefix = ""
) {
  if (!record) {
    return;
  }

  for (const [key, value] of Object.entries(record)) {
    if (target.size >= 16) {
      return;
    }
    if (ignoredMetadataKeys.has(key)) {
      continue;
    }
    const entryKey = prefix ? `${prefix}.${key}` : key;
    const scalar = formatMetadataValue(value);
    if (scalar !== null) {
      addMetadataEntry(target, entryKey, scalar);
      continue;
    }
    const nested = readRecord(value);
    if (nested && !["metadata", "observability"].includes(key)) {
      addRecordEntries(target, nested, entryKey);
    }
  }
}

function addMetadataEntry(
  target: Map<string, LangSmithTraceMetadataEntry>,
  key: string,
  value: unknown
) {
  const formatted = formatMetadataValue(value);
  if (formatted === null) {
    return;
  }
  target.set(key, {
    key,
    label: formatLabel(key.replaceAll(".", " ")),
    value: formatted
  });
}

function metadataGroup(
  id: LangSmithTraceMetadataGroup["id"],
  label: string,
  entries: Map<string, LangSmithTraceMetadataEntry>
): LangSmithTraceMetadataGroup {
  return {
    id,
    label,
    entries: [...entries.values()]
  };
}

function deriveTraceStatus({
  enabled,
  requested,
  rawStatus,
  terminalEventType
}: {
  enabled: boolean;
  requested: boolean;
  rawStatus: string | null;
  terminalEventType: "final" | "error" | null;
}): LangSmithTraceStatus {
  if (terminalEventType === "error") {
    return "error";
  }
  if (terminalEventType === "final") {
    return "complete";
  }
  if (!enabled && requested) {
    return "disabled";
  }
  return normalizeTraceStatus(rawStatus, null);
}

function normalizeTraceStatus(
  value: string | null,
  endedAt: string | null
): LangSmithTraceStatus {
  const normalized = value?.toLowerCase();
  if (["error", "failed", "failure"].includes(normalized ?? "")) {
    return "error";
  }
  if (
    ["complete", "completed", "success", "succeeded", "finished"].includes(
      normalized ?? ""
    ) ||
    endedAt
  ) {
    return "complete";
  }
  if (["disabled", "unavailable", "sampled_out"].includes(normalized ?? "")) {
    return "disabled";
  }
  if (["running", "active", "started", "enabled"].includes(normalized ?? "")) {
    return "running";
  }
  return "idle";
}

function statusFromEvent(
  eventType: WorkflowRuntimeTraceEvent["event"]["event_type"]
): LangSmithTraceStatus {
  if (eventType === "error" || eventType === "node_failed") {
    return "error";
  }
  if (eventType === "node_started") {
    return "running";
  }
  return "complete";
}

function formatTraceStatus(
  status: LangSmithTraceStatus,
  reason: string | null
) {
  if (status === "disabled" && reason) {
    return formatLabel(reason);
  }
  return formatLabel(status);
}

function compareSpans(left: SpanCandidate, right: SpanCandidate) {
  if (left.sortAtMs !== null && right.sortAtMs !== null) {
    return left.sortAtMs - right.sortAtMs || left.sequence - right.sequence;
  }
  if (left.sortAtMs !== null) {
    return -1;
  }
  if (right.sortAtMs !== null) {
    return 1;
  }
  return left.sequence - right.sequence;
}

function differenceMs(startedAt: string | null, endedAt: string | null) {
  const start = parseTimestamp(startedAt);
  const end = parseTimestamp(endedAt);
  if (start === null || end === null || end < start) {
    return null;
  }
  return end - start;
}

function readWorkflowStage(traceEvent: WorkflowRuntimeTraceEvent) {
  const workflow = readRecord(traceEvent.event.payload.workflow);
  return (
    readString(workflow?.step) ??
    readString(workflow?.current_step) ??
    readString(workflow?.currentStep) ??
    readString(traceEvent.event.payload.node)
  );
}

function parseTimestamp(value: string | null) {
  if (!value) {
    return null;
  }
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatMetadataValue(value: unknown): string | null {
  if (typeof value === "string") {
    return value.trim() || null;
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return String(value);
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  if (Array.isArray(value)) {
    const values = value
      .map((item) => formatMetadataValue(item))
      .filter((item): item is string => item !== null);
    return values.length > 0 ? values.join(", ") : null;
  }
  return null;
}

function formatLabel(value: string) {
  return value
    .replace(/[._-]+/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function uniqueStrings(values: string[]) {
  return [...new Set(values.filter(Boolean))];
}

function readRecord(value: unknown): Record<string, unknown> | null {
  return isRecord(value) ? value : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function readBoolean(value: unknown): boolean {
  return value === true;
}

function readNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? value
    : null;
}

function readStringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter(
        (item): item is string => typeof item === "string" && item.trim().length > 0
      )
    : [];
}
