import {
  readEventTimestamp,
  readStreamEventError
} from "./assistant-stream";
import type { WorkflowRuntimeTraceEvent } from "./workflow-runtime";

export type ProviderTelemetryStatus =
  | "idle"
  | "streaming"
  | "complete"
  | "error";

export type ProviderTelemetryProvider = {
  name: string | null;
  model: string | null;
  requestId: string | null;
  responseId: string | null;
  runtime: string | null;
};

export type ProviderTelemetryTokenUsage = {
  inputTokens: number | null;
  outputTokens: number | null;
  totalTokens: number | null;
  cachedInputTokens: number | null;
  reasoningTokens: number | null;
  source: "provider" | "unavailable";
};

export type ProviderTelemetryPricing = {
  currency: string;
  inputCostPerMillionTokens: number | null;
  outputCostPerMillionTokens: number | null;
};

export type ProviderTelemetryCostEstimate = {
  currency: string;
  inputCost: number | null;
  outputCost: number | null;
  totalCost: number | null;
  source: "provider_reported" | "pricing_metadata" | "unavailable";
};

export type ProviderTelemetryTiming = {
  firstEventAt: string | null;
  generationStartedAt: string | null;
  firstTokenAt: string | null;
  completedAt: string | null;
  streamDurationMs: number | null;
  generationDurationMs: number | null;
  timeToFirstTokenMs: number | null;
  firstEventLatencyMs: number | null;
  averageEventLatencyMs: number | null;
  maxEventLatencyMs: number | null;
};

export type ProviderTelemetryLifecycleStep = {
  id: "request" | "generation_input" | "first_token" | "completion";
  label: string;
  state: "complete" | "active" | "pending" | "failed";
  at: string | null;
  offsetMs: number | null;
};

export type ProviderTelemetryStreamSummary = {
  eventCount: number;
  tokenDeltaCount: number;
  streamedCharacterCount: number;
};

export type ProviderTelemetryGenerationMode =
  | "streaming"
  | "non_streaming"
  | "unknown";

export type ProviderTelemetryStreamingState =
  | "idle"
  | "active"
  | "completed"
  | "failed"
  | "disabled"
  | "unknown";

export type ProviderTelemetryIssue = {
  id: string;
  severity: "warning" | "error";
  code: string;
  message: string;
  at: string | null;
  recoverable: boolean | null;
  source: "provider" | "stream";
};

export type ProviderTelemetryRetryEvent = {
  id: string;
  attempt: number | null;
  status: string;
  reason: string | null;
  at: string | null;
};

export type ProviderTelemetryFallbackPath = {
  id: string;
  label: string;
  reason: string;
  source: string | null;
  target: string | null;
  at: string | null;
};

export type ProviderTelemetryExecution = {
  generationMode: ProviderTelemetryGenerationMode;
  streamingEnabled: boolean | null;
  streamingState: ProviderTelemetryStreamingState;
  requestStartedAt: string | null;
  requestCompletedAt: string | null;
  requestDurationMs: number | null;
  retryCount: number | null;
  finishReason: string | null;
  errors: ProviderTelemetryIssue[];
  warnings: ProviderTelemetryIssue[];
  retryEvents: ProviderTelemetryRetryEvent[];
  fallbackPaths: ProviderTelemetryFallbackPath[];
};

export type ProviderTelemetryModel = {
  status: ProviderTelemetryStatus;
  provider: ProviderTelemetryProvider;
  tokenUsage: ProviderTelemetryTokenUsage;
  pricing: ProviderTelemetryPricing | null;
  cost: ProviderTelemetryCostEstimate;
  timing: ProviderTelemetryTiming;
  lifecycle: ProviderTelemetryLifecycleStep[];
  stream: ProviderTelemetryStreamSummary;
  execution: ProviderTelemetryExecution;
  summary: {
    providerLabel: string;
    modelLabel: string;
    tokenLabel: string;
    costLabel: string;
    latencyLabel: string;
    streamLabel: string;
    lifecycleLabel: string;
    generationModeLabel: string;
    streamingStatusLabel: string;
    requestDurationLabel: string;
    retryLabel: string;
    issueLabel: string;
  };
};

type ProviderExecutionRecord = {
  generationMode: ProviderTelemetryGenerationMode;
  streamingEnabled: boolean | null;
  streamingStatus: string | null;
  requestStartedAt: string | null;
  requestCompletedAt: string | null;
  requestDurationMs: number | null;
  retryCount: number | null;
  finishReason: string | null;
};

const emptyExecutionRecord: ProviderExecutionRecord = {
  generationMode: "unknown",
  streamingEnabled: null,
  streamingStatus: null,
  requestStartedAt: null,
  requestCompletedAt: null,
  requestDurationMs: null,
  retryCount: null,
  finishReason: null
};

const emptyProvider: ProviderTelemetryProvider = {
  name: null,
  model: null,
  requestId: null,
  responseId: null,
  runtime: null
};

const emptyTokenUsage: ProviderTelemetryTokenUsage = {
  inputTokens: null,
  outputTokens: null,
  totalTokens: null,
  cachedInputTokens: null,
  reasoningTokens: null,
  source: "unavailable"
};

const unavailableCost: ProviderTelemetryCostEstimate = {
  currency: "USD",
  inputCost: null,
  outputCost: null,
  totalCost: null,
  source: "unavailable"
};

export function buildProviderTelemetryModel(
  traceEvents: WorkflowRuntimeTraceEvent[]
): ProviderTelemetryModel {
  if (traceEvents.length === 0) {
    return buildTelemetryModel({
      cost: unavailableCost,
      provider: emptyProvider,
      pricing: null,
      status: "idle",
      stream: {
        eventCount: 0,
        tokenDeltaCount: 0,
        streamedCharacterCount: 0
      },
      timing: buildTiming(traceEvents),
      tokenUsage: emptyTokenUsage,
      execution: buildExecutionTelemetry({
        executionRecord: emptyExecutionRecord,
        status: "idle",
        traceEvents
      })
    });
  }

  let provider = emptyProvider;
  let tokenUsage = emptyTokenUsage;
  let pricing: ProviderTelemetryPricing | null = null;
  let reportedCost: ProviderTelemetryCostEstimate | null = null;
  let executionRecord = emptyExecutionRecord;
  let streamedCharacterCount = 0;
  let tokenDeltaCount = 0;

  for (const traceEvent of traceEvents) {
    const payload = traceEvent.event.payload;
    const telemetry = readTelemetryRecord(payload);
    const nextProvider = readProviderTelemetry(payload, telemetry);
    const nextUsage = readTokenUsage(payload, telemetry);
    const nextPricing = readPricing(payload, telemetry);
    const nextCost = readReportedCost(payload, telemetry);
    const nextExecution = readExecutionRecord(payload, telemetry);

    provider = mergeProviderTelemetry(provider, nextProvider);
    executionRecord = mergeExecutionRecord(executionRecord, nextExecution);

    if (nextUsage) {
      tokenUsage = mergeTokenUsage(tokenUsage, nextUsage);
    }
    if (nextPricing) {
      pricing = nextPricing;
    }
    if (nextCost.totalCost != null) {
      reportedCost = nextCost;
    }

    if (traceEvent.event.event_type === "token_delta") {
      tokenDeltaCount += 1;
      streamedCharacterCount +=
        typeof payload.text === "string" ? payload.text.length : 0;
    }
  }

  const status = deriveTelemetryStatus(traceEvents);
  const cost = reportedCost ?? estimateCostFromPricing(tokenUsage, pricing);
  const execution = buildExecutionTelemetry({
    executionRecord,
    status,
    traceEvents
  });

  return buildTelemetryModel({
    cost,
    execution,
    provider,
    pricing,
    status,
    stream: {
      eventCount: traceEvents.length,
      tokenDeltaCount,
      streamedCharacterCount
    },
    timing: buildTiming(traceEvents),
    tokenUsage
  });
}

function buildTelemetryModel({
  cost,
  execution,
  provider,
  pricing,
  status,
  stream,
  timing,
  tokenUsage
}: {
  cost: ProviderTelemetryCostEstimate;
  execution: ProviderTelemetryExecution;
  provider: ProviderTelemetryProvider;
  pricing: ProviderTelemetryPricing | null;
  status: ProviderTelemetryStatus;
  stream: ProviderTelemetryStreamSummary;
  timing: ProviderTelemetryTiming;
  tokenUsage: ProviderTelemetryTokenUsage;
}): ProviderTelemetryModel {
  const lifecycle = buildLifecycle(status, timing);

  return {
    status,
    provider,
    tokenUsage,
    pricing,
    cost,
    timing,
    lifecycle,
    stream,
    execution,
    summary: {
      providerLabel: provider.name ?? "Provider pending",
      modelLabel: provider.model ?? "Model pending",
      tokenLabel:
        tokenUsage.totalTokens != null
          ? `${formatCount(tokenUsage.totalTokens)} tokens`
          : "Usage pending",
      costLabel:
        cost.totalCost != null
          ? formatCost(cost.totalCost, cost.currency)
          : "Cost pending",
      latencyLabel:
        timing.firstEventLatencyMs != null
          ? `${formatDuration(timing.firstEventLatencyMs)} latency`
          : "Latency pending",
      streamLabel:
        stream.eventCount > 0
          ? `${stream.eventCount} events / ${stream.tokenDeltaCount} deltas`
          : "No stream events",
      lifecycleLabel: summarizeLifecycle(lifecycle),
      generationModeLabel: formatGenerationMode(execution.generationMode),
      streamingStatusLabel: formatStreamingState(execution.streamingState),
      requestDurationLabel:
        execution.requestDurationMs != null
          ? formatDuration(execution.requestDurationMs)
          : "Duration unavailable",
      retryLabel:
        execution.retryCount != null
          ? `${execution.retryCount} provider ${
              execution.retryCount === 1 ? "retry" : "retries"
            }`
          : "Retry metadata unavailable",
      issueLabel: summarizeIssues(execution)
    }
  };
}

function buildTiming(
  traceEvents: WorkflowRuntimeTraceEvent[]
): ProviderTelemetryTiming {
  const firstEvent = traceEvents[0] ?? null;
  const lastEvent = traceEvents[traceEvents.length - 1] ?? null;
  const generationInput = traceEvents.find(
    (traceEvent) => traceEvent.event.event_type === "generation_input"
  );
  const firstToken = traceEvents.find(
    (traceEvent) => traceEvent.event.event_type === "token_delta"
  );
  const terminalEvent = [...traceEvents]
    .reverse()
    .find((traceEvent) =>
      ["final", "error"].includes(traceEvent.event.event_type)
    );
  const latencies = traceEvents
    .map(readTransportLatency)
    .filter((latency): latency is number => latency !== null);

  return {
    firstEventAt: firstEvent?.receivedAt ?? null,
    generationStartedAt: generationInput?.receivedAt ?? null,
    firstTokenAt: firstToken?.receivedAt ?? null,
    completedAt: terminalEvent?.receivedAt ?? null,
    streamDurationMs:
      firstEvent && lastEvent
        ? Math.max(lastEvent.receivedAtMs - firstEvent.receivedAtMs, 0)
        : null,
    generationDurationMs:
      generationInput && terminalEvent
        ? Math.max(terminalEvent.receivedAtMs - generationInput.receivedAtMs, 0)
        : null,
    timeToFirstTokenMs:
      generationInput && firstToken
        ? Math.max(firstToken.receivedAtMs - generationInput.receivedAtMs, 0)
        : null,
    firstEventLatencyMs: firstEvent ? readTransportLatency(firstEvent) : null,
    averageEventLatencyMs:
      latencies.length > 0
        ? Math.round(
            latencies.reduce((total, latency) => total + latency, 0) /
              latencies.length
          )
        : null,
    maxEventLatencyMs:
      latencies.length > 0 ? Math.max(...latencies) : null
  };
}

function buildLifecycle(
  status: ProviderTelemetryStatus,
  timing: ProviderTelemetryTiming
): ProviderTelemetryLifecycleStep[] {
  const startMs = timing.firstEventAt ? Date.parse(timing.firstEventAt) : null;
  const hasStarted = timing.firstEventAt !== null;
  const hasGenerationInput = timing.generationStartedAt !== null;
  const hasFirstToken = timing.firstTokenAt !== null;
  const hasCompleted = timing.completedAt !== null;

  return [
    {
      id: "request",
      label: "Request",
      state: hasStarted ? "complete" : "pending",
      at: timing.firstEventAt,
      offsetMs: offsetFromStart(timing.firstEventAt, startMs)
    },
    {
      id: "generation_input",
      label: "Generation input",
      state: hasGenerationInput
        ? "complete"
        : hasStarted && status === "streaming"
          ? "active"
          : "pending",
      at: timing.generationStartedAt,
      offsetMs: offsetFromStart(timing.generationStartedAt, startMs)
    },
    {
      id: "first_token",
      label: "First token",
      state: hasFirstToken
        ? "complete"
        : hasGenerationInput && status === "streaming"
          ? "active"
          : "pending",
      at: timing.firstTokenAt,
      offsetMs: offsetFromStart(timing.firstTokenAt, startMs)
    },
    {
      id: "completion",
      label: status === "error" ? "Failure" : "Completion",
      state: hasCompleted ? (status === "error" ? "failed" : "complete") : "pending",
      at: timing.completedAt,
      offsetMs: offsetFromStart(timing.completedAt, startMs)
    }
  ];
}

function offsetFromStart(value: string | null, startMs: number | null) {
  if (!value || startMs === null) {
    return null;
  }
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? Math.max(parsed - startMs, 0) : null;
}

function readTransportLatency(traceEvent: WorkflowRuntimeTraceEvent): number | null {
  const emittedAt = readEventTimestamp(traceEvent.event);
  if (!emittedAt) {
    return null;
  }
  const emittedAtMs = Date.parse(emittedAt);
  if (!Number.isFinite(emittedAtMs)) {
    return null;
  }
  const latency = traceEvent.receivedAtMs - emittedAtMs;
  return latency >= 0 && latency <= 600_000 ? latency : null;
}

function deriveTelemetryStatus(
  traceEvents: WorkflowRuntimeTraceEvent[]
): ProviderTelemetryStatus {
  if (traceEvents.length === 0) {
    return "idle";
  }
  if (traceEvents.some((traceEvent) => traceEvent.event.event_type === "error")) {
    return "error";
  }
  if (traceEvents.some((traceEvent) => traceEvent.event.event_type === "final")) {
    return "complete";
  }
  return "streaming";
}

function buildExecutionTelemetry({
  executionRecord,
  status,
  traceEvents
}: {
  executionRecord: ProviderExecutionRecord;
  status: ProviderTelemetryStatus;
  traceEvents: WorkflowRuntimeTraceEvent[];
}): ProviderTelemetryExecution {
  const generationInput = traceEvents.find(
    (traceEvent) => traceEvent.event.event_type === "generation_input"
  );
  const inferredStreaming = inferStreamingPreference(generationInput);
  const hasTokenDeltas = traceEvents.some(
    (traceEvent) => traceEvent.event.event_type === "token_delta"
  );
  const streamingEnabled =
    executionRecord.streamingEnabled ??
    (executionRecord.generationMode === "streaming"
      ? true
      : executionRecord.generationMode === "non_streaming"
        ? false
        : inferredStreaming ?? (hasTokenDeltas ? true : null));
  const generationMode =
    executionRecord.generationMode !== "unknown"
      ? executionRecord.generationMode
      : streamingEnabled === true
        ? "streaming"
        : streamingEnabled === false
          ? "non_streaming"
          : "unknown";
  const errors = collectExecutionIssues(traceEvents, "error");
  const warnings = collectExecutionIssues(traceEvents, "warning");
  const finishReason = executionRecord.finishReason;

  if (
    finishReason &&
    ["cancelled", "length"].includes(finishReason.toLowerCase())
  ) {
    warnings.push({
      id: `finish-reason:${finishReason.toLowerCase()}`,
      severity: "warning",
      code: `finish_reason_${finishReason.toLowerCase()}`,
      message:
        finishReason.toLowerCase() === "length"
          ? "The provider stopped because the output token limit was reached."
          : "The provider generation was cancelled before normal completion.",
      at: executionRecord.requestCompletedAt,
      recoverable: true,
      source: "provider"
    });
  }

  return {
    generationMode,
    streamingEnabled,
    streamingState: deriveStreamingState({
      explicitStatus: executionRecord.streamingStatus,
      generationMode,
      status
    }),
    requestStartedAt:
      executionRecord.requestStartedAt ?? generationInput?.receivedAt ?? null,
    requestCompletedAt: executionRecord.requestCompletedAt,
    requestDurationMs: executionRecord.requestDurationMs,
    retryCount: executionRecord.retryCount,
    finishReason,
    errors: dedupeIssues(errors),
    warnings: dedupeIssues(warnings),
    retryEvents: collectRetryEvents(traceEvents),
    fallbackPaths: collectFallbackPaths(traceEvents)
  };
}

function readExecutionRecord(
  payload: Record<string, unknown>,
  telemetry: Record<string, unknown> | null
): ProviderExecutionRecord | null {
  const execution =
    readUsageRecord(telemetry?.execution) ??
    readUsageRecord(telemetry?.provider_execution) ??
    readUsageRecord(payload.execution) ??
    readUsageRecord(payload.provider_execution);
  const generationInput = readUsageRecord(payload.generation_input);
  const request = readUsageRecord(generationInput?.request);
  const explicitStreaming =
    readBoolean(execution?.streaming) ??
    readBoolean(execution?.is_streaming) ??
    readBoolean(telemetry?.streaming) ??
    readBoolean(request?.stream);
  const generationMode =
    normalizeGenerationMode(
      readString(execution?.generation_mode) ??
        readString(execution?.generationMode) ??
        readString(telemetry?.generation_mode)
    ) ??
    (explicitStreaming === true
      ? "streaming"
      : explicitStreaming === false
        ? "non_streaming"
        : "unknown");
  const record: ProviderExecutionRecord = {
    generationMode,
    streamingEnabled: explicitStreaming,
    streamingStatus:
      readString(execution?.streaming_status) ??
      readString(execution?.streamingStatus) ??
      readString(execution?.status),
    requestStartedAt:
      readString(execution?.request_started_at) ??
      readString(execution?.requestStartedAt),
    requestCompletedAt:
      readString(execution?.request_completed_at) ??
      readString(execution?.requestCompletedAt),
    requestDurationMs: readFiniteNumber(
      execution ?? {},
      "request_duration_ms",
      "requestDurationMs",
      "duration_ms",
      "durationMs"
    ),
    retryCount: readCount(
      execution ?? {},
      "retry_count",
      "retryCount",
      "retries"
    ),
    finishReason:
      readString(telemetry?.finish_reason) ??
      readString(telemetry?.finishReason) ??
      readString(payload.finish_reason)
  };

  return execution ||
    generationInput ||
    record.finishReason ||
    record.streamingEnabled !== null
    ? record
    : null;
}

function mergeExecutionRecord(
  current: ProviderExecutionRecord,
  next: ProviderExecutionRecord | null
): ProviderExecutionRecord {
  if (!next) {
    return current;
  }

  return {
    generationMode:
      next.generationMode !== "unknown"
        ? next.generationMode
        : current.generationMode,
    streamingEnabled: next.streamingEnabled ?? current.streamingEnabled,
    streamingStatus: next.streamingStatus ?? current.streamingStatus,
    requestStartedAt: next.requestStartedAt ?? current.requestStartedAt,
    requestCompletedAt: next.requestCompletedAt ?? current.requestCompletedAt,
    requestDurationMs: next.requestDurationMs ?? current.requestDurationMs,
    retryCount: next.retryCount ?? current.retryCount,
    finishReason: next.finishReason ?? current.finishReason
  };
}

function collectExecutionIssues(
  traceEvents: WorkflowRuntimeTraceEvent[],
  severity: ProviderTelemetryIssue["severity"]
) {
  const issues: ProviderTelemetryIssue[] = [];

  for (const traceEvent of traceEvents) {
    const telemetry = readTelemetryRecord(traceEvent.event.payload);
    const execution =
      readUsageRecord(telemetry?.execution) ??
      readUsageRecord(telemetry?.provider_execution) ??
      readUsageRecord(traceEvent.event.payload.execution) ??
      readUsageRecord(traceEvent.event.payload.provider_execution);
    const key = severity === "error" ? "errors" : "warnings";
    const values = readUnknownList(execution?.[key] ?? telemetry?.[key]);

    values.forEach((value, index) => {
      const issue = readExecutionIssue({
        at: readTraceTime(traceEvent),
        fallbackCode: `provider_${severity}_${index + 1}`,
        severity,
        value
      });
      if (issue) {
        issues.push(issue);
      }
    });

    if (severity !== "error" || traceEvent.event.event_type !== "error") {
      continue;
    }

    const streamError = readStreamEventError(traceEvent.event);
    if (!streamError || !isProviderError(traceEvent, streamError.subsystem)) {
      continue;
    }

    issues.push({
      id: `stream:${traceEvent.event.sequence}:${streamError.type}`,
      severity: "error",
      code: streamError.type,
      message: streamError.userMessage,
      at: readTraceTime(traceEvent),
      recoverable: streamError.recoverable,
      source: "stream"
    });
  }

  return issues;
}

function readExecutionIssue({
  at,
  fallbackCode,
  severity,
  value
}: {
  at: string | null;
  fallbackCode: string;
  severity: ProviderTelemetryIssue["severity"];
  value: unknown;
}): ProviderTelemetryIssue | null {
  if (typeof value === "string" && value.trim()) {
    const message = value.trim();
    return {
      id: `${severity}:${fallbackCode}:${message}`,
      severity,
      code: fallbackCode,
      message,
      at,
      recoverable: null,
      source: "provider"
    };
  }
  if (!isRecord(value)) {
    return null;
  }

  const code =
    readString(value.code) ??
    readString(value.type) ??
    readString(value.warning_code) ??
    readString(value.error_code) ??
    fallbackCode;
  const message =
    readString(value.message) ??
    readString(value.detail) ??
    readString(value.warning) ??
    readString(value.error);
  if (!message) {
    return null;
  }

  return {
    id: `${severity}:${code}:${message}`,
    severity,
    code,
    message,
    at: readString(value.at) ?? readString(value.emitted_at) ?? at,
    recoverable:
      readBoolean(value.recoverable) ?? readBoolean(value.retryable),
    source: "provider"
  };
}

function collectRetryEvents(
  traceEvents: WorkflowRuntimeTraceEvent[]
): ProviderTelemetryRetryEvent[] {
  const retries: ProviderTelemetryRetryEvent[] = [];

  for (const traceEvent of traceEvents) {
    const telemetry = readTelemetryRecord(traceEvent.event.payload);
    const execution =
      readUsageRecord(telemetry?.execution) ??
      readUsageRecord(telemetry?.provider_execution);
    const values = readUnknownList(
      execution?.retry_events ?? execution?.retryEvents
    );

    values.forEach((value, index) => {
      if (!isRecord(value)) {
        return;
      }
      const status =
        readString(value.status) ??
        readString(value.state) ??
        readString(value.outcome);
      if (!status) {
        return;
      }
      retries.push({
        id: `retry:${traceEvent.event.sequence}:${index}:${status}`,
        attempt: readCount(value, "attempt", "retry_count", "retryCount"),
        status,
        reason:
          readString(value.reason) ??
          readString(value.retry_reason) ??
          readString(value.message),
        at:
          readString(value.at) ??
          readString(value.emitted_at) ??
          readTraceTime(traceEvent)
      });
    });
  }

  return retries;
}

function collectFallbackPaths(
  traceEvents: WorkflowRuntimeTraceEvent[]
): ProviderTelemetryFallbackPath[] {
  const paths: ProviderTelemetryFallbackPath[] = [];

  for (const traceEvent of traceEvents) {
    const payload = traceEvent.event.payload;
    const telemetry = readTelemetryRecord(payload);
    const execution =
      readUsageRecord(telemetry?.execution) ??
      readUsageRecord(telemetry?.provider_execution);
    const explicitPaths = readUnknownList(
      execution?.fallback_paths ??
        execution?.fallbackPaths ??
        execution?.fallback_execution_paths
    );

    explicitPaths.forEach((value, index) => {
      if (!isRecord(value)) {
        return;
      }
      const reason =
        readString(value.reason) ??
        readString(value.decision_reason) ??
        readString(value.message);
      if (!reason) {
        return;
      }
      paths.push({
        id: `fallback:${traceEvent.event.sequence}:${index}:${reason}`,
        label:
          readString(value.label) ??
          readString(value.name) ??
          "Provider fallback",
        reason,
        source: readString(value.source) ?? readString(value.from),
        target: readString(value.target) ?? readString(value.to),
        at:
          readString(value.at) ??
          readString(value.emitted_at) ??
          readTraceTime(traceEvent)
      });
    });

    const decisionReason =
      readString(payload.decision_reason) ??
      readString(readUsageRecord(payload.edge)?.decision_reason);
    if (decisionReason !== "generation_unavailable") {
      continue;
    }
    paths.push({
      id: `fallback:${traceEvent.event.sequence}:generation-unavailable`,
      label: "Generation fallback",
      reason: "Provider execution was unavailable; the workflow continued locally.",
      source:
        readString(payload.transition_source) ??
        readString(readUsageRecord(payload.edge)?.source) ??
        "generation",
      target:
        readString(payload.transition_target) ??
        readString(readUsageRecord(payload.edge)?.target),
      at: readTraceTime(traceEvent)
    });
  }

  return dedupeFallbackPaths(paths);
}

function inferStreamingPreference(
  generationInput: WorkflowRuntimeTraceEvent | undefined
): boolean | null {
  if (!generationInput) {
    return null;
  }
  const input = readUsageRecord(generationInput.event.payload.generation_input);
  const request = readUsageRecord(input?.request);
  return readBoolean(request?.stream);
}

function deriveStreamingState({
  explicitStatus,
  generationMode,
  status
}: {
  explicitStatus: string | null;
  generationMode: ProviderTelemetryGenerationMode;
  status: ProviderTelemetryStatus;
}): ProviderTelemetryStreamingState {
  const normalizedStatus = explicitStatus?.toLowerCase();
  if (normalizedStatus === "failed" || status === "error") {
    return "failed";
  }
  if (generationMode === "non_streaming") {
    return "disabled";
  }
  if (normalizedStatus === "active" || normalizedStatus === "streaming") {
    return "active";
  }
  if (normalizedStatus === "completed" || normalizedStatus === "complete") {
    return "completed";
  }
  if (status === "idle") {
    return "idle";
  }
  if (status === "streaming" && generationMode === "streaming") {
    return "active";
  }
  if (status === "complete" && generationMode === "streaming") {
    return "completed";
  }
  return "unknown";
}

function normalizeGenerationMode(
  value: string | null
): ProviderTelemetryGenerationMode | null {
  switch (value?.toLowerCase().replaceAll("-", "_")) {
    case "stream":
    case "streamed":
    case "streaming":
      return "streaming";
    case "non_stream":
    case "non_streamed":
    case "non_streaming":
    case "synchronous":
      return "non_streaming";
    default:
      return null;
  }
}

function isProviderError(
  traceEvent: WorkflowRuntimeTraceEvent,
  subsystem: string
) {
  const payload = traceEvent.event.payload;
  return (
    subsystem === "generation_provider" ||
    subsystem === "provider" ||
    readString(payload.transition_source) === "generation" ||
    readString(payload.node) === "generation" ||
    String(payload.code ?? "").includes("provider")
  );
}

function dedupeIssues(issues: ProviderTelemetryIssue[]) {
  return [
    ...new Map(
      issues.map((issue) => [
        `${issue.severity}:${issue.code}`,
        issue
      ])
    ).values()
  ];
}

function dedupeFallbackPaths(paths: ProviderTelemetryFallbackPath[]) {
  return [...new Map(paths.map((path) => [path.id, path])).values()];
}

function readTraceTime(traceEvent: WorkflowRuntimeTraceEvent) {
  return readEventTimestamp(traceEvent.event) ?? traceEvent.receivedAt;
}

function readTelemetryRecord(
  payload: Record<string, unknown>
): Record<string, unknown> | null {
  return isRecord(payload.telemetry) ? payload.telemetry : null;
}

function readProviderTelemetry(
  payload: Record<string, unknown>,
  telemetry: Record<string, unknown> | null
): ProviderTelemetryProvider | null {
  return (
    readProviderFromRecord(telemetry?.provider, telemetry) ??
    readProviderFromRecord(payload.provider, payload) ??
    readProviderFromRecord(payload.provider_metadata, payload) ??
    null
  );
}

function readProviderFromRecord(
  value: unknown,
  fallback: Record<string, unknown> | null | undefined
): ProviderTelemetryProvider | null {
  const record = isRecord(value) ? value : fallback;
  const name =
    typeof value === "string"
      ? value
      : readString(record?.name) ??
        readString(record?.provider) ??
        readString(record?.provider_name);
  const model =
    readString(record?.model) ??
    readString(record?.model_name) ??
    readString(fallback?.model);
  const requestId =
    readString(record?.request_id) ??
    readString(record?.requestId) ??
    readString(fallback?.request_id) ??
    readString(fallback?.requestId);
  const responseId =
    readString(record?.response_id) ??
    readString(record?.responseId) ??
    readString(fallback?.response_id) ??
    readString(fallback?.responseId);
  const runtime =
    readString(record?.runtime) ??
    readString(record?.runtime_name) ??
    readString(fallback?.runtime);

  if (!name && !model && !requestId && !responseId && !runtime) {
    return null;
  }

  return {
    name,
    model,
    requestId,
    responseId,
    runtime
  };
}

function readTokenUsage(
  payload: Record<string, unknown>,
  telemetry: Record<string, unknown> | null
): ProviderTelemetryTokenUsage | null {
  const usage =
    readUsageRecord(telemetry?.token_usage) ??
    readUsageRecord(telemetry?.usage) ??
    readUsageRecord(payload.token_usage) ??
    readUsageRecord(payload.usage);

  if (!usage) {
    return null;
  }

  const inputTokens = readCount(usage, "input_tokens", "inputTokens", "prompt_tokens");
  const outputTokens = readCount(
    usage,
    "output_tokens",
    "outputTokens",
    "completion_tokens"
  );
  const totalTokens =
    readCount(usage, "total_tokens", "totalTokens") ??
    (inputTokens != null && outputTokens != null
      ? inputTokens + outputTokens
      : null);
  const cachedInputTokens = readCount(
    usage,
    "cached_input_tokens",
    "cachedInputTokens",
    "cached_tokens"
  );
  const reasoningTokens = readCount(
    usage,
    "reasoning_tokens",
    "reasoningTokens"
  );

  if (
    inputTokens === null &&
    outputTokens === null &&
    totalTokens === null &&
    cachedInputTokens === null &&
    reasoningTokens === null
  ) {
    return null;
  }

  return {
    inputTokens,
    outputTokens,
    totalTokens,
    cachedInputTokens,
    reasoningTokens,
    source: "provider"
  };
}

function readUsageRecord(value: unknown): Record<string, unknown> | null {
  return isRecord(value) ? value : null;
}

function mergeProviderTelemetry(
  current: ProviderTelemetryProvider,
  next: ProviderTelemetryProvider | null
): ProviderTelemetryProvider {
  if (!next) {
    return current;
  }

  return {
    name: next.name ?? current.name,
    model: next.model ?? current.model,
    requestId: next.requestId ?? current.requestId,
    responseId: next.responseId ?? current.responseId,
    runtime: next.runtime ?? current.runtime
  };
}

function mergeTokenUsage(
  current: ProviderTelemetryTokenUsage,
  next: ProviderTelemetryTokenUsage
): ProviderTelemetryTokenUsage {
  return {
    inputTokens: next.inputTokens ?? current.inputTokens,
    outputTokens: next.outputTokens ?? current.outputTokens,
    totalTokens: next.totalTokens ?? current.totalTokens,
    cachedInputTokens: next.cachedInputTokens ?? current.cachedInputTokens,
    reasoningTokens: next.reasoningTokens ?? current.reasoningTokens,
    source: "provider"
  };
}

function readPricing(
  payload: Record<string, unknown>,
  telemetry: Record<string, unknown> | null
): ProviderTelemetryPricing | null {
  const pricing = readUsageRecord(telemetry?.pricing) ?? readUsageRecord(payload.pricing);
  if (!pricing) {
    return null;
  }

  const inputCostPerMillionTokens =
    readFiniteNumber(
      pricing,
      "input_usd_per_million_tokens",
      "inputCostPerMillionTokens",
      "input_cost_per_million_tokens"
    ) ??
    multiplyNullable(
      readFiniteNumber(
        pricing,
        "input_usd_per_token",
        "inputCostPerToken",
        "input_cost_per_token"
      ),
      1_000_000
    );
  const outputCostPerMillionTokens =
    readFiniteNumber(
      pricing,
      "output_usd_per_million_tokens",
      "outputCostPerMillionTokens",
      "output_cost_per_million_tokens"
    ) ??
    multiplyNullable(
      readFiniteNumber(
        pricing,
        "output_usd_per_token",
        "outputCostPerToken",
        "output_cost_per_token"
      ),
      1_000_000
    );

  if (inputCostPerMillionTokens === null && outputCostPerMillionTokens === null) {
    return null;
  }

  return {
    currency: readString(pricing.currency) ?? "USD",
    inputCostPerMillionTokens,
    outputCostPerMillionTokens
  };
}

function readReportedCost(
  payload: Record<string, unknown>,
  telemetry: Record<string, unknown> | null
): ProviderTelemetryCostEstimate {
  const cost = readUsageRecord(telemetry?.cost) ?? readUsageRecord(payload.cost);
  if (!cost) {
    return unavailableCost;
  }

  const totalCost = readFiniteNumber(
    cost,
    "total_usd",
    "estimated_usd",
    "amount_usd",
    "totalCost",
    "total",
    "amount"
  );

  return {
    currency: readString(cost.currency) ?? "USD",
    inputCost: readFiniteNumber(cost, "input_usd", "inputCost"),
    outputCost: readFiniteNumber(cost, "output_usd", "outputCost"),
    totalCost,
    source: totalCost == null ? "unavailable" : "provider_reported"
  };
}

function estimateCostFromPricing(
  usage: ProviderTelemetryTokenUsage,
  pricing: ProviderTelemetryPricing | null
): ProviderTelemetryCostEstimate {
  if (!pricing || usage.source !== "provider") {
    return unavailableCost;
  }

  const inputCost =
    usage.inputTokens != null && pricing.inputCostPerMillionTokens != null
      ? (usage.inputTokens / 1_000_000) * pricing.inputCostPerMillionTokens
      : null;
  const outputCost =
    usage.outputTokens != null && pricing.outputCostPerMillionTokens != null
      ? (usage.outputTokens / 1_000_000) * pricing.outputCostPerMillionTokens
      : null;
  const totalCost =
    inputCost != null || outputCost != null
      ? (inputCost ?? 0) + (outputCost ?? 0)
      : null;

  return {
    currency: pricing.currency,
    inputCost,
    outputCost,
    totalCost,
    source: totalCost == null ? "unavailable" : "pricing_metadata"
  };
}

function readCount(record: Record<string, unknown>, ...keys: string[]) {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "number" && Number.isFinite(value) && value >= 0) {
      return Math.round(value);
    }
  }
  return null;
}

function readFiniteNumber(record: Record<string, unknown>, ...keys: string[]) {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "number" && Number.isFinite(value) && value >= 0) {
      return value;
    }
  }
  return null;
}

function multiplyNullable(value: number | null, multiplier: number) {
  return value == null ? null : value * multiplier;
}

function readString(value: unknown) {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function readBoolean(value: unknown) {
  return typeof value === "boolean" ? value : null;
}

function readUnknownList(value: unknown): unknown[] {
  if (Array.isArray(value)) {
    return value;
  }
  return value == null ? [] : [value];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function formatCount(value: number) {
  return new Intl.NumberFormat("en", {
    maximumFractionDigits: 0
  }).format(value);
}

function formatDuration(valueMs: number) {
  if (valueMs < 1000) {
    return `${Math.round(valueMs)} ms`;
  }
  return `${(valueMs / 1000).toFixed(1)} s`;
}

function formatCost(value: number, currency: string) {
  const precision = value < 0.01 ? 4 : 2;
  return new Intl.NumberFormat("en", {
    currency,
    maximumFractionDigits: precision,
    minimumFractionDigits: precision,
    style: "currency"
  }).format(value);
}

function summarizeLifecycle(lifecycle: ProviderTelemetryLifecycleStep[]) {
  const completed = lifecycle.filter((step) =>
    ["complete", "failed"].includes(step.state)
  ).length;
  return `${completed} of ${lifecycle.length} lifecycle stages`;
}

function formatGenerationMode(mode: ProviderTelemetryGenerationMode) {
  switch (mode) {
    case "streaming":
      return "Streaming generation";
    case "non_streaming":
      return "Non-streaming generation";
    default:
      return "Generation mode unavailable";
  }
}

function formatStreamingState(state: ProviderTelemetryStreamingState) {
  switch (state) {
    case "active":
      return "Stream active";
    case "completed":
      return "Stream completed";
    case "failed":
      return "Stream failed";
    case "disabled":
      return "Streaming disabled";
    case "idle":
      return "Stream idle";
    default:
      return "Stream status unavailable";
  }
}

function summarizeIssues(execution: ProviderTelemetryExecution) {
  const issueCount = execution.errors.length + execution.warnings.length;
  if (issueCount === 0) {
    return "No provider issues";
  }
  return `${execution.errors.length} ${
    execution.errors.length === 1 ? "error" : "errors"
  } / ${execution.warnings.length} ${
    execution.warnings.length === 1 ? "warning" : "warnings"
  }`;
}
