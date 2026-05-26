import { readEventTimestamp } from "./assistant-stream";
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

export type ProviderTelemetryModel = {
  status: ProviderTelemetryStatus;
  provider: ProviderTelemetryProvider;
  tokenUsage: ProviderTelemetryTokenUsage;
  pricing: ProviderTelemetryPricing | null;
  cost: ProviderTelemetryCostEstimate;
  timing: ProviderTelemetryTiming;
  lifecycle: ProviderTelemetryLifecycleStep[];
  stream: ProviderTelemetryStreamSummary;
  summary: {
    providerLabel: string;
    modelLabel: string;
    tokenLabel: string;
    costLabel: string;
    latencyLabel: string;
    streamLabel: string;
    lifecycleLabel: string;
  };
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
      tokenUsage: emptyTokenUsage
    });
  }

  let provider = emptyProvider;
  let tokenUsage = emptyTokenUsage;
  let pricing: ProviderTelemetryPricing | null = null;
  let reportedCost: ProviderTelemetryCostEstimate | null = null;
  let streamedCharacterCount = 0;
  let tokenDeltaCount = 0;

  for (const traceEvent of traceEvents) {
    const payload = traceEvent.event.payload;
    const telemetry = readTelemetryRecord(payload);
    const nextProvider = readProviderTelemetry(payload, telemetry);
    const nextUsage = readTokenUsage(payload, telemetry);
    const nextPricing = readPricing(payload, telemetry);
    const nextCost = readReportedCost(payload, telemetry);

    provider = mergeProviderTelemetry(provider, nextProvider);

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

  return buildTelemetryModel({
    cost,
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
  provider,
  pricing,
  status,
  stream,
  timing,
  tokenUsage
}: {
  cost: ProviderTelemetryCostEstimate;
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
      lifecycleLabel: summarizeLifecycle(lifecycle)
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
