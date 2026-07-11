import {
  readEventTimestamp,
  readWorkflowMetadata
} from "./assistant-stream";
import type {
  ProviderTelemetryGenerationMode,
  ProviderTelemetryModel,
  ProviderTelemetryPricing,
  ProviderTelemetryStatus
} from "./provider-telemetry";
import type { WorkflowRuntimeTraceEvent } from "./workflow-runtime";

export type CreativeCostRunKind = "generation" | "artifact_refinement";

export type CreativeCostRunRecord = {
  id: string;
  status: Extract<ProviderTelemetryStatus, "complete" | "error">;
  completedAt: string | null;
  kind: CreativeCostRunKind;
  providerName: string | null;
  modelName: string | null;
  generationMode: ProviderTelemetryGenerationMode;
  pricing: ProviderTelemetryPricing | null;
  inputTokens: number | null;
  outputTokens: number | null;
  totalTokens: number | null;
  cost: number | null;
  currency: string;
  costSource: ProviderTelemetryModel["cost"]["source"];
  durationMs: number | null;
  retryCount: number | null;
  fallbackCount: number;
  retryCost: number | null;
  fallbackCost: number | null;
  artifactCount: number;
  refinementCount: number;
  critiqueCount: number;
  reviewCount: number;
};

export type CreativeCostEstimate = {
  state: "empty" | "pricing_unavailable" | "ready";
  providerName: string | null;
  modelName: string | null;
  generationMode: ProviderTelemetryGenerationMode;
  promptTokens: number | null;
  contextTokens: number | null;
  requestedArtifactCount: number;
  includesReview: boolean;
  includesRefinement: boolean;
  inputTokenRange: [number, number] | null;
  outputTokenRange: [number, number] | null;
  costRange: [number, number] | null;
  currency: string;
  detail: string;
  assumptions: string[];
};

export type CreativeCostCurrentRun = {
  state: "idle" | "running" | "complete" | "error";
  providerName: string | null;
  modelName: string | null;
  generationMode: ProviderTelemetryGenerationMode;
  inputTokens: number | null;
  outputTokens: number | null;
  totalTokens: number | null;
  cost: number | null;
  currency: string;
  costSource: ProviderTelemetryModel["cost"]["source"];
  durationMs: number | null;
  retryCount: number | null;
  fallbackCount: number;
  retryCost: number | null;
  fallbackCost: number | null;
  artifactCount: number;
  refinementCount: number;
  critiqueCount: number;
  reviewCount: number;
};

export type CreativeCostSessionSummary = {
  runCount: number;
  generationCount: number;
  refinementCount: number;
  inputTokens: number | null;
  outputTokens: number | null;
  totalTokens: number | null;
  tokenedRunCount: number;
  artifactCount: number;
  critiqueCount: number;
  reviewCount: number;
  totalCost: number | null;
  currency: string;
  costedRunCount: number;
  coverage: "none" | "partial" | "complete";
  averagePerGeneration: number | null;
  averagePerArtifact: number | null;
};

export type CreativeCostIntelligenceModel = {
  estimate: CreativeCostEstimate;
  current: CreativeCostCurrentRun;
  session: CreativeCostSessionSummary;
};

export type BuildCreativeCostIntelligenceInput = {
  draftPrompt: string;
  providerTelemetry: ProviderTelemetryModel;
  retrievalChunkCount: number;
  runHistory: CreativeCostRunRecord[];
  traceEvents: WorkflowRuntimeTraceEvent[];
};

const approximateCharactersPerToken = 4;
const approximateTokensPerRetrievalChunk = 180;

export function buildCreativeCostIntelligenceModel({
  draftPrompt,
  providerTelemetry,
  retrievalChunkCount,
  runHistory,
  traceEvents
}: BuildCreativeCostIntelligenceInput): CreativeCostIntelligenceModel {
  const currentRecord = buildCreativeCostRunRecord({
    providerTelemetry,
    traceEvents
  });
  const completedRuns = dedupeRuns([
    ...runHistory,
    ...(currentRecord ? [currentRecord] : [])
  ]);

  return {
    estimate: buildEstimate({
      draftPrompt,
      fallbackRun: completedRuns.at(-1) ?? null,
      providerTelemetry,
      retrievalChunkCount
    }),
    current: buildCurrentRun(providerTelemetry, currentRecord),
    session: buildSessionSummary(completedRuns)
  };
}

export function buildCreativeCostRunRecord({
  providerTelemetry,
  traceEvents
}: {
  providerTelemetry: ProviderTelemetryModel;
  traceEvents: WorkflowRuntimeTraceEvent[];
}): CreativeCostRunRecord | null {
  if (
    traceEvents.length === 0 ||
    (providerTelemetry.status !== "complete" &&
      providerTelemetry.status !== "error")
  ) {
    return null;
  }

  const latestWorkflow = findLatestWorkflowMetadata(traceEvents);
  const firstTrace = traceEvents[0];
  const lastTrace = traceEvents.at(-1);
  const completedAt =
    providerTelemetry.execution.requestCompletedAt ??
    (lastTrace ? readTraceTimestamp(lastTrace) : null);
  const requestIdentity =
    providerTelemetry.provider.requestId ??
    providerTelemetry.provider.responseId ??
    `${firstTrace.event.sequence}:${firstTrace.receivedAt}`;
  const selectedArtifactRefinement = traceEvents.some(({ event }) =>
    hasArtifactRefinementInput(event.payload)
  );
  const retryCost = readLatestOptionalCost(traceEvents, [
    "retry_cost",
    "retry_cost_usd",
    "retryCost",
    "retryCostUsd"
  ]);
  const fallbackCost = readLatestOptionalCost(traceEvents, [
    "fallback_cost",
    "fallback_cost_usd",
    "fallbackCost",
    "fallbackCostUsd"
  ]);

  return {
    id: `${requestIdentity}:${completedAt ?? lastTrace?.receivedAt ?? "complete"}`,
    status: providerTelemetry.status,
    completedAt,
    kind: selectedArtifactRefinement ? "artifact_refinement" : "generation",
    providerName: providerTelemetry.provider.name,
    modelName: providerTelemetry.provider.model,
    generationMode: providerTelemetry.execution.generationMode,
    pricing: providerTelemetry.pricing,
    inputTokens: providerTelemetry.tokenUsage.inputTokens,
    outputTokens: providerTelemetry.tokenUsage.outputTokens,
    totalTokens: providerTelemetry.tokenUsage.totalTokens,
    cost: providerTelemetry.cost.totalCost,
    currency: providerTelemetry.cost.currency,
    costSource: providerTelemetry.cost.source,
    durationMs:
      providerTelemetry.execution.requestDurationMs ??
      providerTelemetry.timing.generationDurationMs,
    retryCount: providerTelemetry.execution.retryCount,
    fallbackCount: providerTelemetry.execution.fallbackPaths.length,
    retryCost,
    fallbackCost,
    artifactCount: Math.max(
      latestWorkflow?.artifact_count ?? 0,
      countArtifactEvents(traceEvents)
    ),
    refinementCount: Math.max(
      latestWorkflow?.refinement_count ?? 0,
      countEventTypes(traceEvents, "refinement_completed"),
      selectedArtifactRefinement ? 1 : 0
    ),
    critiqueCount: Math.max(
      latestWorkflow?.artifact_critique_count ?? 0,
      countCritiqueEvents(traceEvents)
    ),
    reviewCount: Math.max(
      countReviewEvents(traceEvents),
      latestWorkflow?.review_outcome ? 1 : 0
    )
  };
}

function buildEstimate({
  draftPrompt,
  fallbackRun,
  providerTelemetry,
  retrievalChunkCount
}: {
  draftPrompt: string;
  fallbackRun: CreativeCostRunRecord | null;
  providerTelemetry: ProviderTelemetryModel;
  retrievalChunkCount: number;
}): CreativeCostEstimate {
  const prompt = draftPrompt.trim();
  const pricing = providerTelemetry.pricing ?? fallbackRun?.pricing ?? null;
  const providerName =
    providerTelemetry.provider.name ?? fallbackRun?.providerName ?? null;
  const modelName =
    providerTelemetry.provider.model ?? fallbackRun?.modelName ?? null;
  const generationMode =
    providerTelemetry.execution.generationMode !== "unknown"
      ? providerTelemetry.execution.generationMode
      : (fallbackRun?.generationMode ?? "unknown");

  if (!prompt) {
    return {
      state: "empty",
      providerName,
      modelName,
      generationMode,
      promptTokens: null,
      contextTokens: null,
      requestedArtifactCount: 1,
      includesReview: false,
      includesRefinement: false,
      inputTokenRange: null,
      outputTokenRange: null,
      costRange: null,
      currency: pricing?.currency ?? "USD",
      detail: "Add a prompt to preview the likely generation scope.",
      assumptions: []
    };
  }

  const promptTokens = Math.max(
    1,
    Math.ceil(prompt.length / approximateCharactersPerToken)
  );
  const contextTokens =
    retrievalChunkCount > 0
      ? retrievalChunkCount * approximateTokensPerRetrievalChunk
      : 0;
  const requestedArtifactCount = inferRequestedArtifactCount(prompt);
  const includesRefinement = /\b(refine|refinement|revise|improve|fix)\b/i.test(
    prompt
  );
  const includesReview =
    requestedArtifactCount > 1 ||
    /\b(critique|review|compare|evaluate|rank|best)\b/i.test(prompt);
  const inputLower = promptTokens + Math.round(contextTokens * 0.75);
  const inputUpper =
    promptTokens +
    Math.ceil(contextTokens * 1.25) +
    (includesRefinement ? 450 : 180);
  const baseOutputLower = 450 * requestedArtifactCount;
  const baseOutputUpper = 1200 * requestedArtifactCount;
  const outputLower =
    baseOutputLower + (includesReview ? 120 * requestedArtifactCount : 0);
  const outputUpper =
    baseOutputUpper +
    (includesReview ? 320 * requestedArtifactCount : 0) +
    (includesRefinement ? 300 : 0);
  const inputTokenRange: [number, number] = [inputLower, inputUpper];
  const outputTokenRange: [number, number] = [outputLower, outputUpper];
  const costRange = calculateCostRange(
    inputTokenRange,
    outputTokenRange,
    pricing
  );
  const assumptions = [
    "Token counts are approximate and can change after retrieval and prompt rendering.",
    `${requestedArtifactCount} requested ${requestedArtifactCount === 1 ? "artifact" : "artifacts"} inferred from the draft.`
  ];

  if (contextTokens > 0) {
    assumptions.push(
      `${retrievalChunkCount} current retrieval ${retrievalChunkCount === 1 ? "chunk" : "chunks"} used as a context allowance.`
    );
  }
  if (includesReview) {
    assumptions.push("Review or comparison overhead included.");
  }
  if (includesRefinement) {
    assumptions.push("Refinement context overhead included.");
  }

  if (!costRange) {
    return {
      state: "pricing_unavailable",
      providerName,
      modelName,
      generationMode,
      promptTokens,
      contextTokens,
      requestedArtifactCount,
      includesReview,
      includesRefinement,
      inputTokenRange,
      outputTokenRange,
      costRange: null,
      currency: pricing?.currency ?? "USD",
      detail:
        "A token range is available, but cost requires complete provider pricing metadata.",
      assumptions
    };
  }

  return {
    state: "ready",
    providerName,
    modelName,
    generationMode,
    promptTokens,
    contextTokens,
    requestedArtifactCount,
    includesReview,
    includesRefinement,
    inputTokenRange,
    outputTokenRange,
    costRange,
    currency: pricing?.currency ?? "USD",
    detail:
      "Bounded estimate from the draft, current context allowance, and last known provider pricing.",
    assumptions
  };
}

function buildCurrentRun(
  telemetry: ProviderTelemetryModel,
  record: CreativeCostRunRecord | null
): CreativeCostCurrentRun {
  const state =
    telemetry.status === "streaming"
      ? "running"
      : telemetry.status === "complete"
        ? "complete"
        : telemetry.status === "error"
          ? "error"
          : "idle";

  return {
    state,
    providerName: telemetry.provider.name,
    modelName: telemetry.provider.model,
    generationMode: telemetry.execution.generationMode,
    inputTokens: telemetry.tokenUsage.inputTokens,
    outputTokens: telemetry.tokenUsage.outputTokens,
    totalTokens: telemetry.tokenUsage.totalTokens,
    cost: telemetry.cost.totalCost,
    currency: telemetry.cost.currency,
    costSource: telemetry.cost.source,
    durationMs:
      telemetry.execution.requestDurationMs ??
      telemetry.timing.generationDurationMs,
    retryCount: telemetry.execution.retryCount,
    fallbackCount: telemetry.execution.fallbackPaths.length,
    retryCost: record?.retryCost ?? null,
    fallbackCost: record?.fallbackCost ?? null,
    artifactCount: record?.artifactCount ?? 0,
    refinementCount: record?.refinementCount ?? 0,
    critiqueCount: record?.critiqueCount ?? 0,
    reviewCount: record?.reviewCount ?? 0
  };
}

function buildSessionSummary(
  runs: CreativeCostRunRecord[]
): CreativeCostSessionSummary {
  const costedRuns = runs.filter((run) => run.cost != null);
  const tokenedRuns = runs.filter((run) => run.totalTokens != null);
  const currencies = new Set(costedRuns.map((run) => run.currency));
  const hasConsistentCurrency = currencies.size <= 1;
  const totalCost =
    costedRuns.length > 0 && hasConsistentCurrency
      ? costedRuns.reduce((total, run) => total + (run.cost ?? 0), 0)
      : null;
  const costedArtifactCount = costedRuns.reduce(
    (total, run) => total + run.artifactCount,
    0
  );

  return {
    runCount: runs.length,
    generationCount: runs.length,
    refinementCount: runs.reduce(
      (total, run) => total + run.refinementCount,
      0
    ),
    inputTokens:
      tokenedRuns.length > 0
        ? tokenedRuns.reduce((total, run) => total + (run.inputTokens ?? 0), 0)
        : null,
    outputTokens:
      tokenedRuns.length > 0
        ? tokenedRuns.reduce((total, run) => total + (run.outputTokens ?? 0), 0)
        : null,
    totalTokens:
      tokenedRuns.length > 0
        ? tokenedRuns.reduce((total, run) => total + (run.totalTokens ?? 0), 0)
        : null,
    tokenedRunCount: tokenedRuns.length,
    artifactCount: runs.reduce((total, run) => total + run.artifactCount, 0),
    critiqueCount: runs.reduce((total, run) => total + run.critiqueCount, 0),
    reviewCount: runs.reduce((total, run) => total + run.reviewCount, 0),
    totalCost,
    currency: costedRuns[0]?.currency ?? "USD",
    costedRunCount: costedRuns.length,
    coverage:
      costedRuns.length === 0
        ? "none"
        : costedRuns.length === runs.length && hasConsistentCurrency
          ? "complete"
          : "partial",
    averagePerGeneration:
      totalCost != null && costedRuns.length > 0
        ? totalCost / costedRuns.length
        : null,
    averagePerArtifact:
      totalCost != null && costedArtifactCount > 0
        ? totalCost / costedArtifactCount
        : null
  };
}

function calculateCostRange(
  inputRange: [number, number],
  outputRange: [number, number],
  pricing: ProviderTelemetryPricing | null
): [number, number] | null {
  if (
    !pricing ||
    pricing.inputCostPerMillionTokens == null ||
    pricing.outputCostPerMillionTokens == null
  ) {
    return null;
  }

  return [
    (inputRange[0] / 1_000_000) * pricing.inputCostPerMillionTokens +
      (outputRange[0] / 1_000_000) * pricing.outputCostPerMillionTokens,
    (inputRange[1] / 1_000_000) * pricing.inputCostPerMillionTokens +
      (outputRange[1] / 1_000_000) * pricing.outputCostPerMillionTokens
  ];
}

function inferRequestedArtifactCount(prompt: string) {
  const numericMatch = prompt.match(
    /\b([2-8])\b(?:\s+\w+){0,2}\s+(?:artifacts?|alternatives?|options?|variations?|versions?|concepts?)\b/i
  );
  if (numericMatch) {
    return Number(numericMatch[1]);
  }

  const wordCounts: Record<string, number> = {
    two: 2,
    three: 3,
    four: 4,
    five: 5,
    six: 6,
    seven: 7,
    eight: 8
  };
  const wordMatch = prompt.match(
    /\b(two|three|four|five|six|seven|eight)\b(?:\s+\w+){0,2}\s+(?:artifacts?|alternatives?|options?|variations?|versions?|concepts?)\b/i
  );
  if (wordMatch) {
    return wordCounts[wordMatch[1].toLowerCase()] ?? 1;
  }

  return /\b(multiple|several|alternatives|variations|options)\b/i.test(prompt)
    ? 2
    : 1;
}

function findLatestWorkflowMetadata(traceEvents: WorkflowRuntimeTraceEvent[]) {
  for (let index = traceEvents.length - 1; index >= 0; index -= 1) {
    const metadata = readWorkflowMetadata(traceEvents[index].event);
    if (metadata) {
      return metadata;
    }
  }
  return null;
}

function countArtifactEvents(traceEvents: WorkflowRuntimeTraceEvent[]) {
  return traceEvents.reduce((count, { event }) => {
    if (event.event_type !== "artifact_extracted") {
      return count;
    }
    const artifactCount = readFiniteNumber(event.payload.artifact_count);
    return Math.max(count, artifactCount ?? count + 1);
  }, 0);
}

function countCritiqueEvents(traceEvents: WorkflowRuntimeTraceEvent[]) {
  return traceEvents.filter(
    ({ event }) =>
      event.event_type === "artifact_critique" &&
      (event.payload.code === "artifact_scored" ||
        event.payload.code === "critique_completed")
  ).length;
}

function countReviewEvents(traceEvents: WorkflowRuntimeTraceEvent[]) {
  return traceEvents.filter(
    ({ event }) =>
      event.event_type === "review_passed" ||
      event.event_type === "review_failed"
  ).length;
}

function countEventTypes(
  traceEvents: WorkflowRuntimeTraceEvent[],
  eventType: WorkflowRuntimeTraceEvent["event"]["event_type"]
) {
  return traceEvents.filter(({ event }) => event.event_type === eventType).length;
}

function readLatestOptionalCost(
  traceEvents: WorkflowRuntimeTraceEvent[],
  keys: string[]
) {
  for (let index = traceEvents.length - 1; index >= 0; index -= 1) {
    const payload = traceEvents[index].event.payload;
    const telemetry = isRecord(payload.telemetry) ? payload.telemetry : null;
    const execution =
      (telemetry && isRecord(telemetry.execution)
        ? telemetry.execution
        : null) ??
      (isRecord(payload.execution) ? payload.execution : null);

    for (const source of [execution, telemetry, payload]) {
      if (!source) {
        continue;
      }
      for (const key of keys) {
        const value = readFiniteNumber(source[key]);
        if (value != null) {
          return value;
        }
      }
    }
  }
  return null;
}

function hasArtifactRefinementInput(payload: Record<string, unknown>) {
  const promptInput = isRecord(payload.prompt_input)
    ? payload.prompt_input
    : null;
  const userInput =
    promptInput && isRecord(promptInput.user_input)
      ? promptInput.user_input
      : null;
  return userInput ? isRecord(userInput.artifact_refinement) : false;
}

function readTraceTimestamp(traceEvent: WorkflowRuntimeTraceEvent) {
  return readEventTimestamp(traceEvent.event) ?? traceEvent.receivedAt;
}

function readFiniteNumber(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? value
    : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function dedupeRuns(runs: CreativeCostRunRecord[]) {
  return Array.from(new Map(runs.map((run) => [run.id, run])).values());
}
