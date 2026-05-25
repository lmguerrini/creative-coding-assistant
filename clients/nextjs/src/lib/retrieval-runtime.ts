import type {
  RetrievalChunkSummary,
  RetrievalFreshness,
  RetrievalQuality,
  RetrievalSourceSummary,
  RetrievalSummary
} from "./assistant-client";
import { readEventTimestamp, type AssistantStreamEvent } from "./assistant-stream";
import {
  createWorkstationError,
  parseSubsystemErrorPayload,
  type WorkstationError
} from "./workstation-errors";
import type { WorkflowRuntimeTraceEvent } from "./workflow-runtime";

export type RetrievalRuntimeRequest = {
  query: string | null;
  domains: string[];
  domainLabels: string[];
  limit: number | null;
  sourceFilter: string | null;
  sourceTypeFilter: string | null;
  publisherFilter: string | null;
  filterLabels: string[];
};

export type RetrievalRuntimeSummary = {
  state: RetrievalSummary["state"];
  status: string;
  headline: string;
  detail: string;
  providerLabel: string;
  sourceCount: number;
  chunkCount: number;
  domainCount: number;
  domainLabels: string[];
  qualityLabel: string;
  freshnessLabel: string;
  coverageLabel: string;
  warning: string | null;
  updatedAt: string | null;
  error: WorkstationError | null;
};

export type RetrievalRuntimeModel = {
  request: RetrievalRuntimeRequest;
  sources: RetrievalSourceSummary[];
  summary: RetrievalRuntimeSummary;
};

type ParsedRetrievalRequest = {
  query: string | null;
  domains: string[];
  limit: number | null;
  sourceFilter: string | null;
  sourceTypeFilter: string | null;
  publisherFilter: string | null;
};

type ParsedRetrievalChunk = {
  sourceId: string;
  title: string;
  domain: string;
  publisher: string;
  sourceType: string;
  href: string;
  host: string;
  score: number | null;
  snippet: string;
  chunkIndex: number;
};

type ParsedRetrievalContext = {
  emittedAt: string | null;
  provider: string | null;
  request: ParsedRetrievalRequest;
  chunks: ParsedRetrievalChunk[];
  error: WorkstationError | null;
};

const retrievalStateStatusLabels = {
  available: "Grounded",
  pending: "Searching",
  empty: "No matches",
  unavailable: "Unavailable",
  error: "Retrieval failed"
} satisfies Record<RetrievalSummary["state"], string>;

const retrievalQualityOrder = {
  high: 3,
  medium: 2,
  low: 1,
  unknown: 0
} satisfies Record<RetrievalQuality, number>;

const domainLabelOverrides: Record<string, string> = {
  ableton_live: "Ableton Live",
  blender_geometry_nodes: "Blender Geometry Nodes",
  blender_python_api: "Blender Python API",
  cables_gl: "Cables.gl",
  canvas_2d: "Canvas 2D",
  comfyui: "ComfyUI",
  glsl: "GLSL",
  gsap: "GSAP",
  hydra: "Hydra",
  madmapper: "MadMapper",
  matter_js: "Matter.js",
  ml5_js: "ml5.js",
  notch: "Notch",
  openframeworks: "openFrameworks",
  openrndr: "OPENRNDR",
  p5_js: "p5.js",
  p5_sound: "p5.sound",
  pixi_js: "PixiJS",
  pure_data: "Pure Data",
  r3f: "R3F",
  react_three_fiber: "React Three Fiber",
  shadertoy: "Shadertoy",
  sonic_pi: "Sonic Pi",
  stable_diffusion_workflows: "Stable Diffusion Workflows",
  supercollider: "SuperCollider",
  tensorflow_js: "TensorFlow.js",
  three_js: "Three.js",
  tone_js: "Tone.js",
  touchdesigner: "TouchDesigner",
  unreal_blueprints: "Unreal Blueprints",
  vcv_rack: "VCV Rack",
  vvvv: "vvvv",
  web_audio_api: "Web Audio API",
  webgpu_wgsl: "WebGPU / WGSL"
};

const sourceTypeLabels: Record<string, string> = {
  api_reference: "API reference",
  examples: "Examples",
  guide: "Guide",
  specification: "Specification"
};

export function buildRetrievalRuntimeModel(
  baseRetrieval: RetrievalSummary,
  traceEvents: WorkflowRuntimeTraceEvent[]
): RetrievalRuntimeModel {
  const latestRequestEvent = findLatestRetrievalEvent(traceEvents, "retrieval_requested");
  const latestCompletedEvent = findLatestRetrievalEvent(
    traceEvents,
    "retrieval_completed"
  );

  if (
    latestRequestEvent &&
    (!latestCompletedEvent ||
      latestRequestEvent.event.sequence > latestCompletedEvent.event.sequence)
  ) {
    return buildPendingRuntimeModel(baseRetrieval, latestRequestEvent.event);
  }

  if (latestCompletedEvent) {
    return buildCompletedRuntimeModel(
      baseRetrieval,
      latestCompletedEvent.event,
      latestRequestEvent?.event ?? null
    );
  }

  return buildFallbackRuntimeModel(baseRetrieval);
}

function buildFallbackRuntimeModel(
  baseRetrieval: RetrievalSummary
): RetrievalRuntimeModel {
  const request = buildRuntimeRequest({
    domains: baseRetrieval.requestedDomains,
    limit: null,
    publisherFilter: null,
    query: baseRetrieval.query,
    sourceFilter: null,
    sourceTypeFilter: null
  });
  const sources = normalizeFallbackSources(baseRetrieval.sources);
  const summary = buildRuntimeSummary({
    baseRetrieval,
    detail:
      baseRetrieval.detail ||
      "Retrieval context is available in the saved workspace snapshot.",
    error: baseRetrieval.error ?? buildFallbackRetrievalError(baseRetrieval),
    emittedAt: latestUpdatedAt(sources),
    includeBaseWarning: true,
    provider: baseRetrieval.source,
    request,
    sources,
    state: baseRetrieval.state,
    status: baseRetrieval.status || retrievalStateStatusLabels[baseRetrieval.state]
  });

  return {
    request,
    sources,
    summary: {
      ...summary,
      warning: baseRetrieval.warning ?? summary.warning
    }
  };
}

function buildPendingRuntimeModel(
  baseRetrieval: RetrievalSummary,
  event: AssistantStreamEvent
): RetrievalRuntimeModel {
  const request = parseRetrievalRequestFromEvent(event) ?? buildFallbackRuntimeModel(baseRetrieval).request;
  const detail =
    request.query && request.domainLabels.length > 0
      ? `Waiting for retrieval results for “${request.query}” across ${request.domainLabels.join(", ")}.`
      : request.query
        ? `Waiting for retrieval results for “${request.query}”.`
        : request.domainLabels.length > 0
          ? `Waiting for retrieval results across ${request.domainLabels.join(", ")}.`
          : "Waiting for retrieval results from the knowledge base.";

  return {
    request,
    sources: [],
    summary: buildRuntimeSummary({
      baseRetrieval,
      detail,
      error: null,
      emittedAt: readEventTimestamp(event),
      includeBaseWarning: false,
      provider: baseRetrieval.source,
      request,
      sources: [],
      state: "pending",
      status: retrievalStateStatusLabels.pending
    })
  };
}

function buildCompletedRuntimeModel(
  baseRetrieval: RetrievalSummary,
  completedEvent: AssistantStreamEvent,
  requestEvent: AssistantStreamEvent | null
): RetrievalRuntimeModel {
  const context = parseRetrievalContextFromEvent(completedEvent);
  if (!context) {
    return buildFallbackRuntimeModel(baseRetrieval);
  }

  if (context.error) {
    return buildErroredRuntimeModel({
      baseRetrieval,
      emittedAt: context.emittedAt,
      error: context.error,
      provider: context.provider ?? baseRetrieval.source,
      request: buildRuntimeRequest(context.request)
    });
  }

  const request =
    context.request.query ||
    context.request.domains.length > 0 ||
    context.request.limit != null
      ? buildRuntimeRequest(context.request)
      : requestEvent
        ? parseRetrievalRequestFromEvent(requestEvent) ??
          buildRuntimeRequest(context.request)
        : buildRuntimeRequest(context.request);
  const sources = buildRuntimeSources(baseRetrieval.sources, context.chunks, request);
  const state: RetrievalSummary["state"] = sources.length > 0 ? "available" : "empty";
  const detail =
    state === "available"
      ? `${formatRetrievalProviderLabel(context.provider)} returned ${countLabel(
          sources.reduce((total, source) => total + source.chunks.length, 0),
          "chunk"
        )} across ${countLabel(sources.length, "source")}.`
      : request.query
        ? `No retrieval chunks were returned for “${request.query}”.`
        : "No retrieval chunks were returned for this request.";

  return {
    request,
    sources,
    summary: buildRuntimeSummary({
      baseRetrieval,
      detail,
      error: null,
      emittedAt: context.emittedAt,
      includeBaseWarning: false,
      provider: context.provider ?? baseRetrieval.source,
      request,
      sources,
      state,
      status: retrievalStateStatusLabels[state]
    })
  };
}

function buildRuntimeSummary({
  baseRetrieval,
  detail,
  error,
  emittedAt,
  includeBaseWarning,
  provider,
  request,
  sources,
  state,
  status
}: {
  baseRetrieval: RetrievalSummary;
  detail: string;
  error: WorkstationError | null;
  emittedAt: string | null;
  includeBaseWarning: boolean;
  provider: string | null;
  request: RetrievalRuntimeRequest;
  sources: RetrievalSourceSummary[];
  state: RetrievalSummary["state"];
  status: string;
}): RetrievalRuntimeSummary {
  const chunkCount = sources.reduce((total, source) => total + source.chunks.length, 0);
  const domainLabels = Array.from(
    new Set(sources.map((source) => source.domainLabel).filter(Boolean))
  );
  const topScore = sources.reduce<number | null>((currentTop, source) => {
    if (source.score == null) {
      return currentTop;
    }

    return currentTop == null ? source.score : Math.max(currentTop, source.score);
  }, null);
  const freshnessLabel = buildFreshnessSummaryLabel(sources);
  const warning =
    buildMissingDomainWarning(request, domainLabels) ??
    buildStaleSourceWarning(sources) ??
    (includeBaseWarning ? baseRetrieval.warning : null);

  return {
    state,
    status,
    headline: buildSummaryHeadline(state, sources.length, chunkCount, request.query),
    detail,
    providerLabel: formatRetrievalProviderLabel(provider),
    sourceCount: sources.length,
    chunkCount,
    domainCount: domainLabels.length,
    domainLabels,
    qualityLabel:
      topScore != null ? `Top score ${formatScorePercent(topScore)}` : "No relevance score",
    freshnessLabel,
    coverageLabel:
      domainLabels.length > 0
        ? countLabel(domainLabels.length, "domain")
        : request.domainLabels.length > 0
          ? `Requested ${countLabel(request.domainLabels.length, "domain")}`
          : "No domain filter",
    warning,
    updatedAt: emittedAt ?? latestUpdatedAt(sources),
    error
  };
}

function buildErroredRuntimeModel({
  baseRetrieval,
  emittedAt,
  error,
  provider,
  request
}: {
  baseRetrieval: RetrievalSummary;
  emittedAt: string | null;
  error: WorkstationError;
  provider: string | null;
  request: RetrievalRuntimeRequest;
}): RetrievalRuntimeModel {
  return {
    request,
    sources: [],
    summary: buildRuntimeSummary({
      baseRetrieval,
      detail: error.userMessage,
      error,
      emittedAt,
      includeBaseWarning: false,
      provider,
      request,
      sources: [],
      state: "error",
      status: "Retrieval failed"
    })
  };
}

function normalizeFallbackSources(
  sources: RetrievalSourceSummary[]
): RetrievalSourceSummary[] {
  return [...sources]
    .map((source) => ({
      ...source,
      chunks: [...source.chunks].sort(sortChunksByScore)
    }))
    .sort(sortSourcesByScore);
}

function buildRuntimeSources(
  baseSources: RetrievalSourceSummary[],
  chunks: ParsedRetrievalChunk[],
  request: RetrievalRuntimeRequest
): RetrievalSourceSummary[] {
  const baseSourceById = new Map(baseSources.map((source) => [source.sourceId, source]));
  const sourceGroups = new Map<string, ParsedRetrievalChunk[]>();

  for (const chunk of chunks) {
    const existingChunks = sourceGroups.get(chunk.sourceId);
    if (existingChunks) {
      existingChunks.push(chunk);
      continue;
    }

    sourceGroups.set(chunk.sourceId, [chunk]);
  }

  return Array.from(sourceGroups.entries())
    .map(([sourceId, sourceChunks]) =>
      buildRuntimeSourceSummary(baseSourceById.get(sourceId) ?? null, sourceChunks, request)
    )
    .sort(sortSourcesByScore);
}

function buildRuntimeSourceSummary(
  baseSource: RetrievalSourceSummary | null,
  rawChunks: ParsedRetrievalChunk[],
  request: RetrievalRuntimeRequest
): RetrievalSourceSummary {
  const sortedChunks = [...rawChunks].sort(sortParsedChunksByScore);
  const topChunk = sortedChunks[0] ?? null;
  const topScore = topChunk?.score ?? baseSource?.score ?? null;
  const quality = baseSource?.quality ?? deriveQualityFromScore(topScore);

  return {
    sourceId: topChunk?.sourceId ?? baseSource?.sourceId ?? "unknown_source",
    title: topChunk?.title ?? baseSource?.title ?? "Untitled source",
    detail:
      baseSource?.detail ??
      (compactText(topChunk?.snippet ?? "") || "Retrieval context used for grounding."),
    domain: topChunk?.domain ?? baseSource?.domain ?? "",
    domainLabel:
      baseSource?.domainLabel ??
      formatDomainLabel(topChunk?.domain ?? baseSource?.domain ?? ""),
    publisher: topChunk?.publisher ?? baseSource?.publisher ?? "Unknown publisher",
    sourceType: topChunk?.sourceType ?? baseSource?.sourceType ?? "reference",
    sourceTypeLabel:
      baseSource?.sourceTypeLabel ??
      formatSourceTypeLabel(topChunk?.sourceType ?? baseSource?.sourceType ?? "reference"),
    href: topChunk?.href || baseSource?.href || "",
    host: topChunk?.host || baseSource?.host || "",
    score: topScore,
    quality,
    qualityLabel: buildQualityLabel(quality, topScore),
    freshness: baseSource?.freshness ?? "unknown",
    freshnessLabel:
      baseSource?.freshnessLabel ??
      buildFreshnessLabel(baseSource?.freshness ?? "unknown"),
    updatedAt: baseSource?.updatedAt ?? null,
    whyUsed:
      baseSource?.whyUsed ??
      buildWhyUsedCopy({
        domainLabel:
          baseSource?.domainLabel ??
          formatDomainLabel(topChunk?.domain ?? baseSource?.domain ?? ""),
        query: request.query,
        quality,
        sourceTypeLabel:
          baseSource?.sourceTypeLabel ??
          formatSourceTypeLabel(topChunk?.sourceType ?? baseSource?.sourceType ?? "reference")
      }),
    chunks: sortedChunks.map((chunk, index) => ({
      id: `${chunk.sourceId}::chunk-${String(chunk.chunkIndex).padStart(4, "0")}`,
      chunkIndex: chunk.chunkIndex,
      score: chunk.score,
      snippet: compactText(chunk.snippet),
      relevanceLabel:
        index === 0 ? "Best match" : buildChunkRelevanceLabel(chunk.score)
    }))
  };
}

function findLatestRetrievalEvent(
  traceEvents: WorkflowRuntimeTraceEvent[],
  code: "retrieval_requested" | "retrieval_completed"
): WorkflowRuntimeTraceEvent | null {
  for (let index = traceEvents.length - 1; index >= 0; index -= 1) {
    const traceEvent = traceEvents[index];
    if (
      traceEvent?.event.event_type === "retrieval" &&
      traceEvent.event.payload.code === code
    ) {
      return traceEvent;
    }
  }

  return null;
}

function parseRetrievalRequestFromEvent(
  event: AssistantStreamEvent
): RetrievalRuntimeRequest | null {
  const rawRequest = event.payload.request;
  if (!isRecord(rawRequest)) {
    return null;
  }

  return buildRuntimeRequest(parseRetrievalRequest(rawRequest));
}

function parseRetrievalContextFromEvent(
  event: AssistantStreamEvent
): ParsedRetrievalContext | null {
  const rawContext = event.payload.context;
  if (!isRecord(rawContext)) {
    return null;
  }

  const rawChunks = Array.isArray(rawContext.chunks) ? rawContext.chunks : [];
  const parsedChunks = rawChunks
    .map((rawChunk) => parseRetrievalChunk(rawChunk))
    .filter((chunk): chunk is ParsedRetrievalChunk => chunk !== null);
  const rawRequest = isRecord(rawContext.request) ? rawContext.request : null;

  return {
    emittedAt: readEventTimestamp(event),
    provider: readText(rawContext.source),
    request: parseRetrievalRequest(rawRequest),
    chunks: parsedChunks,
    error: buildRetrievalError(event)
  };
}

function parseRetrievalRequest(rawRequest: Record<string, unknown> | null): ParsedRetrievalRequest {
  if (!rawRequest) {
    return {
      query: null,
      domains: [],
      limit: null,
      sourceFilter: null,
      sourceTypeFilter: null,
      publisherFilter: null
    };
  }

  const rawFilters = isRecord(rawRequest.filters) ? rawRequest.filters : null;

  return {
    query: readText(rawRequest.query),
    domains: readDomains(rawFilters),
    limit: readNumber(rawRequest.limit),
    sourceFilter: readText(rawFilters?.source_id),
    sourceTypeFilter: readText(rawFilters?.source_type),
    publisherFilter: readText(rawFilters?.publisher)
  };
}

function parseRetrievalChunk(rawChunk: unknown): ParsedRetrievalChunk | null {
  if (!isRecord(rawChunk)) {
    return null;
  }

  const sourceId = readText(rawChunk.source_id);
  const domain = readText(rawChunk.domain);
  const snippet = readText(rawChunk.excerpt);
  if (!sourceId || !domain || !snippet) {
    return null;
  }

  const href =
    readText(rawChunk.resolved_url) ?? readText(rawChunk.source_url) ?? "";

  return {
    sourceId,
    title:
      readText(rawChunk.document_title) ??
      readText(rawChunk.registry_title) ??
      sourceId,
    domain,
    publisher: readText(rawChunk.publisher) ?? "Unknown publisher",
    sourceType: readText(rawChunk.source_type) ?? "reference",
    href,
    host: readHost(href),
    score: readNumber(rawChunk.score),
    snippet,
    chunkIndex: readInteger(rawChunk.chunk_index) ?? 0
  };
}

function buildRuntimeRequest(
  request: ParsedRetrievalRequest
): RetrievalRuntimeRequest {
  const domainLabels = request.domains.map((domain) => formatDomainLabel(domain));
  const filterLabels = [
    ...domainLabels,
    request.sourceTypeFilter ? formatSourceTypeLabel(request.sourceTypeFilter) : null,
    request.publisherFilter,
    request.sourceFilter ? `Source ${request.sourceFilter}` : null
  ].filter((label): label is string => Boolean(label));

  return {
    ...request,
    domainLabels,
    filterLabels
  };
}

function buildRetrievalError(event: AssistantStreamEvent) {
  const parsed = parseSubsystemErrorPayload(event.payload.error);
  if (!parsed?.message && !parsed?.type) {
    return null;
  }

  const type = parsed.type ?? "retrieval_runtime_failed";
  const recoverable = parsed.recoverable ?? true;

  return createWorkstationError({
    type,
    category: "retrieval",
    subsystem: parsed.subsystem ?? "retrieval_gateway",
    userMessage:
      parsed.message ??
      "Retrieval context could not be loaded for this request.",
    debugMessage: parsed.debugMessage,
    recoverable,
    suggestedAction:
      parsed.suggestedAction ??
      "Retry the request or continue without retrieved references.",
    retryLabel: parsed.retryLabel ?? (recoverable ? "Retry retrieval" : null),
    resetLabel: parsed.resetLabel
  });
}

function buildFallbackRetrievalError(baseRetrieval: RetrievalSummary) {
  if (baseRetrieval.state !== "error") {
    return null;
  }

  return createWorkstationError({
    type: baseRetrieval.error?.type ?? "retrieval_runtime_failed",
    category: "retrieval",
    subsystem: baseRetrieval.error?.subsystem ?? "retrieval_gateway",
    userMessage:
      baseRetrieval.error?.userMessage ??
      baseRetrieval.detail ??
      "Retrieval context could not be loaded for this request.",
    debugMessage: baseRetrieval.error?.debugMessage ?? baseRetrieval.warning,
    recoverable: baseRetrieval.error?.recoverable ?? true,
    suggestedAction:
      baseRetrieval.error?.suggestedAction ??
      "Retry the request or continue without retrieved references.",
    retryLabel: baseRetrieval.error?.retryLabel ?? "Retry retrieval",
    resetLabel: baseRetrieval.error?.resetLabel ?? null
  });
}

function buildSummaryHeadline(
  state: RetrievalSummary["state"],
  sourceCount: number,
  chunkCount: number,
  query: string | null
) {
  switch (state) {
    case "available":
      return `${countLabel(chunkCount, "chunk")} from ${countLabel(sourceCount, "source")}`;
    case "pending":
      return query ? `Searching for “${truncateText(query, 44)}”` : "Searching knowledge base";
    case "empty":
      return "No retrieved context";
    case "error":
      return "Retrieval failed";
    case "unavailable":
      return "Retrieval unavailable";
    default:
      return "Retrieval context";
  }
}

function buildMissingDomainWarning(
  request: RetrievalRuntimeRequest,
  sourceDomainLabels: string[]
) {
  if (request.domainLabels.length === 0) {
    return null;
  }

  if (sourceDomainLabels.length === 0) {
    return `No retrieved chunks for ${request.domainLabels.join(", ")}.`;
  }

  const sourceDomainSet = new Set(sourceDomainLabels);
  const missingDomainLabels = request.domainLabels.filter(
    (domainLabel) => !sourceDomainSet.has(domainLabel)
  );

  if (missingDomainLabels.length === 0) {
    return null;
  }

  return `No retrieved chunks for ${missingDomainLabels.join(", ")}.`;
}

function buildStaleSourceWarning(sources: RetrievalSourceSummary[]) {
  const staleCount = sources.filter((source) => source.freshness === "stale").length;

  if (staleCount === 0) {
    return null;
  }

  return `${countLabel(staleCount, "source")} may be stale.`;
}

function buildFreshnessSummaryLabel(sources: RetrievalSourceSummary[]) {
  const staleCount = sources.filter((source) => source.freshness === "stale").length;
  const freshCount = sources.filter((source) => source.freshness === "fresh").length;
  const unknownCount = sources.filter(
    (source) => source.freshness === "unknown"
  ).length;

  if (staleCount > 0) {
    return `${countLabel(staleCount, "stale source")}`;
  }

  if (freshCount > 0 && unknownCount === 0) {
    return "Freshness current";
  }

  if (freshCount > 0) {
    return "Freshness mixed";
  }

  return "Freshness unverified";
}

function buildWhyUsedCopy({
  domainLabel,
  query,
  quality,
  sourceTypeLabel
}: {
  domainLabel: string;
  query: string | null;
  quality: RetrievalQuality;
  sourceTypeLabel: string;
}) {
  if (query) {
    return `Used to ground ${domainLabel} guidance for “${truncateText(query, 56)}”.`;
  }

  if (quality === "high") {
    return `Used as the highest-confidence ${domainLabel} source in this retrieval pass.`;
  }

  return `Used as supporting ${sourceTypeLabel.toLowerCase()} context for ${domainLabel}.`;
}

function buildQualityLabel(quality: RetrievalQuality, score: number | null) {
  if (score != null) {
    return `${formatScorePercent(score)} match`;
  }

  switch (quality) {
    case "high":
      return "High relevance";
    case "medium":
      return "Relevant grounding";
    case "low":
      return "Low confidence";
    default:
      return "Relevance unknown";
  }
}

function buildFreshnessLabel(freshness: RetrievalFreshness) {
  switch (freshness) {
    case "fresh":
      return "Current";
    case "stale":
      return "Review soon";
    default:
      return "Freshness unverified";
  }
}

function buildChunkRelevanceLabel(score: number | null) {
  if (score == null) {
    return "Supporting context";
  }

  if (score >= 0.85) {
    return "High match";
  }

  if (score >= 0.7) {
    return "Supporting match";
  }

  return "Low-confidence match";
}

function deriveQualityFromScore(score: number | null): RetrievalQuality {
  if (score == null) {
    return "unknown";
  }

  if (score >= 0.85) {
    return "high";
  }

  if (score >= 0.7) {
    return "medium";
  }

  return "low";
}

function formatDomainLabel(value: string) {
  if (!value) {
    return "Unknown domain";
  }

  const override = domainLabelOverrides[value];
  if (override) {
    return override;
  }

  return value
    .split("_")
    .map((segment) => {
      switch (segment) {
        case "api":
          return "API";
        case "js":
          return "JS";
        case "wgsl":
          return "WGSL";
        default:
          return segment.charAt(0).toUpperCase() + segment.slice(1);
      }
    })
    .join(" ");
}

function formatSourceTypeLabel(value: string) {
  return sourceTypeLabels[value] ?? value.replace(/_/g, " ");
}

function formatRetrievalProviderLabel(value: string | null) {
  if (value === "official_kb") {
    return "Official knowledge base";
  }

  if (!value) {
    return "Retrieval context";
  }

  return value.replace(/_/g, " ");
}

function formatScorePercent(score: number) {
  return `${Math.round(score * 100)}%`;
}

function sortSourcesByScore(
  firstSource: RetrievalSourceSummary,
  secondSource: RetrievalSourceSummary
) {
  const firstScore = firstSource.score ?? -1;
  const secondScore = secondSource.score ?? -1;

  if (firstScore !== secondScore) {
    return secondScore - firstScore;
  }

  const firstQuality = retrievalQualityOrder[firstSource.quality];
  const secondQuality = retrievalQualityOrder[secondSource.quality];
  if (firstQuality !== secondQuality) {
    return secondQuality - firstQuality;
  }

  return firstSource.title.localeCompare(secondSource.title);
}

function sortChunksByScore(
  firstChunk: RetrievalChunkSummary,
  secondChunk: RetrievalChunkSummary
) {
  return sortScores(firstChunk.score, secondChunk.score) || firstChunk.chunkIndex - secondChunk.chunkIndex;
}

function sortParsedChunksByScore(
  firstChunk: ParsedRetrievalChunk,
  secondChunk: ParsedRetrievalChunk
) {
  return sortScores(firstChunk.score, secondChunk.score) || firstChunk.chunkIndex - secondChunk.chunkIndex;
}

function sortScores(firstScore: number | null, secondScore: number | null) {
  return (secondScore ?? -1) - (firstScore ?? -1);
}

function countLabel(count: number, noun: string) {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

function latestUpdatedAt(sources: RetrievalSourceSummary[]) {
  return (
    sources
      .map((source) => source.updatedAt)
      .filter((updatedAt): updatedAt is string => Boolean(updatedAt))
      .sort()
      .at(-1) ?? null
  );
}

function truncateText(value: string, limit: number) {
  if (value.length <= limit) {
    return value;
  }

  return `${value.slice(0, Math.max(limit - 1, 0)).trimEnd()}…`;
}

function compactText(value: string) {
  return value.replace(/\s+/g, " ").trim();
}

function readDomains(value: Record<string, unknown> | null) {
  if (!value) {
    return [];
  }

  const rawDomains = value.domains;
  if (Array.isArray(rawDomains)) {
    return Array.from(
      new Set(rawDomains.map((domain) => readText(domain)).filter((domain): domain is string => Boolean(domain)))
    );
  }

  const legacyDomain = readText(value.domain);
  return legacyDomain ? [legacyDomain] : [];
}

function readHost(value: string) {
  if (!value) {
    return "";
  }

  try {
    return new URL(value).host;
  } catch {
    return "";
  }
}

function readText(value: unknown) {
  if (typeof value !== "string") {
    return null;
  }

  const trimmedValue = value.trim();
  return trimmedValue || null;
}

function readNumber(value: unknown) {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }

  return null;
}

function readInteger(value: unknown) {
  if (typeof value === "number" && Number.isInteger(value)) {
    return value;
  }

  return null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
