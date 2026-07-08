import type {
  RetrievalChunkSummary,
  RetrievalFreshness,
  RetrievalQuality,
  RetrievalSourceAvailability,
  RetrievalSourceHealthMetadata,
  RetrievalSourceHealthStatus,
  RetrievalSourceSummary,
  RetrievalSourceSyncOutcome,
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
  coverageDetail: string;
  confidence: RetrievalQuality;
  confidenceLabel: string;
  confidenceDetail: string;
  usedChunkLabel: string;
  usedChunkDetail: string;
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
  rank: number | null;
  originalScore: number | null;
  scoreAdjustment: number | null;
  domainMatch: boolean | null;
  selectionReason: string | null;
  usedInContext: boolean | null;
  health: RetrievalSourceHealthMetadata | null;
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
  const sources = normalizeFallbackSources(baseRetrieval.sources, request);
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
        ? `No retrieved context for this run. No matching chunks were returned for “${request.query}”`
        : "No retrieved context for this run. No matching retrieval chunks were returned for this request.";

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
  const chunks = sources.flatMap((source) => source.chunks);
  const usedChunkCount = chunks.filter((chunk) => chunk.usedInContext !== false).length;
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
  const confidence = buildRetrievalConfidence(chunks);
  const coverage = buildRetrievalCoverage(request, domainLabels);
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
    coverageLabel: coverage.label,
    coverageDetail: coverage.detail,
    confidence: confidence.quality,
    confidenceLabel: confidence.label,
    confidenceDetail: confidence.detail,
    usedChunkLabel: `${usedChunkCount} ${usedChunkCount === 1 ? "chunk" : "chunks"} used`,
    usedChunkDetail:
      usedChunkCount > 0
        ? usedChunkCount === chunkCount
          ? "Every returned chunk was included in the generation context."
          : `${usedChunkCount} of ${chunkCount} returned chunks were included in the generation context.`
        : "No retrieved context for this run.",
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
  sources: RetrievalSourceSummary[],
  request: RetrievalRuntimeRequest
): RetrievalSourceSummary[] {
  const normalizedSources = [...sources]
    .map((source) => ({
      ...source,
      chunks: (Array.isArray(source.chunks) ? [...source.chunks] : []).sort(
        sortChunksByScore
      )
    }));
  const rankByChunkId = buildFallbackChunkRankMap(normalizedSources);

  return normalizedSources
    .map((source) => {
      const domainMatch =
        request.domains.length > 0 ? request.domains.includes(source.domain) : null;
      const sourceSelection = source.selectedForContext;
      const chunks = source.chunks
        .map((chunk) => {
          const rank = rankByChunkId.get(chunk.id) ?? chunk.rank ?? null;
          const scoreAdjustment =
            chunk.scoreAdjustment ??
            buildScoreAdjustment(chunk.score, chunk.originalScore ?? null);

          return {
            ...chunk,
            rank,
            originalScore: chunk.originalScore ?? chunk.score,
            scoreAdjustment,
            domainMatch: chunk.domainMatch ?? domainMatch,
            usedInContext:
              chunk.usedInContext ?? sourceSelection ?? true,
            selectionReason:
              chunk.selectionReason ??
              buildFallbackSelectionReason({
                domainMatch: chunk.domainMatch ?? domainMatch,
                rank,
                scoreAdjustment
              })
          };
        })
        .sort(sortChunksByRank);

      return {
        ...source,
        bestRank: chunks[0]?.rank ?? source.bestRank ?? null,
        selectedForContext:
          sourceSelection ?? chunks.some((chunk) => chunk.usedInContext !== false),
        chunks
      };
    })
    .sort(sortSourcesByRank);
}

function buildRuntimeSources(
  baseSources: RetrievalSourceSummary[],
  chunks: ParsedRetrievalChunk[],
  request: RetrievalRuntimeRequest
): RetrievalSourceSummary[] {
  const rankedChunks = normalizeParsedChunkRanks(chunks);
  const baseSourceById = new Map(baseSources.map((source) => [source.sourceId, source]));
  const sourceGroups = new Map<string, ParsedRetrievalChunk[]>();

  for (const chunk of rankedChunks) {
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
    .sort(sortSourcesByRank);
}

function buildRuntimeSourceSummary(
  baseSource: RetrievalSourceSummary | null,
  rawChunks: ParsedRetrievalChunk[],
  request: RetrievalRuntimeRequest
): RetrievalSourceSummary {
  const sortedChunks = [...rawChunks].sort(sortParsedChunksByRank);
  const topChunk = sortedChunks[0] ?? null;
  const topScore = topChunk?.score ?? baseSource?.score ?? null;
  const quality = baseSource?.quality ?? deriveQualityFromScore(topScore);
  const health = topChunk?.health ?? baseSource?.health ?? null;
  const freshness =
    normalizeRetrievalFreshness(health?.freshnessStatus) ??
    baseSource?.freshness ??
    "unknown";

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
    freshness,
    freshnessLabel:
      baseSource?.freshnessLabel ??
      buildFreshnessLabel(freshness),
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
    bestRank: topChunk?.rank ?? baseSource?.bestRank ?? null,
    selectedForContext: sortedChunks.some(
      (chunk) => chunk.usedInContext !== false
    ),
    health,
    chunks: sortedChunks.map((chunk, index) => ({
      id: `${chunk.sourceId}::chunk-${String(chunk.chunkIndex).padStart(4, "0")}`,
      chunkIndex: chunk.chunkIndex,
      score: chunk.score,
      snippet: compactText(chunk.snippet),
      rank: chunk.rank,
      originalScore: chunk.originalScore,
      scoreAdjustment:
        chunk.scoreAdjustment ??
        buildScoreAdjustment(chunk.score, chunk.originalScore),
      domainMatch: chunk.domainMatch,
      usedInContext: chunk.usedInContext ?? true,
      selectionReason:
        chunk.selectionReason ??
        buildFallbackSelectionReason({
          domainMatch: chunk.domainMatch,
          rank: chunk.rank,
          scoreAdjustment:
            chunk.scoreAdjustment ??
            buildScoreAdjustment(chunk.score, chunk.originalScore)
        }),
      relevanceLabel:
        chunk.rank === 1 || (chunk.rank == null && index === 0)
          ? "Best match"
          : buildChunkRelevanceLabel(chunk.score)
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
    chunkIndex: readInteger(rawChunk.chunk_index) ?? 0,
    rank: readInteger(rawChunk.rank),
    originalScore: readNumber(rawChunk.original_score),
    scoreAdjustment: readNumber(rawChunk.score_adjustment),
    domainMatch: readBoolean(rawChunk.domain_match),
    selectionReason: readText(rawChunk.selection_reason),
    usedInContext: readBoolean(rawChunk.used_in_context),
    health: parseRetrievalSourceHealth(rawChunk)
  };
}

function parseRetrievalSourceHealth(
  rawChunk: Record<string, unknown>
): RetrievalSourceHealthMetadata | null {
  const rawHealth =
    readRecord(rawChunk.source_health) ??
    readRecord(rawChunk.kb_source_health) ??
    readRecord(rawChunk.health);
  const health = rawHealth ?? rawChunk;
  const rawSync = readRecord(health.sync);
  const rawSource = readRecord(health.source);
  const status = normalizeRetrievalHealthStatus(
    readText(health.health_status) ?? readText(health.status)
  );
  const freshnessStatus = normalizeRetrievalFreshness(
    readText(health.freshness_status) ?? readText(health.freshness)
  );
  const availability =
    normalizeRetrievalAvailability(readText(health.availability)) ??
    availabilityFromBoolean(readBoolean(health.available));
  const syncOutcome = normalizeRetrievalSyncOutcome(
    readText(health.sync_outcome) ??
      readText(health.sync_status) ??
      readText(rawSync?.sync_status)
  );
  const warnings = readStringArray(health.warnings ?? rawSync?.warnings);
  const metadata: RetrievalSourceHealthMetadata = {
    status,
    freshnessStatus,
    availability,
    domainOwner:
      readText(health.domain_owner) ??
      readText(rawSource?.domain_owner) ??
      readText(rawSource?.owner),
    indexedChunkCount:
      readInteger(health.indexed_chunk_count) ??
      readInteger(health.chunk_count) ??
      readInteger(rawSync?.chunk_count),
    lastSuccessfulSyncAt:
      readText(health.last_successful_sync_at) ??
      readText(health.last_synced_at) ??
      readText(rawSync?.last_synced_at),
    lastAttemptedSyncAt:
      readText(health.last_attempted_sync_at) ??
      readText(health.requested_at) ??
      readText(rawSync?.requested_at),
    syncOutcome,
    refreshRecommended:
      readBoolean(health.refresh_recommended) ?? null,
    checkedAt: readText(health.checked_at),
    warnings
  };

  return rawHealth || hasHealthMetadata(metadata) ? metadata : null;
}

function buildRuntimeRequest(
  request: ParsedRetrievalRequest
): RetrievalRuntimeRequest {
  const domains = Array.isArray(request.domains) ? request.domains : [];
  const domainLabels = domains.map((domain) => formatDomainLabel(domain));
  const filterLabels = [
    ...domainLabels,
    request.sourceTypeFilter ? formatSourceTypeLabel(request.sourceTypeFilter) : null,
    request.publisherFilter,
    request.sourceFilter ? `Source ${request.sourceFilter}` : null
  ].filter((label): label is string => Boolean(label));

  return {
    ...request,
    domains,
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
      return "No retrieved context for this run.";
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

function buildRetrievalConfidence(chunks: RetrievalChunkSummary[]) {
  const scores = chunks
    .map((chunk) => chunk.score)
    .filter((score): score is number => score != null);

  if (scores.length === 0) {
    return {
      quality: "unknown" as const,
      label: "Confidence unknown",
      detail: "No retrieval scores were available for this context."
    };
  }

  const averageScore =
    scores.reduce((total, score) => total + score, 0) / scores.length;
  const quality = deriveQualityFromScore(averageScore);
  const label =
    quality === "high"
      ? "High confidence"
      : quality === "medium"
        ? "Medium confidence"
        : "Low confidence";

  return {
    quality,
    label,
    detail: `${formatScorePercent(averageScore)} average relevance across ${countLabel(
      scores.length,
      "scored chunk"
    )}.`
  };
}

function buildRetrievalCoverage(
  request: RetrievalRuntimeRequest,
  sourceDomainLabels: string[]
) {
  if (request.domainLabels.length === 0) {
    return sourceDomainLabels.length > 0
      ? {
          label: countLabel(sourceDomainLabels.length, "domain"),
          detail: `${sourceDomainLabels.join(", ")} represented in retrieved context.`
        }
      : {
          label: "No domain filter",
          detail: "The request did not constrain retrieval to a specific domain."
        };
  }

  const sourceDomainSet = new Set(sourceDomainLabels);
  const matchedDomains = request.domainLabels.filter((domainLabel) =>
    sourceDomainSet.has(domainLabel)
  );
  const missingDomains = request.domainLabels.filter(
    (domainLabel) => !sourceDomainSet.has(domainLabel)
  );

  return {
    label: `${matchedDomains.length}/${request.domainLabels.length} domains covered`,
    detail:
      missingDomains.length === 0
        ? `All requested domains are represented: ${matchedDomains.join(", ")}.`
        : `${matchedDomains.length > 0 ? `${matchedDomains.join(", ")} represented. ` : ""}Missing ${missingDomains.join(", ")}.`
  };
}

function buildFallbackChunkRankMap(sources: RetrievalSourceSummary[]) {
  const chunks = sources.flatMap((source) => source.chunks);
  const hasCompleteRanks =
    chunks.length > 0 && chunks.every((chunk) => chunk.rank != null);
  const sortedChunks = [...chunks].sort(
    hasCompleteRanks ? sortChunksByRank : sortChunksByScore
  );

  return new Map(
    sortedChunks.map((chunk, index) => [
      chunk.id,
      hasCompleteRanks ? (chunk.rank ?? index + 1) : index + 1
    ])
  );
}

function normalizeParsedChunkRanks(
  chunks: ParsedRetrievalChunk[]
): ParsedRetrievalChunk[] {
  const hasCompleteRanks =
    chunks.length > 0 && chunks.every((chunk) => chunk.rank != null);

  return [...chunks]
    .sort(hasCompleteRanks ? sortParsedChunksByRank : sortParsedChunksByScore)
    .map((chunk, index) => ({
      ...chunk,
      rank: hasCompleteRanks ? (chunk.rank ?? index + 1) : index + 1
    }));
}

function buildScoreAdjustment(
  score: number | null,
  originalScore: number | null
) {
  if (score == null || originalScore == null || score === originalScore) {
    return null;
  }

  return score - originalScore;
}

function buildFallbackSelectionReason({
  domainMatch,
  rank,
  scoreAdjustment
}: {
  domainMatch: boolean | null;
  rank: number | null;
  scoreAdjustment: number | null;
}) {
  const rankLabel = rank != null ? `Ranked #${rank}` : "Selected";

  if (scoreAdjustment != null) {
    return `${rankLabel} after semantic retrieval and route-specific relevance adjustment.`;
  }

  if (domainMatch) {
    return `${rankLabel} for semantic relevance within the requested domain scope.`;
  }

  if (domainMatch === false) {
    return `${rankLabel} as cross-domain supporting context.`;
  }

  return `${rankLabel} by semantic relevance from the official knowledge base.`;
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

function sortSourcesByRank(
  firstSource: RetrievalSourceSummary,
  secondSource: RetrievalSourceSummary
) {
  const rankDifference =
    (firstSource.bestRank ?? Number.MAX_SAFE_INTEGER) -
    (secondSource.bestRank ?? Number.MAX_SAFE_INTEGER);

  return rankDifference || sortSourcesByScore(firstSource, secondSource);
}

function sortChunksByScore(
  firstChunk: RetrievalChunkSummary,
  secondChunk: RetrievalChunkSummary
) {
  return sortScores(firstChunk.score, secondChunk.score) || firstChunk.chunkIndex - secondChunk.chunkIndex;
}

function sortChunksByRank(
  firstChunk: RetrievalChunkSummary,
  secondChunk: RetrievalChunkSummary
) {
  const rankDifference =
    (firstChunk.rank ?? Number.MAX_SAFE_INTEGER) -
    (secondChunk.rank ?? Number.MAX_SAFE_INTEGER);

  return rankDifference || sortChunksByScore(firstChunk, secondChunk);
}

function sortParsedChunksByRank(
  firstChunk: ParsedRetrievalChunk,
  secondChunk: ParsedRetrievalChunk
) {
  const rankDifference =
    (firstChunk.rank ?? Number.MAX_SAFE_INTEGER) -
    (secondChunk.rank ?? Number.MAX_SAFE_INTEGER);

  return (
    rankDifference ||
    sortScores(firstChunk.score, secondChunk.score) ||
    firstChunk.chunkIndex - secondChunk.chunkIndex
  );
}

function sortParsedChunksByScore(
  firstChunk: ParsedRetrievalChunk,
  secondChunk: ParsedRetrievalChunk
) {
  return (
    sortScores(firstChunk.score, secondChunk.score) ||
    firstChunk.chunkIndex - secondChunk.chunkIndex
  );
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

function readBoolean(value: unknown) {
  return typeof value === "boolean" ? value : null;
}

function readStringArray(value: unknown) {
  return Array.isArray(value)
    ? value
        .map((item) => readText(item))
        .filter((item): item is string => item !== null)
    : [];
}

function readRecord(value: unknown) {
  return isRecord(value) ? value : null;
}

function normalizeRetrievalHealthStatus(
  value: unknown
): RetrievalSourceHealthStatus | "sync_failed" | null {
  return value === "healthy" ||
    value === "warning" ||
    value === "stale" ||
    value === "failed" ||
    value === "sync_failed" ||
    value === "unknown"
    ? value
    : null;
}

function normalizeRetrievalFreshness(value: unknown): RetrievalFreshness | null {
  return value === "fresh" || value === "stale" || value === "unknown"
    ? value
    : null;
}

function normalizeRetrievalAvailability(
  value: unknown
): RetrievalSourceAvailability | null {
  return value === "available" ||
    value === "degraded" ||
    value === "unavailable" ||
    value === "unknown"
    ? value
    : null;
}

function normalizeRetrievalSyncOutcome(
  value: unknown
): RetrievalSourceSyncOutcome | null {
  return value === "succeeded" ||
    value === "failed" ||
    value === "pending" ||
    value === "unknown"
    ? value
    : null;
}

function availabilityFromBoolean(
  value: boolean | null
): RetrievalSourceAvailability | null {
  if (value === true) {
    return "available";
  }

  if (value === false) {
    return "unavailable";
  }

  return null;
}

function hasHealthMetadata(metadata: RetrievalSourceHealthMetadata) {
  return Object.values(metadata).some(
    (value) =>
      value !== null &&
      value !== undefined &&
      (!Array.isArray(value) || value.length > 0)
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
