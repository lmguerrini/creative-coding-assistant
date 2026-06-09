import type {
  RetrievalFreshness,
  RetrievalSourceAvailability,
  RetrievalSourceHealthMetadata,
  RetrievalSourceHealthStatus,
  RetrievalSourceSummary,
  RetrievalSourceSyncOutcome
} from "./assistant-client";
import type { RetrievalRuntimeModel } from "./retrieval-runtime";

export type KbSourceHealthStatus = RetrievalSourceHealthStatus;

export type KbSourceHealthSource = {
  sourceId: string;
  title: string;
  status: KbSourceHealthStatus;
  statusLabel: string;
  statusDetail: string;
  freshness: RetrievalFreshness;
  freshnessLabel: string;
  availability: RetrievalSourceAvailability;
  availabilityLabel: string;
  domainOwner: string;
  indexedChunkCount: number | null;
  indexedChunkLabel: string;
  lastSuccessfulSyncAt: string | null;
  lastSuccessfulSyncLabel: string;
  lastAttemptedSyncAt: string | null;
  lastAttemptedSyncLabel: string;
  syncOutcome: RetrievalSourceSyncOutcome;
  syncOutcomeLabel: string;
  coverageLabel: string;
  coverageDetail: string;
  warnings: readonly string[];
  metadataAvailable: boolean;
};

export type KbSourceHealthDashboardModel = {
  status: KbSourceHealthStatus;
  statusLabel: string;
  statusDetail: string;
  sourceCount: number;
  healthySourceCount: number;
  attentionSourceCount: number;
  availableSourceCount: number;
  availabilityLabel: string;
  indexedChunkCount: number | null;
  indexedChunkLabel: string;
  latestSyncAttemptAt: string | null;
  latestSyncAttemptLabel: string;
  domainOwnerCount: number;
  domainOwnerLabel: string;
  sources: KbSourceHealthSource[];
};

type BuildKbSourceHealthOptions = {
  observedAt?: string | number | Date | null;
};

export function buildKbSourceHealthDashboardModel(
  runtime: RetrievalRuntimeModel,
  options: BuildKbSourceHealthOptions = {}
): KbSourceHealthDashboardModel {
  const observedAtMs = resolveObservedAtMs(runtime.sources, options.observedAt);
  const totalUsedChunks = runtime.sources.reduce(
    (total, source) =>
      total +
      source.chunks.filter(
        (chunk) => chunk.usedInContext ?? source.selectedForContext ?? true
      ).length,
    0
  );
  const sources = runtime.sources.map((source) =>
    buildSourceHealth(source, totalUsedChunks, observedAtMs)
  );
  const status = deriveDashboardStatus(sources);
  const healthySourceCount = sources.filter(
    (source) => source.status === "healthy"
  ).length;
  const attentionSourceCount = sources.filter(
    (source) => source.status !== "healthy"
  ).length;
  const availableSourceCount = sources.filter(
    (source) => source.availability === "available"
  ).length;
  const reportedIndexedCounts = sources
    .map((source) => source.indexedChunkCount)
    .filter((count): count is number => count != null);
  const allIndexedCountsReported =
    sources.length > 0 && reportedIndexedCounts.length === sources.length;
  const indexedChunkCount =
    reportedIndexedCounts.length > 0
      ? reportedIndexedCounts.reduce((total, count) => total + count, 0)
      : null;
  const latestSyncAttemptAt = latestTimestamp(
    sources.map((source) => source.lastAttemptedSyncAt)
  );
  const domainOwnerCount = new Set(
    sources.map((source) => source.domainOwner).filter(Boolean)
  ).size;

  return {
    status,
    statusLabel: formatHealthStatus(status),
    statusDetail: buildDashboardStatusDetail({
      attentionSourceCount,
      healthySourceCount,
      sourceCount: sources.length,
      status
    }),
    sourceCount: sources.length,
    healthySourceCount,
    attentionSourceCount,
    availableSourceCount,
    availabilityLabel:
      sources.length > 0
        ? `${availableSourceCount}/${sources.length} sources available`
        : "No source availability reported",
    indexedChunkCount,
    indexedChunkLabel:
      indexedChunkCount == null
        ? "Indexed total unavailable"
        : allIndexedCountsReported
          ? countLabel(indexedChunkCount, "indexed chunk")
          : `${countLabel(indexedChunkCount, "indexed chunk")} reported`,
    latestSyncAttemptAt,
    latestSyncAttemptLabel: formatTimestampWithRecency(
      latestSyncAttemptAt,
      observedAtMs,
      "No sync attempt reported"
    ),
    domainOwnerCount,
    domainOwnerLabel:
      domainOwnerCount > 0
        ? countLabel(domainOwnerCount, "domain owner")
        : "Domain ownership unreported",
    sources
  };
}

function buildSourceHealth(
  source: RetrievalSourceSummary,
  totalUsedChunks: number,
  observedAtMs: number
): KbSourceHealthSource {
  const metadata = normalizeHealthMetadata(source.health);
  const freshness = metadata.freshnessStatus ?? source.freshness ?? "unknown";
  const availability =
    metadata.availability ??
    (source.href && source.chunks.length > 0 ? "available" : "unknown");
  const syncOutcome = metadata.syncOutcome ?? "unknown";
  const warnings = uniqueText(metadata.warnings ?? []);
  const metadataAvailable = hasReportedHealthMetadata(source.health);
  const indexedChunkCount = normalizeCount(metadata.indexedChunkCount);
  const status = deriveSourceStatus({
    availability,
    freshness,
    metadataAvailable,
    refreshRecommended: metadata.refreshRecommended,
    reportedStatus: metadata.status,
    syncOutcome,
    warnings
  });
  const usedChunkCount = source.chunks.filter(
    (chunk) => chunk.usedInContext ?? source.selectedForContext ?? true
  ).length;
  const contributionPercent =
    totalUsedChunks > 0 ? Math.round((usedChunkCount / totalUsedChunks) * 100) : 0;

  return {
    sourceId: source.sourceId,
    title: source.title,
    status,
    statusLabel: formatHealthStatus(status),
    statusDetail: buildSourceStatusDetail({
      availability,
      freshness,
      metadataAvailable,
      status,
      syncOutcome,
      warnings
    }),
    freshness,
    freshnessLabel: formatFreshness(freshness),
    availability,
    availabilityLabel: formatAvailability(availability),
    domainOwner:
      cleanText(metadata.domainOwner) ??
      cleanText(source.publisher) ??
      "Ownership unreported",
    indexedChunkCount,
    indexedChunkLabel:
      indexedChunkCount != null
        ? countLabel(indexedChunkCount, "indexed chunk")
        : "Not reported",
    lastSuccessfulSyncAt: normalizeTimestamp(metadata.lastSuccessfulSyncAt),
    lastSuccessfulSyncLabel: formatTimestampWithRecency(
      metadata.lastSuccessfulSyncAt,
      observedAtMs,
      "Not reported"
    ),
    lastAttemptedSyncAt: normalizeTimestamp(metadata.lastAttemptedSyncAt),
    lastAttemptedSyncLabel: formatTimestampWithRecency(
      metadata.lastAttemptedSyncAt,
      observedAtMs,
      "Not reported"
    ),
    syncOutcome,
    syncOutcomeLabel: formatSyncOutcome(syncOutcome),
    coverageLabel:
      totalUsedChunks > 0
        ? `${usedChunkCount}/${totalUsedChunks} context chunks`
        : "No context coverage",
    coverageDetail:
      usedChunkCount > 0
        ? `${contributionPercent}% of the active retrieval context came from this source.`
        : "This source did not contribute to the active retrieval context.",
    warnings,
    metadataAvailable
  };
}

function normalizeHealthMetadata(
  metadata: RetrievalSourceHealthMetadata | null | undefined
): {
  status: RetrievalSourceHealthStatus | null;
  freshnessStatus: RetrievalFreshness | null;
  availability: RetrievalSourceAvailability | null;
  domainOwner: string | null;
  indexedChunkCount: number | null;
  lastSuccessfulSyncAt: string | null;
  lastAttemptedSyncAt: string | null;
  syncOutcome: RetrievalSourceSyncOutcome | null;
  refreshRecommended: boolean | null;
  checkedAt: string | null;
  warnings: string[];
} {
  return {
    status: normalizeHealthStatus(metadata?.status),
    freshnessStatus: normalizeFreshness(metadata?.freshnessStatus),
    availability: normalizeAvailability(metadata?.availability),
    domainOwner: cleanText(metadata?.domainOwner),
    indexedChunkCount: normalizeCount(metadata?.indexedChunkCount),
    lastSuccessfulSyncAt: normalizeTimestamp(metadata?.lastSuccessfulSyncAt),
    lastAttemptedSyncAt: normalizeTimestamp(metadata?.lastAttemptedSyncAt),
    syncOutcome: normalizeSyncOutcome(metadata?.syncOutcome),
    refreshRecommended:
      typeof metadata?.refreshRecommended === "boolean"
        ? metadata.refreshRecommended
        : null,
    checkedAt: normalizeTimestamp(metadata?.checkedAt),
    warnings: Array.isArray(metadata?.warnings)
      ? metadata.warnings.filter(
          (warning): warning is string =>
            typeof warning === "string" && warning.trim().length > 0
        )
      : []
  };
}

function deriveSourceStatus({
  availability,
  freshness,
  metadataAvailable,
  refreshRecommended,
  reportedStatus,
  syncOutcome,
  warnings
}: {
  availability: RetrievalSourceAvailability;
  freshness: RetrievalFreshness;
  metadataAvailable: boolean;
  refreshRecommended: boolean | null;
  reportedStatus: RetrievalSourceHealthStatus | null;
  syncOutcome: RetrievalSourceSyncOutcome;
  warnings: readonly string[];
}): KbSourceHealthStatus {
  if (
    reportedStatus === "failed" ||
    syncOutcome === "failed" ||
    availability === "unavailable"
  ) {
    return "failed";
  }

  if (reportedStatus === "stale" || freshness === "stale") {
    return "stale";
  }

  if (
    reportedStatus === "warning" ||
    availability === "degraded" ||
    refreshRecommended ||
    warnings.length > 0
  ) {
    return "warning";
  }

  if (reportedStatus === "healthy") {
    return "healthy";
  }

  if (
    metadataAvailable &&
    syncOutcome === "succeeded" &&
    freshness === "fresh" &&
    availability === "available"
  ) {
    return "healthy";
  }

  if (!metadataAvailable && freshness === "fresh" && availability === "available") {
    return "warning";
  }

  return "unknown";
}

function deriveDashboardStatus(
  sources: KbSourceHealthSource[]
): KbSourceHealthStatus {
  if (sources.length === 0) {
    return "unknown";
  }

  if (sources.some((source) => source.status === "failed")) {
    return "failed";
  }

  if (sources.some((source) => source.status === "stale")) {
    return "stale";
  }

  if (sources.some((source) => source.status === "warning")) {
    return "warning";
  }

  const unknownCount = sources.filter(
    (source) => source.status === "unknown"
  ).length;
  if (unknownCount === sources.length) {
    return "unknown";
  }

  return unknownCount > 0 ? "warning" : "healthy";
}

function buildSourceStatusDetail({
  availability,
  freshness,
  metadataAvailable,
  status,
  syncOutcome,
  warnings
}: {
  availability: RetrievalSourceAvailability;
  freshness: RetrievalFreshness;
  metadataAvailable: boolean;
  status: KbSourceHealthStatus;
  syncOutcome: RetrievalSourceSyncOutcome;
  warnings: readonly string[];
}) {
  if (status === "failed") {
    return syncOutcome === "failed"
      ? "The latest source sync failed; previously indexed content may still be available."
      : "The source is unavailable for knowledge-base maintenance."
  }

  if (status === "stale") {
    return "The source is available, but its indexed content is beyond the freshness window.";
  }

  if (status === "warning") {
    if (!metadataAvailable) {
      return "The source is available, but sync health metadata was not recorded for this legacy session.";
    }

    return (
      warnings[0] ??
      (availability === "degraded"
        ? "The source is partially available and should be reviewed."
        : freshness === "fresh"
          ? "The source is current, but maintenance attention is recommended."
          : "Source health requires operator review.")
    );
  }

  if (status === "healthy") {
    return "The source is available, current, and its latest reported sync succeeded.";
  }

  return "Source health has not been reported for this session.";
}

function buildDashboardStatusDetail({
  attentionSourceCount,
  healthySourceCount,
  sourceCount,
  status
}: {
  attentionSourceCount: number;
  healthySourceCount: number;
  sourceCount: number;
  status: KbSourceHealthStatus;
}) {
  if (sourceCount === 0) {
    return "Source health appears when retrieval returns knowledge-base sources.";
  }

  if (status === "healthy") {
    return `All ${sourceCount} retrieved sources report healthy maintenance state.`;
  }

  if (status === "unknown") {
    return "Retrieved sources do not include maintenance health metadata.";
  }

  return `${healthySourceCount} healthy · ${attentionSourceCount} ${
    attentionSourceCount === 1 ? "needs" : "need"
  } attention or verification.`;
}

function resolveObservedAtMs(
  sources: RetrievalSourceSummary[],
  observedAt: BuildKbSourceHealthOptions["observedAt"]
) {
  const explicitObservedAt = parseTimestamp(observedAt);
  if (explicitObservedAt != null) {
    return explicitObservedAt;
  }

  const checkedAt = latestTimestamp(
    sources.map((source) => source.health?.checkedAt ?? null)
  );

  return parseTimestamp(checkedAt) ?? Date.now();
}

function formatHealthStatus(status: KbSourceHealthStatus) {
  switch (status) {
    case "healthy":
      return "Healthy";
    case "warning":
      return "Warning";
    case "stale":
      return "Stale";
    case "failed":
      return "Failed";
    default:
      return "Unknown";
  }
}

function formatFreshness(freshness: RetrievalFreshness) {
  switch (freshness) {
    case "fresh":
      return "Fresh";
    case "stale":
      return "Stale";
    default:
      return "Unknown";
  }
}

function formatAvailability(availability: RetrievalSourceAvailability) {
  switch (availability) {
    case "available":
      return "Available";
    case "degraded":
      return "Degraded";
    case "unavailable":
      return "Unavailable";
    default:
      return "Unknown";
  }
}

function formatSyncOutcome(outcome: RetrievalSourceSyncOutcome) {
  switch (outcome) {
    case "succeeded":
      return "Succeeded";
    case "failed":
      return "Failed";
    case "pending":
      return "Pending";
    default:
      return "Unknown";
  }
}

function formatTimestampWithRecency(
  timestamp: string | null | undefined,
  observedAtMs: number,
  fallback: string
) {
  const timestampMs = parseTimestamp(timestamp);
  if (timestampMs == null) {
    return fallback;
  }

  const absolute = new Intl.DateTimeFormat("en-US", {
    day: "numeric",
    month: "short",
    year: "numeric"
  }).format(new Date(timestampMs));
  const elapsedMs = Math.max(observedAtMs - timestampMs, 0);
  const elapsedHours = Math.floor(elapsedMs / 3_600_000);
  const recency =
    elapsedHours < 1
      ? "less than an hour ago"
      : elapsedHours < 24
        ? `${elapsedHours}h ago`
        : `${Math.floor(elapsedHours / 24)}d ago`;

  return `${absolute} · ${recency}`;
}

function latestTimestamp(values: Array<string | null>) {
  const timestamps = values
    .map((value) => ({
      value,
      timestampMs: parseTimestamp(value)
    }))
    .filter(
      (entry): entry is { value: string; timestampMs: number } =>
        entry.value != null && entry.timestampMs != null
    )
    .sort((left, right) => right.timestampMs - left.timestampMs);

  return timestamps[0]?.value ?? null;
}

function parseTimestamp(value: string | number | Date | null | undefined) {
  if (value == null) {
    return null;
  }

  const timestampMs =
    value instanceof Date
      ? value.getTime()
      : typeof value === "number"
        ? value
        : Date.parse(value);

  return Number.isFinite(timestampMs) ? timestampMs : null;
}

function normalizeTimestamp(value: unknown) {
  return typeof value === "string" && parseTimestamp(value) != null ? value : null;
}

function normalizeCount(value: unknown) {
  return typeof value === "number" && Number.isInteger(value) && value >= 0
    ? value
    : null;
}

function normalizeHealthStatus(value: unknown): RetrievalSourceHealthStatus | null {
  if (value === "sync_failed") {
    return "failed";
  }

  return value === "healthy" ||
    value === "warning" ||
    value === "stale" ||
    value === "failed" ||
    value === "unknown"
    ? value
    : null;
}

function normalizeFreshness(value: unknown): RetrievalFreshness | null {
  return value === "fresh" || value === "stale" || value === "unknown"
    ? value
    : null;
}

function normalizeAvailability(
  value: unknown
): RetrievalSourceAvailability | null {
  return value === "available" ||
    value === "degraded" ||
    value === "unavailable" ||
    value === "unknown"
    ? value
    : null;
}

function normalizeSyncOutcome(
  value: unknown
): RetrievalSourceSyncOutcome | null {
  return value === "succeeded" ||
    value === "failed" ||
    value === "pending" ||
    value === "unknown"
    ? value
    : null;
}

function hasReportedHealthMetadata(
  metadata: RetrievalSourceHealthMetadata | null | undefined
) {
  if (!metadata || typeof metadata !== "object") {
    return false;
  }

  return Object.values(metadata).some(
    (value) =>
      value !== null &&
      value !== undefined &&
      (!Array.isArray(value) || value.length > 0)
  );
}

function cleanText(value: unknown) {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function uniqueText(values: readonly string[]) {
  return [...new Set(values.map((value) => value.trim()).filter(Boolean))];
}

function countLabel(count: number, noun: string) {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}
