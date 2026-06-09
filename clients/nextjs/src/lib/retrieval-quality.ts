import type {
  RetrievalQuality,
  RetrievalSourceSummary
} from "./assistant-client";
import type { RetrievalRuntimeModel } from "./retrieval-runtime";

export type RetrievalQualityMetricKey =
  | "precision"
  | "diversity"
  | "coverage"
  | "context_sufficiency";

export type RetrievalQualityMetric = {
  key: RetrievalQualityMetricKey;
  label: string;
  level: RetrievalQuality;
  valueLabel: string;
  detail: string;
};

export type RetrievalDomainBalanceStatus =
  | "balanced"
  | "weighted"
  | "focused"
  | "concentrated"
  | "unknown";

export type RetrievalDomainBalanceEntry = {
  domain: string;
  label: string;
  chunkCount: number;
  sharePercent: number;
  shareLabel: string;
  requested: boolean;
};

export type RetrievalDomainBalance = {
  status: RetrievalDomainBalanceStatus;
  label: string;
  detail: string;
  domains: RetrievalDomainBalanceEntry[];
};

export type RetrievalQualityModel = {
  overallLevel: RetrievalQuality;
  overallLabel: string;
  overallDetail: string;
  metrics: RetrievalQualityMetric[];
  domainBalance: RetrievalDomainBalance;
  weaknesses: string[];
  hasEvidence: boolean;
};

type UsedChunk = {
  source: RetrievalSourceSummary;
  score: number | null;
};

export function buildRetrievalQualityModel(
  runtime: RetrievalRuntimeModel
): RetrievalQualityModel {
  const usedChunks = runtime.sources.flatMap((source) =>
    source.chunks
      .filter((chunk) => resolveChunkUsage(source, chunk.usedInContext))
      .map((chunk) => ({
        source,
        score: chunk.score
      }))
  );
  const selectedSources = runtime.sources.filter((source) =>
    usedChunks.some((chunk) => chunk.source.sourceId === source.sourceId)
  );
  const precision = buildPrecisionMetric(usedChunks);
  const diversity = buildDiversityMetric(usedChunks, selectedSources);
  const coverage = buildCoverageMetric(runtime, usedChunks);
  const contextSufficiency = buildContextSufficiencyMetric({
    runtime,
    usedChunks,
    selectedSources,
    precision,
    coverage
  });
  const metrics = [precision, diversity, coverage, contextSufficiency];
  const domainBalance = buildDomainBalance(runtime, usedChunks);
  const weaknesses = buildWeaknesses({
    usedChunks,
    selectedSources,
    diversity,
    coverage,
    domainBalance
  });
  const overallLevel = buildOverallLevel(runtime, metrics, usedChunks.length);

  return {
    overallLevel,
    overallLabel: formatOverallLabel(overallLevel),
    overallDetail: buildOverallDetail({
      runtime,
      overallLevel,
      precision,
      diversity,
      coverage,
      contextSufficiency,
      usedChunkCount: usedChunks.length
    }),
    metrics,
    domainBalance,
    weaknesses,
    hasEvidence: usedChunks.length > 0
  };
}

function buildPrecisionMetric(usedChunks: UsedChunk[]): RetrievalQualityMetric {
  const scores = usedChunks
    .map((chunk) => chunk.score)
    .filter((score): score is number => score != null);

  if (scores.length === 0) {
    return {
      key: "precision",
      label: "Precision",
      level: "unknown",
      valueLabel: "Not scored",
      detail:
        usedChunks.length > 0
          ? "Selected chunks have no relevance scores, so score-based precision cannot be verified."
          : "No selected retrieval context is available to assess precision."
    };
  }

  const averageScore = average(scores);
  const scoreLevel = qualityFromScore(averageScore);
  const level =
    scores.length < usedChunks.length && scoreLevel === "high"
      ? "medium"
      : scoreLevel;
  const highConfidenceCount = scores.filter((score) => score >= 0.85).length;

  return {
    key: "precision",
    label: "Precision",
    level,
    valueLabel: `${formatPercent(averageScore)} average`,
    detail: `${scores.length} of ${usedChunks.length} selected chunks scored; ${highConfidenceCount} ${highConfidenceCount === 1 ? "clears" : "clear"} the 85% high-confidence threshold. This is a relevance-score proxy, not offline evaluation precision.`
  };
}

function buildDiversityMetric(
  usedChunks: UsedChunk[],
  selectedSources: RetrievalSourceSummary[]
): RetrievalQualityMetric {
  if (usedChunks.length === 0) {
    return {
      key: "diversity",
      label: "Diversity",
      level: "unknown",
      valueLabel: "No context",
      detail: "Source and domain diversity can be assessed after context is selected."
    };
  }

  const domainCount = uniqueValues(
    usedChunks.map((chunk) => chunk.source.domain).filter(Boolean)
  ).length;
  const publisherCount = uniqueValues(
    selectedSources.map((source) => source.publisher).filter(Boolean)
  ).length;
  const topSourceShare = largestSourceShare(usedChunks);
  const level: RetrievalQuality =
    selectedSources.length >= 2 &&
    (domainCount >= 2 || publisherCount >= 2) &&
    topSourceShare <= 2 / 3
      ? "high"
      : selectedSources.length >= 2 || domainCount >= 2 || publisherCount >= 2
        ? "medium"
        : "low";

  return {
    key: "diversity",
    label: "Diversity",
    level,
    valueLabel: `${countLabel(selectedSources.length, "source")} · ${countLabel(domainCount, "domain")}`,
    detail:
      selectedSources.length > 0
        ? `${countLabel(publisherCount, "publisher")} represented; the largest source contributes ${formatPercent(topSourceShare)} of selected context.`
        : "Selected context does not identify a contributing source."
  };
}

function buildCoverageMetric(
  runtime: RetrievalRuntimeModel,
  usedChunks: UsedChunk[]
): RetrievalQualityMetric {
  const requestedDomains = runtime.request.domains;
  const usedDomainSet = new Set(
    usedChunks.map((chunk) => chunk.source.domain).filter(Boolean)
  );

  if (requestedDomains.length === 0) {
    return {
      key: "coverage",
      label: "Coverage",
      level: "unknown",
      valueLabel:
        usedDomainSet.size > 0
          ? `${countLabel(usedDomainSet.size, "domain")} represented`
          : "Scope not recorded",
      detail:
        "No requested-domain scope was recorded, so retrieval coverage cannot be scored."
    };
  }

  const matchedDomains = requestedDomains.filter((domain) =>
    usedDomainSet.has(domain)
  );
  const missingLabels = requestedDomains
    .map((domain, index) => ({
      domain,
      label: runtime.request.domainLabels[index] ?? domain
    }))
    .filter(({ domain }) => !usedDomainSet.has(domain))
    .map(({ label }) => label);
  const ratio = matchedDomains.length / requestedDomains.length;
  const level: RetrievalQuality =
    ratio === 1 ? "high" : ratio >= 0.5 ? "medium" : "low";

  return {
    key: "coverage",
    label: "Coverage",
    level,
    valueLabel: `${matchedDomains.length}/${requestedDomains.length} requested domains`,
    detail:
      missingLabels.length === 0
        ? "Every requested domain contributed selected retrieval context."
        : `Selected context is missing ${missingLabels.join(", ")}.`
  };
}

function buildContextSufficiencyMetric({
  runtime,
  usedChunks,
  selectedSources,
  precision,
  coverage
}: {
  runtime: RetrievalRuntimeModel;
  usedChunks: UsedChunk[];
  selectedSources: RetrievalSourceSummary[];
  precision: RetrievalQualityMetric;
  coverage: RetrievalQualityMetric;
}): RetrievalQualityMetric {
  if (usedChunks.length === 0) {
    const retrievalWasAttempted =
      runtime.request.query != null || runtime.request.domains.length > 0;

    return {
      key: "context_sufficiency",
      label: "Context sufficiency",
      level:
        retrievalWasAttempted && runtime.summary.state === "empty"
          ? "low"
          : "unknown",
      valueLabel: retrievalWasAttempted ? "No selected context" : "Not assessed",
      detail: retrievalWasAttempted
        ? "The retrieval pass did not provide context for generation."
        : "Context sufficiency can be assessed after retrieval runs."
    };
  }

  const hasBroadSupport =
    usedChunks.length >= 3 &&
    selectedSources.length >= 2 &&
    coverage.level !== "low" &&
    precision.level !== "low";
  const level: RetrievalQuality = hasBroadSupport
    ? "high"
    : precision.level === "low" || coverage.level === "low"
      ? "low"
      : "medium";

  return {
    key: "context_sufficiency",
    label: "Context sufficiency",
    level,
    valueLabel: `${countLabel(usedChunks.length, "chunk")} selected`,
    detail: `${countLabel(selectedSources.length, "source")} contributed to generation context${
      precision.level === "unknown"
        ? "; relevance scores were not recorded."
        : "."
    }`
  };
}

function buildDomainBalance(
  runtime: RetrievalRuntimeModel,
  usedChunks: UsedChunk[]
): RetrievalDomainBalance {
  const requestedLabelByDomain = new Map(
    runtime.request.domains.map((domain, index) => [
      domain,
      runtime.request.domainLabels[index] ?? domain
    ])
  );
  const sourceLabelByDomain = new Map(
    runtime.sources
      .filter((source) => source.domain)
      .map((source) => [source.domain, source.domainLabel])
  );
  const chunkCountByDomain = new Map<string, number>();

  for (const chunk of usedChunks) {
    const domain = chunk.source.domain || "unknown";
    chunkCountByDomain.set(domain, (chunkCountByDomain.get(domain) ?? 0) + 1);
  }

  const orderedDomains = [
    ...runtime.request.domains,
    ...Array.from(chunkCountByDomain.keys()).filter(
      (domain) => !requestedLabelByDomain.has(domain)
    )
  ];
  const domains = uniqueValues(orderedDomains).map((domain) => {
    const chunkCount = chunkCountByDomain.get(domain) ?? 0;
    const sharePercent =
      usedChunks.length > 0 ? Math.round((chunkCount / usedChunks.length) * 100) : 0;

    return {
      domain,
      label:
        requestedLabelByDomain.get(domain) ??
        sourceLabelByDomain.get(domain) ??
        formatDomainLabel(domain),
      chunkCount,
      sharePercent,
      shareLabel: `${sharePercent}% of context`,
      requested: requestedLabelByDomain.has(domain)
    };
  });

  if (usedChunks.length === 0 || domains.length === 0) {
    return {
      status: "unknown",
      label: "Domain balance unavailable",
      detail: "No selected context is available for a domain distribution.",
      domains
    };
  }

  const representedDomains = domains.filter((domain) => domain.chunkCount > 0);
  const dominantDomain = [...representedDomains].sort(
    (first, second) => second.chunkCount - first.chunkCount
  )[0]!;
  const missingRequestedCount = domains.filter(
    (domain) => domain.requested && domain.chunkCount === 0
  ).length;
  const status: RetrievalDomainBalanceStatus =
    missingRequestedCount > 0
      ? "concentrated"
      : representedDomains.length === 1
        ? "focused"
        : dominantDomain.sharePercent <= 67
          ? "balanced"
          : dominantDomain.sharePercent <= 80
            ? "weighted"
            : "concentrated";

  return {
    status,
    label:
      status === "balanced"
        ? `Balanced across ${countLabel(representedDomains.length, "domain")}`
        : status === "weighted"
          ? `Weighted toward ${dominantDomain.label}`
          : status === "focused"
            ? `Focused on ${dominantDomain.label}`
            : `Concentrated in ${dominantDomain.label}`,
    detail:
      missingRequestedCount > 0
        ? `${countLabel(missingRequestedCount, "requested domain")} did not contribute selected context.`
        : `${dominantDomain.label} contributes ${dominantDomain.shareLabel}.`,
    domains
  };
}

function buildWeaknesses({
  usedChunks,
  selectedSources,
  diversity,
  coverage,
  domainBalance
}: {
  usedChunks: UsedChunk[];
  selectedSources: RetrievalSourceSummary[];
  diversity: RetrievalQualityMetric;
  coverage: RetrievalQualityMetric;
  domainBalance: RetrievalDomainBalance;
}) {
  if (usedChunks.length === 0) {
    return [];
  }

  const weaknesses: string[] = [];
  const scores = usedChunks
    .map((chunk) => chunk.score)
    .filter((score): score is number => score != null);
  if (scores.length === 0) {
    weaknesses.push(
      "Relevance scores were not recorded, so precision cannot be verified."
    );
  } else if (average(scores) < 0.7) {
    weaknesses.push("Average selected-chunk relevance is below 70%.");
  } else if (average(scores) < 0.85) {
    weaknesses.push("Average selected-chunk relevance is moderate rather than high.");
  }

  if (scores.length > 0 && scores.length < usedChunks.length) {
    weaknesses.push(
      `Only ${scores.length} of ${usedChunks.length} selected chunks include relevance scores.`
    );
  }

  if (coverage.level === "medium" || coverage.level === "low") {
    weaknesses.push(coverage.detail);
  }

  if (diversity.level === "low" && selectedSources.length <= 1) {
    weaknesses.push("Generation context depends on a single source.");
  } else {
    const topShare = largestSourceShare(usedChunks);
    if (selectedSources.length > 1 && topShare > 0.75) {
      weaknesses.push(
        `One source supplies ${formatPercent(topShare)} of selected context.`
      );
    }
  }

  if (usedChunks.length === 1) {
    weaknesses.push("Only one chunk was selected for generation context.");
  }

  if (domainBalance.status === "concentrated" && coverage.level === "unknown") {
    weaknesses.push("Selected context is concentrated in one domain.");
  }

  const staleSourceCount = selectedSources.filter(
    (source) => source.freshness === "stale"
  ).length;
  if (staleSourceCount > 0) {
    weaknesses.push(
      `${countLabel(staleSourceCount, "selected source")} may be stale.`
    );
  }

  return uniqueValues(weaknesses);
}

function buildOverallLevel(
  runtime: RetrievalRuntimeModel,
  metrics: RetrievalQualityMetric[],
  usedChunkCount: number
): RetrievalQuality {
  if (usedChunkCount === 0) {
    const retrievalWasAttempted =
      runtime.request.query != null || runtime.request.domains.length > 0;
    return retrievalWasAttempted && runtime.summary.state === "empty"
      ? "low"
      : "unknown";
  }

  const metricByKey = new Map(metrics.map((metric) => [metric.key, metric]));
  const precision = metricByKey.get("precision")!;
  const diversity = metricByKey.get("diversity")!;
  const coverage = metricByKey.get("coverage")!;
  const contextSufficiency = metricByKey.get("context_sufficiency")!;

  if (
    precision.level === "low" ||
    coverage.level === "low" ||
    contextSufficiency.level === "low"
  ) {
    return "low";
  }

  if (
    precision.level === "high" &&
    contextSufficiency.level === "high" &&
    diversity.level !== "low"
  ) {
    return "high";
  }

  return "medium";
}

function buildOverallDetail({
  runtime,
  overallLevel,
  precision,
  diversity,
  coverage,
  contextSufficiency,
  usedChunkCount
}: {
  runtime: RetrievalRuntimeModel;
  overallLevel: RetrievalQuality;
  precision: RetrievalQualityMetric;
  diversity: RetrievalQualityMetric;
  coverage: RetrievalQualityMetric;
  contextSufficiency: RetrievalQualityMetric;
  usedChunkCount: number;
}) {
  if (overallLevel === "unknown") {
    if (runtime.summary.state === "pending") {
      return "Quality will be assessed when the active retrieval pass completes.";
    }

    return "No selected retrieval evidence is available for a quality assessment.";
  }

  if (overallLevel === "low" && usedChunkCount === 0) {
    return "Retrieval was requested, but no context was selected for generation.";
  }

  if (overallLevel === "high") {
    return `Quality is high because ${precision.valueLabel.toLowerCase()} is supported by ${diversity.valueLabel.toLowerCase()} and ${contextSufficiency.valueLabel.toLowerCase()}.`;
  }

  if (overallLevel === "low") {
    const lowSignals = [precision, coverage, contextSufficiency]
      .filter((metric) => metric.level === "low")
      .map((metric) => metric.label.toLowerCase());

    return `Quality is low because ${joinList(lowSignals)} ${
      lowSignals.length === 1 ? "is" : "are"
    } below the reliable range.`;
  }

  const limitingSignals = [precision, diversity, coverage, contextSufficiency]
    .filter((metric) => metric.level === "medium" || metric.level === "unknown")
    .map((metric) => metric.label.toLowerCase());

  return `Quality is medium because ${joinList(limitingSignals)} ${
    limitingSignals.length === 1 ? "remains" : "remain"
  } the limiting signal${limitingSignals.length === 1 ? "" : "s"}.`;
}

function resolveChunkUsage(
  source: RetrievalSourceSummary,
  usedInContext: boolean | null | undefined
) {
  return usedInContext ?? source.selectedForContext ?? true;
}

function largestSourceShare(usedChunks: UsedChunk[]) {
  if (usedChunks.length === 0) {
    return 0;
  }

  const countBySource = new Map<string, number>();
  for (const chunk of usedChunks) {
    countBySource.set(
      chunk.source.sourceId,
      (countBySource.get(chunk.source.sourceId) ?? 0) + 1
    );
  }

  return Math.max(...countBySource.values()) / usedChunks.length;
}

function qualityFromScore(score: number): RetrievalQuality {
  if (score >= 0.85) {
    return "high";
  }

  if (score >= 0.7) {
    return "medium";
  }

  return "low";
}

function formatOverallLabel(level: RetrievalQuality) {
  switch (level) {
    case "high":
      return "High retrieval quality";
    case "medium":
      return "Medium retrieval quality";
    case "low":
      return "Low retrieval quality";
    default:
      return "Retrieval quality unknown";
  }
}

function formatDomainLabel(domain: string) {
  if (domain === "unknown") {
    return "Unknown domain";
  }

  return domain
    .split("_")
    .map((segment) => segment.charAt(0).toUpperCase() + segment.slice(1))
    .join(" ");
}

function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`;
}

function average(values: number[]) {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function uniqueValues<T>(values: T[]) {
  return Array.from(new Set(values));
}

function countLabel(count: number, noun: string) {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

function joinList(values: string[]) {
  if (values.length <= 1) {
    return values[0] ?? "available evidence";
  }

  if (values.length === 2) {
    return values.join(" and ");
  }

  return `${values.slice(0, -1).join(", ")}, and ${values.at(-1)}`;
}
