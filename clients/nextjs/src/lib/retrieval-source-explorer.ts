import type {
  RetrievalChunkSummary,
  RetrievalQuality,
  RetrievalSourceSummary
} from "./assistant-client";
import type { RetrievalRuntimeModel } from "./retrieval-runtime";

export type RetrievalExplorerChunk = RetrievalChunkSummary & {
  confidence: RetrievalQuality;
  confidenceLabel: string;
  usedInContext: boolean;
  contextStatusLabel: string;
};

export type RetrievalExplorerSource = Omit<RetrievalSourceSummary, "chunks"> & {
  chunks: RetrievalExplorerChunk[];
  selectedForContext: boolean;
  contextStatusLabel: string;
  chunkCountLabel: string;
  usedChunkCount: number;
  rankRangeLabel: string;
  coverageLabel: string;
  coverageDetail: string;
  contextReason: string;
  isTopContributor: boolean;
};

export type RetrievalSourceExplorerModel = {
  sources: RetrievalExplorerSource[];
  selectedSourceCount: number;
  ignoredSourceCount: number;
  overviewLabel: string;
  contributionLabel: string;
};

export function buildRetrievalSourceExplorerModel(
  runtime: RetrievalRuntimeModel
): RetrievalSourceExplorerModel {
  const totalUsedChunks = runtime.sources.reduce(
    (total, source) =>
      total +
      source.chunks.filter((chunk) => resolveChunkUsage(chunk, source)).length,
    0
  );
  const topContributorId = findTopContributorId(runtime.sources);
  const sources = runtime.sources.map((source) =>
    buildExplorerSource(source, totalUsedChunks, topContributorId)
  );
  const selectedSourceCount = sources.filter(
    (source) => source.selectedForContext
  ).length;
  const ignoredSourceCount = sources.length - selectedSourceCount;
  const topContributor = sources.find((source) => source.isTopContributor);

  return {
    sources,
    selectedSourceCount,
    ignoredSourceCount,
    overviewLabel: [
      countLabel(selectedSourceCount, "selected source"),
      ignoredSourceCount > 0
        ? countLabel(ignoredSourceCount, "ignored source")
        : "No ignored sources reported"
    ].join(" · "),
    contributionLabel: topContributor
      ? `${topContributor.title} contributed most with ${topContributor.coverageLabel.toLowerCase()}.`
      : "No source contribution is available yet."
  };
}

function buildExplorerSource(
  source: RetrievalSourceSummary,
  totalUsedChunks: number,
  topContributorId: string | null
): RetrievalExplorerSource {
  const chunks = source.chunks.map((chunk) => buildExplorerChunk(chunk, source));
  const usedChunkCount = chunks.filter((chunk) => chunk.usedInContext).length;
  const selectedForContext =
    source.selectedForContext ?? usedChunkCount > 0;
  const ranks = chunks
    .map((chunk) => chunk.rank)
    .filter((rank): rank is number => rank != null)
    .sort((first, second) => first - second);
  const contributionPercent =
    totalUsedChunks > 0 ? Math.round((usedChunkCount / totalUsedChunks) * 100) : 0;

  return {
    ...source,
    chunks,
    selectedForContext,
    contextStatusLabel: selectedForContext
      ? "Selected for context"
      : "Not selected",
    chunkCountLabel: countLabel(chunks.length, "retrieved chunk"),
    usedChunkCount,
    rankRangeLabel: buildRankRangeLabel(ranks),
    coverageLabel:
      totalUsedChunks > 0
        ? `${usedChunkCount}/${totalUsedChunks} context chunks`
        : "No context coverage",
    coverageDetail:
      usedChunkCount > 0
        ? `${contributionPercent}% of the final retrieval context came from this source.`
        : "This source did not contribute a chunk to the final retrieval context.",
    contextReason: selectedForContext
      ? source.whyUsed
      : "Retrieved as a candidate source but not included in the final context; no exclusion reason was recorded.",
    isTopContributor: source.sourceId === topContributorId
  };
}

function buildExplorerChunk(
  chunk: RetrievalChunkSummary,
  source: RetrievalSourceSummary
): RetrievalExplorerChunk {
  const confidence = deriveConfidence(chunk.score);
  const usedInContext = resolveChunkUsage(chunk, source);

  return {
    ...chunk,
    confidence,
    confidenceLabel: formatConfidenceLabel(confidence),
    usedInContext,
    contextStatusLabel: usedInContext ? "Used in context" : "Not used"
  };
}

function findTopContributorId(sources: RetrievalSourceSummary[]) {
  const rankedSources = sources
    .map((source) => ({
      source,
      usedChunkCount: source.chunks.filter((chunk) =>
        resolveChunkUsage(chunk, source)
      ).length
    }))
    .filter(({ usedChunkCount }) => usedChunkCount > 0)
    .sort(
      (first, second) =>
        second.usedChunkCount - first.usedChunkCount ||
        (first.source.bestRank ?? Number.MAX_SAFE_INTEGER) -
          (second.source.bestRank ?? Number.MAX_SAFE_INTEGER)
    );

  return rankedSources[0]?.source.sourceId ?? null;
}

function resolveChunkUsage(
  chunk: RetrievalChunkSummary,
  source: RetrievalSourceSummary
) {
  return chunk.usedInContext ?? source.selectedForContext ?? true;
}

function deriveConfidence(score: number | null): RetrievalQuality {
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

function formatConfidenceLabel(confidence: RetrievalQuality) {
  switch (confidence) {
    case "high":
      return "High confidence";
    case "medium":
      return "Medium confidence";
    case "low":
      return "Low confidence";
    default:
      return "Confidence unknown";
  }
}

function buildRankRangeLabel(ranks: number[]) {
  if (ranks.length === 0) {
    return "Rank unavailable";
  }

  if (ranks.length === 1 || ranks[0] === ranks[ranks.length - 1]) {
    return `Rank #${ranks[0]}`;
  }

  return `Ranks #${ranks[0]}–#${ranks[ranks.length - 1]}`;
}

function countLabel(count: number, noun: string) {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}
