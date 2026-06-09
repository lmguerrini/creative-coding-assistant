"use client";

import { ChevronRight } from "lucide-react";
import { useState } from "react";
import type {
  RetrievalExplorerChunk,
  RetrievalExplorerSource,
  RetrievalSourceExplorerModel
} from "@/lib/retrieval-source-explorer";
import { KbSourceHealthDashboard } from "./kb-source-health-dashboard";

export function RetrievalSourceExplorer({
  model
}: {
  model: RetrievalSourceExplorerModel;
}) {
  const [selectedSourceId, setSelectedSourceId] = useState(
    model.sources[0]?.sourceId ?? null
  );
  const selectedSource =
    model.sources.find((source) => source.sourceId === selectedSourceId) ??
    model.sources[0];

  if (!selectedSource) {
    return null;
  }

  return (
    <section
      aria-label="Retrieval source explorer"
      className="retrievalExplorer"
    >
      <header className="retrievalExplorerHeader">
        <div>
          <span>Source explorer</span>
          <strong>{model.overviewLabel}</strong>
          <p>{model.contributionLabel}</p>
        </div>
      </header>
      <KbSourceHealthDashboard model={model.health} />
      <div aria-label="Retrieved sources" className="retrievalSourceList" role="list">
        {model.sources.map((source) => (
          <span key={source.sourceId} role="listitem">
            <button
              aria-label={`Inspect source ${source.title}`}
              aria-pressed={source.sourceId === selectedSource.sourceId}
              className="retrievalSourceRow"
              data-selected={source.selectedForContext ? "true" : "false"}
              onClick={() => setSelectedSourceId(source.sourceId)}
              type="button"
            >
              <span className="retrievalSourceRowMain">
                <strong>{source.title}</strong>
                <code>{source.sourceId}</code>
              </span>
              <span className="retrievalSourceRowMeta">
                <small
                  className="kbHealthStatusBadge"
                  data-health={source.health.status}
                >
                  {source.health.statusLabel}
                </small>
                <small>{source.contextStatusLabel}</small>
                <small>{source.rankRangeLabel}</small>
                <ChevronRight aria-hidden="true" size={14} />
              </span>
            </button>
          </span>
        ))}
      </div>
      <RetrievalSourceDetail source={selectedSource} />
    </section>
  );
}

function RetrievalSourceDetail({ source }: { source: RetrievalExplorerSource }) {
  return (
    <article
      aria-label={`${source.title} source details`}
      className="retrievalSourceDetail"
      data-selected={source.selectedForContext ? "true" : "false"}
      role="group"
    >
      <header className="retrievalSourceDetailHeader">
        <div>
          <div className="retrievalItemMeta">
            <span className="retrievalDomainBadge">{source.domainLabel}</span>
            <span className="retrievalSourceType">{source.sourceTypeLabel}</span>
            {source.isTopContributor ? (
              <span className="retrievalContributorBadge">Top contributor</span>
            ) : null}
          </div>
          <strong>{source.title}</strong>
          <code className="retrievalSourceId">{source.sourceId}</code>
          <p>
            {source.publisher}
            {source.host ? ` • ${source.host}` : ""}
          </p>
        </div>
        <span
          className="retrievalContextBadge"
          data-selected={source.selectedForContext ? "true" : "false"}
        >
          {source.contextStatusLabel}
        </span>
      </header>
      <div
        aria-label={`${source.title} source metrics`}
        className="retrievalSourceMetrics"
        role="list"
      >
        <SourceMetric label="Confidence" value={source.qualityLabel} />
        <SourceMetric label="Freshness" value={source.freshnessLabel} />
        <SourceMetric label="Retrieved" value={source.chunkCountLabel} />
        <SourceMetric label="Global rank" value={source.rankRangeLabel} />
        <SourceMetric label="Context share" value={source.coverageLabel} />
      </div>
      <SourceHealthDetail source={source} />
      <p className="retrievalCoverageDetail">{source.coverageDetail}</p>
      <p className="retrievalWhyUsed">{source.contextReason}</p>
      <div className="retrievalChunkList" aria-label={`${source.title} chunks`}>
        {source.chunks.map((chunk) => (
          <RetrievalChunkCard
            chunk={chunk}
            domainLabel={source.domainLabel}
            key={chunk.id}
          />
        ))}
      </div>
      {source.href ? (
        <a
          className="retrievalSourceLink"
          href={source.href}
          rel="noreferrer"
          target="_blank"
        >
          Open source reference
        </a>
      ) : null}
    </article>
  );
}

function SourceHealthDetail({ source }: { source: RetrievalExplorerSource }) {
  const health = source.health;

  return (
    <section
      aria-label={`${source.title} source health`}
      className="kbSourceHealthDetail"
      data-health={health.status}
    >
      <header>
        <div>
          <span>Source health</span>
          <strong>{health.statusLabel}</strong>
          <p>{health.statusDetail}</p>
        </div>
        <div className="kbSourceHealthBadges">
          <span className="kbHealthStatusBadge" data-health={health.status}>
            {health.statusLabel}
          </span>
          <span
            className="kbAvailabilityBadge"
            data-availability={health.availability}
          >
            {health.availabilityLabel}
          </span>
        </div>
      </header>
      <div
        aria-label={`${source.title} health metrics`}
        className="kbSourceHealthMetrics"
        role="list"
      >
        <SourceMetric label="Availability" value={health.availabilityLabel} />
        <SourceMetric label="Freshness" value={health.freshnessLabel} />
        <SourceMetric label="Domain owner" value={health.domainOwner} />
        <SourceMetric label="Indexed chunks" value={health.indexedChunkLabel} />
        <SourceMetric label="Last successful sync" value={health.lastSuccessfulSyncLabel} />
        <SourceMetric label="Last attempted sync" value={health.lastAttemptedSyncLabel} />
        <SourceMetric label="Sync outcome" value={health.syncOutcomeLabel} />
        <SourceMetric label="Source coverage" value={health.coverageLabel} />
      </div>
      <p className="kbSourceHealthCoverage">{health.coverageDetail}</p>
      {health.warnings.length > 0 ? (
        <div
          aria-label={`${source.title} health warnings`}
          className="kbSourceHealthWarnings"
          role="list"
        >
          {health.warnings.map((warning) => (
            <p key={warning} role="listitem">
              {warning}
            </p>
          ))}
        </div>
      ) : null}
    </section>
  );
}

function SourceMetric({ label, value }: { label: string; value: string }) {
  return (
    <span className="retrievalSourceMetric" role="listitem">
      <small>{label}</small>
      <strong>{value}</strong>
    </span>
  );
}

function RetrievalChunkCard({
  chunk,
  domainLabel
}: {
  chunk: RetrievalExplorerChunk;
  domainLabel: string;
}) {
  return (
    <article
      aria-label={`${formatRankLabel(chunk.rank)} source chunk ${chunk.chunkIndex + 1}`}
      className="retrievalChunk"
      data-rank={chunk.rank ?? undefined}
      data-used={chunk.usedInContext ? "true" : "false"}
    >
      <header>
        <div className="retrievalChunkRank">
          <strong>{formatRankLabel(chunk.rank)}</strong>
          <span>{`Source chunk ${chunk.chunkIndex + 1}`}</span>
        </div>
        <span
          className="retrievalContextBadge"
          data-selected={chunk.usedInContext ? "true" : "false"}
        >
          {chunk.contextStatusLabel}
        </span>
      </header>
      <div className="retrievalChunkSignals">
        {chunk.score != null ? <span>{`${formatScore(chunk.score)} score`}</span> : null}
        <span data-confidence={chunk.confidence}>{chunk.confidenceLabel}</span>
        <span>{chunk.relevanceLabel}</span>
      </div>
      <div className="retrievalChunkMeta">
        <span data-domain-match={formatDomainMatchState(chunk.domainMatch)}>
          {formatDomainMatchLabel(chunk.domainMatch, domainLabel)}
        </span>
        {chunk.scoreAdjustment != null ? (
          <span>{`Route adjustment ${formatSignedScore(chunk.scoreAdjustment)}`}</span>
        ) : null}
      </div>
      <p>{chunk.snippet || "No chunk preview is available for this saved source."}</p>
      <p className="retrievalSelectionReason">
        <span>{chunk.usedInContext ? "Why selected" : "Selection note"}</span>
        {chunk.selectionReason ?? "Selection reasoning was not recorded for this chunk."}
      </p>
    </article>
  );
}

function formatRankLabel(rank: number | null | undefined) {
  return rank != null ? `Rank #${rank}` : "Rank unavailable";
}

function formatScore(score: number) {
  return `${Math.round(score * 100)}%`;
}

function formatSignedScore(score: number) {
  const points = Math.round(score * 100);
  return `${points > 0 ? "+" : ""}${points} pts`;
}

function formatDomainMatchState(domainMatch: boolean | null | undefined) {
  if (domainMatch === true) {
    return "match";
  }

  if (domainMatch === false) {
    return "cross-domain";
  }

  return "unfiltered";
}

function formatDomainMatchLabel(
  domainMatch: boolean | null | undefined,
  domainLabel: string
) {
  if (domainMatch === true) {
    return `Domain match · ${domainLabel}`;
  }

  if (domainMatch === false) {
    return `Cross-domain support · ${domainLabel}`;
  }

  return `Unfiltered domain · ${domainLabel}`;
}
