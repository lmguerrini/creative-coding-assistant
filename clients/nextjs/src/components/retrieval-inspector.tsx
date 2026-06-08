import type {
  RetrievalChunkSummary,
  RetrievalSourceSummary
} from "@/lib/assistant-client";
import type { RetrievalRuntimeModel } from "@/lib/retrieval-runtime";
import { SubsystemErrorCallout } from "./subsystem-error-callout";

export function RetrievalInspector({ runtime }: { runtime: RetrievalRuntimeModel }) {
  const hasSources = runtime.sources.length > 0;

  return (
    <section
      aria-label="Retrieval inspector"
      className="inspectorPanel retrievalPanel"
      id="retrieval-inspector-panel"
      role="tabpanel"
    >
      <article
        aria-label="Retrieval status"
        className="retrievalSummaryCard"
        data-state={runtime.summary.state}
        role="group"
      >
        <header className="retrievalSummaryHeader">
          <div>
            <span>Retrieval status</span>
            <strong>{runtime.summary.status}</strong>
            <p>{runtime.summary.headline}</p>
          </div>
          <span className="retrievalStateBadge" data-state={runtime.summary.state}>
            {runtime.summary.providerLabel}
          </span>
        </header>
        <div className="retrievalSummaryMeta" aria-label="Retrieval metadata" role="list">
          <span role="listitem">{runtime.summary.sourceCount} sources</span>
          <span role="listitem">{runtime.summary.usedChunkLabel}</span>
          <span role="listitem">{runtime.summary.qualityLabel}</span>
          <span role="listitem">{runtime.summary.freshnessLabel}</span>
        </div>
        <div className="retrievalInsightGrid" aria-label="Retrieval quality signals">
          <article
            aria-label="Retrieval confidence"
            className="retrievalInsightCard"
            data-confidence={runtime.summary.confidence}
            role="group"
          >
            <span>Confidence</span>
            <strong>{runtime.summary.confidenceLabel}</strong>
            <p>{runtime.summary.confidenceDetail}</p>
          </article>
          <article
            aria-label="Retrieval coverage"
            className="retrievalInsightCard"
            role="group"
          >
            <span>Coverage</span>
            <strong>{runtime.summary.coverageLabel}</strong>
            <p>{runtime.summary.coverageDetail}</p>
          </article>
          <article
            aria-label="Retrieval context used"
            className="retrievalInsightCard"
            role="group"
          >
            <span>Generation context</span>
            <strong>{runtime.summary.usedChunkLabel}</strong>
            <p>{runtime.summary.usedChunkDetail}</p>
          </article>
        </div>
        {runtime.request.query ? (
          <p className="retrievalQuery">
            Query
            <code>{runtime.request.query}</code>
          </p>
        ) : null}
        {runtime.request.filterLabels.length > 0 ? (
          <div className="retrievalFilterRow" aria-label="Retrieval filters">
            {runtime.request.filterLabels.map((label) => (
              <span className="retrievalFilterPill" key={label}>
                {label}
              </span>
            ))}
          </div>
        ) : null}
        <p className="retrievalSummaryDetail">{runtime.summary.detail}</p>
        {runtime.summary.error ? (
          <SubsystemErrorCallout
            className="retrievalErrorCallout"
            error={runtime.summary.error}
            title="Retrieval failed"
          />
        ) : null}
        {runtime.summary.warning ? (
          <p className="retrievalWarning" role="status">
            {runtime.summary.warning}
          </p>
        ) : null}
      </article>
      {hasSources ? (
        <div className="retrievalList">
          {runtime.sources.map((source) => (
            <RetrievalSourceCard key={source.sourceId} source={source} />
          ))}
        </div>
      ) : (
        <article
          aria-label="Retrieval empty state"
          className="retrievalEmptyCard"
          data-state={runtime.summary.state}
          role="group"
        >
          <strong>{runtime.summary.headline}</strong>
          <p>{runtime.summary.detail}</p>
        </article>
      )}
    </section>
  );
}

function RetrievalSourceCard({ source }: { source: RetrievalSourceSummary }) {
  return (
    <article className="retrievalItem">
      <header className="retrievalItemHeader">
        <div>
          <div className="retrievalItemMeta">
            <span className="retrievalDomainBadge">{source.domainLabel}</span>
            <span className="retrievalSourceType">{source.sourceTypeLabel}</span>
            <code className="retrievalSourceId">{source.sourceId}</code>
          </div>
          <strong>{source.title}</strong>
          <p>
            {source.publisher}
            {source.host ? ` • ${source.host}` : ""}
          </p>
        </div>
        <div className="retrievalItemSignals">
          {source.bestRank != null ? (
            <span className="retrievalRankBadge">{`Best rank #${source.bestRank}`}</span>
          ) : null}
          <span className="retrievalScoreBadge" data-quality={source.quality}>
            {source.qualityLabel}
          </span>
          <span
            className="retrievalFreshnessBadge"
            data-freshness={source.freshness}
          >
            {source.freshnessLabel}
          </span>
        </div>
      </header>
      <p className="retrievalWhyUsed">{source.whyUsed}</p>
      <div className="retrievalChunkList" aria-label={`${source.title} chunks`}>
        {source.chunks.map((chunk) => (
          <RetrievalChunkCard chunk={chunk} domainLabel={source.domainLabel} key={chunk.id} />
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

function RetrievalChunkCard({
  chunk,
  domainLabel
}: {
  chunk: RetrievalChunkSummary;
  domainLabel: string;
}) {
  return (
    <article className="retrievalChunk" data-rank={chunk.rank ?? undefined}>
      <header>
        <div className="retrievalChunkRank">
          <strong>{chunk.rank != null ? `Rank #${chunk.rank}` : "Rank unavailable"}</strong>
          <span>{`Source chunk ${chunk.chunkIndex + 1}`}</span>
        </div>
        <div className="retrievalChunkSignals">
          {chunk.score != null ? <span>{`${formatScore(chunk.score)} score`}</span> : null}
          <span>{chunk.relevanceLabel}</span>
        </div>
      </header>
      <div className="retrievalChunkMeta">
        <span data-domain-match={formatDomainMatchState(chunk.domainMatch)}>
          {formatDomainMatchLabel(chunk.domainMatch, domainLabel)}
        </span>
        {chunk.scoreAdjustment != null ? (
          <span>{`Route adjustment ${formatSignedScore(chunk.scoreAdjustment)}`}</span>
        ) : null}
      </div>
      <p>{chunk.snippet}</p>
      <p className="retrievalSelectionReason">
        <span>Why selected</span>
        {chunk.selectionReason ?? "Selection reasoning was not recorded for this chunk."}
      </p>
    </article>
  );
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
