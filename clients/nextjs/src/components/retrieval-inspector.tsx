import type { RetrievalRuntimeModel } from "@/lib/retrieval-runtime";
import { buildRetrievalQualityModel } from "@/lib/retrieval-quality";
import { buildRetrievalSourceExplorerModel } from "@/lib/retrieval-source-explorer";
import { RetrievalQualityDeepDive } from "./retrieval-quality-deep-dive";
import { RetrievalSourceExplorer } from "./retrieval-source-explorer";
import { SubsystemErrorCallout } from "./subsystem-error-callout";

export function RetrievalInspector({ runtime }: { runtime: RetrievalRuntimeModel }) {
  const hasSources = runtime.sources.length > 0;
  const quality = buildRetrievalQualityModel(runtime);
  const explorer = buildRetrievalSourceExplorerModel(runtime);

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
      <RetrievalQualityDeepDive model={quality} />
      {hasSources ? (
        <RetrievalSourceExplorer model={explorer} />
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
