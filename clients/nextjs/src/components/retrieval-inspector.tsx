import type { RetrievalRuntimeModel } from "@/lib/retrieval-runtime";
import { buildRetrievalQualityModel } from "@/lib/retrieval-quality";
import { buildRetrievalSourceExplorerModel } from "@/lib/retrieval-source-explorer";
import { KbSourceHealthDashboard } from "./kb-source-health-dashboard";
import { RetrievalQualityDeepDive } from "./retrieval-quality-deep-dive";
import { RetrievalSourceExplorer } from "./retrieval-source-explorer";
import { SubsystemErrorCallout } from "./subsystem-error-callout";

export function RetrievalInspector({
  runtime,
  showDebugPanels = true
}: {
  runtime: RetrievalRuntimeModel;
  showDebugPanels?: boolean;
}) {
  const hasSources = runtime.sources.length > 0;
  const quality = buildRetrievalQualityModel(runtime);
  const explorer = buildRetrievalSourceExplorerModel(runtime);

  if (!showDebugPanels) {
    return (
      <section
        aria-label="Retrieval inspector"
        className="inspectorPanel retrievalPanel"
        id="retrieval-inspector-panel"
        role="tabpanel"
      >
        <RetrievalRunStatusSurface runtime={runtime} />
        <KnowledgeBaseStatusSurface runtime={runtime} showSourceHealth />
      </section>
    );
  }

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

export function RetrievalRunStatusSurface({
  runtime
}: {
  runtime: RetrievalRuntimeModel;
}) {
  const runStatusCopy = buildRunRetrievalStatusCopy(runtime);

  return (
    <article
      aria-label="Retrieval status"
      className="retrievalSummaryCard retrievalSummaryCard--user"
      data-state={runtime.summary.state}
      role="group"
    >
      <header className="retrievalSummaryHeader">
        <div>
          <span>Retrieval for this run</span>
          <strong>{runStatusCopy.title}</strong>
          <p>{runStatusCopy.detail}</p>
        </div>
        <span className="retrievalStateBadge" data-state={runtime.summary.state}>
          {runtime.summary.status}
        </span>
      </header>
      <div className="retrievalSummaryMeta" aria-label="Retrieval run metadata" role="list">
        <span role="listitem">{runtime.summary.sourceCount} retrieved sources</span>
        <span role="listitem">{runtime.summary.chunkCount} retrieved chunks</span>
        <span role="listitem">{runtime.summary.domainCount} retrieved domains</span>
      </div>
      {runtime.summary.error ? (
        <SubsystemErrorCallout
          className="retrievalErrorCallout"
          error={runtime.summary.error}
          title="Retrieval failed"
        />
      ) : null}
    </article>
  );
}

export function KnowledgeBaseStatusSurface({
  runtime,
  showSourceHealth = false
}: {
  runtime: RetrievalRuntimeModel;
  showSourceHealth?: boolean;
}) {
  const explorer = buildRetrievalSourceExplorerModel(runtime);

  return (
    <article
      aria-label="Knowledge Base status"
      className="kbStatusSurface"
      id="retrieval-kb-status"
      role="group"
    >
      <header>
        <div>
          <span>Knowledge Base status</span>
          <strong>{explorer.health.statusLabel}</strong>
          <p>{buildKnowledgeBaseStatusDetail(runtime)}</p>
        </div>
      </header>
      <div className="kbStatusMetrics" aria-label="Knowledge Base metrics" role="list">
        <KbStatusMetric
          label="Indexed sources"
          value={
            explorer.health.sourceCount > 0
              ? `${explorer.health.sourceCount} sources observed`
              : "Not reported in this session"
          }
        />
        <KbStatusMetric
          label="Indexed domains"
          value={
            runtime.summary.domainCount > 0
              ? `${runtime.summary.domainCount} retrieved domains`
              : "No domain metrics for this run"
          }
        />
        <KbStatusMetric
          label="Indexed chunks"
          value={explorer.health.indexedChunkLabel}
        />
        <KbStatusMetric
          label="Last sync/fetch"
          value={explorer.health.latestSyncAttemptLabel}
        />
        <KbStatusMetric
          label="Official docs coverage"
          value={
            explorer.health.availableSourceCount > 0
              ? `${explorer.health.availableSourceCount}/${explorer.health.sourceCount} observed sources available`
              : "Coverage not reported in this session"
          }
        />
        <KbStatusMetric
          label="Local/personal docs"
          value="Not connected to this UI"
        />
      </div>
      <div className="kbStatusActions" aria-label="Knowledge Base actions">
        <a className="kbStatusAction" href="#retrieval-kb-status">
          Check KB status
        </a>
        <button
          disabled
          title="Run the documented backend sync command; UI refresh is future scope."
          type="button"
        >
          Refresh official KB
        </button>
      </div>
      <p className="kbStatusGuidance">
        KB refresh is not executed from this UI. Use the documented local sync
        command before the demo if official docs need to be refreshed.
      </p>
      {showSourceHealth ? <KbSourceHealthDashboard model={explorer.health} /> : null}
    </article>
  );
}

function KbStatusMetric({ label, value }: { label: string; value: string }) {
  return (
    <span className="kbStatusMetric" role="listitem">
      <small>{label}</small>
      <strong>{value}</strong>
    </span>
  );
}

function buildRunRetrievalStatusCopy(runtime: RetrievalRuntimeModel) {
  if (runtime.summary.state === "available") {
    return {
      title: "Retrieved context available",
      detail: `${runtime.summary.chunkCount} retrieved chunks from ${runtime.summary.sourceCount} sources were available for this run.`
    };
  }

  if (runtime.summary.state === "empty") {
    return {
      title: "No retrieved context for this run.",
      detail:
        "This only describes the current workflow. It does not mean the Knowledge Base is empty."
    };
  }

  if (runtime.summary.state === "pending") {
    return {
      title: "Retrieval is running",
      detail: "The current workflow is still waiting for retrieval context."
    };
  }

  if (runtime.summary.state === "unavailable") {
    return {
      title: "Knowledge Base unavailable",
      detail:
        "The app did not receive retrieval availability for this session. Check backend and Chroma setup before relying on RAG."
    };
  }

  if (runtime.summary.state === "error") {
    return {
      title: "Retrieval failed for this run",
      detail:
        runtime.summary.error?.userMessage ??
        "Retrieval failed for the current workflow. This does not prove the Knowledge Base is empty."
    };
  }

  return {
    title: "Knowledge Base exists but was not used",
    detail:
      "No retrieval request is attached to this current workflow. Use a retrieval-grounded scenario to exercise RAG."
  };
}

function buildKnowledgeBaseStatusDetail(runtime: RetrievalRuntimeModel) {
  if (runtime.summary.sourceCount > 0) {
    return `${runtime.summary.sourceCount} retrieved sources and ${runtime.summary.chunkCount} chunks are visible from the current session.`;
  }

  if (runtime.summary.state === "empty") {
    return "No retrieved context for this run; persistent KB inventory is not exposed by this UI without a retrieval result.";
  }

  if (runtime.summary.state === "unavailable" || runtime.summary.state === "error") {
    return "KB availability could not be confirmed from the current workflow.";
  }

  return "KB inventory is setup-ready, but this workflow has not surfaced indexed-source metrics.";
}
