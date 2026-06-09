"use client";

import { ChevronDown } from "lucide-react";
import { useEffect, useState } from "react";
import type {
  RetrievalDomainBalanceEntry,
  RetrievalQualityMetric,
  RetrievalQualityModel
} from "@/lib/retrieval-quality";

export function RetrievalQualityDeepDive({
  model
}: {
  model: RetrievalQualityModel;
}) {
  const [isOpen, setIsOpen] = useState(model.hasEvidence);

  useEffect(() => {
    setIsOpen(model.hasEvidence);
  }, [model.hasEvidence]);

  return (
    <section
      aria-label="Retrieval quality deep dive"
      className="retrievalQualityDeepDive"
      data-open={isOpen ? "true" : "false"}
      data-quality={model.overallLevel}
    >
      <button
        aria-expanded={isOpen}
        aria-label="Toggle retrieval quality deep dive"
        className="retrievalQualityToggle"
        onClick={() => setIsOpen((currentValue) => !currentValue)}
        type="button"
      >
        <div>
          <span>Quality deep dive</span>
          <strong>{model.overallLabel}</strong>
          <p>{model.overallDetail}</p>
        </div>
        <span
          className="retrievalQualityBadge"
          data-quality={model.overallLevel}
        >
          {formatQualityLevel(model.overallLevel)}
        </span>
        <ChevronDown aria-hidden="true" size={15} />
      </button>
      {isOpen ? (
        <div className="retrievalQualityBody">
          <div
            aria-label="Retrieval quality metrics"
            className="retrievalQualityMetricGrid"
            role="list"
          >
            {model.metrics.map((metric) => (
              <QualityMetric metric={metric} key={metric.key} />
            ))}
          </div>
          <section
            aria-label="Retrieval domain balance"
            className="retrievalDomainBalance"
          >
            <header>
              <div>
                <span>Domain balance</span>
                <strong>{model.domainBalance.label}</strong>
                <p>{model.domainBalance.detail}</p>
              </div>
              <span data-balance={model.domainBalance.status}>
                {formatBalanceStatus(model.domainBalance.status)}
              </span>
            </header>
            {model.domainBalance.domains.length > 0 ? (
              <div
                aria-label="Retrieval domain distribution"
                className="retrievalDomainDistribution"
                role="list"
              >
                {model.domainBalance.domains.map((domain) => (
                  <DomainBalanceRow domain={domain} key={domain.domain} />
                ))}
              </div>
            ) : null}
          </section>
          <section
            aria-label="Retrieval weaknesses"
            className="retrievalQualityWeaknesses"
          >
            <span>Detected weaknesses</span>
            {model.weaknesses.length > 0 ? (
              <ul>
                {model.weaknesses.map((weakness) => (
                  <li key={weakness}>{weakness}</li>
                ))}
              </ul>
            ) : (
              <p>
                {model.hasEvidence
                  ? "No material weaknesses were detected in the recorded retrieval evidence."
                  : "Weaknesses can be assessed after retrieval evidence is available."}
              </p>
            )}
          </section>
        </div>
      ) : null}
    </section>
  );
}

function QualityMetric({ metric }: { metric: RetrievalQualityMetric }) {
  return (
    <article
      aria-label={`Retrieval ${metric.label.toLowerCase()}`}
      className="retrievalQualityMetric"
      data-quality={metric.level}
      role="listitem"
    >
      <header>
        <span>{metric.label}</span>
        <small data-quality={metric.level}>{formatQualityLevel(metric.level)}</small>
      </header>
      <strong>{metric.valueLabel}</strong>
      <p>{metric.detail}</p>
    </article>
  );
}

function DomainBalanceRow({ domain }: { domain: RetrievalDomainBalanceEntry }) {
  return (
    <div
      className="retrievalDomainRow"
      data-missing={domain.chunkCount === 0 ? "true" : "false"}
      role="listitem"
    >
      <span>
        <strong>{domain.label}</strong>
        <small>
          {domain.requested ? "Requested" : "Supporting"} ·{" "}
          {domain.chunkCount} {domain.chunkCount === 1 ? "chunk" : "chunks"}
        </small>
      </span>
      <span
        aria-label={`${domain.label} ${domain.shareLabel}`}
        className="retrievalDomainBar"
      >
        <span style={{ width: `${domain.sharePercent}%` }} />
      </span>
      <strong>{domain.shareLabel}</strong>
    </div>
  );
}

function formatQualityLevel(level: RetrievalQualityMetric["level"]) {
  switch (level) {
    case "high":
      return "High";
    case "medium":
      return "Medium";
    case "low":
      return "Low";
    default:
      return "Unknown";
  }
}

function formatBalanceStatus(status: RetrievalQualityModel["domainBalance"]["status"]) {
  switch (status) {
    case "balanced":
      return "Balanced";
    case "weighted":
      return "Weighted";
    case "focused":
      return "Focused";
    case "concentrated":
      return "Concentrated";
    default:
      return "Unknown";
  }
}
