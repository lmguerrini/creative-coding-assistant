import { ChevronDown } from "lucide-react";
import type { KbSourceHealthDashboardModel } from "@/lib/kb-source-health";

export function KbSourceHealthDashboard({
  model
}: {
  model: KbSourceHealthDashboardModel;
}) {
  return (
    <details className="kbHealthDashboard">
      <summary aria-label="Toggle knowledge base source health dashboard">
        <div>
          <span>Knowledge base health</span>
          <strong>{model.statusLabel}</strong>
          <p>{model.statusDetail}</p>
        </div>
        <span className="kbHealthStatusBadge" data-health={model.status}>
          {model.statusLabel}
        </span>
        <ChevronDown aria-hidden="true" size={15} />
      </summary>
      <div
        aria-label="Knowledge base source health metrics"
        className="kbHealthSummaryMetrics"
        role="list"
      >
        <HealthSummaryMetric
          label="Availability"
          value={model.availabilityLabel}
        />
        <HealthSummaryMetric
          label="Indexed coverage"
          value={model.indexedChunkLabel}
        />
        <HealthSummaryMetric
          label="Latest sync attempt"
          value={model.latestSyncAttemptLabel}
        />
        <HealthSummaryMetric
          label="Ownership"
          value={model.domainOwnerLabel}
        />
      </div>
    </details>
  );
}

function HealthSummaryMetric({
  label,
  value
}: {
  label: string;
  value: string;
}) {
  return (
    <span className="kbHealthSummaryMetric" role="listitem">
      <small>{label}</small>
      <strong>{value}</strong>
    </span>
  );
}
