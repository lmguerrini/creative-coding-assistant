import type {
  EvaluationOutcome,
  EvaluationSessionModel,
  EvaluationSessionSignal
} from "@/lib/evaluation-session";

type EvaluationSessionDashboardProps = {
  evaluation: EvaluationSessionModel;
};

export function EvaluationSessionDashboard({
  evaluation
}: EvaluationSessionDashboardProps) {
  const hasEvaluation = evaluation.state === "available";

  return (
    <article
      aria-label="Evaluation session dashboard"
      className="evaluationSessionDashboard telemetryDashboardCard telemetryDashboardCard--wide"
      data-outcome={evaluation.outcome}
      data-state={evaluation.state}
      role="group"
    >
      <header className="evaluationSessionHeader">
        <div>
          <span>Session evaluation</span>
          <strong>{evaluation.statusLabel}</strong>
          <p>{evaluation.detail}</p>
        </div>
        <OutcomeBadge outcome={evaluation.outcome} />
      </header>

      <div className="evaluationSessionOverview">
        <section
          aria-label="Latest evaluation score"
          className="evaluationSessionScore"
          data-outcome={evaluation.outcome}
        >
          <span>Latest score</span>
          <strong>{formatScore(evaluation.score)}</strong>
          <small>
            {evaluation.score == null ? "Score unavailable" : "Session composite"}
          </small>
        </section>

        <dl className="evaluationSessionMeta">
          <Metadata label="Type" value={evaluation.evaluationType} />
          <Metadata
            label="Evaluated"
            value={formatEvaluationTimestamp(evaluation.latestAt)}
          />
          <Metadata label="Run" value={evaluation.runId ?? "Not recorded"} />
          <Metadata label="Dataset" value={evaluation.datasetId ?? "Not recorded"} />
        </dl>
      </div>

      {hasEvaluation ? (
        <section
          aria-label="Evaluation quality signals"
          className="evaluationSignalsSection"
        >
          <header>
            <div>
              <span>Quality signals</span>
              <strong>Current session evidence</strong>
            </div>
            <small>{summarizeSignalCoverage(evaluation.signals)}</small>
          </header>
          <div className="evaluationSignalGrid">
            {evaluation.signals.map((signal) => (
              <SignalCard key={signal.id} signal={signal} />
            ))}
          </div>
        </section>
      ) : (
        <section
          aria-label="Evaluation empty state"
          className="evaluationEmptyState"
        >
          <strong>No session evaluation results yet</strong>
          <p>
            Answer, retrieval, grounding, artifact, and provider/runtime signals
            will appear after an evaluation event is recorded.
          </p>
        </section>
      )}

      {hasEvaluation ? (
        <section
          aria-label="RAGAs evaluation lineage"
          className="evaluationLineage"
        >
          <header>
            <span>Evaluation lineage</span>
            <small>{formatLineageMode(evaluation)}</small>
          </header>
          <dl>
            <Metadata
              label="Rows"
              value={formatNullableNumber(evaluation.resultRows)}
            />
            <Metadata
              label="Failures"
              value={formatNullableNumber(evaluation.metricFailures)}
            />
            <Metadata
              label="Provider calls"
              value={formatOptionalBoolean(evaluation.providerCallsAllowed)}
            />
            <Metadata
              label="Metrics"
              value={
                evaluation.metrics.length > 0
                  ? evaluation.metrics.map(formatMetricLabel).join(" / ")
                  : "Not recorded"
              }
            />
          </dl>
        </section>
      ) : null}
    </article>
  );
}

function SignalCard({ signal }: { signal: EvaluationSessionSignal }) {
  return (
    <article
      aria-label={`${signal.label} evaluation signal`}
      className="evaluationSignalCard"
      data-outcome={signal.outcome}
      role="group"
    >
      <header>
        <span>{signal.label}</span>
        <OutcomeBadge compact outcome={signal.outcome} />
      </header>
      <strong>{formatScore(signal.score)}</strong>
      <p>{signal.detail}</p>
    </article>
  );
}

function Metadata({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function OutcomeBadge({
  compact = false,
  outcome
}: {
  compact?: boolean;
  outcome: EvaluationOutcome;
}) {
  return (
    <span
      aria-label={
        compact
          ? `${formatOutcome(outcome)} signal status`
          : `Evaluation status ${formatOutcome(outcome)}`
      }
      className="evaluationOutcomeBadge"
      data-compact={compact ? "true" : undefined}
      data-outcome={outcome}
    >
      {formatOutcome(outcome)}
    </span>
  );
}

function summarizeSignalCoverage(signals: EvaluationSessionSignal[]) {
  const available = signals.filter((signal) => signal.score != null).length;
  return `${available}/${signals.length} scored`;
}

function formatEvaluationTimestamp(timestamp: string | null) {
  if (!timestamp) {
    return "Not recorded";
  }
  const parsed = new Date(timestamp);
  if (Number.isNaN(parsed.getTime())) {
    return timestamp;
  }
  return new Intl.DateTimeFormat("en-US", {
    day: "numeric",
    hour: "2-digit",
    hour12: false,
    minute: "2-digit",
    month: "short",
    timeZone: "UTC",
    timeZoneName: "short",
    year: "numeric"
  }).format(parsed);
}

function formatLineageMode(evaluation: EvaluationSessionModel) {
  if (evaluation.dryRun === true) {
    return "Dry run";
  }
  if (evaluation.dryRun === false) {
    return "Live evaluation";
  }
  return "Mode not recorded";
}

function formatScore(score: number | null) {
  return score == null ? "—" : `${Math.round(score * 100)}%`;
}

function formatOutcome(outcome: EvaluationOutcome) {
  switch (outcome) {
    case "pass":
      return "Pass";
    case "warn":
      return "Warn";
    case "fail":
      return "Fail";
    default:
      return "Unscored";
  }
}

function formatOptionalBoolean(value: boolean | null) {
  if (value == null) {
    return "Not recorded";
  }
  return value ? "Allowed" : "Disabled";
}

function formatNullableNumber(value: number | null) {
  return value == null ? "Not recorded" : value.toLocaleString("en-US");
}

function formatMetricLabel(value: string) {
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}
