import type {
  ProviderTelemetryFallbackPath,
  ProviderTelemetryIssue,
  ProviderTelemetryModel,
  ProviderTelemetryRetryEvent
} from "@/lib/provider-telemetry";
import { Fragment, type ReactNode } from "react";

type ProviderObservabilityDeepDiveProps = {
  telemetry: ProviderTelemetryModel;
};

export function ProviderObservabilityDeepDive({
  telemetry
}: ProviderObservabilityDeepDiveProps) {
  const execution = telemetry.execution;

  return (
    <article
      aria-label="Provider observability deep dive"
      className="providerObservabilityCard telemetryDashboardCard telemetryDashboardCard--wide"
      data-state={telemetry.status}
      role="group"
    >
      <header className="providerObservabilityHeader">
        <div>
          <span>Provider execution</span>
          <strong>{formatProviderLabel(telemetry)}</strong>
          <p>{executionSummary(telemetry)}</p>
        </div>
        <span className="telemetryStateBadge" data-state={telemetry.status}>
          {formatProviderStatus(telemetry.status)}
        </span>
      </header>

      <div
        aria-label="Provider execution summary"
        className="providerObservabilityMetrics"
      >
        <Metric label="Generation mode" value={telemetry.summary.generationModeLabel} />
        <Metric label="Request duration" value={telemetry.summary.requestDurationLabel} />
        <Metric label="Retries" value={formatRetryCount(execution.retryCount)} />
        <Metric label="Streaming" value={telemetry.summary.streamingStatusLabel} />
      </div>

      <div className="providerObservabilityDetailGrid">
        <section
          aria-label="Provider token transparency"
          className="providerObservabilitySection"
        >
          <header>
            <span>Token transparency</span>
            <strong>{telemetry.summary.tokenLabel}</strong>
          </header>
          <dl>
            <MetricDefinition
              label="Input"
              value={formatTokenCount(telemetry.tokenUsage.inputTokens)}
            />
            <MetricDefinition
              label="Output"
              value={formatTokenCount(telemetry.tokenUsage.outputTokens)}
            />
            <MetricDefinition
              label="Total"
              value={formatTokenCount(telemetry.tokenUsage.totalTokens)}
            />
            <MetricDefinition label="Estimated cost" value={telemetry.summary.costLabel} />
          </dl>
          <small>{formatCostSource(telemetry)}</small>
        </section>

        <section
          aria-label="Provider request identity"
          className="providerObservabilitySection"
        >
          <header>
            <span>Request identity</span>
            <strong>{telemetry.provider.runtime ?? "Provider runtime"}</strong>
          </header>
          <dl>
            <MetricDefinition
              label="Request ID"
              value={telemetry.provider.requestId ?? "Unavailable"}
            />
            <MetricDefinition
              label="Response ID"
              value={telemetry.provider.responseId ?? "Unavailable"}
            />
            <MetricDefinition
              label="Finish reason"
              value={formatCode(execution.finishReason) ?? "Unavailable"}
            />
            <MetricDefinition
              label="First token"
              value={formatDuration(telemetry.timing.timeToFirstTokenMs)}
            />
          </dl>
          <small>
            {execution.requestStartedAt
              ? `Started ${formatTimestamp(execution.requestStartedAt)}`
              : "Request timing metadata unavailable"}
          </small>
        </section>

        <section
          aria-label="Agent configuration"
          className="providerObservabilitySection"
        >
          <header>
            <span>Agent configuration</span>
            <strong>{telemetry.configuration.route ?? "Route unavailable"}</strong>
          </header>
          <dl>
            <MetricDefinition label="Model" value={telemetry.summary.modelLabel} />
            <MetricDefinition
              label="Messages"
              value={formatCount(telemetry.configuration.messageCount)}
            />
            <MetricDefinition
              label="Temperature"
              value={formatParameter(telemetry.configuration.temperature)}
            />
            <MetricDefinition
              label="Top P"
              value={formatParameter(telemetry.configuration.topP)}
            />
            <MetricDefinition
              label="Max output tokens"
              value={formatCount(telemetry.configuration.maxOutputTokens)}
            />
          </dl>
          <small>{formatParameterSource(telemetry)}</small>
        </section>
      </div>

      <section
        aria-label="Provider execution details"
        className="providerObservabilitySection providerObservabilitySection--wide"
      >
        <header>
          <span>Execution details</span>
          <strong>{telemetry.summary.issueLabel}</strong>
        </header>
        <div className="providerExecutionColumns">
          <ExecutionList
            emptyLabel="No provider errors reported."
            items={execution.errors}
            label="Errors"
            renderItem={(issue) => <IssueRow issue={issue} />}
          />
          <ExecutionList
            emptyLabel="No provider warnings reported."
            items={execution.warnings}
            label="Warnings"
            renderItem={(issue) => <IssueRow issue={issue} />}
          />
          <ExecutionList
            emptyLabel={
              execution.retryCount === null
                ? "Provider retry metadata unavailable."
                : "No provider retries reported."
            }
            items={execution.retryEvents}
            label="Retry events"
            renderItem={(retry) => <RetryRow retry={retry} />}
          />
          <ExecutionList
            emptyLabel="No provider fallback path observed."
            items={execution.fallbackPaths}
            label="Fallback paths"
            renderItem={(path) => <FallbackRow path={path} />}
          />
        </div>
      </section>
    </article>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function MetricDefinition({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function ExecutionList<T>({
  emptyLabel,
  items,
  label,
  renderItem
}: {
  emptyLabel: string;
  items: T[];
  label: string;
  renderItem: (item: T) => ReactNode;
}) {
  return (
    <div className="providerExecutionList" role="group" aria-label={label}>
      <header>
        <strong>{label}</strong>
        <span>{items.length}</span>
      </header>
      {items.length > 0 ? (
        <div>
          {items.map((item, index) => (
            <Fragment key={index}>{renderItem(item)}</Fragment>
          ))}
        </div>
      ) : (
        <p>{emptyLabel}</p>
      )}
    </div>
  );
}

function IssueRow({ issue }: { issue: ProviderTelemetryIssue }) {
  return (
    <div
      className="providerExecutionItem"
      data-severity={issue.severity}
      key={issue.id}
    >
      <strong>{formatCode(issue.code) ?? issue.code}</strong>
      <p>{issue.message}</p>
      <small>
        {issue.recoverable === true
          ? "Recoverable"
          : issue.recoverable === false
            ? "Terminal"
            : "Recovery state unavailable"}
        {issue.at ? ` / ${formatTimestamp(issue.at)}` : ""}
      </small>
    </div>
  );
}

function RetryRow({ retry }: { retry: ProviderTelemetryRetryEvent }) {
  return (
    <div className="providerExecutionItem" key={retry.id}>
      <strong>
        {retry.attempt != null ? `Attempt ${retry.attempt}` : "Provider retry"} /{" "}
        {formatCode(retry.status)}
      </strong>
      <p>{retry.reason ?? "No retry reason reported."}</p>
      <small>{retry.at ? formatTimestamp(retry.at) : "Time unavailable"}</small>
    </div>
  );
}

function FallbackRow({ path }: { path: ProviderTelemetryFallbackPath }) {
  const route =
    path.source || path.target
      ? `${formatCode(path.source) ?? "provider"} -> ${
          formatCode(path.target) ?? "fallback"
        }`
      : "Fallback route unavailable";

  return (
    <div className="providerExecutionItem" key={path.id}>
      <strong>{path.label}</strong>
      <p>{path.reason}</p>
      <small>
        {route}
        {path.at ? ` / ${formatTimestamp(path.at)}` : ""}
      </small>
    </div>
  );
}

function formatProviderLabel(telemetry: ProviderTelemetryModel) {
  return `${telemetry.summary.providerLabel} / ${telemetry.summary.modelLabel}`;
}

function executionSummary(telemetry: ProviderTelemetryModel) {
  return `${telemetry.summary.generationModeLabel} / ${telemetry.summary.streamingStatusLabel}`;
}

function formatProviderStatus(status: ProviderTelemetryModel["status"]) {
  switch (status) {
    case "complete":
      return "Complete";
    case "error":
      return "Error";
    case "streaming":
      return "Streaming";
    default:
      return "Idle";
  }
}

function formatRetryCount(value: number | null) {
  if (value === null) {
    return "Unavailable";
  }
  return value === 1 ? "1 retry" : `${value} retries`;
}

function formatTokenCount(value: number | null) {
  return value == null
    ? "Unavailable"
    : new Intl.NumberFormat("en", { maximumFractionDigits: 0 }).format(value);
}

function formatDuration(value: number | null) {
  if (value == null) {
    return "Unavailable";
  }
  return value < 1000 ? `${Math.round(value)}ms` : `${(value / 1000).toFixed(1)}s`;
}

function formatCount(value: number | null) {
  return value == null ? "Unavailable" : value.toLocaleString();
}

function formatParameter(value: number | null) {
  return value == null ? "Not published" : String(value);
}

function formatParameterSource(telemetry: ProviderTelemetryModel) {
  switch (telemetry.configuration.parameterSource) {
    case "provider_reported":
      return "Sampling parameters are provider-reported for this request.";
    case "request_record":
      return "The request record is available; the provider did not publish sampling parameters.";
    default:
      return "Parameter provenance was not published for this request.";
  }
}

function formatCostSource(telemetry: ProviderTelemetryModel) {
  switch (telemetry.cost.source) {
    case "provider_reported":
      return "Provider-reported cost";
    case "pricing_metadata":
      return "Estimated from provider pricing metadata";
    default:
      return "Cost unavailable until usage or pricing metadata arrives";
  }
}

function formatCode(value: string | null) {
  if (!value) {
    return null;
  }
  return value
    .replaceAll("_", " ")
    .replaceAll("-", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function formatTimestamp(value: string) {
  const parsed = Date.parse(value);
  if (!Number.isFinite(parsed)) {
    return value;
  }
  return new Intl.DateTimeFormat("en", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  }).format(parsed);
}
