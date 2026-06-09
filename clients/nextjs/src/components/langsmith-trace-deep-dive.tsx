import type {
  LangSmithTraceMetadataGroup,
  LangSmithTraceModel,
  LangSmithTraceSpan
} from "@/lib/langsmith-trace";
import type { CSSProperties } from "react";

type LangSmithTraceDeepDiveProps = {
  trace: LangSmithTraceModel;
};

export function LangSmithTraceDeepDive({
  trace
}: LangSmithTraceDeepDiveProps) {
  const populatedMetadata = trace.metadataGroups.filter(
    (group) => group.entries.length > 0
  );

  return (
    <article
      aria-label="LangSmith trace deep dive"
      className="langsmithTraceCard telemetryDashboardCard telemetryDashboardCard--wide"
      data-state={trace.state}
      role="group"
    >
      <header className="langsmithTraceHeader">
        <div>
          <span>LangSmith trace</span>
          <strong>{trace.availabilityLabel}</strong>
          <p>
            {trace.traceKind
              ? `${formatLabel(trace.traceKind)} execution trace`
              : "Execution trace metadata appears here when observability is requested."}
          </p>
        </div>
        <span
          aria-label={`Trace status ${trace.statusLabel}`}
          className="langsmithTraceStatus"
          data-status={trace.status}
        >
          {trace.statusLabel}
        </span>
      </header>

      {trace.state === "unavailable" ? (
        <section
          aria-label="LangSmith trace empty state"
          className="langsmithTraceEmpty"
        >
          <strong>No LangSmith trace for this session</strong>
          <p>
            Legacy telemetry remains available. Trace identity, hierarchy, and
            metadata will appear when observability records are present.
          </p>
        </section>
      ) : (
        <>
          <section
            aria-label="LangSmith trace overview"
            className="langsmithTraceOverview"
          >
            <TraceMetric label="Trace ID" value={trace.traceId ?? "Not recorded"} />
            <TraceMetric label="Run ID" value={trace.runId ?? "Not recorded"} />
            <TraceMetric
              label="Started"
              value={formatTimestamp(trace.startedAt)}
            />
            <TraceMetric label="Ended" value={formatTimestamp(trace.endedAt)} />
            <TraceMetric
              label="Duration"
              value={formatDuration(trace.durationMs)}
            />
          </section>

          <div className="langsmithTraceIdentity">
            <span>{trace.providerLabel}</span>
            <span>{trace.projectName ?? "Project not recorded"}</span>
            <span>{trace.runName ?? "Run name not recorded"}</span>
            {trace.parentRunId ? <span>Parent {trace.parentRunId}</span> : null}
          </div>

          <section
            aria-label="LangSmith trace hierarchy"
            className="langsmithTraceHierarchy"
          >
            <header>
              <div>
                <span>Trace hierarchy</span>
                <strong>
                  {trace.summary.spanCount === 1
                    ? "1 recorded span"
                    : `${trace.summary.spanCount} recorded spans`}
                </strong>
              </div>
              <small>
                {trace.summary.transitionCount} transitions /{" "}
                {trace.summary.nestedSpanCount} nested
              </small>
            </header>

            {trace.spans.length > 0 ? (
              <ol>
                {trace.spans.map((span) => (
                  <TraceSpanRow key={span.id} span={span} />
                ))}
              </ol>
            ) : (
              <div className="langsmithTraceHierarchyEmpty">
                <strong>No span hierarchy reported</strong>
                <p>
                  The trace is available, but this session did not expose
                  workflow steps or nested spans.
                </p>
              </div>
            )}
          </section>

          <section
            aria-label="LangSmith trace metadata"
            className="langsmithTraceMetadata"
          >
            <header>
              <div>
                <span>Trace metadata</span>
                <strong>{trace.summary.metadataCount} recorded fields</strong>
              </div>
              <small>
                {trace.tags.length > 0
                  ? `${trace.tags.length} execution tags`
                  : "No execution tags"}
              </small>
            </header>

            {trace.tags.length > 0 ? (
              <div aria-label="LangSmith execution tags" className="langsmithTraceTags">
                {trace.tags.map((tag) => (
                  <span key={tag}>{tag}</span>
                ))}
              </div>
            ) : null}

            {populatedMetadata.length > 0 ? (
              <div className="langsmithTraceMetadataGrid">
                {populatedMetadata.map((group) => (
                  <MetadataGroup group={group} key={group.id} />
                ))}
              </div>
            ) : (
              <div className="langsmithTraceMetadataEmpty">
                No provider, retrieval, evaluation, or execution metadata was
                reported.
              </div>
            )}
          </section>
        </>
      )}
    </article>
  );
}

function TraceMetric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function TraceSpanRow({ span }: { span: LangSmithTraceSpan }) {
  return (
    <li
      className="langsmithTraceSpan"
      data-depth={Math.min(span.depth, 4)}
      data-status={span.status}
      style={{ "--trace-depth": Math.min(span.depth, 4) } as CSSProperties}
    >
      <div className="langsmithTraceRail" aria-hidden="true">
        <span />
      </div>
      <div className="langsmithTraceSpanBody">
        <header>
          <div>
            <strong>{span.name}</strong>
            <span>{formatLabel(span.runType)}</span>
          </div>
          <small>{formatDuration(span.durationMs)}</small>
        </header>
        <div className="langsmithTraceSpanMeta">
          {span.stage ? <span>{formatLabel(span.stage)}</span> : null}
          <span>{formatTraceStatus(span.status)}</span>
          <span>{formatTimestamp(span.startedAt)}</span>
          {span.runId ? <span>Run {span.runId}</span> : null}
        </div>
        {span.transitionFrom ? (
          <p>
            <strong>{formatLabel(span.transitionFrom)}</strong>
            {" -> "}
            {span.stage ? formatLabel(span.stage) : span.name}
            {span.transitionReason ? ` / ${span.transitionReason}` : ""}
          </p>
        ) : null}
      </div>
    </li>
  );
}

function MetadataGroup({ group }: { group: LangSmithTraceMetadataGroup }) {
  return (
    <section
      aria-label={group.label}
      className="langsmithTraceMetadataGroup"
    >
      <header>
        <strong>{group.label}</strong>
        <span>{group.entries.length}</span>
      </header>
      <dl>
        {group.entries.map((entry) => (
          <div key={entry.key}>
            <dt>{entry.label}</dt>
            <dd>{entry.value}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function formatTraceStatus(status: LangSmithTraceSpan["status"]) {
  return formatLabel(status);
}

function formatDuration(durationMs: number | null) {
  if (durationMs === null) {
    return "Not recorded";
  }
  if (durationMs < 1000) {
    return `${Math.round(durationMs)} ms`;
  }
  return `${(durationMs / 1000).toFixed(durationMs >= 10000 ? 0 : 1)} s`;
}

function formatTimestamp(timestamp: string | null) {
  if (!timestamp) {
    return "Not recorded";
  }
  const parsed = Date.parse(timestamp);
  if (!Number.isFinite(parsed)) {
    return timestamp;
  }
  return new Intl.DateTimeFormat("en-US", {
    hour: "2-digit",
    hour12: false,
    minute: "2-digit",
    second: "2-digit",
    timeZone: "UTC"
  }).format(parsed);
}

function formatLabel(value: string) {
  return value
    .replace(/[._-]+/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}
