import type {
  WorkflowTimelineEvent,
  WorkflowTimelineModel,
  WorkflowTimelineStatus
} from "@/lib/workflow-timeline";

type WorkflowTimelineExplorerProps = {
  timeline: WorkflowTimelineModel;
};

export function WorkflowTimelineExplorer({
  timeline
}: WorkflowTimelineExplorerProps) {
  return (
    <article
      aria-label="Workflow timeline explorer"
      className="workflowTimelineExplorer"
      data-state={timeline.state}
      role="group"
    >
      <header className="workflowTimelineHeader">
        <div>
          <span>Execution path</span>
          <strong>Workflow timeline</strong>
          <p>
            {timeline.state === "available"
              ? `${timeline.summary.eventCount} chronological workflow events`
              : "The chronological run path will appear after workflow events arrive."}
          </p>
        </div>
        <span className="workflowTimelineState" data-state={timeline.state}>
          {timeline.state === "available" ? "Recorded" : "Awaiting run"}
        </span>
      </header>

      <div
        aria-label="Workflow timeline summary"
        className="workflowTimelineSummary"
        role="group"
      >
        <SummaryMetric
          label="Started"
          value={formatTimelineTime(timeline.summary.startedAt)}
        />
        <SummaryMetric
          label="Elapsed"
          value={formatDuration(timeline.summary.totalDurationMs)}
        />
        <SummaryMetric
          label="Warnings"
          value={String(timeline.summary.warningCount)}
        />
        <SummaryMetric
          label="Errors"
          value={String(timeline.summary.errorCount)}
        />
      </div>

      {timeline.events.length > 0 ? (
        <ol aria-label="Chronological workflow events" className="workflowTimelineList">
          {timeline.events.map((event, index) => (
            <TimelineEvent
              event={event}
              index={index}
              isLast={index === timeline.events.length - 1}
              key={event.id}
            />
          ))}
        </ol>
      ) : (
        <div aria-label="Workflow timeline empty state" className="workflowTimelineEmpty">
          <strong>No workflow timeline yet</strong>
          <p>
            Start a generation run to inspect routing, retrieval, generation,
            artifacts, review, and finalization in order.
          </p>
        </div>
      )}
    </article>
  );
}

function TimelineEvent({
  event,
  index,
  isLast
}: {
  event: WorkflowTimelineEvent;
  index: number;
  isLast: boolean;
}) {
  return (
    <li
      aria-label={`${event.label} timeline event`}
      className="workflowTimelineEvent"
      data-status={event.status}
    >
      <div className="workflowTimelineRail" aria-hidden="true">
        <span>{String(index + 1).padStart(2, "0")}</span>
        {!isLast ? <i /> : null}
      </div>
      <article>
        <header>
          <div>
            <strong>{event.label}</strong>
            <small>{event.detail}</small>
          </div>
          <StatusBadge status={event.status} />
        </header>
        <div className="workflowTimelineMeta">
          <span>{formatTimelineTime(event.at)}</span>
          <span>{event.stageLabel}</span>
          <span>{formatCode(event.phase)}</span>
          {event.durationMs != null ? (
            <span>{formatDuration(event.durationMs)}</span>
          ) : null}
        </div>
        {event.transitionReason ? (
          <p className="workflowTimelineReason">
            <span>Transition</span>
            {formatCode(event.transitionReason)}
          </p>
        ) : null}
        {event.warning ? (
          <p className="workflowTimelineIssue" data-kind="warning">
            <span>Warning</span>
            {event.warning}
          </p>
        ) : null}
        {event.error ? (
          <p className="workflowTimelineIssue" data-kind="error">
            <span>Error</span>
            {event.error}
          </p>
        ) : null}
      </article>
    </li>
  );
}

function StatusBadge({ status }: { status: WorkflowTimelineStatus }) {
  return (
    <span className="workflowTimelineBadge" data-status={status}>
      {formatStatus(status)}
    </span>
  );
}

function SummaryMetric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function formatStatus(status: WorkflowTimelineStatus) {
  switch (status) {
    case "complete":
      return "Complete";
    case "running":
      return "Running";
    case "skipped":
      return "Skipped";
    case "warning":
      return "Warning";
    case "error":
      return "Error";
    default:
      return "Info";
  }
}

function formatDuration(durationMs: number | null) {
  if (durationMs == null) {
    return "Pending";
  }
  if (durationMs < 1000) {
    return `${durationMs} ms`;
  }
  return `${(durationMs / 1000).toFixed(durationMs >= 10000 ? 0 : 1)} s`;
}

function formatTimelineTime(timestamp: string | null) {
  if (!timestamp) {
    return "Not recorded";
  }
  const parsed = new Date(timestamp);
  if (Number.isNaN(parsed.getTime())) {
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

function formatCode(value: string) {
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}
