import type {
  CreativeTimelineEvent,
  CreativeTimelineModel
} from "@/lib/creative-timeline";

type CreativeTimelineSurfaceProps = {
  timeline: CreativeTimelineModel;
};

export function CreativeTimelineSurface({ timeline }: CreativeTimelineSurfaceProps) {
  return (
    <article
      aria-label="Creative timeline"
      className="creativeTimelineSurface"
      data-state={timeline.state}
      role="group"
    >
      <header className="creativeTimelineHeader">
        <div>
          <span>Creative timeline</span>
          <strong>Request evolution</strong>
          <p>
            {timeline.state === "available"
              ? `${timeline.summary.completeCount} complete / ${timeline.summary.warningCount} warnings / ${timeline.summary.missingCount} missing`
              : "Timeline events appear after workstation metadata is available."}
          </p>
        </div>
      </header>

      <ol aria-label="Creative timeline events" className="creativeTimelineEvents">
        {timeline.events.map((event, index) => (
          <CreativeTimelineEventItem
            event={event}
            index={index}
            isLast={index === timeline.events.length - 1}
            key={event.id}
          />
        ))}
      </ol>
    </article>
  );
}

function CreativeTimelineEventItem({
  event,
  index,
  isLast
}: {
  event: CreativeTimelineEvent;
  index: number;
  isLast: boolean;
}) {
  return (
    <li
      aria-label={`${event.label} creative timeline event`}
      className="creativeTimelineEvent"
      data-status={event.status}
    >
      <div className="creativeTimelineRail" aria-hidden="true">
        <span>{String(index + 1).padStart(2, "0")}</span>
        {!isLast ? <i /> : null}
      </div>
      <article>
        <header>
          <div>
            <strong>{event.label}</strong>
            <small>{event.metadataAvailability}</small>
          </div>
          <span className="creativeTimelineBadge" data-status={event.status}>
            {formatStatus(event.status)}
          </span>
        </header>
        <p>{event.summary}</p>
        <div className="creativeTimelineMeta">
          <span>{`${event.eventCount} events`}</span>
          <span>{`${event.sourceCount} sources`}</span>
        </div>
        {event.warning ? (
          <p className="creativeTimelineWarning">{event.warning}</p>
        ) : null}
      </article>
    </li>
  );
}

function formatStatus(status: CreativeTimelineEvent["status"]) {
  switch (status) {
    case "complete":
      return "Complete";
    case "active":
      return "Active";
    case "warning":
      return "Warning";
    case "error":
      return "Error";
    default:
      return "Missing";
  }
}
