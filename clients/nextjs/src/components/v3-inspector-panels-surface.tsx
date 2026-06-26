import type {
  V3InspectorPanel,
  V3InspectorPanelItem,
  V3InspectorPanelsModel
} from "@/lib/v3-inspector-panels";

type V3InspectorPanelsSurfaceProps = {
  model: V3InspectorPanelsModel;
};

const maxVisibleItems = 5;

export function V3InspectorPanelsSurface({
  model
}: V3InspectorPanelsSurfaceProps) {
  return (
    <article
      aria-label="V3 inspector panels"
      className="v3InspectorSurface"
      data-state={model.state}
      role="group"
    >
      <header className="v3InspectorHeader">
        <div>
          <span>V3 inspector panels</span>
          <strong>Metadata groups</strong>
          <p>
            {`${model.summary.availableItemCount} available / ${model.summary.partialItemCount} partial / ${model.summary.missingItemCount} missing`}
          </p>
        </div>
        <span className="v3InspectorRunState">{formatRunState(model)}</span>
      </header>

      <div
        aria-label="V3 metadata inspector panels"
        className="v3InspectorGrid"
        role="list"
      >
        {model.panels.map((panel) => (
          <V3InspectorPanelCard key={panel.id} panel={panel} />
        ))}
      </div>
    </article>
  );
}

function V3InspectorPanelCard({ panel }: { panel: V3InspectorPanel }) {
  const visibleItems = panel.items.slice(0, maxVisibleItems);
  const hiddenCount = Math.max(0, panel.items.length - visibleItems.length);

  return (
    <section
      aria-label={`${panel.label} inspector panel`}
      className="v3InspectorCard"
      data-state={panel.status}
      role="group"
    >
      <header>
        <div>
          <span>{panel.label}</span>
          <strong>{formatStatus(panel.status)}</strong>
        </div>
        <span className="v3InspectorBadge" data-state={panel.status}>
          {`${panel.availableItemCount}/${panel.items.length}`}
        </span>
      </header>
      <p>{panel.summary}</p>
      <div className="v3InspectorMetrics">
        <span>{`${panel.partialItemCount} partial`}</span>
        <span>{`${panel.missingItemCount} missing`}</span>
      </div>
      <div
        aria-label={`${panel.label} metadata records`}
        className="v3InspectorRecords"
        role="list"
      >
        {visibleItems.map((item) => (
          <V3InspectorRecord item={item} key={item.id} />
        ))}
      </div>
      {hiddenCount > 0 ? (
        <small>{`${hiddenCount} additional metadata record${
          hiddenCount === 1 ? "" : "s"
        } bounded from view.`}</small>
      ) : null}
    </section>
  );
}

function V3InspectorRecord({ item }: { item: V3InspectorPanelItem }) {
  return (
    <div
      aria-label={`${item.label} metadata record`}
      className="v3InspectorRecord"
      data-state={item.status}
      role="listitem"
    >
      <header>
        <strong>{item.label}</strong>
        <span>{formatStatus(item.status)}</span>
      </header>
      <p>{item.summary}</p>
      {item.details.length > 0 ? (
        <ul>
          {item.details.map((detail) => (
            <li key={detail}>{detail}</li>
          ))}
        </ul>
      ) : null}
      <small>
        {item.eventSequence != null
          ? `Event ${item.eventSequence} / ${formatSource(item.source)}`
          : formatSource(item.source)}
      </small>
    </div>
  );
}

function formatRunState(model: V3InspectorPanelsModel) {
  switch (model.summary.currentRunState) {
    case "streaming":
      return "Streaming";
    case "completed":
      return "Completed";
    case "error":
      return "Error";
    default:
      return "Idle";
  }
}

function formatStatus(status: V3InspectorPanelItem["status"]) {
  switch (status) {
    case "available":
      return "Available";
    case "partial":
      return "Partial";
    default:
      return "Missing";
  }
}

function formatSource(source: V3InspectorPanelItem["source"]) {
  switch (source) {
    case "hydrated":
      return "Hydrated";
    case "partial":
      return "Partial payload";
    case "provenance":
      return "Provenance";
    default:
      return "Missing";
  }
}
