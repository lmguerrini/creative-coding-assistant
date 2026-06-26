import type {
  WorkstationDashboardCard,
  WorkstationDashboardModel
} from "@/lib/workstation-dashboard";

type WorkstationDashboardSurfaceProps = {
  dashboard: WorkstationDashboardModel;
};

export function WorkstationDashboardSurface({
  dashboard
}: WorkstationDashboardSurfaceProps) {
  return (
    <section
      aria-label="Workstation dashboard"
      className="workstationDashboard"
      data-state={dashboard.state}
      role="group"
    >
      <header className="workstationDashboardHeader">
        <div>
          <span>Dashboard</span>
          <strong>Workstation signals</strong>
          <p>
            {dashboard.summary.errorCount > 0
              ? `${dashboard.summary.goodCount} good / ${dashboard.summary.watchCount} watch / ${dashboard.summary.missingCount} missing / ${dashboard.summary.errorCount} error`
              : `${dashboard.summary.goodCount} good / ${dashboard.summary.watchCount} watch / ${dashboard.summary.missingCount} missing`}
          </p>
        </div>
      </header>
      <div
        aria-label="Workstation dashboard cards"
        className="workstationDashboardGrid"
        role="list"
      >
        {dashboard.cards.map((card) => (
          <WorkstationDashboardCardView card={card} key={card.id} />
        ))}
      </div>
    </section>
  );
}

function WorkstationDashboardCardView({
  card
}: {
  card: WorkstationDashboardCard;
}) {
  return (
    <article
      aria-label={`${card.label} dashboard card`}
      className="workstationDashboardCard"
      data-tone={card.tone}
      role="listitem"
    >
      <header>
        <span>{card.label}</span>
        <strong>{card.value}</strong>
      </header>
      <p>{card.summary}</p>
      <small>{`${card.detail} / ${card.source}`}</small>
    </article>
  );
}
