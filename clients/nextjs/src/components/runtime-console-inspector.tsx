import type {
  RuntimeConsoleEvent,
  RuntimeConsoleModel
} from "@/lib/runtime-console";
import type { WorkflowRuntimeModel } from "@/lib/workflow-runtime";
import { DashboardDisclosure } from "./dashboard-page-primitives";

export function RuntimeConsoleInspector({
  console,
  productOutcome,
  presentation = "inspector"
}: {
  console: RuntimeConsoleModel;
  productOutcome?: WorkflowRuntimeModel["summary"]["productOutcome"];
  presentation?: "dashboard" | "inspector";
}) {
  const statusMetric = console.metrics.find((metric) => metric.id === "status");

  return (
    <section
      aria-label="Runtime console inspector"
      className={
        presentation === "inspector"
          ? "inspectorPanel runtimeConsolePanel"
          : "runtimeConsoleDashboard"
      }
      data-state={console.hero.tone}
      id="runtime-inspector-panel"
      role="tabpanel"
    >
      <article
        aria-label="Runtime console status"
        className="runtimeConsoleHero"
        data-tone={console.hero.tone}
        role="group"
      >
        <div>
          <span>{console.hero.eyebrow}</span>
          <strong>{console.hero.title}</strong>
          <p>{console.hero.detail}</p>
        </div>
        <small>{console.hero.sessionLabel}</small>
      </article>

      {productOutcome ? <ProductOutcomeRuntimeCard productOutcome={productOutcome} /> : null}

      {console.emptyTitle ? (
        <article
          aria-label="Runtime console empty state"
          className="runtimeConsoleEmptyCard"
          role="group"
        >
          <strong>{console.emptyTitle}</strong>
          <p>{console.emptyDetail}</p>
        </article>
      ) : (
        <>
          <div className="runtimeConsoleGrid">
          <article
            aria-label="Runtime health"
            className="runtimeConsoleCard runtimeConsoleCard--wide runtimeConsoleHealth"
            data-tone={console.health.tone}
            role="group"
          >
            <header>
              <span>Renderer health</span>
              <strong>{console.health.label}</strong>
            </header>
            <p>{console.health.explanation}</p>
            <div className="runtimeConsoleHealthSignals" role="list">
              <div role="listitem">
                <span>Health signal</span>
                <strong>{console.health.label}</strong>
              </div>
              <div role="listitem">
                <span>Execution state</span>
                <strong>{statusMetric?.value ?? "Idle"}</strong>
              </div>
            </div>
          </article>

          <article
            aria-label="Runtime metrics"
            className="runtimeConsoleCard runtimeConsoleCard--wide"
            role="group"
          >
            <header>
              <span>Metrics</span>
              <strong>{statusMetric?.value ?? "Idle"}</strong>
            </header>
            <div className="runtimeConsoleMetricGrid" role="list">
              {console.metrics.map((metric) => (
                <div
                  data-tone={metric.tone}
                  key={metric.id}
                  role="listitem"
                >
                  <span>{metric.label}</span>
                  <strong>{metric.value}</strong>
                </div>
              ))}
            </div>
          </article>

          {presentation === "inspector" ? (
            <>
              <RuntimeDiagnosticsCard console={console} />
              <RuntimeContextCard console={console} />
              <RuntimeReloadHistoryCard console={console} />
              <RuntimeEventHistoryCard console={console} />
            </>
          ) : null}
          </div>
          {presentation === "dashboard" ? (
            <div className="runtimeConsoleDashboardDetails">
              <DashboardDisclosure summary={`Diagnostics and context · ${formatDiagnosticCount(console)}`}>
                <div className="runtimeConsoleGrid runtimeConsoleGrid--secondary">
                  <RuntimeDiagnosticsCard console={console} />
                  <RuntimeContextCard console={console} />
                </div>
              </DashboardDisclosure>
              <DashboardDisclosure summary={`Runtime history · ${formatReloadCount(console.reloadHistory.length)} · ${console.events.length} events`}>
                <div className="runtimeConsoleGrid runtimeConsoleGrid--secondary">
                  <RuntimeReloadHistoryCard console={console} />
                  <RuntimeEventHistoryCard console={console} />
                </div>
              </DashboardDisclosure>
            </div>
          ) : null}
        </>
      )}
    </section>
  );
}

function ProductOutcomeRuntimeCard({
  productOutcome
}: {
  productOutcome: WorkflowRuntimeModel["summary"]["productOutcome"];
}) {
  return (
    <article
      aria-label="Semantic runtime outcome"
      className="runtimeConsoleCard runtimeConsoleCard--wide"
      data-tone={
        productOutcome.product_outcome === "FAILURE"
          ? "danger"
          : productOutcome.product_outcome === "PARTIAL"
            ? "warning"
            : "success"
      }
      role="group"
    >
      <header>
        <span>Product outcome</span>
        <strong>{productOutcome.product_outcome}</strong>
      </header>
      <p>{productOutcome.summary}</p>
      <div className="runtimeConsoleHealthSignals" role="list">
        <div role="listitem">
          <span>Preview</span>
          <strong>{productOutcome.preview_status}</strong>
        </div>
        <div role="listitem">
          <span>Runtime health</span>
          <strong>{productOutcome.runtime_health}</strong>
        </div>
        <div role="listitem">
          <span>Recovery</span>
          <strong>{productOutcome.recovery_action || "None required"}</strong>
        </div>
      </div>
    </article>
  );
}

function RuntimeDiagnosticsCard({ console }: { console: RuntimeConsoleModel }) {
  return (
    <article aria-label="Runtime diagnostics" className="runtimeConsoleCard" role="group">
      <header><span>Diagnostics</span><strong>{formatDiagnosticCount(console)}</strong></header>
      {console.errors.length > 0 ? <DiagnosticList items={console.errors} label="Runtime errors" tone="danger" /> : null}
      {console.warnings.length > 0 ? <DiagnosticList items={console.warnings} label="Runtime warnings" tone="warning" /> : null}
      {console.errors.length === 0 && console.warnings.length === 0 ? (
        <p className="runtimeConsoleMuted">No runtime warnings or errors are active.</p>
      ) : null}
    </article>
  );
}

function RuntimeContextCard({ console }: { console: RuntimeConsoleModel }) {
  return (
    <article aria-label="Runtime context" className="runtimeConsoleCard" role="group">
      <header><span>Context</span><strong>{console.context.runtimeTypeLabel}</strong></header>
      <dl className="runtimeConsoleDetails">
        <div><dt>Artifact</dt><dd>{console.context.artifactName}</dd></div>
        <div><dt>Source</dt><dd>{console.context.sourceName}</dd></div>
        <div><dt>Target</dt><dd>{console.context.targetLabel}</dd></div>
        <div><dt>Renderer</dt><dd>{console.context.rendererLabel}</dd></div>
        <div><dt>Support</dt><dd>{console.context.supportLabel}</dd></div>
        <div><dt>Fingerprint</dt><dd>{console.context.fingerprint}</dd></div>
        <div><dt>Source size</dt><dd>{console.context.lineCountLabel}</dd></div>
      </dl>
    </article>
  );
}

function RuntimeReloadHistoryCard({ console }: { console: RuntimeConsoleModel }) {
  return (
    <article aria-label="Runtime reload history" className="runtimeConsoleCard runtimeConsoleCard--wide" role="group">
      <header><span>Reload history</span><strong>{formatReloadCount(console.reloadHistory.length)}</strong></header>
      {console.reloadHistory.length > 0 ? (
        <RuntimeEventList events={console.reloadHistory} />
      ) : (
        <p className="runtimeConsoleMuted">No preview runtime reloads have been requested.</p>
      )}
    </article>
  );
}

function RuntimeEventHistoryCard({ console }: { console: RuntimeConsoleModel }) {
  return (
    <article aria-label="Runtime event history" className="runtimeConsoleCard runtimeConsoleCard--wide" role="group">
      <header><span>Event history</span><strong>{console.events.length} chronological events</strong></header>
      {console.events.length > 0 ? (
        <RuntimeEventList events={console.events} />
      ) : (
        <p className="runtimeConsoleMuted">Runtime events are captured after the live preview renderer starts.</p>
      )}
    </article>
  );
}

function DiagnosticList({
  items,
  label,
  tone
}: {
  items: readonly string[];
  label: string;
  tone: "warning" | "danger";
}) {
  return (
    <div
      aria-label={label}
      className="runtimeConsoleDiagnosticList"
      data-tone={tone}
      role="list"
    >
      {items.map((item) => (
        <p key={item} role="listitem">
          {item}
        </p>
      ))}
    </div>
  );
}

function RuntimeEventList({ events }: { events: RuntimeConsoleEvent[] }) {
  return (
    <div className="runtimeConsoleEventList">
      {events.map((event) => (
        <article
          className="runtimeConsoleEvent"
          data-event-kind={event.kind}
          data-tone={event.tone}
          key={event.id}
        >
          <div>
            <strong>{event.label}</strong>
            <p>
              <span>{event.stateLabel}</span>
              <span>{event.atLabel}</span>
              {event.runtimeTypeLabel ? <span>{event.runtimeTypeLabel}</span> : null}
            </p>
          </div>
          {event.artifactName ? <small>{event.artifactName}</small> : null}
          <span>{event.detail}</span>
        </article>
      ))}
    </div>
  );
}

function formatDiagnosticCount(console: RuntimeConsoleModel) {
  const count = console.errors.length + console.warnings.length;

  return count === 0 ? "Clear" : `${count} active`;
}

function formatReloadCount(count: number) {
  return `${count} ${count === 1 ? "reload" : "reloads"}`;
}
