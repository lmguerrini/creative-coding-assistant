import type {
  CreativeCostEstimate,
  CreativeCostIntelligenceModel
} from "@/lib/creative-cost-intelligence";

type CreativeCostIntelligenceDashboardProps = {
  intelligence: CreativeCostIntelligenceModel;
};

export function CreativeCostIntelligenceDashboard({
  intelligence
}: CreativeCostIntelligenceDashboardProps) {
  const { current, estimate, session } = intelligence;

  return (
    <article
      aria-label="Creative cost intelligence dashboard"
      className="creativeCostDashboard telemetryDashboardCard telemetryDashboardCard--wide"
      data-state={current.state}
      role="group"
    >
      <header className="creativeCostHeader">
        <div>
          <span>Creative cost intelligence</span>
          <strong>{formatSessionCost(session)}</strong>
          <p>{formatCoverageDetail(session)}</p>
        </div>
        <span className="creativeCostCoverage" data-coverage={session.coverage}>
          {formatCoverageLabel(session.coverage)}
        </span>
      </header>

      <div className="creativeCostSections">
        <section
          aria-label="Pre-generation cost estimate"
          className="creativeCostSection"
          data-state={estimate.state}
        >
          <header>
            <div>
              <span>Before generation</span>
              <strong>{formatEstimateCost(estimate)}</strong>
            </div>
            <small>{formatProviderLabel(estimate)}</small>
          </header>
          <p>{estimate.detail}</p>
          {estimate.inputTokenRange && estimate.outputTokenRange ? (
            <dl>
              <Metric
                label="Prompt"
                value={formatTokens(estimate.promptTokens)}
              />
              <Metric
                label="Context"
                value={formatTokens(estimate.contextTokens)}
              />
              <Metric
                label="Estimated input"
                value={formatTokenRange(estimate.inputTokenRange)}
              />
              <Metric
                label="Estimated output"
                value={formatTokenRange(estimate.outputTokenRange)}
              />
            </dl>
          ) : null}
          {estimate.assumptions.length > 0 ? (
            <ul className="creativeCostAssumptions">
              {estimate.assumptions.map((assumption) => (
                <li key={assumption}>{assumption}</li>
              ))}
            </ul>
          ) : null}
        </section>

        <section
          aria-label="Latest generation cost"
          className="creativeCostSection"
          data-state={current.state}
        >
          <header>
            <div>
              <span>Latest generation</span>
              <strong>
                {current.cost == null
                  ? current.state === "running"
                    ? "Usage pending"
                    : current.state === "idle"
                      ? "No completed run"
                    : "Cost unavailable"
                  : formatCurrency(current.cost, current.currency)}
              </strong>
            </div>
            <small>{formatCurrentProviderLabel(current)}</small>
          </header>
          {current.state === "idle" ? (
            <p>
              Token usage, duration, and creative scope appear after the first
              generation.
            </p>
          ) : (
            <>
              <dl>
                <Metric label="Input" value={formatTokens(current.inputTokens)} />
                <Metric label="Output" value={formatTokens(current.outputTokens)} />
                <Metric label="Total" value={formatTokens(current.totalTokens)} />
                <Metric
                  label="Duration"
                  value={formatDuration(current.durationMs)}
                />
              </dl>
              {current.state === "running" ? (
                <p>Provider usage and creative scope are still being collected.</p>
              ) : (
                <>
                  <div className="creativeCostRunContext">
                    <span>
                      {formatRetryContext(
                        current.retryCount,
                        current.retryCost,
                        current.currency
                      )}
                    </span>
                    <span>
                      {formatFallbackContext(
                        current.fallbackCount,
                        current.fallbackCost,
                        current.currency
                      )}
                    </span>
                    <span>{formatCostSource(current.costSource)}</span>
                  </div>
                  <div
                    aria-label="Latest creative cost context"
                    className="creativeCostRunContext"
                    role="group"
                  >
                    <span>{formatCount(current.artifactCount, "artifact")}</span>
                    <span>
                      {formatCount(current.refinementCount, "refinement")}
                    </span>
                    <span>
                      {formatCount(current.critiqueCount, "critique")} /{" "}
                      {formatCount(current.reviewCount, "review")}
                    </span>
                  </div>
                </>
              )}
            </>
          )}
        </section>

        <section
          aria-label="Session cost summary"
          className="creativeCostSection creativeCostSection--wide"
        >
          <header>
            <div>
              <span>Session summary</span>
              <strong>{formatRunCount(session.runCount)}</strong>
            </div>
            <small>
              {session.costedRunCount} of {session.runCount} runs costed
            </small>
          </header>
          <dl className="creativeCostSessionMetrics">
            <Metric
              label="Generations"
              value={session.generationCount.toLocaleString()}
            />
            <Metric
              label="Refinements"
              value={session.refinementCount.toLocaleString()}
            />
            <Metric
              label="Artifacts"
              value={session.artifactCount.toLocaleString()}
            />
            <Metric
              label="Avg / generation"
              value={formatNullableCurrency(
                session.averagePerGeneration,
                session.currency
              )}
            />
            <Metric
              label="Avg / artifact"
              value={formatNullableCurrency(
                session.averagePerArtifact,
                session.currency
              )}
            />
            <Metric
              label="Review activity"
              value={`${formatCount(
                session.critiqueCount,
                "critique"
              )} / ${formatCount(session.reviewCount, "review")}`}
            />
          </dl>
        </section>
      </div>
    </article>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function formatEstimateCost(estimate: CreativeCostEstimate) {
  if (estimate.state === "empty") {
    return "Awaiting draft";
  }
  if (!estimate.costRange) {
    return "Pricing unavailable";
  }
  return `${formatCurrency(
    estimate.costRange[0],
    estimate.currency
  )} - ${formatCurrency(estimate.costRange[1], estimate.currency)}`;
}

function formatSessionCost(
  session: CreativeCostIntelligenceModel["session"]
) {
  if (session.totalCost == null) {
    return "Session cost unavailable";
  }
  const prefix = session.coverage === "partial" ? "Partial total " : "";
  return `${prefix}${formatCurrency(session.totalCost, session.currency)}`;
}

function formatCoverageDetail(
  session: CreativeCostIntelligenceModel["session"]
) {
  if (session.runCount === 0) {
    return "Completed generation costs will accumulate for this workspace session.";
  }
  if (session.coverage === "complete") {
    return "Every completed run in this session includes compatible cost metadata.";
  }
  if (session.coverage === "partial") {
    return "Only runs with compatible provider cost metadata are included in the total.";
  }
  return "Runs are tracked, but provider pricing or reported cost is unavailable.";
}

function formatCoverageLabel(
  coverage: CreativeCostIntelligenceModel["session"]["coverage"]
) {
  switch (coverage) {
    case "complete":
      return "Complete coverage";
    case "partial":
      return "Partial coverage";
    default:
      return "No cost coverage";
  }
}

function formatProviderLabel(estimate: CreativeCostEstimate) {
  const provider = estimate.providerName ?? "Provider pending";
  const model = estimate.modelName ?? "Model pending";
  const mode =
    estimate.generationMode === "streaming"
      ? "streaming"
      : estimate.generationMode === "non_streaming"
        ? "non-streaming"
        : "mode pending";
  return `${provider} / ${model} / ${mode}`;
}

function formatCurrentProviderLabel(
  current: CreativeCostIntelligenceModel["current"]
) {
  return `${current.providerName ?? "Provider pending"} / ${
    current.modelName ?? "Model pending"
  }`;
}

function formatTokenRange(range: [number, number]) {
  return `${formatNumber(range[0])} - ${formatNumber(range[1])}`;
}

function formatTokens(value: number | null) {
  return value == null ? "Unavailable" : `${formatNumber(value)} tokens`;
}

function formatDuration(value: number | null) {
  if (value == null) {
    return "Unavailable";
  }
  return value < 1000 ? `${Math.round(value)} ms` : `${(value / 1000).toFixed(1)} s`;
}

function formatRetryContext(
  retryCount: number | null,
  retryCost: number | null,
  currency: string
) {
  if (retryCount == null) {
    return "Retry metadata unavailable";
  }
  const countLabel = retryCount === 1 ? "1 retry" : `${retryCount} retries`;
  return retryCost == null
    ? countLabel
    : `${countLabel} / ${formatCurrency(retryCost, currency)}`;
}

function formatFallbackContext(
  fallbackCount: number,
  fallbackCost: number | null,
  currency: string
) {
  const countLabel =
    fallbackCount === 1 ? "1 fallback" : `${fallbackCount} fallbacks`;
  return fallbackCost == null
    ? countLabel
    : `${countLabel} / ${formatCurrency(fallbackCost, currency)}`;
}

function formatCostSource(
  source: CreativeCostIntelligenceModel["current"]["costSource"]
) {
  switch (source) {
    case "provider_reported":
      return "Provider-reported cost";
    case "pricing_metadata":
      return "Estimated from pricing metadata";
    default:
      return "No cost source";
  }
}

function formatRunCount(value: number) {
  return value === 1 ? "1 completed run" : `${value} completed runs`;
}

function formatCount(value: number, label: string) {
  return `${value.toLocaleString()} ${label}${value === 1 ? "" : "s"}`;
}

function formatNullableCurrency(value: number | null, currency: string) {
  return value == null ? "Unavailable" : formatCurrency(value, currency);
}

function formatCurrency(value: number, currency: string) {
  const maximumFractionDigits = value < 0.01 ? 6 : 4;
  return new Intl.NumberFormat("en", {
    style: "currency",
    currency,
    minimumFractionDigits: Math.min(4, maximumFractionDigits),
    maximumFractionDigits
  }).format(value);
}

function formatNumber(value: number) {
  return new Intl.NumberFormat("en", { maximumFractionDigits: 0 }).format(value);
}
