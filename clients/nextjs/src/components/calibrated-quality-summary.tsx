import type {
  CalibratedQualityEvaluation,
  CalibratedQualitySignal
} from "@/lib/assistant-client";

type CalibratedQualitySummaryProps = {
  evaluation: CalibratedQualityEvaluation | null | undefined;
};

export function CalibratedQualitySummary({
  evaluation
}: CalibratedQualitySummaryProps) {
  if (!evaluation) {
    return null;
  }

  return (
    <section
      aria-label="Calibrated quality summary"
      className="calibratedQuality"
      data-band={evaluation.decisionBand}
    >
      <header>
        <div>
          <span>Calibrated quality</span>
          <strong>{formatDecisionBand(evaluation.decisionBand)}</strong>
        </div>
        <span className="calibratedQualityScore">
          {formatScore(evaluation.score)}
        </span>
      </header>

      <p className="calibratedQualitySummary">{evaluation.summary}</p>
      <p className="calibratedQualityRationale">{evaluation.rationale}</p>

      <dl className="calibratedQualityMeta">
        <div>
          <dt>Legacy score</dt>
          <dd>{formatScore(evaluation.legacyScore)}</dd>
        </div>
        <div>
          <dt>Confidence</dt>
          <dd>{evaluation.confidence}</dd>
        </div>
      </dl>

      <div
        aria-label="Calibrated quality signals"
        className="calibratedQualitySignals"
        role="list"
      >
        {evaluation.signals.map((signal) => (
          <CalibratedQualitySignalRow
            key={signal.key}
            signal={signal}
          />
        ))}
      </div>

      {evaluation.adjustments.length > 0 ? (
        <div className="calibratedQualityAdjustments">
          <strong>Conservative adjustments</strong>
          <ul>
            {evaluation.adjustments.map((adjustment) => (
              <li key={adjustment}>{adjustment}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </section>
  );
}

function CalibratedQualitySignalRow({
  signal
}: {
  signal: CalibratedQualitySignal;
}) {
  return (
    <article className="calibratedQualitySignal" role="listitem">
      <div>
        <strong>{signal.label}</strong>
        <span>{`${formatScore(signal.score)} / ${Math.round(signal.weight * 100)}%`}</span>
      </div>
      <p>{signal.rationale}</p>
    </article>
  );
}

function formatDecisionBand(band: CalibratedQualityEvaluation["decisionBand"]) {
  switch (band) {
    case "strong_candidate":
      return "Strong candidate";
    case "usable_candidate":
      return "Usable candidate";
    case "needs_refinement":
      return "Needs refinement";
    case "high_risk":
      return "High risk";
  }
}

function formatScore(score: number) {
  return `${Math.round(Math.min(Math.max(score, 0), 1) * 100)}%`;
}
