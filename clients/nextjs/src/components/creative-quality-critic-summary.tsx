import type {
  CreativeQualityEvaluation,
  CreativeQualityObservation
} from "@/lib/assistant-client";

type CreativeQualityCriticSummaryProps = {
  evaluation: CreativeQualityEvaluation | null | undefined;
};

export function CreativeQualityCriticSummary({
  evaluation
}: CreativeQualityCriticSummaryProps) {
  if (!evaluation) {
    return null;
  }

  const dimensions = [
    ["Composition", evaluation.composition],
    ["Originality", evaluation.originality],
    ["Coherence", evaluation.coherence],
    ["Aesthetic consistency", evaluation.aestheticConsistency],
    ["Expressiveness", evaluation.expressiveness]
  ] as const;

  return (
    <section
      aria-label="Creative quality critic"
      className="creativeQualityCritic"
    >
      <header>
        <div>
          <span>Creative quality critic</span>
          <strong>Bounded artistic analysis</strong>
        </div>
        <span className="creativeQualityScore">
          {formatScore(evaluation.overallScore)}
        </span>
      </header>

      <p className="creativeQualitySummary">{evaluation.summary}</p>

      <div
        aria-label="Creative quality dimensions"
        className="creativeQualityDimensions"
        role="list"
      >
        {dimensions.map(([label, observation]) => (
          <CreativeQualityDimension
            key={label}
            label={label}
            observation={observation}
          />
        ))}
      </div>

      {evaluation.strengths.length > 0 ? (
        <div className="creativeQualityNotes">
          <strong>Strengths</strong>
          <ul>
            {evaluation.strengths.map((strength) => (
              <li key={strength}>{strength}</li>
            ))}
          </ul>
        </div>
      ) : null}

      {evaluation.refinementOpportunities.length > 0 ? (
        <div className="creativeQualityNotes" data-tone="refine">
          <strong>Refinement opportunities</strong>
          <ul>
            {evaluation.refinementOpportunities.map((opportunity) => (
              <li key={opportunity}>{opportunity}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </section>
  );
}

function CreativeQualityDimension({
  label,
  observation
}: {
  label: string;
  observation: CreativeQualityObservation;
}) {
  return (
    <article
      className="creativeQualityDimension"
      data-level={observation.level}
      role="listitem"
    >
      <div>
        <strong>{label}</strong>
        <span>{formatScore(observation.score)}</span>
      </div>
      <p>{observation.observation}</p>
      <small>{observation.level}</small>
    </article>
  );
}

function formatScore(score: number) {
  return `${Math.round(Math.min(Math.max(score, 0), 1) * 100)}%`;
}
