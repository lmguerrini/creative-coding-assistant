import type {
  SacredConsistencyEvaluation,
  SacredConsistencyObservation
} from "@/lib/assistant-client";

type SacredConsistencySummaryProps = {
  evaluation: SacredConsistencyEvaluation | null | undefined;
};

export function SacredConsistencySummary({
  evaluation
}: SacredConsistencySummaryProps) {
  if (!evaluation) {
    return null;
  }

  const dimensions = [
    ["Guidance alignment", evaluation.alignment],
    ["Geometric motifs", evaluation.motifConsistency],
    ["Modality coherence", evaluation.modalityCoherence],
    ["Claim safety", evaluation.claimSafety]
  ] as const;

  return (
    <section
      aria-label="Sacred consistency evaluator"
      className="sacredConsistency"
    >
      <header>
        <div>
          <span>Sacred consistency evaluator</span>
          <strong>Bounded motif analysis</strong>
        </div>
        <span className="sacredConsistencyScore">
          {formatScore(evaluation.overallScore)}
        </span>
      </header>

      <p className="sacredConsistencySummary">{evaluation.summary}</p>

      <div
        aria-label="Sacred consistency dimensions"
        className="sacredConsistencyDimensions"
        role="list"
      >
        {dimensions.map(([label, observation]) => (
          <SacredConsistencyDimension
            key={label}
            label={label}
            observation={observation}
          />
        ))}
      </div>

      {evaluation.strengths.length > 0 ? (
        <div className="sacredConsistencyNotes">
          <strong>Strengths</strong>
          <ul>
            {evaluation.strengths.map((strength) => (
              <li key={strength}>{strength}</li>
            ))}
          </ul>
        </div>
      ) : null}

      {evaluation.refinementOpportunities.length > 0 ? (
        <div className="sacredConsistencyNotes" data-tone="refine">
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

function SacredConsistencyDimension({
  label,
  observation
}: {
  label: string;
  observation: SacredConsistencyObservation;
}) {
  return (
    <article
      className="sacredConsistencyDimension"
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
