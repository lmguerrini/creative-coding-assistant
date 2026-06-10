import type { CreativeTranslationSummary } from "@/lib/assistant-client";

type CreativeTranslationSummaryProps = {
  translation: CreativeTranslationSummary | null | undefined;
};

export function CreativeTranslationSummaryCard({
  translation
}: CreativeTranslationSummaryProps) {
  if (!translation) {
    return (
      <section
        aria-label="Creative translation summary"
        className="creativeTranslationCard"
        data-state="legacy"
      >
        <header>
          <div>
            <span>Creative translation</span>
            <strong>Not recorded</strong>
          </div>
        </header>
        <p>
          This legacy artifact does not include structured creative guidance.
        </p>
      </section>
    );
  }

  const groups = [
    {
      label: "Symbols",
      values: translation.symbolicReferences
    },
    {
      label: "Geometry",
      values: translation.geometricReferences
    },
    {
      label: "Music",
      values: translation.musicalReferences
    },
    {
      label: "Atmosphere",
      values: [
        ...translation.moodAtmosphere,
        ...translation.colorMaterialDirection
      ]
    },
    {
      label: "Movement",
      values: translation.movementLanguage
    },
    {
      label: "Runtime",
      values: translation.runtimeRecommendations
    }
  ].filter((group) => group.values.length > 0);

  return (
    <section
      aria-label="Creative translation summary"
      className="creativeTranslationCard"
      data-state="available"
    >
      <header>
        <div>
          <span>Creative translation</span>
          <strong>{formatModality(translation.outputModality)}</strong>
        </div>
      </header>
      <p>{translation.creativeIntent}</p>
      {groups.length > 0 ? (
        <dl className="creativeTranslationGroups">
          {groups.map((group) => (
            <div key={group.label}>
              <dt>{group.label}</dt>
              <dd>{group.values.join(" / ")}</dd>
            </div>
          ))}
        </dl>
      ) : null}
      {translation.refinementTargets.length > 0 ? (
        <p className="creativeTranslationTargets">
          <strong>Refine</strong>
          {translation.refinementTargets.join(" · ")}
        </p>
      ) : null}
    </section>
  );
}

function formatModality(
  modality: CreativeTranslationSummary["outputModality"]
) {
  switch (modality) {
    case "audiovisual":
      return "Audiovisual";
    case "audio":
      return "Audio";
    case "visual":
      return "Visual";
    default:
      return "Modality open";
  }
}
