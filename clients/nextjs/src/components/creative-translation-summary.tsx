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
      {translation.sacredGeometry ? (
        <section
          aria-label="Sacred geometry guidance"
          className="sacredGeometrySummary"
        >
          <header>
            <span>Sacred geometry</span>
            <strong>{translation.sacredGeometry.concepts.join(" / ")}</strong>
          </header>
          <dl>
            {translation.sacredGeometry.symmetryType.length > 0 ? (
              <div>
                <dt>Symmetry</dt>
                <dd>
                  {translation.sacredGeometry.symmetryType.join(" ")}
                </dd>
              </div>
            ) : null}
            {translation.sacredGeometry.movementBehavior.length > 0 ? (
              <div>
                <dt>Movement</dt>
                <dd>
                  {translation.sacredGeometry.movementBehavior.join(" ")}
                </dd>
              </div>
            ) : null}
            {translation.sacredGeometry.runtimeRecommendations.length > 0 ? (
              <div>
                <dt>Runtime</dt>
                <dd>
                  {translation.sacredGeometry.runtimeRecommendations.join(
                    " / "
                  )}
                </dd>
              </div>
            ) : null}
            {translation.sacredGeometry.audioImplications.length > 0 ? (
              <div>
                <dt>Audio</dt>
                <dd>
                  {translation.sacredGeometry.audioImplications.join(" ")}
                </dd>
              </div>
            ) : null}
          </dl>
        </section>
      ) : null}
      {translation.shaderPresets ? (
        <section
          aria-label="Shader preset guidance"
          className="shaderPresetSummary"
        >
          <header>
            <span>Shader presets</span>
            <strong>{translation.shaderPresets.presets.join(" / ")}</strong>
          </header>
          <dl>
            {translation.shaderPresets.lightMaterialBehavior.length > 0 ? (
              <div>
                <dt>Material</dt>
                <dd>
                  {translation.shaderPresets.lightMaterialBehavior.join(" ")}
                </dd>
              </div>
            ) : null}
            {translation.shaderPresets.motionBehavior.length > 0 ? (
              <div>
                <dt>Motion</dt>
                <dd>{translation.shaderPresets.motionBehavior.join(" ")}</dd>
              </div>
            ) : null}
            {translation.shaderPresets.runtimeSuitability.length > 0 ? (
              <div>
                <dt>Runtime</dt>
                <dd>
                  {translation.shaderPresets.runtimeSuitability.join(" ")}
                </dd>
              </div>
            ) : null}
            {translation.shaderPresets.performanceConstraints.length > 0 ? (
              <div>
                <dt>Budget</dt>
                <dd>
                  {translation.shaderPresets.performanceConstraints[0]}
                </dd>
              </div>
            ) : null}
          </dl>
        </section>
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
