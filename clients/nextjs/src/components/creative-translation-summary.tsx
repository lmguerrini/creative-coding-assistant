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
      label: "Concepts",
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
      <p>{formatPublicCreativeText(translation.creativeIntent)}</p>
      {groups.length > 0 ? (
        <dl className="creativeTranslationGroups">
          {groups.map((group) => (
            <div key={group.label}>
              <dt>{group.label}</dt>
              <dd>{formatPublicCreativeList(group.values).join(" / ")}</dd>
            </div>
          ))}
        </dl>
      ) : null}
      {translation.referenceFusion ? (
        <section
          aria-label="Reference fusion guidance"
          className="referenceFusionSummary"
        >
          <header>
            <span>Reference fusion</span>
            <strong>
              {formatSourceCount(translation.referenceFusion.sourceCount)}
            </strong>
          </header>
          <p>{formatPublicCreativeText(translation.referenceFusion.summary)}</p>
          <dl>
            {translation.referenceFusion.paletteDirection.length > 0 ? (
              <div>
                <dt>Palette</dt>
                <dd>
                  {formatPublicCreativeList(
                    translation.referenceFusion.paletteDirection
                  ).join(" / ")}
                </dd>
              </div>
            ) : null}
            {translation.referenceFusion.composition.length > 0 ? (
              <div>
                <dt>Compose</dt>
                <dd>
                  {formatPublicCreativeList(
                    translation.referenceFusion.composition
                  ).join(" / ")}
                </dd>
              </div>
            ) : null}
            {translation.referenceFusion.textureMaterialCues.length > 0 ? (
              <div>
                <dt>Material</dt>
                <dd>
                  {formatPublicCreativeList(
                    translation.referenceFusion.textureMaterialCues
                  ).join(" / ")}
                </dd>
              </div>
            ) : null}
            {translation.referenceFusion.motionImplications.length > 0 ? (
              <div>
                <dt>Motion</dt>
                <dd>
                  {formatPublicCreativeList(
                    translation.referenceFusion.motionImplications
                  ).join(" / ")}
                </dd>
              </div>
            ) : null}
            {translation.referenceFusion.runtimeStyleImplications.length > 0 ? (
              <div>
                <dt>Runtime</dt>
                <dd>
                  {formatPublicCreativeText(
                    translation.referenceFusion.runtimeStyleImplications[0]
                  )}
                </dd>
              </div>
            ) : null}
          </dl>
          {translation.referenceFusion.safetyConstraints.length > 0 ? (
            <small>
              {formatPublicCreativeText(
                translation.referenceFusion.safetyConstraints[0]
              )}
            </small>
          ) : null}
        </section>
      ) : null}
      {translation.sacredGeometry ? (
        <section
          aria-label="Geometry guidance"
          className="sacredGeometrySummary"
        >
          <header>
            <span>Geometry</span>
            <strong>
              {formatPublicCreativeList(translation.sacredGeometry.concepts).join(
                " / "
              )}
            </strong>
          </header>
          <dl>
            {translation.sacredGeometry.symmetryType.length > 0 ? (
              <div>
                <dt>Symmetry</dt>
                <dd>
                  {formatPublicCreativeList(
                    translation.sacredGeometry.symmetryType
                  ).join(" ")}
                </dd>
              </div>
            ) : null}
            {translation.sacredGeometry.movementBehavior.length > 0 ? (
              <div>
                <dt>Movement</dt>
                <dd>
                  {formatPublicCreativeList(
                    translation.sacredGeometry.movementBehavior
                  ).join(" ")}
                </dd>
              </div>
            ) : null}
            {translation.sacredGeometry.runtimeRecommendations.length > 0 ? (
              <div>
                <dt>Runtime</dt>
                <dd>
                  {formatPublicCreativeList(
                    translation.sacredGeometry.runtimeRecommendations
                  ).join(" / ")}
                </dd>
              </div>
            ) : null}
            {translation.sacredGeometry.audioImplications.length > 0 ? (
              <div>
                <dt>Audio</dt>
                <dd>
                  {formatPublicCreativeList(
                    translation.sacredGeometry.audioImplications
                  ).join(" ")}
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
            <strong>
              {formatPublicCreativeList(translation.shaderPresets.presets).join(
                " / "
              )}
            </strong>
          </header>
          <dl>
            {translation.shaderPresets.lightMaterialBehavior.length > 0 ? (
              <div>
                <dt>Material</dt>
                <dd>
                  {formatPublicCreativeList(
                    translation.shaderPresets.lightMaterialBehavior
                  ).join(" ")}
                </dd>
              </div>
            ) : null}
            {translation.shaderPresets.motionBehavior.length > 0 ? (
              <div>
                <dt>Motion</dt>
                <dd>
                  {formatPublicCreativeList(
                    translation.shaderPresets.motionBehavior
                  ).join(" ")}
                </dd>
              </div>
            ) : null}
            {translation.shaderPresets.runtimeSuitability.length > 0 ? (
              <div>
                <dt>Runtime</dt>
                <dd>
                  {formatPublicCreativeList(
                    translation.shaderPresets.runtimeSuitability
                  ).join(" ")}
                </dd>
              </div>
            ) : null}
            {translation.shaderPresets.performanceConstraints.length > 0 ? (
              <div>
                <dt>Budget</dt>
                <dd>
                  {formatPublicCreativeText(
                    translation.shaderPresets.performanceConstraints[0]
                  )}
                </dd>
              </div>
            ) : null}
          </dl>
        </section>
      ) : null}
      {translation.visualStyle ? (
        <section
          aria-label="Visual style guidance"
          className="visualStyleSummary"
        >
          <header>
            <span>Visual style</span>
            <strong>
              {formatPublicCreativeList(translation.visualStyle.styles).join(
                " / "
              )}
            </strong>
          </header>
          <dl>
            {translation.visualStyle.paletteBehavior.length > 0 ? (
              <div>
                <dt>Palette</dt>
                <dd>
                  {formatPublicCreativeText(
                    translation.visualStyle.paletteBehavior[0]
                  )}
                </dd>
              </div>
            ) : null}
            {translation.visualStyle.compositionTendencies.length > 0 ? (
              <div>
                <dt>Compose</dt>
                <dd>
                  {formatPublicCreativeText(
                    translation.visualStyle.compositionTendencies[0]
                  )}
                </dd>
              </div>
            ) : null}
            {translation.visualStyle.motionTendencies.length > 0 ? (
              <div>
                <dt>Motion</dt>
                <dd>
                  {formatPublicCreativeText(
                    translation.visualStyle.motionTendencies[0]
                  )}
                </dd>
              </div>
            ) : null}
            {translation.visualStyle.runtimeSuitability.length > 0 ? (
              <div>
                <dt>Runtime</dt>
                <dd>
                  {formatPublicCreativeList(
                    translation.visualStyle.runtimeSuitability
                  ).join(" ")}
                </dd>
              </div>
            ) : null}
          </dl>
        </section>
      ) : null}
      {translation.refinementTargets.length > 0 ? (
        <p className="creativeTranslationTargets">
          <strong>Refine</strong>
          {formatPublicCreativeList(translation.refinementTargets).join(" · ")}
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

function formatPublicCreativeList(values: readonly string[]) {
  return values.map(formatPublicCreativeText);
}

function formatPublicCreativeText(value: string) {
  return value
    .replace(/\bsacred geometry\b/gi, "geometry")
    .replace(/\bsacred\b/gi, "geometric")
    .replace(/\bsymbolic references?\b/gi, "concept references")
    .replace(/\bsymbolic\b/gi, "conceptual");
}

function formatSourceCount(sourceCount: number) {
  return `${sourceCount} ${sourceCount === 1 ? "reference" : "references"}`;
}
