import type { CreativeTranslationSummary } from "./assistant-client";

export function normalizeCreativeTranslation(
  value: unknown
): CreativeTranslationSummary | null {
  const record = readRecord(value);
  if (!record) {
    return null;
  }

  const creativeIntent = readString(
    record.creative_intent ?? record.creativeIntent
  );
  if (!creativeIntent) {
    return null;
  }

  return {
    outputModality: readOutputModality(
      record.output_modality ?? record.outputModality
    ),
    creativeIntent,
    symbolicReferences: readStringList(
      record.symbolic_references ?? record.symbolicReferences
    ),
    geometricReferences: readStringList(
      record.geometric_references ?? record.geometricReferences
    ),
    musicalReferences: readStringList(
      record.musical_references ?? record.musicalReferences
    ),
    moodAtmosphere: readStringList(
      record.mood_atmosphere ?? record.moodAtmosphere
    ),
    movementLanguage: readStringList(
      record.movement_language ?? record.movementLanguage
    ),
    colorMaterialDirection: readStringList(
      record.color_material_direction ?? record.colorMaterialDirection
    ),
    runtimeRecommendations: readStringList(
      record.runtime_recommendations ?? record.runtimeRecommendations
    ),
    structureDirection: readStringList(
      record.structure_direction ?? record.structureDirection
    ),
    generationConstraints: readStringList(
      record.generation_constraints ?? record.generationConstraints
    ),
    refinementTargets: readStringList(
      record.refinement_targets ?? record.refinementTargets
    )
  };
}

function readOutputModality(
  value: unknown
): CreativeTranslationSummary["outputModality"] {
  return value === "visual" || value === "audio" || value === "audiovisual"
    ? value
    : null;
}

function readStringList(value: unknown) {
  if (!Array.isArray(value)) {
    return [];
  }

  return Array.from(
    new Set(
      value
        .map((item) => readString(item))
        .filter((item): item is string => item !== null)
    )
  ).slice(0, 8);
}

function readRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null
    ? (value as Record<string, unknown>)
    : null;
}

function readString(value: unknown) {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}
