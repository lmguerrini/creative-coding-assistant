import type {
  CreativeTranslationSummary,
  SacredGeometrySummary,
  ShaderPresetName,
  ShaderPresetSummary,
  VisualStyleName,
  VisualStyleSummary
} from "./assistant-client";

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
    ),
    sacredGeometry: normalizeSacredGeometry(
      record.sacred_geometry ?? record.sacredGeometry
    ),
    shaderPresets: normalizeShaderPresets(
      record.shader_presets ?? record.shaderPresets
    ),
    visualStyle: normalizeVisualStyle(
      record.visual_style ?? record.visualStyle
    )
  };
}

function normalizeSacredGeometry(
  value: unknown
): SacredGeometrySummary | null {
  const record = readRecord(value);
  if (!record) {
    return null;
  }

  const concepts = readStringList(record.concepts);
  if (concepts.length === 0) {
    return null;
  }

  return {
    concepts,
    geometricStructure: readStringList(
      record.geometric_structure ?? record.geometricStructure
    ),
    symmetryType: readStringList(
      record.symmetry_type ?? record.symmetryType
    ),
    movementBehavior: readStringList(
      record.movement_behavior ?? record.movementBehavior
    ),
    visualComposition: readStringList(
      record.visual_composition ?? record.visualComposition
    ),
    colorMaterialDirection: readStringList(
      record.color_material_direction ?? record.colorMaterialDirection
    ),
    runtimeRecommendations: readStringList(
      record.runtime_recommendations ?? record.runtimeRecommendations
    ),
    audioImplications: readStringList(
      record.audio_implications ?? record.audioImplications
    ),
    generationConstraints: readStringList(
      record.generation_constraints ?? record.generationConstraints
    )
  };
}

function normalizeShaderPresets(value: unknown): ShaderPresetSummary | null {
  const record = readRecord(value);
  if (!record) {
    return null;
  }

  const presets = readShaderPresetList(record.presets);
  if (presets.length === 0) {
    return null;
  }

  return {
    presets,
    colorBehavior: readStringList(
      record.color_behavior ?? record.colorBehavior
    ),
    lightMaterialBehavior: readStringList(
      record.light_material_behavior ?? record.lightMaterialBehavior
    ),
    motionBehavior: readStringList(
      record.motion_behavior ?? record.motionBehavior
    ),
    shaderStructure: readStringList(
      record.shader_structure ?? record.shaderStructure
    ),
    runtimeSuitability: readStringList(
      record.runtime_suitability ?? record.runtimeSuitability
    ),
    performanceConstraints: readStringList(
      record.performance_constraints ?? record.performanceConstraints
    )
  };
}

function normalizeVisualStyle(value: unknown): VisualStyleSummary | null {
  const record = readRecord(value);
  if (!record) {
    return null;
  }

  const styles = readVisualStyleList(record.styles);
  if (styles.length === 0) {
    return null;
  }

  return {
    styles,
    paletteBehavior: readStringList(
      record.palette_behavior ?? record.paletteBehavior
    ),
    contrastBehavior: readStringList(
      record.contrast_behavior ?? record.contrastBehavior
    ),
    compositionTendencies: readStringList(
      record.composition_tendencies ?? record.compositionTendencies
    ),
    motionTendencies: readStringList(
      record.motion_tendencies ?? record.motionTendencies
    ),
    textureTendencies: readStringList(
      record.texture_tendencies ?? record.textureTendencies
    ),
    spatialOrganization: readStringList(
      record.spatial_organization ?? record.spatialOrganization
    ),
    runtimeSuitability: readStringList(
      record.runtime_suitability ?? record.runtimeSuitability
    )
  };
}

const shaderPresetNames = new Set<ShaderPresetName>([
  "glow",
  "aura",
  "plasma",
  "bloom-like emission",
  "refraction",
  "glass / crystal",
  "volumetric atmosphere",
  "fractal field",
  "kaleidoscopic symmetry",
  "sacred light / ritual ambience"
]);

function readShaderPresetList(value: unknown): ShaderPresetName[] {
  return readStringList(value).filter(
    (item): item is ShaderPresetName =>
      shaderPresetNames.has(item as ShaderPresetName)
  );
}

const visualStyleNames = new Set<VisualStyleName>([
  "minimal",
  "cyberpunk",
  "organic",
  "ritual",
  "sacred geometry",
  "generative modernism",
  "retro computational",
  "ethereal",
  "psychedelic",
  "architectural",
  "monochrome",
  "maximalist"
]);

function readVisualStyleList(value: unknown): VisualStyleName[] {
  return readStringList(value).filter(
    (item): item is VisualStyleName =>
      visualStyleNames.has(item as VisualStyleName)
  );
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
