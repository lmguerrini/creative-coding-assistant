import type {
  AudioReactiveGuidanceSummary,
  AudioReactiveSource,
  AudioReactiveVisualTarget,
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
    ),
    audioReactive: normalizeAudioReactive(
      record.audio_reactive ??
        record.audioReactive ??
        record.audio_reactive_mappings ??
        record.audioReactiveMappings
    )
  };
}

function normalizeAudioReactive(
  value: unknown
): AudioReactiveGuidanceSummary | null {
  const record = readRecord(value);
  if (!record) {
    return null;
  }

  const mappings = readRecordList(record.mappings).flatMap((mapping) => {
    const source = readAudioReactiveSource(mapping.source);
    const targets = readAudioReactiveTargets(mapping.targets);
    const behavior = readString(mapping.behavior);
    if (!source || targets.length === 0 || !behavior) {
      return [];
    }

    return [
      {
        source,
        targets,
        intensity: readAudioReactiveIntensity(mapping.intensity),
        behavior,
        evidence: readStringList(mapping.evidence)
      }
    ];
  });
  const summary = readString(record.summary);
  if (mappings.length === 0 || !summary) {
    return null;
  }

  return {
    mappings: mappings.slice(0, 6),
    audioRuntime: readString(
      record.audio_runtime ?? record.audioRuntime
    ),
    visualRuntime: readString(
      record.visual_runtime ?? record.visualRuntime
    ),
    activation: "explicit_user_gesture",
    summary
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

const audioReactiveSources = new Set<AudioReactiveSource>([
  "amplitude",
  "bass",
  "mids",
  "highs",
  "rhythm",
  "envelope",
  "drone_intensity"
]);

const audioReactiveTargets = new Set<AudioReactiveVisualTarget>([
  "scale",
  "glow",
  "brightness",
  "pulse",
  "expansion",
  "camera_movement",
  "color_shift",
  "texture_modulation",
  "sparkle",
  "particles",
  "detail",
  "rotation",
  "pattern_phase",
  "scene_transitions",
  "opacity",
  "bloom",
  "geometry_emergence",
  "fog",
  "aura",
  "field_density"
]);

function readVisualStyleList(value: unknown): VisualStyleName[] {
  return readStringList(value).filter(
    (item): item is VisualStyleName =>
      visualStyleNames.has(item as VisualStyleName)
  );
}

function readAudioReactiveSource(
  value: unknown
): AudioReactiveSource | null {
  return typeof value === "string" &&
    audioReactiveSources.has(value as AudioReactiveSource)
    ? (value as AudioReactiveSource)
    : null;
}

function readAudioReactiveTargets(
  value: unknown
): AudioReactiveVisualTarget[] {
  return readStringList(value).filter(
    (target): target is AudioReactiveVisualTarget =>
      audioReactiveTargets.has(target as AudioReactiveVisualTarget)
  );
}

function readAudioReactiveIntensity(
  value: unknown
): AudioReactiveGuidanceSummary["mappings"][number]["intensity"] {
  return value === "subtle" || value === "strong" ? value : "balanced";
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

function readRecordList(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value)
    ? value
        .map((item) => readRecord(item))
        .filter((item): item is Record<string, unknown> => item !== null)
    : [];
}

function readRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null
    ? (value as Record<string, unknown>)
    : null;
}

function readString(value: unknown) {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}
