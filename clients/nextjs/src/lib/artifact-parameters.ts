import type { ArtifactSummary } from "./assistant-client";

export type ArtifactParameterType =
  | "range"
  | "number"
  | "boolean"
  | "enum"
  | "color"
  | "readonly";

export type ArtifactParameterValue = string | number | boolean;

export type ArtifactParameterEffect =
  | "preview_only"
  | "refinement_guidance";

export type ArtifactParameterSource =
  | "runtime"
  | "modality"
  | "creative_translation"
  | "reference_fusion"
  | "sacred_geometry"
  | "shader_preset"
  | "visual_style"
  | "code_hint";

export type ArtifactParameterOption = {
  label: string;
  value: string;
};

export type ArtifactParameterDefinition = {
  id: string;
  label: string;
  type: ArtifactParameterType;
  description: string;
  defaultValue: ArtifactParameterValue;
  effect: ArtifactParameterEffect;
  source: ArtifactParameterSource;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  options?: ArtifactParameterOption[];
};

export type ArtifactParameterModel = {
  artifactId: string;
  artifactTitle: string;
  parameters: ArtifactParameterDefinition[];
  status: "available" | "unsupported";
  summary: string;
};

export type ArtifactParameterValues = Record<
  string,
  ArtifactParameterValue
>;

export type ArtifactParameterChange = {
  id: string;
  label: string;
  previousValue: ArtifactParameterValue;
  value: ArtifactParameterValue;
  unit: string | null;
};

export type ArtifactParameterGuidance = {
  artifactId: string;
  artifactTitle: string;
  changes: ArtifactParameterChange[];
  instruction: string;
};

const maxDerivedParameters = 8;

const visualRuntimes = new Set([
  "p5",
  "three",
  "glsl",
  "hydra",
  "react_three_fiber"
]);

const supportedRuntimes = new Set([...visualRuntimes, "tone"]);

const colorSignals = [
  { tokens: ["cyan", "aqua", "teal", "bioluminescent"], value: "#45d8c8" },
  { tokens: ["blue", "azure", "ocean"], value: "#5b8cff" },
  { tokens: ["violet", "purple", "magenta"], value: "#b26cff" },
  { tokens: ["amber", "gold", "warm"], value: "#f0b85a" },
  { tokens: ["red", "coral"], value: "#ff7668" },
  { tokens: ["green", "lime", "forest"], value: "#79d38b" }
] as const;

export function deriveArtifactParameterModel(
  artifact: ArtifactSummary
): ArtifactParameterModel {
  const runtime = normalizeRuntime(artifact);
  const modality = inferOutputModality(artifact, runtime);
  const metadataText = collectMetadataText(artifact);
  const codeHints = collectSafeCodeHints(artifact.content);
  const hasStructuredMetadata = Boolean(artifact.creativeTranslation);
  const hasSafeSignals =
    supportedRuntimes.has(runtime) ||
    hasStructuredMetadata ||
    codeHints.size > 0;

  if (artifact.type !== "code" || !hasSafeSignals) {
    return {
      artifactId: artifact.id,
      artifactTitle: artifact.title,
      parameters: [],
      status: "unsupported",
      summary:
        "No bounded runtime or creative metadata is available for safe parameter derivation."
    };
  }

  const parameters: ArtifactParameterDefinition[] = [
    readonlyParameter(
      "runtime",
      "Runtime",
      formatRuntimeLabel(runtime, artifact),
      "Runtime identity used for deterministic parameter selection.",
      "runtime"
    ),
    readonlyParameter(
      "modality",
      "Output",
      formatModalityLabel(modality),
      "Output modality inferred from artifact metadata and runtime.",
      "modality"
    )
  ];

  if (modality === "visual" || modality === "audiovisual") {
    parameters.push(
      {
        id: "accent_color",
        label: "Accent color",
        type: "color",
        description:
          "Primary generated accent used as refinement guidance for palette changes.",
        defaultValue: inferAccentColor(metadataText),
        effect: "refinement_guidance",
        source: resolveColorSource(artifact)
      },
      {
        id: "movement_complexity",
        label: "Movement complexity",
        type: "range",
        description:
          "Controls how layered or restrained the requested motion should feel.",
        defaultValue: inferMovementComplexity(metadataText),
        effect: "refinement_guidance",
        source: artifact.creativeTranslation
          ? "creative_translation"
          : "runtime",
        min: 1,
        max: 10,
        step: 1
      }
    );
  }

  appendRuntimeParameters(parameters, {
    artifact,
    codeHints,
    metadataText,
    modality,
    runtime
  });
  appendStructuredMetadataParameters(parameters, artifact, metadataText);

  return {
    artifactId: artifact.id,
    artifactTitle: artifact.title,
    parameters: dedupeParameters(parameters).slice(0, maxDerivedParameters),
    status: "available",
    summary:
      "Controls are derived from bounded metadata and known runtime signals. Changes remain local until refinement is submitted."
  };
}

export function createArtifactParameterValues(
  model: ArtifactParameterModel
): ArtifactParameterValues {
  return Object.fromEntries(
    model.parameters.map((parameter) => [
      parameter.id,
      parameter.defaultValue
    ])
  );
}

export function updateArtifactParameterValue(
  model: ArtifactParameterModel,
  values: ArtifactParameterValues,
  parameterId: string,
  nextValue: unknown
): ArtifactParameterValues {
  const parameter = model.parameters.find(
    (candidate) => candidate.id === parameterId
  );
  if (!parameter || parameter.type === "readonly") {
    return values;
  }

  return {
    ...values,
    [parameterId]: normalizeArtifactParameterValue(parameter, nextValue)
  };
}

export function normalizeArtifactParameterValue(
  parameter: ArtifactParameterDefinition,
  value: unknown
): ArtifactParameterValue {
  if (parameter.type === "boolean") {
    return value === true || value === "true";
  }

  if (parameter.type === "enum") {
    return parameter.options?.some((option) => option.value === value)
      ? String(value)
      : parameter.defaultValue;
  }

  if (parameter.type === "color") {
    return typeof value === "string" && /^#[0-9a-f]{6}$/i.test(value)
      ? value.toLowerCase()
      : parameter.defaultValue;
  }

  if (parameter.type === "range" || parameter.type === "number") {
    const parsedValue =
      typeof value === "number" ? value : Number.parseFloat(String(value));
    if (!Number.isFinite(parsedValue)) {
      return parameter.defaultValue;
    }

    const min = parameter.min ?? parsedValue;
    const max = parameter.max ?? parsedValue;
    const boundedValue = Math.min(max, Math.max(min, parsedValue));
    const step = parameter.step ?? 1;
    const steppedValue =
      Math.round((boundedValue - min) / step) * step + min;

    return Number(steppedValue.toFixed(4));
  }

  return parameter.defaultValue;
}

export function getArtifactParameterChanges(
  model: ArtifactParameterModel,
  values: ArtifactParameterValues
): ArtifactParameterChange[] {
  return model.parameters.flatMap((parameter) => {
    if (
      parameter.type === "readonly" ||
      parameter.effect !== "refinement_guidance"
    ) {
      return [];
    }

    const value = normalizeArtifactParameterValue(
      parameter,
      values[parameter.id]
    );
    if (value === parameter.defaultValue) {
      return [];
    }

    return [
      {
        id: parameter.id,
        label: parameter.label,
        previousValue: parameter.defaultValue,
        value,
        unit: parameter.unit ?? null
      }
    ];
  });
}

export function serializeArtifactParameterGuidance(
  model: ArtifactParameterModel,
  values: ArtifactParameterValues
): ArtifactParameterGuidance | null {
  const changes = getArtifactParameterChanges(model, values);
  if (changes.length === 0) {
    return null;
  }

  const changeLines = changes.map(
    (change) =>
      `- ${change.label}: ${formatParameterValue(
        change.value,
        change.unit
      )} (derived default: ${formatParameterValue(
        change.previousValue,
        change.unit
      )})`
  );

  return {
    artifactId: model.artifactId,
    artifactTitle: model.artifactTitle,
    changes,
    instruction: [
      `Apply these bounded parameter changes to ${model.artifactTitle}:`,
      ...changeLines,
      "Treat these values as refinement guidance. Preserve unrelated behavior and do not assume the source or preview was already mutated."
    ].join("\n")
  };
}

export function buildArtifactRefinementInstruction({
  guidance,
  instruction
}: {
  guidance: ArtifactParameterGuidance | null;
  instruction: string;
}) {
  const trimmedInstruction = instruction.trim();
  if (!guidance) {
    return trimmedInstruction;
  }

  return [
    trimmedInstruction || "Apply the selected artifact parameter changes.",
    guidance.instruction
  ].join("\n\n");
}

function appendRuntimeParameters(
  parameters: ArtifactParameterDefinition[],
  {
    artifact,
    codeHints,
    metadataText,
    modality,
    runtime
  }: {
    artifact: ArtifactSummary;
    codeHints: Set<string>;
    metadataText: string;
    modality: "visual" | "audio" | "audiovisual" | "code";
    runtime: string;
  }
) {
  if (modality === "audiovisual") {
    parameters.push({
      id: "audio_reactivity",
      label: "Audio reactivity",
      type: "boolean",
      description:
        "Requests visual response to audio analysis in a future refinement.",
      defaultValue:
        Boolean(artifact.creativeTranslation?.audioReactive) ||
        inferAudioReactivity(metadataText),
      effect: "refinement_guidance",
      source: "modality"
    });
  }

  if (visualRuntimes.has(runtime) || codeHints.has("rotation")) {
    parameters.push({
      id: "rotation_speed",
      label: "Rotation speed",
      type: "range",
      description:
        "Requested rotational pace for generated geometry or camera motion.",
      defaultValue: inferRotationSpeed(metadataText),
      effect: "refinement_guidance",
      source: codeHints.has("rotation") ? "code_hint" : "runtime",
      min: 0,
      max: 2,
      step: 0.05,
      unit: "x"
    });
  }

  if (
    runtime === "three" ||
    runtime === "react_three_fiber" ||
    codeHints.has("fog")
  ) {
    parameters.push({
      id: "fog_density",
      label: "Fog density",
      type: "range",
      description: "Atmospheric depth requested for the scene refinement.",
      defaultValue: 0.035,
      effect: "refinement_guidance",
      source: codeHints.has("fog") ? "code_hint" : "runtime",
      min: 0,
      max: 0.15,
      step: 0.005
    });
  }

  if (
    runtime === "glsl" ||
    hasShaderPreset(artifact, ["glow", "bloom"]) ||
    codeHints.has("bloom")
  ) {
    parameters.push({
      id: "bloom_intensity",
      label: "Bloom intensity",
      type: "range",
      description:
        "Requested glow contribution for compatible shader or scene refinements.",
      defaultValue: hasShaderPreset(artifact, ["glow", "bloom"]) ? 0.8 : 0.45,
      effect: "refinement_guidance",
      source: hasShaderPreset(artifact, ["glow", "bloom"])
        ? "shader_preset"
        : codeHints.has("bloom")
          ? "code_hint"
          : "runtime",
      min: 0,
      max: 2,
      step: 0.05,
      unit: "x"
    });
  }

  if (runtime === "hydra" || codeHints.has("feedback")) {
    parameters.push({
      id: "feedback_mix",
      label: "Feedback mix",
      type: "range",
      description:
        "Requested amount of bounded visual feedback in the Hydra composition.",
      defaultValue: 0.55,
      effect: "refinement_guidance",
      source: codeHints.has("feedback") ? "code_hint" : "runtime",
      min: 0,
      max: 0.95,
      step: 0.05
    });
  }

  if (runtime === "tone" || modality === "audio" || modality === "audiovisual") {
    parameters.push(
      {
        id: "rhythm_density",
        label: "Rhythm density",
        type: "range",
        description:
          "Requested number of rhythmic events per phrase for the next refinement.",
        defaultValue: inferRhythmDensity(metadataText),
        effect: "refinement_guidance",
        source: runtime === "tone" ? "runtime" : "modality",
        min: 1,
        max: 16,
        step: 1,
        unit: "events"
      },
      {
        id: "drone_intensity",
        label: "Drone intensity",
        type: "range",
        description:
          "Requested prominence of sustained tonal material without starting audio.",
        defaultValue: inferDroneIntensity(metadataText),
        effect: "refinement_guidance",
        source: runtime === "tone" ? "runtime" : "modality",
        min: 0,
        max: 1,
        step: 0.05
      }
    );
  }

}

function appendStructuredMetadataParameters(
  parameters: ArtifactParameterDefinition[],
  artifact: ArtifactSummary,
  metadataText: string
) {
  const sacredGeometry = artifact.creativeTranslation?.sacredGeometry;
  if (
    sacredGeometry &&
    (sacredGeometry.concepts.length > 0 ||
      sacredGeometry.symmetryType.length > 0)
  ) {
    parameters.push({
      id: "symmetry",
      label: "Symmetry",
      type: "enum",
      description:
        "Requested geometric symmetry mode derived from sacred geometry guidance.",
      defaultValue: inferSymmetry(metadataText),
      effect: "refinement_guidance",
      source: "sacred_geometry",
      options: [
        { label: "Radial", value: "radial" },
        { label: "Rotational", value: "rotational" },
        { label: "Bilateral", value: "bilateral" },
        { label: "Tessellated", value: "tessellated" }
      ]
    });
  }

  const visualStyle = artifact.creativeTranslation?.visualStyle;
  if (visualStyle && visualStyle.styles.length > 0) {
    parameters.push({
      id: "palette_mode",
      label: "Palette mode",
      type: "enum",
      description:
        "Refinement palette strategy derived from the selected visual style.",
      defaultValue: inferPaletteMode(metadataText),
      effect: "refinement_guidance",
      source: "visual_style",
      options: [
        { label: "Source palette", value: "source" },
        { label: "Monochrome", value: "monochrome" },
        { label: "Analogous", value: "analogous" },
        { label: "Complementary", value: "complementary" }
      ]
    });
  }

  if (
    artifact.creativeTranslation &&
    (artifact.creativeTranslation.structureDirection.length > 0 ||
      artifact.creativeTranslation.geometricReferences.length > 0)
  ) {
    parameters.push({
      id: "scale",
      label: "Composition scale",
      type: "number",
      description:
        "Relative scale requested for the primary generated composition.",
      defaultValue: 1,
      effect: "refinement_guidance",
      source: "creative_translation",
      min: 0.5,
      max: 2,
      step: 0.1,
      unit: "x"
    });
  }
}

function readonlyParameter(
  id: string,
  label: string,
  value: string,
  description: string,
  source: ArtifactParameterSource
): ArtifactParameterDefinition {
  return {
    id,
    label,
    type: "readonly",
    description,
    defaultValue: value,
    effect: "refinement_guidance",
    source
  };
}

function normalizeRuntime(artifact: ArtifactSummary) {
  const value = (artifact.runtime ?? artifact.domain ?? "").toLowerCase();
  if (value.includes("react_three") || value.includes("r3f")) {
    return "react_three_fiber";
  }
  if (value.includes("three")) {
    return "three";
  }
  if (value.includes("tone")) {
    return "tone";
  }
  if (value.includes("hydra")) {
    return "hydra";
  }
  if (value.includes("glsl") || value.includes("shader")) {
    return "glsl";
  }
  if (value.includes("p5")) {
    return "p5";
  }
  return value || "unknown";
}

function inferOutputModality(
  artifact: ArtifactSummary,
  runtime: string
): "visual" | "audio" | "audiovisual" | "code" {
  const modality = artifact.creativeTranslation?.outputModality;
  if (modality) {
    return modality;
  }
  if (runtime === "tone") {
    return "audio";
  }
  if (visualRuntimes.has(runtime)) {
    return "visual";
  }
  return artifact.previewEligible ? "visual" : "code";
}

function collectMetadataText(artifact: ArtifactSummary) {
  const translation = artifact.creativeTranslation;
  if (!translation) {
    return `${artifact.title} ${artifact.summary}`.toLowerCase();
  }

  return [
    artifact.title,
    artifact.summary,
    translation.creativeIntent,
    ...translation.moodAtmosphere,
    ...translation.movementLanguage,
    ...translation.colorMaterialDirection,
    ...translation.musicalReferences,
    ...(translation.sacredGeometry?.concepts ?? []),
    ...(translation.sacredGeometry?.symmetryType ?? []),
    ...(translation.sacredGeometry?.audioImplications ?? []),
    ...(translation.shaderPresets?.presets ?? []),
    ...(translation.visualStyle?.styles ?? []),
    ...(translation.visualStyle?.paletteBehavior ?? []),
    ...(translation.referenceFusion?.paletteDirection ?? []),
    ...(translation.referenceFusion?.composition ?? []),
    ...(translation.referenceFusion?.lightingContrast ?? []),
    ...(translation.referenceFusion?.textureMaterialCues ?? []),
    ...(translation.referenceFusion?.geometricStructure ?? []),
    ...(translation.referenceFusion?.moodAtmosphere ?? []),
    ...(translation.referenceFusion?.motionImplications ?? []),
    ...(translation.referenceFusion?.runtimeStyleImplications ?? [])
  ]
    .join(" ")
    .toLowerCase();
}

function collectSafeCodeHints(content?: string) {
  const normalized = (content ?? "").toLowerCase();
  const hints = new Set<string>();
  const knownSignals = {
    bloom: ["bloom", "glow"],
    feedback: ["feedback", "modulate", ".out("],
    fog: ["fog", "fogexp2"],
    rotation: ["rotate", "rotation", "angular"]
  } as const;

  for (const [hint, tokens] of Object.entries(knownSignals)) {
    if (tokens.some((token) => normalized.includes(token))) {
      hints.add(hint);
    }
  }

  return hints;
}

function inferAccentColor(metadataText: string) {
  return (
    colorSignals.find((signal) =>
      signal.tokens.some((token) => metadataText.includes(token))
    )?.value ?? "#45d8c8"
  );
}

function inferMovementComplexity(metadataText: string) {
  if (includesAny(metadataText, ["chaotic", "dense", "energetic", "complex"])) {
    return 8;
  }
  if (includesAny(metadataText, ["calm", "slow", "minimal", "restrained"])) {
    return 3;
  }
  return 5;
}

function inferRotationSpeed(metadataText: string) {
  if (includesAny(metadataText, ["fast", "rapid", "energetic"])) {
    return 1.2;
  }
  if (includesAny(metadataText, ["slow", "calm", "meditative"])) {
    return 0.3;
  }
  return 0.65;
}

function inferRhythmDensity(metadataText: string) {
  if (includesAny(metadataText, ["dense", "polyrhythm", "rapid"])) {
    return 12;
  }
  if (includesAny(metadataText, ["drone", "ambient", "slow", "sparse"])) {
    return 4;
  }
  return 8;
}

function inferDroneIntensity(metadataText: string) {
  if (includesAny(metadataText, ["drone", "sustained", "ambient"])) {
    return 0.7;
  }
  return 0.25;
}

function inferAudioReactivity(metadataText: string) {
  return includesAny(metadataText, [
    "audio reactive",
    "audio-reactive",
    "sound reactive",
    "music reactive"
  ]);
}

function inferSymmetry(metadataText: string) {
  if (metadataText.includes("tessellat")) {
    return "tessellated";
  }
  if (metadataText.includes("bilateral")) {
    return "bilateral";
  }
  if (metadataText.includes("rotational")) {
    return "rotational";
  }
  return "radial";
}

function inferPaletteMode(metadataText: string) {
  if (metadataText.includes("monochrom")) {
    return "monochrome";
  }
  if (metadataText.includes("complement")) {
    return "complementary";
  }
  if (metadataText.includes("analogous")) {
    return "analogous";
  }
  return "source";
}

function resolveColorSource(
  artifact: ArtifactSummary
): ArtifactParameterSource {
  if (artifact.creativeTranslation?.visualStyle) {
    return "visual_style";
  }
  if (artifact.creativeTranslation?.referenceFusion) {
    return "reference_fusion";
  }
  if (artifact.creativeTranslation) {
    return "creative_translation";
  }
  return "runtime";
}

function hasShaderPreset(artifact: ArtifactSummary, presets: string[]) {
  return (
    artifact.creativeTranslation?.shaderPresets?.presets.some((preset) =>
      presets.some((candidate) =>
        preset.toLowerCase().includes(candidate.toLowerCase())
      )
    ) ?? false
  );
}

function dedupeParameters(parameters: ArtifactParameterDefinition[]) {
  const seen = new Set<string>();
  return parameters.filter((parameter) => {
    if (seen.has(parameter.id)) {
      return false;
    }
    seen.add(parameter.id);
    return true;
  });
}

function formatRuntimeLabel(runtime: string, artifact: ArtifactSummary) {
  if (runtime === "unknown") {
    return artifact.language || "Unknown";
  }
  if (runtime === "react_three_fiber") {
    return "React Three Fiber";
  }
  if (runtime === "p5") {
    return "p5.js";
  }
  if (runtime === "tone") {
    return "Tone.js";
  }
  if (runtime === "three") {
    return "Three.js";
  }
  if (runtime === "glsl") {
    return "GLSL";
  }
  if (runtime === "hydra") {
    return "Hydra";
  }
  if (runtime === "gsap") {
    return "GSAP";
  }
  return runtime;
}

function formatModalityLabel(
  modality: "visual" | "audio" | "audiovisual" | "code"
) {
  if (modality === "audiovisual") {
    return "Audiovisual";
  }
  if (modality === "audio") {
    return "Audio";
  }
  if (modality === "visual") {
    return "Visual";
  }
  return "Code-only";
}

function formatParameterValue(
  value: ArtifactParameterValue,
  unit: string | null
) {
  const formattedValue =
    typeof value === "boolean" ? (value ? "enabled" : "disabled") : String(value);
  return unit ? `${formattedValue} ${unit}` : formattedValue;
}

function includesAny(value: string, tokens: string[]) {
  return tokens.some((token) => value.includes(token));
}
