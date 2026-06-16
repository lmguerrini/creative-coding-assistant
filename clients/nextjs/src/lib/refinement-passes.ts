import type {
  ArtifactSummary,
  RefinementPassRecord,
  RefinementPassStopReason
} from "./assistant-client";
import type { AssistantArtifactRefinementRequest } from "./assistant-stream";

export const defaultRefinementPassLimit = 2;
export const maxRefinementPassLimit = 3;
export const qualityImprovementThreshold = 0.04;

export function normalizeRefinementPassLimit(value?: number | null) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return defaultRefinementPassLimit;
  }

  return Math.min(Math.max(Math.round(value), 1), maxRefinementPassLimit);
}

export function readArtifactQuality(artifact: ArtifactSummary) {
  return (
    artifact.critique?.calibratedQuality?.score ??
    artifact.qualityScore ??
    artifact.critique?.overallScore ??
    null
  );
}

export function readCompletedRefinementPasses(artifact: ArtifactSummary) {
  return artifact.refinementPasses ?? [];
}

export function nextRefinementPassNumber(artifact: ArtifactSummary) {
  return readCompletedRefinementPasses(artifact).length + 1;
}

export function canRunRefinementPass(artifact: ArtifactSummary) {
  const history = readCompletedRefinementPasses(artifact);
  const latest = history.at(-1);

  return (
    nextRefinementPassNumber(artifact) <= defaultRefinementPassLimit &&
    (!latest || latest.stopReason === "continue_available")
  );
}

export function refinementStoppedBeforeLimit(artifact: ArtifactSummary) {
  const latest = readCompletedRefinementPasses(artifact).at(-1);
  return Boolean(
    latest &&
      latest.stopReason !== "continue_available" &&
      readCompletedRefinementPasses(artifact).length < defaultRefinementPassLimit
  );
}

export function collectRefinementOpportunities(
  artifact: ArtifactSummary,
  limit = 6
) {
  const opportunities: string[] = [];
  const critique = artifact.critique;
  const translation = artifact.creativeTranslation;

  appendUnique(opportunities, critique?.refinementGuidance);
  appendList(opportunities, critique?.creativeEvaluation?.refinementOpportunities);
  appendList(opportunities, critique?.sacredConsistency?.refinementOpportunities);
  appendList(opportunities, critique?.calibratedQuality?.adjustments);
  appendList(opportunities, critique?.reasons);
  appendList(opportunities, translation?.refinementTargets);

  for (const mapping of translation?.audioReactive?.mappings ?? []) {
    appendUnique(
      opportunities,
      `Preserve audio-reactive ${mapping.source} mapping to ${mapping.targets.join(", ")}.`
    );
  }

  return opportunities.slice(0, limit);
}

export function buildRefinementObjective({
  artifact,
  instruction
}: {
  artifact: ArtifactSummary;
  instruction: string;
}) {
  const opportunities = collectRefinementOpportunities(artifact, 4);
  const trimmedInstruction = instruction.trim();
  const explicitInstruction = trimmedInstruction
    ? `Apply the user refinement: ${trimmedInstruction}`
    : "Create a targeted refined version of the selected artifact.";

  if (opportunities.length === 0) {
    return `${explicitInstruction} Preserve source lineage, runtime, and preview compatibility.`;
  }

  return [
    explicitInstruction,
    `Use existing quality signals: ${opportunities.join(" ")}`
  ].join(" ");
}

export function enrichArtifactRefinementRequest(
  request: AssistantArtifactRefinementRequest,
  artifact: ArtifactSummary
): AssistantArtifactRefinementRequest {
  const passNumber = nextRefinementPassNumber(artifact);
  const maxPasses = defaultRefinementPassLimit;
  const qualityBefore = readArtifactQuality(artifact);

  return {
    ...request,
    qualityBefore,
    passNumber,
    maxPasses,
    refinementObjective: buildRefinementObjective({
      artifact,
      instruction: request.instruction
    }),
    refinementPasses: readCompletedRefinementPasses(artifact)
  };
}

export function appendRefinementPassRecord({
  refinement,
  resultArtifact,
  sourceArtifact
}: {
  refinement: AssistantArtifactRefinementRequest;
  resultArtifact: ArtifactSummary;
  sourceArtifact: ArtifactSummary | null;
}): RefinementPassRecord[] {
  const previousPasses = sourceArtifact?.refinementPasses ?? refinement.refinementPasses ?? [];
  const qualityBefore =
    refinement.qualityBefore ??
    (sourceArtifact ? readArtifactQuality(sourceArtifact) : null);
  const qualityAfter = readArtifactQuality(resultArtifact);
  const passNumber = refinement.passNumber ?? previousPasses.length + 1;
  const maxPasses = normalizeRefinementPassLimit(refinement.maxPasses);
  const stopReason = determineStopReason({
    maxPasses,
    passNumber,
    qualityAfter,
    qualityBefore,
    resultArtifact,
    sourceArtifact
  });

  return [
    ...previousPasses,
    {
      passNumber,
      sourceArtifactId: refinement.artifactId,
      sourceArtifactTitle: refinement.title,
      resultArtifactId: resultArtifact.id,
      resultArtifactTitle: resultArtifact.title,
      refinementObjective:
        refinement.refinementObjective ??
        buildRefinementObjective({
          artifact: sourceArtifact ?? resultArtifact,
          instruction: refinement.instruction
        }),
      qualityBefore,
      qualityAfter,
      stopReason,
      summary: buildPassSummary({
        passNumber,
        qualityAfter,
        qualityBefore,
        stopReason
      })
    }
  ];
}

export function formatRefinementStopReason(reason: RefinementPassStopReason) {
  switch (reason) {
    case "continue_available":
      return "Next pass available";
    case "quality_improved":
      return "Quality improved";
    case "no_useful_opportunities":
      return "No useful opportunities";
    case "runtime_preview_safety_failed":
      return "Runtime safety stopped";
    case "max_passes_reached":
      return "Max passes reached";
  }
}

function determineStopReason({
  maxPasses,
  passNumber,
  qualityAfter,
  qualityBefore,
  resultArtifact,
  sourceArtifact
}: {
  maxPasses: number;
  passNumber: number;
  qualityAfter: number | null;
  qualityBefore: number | null;
  resultArtifact: ArtifactSummary;
  sourceArtifact: ArtifactSummary | null;
}): RefinementPassStopReason {
  if (
    qualityBefore !== null &&
    qualityAfter !== null &&
    qualityAfter - qualityBefore >= qualityImprovementThreshold
  ) {
    return "quality_improved";
  }

  if (runtimePreviewSafetyFailed(resultArtifact)) {
    return "runtime_preview_safety_failed";
  }

  if (
    collectRefinementOpportunities(resultArtifact).length === 0 &&
    collectRefinementOpportunities(sourceArtifact ?? resultArtifact).length === 0
  ) {
    return "no_useful_opportunities";
  }

  if (passNumber >= maxPasses) {
    return "max_passes_reached";
  }

  return "continue_available";
}

function runtimePreviewSafetyFailed(artifact: ArtifactSummary) {
  const critique = artifact.critique;
  return Boolean(
    critique &&
      critique.runtimeSuitability.score < 0.5 &&
      critique.previewReadiness.score < 0.5
  );
}

function buildPassSummary({
  passNumber,
  qualityAfter,
  qualityBefore,
  stopReason
}: {
  passNumber: number;
  qualityAfter: number | null;
  qualityBefore: number | null;
  stopReason: RefinementPassStopReason;
}) {
  const quality =
    qualityBefore !== null && qualityAfter !== null
      ? ` Quality ${qualityBefore.toFixed(2)} -> ${qualityAfter.toFixed(2)}.`
      : "";

  return `Pass ${passNumber}: ${formatRefinementStopReason(stopReason)}.${quality}`;
}

function appendList(items: string[], values?: string[] | null) {
  for (const value of values ?? []) {
    appendUnique(items, value);
  }
}

function appendUnique(items: string[], value?: string | null) {
  const normalized = value?.trim();
  if (normalized && !items.includes(normalized)) {
    items.push(normalized);
  }
}
