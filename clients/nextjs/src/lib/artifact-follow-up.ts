import type { ArtifactSummary } from "./assistant-client";
import { canRunRefinementPass } from "./refinement-passes";

export type ArtifactFollowUpResolution =
  | {
      kind: "refinement";
      artifact: ArtifactSummary;
      confidence: "high";
    }
  | {
      kind: "none";
      artifact: null;
      confidence: "low";
    };

const newArtifactRequest = /^(?:please\s+)?(?:(?:can|could|would)\s+you\s+)?(?:create|build|generate|make|design|compose|prototype)\s+(?:me\s+)?(?:a|an|another|new|separate)\b/i;
const directArtifactModification = /^(?:please\s+)?(?:(?:can|could|would)\s+you\s+)?(?:(?:make|turn)\s+(?:it|this|that|the\s+(?:artifact|output|result|sketch|shader|scene|code))\b|(?:adjust|add|brighten|change|convert|darken|decrease|enhance|fix|improve|increase|optimize|reduce|refine|remove|simplify|slow|speed|tweak)\b)/i;
const compactComparativeModification = /^(?:please\s+)?(?:a\s+(?:bit|little)\s+|slightly\s+|much\s+|even\s+)?(?:brighter|darker|faster|slower|calmer|bolder|softer|cleaner|simpler|warmer|cooler|more\s+[a-z-]+|less\s+[a-z-]+)[.!]?$/i;

export function resolveArtifactFollowUp({
  activeArtifact,
  artifacts,
  prompt
}: {
  activeArtifact: ArtifactSummary | null;
  artifacts: readonly ArtifactSummary[];
  prompt: string;
}): ArtifactFollowUpResolution {
  const normalizedPrompt = prompt.trim().replace(/\s+/g, " ");
  const targetArtifact = isRefinableArtifact(activeArtifact)
    ? activeArtifact
    : artifacts.find(isRefinableArtifact) ?? null;

  if (
    !targetArtifact ||
    normalizedPrompt.length < 3 ||
    normalizedPrompt.length > 220 ||
    newArtifactRequest.test(normalizedPrompt)
  ) {
    return noArtifactFollowUp;
  }

  if (
    directArtifactModification.test(normalizedPrompt) ||
    compactComparativeModification.test(normalizedPrompt)
  ) {
    return {
      artifact: targetArtifact,
      confidence: "high",
      kind: "refinement"
    };
  }

  return noArtifactFollowUp;
}

function isRefinableArtifact(
  artifact: ArtifactSummary | null | undefined
): artifact is ArtifactSummary {
  return Boolean(
    artifact &&
      artifact.type === "code" &&
      artifact.id &&
      artifact.title &&
      canRunRefinementPass(artifact)
  );
}

const noArtifactFollowUp: ArtifactFollowUpResolution = {
  artifact: null,
  confidence: "low",
  kind: "none"
};
