export type AssistantRequestMode = "explain" | "generate";

type ResolveAssistantRequestModeOptions = {
  hasArtifactRefinement?: boolean;
  hasClarificationResponse?: boolean;
  prompt: string;
};

const directGenerationPattern =
  /^(?:(?:please|kindly|per favore)\s+)?(?:animate|build|change|code|compose|create|design|draw|fix|generate|implement|improve|make|modify|produce|refine|render|update|write|anima|cambia|crea|disegna|genera|implementa|migliora|modifica|produci|raffina|renderizza|scrivi)\b/i;
const delegatedGenerationPattern =
  /^(?:can|could|would|will)\s+you\s+(?:(?:please|kindly)\s+)?(?:animate|build|change|code|compose|create|design|draw|fix|generate|implement|improve|make|modify|produce|refine|render|update|write)\b/i;
const explanationPattern =
  /^(?:can|could|do|does|did|how|is|are|should|what|what's|when|where|which|who|why|would|will)\b|^(?:can|could|would)\s+you\s+(?:define|describe|explain|tell)\b|^(?:define|describe|explain|tell\s+me|compare)\b|^i\s+(?:want|would\s+like)\s+to\s+(?:know|understand)\b|^(?:che\s+cos(?:a|['’]\s*è)|cos(?:a|['’]\s*è)|come|dove|perch(?:é|e)|qual(?:e|['’]\s*è)|quali|quando|chi)\b|^(?:definisci|descrivi|spiega|confronta)\b/i;

export function resolveAssistantRequestMode({
  hasArtifactRefinement = false,
  hasClarificationResponse = false,
  prompt
}: ResolveAssistantRequestModeOptions): AssistantRequestMode {
  if (hasArtifactRefinement || hasClarificationResponse) {
    return "generate";
  }

  const normalizedPrompt = prompt.trim();
  if (
    directGenerationPattern.test(normalizedPrompt) ||
    delegatedGenerationPattern.test(normalizedPrompt)
  ) {
    return "generate";
  }

  if (explanationPattern.test(normalizedPrompt) || normalizedPrompt.endsWith("?")) {
    return "explain";
  }

  return "generate";
}
