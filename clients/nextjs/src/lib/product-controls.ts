import type { ArtifactSummary } from "./assistant-client";
import type { EvaluationBenchmarkRun } from "./evaluation-benchmark";

export type CreativityProfile = "controlled" | "balanced" | "exploratory";
export type FontScale = "small" | "medium" | "large";
export type FeedbackSentiment = "positive" | "negative";
export type PreferenceCategory =
  | "visual_density"
  | "palette"
  | "runtime"
  | "motion"
  | "interaction"
  | "export";

export type GenerationControls = {
  profile: CreativityProfile;
  requestedTemperature: number;
  providerParameterState: "pending_provider_confirmation";
  detail: string;
};

export type FeedbackSignal = {
  id: string;
  sentiment: FeedbackSentiment;
  comment: string | null;
  sessionId: string;
  artifactId: string | null;
  artifactTitle: string | null;
  domain: string | null;
  workflowMode: string;
  creativity: CreativityProfile;
  categories: PreferenceCategory[];
  createdAt: string;
  promptExcerpt: string | null;
  providerName: string | null;
  providerModel: string | null;
  requestedTemperature: number;
  effectiveTemperature: number | null;
  parameterApplication: "requested_not_confirmed" | "provider_reported";
  productOutcome: string | null;
};

export type PersonalizationContext = {
  enabled: boolean;
  signalCount: number;
  categories: PreferenceCategory[];
  selectedSignalIds: string[];
  detail: string;
};

export type PrivacyContractItem = {
  label: string;
  storage: string;
  boundary: string;
};

export type EvaluationHistoryRecord = {
  id: string;
  runId: string | null;
  datasetId: string | null;
  metrics: string[];
  status: string;
  detail: string;
  evaluatedAt: string;
  resultRows: number | null;
  metricFailures: number | null;
  dryRun: boolean | null;
  providerCallsAllowed: boolean | null;
  benchmark?: EvaluationBenchmarkRun | null;
};

const profileControls: Record<CreativityProfile, Omit<GenerationControls, "profile">> = {
  controlled: {
    requestedTemperature: 0.35,
    providerParameterState: "pending_provider_confirmation",
    detail: "Favor coherent, constrained variations."
  },
  balanced: {
    requestedTemperature: 0.7,
    providerParameterState: "pending_provider_confirmation",
    detail: "Balance distinct ideas with a reliable implementation path."
  },
  exploratory: {
    requestedTemperature: 1,
    providerParameterState: "pending_provider_confirmation",
    detail: "Favor broader visual and interaction variations within runtime boundaries."
  }
};

export const privacyContract: PrivacyContractItem[] = [
  {
    label: "Workspace sessions and artifacts",
    storage: "Local browser and local workspace persistence",
    boundary: "You can rename, delete, or clear them; no background sharing occurs."
  },
  {
    label: "Generation prompts and supported image references",
    storage: "Sent only for an explicit generation request",
    boundary: "Provider transmission depends on the configured provider; unsupported files are rejected."
  },
  {
    label: "Feedback and preference signals",
    storage: "Local profile persistence",
    boundary: "Used only when personalization is enabled; signals can be removed or cleared."
  },
  {
    label: "LangSmith traces and evaluation artifacts",
    storage: "Only when separately configured or explicitly run",
    boundary: "The workspace never silently uploads traces or starts an evaluation."
  },
  {
    label: "Knowledge Base updates",
    storage: "Local Chroma index",
    boundary: "Checks, downloads, rebuilds, and validation require an explicit operator action."
  }
];

export function buildGenerationControls(profile: CreativityProfile): GenerationControls {
  return { profile, ...profileControls[profile] };
}

export function createFeedbackSignal({
  artifact,
  comment,
  creativity,
  domain,
  id,
  promptExcerpt,
  providerName,
  providerModel,
  requestedTemperature,
  effectiveTemperature,
  parameterApplication,
  productOutcome,
  sentiment,
  sessionId,
  workflowMode,
  createdAt = new Date().toISOString()
}: {
  artifact: ArtifactSummary | null;
  comment?: string | null;
  creativity: CreativityProfile;
  domain?: string | null;
  id: string;
  promptExcerpt?: string | null;
  providerName?: string | null;
  providerModel?: string | null;
  requestedTemperature?: number;
  effectiveTemperature?: number | null;
  parameterApplication?: FeedbackSignal["parameterApplication"];
  productOutcome?: string | null;
  sentiment: FeedbackSentiment;
  sessionId: string;
  workflowMode: string;
  createdAt?: string;
}): FeedbackSignal {
  const normalizedComment = comment?.trim() || null;
  return {
    id,
    sentiment,
    comment: normalizedComment,
    sessionId,
    artifactId: artifact?.id ?? null,
    artifactTitle: artifact?.title ?? null,
    domain: domain ?? artifact?.domain ?? null,
    workflowMode,
    creativity,
    categories: inferPreferenceCategories(normalizedComment, artifact),
    createdAt,
    promptExcerpt: truncatePrompt(promptExcerpt),
    providerName: providerName ?? null,
    providerModel: providerModel ?? null,
    requestedTemperature: requestedTemperature ?? buildGenerationControls(creativity).requestedTemperature,
    effectiveTemperature: effectiveTemperature ?? null,
    parameterApplication: parameterApplication ?? "requested_not_confirmed",
    productOutcome: productOutcome ?? null
  };
}

function truncatePrompt(value: string | null | undefined) {
  const normalized = value?.trim() ?? "";
  if (!normalized) {
    return null;
  }
  return normalized.length <= 240 ? normalized : `${normalized.slice(0, 237)}…`;
}

export function selectPersonalizationContext({
  enabled,
  prompt,
  signals
}: {
  enabled: boolean;
  prompt: string;
  signals: FeedbackSignal[];
}): PersonalizationContext {
  if (!enabled) {
    return {
      enabled: false,
      signalCount: 0,
      categories: [],
      selectedSignalIds: [],
      detail: "Personalization is off; no stored preference signals are used."
    };
  }

  const promptTerms = new Set(tokenize(prompt));
  const selected = [...signals]
    .sort((left, right) => right.createdAt.localeCompare(left.createdAt))
    .sort((left, right) => relevance(right, promptTerms) - relevance(left, promptTerms))
    .filter((signal) => relevance(signal, promptTerms) > 0)
    .slice(0, 3);
  const categories = [...new Set(selected.flatMap((signal) => signal.categories))];

  return {
    enabled: true,
    signalCount: selected.length,
    categories,
    selectedSignalIds: selected.map((signal) => signal.id),
    detail:
      selected.length === 0
        ? "No relevant preference signals are stored for this request."
        : `${selected.length} recent, relevant preference signal${selected.length === 1 ? "" : "s"} will inform this request.`
  };
}

function inferPreferenceCategories(
  comment: string | null,
  artifact: ArtifactSummary | null
): PreferenceCategory[] {
  const terms = tokenize([comment ?? "", artifact?.summary ?? "", artifact?.domain ?? ""].join(" "));
  const categories = new Set<PreferenceCategory>();
  if (terms.some((term) => ["dense", "sparse", "minimal", "detail"].includes(term))) {
    categories.add("visual_density");
  }
  if (terms.some((term) => ["palette", "colour", "color", "hue"].includes(term))) {
    categories.add("palette");
  }
  if (terms.some((term) => ["p5", "three", "glsl", "tone", "runtime"].includes(term))) {
    categories.add("runtime");
  }
  if (terms.some((term) => ["motion", "animate", "animation", "speed"].includes(term))) {
    categories.add("motion");
  }
  if (terms.some((term) => ["interact", "mouse", "keyboard", "click"].includes(term))) {
    categories.add("interaction");
  }
  if (terms.some((term) => ["export", "download", "handoff"].includes(term))) {
    categories.add("export");
  }
  return [...categories];
}

function relevance(signal: FeedbackSignal, promptTerms: Set<string>) {
  const signalTerms = tokenize(
    [signal.comment ?? "", signal.domain ?? "", signal.categories.join(" ")].join(" ")
  );
  return signalTerms.reduce((score, term) => score + Number(promptTerms.has(term)), 0);
}

function tokenize(value: string) {
  return value.toLowerCase().match(/[a-z0-9_]+/g) ?? [];
}
