import { demoModeScenarioCatalog } from "./demo-mode";
import {
  domainStarterPromptLibrary,
  homepagePromptLibrary,
  morphogenesisPromptLibrary,
  rhythmicLineStudyPrompt,
  type CuratedPrompt
} from "./curated-prompt-library";
import type { ProductIntelligenceModel } from "./product-intelligence";

export const GOLDEN_EVALUATION_DATASET_VERSION = "golden_eval.v1";
export const EVALUATION_SCHEMA_VERSION = 2 as const;
export const EVALUATION_TARGET_THRESHOLD = 0.8;

export type EvaluationCategory =
  | "rag"
  | "creative_artifact"
  | "workflow"
  | "product_reliability";

export type EvaluationScope =
  | "full"
  | "rag"
  | "creative_artifact"
  | "workflow"
  | "product_reliability"
  | "cases";

export type EvaluationMetricStatus =
  | "pass"
  | "partial"
  | "fail"
  | "blocked"
  | "missing_evidence"
  | "not_run";

export type EvaluationCaseStatus = EvaluationMetricStatus;

export type EvaluationEvidenceClass =
  | "deterministic_contract"
  | "static_analysis"
  | "stream_observation"
  | "runtime_observation"
  | "ragas_provider"
  | "validation_manifest"
  | "human";

export type GoldenEvaluationCase = {
  id: string;
  title: string;
  prompt: string;
  promptVersion: string;
  origins: string[];
  sourceAliases: string[];
  categories: EvaluationCategory[];
  domain: string;
  expectedWorkflowSelection: "auto" | "single_agent" | "multi_agent";
  expectedResolvedMode: "single_agent" | "multi_agent" | "any_published";
  retrievalExpectation: "required" | "optional" | "not_expected";
  expectedArtifactType: "code" | "export" | "recovery" | "retrieval_answer";
  expectedArtifactName: string | null;
  expectedArtifact: string;
  previewContract: string;
  previewExpected: boolean;
  validationCriteria: string[];
  applicableMetricIds: string[];
};

export type GoldenEvaluationDataset = {
  version: string;
  fingerprint: string;
  promptVersion: string;
  cases: GoldenEvaluationCase[];
  rawSourceCount: number;
  duplicateCount: number;
};

export type EvaluationMetricResult = {
  id: string;
  label: string;
  category: EvaluationCategory;
  kind: "ragas" | "product_specific";
  status: EvaluationMetricStatus;
  score: number | null;
  target: number | null;
  gap: number | null;
  measuredValue: string;
  confidence: "high" | "medium" | "low" | "none";
  evidenceClass: EvaluationEvidenceClass;
  detail: string;
  evidence: string[];
};

export type EvaluationCaseResult = {
  caseId: string;
  title: string;
  domain: string;
  origins: string[];
  categories: EvaluationCategory[];
  status: EvaluationCaseStatus;
  score: number | null;
  prompt: string;
  expectedArtifact: string;
  previewContract: string;
  metrics: EvaluationMetricResult[];
  recommendation: EvaluationRecommendation | null;
};

export type EvaluationRecommendation = {
  id: string;
  caseId: string;
  category: EvaluationCategory;
  title: string;
  detail: string;
  candidateConstraint: string;
};

export type EvaluationCategoryResult = {
  category: EvaluationCategory;
  label: string;
  status: EvaluationCaseStatus;
  score: number | null;
  previousScore: number | null;
  delta: number | null;
  target: number;
  gap: number | null;
  measuredMetrics: number;
  applicableMetrics: number;
  blockedMetrics: number;
  missingMetrics: number;
  detail: string;
};

export type EvaluationStatusCounts = {
  pass: number;
  partial: number;
  fail: number;
  blocked: number;
  missing: number;
  notRun: number;
};

export type RagasCaseEvidence = {
  sampleId: string;
  metrics: Record<string, number | null>;
  metricErrors: Record<string, string>;
  sourceIds: string[];
  domains: string[];
};

export type RagasExecutionEvidence = {
  state: "not_requested" | "prepared" | "completed" | "blocked" | "failed";
  runId: string | null;
  evaluatedAt: string | null;
  datasetId: string;
  datasetVersion: string;
  privacyClass: string;
  metrics: string[];
  metricScores: Record<string, number>;
  resultRows: number;
  totalSamples: number;
  eligibleSamples: number;
  skippedSamples: number;
  metricFailures: number;
  provider: string | null;
  model: string | null;
  embeddingModel: string | null;
  ragasVersion: string | null;
  metricContract: string | null;
  durationMs: number | null;
  detail: string;
  caseRows: RagasCaseEvidence[];
};

export type EvaluationBenchmarkRun = {
  schemaVersion: typeof EVALUATION_SCHEMA_VERSION;
  id: string;
  datasetVersion: string;
  datasetFingerprint: string;
  promptVersion: string;
  scope: EvaluationScope;
  selectedCaseIds: string[];
  startedAt: string;
  completedAt: string;
  durationMs: number;
  executionMode: "deterministic_local" | "provider_assisted";
  environmentStatus: "ready" | "partially_available" | "blocked";
  statusLabel:
    | "Excellent"
    | "Demo Ready"
    | "Needs Improvement"
    | "Incomplete Evidence"
    | "Blocked";
  measuredScore: number | null;
  targetThreshold: number;
  evidenceCompleteness: number;
  caseCoverage: number;
  executedCases: number;
  selectedCases: number;
  counts: EvaluationStatusCounts;
  categoryResults: EvaluationCategoryResult[];
  caseResults: EvaluationCaseResult[];
  recommendations: EvaluationRecommendation[];
  missingMetricIds: string[];
  provider: string | null;
  model: string | null;
  workflow: string | null;
  totalTokens: number | null;
  estimatedCost: number | null;
  currency: string;
  ragas: RagasExecutionEvidence;
};

export type EvaluationRunRequest = {
  scope: EvaluationScope;
  caseIds: string[];
  allowProviderCalls: boolean;
  approvedRagasDataset: "sanitized_public" | "redacted_public";
};

export type EvaluationCandidate = {
  id: string;
  caseId: string;
  createdAt: string;
  originalPrompt: string;
  candidatePrompt: string;
  baselineScore: number | null;
  candidateScore: null;
  delta: null;
  recommendationId: string;
};

type CaseSeed = Omit<GoldenEvaluationCase, "promptVersion">;

const creativeMetrics = [
  "prompt_adherence",
  "artifact_extraction",
  "runtime_readiness",
  "preview_success",
  "technical_validity",
  "interaction",
  "visual_clarity",
  "creative_quality",
  "refinement_stability",
  "persistence",
  "terminal_outcome_truthfulness"
] as const;

const workflowMetrics = [
  "route_correctness",
  "node_completion",
  "refinement_count",
  "retries",
  "latency",
  "token_usage",
  "estimated_cost",
  "recovery_outcome"
] as const;

const reliabilityMetrics = [
  "frontend_tests",
  "backend_tests",
  "browser_e2e",
  "runtime_checks",
  "artifact_persistence",
  "session_persistence",
  "product_bug_signals"
] as const;

const ragasMetrics = [
  "context_precision",
  "faithfulness",
  "answer_relevancy",
  "context_recall",
  "context_relevancy"
] as const;

const retrievalCaseSeeds: CaseSeed[] = [
  retrievalCase(
    "runtime_selection_hydra_vs_p5",
    "Runtime selection for fast live visuals",
    "I want feedback-heavy live visuals with fast iteration and layered oscillator textures. Should I use Hydra or p5.js first?",
    "Hydra / p5.js",
    ["hydra_docs", "p5_reference"]
  ),
  retrievalCase(
    "audio_reactive_browser_mapping",
    "Audio-reactive mapping foundations",
    "How should I map bass, mids, amplitude, and FFT data into a browser visual system?",
    "p5.sound / Web Audio / Tone.js",
    ["p5_sound_analysis_reference", "web_audio_analyser_node", "web_audio_visualization_guide", "tone_js_analysis_reference"]
  ),
  retrievalCase(
    "shader_post_fx_pipeline",
    "Shader and post-processing pipeline choices",
    "How do I build a glow-heavy kaleidoscopic browser visual with shaders or post-processing passes?",
    "Three.js / GLSL / Shadertoy",
    ["three_manual_effects", "glsl_mdn_webgl_examples", "shadertoy_howto"]
  ),
  retrievalCase(
    "audiovisual_composition_browser_set",
    "Audiovisual composition timing",
    "How should I coordinate loop timing, sample playback, and visual accents across Tone.js and a browser visual runtime?",
    "Tone.js / p5.js / Three.js",
    ["tone_js_analysis_reference", "tone_js_docs", "p5_reference", "three_manual"]
  ),
  retrievalCase(
    "creative_debugging_silent_audio",
    "Creative debugging for silent browser audio",
    "My browser audio-reactive sketch stays silent until a click and the analyser looks flat. What should I check first?",
    "Web Audio / p5.js / p5.sound",
    ["web_audio_visualization_guide", "web_audio_analyser_node", "p5_reference", "p5_sound_reference"]
  ),
  retrievalCase(
    "creative_debugging_three_effects",
    "Creative debugging for Three.js effects",
    "My Three.js post-processing scene is slow and the shadows look wrong. Which render-pipeline parts should I inspect?",
    "Three.js",
    ["three_manual_effects", "three_manual"]
  ),
  retrievalCase(
    "symbol_to_art_operational_translation",
    "Operational symbol-to-art translation",
    "Turn a concentric mandala motif into a practical browser visual system with motion, rhythm, and runtime choices.",
    "p5.js / GLSL / Tone.js / Three.js",
    ["p5_reference", "glsl_mdn_webgl_examples", "tone_js_analysis_reference", "three_manual_effects"]
  )
];

export const evaluationMetricCatalog = {
  rag: [...ragasMetrics],
  creative_artifact: [...creativeMetrics],
  workflow: [...workflowMetrics],
  product_reliability: [...reliabilityMetrics]
} satisfies Record<EvaluationCategory, string[]>;

export function buildGoldenEvaluationDataset(): GoldenEvaluationDataset {
  const seeds = [
    ...demoModeScenarioCatalog.map(demoCase),
    ...homepagePromptLibrary.map((prompt) => curatedCase(prompt, "homepage")),
    ...domainStarterPromptLibrary.map((prompt) => curatedCase(prompt, "domain_starter")),
    ...morphogenesisPromptLibrary.map((prompt) => curatedCase(prompt, "morphogenesis")),
    curatedCase(rhythmicLineStudyPrompt, "prompt_pack"),
    ...retrievalCaseSeeds
  ];
  const deduplicated = new Map<string, CaseSeed>();

  for (const seed of seeds) {
    const key = normalizePrompt(seed.prompt);
    const current = deduplicated.get(key);
    if (!current) {
      deduplicated.set(key, seed);
      continue;
    }
    deduplicated.set(key, {
      ...current,
      origins: unique([...current.origins, ...seed.origins]),
      sourceAliases: unique([
        ...current.sourceAliases,
        seed.id,
        ...seed.sourceAliases
      ])
    });
  }

  const cases = [...deduplicated.values()].map((item) => ({
    ...item,
    promptVersion: stableFingerprint(item.prompt)
  }));
  const fingerprint = stableFingerprint(
    cases
      .map((item) => `${item.id}:${item.promptVersion}:${item.applicableMetricIds.join(",")}`)
      .sort()
      .join("|")
  );

  return {
    version: GOLDEN_EVALUATION_DATASET_VERSION,
    fingerprint,
    promptVersion: `exact-prompts.${stableFingerprint(cases.map((item) => item.promptVersion).join("|"))}`,
    cases,
    rawSourceCount: seeds.length,
    duplicateCount: seeds.length - cases.length
  };
}

export function buildEvaluationBenchmarkRun({
  model,
  now = new Date(),
  previousRun = null,
  ragas = emptyRagasEvidence(),
  request
}: {
  model: ProductIntelligenceModel;
  now?: Date;
  previousRun?: EvaluationBenchmarkRun | null;
  ragas?: RagasExecutionEvidence;
  request: EvaluationRunRequest;
}): EvaluationBenchmarkRun {
  const startedAt = now.toISOString();
  const dataset = buildGoldenEvaluationDataset();
  const selectedCases = selectEvaluationCases(dataset, request);
  const currentPrompt = latestUserPrompt(model);
  const currentCase = selectedCases.find(
    (item) => normalizePrompt(item.prompt) === normalizePrompt(currentPrompt)
  ) ?? null;
  const caseResults = selectedCases.map((item) =>
    item.id === currentCase?.id
      ? evaluateCurrentCase(item, model)
      : notRunCase(item)
  );

  if (scopeIncludes(request.scope, "rag")) {
    caseResults.push(ragasEvidenceCase(ragas));
  }
  if (scopeIncludes(request.scope, "product_reliability")) {
    caseResults.push(reliabilityEvidenceCase(model));
  }

  const previousComparable = previousRun && comparableRuns(previousRun, request, dataset, ragas)
    ? previousRun
    : null;
  const categoryResults = evaluationCategoriesForScope(request.scope).map((category) =>
    summarizeCategory(category, caseResults, previousComparable)
  );
  const selectedGoldenResults = caseResults.filter((item) =>
    selectedCases.some((selected) => selected.id === item.caseId)
  );
  const executedCases = selectedGoldenResults.filter((item) => item.status !== "not_run").length;
  const caseCoverage = selectedCases.length ? executedCases / selectedCases.length : 0;
  const allMetrics = caseResults.flatMap((item) => item.metrics);
  const applicableMetrics = allMetrics.length;
  const measuredMetrics = allMetrics.filter((item) => item.score != null).length;
  const evidenceCompleteness = applicableMetrics ? measuredMetrics / applicableMetrics : 0;
  const counts = countCaseStatuses(caseResults);
  // Category scores measure different constructs and are never averaged into a
  // cross-category product score. Keep the legacy field null for persisted-run
  // schema compatibility.
  const measuredScore = null;
  const recommendations = uniqueRecommendations(caseResults);
  const completedAt = new Date(now.getTime() + 1).toISOString();
  const providerTelemetry = model.details?.providerTelemetry;
  const creativeCost = model.details?.telemetryDashboard.creativeCost.current;
  const missingMetricIds = unique(
    allMetrics
      .filter((item) => ["blocked", "missing_evidence", "not_run"].includes(item.status))
      .map((item) => item.id)
  );

  return {
    schemaVersion: EVALUATION_SCHEMA_VERSION,
    id: `evaluation-${now.getTime()}`,
    datasetVersion: dataset.version,
    datasetFingerprint: dataset.fingerprint,
    promptVersion: dataset.promptVersion,
    scope: request.scope,
    selectedCaseIds: selectedCases.map((item) => item.id),
    startedAt,
    completedAt,
    durationMs: 1 + (ragas.durationMs ?? 0),
    executionMode: request.allowProviderCalls ? "provider_assisted" : "deterministic_local",
    environmentStatus:
      ragas.state === "blocked"
        ? measuredMetrics > 0 ? "partially_available" : "blocked"
        : missingMetricIds.length ? "partially_available" : "ready",
    statusLabel: statusLabelForRun({ counts, evidenceCompleteness }),
    measuredScore,
    targetThreshold: EVALUATION_TARGET_THRESHOLD,
    evidenceCompleteness,
    caseCoverage,
    executedCases,
    selectedCases: selectedCases.length,
    counts,
    categoryResults,
    caseResults,
    recommendations,
    missingMetricIds,
    provider: ragas.provider ?? providerTelemetry?.provider.name ?? null,
    model: ragas.model ?? providerTelemetry?.provider.model ?? null,
    workflow: model.details?.workflowExecution.requestedMode ?? null,
    totalTokens: providerTelemetry?.tokenUsage.totalTokens ?? null,
    estimatedCost: creativeCost?.cost ?? null,
    currency: creativeCost?.currency ?? "USD",
    ragas
  };
}

export function selectEvaluationCases(
  dataset: GoldenEvaluationDataset,
  request: Pick<EvaluationRunRequest, "scope" | "caseIds">
) {
  if (request.scope === "cases") {
    const selected = new Set(request.caseIds);
    return dataset.cases.filter((item) => selected.has(item.id));
  }
  if (request.scope === "full") return dataset.cases;
  const category = request.scope as EvaluationCategory;
  return dataset.cases.filter((item) => item.categories.includes(category));
}

export function createEvaluationCandidate({
  caseResult,
  createdAt = new Date()
}: {
  caseResult: EvaluationCaseResult;
  createdAt?: Date;
}): EvaluationCandidate | null {
  const recommendation = caseResult.recommendation;
  if (!recommendation) return null;
  return {
    id: `candidate-${caseResult.caseId}-${createdAt.getTime()}`,
    caseId: caseResult.caseId,
    createdAt: createdAt.toISOString(),
    originalPrompt: caseResult.prompt,
    candidatePrompt: `${caseResult.prompt}\n\nEvaluation candidate constraint: ${recommendation.candidateConstraint}`,
    baselineScore: caseResult.score,
    candidateScore: null,
    delta: null,
    recommendationId: recommendation.id
  };
}

export function emptyRagasEvidence(): RagasExecutionEvidence {
  return {
    state: "not_requested",
    runId: null,
    evaluatedAt: null,
    datasetId: "not_selected",
    datasetVersion: "ragas-live-session.v1",
    privacyClass: "not_selected",
    metrics: [
      "context_precision",
      "faithfulness",
      "answer_relevancy",
      "context_relevancy"
    ],
    metricScores: {},
    resultRows: 0,
    totalSamples: 0,
    eligibleSamples: 0,
    skippedSamples: 0,
    metricFailures: 0,
    provider: null,
    model: null,
    embeddingModel: null,
    ragasVersion: null,
    metricContract: null,
    durationMs: null,
    detail: "RAGAS was not requested for this evaluation scope.",
    caseRows: []
  };
}

export function formatEvaluationCategory(category: EvaluationCategory) {
  switch (category) {
    case "rag": return "RAG / Retrieval";
    case "creative_artifact": return "Creative Artifacts";
    case "workflow": return "Agents & Workflow";
    case "product_reliability": return "Product Reliability";
  }
}

export function formatEvaluationMetric(metricId: string) {
  return metricId.replace(/_/g, " ").replace(/\b\w/g, (value) => value.toUpperCase());
}

function demoCase(scenario: (typeof demoModeScenarioCatalog)[number]): CaseSeed {
  const retrievalRequired = scenario.id === "retrieval-grounded-design-brief";
  const failureCase = scenario.id === "failure-recovery-rehearsal";
  const expectedArtifactType = failureCase
    ? "recovery"
    : scenario.id === "export-handoff-package"
      ? "export"
      : "code";
  const categories: EvaluationCategory[] = ["workflow", "product_reliability"];
  if (!failureCase) categories.unshift("creative_artifact");
  if (retrievalRequired) categories.unshift("rag");
  return {
    id: `demo/${scenario.id}`,
    title: scenario.title,
    prompt: scenario.prompt,
    origins: ["demo"],
    sourceAliases: [],
    categories,
    domain: inferDomain(scenario.runtime),
    expectedWorkflowSelection: scenario.workflowMode,
    expectedResolvedMode:
      scenario.workflowMode === "auto" ? "any_published" : scenario.workflowMode,
    retrievalExpectation: retrievalRequired
      ? "required"
      : /optional/i.test(scenario.retrievalRequirement)
        ? "optional"
        : "not_expected",
    expectedArtifactType,
    expectedArtifactName: extractArtifactName(scenario.expectedArtifact, scenario.prompt),
    expectedArtifact: scenario.expectedArtifact,
    previewContract: scenario.expectedPreview,
    previewExpected: !/^no\b/i.test(scenario.expectedPreview),
    validationCriteria: [
      scenario.expectedValidation,
      scenario.expectedInteraction,
      scenario.sourceBoundary
    ],
    applicableMetricIds: applicableMetrics(categories)
  };
}

function curatedCase(prompt: CuratedPrompt, origin: string): CaseSeed {
  const categories: EvaluationCategory[] = [
    "creative_artifact",
    "workflow",
    "product_reliability"
  ];
  return {
    id: `prompt/${prompt.id}`,
    title: prompt.title,
    prompt: prompt.prompt,
    origins: [origin],
    sourceAliases: [],
    categories,
    domain: inferDomain(prompt.runtime),
    expectedWorkflowSelection: "single_agent",
    expectedResolvedMode: "single_agent",
    retrievalExpectation: "not_expected",
    expectedArtifactType: /export|handoff/i.test(prompt.runtime) ? "export" : "code",
    expectedArtifactName: extractArtifactName(prompt.expectedArtifact, prompt.prompt),
    expectedArtifact: prompt.expectedArtifact,
    previewContract: prompt.previewBoundary,
    previewExpected: !/code\/export|code-only|inspect/i.test(prompt.previewBoundary),
    validationCriteria: [prompt.previewBoundary, prompt.fallback],
    applicableMetricIds: applicableMetrics(categories)
  };
}

function retrievalCase(
  id: string,
  title: string,
  prompt: string,
  domain: string,
  expectedSourceIds: string[]
): CaseSeed {
  const categories: EvaluationCategory[] = ["rag", "workflow", "product_reliability"];
  return {
    id: `retrieval/${id}`,
    title,
    prompt,
    origins: ["retrieval_pack"],
    sourceAliases: [],
    categories,
    domain,
    expectedWorkflowSelection: "auto",
    expectedResolvedMode: "any_published",
    retrievalExpectation: "required",
    expectedArtifactType: "retrieval_answer",
    expectedArtifactName: null,
    expectedArtifact: "Source-grounded answer with published current-run retrieval evidence",
    previewContract: "No preview contract; this case evaluates retrieval and answer evidence.",
    previewExpected: false,
    validationCriteria: [
      `Registered, domain-aligned source candidates: ${expectedSourceIds.join(", ")}.`,
      "Do not require every candidate source at top-k unless the measured retrieval run returned it.",
      "RAGAS applies only after an answer and non-empty retrieved contexts are recorded."
    ],
    applicableMetricIds: applicableMetrics(categories)
  };
}

function applicableMetrics(categories: EvaluationCategory[]) {
  return unique(categories.flatMap((category) => {
    if (category === "rag") return [...ragasMetrics];
    if (category === "creative_artifact") return [...creativeMetrics];
    if (category === "workflow") return [...workflowMetrics];
    return [...reliabilityMetrics];
  }));
}

function evaluateCurrentCase(
  item: GoldenEvaluationCase,
  model: ProductIntelligenceModel
): EvaluationCaseResult {
  const details = model.details;
  const artifact = item.expectedArtifactName
    ? model.artifactRegistry.find((candidate) => candidate.title === item.expectedArtifactName) ?? null
    : model.artifactRegistry[0] ?? null;
  const metrics: EvaluationMetricResult[] = [];

  if (item.categories.includes("creative_artifact")) {
    metrics.push(...creativeMetricResults(item, model, artifact));
  }
  if (item.categories.includes("workflow")) {
    metrics.push(...workflowMetricResults(item, model));
  }
  if (item.categories.includes("rag")) {
    metrics.push(
      productMetric({
        id: "retrieval_evidence",
        label: "Current-run retrieval evidence",
        category: "rag",
        score: details?.retrievalRuntime.summary.chunkCount
          ? 1
          : null,
        status: details?.retrievalRuntime.summary.chunkCount
          ? "pass"
          : "missing_evidence",
        measuredValue: details
          ? `${details.retrievalRuntime.summary.chunkCount} chunks`
          : "Not published",
        detail: details?.retrievalRuntime.summary.chunkCount
          ? "The current run published non-empty retrieval context."
          : "No retrieved context was published for this run.",
        evidenceClass: "stream_observation",
        confidence: details?.retrievalRuntime.summary.chunkCount ? "high" : "none"
      })
    );
  }

  const result = buildCaseResult(item, metrics);
  return { ...result, recommendation: recommend(result) };
}

function creativeMetricResults(
  item: GoldenEvaluationCase,
  model: ProductIntelligenceModel,
  artifact: ProductIntelligenceModel["artifactRegistry"][number] | null
): EvaluationMetricResult[] {
  const details = model.details;
  const critique = artifact?.critique;
  const preview = details?.telemetryDashboard.preview;
  const runtime = details?.runtimeConsole;
  const refinements = artifact?.refinementPasses ?? [];
  const filenameMatch = artifact && item.expectedArtifactName
    ? artifact.title === item.expectedArtifactName
    : Boolean(artifact);
  const qualityScore = normalizeScore(
    critique?.creativeEvaluation?.overallScore ?? critique?.creativeQuality.score ?? artifact?.qualityScore
  );
  const visualScore = normalizeScore(critique?.creativeEvaluation?.composition.score);
  const refinementStable = refinements.length
    ? refinements.every((pass) =>
        pass.qualityAfter == null || pass.qualityBefore == null || pass.qualityAfter >= pass.qualityBefore
      )
    : null;
  const terminal = details?.workflowRuntime.summary.productOutcome;

  return [
    scoredProductMetric("prompt_adherence", "Prompt adherence", "creative_artifact", critique?.promptAlignment.score ?? (filenameMatch ? 1 : artifact ? 0.4 : null), critique?.promptAlignment.rationale ?? "Checked against exact artifact filename and available critique evidence.", "static_analysis"),
    scoredProductMetric("artifact_extraction", "Artifact extraction", "creative_artifact", artifact ? (filenameMatch ? 1 : 0.4) : null, artifact ? `${artifact.title} was extracted.` : "No matching artifact was extracted.", "stream_observation"),
    scoredProductMetric("runtime_readiness", "Runtime readiness", "creative_artifact", item.previewExpected ? (artifact?.previewEligible ? 1 : critique?.runtimeSuitability.score ?? null) : 1, item.previewExpected ? "Uses published artifact preview eligibility and runtime suitability." : "This case explicitly has no internal preview contract.", "static_analysis"),
    scoredProductMetric("preview_success", "Preview success", "creative_artifact", item.previewExpected ? (preview?.available && !preview.error ? 1 : preview?.error ? 0 : null) : 1, item.previewExpected ? preview?.detail ?? "No browser Preview evidence published." : "No-preview boundary preserved.", "runtime_observation"),
    scoredProductMetric("technical_validity", "Technical validity", "creative_artifact", critique?.codeQuality.score ?? (runtime?.health.signal === "healthy" ? 1 : runtime?.health.signal === "failed" ? 0 : null), critique?.codeQuality.rationale ?? runtime?.health.explanation ?? "No technical validity evidence published.", runtime?.hasRuntimeActivity ? "runtime_observation" : "static_analysis"),
    missingMetric("interaction", "Interaction", "creative_artifact", "Interaction intent is documented, but successful interaction is not automatically measured."),
    visualScore == null
      ? missingMetric("visual_clarity", "Visual clarity", "creative_artifact", "No rendered-frame or reviewed composition score is available.")
      : scoredProductMetric("visual_clarity", "Visual clarity", "creative_artifact", visualScore, critique?.creativeEvaluation?.composition.observation ?? "Static composition evidence.", "static_analysis"),
    qualityScore == null
      ? missingMetric("creative_quality", "Creative quality", "creative_artifact", "No bounded creative-quality analysis is available.")
      : scoredProductMetric("creative_quality", "Creative quality", "creative_artifact", qualityScore, critique?.creativeEvaluation?.summary ?? critique?.creativeQuality.rationale ?? "Bounded static creative-quality evidence.", "static_analysis"),
    refinementStable == null
      ? missingMetric("refinement_stability", "Refinement stability", "creative_artifact", "No refinement passes were recorded for this artifact.")
      : scoredProductMetric("refinement_stability", "Refinement stability", "creative_artifact", refinementStable ? 1 : 0, `${refinements.length} explicit refinement pass${refinements.length === 1 ? "" : "es"} recorded.`, "stream_observation"),
    productMetric({
      id: "persistence",
      label: "Persistence",
      category: "creative_artifact",
      score: artifact ? 0.5 : null,
      status: artifact ? "partial" : "missing_evidence",
      measuredValue: artifact ? "Current snapshot" : "Not observed",
      detail: artifact ? "Artifact is retained in the current snapshot; reload restoration was not executed by this run." : "No artifact is available to inspect.",
      evidenceClass: "stream_observation",
      confidence: artifact ? "medium" : "none"
    }),
    terminal?.product_outcome && terminal.product_outcome !== "IN_PROGRESS"
      ? scoredProductMetric(
          "terminal_outcome_truthfulness",
          "Terminal outcome truthfulness",
          "creative_artifact",
          terminalTruthScore(terminal, preview?.available ?? false),
          terminal.summary,
          "stream_observation"
        )
      : missingMetric("terminal_outcome_truthfulness", "Terminal outcome truthfulness", "creative_artifact", "No terminal product outcome was published.")
  ];
}

function workflowMetricResults(
  item: GoldenEvaluationCase,
  model: ProductIntelligenceModel
): EvaluationMetricResult[] {
  const details = model.details;
  const execution = details?.workflowExecution;
  const runtime = details?.workflowRuntime;
  const telemetry = details?.providerTelemetry;
  const creativeCost = details?.telemetryDashboard.creativeCost.current;
  const routeScore = execution?.state === "available"
    ? item.expectedResolvedMode === "any_published"
      ? execution.requestedMode === item.expectedWorkflowSelection && execution.resolvedMode != null ? 1 : 0
      : execution.requestedMode === item.expectedWorkflowSelection && execution.resolvedMode === item.expectedResolvedMode ? 1 : 0
    : null;
  const nodeScore = runtime?.summary.activity.terminal && runtime.summary.total
    ? runtime.summary.reached / runtime.summary.total
    : null;
  const refinementCount = creativeCost?.refinementCount ?? null;
  const retryCount = runtime?.summary.retryCount ?? telemetry?.execution.retryCount ?? null;
  const recovery = runtime?.summary.productOutcome;
  return [
    routeScore == null
      ? missingMetric("route_correctness", "Route correctness", "workflow", "No requested/resolved route evidence was published.")
      : scoredProductMetric("route_correctness", "Route correctness", "workflow", routeScore, execution?.rationale ?? "Published execution decision.", "stream_observation"),
    nodeScore == null
      ? missingMetric("node_completion", "Successful node completion", "workflow", "The workflow has not published a terminal node summary.")
      : scoredProductMetric("node_completion", "Successful node completion", "workflow", nodeScore, `${runtime?.summary.reached}/${runtime?.summary.total} workflow nodes reached.`, "stream_observation"),
    measuredOnlyMetric("refinement_count", "Refinement count", "workflow", refinementCount == null ? "Not published" : String(refinementCount), refinementCount == null ? "No refinement count was published." : "Count reported by workflow/cost telemetry."),
    measuredOnlyMetric("retries", "Retries", "workflow", retryCount == null ? "Not published" : String(retryCount), retryCount == null ? "No retry count was published." : "Retry count is reported without treating every retry as failure."),
    measuredOnlyMetric("latency", "Latency", "workflow", runtime?.summary.totalRuntimeMs == null ? "Not published" : `${runtime.summary.totalRuntimeMs} ms`, "Measured duration is reported without an invented pass/fail target."),
    measuredOnlyMetric("token_usage", "Token usage", "workflow", telemetry?.tokenUsage.totalTokens == null ? "Not published" : telemetry.tokenUsage.totalTokens.toLocaleString(), "Only provider-published usage is shown."),
    measuredOnlyMetric("estimated_cost", "Estimated cost", "workflow", creativeCost?.cost == null ? "Not published" : `${creativeCost.currency} ${creativeCost.cost.toFixed(4)}`, "Cost remains unavailable when provider usage or pricing was not published."),
    recovery?.product_outcome && recovery.product_outcome !== "IN_PROGRESS"
      ? scoredProductMetric("recovery_outcome", "Recovery outcome", "workflow", recovery.product_outcome === "FAILURE" ? 0 : recovery.product_outcome === "PARTIAL" ? 0.65 : 1, recovery.summary, "stream_observation")
      : missingMetric("recovery_outcome", "Recovery outcome", "workflow", "No terminal recovery outcome was published.")
  ];
}

function reliabilityEvidenceCase(model: ProductIntelligenceModel): EvaluationCaseResult {
  const workstation = model.details?.workstationDashboard;
  const runtime = model.details?.runtimeConsole;
  const snapshot = model.details?.snapshot;
  const metrics: EvaluationMetricResult[] = [
    blockedMetric("frontend_tests", "Frontend tests", "product_reliability", "Browser execution cannot run the frontend test suite; a current validation manifest was not provided."),
    blockedMetric("backend_tests", "Backend tests", "product_reliability", "Browser execution cannot run the backend test suite; a current validation manifest was not provided."),
    blockedMetric("browser_e2e", "Browser E2E", "product_reliability", "The evaluation action does not launch Playwright. Historical QA remains separate evidence."),
    runtime?.hasRuntimeActivity
      ? scoredProductMetric("runtime_checks", "Runtime checks", "product_reliability", runtime.health.signal === "healthy" ? 1 : runtime.health.signal === "degraded" ? 0.6 : 0, runtime.health.explanation, "runtime_observation")
      : missingMetric("runtime_checks", "Runtime checks", "product_reliability", "No browser runtime activity is available for this session."),
    snapshot?.artifacts.length
      ? productMetric({
          id: "artifact_persistence",
          label: "Artifact persistence",
          category: "product_reliability",
          score: 0.5,
          status: "partial",
          measuredValue: `${snapshot.artifacts.length} retained`,
          detail: "Artifacts are present in the current snapshot; reload restoration was not executed by this evaluation action.",
          evidenceClass: "stream_observation",
          confidence: "medium"
        })
      : missingMetric("artifact_persistence", "Artifact persistence", "product_reliability", "No artifact is retained in the current snapshot."),
    blockedMetric("session_persistence", "Session persistence", "product_reliability", "Session reload persistence requires an explicit reload/E2E check and was not run here."),
    workstation
      ? scoredProductMetric("product_bug_signals", "Product Bug signals", "product_reliability", workstation.summary.errorCount === 0 ? 1 : 0, `${workstation.summary.errorCount} current error signal${workstation.summary.errorCount === 1 ? "" : "s"}. This is current-state evidence, not a bug-suite pass.`, "stream_observation")
      : missingMetric("product_bug_signals", "Product Bug signals", "product_reliability", "No workstation signal model is available.")
  ];
  return buildCaseResult(
    {
      id: "system/current-workspace-reliability",
      title: "Current workspace reliability evidence",
      prompt: "",
      promptVersion: "not_applicable",
      origins: ["current_workspace"],
      sourceAliases: [],
      categories: ["product_reliability"],
      domain: "Product runtime",
      expectedWorkflowSelection: "auto",
      expectedResolvedMode: "any_published",
      retrievalExpectation: "optional",
      expectedArtifactType: "recovery",
      expectedArtifactName: null,
      expectedArtifact: "Current workstation, runtime, persistence, and Product Bug evidence",
      previewContract: "Current browser runtime only",
      previewExpected: false,
      validationCriteria: [],
      applicableMetricIds: [...reliabilityMetrics]
    },
    metrics
  );
}

function ragasEvidenceCase(ragas: RagasExecutionEvidence): EvaluationCaseResult {
  const metrics: EvaluationMetricResult[] = ragasMetrics.map((metricId) => {
    if (metricId === "context_recall") {
      return missingMetric(metricId, "Context Recall", "rag", "Not implemented: approved datasets have no justified reference answers.", "ragas");
    }
    const score = ragas.metricScores[metricId];
    if (score != null) {
      return metricResult({
        id: metricId,
        label: formatEvaluationMetric(metricId),
        category: "rag",
        kind: "ragas",
        score,
        detail: `Real RAGAS score across ${ragas.resultRows} approved retrieval row${ragas.resultRows === 1 ? "" : "s"}.`,
        evidenceClass: "ragas_provider",
        confidence: ragas.resultRows > 1 ? "high" : "medium",
        measuredValue: `${Math.round(score * 100)}%`
      });
    }
    if (ragas.state === "blocked" || ragas.state === "prepared" || ragas.state === "not_requested") {
      return blockedMetric(
        metricId,
        formatEvaluationMetric(metricId),
        "rag",
        ragas.state === "prepared"
          ? "Dataset eligibility was prepared locally; evaluator providers were not called."
          : ragas.detail,
        "ragas"
      );
    }
    return missingMetric(metricId, formatEvaluationMetric(metricId), "rag", ragas.detail, "ragas");
  });
  return buildCaseResult(
    {
      id: `ragas/${ragas.datasetId}`,
      title: "Approved RAGAS retrieval dataset",
      prompt: "",
      promptVersion: ragas.datasetVersion,
      origins: [ragas.privacyClass],
      sourceAliases: [],
      categories: ["rag"],
      domain: "Recorded retrieval sessions",
      expectedWorkflowSelection: "auto",
      expectedResolvedMode: "any_published",
      retrievalExpectation: "required",
      expectedArtifactType: "retrieval_answer",
      expectedArtifactName: null,
      expectedArtifact: "Recorded answer with non-empty retrieved contexts",
      previewContract: "Not applicable",
      previewExpected: false,
      validationCriteria: [],
      applicableMetricIds: [...ragasMetrics]
    },
    metrics
  );
}

function notRunCase(item: GoldenEvaluationCase): EvaluationCaseResult {
  return {
    caseId: item.id,
    title: item.title,
    domain: item.domain,
    origins: item.origins,
    categories: item.categories,
    status: "not_run",
    score: null,
    prompt: item.prompt,
    expectedArtifact: item.expectedArtifact,
    previewContract: item.previewContract,
    metrics: item.applicableMetricIds.map((metricId) => ({
      id: metricId,
      label: formatEvaluationMetric(metricId),
      category: categoryForMetric(metricId),
      kind: ragasMetrics.includes(metricId as (typeof ragasMetrics)[number]) ? "ragas" : "product_specific",
      status: "not_run",
      score: null,
      target: EVALUATION_TARGET_THRESHOLD,
      gap: null,
      measuredValue: "Not run",
      confidence: "none",
      evidenceClass: "deterministic_contract",
      detail: "The canonical case is defined, but this prompt was not executed in the current session.",
      evidence: []
    })),
    recommendation: null
  };
}

function buildCaseResult(
  item: GoldenEvaluationCase,
  metrics: EvaluationMetricResult[]
): EvaluationCaseResult {
  const scoreValues = metrics.flatMap((metric) => metric.score == null ? [] : [metric.score]);
  const score = scoreValues.length ? average(scoreValues) : null;
  const status = statusFromMetrics(metrics);
  return {
    caseId: item.id,
    title: item.title,
    domain: item.domain,
    origins: item.origins,
    categories: item.categories,
    status,
    score,
    prompt: item.prompt,
    expectedArtifact: item.expectedArtifact,
    previewContract: item.previewContract,
    metrics,
    recommendation: null
  };
}

function recommend(result: EvaluationCaseResult): EvaluationRecommendation | null {
  const weak = result.metrics.find((metric) => metric.status === "fail" || metric.status === "partial");
  if (!weak) return null;
  const content = recommendationForMetric(weak.id, result);
  return {
    id: `recommendation-${result.caseId}-${weak.id}`,
    caseId: result.caseId,
    category: weak.category,
    ...content
  };
}

function recommendationForMetric(metricId: string, result: EvaluationCaseResult) {
  if (["context_precision", "faithfulness", "answer_relevancy", "context_relevancy", "retrieval_evidence"].includes(metricId)) {
    return { title: "Improve retrieval query and context selection", detail: "Inspect query specificity, domain filters, and which chunks were used before changing the answer prompt.", candidateConstraint: "Use current-run retrieval only when sources are published; state the source boundary and exclude irrelevant contexts." };
  }
  if (metricId === "route_correctness") {
    return { title: "Correct the workflow route", detail: "The published requested/resolved route did not match this case contract.", candidateConstraint: "Use the benchmark’s explicit workflow route and preserve the published route evidence in the result." };
  }
  if (["runtime_readiness", "preview_success", "technical_validity"].includes(metricId)) {
    return { title: "Tighten runtime compatibility", detail: `Reassert the supported runtime and Preview boundary for ${result.domain}.`, candidateConstraint: `Return only the exact supported artifact for ${result.domain}; do not add wrappers, imports, or unsupported runtime features.` };
  }
  if (["creative_quality", "visual_clarity"].includes(metricId)) {
    return { title: "Clarify the creative direction", detail: "Make composition, contrast, motion, and hierarchy requirements more concrete without changing the canonical original.", candidateConstraint: "Strengthen composition hierarchy, contrast, motion rhythm, and visual legibility while preserving the runtime contract." };
  }
  return { title: "Clarify the expected artifact contract", detail: "The measured output only partially satisfied the exact case boundary.", candidateConstraint: `Return the exact expected artifact (${result.expectedArtifact}) and preserve every stated output constraint.` };
}

function summarizeCategory(
  category: EvaluationCategory,
  caseResults: EvaluationCaseResult[],
  previousRun: EvaluationBenchmarkRun | null
): EvaluationCategoryResult {
  const metrics = caseResults
    .filter((item) => item.status !== "not_run" && item.categories.includes(category))
    .flatMap((item) => item.metrics.filter((metric) => metric.category === category));
  const scored = metrics.flatMap((metric) => metric.score == null ? [] : [metric.score]);
  const evidenceRatio = metrics.length ? scored.length / metrics.length : 0;
  const score = scored.length && evidenceRatio >= 0.5 ? average(scored) : null;
  const previousScore = previousRun?.categoryResults.find((item) => item.category === category)?.score ?? null;
  const blocked = metrics.filter((item) => item.status === "blocked").length;
  const missing = metrics.filter((item) => ["missing_evidence", "not_run"].includes(item.status)).length;
  return {
    category,
    label: formatEvaluationCategory(category),
    status: statusFromMetrics(metrics),
    score,
    previousScore,
    delta: score != null && previousScore != null ? score - previousScore : null,
    target: EVALUATION_TARGET_THRESHOLD,
    gap: score == null ? null : Math.max(0, EVALUATION_TARGET_THRESHOLD - score),
    measuredMetrics: scored.length,
    applicableMetrics: metrics.length,
    blockedMetrics: blocked,
    missingMetrics: missing,
    detail: score == null
      ? scored.length > 0
        ? `${scored.length}/${metrics.length} observations were measured; the category score is withheld because evidence coverage is below 50%.`
        : blocked > 0
        ? "Execution or evidence is blocked; no zero score was substituted."
        : "No defensible measured score is available yet."
      : `${scored.length}/${metrics.length} applicable metric observations contributed to this category only.`
  };
}

function productMetric(input: Omit<EvaluationMetricResult, "kind" | "target" | "gap" | "evidence"> & { evidence?: string[] }): EvaluationMetricResult {
  return {
    ...input,
    kind: "product_specific",
    target: EVALUATION_TARGET_THRESHOLD,
    gap: input.score == null ? null : Math.max(0, EVALUATION_TARGET_THRESHOLD - input.score),
    evidence: input.evidence ?? []
  };
}

function scoredProductMetric(
  id: string,
  label: string,
  category: EvaluationCategory,
  rawScore: number | null | undefined,
  detail: string,
  evidenceClass: EvaluationEvidenceClass
) {
  const score = normalizeScore(rawScore);
  if (score == null) return missingMetric(id, label, category, detail);
  return metricResult({ id, label, category, kind: "product_specific", score, detail, evidenceClass, confidence: evidenceClass === "runtime_observation" ? "high" : "medium", measuredValue: `${Math.round(score * 100)}%` });
}

function measuredOnlyMetric(
  id: string,
  label: string,
  category: EvaluationCategory,
  value: string,
  detail: string
): EvaluationMetricResult {
  const missing = value === "Not published";
  return productMetric({
    id,
    label,
    category,
    status: missing ? "missing_evidence" : "pass",
    score: null,
    measuredValue: value,
    confidence: missing ? "none" : "high",
    evidenceClass: "stream_observation",
    detail
  });
}

function missingMetric(
  id: string,
  label: string,
  category: EvaluationCategory,
  detail: string,
  kind: "ragas" | "product_specific" = "product_specific"
): EvaluationMetricResult {
  return {
    id,
    label,
    category,
    kind,
    status: "missing_evidence",
    score: null,
    target: EVALUATION_TARGET_THRESHOLD,
    gap: null,
    measuredValue: "Missing evidence",
    confidence: "none",
    evidenceClass: kind === "ragas" ? "ragas_provider" : "deterministic_contract",
    detail,
    evidence: []
  };
}

function blockedMetric(
  id: string,
  label: string,
  category: EvaluationCategory,
  detail: string,
  kind: "ragas" | "product_specific" = "product_specific"
): EvaluationMetricResult {
  return {
    ...missingMetric(id, label, category, detail, kind),
    status: "blocked",
    measuredValue: "BLOCKED_BY_EXECUTION_ENVIRONMENT"
  };
}

function metricResult(input: {
  id: string;
  label: string;
  category: EvaluationCategory;
  kind: "ragas" | "product_specific";
  score: number;
  detail: string;
  evidenceClass: EvaluationEvidenceClass;
  confidence: EvaluationMetricResult["confidence"];
  measuredValue: string;
}): EvaluationMetricResult {
  const score = normalizeScore(input.score) ?? 0;
  return {
    ...input,
    score,
    status: score >= EVALUATION_TARGET_THRESHOLD ? "pass" : score >= 0.6 ? "partial" : "fail",
    target: EVALUATION_TARGET_THRESHOLD,
    gap: Math.max(0, EVALUATION_TARGET_THRESHOLD - score),
    evidence: []
  };
}

function categoryForMetric(metricId: string): EvaluationCategory {
  for (const [category, metrics] of Object.entries(evaluationMetricCatalog) as [EvaluationCategory, string[]][]) {
    if (metrics.includes(metricId)) return category;
  }
  return "creative_artifact";
}

function scopeIncludes(scope: EvaluationScope, category: EvaluationCategory) {
  return scope === "full" || scope === "cases" || scope === category;
}

function evaluationCategoriesForScope(scope: EvaluationScope): EvaluationCategory[] {
  if (scope === "full" || scope === "cases") {
    return ["rag", "creative_artifact", "workflow", "product_reliability"];
  }
  return [scope];
}

function comparableRuns(
  previous: EvaluationBenchmarkRun,
  request: EvaluationRunRequest,
  dataset: GoldenEvaluationDataset,
  ragas: RagasExecutionEvidence
) {
  const selected = selectEvaluationCases(dataset, request).map((item) => item.id);
  const ragasComparable = !scopeIncludes(request.scope, "rag") || (
    previous.ragas.datasetId === ragas.datasetId &&
    previous.ragas.datasetVersion === ragas.datasetVersion &&
    previous.ragas.privacyClass === ragas.privacyClass &&
    [...previous.ragas.metrics].sort().join("|") === [...ragas.metrics].sort().join("|") &&
    previous.ragas.model === ragas.model &&
    previous.ragas.provider === ragas.provider &&
    previous.ragas.embeddingModel === ragas.embeddingModel &&
    previous.ragas.ragasVersion === ragas.ragasVersion &&
    previous.ragas.metricContract === ragas.metricContract
  );
  return ragasComparable &&
    previous.datasetFingerprint === dataset.fingerprint &&
    previous.scope === request.scope &&
    previous.selectedCaseIds.join("|") === selected.join("|");
}

function statusLabelForRun({
  counts,
  evidenceCompleteness
}: {
  counts: EvaluationStatusCounts;
  evidenceCompleteness: number;
}): EvaluationBenchmarkRun["statusLabel"] {
  if (counts.blocked > 0 && counts.pass + counts.partial + counts.fail === 0) return "Blocked";
  if (evidenceCompleteness < 1 || counts.missing > 0 || counts.notRun > 0 || counts.blocked > 0) return "Incomplete Evidence";
  if (counts.fail > 0 || counts.partial > 0) return "Needs Improvement";
  return "Demo Ready";
}

function statusFromMetrics(metrics: EvaluationMetricResult[]): EvaluationMetricStatus {
  if (metrics.some((item) => item.status === "fail")) return "fail";
  if (metrics.some((item) => item.status === "partial")) return "partial";
  const hasMeasuredPass = metrics.some((item) => item.status === "pass");
  const hasBlocked = metrics.some((item) => item.status === "blocked");
  const hasMissing = metrics.some((item) => item.status === "missing_evidence");
  if (hasBlocked && !hasMeasuredPass) return "blocked";
  if (hasBlocked || hasMissing) return "missing_evidence";
  if (hasMeasuredPass) return "pass";
  return "not_run";
}

function countCaseStatuses(results: EvaluationCaseResult[]): EvaluationStatusCounts {
  return {
    pass: results.filter((item) => item.status === "pass").length,
    partial: results.filter((item) => item.status === "partial").length,
    fail: results.filter((item) => item.status === "fail").length,
    blocked: results.filter((item) => item.status === "blocked").length,
    missing: results.filter((item) => item.status === "missing_evidence").length,
    notRun: results.filter((item) => item.status === "not_run").length
  };
}

function uniqueRecommendations(caseResults: EvaluationCaseResult[]) {
  const recommendations = caseResults.flatMap((item) => item.recommendation ? [item.recommendation] : []);
  return [...new Map(recommendations.map((item) => [item.id, item])).values()];
}

function normalizeScore(value: number | null | undefined) {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  return Math.max(0, Math.min(1, value > 1 ? value / 100 : value));
}

function terminalTruthScore(
  outcome: NonNullable<ProductIntelligenceModel["details"]>["workflowRuntime"]["summary"]["productOutcome"],
  previewAvailable: boolean
) {
  if (outcome.product_outcome === "SUCCESS" && outcome.preview_status.toLowerCase().includes("failed")) return 0;
  if (outcome.product_outcome === "SUCCESS" && outcome.preview_status.toLowerCase().includes("ready") && !previewAvailable) return 0.5;
  return 1;
}

function latestUserPrompt(model: ProductIntelligenceModel) {
  return [...(model.details?.snapshot.messages ?? [])]
    .reverse()
    .find((message) => message.role === "user")?.content ?? "";
}

function extractArtifactName(expectedArtifact: string, prompt: string) {
  const text = `${expectedArtifact} ${prompt}`;
  return text.match(/[a-z0-9][a-z0-9-]*\.(?:p5\.js|three\.js|tone\.js|hydra\.js|gsap\.js|canvas\.js|frag|glsl|svg|md)/i)?.[0] ?? null;
}

function inferDomain(runtime: string) {
  const value = runtime.toLowerCase();
  if (value.includes("p5")) return "p5.js";
  if (value.includes("three")) return "Three.js";
  if (value.includes("glsl") || value.includes("shader")) return "GLSL";
  if (value.includes("tone") || value.includes("audio")) return "Tone.js / Web Audio";
  if (value.includes("hydra")) return "Hydra";
  if (value.includes("gsap")) return "GSAP";
  if (value.includes("svg")) return "SVG";
  if (value.includes("canvas")) return "Canvas 2D";
  if (value.includes("export") || value.includes("handoff")) return "External handoff";
  return runtime;
}

function normalizePrompt(value: string) {
  return value.replace(/\s+/g, " ").trim();
}

function stableFingerprint(value: string) {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

function unique<T>(values: T[]) {
  return [...new Set(values)];
}

function average(values: number[]) {
  return values.reduce((total, value) => total + value, 0) / values.length;
}
