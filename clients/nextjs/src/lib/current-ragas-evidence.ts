import canonicalRetrievalReport from "../../../../demo/evaluation/canonical_retrieval_report.json";
import currentProductRagasEvidence from "../../../../demo/evaluation/current_product_ragas_evidence.json";
import type {
  RagasCaseEvidence,
  RagasExecutionEvidence
} from "./evaluation-benchmark";

export const CURRENT_APPROVED_RAGAS_EVALUATED_AT = "2026-07-13T03:00:18.124301Z";

export const CURRENT_CANONICAL_RETRIEVAL_REPORT = {
  evaluatedAt: canonicalRetrievalReport.evaluatedAt,
  verificationState: "verified_current_worktree",
  reportArtifact: "demo/evaluation/canonical_retrieval_report.json",
  selectionFingerprint: canonicalRetrievalReport.selectionFingerprint,
  kbSnapshot: canonicalRetrievalReport.kbSnapshot,
  benchmarkId: canonicalRetrievalReport.benchmarkId,
  retrievalPackCases: canonicalRetrievalReport.benchmarkCaseCount,
  casesWithResults: canonicalRetrievalReport.summary.casesWithResults,
  ragScopedGoldenContracts: 8,
  expectedSourceOverlap: canonicalRetrievalReport.summary.expectedSourceOverlap,
  requestedDomainCoverage: canonicalRetrievalReport.summary.requestedDomainCoverage,
  embeddingModel: canonicalRetrievalReport.embeddingModel,
  retrievalLimit: canonicalRetrievalReport.retrievalLimit,
  interpretation: canonicalRetrievalReport.interpretation,
  qualityInterpretation:
    "Heading-only chunks and verified index-only sources are excluded; bounded candidate headroom then recovered substantive Three.js manual evidence without source pinning."
} as const;

export const CANONICAL_RETRIEVAL_EVOLUTION = [
  {
    id: "pre_loop_baseline",
    label: "Pre-loop baseline",
    sourceCoverage: { covered: 9, expected: 23, ratio: 9 / 23 },
    domainCoverage: { covered: 7, expected: 19, ratio: 7 / 19 },
    finding: "Intent ambiguity and a relevance-only candidate pool obscured requested domains."
  },
  {
    id: "intent_and_diversity",
    label: "Intent + result diversity",
    sourceCoverage: { covered: 12, expected: 23, ratio: 12 / 23 },
    domainCoverage: { covered: 11, expected: 19, ratio: 11 / 19 },
    finding: "Bounded domain intent and diverse final selection recovered relevant cross-domain evidence."
  },
  {
    id: "balanced_candidates",
    label: "Balanced candidate search",
    sourceCoverage: { covered: 15, expected: 23, ratio: 15 / 23 },
    domainCoverage: { covered: 17, expected: 19, ratio: 17 / 19 },
    finding: "Per-domain nearest-neighbour pools prevented one strong domain from monopolizing candidates."
  },
  {
    id: "requested_domain_fallback",
    label: "Requested-domain fallback",
    sourceCoverage: { covered: 15, expected: 23, ratio: 15 / 23 },
    domainCoverage: { covered: 18, expected: 19, ratio: 18 / 19 },
    finding: "A bounded fallback retained the best available evidence when generic-source filtering erased a requested domain."
  },
  {
    id: "source_diversity",
    label: "Source-diverse context",
    sourceCoverage: { covered: 17, expected: 23, ratio: 17 / 23 },
    domainCoverage: { covered: 18, expected: 19, ratio: 18 / 19 },
    finding: "A two-chunk primary cap stopped one source from consuming remaining context slots before distinct evidence was considered."
  },
  {
    id: "manual_and_unseen_source",
    label: "Manual + unseen-source selection",
    sourceCoverage: { covered: 19, expected: 23, ratio: 19 / 23 },
    domainCoverage: { covered: 18, expected: 19, ratio: 18 / 19 },
    finding: "Chunk-level manual filtering and unseen-source priority recovered useful Three.js and p5.sound evidence; lineage also exposed false heading/index coverage."
  },
  {
    id: "substantive_context_only",
    label: "Substantive context only",
    sourceCoverage: { covered: 15, expected: 23, ratio: 15 / 23 },
    domainCoverage: { covered: 18, expected: 19, ratio: 18 / 19 },
    finding: "Heading-only chunks and the Tone.js API-name index were removed even though their removal lowered anchor overlap."
  },
  {
    id: "bounded_filter_headroom",
    label: "Bounded filter headroom",
    sourceCoverage: { covered: 16, expected: 23, ratio: 16 / 23 },
    domainCoverage: { covered: 18, expected: 19, ratio: 18 / 19 },
    finding: "Bounded candidate headroom recovered substantive Three.js manual guidance after quality filtering, without changing top-k or pinning sources."
  }
] as const;

export const CURRENT_APPROVED_RAGAS_EVIDENCE: RagasExecutionEvidence = {
  schemaVersion: "historical-ragas-evidence.v1",
  scope: "historical_fixture",
  state: "completed",
  runId: "a85a9f445df7481eb2d327ccc0b6f055",
  evaluatedAt: CURRENT_APPROVED_RAGAS_EVALUATED_AT,
  datasetId: "sanitized_public",
  datasetVersion: "sanitized-ragas.v1",
  privacyClass: "committed_synthetic_public",
  metrics: [
    "context_precision",
    "faithfulness",
    "answer_relevancy",
    "context_relevancy"
  ],
  metricScores: {
    context_precision: 0.999999999925,
    faithfulness: 0.29583333333333334,
    answer_relevancy: 0.4742546883775048,
    context_relevancy: 0.6875
  },
  retrievalScore: 0.6143970054089595,
  resultRows: 4,
  totalSamples: 4,
  eligibleSamples: 4,
  skippedSamples: 0,
  metricFailures: 0,
  provider: "OpenAI evaluator",
  model: "gpt-4o-mini",
  embeddingModel: "text-embedding-3-small",
  ragasVersion: "0.4.3",
  metricContract: "ragas-supported.v2",
  durationMs: 46_880,
  detail:
    "Committed transcribed summary of the approved provider-scored fixture. It is separate from canonical golden-case execution coverage.",
  benchmarkMode: "historical_fixture",
  scoreOrigin: "historical_fixture",
  benchmarkVersion: "sanitized-ragas.v1",
  selectedCaseIds: [],
  datasetFingerprint: null,
  retrievalFingerprint: null,
  promptFingerprint: null,
  generationFingerprint: null,
  outputFingerprint: null,
  selectionFingerprint: null,
  kbFingerprint: null,
  generationModel: null,
  evaluator: "OpenAI evaluator / gpt-4o-mini",
  evaluatorModel: "gpt-4o-mini",
  timestamp: CURRENT_APPROVED_RAGAS_EVALUATED_AT,
  caseRows: [
    {
      sampleId: "sanitized_ragas_p5_setup_draw",
      metrics: {
        context_precision: 0.9999999999,
        faithfulness: 0.6,
        answer_relevancy: 0.5908416371437247,
        context_relevancy: 0.75
      },
      metricErrors: {},
      sourceIds: ["p5_reference_setup_draw", "cca_geometry_boundary"],
      domains: ["p5_js"],
      promptFingerprint: null,
      generationFingerprint: null
    },
    {
      sampleId: "sanitized_ragas_three_audio_postfx",
      metrics: {
        context_precision: 0.9999999999,
        faithfulness: 0,
        answer_relevancy: 0.5645202957904727,
        context_relevancy: 0.5
      },
      metricErrors: {},
      sourceIds: ["three_manual_post_processing", "three_manual_fundamentals", "mdn_web_audio_analyser"],
      domains: ["three_js", "web_audio_api"],
      promptFingerprint: null,
      generationFingerprint: null
    },
    {
      sampleId: "sanitized_ragas_glsl_kaleidoscope",
      metrics: {
        context_precision: 0.99999999995,
        faithfulness: 0.3333333333333333,
        answer_relevancy: 0,
        context_relevancy: 0.5
      },
      metricErrors: {},
      sourceIds: ["opengl_glsl_fragment_shader", "shadertoy_uniforms_guide", "three_shader_material_reference"],
      domains: ["glsl", "shadertoy", "three_js"],
      promptFingerprint: null,
      generationFingerprint: null
    },
    {
      sampleId: "sanitized_ragas_hydra_boundary",
      metrics: {
        context_precision: 0.99999999995,
        faithfulness: 0.25,
        answer_relevancy: 0.7416568205758219,
        context_relevancy: 1
      },
      metricErrors: {},
      sourceIds: ["cca_demo_pack_boundaries", "cca_capstone_evidence_hydra_boundary"],
      domains: ["hydra"],
      promptFingerprint: null,
      generationFingerprint: null
    }
  ]
};

type CanonicalCurrentProductCase = Omit<RagasCaseEvidence, "sampleId"> & {
  caseId: string;
};

type CanonicalCurrentProductEvidence = Omit<
  RagasExecutionEvidence,
  "state" | "caseRows"
> & {
  status: RagasExecutionEvidence["state"];
  caseResults: CanonicalCurrentProductCase[];
};

// The committed canonical JSON is the sole static source of current-product
// evidence. The Evaluation workspace treats it like a persisted run, then lets
// any newer successful persisted run win by timestamp. Historical fixture
// evidence remains separate above.
const {
  status: currentProductStatus,
  caseResults: currentProductCaseResults,
  ...currentProductEvidence
} = currentProductRagasEvidence as CanonicalCurrentProductEvidence;

export const CURRENT_PRODUCT_RAGAS_EVIDENCE: RagasExecutionEvidence = {
  ...currentProductEvidence,
  state: currentProductStatus,
  caseRows: currentProductCaseResults.map(({ caseId, ...row }) => ({
    ...row,
    sampleId: caseId
  }))
};
