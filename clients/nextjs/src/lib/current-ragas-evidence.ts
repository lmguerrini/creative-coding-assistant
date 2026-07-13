import type { RagasExecutionEvidence } from "./evaluation-benchmark";

export const CURRENT_APPROVED_RAGAS_EVALUATED_AT = "2026-07-13T03:00:18.124301Z";

export const CURRENT_CANONICAL_RETRIEVAL_REPORT = {
  evaluatedAt: "2026-07-13T05:05:33.306298+00:00",
  verificationState: "verified_current_worktree",
  reportArtifact: "demo/evaluation/canonical_retrieval_report.json",
  selectionFingerprint: "sha256:74acf5d62f669eff64fd5fe4fe176bff04da4fcbdc7a7588e18b85a8a418d1c7",
  kbSnapshot: {
    recordCount: 1445,
    metadataFingerprint: "sha256:b64323bf14246d63a2294794d5948da6abe130d8dd4a0c7ad5a4b3ac3bca11ae"
  },
  benchmarkId: "capstone_kb_expansion_retrieval_demo_pack",
  retrievalPackCases: 7,
  casesWithResults: 7,
  ragScopedGoldenContracts: 8,
  expectedSourceOverlap: { covered: 16, expected: 23, ratio: 16 / 23 },
  requestedDomainCoverage: { covered: 18, expected: 19, ratio: 18 / 19 },
  embeddingModel: "text-embedding-3-small",
  retrievalLimit: 5,
  interpretation:
    "Expected source IDs are coverage anchors, not a requirement that every listed source appear in the top-k results.",
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
    "Latest committed approved-fixture evidence. It is separate from canonical golden-case execution coverage.",
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
      domains: ["p5_js"]
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
      domains: ["three_js", "web_audio_api"]
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
      domains: ["glsl", "shadertoy", "three_js"]
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
      domains: ["hydra"]
    }
  ]
};
