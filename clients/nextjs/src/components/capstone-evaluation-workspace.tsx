"use client";

import { useMemo, useState } from "react";
import { Activity, ArrowRight, BarChart3, CheckCircle2, ChevronDown, Database, FlaskConical, Gauge, History, LineChart, ShieldCheck, Sparkles } from "lucide-react";
import {
  buildGoldenEvaluationDataset,
  createEvaluationCandidate,
  formatEvaluationCategory,
  formatEvaluationMetric,
  type EvaluationBenchmarkRun,
  type EvaluationCandidate,
  type EvaluationCaseResult,
  type EvaluationCategory,
  type EvaluationMetricResult,
  type EvaluationMetricStatus,
  type EvaluationRunRequest,
  type EvaluationScope
} from "@/lib/evaluation-benchmark";
import type { EvaluationHistoryRecord } from "@/lib/product-controls";
import type { ProductIntelligenceModel } from "@/lib/product-intelligence";
import {
  CANONICAL_RETRIEVAL_EVOLUTION,
  CURRENT_APPROVED_RAGAS_EVALUATED_AT,
  CURRENT_APPROVED_RAGAS_EVIDENCE,
  CURRENT_CANONICAL_RETRIEVAL_REPORT
} from "@/lib/current-ragas-evidence";

type Props = {
  history: EvaluationHistoryRecord[];
  model: ProductIntelligenceModel;
  onRun: (request: EvaluationRunRequest) => Promise<void>;
  running: boolean;
};

const categories: EvaluationCategory[] = ["rag", "creative_artifact", "workflow", "product_reliability"];
const scopes: { id: EvaluationScope; label: string; detail: string }[] = [
  { id: "full", label: "Full benchmark", detail: "All four categories and all canonical cases" },
  { id: "rag", label: "RAG / Retrieval", detail: "Retrieval contracts and optional RAGAS evidence" },
  { id: "creative_artifact", label: "Creative Artifacts", detail: "Artifact, preview, runtime, and quality evidence" },
  { id: "workflow", label: "Agents & Workflow", detail: "Route, nodes, latency, retries, and recovery" },
  { id: "product_reliability", label: "Product Reliability", detail: "Runtime, persistence, tests, and bug signals" },
  { id: "cases", label: "Selected cases", detail: "A focused subset of canonical cases" }
];

const retrievalMetrics = [
  { id: "faithfulness", label: "Faithfulness", detail: "How well the answer is grounded in the retrieved context." },
  { id: "answer_relevancy", label: "Answer Relevancy", detail: "How directly the answer addresses the benchmark question." },
  { id: "context_precision", label: "Context Precision", detail: "How much of the retrieved context is useful and correctly ranked." },
  { id: "context_recall", label: "Context Recall", detail: "Requires justified reference answers, which the approved fixture does not yet publish." },
  { id: "context_relevancy", label: "Context Relevancy", detail: "How consistently the retrieved context contains information useful for answering the query." }
] as const;

const currentRetrievalEvolution = CANONICAL_RETRIEVAL_EVOLUTION[CANONICAL_RETRIEVAL_EVOLUTION.length - 1];

const retrievalMetricDiagnostics = {
  faithfulness: {
    target: "At least 80% under the same approved fixture, evaluator model, embedding model, and RAGAS version.",
    rootCause: "Approved baseline answers contain compound claims whose supporting excerpts do not entail every clause.",
    improvement: "Generation now requests short, direct, excerpt-grounded claims, source IDs, explicit inference labels, and visible evidence gaps.",
    delta: "NOT_COMPARABLE — the comparable current-product provider rerun is blocked.",
    limitation: "The approved fixture remains the latest scored run; local knowledge-base excerpts cannot cross the provider boundary.",
    nextStep: "On the next eligible sanitized run, retain per-claim support verdicts and inspect unsupported clauses before making another product change."
  },
  answer_relevancy: {
    target: "At least 80% under the same approved fixture, evaluator model, embedding model, and RAGAS version.",
    rootCause: "Several approved baseline answers broaden the response beyond the question instead of leading with the requested answer.",
    improvement: "The product prompt contract now prioritizes the direct answer before supporting detail and separates evidence from inference.",
    delta: "NOT_COMPARABLE — no current-product run exists under the same scored contract.",
    limitation: "The impact is not claimed until the same approved evaluation contract can run in an eligible environment.",
    nextStep: "Retain RAGAS reverse-generated questions and noncommittal flags on the next comparable sanitized run, then diagnose only repeatable failure patterns."
  },
  context_precision: {
    target: "Maintain at least 80% under an unchanged, broader approved fixture; the current four-row result is 100%.",
    rootCause: "No material weakness was observed in the four-row approved fixture; useful evidence was already ranked first.",
    improvement: "No score-seeking change was made. Retrieval work targeted breadth and multi-domain coverage instead.",
    delta: "NOT_COMPARABLE — the approved baseline is 100%, but no eligible current-product rerun exists.",
    limitation: "Four sanitized rows are too narrow to establish broad precision across the full product. No reviewed graded relevance judgments exist for calibrating a similarity threshold.",
    nextStep: "Retain the current bounded semantic pool. Add a threshold only in a future reviewed graded-relevance benchmark, never to improve anchor overlap."
  },
  context_recall: {
    target: "MISSING_EVIDENCE — no numerical target is defensible until reviewed reference answers exist.",
    rootCause: "The approved fixture does not publish justified reference answers required for this metric.",
    improvement: "The gap remains visible as MISSING_EVIDENCE rather than being converted to zero or inferred from another metric.",
    delta: "NOT_COMPARABLE — the metric has no defensible score or prior value.",
    limitation: "Reference-answer evidence must be authored and reviewed for a future benchmark version, not retrofitted to raise this score.",
    nextStep: "Author and independently review reference answers for a future version before enabling Context Recall; keep the current baseline unchanged."
  },
  context_relevancy: {
    target: "At least 80% under the same approved fixture, evaluator model, embedding model, and RAGAS version.",
    rootCause: "The exact RAGAS cause is unproven because evaluator votes and explanations were not retained. Separately, the local benchmark exposed one-domain candidate-pool saturation.",
    improvement: "Explicit multi-domain requests now fan out across domains, merge candidates by distance, reject filtered navigation/index evidence, and prefer unseen sources in the final top-k.",
    delta: "NOT_COMPARABLE as RAGAS; current verified local domain coverage improved +57.89 pts under the fixed retrieval pack.",
    limitation: "Context Relevancy remains 68.75% until the approved provider evaluation can rerun comparably. A distinct but weaker semantic candidate can still displace a stronger repeated-source chunk.",
    nextStep: "On a future eligible sanitized run, retain evaluator relevance votes and explanations; continue only product-wide ranking changes supported by repeated graded failures."
  }
} as const;

type EvaluationProgress = {
  executionMode: "deterministic_local" | "provider_assisted";
  phase: "running" | "response_received" | "ended";
  previousRunId: string | null;
  selectedContracts: number;
};

const coverageRows = [
  ["RAG architecture", "Retrieval-required cases, source contracts, RAGAS precision/faithfulness/relevancy", "Measured only from recorded answer + context evidence"],
  ["Creative quality", "Prompt adherence, extraction, preview, technical and creative critique", "Rendered clarity and interaction stay missing without observation"],
  ["Multi-agent workflow", "Requested/resolved route, node completion, refinements, retries, latency", "Published workflow telemetry only"],
  ["Reliability", "Runtime health, persistence evidence, Product Bug signals, validation manifests", "Test suites are blocked unless explicitly executed"],
  ["Provider observability", "Provider/model, token usage, cost, duration, evaluator dataset", "Unavailable values are not estimated or fabricated"],
  ["Reproducibility", "Versioned prompt contracts, stable fingerprint, append-only local history", "Canonical prompts remain immutable"],
  ["Safety & privacy", "Explicit provider consent and approved public-safe RAGAS fixtures", "Raw local sessions never enter provider evaluation"]
] as const;

export function CapstoneEvaluationWorkspace({ history, model: _model, onRun, running }: Props) {
  const dataset = useMemo(buildGoldenEvaluationDataset, []);
  const benchmarkHistory = useMemo(
    () => history.flatMap((entry) => entry.benchmark ? [entry.benchmark] : []),
    [history]
  );
  const latest = benchmarkHistory.at(-1) ?? null;
  const [scope, setScope] = useState<EvaluationScope>("full");
  const [caseIds, setCaseIds] = useState<string[]>(() => dataset.cases.slice(0, 3).map((item) => item.id));
  const [allowProviderCalls, setAllowProviderCalls] = useState(false);
  const [approvedRagasDataset, setApprovedRagasDataset] = useState<"sanitized_public" | "redacted_public">("sanitized_public");
  const [runOpen, setRunOpen] = useState(false);
  const [categoryFilter, setCategoryFilter] = useState<"all" | EvaluationCategory>("all");
  const [statusFilter, setStatusFilter] = useState<"all" | EvaluationMetricStatus>("all");
  const [query, setQuery] = useState("");
  const [selectedHistoryId, setSelectedHistoryId] = useState<string | null>(null);
  const [candidates, setCandidates] = useState<EvaluationCandidate[]>([]);
  const [progress, setProgress] = useState<EvaluationProgress | null>(null);
  const selectedRun = benchmarkHistory.find((item) => item.id === selectedHistoryId) ?? latest;
  const ragCaseIds = dataset.cases.filter((item) => item.categories.includes("rag")).map((item) => item.id);
  const comparableHistory = selectedRun
    ? benchmarkHistory.filter((item) => isComparableHistoryRun(item, selectedRun))
    : [];
  const selectedCount = scope === "cases"
    ? caseIds.length
    : scope === "full"
      ? dataset.cases.length
      : dataset.cases.filter((item) => item.categories.includes(scope)).length;
  const selectedHasRag = scope === "full" || scope === "rag" || (scope === "cases" && dataset.cases.some((item) => caseIds.includes(item.id) && item.categories.includes("rag")));
  const isRunning = running || progress?.phase === "running";

  async function run() {
    const selectedCases = scope === "full"
      ? dataset.cases
      : scope === "cases"
        ? dataset.cases.filter((item) => caseIds.includes(item.id))
        : dataset.cases.filter((item) => item.categories.includes(scope));
    const executionMode = selectedHasRag && allowProviderCalls ? "provider_assisted" : "deterministic_local";
    setProgress({ executionMode, phase: "running", previousRunId: latest?.id ?? null, selectedContracts: selectedCases.length });
    try {
      await onRun({
        scope,
        caseIds: scope === "cases" ? caseIds : [],
        allowProviderCalls: selectedHasRag && allowProviderCalls,
        approvedRagasDataset
      });
      setProgress((current) => current ? { ...current, phase: "response_received" } : current);
    } catch {
      setProgress((current) => current ? { ...current, phase: "ended" } : current);
    }
  }

  const cases = (selectedRun?.caseResults ?? []).filter((item) => {
    if (categoryFilter !== "all" && !item.categories.includes(categoryFilter)) return false;
    if (statusFilter !== "all" && item.status !== statusFilter) return false;
    const haystack = `${item.title} ${item.domain} ${item.origins.join(" ")}`.toLowerCase();
    return haystack.includes(query.toLowerCase());
  });

  return (
    <section aria-label="Capstone evaluation workspace" className="capstoneEvaluation">
      <header className="capstoneEvaluationHero">
        <div>
          <span className="capstoneEyebrow"><FlaskConical aria-hidden="true" size={14} /> AI Engineering Lab</span>
          <h2>Measure retrieval. Diagnose weaknesses. Improve the real system.</h2>
          <p>Follow the engineering loop from fixed benchmark evidence to root cause, product change, and comparable rerun. RAG, creative, workflow, and product quality stay separate so every claim remains defensible.</p>
          <div className="capstoneHeroMeta">
            <span><strong>{dataset.cases.length}</strong> unique cases</span>
            <span><strong>{dataset.version}</strong> dataset</span>
            <span><strong>{dataset.fingerprint.slice(0, 10)}</strong> fingerprint</span>
          </div>
        </div>
        <div className="capstoneHeroAction">
          <span className="evaluationStatusPill" data-status="partial">Approved RAGAS baseline</span>
          <strong>61.44% approved-fixture RAGAS macro</strong>
          <small>Equal-weight mean · four metrics · four sanitized rows · not current-product quality</small>
          <button className="capstonePrimaryButton" disabled={isRunning} onClick={() => setRunOpen(true)} type="button">
            <Sparkles aria-hidden="true" size={16} /> {isRunning ? "Evaluation running…" : "Run Evaluation"}
          </button>
        </div>
      </header>

      <QualityBoundaryMap ragCaseCount={ragCaseIds.length} run={selectedRun} />

      <RetrievalEvaluation
        ragCaseIds={ragCaseIds}
        run={selectedRun}
      />

      <div className="evaluationCountStrip" aria-label="Latest result counts">
        <Count label="Pass" value={selectedRun?.counts.pass ?? 0} tone="pass" />
        <Count label="Partial" value={selectedRun?.counts.partial ?? 0} tone="partial" />
        <Count label="Fail" value={selectedRun?.counts.fail ?? 0} tone="fail" />
        <Count label="Blocked" value={selectedRun?.counts.blocked ?? 0} tone="blocked" />
        <Count label="Missing" value={selectedRun?.counts.missing ?? 0} tone="missing_evidence" />
        <Count label="Not run" value={selectedRun?.counts.notRun ?? dataset.cases.length} tone="not_run" />
      </div>

      <section aria-label="Evaluation categories" className="evaluationCategoryGrid">
        {categories.map((category) => {
          const result = selectedRun?.categoryResults.find((item) => item.category === category);
          return (
            <article className="evaluationCategoryCard" key={category}>
              <header><span>{formatQualityLane(category)}</span><Status status={result?.status ?? "not_run"} /></header>
              <div className="evaluationCategoryScore"><strong>{formatScore(result?.score ?? null)}</strong><span>target {Math.round((result?.target ?? .8) * 100)}%</span></div>
              <div className="evaluationScoreTrack"><i style={{ width: `${(result?.score ?? 0) * 100}%` }} /></div>
              <dl>
                <div><dt>Previous</dt><dd>{formatScore(result?.previousScore ?? null)}</dd></div>
                <div><dt>Delta</dt><dd>{formatDelta(result?.delta ?? null)}</dd></div>
                <div><dt>Gap</dt><dd>{result?.gap == null ? "—" : `${Math.round(result.gap * 100)} pts`}</dd></div>
                <div><dt>Evidence</dt><dd>{result ? `${result.measuredMetrics}/${result.applicableMetrics}` : "0/0"}</dd></div>
              </dl>
              <p>{result?.detail ?? "Run this category to create current, comparable evidence."}</p>
            </article>
          );
        })}
      </section>

      {runOpen ? (
        <section aria-label="Evaluation preflight" className="evaluationRunPanel">
          <header><div><span>Run preflight</span><strong>Choose evidence—not a marketing score</strong></div><button disabled={isRunning} onClick={() => setRunOpen(false)} type="button">Close</button></header>
          <div className="evaluationScopeGrid">
            {scopes.map((item) => <button aria-pressed={scope === item.id} key={item.id} onClick={() => setScope(item.id)} type="button"><strong>{item.label}</strong><span>{item.detail}</span></button>)}
          </div>
          {scope === "cases" ? (
            <details className="evaluationCasePicker" open>
              <summary>{caseIds.length} cases selected</summary>
              <div>{dataset.cases.map((item) => <label key={item.id}><input checked={caseIds.includes(item.id)} onChange={(event) => setCaseIds((current) => event.target.checked ? [...current, item.id] : current.filter((id) => id !== item.id))} type="checkbox" /><span><strong>{item.title}</strong><small>{item.domain} · {item.origins.join(", ")}</small></span></label>)}</div>
            </details>
          ) : null}
          <div className="evaluationPreflightGrid">
            <Preflight label="Execution" value="Deterministic local benchmark" detail="Always available; no provider call" />
            <Preflight label="Selection" value={`${selectedCount} canonical contracts`} detail={`${scope.replace(/_/g, " ")} scope · current local snapshot is analyzed`} />
            <Preflight label="History" value="Append locally" detail="Up to 24 run records survive reload" />
            <Preflight label="Cost" value={allowProviderCalls ? "Estimate unavailable" : "$0 provider evaluation"} detail={allowProviderCalls ? "Actual provider usage is reported when published" : "Local evidence only"} />
          </div>
          {selectedHasRag ? (
            <div className="evaluationProviderGate">
              <ShieldCheck aria-hidden="true" size={20} />
              <div><strong>Provider-assisted RAGAS is optional</strong><p>Only a committed public-safe benchmark fixture may leave the app. Raw local sessions are never eligible.</p></div>
              <select aria-label="Approved RAGAS dataset" disabled={!allowProviderCalls} onChange={(event) => setApprovedRagasDataset(event.target.value as typeof approvedRagasDataset)} value={approvedRagasDataset}><option value="sanitized_public">Sanitized public fixture</option><option value="redacted_public">Redacted public fixture</option></select>
              <label><input checked={allowProviderCalls} onChange={(event) => setAllowProviderCalls(event.target.checked)} type="checkbox" /> I explicitly authorize evaluator provider calls for this approved fixture.</label>
            </div>
          ) : null}
          {progress ? <EvaluationRunProgress datasetVersion={dataset.version} progress={progress} run={latest} /> : null}
          <footer><span>{selectedCount === 0 ? "Select at least one case." : allowProviderCalls ? "Provider call authorized for approved RAGAS fixture." : "No evaluator provider will be called."}</span><button className="capstonePrimaryButton" disabled={isRunning || selectedCount === 0} onClick={() => void run()} type="button">{isRunning ? "Running…" : `Run ${selectedCount} case${selectedCount === 1 ? "" : "s"}`}</button></footer>
        </section>
      ) : null}

      <section aria-label="Evaluation results" className="evaluationResultsSection">
        <SectionHeading icon={<BarChart3 aria-hidden="true" size={16} />} eyebrow="Results" title="Measured outcomes and evidence gaps" detail="Every category keeps its own metric family, threshold, and evidence boundary." />
        <div className="evaluationTrendGrid">
          {categories.map((category) => <Trend key={category} category={category} runs={comparableHistory} />)}
        </div>
        <div className="evaluationFilters">
          <input aria-label="Filter evaluation cases" onChange={(event) => setQuery(event.target.value)} placeholder="Filter by case, domain, or source…" value={query} />
          <select aria-label="Filter by category" onChange={(event) => setCategoryFilter(event.target.value as typeof categoryFilter)} value={categoryFilter}><option value="all">All categories</option>{categories.map((item) => <option key={item} value={item}>{formatEvaluationCategory(item)}</option>)}</select>
          <select aria-label="Filter by status" onChange={(event) => setStatusFilter(event.target.value as typeof statusFilter)} value={statusFilter}><option value="all">All statuses</option>{["pass", "partial", "fail", "blocked", "missing_evidence", "not_run"].map((item) => <option key={item} value={item}>{item.replace(/_/g, " ")}</option>)}</select>
        </div>
        {selectedRun ? <div className="evaluationCaseTable">{cases.map((item) => <CaseRow caseResult={item} key={item.caseId} onCandidate={(caseResult) => { const candidate = createEvaluationCandidate({ caseResult }); if (candidate) setCandidates((current) => [...current, candidate]); }} />)}{cases.length === 0 ? <p className="evaluationEmpty">No cases match the active filters.</p> : null}</div> : <EmptyResults />}
      </section>

      {candidates.length ? <section aria-label="Improvement candidates" className="evaluationCandidates"><SectionHeading icon={<Sparkles aria-hidden="true" size={16} />} eyebrow="Improve" title="Non-destructive prompt candidates" detail="The canonical prompt stays unchanged; candidates begin with no score or delta until rerun." />{candidates.map((item) => <article key={item.id}><div><span>Original</span><p>{item.originalPrompt}</p><strong>{formatScore(item.baselineScore)}</strong></div><div><span>Candidate</span><p>{item.candidatePrompt}</p><strong>Pending rerun · delta —</strong></div></article>)}</section> : null}

      <section aria-label="Evaluation history and trends" className="evaluationHistorySection">
        <SectionHeading icon={<History aria-hidden="true" size={16} />} eyebrow="History" title="Comparable local runs" detail="Deltas appear only when dataset fingerprint, scope, selected cases, and metrics are comparable." />
        {benchmarkHistory.length ? <div className="evaluationRunHistory">{[...benchmarkHistory].reverse().map((run) => <button aria-pressed={selectedRun?.id === run.id} key={run.id} onClick={() => setSelectedHistoryId(run.id)} type="button"><Status status={statusTone(run.statusLabel)} /><span><strong>{run.scope.replace(/_/g, " ")}</strong><small>{formatDate(run.completedAt)} · {run.executedCases}/{run.selectedCases} cases</small></span><span><strong>{run.provider ?? "Local only"}</strong><small>{run.totalTokens == null ? "Tokens unavailable" : `${run.totalTokens.toLocaleString()} tokens`} · {run.estimatedCost == null ? "Cost unavailable" : `${run.currency} ${run.estimatedCost.toFixed(4)}`}</small></span></button>)}</div> : <p className="evaluationEmpty">No rich benchmark run is stored yet. Existing legacy attempts remain preserved but are not comparable.</p>}
      </section>

      <section aria-label="Capstone evaluation mapping" className="evaluationCoverage">
        <SectionHeading icon={<CheckCircle2 aria-hidden="true" size={16} />} eyebrow="Capstone coverage" title="What each evaluation claim is built from" detail="The matrix separates measurable product evidence from checks that require another environment or human observation." />
        <div className="evaluationCoverageTable"><div className="evaluationCoverageHead"><span>Capability</span><span>Evidence</span><span>Truth boundary</span></div>{coverageRows.map((row) => <div key={row[0]}><strong>{row[0]}</strong><span>{row[1]}</span><small>{row[2]}</small></div>)}</div>
      </section>

      <details className="evaluationMethodology"><summary><ChevronDown aria-hidden="true" size={16} /> Methodology, scoring, and limitations</summary><div><p><strong>Golden dataset.</strong> {dataset.rawSourceCount} product-authored records are normalized into {dataset.cases.length} unique prompt contracts; {dataset.duplicateCount} aliases are deduplicated. IDs and a deterministic fingerprint make changes visible.</p><p><strong>Metric separation.</strong> RAGAS metrics apply only to recorded answers with contexts. Product-specific creative, workflow, runtime, and persistence signals retain their own labels. No cross-category global score is calculated because these lanes measure different constructs.</p><p><strong>Missing evidence.</strong> BLOCKED_BY_EXECUTION_ENVIRONMENT means a required evaluator, credential, network, or test runner was unavailable. MISSING_EVIDENCE means the current product did not publish defensible proof. Neither is converted to zero.</p><p><strong>Known limits.</strong> Context recall has no justified reference answers in the approved fixtures. Visual clarity and interaction success need rendered or human evidence. Historic QA can inform review but never becomes a current run result.</p></div></details>
    </section>
  );
}

function Count({ label, tone, value }: { label: string; tone: EvaluationMetricStatus; value: number }) { return <span data-status={tone}><strong>{value}</strong>{label}</span>; }
function Status({ status }: { status: EvaluationMetricStatus }) { return <span className="evaluationStatusPill" data-status={status}>{status.replace(/_/g, " ")}</span>; }
function Preflight({ detail, label, value }: { detail: string; label: string; value: string }) { return <div><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>; }
function SectionHeading({ detail, eyebrow, icon, title }: { detail: string; eyebrow: string; icon: React.ReactNode; title: string }) { return <header className="evaluationSectionHeading"><span>{icon}{eyebrow}</span><strong>{title}</strong><p>{detail}</p></header>; }

function formatQualityLane(category: EvaluationCategory) {
  if (category === "rag") return "Retrieval Quality";
  if (category === "creative_artifact") return "Creative Quality";
  if (category === "workflow") return "Workflow Quality";
  return "Product Reliability";
}

function QualityBoundaryMap({ ragCaseCount, run }: { ragCaseCount: number; run: EvaluationBenchmarkRun | null }) {
  const approvedScores = ["faithfulness", "answer_relevancy", "context_precision", "context_relevancy"].flatMap((id) => {
    const score = CURRENT_APPROVED_RAGAS_EVIDENCE.metricScores[id];
    return score == null ? [] : [score];
  });
  const approvedRetrievalScore = approvedScores.length
    ? approvedScores.reduce((sum, score) => sum + score, 0) / approvedScores.length
    : null;
  const categorySignal = (category: EvaluationCategory) => run?.categoryResults.find((result) => result.category === category)?.score ?? null;
  const lanes = [
    {
      classification: "MEASURED BASELINE",
      detail: "Equal-weight mean of four supported RAGAS dimensions on the committed public-safe fixture only.",
      label: "Retrieval Quality",
      value: formatPreciseScore(approvedRetrievalScore)
    },
    {
      classification: "SEPARATE LANE",
      detail: "Objective artifact evidence stays here; artistic judgement is SUBJECTIVE and excluded from aggregate scoring.",
      label: "Creative Quality",
      value: categorySignal("creative_artifact") == null ? "Not measured" : formatScore(categorySignal("creative_artifact"))
    },
    {
      classification: "SEPARATE LANE",
      detail: "Route, node, retry, recovery, and latency evidence is not mixed with Retrieval Quality.",
      label: "Workflow Quality",
      value: categorySignal("workflow") == null ? "Not measured" : formatScore(categorySignal("workflow"))
    },
    {
      classification: "SEPARATE LANE",
      detail: "Runtime, persistence, validation, and bug evidence remains independent from AI answer quality.",
      label: "Product Reliability",
      value: categorySignal("product_reliability") == null ? "Not measured" : formatScore(categorySignal("product_reliability"))
    },
    {
      classification: "COVERAGE ONLY",
      detail: `Current verified: ${CURRENT_CANONICAL_RETRIEVAL_REPORT.casesWithResults}/${CURRENT_CANONICAL_RETRIEVAL_REPORT.retrievalPackCases} canonical retrieval-pack queries returned results; 0/${ragCaseCount} golden RAG contracts have exact end-to-end RAGAS evidence. Coverage is not quality.`,
      label: "Benchmark Coverage",
      value: `${CURRENT_CANONICAL_RETRIEVAL_REPORT.casesWithResults}/${CURRENT_CANONICAL_RETRIEVAL_REPORT.retrievalPackCases} retrieval queries`
    },
    {
      classification: "COVERAGE ONLY",
      detail: "Four of five requested Retrieval dimensions are measured; missing evidence is not a zero.",
      label: "Evidence Coverage",
      value: "4/5 RAG metrics"
    }
  ] as const;

  return (
    <section aria-label="Evaluation score boundaries" className="evaluationQualityMap">
      <header><div><span><ShieldCheck aria-hidden="true" size={14} /> Score boundaries</span><strong>Six independent signals. No global score.</strong></div><small>Quality and coverage answer different questions and never share a denominator.</small></header>
      <div>
        {lanes.map((lane) => <article key={lane.label}><span>{lane.label}</span><strong>{lane.value}</strong><small>{lane.classification}</small><p>{lane.detail}</p></article>)}
      </div>
    </section>
  );
}

function RetrievalEvaluation({ ragCaseIds, run }: { ragCaseIds: string[]; run: EvaluationBenchmarkRun | null }) {
  const supportedMetricIds = ["faithfulness", "answer_relevancy", "context_precision", "context_relevancy"] as const;
  const runHasMeasuredRagas = supportedMetricIds.some((id) => run?.ragas.metricScores[id] != null);
  const evidence = runHasMeasuredRagas && run ? run.ragas : CURRENT_APPROVED_RAGAS_EVIDENCE;
  const usingCommittedBaseline = !runHasMeasuredRagas;
  const requestedSupportedIds = supportedMetricIds.filter((id) => evidence.metrics.includes(id));
  const supportedScores = requestedSupportedIds.flatMap((id) => {
    const score = evidence.metricScores[id] ?? findRetrievalMetric(run, id)?.score ?? null;
    return score == null ? [] : [score];
  });
  const retrievalScore = supportedScores.length
    ? supportedScores.reduce((total, score) => total + score, 0) / supportedScores.length
    : null;
  const evidenceCoverage = supportedScores.length / retrievalMetrics.length;
  const productReliability = run?.categoryResults.find((result) => result.category === "product_reliability") ?? null;
  const retrievalStatus = retrievalScore == null
    ? ["blocked", "prepared", "not_requested"].includes(evidence.state) ? "blocked" : "missing_evidence"
    : scoreStatus(retrievalScore);

  return (
    <section aria-label="RAGAS retrieval evaluation" className="retrievalEvaluation">
      <header className="retrievalEvaluationHeader">
        <div>
          <span><Database aria-hidden="true" size={15} /> Real RAGAS evaluation</span>
          <h3>Retrieval quality, with every evidence boundary visible.</h3>
          <p>The retrieval score uses only supported RAGAS measurements from the approved public-safe fixture. Missing or unsupported metrics are shown explicitly and never entered as zero.</p>
        </div>
        <div className="retrievalExecutionBadge">
          <Status status={retrievalStatus} />
          <strong>{formatRagasDataset(evidence.datasetId)}</strong>
          <small>{usingCommittedBaseline ? `latest scored approved baseline · ${formatDate(CURRENT_APPROVED_RAGAS_EVALUATED_AT)}${run ? ` · current attempt ${run.ragas.state.replace(/_/g, " ")}` : ""}` : `${evidence.state.replace(/_/g, " ")} · ${evidence.provider ?? "provider not reported"}`}</small>
        </div>
      </header>

      <div className="retrievalScoreSummary">
        <article className="retrievalOverallScore" data-status={retrievalStatus}>
          <span><Gauge aria-hidden="true" size={15} /> Approved-fixture Overall Retrieval Score</span>
          <strong>{formatPreciseScore(retrievalScore)}</strong>
          <div className="evaluationScoreTrack"><i style={{ width: `${(retrievalScore ?? 0) * 100}%` }} /></div>
          <p>{supportedScores.length}/{requestedSupportedIds.length} requested, supported RAGAS dimensions measured. {supportedScores.length < requestedSupportedIds.length ? "The score is provisional until supported evidence is complete." : "Each measured dimension contributes equally."} {usingCommittedBaseline ? "Shown from the latest committed approved-fixture run." : "Shown from this workspace run."}</p>
        </article>
        <EvaluationBoundaryCard
          detail={`${CURRENT_CANONICAL_RETRIEVAL_REPORT.casesWithResults}/${CURRENT_CANONICAL_RETRIEVAL_REPORT.retrievalPackCases} canonical retrieval-pack queries returned ranked results; 0/${ragCaseIds.length} exact golden contracts have end-to-end RAGAS evidence`}
          label="Benchmark coverage"
          value={`${CURRENT_CANONICAL_RETRIEVAL_REPORT.casesWithResults}/${CURRENT_CANONICAL_RETRIEVAL_REPORT.retrievalPackCases} retrieval queries`}
        />
        <EvaluationBoundaryCard
          caution
          detail={`${supportedScores.length}/${retrievalMetrics.length} requested RAG metric dimensions measured. This is not an overall product or project score.`}
          label="Evidence coverage"
          value={formatScore(evidenceCoverage)}
        />
        <EvaluationBoundaryCard
          detail={productReliability?.score == null ? "Not measured in the selected run. Runtime, persistence, validation, and bug evidence remain a separate quality lane." : "Measured from product reliability evidence only; never included in the Retrieval Quality baseline."}
          label="Product Reliability"
          value={productReliability?.score == null ? "Not measured" : formatScore(productReliability.score)}
        />
      </div>

      <ScoreCompositionContract evidence={evidence} />

      <RetrievalEngineeringEvolution />

      <div aria-label="RAGAS metric scores" className="retrievalMetricGrid">
        {retrievalMetrics.map((definition) => {
          const metric = findRetrievalMetric(run, definition.id);
          const score = evidence.metricScores[definition.id] ?? metric?.score ?? null;
          const status = score == null
            ? definition.id === "context_recall"
              ? "missing_evidence"
              : metric?.status ?? (evidence.state === "blocked" ? "blocked" : "missing_evidence")
            : scoreStatus(score);
          const value = score == null
            ? status === "blocked" ? "BLOCKED" : status === "missing_evidence" ? "MISSING EVIDENCE" : "NOT RUN"
            : formatPreciseScore(score);
          const evidenceDetail = score != null
            ? `Real RAGAS score across ${evidence.resultRows} approved retrieval rows.`
            : metric?.detail ?? "No defensible measurement is published for this dimension.";
          return (
            <article data-status={status} key={definition.id}>
              <header><span>{definition.label}</span><Status status={status} /></header>
              <strong>{value}</strong>
              <p>{definition.detail}</p>
              <small>{evidenceDetail}</small>
            </article>
          );
        })}
      </div>

      <MetricDiagnostics />

      <div className="retrievalEvidenceGrid">
        <EvidenceLineage evidence={evidence} />
        <RetrievalExecutionTimeline />
      </div>

      <footer>
        <ShieldCheck aria-hidden="true" size={16} />
        <span><strong>Why this score is trustworthy.</strong> {evidence.resultRows} evaluated rows from {evidence.eligibleSamples}/{evidence.totalSamples} eligible fixture samples; raw workspace sessions are excluded. {evidence.model ?? "Evaluator model not reported"} · RAGAS {evidence.ragasVersion ?? "version unavailable"}. The separate current verified canonical retrieval report returned results for {CURRENT_CANONICAL_RETRIEVAL_REPORT.casesWithResults}/{CURRENT_CANONICAL_RETRIEVAL_REPORT.retrievalPackCases} queries, with {formatPreciseScore(currentRetrievalEvolution.sourceCoverage.ratio)} substantive expected-source overlap and {formatPreciseScore(currentRetrievalEvolution.domainCoverage.ratio)} requested-domain coverage. {CURRENT_CANONICAL_RETRIEVAL_REPORT.interpretation} {CURRENT_CANONICAL_RETRIEVAL_REPORT.qualityInterpretation} Context Recall stays visible until justified reference answers exist.</span>
      </footer>
    </section>
  );
}

function RetrievalEngineeringEvolution() {
  const baseline = CANONICAL_RETRIEVAL_EVOLUTION[0];
  const current = currentRetrievalEvolution;
  const sourceDelta = current.sourceCoverage.ratio - baseline.sourceCoverage.ratio;
  const domainDelta = current.domainCoverage.ratio - baseline.domainCoverage.ratio;
  return (
    <section aria-label="Retrieval engineering evolution" className="retrievalEvolutionPanel">
      <header>
        <div><span><LineChart aria-hidden="true" size={14} /> Engineering evolution</span><strong>Real retrieval gains, kept separate from RAGAS</strong></div>
        <small>Current verified · fixed canonical retrieval pack · same 7 queries · top 5 results</small>
      </header>
      <div className="retrievalRagasComparison">
        <div><span>Approved-fixture RAGAS</span><strong>61.44%</strong><small>Latest scored baseline</small></div>
        <ArrowRight aria-hidden="true" size={17} />
        <div data-status="blocked"><span>Current-product RAGAS rerun</span><strong>BLOCKED_BY_EXECUTION_ENVIRONMENT</strong><small>Local KB excerpts cannot cross the provider boundary</small></div>
        <div><span>Comparable RAGAS delta</span><strong>—</strong><small>Withheld, never inferred from local coverage</small></div>
      </div>
      <div className="retrievalEvolutionStages">
        {CANONICAL_RETRIEVAL_EVOLUTION.map((stage, index) => {
          const sourceRatio = stage.sourceCoverage.ratio;
          const domainRatio = stage.domainCoverage.ratio;
          return (
            <article key={stage.label}>
              <header><span>0{index + 1}</span><strong>{stage.label}</strong></header>
              <p>{stage.finding}</p>
              <div><span>Expected-source overlap</span><strong>{stage.sourceCoverage.covered}/{stage.sourceCoverage.expected} · {formatScore(sourceRatio)}</strong><i><b style={{ width: `${sourceRatio * 100}%` }} /></i></div>
              <div><span>Requested-domain coverage</span><strong>{stage.domainCoverage.covered}/{stage.domainCoverage.expected} · {formatScore(domainRatio)}</strong><i><b style={{ width: `${domainRatio * 100}%` }} /></i></div>
            </article>
          );
        })}
      </div>
      <footer>
        <span><strong>{formatPreciseDelta(sourceDelta)}</strong> expected-source overlap</span>
        <span><strong>{formatPreciseDelta(domainDelta)}</strong> requested-domain coverage</span>
        <small>Coverage anchors describe retrieval breadth. They are not answer quality, evidence completeness, or an overall product score. {CURRENT_CANONICAL_RETRIEVAL_REPORT.qualityInterpretation}</small>
        <small className="retrievalReportProof">Machine evidence: {CURRENT_CANONICAL_RETRIEVAL_REPORT.reportArtifact} · {CURRENT_CANONICAL_RETRIEVAL_REPORT.kbSnapshot.recordCount.toLocaleString()} chunks · KB {shortFingerprint(CURRENT_CANONICAL_RETRIEVAL_REPORT.kbSnapshot.metadataFingerprint)} · selection {shortFingerprint(CURRENT_CANONICAL_RETRIEVAL_REPORT.selectionFingerprint)}</small>
      </footer>
    </section>
  );
}

function ScoreCompositionContract({ evidence }: { evidence: EvaluationBenchmarkRun["ragas"] }) {
  const includedMetricIds = ["faithfulness", "answer_relevancy", "context_precision", "context_relevancy"] as const;
  const exclusions = [
    {
      classification: "MISSING_EVIDENCE",
      detail: "The approved fixture has no justified reference answers, so Context Recall has no defensible value and contributes neither zero nor weight.",
      label: "Context Recall"
    },
    {
      classification: "BLOCKED_BY_EXECUTION_ENVIRONMENT",
      detail: "Current local knowledge-base excerpts cannot cross the evaluator-provider boundary, so no current-product provider RAGAS score is claimed.",
      label: "Current-product provider RAGAS rerun"
    },
    {
      classification: "NOT_COMPARABLE",
      detail: "The approved provider fixture score and the current local retrieval-coverage report use different evidence, methods, and contracts; their delta is withheld.",
      label: "Approved baseline versus current local retrieval"
    },
    {
      classification: "SUBJECTIVE",
      detail: "Artistic merit, visual clarity, and interaction quality require rendered or human observation and never enter the Retrieval Quality baseline.",
      label: "Artistic and visual judgement"
    },
    {
      classification: "NOT_COMPARABLE",
      detail: "Retrieval, creative, workflow, reliability, benchmark coverage, and evidence coverage measure different constructs; a global score is not calculated.",
      label: "Cross-category aggregate"
    }
  ] as const;

  return (
    <details className="retrievalScoreContract" open>
      <summary><div><span><ShieldCheck aria-hidden="true" size={14} /> Retrieval score contract</span><strong>Exactly what enters the approved baseline</strong></div><small>Only measurable · reproducible · defensible · comparable evidence</small><ChevronDown aria-hidden="true" size={14} /></summary>
      <div>
        <section aria-label="Included retrieval metrics">
          <header><span>Included</span><strong>Equal weight inside Retrieval Quality only</strong></header>
          {includedMetricIds.map((metricId) => {
            const metric = retrievalMetrics.find((item) => item.id === metricId);
            const score = evidence.metricScores[metricId] ?? null;
            return (
              <article key={metricId}>
                <div><strong>{metric?.label}</strong><small>{score == null ? "No published comparable value" : "Same approved fixture, evaluator, embedding, and RAGAS contract"}</small></div>
                <span>{formatPreciseScore(score)}</span>
                <em data-classification={score == null ? "MISSING_EVIDENCE" : "INCLUDED"}>{score == null ? "MISSING_EVIDENCE" : "INCLUDED"}</em>
              </article>
            );
          })}
        </section>
        <section aria-label="Excluded evaluation evidence">
          <header><span>Excluded</span><strong>Visible, justified, never coerced to zero</strong></header>
          {exclusions.map((exclusion) => (
            <article key={exclusion.label}>
              <div><strong>{exclusion.label}</strong><small>{exclusion.detail}</small></div>
              <em data-classification={exclusion.classification}>{exclusion.classification}</em>
            </article>
          ))}
        </section>
      </div>
    </details>
  );
}

function MetricDiagnostics() {
  return (
    <section aria-label="Metric engineering diagnostics" className="retrievalDiagnostics">
      <header><div><span><Activity aria-hidden="true" size={14} /> Metric diagnostics</span><strong>What the baseline exposed—and what changed</strong></div><small>Open a metric for root cause, product change, delta, and remaining limit.</small></header>
      <div>
        {retrievalMetrics.map((metric) => {
          const diagnosis = retrievalMetricDiagnostics[metric.id];
          const baselineScore = CURRENT_APPROVED_RAGAS_EVIDENCE.metricScores[metric.id] ?? null;
          return (
            <details key={metric.id} open={metric.id === "faithfulness" || metric.id === "answer_relevancy" || metric.id === "context_relevancy"}>
              <summary><span>{metric.label}</span><strong>{formatPreciseScore(baselineScore)}</strong><ChevronDown aria-hidden="true" size={14} /></summary>
              <dl>
                <div><dt>Current approved score</dt><dd>{baselineScore == null ? "MISSING_EVIDENCE" : formatPreciseScore(baselineScore)}</dd></div>
                <div><dt>Target</dt><dd>{diagnosis.target}</dd></div>
                <div><dt>Root cause</dt><dd>{diagnosis.rootCause}</dd></div>
                <div><dt>Product improvement</dt><dd>{diagnosis.improvement}</dd></div>
                <div><dt>Comparable benchmark delta</dt><dd>{diagnosis.delta}</dd></div>
                <div><dt>Remaining limitation</dt><dd>{diagnosis.limitation}</dd></div>
                <div><dt>Recommended next engineering step</dt><dd>{diagnosis.nextStep}</dd></div>
              </dl>
            </details>
          );
        })}
      </div>
    </section>
  );
}

function EvidenceLineage({ evidence }: { evidence: EvaluationBenchmarkRun["ragas"] }) {
  return (
    <details className="retrievalLineage" open>
      <summary><span><Database aria-hidden="true" size={14} /> Evidence lineage</span><small>Run, contract, provider, and source trace</small><ChevronDown aria-hidden="true" size={14} /></summary>
      <dl>
        <div><dt>Run</dt><dd>{evidence.runId ?? "Unavailable"}</dd></div>
        <div><dt>Evaluated</dt><dd>{evidence.evaluatedAt ? formatDate(evidence.evaluatedAt) : "Unavailable"}</dd></div>
        <div><dt>Dataset</dt><dd>{evidence.datasetVersion || evidence.datasetId} · {evidence.privacyClass.replace(/_/g, " ")}</dd></div>
        <div><dt>Evaluation stack</dt><dd>{evidence.provider ?? "Provider unavailable"} · {evidence.model ?? "model unavailable"} · RAGAS {evidence.ragasVersion ?? "version unavailable"}</dd></div>
        <div><dt>Embedding</dt><dd>{evidence.embeddingModel ?? "Unavailable"}</dd></div>
        <div><dt>Metric contract</dt><dd>{evidence.metricContract ?? "Unavailable"} · {evidence.resultRows} rows · {evidence.metricFailures} failures</dd></div>
      </dl>
      {evidence.caseRows.length ? (
        <div className="retrievalLineageRows">
          {evidence.caseRows.map((row) => <article key={row.sampleId}><strong>{row.sampleId}</strong><span>{row.domains.join(" · ") || "Domains unavailable"}</span><small>{row.sourceIds.join(" · ") || "Source IDs unavailable"}</small></article>)}
        </div>
      ) : <p>No per-row source lineage was published for this selected run.</p>}
    </details>
  );
}

function RetrievalExecutionTimeline() {
  return (
    <section aria-label="Retrieval evaluation execution timeline" className="retrievalTimeline">
      <header><span><Activity aria-hidden="true" size={14} /> Execution timeline</span><strong>Current engineering loop</strong></header>
      <ol>
        <li data-status="pass"><i /><div><span>Current canonical retrieval run</span><strong>COMPLETE · 7/7 queries returned results</strong><small>Local excerpts stayed local; public query embedding used the configured embedding provider.</small></div></li>
        <li data-status="pass"><i /><div><span>Truthful evidence correction</span><strong>19/23 RAW ANCHORS → 15/23 SUBSTANTIVE → 16/23 FINAL</strong><small>False heading/index coverage was removed before bounded headroom recovered useful Three.js guidance.</small></div></li>
        <li data-status="pass"><i /><div><span>Current retrieval evidence</span><strong>16/23 SUBSTANTIVE ANCHORS · 18/19 DOMAINS</strong><small>{CURRENT_CANONICAL_RETRIEVAL_REPORT.qualityInterpretation}</small></div></li>
        <li data-status="blocked"><i /><div><span>Current-product RAGAS rerun</span><strong>BLOCKED_BY_EXECUTION_ENVIRONMENT</strong><small>Local knowledge-base excerpts are not eligible to cross the provider boundary.</small></div></li>
        <li data-status="missing_evidence"><i /><div><span>Context Recall</span><strong>MISSING_EVIDENCE</strong><small>The approved fixture has no justified reference answers for this metric.</small></div></li>
      </ol>
    </section>
  );
}

function EvaluationBoundaryCard({ caution = false, detail, label, value }: { caution?: boolean; detail: string; label: string; value: string }) {
  return <article className="evaluationBoundaryCard" data-caution={caution}><span>{label}</span><strong>{value}</strong><p>{detail}</p></article>;
}

function EvaluationRunProgress({ datasetVersion, progress, run }: { datasetVersion: string; progress: EvaluationProgress; run: EvaluationBenchmarkRun | null }) {
  const publishedRun = run && run.id !== progress.previousRunId ? run : null;
  const providerAssisted = progress.executionMode === "provider_assisted";
  const total = providerAssisted ? publishedRun?.ragas.totalSamples || 4 : 1;
  const completed = providerAssisted
    ? publishedRun?.ragas.state === "completed" ? publishedRun.ragas.resultRows : 0
    : publishedRun ? 1 : 0;
  const remaining = Math.max(0, total - completed);
  const exactProgressAvailable = publishedRun != null;
  const percent = exactProgressAvailable && total ? Math.round(completed / total * 100) : null;
  const currentCase = providerAssisted
    ? "Approved fixture rows · per-row state unavailable"
    : "Current workspace snapshot";
  const currentMetric = providerAssisted
    ? "RAGAS metric batch · per-metric state unavailable"
    : "Local product evidence contracts";
  const phaseLabel = publishedRun
    ? `${publishedRun.ragas.state.replace(/_/g, " ")} · stored result published`
    : progress.phase === "running"
      ? providerAssisted ? "Authorized provider batch in progress" : "Local snapshot analysis in progress · provider off"
      : progress.phase === "response_received" ? "Batch response received · awaiting stored result" : "Run ended · inspect the published result state";
  return (
    <section aria-label="Live evaluation progress" aria-live="polite" className="evaluationLiveProgress">
      <header><span><Activity aria-hidden="true" size={15} /> Live run</span><strong>{percent == null ? "Estimated progress: indeterminate" : `${percent}% confirmed`}</strong></header>
      <div aria-label="Estimated evaluation progress" aria-valuemax={100} aria-valuemin={0} {...(percent == null ? {} : { "aria-valuenow": percent })} className="evaluationProgressTrack" data-indeterminate={percent == null} role="progressbar"><i style={{ width: percent == null ? "28%" : `${percent}%` }} /></div>
      <dl>
        <div><dt>Current benchmark</dt><dd>{datasetVersion} · {progress.selectedContracts} contracts enumerated</dd></div>
        <div><dt>Current case</dt><dd>{currentCase}</dd></div>
        <div><dt>Current metric</dt><dd>{currentMetric}</dd></div>
        <div><dt>Completed / remaining</dt><dd>{completed} confirmed / {remaining} unresolved {providerAssisted ? "fixture rows" : "snapshot"}</dd></div>
        <div><dt>Execution state</dt><dd>{phaseLabel}</dd></div>
      </dl>
      <small>The synchronous evaluator exposes no per-case callback, so progress stays indeterminate until a stored response arrives. Golden contracts are enumerated for coverage; they are not presented as 31 executed generations.</small>
    </section>
  );
}

function findRetrievalMetric(run: EvaluationBenchmarkRun | null, metricId: string): EvaluationMetricResult | null {
  if (!run) return null;
  const matches = run.caseResults.flatMap((caseResult) => caseResult.metrics).filter((metric) => metric.kind === "ragas" && metric.id === metricId);
  return matches.find((metric) => metric.score != null) ?? matches.find((metric) => metric.status !== "not_run") ?? null;
}

function isComparableHistoryRun(run: EvaluationBenchmarkRun, anchor: EvaluationBenchmarkRun) {
  return run.datasetFingerprint === anchor.datasetFingerprint &&
    run.scope === anchor.scope &&
    run.selectedCaseIds.join("|") === anchor.selectedCaseIds.join("|") &&
    run.ragas.datasetId === anchor.ragas.datasetId &&
    run.ragas.datasetVersion === anchor.ragas.datasetVersion &&
    run.ragas.privacyClass === anchor.ragas.privacyClass &&
    [...run.ragas.metrics].sort().join("|") === [...anchor.ragas.metrics].sort().join("|") &&
    run.ragas.model === anchor.ragas.model &&
    run.ragas.provider === anchor.ragas.provider &&
    run.ragas.embeddingModel === anchor.ragas.embeddingModel &&
    run.ragas.ragasVersion === anchor.ragas.ragasVersion &&
    run.ragas.metricContract === anchor.ragas.metricContract;
}

function Trend({ category, runs }: { category: EvaluationCategory; runs: EvaluationBenchmarkRun[] }) {
  const points = runs.slice(-8).map((run) => run.categoryResults.find((item) => item.category === category)?.score ?? null);
  return <article><header><strong>{formatEvaluationCategory(category)}</strong><span>{points.filter((item) => item != null).length} measured</span></header><div>{points.length ? points.map((point, index) => <i aria-label={point == null ? "Missing evidence" : `${Math.round(point * 100)} percent`} key={index} style={{ height: `${point == null ? 6 : Math.max(8, point * 100)}%` }} data-missing={point == null} />) : <small>Run to start trend</small>}</div></article>;
}

function CaseRow({ caseResult, onCandidate }: { caseResult: EvaluationCaseResult; onCandidate: (value: EvaluationCaseResult) => void }) {
  return <details><summary><Status status={caseResult.status} /><span><strong>{caseResult.title}</strong><small>{caseResult.domain} · {caseResult.origins.join(", ")}</small></span><strong>{formatScore(caseResult.score)}</strong><ChevronDown aria-hidden="true" size={15} /></summary><div className="evaluationCaseDetail"><div className="evaluationExpected"><span>Expected</span><strong>{caseResult.expectedArtifact}</strong><p>{caseResult.previewContract}</p></div><div className="evaluationMetricList">{caseResult.metrics.map((metric) => <article key={`${caseResult.caseId}-${metric.id}`}><header><Status status={metric.status} /><strong>{formatEvaluationMetric(metric.id)}</strong><span>{formatScore(metric.score)}</span></header><p>{metric.detail}</p><small>{metric.kind === "ragas" ? "RAGAS" : "Product-specific"} · {metric.evidenceClass.replace(/_/g, " ")} · target {metric.target == null ? "n/a" : `${Math.round(metric.target * 100)}%`} · gap {metric.gap == null ? "n/a" : `${Math.round(metric.gap * 100)} pts`}</small></article>)}</div>{caseResult.recommendation ? <aside><div><span>Recommended next step</span><strong>{caseResult.recommendation.title}</strong><p>{caseResult.recommendation.detail}</p></div><button onClick={() => onCandidate(caseResult)} type="button">Create improved candidate</button></aside> : null}</div></details>;
}

function EmptyResults() { return <div className="evaluationEmptyState"><FlaskConical aria-hidden="true" size={24} /><strong>No benchmark result yet</strong><p>Run the deterministic local benchmark. Provider access is optional and affects only approved RAGAS evidence.</p></div>; }
function formatScore(value: number | null) { return value == null ? "—" : `${Math.round(value * 100)}%`; }
function formatPreciseScore(value: number | null) { return value == null ? "—" : `${(value * 100).toFixed(2)}%`; }
function formatDelta(value: number | null) { return value == null ? "—" : `${value > 0 ? "+" : ""}${Math.round(value * 100)} pts`; }
function formatPreciseDelta(value: number | null) { return value == null ? "—" : `${value > 0 ? "+" : ""}${(value * 100).toFixed(2)} pts`; }
function formatDate(value: string) { const date = new Date(value); return Number.isNaN(date.getTime()) ? value : date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }); }
function formatRagasDataset(value: string) { return value === "sanitized_public" ? "Sanitized public fixture" : value === "redacted_public" ? "Redacted public fixture" : value.replace(/_/g, " "); }
function scoreStatus(value: number): EvaluationMetricStatus { return value >= .8 ? "pass" : value >= .6 ? "partial" : "fail"; }
function shortFingerprint(value: string) { return value.replace(/^sha256:/, "").slice(0, 12); }
function statusTone(value: EvaluationBenchmarkRun["statusLabel"]): EvaluationMetricStatus { if (value === "Blocked") return "blocked"; if (value === "Needs Improvement") return "fail"; if (value === "Incomplete Evidence") return "missing_evidence"; return "pass"; }
