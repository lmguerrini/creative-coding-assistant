"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Activity, ArrowRight, BarChart3, CheckCircle2, ChevronDown, Database, FlaskConical, Gauge, History, LineChart, ShieldCheck, Sparkles } from "lucide-react";
import {
  buildGoldenEvaluationDataset,
  buildEvaluationBenchmarkRun,
  CURRENT_PRODUCT_RETRIEVAL_GOLDEN_CASE_IDS,
  createEvaluationCandidate,
  currentProductRetrievalScoreFromEvidence,
  formatEvaluationCategory,
  formatEvaluationMetric,
  matchesCurrentProductEvidenceIdentity,
  type EvaluationBenchmarkRun,
  type EvaluationCandidate,
  type EvaluationCaseResult,
  type EvaluationCategory,
  type EvaluationExecutionProgress,
  type EvaluationMetricResult,
  type EvaluationMetricStatus,
  type EvaluationProgressCallback,
  type EvaluationRunRequest,
  type EvaluationScope,
  type RagasExecutionEvidence
} from "@/lib/evaluation-benchmark";
import type { EvaluationHistoryRecord } from "@/lib/product-controls";
import type { ProductIntelligenceModel } from "@/lib/product-intelligence";
import {
  CURRENT_APPROVED_RAGAS_EVALUATED_AT,
  CURRENT_APPROVED_RAGAS_EVIDENCE,
  CURRENT_PRODUCT_RAGAS_EVIDENCE
} from "@/lib/current-ragas-evidence";
import {
  DashboardCallout,
  DashboardCardGrid,
  DashboardDisclosure,
  DashboardInfoCard,
  DashboardMetricGrid,
  DashboardSection,
  DashboardSectionHeader,
  DashboardTableFrame
} from "./dashboard-page-primitives";

type Props = {
  currentProductEvidence?: RagasExecutionEvidence | null;
  history: EvaluationHistoryRecord[];
  model: ProductIntelligenceModel;
  onRun: (
    request: EvaluationRunRequest,
    onProgress: EvaluationProgressCallback
  ) => Promise<void>;
  running: boolean;
};

const categories: EvaluationCategory[] = ["rag", "creative_artifact", "workflow", "product_reliability"];
const scopes: { id: EvaluationScope; label: string; detail: string }[] = [
  { id: "full", label: "Full evaluation", detail: "Seven canonical RAG cases plus current creative, workflow, and reliability workspace snapshots" },
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
  { id: "context_recall", label: "Context Recall", detail: "How completely retrieved evidence covers the independently reviewed reference answer." },
  { id: "context_relevancy", label: "Context Relevancy", detail: "How consistently the retrieved context contains information useful for answering the query." }
] as const;

const currentProductMetricIds = [
  "faithfulness",
  "answer_relevancy",
  "context_precision",
  "context_recall",
  "context_relevancy"
] as const;

const coverageRows = [
  ["RAG architecture", "Retrieval-required cases, source contracts, RAGAS precision/faithfulness/relevancy", "Measured only from recorded answer + context evidence"],
  ["Creative quality", "Prompt adherence, extraction, preview, technical and creative critique", "Rendered clarity and interaction stay missing without observation"],
  ["Multi-agent workflow", "Requested/resolved route, node completion, refinements, retries, latency", "Published workflow telemetry only"],
  ["Reliability", "Runtime health, persistence evidence, Product Bug signals, validation manifests", "Test suites are blocked unless explicitly executed"],
  ["Provider observability", "Provider/model, token usage, cost, duration, evaluator dataset", "Unavailable values are not estimated or fabricated"],
  ["Reproducibility", "Versioned prompt contracts, stable fingerprint, append-only local history", "Canonical prompts remain immutable"],
  ["Safety & privacy", "Explicit provider consent and public-only current-product benchmark evidence", "Raw local sessions and private KB content never enter provider evaluation"]
] as const;

export function CapstoneEvaluationWorkspace({
  currentProductEvidence = CURRENT_PRODUCT_RAGAS_EVIDENCE,
  history,
  model,
  onRun,
  running
}: Props) {
  const dataset = useMemo(buildGoldenEvaluationDataset, []);
  const committedCurrentProductRun = useMemo(() => {
    if (
      !currentProductEvidence?.timestamp ||
      currentProductRetrievalScoreFromEvidence(currentProductEvidence) == null
    ) return null;
    return buildEvaluationBenchmarkRun({
      model,
      now: new Date(currentProductEvidence.timestamp),
      ragas: currentProductEvidence,
      request: {
        scope: "rag",
        caseIds: [],
        allowProviderCalls: true,
        approvedRagasDataset: "sanitized_public"
      }
    });
  }, [currentProductEvidence, model]);
  const benchmarkHistory = useMemo(
    () => {
      const persisted = history.flatMap((entry) => entry.benchmark ? [entry.benchmark] : []);
      const combined = committedCurrentProductRun
        ? [...persisted.filter((run) => run.runId !== committedCurrentProductRun.runId), committedCurrentProductRun]
        : persisted;
      return combined.sort((left, right) => runTimestamp(left) - runTimestamp(right));
    },
    [committedCurrentProductRun, history]
  );
  const latestAttempt = benchmarkHistory.at(-1) ?? null;
  const latestCurrentProductAttempt = [...benchmarkHistory]
    .reverse()
    .find((run) => run.ragas.benchmarkMode === "current_product") ?? null;
  const currentRetrievalRun = [...benchmarkHistory]
    .reverse()
    .find(isFullyCompletedCurrentProductRun) ?? null;
  const [scope, setScope] = useState<EvaluationScope>("full");
  const [caseIds, setCaseIds] = useState<string[]>(() => dataset.cases.slice(0, 3).map((item) => item.id));
  const [allowProviderCalls, setAllowProviderCalls] = useState(false);
  const [runOpen, setRunOpen] = useState(false);
  const [categoryFilter, setCategoryFilter] = useState<"all" | EvaluationCategory>("all");
  const [statusFilter, setStatusFilter] = useState<"all" | EvaluationMetricStatus>("all");
  const [query, setQuery] = useState("");
  const [selectedHistoryId, setSelectedHistoryId] = useState<string | null>(null);
  const [candidates, setCandidates] = useState<EvaluationCandidate[]>([]);
  const [progress, setProgress] = useState<EvaluationExecutionProgress | null>(null);
  const runInFlight = useRef(false);
  const previousCurrentAttemptId = useRef<string | null>(latestCurrentProductAttempt?.id ?? null);
  const selectedHistoryRun = benchmarkHistory.find((item) => item.id === selectedHistoryId) ?? null;
  const selectedRun = selectedHistoryRun ?? latestAttempt;
  const viewingHistorical = Boolean(
    selectedHistoryRun && selectedHistoryRun.id !== currentRetrievalRun?.id
  );
  const ragCaseIds = [...CURRENT_PRODUCT_RETRIEVAL_GOLDEN_CASE_IDS];
  const comparableHistory = selectedRun
    ? benchmarkHistory.filter((item) => isComparableHistoryRun(item, selectedRun))
    : [];
  const selectedCount = scope === "cases"
    ? caseIds.length
    : scope === "full"
      ? CURRENT_PRODUCT_RETRIEVAL_GOLDEN_CASE_IDS.length
      : scope === "rag"
        ? CURRENT_PRODUCT_RETRIEVAL_GOLDEN_CASE_IDS.length
        : dataset.cases.filter((item) => item.categories.includes(scope)).length;
  const selectedScopeLabel = scopes.find((item) => item.id === scope)?.label ?? "Selected evaluation";
  const selectedHasRag = scope === "full" || scope === "rag" || (scope === "cases" && caseIds.some((caseId) => CURRENT_PRODUCT_RETRIEVAL_GOLDEN_CASE_IDS.includes(caseId)));
  const providerAuthorized = selectedHasRag && allowProviderCalls;
  const isRunning = running || runInFlight.current;

  useEffect(() => {
    const nextId = latestCurrentProductAttempt?.id ?? null;
    if (nextId && previousCurrentAttemptId.current !== nextId) {
      setSelectedHistoryId(null);
    }
    previousCurrentAttemptId.current = nextId;
  }, [latestCurrentProductAttempt?.id]);

  async function run() {
    if (runInFlight.current || running || selectedCount === 0) return;
    const selectedCases = scope === "full"
      ? dataset.cases.filter((item) => CURRENT_PRODUCT_RETRIEVAL_GOLDEN_CASE_IDS.includes(item.id))
      : scope === "cases"
        ? dataset.cases.filter((item) => caseIds.includes(item.id))
        : scope === "rag"
          ? dataset.cases.filter((item) => CURRENT_PRODUCT_RETRIEVAL_GOLDEN_CASE_IDS.includes(item.id))
          : dataset.cases.filter((item) => item.categories.includes(scope));
    runInFlight.current = true;
    setRunOpen(true);
    setProgress({
      runId: null,
      status: "preflight",
      phase: "preflight",
      lane: selectedScopeLabel,
      currentCaseId: null,
      currentCaseLabel: providerAuthorized ? "Preparing current-product benchmark" : "Preparing local preflight / workspace snapshot",
      completedCases: 0,
      totalCases: selectedCases.length,
      remainingCases: selectedCases.length,
      percent: 0,
      executionState: providerAuthorized ? "provider_authorized" : "local_preflight",
      detail: providerAuthorized
        ? "The current-product benchmark request is being validated before retrieval, generation, and evaluation."
        : "No retrieval, generation, or evaluator provider calls will run; this cannot publish a new Retrieval Quality score."
    });
    try {
      await onRun({
        scope,
        caseIds: scope === "cases" ? caseIds : [],
        allowProviderCalls: providerAuthorized,
        approvedRagasDataset: "sanitized_public"
      }, setProgress);
    } catch (error) {
      setProgress((current) => ({
        runId: current?.runId ?? null,
        status: "failed",
        phase: "terminal",
        lane: current?.lane ?? selectedScopeLabel,
        currentCaseId: current?.currentCaseId ?? null,
        currentCaseLabel: "Evaluation stopped",
        completedCases: current?.completedCases ?? 0,
        totalCases: current?.totalCases ?? selectedCases.length,
        remainingCases: current?.remainingCases ?? selectedCases.length,
        percent: current?.percent ?? null,
        executionState: current?.executionState ?? "unavailable",
        detail: error instanceof Error ? error.message : "The evaluation ended unexpectedly."
      }));
    } finally {
      runInFlight.current = false;
    }
  }

  const cases = (selectedRun?.caseResults ?? []).filter((item) => {
    if (categoryFilter !== "all" && !item.categories.includes(categoryFilter)) return false;
    if (statusFilter !== "all" && item.status !== statusFilter) return false;
    const haystack = `${item.title} ${item.domain} ${item.origins.join(" ")}`.toLowerCase();
    return haystack.includes(query.toLowerCase());
  });

  return (
    <section aria-label="Capstone evaluation workspace" className="evaluationWorkspace">
      <DashboardSection
        action={(
          <div className="evaluationLaunchActions">
            <button className="capstonePrimaryButton" disabled={isRunning || selectedCount === 0} onClick={() => void run()} type="button">
              <Sparkles aria-hidden="true" size={16} /> {isRunning ? "Evaluation running…" : "Run Evaluation"}
            </button>
            <button className="capstoneConfigureButton" disabled={isRunning} onClick={() => setRunOpen(true)} type="button">Configure run</button>
          </div>
        )}
        className="evaluationLabIntro"
        detail="Move from fixed benchmark evidence to root cause, product change, and a comparable rerun while keeping retrieval, creative, workflow, and reliability claims separate."
        eyebrow="AI Engineering Lab"
        icon={FlaskConical}
        label="AI Engineering Lab"
        title="Measure retrieval. Diagnose weaknesses. Improve the real system."
      >
        <DashboardMetricGrid
          className="evaluationLabMetrics"
          label="Evaluation benchmark summary"
          metrics={[
            {
              detail: "Frozen contract coverage only; Full does not generate all 35 prompts.",
              label: "Benchmark corpus",
              value: `${dataset.cases.length} unique cases`
            },
            {
              detail: currentRetrievalRun ? "Newest fully completed current-product retrieval run" : "Current-product benchmark has not completed",
              label: "Benchmark",
              value: currentRetrievalRun?.benchmarkVersion ?? "Awaiting current run"
            },
            {
              detail: currentRetrievalRun?.runId ?? "No current-product run identifier",
              label: "Dataset fingerprint",
              value: currentRetrievalRun ? shortFingerprint(currentRetrievalRun.datasetFingerprint) : "—"
            },
            {
              detail: currentRetrievalRun
                ? `${currentRetrievalRun.ragas.resultRows}/${currentRetrievalRun.ragas.totalSamples} current-product cases · target 85% · stretch 90%`
                : "Run the current product to establish a score · target 85% · stretch 90%",
              label: "Current Retrieval Quality",
              tone: currentRetrievalRun ? "good" : "warning",
              value: formatPreciseScore(currentProductRetrievalScore(currentRetrievalRun))
            }
          ]}
        />
        <DashboardCallout
          detail={scope === "full"
            ? "Selected: Full evaluation · seven canonical current-product RAG cases plus current local creative, workflow, and reliability workspace snapshots. The snapshots are not additional generated or evaluator-scored cases. A new RAGAS score requires provider authorization."
            : `Selected: ${selectedScopeLabel} · ${selectedCount} defined contract${selectedCount === 1 ? "" : "s"} · local preflight/workspace evidence by default. A new current-product RAGAS score requires provider authorization. Defined contracts are not claimed executed until live progress confirms them.`}
          icon={ShieldCheck}
          title="Scope and evidence stay explicit"
          tone="warning"
        />
      </DashboardSection>

      <QualityBoundaryMap ragCaseCount={ragCaseIds.length} run={currentRetrievalRun} />

      <RetrievalEvaluation
        ragCaseIds={ragCaseIds}
        run={currentRetrievalRun}
      />

      <DashboardDisclosure
        className="evaluationWorkspaceDisclosure"
        summary="Current workspace lanes and latest result counts"
      >
        <DashboardMetricGrid
          label="Latest result counts"
          metrics={[
            { label: "Pass", tone: "good", value: selectedRun?.counts.pass ?? 0 },
            { label: "Partial", tone: "warning", value: selectedRun?.counts.partial ?? 0 },
            { label: "Fail", tone: "danger", value: selectedRun?.counts.fail ?? 0 },
            { label: "Blocked", value: selectedRun?.counts.blocked ?? 0 },
            { label: "Missing", value: selectedRun?.counts.missing ?? 0 },
            { label: "Not-run result rows", value: selectedRun?.counts.notRun ?? 0 }
          ]}
        />
        <section aria-label="Evaluation categories" className="evaluationCategoriesContent">
          <DashboardCardGrid label="Evaluation quality lanes" layout="equal" role="list">
          {categories.map((category) => {
            const result = selectedRun?.categoryResults.find((item) => item.category === category);
            const measured = result?.score != null;
            return (
              <article className="dashboardInnerCard evaluationCategoryCard" data-measured={measured ? "true" : "false"} key={category} role="listitem">
                <header><span>{formatQualityLane(category)}</span><Status status={result?.status ?? "not_run"} /></header>
                {result?.score != null ? (
                  <>
                    <div className="evaluationCategoryScore"><strong>{formatScore(result.score)}</strong><span>target {Math.round(result.target * 100)}%</span></div>
                    <div aria-label={`${formatQualityLane(category)} measured score`} className="evaluationScoreTrack"><i style={{ width: `${result.score * 100}%` }} /></div>
                    <dl>
                      <div><dt>Previous</dt><dd>{formatScore(result.previousScore)}</dd></div>
                      <div><dt>Delta</dt><dd>{formatDelta(result.delta)}</dd></div>
                      <div><dt>Gap</dt><dd>{result.gap == null ? "—" : `${Math.round(result.gap * 100)} pts`}</dd></div>
                      <div><dt>Evidence</dt><dd>{result.measuredMetrics}/{result.applicableMetrics}</dd></div>
                    </dl>
                  </>
                ) : (
                  <div className="evaluationCategoryUnmeasured">
                    <strong>Not measured</strong>
                    <span>{result
                      ? `${result.measuredMetrics}/${result.applicableMetrics} metrics published; no category score is defensible.`
                      : "No run has published evidence for this category."}</span>
                  </div>
                )}
                <p>{result?.detail ?? "Run this category to create current, comparable evidence."}</p>
              </article>
            );
          })}
          </DashboardCardGrid>
        </section>
      </DashboardDisclosure>

      {runOpen ? (
        <DashboardSection
          action={<button className="evaluationCloseButton" disabled={isRunning} onClick={() => setRunOpen(false)} type="button">Close</button>}
          className="evaluationRunControls"
          detail="Choose the evidence scope and execution boundary. Provider authorization is required whenever the selection includes current-product RAGAS scoring."
          eyebrow="Run controls"
          icon={Gauge}
          label="Evaluation preflight"
          title="Choose evidence—not a marketing score"
        >
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
            <Preflight
              label="Execution"
              value={providerAuthorized ? "Provider-assisted current-product benchmark" : "Local preflight / workspace snapshot"}
              detail={providerAuthorized
                ? "Public official KB excerpts only; explicitly authorized"
                : selectedHasRag
                  ? "Zero retrieval, generation, or evaluator provider calls; no new Retrieval Quality score"
                  : "Non-RAG workspace-lane evidence only; zero provider calls"}
            />
            <Preflight
              label="Defined selection"
              value={scope === "full" ? "7 canonical RAG cases + workspace snapshots" : `${selectedCount} canonical contracts`}
              detail={scope === "full"
                ? "Live case progress covers the seven RAG cases; local creative, workflow, and reliability evidence is snapshotted separately."
                : `${scope.replace(/_/g, " ")} scope · live progress reports the cases actually executed`}
            />
            <Preflight label="History" value="Append locally" detail="Up to 24 run records survive reload" />
            <Preflight label="Cost" value={providerAuthorized ? "Estimate unavailable" : "$0 provider calls"} detail={providerAuthorized ? "Actual provider usage is reported when published" : "Local preflight/workspace-lane evidence only"} />
          </div>
          {selectedHasRag ? (
            <div className="evaluationProviderGate">
              <ShieldCheck aria-hidden="true" size={20} />
              <div><strong>Provider authorization is required for a new current-product RAGAS score</strong><p>The benchmark may send only committed public questions and public official-documentation excerpts. Raw sessions and private KB content remain ineligible. Non-RAG workspace lanes can run without provider authorization.</p></div>
              <label><input checked={allowProviderCalls} onChange={(event) => setAllowProviderCalls(event.target.checked)} type="checkbox" /> I explicitly authorize evaluator provider calls for this current-product public benchmark.</label>
            </div>
          ) : null}
          {progress ? <EvaluationRunProgress datasetVersion={dataset.version} progress={progress} /> : null}
          <footer className="evaluationRunActions"><span>{selectedCount === 0 ? "Select at least one case." : providerAuthorized ? "Provider authorization is saved for the next above-fold Run Evaluation action." : selectedHasRag ? "Without provider authorization, the next run publishes only local preflight/workspace evidence and no new Retrieval Quality score." : "This non-RAG workspace lane runs without provider calls. Use the above-fold Run Evaluation action to start."}</span></footer>
        </DashboardSection>
      ) : null}

      <DashboardDisclosure
        className="evaluationWorkspaceDisclosure"
        defaultOpen={Boolean(selectedRun)}
        summary={selectedRun ? "Measured case results and evidence gaps" : "Case results — run a benchmark to publish current evidence"}
      >
        <section aria-label="Evaluation results" className="evaluationResultsContent">
          <DashboardSectionHeader
            detail="Every category keeps its own metric family, threshold, and evidence boundary."
            eyebrow="Results"
            icon={BarChart3}
            title="Measured outcomes and evidence gaps"
          />
          {viewingHistorical && selectedHistoryRun ? (
            <DashboardCallout
              detail={`You are inspecting historical run ${selectedHistoryRun.runId ?? selectedHistoryRun.id}. Primary Retrieval Quality remains bound to the newest fully completed current-product run.`}
              icon={History}
              title="Historical run selected"
              tone="warning"
            />
          ) : null}
          {comparableHistory.length >= 2 ? (
            <div className="evaluationTrendGrid">
              {categories.map((category) => <Trend key={category} category={category} runs={comparableHistory} />)}
            </div>
          ) : null}
          <div className="evaluationFilters">
            <input aria-label="Filter evaluation cases" onChange={(event) => setQuery(event.target.value)} placeholder="Filter by case, domain, or source…" value={query} />
            <select aria-label="Filter by category" onChange={(event) => setCategoryFilter(event.target.value as typeof categoryFilter)} value={categoryFilter}><option value="all">All categories</option>{categories.map((item) => <option key={item} value={item}>{formatEvaluationCategory(item)}</option>)}</select>
            <select aria-label="Filter by status" onChange={(event) => setStatusFilter(event.target.value as typeof statusFilter)} value={statusFilter}><option value="all">All statuses</option>{["pass", "partial", "fail", "blocked", "missing_evidence", "not_run"].map((item) => <option key={item} value={item}>{item.replace(/_/g, " ")}</option>)}</select>
          </div>
          {selectedRun ? <div className="evaluationCaseTable">{cases.map((item) => <CaseRow caseResult={item} key={item.caseId} onCandidate={(caseResult) => { const candidate = createEvaluationCandidate({ caseResult }); if (candidate) setCandidates((current) => [...current, candidate]); }} />)}{cases.length === 0 ? <p className="evaluationEmpty">No cases match the active filters.</p> : null}</div> : <EmptyResults />}
        </section>
      </DashboardDisclosure>

      {candidates.length ? (
        <DashboardSection
          className="evaluationCandidates"
          detail="The canonical prompt stays unchanged; candidates begin with no score or delta until rerun."
          eyebrow="Improve"
          icon={Sparkles}
          label="Improvement candidates"
          title="Non-destructive prompt candidates"
        >
          {candidates.map((item) => <article className="dashboardInnerCard" key={item.id}><div><span>Original</span><p>{item.originalPrompt}</p><strong>{formatScore(item.baselineScore)}</strong></div><div><span>Candidate</span><p>{item.candidatePrompt}</p><strong>Pending rerun · delta —</strong></div></article>)}
        </DashboardSection>
      ) : null}

      <DashboardDisclosure className="evaluationWorkspaceDisclosure" summary="Comparable stored runs">
        <section aria-label="Evaluation history and trends" className="evaluationHistoryContent">
          <DashboardSectionHeader
            detail="Deltas appear only when pipeline fingerprints, scope, selected cases, and evaluator contracts are comparable."
            eyebrow="History"
            icon={History}
            title="Comparable stored runs"
          />
          <DashboardInfoCard
            className="evaluationHistoricalBaseline"
            detail={`Equal-weight mean of four measured dimensions on four committed sanitized rows. Context Recall is absent. Evaluated ${formatDate(CURRENT_APPROVED_RAGAS_EVALUATED_AT)} with ${CURRENT_APPROVED_RAGAS_EVIDENCE.model ?? "model unavailable"}; never used as current-product Retrieval Quality.`}
            eyebrow="Historical approved fixture · not current product"
            title="61.44%"
            tone="warning"
          >
            <small>Run {CURRENT_APPROVED_RAGAS_EVIDENCE.runId} · {CURRENT_APPROVED_RAGAS_EVIDENCE.datasetVersion} · four-metric limitation</small>
          </DashboardInfoCard>
          {benchmarkHistory.length ? <div className="evaluationRunHistory">{[...benchmarkHistory].reverse().map((run) => {
            const isCurrent = run.id === currentRetrievalRun?.id;
            const isLatestAttempt = run.id === latestAttempt?.id;
            const originLabel = isCurrent
              ? "CURRENT PRODUCT"
              : run.ragas.benchmarkMode === "current_product"
                ? run.scoreOrigin === "current_product"
                  ? "PRIOR CURRENT-PRODUCT RUN"
                  : "UNSCORED CURRENT-PRODUCT RUN"
                : run.ragas.benchmarkMode === "historical_fixture"
                  ? "HISTORICAL FIXTURE"
                  : "LOCAL WORKSPACE RUN";
            const completion = run.scope === "full"
              ? `${run.ragas.resultRows}/7 RAG cases evaluated · local workspace snapshots recorded separately`
              : `${run.executedCases}/${run.selectedCases} selected contracts observed`;
            return <button aria-pressed={selectedRun?.id === run.id} key={run.id} onClick={() => setSelectedHistoryId(run.id)} type="button"><Status status={statusTone(run.statusLabel)} /><span><strong>{originLabel} · {run.scope.replace(/_/g, " ")}</strong><small>{formatDate(run.timestamp ?? run.completedAt)} · {completion}{isLatestAttempt ? " · latest attempt" : ""}</small></span><span><strong>{run.evaluator ?? run.provider ?? "Local only"}</strong><small>{run.runId ?? run.id} · {shortFingerprint(run.ragas.datasetFingerprint)}</small></span></button>;
          })}</div> : <p className="evaluationEmpty">No current-product benchmark run is stored yet. The historical approved fixture above remains comparison evidence only.</p>}
        </section>
      </DashboardDisclosure>

      <DashboardDisclosure className="evaluationWorkspaceDisclosure" summary="Evaluation claim mapping and methodology">
        <section aria-label="Capstone evaluation mapping" className="evaluationCoverageContent">
          <DashboardSectionHeader
            detail="The matrix separates measurable product evidence from checks that require another environment or human observation."
            eyebrow="Capstone coverage"
            icon={CheckCircle2}
            title="What each evaluation claim is built from"
          />
          <DashboardTableFrame>
            <table>
              <thead><tr><th>Capability</th><th>Evidence</th><th>Truth boundary</th></tr></thead>
              <tbody>{coverageRows.map((row) => <tr key={row[0]}><th scope="row">{row[0]}</th><td>{row[1]}</td><td>{row[2]}</td></tr>)}</tbody>
            </table>
          </DashboardTableFrame>
          <DashboardDisclosure className="evaluationMethodology" summary="Methodology, scoring, and limitations">
            <div><p><strong>Golden dataset.</strong> {dataset.rawSourceCount} product-authored records are normalized into {dataset.cases.length} unique prompt contracts; {dataset.duplicateCount} aliases are deduplicated. IDs and a deterministic fingerprint make selection changes visible.</p><p><strong>Metric separation.</strong> RAGAS metrics apply only to current generated answers with current retrieved contexts. Product-specific creative, workflow, runtime, and persistence signals retain their own labels. No cross-category global score is calculated.</p><p><strong>Missing evidence.</strong> BLOCKED_BY_EXECUTION_ENVIRONMENT means a required evaluator, credential, network, or runner was unavailable. MISSING_EVIDENCE means the current product did not publish defensible proof. Neither becomes zero.</p><p><strong>Known limits.</strong> Context Recall requires independently reviewed reference answers. Visual clarity and interaction success need rendered or human evidence. Historical QA remains comparison evidence and never becomes a current result.</p></div>
          </DashboardDisclosure>
        </section>
      </DashboardDisclosure>
    </section>
  );
}

function Status({ status }: { status: EvaluationMetricStatus }) { return <span className="evaluationStatusPill" data-status={status}>{status.replace(/_/g, " ")}</span>; }
function Preflight({ detail, label, value }: { detail: string; label: string; value: string }) { return <div><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>; }

function formatQualityLane(category: EvaluationCategory) {
  if (category === "rag") return "Retrieval Quality";
  if (category === "creative_artifact") return "Creative Quality";
  if (category === "workflow") return "Workflow Quality";
  return "Product Reliability";
}

function QualityBoundaryMap({ ragCaseCount, run }: { ragCaseCount: number; run: EvaluationBenchmarkRun | null }) {
  const retrievalScore = currentProductRetrievalScore(run);
  const categorySignal = (category: EvaluationCategory) => run?.categoryResults.find((result) => result.category === category)?.score ?? null;
  const measuredMetrics = currentProductMetricIds.filter((id) => run?.ragas.metricScores[id] != null).length;
  const completedCases = run?.ragas.resultRows ?? 0;
  const totalCases = run?.ragas.totalSamples ?? ragCaseCount;
  const lanes = [
    {
      classification: run ? "CURRENT PRODUCT" : "NOT MEASURED",
      detail: run
        ? "Equal-weight mean of all five justified RAGAS dimensions from the newest fully completed current-product benchmark."
        : "No fully completed current-product retrieval benchmark is available. Historical fixture evidence is kept in History only.",
      label: "Retrieval Quality",
      value: formatPreciseScore(retrievalScore)
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
      detail: run
        ? `${completedCases}/${totalCases} current-product retrieval cases published terminal evidence. Coverage is not quality.`
        : `${ragCaseCount} retrieval contracts are defined; none is claimed executed until a current-product run completes.`,
      label: "Benchmark Coverage",
      value: run ? `${completedCases}/${totalCases} executed` : "Awaiting run"
    },
    {
      classification: "COVERAGE ONLY",
      detail: `${measuredMetrics}/5 justified current-product Retrieval dimensions are measured; missing evidence is not a zero.`,
      label: "Evidence Coverage",
      value: `${measuredMetrics}/5 RAG metrics`
    }
  ] as const;

  return (
    <DashboardSection
      className="evaluationQualityBoundaries"
      detail="Quality and coverage answer different questions and never share a denominator."
      eyebrow="Score boundaries"
      icon={ShieldCheck}
      label="Evaluation score boundaries"
      title="Six independent signals. No global score."
    >
      <DashboardCardGrid label="Independent evaluation signals" layout="equal" role="list">
        {lanes.map((lane) => (
          <DashboardInfoCard
            detail={lane.detail}
            eyebrow={lane.label}
            key={lane.label}
            role="listitem"
            title={lane.value}
          >
            <small className="evaluationLaneClassification">{lane.classification}</small>
          </DashboardInfoCard>
        ))}
      </DashboardCardGrid>
      <DashboardCallout
        detail="Retrieval quality, creative quality, workflow quality, product reliability, benchmark coverage, and evidence coverage remain independent claims. Coverage is not quality."
        icon={ShieldCheck}
        title="No cross-category aggregate"
      />
    </DashboardSection>
  );
}

function RetrievalEvaluation({ ragCaseIds, run }: { ragCaseIds: string[]; run: EvaluationBenchmarkRun | null }) {
  const evidence = run?.ragas ?? null;
  const supportedScores = currentProductMetricIds.flatMap((id) => {
    const score = evidence?.metricScores[id] ?? null;
    return score == null ? [] : [score];
  });
  const retrievalScore = currentProductRetrievalScore(run);
  const evidenceCoverage = supportedScores.length / currentProductMetricIds.length;
  const productReliability = run?.categoryResults.find((result) => result.category === "product_reliability") ?? null;
  const retrievalStatus = retrievalScore == null
    ? run ? "missing_evidence" : "not_run"
    : scoreStatus(retrievalScore);

  return (
    <DashboardSection
      action={(
        <div className="retrievalExecutionBadge">
          <Status status={retrievalStatus} />
          <strong>{run?.benchmarkVersion ?? "No current-product score"}</strong>
          <small>{run ? `${run.ragas.state.replace(/_/g, " ")} · ${run.evaluator ?? "evaluator not reported"}` : "Run the current product to publish Retrieval Quality"}</small>
        </div>
      )}
      className="evaluationRetrievalSection"
      detail="Only a fully completed current-product run with all five justified dimensions can become primary Retrieval Quality. Missing or partial evidence remains visible and never becomes zero."
      eyebrow="Current-product RAGAS"
      icon={Database}
      label="RAGAS retrieval evaluation"
      title="Current Retrieval Quality, with complete provenance"
    >

      <div className="evaluationRetrievalOverview">
        <article className="dashboardInnerCard retrievalOverallScore" data-status={retrievalStatus}>
          <span><Gauge aria-hidden="true" size={15} /> Current-product Overall Retrieval Score</span>
          <strong>{formatPreciseScore(retrievalScore)}</strong>
          <div className="evaluationScoreTrack"><i style={{ width: `${(retrievalScore ?? 0) * 100}%` }} /></div>
          <p>{supportedScores.length}/5 justified RAGAS dimensions measured. {retrievalScore == null ? "No partial metric set is promoted to the primary score." : "Each dimension contributes equally; target 85%, stretch 90%."}</p>
        </article>
        <DashboardCardGrid className="evaluationRetrievalBoundaries" label="Retrieval score boundaries" layout="equal" role="list">
          <EvaluationBoundaryCard
            detail={run ? `${run.ragas.resultRows}/${run.ragas.totalSamples} current-product retrieval cases published scored evidence.` : `${ragCaseIds.length} retrieval contracts are defined; no execution is claimed before a run completes.`}
            label="Benchmark coverage"
            value={run ? `${run.ragas.resultRows}/${run.ragas.totalSamples} executed` : "Awaiting run"}
          />
          <EvaluationBoundaryCard
            caution
            detail={`${supportedScores.length}/5 justified current-product RAG metric dimensions measured. This is not an overall product or project score.`}
            label="Evidence coverage"
            value={formatScore(evidenceCoverage)}
          />
          <EvaluationBoundaryCard
            detail={productReliability?.score == null ? "Not measured in the selected run. Runtime, persistence, validation, and bug evidence remain a separate quality lane." : "Measured from product reliability evidence only; never included in current-product Retrieval Quality."}
            label="Product Reliability"
            value={productReliability?.score == null ? "Not measured" : formatScore(productReliability.score)}
          />
        </DashboardCardGrid>
      </div>

      {run ? <CurrentRunProvenance run={run} /> : null}

      {evidence ? <ScoreCompositionContract evidence={evidence} /> : null}

      <DashboardDisclosure className="evaluationDeepDisclosure" summary="Current-product retrieval engineering evolution">
        <RetrievalEngineeringEvolution run={run} />
      </DashboardDisclosure>

      <DashboardCardGrid className="evaluationRagasMetricGrid" label="RAGAS metric scores" layout="compact" role="list">
        {retrievalMetrics.map((definition) => {
          const metric = findRetrievalMetric(run, definition.id);
          const score = evidence?.metricScores[definition.id] ?? metric?.score ?? null;
          const status = score == null
            ? metric?.status ?? (run ? "missing_evidence" : "not_run")
            : scoreStatus(score);
          const value = score == null
            ? status === "blocked" ? "BLOCKED" : status === "missing_evidence" ? "MISSING EVIDENCE" : "NOT RUN"
            : formatPreciseScore(score);
          const evidenceDetail = score != null
            ? `Current-product RAGAS score across ${evidence?.resultRows ?? 0} retrieval cases.`
            : metric?.detail ?? "No defensible measurement is published for this dimension.";
          return (
            <article className="dashboardInnerCard evaluationRagasMetricCard" data-status={status} key={definition.id} role="listitem">
              <header><span>{definition.label}</span><Status status={status} /></header>
              <strong>{value}</strong>
              <p>{definition.detail}</p>
              <small>{evidenceDetail}</small>
            </article>
          );
        })}
      </DashboardCardGrid>

      <DashboardDisclosure className="evaluationDeepDisclosure" summary="Metric diagnostics — root cause, change, delta, and next step">
        <MetricDiagnostics run={run} />
      </DashboardDisclosure>

      <DashboardDisclosure className="evaluationDeepDisclosure" summary="Evidence lineage and evaluation execution timeline">
        <div className="retrievalEvidenceGrid">
          {evidence ? <EvidenceLineage evidence={evidence} /> : <p>No current-product evidence lineage has been published.</p>}
          <RetrievalExecutionTimeline run={run} />
        </div>
      </DashboardDisclosure>

      <DashboardCallout
        as="footer"
        detail={run ? `${run.ragas.resultRows} evaluated current-product cases from ${run.ragas.eligibleSamples}/${run.ragas.totalSamples} eligible cases. ${run.evaluator ?? "Evaluator not reported"} · ${run.embeddingModel ?? "embedding not reported"} · RAGAS ${run.ragas.ragasVersion ?? "version unavailable"}.` : "No current-product Retrieval Quality evidence is published. Historical approved-fixture evidence remains available in History for comparison only."}
        icon={ShieldCheck}
        title="Evidence provenance and limits"
      />
    </DashboardSection>
  );
}

function CurrentRunProvenance({ run }: { run: EvaluationBenchmarkRun }) {
  return (
    <DashboardMetricGrid
      className="evaluationRunProvenance"
      label="Current Retrieval Quality provenance"
      metrics={[
        { label: "Score origin", value: run.scoreOrigin.replace(/_/g, " ") },
        { label: "Benchmark version", value: run.benchmarkVersion },
        { label: "Dataset fingerprint", value: shortFingerprint(run.datasetFingerprint) },
        { label: "Retrieval fingerprint", value: shortFingerprint(run.retrievalFingerprint) },
        { label: "Prompt fingerprint", value: shortFingerprint(run.promptFingerprint) },
        { label: "Generation fingerprint", value: shortFingerprint(run.generationFingerprint) },
        { label: "Generation model", value: run.generationModel ?? "Unavailable" },
        { label: "Evaluator", value: run.evaluator ?? "Unavailable" },
        { label: "Embedding model", value: run.embeddingModel ?? "Unavailable" },
        { label: "Timestamp", value: formatDate(run.timestamp) },
        { label: "Run identifier", value: run.runId }
      ]}
    />
  );
}

function RetrievalEngineeringEvolution({ run }: { run: EvaluationBenchmarkRun | null }) {
  return (
    <section aria-label="Retrieval engineering evolution" className="retrievalEvolutionPanel">
      <header>
        <div><span><LineChart aria-hidden="true" size={14} /> Current run evidence</span><strong>Per-case retrieval measurements from the selected current-product run</strong></div>
        <small>{run ? `${run.ragas.resultRows}/${run.ragas.totalSamples} cases · ${formatDate(run.timestamp)}` : "No completed current-product run"}</small>
      </header>
      <div className="retrievalRagasComparison">
        <div><span>Current Retrieval Quality</span><strong>{formatPreciseScore(currentProductRetrievalScore(run))}</strong><small>Target 85% · stretch 90%</small></div>
        <ArrowRight aria-hidden="true" size={17} />
        <div><span>Current benchmark</span><strong>{run?.benchmarkVersion ?? "Awaiting run"}</strong><small>{run?.runId ?? "No run identifier"}</small></div>
        <div><span>Comparable delta</span><strong>{formatDelta(run?.categoryResults.find((item) => item.category === "rag")?.delta ?? null)}</strong><small>Only same dataset and evaluator contracts compare</small></div>
      </div>
      <div className="retrievalEvolutionStages">
        {(run?.ragas.caseRows ?? []).map((row, index) => {
          const scores = Object.values(row.metrics).filter((score): score is number => typeof score === "number");
          const caseScore = scores.length ? scores.reduce((sum, score) => sum + score, 0) / scores.length : null;
          return (
            <article key={row.sampleId}>
              <header><span>{String(index + 1).padStart(2, "0")}</span><strong>{row.sampleId}</strong></header>
              <p>{row.domains.join(" · ") || "Domains unavailable"}</p>
              <div><span>Measured metric mean</span><strong>{formatPreciseScore(caseScore)}</strong><i><b style={{ width: `${(caseScore ?? 0) * 100}%` }} /></i></div>
              <div><span>Retrieved sources</span><strong>{row.sourceIds.length}</strong></div>
            </article>
          );
        })}
        {!run?.ragas.caseRows.length ? <p>No per-case current-product evidence has been published.</p> : null}
      </div>
      <footer>
        <small>Case evidence, metric means, and source counts are read directly from the current run. They are never substituted with the historical approved fixture.</small>
        <small className="retrievalReportProof">Retrieval {shortFingerprint(run?.retrievalFingerprint ?? null)} · prompt {shortFingerprint(run?.promptFingerprint ?? null)} · generation {shortFingerprint(run?.generationFingerprint ?? null)}</small>
      </footer>
    </section>
  );
}

function ScoreCompositionContract({ evidence }: { evidence: EvaluationBenchmarkRun["ragas"] }) {
  const includedMetricIds = currentProductMetricIds;
  const exclusions = [
    {
      classification: "SUBJECTIVE",
      detail: "Artistic merit, visual clarity, and interaction quality require rendered or human observation and never enter current-product Retrieval Quality.",
      label: "Artistic and visual judgement"
    },
    {
      classification: "NOT_COMPARABLE",
      detail: "Retrieval, creative, workflow, reliability, benchmark coverage, and evidence coverage measure different constructs; a global score is not calculated.",
      label: "Cross-category aggregate"
    }
  ] as const;
  const missingMetrics = includedMetricIds
    .filter((metricId) => evidence.metricScores[metricId] == null)
    .map((metricId) => ({
      classification: "MISSING_EVIDENCE" as const,
      detail: `${retrievalMetrics.find((item) => item.id === metricId)?.label ?? metricId} is absent from this run and contributes neither zero nor weight. The run cannot become primary Retrieval Quality.`,
      label: retrievalMetrics.find((item) => item.id === metricId)?.label ?? metricId
    }));

  return (
    <DashboardDisclosure
      className="retrievalScoreContract"
      summary={<div><span><ShieldCheck aria-hidden="true" size={14} /> Retrieval score contract</span><strong>Exactly what enters current-product Retrieval Quality</strong></div>}
    >
        <section aria-label="Included retrieval metrics">
          <header><span>Included</span><strong>Equal weight inside Retrieval Quality only</strong></header>
          {includedMetricIds.map((metricId) => {
            const metric = retrievalMetrics.find((item) => item.id === metricId);
            const score = evidence.metricScores[metricId] ?? null;
            return (
              <article key={metricId}>
                <div><strong>{metric?.label}</strong><small>{score == null ? "No published current-product value" : "Current-product benchmark under the visible evaluator and embedding contract"}</small></div>
                <span>{formatPreciseScore(score)}</span>
                <em data-classification={score == null ? "MISSING_EVIDENCE" : "INCLUDED"}>{score == null ? "MISSING_EVIDENCE" : "INCLUDED"}</em>
              </article>
            );
          })}
        </section>
        <section aria-label="Excluded evaluation evidence">
          <header><span>Excluded</span><strong>Visible, justified, never coerced to zero</strong></header>
          {[...missingMetrics, ...exclusions].map((exclusion) => (
            <article key={exclusion.label}>
              <div><strong>{exclusion.label}</strong><small>{exclusion.detail}</small></div>
              <em data-classification={exclusion.classification}>{exclusion.classification}</em>
            </article>
          ))}
        </section>
    </DashboardDisclosure>
  );
}

function MetricDiagnostics({ run }: { run: EvaluationBenchmarkRun | null }) {
  const ragResult = run?.categoryResults.find((item) => item.category === "rag") ?? null;
  const recommendation = run?.recommendations.find((item) => item.category === "rag") ?? null;
  return (
    <section aria-label="Metric engineering diagnostics" className="retrievalDiagnostics">
      <header><div><span><Activity aria-hidden="true" size={14} /> Metric diagnostics</span><strong>What the current-product run measured</strong></div><small>Open a metric for current evidence, target, provenance change, and next step.</small></header>
      <div>
        {retrievalMetrics.map((metric) => {
          const score = run?.ragas.metricScores[metric.id] ?? null;
          const belowTarget = score != null && score < .85;
          return (
            <details key={metric.id}>
              <summary><span>{metric.label}</span><strong>{formatPreciseScore(score)}</strong><ChevronDown aria-hidden="true" size={14} /></summary>
              <dl>
                <div><dt>Current-product score</dt><dd>{score == null ? "MISSING_EVIDENCE" : formatPreciseScore(score)}</dd></div>
                <div><dt>Target</dt><dd>85% acceptance · 90% stretch under the same benchmark and evaluator contract.</dd></div>
                <div><dt>Root cause</dt><dd>{score == null ? "This run did not publish a defensible score for the metric." : belowTarget ? "The current metric remains below the acceptance target; inspect the per-case evidence before changing the product." : "No below-target defect is established by the current metric."}</dd></div>
                <div><dt>Product version</dt><dd>Retrieval {shortFingerprint(run?.retrievalFingerprint ?? null)} · prompt {shortFingerprint(run?.promptFingerprint ?? null)} · generation {shortFingerprint(run?.generationFingerprint ?? null)}</dd></div>
                <div><dt>Comparable benchmark delta</dt><dd>{formatDelta(ragResult?.delta ?? null)}</dd></div>
                <div><dt>Remaining limitation</dt><dd>{run?.ragas.detail ?? "No current-product benchmark has completed."}</dd></div>
                <div><dt>Recommended next engineering step</dt><dd>{recommendation?.detail ?? (belowTarget ? `Review the lowest ${metric.label} case evidence and improve the real retrieval or generation path.` : "Retain the current behavior and broaden independently reviewed benchmark coverage.")}</dd></div>
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
    <details className="retrievalLineage">
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

function RetrievalExecutionTimeline({ run }: { run: EvaluationBenchmarkRun | null }) {
  const categoryResults = run?.categoryResults ?? [];
  return (
    <section aria-label="Retrieval evaluation execution timeline" className="retrievalTimeline">
      <header><span><Activity aria-hidden="true" size={14} /> Execution timeline</span><strong>{run ? "Newest current-product terminal run" : "Awaiting current-product run"}</strong></header>
      <ol>
        {categoryResults.map((category) => (
          <li data-status={category.status} key={category.category}><i /><div><span>{formatQualityLane(category.category)}</span><strong>{category.status.replace(/_/g, " ").toUpperCase()} · {category.measuredMetrics}/{category.applicableMetrics} metrics</strong><small>{category.detail}</small></div></li>
        ))}
        {run ? <li data-status={run.ragas.state === "completed" ? "pass" : run.ragas.state}><i /><div><span>Terminal evidence publication</span><strong>{run.ragas.state.replace(/_/g, " ").toUpperCase()} · {run.ragas.resultRows}/{run.ragas.totalSamples} retrieval cases</strong><small>{run.ragas.detail}</small></div></li> : <li data-status="not_run"><i /><div><span>Current-product benchmark</span><strong>NOT RUN</strong><small>Use Run Evaluation to begin the selected current-product scope.</small></div></li>}
      </ol>
    </section>
  );
}

function EvaluationBoundaryCard({ caution = false, detail, label, value }: { caution?: boolean; detail: string; label: string; value: string }) {
  return <article className="dashboardInnerCard evaluationBoundaryCard" data-caution={caution} role="listitem"><span>{label}</span><strong>{value}</strong><p>{detail}</p></article>;
}

function EvaluationRunProgress({ datasetVersion, progress }: { datasetVersion: string; progress: EvaluationExecutionProgress }) {
  const percent = progress.percent;
  const indeterminate = percent == null && !["terminal", "completed"].includes(progress.phase.toLowerCase());
  return (
    <section aria-label="Live evaluation progress" aria-live="polite" className="evaluationLiveProgress">
      <header><span><Activity aria-hidden="true" size={15} /> Live run</span><strong>{percent == null ? "Progress pending" : `${percent}% complete`}</strong></header>
      <div aria-label="Evaluation progress" aria-valuemax={100} aria-valuemin={0} {...(percent == null ? {} : { "aria-valuenow": percent })} className="evaluationProgressTrack" data-indeterminate={indeterminate} role="progressbar"><i style={{ width: percent == null ? "28%" : `${percent}%` }} /></div>
      <dl>
        <div><dt>Current benchmark</dt><dd>{datasetVersion} · {progress.runId ?? "run ID pending"}</dd></div>
        <div><dt>Current lane</dt><dd>{progress.lane}</dd></div>
        <div><dt>Current case</dt><dd>{progress.currentCaseLabel}{progress.currentCaseId ? ` · ${progress.currentCaseId}` : ""}</dd></div>
        <div><dt>Completed / remaining</dt><dd>{progress.completedCases} completed / {progress.remainingCases} remaining of {progress.totalCases}</dd></div>
        <div><dt>Execution phase</dt><dd>{progress.phase.replace(/_/g, " ")} · {progress.status.replace(/_/g, " ")}</dd></div>
        <div><dt>Local / provider state</dt><dd>{progress.executionState.replace(/_/g, " ")}</dd></div>
      </dl>
      <small>{progress.detail}</small>
    </section>
  );
}

function findRetrievalMetric(run: EvaluationBenchmarkRun | null, metricId: string): EvaluationMetricResult | null {
  if (!run) return null;
  const matches = run.caseResults.flatMap((caseResult) => caseResult.metrics).filter((metric) => metric.kind === "ragas" && metric.id === metricId);
  return matches.find((metric) => metric.score != null) ?? matches.find((metric) => metric.status !== "not_run") ?? null;
}

function isComparableHistoryRun(run: EvaluationBenchmarkRun, anchor: EvaluationBenchmarkRun) {
  return matchesCurrentProductEvidenceIdentity(run.ragas, anchor.ragas) &&
    run.scope === anchor.scope &&
    run.selectedCaseIds.join("|") === anchor.selectedCaseIds.join("|") &&
    run.ragas.privacyClass === anchor.ragas.privacyClass;
}

function Trend({ category, runs }: { category: EvaluationCategory; runs: EvaluationBenchmarkRun[] }) {
  const points = runs.slice(-8).map((run) => run.categoryResults.find((item) => item.category === category)?.score ?? null);
  return <article><header><strong>{formatEvaluationCategory(category)}</strong><span>{points.filter((item) => item != null).length} measured</span></header><div>{points.length ? points.map((point, index) => <i aria-label={point == null ? "Missing evidence" : `${Math.round(point * 100)} percent`} key={index} style={{ height: `${point == null ? 6 : Math.max(8, point * 100)}%` }} data-missing={point == null} />) : <small>Run to start trend</small>}</div></article>;
}

function CaseRow({ caseResult, onCandidate }: { caseResult: EvaluationCaseResult; onCandidate: (value: EvaluationCaseResult) => void }) {
  return <details><summary><Status status={caseResult.status} /><span><strong>{caseResult.title}</strong><small>{caseResult.domain} · {caseResult.origins.join(", ")}</small></span><strong>{formatScore(caseResult.score)}</strong><ChevronDown aria-hidden="true" size={15} /></summary><div className="evaluationCaseDetail"><div className="evaluationExpected"><span>Expected</span><strong>{caseResult.expectedArtifact}</strong><p>{caseResult.previewContract}</p></div><div className="evaluationMetricList">{caseResult.metrics.map((metric) => <article key={`${caseResult.caseId}-${metric.id}`}><header><Status status={metric.status} /><strong>{formatEvaluationMetric(metric.id)}</strong><span>{formatScore(metric.score)}</span></header><p>{metric.detail}</p><small>{metric.kind === "ragas" ? "RAGAS" : "Product-specific"} · {metric.evidenceClass.replace(/_/g, " ")} · target {metric.target == null ? "n/a" : `${Math.round(metric.target * 100)}%`} · gap {metric.gap == null ? "n/a" : `${Math.round(metric.gap * 100)} pts`}</small></article>)}</div>{caseResult.recommendation ? <aside><div><span>Recommended next step</span><strong>{caseResult.recommendation.title}</strong><p>{caseResult.recommendation.detail}</p></div><button onClick={() => onCandidate(caseResult)} type="button">Create improved candidate</button></aside> : null}</div></details>;
}

function EmptyResults() { return <div className="evaluationEmptyState"><FlaskConical aria-hidden="true" size={24} /><strong>No benchmark result yet</strong><p>Provider authorization is required to publish a new current-product Retrieval Quality score. Non-RAG workspace lanes can run without provider calls.</p></div>; }
function formatScore(value: number | null) { return value == null ? "—" : `${Math.round(value * 100)}%`; }
function formatPreciseScore(value: number | null) { return value == null ? "—" : `${(value * 100).toFixed(2)}%`; }
function formatDelta(value: number | null) { return value == null ? "—" : `${value > 0 ? "+" : ""}${Math.round(value * 100)} pts`; }
function formatDate(value: string) { const date = new Date(value); return Number.isNaN(date.getTime()) ? value : date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }); }
function runTimestamp(run: EvaluationBenchmarkRun) { const value = new Date(run.timestamp ?? run.completedAt).getTime(); return Number.isNaN(value) ? 0 : value; }
function scoreStatus(value: number): EvaluationMetricStatus { return value >= .85 ? "pass" : value >= .6 ? "partial" : "fail"; }
function shortFingerprint(value: string | null) { return value ? value.replace(/^sha256:/, "").slice(0, 12) : "Unavailable"; }
function statusTone(value: EvaluationBenchmarkRun["statusLabel"]): EvaluationMetricStatus { if (value === "Blocked") return "blocked"; if (value === "Needs Improvement") return "fail"; if (value === "Incomplete Evidence") return "missing_evidence"; return "pass"; }

function isFullyCompletedCurrentProductRun(run: EvaluationBenchmarkRun) {
  const retrievalScore = currentProductRetrievalScoreFromEvidence(run.ragas);
  return (run.scope === "full" || run.scope === "rag") &&
    retrievalScore != null &&
    run.scoreOrigin === "current_product" &&
    run.benchmarkVersion === run.ragas.benchmarkVersion &&
    run.datasetFingerprint === run.ragas.datasetFingerprint &&
    run.retrievalFingerprint === run.ragas.retrievalFingerprint &&
    run.promptFingerprint === run.ragas.promptFingerprint &&
    run.generationFingerprint === run.ragas.generationFingerprint &&
    run.outputFingerprint === run.ragas.outputFingerprint &&
    run.selectionFingerprint === run.ragas.selectionFingerprint &&
    run.kbFingerprint === run.ragas.kbFingerprint &&
    run.generationModel === run.ragas.generationModel &&
    run.evaluator === run.ragas.evaluator &&
    run.evaluatorModel === run.ragas.evaluatorModel &&
    run.embeddingModel === run.ragas.embeddingModel &&
    run.retrievalScore === retrievalScore &&
    run.timestamp === run.ragas.timestamp &&
    run.runId === run.ragas.runId;
}

function currentProductRetrievalScore(run: EvaluationBenchmarkRun | null) {
  if (!run || !isFullyCompletedCurrentProductRun(run)) return null;
  return currentProductRetrievalScoreFromEvidence(run.ragas);
}
