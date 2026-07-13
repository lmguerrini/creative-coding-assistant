"use client";

import { useMemo, useState } from "react";
import { BarChart3, CheckCircle2, ChevronDown, FlaskConical, History, ShieldCheck, Sparkles } from "lucide-react";
import {
  buildGoldenEvaluationDataset,
  createEvaluationCandidate,
  formatEvaluationCategory,
  formatEvaluationMetric,
  type EvaluationBenchmarkRun,
  type EvaluationCandidate,
  type EvaluationCaseResult,
  type EvaluationCategory,
  type EvaluationMetricStatus,
  type EvaluationRunRequest,
  type EvaluationScope
} from "@/lib/evaluation-benchmark";
import type { EvaluationHistoryRecord } from "@/lib/product-controls";
import type { ProductIntelligenceModel } from "@/lib/product-intelligence";

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
  const selectedRun = benchmarkHistory.find((item) => item.id === selectedHistoryId) ?? latest;
  const selectedCount = scope === "cases"
    ? caseIds.length
    : scope === "full"
      ? dataset.cases.length
      : dataset.cases.filter((item) => item.categories.includes(scope)).length;
  const selectedHasRag = scope === "full" || scope === "rag" || (scope === "cases" && dataset.cases.some((item) => caseIds.includes(item.id) && item.categories.includes("rag")));

  async function run() {
    await onRun({
      scope,
      caseIds: scope === "cases" ? caseIds : [],
      allowProviderCalls: selectedHasRag && allowProviderCalls,
      approvedRagasDataset
    });
    setRunOpen(false);
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
          <span className="capstoneEyebrow"><FlaskConical aria-hidden="true" size={14} /> Canonical product evaluation</span>
          <h2>Evidence you can inspect, rerun, and defend.</h2>
          <p>Four separate evaluation systems share one versioned golden dataset. Missing or unavailable proof stays visible—it never becomes a zero or a fabricated pass.</p>
          <div className="capstoneHeroMeta">
            <span><strong>{dataset.cases.length}</strong> unique cases</span>
            <span><strong>{dataset.version}</strong> dataset</span>
            <span><strong>{dataset.fingerprint.slice(0, 10)}</strong> fingerprint</span>
          </div>
        </div>
        <div className="capstoneHeroAction">
          <span className="evaluationStatusPill" data-status={latest ? statusTone(latest.statusLabel) : "not_run"}>{latest?.statusLabel ?? "Not run"}</span>
          <strong>{latest ? `${Math.round(latest.evidenceCompleteness * 100)}% evidence complete` : "Ready for a local run"}</strong>
          <small>{latest ? formatDate(latest.completedAt) : "No benchmark history yet"}</small>
          <button className="capstonePrimaryButton" disabled={running} onClick={() => setRunOpen(true)} type="button">
            <Sparkles aria-hidden="true" size={16} /> {running ? "Evaluation running…" : "Run Evaluation"}
          </button>
        </div>
      </header>

      <div className="evaluationCountStrip" aria-label="Latest result counts">
        <Count label="Pass" value={latest?.counts.pass ?? 0} tone="pass" />
        <Count label="Partial" value={latest?.counts.partial ?? 0} tone="partial" />
        <Count label="Fail" value={latest?.counts.fail ?? 0} tone="fail" />
        <Count label="Blocked" value={latest?.counts.blocked ?? 0} tone="blocked" />
        <Count label="Missing" value={latest?.counts.missing ?? 0} tone="missing_evidence" />
        <Count label="Not run" value={latest?.counts.notRun ?? dataset.cases.length} tone="not_run" />
      </div>

      <section aria-label="Evaluation categories" className="evaluationCategoryGrid">
        {categories.map((category) => {
          const result = latest?.categoryResults.find((item) => item.category === category);
          return (
            <article className="evaluationCategoryCard" key={category}>
              <header><span>{formatEvaluationCategory(category)}</span><Status status={result?.status ?? "not_run"} /></header>
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
          <header><div><span>Run preflight</span><strong>Choose evidence—not a marketing score</strong></div><button onClick={() => setRunOpen(false)} type="button">Close</button></header>
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
            <Preflight label="Selection" value={`${selectedCount} canonical cases`} detail={`${scope.replace(/_/g, " ")} scope`} />
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
          <footer><span>{selectedCount === 0 ? "Select at least one case." : allowProviderCalls ? "Provider call authorized for approved RAGAS fixture." : "No evaluator provider will be called."}</span><button className="capstonePrimaryButton" disabled={running || selectedCount === 0} onClick={() => void run()} type="button">{running ? "Running…" : `Run ${selectedCount} case${selectedCount === 1 ? "" : "s"}`}</button></footer>
        </section>
      ) : null}

      <section aria-label="Evaluation results" className="evaluationResultsSection">
        <SectionHeading icon={<BarChart3 aria-hidden="true" size={16} />} eyebrow="Results" title="Measured outcomes and evidence gaps" detail="Every category keeps its own metric family, threshold, and evidence boundary." />
        <div className="evaluationTrendGrid">
          {categories.map((category) => <Trend key={category} category={category} runs={benchmarkHistory} />)}
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

      <details className="evaluationMethodology"><summary><ChevronDown aria-hidden="true" size={16} /> Methodology, scoring, and limitations</summary><div><p><strong>Golden dataset.</strong> {dataset.rawSourceCount} product-authored records are normalized into {dataset.cases.length} unique prompt contracts; {dataset.duplicateCount} aliases are deduplicated. IDs and a deterministic fingerprint make changes visible.</p><p><strong>Metric separation.</strong> RAGAS metrics apply only to recorded answers with contexts. Product-specific creative, workflow, runtime, and persistence signals retain their own labels. A combined score is withheld unless all four categories have scores and at least 80% case coverage.</p><p><strong>Missing evidence.</strong> BLOCKED_BY_EXECUTION_ENVIRONMENT means a required evaluator, credential, network, or test runner was unavailable. MISSING_EVIDENCE means the current product did not publish defensible proof. Neither is converted to zero.</p><p><strong>Known limits.</strong> Context recall has no justified reference answers in the approved fixtures. Visual clarity and interaction success need rendered or human evidence. Historic QA can inform review but never becomes a current run result.</p></div></details>
    </section>
  );
}

function Count({ label, tone, value }: { label: string; tone: EvaluationMetricStatus; value: number }) { return <span data-status={tone}><strong>{value}</strong>{label}</span>; }
function Status({ status }: { status: EvaluationMetricStatus }) { return <span className="evaluationStatusPill" data-status={status}>{status.replace(/_/g, " ")}</span>; }
function Preflight({ detail, label, value }: { detail: string; label: string; value: string }) { return <div><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>; }
function SectionHeading({ detail, eyebrow, icon, title }: { detail: string; eyebrow: string; icon: React.ReactNode; title: string }) { return <header className="evaluationSectionHeading"><span>{icon}{eyebrow}</span><strong>{title}</strong><p>{detail}</p></header>; }

function Trend({ category, runs }: { category: EvaluationCategory; runs: EvaluationBenchmarkRun[] }) {
  const points = runs.slice(-8).map((run) => run.categoryResults.find((item) => item.category === category)?.score ?? null);
  return <article><header><strong>{formatEvaluationCategory(category)}</strong><span>{points.filter((item) => item != null).length} measured</span></header><div>{points.length ? points.map((point, index) => <i aria-label={point == null ? "Missing evidence" : `${Math.round(point * 100)} percent`} key={index} style={{ height: `${point == null ? 6 : Math.max(8, point * 100)}%` }} data-missing={point == null} />) : <small>Run to start trend</small>}</div></article>;
}

function CaseRow({ caseResult, onCandidate }: { caseResult: EvaluationCaseResult; onCandidate: (value: EvaluationCaseResult) => void }) {
  return <details><summary><Status status={caseResult.status} /><span><strong>{caseResult.title}</strong><small>{caseResult.domain} · {caseResult.origins.join(", ")}</small></span><strong>{formatScore(caseResult.score)}</strong><ChevronDown aria-hidden="true" size={15} /></summary><div className="evaluationCaseDetail"><div className="evaluationExpected"><span>Expected</span><strong>{caseResult.expectedArtifact}</strong><p>{caseResult.previewContract}</p></div><div className="evaluationMetricList">{caseResult.metrics.map((metric) => <article key={`${caseResult.caseId}-${metric.id}`}><header><Status status={metric.status} /><strong>{formatEvaluationMetric(metric.id)}</strong><span>{formatScore(metric.score)}</span></header><p>{metric.detail}</p><small>{metric.kind === "ragas" ? "RAGAS" : "Product-specific"} · {metric.evidenceClass.replace(/_/g, " ")} · target {metric.target == null ? "n/a" : `${Math.round(metric.target * 100)}%`} · gap {metric.gap == null ? "n/a" : `${Math.round(metric.gap * 100)} pts`}</small></article>)}</div>{caseResult.recommendation ? <aside><div><span>Recommended next step</span><strong>{caseResult.recommendation.title}</strong><p>{caseResult.recommendation.detail}</p></div><button onClick={() => onCandidate(caseResult)} type="button">Create improved candidate</button></aside> : null}</div></details>;
}

function EmptyResults() { return <div className="evaluationEmptyState"><FlaskConical aria-hidden="true" size={24} /><strong>No benchmark result yet</strong><p>Run the deterministic local benchmark. Provider access is optional and affects only approved RAGAS evidence.</p></div>; }
function formatScore(value: number | null) { return value == null ? "—" : `${Math.round(value * 100)}%`; }
function formatDelta(value: number | null) { return value == null ? "—" : `${value > 0 ? "+" : ""}${Math.round(value * 100)} pts`; }
function formatDate(value: string) { const date = new Date(value); return Number.isNaN(date.getTime()) ? value : date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }); }
function statusTone(value: EvaluationBenchmarkRun["statusLabel"]): EvaluationMetricStatus { if (value === "Blocked") return "blocked"; if (value === "Needs Improvement") return "fail"; if (value === "Incomplete Evidence") return "missing_evidence"; return "pass"; }
