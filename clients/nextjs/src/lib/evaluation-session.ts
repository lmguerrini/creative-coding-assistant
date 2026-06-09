import { readEventTimestamp, type AssistantStreamEvent } from "./assistant-stream";
import type { WorkflowRuntimeTraceEvent } from "./workflow-runtime";

export type EvaluationOutcome = "pass" | "warn" | "fail" | "unscored";

export type EvaluationSignalId =
  | "answer"
  | "retrieval"
  | "grounding"
  | "artifact"
  | "provider-runtime";

export type EvaluationSessionSignal = {
  id: EvaluationSignalId;
  label: string;
  score: number | null;
  outcome: EvaluationOutcome;
  metrics: string[];
  detail: string;
};

export type EvaluationSessionModel = {
  state: "available" | "pending" | "unavailable";
  runId: string | null;
  datasetId: string | null;
  metrics: string[];
  resultRows: number | null;
  metricFailures: number | null;
  dryRun: boolean | null;
  providerCallsAllowed: boolean | null;
  statusLabel: string;
  detail: string;
  latestAt: string | null;
  evaluationType: string;
  score: number | null;
  outcome: EvaluationOutcome;
  signals: EvaluationSessionSignal[];
};

type EvaluationObservability = {
  traceKind: string | null;
  latestAt: string | null;
};

type EvaluationRecord = {
  runId: string | null;
  datasetId: string | null;
  metrics: string[];
  metricScores: Record<string, number>;
  resultRows: number | null;
  metricFailures: number | null;
  dryRun: boolean | null;
  providerCallsAllowed: boolean | null;
  status: string | null;
  detail: string | null;
  evaluationType: string | null;
  score: number | null;
  outcome: EvaluationOutcome | null;
  evaluatedAt: string | null;
};

type SignalDefinition = {
  id: EvaluationSignalId;
  label: string;
  metricNames: string[];
};

const PASS_THRESHOLD = 0.8;
const WARN_THRESHOLD = 0.6;

const signalDefinitions: SignalDefinition[] = [
  {
    id: "answer",
    label: "Answer quality",
    metricNames: [
      "answer_relevancy",
      "answer_relevance",
      "answer_quality",
      "response_quality",
      "response_relevancy"
    ]
  },
  {
    id: "retrieval",
    label: "Retrieval quality",
    metricNames: [
      "context_precision",
      "context_recall",
      "retrieval_quality",
      "retrieval_precision",
      "retrieval_coverage"
    ]
  },
  {
    id: "grounding",
    label: "Grounding quality",
    metricNames: [
      "faithfulness",
      "groundedness",
      "grounding_quality",
      "context_relevance"
    ]
  },
  {
    id: "artifact",
    label: "Artifact quality",
    metricNames: [
      "artifact_quality",
      "artifact_score",
      "code_quality",
      "creative_quality"
    ]
  },
  {
    id: "provider-runtime",
    label: "Provider / runtime",
    metricNames: [
      "provider_quality",
      "runtime_quality",
      "execution_quality",
      "provider_runtime_quality"
    ]
  }
];

const knownMetricNames = new Set(
  signalDefinitions.flatMap((definition) => definition.metricNames)
);

export function buildEvaluationSessionModel(
  traceEvents: WorkflowRuntimeTraceEvent[],
  observability: EvaluationObservability
): EvaluationSessionModel {
  const latest = findLatestEvaluationRecord(traceEvents);
  if (latest) {
    const score =
      latest.record.score ?? averageScores(Object.values(latest.record.metricScores));
    const outcome = latest.record.outcome ?? outcomeFromScore(score);

    return {
      state: "available",
      runId: latest.record.runId,
      datasetId: latest.record.datasetId,
      metrics: latest.record.metrics,
      resultRows: latest.record.resultRows,
      metricFailures: latest.record.metricFailures,
      dryRun: latest.record.dryRun,
      providerCallsAllowed: latest.record.providerCallsAllowed,
      statusLabel: latest.record.status ?? "Evaluation results captured",
      detail:
        latest.record.detail ??
        (latest.record.datasetId
          ? `Dataset ${latest.record.datasetId}`
          : "Evaluation results were captured for this session."),
      latestAt: latest.record.evaluatedAt ?? latest.at,
      evaluationType:
        latest.record.evaluationType ??
        inferEvaluationType(latest.event, latest.record.metrics),
      score,
      outcome,
      signals: buildSignals(latest.record.metricScores, latest.record.metrics)
    };
  }

  if (observability.traceKind === "ragas_evaluation") {
    return {
      state: "available",
      runId: null,
      datasetId: null,
      metrics: [],
      resultRows: null,
      metricFailures: null,
      dryRun: null,
      providerCallsAllowed: null,
      statusLabel: "RAGAs trace linked",
      detail:
        "Evaluation lineage is available through the LangSmith trace metadata.",
      latestAt: observability.latestAt,
      evaluationType: "RAGAs",
      score: null,
      outcome: "unscored",
      signals: buildSignals({}, [])
    };
  }

  const state = traceEvents.length > 0 ? "pending" : "unavailable";
  return {
    state,
    runId: null,
    datasetId: null,
    metrics: [],
    resultRows: null,
    metricFailures: null,
    dryRun: null,
    providerCallsAllowed: null,
    statusLabel:
      state === "pending" ? "No evaluation event in stream" : "No evaluation run",
    detail:
      "Session evaluation appears here when eval_update events or linked evaluation traces are available.",
    latestAt: null,
    evaluationType: "Not evaluated",
    score: null,
    outcome: "unscored",
    signals: buildSignals({}, [])
  };
}

function findLatestEvaluationRecord(
  traceEvents: WorkflowRuntimeTraceEvent[]
) {
  for (let index = traceEvents.length - 1; index >= 0; index -= 1) {
    const traceEvent = traceEvents[index];
    const record = parseEvaluationRecord(traceEvent.event);
    if (record) {
      return {
        at: readEventTimestamp(traceEvent.event) ?? traceEvent.receivedAt,
        event: traceEvent.event,
        record
      };
    }
  }
  return null;
}

function parseEvaluationRecord(event: AssistantStreamEvent): EvaluationRecord | null {
  if (event.event_type !== "eval_update") {
    return null;
  }

  const payload = event.payload;
  const record =
    readRecord(payload.evaluation) ??
    readRecord(payload.ragas) ??
    readRecord(payload.result) ??
    payload;
  const dataset = readRecord(record.dataset);
  const observability = readRecord(record.observability) ?? readRecord(record.langsmith);
  const observabilityMetadata = readRecord(observability?.metadata);
  const summary = readRecord(record.summary);
  const metricScores = readMetricScores(record);
  const declaredMetrics = readMetricNames(record.metrics);
  const datasetMetrics = readMetricNames(dataset?.metrics);
  const metrics = uniqueStrings([
    ...declaredMetrics,
    ...datasetMetrics,
    ...Object.keys(metricScores)
  ]);

  return {
    runId:
      readString(record.run_id) ??
      readString(record.runId) ??
      readString(observabilityMetadata?.run_id),
    datasetId:
      readString(record.dataset_id) ??
      readString(record.datasetId) ??
      readString(dataset?.dataset_id) ??
      readString(dataset?.datasetId),
    metrics,
    metricScores,
    resultRows:
      readNumber(record.result_rows) ??
      readNumber(record.resultRows) ??
      readArrayLength(record.result_rows) ??
      readArrayLength(record.resultRows),
    metricFailures:
      readNumber(record.metric_failures) ?? readNumber(record.metricFailures),
    dryRun: readOptionalBoolean(record.dry_run) ?? readOptionalBoolean(record.dryRun),
    providerCallsAllowed:
      readOptionalBoolean(record.provider_calls_allowed) ??
      readOptionalBoolean(record.providerCallsAllowed),
    status: readString(record.status) ?? readString(payload.code),
    detail: readString(record.detail) ?? readString(payload.message),
    evaluationType:
      readString(record.evaluation_type) ??
      readString(record.evaluationType) ??
      readString(record.type) ??
      readString(record.kind),
    score:
      readScore(record.score) ??
      readScore(record.overall_score) ??
      readScore(record.overallScore) ??
      readScore(record.evaluation_score) ??
      readScore(record.evaluationScore) ??
      readScore(summary?.score) ??
      readScore(summary?.overall_score),
    outcome:
      readOutcome(record.outcome) ??
      readOutcome(record.verdict) ??
      readOutcome(record.quality_status) ??
      readOutcome(record.result_status) ??
      readOutcome(record.status),
    evaluatedAt:
      readString(record.evaluated_at) ??
      readString(record.evaluatedAt) ??
      readString(summary?.evaluated_at)
  };
}

function readMetricScores(record: Record<string, unknown>) {
  const scoreMaps = [
    record.metric_scores,
    record.metricScores,
    record.scores,
    record.results,
    readRecord(record.summary)?.scores,
    readRecord(record.summary)?.metric_scores,
    isRecord(record.metrics) ? record.metrics : null,
    record.signals
  ];
  const scores: Record<string, number[]> = {};

  for (const scoreMap of scoreMaps) {
    collectScoreMap(scoreMap, scores);
  }
  collectResultRows(record.result_rows, scores);
  collectResultRows(record.resultRows, scores);
  collectResultRows(record.results, scores);

  for (const metricName of knownMetricNames) {
    const score = readScore(record[metricName]);
    if (score != null) {
      (scores[metricName] ??= []).push(score);
    }
  }

  return Object.fromEntries(
    Object.entries(scores)
      .map(([metricName, values]) => [metricName, averageScores(values)])
      .filter((entry): entry is [string, number] => entry[1] != null)
  );
}

function collectScoreMap(
  value: unknown,
  scores: Record<string, number[]>
) {
  if (!isRecord(value)) {
    return;
  }

  for (const [rawName, rawValue] of Object.entries(value)) {
    const metricName = canonicalSignalMetricName(rawName);
    const nested = readRecord(rawValue);
    const score =
      readScore(rawValue) ??
      readScore(nested?.score) ??
      readScore(nested?.value) ??
      readScore(nested?.overall_score);
    if (score != null) {
      (scores[metricName] ??= []).push(score);
    }
  }
}

function collectResultRows(
  value: unknown,
  scores: Record<string, number[]>
) {
  if (!Array.isArray(value)) {
    return;
  }

  for (const row of value) {
    const record = readRecord(row);
    if (record) {
      collectScoreMap(record.metrics ?? record.scores, scores);
    }
  }
}

function buildSignals(
  metricScores: Record<string, number>,
  declaredMetrics: string[]
): EvaluationSessionSignal[] {
  return signalDefinitions.map((definition) => {
    const matchingMetrics = definition.metricNames.filter(
      (metricName) =>
        metricScores[metricName] != null || declaredMetrics.includes(metricName)
    );
    const matchingScores = definition.metricNames
      .map((metricName) => ({
        metricName,
        score: metricScores[metricName]
      }))
      .filter(
        (entry): entry is { metricName: string; score: number } =>
          entry.score != null
      );
    const score = averageScores(matchingScores.map((entry) => entry.score));

    return {
      id: definition.id,
      label: definition.label,
      score,
      outcome: outcomeFromScore(score),
      metrics: matchingMetrics,
      detail:
        matchingScores.length > 0
          ? `${matchingScores.map((entry) => formatMetricLabel(entry.metricName)).join(" / ")}`
          : matchingMetrics.length > 0
            ? `${matchingMetrics.map(formatMetricLabel).join(" / ")} recorded without a score.`
          : `No ${definition.label.toLowerCase()} metric in the latest evaluation.`
    };
  });
}

function inferEvaluationType(
  event: AssistantStreamEvent,
  metrics: string[]
) {
  const code = readString(event.payload.code)?.toLowerCase() ?? "";
  if (
    code.includes("ragas") ||
    metrics.some((metric) =>
      ["answer_relevancy", "context_precision", "context_recall", "faithfulness"].includes(
        metric
      )
    )
  ) {
    return "RAGAs";
  }
  return "Session evaluation";
}

function outcomeFromScore(score: number | null): EvaluationOutcome {
  if (score == null) {
    return "unscored";
  }
  if (score >= PASS_THRESHOLD) {
    return "pass";
  }
  if (score >= WARN_THRESHOLD) {
    return "warn";
  }
  return "fail";
}

function readOutcome(value: unknown): EvaluationOutcome | null {
  const normalized = readString(value)?.toLowerCase();
  if (normalized === "pass" || normalized === "passed") {
    return "pass";
  }
  if (
    normalized === "warn" ||
    normalized === "warning" ||
    normalized === "degraded"
  ) {
    return "warn";
  }
  if (normalized === "fail" || normalized === "failed" || normalized === "error") {
    return "fail";
  }
  return null;
}

function readMetricNames(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value
      .filter((item): item is string => typeof item === "string")
      .map(normalizeMetricName);
  }
  return isRecord(value) ? Object.keys(value).map(normalizeMetricName) : [];
}

function normalizeMetricName(value: string) {
  return value.trim().toLowerCase().replace(/[\s-]+/g, "_");
}

function canonicalSignalMetricName(value: string) {
  const normalized = normalizeMetricName(value);
  switch (normalized) {
    case "answer":
      return "answer_quality";
    case "retrieval":
      return "retrieval_quality";
    case "grounding":
      return "grounding_quality";
    case "artifact":
      return "artifact_quality";
    case "provider":
    case "provider_runtime":
    case "runtime":
      return "provider_runtime_quality";
    default:
      return normalized;
  }
}

function formatMetricLabel(value: string) {
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function averageScores(scores: number[]): number | null {
  if (scores.length === 0) {
    return null;
  }
  return scores.reduce((total, score) => total + score, 0) / scores.length;
}

function uniqueStrings(values: string[]) {
  return [...new Set(values)];
}

function readScore(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    return null;
  }
  if (value <= 1) {
    return value;
  }
  return value <= 100 ? value / 100 : null;
}

function readArrayLength(value: unknown): number | null {
  return Array.isArray(value) ? value.length : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readRecord(value: unknown): Record<string, unknown> | null {
  return isRecord(value) ? value : null;
}

function readString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function readOptionalBoolean(value: unknown): boolean | null {
  return typeof value === "boolean" ? value : null;
}

function readNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}
