import type { WorkflowRuntimeTraceEvent } from "./workflow-runtime";

export type WorkflowExecutionMode =
  | "auto"
  | "single_agent"
  | "multi_agent";

export type WorkflowExecutionState = "idle" | "available" | "unavailable";

export type WorkflowExecutionModel = {
  state: WorkflowExecutionState;
  requestedMode: WorkflowExecutionMode;
  resolvedMode: Exclude<WorkflowExecutionMode, "auto"> | null;
  rationale: string;
  agentRoles: string[];
  researcherRequired: boolean | null;
  researcherReason: string | null;
  maxRefinementLoops: number | null;
  source: "stream" | "none";
};

const validModes = new Set<WorkflowExecutionMode>([
  "auto",
  "single_agent",
  "multi_agent"
]);

export function buildWorkflowExecutionModel(
  traceEvents: readonly WorkflowRuntimeTraceEvent[]
): WorkflowExecutionModel {
  for (let index = traceEvents.length - 1; index >= 0; index -= 1) {
    const plan = readExecutionPlan(traceEvents[index]?.event.payload ?? {});
    if (!plan) {
      continue;
    }

    return {
      state: "available",
      requestedMode: plan.requestedMode,
      resolvedMode: plan.resolvedMode,
      rationale: plan.rationale,
      agentRoles: plan.agentRoles,
      researcherRequired: plan.researcherRequired,
      researcherReason: plan.researcherReason,
      maxRefinementLoops: plan.maxRefinementLoops,
      source: "stream"
    };
  }

  return {
    state: traceEvents.length === 0 ? "idle" : "unavailable",
    requestedMode: "auto",
    resolvedMode: null,
    rationale:
      traceEvents.length === 0
        ? "Auto is ready for the next run."
        : "This run did not publish an execution decision.",
    agentRoles: [],
    researcherRequired: null,
    researcherReason: null,
    maxRefinementLoops: null,
    source: "none"
  };
}

function readExecutionPlan(payload: Record<string, unknown>) {
  const route = asRecord(payload.route);
  const execution = asRecord(route?.execution) ?? asRecord(payload.execution);
  if (!execution) {
    return null;
  }

  const requestedMode = readMode(execution.requested_mode ?? execution.requestedMode);
  const resolvedMode = readMode(execution.resolved_mode ?? execution.resolvedMode);
  if (!requestedMode || resolvedMode === "auto" || !resolvedMode) {
    return null;
  }

  return {
    requestedMode,
    resolvedMode,
    rationale:
      readString(execution.rationale) ?? "The runtime selected this bounded route.",
    agentRoles: readStringList(execution.agent_roles ?? execution.agentRoles),
    researcherRequired: readBoolean(
      execution.researcher_required ?? execution.researcherRequired
    ),
    researcherReason:
      readString(execution.researcher_reason ?? execution.researcherReason) ?? null,
    maxRefinementLoops: readNumber(
      execution.max_refinement_loops ?? execution.maxRefinementLoops
    )
  };
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function readMode(value: unknown): WorkflowExecutionMode | null {
  return typeof value === "string" && validModes.has(value as WorkflowExecutionMode)
    ? (value as WorkflowExecutionMode)
    : null;
}

function readString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function readStringList(value: unknown): string[] {
  return Array.isArray(value)
    ? value.flatMap((item) => (typeof item === "string" && item.trim() ? [item] : []))
    : [];
}

function readBoolean(value: unknown): boolean | null {
  return typeof value === "boolean" ? value : null;
}

function readNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}
