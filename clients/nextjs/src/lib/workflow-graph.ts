import type { WorkflowNodeId } from "./assistant-client";
import type {
  WorkflowExecutionMode,
  WorkflowExecutionModel
} from "./workflow-execution";
import type { WorkflowRuntimeStep } from "./workflow-runtime";

export type ResolvedWorkflowGraphRoute = Exclude<WorkflowExecutionMode, "auto">;

const multiAgentOnlyNodes = new Set<WorkflowNodeId>([
  "planning",
  "director",
  "reasoning",
  "artifact_critique",
  "review",
  "refinement"
]);

export function resolveWorkflowGraphRoute({
  execution,
  requestedMode
}: {
  execution: WorkflowExecutionModel;
  requestedMode: WorkflowExecutionMode;
}): ResolvedWorkflowGraphRoute | null {
  if (requestedMode !== "auto") {
    return requestedMode;
  }

  if (
    execution.state === "available" &&
    execution.resolvedMode
  ) {
    return execution.resolvedMode;
  }

  return null;
}

export function selectWorkflowGraphSteps({
  execution,
  requestedMode,
  steps
}: {
  execution: WorkflowExecutionModel;
  requestedMode: WorkflowExecutionMode;
  steps: readonly WorkflowRuntimeStep[];
}): WorkflowRuntimeStep[] {
  const route = resolveWorkflowGraphRoute({ execution, requestedMode });

  if (route !== "single_agent") {
    return [...steps];
  }

  return steps.filter((step) => !multiAgentOnlyNodes.has(step.nodeId));
}

export function formatWorkflowGraphRoute({
  execution,
  requestedMode
}: {
  execution: WorkflowExecutionModel;
  requestedMode: WorkflowExecutionMode;
}) {
  const route = resolveWorkflowGraphRoute({ execution, requestedMode });

  if (requestedMode === "auto") {
    return route
      ? `Auto (${formatResolvedRoute(route)})`
      : "Auto · route pending";
  }

  return formatResolvedRoute(route ?? requestedMode);
}

function formatResolvedRoute(route: ResolvedWorkflowGraphRoute) {
  return route === "single_agent" ? "Single Agent" : "Multi Agent";
}
