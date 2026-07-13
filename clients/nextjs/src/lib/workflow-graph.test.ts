import { describe, expect, it } from "vitest";
import type { WorkflowExecutionModel } from "./workflow-execution";
import {
  formatWorkflowGraphRoute,
  selectWorkflowGraphSteps
} from "./workflow-graph";
import type { WorkflowRuntimeStep } from "./workflow-runtime";

const idleExecution: WorkflowExecutionModel = {
  state: "idle",
  requestedMode: "auto",
  resolvedMode: null,
  rationale: "Auto is ready for the next run.",
  agentRoles: [],
  researcherRequired: null,
  researcherReason: null,
  maxRefinementLoops: null,
  source: "none"
};

const steps = [
  workflowStep("intake"),
  workflowStep("planning"),
  workflowStep("generation"),
  workflowStep("artifact_critique"),
  workflowStep("finalization")
];

describe("workflow graph route projection", () => {
  it("shows only nodes that exist on the selected Single Agent graph", () => {
    expect(
      selectWorkflowGraphSteps({
        execution: idleExecution,
        requestedMode: "single_agent",
        steps
      }).map((step) => step.nodeId)
    ).toEqual(["intake", "generation", "finalization"]);
  });

  it("keeps the additional planning and review nodes for Multi Agent", () => {
    expect(
      selectWorkflowGraphSteps({
        execution: idleExecution,
        requestedMode: "multi_agent",
        steps
      }).map((step) => step.nodeId)
    ).toEqual(steps.map((step) => step.nodeId));
  });

  it("projects Auto to the route actually published by the runtime", () => {
    const execution: WorkflowExecutionModel = {
      ...idleExecution,
      state: "available",
      requestedMode: "auto",
      resolvedMode: "single_agent",
      rationale: "Auto selected the bounded single-agent route.",
      agentRoles: ["generator"],
      researcherRequired: false,
      researcherReason: "No separate researcher was selected.",
      maxRefinementLoops: 0,
      source: "stream"
    };

    expect(formatWorkflowGraphRoute({ execution, requestedMode: "auto" })).toBe(
      "Auto → Single Agent"
    );
    expect(
      selectWorkflowGraphSteps({ execution, requestedMode: "auto", steps }).map(
        (step) => step.nodeId
      )
    ).toEqual(["intake", "generation", "finalization"]);
  });
});

function workflowStep(nodeId: WorkflowRuntimeStep["nodeId"]): WorkflowRuntimeStep {
  return {
    nodeId,
    displayLabel: nodeId,
    detail: "Runtime node",
    state: "queued",
    attemptCount: 0,
    eventCount: 0,
    startedAt: null,
    completedAt: null,
    durationMs: null,
    lastUpdatedAt: null,
    lastEventLabel: null,
    lastEventDetail: null
  };
}
