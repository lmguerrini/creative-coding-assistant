import { describe, expect, it } from "vitest";
import { buildWorkflowExecutionModel } from "./workflow-execution";
import type { WorkflowRuntimeTraceEvent } from "./workflow-runtime";

describe("workflow execution observability", () => {
  it("reads the resolved bounded execution plan from runtime evidence", () => {
    const model = buildWorkflowExecutionModel([
      traceEvent({
        requested_mode: "auto",
        resolved_mode: "multi_agent",
        rationale: "Auto selected the bounded route.",
        agent_roles: ["planner", "researcher", "generator", "critic", "reviewer"],
        researcher_required: true,
        researcher_reason: "Planner requested bounded retrieval.",
        max_refinement_loops: 1
      })
    ]);

    expect(model).toMatchObject({
      state: "available",
      requestedMode: "auto",
      resolvedMode: "multi_agent",
      researcherRequired: true,
      maxRefinementLoops: 1,
      source: "stream"
    });
    expect(model.agentRoles).toEqual([
      "planner",
      "researcher",
      "generator",
      "critic",
      "reviewer"
    ]);
  });

  it("does not invent a route when no execution evidence exists", () => {
    expect(buildWorkflowExecutionModel([])).toMatchObject({
      state: "idle",
      requestedMode: "auto",
      resolvedMode: null,
      source: "none"
    });
  });
});

function traceEvent(execution: Record<string, unknown>): WorkflowRuntimeTraceEvent {
  const at = "2026-07-11T10:00:00Z";
  return {
    event: {
      event_type: "status",
      sequence: 2,
      payload: {
        code: "route_selected",
        emitted_at: at,
        route: { execution }
      }
    },
    receivedAt: at,
    receivedAtMs: Date.parse(at)
  };
}
