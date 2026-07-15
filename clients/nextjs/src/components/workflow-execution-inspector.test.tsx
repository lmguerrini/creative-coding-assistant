import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { WorkflowExecutionInspector } from "./workflow-execution-inspector";

describe("WorkflowExecutionInspector", () => {
  it("publishes the two-attempt refinement and three-call generation budgets", () => {
    render(
      <WorkflowExecutionInspector
        execution={{
          state: "available",
          requestedMode: "auto",
          resolvedMode: "multi_agent",
          rationale: "The request benefits from bounded review.",
          agentRoles: ["planner", "researcher", "generator", "critic", "reviewer"],
          researcherRequired: true,
          researcherReason: "Approved context is available.",
          maxRefinementLoops: 2,
          source: "stream"
        }}
      />
    );

    expect(screen.getByRole("group", { name: "Workflow execution decision" }))
      .toHaveTextContent(
        "Refinement budget: 2 bounded attempts. Text-generation budget: up to 3 provider calls."
      );
  });
});
