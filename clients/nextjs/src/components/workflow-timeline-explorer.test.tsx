import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { WorkflowTimelineModel } from "@/lib/workflow-timeline";
import { WorkflowTimelineExplorer } from "./workflow-timeline-explorer";

describe("WorkflowTimelineExplorer", () => {
  it("renders a clean first-run empty state", () => {
    render(<WorkflowTimelineExplorer timeline={emptyTimeline()} />);

    const explorer = screen.getByRole("group", {
      name: "Workflow timeline explorer"
    });
    expect(within(explorer).getByText("No workflow timeline yet")).toBeVisible();
    expect(within(explorer).getByText("Awaiting run")).toBeVisible();
    expect(
      within(explorer).queryByRole("list", {
        name: "Chronological workflow events"
      })
    ).not.toBeInTheDocument();
  });

  it("renders ordered timeline metadata, duration, and transition reasons", () => {
    render(<WorkflowTimelineExplorer timeline={recordedTimeline()} />);

    const list = screen.getByRole("list", {
      name: "Chronological workflow events"
    });
    const events = within(list).getAllByRole("listitem");
    expect(events).toHaveLength(3);
    expect(events[0]).toHaveTextContent("Request received");
    expect(events[1]).toHaveTextContent("Provider generation completed");
    expect(events[1]).toHaveTextContent("Generation");
    expect(events[1]).toHaveTextContent("1.5 s");
    expect(events[1]).toHaveTextContent("Generation Completed");
    expect(events[2]).toHaveTextContent("Final response");
    expect(
      screen.getByRole("group", { name: "Workflow timeline summary" })
    ).toHaveTextContent("3.0 s");
  });

  it("renders warning and error callouts with explicit status", () => {
    const timeline = recordedTimeline();
    timeline.events[1] = {
      ...timeline.events[1],
      status: "warning",
      warning: "Review requested refinement."
    };
    timeline.events[2] = {
      ...timeline.events[2],
      status: "error",
      error: "Finalization failed."
    };
    timeline.summary.warningCount = 1;
    timeline.summary.errorCount = 1;

    render(<WorkflowTimelineExplorer timeline={timeline} />);

    expect(screen.getByText("Review requested refinement.")).toBeVisible();
    expect(screen.getByText("Finalization failed.")).toBeVisible();
    expect(screen.getAllByText("Warning").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Error").length).toBeGreaterThan(0);
  });
});

function emptyTimeline(): WorkflowTimelineModel {
  return {
    state: "empty",
    events: [],
    summary: {
      eventCount: 0,
      warningCount: 0,
      errorCount: 0,
      startedAt: null,
      completedAt: null,
      totalDurationMs: null
    }
  };
}

function recordedTimeline(): WorkflowTimelineModel {
  return {
    state: "available",
    events: [
      {
        id: "0-status",
        sequence: 0,
        eventType: "status",
        label: "Request received",
        detail: "Request accepted.",
        at: "2026-06-01T10:00:00Z",
        phase: "running",
        nodeId: "intake",
        stageLabel: "Intake",
        status: "info",
        durationMs: null,
        warning: null,
        error: null,
        transitionReason: null
      },
      {
        id: "1-node",
        sequence: 1,
        eventType: "node_completed",
        label: "Provider generation completed",
        detail: "Generation completed.",
        at: "2026-06-01T10:00:01.500Z",
        phase: "completed",
        nodeId: "generation",
        stageLabel: "Generation",
        status: "complete",
        durationMs: 1500,
        warning: null,
        error: null,
        transitionReason: "generation_completed"
      },
      {
        id: "2-final",
        sequence: 2,
        eventType: "final",
        label: "Final response",
        detail: "The final response was emitted to the creative session.",
        at: "2026-06-01T10:00:03Z",
        phase: "completed",
        nodeId: "finalization",
        stageLabel: "Finalization",
        status: "complete",
        durationMs: null,
        warning: null,
        error: null,
        transitionReason: null
      }
    ],
    summary: {
      eventCount: 3,
      warningCount: 0,
      errorCount: 0,
      startedAt: "2026-06-01T10:00:00Z",
      completedAt: "2026-06-01T10:00:03Z",
      totalDurationMs: 3000
    }
  };
}
