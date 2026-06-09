import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { LangSmithTraceModel } from "@/lib/langsmith-trace";
import { LangSmithTraceDeepDive } from "./langsmith-trace-deep-dive";

describe("LangSmithTraceDeepDive", () => {
  it("renders a clean empty state for sessions without a trace", () => {
    render(<LangSmithTraceDeepDive trace={emptyTrace()} />);

    expect(
      screen.getByRole("group", { name: "LangSmith trace deep dive" })
    ).toBeVisible();
    expect(screen.getByText("No LangSmith trace for this session")).toBeVisible();
    expect(
      screen.getByText(/Legacy telemetry remains available/)
    ).toBeVisible();
    expect(
      screen.queryByRole("region", { name: "LangSmith trace hierarchy" })
    ).not.toBeInTheDocument();
  });

  it("renders overview, hierarchy, transitions, tags, and metadata", () => {
    render(<LangSmithTraceDeepDive trace={populatedTrace()} />);

    const dashboard = screen.getByRole("group", {
      name: "LangSmith trace deep dive"
    });
    expect(within(dashboard).getByText("LangSmith linked")).toBeVisible();
    expect(within(dashboard).getByLabelText("Trace status Complete")).toBeVisible();

    const overview = within(dashboard).getByRole("region", {
      name: "LangSmith trace overview"
    });
    expect(within(overview).getByText("trace-123")).toBeVisible();
    expect(within(overview).getByText("run-root-1")).toBeVisible();
    expect(within(overview).getByText("3.0 s")).toBeVisible();

    const hierarchy = within(dashboard).getByRole("region", {
      name: "LangSmith trace hierarchy"
    });
    expect(within(hierarchy).getByText("assistant.workflow")).toBeVisible();
    expect(within(hierarchy).getByText("retrieve.context")).toBeVisible();
    expect(hierarchy).toHaveTextContent("Intake -> Retrieval");
    expect(within(hierarchy).getByText(/Official context selected/)).toBeVisible();

    const tags = within(dashboard).getByLabelText("LangSmith execution tags");
    expect(within(tags).getByText("assistant")).toBeVisible();
    expect(within(tags).getByText("generate")).toBeVisible();

    const provider = within(dashboard).getByRole("region", {
      name: "Provider metadata"
    });
    expect(within(provider).getByText("gpt-5-mini")).toBeVisible();
    const retrieval = within(dashboard).getByRole("region", {
      name: "Retrieval metadata"
    });
    expect(within(retrieval).getByText("official_kb")).toBeVisible();
  });

  it("keeps linked traces readable when hierarchy and metadata are absent", () => {
    render(
      <LangSmithTraceDeepDive
        trace={{
          ...populatedTrace(),
          spans: [],
          metadataGroups: populatedTrace().metadataGroups.map((group) => ({
            ...group,
            entries: []
          })),
          tags: [],
          summary: {
            metadataCount: 0,
            nestedSpanCount: 0,
            spanCount: 0,
            transitionCount: 0
          }
        }}
      />
    );

    expect(screen.getByText("No span hierarchy reported")).toBeVisible();
    expect(
      screen.getByText(/No provider, retrieval, evaluation, or execution metadata/)
    ).toBeVisible();
  });
});

function emptyTrace(): LangSmithTraceModel {
  return {
    state: "unavailable",
    status: "idle",
    availabilityLabel: "Trace unavailable",
    statusLabel: "No LangSmith trace",
    traceId: null,
    runId: null,
    parentRunId: null,
    runName: null,
    traceKind: null,
    projectName: null,
    providerLabel: "langsmith",
    startedAt: null,
    endedAt: null,
    durationMs: null,
    tags: [],
    spans: [],
    metadataGroups: [
      { id: "provider", label: "Provider metadata", entries: [] },
      { id: "retrieval", label: "Retrieval metadata", entries: [] },
      { id: "evaluation", label: "Evaluation metadata", entries: [] },
      { id: "execution", label: "Execution metadata", entries: [] }
    ],
    summary: {
      spanCount: 0,
      nestedSpanCount: 0,
      transitionCount: 0,
      metadataCount: 0
    }
  };
}

function populatedTrace(): LangSmithTraceModel {
  return {
    ...emptyTrace(),
    state: "linked",
    status: "complete",
    availabilityLabel: "LangSmith linked",
    statusLabel: "Complete",
    traceId: "trace-123",
    runId: "run-root-1",
    runName: "assistant.workflow",
    traceKind: "assistant_workflow",
    projectName: "creative-prod",
    providerLabel: "langsmith",
    startedAt: "2026-06-09T10:00:00Z",
    endedAt: "2026-06-09T10:00:03Z",
    durationMs: 3000,
    tags: ["assistant", "generate"],
    spans: [
      {
        id: "root",
        parentId: null,
        runId: "run-root-1",
        name: "assistant.workflow",
        runType: "chain",
        stage: "intake",
        status: "complete",
        startedAt: "2026-06-09T10:00:00Z",
        endedAt: "2026-06-09T10:00:03Z",
        durationMs: 3000,
        depth: 0,
        transitionFrom: null,
        transitionReason: null
      },
      {
        id: "retrieval",
        parentId: "root",
        runId: "run-retrieval-1",
        name: "retrieve.context",
        runType: "retriever",
        stage: "retrieval",
        status: "complete",
        startedAt: "2026-06-09T10:00:01Z",
        endedAt: "2026-06-09T10:00:01.600Z",
        durationMs: 600,
        depth: 1,
        transitionFrom: "intake",
        transitionReason: "Official context selected."
      }
    ],
    metadataGroups: [
      {
        id: "provider",
        label: "Provider metadata",
        entries: [{ key: "model", label: "Model", value: "gpt-5-mini" }]
      },
      {
        id: "retrieval",
        label: "Retrieval metadata",
        entries: [{ key: "source", label: "Source", value: "official_kb" }]
      },
      { id: "evaluation", label: "Evaluation metadata", entries: [] },
      {
        id: "execution",
        label: "Execution metadata",
        entries: [{ key: "mode", label: "Mode", value: "generate" }]
      }
    ],
    summary: {
      spanCount: 2,
      nestedSpanCount: 1,
      transitionCount: 1,
      metadataCount: 3
    }
  };
}
