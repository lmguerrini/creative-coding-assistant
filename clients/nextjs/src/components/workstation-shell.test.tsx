import {
  act,
  fireEvent,
  render,
  screen,
  within
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { WorkstationShell } from "./workstation-shell";
import {
  getLocalWorkspaceSnapshot,
  type AssistantWorkspaceSnapshot,
  type InspectorTabName
} from "@/lib/assistant-client";
import type { AssistantStreamEvent } from "@/lib/assistant-stream";

function snapshotWithActiveTab(
  activeTab: InspectorTabName
): AssistantWorkspaceSnapshot {
  const snapshot = getLocalWorkspaceSnapshot();

  return {
    ...snapshot,
    inspectorTabs: snapshot.inspectorTabs.map((tab) => ({
      ...tab,
      active: tab.label === activeTab
    }))
  };
}

async function* streamEvents(
  events: AssistantStreamEvent[]
): AsyncGenerator<AssistantStreamEvent> {
  for (const event of events) {
    yield event;
  }
}

async function* failingStream(): AsyncGenerator<AssistantStreamEvent> {
  throw new Error("offline");
}

describe("WorkstationShell", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("renders the three-zone creative workspace shell", () => {
    render(<WorkstationShell snapshot={getLocalWorkspaceSnapshot()} />);

    expect(screen.getByText("Creative Coding Assistant")).toBeVisible();
    expect(screen.getByRole("region", { name: "Creative session" })).toBeVisible();
    expect(screen.getByRole("region", { name: "Preview workspace" })).toBeVisible();
    expect(screen.getByRole("complementary", { name: "Right inspector" })).toBeVisible();
    expect(screen.getByRole("tablist", { name: "Inspector tabs" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Dashboard" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Theme" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Settings" })).toBeVisible();
  });

  it("defaults to a single Overview inspector panel", () => {
    render(<WorkstationShell snapshot={getLocalWorkspaceSnapshot()} />);

    for (const tab of ["Overview", "Code", "Workflow", "Artifacts", "Retrieval"]) {
      expect(screen.getByRole("tab", { name: tab })).toBeVisible();
    }

    expect(screen.getByRole("tab", { name: "Overview" })).toHaveAttribute(
      "aria-selected",
      "true"
    );
    expect(screen.getAllByRole("tabpanel")).toHaveLength(1);
    expect(screen.getByRole("tabpanel", { name: "Overview inspector" })).toBeVisible();
    expect(screen.getByRole("group", { name: "Workflow summary" })).toBeVisible();
    expect(screen.getByRole("group", { name: "Artifacts summary" })).toBeVisible();
    expect(screen.getByRole("group", { name: "Preview summary" })).toBeVisible();
    expect(screen.getByRole("group", { name: "Retrieval summary" })).toBeVisible();
    expect(
      screen.getByRole("progressbar", { name: "Overview workflow progress" })
    ).toHaveAttribute("aria-valuetext", "8 of 11 workflow nodes reached");
    expect(screen.queryByRole("tabpanel", { name: "Code inspector" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Preview" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Review" })).not.toBeInTheDocument();
  });

  it("switches inspector tabs without stacking panels", () => {
    render(<WorkstationShell snapshot={getLocalWorkspaceSnapshot()} />);

    fireEvent.click(screen.getByRole("tab", { name: "Code" }));

    expect(screen.getByRole("tab", { name: "Code" })).toHaveAttribute(
      "aria-selected",
      "true"
    );
    expect(screen.getByRole("tab", { name: "Code" })).toHaveAttribute(
      "data-active",
      "true"
    );
    expect(screen.getAllByRole("tabpanel")).toHaveLength(1);
    expect(screen.getByRole("tabpanel", { name: "Code inspector" })).toBeVisible();
    expect(screen.queryByRole("tabpanel", { name: "Overview inspector" })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "Retrieval" }));

    expect(screen.getByRole("tab", { name: "Retrieval" })).toHaveAttribute(
      "aria-selected",
      "true"
    );
    expect(screen.getAllByRole("tabpanel")).toHaveLength(1);
    expect(screen.getByRole("tabpanel", { name: "Retrieval inspector" })).toBeVisible();
  });

  it("streams backend events into the conversation and workflow state", async () => {
    const backendStream = vi.fn(() =>
      streamEvents([
        {
          event_type: "status",
          sequence: 0,
          payload: { code: "request_received", message: "Request accepted." }
        },
        {
          event_type: "status",
          sequence: 1,
          payload: { code: "route_selected", message: "Route selected." }
        },
        {
          event_type: "token_delta",
          sequence: 2,
          payload: { text: "Streaming " }
        },
        {
          event_type: "token_delta",
          sequence: 3,
          payload: { text: "answer." }
        },
        {
          event_type: "preview_artifact",
          sequence: 4,
          payload: { artifact_id: "preview-manifest", status: "ready" }
        },
        {
          event_type: "final",
          sequence: 5,
          payload: { answer: "Final backend answer." }
        }
      ])
    );

    render(
      <WorkstationShell
        snapshot={getLocalWorkspaceSnapshot()}
        streamAssistantEvents={backendStream}
      />
    );

    const promptInput = screen.getByLabelText("Assistant prompt");
    const sendButton = screen.getByRole("button", { name: "Send prompt" });

    expect(sendButton).toBeDisabled();
    expect(sendButton).toHaveAttribute("data-ready", "false");
    expect(screen.getByText("Type to stream backend")).toBeVisible();

    fireEvent.change(promptInput, {
      target: { value: "Make the low-frequency motion calmer." }
    });
    expect(sendButton).toHaveAttribute("data-ready", "true");
    expect(screen.getByText("Ready to stream")).toBeVisible();

    fireEvent.click(sendButton);

    expect(promptInput).toHaveValue("");
    expect(await screen.findByText("Final backend answer.")).toBeVisible();
    expect(backendStream).toHaveBeenCalledWith(
      expect.objectContaining({
        conversationId: "local-nextjs-session",
        domain: "webgpu_wgsl",
        mode: "generate",
        projectId: "local-nextjs-workspace",
        query: "Make the low-frequency motion calmer."
      })
    );
    expect(screen.getByLabelText("Current session")).toHaveTextContent(
      "Finalization"
    );
    expect(
      screen.getByRole("progressbar", { name: "Overview workflow progress" })
    ).toHaveAttribute("aria-valuenow", "11");

    const preview = screen.getByRole("region", { name: "Preview workspace" });
    expect(within(preview).getByText("preview-request.json")).toBeVisible();
    expect(preview.querySelector("details")).toHaveAttribute("open");
  });

  it("falls back to the local mock path when the backend stream is unavailable", async () => {
    vi.useFakeTimers();
    render(
      <WorkstationShell
        snapshot={getLocalWorkspaceSnapshot()}
        streamAssistantEvents={failingStream}
      />
    );

    const promptInput = screen.getByLabelText("Assistant prompt");
    const sendButton = screen.getByRole("button", { name: "Send prompt" });

    expect(sendButton).toBeDisabled();
    expect(sendButton).toHaveAttribute("data-ready", "false");
    expect(screen.getByText("Type to stream backend")).toBeVisible();

    fireEvent.change(promptInput, {
      target: { value: "Make the low-frequency motion calmer." }
    });
    expect(sendButton).toHaveAttribute("data-ready", "true");
    expect(screen.getByText("Ready to stream")).toBeVisible();

    fireEvent.click(sendButton);

    await act(async () => {
      await Promise.resolve();
    });

    expect(promptInput).toHaveValue("");
    expect(screen.getByText(/Backend stream unavailable/)).toBeVisible();
    const userMessage = screen
      .getByText("Make the low-frequency motion calmer.")
      .closest("article");
    const assistantMessage = screen
      .getByText(/Mock orchestration pass started/)
      .closest("article");

    expect(userMessage).toHaveAttribute("data-fresh", "true");
    expect(assistantMessage).toHaveAttribute("data-fresh", "true");
    expect(screen.getByLabelText("Current session")).toHaveTextContent("Intake");
    expect(screen.getByText("Backend fallback active")).toBeVisible();
    expect(
      screen.getByRole("progressbar", { name: "Overview workflow progress" })
    ).toHaveAttribute("aria-valuenow", "1");

    act(() => {
      vi.advanceTimersByTime(850);
    });

    expect(screen.getByLabelText("Current session")).toHaveTextContent("Routing");
    expect(
      screen.getByRole("progressbar", { name: "Overview workflow progress" })
    ).toHaveAttribute("aria-valuenow", "2");
  });

  it("surfaces backend error events without losing the user message", async () => {
    render(
      <WorkstationShell
        snapshot={getLocalWorkspaceSnapshot()}
        streamAssistantEvents={() =>
          streamEvents([
            {
              event_type: "status",
              sequence: 0,
              payload: { code: "request_received", message: "Request accepted." }
            },
            {
              event_type: "error",
              sequence: 1,
              payload: {
                code: "provider_unavailable",
                message: "Provider unavailable."
              }
            }
          ])
        }
      />
    );

    fireEvent.change(screen.getByLabelText("Assistant prompt"), {
      target: { value: "Generate a reactive sketch." }
    });
    fireEvent.click(screen.getByRole("button", { name: "Send prompt" }));

    expect(screen.getByText("Generate a reactive sketch.")).toBeVisible();
    expect(
      await screen.findByText("Backend stream error: Provider unavailable.")
    ).toBeVisible();
    expect(screen.getByText("Backend fallback active")).toBeVisible();
  });

  it("keeps preview available, on demand, and collapsible in the main column", () => {
    render(<WorkstationShell snapshot={getLocalWorkspaceSnapshot()} />);

    const preview = screen.getByRole("region", { name: "Preview workspace" });
    const details = preview.querySelector("details");
    const summary = within(preview).getByText("Preview available").closest("summary");

    expect(within(preview).getByText("Preview available")).toBeVisible();
    expect(within(preview).getByText("webgpu-particle-field.ts")).toBeVisible();
    expect(within(preview).getByText("Ready when opened")).toBeVisible();
    expect(within(preview).getByText("preview.noop")).not.toBeVisible();
    expect(details).not.toHaveAttribute("open");
    expect(details).toHaveAttribute("data-state", "closed");
    expect(screen.queryByRole("tabpanel", { name: "Preview inspector" })).not.toBeInTheDocument();

    expect(summary).not.toBeNull();
    fireEvent.click(summary as HTMLElement);

    expect(details).toHaveAttribute("open");
    expect(details).toHaveAttribute("data-state", "open");
    expect(summary).toHaveAttribute("aria-expanded", "true");
  });

  it("opens artifacts, highlights the active artifact, and targets preview actions", () => {
    render(<WorkstationShell snapshot={getLocalWorkspaceSnapshot()} />);

    fireEvent.click(screen.getByRole("tab", { name: "Artifacts" }));
    fireEvent.click(screen.getByRole("button", { name: "Open webgpu-particle-field.ts" }));

    const codePanel = screen.getByRole("tabpanel", { name: "Code inspector" });

    expect(screen.getByRole("tab", { name: "Code" })).toHaveAttribute(
      "aria-selected",
      "true"
    );
    expect(codePanel).toHaveAttribute(
      "data-opened-artifact",
      "webgpu-particle-field.ts"
    );

    fireEvent.click(screen.getByRole("tab", { name: "Artifacts" }));
    fireEvent.click(screen.getByRole("button", { name: "Open projection-notes.md" }));

    expect(screen.getByLabelText("Active artifact")).toHaveTextContent(
      "projection-notes.md"
    );
    const selectedArtifact = screen.getByLabelText("projection-notes.md artifact");

    expect(selectedArtifact).toHaveAttribute("data-active", "true");
    expect(within(selectedArtifact).getByText("Selected")).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Preview preview-request.json" }));

    const preview = screen.getByRole("region", { name: "Preview workspace" });

    expect(screen.getByRole("tab", { name: "Overview" })).toHaveAttribute(
      "aria-selected",
      "true"
    );
    expect(within(preview).getByText("preview-request.json")).toBeVisible();
    expect(within(preview).getByText("Preview open")).toBeVisible();
    expect(within(preview).getByText("preview.noop")).toBeVisible();
    expect(preview.querySelector("details")).toHaveAttribute("open");
    expect(preview.querySelector("details")).toHaveAttribute("data-state", "open");
  });

  it("uses the full inspector panel for code when Code is active", () => {
    render(<WorkstationShell snapshot={snapshotWithActiveTab("Code")} />);

    expect(screen.getAllByRole("tabpanel")).toHaveLength(1);
    const codePanel = screen.getByRole("tabpanel", { name: "Code inspector" });

    expect(codePanel).toBeVisible();
    expect(screen.getByText("TypeScript + WGSL / Draft artifact")).toBeVisible();
    expect(
      within(codePanel).getByText((content) =>
        content.includes("renderer.present({ palette, projectionScale });")
      )
    ).toBeVisible();
    expect(screen.queryByRole("tabpanel", { name: "Overview inspector" })).not.toBeInTheDocument();
  });

  it("shows an elegant workflow inspector with live graph states", () => {
    render(<WorkstationShell snapshot={snapshotWithActiveTab("Workflow")} />);

    expect(screen.getAllByRole("tabpanel")).toHaveLength(1);
    expect(screen.getByRole("tabpanel", { name: "Workflow inspector" })).toBeVisible();
    const graph = screen.getByRole("group", {
      name: "LangGraph workflow visualization"
    });

    expect(graph).toBeVisible();
    expect(
      screen.getByRole("progressbar", { name: "Workflow inspector progress" })
    ).toHaveAttribute("aria-valuetext", "8 of 11 workflow nodes reached");
    expect(within(graph).getByText("Generation")).toBeVisible();
    expect(within(graph).getByText("Generation").closest("article")).toHaveAttribute(
      "aria-current",
      "step"
    );
    expect(within(graph).getByText("context_assembly")).toBeVisible();
    expect(within(graph).getByText("prompt_rendering")).toBeVisible();
    expect(within(graph).getByText("failure")).toBeVisible();
    expect(screen.queryByText("Preview request")).not.toBeInTheDocument();
    expect(
      screen.getByText("Real retry edge: refinement -> generation, bounded by review state.")
    ).toBeVisible();
    expect(screen.queryByRole("tab", { name: "Review" })).not.toBeInTheDocument();
  });
});
