import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { WorkstationShell } from "./workstation-shell";
import {
  getLocalWorkspaceSnapshot,
  type AssistantWorkspaceSnapshot,
  type InspectorTabName
} from "@/lib/assistant-client";

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

describe("WorkstationShell", () => {
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
    expect(screen.queryByRole("tabpanel", { name: "Code inspector" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Preview" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Review" })).not.toBeInTheDocument();
  });

  it("keeps preview available, on demand, and collapsible in the main column", () => {
    render(<WorkstationShell snapshot={getLocalWorkspaceSnapshot()} />);

    const preview = screen.getByRole("region", { name: "Preview workspace" });

    expect(within(preview).getByText("Preview available")).toBeVisible();
    expect(within(preview).getByText("webgpu-particle-field.ts")).toBeVisible();
    expect(within(preview).getByText("Ready when opened")).toBeVisible();
    expect(within(preview).getByText("preview.noop")).not.toBeVisible();
    expect(preview.querySelector("details")).not.toHaveAttribute("open");
    expect(screen.queryByRole("tabpanel", { name: "Preview inspector" })).not.toBeInTheDocument();
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
    expect(within(graph).getByText("Generation")).toBeVisible();
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
