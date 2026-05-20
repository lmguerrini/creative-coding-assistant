import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { WorkstationShell } from "./workstation-shell";
import { getLocalWorkspaceSnapshot } from "@/lib/assistant-client";

describe("WorkstationShell", () => {
  it("renders the multi-panel creative workflow shell", () => {
    render(<WorkstationShell snapshot={getLocalWorkspaceSnapshot()} />);

    expect(screen.getByText("Creative Coding Assistant")).toBeVisible();
    expect(screen.getByRole("navigation", { name: "Assistant mode" })).toBeVisible();
    expect(screen.getByRole("region", { name: "Chat Workflow" })).toBeVisible();
    expect(screen.getByRole("region", { name: "Artifacts" })).toBeVisible();
    expect(screen.getByRole("region", { name: "Preview" })).toBeVisible();
    expect(screen.getByRole("region", { name: "Workflow Debug" })).toBeVisible();
  });

  it("keeps artifact and preview state visible", () => {
    render(<WorkstationShell snapshot={getLocalWorkspaceSnapshot()} />);

    expect(screen.getAllByText("webgpu-particle-field.ts").length).toBeGreaterThan(1);
    expect(screen.getByText("preview-request.json")).toBeVisible();
    expect(screen.getByText("preview.noop")).toBeVisible();
    expect(screen.getByText(/browser_sandbox/)).toBeVisible();
  });
});
