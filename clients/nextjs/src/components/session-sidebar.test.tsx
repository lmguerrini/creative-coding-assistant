import { useState } from "react";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { WorkspaceSessionSummary } from "@/lib/workspace-persistence";
import { SessionSidebar } from "./session-sidebar";

const sessions: WorkspaceSessionSummary[] = [
  {
    artifactCount: 2,
    projectId: "project-1",
    sessionId: "session-1",
    title: "Creative workspace",
    updatedAt: "2026-07-13T16:15:00.000Z"
  },
  {
    artifactCount: 0,
    projectId: "project-1",
    sessionId: "session-2",
    title: "Line study",
    updatedAt: null
  }
];

function buildProps() {
  return {
    activeSessionId: "session-1",
    collapsed: false,
    onCreate: vi.fn(),
    onDelete: vi.fn(),
    onRename: vi.fn(),
    onSelect: vi.fn(),
    onToggle: vi.fn(),
    sessions
  };
}

describe("SessionSidebar", () => {
  it("presents browser-local sessions through the shared sidebar hierarchy", async () => {
    const props = buildProps();
    render(<SessionSidebar {...props} />);

    const sidebar = screen.getByRole("complementary", { name: "Sessions" });
    expect(within(sidebar).getByText("Workspace")).toBeVisible();
    expect(within(sidebar).getByText("Browser-local history")).toBeVisible();
    expect(
      within(within(sidebar).getByRole("list", { name: "Saved sessions" }))
        .getAllByRole("listitem")
    ).toHaveLength(2);

    const currentSession = within(sidebar).getByRole("button", {
      name: "Creative workspace"
    });
    expect(currentSession).toHaveAttribute("aria-current", "true");
    expect(currentSession).not.toHaveAttribute("aria-pressed");
    expect(currentSession).toHaveAccessibleDescription(/2 artifacts.*Saved/i);

    fireEvent.click(within(sidebar).getByRole("button", { name: "New session" }));
    expect(props.onCreate).toHaveBeenCalledOnce();

    fireEvent.click(within(sidebar).getByRole("button", { name: "Line study" }));
    expect(props.onSelect).toHaveBeenCalledWith("session-2");

    fireEvent.click(
      within(sidebar).getByRole("button", { name: "Delete Creative workspace" })
    );
    expect(props.onDelete).toHaveBeenCalledWith("session-1");

    fireEvent.click(
      within(sidebar).getByRole("button", { name: "Rename Creative workspace" })
    );
    const nameInput = within(sidebar).getByRole("textbox", { name: "Session name" });
    expect(nameInput).toHaveFocus();
    fireEvent.change(nameInput, { target: { value: "  Orbit archive  " } });
    fireEvent.click(within(sidebar).getByRole("button", { name: "Save" }));
    expect(props.onRename).toHaveBeenCalledWith("session-1", "Orbit archive");
    await waitFor(() => {
      expect(
        within(sidebar).getByRole("button", { name: "Rename Creative workspace" })
      ).toHaveFocus();
    });
  });

  it("cancels rename with Escape and returns focus to the rename action", async () => {
    render(<SessionSidebar {...buildProps()} />);

    fireEvent.click(screen.getByRole("button", { name: "Rename Creative workspace" }));
    fireEvent.keyDown(screen.getByRole("textbox", { name: "Session name" }), {
      key: "Escape"
    });

    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: "Rename Creative workspace" })
      ).toHaveFocus();
    });
  });

  it("preserves keyboard focus across its open and collapsed rail states", async () => {
    function Harness() {
      const [collapsed, setCollapsed] = useState(false);
      return (
        <SessionSidebar
          {...buildProps()}
          collapsed={collapsed}
          onToggle={() => setCollapsed((current) => !current)}
        />
      );
    }

    render(<Harness />);

    fireEvent.click(screen.getByRole("button", { name: "Collapse session sidebar" }));
    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: "Expand session sidebar" })
      ).toHaveFocus();
    });
    expect(screen.getByText("2 sessions")).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Expand session sidebar" }));
    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: "Collapse session sidebar" })
      ).toHaveFocus();
    });
  });
});
