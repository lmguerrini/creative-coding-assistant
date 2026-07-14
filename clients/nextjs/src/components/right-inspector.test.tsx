import { useState } from "react";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { Activity, Eye, FileCode2, LayoutDashboard } from "lucide-react";
import { describe, expect, it, vi } from "vitest";
import {
  RightInspector,
  type RightInspectorTab
} from "./right-inspector";

type TabId = "overview" | "runtime" | "preview" | "artifacts";

const tabs = [
  {
    closeable: false,
    icon: LayoutDashboard,
    id: "overview",
    label: "Overview",
    panelId: "overview-panel"
  },
  {
    closeable: true,
    icon: Activity,
    id: "runtime",
    label: "Runtime",
    panelId: "runtime-panel"
  },
  {
    closeable: true,
    icon: Eye,
    id: "preview",
    label: "Preview",
    panelId: "preview-panel"
  }
] as const satisfies readonly RightInspectorTab<TabId>[];

const availableTabs = [
  {
    closeable: true,
    icon: FileCode2,
    id: "artifacts",
    label: "Artifacts",
    panelId: "artifacts-panel"
  }
] as const satisfies readonly RightInspectorTab<TabId>[];

function createProps() {
  return {
    activeTab: "overview" as TabId,
    addMenuOpen: false,
    availableTabs,
    collapsed: false,
    detail: "Current workspace evidence",
    onAddMenuOpenChange: vi.fn(),
    onAddTab: vi.fn(),
    onCloseTab: vi.fn(),
    onOpenDashboard: vi.fn(),
    onSelectTab: vi.fn(),
    onToggle: vi.fn(),
    tabs
  };
}

function InspectorPanel({ activeTab }: { activeTab: TabId }) {
  const active = [...tabs, ...availableTabs].find((tab) => tab.id === activeTab);

  return (
    <section
      aria-labelledby={`${active?.panelId}-tab`}
      id={active?.panelId}
      role="tabpanel"
    >
      {active?.label} evidence
    </section>
  );
}

describe("RightInspector", () => {
  it("exposes one selected tab and its controlled panel relationship", () => {
    render(
      <RightInspector {...createProps()} activeTab="runtime">
        <InspectorPanel activeTab="runtime" />
      </RightInspector>
    );

    const tablist = screen.getByRole("tablist", { name: "Inspector tabs" });
    const overview = within(tablist).getByRole("tab", { name: "Overview" });
    const runtime = within(tablist).getByRole("tab", { name: "Runtime" });
    const panel = screen.getByRole("tabpanel", { name: "Runtime" });
    const inspector = screen.getByRole("complementary", { name: "Right inspector" });

    expect(runtime).toHaveAttribute("aria-selected", "true");
    expect(runtime).toHaveAttribute("aria-controls", "runtime-panel");
    expect(runtime).toHaveAttribute("tabindex", "0");
    expect(panel).toHaveAttribute("aria-labelledby", "runtime-panel-tab");
    expect(overview).toHaveAttribute("aria-selected", "false");
    expect(overview).toHaveAttribute("tabindex", "-1");
    expect(
      inspector.querySelector(".dashboardSidebarHeaderIcon .lucide-search")
    ).not.toBeNull();
  });

  it("moves and selects tabs with ArrowLeft, ArrowRight, Home, and End", () => {
    function Harness() {
      const [activeTab, setActiveTab] = useState<TabId>("overview");

      return (
        <RightInspector
          {...createProps()}
          activeTab={activeTab}
          onSelectTab={setActiveTab}
        >
          <InspectorPanel activeTab={activeTab} />
        </RightInspector>
      );
    }

    render(<Harness />);

    const overview = screen.getByRole("tab", { name: "Overview" });
    const runtime = screen.getByRole("tab", { name: "Runtime" });
    const preview = screen.getByRole("tab", { name: "Preview" });

    overview.focus();
    fireEvent.keyDown(overview, { key: "ArrowRight" });
    expect(runtime).toHaveFocus();
    expect(runtime).toHaveAttribute("aria-selected", "true");

    fireEvent.keyDown(runtime, { key: "ArrowRight" });
    expect(preview).toHaveFocus();

    fireEvent.keyDown(preview, { key: "ArrowRight" });
    expect(overview).toHaveFocus();

    fireEvent.keyDown(overview, { key: "ArrowLeft" });
    expect(preview).toHaveFocus();

    fireEvent.keyDown(preview, { key: "Home" });
    expect(overview).toHaveFocus();

    fireEvent.keyDown(overview, { key: "End" });
    expect(preview).toHaveFocus();
    expect(preview).toHaveAttribute("aria-selected", "true");
  });

  it("restores focus to the toggle across controlled collapse and expand states", async () => {
    function Harness() {
      const [collapsed, setCollapsed] = useState(false);

      return (
        <RightInspector
          {...createProps()}
          collapsed={collapsed}
          onToggle={() => setCollapsed((current) => !current)}
        >
          <InspectorPanel activeTab="overview" />
        </RightInspector>
      );
    }

    render(<Harness />);

    fireEvent.click(screen.getByRole("button", { name: "Collapse inspector" }));
    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: "Expand inspector" })
      ).toHaveFocus();
    });
    expect(screen.getByRole("complementary", { name: "Right inspector" }))
      .toHaveAttribute("data-state", "collapsed");

    fireEvent.click(screen.getByRole("button", { name: "Expand inspector" }));
    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: "Collapse inspector" })
      ).toHaveFocus();
    });
  });

  it("focuses the first Add menu item, returns focus on Escape, and delegates selection", async () => {
    const onAddTab = vi.fn();

    function Harness() {
      const [addMenuOpen, setAddMenuOpen] = useState(false);

      return (
        <RightInspector
          {...createProps()}
          addMenuOpen={addMenuOpen}
          onAddMenuOpenChange={setAddMenuOpen}
          onAddTab={onAddTab}
        >
          <InspectorPanel activeTab="overview" />
        </RightInspector>
      );
    }

    render(<Harness />);

    const addButton = screen.getByRole("button", { name: "Add Inspector tab" });
    fireEvent.click(addButton);

    const menu = screen.getByRole("menu", { name: "Available Inspector tabs" });
    const artifacts = within(menu).getByRole("menuitem", { name: "Artifacts" });
    await waitFor(() => expect(artifacts).toHaveFocus());

    fireEvent.keyDown(artifacts, { key: "Escape" });
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
    expect(addButton).toHaveFocus();

    fireEvent.click(addButton);
    fireEvent.click(screen.getByRole("menuitem", { name: "Artifacts" }));
    expect(onAddTab).toHaveBeenCalledOnce();
    expect(onAddTab).toHaveBeenCalledWith("artifacts");
  });

  it("delegates closing a closeable tab without selecting it", () => {
    const props = createProps();
    render(
      <RightInspector {...props}>
        <InspectorPanel activeTab="overview" />
      </RightInspector>
    );

    fireEvent.click(
      screen.getByRole("button", { name: "Close Runtime Inspector tab" })
    );

    expect(props.onCloseTab).toHaveBeenCalledOnce();
    expect(props.onCloseTab).toHaveBeenCalledWith("runtime");
    expect(props.onSelectTab).not.toHaveBeenCalled();
  });
});
