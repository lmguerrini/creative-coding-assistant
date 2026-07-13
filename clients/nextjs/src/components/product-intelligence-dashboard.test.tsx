import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import {
  productIntelligenceCategories,
  type ProductIntelligenceModel
} from "@/lib/product-intelligence";
import { loadingDomainExperienceCatalog } from "@/lib/domain-experience";
import {
  defaultWorkspaceLayoutState,
  defaultWorkspacePreferences
} from "@/lib/workspace-persistence";
import {
  ProductIntelligenceDashboard,
  ProductIntelligenceInspector
} from "./product-intelligence-dashboard";

function buildModel(): ProductIntelligenceModel {
  return {
    activeDomainId: null,
    domainExperience: {
      ...loadingDomainExperienceCatalog,
      creativeKnowledge: {
        status: "available",
        detail: "Typed creative guidance.",
        authorityBoundary: "No private reasoning is exposed.",
        recordCount: 1,
        records: [
          {
            id: "creative_knowledge::runtime_selection_hydra_vs_p5",
            kind: "workflow",
            title: "Live visual runtime triage",
            summary: "Choose a controlled sketch path.",
            domains: ["hydra", "p5_js"],
            techniqueTags: ["runtime_triage"],
            workflowSteps: ["Compare runtime boundaries"],
            patternTags: ["runtime_selection"],
            taxonomyPath: ["creative production", "runtime choice"],
            sourceIds: ["p5_reference"],
            provenanceCount: 1,
            confidence: { score: 0.8, band: "high", caveats: [] }
          }
        ]
      }
    },
    artifactRegistry: [],
    session: { id: "session-1", title: "Creative workspace" },
    sections: productIntelligenceCategories.map((category) => ({
      category,
      tone: category === "Workflow" ? "active" : "ready",
      summary: `${category} summary`,
      detail: `${category} detailed product information`,
      metrics: [
        { label: "Primary", value: `${category} value` },
        { label: "Secondary", value: "Available" }
      ],
      notes: [`${category} is derived from the shared product model.`]
    })),
    summary: { activeCount: 1, attentionCount: 0, readyCount: 18 }
  };
}

describe("Product Intelligence surfaces", () => {
  it("separates Dashboard categories into focused decision pages", () => {
    const onCategoryChange = vi.fn();
    const onClose = vi.fn();

    const { rerender } = render(
      <ProductIntelligenceDashboard
        activeCategory="Overview"
        model={buildModel()}
        onCategoryChange={onCategoryChange}
        onClose={onClose}
      />
    );

    const navigation = screen.getByRole("navigation", { name: "Dashboard categories" });
    expect(navigation)
      .toHaveTextContent("User Guide");
    expect(navigation)
      .toHaveTextContent("Knowledge Base");
    expect(navigation).not.toHaveTextContent("Current workspace outcome and selected artifact.");
    expect(screen.getByRole("heading", { name: "Overview" })).toBeVisible();
    expect(screen.getByLabelText("Overview live signal board")).toBeVisible();
    expect(screen.getByLabelText("Help with Overview live signal board")).toBeVisible();

    fireEvent.click(screen.getAllByLabelText("Help with Overview")[0]!);
    expect(screen.getAllByRole("note")[0]).toHaveTextContent(
      "Review the metric cards for the current values"
    );

    const overviewHelp = screen.getAllByLabelText("Help with Overview")[0]!
      .closest("details");
    expect(overviewHelp).not.toBeNull();
    fireEvent.mouseLeave(overviewHelp!);
    expect(overviewHelp).not.toHaveAttribute("open");
    fireEvent.mouseEnter(overviewHelp!);
    expect(overviewHelp).toHaveAttribute("open");
    fireEvent.mouseLeave(overviewHelp!);
    expect(overviewHelp).not.toHaveAttribute("open");

    fireEvent.click(
      screen.getByRole("button", { name: /Architecture/ })
    );
    expect(onCategoryChange).toHaveBeenCalledWith("Architecture");

    fireEvent.click(screen.getByRole("button", { name: "Knowledge Base" }));
    expect(onCategoryChange).toHaveBeenCalledWith("Knowledge Base");
    rerender(
      <ProductIntelligenceDashboard
        activeCategory="Knowledge Base"
        model={buildModel()}
        onCategoryChange={onCategoryChange}
        onClose={onClose}
      />
    );
    expect(screen.getByText("Technical knowledge")).toBeVisible();
    expect(screen.queryByText("Knowledge Base inventory")).not.toBeInTheDocument();
    expect(screen.getAllByText("Creative Knowledge Base")[0]).toBeVisible();
    expect(screen.getByText("Live visual runtime triage")).toBeVisible();
    expect(screen.getByLabelText("Curated creative studies")).toBeVisible();
    expect(screen.getByText("Published request, source, chunk, quality, and freshness signals for this run.")).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "User Guide" }));
    expect(screen.getByRole("button", { name: "User Guide" })).toHaveAttribute(
      "aria-current",
      "page"
    );
    expect(screen.getByRole("heading", { name: "User Guide" })).toBeVisible();
    expect(screen.getByRole("region", { name: "User Guide" })).toBeVisible();
    expect(screen.getByRole("heading", { name: "Your first run" })).toBeVisible();
    expect(screen.getByRole("heading", { name: "Every Dashboard page" })).toBeVisible();
    expect(screen.getByRole("heading", { name: "Workflows and Demo Mode" })).toBeVisible();
    expect(screen.getAllByText("Knowledge Base").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Retrieval").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Sessions").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Artifacts").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Preview").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Runtime").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Settings").length).toBeGreaterThan(0);
    expect(screen.getByRole("heading", { name: "Troubleshooting" })).toBeVisible();
    expect(
      within(screen.getByRole("list", { name: "Dashboard page reference" }))
        .getAllByRole("listitem")
    ).toHaveLength(16);

    const previewDisclosure = screen.getByText(
      "Supported live Preview runtimes and their boundaries"
    ).closest("details");
    expect(previewDisclosure).not.toHaveAttribute("open");
    fireEvent.click(screen.getByText("Supported live Preview runtimes and their boundaries"));
    expect(previewDisclosure).toHaveAttribute("open");
    expect(screen.getByRole("row", { name: /Tone.js/ })).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Close dashboard" }));
    expect(onClose).toHaveBeenCalledOnce();
  });

  it("keeps output feedback in the AI & agents Dashboard group", () => {
    const onSubmitFeedback = vi.fn();
    const onCategoryChange = vi.fn();

    const { rerender } = render(
      <ProductIntelligenceDashboard
        activeCategory="Overview"
        feedback={{ artifactTitle: "aurora-field.p5.js", onSubmit: onSubmitFeedback }}
        model={buildModel()}
        onCategoryChange={onCategoryChange}
        onClose={vi.fn()}
      />
    );

    expect(screen.queryByLabelText("Output feedback")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /AI & agents/ }));
    expect(onCategoryChange).toHaveBeenCalledWith("Agents");
    rerender(
      <ProductIntelligenceDashboard
        activeCategory="Agents"
        feedback={{ artifactTitle: "aurora-field.p5.js", onSubmit: onSubmitFeedback }}
        model={buildModel()}
        onCategoryChange={onCategoryChange}
        onClose={vi.fn()}
      />
    );

    expect(screen.getByLabelText("Output feedback")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Mark output helpful" }));
    expect(onSubmitFeedback).toHaveBeenCalledWith("positive", "");
  });

  it("gives every retained artifact a visual surface or an explicit export boundary", () => {
    const model = buildModel();
    model.activeArtifactId = "export-only";
    model.artifactRegistry = [
      {
        actions: ["Open", "Preview", "Copy"],
        content: "function draw() { background(8); }",
        id: "preview-ready",
        language: "JavaScript + p5.js",
        previewEligible: true,
        previewTarget: "browser_sandbox",
        status: "Generated",
        summary: "A browser-native visual study.",
        title: "orbit-study.p5.js",
        type: "code"
      },
      {
        actions: ["Open", "Export", "Copy"],
        content: "@fragment fn main() -> @location(0) vec4f { return vec4f(1.0); }",
        id: "export-only",
        language: "WGSL",
        previewEligible: false,
        status: "Generated",
        summary: "A WebGPU shader export.",
        title: "light-field.wgsl",
        type: "export"
      }
    ];

    render(
      <ProductIntelligenceDashboard
        activeCategory="Artifacts"
        model={model}
        onCategoryChange={vi.fn()}
        onClose={vi.fn()}
      />
    );

    const registry = screen.getByRole("region", { name: "Artifact registry" });
    expect(within(registry).getAllByRole("figure")).toHaveLength(2);
    expect(within(registry).getByLabelText("orbit-study.p5.js export boundary"))
      .toHaveTextContent("Preview evidence unavailable");
    expect(within(registry).getByLabelText("light-field.wgsl export boundary"))
      .toHaveTextContent(/Code \/ export boundary.*does not simulate an unsupported runtime/i);
    expect(within(registry).getByText("export · selected").closest("article"))
      .toHaveAttribute("aria-current", "true");
  });

  it("uses the published active artifact on the Workspace page", () => {
    const model = buildModel();
    model.activeArtifactId = "selected-artifact";
    model.artifactRegistry = [
      {
        actions: ["Open"],
        id: "older-artifact",
        language: "JavaScript",
        status: "Generated",
        summary: "Earlier source.",
        title: "older.js",
        type: "code"
      },
      {
        actions: ["Open", "Preview"],
        content: "function draw() {}",
        id: "selected-artifact",
        language: "JavaScript + p5.js",
        previewEligible: true,
        status: "Generated",
        summary: "Current selected source.",
        title: "selected.p5.js",
        type: "code"
      }
    ];

    render(
      <ProductIntelligenceDashboard
        activeCategory="Code"
        model={model}
        onCategoryChange={vi.fn()}
        onClose={vi.fn()}
      />
    );

    expect(screen.getByRole("region", { name: "Active document" }))
      .toHaveTextContent("selected.p5.js");
    expect(screen.getByLabelText("Active document source excerpt"))
      .toHaveTextContent("function draw() {}");
  });

  it("covers every saved typography category in Dashboard settings", () => {
    const onPreferencesChange = vi.fn();
    const onWorkflowModeChange = vi.fn();

    render(
      <ProductIntelligenceDashboard
        activeCategory="Settings"
        model={buildModel()}
        onCategoryChange={vi.fn()}
        onClose={vi.fn()}
        settings={{
          isFocusMode: false,
          isPreviewOpen: false,
          layoutState: defaultWorkspaceLayoutState,
          onDensityChange: vi.fn(),
          onFocusModeToggle: vi.fn(),
          onInspectorToggle: vi.fn(),
          onPreferencesChange,
          onPreviewToggle: vi.fn(),
          onSidebarToggle: vi.fn(),
          onWorkflowModeChange,
          preferences: defaultWorkspacePreferences,
          workflowMode: "auto"
        }}
      />
    );

    expect(screen.getByText("Headings")).toBeVisible();
    expect(screen.getAllByText("Body text")).toHaveLength(2);
    expect(screen.getByText("Labels and controls")).toBeVisible();
    expect(screen.getByText("Code text")).toBeVisible();
    expect(screen.getByText("Heading")).toBeVisible();
    expect(screen.getByText("Label")).toBeVisible();
    expect(screen.getByText("Code")).toBeVisible();
    expect(screen.getByRole("group", { name: "Colour themes" })).toBeVisible();
    expect(screen.getByRole("group", { name: "Black & white" })).toHaveTextContent(
      "DarkLight"
    );
    expect(screen.getByRole("group", { name: "Generation defaults" })).toHaveTextContent(
      /Workflow.*AI Providers.*OpenAI.*Creativity/
    );
    expect(screen.getByRole("group", { name: "AI Providers" })).toHaveTextContent(
      "OpenAI"
    );
    fireEvent.click(screen.getByLabelText("Selected AI provider: OpenAI"));
    expect(screen.getByText("Selected provider")).toBeVisible();

    const largeChoices = screen.getAllByRole("button", { name: "large" });
    fireEvent.click(largeChoices[0]!);
    fireEvent.click(largeChoices[2]!);

    expect(onPreferencesChange).toHaveBeenNthCalledWith(1, {
      headingFontSize: "large"
    });
    expect(onPreferencesChange).toHaveBeenNthCalledWith(2, {
      labelFontSize: "large"
    });

    fireEvent.change(screen.getByRole("combobox", { name: "Default workflow" }), {
      target: { value: "multi_agent" }
    });
    fireEvent.change(screen.getByRole("combobox", { name: "Default creativity" }), {
      target: { value: "exploratory" }
    });

    expect(onWorkflowModeChange).toHaveBeenCalledWith("multi_agent");
    expect(onPreferencesChange).toHaveBeenLastCalledWith({
      creativity: "exploratory"
    });
  });

  it("uses the same model for a compact Inspector category", () => {
    render(<ProductIntelligenceInspector category="Providers" model={buildModel()} />);

    expect(screen.getByRole("tabpanel", { name: "Providers inspector" }))
      .toHaveTextContent("Providers detailed product information");
    expect(screen.getByText("Providers value")).toBeVisible();
    expect(screen.getByLabelText("Help with Providers")).toBeVisible();
  });
});
