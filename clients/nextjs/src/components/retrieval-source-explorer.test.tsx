import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { getLocalWorkspaceSnapshot } from "@/lib/assistant-client";
import { buildRetrievalRuntimeModel } from "@/lib/retrieval-runtime";
import { buildRetrievalSourceExplorerModel } from "@/lib/retrieval-source-explorer";
import { RetrievalSourceExplorer } from "./retrieval-source-explorer";

describe("RetrievalSourceExplorer source health", () => {
  it("renders selected-source health, sync metrics, ownership, and coverage", () => {
    renderExplorer();

    const webgpuHealth = screen.getByRole("region", {
      name: "WebGPU API source health"
    });

    expect(webgpuHealth).toHaveAttribute("data-health", "healthy");
    expect(webgpuHealth).toHaveTextContent("Healthy");
    expect(webgpuHealth).toHaveTextContent("Available");
    expect(webgpuHealth).toHaveTextContent("Fresh");
    expect(webgpuHealth).toHaveTextContent("Web platform / MDN");
    expect(webgpuHealth).toHaveTextContent("184 indexed chunks");
    expect(webgpuHealth).toHaveTextContent("May 20, 2026 · 20d ago");
    expect(webgpuHealth).toHaveTextContent("Succeeded");
    expect(webgpuHealth).toHaveTextContent("2/3 context chunks");
    expect(
      within(webgpuHealth).getByText("Available", {
        selector: ".kbAvailabilityBadge"
      })
    ).toHaveAttribute("data-availability", "available");

    fireEvent.click(
      screen.getByRole("button", {
        name: "Inspect source OpenGL Shading Language 4.60 Specification"
      })
    );

    const glslHealth = screen.getByRole("region", {
      name: "OpenGL Shading Language 4.60 Specification source health"
    });

    expect(glslHealth).toHaveAttribute("data-health", "stale");
    expect(glslHealth).toHaveTextContent("Stale");
    expect(glslHealth).toHaveTextContent("Graphics standards / Khronos Group");
    expect(
      within(glslHealth).getByRole("list", {
        name: "OpenGL Shading Language 4.60 Specification health warnings"
      })
    ).toHaveTextContent("preferred shader guidance refresh window");
  });

  it("renders safe legacy health defaults without fabricated sync data", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const retrieval = {
      ...snapshot.retrieval,
      sources: snapshot.retrieval.sources.map(({ health: _health, ...source }) => source)
    };
    const model = buildRetrievalSourceExplorerModel(
      buildRetrievalRuntimeModel(retrieval, [])
    );

    render(<RetrievalSourceExplorer model={model} />);

    const webgpuHealth = screen.getByRole("region", {
      name: "WebGPU API source health"
    });

    expect(webgpuHealth).toHaveAttribute("data-health", "warning");
    expect(webgpuHealth).toHaveTextContent("legacy session");
    expect(webgpuHealth).toHaveTextContent("MDN");
    expect(webgpuHealth).toHaveTextContent("Not reported");
    expect(webgpuHealth).toHaveTextContent("Unknown");
  });
});

function renderExplorer() {
  const snapshot = getLocalWorkspaceSnapshot();
  const model = buildRetrievalSourceExplorerModel(
    buildRetrievalRuntimeModel(snapshot.retrieval, [])
  );

  return render(<RetrievalSourceExplorer model={model} />);
}
