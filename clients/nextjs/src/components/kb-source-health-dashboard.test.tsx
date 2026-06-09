import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type {
  KbSourceHealthDashboardModel,
  KbSourceHealthStatus
} from "@/lib/kb-source-health";
import { KbSourceHealthDashboard } from "./kb-source-health-dashboard";

describe("KbSourceHealthDashboard", () => {
  it.each([
    ["healthy", "Healthy"],
    ["warning", "Warning"],
    ["stale", "Stale"],
    ["failed", "Failed"],
    ["unknown", "Unknown"]
  ] as const)("renders the %s health indicator", (status, label) => {
    render(
      <KbSourceHealthDashboard model={createHealthModel(status, label)} />
    );

    const badge = screen.getByText(label, { selector: ".kbHealthStatusBadge" });

    expect(badge).toHaveAttribute("data-health", status);
  });

  it("keeps aggregate metrics compact until expanded", () => {
    render(
      <KbSourceHealthDashboard model={createHealthModel("warning", "Warning")} />
    );

    const toggle = screen.getByLabelText(
      "Toggle knowledge base source health dashboard"
    );
    const dashboard = toggle.closest("details");

    expect(dashboard).not.toHaveAttribute("open");

    fireEvent.click(toggle);

    expect(dashboard).toHaveAttribute("open");
    expect(
      screen.getByRole("list", {
        name: "Knowledge base source health metrics"
      })
    ).toHaveTextContent("2/2 sources available");
    expect(screen.getByText("280 indexed chunks")).toBeVisible();
    expect(screen.getByText("20d ago", { exact: false })).toBeVisible();
    expect(screen.getByText("2 domain owners")).toBeVisible();
  });
});

function createHealthModel(
  status: KbSourceHealthStatus,
  statusLabel: string
): KbSourceHealthDashboardModel {
  return {
    status,
    statusLabel,
    statusDetail: "Source health summary.",
    sourceCount: 2,
    healthySourceCount: status === "healthy" ? 2 : 1,
    attentionSourceCount: status === "healthy" ? 0 : 1,
    availableSourceCount: 2,
    availabilityLabel: "2/2 sources available",
    indexedChunkCount: 280,
    indexedChunkLabel: "280 indexed chunks",
    latestSyncAttemptAt: "2026-05-20T08:30:00Z",
    latestSyncAttemptLabel: "May 20, 2026 · 20d ago",
    domainOwnerCount: 2,
    domainOwnerLabel: "2 domain owners",
    sources: []
  };
}
