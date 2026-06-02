import { describe, expect, it } from "vitest";
import {
  createAssistantClient,
  getInitialWorkspaceSnapshot,
  getLocalWorkspaceSnapshot
} from "./assistant-client";

describe("assistant frontend client", () => {
  it("provides a typed local workspace snapshot", () => {
    const snapshot = getLocalWorkspaceSnapshot();

    expect(snapshot.session).toMatchObject({
      userId: "local-user",
      sessionId: "local-nextjs-session",
      projectId: "local-nextjs-workspace"
    });
    expect(snapshot.inspectorTabs.map((tab) => tab.label)).toEqual([
      "Overview",
      "Preview",
      "Code",
      "Workflow",
      "Telemetry",
      "Artifacts",
      "Retrieval"
    ]);
    expect(snapshot.inspectorTabs).toContainEqual(
      expect.objectContaining({ label: "Overview", active: true })
    );
    expect(snapshot.inspectorTabs).toContainEqual(
      expect.objectContaining({ label: "Preview", active: false })
    );
    expect(snapshot.preview.available).toBe(true);
    expect(snapshot.preview.active).toBe(false);
    expect(snapshot.preview.collapsed).toBe(true);
    expect(snapshot.workflow.steps.some((step) => step.state === "active")).toBe(
      true
    );
    expect(snapshot.workflow.steps.map((step) => step.nodeId)).toEqual([
      "intake",
      "routing",
      "memory",
      "retrieval",
      "context_assembly",
      "prompt_input",
      "prompt_rendering",
      "generation",
      "artifact_extraction",
      "preview_preparation",
      "review",
      "refinement",
      "finalization",
      "failure"
    ]);
    expect(snapshot.workflow.steps).toContainEqual(
      expect.objectContaining({
        nodeId: "context_assembly",
        displayLabel: "Context assembly"
      })
    );
    expect(snapshot.workflow.steps).toContainEqual(
      expect.objectContaining({ nodeId: "refinement", displayLabel: "Refinement" })
    );
    expect(snapshot.workflow.steps.map((step) => step.nodeId)).not.toContain(
      "preview_request"
    );
    expect(snapshot.artifacts.map((artifact) => artifact.type)).toEqual([
      "code",
      "preview",
      "export"
    ]);
    expect(snapshot.artifacts[0].actions).toContain("Preview");
    expect(snapshot.multimodal).toMatchObject({
      state: "empty",
      status: "No image references",
      imageAttachments: []
    });
    expect(snapshot.preview.targetId).toBe("browser_sandbox");
    expect(snapshot.preview.target).toContain("Browser preview");
    expect(snapshot.retrieval.state).toBe("available");
    expect(snapshot.retrieval.sources[0]).toMatchObject({
      sourceId: "webgpu_mdn_api",
      domainLabel: "WebGPU / WGSL",
      quality: "high",
      freshness: "fresh"
    });
    expect(snapshot.retrieval.sources[0].chunks[0]).toMatchObject({
      chunkIndex: 0,
      relevanceLabel: "Best match"
    });
  });

  it("exposes an async boundary for future backend integration", async () => {
    const client = createAssistantClient();

    await expect(client.getWorkspaceSnapshot()).resolves.toMatchObject({
      artifacts: [],
      debug: {
        status: "Ready"
      },
      preview: {
        available: false
      }
    });
  });

  it("provides a polished first-run workspace snapshot", () => {
    const snapshot = getInitialWorkspaceSnapshot();

    expect(snapshot.messages).toEqual([]);
    expect(snapshot.artifacts).toEqual([]);
    expect(snapshot.preview).toMatchObject({
      available: false,
      active: false,
      collapsed: true,
      state: "unavailable"
    });
    expect(snapshot.workflow.status).toBe("Idle");
    expect(snapshot.workflow.steps.some((step) => step.state === "active")).toBe(
      false
    );
    expect(snapshot.retrieval.state).toBe("empty");
  });
});
