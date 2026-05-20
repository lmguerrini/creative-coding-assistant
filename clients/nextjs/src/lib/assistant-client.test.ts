import { describe, expect, it } from "vitest";
import { createAssistantClient, getLocalWorkspaceSnapshot } from "./assistant-client";

describe("assistant frontend client", () => {
  it("provides a typed local workspace snapshot", () => {
    const snapshot = getLocalWorkspaceSnapshot();

    expect(snapshot.modes).toContainEqual({ label: "Generate", active: true });
    expect(snapshot.workflow.steps.some((step) => step.state === "active")).toBe(
      true
    );
    expect(snapshot.artifacts.map((artifact) => artifact.type)).toEqual([
      "code",
      "preview",
      "export"
    ]);
    expect(snapshot.preview.target).toContain("Browser sandbox");
  });

  it("exposes an async boundary for future backend integration", async () => {
    const client = createAssistantClient();

    await expect(client.getWorkspaceSnapshot()).resolves.toMatchObject({
      debug: {
        status: "Live"
      }
    });
  });
});
