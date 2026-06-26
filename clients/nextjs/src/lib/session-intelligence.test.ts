import { describe, expect, it } from "vitest";
import {
  getInitialWorkspaceSnapshot,
  getLocalWorkspaceSnapshot
} from "./assistant-client";
import {
  buildSessionIntelligenceModel,
  readSessionIntelligenceMetadata
} from "./session-intelligence";
import { buildWorkstationState } from "./workstation-state";
import { createWorkstationError } from "./workstation-errors";

describe("session intelligence", () => {
  it("derives first-run session metadata from workstation state", () => {
    const snapshot = getInitialWorkspaceSnapshot();
    const workstationState = buildWorkstationState({ snapshot });
    const intelligence = buildSessionIntelligenceModel({
      snapshot,
      workstationState
    });

    expect(intelligence.source).toBe("derived");
    expect(intelligence.metadata).toMatchObject({
      active_request_summary: "No active request has been submitted yet.",
      completion_status: "idle",
      session_summary:
        "Creative workspace is ready for the first creative request."
    });
    expect(intelligence.metadata.available_metadata_groups).toEqual([
      "Session",
      "Selected workflow"
    ]);
    expect(intelligence.metadata.session_warnings).toEqual([]);
    expect(intelligence.metadata.recommended_next_user_actions).toContain(
      "Send a creative prompt to start the session."
    );
  });

  it("hydrates snake_case session intelligence fields from stream payloads", () => {
    const metadata = readSessionIntelligenceMetadata({
      session_intelligence: {
        active_request_summary: "Generating a p5 sketch.",
        available_metadata_groups: ["Session", "Preview", "Workflow"],
        completion_status: "running",
        recommended_next_user_actions: [
          "Wait for the active response to finish before sending another request."
        ],
        session_summary: "Live run metadata is available.",
        session_warnings: ["Preview metadata is pending."]
      }
    });

    expect(metadata).toEqual({
      active_request_summary: "Generating a p5 sketch.",
      available_metadata_groups: ["Session", "Preview", "Workflow"],
      completion_status: "running",
      recommended_next_user_actions: [
        "Wait for the active response to finish before sending another request."
      ],
      session_summary: "Live run metadata is available.",
      session_warnings: ["Preview metadata is pending."]
    });
  });

  it("merges streamed partial metadata with derived fallback fields", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const workstationState = buildWorkstationState({ snapshot });
    const intelligence = buildSessionIntelligenceModel({
      snapshot,
      streamedMetadata: {
        completion_status: "completed",
        session_summary: "Stream provided a compact session summary."
      },
      workstationState
    });

    expect(intelligence.source).toBe("stream");
    expect(intelligence.statusLabel).toBe("Completed");
    expect(intelligence.metadata.session_summary).toBe(
      "Stream provided a compact session summary."
    );
    expect(intelligence.metadata.active_request_summary).toContain(
      "Build a luminous particle field"
    );
    expect(intelligence.metadata.available_metadata_groups).toContain("Preview");
  });

  it("marks degraded sessions with warnings and text-only recommendations", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const workstationState = buildWorkstationState({
      snapshot,
      streamError: createWorkstationError({
        category: "stream",
        recoverable: true,
        subsystem: "assistant_stream",
        suggestedAction: "Retry the request from the composer.",
        type: "assistant_stream_failed",
        userMessage: "The live response stopped before completion."
      })
    });
    const intelligence = buildSessionIntelligenceModel({
      snapshot,
      workstationState
    });

    expect(intelligence.metadata.completion_status).toBe("needs_attention");
    expect(intelligence.metadata.session_warnings).toContain(
      "The live response stopped before completion."
    );
    expect(intelligence.metadata.recommended_next_user_actions).toContain(
      "Review session warnings before continuing."
    );
  });
});
