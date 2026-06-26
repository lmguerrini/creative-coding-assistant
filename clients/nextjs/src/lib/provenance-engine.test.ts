import { describe, expect, it } from "vitest";
import {
  getInitialWorkspaceSnapshot,
  getLocalWorkspaceSnapshot,
  type WorkflowNodeId
} from "./assistant-client";
import type { AssistantStreamEvent, AssistantStreamEventType } from "./assistant-stream";
import { buildProvenanceEngineModel } from "./provenance-engine";
import type { WorkflowRuntimeTraceEvent } from "./workflow-runtime";
import { buildWorkstationState } from "./workstation-state";

describe("provenance engine", () => {
  it("reports missing provenance without inventing sources", () => {
    const snapshot = getInitialWorkspaceSnapshot();
    const workstationState = buildWorkstationState({ snapshot });
    const provenance = buildProvenanceEngineModel({
      snapshot,
      traceEvents: [],
      workstationState
    });

    expect(provenance.evidence_sources).toEqual([]);
    expect(provenance.dependency_sources).toEqual([]);
    expect(provenance.artifact_sources).toEqual([]);
    expect(provenance.evaluation_sources).toEqual([]);
    expect(provenance.unsupported_or_missing_sources.map((source) => source.id)).toEqual(
      expect.arrayContaining([
        "missing:retrieval_evidence",
        "missing:dependency_sources",
        "missing:evaluation_sources",
        "missing:artifact_sources"
      ])
    );
  });

  it("aggregates retrieval and artifact provenance from the workspace snapshot", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const workstationState = buildWorkstationState({ snapshot });
    const provenance = buildProvenanceEngineModel({
      snapshot,
      traceEvents: [],
      workstationState
    });

    expect(provenance.evidence_sources[0]).toMatchObject({
      kind: "retrieval",
      label: "WebGPU API",
      status: "available"
    });
    expect(provenance.artifact_sources.map((source) => source.label)).toEqual([
      "aurora-field.p5.js",
      "preview-request.json",
      "projection-notes.md"
    ]);
    expect(provenance.provenance_summary).toContain("provenance sources available");
  });

  it("aggregates dependencies, evaluation, artifacts, and final payload provenance", () => {
    const snapshot = getInitialWorkspaceSnapshot();
    const traceEvents = [
      traceEvent({
        eventType: "planning",
        nodeId: "planning",
        payload: {
          creative_reasoning: {
            role: "creative_reasoning_engine",
            rationale: "Use evidence from the prompt and runtime metadata."
          },
          artifact_dependency_graph: {
            role: "artifact_dependency_graph",
            summary: "Primary artifact depends on the preview manifest."
          }
        },
        sequence: 1
      }),
      traceEvent({
        eventType: "artifact_extracted",
        nodeId: "artifact_extraction",
        payload: {
          artifacts: [
            {
              id: "generated-sketch",
              language: "p5.js",
              summary: "Generated sketch artifact.",
              title: "generated-sketch.p5.js"
            }
          ]
        },
        sequence: 2
      }),
      traceEvent({
        eventType: "eval_update",
        nodeId: "review",
        payload: {
          evaluation: {
            evaluation_summary: "Evaluation confirmed preview readiness.",
            run_id: "eval-run-1"
          }
        },
        sequence: 3
      }),
      traceEvent({
        eventType: "final",
        nodeId: "finalization",
        payload: {
          answer: "Final answer."
        },
        sequence: 4
      })
    ];
    const workstationState = buildWorkstationState({
      activeWorkflowNodeId: "finalization",
      snapshot,
      traceEvents
    });
    const provenance = buildProvenanceEngineModel({
      snapshot,
      traceEvents,
      workstationState
    });

    expect(provenance.evidence_sources[0]).toMatchObject({
      kind: "reasoning",
      summary: "Use evidence from the prompt and runtime metadata."
    });
    expect(provenance.dependency_sources[0]).toMatchObject({
      label: "Artifact dependency graph",
      status: "available"
    });
    expect(provenance.artifact_sources[0]).toMatchObject({
      label: "generated-sketch.p5.js",
      summary: "Generated sketch artifact."
    });
    expect(provenance.evaluation_sources).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          label: "Live evaluation",
          summary: "Evaluation confirmed preview readiness."
        }),
        expect.objectContaining({
          label: "Final payload",
          summary: "Final answer."
        })
      ])
    );
  });
});

function traceEvent({
  eventType,
  nodeId,
  payload,
  sequence
}: {
  eventType: AssistantStreamEventType;
  nodeId: WorkflowNodeId;
  payload: Record<string, unknown>;
  sequence: number;
}): WorkflowRuntimeTraceEvent {
  const event: AssistantStreamEvent = {
    event_type: eventType,
    sequence,
    payload: {
      ...payload,
      emitted_at: `2026-06-26T09:10:0${sequence}.000Z`,
      workflow: {
        current_step: nodeId,
        phase: eventType === "final" ? "completed" : "running",
        status: eventType === "final" ? "completed" : "running",
        step: nodeId
      }
    }
  };

  return {
    event,
    receivedAt: `2026-06-26T09:10:0${sequence}.000Z`,
    receivedAtMs: Date.parse(`2026-06-26T09:10:0${sequence}.000Z`)
  };
}
