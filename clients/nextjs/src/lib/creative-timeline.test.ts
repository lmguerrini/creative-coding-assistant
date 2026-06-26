import { describe, expect, it } from "vitest";
import {
  getInitialWorkspaceSnapshot,
  getLocalWorkspaceSnapshot,
  type AssistantWorkspaceSnapshot,
  type WorkflowNodeId
} from "./assistant-client";
import type { AssistantStreamEvent, AssistantStreamEventType } from "./assistant-stream";
import {
  buildCreativeTimelineModel,
  type CreativeTimelineModel
} from "./creative-timeline";
import { buildProvenanceEngineModel } from "./provenance-engine";
import { buildWorkflowExplorerModel } from "./workflow-explorer";
import {
  buildWorkflowRuntimeModel,
  type WorkflowRuntimeTraceEvent
} from "./workflow-runtime";
import { buildWorkstationState } from "./workstation-state";

describe("creative timeline model", () => {
  it("orders the full request evolution and protects missing metadata", () => {
    const timeline = buildTimeline(getInitialWorkspaceSnapshot());

    expect(timeline.events.map((event) => event.label)).toEqual([
      "Request intake",
      "Planning",
      "Retrieval",
      "Creative intelligence",
      "Generative design",
      "Artifact intelligence",
      "Creative evaluation",
      "Final synthesis"
    ]);
    expect(timeline.state).toBe("available");
    expect(event(timeline, "request_intake")).toMatchObject({
      metadataAvailability: "Session metadata available",
      sourceCount: 1,
      status: "complete",
      warning: null
    });
    expect(event(timeline, "planning")).toMatchObject({
      metadataAvailability: "0/3 metadata groups",
      sourceCount: 0,
      status: "missing",
      warning: "No metadata has been captured for this stage."
    });
    expect(timeline.summary).toMatchObject({
      completeCount: 1,
      missingCount: 7
    });
  });

  it("surfaces partial snapshot-backed stages with metadata and provenance counts", () => {
    const timeline = buildTimeline(getLocalWorkspaceSnapshot());

    expect(timeline.state).toBe("available");
    expect(event(timeline, "retrieval")).toMatchObject({
      metadataAvailability: "1/2 metadata groups",
      sourceCount: 2,
      status: "warning"
    });
    expect(event(timeline, "artifact_intelligence")).toMatchObject({
      metadataAvailability: "1/3 metadata groups",
      sourceCount: 3,
      status: "warning"
    });
    expect(event(timeline, "final_synthesis").summary).toContain(
      "Drafting a p5.js sketch"
    );
    expect(timeline.summary.warningCount).toBeGreaterThanOrEqual(3);
  });

  it("links streamed workstation metadata into completed timeline stages", () => {
    const snapshot = getInitialWorkspaceSnapshot();
    const traceEvents = [
      traceEvent({
        eventType: "planning",
        nodeId: "planning",
        payload: {
          artifact_dependency_graph: {
            summary: "The sketch depends on the p5 runtime and preview route."
          },
          creative_constraints: ["Keep the code browser-safe."],
          creative_intent: {
            summary: "Translate the request into a kinetic field."
          },
          creative_plan: {
            generationStrategy: "Plan one responsive sketch."
          },
          creative_reasoning: {
            rationale: "Use prompt evidence and runtime metadata."
          },
          creative_strategy: {
            rationale: "Layer recursive motion and restrained color."
          },
          cross_modality: {
            summary: "Optional audio energy can scale particle motion."
          },
          procedural_structure: {
            summary: "Recursive lattice with bounded particle trails."
          },
          runtime_capabilities: {
            summary: "p5.js browser runtime is available."
          },
          semantic_motif: {
            summary: "Aurora orbit motif."
          }
        },
        sequence: 1
      }),
      traceEvent({
        eventType: "retrieval",
        nodeId: "retrieval",
        payload: {
          assembled_context: {
            summary: "Assembled creative coding context."
          },
          retrieval: {
            summary: "Retrieved p5 runtime notes."
          }
        },
        sequence: 2
      }),
      traceEvent({
        eventType: "artifact_extracted",
        nodeId: "artifact_extraction",
        payload: {
          artifact_capability_matrix: {
            summary: "The generated sketch is previewable."
          },
          artifact_critic: {
            rationale: "Candidate is coherent and runnable."
          },
          artifact_plan: {
            summary: "Generate a single p5.js artifact."
          },
          artifacts: [
            {
              id: "kinetic-field",
              language: "p5.js",
              summary: "Kinetic field sketch.",
              title: "kinetic-field.p5.js"
            }
          ]
        },
        sequence: 3
      }),
      traceEvent({
        eventType: "eval_update",
        nodeId: "review",
        payload: {
          evaluation: {
            evaluation_summary: "Live evaluation passed."
          },
          evaluation_report: {
            summary: "No blocking quality issues."
          },
          creative_confidence: {
            summary: "Quality signal confidence is high."
          },
          self_evaluation: {
            evaluation_summary: "The output satisfies the prompt."
          }
        },
        sequence: 4
      }),
      traceEvent({
        eventType: "final",
        nodeId: "finalization",
        payload: {
          answer: "Final answer.",
          outputs: [
            {
              id: "kinetic-field",
              summary: "Kinetic field sketch.",
              title: "kinetic-field.p5.js"
            }
          ],
          session_intelligence: {
            session_summary: "Completed kinetic field request."
          }
        },
        sequence: 5
      })
    ];
    const timeline = buildTimeline(snapshot, {
      activeWorkflowNodeId: "finalization",
      traceEvents
    });

    expect(timeline.summary).toMatchObject({
      completeCount: 8,
      missingCount: 0,
      warningCount: 0
    });
    expect(event(timeline, "planning")).toMatchObject({
      metadataAvailability: "3/3 metadata groups",
      status: "complete"
    });
    expect(event(timeline, "artifact_intelligence").sourceCount).toBeGreaterThan(1);
    expect(event(timeline, "creative_evaluation")).toMatchObject({
      metadataAvailability: "3/3 metadata groups",
      status: "complete"
    });
    expect(event(timeline, "final_synthesis")).toMatchObject({
      metadataAvailability: "3/3 metadata groups",
      sourceCount: 1,
      status: "complete",
      warning: null
    });
  });
});

function buildTimeline(
  snapshot: AssistantWorkspaceSnapshot,
  options: {
    activeWorkflowNodeId?: WorkflowNodeId;
    traceEvents?: WorkflowRuntimeTraceEvent[];
  } = {}
): CreativeTimelineModel {
  const traceEvents = options.traceEvents ?? [];
  const runtime = buildWorkflowRuntimeModel(snapshot.workflow, traceEvents);
  const workstationState = buildWorkstationState({
    activeWorkflowNodeId: options.activeWorkflowNodeId,
    snapshot,
    traceEvents
  });
  const explorer = buildWorkflowExplorerModel({
    runtime,
    snapshot,
    traceEvents,
    workstationState
  });
  const provenance = buildProvenanceEngineModel({
    snapshot,
    traceEvents,
    workstationState
  });

  return buildCreativeTimelineModel({
    explorer,
    provenance,
    runtime,
    workstationState
  });
}

function event(timeline: CreativeTimelineModel, id: string) {
  const match = timeline.events.find((candidate) => candidate.id === id);
  expect(match).toBeDefined();
  return match!;
}

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
      emitted_at: `2026-06-26T09:20:0${sequence}.000Z`,
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
    receivedAt: `2026-06-26T09:20:0${sequence}.000Z`,
    receivedAtMs: Date.parse(`2026-06-26T09:20:0${sequence}.000Z`)
  };
}
