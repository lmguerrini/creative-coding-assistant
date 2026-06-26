import { describe, expect, it } from "vitest";
import {
  getInitialWorkspaceSnapshot,
  type AssistantWorkspaceSnapshot,
  type WorkflowNodeId
} from "./assistant-client";
import type {
  AssistantStreamEvent,
  AssistantStreamEventType
} from "./assistant-stream";
import { buildCreativeTimelineModel } from "./creative-timeline";
import { buildProvenanceEngineModel } from "./provenance-engine";
import {
  buildSessionIntelligenceModel,
  readSessionIntelligenceMetadata
} from "./session-intelligence";
import { buildV3InspectorPanelsModel } from "./v3-inspector-panels";
import { buildWorkflowExplorerModel } from "./workflow-explorer";
import {
  buildWorkflowRuntimeModel,
  type WorkflowRuntimeTraceEvent
} from "./workflow-runtime";
import { buildWorkstationDashboardModel } from "./workstation-dashboard";
import { buildWorkstationState } from "./workstation-state";

describe("workstation surface integration", () => {
  it("builds all seven workstation surfaces from one fixture in dependency order", () => {
    const fixture = workstationSurfaceFixture();
    const runtime = buildWorkflowRuntimeModel(
      fixture.snapshot.workflow,
      fixture.traceEvents
    );

    const workstationState = buildWorkstationState({
      activeWorkflowNodeId: "finalization",
      snapshot: fixture.snapshot,
      traceEvents: fixture.traceEvents
    });
    const sessionIntelligence = buildSessionIntelligenceModel({
      snapshot: fixture.snapshot,
      streamedMetadata: readSessionIntelligenceMetadata(
        fixture.traceEvents.at(-1)?.event.payload ?? {}
      ),
      workstationState
    });
    const workflowExplorer = buildWorkflowExplorerModel({
      runtime,
      snapshot: fixture.snapshot,
      traceEvents: fixture.traceEvents,
      workstationState
    });
    const provenance = buildProvenanceEngineModel({
      snapshot: fixture.snapshot,
      traceEvents: fixture.traceEvents,
      workstationState
    });
    const creativeTimeline = buildCreativeTimelineModel({
      explorer: workflowExplorer,
      provenance,
      runtime,
      workstationState
    });
    const v3InspectorPanels = buildV3InspectorPanelsModel({
      provenance,
      traceEvents: fixture.traceEvents,
      workstationState
    });
    const workstationDashboard = buildWorkstationDashboardModel({
      runtime,
      snapshot: fixture.snapshot,
      v3InspectorPanels,
      workstationState
    });

    expect(workstationState.currentRun.state).toBe("completed");
    expect(sessionIntelligence.source).toBe("stream");
    expect(sessionIntelligence.metadata.session_summary).toBe(
      "Completed workstation integration request."
    );
    expect(workflowExplorer.state).toBe("available");
    expect(workflowExplorer.summary.availableMetadataGroupCount).toBeGreaterThan(0);
    expect(provenance.totals.availableSourceCount).toBeGreaterThan(0);
    expect(creativeTimeline.events.map((event) => event.id)).toEqual([
      "request_intake",
      "planning",
      "retrieval",
      "creative_intelligence",
      "generative_design",
      "artifact_intelligence",
      "creative_evaluation",
      "final_synthesis"
    ]);
    expect(creativeTimeline.summary.missingCount).toBe(0);
    expect(v3InspectorPanels.summary.availableItemCount).toBeGreaterThan(0);
    expect(workstationDashboard.cards).toHaveLength(8);
    expect(workstationDashboard.summary.goodCount).toBeGreaterThan(0);
  });
});

function workstationSurfaceFixture(): {
  snapshot: AssistantWorkspaceSnapshot;
  traceEvents: WorkflowRuntimeTraceEvent[];
} {
  return {
    snapshot: getInitialWorkspaceSnapshot(),
    traceEvents: [
      traceEvent({
        eventType: "planning",
        nodeId: "planning",
        payload: {
          creative_constraints: {
            summary: "Keep the sketch browser-safe."
          },
          creative_intent: {
            normalizedIntent: "Build an aurora particle field."
          },
          creative_plan: {
            generationStrategy: "Generate one p5.js sketch."
          },
          creative_reasoning: {
            recommendedCreativeDirection: "Use layered luminous particles."
          },
          creative_score: {
            scoreSummary: "Creative quality is strong for review."
          },
          creative_strategy: {
            rationale: "Use recursive motion and restrained color."
          },
          cross_modality: {
            compositionPattern: "Visual motion can follow ambient energy."
          },
          procedural_structure: {
            spatialStructurePlan: "Layered particle lattice."
          },
          runtime_capabilities: {
            summary: "p5.js preview runtime is available."
          },
          semantic_motif: {
            motifSystemName: "Aurora orbit motif."
          }
        },
        sequence: 1
      }),
      traceEvent({
        eventType: "retrieval",
        nodeId: "retrieval",
        payload: {
          assembled_context: {
            summary: "Assembled p5.js runtime context."
          },
          retrieval: {
            summary: "Retrieved p5.js setup and draw references."
          }
        },
        sequence: 2
      }),
      traceEvent({
        eventType: "artifact_extracted",
        nodeId: "artifact_extraction",
        payload: {
          artifact_capability_matrix: {
            strongestTargets: ["p5.js"],
            summary: "The generated sketch is previewable."
          },
          artifact_critic: {
            rationale: "Candidate is coherent and runnable."
          },
          artifact_dependency_graph: {
            summary: "The sketch depends on p5 setup and draw lifecycle."
          },
          artifact_plan: {
            primaryArtifactIntent: "Generate a single p5.js artifact."
          },
          artifacts: [
            {
              id: "aurora-field",
              language: "p5.js",
              summary: "Aurora particle sketch.",
              title: "aurora-field.p5.js"
            }
          ]
        },
        sequence: 3
      }),
      traceEvent({
        eventType: "eval_update",
        nodeId: "review",
        payload: {
          consistency_validation: {
            summary: "The artifact stays aligned with the creative brief."
          },
          creative_confidence: {
            confidenceSummary: "Confidence is high with normal review."
          },
          evaluation: {
            evaluation_summary: "Live evaluation passed."
          },
          evaluation_report: {
            executiveSummary: "No blocking quality issues."
          },
          self_evaluation: {
            evaluation_summary: "The response satisfies the prompt."
          }
        },
        sequence: 4
      }),
      traceEvent({
        eventType: "final",
        nodeId: "finalization",
        payload: {
          answer: "Final aurora particle field response.",
          outputs: [
            {
              id: "aurora-field",
              summary: "Aurora particle sketch.",
              title: "aurora-field.p5.js"
            }
          ],
          session_intelligence: {
            active_request_summary: "Completed aurora particle field.",
            available_metadata_groups: [
              "Session",
              "Selected workflow",
              "Creative plan",
              "Preview"
            ],
            completion_status: "completed",
            recommended_next_user_actions: [
              "Review the generated artifact in the preview."
            ],
            session_summary: "Completed workstation integration request.",
            session_warnings: []
          }
        },
        sequence: 5
      })
    ]
  };
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
      emitted_at: `2026-06-26T10:00:0${sequence}.000Z`,
      workflow: {
        ...payload,
        current_step: nodeId,
        phase: eventType === "final" ? "completed" : "running",
        status: eventType === "final" ? "completed" : "running",
        step: nodeId
      }
    }
  };

  return {
    event,
    receivedAt: `2026-06-26T10:00:0${sequence}.000Z`,
    receivedAtMs: Date.parse(`2026-06-26T10:00:0${sequence}.000Z`)
  };
}
