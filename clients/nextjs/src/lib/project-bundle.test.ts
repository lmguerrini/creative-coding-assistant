import { describe, expect, it } from "vitest";
import {
  getLocalWorkspaceSnapshot,
  type AssistantWorkspaceSnapshot
} from "./assistant-client";
import { buildPreviewControllerModel } from "./preview-controller";
import { buildProjectBundle } from "./project-bundle";
import { buildPreviewRendererRoute } from "./preview-renderers";
import { buildPreviewRuntimeSource } from "./preview-runtime-adapters";
import { buildRetrievalRuntimeModel } from "./retrieval-runtime";
import { buildWorkflowRuntimeModel } from "./workflow-runtime";
import { createWorkspaceSessionRecord } from "./workspace-persistence";

const textDecoder = new TextDecoder();

describe("project bundle export", () => {
  it("builds a zip-ready bundle with runtime, session, and multimodal exports", () => {
    const snapshot = createSnapshotWithImageReference();
    const bundle = buildBundle(snapshot);

    expect(bundle.fileName).toBe("local-nextjs-workspace-bundle.zip");
    expect(bundle.files.map((file) => file.path)).toEqual(
      expect.arrayContaining([
        "manifest.json",
        "README.md",
        "artifacts/aurora-field.p5.js",
        "artifacts/preview-request.json",
        "artifacts/projection-notes.md",
        "session/workspace-session.json",
        "runtime/workflow-summary.json",
        "runtime/retrieval-summary.json",
        "runtime/preview-config.json",
        "runtime/operator-checkpoints.json",
        "multimodal/image-references.json",
        "multimodal/images/reference-board.png"
      ])
    );

    expect(bundle.manifest.fileCount).toBe(bundle.files.length);
    expect(
      bundle.manifest.files.some((file) => file.path === "manifest.json")
    ).toBe(true);

    const workflowSummary = readJsonFile<{
      workflow: { steps: Array<{ nodeId: string; displayLabel: string }> };
      runtime: { summary: { currentNode: string } };
    }>(bundle, "runtime/workflow-summary.json");
    expect(workflowSummary.workflow.steps.map((step) => step.nodeId)).toEqual(
      expect.arrayContaining([
        "intake",
        "context_assembly",
        "reasoning",
        "prompt_rendering",
        "generation",
        "review",
        "refinement",
        "finalization",
        "failure"
      ])
    );
    expect(workflowSummary.runtime.summary.currentNode).toBe("generation");

    const previewConfig = readJsonFile<{
      route: { rendererLabel: string };
      runtimeSource: { title: string; lineCount: number };
    }>(bundle, "runtime/preview-config.json");
    expect(previewConfig.route.rendererLabel).toBe("JSON panel surface");
    expect(previewConfig.runtimeSource.title).toBe("aurora-field.p5.js");

    const multimodalMetadata = readJsonFile<{
      images: Array<{
        id: string;
        included: boolean;
        mimeType: string;
        path: string | null;
      }>;
    }>(bundle, "multimodal/image-references.json");
    expect(multimodalMetadata.images).toMatchObject([
      {
        id: "img-1",
        included: true,
        mimeType: "image/png",
        path: "multimodal/images/reference-board.png"
      }
    ]);
    expect(
      textDecoder.decode(readFile(bundle, "multimodal/images/reference-board.png").bytes)
    ).toBe("palette");
  });

  it("keeps image references as metadata when binary payloads are unavailable", () => {
    const snapshot = createSnapshotWithImageReference("not-a-data-url");
    const bundle = buildBundle(snapshot);

    expect(bundle.manifest.multimodal.includedImageCount).toBe(0);
    expect(bundle.manifest.warnings).toEqual([
      "Image reference reference-board was exported as metadata only."
    ]);
    expect(
      bundle.files.some((file) => file.path.startsWith("multimodal/images/"))
    ).toBe(false);

    const metadata = readJsonFile<{
      images: Array<{ id: string; included: boolean; path: string | null }>;
    }>(bundle, "multimodal/image-references.json");
    expect(metadata.images).toMatchObject([
      {
        id: "img-1",
        included: false,
        path: null
      }
    ]);
  });
});

function buildBundle(snapshot: AssistantWorkspaceSnapshot) {
  const previewRoute = buildPreviewRendererRoute({
    artifacts: snapshot.artifacts,
    preview: snapshot.preview,
    previewArtifactId: snapshot.artifacts[1]?.id ?? snapshot.artifacts[0]?.id ?? ""
  });

  return buildProjectBundle({
    approvalSummary: {
      activeRequest: null,
      latestRequest: null,
      pendingCount: 0
    },
    exportedAt: "2026-05-26T10:00:00.000Z",
    persistenceRecord: createWorkspaceSessionRecord({
      activeArtifactId: snapshot.artifacts[0]?.id ?? "",
      activeInspectorTab: "Overview",
      previewArtifactId: snapshot.artifacts[1]?.id ?? "",
      previewOpen: snapshot.preview.active,
      snapshot
    }),
    previewController: buildPreviewControllerModel({
      isFullscreen: false,
      preview: snapshot.preview,
      route: previewRoute,
      sessionOverride: null
    }),
    previewRoute,
    previewRuntimeSource: buildPreviewRuntimeSource({
      code: snapshot.code,
      route: previewRoute
    }),
    retrievalRuntime: buildRetrievalRuntimeModel(snapshot.retrieval, []),
    snapshot,
    workflowRuntime: buildWorkflowRuntimeModel(snapshot.workflow, [])
  });
}

function createSnapshotWithImageReference(
  dataUrl = "data:image/png;base64,cGFsZXR0ZQ=="
): AssistantWorkspaceSnapshot {
  const snapshot = getLocalWorkspaceSnapshot();

  return {
    ...snapshot,
    multimodal: {
      ...snapshot.multimodal,
      state: "ready",
      status: "1 image reference attached",
      detail: "Reference board is pinned for palette and composition cues.",
      imageAttachments: [
        {
          id: "img-1",
          kind: "image",
          name: "reference-board",
          mimeType: "image/png",
          sizeBytes: 7,
          dataUrl,
          createdAt: "2026-05-26T09:58:00.000Z"
        }
      ]
    }
  };
}

function readJsonFile<T>(bundle: ReturnType<typeof buildBundle>, path: string) {
  return JSON.parse(textDecoder.decode(readFile(bundle, path).bytes)) as T;
}

function readFile(bundle: ReturnType<typeof buildBundle>, path: string) {
  const file = bundle.files.find((candidate) => candidate.path === path);
  expect(file).toBeDefined();
  return file!;
}
