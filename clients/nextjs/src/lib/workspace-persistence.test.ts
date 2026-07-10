import { describe, expect, it, vi } from "vitest";
import { getLocalWorkspaceSnapshot } from "./assistant-client";
import {
  createWorkspacePersistenceClient,
  createWorkspaceSessionRecord,
  defaultWorkspacePreferences,
  defaultWorkspaceLayoutState,
  fingerprintWorkspaceSessionRecord,
  normalizeWorkspacePreferences,
  normalizeWorkspaceLayoutState,
  snapshotFromWorkspaceSessionRecord,
  type WorkspaceSessionRecord
} from "./workspace-persistence";
import { getP5RuntimeSourceSupportIssue } from "./preview-source-classification";

describe("workspace persistence client", () => {
  it("builds a typed session record from the workspace snapshot", () => {
    const snapshot = getLocalWorkspaceSnapshot();

    const record = createWorkspaceSessionRecord({
      activeArtifactId: "source-sketch",
      activeInspectorTab: "Code",
      previewArtifactId: "preview-manifest",
      previewOpen: true,
      snapshot
    });

    expect(record).toMatchObject({
      schemaVersion: 4,
      userId: "local-user",
      sessionId: "local-nextjs-session",
      projectId: "local-nextjs-workspace",
      activeArtifactId: "source-sketch",
      activeInspectorTab: "Code",
      previewOpen: true,
      previewArtifactId: "preview-manifest",
      layout: defaultWorkspaceLayoutState,
      preferences: defaultWorkspacePreferences,
      multimodal: {
        state: "empty",
        imageAttachments: []
      }
    });
    expect(record.messages).toEqual(snapshot.messages);
    expect(record.artifacts).toHaveLength(3);
    expect(record.updatedAt).toBeDefined();
  });

  it("compacts oversized live workspace snapshots for remote persistence", () => {
    const largeText = "live generated code and reasoning ".repeat(15_000);
    const snapshot = {
      ...getLocalWorkspaceSnapshot(),
      messages: [
        {
          role: "user" as const,
          time: "12:00",
          content: largeText
        },
        {
          role: "assistant" as const,
          time: "12:01",
          content: largeText
        }
      ],
      artifacts: getLocalWorkspaceSnapshot().artifacts.map((artifact) => ({
        ...artifact,
        content: largeText
      })),
      code: {
        ...getLocalWorkspaceSnapshot().code,
        excerpt: [largeText, largeText]
      },
      debug: {
        ...getLocalWorkspaceSnapshot().debug,
        events: Array.from({ length: 80 }, (_, index) => ({
          code: `event_${index}`,
          label: `Event ${index}`,
          detail: largeText
        }))
      }
    };

    const record = createWorkspaceSessionRecord({
      activeArtifactId: "source-sketch",
      activeInspectorTab: "Code",
      previewArtifactId: "preview-manifest",
      previewOpen: true,
      snapshot
    });

    expect(JSON.stringify(record).length).toBeLessThan(256 * 1024);
    expect(record.messages).toHaveLength(2);
    expect(record.messages[1]?.content).toContain("[truncated]");
    expect(record.snapshot.debug.events).toHaveLength(40);
    expect(record.snapshot.code.excerpt.join("\n")).toContain(
      "exceeds the local session restore limit"
    );
    expect(record.artifacts[0]?.content).toBeUndefined();
  });

  it("does not restore a runnable preview when its oversized source was omitted", () => {
    const baseSnapshot = getLocalWorkspaceSnapshot();
    const source = [
      "function setup() { createCanvas(320, 180); }",
      "function draw() { background(10); }",
      ...Array.from({ length: 10_000 }, () => "// retained only in the original artifact")
    ].join("\n");
    const snapshot: typeof baseSnapshot = {
      ...baseSnapshot,
      artifacts: [
        {
          ...baseSnapshot.artifacts[0],
          content: source,
          domain: "p5",
          previewEligible: true,
          previewTarget: "browser_sandbox",
          rendererId: "surface.p5",
          runtime: "p5"
        }
      ],
      code: {
        ...baseSnapshot.code,
        excerpt: source.split("\n")
      }
    };
    const record = createWorkspaceSessionRecord({
      activeArtifactId: "source-sketch",
      activeInspectorTab: "Preview",
      previewArtifactId: "source-sketch",
      previewOpen: true,
      snapshot
    });

    const restored = snapshotFromWorkspaceSessionRecord(baseSnapshot, record);

    expect(record.artifacts[0]?.content).toBeUndefined();
    expect(record.snapshot.code.excerpt.join("\n")).toContain(
      "exceeds the local session restore limit"
    );
    expect(restored.preview).toMatchObject({
      available: false,
      active: false,
      collapsed: true,
      state: "unavailable",
      title: "Preview unavailable"
    });
    expect(restored.preview.summary).toContain("not restored");
  });

  it("restores a complete p5 source without collapsing its executable structure", () => {
    const source = [
      "function setup() {",
      "  createCanvas(windowWidth, windowHeight);",
      "  colorMode(HSL, 360, 100, 100, 1);",
      "}",
      "",
      "function draw() {",
      "  background(220, 24, 7, 0.08);",
      "  const phase = noise(mouseX * 0.01, mouseY * 0.01, frameCount * 0.01);",
      "  line(0, 0, phase * width, height);",
      "}"
    ];
    const baseSnapshot = getLocalWorkspaceSnapshot();
    const snapshot: typeof baseSnapshot = {
      ...baseSnapshot,
      artifacts: [
        {
          ...baseSnapshot.artifacts[0],
          id: "p5-flow-field",
          title: "generated-sketch-1.p5.js",
          language: "JavaScript",
          content: source.join("\n"),
          domain: "p5",
          runtime: "p5",
          rendererId: "preview.p5",
          previewEligible: true
        }
      ],
      code: {
        ...baseSnapshot.code,
        title: "generated-sketch-1.p5.js",
        language: "JavaScript",
        excerpt: source
      }
    };

    const record = createWorkspaceSessionRecord({
      activeArtifactId: "p5-flow-field",
      activeInspectorTab: "Preview",
      previewArtifactId: "p5-flow-field",
      previewOpen: true,
      snapshot
    });
    const restored = snapshotFromWorkspaceSessionRecord(snapshot, record);

    expect(record.artifacts[0]?.content).toBe(source.join("\n"));
    expect(record.snapshot.code.excerpt).toEqual(source);
    expect(restored.artifacts[0]?.content).toBe(source.join("\n"));
    expect(restored.code.excerpt).toEqual(source);
    expect(getP5RuntimeSourceSupportIssue(restored.code.excerpt.join("\n"))).toBeNull();
  });

  it("restores stale React Three Fiber metadata as code-only", () => {
    const baseSnapshot = getLocalWorkspaceSnapshot();
    const source = [
      'import { Canvas, useFrame } from "@react-three/fiber";',
      "function Orb() { useFrame(() => {}); return <mesh />; }",
      "export default function Study() { return <Canvas><Orb /></Canvas>; }"
    ].join("\n");
    const snapshot: typeof baseSnapshot = {
      ...baseSnapshot,
      artifacts: [
        {
          ...baseSnapshot.artifacts[0],
          id: "react-three-fiber-study",
          title: "generated-scene-1.three.ts",
          language: "TypeScript + Three.js",
          content: source,
          domain: "react_three_fiber",
          runtime: "three",
          rendererId: "surface.three",
          previewEligible: true,
          previewTarget: "browser_sandbox",
          actions: ["Open", "Preview", "Copy", "Download"],
          summary: "Extracted from the generation result; matched Three.js creative runtime."
        }
      ],
      preview: {
        ...baseSnapshot.preview,
        available: true,
        renderer: "surface.three",
        sourceArtifactId: "react-three-fiber-study",
        state: "ready"
      }
    };
    const record = createWorkspaceSessionRecord({
      activeArtifactId: "react-three-fiber-study",
      activeInspectorTab: "Code",
      previewArtifactId: "react-three-fiber-study",
      previewOpen: true,
      snapshot
    });

    const restored = snapshotFromWorkspaceSessionRecord(baseSnapshot, record);

    expect(restored.artifacts[0]).toMatchObject({
      title: "generated-study-1.r3f.tsx",
      language: "TypeScript + React Three Fiber",
      runtime: null,
      rendererId: null,
      previewEligible: false,
      previewTarget: "",
      actions: ["Open", "Copy", "Download"]
    });
    expect(restored.preview).toMatchObject({
      available: false,
      active: false,
      collapsed: true,
      state: "unavailable",
      title: "Preview unavailable"
    });
    expect(restored.preview.summary).toContain(
      "React Three Fiber components need their own bundle runtime"
    );
  });

  it("normalizes layout preferences into safe persisted values", () => {
    expect(
      normalizeWorkspaceLayoutState({
        density: "compact",
        inspectorCollapsed: true,
        inspectorWidth: 999,
        previewHeight: 120
      })
    ).toEqual({
      density: "compact",
      inspectorCollapsed: true,
      inspectorWidth: 560,
      previewHeight: 220
    });
    expect(normalizeWorkspaceLayoutState()).toEqual(defaultWorkspaceLayoutState);
  });

  it("normalizes workspace preferences into safe persisted values", () => {
    expect(
      normalizeWorkspacePreferences({
        theme: "matrix",
        autoOpenPreview: false,
        showDebugPanels: false
      })
    ).toEqual({
      theme: "matrix",
      autoOpenPreview: false,
      showDebugPanels: false
    });
    expect(
      normalizeWorkspacePreferences({
        theme: "invalid" as never
      })
    ).toEqual(defaultWorkspacePreferences);
  });

  it("restores messages, active tab, artifact, and preview state", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const record = {
      ...createWorkspaceSessionRecord({
        activeArtifactId: "session-notes",
        activeInspectorTab: "Artifacts",
        previewArtifactId: "preview-manifest",
        previewOpen: true,
        snapshot
      }),
      messages: [
        {
          role: "user",
          time: "11:00",
          content: "Restore this."
        }
      ]
    } satisfies WorkspaceSessionRecord;

    const restored = snapshotFromWorkspaceSessionRecord(snapshot, record);

    expect(restored.messages).toEqual(record.messages);
    expect(
      restored.inspectorTabs.find((tab) => tab.label === "Artifacts")?.active
    ).toBe(true);
    expect(restored.preview.active).toBe(true);
    expect(restored.preview.collapsed).toBe(false);
  });

  it("normalizes legacy workflow snapshots with newly introduced nodes", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const record = createWorkspaceSessionRecord({
      activeArtifactId: "source-sketch",
      activeInspectorTab: "Workflow",
      previewArtifactId: "preview-manifest",
      previewOpen: false,
      snapshot
    });
    const legacyWorkflow = {
      ...record.workflow,
      currentNode: "generation" as const,
      currentStep: "Generation",
      steps: record.workflow.steps
        .filter((step) => step.nodeId !== "planning")
        .map((step) =>
          step.nodeId === "prompt_rendering" || step.nodeId === "generation"
            ? { ...step, state: "active" as const }
            : step
        )
    };

    const restored = snapshotFromWorkspaceSessionRecord(snapshot, {
      ...record,
      workflow: legacyWorkflow,
      snapshot: {
        ...record.snapshot,
        workflow: legacyWorkflow
      }
    });

    expect(restored.workflow.steps.map((step) => step.nodeId)).toContain(
      "planning"
    );
    expect(restored.workflow.steps.map((step) => step.nodeId)).toEqual(
      snapshot.workflow.steps.map((step) => step.nodeId)
    );
    expect(
      restored.workflow.steps.find((step) => step.nodeId === "planning")
    ).toMatchObject({
      displayLabel: "Planning",
      state: "complete"
    });
  });

  it("loads a remote session and mirrors it to local storage", async () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const storage = new MemoryStorage();
    const record = createWorkspaceSessionRecord({
      activeArtifactId: "source-sketch",
      activeInspectorTab: "Overview",
      previewArtifactId: "source-sketch",
      previewOpen: false,
      snapshot
    });
    const fetchImpl = vi.fn(async () => new Response(JSON.stringify(record)));
    const client = createWorkspacePersistenceClient({ fetchImpl, storage });

    await expect(client.load()).resolves.toEqual({
      error: null,
      record,
      source: "remote"
    });

    expect(fetchImpl).toHaveBeenCalledWith(
      expect.stringContaining("/api/workspace/session?userId=local-user"),
      expect.objectContaining({ method: "GET" })
    );
    expect(storage.length).toBe(1);
  });

  it("saves remotely while keeping a local fallback copy", async () => {
    const storage = new MemoryStorage();
    const record = createWorkspaceSessionRecord({
      activeArtifactId: "source-sketch",
      activeInspectorTab: "Overview",
      previewArtifactId: "source-sketch",
      previewOpen: false,
      snapshot: getLocalWorkspaceSnapshot()
    });
    const fetchImpl = vi.fn(async () => new Response("{}"));
    const client = createWorkspacePersistenceClient({ fetchImpl, storage });

    await expect(client.save(record)).resolves.toEqual({
      error: null,
      target: "remote"
    });

    expect(fetchImpl).toHaveBeenCalledWith(
      "http://localhost:8000/api/workspace/session",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({ "Content-Type": "application/json" })
      })
    );
    expect(storage.length).toBe(1);
  });

  it("falls back to local session storage when the endpoint is unavailable", async () => {
    const storage = new MemoryStorage();
    const record = createWorkspaceSessionRecord({
      activeArtifactId: "source-sketch",
      activeInspectorTab: "Workflow",
      previewArtifactId: "preview-manifest",
      previewOpen: true,
      snapshot: getLocalWorkspaceSnapshot()
    });
    const offlineFetch = vi.fn(async () => {
      throw new Error("offline");
    });
    const client = createWorkspacePersistenceClient({
      fetchImpl: offlineFetch,
      storage
    });

    await expect(client.save(record)).resolves.toMatchObject({
      error: expect.objectContaining({
        category: "persistence",
        type: "session_save_unavailable"
      }),
      target: "local"
    });
    await expect(client.load()).resolves.toMatchObject({
      error: expect.objectContaining({
        category: "persistence",
        type: "session_restore_unavailable"
      }),
      record: expect.objectContaining({
        activeInspectorTab: "Workflow",
        previewOpen: true,
        layout: defaultWorkspaceLayoutState
      }),
      source: "local"
    });
  });

  it("ignores a malformed local session when the remote endpoint is unavailable", async () => {
    const storage = new MemoryStorage();
    storage.setItem(
      "cca.workspace.local-user.local-nextjs-session",
      '{"schemaVersion":4,"artifacts":"not-an-array"}'
    );
    const offlineFetch = vi.fn(async () => {
      throw new Error("offline");
    });
    const client = createWorkspacePersistenceClient({
      fetchImpl: offlineFetch,
      storage
    });

    await expect(client.load()).resolves.toMatchObject({
      error: expect.objectContaining({
        category: "persistence",
        type: "session_restore_unavailable"
      }),
      record: null,
      source: "none"
    });
  });

  it("fingerprints records without timestamp churn", () => {
    const record = createWorkspaceSessionRecord({
      activeArtifactId: "source-sketch",
      activeInspectorTab: "Overview",
      previewArtifactId: "source-sketch",
      previewOpen: false,
      snapshot: getLocalWorkspaceSnapshot()
    });

    expect(
      fingerprintWorkspaceSessionRecord({
        ...record,
        updatedAt: "later",
        snapshot: {
          ...record.snapshot,
          session: {
            ...record.snapshot.session,
            updatedAt: "later"
          }
        }
      })
    ).toBe(fingerprintWorkspaceSessionRecord(record));
  });
});

class MemoryStorage implements Storage {
  private readonly items = new Map<string, string>();

  get length() {
    return this.items.size;
  }

  clear() {
    this.items.clear();
  }

  getItem(key: string) {
    return this.items.get(key) ?? null;
  }

  key(index: number) {
    return Array.from(this.items.keys())[index] ?? null;
  }

  removeItem(key: string) {
    this.items.delete(key);
  }

  setItem(key: string, value: string) {
    this.items.set(key, value);
  }
}
