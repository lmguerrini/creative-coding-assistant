import { describe, expect, it, vi } from "vitest";
import { getLocalWorkspaceSnapshot } from "./assistant-client";
import {
  buildArtifactDocument,
  copyArtifactDocument,
  downloadArtifactDocument,
  formatArtifactActionLabel,
  highlightArtifactDocument
} from "./artifact-inspector";

describe("artifact inspector helpers", () => {
  it("builds readable source and preview documents for the inspector", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const codeArtifact = snapshot.artifacts[0];
    const previewArtifact = snapshot.artifacts[1];
    const exportArtifact = snapshot.artifacts[2];
    const previewDocument = buildArtifactDocument(snapshot, previewArtifact);
    const parsedPreviewDocument = JSON.parse(previewDocument.content) as {
      preview: { targetId: string };
      route: { surfaceKind: string; rendererLabel: string };
    };

    expect(buildArtifactDocument(snapshot, codeArtifact)).toMatchObject({
      fileName: "webgpu-particle-field.ts",
      languageLabel: "TypeScript + WGSL",
      lineCount: 7,
      mimeType: "text/typescript;charset=utf-8",
      typeLabel: "Source code"
    });
    expect(previewDocument).toMatchObject({
      fileName: "preview-request.json",
      mimeType: "application/json;charset=utf-8",
      typeLabel: "Preview manifest"
    });
    expect(parsedPreviewDocument.preview.targetId).toBe("browser_sandbox");
    expect(parsedPreviewDocument.route.surfaceKind).toBe("json_panel");
    expect(parsedPreviewDocument.route.rendererLabel).toBe("JSON panel surface");
    expect(buildArtifactDocument(snapshot, exportArtifact)).toMatchObject({
      fileName: "projection-notes.md",
      mimeType: "text/markdown;charset=utf-8",
      typeLabel: "Markdown export"
    });
  });

  it("applies lightweight token highlighting for script and json artifacts", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const codeDocument = buildArtifactDocument(snapshot, snapshot.artifacts[0]);
    const previewDocument = buildArtifactDocument(snapshot, snapshot.artifacts[1]);

    const codeLines = highlightArtifactDocument(codeDocument);
    const previewLines = highlightArtifactDocument(previewDocument);

    expect(codeLines[0].tokens).toContainEqual({
      kind: "keyword",
      text: "const"
    });
    expect(codeLines[0].tokens).toContainEqual({
      kind: "function",
      text: "sampleBand"
    });
    expect(previewLines[1].tokens).toContainEqual({
      kind: "property",
      text: '"artifactId":'
    });
  });

  it("copies, downloads, and labels artifact actions clearly", async () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const document = buildArtifactDocument(snapshot, snapshot.artifacts[0]);
    const clipboard = {
      writeText: vi.fn(async () => undefined)
    };
    const click = vi.fn();
    const anchor = {
      click,
      download: "",
      href: ""
    };
    const downloadApi = {
      createAnchor: vi.fn(() => anchor),
      createObjectURL: vi.fn(() => "blob:artifact"),
      revokeObjectURL: vi.fn()
    };

    await expect(copyArtifactDocument(document, clipboard)).resolves.toBe(true);
    expect(clipboard.writeText).toHaveBeenCalledWith(document.content);
    expect(downloadArtifactDocument(document, downloadApi)).toBe(true);
    expect(anchor.download).toBe("webgpu-particle-field.ts");
    expect(anchor.href).toBe("blob:artifact");
    expect(click).toHaveBeenCalledTimes(1);
    expect(downloadApi.revokeObjectURL).toHaveBeenCalledWith("blob:artifact");
    expect(formatArtifactActionLabel("Open")).toBe("Open in Code");
    expect(formatArtifactActionLabel("Export")).toBe("Export File");
    expect(
      formatArtifactActionLabel("Export", { type: "export" })
    ).toBe("Export Bundle");
  });
});
