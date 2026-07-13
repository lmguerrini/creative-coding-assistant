import { describe, expect, it } from "vitest";
import { getLocalWorkspaceSnapshot } from "./assistant-client";
import {
  buildMultimodalSummary,
  createImageAttachmentFromFile,
  maxImageAttachmentBytes,
  maxImageAttachmentCount,
  normalizeImageAttachments,
  toAssistantRequestImageAttachments
} from "./multimodal-attachments";

const pngSignature = new Uint8Array([
  0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a
]);
const webpSignature = new Uint8Array([
  0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50
]);
const pngSignatureDataUrl = "data:image/png;base64,iVBORw0KGgo=";

describe("multimodal attachments", () => {
  it("creates bounded image attachments from supported uploads", async () => {
    const result = await createImageAttachmentFromFile({
      createdAt: "2026-05-25T10:00:00Z",
      existingCount: 0,
      file: new File([pngSignature], "reference.png", { type: "image/png" }),
      id: "image-reference-1"
    });

    expect(result).toMatchObject({ ok: true });
    if (!result.ok) {
      throw new Error("Expected image attachment to be created.");
    }

    expect(result.attachment).toMatchObject({
      id: "image-reference-1",
      kind: "image",
      name: "reference.png",
      mimeType: "image/png",
      sizeBytes: pngSignature.byteLength
    });
    expect(result.attachment.dataUrl).toMatch(/^data:image\/png;base64,/);
    expect(toAssistantRequestImageAttachments([result.attachment])).toEqual([
      expect.objectContaining({
        type: "image",
        id: "image-reference-1",
        name: "reference.png",
        dataUrl: result.attachment.dataUrl
      })
    ]);
  });

  it("returns workstation errors for unsupported or oversized images", async () => {
    await expect(
      createImageAttachmentFromFile({
        createdAt: "2026-05-25T10:00:00Z",
        existingCount: 0,
        file: new File(["notes"], "notes.txt", { type: "text/plain" }),
        id: "bad-reference"
      })
    ).resolves.toMatchObject({
      ok: false,
      error: {
        category: "multimodal",
        subsystem: "image_upload",
        type: "unsupported_image_reference"
      }
    });

    await expect(
      createImageAttachmentFromFile({
        createdAt: "2026-05-25T10:00:00Z",
        existingCount: maxImageAttachmentCount,
        file: new File(["image-bytes"], "reference.png", { type: "image/png" }),
        id: "too-many"
      })
    ).resolves.toMatchObject({
      ok: false,
      error: {
        type: "image_upload_limit_reached"
      }
    });

    await expect(
      createImageAttachmentFromFile({
        createdAt: "2026-05-25T10:00:00Z",
        existingCount: 0,
        file: new File([new Uint8Array(maxImageAttachmentBytes + 1)], "large.png", {
          type: "image/png"
        }),
        id: "large-reference"
      })
    ).resolves.toMatchObject({
      ok: false,
      error: {
        type: "image_reference_too_large"
      }
    });
  });

  it("summarizes ready image context for the workspace", async () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const result = await createImageAttachmentFromFile({
      createdAt: "2026-05-25T10:00:00Z",
      existingCount: 0,
      file: new File([webpSignature], "palette.webp", { type: "image/webp" }),
      id: "image-reference-1"
    });

    if (!result.ok) {
      throw new Error("Expected image attachment to be created.");
    }

    expect(
      buildMultimodalSummary({
        baseMultimodal: snapshot.multimodal,
        imageAttachments: [result.attachment],
        uploadError: null
      })
    ).toMatchObject({
      state: "ready",
      status: "1 image reference",
      detail:
        "palette.webp stays local until you submit a generation request. The browser does not perform pixel analysis; the backend includes accepted pixels in the provider request payload for that explicit request. Provider receipt, use, and influence still need live evidence."
    });
  });

  it("revalidates local image bytes before serializing an explicit request", () => {
    const validAttachment = {
      id: "image-reference-valid",
      kind: "image" as const,
      name: "palette.png",
      mimeType: "image/png",
      sizeBytes: pngSignature.byteLength,
      dataUrl: pngSignatureDataUrl,
      createdAt: "2026-05-25T10:00:00Z"
    };
    const forgedAttachment = {
      ...validAttachment,
      id: "image-reference-forged",
      sizeBytes: 9
    };

    expect(normalizeImageAttachments([validAttachment, forgedAttachment])).toEqual([
      validAttachment
    ]);
    expect(
      toAssistantRequestImageAttachments([validAttachment, forgedAttachment])
    ).toEqual([
      {
        type: "image",
        id: "image-reference-valid",
        name: "palette.png",
        mimeType: "image/png",
        sizeBytes: pngSignature.byteLength,
        dataUrl: pngSignatureDataUrl
      }
    ]);
    expect(toAssistantRequestImageAttachments([])).toEqual([]);
  });

  it("rejects a file whose bytes do not match its declared image format", async () => {
    await expect(
      createImageAttachmentFromFile({
        createdAt: "2026-05-25T10:00:00Z",
        existingCount: 0,
        file: new File(["plain text"], "forged.png", { type: "image/png" }),
        id: "forged-reference"
      })
    ).resolves.toMatchObject({
      ok: false,
      error: {
        type: "image_reference_signature_mismatch"
      }
    });
  });
});
