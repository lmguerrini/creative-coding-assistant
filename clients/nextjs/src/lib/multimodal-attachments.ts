import type {
  ImageAttachmentSummary,
  MultimodalSummary
} from "./assistant-client";
import {
  createWorkstationError,
  type WorkstationError
} from "./workstation-errors";

export const supportedImageMimeTypes = [
  "image/png",
  "image/jpeg",
  "image/webp",
  "image/gif"
] as const;

type SupportedImageMimeType = (typeof supportedImageMimeTypes)[number];

export const supportedImageUploadAccept = supportedImageMimeTypes.join(",");
export const maxImageAttachmentBytes = 1024 * 1024;
export const maxImageAttachmentCount = 4;

export type AssistantRequestImageAttachment = {
  type: "image";
  id: string;
  name: string;
  mimeType: string;
  sizeBytes: number;
  dataUrl: string;
};

export type CreateImageAttachmentInput = {
  createdAt: string;
  file: File;
  id: string;
  existingCount: number;
};

export type CreateImageAttachmentResult =
  | { ok: true; attachment: ImageAttachmentSummary }
  | { ok: false; error: WorkstationError };

export async function createImageAttachmentFromFile({
  createdAt,
  existingCount,
  file,
  id
}: CreateImageAttachmentInput): Promise<CreateImageAttachmentResult> {
  const validationError = validateImageAttachmentFile(file, existingCount);
  if (validationError) {
    return { error: validationError, ok: false };
  }

  try {
    return {
      ok: true,
      attachment: {
        id,
        kind: "image",
        name: file.name || "Untitled image reference",
        mimeType: file.type,
        sizeBytes: file.size,
        dataUrl: await readFileAsDataUrl(file),
        createdAt
      }
    };
  } catch (error) {
    return {
      ok: false,
      error: createImageUploadError({
        type: "image_upload_read_failed",
        userMessage: "The image reference could not be read.",
        debugMessage: error instanceof Error ? error.message : null,
        suggestedAction:
          "Try attaching the image again, or choose a smaller PNG, JPEG, WebP, or GIF file."
      })
    };
  }
}

export function buildMultimodalSummary({
  baseMultimodal,
  imageAttachments,
  uploadError
}: {
  baseMultimodal: MultimodalSummary;
  imageAttachments: ImageAttachmentSummary[];
  uploadError: WorkstationError | null;
}): MultimodalSummary {
  if (uploadError) {
    return {
      ...baseMultimodal,
      state: "error",
      status: "Image upload issue",
      detail: uploadError.userMessage,
      imageAttachments,
      error: uploadError
    };
  }

  if (imageAttachments.length > 0) {
    return {
      ...baseMultimodal,
      state: "ready",
      status: `${imageAttachments.length} ${pluralize(
        imageAttachments.length,
        "image reference",
        "image references"
      )}`,
      detail: `${formatAttachmentNames(imageAttachments)} will be sent with the next request.`,
      imageAttachments,
      error: null
    };
  }

  return {
    ...baseMultimodal,
    state: "empty",
    status: "No image references",
    detail:
      baseMultimodal.detail ||
      "Attach image references to ground the next creative coding request visually.",
    imageAttachments: [],
    error: null
  };
}

export function normalizeImageAttachments(
  value: unknown
): ImageAttachmentSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.filter(isImageAttachmentSummary).slice(0, maxImageAttachmentCount);
}

export function toAssistantRequestImageAttachments(
  attachments: ImageAttachmentSummary[]
): AssistantRequestImageAttachment[] {
  return attachments.map((attachment) => ({
    type: "image",
    id: attachment.id,
    name: attachment.name,
    mimeType: attachment.mimeType,
    sizeBytes: attachment.sizeBytes,
    dataUrl: attachment.dataUrl
  }));
}

export function formatImageAttachmentSize(sizeBytes: number) {
  if (sizeBytes >= 1024 * 1024) {
    return `${(sizeBytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  return `${Math.max(1, Math.round(sizeBytes / 1024))} KB`;
}

function validateImageAttachmentFile(
  file: File,
  existingCount: number
): WorkstationError | null {
  if (existingCount >= maxImageAttachmentCount) {
    return createImageUploadError({
      type: "image_upload_limit_reached",
      userMessage: `Attach up to ${maxImageAttachmentCount} image references per session.`,
      suggestedAction:
        "Remove an existing image reference before attaching another one."
    });
  }

  if (!isSupportedImageMimeType(file.type)) {
    return createImageUploadError({
      type: "unsupported_image_reference",
      userMessage:
        "Only PNG, JPEG, WebP, or GIF image references can be attached.",
      debugMessage: file.type || "unknown file type",
      suggestedAction:
        "Choose a supported image file before sending the multimodal request."
    });
  }

  if (file.size <= 0) {
    return createImageUploadError({
      type: "empty_image_reference",
      userMessage: "The selected image reference is empty.",
      suggestedAction: "Choose a non-empty image file."
    });
  }

  if (file.size > maxImageAttachmentBytes) {
    return createImageUploadError({
      type: "image_reference_too_large",
      userMessage: `Image references must be ${formatImageAttachmentSize(
        maxImageAttachmentBytes
      )} or smaller.`,
      debugMessage: `${file.name || "image"} is ${file.size} bytes.`,
      suggestedAction:
        "Compress the image or choose a smaller reference before uploading."
    });
  }

  return null;
}

function createImageUploadError({
  debugMessage = null,
  suggestedAction,
  type,
  userMessage
}: {
  debugMessage?: string | null;
  suggestedAction: string;
  type: string;
  userMessage: string;
}) {
  return createWorkstationError({
    type,
    category: "multimodal",
    subsystem: "image_upload",
    userMessage,
    debugMessage,
    recoverable: true,
    suggestedAction,
    retryLabel: "Attach image again",
    resetLabel: "Remove image references"
  });
}

async function readFileAsDataUrl(file: File) {
  if (typeof file.arrayBuffer !== "function") {
    return readFileWithFileReader(file);
  }

  const buffer = await file.arrayBuffer();
  return `data:${file.type};base64,${arrayBufferToBase64(buffer)}`;
}

function readFileWithFileReader(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    if (typeof FileReader === "undefined") {
      reject(new Error("FileReader is unavailable in this environment."));
      return;
    }

    const reader = new FileReader();
    reader.onerror = () => reject(reader.error ?? new Error("File read failed."));
    reader.onload = () => {
      if (typeof reader.result === "string") {
        resolve(reader.result);
        return;
      }

      reject(new Error("FileReader did not return a data URL."));
    };
    reader.readAsDataURL(file);
  });
}

function arrayBufferToBase64(buffer: ArrayBuffer) {
  const bytes = new Uint8Array(buffer);
  let binary = "";

  for (let offset = 0; offset < bytes.length; offset += 0x8000) {
    const chunk = bytes.subarray(offset, offset + 0x8000);
    binary += String.fromCharCode(...chunk);
  }

  return btoa(binary);
}

function isImageAttachmentSummary(value: unknown): value is ImageAttachmentSummary {
  if (!isRecord(value)) {
    return false;
  }

  return (
    value.kind === "image" &&
    typeof value.id === "string" &&
    typeof value.name === "string" &&
    typeof value.mimeType === "string" &&
    isSupportedImageMimeType(value.mimeType) &&
    typeof value.sizeBytes === "number" &&
    Number.isFinite(value.sizeBytes) &&
    value.sizeBytes > 0 &&
    value.sizeBytes <= maxImageAttachmentBytes &&
    typeof value.dataUrl === "string" &&
    value.dataUrl.startsWith(`data:${value.mimeType};base64,`) &&
    typeof value.createdAt === "string"
  );
}

function formatAttachmentNames(attachments: ImageAttachmentSummary[]) {
  const names = attachments.map((attachment) => attachment.name);
  if (names.length <= 2) {
    return names.join(" and ");
  }

  return `${names.slice(0, 2).join(", ")} and ${names.length - 2} more`;
}

function pluralize(count: number, singular: string, plural: string) {
  return count === 1 ? singular : plural;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isSupportedImageMimeType(value: unknown): value is SupportedImageMimeType {
  return (
    typeof value === "string" &&
    supportedImageMimeTypes.includes(value as SupportedImageMimeType)
  );
}
