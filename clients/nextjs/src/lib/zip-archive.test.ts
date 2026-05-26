import { describe, expect, it, vi } from "vitest";
import { buildZipArchive, downloadZipArchive } from "./zip-archive";

const textDecoder = new TextDecoder();

describe("zip archive helpers", () => {
  it("builds a stored zip archive with readable file entries", () => {
    const bytes = buildZipArchive([
      {
        path: "manifest.json",
        bytes: new TextEncoder().encode('{"schemaVersion":1}')
      },
      {
        path: "artifacts/demo.ts",
        bytes: new TextEncoder().encode("export const demo = true;\n")
      }
    ]);
    const entries = readStoredZipEntries(bytes);

    expect(entries.get("manifest.json")).toBe('{"schemaVersion":1}');
    expect(entries.get("artifacts/demo.ts")).toBe("export const demo = true;\n");
  });

  it("downloads zip archives through the browser download boundary", () => {
    const click = vi.fn();
    const anchor = {
      click,
      download: "",
      href: ""
    };
    const downloadApi = {
      createAnchor: vi.fn(() => anchor),
      createObjectURL: vi.fn(() => "blob:bundle"),
      revokeObjectURL: vi.fn()
    };

    expect(
      downloadZipArchive("workspace-bundle.zip", new Uint8Array([1, 2, 3]), downloadApi)
    ).toBe(true);
    expect(anchor.download).toBe("workspace-bundle.zip");
    expect(anchor.href).toBe("blob:bundle");
    expect(click).toHaveBeenCalledTimes(1);
    expect(downloadApi.revokeObjectURL).toHaveBeenCalledWith("blob:bundle");

    const createObjectUrlCalls = downloadApi.createObjectURL.mock
      .calls as unknown as Array<[Blob]>;
    expect(createObjectUrlCalls).toHaveLength(1);
    expect(createObjectUrlCalls[0][0].type).toBe("application/zip");
  });
});

function readStoredZipEntries(bytes: Uint8Array) {
  const entries = new Map<string, string>();
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let offset = 0;

  while (offset + 4 <= bytes.byteLength) {
    const signature = view.getUint32(offset, true);

    if (signature === 0x04034b50) {
      const fileNameLength = view.getUint16(offset + 26, true);
      const extraFieldLength = view.getUint16(offset + 28, true);
      const compressedSize = view.getUint32(offset + 18, true);
      const fileNameStart = offset + 30;
      const fileNameEnd = fileNameStart + fileNameLength;
      const fileName = textDecoder.decode(bytes.slice(fileNameStart, fileNameEnd));
      const fileStart = fileNameEnd + extraFieldLength;
      const fileEnd = fileStart + compressedSize;

      entries.set(fileName, textDecoder.decode(bytes.slice(fileStart, fileEnd)));
      offset = fileEnd;
      continue;
    }

    if (signature === 0x02014b50 || signature === 0x06054b50) {
      break;
    }

    throw new Error(`Unexpected ZIP signature ${signature.toString(16)}.`);
  }

  return entries;
}
