export type ZipArchiveEntry = {
  path: string;
  bytes: Uint8Array;
};

type DownloadAnchor = {
  download: string;
  href: string;
  click: () => void;
};

type DownloadApi = {
  createAnchor: () => DownloadAnchor;
  createObjectURL: (blob: Blob) => string;
  revokeObjectURL: (url: string) => void;
};

const crcTable = buildCrcTable();

export function buildZipArchive(entries: ZipArchiveEntry[]): Uint8Array {
  const encoder = new TextEncoder();
  const localFiles: Uint8Array[] = [];
  const centralDirectory: Uint8Array[] = [];
  let offset = 0;
  const dosTimestamp = toDosTimestamp(new Date());

  for (const entry of entries) {
    const name = encoder.encode(normalizeZipPath(entry.path));
    const crc = crc32(entry.bytes);
    const localFile = new Uint8Array(30 + name.length + entry.bytes.length);
    const localView = new DataView(localFile.buffer);

    localView.setUint32(0, 0x04034b50, true);
    localView.setUint16(4, 20, true);
    localView.setUint16(6, 0, true);
    localView.setUint16(8, 0, true);
    localView.setUint16(10, dosTimestamp.time, true);
    localView.setUint16(12, dosTimestamp.date, true);
    localView.setUint32(14, crc, true);
    localView.setUint32(18, entry.bytes.byteLength, true);
    localView.setUint32(22, entry.bytes.byteLength, true);
    localView.setUint16(26, name.length, true);
    localView.setUint16(28, 0, true);
    localFile.set(name, 30);
    localFile.set(entry.bytes, 30 + name.length);
    localFiles.push(localFile);

    const centralFile = new Uint8Array(46 + name.length);
    const centralView = new DataView(centralFile.buffer);

    centralView.setUint32(0, 0x02014b50, true);
    centralView.setUint16(4, 20, true);
    centralView.setUint16(6, 20, true);
    centralView.setUint16(8, 0, true);
    centralView.setUint16(10, 0, true);
    centralView.setUint16(12, dosTimestamp.time, true);
    centralView.setUint16(14, dosTimestamp.date, true);
    centralView.setUint32(16, crc, true);
    centralView.setUint32(20, entry.bytes.byteLength, true);
    centralView.setUint32(24, entry.bytes.byteLength, true);
    centralView.setUint16(28, name.length, true);
    centralView.setUint16(30, 0, true);
    centralView.setUint16(32, 0, true);
    centralView.setUint16(34, 0, true);
    centralView.setUint16(36, 0, true);
    centralView.setUint32(38, 0, true);
    centralView.setUint32(42, offset, true);
    centralFile.set(name, 46);
    centralDirectory.push(centralFile);

    offset += localFile.byteLength;
  }

  const centralDirectoryOffset = offset;
  for (const centralFile of centralDirectory) {
    offset += centralFile.byteLength;
  }

  const endRecord = new Uint8Array(22);
  const endView = new DataView(endRecord.buffer);
  const centralDirectorySize = offset - centralDirectoryOffset;

  endView.setUint32(0, 0x06054b50, true);
  endView.setUint16(4, 0, true);
  endView.setUint16(6, 0, true);
  endView.setUint16(8, entries.length, true);
  endView.setUint16(10, entries.length, true);
  endView.setUint32(12, centralDirectorySize, true);
  endView.setUint32(16, centralDirectoryOffset, true);
  endView.setUint16(20, 0, true);

  return concatenateBuffers([...localFiles, ...centralDirectory, endRecord]);
}

export function downloadZipArchive(
  fileName: string,
  bytes: Uint8Array,
  api: DownloadApi | undefined = resolveDownloadApi()
): boolean {
  if (!api) {
    return false;
  }

  try {
    const href = api.createObjectURL(
      new Blob([bytes], { type: "application/zip" })
    );
    const anchor = api.createAnchor();

    anchor.download = fileName;
    anchor.href = href;
    anchor.click();
    api.revokeObjectURL(href);
    return true;
  } catch {
    return false;
  }
}

function normalizeZipPath(path: string) {
  return path.replace(/\\/g, "/").replace(/^\/+/, "");
}

function concatenateBuffers(buffers: Uint8Array[]) {
  const totalBytes = buffers.reduce((sum, buffer) => sum + buffer.byteLength, 0);
  const combined = new Uint8Array(totalBytes);
  let offset = 0;

  for (const buffer of buffers) {
    combined.set(buffer, offset);
    offset += buffer.byteLength;
  }

  return combined;
}

function resolveDownloadApi(): DownloadApi | undefined {
  if (
    typeof document === "undefined" ||
    typeof URL === "undefined" ||
    typeof URL.createObjectURL !== "function" ||
    typeof URL.revokeObjectURL !== "function"
  ) {
    return undefined;
  }

  return {
    createAnchor: () => document.createElement("a"),
    createObjectURL: (blob) => URL.createObjectURL(blob),
    revokeObjectURL: (url) => URL.revokeObjectURL(url)
  };
}

function toDosTimestamp(date: Date) {
  const year = Math.max(date.getUTCFullYear(), 1980);
  const month = date.getUTCMonth() + 1;
  const day = date.getUTCDate();
  const hours = date.getUTCHours();
  const minutes = date.getUTCMinutes();
  const seconds = Math.floor(date.getUTCSeconds() / 2);

  return {
    date: ((year - 1980) << 9) | (month << 5) | day,
    time: (hours << 11) | (minutes << 5) | seconds
  };
}

function crc32(bytes: Uint8Array) {
  let crc = 0xffffffff;

  for (const byte of bytes) {
    crc = (crc >>> 8) ^ crcTable[(crc ^ byte) & 0xff];
  }

  return (crc ^ 0xffffffff) >>> 0;
}

function buildCrcTable() {
  const table = new Uint32Array(256);

  for (let index = 0; index < 256; index += 1) {
    let value = index;

    for (let bit = 0; bit < 8; bit += 1) {
      value =
        (value & 1) === 1 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
    }

    table[index] = value >>> 0;
  }

  return table;
}
