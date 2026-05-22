import type {
  ArtifactSummary,
  AssistantWorkspaceSnapshot
} from "./assistant-client";
import { buildPreviewRendererRoute } from "./preview-renderers";

export type ArtifactDocument = {
  artifactId: string;
  fileName: string;
  typeLabel: string;
  languageLabel: string;
  mimeType: string;
  content: string;
  lineCount: number;
  status: string;
  summary: string;
};

export type HighlightTokenKind =
  | "plain"
  | "comment"
  | "heading"
  | "inline_code"
  | "keyword"
  | "number"
  | "property"
  | "string"
  | "symbol"
  | "function";

export type HighlightToken = {
  kind: HighlightTokenKind;
  text: string;
};

export type HighlightedLine = {
  lineNumber: number;
  tokens: HighlightToken[];
};

type ClipboardLike = {
  writeText: (text: string) => Promise<void>;
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

const tsKeywords = new Set([
  "await",
  "const",
  "export",
  "false",
  "from",
  "function",
  "if",
  "import",
  "let",
  "null",
  "return",
  "true",
  "undefined"
]);

export function buildArtifactDocument(
  snapshot: AssistantWorkspaceSnapshot,
  artifact: ArtifactSummary
): ArtifactDocument {
  if (artifact.type === "code") {
    const content = snapshot.code.excerpt.join("\n");

    return {
      artifactId: artifact.id,
      content,
      fileName: artifact.title,
      languageLabel: snapshot.code.language,
      lineCount: content.split("\n").length,
      mimeType: resolveMimeType(artifact.title),
      status: artifact.status,
      summary: artifact.summary,
      typeLabel: "Source code"
    };
  }

  if (artifact.type === "preview") {
    const previewRoute = buildPreviewRendererRoute({
      artifacts: snapshot.artifacts,
      preview: snapshot.preview,
      previewArtifactId: artifact.id
    });
    const content = JSON.stringify(
      {
        artifactId: artifact.id,
        artifactTitle: artifact.title,
        preview: {
          renderer: snapshot.preview.renderer,
          status: snapshot.preview.status,
          target: snapshot.preview.target,
          targetId: snapshot.preview.targetId,
          trigger: snapshot.preview.trigger,
          version: snapshot.preview.version
        },
        route: {
          selectedArtifactId: previewRoute.selectedArtifactId,
          selectedArtifactName: previewRoute.selectedArtifactName,
          sourceArtifactId: previewRoute.sourceArtifactId,
          sourceArtifactName: previewRoute.sourceArtifactName,
          rendererId: previewRoute.rendererId,
          rendererLabel: previewRoute.rendererLabel,
          supportState: previewRoute.supportState,
          supportLabel: previewRoute.supportLabel,
          supportReason: previewRoute.supportReason,
          surfaceKind: previewRoute.surfaceKind,
          surfaceTitle: previewRoute.surfaceTitle,
          targetId: previewRoute.targetId,
          targetLabel: previewRoute.targetLabel,
          notes: previewRoute.notes
        },
        session: {
          projectId: snapshot.session.projectId,
          sessionId: snapshot.session.sessionId,
          userId: snapshot.session.userId
        }
      },
      null,
      2
    );

    return {
      artifactId: artifact.id,
      content,
      fileName: artifact.title,
      languageLabel: artifact.language,
      lineCount: content.split("\n").length,
      mimeType: resolveMimeType(artifact.title),
      status: artifact.status,
      summary: artifact.summary,
      typeLabel: "Preview manifest"
    };
  }

  const content = [
    `# ${artifact.title}`,
    "",
    "## Workspace",
    `- Session: ${snapshot.workspace.name}`,
    `- Focus: ${snapshot.workspace.focus}`,
    `- Selected artifact: ${artifact.title}`,
    "",
    "## Constraints",
    ...snapshot.retrieval.sources.map((source) => `- ${source.detail}`),
    "",
    "## Notes",
    artifact.summary
  ].join("\n");

  return {
    artifactId: artifact.id,
    content,
    fileName: artifact.title,
    languageLabel: artifact.language,
    lineCount: content.split("\n").length,
    mimeType: resolveMimeType(artifact.title),
    status: artifact.status,
    summary: artifact.summary,
    typeLabel: "Markdown export"
  };
}

export function highlightArtifactDocument(
  document: ArtifactDocument
): HighlightedLine[] {
  return document.content.split("\n").map((line, index) => ({
    lineNumber: index + 1,
    tokens: tokenizeArtifactLine(line, document.fileName)
  }));
}

export async function copyArtifactDocument(
  document: ArtifactDocument,
  clipboard: ClipboardLike | undefined = globalThis.navigator?.clipboard
): Promise<boolean> {
  if (!clipboard) {
    return false;
  }

  try {
    await clipboard.writeText(document.content);
    return true;
  } catch {
    return false;
  }
}

export function downloadArtifactDocument(
  document: ArtifactDocument,
  api: DownloadApi | undefined = resolveDownloadApi()
): boolean {
  if (!api) {
    return false;
  }

  try {
    const blob = new Blob([document.content], { type: document.mimeType });
    const href = api.createObjectURL(blob);
    const anchor = api.createAnchor();
    anchor.download = document.fileName;
    anchor.href = href;
    anchor.click();
    api.revokeObjectURL(href);
    return true;
  } catch {
    return false;
  }
}

export function formatArtifactActionLabel(action: string): string {
  switch (action) {
    case "Open":
      return "Open in Code";
    case "Preview":
      return "Open Preview";
    case "Copy":
      return "Copy Text";
    case "Download":
      return "Download File";
    case "Export":
      return "Export File";
    default:
      return action;
  }
}

function tokenizeArtifactLine(
  line: string,
  fileName: string
): HighlightToken[] {
  if (fileName.endsWith(".md")) {
    return tokenizeMarkdownLine(line);
  }

  if (fileName.endsWith(".json")) {
    return tokenizeJsonLine(line);
  }

  return tokenizeScriptLine(line);
}

function tokenizeMarkdownLine(line: string): HighlightToken[] {
  if (!line) {
    return [{ kind: "plain", text: "" }];
  }

  const trimmed = line.trimStart();
  if (trimmed.startsWith("#")) {
    return [{ kind: "heading", text: line }];
  }

  const tokens: HighlightToken[] = [];
  const pattern = /(`[^`]+`|^\s*-\s)/g;
  let lastIndex = 0;

  for (const match of line.matchAll(pattern)) {
    const index = match.index ?? 0;
    if (index > lastIndex) {
      tokens.push({ kind: "plain", text: line.slice(lastIndex, index) });
    }

    const text = match[0];
    tokens.push({
      kind: text.startsWith("`") ? "inline_code" : "symbol",
      text
    });
    lastIndex = index + text.length;
  }

  if (lastIndex < line.length) {
    tokens.push({ kind: "plain", text: line.slice(lastIndex) });
  }

  return tokens;
}

function tokenizeJsonLine(line: string): HighlightToken[] {
  const pattern =
    /("(?:[^"\\]|\\.)*"\s*:|"(?:[^"\\]|\\.)*"|true|false|null|-?\d+(?:\.\d+)?|[{}[\],:])/g;
  return tokenizeByPattern(line, pattern, classifyJsonToken);
}

function tokenizeScriptLine(line: string): HighlightToken[] {
  const pattern =
    /(\/\/.*$|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|\b\d+(?:\.\d+)?\b|\b[A-Za-z_$][\w$]*(?=\()|\b(?:await|const|export|false|from|function|if|import|let|null|return|true|undefined)\b|[()[\]{}.,;:=])/g;
  return tokenizeByPattern(line, pattern, classifyScriptToken);
}

function tokenizeByPattern(
  line: string,
  pattern: RegExp,
  classifyToken: (token: string) => HighlightTokenKind
): HighlightToken[] {
  if (!line) {
    return [{ kind: "plain", text: "" }];
  }

  const tokens: HighlightToken[] = [];
  let lastIndex = 0;

  for (const match of line.matchAll(pattern)) {
    const index = match.index ?? 0;
    const text = match[0];

    if (index > lastIndex) {
      tokens.push({ kind: "plain", text: line.slice(lastIndex, index) });
    }

    tokens.push({ kind: classifyToken(text), text });
    lastIndex = index + text.length;
  }

  if (lastIndex < line.length) {
    tokens.push({ kind: "plain", text: line.slice(lastIndex) });
  }

  return tokens;
}

function classifyJsonToken(token: string): HighlightTokenKind {
  if (token.startsWith('"')) {
    return token.trimEnd().endsWith(":") ? "property" : "string";
  }

  if (/^-?\d/.test(token)) {
    return "number";
  }

  if (token === "true" || token === "false" || token === "null") {
    return "keyword";
  }

  return "symbol";
}

function classifyScriptToken(token: string): HighlightTokenKind {
  if (token.startsWith("//")) {
    return "comment";
  }

  if (token.startsWith('"') || token.startsWith("'")) {
    return "string";
  }

  if (/^\d/.test(token)) {
    return "number";
  }

  if (tsKeywords.has(token)) {
    return "keyword";
  }

  if (/^[A-Za-z_$]/.test(token) && !tsKeywords.has(token)) {
    return "function";
  }

  return "symbol";
}

function resolveMimeType(fileName: string): string {
  if (fileName.endsWith(".ts")) {
    return "text/typescript;charset=utf-8";
  }

  if (fileName.endsWith(".json")) {
    return "application/json;charset=utf-8";
  }

  if (fileName.endsWith(".md")) {
    return "text/markdown;charset=utf-8";
  }

  return "text/plain;charset=utf-8";
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
    createAnchor() {
      const anchor = document.createElement("a");
      anchor.rel = "noreferrer";
      return anchor;
    },
    createObjectURL: (blob) => URL.createObjectURL(blob),
    revokeObjectURL: (url) => URL.revokeObjectURL(url)
  };
}
