import type {
  ArtifactAction,
  ArtifactSummary,
  AssistantWorkspaceSnapshot,
  PreviewSummary
} from "./assistant-client";
import type { AssistantStreamEvent } from "./assistant-stream";

export type LiveArtifactHydrationResult = {
  activeArtifactId: string;
  artifact: ArtifactSummary | null;
  previewArtifactId: string;
  previewAvailable: boolean;
  snapshot: AssistantWorkspaceSnapshot;
};

export type LiveArtifactHydrationOptions = {
  skipPlainTextArtifact?: boolean;
};

type GeneratedArtifactSource = {
  content: string;
  id?: string;
  language?: string;
  origin?: "answer" | "code_fence" | "structured";
  title?: string;
  type?: ArtifactSummary["type"];
};

type CreativeRuntimeKind = "p5" | "three" | "glsl";

type ArtifactInference = {
  content: string;
  id: string;
  language: string;
  previewKind: CreativeRuntimeKind | null;
  title: string;
  type: ArtifactSummary["type"];
};

const liveGeneratedArtifactId = "live-generated-artifact";
const liveResponseArtifactId = "live-response-artifact";
const previewRendererLabels: Record<CreativeRuntimeKind, string> = {
  glsl: "GLSL",
  p5: "p5.js",
  three: "Three.js"
};
const previewRendererIds: Record<CreativeRuntimeKind, string> = {
  glsl: "surface.glsl",
  p5: "surface.p5",
  three: "surface.three"
};

export function hydrateWorkspaceFromFinalEvent(
  snapshot: AssistantWorkspaceSnapshot,
  event: AssistantStreamEvent,
  options: LiveArtifactHydrationOptions = {}
): LiveArtifactHydrationResult {
  if (event.event_type !== "final") {
    return {
      activeArtifactId: snapshot.artifacts[0]?.id ?? "",
      artifact: null,
      previewArtifactId: "",
      previewAvailable: snapshot.preview.available,
      snapshot
    };
  }

  const answer = readString(event.payload.answer);
  const source = readStructuredArtifactSources(event.payload)[0] ?? sourceFromAnswer(answer);
  if (options.skipPlainTextArtifact && source?.origin === "answer") {
    return {
      activeArtifactId: snapshot.artifacts[0]?.id ?? "",
      artifact: null,
      previewArtifactId: "",
      previewAvailable: snapshot.preview.available,
      snapshot
    };
  }

  if (!source) {
    return {
      activeArtifactId: snapshot.artifacts[0]?.id ?? "",
      artifact: null,
      previewArtifactId: "",
      previewAvailable: false,
      snapshot: {
        ...snapshot,
        preview: buildUnavailablePreviewSummary({
          artifact: null,
          basePreview: snapshot.preview
        })
      }
    };
  }

  const inferred = inferGeneratedArtifact(source);
  const artifact = buildArtifactSummary(inferred);
  const artifacts = upsertLiveArtifact(snapshot.artifacts, artifact);
  const preview = inferred.previewKind
    ? buildPreviewableSummary({
        artifact,
        basePreview: snapshot.preview,
        kind: inferred.previewKind
      })
    : buildUnavailablePreviewSummary({
        artifact,
        basePreview: snapshot.preview
      });
  const code =
    artifact.type === "code"
      ? {
          title: artifact.title,
          language: artifact.language,
          status: artifact.status,
          excerpt: splitContentLines(inferred.content)
        }
      : snapshot.code;

  return {
    activeArtifactId: artifact.id,
    artifact,
    previewArtifactId: inferred.previewKind ? artifact.id : "",
    previewAvailable: inferred.previewKind !== null,
    snapshot: {
      ...snapshot,
      artifacts,
      code,
      preview
    }
  };
}

function readStructuredArtifactSources(
  payload: Record<string, unknown>
): GeneratedArtifactSource[] {
  const rawArtifacts = [
    ...readRecordList(payload.artifacts),
    ...readRecordList(payload.generated_artifacts),
    ...readRecordList(payload.outputs),
    ...readRecordList(payload.artifact ? [payload.artifact] : []),
    ...readRecordList(payload.generated_artifact ? [payload.generated_artifact] : [])
  ];

  const sources: GeneratedArtifactSource[] = [];

  for (const artifact of rawArtifacts) {
    const content =
      readString(artifact.content) ??
      readString(artifact.code) ??
      readString(artifact.source) ??
      readString(artifact.text);
    if (!content?.trim()) {
      continue;
    }

    sources.push({
      content,
      id: readString(artifact.id) ?? undefined,
      language:
        readString(artifact.language) ??
        readString(artifact.lang) ??
        readString(artifact.mime_type) ??
        undefined,
      origin: "structured",
      title:
        readString(artifact.title) ??
        readString(artifact.file_name) ??
        readString(artifact.filename) ??
        readString(artifact.name) ??
        undefined,
      type: readArtifactType(artifact.type)
    });
  }

  return sources;
}

function sourceFromAnswer(answer: string | null): GeneratedArtifactSource | null {
  if (!answer?.trim()) {
    return null;
  }

  const codeBlocks = parseMarkdownCodeBlocks(answer);
  const preferredBlock =
    codeBlocks.find((block) => inferRuntimeKind(block.content, block.language, block.title)) ??
    codeBlocks[0];

  if (preferredBlock) {
    return preferredBlock;
  }

  return {
    content: answer,
    language: "markdown",
    origin: "answer",
    title: "assistant-response.md",
    type: "export"
  };
}

function parseMarkdownCodeBlocks(answer: string): GeneratedArtifactSource[] {
  const blocks: GeneratedArtifactSource[] = [];
  const fencePattern = /```([^\n`]*)\n([\s\S]*?)```/g;
  let match: RegExpExecArray | null;

  while ((match = fencePattern.exec(answer)) !== null) {
    const info = parseFenceInfo(match[1] ?? "");
    const content = trimCodeBlock(match[2] ?? "");
    if (!content.trim()) {
      continue;
    }

    blocks.push({
      content,
      language: info.language,
      origin: "code_fence",
      title: info.title,
      type: "code"
    });
  }

  return blocks;
}

function parseFenceInfo(info: string) {
  const tokens = info.trim().split(/\s+/).filter(Boolean);
  const language = normalizeLanguageToken(tokens[0] ?? "");
  const titleToken = tokens.find((token) => token.includes("."));
  const namedTitle = tokens
    .map((token) => token.match(/^(?:file|filename|name)=(.+)$/i)?.[1])
    .find((value): value is string => Boolean(value));

  return {
    language: language || undefined,
    title: sanitizeFileName(namedTitle ?? titleToken ?? "") || undefined
  };
}

function inferGeneratedArtifact(source: GeneratedArtifactSource): ArtifactInference {
  const normalizedLanguage = normalizeLanguageToken(source.language ?? "");
  const type = source.type ?? (normalizedLanguage === "markdown" ? "export" : "code");
  const previewKind =
    type === "code"
      ? inferRuntimeKind(source.content, normalizedLanguage, source.title)
      : null;
  const title =
    sanitizeFileName(source.title ?? "") ||
    defaultArtifactTitle({
      content: source.content,
      language: normalizedLanguage,
      previewKind,
      type
    });
  const language = formatLanguageLabel(normalizedLanguage, previewKind, title);

  return {
    content: trimCodeBlock(source.content),
    id:
      sanitizeArtifactId(source.id ?? "") ||
      (type === "code" ? liveGeneratedArtifactId : liveResponseArtifactId),
    language,
    previewKind,
    title,
    type
  };
}

function buildArtifactSummary(inferred: ArtifactInference): ArtifactSummary {
  const actions: ArtifactAction[] =
    inferred.type === "code"
      ? inferred.previewKind
        ? ["Open", "Preview", "Copy", "Download"]
        : ["Open", "Copy", "Download"]
      : ["Open", "Copy", "Download"];
  const runtimeSummary = inferred.previewKind
    ? `${previewRendererLabels[inferred.previewKind]} runtime signals matched from the generated artifact.`
    : "No supported p5.js, Three.js, or GLSL preview runtime matched this output.";

  return {
    id: inferred.id,
    title: inferred.title,
    type: inferred.type,
    language: inferred.language,
    status: "Generated",
    summary:
      inferred.type === "code"
        ? `Hydrated from the latest live generation output. ${runtimeSummary}`
        : "Hydrated from the latest live generation output as a readable response artifact.",
    content: inferred.content,
    actions
  };
}

function upsertLiveArtifact(
  artifacts: ArtifactSummary[],
  artifact: ArtifactSummary
): ArtifactSummary[] {
  const nextArtifacts = artifacts.filter((currentArtifact) => {
    if (currentArtifact.id === artifact.id) {
      return false;
    }

    return !(
      currentArtifact.id === liveGeneratedArtifactId ||
      currentArtifact.id === liveResponseArtifactId
    );
  });

  return [artifact, ...nextArtifacts];
}

function buildPreviewableSummary({
  artifact,
  basePreview,
  kind
}: {
  artifact: ArtifactSummary;
  basePreview: PreviewSummary;
  kind: CreativeRuntimeKind;
}): PreviewSummary {
  const rendererLabel = previewRendererLabels[kind];

  return {
    ...basePreview,
    active: false,
    artifactName: artifact.title,
    available: true,
    collapsed: true,
    error: null,
    outputArtifactName: artifact.title,
    renderer: previewRendererIds[kind],
    sourceArtifactId: artifact.id,
    sourceArtifactName: artifact.title,
    state: "ready",
    status: "Ready when opened",
    summary: `${rendererLabel} preview routing was inferred from the latest generated artifact. Open the shelf to mount the controlled runtime surface.`,
    target: `Browser sandbox / ${rendererLabel}`,
    targetId: "browser_sandbox",
    title: "Preview available",
    trigger: "Final generation output"
  };
}

function buildUnavailablePreviewSummary({
  artifact,
  basePreview
}: {
  artifact: ArtifactSummary | null;
  basePreview: PreviewSummary;
}): PreviewSummary {
  const artifactName = artifact?.title ?? "No runnable artifact";

  return {
    ...basePreview,
    active: false,
    artifactName,
    available: false,
    collapsed: true,
    error: null,
    outputArtifactName: "",
    renderer: "",
    sourceArtifactId: artifact?.id ?? "",
    sourceArtifactName: artifact?.title ?? "",
    state: "unavailable",
    status: "Unavailable",
    summary: `${artifactName} is available in the workspace, but it does not match the supported live p5.js, Three.js, or GLSL preview runtimes yet.`,
    target: "",
    targetId: "",
    title: "Preview unavailable",
    trigger: "Final generation output"
  };
}

function inferRuntimeKind(
  content: string,
  language: string | undefined,
  title: string | undefined
): CreativeRuntimeKind | null {
  const haystack = [content, language, title].join(" ").toLowerCase();
  const normalizedTitle = (title ?? "").toLowerCase();

  if (
    normalizedTitle.endsWith(".frag") ||
    normalizedTitle.endsWith(".glsl") ||
    language === "glsl" ||
    haystack.includes("gl_fragcolor") ||
    haystack.includes("fragment shader")
  ) {
    return "glsl";
  }

  if (
    normalizedTitle.endsWith(".three.ts") ||
    normalizedTitle.endsWith(".three.js") ||
    haystack.includes("three") ||
    haystack.includes("webglrenderer") ||
    haystack.includes("perspectivecamera") ||
    haystack.includes("@react-three/fiber")
  ) {
    return "three";
  }

  if (
    normalizedTitle.endsWith(".p5.ts") ||
    normalizedTitle.endsWith(".p5.js") ||
    haystack.includes("createcanvas") ||
    (haystack.includes("function setup") && haystack.includes("function draw")) ||
    haystack.includes("new p5")
  ) {
    return "p5";
  }

  return null;
}

function defaultArtifactTitle({
  content,
  language,
  previewKind,
  type
}: {
  content: string;
  language: string;
  previewKind: CreativeRuntimeKind | null;
  type: ArtifactSummary["type"];
}) {
  if (type !== "code") {
    return "assistant-response.md";
  }

  if (previewKind === "three") {
    return language === "javascript"
      ? "generated-scene.three.js"
      : "generated-scene.three.ts";
  }

  if (previewKind === "p5") {
    return language === "javascript"
      ? "generated-sketch.p5.js"
      : "generated-sketch.p5.ts";
  }

  if (previewKind === "glsl") {
    return "generated-shader.frag";
  }

  if (language === "javascript") {
    return "generated-artifact.js";
  }

  if (language === "typescript" || content.includes("export ")) {
    return "generated-artifact.ts";
  }

  if (language === "json") {
    return "generated-artifact.json";
  }

  if (language === "html") {
    return "generated-artifact.html";
  }

  if (language === "css") {
    return "generated-artifact.css";
  }

  if (language === "python") {
    return "generated-artifact.py";
  }

  return "generated-artifact.txt";
}

function formatLanguageLabel(
  language: string,
  previewKind: CreativeRuntimeKind | null,
  title: string
) {
  if (previewKind === "three") {
    return language === "javascript" ? "JavaScript + Three.js" : "TypeScript + Three.js";
  }

  if (previewKind === "p5") {
    return language === "javascript" ? "JavaScript + p5.js" : "TypeScript + p5.js";
  }

  if (previewKind === "glsl") {
    return "GLSL";
  }

  switch (language) {
    case "css":
      return "CSS";
    case "glsl":
      return "GLSL";
    case "html":
      return "HTML";
    case "javascript":
      return "JavaScript";
    case "json":
      return "JSON";
    case "markdown":
      return "Markdown";
    case "python":
      return "Python";
    case "typescript":
      return title.endsWith(".tsx") ? "TypeScript + React" : "TypeScript";
    default:
      return language ? sentenceCase(language) : "Text";
  }
}

function normalizeLanguageToken(value: string) {
  const normalized = value.trim().toLowerCase();

  switch (normalized) {
    case "frag":
    case "fragment":
    case "fragment-shader":
      return "glsl";
    case "js":
    case "jsx":
    case "mjs":
      return "javascript";
    case "md":
    case "markdown":
      return "markdown";
    case "py":
      return "python";
    case "ts":
    case "tsx":
      return "typescript";
    default:
      return normalized;
  }
}

function splitContentLines(content: string) {
  const lines = content.replace(/\r\n/g, "\n").split("\n");
  return lines.length > 0 ? lines : [""];
}

function trimCodeBlock(content: string) {
  return content.replace(/^\s*\n/, "").replace(/\n\s*$/, "");
}

function readArtifactType(value: unknown): ArtifactSummary["type"] | undefined {
  return value === "code" || value === "preview" || value === "export"
    ? value
    : undefined;
}

function readRecordList(value: unknown): Record<string, unknown>[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.filter(
    (entry): entry is Record<string, unknown> =>
      typeof entry === "object" && entry !== null && !Array.isArray(entry)
  );
}

function readString(value: unknown) {
  return typeof value === "string" ? value : null;
}

function sanitizeFileName(value: string) {
  return value
    .trim()
    .replace(/^["'`]+|["'`]+$/g, "")
    .replace(/[\\/:*?"<>|]+/g, "-");
}

function sanitizeArtifactId(value: string) {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_.-]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function sentenceCase(value: string) {
  return value.replace(/[_-]+/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}
