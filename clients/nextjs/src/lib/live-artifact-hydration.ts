import type {
  ArtifactAction,
  ArtifactCritique,
  ArtifactCritiqueDimension,
  ArtifactSummary,
  AssistantWorkspaceSnapshot,
  PreviewSummary,
  PreviewTargetId
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
  critique?: ArtifactCritique;
  domain?: string | null;
  id?: string;
  isDefault?: boolean;
  language?: string;
  origin?: "answer" | "code_fence" | "structured";
  previewEligible?: boolean;
  previewTarget?: string | null;
  rendererId?: string | null;
  qualityRank?: number | null;
  qualityScore?: number | null;
  isRecommended?: boolean;
  refinementReason?: string | null;
  runtime?: string | null;
  sourceOrder?: number;
  status?: string;
  summary?: string;
  title?: string;
  type?: ArtifactSummary["type"];
};

type CreativeRuntimeKind = "p5" | "three" | "glsl" | "hydra";

type ArtifactInference = {
  content: string;
  domain: string | null;
  critique: ArtifactCritique | null;
  id: string;
  isDefault: boolean;
  language: string;
  previewEligible: boolean;
  previewKind: CreativeRuntimeKind | null;
  previewTarget: PreviewTargetId | "";
  rendererId: string | null;
  qualityRank: number | null;
  qualityScore: number | null;
  isRecommended: boolean;
  refinementReason: string | null;
  sourceOrder: number;
  status: string;
  summary: string | null;
  title: string;
  type: ArtifactSummary["type"];
};

const liveGeneratedArtifactId = "live-generated-artifact";
const liveResponseArtifactId = "live-response-artifact";
const previewRendererLabels: Record<CreativeRuntimeKind, string> = {
  glsl: "GLSL",
  hydra: "Hydra",
  p5: "p5.js",
  three: "Three.js"
};
const previewRendererIds: Record<CreativeRuntimeKind, string> = {
  glsl: "surface.glsl",
  hydra: "surface.hydra",
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
  const structuredSources = readStructuredArtifactSources(event.payload);
  const sources =
    structuredSources.length > 0 ? structuredSources : sourcesFromAnswer(answer);
  if (
    options.skipPlainTextArtifact &&
    sources.length > 0 &&
    sources.every((source) => source.origin === "answer")
  ) {
    return {
      activeArtifactId: snapshot.artifacts[0]?.id ?? "",
      artifact: null,
      previewArtifactId: "",
      previewAvailable: snapshot.preview.available,
      snapshot
    };
  }

  return hydrateWorkspaceFromSources(snapshot, sources);
}

export function hydrateWorkspaceFromArtifactExtractedEvent(
  snapshot: AssistantWorkspaceSnapshot,
  event: AssistantStreamEvent
): LiveArtifactHydrationResult {
  if (event.event_type !== "artifact_extracted") {
    return {
      activeArtifactId: snapshot.artifacts[0]?.id ?? "",
      artifact: null,
      previewArtifactId: "",
      previewAvailable: snapshot.preview.available,
      snapshot
    };
  }

  const sources = readStructuredArtifactSources(event.payload);
  return hydrateWorkspaceFromSources(snapshot, sources, {
    artifactHydrationSource: "graph-owned artifact extraction",
    previewTrigger: "Artifact extraction"
  });
}

function hydrateWorkspaceFromSources(
  snapshot: AssistantWorkspaceSnapshot,
  sources: GeneratedArtifactSource[],
  options: {
    artifactHydrationSource?: string;
    previewTrigger?: string;
  } = {}
): LiveArtifactHydrationResult {
  if (sources.length === 0) {
    return {
      activeArtifactId: snapshot.artifacts[0]?.id ?? "",
      artifact: null,
      previewArtifactId: "",
      previewAvailable: false,
      snapshot: {
        ...snapshot,
        preview: buildUnavailablePreviewSummary({
          artifact: null,
          basePreview: snapshot.preview,
          trigger: options.previewTrigger
        })
      }
    };
  }

  const inferredArtifacts = inferGeneratedArtifacts(sources);
  const artifactSummaries = inferredArtifacts.map((inferred) =>
    buildArtifactSummary(inferred, options.artifactHydrationSource)
  );
  const activeInference =
    inferredArtifacts.find((inferred) => inferred.isDefault) ??
    inferredArtifacts.find((inferred) => inferred.previewKind) ??
    inferredArtifacts[0];
  const activeArtifact =
    artifactSummaries.find((artifact) => artifact.id === activeInference.id) ??
    artifactSummaries[0];
  const previewInference =
    activeInference.previewKind || activeInference.previewEligible
      ? activeInference
      : inferredArtifacts.find(
          (inferred) => inferred.previewKind || inferred.previewEligible
        ) ?? null;
  const previewArtifact = previewInference
    ? artifactSummaries.find((artifact) => artifact.id === previewInference.id) ?? null
    : null;
  const artifacts = upsertLiveArtifacts(snapshot.artifacts, artifactSummaries);
  const preview = previewInference && previewArtifact
    ? buildPreviewableSummary({
        artifact: previewArtifact,
        basePreview: snapshot.preview,
        inferred: previewInference,
        trigger: options.previewTrigger
      })
    : buildUnavailablePreviewSummary({
        artifact: activeArtifact,
        basePreview: snapshot.preview,
        trigger: options.previewTrigger
      });
  const code =
    activeArtifact.type === "code"
      ? {
          title: activeArtifact.title,
          language: activeArtifact.language,
          status: activeArtifact.status,
          excerpt: splitContentLines(activeInference.content)
        }
      : snapshot.code;

  return {
    activeArtifactId: activeArtifact.id,
    artifact: activeArtifact,
    previewArtifactId: previewInference && previewArtifact ? previewArtifact.id : "",
    previewAvailable: previewInference !== null,
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
      critique: readArtifactCritique(artifact.critique),
      domain: readString(artifact.domain) ?? null,
      id: readString(artifact.id) ?? undefined,
      isDefault:
        readBoolean(artifact.is_default) ??
        readBoolean(artifact.isDefault) ??
        readBoolean(artifact.default) ??
        undefined,
      language:
        readString(artifact.source_language) ??
        readString(artifact.language) ??
        readString(artifact.lang) ??
        readString(artifact.mime_type) ??
        undefined,
      previewEligible:
        readBoolean(artifact.preview_eligible) ??
        readBoolean(artifact.previewEligible) ??
        undefined,
      previewTarget:
        readString(artifact.preview_target) ??
        readString(artifact.previewTarget) ??
        readString(artifact.target_id) ??
        readString(artifact.targetId) ??
        null,
      rendererId:
        readString(artifact.renderer_id) ??
        readString(artifact.rendererId) ??
        null,
      qualityRank:
        readNumber(artifact.quality_rank) ??
        readNumber(artifact.qualityRank) ??
        null,
      qualityScore:
        readNumber(artifact.quality_score) ??
        readNumber(artifact.qualityScore) ??
        null,
      isRecommended:
        readBoolean(artifact.is_recommended) ??
        readBoolean(artifact.isRecommended) ??
        undefined,
      refinementReason:
        readString(artifact.refinement_reason) ??
        readString(artifact.refinementReason) ??
        null,
      runtime: readString(artifact.runtime) ?? null,
      sourceOrder:
        readNumber(artifact.source_order) ??
        readNumber(artifact.sourceOrder) ??
        undefined,
      status: readString(artifact.status) ?? undefined,
      summary: readString(artifact.summary) ?? undefined,
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

function sourcesFromAnswer(answer: string | null): GeneratedArtifactSource[] {
  if (!answer?.trim()) {
    return [];
  }

  const codeBlocks = parseMarkdownCodeBlocks(answer);
  if (codeBlocks.length > 0) {
    return codeBlocks;
  }

  return [
    {
      content: answer,
      language: "markdown",
      origin: "answer",
      title: "assistant-response.md",
      type: "export"
    }
  ];
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

function inferGeneratedArtifacts(
  sources: GeneratedArtifactSource[]
): ArtifactInference[] {
  const inferredArtifacts = sources.map((source, index) =>
    inferGeneratedArtifact(source, {
      sourceOrder: source.sourceOrder ?? index + 1,
      totalSources: sources.length
    })
  );
  const defaultIndex = inferredArtifacts.findIndex((artifact) => artifact.isDefault);
  const fallbackDefaultIndex =
    inferredArtifacts.findIndex((artifact) => artifact.previewKind) >= 0
      ? inferredArtifacts.findIndex((artifact) => artifact.previewKind)
      : 0;
  const defaultArtifactIndex =
    defaultIndex >= 0 ? defaultIndex : fallbackDefaultIndex;

  return ensureUniqueInferredArtifacts(
    inferredArtifacts.map((artifact, index) => ({
      ...artifact,
      isDefault: index === defaultArtifactIndex
    }))
  );
}

function inferGeneratedArtifact(
  source: GeneratedArtifactSource,
  {
    sourceOrder,
    totalSources
  }: {
    sourceOrder: number;
    totalSources: number;
  }
): ArtifactInference {
  const normalizedLanguage = normalizeLanguageToken(source.language ?? "");
  const type = source.type ?? (normalizedLanguage === "markdown" ? "export" : "code");
  const runtimeKind = normalizeRuntimeKind(source.runtime ?? source.rendererId ?? "");
  const previewKind =
    type === "code"
      ? runtimeKind ??
        inferRuntimeKind(source.content, normalizedLanguage, source.title)
      : null;
  const title =
    sanitizeFileName(source.title ?? "") ||
    defaultArtifactTitle({
      content: source.content,
      language: normalizedLanguage,
      previewKind,
      sourceOrder: totalSources > 1 ? sourceOrder : undefined,
      type
    });
  const language = formatLanguageLabel(normalizedLanguage, previewKind, title);
  const previewTarget = normalizePreviewTarget(source.previewTarget);
  const previewEligible =
    source.previewEligible ??
    (previewKind !== null || previewTarget === "browser_sandbox");
  const fallbackId =
    totalSources > 1
      ? sanitizeArtifactId(title) || `${liveGeneratedArtifactId}-${sourceOrder}`
      : type === "code"
        ? liveGeneratedArtifactId
        : liveResponseArtifactId;

  return {
    content: trimCodeBlock(source.content),
    critique: source.critique ?? null,
    domain: source.domain ?? null,
    id:
      sanitizeArtifactId(source.id ?? "") ||
      fallbackId,
    isDefault: source.isDefault ?? false,
    language,
    previewEligible,
    previewKind,
    previewTarget,
    qualityRank: source.qualityRank ?? source.critique?.rank ?? null,
    qualityScore: source.qualityScore ?? source.critique?.overallScore ?? null,
    isRecommended: source.isRecommended ?? source.critique?.recommended ?? false,
    refinementReason:
      source.refinementReason ?? source.critique?.refinementGuidance ?? null,
    rendererId: source.rendererId ?? (previewKind ? previewRendererIds[previewKind] : null),
    sourceOrder,
    status: source.status ?? "Generated",
    summary: source.summary ?? null,
    title,
    type
  };
}

function ensureUniqueInferredArtifacts(
  artifacts: ArtifactInference[]
): ArtifactInference[] {
  const usedIds = new Set<string>();

  return artifacts.map((artifact) => {
    let nextId = artifact.id;
    let suffix = 2;
    while (usedIds.has(nextId)) {
      nextId =
        sanitizeArtifactId(`${artifact.id}-${suffix}`) ||
        `${liveGeneratedArtifactId}-${artifact.sourceOrder}-${suffix}`;
      suffix += 1;
    }
    usedIds.add(nextId);
    return nextId === artifact.id ? artifact : { ...artifact, id: nextId };
  });
}

function buildArtifactSummary(
  inferred: ArtifactInference,
  hydrationSource = "latest live generation output"
): ArtifactSummary {
  const actions: ArtifactAction[] =
    inferred.type === "code"
      ? inferred.previewKind || inferred.previewEligible
        ? ["Open", "Preview", "Copy", "Download"]
        : ["Open", "Copy", "Download"]
      : ["Open", "Copy", "Download"];
  const runtimeSummary = inferred.previewKind
    ? `${previewRendererLabels[inferred.previewKind]} runtime signals matched from the generated artifact.`
    : inferred.previewEligible
      ? "Preview target metadata is available, but no supported creative runtime matched this output."
    : "No supported p5.js, Three.js, GLSL, or Hydra preview runtime matched this output.";
  const summary =
    inferred.summary ??
    (inferred.type === "code"
      ? `Hydrated from ${hydrationSource}. ${runtimeSummary}`
      : `Hydrated from ${hydrationSource} as a readable response artifact.`);

  return {
    id: inferred.id,
    title: inferred.title,
    type: inferred.type,
    language: inferred.language,
    status: inferred.status,
    summary,
    content: inferred.content,
    critique: inferred.critique ?? undefined,
    domain: inferred.domain,
    isDefault: inferred.isDefault,
    isRecommended: inferred.isRecommended,
    previewEligible: inferred.previewEligible,
    previewTarget: inferred.previewTarget,
    qualityRank: inferred.qualityRank,
    qualityScore: inferred.qualityScore,
    refinementReason: inferred.refinementReason,
    rendererId: inferred.rendererId,
    runtime: inferred.previewKind,
    sourceOrder: inferred.sourceOrder,
    actions
  };
}

function upsertLiveArtifacts(
  artifacts: ArtifactSummary[],
  hydratedArtifacts: ArtifactSummary[]
): ArtifactSummary[] {
  const hydratedIds = new Set(hydratedArtifacts.map((artifact) => artifact.id));
  const nextArtifacts = artifacts.filter((currentArtifact) => {
    if (hydratedIds.has(currentArtifact.id)) {
      return false;
    }

    return !(
      currentArtifact.id === liveGeneratedArtifactId ||
      currentArtifact.id === liveResponseArtifactId
    );
  });

  return [...hydratedArtifacts, ...nextArtifacts];
}

function buildPreviewableSummary({
  artifact,
  basePreview,
  inferred,
  trigger = "Final generation output"
}: {
  artifact: ArtifactSummary;
  basePreview: PreviewSummary;
  inferred: ArtifactInference;
  trigger?: string;
}): PreviewSummary {
  const rendererLabel = inferred.previewKind
    ? previewRendererLabels[inferred.previewKind]
    : "Browser";
  const rendererId =
    inferred.rendererId ??
    (inferred.previewKind ? previewRendererIds[inferred.previewKind] : "");
  const targetId = inferred.previewTarget || "browser_sandbox";

  return {
    ...basePreview,
    active: false,
    artifactName: artifact.title,
    available: true,
    collapsed: true,
    error: null,
    outputArtifactName: artifact.title,
    renderer: rendererId,
    sourceArtifactId: artifact.id,
    sourceArtifactName: artifact.title,
    state: "ready",
    status: "Ready when opened",
    summary:
      trigger === "Artifact extraction"
        ? `${rendererLabel} preview routing was prepared from graph-owned artifact metadata. Open the shelf to mount the live preview surface.`
        : `${rendererLabel} preview routing was inferred from the latest generated artifact. Open the shelf to mount the live preview surface.`,
    target: `Browser preview / ${rendererLabel}`,
    targetId,
    title: "Preview available",
    trigger
  };
}

function buildUnavailablePreviewSummary({
  artifact,
  basePreview,
  trigger = "Final generation output"
}: {
  artifact: ArtifactSummary | null;
  basePreview: PreviewSummary;
  trigger?: string;
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
    trigger
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

  if (
    normalizedTitle.endsWith(".hydra.ts") ||
    normalizedTitle.endsWith(".hydra.js") ||
    haystack.includes("hydra") ||
    (haystack.includes("osc(") && haystack.includes("out("))
  ) {
    return "hydra";
  }

  return null;
}

function defaultArtifactTitle({
  content,
  language,
  previewKind,
  sourceOrder,
  type
}: {
  content: string;
  language: string;
  previewKind: CreativeRuntimeKind | null;
  sourceOrder?: number;
  type: ArtifactSummary["type"];
}) {
  const orderSuffix = sourceOrder ? `-${sourceOrder}` : "";

  if (type !== "code") {
    return "assistant-response.md";
  }

  if (previewKind === "three") {
    return language === "javascript"
      ? `generated-scene${orderSuffix}.three.js`
      : `generated-scene${orderSuffix}.three.ts`;
  }

  if (previewKind === "p5") {
    return language === "javascript"
      ? `generated-sketch${orderSuffix}.p5.js`
      : `generated-sketch${orderSuffix}.p5.ts`;
  }

  if (previewKind === "glsl") {
    return `generated-shader${orderSuffix}.frag`;
  }

  if (previewKind === "hydra") {
    return language === "typescript"
      ? `generated-patch${orderSuffix}.hydra.ts`
      : `generated-patch${orderSuffix}.hydra.js`;
  }

  if (language === "javascript") {
    return `generated-artifact${orderSuffix}.js`;
  }

  if (language === "typescript" || content.includes("export ")) {
    return `generated-artifact${orderSuffix}.ts`;
  }

  if (language === "json") {
    return `generated-artifact${orderSuffix}.json`;
  }

  if (language === "html") {
    return `generated-artifact${orderSuffix}.html`;
  }

  if (language === "css") {
    return `generated-artifact${orderSuffix}.css`;
  }

  if (language === "python") {
    return `generated-artifact${orderSuffix}.py`;
  }

  return `generated-artifact${orderSuffix}.txt`;
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

  if (previewKind === "hydra") {
    return language === "typescript" ? "TypeScript + Hydra" : "JavaScript + Hydra";
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

function normalizeRuntimeKind(value: string): CreativeRuntimeKind | null {
  const normalized = value.trim().toLowerCase();

  switch (normalized) {
    case "surface.glsl":
    case "glsl":
      return "glsl";
    case "surface.hydra":
    case "hydra":
      return "hydra";
    case "surface.p5":
    case "p5":
    case "p5.js":
      return "p5";
    case "surface.three":
    case "three":
    case "three.js":
      return "three";
    default:
      return null;
  }
}

function normalizePreviewTarget(
  value: string | null | undefined
): PreviewTargetId | "" {
  switch (value) {
    case "audio_asset":
    case "browser_sandbox":
    case "image_asset":
    case "json_panel":
    case "text_panel":
    case "video_asset":
      return value;
    default:
      return "";
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

function readArtifactCritique(value: unknown): ArtifactCritique | undefined {
  const record = isRecord(value) ? value : null;
  if (!record) {
    return undefined;
  }
  const artifactId = readString(record.artifact_id) ?? readString(record.artifactId);
  const artifactTitle =
    readString(record.artifact_title) ?? readString(record.artifactTitle);
  const overallScore =
    readNumber(record.overall_score) ?? readNumber(record.overallScore);
  const rank = readNumber(record.rank);
  const sourceOrder =
    readNumber(record.source_order) ?? readNumber(record.sourceOrder);

  if (!artifactId || !artifactTitle || overallScore === null || rank === null || sourceOrder === null) {
    return undefined;
  }

  return {
    artifactId,
    artifactTitle,
    sourceOrder,
    overallScore,
    rank,
    passed: readBoolean(record.passed) ?? false,
    recommended: readBoolean(record.recommended) ?? false,
    promptAlignment: readCritiqueDimension(record.prompt_alignment),
    creativeQuality: readCritiqueDimension(record.creative_quality),
    runtimeSuitability: readCritiqueDimension(record.runtime_suitability),
    codeQuality: readCritiqueDimension(record.code_quality),
    previewReadiness: readCritiqueDimension(record.preview_readiness),
    domainAppropriateness: readCritiqueDimension(record.domain_appropriateness),
    reasons: readStringList(record.reasons),
    rationale: readString(record.rationale) ?? "Artifact critique completed.",
    refinementGuidance:
      readString(record.refinement_guidance) ??
      readString(record.refinementGuidance) ??
      null
  };
}

function readCritiqueDimension(value: unknown): ArtifactCritiqueDimension {
  const record = isRecord(value) ? value : null;
  return {
    score: readNumber(record?.score) ?? 0,
    rationale: readString(record?.rationale) ?? "No critique rationale available."
  };
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

function readStringList(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readBoolean(value: unknown) {
  return typeof value === "boolean" ? value : null;
}

function readNumber(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
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
