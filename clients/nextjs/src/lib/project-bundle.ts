import type {
  ArtifactSummary,
  AssistantWorkspaceSnapshot,
  ImageAttachmentSummary
} from "./assistant-client";
import {
  buildArtifactDocument,
  type ArtifactDocument
} from "./artifact-inspector";
import type { HitlApprovalSummary } from "./hitl-runtime";
import type { PreviewControllerModel } from "./preview-controller";
import type { PreviewRuntimeSource } from "./preview-runtime-adapters";
import type { PreviewRendererRoute } from "./preview-renderers";
import type { RetrievalRuntimeModel } from "./retrieval-runtime";
import type { WorkflowRuntimeModel } from "./workflow-runtime";
import type { WorkspaceSessionRecord } from "./workspace-persistence";
import type { DomainExperienceRecord } from "./domain-experience";

export type ProjectBundleFileKind =
  | "artifact"
  | "manifest"
  | "multimodal"
  | "preview"
  | "readme"
  | "runtime"
  | "session"
  | "handoff";

export type ProjectBundleFile = {
  path: string;
  mimeType: string;
  bytes: Uint8Array;
  kind: ProjectBundleFileKind;
};

export type ProjectBundleArtifactManifest = {
  id: string;
  title: string;
  type: ArtifactSummary["type"];
  language: string;
  status: string;
  summary: string;
  path: string;
};

export type ProjectBundleImageManifest = {
  id: string;
  name: string;
  mimeType: string;
  sizeBytes: number;
  createdAt: string;
  path: string | null;
  included: boolean;
};

export type ProjectBundleFileManifest = {
  path: string;
  kind: ProjectBundleFileKind;
  mimeType: string;
  sizeBytes: number;
};

export type ProjectBundleExternalHandoffManifest = {
  domain: string;
  displayName: string;
  artifactTitle: string;
  files: string[];
  boundary: string;
};

export type ProjectBundleManifest = {
  schemaVersion: 1;
  bundleName: string;
  exportedAt: string;
  workspace: {
    name: string;
    focus: string;
  };
  session: AssistantWorkspaceSnapshot["session"];
  debug: {
    status: string;
    eventCount: number;
  };
  artifacts: {
    count: number;
    items: ProjectBundleArtifactManifest[];
  };
  workflow: {
    status: string;
    currentNode: string;
    currentStep: string;
    retryCount: number;
    transitionCount: number;
    traceEventCount: number;
    exportPath: string;
  };
  retrieval: {
    state: string;
    sourceCount: number;
    chunkCount: number;
    providerLabel: string;
    exportPath: string;
  };
  preview: {
    available: boolean;
    state: string;
    rendererLabel: string;
    supportState: string;
    targetLabel: string;
    exportPath: string;
  };
  multimodal: {
    state: string;
    imageReferenceCount: number;
    includedImageCount: number;
    metadataPath: string;
    images: ProjectBundleImageManifest[];
  };
  approvals: {
    pendingCount: number;
    latestTitle: string | null;
    activeTitle: string | null;
    exportPath: string;
  };
  handoffs: {
    count: number;
    items: ProjectBundleExternalHandoffManifest[];
  };
  readmePath: string;
  sessionPath: string;
  fileCount: number;
  files: ProjectBundleFileManifest[];
  warnings: string[];
};

export type ProjectBundle = {
  fileName: string;
  manifest: ProjectBundleManifest;
  files: ProjectBundleFile[];
};

export type ProjectBundleBuildInput = {
  snapshot: AssistantWorkspaceSnapshot;
  persistenceRecord: WorkspaceSessionRecord;
  workflowRuntime: WorkflowRuntimeModel;
  retrievalRuntime: RetrievalRuntimeModel;
  previewRoute: PreviewRendererRoute;
  previewController: PreviewControllerModel;
  previewRuntimeSource: PreviewRuntimeSource;
  approvalSummary: HitlApprovalSummary;
  domainContracts?: DomainExperienceRecord[];
  exportedAt?: string;
};

type ImageBundleResult = {
  items: ProjectBundleImageManifest[];
  files: ProjectBundleFile[];
  warnings: string[];
};

type ExternalHandoffBundleResult = {
  items: ProjectBundleExternalHandoffManifest[];
  files: ProjectBundleFile[];
};

type ProjectBundleManifestSummary = Omit<
  ProjectBundleManifest,
  "fileCount" | "files"
>;

const textEncoder = new TextEncoder();
const manifestPath = "manifest.json";
const readmePath = "README.md";
const sessionPath = "session/workspace-session.json";
const workflowPath = "runtime/workflow-summary.json";
const retrievalPath = "runtime/retrieval-summary.json";
const previewPath = "runtime/preview-config.json";
const approvalPath = "runtime/operator-checkpoints.json";
const multimodalPath = "multimodal/image-references.json";

export function buildProjectBundle({
  approvalSummary,
  domainContracts = [],
  exportedAt = new Date().toISOString(),
  persistenceRecord,
  previewController,
  previewRoute,
  previewRuntimeSource,
  retrievalRuntime,
  snapshot,
  workflowRuntime
}: ProjectBundleBuildInput): ProjectBundle {
  const fileName = buildProjectBundleFileName(snapshot.session.projectId);
  const warnings: string[] = [];
  const bundleFiles: ProjectBundleFile[] = [];
  const usedPaths = new Set<string>([
    manifestPath,
    readmePath,
    sessionPath,
    workflowPath,
    retrievalPath,
    previewPath,
    approvalPath,
    multimodalPath
  ]);
  const artifactDocuments = snapshot.artifacts.map((artifact) =>
    buildArtifactDocument(snapshot, artifact)
  );
  const artifactItems = snapshot.artifacts.map((artifact, index) => {
    const path = createUniqueBundlePath(
      `artifacts/${sanitizeBundleSegment(artifactDocuments[index].fileName)}`,
      usedPaths
    );
    bundleFiles.push(
      createTextBundleFile({
        content: artifactDocuments[index].content,
        kind: "artifact",
        mimeType: artifactDocuments[index].mimeType,
        path
      })
    );

    return {
      id: artifact.id,
      title: artifact.title,
      type: artifact.type,
      language: artifact.language,
      status: artifact.status,
      summary: artifact.summary,
      path
    } satisfies ProjectBundleArtifactManifest;
  });
  const imageBundle = buildImageBundle(snapshot.multimodal.imageAttachments, usedPaths);
  warnings.push(...imageBundle.warnings);
  bundleFiles.push(...imageBundle.files);
  const handoffBundle = buildExternalHandoffBundle({
    artifacts: snapshot.artifacts,
    domainContracts,
    usedPaths
  });
  bundleFiles.push(...handoffBundle.files);

  const sessionPayload = {
    schemaVersion: persistenceRecord.schemaVersion,
    session: {
      userId: persistenceRecord.userId,
      sessionId: persistenceRecord.sessionId,
      projectId: persistenceRecord.projectId,
      title: persistenceRecord.title,
      updatedAt:
        persistenceRecord.updatedAt ?? persistenceRecord.snapshot.session.updatedAt ?? null
    },
    workspace: persistenceRecord.workspace,
    debug: persistenceRecord.snapshot.debug,
    activeInspectorTab: persistenceRecord.activeInspectorTab,
    activeArtifactId: persistenceRecord.activeArtifactId,
    previewArtifactId: persistenceRecord.previewArtifactId,
    previewOpen: persistenceRecord.previewOpen,
    layout: persistenceRecord.layout,
    preferences: persistenceRecord.preferences,
    createdAt: persistenceRecord.createdAt ?? null,
    updatedAt: persistenceRecord.updatedAt ?? null,
    snapshot: persistenceRecord.snapshot
  };
  bundleFiles.push(
    createJsonBundleFile({
      content: sessionPayload,
      kind: "session",
      path: sessionPath
    })
  );
  bundleFiles.push(
    createJsonBundleFile({
      content: {
        workflow: snapshot.workflow,
        runtime: workflowRuntime
      },
      kind: "runtime",
      path: workflowPath
    })
  );
  bundleFiles.push(
    createJsonBundleFile({
      content: retrievalRuntime,
      kind: "runtime",
      path: retrievalPath
    })
  );
  bundleFiles.push(
    createJsonBundleFile({
      content: {
        preview: snapshot.preview,
        route: previewRoute,
        controller: previewController,
        runtimeSource: {
          fingerprint: previewRuntimeSource.fingerprint,
          lineCount: previewRuntimeSource.lineCount,
          title: previewRuntimeSource.title
        }
      },
      kind: "preview",
      path: previewPath
    })
  );
  bundleFiles.push(
    createJsonBundleFile({
      content: approvalSummary,
      kind: "runtime",
      path: approvalPath
    })
  );
  bundleFiles.push(
    createJsonBundleFile({
      content: {
        state: snapshot.multimodal.state,
        status: snapshot.multimodal.status,
        detail: snapshot.multimodal.detail,
        error: snapshot.multimodal.error ?? null,
        images: imageBundle.items
      },
      kind: "multimodal",
      path: multimodalPath
    })
  );

  const manifestSummary: ProjectBundleManifestSummary = {
    schemaVersion: 1,
    bundleName: fileName,
    exportedAt,
    workspace: {
      name: snapshot.workspace.name,
      focus: snapshot.workspace.focus
    },
    session: snapshot.session,
    debug: {
      status: snapshot.debug.status,
      eventCount: snapshot.debug.events.length
    },
    artifacts: {
      count: artifactItems.length,
      items: artifactItems
    },
    workflow: {
      status: workflowRuntime.summary.status,
      currentNode: workflowRuntime.summary.currentNode,
      currentStep: workflowRuntime.summary.currentStep,
      retryCount: workflowRuntime.summary.retryCount,
      transitionCount: workflowRuntime.summary.transitionCount,
      traceEventCount: workflowRuntime.summary.traceEventCount,
      exportPath: workflowPath
    },
    retrieval: {
      state: retrievalRuntime.summary.state,
      sourceCount: retrievalRuntime.summary.sourceCount,
      chunkCount: retrievalRuntime.summary.chunkCount,
      providerLabel: retrievalRuntime.summary.providerLabel,
      exportPath: retrievalPath
    },
    preview: {
      available: snapshot.preview.available,
      state: snapshot.preview.state,
      rendererLabel: previewRoute.rendererLabel,
      supportState: previewRoute.supportState,
      targetLabel: previewRoute.targetLabel,
      exportPath: previewPath
    },
    multimodal: {
      state: snapshot.multimodal.state,
      imageReferenceCount: snapshot.multimodal.imageAttachments.length,
      includedImageCount: imageBundle.items.filter((image) => image.included).length,
      metadataPath: multimodalPath,
      images: imageBundle.items
    },
    approvals: {
      pendingCount: approvalSummary.pendingCount,
      latestTitle: approvalSummary.latestRequest?.title ?? null,
      activeTitle: approvalSummary.activeRequest?.title ?? null,
      exportPath: approvalPath
    },
    handoffs: {
      count: handoffBundle.items.length,
      items: handoffBundle.items
    },
    readmePath,
    sessionPath,
    warnings
  };

  const readme = buildProjectBundleReadme({
    artifactDocuments,
    exportedAt,
    manifest: manifestSummary,
    snapshot
  });
  const readmeFile = createTextBundleFile({
    content: readme,
    kind: "readme",
    mimeType: "text/markdown;charset=utf-8",
    path: readmePath
  });
  const contentFiles = [readmeFile, ...bundleFiles];
  const manifest = finalizeProjectBundleManifest(manifestSummary, contentFiles);
  const manifestFile = createJsonBundleFile({
    content: manifest,
    kind: "manifest",
    path: manifestPath
  });

  return {
    fileName,
    manifest,
    files: [manifestFile, ...contentFiles]
  };
}

function buildProjectBundleReadme({
  artifactDocuments,
  exportedAt,
  manifest,
  snapshot
}: {
  artifactDocuments: ArtifactDocument[];
  exportedAt: string;
  manifest: ProjectBundleManifestSummary;
  snapshot: AssistantWorkspaceSnapshot;
}) {
  const lines = [
    "# Creative Coding Assistant Project Bundle",
    "",
    `This bundle captures the current workspace state for **${snapshot.workspace.name}**.`,
    "",
    "## Session",
    `- Exported at: ${exportedAt}`,
    `- Project ID: ${snapshot.session.projectId}`,
    `- Session ID: ${snapshot.session.sessionId}`,
    `- User ID: ${snapshot.session.userId}`,
    `- Workspace focus: ${snapshot.workspace.focus}`,
    "",
    "## Workflow",
    `- Status: ${manifest.workflow.status}`,
    `- Current step: ${manifest.workflow.currentStep}`,
    `- Retries: ${manifest.workflow.retryCount}`,
    `- Trace events: ${manifest.workflow.traceEventCount}`,
    "",
    "## Retrieval",
    `- State: ${manifest.retrieval.state}`,
    `- Sources: ${manifest.retrieval.sourceCount}`,
    `- Chunks: ${manifest.retrieval.chunkCount}`,
    `- Provider: ${manifest.retrieval.providerLabel}`,
    "",
    "## Preview",
    `- State: ${manifest.preview.state}`,
    `- Renderer: ${manifest.preview.rendererLabel}`,
    `- Support: ${manifest.preview.supportState}`,
    `- Target: ${manifest.preview.targetLabel}`,
    "",
    "## Multimodal",
    `- Image references: ${manifest.multimodal.imageReferenceCount}`,
    `- Included image files: ${manifest.multimodal.includedImageCount}`,
    "",
    "## Artifact Files"
  ];

  for (const artifact of manifest.artifacts.items) {
    lines.push(`- ${artifact.path}`);
  }

  if (artifactDocuments.length > 0) {
    lines.push("");
    lines.push("## Notes");
    lines.push(
      `- ${artifactDocuments[artifactDocuments.length - 1].summary}`
    );
  }

  if (manifest.handoffs.count > 0) {
    lines.push("");
    lines.push("## External Tool Handoffs");
    lines.push(
      "These packages continue in the named external tool; they are not internal live previews."
    );
    for (const handoff of manifest.handoffs.items) {
      lines.push(`- ${handoff.displayName}: ${handoff.artifactTitle}`);
      lines.push(`  - Boundary: ${handoff.boundary}`);
      for (const file of handoff.files) {
        lines.push(`  - ${file}`);
      }
    }
  }

  if (manifest.warnings.length > 0) {
    lines.push("");
    lines.push("## Warnings");
    for (const warning of manifest.warnings) {
      lines.push(`- ${warning}`);
    }
  }

  return lines.join("\n");
}

function buildImageBundle(
  attachments: ImageAttachmentSummary[],
  usedPaths: Set<string>
): ImageBundleResult {
  const files: ProjectBundleFile[] = [];
  const items: ProjectBundleImageManifest[] = [];
  const warnings: string[] = [];

  for (const attachment of attachments) {
    const candidatePath = createUniqueBundlePath(
      `multimodal/images/${sanitizeImageFileName(attachment)}`,
      usedPaths
    );
    const decoded = decodeImageDataUrl(attachment.dataUrl);

    if (!decoded.ok) {
      warnings.push(
        `Image reference ${attachment.name} was exported as metadata only.`
      );
      items.push({
        id: attachment.id,
        name: attachment.name,
        mimeType: attachment.mimeType,
        sizeBytes: attachment.sizeBytes,
        createdAt: attachment.createdAt,
        path: null,
        included: false
      });
      continue;
    }

    files.push({
      path: candidatePath,
      mimeType: attachment.mimeType,
      bytes: decoded.bytes,
      kind: "multimodal"
    });
    items.push({
      id: attachment.id,
      name: attachment.name,
      mimeType: attachment.mimeType,
      sizeBytes: attachment.sizeBytes,
      createdAt: attachment.createdAt,
      path: candidatePath,
      included: true
    });
  }

  return { items, files, warnings };
}

function buildExternalHandoffBundle({
  artifacts,
  domainContracts,
  usedPaths
}: {
  artifacts: ArtifactSummary[];
  domainContracts: DomainExperienceRecord[];
  usedPaths: Set<string>;
}): ExternalHandoffBundleResult {
  const contractsByDomain = new Map(
    domainContracts.map((contract) => [contract.id, contract])
  );
  const files: ProjectBundleFile[] = [];
  const items: ProjectBundleExternalHandoffManifest[] = [];

  for (const artifact of artifacts) {
    const contract = artifact.domain
      ? contractsByDomain.get(artifact.domain) ?? null
      : null;
    if (!contract || contract.deliveryKind !== "external_handoff") {
      continue;
    }

    const directory = `handoff/${sanitizeBundleSegment(contract.id) || "external-domain"}`;
    const paths = {
      brief: createUniqueBundlePath(`${directory}/creative-brief.md`, usedPaths),
      specification: createUniqueBundlePath(
        `${directory}/system-specification.json`,
        usedPaths
      ),
      parameters: createUniqueBundlePath(`${directory}/parameter-schema.json`, usedPaths),
      notes: createUniqueBundlePath(`${directory}/implementation-notes.md`, usedPaths),
      checklist: createUniqueBundlePath(`${directory}/validation-checklist.md`, usedPaths),
      boundaries: createUniqueBundlePath(`${directory}/handoff-boundaries.md`, usedPaths)
    };
    files.push(
      createTextBundleFile({
        content: buildExternalHandoffBrief({ artifact, contract }),
        kind: "handoff",
        mimeType: "text/markdown;charset=utf-8",
        path: paths.brief
      }),
      createJsonBundleFile({
        content: {
          artifact: {
            language: artifact.language,
            title: artifact.title,
            type: artifact.type
          },
          artifactTypes: contract.artifactTypes,
          domain: contract.displayName,
          expectedExtensions: contract.filenameExtensions,
          workflowCompatibility: contract.workflowCompatibility
        },
        kind: "handoff",
        path: paths.specification
      }),
      createJsonBundleFile({
        content: {
          parameters: contract.intentTriggers.map((trigger) => ({
            name: sanitizeBundleSegment(trigger).replace(/-/g, "_") || "creative_parameter",
            purpose: `External-tool parameter informed by the ${trigger} intent.`,
            value: "Set in the external tool",
            required: false
          })),
          schemaVersion: 1
        },
        kind: "handoff",
        path: paths.parameters
      }),
      createTextBundleFile({
        content: buildExternalImplementationNotes({ artifact, contract }),
        kind: "handoff",
        mimeType: "text/markdown;charset=utf-8",
        path: paths.notes
      }),
      createTextBundleFile({
        content: buildExternalValidationChecklist(contract),
        kind: "handoff",
        mimeType: "text/markdown;charset=utf-8",
        path: paths.checklist
      }),
      createTextBundleFile({
        content: buildExternalHandoffBoundary(contract),
        kind: "handoff",
        mimeType: "text/markdown;charset=utf-8",
        path: paths.boundaries
      })
    );
    items.push({
      domain: contract.id,
      displayName: contract.displayName,
      artifactTitle: artifact.title,
      files: Object.values(paths),
      boundary: contract.publicClaimBoundary
    });
  }

  return { items, files };
}

function buildExternalHandoffBrief({
  artifact,
  contract
}: {
  artifact: ArtifactSummary;
  contract: DomainExperienceRecord;
}) {
  return [
    `# ${contract.displayName} Creative Handoff`,
    "",
    "## Creative brief",
    artifact.summary,
    "",
    "## Source artifact",
    `- Title: ${artifact.title}`,
    `- Language: ${artifact.language}`,
    `- Expected extensions: ${contract.filenameExtensions.join(", ")}`,
    "",
    "## Intended workflow",
    ...contract.workflowCompatibility.map((step) => `- ${step}`),
    "",
    "## Runtime assumptions",
    ...contract.runtimeRequirements.map((requirement) => `- ${requirement}`)
  ].join("\n");
}

function buildExternalImplementationNotes({
  artifact,
  contract
}: {
  artifact: ArtifactSummary;
  contract: DomainExperienceRecord;
}) {
  return [
    "# Implementation Notes",
    "",
    `Use ${artifact.title} as the source reference for the ${contract.displayName} handoff.`,
    "",
    "- Translate the exported source and parameter schema inside the external tool.",
    "- Keep the generated artifact attached to the external project for provenance.",
    "- Record changes made after import in the external project rather than claiming the workstation executed them.",
    "- Review the validation checklist before presenting the external result."
  ].join("\n");
}

function buildExternalValidationChecklist(contract: DomainExperienceRecord) {
  return [
    "# External Validation Checklist",
    "",
    `- [ ] Open the package in a licensed or installed ${contract.displayName} environment.`,
    "- [ ] Map the parameter schema to the external project controls.",
    "- [ ] Confirm source import, project save, and target runtime assumptions.",
    "- [ ] Render or play the result in the external tool.",
    "- [ ] Record any unsupported feature and use the documented fallback.",
    "- [ ] Do not describe this external output as an internal workstation preview."
  ].join("\n");
}

function buildExternalHandoffBoundary(contract: DomainExperienceRecord) {
  return [
    "# Handoff Boundaries",
    "",
    contract.publicClaimBoundary,
    "",
    "## Fallback",
    contract.fallback
  ].join("\n");
}

function buildProjectBundleFileName(projectId: string) {
  const safeProjectId =
    sanitizeBundleSegment(projectId).replace(/\.zip$/i, "") || "workspace";
  return `${safeProjectId}-bundle.zip`;
}

function sanitizeImageFileName(attachment: ImageAttachmentSummary) {
  const normalized = sanitizeBundleSegment(attachment.name);
  if (/\.[a-z0-9]+$/i.test(normalized)) {
    return normalized;
  }

  return `${normalized || "image-reference"}${mimeTypeToExtension(attachment.mimeType)}`;
}

function mimeTypeToExtension(mimeType: string) {
  switch (mimeType) {
    case "image/png":
      return ".png";
    case "image/jpeg":
      return ".jpg";
    case "image/webp":
      return ".webp";
    case "image/gif":
      return ".gif";
    default:
      return "";
  }
}

function sanitizeBundleSegment(value: string) {
  return value
    .trim()
    .replace(/\\/g, "/")
    .split("/")
    .filter(Boolean)
    .join("-")
    .replace(/[^A-Za-z0-9._-]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");
}

function createUniqueBundlePath(path: string, usedPaths: Set<string>) {
  let nextPath = path;
  let counter = 2;

  while (usedPaths.has(nextPath)) {
    nextPath = appendCounterToPath(path, counter);
    counter += 1;
  }

  usedPaths.add(nextPath);
  return nextPath;
}

function appendCounterToPath(path: string, counter: number) {
  const lastSlashIndex = path.lastIndexOf("/");
  const directory = lastSlashIndex >= 0 ? path.slice(0, lastSlashIndex + 1) : "";
  const fileName = lastSlashIndex >= 0 ? path.slice(lastSlashIndex + 1) : path;
  const extensionIndex = fileName.lastIndexOf(".");

  if (extensionIndex <= 0) {
    return `${directory}${fileName}-${counter}`;
  }

  return `${directory}${fileName.slice(0, extensionIndex)}-${counter}${fileName.slice(
    extensionIndex
  )}`;
}

function createJsonBundleFile({
  content,
  kind,
  path
}: {
  content: unknown;
  kind: ProjectBundleFileKind;
  path: string;
}) {
  return createTextBundleFile({
    content: JSON.stringify(content, null, 2),
    kind,
    mimeType: "application/json;charset=utf-8",
    path
  });
}

function createTextBundleFile({
  content,
  kind,
  mimeType,
  path
}: {
  content: string;
  kind: ProjectBundleFileKind;
  mimeType: string;
  path: string;
}) {
  return {
    path,
    mimeType,
    bytes: textEncoder.encode(content),
    kind
  } satisfies ProjectBundleFile;
}

function finalizeProjectBundleManifest(
  summary: ProjectBundleManifestSummary,
  contentFiles: ProjectBundleFile[]
) {
  const fileManifests = contentFiles.map(toBundleFileManifest);
  let manifestSizeBytes = 0;

  for (let attempt = 0; attempt < 8; attempt += 1) {
    const nextManifest: ProjectBundleManifest = {
      ...summary,
      fileCount: fileManifests.length + 1,
      files: [
        {
          path: manifestPath,
          kind: "manifest",
          mimeType: "application/json;charset=utf-8",
          sizeBytes: manifestSizeBytes
        },
        ...fileManifests
      ]
    };
    const nextManifestSizeBytes = textEncoder.encode(
      JSON.stringify(nextManifest, null, 2)
    ).byteLength;

    if (nextManifestSizeBytes === manifestSizeBytes) {
      return nextManifest;
    }

    manifestSizeBytes = nextManifestSizeBytes;
  }

  return {
    ...summary,
    fileCount: fileManifests.length + 1,
    files: [
      {
        path: manifestPath,
        kind: "manifest",
        mimeType: "application/json;charset=utf-8",
        sizeBytes: manifestSizeBytes
      },
      ...fileManifests
    ]
  } satisfies ProjectBundleManifest;
}

function toBundleFileManifest(file: ProjectBundleFile) {
  return {
    path: file.path,
    kind: file.kind,
    mimeType: file.mimeType,
    sizeBytes: file.bytes.byteLength
  } satisfies ProjectBundleFileManifest;
}

function decodeImageDataUrl(dataUrl: string):
  | { ok: true; bytes: Uint8Array }
  | { ok: false } {
  const match = dataUrl.match(/^data:[^;]+;base64,(.+)$/);
  if (!match) {
    return { ok: false };
  }

  const base64 = match[1];

  try {
    if (typeof atob === "function") {
      const binary = atob(base64);
      const bytes = new Uint8Array(binary.length);

      for (let index = 0; index < binary.length; index += 1) {
        bytes[index] = binary.charCodeAt(index);
      }

      return { ok: true, bytes };
    }

    const nodeBuffer = (
      globalThis as {
        Buffer?: {
          from: (value: string, encoding: string) => Uint8Array;
        };
      }
    ).Buffer;

    if (nodeBuffer) {
      return {
        ok: true,
        bytes: Uint8Array.from(nodeBuffer.from(base64, "base64"))
      };
    }
  } catch {
    return { ok: false };
  }

  return { ok: false };
}
