import type { ArtifactSummary, AssistantWorkspaceSnapshot } from "./assistant-client";

const genericTitlePattern = /^(?:assistant-response|generated(?:-[a-z0-9]+)*|artifact|output|untitled)(?:-\d+)?(?:\.[a-z0-9]+){1,3}$/i;

const ignoredPromptTokens = new Set([
  "a",
  "an",
  "and",
  "artifact",
  "browser",
  "build",
  "code",
  "component",
  "create",
  "for",
  "from",
  "generate",
  "generated",
  "implementation",
  "in",
  "into",
  "is",
  "javascript",
  "js",
  "make",
  "of",
  "p5",
  "please",
  "preview",
  "react",
  "runnable",
  "scene",
  "sketch",
  "the",
  "that",
  "this",
  "to",
  "typescript",
  "use",
  "with"
]);

export function assignSemanticArtifactTitles({
  artifacts,
  eligibleArtifactIds,
  existingTitles,
  prompt
}: {
  artifacts: ArtifactSummary[];
  eligibleArtifactIds?: string[];
  existingTitles: string[];
  prompt: string;
}): ArtifactSummary[] {
  const usedTitles = new Set(existingTitles.map((title) => title.toLowerCase()));
  const eligibleIds = eligibleArtifactIds ? new Set(eligibleArtifactIds) : null;

  return artifacts.map((artifact) => {
    if (eligibleIds && !eligibleIds.has(artifact.id)) {
      usedTitles.add(artifact.title.toLowerCase());
      return artifact;
    }

    if (!isGenericArtifactTitle(artifact.title)) {
      usedTitles.add(artifact.title.toLowerCase());
      return artifact;
    }

    const semanticPrompt = artifact.creativeTranslation?.creativeIntent ?? prompt;
    const title = uniqueArtifactTitle({
      extension: artifactFileExtension(artifact),
      stem: semanticArtifactStem(semanticPrompt),
      usedTitles
    });
    usedTitles.add(title.toLowerCase());
    return { ...artifact, title };
  });
}

export function renameWorkspaceArtifact({
  artifactId,
  requestedTitle,
  snapshot
}: {
  artifactId: string;
  requestedTitle: string;
  snapshot: AssistantWorkspaceSnapshot;
}): { snapshot: AssistantWorkspaceSnapshot; title: string } | null {
  const artifact = snapshot.artifacts.find((item) => item.id === artifactId);
  if (!artifact) {
    return null;
  }

  const stem = sanitizeTitleStem(requestedTitle, artifactFileExtension(artifact));
  if (!stem) {
    return null;
  }

  const usedTitles = new Set(
    snapshot.artifacts
      .filter((item) => item.id !== artifactId)
      .map((item) => item.title.toLowerCase())
  );
  const title = uniqueArtifactTitle({
    extension: artifactFileExtension(artifact),
    stem,
    usedTitles
  });
  const previousTitle = artifact.title;

  return {
    snapshot: {
      ...snapshot,
      artifacts: snapshot.artifacts.map((item) =>
        updateArtifactTitleReferences({
          artifactId,
          item,
          previousTitle,
          title
        })
      ),
      code:
        snapshot.code.title === previousTitle
          ? { ...snapshot.code, title }
          : snapshot.code,
      preview: updatePreviewTitleReferences({
        artifactId,
        preview: snapshot.preview,
        previousTitle,
        title
      })
    },
    title
  };
}

export function isGenericArtifactTitle(title: string) {
  return genericTitlePattern.test(title.trim());
}

function updateArtifactTitleReferences({
  artifactId,
  item,
  previousTitle,
  title
}: {
  artifactId: string;
  item: ArtifactSummary;
  previousTitle: string;
  title: string;
}) {
  return {
    ...item,
    refinedFromTitle:
      item.refinedFromArtifactId === artifactId || item.refinedFromTitle === previousTitle
        ? title
        : item.refinedFromTitle,
    refinementPasses: item.refinementPasses?.map((pass) => ({
      ...pass,
      resultArtifactTitle:
        pass.resultArtifactId === artifactId || pass.resultArtifactTitle === previousTitle
          ? title
          : pass.resultArtifactTitle,
      sourceArtifactTitle:
        pass.sourceArtifactId === artifactId || pass.sourceArtifactTitle === previousTitle
          ? title
          : pass.sourceArtifactTitle
    })),
    title: item.id === artifactId ? title : item.title
  };
}

function updatePreviewTitleReferences({
  artifactId,
  preview,
  previousTitle,
  title
}: {
  artifactId: string;
  preview: AssistantWorkspaceSnapshot["preview"];
  previousTitle: string;
  title: string;
}) {
  const matchesSource =
    preview.sourceArtifactId === artifactId || preview.sourceArtifactName === previousTitle;
  const matchesOutput =
    preview.sourceArtifactId === artifactId || preview.outputArtifactName === previousTitle;

  return {
    ...preview,
    artifactName:
      matchesSource || preview.artifactName === previousTitle
        ? title
        : preview.artifactName,
    outputArtifactName: matchesOutput ? title : preview.outputArtifactName,
    sourceArtifactName: matchesSource ? title : preview.sourceArtifactName
  };
}

function artifactFileExtension(artifact: ArtifactSummary) {
  const title = artifact.title.toLowerCase();
  const runtime = `${artifact.runtime ?? ""} ${artifact.rendererId ?? ""} ${artifact.domain ?? ""}`.toLowerCase();

  if (runtime.includes("react_three_fiber") || title.endsWith(".r3f.tsx")) {
    return ".r3f.tsx";
  }
  if (runtime.includes("p5") || title.endsWith(".p5.js") || title.endsWith(".p5.ts")) {
    return ".p5.js";
  }
  if (runtime.includes("three") || title.includes(".three.")) {
    return artifact.language.toLowerCase().includes("javascript") ? ".three.js" : ".three.ts";
  }
  if (runtime.includes("glsl") || title.endsWith(".frag") || title.endsWith(".glsl")) {
    return ".frag";
  }
  if (runtime.includes("hydra") || title.includes(".hydra.")) {
    return ".hydra.js";
  }
  if (runtime.includes("tone") || title.includes(".tone.")) {
    return ".tone.js";
  }
  if (runtime.includes("gsap") || title.includes(".gsap.")) {
    return artifact.language.toLowerCase().includes("javascript") ? ".gsap.js" : ".gsap.ts";
  }
  if (runtime.includes("svg") || title.endsWith(".svg")) {
    return ".svg";
  }
  if (runtime.includes("canvas") || title.includes(".canvas.")) {
    return artifact.language.toLowerCase().includes("javascript") ? ".canvas.js" : ".canvas.ts";
  }

  const extension = title.match(/(\.[a-z0-9]+)$/)?.[1];
  return extension ?? (artifact.type === "export" ? ".md" : ".txt");
}

function semanticArtifactStem(prompt: string) {
  const tokens = prompt
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[^a-z0-9]+/g, " ")
    .split(" ")
    .filter((token) => token.length > 1 && !ignoredPromptTokens.has(token))
    .slice(0, 5);

  return tokens.join("-") || "creative-study";
}

function sanitizeTitleStem(value: string, extension: string) {
  const lowerValue = value.trim().toLowerCase();
  const withoutExtension = lowerValue.endsWith(extension)
    ? lowerValue.slice(0, -extension.length)
    : lowerValue.replace(/(?:\.[a-z0-9]+){1,3}$/i, "");
  return withoutExtension
    .normalize("NFKD")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 64);
}

function uniqueArtifactTitle({
  extension,
  stem,
  usedTitles
}: {
  extension: string;
  stem: string;
  usedTitles: Set<string>;
}) {
  const normalizedStem = sanitizeTitleStem(stem, extension) || "creative-study";
  let candidate = `${normalizedStem}${extension}`;
  let index = 2;
  while (usedTitles.has(candidate.toLowerCase())) {
    candidate = `${normalizedStem}-${index}${extension}`;
    index += 1;
  }
  return candidate;
}
