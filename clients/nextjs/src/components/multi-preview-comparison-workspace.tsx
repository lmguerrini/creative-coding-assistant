"use client";

import {
  Check,
  Code2,
  Play,
  ShieldCheck,
  Sparkles,
  Volume2
} from "lucide-react";
import type {
  ArtifactAction,
  ArtifactSummary
} from "@/lib/assistant-client";
import type {
  MultiPreviewCandidate,
  MultiPreviewComparisonModel
} from "@/lib/multi-preview-comparison";
import { PreviewRendererSurface } from "./preview-renderer-surface";

type MultiPreviewComparisonWorkspaceProps = {
  comparison: MultiPreviewComparisonModel;
  onArtifactAction: (action: ArtifactAction, artifact: ArtifactSummary) => void;
  onArtifactSelect: (artifact: ArtifactSummary) => void;
};

export function MultiPreviewComparisonWorkspace({
  comparison,
  onArtifactAction,
  onArtifactSelect
}: MultiPreviewComparisonWorkspaceProps) {
  return (
    <section
      aria-label="Artifact comparison"
      className="multiPreviewWorkspace"
      data-layout={comparison.layout}
    >
      <header className="multiPreviewWorkspaceHeader">
        <div>
          <span>Multi-preview workspace</span>
          <strong>{`${comparison.candidates.length} candidate${
            comparison.candidates.length === 1 ? "" : "s"
          }`}</strong>
          <p>
            Compare generated outputs directly, then select the artifact that
            should drive Code, Preview, and refinement.
          </p>
        </div>
        {comparison.recommendedTitle ? (
          <div
            aria-label="Recommended artifact comparison"
            className="multiPreviewRecommendation"
            role="group"
          >
            <Sparkles size={14} aria-hidden="true" />
            <span>Recommended</span>
            <strong>{comparison.recommendedTitle}</strong>
          </div>
        ) : null}
      </header>

      {comparison.recommendedTitle ? (
        <p className="multiPreviewRecommendationReason">
          {comparison.recommendedReason}
        </p>
      ) : null}

      {comparison.candidates.length > 0 ? (
        <div className="multiPreviewGrid" role="list">
          {comparison.candidates.map((candidate) => (
            <MultiPreviewCandidateCard
              candidate={candidate}
              key={candidate.artifact.id}
              onArtifactAction={onArtifactAction}
              onArtifactSelect={onArtifactSelect}
            />
          ))}
        </div>
      ) : (
        <div className="multiPreviewEmpty">
          <strong>No candidates yet</strong>
          <p>
            Generate multiple artifacts to compare their live output, runtime
            support, and creative metadata here.
          </p>
        </div>
      )}
    </section>
  );
}

function MultiPreviewCandidateCard({
  candidate,
  onArtifactAction,
  onArtifactSelect
}: {
  candidate: MultiPreviewCandidate;
  onArtifactAction: (action: ArtifactAction, artifact: ArtifactSummary) => void;
  onArtifactSelect: (artifact: ArtifactSummary) => void;
}) {
  const canOpen = candidate.artifact.actions.includes("Open");
  const canOpenPreview =
    candidate.canRender && candidate.artifact.actions.includes("Preview");

  return (
    <article
      aria-current={candidate.row.isActive ? "true" : undefined}
      aria-label={`${candidate.row.title} comparison candidate`}
      className="multiPreviewCandidate"
      data-active={candidate.row.isActive}
      data-output-kind={candidate.outputKind}
      data-recommended={candidate.row.isRecommended}
      data-runtime-support={candidate.row.runtimeSupport.state}
      role="listitem"
    >
      <header className="multiPreviewCandidateHeader">
        <div>
          <strong>{candidate.row.title}</strong>
          <span>{`${candidate.outputLabel} / ${candidate.row.runtimeLabel}`}</span>
        </div>
        <div className="multiPreviewCandidateBadges">
          {candidate.row.isRecommended ? <span>Recommended</span> : null}
          {candidate.row.isActive ? <span>Selected</span> : null}
          <span data-support={candidate.row.runtimeSupport.state}>
            {candidate.row.runtimeSupport.label}
          </span>
        </div>
      </header>

      <div
        aria-label={`${candidate.row.title} comparison preview`}
        className="multiPreviewCanvas"
      >
        {candidate.canRender ? (
          <PreviewRendererSurface
            chrome="comparison"
            preview={candidate.preview}
            route={candidate.route}
            runtimeSessionKey={candidate.runtimeSessionKey}
            runtimeSource={candidate.runtimeSource}
          />
        ) : (
          <MultiPreviewFallback candidate={candidate} />
        )}
      </div>

      <div className="multiPreviewCandidateSignals">
        <span>
          <ShieldCheck size={12} aria-hidden="true" />
          {candidate.audioSafetyLabel}
        </span>
        <span>{candidate.row.scoreLabel}</span>
        <span>{candidate.row.rankLabel}</span>
      </div>

      <MultiPreviewCreativeMetadata candidate={candidate} />

      <div className="multiPreviewCandidateActions">
        <button
          aria-label={`Select ${candidate.row.title} as preferred candidate`}
          data-action="select"
          onClick={() => onArtifactSelect(candidate.artifact)}
          type="button"
        >
          {candidate.row.isActive ? (
            <Check size={14} aria-hidden="true" />
          ) : (
            <Sparkles size={14} aria-hidden="true" />
          )}
          {candidate.row.isActive ? "Selected" : "Use candidate"}
        </button>
        {canOpen ? (
          <button
            aria-label={`Open code for ${candidate.row.title}`}
            onClick={() => onArtifactAction("Open", candidate.artifact)}
            type="button"
          >
            <Code2 size={14} aria-hidden="true" />
            Code
          </button>
        ) : null}
        {canOpenPreview ? (
          <button
            aria-label={`Preview ${candidate.row.title} from comparison`}
            onClick={() => onArtifactAction("Preview", candidate.artifact)}
            type="button"
          >
            <Play size={14} aria-hidden="true" />
            Preview
          </button>
        ) : null}
      </div>
    </article>
  );
}

function MultiPreviewFallback({
  candidate
}: {
  candidate: MultiPreviewCandidate;
}) {
  return (
    <div
      aria-label={`${candidate.row.title} safe preview fallback`}
      className="multiPreviewFallback"
      data-support={candidate.row.runtimeSupport.state}
    >
      {candidate.outputKind === "audio" ? (
        <Volume2 size={26} aria-hidden="true" />
      ) : (
        <Code2 size={26} aria-hidden="true" />
      )}
      <strong>{candidate.row.runtimeSupport.label}</strong>
      <p>{candidate.row.runtimeSupport.detail}</p>
      <span>{candidate.row.previewLabel}</span>
    </div>
  );
}

function MultiPreviewCreativeMetadata({
  candidate
}: {
  candidate: MultiPreviewCandidate;
}) {
  const groups = [
    {
      label: "Style",
      values: candidate.visualStyleLabels
    },
    {
      label: "Shader",
      values: candidate.shaderPresetLabels
    },
    {
      label: "Geometry",
      values: candidate.geometryLabels
    }
  ].filter((group) => group.values.length > 0);

  if (groups.length === 0) {
    return (
      <p className="multiPreviewLegacyMetadata">
        No structured creative metadata recorded.
      </p>
    );
  }

  return (
    <dl className="multiPreviewCreativeMetadata">
      {groups.map((group) => (
        <div key={group.label}>
          <dt>{group.label}</dt>
          <dd>{group.values.join(" / ")}</dd>
        </div>
      ))}
    </dl>
  );
}
