"use client";

import {
  RotateCcw,
  SlidersHorizontal,
  Sparkles
} from "lucide-react";
import {
  useMemo,
  useState,
  type FormEvent
} from "react";
import type { ArtifactSummary } from "@/lib/assistant-client";
import {
  buildArtifactRefinementInstruction,
  createArtifactParameterValues,
  deriveArtifactParameterModel,
  serializeArtifactParameterGuidance,
  updateArtifactParameterValue,
  type ArtifactParameterDefinition,
  type ArtifactParameterModel,
  type ArtifactParameterValues
} from "@/lib/artifact-parameters";

type ArtifactRefinementPanelProps = {
  artifact: ArtifactSummary;
  disabled: boolean;
  onArtifactRefine: (
    artifact: ArtifactSummary,
    instruction: string
  ) => Promise<void>;
};

const artifactRefinementSuggestions = [
  "Make this faster",
  "Make this more organic",
  "Add audio-reactive behavior",
  "Convert this to a calmer version",
  "Improve performance"
] as const;

export function ArtifactRefinementPanel({
  artifact,
  disabled,
  onArtifactRefine
}: ArtifactRefinementPanelProps) {
  const parameterModel = useMemo(
    () => deriveArtifactParameterModel(artifact),
    [artifact]
  );
  const [parameterValues, setParameterValues] = useState(() =>
    createArtifactParameterValues(parameterModel)
  );
  const [instruction, setInstruction] = useState("");
  const parameterGuidance = useMemo(
    () => serializeArtifactParameterGuidance(parameterModel, parameterValues),
    [parameterModel, parameterValues]
  );
  const trimmedInstruction = instruction.trim();
  const canSubmit =
    !disabled && (trimmedInstruction.length > 0 || parameterGuidance !== null);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!canSubmit) {
      return;
    }

    await onArtifactRefine(
      artifact,
      buildArtifactRefinementInstruction({
        guidance: parameterGuidance,
        instruction: trimmedInstruction
      })
    );
    setInstruction("");
    setParameterValues(createArtifactParameterValues(parameterModel));
  }

  function handleParameterChange(
    parameterId: string,
    value: unknown
  ) {
    setParameterValues((currentValues) =>
      updateArtifactParameterValue(
        parameterModel,
        currentValues,
        parameterId,
        value
      )
    );
  }

  return (
    <section
      aria-label="Selected artifact refinement"
      className="artifactRefinementCard"
    >
      <header>
        <div>
          <span>Iterate</span>
          <strong>Refine selected artifact</strong>
          <p>{`Target ${artifact.title} without regenerating every candidate.`}</p>
        </div>
        {artifact.refinedFromTitle ? (
          <span className="artifactRefinedBadge">Refined</span>
        ) : null}
      </header>
      {artifact.refinedFromTitle ? (
        <p className="artifactRefinementHistory">
          {`Refined from ${artifact.refinedFromTitle}`}
          {artifact.refinementInstruction
            ? ` with "${artifact.refinementInstruction}"`
            : ""}
        </p>
      ) : null}
      <form className="artifactRefinementForm" onSubmit={handleSubmit}>
        <ArtifactParameterControlPanel
          disabled={disabled}
          model={parameterModel}
          onChange={handleParameterChange}
          onReset={() =>
            setParameterValues(createArtifactParameterValues(parameterModel))
          }
          values={parameterValues}
        />
        <label htmlFor={`artifact-refinement-${artifact.id}`}>
          Refinement instruction
        </label>
        <textarea
          disabled={disabled}
          id={`artifact-refinement-${artifact.id}`}
          onChange={(event) => setInstruction(event.target.value)}
          placeholder="Describe an additional targeted improvement, or submit the parameter changes directly."
          rows={3}
          value={instruction}
        />
        <div className="artifactRefinementSuggestions" aria-label="Refinement examples">
          {artifactRefinementSuggestions.map((suggestion) => (
            <button
              disabled={disabled}
              key={suggestion}
              onClick={() => setInstruction(suggestion)}
              type="button"
            >
              {suggestion}
            </button>
          ))}
        </div>
        <button
          className="artifactRefinementSubmit"
          disabled={!canSubmit}
          type="submit"
        >
          {disabled
            ? "Refinement running"
            : parameterGuidance
              ? "Refine with parameter changes"
              : "Refine selected artifact"}
        </button>
      </form>
    </section>
  );
}

function ArtifactParameterControlPanel({
  disabled,
  model,
  onChange,
  onReset,
  values
}: {
  disabled: boolean;
  model: ArtifactParameterModel;
  onChange: (parameterId: string, value: unknown) => void;
  onReset: () => void;
  values: ArtifactParameterValues;
}) {
  const guidance = serializeArtifactParameterGuidance(model, values);
  const adjustableParameters = model.parameters.filter(
    (parameter) => parameter.type !== "readonly"
  );
  const readonlyParameters = model.parameters.filter(
    (parameter) => parameter.type === "readonly"
  );

  return (
    <section
      aria-label="Artifact parameter controls"
      className="artifactParameterPanel"
      data-status={model.status}
    >
      <header className="artifactParameterHeader">
        <div>
          <span>Generated controls</span>
          <strong>
            <SlidersHorizontal size={15} aria-hidden="true" />
            Artifact parameters
          </strong>
        </div>
        <span className="artifactParameterMode">Refinement guidance</span>
      </header>
      <p className="artifactParameterSafety">
        Local draft only. Values do not mutate source code or the running
        preview; submit refinement to apply them.
      </p>

      {model.status === "unsupported" ? (
        <div className="artifactParameterFallback">
          <strong>No safe parameters derived</strong>
          <p>{model.summary}</p>
          <span>Manual refinement remains available below.</span>
        </div>
      ) : (
        <>
          {readonlyParameters.length > 0 ? (
            <dl
              aria-label="Artifact parameter context"
              className="artifactParameterContext"
            >
              {readonlyParameters.map((parameter) => (
                <div key={parameter.id}>
                  <dt>{parameter.label}</dt>
                  <dd>{String(parameter.defaultValue)}</dd>
                </div>
              ))}
            </dl>
          ) : null}
          <div className="artifactParameterGrid">
            {adjustableParameters.map((parameter) => (
              <ArtifactParameterField
                disabled={disabled}
                key={parameter.id}
                onChange={(value) => onChange(parameter.id, value)}
                parameter={parameter}
                value={values[parameter.id] ?? parameter.defaultValue}
              />
            ))}
          </div>
          <footer className="artifactParameterFooter">
            <div aria-live="polite">
              <Sparkles size={13} aria-hidden="true" />
              <span>
                {guidance
                  ? `${guidance.changes.length} local change${
                      guidance.changes.length === 1 ? "" : "s"
                    } ready for refinement`
                  : "Using derived defaults"}
              </span>
            </div>
            <button
              disabled={disabled || !guidance}
              onClick={onReset}
              type="button"
            >
              <RotateCcw size={13} aria-hidden="true" />
              Reset parameters
            </button>
          </footer>
        </>
      )}
    </section>
  );
}

function ArtifactParameterField({
  disabled,
  onChange,
  parameter,
  value
}: {
  disabled: boolean;
  onChange: (value: unknown) => void;
  parameter: ArtifactParameterDefinition;
  value: string | number | boolean;
}) {
  const controlId = `artifact-parameter-${parameter.id}`;
  const label = `${parameter.label} parameter`;

  return (
    <div
      className="artifactParameterField"
      data-parameter-source={parameter.source}
      data-parameter-type={parameter.type}
    >
      <label htmlFor={controlId}>
        <span>{parameter.label}</span>
        <small>{formatParameterSource(parameter.source)}</small>
      </label>
      <p>{parameter.description}</p>
      {parameter.type === "range" ? (
        <div className="artifactParameterRange">
          <input
            aria-label={label}
            disabled={disabled}
            id={controlId}
            max={parameter.max}
            min={parameter.min}
            onChange={(event) => onChange(event.target.value)}
            step={parameter.step}
            type="range"
            value={Number(value)}
          />
          <output htmlFor={controlId}>
            {formatControlValue(value, parameter.unit)}
          </output>
        </div>
      ) : null}
      {parameter.type === "number" ? (
        <div className="artifactParameterNumber">
          <input
            aria-label={label}
            disabled={disabled}
            id={controlId}
            max={parameter.max}
            min={parameter.min}
            onChange={(event) => onChange(event.target.value)}
            step={parameter.step}
            type="number"
            value={Number(value)}
          />
          {parameter.unit ? <span>{parameter.unit}</span> : null}
        </div>
      ) : null}
      {parameter.type === "boolean" ? (
        <label
          className="artifactParameterSwitch"
          data-disabled={disabled}
          htmlFor={controlId}
        >
          <input
            aria-label={label}
            checked={Boolean(value)}
            disabled={disabled}
            id={controlId}
            onChange={(event) => onChange(event.target.checked)}
            type="checkbox"
          />
          <span aria-hidden="true" />
          <strong>{Boolean(value) ? "Enabled" : "Disabled"}</strong>
        </label>
      ) : null}
      {parameter.type === "enum" ? (
        <select
          aria-label={label}
          disabled={disabled}
          id={controlId}
          onChange={(event) => onChange(event.target.value)}
          value={String(value)}
        >
          {parameter.options?.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      ) : null}
      {parameter.type === "color" ? (
        <div className="artifactParameterColor">
          <input
            aria-label={label}
            disabled={disabled}
            id={controlId}
            onChange={(event) => onChange(event.target.value)}
            type="color"
            value={String(value)}
          />
          <code>{String(value)}</code>
        </div>
      ) : null}
    </div>
  );
}

function formatParameterSource(
  source: ArtifactParameterDefinition["source"]
) {
  switch (source) {
    case "creative_translation":
      return "Creative translation";
    case "sacred_geometry":
      return "Sacred geometry";
    case "shader_preset":
      return "Shader preset";
    case "visual_style":
      return "Visual style";
    case "code_hint":
      return "Safe code hint";
    case "modality":
      return "Modality";
    case "runtime":
      return "Runtime";
  }
}

function formatControlValue(
  value: string | number | boolean,
  unit?: string
) {
  const formatted = typeof value === "boolean" ? (value ? "On" : "Off") : value;
  return unit ? `${formatted} ${unit}` : formatted;
}
