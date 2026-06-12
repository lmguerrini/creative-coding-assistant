import {
  Activity,
  ArrowRight,
  ShieldCheck
} from "lucide-react";
import type {
  AudioReactiveSource,
  AudioReactiveVisualTarget,
  CreativeTranslationSummary
} from "@/lib/assistant-client";

type AudioReactiveMappingSummaryProps = {
  translation: CreativeTranslationSummary | null | undefined;
};

export function AudioReactiveMappingSummaryCard({
  translation
}: AudioReactiveMappingSummaryProps) {
  const guidance = translation?.audioReactive;

  if (!translation) {
    return (
      <FallbackCard
        detail="This legacy artifact does not include audiovisual mapping metadata."
        label="Not recorded"
        state="legacy"
      />
    );
  }

  if (translation.outputModality !== "audiovisual") {
    return (
      <FallbackCard
        detail={
          translation.outputModality === "audio"
            ? "Audio-only output has no visual target layer to drive."
            : "Audio-reactive mappings are only derived for explicit audiovisual intent."
        }
        label="Not active"
        state="inactive"
      />
    );
  }

  if (!guidance) {
    return (
      <FallbackCard
        detail="No bounded mapping evidence was recorded. Existing source and preview behavior remain unchanged."
        label="Mapping unavailable"
        state="legacy"
      />
    );
  }

  return (
    <section
      aria-label="Audio-reactive mapping summary"
      className="audioReactiveCard"
      data-state="available"
    >
      <header className="audioReactiveHeader">
        <div>
          <span>Audio-reactive mapping</span>
          <strong>
            <Activity aria-hidden="true" size={15} />
            {`${guidance.mappings.length} bounded link${
              guidance.mappings.length === 1 ? "" : "s"
            }`}
          </strong>
        </div>
        <span className="audioReactiveStatus">Audiovisual</span>
      </header>

      <p className="audioReactiveSummary">{guidance.summary}</p>

      <div className="audioReactiveMap" role="list">
        {guidance.mappings.map((mapping) => (
          <article
            className="audioReactiveMapping"
            key={mapping.source}
            role="listitem"
          >
            <div className="audioReactiveRoute">
              <strong>{formatSource(mapping.source)}</strong>
              <ArrowRight aria-hidden="true" size={14} />
              <span>
                {mapping.targets.map(formatTarget).join(" / ")}
              </span>
            </div>
            <p>{mapping.behavior}</p>
            <footer>
              <span data-intensity={mapping.intensity}>
                {mapping.intensity}
              </span>
              {mapping.evidence.length > 0 ? (
                <small>{mapping.evidence.join(" · ")}</small>
              ) : null}
            </footer>
          </article>
        ))}
      </div>

      <dl className="audioReactiveRuntimes">
        <div>
          <dt>Audio</dt>
          <dd>{guidance.audioRuntime ?? "Existing audio runtime"}</dd>
        </div>
        <div>
          <dt>Visual</dt>
          <dd>{guidance.visualRuntime ?? "Compatible visual runtime"}</dd>
        </div>
      </dl>

      <p className="audioReactiveSafety">
        <ShieldCheck aria-hidden="true" size={14} />
        Audio remains silent until explicit start. Mappings guide generation;
        they do not mutate source or start playback.
      </p>
    </section>
  );
}

function FallbackCard({
  detail,
  label,
  state
}: {
  detail: string;
  label: string;
  state: "inactive" | "legacy";
}) {
  return (
    <section
      aria-label="Audio-reactive mapping summary"
      className="audioReactiveCard"
      data-state={state}
    >
      <header className="audioReactiveHeader">
        <div>
          <span>Audio-reactive mapping</span>
          <strong>{label}</strong>
        </div>
      </header>
      <p className="audioReactiveSummary">{detail}</p>
    </section>
  );
}

function formatSource(source: AudioReactiveSource) {
  return sentenceCase(source);
}

function formatTarget(target: AudioReactiveVisualTarget) {
  return sentenceCase(target);
}

function sentenceCase(value: string) {
  const normalized = value.replaceAll("_", " ");
  return normalized.charAt(0).toUpperCase() + normalized.slice(1);
}
