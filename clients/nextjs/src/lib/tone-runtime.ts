export type ToneRuntimeVoiceKind =
  | "synth"
  | "fm"
  | "am"
  | "membrane"
  | "metal"
  | "noise"
  | "oscillator";

export type ToneRuntimeWaveform =
  | "sine"
  | "square"
  | "triangle"
  | "sawtooth";

export type ToneRuntimeEnvelope = {
  attack: number;
  decay: number;
  sustain: number;
  release: number;
};

export type ToneRuntimeVoice = {
  kind: ToneRuntimeVoiceKind;
  waveform: ToneRuntimeWaveform;
  frequency: number | null;
  envelope: ToneRuntimeEnvelope;
};

export type ToneRuntimeEffect = {
  kind: "chorus" | "delay" | "distortion" | "filter" | "reverb";
  amount: number;
};

export type ToneRuntimePattern = {
  notes: Array<string | number>;
  subdivision: string;
};

export type ToneRuntimeProgram = {
  version: 1;
  tempo: number;
  volumeDb: number;
  voices: ToneRuntimeVoice[];
  effects: ToneRuntimeEffect[];
  patterns: ToneRuntimePattern[];
};

export type ToneRuntimeParseResult =
  | { ok: true; program: ToneRuntimeProgram }
  | { ok: false; message: string };

const maxToneSourceLength = 16_000;
const maxToneVoices = 8;
const maxToneEffects = 6;
const maxTonePatterns = 6;
const maxTonePatternSteps = 32;
const supportedSubdivisions = new Set([
  "1m",
  "2n",
  "4n",
  "8n",
  "16n",
  "32n"
]);

const voiceKindByConstructor: Record<string, ToneRuntimeVoiceKind> = {
  AMSynth: "am",
  DuoSynth: "am",
  FMSynth: "fm",
  MembraneSynth: "membrane",
  MetalSynth: "metal",
  MonoSynth: "synth",
  NoiseSynth: "noise",
  Oscillator: "oscillator",
  PolySynth: "synth",
  Synth: "synth"
};

const effectKindByConstructor: Record<string, ToneRuntimeEffect["kind"]> = {
  Chorus: "chorus",
  Distortion: "distortion",
  FeedbackDelay: "delay",
  Filter: "filter",
  Freeverb: "reverb",
  JCReverb: "reverb",
  PingPongDelay: "delay",
  Reverb: "reverb"
};

const defaultEnvelope: ToneRuntimeEnvelope = {
  attack: 0.02,
  decay: 0.18,
  sustain: 0.42,
  release: 0.6
};

export function parseToneRuntimeSource(source: string): ToneRuntimeParseResult {
  if (source.length > maxToneSourceLength) {
    return {
      ok: false,
      message: `Tone.js source exceeds the ${maxToneSourceLength}-character sandbox limit.`
    };
  }

  if (!/\bTone(?:\.|\s)/.test(source)) {
    return {
      ok: false,
      message: "Tone.js source must reference the Tone namespace."
    };
  }

  const voices = parseVoices(source);
  if (voices.length === 0) {
    return {
      ok: false,
      message:
        "Tone.js source must define a supported synth, oscillator, or noise voice."
    };
  }

  const patterns = parsePatterns(source, voices);
  return {
    ok: true,
    program: {
      version: 1,
      tempo: parseTempo(source),
      volumeDb: parseVolume(source),
      voices,
      effects: parseEffects(source),
      patterns
    }
  };
}

export function prepareToneRuntimeSource(source: string) {
  const result = parseToneRuntimeSource(source);
  return JSON.stringify(
    result.ok
      ? result.program
      : {
          version: 1,
          error: result.message
        }
  );
}

function parseVoices(source: string): ToneRuntimeVoice[] {
  const envelope = parseEnvelope(source);
  const voices: ToneRuntimeVoice[] = [];
  const constructorPattern =
    /new\s+Tone\.(AMSynth|DuoSynth|FMSynth|MembraneSynth|MetalSynth|MonoSynth|NoiseSynth|Oscillator|PolySynth|Synth)\s*\(([^;\n]*)/g;
  let match: RegExpExecArray | null;

  while ((match = constructorPattern.exec(source)) !== null) {
    const constructorName = match[1];
    const constructorArgs = match[2] ?? "";
    voices.push({
      envelope,
      frequency:
        constructorName === "Oscillator"
          ? clamp(readFirstNumber(constructorArgs) ?? 220, 24, 4_000)
          : null,
      kind: voiceKindByConstructor[constructorName] ?? "synth",
      waveform: parseWaveform(constructorArgs)
    });
    if (voices.length >= maxToneVoices) {
      break;
    }
  }

  return voices;
}

function parseEffects(source: string): ToneRuntimeEffect[] {
  const effects: ToneRuntimeEffect[] = [];
  const effectPattern =
    /new\s+Tone\.(Chorus|Distortion|FeedbackDelay|Filter|Freeverb|JCReverb|PingPongDelay|Reverb)\s*\(([^;\n]*)/g;
  let match: RegExpExecArray | null;

  while ((match = effectPattern.exec(source)) !== null) {
    effects.push({
      amount: clamp(readFirstNumber(match[2] ?? "") ?? 0.35, 0, 1),
      kind: effectKindByConstructor[match[1]] ?? "filter"
    });
    if (effects.length >= maxToneEffects) {
      break;
    }
  }

  return effects;
}

function parsePatterns(
  source: string,
  voices: ToneRuntimeVoice[]
): ToneRuntimePattern[] {
  const patterns: ToneRuntimePattern[] = [];
  const sequencePattern =
    /new\s+Tone\.(?:Pattern|Sequence)\s*\([\s\S]*?\[([\s\S]*?)\]\s*,\s*["']([^"']+)["']/g;
  let match: RegExpExecArray | null;

  while ((match = sequencePattern.exec(source)) !== null) {
    const notes = parsePatternValues(match[1] ?? "");
    if (notes.length > 0) {
      patterns.push({
        notes: notes.slice(0, maxTonePatternSteps),
        subdivision: normalizeSubdivision(match[2])
      });
    }
    if (patterns.length >= maxTonePatterns) {
      return patterns;
    }
  }

  const triggerNotes = Array.from(
    source.matchAll(
      /triggerAttackRelease\s*\(\s*(?:\[)?\s*["']([A-Ga-g][#b]?\d)["']/g
    ),
    (entry) => normalizeNote(entry[1])
  );
  if (triggerNotes.length > 0) {
    patterns.push({
      notes: triggerNotes.slice(0, maxTonePatternSteps),
      subdivision: normalizeSubdivision(
        source.match(/triggerAttackRelease\s*\([^,]+,\s*["']([^"']+)["']/)?.[1]
      )
    });
  }

  if (patterns.length === 0) {
    const oscillatorFrequencies = voices
      .map((voice) => voice.frequency)
      .filter((frequency): frequency is number => frequency !== null);
    patterns.push({
      notes:
        oscillatorFrequencies.length > 0
          ? oscillatorFrequencies
          : ["C4", "E4", "G4", "B4"],
      subdivision: normalizeSubdivision(
        source.match(/new\s+Tone\.Loop\s*\([\s\S]*?,\s*["']([^"']+)["']/)?.[1]
      )
    });
  }

  return patterns.slice(0, maxTonePatterns);
}

function parsePatternValues(value: string): Array<string | number> {
  const entries: Array<string | number> = [];
  const tokenPattern = /["']([A-Ga-g][#b]?\d)["']|(-?\d+(?:\.\d+)?)/g;
  let match: RegExpExecArray | null;

  while ((match = tokenPattern.exec(value)) !== null) {
    if (match[1]) {
      entries.push(normalizeNote(match[1]));
    } else {
      const frequency = Number(match[2]);
      if (Number.isFinite(frequency)) {
        entries.push(clamp(frequency, 24, 4_000));
      }
    }
    if (entries.length >= maxTonePatternSteps) {
      break;
    }
  }
  return entries;
}

function parseEnvelope(source: string): ToneRuntimeEnvelope {
  return {
    attack: clamp(readPropertyNumber(source, "attack") ?? defaultEnvelope.attack, 0.005, 2),
    decay: clamp(readPropertyNumber(source, "decay") ?? defaultEnvelope.decay, 0.01, 3),
    sustain: clamp(
      readPropertyNumber(source, "sustain") ?? defaultEnvelope.sustain,
      0,
      1
    ),
    release: clamp(
      readPropertyNumber(source, "release") ?? defaultEnvelope.release,
      0.02,
      4
    )
  };
}

function parseTempo(source: string) {
  const tempo =
    source.match(/Transport\.bpm\.value\s*=\s*(-?\d+(?:\.\d+)?)/)?.[1] ??
    source.match(/bpm\s*:\s*(-?\d+(?:\.\d+)?)/)?.[1];
  return clamp(Number(tempo) || 96, 40, 220);
}

function parseVolume(source: string) {
  const volume =
    source.match(/Destination\.volume\.value\s*=\s*(-?\d+(?:\.\d+)?)/)?.[1] ??
    source.match(/volume\s*:\s*(-?\d+(?:\.\d+)?)/)?.[1];
  return clamp(Number(volume) || -12, -48, 0);
}

function parseWaveform(value: string): ToneRuntimeWaveform {
  const waveform = value.match(/["'](sine|square|triangle|sawtooth)(?:\d+)?["']/i)?.[1];
  switch (waveform?.toLowerCase()) {
    case "square":
    case "triangle":
    case "sawtooth":
      return waveform.toLowerCase() as ToneRuntimeWaveform;
    default:
      return "sine";
  }
}

function normalizeSubdivision(value: string | undefined) {
  const normalized = value?.trim().toLowerCase() ?? "8n";
  return supportedSubdivisions.has(normalized) ? normalized : "8n";
}

function normalizeNote(value: string) {
  return value.slice(0, 1).toUpperCase() + value.slice(1);
}

function readPropertyNumber(source: string, property: string) {
  const match = source.match(
    new RegExp(`${property}\\s*:\\s*(-?\\d+(?:\\.\\d+)?)`, "i")
  );
  if (!match) {
    return null;
  }
  const value = Number(match[1]);
  return Number.isFinite(value) ? value : null;
}

function readFirstNumber(value: string) {
  const match = value.match(/-?\d+(?:\.\d+)?/);
  if (!match) {
    return null;
  }
  const number = Number(match[0]);
  return Number.isFinite(number) ? number : null;
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value));
}
