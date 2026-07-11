import { describe, expect, it } from "vitest";
import {
  parseToneRuntimeSource,
  prepareToneRuntimeSource
} from "./tone-runtime";

describe("Tone.js runtime model", () => {
  it("extracts bounded voices, envelopes, sequences, effects, and transport tempo", () => {
    const result = parseToneRuntimeSource(
      [
        "const synth = new Tone.FMSynth({",
        "  envelope: { attack: 0.04, decay: 0.2, sustain: 0.45, release: 0.8 }",
        "}).toDestination();",
        "const delay = new Tone.FeedbackDelay('8n', 0.3).toDestination();",
        "const sequence = new Tone.Sequence(",
        "  (time, note) => synth.triggerAttackRelease(note, '8n', time),",
        "  ['C4', 'Eb4', 'G4', 'Bb4'],",
        "  '8n'",
        ").start(0);",
        "Tone.Transport.bpm.value = 112;",
        "Tone.Transport.start();"
      ].join("\n")
    );

    expect(result).toMatchObject({
      ok: true,
      program: {
        tempo: 112,
        version: 1,
        voices: [
          {
            kind: "fm",
            envelope: {
              attack: 0.04,
              decay: 0.2,
              sustain: 0.45,
              release: 0.8
            }
          }
        ],
        effects: [{ kind: "delay" }],
        patterns: [
          {
            notes: ["C4", "Eb4", "G4", "Bb4"],
            subdivision: "8n"
          }
        ]
      }
    });
  });

  it("models oscillator loops without executing source", () => {
    const source = [
      "const drone = new Tone.Oscillator(220, 'triangle').toDestination();",
      "const loop = new Tone.Loop(() => drone.frequency.rampTo(330), '4n');",
      "loop.start(0);"
    ].join("\n");
    const prepared = prepareToneRuntimeSource(source);
    const program = JSON.parse(prepared);

    expect(program).toMatchObject({
      version: 1,
      voices: [
        {
          frequency: 220,
          kind: "oscillator",
          waveform: "triangle"
        }
      ],
      patterns: [
        {
          notes: [220],
          subdivision: "4n"
        }
      ]
    });
    expect(prepared).not.toContain("rampTo");
  });

  it("selects the deterministic Cymatics visualization only through an explicit source marker", () => {
    const source = [
      "// CCA_VISUAL: cymatics",
      "const synth = new Tone.FMSynth().toDestination();",
      "new Tone.Sequence((time, note) => synth.triggerAttackRelease(note, '8n', time), ['C3', 'G3', 'D4', 'A3'], '8n').start(0);",
      "Tone.Transport.bpm.value = 96;",
      "Tone.Transport.start();"
    ].join("\n");

    expect(parseToneRuntimeSource(source)).toMatchObject({
      ok: true,
      program: {
        tempo: 96,
        visualization: "cymatics"
      }
    });
    expect(
      JSON.parse(prepareToneRuntimeSource(source)).visualization
    ).toBe("cymatics");
    expect(
      JSON.parse(
        prepareToneRuntimeSource(
          "const synth = new Tone.Synth().toDestination(); Tone.Transport.start();"
        )
      ).visualization
    ).toBe("spectrum");
  });

  it("rejects source without a supported audio voice", () => {
    expect(
      parseToneRuntimeSource("Tone.Transport.bpm.value = 90; Tone.Transport.start();")
    ).toEqual({
      ok: false,
      message:
        "Tone.js source must define a supported synth, oscillator, or noise voice."
    });
  });
});
