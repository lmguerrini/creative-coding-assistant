import { describe, expect, it } from "vitest";
import {
  parseHydraRuntimeSource,
  prepareHydraRuntimeSource
} from "./hydra-runtime";

describe("Hydra runtime model", () => {
  it("parses common sources, operators, output buffers, and render selection", () => {
    const result = parseHydraRuntimeSource(
      [
        "speed = 0.75;",
        "osc(12, 0.08, 1.4).kaleid(5).modulate(noise(3, 0.2), 0.14).out(o1);",
        "src(o1).blend(shape(4, 0.36, 0.02), 0.24).color(0.4, 0.9, 1).out(o0);",
        "render(o0);"
      ].join("\n")
    );

    expect(result).toMatchObject({
      ok: true,
      program: {
        renderTarget: "o0",
        speed: 0.75,
        version: 1
      }
    });
    if (!result.ok) {
      return;
    }
    expect(result.program.outputs.o1?.source.name).toBe("osc");
    expect(result.program.outputs.o1?.operators.map((operator) => operator.name)).toEqual([
      "kaleid",
      "modulate"
    ]);
    expect(result.program.outputs.o0?.source).toEqual({
      args: ["o1"],
      name: "src"
    });
  });

  it("parses readable multiline method chains as one bounded output chain", () => {
    const result = parseHydraRuntimeSource(
      [
        "osc(12, 0.08, 1.4)",
        "  .kaleid(5)",
        "  .modulate(noise(3, 0.2), 0.14)",
        "  .blend(shape(4, 0.36, 0.02), 0.24)",
        "  .out(o0);"
      ].join("\n")
    );

    expect(result).toMatchObject({
      ok: true,
      program: {
        renderTarget: "o0"
      }
    });
    if (!result.ok) {
      return;
    }
    expect(result.program.outputs.o0?.operators.map((operator) => operator.name)).toEqual([
      "kaleid",
      "modulate",
      "blend"
    ]);
  });

  it("serializes parse failures instead of executable user source", () => {
    const prepared = prepareHydraRuntimeSource(
      "osc(10).constructor.constructor('return window')().out();"
    );

    expect(JSON.parse(prepared)).toMatchObject({
      version: 1,
      error: expect.stringContaining("Unsupported Hydra syntax")
    });
    expect(prepared).not.toContain("return window");
  });

  it("rejects programs without a bounded output chain", () => {
    expect(parseHydraRuntimeSource("osc(8, 0.1, 1.2)")).toEqual({
      ok: false,
      message: "Each Hydra source chain must end with out()."
    });
  });
});
