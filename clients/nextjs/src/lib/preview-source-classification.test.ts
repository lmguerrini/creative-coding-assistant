import { describe, expect, it } from "vitest";
import {
  getGlslRuntimeSourceSupportIssue,
  getP5RuntimeSourceSupportIssue,
  getThreeRuntimeSourceSupportIssue,
  prepareThreeJavaScriptSource,
  threeHtmlSourceMismatchMessage
} from "./preview-source-classification";

describe("Three.js preview source classification", () => {
  it("rejects standalone HTML before it reaches the JavaScript sandbox", () => {
    expect(
      getThreeRuntimeSourceSupportIssue(
        "<!-- document -->\n<!doctype html><html><script>new THREE.Scene()</script></html>"
      )
    ).toBe(threeHtmlSourceMismatchMessage);
  });

  it("normalizes a plain Three.js module scene into executable runtime source", () => {
    const source = [
      "import * as THREE from 'three';",
      "const scene = new THREE.Scene();",
      "const renderer = new THREE.WebGLRenderer();"
    ].join("\n");

    expect(getThreeRuntimeSourceSupportIssue(source)).toBeNull();
    expect(prepareThreeJavaScriptSource(source)).toBe(
      "const scene = new THREE.Scene();\nconst renderer = new THREE.WebGLRenderer();"
    );
  });
});

describe("p5 preview source classification", () => {
  it("accepts common p5 helpers and ordinary arrow callbacks in a global-mode sketch", () => {
    const source = [
      "const particles = [{ life: 1 }];",
      "function setup() { createCanvas(320, 180); }",
      "function draw() {",
      "  const value = constrain(int(noise(frameCount * 0.01) * 320), 0, 320);",
      "  const orbit = sin(TAU * frameCount * 0.01);",
      "  const active = particles.filter(p => p.life > 0);",
      "  background(8);",
      "  circle(value, 90 + orbit, active.length * 18);",
      "}"
    ].join("\n");

    expect(getP5RuntimeSourceSupportIssue(source)).toBeNull();
  });
});

describe("GLSL preview source classification", () => {
  it("accepts a compact bounded fragment shader and rejects unsupported source", () => {
    expect(
      getGlslRuntimeSourceSupportIssue(
        "void main() { gl_FragColor = vec4(0.5 + 0.5 * sin(u_time)); }"
      )
    ).toBeNull();
    expect(
      getGlslRuntimeSourceSupportIssue(
        "uniform sampler2D sourceTexture; void main() { gl_FragColor = texture2D(sourceTexture, vec2(0.5)); }"
      )
    ).toContain("outside the current bounded runtime subset");
    expect(
      getGlslRuntimeSourceSupportIssue(
        "#version 300 es\nout vec4 color; void main() { color = vec4(1.0); }"
      )
    ).toContain("#version declarations");
  });
});
