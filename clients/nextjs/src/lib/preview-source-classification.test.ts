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
      "const retainParticle = function (particle) { return particle.life > 0; };",
      "function setup() { createCanvas(320, 180); }",
      "function draw() {",
      "  const value = constrain(int(noise(frameCount * 0.01) * 320), 0, 320);",
      "  const orbit = sin(TAU * frameCount * 0.01);",
      "  const eased = exp(-value * 0.01);",
      "  const active = particles.filter(retainParticle);",
      "  strokeCap(ROUND);",
      "  strokeJoin(ROUND);",
      "  blendMode(ADD);",
      "  rectMode(CENTER);",
      "  smooth();",
      "  const swatch = color(20, 40, 60);",
      "  fill(red(swatch), green(swatch), blue(swatch));",
      "  beginShape(); curveVertex(value, 90 + orbit); endShape();",
      "  background(8);",
      "  circle(value, 90 + orbit, active.length * 18 * eased);",
      "}"
    ].join("\n");

    expect(getP5RuntimeSourceSupportIssue(source)).toBeNull();
  });

  it("accepts the standard degrees helper used to derive a p5 hue", () => {
    const source = [
      "function setup() { createCanvas(320, 180); colorMode(HSL, 360, 100, 100); }",
      "function draw() {",
      "  const hue = degrees(noise(frameCount * 0.01) * TWO_PI) % 360;",
      "  background(hue, 48, 12);",
      "  circle(width / 2, height / 2, 40);",
      "}"
    ].join("\n");

    expect(getP5RuntimeSourceSupportIssue(source)).toBeNull();
  });

  it("does not mistake a parenthesized return expression for a function call", () => {
    const source = [
      "function setup() { createCanvas(320, 180); }",
      "function boundedMax(a, b) { return (a > b) ? a : b; }",
      "function draw() {",
      "  background(8);",
      "  circle(width / 2, height / 2, boundedMax(24, 32));",
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
    expect(
      getGlslRuntimeSourceSupportIssue(
        "// texture is not used here\nvoid main() { gl_FragColor = vec4(1.0); }"
      )
    ).toBeNull();
  });

  it("rejects common refinement contamination before WebGL compilation", () => {
    expect(
      getGlslRuntimeSourceSupportIssue(
        "float field(vec2 uv): float { return uv.x; }\nvoid main() { gl_FragColor = vec4(field(vec2(0.5))); }"
      )
    ).toContain("non-GLSL colon syntax");
    expect(
      getGlslRuntimeSourceSupportIssue(
        "```glsl\nvoid main() { gl_FragColor = vec4(1.0); }\n```"
      )
    ).toContain("Markdown fences");
  });

  it("keeps valid GLSL ternary expressions previewable", () => {
    expect(
      getGlslRuntimeSourceSupportIssue(
        "void main() { float value = u_time > 1.0 ? 1.0 : 0.0; gl_FragColor = vec4(value); }"
      )
    ).toBeNull();
  });
});
