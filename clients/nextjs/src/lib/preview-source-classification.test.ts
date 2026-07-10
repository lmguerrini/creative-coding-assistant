import { describe, expect, it } from "vitest";
import {
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
      "  const active = particles.filter(p => p.life > 0);",
      "  background(8);",
      "  circle(value, 90, active.length * 18);",
      "}"
    ].join("\n");

    expect(getP5RuntimeSourceSupportIssue(source)).toBeNull();
  });
});
