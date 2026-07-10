import { describe, expect, it } from "vitest";
import {
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
