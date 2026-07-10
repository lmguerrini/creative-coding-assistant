const { test, expect } = require("@playwright/test");

test("executes a bounded Three.js scene with common scene-graph APIs", async ({ page }) => {
  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));

  const source = [
    "const scene = new THREE.Scene();",
    "scene.background = new THREE.Color(0x101418);",
    "const camera = new THREE.PerspectiveCamera(50, 4 / 3, 0.1, 100);",
    "const pivot = new THREE.Object3D();",
    "pivot.add(camera);",
    "scene.add(pivot);",
    "const group = new THREE.Group();",
    "scene.add(group);",
    "scene.add(new THREE.HemisphereLight(0xffffff, 0x111111, 1));",
    "const floor = new THREE.Mesh(new THREE.PlaneGeometry(12, 12), new THREE.MeshStandardMaterial({ color: 0x1d2933, side: THREE.DoubleSide }));",
    "group.add(floor);",
    "const mesh = new THREE.Mesh(new THREE.TorusGeometry(1.2, 0.28, 18, 48), new THREE.MeshStandardMaterial({ color: 0x4cd7c8, emissive: 0x1d6f78 }));",
    "group.add(mesh);",
    "const renderer = new THREE.WebGLRenderer({ antialias: true });",
    "renderer.outputEncoding = THREE.sRGBEncoding;",
    "renderer.shadowMap.enabled = true;",
    "renderer.shadowMap.type = THREE.PCFSoftShadowMap;",
    "function animate() { mesh.rotation.y += 0.015; renderer.render(scene, camera); requestAnimationFrame(animate); }",
    "requestAnimationFrame(animate);"
  ].join("\n");
  const runtime = {
    kind: "three",
    runtimeId: "three-common-apis",
    source: {
      fingerprint: "three-common-apis",
      lineCount: source.split("\n").length,
      source,
      title: "kinetic-studio-sculpture.three.js"
    }
  };

  await page.goto(`/preview-sandbox.html#${encodeURIComponent(JSON.stringify(runtime))}`);

  await expect(page.locator("body")).toHaveAttribute("data-runtime-state", "running");
  await expect(page.locator("#preview-canvas")).toBeVisible();
  expect(pageErrors).toEqual([]);
});
