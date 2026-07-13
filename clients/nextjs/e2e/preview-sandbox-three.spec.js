const { test, expect } = require("@playwright/test");
const {
  kineticOrbitSculptureSource
} = require("./support/demo-fixtures");

test("renders the Kinetic Orbit Sculpture with real Three.js scene-graph motion", async ({ page }) => {
  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));

  const runtime = {
    kind: "three",
    runtimeId: "kinetic-orbit-sculpture",
    source: {
      fingerprint: "kinetic-orbit-sculpture",
      lineCount: kineticOrbitSculptureSource.split("\n").length,
      source: kineticOrbitSculptureSource,
      title: "kinetic-orbit-sculpture.three.js"
    }
  };

  await page.goto(`/preview-sandbox.html#${encodeURIComponent(JSON.stringify(runtime))}`);

  await expect(page.locator("body")).toHaveAttribute("data-runtime-state", "running");
  await expect(page.locator("body")).toHaveAttribute("data-three-runtime-revision", "176");
  await expect(page.locator("#preview-canvas")).toBeVisible();
  await expect
    .poll(async () => Number(await page.locator("body").getAttribute("data-three-frame-energy")))
    .toBeGreaterThan(80);

  const firstSignature = await page
    .locator("body")
    .getAttribute("data-three-frame-signature");
  expect(firstSignature).toBeTruthy();
  await expect
    .poll(() => page.locator("body").getAttribute("data-three-frame-signature"))
    .not.toBe(firstSignature);

  expect(kineticOrbitSculptureSource).toContain("new THREE.TorusKnotGeometry");
  expect(kineticOrbitSculptureSource).toContain("cameraRig.add(camera)");
  expect(kineticOrbitSculptureSource).toContain("sculptureRig.add(orbitRig)");
  expect(pageErrors).toEqual([]);
});
