const { test, expect } = require("@playwright/test");
const {
  expectLoadedWorkstation,
  installApiMocks,
  installConsoleGate
} = require("./support/quality-gates");
const { showcaseSmokeCases } = require("./support/demo-fixtures");

const localFixtureBoundary =
  "Local deterministic showcase fixture for browser smoke; this is not provider-backed generation evidence.";

test.describe("V9.8 canonical showcase browser smoke (local deterministic fixtures)", () => {
  for (const showcase of showcaseSmokeCases) {
    test(`${showcase.title}: request, runtime, fullscreen, refinement, reload, and quality`, async ({
      page
    }) => {
      const consoleGate = installConsoleGate(page);
      await installApiMocks(page, `showcase:${showcase.id}`);
      await expectLoadedWorkstation(page);

      const generationRequest = await runShowcaseFromReviewedDemoDetail(page, showcase);
      const generationPayload = generationRequest.postDataJSON();

      expect(generationPayload).toMatchObject({
        mode: "generate",
        workflowMode: "single_agent"
      });
      expect(generationPayload.attachments).toBeUndefined();
      for (const token of showcase.requestTokens) {
        expect(generationPayload.query).toContain(token);
      }

      await expect(page.getByRole("form", { name: "Creative request composer" })).toHaveAttribute(
        "aria-busy",
        "false"
      );
      await expandInspector(page);
      await assertHydratedArtifactAndSource(page, showcase);
      await assertLiveRuntimeSignal(page, showcase);
      await assertCreativeSessionFullscreenRestore(page);

      const refinementRequest = await submitShowcaseRefinement(page, showcase);
      const refinementPayload = refinementRequest.postDataJSON();

      expect(refinementPayload).toMatchObject({
        domain: showcase.artifact.domain,
        domains: [showcase.artifact.domain],
        mode: "generate",
        query: showcase.followUp,
        workflowMode: "auto",
        artifactRefinement: {
          artifactId: showcase.artifact.id,
          content: showcase.artifact.content,
          domain: showcase.artifact.domain,
          instruction: showcase.followUp,
          rendererId: showcase.artifact.renderer_id,
          runtime: showcase.artifact.runtime,
          title: showcase.artifact.title
        }
      });
      await expect(page.getByRole("form", { name: "Creative request composer" })).toHaveAttribute(
        "aria-busy",
        "false"
      );

      const refinedTitle = createRefinedArtifactTitle(showcase.artifact.title);
      await page.getByRole("tab", { exact: true, name: "Artifacts" }).click();
      await expect(page.getByRole("group", { name: "Active artifact details" })).toContainText(
        refinedTitle
      );
      await expect
        .poll(() =>
          page.evaluate(
            (title) =>
              Object.values(window.localStorage).some((value) => value.includes(title)),
            refinedTitle
          )
        )
        .toBe(true);

      await page.reload();
      await expect(page.getByRole("region", { name: "Creative workspace" })).toBeVisible();
      await expect(page.getByRole("region", { name: "Creative session" })).toBeVisible();
      await expandInspector(page);
      await page.getByRole("tab", { exact: true, name: "Artifacts" }).click();
      await expect(page.getByRole("group", { name: "Active artifact details" })).toContainText(
        refinedTitle
      );
      await assertLiveRuntimeSignal(page, showcase, refinedTitle);

      consoleGate.assertClean();
    });
  }
});

async function runShowcaseFromReviewedDemoDetail(page, showcase) {
  await page.getByRole("button", { name: "Demo Mode" }).click();
  const demoMode = page.getByRole("region", { name: "Demo Mode" });
  const scenarioList = demoMode.getByRole("list", { name: "Demo Mode scenarios" });
  const scenario = scenarioList.getByRole("button").filter({ hasText: showcase.title });

  await expect(scenario).toHaveCount(1);
  await scenario.click();
  const detail = demoMode.getByRole("article", { name: "Selected demo scenario" });
  await expect(detail).toContainText(showcase.title);
  await expect(detail).toContainText(showcase.artifact.title);

  const requestPromise = page.waitForRequest("**/api/assistant/stream");
  await detail.getByRole("button", { name: "Load prompt & run" }).click();
  const request = await requestPromise;

  await expect(demoMode).toHaveCount(0);
  return request;
}

async function assertHydratedArtifactAndSource(page, showcase) {
  await page.getByRole("tab", { exact: true, name: "Artifacts" }).click();
  const details = page.getByRole("group", { name: "Active artifact details" });

  await expect(details).toContainText(showcase.artifact.title);
  await expect(details).toContainText(localFixtureBoundary);
  await expect(details).toContainText(showcase.runtimeLabel);

  await page.getByRole("tab", { exact: true, name: "Code" }).click();
  const code = page.getByRole("tabpanel", { exact: true, name: "Code" });
  await expect(code).toHaveAttribute("data-opened-artifact", showcase.artifact.title);
  for (const token of showcase.qualityTokens) {
    await expect(code).toContainText(token);
  }
}

async function assertLiveRuntimeSignal(page, showcase, expectedTitle = showcase.artifact.title) {
  await page.getByRole("tab", { exact: true, name: "Preview" }).click();
  const previewInspector = page.getByRole("tabpanel", { exact: true, name: "Preview" });
  await expect(previewInspector).toContainText(expectedTitle);

  const runtime = page.getByRole("group", {
    name: `${showcase.runtimeLabel} live runtime`
  });
  await expect(runtime).toBeVisible();
  await expect(runtime).toHaveAttribute(
    "data-runtime-kind",
    showcase.artifact.runtime
  );
  await expect(runtime).toHaveAttribute(
    "data-runtime-state",
    showcase.expectedRuntimeState
  );
  const frame = page.frameLocator(
    `iframe[title="${showcase.runtimeLabel} preview runtime"]`
  );
  await expect(frame.locator("canvas")).toBeVisible();

  if (showcase.artifact.runtime === "three") {
    await expect(frame.locator("body")).toHaveAttribute(
      "data-three-runtime-revision",
      "176"
    );
    await expect
      .poll(async () => Number(await frame.locator("body").getAttribute("data-three-frame-energy")))
      .toBeGreaterThan(80);
    const firstSignature = await frame
      .locator("body")
      .getAttribute("data-three-frame-signature");
    expect(firstSignature).toBeTruthy();
    await expect
      .poll(() => frame.locator("body").getAttribute("data-three-frame-signature"))
      .not.toBe(firstSignature);
    return;
  }

  if (showcase.artifact.runtime === "p5" || showcase.artifact.runtime === "tone") {
    await expect
      .poll(() => readTwoDimensionalCanvasEnergy(frame.locator("canvas")))
      .toBeGreaterThan(8);
    if (showcase.artifact.runtime === "tone") {
      await expect(frame.getByRole("button", { name: "Start audio" })).toBeVisible();
      await expect(runtime).toContainText("Audio remains silent until Start audio is selected.");
    }
    return;
  }

  const frameMetric = runtime
    .getByRole("listitem")
    .filter({ hasText: "Frame" });
  await expect(frameMetric).toContainText(/\d+\.\d+ ms/);
}

async function assertCreativeSessionFullscreenRestore(page) {
  await expandInspector(page);
  const workstation = page.locator(".workstation");
  const before = await workstation.evaluate((element) => ({
    inspector: element.getAttribute("data-inspector-state"),
    preview: element.getAttribute("data-preview"),
    sidebar: element.getAttribute("data-sidebar-state")
  }));

  expect(before).toEqual({ inspector: "open", preview: "open", sidebar: "open" });
  await page.getByRole("button", { name: "Settings" }).click();
  await page
    .getByRole("button", {
      name: "Toggle Fullscreen Creative Session from quick actions"
    })
    .click();
  await expect(workstation).toHaveAttribute("data-focus-mode", "true");
  await expect(workstation).toHaveAttribute("data-inspector-state", "collapsed");
  await expect(workstation).toHaveAttribute("data-preview", "closed");
  await expect(workstation).toHaveAttribute("data-sidebar-state", "collapsed");
  await expect(page.getByRole("region", { name: "Creative session" })).toBeVisible();

  await page.getByRole("button", { name: "Settings" }).click();
  await page
    .getByRole("button", {
      name: "Toggle Fullscreen Creative Session from quick actions"
    })
    .click();
  await expect(workstation).toHaveAttribute("data-focus-mode", "false");
  const restored = await workstation.evaluate((element) => ({
    inspector: element.getAttribute("data-inspector-state"),
    preview: element.getAttribute("data-preview"),
    sidebar: element.getAttribute("data-sidebar-state")
  }));
  expect(restored).toEqual(before);
}

async function submitShowcaseRefinement(page, showcase) {
  await page.getByRole("tab", { exact: true, name: "Artifacts" }).click();
  const details = page.getByRole("group", { name: "Active artifact details" });
  const refinement = details.getByRole("region", {
    name: "Selected artifact refinement"
  });

  await refinement.getByLabel("Refinement instruction").fill(showcase.followUp);
  const requestPromise = page.waitForRequest("**/api/assistant/stream");
  const submit = refinement.getByRole("button", { name: "Apply refinement" });
  await submit.click();
  return requestPromise;
}

async function expandInspector(page) {
  const expand = page.getByRole("button", { name: "Expand inspector" });
  if (await expand.isVisible().catch(() => false)) {
    await expand.click();
  }
  await expect(page.getByRole("complementary", { name: "Right inspector" })).toHaveAttribute(
    "data-state",
    "open"
  );
}

async function readTwoDimensionalCanvasEnergy(canvas) {
  return canvas.evaluate((element) => {
    const context = element.getContext("2d");
    if (!context || element.width < 2 || element.height < 2) {
      return 0;
    }
    const pixels = context.getImageData(0, 0, element.width, element.height).data;
    const stride = Math.max(4, Math.floor(pixels.length / 16_000 / 4) * 4);
    let energy = 0;
    let samples = 0;
    for (let index = 0; index < pixels.length; index += stride) {
      energy += pixels[index] + pixels[index + 1] + pixels[index + 2];
      samples += 1;
    }
    return samples > 0 ? energy / samples : 0;
  });
}

function createRefinedArtifactTitle(title) {
  const extensionIndex = title.lastIndexOf(".");
  return extensionIndex > 0
    ? `${title.slice(0, extensionIndex)}.refined${title.slice(extensionIndex)}`
    : `${title}-refined`;
}
