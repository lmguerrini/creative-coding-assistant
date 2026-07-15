const { test, expect } = require("@playwright/test");
const {
  expectLoadedWorkstation,
  installApiMocks,
  installConsoleGate,
  submitCreativePrompt
} = require("./support/quality-gates");

const previewViewports = [
  { height: 900, label: "desktop", width: 1440 },
  { height: 768, label: "laptop", width: 1024 },
  { height: 900, label: "compact", width: 720 }
];

test.describe("Preview workspace design-system boundary", () => {
  for (const viewport of previewViewports) {
    test(`${viewport.label} keeps the ready artwork bounded across themes and modes`, async ({
      page
    }) => {
      await page.setViewportSize({ height: viewport.height, width: viewport.width });
      const consoleGate = installConsoleGate(page);
      await installApiMocks(page, "success");
      await expectLoadedWorkstation(page);

      await submitCreativePrompt(
        page,
        `Create the deterministic ${viewport.label} p5 preview workspace fixture.`
      );
      await expectReadyPreview(page);

      await setDisplayMode(page, "Developer");
      await selectTheme(page, "Deep Blue", "codex");
      await expectDeveloperPreviewBoundary(page);
      await expectBoundedPreviewWorkspace(page, viewport);

      await setDisplayMode(page, "User");
      await expectUserPreviewBoundary(page);
      await expectBoundedPreviewWorkspace(page, viewport);

      await selectTheme(page, "Light", "light");
      await expectUserPreviewBoundary(page);
      await expectBoundedPreviewWorkspace(page, viewport);

      await setDisplayMode(page, "Developer");
      await expectDeveloperPreviewBoundary(page);
      await expectBoundedPreviewWorkspace(page, viewport);

      if (viewport.label === "desktop") {
        await expectRendererFailureReloadRecovery(page);
      }

      await expectFullscreenArtworkLifecycle(page, viewport);

      consoleGate.assertClean();
    });
  }
});

async function expectReadyPreview(page) {
  const workspace = page.getByRole("region", { name: "Preview workspace" });
  await expect(workspace).toBeVisible();

  const expandInspector = page.getByRole("button", { name: "Expand inspector" });
  if (await expandInspector.isVisible().catch(() => false)) {
    await expandInspector.click();
  }
  await expect(page.getByRole("tab", { exact: true, name: "Preview" })).toHaveAttribute(
    "aria-selected",
    "true"
  );
  await expect(
    page.getByRole("tabpanel", { exact: true, name: "Preview" })
  ).toContainText("P5 sketch surface");

  const runtime = page.getByRole("group", { name: "p5.js live runtime" });
  await expect(runtime).toHaveAttribute("data-runtime-state", "running");
  await expect(runtime.locator('iframe[title="p5.js preview runtime"]')).toBeVisible();
  await expect(
    page.frameLocator('iframe[title="p5.js preview runtime"]').locator("canvas")
  ).toBeVisible();
}

async function selectTheme(page, themeLabel, themeValue) {
  await page.getByRole("button", { name: "Theme" }).click();
  const presets = page.getByRole("dialog", { name: "Theme presets" });
  await expect(presets).toBeVisible();
  await presets.getByRole("button", { name: `Use ${themeLabel} theme` }).click();
  await expect(presets).toHaveCount(0);
  await expect(page.locator(".workstation")).toHaveAttribute("data-theme", themeValue);
  await expect(page.locator("html")).toHaveAttribute("data-cca-theme", themeValue);
}

async function setDisplayMode(page, targetMode) {
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  const settings = page.getByRole("dialog", { name: "Workspace settings" });
  await expect(settings).toBeVisible();
  const displayMode = settings.getByRole("button", { name: "Display mode" });
  const currentMode = (await displayMode.textContent())?.trim();
  if (currentMode !== targetMode) {
    await displayMode.click();
  }
  await expect(displayMode).toHaveText(targetMode);
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  await expect(settings).toHaveCount(0);
  await expect(page.getByRole("group", { name: "p5.js live runtime" })).toHaveAttribute(
    "data-runtime-state",
    "running"
  );
}

async function expectDeveloperPreviewBoundary(page) {
  const shelf = page.locator(".previewShelf");
  await expect(shelf).toHaveAttribute("data-user-mode", "false");
  await expect(shelf.locator(".previewRuntimeStatus")).toBeVisible();
  await expect(shelf.getByRole("list", { name: "Renderer health overlay" })).toBeVisible();
  await expect(shelf.locator('[aria-label="Preview runtime source"]')).toHaveCount(1);
}

async function expectUserPreviewBoundary(page) {
  const shelf = page.locator(".previewShelf");
  await expect(shelf).toHaveAttribute("data-user-mode", "true");
  await expect(shelf.locator(".previewRuntimeStatus")).toBeVisible();
  await expect(shelf.getByRole("list", { name: "Renderer health overlay" })).toHaveCount(0);
  await expect(shelf.locator('[aria-label="Preview runtime source"]')).toHaveCount(0);
  await expect(shelf).not.toContainText("e2e-orbit-sketch.p5.js");
}

async function expectBoundedPreviewWorkspace(page, viewport) {
  const workspace = page.getByRole("region", { name: "Preview workspace" });
  const shelf = workspace.locator(".previewShelf");
  const surface = shelf.locator('.previewSurface[data-chrome="immersive"]');
  const controls = shelf.getByLabel("Preview controls");
  await expect(shelf).toHaveAttribute("data-state", "open");
  await expect(surface).toBeVisible();
  await expect(controls).toBeVisible();

  const geometry = await page.evaluate(() => {
    const selectors = {
      controls: '[aria-label="Preview controls"]',
      shelf: '.previewZone .previewShelf',
      surface: '.previewZone .previewSurface[data-chrome="immersive"]',
      workspace: '[aria-label="Preview workspace"]'
    };
    const read = (selector) => {
      const element = document.querySelector(selector);
      if (!(element instanceof HTMLElement)) {
        throw new Error(`Missing preview geometry target: ${selector}`);
      }
      const rect = element.getBoundingClientRect();
      return {
        bottom: rect.bottom,
        clientWidth: element.clientWidth,
        height: rect.height,
        left: rect.left,
        right: rect.right,
        scrollWidth: element.scrollWidth,
        top: rect.top,
        width: rect.width
      };
    };

    return {
      clientWidth: document.documentElement.clientWidth,
      controls: read(selectors.controls),
      documentScrollWidth: document.documentElement.scrollWidth,
      shelf: read(selectors.shelf),
      surface: read(selectors.surface),
      viewportHeight: window.innerHeight,
      viewportWidth: window.innerWidth,
      workspace: read(selectors.workspace)
    };
  });

  expect(geometry.viewportWidth).toBe(viewport.width);
  expect(geometry.viewportHeight).toBe(viewport.height);
  expect(geometry.documentScrollWidth).toBeLessThanOrEqual(geometry.clientWidth);

  for (const target of [geometry.workspace, geometry.shelf, geometry.surface]) {
    expect(target.width).toBeGreaterThan(240);
    expect(target.height).toBeGreaterThan(160);
    expect(target.left).toBeGreaterThanOrEqual(0);
    expect(target.right).toBeLessThanOrEqual(geometry.viewportWidth + 1);
    expect(target.scrollWidth).toBeLessThanOrEqual(target.clientWidth);
  }

  expect(geometry.controls.left).toBeGreaterThanOrEqual(geometry.surface.left - 1);
  expect(geometry.controls.right).toBeLessThanOrEqual(geometry.surface.right + 1);
  expect(geometry.controls.top).toBeGreaterThanOrEqual(geometry.surface.top - 1);
  expect(geometry.controls.bottom).toBeLessThanOrEqual(geometry.surface.bottom + 1);

  const buttonBounds = await controls.getByRole("button").evaluateAll((buttons) =>
    buttons.map((button) => {
      const rect = button.getBoundingClientRect();
      return { bottom: rect.bottom, left: rect.left, right: rect.right, top: rect.top };
    })
  );
  expect(buttonBounds).toHaveLength(4);
  for (const bounds of buttonBounds) {
    expect(bounds.left).toBeGreaterThanOrEqual(geometry.controls.left - 1);
    expect(bounds.right).toBeLessThanOrEqual(geometry.controls.right + 1);
    expect(bounds.top).toBeGreaterThanOrEqual(geometry.controls.top - 1);
    expect(bounds.bottom).toBeLessThanOrEqual(geometry.controls.bottom + 1);
  }
}

async function expectFullscreenArtworkLifecycle(page, viewport) {
  const enter = page.getByRole("button", { name: "Enter preview fullscreen" });
  await enter.focus();
  await enter.press("Enter");

  const fullscreen = page.getByRole("dialog", { name: "Fullscreen artwork canvas" });
  const exit = fullscreen.getByRole("button", { name: "Exit preview fullscreen" });
  await expect(fullscreen).toBeVisible();
  await expect(exit).toBeFocused();
  const focusedExitStyle = await exit.evaluate((element) => {
    const style = window.getComputedStyle(element);
    return style.boxShadow;
  });
  expect(focusedExitStyle).toContain("rgb(103, 185, 255)");
  await expectFullscreenGeometry(fullscreen, viewport);
  await expectFullscreenRuntime(page, fullscreen);
  await expectFullscreenSandboxKeyboardBridge(page, fullscreen, enter, exit);

  await enter.click();
  await expect(exit).toBeFocused();
  await exit.click();
  await expect(fullscreen).toHaveCount(0);
  await expect(enter).toBeFocused();
  await expect(page.getByRole("group", { name: "p5.js live runtime" })).toHaveAttribute(
    "data-runtime-state",
    "running"
  );
}

async function expectFullscreenSandboxKeyboardBridge(page, fullscreen, enter, exit) {
  const runtimeFrame = fullscreen.locator('iframe[title="p5.js preview runtime"]');
  const sandboxRoot = page
    .frameLocator('iframe[title="p5.js preview runtime"]')
    .locator("#preview-root");
  await expect(sandboxRoot).toBeVisible();

  await page.keyboard.press("Tab");
  await expect(runtimeFrame).toBeFocused();
  await expect(sandboxRoot).toBeFocused();

  await page.keyboard.press("Tab");
  await expect(exit).toBeFocused();

  await page.keyboard.press("Tab");
  await expect(sandboxRoot).toBeFocused();
  await page.keyboard.press("Shift+Tab");
  await expect(exit).toBeFocused();

  await page.keyboard.press("Tab");
  await expect(sandboxRoot).toBeFocused();
  await sandboxRoot.click({ position: { x: 4, y: 4 } });
  await expect(sandboxRoot).toBeFocused();
  await page.keyboard.press("Escape");

  await expect(fullscreen).toHaveCount(0);
  await expect(enter).toBeFocused();
  await expect(page.getByRole("group", { name: "p5.js live runtime" })).toHaveAttribute(
    "data-runtime-state",
    "running"
  );
}

async function expectFullscreenGeometry(fullscreen, viewport) {
  await expect
    .poll(() =>
      fullscreen.evaluate((layer) => {
        const targets = [
          layer,
          layer.querySelector('.previewShelf[data-fullscreen="true"]'),
          layer.querySelector(".previewPanel"),
          layer.querySelector('.previewSurface[data-chrome="immersive"]'),
          layer.querySelector(".previewRuntimeStage")
        ];
        return Math.min(
          ...targets.map((target) =>
            target instanceof HTMLElement ? target.getBoundingClientRect().bottom : 0
          )
        );
      })
    )
    .toBeGreaterThanOrEqual(viewport.height - 1);

  const geometry = await fullscreen.evaluate((layer) => {
    const read = (selector) => {
      const element = selector ? layer.querySelector(selector) : layer;
      if (!(element instanceof HTMLElement)) {
        throw new Error(`Missing fullscreen geometry target: ${selector}`);
      }
      const rect = element.getBoundingClientRect();
      return {
        bottom: rect.bottom,
        height: rect.height,
        left: rect.left,
        right: rect.right,
        top: rect.top,
        width: rect.width
      };
    };

    return {
      bodyOverflow: document.body.style.overflow,
      layer: read(null),
      panel: read(".previewPanel"),
      runtime: read(".previewRuntimeStage"),
      shelf: read('.previewShelf[data-fullscreen="true"]'),
      surface: read('.previewSurface[data-chrome="immersive"]'),
      viewportHeight: window.innerHeight,
      viewportWidth: window.innerWidth
    };
  });

  expect(geometry.viewportWidth).toBe(viewport.width);
  expect(geometry.viewportHeight).toBe(viewport.height);
  expect(geometry.bodyOverflow).toBe("hidden");
  for (const target of [
    geometry.layer,
    geometry.shelf,
    geometry.panel,
    geometry.surface,
    geometry.runtime
  ]) {
    expect(target.left).toBeLessThanOrEqual(1);
    expect(target.top).toBeLessThanOrEqual(1);
    expect(target.right).toBeGreaterThanOrEqual(viewport.width - 1);
    expect(target.bottom).toBeGreaterThanOrEqual(viewport.height - 1);
    expect(target.width).toBeGreaterThanOrEqual(viewport.width - 1);
    expect(target.height).toBeGreaterThanOrEqual(viewport.height - 1);
  }
}

async function expectFullscreenRuntime(page, fullscreen) {
  const runtime = fullscreen.getByRole("group", { name: "p5.js live runtime" });
  await expect(runtime).toHaveAttribute("data-runtime-state", "running");
  const frame = runtime.locator('iframe[title="p5.js preview runtime"]');
  await expect(frame).toBeVisible();
  const frameBox = await frame.boundingBox();
  expect(frameBox?.width).toBeGreaterThanOrEqual((await fullscreen.boundingBox()).width - 1);
  expect(frameBox?.height).toBeGreaterThanOrEqual((await fullscreen.boundingBox()).height - 1);

  const canvas = page.frameLocator('iframe[title="p5.js preview runtime"]').locator("canvas");
  await expect(canvas).toBeVisible();
  const canvasGeometry = await canvas.evaluate((element) => {
    const rect = element.getBoundingClientRect();
    return {
      height: rect.height,
      viewportHeight: window.innerHeight,
      viewportWidth: window.innerWidth,
      width: rect.width
    };
  });
  expect(canvasGeometry.width).toBeGreaterThanOrEqual(canvasGeometry.viewportWidth - 1);
  expect(canvasGeometry.height).toBeGreaterThanOrEqual(canvasGeometry.viewportHeight - 1);
}

async function expectRendererFailureReloadRecovery(page) {
  const runtime = page.getByRole("group", { name: "p5.js live runtime" });
  const frameMetric = runtime.getByRole("listitem").filter({ hasText: "Frame" });
  await expect(frameMetric).not.toContainText("—");
  const iframe = runtime.locator('iframe[title="p5.js preview runtime"]');
  const runtimeId = await iframe.getAttribute("data-runtime-id");
  expect(runtimeId).toBeTruthy();
  const iframeHandle = await iframe.elementHandle();
  const frame = await iframeHandle?.contentFrame();
  if (!frame) {
    throw new Error("The p5 preview iframe did not expose a content frame.");
  }
  const handshakeId = await iframeHandle.evaluate(
    (element, activeRuntimeId) =>
      new Promise((resolve, reject) => {
        const timeout = window.setTimeout(() => {
          window.removeEventListener("message", handleMessage);
          reject(new Error("Timed out waiting for the preview handshake nonce."));
        }, 8_000);

        function handleMessage(event) {
          if (
            event.source !== element.contentWindow ||
            event.origin !== "null" ||
            !event.data ||
            event.data.runtimeId !== activeRuntimeId ||
            typeof event.data.handshakeId !== "string"
          ) {
            return;
          }

          window.clearTimeout(timeout);
          window.removeEventListener("message", handleMessage);
          resolve(event.data.handshakeId);
        }

        window.addEventListener("message", handleMessage);
      }),
    runtimeId
  );

  await frame.evaluate(({ activeRuntimeId, activeHandshakeId }) => {
    window.parent.postMessage(
      {
        handshakeId: activeHandshakeId,
        runtimeId: activeRuntimeId,
        source: "cca-preview-runtime",
        status: {
          detail: "E2E injected renderer failure",
          diagnostics: ["E2E injected renderer failure"],
          error: {
            debugMessage: "Deterministic one-shot browser failure",
            message: "E2E injected renderer failure",
            type: "preview_sandbox_runtime_failed"
          },
          label: "p5 runtime failed",
          state: "error"
        },
        type: "status"
      },
      "*"
    );
  }, { activeHandshakeId: handshakeId, activeRuntimeId: runtimeId });

  const error = runtime.getByRole("alert");
  await expect(error).toBeVisible();
  await expect(runtime).toHaveAttribute("data-runtime-state", "error");
  await expect(error).toContainText("Preview could not start");
  await expect(error).toContainText("E2E injected renderer failure");
  const reload = runtime.getByRole("button", { name: "Reload preview runtime" });
  await expect(reload).toBeVisible();
  await reload.click();

  await expect
    .poll(() => iframe.getAttribute("data-runtime-id"))
    .not.toBe(runtimeId);
  await expect(runtime).toHaveAttribute("data-runtime-state", "running");
  await expect(
    page.frameLocator('iframe[title="p5.js preview runtime"]').locator("canvas")
  ).toBeVisible();
  await expect(error).toHaveCount(0);
}
