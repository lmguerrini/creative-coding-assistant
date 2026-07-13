const { test, expect } = require("@playwright/test");
const {
  expectLoadedWorkstation,
  installApiMocks,
  installConsoleGate
} = require("./support/quality-gates");

const requestControlViewports = [
  { height: 900, label: "desktop", width: 1440 },
  { height: 768, label: "laptop", width: 1024 },
  { height: 900, label: "compact", width: 720 }
];

const visualStates = [
  { mode: "Developer", themeLabel: "Deep Blue", themeValue: "codex" },
  { mode: "User", themeLabel: "Deep Blue", themeValue: "codex" },
  { mode: "User", themeLabel: "Light", themeValue: "light" },
  { mode: "Developer", themeLabel: "Light", themeValue: "light" }
];

const onePixelPng = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl0sZ0AAAAASUVORK5CYII=",
  "base64"
);

test.describe("Attachment and workflow control design-system boundary", () => {
  for (const viewport of requestControlViewports) {
    test(`${viewport.label} keeps request controls bounded, persistent, and request-scoped`, async ({
      page
    }) => {
      await page.setViewportSize({ height: viewport.height, width: viewport.width });
      const consoleGate = installConsoleGate(page);
      await installApiMocks(page, "success");
      const streamGate = await installOneShotStreamGate(page);
      await expectLoadedWorkstation(page);

      const workflow = page.getByRole("combobox", { name: "Workflow" });
      const creativity = page.getByRole("combobox", { name: "Creativity" });
      await workflow.selectOption("multi_agent");
      await creativity.selectOption("exploratory");

      for (const state of visualStates) {
        await selectTheme(page, state.themeLabel, state.themeValue);
        await setDisplayMode(page, state.mode);
        await expectGenerationControlContract(page, viewport, state);
        await expectAttachmentMenuLifecycle(page, viewport);
        await expectProviderPopoverLifecycle(page, viewport);
      }

      await expectPreferencesStored(page);
      await page.reload();
      await expectLoadedWorkspaceAfterReload(page);
      await expect(workflow).toHaveValue("multi_agent");
      await expect(creativity).toHaveValue("exploratory");
      await expect(page.locator(".workstation")).toHaveAttribute("data-theme", "light");
      await expect(page.getByRole("form", { name: "Creative request composer" })).toHaveAttribute(
        "data-mode",
        "developer"
      );

      await expectUnsupportedUploadLifecycle(page);
      await expectDeferredUploadLifecycle(page, viewport);

      if (viewport.label === "desktop") {
        await expectRequestPayloadAndInFlightBoundary(page, streamGate);
      }

      consoleGate.assertClean();
    });
  }
});

async function expectGenerationControlContract(page, viewport, state) {
  const composer = page.getByRole("form", { name: "Creative request composer" });
  const controls = page.getByRole("group", { name: "Generation controls" });
  const workflow = page.getByRole("combobox", { name: "Workflow" });
  const creativity = page.getByRole("combobox", { name: "Creativity" });
  const provider = page.getByRole("button", {
    name: "Selected AI provider: OpenAI"
  });

  await expect(composer).toHaveAttribute("data-mode", state.mode.toLowerCase());
  await expect(page.locator(".workstation")).toHaveAttribute(
    "data-theme",
    state.themeValue
  );
  await expect(controls).toBeVisible();
  await expect(workflow).toHaveValue("multi_agent");
  await expect(creativity).toHaveValue("exploratory");
  await expect(provider).toBeEnabled();
  await expect(workflow).toBeEnabled();
  await expect(creativity).toBeEnabled();

  const geometry = await page.evaluate(() => {
    const read = (selector) => {
      const element = document.querySelector(selector);
      if (!(element instanceof HTMLElement)) {
        throw new Error(`Missing request-control geometry target: ${selector}`);
      }
      const rect = element.getBoundingClientRect();
      return {
        clientWidth: element.clientWidth,
        left: rect.left,
        right: rect.right,
        scrollWidth: element.scrollWidth,
        width: rect.width
      };
    };

    return {
      clientWidth: document.documentElement.clientWidth,
      composer: read(".workspaceComposer"),
      controls: read(".workspaceGenerationControls"),
      documentScrollWidth: document.documentElement.scrollWidth,
      viewportWidth: window.innerWidth
    };
  });

  expect(geometry.viewportWidth).toBe(viewport.width);
  expect(geometry.documentScrollWidth).toBeLessThanOrEqual(geometry.clientWidth);
  for (const target of [geometry.composer, geometry.controls]) {
    expect(target.width).toBeGreaterThan(120);
    expect(target.left).toBeGreaterThanOrEqual(0);
    expect(target.right).toBeLessThanOrEqual(viewport.width + 1);
    expect(target.scrollWidth).toBeLessThanOrEqual(target.clientWidth + 1);
  }
}

async function expectAttachmentMenuLifecycle(page, viewport) {
  const trigger = page.getByRole("button", { name: "Add attachment" });
  await trigger.focus();
  await trigger.press("ArrowDown");

  const menu = page.getByRole("menu", { name: "Attachment options" });
  const upload = menu.getByRole("menuitem", { name: /Upload image reference/ });
  const audio = menu.getByRole("menuitem", { name: /Audio input unavailable/ });
  await expect(menu).toBeVisible();
  await expect(trigger).toHaveAttribute("aria-expanded", "true");
  await expect(upload).toBeFocused();
  await expect(audio).toHaveAttribute("aria-disabled", "true");
  await expectPopupUnclipped(page, menu, viewport);

  await page.keyboard.press("Escape");
  await expect(menu).toHaveCount(0);
  await expect(trigger).toHaveAttribute("aria-expanded", "false");
  await expect(trigger).toBeFocused();
}

async function expectProviderPopoverLifecycle(page, viewport) {
  const trigger = page.getByRole("button", {
    name: "Selected AI provider: OpenAI"
  });
  await trigger.focus();
  await trigger.press("Enter");

  const popover = page.getByRole("region", { name: "AI provider configuration" });
  await expect(popover).toBeVisible();
  await expect(trigger).toHaveAttribute("aria-expanded", "true");
  await expect(popover).toContainText("Configured server-side");
  await expect(popover).toContainText("Credentials and live routing remain server-owned.");
  await expectPopupUnclipped(page, popover, viewport);

  await page.keyboard.press("Escape");
  await expect(popover).toHaveCount(0);
  await expect(trigger).toHaveAttribute("aria-expanded", "false");
  await expect(trigger).toBeFocused();
}

async function expectPopupUnclipped(page, popup, viewport) {
  const geometry = await popup.evaluate((element) => {
    const rect = element.getBoundingClientRect();
    const clippingAncestors = [];
    let ancestor = element.parentElement;

    while (ancestor) {
      const style = window.getComputedStyle(ancestor);
      const clipsX = style.overflowX !== "visible";
      const clipsY = style.overflowY !== "visible";
      if (clipsX || clipsY) {
        const ancestorRect = ancestor.getBoundingClientRect();
        clippingAncestors.push({
          bottom: ancestorRect.bottom,
          clipsX,
          clipsY,
          left: ancestorRect.left,
          right: ancestorRect.right,
          top: ancestorRect.top
        });
      }
      ancestor = ancestor.parentElement;
    }

    const style = window.getComputedStyle(element);
    return {
      backgroundColor: style.backgroundColor,
      borderRadius: Number.parseFloat(style.borderRadius),
      bottom: rect.bottom,
      clippingAncestors,
      height: rect.height,
      left: rect.left,
      right: rect.right,
      scrollHeight: element.scrollHeight,
      scrollWidth: element.scrollWidth,
      top: rect.top,
      width: rect.width
    };
  });

  expect(geometry.width).toBeGreaterThan(220);
  expect(geometry.height).toBeGreaterThan(80);
  expect(geometry.left).toBeGreaterThanOrEqual(0);
  expect(geometry.right).toBeLessThanOrEqual(viewport.width + 1);
  expect(geometry.top).toBeGreaterThanOrEqual(0);
  expect(geometry.bottom).toBeLessThanOrEqual(viewport.height + 1);
  expect.soft(geometry.scrollWidth).toBeLessThanOrEqual(Math.ceil(geometry.width));
  expect.soft(geometry.scrollHeight).toBeLessThanOrEqual(Math.ceil(geometry.height));
  expect(geometry.backgroundColor).not.toBe("rgba(0, 0, 0, 0)");
  expect(geometry.borderRadius).toBeGreaterThanOrEqual(10);

  for (const ancestor of geometry.clippingAncestors) {
    if (ancestor.clipsX) {
      expect.soft(geometry.left).toBeGreaterThanOrEqual(ancestor.left - 1);
      expect.soft(geometry.right).toBeLessThanOrEqual(ancestor.right + 1);
    }
    if (ancestor.clipsY) {
      expect.soft(geometry.top).toBeGreaterThanOrEqual(ancestor.top - 1);
      expect.soft(geometry.bottom).toBeLessThanOrEqual(ancestor.bottom + 1);
    }
  }
}

async function expectUnsupportedUploadLifecycle(page) {
  const trigger = page.getByRole("button", { name: "Add attachment" });
  await trigger.click();
  const chooserPromise = page.waitForEvent("filechooser");
  await page
    .getByRole("menu", { name: "Attachment options" })
    .getByRole("menuitem", { name: /Upload image reference/ })
    .click();
  const chooser = await chooserPromise;
  await chooser.setFiles({
    buffer: Buffer.from("not an image"),
    mimeType: "text/plain",
    name: "review-notes.txt"
  });

  const shelf = page.getByRole("region", { name: "Image references" });
  await expect(shelf).toHaveAttribute("data-state", "error");
  await expect(shelf).toContainText("Reference not added");
  await expect(shelf).toContainText("Only PNG, JPEG, WebP, or GIF");
  await expect(shelf).not.toContainText("review-notes.txt image reference");

  await shelf.getByRole("button", { name: "Dismiss image upload issue" }).click();
  await expect(shelf).toHaveCount(0);
  await expect(trigger).toBeFocused();
}

async function expectDeferredUploadLifecycle(page, viewport) {
  const composer = page.getByRole("form", { name: "Creative request composer" });
  const add = page.getByRole("button", { name: "Add attachment" });
  const send = page.getByRole("button", { name: "Send prompt" });
  const workflow = page.getByRole("combobox", { name: "Workflow" });
  const creativity = page.getByRole("combobox", { name: "Creativity" });
  const provider = page.getByRole("button", {
    name: "Selected AI provider: OpenAI"
  });

  await page.getByRole("textbox", { name: "Assistant prompt" }).fill(
    `Use the delayed ${viewport.label} reference without sending early.`
  );
  await beginDeferredImageUpload(page, `queued-${viewport.label}.png`);

  await expect(composer).toHaveAttribute("data-upload-state", "processing");
  await expect(composer).toHaveAttribute("aria-busy", "true");
  await expect(add).toBeDisabled();
  await expect(send).toBeDisabled();
  await expect(workflow).toBeEnabled();
  await expect(creativity).toBeEnabled();
  await expect(provider).toBeEnabled();
  await expect(page.getByText("Preparing image reference. Send is paused.")).toBeAttached();

  await page.evaluate(() => window.__resolveE2EImageRead?.());

  await expect(composer).toHaveAttribute("data-upload-state", "idle");
  await expect(composer).toHaveAttribute("aria-busy", "false");
  await expect(add).toBeEnabled();
  await expect(send).toBeEnabled();
  await expect(workflow).toBeEnabled();
  await expect(creativity).toBeEnabled();
  await expect(provider).toBeEnabled();

  const shelf = page.getByRole("region", { name: "Image references" });
  await expect(shelf).toHaveAttribute("data-state", "ready");
  await expect(shelf).toContainText(`queued-${viewport.label}.png`);
  await expect(shelf).toContainText("next explicit request");
  await expectShelfBounded(page, shelf, viewport);

  const remove = shelf.getByRole("button", {
    name: `Remove image reference queued-${viewport.label}.png`
  });
  await remove.click();
  await expect(shelf).toHaveCount(0);
  await expect(add).toBeFocused();
  await expect(composer).toHaveAttribute("data-has-images", "false");

  await attachImageFromChooser(page, `request-${viewport.label}.png`);
  await expect(page.getByRole("region", { name: "Image references" })).toHaveAttribute(
    "data-state",
    "ready"
  );
  await expect(composer).toHaveAttribute("data-has-images", "true");
}

async function expectRequestPayloadAndInFlightBoundary(page, streamGate) {
  const composer = page.getByRole("form", { name: "Creative request composer" });
  const add = page.getByRole("button", { name: "Add attachment" });
  const send = page.getByRole("button", { name: "Send prompt" });
  const workflow = page.getByRole("combobox", { name: "Workflow" });
  const creativity = page.getByRole("combobox", { name: "Creativity" });
  const provider = page.getByRole("button", {
    name: "Selected AI provider: OpenAI"
  });
  const shelf = page.getByRole("region", { name: "Image references" });

  await page.getByRole("textbox", { name: "Assistant prompt" }).fill(
    "Create a restrained orbit study using the queued image only as explicit visual guidance."
  );
  const requestPromise = page.waitForRequest("**/api/assistant/stream");
  const gate = streamGate.arm();
  const sendAction = send.click();
  await gate.waitUntilBlocked;
  const request = await requestPromise;

  await expect(composer).toHaveAttribute("aria-busy", "true");
  await expect(add).toBeDisabled();
  await expect(send).toBeDisabled();
  await expect(workflow).toBeDisabled();
  await expect(creativity).toBeDisabled();
  await expect(provider).toBeDisabled();
  await expect(shelf).toHaveCount(0);
  await expect(composer).toHaveAttribute("data-has-images", "false");

  const payload = request.postDataJSON();
  expect(payload.workflowMode).toBe("multi_agent");
  expect(payload.generationControls).toEqual({ profile: "exploratory" });
  expect(payload.attachments).toHaveLength(1);
  expect(payload.attachments[0]).toMatchObject({
    mimeType: "image/png",
    name: "request-desktop.png",
    sizeBytes: onePixelPng.length,
    type: "image"
  });
  expect(payload.attachments[0].dataUrl).toBe(
    `data:image/png;base64,${onePixelPng.toString("base64")}`
  );

  gate.release();
  await sendAction;
  await expect(page.locator('.workspaceMessage[data-role="assistant"]').last()).toHaveAttribute(
    "data-stream-phase",
    "completed"
  );
  await expect(composer).toHaveAttribute("aria-busy", "false");
  await expect(add).toBeEnabled();
  await expect(workflow).toBeEnabled();
  await expect(creativity).toBeEnabled();
  await expect(provider).toBeEnabled();
}

async function beginDeferredImageUpload(page, fileName) {
  const bytes = Array.from(onePixelPng);
  await page.evaluate(
    ({ bytes: imageBytes, name }) => {
      const input = document.querySelector('input[aria-label="Upload image attachment"]');
      if (!(input instanceof HTMLInputElement)) {
        throw new Error("Missing image attachment input.");
      }

      const file = new File([new Uint8Array(imageBytes)], name, {
        type: "image/png"
      });
      file.arrayBuffer = () =>
        new Promise((resolve) => {
          window.__resolveE2EImageRead = () => {
            resolve(new Uint8Array(imageBytes).buffer);
            delete window.__resolveE2EImageRead;
          };
        });

      const transfer = new DataTransfer();
      transfer.items.add(file);
      Object.defineProperty(input, "files", {
        configurable: true,
        value: transfer.files
      });
      input.dispatchEvent(new Event("change", { bubbles: true }));
      delete input.files;
    },
    { bytes, name: fileName }
  );
}

async function attachImageFromChooser(page, fileName) {
  await page.getByRole("button", { name: "Add attachment" }).click();
  const chooserPromise = page.waitForEvent("filechooser");
  await page
    .getByRole("menu", { name: "Attachment options" })
    .getByRole("menuitem", { name: /Upload image reference/ })
    .click();
  const chooser = await chooserPromise;
  await chooser.setFiles({
    buffer: onePixelPng,
    mimeType: "image/png",
    name: fileName
  });
}

async function expectShelfBounded(page, shelf, viewport) {
  const geometry = await shelf.evaluate((element) => {
    const rect = element.getBoundingClientRect();
    return {
      clientWidth: element.clientWidth,
      left: rect.left,
      right: rect.right,
      scrollWidth: element.scrollWidth,
      width: rect.width
    };
  });

  expect(geometry.width).toBeGreaterThan(200);
  expect(geometry.left).toBeGreaterThanOrEqual(0);
  expect(geometry.right).toBeLessThanOrEqual(viewport.width + 1);
  expect(geometry.scrollWidth).toBeLessThanOrEqual(geometry.clientWidth + 1);
  const card = shelf.getByRole("listitem");
  await expect(card).toBeVisible();
  expect(await card.evaluate((element) => element.scrollWidth <= element.clientWidth + 1)).toBe(
    true
  );
}

async function expectPreferencesStored(page) {
  await expect
    .poll(() =>
      page.evaluate(() =>
        Object.values(localStorage).some((rawValue) => {
          try {
            const value = JSON.parse(rawValue);
            return (
              value?.preferences?.workflowMode === "multi_agent" &&
              value?.preferences?.creativity === "exploratory" &&
              value?.preferences?.theme === "light" &&
              value?.preferences?.showDebugPanels === true
            );
          } catch {
            return false;
          }
        })
      )
    )
    .toBe(true);
}

async function expectLoadedWorkspaceAfterReload(page) {
  await expect(page.getByRole("region", { name: "Creative workspace" })).toBeVisible();
  await expect(page.getByRole("region", { name: "Creative session" })).toBeVisible();
  await expect(page.getByRole("textbox", { name: "Assistant prompt" })).toBeVisible();
}

async function installOneShotStreamGate(page) {
  let armed = false;
  let blockedResolve;
  let releaseRequest;

  await page.route("**/api/assistant/stream", async (route, request) => {
    if (request.method() !== "POST" || !armed) {
      await route.fallback();
      return;
    }

    armed = false;
    blockedResolve();
    await new Promise((resolve) => {
      releaseRequest = resolve;
    });
    releaseRequest = undefined;
    await route.fallback();
  });

  return {
    arm() {
      if (armed) {
        throw new Error("The controlled assistant stream is already armed.");
      }
      armed = true;
      const waitUntilBlocked = new Promise((resolve) => {
        blockedResolve = resolve;
      });
      return {
        release() {
          if (!releaseRequest) {
            throw new Error("The controlled assistant stream has not reached the gate.");
          }
          releaseRequest();
        },
        waitUntilBlocked
      };
    }
  };
}

async function selectTheme(page, themeLabel, themeValue) {
  if ((await page.locator(".workstation").getAttribute("data-theme")) === themeValue) {
    return;
  }
  await page.getByRole("button", { name: "Theme" }).click();
  const presets = page.getByRole("dialog", { name: "Theme presets" });
  await expect(presets).toBeVisible();
  await presets.getByRole("button", { name: `Use ${themeLabel} theme` }).click();
  await expect(presets).toHaveCount(0);
  await expect(page.locator(".workstation")).toHaveAttribute("data-theme", themeValue);
  await expect(page.locator("html")).toHaveAttribute("data-cca-theme", themeValue);
}

async function setDisplayMode(page, targetMode) {
  const expectedMode = targetMode.toLowerCase();
  const composer = page.getByRole("form", { name: "Creative request composer" });
  if ((await composer.getAttribute("data-mode")) === expectedMode) {
    return;
  }

  await page.getByRole("button", { name: "Settings", exact: true }).click();
  const settings = page.getByRole("dialog", { name: "Workspace settings" });
  await expect(settings).toBeVisible();
  const displayMode = settings.getByRole("button", { name: "Display mode" });
  await displayMode.click();
  await expect(displayMode).toHaveText(targetMode);
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  await expect(settings).toHaveCount(0);
  await expect(composer).toHaveAttribute("data-mode", expectedMode);
}
