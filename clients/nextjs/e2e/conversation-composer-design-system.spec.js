const { test, expect } = require("@playwright/test");
const {
  expectLoadedWorkstation,
  installApiMocks,
  installConsoleGate
} = require("./support/quality-gates");

const conversationViewports = [
  { height: 900, label: "desktop", width: 1440 },
  { height: 768, label: "laptop", width: 1024 },
  { height: 900, label: "compact", width: 720 }
];

const populatedPrompts = [
  "Build a layered orbit field with deliberate visual hierarchy, generous negative space, and a calm motion system that remains legible throughout the full creative workspace.",
  "Refine the same composition with a restrained luminous palette, clearer depth cues, and a stable browser-native runtime while preserving the original interaction language.",
  "Complete the study with a subtle responsive rhythm, accessible contrast, and enough structural detail for a reviewer to understand the creative intent at a glance."
];

test.describe("Conversation and composer design-system boundary", () => {
  for (const viewport of conversationViewports) {
    test(`${viewport.label} keeps a populated conversation coherent across themes and modes`, async ({
      page
    }) => {
      await page.setViewportSize({ height: viewport.height, width: viewport.width });
      const consoleGate = installConsoleGate(page);
      await installApiMocks(page, "long");
      const streamGate = await installOneShotStreamGate(page);
      await expectLoadedWorkstation(page);

      await selectTheme(page, "Deep Blue", "codex");
      await setDisplayMode(page, "Developer");
      await expectComposerInputContract(page);

      const gate = streamGate.arm();
      const firstExchange = page.locator(".workspaceMessage");
      const textarea = page.getByRole("textbox", { name: "Assistant prompt" });
      await textarea.fill(populatedPrompts[0]);
      await expect(page.getByRole("button", { name: "Send prompt" })).toBeEnabled();
      const enterAction = textarea.press("Enter");
      await gate.waitUntilBlocked;
      await expect(page.getByRole("form", { name: "Creative request composer" })).toHaveAttribute(
        "aria-busy",
        "true"
      );
      await expect(page.getByRole("button", { name: "Send prompt" })).toBeDisabled();
      gate.release();
      await enterAction;
      await expectCompletedExchange(page, firstExchange, 0);
      await closePreviewIfOpen(page);

      for (const prompt of populatedPrompts.slice(1)) {
        await submitPromptAndWait(page, prompt);
        await closePreviewIfOpen(page);
      }

      await expectMessageRolesAndBounds(page, viewport);
      await expectPinnedComposer(page, viewport);
      if (viewport.width >= 761) {
        await expectConversationScrollContract(page, viewport.label);
      }

      for (const state of [
        { mode: "Developer", themeLabel: "Deep Blue", themeValue: "codex" },
        { mode: "User", themeLabel: "Deep Blue", themeValue: "codex" },
        { mode: "User", themeLabel: "Light", themeValue: "light" },
        { mode: "Developer", themeLabel: "Light", themeValue: "light" }
      ]) {
        await selectTheme(page, state.themeLabel, state.themeValue);
        await setDisplayMode(page, state.mode);
        await expectConversationBoundary(page, viewport, state);
        await expectComposerFocusTreatment(page);
      }

      consoleGate.assertClean();
    });
  }
});

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

async function expectComposerInputContract(page) {
  const composer = page.getByRole("form", { name: "Creative request composer" });
  const textarea = page.getByRole("textbox", { name: "Assistant prompt" });
  const send = page.getByRole("button", { name: "Send prompt" });

  await expect(composer).toHaveAttribute("data-mode", "developer");
  await expect(composer).toHaveAttribute("data-ready", "false");
  await expect(send).toBeDisabled();

  await textarea.fill("   ");
  await expect(composer).toHaveAttribute("data-ready", "false");
  await expect(send).toBeDisabled();

  await textarea.fill("Keep this line and add another intentionally.");
  await textarea.press("Shift+Enter");
  await expect(textarea).toHaveValue(/\n$/);
  await expect(composer).toHaveAttribute("data-ready", "true");
  await expect(send).toBeEnabled();

  await textarea.fill("");
  const baseline = await readTextareaMetrics(textarea);

  await textarea.fill("Line one\nLine two\nLine three\nLine four\nLine five");
  await expect
    .poll(async () => (await readTextareaMetrics(textarea)).height)
    .toBeGreaterThan(baseline.height + 2);
  const grown = await readTextareaMetrics(textarea);
  expect(grown.height).toBeLessThan(168);
  expect(grown.overflowY).toBe("hidden");

  await textarea.fill(Array.from({ length: 24 }, (_, index) => `Detail ${index + 1}`).join("\n"));
  await expect
    .poll(async () => (await readTextareaMetrics(textarea)).overflowY)
    .toBe("auto");
  const clamped = await readTextareaMetrics(textarea);
  expect(clamped.height).toBeLessThanOrEqual(169);
  expect(clamped.scrollHeight).toBeGreaterThan(clamped.clientHeight);

  await textarea.fill("");
  await expect
    .poll(async () => (await readTextareaMetrics(textarea)).height)
    .toBeLessThanOrEqual(baseline.height + 1);
  const reset = await readTextareaMetrics(textarea);
  expect(reset.height).toBeGreaterThanOrEqual(baseline.height - 1);
  expect(reset.overflowY).toBe("hidden");
  await expect(send).toBeDisabled();
}

async function readTextareaMetrics(textarea) {
  return textarea.evaluate((element) => {
    const rect = element.getBoundingClientRect();
    return {
      clientHeight: element.clientHeight,
      height: rect.height,
      overflowY: window.getComputedStyle(element).overflowY,
      scrollHeight: element.scrollHeight
    };
  });
}

async function submitPromptAndWait(page, prompt) {
  const messages = page.locator(".workspaceMessage");
  const initialCount = await messages.count();
  await page.getByRole("textbox", { name: "Assistant prompt" }).fill(prompt);
  await page.getByRole("button", { name: "Send prompt" }).click();
  await expectCompletedExchange(page, messages, initialCount);
}

async function expectCompletedExchange(page, messages, initialCount) {
  await expect(messages).toHaveCount(initialCount + 2);
  await expect(messages.nth(initialCount)).toHaveAttribute("data-role", "user");
  await expect(messages.nth(initialCount + 1)).toHaveAttribute("data-role", "assistant");
  await expect(messages.nth(initialCount + 1)).toHaveAttribute(
    "data-stream-phase",
    "completed"
  );
  await expect(page.getByRole("form", { name: "Creative request composer" })).toHaveAttribute(
    "aria-busy",
    "false"
  );
  await expect(page.getByRole("textbox", { name: "Assistant prompt" })).toHaveValue("");
  await expect(page.getByRole("button", { name: "Send prompt" })).toBeDisabled();
}

async function closePreviewIfOpen(page) {
  const shelf = page.locator(".mainColumn > .previewZone .previewShelf");
  if ((await shelf.count()) === 0) {
    return;
  }
  if ((await shelf.getAttribute("data-state")) === "open") {
    await shelf.locator(":scope > summary").click();
    await expect(shelf).toHaveAttribute("data-state", "closed");
  }
}

async function expectMessageRolesAndBounds(page, viewport) {
  const geometry = await readConversationGeometry(page);
  expect(geometry.messages.length).toBeGreaterThanOrEqual(6);
  expect(geometry.messages.filter((message) => message.role === "user")).not.toHaveLength(0);
  expect(geometry.messages.filter((message) => message.role === "assistant")).not.toHaveLength(0);

  for (const message of geometry.messages) {
    expect(message.left).toBeGreaterThanOrEqual(geometry.conversation.left - 1);
    expect(message.right).toBeLessThanOrEqual(geometry.conversation.right + 1);
    expect(message.scrollWidth).toBeLessThanOrEqual(message.clientWidth);
    expect(message.width).toBeGreaterThan(120);
    if (message.role === "user") {
      expect(message.justifySelf).toBe("end");
    } else {
      expect(message.justifySelf).toBe("start");
    }
  }

  expect(geometry.viewportWidth).toBe(viewport.width);
  expect(geometry.viewportHeight).toBe(viewport.height);
}

async function expectPinnedComposer(page, viewport) {
  const conversation = page.getByRole("log", { name: "Conversation" });
  const composer = page.getByRole("form", { name: "Creative request composer" });
  const before = await composer.boundingBox();

  if (viewport.width >= 761) {
    const overflow = await conversation.evaluate(
      (element) => element.scrollHeight - element.clientHeight
    );
    expect(overflow).toBeGreaterThan(24);
    await conversation.evaluate((element) => {
      element.scrollTop = 0;
      element.dispatchEvent(new Event("scroll"));
    });
  }

  const after = await composer.boundingBox();
  expect(before).not.toBeNull();
  expect(after).not.toBeNull();
  expect(Math.abs((after?.x ?? 0) - (before?.x ?? 0))).toBeLessThanOrEqual(1);
  expect(Math.abs((after?.y ?? 0) - (before?.y ?? 0))).toBeLessThanOrEqual(1);
  expect(Math.abs((after?.width ?? 0) - (before?.width ?? 0))).toBeLessThanOrEqual(1);
  expect(Math.abs((after?.height ?? 0) - (before?.height ?? 0))).toBeLessThanOrEqual(1);

  await conversation.evaluate((element) => {
    element.scrollTop = element.scrollHeight;
    element.dispatchEvent(new Event("scroll"));
  });
}

async function expectConversationScrollContract(page, viewportLabel) {
  const conversation = page.getByRole("log", { name: "Conversation" });
  await expect.poll(async () => distanceFromBottom(conversation)).toBeLessThanOrEqual(2);

  await conversation.evaluate((element) => {
    element.scrollTop = 0;
    element.dispatchEvent(new Event("scroll"));
  });
  await expect(page.getByRole("button", { name: "Jump to latest" })).toBeVisible();
  const preservedTop = await conversation.evaluate((element) => element.scrollTop);

  await submitPromptAndWait(
    page,
    `Keep this ${viewportLabel} review anchored while appending another deterministic conversation result.`
  );
  await closePreviewIfOpen(page);
  expect(await conversation.evaluate((element) => element.scrollTop)).toBeLessThanOrEqual(
    preservedTop + 4
  );

  await page.getByRole("button", { name: "Jump to latest" }).click();
  await expect.poll(async () => distanceFromBottom(conversation)).toBeLessThanOrEqual(2);

  await submitPromptAndWait(
    page,
    `Auto-follow the latest ${viewportLabel} result from the bottom of the conversation.`
  );
  await closePreviewIfOpen(page);
  await expect.poll(async () => distanceFromBottom(conversation)).toBeLessThanOrEqual(2);
}

async function distanceFromBottom(conversation) {
  return conversation.evaluate(
    (element) => element.scrollHeight - element.scrollTop - element.clientHeight
  );
}

async function expectConversationBoundary(page, viewport, state) {
  const composer = page.getByRole("form", { name: "Creative request composer" });
  await expect(composer).toHaveAttribute("data-mode", state.mode.toLowerCase());
  await expect(page.locator(".workstation")).toHaveAttribute("data-theme", state.themeValue);
  await expect(page.locator("html")).toHaveAttribute("data-cca-theme", state.themeValue);

  const geometry = await readConversationGeometry(page);
  expect(geometry.documentScrollWidth).toBeLessThanOrEqual(geometry.documentClientWidth);

  for (const [name, target] of [
    ["composer", geometry.composer],
    ["composer actions", geometry.actions],
    ["conversation", geometry.conversation],
    ["conversation frame", geometry.frame],
    ["session", geometry.session]
  ]) {
    const boundary = `${viewport.label} ${state.themeLabel} ${state.mode} ${name}`;
    expect(target.left, `${boundary} stays inside the left viewport edge`).toBeGreaterThanOrEqual(
      0
    );
    expect(target.right, `${boundary} stays inside the right viewport edge`).toBeLessThanOrEqual(
      viewport.width + 1
    );
    expect(target.scrollWidth, `${boundary} has no horizontal overflow`).toBeLessThanOrEqual(
      target.clientWidth
    );
  }

  expect(geometry.composer.left).toBeGreaterThanOrEqual(geometry.session.left - 1);
  expect(geometry.composer.right).toBeLessThanOrEqual(geometry.session.right + 1);
  expect(geometry.frame.bottom).toBeLessThanOrEqual(geometry.composer.top + 1);
  expect(geometry.actions.left).toBeGreaterThanOrEqual(geometry.composer.left - 1);
  expect(geometry.actions.right).toBeLessThanOrEqual(geometry.composer.right + 1);
  expect(geometry.send.left).toBeGreaterThanOrEqual(geometry.actions.left - 1);
  expect(geometry.send.right).toBeLessThanOrEqual(geometry.actions.right + 1);

  for (const message of geometry.messages) {
    expect(message.left).toBeGreaterThanOrEqual(geometry.conversation.left - 1);
    expect(message.right).toBeLessThanOrEqual(geometry.conversation.right + 1);
    expect(message.scrollWidth).toBeLessThanOrEqual(message.clientWidth);
  }
}

async function readConversationGeometry(page) {
  return page.evaluate(() => {
    const read = (selector) => {
      const element = document.querySelector(selector);
      if (!(element instanceof HTMLElement)) {
        throw new Error(`Missing conversation geometry target: ${selector}`);
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

    const conversation = read(".workspaceConversation");
    return {
      actions: read(".workspaceComposerActions"),
      composer: read(".workspaceComposer"),
      conversation,
      documentClientWidth: document.documentElement.clientWidth,
      documentScrollWidth: document.documentElement.scrollWidth,
      frame: read(".workspaceConversationFrame"),
      messages: [...document.querySelectorAll(".workspaceMessage")].map((element) => {
        const rect = element.getBoundingClientRect();
        return {
          clientWidth: element.clientWidth,
          justifySelf: window.getComputedStyle(element).justifySelf,
          left: rect.left,
          right: rect.right,
          role: element.dataset.role,
          scrollWidth: element.scrollWidth,
          width: rect.width
        };
      }),
      send: read(".workspaceComposerSend"),
      session: read(".sessionPanel"),
      viewportHeight: window.innerHeight,
      viewportWidth: window.innerWidth
    };
  });
}

async function expectComposerFocusTreatment(page) {
  const composer = page.getByRole("form", { name: "Creative request composer" });
  const textarea = page.getByRole("textbox", { name: "Assistant prompt" });
  await textarea.evaluate((element) => element.blur());
  const unfocused = await readComposerFocusStyle(composer);

  await textarea.focus();
  await expect(textarea).toBeFocused();
  await expect
    .poll(async () => JSON.stringify(await readComposerFocusStyle(composer)))
    .not.toBe(JSON.stringify(unfocused));
  expect(await composer.evaluate((element) => element.matches(":focus-within"))).toBe(true);
}

async function readComposerFocusStyle(composer) {
  return composer.evaluate((element) => {
    const style = window.getComputedStyle(element);
    return { borderColor: style.borderColor, boxShadow: style.boxShadow };
  });
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
