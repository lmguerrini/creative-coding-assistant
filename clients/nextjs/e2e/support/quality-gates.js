const { expect } = require("@playwright/test");

const corsHeaders = {
  "Access-Control-Allow-Headers": "Accept, Content-Type",
  "Access-Control-Allow-Methods": "DELETE, GET, OPTIONS, POST",
  "Access-Control-Allow-Origin": "*"
};

const generatedArtifact = {
  id: "e2e-orbit-sketch",
  title: "e2e-orbit-sketch.p5.js",
  type: "code",
  language: "javascript",
  runtime: "p5",
  renderer_id: "surface.p5",
  preview_eligible: true,
  preview_target: "browser_sandbox",
  status: "Generated",
  summary: "E2E smoke p5 orbit field with deterministic preview routing.",
  content: [
    "let phase = 0;",
    "function setup() {",
    "  createCanvas(windowWidth, windowHeight);",
    "  pixelDensity(1);",
    "  colorMode(HSL, 360, 100, 100, 1);",
    "  noiseDetail(3, 0.5);",
    "  noStroke();",
    "}",
    "function draw() {",
    "  phase += 0.012;",
    "  background(220, 38, 8, 0.14);",
    "  for (let i = 0; i < 18; i += 1) {",
    "    const x = map(i, 0, 17, 32, width - 32);",
    "    const angle = noise(i * 0.12, phase) * TWO_PI;",
    "    fill(168 + i * 4, 72, 64, 0.76);",
    "    push();",
    "    translate(x, height * 0.5 + sin(phase + i) * 64);",
    "    rotate(angle);",
    "    beginShape();",
    "    vertex(-9, -4);",
    "    vertex(12, 0);",
    "    vertex(-9, 4);",
    "    endShape(CLOSE);",
    "    pop();",
    "  }",
    "}"
  ].join("\n")
};

const retrievalSource = {
  id: "e2e-source-p5-reference",
  title: "p5.js createCanvas reference",
  url: "https://p5js.org/reference/p5/createCanvas/",
  domain: "p5js",
  score: 0.92,
  excerpt: "createCanvas initializes the sketch drawing surface."
};

function installConsoleGate(page) {
  const errors = [];

  page.on("console", (message) => {
    if (
      message.type() === "error" &&
      message.text().includes("Failed to load resource: the server responded with a status of 404")
    ) {
      return;
    }
    if (message.type() === "error") {
      errors.push(`console:${message.text()}`);
    }
  });
  page.on("response", (response) => {
    if (response.status() < 400) {
      return;
    }
    if (
      response.status() === 404 &&
      response.url().includes("/api/workspace/session")
    ) {
      return;
    }
    errors.push(`http:${response.status()} ${response.url()}`);
  });
  page.on("pageerror", (error) => {
    errors.push(`pageerror:${error.message}`);
  });
  page.on("requestfailed", (request) => {
    const failureText = request.failure()?.errorText ?? "unknown failure";
    if (failureText.includes("net::ERR_ABORTED")) {
      return;
    }
    errors.push(`requestfailed:${request.method()} ${request.url()} ${failureText}`);
  });

  return {
    assertClean() {
      expect(errors).toEqual([]);
    },
    errors
  };
}

async function installApiMocks(page, scenario = "success") {
  await page.route("**/api/workspace/session**", handleWorkspaceSessionRoute);
  await page.route("**/api/assistant/stream", async (route, request) => {
    if (request.method() === "OPTIONS") {
      await fulfillOptions(route);
      return;
    }

    if (scenario === "failure") {
      await route.fulfill({
        body: buildFailureNdjson(),
        contentType: "application/x-ndjson",
        headers: corsHeaders,
        status: 200
      });
      return;
    }

    if (scenario === "provider-fallback") {
      await route.fulfill({
        body: buildProviderFallbackNdjson(),
        contentType: "application/x-ndjson",
        headers: corsHeaders,
        status: 200
      });
      return;
    }

    await route.fulfill({
      body: buildAssistantNdjson(scenario),
      contentType: "application/x-ndjson",
      headers: corsHeaders,
      status: 200
    });
  });
}

async function expectLoadedWorkstation(page) {
  await page.goto("/");
  await expect(page.getByRole("region", { name: "Creative workspace" })).toBeVisible();
  await expect(page.getByRole("region", { name: "Creative session" })).toBeVisible();
  await expect(page.getByRole("complementary", { name: "Right inspector" })).toBeVisible();
  await expect(page.getByRole("textbox", { name: "Assistant prompt" })).toBeVisible();
}

async function submitCreativePrompt(page, prompt) {
  await page.getByRole("textbox", { name: "Assistant prompt" }).fill(prompt);
  await page.getByRole("button", { name: "Send prompt" }).click();
}

async function expectGeneratedPreview(page) {
  await expect(page.getByRole("region", { name: "Preview workspace" })).toBeVisible();
  await expandInspectorIfCollapsed(page);
  await expect(page.getByRole("tab", { name: "Preview" })).toHaveAttribute(
    "aria-selected",
    "true"
  );
  await expect(page.getByRole("tabpanel", { name: "Preview inspector" })).toContainText(
    "P5 sketch surface"
  );
  await expect(page.getByRole("tabpanel", { name: "Preview inspector" })).not.toContainText(
    "e2e-orbit-sketch.p5.js"
  );
  await expect(page.getByRole("tab", { name: "Saved" })).toBeVisible();
  await page.getByRole("tab", { name: "Saved" }).click();
  await expect(page.getByRole("tabpanel", { name: "Saved outputs inspector" })).toContainText(
    "P5 Sketch"
  );
  await expect(page.getByRole("tabpanel", { name: "Saved outputs inspector" })).not.toContainText(
    "e2e-orbit-sketch.p5.js"
  );
  await page.getByRole("tab", { name: "Code" }).click();
  await expect(page.getByRole("tabpanel", { name: "Code inspector" })).toContainText(
    "createCanvas"
  );
}

async function expectWorkspacePersistence(page) {
  await page.getByRole("button", { name: "Settings" }).click();
  await page.getByRole("button", { name: "Use Matrix theme" }).click();
  await page.getByRole("button", { name: "Compact" }).click();
  await expect(page.locator(".workstation")).toHaveAttribute("data-theme", "matrix");
  await page.reload();
  await expect(page.locator(".workstation")).toHaveAttribute("data-theme", "matrix");
  await expect(page.locator(".workstation")).toHaveAttribute("data-density", "compact");
}

async function expectRetrievalRegressionSurface(page) {
  await switchToDeveloperMode(page);
  await page.getByRole("tab", { name: "Retrieval" }).click();
  await expect(page.getByRole("tabpanel", { name: "Retrieval inspector" })).toContainText(
    "p5.js createCanvas reference"
  );
}

async function expectStableVisualLayout(page) {
  const workspace = page.getByRole("region", { name: "Creative workspace" });
  const session = page.getByRole("region", { name: "Creative session" });
  const inspector = page.getByRole("complementary", { name: "Right inspector" });
  await expect(workspace).toBeVisible();
  await expect(session).toBeVisible();
  await expect(inspector).toBeVisible();

  const [workspaceBox, sessionBox, inspectorBox] = await Promise.all([
    workspace.boundingBox(),
    session.boundingBox(),
    inspector.boundingBox()
  ]);
  for (const box of [workspaceBox, sessionBox]) {
    expect(box?.width).toBeGreaterThan(240);
    expect(box?.height).toBeGreaterThan(240);
  }
  expect(inspectorBox?.height).toBeGreaterThan(240);
  const inspectorState = await inspector.getAttribute("data-state");
  expect(inspectorBox?.width).toBeGreaterThan(inspectorState === "collapsed" ? 40 : 240);
}

async function expandInspectorIfCollapsed(page) {
  const expandInspector = page.getByRole("button", { name: "Expand inspector" });
  if (await expandInspector.isVisible().catch(() => false)) {
    await expandInspector.click();
  }
}

async function switchToDeveloperMode(page) {
  const displayMode = page.getByRole("button", { name: "Display mode" });
  const label = await displayMode.textContent();
  if (label?.includes("User")) {
    await displayMode.click();
  }
  await expect(displayMode).toContainText("Developer");
}

async function handleWorkspaceSessionRoute(route, request) {
  if (request.method() === "OPTIONS") {
    await fulfillOptions(route);
    return;
  }

  if (request.method() === "GET") {
    await route.fulfill({
      body: JSON.stringify({
        error: {
          type: "session_not_found",
          message: "No remote E2E workspace session exists."
        }
      }),
      contentType: "application/json",
      headers: corsHeaders,
      status: 404
    });
    return;
  }

  if (request.method() === "POST") {
    await route.fulfill({
      body: JSON.stringify({ ok: true, target: "e2e-memory" }),
      contentType: "application/json",
      headers: corsHeaders,
      status: 200
    });
    return;
  }

  await route.fulfill({
    body: JSON.stringify({ error: "method not allowed" }),
    contentType: "application/json",
    headers: corsHeaders,
    status: 405
  });
}

async function fulfillOptions(route) {
  await route.fulfill({
    body: "",
    headers: corsHeaders,
    status: 204
  });
}

function buildAssistantNdjson(scenario) {
  const events = [
    streamEvent("status", 0, {
      message: "E2E request received.",
      status: "request_received",
      workflow: { current_step: "intake" }
    }),
    streamEvent("retrieval", 1, {
      code: "retrieval_requested",
      message: "E2E retrieval requested.",
      request: {
        filters: {
          domains: ["p5_js"]
        },
        limit: 3,
        query: "p5 orbit field"
      },
      workflow: { current_step: "retrieval" }
    }),
    streamEvent("retrieval", 2, {
      code: "retrieval_completed",
      context: {
        chunks: [
          {
            chunk_index: 0,
            domain: "p5_js",
            domain_match: true,
            excerpt: retrievalSource.excerpt,
            original_score: 0.9,
            publisher: "p5.js",
            rank: 1,
            registry_title: retrievalSource.title,
            resolved_url: retrievalSource.url,
            score: retrievalSource.score,
            score_adjustment: 0.02,
            selection_reason: "Selected for E2E retrieval regression coverage.",
            source_health: {
              availability: "available",
              checked_at: new Date(0).toISOString(),
              domain_owner: "p5.js",
              freshness_status: "fresh",
              health_status: "healthy",
              refresh_recommended: false
            },
            source_id: retrievalSource.id,
            source_type: "api_reference",
            source_url: retrievalSource.url,
            used_in_context: true
          }
        ],
        request: {
          filters: {
            domains: ["p5_js"]
          },
          limit: 3,
          query: "p5 orbit field"
        },
        source: "official_kb"
      },
      emitted_at: new Date(0).toISOString(),
      message: "E2E retrieval completed.",
      workflow: { current_step: "retrieval" }
    }),
    streamEvent("planning", 3, {
      message: "E2E creative plan prepared.",
      status: "creative_plan_prepared",
      creative_plan: {
        outputModality: "visual",
        generationStrategy: "Generate one deterministic p5 candidate for E2E smoke.",
        recommendedRuntime: "p5",
        recommendedRendererId: "surface.p5",
        recommendedPreviewTarget: "browser_sandbox",
        recommendedShaderStyle: "glow",
        candidateCount: 1,
        refinementBudget: 1,
        expectedComplexity: "medium",
        estimatedTokenCost: 1800,
        exportReadiness: "ready",
        runtimeAvailable: true,
        runtimeSupportSummary: "p5 browser preview is available.",
        planSteps: ["Use p5 setup/draw.", "Keep the smoke artifact deterministic."],
        constraints: ["No external assets."],
        evidence: ["E2E mocked stream"]
      },
      workflow: { current_step: "planning" }
    }),
    streamEvent("artifact_extracted", 4, {
      message: "E2E artifact extracted.",
      status: "artifact_extracted",
      artifacts: [generatedArtifact],
      workflow: { current_step: "artifact_extraction" }
    }),
    streamEvent("preview_artifact", 5, {
      artifact_id: generatedArtifact.id,
      emitted_at: new Date(0).toISOString(),
      result: {
        completed_at: new Date(1).toISOString(),
        preview_artifact_id: generatedArtifact.id,
        request: {
          target: "browser_sandbox"
        },
        provenance: {
          renderer_id: "surface.p5"
        },
        summary: "E2E preview artifact prepared."
      },
      status: "succeeded",
      workflow: { current_step: "preview_preparation" }
    }),
    streamEvent("final", 6, {
      answer: "Generated the E2E p5 orbit sketch with preview routing.",
      artifacts: [generatedArtifact],
      workflow: { current_step: "finalization" }
    })
  ];

  if (scenario === "long") {
    events.splice(
      3,
      0,
      streamEvent("token_delta", 30, {
        text: "Layering orbit fields. ",
        workflow: { current_step: "generation" }
      }),
      streamEvent("token_delta", 31, {
        text: "Preparing deterministic preview. ",
        workflow: { current_step: "generation" }
      })
    );
  }

  return `${events.map((event) => JSON.stringify(event)).join("\n")}\n`;
}

function buildFailureNdjson() {
  return `${[
    streamEvent("status", 0, {
      message: "E2E request received.",
      status: "request_received",
      workflow: { current_step: "intake" }
    }),
    streamEvent("error", 1, {
      code: "assistant_stream_failed",
      message: "The live response stopped before completion.",
      recoverable: true,
      workflow: { current_step: "failure" }
    })
  ]
    .map((event) => JSON.stringify(event))
    .join("\n")}\n`;
}

function buildProviderFallbackNdjson() {
  return `${[
    streamEvent("status", 0, {
      message: "E2E request received.",
      status: "request_received",
      workflow: { current_step: "intake" }
    }),
    streamEvent("status", 1, {
      message: "Primary provider unavailable; using bounded local fallback.",
      status: "provider_fallback_selected",
      workflow: { current_step: "generation" }
    }),
    streamEvent("final", 2, {
      answer:
        "Provider fallback completed with a local draft while preserving the workspace session.",
      workflow: { current_step: "finalization" }
    })
  ]
    .map((event) => JSON.stringify(event))
    .join("\n")}\n`;
}

function streamEvent(eventType, sequence, payload) {
  return {
    event_type: eventType,
    payload,
    sequence
  };
}

module.exports = {
  expectGeneratedPreview,
  expectLoadedWorkstation,
  expectRetrievalRegressionSurface,
  expectStableVisualLayout,
  expectWorkspacePersistence,
  installApiMocks,
  installConsoleGate,
  submitCreativePrompt
};
