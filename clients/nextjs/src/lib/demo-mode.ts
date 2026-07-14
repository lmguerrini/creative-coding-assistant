import {
  boundedP5DemoSurfaceContract,
  demoShowcasePromptLibrary,
  rhythmicLineStudyPrompt
} from "./curated-prompt-library";
import type { WorkflowExecutionMode } from "./workflow-execution";

export type DemoModeScenario = {
  id: string;
  title: string;
  concept: string;
  purpose: string;
  description: string;
  category: string;
  runtime: string;
  workflowMode: WorkflowExecutionMode;
  workflow: string;
  inputRequirement: string;
  requiresImageAttachment?: boolean;
  prompt: string;
  expectedArtifact: string;
  expectedPreview: string;
  expectedInteraction: string;
  expectedValidation: string;
  fallback: string;
  estimatedGenerationTime: string;
  providerRequirement: string;
  retrievalRequirement: string;
  sourceBoundary: string;
};

export type DemoModeRecommendation = {
  role: string;
  scenarioId: DemoModeScenario["id"];
  title: string;
  rationale: string;
};

const [
  audioShowcasePrompt,
  p5ShowcasePrompt,
  threeShowcasePrompt,
  glslShowcasePrompt
] = demoShowcasePromptLibrary;

export const demoModeScenarioCatalog = [
  {
    id: "cymatic-chladni-audiovisual",
    title: audioShowcasePrompt.title,
    concept: audioShowcasePrompt.concept,
    purpose: "Open with a memorable audio-visual system that remains safe in a silent room.",
    description:
      "A complementary two-voice score drives a polished spectrum while remaining silent until the presenter opts in.",
    category: "Audio-visual browser scene",
    runtime: audioShowcasePrompt.runtime,
    workflowMode: "single_agent",
    workflow: "Single-agent executable audio artifact",
    inputRequirement: "No upload is needed; audio stays optional and muted until the presenter chooses Start audio.",
    prompt: audioShowcasePrompt.prompt,
    expectedArtifact: audioShowcasePrompt.expectedArtifact,
    expectedPreview:
      "Muted constellation spectrum on load; controlled Tone.js playback after Start audio.",
    expectedInteraction:
      "Use Start audio for optional playback, try fullscreen, then ask for a slower tempo variation. Stop or Mute before leaving; no microphone is requested.",
    expectedValidation:
      "Verify generation, artifact identity, parsed Tone runtime, silent-first preview, fullscreen, tempo follow-up, visual quality, controls, and reload.",
    fallback: audioShowcasePrompt.fallback,
    estimatedGenerationTime: "Target: 90 seconds for generation and 30 seconds for the muted preview check.",
    providerRequirement: "Configured provider for a live generation run",
    retrievalRequirement: "Optional runtime guidance; no microphone or uploaded audio",
    sourceBoundary:
      "The visual maps the parsed score and tempo; it does not analyze live microphone input or claim scientific measurement."
  },
  {
    id: "physarum-p5-hero",
    title: p5ShowcasePrompt.title,
    concept: p5ShowcasePrompt.concept,
    purpose: "Show the quickest path from a clear prompt to a living browser sketch.",
    description:
      "A golden-angle garden combines compact recursive-looking growth, luminous trails, and immediate pointer parallax.",
    category: "Generative browser sketch",
    runtime: p5ShowcasePrompt.runtime,
    workflowMode: "single_agent",
    workflow: "Single-agent runnable-code generation",
    inputRequirement: "No upload is needed; the pointer is the only live input.",
    prompt: p5ShowcasePrompt.prompt,
    expectedArtifact: p5ShowcasePrompt.expectedArtifact,
    expectedPreview: "One running p5.js canvas with an animated aurora garden and pointer parallax.",
    expectedInteraction: "Move the pointer, try fullscreen, then ask for a colder palette variation.",
    expectedValidation:
      "Verify generation, artifact identity, visible p5 runtime, pointer preview, fullscreen, palette follow-up, visual quality, and reload.",
    fallback: p5ShowcasePrompt.fallback,
    estimatedGenerationTime: "Target: 90 seconds for generation and 30 seconds for the canvas check.",
    providerRequirement: "Configured provider for a live generation run",
    retrievalRequirement: "Optional creative-code source grounding",
    sourceBoundary:
      "The recursive growth is an artistic construction, not a biological-growth simulation."
  },
  {
    id: "kinetic-three-hero",
    title: "Kinetic orbit sculpture",
    concept: threeShowcasePrompt.concept,
    purpose: "Show a spatial, browser-native hero piece with a truthful Three.js route.",
    description:
      "A self-contained scene uses authored geometry, nested parent transforms, lights, and a moving camera in the real bundled Three.js runtime.",
    category: "3D browser scene",
    runtime: threeShowcasePrompt.runtime,
    workflowMode: "single_agent",
    workflow: "Single-agent runnable-code generation",
    inputRequirement: "No upload is needed; fullscreen is the optional presentation interaction.",
    prompt: threeShowcasePrompt.prompt,
    expectedArtifact: threeShowcasePrompt.expectedArtifact,
    expectedPreview: "One nonblank animated WebGL scene whose sculpture, orbit rigs, parent transforms, lights, and camera motion are all visible.",
    expectedInteraction: "Use fullscreen, then ask for slower orbit motion; reload keeps the selected artifact.",
    expectedValidation:
      "Verify generation, artifact identity, bundled Three.js r176, nonblank dynamic frame evidence, authored parent/camera transforms, fullscreen, motion follow-up, visual quality, and reload.",
    fallback: threeShowcasePrompt.fallback,
    estimatedGenerationTime: "Target: 90 seconds for generation and 45 seconds for fullscreen and reload.",
    providerRequirement: "Configured provider for a live generation run",
    retrievalRequirement: "Optional Three.js source grounding",
    sourceBoundary:
      "The local Three.js r176 runtime executes only the controlled JavaScript artifact. React Three Fiber, standalone HTML, and remote modules remain code/export-only."
  },
  {
    id: "chladni-glsl-hero",
    title: glslShowcasePrompt.title,
    concept: glslShowcasePrompt.concept,
    purpose: "Show technical depth through a compact, visibly running fragment shader.",
    description:
      "A compact analytical shader stays inside the documented WebGL 1 contract while producing a polished recursive-looking bloom.",
    category: "Shader browser scene",
    runtime: glslShowcasePrompt.runtime,
    workflowMode: "single_agent",
    workflow: "Single-agent runnable-code generation",
    inputRequirement: "No upload is needed; the shader runs from its bounded source alone.",
    prompt: glslShowcasePrompt.prompt,
    expectedArtifact: glslShowcasePrompt.expectedArtifact,
    expectedPreview: "One nonblank animated WebGL fragment field.",
    expectedInteraction: "Use fullscreen, then ask for a higher-contrast color variation; the shader has no hidden external controls.",
    expectedValidation:
      "Verify generation, artifact source contract, shader compile/link, nonblank runtime preview, fullscreen, color follow-up, visual quality, and reload.",
    fallback: glslShowcasePrompt.fallback,
    estimatedGenerationTime: "Target: 90 seconds for generation and 30 seconds for the frame check.",
    providerRequirement: "Configured provider for a live generation run",
    retrievalRequirement: "Optional GLSL source grounding",
    sourceBoundary:
      "This is a bounded WebGL 1 fragment field, not a texture pipeline or display-performance benchmark."
  },
  {
    id: "retrieval-grounded-design-brief",
    title: "Source-grounded design brief",
    concept: "Creative direction tied to retrieved project sources",
    purpose: "Explain how retrieval supports a creative decision without confusing it with persistent knowledge inventory.",
    description:
      "A source-grounded p5.js request makes retrieval evidence, the chosen browser boundary, and the resulting artifact visible in one honest flow.",
    category: "Retrieval workflow",
    runtime: "p5.js browser preview with retrieval",
    workflowMode: "auto",
    workflow: "Auto-routed retrieval-grounded runnable-code generation",
    inputRequirement: "No upload is needed; the request draws only on current-run retrieval when available.",
    prompt: `Use current-run retrieval only when sources exist. Return only one global .p5.js artifact named source-grounded-chladni.p5.js with setup() and draw(): a compact Chladni line field with pointer attraction and a source-boundary comment. ${boundedP5DemoSurfaceContract}`,
    expectedArtifact: "source-grounded-chladni.p5.js with current-run retrieval evidence",
    expectedPreview: "One controlled p5.js canvas; retrieval remains visible as evidence rather than an external execution claim.",
    expectedInteraction: "Move the pointer in the canvas, then open Retrieval in Developer Mode to inspect current-run source grounding.",
    expectedValidation:
      "Exact prompt source contract, retrieval diagnostics, visible p5 canvas, runtime health, and a truthful retrieval boundary.",
    fallback: "Use the visible source boundary and local retrieval status; do not invent citations or retrieval results.",
    estimatedGenerationTime: "Target: 90 seconds for generation and 30 seconds for retrieval inspection.",
    providerRequirement: "Configured provider for a live runnable artifact",
    retrievalRequirement: "Current-run retrieval when sources are available",
    sourceBoundary:
      "Persistent Knowledge Base inventory and request-scoped retrieval are separate product surfaces."
  },
  {
    id: "multi-agent-production-plan",
    title: "Multi-agent production plan",
    concept: "A bounded creative plan with explicit specialist roles",
    purpose: "Demonstrate the visible multi-agent workflow before requesting a final visual artifact.",
    description:
      "The scenario switches the composer to Multi-Agent and makes its roles visible while producing one bounded browser artifact rather than a plan with no deliverable.",
    category: "Agent workflow",
    runtime: "p5.js browser preview with multi-agent workflow evidence",
    workflowMode: "multi_agent",
    workflow: "Multi-agent runnable-code generation with visible route evidence",
    inputRequirement: "No upload is needed; the workflow selection is the input under demonstration.",
    prompt: `Use the Multi-Agent workflow. Return only one global .p5.js artifact named multi-agent-orbit-study.p5.js with setup() and draw(): a dark orbit study with pointer input and a comment naming researcher, creative director, generator, and reviewer. ${boundedP5DemoSurfaceContract}`,
    expectedArtifact: "multi-agent-orbit-study.p5.js with visible role-aware route evidence",
    expectedPreview: "One controlled p5.js canvas after the multi-agent route completes.",
    expectedInteraction: "Keep Developer Mode visible to inspect the selected execution route and agent roles, then move the pointer in the canvas.",
    expectedValidation:
      "Exact prompt workflow mode, route evidence, role comment, visible p5 canvas, and runtime health.",
    fallback: "Use Single-Agent for the direct line-study artifact if the multi-agent route is unavailable.",
    estimatedGenerationTime: "Target: 90 seconds for generation and 30 seconds for route inspection.",
    providerRequirement: "Configured provider for a live multi-agent artifact",
    retrievalRequirement: "Optional source grounding based on the active request",
    sourceBoundary:
      "The plan describes the application's own workflow; it does not execute external creative tools."
  },
  {
    id: "single-agent-line-study",
    title: "Single-agent line study",
    concept: rhythmicLineStudyPrompt.concept,
    purpose: "Contrast a fast, direct generation path with multi-agent planning.",
    description:
      "A small generative line study makes the selected Single-Agent route easy to explain and compare.",
    category: "Agent workflow",
    runtime: rhythmicLineStudyPrompt.runtime,
    workflowMode: "single_agent",
    workflow: "Single-agent runnable-code generation",
    inputRequirement: "No upload is needed; the pointer is the optional canvas interaction.",
    prompt: rhythmicLineStudyPrompt.prompt,
    expectedArtifact: rhythmicLineStudyPrompt.expectedArtifact,
    expectedPreview: "One controlled p5.js line study.",
    expectedInteraction: "Move the pointer to disturb the line field.",
    expectedValidation:
      "Exact prompt route, global-mode source contract, visible p5 canvas, and runtime health.",
    fallback: rhythmicLineStudyPrompt.fallback,
    estimatedGenerationTime: "Target: 90 seconds for generation and 30 seconds for the canvas check.",
    providerRequirement: "Configured provider for a live generation run",
    retrievalRequirement: "Optional creative-code source grounding",
    sourceBoundary:
      "This is an original prompt pack direction, not a claim about an external artist's implementation."
  },
  {
    id: "export-handoff-package",
    title: "Export handoff package",
    concept: "An inspectable external-tool handoff",
    purpose: "Show that a strong export is different from a live browser preview.",
    description:
      "The generated brief can be exported as a package with implementation notes and validation checklist for a supported external handoff.",
    category: "Export workflow",
    runtime: "Code/export-only handoff",
    workflowMode: "auto",
    workflow: "Artifact generation followed by project-bundle export",
    inputRequirement: "No upload is needed; export is an explicit local operator action.",
    prompt:
      "Return only one fenced markdown block named chladni-touchdesigner-handoff.md. Write an audio-visual Chladni TouchDesigner handoff: concept, parameter names, web prototype boundary, implementation notes, validation, and fallback. Do not claim TouchDesigner runs here.",
    expectedArtifact:
      "chladni-touchdesigner-handoff.md and an inspectable exported project bundle",
    expectedPreview: "No internal TouchDesigner live preview; inspect Code and the export package.",
    expectedInteraction: "Open the artifact, choose export, and inspect the handoff files in the package.",
    expectedValidation:
      "Exact prompt artifact, project-bundle contents, explicit code/export-only route, and reload preservation.",
    fallback: "Use the browser-native Cymatic study instead of implying external execution.",
    estimatedGenerationTime: "Target: 60 seconds for the brief and 45 seconds for export inspection.",
    providerRequirement: "Configured provider for a live handoff brief",
    retrievalRequirement: "Optional external-domain guidance",
    sourceBoundary:
      "The application creates an inspectable handoff; it does not install, launch, or execute TouchDesigner."
  },
  {
    id: "multimodal-reference-study",
    title: "Reference-guided palette study",
    concept: "A supplied image reference guides palette and composition without becoming a live texture dependency",
    purpose: "Show the image-reference workflow while keeping the generated browser artifact self-contained.",
    description:
      "The presenter attaches a small reference image, then asks for a p5.js palette study that records the image as a creative cue rather than treating it as an executable asset.",
    category: "Multimodal creative workflow",
    runtime: "p5.js browser preview with image-reference context",
    workflowMode: "single_agent",
    workflow: "Single-agent image-guided runnable-code generation",
    inputRequirement: "Attach one PNG, JPEG, WebP, or GIF reference image before Send; do not attach private material for a public demo.",
    requiresImageAttachment: true,
    prompt: `Using the attached image only as palette and composition guidance, create exactly one global-mode .p5.js artifact named reference-palette-study.p5.js with setup() and draw(). Make a self-contained abstract field that never loads or embeds the image. ${boundedP5DemoSurfaceContract} Return only the artifact.`,
    expectedArtifact: "reference-palette-study.p5.js guided by the request-scoped image; no attachment record is persisted with the session",
    expectedPreview: "One self-contained p5.js canvas; the source must not fetch or embed the uploaded image.",
    expectedInteraction: "Attach the reference, inspect the image-reference status, then move the pointer over the generated canvas.",
    expectedValidation:
      "Attachment acceptance, image bytes included in the backend provider request payload, request-scoped metadata cleared after submission, exact prompt source contract, visible p5 canvas, and no attachment/session persistence. Provider receipt, use, and influence require separate live evidence. Reference metadata can enter a bundle only when export is explicitly requested before Send.",
    fallback: "Run the same palette-study prompt without an attachment and state that the result is text-guided rather than reference-guided.",
    estimatedGenerationTime: "Target: 90 seconds for generation and 45 seconds for attachment and canvas inspection.",
    providerRequirement: "Configured image-capable provider for live receipt, use, and influence evidence; local validation proves bounded request construction only",
    retrievalRequirement: "Not required; image-reference context remains separate from retrieval sources",
    sourceBoundary:
      "The attachment is request-scoped creative guidance and is cleared after submission rather than persisted in the session. The generated preview does not fetch or expose the original image as a runtime asset; an export can include reference metadata only if requested before Send."
  },
  {
    id: "failure-recovery-rehearsal",
    title: "Failure-recovery rehearsal",
    concept: "An honest provider and offline fallback boundary",
    purpose: "Show what stays available when a live provider cannot complete a request.",
    description:
      "This is a controlled recovery rehearsal: the validation fixture produces the same provider-fallback state that a real unavailable provider would surface, without pretending a fallback is a live preview.",
    category: "Failure-recovery workflow",
    runtime: "Controlled provider-fallback and local-draft state",
    workflowMode: "auto",
    workflow: "Provider-failure recovery with retry and code-only fallback",
    inputRequirement: "No upload is needed. Use the controlled failure fixture in validation; never simulate a provider failure in a normal user session.",
    prompt:
      "Create a concise fallback-ready design brief for a browser-native generative study. State the preferred supported runtime, one validation step, and a code-only fallback. Do not claim that a preview ran if the provider is unavailable.",
    expectedArtifact: "A clearly labeled local draft or usable code artifact, depending on the recovery state",
    expectedPreview: "No live preview is claimed after an unavailable-provider fallback.",
    expectedInteraction: "Inspect the recovery message, choose Retry when the provider returns, or open Code for the local draft.",
    expectedValidation:
      "Controlled provider-failure stream, truthful Partial outcome, retry affordance, no false preview, and session persistence.",
    fallback: "Keep the local draft visible and retry later; do not relabel it as a provider-generated live artifact.",
    estimatedGenerationTime: "Target: 30 seconds for the controlled fallback state and 30 seconds for recovery inspection.",
    providerRequirement: "Controlled failure fixture for validation; a real provider remains optional for the normal brief",
    retrievalRequirement: "Not required for the fallback assertion",
    sourceBoundary:
      "The application distinguishes a provider response, a local draft, and a live renderer. Recovery never upgrades an unavailable preview to Success."
  }
] as const satisfies readonly DemoModeScenario[];

export const demoModeScenarios = demoModeScenarioCatalog;

export const demoModeScenarioCount = demoModeScenarios.length;

export const demoModeRecommendedLiveSequence = [
  {
    role: "Audio-visual opener",
    scenarioId: "cymatic-chladni-audiovisual",
    title: "Polyrhythmic constellation",
    rationale: "Silent-by-default spectrum with an original two-voice score and explicit Tone.js playback."
  },
  {
    role: "Fast browser artifact",
    scenarioId: "physarum-p5-hero",
    title: "Recursive aurora garden",
    rationale: "A concise global-mode p5.js showcase with golden-angle growth and pointer parallax."
  },
  {
    role: "Spatial hero",
    scenarioId: "kinetic-three-hero",
    title: "Kinetic orbit sculpture",
    rationale: "A bounded Three.js scene with fullscreen-ready presentation."
  },
  {
    role: "Technical close-up",
    scenarioId: "chladni-glsl-hero",
    title: "Fractal solar bloom",
    rationale: "A compact analytical shader that makes its WebGL boundary explicit."
  }
] as const satisfies readonly DemoModeRecommendation[];

export function getDefaultDemoModeScenario(): DemoModeScenario {
  const scenario = demoModeScenarios[0];
  if (!scenario) {
    throw new Error("Demo Mode requires at least one validated scenario.");
  }
  return scenario;
}
