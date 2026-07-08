export type DemoModeScenario = {
  id: string;
  title: string;
  description: string;
  category: string;
  runtime: string;
  prompt: string;
  estimatedGenerationTime: string;
  estimatedTokenUsage: string;
  workflowType: string;
  providerRequirement: string;
  retrievalRequirement: string;
  previewAvailability: string;
  fallbackAvailability: string;
  expectedOutput: string;
  complexity: string;
  recommendedForDemo: string;
  presentationTime: string;
  talkingPoint: string;
  expectedBehavior: string;
  fallback: string;
  outputGuidance: string;
  evidence: string[];
  sourceBoundary: string;
  validationPath: string;
};

export const demoModeScenarios = [
  {
    id: "three-audio-reactive-visual-system",
    title: "Three.js Audio-Reactive Visual System",
    description:
      "High-impact 3D browser scene for showing prompt-to-artifact generation, audio boundaries, and visual ambition.",
    category: "3D browser scene",
    runtime: "Three.js",
    prompt:
      "Create an audio-reactive Three.js visual system for a capstone demo. Use concentric geometry, subtle bloom, FFT-driven motion accents, camera movement, browser-safe runtime notes, interaction guidance, and a clear fallback if live audio or preview is unavailable.",
    estimatedGenerationTime: "94.1s measured full-app smoke",
    estimatedTokenUsage: "41,960 total / 4,000 output tokens",
    workflowType: "Single-domain generation with retrieval",
    providerRequirement: "OpenAI provider configured for live generation",
    retrievalRequirement: "3 retrieved contexts in measured smoke",
    previewAvailability:
      "Golden Three.js browser QA; generated HTML/code may need Code panel handoff",
    fallbackAvailability: "Committed Three.js artifact QA and static launcher",
    expectedOutput:
      "Single-file Three.js scene with optional user-gesture audio controls",
    complexity: "High",
    recommendedForDemo: "Primary wow moment",
    presentationTime: "90-120s",
    talkingPoint:
      "Shows visual ambition, browser audio constraints, and conservative fallback handling.",
    expectedBehavior:
      "The assistant should produce source-grounded Three.js guidance, artifact direction, runtime constraints, and a fallback plan suitable for the normal preview workflow.",
    fallback:
      "Use the committed Three.js golden artifact QA evidence and explain that live audio remains opt-in and browser-controlled.",
    outputGuidance:
      "Show geometry, animation mapping, runtime choice, preview route, and the exact boundary between generated code guidance and local browser QA evidence.",
    evidence: [
      "demo/golden_artifacts/browser_full_runtime_qa_results.json",
      "demo/golden_artifacts/qa_manifest.json",
      "docs/V8_CAPSTONE_EVIDENCE_MATRIX.md"
    ],
    sourceBoundary:
      "Uses registered creative-coding/runtime guidance and local browser QA evidence; does not claim cloud deployment or external DCC execution.",
    validationPath: "Three.js golden artifact rendered nonblank in local browser QA."
  },
  {
    id: "p5-generative-morphogenesis-sketch",
    title: "p5.js Generative Morphogenesis Sketch",
    description:
      "Rule-based 2D growth system for demonstrating generative form, interaction controls, and browser-safe sketch guidance.",
    category: "2D generative sketch",
    runtime: "p5.js",
    prompt:
      "Create a p5.js generative morphogenesis sketch using reaction diffusion, cellular automata, L-systems, flow fields, particle systems, differential growth, diffusion-limited aggregation, branching, self-organization, and emergent form. Keep it browser-safe, explain interaction controls, and cite retrieval/source boundaries conservatively.",
    estimatedGenerationTime: "100.5s measured full-app smoke",
    estimatedTokenUsage: "40,760 total / 4,000 output tokens",
    workflowType: "Single-domain generation with retrieval",
    providerRequirement: "OpenAI provider configured for live generation",
    retrievalRequirement: "5 retrieved contexts in measured smoke",
    previewAvailability:
      "Golden p5.js browser QA; generated HTML/code can be inspected as an artifact",
    fallbackAvailability: "Committed p5.js artifact QA and offline demo dataset",
    expectedOutput:
      "Browser-safe p5.js sketch with growth techniques and controls",
    complexity: "High",
    recommendedForDemo: "Primary creative-quality proof",
    presentationTime: "90-120s",
    talkingPoint:
      "Connects computational growth techniques to concrete p5.js implementation choices.",
    expectedBehavior:
      "The assistant should translate morphogenesis techniques into practical p5.js structure, visual controls, a clear visual growth story, and reviewable implementation notes.",
    fallback:
      "Use the committed p5.js golden artifact QA record and the offline demo dataset if live generation is unavailable.",
    outputGuidance:
      "Emphasize computational growth, rule systems, visual texture, and implementation constraints rather than metaphysical or authority claims.",
    evidence: [
      "demo/golden_artifacts/browser_full_runtime_qa_results.json",
      "demo/golden_demo_dataset.json",
      "docs/CAPSTONE_EVALUATION_ETHICS.md"
    ],
    sourceBoundary:
      "Generic morphogenesis techniques are represented in code/catalog signals; no Jason Webb or morphogenesis-resources source coverage is claimed.",
    validationPath: "p5.js golden artifact rendered nonblank in local browser QA."
  },
  {
    id: "glsl-shader-post-processing-visual",
    title: "GLSL Shader And Post-Processing Visual",
    description:
      "Shader-focused flow for proving technical depth, WebGL constraints, and nonblank render evidence.",
    category: "Shader visual",
    runtime: "GLSL",
    prompt:
      "Design a GLSL fragment shader or post-processing visual for a browser creative-coding scene. Include uniforms, resolution/time handling, glow, texture movement, WebGL failure risks, and a static-code fallback if runtime preview is unavailable.",
    estimatedGenerationTime: "97.5s measured full-app smoke",
    estimatedTokenUsage: "40,348 total / 3,368 output tokens",
    workflowType: "Single-domain generation with retrieval",
    providerRequirement: "OpenAI provider configured for live generation",
    retrievalRequirement: "5 retrieved contexts in measured smoke",
    previewAvailability: "GLSL/WebGL golden artifact compiled and drew nonblank",
    fallbackAvailability: "Committed GLSL artifact and WebGL QA record",
    expectedOutput:
      "Fragment shader or post-processing visual with runtime failure notes",
    complexity: "Medium-high",
    recommendedForDemo: "Primary technical proof",
    presentationTime: "60-90s",
    talkingPoint:
      "Shows direct WebGL validation without claiming display-FPS benchmarking.",
    expectedBehavior:
      "The assistant should provide shader structure, runtime tradeoffs, and fallback implementation options with conservative preview claims.",
    fallback:
      "Use the committed GLSL golden artifact and WebGL QA evidence if live preview cannot run.",
    outputGuidance:
      "Keep the answer grounded in browser shader constraints, nonblank render evidence, and graceful failure handling.",
    evidence: [
      "demo/golden_artifacts/browser_full_runtime_qa_results.json",
      "demo/golden_artifacts/glsl_kaleidoscope_field.frag",
      "docs/V8_GRAND_ENGINEERING_REVIEW.md"
    ],
    sourceBoundary:
      "Uses local WebGL QA and static shader checks; does not claim display-FPS benchmarking.",
    validationPath: "GLSL compiled, linked, drew, and pixel-checked nonblank in local WebGL QA."
  },
  {
    id: "hydra-feedback-pattern-demo",
    title: "Hydra Feedback-Pattern Demo",
    description:
      "Bounded live-code visual pattern demo for showing validated Hydra support without overstating editor/runtime scope.",
    category: "Live-code visual pattern",
    runtime: "Hydra",
    prompt:
      "Create a Hydra feedback-pattern demo for a browser visual performance. Use oscillator layers, modulation, feedback, moire-like pattern motion, output routing, visual pattern explanation, and a fallback to GLSL or static artifact evidence when the Hydra runtime is unavailable.",
    estimatedGenerationTime: "Not live-smoke measured; artifact QA only",
    estimatedTokenUsage: "Not measured in full-app provider smoke",
    workflowType: "Artifact QA with optional assistant explanation",
    providerRequirement: "No provider required for golden artifact; optional for live explanation",
    retrievalRequirement: "No retrieval required for artifact QA",
    previewAvailability:
      "Validated local hydra-synth browser artifact path only",
    fallbackAvailability: "Hydra QA record, GLSL guidance, or static artifact evidence",
    expectedOutput:
      "Hydra feedback lattice rendered nonblank through local hydra-synth QA",
    complexity: "High boundary sensitivity",
    recommendedForDemo: "Optional if time permits",
    presentationTime: "30-45s",
    talkingPoint:
      "Hydra is supported only through the validated local hydra-synth artifact path.",
    expectedBehavior:
      "The assistant should describe a bounded Hydra-compatible chain and clearly explain that support is limited to the validated local hydra-synth artifact path.",
    fallback:
      "Use the Hydra golden artifact QA result, or pivot to GLSL guidance if the runtime cannot load.",
    outputGuidance:
      "Show runtime load assumptions, nonblank render evidence, and no microphone or full editor claim.",
    evidence: [
      "demo/golden_artifacts/hydra_feedback_lattice.js",
      "demo/golden_artifacts/browser_full_runtime_qa_results.json",
      "demo/golden_artifacts/README.md"
    ],
    sourceBoundary:
      "Hydra support is bounded to the validated local hydra-synth browser artifact path.",
    validationPath: "Hydra golden artifact rendered nonblank through hydra-synth local browser QA."
  },
  {
    id: "retrieval-grounded-creative-coding-answer",
    title: "Retrieval-Grounded Creative Coding Answer",
    description:
      "Evidence-centered answer path for showing retrieval, source boundaries, and evaluator-safe RAGAs evidence.",
    category: "RAG answer",
    runtime: "Assistant workflow",
    prompt:
      "Answer a creative-coding runtime question with registered source grounding. Explain which retrieved sources shaped the response, what the source boundaries are, and how the answer should be validated before using it in a browser sketch.",
    estimatedGenerationTime: "76.5s measured full-app smoke",
    estimatedTokenUsage: "40,394 total / 2,813 output tokens",
    workflowType: "Hybrid retrieval-grounded generation",
    providerRequirement: "OpenAI provider configured for live answer",
    retrievalRequirement: "5 retrieved contexts plus redacted/sanitized RAGAs evidence",
    previewAvailability: "Answer/evidence flow; preview is not required",
    fallbackAvailability: "Redacted latest-live RAGAs and retrieval smoke evidence",
    expectedOutput:
      "Source-grounded creative-coding answer with visible boundaries",
    complexity: "Medium",
    recommendedForDemo: "Q&A credibility proof",
    presentationTime: "60-90s",
    talkingPoint:
      "Raw private rows stay local while reviewer-safe fixtures carry evaluator evidence.",
    expectedBehavior:
      "The assistant should run the normal retrieval-grounded answer path, surface citations/evidence, and keep privacy boundaries clear.",
    fallback:
      "Use the redacted latest-live RAGAs results, sanitized RAGAs evidence, and retrieval smoke records if live retrieval is unavailable.",
    outputGuidance:
      "Make source grounding visible, avoid unsupported library/runtime claims, and explain what remains to verify locally.",
    evidence: [
      "demo/evaluation/redacted_live_session_ragas_latest4_results.jsonl",
      "demo/evaluation/sanitized_ragas_context_precision_results_external.jsonl",
      "docs/eval_pipeline.md"
    ],
    sourceBoundary:
      "Raw private live-session rows are not sent externally; public evidence uses sanitized or redacted fixtures.",
    validationPath: "Redacted latest-live RAGAs passed with zero skipped rows and zero metric failures."
  },
  {
    id: "concept-to-visual-translation",
    title: "Concept-To-Visual Translation",
    description:
      "Creative-direction flow for converting abstract language into concrete browser visual decisions.",
    category: "Creative translation",
    runtime: "Assistant workflow",
    prompt:
      "Translate the concept of threshold, recursion, and return into a practical browser visual system. Use geometry, motion, color, runtime choice, interaction, and implementation constraints without making spiritual, therapeutic, or authority claims.",
    estimatedGenerationTime: "103.4s measured full-app smoke",
    estimatedTokenUsage: "40,381 total / 3,624 output tokens",
    workflowType: "Single-domain creative translation with retrieval",
    providerRequirement: "OpenAI provider configured for live generation",
    retrievalRequirement: "5 retrieved contexts in measured smoke",
    previewAvailability:
      "Generated browser artifact can be inspected when extraction succeeds",
    fallbackAvailability: "Prompt library, generated artifact QA, and claim-safety explanation",
    expectedOutput:
      "Operational visual system guidance with runtime and interaction choices",
    complexity: "Medium",
    recommendedForDemo: "Narrative bridge",
    presentationTime: "60-90s",
    talkingPoint:
      "Demonstrates creative translation while keeping interpretation aesthetic and operational.",
    expectedBehavior:
      "The assistant should convert abstract concept language into operational visual design guidance and code-ready structure.",
    fallback:
      "Use the prompt library and generated artifact QA evidence to explain the translation path without running a live provider call.",
    outputGuidance:
      "Frame the result as concept-to-visual creative direction with implementation notes, not as objective interpretation.",
    evidence: [
      "demo/demo_prompt_library.md",
      "docs/CAPSTONE_EVALUATION_ETHICS.md",
      "docs/V8_CAPSTONE_EVIDENCE_MATRIX.md"
    ],
    sourceBoundary:
      "Uses internal creative translation signals as review evidence; it is not treated as objective truth.",
    validationPath: "Creative translation surfaces are covered by existing frontend tests and evidence documentation."
  },
  {
    id: "geometry-morphogenesis-visual-system",
    title: "Geometry And Morphogenesis Visual System",
    description:
      "Multi-runtime generative-system flow for showing geometry, growth rules, and practical preview strategy.",
    category: "Generative systems",
    runtime: "p5.js / GLSL",
    prompt:
      "Design a geometry and morphogenesis visual system for the browser. Combine radial structures, recursive growth, reaction diffusion, diffusion-limited aggregation, branching, flow fields, and particle trails. Include runtime selection, preview strategy, source boundaries, and a graceful fallback plan.",
    estimatedGenerationTime: "97.4s measured full-app smoke",
    estimatedTokenUsage: "40,345 total / 3,128 output tokens",
    workflowType: "Multi-domain generation with retrieval",
    providerRequirement: "OpenAI provider configured for live generation",
    retrievalRequirement: "5 retrieved contexts in measured smoke",
    previewAvailability: "p5.js and GLSL golden artifacts have browser QA evidence",
    fallbackAvailability: "p5.js and GLSL QA records plus offline prompts",
    expectedOutput:
      "Browser-oriented generative system plan with runtime selection and controls",
    complexity: "High",
    recommendedForDemo: "Secondary systems-depth proof",
    presentationTime: "60-90s",
    talkingPoint:
      "Shows emergent form through inspectable rules rather than unsupported authority claims.",
    expectedBehavior:
      "The assistant should describe a cohesive generative system, choose a practical browser runtime, and keep claims bounded to implementation guidance.",
    fallback:
      "Use the p5.js and GLSL golden QA records plus offline prompts if live generation or preview fails.",
    outputGuidance:
      "Show how form emerges from rules, how the viewer can inspect parameters, and how to keep performance predictable.",
    evidence: [
      "demo/golden_artifacts/browser_full_runtime_qa_results.json",
      "demo/demo_prompt_library.md",
      "docs/V8_GRAND_ENGINEERING_REVIEW.md"
    ],
    sourceBoundary:
      "Generic geometry/morphogenesis coverage exists in the local knowledge/catalog code; no Jason Webb-specific source claim is made.",
    validationPath: "p5.js and GLSL golden artifacts have local browser QA evidence."
  },
  {
    id: "installation-immersive-scene-planning",
    title: "Installation And Immersive Scene Planning",
    description:
      "Planning and handoff flow for showing local-demo scope, fallback routes, and reviewer-ready project sequencing.",
    category: "Planning workflow",
    runtime: "Assistant workflow",
    prompt:
      "Plan a browser-based installation or immersive scene for a gallery demo. Include concept, geometry, audience movement, runtimes, retrieval needs, preview plan, artifact package, evaluation checks, fallback route, and handoff boundaries.",
    estimatedGenerationTime: "Not live-smoke measured; planning evidence only",
    estimatedTokenUsage: "Not measured in full-app provider smoke",
    workflowType: "Planning workflow with advisory evidence",
    providerRequirement: "Provider optional for live planning; offline evidence is available",
    retrievalRequirement: "Retrieval optional; source boundaries must be stated if used",
    previewAvailability: "Preview depends on generated artifact choice",
    fallbackAvailability:
      "Integrated Demo Mode, static launcher, offline dataset, and evidence docs",
    expectedOutput:
      "Local browser installation plan with demo sequence and handoff boundaries",
    complexity: "Medium-high",
    recommendedForDemo: "Fallback or Q&A planning proof",
    presentationTime: "45-75s",
    talkingPoint:
      "Shows delivery judgment: local browser demo target, not public deployment or external execution.",
    expectedBehavior:
      "The assistant should create a bounded project plan with demo sequence, preview strategy, evidence path, and implementation handoff notes.",
    fallback:
      "Use the integrated Demo Mode scenario list, external launcher fallback, evidence docs, and offline dataset if any live service fails.",
    outputGuidance:
      "Keep the planning answer practical: browser runtimes, reviewer walkthrough, validation evidence, and explicit limits.",
    evidence: [
      "demo/final_demo_suite.json",
      "demo/final_demo_launcher.html",
      "docs/CAPSTONE_DEMO_SHOWCASE.md"
    ],
    sourceBoundary:
      "Planning guidance only; no public deployment, live venue scan, engineering certification, or external tool execution is claimed.",
    validationPath: "Demo plan and fallback paths are documented in the final demo suite."
  }
] as const satisfies readonly DemoModeScenario[];

export const demoModeScenarioCount = demoModeScenarios.length;

export function getDefaultDemoModeScenario(): DemoModeScenario {
  return demoModeScenarios[0];
}
