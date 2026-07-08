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

export type DemoModeRecommendation = {
  role: string;
  scenarioId: DemoModeScenario["id"];
  title: string;
  rationale: string;
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
      "Create a concise Three.js capstone artifact: concentric audio-reactive geometry, subtle bloom-like glow, FFT motion accents, camera movement, browser-safe audio controls, and a fallback if live audio or preview is unavailable. Return a brief plan, compact browser code/artifact direction, validation notes, and fallback. Keep it demo-ready and under 120 code lines or 700 words.",
    estimatedGenerationTime: "68.8s optimized live smoke",
    estimatedTokenUsage: "41,518 total / 2,817 output tokens",
    workflowType: "Single-domain generation with retrieval",
    providerRequirement: "OpenAI provider configured for live generation",
    retrievalRequirement: "4 retrieved contexts in optimized smoke",
    previewAvailability:
      "Optimized smoke produced artifact events; golden Three.js browser QA remains fallback",
    fallbackAvailability: "Committed Three.js artifact QA and static launcher",
    expectedOutput:
      "Single-file Three.js scene with optional user-gesture audio controls",
    complexity: "High",
    recommendedForDemo: "Primary wow moment",
    presentationTime: "75-90s",
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
      "Create a concise p5.js generative morphogenesis sketch using reaction diffusion, flow fields, particles, branching, and emergent form. Include interaction controls, browser-safe notes, source boundaries, compact code/artifact direction, and fallback. Keep the p5.js generative morphogenesis sketch demo-ready and under 120 code lines or 700 words.",
    estimatedGenerationTime: "33.9s optimized live smoke",
    estimatedTokenUsage: "39,645 total / 2,839 output tokens",
    workflowType: "Single-domain generation with retrieval",
    providerRequirement: "OpenAI provider configured for live generation",
    retrievalRequirement: "5 retrieved contexts in optimized smoke",
    previewAvailability:
      "Optimized smoke produced artifact events; golden p5.js browser QA remains fallback",
    fallbackAvailability: "Committed p5.js artifact QA and offline demo dataset",
    expectedOutput:
      "Browser-safe p5.js sketch with growth techniques and controls",
    complexity: "High",
    recommendedForDemo: "Primary creative-quality proof",
    presentationTime: "60-75s",
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
      "Design a concise GLSL fragment shader or post-processing visual for a browser creative-coding scene. Include uniforms, resolution/time handling, glow, texture movement, WebGL risks, compact shader/artifact direction, and static fallback. Keep it demo-ready and under 100 shader lines or 600 words.",
    estimatedGenerationTime: "32.8s optimized live smoke",
    estimatedTokenUsage: "39,400 total / 2,343 output tokens",
    workflowType: "Single-domain generation with retrieval",
    providerRequirement: "OpenAI provider configured for live generation",
    retrievalRequirement: "5 retrieved contexts in optimized smoke",
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
      "Create a concise Hydra feedback-pattern demo for the validated local hydra-synth browser artifact path. Use oscillator layers, modulation, feedback, moire-like motion, output routing, bounded runtime assumptions, and fallback to GLSL/static evidence. Keep it demo-ready and under 80 Hydra lines or 500 words.",
    estimatedGenerationTime: "0.4s optimized bounded route; no provider call",
    estimatedTokenUsage: "N/A; no provider token usage captured",
    workflowType: "Bounded multi-domain route; artifact QA support only",
    providerRequirement: "No provider call in optimized smoke; optional for live explanation",
    retrievalRequirement: "5 retrieved contexts in optimized bounded smoke",
    previewAvailability:
      "Validated local hydra-synth browser artifact path only",
    fallbackAvailability: "Hydra QA record, GLSL guidance, or static artifact evidence",
    expectedOutput:
      "Hydra feedback lattice rendered nonblank through local hydra-synth QA",
    complexity: "High boundary sensitivity",
    recommendedForDemo: "Artifact fallback only",
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
      "Answer a creative-coding runtime question with registered source grounding. In under 600 words, explain the retrieved sources, source boundaries, browser validation steps, and fallback if retrieval is unavailable.",
    estimatedGenerationTime: "19.2s optimized live smoke",
    estimatedTokenUsage: "38,778 total / 1,286 output tokens",
    workflowType: "Hybrid retrieval-grounded generation",
    providerRequirement: "OpenAI provider configured for live answer",
    retrievalRequirement: "5 retrieved contexts plus redacted/sanitized RAGAs evidence",
    previewAvailability: "Answer/evidence flow; preview is not required",
    fallbackAvailability: "Redacted latest-live RAGAs and retrieval smoke evidence",
    expectedOutput:
      "Source-grounded creative-coding answer with visible boundaries",
    complexity: "Medium",
    recommendedForDemo: "Q&A credibility proof",
    presentationTime: "45-60s",
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
      "Translate threshold, recursion, and return into a practical browser visual system. Keep the answer concise: geometry, motion, color, runtime choice, interaction, implementation constraints, and claim boundaries. Avoid spiritual, therapeutic, or authority claims. Target under 650 words.",
    estimatedGenerationTime: "26.3s optimized live smoke",
    estimatedTokenUsage: "38,919 total / 2,109 output tokens",
    workflowType: "Single-domain creative translation with retrieval",
    providerRequirement: "OpenAI provider configured for live generation",
    retrievalRequirement: "5 retrieved contexts in optimized smoke",
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
      "Design a concise geometry and morphogenesis browser visual system. Combine radial structures, recursive growth, reaction diffusion, branching, flow fields, and particle trails. Include runtime selection, preview strategy, source boundaries, compact artifact direction, and graceful fallback. Target under 750 words.",
    estimatedGenerationTime: "21.5s optimized live smoke",
    estimatedTokenUsage: "38,556 total / 1,639 output tokens",
    workflowType: "Multi-domain generation with retrieval",
    providerRequirement: "OpenAI provider configured for live generation",
    retrievalRequirement: "5 retrieved contexts in optimized smoke",
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
      "Plan a concise browser-based installation or immersive scene for a gallery demo. Include concept, geometry, audience movement, runtimes, retrieval needs, preview plan, artifact package, evaluation checks, fallback route, and handoff boundaries. Target under 700 words.",
    estimatedGenerationTime: "52.2s optimized live smoke",
    estimatedTokenUsage: "37,699 total / 1,370 output tokens",
    workflowType: "Planning workflow with retrieval",
    providerRequirement: "OpenAI provider configured for live planning",
    retrievalRequirement: "5 retrieved contexts in optimized smoke",
    previewAvailability: "Preview depends on generated artifact choice",
    fallbackAvailability:
      "Integrated Demo Mode, static launcher, offline dataset, and evidence docs",
    expectedOutput:
      "Local browser installation plan with demo sequence and handoff boundaries",
    complexity: "Medium-high",
    recommendedForDemo: "Best Q&A demo",
    presentationTime: "45-60s",
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

export const demoModeRecommendedLiveSequence = [
  {
    role: "Fastest reliable demo",
    scenarioId: "retrieval-grounded-creative-coding-answer",
    title: "Retrieval-grounded answer",
    rationale: "19.2s optimized smoke, 5 retrieved contexts, best quick proof."
  },
  {
    role: "Most visually impressive demo",
    scenarioId: "three-audio-reactive-visual-system",
    title: "Three.js audio-reactive visual",
    rationale: "68.8s optimized smoke with artifact events and strongest wow factor."
  },
  {
    role: "Safest fallback demo",
    scenarioId: "glsl-shader-post-processing-visual",
    title: "GLSL shader visual",
    rationale: "32.8s optimized smoke plus direct WebGL golden artifact QA."
  },
  {
    role: "Best RAG demo",
    scenarioId: "retrieval-grounded-creative-coding-answer",
    title: "Source-grounded creative answer",
    rationale: "Shortest provider-backed route with clear retrieval and RAGAs evidence."
  },
  {
    role: "Best Q&A demo",
    scenarioId: "installation-immersive-scene-planning",
    title: "Installation planning",
    rationale: "52.2s optimized smoke, useful for scope, ethics, and handoff questions."
  }
] as const satisfies readonly DemoModeRecommendation[];

export function getDefaultDemoModeScenario(): DemoModeScenario {
  return demoModeScenarios[0];
}
