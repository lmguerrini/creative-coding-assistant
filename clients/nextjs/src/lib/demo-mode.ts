export type DemoModeScenario = {
  id: string;
  title: string;
  category: string;
  runtime: string;
  prompt: string;
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
    category: "3D browser scene",
    runtime: "Three.js",
    prompt:
      "Create an audio-reactive Three.js visual system for a capstone demo. Use concentric geometry, subtle bloom, FFT-driven motion accents, camera movement, browser-safe runtime notes, interaction guidance, and a clear fallback if live audio or preview is unavailable.",
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
    category: "2D generative sketch",
    runtime: "p5.js",
    prompt:
      "Create a p5.js generative morphogenesis sketch using reaction diffusion, cellular automata, L-systems, flow fields, particle systems, differential growth, diffusion-limited aggregation, branching, self-organization, and emergent form. Keep it browser-safe, explain interaction controls, and cite retrieval/source boundaries conservatively.",
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
    category: "Shader visual",
    runtime: "GLSL",
    prompt:
      "Design a GLSL fragment shader or post-processing visual for a browser creative-coding scene. Include uniforms, resolution/time handling, glow, texture movement, WebGL failure risks, and a static-code fallback if runtime preview is unavailable.",
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
    category: "Live-code visual pattern",
    runtime: "Hydra",
    prompt:
      "Create a Hydra feedback-pattern demo for a browser visual performance. Use oscillator layers, modulation, feedback, moire-like pattern motion, output routing, visual pattern explanation, and a fallback to GLSL or static artifact evidence when the Hydra runtime is unavailable.",
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
    category: "RAG answer",
    runtime: "Assistant workflow",
    prompt:
      "Answer a creative-coding runtime question with registered source grounding. Explain which retrieved sources shaped the response, what the source boundaries are, and how the answer should be validated before using it in a browser sketch.",
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
    category: "Creative translation",
    runtime: "Assistant workflow",
    prompt:
      "Translate the concept of threshold, recursion, and return into a practical browser visual system. Use geometry, motion, color, runtime choice, interaction, and implementation constraints without making spiritual, therapeutic, or authority claims.",
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
    category: "Generative systems",
    runtime: "p5.js / GLSL",
    prompt:
      "Design a geometry and morphogenesis visual system for the browser. Combine radial structures, recursive growth, reaction diffusion, diffusion-limited aggregation, branching, flow fields, and particle trails. Include runtime selection, preview strategy, source boundaries, and a graceful fallback plan.",
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
    category: "Planning workflow",
    runtime: "Assistant workflow",
    prompt:
      "Plan a browser-based installation or immersive scene for a gallery demo. Include concept, geometry, audience movement, runtimes, retrieval needs, preview plan, artifact package, evaluation checks, fallback route, and handoff boundaries.",
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
