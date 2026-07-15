export type CuratedPrompt = {
  id: string;
  title: string;
  description: string;
  concept: string;
  runtime: string;
  prompt: string;
  expectedArtifact: string;
  previewBoundary: string;
  fallback: string;
};

export const boundedP5DemoSurfaceContract =
  "Use only these supported p5 calls: createCanvas, pixelDensity, colorMode, background, push, pop, translate, rotate, stroke, strokeWeight, strokeCap, noStroke, fill, noFill, line, circle, sin, cos, sqrt, noise, map, constrain, lerp, floor, min, and max. Use frameCount for time, use p5's PI constant directly, and declare every custom helper function. Never redeclare p5 globals. Do not use createGraphics, millis, lerpColor, drawingContext, shaders, DOM, loading APIs, imports, HTML, assets, Markdown, or prose.";

export const recursiveAuroraGardenFallbackSource = [
  "let phase = 0;",
  "function setup() {",
  "  createCanvas(windowWidth, windowHeight);",
  "  pixelDensity(1);",
  "  colorMode(HSL, 360, 100, 100, 1);",
  "  noStroke();",
  "}",
  "function draw() {",
  "  phase += 0.008;",
  "  background(232, 54, 6, 0.2);",
  "  translate(width * 0.5, height * 0.52);",
  "  for (let seed = 0; seed < 160; seed += 1) {",
  "    const goldenAngle = seed * 2.399963;",
  "    const radius = 7.2 * sqrt(seed);",
  "    const breath = 1 + 0.08 * sin(phase * 2 + seed * 0.11);",
  "    const x = cos(goldenAngle + phase) * radius * breath + (mouseX - width * 0.5) * 0.025;",
  "    const y = sin(goldenAngle + phase) * radius * 0.62 * breath + (mouseY - height * 0.5) * 0.018;",
  "    fill((178 + seed * 0.72) % 360, 82, 64, 0.54);",
  "    circle(x, y, 4 + 4 * sin(seed * 0.17 + phase));",
  "  }",
  "}"
].join("\n");

export const multiAgentOrbitStudyFallbackSource = [
  "// researcher, creative director, generator, reviewer",
  "function setup() {",
  "  createCanvas(windowWidth, windowHeight);",
  "  pixelDensity(1);",
  "  colorMode(HSL, 360, 100, 100, 1);",
  "  strokeCap(ROUND);",
  "}",
  "function draw() {",
  "  const time = frameCount * 0.008;",
  "  background(228, 58, 6, 0.24);",
  "  translate(width * 0.5 + (mouseX - width * 0.5) * 0.035, height * 0.5 + (mouseY - height * 0.5) * 0.025);",
  "  noFill();",
  "  for (let ring = 0; ring < 18; ring += 1) {",
  "    const radius = 24 + ring * min(width, height) * 0.024;",
  "    const hue = 178 + ring * 4.6;",
  "    stroke(hue, 88, 68, 0.38);",
  "    strokeWeight(1 + (ring % 4) * 0.45);",
  "    let px = cos(time + ring * 0.31) * radius;",
  "    let py = sin(time + ring * 0.31) * radius * 0.62;",
  "    for (let step = 1; step <= 52; step += 1) {",
  "      const angle = step * PI / 26 + time * (0.32 + ring * 0.012) + ring * 0.31;",
  "      const pulse = 1 + 0.1 * sin(time * 2 + step * 0.24 + ring);",
  "      const x = cos(angle) * radius * pulse;",
  "      const y = sin(angle) * radius * 0.62 * pulse;",
  "      line(px, py, x, y);",
  "      px = x;",
  "      py = y;",
  "    }",
  "  }",
  "  noStroke();",
  "  fill(42, 96, 70, 0.9);",
  "  circle(0, 0, 12 + 4 * sin(time * 3));",
  "}"
].join("\n");

export const homepagePromptLibrary = [
  {
    id: "physarum-drift",
    title: "Physarum drift",
    description: "A living trail field for the p5.js preview.",
    concept: "Physarum-inspired collective motion",
    runtime: "p5.js browser preview",
    prompt:
      "Create exactly one runnable .p5.js artifact named physarum-drift.p5.js. Make a graceful Physarum-inspired trail field with setup(), draw(), 180 agents, noise-driven steering, rounded translucent paths, and pointer attraction. Use only compact global-mode p5.js JavaScript. No imports or HTML. Return only one closed ```javascript filename=physarum-drift.p5.js code fence with no prose, and keep the source below 2,800 characters.",
    expectedArtifact: "physarum-drift.p5.js",
    previewBoundary: "Runs in the controlled p5.js browser preview.",
    fallback: "Inspect the code artifact and retry with the same global-mode contract."
  },
  {
    id: "kinetic-orbit-sculpture",
    title: "Kinetic orbit sculpture",
    description: "A luminous spatial study for the Three.js preview.",
    concept: "Kinetic sculpture, orbital light, and camera motion",
    runtime: "Three.js browser preview",
    prompt:
      "Create exactly one self-contained .three.js artifact named kinetic-orbit-sculpture.three.js. Use plain JavaScript only: THREE.Scene, PerspectiveCamera, WebGLRenderer, a central TorusKnotGeometry or sphere sculpture, three thin orbit rings, two PointLights, an AmbientLight, a deep navy background, and a requestAnimationFrame loop. Compose it as a polished hero: warm-gold sculpture, cool-blue rim light, slow object rotation, gentle ring counter-rotation, and a slow camera orbit that keeps the subject framed. Use hexadecimal color strings or numeric hex values only; never call setHSL. Do not declare canvas, width, height, or pixelRatio. Do not use imports, HTML, React, TypeScript, Points, BufferGeometry, external assets, Markdown, or prose. Return only the artifact.",
    expectedArtifact: "kinetic-orbit-sculpture.three.js",
    previewBoundary: "Runs in the controlled Three.js browser preview.",
    fallback: "Inspect the code artifact; standalone HTML and React exports remain code-only."
  },
  {
    id: "chladni-light-field",
    title: "Chladni light field",
    description: "A precise interference study for the GLSL preview.",
    concept: "Cymatic interference pattern",
    runtime: "GLSL browser preview",
    prompt:
      "Create exactly one compact .frag artifact named chladni-light-field.frag. Return only WebGL 1 fragment shader source with void main(), u_time, u_resolution, and gl_FragColor. Render an animated Chladni-inspired interference field with two colors and a soft glow. Do not use #version, textures, samplers, loops, discard, HTML, Markdown, or prose.",
    expectedArtifact: "chladni-light-field.frag",
    previewBoundary: "Runs only in the bounded WebGL 1 fragment preview.",
    fallback: "Inspect the shader artifact if the browser cannot provide WebGL."
  },
  {
    id: "cymatic-audio-study",
    title: "Cymatic audio study",
    description: "A silent-by-default Tone.js pattern with optional playback.",
    concept: "Chladni-style audio-visual mapping",
    runtime: "Tone.js browser preview",
    prompt:
      "Create exactly one executable .tone.js artifact named cymatic-chladni.tone.js. Start with the comment // CCA_VISUAL: cymatics. Use a Tone.FMSynth or Tone.Synth, one Tone.Sequence with the notes ['C3', 'G3', 'D4', 'A3'], Tone.Transport.bpm.value = 96, and Tone.Transport.start(). Use only variable declarations and Tone namespace calls: do not create a canvas, DOM, or button, and do not use document, window, CSS, requestAnimationFrame, Tone.start, addEventListener, a microphone, files, imports, HTML, Markdown, or prose. The controlled preview supplies the silent visual and explicit Start audio action. Return only the artifact.",
    expectedArtifact: "cymatic-chladni.tone.js",
    previewBoundary: "Visuals mount muted; audio begins only after an explicit Start audio action.",
    fallback: "Use the silent Chladni visual and inspect the generated Tone.js artifact."
  }
] as const satisfies readonly CuratedPrompt[];

export const demoShowcasePromptLibrary = [
  {
    id: "polyrhythmic-constellation",
    title: "Polyrhythmic constellation",
    description: "A silent-first generative score with a luminous spectrum preview.",
    concept: "Generative audio and orbital rhythm",
    runtime: "Tone.js browser preview",
    prompt:
      "Create exactly one executable .tone.js artifact named polyrhythmic-constellation.tone.js. Use a Tone.FMSynth and a Tone.MembraneSynth, two Tone.Sequence patterns with complementary notes and 8n/16n subdivisions, Tone.Transport.bpm.value = 108, and Tone.Transport.start(). Keep the score elegant and sparse. Use only variable declarations and Tone namespace calls; do not create DOM, canvas, buttons, listeners, files, imports, HTML, Markdown, or prose. The controlled preview supplies a silent-first spectrum and explicit Start audio action. Return only the artifact.",
    expectedArtifact: "polyrhythmic-constellation.tone.js",
    previewBoundary: "Visuals mount silently; audio starts only after an explicit Start audio action.",
    fallback: "Use the silent spectrum and inspect the generated Tone.js score."
  },
  {
    id: "recursive-aurora-garden",
    title: "Recursive aurora garden",
    description: "A golden-angle field that grows into an interactive luminous garden.",
    concept: "Recursive growth, phyllotaxis, and pointer parallax",
    runtime: "p5.js browser preview",
    prompt: `Create exactly one runnable .p5.js artifact named recursive-aurora-garden.p5.js. In compact global-mode JavaScript with setup() and draw(), compose 160 golden-angle seeds into a deep-night aurora garden with recursive-looking branching ribbons, translucent cyan/gold glow, slow breathing motion, and pointer parallax. ${boundedP5DemoSurfaceContract} Return exactly one closed \`\`\`javascript filename=recursive-aurora-garden.p5.js code fence and keep source below 3,000 characters.`,
    expectedArtifact: "recursive-aurora-garden.p5.js",
    previewBoundary: "Runs in the controlled global-mode p5.js preview.",
    fallback: "Inspect the source and retry with the same compact global-mode contract."
  },
  {
    id: "kinetic-orbit-capstone",
    title: "Kinetic orbit sculpture",
    description: "A gallery-lit orbital sculpture authored as a real Three.js scene graph.",
    concept: "Kinetic sculpture, nested orbital rigs, and cinematic camera motion",
    runtime: "Three.js browser preview",
    prompt:
      "Create exactly one self-contained .three.js artifact named kinetic-orbit-capstone.three.js. Use plain JavaScript and real scene-graph authorship: a warm-gold TorusKnotGeometry inside sculptureRig, three thin TorusGeometry meshes inside orbitRig, and a PerspectiveCamera parented to cameraRig. Give every rig intentional position/rotation/scale, add two PointLights plus AmbientLight, a deep navy background, WebGLRenderer, and a requestAnimationFrame loop that rotates sculpture/rings, moves cameraRig, calls camera.lookAt(), and renders every frame. Use numeric hex colors; do not declare canvas, width, height, or pixelRatio. No imports, HTML, React, TypeScript, external assets, Markdown, or prose. Return only the artifact.",
    expectedArtifact: "kinetic-orbit-capstone.three.js",
    previewBoundary: "Runs with the locally bundled Three.js r176 WebGL runtime.",
    fallback: "Inspect the scene source and retry without changing the bounded Three.js contract."
  },
  {
    id: "fractal-solar-bloom",
    title: "Fractal solar bloom",
    description: "A precise, animated interference bloom for the WebGL shader surface.",
    concept: "Analytical fractal repetition and solar interference",
    runtime: "GLSL browser preview",
    prompt:
      "Create fractal-solar-bloom.frag as compile-ready WebGL 1 source under 80 lines. Return only code with uniform float u_time; uniform vec2 u_resolution; and void main() writing gl_FragColor. Before main define float spiral(vec2 p,float scale,float twist) using only length, atan, log, sin and smoothstep; call it exactly 6 times. Fill frame with asymmetric logarithmic nautilus curls, nested clusters and scalloped filigree. Palette: navy-black, cobalt/ultramarine, cyan-white rims and amber-gold cores; avoid a centered flower, flat rings, radial symmetry or empty areas. End every statement with a semicolon; balance braces/parentheses and end with main's closing brace. No #version, arrays, structs, macros, recursion, loops, textures, samplers, discard, HTML, Markdown or prose.",
    expectedArtifact: "fractal-solar-bloom.frag",
    previewBoundary: "Runs only in the bounded WebGL 1 fragment preview.",
    fallback: "Inspect the shader source if WebGL is unavailable."
  }
] as const satisfies readonly CuratedPrompt[];

export const domainStarterPromptLibrary = [
  ...homepagePromptLibrary,
  {
    id: "feedback-lattice-hydra",
    title: "Feedback lattice",
    description: "A Hydra-compatible code starter kept out of the visible demo sequence.",
    concept: "Feedback and modulation study",
    runtime: "Hydra-compatible code/export boundary",
    prompt:
      "Create exactly one .hydra.js artifact named feedback-lattice.hydra.js using a compact osc(), noise(), shape(), modulate(), and out(o0) chain. Do not use HTML, imports, external textures, Markdown, or prose. Return only the artifact.",
    expectedArtifact: "feedback-lattice.hydra.js",
    previewBoundary: "Keep this starter code/export-only in Demo Mode until dedicated browser evidence is recorded.",
    fallback: "Inspect the Hydra source and use a p5.js or GLSL hero for a live preview."
  },
  {
    id: "signal-bloom-gsap",
    title: "Signal bloom",
    description: "A bounded GSAP motion-stage starter.",
    concept: "Layered motion and stagger",
    runtime: "GSAP browser preview",
    prompt:
      "Create exactly one .gsap.js artifact named signal-bloom.gsap.js. Use a gsap.timeline with repeat, yoyo, stagger, and transforms for .particle and .ring selectors only. No imports, HTML, Markdown, or prose. Return only the artifact.",
    expectedArtifact: "signal-bloom.gsap.js",
    previewBoundary: "Runs only in the controlled GSAP motion stage.",
    fallback: "Inspect the motion source if the bounded preview cannot mount."
  },
  {
    id: "signal-markup-svg",
    title: "Signal markup",
    description: "An inline SVG starter for the vector surface.",
    concept: "Animated vector rhythm",
    runtime: "SVG browser preview",
    prompt:
      "Create exactly one .svg artifact named signal-markup.svg. Return only a self-contained <svg> with a viewBox, two shapes, and one SVG animate element. Do not use scripts, external assets, HTML wrappers, Markdown, or prose.",
    expectedArtifact: "signal-markup.svg",
    previewBoundary: "Runs only as sanitized inline SVG in the controlled vector surface.",
    fallback: "Inspect or export the SVG artifact."
  },
  {
    id: "signal-grid-canvas",
    title: "Signal grid",
    description: "A Canvas 2D starter for the isolated canvas surface.",
    concept: "Abstract grid motion",
    runtime: "Canvas 2D browser preview",
    prompt:
      "Create exactly one .canvas.js artifact named signal-grid.canvas.js. Use only a provided canvas context, requestAnimationFrame, clearRect, fillRect, and deterministic trigonometric motion. Do not create DOM nodes, load assets, use HTML, Markdown, or prose. Return only the artifact.",
    expectedArtifact: "signal-grid.canvas.js",
    previewBoundary: "Runs only in the controlled Canvas 2D surface.",
    fallback: "Inspect the canvas source if the preview cannot mount."
  }
] as const satisfies readonly CuratedPrompt[];

export const morphogenesisPromptLibrary = [
  homepagePromptLibrary[0],
  {
    id: "cellular-tide",
    title: "Cellular tide",
    description: "A restrained cellular-automata-inspired p5.js study.",
    concept: "Cellular automata",
    runtime: "p5.js browser preview",
    prompt:
      "Create one global-mode .p5.js artifact named cellular-tide.p5.js with setup() and draw(). Make a cellular-automata-inspired grid using simple local neighbor rules, a calm two-color palette, and a click-to-reseed interaction. Return only JavaScript source.",
    expectedArtifact: "cellular-tide.p5.js",
    previewBoundary: "Controlled p5.js preview only.",
    fallback: "Inspect the runnable code artifact."
  },
  homepagePromptLibrary[3],
  {
    id: "fibonacci-orbit",
    title: "Fibonacci orbit",
    description: "A p5.js spiral built from a simple numerical rhythm.",
    concept: "Fibonacci sequence",
    runtime: "p5.js browser preview",
    prompt:
      "Create one global-mode .p5.js artifact named fibonacci-orbit.p5.js with setup() and draw(). Draw a slowly rotating Fibonacci-inspired spiral of circles using a compact loop, a dark background, and pointer-controlled scale. Return only JavaScript source.",
    expectedArtifact: "fibonacci-orbit.p5.js",
    previewBoundary: "Controlled p5.js preview only.",
    fallback: "Inspect the runnable code artifact."
  },
  {
    id: "fractal-bloom",
    title: "Fractal bloom",
    description: "A compact analytical fractal field for WebGL.",
    concept: "Fractals",
    runtime: "GLSL browser preview",
    prompt:
      "Create one compact .frag artifact named fractal-bloom.frag using only void main(), u_time, u_resolution, and gl_FragColor. Make a fractal-inspired repeating bloom without texture sampling, loops, #version, or discard. Return only shader source.",
    expectedArtifact: "fractal-bloom.frag",
    previewBoundary: "Bounded WebGL 1 preview only.",
    fallback: "Inspect the shader artifact."
  },
  {
    id: "golden-angle-garden",
    title: "Golden-angle garden",
    description: "A phyllotactic p5.js field with readable growth.",
    concept: "Golden ratio",
    runtime: "p5.js browser preview",
    prompt:
      "Create one global-mode .p5.js artifact named golden-angle-garden.p5.js with setup() and draw(). Use the golden angle to arrange growing dots, add a gentle color drift, and keep the code self-contained. Return only JavaScript source.",
    expectedArtifact: "golden-angle-garden.p5.js",
    previewBoundary: "Controlled p5.js preview only.",
    fallback: "Inspect the runnable code artifact."
  },
  {
    id: "phyllotaxis-pulse",
    title: "Phyllotaxis pulse",
    description: "A radial p5.js seed pattern with a light interaction.",
    concept: "Phyllotaxis",
    runtime: "p5.js browser preview",
    prompt:
      "Create one global-mode .p5.js artifact named phyllotaxis-pulse.p5.js with setup() and draw(). Draw a phyllotaxis seed pattern that breathes slowly and responds to pointer distance. Return only JavaScript source.",
    expectedArtifact: "phyllotaxis-pulse.p5.js",
    previewBoundary: "Controlled p5.js preview only.",
    fallback: "Inspect the runnable code artifact."
  },
  {
    id: "attractor-ribbons",
    title: "Attractor ribbons",
    description: "A bounded shader study of folded motion.",
    concept: "Strange attractors",
    runtime: "GLSL browser preview",
    prompt:
      "Create one compact .frag artifact named attractor-ribbons.frag using only void main(), u_time, u_resolution, and gl_FragColor. Make a strange-attractor-inspired ribbon field with analytic sine and cosine folds. No textures, loops, #version, or discard. Return only shader source.",
    expectedArtifact: "attractor-ribbons.frag",
    previewBoundary: "Bounded WebGL 1 preview only.",
    fallback: "Inspect the shader artifact."
  },
  {
    id: "superformula-petals",
    title: "Superformula petals",
    description: "An expressive radial p5.js contour.",
    concept: "Superformula",
    runtime: "p5.js browser preview",
    prompt:
      "Create one global-mode .p5.js artifact named superformula-petals.p5.js with setup() and draw(). Draw a superformula-inspired animated petal contour using beginShape(), vertex(), and endShape(). Return only JavaScript source.",
    expectedArtifact: "superformula-petals.p5.js",
    previewBoundary: "Controlled p5.js preview only.",
    fallback: "Inspect the runnable code artifact."
  },
  {
    id: "reaction-drift",
    title: "Reaction drift",
    description: "A visual reaction study, not a scientific simulator.",
    concept: "Belousov–Zhabotinsky reaction",
    runtime: "GLSL browser preview",
    prompt:
      "Create one compact .frag artifact named reaction-drift.frag using only void main(), u_time, u_resolution, and gl_FragColor. Make a Belousov–Zhabotinsky-inspired traveling color field without claiming a scientific simulation. No textures, loops, #version, or discard. Return only shader source.",
    expectedArtifact: "reaction-drift.frag",
    previewBoundary: "Bounded WebGL 1 preview only.",
    fallback: "Inspect the shader artifact."
  },
  {
    id: "viscous-gap",
    title: "Viscous gap",
    description: "A Hele-Shaw-inspired interference field.",
    concept: "Hele-Shaw flow",
    runtime: "GLSL browser preview",
    prompt:
      "Create one compact .frag artifact named viscous-gap.frag using only void main(), u_time, u_resolution, and gl_FragColor. Make a Hele-Shaw-inspired radial displacement field without claiming physical simulation. No textures, loops, #version, or discard. Return only shader source.",
    expectedArtifact: "viscous-gap.frag",
    previewBoundary: "Bounded WebGL 1 preview only.",
    fallback: "Inspect the shader artifact."
  },
  {
    id: "fluid-echo",
    title: "Fluid echo",
    description: "An analytical flow field framed honestly as visual inspiration.",
    concept: "Fluid simulation",
    runtime: "GLSL browser preview",
    prompt:
      "Create one compact .frag artifact named fluid-echo.frag using only void main(), u_time, u_resolution, and gl_FragColor. Make a fluid-simulation-inspired flow field with analytic curves; do not claim a physical solver. No textures, loops, #version, or discard. Return only shader source.",
    expectedArtifact: "fluid-echo.frag",
    previewBoundary: "Bounded WebGL 1 preview only.",
    fallback: "Inspect the shader artifact."
  }
] as const satisfies readonly CuratedPrompt[];

export const rhythmicLineStudyPrompt = {
  id: "rhythmic-line-study",
  title: "Rhythmic line study",
  description: "A modular line-system prompt inspired by open generative practice.",
  concept: "Modular moving lines",
  runtime: "p5.js browser preview",
  prompt: `Create one global-mode .p5.js artifact named rhythmic-line-study.p5.js with setup() and draw(). Compose a field of independently drifting lines with a limited palette, rounded caps, and a gentle pointer disturbance. ${boundedP5DemoSurfaceContract} Return only JavaScript source.`,
  expectedArtifact: "rhythmic-line-study.p5.js",
  previewBoundary: "Controlled p5.js preview only.",
  fallback: "Inspect the runnable code artifact."
} as const satisfies CuratedPrompt;
