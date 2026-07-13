# Demo Prompt Library

> **Historical V8 prompt set.** These prompts support the archived eight-flow
> demo material. The current in-app source of truth is
> `clients/nextjs/src/lib/demo-mode.ts`; the current operator copy is
> [V9.5 Exact Prompt Library](v9_5_exact_prompt_library.md).

Use these prompts for the 10-minute demo, Q&A recovery, and offline fallback.
They are operator assets, not product prompt-template mutations.

## Primary Prompt

**Luminous audio-reactive Three.js scene**

```text
Create a luminous audio-reactive Three.js scene for a capstone demo:
concentric geometry, subtle bloom, FFT-driven motion accents, and a clear
explanation of provider choice, execution mode, estimates, fallback, and
escalation boundaries.
```

Demo value: shows Case 5 as the main story, supported by Case 1 retrieval
grounding and Case 6 advanced tool architecture.

Fallback: use `assets/preview_current.png` and `demo/golden_demo_dataset.json`
if provider, retrieval, or preview access fails.

## Secondary Prompts

**Concept-to-visual browser system**

```text
Translate the concept of threshold, recursion, and return into a practical
browser visual system with geometry, motion, color, runtime choice,
interaction, implementation constraints, and clear authority boundaries.
```

Use when explaining safe concept-to-visual translation and no authority claims.

**Debug silent browser audio**

```text
My browser audio-reactive sketch stays silent until a click and the analyser
looks flat. What should I check first, and how should I explain the
browser-audio constraints?
```

Use for Q&A if asked how the assistant handles practical debugging.

**Shader and post-processing pipeline**

```text
How do I build a glow-heavy kaleidoscopic browser visual with shaders or
post-processing passes? Give source-grounded runtime tradeoffs, failure risks,
and fallback implementation options.
```

Use when asked about retrieval quality, runtime tradeoffs, or technical depth.

**Offline showcase walkthrough**

```text
Walk through the prepared Creative Coding Assistant demo offline: problem,
solution, data, evaluation, ethical considerations, fallback plan, challenges,
and next steps.
```

Use if live systems are unavailable or timing requires a shorter path.

**p5.js generative morphogenesis sketch**

```text
Create a p5.js generative morphogenesis sketch using reaction diffusion,
cellular automata, L-systems, flow fields, particle systems, differential
growth, diffusion-limited aggregation, branching, self-organization, and
emergent form. Keep it browser-safe, explain interaction controls, and cite
retrieval/source boundaries conservatively.
```

Use when the demo needs a simpler canvas-based fallback for p5.js and
generative systems.

**Hydra and GLSL feedback texture chain**

```text
Design a Hydra feedback texture chain for the validated local hydra-synth
browser path and a GLSL fragment-shader fallback for a luminous kaleidoscopic
scene. Include oscillator layers, modulation, feedback, moire-like pattern
motion, and output routing. Explain that Hydra support is bounded to the
committed hydra-synth artifact QA path and how to recover if a runtime is
unavailable.
```

Use when asked about Hydra support, shader tradeoffs, or runtime fallbacks.

**Installation and immersive scene plan**

```text
Plan a browser-based installation or immersive scene for a gallery demo.
Include concept, geometry, audience movement, runtimes, retrieval needs,
preview plan, artifact package, evaluation checks, fallback route, and handoff
boundaries.
```

Use for installation planning and portfolio story questions.

**Geometry and morphogenesis visual system**

```text
Design a geometry and morphogenesis visual system for the browser. Combine
radial structures, recursive growth, reaction diffusion, diffusion-limited
aggregation, branching, flow fields, and particle trails. Include runtime
selection, preview strategy, source boundaries, and a graceful fallback plan.
```

Use when reviewers ask about generative structures, emergent form, and output
quality.

**Project handoff and reviewer evidence plan**

```text
Plan a Creative Coding Assistant project handoff for a browser installation:
concept, runtimes, retrieval needs, preview plan, artifact package, evaluation
checks, fallback route, and handoff boundaries. Do not claim live DCC/MCP
execution, autonomous delivery, or public deployment.
```

Use for senior-reviewer questions about system direction and next steps.
