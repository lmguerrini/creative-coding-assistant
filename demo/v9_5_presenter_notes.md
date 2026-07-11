# V9.5 Demo Mode Presenter Notes

Use the in-app Demo Mode as the presenter surface. Each scenario states its
input, runtime, expected artifact, preview boundary, validation, and fallback.
The dataset at `demo/v9_5_golden_demo_dataset.json` is rehearsal metadata—not
a substitute for a live run.

## Recommended 8-minute path

1. Cymatic Chladni study — show the muted visual first. Start audio only when
   appropriate for the room; no microphone is involved.
2. Physarum drift — show a fast p5.js generation and pointer interaction.
3. Kinetic orbit sculpture — show a bounded Three.js preview and fullscreen.
4. Chladni light field — show the WebGL 1 shader boundary and nonblank frame.
5. Retrieval brief — inspect source grounding without claiming unavailable
   sources.
6. Multi-agent plan — show selected roles and workflow evidence, not a fake
   rendered artifact.
7. Export handoff — distinguish an inspectable export from a live preview.
8. Failure-recovery rehearsal — use only the controlled validation fixture;
   label a local draft or unavailable renderer honestly.

## Optional extensions

- Single-agent line study contrasts direct p5.js generation with the plan.
- Reference-guided palette study requires a presenter-supplied, non-sensitive
  image. If no image is available, use the text-guided fallback and say so.

## Boundaries to state plainly

- p5.js, Three.js, GLSL, and Tone.js run only through controlled browser
  surfaces.
- Tone.js output stays muted until the presenter explicitly starts it.
- TouchDesigner, React Three Fiber, and other external environments are
  code/export handoffs, not live execution claims.
- Hydra remains out of the visible V9.5 sequence pending its own dedicated
  browser evidence.
- An unavailable provider, retrieval source, or renderer is a limitation, not
  a success state.
