# Golden Runtime Artifacts

These artifacts are public creative-code examples generated for static browser
QA and user inspection. They demonstrate only the runtime paths and boundaries
recorded below.

Included:

- `browser_full_runtime_qa.html`: offline browser QA harness for real temporary
  `p5`, `three`, and `hydra-synth` packages plus GLSL WebGL rendering.
- `browser_full_runtime_qa_results.json`: captured full-runtime browser QA
  evidence and accepted boundaries.
- `browser_render_qa.html`: fallback browser QA harness for p5.js shim render,
  GLSL WebGL render, and Three.js dependency-boundary validation when packages
  are unavailable.
- `p5_generative_morphogenesis_sketch.js`: p5.js browser sketch with
  bounded geometry and growth-system language.
- `three_audio_reactive_scene.js`: Three.js scene module with Web Audio
  analyser hooks and no required microphone capture.
- `glsl_kaleidoscope_field.frag`: GLSL fragment shader intended for a browser
  shader host or Shadertoy-style adapter.
- `hydra_feedback_lattice.js`: Hydra synth chain validated through real
  `hydra-synth` in the temporary QA workspace.
- `qa_manifest.json`: validation evidence and conservative runtime boundary.

Full-runtime browser QA is conservative:

- p5.js was rendered through a temporary QA install of `p5@2.3.0`.
- Three.js was rendered through a temporary QA install of `three@0.185.1`.
- GLSL is compiled, linked, drawn, and pixel-checked through WebGL.
- Hydra was rendered through a temporary QA install of `hydra-synth@1.4.0`
  with audio detection disabled and no microphone permission request.
- The temporary packages are validation dependencies only and were not added to
  the application dependency graph.
- Frame timing is an uncapped local draw-loop measurement, not a display-FPS,
  load, soak, or production performance benchmark.
