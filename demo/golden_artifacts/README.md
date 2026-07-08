# Golden Runtime Artifacts

These artifacts are public V8 release-candidate examples generated for static
QA and reviewer inspection. They are not a claim that every runtime named in
the roadmap is executable in V8.

Included:

- `browser_full_runtime_qa.html`: offline browser QA harness for real temporary
  `p5` and `three` packages plus GLSL WebGL rendering.
- `browser_full_runtime_qa_results.json`: captured full-runtime browser QA
  evidence and accepted boundaries.
- `browser_render_qa.html`: fallback browser QA harness for p5.js shim render,
  GLSL WebGL render, and Three.js dependency-boundary validation when packages
  are unavailable.
- `p5_sacred_geometry_sketch.js`: p5.js browser sketch with bounded sacred
  geometry language.
- `three_audio_reactive_scene.js`: Three.js scene module with Web Audio
  analyser hooks and no required microphone capture.
- `glsl_kaleidoscope_field.frag`: GLSL fragment shader intended for a browser
  shader host or Shadertoy-style adapter.
- `qa_manifest.json`: validation evidence and conservative runtime boundary.

Hydra is intentionally not generated in this pass because this repository has
not validated a live Hydra execution path for V8. Hydra remains guidance-only
unless installed, wired, and QA tested behind an explicit HITL decision.

Full-runtime browser QA is conservative:

- p5.js was rendered through a temporary QA install of `p5@2.3.0`.
- Three.js was rendered through a temporary QA install of `three@0.185.1`.
- GLSL is compiled, linked, drawn, and pixel-checked through WebGL.
- The temporary packages are validation dependencies only and were not added to
  the application dependency graph.
- Frame timing is an uncapped local draw-loop measurement, not a display-FPS,
  load, soak, or production performance benchmark.
