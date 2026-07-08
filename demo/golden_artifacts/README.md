# Golden Runtime Artifacts

These artifacts are public V8 release-candidate examples generated for static
QA and reviewer inspection. They are not a claim that every runtime named in
the roadmap is executable in V8.

Included:

- `browser_render_qa.html`: offline browser QA harness for p5.js shim render,
  GLSL WebGL render, and Three.js dependency-boundary validation.
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

Browser QA is conservative:

- p5.js is rendered through the local harness' minimal p5-compatible canvas
  shim because a full local p5 package is not installed.
- GLSL is compiled, linked, drawn, and pixel-checked through WebGL.
- Three.js is rendered only if a local Three package is present. In the current
  V8 review environment, no local Three package is installed, so the harness
  validates module export and graceful dependency failure rather than claiming
  a Three.js browser render.
