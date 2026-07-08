# Golden Runtime Artifacts

These artifacts are public V8 release-candidate examples generated for static
QA and reviewer inspection. They are not a claim that every runtime named in
the roadmap is executable in V8.

Included:

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
