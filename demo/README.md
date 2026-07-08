# V8.8 Demo Pack

This folder contains operator-facing material for the final Turing College AI
Capstone demo and showcase. It prepares the demo; it does not execute the
assistant, call providers, run retrieval, render previews, or publish showcase
assets.

## Demo Mode

Use demo mode as a manual operating posture:

1. Start from `docs/CAPSTONE_DEMO_SHOWCASE.md`.
2. Keep `demo/demo_prompt_library.md` open.
3. Use `demo/golden_demo_dataset.json` for rehearsal and offline fallback.
4. Keep `demo/golden_artifacts/` ready as generated p5.js, Three.js, and GLSL
   artifact evidence.
5. Keep `demo/evaluation/` ready as the sanitized RAGAs evidence path.
6. Keep `assets/preview_current.png` ready if live preview is unavailable.
7. Use `docs/CAPSTONE_EVALUATION_ETHICS.md` for evaluation and ethics answers.

## Golden Flow

The primary flow is:

Prompt -> retrieval grounding -> creative planning -> code/artifact direction ->
internal preview or screenshot -> critique/refinement explanation -> final
project output narrative.

## Boundaries

- No live DCC/MCP execution is claimed.
- No HoloMind or HOLOiVERSE behavior is claimed.
- No autonomous agent swarm or unattended workflow control is claimed.
- Case 2 and Case 3 are guarded support claims only.
- Merge, push, tag, and freeze remain HITL-blocked.
- V8 Grand Review is handled separately on `version-review/v8` as a
  release-candidate validation program.
- Hydra is guidance-only unless a live execution path is installed, wired, and
  QA tested.
