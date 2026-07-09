# V8.8 Demo Pack

This folder contains operator-facing material for the final Turing College AI
Capstone demo and showcase. It prepares the demo; it does not execute the
assistant, run live retrieval, or publish showcase assets. The committed
evidence includes local browser QA artifacts and provider-backed RAGAs result
rows for public/redacted fixtures.

## Demo Mode

Use the integrated Demo Mode inside Creative Coding Assistant as the primary
Capstone presenter path:

1. Start the backend:
   `.venv/bin/python -m creative_coding_assistant.api.dev_server --host 127.0.0.1 --port 8000`
2. Start the frontend: `cd clients/nextjs && npm run dev`
3. Open `http://127.0.0.1:3000`.
4. Select `Demo Mode` in the workstation top bar.
5. Choose one of the 8 curated scenarios. The app loads the scenario prompt
   into the normal assistant composer and shows expected behavior, fallback,
   evidence, source boundaries, and output guidance.

Use the static demo pack as fallback/reviewer evidence:

1. Start from `docs/CAPSTONE_DEMO_SHOWCASE.md`.
2. Keep `demo/demo_prompt_library.md` open.
3. Use `demo/golden_demo_dataset.json` for rehearsal and offline fallback.
4. Open `demo/final_demo_launcher.html` from the local static QA server for the
   static final demo launcher if the frontend is unavailable.
5. Keep `demo/final_demo_suite.json` open for exact flow starts, success
   criteria, fallbacks, and reviewer talking points.
6. Keep `demo/golden_artifacts/` ready as generated p5.js, Three.js, GLSL, and
   Hydra artifact evidence, including `browser_full_runtime_qa_results.json`.
7. Keep `demo/evaluation/` ready as the sanitized and redacted latest-live
   RAGAs evidence paths.
8. Keep `assets/preview_current.png` ready if live preview is unavailable.
9. Use `docs/CAPSTONE_EVALUATION_ETHICS.md` for evaluation and ethics answers.

## Golden Flow

The primary flow is:

Prompt -> retrieval grounding -> creative planning -> code/artifact direction ->
internal preview or screenshot -> critique/refinement explanation -> final
project output narrative.

## Boundaries

- No live DCC/MCP execution is claimed.
- No autonomous immersive platform behavior is claimed.
- No autonomous agent swarm or unattended workflow control is claimed.
- Case 2 and Case 3 are guarded support claims only.
- Merge, push, tag, and freeze remain blocked until maintainer approval.
- V8 Grand Review is handled separately on `version-review/v8` as a
  release-candidate validation program.
- Hydra support is limited to the validated local `hydra-synth` browser
  artifact path; no broader Hydra editor, microphone, or DCC/MCP claim is made.
